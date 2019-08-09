# ################################################
# Extract Text Features from PubMed XML files
# Xiaonan Ji (ji.62@osu.edu)
# Original: 2017.5.23
# Last Update: 2017.12.20
#
# PubMed XML Element Description:
# https://www.nlm.nih.gov/bsd/licensee/elements_descriptions.html
#
# PubMed XML data fields to be used:
# Year: <DateCreated>-<Year>
# PMID: <PMID>
# Title: <Article>-<ArticleTitle>,
# Abstract: <Article>-<Abstract>-<AbstractText>
# Author: <Article>-<AuthorList>-<Author>-<LastName> & <ForeName>
# Journal Type/Name: <Article>-<Journal>-<Title>
# Author Keywords: <KeywordList>-<Keyword>
# MeSH Keywords: <MeshHeadingList>-<MeshHeading>-<DescriptorName>
# Publication Type: <>Article>-<PublicationTypeList>-<PublicationType>
#
# ################################################
from __future__ import division
import xml.etree.ElementTree as ET
import csv
import numpy as np
import scipy
from scipy import spatial
import string
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from sklearn.manifold import MDS
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import SpectralEmbedding
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise
from sklearn import preprocessing
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.sparse import hstack
import nltk
from rake_nltk import Rake
import random
import os.path
import math
import cairocffi as cairo
from igraph import *
from gensim import models
from random import shuffle
import operator

# remove non-ascii characters
def removeNonAscii(s): return "".join(i for i in s if ord(i) < 128)

# define the function to extract Noun Phrases (NP) from text i.e. title and abstract
def NPextractor(text):
	text = removeNonAscii(text)
	if len(text) == 0:
		return text
	
	tok = nltk.word_tokenize(text)
	pos = nltk.pos_tag(tok)	
	grammar = r"""
	  NP: {<DT|PP\$>?<JJ>*<NN>}   # chunk determiner/possessive, adjectives and noun
		  {<NNP>+}                # chunk sequences of proper nouns
	"""
	chunker = nltk.RegexpParser(grammar)
	tree = chunker.parse(pos)
	nps = [] # word and pos_tag
	nps_words = [] # only word
	for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
		nps.append(subtree.leaves())
		current_np = []
		for item in subtree.leaves():
			current_np.append(item[0])
		nps_words.append(current_np)	
	
	refined_words = []
	stopwords = nltk.corpus.stopwords.words('english')
	for np in nps_words:
		current_np = []
		for word in np:
			if (2 <= len(word) <= 40) and (word.lower() not in stopwords):
				current_np.append(word.lower())
		refined_words.append(current_np)		
	
	stemmed_words = []
	stemmer = nltk.stem.porter.PorterStemmer()
	lemmatizer = nltk.WordNetLemmatizer()
	for np in refined_words:
		current_np = []
		for word in np:
			if word == "aeds" or word == "aed":
				continue
			word = stemmer.stem(word)
			word = lemmatizer.lemmatize(word)
			if (2 <= len(word) <= 40) and (word not in stopwords):
				current_np.append(word)
		stemmed_words.append(current_np)		
	
	refined_text = ""
	for np in stemmed_words:
		for word in np:
			refined_text += word
			refined_text += " "
		refined_text = refined_text.strip()	
		refined_text += ". "
	refined_text = refined_text.strip()		
	return refined_text

# (update 2017/10/05) define the function to extract Noun Phrases (NP), non-stemming, in the format of adj+noun
def NPextractor2(text):
	text = removeNonAscii(text)
	if len(text) == 0:
		return text
	
	tok = nltk.word_tokenize(text)
	pos = nltk.pos_tag(tok)
	
	# the original grammar, to get shorter NPs
	grammar1 = r"""
	  NP: {<DT|PP\$>?<JJ>*<NN>}   # chunk determiner/possessive, adjectives and noun
		  {<NNP>+}                # chunk sequences of proper nouns
	"""

	
	# the self defined grammar based on the previous version above, to get longer NPs (as supplements)
	grammar2 = r"""
	  NP: {<DT|PP\$>?<JJ>*<NN|NNS|NNP|NNPS>+}   # chunk determiner/possessive, adjectives and noun(s)
		  {<NNP>+}                # chunk sequences of proper nouns
	"""
	
	chunker1 = nltk.RegexpParser(grammar1)
	tree1 = chunker1.parse(pos)
	chunker2 = nltk.RegexpParser(grammar2)
	tree2 = chunker2.parse(pos)
	
	nps = [] # word and pos_tag
	nps_words = [] # only word
	
	for subtree in tree1.subtrees(filter=lambda t: t.label() == 'NP'):
		nps.append(subtree.leaves())
		current_np = []
		for item in subtree.leaves():
			current_np.append(item[0])
		nps_words.append(current_np)

	for subtree in tree2.subtrees(filter=lambda t: t.label() == 'NP'):
		if subtree.leaves() in nps:
			continue
		nps.append(subtree.leaves())
		current_np = []
		for item in subtree.leaves():
			current_np.append(item[0])
		nps_words.append(current_np)
	
	refined_words = []
	#stopwords = nltk.corpus.stopwords.words('english')
	stopwords = my_stopwords
	for np in nps_words:
		if len(np) < 1:
			continue
		current_np = []
		for word in np:
			if (2 <= len(word) <= 40) and (word.lower() not in stopwords):
				current_np.append(word.lower())
		if len(current_np) >= 1:		
			refined_words.append(current_np)
	return refined_words

# define the function to process raw text via tokenization, converting words to lower case, removing stopwords & digits, stemming, etc.	
def textProcessor(text):
	text = removeNonAscii(text)
	if len(text) == 0:
		return text
		
	#text.replace("-", " ") # update 2017/12/31 (used to ACE/ADHD/BetaBlocker/Opioids/Proton/... not used for Estrogens/NSAIDS/Statins)
	tok = nltk.word_tokenize(text)
	stopwords = nltk.corpus.stopwords.words('english')
	refined_words = []
	for word in tok:
		word = str(word).translate(None, string.punctuation)
		word = word.lower()
		if (2 <= len(word) <= 40) and (word not in stopwords) and (not word.isdigit()):
			refined_words.append(word)	
	
	stemmed_words = []
	stemmer = nltk.stem.porter.PorterStemmer()
	lemmatizer = nltk.WordNetLemmatizer()
	for word in refined_words:
		if word == "aeds" or word == "aed":
			continue
		word = stemmer.stem(word)
		word = lemmatizer.lemmatize(word)
		if (2 <= len(word) <= 40) and (word not in stopwords):
			stemmed_words.append(word)		
	
	refined_text = ""
	for word in stemmed_words:
		refined_text += word
		refined_text += " "
	refined_text = refined_text.strip()	
	refined_text += ". "		
	return refined_text
	
# (Update 2017/12/16)
# define the function to (simply) process raw text via tokenization, converting words to lower case, removing digits, stemming, etc.
# stopwords are kept to preserve necessary context information 
# this function also distinguishes the break between sentence. The sentence break ". " will be reserved.
# for example, the processed texts will be fed into doc2vec (Paragraph Vector model)	
def textProcessor_simple(text):
	text = removeNonAscii(text)
	if len(text) == 0:
		return text
	
	#text.replace("-", " ") # update 2017/12/31 (especially used to ACE/ADHD/BetaBlocker/Opioids/Proton/... not used for Estrogens/NSAIDS/Statins)
	tok = nltk.word_tokenize(text)
	stopwords = nltk.corpus.stopwords.words('english')
	refined_words = []
	for word in tok:
		if word in ['.', '?', '!']:
			word = "mysentencebreak"
		word = str(word).translate(None, string.punctuation)
		word = word.lower()
		word = word.strip()
		if (1 <= len(word) <= 40) and (not word.isdigit()):
			refined_words.append(word)	
	
	stemmed_words = []
	stemmer = nltk.stem.porter.PorterStemmer()
	lemmatizer = nltk.WordNetLemmatizer()
	for word in refined_words:
		if word == "aeds" or word == "aed":
			continue
		if word == "mysentencebreak":
			stemmed_words.append(word)
			continue
		word = stemmer.stem(word)
		word = lemmatizer.lemmatize(word)
		if (1 <= len(word) <= 40) and (not word.isdigit()):
			stemmed_words.append(word)
	refined_words = stemmed_words			
	
	refined_text = ""
	for word in refined_words:
		if word == "mysentencebreak":
			refined_text = refined_text.strip()
			refined_text += ". "
		else:	
			refined_text += word
			refined_text += " "
	#refined_text = refined_text.strip()		
	return refined_text	

# (Update 2017/12/19)
# define the function to stem a word or text piece
def myStemmer(text):
	stemmer = nltk.stem.porter.PorterStemmer()
	lemmatizer = nltk.WordNetLemmatizer()
	stemmed_text = ""
	for word in text.split():
		word = lemmatizer.lemmatize(stemmer.stem(word)).strip()
		stemmed_text += word
		stemmed_text += " "
	return stemmed_text.strip()

# define the function to extract the top words of topics resulted from LDA
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

# (update 2017/10/05) functions to extract keywords from texts, implementing the RAKE method
def isPunct(word):
    return len(word) == 1 and word in string.punctuation

def isNumeric(word):
    try:
        float(word) if '.' in word else int(word)
        return True
    except ValueError:
        return False
        
def containNumeric(word):
    return any(char.isdigit() for char in word)

# define my own stop words list. Based on the nltk stopwords and Rake stopwords     
my_stopwords = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "about", "above", "addition", "after", "again", "against", "ain", "all", 
"also", "although", "am", "among", "an", "and", "any", "approach", "approached", "approaches", "approaching", "are", "aren", "as", "at", "b", "based", "be", 
"because", "been", "before", "being", "below", "between", "both", "but", "by", "c", "called", "can", 
"consider", "considers", "consideres", "considering", "corresponding", "could", "couldn", "d", 
"develop", "developed", "developing", "develops", "did", "didn", "do", "does", "doesn", "doing", "don", "down", "during", "e", 
"each", "f", "few", "first", "for", "from", "further", "g", "go", "goes", "h", "had", "hadn", "has", "hasn", "have", "haven", 
"having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "however", "i", "if", "in", "include", "included", 
"includes", "including", "into", "is", "isn", "it", "its", "itself", "j", "just", "k", "l", "ll", "m", "m", "ma", "many", "may", "me", 
"mg", "might", "mightn", "more", "most", "much", "must", "mustn", "my", "myself", "n", "needn", "never", "new", "no", "none", "nor", 
"not", "now", "o", "of", "off", "on", "once", "one", "ones", "only", "or", "other", "others", "otherwise", "our", "ours", "ourselves", 
"out", "over", "over", "own", "p", "particular", "present", "presented", "presenting", "presents", "propose", "proposed", "proposes", 
"proposing", "provide", "provided", "provides", "providing", "q", "r", "re", "result", "resulted", "resulting", "results", "s", "same", 
"shall", "shalln", "shan", "she", "should", "shouldn", "show", "showed", "showing", "shows", "since", "so", "some", "studied", "studies", 
"study", "studying", "sub", "such", "sup", "t", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", 
"they", "this", "those", "though", "through", "throughout", "to", "too", "two", "u", "under", "until", "up", "use", "used", "uses", "using", 
"v", "ve", "very", "via", "w", "was", "wasn", "we", "well", "were", "weren", "what", "when", "where", "whether", "which", "while", "who", 
"whom", "why", "will", "with", "without", "won", "would", "wouldn", "x", "y", "you", "your", "yours", "yourself", "yourselves", "z"]

class RakeKeywordExtractor:
	def __init__(self):
		#self.stopwords = set(nltk.corpus.stopwords.words())
		self.stopwords = set(my_stopwords)
		self.top_fraction = 1 # consider top third candidate keywords by score

	def _generate_candidate_keywords(self, sentences, upper_length, mode):
		phrase_list = []
		for sentence in sentences:
			# Additional Noun phrases if they won't be detected by the Rake method below
			nps = NPextractor2(sentence)
			if mode == "np":
				if len(nps) > 0:
					for item in nps:
						if len(item) > 0 and len(item) <= upper_length:  # restrict the length of phrase to be 1~5
							phrase_list.append(item)
				continue
				
			words = map(lambda x: "|" if x in my_stopwords else x, nltk.word_tokenize(sentence.lower()))
			phrase = []
			for word in words:
				#if word == "|" or isPunct(word):
				if word == "|" or isPunct(word) or isNumeric(word) or containNumeric(word):
					#if len(phrase) > 0:
					if len(phrase) > 0 and len(phrase) <= upper_length: # restrict the length of phrase to be 1~5
						if phrase not in nps:
							phrase_list.append(phrase)
						phrase = []
				else:
					phrase.append(word)
			if len(nps) > 0:
				#phrase_list += nps
				for item in nps:
					if len(item) > 0 and len(item) <= upper_length:  # restrict the length of phrase to be 1~5
						phrase_list.append(item)
		return phrase_list
        
	def _calculate_word_scores(self, phrase_list):
		word_freq = nltk.FreqDist()
		word_degree = nltk.FreqDist()
		for phrase in phrase_list:
			degree = len(filter(lambda x: not isNumeric(x) and not containNumeric(x), phrase)) - 1
			for word in phrase:
				#word_freq.inc(word)
				word_freq[word] += 1
				#word_degree.inc(word, degree) # other words
				word_degree[word] += degree
		for word in word_freq.keys():
			word_degree[word] = word_degree[word] + word_freq[word] # itself
		# word score = deg(w) / freq(w) (favor long phrases), or word score = deg(w) (not that favor long phrases)
		word_scores = {}
		for word in word_freq.keys():
			#word_scores[word] = word_degree[word] / word_freq[word] # (favor long phrases)
			word_scores[word] = word_degree[word]
		return word_scores

	def _calculate_phrase_scores(self, phrase_list, word_scores):
		phrase_scores = {}
		for phrase in phrase_list:
			phrase_score = 0
			for word in phrase:
				phrase_score += word_scores[word]
			phrase_scores[" ".join(phrase)] = phrase_score
			#phrase_scores[" ".join(phrase)] = phrase_score/len(phrase)
		return phrase_scores
                    
	def extract(self, text, incl_scores=False, number=30, upper_length=5, mode="all"):
		sentences = nltk.sent_tokenize(text) # break a text (paragraph) into an array of single sentences ending with a period
		phrase_list = self._generate_candidate_keywords(sentences, upper_length, mode)
		word_scores = self._calculate_word_scores(phrase_list)
		phrase_scores = self._calculate_phrase_scores(phrase_list, word_scores)
		sorted_phrase_scores = sorted(phrase_scores.iteritems(), key=operator.itemgetter(1), reverse=True)
		n_phrases = len(sorted_phrase_scores)
		if incl_scores:
			#return sorted_phrase_scores[0:int(n_phrases/self.top_fraction)]
			return sorted_phrase_scores[0:number]
		else:
			#return map(lambda x: x[0], sorted_phrase_scores[0:int(n_phrases/self.top_fraction)])
			return map(lambda x: x[0], sorted_phrase_scores[0:number])
	
# the file directory
#directory = "./"
#directory = "./DERP/ACEInhibitors/"
#directory = "./DERP/ADHD/"
#directory = "./DERP/Antihistamines/"
#directory = "./DERP/AtypicalAntipsychotics/"
#directory = "./DERP/BetaBlockers/"
#directory = "./DERP/CalciumChannelBlockers/"
#directory = "./DERP/Estrogens/"
#directory = "./DERP/NSAIDS/"
#directory = "./DERP/Opioids/"
#directory = "./DERP/OralHypoglycemics/"
#directory = "./DERP/ProtonPumpInhibitors/"
#directory = "./DERP/SkeletalMuscleRelaxants/"
#directory = "./DERP/Statins/"
#directory = "./DERP/Triptans/"
#directory = "./DERP/UrinaryIncontinence/"
#directory = "./DERP/VisPubData/"
directory = "./DERP/MobileHealthApps/"

# read in the article IDs to be highlighted
highlight = []
if os.path.isfile(directory + "highlight.txt"):
	with open(directory + "highlight.txt") as f:
		highlight = f.readlines()
	highlight = [x.strip() for x in highlight]
	
# red in the prior-provided keywords
'''
prior-keywords = ["adderall", "adhd", "adult", "amphetamine", "aripiprazole", "atomoxetine", "attention",
				"bupropion", "catapres", "child", "clonidine", "clozapine", "clozaril", "concerta", "cylert",
				"deficit", "disorder", "dexedrine", "dexmethylphenidate", "dextroamphetamine", "focalin", "geodon", 
				"guanfacine", "hyperactivity", "metadate", "methylin", "methylphenidate", "modafinil", "olanzapine", 
				"pediatric", "pemoline", "provigil", "quetiapine", "risperdal",
				"risperidone", "ritalin", "seroquel", "strattera", "tenex", "wellbutrin", "ziprasidone", "zyprexa", "adverse",
				"itch", "poison", "toxic", "functional", "withdrawal", "safety",
				"mortality", "hospitalization", "hyperactivity", "emergency", "discontinuation", "employ", "nausea",
				"vomit", "headache", "tolerance", "death", "dead", "placebo", "attention deficit hyperactivity disorder", 
				"quality of life", "life quality", "dry mouth", "mental state"]	
'''
prior_keywords = []
if os.path.isfile(directory + "keywords.txt"):
	with open(directory + "keywords.txt") as f:
		keywords = f.readlines()
	prior_keywords = [x.strip() for x in keywords]				
	
# Container for extracted key information	
years = []
ids = [] # PubMed publication ID
types = [] # jouranl type or journal name

titles_np = [] # noun phrases from title
titles_all = [] # all processed terms from title (lower case, no punctuation/digit/stopword, stemmer)
titles_all_simple = [] # (update 20171216) all simply processed terms from title (lower case, no punctuation/digit, stemmer), keep sentence breaks
titles_raw = [] # all raw terms from title
abstracts_np = [] # noun phrases from abstract
abstracts_all = [] # all processed terms from abstract
abstracts_all_simple = [] # (update 20171216) all simply processed terms from abstract (lower case, no punctuation/digit, stemmer), keep sentence breaks
abstracts_raw = [] # all raw terms from abstract

authors = [] # author names in the format of string, each author name is written as "Lastname-FirstNameInitial"
authors2 = [] # author names in the format of array, each author name is written as "Lastname Firstname-Initial"

keywords = [] # author provided keywords in the format of string
keywords2 = [] # author provided keywords in the format of array
terms = [] # MeSH terms in the format of string
terms2 = [] # MeSH terms in the format of array

pubtypes = [] # publication types in the format of string
pubtypes2 = [] # publications types in the format of array

references = [] # refernece list
index = {}
current_index = 0
markers = [] # shape of visualized data points
if_highlight = [] # if articles are highlighted per prior knowledge

flag_keywords = 0 # record the number of articles having keywords (in case there is little article containing keywords)
flag_terms = 0 # record the number of articles having terms (in case there is little article containing MeSH terms)

#--------------------------------File Type 1: XML from PubMed------------------------------#
# Process the .xml file and extract information (update 09/21/2017: handle both general paper articles and book articles from PubMed)
###

tree = ET.parse(directory + 'data.xml')
root = tree.getroot()
for element in root:
	if not (((element.find('MedlineCitation') is not None) and (element.find('MedlineCitation').find('PMID') is not None) and (element.find('MedlineCitation').find('Article') is not None) and (element.find('MedlineCitation').find('Article').find('ArticleTitle') is not None)) or ((element.find('BookDocument') is not None) and (element.find('BookDocument').find('PMID') is not None) and (element.find('BookDocument').find('ArticleTitle') is not None))):
		continue
	
	#medline = element.find('MedlineCitation')
	medline = element.find('MedlineCitation') if element.find('MedlineCitation') is not None else element.find('BookDocument')
	
	#article = medline.find('Article')
	article = medline.find('Article') if medline.find('Article') is not None else medline
	if article.find('ArticleTitle') is None or article.find('ArticleTitle').text == "" or article.find('ArticleTitle').text is None:
		continue	
	
	ids.append(medline.find('PMID').text.strip())
	#print medline.find('PMID').text.strip()
	#years.append(medline.find('DateCreated').find('Year').text if medline.find('DateCreated') is not None else medline.find('Book').find('PubDate').find('Year').text)
	
	####It seems that PubMed changed the coding/format of the exported XML file, especially regarding publication time. 	
	####yz updated: 01/10/2018 Location of year: MedlineCitation->Article->Journal->JournalIssue->PubDate->Year##########
	if medline.find('Article') is not None:
		if article.find('Journal').find('JournalIssue').find('PubDate').find('Year') is not None:
			years.append(article.find('Journal').find('JournalIssue').find('PubDate').find('Year').text)
		else:
			years.append(filter(str.isdigit, article.find('Journal').find('JournalIssue').find('PubDate').find('MedlineDate').text))
			#print years
	else:
		years.append(article.find('Book').find('PubDate').find('Year').text)
	#############################################
	
	types.append(article.find('Journal').find('Title').text if article.find('Journal') is not None else "book")
	
	current_title = article.find('ArticleTitle').text
	#titles_raw.append(removeNonAscii(current_title).replace("\"", "").replace("\'", "").strip())
	titles_raw.append(removeNonAscii(current_title).replace("\"", "").replace("\'", "").replace("\\", "-").replace("\n", " ").strip())
	titles_all.append(textProcessor(current_title.strip()))
	titles_all_simple.append(textProcessor_simple(current_title.strip()))
	titles_np.append(NPextractor(current_title.strip()))
	
	current_abstract = ""
	if article.find('Abstract') is not None:
		for child in article.find('Abstract'):
			if child is not None and child.text is not None:
				current_abstract += child.text
				current_abstract += " "
		#abstracts_raw.append(removeNonAscii(current_abstract).replace("\"", "").replace("\'", "").strip())
		abstracts_raw.append(removeNonAscii(current_abstract).replace("\"", "").replace("\'", "").replace("\\", "-").replace("\n", " ").strip())
		abstracts_all.append(textProcessor(current_abstract.strip()))
		abstracts_all_simple.append(textProcessor_simple(current_abstract.strip()))
		abstracts_np.append(NPextractor(current_abstract.strip()))
	else:
		abstracts_raw.append("")
		abstracts_all.append("")
		abstracts_all_simple.append("")
		abstracts_np.append("")
	
	author_string = ""
	author_list = []
	if article.find('AuthorList') is not None:
		for author in article.find('AuthorList').findall('Author'):
			if (author.find('LastName') is None) or (author.find('ForeName') is None) or (author.find('Initials') is None):
				continue
			author_list.append(removeNonAscii(author.find('LastName').text).strip() + "_" + removeNonAscii(author.find('Initials').text).strip())
			author_string += (removeNonAscii(author.find('LastName').text).strip()) + "_" + (removeNonAscii(author.find('Initials').text).strip()) + "; "
			#author_string += (author.find('LastName').text.strip()) + "_" + (author.find('ForeName').text.strip()) + "; "
	authors.append(author_string)
	authors2.append(author_list)
	
	keyword_string = ""
	keyword_list = []
	if medline.find('KeywordList') is not None:
		flag_keywords += 1
		for keyword in medline.find('KeywordList'):
			keyword_list.append(removeNonAscii(keyword.text))
			#keyword_string += removeNonAscii(keyword.text.strip().replace(" ", "_").lower())
			keyword_string += removeNonAscii(keyword.text.strip().replace(" ", "_").replace("\\", "-").lower())
			keyword_string += ", "
	keywords.append(keyword_string)
	keywords2.append(keyword_list)
	
	term_string = ""
	term_list = []
	if medline.find('MeshHeadingList') is not None:
		flag_terms += 1
		for mesh in medline.find('MeshHeadingList'):
			term_list.append(removeNonAscii(mesh.find('DescriptorName').text))
			#term_string += removeNonAscii(mesh.find('DescriptorName').text.strip().translate(None, string.punctuation).replace(" ", "_").lower())
			term_string += removeNonAscii(mesh.find('DescriptorName').text.strip().translate(None, string.punctuation).replace(" ", "_").replace("\\", "-").lower())
			term_string += ", "
	terms.append(term_string)
	terms2.append(term_list)
	
	pubtype_string = ""
	pubtype_list = []
	if article.find('PublicationType') is not None:
		pubtype_list.append(removeNonAscii(article.find('PublicationType').text))
		#pubtype_string += removeNonAscii(article.find('PublicationType').text.strip().translate(None, string.punctuation).replace(" ", "_").lower())
		pubtype_string += removeNonAscii(article.find('PublicationType').text.strip().translate(None, string.punctuation).replace(" ", "_").replace("\\", "-").lower())
		pubtype_string += ". "
	elif article.find('PublicationTypeList') is not None:
		for pubtype in article.find('PublicationTypeList'):
			pubtype_list.append(removeNonAscii(pubtype.text))
			#pubtype_string += removeNonAscii(pubtype.text.strip().translate(None, string.punctuation).replace(" ", "_").lower())
			pubtype_string += removeNonAscii(pubtype.text.strip().translate(None, string.punctuation).replace(" ", "_").replace("\\", "-").lower())
			pubtype_string += ", "
	pubtypes.append(pubtype_string)
	pubtypes2.append(pubtype_list)
	
	refer_list = []
	if medline.find('CommentsCorrectionsList') is not None:
		for item in medline.find('CommentsCorrectionsList').findall('CommentsCorrections'):
			if item.get('RefType') != "Cites":
				continue
			if item.find('PMID') is not None:
				refer_list.append(item.find('PMID').text.strip())
	references.append(refer_list)
			
	index[medline.find('PMID').text.strip()] = current_index # mapping between PMID (publication ID) and index (the local index)
	current_index += 1	
		
	markers.append('o')
	if medline.find('PMID').text.strip() in highlight:
		if_highlight.append("1")
	else:
		if_highlight.append("0")
###


#------------------------------------File Type 2: CSV for VisPub------------------------------------#
###
# Process the .csv file and extract information
# 2-Title, 10-Authors, 1-Year, 9-Abstract, 3-DOI
# 13-Author Keywords, ?-IEEE Terms , 0-Type (InfoVis, VAST, SciVis)
'''
with open(directory + 'data.csv') as csvfile:
	myreader = csv.reader(csvfile, delimiter = ',')
	for row in myreader:
		years.append(row[1].strip())
		ids.append(row[3].strip())
		types.append(row[0].strip())
		
		author_list = removeNonAscii(row[10]).strip().split(";")
		authors2.append(author_list)
		author_string = ""
		for item in author_list:
			author_string += item.strip().replace(", ", "_").replace(" ", "_").replace(".", "").lower()
			author_string += ", "
		authors.append(author_string)
		
		current_title = row[2].strip()
		titles_raw.append(removeNonAscii(current_title).replace("\"", "").replace("\'", "").replace("\\", "-").strip()) # all raw terms
		titles_all.append(textProcessor(current_title.strip())) # all processed terms
		titles_all_simple.append(textProcessor_simple(current_title.strip()))
		titles_np.append(NPextractor(current_title.strip())) # only noun phrases
	
		current_abstract = row[9].strip()
		if len(current_abstract) > 0:
			abstracts_raw.append(removeNonAscii(current_abstract).replace("\"", "").replace("\'", "").replace("\\", "-").strip()) # all raw terms
			abstracts_all.append(textProcessor(current_abstract.strip())) # all processed terms
			abstracts_all_simple.append(textProcessor_simple(current_abstract.strip()))
			abstracts_np.append(NPextractor(current_abstract.strip())) # only noun phrases
		else:
			abstracts_raw.append("")
			abstracts_all.append("")
			abstracts_all_simple.append("")
			abstracts_np.append("")
	
		
		if len(row[13].strip()) > 0:
			flag_keywords += 1
		keyword_list_raw = removeNonAscii(row[13]).strip().split(",")
		keyword_list = []
		keyword_string = ""
		#keywords2.append(keyword_list)
		for item in keyword_list_raw:
			keyword_list.append(item.strip().lower())
			keyword_string += item.strip().replace(" ", "_").replace("\\", "-").lower()
			keyword_string += ", "
		keywords.append(keyword_string)
		keywords2.append(keyword_list)
		
		pubtypes.append(row[0].strip())
		pubtypes2.append(row[0].strip())		
		
		refer_list = row[12].strip().split(";") # The reference information is not available from the IEEE exported CSV data
		references.append(refer_list)
			
		index[row[3]] = current_index # mapping between DOI (publication ID) and index (the local index)
		current_index += 1	
		
		if "Visual" in row[3]:
			markers.append('v')
		elif "Big Data" in row[3]:
			markers.append('o')
		else:
			markers.append('*')
			
		if row[3].strip() in highlight:
			if_highlight.append("1")
		else:
			if_highlight.append("0")
'''	
###

	
# Get the citation list based on the reference list
citations = np.empty(current_index, dtype = object)
for i in range(0, current_index):
	citations[i] = []
for i in range(0, current_index):
	for item in references[i]:
		if item in index.keys():
			citations[index[item]].append(ids[i])

# simplify references and citations with local index (instead of long publication IDs)
references2 = np.empty(current_index, dtype = object)
for i in range(0, current_index):
	references2[i] = []
	for item in references[i]:
		if item in index.keys():
			references2[i].append(index[item])
		
citations2 = np.empty(current_index, dtype = object)
for i in range(0, current_index):
	citations2[i] = []
	for item in citations[i]:
		if item in index.keys():
			citations2[i].append(index[item])		

# Handle the text features and dimensionality reduction for title, abstract, keywords, and terms respectively
		
# Get the textual features with the bag-of-words approach	
titles_vectorizer = CountVectorizer(max_df = 0.5, min_df = 2, stop_words='english', ngram_range=(1, 1)) # term frequency (TF)
#titles_vectorizer = TfidfVectorizer(max_df = 0.1, min_df = 1, stop_words='english', ngram_range=(1, 1)) # term frequency-inverse document frequency (TF-IDF)
titles_vec = titles_vectorizer.fit_transform(titles_all)
# Calculate the pairwise similarities (cosine similarity)
#titles_simi = pairwise.cosine_similarity(titles_vec)
# Call truncated SVD to handle the sparse matrix and initial dimensionality reduction
#svd_ti = TruncatedSVD(n_components=min(100, titles_vec.shape[1] - 1), n_iter=100, random_state=0)
#titles_svd = svd_ti.fit_transform(titles_vec.toarray())
# Call t-SNE for further higher-quality dimensionality reduction
model = TSNE(n_components = 2, init='pca', perplexity=100, random_state = 0)
np.set_printoptions(suppress = True)
#titles_2d = model.fit_transform(titles_svd)

abstracts_vectorizer = CountVectorizer(max_df = 0.5, min_df = 2, stop_words='english', ngram_range=(1, 1))
#abstracts_vectorizer = TfidfVectorizer(max_df = 0.1, min_df = 2, stop_words='english', ngram_range=(1, 1))
abstracts_vec = abstracts_vectorizer.fit_transform(abstracts_all)
#abstracts_simi = pairwise.cosine_similarity(abstracts_vec)
#svd_ab = TruncatedSVD(n_components=min(100, abstracts_vec.shape[1] - 1), n_iter=100, random_state=0)
#abstracts_svd = svd_ab.fit_transform(abstracts_vec.toarray())
#abstracts_2d = model.fit_transform(abstracts_svd)

# (Update 2017/9/18) [additionally for abstract] apply abstract tf-idf features for topics (LAD and NMF respectively)
#################################
topic_number = 20
n_top_words = 10
abstracts_vectorizer_tf = TfidfVectorizer(max_df = 0.4, min_df = 2, stop_words='english', ngram_range=(1, 1))
abstracts_tf = abstracts_vectorizer_tf.fit_transform(abstracts_all)
abstracts_tf_feature_names = abstracts_vectorizer_tf.get_feature_names()
# LDA
lda = LatentDirichletAllocation(n_components=topic_number, max_iter=10, learning_method='online', learning_offset=10., random_state=0)
lda.fit(abstracts_tf)
# get the top words for each topic
topic_lda_topwords = []
for topic_idx, topic in enumerate(lda.components_):
	current_top_words = " ".join([abstracts_tf_feature_names[i]
									for i in topic.argsort()[:-n_top_words - 1:-1]])
	topic_lda_topwords.append(current_top_words)
# get the topic of each documents
topics_lda = []
doc_topic_distribution = lda.transform(abstracts_tf)
for doc in doc_topic_distribution:
	index = np.argmax(doc)
	topics_lda.append(topic_lda_topwords[index])
# NMF	
nmf = NMF(n_components=topic_number, random_state=1, beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1, l1_ratio=.5)
nmf.fit(abstracts_tf)
topic_nmf_topwords = []
for topic_idx, topic in enumerate(nmf.components_):
	current_top_words = " ".join([abstracts_tf_feature_names[i]
									for i in topic.argsort()[:-n_top_words - 1:-1]])
	topic_nmf_topwords.append(current_top_words)
# get the topic of each documents
topics_nmf = []
doc_topic_distribution = nmf.transform(abstracts_tf)
for doc in doc_topic_distribution:
	index = np.argmax(doc)
	topics_nmf.append(topic_nmf_topwords[index])
#################################

# (Update 2017/10/07) [additionally for title and abstract] get keywords for each article with references to the Rake method
#################################
'''
title_keywords_rake = []
rake_title = RakeKeywordExtractor()
for index in range(0, current_index):
	title_keywords_rake.append(rake_title.extract(titles_raw[index], incl_scores=False, number=15, upper_length=3, mode="np"))

abstract_keywords_rake = []
rake_abstract = RakeKeywordExtractor()
for index in range(0, current_index):
	abstract_keywords_rake.append(rake_abstract.extract(abstracts_raw[index], incl_scores=False, number=15, upper_length=3, mode="np"))
'''
# for title and abstract jointly	
keywords_rake = []
rake = RakeKeywordExtractor()
for index in range(0, current_index):
	keywords_rake.append(rake.extract(titles_raw[index] + abstracts_raw[index], incl_scores=False, number=30, upper_length=3, mode="np"))	
#################################

'''
keywords_vec = np.zeros((current_index, 1))
keywords_2d = np.zeros((current_index, 2))
keywords_simi = np.zeros((current_index, current_index))
if flag_keywords > 1:
	keywords_vectorizer = CountVectorizer(max_df = 0.9, min_df = 1, stop_words='english')
	#keywords_vectorizer = TfidfVectorizer(max_df = 0.1, min_df = 1, stop_words='english')
	keywords_vec = keywords_vectorizer.fit_transform(keywords)
	keywords_simi = pairwise.cosine_similarity(keywords_vec)
	svd_ky = TruncatedSVD(n_components=min(100, keywords_vec.shape[1] - 1), n_iter=100, random_state=0)
	#keywords_svd = svd_ky.fit_transform(keywords_vec.toarray())
	#keywords_2d = model.fit_transform(keywords_svd)

terms_vec = np.zeros((current_index, 1))	
terms_2d = np.zeros((current_index, 2))
terms_simi = np.zeros((current_index, current_index))
if flag_terms > 1:	
	terms_vectorizer = CountVectorizer(max_df = 0.9, min_df = 1, stop_words='english') # term frequency (TF)
	#terms_vectorizer = TfidfVectorizer(max_df = 0.1, min_df = 1, stop_words='english') # term frequency-inverse document frequency (TF-IDF)
	terms_vec = terms_vectorizer.fit_transform(terms)
	terms_simi = pairwise.cosine_similarity(terms_vec)
	svd_tm = TruncatedSVD(n_components=min(100, terms_vec.shape[1] - 1), n_iter=100, random_state=0)
	#terms_svd = svd_tm.fit_transform(terms_vec.toarray())
	#terms_2d = model.fit_transform(terms_svd)

authors_2d = np.zeros((current_index, 2))
authors_vectorizer = CountVectorizer(max_df = 0.9, min_df = 3, stop_words='english') # term frequency (TF)
#authors_vectorizer = TfidfVectorizer(max_df = 0.05, min_df = 1, stop_words='english') # term frequency-inverse document frequency (TF-IDF)
authors_vec = authors_vectorizer.fit_transform(authors)
authors_simi = pairwise.cosine_similarity(authors_vec)
svd_au = TruncatedSVD(n_components=min(100, authors_vec.shape[1] - 1), n_iter=100, random_state=0)
#authors_svd = svd_au.fit_transform(authors_vec.toarray())
#authors_2d = model.fit_transform(authors_svd)	

pubtypes_2d = np.zeros((current_index, 2))
pubtypes_vectorizer = CountVectorizer(max_df = 0.9, min_df = 1, stop_words='english') # term frequency (TF)
#pubtypes_vectorizer = TfidfVectorizer(max_df = 0.05, min_df = 1, stop_words='english') # term frequency-inverse document frequency (TF-IDF)
pubtypes_vec = pubtypes_vectorizer.fit_transform(pubtypes)
pubtypes_simi = pairwise.cosine_similarity(pubtypes_vec)
svd_pt = TruncatedSVD(n_components=min(100, pubtypes_vec.shape[1] - 1), n_iter=100, random_state=0)
#pubtypes_svd = svd_pt.fit_transform(pubtypes_vec.toarray())
#pubtypes_2d = model.fit_transform(pubtypes_svd)
'''

'''
# Combine all features into a feature space
combine_vec = hstack((titles_vec, abstracts_vec, terms_vec, authors_vec, pubtypes_vec)) # all elements
combine_text_vec = hstack((titles_vec, abstracts_vec)) # text only (title and abstract)
#combine_simi = pairwise.cosine_similarity(combine_vec)
combine_simi = titles_simi*0.6 + abstracts_simi + terms_simi*0.5 + authors_simi*0.5 + pubtypes_simi*0.4

# Call SVD (high dimension->100) and t-SNE (100->2) for dimensionality reduction 
#svd_com = TruncatedSVD(n_components=min(100, combine_vec.shape[1] - 1), n_iter=1000, random_state=0)
#combine_svd = svd_com.fit_transform(combine_vec.toarray())
#combine_2d = model.fit_transform(combine_svd)

# [in case of python memory error] output the feature matrix to matlab for t-sne
np.savetxt(directory + 'matlab_input.txt', combine_vec.toarray(), delimiter=' ', fmt='%.2f')
# After implementing Matlab for t-sne, read in the resulted 2D map

matlab_index = 0
combine_2d = np.zeros((current_index, 2))
if os.path.isfile(directory + "matlab_output.txt"):
	#combine_2d = np.zeros((current_index, 2))
	with open(directory + "matlab_output.txt") as f:
		for line in f:
			if matlab_index >= current_index:
				break
			combine_2d[matlab_index][0] = float(line.split()[0])
			combine_2d[matlab_index][1] = float(line.split()[1])
			matlab_index += 1
'''	

#model2 = TSNE(n_components = 2, init='pca', perplexity=50, n_iter=1000, early_exaggeration=10.0, random_state = 0)
#model2 = TSNE(n_components = 2, init='pca', perplexity=100, n_iter=1000)
#combine_vec2 = hstack((titles_vec, abstracts_vec, terms_vec))
#combine_2d = model2.fit_transform(combine_vec2.toarray())
	
'''
# standardize the data (might not be used)
min_max_scaler = preprocessing.MinMaxScaler()
titles_scaled = min_max_scaler.fit_transform(titles_vec.toarray())

# experiment of using different model for dimensionality reduction (experiment only)
model = TSNE(n_components = 2, init='pca', random_state = 0)
model = Isomap(n_components = 2, n_neighbors = 100)
model = MDS(n_components=2, max_iter=1000)
model = SpectralEmbedding(n_components=2)
np.set_printoptions(suppress = True)

titles_2d = model.fit_transform(titles_vec.toarray())
plt.scatter(titles_2d[:, 0], titles_2d[:, 1])
plt.show()

abstracts_2d = model.fit_transform(abstracts_vec.toarray())
plt.scatter(abstracts_2d[:, 0], abstracts_2d[:, 1])
plt.show()

keywords_2d = model.fit_transform(keywords_vec.toarray())
plt.scatter(keywords_2d[:, 0], keywords_2d[:, 1])
plt.show()

'''

'''
# Visualize the clustering (experiment only)
def plot_clustering(X_2d, X, labels, title=None):
    x_min, x_max = np.min(X_2d, axis=0), np.max(X_2d, axis=0)
    X_2d = (X_2d - x_min) / (x_max - x_min)

    plt.figure(figsize=(6, 4))
    for i in range(X_2d.shape[0]):
        plt.scatter(X_2d[i, 0], X_2d[i, 1], marker = markers[i], s = 75, color = plt.cm.spectral(labels[i] / 20.))
    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title, size=17)
    plt.axis('off')
    plt.tight_layout()

for linkage in ('ward', 'average', 'complete'):
	clustering = AgglomerativeClustering(linkage = linkage, n_clusters = 9)
	#clustering.fit(combine_2d)
	clustering.fit(combine_vec.toarray())
	plot_clustering(combine_2d, combine_vec.toarray(), clustering.labels_, "%s linkage" % linkage)
#plt.show()
'''

# (Update 2017/9/20) Use Paragraph Vector (doc2vec from gensim) to get low-dimensional (i.e. 200) representations of title/abstract, then call t-SNE for 2D representations.
#################################
vector_dimension = 200
#vector_dimension = 400
'''
# For title only
titles_sentences = []
label_index = 0
for title in titles_all:
	sentence = models.doc2vec.LabeledSentence(words = title.replace('.', '').split(), tags = ['Title_%s' % label_index])
	titles_sentences.append(sentence)
	label_index += 1
# format: model = Doc2Vec(documents, size=100, window=8, min_count=5, workers=4, iter=50)
model_doc2vec = models.Doc2Vec(alpha=0.025, min_alpha=0.025, size=vector_dimension, window=8, min_count=2)  # use fixed learning rate
model_doc2vec.build_vocab(titles_sentences)
for epoch in range(20): # run for 20 passes for better performance
	model_doc2vec.train(titles_sentences, total_examples=len(titles_sentences), epochs=1)
	model_doc2vec.alpha -= 0.001
	model_doc2vec.min_alpha = model_doc2vec.alpha
	shuffle(titles_sentences) # shuffle for better performance
# model_doc2vec.docvecs.most_similar(["Title_0"])	
titles_doc2vec = []	
# arrage the resulting 100D representations of titles
for index in range(current_index):
	titles_doc2vec.append(model_doc2vec.docvecs['Title_%s' % index])
# call t-SNE to get 2D representations
title_2d = model.fit_transform(titles_doc2vec)

# For abstract only	
abstracts_sentences = []
label_index = 0
for abstract in abstracts_all:
	sentence = models.doc2vec.LabeledSentence(words = abstract.replace('.', '').split(), tags = ['Abstract_%s' % label_index])
	abstracts_sentences.append(sentence)
	label_index += 1
model_doc2vec = models.Doc2Vec(alpha=0.025, min_alpha=0.025, size=vector_dimension, window=8, min_count=2)  # use fixed learning rate
model_doc2vec.build_vocab(abstracts_sentences)
for epoch in range(20): # run for 20 passes for better performance
	model_doc2vec.train(abstracts_sentences, total_examples=len(abstracts_sentences), epochs=1)
	model_doc2vec.alpha -= 0.001
	model_doc2vec.min_alpha = model_doc2vec.alpha
	shuffle(abstracts_sentences) # shuffle for better performance
# model_doc2vec.docvecs.most_similar(["Abstract_0"])	
abstracts_doc2vec = []	
# arrage the resulting 100D representations of titles
for index in range(current_index):
	abstracts_doc2vec.append(model_doc2vec.docvecs['Abstract_%s' % index])
# call t-SNE to get 2D representations
abstract_2d = model.fit_transform(abstracts_doc2vec)
'''

# For the combination of title and abstract
texts_sentences = [] # combine title and abstract
label_index = 0
#for (title, abstract) in zip(titles_all, abstracts_all):
for (title, abstract) in zip(titles_all_simple, abstracts_all_simple):
	sentence = models.doc2vec.LabeledSentence(words = title.replace('.', '').split() + abstract.replace('.', '').split(), tags = ['Text_%s' % label_index])
	texts_sentences.append(sentence)
	label_index += 1
model_doc2vec = models.Doc2Vec(alpha=0.025, min_alpha=0.025, size=vector_dimension, window=8, min_count=2)
model_doc2vec.build_vocab(texts_sentences)
for epoch in range(20): # run for 20 passes for better performance
	model_doc2vec.train(texts_sentences, total_examples=len(texts_sentences), epochs=1)
	model_doc2vec.alpha -= 0.001
	model_doc2vec.min_alpha = model_doc2vec.alpha
	shuffle(texts_sentences) # shuffle for better performance
# model_doc2vec.docvecs.most_similar(["Abstract_0"])	
texts_doc2vec = []
texts_doc2vec_normArticle = [] # normalization for each article vector
# arrange the resulting representations (e.g. 200 dimensional) of titles
for index in range(current_index):
	current_vector = model_doc2vec.docvecs['Text_%s' % index];
	texts_doc2vec.append(current_vector)	
	#texts_doc2vec_normArticle.append(current_vector / np.linalg.norm(current_vector))
	texts_doc2vec_normArticle.append(np.divide(np.subtract(current_vector, min(current_vector)), (max(current_vector) - min(current_vector))))
	
texts_doc2vec_normDim = [[] for i in range(current_index)] # normalization for each dimension
for i in range(vector_dimension):
	temp_dimension_vector = []
	for index in range(current_index):
		temp_dimension_vector.append(model_doc2vec.docvecs['Text_%s' % index][i])
	#temp_dimension_vector_norm = temp_dimension_vector / np.linalg.norm(temp_dimension_vector)
	temp_dimension_vector_norm = np.divide(np.subtract(temp_dimension_vector, min(temp_dimension_vector)), (max(temp_dimension_vector) - min(temp_dimension_vector)))
	for index in range(current_index):
		texts_doc2vec_normDim[index].append(temp_dimension_vector_norm[index])
	
# call t-SNE to get 2D representations
text_2d = model.fit_transform(texts_doc2vec)
#################################


# (Update 2017/12/16) Use our advanced "Paragraph Vector + Attention Mechanism" model (leverage doc2vec from gensim) to get low-dimensional (i.e. 100) representations of title/abstract, then call t-SNE for 2D representations.
# I will call this model as "PA"
#################################

# Sample attention words (Notes: for some terms, better to list the version with/without "-")
# ACE inhibitors
'''
attentions = ["adult", "hypertension", "heart failure", "high cardiovascular risk factors", "diabetic nephropathy", "nondiabetic nephropathy", "myocardial infarction", 
				"angiotensin converting enzyme", "angiotensin-converting enzyme", "angiotensin-converting-enzyme", "ACE",
				"effectiveness", "safety", "adverse effect", 
				"long-term", "short-term",
				"demographics", "age", "racial", "gender", "co-morbidities", "comorbidities"]
'''
# ADHD

attentions = ["attention deficit hyperactivity disorder", "adhd", "disorder", "deficit", "hyperactivity",
				"adult", "child",
				"treat", "treatment",
				"quality of life", "life quality"]
		
# Antihistamines
'''
attentions = ["outpatient", "allergic rhinitis", "urticaria",
				"antihistamines", "safety", "adverse event", "adverse effect", "demographics", "age", "racial", "gender",
				"drug-drug interaction", "drug-disease interaction", "co-morbidities", "comorbidities", "pregnancy", "effective"]
'''
# Atypical Antipsychotics
'''
attentions = ["adult", "schizophrenia", "psychoses", "bipolar mania", "dementia", "youth", "autism",
				"behavioral", "disorder", "disruptive behavior disorder", "attention deficit hyperactivity disorder", "atypical antipsychotic drug", 
				"efficacy", "effectiveness", "safety", "adverse effect",
				"demographics", "age", "racial", "gender",
				"co-morbidities", "comorbidities"]
'''		
# Beta Blockers
'''
attentions = ["adult",
				"hypertension", "angina", "coronary artery bypass graft", "myocardial infarction", "heart failure", "atrial arrhythmia", "migraine", "bleeding esophageal varices", 
				"beta blocker", "betablocker", "beta-blocker",
				"effectiveness", "safety", "adverse",
				"demographics", "age", "racial", "gender",
				"drug-drug interaction", "drug-disease interaction", "co-morbidities", "comorbidities"]
'''	
# Calcium Channel Blockers
'''		
attentions = ["adult", "hypertension", "angina", "supraventricular arrhythmias", "systolic dysfunction",
				"blood pressure", "left ventricular ejection fraction", "LVEF",
				"calcium channel blocker", "CCB",
				"effectiveness", "safety", "adverse effect", "adverse outcome",
				"demographics", "age", "racial", "gender",
				"drug-drug interaction", "drug-disease interaction", "co-morbidities", "comorbidities"]	
'''						
# Estrogen
'''
attentions = [
				#"women", "perimenopausal", "postmenopausal", "peri menopausal", "post menopausal",
				"perimenopausal women", "postmenopausal women",
				"estrogen", "estrogen preparation",
				"menopause", "symptom", "bone", "fractures",
				#"reduce symptom of menopause", "reduce menopausal symptom", "prevent low bone density and fractures",
				"short-term use", "long-term use",
				"effectiveness", "safety", "adverse",
				"demographics", "other medication", "co-morbidities", "comorbidities"]
'''				
# NSAIDS
'''
attentions = ["coxibs", "NSAID", "Non-steroidal Anti-inflammatory", "Nonsteroidal Antiinflammatory",
				"Cyclo-oxygenase (COX)-2 Inhibitors", "Cyclo-oxygenase-2 (COX)-2 Inhibitors", "Cyclo-oxygenase-2 Inhibitors", "COX-2 Inhibitors",
				"Cyclooxygenase-2", "COX-2",
				"Cyclooxygenase2", "COX2",
				#"inhibitors",
				#"Chronic pain from osteoarthritis", "Rheumatoid arthritis", "Soft-tissue pain", "Back pain", "Ankylosing spondylitis",
				"combination", "antiulcer", "musculoskeletal pain",
				"effectiveness", "efficacy", "tolerability", "safety", "adverse",
				#"cardiovascular events", "GI events", "Tolerability",
				"demographics", "other medication", "aspirin", "co-morbidities", "comorbidities"]
'''
# Opioids
'''
attentions = [	"adult", "pain",
				"opioids", "placebo", 
				"effectiveness", "safety", "pattern", "superior", "adverse", 
				"reduce pain", "improve functional outcomes",
				"race", "age", "sex", "type of pain"]
'''
# Oral Hypoglycemics
'''
attentions = ["adult", "Type 2 diabetes", 
				"oral hypoglycemics", "sulfonylureas", "short-acting secretagogues", "short acting secretagogues",
				"clinically relevant outcome", "reduce HbA1C level", "safety", "adverse effect",
				"demographics", "age", "racial", "gender",
				"concomitant medications", "drug-drug interaction", "co-morbidities", "comorbidities", "obesity", "history of hypoglycemic",
				"effective", "adverse effect"]
'''	
# Proton Pump Inhibitors
'''
attentions = ["proton pump inhibitor", "proton-pump-inhibitor", "PPI", "H2-RA",
				"GERD", "gastroesophageal reflux", "reflux gastroesophageal", "ulcer", "Helicobacter pylori",
				"effectiveness", "adverse", "incidence", "complication", 
				#"healing esophagitis", "reduce symptom", "prevent relapse",
				#"improve endoscopic healing", "prevent NSAID-induced ulcer", "improve eradication rates", "gastroesophageal reflux"
				"demographics", "other medication", "co-morbidities", "comorbidities"]
'''
# Skeletal Muscle Relaxants
'''
attentions = [	"adult", "pediatric",
				"muscle relaxant", "different",
				"diazepam", "clonazepam", "clorazepate", "carisoprodol", "methocarbamol", "baclofen", "chlorzoxazone", "cyclobenzaprine", "dantrolene", "metaxalone", "orphenadrine", "tizanidine",
				"neurologic", "psychosis", "neuropathic",
				"spasticity", "musculoskeletal", "spasms", "pain",
				"effectiveness", "efficacy", "incidence", "adverse", "addiction", "abuse", "quality of life", "withdrawal",
				"reduce symptom", "improve functional outcome",
				"subpopulation", "subgroup"]
'''
'''
attentions = [
				"muscle relaxant",
				"diazepam", "clonazepam", "clorazepate", "carisoprodol", "methocarbamol", "baclofen", "chlorzoxazone", "cyclobenzaprine", "dantrolene", "metaxalone", "orphenadrine", "tizanidine",
				"neurologic", "psychosis", "neuropathic"
				#"abuse", "addiction", "reduce", "relieve"
				]				
'''
# Triptans
'''
attentions = [	"adult", "migraine", "headache",
				"triptans", 
				"Almotriptan", "Axert",
				"Eletriptan", "Relpax",
				"Frovatriptan", "Frova",
				"Naratriptan", "Amerge",
				"Rizatriptan", "Maxalt",
				"Rizatriptan", 
				"Sumatriptan", "Imitrex",
				"Zolmitriptan", "Zomig",
				"Zolmitriptan",
				#"reduce the severity", "reduce the duration of symptom", "improve functional outcome", "improve quality of life",
				"severity", "duration", "functional outcome", "quality of life",
				"efficacy", "effectiveness", "safety", "tolerability", "incidence", "complication", "adverse", 
				"subgroup", "demographics", "other medication", "co-morbidities", "comorbidities"]
'''
# Urinaryincontinence
'''
attentions = ["adult", "urinary urge incontinence", "urinary incontinence", "overactive bladder", "anticholinergic incontinence", "effectiveness",
				"safety", "adverse effect", "demographics", "age", "racial", "gender", "co-morbidities", "comorbidities"]
				
'''
# Statins
'''
attentions = ["statins", 
				"LDL-c", "LDLc", "LDL", "National Cholesterol Education Panel", "NCEP",
				"HDL-c", "HDLc", "HDL",
				"myocardial infarction", "CHD", "angina", "mortality", "stroke", "revascularization", "graft", "angioplasty", "stenting",
				"adverse", "effectiveness", "safety", "demographic"
				#"demographic", "age", "sex", "race",
				#"population", "other medication", "drug-drug interaction",
				]
'''	

'''							
# (1) The basic doc2vec model - Train the model based on the context of documents. One additional consideration is: consider a document as multiple sentences (break a document into sentences, rather than consider a document as a whole piece of texts)
texts_sentences_PA = []
label_index = 0
# generate context sentences (as gensim labeled sentences)
for (title, abstract) in zip(titles_all_simple, abstracts_all_simple): # take the whole document (multiple sentences)
	# context from title and abstract (abstract as a whole text piece) respectively
	sentence_labeled = models.doc2vec.LabeledSentence(words = title.replace(".", "").split(), tags = ['Text_%s' % label_index])
	texts_sentences_PA.append(sentence_labeled)
	sentence_labeled = models.doc2vec.LabeledSentence(words = abstract.replace(".", "").split(), tags = ['Text_%s' % label_index])
	texts_sentences_PA.append(sentence_labeled)
	
	 # break a document's abstract into sentences. Each sentence has the same document label
	for sentence in abstract.split("."): # break a document into sentences. Each of a document's sentences has the same document label
		sentence = sentence.strip()
		if len(sentence) <= 0:
			continue
		sentence_labeled = models.doc2vec.LabeledSentence(words = sentence.split(), tags = ['Text_%s' % label_index])
		texts_sentences_PA.append(sentence_labeled)

		# additionally, for attentions (key phrases) consisting of multiple words, consider that as a "special word".
		# for attentions as a single word, this will perform as an up-sampling.
		#for attention in attentions:
		#	attention = myStemmer(attention)
		#	if attention in sentence:
		#		words = []
		#		for word in sentence.split():
		#			if word not in attention and sentence.find(word) != sentence.find(attention):
		#				words.append(word)
		#			elif word in attention and sentence.find(word) == sentence.find(attention):
		#				words.append(attention.replace(" ", "-"))
		#		sentence_labeled = models.doc2vec.LabeledSentence(words = words, tags = ['Text_%s' % label_index])
		#		texts_sentences_PA.append(sentence_labeled)		
	label_index += 1
# define the model	
model_doc2vec_PA = models.Doc2Vec(alpha=0.025, min_alpha=0.025, size=vector_dimension, window=8, min_count=2)
# pre-train the word vectors
model_doc2vec_PA.build_vocab(texts_sentences_PA) # build the vocabulary
# check the word vocabulary (learned word vectors) - (model_doc2vec_PA.wv is a) gensim.models.keyedvectors.KeyedVectors object
#model_doc2vec_PA.wv.vocab['adhd'].count
#model_doc2vec_PA.wv.similarity('adult', 'child')
#model_doc2vec_PA.wv.most_similar('child')
# train the doc vectors
for epoch in range(20): # run for 20 passes for better performance
	model_doc2vec_PA.train(texts_sentences_PA, total_examples=len(texts_sentences_PA), epochs=1)
	model_doc2vec_PA.alpha -= 0.001
	model_doc2vec_PA.min_alpha = model_doc2vec_PA.alpha
	shuffle(texts_sentences_PA) # shuffle for better performance

# (2) Emphasis the Attention - the context (sentence enclosing) of attention words
texts_attentions_PA = []
#distance = 8 # consider words within a certain distance to the attention (specify a window size); set distance=0 if considering all words in the sentence (unlimited window size)
for sentence in texts_sentences_PA:
	for attention in attentions:
		attention = myStemmer(attention)
		if attention in sentence.words: # attention word is found in a sentence
			for word in sentence.words:
				# check the distance of the word to the attention in the sentence
				#positions = [i for i,x in enumerate(sentence.words) if x == word]
				if word not in attention and word in model_doc2vec_PA.wv.vocab.keys() and 0.5*len(texts_sentences_PA) > model_doc2vec_PA.wv.vocab[word].count > 0.01*len(texts_sentences_PA):
					words = []
					words.append(attention)
					words.append(word)
					sentence_labeled = models.doc2vec.LabeledSentence(words = words, tags = sentence.tags)				
					texts_attentions_PA.append(sentence_labeled)
# use document topken + attention to predict the attention's context word, one by one
model_doc2vec_PA.window = 2
model_doc2vec_PA.alpha = 0.025 # 0.025
model_doc2vec_PA.min_alpha = 0.025 # 0.025
for epoch in range(5): # run for 5 passes for better performance
	model_doc2vec_PA.train(texts_attentions_PA, total_examples=len(texts_attentions_PA), epochs=1)
	model_doc2vec_PA.alpha -= 0.005
	model_doc2vec_PA.min_alpha = model_doc2vec_PA.alpha
	shuffle(texts_attentions_PA) # shuffle for better performance

# model_doc2vec_PA.docvecs.most_similar(["Abstract_0"])	
texts_doc2vec_PA = []
texts_doc2vec_PA_normArticle = [] # normalization for each article
# arrange the resulting representations (e.g. 200 dimensional) of titles
for index in range(current_index):
	current_vector = model_doc2vec_PA.docvecs['Text_%s' % index]
	texts_doc2vec_PA.append(current_vector)
	texts_doc2vec_PA_normArticle.append(current_vector / np.linalg.norm(current_vector))
	
texts_doc2vec_PA_normDim = [[] for i in range(current_index)] # normalization for each dimension
for i in range(vector_dimension):
	temp_dimension_vector = []
	for index in range(current_index):
		temp_dimension_vector.append(model_doc2vec_PA.docvecs['Text_%s' % index][i])
	temp_dimension_vector_norm = temp_dimension_vector / np.linalg.norm(temp_dimension_vector)
	for index in range(current_index):
		texts_doc2vec_PA_normDim[index].append(temp_dimension_vector_norm[index])
			
# call t-SNE to get 2D representations
text_2d_PA = model.fit_transform(texts_doc2vec_PA)
#################################
'''

# (update 2017/12/19) Optional Codes - to evaluate the performance based on external (gold-standard) labels of documents e.g. the set of highlighted/relevant documents
# Simulate an active article recommendation process, and record dynamic precision & recall, and final WSS95
'''
############################################
# (1) score all articles based on the abstract
keyword_scores = []
for index in range(current_index):
	element = {}
	element["id"] = index
	score = 0
	for keyword in prior_keywords:
		score += abstracts_all[index].count(myStemmer(keyword))
	element["score"] = score
	keyword_scores.append(element)
keyword_scores_sorted = sorted(keyword_scores, key=lambda k: k["score"], reverse=True)
# (2) get the initial article based on the matching with prior-keywords - until a relevant (highlighed) article is reached
statuses = np.zeros(current_index) + 1 # record whether an article is recommended or not
recommendations = [] # the sequence of recommendations based on the original Paragraph Vector model
relevancies = [] # recommended and relevant (based on the prior highlight list) articles
for item in keyword_scores_sorted:
	if ids[item['id']] in highlight:
		recommendations.append(item['id'])
		relevancies.append(item['id'])
		statuses[item['id']] = 0
		break
	else:
		recommendations.append(item['id'])
		statuses[item['id']] = 0		
# (3) simulate an iterative article recommendation process, starting with the best_match, and recommending the most similar article(s)
# For the original Paragraph Vector model
recommendations_PV = recommendations[:]
relevancies_PV = relevancies[:]
statuses_PV = list(statuses)
for i in range(len(recommendations_PV), current_index): # iterations
	#relevancies_PV_labels = ["Text_%s" % index for index in relevancies_PV]
	#model_doc2vec.docvecs.most_similar(relevancies_PV_labels, topn=10)
	relevancy_vector = sum(x for i, x in enumerate(texts_doc2vec) if i in relevancies_PV)
	#print relevancies_PV
	relevancy_vector = relevancy_vector/len(relevancies_PV)
	best_similarity = -1
	recommendation = -1
	for index in range(current_index): # find the most similar articles to the list of relevancies
		if statuses_PV[index] == 0:
			continue
		similarity = 1 - spatial.distance.cosine(relevancy_vector, texts_doc2vec[index])
		if similarity > best_similarity:
			best_similarity = similarity
			recommendation = index
	recommendations_PV.append(recommendation)
	statuses_PV[recommendation] = 0
	if ids[recommendation] in highlight:
		relevancies_PV.append(recommendation)
# For our Paragraph Vector + Attention Mechanism (PA) model
recommendations_PA = recommendations[:]
relevancies_PA = relevancies[:]
statuses_PA = list(statuses)
for i in range(len(recommendations_PA), current_index): # iterations
	#relevancies_PV_labels = ["Text_%s" % index for index in relevancies_PV]
	#model_doc2vec.docvecs.most_similar(relevancies_PV_labels, topn=10)
	relevancy_vector = sum(x for i, x in enumerate(texts_doc2vec_PA) if i in relevancies_PA)
	#print relevancies_PA
	relevancy_vector = relevancy_vector/len(relevancies_PA)
	best_similarity = -1
	recommendation = -1
	for index in range(current_index): # find the most similar articles to the list of relevancies
		if statuses_PA[index] == 0:
			continue
		similarity = 1 - spatial.distance.cosine(relevancy_vector, texts_doc2vec_PA[index])
		if similarity > best_similarity:
			best_similarity = similarity
			recommendation = index
	recommendations_PA.append(recommendation)
	statuses_PA[recommendation] = 0
	if ids[recommendation] in highlight:
		relevancies_PA.append(recommendation)
# (4) calculate precision, recall, and WSS95 as performance measurements
# For the original Paragraph Vector model
precisions_PV = []
recalls_PV = []
wss95_PV = 0
wss95_flag = 1
current_relevant = 0
for index in range(current_index):
	if ids[recommendations_PV[index]] in highlight:
		current_relevant += 1
	precision = float(current_relevant)/float(index + 1)
	recall = float(current_relevant)/float(len(highlight))	
	precisions_PV.append(precision)
	recalls_PV.append(recall)
	if recall >= 0.95 and wss95_flag == 1:
		wss95_PV = float(current_index - index)/float(current_index + 1)
		wss95_flag = 0
# For our Paragraph Vector + Attention Mechanism (PA) model
precisions_PA = []
recalls_PA = []
wss95_PA = 0
wss95_flag = 1
current_relevant = 0
for index in range(current_index):
	if ids[recommendations_PA[index]] in highlight:
		current_relevant += 1
	precision = float(current_relevant)/float(index + 1)
	recall = float(current_relevant)/float(len(highlight))	
	precisions_PA.append(precision)
	recalls_PA.append(recall)
	if recall >= 0.95 and wss95_flag == 1:
		wss95_PA = float(current_index - index)/float(current_index + 1)
		wss95_flag = 0
##########################################
'''

# (Update 2017/12/20)
# for semantic vectors with a dimensionality of n (e.g. 200), get the most active m (e.g. 50) dimensions
# here, "active" means a large difference across all document instances
#################################
# For the original Paragraph Vector model
#np.shape(texts_doc2vec)
value_differences = [] # the difference of each dimension
for i in range(vector_dimension):
	current_column = [item[i] for item in texts_doc2vec] # get each dimension (column)
	diff = max(current_column) - min(current_column)
	element = {}
	element['dimensionID'] = i
	element['difference'] = diff
	value_differences.append(element)
value_differences_sorted = sorted(value_differences, key=lambda k: k['difference'], reverse=True)
active_dimension = 50
active_index = [] # the index of the active dimensions
for i in range(active_dimension):
	active_index.append(value_differences_sorted[i]['dimensionID'])
texts_doc2vec_active = [] # the "semantic vectors" with the most active dimensions	
for i in range(current_index):
	texts_doc2vec_active.append([x for j, x in enumerate(texts_doc2vec[i]) if j in active_index])
	
# For our Paragraph Vector + Attention Mechanism (PA) model
'''
#np.shape(texts_doc2vec)
value_differences_PA = [] # the difference of each dimension
for i in range(vector_dimension):
	current_column = [item[i] for item in texts_doc2vec_PA] # get each dimension (column)
	diff = max(current_column) - min(current_column)
	element = {}
	element['dimensionID'] = i
	element['difference'] = diff
	value_differences_PA.append(element)
value_differences_sorted_PA = sorted(value_differences_PA, key=lambda k: k['difference'], reverse=True)
active_dimension = 50
active_index_PA = [] # the index of the active dimensions
for i in range(active_dimension):
	active_index_PA.append(value_differences_sorted_PA[i]['dimensionID'])
texts_doc2vec_active_PA = [] # the "semantic vectors" with the most active dimensions	
for i in range(current_index):
	texts_doc2vec_active_PA.append([x for j, x in enumerate(texts_doc2vec_PA[i]) if j in active_index_PA])	
'''
#################################


# Graphs with force-directed layouts
'''
#################################
# handle the overlap of dots
offset_x = max(keywords_2d[:, 0]) - min(keywords_2d[:, 0])
offset_y = max(keywords_2d[:, 1]) - min(keywords_2d[:, 1])

# Create a graph and place vertices based on force-directed layout
g = Graph()
g.add_vertices(current_index)
# edge sampling with the AS scheme - keep the top 10% edges first, then keep top deg^0.5 edges for each node.
combine_simi_copy = np.zeros((current_index, current_index))
combine_simi_copy = combine_simi
for index1 in range(0, current_index):
		combine_simi_copy[index1][index1] = 0
# initial prune
# keep the top 10% edges for each node
flag_10percent_value = np.zeros(current_index)
for index1 in range(0, current_index):
	flag_10percent_index = int(0.1 * (current_index-1))
	flag_10percent_value[index1] = np.partition(combine_simi_copy[index1], int(-flag_10percent_index))[int(-flag_10percent_index)]
	for index2 in range(0, current_index):
		if combine_simi_copy[index1][index2] < flag_10percent_value[index1]:
			combine_simi_copy[index1][index2] = 0

# get the degree after initial prune
degree = np.zeros(current_index)
for index in range(0, current_index):
	degree[index] = np.count_nonzero(combine_simi_copy[index])
# edge samlpling - keep the top degree^0.5 edges for each node
count_edge = np.zeros(current_index)
combine_simi_final = np.zeros((current_index, current_index))
for index1 in range(0, current_index):
	count_edge[index1] = int(pow(degree[index1], 0.5)) + 1
	#if count_edge[index1] == 0:
	#	count_edge[index1] = 1
	threshold_value = np.partition(combine_simi_copy[index1], int(-count_edge[index1]))[int(-count_edge[index1])]
	for index2 in range(0, current_index):
		#if combine_simi_copy[index1][index2] >= threshold_value and index2 > index1:
		if combine_simi_copy[index1][index2] >= threshold_value:
			g.add_edge(index1, index2, weight = combine_simi_copy[index1][index2])
			combine_simi_final[index1][index2] = combine_simi_copy[index1][index2]
weight = g.es['weight']
#layout_fr = g.layout("fr")
layout_fr = g.layout_fruchterman_reingold(maxiter=1000, weights=weight)
layout_kk = g.layout("kk")
layout_lgl = g.layout("lgl")
#plot(g, layout = layout)
#################################
'''

# Community Detection (clustering)
'''
#################################
# Optimal single level
community = g.community_multilevel(weights=weight) # Louvain method (Blondel 2008)
modularity = community.modularity
membership = community.membership

# Multilevel Community Detection with Louvain method (network clustering) for sparsified article networks
communities_raw = g.community_multilevel(weights=weight, return_levels=True) # Louvain method (Blondel 2008)
communities = [] # Make 3 clustering levels
if(len(communities_raw) == 3):
	communities = communities_raw
elif(len(communities_raw) > 3):
	for i in range(len(communities_raw) - 2, len(communities_raw)):
		communities.append(communities[i])
elif(len(communities_raw) == 2):
	communities.append(communities_raw[0])
	communities.append(communities_raw[0])
	communities.append(communities_raw[1])
elif(len(communities_raw) == 1):
	communities.append(communities_raw[0])
	communities.append(communities_raw[0])
	communities.append(communities_raw[0])
else:
	print "error in community detection"

modularities = []
memberships = [] 
number_level = 0 # the total nummber of clustering levels
optimal_level_index = 0
optimal_modularity = 0
for level in communities: # iterate through different levels of clustering
	number_level += 1
	if level.modularity > optimal_modularity:
		optimal_modularity = level.modularity
		optimal_level_index = number_level - 1
	modularities.append(level.modularity)
	memberships.append(level.membership)
level_size = np.zeros(number_level) # the number of communities for each clustering level
for i in range(0, number_level):
	level_size [i] = len(set(memberships[i]))

# Get the mapping of community IDs among different clustering levels
community_map = np.zeros((int(max(level_size)), number_level)) # global map: level0, level1, level2
community_map = community_map - 1
for index in range(0, current_index):
	community = int(memberships[0][index])
	if community not in community_map[:,0]:
		community_map[community][0] = community
		for level in range(1, number_level):
			community_map[community][level] = memberships[level][index]
community_maps = [] # the map between every level to the top level, the format is: current_level_cluster_id, top_level_cluster_id, sub_id_within_the_top_cluster
for level in range(0, number_level):
	community_map_current = np.zeros((int(level_size[level]), 3))
	community_map_current = community_map_current - 1
	cluster_count = np.zeros(int(level_size[number_level - 1]))
	for index in range(0, current_index):
		community = int(memberships[level][index])
		if community not in community_map_current[:,0]:
			community_map_current[community][0] = community
			top_community = memberships[number_level-1][index]
			community_map_current[community][1] = top_community
			community_map_current[community][2] = cluster_count[top_community]
			cluster_count[top_community] += 1
	community_maps.append(community_map_current)

# (update 2017/10/02) idenitfy the keywords for each community/cluster, using the nltk-rake library
cluster_titles = [["" for j in range(len(community_maps[i]))] for i in range(number_level)]
cluster_abstracts = [["" for j in range(len(community_maps[i]))] for i in range(number_level)]
cluster_title_keywords = [[[] for j in range(len(community_maps[i]))] for i in range(number_level)]
cluster_abstract_keywords = [[[] for j in range(len(community_maps[i]))] for i in range(number_level)]
for level in range(0, number_level):
    for index in range(0, current_index):
        cluster_id = memberships[level][index]
        cluster_titles[level][cluster_id] += titles_raw[index]
        cluster_titles[level][cluster_id] += " "
        cluster_abstracts[level][cluster_id] += abstracts_raw[index]
        cluster_abstracts[level][cluster_id] += " "
#r = Rake() # Uses stopwords for english from NLTK, and all puntuation characters.
rake = RakeKeywordExtractor()
# If you want to provide your own set of stop words and punctuations to
# r = Rake(<list of stopwords>, <string of puntuations to ignore>)
for level in range(0, number_level):
	for cluster in range(len(community_maps[level])):
		cluster_title_keywords[level][cluster] = rake.extract(cluster_titles[level][cluster], incl_scores=False)
		cluster_abstract_keywords[level][cluster] = rake.extract(cluster_abstracts[level][cluster], incl_scores=False)
						
# Implement hierarchical clustering on t-SNE based article maps, based on the community detection results (number of clusters on each level)
# For all elements (SVD + t-SNE)
memberships2 = []
for level in range(0, number_level):
	clustering = AgglomerativeClustering(linkage = 'ward', n_clusters = int(level_size[level]))
	#clustering.fit(combine_2d)
	clustering.fit(combine_vec.toarray())
	#plot_clustering(combine_2d, combine_vec.toarray(), clustering.labels_, "%s linkage" % 'ward')
	memberships2.append(clustering.labels_)
# Again, get the mapping of community IDs among different clustering levels
community_map2 = np.zeros((int(max(level_size)), number_level)) # global map: level0, level1, level2
community_map2 = community_map2 - 1
for index in range(0, current_index):
	community = int(memberships2[0][index])
	if community not in community_map2[:,0]:
		#print ("index-" + str(index) + ", community0-" + str(community) + ", community1-" + str(memberships2[1][index]) + ", community2-" + str(memberships2[2][index]));
		community_map2[community][0] = community
		for level in range(1, number_level):
			community_map2[community][level] = memberships2[level][index]
community_maps2 = [] # the map between every level to the top level, the format is: current_level_cluster_id, top_level_cluster_id, sub_id_within_the_top_cluster
for level in range(0, number_level):
	community_map_current = np.zeros((int(level_size[level]), 3))
	community_map_current = community_map_current - 1
	cluster_count = np.zeros(int(level_size[number_level - 1]))
	for index in range(0, current_index):
		community = int(memberships2[level][index])
		if community not in community_map_current[:,0]:
			community_map_current[community][0] = community
			top_community = memberships2[number_level-1][index]
			community_map_current[community][1] = top_community
			community_map_current[community][2] = cluster_count[top_community]
			cluster_count[top_community] += 1
	community_maps2.append(community_map_current)
	
# (update 2017/10/02) idenitfy the keywords for each community/cluster, using the nltk-rake library
cluster_titles2 = [["" for j in range(len(community_maps2[i]))] for i in range(number_level)]
cluster_abstracts2 = [["" for j in range(len(community_maps2[i]))] for i in range(number_level)]
cluster_title_keywords2 = [[[] for j in range(len(community_maps2[i]))] for i in range(number_level)]
cluster_abstract_keywords2 = [[[] for j in range(len(community_maps2[i]))] for i in range(number_level)]
for level in range(0, number_level):
    for index in range(0, current_index):
        cluster_id = memberships2[level][index]
        cluster_titles2[level][cluster_id] += titles_raw[index]
        cluster_titles2[level][cluster_id] += " "
        cluster_abstracts2[level][cluster_id] += abstracts_raw[index]
        cluster_abstracts2[level][cluster_id] += " "
#r = Rake() # Uses stopwords for english from NLTK, and all puntuation characters.
rake = RakeKeywordExtractor()
# If you want to provide your own set of stop words and punctuations to
# r = Rake(<list of stopwords>, <string of puntuations to ignore>)
for level in range(0, number_level):
	for cluster in range(len(community_maps2[level])):
		cluster_title_keywords2[level][cluster] = rake.extract(cluster_titles2[level][cluster], incl_scores=False)
		cluster_abstract_keywords2[level][cluster] = rake.extract(cluster_abstracts2[level][cluster], incl_scores=False)	

# For the original doc2vec method, using texts only - title and abstract (doc2vec + t-SNE)
memberships3 = []
silhouettes3 = [] # (update 2017/12/21) get the silhouette coefficient for each article sample
for level in range(0, number_level):
	clustering = AgglomerativeClustering(linkage = 'ward', n_clusters = int(level_size[level]))
	#clustering.fit(text_2d)
	clustering.fit(texts_doc2vec)
	#plot_clustering(combine_2d, combine_vec.toarray(), clustering.labels_, "%s linkage" % 'ward')
	memberships3.append(clustering.labels_)
	silhouettes3.append(silhouette_samples(texts_doc2vec, clustering.labels_))
# Again, get the mapping of community IDs among different clustering levels
community_map3 = np.zeros((int(max(level_size)), number_level)) # global map: level0, level1, level2
community_map3 = community_map3 - 1
for index in range(0, current_index):
	community = int(memberships3[0][index])
	if community not in community_map3[:,0]:
		#print ("index-" + str(index) + ", community0-" + str(community) + ", community1-" + str(memberships2[1][index]) + ", community2-" + str(memberships2[2][index]));
		community_map3[community][0] = community
		for level in range(1, number_level):
			community_map3[community][level] = memberships3[level][index]
community_maps3 = [] # the map between every level to the top level, the format is: current_level_cluster_id, top_level_cluster_id, sub_id_within_the_top_cluster
for level in range(0, number_level):
	community_map_current = np.zeros((int(level_size[level]), 3))
	community_map_current = community_map_current - 1
	cluster_count = np.zeros(int(level_size[number_level - 1]))
	for index in range(0, current_index):
		community = int(memberships3[level][index])
		if community not in community_map_current[:,0]:
			community_map_current[community][0] = community
			top_community = memberships3[number_level-1][index]
			community_map_current[community][1] = top_community
			community_map_current[community][2] = cluster_count[top_community]
			cluster_count[top_community] += 1
	community_maps3.append(community_map_current)
	
# (update 2017/10/02) idenitfy the keywords for each community/cluster, using the nltk-rake library
cluster_titles3 = [["" for j in range(len(community_maps3[i]))] for i in range(number_level)]
cluster_abstracts3 = [["" for j in range(len(community_maps3[i]))] for i in range(number_level)]
cluster_title_keywords3 = [[[] for j in range(len(community_maps3[i]))] for i in range(number_level)]
cluster_abstract_keywords3 = [[[] for j in range(len(community_maps3[i]))] for i in range(number_level)]
for level in range(0, number_level):
    for index in range(0, current_index):
        cluster_id = memberships3[level][index]
        cluster_titles3[level][cluster_id] += titles_raw[index]
        cluster_titles3[level][cluster_id] += " "
        cluster_abstracts3[level][cluster_id] += abstracts_raw[index]
        cluster_abstracts3[level][cluster_id] += " "
#r = Rake() # Uses stopwords for english from NLTK, and all puntuation characters.
rake = RakeKeywordExtractor()
# If you want to provide your own set of stop words and punctuations to
# r = Rake(<list of stopwords>, <string of puntuations to ignore>)
for level in range(0, number_level):
	for cluster in range(len(community_maps3[level])):
		cluster_title_keywords3[level][cluster] = rake.extract(cluster_titles3[level][cluster], incl_scores=False)
		cluster_abstract_keywords3[level][cluster] = rake.extract(cluster_abstracts3[level][cluster], incl_scores=False)				

# (Update 2017/12/19) Clustering and keywords detection for our advanced doc2vec method: Paragraph Vector + Attention (PA)
# using abstract (PA + t-SNE)
memberships4 = []
silhouettes4 = [] # (update 2017/12/21) get the silhouette coefficient for each article sample
for level in range(0, number_level):
	clustering = AgglomerativeClustering(linkage = 'ward', n_clusters = int(level_size[level]))
	#clustering.fit(text_2d_PA)
	clustering.fit(texts_doc2vec_PA)
	memberships4.append(clustering.labels_)
	silhouettes4.append(silhouette_samples(texts_doc2vec_PA, clustering.labels_))
# Again, get the mapping of community IDs among different clustering levels
community_map4 = np.zeros((int(max(level_size)), number_level)) # global map: level0, level1, level2
community_map4 = community_map4 - 1
for index in range(0, current_index):
	community = int(memberships4[0][index])
	if community not in community_map4[:,0]:
		#print ("index-" + str(index) + ", community0-" + str(community) + ", community1-" + str(memberships2[1][index]) + ", community2-" + str(memberships2[2][index]));
		community_map4[community][0] = community
		for level in range(1, number_level):
			community_map4[community][level] = memberships4[level][index]
community_maps4 = [] # the map between every level to the top level, the format is: current_level_cluster_id, top_level_cluster_id, sub_id_within_the_top_cluster
for level in range(0, number_level):
	community_map_current = np.zeros((int(level_size[level]), 3))
	community_map_current = community_map_current - 1
	cluster_count = np.zeros(int(level_size[number_level - 1]))
	for index in range(0, current_index):
		community = int(memberships4[level][index])
		if community not in community_map_current[:,0]:
			community_map_current[community][0] = community
			top_community = memberships4[number_level-1][index]
			community_map_current[community][1] = top_community
			community_map_current[community][2] = cluster_count[top_community]
			cluster_count[top_community] += 1
	community_maps4.append(community_map_current)
	
# idenitfy the keywords for each community/cluster, using the nltk-rake library
cluster_titles4 = [["" for j in range(len(community_maps4[i]))] for i in range(number_level)]
cluster_abstracts4 = [["" for j in range(len(community_maps4[i]))] for i in range(number_level)]
cluster_title_keywords4 = [[[] for j in range(len(community_maps4[i]))] for i in range(number_level)]
cluster_abstract_keywords4 = [[[] for j in range(len(community_maps4[i]))] for i in range(number_level)]
for level in range(0, number_level):
    for index in range(0, current_index):
        cluster_id = memberships4[level][index]
        cluster_titles4[level][cluster_id] += titles_raw[index]
        cluster_titles4[level][cluster_id] += " "
        cluster_abstracts4[level][cluster_id] += abstracts_raw[index]
        cluster_abstracts4[level][cluster_id] += " "
#r = Rake() # Uses stopwords for english from NLTK, and all puntuation characters.
rake = RakeKeywordExtractor()
# If you want to provide your own set of stop words and punctuations to
# r = Rake(<list of stopwords>, <string of puntuations to ignore>)
for level in range(0, number_level):
	for cluster in range(len(community_maps4[level])):
		cluster_title_keywords4[level][cluster] = rake.extract(cluster_titles4[level][cluster], incl_scores=False)
		cluster_abstract_keywords4[level][cluster] = rake.extract(cluster_abstracts4[level][cluster], incl_scores=False)				
'''

'''
# Reorganize the similarity matrix of combine_simi_final to avoid repeating edges (as an undirected graph, keep edges with source_id < target_id)
combine_simi_edge = np.zeros((current_index, current_index))
for index1 in range(0, current_index):
	for index2 in range(0, current_index):
		if index1 < index2 and combine_simi_final[index1][index2] > 0:
			combine_simi_edge[index1][index2] = combine_simi_final[index1][index2]
		if index1 > index2 and combine_simi_final[index1][index2] > 0 and combine_simi_final[index2][index1] == 0:
			#print(str(index1) + "-" + str(index2))
			combine_simi_edge[index2][index1] = combine_simi_final[index1][index2]	
			
# Output a gephi .gexf file (similar to XML) for Force Atlas Layout
edge_id = 0;
with open(directory + 'gephi_input.gexf', 'w') as gephi:
	gephi.write("<gexf xmlns=\"http://www.gexf.net/1.1draft\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:schemaLocation=\"http://www.gexf.net/1.1draft http://www.gexf.net/1.1draft/gexf.xsd\" version=\"1.1\">\n")
	gephi.write("<graph mode=\"static\" defaultedgetype=\"undirected\">\n")
	gephi.write("<attributes class=\"node\">\n")
	gephi.write("<attribute id=\"0\" title=\"Class\" type=\"string\"/>\n")
	gephi.write("</attributes>\n")
	gephi.write("<nodes>\n")
	for index in range(0, current_index):
		gephi.write("<node id=\"" + str(index+1) + "\" label=\"" + str(index+1) + "\" >\n")
		gephi.write("<attvalues><attvalue for=\"0\" value=\"" + str(if_highlight[index]) + "\"/></attvalues>\n")
		gephi.write("</node>\n")
	gephi.write("</nodes>\n")
	gephi.write("<edges>\n")
	for index1 in range(0, current_index):
		for index2 in range(0, current_index):
			if combine_simi_final[index1][index2] > 0:
				gephi.write("<edge id=\"" + str(edge_id+1) + "\" source=\"" + str(index1+1) + "\" target=\"" + str(index2+1) + "\" weight=\"" + str(combine_simi_final[index1][index2]) + "\"/>\n")
				edge_id += 1
	gephi.write("</edges>\n")
	gephi.write("</graph>\n")
	gephi.write("</gexf>\n")

# After the utilization of Gephi, Read in Gephi generated .gexf for node positions of Force Atlas Layout (similar to XML)
layout_fa = np.zeros((current_index, 2))
membership_gephi = np.zeros(current_index)
if os.path.isfile(directory + "gephi_output.gexf"):
	tree_gephi = ET.parse(directory + 'gephi_output.gexf')
	root_gephi = tree_gephi.getroot()
	node_count = 0
	version = "{http://www.gexf.net/1.3}"
	version_viz = "{http://www.gexf.net/1.3/viz}"
	if root_gephi.find(version + 'graph') is not None:
		graph = root_gephi.find(version + 'graph')
		nodes = graph.find(version + 'nodes')
		for node in nodes.findall(version + 'node'):
			for attvalue in node.find(version + 'attvalues').findall(version + 'attvalue'):
				if attvalue.get('for') == "modularity_class":
					membership_gephi[node_count] = attvalue.get('value')
			layout_fa[node_count][0] = node.find(version_viz + 'position').get('x')
			layout_fa[node_count][1] = node.find(version_viz + 'position').get('y')
			node_count += 1
'''			

# Modify the Force Atlas (FA) Layout to make communities/clusters more seperate with each other
'''
layout_fa_adjusts = []
layout_fa_adjusts2 = []
aggregate_rate = 1
aggregate_rate2 = 2
# get the graph center
graph_center = [0,0]
for index in range(0, current_index):
	graph_center += layout_fa[index]
graph_center = graph_center/current_index
# get the cluster centers (consider different clustering levels)
for level in range(0, number_level):
	cluster_number = int(level_size[level])
	cluster_center = np.zeros((cluster_number, 2))
	cluster_diff = np.zeros((cluster_number, 2))
	cluster_member_count = np.zeros(cluster_number)
	for index in range(0, current_index):
		cluster_id = memberships[level][index]
		cluster_center[cluster_id] += layout_fa[index]
		cluster_member_count[cluster_id] += 1	
	for index in range(0, cluster_number):
		cluster_center[index] = cluster_center[index]/cluster_member_count[index]
		cluster_diff[index] = cluster_center[index] - graph_center
		
	layout_adjust = np.zeros((current_index, 2))
	for index in range(0, current_index):
		cluster_id = memberships[level][index]
		layout_adjust[index] = layout_fa[index] + aggregate_rate*cluster_diff[cluster_id]
	layout_fa_adjusts.append(layout_adjust)
	
	layout_adjust2 = np.zeros((current_index, 2))
	for index in range(0, current_index):
		cluster_id = memberships[level][index]
		layout_adjust2[index] = layout_fa[index] + aggregate_rate2*cluster_diff[cluster_id]
	layout_fa_adjusts2.append(layout_adjust2)
	
# Modify the Fruchterman Reingold (FR) Layout to make communities/clusters more seperate with each other
layout_fr_adjusts = []
layout_fr_adjusts2 = []
# convert igraph generated layout to a 2D array
layout_fr_array = np.zeros((current_index, 2))
for index in range(0, current_index):
	layout_fr_array[index][0] = layout_fr[index][0]
	layout_fr_array[index][1] = layout_fr[index][1]
# get the graph center
graph_center = [0,0]
for index in range(0, current_index):
	graph_center += layout_fr_array[index]
graph_center = graph_center/current_index	
# get the cluster centers (consider different clustering levels)
for level in range(0, number_level):
	cluster_number = int(level_size[level])
	cluster_center = np.zeros((cluster_number, 2))
	cluster_diff = np.zeros((cluster_number, 2))
	cluster_member_count = np.zeros(cluster_number)
	for index in range(0, current_index):
		cluster_id = memberships[level][index]
		cluster_center[cluster_id] += layout_fr_array[index]
		cluster_member_count[cluster_id] += 1	
	for index in range(0, cluster_number):
		cluster_center[index] = cluster_center[index]/cluster_member_count[index]
		cluster_diff[index] = cluster_center[index] - graph_center
		
	layout_adjust = np.zeros((current_index, 2))
	for index in range(0, current_index):
		cluster_id = memberships[level][index]
		layout_adjust[index] = layout_fr_array[index] + aggregate_rate*cluster_diff[cluster_id]
	layout_fr_adjusts.append(layout_adjust)
	
	layout_adjust2 = np.zeros((current_index, 2))
	for index in range(0, current_index):
		cluster_id = memberships[level][index]
		layout_adjust2[index] = layout_fr_array[index] + aggregate_rate2*cluster_diff[cluster_id]
	layout_fr_adjusts2.append(layout_adjust2)

# Modify the t-SNE "Layout" to make clusters more seperate with each other
layout_tsne = combine_2d
layout_tsne_adjusts = []
layout_tsne_adjusts2 = []
# get the graph center
graph_center = [0,0]
for index in range(0, current_index):
	graph_center += layout_tsne[index]
graph_center = graph_center/current_index	
# get the cluster centers (consider different clustering levels)
for level in range(0, number_level):
	cluster_number = int(level_size[level])
	cluster_center = np.zeros((cluster_number, 2))
	cluster_diff = np.zeros((cluster_number, 2))
	cluster_member_count = np.zeros(cluster_number)
	for index in range(0, current_index):
		cluster_id = memberships2[level][index]
		cluster_center[cluster_id] += layout_tsne[index]
		cluster_member_count[cluster_id] += 1	
	for index in range(0, cluster_number):
		cluster_center[index] = cluster_center[index]/cluster_member_count[index]
		cluster_diff[index] = cluster_center[index] - graph_center
		
	layout_adjust = np.zeros((current_index, 2))
	for index in range(0, current_index):
		cluster_id = memberships2[level][index]
		layout_adjust[index] = layout_tsne[index] + aggregate_rate*cluster_diff[cluster_id]
	layout_tsne_adjusts.append(layout_adjust)
	
	layout_adjust2 = np.zeros((current_index, 2))
	for index in range(0, current_index):
		cluster_id = memberships2[level][index]
		layout_adjust2[index] = layout_tsne[index] + aggregate_rate2*cluster_diff[cluster_id]
	layout_tsne_adjusts2.append(layout_adjust2)
'''

# layout for the PV method, make the adjustment modification, and calculate cluster center, size, averaged vector, and silhouette coefficient
layout_tsne_text = text_2d
'''
layout_tsne_text_adjusts = []
layout_tsne_text_adjusts2 = []
# (update 2017/12/20) record cluster center, size, averaged cluster vector, and silhouette coefficient as well
layout_tsne_text_clusterCenter = []
layout_tsne_text_clusterSize = [] 
layout_tsne_text_clusterVector = []
layout_tsne_text_clusterVector_active = []
layout_tsne_text_clusterSilhouette = []
# (update 2018/01/13) record the number of highlighted (relevant) articles in each cluster, and also the detailed list of highlighted articles
layout_tsne_text_clusterHighlightSize = []
layout_tsne_text_clusterHighlight = [] # list of article index
layout_tsne_text_clusterHighlightPMID = [] # list of article PMID
# (update 2017/12/21) calculate pairwise cluster similarity	
layout_tsne_text_clusterSimi = [] # based on average cluster vector
layout_tsne_text_clusterSimi_active = [] # based on the active version of averaged cluster vector
# get the graph center
graph_center = [0,0]
for index in range(0, current_index):
	graph_center += layout_tsne_text[index]
graph_center = graph_center/current_index	
# get the cluster centers (consider different clustering levels)
for level in range(0, number_level):
	cluster_number = int(level_size[level])
	cluster_center = np.zeros((cluster_number, 2))
	cluster_diff = np.zeros((cluster_number, 2))
	cluster_member_count = np.zeros(cluster_number)
	cluster_vector = np.zeros((cluster_number, vector_dimension))
	cluster_vector_active = np.zeros((cluster_number, active_dimension))
	cluster_silhouette = np.zeros(cluster_number)
	cluster_highlight_count = np.zeros(cluster_number)
	cluster_highlight_list = [[] for _ in range(cluster_number)]
	cluster_highlight_listPMID = [[] for _ in range(cluster_number)]
	for index in range(0, current_index):
		cluster_id = memberships3[level][index]
		cluster_center[cluster_id] += layout_tsne_text[index]
		cluster_member_count[cluster_id] += 1
		if ids[index] in highlight:
			cluster_highlight_count[cluster_id] += 1
			cluster_highlight_list[cluster_id].append(index)
			cluster_highlight_listPMID[cluster_id].append(ids[index])
		cluster_vector[cluster_id] += texts_doc2vec[index]
		cluster_vector_active[cluster_id] += texts_doc2vec_active[index]
		cluster_silhouette[cluster_id] += silhouettes3[level][index]
	for index in range(0, cluster_number):
		cluster_center[index] = cluster_center[index]/cluster_member_count[index]
		cluster_diff[index] = cluster_center[index] - graph_center
		cluster_vector[index] = cluster_vector[index]/cluster_member_count[index]
		cluster_vector_active[index] = cluster_vector_active[index]/cluster_member_count[index]
		cluster_silhouette[index] = cluster_silhouette[index]/cluster_member_count[index]
	
	layout_tsne_text_clusterCenter.append(cluster_center)
	layout_tsne_text_clusterSize.append(cluster_member_count)
	layout_tsne_text_clusterVector.append(cluster_vector)
	layout_tsne_text_clusterVector_active.append(cluster_vector_active)
	layout_tsne_text_clusterSilhouette.append(cluster_silhouette)
	layout_tsne_text_clusterHighlightSize.append(cluster_highlight_count)
	layout_tsne_text_clusterHighlight.append(cluster_highlight_list)
	layout_tsne_text_clusterHighlightPMID.append(cluster_highlight_listPMID)
	
	# (update 2017/12/21) calculate pairwise cluster similarity	
	cluster_simi = np.zeros((cluster_number, cluster_number))
	cluster_simi_active = np.zeros((cluster_number, cluster_number))
	for index1 in range(0, cluster_number):
		for index2 in range(index1, cluster_number):
			if index1 == index2:
				cluster_simi[index1][index2] = 1
			else:
				cluster_simi[index1][index2] = 	1 - spatial.distance.cosine(cluster_vector[index1], cluster_vector[index2])
				cluster_simi[index2][index1] = cluster_simi[index1][index2]
				cluster_simi_active[index1][index2] = 	1 - spatial.distance.cosine(cluster_vector_active[index1], cluster_vector_active[index2])
				cluster_simi_active[index2][index1] = cluster_simi_active[index1][index2]
	layout_tsne_text_clusterSimi.append(cluster_simi)
	layout_tsne_text_clusterSimi_active.append(cluster_simi_active)			
		
	layout_adjust = np.zeros((current_index, 2))
	for index in range(0, current_index):
		cluster_id = memberships3[level][index]
		layout_adjust[index] = layout_tsne_text[index] + aggregate_rate*cluster_diff[cluster_id]
	layout_tsne_text_adjusts.append(layout_adjust)
	
	layout_adjust2 = np.zeros((current_index, 2))
	for index in range(0, current_index):
		cluster_id = memberships3[level][index]
		layout_adjust2[index] = layout_tsne_text[index] + aggregate_rate2*cluster_diff[cluster_id]
	layout_tsne_text_adjusts2.append(layout_adjust2)

	
# (Update 2017/12/19) layout for PA method, and the adjustment modification, and calculate cluster center, size, averaged vector, and silhouette coefficient
layout_tsnePA_text = text_2d_PA
layout_tsnePA_text_adjusts = []
layout_tsnePA_text_adjusts2 = []
# (update 2017/12/20) record cluster center, size, averaged cluster vector, and silhouette coefficient as well
layout_tsnePA_text_clusterCenter = []
layout_tsnePA_text_clusterSize = []
layout_tsnePA_text_clusterVector = []
layout_tsnePA_text_clusterVector_active = []
layout_tsnePA_text_clusterSilhouette = []
# (update 2018/01/13) record the number of highlighted (relevant) articles in each cluster, and also the detailed list of highlighted articles
layout_tsnePA_text_clusterHighlightSize = []
layout_tsnePA_text_clusterHighlight = [] # list of article index
layout_tsnePA_text_clusterHighlightPMID = [] # list of article PMID
# (update 2017/12/21) calculate pairwise cluster similarity	
layout_tsnePA_text_clusterSimi = [] # based on average cluster vector
layout_tsnePA_text_clusterSimi_active = [] # based on the active version of averaged cluster vector
# get the graph center
graph_center = [0,0]
for index in range(0, current_index):
	graph_center += layout_tsnePA_text[index]
graph_center = graph_center/current_index	
# get the cluster centers (consider different clustering levels)
for level in range(0, number_level):
	cluster_number = int(level_size[level])
	cluster_center = np.zeros((cluster_number, 2))
	cluster_diff = np.zeros((cluster_number, 2))
	cluster_member_count = np.zeros(cluster_number)
	cluster_vector = np.zeros((cluster_number, vector_dimension))
	cluster_vector_active = np.zeros((cluster_number, active_dimension))
	cluster_silhouette = np.zeros(cluster_number)
	cluster_highlight_count = np.zeros(cluster_number)
	cluster_highlight_list = [[] for _ in range(cluster_number)]
	cluster_highlight_listPMID = [[] for _ in range(cluster_number)]
	for index in range(0, current_index):
		cluster_id = memberships4[level][index]
		cluster_center[cluster_id] += layout_tsnePA_text[index]
		cluster_member_count[cluster_id] += 1
		if ids[index] in highlight:
			cluster_highlight_count[cluster_id] += 1
			cluster_highlight_list[cluster_id].append(index)
			cluster_highlight_listPMID[cluster_id].append(ids[index])
		cluster_vector[cluster_id] += texts_doc2vec_PA[index]
		cluster_vector_active[cluster_id] += texts_doc2vec_active_PA[index]
		cluster_silhouette[cluster_id] += silhouettes3[level][index]
	for index in range(0, cluster_number):
		cluster_center[index] = cluster_center[index]/cluster_member_count[index]
		cluster_diff[index] = cluster_center[index] - graph_center
		cluster_vector[index] = cluster_vector[index]/cluster_member_count[index]
		cluster_vector_active[index] = cluster_vector_active[index]/cluster_member_count[index]
		cluster_silhouette[index] = cluster_silhouette[index]/cluster_member_count[index]
		
	layout_tsnePA_text_clusterCenter.append(cluster_center)
	layout_tsnePA_text_clusterSize.append(cluster_member_count)
	layout_tsnePA_text_clusterVector.append(cluster_vector)
	layout_tsnePA_text_clusterVector_active.append(cluster_vector_active)	
	layout_tsnePA_text_clusterSilhouette.append(cluster_silhouette)
	layout_tsnePA_text_clusterHighlightSize.append(cluster_highlight_count)
	layout_tsnePA_text_clusterHighlight.append(cluster_highlight_list)
	layout_tsnePA_text_clusterHighlightPMID.append(cluster_highlight_listPMID)
	
	# (update 2017/12/21) calculate pairwise cluster similarity	
	cluster_simi = np.zeros((cluster_number, cluster_number))
	cluster_simi_active = np.zeros((cluster_number, cluster_number))
	for index1 in range(0, cluster_number):
		for index2 in range(index1, cluster_number):
			if index1 == index2:
				cluster_simi[index1][index2] = 1
			else:
				cluster_simi[index1][index2] = 	1 - spatial.distance.cosine(cluster_vector[index1], cluster_vector[index2])
				cluster_simi[index2][index1] = cluster_simi[index1][index2]
				cluster_simi_active[index1][index2] = 	1 - spatial.distance.cosine(cluster_vector_active[index1], cluster_vector_active[index2])
				cluster_simi_active[index2][index1] = cluster_simi_active[index1][index2]
	layout_tsnePA_text_clusterSimi.append(cluster_simi)
	layout_tsnePA_text_clusterSimi_active.append(cluster_simi_active)	
		
	layout_adjust = np.zeros((current_index, 2))
	for index in range(0, current_index):
		cluster_id = memberships4[level][index]
		layout_adjust[index] = layout_tsnePA_text[index] + aggregate_rate*cluster_diff[cluster_id]
	layout_tsnePA_text_adjusts.append(layout_adjust)
	
	layout_adjust2 = np.zeros((current_index, 2))
	for index in range(0, current_index):
		cluster_id = memberships4[level][index]
		layout_adjust2[index] = layout_tsnePA_text[index] + aggregate_rate2*cluster_diff[cluster_id]
	layout_tsnePA_text_adjusts2.append(layout_adjust2)				
'''
			
# Write the preprocessed features to a JSON file
with open(directory + 'features.json', 'w') as output:
	output.write("{")
	output.write("\"nodes\":\n")
	output.write("[\n")
	for i in range(0, current_index):
		output.write("{\n")
		output.write("\"id\": \"" + ids[i] + "\",\n")
		output.write("\"index\": " + str(i) + ",\n")
		output.write("\"type\": \"" + types[i] + "\",\n")
		output.write("\"year\": \"" + years[i] + "\",\n")
		output.write("\"if_highlight\": \"" + if_highlight[i] + "\",\n")
		output.write("\"title_all\": \"" + titles_all[i] + "\",\n")
		output.write("\"title_raw\": \"" + titles_raw[i] + "\",\n")
		output.write("\"abstract_all\": \"" + abstracts_all[i] + "\",\n")
		output.write("\"abstract_raw\": \"" + abstracts_raw[i] + "\",\n")
		output.write("\"topic_lda\": \"" + topics_lda[i] + "\",\n")
		output.write("\"topic_nmf\": \"" + topics_nmf[i] + "\",\n")
		#output.write("\"keyword_rake\": \"" + ", ".join(abstract_keywords_rake[i]) + "\",\n")
		output.write("\"keyword_rake\": \"" + ", ".join(keywords_rake[i]) + "\",\n")
		if flag_terms > 0:
			output.write("\"keyword_display\": \"" + terms[i].replace("_", " ") + "\",\n")
		elif flag_keywords > 0:
			output.write("\"keyword_display\": \"" + keywords[i].replace("_", " ") + "\",\n")
		if flag_keywords > 0:
			output.write("\"keywords\": [")
			for j in range(0, len(keywords2[i])):
				if(len(keywords2[i][j].strip()) > 0):
					output.write("\"" + keywords2[i][j].strip() + "\"")
					if j != len(keywords2[i]) - 1:
						output.write(",")
			output.write("],\n")
		if flag_terms > 0:	
			output.write("\"terms\": [")
			for j in range(0, len(terms2[i])):
				if(len(terms2[i][j].strip()) > 0):
					output.write("\"" + terms2[i][j].strip() + "\"")
					if j != len(terms2[i]) - 1:
						output.write(",")
			output.write("],\n")
		output.write("\"authors\": [")
		for j in range(0, len(authors2[i])):
			if(len(authors2[i][j].strip()) > 0):
				output.write("\"" + authors2[i][j].strip() + "\"")
				if j != len(authors2[i]) - 1:
					output.write(",")
		output.write("],\n")
		output.write("\"pubtypes\": [")
		for j in range(0, len(pubtypes2[i])):
			if(len(pubtypes2[i][j].strip()) > 0):
				output.write("\"" + pubtypes2[i][j].strip() + "\"")
				if j != len(pubtypes2[i]) - 1:
					output.write(",")
		output.write("],\n")
		output.write("\"references\": [")
		for j in range(0, len(references2[i])):
			output.write(str(references2[i][j]))
			if j != len(references2[i]) - 1:
				output.write(",")
		output.write("],\n")
		output.write("\"citations\": [")
		for j in range(0, len(citations2[i])):
			output.write(str(citations2[i][j]))
			if j != len(citations2[i]) - 1:
				output.write(",")
		output.write("],\n")
		output.write("\"number_of_citations\": " + str(len(citations2[i])) + ",\n")
		'''
		number_edge = np.count_nonzero(combine_simi_edge[i])
		output.write("\"edges\": [")
		add_edge = 0
		for j in range(0, current_index):
			if combine_simi_edge[i][j] > 0:
				output.write(str(j))
				add_edge += 1
				if add_edge < number_edge:
					output.write(",")
		output.write("],\n")
		'''	
		'''
		output.write("\"weights\": [")
		add_weight = 0
		for j in range(0, current_index):
			if combine_simi_edge[i][j] > 0:
				output.write(str(combine_simi_edge[i][j]))
				add_weight += 1
				if add_weight < number_edge:
					output.write(",")
		output.write("],\n")
		'''
		'''
		#output.write("\"title_x\": \"" + str(titles_2d[i][0]) + "\", ")
		#output.write("\"title_y\": \"" + str(titles_2d[i][1]) + "\", ")
		#output.write("\"abstract_x\": \"" + str(abstracts_2d[i][0]) + "\", ")
		#output.write("\"abstract_y\": \"" + str(abstracts_2d[i][1]) + "\", ")
		#output.write("\"keyword_x\": \"" + str(keywords_2d[i][0] + random.uniform(-offset_x/100, offset_x/100)) + "\", ")
		#output.write("\"keyword_y\": \"" + str(keywords_2d[i][1] + random.uniform(-offset_y/100, offset_y/100)) + "\", ")
		#output.write("\"term_x\": \"" + str(terms_2d[i][0] + random.uniform(-offset_x/100, offset_x/100)) + "\", ")
		#output.write("\"term_y\": \"" + str(terms_2d[i][1] + random.uniform(-offset_y/100, offset_y/100)) + "\", ")
		#output.write("\"author_x\": \"" + str(authors_2d[i][0]) + "\", ")
		#output.write("\"author_y\": \"" + str(authors_2d[i][1]) + "\", ")
		#output.write("\"pubtype_x\": \"" + str(pubtypes_2d[i][0]) + "\", ")
		#output.write("\"pubtype_y\": \"" + str(pubtypes_2d[i][1]) + "\", ")
		output.write("\"combine_x\": \"" + str(combine_2d[i][0]) + "\", ")
		output.write("\"combine_y\": \"" + str(combine_2d[i][1]) + "\", ")
		output.write("\"layout_tsne_x\": \"" + str(layout_tsne[i][0]) + "\", ") # combination of all elements using SVD and t-SNE
		output.write("\"layout_tsne_y\": \"" + str(layout_tsne[i][1]) + "\", ")
		'''
		output.write("\"layout_tsne_text_x\": \"" + str(layout_tsne_text[i][0]) + "\", ") # title and abstract with doc2vec and t-SNE
		output.write("\"layout_tsne_text_y\": \"" + str(layout_tsne_text[i][1]) + "\", \n")
		'''
		output.write("\"layout_tsne_title_x\": \"" + str(title_2d[i][0]) + "\", ") # title alone with doc2vec and t-SNE
		output.write("\"layout_tsne_title_y\": \"" + str(title_2d[i][1]) + "\", ")
		output.write("\"layout_tsne_abstract_x\": \"" + str(abstract_2d[i][0]) + "\", ") # abstract alone with doc2vec andt-SNE
		output.write("\"layout_tsne_abstract_y\": \"" + str(abstract_2d[i][1]) + "\", ")
		output.write("\"layout_tsnePA_text_x\": \"" + str(layout_tsnePA_text[i][0]) + "\", ") # abstract alone with doc2vec andt-SNE
		output.write("\"layout_tsnePA_text_y\": \"" + str(layout_tsnePA_text[i][1]) + "\", ")
		output.write("\"layout_fr_x\": \"" + str(layout_fr[i][0]) + "\", ")
		output.write("\"layout_fr_y\": \"" + str(layout_fr[i][1]) + "\", ")
		output.write("\"layout_kk_x\": \"" + str(layout_kk[i][0]) + "\", ")
		output.write("\"layout_kk_y\": \"" + str(layout_kk[i][1]) + "\", ")
		output.write("\"layout_lgl_x\": \"" + str(layout_lgl[i][0]) + "\", ")
		output.write("\"layout_lgl_y\": \"" + str(layout_lgl[i][1]) + "\", ")
		output.write("\"layout_fa_x\": \"" + str(layout_fa[i][0]) + "\", ")
		output.write("\"layout_fa_y\": \"" + str(layout_fa[i][1]) + "\",\n")
		for level in range(0, number_level):
			output.write("\"layout_fa_adjust_level" + str(level) + "_x\": " + str(layout_fa_adjusts[level][i][0]) + ", ")
			output.write("\"layout_fa_adjust_level" + str(level) + "_y\": " + str(layout_fa_adjusts[level][i][1]) + ", ")
			output.write("\"layout_fa_adjust2_level" + str(level) + "_x\": " + str(layout_fa_adjusts2[level][i][0]) + ", ")
			output.write("\"layout_fa_adjust2_level" + str(level) + "_y\": " + str(layout_fa_adjusts2[level][i][1]) + ",\n")
		for level in range(0, number_level):
			output.write("\"layout_fr_adjust_level" + str(level) + "_x\": " + str(layout_fr_adjusts[level][i][0]) + ", ")
			output.write("\"layout_fr_adjust_level" + str(level) + "_y\": " + str(layout_fr_adjusts[level][i][1]) + ", ")
			output.write("\"layout_fr_adjust2_level" + str(level) + "_x\": " + str(layout_fr_adjusts2[level][i][0]) + ", ")
			output.write("\"layout_fr_adjust2_level" + str(level) + "_y\": " + str(layout_fr_adjusts2[level][i][1]) + ",\n")
		for level in range(0, number_level):
			output.write("\"layout_tsne_adjust_level" + str(level) + "_x\": " + str(layout_tsne_adjusts[level][i][0]) + ", ")
			output.write("\"layout_tsne_adjust_level" + str(level) + "_y\": " + str(layout_tsne_adjusts[level][i][1]) + ", ")
			output.write("\"layout_tsne_adjust2_level" + str(level) + "_x\": " + str(layout_tsne_adjusts2[level][i][0]) + ", ")
			output.write("\"layout_tsne_adjust2_level" + str(level) + "_y\": " + str(layout_tsne_adjusts2[level][i][1]) + ",\n")
		for level in range(0, number_level):
			output.write("\"layout_tsne_text_adjust_level" + str(level) + "_x\": " + str(layout_tsne_text_adjusts[level][i][0]) + ", ")
			output.write("\"layout_tsne_text_adjust_level" + str(level) + "_y\": " + str(layout_tsne_text_adjusts[level][i][1]) + ", ")
			output.write("\"layout_tsne_text_adjust2_level" + str(level) + "_x\": " + str(layout_tsne_text_adjusts2[level][i][0]) + ", ")
			output.write("\"layout_tsne_text_adjust2_level" + str(level) + "_y\": " + str(layout_tsne_text_adjusts2[level][i][1]) + ",\n")
		for level in range(0, number_level):
			output.write("\"layout_tsnePA_text_adjust_level" + str(level) + "_x\": " + str(layout_tsnePA_text_adjusts[level][i][0]) + ", ")
			output.write("\"layout_tsnePA_text_adjust_level" + str(level) + "_y\": " + str(layout_tsnePA_text_adjusts[level][i][1]) + ", ")
			output.write("\"layout_tsnePA_text_adjust2_level" + str(level) + "_x\": " + str(layout_tsnePA_text_adjusts2[level][i][0]) + ", ")
			output.write("\"layout_tsnePA_text_adjust2_level" + str(level) + "_y\": " + str(layout_tsnePA_text_adjusts2[level][i][1]) + ",\n")
		'''
		
		#(Update 2017/12/20) output semantic vectors learned with PV and PA respectively
		output.write("\"vector_PV\": [")
		for dim in range(vector_dimension):
			output.write(str(texts_doc2vec[i][dim]))
			if dim != vector_dimension - 1:
				output.write(",")
		output.write("],\n")
		output.write("\"vector_PV_active\": [")
		for dim in range(active_dimension):
			output.write(str(texts_doc2vec_active[i][dim]))
			if dim != active_dimension - 1:
				output.write(",")
		output.write("],\n")	
		output.write("\"vector_PV_normArticle\": [")
		for dim in range(vector_dimension):
			output.write(str(texts_doc2vec_normArticle[i][dim]))
			if dim != vector_dimension - 1:
				output.write(",")
		output.write("],\n")	
		output.write("\"vector_PV_normDim\": [")
		for dim in range(vector_dimension):
			output.write(str(texts_doc2vec_normDim[i][dim]))
			if dim != vector_dimension - 1:
				output.write(",")
		output.write("]\n")
		output.write("}\n")
		'''
		output.write("\"vector_PA\": [")
		for dim in range(vector_dimension):
			output.write(str(texts_doc2vec_PA[i][dim]))
			if dim != vector_dimension - 1:
				output.write(",")
		output.write("],\n")
		output.write("\"vector_PA_active\": [")
		for dim in range(active_dimension):
			output.write(str(texts_doc2vec_active_PA[i][dim]))
			if dim != active_dimension - 1:
				output.write(",")
		output.write("],\n")
		output.write("\"vector_PA_normArticle\": [")
		for dim in range(vector_dimension):
			output.write(str(texts_doc2vec_PA_normArticle[i][dim]))
			if dim != vector_dimension - 1:
				output.write(",")
		output.write("],\n")
		output.write("\"vector_PA_normDim\": [")
		for dim in range(vector_dimension):
			output.write(str(texts_doc2vec_PA_normDim[i][dim]))
			if dim != vector_dimension - 1:
				output.write(",")
		output.write("],\n")
		'''	
		'''
		#output.write("\"cluster\": \"" + str(clustering.labels_[i]) + "\"\n}")
		output.write("\"cluster\": " + str(membership[i]) + ", ")
		#output.write("\"cluster_gephi\": \"" + str(membership_gephi[i]) + "\", ")
		output.write("\"clusters\": [")
		for level in range(0, number_level):
			output.write(str(memberships[level][i]))
			if level != number_level - 1:
				output.write(",")
		output.write("],")
		output.write("\"clusters2\": [")
		for level in range(0, number_level):
			output.write(str(memberships2[level][i]))
			if level != number_level - 1:
				output.write(",")
		output.write("],")
		output.write("\"clusters3\": [")
		for level in range(0, number_level):
			output.write(str(memberships3[level][i]))
			if level != number_level - 1:
				output.write(",")
		output.write("],")
		output.write("\"clusters4\": [")
		for level in range(0, number_level):
			output.write(str(memberships4[level][i]))
			if level != number_level - 1:
				output.write(",")
		output.write("]\n}")
		'''	
		if i != current_index - 1:
			output.write(",")
		output.write("\n")
	output.write("]\n")
	
	'''
	output.write("\"links\":\n")
	output.write("[\n")
	total_number_edge = np.count_nonzero(combine_simi_edge.flatten())
	add_edge = 0
	for index1 in range(0, current_index):
		for index2 in range(0, current_index):
			if combine_simi_edge[index1][index2] > 0:
				output.write("{\"source\":\"" + str(ids[index1]) + "\", \"target\":\"" + str(ids[index2]) + "\", \"value\":\"" + str(combine_simi_edge[index1][index2]) + "\", \"combine_x\":\"" + str(combine_2d[index1][0]) + "\", \"combine_y\":\"" + str(combine_2d[index1][1]) + "\"}")
				add_edge += 1
				if(add_edge < total_number_edge):
					output.write(",")
				output.write("\n")
	output.write("],\n")
	'''
	'''
	output.write("\"clusters\":\n")
	output.write("{\n")
	output.write("\"number_highlight\": " + str(len(highlight)) + ",\n")
	output.write("\"number_level\": " + str(number_level) + ",\n")
	output.write("\"level_size\": [")
	for level in range(0, number_level):
		output.write(str(int(level_size[level])))
		if level != number_level - 1:
			output.write(",")
	output.write("],\n")
	output.write("\"community_map\": [")
	for index in range(0, len(community_map)):
		output.write("{")
		for level in range(0, number_level):
			output.write("\"level" + str(level) + "\":" + str(int(community_map[index][level])))
			if level != number_level - 1:
				output.write(",")
		output.write("}")
		if index != len(community_map) - 1:
			output.write(", ")
	output.write("],\n")
	for level in range(0, number_level):
		output.write("\"community_map_level" + str(level) + "\": [")
		for index in range(0, len(community_maps[level])):
			output.write(str(int(community_maps[level][index][1])))
			if index != len(community_maps[level]) - 1:
				output.write(",")
		output.write("],\n")
	for level in range(0, number_level):
		output.write("\"community_subid_level" + str(level) + "\": [")
		for index in range(0, len(community_maps[level])):
			output.write(str(int(community_maps[level][index][2])))
			if index != len(community_maps[level]) - 1:
				output.write(",")
		output.write("],\n")
	output.write("\"cluster_title_keywords\": [")	
	for level in range(0, number_level):
		output.write("[")	
		for cluster in range(len(community_maps[level])):
			output.write('\"%s\"' % ', '.join(map(str, cluster_title_keywords[level][cluster])))
			if cluster < len(community_maps[level]) - 1:
				output.write(",")
			output.write("\n")
		if level < number_level - 1:
			output.write("], \n")
		else:
			output.write("]\n")	
	output.write("], \n")		
	output.write("\"cluster_abstract_keywords\": [")	
	for level in range(0, number_level):
		output.write("[")
		for cluster in range(len(community_maps[level])):
			output.write('\"%s\"' % ', '.join(map(str, cluster_abstract_keywords[level][cluster])))
			if cluster < len(community_maps[level]) - 1:
				output.write(",")
			output.write("\n")
		if level < number_level - 1:
			output.write("], \n")
		else:
			output.write("]\n")	
	output.write("], \n")			
			
	output.write("\"community_map2\": [")
	for index in range(0, len(community_map2)):
		output.write("{")
		for level in range(0, number_level):
			output.write("\"level" + str(level) + "\":" + str(int(community_map2[index][level])))
			if level != number_level - 1:
				output.write(",")
		output.write("}")
		if index != len(community_map2) - 1:
			output.write(", ")
	output.write("],\n")
	for level in range(0, number_level):
		output.write("\"community_map2_level" + str(level) + "\": [")
		for index in range(0, len(community_maps2[level])):
			output.write(str(int(community_maps2[level][index][1])))
			if index != len(community_maps2[level]) - 1:
				output.write(",")
		output.write("],\n")
	for level in range(0, number_level):
		output.write("\"community_subid2_level" + str(level) + "\": [")
		for index in range(0, len(community_maps2[level])):
			output.write(str(int(community_maps2[level][index][2])))
			if index != len(community_maps2[level]) - 1:
				output.write(",")
		output.write("],\n")
	output.write("\"cluster_title_keywords2\": [")	
	for level in range(0, number_level):
		output.write("[")	
		for cluster in range(len(community_maps2[level])):
			output.write('\"%s\"' % ', '.join(map(str, cluster_title_keywords2[level][cluster])))
			if cluster < len(community_maps2[level]) - 1:
				output.write(",")
			output.write("\n")
		if level < number_level - 1:
			output.write("], \n")
		else:
			output.write("]\n")	
	output.write("], \n")		
	output.write("\"cluster_abstract_keywords2\": [")	
	for level in range(0, number_level):
		output.write("[")
		for cluster in range(len(community_maps2[level])):
			output.write('\"%s\"' % ', '.join(map(str, cluster_abstract_keywords2[level][cluster])))
			if cluster < len(community_maps2[level]) - 1:
				output.write(",")
			output.write("\n")
		if level < number_level - 1:
			output.write("], \n")
		else:
			output.write("]\n")	
	output.write("], \n")
	
	output.write("\"community_map3\": [")
	for index in range(0, len(community_map3)):
		output.write("{")
		for level in range(0, number_level):
			output.write("\"level" + str(level) + "\":" + str(int(community_map3[index][level])))
			if level != number_level - 1:
				output.write(",")
		output.write("}")
		if index != len(community_map3) - 1:
			output.write(", ")
	output.write("],\n")
	for level in range(0, number_level):
		output.write("\"community_map3_level" + str(level) + "\": [")
		for index in range(0, len(community_maps3[level])):
			output.write(str(int(community_maps3[level][index][1])))
			if index != len(community_maps3[level]) - 1:
				output.write(",")
		output.write("],\n")
	for level in range(0, number_level):
		output.write("\"community_subid3_level" + str(level) + "\": [")
		for index in range(0, len(community_maps3[level])):
			output.write(str(int(community_maps3[level][index][2])))
			if index != len(community_maps3[level]) - 1:
				output.write(",")
		output.write("],\n")
	output.write("\"cluster_title_keywords3\": [")	
	for level in range(0, number_level):
		output.write("[")	
		for cluster in range(len(community_maps3[level])):
			output.write('\"%s\"' % ', '.join(map(str, cluster_title_keywords3[level][cluster])))
			if cluster < len(community_maps3[level]) - 1:
				output.write(",")
			output.write("\n")
		if level < number_level - 1:
			output.write("], \n")
		else:
			output.write("]\n")	
	output.write("], \n")		
	output.write("\"cluster_abstract_keywords3\": [")	
	for level in range(0, number_level):
		output.write("[")
		for cluster in range(len(community_maps3[level])):
			output.write('\"%s\"' % ', '.join(map(str, cluster_abstract_keywords3[level][cluster])))
			if cluster < len(community_maps3[level]) - 1:
				output.write(",")
			output.write("\n")
		if level < number_level - 1:
			output.write("], \n")
		else:
			output.write("]\n")	
	output.write("], \n")				

	output.write("\"community_map4\": [")
	for index in range(0, len(community_map4)):
		output.write("{")
		for level in range(0, number_level):
			output.write("\"level" + str(level) + "\":" + str(int(community_map4[index][level])))
			if level != number_level - 1:
				output.write(",")
		output.write("}")
		if index != len(community_map4) - 1:
			output.write(", ")
	output.write("],\n")
	for level in range(0, number_level):
		output.write("\"community_map4_level" + str(level) + "\": [")
		for index in range(0, len(community_maps4[level])):
			output.write(str(int(community_maps4[level][index][1])))
			if index != len(community_maps4[level]) - 1:
				output.write(",")
		output.write("],\n")
	for level in range(0, number_level):
		output.write("\"community_subid4_level" + str(level) + "\": [")
		for index in range(0, len(community_maps4[level])):
			output.write(str(int(community_maps4[level][index][2])))
			if index != len(community_maps4[level]) - 1:
				output.write(",")
		output.write("], \n")
	output.write("\"cluster_title_keywords4\": [")	
	for level in range(0, number_level):
		output.write("[")	
		for cluster in range(len(community_maps4[level])):
			output.write('\"%s\"' % ', '.join(map(str, cluster_title_keywords4[level][cluster])))
			if cluster < len(community_maps4[level]) - 1:
				output.write(",")
			output.write("\n")
		if level < number_level - 1:
			output.write("], \n")
		else:
			output.write("]\n")	
	output.write("], \n")		
	output.write("\"cluster_abstract_keywords4\": [")	
	for level in range(0, number_level):
		output.write("[")
		for cluster in range(len(community_maps4[level])):
			output.write('\"%s\"' % ', '.join(map(str, cluster_abstract_keywords4[level][cluster])))
			if cluster < len(community_maps4[level]) - 1:
				output.write(",")
			output.write("\n")
		if level < number_level - 1:
			output.write("], \n")
		else:
			output.write("]\n")	
	output.write("], \n")
	
	# (Update 2017/12/20) output cluster centers and cluster size
	output.write("\"cluster_center3\": [")	
	for level in range(0, number_level):
		output.write("[")
		for index in range(len(layout_tsne_text_clusterCenter[level])):
			output.write("[" + str(layout_tsne_text_clusterCenter[level][index][0]) + "," + str(layout_tsne_text_clusterCenter[level][index][1]) + "]")
			if index < len(layout_tsne_text_clusterCenter[level]) - 1:
				output.write(",")
		output.write("]")		
		if level < number_level - 1:
			output.write(",\n")
	output.write("], \n")		
	output.write("\"cluster_size3\": [")	
	for level in range(0, number_level):
		output.write("[")
		for index in range(len(layout_tsne_text_clusterSize[level])):
			output.write(str(layout_tsne_text_clusterSize[level][index]))
			if index < len(layout_tsne_text_clusterSize[level]) - 1:
				output.write(",")
		output.write("]")		
		if level < number_level - 1:
			output.write(",\n")
	output.write("], \n")
	output.write("\"cluster_silhouette3\": [")	
	for level in range(0, number_level):
		output.write("[")
		for index in range(len(layout_tsne_text_clusterSilhouette[level])):
			output.write(str(layout_tsne_text_clusterSilhouette[level][index]))
			if index < len(layout_tsne_text_clusterSilhouette[level]) - 1:
				output.write(",")
		output.write("]")		
		if level < number_level - 1:
			output.write(",\n")
	output.write("], \n")
	output.write("\"cluster_highlight_size3\": [")	
	for level in range(0, number_level):
		output.write("[")
		for index in range(len(layout_tsne_text_clusterHighlightSize[level])):
			output.write(str(layout_tsne_text_clusterHighlightSize[level][index]))
			if index < len(layout_tsne_text_clusterHighlightSize[level]) - 1:
				output.write(",")
		output.write("]")		
		if level < number_level - 1:
			output.write(",\n")
	output.write("], \n")
	output.write("\"cluster_vector3\": [")	
	for level in range(0, number_level):
		output.write("[")
		for index in range(len(layout_tsne_text_clusterVector[level])):
			#output.write(str(layout_tsnePA_text_clusterVector[level][index]))
			output.write("[")
			for dim in range(len(layout_tsne_text_clusterVector[level][index])):
				output.write(str(layout_tsne_text_clusterVector[level][index][dim]))
				if dim < len(layout_tsne_text_clusterVector[level][index]) - 1:
					output.write(",")
			output.write("]")		
			if index < len(layout_tsne_text_clusterVector[level]) - 1:
				output.write(",")
		output.write("]")		
		if level < number_level - 1:
			output.write(",\n")
	output.write("], \n")
	output.write("\"cluster_vector_active3\": [")	
	for level in range(0, number_level):
		output.write("[")
		for index in range(len(layout_tsne_text_clusterVector_active[level])):
			output.write("[")
			for dim in range(len(layout_tsne_text_clusterVector_active[level][index])):
				output.write(str(layout_tsne_text_clusterVector_active[level][index][dim]))
				if dim < len(layout_tsne_text_clusterVector_active[level][index]) - 1:
					output.write(",")
			output.write("]")		
			if index < len(layout_tsne_text_clusterVector_active[level]) - 1:
				output.write(",")
		output.write("]")		
		if level < number_level - 1:
			output.write(",\n")
	output.write("], \n")
	output.write("\"cluster_simi3\": [")	
	for level in range(0, number_level):
		output.write("[")
		for index1 in range(len(layout_tsne_text_clusterSimi[level])):
			output.write("[")
			for index2 in range(len(layout_tsne_text_clusterSimi[level][index1])):
				output.write(str(layout_tsne_text_clusterSimi[level][index1][index2]))
				if index2 < len(layout_tsne_text_clusterSimi[level][index1]) - 1:
					output.write(",")	
			if index1 < len(layout_tsne_text_clusterSimi[level]) - 1:
				output.write("],\n")
			else:
				output.write("]\n")		
		if level < number_level - 1:
			output.write("],\n")
		else:
			output.write("]\n")	
	output.write("], \n")
	
	output.write("\"cluster_center4\": [")	
	for level in range(0, number_level):
		output.write("[")
		for index in range(len(layout_tsnePA_text_clusterCenter[level])):
			output.write("[" + str(layout_tsnePA_text_clusterCenter[level][index][0]) + "," + str(layout_tsnePA_text_clusterCenter[level][index][1]) + "]")
			if index < len(layout_tsnePA_text_clusterCenter[level]) - 1:
				output.write(",")
		output.write("]")		
		if level < number_level - 1:
			output.write(",\n")
	output.write("], \n")		
	output.write("\"cluster_size4\": [")	
	for level in range(0, number_level):
		output.write("[")
		for index in range(len(layout_tsnePA_text_clusterSize[level])):
			output.write(str(layout_tsnePA_text_clusterSize[level][index]))
			if index < len(layout_tsnePA_text_clusterSize[level]) - 1:
				output.write(",")
		output.write("]")		
		if level < number_level - 1:
			output.write(",\n")
	output.write("], \n")
	output.write("\"cluster_highlight_size4\": [")	
	for level in range(0, number_level):
		output.write("[")
		for index in range(len(layout_tsnePA_text_clusterHighlightSize[level])):
			output.write(str(layout_tsnePA_text_clusterHighlightSize[level][index]))
			if index < len(layout_tsnePA_text_clusterHighlightSize[level]) - 1:
				output.write(",")
		output.write("]")		
		if level < number_level - 1:
			output.write(",\n")
	output.write("], \n")
	output.write("\"cluster_silhouette4\": [")	
	for level in range(0, number_level):
		output.write("[")
		for index in range(len(layout_tsnePA_text_clusterSilhouette[level])):
			output.write(str(layout_tsnePA_text_clusterSilhouette[level][index]))
			if index < len(layout_tsnePA_text_clusterSilhouette[level]) - 1:
				output.write(",")
		output.write("]")		
		if level < number_level - 1:
			output.write(",\n")
	output.write("], \n")
	output.write("\"cluster_vector4\": [")	
	for level in range(0, number_level):
		output.write("[")
		for index in range(len(layout_tsnePA_text_clusterVector[level])):
			#output.write(str(layout_tsnePA_text_clusterVector[level][index]))
			output.write("[")
			for dim in range(len(layout_tsnePA_text_clusterVector[level][index])):
				output.write(str(layout_tsnePA_text_clusterVector[level][index][dim]))
				if dim < len(layout_tsnePA_text_clusterVector[level][index]) - 1:
					output.write(",")
			output.write("]")		
			if index < len(layout_tsnePA_text_clusterVector[level]) - 1:
				output.write(",")
		output.write("]")		
		if level < number_level - 1:
			output.write(",\n")
	output.write("], \n")
	output.write("\"cluster_vector_active4\": [")	
	for level in range(0, number_level):
		output.write("[")
		for index in range(len(layout_tsnePA_text_clusterVector_active[level])):
			output.write("[")
			for dim in range(len(layout_tsnePA_text_clusterVector_active[level][index])):
				output.write(str(layout_tsnePA_text_clusterVector_active[level][index][dim]))
				if dim < len(layout_tsnePA_text_clusterVector_active[level][index]) - 1:
					output.write(",")
			output.write("]")		
			if index < len(layout_tsnePA_text_clusterVector_active[level]) - 1:
				output.write(",")
		output.write("]")		
		if level < number_level - 1:
			output.write(",\n")
	output.write("], \n")
	output.write("\"cluster_simi4\": [")	
	for level in range(0, number_level):
		output.write("[")
		for index1 in range(len(layout_tsnePA_text_clusterSimi[level])):
			output.write("[")
			for index2 in range(len(layout_tsnePA_text_clusterSimi[level][index1])):
				output.write(str(layout_tsnePA_text_clusterSimi[level][index1][index2]))
				if index2 < len(layout_tsnePA_text_clusterSimi[level][index1]) - 1:
					output.write(",")	
			if index1 < len(layout_tsnePA_text_clusterSimi[level]) - 1:
				output.write("],\n")
			else:
				output.write("]\n")		
		if level < number_level - 1:
			output.write("],\n")
		else:
			output.write("]\n")	
	output.write("] \n")
	'''
		
	#output.write("}\n")
	output.write("}")
output.close()

# output all semantic vectors into a csv file, with ArticleID as columns and Dimensions as rows
with open(directory + 'PV_forArticle.csv', 'w') as output:
	output.write("DimensionID" + ",")
	for index in range(current_index):
		output.write("Article" + str(index) + ",")
	output.write("\n")	
	for dim in range(vector_dimension):
		output.write("Dimension" + str(dim) + ",")
		for index in range(current_index):
			output.write(str(texts_doc2vec[index][dim]) + ",")
		output.write("\n")	
output.close()

with open(directory + 'PV_normArticle_forArticle.csv', 'w') as output:
	output.write("DimensionID" + ",")
	for index in range(current_index):
		output.write("Article" + str(index) + ",")
	output.write("\n")	
	for dim in range(vector_dimension):
		output.write("Dimension" + str(dim) + ",")
		for index in range(current_index):
			output.write(str(texts_doc2vec_normArticle[index][dim]) + ",")
		output.write("\n")	
output.close()

with open(directory + 'PV_normDim_forArticle.csv', 'w') as output:
	output.write("DimensionID" + ",")
	for index in range(current_index):
		output.write("Article" + str(index) + ",")
	output.write("\n")	
	for dim in range(vector_dimension):
		output.write("Dimension" + str(dim) + ",")
		for index in range(current_index):
			output.write(str(texts_doc2vec_normDim[index][dim]) + ",")
		output.write("\n")	
output.close()
	
# output all semantic vectors into a csv file, with Dimensions as columns and ArticleIDs as rows
with open(directory + 'PV_forDim.csv', 'w') as output:
	output.write("ArticleID" + ",")
	for dim in range(vector_dimension):
		output.write("Dimension" + str(dim) + ",")
	output.write("\n")
	for index in range(current_index):
		output.write("Article" + str(index) + ",")
		for dim in range(vector_dimension):
			output.write(str(texts_doc2vec[index][dim]) + ",")
		output.write("\n")
output.close()

with open(directory + 'PV_normArticle_forDim.csv', 'w') as output:
	output.write("ArticleID" + ",")
	for dim in range(vector_dimension):
		output.write("Dimension" + str(dim) + ",")
	output.write("\n")
	for index in range(current_index):
		output.write("Article" + str(index) + ",")
		for dim in range(vector_dimension):
			output.write(str(texts_doc2vec_normArticle[index][dim]) + ",")
		output.write("\n")
output.close()

with open(directory + 'PV_normDim_forDim.csv', 'w') as output:
	output.write("ArticleID" + ",")
	for dim in range(vector_dimension):
		output.write("Dimension" + str(dim) + ",")
	output.write("\n")
	for index in range(current_index):
		output.write("Article" + str(index) + ",")
		for dim in range(vector_dimension):
			output.write(str(texts_doc2vec_normDim[index][dim]) + ",")
		output.write("\n")
output.close()
	