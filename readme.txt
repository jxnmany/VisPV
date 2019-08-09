(1) feature_extractor_XML.py
Reads a document corpus (in PubMed XML format, see an example in DERP/ADHD/data.xml), implements text parsing/processing and generates structured features (including neural document embedding), and has a JSON file as the main output (see an example in DERP/ADHD/features.json)
Note: I have commented-off many components which are not used in the current workflow. 

(2) vis.html (with JavaScript)
Generates visualizations and enables interactions to explore a document corpus and the corresponding neural embedding - as provided in a JSON file (see an example in DERP/ADHD/features.json)
Note: the text introduction/description about this VAST system (as displayed below the visualization), is not necessarily up-to-date.

(3) support
.css and other supporting .js files

(4) DERP/ADHD
a sample dataset about ADHD treatment efficacy. The document corpus (data.XML) contains 851 biomedical articles (abstract-level information) and was downloaded from PubMed. 

(5) img
images being displayed, not necessarily up-to-date.

--Xiaonan Ji (Feb 16, 2019)