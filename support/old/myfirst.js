var http = require('http');

var clusterMaker = require('clusters');

//number of clusters, defaults to undefined
clusterMaker.k(3);

//number of iterations (higher number gives more time to converge), defaults to 1000
clusterMaker.iterations(750);

//data from which to identify clusters, defaults to []
clusterMaker.data([[1, 0], [0, 1], [0, 0], [-10, 10], [-9, 11], [10, 10], [11, 12]]);

console.log(clusterMaker.clusters());

http.createServer(function (req, res) {
    res.writeHead(200, {'Content-Type': 'text/html'});
    res.end('Hello World!');
}).listen(8080);