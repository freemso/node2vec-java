# node2vec-java

This repository is an implementation of *node2vec* using Java.

The *node2vec* algorithm learns continuous representations for nodes in any (un)directed, (un)weighted graph. Please check the [project page](https://snap.stanford.edu/node2vec/) for more details.

>[node2vec: Scalable Feature Learning for Networks](http://arxiv.org/abs/1607.00653). A. Grover, J. Leskovec. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2016.

The original oficial Python implementation can be found in [this repository](https://github.com/aditya-grover/node2vec).

### Basic Usage

#### Example
To run *node2vec* on Zachary's karate club network, execute the following command from the project home directory:<br/>
	``java -jar node2vec.jar --input graph/karate.edgelist --output emb/karate.emd``

#### Options
You can check out the other options available to use with *node2vec* using:<br/>
	``java -jar node2vec.jar --help``

#### Input
The supported input format is an edgelist:

	node1_id_int node2_id_int <weight_float, optional>

The graph is assumed to be undirected and unweighted by default. These options can be changed by setting the appropriate flags.

#### Output
The output file has *n+1* lines for a graph with *n* vertices.
The first line has the following format:

	num_of_nodes dim_of_representation

The next *n* lines are as follows:

	node_id dim1 dim2 ... dimd

where dim1, ... , dimd is the *d*-dimensional representation learned by *node2vec*.

### Library
- [argparse4j](https://github.com/tatsuhiro-t/argparse4j) by [tatsuhiro-t](https://github.com/tatsuhiro-t)
- [AliasMethod.java](http://www.keithschwarz.com/interesting/code/?dir=alias-method) by Keith Schwarz(htiek@cs.stanford.edu).
- [Word2VEC_java](https://github.com/NLPchina/Word2VEC_java) by [NLPchina](https://github.com/NLPchina)
