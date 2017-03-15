package node2vec;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

/**
 * Created by freemso on 17-3-14.
 */
public class Graph {

    private static final double DEFAULT_WEIGHT = 1;

    private Set<Node> nodeSet = new HashSet<>();
    private Set<Edge> edgeSet = new HashSet<>();

    private boolean directed;

    private double p, q;

    private Map<Graph.Node, AliasMethod> aliasNodes = new HashMap<>();
    private Map<Graph.Edge, AliasMethod> aliasEdges = new HashMap<>();

    public Graph(String file, boolean directed, double p, double q) throws IOException {
        this.directed = directed;
        this.p = p;
        this.q = q;

        loadGraphFrom(file);
        preprocess();
    }

    /**
     * load graph data from file
     * input format: node1_id_int node2_id_int <weight_float, optional>
     * @param file path of the input file
     * @throws IOException file not found or file format not fit
     */
    private void loadGraphFrom(String file) throws IOException {
        // read graph info from file
        FileReader fr = new FileReader(file);
        BufferedReader br = new BufferedReader(fr);
        String lineTxt;
        while ((lineTxt = br.readLine()) != null) {
            // parse the line text to get the edge info
            String[] strList = lineTxt.split(" ");
            int node1ID = Integer.parseInt(strList[0]);
            int node2ID = Integer.parseInt(strList[1]);
            // add the nodes to the graph
            Node node1 = this.addNode(node1ID);
            Node node2 = this.addNode(node2ID);
            // add the edge to the graph
            if (strList.length > 2) {
                double weight = Double.parseDouble((strList[2]));
                this.addEdge(node1, node2, weight);
            } else {
                this.addEdge(node1, node2, DEFAULT_WEIGHT);
            }
        }
    }

    /**
     * pre-processing of transition probabilities for guiding the random walks
     */
    private void preprocess() {
        for (Node node : nodeSet) {
            List<Node> neighbors = this.sortedNeighborList(node);
            List<Double> probs = new ArrayList<>();
            double weightSum = 0;
            for (Node neighbor : neighbors) {
                // assert has an edge
                double weight = this.getEdge(node, neighbor).weight;
                probs.add(weight);
                weightSum += weight;
            }
            double norm = weightSum;
            probs.forEach(aDouble -> aDouble /= norm);
            aliasNodes.put(node, new AliasMethod(probs));
        }
        for (Edge edge :
                edgeSet) {
            aliasEdges.put(edge, this.computeAliasEdge(edge));
        }
    }

    /**
     * to compute the alias method for an edge
     * @param edge the edge to compute
     * @return the node2vec.AliasMethod object that store distribution information
     */
    private AliasMethod computeAliasEdge(Edge edge) {
        List<Node> neighbors = this.sortedNeighborList(edge.dst);
        List<Double> probs = new ArrayList<>();
        double weightSum = 0;
        for (Node neighbor :
                neighbors) {
            double weight;
            if (neighbor == edge.src)
                weight = edge.weight / this.p;
            else if (this.hasEdge(neighbor, edge.src))
                weight = edge.weight;
            else weight = edge.weight / this.q;
            weightSum += weight;
            probs.add(weight);
        }
        double norm = weightSum;
        probs.forEach(aDouble -> aDouble /= norm);
        return new AliasMethod(probs);
    }

    /**
     * random walk in the graph starting from a node
     * @param walkLength the steps of this walk
     * @param startNode the start node of this walk
     * @return the path that we pass, expressed as a Node List
     */
    private List<Node> walk(int walkLength, Node startNode) {
        List<Node> path = new ArrayList<>();
        path.add(startNode);

        while (path.size() < walkLength) {
            Node current = path.get(path.size()-1); // the last node on the path
            List<Node> neighbors = this.sortedNeighborList(current);
            if (neighbors.size() > 0) {
                if (path.size() == 1) {
                    int nextIndex = aliasNodes.get(current).next();
                    path.add(neighbors.get(nextIndex));
                } else {
                    Node prev = path.get(path.size()-2);
                    int nextIndex = aliasEdges.get(this.getEdge(prev, current)).next();
                    path.add(neighbors.get(nextIndex));
                }
            } else break;
        }
        return path;
    }

    /**
     * simulation of a bunch of walks
     * @param numWalks iteration times
     * @param walkLength steps of every walk
     * @return the list of paths that we've walked
     */
    public List<List> simulateWalks(int numWalks, int walkLength) {
        List<List> pathList = new ArrayList<>();
        System.out.println("Walk iteration:");
        List<Node> nodeList = new ArrayList<>(nodeSet);
        for (int i = 0; i < numWalks; i++) {
            System.out.println(i+1+"/"+numWalks);
            Collections.shuffle(nodeList);
            for (Node node :
                    nodeList) {
                pathList.add(walk(walkLength, node));
            }
        }
        return pathList;
    }

    /**
     * get a node's neighbors in a sorted list
     * the set of the neighbors of node is defined as {x|node-->x}
     * sort the nodes according to its ids
     * @param node the node
     * @return a sorted list of nodes
     */
    private List<Node> sortedNeighborList(Node node) {
        List<Node> neighborList = new ArrayList<>();
        for (Node n : nodeSet) {
            if (this.hasEdge(node, n)) neighborList.add(n); // only node-->n
        }
        neighborList.sort(Comparator.comparingInt(n -> n.id));
        return neighborList;
    }

    /**
     * check whether there is an edge between two nodes
     * note that all the edges in the graph are directive
     * @param src node1
     * @param dst node2
     * @return true is there is an edge
     */
    private boolean hasEdge(Node src, Node dst) {
        for (Edge edge : edgeSet) {
            if (edge.equals(new Edge(src, dst))) {
                return true;
            }
        }
        return false;
    }

    /**
     * get the edge between two nodes
     * @param src node1
     * @param dst node2
     * @return the edge, null is not exist such an edge
     */
    private Edge getEdge(Node src, Node dst) {
        for (Edge edge : edgeSet) {
            if (edge.equals(new Edge(src, dst))) {
                return edge;
            }
        }
        throw new NoSuchElementException();
    }

    /**
     * add a new edge to the graph
     * if such an edge already exists, update the weight
     * note that all the edges in the graph are directed
     * if the graph is not directed,
     * we just simply add two directed edges with the opposite directions
     * that connect two nodes
     * @param src first node of the edge
     * @param dst second node of the edge
     * @param weight of the edge
     */
    private void addEdge(Node src, Node dst, double weight) {
        if (directed) {
            Edge edge;
            if (hasEdge(src, dst)) {
                edge = getEdge(src, dst);
                edge.weight = weight; // update the weight of the edge
            } else {
                edge = new Edge(src, dst, weight);
                edgeSet.add(edge); // add it to edge set
            }
        } else {
            Edge edge1;
            Edge edge2;
            if (hasEdge(src, dst)) {
                edge1 = getEdge(src, dst);
                edge2 = getEdge(dst, src);
                // update the weight of the edges
                edge1.weight = weight;
                edge2.weight = weight;
            } else {
                edge1 = new Edge(src, dst, weight);
                edge2 = new Edge(dst, src, weight);
                // add it to edge set
                edgeSet.add(edge1);
                edgeSet.add(edge2);
            }
        }
    }

    /**
     * add a node with the id to the graph
     * if such a node already exists, return it and do nothing
     * if not, create a new node, add it to the graph and return it
     * @param id the id of the node
     * @return the node found
     */
    private Node addNode(int id) {
        for (Node node : nodeSet) {
            if (node.id == id) {
                return node;
            }
        }
        // not exists, create a new node with the id
        Node node = new Node(id);
        // add it to the nodeSet
        nodeSet.add(node);
        return node;
    }

    class Node {

        private int id;

        Node(int id) {
            this.id = id;
        }

        boolean equals(Node that) {
            return this.id == that.id;
        }

        public int getId() {
            return id;
        }
    }

    class Edge {

        private Node src, dst;
        private double weight;

        Edge(Node src, Node dst) {
            if (src == null || dst == null)
                throw new IllegalArgumentException();
            this.src = src;
            this.dst = dst;
        }

        Edge(Node src, Node dst, double weight) {
            if (src == null || dst == null)
                throw new IllegalArgumentException();
            this.src = src;
            this.dst = dst;
            this.weight = weight;
        }

        /**
         * two edges are equal if and only if they start at the same node
         * and end at the same node
         * @param that the node to compare
         * @return true if two are equal
         */
        boolean equals(Edge that) {
            return this.src.equals(that.src)
                    && this.dst.equals(that.dst);
        }
    }

}


