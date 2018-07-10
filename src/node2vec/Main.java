package node2vec;

import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.impl.Arguments;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.ArgumentParserException;
import net.sourceforge.argparse4j.inf.Namespace;
import word2vec.Model;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

/**
 * Created by freemso on 17-3-14.
 */
public class Main {
    public static void main(String[] args) {
        // parse arguments
        ArgumentParser parser = ArgumentParsers
                .newArgumentParser("node2vec")
                .defaultHelp(true)
                .description("Run node2vec");
        parser.addArgument("-i", "--input")
                .nargs("?")
                .setDefault("graph/karate.edgelist")
                .help("Input graph edge information path");
        parser.addArgument("-o", "--output")
                .nargs("?")
                .setDefault("emb/karate.emb")
                .help("Output embedding path");
        parser.addArgument("--dimensions")
                .type(Integer.class)
                .setDefault(128)
                .help("Number of dimensions. Default is 128");
        parser.addArgument("--walkLength")
                .type(Integer.class)
                .setDefault(80)
                .help("Length og walk per source. Default is 80");
        parser.addArgument("--numWalks")
                .type(Integer.class)
                .setDefault(10)
                .help("Number of walks per source. Default is 10");
        parser.addArgument("--windowSize")
                .type(Integer.class)
                .setDefault(10)
                .help("Context size for optimization. Default is 10");
        parser.addArgument("--iter")
                .type(Integer.class)
                .setDefault(1)
                .help("Number of epochs in SGD");
        parser.addArgument("--workers")
                .type(Integer.class)
                .setDefault(8)
                .help("Number of parallel workers. Default is 8");
        parser.addArgument("-p", "--p")
                .type(Double.class)
                .setDefault(1.0)
                .help("Return hyperparameter. Default is 1");
        parser.addArgument("-q", "--q")
                .type(Double.class)
                .setDefault(1.0)
                .help("Inout hyperparameter. Default is 1");
        parser.addArgument("--weighted")
                .dest("weighted")
                .action(Arguments.storeTrue())
                .help("Boolean specifying (un)weighted. Default is unweighted");
        parser.addArgument("--unweighted")
                .dest("weighted")
                .action(Arguments.storeFalse());
        parser.setDefault("weighted", false);
        parser.addArgument("--directed")
                .dest("directed")
                .action(Arguments.storeTrue())
                .help("node2vec.Graph is (un)directed. Default is undirected");
        parser.addArgument("--undirected")
                .dest("directed")
                .action(Arguments.storeFalse());
        parser.setDefault("directed", false);

        try {
            Namespace ns = parser.parseArgs(args);
            Graph graph = new Graph(ns.get("input"),
                    ns.getBoolean("directed"),
                    ns.getDouble("p"),
                    ns.getDouble("q"));
            List<List> pathList = graph.
                    simulateWalks(ns.getInt("numWalks"),
                    ns.getInt("walkLength"));

            System.out.println("Learning Embedding...");

            // convert path list to string
            String sentList = "";
            for (List<Graph.Node> path :
                    pathList) {
                String sent = "";
                for (Graph.Node node :
                        path) {
                    sent += node.getId() + " ";
                }
                sentList += sent + "\n";
            }
            // write to temp file
            String tempPath = System.getProperty("java.io.tmpdir");
            File tempFile = File.createTempFile("pathList", "txt", new File(tempPath));
            FileWriter fw = new FileWriter(tempFile);
            fw.write(sentList);
            fw.flush();
            fw.close();
            // use word2vec to do word embedding
            Model model = new Model(false, ns.getInt("dimensions"), ns.getInt("windowSize"), null, null);
            model.learnFile(tempFile);
            model.storeModel(new File(ns.getString("output")));

        } catch (ArgumentParserException e) {
            parser.handleError(e);
        } catch (IOException e) {
            System.err.println("invalid arguments");
        }

    }
}
