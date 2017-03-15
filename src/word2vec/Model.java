package word2vec;

import word2vec.domain.HiddenNeuron;
import word2vec.domain.Neuron;
import word2vec.domain.WordNeuron;
import word2vec.util.Haffman;
import word2vec.util.MapCount;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

public class Model {
    private static final int DEFAULT_LAYER_SIZE = 200;
    private static final int DEFAULT_WINDOW_SIZE = 5;
    private static final double DEFAULT_SAMPLE = 1e-3;
    private static final double DEFAULT_ALPHA = 0.025;
    
    private static final int EXP_TABLE_SIZE = 1000;

    private Map<String, Neuron> word2neuron = new HashMap<>();

    private int layerSize, windowSize;
    private double sample, alpha, startingAlpha;
    private boolean isCBOW;
    
    private double[] expTable = new double[EXP_TABLE_SIZE];
    
    private int trainWordsCount = 0;
    
    private int MAX_EXP = 6;
    
    public Model(Boolean isCBOW, Integer layerSize, Integer windowSize, Double alpha, Double sample) {
        createExpTable();
        if (isCBOW != null) this.isCBOW = isCBOW;
        else this.isCBOW = false;
        if (layerSize != null) this.layerSize = layerSize;
        else this.layerSize = DEFAULT_LAYER_SIZE;
        if (windowSize != null) this.windowSize = windowSize;
        else this.windowSize = DEFAULT_WINDOW_SIZE;
        if (alpha != null) this.alpha = alpha;
        else this.alpha = DEFAULT_ALPHA;
        if (sample != null) this.sample = sample;
        else this.sample = DEFAULT_SAMPLE;
    }
    
    /**
    * train model with the file data
    *
    * @throws IOException
    */
    private void trainModel(File file) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file)));
        long nextRandom = 5;
        int wordCount = 0;
        int lastWordCount = 0;
        int wordCountActual = 0;
        String lineTxt;
        while ((lineTxt = br.readLine()) != null) {
            if (wordCount - lastWordCount > 10000) {
                System.out.println("alpha:" + alpha + "\tProgress: "
                        + (int) (wordCountActual / (double) (trainWordsCount + 1) * 100)
                        + "%");
                wordCountActual += wordCount - lastWordCount;
                lastWordCount = wordCount;
                alpha = startingAlpha * (1 - wordCountActual / (double) (trainWordsCount + 1));
                if (alpha < startingAlpha * 0.0001) {
                    alpha = startingAlpha * 0.0001;
                }
            }
            String[] strs = lineTxt.split(" ");
            wordCount += strs.length;
            List<WordNeuron> sentence = new ArrayList<WordNeuron>();
            for (int i = 0; i < strs.length; i++) {
                Neuron entry = word2neuron.get(strs[i]);
                if (entry == null) {
                continue;
                }
                // The subsampling randomly discards frequent words while keeping the
                // ranking same
                if (sample > 0) {
                    double ran = (Math.sqrt(entry.freq / (sample * trainWordsCount)) + 1)
                                * (sample * trainWordsCount) / entry.freq;
                    nextRandom = nextRandom * 25214903917L + 11;
                    if (ran < (nextRandom & 0xFFFF) / (double) 65536) {
                        continue;
                    }
                }
                sentence.add((WordNeuron) entry);
            }
            for (int index = 0; index < sentence.size(); index++) {
                nextRandom = nextRandom * 25214903917L + 11;
                if (isCBOW) cbowGram(index, sentence, (int) nextRandom % windowSize);
                else skipGram(index, sentence, (int) nextRandom % windowSize);
            }

        }
        System.out.println("Vocab size: " + word2neuron.size());
        System.out.println("Words in train file: " + trainWordsCount);
        System.out.println("success train over!");
    }
    
    /**
    * skip gram train
    *
    */
    private void skipGram(int index, List<WordNeuron> sentence, int b) {
        WordNeuron word = sentence.get(index);
        int a, c;
        for (a = b; a < windowSize * 2 + 1 - b; a++) {
            if (a == windowSize) {
                continue;
            }
            c = index - windowSize + a;
            if (c < 0 || c >= sentence.size()) {
                continue;
            }

            double[] neu1e = new double[layerSize];// 误差项
            // HIERARCHICAL SOFTMAX
            List<Neuron> neurons = word.neurons;
            WordNeuron we = sentence.get(c);
            for (int i = 0; i < neurons.size(); i++) {
                HiddenNeuron out = (HiddenNeuron) neurons.get(i);
                double f = 0;
                // Propagate hidden -> output
                for (int j = 0; j < layerSize; j++) {
                    f += we.syn0[j] * out.syn1[j];
                }
                if (f <= -MAX_EXP || f >= MAX_EXP) {
                    continue;
                } else {
                    f = (f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2);
                    f = expTable[(int) f];
                }
                // 'g' is the gradient multiplied by the learning rate
                double g = (1 - word.codeArr[i] - f) * alpha;
                // Propagate errors output -> hidden
                for (c = 0; c < layerSize; c++) {
                    neu1e[c] += g * out.syn1[c];
                }
                // Model weights hidden -> output
                for (c = 0; c < layerSize; c++) {
                    out.syn1[c] += g * we.syn0[c];
                }
            }

            // Model weights input -> hidden
            for (int j = 0; j < layerSize; j++) {
                we.syn0[j] += neu1e[j];
            }
        }
    
    }
    
    /**
    * bag of words
    *
    */
    private void cbowGram(int index, List<WordNeuron> sentence, int b) {
        WordNeuron word = sentence.get(index);
        int a, c;

        List<Neuron> neurons = word.neurons;
        double[] neu1e = new double[layerSize];
        double[] neu1 = new double[layerSize];
        WordNeuron last_word;

        for (a = b; a < windowSize * 2 + 1 - b; a++)
            if (a != windowSize) {
                c = index - windowSize + a;
                if (c < 0) continue;
                if (c >= sentence.size()) continue;
                last_word = sentence.get(c);
                if (last_word == null) continue;
                for (c = 0; c < layerSize; c++) neu1[c] += last_word.syn0[c];
            }

        // HIERARCHICAL SOFTMAX
        for (int d = 0; d < neurons.size(); d++) {
            HiddenNeuron out = (HiddenNeuron) neurons.get(d);
            double f = 0;
            // Propagate hidden -> output
            for (c = 0; c < layerSize; c++) f += neu1[c] * out.syn1[c];
            if (f <= -MAX_EXP) continue;
            else if (f >= MAX_EXP) continue;
            else f = expTable[(int) ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
            // 'g' is the gradient multiplied by the learning rate
            // double g = (1 - word.codeArr[d] - f) * alpha;
            // double g = f*(1-f)*( word.codeArr[i] - f) * alpha;
            double g = f * (1 - f) * (word.codeArr[d] - f) * alpha;
            for (c = 0; c < layerSize; c++) {
                neu1e[c] += g * out.syn1[c];
            }
            // Model weights hidden -> output
            for (c = 0; c < layerSize; c++) {
                out.syn1[c] += g * neu1[c];
            }
        }
        for (a = b; a < windowSize * 2 + 1 - b; a++) {
            if (a != windowSize) {
                c = index - windowSize + a;
                if (c < 0) continue;
                if (c >= sentence.size()) continue;
                last_word = sentence.get(c);
                if (last_word == null) continue;
                for (c = 0; c < layerSize; c++) last_word.syn0[c] += neu1e[c];
            }

        }
    }
    
    /**
    *
    * count word frequency in a file
    * @param file
    * @throws IOException
    */
    private void countWordFreq(File file) throws IOException {
        MapCount<String> mc = new MapCount<>();
        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file)));
        String lineTxt;
        while ((lineTxt = br.readLine()) != null) {
            String[] split = lineTxt.split(" ");
            trainWordsCount += split.length;
            for (String string : split) {
                mc.add(string);
            }
        }
        for (Entry<String, Integer> element : mc.get().entrySet()) {
            word2neuron.put(element.getKey(), new WordNeuron(element.getKey(),
            (double) element.getValue() / mc.size(), layerSize));
        }
    }
    
    /**
    * Pre-compute the exp() table f(x) = x / (x + 1)
    */
    private void createExpTable() {
        for (int i = 0; i < EXP_TABLE_SIZE; i++) {
            expTable[i] = Math.exp(((i / (double) EXP_TABLE_SIZE * 2 - 1) * MAX_EXP));
            expTable[i] = expTable[i] / (expTable[i] + 1);
        }
    }
    
    /**
    *
    * learn from the data in the file
    */
    public void learnFile(File file) throws IOException {
        countWordFreq(file);
        new Haffman(layerSize).make(word2neuron.values());

        for (Neuron neuron : word2neuron.values()) {
            ((WordNeuron) neuron).makeNeurons();
        }

        trainModel(file);
    }
    
    public void storeModel(File file) {
        try {
            FileWriter fw = new FileWriter(file);
            fw.write(word2neuron.size()+" "+layerSize+"\n");
            double[] syn0;
            for (Entry<String, Neuron> element : word2neuron.entrySet()) {
                fw.write(element.getKey()+" ");
                syn0 = ((WordNeuron) element.getValue()).syn0;
                for (double d : syn0) {
                    fw.write(((Double) d).floatValue()+" ");
                }
                fw.write("\n");
            }
            fw.flush();
            fw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
