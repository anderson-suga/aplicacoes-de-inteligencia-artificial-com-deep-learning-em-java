import org.apache.commons.lang3.tuple.ImmutableTriple;
import org.apache.commons.lang3.tuple.Triple;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.*;
import java.util.List;

public class AutoencoderDeteccaoAnomalia {

    public static void main(String[] args) throws Exception {

        // Define a rede. 784 na camada de entrada e na camada de saída (MNIST - 28x28).
        //784 -> 250 -> 10 -> 250 -> 784
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .iterations(1)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.ADAGRAD)
                .activation(Activation.RELU)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(0.05)
                .regularization(true).l2(0.0001)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(784).nOut(250)
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(250).nOut(10)
                        .build())
                .layer(2, new DenseLayer.Builder().nIn(10).nOut(250)
                        .build())
                .layer(3, new OutputLayer.Builder().nIn(250).nOut(784)
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .build())
                .pretrain(false).backprop(true)
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(1)));

        //Carrega os dados e divide em treino e teste. 40000 treino, 10000 teste
        DataSetIterator iter = new MnistDataSetIterator(100,
                50000,false);

        List<INDArray> featuresTrain = new ArrayList<>();
        List<INDArray> featuresTest = new ArrayList<>();
        List<INDArray> labelsTest = new ArrayList<>();

        Random r = new Random(12345);
        while(iter.hasNext()){
            DataSet ds = iter.next();
            SplitTestAndTrain split = ds.splitTestAndTrain(80, r);
            featuresTrain.add(split.getTrain().getFeatureMatrix());
            DataSet dsTest = split.getTest();
            featuresTest.add(dsTest.getFeatureMatrix());
            INDArray indexes = Nd4j.argMax(dsTest.getLabels(),1); //Converte representaão one-hot -> index
            labelsTest.add(indexes);
        }

        //Treina o modelo
        int nEpochs = 30;
        for( int epoch=0; epoch<nEpochs; epoch++ ){
            for(INDArray data : featuresTrain){
                net.fit(data,data);
            }
            System.out.println("Epoch " + epoch + " completo");
        }

        //Avalia o modelo nos dados de teste
        //Calcula o score de cada digito/exemplo nos dados de teste separadamente
        //Cria uma tripla: (score, digit, and INDArray data) e ordena por score
        //Isso nos permite obter os N melhores e piores dígitos de cada tipo
        Map<Integer,List<Triple<Double,Integer,INDArray>>> listsByDigit = new HashMap<>();
        for( int i=0; i<10; i++ ) listsByDigit.put(i,new ArrayList<Triple<Double,Integer,INDArray>>());

        int count = 0;
        for( int i=0; i<featuresTest.size(); i++ ){
            INDArray testData = featuresTest.get(i);
            INDArray labels = labelsTest.get(i);
            int nRows = testData.rows();
            for( int j=0; j<nRows; j++){
                INDArray example = testData.getRow(j);
                int label = (int)labels.getDouble(j);
                double score = net.score(new DataSet(example,example));
                listsByDigit.get(label).add(new ImmutableTriple<>(score, count++, example));
            }
        }

        //Ordena os dados de cada dígito por score
        Comparator<Triple<Double, Integer, INDArray>> c = new Comparator<Triple<Double, Integer, INDArray>>() {
            @Override
            public int compare(Triple<Double, Integer, INDArray> o1, Triple<Double, Integer, INDArray> o2) {
                return Double.compare(o1.getLeft(),o2.getLeft());
            }
        };

        for(List<Triple<Double, Integer, INDArray>> list : listsByDigit.values()){
            Collections.sort(list, c);
        }

        //Seleciona as 5 melhores e as 5 piores imagens de cada dígitos
        List<INDArray> best = new ArrayList<>(50);
        List<INDArray> worst = new ArrayList<>(50);
        for( int i=0; i<10; i++ ){
            List<Triple<Double,Integer,INDArray>> list = listsByDigit.get(i);
            for( int j=0; j<5; j++ ){
                best.add(list.get(j).getRight());
                worst.add(list.get(list.size()-j-1).getRight());
            }
        }

        //Visualiza os melhores e os piores dígitos
        MNISTVisualizer bestVisualizer = new MNISTVisualizer(2.0,best,"Melhor (Low Rec. Error)");
        bestVisualizer.visualize();

        MNISTVisualizer worstVisualizer = new MNISTVisualizer(2.0,worst,"Pior (High Rec. Error)");
        worstVisualizer.visualize();
    }

    public static class MNISTVisualizer {
        private double imageScale;
        private List<INDArray> digits;
        private String title;
        private int gridWidth;

        public MNISTVisualizer(double imageScale, List<INDArray> digits, String title ) {
            this(imageScale, digits, title, 5);
        }

        public MNISTVisualizer(double imageScale, List<INDArray> digits, String title, int gridWidth ) {
            this.imageScale = imageScale;
            this.digits = digits;
            this.title = title;
            this.gridWidth = gridWidth;
        }

        public void visualize(){
            JFrame frame = new JFrame();
            frame.setTitle(title);
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

            JPanel panel = new JPanel();
            panel.setLayout(new GridLayout(0,gridWidth));

            List<JLabel> list = getComponents();
            for(JLabel image : list){
                panel.add(image);
            }

            frame.add(panel);
            frame.setVisible(true);
            frame.pack();
        }

        private List<JLabel> getComponents(){
            List<JLabel> images = new ArrayList<>();
            for( INDArray arr : digits ){
                BufferedImage bi = new BufferedImage(28,28,BufferedImage.TYPE_BYTE_GRAY);
                for( int i=0; i<784; i++ ){
                    bi.getRaster().setSample(i % 28, i / 28, 0, (int)(255*arr.getDouble(i)));
                }
                ImageIcon orig = new ImageIcon(bi);
                Image imageScaled = orig.getImage().getScaledInstance((int)(imageScale*28),(int)(imageScale*28),Image.SCALE_REPLICATE);
                ImageIcon scaled = new ImageIcon(imageScaled);
                images.add(new JLabel(scaled));
            }
            return images;
        }
    }
}