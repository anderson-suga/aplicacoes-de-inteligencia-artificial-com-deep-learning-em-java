import org.apache.commons.io.FilenameUtils;
import org.datavec.image.loader.CifarLoader;
import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import java.io.File;
import java.io.IOException;

public class Cifar {
    private static final String DATA_PATH = FilenameUtils.concat(System.getProperty("user.dir"),
            "/");

    private static String labelStr = "[]";
    private static int altura = 32;
    private static int largura = 32;
    private static int canais = 3;
    private static int nClasses = CifarLoader.NUM_LABELS;
    private static int nExemplos = 50000;
    private static int batchSize = 100;
    private static int iterations = 1;
    private static int seed = 123;
    private static boolean preProcessCifar = false;
    private static int epochs = 1;

    public static void main(String[] args) throws Exception {

        DataTypeUtil.setDTypeForContext(DataBuffer.Type.FLOAT);
        MultiLayerNetwork modelo = criarRede();

        CifarDataSetIterator cifar = new CifarDataSetIterator(batchSize, nExemplos,
                new int[] {altura, largura, canais}, preProcessCifar, true);
        CifarDataSetIterator cifarEval = new CifarDataSetIterator(batchSize, 10000,
                new int[] {altura, largura, canais}, preProcessCifar, false);

        labelStr = String.join(",", cifar.getLabels().toArray(
                new String[cifar.getLabels().size()]));
        System.out.println("=====Treinando o modelo========");
        for ( int i = 0; i < epochs; i ++ ) {
            System.out.println("Epoch=====================" + i);
            modelo.fit(cifar);
        }
        
        System.out.println("=====Avaliando o modelo========");
        Evaluation eval = new Evaluation(cifarEval.getLabels());
        while(cifarEval.hasNext()) {
            DataSet testDS = cifarEval.next(batchSize);
            INDArray output = modelo.output(testDS.getFeatureMatrix());
            eval.eval(testDS.getLabels(), output);
        }
        System.out.println(eval.stats());

        salvarModelo(modelo, "cifar.json");
    }


    public static MultiLayerNetwork criarRede() throws IOException {
        System.out.println("Criando a rede...");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .cacheMode(CacheMode.DEVICE)
                .updater(Updater.ADAM)
                .iterations(iterations)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .l1(1e-4)
                .regularization(true)
                .l2(5 * 1e-4)
                .list()
                .layer(0, new ConvolutionLayer.Builder(new int[]{4, 4}, new int[]{1, 1},
                        new int[]{0, 0}).name("cnn1").convolutionMode(ConvolutionMode.Same)
                    .nIn(3).nOut(64).weightInit(WeightInit.XAVIER_UNIFORM).activation(Activation.RELU)
                    .learningRate(1e-2).biasInit(1e-2).biasLearningRate(1e-2*2).build())
                .layer(1, new ConvolutionLayer.Builder(new int[]{4,4}, new int[] {1,1}, new int[] {0,0})
                    .name("cnn2").convolutionMode(ConvolutionMode.Same)
                    .nOut(64).weightInit(WeightInit.XAVIER_UNIFORM).activation(Activation.RELU)
                    .learningRate(1e-2).biasInit(1e-2).biasLearningRate(1e-2*2).build())
                .layer(2, new SubsamplingLayer.Builder(PoolingType.MAX, new int[]{2,2})
                    .name("maxpool2").build())
                .layer(3, new ConvolutionLayer.Builder(new int[]{4,4}, new int[] {1,1}, new int[] {0,0})
                    .name("cnn3").convolutionMode(ConvolutionMode.Same)
                    .nOut(96).weightInit(WeightInit.XAVIER_UNIFORM).activation(Activation.RELU)
                    .learningRate(1e-2).biasInit(1e-2).biasLearningRate(1e-2*2).build())
                .layer(4, new ConvolutionLayer.Builder(new int[]{4,4}, new int[] {1,1}, new int[] {0,0})
                    .name("cnn4").convolutionMode(ConvolutionMode.Same)
                    .nOut(96).weightInit(WeightInit.XAVIER_UNIFORM).activation(Activation.RELU)
                    .learningRate(1e-2).biasInit(1e-2).biasLearningRate(1e-2*2).build())
                .layer(5, new ConvolutionLayer.Builder(new int[]{3,3}, new int[] {1,1}, new int[] {0,0})
                    .name("cnn5").convolutionMode(ConvolutionMode.Same)
                    .nOut(128).weightInit(WeightInit.XAVIER_UNIFORM).activation(Activation.RELU)
                    .learningRate(1e-2).biasInit(1e-2).biasLearningRate(1e-2*2).build())
                .layer(6, new ConvolutionLayer.Builder(new int[]{3,3}, new int[] {1,1}, new int[] {0,0})
                    .name("cnn6").convolutionMode(ConvolutionMode.Same)
                    .nOut(128).weightInit(WeightInit.XAVIER_UNIFORM).activation(Activation.RELU)
                    .learningRate(1e-2).biasInit(1e-2).biasLearningRate(1e-2*2).build())
                .layer(7, new ConvolutionLayer.Builder(new int[]{2,2}, new int[] {1,1}, new int[] {0,0})
                    .name("cnn7").convolutionMode(ConvolutionMode.Same)
                    .nOut(256).weightInit(WeightInit.XAVIER_UNIFORM).activation(Activation.RELU)
                    .learningRate(1e-2).biasInit(1e-2).biasLearningRate(1e-2*2).build())
                .layer(8, new ConvolutionLayer.Builder(new int[]{2,2}, new int[] {1,1}, new int[] {0,0})
                    .name("cnn8").convolutionMode(ConvolutionMode.Same)
                    .nOut(256).weightInit(WeightInit.XAVIER_UNIFORM).activation(Activation.RELU)
                    .learningRate(1e-2).biasInit(1e-2).biasLearningRate(1e-2*2).build())
                .layer(9, new SubsamplingLayer.Builder(PoolingType.MAX, new int[]{2,2})
                    .name("maxpool8").build())
                .layer(10, new DenseLayer.Builder().name("ffn1").nOut(1024).learningRate(1e-3)
                    .biasInit(1e-3).biasLearningRate(1e-3*2).build())
                .layer(11,new DropoutLayer.Builder().name("dropout1").dropOut(0.2).build())
                .layer(12, new DenseLayer.Builder().name("ffn2").nOut(1024).learningRate(1e-2)
                    .biasInit(1e-2).biasLearningRate(1e-2*2).build())
                .layer(13,new DropoutLayer.Builder().name("dropout2").dropOut(0.2).build())
                .layer(14, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(nClasses)
                        .activation(Activation.SOFTMAX)
                        .build())
                .backprop(true)
                .pretrain(false)
                .setInputType(InputType.convolutional(altura, largura, canais))
                .build();

        MultiLayerNetwork modelo = new MultiLayerNetwork(conf);
        modelo.init();
        return modelo;
    }


    public static MultiLayerNetwork salvarModelo(MultiLayerNetwork model, String fileName) {
        File locationModelFile = new File(DATA_PATH + fileName);
        boolean saveUpdater = false;
        try {
            ModelSerializer.writeModel(model,locationModelFile,saveUpdater);
        } catch (Exception e) {
            System.out.println("Erro ao salvar o modelo !");
        }
        return model;
    }

}
