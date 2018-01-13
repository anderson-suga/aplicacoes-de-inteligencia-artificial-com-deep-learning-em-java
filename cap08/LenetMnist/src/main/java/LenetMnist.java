import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;

/**
 * Usando Lenet para classificar dados do MNIST
 */
public class LenetMnist {

    public static void main(String[] args) throws Exception {
        int nChannels = 1; // Número de canais de cores
        int outputNum = 10; // Número de saídas possíveis
        int batchSize = 64; // Tamanho do batch
        int nEpochs = 1; // Número de epochs
        int iterations = 1; // Número de iterações
        int seed = 123; // Seed


        System.out.println("Carregando os dados....");
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize,true,12345);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize,false,12345);


        System.out.println("Construindo o modelo....");

        // Definindo as taxas de aprendizagem que vão variar conforme o tempo
        Map<Integer, Double> lrSchedule = new HashMap<>();
        lrSchedule.put(0, 0.01);
        lrSchedule.put(1000, 0.005);
        lrSchedule.put(3000, 0.001);

        // Configurando a rede
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations) 
                .regularization(true).l2(0.0005)
                .learningRate(.01)
                .learningRateDecayPolicy(LearningRatePolicy.Schedule) // Taxa de aprendizado que varia com o tempo
                .learningRateSchedule(lrSchedule)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5) // Primeira camada convolucional: kernel de 5 por 5
                        .nIn(nChannels) // Número de canais de entradas
                        .stride(1, 1) // Stride
                        .nOut(20) // Número de saídas = número de filtros
                        .activation(Activation.IDENTITY) // Função de ativação IDENTITY
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX) // Camada de max pooling
                        .kernelSize(2,2) // Kernel de 2 por 2
                        .stride(2,2) // Stride de 2 por 2
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5) // Camada convolucional: kernel de 5 por 5
                        .stride(1, 1) // Stride
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX) // Camada de max pooling
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(4, new DenseLayer.Builder().activation(Activation.RELU) // Camada totalmente conectada
                        .nOut(500).build()) // 500 neurônios = 500 saídas
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD) // Camada de saída
                        .nOut(outputNum) // 10 neurônios definidos acima = 10 saídas
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(28,28,1)) // Definindo o formato das entradas da rede 
                .backprop(true).pretrain(false).build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();


        System.out.println("Treinando o modelo....");
        for( int i=0; i<nEpochs; i++ ) {
            model.fit(mnistTrain);
            System.out.println("*** Epoch " + i + "***");

            System.out.println("Avaliando o modelo....");
            Evaluation eval = model.evaluate(mnistTest);
            System.out.println(eval.stats());
            mnistTest.reset();
        }
        System.out.println("****************Fim do exemplo********************");
    }
}