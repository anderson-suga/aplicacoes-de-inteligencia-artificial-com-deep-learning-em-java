import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
* Multilayer perceptron com duas camadas
**/
public class PerceptronDuasCamadas {

    public static void main(String[] args) throws Exception {
        //Número de linhas e colunas das imagens
        final int nLinhas = 28;
        final int nColunas = 28;
        int nSaidas = 10; // número de classes de saída
        int batchSize = 64; // Tamanho do batch
        int seed = 123; // seed para permitir reprodutibilidade
        int numEpochs = 15; // Número de epochs
        double taxaAprendizado = 0.0015; // Taxa de aprendizado

        // Dados de treino e de teste
        DataSetIterator mnistTreino = new MnistDataSetIterator(batchSize, true, seed);
        DataSetIterator mnistTeste = new MnistDataSetIterator(batchSize, false, seed);


        System.out.println("Construindo o modelo....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed) // Incluindo o seed para reprodutibilidade
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) // Gradiente descendente estocástico
            .iterations(1)
            .activation(Activation.RELU)
            .weightInit(WeightInit.XAVIER)
            .learningRate(taxaAprendizado) // taxa de aprendizado
            .updater(new Nesterovs(0.98))
            .regularization(true).l2(taxaAprendizado * 0.005) // regularização
            .list()
            .layer(0, new DenseLayer.Builder() // Camada 0
                    .nIn(nLinhas * nColunas)
                    .nOut(500)
                    .build())
            .layer(1, new DenseLayer.Builder() // Camada 1
                    .nIn(500)
                    .nOut(100)
                    .build())
            .layer(2, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) // Camada 2
                    .activation(Activation.SOFTMAX)
                    .nIn(100)
                    .nOut(nSaidas)
                    .build())
            .pretrain(false).backprop(true) // Usando backpropagation
            .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        System.out.println("Trainando o modelo....");
        for( int i=0; i<numEpochs; i++ ){
        	System.out.println("Epoch " + i);
            model.fit(mnistTreino);
        }


        System.out.println("Avaliando o modelo....");
        Evaluation eval = new Evaluation(nSaidas); 
        while(mnistTeste.hasNext()){
            DataSet next = mnistTeste.next();
            INDArray output = model.output(next.getFeatureMatrix());
            eval.eval(next.getLabels(), output);
        }

        System.out.println(eval.stats());
        System.out.println("****************Fim do exemplo********************");

    }

}