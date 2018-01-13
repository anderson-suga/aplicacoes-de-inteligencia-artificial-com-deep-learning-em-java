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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;


/** Construindo umMultilayer perceptron de única camada
 * para treinar o MNIST
 *
 * Na rede existe apenas a camada de entrada e uma camada oculta
 *
 * A camada de entrada possui a dimensão de numeroLinhas * numeroColunas
 * que são o número de pixels na horizontal e na vertical da imagem
 * Nessa camada usamos a função de ativação ReLu. Os pesos são inicializados usando Xavier initialization
 *
 * A camada oculta possui um  input de 1000. Eles são gerados pela camada de entrada. Os pesos são, também,
 * inicializados The weights usando Xavier initialization. A função de ativação dessa camada
 * é a softmax que faz com que saída seja normalizada para que
 * a soma das saídas de cada um dos dez neurônios da camada de saída seja um. O neurônio que possuir
 * o maior valor será escolhido como resultado da predissão da rede.
 *
 */
public class MnistCamadaUnica {

    public static void main(String[] args) throws Exception {
        // número de linhas e colunas nas imagens
        final int numeroLinhas = 28;
        final int numeroColunas = 28;
        int outputNum = 10; // número de classes de saída
        int batchSize = 128; // tamanho do batch para cada iteração
        int seed = 123; // seed para manter a reprodutibilidade
        int numEpochs = 15; // número de epochs

        // pegando os DataSetIterators:
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, seed);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, seed);


        System.out.println("Construindo o modelo....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed) //seed para manter a reprodutibilidade
                // gradiente descendente estocástico
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .learningRate(0.006) //taxa de aprendizagem
                .updater(Updater.NESTEROVS)
                .regularization(true).l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder() //primeira camada usando xavier para inicializar os pesos
                        .nIn(numeroLinhas * numeroColunas)
                        .nOut(1000)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) //camada oculta
                        .nIn(1000)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .pretrain(false).backprop(true) // usando backpropagation to treinar a rede
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        System.out.println("Treinando o modelo....");
        for( int i=0; i<numEpochs; i++ ){
            model.fit(mnistTrain);
        }


        System.out.println("Avaliando o modelo....");
        Evaluation eval = new Evaluation(outputNum);
        while(mnistTest.hasNext()){
            DataSet next = mnistTest.next();
            INDArray output = model.output(next.getFeatureMatrix()); //pegando a predição da rede
            eval.eval(next.getLabels(), output); //comparando a predição com os valores esperados
        }

        System.out.println(eval.stats());
        System.out.println("****************Fim do exemplo********************");

    }

}