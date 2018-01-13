import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer.Builder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Nesse exemplo veremos como criar manualmente um DataSet e treiná-lo em uma rede neural simples
 * A rede possuirá dois neurônios de entreda, uma camada oculta com 4 neurônios e dois neurônios de saída
 *
 */
public class ExemploXor {
    public static void main(String[] args) {

        // definição dos valores de entrada, 4 valores de treino com dois neurônios de entrada
        INDArray input = Nd4j.zeros(4, 2);

        // definição dos valores de saída 
        INDArray labels = Nd4j.zeros(4, 2);

        // criação do nosso primeiro DataSet
        // quando as duas entradas são 0
        input.putScalar(new int[]{0, 0}, 0);
        input.putScalar(new int[]{0, 1}, 0);
        // o primeiro neurônio de saída que corresponde ao valor falso, recebe 1, o segundo, que corresponde ao valor verdadeiro, recebe 0
        labels.putScalar(new int[]{0, 0}, 1);
        labels.putScalar(new int[]{0, 1}, 0);

        // quando a primeira entrada é 1 e a segunda é 0
        input.putScalar(new int[]{1, 0}, 1);
        input.putScalar(new int[]{1, 1}, 0);
        // então o xor é verdadeiro e, por isso, o segundo neurônio de saída recebe 1
        labels.putScalar(new int[]{1, 0}, 0);
        labels.putScalar(new int[]{1, 1}, 1);

        // o mesmo de cima
        input.putScalar(new int[]{2, 0}, 0);
        input.putScalar(new int[]{2, 1}, 1);
        labels.putScalar(new int[]{2, 0}, 0);
        labels.putScalar(new int[]{2, 1}, 1);

        // quando as duas entradas são 1, o xor é falso, então o primeiro neurônio de saída recebe 1
        input.putScalar(new int[]{3, 0}, 1);
        input.putScalar(new int[]{3, 1}, 1);
        labels.putScalar(new int[]{3, 0}, 1);
        labels.putScalar(new int[]{3, 1}, 0);

        // criando o objeto DataSet usando as entradas e as saídas
        DataSet ds = new DataSet(input, labels);

        // configurando a rede neural
        NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
        // definindo o número de iterações
        builder.iterations(10000);
        // a taxa de aprendizagem
        builder.learningRate(0.1);
        // fixando um seed para permitir reprodutibilidade na geração de valores aleatórios
        builder.seed(1234);
        // aqui não usaremos dropout, porém, em redes maiores pode ser útil para evitar overfitting
        builder.useDropConnect(false);
        // o algoritmo que utilizaremos para treinar a rede, no nosso caso o gradiente descendente estocástico
        builder.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
        // inicializando o bias com 0
        builder.biasInit(0);
        // como a rede é pequena, não usaremos mini-batchs
        builder.miniBatch(false);

        // criação de um perceptron com duas camadas (incluindo a camada de saída, excluindo a camada de entrada)
        ListBuilder listBuilder = builder.list();

        // nossa camada oculta
        DenseLayer.Builder hiddenLayerBuilder = new DenseLayer.Builder();
        // duas conexões de entrada - nIn ao mesmo tempo define o número de entradas da rede
        hiddenLayerBuilder.nIn(2);
        // número de conexões de saída, nOut ao mesmo tempo define o número de neurônios dessa camada
        hiddenLayerBuilder.nOut(4);
        // usando a função de ativação SIGMOID para manter os resultados entre 0 e 1
        hiddenLayerBuilder.activation(Activation.SIGMOID);
        // inicializando os pesos com números randomicos entre 0 e 1
        hiddenLayerBuilder.weightInit(WeightInit.DISTRIBUTION);
        hiddenLayerBuilder.dist(new UniformDistribution(0, 1));

        // definindo a camada como sendo a camada 0
        listBuilder.layer(0, hiddenLayerBuilder.build());


        // escolhendo a função de custo/perda
        Builder outputLayerBuilder = new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD);
        // nIn precisa possuir o mesmo número de neurônios da saída da camada anterior
        outputLayerBuilder.nIn(4);
        // 2 neurônios para a saída dessa camada
        outputLayerBuilder.nOut(2);
        // função de ativação SOFTMAX
        outputLayerBuilder.activation(Activation.SOFTMAX);
        outputLayerBuilder.weightInit(WeightInit.DISTRIBUTION);
        outputLayerBuilder.dist(new UniformDistribution(0, 1));

        // definindo a camada como sendo a camada 1
        listBuilder.layer(1, outputLayerBuilder.build());

        // essa rede não possuirá fase de pré-treino
        listBuilder.pretrain(false);

        // definindo o uso do backpropagation
        listBuilder.backprop(true);

        // construindo e inicializando a rede
        MultiLayerConfiguration conf = listBuilder.build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        // Imprimindo os parâmetros de cada camada e da rede
        Layer[] layers = net.getLayers();
        int totalNumParams = 0;
        for (int i = 0; i < layers.length; i++) {
            int nParams = layers[i].numParams();
            System.out.println("Número de parâmetros na camada " + i + ": " + nParams);
            totalNumParams += nParams;
        }
        System.out.println("Total de parâmetros na rede: " + totalNumParams);


        // Inicializando o uiServer
        UIServer uiServer = UIServer.getInstance();

        // Configurando onde os dados serão armazenados
        StatsStorage statsStorage = new InMemoryStatsStorage();

        // Configurando o uiServer com o statsStorage
        uiServer.attach(statsStorage);

        // Adicionando o listener na rede
        net.setListeners(new StatsListener(statsStorage));

        // treinando a rede
        net.fit(ds);

        // criando output para cada exemplo de treino
        INDArray output = net.output(ds.getFeatureMatrix());
        System.out.println(output);

        // Avaliando a rede
        Evaluation eval = new Evaluation(2);
        eval.eval(ds.getLabels(), output);
        System.out.println(eval.stats());

    }
}