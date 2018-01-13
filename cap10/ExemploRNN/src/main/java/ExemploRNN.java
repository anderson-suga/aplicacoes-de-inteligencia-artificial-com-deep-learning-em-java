
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Random;

/**
 * Nesse exemplo treinamos uma RNN. Depois de treinada nós precisamos apenas
 * informar o primeiro caractere da FRASE e ela conseguirá prever todos os demais caracteres
 *
 */
public class ExemploRNN {

    // Definindo a FRASE que a rede vai aprender. O * serve para marcar o início da frase
    private static final char[] FRASE =
            "*A vingança nunca é plena, mata a alma e envenena. - Madruga, Seu".toCharArray();

    // Uma lista com todos os caracteres possíveis
    private static final List<Character> LISTA_CARACTERES_FRASE = new ArrayList<>();

    // Dimensões da RNN
    private static final int CAMADAS_OCULTAS_WIDTH = 50;
    private static final int CAMADAS_OCULTAS_CONT = 2;

    public static void main(String[] args) {

        // Criando uma lista dedicada para todos os caracteres presentes em
        // LISTA_CARACTERES_FRASE
        LinkedHashSet<Character> FRASE_CARACTERES = new LinkedHashSet<>();
        for (char c : FRASE)
            FRASE_CARACTERES.add(c);
        LISTA_CARACTERES_FRASE.addAll(FRASE_CARACTERES);

        // Alguns parâmetros comuns
        NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
        builder.iterations(10);
        builder.learningRate(0.001);
        builder.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
        builder.seed(123);
        builder.biasInit(0);
        builder.miniBatch(false);
        builder.updater(Updater.RMSPROP);
        builder.weightInit(WeightInit.XAVIER);

        ListBuilder listBuilder = builder.list();

        // Primeira diferença: para RNNs, usamos use GravesLSTM.Builder
        for (int i = 0; i < CAMADAS_OCULTAS_CONT; i++) {
            GravesLSTM.Builder hiddenLayerBuilder = new GravesLSTM.Builder();
            hiddenLayerBuilder.nIn(i == 0 ? FRASE_CARACTERES.size() : CAMADAS_OCULTAS_WIDTH);
            hiddenLayerBuilder.nOut(CAMADAS_OCULTAS_WIDTH);
            hiddenLayerBuilder.activation(Activation.TANH);
            listBuilder.layer(i, hiddenLayerBuilder.build());
        }

        // Precisamos usar RnnOutputLayer para nossa RNN
        RnnOutputLayer.Builder outputLayerBuilder = new RnnOutputLayer.Builder(
                LossFunction.MCXENT);
        // Função de ativação softmax
        outputLayerBuilder.activation(Activation.SOFTMAX);
        outputLayerBuilder.nIn(CAMADAS_OCULTAS_WIDTH);
        outputLayerBuilder.nOut(FRASE_CARACTERES.size());
        listBuilder.layer(CAMADAS_OCULTAS_CONT, outputLayerBuilder.build());

        // Finalizando as configurações
        listBuilder.pretrain(false);
        listBuilder.backprop(true);

        // Criando a rede
        MultiLayerConfiguration conf = listBuilder.build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

		/*
		 * Criando os dados de treino
		 */
        // Criando arrays de entrada e de saída
        INDArray input = Nd4j.zeros(1, LISTA_CARACTERES_FRASE.size(), FRASE.length);
        INDArray labels = Nd4j.zeros(1, LISTA_CARACTERES_FRASE.size(), FRASE.length);
        // Loop por cada elemento da frase
        int samplePos = 0;
        for (char currentChar : FRASE) {
            char nextChar = FRASE[(samplePos + 1) % (FRASE.length)];
            // Configura o caractere atual como sendo a entrada
            input.putScalar(new int[] { 0, LISTA_CARACTERES_FRASE.indexOf(currentChar),
                    samplePos }, 1);
            // Configura o caractere seguinte como sendo a saída
            labels.putScalar(new int[] { 0, LISTA_CARACTERES_FRASE.indexOf(nextChar),
                    samplePos }, 1);
            samplePos++;
        }
        DataSet trainingData = new DataSet(input, labels);

        // Alguns epochs
        for (int epoch = 0; epoch < 100; epoch++) {

            System.out.println("Epoch " + epoch);

            // Treinando os dados
            net.fit(trainingData);

            // Limpa o status corrente
            net.rnnClearPreviousState();

            // Coloca o primeiro caractere como inicializador da rede para teste
            INDArray testInit = Nd4j.zeros(LISTA_CARACTERES_FRASE.size());
            testInit.putScalar(LISTA_CARACTERES_FRASE.indexOf(FRASE[0]), 1);

            // Roda uma vez com  rnnTimeStep()
            INDArray output = net.rnnTimeStep(testInit);

            // Agora a rede deve predizer os próximos caracteres por n vezes
            for (char dummy : FRASE) {

                // Processa a última saída da rede
                int sampledCharacterIdx = Nd4j.getExecutioner().exec(new IMax(output),
                        1).getInt(0);

                // Imprime a saída
                System.out.print(LISTA_CARACTERES_FRASE.get(sampledCharacterIdx));

                // Usa a última saída como entrada
                INDArray nextInput = Nd4j.zeros(LISTA_CARACTERES_FRASE.size());
                nextInput.putScalar(sampledCharacterIdx, 1);
                output = net.rnnTimeStep(nextInput);

            }
            System.out.print("\n");
        }
    }
}