import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import java.nio.charset.Charset;
import java.util.Random;

public class LSTMMachadoAssis {
    public static void main( String[] args ) throws Exception {
        int tamanhoCamadaLstm = 200;				// Número de unidades em cada camada GravesLSTM
        int tamanhoMiniBatch = 32;					// Tamanho do mini batch para treino
        int tamanhoExemplo = 1000;					// Tamanho de cada sequência de exemplo. Você pode mudar esse valor.
        int tbpttTamanho = 50;                      // Tamanho para truncated backpropagation through time. i.e., atualiza os pesos a cada 50 caracteres
        int numEpochs = 50;						    // Total de epochs
        int gerarExemplosACadaNMiniBatchs = 10;     // Frequência com que geraremos exemplos usando a rede
        int nExemplosGerar = 4;					    // Número de exemplos que serão gerados de cada vez
        int nCaracteresExemplo = 800;				// Tamanho de cada exemplo a ser gerado
        String caracterInicialGerar = null;		    // Caractere para inicialização dos exemplos. Um caractere aleatório será usado se definirmos como null

        Random rng = new Random(12345);

        // Obter um DataSetIterator que vetoriza o texto transformando em algo que podemos usar para treinar nossa rede
        CharacterIterator iter = getTextoIterator(tamanhoMiniBatch,tamanhoExemplo);
        int nOut = iter.totalOutcomes();

        // Configuração inicial da rede:
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .learningRate(0.1)
                .seed(12345)
                .regularization(true)
                .l2(0.001)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.RMSPROP)
                .list()
                .layer(0, new GravesLSTM.Builder()
                        .nIn(iter.inputColumns())
                        .nOut(tamanhoCamadaLstm)
                        .activation(Activation.TANH)
                        .build())
                .layer(1, new GravesLSTM.Builder()
                        .nIn(tamanhoCamadaLstm)
                        .nOut(tamanhoCamadaLstm)
                        .activation(Activation.TANH)
                        .build())
                .layer(2, new RnnOutputLayer.Builder(LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(tamanhoCamadaLstm)
                        .nOut(nOut)
                        .build())
                .backpropType(BackpropType.TruncatedBPTT)
                .tBPTTForwardLength(tbpttTamanho)
                .tBPTTBackwardLength(tbpttTamanho)
                .pretrain(false)
                .backprop(true)
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        // Imprime o número de parâmetros da rede inteira e de cada camada
        Layer[] camadas = net.getLayers();
        int totalNumParams = 0;
        for( int i=0; i<camadas.length; i++ ){
            int nParams = camadas[i].numParams();
            System.out.println("Número de parâmetros na camada " + i + ": " + nParams);
            totalNumParams += nParams;
        }
        System.out.println("Total de parâmetros da rede: " + totalNumParams);

        // Realiza o treinamento e, depois, gera e imprime exemplo fazendo uso da rede
        int miniBatchN = 0;
        for( int i=0; i<numEpochs; i++ ){
            while(iter.hasNext()){
                DataSet ds = iter.next();
                net.fit(ds);
                if(++miniBatchN % gerarExemplosACadaNMiniBatchs == 0){
                    System.out.println("--------------------");
                    System.out.println("Completo " + miniBatchN + " minibatches de tamanho "
                            + tamanhoMiniBatch + "x" + tamanhoExemplo + " caracteres" );
                    System.out.println("Obtendo exemplos da rede dada a inicialização \"" +
                            (caracterInicialGerar == null ? "" : caracterInicialGerar) + "\"");
                    String[] exemplos = gerarCaracteresUsandoARede(
                            caracterInicialGerar,net,iter,rng,nCaracteresExemplo,nExemplosGerar);
                    for( int j=0; j<exemplos.length; j++ ){
                        System.out.println("----- Exemplo " + j + " -----");
                        System.out.println(exemplos[j]);
                        System.out.println();
                    }
                }
            }

            iter.reset();	// Reseta o iterador para o próximo epoch
        }

        System.out.println("\n\nExamplo completo");
    }


    public static CharacterIterator getTextoIterator(int tamanhoMiniBatch, int tamanhoSequencia) throws Exception{
        // Localização do arquivo txt
        String localizacaoArquivo = System.getProperty("user.dir") + "/src/main/resources/MachadoAssis.txt";

        // Filtra os caracteres inválidos deixando, apenas, os válidos
        char[] caracteresValidos = CharacterIterator.getMinimalCharacterSet();
        return new CharacterIterator(localizacaoArquivo, Charset.forName("ISO-8859-1"),
                tamanhoMiniBatch, tamanhoSequencia, caracteresValidos, new Random(12345));
    }


    private static String[] gerarCaracteresUsandoARede(String inicializacao, MultiLayerNetwork net,
                                                       CharacterIterator iter,
                                                       Random rng, int nCaracteresExemplo, int numeroExemplos ){
        // Define o caracter inicial. Se não houver caracter inicial, usa um caracter aleatório
        if( inicializacao == null ){
            inicializacao = String.valueOf(iter.getRandomCharacter());
        }

        // Cria a entrada para a inicialização
        INDArray inicializacaoEntrada = Nd4j.zeros(numeroExemplos, iter.inputColumns(), inicializacao.length());
        char[] init = inicializacao.toCharArray();
        for( int i=0; i<init.length; i++ ){
            int idx = iter.convertCharacterToIndex(init[i]);
            for( int j=0; j<numeroExemplos; j++ ){
                inicializacaoEntrada.putScalar(new int[]{j,idx,i}, 1.0f);
            }
        }

        StringBuilder[] sb = new StringBuilder[numeroExemplos];
        for( int i=0; i<numeroExemplos; i++ )
            sb[i] = new StringBuilder(inicializacao);

        // Gera exemplos usando a rede (e alimenta a rede de volta com os exemplos gerados) um caracter por vez
        net.rnnClearPreviousState();
        INDArray output = net.rnnTimeStep(inicializacaoEntrada);
        output = output.tensorAlongDimension(output.size(2)-1,1,0);	// Pega o último passo de saída

        // Para cada caracter a ser gerado de exemplo
        for( int i=0; i<nCaracteresExemplo; i++ ){
            // Configura o próximo input (single time step) a partir de exemplos do último output
            INDArray nextInput = Nd4j.zeros(numeroExemplos,iter.inputColumns());
            for( int s=0; s<numeroExemplos; s++ ){
                double[] outputProbDistribution = new double[iter.totalOutcomes()];
                for( int j=0; j<outputProbDistribution.length; j++ ) outputProbDistribution[j] = output.getDouble(s,j);
                int sampledCharacterIdx = sampleFromDistribution(outputProbDistribution,rng);

                nextInput.putScalar(new int[]{s,sampledCharacterIdx}, 1.0f);
                sb[s].append(iter.convertIndexToCharacter(sampledCharacterIdx));
            }

            output = net.rnnTimeStep(nextInput);	// Faz uma passada para frente
        }

        String[] out = new String[numeroExemplos];
        for( int i=0; i<numeroExemplos; i++ ) out[i] = sb[i].toString();
        return out;
    }


    public static int sampleFromDistribution( double[] distribution, Random rng ){
        double d = 0.0;
        double sum = 0.0;
        for( int t=0; t<10; t++ ) {
            d = rng.nextDouble();
            sum = 0.0;
            for( int i=0; i<distribution.length; i++ ){
                sum += distribution[i];
                if( d <= sum ) return i;
            }

        }
        throw new IllegalArgumentException("Distribution is invalid? d="+d+", sum="+sum);
    }
}