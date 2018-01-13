package com.exemplo;

import com.univocity.parsers.csv.CsvParser;
import com.univocity.parsers.csv.CsvParserSettings;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.util.*;

import static org.nd4j.linalg.ops.transforms.Transforms.normalizeZeroMeanAndUnitVariance;


public class App {

    public static void main(String[] args) throws FileNotFoundException, UnsupportedEncodingException {
        

        // Armazena o local do arquivo .csv
        String localizacaoArquivo = "./Boston.csv";

        // Configura o parser de csv
        CsvParserSettings settings = new CsvParserSettings();
        settings.getFormat().setLineSeparator("\n");
        CsvParser parser = new CsvParser(settings);

        // Extrai os valores do arquivo .csv
        List<String[]> allRows = parser.parseAll(getReader(localizacaoArquivo));

        // Converte para matriz de strings
        String[][] w = allRows.toArray(new String[][] {});

        // Converte para matriz de double
        double array [][] = new double[w.length][w[0].length];

        for (int i = 0; i < array.length; i++) {
            for(int j=0; j < array[0].length; j++)
            array[i][j] = Double.parseDouble(w[i][j]);
        }

        // Cria objeto INDArray com todos os valores
        INDArray array_valores = Nd4j.create(array);

        // Armazena os valores de X normalizados
        INDArray X_ = normalizeZeroMeanAndUnitVariance(array_valores.getColumns(1, 2, 3, 4, 5, 6, 7, 8,
                9, 10, 11, 12, 13));


        // Armazena o valor da saída
        INDArray y_ = array_valores.getColumns(14);
        // Armazena a quantidade de features
        int n_features = X_.shape()[1];
        // Define a quantidade de neurônios na camada oculta
        int n_hidden = 20;

        // Inicializa os pesos W1 com valor aleatórios
        INDArray W1_ = Nd4j.rand(n_features, n_hidden);
        // Inicializa o bias b1 com zeros
        INDArray b1_ = Nd4j.zeros(n_hidden);
        // Inicializa os pesos W2 com valor aleatórios
        INDArray W2_ = Nd4j.randn(n_hidden, 1);
        // Inicializa o bias b2 com zeros
        INDArray b2_ = Nd4j.zeros(1);

        // Cria os nós da rede
        Entrada X = new Entrada("X");
        Entrada y = new Entrada("y");
        Entrada W1 = new Entrada("W1");
        Entrada W2 = new Entrada("W2");
        Entrada b1 = new Entrada("b1");
        Entrada b2 = new Entrada("b2");
        Linear l1 = new Linear("l1", X, W1, b1);
        Sigmoide s1 = new Sigmoide("s1", l1);
        Linear l2 = new Linear("l2", s1, W2, b2);
        MSE custo = new MSE("MSE", y, l2);

        // Cria o feed_dict q relaciona os nós de entrada aos seus valores
        Map<No, INDArray> feed_dict = new HashMap<No, INDArray>();
        feed_dict.put(X, X_);
        feed_dict.put(y, y_);
        feed_dict.put(W1, W1_);
        feed_dict.put(b1, b1_);
        feed_dict.put(W2, W2_);
        feed_dict.put(b2, b2_);

        // Define o número de epochs
        int epochs = 10000;
        // Armazena a quantidade de exemplos
        int m = X_.shape()[0];
        // Cria o grafo da rede ordenando os nós criados
        ArrayList<No> grafo = RedeNeural.ordenacao(feed_dict);
        // Define quais nós serão treináveis
        No[] treinaveis = {W1, b1, W2, b2};
        // Imprime o grafo
        System.out.println(grafo);
        // Imprime o total de exemplos
        System.out.println("Número total de exemplos: " + m);

        // Para cada epochs...
        for(int i=0; i<epochs; i++){
            // Faz passada para frente e para trás
            RedeNeural.passada_frente_tras(grafo);
            // Atualiza os valores dos treinavéis usando gradiente_update
            RedeNeural.gradiente_update(treinaveis, 0.1);
            // Calcula a média da perda total do epoch
            System.out.println("Epoch: " + i + " Erro: " + custo.valor);
        }
    }


    // Método auxiliar do parser de csv
    public static Reader getReader(String csvFile) throws FileNotFoundException,
            UnsupportedEncodingException {
        return new InputStreamReader(new FileInputStream(new File(csvFile)), "UTF-8");
    }

}
