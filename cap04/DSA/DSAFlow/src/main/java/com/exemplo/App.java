package com.exemplo;

import com.univocity.parsers.csv.CsvParser;
import com.univocity.parsers.csv.CsvParserSettings;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.util.List;

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
        String[][] w = allRows.toArray(new String[][]{});

        // Converte para matriz de double
        double array[][] = new double[w.length][w[0].length];

        // Cria objeto INDArray com todos os valores
        INDArray array_valores = Nd4j.create(array);

        // Armazena os valores de X normalizados
        INDArray X_ = normalizeZeroMeanAndUnitVariance(array_valores.getColumns(1, 2, 3, 4, 5, 6, 7,
                8, 9, 10, 11, 12, 13));

        // Armazena o valor da saída
        INDArray y_ = array_valores.getColumns(14);
        // Armazena a quantidade de features
        int n_features = X_.shape()[1];
        // Define a quantidade de neurônios na camada oculta
        int n_hidden = 20;

        DSAFlow minhaRede = new DSAFlow("Minha Rede");
        minhaRede.adicionaCamada("Camada1", n_features, n_hidden, true);
        minhaRede.adicionaCamada("CamadaFinal", n_hidden, 1, false);
        minhaRede.treinaRede(X_, y_, 10, 0.05);
    }



    // Método auxiliar do parser de csv
    public static Reader getReader(String csvFile) throws FileNotFoundException,
            UnsupportedEncodingException {
        return new InputStreamReader(new FileInputStream(new File(csvFile)), "UTF-8");
    }

}
