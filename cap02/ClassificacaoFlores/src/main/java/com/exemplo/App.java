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

        String localizacaoArquivo = "./iris.csv";

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
        for(int i=0; i<w.length; i++){
            for (int j=0; j<w[i].length; j++){
                // Na coluna de índice 4...
                if(j == 4){
                    // Se for "Iris-setosa", armazena 0.0
                    if(w[i][j].equals("Iris-setosa")) {
                        array[i][j] = 0.0;
                    }
                    // Se não, armazena 1.0
                    else {
                        array[i][j] = 1.0;
                    }
                } else
                array[i][j] = Double.valueOf(w[i][j]);
            }
        }


        // Cria objeto INDArray com todos os valores
        INDArray array_valores = Nd4j.create(array);
        System.out.println(array_valores);

        // Cria neurônio com 4 entradas
        Neuronio meuneuronio = new Neuronio(4);

        // Armazena em x o valor das entradas normalizadas
        INDArray x = normalizeZeroMeanAndUnitVariance(array_valores.getColumns(0, 1, 2, 3));
        // Armazena em y o valor das saídas
        INDArray y = array_valores.getColumns(4);
        // Treina o neurônio
        meuneuronio.treinaNeuronio(x, y, 0.5, 10);

    }

    // Método auxiliar do parser de csv
    public static Reader getReader(String csvFile) throws FileNotFoundException, UnsupportedEncodingException {
        return new InputStreamReader(new FileInputStream(new File(csvFile)), "UTF-8");
    }
}