package com.exemplo;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class App 
{
    public static void main( String[] args ) {
        // Neurônio AND
        Neuronio and = new Neuronio(2);
        double[][] entradas = {{0,0},
                {0,1},
                {1,0},
                {1,1}};
        double[][] saidas = {{0},{0},{0},{1}};
        INDArray x = Nd4j.create(entradas);
        INDArray y = Nd4j.create(saidas);
        and.treinaNeuronio(x, y, 0.5, 100);

        // Neurônio OR
        Neuronio or = new Neuronio(2);
        entradas = new double[][]{{0, 0},
                {0, 1},
                {1, 0},
                {1, 1}};
        saidas = new double[][]{{0},{1},{1},{1}};
        x = Nd4j.create(entradas);
        y = Nd4j.create(saidas);
        or.treinaNeuronio(x, y, 0.5, 100);

        // Neurônio NOT
        Neuronio not = new Neuronio(1);
        entradas = new double[][]{{0},
                {1}};
        saidas = new double[][]{{1},{0}};
        x = Nd4j.create(entradas);
        y = Nd4j.create(saidas);
        not.treinaNeuronio(x, y, 0.5, 100);

        // Neurônio XOR - Não treina
        Neuronio xor = new Neuronio(2);
        entradas = new double[][]{{0, 0},
                {0, 1},
                {1, 0},
                {1, 1}};
        saidas = new double[][]{{0},{1},{1},{0}};
        x = Nd4j.create(entradas);
        y = Nd4j.create(saidas);
        xor.treinaNeuronio(x, y, 0.5, 100);


        // Entradas para testes
        entradas = new double[][]{{0, 0},
                {0, 1},
                {1, 0},
                {1, 1}};
        x = Nd4j.create(entradas);

        // Testa AND
        System.out.println("Resultados do AND: " + and.calculaResultado(x));
        // Testa OR
        System.out.println("Resultados do OR: " + or.calculaResultado(x));
        // Testa XOR
        System.out.println("Resultados do XOR: " + xor.calculaResultado(x));

        // Entradas para testar NOT
        entradas = new double[][]{{0},
                {1}};
        x = Nd4j.create(entradas);

        // Testa NOT
        System.out.println("Resultados do NOT: " + not.calculaResultado(x));


    }
}
