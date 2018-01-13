package com.exemplo;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * Created by Higor Eduardo
 */
public class Linear extends No{
	// Nó do tipo linear. Recebe entradas, pesos e bias
    public Linear(String nome, No entradas, No pesos, No bias){
    	// Cria arraylist com entradas, pesos e bias e chama o construtor da classe mãe
        super(nome, new ArrayList<No>(Arrays.asList(entradas, pesos, bias)));
    }

    // Método foward, faz multiplicação de matrizes dos valores dos nós de entrada
    // pelos valores dos nós dos pesos, soma com o bias e armazena o resultado em valor.

    public void foward() {
    	// Captura os valores de entradas, pesos e bias
        INDArray entradas = this.nos_entrada.get(0).valor;
        INDArray pesos = this.nos_entrada.get(1).valor;
        INDArray bias = this.nos_entrada.get(2).valor;
        // Faz multiplicação de matrizes das entradas pelos pesos, soma o bias
        // E armazena o resultado no valor do nó
        this.valor =  entradas.mmul(pesos).addRowVector(bias);
    }
}
