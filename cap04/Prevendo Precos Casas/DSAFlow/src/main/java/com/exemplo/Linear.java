package com.exemplo;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

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

    // Método backward. Calcula os gradientes usando os nós de saída
    public void backward() {
        // Inicializa o gradiente referente a cada nó de entrada com 0.
        for(No n: this.nos_entrada){
            this.gradientes.put(n, Nd4j.zeros(n.valor.shape()));
        }

        // Para cada nó de saída
        for(No n: this.nos_saida) {
            // Armazena o gradiente do nó de saída relacionado ao nó atual e armazena em grad_cost
            INDArray grad_cost = n.gradientes.get(this);
            // Captura nas entradas do nó atual os valores de x, w e bias
            INDArray x = this.nos_entrada.get(0).valor;
            INDArray w = this.nos_entrada.get(1).valor;
            INDArray bias = this.nos_entrada.get(2).valor;

            // Calcula o valor do gradiente do primeiro nó de entrada somando ao seu valor atual
            // o valor da multiplicação de matrizes entre grad_cost e a transposta de w
            this.gradientes.get(this.nos_entrada.get(0)).addi(grad_cost.mmul(w.transpose()));
            // Calcula o valor do gradiente do segundo nó de entrada somando ao seu valor atual
            // o valor da multiplicação de matrizes entre a transposta de x e grad_cost
            this.gradientes.get(this.nos_entrada.get(1)).addi(x.transpose().mmul(grad_cost));
            // Calcula o valor do gradiente do terceiro nó de entrada somando ao seu valor atual
            // o valor da soma dos valores de grad_cost
            this.gradientes.get(this.nos_entrada.get(2)).addi(Nd4j.sum(grad_cost));
        }

    }
}
