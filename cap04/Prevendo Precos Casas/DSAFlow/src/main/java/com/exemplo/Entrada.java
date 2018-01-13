package com.exemplo;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by Higor Eduardo
 */

public class Entrada extends No {

	// Nó de entrada. Não recebe nós de entrada
    public Entrada(String nome) {
        super(nome);
    }

	// Não faz nada em foward. Os valores dos nós de entrada são inicializados 
	// no momento da ordenação da rede    
    public void foward() {
    }

    // Método backward. Calcula os gradientes usando os nós de saída
    public void backward() {
        // Inicializa os gradientes referentes ao próprio nó com zero
        this.gradientes.put(this, Nd4j.zeros(this.valor.shape()));
        // Para cada nó de saída
        for(No n: this.nos_saida){
            // Armazena o gradiente do nó de saída relacionado ao nó atual e armazena em grad_cost
            INDArray grad_cost = n.gradientes.get(this);
            // Adiciona ao gradiente do nó atual o valor de grad_cost
            this.gradientes.get(this).addi(grad_cost);
        }
    }
}
