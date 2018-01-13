package com.exemplo;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Arrays;

import static org.nd4j.linalg.ops.transforms.Transforms.sigmoid;

/**
 * Created by Higor Eduardo
 */
public class Sigmoide extends No{
    // Nó do tipo sigmoide. Recebe apenas um valor para calcular seu sigmoide
    public Sigmoide(String nome, No n){
        super(nome, new ArrayList<No>(Arrays.asList(n)));
    }

    // Método auxiliar para calcular um sigmoide
    private INDArray _sigmoide(INDArray h){
        return sigmoid(h);
    }
    // Calcula o sigmoide do valor do nó de entrada e armazena no valor do nó
    public void foward() {
        // Captura o valor do nó de entrada e armazena em h
        INDArray h = this.nos_entrada.get(0).valor;
        // Armazena o valor de sigmoide de h e armazena no valor do nó atual.
        this.valor = this._sigmoide(h);
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
            // Captura na entrada do nó atual o valor de h
            INDArray h = this.nos_entrada.get(0).valor;
            // Cria vetor de uns que será usado para calcular o gradiente
            INDArray vetor_uns = Nd4j.ones(h.shape());
            // Calcula o gradiente multiplicando o valor de grad_cost pelo valor do gradiente da função
            // sigmoide definido por sigmoide vezes (1 menos sigmoide) e armazena em g
            INDArray g = grad_cost.mul(this._sigmoide(h).mul(vetor_uns.sub(this._sigmoide(h))));

            // Calcula o valor do gradiente do nó de entrada somando ao seu valor atual
            // o valor de g
            this.gradientes.get(this.nos_entrada.get(0)).addi(g);
        }
    }
}
