package com.exemplo;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * Created by Higor Eduardo
 */
public class MSE extends No{
    INDArray diff;
    double m;

	// No do tipo MSE. Recebe o nó da saída esperada e o nó da saída calculada
    public MSE(String nome, No y, No a){
        super(nome, new ArrayList<No>(Arrays.asList(y, a)));
    }
    // Calcula a média dos erros quadrados
    public void foward() {

    	// Captura o valor dos nós da saída esperada e da saída calculada e subtrai um do outro
        INDArray y = this.nos_entrada.get(0).valor;
        INDArray a = this.nos_entrada.get(1).valor;
        this.m = this.nos_entrada.get(0).valor.shape()[0];
        this.diff = y.sub(a);
        // Calcula a média dos valores elevados ao quadrado
        this.valor = Nd4j.mean(diff.mul(diff));

    }
    // Método backward. Calcula os gradientes
    public void backward() {
        this.gradientes.put(this.nos_entrada.get(0), this.diff.mul(2/this.m));
        this.gradientes.put(this.nos_entrada.get(1), this.diff.mul(-2/this.m));
    }
}
