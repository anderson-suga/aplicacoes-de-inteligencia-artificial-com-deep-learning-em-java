package com.exemplo;

import org.nd4j.linalg.api.ndarray.INDArray;

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
}
