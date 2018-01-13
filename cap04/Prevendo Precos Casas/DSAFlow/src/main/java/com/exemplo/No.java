package com.exemplo;
/**
 * Created by Higor Eduardo
 */

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.util.ArrayList;
import java.util.HashMap;

public abstract class No {
    // Nós de entrada e nós de saída desse nó
    protected ArrayList<No> nos_entrada;
    protected ArrayList<No> nos_saida;

    // Nome desse nó
    String nome;

    // Valor desse nó
    protected INDArray valor;
    // Gradientes do nó
    protected HashMap<No, INDArray> gradientes;

    public No(String nome, ArrayList<No> nos_entrada){
        // Inicializa nome e nós de entrada
        this.nome = nome;
        this.nos_entrada = nos_entrada;

        // Define nó como sendo inicialmente zero
        this.valor = Nd4j.zeros(1);
        this.gradientes = new HashMap<No, INDArray>();
        this.nos_saida = new ArrayList<No>();
        // Para cada nó de saída...
        for(No n: nos_entrada){
            // Configura esse nó como sendo um nó de entrada
            n.nos_saida.add(this);
        }

    }


    public No(String nome){
        this.nome = nome;
        this.valor = Nd4j.zeros(1);
        this.gradientes = new HashMap<No, INDArray>();
        this.nos_saida = new ArrayList<No>();
    }

    // Método abstrato foward
    public abstract void foward();
    // Método abstrato backward
    public abstract void backward();
}
