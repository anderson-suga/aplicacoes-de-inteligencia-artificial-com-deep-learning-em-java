package com.exemplo;

/**
 * Created by Higor Eduardo
 */
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.Arrays;


public class Add extends No {

	// Nó do tipo add, recebe dois nós de entrada
    public Add(String nome, No x, No y){
    	// Cria um arraylist com os nós que recebeu e chama o construtor
        super(nome, new ArrayList<No>(Arrays.asList(x, y)));
    }

    // Método foward - captura os valores dos nós de entrada e soma
    public void foward(){
    	// Armazena os valores dos nós de entrada
        INDArray valor_x = this.nos_entrada.get(0).valor;
        INDArray valor_y = this.nos_entrada.get(1).valor;
        
        // Soma os valores dos nós de entrada e guarda o resultado no valor do nó
        this.valor = valor_x.add(valor_y);
    }

}
