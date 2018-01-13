package com.exemplo;

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
}
