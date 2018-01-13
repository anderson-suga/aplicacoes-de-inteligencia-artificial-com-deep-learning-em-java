package com.exemplo;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;


public class App {

    public static void main(String[] args) {
        // Criando uma rede que soma dois valores
        // Define nós de entrada x e y
        Entrada x = new Entrada("x");
        Entrada y = new Entrada("y");

        // Define um nó Add, sendo os dois nós do tipo Entrada.
        Add f = new Add("add", x, y);

        // Os valores de x e y serão definidos como 10 e 20, respectivamente.
        Map<No, INDArray> feed_dict = new HashMap<No, INDArray>();
        double[] x_d = {10.0};
        double[] y_d = {20.0};
        INDArray x_v = Nd4j.create(x_d);
        INDArray y_v = Nd4j.create(y_d);

        feed_dict.put(x, x_v);
        feed_dict.put(y, y_v);

        // Ordena os nós.
        ArrayList<No> sorted_nodes = RedeNeural.ordenacao(feed_dict);
        // Imprime os nós ordenados
        System.out.println(sorted_nodes);
        // Calcula a saída da rede usando passada_frente
        INDArray output = RedeNeural.passada_frente(f, sorted_nodes);
        // Imprime o valor da saída da rede
        System.out.println(output);

        // Criando uma rede que faz uma transformação linear
        // Define os nós de entrada X, W e b
        Entrada X = new Entrada("X");
        Entrada W = new Entrada("W");
        Entrada b = new Entrada("b");

        // Define um nó do tipo Linear usando os nós X, W e b como entrada
        Linear l = new Linear("Linear", X, W, b);

        double[][] X_d = {{-1, -2}, {-1, -2}};
        INDArray X_v = Nd4j.create(X_d);
        double[][] W_d = {{2, -3}, {2, -3}};
        INDArray W_v = Nd4j.create(W_d);
        double[] b_d = {-3, -5};
        INDArray b_v = Nd4j.create(b_d);

        feed_dict = new HashMap<No, INDArray>();
        feed_dict.put(X, X_v);
        feed_dict.put(W, W_v);
        feed_dict.put(b, b_v);

        ArrayList<No> grafo = RedeNeural.ordenacao(feed_dict);
        System.out.println(grafo);
        INDArray saida = RedeNeural.passada_frente(l, grafo);
        System.out.println(saida);

        // Testando um nó do tipo MSE
        y = new Entrada("y");
        No a = new Entrada("a");
        No custo = new MSE("custo", y, a);

        double[] y_d_ = {1,2,3};
        INDArray y_v_ = Nd4j.create(y_d_);
        double[] a_d = {4.5, 5, 10};
        INDArray a_v = Nd4j.create(a_d);

        feed_dict = new HashMap<No, INDArray>();
        feed_dict.put(y, y_v_);
        feed_dict.put(a, a_v);

        grafo = RedeNeural.ordenacao(feed_dict);
        System.out.println(grafo);

        saida = RedeNeural.passada_frente(custo, grafo);
        System.out.println(saida);

    }

}
