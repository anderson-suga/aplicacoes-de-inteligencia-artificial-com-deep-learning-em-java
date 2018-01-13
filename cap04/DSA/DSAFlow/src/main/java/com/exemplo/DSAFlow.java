package com.exemplo;

/**
 * Created by Higor Eduardo
 */

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

public class DSAFlow {
	String nome;
	Entrada X;
	No ultimaSaida;
	int pos_oculta;
	Map<No, INDArray> feed_dict;
	ArrayList<No> treinaveis;

	public DSAFlow(String nome){
		this.nome = nome;
		pos_oculta = 0;
		X = new Entrada("X");
		this.ultimaSaida = X;
		feed_dict = new HashMap<No, INDArray>();
		this.treinaveis = new ArrayList<No>();
	}

	public void adicionaCamada(String nome, int f, int h,
		Boolean ativacao) {

		Entrada W = new Entrada("W_" + this.pos_oculta);
		Entrada b = new Entrada("b_" + this.pos_oculta);
		Linear l = new Linear("l_" + pos_oculta, ultimaSaida, W, b);
		if(ativacao){
			Sigmoide s = new Sigmoide("s_" + pos_oculta, l);
			ultimaSaida = s;
		} else {
			ultimaSaida = l;
		}
        this.treinaveis.add(W);
		this.treinaveis.add(b);

		INDArray W_ = Nd4j.rand(f, h);
        INDArray b_ = Nd4j.zeros(h);
        feed_dict.put(W, W_);
        feed_dict.put(b, b_);

		this.pos_oculta = this.pos_oculta + 1;

	}


	public void treinaRede(INDArray X_, INDArray y_, int epochs, double taxa_aprendizagem){
		feed_dict.put(this.X, X_);
		Entrada y = new Entrada("y");
		MSE custo = new MSE("custo", y, ultimaSaida);
		feed_dict.put(y, y_);
        ArrayList<No> grafo = this.ordenacao(feed_dict);
		for(int i=0; i<epochs; i++){
			this.passada_frente_tras(grafo);
			this.gradiente_update(taxa_aprendizagem);
			System.out.println("Epoch: " + i + " Perda: " + custo.valor);
		}
	}

    private void gradiente_update(double taxa_aprendizagem){
        // Itera por cada nó treinável
        for(No t: treinaveis){
            INDArray parcial = t.gradientes.get(t);
            // Subtrai do nó treinável seu gradiente multiplicado pela taxa de aprendizagem
            t.valor.subi(parcial.mul(taxa_aprendizagem));
        }
    }

    public static void passada_frente_tras(ArrayList<No> nos_ordenados) {
        // Para cada nó dentre os nós já ordenados
        for(No n: nos_ordenados){
            // Executa o método foward do nó
            n.foward();
        }
        // Faz iteração entre os nós ordenados de trás para frente
        for(int j = nos_ordenados.size() - 1; j >= 0; j--){
            // Executa o método backward do nó
            nos_ordenados.get(j).backward();
        }
    }

    private ArrayList<No> ordenacao(Map<No, INDArray> feed_dict){
        Set<No> nos_entrada = feed_dict.keySet();
        ArrayList<No> nos = new ArrayList<No>(nos_entrada);
        Map<No, Map<String, HashSet<No>>> G = new HashMap<No, Map<String, HashSet<No>>>();

        while(nos.size() > 0) {
            No n = (No) nos.get(0);
            nos.remove(0);
            if(!G.containsKey(n)) {
                Map<String, HashSet<No>> in_out = new HashMap<String, HashSet<No>>();
                HashSet<No> in = new HashSet<No>();
                HashSet<No> out = new HashSet<No>();
                in_out.put("in", in);
                in_out.put("out", out);
                G.put(n, in_out);
            }

            for(int j = 0; j < n.nos_saida.size(); j++){
                No m = n.nos_saida.get(j);
                if(!G.containsKey(m)){
                    HashMap<String, HashSet<No>> in_out = new HashMap<String, HashSet<No>>();
                    HashSet<No> in = new HashSet<No>();
                    HashSet<No> out = new HashSet<No>();
                    in_out.put("in", in);
                    in_out.put("out", out);
                    G.put(m, in_out);
                }
                (G.get(n).get("out")).add(m);
                (G.get(m).get("in")).add(n);
                nos.add(m);
            }
        }
        HashSet<No> S = new HashSet<No>(nos_entrada);
        ArrayList<No> L = new ArrayList<No>();
        while(S.size() > 0){
            No n = S.iterator().next();
            if(n instanceof Entrada){
                n.valor = feed_dict.get(n);
            }
            L.add(n);
            for(int j=0; j < n.nos_saida.size(); j++){
                No m = n.nos_saida.get(j);
                G.get(n).get("out").remove(m);
                (G.get(m).get("in")).remove(n);
                if(G.get(m).get("in").size() == 0){
                    S.add(m);
                }
            }
            S.remove(n);
        }
        return L;
    }
}