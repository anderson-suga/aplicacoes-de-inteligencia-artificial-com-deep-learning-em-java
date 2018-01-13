package com.exemplo;

/**
 * Created by Higor Eduardo
 */
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.*;

public class RedeNeural {
    public RedeNeural(){
    }

    // Método que faz a passada para frente
    public static INDArray passada_frente(No no_saida, ArrayList<No> nos_ordenados) {
        // Para cada nó dentre os nós já ordenados
        for(No n: nos_ordenados){
            // Executa o método foward do nó
            n.foward();
        }
        // Retorna o valor do último nó da rede
        return no_saida.valor;
    }

    // Método que ordena os nós usando o algoritmo de Kahn 
    // https://en.wikipedia.org/wiki/Topological_sorting#Kahn.27s_algorithm
    public static ArrayList<No> ordenacao(Map<No, INDArray> feed_dict){
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
