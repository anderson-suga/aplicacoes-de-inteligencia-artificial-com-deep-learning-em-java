package com.exemplo;

/**
 * Created by higoreduardo
 */


import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import static org.nd4j.linalg.ops.transforms.Transforms.*;

public class Neuronio {
    // Pesos do neurônio e Número de entradas do neurônio
    INDArray w;
    int n;

    public Neuronio(int n){

        this.n = n;

        // Inicializa os pesos passando o número de entradas + 1 (bias)
        inicializaPesos(this.n + 1);
    }

    private void inicializaPesos(int n_pesos) {
        // Inicializa os pesos com valores aleatórios
        this.w = Nd4j.rand(n_pesos, 1);
    }


    public void treinaNeuronio(INDArray x, INDArray y, double taxaAprendizagem, int epoch) {
        // Armazena o número de exemplos enviados para treinamento
        int n_examples = x.size(0);
        // Cria a coluna de bias com todos os valores sendo 1
        INDArray x_bias = Nd4j.ones(n_examples, 1);
        INDArray X = Nd4j.hstack(x_bias, x);

        // Para cada epoch...
        for(int i=0; i < epoch; i++) {
            // Encontra o valor de h fazendo uma multiplicação de matrizes entre as entradas e os pesos
            INDArray h = X.mmul(this.w);
            // Encontra o resultado da função de ativação sigmoide
            INDArray sigmoide  = sigmoid(h);
            // Encontra o erro
            INDArray erro = sigmoide.sub(y);
            // Encontra a soma dos erros quadrados
            INDArray soma_erro_quadrado = Nd4j.sum(erro.mul(erro), 0);
            // Cria um vetor de 1s que serão usados para encontrar o gradiente
            INDArray vetor_uns = Nd4j.ones(1, n_examples);
            // Calcula o gradiente: g(h) = f(h) * (1 - f(h))
            INDArray gradiente = sigmoide.mul(vetor_uns.sub(sigmoide));
            // Calcula o termo de atualização multiplicando o gradiente pelo erro
            INDArray termo_atualizacao = gradiente.mul(erro);
            // Calcula a matriz delta_w multiplicando as entradas pelo termo de atualização
            INDArray delta_w = X.mulColumnVector(termo_atualizacao);
            // Calcula o valor de delta
            INDArray delta = Nd4j.sum(delta_w, 0);
            // Multiplica delta pela taxa de aprendizagem
            delta = delta.mul(taxaAprendizagem);
            // Calcula os novos pesos, subtraindo delta dos antigos
            this.w = this.w.sub(delta.transpose());

            // Log
//            if(i%1==0) {
//                System.out.println("Epoch: " + i);
//                System.out.println("Soma dos erros quadrados: " + soma_erro_quadrado);
//                System.out.println("Novos Pesos: " + this.w);
//            }
        }
    }

    public INDArray calculaResultado(INDArray x){
        int n_examples = x.size(0);
        INDArray x_bias = Nd4j.ones(n_examples, 1);
        INDArray X = Nd4j.hstack(x_bias, x);

        // Calcula h
        INDArray h = X.mmul(this.w);
        // Calcula a saída aplicando sigmoide
        INDArray saida  = sigmoid(h);

        // Arredonda para que a saída seja sempre 0 ou 1
        return round(saida);
    }
}

