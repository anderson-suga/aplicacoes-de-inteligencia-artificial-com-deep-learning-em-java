/**
 * Created by Higor Eduardo.
 */


import java.util.Random;

class Perceptron {

    // Vetor de pesos
    double w[];
    // Número de entradas
    int n;

    // Construtor - Recebe o número de entradas e inicializa os pesos
    public Perceptron(int n) {
        this.n = n;
        inicializaPesos(this.n);
    }

    // Inicializa os pesos com valores aleatórios entre 0 e 1
    private void inicializaPesos(int n) {
        Random r = new Random();
        this.w = new double[n+1];
        for(int i=0; i<=n; i++) {
            this.w[i] = r.nextDouble();
        }
    }

    // Treina o perceptron usando exemplos, a taxa de aprendizagem e o número de epochs
    public void treinaPerceptron(int x[][], int y[], double taxaAprendizagem, int epoch) {
        // Para cada epoch...
        for(int i=0; i < epoch; i++) {
            // Para cada exemplo...
            for(int j=0; j < y.length; j++){
                // Passo 1: calcula o resultado
                int y_calc = calculaResultado(x[j]);
                // Passo 2: calcula o erro
                double erro =  y_calc - y[j];
                // Para cada peso...
                for(int k=0; k<=n; k++){
                    // Passo 3: atualiza o peso
                    if(k==0){
                        w[k] = w[k] - erro * taxaAprendizagem;
                    } else {
                        w[k] = w[k] - x[j][k-1] * erro * taxaAprendizagem;
                    }
                }
            }
        }
    }

    // Retorna os valores dos pesos
    public double[] getW() {
        return w;
    }

    // Retorna o número de entradas
    public int getN() {
        return n;
    }

    // Calcula resultado
    public int calculaResultado(int x[]) {
        double h = 0;
        // Calcula h
        for(int i=0; i<=n; i++){
            if(i == 0){
                h = h + w[i];
            } else {
                h = h + x[i-1] * w[i];
            }
        }
        // Retorna o resultado da função de passo
        return stepFunction(h);
    }

    // Função de passo
    private int stepFunction(double h) {
        // Se h for menor que 0, retorna 0
        if(h < 0)
            return 0;
        // Se não, retorna 1
        else
            return 1;
    }
}