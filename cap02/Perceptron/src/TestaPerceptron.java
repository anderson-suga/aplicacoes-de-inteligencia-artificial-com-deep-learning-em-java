/**
 * Created by Higor Eduardo.
 */

public class TestaPerceptron {

    public static void main(String[] args){
        int[][] entradas = {
                {0,0},
                {0,1},
                {1,0},
                {1,1}
                };
        int[] saidas = {0,0,0,1};

        Perceptron and = new Perceptron(2);
        System.out.println("Treinando um AND...");
        and.treinaPerceptron(entradas, saidas, 0.1, 1000);
        System.out.println("Imprimindo os pesos da rede AND...");
        double[] w = and.getW();
        for(int i=0; i< w.length; i++){
            System.out.println(w[i]);
        }
        System.out.println("Testando o AND...");
        System.out.println("Entrada: 0, Entrada 0 = Saída " + and.calculaResultado(new int[]{0, 0}));
        System.out.println("Entrada: 0, Entrada 1 = Saída " + and.calculaResultado(new int[]{0, 1}));
        System.out.println("Entrada: 1, Entrada 0 = Saída " + and.calculaResultado(new int[]{1, 0}));
        System.out.println("Entrada: 1, Entrada 1 = Saída " + and.calculaResultado(new int[]{1, 1}));


        entradas = new int[][]{
                {0, 0},
                {0, 1},
                {1, 0},
                {1, 1}
        };
        saidas = new int[]{0, 1, 1, 1};

        Perceptron or = new Perceptron(2);
        System.out.println("\n============================\n");
        System.out.println("Treinando um OR...");
        or.treinaPerceptron(entradas, saidas, 0.1, 1000);
        System.out.println("Imprimindo os pesos da rede OR...");
        w = or.getW();
        for(int i=0; i< w.length; i++){
            System.out.println(w[i]);
        }
        System.out.println("Testando o OR...");
        System.out.println("Entrada: 0, Entrada 0 = Saída " + or.calculaResultado(new int[]{0, 0}));
        System.out.println("Entrada: 0, Entrada 1 = Saída " + or.calculaResultado(new int[]{0, 1}));
        System.out.println("Entrada: 1, Entrada 0 = Saída " + or.calculaResultado(new int[]{1, 0}));
        System.out.println("Entrada: 1, Entrada 1 = Saída " + or.calculaResultado(new int[]{1, 1}));

        entradas = new int[][]{
                {0},
                {1}
        };
        saidas = new int[]{1, 0};

        Perceptron not = new Perceptron(1);
        System.out.println("\n============================\n");
        System.out.println("Treinando um NOT...");
        not.treinaPerceptron(entradas, saidas, 0.1, 1000);
        System.out.println("Imprimindo os pesos da rede NOT...");
        w = not.getW();
        for(int i=0; i< w.length; i++){
            System.out.println(w[i]);
        }
        System.out.println("Entrada: 0 = " + not.calculaResultado(new int[]{0}));
        System.out.println("Entrada: 1 = " + not.calculaResultado(new int[]{1}));

        entradas = new int[][]{
                {0, 0},
                {0, 1},
                {1, 0},
                {1, 1}
        };
        saidas = new int[]{0, 1, 1, 0};

        Perceptron xor = new Perceptron(2);
        System.out.println("\n============================\n");
        System.out.println("Treinando um XOR...");
        xor.treinaPerceptron(entradas, saidas, 0.1, 1000);
        System.out.println("Imprimindo os pesos da rede XOR...");
        w = xor.getW();
        for(int i=0; i< w.length; i++){
            System.out.println(w[i]);
        }

        System.out.println("Testando o XOR...");
        System.out.println("Entrada: 0, Entrada 0 = Saída " + xor.calculaResultado(new int[]{0, 0}));
        System.out.println("Entrada: 0, Entrada 1 = Saída " + xor.calculaResultado(new int[]{0, 1}));
        System.out.println("Entrada: 1, Entrada 0 = Saída " + xor.calculaResultado(new int[]{1, 0}));
        System.out.println("Entrada: 1, Entrada 1 = Saída " + xor.calculaResultado(new int[]{1, 1}));

    }

}