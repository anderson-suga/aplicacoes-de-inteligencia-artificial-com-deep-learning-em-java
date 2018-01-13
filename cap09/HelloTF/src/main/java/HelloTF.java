import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;

public class HelloTF {
    public static void main(String[] args) throws Exception {
        try (Graph g = new Graph()) {
            final String value = "Olá de tensorflow versão " + TensorFlow.version();

            // Construindo um grafo computacional com uma única operação, uma constante chamada MinhaConstante
            // com o valor value

            try (Tensor t = Tensor.create(value.getBytes("UTF-8"))) {
                g.opBuilder("Const", "MinhaConstante").setAttr("dtype", t.dataType())
                        .setAttr("value", t).build();
            }

            // Executa a operação MinhaConstante em uma Sessão
            try (Session s = new Session(g);
                 Tensor output = s.runner().fetch("MinhaConstante").run().get(0))
            {
                System.out.println(new String(output.bytesValue(), "UTF-8"));
            }
        }
    }
}