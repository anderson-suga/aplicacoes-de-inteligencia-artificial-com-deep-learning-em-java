import org.tensorflow.*;

public class Operacoes {
    public static void main(String args[]){
        try (Graph g = new Graph()) {
            GraphBuilder b = new GraphBuilder(g);

            final float x = 10f;
            final float y = 4f;
            final float z = 2f;
            final Output input = b.constant("input", 5f);
            final Output output =
                    b.div(
                            b.sub(
                                    b.constant("x", x),
                                    b.constant("y", y)
                            ),
                            b.constant("z", z)
                    );

            try (Session s = new Session(g)) {
                System.out.println(
                        s.runner().fetch(output.op().name()).run().get(0).floatValue()
		        );
            }
        }

    }
    static class GraphBuilder {
        GraphBuilder(Graph g) {
            this.g = g;
        }

        Output div(Output x, Output y) {
            return binaryOp("Div", x, y);
        }

        Output sub(Output x, Output y) {
            return binaryOp("Sub", x, y);
        }
        
        Output constant(String name, Object value) {
            try (Tensor t = Tensor.create(value)) {
                return g.opBuilder("Const", name)
                        .setAttr("dtype", t.dataType())
                        .setAttr("value", t)
                        .build()
                        .output(0);
            }
        }

        private Output binaryOp(String type, Output in1, Output in2) {
            return g.opBuilder(type, type).addInput(in1).addInput(in2).build().output(0);
        }

        private Graph g;
    }
}
