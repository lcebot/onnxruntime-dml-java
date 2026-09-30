// Checks a build of ONNX Runtime Java with DirectML from .github/workflows/build.yml.
//
// Run with the JDK source launcher (JDK 17 or newer), the native folder of the release zip and the jar on the class path:
//   java -Donnxruntime.native.path=<zip>\native -cp <zip>\onnxruntime-<version>.jar smoke\OrtSmoke.java [model.onnx] [--dml-optional]
//
// Without a model it only checks that the native libraries load and list the DirectML provider. With a model
// it runs the model once on the CPU and once on DirectML with the same random inputs and prints the largest
// difference per output. Batch dimensions (the dynamic ones) are set to 2, int64 inputs to 4 and bool
// inputs to true, which suits models whose int64 inputs are small indices or counts. --dml-optional reports a DirectML failure without failing, for CI runners that
// have no GPU.
import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtProvider;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

public class OrtSmoke {
    public static void main(String[] args) throws Exception {
        String model = null;
        boolean dmlOptional = false;
        for (String a : args) {
            if (a.equals("--dml-optional")) dmlOptional = true;
            else model = a;
        }
        if (System.getProperty("onnxruntime.native.path") == null) {
            // Without it the loader extracts onnxruntime.dll to a temp folder that has no DirectML.dll, and
            // Windows may then pick the older DirectML.dll in System32.
            fail("set -Donnxruntime.native.path to the folder that holds onnxruntime.dll and DirectML.dll");
        }
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        EnumSet<OrtProvider> providers = OrtEnvironment.getAvailableProviders();
        System.out.println("providers " + providers);
        if (!providers.contains(OrtProvider.DIRECT_ML)) fail("the DirectML provider is missing from this build");
        if (model == null) return;

        Map<String, float[]> cpu;
        try (OrtSession.SessionOptions so = new OrtSession.SessionOptions();
             OrtSession s = env.createSession(model, so)) {
            cpu = runOnce(env, s, "cpu");
        }
        Map<String, float[]> dml;
        try (OrtSession.SessionOptions so = new OrtSession.SessionOptions()) {
            // The DirectML provider does not support memory patterns or parallel execution.
            so.setMemoryPatternOptimization(false);
            so.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL);
            so.addDirectML(0);
            try (OrtSession s = env.createSession(model, so)) {
                runOnce(env, s, "dml, first run");
                dml = runOnce(env, s, "dml");
            }
        } catch (OrtException e) {
            if (!dmlOptional) throw e;
            System.out.println("DirectML failed (allowed by --dml-optional): " + e.getMessage());
            return;
        }
        for (String name : cpu.keySet()) {
            float[] a = cpu.get(name), b = dml.get(name);
            double diff = 0, ref = 0;
            for (int i = 0; i < a.length; i++) {
                diff = Math.max(diff, Math.abs(a[i] - b[i]));
                ref = Math.max(ref, Math.abs(a[i]));
            }
            System.out.printf("  %s: %d values, max abs diff %.3g, ref max abs %.3g%n", name, a.length, diff, ref);
        }
    }

    // Returns every float output, keyed by name. The inputs come from a fixed seed so both runs see the same data.
    static Map<String, float[]> runOnce(OrtEnvironment env, OrtSession s, String label) throws OrtException {
        Random rnd = new Random(1);
        Map<String, OnnxTensor> inputs = new LinkedHashMap<>();
        try {
            for (Map.Entry<String, NodeInfo> e : s.getInputInfo().entrySet()) {
                TensorInfo info = (TensorInfo) e.getValue().getInfo();
                long[] shape = info.getShape().clone();
                for (int i = 0; i < shape.length; i++) if (shape[i] < 0) shape[i] = 2;
                int n = (int) Arrays.stream(shape).reduce(1, (x, y) -> x * y);
                OnnxTensor t;
                if (info.type == OnnxJavaType.FLOAT) {
                    float[] v = new float[n];
                    for (int i = 0; i < n; i++) v[i] = (float) (rnd.nextGaussian() * 0.5);
                    t = OnnxTensor.createTensor(env, FloatBuffer.wrap(v), shape);
                } else if (info.type == OnnxJavaType.INT64) {
                    long[] v = new long[n];
                    Arrays.fill(v, 4);
                    t = OnnxTensor.createTensor(env, LongBuffer.wrap(v), shape);
                } else if (info.type == OnnxJavaType.BOOL) {
                    byte[] v = new byte[n];
                    Arrays.fill(v, (byte) 1);
                    t = OnnxTensor.createTensor(env, ByteBuffer.wrap(v), shape, OnnxJavaType.BOOL);
                } else {
                    throw new IllegalStateException("unsupported input type " + info.type + " of " + e.getKey());
                }
                inputs.put(e.getKey(), t);
            }
            long t0 = System.nanoTime();
            Map<String, float[]> out = new LinkedHashMap<>();
            try (OrtSession.Result r = s.run(inputs)) {
                for (Map.Entry<String, OnnxValue> e : r) {
                    if (e.getValue() instanceof OnnxTensor t && t.getInfo().type == OnnxJavaType.FLOAT) {
                        FloatBuffer fb = t.getFloatBuffer();
                        float[] v = new float[fb.remaining()];
                        fb.get(v);
                        out.put(e.getKey(), v);
                    }
                }
            }
            System.out.printf("%s: %.1f ms%n", label, (System.nanoTime() - t0) / 1e6);
            return out;
        } finally {
            for (OnnxTensor t : inputs.values()) t.close();
        }
    }

    static void fail(String msg) {
        System.err.println("FAILED: " + msg);
        System.exit(1);
    }
}
