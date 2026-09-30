# onnxruntime-dml-java

ONNX Runtime for Java with the DirectML execution provider, for Windows x64. DirectML runs models on any DirectX 12 GPU (NVIDIA, AMD, Intel) without a CUDA install.

Maven Central only has the CPU (`com.microsoft.onnxruntime:onnxruntime`) and CUDA (`onnxruntime_gpu`) builds of the Java binding. This repository builds the official ONNX Runtime source with `--use_dml --build_java` in GitHub Actions and publishes the result as a release. Nothing in ONNX Runtime is patched.

## Release contents

Each release `v<version>` matches ONNX Runtime `v<version>` and has one asset, `onnxruntime-dml-java-<version>-win-x64.zip`:

```
onnxruntime-<version>.jar          Java classes, without native libraries
native/
  onnxruntime.dll
  onnxruntime4j_jni.dll
  onnxruntime_providers_shared.dll   when the build produces it
  DirectML.dll                       from the Microsoft.AI.DirectML NuGet package the build restores
licenses/                          ONNX Runtime and DirectML license texts
VERSIONS.txt                       ONNX Runtime and DirectML versions
SHA256SUMS.txt                     SHA-256 of every file above
```

The SHA-256 of the zip is in the release notes. A published release is never replaced, so the hash can be pinned.

## Usage

Put the jar on the class path and keep the dlls of `native/` in one folder. Point the loader at that folder:

```
java -Donnxruntime.native.path=C:\path\to\native -cp onnxruntime-<version>.jar;... Main
```

The folder matters: `onnxruntime.dll` loads `DirectML.dll` from its own folder. The official jar extracts its native libraries to a temp folder that has no `DirectML.dll`, and Windows may then load the older `DirectML.dll` in System32, which this build does not support. The jar here has no native libraries, so always set the property: without it, Java searches `java.library.path` and PATH, and may pick up another `onnxruntime.dll` (some Windows versions ship one in System32).

Session options for DirectML:

```kotlin
OrtSession.SessionOptions().apply {
    // DirectML supports neither memory patterns nor parallel execution.
    setMemoryPatternOptimization(false)
    setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
    addDirectML(0)                     // adapter 0 is the default GPU
}
```

Operators DirectML lacks fall back to the CPU inside the same session. Long single operators (a large STFT, for example) can exceed the Windows GPU timeout and reset the device (`DXGI_ERROR_DEVICE_HUNG`); run such graphs in a CPU session.

## Smoke test

`smoke/OrtSmoke.java` checks that DirectML is listed, then runs a model once on the CPU and once on DirectML with the same random inputs and prints the largest difference per output. With JDK 17 or newer:

```
java -Donnxruntime.native.path=<unzipped>\native -cp <unzipped>\onnxruntime-<version>.jar smoke\OrtSmoke.java model.onnx
```

Dynamic dimensions are set to 2, float inputs are random, int64 inputs are 4 and bool inputs are true. CI runs it on a small model with `--dml-optional`, because the runner has no GPU.

## Building

Run the `build` workflow by hand (Actions, build, Run workflow) with an ONNX Runtime version and a Visual Studio version. It takes one to three hours on a hosted runner and publishes release `v<version>`. To rebuild a version, delete its release and tag first.

- Visual Studio 2026 (the `windows-2025` runner) is the default. ONNX Runtime's own CI compiles with Visual Studio 2022, so choose 2022 (the `windows-2022` runner) if 2026 fails to compile.
- JDK 21 builds the Java binding: the Gradle wrapper that ONNX Runtime pins (8.7) runs on JDK 21 at most. The jar targets Java 8 bytecode, so it runs on any newer JDK.
- The official DirectML packages (`onnxruntime-directml` on PyPI, `Microsoft.ML.OnnxRuntime.DirectML` on NuGet) stop at 1.24.4, because DirectML moved to sustained engineering. The DirectML provider is still in the ONNX Runtime source and still receives fixes, which only a source build like this one delivers.

## License

The workflow and the smoke test are under the MIT License (see `LICENSE`). The released binaries keep their own licenses: ONNX Runtime is under the MIT License, and DirectML is under the Microsoft DirectML license included in the zip.
