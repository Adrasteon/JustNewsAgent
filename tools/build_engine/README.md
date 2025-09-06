# Build engine scaffold

This folder contains a host-native scaffold for building TensorRT engines.

Usage (scaffold):

```bash
python tools/build_engine/build_engine.py --check-only
python tools/build_engine/build_engine.py --build-markers
python tools/build_engine/build_engine.py --build --model sentiment --precision fp16
```

Notes:
- This scaffold tries to call `NativeTensorRTCompiler` when the native toolchain is present.
- It will fallback to marker-engine creation when TensorRT/CUDA isn't available.
- Full engine building requires a GPU host with TensorRT, CUDA, and PyTorch installed.
