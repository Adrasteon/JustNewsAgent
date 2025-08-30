# TensorRT Quickstart (safe, no-GPU stub)

This file explains how to run a safe, developer-friendly stub for the TensorRT engine build process.

Purpose
- Provide a predictable local flow that documents what the native TensorRT compiler would do.
- Avoid requiring GPUs, TensorRT, or CUDA to run a quick "build check" during development.

Files
- `scripts/compile_tensorrt_stub.py` — safe stub that either calls the real compiler (if available) or creates marker engine files to emulate a successful build.

Quick checks
1. Check-only (no changes):

   python scripts/compile_tensorrt_stub.py --check-only

   This verifies whether the real compiler and runtime are importable and reports what would be built.

2. Create marker engines (safe, no GPU required):

   python scripts/compile_tensorrt_stub.py --build-markers

   This creates small marker `.engine` files and matching metadata JSON in `agents/analyst/tensorrt_engines/` so runtime code paths that check for engine artifacts will see them.

Notes
- The stub is intentionally conservative: it will only call the real compiler if the required native packages are present. Otherwise it writes marker files and returns success.
- Use this when running CI jobs or developer checks that must not require GPUs.

Recommended next steps
- Add a CI job that runs `--check-only` to assert environment capability.
- Add unit tests that mock `tensorrt`/`torch` to validate the logic in `native_tensorrt_compiler.py` without hardware.
