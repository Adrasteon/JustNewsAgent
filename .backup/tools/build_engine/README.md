---
title: Build engine scaffold
description: Auto-generated description for Build engine scaffold
tags: [documentation]
status: current
last_updated: 2025-09-12
---

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

## See also

- Technical Architecture: markdown_docs/TECHNICAL_ARCHITECTURE.md
- Documentation Catalogue: docs/DOCUMENTATION_CATALOGUE.md

