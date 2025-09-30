#!/usr/bin/env python3
"""GPU-enabled runner for performing ONNX->TensorRT build with optional INT8 calibration.

This script is intended to be run on a GPU host with CUDA, TensorRT and PyCUDA
installed. It wraps `agents.analyst.native_tensorrt_compiler.NativeTensorRTCompiler`
and exposes a simple CLI to run builds and control calibration.

Usage:
  python scripts/gpu_runner.py --precision int8 --calib-data path/to/calib.jsonl --sentiment

Notes:
- Ensure `MODEL_STORE_ROOT` env var is set to the path where ModelStore writes are permitted.
- Run inside the project's Python environment (conda env recommended).
"""

import argparse

from agents.analyst.native_tensorrt_compiler import NativeTensorRTCompiler


def main():
    parser = argparse.ArgumentParser(description='GPU runner for TensorRT builds')
    parser.add_argument('--precision', choices=['fp16', 'int8', 'fp32'], default='fp16')
    parser.add_argument('--calib-data', type=str, default=None, help='Path to calibration dataset (jsonl or text or dir)')
    parser.add_argument('--sentiment', action='store_true')
    parser.add_argument('--bias', action='store_true')
    parser.add_argument('--max-batch-size', type=int, default=100)
    args = parser.parse_args()

    compiler = NativeTensorRTCompiler()
    compiler.optimization_config['precision'] = args.precision
    compiler.optimization_config['max_batch_size'] = args.max_batch_size
    if args.calib_data:
        compiler.optimization_config['calibration_data'] = args.calib_data

    # Run requested builds
    if args.sentiment:
        ok = compiler.compile_sentiment_model()
        print('Sentiment build:', 'OK' if ok else 'FAILED')
    if args.bias:
        ok = compiler.compile_bias_model()
        print('Bias build:', 'OK' if ok else 'FAILED')
    if not args.sentiment and not args.bias:
        results = compiler.compile_all_models()
        print('Build results:', results)


if __name__ == '__main__':
    main()
