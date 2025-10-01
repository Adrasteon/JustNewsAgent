#!/usr/bin/env python3
"""
Safe TensorRT engine build stub for development and CI.

This script will do one of the following depending on the environment:
- If the native compiler (`agents/analyst/native_tensorrt_compiler.py`) is importable and TensorRT is available,
  it will call the real compiler (useful on GPU-enabled developer machines).
- Otherwise it will create small marker `.engine` files and matching `.json` metadata in
  `agents/analyst/tensorrt_engines/` to emulate a successful build.

The marker files are text placeholders and do not require GPU or TensorRT.
"""

import argparse
import json
import time
from pathlib import Path

ENGINES_DIR = Path(__file__).parent.parent / "agents" / "analyst" / "tensorrt_engines"
ENGINES_DIR.mkdir(parents=True, exist_ok=True)


def create_marker(engine_name: str, task: str, precision: str = "fp16"):
    engine_path = ENGINES_DIR / engine_name
    metadata_path = engine_path.with_suffix(".json")

    with open(engine_path, "w", encoding="utf-8") as f:
        f.write(f"Marker TensorRT engine for {task}\n")
        f.write(f"Created: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Precision: {precision}\n")

    metadata = {
        "task": task,
        "precision": precision,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Created marker engine: {engine_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-only", action="store_true", help="Only check environment capability"
    )
    parser.add_argument(
        "--build-markers", action="store_true", help="Create marker engine files (safe)"
    )
    parser.add_argument(
        "--precision",
        default="fp16",
        choices=["fp32", "fp16", "int8"],
        help="Precision for marker files",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Record calibration requested for int8 markers",
    )
    parser.add_argument(
        "--calib-data",
        type=str,
        help="Path to calibration dataset to record in metadata",
    )
    args = parser.parse_args()

    if args.check_only:
        try:
            import tensorrt  # noqa: F401
            import torch  # noqa: F401

            print("TensorRT and CUDA appear to be available on this machine.")
            print(
                "You can run the real compiler in agents/analyst/native_tensorrt_compiler.py"
            )
        except Exception:
            print(
                "TensorRT/CUDA not available. Use --build-markers to create safe marker files."
            )
        return

    if args.build_markers:
        # Create marker engines for sentiment and bias with requested precision
        create_marker(
            "native_sentiment_roberta.engine", "sentiment", precision=args.precision
        )
        create_marker("native_bias_bert.engine", "bias", precision=args.precision)
        # If int8 calibration requested, write calibration metadata
        if args.precision == "int8" and args.calibrate:
            # Update the JSON metadata files to include calibration hints
            for name in ("native_sentiment_roberta", "native_bias_bert"):
                meta_path = ENGINES_DIR / f"{name}.json"
                try:
                    with open(meta_path, encoding="utf-8") as f:
                        meta = json.load(f)
                except Exception:
                    meta = {}
                meta["calibrated"] = False
                meta["calib_data"] = args.calib_data or None
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
        print("Marker engine creation complete.")
        return

    # Default behavior: try to run native compiler if present
    try:
        from agents.analyst.native_tensorrt_compiler import NativeTensorRTCompiler

        compiler = NativeTensorRTCompiler()
        print("Native compiler initialized. Starting compilation...")
        results = compiler.compile_all_models()
        print("Compilation results:", results)
    except Exception as e:
        print("Native compiler not available or failed:", e)
        print("Falling back to creating marker engine files...")
        create_marker("native_sentiment_roberta.engine", "sentiment", precision="fp16")
        create_marker("native_bias_bert.engine", "bias", precision="fp16")
        print("Marker engine creation complete.")


if __name__ == "__main__":
    main()
