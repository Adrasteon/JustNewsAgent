"""Generate parity report artifacts for diagnostics and CI.

This script replays the sample dataset through both adapters and emits
normalized JSON files for each run. When differences are detected a
"diff.json" file is written containing the mismatched IDs and their
respective normalized records.

Exit codes:
- 0: parity OK
- 2: parity differences found (diff.json written)
- 3: unexpected error
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from kafka.tests.parity.replay_runner import run_replay, normalize_store, compare_stores
from kafka.src.agents.adapter_template import MpcAdapter, KafkaAdapter


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate parity artifacts for a sample dataset")
    parser.add_argument("--sample", required=True, help="Path to sample jsonl file")
    parser.add_argument("--out", required=True, help="Output directory for artifacts")
    parser.add_argument("--tolerance", required=False, type=int, default=5, help="Time tolerance in seconds for time-like fields when comparing stores")
    args = parser.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)

    try:
        mpc_adapter = MpcAdapter()
        mpc_store = run_replay(mpc_adapter, args.sample)
        mpc_norm = normalize_store(mpc_store, preserve_time_keys=True)

        kafka_adapter = KafkaAdapter()
        # Ensure the KafkaAdapter will use the real Kafka client if available
        kafka_adapter._use_real = True
        kafka_store = run_replay(kafka_adapter, args.sample)
        kafka_norm = normalize_store(kafka_store, preserve_time_keys=True)

        # Write artifacts
        mpc_path = os.path.join(args.out, "mpc_store.json")
        kafka_path = os.path.join(args.out, "kafka_store.json")
        with open(mpc_path, "w", encoding="utf-8") as fh:
            json.dump(mpc_norm, fh, indent=2, ensure_ascii=False, sort_keys=True)
        with open(kafka_path, "w", encoding="utf-8") as fh:
            json.dump(kafka_norm, fh, indent=2, ensure_ascii=False, sort_keys=True)

        # Compute diffs using tolerance-aware comparator
        diffs = compare_stores(mpc_norm, kafka_norm, time_tolerance_seconds=args.tolerance)

        if diffs:
            diff_path = os.path.join(args.out, "diff.json")
            with open(diff_path, "w", encoding="utf-8") as fh:
                json.dump(diffs, fh, indent=2, ensure_ascii=False, sort_keys=True)
            print(f"Parity differences found, written to {diff_path}")
            return 2

        print("Parity OK: no differences")
        return 0

    except Exception as e:
        print(f"Unexpected error while generating parity artifacts: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
