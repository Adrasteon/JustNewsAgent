"""
Lightweight GPU monitor for live systems.
Writes JSONL entries to logs/gpu_monitor.jsonl with timestamps and metrics.

Usage (in the activated conda env):
python tools/gpu_monitor.py --interval 2 --duration 300

It uses nvidia-smi via subprocess (no extra deps) and falls back to PyTorch if available.
"""
import argparse
import json
import os
import subprocess
import time
from datetime import datetime

OUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'logs', 'gpu_monitor.jsonl')
OUT_PATH = os.path.abspath(OUT_PATH)


def read_nvidia_smi():
    """Read a compact JSON output from nvidia-smi if available."""
    try:
        proc = subprocess.run([
            'nvidia-smi',
            '--query-gpu=index,name,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,temperature.gpu,power.draw',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=5)
        if proc.returncode != 0:
            return None
        lines = proc.stdout.strip().splitlines()
        gpus = []
        for line in lines:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 9:
                continue
            gpus.append({
                'index': int(parts[0]),
                'name': parts[1],
                'utilization_gpu_pct': float(parts[2]),
                'utilization_mem_pct': float(parts[3]),
                'memory_total_mb': float(parts[4]),
                'memory_used_mb': float(parts[5]),
                'memory_free_mb': float(parts[6]),
                'temperature_c': float(parts[7]),
                'power_draw_w': float(parts[8]) if parts[8] else None
            })
        return gpus
    except Exception:
        return None


def read_torch():
    try:
        import torch
        gpus = []
        if not torch.cuda.is_available():
            return None
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            used = torch.cuda.memory_allocated(i) / 1024**2
            reserved = torch.cuda.memory_reserved(i) / 1024**2
            gpus.append({
                'index': i,
                'name': props.name,
                'memory_total_mb': props.total_memory / 1024**2,
                'memory_allocated_mb': used,
                'memory_reserved_mb': reserved
            })
        return gpus
    except Exception:
        return None


def sample_once():
    now = datetime.now(datetime.UTC).isoformat()
    record = {'timestamp': now}
    nvsmi = read_nvidia_smi()
    if nvsmi is not None:
        record['nvidia_smi'] = nvsmi
    else:
        torch_data = read_torch()
        if torch_data is not None:
            record['torch'] = torch_data
        else:
            record['error'] = 'nvidia-smi and torch unavailable'
    return record


def ensure_out_dir():
    d = os.path.dirname(OUT_PATH)
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def main(interval: float, duration: float):
    ensure_out_dir()
    end_time = time.time() + duration if duration and duration > 0 else None
    print(f"Writing GPU samples to: {OUT_PATH}")
    with open(OUT_PATH, 'a', encoding='utf-8') as f:
        while True:
            rec = sample_once()
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if end_time and time.time() >= end_time:
                break
            time.sleep(interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--interval', type=float, default=2.0, help='Seconds between samples')
    parser.add_argument('--duration', type=float, default=0.0, help='Total duration in seconds (0 for infinite)')
    args = parser.parse_args()
    try:
        main(args.interval, args.duration)
    except KeyboardInterrupt:
        print('Interrupted')
