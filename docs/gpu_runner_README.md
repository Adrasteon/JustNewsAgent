GPU Runner README

This document describes how to run the GPU-enabled build runner on a machine
with CUDA/TensorRT/PyCUDA and RAPIDS installed.

**✅ GPU Management Status:** All agents now use production MultiAgentGPUManager for conflict-free operation

Requirements
- CUDA toolkit 12.4+ (matching your GPU)
- NVIDIA drivers
- TensorRT (tested with 8.x)
- PyCUDA
- RAPIDS 25.04+ (includes cudf, cuml, cugraph, cuspatial, cuvs)
- Python packages: torch, transformers, numpy

Recommended conda environment setup (RAPIDS-integrated):

```bash
# Use the main RAPIDS environment
conda activate justnews-v2-py312

# Verify RAPIDS installation
python -c "import cudf, cuml, cugraph; print('RAPIDS libraries loaded successfully')"

# Check GPU memory
nvidia-smi
```

**Alternative: Legacy separate GPU environment**
```bash
conda create -n justnews-gpu python=3.11 -y
conda activate justnews-gpu
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers numpy pycuda
# Install TensorRT python wheel matching your system (downloaded from NVIDIA)
```

How to run

export MODEL_STORE_ROOT=/path/to/models
python scripts/gpu_runner.py --precision int8 --calib-data /path/to/calib.jsonl --sentiment

Notes
- The script will attempt to perform INT8 calibration if `--precision int8` and
  `--calib-data` are provided. Calibration requires a GPU and the TensorRT/PyCUDA
  runtime.
- If the environment lacks TensorRT/PyCUDA the compiler will create placeholder
  calibration files and still upload markers to the ModelStore (if configured).

---

## ✅ **Production GPU Management - IMPLEMENTED**

The JustNewsAgent system now features **production-grade GPU resource management**:

### Key Features
- **Multi-Agent Support:** Concurrent GPU allocation for all 6 GPU-enabled agents
- **Conflict Prevention:** Coordinated resource allocation prevents GPU conflicts
- **Dynamic Allocation:** Automatic GPU device assignment based on availability
- **Memory Management:** Intelligent memory allocation (2-8GB per agent)
- **Health Monitoring:** Real-time GPU usage tracking and error recovery
- **Fallback Support:** Automatic CPU fallback when GPU unavailable

### Agent GPU Allocations
| Agent | Memory Allocation | GPU Manager Status |
|-------|------------------|-------------------|
| Synthesizer | 6-8GB | ✅ Production Manager |
| Analyst | 4-6GB | ✅ Production Manager |
| Scout | 4-6GB | ✅ Production Manager |
| Fact Checker | 4-6GB | ✅ Production Manager |
| Memory | 2-4GB | ✅ Production Manager |
| Newsreader | 4-8GB | ✅ Production Manager |

### Integration Pattern
```python
# All agents now follow this production pattern
from agents.common.gpu_manager import request_agent_gpu, release_agent_gpu

def __init__(self):
    self.gpu_device = request_agent_gpu(f"{agent_name}_agent", memory_gb=X)
    if self.gpu_device is not None:
        self.device = torch.device(f"cuda:{self.gpu_device}")
    else:
        self.device = torch.device("cpu")  # Fallback

def cleanup(self):
    if self.gpu_device is not None:
        release_agent_gpu(f"{agent_name}_agent")
```

---

RAPIDS Integration
------------------

The environment now includes RAPIDS 25.04 for GPU-accelerated data processing:

**Available RAPIDS Libraries:**
- `cudf`: GPU DataFrames (drop-in pandas replacement)
- `cuml`: GPU machine learning (scikit-learn compatible)
- `cugraph`: GPU graph analytics
- `cuspatial`: GPU spatial computations
- `cuvs`: GPU vector search and similarity

**Example Usage:**
```python
import cudf
import cuml

# GPU DataFrame operations
df = cudf.read_csv('data.csv')
result = df.groupby('category').mean()

# GPU machine learning
from cuml.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)
```

**Memory Management:**
- RAPIDS automatically manages GPU memory
- Monitor usage with `nvidia-smi`
- Set memory limits if needed: `cudf.set_allocator("managed")`
