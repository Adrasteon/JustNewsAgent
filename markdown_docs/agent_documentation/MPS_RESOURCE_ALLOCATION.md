---
title: MPS Resource Allocation System
description: Machine-readable GPU memory allocation configuration for NVIDIA MPS - calculated from agent model requirements
tags: [mps, gpu, allocation, memory, configuration]
status: current
last_updated: 2025-09-17
---

# MPS Resource Allocation System

## Overview

The JustNewsAgent system uses NVIDIA Multi-Process Service (MPS) for enterprise-grade GPU resource isolation and management. This document describes the machine-readable configuration system that calculates and allocates fixed GPU memory limits per agent based on their model requirements.

## Architecture

### Configuration Files

**Primary Configuration**: `config/gpu/mps_allocation_config.json`
- Machine-readable format for MPS resource allocation
- Calculated from actual model memory requirements
- Includes safety margins and priority levels
- Used by GPU orchestrator for policy enforcement

**Model Memory Database**: `config/gpu/model_config.json`
- Per-model memory usage specifications
- Agent-to-model mappings
- Batch size recommendations
- Memory requirement calculations

### Memory Calculation Methodology

#### 1. Model Memory Assessment
Each model is profiled for GPU memory usage under typical workloads:
- **Base Memory**: Model weights and KV cache
- **Working Memory**: Batch processing overhead
- **Safety Margins**: 50-100% buffer for variance

#### 2. Agent Aggregation
Memory requirements are summed across all models used by each agent:
```python
agent_memory_gb = sum(model_memory_gb for model in agent_models)
```

#### 3. MPS Allocation Strategy
Final MPS limits include calculated requirements plus safety margins:
```python
mps_limit_gb = calculated_memory_gb + safety_margin_gb
```

## Agent Memory Requirements

### Calculated Allocations

| Agent | Models | Calculated Memory | MPS Limit | Safety Margin |
|-------|--------|------------------|-----------|---------------|
| **analyst** | google/bert_uncased_L-2_H-128_A-2 | 0.5GB | 1.0GB | 0.5GB |
| **balancer** | google/bert_uncased_L-2_H-128_A-2 | 0.5GB | 1.0GB | 0.5GB |
| **chief_editor** | distilbert-base-uncased | 1.0GB | 2.0GB | 1.0GB |
| **critic** | unitary/unbiased-toxic-roberta, unitary/toxic-bert | 3.5GB | 4.0GB | 0.5GB |
| **fact_checker** | distilbert-base-uncased, roberta-base, sentence-transformers/all-mpnet-base-v2 | 4.0GB | 5.0GB | 1.0GB |
| **memory** | sentence-transformers/all-MiniLM-L6-v2 | 0.5GB | 1.0GB | 0.5GB |
| **newsreader** | sentence-transformers/all-MiniLM-L6-v2 | 0.5GB | 1.0GB | 0.5GB |
| **scout** | google/bert_uncased_L-2_H-128_A-2, cardiffnlp/twitter-roberta-base-sentiment-latest, martin-ha/toxic-comment-model | 3.0GB | 4.0GB | 1.0GB |
| **synthesizer** | distilgpt2, google/flan-t5-small | 3.0GB | 4.0GB | 1.0GB |

### System Summary
- **Total Agents**: 9
- **Total Calculated Memory**: 16.0GB
- **Total MPS Allocation**: 23.0GB
- **Total Safety Margin**: 7.0GB
- **Memory Efficiency**: 69.6%
- **GPU Utilization**: 95.8% of 24GB RTX 3090

## MPS Configuration API

### GPU Orchestrator Endpoints

**GET /mps/allocation**
Returns the complete MPS allocation configuration:
```json
{
  "mps_resource_allocation": {
    "agent_allocations": {
      "analyst": {
        "mps_memory_limit_gb": 1.0,
        "calculated_requirement_gb": 0.5,
        "safety_margin_gb": 0.5,
        "priority": "medium",
        "models": ["google/bert_uncased_L-2_H-128_A-2"]
      }
    }
  }
}
```

**GET /gpu/info**
Includes MPS status and detection:
```json
{
  "mps_enabled": true,
  "mps": {
    "enabled": true,
    "pipe_dir": "/tmp/nvidia-mps",
    "control_process": true
  }
}
```

### MPS Control Commands

```bash
# Start MPS control daemon
nvidia-cuda-mps-control -d

# Check MPS status
curl -s http://localhost:8014/gpu/info | jq '.mps'

# Get allocation configuration
curl -s http://localhost:8014/mps/allocation | jq '.mps_resource_allocation.agent_allocations'
```

## Implementation Details

### Memory Limit Enforcement

MPS enforces per-process GPU memory limits:
```bash
# Set MPS memory limit for a process
export CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=1GB
export CUDA_VISIBLE_DEVICES=0

# Launch agent with memory constraints
python agent_script.py
```

### Resource Isolation Benefits

1. **Process Separation**: Each agent runs in isolated MPS context
2. **Memory Protection**: Prevents memory leaks from affecting other agents
3. **Fair Allocation**: Equal GPU access regardless of launch order
4. **Crash Isolation**: GPU hangs contained to individual agent processes
5. **Debugging**: Per-client GPU usage tracking and error isolation

### Performance Characteristics

- **Memory Overhead**: ~0.5GB for MPS control daemon
- **Context Switching**: Minimal latency between MPS clients
- **Throughput**: Near-native GPU performance with isolation
- **Scalability**: Supports up to 48 concurrent MPS clients

## Configuration Management

### Updating Allocations

When new models are added or memory requirements change:

1. **Update Model Config**: Modify `config/gpu/model_config.json`
2. **Recalculate Requirements**: Run memory calculation script
3. **Update MPS Config**: Regenerate `mps_allocation_config.json`
4. **Restart Services**: Apply new limits to running agents

### Validation Checks

```bash
# Validate configuration
python config/validate_config.py

# Check MPS status
curl -s http://localhost:8014/mps/allocation | jq '.mps_resource_allocation.system_summary'
```

## Troubleshooting

### Common Issues

**MPS Not Enabled**:
```bash
# Check MPS daemon
pgrep -x nvidia-cuda-mps-control

# Start MPS if needed
nvidia-cuda-mps-control -d
```

**Memory Limit Exceeded**:
```bash
# Check current allocations
curl -s http://localhost:8014/mps/allocation

# Increase limits in configuration
# Restart affected agents
```

**Resource Conflicts**:
```bash
# Monitor MPS clients
ls /tmp/nvidia-mps/ | grep client

# Check GPU usage per client
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
```

## Future Enhancements

### Planned Features
- **Dynamic Allocation**: Runtime memory limit adjustments
- **Priority Scheduling**: Agent priority-based resource allocation
- **Memory Profiling**: Automated memory usage analysis
- **Auto-scaling**: Memory limit optimization based on usage patterns

### Integration Points
- **Kubernetes**: MPS-aware GPU resource management
- **Monitoring**: Real-time MPS client performance tracking
- **Orchestration**: Automated MPS configuration deployment

---

**Configuration Version**: 1.0
**Last Updated**: September 17, 2025
**Total System Memory**: 23.0GB allocated, 16.0GB calculated
**MPS Efficiency**: 69.6% memory utilization with safety margins

**Deployment Note**: MPS configuration is now included in systemd environment examples and deployment guides.
