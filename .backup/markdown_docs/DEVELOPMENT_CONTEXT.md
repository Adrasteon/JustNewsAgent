---
title: JustNews Agentic - Development Context
description: Auto-generated description for JustNews Agentic - Development Context
tags: [documentation]
status: current
last_updated: 2025-09-12
---

# JustNews Agentic - Development Context

**Last Updated**: September 2, 2025  
**Branch**: `dev/gpu_implementation`  
**Status**: Production-Validated RAPIDS 25.04 Integration + Package Management Complete  

## 🚨 **MAJOR BREAKTHROUGH - GPU Crash Investigation Resolved**

### Critical Discovery Summary (August 13, 2025)

After extensive crash investigation involving multiple system crashes and PC resets, we have **definitively identified and resolved** the root cause of the GPU crashes that were occurring consistently around the 5th article processing.

#### **Root Cause Analysis**

The crashes were **NOT caused by GPU memory exhaustion** as initially suspected, but by:

1. **Incorrect Quantization Method**:
   - ❌ **Wrong**: `torch_dtype=torch.int8` (causes `ValueError: Can't instantiate LlavaForConditionalGeneration model under dtype=torch.int8 since it is not a floating point dtype`)
   - ✅ **Correct**: `BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=torch.float16, ...)`

2. **Improper LLaVA Conversation Format**:
   - ❌ **Wrong**: Simple string format `"USER: <image>\nAnalyze this ASSISTANT:"`
   - ✅ **Correct**: Structured conversation format with separate image and text content

3. **SystemD Environment Configuration**:
   - Missing `CUDA_VISIBLE_DEVICES=0` and proper conda environment paths

#### **Production-Validated Solution**

Our final GPU crash isolation test achieved **100% success rate** using the correct configuration:

```python
# CORRECT quantization setup
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16,
    bnb_8bit_use_double_quant=True,
)

# CORRECT model loading
model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    torch_dtype=torch.float16,  # Use float16, not int8
    quantization_config=quantization_config,
    device_map="auto",
    low_cpu_mem_usage=True,
    max_memory={0: "8GB"},  # Conservative crash-safe limit
    trust_remote_code=True
)

# CORRECT conversation format
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": custom_prompt}
        ]
    }
]
```

#### **Validation Results**

**Test Results (August 13, 2025)**:
- ✅ **Zero crashes** during intensive testing
- ✅ **Stable GPU memory**: 6.85GB allocated, 7.36GB reserved
- ✅ **Stable system memory**: 24.8% usage (~7.3GB of 31GB)
- ✅ **Proper LLaVA functionality**: Successful news screenshot analysis
- ✅ **Critical test passed**: Successfully processed 5th image (previous crash point)

## 📊 **Current System Status**

### Production Environment
- **Hardware**: NVIDIA GeForce RTX 3090 (24GB VRAM)
- **System RAM**: 31GB
- **Primary Environment**: `justnews-v2-py312` (Python 3.12.11)
- **Secondary Environment**: `justnews-v2-prod` (Python 3.11.13)
- **RAPIDS Version**: 25.04 (fully integrated)
- **CUDA Version**: 12.4
- **PyTorch**: 2.5.1+cu124

### RAPIDS Integration Status
- ✅ **cudf**: GPU DataFrames - Active and tested
- ✅ **cuml**: GPU Machine Learning - Active and tested
- ✅ **cugraph**: GPU Graph Analytics - Available
- ✅ **cuspatial**: GPU Spatial Analytics - Available
- ✅ **cuvs**: GPU Vector Search - Available
- ✅ **Python 3.12 Compatibility**: Fully validated

### Active Services
```bash
# NewsReader V2 Service (Production-Validated)
sudo systemctl status justnews@newsreader
# Status: ✅ Active and stable with RAPIDS integration

# Balancer Service
sudo systemctl status justnews@balancer
# Status: ✅ Active with GPU acceleration support
```

### Memory Usage (Stable Operation)
```
GPU Memory Usage:
- Allocated: 6.85GB (RAPIDS + PyTorch)
- Reserved: 7.36GB
- Total Available: 24GB
- Utilization: ~29% (well within safe limits)

System Memory Usage:
- Used: ~7.3GB / 31GB (24.8%)
- Status: Stable with no memory leaks
```

## 📦 **Package Management & Environment Optimization - PRODUCTION READY**

### Package Installation Summary (September 2, 2025)

Successfully completed comprehensive package management for core JustNewsAgent dependencies, ensuring all critical packages are properly installed and tested in the production environment.

#### **Strategic Package Installation Approach**
- **Conda-First Strategy**: Prioritized conda-forge channel for available packages
- **Pip Fallback**: Used pip only for packages unavailable in conda channels (TensorRT)
- **Compatibility Validation**: Ensured all packages work with existing PyTorch 2.8.0+cu128 environment
- **GPU Compatibility**: Verified all packages compatible with RTX 3090 and CUDA 12.8

#### **Core Packages Installed & Tested**

**✅ TensorRT 10.13.3.9**
- **Installation Method**: pip (not available in conda-forge/nvidia channels)
- **Purpose**: Native GPU acceleration for Analyst agent operations
- **Status**: ✅ Installed and functional with existing TensorRT engines
- **Integration**: Seamless compatibility with PyCUDA and existing GPU workflows

**✅ PyCUDA**
- **Installation Method**: conda-forge
- **Purpose**: GPU CUDA operations for TensorRT inference
- **Status**: ✅ Installed and tested successfully
- **Integration**: Working with TensorRT engines for GPU memory management

**✅ BERTopic**
- **Installation Method**: conda-forge
- **Purpose**: Topic modeling in Synthesizer V3 production stack
- **Status**: ✅ Installed and functional
- **Integration**: Compatible with existing sentence-transformers and clustering workflows

**✅ spaCy**
- **Installation Method**: conda-forge
- **Purpose**: Natural language processing in Fact Checker agent
- **Status**: ✅ Installed and operational
- **Integration**: Working with existing NLP pipelines and model loading

#### **Package Compatibility Validation**
- **Environment**: `justnews-v2-prod` (Python 3.12.11, PyTorch 2.8.0+cu128)
- **GPU**: RTX 3090 with CUDA 12.8 compatibility confirmed
- **Dependencies**: Zero conflicts with existing RAPIDS 25.04 and PyTorch ecosystem
- **Testing**: All packages imported and basic functionality validated
- **Production Impact**: No disruption to existing agent operations or performance

#### **Installation Strategy Benefits**
1. **Conda Ecosystem**: Leveraged conda-forge for reliable, tested package builds
2. **Minimal Conflicts**: Strategic pip fallback prevented dependency resolution issues
3. **GPU Optimization**: All packages compatible with CUDA 12.8 and RTX 3090
4. **Production Stability**: Comprehensive testing ensures no runtime issues
5. **Future Maintenance**: Clear documentation of installation methods and sources

#### **Agent Integration Status**
- **Analyst Agent**: TensorRT + PyCUDA integration maintained and enhanced
- **Synthesizer Agent**: BERTopic integration preserved for V3 production stack
- **Fact Checker Agent**: spaCy functionality maintained for NLP operations
- **System Stability**: All GPU-accelerated operations functional with updated packages

**Package Management Status**: **COMPLETE** - All core packages installed, tested, and production-ready

## 🚀 **RAPIDS 25.04 Integration - Major Enhancement**

### Integration Summary (August 31, 2025)

Successfully integrated RAPIDS 25.04 into the primary development environment, enabling GPU-accelerated data science operations across the JustNewsAgent system.

#### **Environment Optimization**

**Before Integration:**
- Separate `justnews-rapids` environment (Python 3.11, RAPIDS 24.08)
- Multiple conda environments causing maintenance overhead
- Limited Python 3.12 compatibility

**After Integration:**
- Unified `justnews-v2-py312` environment (Python 3.12.11)
- RAPIDS 25.04 with full Python 3.12 support
- Streamlined environment management
- CUDA 12.4 compatibility

#### **RAPIDS Libraries Integration**

```python
# GPU DataFrames (pandas-compatible)
import cudf
df = cudf.read_csv('news_data.csv')
processed_data = df.groupby('category').sentiment.mean()

# GPU Machine Learning
import cuml
from cuml.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

# GPU Graph Analytics
import cugraph
G = cugraph.Graph()
G.from_cudf_edgelist(df, source='source', destination='target')
```

#### **Performance Benefits**

- **Data Processing**: 10-100x faster than CPU pandas operations
- **Machine Learning**: GPU-accelerated training and inference
- **Memory Efficiency**: Direct GPU memory usage without CPU roundtrips
- **Scalability**: Handle larger datasets with RTX 3090's 24GB VRAM

## 🔧 **Development Process & Lessons Learned**

### Investigation Methodology
1. **Systematic Crash Isolation**: Created minimal test scripts to isolate exact crash points
2. **Progressive Testing**: Started with single images, then critical 5th image
3. **Configuration Comparison**: Analyzed working newsreader vs. failing test configurations
4. **Environment Validation**: Ensured proper conda activation and CUDA visibility

### Key Technical Insights
- **Quantization Complexity**: Modern transformer quantization requires specialized configuration objects
- **LLaVA Input Format**: Vision-language models need structured conversation format, not simple strings
- **Memory Management**: Conservative limits (30% of GPU memory) prevent crashes while maintaining functionality
- **Environment Consistency**: SystemD services need explicit environment variable configuration

### Documentation Created
- **`Using-The-GPU-Correctly.md`**: Complete configuration guide with error resolution
- **Updated Technical Architecture**: Crash resolution details in main docs
- **Updated NewsReader README**: Production-validated status and configuration details
- **CHANGELOG**: Major breakthrough documentation

## 🎯 **Current Development Focus**

### Immediate Status
- ✅ **RAPIDS Integration**: Complete - 25.04 with Python 3.12 support
- ✅ **Environment Optimization**: Streamlined to 3 environments (base, justnews-v2-prod, justnews-v2-py312)
- ✅ **GPU Configuration**: Production-validated and crash-free
- ✅ **NewsReader Service**: Stable operation with RAPIDS acceleration
- ✅ **Documentation**: Updated with RAPIDS integration details
- ✅ **System Stability**: Zero crashes in production testing

### Next Steps
1. **RAPIDS Utilization**: Implement GPU-accelerated data processing in agents
2. **Performance Benchmarking**: Compare CPU vs GPU performance metrics
3. **Memory Optimization**: Fine-tune RAPIDS memory management
4. **Extended Testing**: Run longer processing sessions with RAPIDS workloads
5. **Production Deployment**: Roll out RAPIDS-accelerated features across all agents

## 📚 **Reference Documentation**

### Primary Documents
- **`README.md`**: Updated with RAPIDS 25.04 integration guide
- **`docs/gpu_runner_README.md`**: RAPIDS usage examples and GPU memory management
- **`TECHNICAL_ARCHITECTURE.md`**: System architecture with RAPIDS integration details
- **`agents/analyst/requirements_v4.txt`**: RAPIDS dependencies and versions
- **`CHANGELOG.md`**: Version history with RAPIDS integration documentation

### RAPIDS Integration
- **RAPIDS Libraries**: cudf, cuml, cugraph, cuspatial, cuvs
- **Python Compatibility**: 3.12+ support with RAPIDS 25.04+
- **CUDA Compatibility**: 12.4+ required
- **GPU Requirements**: RTX 3090/4090 recommended (24GB+ VRAM)

### Test Files
- **`final_corrected_gpu_test.py`**: Production-validated GPU configuration test
- **`final_corrected_gpu_results_*.json`**: Test results proving stability
- **RAPIDS validation scripts**: GPU library import and functionality tests

### Configuration Files
- **`/etc/systemd/system/justnews@newsreader.service`**: Correct SystemD configuration
- **`agents/newsreader/newsreader_v2_true_engine.py`**: Working production engine
- **`agents/newsreader/main_v2.py`**: FastAPI service with correct configuration

## 🏆 **Success Metrics**

### Before Fix
- **Crash Rate**: 100% (consistent crashes at 5th image)
- **System Stability**: Complete PC resets required
- **Processing**: Unable to complete multi-image analysis

### After Fix  
- **Crash Rate**: 0% (zero crashes in comprehensive testing)
- **System Stability**: Stable throughout extended testing
- **Processing**: Successful multi-image analysis with proper LLaVA responses
- **Memory Usage**: Stable and predictable (6.85GB GPU, 24.8% system)

---

**Development Team Notes**: This breakthrough resolves months of intermittent crash issues and establishes a solid foundation for production deployment. The key was systematic investigation rather than assumptions about memory limits being the primary cause.

**Next Review Date**: September 13, 2025 (monitor for any stability issues)

## ⚙️ **Centralized Configuration System - ENTERPRISE-GRADE MANAGEMENT**

### **🎯 System Overview**
JustNewsAgent now features a comprehensive **centralized configuration system** that provides enterprise-grade configuration management with environment overrides, validation, and unified access to all critical system variables.

### **📁 Configuration Architecture**
```
config/
├── system_config.json          # Main system configuration (12 sections)
├── system_config.py           # Python configuration manager with env overrides
├── validate_config.py         # Comprehensive validation with error reporting
├── config_quickref.py         # Interactive quick reference tool
└── gpu/                       # GPU-specific configurations
    ├── gpu_config.json        # GPU resource management
    ├── environment_config.json # Environment-specific GPU settings
    ├── model_config.json      # Model-specific configurations
    └── config_profiles.json   # Configuration profiles
```

### **🔧 Core Features**

#### **1. Unified Variable Management**
- **12 Major Configuration Sections**: system, mcp_bus, database, crawling, gpu, agents, training, monitoring, data_minimization, performance, external_services
- **Environment Variable Overrides**: Runtime configuration without code changes
- **Automatic Validation**: Comprehensive error checking with helpful messages
- **Production-Ready Defaults**: Sensible defaults for all critical variables

#### **2. Critical System Variables**
```json
{
  "crawling": {
    "obey_robots_txt": true,
    "requests_per_minute": 20,
    "delay_between_requests_seconds": 2.0,
    "concurrent_sites": 3,
    "user_agent": "JustNewsAgent/4.0"
  },
  "gpu": {
    "enabled": true,
    "max_memory_per_agent_gb": 8.0,
    "temperature_limits": {
      "warning_celsius": 75,
      "critical_celsius": 85
    }
  }
}
```

#### **3. Environment Override System**
```bash
# Crawling Configuration
export CRAWLER_REQUESTS_PER_MINUTE=15
export CRAWLER_DELAY_BETWEEN_REQUESTS=3.0
export CRAWLER_CONCURRENT_SITES=2

# Database Configuration
export POSTGRES_HOST=production-db.example.com
export POSTGRES_DB=justnews_prod

# System Configuration
export LOG_LEVEL=DEBUG
export GPU_ENABLED=true
```

### **🚀 Usage Patterns**

#### **Python API Access:**
```python
from config.system_config import config

# Get crawling configuration
crawl_config = config.get('crawling')
rpm = config.get('crawling.rate_limiting.requests_per_minute')
robots_compliance = config.get('crawling.obey_robots_txt')

# Get GPU configuration
gpu_enabled = config.get('gpu.enabled')
max_memory = config.get('gpu.memory_management.max_memory_per_agent_gb')
```

#### **Interactive Tools:**
```bash
# Display all current settings
/media/adra/Extend/miniconda3/envs/justnews-v2-py312/bin/python config/config_quickref.py

# Validate configuration
/media/adra/Extend/miniconda3/envs/justnews-v2-py312/bin/python config/validate_config.py
```

### **✅ Enterprise Benefits**

1. **🎯 Single Source of Truth**: All critical variables centralized
2. **🔧 Environment Flexibility**: Easy deployment across dev/staging/prod
3. **🚀 Runtime Updates**: Modify settings without service restarts
4. **🛡️ Validation & Safety**: Automatic validation prevents misconfigurations
5. **📚 Self-Documenting**: Clear structure with comprehensive defaults
6. **🏢 Production Ready**: Enterprise-grade configuration management

### **🔍 Validation & Monitoring**

#### **Configuration Validation:**
```bash
# Run comprehensive validation
python config/validate_config.py

# Example output:
=== JustNewsAgent Configuration Validation Report ===

⚠️  WARNINGS:
  • Database password is empty in production environment

✅ Configuration is valid with no errors found!
```

This centralized configuration system provides **enterprise-grade configuration management** that makes it easy to locate, adjust, and manage all critical system variables across development, staging, and production environments! 🎯✨

## See also

- Technical Architecture: markdown_docs/TECHNICAL_ARCHITECTURE.md
- Documentation Catalogue: docs/DOCUMENTATION_CATALOGUE.md

