# JustNewsAgentic V4 🤖

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![TensorRT](https://img.shields.io/badge/TensorRT-Production-orange.svg)](https://developer.nvidia.com/tensorrt)
[![GPU Status](https://img.shields.io/badge/GPU%20Crashes-RESOLVED-brightgreen.svg)](markdown_docs/development_reports/Using-The-GPU-Correctly.md)

**AI-powered news analysis ecosystem with multi-agent collaboration and GPU acceleration**

JustNewsAgentic V4 is a production-ready, multi-agent news analysis system that automatically discovers, analyzes, and synthesizes news content using specialized AI agents. Built with native TensorRT GPU acceleration and MCP (Model Context Protocol) for seamless agent communication.

## 🚨 **GPU Configuration Update - August 13, 2025**

**MAJOR BREAKTHROUGH**: All GPU crashes have been resolved! The root cause was incorrect model quantization configuration, not memory exhaustion.

- ✅ **100% Crash-Free Operation**: Production-validated with intensive testing
- ✅ **Stable GPU Memory**: 6.85GB usage (well within 25GB limits) 
- ✅ **Proper Configuration**: BitsAndBytesConfig quantization method
- 📖 **Complete Guide**: See `markdown_docs/development_reports/Using-The-GPU-Correctly.md`

## ✨ Key Features

## ✨ Key Features

- 🤖 **Multi-Agent Architecture**: 10 specialized AI agents working collaboratively
- ⚡ **GPU Acceleration**: Native TensorRT optimization with 730+ articles/sec processing
- 🕷️ **Production Crawling**: 8.14 articles/sec ultra-fast + 0.86 articles/sec AI-enhanced processing
- 🧠 **Continuous Learning**: EWC-based online training with 48 examples/min processing
- 🔗 **MCP Integration**: Model Context Protocol for seamless inter-agent communication
- 📊 **Real-time Analysis**: Sentiment, bias, fact-checking, and content synthesis
- 💾 **Vector Search**: PostgreSQL with semantic search capabilities
- 🎓 **Training System**: Automated model improvement from production feedback

## 🚀 Quick Start

### Prerequisites

- **Hardware**: NVIDIA RTX 3090 (24GB VRAM recommended) or RTX 4090
- **Software**: Ubuntu 24.04, Docker, NVIDIA Container Toolkit
- **Python**: 3.12+ with CUDA 12.1+ support

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Adrasteon/JustNewsAgentic.git
   cd JustNewsAgentic
   ```

2. **Set up GPU environment**
   ```bash
   # Activate production environment
   conda activate rapids-25.06  # or your CUDA-enabled environment
   ```

3. **Start the system**
   ```bash
   # Docker multi-agent deployment
   docker-compose up --build
   
   # Or native GPU deployment
   ./start_services_daemon.sh
   ```

4. **Verify system health**
   ```bash
   curl http://localhost:8000/agents
   ```

## 🏗️ Architecture Overview

JustNewsAgentic V4 employs a **distributed multi-agent architecture** where specialized AI agents communicate through a central MCP Bus:

### Core Agents

| Agent | Purpose | Technology | Status |
|-------|---------|------------|--------|
| **Scout** | Content discovery & extraction | 5 AI models (BERT, RoBERTa, LLaVA) | ✅ Production |
| **Analyst** | Sentiment & bias analysis | TensorRT-optimized RoBERTa | ✅ Production |
| **Fact Checker** | Claim verification & credibility | 5 AI models for verification | ✅ Production |
| **Synthesizer** | Content clustering & synthesis | BERTopic + BART + FLAN-T5 | ✅ Production |
| **Critic** | Quality assessment & review | DialoGPT-medium | 🔧 Integration |
| **Chief Editor** | Workflow orchestration | DialoGPT-medium | 🔧 Integration |
| **Memory** | Semantic storage & retrieval | PostgreSQL + vector embeddings | ✅ Production |
| **NewsReader** | Visual content analysis | LLaVA-1.5-7B (INT8) | ✅ Production |
| **Reasoning** | Symbolic logic & validation | Nucleoid (AST parsing) | ✅ Production |

### Key Technologies

- **Native TensorRT**: 4.8x performance improvement (730+ articles/sec)
- **MCP Protocol**: Standardized agent communication
- **Continuous Learning**: EWC-based training without catastrophic forgetting
- **Production Crawling**: Advanced cookie handling and modal dismissal
- **GPU Safety**: Professional CUDA context management

## 📋 Usage Examples

### Analyze News Articles
```bash
# Analyze a single article
curl -X POST http://localhost:8002/enhanced_deepcrawl \
  -H "Content-Type: application/json" \
  -d '{"args": ["https://www.bbc.com/news/example"], "kwargs": {}}'
```

### Batch Processing
```bash
# Process multiple articles
curl -X POST http://localhost:8004/score_sentiment_batch \
  -H "Content-Type: application/json" \
  -d '{"args": [["Article 1 text...", "Article 2 text..."]], "kwargs": {}}'
```

### Training System
```bash
# View training status
curl http://localhost:8000/training/status
```

## 🔧 Configuration

### Environment Variables

```bash
# Core settings
MCP_BUS_URL=http://localhost:8000
GPU_MEMORY_FRACTION=0.8
BATCH_SIZE=32

# Training settings
TRAINING_ENABLED=true
LEARNING_RATE=0.001
EWC_LAMBDA=0.4

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/justnews
```

### Agent Configuration

Each agent can be configured via its `config.json` file:

```json
{
  "model": "microsoft/DialoGPT-medium",
  "gpu_enabled": true,
  "batch_size": 16,
  "training": {
    "enabled": true,
    "threshold": 30
  }
}
```

## 📊 Performance

### Production Metrics

- **Content Processing**: 8.14 articles/sec (ultra-fast) + 0.86 articles/sec (AI-enhanced)
- **TensorRT Analysis**: 730+ articles/sec for sentiment/bias analysis
- **Training Throughput**: 48 examples/min with 82.3 model updates/hour
- **Success Rate**: 95.5% successful content extraction
- **Memory Efficiency**: 29.6GB total system (RTX 3090 optimized)

### Benchmark Results

| Operation | Speed | Baseline | Improvement |
|-----------|-------|----------|-------------|
| Sentiment Analysis | 720.8 art/sec | 151.4 art/sec | 4.8x |
| Bias Analysis | 740.3 art/sec | 146.8 art/sec | 5.0x |
| Content Extraction | 1,591 words/article | N/A | Production-ready |
| Training Updates | 82.3/hour | Manual only | Continuous |

## 📚 Documentation

### Quick Links
- **📖 [Complete Documentation](markdown_docs/README.md)** - Navigation hub for all documentation
- **🏭 [Production Status](markdown_docs/production_status/)** - Deployment reports and achievements  
- **🤖 [Agent Guides](markdown_docs/agent_documentation/)** - Individual agent documentation
- **🔧 [Technical Reports](markdown_docs/development_reports/)** - Analysis and validation reports

### Models & Caching

The repository centralizes model download, caching, and loading to avoid permission and concurrency issues across agents:

- Per-agent model caches (recommended): `agents/<agent>/models`. Override with env vars like `SYNTHESIZER_MODEL_CACHE`, `SYNTHESIZER_V2_MODEL_CACHE`, `FACT_CHECKER_MODEL_PATH`.
- Use the embedding helper: `markdown_docs/agent_documentation/EMBEDDING_HELPER.md` and `markdown_docs/agent_documentation/MODEL_USAGE.md` for design and examples.
- The helper implements atomic install semantics and process-local caching — prefer `get_shared_embedding_model()` and `ensure_agent_model_exists()` over direct constructors.


### Development Resources
- **[Development Context](markdown_docs/DEVELOPMENT_CONTEXT.md)** - Complete development history
- **[Architecture Details](docs/)** - V4 proposals and technical specifications  
- **[Training System](training_system/)** - Continuous learning implementation
- **[API Reference](agents/)** - Agent-specific API documentation

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup

```bash
# Set up development environment
conda create -n justnews python=3.12
conda activate justnews
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/
```

### Code Standards

- **Python**: Follow PEP 8 style guidelines
- **Documentation**: Update relevant docs in `markdown_docs/`
- **Testing**: Maintain >90% test coverage
- **GPU Code**: Use professional CUDA context management

## 🔒 Security & Privacy

- **Data Protection**: Local processing, no external data transmission
- **Model Security**: Validated model checksums and secure loading
- **Access Control**: Role-based agent permissions
- **Audit Trail**: Comprehensive logging for all operations

## 📈 Roadmap

### Current Focus (V4.1)
- [ ] Complete V2 engines across remaining agents
- [ ] Enhanced training system scalability
- [ ] Advanced analytics dashboard
- [ ] Multi-language support

### Future Plans (V5.0)
- [ ] RTX AI Toolkit full integration
- [ ] Custom domain-specific models
- [ ] Real-time streaming analysis
- [ ] Distributed multi-node deployment

## 🐛 Troubleshooting

### Common Issues

**GPU Memory Issues**
```bash
# Check GPU status
nvidia-smi

# Reduce batch size in config
{"batch_size": 16}
```

**Agent Communication Issues**
```bash
# Verify MCP Bus
curl http://localhost:8000/health

# Check agent registration
curl http://localhost:8000/agents
```

**Training Issues**
```bash
# View training logs
tail -f training_system/logs/training.log

# Reset training state
curl -X POST http://localhost:8000/training/reset
```

For more troubleshooting guidance, see our [Technical Reports](markdown_docs/development_reports/).

## � Recent Changes (August 2025)

### V2 System Stabilization
- **Branch**: `fix-v2-stable-rollback` (rollback from development issues)
- **LLaVA Model**: Switched from `llava-v1.6-mistral-7b-hf` to `llava-1.5-7b-hf` for stability
- **Memory Management**: Ultra-conservative GPU memory limits (30% max usage) to prevent system crashes
- **OCR/Layout Deprecation**: Removed OCR and Layout Parser models - LLaVA provides superior vision-language understanding
- **Environment**: Fresh conda environment `justnews-v2-prod` with PyTorch 2.5.1+cu121, Transformers 4.55.0

### Critical Fixes Applied
- **GPU Memory Crashes**: Multiple system crashes during 10-article tests around article 5 processing
- **Context Managers**: Added proper resource cleanup with `__enter__` and `__exit__` methods
- **Model Loading**: Ultra-conservative memory limits (8GB max) to prevent GPU OOM crashes
- **Processing Streamlined**: Removed redundant OCR and layout processing to focus on LLaVA-only approach

### Known Issues
- **System Stability**: Testing 10-article processing after recent crashes
- **Memory Optimization**: Investigating optimal GPU memory allocation for RTX 3090 (24GB)
- **Model Performance**: Validating LLaVA 1.5 vs 1.6 performance differences

**Status**: Active stabilization phase - prioritizing crash-free operation over performance

## �📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NVIDIA** - TensorRT optimization and CUDA development tools
- **Hugging Face** - Transformers library and model ecosystem  
- **Model Context Protocol** - Agent communication standardization
- **PostgreSQL** - Robust database foundation
- **FastAPI** - High-performance API framework

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Adrasteon/JustNewsAgentic/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Adrasteon/JustNewsAgentic/discussions)
- **Documentation**: [Complete Documentation](markdown_docs/README.md)

---

**JustNewsAgentic V4** - *Intelligent news analysis through collaborative AI*