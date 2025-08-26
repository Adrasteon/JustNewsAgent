# JustNewsAgentic V4 🤖

```markdown
# JustNewsAgent

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![TensorRT](https://img.shields.io/badge/TensorRT-Production-orange.svg)](https://developer.nvidia.com/tensorrt)

AI-powered news analysis system using a distributed multi-agent architecture, GPU acceleration, and continuous learning.

Overview
--------

JustNewsAgent (V4) is a modular multi-agent system that discovers, analyzes, verifies, and synthesizes news content. Agents communicate via the Model Context Protocol (MCP). The project emphasizes performance (native TensorRT acceleration), modularity, and operational observability.

Highlights
----------
- Multi-agent architecture with specialized agents for crawling, analysis, fact-checking, synthesis, and storage
- Native TensorRT GPU optimizations for model inference
- MCP (Model Context Protocol) for standardized inter-agent communication
- Continuous online training pipeline for incremental model improvements
- PostgreSQL-backed vector storage for semantic search

Quick links
-----------
- Documentation: `markdown_docs/README.md`
- Agent guides: `markdown_docs/agent_documentation/`
- Technical reports: `markdown_docs/development_reports/`
- License: `LICENSE`

Quick start
-----------

```markdown
# JustNewsAgent

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![TensorRT](https://img.shields.io/badge/TensorRT-Production-orange.svg)](https://developer.nvidia.com/tensorrt)

AI-powered news analysis system using a distributed multi-agent architecture, GPU acceleration, and continuous learning.

Overview
--------

JustNewsAgent (V4) is a modular multi-agent system that discovers, analyzes, verifies, and synthesizes news content. Agents communicate via the Model Context Protocol (MCP). The project emphasizes performance (native TensorRT acceleration), modularity, and operational observability.

Highlights
----------
- Multi-agent architecture with specialized agents for crawling, analysis, fact-checking, synthesis, and storage
- Native TensorRT GPU optimizations for model inference
- MCP (Model Context Protocol) for standardized inter-agent communication
- Continuous online training pipeline for incremental model improvements
- PostgreSQL-backed vector storage for semantic search

Quick links
-----------
- Documentation: `markdown_docs/README.md`
- Agent guides: `markdown_docs/agent_documentation/`
- Technical reports: `markdown_docs/development_reports/`
- License: `LICENSE`

Quick start
-----------

Prerequisites
-------------
- Linux (Ubuntu recommended)
- Python 3.12+
- Optional: NVIDIA GPU with CUDA 12.1+ for acceleration (RTX 3090/4090 recommended)
- Conda or virtualenv for environment management

Installation
------------

1. Clone the repository

```bash
git clone https://github.com/Adrasteon/JustNewsAgent.git
cd JustNewsAgent
```

2. Create and activate a Python environment

```bash
conda create -n justnews python=3.12 -y
conda activate justnews
pip install -r tests/requirements.txt
```

3. (Optional) GPU setup

- Install NVIDIA drivers, CUDA 12.1, and NVIDIA Container Toolkit if you plan to use GPU acceleration

Starting the system (development)
-------------------------------

This repository contains multiple agent services. For development, run individual agents using their FastAPI entrypoints (see `agents/<agent>/main.py`). A convenience script is available for local runs (may require customization):

```bash
./scripts/run_ultra_fast_crawl_and_store.py
# or run a single agent
python -m agents.mcp_bus.main
```

Usage examples
--------------

Check MCP bus agents list

```bash
curl http://localhost:8000/agents
```

Analyze a single article (example agent endpoint)

```bash
curl -X POST http://localhost:8002/enhanced_deepcrawl \
	-H "Content-Type: application/json" \
	-d '{"args": ["https://www.bbc.com/news/example"], "kwargs": {}}'
```

Configuration
-------------

Core configuration is controlled via environment variables. Example values:

```bash
MCP_BUS_URL=http://localhost:8000
GPU_MEMORY_FRACTION=0.8
BATCH_SIZE=32
DATABASE_URL=postgresql://user:password@localhost:5432/justnews
```

Agent configuration files are located in each agent folder (e.g. `agents/synthesizer/`).

Contributing
------------

Contributions are welcome. Please follow the workflow below:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes with clear messages
4. Push to your fork and open a PR against `main`

Developer setup (short)

```bash
conda activate justnews
pip install -r tests/requirements.txt
pre-commit install
pytest tests/
```

Security & privacy
------------------

- The project is designed for local processing. Configure access controls and secure your database credentials.
- Validate and checksum any models you download before use.

Roadmap
-------

- Continue expanding agent capabilities and improving training integration
- Add multi-node deployment and easier local dev workflows
- Improve documentation and API references

Support & contacts
------------------

- Issues: https://github.com/Adrasteon/JustNewsAgent/issues
- Documentation: `markdown_docs/README.md`

License
-------

This project is licensed under the Apache 2.0 License — see the `LICENSE` file for details.

Acknowledgments
---------------

- NVIDIA, Hugging Face, PostgreSQL, FastAPI, and the open-source community for libraries and tooling.

```