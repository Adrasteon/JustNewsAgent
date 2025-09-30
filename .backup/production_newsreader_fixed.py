#!/usr/bin/env python3
"""
Production NewsReader Fixed - GPU Crash Resolved Implementation

This is a production-ready version of the NewsReader that uses the crash-resolved
GPU configuration with proper BitsAndBytesConfig and conversation formatting.

Key Features:
- BitsAndBytesConfig INT8 quantization (not torch_dtype=torch.int8)
- Correct LLaVA conversation template usage
- Memory management and cleanup
- Production fallback strategies
"""

import sys
from pathlib import Path

# Add the newsreader agent path
newsreader_path = Path(__file__).parent / "agents" / "newsreader" / "main_options"
sys.path.insert(0, str(newsreader_path))

try:
    from practical_newsreader_solution import (
        PracticalNewsReader as ProductionNewsReader,
    )
except ImportError:
    # Fallback to the main newsreader agent
    newsreader_agent_path = Path(__file__).parent / "agents" / "newsreader"
    sys.path.insert(0, str(newsreader_agent_path))
    from newsreader_agent import PracticalNewsReader as ProductionNewsReader

__all__ = ["ProductionNewsReader"]
