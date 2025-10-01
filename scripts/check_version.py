#!/usr/bin/env python3
"""
Version validation script for JustNewsAgent
"""

import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from justnews import VERSION_INFO, __version__

    print(f"✅ JustNewsAgent Version: {__version__}")
    print(f"📊 Status: {VERSION_INFO['status']}")
    print(f"📅 Release Date: {VERSION_INFO['release_date']}")
    print(f"📝 Description: {VERSION_INFO['description']}")
    sys.exit(0)
except ImportError as e:
    print(f"❌ Version import failed: {e}")
    print(f"   Project root: {project_root}")
    print(f"   Python path: {sys.path}")
    sys.exit(1)
