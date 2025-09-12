---
title: Centralized Logging Migration Guide
description: Auto-generated description for Centralized Logging Migration Guide
tags: [documentation]
status: current
last_updated: 2025-09-12
---

# Centralized Logging Migration Guide

## Overview
JustNewsAgent now has a centralized logging system that provides:
- Structured JSON logging for production
- Automatic log rotation and file management
- Environment-specific configuration
- Performance and error tracking
- Agent-specific log files

## Migration Steps

### 1. Replace Basic Logging Setup
**BEFORE:**
```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

**AFTER:**
```python
from common.observability import get_logger
logger = get_logger(__name__)
```

### 2. Use Performance Logging
**BEFORE:**
```python
start_time = time.time()
# ... your code ...
logger.info(f"Operation took {time.time() - start_time:.3f}s")
```

**AFTER:**
```python
from common.observability import log_performance
start_time = time.time()
# ... your code ...
log_performance("your_operation", time.time() - start_time)
```

### 3. Use Error Logging
**BEFORE:**
```python
try:
    # ... your code ...
except Exception as e:
    logger.error(f"Error: {e}", exc_info=True)
```

**AFTER:**
```python
from common.observability import log_error
try:
    # ... your code ...
except Exception as e:
    log_error(e, "operation_context")
```

## Configuration

### Environment Variables
- `LOG_LEVEL`: DEBUG, INFO, WARNING, ERROR
- `LOG_FORMAT`: structured (JSON) or readable (human)
- `LOG_DIR`: Directory for log files (default: ./logs)
- `LOG_MAX_BYTES`: Max size per log file (default: 10MB)
- `LOG_BACKUP_COUNT`: Number of backup files (default: 5)

### Log Files Created
- `justnews.log`: General application logs
- `justnews_error.log`: Error-only logs
- `scout.log`, `analyst.log`, etc.: Agent-specific logs

## Benefits
- ✅ Consistent logging across all modules
- ✅ Automatic log rotation and cleanup
- ✅ Structured logs for production monitoring
- ✅ Performance tracking built-in
- ✅ Easy debugging with agent-specific logs

## See also

- Technical Architecture: markdown_docs/TECHNICAL_ARCHITECTURE.md
- Documentation Catalogue: docs/DOCUMENTATION_CATALOGUE.md

