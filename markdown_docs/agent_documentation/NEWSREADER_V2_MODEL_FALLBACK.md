---
title: NewsReader V2 Vision-Language Model Fallback Logic
description: Auto-generated description for NewsReader V2 Vision-Language Model Fallback Logic
tags: [documentation]
status: current
last_updated: 2025-09-12
---

# NewsReader V2 Vision-Language Model Fallback Logic

## Overview
The NewsReader V2 agent now implements robust fallback logic for vision-language model initialization. If the primary LLaVA model fails to load, the agent automatically attempts to load BLIP-2 as a fallback. This ensures reliable screenshot-based article extraction even if the preferred model is unavailable or fails due to resource constraints.

## Implementation Details
- **Primary Model:** LLaVA (LLaVA-Next)
- **Fallback Model:** BLIP-2 (Salesforce/blip2-opt-2.7b)
- **Logic:**
    - On agent startup, attempts to load LLaVA.
    - If LLaVA fails, logs warning and attempts BLIP-2.
    - If both fail, logs error and disables vision-language extraction.
- **Error Handling:**
    - All model loading exceptions are logged with details.
    - GPU memory usage is monitored and logged.
    - Fallback logic is fully MCP-compliant and production-grade.

## Deprecation Notes
- **easyocr** and **layoutparser** are deprecated and not required for NewsReader V2 operation.
- All references to these libraries are commented out in requirements and code.

## References
- See `agents/newsreader/newsreader_v2_true_engine.py` for implementation.
- See `agents/newsreader/requirements.txt` for dependency notes.
- See `markdown_docs/optimization_reports/OCR_REDUNDANCY_ANALYSIS.md` for analysis of OCR redundancy.

---
Last updated: 2025-08-12

## See also

- Technical Architecture: markdown_docs/TECHNICAL_ARCHITECTURE.md
- Documentation Catalogue: docs/DOCUMENTATION_CATALOGUE.md

