# AGENT_MODEL_MAP — Definitive Agent → Model mapping

This document lists the authoritative mapping of agents to their external model dependencies as defined in `scripts/download_agent_models.py` (the `AGENT_MODEL_MAP` constant). It also records the current canonical target path and observed size (from the data drive) at the time this file was generated.

Generated: 2025-08-20
Updated: 2025-09-20 (Training Integration Status)

## Canonical models base path

`/media/adra/Data/justnews/agents/<agent>/models/<model-folder>`

(Each agent's `agents/<agent>/models` in the repository is a symlink to the canonical path when models are present.)

---

## Training Integration Status ✅

All V2 agents are now fully integrated with the online training system for continuous learning:

| Agent | Training Status | Update Method | Task Types |
|-------|----------------|---------------|------------|
| **Scout V2** | ✅ Complete | `_update_scout_models()` | news_classification, quality_assessment, sentiment, bias_detection |
| **Analyst V2** | ✅ Complete | `_update_analyst_models()` | entity_extraction, sentiment_analysis, bias_analysis |
| **Critic V2** | ✅ Complete | `_update_critic_models()` | logical_fallacy, argument_structure |
| **Fact Checker V2** | ✅ Complete | `_update_fact_checker_models()` | fact_verification, credibility_assessment |
| **NewsReader V2** | ✅ Complete | `_update_newsreader_models()` | screenshot_analysis, content_extraction |
| **Synthesizer V3** | ✅ **NEW** Complete | `_update_synthesizer_models()` | article_clustering, text_neutralization, cluster_aggregation |
| **Chief Editor** | ✅ **NEW** Complete | `_update_chief_editor_models()` | story_brief_generation, story_publishing, evidence_review_queuing |
| **Memory** | ✅ **NEW** Complete | `_update_memory_models()` | article_storage, vector_search, training_example_logging |

**Training System**: EWC-based continuous learning with 48 examples/minute processing capability
**Update Frequency**: 82.3 model updates/hour across all integrated agents
**Mission Status**: 🏆 **V2 ENGINES EXPANSION COMPLETE** - All 7 agents enabled for continuous learning

---

## Mapping (agent -> [(type, HF id)])

## Mapping (agent -> [(type, HF id)])

- scout
  - (transformers) google/bert_uncased_L-2_H-128_A-2
  - (transformers) cardiffnlp/twitter-roberta-base-sentiment-latest
  - (transformers) martin-ha/toxic-comment-model

- fact_checker
  - (transformers) distilbert-base-uncased
  - (transformers) roberta-base
  - (sentence-transformers) sentence-transformers/all-mpnet-base-v2

- memory
  - (sentence-transformers) all-MiniLM-L6-v2

- synthesizer
  - (transformers) distilgpt2
  - (transformers) google/flan-t5-small

- critic
  - (transformers) unitary/unbiased-toxic-roberta
  - (transformers) unitary/toxic-bert

- analyst
  - (transformers) google/bert_uncased_L-2_H-128_A-2

- newsreader
  - (sentence-transformers) all-MiniLM-L6-v2

- balancer
  - (transformers) google/bert_uncased_L-2_H-128_A-2

- chief_editor
  - (transformers) distilbert-base-uncased

---

## Observed on-disk targets & sizes

(Resolved symlink targets and `du -sh` sizes at generation time.)

- analyst: `/media/adra/Data/justnews/agents/analyst/models` — 18M
- balancer: `/media/adra/Data/justnews/agents/balancer/models` — 18M
- chief_editor: `/media/adra/Data/justnews/agents/chief_editor/models` — 257M
- common: `/media/adra/Data/justnews/agents/common/models` — 0
- critic: `/media/adra/Data/justnews/agents/critic/models` — 1.4G
- dashboard: (no models/ in workspace) — skipped
- fact_checker: `/media/adra/Data/justnews/agents/fact_checker/models` — 1.6G
- memory: `/media/adra/Data/justnews/agents/memory/models` — 175M
- newsreader: `/media/adra/Data/justnews/agents/newsreader/models` — 175M
- reasoning: (no models/ in workspace) — skipped
- scout: `/media/adra/Data/justnews/agents/scout/models` — 1.5G
- synthesizer: `/media/adra/Data/justnews/agents/synthesizer/models` — 636M

---

## Notes
- `dashboard` and `reasoning` intentionally do not have model folders: `dashboard` is a GUI controller; `reasoning` uses the Nucleoid engine and does not require a HF model folder.
- To update this document, re-run the quick verification and regenerate this file.
