# Action Plan: JustNews V4 RTX-Accelerated Development

**Current Status**: Enhanced Scout Agent + TensorRT-LLM Integration Complete - Ready for Multi-Agent GPU Expansion

This action plan outlines the next phases for JustNews V4 development now that both the RTX AI Toolkit foundation and Enhanced Scout Agent integration are operational.

---

## ✅ Phase 0: RTX Foundation (COMPLETED - July 26, 2025)

### Infrastructure Complete
- **TensorRT-LLM 0.20.0**: ✅ Fully operational on RTX 3090
- **NVIDIA RAPIDS 25.6.0**: ✅ GPU data processing suite ready
- **Hardware Validation**: ✅ RTX 3090 performance confirmed (24GB VRAM)
- **Environment Setup**: ✅ Professional-grade GPU stability

### Test Results: 6/6 PASS (100% Success Rate)
- Basic Imports, CUDA Support, MPI Support, TensorRT, Transformers, TensorRT-LLM

---

## ✅ Phase 0.5: Enhanced Scout Agent Integration (COMPLETED - July 29, 2025)

### Native Crawl4AI Integration Complete
- **BestFirstCrawlingStrategy**: ✅ Native Crawl4AI 0.7.2 integration deployed
- **Scout Intelligence Engine**: ✅ LLaMA-3-8B GPU-accelerated content analysis
- **User Parameters**: ✅ max_depth=3, max_pages=100, word_count_threshold=500 implemented
- **Quality Filtering**: ✅ Dynamic threshold-based content selection operational
- **MCP Bus Integration**: ✅ Full agent registration and communication validated

### Performance Validation Complete
- **Sky News Test**: ✅ 148k characters crawled in 1.3 seconds
- **Scout Intelligence**: ✅ Content analysis with quality scoring operational
- **Integration Testing**: ✅ MCP Bus and direct API validation completed
- **Production Ready**: ✅ Enhanced deep crawl functionality fully operational

---

## 🚀 Phase 1: Model Integration (CURRENT PRIORITY)

### 1.1 Download Optimized Models
- **Primary Focus**: News analysis models optimized for TensorRT-LLM
  - BERT variants for sentiment analysis and classification
  - Summarization models (T5, BART variants)
  - Named Entity Recognition models for news processing
- **Quantization**: Apply INT4_AWQ for 3x compression without quality loss
- **Timeline**: 2-3 days

### 1.2 Engine Building
- Convert models to TensorRT engines optimized for RTX 3090
- Test inference performance with target 10-20x speedup
- Implement model caching and management
- **Timeline**: 3-5 days

---

## 🔧 Phase 2: Multi-Agent GPU Expansion (HIGH PRIORITY)

### 2.1 Fact Checker Agent GPU Enhancement
- **Integrate GPU-accelerated claim verification** using TensorRT-LLM
- **Implement Scout Intelligence pre-filtering** for optimized downstream processing
- **Add performance monitoring** with real-time metrics
- **Hybrid routing**: GPU primary, Docker fallback
- **Timeline**: 4-6 days

### 2.2 Synthesizer Agent GPU Enhancement
- **Migrate clustering to GPU** using RAPIDS cuML
- **Implement TensorRT-LLM content synthesis** with batch processing
- **Add Scout pre-filtered content handling** for efficiency gains
- **Performance optimization**: GPU memory management and batching
- **Timeline**: 5-7 days

### 2.3 Critic Agent GPU Enhancement
- **Implement GPU-accelerated quality assessment** using TensorRT-LLM
- **Integrate with Scout Intelligence scoring** for consistent quality metrics
- **Add real-time performance monitoring** and feedback loops
- **Batch processing**: Optimize for RTX 3090 memory utilization
- **Timeline**: 4-5 days

---

## 2. Scout Agent
- **Replace stubs with real implementations:**
  - Integrate a real web search API (e.g., Google/Bing Custom Search, SerpAPI) for `discover_sources`.
  - Implement robust web crawling and content extraction for `crawl_url` and `deep_crawl_site`.
- **Add error handling and feedback logging:**
  - Log failed searches/crawls and user feedback for continual improvement.
- **Support extraction prompts:**
  - Allow custom extraction prompts to guide content extraction.
- **Testing:**
  - Add tests for search, crawl, and extraction logic.

---

## 3. Fact-Checker Agent
- **Replace rule-based logic with ML/LLM:**
  - Use an LLM or claim verification model for `validate_is_news` and `verify_claims`.
  - Integrate with external fact-checking APIs if available.
- **Add feedback logging:**
  - Log fact-check outcomes and user/editor feedback for retraining.
- **Testing:**
  - Add tests for claim validation and verification.

---

## 4. Analyst Agent
- **Replace rule-based logic with ML/LLM:**
  - Use LLM or ML models for `score_bias`, `score_sentiment`, and `identify_entities`.
  - Integrate with NER and sentiment analysis libraries (spaCy, transformers, etc.).
- **Add feedback logging:**
  - Log analysis results and feedback for model improvement.
- **Testing:**
  - Add tests for bias, sentiment, and entity recognition.

---

## 5. Synthesizer Agent
- **Enhance clustering and aggregation:**
  - Add error handling and validation for clustering and LLM calls.
  - Support additional clustering algorithms (BERTopic, HDBSCAN).
- **Ensure feedback loop is used in retraining:**
  - Automate periodic retraining using logged feedback.
- **Testing:**
  - Add tests for clustering, neutralization, and aggregation.

---

## 6. Critic Agent
- **Enhance critique logic:**
  - Add error handling for LLM pipeline.
  - Integrate optional fact-checking pipeline for cross-referencing.
- **Ensure feedback loop is used in retraining:**
  - Automate periodic retraining using logged feedback.
- **Testing:**
  - Add tests for critique synthesis and neutrality.

---

## 7. Memory Agent
- **Clarify tool interface:**
  - Move or mirror key tool interfaces from `main.py` to `tools.py` for clarity and maintainability.
- **Enhance error handling:**
  - Add robust error handling for DB and embedding/model calls.
- **Ensure feedback loop is used for learning-to-rank:**
  - Use logged retrievals and outcomes to improve ranking models.
- **Testing:**
  - Add tests for semantic retrieval, vector search, and feedback logging.

---

## 8. General/All Agents
- **Documentation:**
  - Update docstrings and README sections for all new/changed logic.
- **Feedback Loop:**
  - Standardize feedback logging format and location across agents.
  - Document retraining and continual learning procedures.
- **CI/CD:**
  - Add/expand tests to cover new ML/LLM logic and feedback mechanisms.

---

## 9. Timeline & Milestones
1. **Week 1:** Replace stubs/rule-based logic in Scout, Fact-Checker, Analyst. Add feedback logging to all agents.
2. **Week 2:** Implement real MCP bus integration for Chief Editor. Enhance orchestration and error handling.
3. **Week 3:** Expand clustering/aggregation in Synthesizer. Add fact-checking pipeline to Critic. Move tool interfaces in Memory.
4. **Week 4:** Standardize feedback loop, automate retraining, finalize documentation, and expand tests.

---

## 10. Success Criteria
- All agents use ML/LLM-based logic for their core tools.
- All feedback is logged and used for continual learning.
- All stubs and rule-based placeholders are replaced.
- Documentation and tests are up to date.

---

*For details, see the latest `CHANGELOG.md`, `JustNews_Plan_V3.md`, and `JustNews_Proposal_V3.md`.*
