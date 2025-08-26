# JustNewsAgentic — Implementation Plan for Evidence, KG, Fact‑Checking & Conservative Generation

Date: 2025-08-21  
Branch: experimental-stripdown

This document records the design and implementation plan for the evidence ledger, knowledge graph (KG), fact-checker, conservative generator (article contract), multimedia forensics, source registry, and editorial UI discussed earlier. Use this as a reference for incremental implementation and testing.

---

## 1. Goals (high level)
- Build a provable, auditable pipeline that produces evidence-backed, neutral news articles.
- Ensure every factual claim links to recorded evidence (snapshots + metadata).
- Use a KG / neuro-symbolic layer for factual grounding and contradiction detection.
- Provide human-in-the-loop editorial controls and an exportable audit bundle.
- Keep CI and tests independent of external LLM providers (mock LLM clients for tests).

---

## 2. Checklist of features in this plan
- Evidence ledger (SQLite + raw snapshots)
- Knowledge Graph starter (rdflib; Turtle persistence)
- Fact-checker agent (KG-first verification; LLM fallback)
- Source registry & scoring (domain-level heuristics + cache)
- Multimedia forensics scaffolding (image/video/audio)
- Conservative generator & article contract (article dataclass + evidence linking)
- Editorial UI (FastAPI stub)
- Tests (pytest) using MockLLM and local sample artifacts

---

## 3. Data models (summary)
- Evidence (JSON schema)
  - id, url, snapshot_hash, timestamp, extractor, text_snippet, start_char, end_char, metadata, confidence
- ClaimVerificationResult
  - claim, verdict ('true'|'false'|'uncertain'), evidence_ids, confidence, notes
- Article
  - id, title, lede, body_paragraphs (text + evidence_ids), claims_table, provenance_bundle, generated_at

(See `agents/types.py` for dataclass suggestions.)

---

## 4. Module contracts & locations
- `agents/evidence_store.py`
  - record_evidence(evidence: Evidence, raw_html: Optional[str]) -> str
  - get_evidence(evidence_id: str) -> Optional[Evidence]
  - query_evidence_by_url(url: str) -> List[Evidence]
  - export_evidence_bundle(evidence_ids: List[str], dest_dir: str) -> str
  - Storage: `memory_v2_vectordb/evidence_store.sqlite`; raw snapshots -> `memory_v2_vectordb/evidence_raw/`

- `kg/loader.py`, `kg/query.py`, `kg/rules.py`
  - ingest_evidence(evidence: Evidence) -> List[str]  # returns created node URIs
  - add_claim(claim_id, claim_text, evidence_ids) -> str
  - run_rules() -> List[dict]  # flagged contradictions, inconsistencies
  - Storage: `kg/graph.ttl` (rdflib Turtle file). Start with `rdflib` then migrate to Neo4j/Dgraph for scale.

- `agents/fact_checker.py`
  - verify_claim(claim_text: str, context_urls: Optional[List[str]] = None) -> ClaimVerificationResult
  - Strategy: 1) Canonicalize claim 2) Query KG 3) Vector retrieval of evidence 4) Deterministic rules 5) LLM fallback (injected client)

- `sources/registry.py`
  - register_source(url) -> domain
  - score_source(domain) -> float
  - update_source_features(domain, features: dict)
  - Storage: `memory_v2_vectordb/sources.sqlite`

- `multimedia/` (image_forensics.py, video_forensics.py, audio_forensics.py)
  - Image: EXIF extraction, pHash, simple manipulation detection
  - Video: frame extraction, basic face/frame checks
  - Audio: fingerprinting basics

- `agents/generator.py` and `agents/article.py`
  - generate_article(topic, claims, evidence_ids, llm_client) -> Article
  - Article output must include inline evidence IDs and a claims table; enforce conservative style policies.

- `agents/editor_ui.py`
  - FastAPI stub with endpoints to list drafts, view draft, approve/reject, export audit bundle.

---

## 5. Testing & CI rules
- All unit tests must mock any LLM client (inject MockLLM).
- Integration smoke test (local only):
  1. Use a local HTML sample (or one saved snapshot).
  2. Create an Evidence record with raw HTML.
  3. Ingest into KG and create a test claim.
  4. Run `fact_checker.verify_claim()` with MockLLM fallback.
  5. Run `generator.generate_article()` to ensure claims table and evidence links exist.
- Add tests under `tests/`:
  - `tests/test_evidence_store.py`
  - `tests/test_kg_ingest.py`
  - `tests/test_fact_checker.py`
- CI: run `pytest -q` with the venv; do not call external networks.

---

## 6. Implementation milestones (prioritized)
Sprint 0 — Evidence ledger (3 days)
- Implement `agents/evidence_store.py` and `agents/types.py`.
- Add tests to record/get evidence and snapshot writing.

Sprint 1 — KG ingest & rules (5 days)
- `kg/loader.py`, `kg/rules.py`. Add a sample rule (date/number contradiction).

Sprint 2 — Fact-checker (5 days)
- `agents/fact_checker.py` using KG-first pipeline and injected LLM client.

Sprint 3 — Generator + Editorial UI (7 days)
- `agents/generator.py`, `agents/article.py`, `agents/editor_ui.py`. Add conservative generation rules.

Sprint 4 — Multimedia forensics (7–10 days)
- Build `multimedia/*` modules and integrate forensic outputs into Evidence records.

Sprint 5 — Evaluation & CI (5 days)
- Add FEVER-style evaluation harness, mock LLMs, metrics collection.

---

## 7. Example commands (dev)
Create and run the editorial UI (dev):
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r [`requirements.txt`](requirements.txt )
python -m uvicorn agents.editor_ui:app --reload --port 8010
```

Run tests:
```bash
pytest -q
```

Add a local smoke script pattern:
```bash
# scripts/run_local_smoke.sh (example)
python scripts/smoke_demo.py  # demo uses MockLLM and local sample article
```

---

## 8. Security & legal notes
- Verify model licenses before using any open-source LLM (Llama/Grok/others).
- Respect robots.txt and publishers' Terms of Service for scraping; log and audit scraped sources.
- Treat PII carefully: redact or secure PII found during ingestion per policy.
- Maintain exportable evidence bundles to support audits.

---

## 9. Next steps (recommended immediate actions)
1. Implement Sprint 0: create `agents/evidence_store.py`, `agents/types.py`, add `tests/test_evidence_store.py`.
2. Run tests locally (they should use local sample content and MockLLM).
3. Iterate: add minimal KG ingestion and a single rule to detect date conflicts.

---

## 10. Contact & provenance
- This file was produced programmatically on 2025-08-21 as the reference design for the features scoped to JustNewsAgentic's `experimental-stripdown` branch.
- For implementation help, copy the scaffolding code from the plan into the repo, run tests, and request follow-up changes.

---
