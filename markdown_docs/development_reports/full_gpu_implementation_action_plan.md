# Full GPU Implementation Action Plan

Goal: take JustNewsAgent from the current hybrid/partial TensorRT implementation to a robust, reproducible, production-ready GPU-enhanced system that uses the central Model Store and respects the repo's updated ingestion/canonicalization DB schema.

Scope and constraints
- Docker is deprecated for this project: the plan uses container-free, host-native builds or controlled VM images (OCI images for ops are noted but not required).
- Use the central `ModelStore` (see `agents/common/model_store.py` and `markdown_docs/agent_documentation/MODEL_STORE_GUIDELINES.md`) as the canonical source for model artifacts (tokenizers, ONNX artifacts, engine artifacts).
- Preserve the ingestion/evidence-first workflow: engine-driven outputs that affect canonical selection or editorial decisions must be recorded in the evidence trail (article_source_map, evidence manifest). See `agents/common/ingest.py` and `deploy/sql/canonical_selection.sql`.

High-level Phases (ordered)
1. Developer & CI safety (quick wins)
2. Reproducible HF → ONNX → TensorRT build pipeline (host-native, non-Docker) + INT8 calibration
3. Engine artifact management & ModelStore integration
4. Runtime & multi-GPU deployment patterns (pinning, process per GPU, context safety)
5. Tests, benchmarks and operational runbooks
6. Production rollout & monitoring

Phase 1 — Developer & CI safety (0.5–2 days)
Actions:
- Add and run non-GPU-friendly checks in CI. Use marker-engine approach so default CI runners pass:
  - `scripts/compile_tensorrt_stub.py --check-only` and `--build-markers` (stub exists in `scripts/`).
  - Add CI job `ci/tensorrt-check.yml` that runs the stub and unit tests in a non-GPU environment.
- Add unit tests that mock missing native packages:
  - `tests/test_tensorrt_stub.py` — marker creation verification.
  - `tests/test_native_compiler_mocked.py` — ensure compiler behaves correctly when `tensorrt`/`pycuda` are absent.

Why first: prevents CI from breaking, allows everyday development without GPUs, and enables automated safety gates.

Phase 2 — Reproducible build pipeline (HF → ONNX → TRT) (3–6 days)
Design constraints & assumptions:
- No Docker: build pipeline must be host-native or run inside a controlled VM image. Provide an optional containerized VM image recipe for ops (OCI artifacts) but the canonical tooling expects a developer/ops host with known CUDA/TensorRT versions.
- Use pinned, documented toolchain versions (CUDA, cuDNN, TensorRT, tensorrt-llm, PyTorch, transformers).

Actions:
- Create `tools/build_engine/` with:
  - `build_engine.py` — CLI wrapper that orchestrates:
    - Fetch model from HF or `ModelStore` (prefer `ModelStore` via `agents/common/model_store.py`).
    - Convert HF model to ONNX (with dynamic axes where appropriate).
    - Run host-native TRT build using `tensorrt`/`trt.Builder` and `tensorrt-llm` when applicable.
    - Emit `.engine` binary and a metadata JSON (naming/fields described below).
  - `build_engine.local.sh` — example script to run on a GPU host.
  - `README.md` listing exact required versions and environment setup steps.
- Implement ONNX conversion robustness:
  - Use `native_tensorrt_compiler.py` functions as the canonical code path, but wrap them in the new CLI with clearly documented flags: `--precision {fp32,fp16,int8}`, `--max-batch`, `--sequence-length`, `--calibrate <calib-dataset>`.
- Calibration flow for INT8:
  - Add a `calibration/` helper to collect representative inputs from a sample article set and produce an INT8 calibration cache.
  - CLI flag `--calibrate` triggers calibration run and saves calibration cache (used by TRT builder).

Deliverable acceptance:
- A host-native run produces a valid `.engine` and `.json` metadata on a GPU dev host with pinned versions.

Phase 3 — Engine artifact management & ModelStore integration (1–2 days)
Actions:
- Define engine naming and metadata schema (enforce via `tools/build_engine/verify_engine.py`):
  - Engine filename pattern: `<task>.<model>-<hf-rev>-trt-<trt-ver>-<precision>.engine`
  - Metadata JSON: { model_name, hf_revision, trt_version, precision, build_options, max_batch_size, seq_len, checksum, created_at }
- Integrate with `ModelStore` APIs:
  - Build CLI should prefer uploading outputs to `ModelStore` with atomic finalize (use `agents/common/model_store.py`).
  - Runtime processes must read engines and tokenizers from `MODEL_STORE_ROOT`/`ModelStore` symlink.

Why: explicit artifact versioning avoids runtime mismatches and supports auditability.

Phase 4 — Runtime & multi-GPU deployment patterns (2–4 days)
Actions:
- Robust runtime loader improvements:
  - Ensure `rtx_manager.py` and `native_tensorrt_engine.py` read metadata JSON and verify compatibility before loading an engine.
  - Add a `verify_engine_compatibility(engine_path, runtime_trt_version)` function to return safe errors.
- Multi-GPU strategies:
  - Process-per-device: recommended default — run N worker processes each pinned to a different GPU (ensures isolated CUDA contexts and simple lifecycle management).
  - Engine-to-device mapping file: provide `conf/engine_device_map.yaml` mapping engine name → device id.
  - Optional: a lightweight device pool manager in `agents/analyst/device_manager.py` to allocate contexts when process-per-device is not feasible.
- Resource safety:
  - Ensure `NativeTensorRTInferenceEngine` and `GPUAcceleratedAnalyst` expose `cleanup()` and safe context teardown for systemd/healthchecks.

Phase 5 — Tests, benchmarks and QA (2–4 days)
Actions:
- Add unit/integration tests:
  - Mocked RTT tests for `native_tensorrt_engine` (simulate `tensorrt` and `pycuda` APIs).
  - Smoke integration `tests/smoke_tensorrt_runtime_marker.py` that uses marker `.engine` files and exercises `tensorrt_tools.get_tensorrt_engine()` path.
- Benchmarks:
  - Add `benchmarks/` scripts to measure throughput/latency for: native engines, TRT-framework mode, fallback HF pipelines.
  - Record and save benchmark artifacts in `logs/benchmarks/` for comparison.

Phase 6 — Production rollout & monitoring (ongoing)
Actions:
- Gradual rollout plan:
  - Canary on a small number of servers using production traffic with A/B (native vs fallback)
  - Observe canonical selection/confidence deltas and evidence logs to ensure no negative impact.
- Monitoring & telemetry:
  - Integrate `rtx_manager._log_performance` outputs into central observability (Prometheus metrics or log aggregation), and ensure GPU health and memory metrics are exported.
  - Record model id and version in evidence trail any time a model's output influences editorial decisions or canonical selection (add fields to evidence manifest). See `agents/common/evidence.py` and `agents/chief_editor/handler.py`.

Cross-cutting requirements
- ModelStore behavior:
  - All build artifacts (ONNX, engines, metadata) are placed into `ModelStore` with atomic finalize. Runtimes read from `MODEL_STORE_ROOT` or `ModelStore` symlink (see `agents/common/model_store.py`).
- Database/evidence integration:
  - Whenever model outputs change `article_source_map` scoring or canonical selection, write a stable evidence manifest and enqueue a review event (use `agents/common/evidence.py` patterns). Log model id/version in the evidence manifest.
  - Coordinate schema migrations with the DB team if new columns are required (e.g., `article_source_map.model_id`, `article_source_map.model_version`).
- Security & reproducibility:
  - Pin versions of TensorRT, cuDNN, PyTorch and tensorrt-llm in the `tools/build_engine/README.md` to ensure reproducible binary engines.
  - Create a `tools/toolchain_versions.md` manifest listing tested versions.

Risk mitigation and fallbacks
- If a real TRT build cannot be run on a host, use marker-engines and the HuggingFace GPU fallback path (already present in `hybrid_tools_v4.py` and `tensorrt_acceleration.py`).
- Keep Docker Model Runner fallback logic for model-serving via HTTP where `rtx_manager` already supports a docker (but do not create new Docker-based flows — mark as deprecated).

Appendix — Concrete file/action checklist (first sprint)
- Add CI job `ci/tensorrt-check.yml` (create file)
- Add tests:
  - `tests/test_tensorrt_stub.py`
  - `tests/test_native_compiler_mocked.py`
  - `tests/test_native_engine_mocked.py`
  - `tests/smoke_tensorrt_runtime_marker.py`
- Add tooling and docs:
  - `tools/build_engine/build_engine.py` (host-native CLI)
  - `tools/build_engine/README.md` (versions & steps)
  - `tools/build_engine/verify_engine.py`
  - `conf/engine_device_map.yaml` (example)
  - `scripts/compile_tensorrt_stub.py` (already added)

Estimated timeline (conservative)
- Sprint 0 (1–3 days): CI + tests (Phase 1) — green CI for non-GPU runners
- Sprint 1 (3–7 days): build pipeline prototype (Phase 2) + metadata & ModelStore upload (Phase 3)
- Sprint 2 (3–7 days): calibration + runtime multi-GPU patterns (Phase 4)
- Sprint 3 (2–5 days): tests, benchmarks, ops runbook, slow rollout (Phases 5–6)

Next step (recommended): I will create the minimal CI job and the unit tests in Phase 1 so we have a safe developer/test baseline. Confirm and I'll implement them now.
