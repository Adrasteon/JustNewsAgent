# Kafka-First JustNews (Kafka_Justnews_Proposal)

This document describes an ideal, fully open-source, Kafka-centric architecture for the entire JustNews system. The goal is to re-imagine JustNews as a resilient, auditable, agentic news-processing platform that is fully self-hosted (no third-party cloud or managed LLM services), transparent, traceable, and censorship-resistant while reusing and improving the existing agent building blocks.

Principles
- Agentic: each capability is an autonomous agent (Scout, Crawler, NewsReader, Synthesizer, FactChecker, Analyst, Memory, GPU-Orchestrator, Critic, Chief Editor) implemented as a producer/consumer that subscribes to and emits Kafka events.
- Modular & Composable: agents communicate via well-defined topics and schemas. Business logic is separated from transport and storage.
- Fully open-source & self-hosted: all infrastructure and models run on self-managed hardware or k3s/k8s clusters. No third-party managed LLMs or cloud-only services.
- Canonical persistence: the Memory Agent remains the authoritative store; all writes that change canonical state are mediated by Memory and acknowledged via events.
- Transparency & traceability: every event is cryptographically signed by the originating agent, content snapshots can be optionally stored in IPFS/MinIO for immutability and public audit.
- Defensive by design: idempotency, deduplication, DLQs, observable metrics, and automated rollbacks.

Why Kafka?
- Durable, ordered, partitioned logs provide a natural event-sourcing backbone for an agentic platform.
- Consumer groups and partitions enable independent horizontal scaling of agents (many workers can share load deterministically keyed by article_id).
- Schema evolution via a registry (Avro/Protobuf/JSON Schema) gives backward/forward compatibility guarantees.
- Native tooling (Kafka Connect, MirrorMaker, Streams, ksqlDB) supports connectors, stream processing and geo-replication while remaining OSS.

High-level Architecture
- Kafka cluster (KRaft or Zookeeper-less modern Kafka) as central event backbone.
- Schema Registry (self-hosted, e.g., Apicurio or Karapace) for controlled schema evolution.
- Agents as lightweight microservices (Python/asyncio or Go/Java) that produce/consume events and call canonical APIs when necessary.
- Memory Agent: consumes article extraction events and persists to PostgreSQL + vector index (FAISS/Milvus/Weaviate self-hosted). Emits `article.persisted` events.
- GPU Orchestrator: coordinates GPU leases via `gpu.lease.request` / `gpu.lease.granted` / `gpu.lease.released` topics.
- Object store: MinIO for images, raw HTML snapshots and attachments.
- Self-hosted LLM inference cluster: vLLM, vLLM+Ray, or native inference via ggml/gguf runners; model binaries downloaded and stored by the ops team and managed in a model registry.
- Optional decentralization layer: IPFS for public, tamper-evident snapshots; an append-only transparency log for signed events.

Core Topics (suggested names; follow semantic versioning)
- `justnews.crawl.job.v1` — crawl job commands (command topic) and lifecycle events (event topic) keyed by job_id
- `justnews.article.created.v1` — Article extraction result (metadata, content hash, minimal text) keyed by article_id
- `justnews.article.persisted.v1` — Memory acknowledged persistence (with DB id)
- `justnews.analysis.request.v1` — Requests for Analyst agent to run NER/sentiment/bias/depth
- `justnews.analysis.result.v1` — Analyst outcomes (NER, sentiment, bias scores, references)
- `justnews.synthesizer.request.v1` / `justnews.synthesizer.result.v1` — Synthesis and long-form outputs
- `justnews.gpu.lease.v1` — GPU lease lifecycle and status
- `justnews.audit.event.v1` — Signed audit trail events for transparency and tracing
- `justnews.deadletter.v1` — Cross-agent DLQ for unprocessable events

Event & Schema Strategy
- Use Avro/Protobuf with a Schema Registry. Schemas must include `event_id`, `causation_id`, `correlation_id`, `producer_id`, `producer_signature`, `created_at`, `version` and `encoding`.
- Example simplified JSON schema for `article.created` (human-readable):

```
{
  "type": "object",
  "required": ["article_id","url","crawl_timestamp"],
  "properties": {
    "article_id": {"type":"string"},
    "url": {"type":"string"},
    "crawl_timestamp": {"type":"string","format":"date-time"},
    "title": {"type":["string","null"]},
    "text_snippet": {"type":["string","null"]},
    "content_hash": {"type":"string"},
    "attachments": {"type":"array","items":{"type":"string"}}  
  }
}
```

Event-sourcing patterns
- Produce immutable events for every state transition. For canonical state changes Memory consumes commands or events and emits a persisted event as the single source-of-truth.
- Use event keys (article_id) to ensure partitioning for ordered per-article processing.
- Producers must be idempotent and provide a stable message key and message id to allow brokers to deduplicate when possible.

Agent Interactions & Workflows (example)
1. Scout discovers promising URLs and publishes `crawl.job` commands.
2. Crawler consumes `crawl.job` commands, executes rendering (Playwright) and extraction; on success it uploads snapshots to MinIO and emits `article.created` with content_hash and object store keys.
3. Memory Agent consumes `article.created`, validates and persists canonical records (Postgres + embedding store). Memory emits `article.persisted` with DB ids.
4. Analyst consumes `article.persisted` and `article.created`, requests deep analysis (NER, bias, claims) and emits `analysis.result`.
5. Synthesizer consumes `analysis.result` and `article.persisted` and creates human-readable syntheses, emits `synthesizer.result`.
6. Chief Editor/ Critic can subscribe to `synthesizer.result` to apply editorial policies and then emit `publication.ready` for distribution.

GPU Orchestration
- Agents should not directly hold GPU resources. They request leases by producing messages to `justnews.gpu.lease.request.v1` and wait for `gpu.lease.granted.v1` with a lease token.
- GPU Orchestrator manages a consistent view of available devices and enforces TTLs. Lease state is maintained in an internal topic and persisted for audit.

Backpressure, Retries & DLQ
- Use consumer backoff strategies and tune max.poll.records for Java-based consumers or asyncio batch sizes for Python.
- Implement retry topics (e.g., `justnews.analysis.request.retry.v1`) with exponential backoff (or use Kafka's delayed processing pattern via tombstone + scheduled processing if supported).
- Unrecoverable events go to `justnews.deadletter.v1` with rich metadata for debugging.

Idempotency & Exactly-once
- For critical flows (e.g., Memory writes), use idempotent producer semantics and atomic DB upserts keyed by `article_id`.
- Consider Kafka exactly-once semantics (EOS) for stream processing where necessary, or rely on idempotent consumers + transactional writes to state stores.

Stream Processing & Aggregation
- Use ksqlDB or Kafka Streams for low-latency enrichment and join operations (e.g., enrich article streams with source metadata, maintain per-source metrics as materialized views).
- For heavy custom processing or multi-source joins, integrate Apache Flink or a Python-based stream processor depending on team skillset.

Local LLMs & Model Hosting (fully self-hosted)
- Use open-source LLMs and model families that permit local hosting (ensure license compliance). Examples include: BLOOM-family variants, RedPajama, and other permissively re-licensed community models.
- Host models on self-managed inference infra using vLLM, orchestrated via Ray/Kubernetes or other scheduler. For CPU-only or small-scale inference use GGML/gguf-based runners.
- Model registry: store and version model binaries and metadata in a Git-LFS-like or object-store backed registry (MinIO + manifest DB).
- Inference requests: agents make inference calls to a local inference endpoint (gRPC/HTTP) and the inference cluster enforces rate limits and GPU quotas.

Data Stores (self-hosted)
- Canonical DB: PostgreSQL with strong constraints for canonical article metadata and provenance.
- Vector search: FAISS, Milvus, or Weaviate (self-hosted) for embeddings.
- Object storage: MinIO for snapshots, images, attachments.
- Index snapshots for audit: periodic dumps to an immutable storage (optionally IPFS) or stored behind `audit/` topics and snapshots.

Security & Trust
- TLS+SASL (SCRAM) for Kafka cluster communication.
- Agent identities provisioned with cryptographic keys; every agent signs produced events. Signatures stored in the event envelope for non-repudiation.
- Schema registry access controlled; use RBAC for topic writes/reads.
- Secrets stored in a self-hosted secret manager (HashiCorp Vault, or at minimum a well-protected keystore).

Observability & Monitoring
- Metrics: JMX exporter (Kafka), Prometheus scrape targets for agents, GPU Orchestrator exposes GPU metrics.
- Tracing: OpenTelemetry across agents, self-hosted Jaeger/Tempo for traces.
- Alerts on consumer lag, DLQ growth, persistent errors, and GPU exhaustion.

Dev / Local Workflow
- Provide `docker-compose.kafka.yml` for dev that brings up single-node Kafka (KRaft), Schema Registry, MinIO, PostgreSQL, and a simple inference mock.
- CI: use Testcontainers or ephemeral Kafka in the CI runner; add contract tests validating schema compatibility and event formats.

Privacy, Censorship Resistance & Transparency
- Event payloads include provenance and signatures. For public transparency, optionally publish article snapshots and signed events to IPFS or an independent append-only log.
- Provide an open verification tool to fetch an event and validate signatures and hash chains.
- Allow selective redaction in persisted canonical records but keep the signed original event and snapshot in a separate, controlled transparency store.

Operational Considerations
- Runbook for incident response: steps to pause consumers, switch to MCP fallback, inspect DLQ, and resume.
- Upgrade strategy: schema compatibility gates in CI; rolling upgrade of Kafka brokers with low downtime.

Migration Path (practical)
1. Design & schema workshop: define core topics and their schemas; create a schema registry and topic naming policy.
2. Spin up dev Kafka stack and a `kafka_bridge` adapter that dual-writes: MCP→Kafka and Kafka→MCP for compatibility.
3. Pilot one workflow (crawl→article→persist) with dual-write and consumer that writes to Memory.
4. Monitor and validate parity between MCP and Kafka paths; iterate on schema and consumer idempotency.
5. Incrementally onboard other agents, replace MCP endpoints with Kafka consumers and producers.
6. Final cutover: disable MCP writes once parity and reliability are proven.

Testing & Acceptance Criteria
- Contracts: Schema registry compatibility tests and contract-driven consumer/producer tests.
- Performance: e2e 95th percentile latency targets for the crawl→persist→analysis pipeline (define numeric SLAs per deploy default infra).
- Reliability: Consumer group lag must be bounded; DLQ growth must be monitored and tolerated ≤ threshold.
- Integrity: All persisted articles must have verifiable content_hash and signature; replaying events produces identical canonical records.

Roadmap & Phasing (6–12 months optimistic)
- Month 0–1: Design, schema registry, dev compose + kafka bridge POC.
- Month 2–3: Pilot crawl→memory→analysis pipeline with dual-write, add tests and metrics.
- Month 4–6: Incremental onboarding for Synthesizer, NewsReader and GPU orchestrator, add model registry and inference infra.
- Month 7–9: Harden security (TLS/SASL, Vault), add observability & tracing, add DLQ/operational runbooks.
- Month 10–12: Cutover to Kafka-first; decommission MCP; add transparency/public snapshotting via IPFS.

Integration with existing agents and databases
- Overview
  - The Kafka-first design must reuse the existing JustNews agent
    building blocks while allowing them to evolve into Kafka
    producers/consumers. Each agent should be migrated via a small
    adapter layer that maps current MCP tool calls and in-process
    functions into event producers and event handlers. This keeps code
    recognizable and permits incremental rollout.

- Agent-by-agent integration notes
  - Scout
    - Role: discovery and candidate URL emitters.
    - Migration: convert URL discovery outputs into `justnews.crawl.job.v1`
      commands. Include discovery metadata (source, score, discovered_at,
      discovery_id) so downstream agents can prioritize.
    - Backwards compatibility: keep an MCP shim that consumes
      `justnews.crawl.job.v1` and translates to existing crawler API
      until the crawler is migrated.

  - Crawler (rendering + extraction)
    - Role: perform rendering (Playwright), produce HTML snapshots and
      lightweight extraction.
    - Migration: Crawler agents consume `justnews.crawl.job.v1` and
      produce `justnews.article.created.v1` plus `justnews.media.ingest.v1`
      when multimedia is detected.
    - Snapshot handling: store raw snapshots in MinIO and include object
      keys and content hashes in the article events.
    - Edge cases: rendering-only failures should emit `crawl.job.failed.v1`
      with rich failure metadata.

  - Crawler Control / Scheduler
    - Role: job orchestration, scheduling, retries.
    - Migration: scheduler becomes a stateful consumer with a local
      job store (or compacted Kafka-sourced state) that enqueues jobs on
      `justnews.crawl.job.v1`. It also watches `justnews.deadletter.v1`.

  - NewsReader (multimodal reader)
    - Role: high-fidelity extraction across text, images, audio and video.
    - Migration: NewsReader evolves into a set of specialized consumers
      and producers: a text reader for articles, an image reader for
      figures, and a media reader for audio/video segments. Each
      component emits domain events (e.g., `article.extraction.detail.v1`,
      `media.segment.annotation.v1`).

  - Synthesizer
    - Role: produce human-readable syntheses from analysis results.
    - Migration: subscribe to `analysis.result.v1` and `article.persisted.v1`.
      For long-running synthesis jobs, create `synthesizer.request.v1`
      and produce incremental `synthesizer.progress.v1` events as the
      output is generated.

  - FactChecker & Analyst
    - Role: factual verification, claim extraction, NER and bias analysis.
    - Migration: both consume `article.persisted.v1` and `media.persisted.v1`.
      They produce `analysis.request.v1`/`analysis.result.v1` and
      `verification.result.v1`. Results must reference `article_id` and
      `media_segment_id` when applicable so Memory can persist canonical
      associations.

  - Memory (canonical store)
    - Role: the authoritative persistence layer for articles and media.
    - Migration: Memory acts as a command processor for events that
      change canonical state. It consumes `article.created.v1`,
      `media.segment.annotation.v1`, `analysis.result.v1` and performs
      transactional upserts into PostgreSQL and the vector store. After a
      successful transaction, Memory emits `article.persisted.v1` and
      `media.persisted.v1` events that other agents trust as the single
      source of truth.
    - Database design (core tables — simplified)
      - articles
        - id (UUID, PK)
        - article_id (string, unique)
        - url (text)
        - title (text)
        - canonical_text (text)
        - content_hash (text)
        - first_seen_at (timestamp)
        - last_updated_at (timestamp)
        - provenance jsonb (producer ids/causation/correlation)
      - media_objects
        - id (UUID, PK)
        - media_id (string, unique)
        - article_id (FK -> articles.article_id) nullable
        - object_store_key (text)
        - media_type (enum: video,audio,image,other)
        - duration_seconds (float) nullable
        - codec (text)
        - content_hash (text)
        - created_at (timestamp)
        - metadata jsonb (detected language, sample_rate, frame_rate)
      - media_segments
        - id (UUID, PK)
        - media_segment_id (string, unique)
        - media_id (FK -> media_objects.media_id)
        - segment_index (int)
        - start_time (float) # seconds
        - end_time (float) # seconds
        - transcript text nullable
        - embeddings_meta jsonb
        - created_at timestamp
      - analysis_results
        - id (UUID)
        - subject_id (article_id or media_segment_id)
        - subject_type (enum: article,media_segment)
        - analysis_type (text)
        - result jsonb
        - created_at timestamp
      - audit_log
        - id (UUID)
        - event_id
        - raw_event jsonb
        - producer_signature text
        - created_at timestamp

    - Transaction pattern
      - Memory consumes `article.created` events and performs a DB
        transaction that upserts `articles`, writes relevant `analysis`
        rows as placeholders, and emits `article.persisted` only after
        commit. This ensures consumers downstream only act on durable
        canonical state.

  - GPU Orchestrator
    - Role: allocate GPU resources for heavy processing tasks like media
      frame inference and ASR.
    - Integration: GPU leases are represented as events. Media processing
      agents request leases and the Orchestrator emits grant/revoke
      events. Leases encode device ids, reservation tokens and TTLs.

- Adapter & migration strategy (practical)
  - Build small adapter modules per agent (e.g.,
    `agents/{agent_name}/kafka_adapter.py`) that present a compatibility
    layer: existing function signatures call into adapters; adapters
    translate into event production. This keeps the behavioural tests
    and unit tests stable while migrating transport.
  - Add a `mcp_compat` consumer that listens for a compacted set of
    topics and re-exposes them as the old MCP API for legacy consumers
    during the transition.

Multimedia (Video & Audio) Ingress and Processing
- Goals
  - Treat multimedia as first-class content: support ingest, storage,
    time-aligned annotation, search, and synthesis.
  - Preserve full fidelity snapshots in object storage and store the
    minimal canonical representation in Memory with links to raw
    assets.

- Ingress topics & events
  - `justnews.media.ingest.v1` (manifest) — emitted when the crawler or
    an external feed enqueues a new media file. Fields: media_id,
    media_type, source_url, object_store_key, duration_seconds, codec,
    content_hash, discovered_at, source_credits, license
  - `justnews.media.segment.v1` — emitted for each processing segment
    (e.g., 10s chunks, scene cuts). Fields: media_segment_id, media_id,
    start_time, end_time, segment_hash, object_store_key (segment),
    frame_count, audio_sample_rate
  - `justnews.media.transcode.v1` — transcode job requests/results
  - `justnews.media.asr.request.v1` / `justnews.media.asr.result.v1`
  - `justnews.media.vision.request.v1` / `justnews.media.vision.result.v1`
  - `justnews.media.caption.v1` — subtitles / VTT generation results

- Ingress pipeline (example flow)
  1. Crawler or external feed emits `justnews.media.ingest.v1` with
     object store pointers to raw media.
  2. Transcoder consumer ensures canonical codec/container and
     produces `justnews.media.segment.v1` describing chunks or scene
     boundaries; each segment has its own object store key for a
     transcode chunk.
  3. ASR consumers pull segments and produce `justnews.media.asr.result.v1`
     containing transcripts and word timings. ASR jobs request GPU
     leases as needed.
  4. FrameExtractor/VideoReader extracts keyframes per segment and
     produces `justnews.media.vision.request.v1` for object detection,
     OCR and image captioning.
  5. MediaAnnotator aggregates ASR + vision outputs and emits
     `media.segment.annotation.v1` containing time-aligned structured
     metadata and embeddings that Memory will persist.
  6. Memory consumes media annotation events and links them to article
     records: either update existing article (if media belongs to an
     article) or create a stub article that references the media.

- Segmenting strategy & keying
  - Use deterministic segment IDs (sha256(media_id + start + end)) so
    processing can be safely retried and deduplicated.
  - Key events by `media_id` to keep segment processing ordered per
    media file when necessary.

- Storage & retention
  - Raw media: stored in MinIO in a structured path
    (`/media/raw/{year}/{month}/{media_id}`) and content addressed by
    content_hash for immutability.
  - Transcoded segments: `/media/segments/{media_id}/{segment_index}`.
  - Retention policy: raw media may be retained longer than segments
    depending on storage budgets; enforce lifecycle rules (cold
    storage / archival to offline object store or IPFS depending on
    transparency policy).

- ASR & Vision models (self-hosted)
  - ASR: Whisper-like models (or open-source optimized variants) hosted
    on inference nodes. For low-resource runs use Whisper.cpp or
    Silero-like models. Ensure model provenance and license metadata
    are part of the model registry.
  - Vision: object detection, OCR and image-captioning using open
    models (e.g. DETR, Tesseract for OCR with tuned models, CLIP/ViT
    families for captioning and visual embeddings).
  - GPU orchestration: segment processors request leases and release
    them when done; Orchestrator enforces quota and balances work.

- Indexing & timeline UX
  - Memory stores segment-level transcripts, visual annotations and
    embeddings; a timeline index provides time-aligned search and
    snippet extraction (e.g., "Show me mentions of 'eviction' between
    5:00 and 7:00 in the video").
  - UI clients subscribe to `justnews.media.annotation.indexed.v1` and
    query the timeline index for fast retrieval.

- Search & Retrieval
  - Create embeddings for audio transcripts and visual frames; store in
    vector store with keys that link back to segment IDs and article
    IDs. Use approximate nearest neighbor (ANN) searches for multimedia
    retrieval.

- Privacy & legal
  - Provide redaction workflows: PII detection in transcripts triggers
    `policy.redaction.request.v1` which the Chief Editor agent reviews
    and issues `policy.redaction.approved.v1` or reject events.
  - Ensure audio/video ingestion respects source licensing and
    do-not-distribute flags contributed by Scout or external feeds.

- Example media schemas (human-readable snippets)
  - `justnews.media.ingest.v1`

```
{
  "type": "object",
  "required": ["media_id","object_store_key","media_type","duration_seconds"],
  "properties": {
    "media_id":{"type":"string"},
    "object_store_key":{"type":"string"},
    "media_type":{"type":"string"},
    "duration_seconds":{"type":"number"},
    "content_hash":{"type":"string"},
    "source_url":{"type":["string","null"]},
    "discovered_at":{"type":"string","format":"date-time"},
    "license":{"type":["string","null"]},
    "source_credits":{"type":["string","null"]}
  }
}
```

  - `justnews.media.segment.v1`

```
{
  "type":"object",
  "required":["media_segment_id","media_id","start_time","end_time","segment_hash"],
  "properties":{
    "media_segment_id":{"type":"string"},
    "media_id":{"type":"string"},
    "start_time":{"type":"number"},
    "end_time":{"type":"number"},
    "segment_hash":{"type":"string"},
    "object_store_key":{"type":["string","null"]},
    "frame_count":{"type":["integer","null"]},
    "sample_rate":{"type":["integer","null"]}
  }
}
```

Operational concerns for multimedia
- Bandwidth & storage: heavy; recommend setting cluster-level quotas
  and separate media storage tiers. Use lifecycle policies and cold
  storage for older assets.
- Cost of inference: batch and off-peak processing windows, and use
  scheduling to avoid saturating GPUs with non-urgent jobs.
- Monitoring: track segment queues, transcode latency, ASR error rates,
  and per-model throughput.

Appendix: Tools & OSS components (self-hosted only)
- Kafka (KRaft) — cluster
- Schema Registry — Apicurio or Karapace
- Kafka Connect — connectors
- ksqlDB / Kafka Streams / Flink — stream processing
- MinIO — S3-compatible object store
- PostgreSQL — canonical metadata
- FAISS / Milvus / Weaviate — vector store
- Prometheus + Grafana — metrics + dashboards
- OpenTelemetry + Jaeger — traces
- HashiCorp Vault — secrets (self-hosted)
- IPFS — optional public transparency snapshots
- Inference: vLLM, ggml/gguf runners, Ray or Kubernetes for orchestration

References & Context
- Kafka quickstart, streams and ops guides inspired implementation patterns from Apache Kafka (see project docs and examples). The topics/partitioning, schema registry and stream processing patterns are aligned with Kafka best practices and the KRaft deployment model for self-hosting.




