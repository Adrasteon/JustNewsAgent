## Crawl4AI — API & Key Interfaces (concise reference)

This short reference summarises the Crawl4AI programmatic APIs, dispatcher classes, REST endpoints, and common usage patterns (extracted from the project's Crawl4AI docs).

### Libraries / Classes

- `AsyncWebCrawler`
  - Main async crawler class used for single or multi-URL crawling via `arun` / `arun_many`.
- `AdaptiveCrawler` / `AdaptiveConfig`
  - High-level adaptive crawler with `digest()` method that supports `save_state` and `resume_from` parameters for checkpointing and resuming crawls.
- `MemoryAdaptiveDispatcher`
  - Dispatcher that monitors memory usage, pauses dispatching when memory usage exceeds `memory_threshold_percent` and resumes when memory is available.
- `SemaphoreDispatcher`
  - Simple fixed-concurrency dispatcher (useful for rate-limited crawling).
- `CrawlerRunConfig`, `BrowserConfig`
  - Configuration objects for page-level settings (timeouts, wait_for selectors, stream mode, cache_mode, etc.).

### Key Methods & Parameters

- `AsyncWebCrawler.arun(url, config)`
  - Run a single crawl and return a result (supports per-page options).
- `AsyncWebCrawler.arun_many(urls, config, dispatcher)`
  - Crawl many URLs in batch or stream mode. Can be called in streaming mode (`stream=True`) to iterate results as they become available.
- `AdaptiveCrawler.digest(start_url, query, save_state=False, state_path=None, resume_from=None)`
  - High-level digest call for adaptive crawling. If `save_state=True` and `state_path` is set, progress will be persisted and later resumable via `resume_from`.
- `MemoryAdaptiveDispatcher(memory_threshold_percent, check_interval, max_session_permit, memory_wait_timeout)`
  - Create and pass to `arun_many` to auto-pause when memory is high.

### REST API Endpoints (server)

- `POST /crawl` — Initiate a crawl. Request body includes `urls`, `browser_config`, `crawler_config`.
- `POST /crawl/stream` — Start a streaming crawl returning NDJSON lines for results.
- `POST /crawl/job` and `GET /crawl/job/{id}` — Submit and check asynchronous crawl jobs.
- `POST /html`, `POST /screenshot`, `POST /pdf`, `POST /execute_js`, `POST /md` — Extraction endpoints for different content types.
- `GET /health`, `GET /schema`, `GET /metrics` — Utility endpoints.

### Save / Resume Example (python)

```python
config = AdaptiveConfig(save_state=True, state_path="my_crawl_state.json")
result = await adaptive.digest(start_url, query, config=config)

# Later: resume
result = await adaptive.digest(start_url, query, resume_from="my_crawl_state.json")
```

### Memory-adaptive dispatch example

```python
dispatcher = MemoryAdaptiveDispatcher(memory_threshold_percent=80.0, check_interval=1.0, max_session_permit=15)
results = await crawler.arun_many(urls=large_list, config=CrawlerRunConfig(stream=False), dispatcher=dispatcher)
```

### Streaming example (process results as available)

```python
async for result in await crawler.arun_many(urls=urls, config=CrawlerRunConfig(stream=True), dispatcher=dispatcher):
    if result.success:
        await process_result(result)

### Interactive page-control & overlays

Crawl4AI exposes C4A-Script style interactive commands and programmatic helpers to manipulate pages before extraction — useful for closing cookie consent dialogs, sign-in overlays, modal popups, cookie banners, and other interactive UI obstacles.

Core interactive primitives:

- `GO <url>` — navigate to a URL.
- `WAIT <seconds>` or `WAIT `<selector>` <timeout>` — wait for time or for a CSS selector to appear.
- `CLICK <selector>` — click an element (useful for "Accept" buttons on cookie popups).
- `PRESS <key>` — simulate keyboard presses (e.g., Escape to close modals).
- `DRAG <x1> <y1> <x2> <y2>` — perform drag operations for sliders or custom dismiss gestures.
- `REPEAT(<command>, `<condition>`)` — repeat a command until a JS condition is met (helpful for infinite-scroll or load-more flows).
- `EXECUTE_JS` / `POST /execute_js` — run arbitrary JS to remove elements or change page state.

Examples (C4A-Script / SDK style):

1) Close cookie banner by clicking an "Accept" button (CSS-driven):

```python
# Wait for cookie button then click
config = CrawlerRunConfig(wait_for="css:button.cookie-accept", wait_for_timeout=8000)
result = await crawler.arun(url="https://example.com", config=config)
# If using scripting capabilities (C4A script):
# CLICK `button.cookie-accept`
```

2) Dismiss sign-in overlay by sending Escape key or clicking close:

```python
# preferred: click close button when present
# C4A-Script: WAIT `css:button.modal-close` 5
# C4A-Script: CLICK `css:button.modal-close`

# fallback: press Escape to try close keyboard-driven modals
# C4A-Script: PRESS Escape
```

3) Remove stubborn elements via JS then extract:

```python
js = "document.querySelectorAll('.cookie-banner, .overlay--modal').forEach(e => e.remove())"
await crawler.execute_js(url, js)
result = await crawler.arun(url)
```

4) Robust handling pattern (best practice)

- 1) Wait for page to stabilize with `wait_for` (selector or timeout).
- 2) Attempt targeted `CLICK` on 'Accept' / 'Close' selectors (try multiple selectors in priority order).
- 3) If selectors not found, run small JS to hide elements (use conservative selectors and timeouts).
- 4) If overlay persists, `PRESS Escape` or `CLICK` an area outside modal (e.g., `.modal-backdrop`).
- 5) Re-check main content selector (e.g., `.article-body`) and only proceed to extraction when present.

Notes and tips

- Use `wait_for` with meaningful selectors (e.g., `css:.article-body`) to avoid removing elements too early.
- Prefer clicking explicit "Accept" or "Close" buttons rather than broad JS removals — safer and less likely to alter content priorities.
- Keep a small selector fallback list: cookie accept buttons often use `button[aria-label*="accept"]`, `button[class*="cookie"]`, `button:contains("Accept")` (Crawl4AI supports `wait_for` CSS selectors; for `:contains()` you may need to evaluate JS).
- Combine `EXECUTE_JS` with conservative timeouts and logging so you can audit when JS removals were used.
- For repeatable flows (infinite scroll / load more), use `REPEAT(SCROLL_DOWN, condition)` or `arun_many` streaming with `stream=True`.

```

### Notes & Integrations

- Crawl4AI provides both native Python SDK and a REST API; the project uses native import where available and falls back to Docker-based calls.
- The AdaptiveCrawler `save_state` / `resume_from` mechanism provides an out-of-the-box checkpointing primitive — suitable when you want crawls to pick up where they left off.
- Use `MemoryAdaptiveDispatcher` to avoid OOM and to achieve pause-resume behavior tied to resource pressure.
- Keep DB-level dedupe (e.g., `crawled_urls`) as a safe guard even when using Crawl4AI resume — this prevents duplicate ingestion when jobs are restarted manually or re-run with overlapping frontiers.

### Useful docs (local pointers)

- `docs/md_v2/core/adaptive-crawling.md` — AdaptiveCrawler behavior and examples
- `docs/md_v2/api/arun_many.md` — `arun_many` + dispatcher examples
- `docs/md_v2/api/digest.md` — `digest()` method and `resume_from` usage
- `docs/md_v2/assets/llm.txt` — API endpoints and Docker deployment notes

---

Generated: 2025-08-27 — brief summary created from project documentation and Crawl4AI docs.
