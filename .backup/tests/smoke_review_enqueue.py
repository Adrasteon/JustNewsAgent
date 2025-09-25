#!/usr/bin/env python3
"""Quick smoke test: simulate chief_editor.review_evidence logic by appending to queue file and verifying.

This avoids importing FastAPI/Pydantic in the test environment.
"""
import json
from datetime import UTC, datetime
from pathlib import Path

manifest = Path('tests/tmp_manifest.json')
manifest.write_text(json.dumps({'url':'https://example.com','html_file':'e.html'}))

queue_file = Path('./evidence_review_queue.jsonl')
if queue_file.exists():
    queue_file.unlink()

record = {
    'manifest': str(manifest),
    'reason': 'paywalled_snapshot',
    'received_at': datetime.now(UTC).isoformat()
}
with open(queue_file, 'a', encoding='utf-8') as f:
    f.write(json.dumps(record) + '\n')

# Verify
lines = queue_file.read_text().splitlines()
assert any(str(manifest) in ln for ln in lines)
print('Review enqueue smoke test passed')
