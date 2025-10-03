"""Parity replay runner utilities.

Provides a small, testable harness to replay sample datasets through a
transport adapter and collect the canonical Memory outputs for comparison.

Functions follow the project's type-hinting and docstring standards so they
can be reused by unit tests and future CI parity runners.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List
from datetime import datetime

from kafka.src.agents.adapter_template import (
    EventEnvelope,
    TransportAdapter,
)
from kafka.src.agents.crawler_adapter_pilot import start_pilot_crawler
from kafka.src.agents.memory_adapter import InMemoryStorage, start_memory_consumer


def load_sample(path: str) -> Iterable[Dict[str, Any]]:
    """Yield JSON objects from a newline-delimited JSON sample file.

    Args:
        path: Filesystem path to a jsonl sample file.

    Yields:
        Parsed JSON objects as dictionaries.
    """
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            yield json.loads(line)


def run_replay(adapter: TransportAdapter, sample_path: str) -> Dict[str, Dict[str, Any]]:
    """Run the pilot pipeline (Scout->Crawler->Memory) using ``adapter``.

    The function registers the crawler and memory handlers on the provided
    adapter, replays every event from ``sample_path`` and returns the in-memory
    storage map from article id to persisted record.

    Args:
        adapter: Transport adapter instance implementing produce/consume.
        sample_path: Path to the newline-delimited JSON sample dataset.

    Returns:
        Mapping of article id -> persisted record dictionaries.
    """
    # Register pipeline handlers
    start_pilot_crawler(adapter=adapter)
    storage = start_memory_consumer(adapter=adapter, storage=InMemoryStorage())

    # Produce sample events
    for obj in load_sample(sample_path):
        env = EventEnvelope(
            event_id=obj["id"],
            event_type="scout.article.created",
            payload=obj,
            metadata={"source": "parity-runner"},
        )
        adapter.produce("scout.article.created", env)

    return storage.store


def _is_time_like(key: str) -> bool:
    """Return True if the key looks like a timestamp/time-like field."""
    return bool(re.match(r".*(_at|_time|timestamp)$", key, flags=re.IGNORECASE))


def _parse_iso(s: str):
    """Parse an ISO-like timestamp string into a timezone-aware datetime.

    Returns None if parsing fails.
    """
    try:
        return datetime.fromisoformat(s.replace('Z', '+00:00'))
    except Exception:
        return None


def _json_equal(a, b) -> bool:
    """Return True if JSON-stable serialization of a and b are equal."""
    try:
        return json.dumps(a, sort_keys=True, ensure_ascii=False) == json.dumps(b, sort_keys=True, ensure_ascii=False)
    except Exception:
        return False


def _normalize_record(
    record: Dict[str, Any], remove_fields: List[str], remove_time_like: bool
) -> Dict[str, Any]:
    """Normalize a single record (shallow) for deterministic comparisons."""
    rec: Dict[str, Any] = {}
    for kk, vv in (record.items() if isinstance(record, dict) else []):
        if kk in remove_fields:
            continue
        if remove_time_like and _is_time_like(kk):
            continue
        if kk == "url" and isinstance(vv, str):
            vv = vv.rstrip("/")
        if kk == "source" and isinstance(vv, str):
            vv = vv.lower()
        if isinstance(vv, (dict, list)):
            try:
                vv = json.loads(json.dumps(vv, sort_keys=True, ensure_ascii=False))
            except Exception:
                pass
        rec[kk] = vv
    return rec


def _compare_record_fields(lcmp: Dict[str, Any], rcmp: Dict[str, Any], time_tolerance_seconds: int) -> Dict[str, Any]:
    """Compare fields of two normalized records and return field-level diffs."""
    local_diffs = {}
    for kk in sorted(lcmp.keys()):
        lv = lcmp.get(kk)
        rv = rcmp.get(kk)
        if lv == rv:
            continue
        if isinstance(lv, str) and isinstance(rv, str) and _is_time_like(kk):
            ldt = _parse_iso(lv)
            rdt = _parse_iso(rv)
            if ldt is not None and rdt is not None:
                delta = abs((ldt - rdt).total_seconds())
                if delta <= float(time_tolerance_seconds):
                    continue
                local_diffs[kk] = {"left": lv, "right": rv, "delta_seconds": delta}
                continue
        if isinstance(lv, (dict, list)) and isinstance(rv, (dict, list)):
            if _json_equal(lv, rv):
                continue
        local_diffs[kk] = {"left": lv, "right": rv}
    return local_diffs


def normalize_store(
    store: Dict[str, Dict[str, Any]],
    remove_fields: List[str] | None = None,
    remove_time_like: bool = True,
    preserve_time_keys: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Normalize a storage map for deterministic comparisons.

    This function performs several deterministic transformations to make
    equality comparisons robust across timing and environmental differences:
    - Removes keys listed in ``remove_fields`` (defaults to ["persisted_at"]).
    - Optionally removes all keys that look like timestamps (ending with
      ``_at``, ``_time`` or ``timestamp``) when ``remove_time_like`` is True.
    - Canonicalizes URLs by removing trailing slashes.
    - Normalizes ``source`` to lower-case to avoid case diffs.
    - Ensures nested structures are JSON-serializable with stable key ordering.

    Args:
        store: The raw in-memory store mapping id -> record.
        remove_fields: Optional list of keys to remove from each record.
        remove_time_like: If True remove keys that look like timestamps.
        preserve_time_keys: If True, preserve time-like keys for tolerance checks.

    Returns:
        A new normalized mapping suitable for equality checks.
    """
    remove_fields = remove_fields or ["persisted_at"]
    normalized: Dict[str, Dict[str, Any]] = {}

    for k, v in store.items():
        normalized[k] = _normalize_record(v, remove_fields, remove_time_like)

    if preserve_time_keys:
        for k, v in store.items():
            rec = normalized.get(k, {})
            for kk, vv in (v.items() if isinstance(v, dict) else []):
                if kk in remove_fields:
                    continue
                if not remove_time_like or not _is_time_like(kk):
                    rec[kk] = vv
            normalized[k] = rec

    return normalized


def compare_stores(
    left: Dict[str, Dict[str, Any]],
    right: Dict[str, Dict[str, Any]],
    time_tolerance_seconds: int = 5,
    ignore_keys: List[str] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Compare two storage maps with time-tolerance for time-like fields.

    This refactored implementation delegates time parsing and JSON comparisons
    to small helpers to reduce the complexity of the main comparison loop.
    """
    ignore_keys = set(ignore_keys or [])

    diffs: Dict[str, Dict[str, Any]] = {}

    all_keys = set(left.keys()) | set(right.keys())
    for k in sorted(all_keys):
        lrec = left.get(k, {})
        rrec = right.get(k, {})
        # Build comparison dicts excluding ignored keys
        lcmp = {kk: vv for kk, vv in lrec.items() if kk not in ignore_keys}
        rcmp = {kk: vv for kk, vv in rrec.items() if kk not in ignore_keys}

        if set(lcmp.keys()) != set(rcmp.keys()):
            diffs[k] = {"left": lcmp, "right": rcmp, "reason": "key-set-difference"}
            continue

        local_diffs = _compare_record_fields(lcmp, rcmp, time_tolerance_seconds)
        if local_diffs:
            diffs[k] = {"left": lcmp, "right": rcmp, "field_diffs": local_diffs}

    return diffs
