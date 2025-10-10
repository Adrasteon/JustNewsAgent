#!/usr/bin/env python3
"""
select_beta_release_candidates.py

Scan the repository workspace and heuristically select files that best
represent the most recent and relevant code and documentation for a
candidate "beta" minimal release. The script is conservative by default
and writes a detailed JSON/MD report. Optionally it can copy the selected
files into a target directory preserving the workspace structure.

Usage:
  python3 scripts/select_beta_release_candidates.py --dry-run
  python3 scripts/select_beta_release_candidates.py --target release_beta_minimal --copy

The selection is based on file modification timestamps, content heuristics
(frontmatter last_updated, presence of deprecation markers), and duplicate
content detection. Exclude patterns are conservative to avoid including
backup or archive folders.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import stat
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

LOG = logging.getLogger("select_beta_release_candidates")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


@dataclass
class FileInfo:
    path: str
    mtime: float
    size: int
    sha256: str
    ext: str
    score: float
    reasons: List[str]
    last_updated_field: Optional[str] = None


DEFAULT_EXCLUDES = [
    ".git",
    "__pycache__",
    "archive_obsolete_files",
    "quality_backups",
    "docs_catalogue_v2.json.bak",
    "logs",
    "tmp",
    "large_scale_crawl_results",
    "node_modules",
    "venv",
    "env",
]

# Exclude by file extension (common model / binary / archive formats and DB files)
DEFAULT_EXT_EXCLUDES = [
    ".db",
    ".pt",
    ".bin",
    ".engine",
    ".onnx",
    ".tar",
    ".gz",
    ".zip",
    ".h5",
    ".ckpt",
    ".tflite",
]


def compute_sha256(path: str, *, block_size: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()


def read_frontmatter_last_updated(text: str) -> Optional[str]:
    """If Markdown/YAML frontmatter exists, return a last_updated-like field."""
    if not text.startswith("---"):
        return None
    frontmatter = text.split("---", 2)
    if len(frontmatter) < 3:
        return None
    yaml_block = frontmatter[1]
    # look for keys like last_updated, last_modified, date
    m = re.search(r"(?im)^(last_updated|last_modified|date)\s*:\s*(.+)$", yaml_block)
    if m:
        return m.group(2).strip().strip('"\'')
    return None


def collect_files(root: str, excludes: Iterable[str]) -> List[str]:
    excludes_set = set(excludes)
    selected: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # prune excluded directories
        rel = os.path.relpath(dirpath, root)
        parts = rel.split(os.sep) if rel != "." else []
        # remove any excluded directory names from dirnames so os.walk won't descend
        dirnames[:] = [d for d in dirnames if d not in excludes_set]
        # if current directory is itself excluded, skip processing files
        if any(p in excludes_set for p in parts):
            dirnames[:] = []  # don't walk further
            continue
        for fn in filenames:
            if fn.startswith("."):
                # skip hidden files
                continue
            # skip specific basenames requested via excludes
            if fn in excludes_set:
                continue
            # skip files by extension (model binaries, DBs, archives)
            ext = os.path.splitext(fn)[1].lower()
            if ext in DEFAULT_EXT_EXCLUDES:
                continue
            abspath = os.path.join(dirpath, fn)
            # skip symlinks that point outside
            try:
                st = os.lstat(abspath)
            except OSError:
                continue
            if stat.S_ISLNK(st.st_mode):
                # include symlink file, but avoid broken
                if not os.path.exists(os.path.realpath(abspath)):
                    continue
            selected.append(abspath)
    return selected


def file_is_text(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            chunk = f.read(2048)
            if b"\0" in chunk:
                return False
    except OSError:
        return False
    return True


def score_file(path: str, now: datetime) -> Tuple[float, List[str], Optional[str]]:
    """Return (score, reasons, last_updated_field) for a file.

    Score is heuristic: recency contributes strongly; presence in key
    directories adds small boost; negative markers reduce score.
    """
    reasons: List[str] = []
    try:
        st = os.stat(path)
    except OSError:
        return 0.0, ["stat-failed"], None
    mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)
    age_days = (now - mtime).days
    # recency: up to 365 days, newer files score higher
    recency_score = max(0.0, 1.0 - (age_days / 365.0))
    score = recency_score * 60.0  # weight recency as 60/100
    if path.startswith(os.path.join(os.getcwd(), "agents")):
        score += 15.0
        reasons.append("agent-path")
    if path.startswith(os.path.join(os.getcwd(), "scripts")):
        score += 6.0
        reasons.append("scripts-path")
    if "/tests/" in path or path.endswith("_test.py") or path.startswith(os.path.join(os.getcwd(), "tests")):
        score += 4.0
        reasons.append("test")
    if path.endswith(".md") or path.endswith(".rst"):
        score += 6.0
        reasons.append("doc")
    # text heuristics
    last_updated_field: Optional[str] = None
    if file_is_text(path):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                txt = f.read()
        except OSError:
            txt = ""
        # deprecation markers
        if re.search(r"(?i)\b(deprecated|do not use|obsolete|archive)\b", txt):
            score -= 50.0
            reasons.append("deprecated-marker")
        # TODOs are slightly negative
        if "TODO" in txt or "FIXME" in txt:
            score -= 3.0
            reasons.append("todo")
        # if markdown frontmatter contains last_updated, prefer recent
        if path.endswith(".md"):
            last_updated_field = read_frontmatter_last_updated(txt)
            if last_updated_field:
                reasons.append("frontmatter-last-updated")
                # try to parse a year if it's present
                year_match = re.search(r"(20\d{2})", last_updated_field)
                if year_match:
                    year = int(year_match.group(1))
                    # prefer recent years
                    score += min(5.0, max(0.0, (year - 2020)))
    # ensure score within 0..100
    score = max(0.0, min(100.0, score))
    return score, reasons, last_updated_field


def select_candidates(root: str, excludes: Iterable[str], max_items: int = 800, since_days: Optional[int] = None) -> Dict[str, FileInfo]:
    now = datetime.now(timezone.utc)
    files = collect_files(root, excludes)
    LOG.info("scanned %d files", len(files))
    infos: Dict[str, FileInfo] = {}
    hash_map: Dict[str, List[Tuple[str, float]]] = {}
    for p in files:
        try:
            st = os.stat(p)
        except OSError:
            continue
        size = st.st_size
        mtime = st.st_mtime
        ext = os.path.splitext(p)[1].lower()
        # optional since filter
        if since_days is not None:
            if datetime.now(timezone.utc) - datetime.fromtimestamp(mtime, tz=timezone.utc) > timedelta(days=since_days):
                continue
        # compute sha256 for dedupe
        try:
            sha = compute_sha256(p) if size > 0 and file_is_text(p) else "binary-" + str(size)
        except Exception:
            sha = "<error>"
        score, reasons, last_updated_field = score_file(p, now)
        fi = FileInfo(path=p, mtime=mtime, size=size, sha256=sha, ext=ext, score=score, reasons=reasons, last_updated_field=last_updated_field)
        infos[p] = fi
        hash_map.setdefault(sha, []).append((p, mtime))

    # Deduplicate: if multiple files have same sha, keep the newest only
    selected: Dict[str, FileInfo] = {}
    for sha, entries in hash_map.items():
        if not entries:
            continue
        # choose newest mtime
        entries_sorted = sorted(entries, key=lambda e: e[1], reverse=True)
        chosen_path = entries_sorted[0][0]
        selected[chosen_path] = infos[chosen_path]
        # annotate duplicates as reason on the chosen file
        if len(entries) > 1:
            dup_count = len(entries)
            selected[chosen_path].reasons.append(f"deduped_{dup_count}")

    # Sort by score descending
    ordered = sorted(selected.values(), key=lambda f: f.score, reverse=True)
    # Keep top max_items
    ordered = ordered[:max_items]
    result = {f.path: f for f in ordered}
    LOG.info("selected %d candidate files (top %d)", len(result), max_items)
    return result


def detect_vendor_dirs(root: str, search_root: str = "agents") -> List[str]:
    """Detect likely third-party vendored sub-repositories under `search_root`.

    Heuristics: directory contains a package.json or pyproject.toml and either
    a .github folder or LICENSE file.
    Return a list of directory basenames to exclude.
    """
    vendor_excludes: List[str] = []
    base = os.path.join(root, search_root)
    if not os.path.isdir(base):
        return vendor_excludes
    for dirpath, dirnames, filenames in os.walk(base):
        rel = os.path.relpath(dirpath, base)
        if rel == ".":
            continue
        parts = rel.split(os.sep)
        # only consider nested directories (depth >= 2 under agents)
        if len(parts) < 2:
            continue
        has_package = os.path.exists(os.path.join(dirpath, "package.json"))
        has_pyproject = os.path.exists(os.path.join(dirpath, "pyproject.toml"))
        has_license = os.path.exists(os.path.join(dirpath, "LICENSE"))
        has_github = os.path.exists(os.path.join(dirpath, ".github"))
        if (has_package or has_pyproject) and (has_license or has_github):
            vendor_excludes.append(parts[-1])
    return vendor_excludes


def write_report(candidates: Dict[str, FileInfo], out_json: str, out_md: str) -> None:
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    payload = {p: asdict(fi) for p, fi in candidates.items()}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"generated_at": datetime.now(timezone.utc).isoformat(), "candidates": payload}, f, indent=2)
    # Markdown summary
    lines: List[str] = ["# Beta release candidate files\n", f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n"]
    for p, fi in sorted(candidates.items(), key=lambda kv: kv[1].score, reverse=True):
        reasons = ", ".join(fi.reasons)
        mtime = datetime.fromtimestamp(fi.mtime, tz=timezone.utc).isoformat()
        lines.append(f"- `{os.path.relpath(p)}` (score={fi.score:.1f}) — {reasons} — mtime={mtime}\n")
    with open(out_md, "w", encoding="utf-8") as f:
        f.writelines(lines)
    LOG.info("wrote report: %s and %s", out_json, out_md)


def copy_candidates(candidates: Dict[str, FileInfo], root: str, target: str) -> None:
    if os.path.exists(target):
        LOG.info("target exists: %s — files will be overwritten where necessary", target)
    for p, fi in candidates.items():
        rel = os.path.relpath(p, root)
        dest = os.path.join(target, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        # copy preserving mode
        shutil.copy2(p, dest)
    LOG.info("copied %d files to %s", len(candidates), target)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select beta release candidate files")
    p.add_argument("--dry-run", action="store_true", default=False, help="Don't copy files; only generate reports")
    p.add_argument("--target", type=str, default="release_beta_minimal", help="Target directory to copy candidates into")
    p.add_argument("--max-items", type=int, default=800, help="Maximum number of candidate files to include")
    p.add_argument("--since-days", type=int, default=None, help="Only consider files modified in the last N days")
    p.add_argument("--exclude", action="append", default=[], help="Additional exclude patterns (directories or filenames)")
    p.add_argument("--copy", action="store_true", default=False, help="Copy selected files into the target directory")
    p.add_argument("--report-json", default="reports/beta_candidate_report.json")
    p.add_argument("--report-md", default="reports/beta_candidate_summary.md")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = os.getcwd()
    excludes = list(DEFAULT_EXCLUDES) + args.exclude
    # auto-detect vendored third-party sub-repos to exclude (e.g., nucleoid_repo)
    auto_vendors = detect_vendor_dirs(root, search_root="agents")
    if auto_vendors:
        LOG.info("auto-detected vendor dirs to exclude: %s", auto_vendors)
        excludes.extend(auto_vendors)
    LOG.info("root=%s excludes=%s", root, excludes)
    candidates = select_candidates(root, excludes, max_items=args.max_items, since_days=args.since_days)
    write_report(candidates, args.report_json, args.report_md)
    if args.copy and not args.dry_run:
        copy_candidates(candidates, root, args.target)
    elif args.copy and args.dry_run:
        LOG.info("--copy requested but --dry-run true, skipping actual copy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
