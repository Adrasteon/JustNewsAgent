#!/usr/bin/env python3
"""
Verify downloaded agent models: check folders, sizes, and attempt lightweight load for transformers/sentence-transformers.
"""
import logging
import sys
from pathlib import Path

from common.observability import get_logger

# Configure basic logging for the script (timestamp + level + message)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENTS_DIR = REPO_ROOT / "agents"

# Gather agent model folders
agents = [p for p in AGENTS_DIR.iterdir() if p.is_dir()]

results = []
for a in agents:
    models_dir = a / "models"
    if not models_dir.exists():
        logger.warning("Agent %s has no models/ directory", a.name)
        continue
    for m in sorted(models_dir.iterdir()):
        # ignore hidden/system dirs (like .locks) and non-dirs
        if m.name.startswith("."):
            logger.debug("Skipping hidden entry %s for agent %s", m.name, a.name)
            continue
        if not m.is_dir():
            continue
        size = sum(f.stat().st_size for f in m.rglob("*") if f.is_file())
        results.append((a.name, m.name, size, str(m)))

# Print basic table
for agent, name, size, path in results:
    logger.info("%s : %s -> %s (%.1f MB)", agent, name, path, size / 1024.0 / 1024.0)

# Attempt lightweight loads
# For sentence-transformers: try to import SentenceTransformer and point cache_folder at path
# For transformers: try to load tokenizer only (cheaper)

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None

checks = []
for agent, name, _size, path in results:
    path_obj = Path(path)
    # quick existence check
    if not path_obj.exists():
        checks.append((agent, name, f"failed: path missing {path}"))
        continue

    # Some downloaders put a nested repo layout like 'models--org--name' inside the folder
    # If we find a single child directory matching that pattern, use it as the model root.
    nested_candidates = [
        p for p in path_obj.iterdir() if p.is_dir() and p.name.startswith("models--")
    ]
    if len(nested_candidates) == 1:
        logger.debug(
            "Using nested model folder %s for %s/%s", nested_candidates[0], agent, name
        )
        path_obj = nested_candidates[0]
        path = str(path_obj)

    # Heuristics based on files present in the folder
    present = {p.name for p in path_obj.iterdir() if p.is_file()}
    # Files that usually indicate a sentence-transformers package
    st_indicators = {
        "modules.json",
        "sentence_transformers_config.json",
        "model.safetensors",
    }
    # Files that usually indicate a transformers tokenizer/model
    tf_indicators = {
        "pytorch_model.bin",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "merges.txt",
    }

    is_st = (
        bool(st_indicators & present)
        or "sentence-transformers" in name
        or name.lower().startswith("all-")
        or "mpnet" in name.lower()
    )
    is_tf = bool(tf_indicators & present) or any(
        fn.endswith((".bin", ".safetensors")) for fn in present
    )

    # If the folder looks like a non-model (e.g. leftover metadata), skip with message
    if not is_st and not is_tf:
        # we'll still attempt a gentle probe: try ST first then tokenizer as fallback
        probe_order = ["st", "tf"]
    else:
        probe_order = ["st", "tf"] if is_st else ["tf", "st"]

    loaded = False
    st_error = None
    tf_error = None
    for probe in probe_order:
        if probe == "st" and SentenceTransformer is not None:
            logger.debug(
                "Probing sentence-transformer model load (no sensitive data logged)"
            )
            try:
                st = SentenceTransformer(str(path))
                v = st.encode(["test"], show_progress_bar=False)
                checks.append((agent, name, f"ok (embed len {len(v[0])})"))
                loaded = True
                break
            except Exception as e:
                # Don't log full path or exception message at info level; keep details in debug with exc_info
                logger.debug("SentenceTransformer load failed", exc_info=True)
                st_error = e
        elif probe == "tf" and AutoTokenizer is not None:
            logger.debug("Probing tokenizer load (no sensitive data logged)")
            try:
                tok = AutoTokenizer.from_pretrained(str(path))
                checks.append((agent, name, "ok (tokenizer loaded)"))
                loaded = True
                break
            except Exception as e:
                logger.debug("AutoTokenizer load failed", exc_info=True)
                tf_error = e

    if not loaded:
        # Prefer the most informative error if available
        err_msg = None
        if "st_error" in locals():
            err_msg = f"st failed: {st_error}"
        if "tf_error" in locals():
            # prefer tokenizer error if st wasn't present
            if err_msg:
                err_msg = f"{err_msg} ; tf failed: {tf_error}"
            else:
                err_msg = f"tf failed: {tf_error}"
        if err_msg is None:
            err_msg = "failed: unrecognized or incomplete model folder"
        checks.append((agent, name, err_msg))

for c in checks:
    logger.info("Verification: %s / %s -> %s", c[0], c[1], c[2])

# return non-zero on failures
fails = [c for c in checks if not c[2].startswith("ok")]
if fails:
    logger.error("Some model verifications failed: %d", len(fails))
    for f in fails:
        logger.error("  %s / %s -> %s", f[0], f[1], f[2])
    sys.exit(2)

logger.info("All verifications passed")
sys.exit(0)
