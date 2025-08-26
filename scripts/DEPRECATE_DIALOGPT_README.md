Deprecate DialoGPT (deprecated) helper
==========================

This repository includes a small helper script `scripts/deprecate_dialogpt.py` to safely locate and optionally replace occurrences of DialoGPT (deprecated) model identifiers and mentions across the codebase.

What it does
- Performs a dry-run by default and prints files that reference `DialoGPT (deprecated)` or `microsoft/DialoGPT (deprecated)-*`.
- When run with `--apply` it will:
  - For Python files: replace quoted literal model ids like `"distilgpt2 (deprecated)"` with
    `os.environ.get("DIALOGPT_REPLACEMENT_MODEL", "distilgpt2")` and add `import os` if missing.
  - Annotate `DialoGPT (deprecated)` mentions in text files and docs as `DialoGPT (deprecated) (deprecated)` and replace model ids with `distilgpt2 (deprecated)`.
  - Create simple backups of changed files with `.bak` (or `.bak2` if backups already exist).

Usage
-----
Dry-run (recommended first):

```bash
python scripts/deprecate_dialogpt.py
```

Apply changes:

```bash
python scripts/deprecate_dialogpt.py --apply
```

How to review & make a PR
-------------------------
1. Run the dry-run and inspect proposed files.
2. Commit the changes from `--apply` on a feature branch, run tests and lint.
3. Create a PR titled "Deprecate DialoGPT (deprecated): replace literals with env-driven fallback" and include this README as the PR description or attach the output of the dry-run.

Notes & limitations
-------------------
- This helper uses conservative text transformations and is intentionally simple. It will not perform AST-aware refactors for all edge cases.
- After applying changes, run the test suite and start a few agents to ensure runtime behavior remains correct.
- The default fallback model is `distilgpt2`. Set the environment variable `DIALOGPT_REPLACEMENT_MODEL` to change the runtime replacement.
