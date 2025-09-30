---
title: Testing & Dependency Upgrade: Paused (2025-08-24)
description: Auto-generated description for Testing & Dependency Upgrade: Paused (2025-08-24)
tags: [documentation]
status: current
last_updated: 2025-09-12
---

# Testing & Dependency Upgrade: Paused (2025-08-24)

Summary
-------
This document records the dependency-testing work performed and the reason we paused further
efforts. The goal was to remove upstream DeprecationWarnings triggered during pytest collection
without masking them.

What we did
- Created a disposable conda environment `justnews-upgrade-test` to safely iterate on upgrades.
- Deferred import-time heavy initializations in repository files (spacy/transformers/Trainer/etc.).
- Captured exact installed package set into `requirements-pinned.txt` for reproducibility.
- Added `environment.yml` to create a conda environment (conda-forge + pip block) that uses the
  pinned requirements file.
- Added a lightweight GitHub Actions workflow (manual dispatch only) to create the env and run
  pytest non-integration tests.
- Performed a stepwise trial: attempted safe upgrades and then tried a spaCy 4.x pre-release in the
  disposable env to assess compatibility.

Key findings
- Several DeprecationWarnings originate from upstream packages (spaCy/weasel) importing
  `click.parser.split_arg_string` and from SWIG-generated compiled types. These are not caused by
  repository code and require upstream fixes or a controlled upgrade to newer major releases.
- Installing the spaCy 4 pre-release changed the warning surface but did not fully eliminate
  warnings; spaCy 4 is pre-release and introduces additional compatibility work.

Why we paused
- Upgrading to spaCy 4.x and related packages is a breaking change that requires a dedicated
  compatibility effort (code changes, extensive test runs, and possibly model artifact updates).
- The immediate testing goal (make pytest collection fast and reduce noise) was achieved by
  deferring heavy import-time initializations and producing a pinned environment for reproducible
  runs.

Next recommended steps when resuming
1. Create a feature branch and plan a controlled spaCy 4 upgrade: bump packages, run the full
   test matrix, and fix API changes. Use the pinned environment as the starting point.
2. Add CI jobs for staged upgrades (unit -> integration -> e2e) and a rollback plan.
3. Consider engaging upstream maintainers if a minimal non-breaking fix exists for the
   split_arg_string usage in weasel/spaCy.

Contact
-------
If you want me to continue, I can open the feature branch and start a guided upgrade to spaCy 4.x
or revert the pre-release trial and freeze the current pinned environment for production use.

## See also

- Technical Architecture: markdown_docs/TECHNICAL_ARCHITECTURE.md
- Documentation Catalogue: docs/DOCUMENTATION_CATALOGUE.md

