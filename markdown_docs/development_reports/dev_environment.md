# Developer environment for JustNewsAgent

This document explains how to create or update a development environment for
contributors. The dev environment contains developer-only tools such as formatters,
linters and pre-commit hooks and is intentionally separate from the runtime
environment to reduce the risk of dependency drift.

Files added:
- `dev-environment.yml` — conda environment specification for the dev env
- `scripts/setup_dev_environment.sh` — helper script to create the dev env or
  install dev tools into the existing runtime env

Recommended workflows

1. Create a dedicated dev environment (recommended):

   # Use mamba if available for speed
   mamba env create -f dev-environment.yml
   conda activate justnews-v2-py312-dev
   pre-commit install

2. Install dev tools into the existing runtime environment

   # Export a snapshot to revert if necessary
   conda env export -n justnews-v2-py312 > env-snapshot-before-devtools.yml
   # Install tooling into the existing env
   ./scripts/setup_dev_environment.sh --install-into-existing

Notes and policy

- After adding or changing dependencies in any environment files, the repository
  policy requires running Codacy CLI with the `trivy` tool to scan for new
  vulnerabilities. The Codacy analysis is triggered automatically by CI, but
  maintainers should also run it locally after changing environment files.
- If you add new dev dependencies, update `dev-environment.yml` and create a
  short PR documenting the reasons and the added tools.
