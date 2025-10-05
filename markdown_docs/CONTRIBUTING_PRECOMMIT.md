# Pre-commit and local linting (quick start)

This short guide explains how to install and run the pre-commit hooks used by this repository so you can catch ShellCheck/yamllint issues locally before CI.

Recommended (safe) setup

1) Install pre-commit (recommended via pipx):

   ```bash
   # using pipx (recommended)
   pipx install pre-commit
   ```

   ```bash
   # or fallback (per-user pip install)
   python3 -m pip install --user pre-commit
   ```

2) Install pre-commit hooks for this repo (one-time per-repo):

   ```bash
   cd /path/to/JustNewsAgent
   pre-commit install
   ```

3) Run the hooks across all files (useful before opening a PR):

   ```bash
   pre-commit run --all-files
   ```

Key developer tooling to install

- ShellCheck (shell script linter)
  - Ubuntu/Debian: `sudo apt-get install -y shellcheck`
  - macOS (Homebrew): `brew install shellcheck`
  - Manual run: `shellcheck -x ./square-one.sh`

- yamllint (YAML linter)
  - Install in a virtualenv or via pipx: `pipx install yamllint`
  - Manual run (CI config): `yamllint -c .yamllint .github/workflows/ci.yml`

Notes about CI

- CI runs pre-commit (includes ShellCheck and yamllint) early in the job and treats ShellCheck failures as fatal. Fix shellcheck errors locally before pushing to avoid CI failures.
- A lightweight `.yamllint` file is included to relax a few rules for workflow files (long lines, certain truthy syntaxes). If you want stricter YAML checks, propose changes and we can tighten that config.

Troubleshooting

- If a hook fails in CI but passes locally, ensure you have the same versions of the tooling (use pipx or create a virtualenv per the project's recommendations).
- To inspect ShellCheck suggestions, see the ShellCheck website: `https://www.shellcheck.net/` and search the SC#### code shown by the tool.

Quick checklist before opening PRs

- Run: `pre-commit run --all-files`
- Run: `pytest -q` (run unit tests)
- Ensure no ShellCheck warnings remain (fix quoting/unused functions as shown by shellcheck)

Thanks — these checks keep the repo consistent and protect CI from avoidable failures
