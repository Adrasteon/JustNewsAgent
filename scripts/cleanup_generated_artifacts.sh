#!/usr/bin/env bash
# Cleanup generated analysis artifacts from repository root.
# Moves discovered generated files into .backup/analysis_outputs and untracks
# them so Git clients (GitKraken, etc.) are no longer blocked by huge root files.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

# Ensure backup location exists
BACKUP_DIR="$REPO_ROOT/.backup/analysis_outputs"
mkdir -p "$BACKUP_DIR"

# Patterns to target (explicit files and simple globs)
TARGET_FILES=(
  "codacy_semgrep.json"
  ".md5_rewrite_report.json"
)
GLOBS=(
  "codacy_*.json"
  "*.sarif"
  ".md5_*.json"
)

# Enable nullglob so globs that match nothing expand to nothing (no errors)
shopt -s nullglob

echo "Cleanup generated analysis artifacts in: $REPO_ROOT"

# Process explicit filenames
for f in "${TARGET_FILES[@]}"; do
  if [[ -e "$f" ]]; then
    echo "Found $f"
    if [[ $DRY_RUN -eq 0 ]]; then
      # If tracked, remove from index (stages deletion). If not tracked, rm --cached will fail harmlessly.
      git rm --cached --ignore-unmatch -- "$f" || true
      # Move file to backup location so users still have the artifact locally
      mv -f -- "$f" "$BACKUP_DIR/" || true
      echo "Moved $f -> $BACKUP_DIR/"
    fi
  fi
done

# Process globs
for g in "${GLOBS[@]}"; do
  for file in $g; do
    echo "Found $file"
    if [[ $DRY_RUN -eq 0 ]]; then
      git rm --cached --ignore-unmatch -- "$file" || true
      mv -f -- "$file" "$BACKUP_DIR/" || true
      echo "Moved $file -> $BACKUP_DIR/"
    fi
  done
done

# If any changes staged (deletions), commit them
if [[ $DRY_RUN -eq 0 ]]; then
  # Check for staged deletions
  if git diff --cached --name-only | grep -qE "codacy_semgrep|md5_rewrite_report|\.sarif|codacy_"; then
    git commit -m "chore(scripts): untrack generated analysis artifacts and move to .backup/analysis_outputs" || true
    echo "Committed removal of generated artifacts."
  else
    echo "No generated artifacts were staged for removal."
  fi
fi

echo "Cleanup complete. Backup dir: $BACKUP_DIR"
