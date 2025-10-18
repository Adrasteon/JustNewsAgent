---
title: Repository cleanup summary
created: 2025-10-18
---

# Repository cleanup performed (brief)

This file summarizes the recent housekeeping work performed on the repository and suggests recommended next steps for maintainers.

Summary of actions performed

- Created branch `maint/cleanup` to stage and review cleanup changes.
- Removed or archived large tracked artifacts by moving them (with history preserved) into `archive/`:
  - `archive/backups/` contains historical backups that were previously checked in.
  - `archive/reports/` contains generated reports and export artifacts.
  - `archive/release_preview/` contains the release preview tree and companion README files.
- Regenerated the documentation index: `docs_index.json` (from frontmatter in `markdown_docs`).
- Ran the documentation linter with safe fixes (`doc_linter.py --fix`) and applied deterministic frontmatter/fence normalizations where missing.
- Ran the doc link/anchor/stub checker (`scripts/doc_check.py`) and created a script to apply deterministic link fixes for references to files moved into `archive/`.
- Committed deterministic link fixes to `maint/cleanup` (commit message: `chore(docs): auto-fix archive links`).

What remains / recommended next steps

1. Review and repair the central documentation catalogue files:
   - `markdown_docs/development_reports/DOCUMENTATION_CATALOGUE.md`
   - `docs/DOCUMENTATION_CATALOGUE.md`
   These two files reference many targets. Fixing them (either by updating links to canonical locations or by pruning/organising references) will reduce the majority of broken link reports.

2. Fix missing anchors and heading slugs
   - The doc checker reported ~50 missing anchors. Decide whether to:
     - add the missing headings to the target documents, or
     - update links to point to existing headings.
   Anchor fixes typically require human review.

3. Review 'probable stubs'
   - 50+ files were flagged as potential stubs. Review and expand or remove them. Consider a short grooming task to triage these.

4. Optionally run external link checks
   - Running `scripts/doc_check.py --check-external` will validate internet links. This takes longer and requires network access.

5. Prepare a PR from `maint/cleanup`
   - Open a PR for the `maint/cleanup` branch with this summary, the archive moves, the linter fixes, and the deterministic link fixes. Add a clear list of remaining manual tasks.

6. (Optional) Remove untracked caches after review
   - After maintainers confirm, run `git clean -n` to preview and `git clean -fdx` to remove local build caches and untracked temporary files.

Notes and rationale

- Tracked `git mv` moves were used when archiving to preserve history and to make the changes reviewable via a single PR.
- The auto-fixer script only applies deterministic maps where a basename matches exactly one file under `archive/`. This avoids risky, ambiguous rewrites.
- The next phase should prioritize the documentation catalogue and anchor fixes: these will have the largest impact on the checker results.

If you want, I can:

- generate a prioritized remediation report (group broken links by target), or
- attempt a second automated pass that proposes edits to the catalogue files (changes shown as a dry-run first), or
- open the PR for `maint/cleanup` with notes and a checklist.

-- cleanup bot
