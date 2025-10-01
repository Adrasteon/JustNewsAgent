#!/usr/bin/env python3
"""
Verify required systemd docs exist and are wired in catalogue and index.

Checks:
- Files exist: deploy/systemd/{README.md, QUICK_REFERENCE.md, COMPREHENSIVE_SYSTEMD_GUIDE.md, postgresql_integration.md}
- docs_catalogue_v2.json contains category `deployment_systemd` and lists those paths
- docs_index.json contains category "Deployment (systemd)" and lists those files (accepts absolute or relative paths)

Exit code 0 on success, 1 on any failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REQUIRED_REL_PATHS = [
    "deploy/systemd/README.md",
    "deploy/systemd/QUICK_REFERENCE.md",
    "deploy/systemd/COMPREHENSIVE_SYSTEMD_GUIDE.md",
    "deploy/systemd/postgresql_integration.md",
]

REQUIRED_ARTIFACTS = {
    "restart_script": {
        "path": "deploy/systemd/reset_and_start.sh",
        "must_be_executable": True,
        "must_be_referenced_in": [
            "deploy/systemd/QUICK_REFERENCE.md",
            "README.md",
        ],
    },
    "cold_start_script": {
        "path": "deploy/systemd/cold_start.sh",
        "must_be_executable": True,
        "must_be_referenced_in": [
            "deploy/systemd/QUICK_REFERENCE.md",
            "README.md",
            "deploy/systemd/COMPREHENSIVE_SYSTEMD_GUIDE.md",
        ],
    },
    "cold_start_wrapper": {
        "path": "deploy/systemd/scripts/justnews-cold-start.sh",
        "must_be_executable": True,
        "must_be_referenced_in": [
            "deploy/systemd/QUICK_REFERENCE.md",
            "deploy/systemd/COMPREHENSIVE_SYSTEMD_GUIDE.md",
        ],
    },
    "cold_start_service": {
        "path": "deploy/systemd/units/justnews-cold-start.service",
        "must_be_referenced_in": [
            "deploy/systemd/COMPREHENSIVE_SYSTEMD_GUIDE.md",
        ],
    },
    "cold_start_timer": {
        "path": "deploy/systemd/units/justnews-cold-start.timer",
        "must_be_referenced_in": [
            "deploy/systemd/COMPREHENSIVE_SYSTEMD_GUIDE.md",
            "deploy/systemd/QUICK_REFERENCE.md",
        ],
    },
    "boot_smoke_helper": {
        "path": "deploy/systemd/helpers/boot_smoke_test.sh",
        "must_be_executable": True,
        "must_be_referenced_in": [
            "deploy/systemd/COMPREHENSIVE_SYSTEMD_GUIDE.md",
            "deploy/systemd/QUICK_REFERENCE.md",
        ],
    },
    "boot_smoke_wrapper": {
        "path": "deploy/systemd/scripts/justnews-boot-smoke.sh",
        "must_be_executable": True,
        "must_be_referenced_in": [
            "deploy/systemd/COMPREHENSIVE_SYSTEMD_GUIDE.md",
            "deploy/systemd/QUICK_REFERENCE.md",
        ],
    },
    "boot_smoke_service": {
        "path": "deploy/systemd/units/justnews-boot-smoke.service",
        "must_be_referenced_in": [
            "deploy/systemd/COMPREHENSIVE_SYSTEMD_GUIDE.md",
            "deploy/systemd/QUICK_REFERENCE.md",
        ],
    },
    "boot_smoke_timer": {
        "path": "deploy/systemd/units/justnews-boot-smoke.timer",
        "must_be_referenced_in": [
            "deploy/systemd/COMPREHENSIVE_SYSTEMD_GUIDE.md",
            "deploy/systemd/QUICK_REFERENCE.md",
        ],
    },
    "path_wrapper_enable_all": {
        "path": "deploy/systemd/scripts/enable_all.sh",
        "must_be_executable": True,
        "must_be_referenced_in": [
            "deploy/systemd/QUICK_REFERENCE.md",
        ],
    },
    "path_wrapper_health_check": {
        "path": "deploy/systemd/scripts/health_check.sh",
        "must_be_executable": True,
        "must_be_referenced_in": [
            "deploy/systemd/QUICK_REFERENCE.md",
        ],
    },
    "path_wrapper_reset_and_start": {
        "path": "deploy/systemd/scripts/reset_and_start.sh",
        "must_be_executable": True,
        "must_be_referenced_in": [
            "deploy/systemd/QUICK_REFERENCE.md",
        ],
    },
    "path_wrapper_cold_start": {
        "path": "deploy/systemd/scripts/cold_start.sh",
        "must_be_executable": True,
        "must_be_referenced_in": [
            "deploy/systemd/QUICK_REFERENCE.md",
        ],
    },
    "enable_all_alias": {
        "path": "deploy/systemd/enable_all.sh",
        "must_contain": "--fresh",
    },
}


def repo_root_from_this_file() -> Path:
    # .../scripts/ci/verify_docs_links.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


def check_files_exist(root: Path) -> tuple[bool, list[str]]:
    missing: list[str] = []
    for rel in REQUIRED_REL_PATHS:
        if not (root / rel).is_file():
            missing.append(rel)
    return (len(missing) == 0, missing)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def check_catalogue(root: Path) -> tuple[bool, str]:
    cat_path = root / "docs_catalogue_v2.json"
    if not cat_path.is_file():
        return False, f"Missing catalogue file: {cat_path}"
    data = load_json(cat_path)
    categories = data.get("categories", [])
    dep = next((c for c in categories if c.get("id") == "deployment_systemd"), None)
    if not dep:
        return (
            False,
            "Category 'deployment_systemd' not found in docs_catalogue_v2.json",
        )
    listed = {doc.get("path") for doc in dep.get("documents", [])}
    missing = [p for p in REQUIRED_REL_PATHS if p not in listed]
    if missing:
        return False, f"Catalogue missing paths: {missing}"
    return True, "ok"


def normalize_to_rel(raw_path: str, root: Path) -> str:
    # Accept absolute or relative; prefer extracting from 'deploy/systemd/' anchor
    anchor = "deploy/systemd/"
    if anchor in raw_path:
        return raw_path[raw_path.index(anchor) :]
    p = Path(raw_path)
    try:
        return str(p.resolve().relative_to(root))
    except Exception:
        return raw_path.lstrip("/")


def check_index(root: Path) -> tuple[bool, str]:
    idx_path = root / "docs_index.json"
    if not idx_path.is_file():
        return False, f"Missing index file: {idx_path}"
    data = load_json(idx_path)
    if not isinstance(data, list):
        return False, "docs_index.json must be a JSON array"
    dep = next((c for c in data if c.get("category") == "Deployment (systemd)"), None)
    if not dep:
        return False, "Category 'Deployment (systemd)' not found in docs_index.json"
    files = dep.get("files", [])
    rels = {normalize_to_rel(f.get("path", ""), root) for f in files}
    missing = [p for p in REQUIRED_REL_PATHS if p not in rels]
    if missing:
        return False, f"Index missing paths: {missing}"
    # Optional: ensure these files actually exist
    missing_fs = [p for p in REQUIRED_REL_PATHS if not (root / p).is_file()]
    if missing_fs:
        return False, f"Index references non-existent files: {missing_fs}"
    return True, "ok"


def main() -> int:
    root = repo_root_from_this_file()
    ok_files, missing = check_files_exist(root)
    ok_cat, msg_cat = check_catalogue(root)
    ok_idx, msg_idx = check_index(root)

    errors: list[str] = []
    if not ok_files:
        errors.append(f"Missing required files: {missing}")
    if not ok_cat:
        errors.append(msg_cat)
    if not ok_idx:
        errors.append(msg_idx)

    if errors:
        print("[docs-verify] FAIL:")
        for e in errors:
            print(f" - {e}")
        return 1

    # Extended operator-simplicity checks
    ext_errors: list[str] = []

    # Validate a set of artifacts with shared logic
    def validate_artifact(key: str, needle: str):
        spec = REQUIRED_ARTIFACTS[key]
        path = root / spec["path"]
        if not path.is_file():
            ext_errors.append(f"Missing {key.replace('_', ' ')}: {spec['path']}")
            return
        try:
            mode = path.stat().st_mode
            if spec.get("must_be_executable") and not (mode & 0o111):
                ext_errors.append(f"{spec['path']} not executable")
        except Exception:
            ext_errors.append(f"Unable to stat {spec['path']}")
        for rel_doc in spec.get("must_be_referenced_in", []):
            dpath = root / rel_doc
            if not dpath.is_file():
                ext_errors.append(f"Doc missing for reference check: {rel_doc}")
                continue
            content = dpath.read_text(encoding="utf-8", errors="ignore")
            if needle not in content:
                ext_errors.append(f"Doc does not reference {spec['path']}: {rel_doc}")

    validate_artifact("restart_script", "reset_and_start.sh")
    validate_artifact("cold_start_script", "cold_start.sh")
    validate_artifact("cold_start_wrapper", "justnews-cold-start.sh")
    validate_artifact("cold_start_service", "justnews-cold-start.service")
    validate_artifact("cold_start_timer", "justnews-cold-start.timer")
    validate_artifact("boot_smoke_helper", "boot_smoke_test.sh")
    validate_artifact("boot_smoke_wrapper", "justnews-boot-smoke.sh")
    validate_artifact("boot_smoke_service", "justnews-boot-smoke.service")
    validate_artifact("boot_smoke_timer", "justnews-boot-smoke.timer")
    validate_artifact("path_wrapper_enable_all", "enable_all.sh")
    validate_artifact("path_wrapper_health_check", "health_check.sh")
    validate_artifact("path_wrapper_reset_and_start", "reset_and_start.sh")
    validate_artifact("path_wrapper_cold_start", "cold_start.sh")

    ealias = REQUIRED_ARTIFACTS["enable_all_alias"]
    epath = root / ealias["path"]
    if not epath.is_file():
        ext_errors.append(f"Missing enable script: {ealias['path']}")
    else:
        etext = epath.read_text(encoding="utf-8", errors="ignore")
        if ealias.get("must_contain") and ealias["must_contain"] not in etext:
            ext_errors.append(f"enable_all.sh missing alias '{ealias['must_contain']}'")

    if ext_errors:
        print("[docs-verify] FAIL (operator-simplicity):")
        for e in ext_errors:
            print(f" - {e}")
        return 1

    print(
        "[docs-verify] PASS: systemd docs present and wired in catalogue/index + operator simplicity checks"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
