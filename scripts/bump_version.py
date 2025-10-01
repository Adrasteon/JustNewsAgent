#!/usr/bin/env python3
"""
Version Bump Automation Script for JustNewsAgent

This script automates the process of updating version references across the entire codebase
when releasing a new version. It ensures all version references stay synchronized.

Usage:
    python scripts/bump_version.py <new_version> [--dry-run] [--verbose]

Examples:
    python scripts/bump_version.py 0.9.0
    python scripts/bump_version.py 1.0.0 --dry-run --verbose
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class VersionBumper:
    """Automates version updates across the JustNewsAgent codebase"""

    def __init__(self, new_version: str, dry_run: bool = False, verbose: bool = False):
        self.new_version = new_version
        self.dry_run = dry_run
        self.verbose = verbose
        self.project_root = Path(__file__).parent.parent
        self.files_modified: List[str] = []

        # Validate version format
        if not re.match(r"^\d+\.\d+\.\d+$", new_version):
            raise ValueError(f"Invalid version format: {new_version}. Expected: x.y.z")

        # Load current version
        try:
            sys.path.insert(0, str(self.project_root))
            from justnews import __version__ as current_version

            self.current_version = current_version
            if self.verbose:
                print(f"📋 Current version: {current_version}")
                print(f"📋 New version: {new_version}")
        except ImportError as e:
            raise RuntimeError(f"Cannot load current version: {e}")

        if current_version == new_version:
            raise ValueError(
                f"New version {new_version} is the same as current version"
            )

    def log(self, message: str):
        """Log a message if verbose mode is enabled"""
        if self.verbose:
            print(f"[INFO] {message}")

    def backup_file(self, file_path: Path) -> Path:
        """Create a backup of a file"""
        backup_path = file_path.with_suffix(file_path.suffix + ".backup")
        if not self.dry_run:
            backup_path.write_text(file_path.read_text())
        self.log(f"Created backup: {backup_path}")
        return backup_path

    def update_central_version(self) -> bool:
        """Update the central version definition in justnews/__init__.py"""
        self.log("Updating central version definition...")

        init_file = self.project_root / "justnews" / "__init__.py"
        if not init_file.exists():
            print(f"❌ Error: {init_file} not found")
            return False

        content = init_file.read_text()

        # Update __version__
        new_content = re.sub(
            r'(__version__\s*=\s*)["\']([^"\']+)["\']',
            f'\\1"{self.new_version}"',
            content,
        )

        # Update VERSION_INFO version
        new_content = re.sub(
            r'("version":\s*)["\']([^"\']+)["\']',
            f'\\1"{self.new_version}"',
            new_content,
        )

        if not self.dry_run:
            self.backup_file(init_file)
            init_file.write_text(new_content)
            self.files_modified.append(str(init_file))

        self.log("✅ Central version updated")
        return True

    def update_readme_badges(self) -> bool:
        """Update version badges in README.md"""
        self.log("Updating README.md version badges...")

        readme_file = self.project_root / "README.md"
        if not readme_file.exists():
            print(f"❌ Error: {readme_file} not found")
            return False

        content = readme_file.read_text()

        # Update version badge - simple replacement
        new_content = content.replace(
            f"version-{self.current_version}", f"version-{self.new_version}"
        )

        # Update version text - simple replacement
        new_content = new_content.replace(
            f"**Version:** {self.current_version}", f"**Version:** {self.new_version}"
        )

        if content != new_content and not self.dry_run:
            self.backup_file(readme_file)
            readme_file.write_text(new_content)
            self.files_modified.append(str(readme_file))

        self.log("✅ README.md updated")
        return True

    def update_documentation_versions(self) -> bool:
        """Update version references in documentation files"""
        self.log("Updating documentation version references...")

        doc_files = ["docs/RELEASE_PROCESS.md", "CHANGELOG.md"]

        for doc_file in doc_files:
            file_path = self.project_root / doc_file
            if not file_path.exists():
                self.log(f"Warning: {doc_file} not found")
                continue

            content = file_path.read_text()

            # Update JSON version references
            new_content = re.sub(
                rf'"version":\s*"{re.escape(self.current_version)}"',
                f'"version": "{self.new_version}"',
                content,
            )

            if content != new_content and not self.dry_run:
                self.backup_file(file_path)
                file_path.write_text(new_content)
                self.files_modified.append(str(file_path))

        self.log("✅ Documentation files updated")
        return True

    def update_agent_documentation_versions(self) -> bool:
        """Update version references in agent documentation"""
        self.log("Updating agent documentation version references...")

        agent_docs_dir = self.project_root / "markdown_docs" / "agent_documentation"
        if not agent_docs_dir.exists():
            self.log("Warning: Agent documentation directory not found")
            return True

        for md_file in agent_docs_dir.glob("*.md"):
            content = md_file.read_text()

            # Update JSON version references
            new_content = re.sub(
                rf'"version":\s*"{re.escape(self.current_version)}"',
                f'"version": "{self.new_version}"',
                content,
            )

            if content != new_content and not self.dry_run:
                self.backup_file(md_file)
                md_file.write_text(new_content)
                self.files_modified.append(str(md_file))

        self.log("✅ Agent documentation updated")
        return True

    def update_package_files(self) -> bool:
        """Update version in package configuration files"""
        self.log("Updating package configuration files...")

        package_files = ["setup.py", "pyproject.toml"]

        for pkg_file in package_files:
            file_path = self.project_root / pkg_file
            if not file_path.exists():
                continue

            content = file_path.read_text()

            # Update version references
            new_content = re.sub(
                rf'(version\s*=\s*)["\']{re.escape(self.current_version)}["\']',
                f'\\1"{self.new_version}"',
                content,
                flags=re.IGNORECASE,
            )

            if content != new_content and not self.dry_run:
                self.backup_file(file_path)
                file_path.write_text(new_content)
                self.files_modified.append(str(file_path))

        self.log("✅ Package files updated")
        return True

    def run_all_updates(self) -> bool:
        """Run all version update operations"""
        print(f"🔄 Bumping version from {self.current_version} to {self.new_version}")
        if self.dry_run:
            print("🔍 DRY RUN MODE - No files will be modified")
        print("=" * 60)

        updates = [
            ("Central Version", self.update_central_version),
            ("README Badges", self.update_readme_badges),
            ("Documentation", self.update_documentation_versions),
            ("Agent Documentation", self.update_agent_documentation_versions),
            ("Package Files", self.update_package_files),
        ]

        all_success = True

        for update_name, update_func in updates:
            try:
                success = update_func()
                status = "✅ PASS" if success else "❌ FAIL"
                print(f"{status} {update_name}")
                if not success:
                    all_success = False
            except Exception as e:
                print(f"❌ ERROR {update_name}: {e}")
                all_success = False

        print("\n" + "=" * 60)

        if self.files_modified:
            print("📝 Files modified:")
            for file in self.files_modified:
                print(f"  • {file}")
        else:
            print("📝 No files were modified")

        if all_success:
            if self.dry_run:
                print("🎉 DRY RUN COMPLETED - All updates would succeed!")
            else:
                print("🎉 VERSION BUMP COMPLETED SUCCESSFULLY!")
                print("\nNext steps:")
                print("1. Run tests: pytest")
                print(
                    "2. Validate version: python scripts/validate_version_compliance.py"
                )
                print("3. Update CHANGELOG.md with release notes")
                print(
                    "4. Commit changes: git add . && git commit -m 'Bump version to {self.new_version}'"
                )
        else:
            print("⚠️  Some updates failed - please check errors above")

        return all_success


def main():
    parser = argparse.ArgumentParser(
        description="Bump version across JustNewsAgent codebase"
    )
    parser.add_argument("new_version", help="New version number (format: x.y.z)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed output"
    )

    args = parser.parse_args()

    try:
        bumper = VersionBumper(
            args.new_version, dry_run=args.dry_run, verbose=args.verbose
        )
        success = bumper.run_all_updates()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
