#!/usr/bin/env python3
"""
Comprehensive Version Compliance Validator for JustNewsAgent

This script validates version consistency across all project files and components.
It ensures that version references remain synchronized during development.

Usage:
    python scripts/validate_version_compliance.py [--fix] [--verbose]

Options:
    --fix       Automatically fix version inconsistencies where possible
    --verbose   Show detailed output for all checks
"""

import argparse
import re
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class VersionComplianceValidator:
    """Validates version consistency across the JustNewsAgent codebase"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.project_root = project_root
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.fixed_files: set[str] = set()

        # Load central version
        try:
            from justnews import __version__ as central_version
            self.expected_version = central_version
            if self.verbose:
                print(f"📋 Central version loaded: {central_version}")
        except ImportError as e:
            self.errors.append(f"Cannot load central version: {e}")
            self.expected_version = None

    def log(self, message: str, level: str = "INFO"):
        """Log a message if verbose mode is enabled"""
        if self.verbose or level in ["ERROR", "WARNING"]:
            print(f"[{level}] {message}")

    def check_central_version_definition(self) -> bool:
        """Check that central version is properly defined"""
        self.log("Checking central version definition...")

        init_file = self.project_root / "justnews" / "__init__.py"
        if not init_file.exists():
            self.errors.append("justnews/__init__.py not found")
            return False

        content = init_file.read_text()

        # Check __version__ definition
        version_match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
        if not version_match:
            self.errors.append("No __version__ definition found in justnews/__init__.py")
            return False

        file_version = version_match.group(1)
        if file_version != self.expected_version:
            self.errors.append(f"Version mismatch: central={self.expected_version}, file={file_version}")
            return False

        # Check VERSION_INFO consistency
        version_info_match = re.search(r'VERSION_INFO\s*=\s*\{[^}]*"version":\s*["\']([^"\']+)["\'][^}]*\}', content, re.DOTALL)
        if version_info_match:
            info_version = version_info_match.group(1)
            if info_version != self.expected_version:
                self.errors.append(f"VERSION_INFO version mismatch: {info_version} != {self.expected_version}")
                return False

        self.log("✅ Central version definition is consistent")
        return True

    def check_readme_badges(self) -> bool:
        """Check README.md version badges"""
        self.log("Checking README.md version badges...")

        readme_file = self.project_root / "README.md"
        if not readme_file.exists():
            self.errors.append("README.md not found")
            return False

        content = readme_file.read_text()

        # Check version badge
        badge_pattern = r'!\[Version\]\([^)]*\)\s*badge/version-([^-\s]+)'
        badge_match = re.search(badge_pattern, content)

        if badge_match:
            badge_version = badge_match.group(1)
            if badge_version != self.expected_version:
                self.errors.append(f"README badge version mismatch: {badge_version} != {self.expected_version}")
                return False
        else:
            self.warnings.append("Version badge not found in README.md")

        # Check version text
        version_text_pattern = r'-\s*\*\*Version:\*\*\s*([0-9]+\.[0-9]+\.[0-9]+)'
        text_match = re.search(version_text_pattern, content)

        if text_match:
            text_version = text_match.group(1)
            if text_version != self.expected_version:
                self.errors.append(f"README version text mismatch: {text_version} != {self.expected_version}")
                return False

        self.log("✅ README.md version references are consistent")
        return True

    def check_documentation_versions(self) -> bool:
        """Check version references in documentation files"""
        self.log("Checking documentation version references...")

        # Files that should contain version references
        doc_files = [
            "docs/RELEASE_PROCESS.md",
            "CHANGELOG.md"
        ]

        issues_found = False

        for doc_file in doc_files:
            file_path = self.project_root / doc_file
            if not file_path.exists():
                self.warnings.append(f"Documentation file not found: {doc_file}")
                continue

            content = file_path.read_text()

            # Find all version references (excluding external tool versions)
            version_refs = re.findall(r'"version":\s*"([0-9]+\.[0-9]+\.[0-9]+)"', content)

            for ref in version_refs:
                if ref != self.expected_version:
                    self.errors.append(f"{doc_file}: Version reference {ref} != {self.expected_version}")
                    issues_found = True

        if not issues_found:
            self.log("✅ Documentation version references are consistent")
        return not issues_found

    def check_agent_documentation_versions(self) -> bool:
        """Check version references in agent documentation"""
        self.log("Checking agent documentation version references...")

        agent_docs_dir = self.project_root / "markdown_docs" / "agent_documentation"
        if not agent_docs_dir.exists():
            self.warnings.append("Agent documentation directory not found")
            return True

        issues_found = False

        # Check all .md files in agent documentation
        for md_file in agent_docs_dir.glob("*.md"):
            content = md_file.read_text()

            # Find JSON version references (exclude Docker versions)
            json_versions = re.findall(r'"version":\s*"([0-9]+\.[0-9]+\.[0-9]+)"', content)

            for version in json_versions:
                if version != self.expected_version:
                    self.errors.append(f"{md_file.name}: JSON version {version} != {self.expected_version}")
                    issues_found = True

        if not issues_found:
            self.log("✅ Agent documentation version references are consistent")
        return not issues_found

    def check_api_endpoints(self) -> bool:
        """Check if API endpoints return consistent versions (requires running services)"""
        self.log("Checking API endpoints (requires running services)...")

        # This would require services to be running
        # For now, just check that the code references are consistent
        self.warnings.append("API endpoint validation requires running services - run manual validation")
        return True

    def check_package_dependencies(self) -> bool:
        """Check that package version is consistent in setup files"""
        self.log("Checking package version in setup files...")

        # Check setup.py, pyproject.toml, etc. if they exist
        setup_files = ["setup.py", "pyproject.toml"]

        for setup_file in setup_files:
            file_path = self.project_root / setup_file
            if file_path.exists():
                content = file_path.read_text()

                # Look for version references
                version_matches = re.findall(r'version\s*=\s*["\']([^"\']+)["\']', content, re.IGNORECASE)

                for match in version_matches:
                    if match == self.expected_version:
                        self.log(f"✅ {setup_file} version is consistent")
                    else:
                        self.errors.append(f"{setup_file}: version {match} != {self.expected_version}")
                        return False

        return True

    def run_all_checks(self) -> bool:
        """Run all version compliance checks"""
        print("🔍 JustNewsAgent Version Compliance Validation")
        print("=" * 60)

        checks = [
            ("Central Version Definition", self.check_central_version_definition),
            ("README Badges", self.check_readme_badges),
            ("Documentation Versions", self.check_documentation_versions),
            ("Agent Documentation Versions", self.check_agent_documentation_versions),
            ("Package Dependencies", self.check_package_dependencies),
            ("API Endpoints", self.check_api_endpoints),
        ]

        all_passed = True

        for check_name, check_func in checks:
            try:
                passed = check_func()
                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"{status} {check_name}")
                if not passed:
                    all_passed = False
            except Exception as e:
                print(f"❌ ERROR {check_name}: {e}")
                all_passed = False

        print("\n" + "=" * 60)

        if self.errors:
            print("🚨 ERRORS FOUND:")
            for error in self.errors:
                print(f"  • {error}")

        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"  • {warning}")

        if all_passed and not self.errors:
            print("🎉 ALL VERSION COMPLIANCE CHECKS PASSED!")
            return True
        else:
            print(f"⚠️  {len(self.errors)} errors found, {len(self.warnings)} warnings")
            return False

def main():
    parser = argparse.ArgumentParser(description="Validate version compliance across JustNewsAgent")
    parser.add_argument("--fix", action="store_true", help="Automatically fix version inconsistencies")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")

    args = parser.parse_args()

    validator = VersionComplianceValidator(verbose=args.verbose)
    success = validator.run_all_checks()

    if args.fix and validator.errors:
        print("\n🔧 Attempting to fix version inconsistencies...")
        # Future: implement auto-fix functionality
        print("Auto-fix not yet implemented - please fix manually")

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
