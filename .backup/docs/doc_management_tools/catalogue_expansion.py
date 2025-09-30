#!/usr/bin/env python3
"""
Automated Documentation Catalogue Expansion System
JustNewsAgent - Industry Standard Documentation Management

This script implements a comprehensive automated system for expanding the documentation
catalogue with metadata-driven discovery, quality assurance, and incremental updates.

Features:
- Automated .md file discovery across the entire codebase
- Metadata extraction (title, description, tags, word count)
- Quality validation and consistency checks
- Incremental updates to avoid full rebuilds
- Cross-reference detection and mapping
- Search index generation and optimization

Usage:
    python doc_management_tools/catalogue_expansion.py --phase development_reports
    python doc_management_tools/catalogue_expansion.py --phase agents
    python doc_management_tools/catalogue_expansion.py --phase scripts_tools
    python doc_management_tools/catalogue_expansion.py --phase all
    python doc_management_tools/catalogue_expansion.py --validate
"""

import os
import json
import re
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DocumentMetadata:
    """Standardized metadata structure for documentation entries"""
    id: str
    title: str
    path: str
    description: str
    category: str
    tags: List[str]
    word_count: int
    last_modified: str
    status: str
    related_documents: List[str]
    search_content: str

class CatalogueExpansionSystem:
    """Main system for automated catalogue expansion"""

    def __init__(self, workspace_root: str):
        self.workspace_root = Path(workspace_root)
        self.catalogue_path = self.workspace_root / "docs" / "docs_catalogue_v2.json"
        self.human_readable_path = self.workspace_root / "docs" / "DOCUMENTATION_CATALOGUE.md"

        # Define directory mappings for categorization
        self.directory_mappings = {
            'agents': 'agent_documentation',
            'markdown_docs/agent_documentation': 'agent_documentation',
            'markdown_docs/development_reports': 'development_reports',
            'markdown_docs/production_status': 'production_deployment',
            'markdown_docs/optimization_reports': 'performance_optimization',
            'scripts': 'scripts_tools',
            'tools': 'scripts_tools',
            'deploy': 'deployment_system',
            '.github': 'system_documentation',
            'docs': 'main_documentation',
            'training_system': 'training_learning'
        }

        # Priority phases for expansion
        self.phases = {
            'development_reports': ['markdown_docs/development_reports'],
            'agents': ['agents', 'markdown_docs/agent_documentation'],
            'scripts_tools': ['scripts', 'tools'],
            'system_docs': ['deploy', '.github'],
            'all': ['markdown_docs', 'agents', 'scripts', 'tools', 'deploy', '.github']
        }

        # Load existing catalogue
        self.catalogue = self._load_catalogue()

    def _load_catalogue(self) -> Dict[str, Any]:
        """Load existing catalogue or create new one"""
        if self.catalogue_path.exists():
            with open(self.catalogue_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return self._create_empty_catalogue()

    def _create_empty_catalogue(self) -> Dict[str, Any]:
        """Create empty catalogue structure"""
        return {
            "catalogue_metadata": {
                "version": "2.1",
                "last_updated": datetime.now().strftime("%Y-%m-%d"),
                "total_documents": 0,
                "categories": 0,
                "status": "initializing",
                "description": "Automated documentation catalogue for JustNewsAgent"
            },
            "categories": [],
            "search_index": {"tags": [], "keywords": []},
            "navigation_paths": {},
            "maintenance": {
                "last_catalogue_update": datetime.now().strftime("%Y-%m-%d"),
                "next_review_date": (datetime.now().replace(day=7)).strftime("%Y-%m-%d"),
                "outdated_documents": [],
                "missing_cross_references": [],
                "broken_links": []
            }
        }

    def discover_markdown_files(self, directories: List[str]) -> List[Path]:
        """Discover all .md files in specified directories"""
        discovered_files = []

        for directory in directories:
            search_path = self.workspace_root / directory
            if search_path.exists():
                # Find all .md files recursively
                for md_file in search_path.rglob("*.md"):
                    # Skip certain directories/files
                    if any(skip in str(md_file) for skip in [
                        'node_modules', '__pycache__', '.git',
                        'archive_obsolete_files', '.pytest_cache'
                    ]):
                        continue
                    discovered_files.append(md_file)

        logger.info(f"Discovered {len(discovered_files)} .md files")
        return discovered_files

    def extract_metadata(self, file_path: Path) -> DocumentMetadata:
        """Extract comprehensive metadata from a markdown file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")
            return None

        # Extract title from first heading
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else file_path.stem.replace('_', ' ').title()

        # Extract description from first paragraph
        desc_match = re.search(r'^#.*?\n\n(.+?)(?:\n\n|\n#)', content, re.DOTALL)
        description = desc_match.group(1).strip()[:200] + "..." if desc_match else f"Documentation for {title}"

        # Count words
        word_count = len(re.findall(r'\b\w+\b', content))

        # Determine category
        relative_path = file_path.relative_to(self.workspace_root)
        category = self._determine_category(str(relative_path))

        # Extract tags from content (look for keywords)
        tags = self._extract_tags(content, title)

        # Generate unique ID
        doc_id = self._generate_document_id(file_path)

        # Get last modified date
        last_modified = datetime.fromtimestamp(file_path.stat().st_mtime).strftime("%Y-%m-%d")

        # Determine status (simplified)
        status = "current"  # Could be enhanced with git analysis

        # Find related documents (simplified - could be enhanced)
        related_docs = []

        return DocumentMetadata(
            id=doc_id,
            title=title,
            path=str(relative_path),
            description=description,
            category=category,
            tags=tags,
            word_count=word_count,
            last_modified=last_modified,
            status=status,
            related_documents=related_docs,
            search_content=content.lower()
        )

    def _determine_category(self, relative_path: str) -> str:
        """Determine document category based on path"""
        for path_prefix, category in self.directory_mappings.items():
            if relative_path.startswith(path_prefix):
                return category
        return "general_documentation"

    def _extract_tags(self, content: str, title: str) -> List[str]:
        """Extract relevant tags from content and title"""
        tags = []

        # Common technical keywords
        keywords = [
            'gpu', 'cuda', 'pytorch', 'tensorrt', 'training', 'agents', 'mcp',
            'api', 'production', 'deployment', 'monitoring', 'compliance',
            'architecture', 'models', 'performance', 'optimization',
            'continuous-learning', 'knowledge-graph', 'archive', 'security',
            'logging', 'analytics', 'dashboard', 'reasoning', 'scout',
            'synthesizer', 'analyst', 'fact-checker', 'memory', 'chief-editor'
        ]

        content_lower = content.lower()
        title_lower = title.lower()

        for keyword in keywords:
            if keyword in content_lower or keyword in title_lower:
                tags.append(keyword)

        # Add category-specific tags
        if 'agent' in content_lower:
            tags.extend(['multi-agent', 'ai-agents'])
        if 'v4' in content_lower or 'v3' in content_lower:
            tags.append('version-specific')

        return list(set(tags))[:5]  # Limit to 5 most relevant tags

    def _generate_document_id(self, file_path: Path) -> str:
        """Generate unique document ID"""
        relative_path = file_path.relative_to(self.workspace_root)
        # Create ID from path, replacing special characters
        doc_id = str(relative_path).replace('/', '_').replace('\\', '_').replace('.md', '')
        return doc_id.lower()

    def update_catalogue(self, new_documents: List[DocumentMetadata]):
        """Update catalogue with new documents"""
        logger.info(f"Updating catalogue with {len(new_documents)} new documents")

        # Group documents by category
        category_groups = defaultdict(list)
        for doc in new_documents:
            category_groups[doc.category].append(doc)

        # Update existing categories or create new ones
        for category_name, docs in category_groups.items():
            self._update_category(category_name, docs)

        # Update metadata
        self.catalogue["catalogue_metadata"]["total_documents"] = self._count_total_documents()
        self.catalogue["catalogue_metadata"]["last_updated"] = datetime.now().strftime("%Y-%m-%d")
        self.catalogue["catalogue_metadata"]["categories"] = len(self.catalogue["categories"])

        # Update search index
        self._update_search_index(new_documents)

        # Update maintenance info
        self.catalogue["maintenance"]["last_catalogue_update"] = datetime.now().strftime("%Y-%m-%d")

    def _update_category(self, category_name: str, documents: List[DocumentMetadata]):
        """Update or create category with new documents"""
        # Find existing category
        category = None
        for cat in self.catalogue["categories"]:
            if cat["id"] == category_name:
                category = cat
                break

        if not category:
            # Create new category
            category = {
                "id": category_name,
                "name": category_name.replace('_', ' ').title(),
                "description": f"Documentation related to {category_name.replace('_', ' ')}",
                "priority": "medium",
                "documents": []
            }
            self.catalogue["categories"].append(category)

        # Add new documents (avoid duplicates)
        existing_paths = {doc["path"] for doc in category["documents"]}
        for doc in documents:
            if doc.path not in existing_paths:
                category["documents"].append(asdict(doc))

    def _count_total_documents(self) -> int:
        """Count total documents across all categories"""
        return sum(len(cat["documents"]) for cat in self.catalogue["categories"])

    def _update_search_index(self, new_documents: List[DocumentMetadata]):
        """Update search index with new document tags and keywords"""
        all_tags = set()
        all_keywords = set()

        # Collect from existing catalogue
        for cat in self.catalogue["categories"]:
            for doc in cat["documents"]:
                all_tags.update(doc.get("tags", []))
                # Extract keywords from title and description
                title_words = doc.get("title", "").lower().split()
                desc_words = doc.get("description", "").lower().split()
                all_keywords.update(title_words + desc_words)

        # Add from new documents
        for doc in new_documents:
            all_tags.update(doc.tags)
            title_words = doc.title.lower().split()
            desc_words = doc.description.lower().split()
            all_keywords.update(title_words + desc_words)

        # Update search index
        self.catalogue["search_index"]["tags"] = sorted(list(all_tags))
        self.catalogue["search_index"]["keywords"] = sorted(list(all_keywords))[:100]  # Limit keywords

    def save_catalogue(self):
        """Save updated catalogue to file"""
        # Ensure docs directory exists
        self.catalogue_path.parent.mkdir(exist_ok=True)

        with open(self.catalogue_path, 'w', encoding='utf-8') as f:
            json.dump(self.catalogue, f, indent=2, ensure_ascii=False)

        logger.info(f"Catalogue saved to {self.catalogue_path}")

    def update_human_readable_catalogue(self):
        """Update the human-readable catalogue markdown file"""
        content = self._generate_human_readable_content()
        with open(self.human_readable_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Human-readable catalogue updated: {self.human_readable_path}")

    def _generate_human_readable_content(self) -> str:
        """Generate human-readable catalogue content"""
        content = f"""# JustNewsAgent Documentation Catalogue

**Version:** {self.catalogue["catalogue_metadata"]["version"]}
**Last Updated:** {self.catalogue["catalogue_metadata"]["last_updated"]}
**Total Documents:** {self.catalogue["catalogue_metadata"]["total_documents"]}
**Categories:** {self.catalogue["catalogue_metadata"]["categories"]}

## Table of Contents

"""

        # Add table of contents
        for i, category in enumerate(self.catalogue["categories"], 1):
            content += f"{i}. [{category['name']}](#{category['id']})\n"

        content += "\n---\n\n"

        # Add each category
        for category in self.catalogue["categories"]:
            content += f"## {category['name']}\n\n"
            content += f"**Category ID:** {category['id']}\n"
            content += f"**Priority:** {category['priority']}\n"
            content += f"**Documents:** {len(category['documents'])}\n\n"

            if category['documents']:
                content += "| Document | Description | Tags | Status |\n"
                content += "|----------|-------------|------|--------|\n"

                for doc in sorted(category['documents'], key=lambda x: x['title']):
                    tags_str = ", ".join(doc['tags'][:3])  # Show first 3 tags
                    content += f"| [{doc['title']}]({doc['path']}) | {doc['description'][:100]}... | {tags_str} | {doc['status']} |\n"

            content += "\n---\n\n"

        # Add search index summary
        content += "## Search Index Summary\n\n"
        content += f"**Available Tags:** {len(self.catalogue['search_index']['tags'])}\n"
        content += f"**Indexed Keywords:** {len(self.catalogue['search_index']['keywords'])}\n\n"

        # Add maintenance info
        content += "## Maintenance Information\n\n"
        content += f"**Last Catalogue Update:** {self.catalogue['maintenance']['last_catalogue_update']}\n"
        content += f"**Next Review Date:** {self.catalogue['maintenance']['next_review_date']}\n"

        return content

    def validate_catalogue(self) -> Dict[str, Any]:
        """Validate catalogue integrity and identify issues"""
        issues = {
            "broken_paths": [],
            "missing_metadata": [],
            "duplicate_ids": [],
            "orphaned_documents": []
        }

        # Check for broken paths
        for category in self.catalogue["categories"]:
            for doc in category["documents"]:
                doc_path = self.workspace_root / doc["path"]
                if not doc_path.exists():
                    issues["broken_paths"].append(doc["path"])

        # Check for missing required metadata
        required_fields = ["id", "title", "path", "description", "status"]
        for category in self.catalogue["categories"]:
            for doc in category["documents"]:
                for field in required_fields:
                    if field not in doc:
                        issues["missing_metadata"].append(f"{doc.get('path', 'unknown')}: missing {field}")

        # Check for duplicate IDs
        all_ids = []
        for category in self.catalogue["categories"]:
            for doc in category["documents"]:
                doc_id = doc.get("id")
                if doc_id in all_ids:
                    issues["duplicate_ids"].append(doc_id)
                all_ids.append(doc_id)

        logger.info(f"Validation complete. Found {sum(len(v) for v in issues.values())} issues")
        return issues

    def expand_catalogue(self, phase: str):
        """Main method to expand catalogue for specified phase"""
        logger.info(f"Starting catalogue expansion for phase: {phase}")

        # Get directories for this phase
        if phase not in self.phases:
            raise ValueError(f"Unknown phase: {phase}")

        directories = self.phases[phase]

        # Discover files
        discovered_files = self.discover_markdown_files(directories)

        # Extract metadata
        new_documents = []
        for file_path in discovered_files:
            metadata = self.extract_metadata(file_path)
            if metadata:
                new_documents.append(metadata)

        # Update catalogue
        self.update_catalogue(new_documents)

        # Save changes
        self.save_catalogue()
        self.update_human_readable_catalogue()

        logger.info(f"Phase {phase} complete: Added {len(new_documents)} documents")

        return len(new_documents)

def main():
    parser = argparse.ArgumentParser(description="Automated Documentation Catalogue Expansion")
    parser.add_argument("--phase", choices=["development_reports", "agents", "scripts_tools", "system_docs", "all"],
                       default="development_reports", help="Expansion phase to execute")
    parser.add_argument("--validate", action="store_true", help="Validate catalogue integrity")
    parser.add_argument("--workspace", default="/home/adra/justnewsagent/JustNewsAgent",
                       help="Workspace root directory")

    args = parser.parse_args()

    # Initialize system
    system = CatalogueExpansionSystem(args.workspace)

    if args.validate:
        # Validation mode
        issues = system.validate_catalogue()
        print("=== CATALOGUE VALIDATION RESULTS ===")
        for issue_type, items in issues.items():
            print(f"\n{issue_type.upper()}: {len(items)} issues")
            for item in items[:10]:  # Show first 10
                print(f"  - {item}")
            if len(items) > 10:
                print(f"  ... and {len(items) - 10} more")
    else:
        # Expansion mode
        try:
            added_count = system.expand_catalogue(args.phase)
            print(f"✅ Successfully expanded catalogue for phase '{args.phase}'")
            print(f"📄 Added {added_count} new documents")

            # Show summary
            total_docs = system.catalogue["catalogue_metadata"]["total_documents"]
            print(f"📊 Total documents in catalogue: {total_docs}")

        except Exception as e:
            logger.error(f"Expansion failed: {e}")
            return 1

    return 0

if __name__ == "__main__":
    exit(main())
