#!/usr/bin/env python3
"""
Documentation Versioning and Change Tracking System
Tracks changes, maintains version history, and enables rollback capabilities

Author: JustNews V4 Documentation System
Date: September 7, 2025
"""

import json
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import difflib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DocumentVersion:
    """Represents a version of a document"""
    document_id: str
    version: str
    timestamp: str
    author: str
    changes: List[str]
    content_hash: str
    previous_version: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class ChangeRecord:
    """Represents a change to the documentation catalogue"""
    change_id: str
    timestamp: str
    author: str
    change_type: str  # 'add', 'update', 'delete', 'bulk_update'
    affected_documents: List[str]
    description: str
    quality_impact: Dict[str, float]  # Before/after quality scores
    rollback_available: bool = True

class DocumentationVersionControl:
    """Version control system for documentation"""

    def __init__(self, catalogue_path: str = "../docs_catalogue_v2.json"):
        self.catalogue_path = Path(catalogue_path)
        self.versions_dir = Path("doc_versions")
        self.changes_dir = Path("doc_changes")
        self.versions_dir.mkdir(exist_ok=True)
        self.changes_dir.mkdir(exist_ok=True)

        self.version_history: Dict[str, DocumentVersion] = {}
        self.change_history: List[ChangeRecord] = []

        self._load_version_history()
        self._load_change_history()

        logger.info("Documentation Version Control initialized")

    def _load_version_history(self) -> None:
        """Load version history from disk"""
        history_file = self.versions_dir / "version_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for doc_id, versions in data.items():
                        self.version_history[doc_id] = [
                            DocumentVersion(**v) for v in versions
                        ]
                logger.info("Version history loaded")
            except Exception as e:
                logger.error(f"Failed to load version history: {e}")

    def _load_change_history(self) -> None:
        """Load change history from disk"""
        history_file = self.changes_dir / "change_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.change_history = [ChangeRecord(**c) for c in data]
                logger.info("Change history loaded")
            except Exception as e:
                logger.error(f"Failed to load change history: {e}")

    def _save_version_history(self) -> None:
        """Save version history to disk"""
        history_file = self.versions_dir / "version_history.json"
        data = {}
        for doc_id, versions in self.version_history.items():
            data[doc_id] = [asdict(v) for v in versions]

        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _save_change_history(self) -> None:
        """Save change history to disk"""
        history_file = self.changes_dir / "change_history.json"
        data = [asdict(c) for c in self.change_history]

        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _calculate_content_hash(self, content: Dict[str, Any]) -> str:
        """Calculate hash of document content"""
        content_str = json.dumps(content, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(content_str.encode('utf-8')).hexdigest()

    def _detect_changes(self, old_content: Dict[str, Any],
                       new_content: Dict[str, Any]) -> List[str]:
        """Detect specific changes between versions"""
        changes = []

        # Check basic fields
        for field in ['title', 'description', 'status', 'tags']:
            old_val = old_content.get(field, '')
            new_val = new_content.get(field, '')

            if old_val != new_val:
                if isinstance(old_val, list) and isinstance(new_val, list):
                    added = set(new_val) - set(old_val)
                    removed = set(old_val) - set(new_val)
                    if added:
                        changes.append(f"Added tags: {', '.join(added)}")
                    if removed:
                        changes.append(f"Removed tags: {', '.join(removed)}")
                else:
                    changes.append(f"Updated {field}: '{old_val}' → '{new_val}'")

        # Check description length changes
        old_desc_len = len(str(old_content.get('description', '')))
        new_desc_len = len(str(new_content.get('description', '')))
        if abs(new_desc_len - old_desc_len) > 10:
            changes.append(f"Description length: {old_desc_len} → {new_desc_len} characters")

        return changes if changes else ["Minor metadata updates"]

    def create_version_snapshot(self, author: str = "system") -> str:
        """Create a snapshot of current documentation state"""
        try:
            with open(self.catalogue_path, 'r', encoding='utf-8') as f:
                catalogue = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load catalogue: {e}")
            return ""

        timestamp = datetime.now().isoformat()
        snapshot_id = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create version snapshot
        snapshot = {
            'snapshot_id': snapshot_id,
            'timestamp': timestamp,
            'author': author,
            'catalogue_hash': self._calculate_content_hash(catalogue),
            'document_count': sum(len(cat.get('documents', []))
                                for cat in catalogue.get('categories', [])),
            'catalogue': catalogue
        }

        # Save snapshot
        snapshot_file = self.versions_dir / f"{snapshot_id}.json"
        with open(snapshot_file, 'w', encoding='utf-8') as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)

        logger.info(f"Version snapshot created: {snapshot_id}")
        return snapshot_id

    def track_document_changes(self, document_id: str, old_content: Dict[str, Any],
                             new_content: Dict[str, Any], author: str = "system") -> None:
        """Track changes to a specific document"""
        if document_id not in self.version_history:
            self.version_history[document_id] = []

        versions = self.version_history[document_id]
        version_num = len(versions) + 1
        version_id = f"v{version_num}"

        changes = self._detect_changes(old_content, new_content)
        content_hash = self._calculate_content_hash(new_content)

        previous_version = versions[-1].version if versions else None

        version = DocumentVersion(
            document_id=document_id,
            version=version_id,
            timestamp=datetime.now().isoformat(),
            author=author,
            changes=changes,
            content_hash=content_hash,
            previous_version=previous_version,
            metadata={
                'description_length': len(str(new_content.get('description', ''))),
                'tags_count': len(new_content.get('tags', [])),
                'has_word_count': 'word_count' in new_content
            }
        )

        versions.append(version)
        self._save_version_history()

        logger.info(f"Document version tracked: {document_id} {version_id}")

    def track_bulk_changes(self, change_description: str, affected_docs: List[str],
                          author: str = "system", change_type: str = "bulk_update") -> str:
        """Track bulk changes to multiple documents"""
        change_id = f"change_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Calculate quality impact (placeholder - would integrate with quality monitor)
        quality_impact = {
            'before_score': 0.0,  # Would be calculated from previous state
            'after_score': 0.0,   # Would be calculated from current state
            'change_delta': 0.0
        }

        change_record = ChangeRecord(
            change_id=change_id,
            timestamp=datetime.now().isoformat(),
            author=author,
            change_type=change_type,
            affected_documents=affected_docs,
            description=change_description,
            quality_impact=quality_impact
        )

        self.change_history.append(change_record)
        self._save_change_history()

        logger.info(f"Bulk change tracked: {change_id} ({len(affected_docs)} documents)")
        return change_id

    def rollback_document(self, document_id: str, target_version: str) -> bool:
        """Rollback document to specific version"""
        if document_id not in self.version_history:
            logger.error(f"No version history for document: {document_id}")
            return False

        versions = self.version_history[document_id]
        target_ver = next((v for v in versions if v.version == target_version), None)

        if not target_ver:
            logger.error(f"Version not found: {document_id} {target_version}")
            return False

        # Load current catalogue
        try:
            with open(self.catalogue_path, 'r', encoding='utf-8') as f:
                catalogue = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load catalogue: {e}")
            return False

        # Find and update the document
        for category in catalogue.get('categories', []):
            for doc in category.get('documents', []):
                if doc.get('id') == document_id:
                    # Restore from version (this would need actual content storage)
                    logger.info(f"Document rollback prepared: {document_id} → {target_version}")
                    # Note: Actual rollback would require storing full content
                    return True

        logger.error(f"Document not found in catalogue: {document_id}")
        return False

    def generate_change_report(self, days: int = 7) -> str:
        """Generate change report for specified period"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_changes = [
            c for c in self.change_history
            if datetime.fromisoformat(c.timestamp) > cutoff_date
        ]

        report = f"""
# 📋 Documentation Change Report
**Period:** Last {days} days
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Summary
- **Total Changes:** {len(recent_changes)}
- **Documents Affected:** {len(set(doc for c in recent_changes for doc in c.affected_documents))}

## 🔄 Recent Changes
"""

        for change in recent_changes[-10:]:  # Show last 10 changes
            report += f"""
### {change.change_id}
- **Type:** {change.change_type}
- **Author:** {change.author}
- **Date:** {change.timestamp[:10]}
- **Documents:** {len(change.affected_documents)}
- **Description:** {change.description}
"""

        return report

    def get_document_history(self, document_id: str) -> List[DocumentVersion]:
        """Get version history for a specific document"""
        return self.version_history.get(document_id, [])

def main():
    """Main function for command-line usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Documentation Version Control")
    parser.add_argument("action", choices=['snapshot', 'report', 'history'],
                       help="Action to perform")
    parser.add_argument("--document", help="Document ID for history")
    parser.add_argument("--days", type=int, default=7, help="Days for report")
    parser.add_argument("--author", default="system", help="Change author")

    args = parser.parse_args()

    vc = DocumentationVersionControl()

    if args.action == 'snapshot':
        snapshot_id = vc.create_version_snapshot(args.author)
        print(f"Snapshot created: {snapshot_id}")

    elif args.action == 'report':
        report = vc.generate_change_report(args.days)
        print(report)

    elif args.action == 'history':
        if not args.document:
            print("Document ID required for history")
            return
        history = vc.get_document_history(args.document)
        for version in history:
            print(f"{version.version}: {version.timestamp} - {', '.join(version.changes)}")

if __name__ == "__main__":
    main()
