#!/usr/bin/env python3
"""
JustNewsAgent Catalogue Cross-Reference Repair Tool
Phase 1 Implementation: Fix broken cross-references and orphaned documents
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple
from datetime import datetime

class CrossReferenceRepair:
    def __init__(self, catalogue_path: str = "../docs_catalogue_v2.json"):
        self.catalogue_path = Path(catalogue_path)
        with open(self.catalogue_path, 'r') as f:
            self.catalogue = json.load(f)

        self.all_doc_ids = set()
        self.referenced_docs = set()
        self.broken_refs = set()

        self._analyze_references()

    def _analyze_references(self):
        """Analyze all references in the catalogue"""
        # Collect all document IDs
        for category in self.catalogue['categories']:
            for doc in category['documents']:
                self.all_doc_ids.add(doc['id'])

        # Find all referenced documents
        for category in self.catalogue['categories']:
            for doc in category['documents']:
                if 'related_documents' in doc:
                    for ref in doc['related_documents']:
                        if ref in self.all_doc_ids:
                            self.referenced_docs.add(ref)
                        else:
                            self.broken_refs.add(ref)

    def get_broken_references(self) -> Set[str]:
        """Get all broken reference IDs"""
        return self.broken_refs

    def get_orphaned_documents(self) -> Set[str]:
        """Get all orphaned document IDs"""
        return self.all_doc_ids - self.referenced_docs

    def get_document_info(self, doc_id: str) -> Dict:
        """Get document information by ID"""
        for category in self.catalogue['categories']:
            for doc in category['documents']:
                if doc['id'] == doc_id:
                    return {
                        'id': doc['id'],
                        'title': doc['title'],
                        'path': doc['path'],
                        'category': category['name'],
                        'tags': doc.get('tags', [])
                    }
        return None

    def suggest_repairs(self) -> Dict[str, List[str]]:
        """Suggest repairs for broken references"""
        suggestions = {}

        for broken_ref in self.broken_refs:
            # Try to find similar document names
            suggestions[broken_ref] = []
            broken_lower = broken_ref.lower()

            for doc_id in self.all_doc_ids:
                doc_info = self.get_document_info(doc_id)
                if doc_info:
                    title_lower = doc_info['title'].lower()
                    path_lower = doc_info['path'].lower()

                    # Check for similarity
                    if (broken_lower in title_lower or
                        broken_lower in path_lower or
                        any(word in title_lower for word in broken_lower.split('_'))):
                        suggestions[broken_ref].append(doc_id)

        return suggestions

    def generate_repair_report(self) -> str:
        """Generate comprehensive repair report"""
        broken_refs = self.get_broken_references()
        orphaned_docs = self.get_orphaned_documents()
        suggestions = self.suggest_repairs()

        report = [
            "=" * 60,
            "CROSS-REFERENCE REPAIR ANALYSIS",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "CURRENT STATUS:",
            f"- Total Documents: {len(self.all_doc_ids)}",
            f"- Documents with References: {len(self.referenced_docs)}",
            f"- Orphaned Documents: {len(orphaned_docs)}",
            f"- Broken References: {len(broken_refs)}",
            ""
        ]

        if broken_refs:
            report.extend([
                "BROKEN REFERENCES:",
                "-" * 20
            ])
            for ref in sorted(broken_refs):
                report.append(f"- {ref}")
                if ref in suggestions and suggestions[ref]:
                    report.append(f"  Suggested replacements: {', '.join(suggestions[ref][:3])}")
                report.append("")

        if len(orphaned_docs) > 0:
            report.extend([
                "ORPHANED DOCUMENTS (TOP 20):",
                "-" * 30
            ])
            for doc_id in sorted(list(orphaned_docs))[:20]:
                doc_info = self.get_document_info(doc_id)
                if doc_info:
                    report.append(f"- {doc_info['title']}")
                    report.append(f"  Path: {doc_info['path']}")
                    report.append(f"  Category: {doc_info['category']}")
                report.append("")

        report.extend([
            "RECOMMENDED ACTIONS:",
            "-" * 22,
            "1. Fix Broken References:",
            "   - Replace broken refs with suggested alternatives",
            "   - Remove invalid references",
            "",
            "2. Add Cross-References to Key Documents:",
            "   - Link orphaned docs to related documents",
            "   - Focus on core architectural documents first",
            "",
            "3. Update Catalogue:",
            "   - Apply fixes to docs_catalogue_v2.json",
            "   - Regenerate DOCUMENTATION_CATALOGUE.md",
            "",
            "4. Validate Results:",
            "   - Run cross-reference analysis again",
            "   - Ensure no new broken references introduced"
        ])

        return "\n".join(report)

def main():
    """Execute cross-reference repair analysis"""
    repair_tool = CrossReferenceRepair()

    print("🔧 Analyzing Cross-Reference Issues...")
    report = repair_tool.generate_repair_report()

    # Save detailed report
    report_path = Path("docs/CROSS_REFERENCE_REPAIR_REPORT.md")
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"✅ Repair analysis saved to: {report_path}")
    print("\n" + "="*60)
    print(report)

    # Show summary
    broken = repair_tool.get_broken_references()
    orphaned = repair_tool.get_orphaned_documents()

    print(f"\n📊 SUMMARY:")
    print(f"- Broken References: {len(broken)}")
    print(f"- Orphaned Documents: {len(orphaned)}")
    print(f"- Documents with References: {len(repair_tool.referenced_docs)}")

if __name__ == "__main__":
    main()
