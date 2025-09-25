#!/usr/bin/env python3
"""
JustNewsAgent Documentation Catalogue - Maintenance Action Plan
Generated: September 7, 2025

This script addresses the maintenance recommendations from the performance report:
1. Category 'Agent Documentation' has 36 documents (above average)
2. Category 'Development Reports' has 53 documents (above average)
3. Consider reviewing oldest document: agents/scout/README.md

Additional issues identified:
- 16 broken cross-references
- 129 orphaned documents
"""

import json
from datetime import datetime
from pathlib import Path


class CatalogueMaintenanceActions:
    def __init__(self, catalogue_path: str = "../docs_catalogue_v2.json"):
        self.catalogue_path = Path(catalogue_path)
        with open(self.catalogue_path) as f:
            self.catalogue = json.load(f)

    def analyze_category_distribution(self) -> dict[str, int]:
        """Analyze document distribution across categories"""
        distribution = {}
        for category in self.catalogue['categories']:
            distribution[category['name']] = len(category['documents'])
        return distribution

    def identify_large_categories(self) -> list[str]:
        """Identify categories with above-average document counts"""
        distribution = self.analyze_category_distribution()
        total_docs = sum(distribution.values())
        avg_docs = total_docs / len(distribution)

        large_categories = []
        for category, count in distribution.items():
            if count > avg_docs * 1.5:  # 50% above average
                large_categories.append(f"{category}: {count} documents")

        return large_categories

    def analyze_orphaned_documents(self) -> dict[str, list[str]]:
        """Analyze orphaned documents and suggest cross-references"""
        all_docs = set()
        referenced_docs = set()

        # Collect all document IDs
        for category in self.catalogue['categories']:
            for doc in category['documents']:
                all_docs.add(doc['id'])

        # Find referenced documents
        for category in self.catalogue['categories']:
            for doc in category['documents']:
                if 'related_documents' in doc:
                    referenced_docs.update(doc['related_documents'])

        orphaned = all_docs - referenced_docs
        broken_refs = referenced_docs - all_docs

        return {
            'orphaned_count': len(orphaned),
            'broken_refs_count': len(broken_refs),
            'orphaned_sample': list(orphaned)[:10],  # First 10 for brevity
            'broken_refs_sample': list(broken_refs)[:10]
        }

    def suggest_category_reorganization(self) -> list[str]:
        """Suggest reorganization for large categories"""
        suggestions = []

        # Agent Documentation category analysis
        agent_docs = []
        for category in self.catalogue['categories']:
            if category['name'] == 'Agent Documentation':
                for doc in category['documents']:
                    agent_docs.append(doc['title'])

        # Group by agent type
        agent_groups = {}
        for title in agent_docs:
            if 'scout' in title.lower():
                agent_groups.setdefault('Scout Agent', []).append(title)
            elif 'analyst' in title.lower():
                agent_groups.setdefault('Analyst Agent', []).append(title)
            elif 'synthesizer' in title.lower():
                agent_groups.setdefault('Synthesizer Agent', []).append(title)
            elif 'fact' in title.lower():
                agent_groups.setdefault('Fact Checker Agent', []).append(title)
            elif 'critic' in title.lower():
                agent_groups.setdefault('Critic Agent', []).append(title)
            elif 'chief' in title.lower():
                agent_groups.setdefault('Chief Editor Agent', []).append(title)
            elif 'memory' in title.lower():
                agent_groups.setdefault('Memory Agent', []).append(title)
            elif 'reasoning' in title.lower():
                agent_groups.setdefault('Reasoning Agent', []).append(title)
            else:
                agent_groups.setdefault('General Agent Docs', []).append(title)

        suggestions.append("Agent Documentation Category Reorganization:")
        for group, docs in agent_groups.items():
            if len(docs) > 3:
                suggestions.append(f"  - {group}: {len(docs)} documents - Consider separate subcategory")

        # Development Reports analysis
        dev_reports = []
        for category in self.catalogue['categories']:
            if category['name'] == 'Development Reports':
                for doc in category['documents']:
                    dev_reports.append(doc['title'])

        # Group by report type
        report_groups = {}
        for title in dev_reports:
            if 'gpu' in title.lower() or 'tensorrt' in title.lower():
                report_groups.setdefault('GPU/TensorRT Reports', []).append(title)
            elif 'training' in title.lower() or 'learning' in title.lower():
                report_groups.setdefault('Training System Reports', []).append(title)
            elif 'mcp' in title.lower() or 'bus' in title.lower():
                report_groups.setdefault('MCP Bus Reports', []).append(title)
            elif 'agent' in title.lower():
                report_groups.setdefault('Agent Development Reports', []).append(title)
            else:
                report_groups.setdefault('General Development Reports', []).append(title)

        suggestions.append("\nDevelopment Reports Category Reorganization:")
        for group, docs in report_groups.items():
            if len(docs) > 5:
                suggestions.append(f"  - {group}: {len(docs)} documents - Consider separate subcategory")

        return suggestions

    def generate_maintenance_report(self) -> str:
        """Generate comprehensive maintenance report"""
        report_lines = [
            "=" * 60,
            "JUSTNEWSAGENT DOCUMENTATION CATALOGUE",
            "MAINTENANCE ACTION PLAN",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "CURRENT STATUS:",
            f"- Total Documents: {self.catalogue['catalogue_metadata']['total_documents']}",
            f"- Total Categories: {len(self.catalogue['categories'])}",
            f"- Last Updated: {self.catalogue['catalogue_metadata']['last_updated']}",
            ""
        ]

        # Category distribution
        distribution = self.analyze_category_distribution()
        report_lines.extend([
            "CATEGORY DISTRIBUTION:",
            "-" * 30
        ])
        for category, count in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
            report_lines.append(f"{category}: {count} documents")

        # Large categories
        large_cats = self.identify_large_categories()
        if large_cats:
            report_lines.extend([
                "",
                "CATEGORIES ABOVE AVERAGE SIZE:",
                "-" * 35
            ])
            report_lines.extend(large_cats)

        # Cross-reference analysis
        orphan_analysis = self.analyze_orphaned_documents()
        report_lines.extend([
            "",
            "CROSS-REFERENCE ANALYSIS:",
            "-" * 30,
            f"Orphaned Documents: {orphan_analysis['orphaned_count']}",
            f"Broken References: {orphan_analysis['broken_refs_count']}",
            "",
            "SAMPLE ORPHANED DOCUMENTS:",
            "-" * 30
        ])
        for doc in orphan_analysis['orphaned_sample']:
            report_lines.append(f"- {doc}")

        if orphan_analysis['broken_refs_sample']:
            report_lines.extend([
                "",
                "SAMPLE BROKEN REFERENCES:",
                "-" * 30
            ])
            for ref in orphan_analysis['broken_refs_sample']:
                report_lines.append(f"- {ref}")

        # Reorganization suggestions
        suggestions = self.suggest_category_reorganization()
        if suggestions:
            report_lines.extend([
                "",
                "REORGANIZATION SUGGESTIONS:",
                "-" * 32
            ])
            report_lines.extend(suggestions)

        # Action plan
        report_lines.extend([
            "",
            "RECOMMENDED ACTIONS:",
            "-" * 22,
            "1. Fix Broken Cross-References:",
            "   - Review and repair 16 broken references",
            "   - Add cross-references to 129 orphaned documents",
            "",
            "2. Category Reorganization:",
            "   - Split 'Agent Documentation' into agent-specific subcategories",
            "   - Split 'Development Reports' by report type",
            "",
            "3. Documentation Enhancement:",
            "   - Add cross-reference sections to key documents",
            "   - Update oldest documents with current information",
            "",
            "4. Quality Improvements:",
            "   - Review documents with short descriptions",
            "   - Add missing tags to improve searchability",
            "",
            "5. Maintenance Automation:",
            "   - Schedule regular cross-reference validation",
            "   - Monitor category sizes for optimal organization"
        ])

        return "\n".join(report_lines)

def main():
    """Execute maintenance analysis and generate report"""
    maintainer = CatalogueMaintenanceActions()

    print("🔧 Generating Maintenance Action Plan...")
    report = maintainer.generate_maintenance_report()

    # Save to file
    report_path = Path("../MAINTENANCE_ACTION_PLAN.md")
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"✅ Maintenance report saved to: {report_path}")
    print("\n" + "="*60)
    print(report)

if __name__ == "__main__":
    main()
