#!/usr/bin/env python3
"""
JustNewsAgent Catalogue Cross-Reference Fixer
Phase 1 Implementation: Fix broken cross-references in catalogue
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple
from datetime import datetime

class CrossReferenceFixer:
    def __init__(self, catalogue_path: str = "../docs_catalogue_v2.json"):
        self.catalogue_path = Path(catalogue_path)
        with open(self.catalogue_path, 'r') as f:
            self.catalogue = json.load(f)

        # Manual mapping of broken references to correct ones
        self.reference_fixes = {
            "analytics_system": "markdown_docs_development_reports_system_architecture_assessment",
            "archive_integration": "markdown_docs_development_reports_synthesizer_training_integration_success",
            "deployment_success": "markdown_docs_production_status_synthesizer_v3_production_success",
            "ewc_optimization": "markdown_docs_production_status_memory_optimization_success_summary",
            "gpu_dashboard": "gpu_runner_readme",
            "gpu_validation": "markdown_docs_development_reports_production_validation_summary",
            "legal_compliance": "legal_compliance_framework",
            "mcp_bus": "mcp_bus_architecture",
            "monitoring": "markdown_docs_development_reports_system_architecture_assessment",  # fallback
            "scout_enhanced_crawl": "markdown_docs_agent_documentation_scout_enhanced_deep_crawl_documentation",
            "security": "markdown_docs_development_reports_system_architecture_assessment",  # fallback
            "synthesizer_v3_success": "markdown_docs_production_status_synthesizer_v3_production_success",
            "training_integration": "markdown_docs_development_reports_synthesizer_training_integration_success",
            "training_system": "markdown_docs_development_reports_training_system_documentation"
        }

    def fix_broken_references(self) -> Dict[str, int]:
        """Fix all broken cross-references in the catalogue"""
        fixes_applied = {}

        for category in self.catalogue['categories']:
            for doc in category['documents']:
                if 'related_documents' in doc:
                    original_refs = doc['related_documents'].copy()
                    fixed_refs = []

                    for ref in original_refs:
                        if ref in self.reference_fixes:
                            # Replace with correct reference
                            correct_ref = self.reference_fixes[ref]
                            fixed_refs.append(correct_ref)
                            fixes_applied[ref] = fixes_applied.get(ref, 0) + 1
                            print(f"✅ Fixed: {ref} → {correct_ref}")
                        else:
                            # Keep as-is if not broken
                            fixed_refs.append(ref)

                    doc['related_documents'] = fixed_refs

        return fixes_applied

    def add_missing_cross_references(self) -> Dict[str, int]:
        """Add cross-references to orphaned documents"""
        # Find orphaned documents
        all_doc_ids = set()
        referenced_docs = set()

        for category in self.catalogue['categories']:
            for doc in category['documents']:
                all_doc_ids.add(doc['id'])
                if 'related_documents' in doc:
                    referenced_docs.update(doc['related_documents'])

        orphaned = all_doc_ids - referenced_docs

        # Add cross-references to key orphaned documents
        key_orphaned_fixes = {
            "readme": ["technical_architecture", "project_status", "changelog"],
            "agents_scout_readme": ["scout_agent_v2", "technical_architecture", "mcp_bus_architecture"],
            "agents_analyst_native_agent_readme": ["markdown_docs_production_status_fact_checker_fixes_success", "technical_architecture"],
            "agents_analyst_native_tensorrt_readme": ["markdown_docs_production_status_fact_checker_fixes_success", "gpu_runner_readme"],
            "agents_analyst_tensorrt_quickstart": ["agents_analyst_native_tensorrt_readme", "gpu_runner_readme"],
            "agents_newsreader_readme": ["markdown_docs_agent_documentation_scout_enhanced_deep_crawl_documentation", "technical_architecture"],
            "agents_reasoning_readme": ["markdown_docs_agent_documentation_reasoning_agent_complete_implementation", "technical_architecture"],
            "markdown_docs_agent_documentation_readme": ["markdown_docs_agent_documentation_agent_model_map", "technical_architecture"]
        }

        fixes_applied = {}
        for doc_id, refs in key_orphaned_fixes.items():
            if doc_id in orphaned:
                # Find the document and add references
                for category in self.catalogue['categories']:
                    for doc in category['documents']:
                        if doc['id'] == doc_id:
                            if 'related_documents' not in doc:
                                doc['related_documents'] = []
                            doc['related_documents'].extend(refs)
                            fixes_applied[doc_id] = len(refs)
                            print(f"✅ Added {len(refs)} cross-references to orphaned doc: {doc_id}")
                            break

        return fixes_applied

    def validate_fixes(self) -> Dict[str, int]:
        """Validate that fixes resolved the issues"""
        # Check for remaining broken references
        all_doc_ids = set()
        referenced_docs = set()
        broken_refs = set()

        for category in self.catalogue['categories']:
            for doc in category['documents']:
                all_doc_ids.add(doc['id'])
                if 'related_documents' in doc:
                    for ref in doc['related_documents']:
                        if ref in all_doc_ids:
                            referenced_docs.add(ref)
                        else:
                            broken_refs.add(ref)

        orphaned = all_doc_ids - referenced_docs

        return {
            'total_docs': len(all_doc_ids),
            'referenced_docs': len(referenced_docs),
            'orphaned_docs': len(orphaned),
            'broken_refs': len(broken_refs),
            'broken_ref_list': list(broken_refs) if broken_refs else []
        }

    def save_catalogue(self):
        """Save the fixed catalogue"""
        # Update metadata
        self.catalogue['catalogue_metadata']['last_updated'] = datetime.now().strftime('%Y-%m-%d')
        self.catalogue['catalogue_metadata']['description'] = "Comprehensive documentation catalogue with fixed cross-references and enhanced navigation"

        with open(self.catalogue_path, 'w') as f:
            json.dump(self.catalogue, f, indent=2)

        print(f"✅ Catalogue saved to: {self.catalogue_path}")

def main():
    """Execute Phase 1: Cross-reference fixes"""
    print("🔧 Starting Phase 1: Cross-Reference Fixes")
    print("=" * 50)

    fixer = CrossReferenceFixer()

    # Step 1: Fix broken references
    print("\n📋 Step 1: Fixing broken cross-references...")
    broken_fixes = fixer.fix_broken_references()
    print(f"✅ Applied {sum(broken_fixes.values())} fixes for {len(broken_fixes)} broken references")

    # Step 2: Add cross-references to key orphaned documents
    print("\n📋 Step 2: Adding cross-references to orphaned documents...")
    orphaned_fixes = fixer.add_missing_cross_references()
    print(f"✅ Added cross-references to {len(orphaned_fixes)} key orphaned documents")

    # Step 3: Validate fixes
    print("\n📋 Step 3: Validating fixes...")
    validation = fixer.validate_fixes()
    print("Validation Results:")
    print(f"  - Total Documents: {validation['total_docs']}")
    print(f"  - Referenced Documents: {validation['referenced_docs']}")
    print(f"  - Orphaned Documents: {validation['orphaned_docs']}")
    print(f"  - Broken References: {validation['broken_refs']}")

    if validation['broken_refs'] > 0:
        print(f"  ⚠️  Remaining broken references: {validation['broken_ref_list']}")

    # Step 4: Save catalogue
    print("\n📋 Step 4: Saving updated catalogue...")
    fixer.save_catalogue()

    # Summary
    print("\n" + "=" * 50)
    print("🎉 PHASE 1 COMPLETE!")
    print("=" * 50)
    print("Summary of fixes applied:")
    print(f"- Broken references fixed: {sum(broken_fixes.values())}")
    print(f"- Orphaned documents enhanced: {len(orphaned_fixes)}")
    print(f"- Final broken references: {validation['broken_refs']}")
    print(f"- Final orphaned documents: {validation['orphaned_docs']}")

    if validation['broken_refs'] == 0:
        print("✅ SUCCESS: All broken cross-references resolved!")
    else:
        print("⚠️  PARTIAL: Some broken references remain - manual review needed")

if __name__ == "__main__":
    main()
