#!/usr/bin/env python3
"""
Fix the final 2 broken cross-references
"""

import json
from pathlib import Path

def fix_final_broken_refs():
    catalogue_path = Path("docs/docs_catalogue_v2.json")
    with open(catalogue_path, 'r') as f:
        catalogue = json.load(f)

    # Fix the 2 remaining broken references
    fixes = {
        "markdown_docs_agent_documentation_agent_model_map": "agent_model_map",
        "markdown_docs_agent_documentation_reasoning_agent_complete_implementation": "markdown_docs_agent_documentation_reasoning_agent_complete_implementation"
    }

    fixed_count = 0
    for category in catalogue['categories']:
        for doc in category['documents']:
            if 'related_documents' in doc:
                original_refs = doc['related_documents'].copy()
                fixed_refs = []

                for ref in original_refs:
                    if ref in fixes:
                        fixed_refs.append(fixes[ref])
                        fixed_count += 1
                        print(f"✅ Fixed: {ref} → {fixes[ref]}")
                    else:
                        fixed_refs.append(ref)

                doc['related_documents'] = fixed_refs

    # Save the catalogue
    with open(catalogue_path, 'w') as f:
        json.dump(catalogue, f, indent=2)

    print(f"✅ Fixed {fixed_count} final broken references")

if __name__ == "__main__":
    fix_final_broken_refs()
