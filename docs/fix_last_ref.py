#!/usr/bin/env python3
"""
Fix the final broken cross-reference with correct document ID
"""

import json
from pathlib import Path

def fix_last_broken_ref():
    catalogue_path = Path("docs/docs_catalogue_v2.json")
    with open(catalogue_path, 'r') as f:
        catalogue = json.load(f)

    # The broken reference is pointing to the wrong ID
    # It should point to "reasoning_agent" instead of "markdown_docs_agent_documentation_reasoning_agent_complete_implementation"
    wrong_id = "markdown_docs_agent_documentation_reasoning_agent_complete_implementation"
    correct_id = "reasoning_agent"

    fixed_count = 0
    for category in catalogue['categories']:
        for doc in category['documents']:
            if 'related_documents' in doc:
                if wrong_id in doc['related_documents']:
                    # Replace the wrong ID with the correct one
                    doc['related_documents'] = [
                        correct_id if ref == wrong_id else ref
                        for ref in doc['related_documents']
                    ]
                    fixed_count += 1
                    print(f"✅ Fixed: {wrong_id} → {correct_id}")

    # Save the catalogue
    with open(catalogue_path, 'w') as f:
        json.dump(catalogue, f, indent=2)

    print(f"✅ Fixed {fixed_count} final broken reference")

if __name__ == "__main__":
    fix_last_broken_ref()
