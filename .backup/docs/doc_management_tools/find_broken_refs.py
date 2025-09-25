#!/usr/bin/env python3
"""
Quick script to identify remaining broken references
"""

import json
from pathlib import Path

def find_broken_refs():
    catalogue_path = Path("../docs_catalogue_v2.json")
    with open(catalogue_path, 'r') as f:
        catalogue = json.load(f)

    all_doc_ids = set()
    broken_refs = set()

    # Collect all document IDs
    for category in catalogue['categories']:
        for doc in category['documents']:
            all_doc_ids.add(doc['id'])

    # Find broken references
    for category in catalogue['categories']:
        for doc in category['documents']:
            if 'related_documents' in doc:
                for ref in doc['related_documents']:
                    if ref not in all_doc_ids:
                        broken_refs.add(ref)

    print(f"Broken references found: {len(broken_refs)}")
    for ref in sorted(broken_refs):
        print(f"- {ref}")

if __name__ == "__main__":
    find_broken_refs()
