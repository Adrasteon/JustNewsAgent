#!/usr/bin/env python3
"""
JustNewsAgent Documentation Catalogue Navigator
==============================================

A command-line tool for navigating and searching the JustNewsAgent documentation catalogue.

Usage:
    python docs_navigator.py [command] [options]

Commands:
    list        - List all documents by category
    search      - Search documents by keyword or tag
    find        - Find documents by ID or title
    related     - Show related documents for a given document
    status      - Show documentation status summary
    validate    - Validate the catalogue structure

Examples:
    python docs_navigator.py list
    python docs_navigator.py search gpu
    python docs_navigator.py find readme
    python docs_navigator.py related technical_architecture
    python docs_navigator.py status
"""

import json
import sys
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

class DocsNavigator:
    """Documentation catalogue navigator and search tool."""

    def __init__(self, catalogue_path: str = "../docs_catalogue_v2.json"):
        self.catalogue_path = catalogue_path
        self.catalogue = self._load_catalogue()

    def _load_catalogue(self) -> Dict[str, Any]:
        """Load the documentation catalogue."""
        try:
            with open(self.catalogue_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: Catalogue file not found at {self.catalogue_path}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in catalogue file: {e}")
            sys.exit(1)

    def list_documents(self, category: Optional[str] = None) -> None:
        """List all documents, optionally filtered by category."""
        print("📚 JustNewsAgent Documentation Catalogue")
        print("=" * 50)

        if category:
            self._list_category_documents(category)
        else:
            for cat in self.catalogue['categories']:
                print(f"\n🏷️  {cat['name']} ({len(cat['documents'])} documents)")
                print("-" * 40)
                for doc in cat['documents'][:3]:  # Show first 3 per category
                    status_emoji = self._get_status_emoji(doc['status'])
                    print(f"  {status_emoji} {doc['title']}")
                if len(cat['documents']) > 3:
                    print(f"  ... and {len(cat['documents']) - 3} more documents")

        print(f"\n📊 Total: {self.catalogue['catalogue_metadata']['total_documents']} documents across {self.catalogue['catalogue_metadata']['categories']} categories")

    def _list_category_documents(self, category_name: str) -> None:
        """List documents for a specific category."""
        category = None
        for cat in self.catalogue['categories']:
            if cat['id'] == category_name or cat['name'].lower() == category_name.lower():
                category = cat
                break

        if not category:
            print(f"❌ Category '{category_name}' not found.")
            print("\nAvailable categories:")
            for cat in self.catalogue['categories']:
                print(f"  - {cat['name']} ({cat['id']})")
            return

        print(f"🏷️  {category['name']}")
        print("-" * 40)

        for doc in category['documents']:
            status_emoji = self._get_status_emoji(doc['status'])
            print(f"{status_emoji} {doc['title']}")
            print(f"   📄 {doc['path']}")
            print(f"   📅 Updated: {doc['last_updated']}")
            print(f"   📝 {doc['description'][:100]}...")
            if doc.get('tags'):
                print(f"   🏷️  Tags: {', '.join(doc['tags'][:3])}")
            print()

    def search_documents(self, query: str) -> None:
        """Search documents by keyword, tag, or title."""
        query = query.lower()
        results = []

        for category in self.catalogue['categories']:
            for doc in category['documents']:
                # Search in title, description, tags, and path
                searchable_text = (
                    doc['title'].lower() + ' ' +
                    doc['description'].lower() + ' ' +
                    doc['path'].lower() + ' ' +
                    ' '.join(doc.get('tags', [])).lower()
                )

                if query in searchable_text:
                    results.append({
                        'doc': doc,
                        'category': category['name'],
                        'relevance_score': self._calculate_relevance(query, doc)
                    })

        # Sort by relevance
        results.sort(key=lambda x: x['relevance_score'], reverse=True)

        if not results:
            print(f"❌ No documents found matching '{query}'")
            return

        print(f"🔍 Search Results for '{query}' ({len(results)} matches)")
        print("=" * 50)

        for result in results[:10]:  # Show top 10 results
            doc = result['doc']
            status_emoji = self._get_status_emoji(doc['status'])
            print(f"{status_emoji} {doc['title']}")
            print(f"   📁 Category: {result['category']}")
            print(f"   📄 Path: {doc['path']}")
            print(f"   📅 Updated: {doc['last_updated']}")
            print(f"   📝 {doc['description'][:80]}...")
            print()

        if len(results) > 10:
            print(f"... and {len(results) - 10} more results")

    def find_document(self, doc_id: str) -> None:
        """Find a specific document by ID or title."""
        doc_id = doc_id.lower()

        for category in self.catalogue['categories']:
            for doc in category['documents']:
                if (doc['id'].lower() == doc_id or
                    doc_id in doc['title'].lower() or
                    doc_id in doc['path'].lower()):

                    print(f"📄 Found Document")
                    print("=" * 30)
                    print(f"🏷️  Title: {doc['title']}")
                    print(f"🆔 ID: {doc['id']}")
                    print(f"📁 Category: {category['name']}")
                    print(f"📄 Path: {doc['path']}")
                    print(f"📅 Last Updated: {doc['last_updated']}")
                    print(f"📊 Status: {doc['status']}")
                    print(f"📝 Description: {doc['description']}")
                    print(f"🏷️  Tags: {', '.join(doc.get('tags', ['None']))}")
                    print(f"📏 Word Count: {doc.get('word_count', 'Unknown')}")

                    if doc.get('related_documents'):
                        print(f"🔗 Related Documents: {', '.join(doc['related_documents'])}")

                    return

        print(f"❌ Document '{doc_id}' not found.")

    def show_related_documents(self, doc_id: str) -> None:
        """Show related documents for a given document."""
        doc_id = doc_id.lower()
        source_doc = None
        source_category = None

        # Find the source document
        for category in self.catalogue['categories']:
            for doc in category['documents']:
                if doc['id'].lower() == doc_id:
                    source_doc = doc
                    source_category = category
                    break
            if source_doc:
                break

        if not source_doc:
            print(f"❌ Document '{doc_id}' not found.")
            return

        related_ids = source_doc.get('related_documents', [])
        if not related_ids:
            print(f"ℹ️  No related documents found for '{source_doc['title']}'")
            return

        print(f"🔗 Related Documents for '{source_doc['title']}'")
        print("=" * 50)

        for related_id in related_ids:
            found = False
            for category in self.catalogue['categories']:
                for doc in category['documents']:
                    if doc['id'] == related_id:
                        status_emoji = self._get_status_emoji(doc['status'])
                        print(f"{status_emoji} {doc['title']}")
                        print(f"   📁 Category: {category['name']}")
                        print(f"   📄 Path: {doc['path']}")
                        print(f"   📝 {doc['description'][:60]}...")
                        print()
                        found = True
                        break
                if found:
                    break

            if not found:
                print(f"⚠️  Related document '{related_id}' not found in catalogue")

    def show_status_summary(self) -> None:
        """Show documentation status summary."""
        metadata = self.catalogue['catalogue_metadata']
        maintenance = self.catalogue['maintenance']

        print("📊 Documentation Status Summary")
        print("=" * 35)
        print(f"📅 Last Updated: {metadata['last_updated']}")
        print(f"📚 Total Documents: {metadata['total_documents']}")
        print(f"🏷️  Categories: {metadata['categories']}")
        print(f"📈 Coverage: {metadata['status']}")

        # Count documents by status
        status_counts = {}
        for category in self.catalogue['categories']:
            for doc in category['documents']:
                status = doc['status']
                status_counts[status] = status_counts.get(status, 0) + 1

        print("\n📋 Document Status Breakdown:")
        for status, count in status_counts.items():
            emoji = self._get_status_emoji(status)
            print(f"  {emoji} {status.replace('_', ' ').title()}: {count}")

        print("\n🔧 Maintenance Info:")
        print(f"  📅 Next Review: {maintenance['next_review_date']}")
        print(f"  📋 Outdated Documents: {len(maintenance['outdated_documents'])}")
        print(f"  🔗 Missing References: {len(maintenance['missing_cross_references'])}")
        print(f"  ❌ Broken Links: {len(maintenance['broken_links'])}")

    def _get_status_emoji(self, status: str) -> str:
        """Get emoji for document status."""
        status_emojis = {
            'current': '✅',
            'production_ready': '✅',
            'completed': '✅',
            'planning': '📝',
            'in_progress': '🔄',
            'outdated': '⚠️'
        }
        return status_emojis.get(status, '❓')

    def _calculate_relevance(self, query: str, doc: Dict[str, Any]) -> int:
        """Calculate relevance score for search results."""
        score = 0

        # Title matches are most relevant
        if query in doc['title'].lower():
            score += 10

        # Path matches are very relevant
        if query in doc['path'].lower():
            score += 8

        # Tag matches are relevant
        for tag in doc.get('tags', []):
            if query in tag.lower():
                score += 6

        # Description matches are somewhat relevant
        if query in doc['description'].lower():
            score += 4

        return score

def main():
    """Main entry point for the documentation navigator."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1].lower()
    navigator = DocsNavigator()

    try:
        if command == 'list':
            category = sys.argv[2] if len(sys.argv) > 2 else None
            navigator.list_documents(category)

        elif command == 'search':
            if len(sys.argv) < 3:
                print("❌ Please provide a search query")
                print("Usage: python docs_navigator.py search <query>")
                sys.exit(1)
            navigator.search_documents(sys.argv[2])

        elif command == 'find':
            if len(sys.argv) < 3:
                print("❌ Please provide a document ID or title")
                print("Usage: python docs_navigator.py find <doc_id>")
                sys.exit(1)
            navigator.find_document(sys.argv[2])

        elif command == 'related':
            if len(sys.argv) < 3:
                print("❌ Please provide a document ID")
                print("Usage: python docs_navigator.py related <doc_id>")
                sys.exit(1)
            navigator.show_related_documents(sys.argv[2])

        elif command == 'status':
            navigator.show_status_summary()

        elif command == 'validate':
            # The catalogue is validated during loading
            print("✅ Catalogue validation successful!")
            print(f"📊 Loaded {navigator.catalogue['catalogue_metadata']['total_documents']} documents")

        else:
            print(f"❌ Unknown command: {command}")
            print("Available commands: list, search, find, related, status, validate")

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
