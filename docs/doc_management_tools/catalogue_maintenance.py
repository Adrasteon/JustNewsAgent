#!/usr/bin/env python3
"""
JustNewsAgent Documentation Catalogue Maintenance Tools
Industry-standard utilities for catalogue management and analysis

This module provides comprehensive tools for:
- Catalogue health monitoring and maintenance
- Advanced search and filtering capabilities
- Cross-reference analysis and validation
- Performance metrics and reporting
- Automated maintenance workflows

Usage:
    python docs/catalogue_maintenance.py --health-check
    python docs/catalogue_maintenance.py --search "gpu" --category "agent_documentation"
    python docs/catalogue_maintenance.py --cross-references
    python docs/catalogue_maintenance.py --performance-report
"""

import os
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict, Counter
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class CatalogueStats:
    """Statistics for catalogue analysis"""
    total_documents: int
    total_categories: int
    oldest_document: str
    newest_document: str
    avg_word_count: float
    largest_document: str
    smallest_document: str
    most_common_tags: List[tuple]
    documents_per_category: Dict[str, int]
    status_distribution: Dict[str, int]

class CatalogueMaintenanceTools:
    """Comprehensive maintenance and analysis tools for the documentation catalogue"""

    def __init__(self, workspace_root: str):
        self.workspace_root = Path(workspace_root)
        self.catalogue_path = self.workspace_root / "docs" / "docs_catalogue_v2.json"
        self.catalogue = self._load_catalogue()

    def _load_catalogue(self) -> Dict[str, Any]:
        """Load the catalogue from file"""
        if not self.catalogue_path.exists():
            raise FileNotFoundError(f"Catalogue not found: {self.catalogue_path}")

        with open(self.catalogue_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of the catalogue"""
        logger.info("Performing comprehensive catalogue health check...")

        issues = {
            "broken_paths": [],
            "missing_metadata": [],
            "duplicate_ids": [],
            "orphaned_documents": [],
            "stale_documents": [],
            "quality_issues": []
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

        # Check for stale documents (older than 6 months)
        six_months_ago = datetime.now() - timedelta(days=180)
        for category in self.catalogue["categories"]:
            for doc in category["documents"]:
                last_updated = doc.get("last_modified", "")
                if last_updated:
                    try:
                        doc_date = datetime.strptime(last_updated, "%Y-%m-%d")
                        if doc_date < six_months_ago:
                            issues["stale_documents"].append(doc["path"])
                    except ValueError:
                        pass

        # Quality checks
        for category in self.catalogue["categories"]:
            for doc in category["documents"]:
                # Check for very short descriptions
                if len(doc.get("description", "")) < 50:
                    issues["quality_issues"].append(f"{doc['path']}: description too short")

                # Check for documents with no tags
                if not doc.get("tags"):
                    issues["quality_issues"].append(f"{doc['path']}: no tags assigned")

        total_issues = sum(len(v) for v in issues.values())

        result = {
            "status": "healthy" if total_issues == 0 else "issues_found",
            "total_issues": total_issues,
            "issues": issues,
            "catalogue_info": {
                "version": self.catalogue["catalogue_metadata"]["version"],
                "last_updated": self.catalogue["catalogue_metadata"]["last_updated"],
                "total_documents": self.catalogue["catalogue_metadata"]["total_documents"],
                "categories": self.catalogue["catalogue_metadata"]["categories"]
            }
        }

        logger.info(f"Health check complete: {total_issues} issues found")
        return result

    def advanced_search(self, query: str, category: Optional[str] = None,
                       tags: Optional[List[str]] = None, status: Optional[str] = None) -> List[Dict]:
        """Advanced search with multiple filters"""
        logger.info(f"Performing advanced search: query='{query}', category='{category}', tags={tags}, status='{status}'")

        results = []

        for cat in self.catalogue["categories"]:
            if category and cat["id"] != category:
                continue

            for doc in cat["documents"]:
                # Text search in title, description, and content
                searchable_text = (
                    doc.get("title", "").lower() + " " +
                    doc.get("description", "").lower() + " " +
                    doc.get("search_content", "").lower()
                )

                # Check if query matches
                query_match = query.lower() in searchable_text

                # Tag filtering
                tag_match = True
                if tags:
                    doc_tags = set(doc.get("tags", []))
                    tag_match = any(tag.lower() in [t.lower() for t in doc_tags] for tag in tags)

                # Status filtering
                status_match = True
                if status:
                    status_match = doc.get("status", "").lower() == status.lower()

                if query_match and tag_match and status_match:
                    results.append({
                        "document": doc,
                        "category": cat["name"],
                        "relevance_score": self._calculate_relevance(query, doc)
                    })

        # Sort by relevance
        results.sort(key=lambda x: x["relevance_score"], reverse=True)

        logger.info(f"Search complete: {len(results)} results found")
        return results

    def _calculate_relevance(self, query: str, doc: Dict) -> float:
        """Calculate relevance score for search results"""
        score = 0.0
        query_lower = query.lower()

        # Title matches are most important
        if query_lower in doc.get("title", "").lower():
            score += 10.0

        # Description matches are important
        if query_lower in doc.get("description", "").lower():
            score += 5.0

        # Tag matches
        for tag in doc.get("tags", []):
            if query_lower in tag.lower():
                score += 3.0

        # Content matches
        content = doc.get("search_content", "").lower()
        occurrences = content.count(query_lower)
        score += min(occurrences * 0.5, 5.0)  # Cap at 5 points

        return score

    def analyze_cross_references(self) -> Dict[str, Any]:
        """Analyze cross-reference patterns in the catalogue"""
        logger.info("Analyzing cross-reference patterns...")

        cross_refs = defaultdict(list)
        orphaned_docs = []
        broken_refs = []

        # Collect all document IDs
        all_doc_ids = set()
        for category in self.catalogue["categories"]:
            for doc in category["documents"]:
                all_doc_ids.add(doc["id"])

        # Analyze references
        for category in self.catalogue["categories"]:
            for doc in category["documents"]:
                doc_refs = doc.get("related_documents", [])

                # Check for broken references
                for ref in doc_refs:
                    if ref not in all_doc_ids:
                        broken_refs.append({
                            "from": doc["path"],
                            "to": ref,
                            "broken": True
                        })

                # Build cross-reference graph
                for ref in doc_refs:
                    cross_refs[ref].append(doc["id"])

                # Check for orphaned documents (no references to them)
                if not cross_refs[doc["id"]]:
                    orphaned_docs.append(doc["path"])

        result = {
            "total_documents": len(all_doc_ids),
            "documents_with_references": len([k for k, v in cross_refs.items() if v]),
            "orphaned_documents": len(orphaned_docs),
            "broken_references": len(broken_refs),
            "most_referenced": sorted(cross_refs.items(), key=lambda x: len(x[1]), reverse=True)[:10],
            "details": {
                "orphaned": orphaned_docs[:20],  # Show first 20
                "broken_refs": broken_refs[:20]  # Show first 20
            }
        }

        logger.info(f"Cross-reference analysis complete: {len(broken_refs)} broken references found")
        return result

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance and statistics report"""
        logger.info("Generating performance report...")

        stats = CatalogueStats(
            total_documents=self.catalogue["catalogue_metadata"]["total_documents"],
            total_categories=self.catalogue["catalogue_metadata"]["categories"],
            oldest_document="",
            newest_document="",
            avg_word_count=0.0,
            largest_document="",
            smallest_document="",
            most_common_tags=[],
            documents_per_category={},
            status_distribution={}
        )

        # Calculate statistics
        all_word_counts = []
        all_tags = []
        oldest_date = datetime.max
        newest_date = datetime.min
        max_words = 0
        min_words = float('inf')

        for category in self.catalogue["categories"]:
            cat_docs = len(category["documents"])
            stats.documents_per_category[category["name"]] = cat_docs

            for doc in category["documents"]:
                # Word count stats
                word_count = doc.get("word_count", 0)
                all_word_counts.append(word_count)

                if word_count > max_words:
                    max_words = word_count
                    stats.largest_document = doc["path"]

                if word_count < min_words and word_count > 0:
                    min_words = word_count
                    stats.smallest_document = doc["path"]

                # Date stats
                last_modified = doc.get("last_modified", "")
                if last_modified:
                    try:
                        doc_date = datetime.strptime(last_modified, "%Y-%m-%d")
                        if doc_date < oldest_date:
                            oldest_date = doc_date
                            stats.oldest_document = doc["path"]
                        if doc_date > newest_date:
                            newest_date = doc_date
                            stats.newest_document = doc["path"]
                    except ValueError:
                        pass

                # Tag stats
                all_tags.extend(doc.get("tags", []))

                # Status stats
                status = doc.get("status", "unknown")
                if status not in stats.status_distribution:
                    stats.status_distribution[status] = 0
                stats.status_distribution[status] += 1

        # Calculate averages
        if all_word_counts:
            stats.avg_word_count = sum(all_word_counts) / len(all_word_counts)

        # Most common tags
        tag_counts = Counter(all_tags)
        stats.most_common_tags = tag_counts.most_common(10)

        # Handle edge cases
        if oldest_date == datetime.max:
            stats.oldest_document = "N/A"
        if newest_date == datetime.min:
            stats.newest_document = "N/A"
        if min_words == float('inf'):
            stats.smallest_document = "N/A"

        result = {
            "catalogue_stats": stats.__dict__,
            "tag_analysis": {
                "total_unique_tags": len(tag_counts),
                "most_common_tags": stats.most_common_tags,
                "tags_per_document_avg": len(all_tags) / max(1, stats.total_documents)
            },
            "content_analysis": {
                "avg_word_count": round(stats.avg_word_count, 1),
                "total_words": sum(all_word_counts),
                "word_count_distribution": {
                    "under_500": len([w for w in all_word_counts if w < 500]),
                    "500_2000": len([w for w in all_word_counts if 500 <= w <= 2000]),
                    "2000_5000": len([w for w in all_word_counts if 2000 < w <= 5000]),
                    "over_5000": len([w for w in all_word_counts if w > 5000])
                }
            },
            "maintenance_recommendations": self._generate_maintenance_recommendations(stats)
        }

        logger.info("Performance report generated successfully")
        return result

    def _generate_maintenance_recommendations(self, stats: CatalogueStats) -> List[str]:
        """Generate maintenance recommendations based on statistics"""
        recommendations = []

        # Check for unbalanced categories
        avg_docs_per_cat = stats.total_documents / stats.total_categories
        for cat_name, doc_count in stats.documents_per_category.items():
            if doc_count > avg_docs_per_cat * 2:
                recommendations.append(f"Category '{cat_name}' has {doc_count} documents (above average)")

        # Check for old documents
        if stats.oldest_document != "N/A":
            recommendations.append(f"Consider reviewing oldest document: {stats.oldest_document}")

        # Check for status distribution
        draft_count = stats.status_distribution.get("draft", 0)
        if draft_count > stats.total_documents * 0.1:
            recommendations.append(f"High number of draft documents: {draft_count}")

        # Check for content quality
        if stats.avg_word_count < 500:
            recommendations.append("Average document length is low - consider expanding content")

        return recommendations if recommendations else ["Catalogue is well-maintained"]

def main():
    parser = argparse.ArgumentParser(description="JustNewsAgent Documentation Catalogue Maintenance Tools")
    parser.add_argument("--health-check", action="store_true", help="Perform comprehensive health check")
    parser.add_argument("--search", help="Search query")
    parser.add_argument("--category", help="Filter by category")
    parser.add_argument("--tags", nargs="+", help="Filter by tags")
    parser.add_argument("--status", help="Filter by status")
    parser.add_argument("--cross-references", action="store_true", help="Analyze cross-reference patterns")
    parser.add_argument("--performance-report", action="store_true", help="Generate performance report")
    parser.add_argument("--workspace", default="/home/adra/justnewsagent/JustNewsAgent",
                       help="Workspace root directory")

    args = parser.parse_args()

    # Initialize tools
    tools = CatalogueMaintenanceTools(args.workspace)

    if args.health_check:
        # Health check
        result = tools.health_check()
        print("=== CATALOGUE HEALTH CHECK ===")
        print(f"Status: {result['status']}")
        print(f"Total Issues: {result['total_issues']}")
        print(f"Total Documents: {result['catalogue_info']['total_documents']}")
        print(f"Categories: {result['catalogue_info']['categories']}")
        print()

        for issue_type, items in result["issues"].items():
            if items:
                print(f"{issue_type.upper()}: {len(items)} issues")
                for item in items[:5]:  # Show first 5
                    print(f"  - {item}")
                if len(items) > 5:
                    print(f"  ... and {len(items) - 5} more")
                print()

    elif args.search:
        # Advanced search
        results = tools.advanced_search(
            query=args.search,
            category=args.category,
            tags=args.tags,
            status=args.status
        )

        print(f"=== SEARCH RESULTS: '{args.search}' ===")
        print(f"Found {len(results)} matching documents")
        print()

        for i, result in enumerate(results[:20], 1):  # Show first 20
            doc = result["document"]
            print(f"{i}. [{doc['title']}]({doc['path']})")
            print(f"   Category: {result['category']}")
            print(f"   Status: {doc['status']}")
            print(f"   Tags: {', '.join(doc['tags'][:3])}")
            print(f"   Relevance: {result['relevance_score']:.1f}")
            print(f"   Description: {doc['description'][:100]}...")
            print()

    elif args.cross_references:
        # Cross-reference analysis
        result = tools.analyze_cross_references()

        print("=== CROSS-REFERENCE ANALYSIS ===")
        print(f"Total Documents: {result['total_documents']}")
        print(f"Documents with References: {result['documents_with_references']}")
        print(f"Orphaned Documents: {result['orphaned_documents']}")
        print(f"Broken References: {result['broken_references']}")
        print()

        if result["most_referenced"]:
            print("Most Referenced Documents:")
            for doc_id, refs in result["most_referenced"][:10]:
                print(f"  {doc_id}: referenced by {len(refs)} documents")

        if result["details"]["orphaned"]:
            print(f"\nOrphaned Documents ({len(result['details']['orphaned'])}):")
            for doc in result["details"]["orphaned"][:10]:
                print(f"  - {doc}")

    elif args.performance_report:
        # Performance report
        result = tools.generate_performance_report()

        print("=== PERFORMANCE REPORT ===")
        stats = result["catalogue_stats"]
        print(f"Total Documents: {stats['total_documents']}")
        print(f"Total Categories: {stats['total_categories']}")
        print(f"Average Word Count: {stats['avg_word_count']:.0f}")
        print(f"Oldest Document: {stats['oldest_document']}")
        print(f"Newest Document: {stats['newest_document']}")
        print(f"Largest Document: {stats['largest_document']}")
        print(f"Smallest Document: {stats['smallest_document']}")
        print()

        print("Documents per Category:")
        for cat, count in stats["documents_per_category"].items():
            print(f"  {cat}: {count}")
        print()

        print("Status Distribution:")
        for status, count in stats["status_distribution"].items():
            print(f"  {status}: {count}")
        print()

        print("Most Common Tags:")
        for tag, count in result["tag_analysis"]["most_common_tags"][:10]:
            print(f"  {tag}: {count}")
        print()

        print("Content Analysis:")
        wc_dist = result["content_analysis"]["word_count_distribution"]
        print(f"  Under 500 words: {wc_dist['under_500']}")
        print(f"  500-2000 words: {wc_dist['500_2000']}")
        print(f"  2000-5000 words: {wc_dist['2000_5000']}")
        print(f"  Over 5000 words: {wc_dist['over_5000']}")
        print()

        print("Maintenance Recommendations:")
        for rec in result["maintenance_recommendations"]:
            print(f"  • {rec}")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
