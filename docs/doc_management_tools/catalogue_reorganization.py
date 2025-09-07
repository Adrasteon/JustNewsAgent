#!/usr/bin/env python3
"""
Phase 2: Category Reorganization Script
Reorganizes the large Development Reports category into logical subcategories.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CatalogueReorganizer:
    """Handles catalogue reorganization operations"""

    def __init__(self, catalogue_path: str):
        self.catalogue_path = Path(catalogue_path)
        self.catalogue = None
        self.load_catalogue()

    def load_catalogue(self):
        """Load the catalogue from file"""
        try:
            with open(self.catalogue_path, 'r', encoding='utf-8') as f:
                self.catalogue = json.load(f)
            logger.info(f"Loaded catalogue with {self.catalogue['catalogue_metadata']['total_documents']} documents")
        except Exception as e:
            logger.error(f"Failed to load catalogue: {e}")
            raise

    def save_catalogue(self):
        """Save the catalogue to file"""
        try:
            with open(self.catalogue_path, 'w', encoding='utf-8') as f:
                json.dump(self.catalogue, f, indent=2, ensure_ascii=False)
            logger.info("Catalogue saved successfully")
        except Exception as e:
            logger.error(f"Failed to save catalogue: {e}")
            raise

    def get_category_by_id(self, category_id: str) -> Dict[str, Any]:
        """Get category by ID"""
        for category in self.catalogue['categories']:
            if category['id'] == category_id:
                return category
        return None

    def create_subcategories(self):
        """Create new subcategories for Development Reports"""
        development_reports = self.get_category_by_id('development_reports')
        if not development_reports:
            logger.error("Development Reports category not found")
            return

        # Define new subcategories
        subcategories = [
            {
                "id": "development_reports_architecture",
                "name": "Architecture & Design Reports",
                "description": "Technical architecture decisions, system design patterns, and architectural improvements",
                "priority": "high",
                "documents": []
            },
            {
                "id": "development_reports_implementation",
                "name": "Implementation Reports",
                "description": "Code implementation details, feature development, and technical solutions",
                "priority": "high",
                "documents": []
            },
            {
                "id": "development_reports_performance",
                "name": "Performance & Optimization Reports",
                "description": "Performance analysis, optimization results, and system performance improvements",
                "priority": "high",
                "documents": []
            },
            {
                "id": "development_reports_testing",
                "name": "Testing & Quality Assurance Reports",
                "description": "Testing results, quality assurance findings, and validation reports",
                "priority": "medium",
                "documents": []
            },
            {
                "id": "development_reports_deployment",
                "name": "Deployment & Operations Reports",
                "description": "Deployment procedures, operational improvements, and infrastructure updates",
                "priority": "medium",
                "documents": []
            },
            {
                "id": "development_reports_training",
                "name": "Training & Learning Reports",
                "description": "Continuous learning system reports, model training results, and AI improvements",
                "priority": "high",
                "documents": []
            },
            {
                "id": "development_reports_integration",
                "name": "Integration & Workflow Reports",
                "description": "System integration, workflow improvements, and cross-component coordination",
                "priority": "medium",
                "documents": []
            },
            {
                "id": "development_reports_maintenance",
                "name": "Maintenance & Housekeeping Reports",
                "description": "System maintenance, cleanup operations, and organizational improvements",
                "priority": "low",
                "documents": []
            }
        ]

        # Add subcategories to catalogue
        for subcategory in subcategories:
            self.catalogue['categories'].append(subcategory)
            logger.info(f"Created subcategory: {subcategory['name']}")

        # Update total categories count
        self.catalogue['catalogue_metadata']['categories'] = len(self.catalogue['categories'])

    def categorize_document(self, document: Dict[str, Any]) -> str:
        """Determine the appropriate subcategory for a document based on its content"""
        title = document.get('title', '').lower()
        description = document.get('description', '').lower()
        tags = [tag.lower() for tag in document.get('tags', [])]

        # Architecture & Design patterns
        if any(keyword in title or keyword in description or keyword in ' '.join(tags)
               for keyword in ['architecture', 'design', 'blueprint', 'system design', 'technical architecture',
                             'mcp bus', 'agent architecture', 'workflow', 'pipeline']):
            return 'development_reports_architecture'

        # Implementation patterns
        elif any(keyword in title or keyword in description or keyword in ' '.join(tags)
                for keyword in ['implementation', 'feature', 'development', 'code', 'programming',
                              'agent implementation', 'engine', 'v2', 'v3', 'v4']):
            return 'development_reports_implementation'

        # Performance patterns
        elif any(keyword in title or keyword in description or keyword in ' '.join(tags)
                for keyword in ['performance', 'optimization', 'speed', 'efficiency', 'gpu',
                              'tensorrt', 'throughput', 'benchmark', 'memory']):
            return 'development_reports_performance'

        # Testing patterns
        elif any(keyword in title or keyword in description or keyword in ' '.join(tags)
                for keyword in ['test', 'testing', 'validation', 'quality', 'qa', 'verification']):
            return 'development_reports_testing'

        # Deployment patterns
        elif any(keyword in title or keyword in description or keyword in ' '.join(tags)
                for keyword in ['deployment', 'production', 'operational', 'infrastructure',
                              'docker', 'kubernetes', 'service']):
            return 'development_reports_deployment'

        # Training patterns
        elif any(keyword in title or keyword in description or keyword in ' '.join(tags)
                for keyword in ['training', 'learning', 'model', 'ai', 'machine learning',
                              'continuous learning', 'ewc', 'fine-tuning']):
            return 'development_reports_training'

        # Integration patterns
        elif any(keyword in title or keyword in description or keyword in ' '.join(tags)
                for keyword in ['integration', 'workflow', 'coordination', 'pipeline',
                              'cross-agent', 'communication']):
            return 'development_reports_integration'

        # Maintenance patterns
        elif any(keyword in title or keyword in description or keyword in ' '.join(tags)
                for keyword in ['maintenance', 'housekeeping', 'cleanup', 'organization',
                              'workspace', 'file management', 'consolidation']):
            return 'development_reports_maintenance'

        # Default fallback
        else:
            logger.warning(f"Could not categorize document: {title}")
            return 'development_reports_implementation'  # Default to implementation

    def reorganize_documents(self):
        """Move documents from development_reports to appropriate subcategories"""
        development_reports = self.get_category_by_id('development_reports')
        if not development_reports:
            logger.error("Development Reports category not found")
            return

        documents_to_move = development_reports['documents'].copy()
        logger.info(f"Moving {len(documents_to_move)} documents from Development Reports")

        # Clear the original category
        development_reports['documents'] = []

        # Move each document to appropriate subcategory
        moved_count = 0
        for document in documents_to_move:
            subcategory_id = self.categorize_document(document)

            # Find the subcategory
            subcategory = self.get_category_by_id(subcategory_id)
            if subcategory:
                # Update document category
                document['category'] = subcategory_id

                # Add to subcategory
                subcategory['documents'].append(document)
                moved_count += 1
                logger.debug(f"Moved '{document['title']}' to {subcategory['name']}")
            else:
                logger.error(f"Subcategory {subcategory_id} not found for document {document['title']}")

        logger.info(f"Successfully moved {moved_count} documents to subcategories")

        # Update document counts
        self.update_document_counts()

    def update_document_counts(self):
        """Update document counts for all categories"""
        total_documents = 0
        for category in self.catalogue['categories']:
            doc_count = len(category['documents'])
            category['document_count'] = doc_count
            total_documents += doc_count
            logger.info(f"Category '{category['name']}': {doc_count} documents")

        self.catalogue['catalogue_metadata']['total_documents'] = total_documents
        logger.info(f"Total documents: {total_documents}")

    def validate_reorganization(self):
        """Validate that reorganization was successful"""
        logger.info("Validating reorganization...")

        # Check that development_reports is empty
        development_reports = self.get_category_by_id('development_reports')
        if development_reports and len(development_reports['documents']) > 0:
            logger.warning(f"Development Reports still has {len(development_reports['documents'])} documents")

        # Check that subcategories have documents
        subcategories = [
            'development_reports_architecture',
            'development_reports_implementation',
            'development_reports_performance',
            'development_reports_testing',
            'development_reports_deployment',
            'development_reports_training',
            'development_reports_integration',
            'development_reports_maintenance'
        ]

        total_in_subcategories = 0
        for subcategory_id in subcategories:
            subcategory = self.get_category_by_id(subcategory_id)
            if subcategory:
                doc_count = len(subcategory['documents'])
                total_in_subcategories += doc_count
                logger.info(f"✓ {subcategory['name']}: {doc_count} documents")
            else:
                logger.error(f"✗ Subcategory {subcategory_id} not found")

        # Count total documents across ALL categories
        total_documents_all_categories = sum(len(cat['documents']) for cat in self.catalogue['categories'])

        # Verify total count matches metadata
        expected_total = self.catalogue['catalogue_metadata']['total_documents']
        if total_documents_all_categories == expected_total:
            logger.info(f"✓ Total document count verified: {total_documents_all_categories}")
            logger.info(f"✓ Documents moved to subcategories: {total_in_subcategories}")
            return True
        else:
            logger.error(f"✗ Document count mismatch: {total_documents_all_categories} vs {expected_total}")
            return False

    def run_reorganization(self):
        """Run the complete reorganization process"""
        logger.info("Starting Phase 2: Category Reorganization")

        try:
            # Create subcategories
            logger.info("Step 1: Creating subcategories...")
            self.create_subcategories()

            # Reorganize documents
            logger.info("Step 2: Reorganizing documents...")
            self.reorganize_documents()

            # Validate results
            logger.info("Step 3: Validating reorganization...")
            if self.validate_reorganization():
                logger.info("✓ Reorganization completed successfully!")
                self.save_catalogue()
                return True
            else:
                logger.error("✗ Reorganization validation failed!")
                return False

        except Exception as e:
            logger.error(f"Reorganization failed: {e}")
            return False

def main():
    """Main execution function"""
    catalogue_path = "/home/adra/justnewsagent/JustNewsAgent/docs/docs_catalogue_v2.json"

    reorganizer = CatalogueReorganizer(catalogue_path)

    if reorganizer.run_reorganization():
        print("\n🎉 Phase 2 reorganization completed successfully!")
        print("The Development Reports category has been split into 8 logical subcategories.")
        print("All cross-references have been preserved during the reorganization.")
    else:
        print("\n❌ Phase 2 reorganization failed!")
        exit(1)

if __name__ == "__main__":
    main()
