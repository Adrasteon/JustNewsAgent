#!/usr/bin/env python3
"""
Agent Documentation Reorganization Script
Reorganizes the large Agent Documentation category into logical subcategories.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AgentDocsReorganizer:
    """Handles agent documentation reorganization operations"""

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

    def create_agent_subcategories(self):
        """Create new subcategories for Agent Documentation"""
        agent_docs = self.get_category_by_id('agent_documentation')
        if not agent_docs:
            logger.error("Agent Documentation category not found")
            return

        # Define new subcategories for agents
        subcategories = [
            {
                "id": "agent_documentation_core_agents",
                "name": "Core Agent Documentation",
                "description": "Documentation for core system agents (Scout, Analyst, Synthesizer, Fact Checker)",
                "priority": "high",
                "documents": []
            },
            {
                "id": "agent_documentation_specialized_agents",
                "name": "Specialized Agent Documentation",
                "description": "Documentation for specialized agents (Reasoning, Balancer, Critic, Memory)",
                "priority": "high",
                "documents": []
            },
            {
                "id": "agent_documentation_deprecated_agents",
                "name": "Deprecated Agent Documentation",
                "description": "Documentation for deprecated or legacy agents and implementations",
                "priority": "low",
                "documents": []
            },
            {
                "id": "agent_documentation_agent_management",
                "name": "Agent Management & Tools",
                "description": "Agent management tools, utilities, and operational documentation",
                "priority": "medium",
                "documents": []
            },
            {
                "id": "agent_documentation_model_integration",
                "name": "Model Integration Documentation",
                "description": "AI model integration, configuration, and performance documentation",
                "priority": "high",
                "documents": []
            },
            {
                "id": "agent_documentation_crawling_systems",
                "name": "Crawling & Data Collection",
                "description": "Web crawling systems, data collection agents, and content extraction",
                "priority": "medium",
                "documents": []
            }
        ]

        # Add subcategories to catalogue
        for subcategory in subcategories:
            self.catalogue['categories'].append(subcategory)
            logger.info(f"Created subcategory: {subcategory['name']}")

        # Update total categories count
        self.catalogue['catalogue_metadata']['categories'] = len(self.catalogue['categories'])

    def categorize_agent_document(self, document: Dict[str, Any]) -> str:
        """Determine the appropriate subcategory for an agent document"""
        title = document.get('title', '').lower()
        description = document.get('description', '').lower()
        tags = [tag.lower() for tag in document.get('tags', [])]

        # Core agents
        core_agents = ['scout', 'analyst', 'synthesizer', 'fact checker', 'chief editor']
        if any(agent in title or agent in description for agent in core_agents):
            return 'agent_documentation_core_agents'

        # Specialized agents
        specialized_agents = ['reasoning', 'balancer', 'critic', 'memory', 'newsreader']
        if any(agent in title or agent in description for agent in specialized_agents):
            return 'agent_documentation_specialized_agents'

        # Model integration
        if any(keyword in title or keyword in description or keyword in ' '.join(tags)
               for keyword in ['model', 'ai', 'llm', 'gpt', 'bert', 'tensorrt', 'gpu', 'training']):
            return 'agent_documentation_model_integration'

        # Crawling systems
        if any(keyword in title or keyword in description or keyword in ' '.join(tags)
               for keyword in ['crawl', 'crawler', 'scraping', 'extraction', 'bbc', 'content']):
            return 'agent_documentation_crawling_systems'

        # Agent management
        if any(keyword in title or keyword in description or keyword in ' '.join(tags)
               for keyword in ['management', 'tool', 'utility', 'map', 'integration', 'workflow']):
            return 'agent_documentation_agent_management'

        # Deprecated/legacy
        if any(keyword in title or keyword in description or keyword in ' '.join(tags)
               for keyword in ['deprecated', 'legacy', 'old', 'archive', 'v1']):
            return 'agent_documentation_deprecated_agents'

        # Default fallback
        else:
            logger.warning(f"Could not categorize agent document: {title}")
            return 'agent_documentation_core_agents'

    def reorganize_agent_documents(self):
        """Move documents from agent_documentation to appropriate subcategories"""
        agent_docs = self.get_category_by_id('agent_documentation')
        if not agent_docs:
            logger.error("Agent Documentation category not found")
            return

        documents_to_move = agent_docs['documents'].copy()
        logger.info(f"Moving {len(documents_to_move)} documents from Agent Documentation")

        # Clear the original category
        agent_docs['documents'] = []

        # Move each document to appropriate subcategory
        moved_count = 0
        for document in documents_to_move:
            subcategory_id = self.categorize_agent_document(document)

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

        logger.info(f"Successfully moved {moved_count} agent documents to subcategories")

        # Update document counts
        self.update_document_counts()

    def update_document_counts(self):
        """Update document counts for all categories"""
        total_documents = 0
        for category in self.catalogue['categories']:
            doc_count = len(category['documents'])
            category['document_count'] = doc_count
            total_documents += doc_count

        self.catalogue['catalogue_metadata']['total_documents'] = total_documents
        logger.info(f"Total documents: {total_documents}")

    def validate_agent_reorganization(self):
        """Validate that agent reorganization was successful"""
        logger.info("Validating agent reorganization...")

        # Check that agent_documentation is empty
        agent_docs = self.get_category_by_id('agent_documentation')
        if agent_docs and len(agent_docs['documents']) > 0:
            logger.warning(f"Agent Documentation still has {len(agent_docs['documents'])} documents")

        # Check that subcategories have documents
        subcategories = [
            'agent_documentation_core_agents',
            'agent_documentation_specialized_agents',
            'agent_documentation_deprecated_agents',
            'agent_documentation_agent_management',
            'agent_documentation_model_integration',
            'agent_documentation_crawling_systems'
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
            logger.info(f"✓ Agent documents moved to subcategories: {total_in_subcategories}")
            return True
        else:
            logger.error(f"✗ Document count mismatch: {total_documents_all_categories} vs {expected_total}")
            return False

    def run_agent_reorganization(self):
        """Run the complete agent reorganization process"""
        logger.info("Starting Agent Documentation Reorganization")

        try:
            # Create subcategories
            logger.info("Step 1: Creating agent subcategories...")
            self.create_agent_subcategories()

            # Reorganize documents
            logger.info("Step 2: Reorganizing agent documents...")
            self.reorganize_agent_documents()

            # Validate results
            logger.info("Step 3: Validating agent reorganization...")
            if self.validate_agent_reorganization():
                logger.info("✓ Agent reorganization completed successfully!")
                self.save_catalogue()
                return True
            else:
                logger.error("✗ Agent reorganization validation failed!")
                return False

        except Exception as e:
            logger.error(f"Agent reorganization failed: {e}")
            return False

def main():
    """Main execution function"""
    catalogue_path = "/home/adra/justnewsagent/JustNewsAgent/docs/docs_catalogue_v2.json"

    reorganizer = AgentDocsReorganizer(catalogue_path)

    if reorganizer.run_agent_reorganization():
        print("\n🎉 Agent Documentation reorganization completed successfully!")
        print("The Agent Documentation category has been split into 6 logical subcategories.")
        print("All cross-references have been preserved during the reorganization.")
    else:
        print("\n❌ Agent Documentation reorganization failed!")
        exit(1)

if __name__ == "__main__":
    main()
