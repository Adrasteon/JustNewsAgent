from common.observability import get_logger
#!/usr/bin/env python3
"""
Database migration script for JustNewsAgent
Applies performance improvements and indexes to the database
"""

import sys

from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.common.database import execute_query


logger = get_logger(__name__)

def apply_migration(migration_file: str):
    """Apply a database migration from SQL file"""
    try:
        with open(migration_file, 'r') as f:
            sql_content = f.read()

        # Split into individual statements
        statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]

        for statement in statements:
            if statement:
                logger.info(f"Executing: {statement[:50]}...")
                execute_query(statement, fetch=False)

        logger.info(f"✅ Migration {migration_file} applied successfully")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to apply migration {migration_file}: {e}")
        return False

def main():
    """Main migration function"""
    logger.info("🚀 Starting database performance migration")

    # Path to the migration file
    migration_file = Path(__file__).parent.parent / "agents" / "memory" / "db_migrations" / "004_add_performance_indexes.sql"

    if not migration_file.exists():
        logger.error(f"Migration file not found: {migration_file}")
        return False

    # Apply the migration
    success = apply_migration(str(migration_file))

    if success:
        logger.info("🎉 Database performance migration completed successfully!")
        logger.info("📊 Performance improvements:")
        logger.info("   • Added GIN indexes for metadata and embedding searches")
        logger.info("   • Added text search index on article content")
        logger.info("   • Added indexes for training examples table")
        logger.info("   • Added composite indexes for common query patterns")
        logger.info("   • Added partial indexes for articles with embeddings")
    else:
        logger.error("💥 Database performance migration failed!")
        return False

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)