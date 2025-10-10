#!/usr/bin/env python3
"""
Interactive database operations script for JustNewsAgent
Provides convenient database operations using the connection pooling utilities
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set database environment variables
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_DB", "justnews")
os.environ.setdefault("POSTGRES_USER", "justnews_user")
os.environ.setdefault("POSTGRES_PASSWORD", "password123")

from agents.common.database import (
    execute_query,
    execute_query_single,
    get_pool_stats,
    initialize_connection_pool,
)

def show_menu():
    """Display the main menu"""
    print("\n🔧 JustNewsAgent Database Operations")
    print("=" * 40)
    print("1. Show database statistics")
    print("2. List recent articles")
    print("3. Search articles by content")
    print("4. Show training examples")
    print("5. Show database schema")
    print("6. Test vector search")
    print("7. Show connection pool status")
    print("8. Exit")
    print()

def show_database_stats():
    """Show database statistics"""
    print("\n📊 Database Statistics:")
    print("-" * 25)

    try:
        # Article count
        result = execute_query_single("SELECT COUNT(*) as count FROM articles")
        print(f"📄 Total articles: {result['count']}")

        # Articles with embeddings
        result = execute_query_single("SELECT COUNT(*) as count FROM articles WHERE embedding IS NOT NULL")
        print(f"🧠 Articles with embeddings: {result['count']}")

        # Training examples count
        result = execute_query_single("SELECT COUNT(*) as count FROM training_examples")
        print(f"🎓 Training examples: {result['count']}")

        # Sources count
        result = execute_query_single("SELECT COUNT(*) as count FROM sources")
        print(f"📰 News sources: {result['count']}")

        # Crawled URLs count
        result = execute_query_single("SELECT COUNT(*) as count FROM crawled_urls")
        print(f"🔗 Crawled URLs: {result['count']}")

    except Exception as e:
        print(f"❌ Error getting statistics: {e}")


def list_recent_articles(limit=5):
    """List recent articles"""
    print(f"\n📰 Recent Articles (last {limit}):")
    print("-" * 30)

    try:
        articles = execute_query("""
            SELECT id, LEFT(content, 100) as content_preview,
                   created_at, metadata->>'source' as source
            FROM articles
            ORDER BY created_at DESC
            LIMIT %s
        """, (limit,))

        if articles:
            for article in articles:
                print(f"ID: {article['id']}")
                print(f"Source: {article['source'] or 'Unknown'}")
                print(f"Date: {article['created_at']}")
                print(f"Content: {article['content_preview']}...")
                print("-" * 50)
        else:
            print("No articles found")

    except Exception as e:
        print(f"❌ Error listing articles: {e}")


def search_articles(query, limit=5):
    """Search articles by content"""
    print(f"\n🔍 Search Results for '{query}' (limit {limit}):")
    print("-" * 40)

    try:
        articles = execute_query("""
            SELECT id, LEFT(content, 150) as content_preview,
                   ts_rank_cd(to_tsvector('english', content), plainto_tsquery('english', %s)) as rank,
                   metadata->>'source' as source
            FROM articles
            WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
            ORDER BY rank DESC
            LIMIT %s
        """, (query, query, limit))

        if articles:
            for article in articles:
                print(f"ID: {article['id']} | Rank: {article['rank']:.3f}")
                print(f"Source: {article['source'] or 'Unknown'}")
                print(f"Content: {article['content_preview']}...")
                print("-" * 50)
        else:
            print("No matching articles found")

    except Exception as e:
        print(f"❌ Error searching articles: {e}")


def show_training_examples(limit=5):
    """Show training examples"""
    print(f"\n🎓 Training Examples (last {limit}):")
    print("-" * 30)

    try:
        examples = execute_query("""
            SELECT id, task, created_at,
                   LEFT(input::text, 50) as input_preview,
                   LEFT(output::text, 50) as output_preview
            FROM training_examples
            ORDER BY created_at DESC
            LIMIT %s
        """, (limit,))

        if examples:
            for example in examples:
                print(f"ID: {example['id']} | Task: {example['task']}")
                print(f"Date: {example['created_at']}")
                print(f"Input: {example['input_preview']}...")
                print(f"Output: {example['output_preview']}...")
                print("-" * 50)
        else:
            print("No training examples found")

    except Exception as e:
        print(f"❌ Error showing training examples: {e}")


def show_database_schema():
    """Show database schema"""
    print("\n📋 Database Schema:")
    print("-" * 20)

    try:
        tables = execute_query("""
            SELECT tablename
            FROM pg_tables
            WHERE schemaname = 'public'
            ORDER BY tablename
        """)

        if tables:
            for table in tables:
                table_name = table['tablename']
                print(f"\n📊 {table_name}")

                # Get column information
                columns = execute_query("""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_name = %s AND table_schema = 'public'
                    ORDER BY ordinal_position
                """, (table_name,))

                if columns:
                    for col in columns:
                        nullable = "NULL" if col['is_nullable'] == 'YES' else "NOT NULL"
                        print(f"   {col['column_name']} ({col['data_type']}) {nullable}")
        else:
            print("No tables found")

    except Exception as e:
        print(f"❌ Error showing schema: {e}")


def run_vector_search_test():
    """Test vector search functionality"""
    print("\n🧠 Vector Search Test:")
    print("-" * 25)

    try:
        # Get a sample article with embedding
        result = execute_query_single("""
            SELECT id, LEFT(content, 50) as content_preview
            FROM articles
            WHERE embedding IS NOT NULL
            LIMIT 1
        """)

        if result:
            print(f"Sample article ID: {result['id']}")
            print(f"Content preview: {result['content_preview']}...")

            # Get similar articles using basic similarity (would use pgvector in production)
            similar = execute_query("""
                SELECT id, LEFT(content, 50) as content_preview
                FROM articles
                WHERE embedding IS NOT NULL AND id != %s
                LIMIT 3
            """, (result['id'],))

            if similar:
                print("\n📄 Similar articles:")
                for article in similar:
                    print(f"   ID {article['id']}: {article['content_preview']}...")
            else:
                print("No similar articles found")
        else:
            print("No articles with embeddings found")

    except Exception as e:
        print(f"❌ Error testing vector search: {e}")


def show_connection_pool_status():
    """Show connection pool status"""
    print("\n🔌 Connection Pool Status:")
    print("-" * 25)

    try:
        stats = get_pool_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")
    except Exception as e:
        print(f"❌ Error getting pool status: {e}")


def main():
    """Main interactive loop"""
    print("🚀 JustNewsAgent Database Operations Tool")
    print("Connected to justnews database using connection pooling")

    # Initialize connection pool
    try:
        initialize_connection_pool()
        print("✅ Database connection pool initialized")
    except Exception as e:
        print(f"❌ Failed to initialize database connection: {e}")
        return

    while True:
        show_menu()
        try:
            choice = input("Select an option (1-8): ").strip()

            if choice == "1":
                show_database_stats()
            elif choice == "2":
                limit = input("Number of articles to show (default 5): ").strip()
                limit = int(limit) if limit.isdigit() else 5
                list_recent_articles(limit)
            elif choice == "3":
                query = input("Search query: ").strip()
                if query:
                    limit = input("Number of results (default 5): ").strip()
                    limit = int(limit) if limit.isdigit() else 5
                    search_articles(query, limit)
                else:
                    print("❌ Please enter a search query")
            elif choice == "4":
                limit = input("Number of examples to show (default 5): ").strip()
                limit = int(limit) if limit.isdigit() else 5
                show_training_examples(limit)
            elif choice == "5":
                show_database_schema()
            elif choice == "6":
                run_vector_search_test()
            elif choice == "7":
                show_connection_pool_status()
            elif choice == "8":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid option. Please select 1-8.")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()

