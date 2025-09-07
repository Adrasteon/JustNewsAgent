# Database Schema & Operations Documentation

## Overview

JustNews V4 implements a comprehensive PostgreSQL database architecture designed for high-performance news processing, vector search, and continuous learning. The system uses connection pooling, optimized indexing, and supports both traditional relational queries and modern vector similarity search.

**Status**: Production Ready (August 2025)  
**Database**: PostgreSQL 13+  
**Architecture**: Connection Pooling + Vector Extensions  
**Performance**: 1000+ articles/second processing capability

## Core Database Schema

### Articles Table
Primary content storage with vector embeddings support:

```sql
CREATE TABLE articles (
    id INTEGER PRIMARY KEY DEFAULT 1,
    content TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    embedding NUMERIC[]
);
```

**Key Features:**
- **Content Storage**: Full article text with metadata
- **Vector Embeddings**: NUMERIC[] array for similarity search
- **JSONB Metadata**: Flexible metadata storage (source, entities, scores)
- **Timestamps**: Automatic creation tracking

### Training Examples Table
Continuous learning data storage:

```sql
CREATE TABLE training_examples (
    id SERIAL PRIMARY KEY,
    task TEXT NOT NULL,
    input JSONB NOT NULL,
    output JSONB NOT NULL,
    critique TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

**Key Features:**
- **Task Classification**: Categorizes training examples by agent/task
- **JSONB Storage**: Flexible input/output format for different data types
- **Critique System**: Stores human feedback and corrections
- **Temporal Tracking**: Creation timestamps for training evolution

### Article Vectors Table
Optimized vector storage for similarity search:

```sql
CREATE TABLE article_vectors (
    article_id INTEGER PRIMARY KEY REFERENCES articles(id),
    vector VECTOR(768)
);
```

**Key Features:**
- **Foreign Key Relationship**: Links to articles table
- **VECTOR Type**: Uses pgvector extension for optimized similarity search
- **768 Dimensions**: Configured for sentence transformer embeddings
- **Primary Key Constraint**: One vector per article

### Crawled URLs Table
Deduplication and crawl tracking:

```sql
CREATE TABLE crawled_urls (
    id SERIAL PRIMARY KEY,
    url TEXT UNIQUE NOT NULL,
    url_hash VARCHAR(64) UNIQUE NOT NULL,
    first_seen TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

**Key Features:**
- **URL Deduplication**: SHA256 hash-based duplicate detection
- **Temporal Tracking**: First and last seen timestamps
- **Unique Constraints**: Prevents duplicate URL processing

### Sources Table
News source management and scoring:

```sql
CREATE TABLE sources (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    domain VARCHAR(255) UNIQUE NOT NULL,
    credibility_score DECIMAL(3,2),
    article_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

### Source Scores Table
Dynamic source credibility tracking:

```sql
CREATE TABLE source_scores (
    id SERIAL PRIMARY KEY,
    source_id INTEGER REFERENCES sources(id),
    score DECIMAL(3,2) NOT NULL,
    calculated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

## Connection Pooling Architecture

### Pool Configuration
```python
# Environment-based configuration
POOL_MIN_CONNECTIONS = int(os.environ.get("DB_POOL_MIN_CONNECTIONS", "2"))
POOL_MAX_CONNECTIONS = int(os.environ.get("DB_POOL_MAX_CONNECTIONS", "10"))

# Global connection pool
_connection_pool: Optional[pool.ThreadedConnectionPool] = None
```

### Connection Management
```python
@contextmanager
def get_db_connection():
    """Context manager for safe connection handling"""
    pool = get_connection_pool()
    conn = None
    try:
        conn = pool.getconn()
        yield conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise
    finally:
        if conn:
            pool.putconn(conn)
```

### Pool Statistics
```python
def get_pool_stats() -> dict:
    """Real-time connection pool monitoring"""
    return {
        "min_connections": POOL_MIN_CONNECTIONS,
        "max_connections": POOL_MAX_CONNECTIONS,
        "connections_in_use": len(pool._used),
        "connections_available": len(pool._rused) - len(pool._used),
        "total_connections": len(pool._rused)
    }
```

## Performance Optimizations

### Indexing Strategy

#### GIN Indexes for JSONB Operations
```sql
-- Metadata search optimization
CREATE INDEX idx_articles_metadata ON articles USING GIN (metadata);

-- Embedding vector operations
CREATE INDEX idx_articles_embedding ON articles USING GIN (embedding);
```

#### Full-Text Search Indexes
```sql
-- Content search optimization
CREATE INDEX idx_articles_content ON articles USING GIN (to_tsvector('english', content));
```

#### Composite Indexes
```sql
-- Common query pattern optimization
CREATE INDEX idx_articles_content_metadata ON articles (id, content, metadata);
```

#### Partial Indexes
```sql
-- Optimize queries for articles with embeddings
CREATE INDEX idx_articles_with_embedding ON articles (id, embedding)
WHERE embedding IS NOT NULL;
```

### Query Optimization Patterns

#### Vector Similarity Search
```python
def find_similar_articles(embedding: list, limit: int = 10) -> list:
    """Optimized vector similarity search"""
    query = """
        SELECT id, content,
               1 - (embedding <=> %s::vector) as similarity
        FROM articles
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    return execute_query(query, (embedding, embedding, limit))
```

#### Full-Text Search
```python
def search_articles(query: str, limit: int = 20) -> list:
    """Full-text search with ranking"""
    sql = """
        SELECT id, content,
               ts_rank_cd(to_tsvector('english', content),
                         plainto_tsquery('english', %s)) as rank
        FROM articles
        WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
        ORDER BY rank DESC
        LIMIT %s
    """
    return execute_query(sql, (query, query, limit))
```

## Database Operations API

### Core Query Functions

#### Execute Query with Results
```python
def execute_query(query: str, params: tuple = None, fetch: bool = True) -> Optional[list]:
    """Execute query with automatic connection management"""
    with get_db_cursor(commit=True) as (conn, cursor):
        cursor.execute(query, params or ())
        if fetch and query.strip().upper().startswith('SELECT'):
            return cursor.fetchall()
        return None
```

#### Single Result Queries
```python
def execute_query_single(query: str, params: tuple = None) -> Optional[dict]:
    """Execute query returning single result"""
    with get_db_cursor() as (conn, cursor):
        cursor.execute(query, params or ())
        result = cursor.fetchone()
        return dict(result) if result else None
```

### Health Monitoring
```python
def health_check() -> bool:
    """Database connectivity health check"""
    try:
        result = execute_query_single("SELECT 1 as health_check")
        return result is not None and result.get('health_check') == 1
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False
```

## Data Management Operations

### Article Storage
```python
def store_article(content: str, metadata: dict, embedding: list = None) -> int:
    """Store article with optional embedding"""
    query = """
        INSERT INTO articles (content, metadata, embedding)
        VALUES (%s, %s, %s)
        RETURNING id
    """
    result = execute_query_single(query, (content, metadata, embedding))
    return result['id'] if result else None
```

### Training Data Management
```python
def store_training_example(task: str, input_data: dict, output_data: dict, critique: str = None) -> int:
    """Store training example for continuous learning"""
    query = """
        INSERT INTO training_examples (task, input, output, critique)
        VALUES (%s, %s, %s, %s)
        RETURNING id
    """
    result = execute_query_single(query, (task, input_data, output_data, critique))
    return result['id'] if result else None
```

### Deduplication System
```python
def register_url(url: str) -> bool:
    """Register URL with deduplication"""
    url_hash = hashlib.sha256(url.encode('utf-8')).hexdigest()
    query = """
        INSERT INTO crawled_urls (url, url_hash, first_seen, last_seen)
        VALUES (%s, %s, now(), now())
        ON CONFLICT (url) DO UPDATE SET last_seen = EXCLUDED.last_seen
        RETURNING (xmax = 0) as inserted
    """
    result = execute_query_single(query, (url, url_hash))
    return result and result.get('inserted', False)
```

## Migration System

### Migration File Structure
```
agents/memory/db_migrations/
├── 001_create_articles_table.sql
├── 002_create_training_examples_table.sql
├── 003_create_article_vectors_table.sql
└── 004_add_performance_indexes.sql
```

### Migration Execution
```python
def apply_migration(migration_file: str) -> bool:
    """Apply database migration from SQL file"""
    try:
        with open(migration_file, 'r') as f:
            sql_content = f.read()

        statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]

        for statement in statements:
            if statement:
                execute_query(statement, fetch=False)

        return True
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False
```

## Environment Configuration

### Required Environment Variables
```bash
# Database Connection
POSTGRES_HOST=localhost
POSTGRES_DB=justnews
POSTGRES_USER=justnews_user
POSTGRES_PASSWORD=your_secure_password

# Connection Pool
DB_POOL_MIN_CONNECTIONS=2
DB_POOL_MAX_CONNECTIONS=10
```

### Docker Configuration
```yaml
version: '3.8'
services:
  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: justnews
      POSTGRES_USER: justnews_user
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init_database.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
```

## Monitoring and Maintenance

### Database Statistics
```python
def get_database_stats() -> dict:
    """Comprehensive database statistics"""
    stats = {}

    # Article counts
    result = execute_query_single("SELECT COUNT(*) as count FROM articles")
    stats['total_articles'] = result['count']

    # Articles with embeddings
    result = execute_query_single("SELECT COUNT(*) as count FROM articles WHERE embedding IS NOT NULL")
    stats['articles_with_embeddings'] = result['count']

    # Training examples
    result = execute_query_single("SELECT COUNT(*) as count FROM training_examples")
    stats['training_examples'] = result['count']

    # Connection pool stats
    stats.update(get_pool_stats())

    return stats
```

### Index Maintenance
```sql
-- Reindex for performance maintenance
REINDEX INDEX idx_articles_metadata;
REINDEX INDEX idx_articles_content;

-- Analyze tables for query optimization
ANALYZE articles;
ANALYZE training_examples;
ANALYZE article_vectors;
```

### Backup Strategy
```bash
# Full database backup
pg_dump -h localhost -U justnews_user -d justnews > backup_$(date +%Y%m%d_%H%M%S).sql

# Continuous archiving setup
wal_level = replica
archive_mode = on
archive_command = 'cp %p /var/lib/postgresql/archive/%f'
```

## Performance Benchmarks

### Throughput Metrics (August 2025)
- **Article Ingestion**: 1000+ articles/second
- **Vector Search**: <100ms average query time
- **Full-Text Search**: <50ms average query time
- **Training Data Storage**: 500+ examples/second

### Memory Usage
- **Connection Pool**: 2-10 connections (configurable)
- **Index Size**: ~30% of total database size
- **Vector Storage**: ~40% of total database size

## Troubleshooting

### Common Issues

#### Connection Pool Exhaustion
**Symptoms:** Database connection errors, slow queries
**Causes:** High concurrent load, connection leaks
**Solutions:**
```python
# Monitor pool status
stats = get_pool_stats()
if stats['connections_in_use'] > stats['max_connections'] * 0.8:
    logger.warning("Connection pool near capacity")

# Implement connection cleanup
close_connection_pool()
initialize_connection_pool()
```

#### Slow Query Performance
**Symptoms:** Queries taking >1 second
**Causes:** Missing indexes, large result sets
**Solutions:**
```sql
-- Analyze query performance
EXPLAIN ANALYZE SELECT * FROM articles WHERE metadata->>'source' = 'bbc';

-- Add missing indexes
CREATE INDEX idx_articles_source ON articles ((metadata->>'source'));
```

#### Vector Search Issues
**Symptoms:** Similarity search returning poor results
**Causes:** Incorrect vector dimensions, normalization issues
**Solutions:**
```python
# Validate vector dimensions
def validate_embedding_dimensions(embedding: list) -> bool:
    return len(embedding) == 768  # Expected dimension

# Normalize embeddings before storage
import numpy as np
normalized_embedding = np.array(embedding) / np.linalg.norm(embedding)
```

## Security Considerations

### Connection Security
- **SSL/TLS**: Enable SSL for production connections
- **Authentication**: Strong passwords and user permissions
- **Network Security**: Restrict database access to application servers

### Data Protection
- **Encryption**: Encrypt sensitive metadata fields
- **Access Control**: Implement row-level security for multi-tenant scenarios
- **Audit Logging**: Log all data access and modifications

### Backup Security
- **Encrypted Backups**: Encrypt database backups
- **Secure Storage**: Store backups in secure, redundant locations
- **Access Controls**: Restrict backup file access

## Development Guidelines

### Schema Changes
1. **Create Migration Files**: Add numbered SQL files to `db_migrations/`
2. **Test Migrations**: Test on development database first
3. **Document Changes**: Update this documentation
4. **Version Control**: Commit migrations with application code

### Query Optimization
1. **Use Indexes**: Ensure proper indexing for query patterns
2. **Limit Results**: Use LIMIT clauses for large datasets
3. **Batch Operations**: Use batch inserts for bulk data
4. **Monitor Performance**: Log slow queries for optimization

### Connection Management
1. **Use Context Managers**: Always use connection context managers
2. **Handle Exceptions**: Properly handle database exceptions
3. **Monitor Pool Usage**: Track connection pool statistics
4. **Graceful Shutdown**: Close connections on application shutdown

---

**Last Updated:** September 7, 2025  
**Version:** 1.0  
**Authors:** JustNews Development Team</content>
<parameter name="filePath">/home/adra/justnewsagent/JustNewsAgent/markdown_docs/agent_documentation/database_schema_operations.md
