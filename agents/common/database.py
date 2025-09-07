"""
Database connection utilities for JustNewsAgent
Provides connection pooling and async database operations
"""

import os
from common.observability import get_logger
from contextlib import contextmanager
from typing import Optional
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

# Configure centralized logging
logger = get_logger(__name__)

# Environment variables
POSTGRES_HOST = os.environ.get("POSTGRES_HOST")
POSTGRES_DB = os.environ.get("POSTGRES_DB")
POSTGRES_USER = os.environ.get("POSTGRES_USER")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD")

# Connection pool configuration
POOL_MIN_CONNECTIONS = int(os.environ.get("DB_POOL_MIN_CONNECTIONS", "2"))
POOL_MAX_CONNECTIONS = int(os.environ.get("DB_POOL_MAX_CONNECTIONS", "10"))

# Global connection pool
_connection_pool: Optional[pool.ThreadedConnectionPool] = None

def initialize_connection_pool():
    """
    Initialize the PostgreSQL connection pool.
    Should be called once at application startup.
    """
    global _connection_pool

    if _connection_pool is not None:
        return _connection_pool

    try:
        _connection_pool = pool.ThreadedConnectionPool(
            minconn=POOL_MIN_CONNECTIONS,
            maxconn=POOL_MAX_CONNECTIONS,
            host=POSTGRES_HOST,
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            connect_timeout=3,
            options='-c search_path=public'
        )
        logger.info(f"✅ Database connection pool initialized with {POOL_MIN_CONNECTIONS}-{POOL_MAX_CONNECTIONS} connections")
        return _connection_pool
    except Exception as e:
        logger.error(f"❌ Failed to initialize connection pool: {e}")
        raise

def get_connection_pool():
    """
    Get the global connection pool instance.
    Initializes it if not already done.
    """
    if _connection_pool is None:
        return initialize_connection_pool()
    return _connection_pool

@contextmanager
def get_db_connection():
    """
    Context manager for getting a database connection from the pool.
    Automatically returns the connection to the pool when done.
    """
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

@contextmanager
def get_db_cursor(commit: bool = False):
    """
    Context manager for getting a database cursor with connection.
    Automatically handles connection pooling and optional commit.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        try:
            yield conn, cursor
            if commit:
                conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database operation error: {e}")
            raise
        finally:
            cursor.close()

def execute_query(query: str, params: tuple = None, fetch: bool = True) -> Optional[list]:
    """
    Execute a database query with automatic connection management.

    Args:
        query: SQL query string
        params: Query parameters
        fetch: Whether to fetch results (for SELECT queries)

    Returns:
        List of results if fetch=True and it's a SELECT query, None otherwise
    """
    with get_db_cursor(commit=True) as (conn, cursor):
        cursor.execute(query, params or ())
        if fetch and query.strip().upper().startswith('SELECT'):
            return cursor.fetchall()
        return None

def execute_query_single(query: str, params: tuple = None) -> Optional[dict]:
    """
    Execute a query and return a single result row.

    Args:
        query: SQL query string
        params: Query parameters

    Returns:
        Single result row as dict, or None if no results
    """
    with get_db_cursor() as (conn, cursor):
        cursor.execute(query, params or ())
        result = cursor.fetchone()
        return dict(result) if result else None

def health_check() -> bool:
    """
    Perform a database health check.

    Returns:
        True if database is healthy, False otherwise
    """
    try:
        result = execute_query_single("SELECT 1 as health_check")
        return result is not None and result.get('health_check') == 1
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False

def get_pool_stats() -> dict:
    """
    Get connection pool statistics.

    Returns:
        Dictionary with pool statistics
    """
    pool = get_connection_pool()
    return {
        "min_connections": POOL_MIN_CONNECTIONS,
        "max_connections": POOL_MAX_CONNECTIONS,
        "connections_in_use": len(pool._used),
        "connections_available": len(pool._rused) - len(pool._used),
        "total_connections": len(pool._rused)
    }

# Cleanup function for graceful shutdown
def close_connection_pool():
    """Close all connections in the pool. Call during application shutdown."""
    global _connection_pool
    if _connection_pool:
        _connection_pool.closeall()
        logger.info("Database connection pool closed")
        _connection_pool = None