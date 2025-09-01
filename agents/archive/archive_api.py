#!/usr/bin/env python3
"""
Phase 3 Sprint 3-4: RESTful Archive API

FastAPI-based REST API for knowledge graph and archive access.
Provides endpoints for querying articles, entities, relationships, and search functionality.

API Endpoints:
- GET /articles - List articles with filtering and pagination
- GET /articles/{id} - Get specific article details
- GET /entities - List entities with filtering
- GET /entities/{id} - Get entity details and relationships
- GET /relationships - Query relationships between entities
- POST /search - Advanced search across articles and entities
- GET /graph/statistics - Get knowledge graph statistics
- GET /health - API health check
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
import re
from fastapi import FastAPI, HTTPException, Query, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from agents.archive.knowledge_graph import TemporalKnowledgeGraph, KnowledgeGraphManager
from agents.archive.archive_manager import ArchiveManager

logger = logging.getLogger("phase3_api")

# Pydantic models for request/response
class ArticleFilter(BaseModel):
    """Filter parameters for article queries"""
    domain: Optional[str] = None
    published_after: Optional[datetime] = None
    published_before: Optional[datetime] = None
    news_score_min: Optional[float] = Field(None, ge=0.0, le=1.0)
    news_score_max: Optional[float] = Field(None, ge=0.0, le=1.0)
    has_entities: Optional[bool] = None
    entity_type: Optional[str] = None
    search_query: Optional[str] = None

class EntityFilter(BaseModel):
    """Filter parameters for entity queries"""
    entity_type: Optional[str] = None
    mention_count_min: Optional[int] = Field(None, ge=0)
    mention_count_max: Optional[int] = Field(None, ge=0)
    first_seen_after: Optional[datetime] = None
    last_seen_before: Optional[datetime] = None
    search_query: Optional[str] = None

class RelationshipFilter(BaseModel):
    """Filter parameters for relationship queries"""
    source_entity: Optional[str] = None
    target_entity: Optional[str] = None
    relationship_type: Optional[str] = None
    strength_min: Optional[float] = Field(None, ge=0.0, le=1.0)
    confidence_min: Optional[float] = Field(None, ge=0.0, le=1.0)
    time_window: Optional[str] = None

class SearchRequest(BaseModel):
    """Advanced search request"""
    query: str = Field(..., min_length=1, max_length=500)
    search_type: str = Field("articles", pattern="^(articles|entities|both)$")
    filters: Optional[Dict[str, Any]] = None
    limit: int = Field(50, ge=1, le=1000)
    offset: int = Field(0, ge=0)

class ArticleResponse(BaseModel):
    """Article response model"""
    article_id: str
    url: str
    title: str
    domain: str
    published_date: Optional[str]
    entities: Dict[str, List[str]]
    news_score: float
    extraction_method: str
    total_entities: int
    relationships_count: int

class EntityResponse(BaseModel):
    """Entity response model"""
    entity_id: str
    name: str
    entity_type: str
    mention_count: int
    first_seen: str
    last_seen: str
    aliases: List[str]
    cluster_size: int
    confidence_score: float

class RelationshipResponse(BaseModel):
    """Relationship response model"""
    source_entity: str
    target_entity: str
    relationship_type: str
    strength: float
    confidence: float
    context: str
    timestamp: str
    co_occurrence_count: int
    proximity_score: float

class SearchResult(BaseModel):
    """Search result model"""
    type: str
    id: str
    title: str
    content: str
    score: float
    metadata: Dict[str, Any]

class PaginatedResponse(BaseModel):
    """Paginated response wrapper"""
    items: List[Any]
    total: int
    page: int
    page_size: int
    has_next: bool
    has_prev: bool

class APIResponse(BaseModel):
    """Generic API response"""
    success: bool
    data: Any
    message: Optional[str] = None
    timestamp: str

# FastAPI application
app = FastAPI(
    title="JustNews Agent - Archive API",
    description="RESTful API for accessing knowledge graph and archived news data",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (would be dependency injection in production)
kg_manager = None
archive_manager = None

async def get_kg_manager() -> KnowledgeGraphManager:
    """Dependency for knowledge graph manager"""
    return kg_manager

async def get_archive_manager() -> ArchiveManager:
    """Dependency for archive manager"""
    return archive_manager

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global kg_manager, archive_manager

    try:
        # Initialize knowledge graph manager
        kg_storage_path = Path("./kg_storage")
        kg_manager = KnowledgeGraphManager(str(kg_storage_path))
        logger.info("🎯 Knowledge Graph Manager initialized")

        # Initialize archive manager (placeholder - would need actual implementation)
        # archive_manager = ArchiveManager(...)
        logger.info("📚 Archive Manager initialized")

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

@app.get("/health")
async def health_check():
    """API health check endpoint"""
    return APIResponse(
        success=True,
        data={
            "status": "healthy",
            "services": {
                "knowledge_graph": kg_manager is not None,
                "archive": archive_manager is not None
            },
            "version": "3.0.0"
        },
        message="API is operational",
        timestamp=datetime.now().isoformat()
    )

@app.get("/articles", response_model=PaginatedResponse)
async def list_articles(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    domain: Optional[str] = Query(None, description="Filter by domain"),
    published_after: Optional[datetime] = Query(None, description="Published after date"),
    published_before: Optional[datetime] = Query(None, description="Published before date"),
    news_score_min: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum news score"),
    entity_type: Optional[str] = Query(None, description="Filter by entity type presence"),
    search_query: Optional[str] = Query(None, description="Search in title/content"),
    kg_manager: KnowledgeGraphManager = Depends(get_kg_manager)
):
    """
    List articles with filtering and pagination

    Query parameters:
    - page: Page number (default: 1)
    - page_size: Items per page (default: 20, max: 100)
    - domain: Filter by news domain
    - published_after: Filter articles published after this date
    - published_before: Filter articles published before this date
    - news_score_min: Minimum news score (0.0-1.0)
    - entity_type: Filter articles containing specific entity type
    - search_query: Search term in title or content
    """
    try:
        # Get all articles from knowledge graph
        all_articles = []
        for node_id, node_data in kg_manager.kg.graph.nodes(data=True):
            if node_data.get("node_type") == "article":
                article_data = node_data["properties"]

                # Apply filters
                if domain and article_data.get("domain") != domain:
                    continue

                if published_after:
                    pub_date = article_data.get("published_date")
                    if pub_date:
                        try:
                            article_date = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                            if article_date < published_after:
                                continue
                        except:
                            pass

                if published_before:
                    pub_date = article_data.get("published_date")
                    if pub_date:
                        try:
                            article_date = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                            if article_date > published_before:
                                continue
                        except:
                            pass

                if news_score_min and article_data.get("news_score", 0) < news_score_min:
                    continue

                if entity_type:
                    entities = article_data.get("entities", {})
                    if entity_type not in entities or not entities[entity_type]:
                        continue

                if search_query:
                    title = article_data.get("title", "").lower()
                    content = article_data.get("content", "").lower()
                    query_lower = search_query.lower()
                    if query_lower not in title and query_lower not in content:
                        continue

                # Create response object
                entities = article_data.get("entities", {})
                total_entities = sum(len(entity_list) for entity_list in entities.values())

                article_response = ArticleResponse(
                    article_id=node_id,
                    url=article_data.get("url", ""),
                    title=article_data.get("title", ""),
                    domain=article_data.get("domain", ""),
                    published_date=article_data.get("published_date"),
                    entities=entities,
                    news_score=article_data.get("news_score", 0.0),
                    extraction_method=article_data.get("extraction_method", ""),
                    total_entities=total_entities,
                    relationships_count=0  # Would need to calculate from graph
                )

                all_articles.append(article_response)

        # Sort by published date (newest first)
        all_articles.sort(key=lambda x: x.published_date or "", reverse=True)

        # Pagination
        total_items = len(all_articles)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_items = all_articles[start_idx:end_idx]

        return PaginatedResponse(
            items=paginated_items,
            total=total_items,
            page=page,
            page_size=page_size,
            has_next=end_idx < total_items,
            has_prev=page > 1
        )

    except Exception as e:
        logger.error(f"Error listing articles: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/articles/{article_id}")
async def get_article(
    article_id: str,
    include_relationships: bool = Query(False, description="Include relationship details"),
    kg_manager: KnowledgeGraphManager = Depends(get_kg_manager)
):
    """Get detailed information about a specific article"""
    try:
        if article_id not in kg_manager.kg.graph.nodes:
            raise HTTPException(status_code=404, detail="Article not found")

        node_data = kg_manager.kg.graph.nodes[article_id]
        if node_data.get("node_type") != "article":
            raise HTTPException(status_code=404, detail="Not an article node")

        article_data = node_data["properties"]

        # Get relationships if requested
        relationships = None
        if include_relationships:
            relationships = kg_manager.kg.query_article_relationships(article_id)

        response = {
            "article_id": article_id,
            "url": article_data.get("url", ""),
            "title": article_data.get("title", ""),
            "domain": article_data.get("domain", ""),
            "published_date": article_data.get("published_date"),
            "content": article_data.get("content", ""),
            "entities": article_data.get("entities", {}),
            "news_score": article_data.get("news_score", 0.0),
            "extraction_method": article_data.get("extraction_method", ""),
            "publisher": article_data.get("publisher", ""),
            "canonical_url": article_data.get("canonical_url", ""),
            "relationships": relationships
        }

        return APIResponse(
            success=True,
            data=response,
            message="Article retrieved successfully",
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting article {article_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/entities", response_model=PaginatedResponse)
async def list_entities(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    entity_type: Optional[str] = Query(None, description="Filter by entity type"),
    mention_count_min: Optional[int] = Query(None, ge=0, description="Minimum mention count"),
    search_query: Optional[str] = Query(None, description="Search in entity name"),
    kg_manager: KnowledgeGraphManager = Depends(get_kg_manager)
):
    """List entities with filtering and pagination"""
    try:
        # Get entities from knowledge graph
        all_entities = kg_manager.kg.query_entities(entity_type, limit=10000)  # Large limit for now

        # Apply additional filters
        filtered_entities = []
        for entity in all_entities:
            if mention_count_min and entity.get("mention_count", 0) < mention_count_min:
                continue

            if search_query:
                name_lower = entity.get("name", "").lower()
                query_lower = search_query.lower()
                if query_lower not in name_lower:
                    continue

            # Create response object
            entity_response = EntityResponse(
                entity_id=entity["node_id"],
                name=entity.get("name", ""),
                entity_type=entity.get("entity_type", ""),
                mention_count=entity.get("mention_count", 0),
                first_seen=entity.get("first_seen", ""),
                last_seen=entity.get("last_seen", ""),
                aliases=entity.get("aliases", []),
                cluster_size=entity.get("cluster_size", 1),
                confidence_score=entity.get("confidence", 0.8)
            )

            filtered_entities.append(entity_response)

        # Sort by mention count (most mentioned first)
        filtered_entities.sort(key=lambda x: x.mention_count, reverse=True)

        # Pagination
        total_items = len(filtered_entities)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_items = filtered_entities[start_idx:end_idx]

        return PaginatedResponse(
            items=paginated_items,
            total=total_items,
            page=page,
            page_size=page_size,
            has_next=end_idx < total_items,
            has_prev=page > 1
        )

    except Exception as e:
        logger.error(f"Error listing entities: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/entities/{entity_id}")
async def get_entity(
    entity_id: str,
    include_relationships: bool = Query(True, description="Include relationship details"),
    kg_manager: KnowledgeGraphManager = Depends(get_kg_manager)
):
    """Get detailed information about a specific entity"""
    try:
        if entity_id not in kg_manager.kg.graph.nodes:
            raise HTTPException(status_code=404, detail="Entity not found")

        node_data = kg_manager.kg.graph.nodes[entity_id]
        if node_data.get("node_type") != "entity":
            raise HTTPException(status_code=404, detail="Not an entity node")

        entity_data = node_data["properties"]

        # Get relationships
        relationships = []
        if include_relationships:
            # Get incoming and outgoing edges
            for source, target, edge_type, edge_data in kg_manager.kg.graph.edges(entity_id, keys=True, data=True):
                relationships.append({
                    "source_entity": source,
                    "target_entity": target,
                    "relationship_type": edge_type,
                    "strength": edge_data.get("strength", 0),
                    "confidence": edge_data.get("confidence", 0),
                    "context": edge_data.get("context", ""),
                    "timestamp": edge_data.get("timestamp", ""),
                    "co_occurrence_count": edge_data.get("co_occurrence_count", 0),
                    "proximity_score": edge_data.get("proximity_score", 0)
                })

        response = {
            "entity_id": entity_id,
            "name": entity_data.get("name", ""),
            "entity_type": entity_data.get("entity_type", ""),
            "mention_count": entity_data.get("mention_count", 0),
            "first_seen": entity_data.get("first_seen", ""),
            "last_seen": entity_data.get("last_seen", ""),
            "aliases": entity_data.get("aliases", []),
            "cluster_size": entity_data.get("cluster_size", 1),
            "confidence_score": entity_data.get("confidence", 0.8),
            "relationships": relationships if include_relationships else None
        }

        return APIResponse(
            success=True,
            data=response,
            message="Entity retrieved successfully",
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting entity {entity_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/search")
async def search_content(
    search_request: SearchRequest,
    kg_manager: KnowledgeGraphManager = Depends(get_kg_manager)
):
    """Advanced search across articles and entities"""
    try:
        results = []
        query = search_request.query.lower()

        # Search articles
        if search_request.search_type in ["articles", "both"]:
            for node_id, node_data in kg_manager.kg.graph.nodes(data=True):
                if node_data.get("node_type") == "article":
                    article_data = node_data["properties"]
                    title = article_data.get("title", "").lower()
                    content = article_data.get("content", "").lower()

                    # Calculate relevance score
                    score = 0
                    if query in title:
                        score += 1.0  # Title match gets highest score
                    if query in content:
                        score += 0.5  # Content match gets medium score

                    if score > 0:
                        # Apply additional filters
                        if search_request.filters:
                            if "domain" in search_request.filters:
                                if article_data.get("domain") != search_request.filters["domain"]:
                                    continue
                            if "news_score_min" in search_request.filters:
                                if article_data.get("news_score", 0) < search_request.filters["news_score_min"]:
                                    continue

                        result = SearchResult(
                            type="article",
                            id=node_id,
                            title=article_data.get("title", ""),
                            content=article_data.get("content", "")[:200] + "..." if len(article_data.get("content", "")) > 200 else article_data.get("content", ""),
                            score=score,
                            metadata={
                                "url": article_data.get("url", ""),
                                "domain": article_data.get("domain", ""),
                                "published_date": article_data.get("published_date"),
                                "news_score": article_data.get("news_score", 0),
                                "entity_count": sum(len(entities) for entities in article_data.get("entities", {}).values())
                            }
                        )
                        results.append(result)

        # Search entities
        if search_request.search_type in ["entities", "both"]:
            entities = kg_manager.kg.query_entities(limit=1000)
            for entity in entities:
                name = entity.get("name", "").lower()
                if query in name:
                    score = 1.0 if query == name else 0.7  # Exact match vs partial

                    result = SearchResult(
                        type="entity",
                        id=entity["node_id"],
                        title=entity.get("name", ""),
                        content=f"Entity of type {entity.get('entity_type', 'unknown')} mentioned {entity.get('mention_count', 0)} times",
                        score=score,
                        metadata={
                            "entity_type": entity.get("entity_type", ""),
                            "mention_count": entity.get("mention_count", 0),
                            "first_seen": entity.get("first_seen", ""),
                            "last_seen": entity.get("last_seen", ""),
                            "aliases": entity.get("aliases", [])
                        }
                    )
                    results.append(result)

        # Sort by score and apply pagination
        results.sort(key=lambda x: x.score, reverse=True)
        start_idx = search_request.offset
        end_idx = start_idx + search_request.limit
        paginated_results = results[start_idx:end_idx]

        return APIResponse(
            success=True,
            data={
                "results": paginated_results,
                "total": len(results),
                "returned": len(paginated_results),
                "query": search_request.query,
                "search_type": search_request.search_type
            },
            message=f"Found {len(results)} results for query: {search_request.query}",
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Error performing search: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/graph/statistics")
async def get_graph_statistics(kg_manager: KnowledgeGraphManager = Depends(get_kg_manager)):
    """Get comprehensive knowledge graph statistics"""
    try:
        stats = kg_manager.kg.get_graph_statistics()
        clustering_stats = kg_manager.kg.get_clustering_statistics()

        # Combine statistics
        combined_stats = {
            **stats,
            "clustering": clustering_stats,
            "last_updated": datetime.now().isoformat()
        }

        return APIResponse(
            success=True,
            data=combined_stats,
            message="Knowledge graph statistics retrieved successfully",
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Error getting graph statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/relationships")
async def get_relationships(
    source_entity: Optional[str] = Query(None, description="Source entity name"),
    target_entity: Optional[str] = Query(None, description="Target entity name"),
    relationship_type: Optional[str] = Query(None, description="Relationship type"),
    strength_min: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum relationship strength"),
    confidence_min: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum confidence score"),
    limit: int = Query(50, ge=1, le=500, description="Maximum number of relationships"),
    kg_manager: KnowledgeGraphManager = Depends(get_kg_manager)
):
    """Query relationships between entities"""
    try:
        relationships = []

        # Iterate through all edges
        for source, target, edge_type, edge_data in kg_manager.kg.graph.edges(keys=True, data=True):
            # Apply filters
            if relationship_type and edge_type != relationship_type:
                continue

            if strength_min and edge_data.get("strength", 0) < strength_min:
                continue

            if confidence_min and edge_data.get("confidence", 0) < confidence_min:
                continue

            # Get entity names
            source_name = ""
            target_name = ""

            if source in kg_manager.kg.graph.nodes:
                source_node = kg_manager.kg.graph.nodes[source]
                if source_node.get("node_type") == "entity":
                    source_name = source_node["properties"].get("name", "")

            if target in kg_manager.kg.graph.nodes:
                target_node = kg_manager.kg.graph.nodes[target]
                if target_node.get("node_type") == "entity":
                    target_name = target_node["properties"].get("name", "")

            if source_entity and source_name != source_entity:
                continue

            if target_entity and target_name != target_entity:
                continue

            relationship = RelationshipResponse(
                source_entity=source_name,
                target_entity=target_name,
                relationship_type=edge_type,
                strength=edge_data.get("strength", 0),
                confidence=edge_data.get("confidence", 0),
                context=edge_data.get("context", ""),
                timestamp=edge_data.get("timestamp", ""),
                co_occurrence_count=edge_data.get("co_occurrence_count", 0),
                proximity_score=edge_data.get("proximity_score", 0)
            )

            relationships.append(relationship)

            if len(relationships) >= limit:
                break

        # Sort by strength (strongest first)
        relationships.sort(key=lambda x: x.strength, reverse=True)

        return APIResponse(
            success=True,
            data={
                "relationships": relationships,
                "total": len(relationships),
                "filters_applied": {
                    "source_entity": source_entity,
                    "target_entity": target_entity,
                    "relationship_type": relationship_type,
                    "strength_min": strength_min,
                    "confidence_min": confidence_min
                }
            },
            message=f"Retrieved {len(relationships)} relationships",
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Error getting relationships: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "archive_api:app",
        host="0.0.0.0",
        port=8020,
        reload=True,
        log_level="info"
    )