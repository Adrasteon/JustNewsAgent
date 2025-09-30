---
title: JustNews Agent - API Documentation
description: Auto-generated description for JustNews Agent - API Documentation
tags: [documentation]
status: current
last_updated: 2025-09-12
---

# JustNews Agent - API Documentation

## Phase 3 Sprint 3-4: Advanced Knowledge Graph APIs

This document provides comprehensive documentation for the RESTful Archive API and GraphQL Query Interface implemented in Phase 3 Sprint 3-4.

## Table of Contents

1. [Overview](#overview)
2. [RESTful Archive API](#restful-archive-api)
3. [GraphQL Query Interface](#graphql-query-interface)
4. [API Examples](#api-examples)
5. [Entity Types](#entity-types)
6. [Authentication](#authentication)
7. [Error Handling](#error-handling)
8. [Performance Considerations](#performance-considerations)

## Overview

The JustNews Agent now provides two powerful APIs for accessing the knowledge graph and archived news data:

- **RESTful Archive API** (Port 8000): Traditional REST endpoints for straightforward queries
- **GraphQL Query Interface** (Port 8020): Advanced GraphQL interface for complex, nested queries

Both APIs provide access to:
- Articles with metadata and entity relationships
- Entities with clustering and confidence scores
- Relationships between entities with strength analysis
- Search functionality across articles and entities
- Knowledge graph statistics and analytics

## RESTful Archive API

### Base URL
```
http://localhost:8021
```

### Endpoints

#### Health Check
```http
GET /health
```

**Response:**
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "services": {
      "knowledge_graph": true,
      "archive": false
    },
    "version": "3.0.0"
  },
  "message": "API is operational",
  "timestamp": "2025-09-01T17:34:39.312650"
}
```

#### List Articles
```http
GET /articles
```

**Query Parameters:**
- `page` (int, default: 1): Page number
- `page_size` (int, default: 20, max: 100): Items per page
- `domain` (string): Filter by news domain
- `published_after` (datetime): Filter articles published after this date
- `published_before` (datetime): Filter articles published before this date
- `news_score_min` (float, 0.0-1.0): Minimum news score
- `news_score_max` (float, 0.0-1.0): Maximum news score
- `entity_type` (string): Filter articles containing specific entity type
- `search_query` (string): Search in title/content

**Response:**
```json
{
  "items": [
    {
      "article_id": "article_001",
      "url": "https://example.com/article",
      "title": "Article Title",
      "domain": "example.com",
      "published_date": "2025-09-01T10:00:00Z",
      "entities": {
        "PERSON": ["John Doe", "Jane Smith"],
        "ORG": ["Example Corp"],
        "GPE": ["New York"]
      },
      "news_score": 0.85,
      "extraction_method": "advanced",
      "total_entities": 4,
      "relationships_count": 3
    }
  ],
  "total": 150,
  "page": 1,
  "page_size": 20,
  "has_next": true,
  "has_prev": false
}
```

#### Get Article Details
```http
GET /articles/{article_id}
```

**Query Parameters:**
- `include_relationships` (boolean, default: false): Include relationship details

**Response:**
```json
{
  "success": true,
  "data": {
    "article_id": "article_001",
    "url": "https://example.com/article",
    "title": "Article Title",
    "domain": "example.com",
    "published_date": "2025-09-01T10:00:00Z",
    "content": "Full article content...",
    "entities": {
      "PERSON": ["John Doe"],
      "ORG": ["Example Corp"]
    },
    "news_score": 0.85,
    "extraction_method": "advanced",
    "publisher": "Example News",
    "canonical_url": "https://example.com/article",
    "relationships": null
  },
  "message": "Article retrieved successfully",
  "timestamp": "2025-09-01T17:34:39.312650"
}
```

#### List Entities
```http
GET /entities
```

**Query Parameters:**
- `page` (int, default: 1): Page number
- `page_size` (int, default: 20, max: 100): Items per page
- `entity_type` (string): Filter by entity type
- `mention_count_min` (int): Minimum mention count
- `mention_count_max` (int): Maximum mention count
- `first_seen_after` (datetime): Filter entities first seen after this date
- `last_seen_before` (datetime): Filter entities last seen before this date
- `search_query` (string): Search in entity name

**Response:**
```json
{
  "items": [
    {
      "entity_id": "entity_001",
      "name": "John Doe",
      "entity_type": "PERSON",
      "mention_count": 15,
      "first_seen": "2025-08-15T08:30:00Z",
      "last_seen": "2025-09-01T14:20:00Z",
      "aliases": ["J. Doe", "Johnathan Doe"],
      "cluster_size": 3,
      "confidence_score": 0.92
    }
  ],
  "total": 68,
  "page": 1,
  "page_size": 20,
  "has_next": true,
  "has_prev": false
}
```

#### Get Entity Details
```http
GET /entities/{entity_id}
```

**Query Parameters:**
- `include_relationships` (boolean, default: true): Include relationship details

**Response:**
```json
{
  "success": true,
  "data": {
    "entity_id": "entity_001",
    "name": "John Doe",
    "entity_type": "PERSON",
    "mention_count": 15,
    "first_seen": "2025-08-15T08:30:00Z",
    "last_seen": "2025-09-01T14:20:00Z",
    "aliases": ["J. Doe"],
    "cluster_size": 3,
    "confidence_score": 0.92,
    "relationships": [
      {
        "source_entity": "entity_001",
        "target_entity": "entity_002",
        "relationship_type": "mentioned_at_time",
        "strength": 0.85,
        "confidence": 0.78,
        "context": "John Doe was mentioned in relation to Example Corp",
        "timestamp": "2025-09-01T10:15:00Z",
        "co_occurrence_count": 5,
        "proximity_score": 0.72
      }
    ]
  },
  "message": "Entity retrieved successfully",
  "timestamp": "2025-09-01T17:34:39.312650"
}
```

#### Advanced Search
```http
POST /search
```

**Request Body:**
```json
{
  "query": "Microsoft",
  "search_type": "both",
  "filters": {
    "domain": "bbc.com",
    "news_score_min": 0.7
  },
  "limit": 50,
  "offset": 0
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "results": [
      {
        "type": "article",
        "id": "article_001",
        "title": "Microsoft Announces New AI Features",
        "content": "Microsoft Corporation today announced...",
        "score": 1.0,
        "metadata": {
          "url": "https://example.com/microsoft-ai",
          "domain": "example.com",
          "published_date": "2025-09-01T09:00:00Z",
          "news_score": 0.88,
          "entity_count": 3
        }
      },
      {
        "type": "entity",
        "id": "entity_045",
        "title": "Microsoft Corporation",
        "content": "Entity of type ORG mentioned 12 times",
        "score": 1.0,
        "metadata": {
          "entity_type": "ORG",
          "mention_count": 12,
          "first_seen": "2025-08-20T11:30:00Z",
          "last_seen": "2025-09-01T16:45:00Z",
          "aliases": ["Microsoft Corp", "MSFT"]
        }
      }
    ],
    "total": 15,
    "returned": 10,
    "query": "Microsoft",
    "search_type": "both"
  },
  "message": "Found 15 results for query: Microsoft",
  "timestamp": "2025-09-01T17:34:39.312650"
}
```

#### Get Graph Statistics
```http
GET /graph/statistics
```

**Response:**
```json
{
  "success": true,
  "data": {
    "total_nodes": 73,
    "total_edges": 108,
    "node_types": {
      "article": 5,
      "entity": 68
    },
    "edge_types": {
      "mentions": 54,
      "mentioned_at_time": 54
    },
    "entity_types": {
      "PERSON": 23,
      "GPE": 43,
      "ORG": 2
    },
    "temporal_coverage": {},
    "clustering": {
      "total_entity_nodes": 68,
      "clustered_entities": 0,
      "clusters_by_type": {},
      "cluster_sizes": [],
      "average_cluster_size": 0,
      "largest_cluster": 0
    },
    "last_updated": "2025-09-01T17:34:39.312650"
  },
  "message": "Knowledge graph statistics retrieved successfully",
  "timestamp": "2025-09-01T17:34:39.312650"
}
```

#### Query Relationships
```http
GET /relationships
```

**Query Parameters:**
- `source_entity` (string): Source entity name
- `target_entity` (string): Target entity name
- `relationship_type` (string): Relationship type
- `strength_min` (float, 0.0-1.0): Minimum relationship strength
- `confidence_min` (float, 0.0-1.0): Minimum confidence score
- `limit` (int, default: 50, max: 500): Maximum number of relationships

**Response:**
```json
{
  "success": true,
  "data": {
    "relationships": [
      {
        "source_entity": "John Doe",
        "target_entity": "Example Corp",
        "relationship_type": "mentioned_at_time",
        "strength": 0.85,
        "confidence": 0.78,
        "context": "John Doe was mentioned in relation to Example Corp",
        "timestamp": "2025-09-01T10:15:00Z",
        "co_occurrence_count": 5,
        "proximity_score": 0.72
      }
    ],
    "total": 25,
    "filters_applied": {
      "source_entity": null,
      "target_entity": null,
      "relationship_type": null,
      "strength_min": 0.0,
      "confidence_min": 0.0
    }
  },
  "message": "Retrieved 25 relationships",
  "timestamp": "2025-09-01T17:34:39.312650"
}
```

## GraphQL Query Interface

### Base URL
```
http://localhost:8020
```

### GraphQL Schema

#### Query Root
```graphql
type Query {
  # Health check
  health: String

  # Single article
  article(id: String!): ArticleType

  # List articles with filtering
  articles(
    limit: Int = 20,
    offset: Int = 0,
    domain: String,
    publishedAfter: DateTime,
    publishedBefore: DateTime,
    newsScoreMin: Float = 0.0,
    newsScoreMax: Float = 1.0,
    entityType: EntityTypeEnum,
    searchQuery: String,
    sortBy: String = "published_date",
    sortOrder: String = "desc"
  ): [ArticleType]

  # Single entity
  entity(id: String!): EntityType

  # List entities with filtering
  entities(
    limit: Int = 20,
    offset: Int = 0,
    entityType: EntityTypeEnum,
    mentionCountMin: Int = 0,
    mentionCountMax: Int = 1000,
    firstSeenAfter: DateTime,
    lastSeenBefore: DateTime,
    searchQuery: String,
    sortBy: String = "mention_count",
    sortOrder: String = "desc"
  ): [EntityType]

  # List relationships
  relationships(
    limit: Int = 50,
    offset: Int = 0,
    sourceEntity: String,
    targetEntity: String,
    relationshipType: String,
    strengthMin: Float = 0.0,
    confidenceMin: Float = 0.0,
    sortBy: String = "strength",
    sortOrder: String = "desc"
  ): [RelationshipType]

  # Advanced search
  search(
    query: String!,
    searchType: String = "both",
    limit: Int = 50,
    offset: Int = 0,
    filters: JSONString
  ): [SearchResultType]

  # Graph statistics
  graphStatistics: GraphStatisticsType
}
```

#### Types

```graphql
enum EntityTypeEnum {
  PERSON
  ORG
  GPE
  EVENT
  MONEY
  DATE
  TIME
  PERCENT
  QUANTITY
}

type ArticleType {
  articleId: String!
  url: String!
  title: String!
  domain: String
  publishedDate: DateTime
  content: String
  entities: JSONString
  newsScore: Float
  extractionMethod: String
  publisher: String
  canonicalUrl: String

  # Related entities
  relatedEntities(
    limit: Int = 20,
    offset: Int = 0,
    entityType: EntityTypeEnum,
    minConfidence: Float = 0.0
  ): [EntityType]
}

type EntityType {
  entityId: String!
  name: String!
  entityType: EntityTypeEnum!
  mentionCount: Int!
  firstSeen: DateTime
  lastSeen: DateTime
  aliases: [String]
  clusterSize: Int
  confidenceScore: Float

  # Relationships
  relationships(
    limit: Int = 50,
    offset: Int = 0,
    relationshipType: String,
    minStrength: Float = 0.0,
    minConfidence: Float = 0.0
  ): [RelationshipType]
}

type RelationshipType {
  sourceEntityId: String!
  targetEntityId: String!
  relationshipType: String!
  strength: Float!
  confidence: Float!
  context: String
  timestamp: DateTime
  coOccurrenceCount: Int
  proximityScore: Float
}

type GraphStatisticsType {
  totalNodes: Int
  totalEdges: Int
  nodeTypes: JSONString
  edgeTypes: JSONString
  entityTypes: JSONString
  temporalCoverage: JSONString
  clusteringStats: JSONString
  lastUpdated: DateTime
}

type SearchResultType {
  type: String!
  id: String!
  title: String!
  content: String
  score: Float!
  metadata: JSONString
}
```

## API Examples

### REST API Examples

#### Get Recent Articles from BBC
```bash
curl "http://localhost:8021/articles?domain=bbc.com&page=1&page_size=10&sort=published_date&order=desc"
```

#### Search for Articles About AI
```bash
curl -X POST http://localhost:8021/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "artificial intelligence",
    "search_type": "articles",
    "limit": 20,
    "filters": {
      "news_score_min": 0.7,
      "published_after": "2025-08-01T00:00:00Z"
    }
  }'
```

#### Get All PERSON Entities
```bash
curl "http://localhost:8021/entities?entity_type=PERSON&page=1&page_size=50&sort=mention_count&order=desc"
```

#### Get Relationships for Microsoft
```bash
curl "http://localhost:8021/relationships?source_entity=Microsoft&limit=20"
```

### GraphQL Examples

#### Basic Health Check
```graphql
{
  health
}
```

#### Get Recent Articles with Entities
```graphql
{
  articles(limit: 10, sortBy: "published_date", sortOrder: "desc") {
    articleId
    title
    domain
    publishedDate
    newsScore
    entities
  }
}
```

#### Complex Entity Query with Relationships
```graphql
{
  entities(limit: 5, entityType: PERSON, sortBy: "mention_count", sortOrder: "desc") {
    entityId
    name
    mentionCount
    confidenceScore
    relationships(limit: 10, minStrength: 0.5) {
      targetEntityId
      relationshipType
      strength
      confidence
      context
    }
  }
}
```

#### Advanced Search with Filtering
```graphql
{
  search(query: "climate change", searchType: "both", limit: 20) {
    type
    title
    content
    score
    metadata
  }
}
```

#### Get Article with Related Entities
```graphql
{
  article(id: "article_001") {
    articleId
    title
    content
    entities
    relatedEntities(limit: 10, entityType: PERSON) {
      name
      entityType
      mentionCount
      confidenceScore
    }
  }
}
```

#### Comprehensive Graph Statistics
```graphql
{
  graphStatistics {
    totalNodes
    totalEdges
    nodeTypes
    edgeTypes
    entityTypes
    clusteringStats
    lastUpdated
  }
}
```

## Entity Types

The system supports the following entity types:

| Type | Description | Examples |
|------|-------------|----------|
| PERSON | People and individuals | "John Doe", "Jane Smith", "Dr. Robert Johnson" |
| ORG | Organizations and companies | "Microsoft Corporation", "United Nations", "BBC" |
| GPE | Geographic locations | "New York", "London", "California", "Europe" |
| EVENT | Events and occurrences | "World War II", "Olympic Games", "COVID-19 Pandemic" |
| MONEY | Monetary values | "$2.5 billion", "€1.2 million", "£500,000" |
| DATE | Dates and time periods | "September 1, 2025", "Q3 2025", "2025" |
| TIME | Time expressions | "3:30 PM", "morning", "afternoon" |
| PERCENT | Percentage values | "15%", "2.5 percent", "75.3%" |
| QUANTITY | Quantities and measurements | "100 tons", "5 kilometers", "2 hours" |

## Authentication ✅ **COMPLETED**

The JustNews Agent now includes a complete JWT-based authentication system with role-based access control. The authentication API runs on port 8021 and provides comprehensive user management capabilities.

### Authentication Architecture

- **JWT-Based Authentication**: Secure token-based authentication with access tokens (30min) and refresh tokens (7 days)
- **Role-Based Access Control**: Three-tier system (ADMIN, RESEARCHER, VIEWER) with hierarchical permissions
- **Secure Database Separation**: Dedicated `justnews_auth` PostgreSQL database for complete security isolation
- **Security Standards**: PBKDF2 password hashing, account lockout (30min after 5 failed attempts), secure token refresh
- **Session Management**: Refresh token storage, validation, and secure session revocation

### Authentication API Endpoints

#### Base URL
```
http://localhost:8021/auth
```

#### User Registration
```http
POST /auth/register
```

**Request Body:**
```json
{
  "email": "researcher@example.com",
  "username": "researcher1",
  "full_name": "Research User",
  "password": "securepassword123",
  "role": "researcher"
}
```

**Response:**
```json
{
  "message": "User registered successfully. Please check your email for activation instructions.",
  "user_id": 7,
  "requires_activation": true
}
```

#### User Login
```http
POST /auth/login
```

**Request Body:**
```json
{
  "username_or_email": "researcher1",
  "password": "securepassword123"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800,
  "user": {
    "user_id": 7,
    "username": "researcher1",
    "email": "researcher@example.com",
    "full_name": "Research User",
    "role": "researcher",
    "last_login": "2025-09-01T18:30:00Z"
  }
}
```

#### Get Current User Info
```http
GET /auth/me
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "user_id": 7,
  "email": "researcher@example.com",
  "username": "researcher1",
  "full_name": "Research User",
  "role": "researcher",
  "status": "active",
  "created_at": "2025-09-01T17:45:00Z",
  "last_login": "2025-09-01T18:30:00Z"
}
```

#### Refresh Access Token
```http
POST /auth/refresh
```

**Request Body:**
```json
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

#### Admin: List Users
```http
GET /auth/users
Authorization: Bearer <admin_access_token>
```

**Query Parameters:**
- `limit` (int, default: 100): Maximum number of users
- `offset` (int, default: 0): Pagination offset

**Response:**
```json
[
  {
    "user_id": 7,
    "email": "researcher@example.com",
    "username": "researcher1",
    "full_name": "Research User",
    "role": "researcher",
    "status": "active",
    "created_at": "2025-09-01T17:45:00Z",
    "last_login": "2025-09-01T18:30:00Z"
  }
]
```

#### Admin: Activate User Account
```http
PUT /auth/users/{user_id}/activate
Authorization: Bearer <admin_access_token>
```

**Response:**
```json
{
  "message": "User account activated successfully"
}
```

#### Admin: Deactivate User Account
```http
PUT /auth/users/{user_id}/deactivate
Authorization: Bearer <admin_access_token>
```

**Response:**
```json
{
  "message": "User account deactivated successfully"
}
```

#### Password Reset Request
```http
POST /auth/password-reset
```

**Request Body:**
```json
{
  "email": "researcher@example.com"
}
```

**Response:**
```json
{
  "message": "If an account with this email exists, a password reset link has been sent."
}
```

#### Password Reset Confirmation
```http
POST /auth/password-reset/confirm
```

**Request Body:**
```json
{
  "token": "reset_token_here",
  "new_password": "new_secure_password123"
}
```

**Response:**
```json
{
  "message": "Password reset successfully"
}
```

### Authentication Integration

#### Using Authentication with Archive APIs

All archive API endpoints now support authentication. Include the access token in the Authorization header:

```bash
# Example: Get articles with authentication
curl -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  "http://localhost:8021/articles?page=1&page_size=10"

# Example: Search with authentication
curl -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -X POST http://localhost:8021/search \
  -H "Content-Type: application/json" \
  -d '{"query": "climate change", "search_type": "both", "limit": 10}'
```

#### Role-Based Permissions

| Role | Archive API Access | Admin Functions | Description |
|------|-------------------|-----------------|-------------|
| **VIEWER** | Read-only access to articles and entities | ❌ No admin access | Basic research access |
| **RESEARCHER** | Full read access, search, and analytics | ❌ No admin access | Full research capabilities |
| **ADMIN** | Full access to all endpoints | ✅ User management, activation, deactivation | System administration |

### Authentication Error Handling

#### Common Error Responses

**Invalid Credentials:**
```json
{
  "detail": "Invalid username or password"
}
```

**Account Not Active:**
```json
{
  "detail": "Account is not active"
}
```

**Invalid Token:**
```json
{
  "detail": "Invalid authentication credentials",
  "headers": {"WWW-Authenticate": "Bearer"}
}
```

**Insufficient Permissions:**
```json
{
  "detail": "Admin access required"
}
```

**Account Locked:**
```json
{
  "detail": "Account is temporarily locked due to too many failed login attempts"
}
```

### Security Features

- **Password Security**: PBKDF2 hashing with salt and 100,000 iterations
- **Account Protection**: Automatic lockout after 5 failed login attempts (30-minute cooldown)
- **Token Security**: Short-lived access tokens (30 minutes) with secure refresh mechanism
- **Session Management**: Secure refresh token storage and validation
- **Audit Logging**: Complete logging of authentication events and API access
- **Database Isolation**: Separate authentication database for security compliance

### Getting Started with Authentication

1. **Start the Authentication API:**
```bash
cd /home/adra/JustNewsAgent
conda run --name justnews-v2-prod uvicorn agents.archive.archive_api:app --reload --port 8021
```

2. **API Documentation:**
```bash
# Interactive API docs
curl http://localhost:8021/docs

# Health check
curl http://localhost:8021/auth/health
```

3. **Create Admin User (First Time Setup):**
```bash
curl -X POST http://localhost:8021/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "admin@justnewsagent.com",
    "username": "admin",
    "full_name": "System Administrator",
    "password": "secure_admin_password",
    "role": "admin"
  }'
```

4. **Activate Admin Account:**
```bash
# Login as admin first, then use the returned access token
curl -X PUT http://localhost:8021/auth/users/1/activate \
  -H "Authorization: Bearer ADMIN_ACCESS_TOKEN"
```

5. **Test Authentication:**
```bash
# Login to get tokens
curl -X POST http://localhost:8021/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username_or_email": "admin",
    "password": "secure_admin_password"
  }'

# Use access token with archive API
curl -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  http://localhost:8021/articles
```

## Analytics Dashboard API

### Base URL
```
http://localhost:8012
```

### Endpoints

#### Health Check
```http
GET /api/health
```

**Response:**
```json
{
  "overall_health_score": 85.5,
  "status": "healthy",
  "total_operations": 1250,
  "success_rate_pct": 94.2,
  "avg_processing_time_s": 1.8,
  "avg_throughput_items_per_s": 45.6,
  "peak_gpu_memory_mb": 8192,
  "avg_gpu_utilization_pct": 72.3,
  "active_agents": 6
}
```

#### Get Real-time Analytics
```http
GET /api/realtime/{hours}
```

**Parameters:**
- `hours` (int): Number of hours to analyze (1-24)

**Response:**
```json
{
  "total_operations": 1250,
  "success_rate_pct": 94.2,
  "avg_processing_time_s": 1.8,
  "avg_throughput_items_per_s": 45.6,
  "peak_gpu_memory_mb": 8192,
  "avg_gpu_utilization_pct": 72.3,
  "bottleneck_indicators": [
    "GPU memory usage approaching 80%",
    "Agent queue depth increasing"
  ],
  "optimization_recommendations": [
    "Consider increasing batch sizes for GPU efficiency",
    "Optimize memory allocation for synthesizer agent"
  ],
  "performance_trends": {
    "throughput_trend": "increasing",
    "memory_trend": "stable",
    "latency_trend": "decreasing"
  }
}
```

#### Get Agent Performance Profile
```http
GET /api/agent/{agent_name}/{hours}
```

**Parameters:**
- `agent_name` (string): Name of the agent (scout, analyst, synthesizer, fact_checker, newsreader, memory)
- `hours` (int): Number of hours to analyze (1-168)

**Response:**
```json
{
  "agent_name": "scout",
  "performance_stats": {
    "avg_processing_time_s": 2.1,
    "success_rate_pct": 96.5,
    "avg_throughput_items_per_s": 38.7,
    "peak_memory_mb": 4096,
    "total_operations": 450,
    "error_count": 15
  },
  "resource_usage": {
    "avg_cpu_percent": 45.2,
    "avg_gpu_percent": 68.3,
    "avg_memory_mb": 2048,
    "peak_memory_mb": 4096
  },
  "optimization_suggestions": [
    "Consider increasing GPU memory allocation",
    "Batch size could be optimized for better throughput"
  ]
}
```

#### Get Performance Trends
```http
GET /api/trends/{hours}
```

**Parameters:**
- `hours` (int): Number of hours to analyze (1-168)

**Response:**
```json
{
  "trends": {
    "throughput_trend": "increasing",
    "memory_trend": "stable",
    "latency_trend": "decreasing",
    "error_rate_trend": "stable"
  },
  "bottlenecks": [
    "GPU memory utilization peaking at 85%",
    "Network latency affecting API response times"
  ],
  "recommendations": [
    "Increase GPU memory allocation for peak loads",
    "Implement response caching for frequently accessed data",
    "Consider load balancing for high-traffic periods"
  ]
}
```

#### Get Analytics Report
```http
GET /api/report/{hours}
```

**Parameters:**
- `hours` (int): Number of hours to analyze (1-168)

**Response:**
```json
{
  "summary": {
    "total_operations": 1250,
    "success_rate": 94.2,
    "avg_throughput": 45.6,
    "peak_memory_usage": 8192,
    "analysis_period_hours": 24
  },
  "agent_performance": {
    "scout": {
      "operations": 300,
      "success_rate": 96.5,
      "avg_time": 2.1
    },
    "analyst": {
      "operations": 250,
      "success_rate": 95.2,
      "avg_time": 1.8
    }
  },
  "system_health": {
    "overall_score": 85.5,
    "critical_issues": 0,
    "warnings": 2,
    "recommendations": 3
  },
  "performance_metrics": {
    "cpu_utilization": 45.2,
    "gpu_utilization": 72.3,
    "memory_utilization": 68.7,
    "network_latency": 15.3
  },
  "generated_at": "2025-09-02T10:30:00Z"
}
```

#### Get Optimization Recommendations
```http
GET /api/optimization-recommendations
```

**Query Parameters:**
- `hours` (int, default: 24): Number of hours to analyze

**Response:**
```json
[
  {
    "id": "gpu_memory_optimization",
    "category": "memory",
    "priority": "high",
    "title": "GPU Memory Optimization",
    "description": "GPU memory usage is approaching 80% capacity. Consider optimizing memory allocation.",
    "impact_score": 85,
    "confidence_score": 92,
    "complexity": "medium",
    "time_savings": 2.5,
    "affected_agents": ["synthesizer", "analyst"],
    "implementation_steps": [
      "Increase GPU memory allocation from 6GB to 8GB",
      "Implement memory pooling for tensor operations",
      "Add memory cleanup after batch processing"
    ]
  }
]
```

#### Get Optimization Insights
```http
GET /api/optimization-insights
```

**Response:**
```json
{
  "total_recommendations_generated": 12,
  "average_impact_score": 78.5,
  "most_common_category": "performance",
  "recommendations_by_priority": {
    "critical": 2,
    "high": 4,
    "medium": 5,
    "low": 1
  }
}
```

#### Record Custom Metric
```http
POST /api/record-metric
```

**Request Body:**
```json
{
  "agent_name": "scout",
  "operation": "crawl",
  "processing_time_s": 2.5,
  "batch_size": 10,
  "success": true,
  "gpu_memory_allocated_mb": 2048.0,
  "gpu_memory_reserved_mb": 3072.0,
  "gpu_utilization_pct": 75.0,
  "temperature_c": 65.0,
  "power_draw_w": 180.0,
  "throughput_items_per_s": 4.0
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Metric recorded successfully"
}
```

### Dashboard Web Interface

The analytics dashboard provides a comprehensive web interface accessible at:
```
http://localhost:8012/
```

#### Features
- **Automatic Data Loading**: Dashboard loads data automatically on page load
- **Real-time Updates**: Live data refresh with manual refresh capability
- **Interactive Controls**: Time range selection and agent filtering
- **Visual Analytics**: Chart.js visualizations for performance trends
- **Export Reports**: JSON export functionality for analytics data
- **Responsive Design**: Mobile-friendly dashboard interface

#### Dashboard Sections
1. **System Health**: Overall health score and key metrics
2. **Performance Overview**: Throughput, processing time, and resource usage
3. **Performance Trends**: Historical charts and trend analysis
4. **GPU Resource Usage**: GPU utilization and memory tracking
5. **Bottlenecks & Recommendations**: Current issues and optimization suggestions
6. **Advanced Optimization**: Detailed recommendations with implementation steps
7. **Agent Profiles**: Per-agent performance analysis

### Usage Examples

#### Get System Health
```bash
curl http://localhost:8012/api/health
```

#### Get Real-time Analytics (Last Hour)
```bash
curl http://localhost:8012/api/realtime/1
```

#### Get Scout Agent Performance (Last 24 Hours)
```bash
curl http://localhost:8012/api/agent/scout/24
```

#### Get Optimization Recommendations
```bash
curl http://localhost:8012/api/optimization-recommendations?hours=24
```

## Error Handling

### REST API Error Responses

#### Standard Error Format
```json
{
  "success": false,
  "message": "Error description",
  "timestamp": "2025-09-01T17:34:39.312650"
}
```

#### Common HTTP Status Codes
- `200` - Success
- `400` - Bad Request (invalid parameters)
- `404` - Not Found (resource doesn't exist)
- `500` - Internal Server Error

#### Validation Errors
```json
{
  "success": false,
  "message": "Validation error: page_size must be between 1 and 100",
  "timestamp": "2025-09-01T17:34:39.312650"
}
```

### GraphQL Error Handling

GraphQL returns errors in the `errors` array alongside any successful data:

```json
{
  "data": { ... },
  "errors": [
    {
      "message": "Field 'invalidField' is not defined",
      "locations": [{"line": 3, "column": 5}]
    }
  ]
}
```

## Performance Considerations

### Current Performance Metrics
- **Knowledge Graph Size**: 73 nodes, 108 relationships
- **Entity Count**: 68 entities across 9 types
- **Article Count**: 5 articles with full entity extraction
- **Response Times**: < 100ms for most queries
- **Concurrent Users**: Supports multiple simultaneous connections

### Optimization Features
- **Pagination**: All list endpoints support pagination to manage large result sets
- **Filtering**: Server-side filtering reduces data transfer and processing
- **Indexing**: Graph database provides efficient query execution
- **Caching**: Future implementation planned for frequently accessed data

### Best Practices

#### REST API
1. Use appropriate page sizes (20-50 items recommended)
2. Apply filters to reduce result sets
3. Use specific entity IDs when possible
4. Cache frequently accessed data client-side

#### GraphQL
1. Request only needed fields to minimize response size
2. Use aliases for multiple similar queries
3. Leverage fragments for reusable field selections
4. Consider query complexity and depth limits

### Rate Limiting
**Current Status:** No rate limiting implemented (development mode)
**Planned:** Token bucket algorithm with configurable limits per user/API key

### Monitoring
- All API calls are logged with timestamps and parameters
- Performance metrics are collected for optimization
- Error rates and response times are tracked
- GraphQL query complexity is monitored

## Getting Started

### Starting the APIs

1. **RESTful Archive API** (Port 8000):
```bash
cd /home/adra/JustNewsAgent
PYTHONPATH=/home/adra/JustNewsAgent python agents/archive/archive_api.py
```

2. **GraphQL Query Interface** (Port 8020):
```bash
cd /home/adra/JustNewsAgent
PYTHONPATH=/home/adra/JustNewsAgent python agents/archive/archive_graphql.py
```

### Testing the APIs

1. **Health Checks**:
```bash
# REST API
curl http://localhost:8021/health

# GraphQL API
curl http://localhost:8020/health
```

2. **GraphQL Playground**:
   - Open browser to: `http://localhost:8020/graphql`
   - Interactive query interface with documentation

3. **API Testing Scripts**:
```bash
# Test REST API
python test_archive_api.py

# Test GraphQL API
python test_graphql_api.py
```

## Future Enhancements

### Phase 3 Sprint 4-4 (Planned)
- **Researcher Authentication**: JWT-based auth with role management
- **Legal Compliance**: GDPR compliance and data retention policies
- **Performance Optimization**: Caching, indexing, and query optimization
- **API Documentation**: Interactive API docs with examples

### Long-term Roadmap
- **Real-time Updates**: WebSocket support for live data updates
- **Advanced Analytics**: Query analytics and usage statistics
- **Federation**: Distributed knowledge graph support
- **Machine Learning**: AI-powered query suggestions and insights
- **Multi-tenancy**: Support for multiple research organizations

---

**Version:** 3.0.0
**Last Updated:** September 1, 2025
**API Status:** Production Ready (Development Environment)
**Knowledge Graph:** 73 nodes, 108 relationships, 68 entities</content>
<parameter name="filePath">/home/adra/JustNewsAgent/docs/PHASE3_API_DOCUMENTATION.md

## See also

- Technical Architecture: markdown_docs/TECHNICAL_ARCHITECTURE.md
- Documentation Catalogue: docs/DOCUMENTATION_CATALOGUE.md

