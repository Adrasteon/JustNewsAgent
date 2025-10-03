"""Pydantic models for Kafka pilot events.

These models are intentionally lightweight and used by adapter skeletons and unit tests
as a local validation layer before integrating with Avro/Schema Registry-based serialization.
"""
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class ArticleCreated(BaseModel):
    """Model representing the creation of an article."""

    id: str
    title: str
    url: str
    published_at: Optional[str] = None
    source: str
    summary: Optional[str] = Field(default=None, description="Optional summary of the article")


class ArticlePersisted(BaseModel):
    """Model representing that an article has been persisted to Memory."""

    id: str
    url: str
    persisted_at: str
    source: Optional[str] = None


class TrainingExample(BaseModel):
    id: str
    label: str
    features: Dict[str, Any] = Field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = None
