from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class NeuralAssessment(BaseModel):
    """Standardized payload produced by the Fact Checker agent for reasoning."""
    version: str = "1.0"
    confidence: float
    source_credibility: Optional[float] = None
    extracted_claims: List[str] = []
    evidence_matches: List[Dict[str, Any]] = []
    processing_metadata: Dict[str, Any] = {}


class ReasoningInput(BaseModel):
    """Input wrapper for the reasoning pipeline containing a neural assessment and article metadata."""
    assessment: NeuralAssessment
    article_metadata: Optional[Dict[str, Any]] = {}


class PipelineResult(BaseModel):
    version: str = "1.0"
    overall_confidence: float
    verification_status: str
    explanation: Any
    neural_assessment: NeuralAssessment
    logical_validation: Dict[str, Any]
    processing_summary: Dict[str, Any]
