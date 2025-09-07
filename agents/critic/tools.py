"""
Critic Agent V2 - Specialized Logical Analysis Tools
Production-ready implementation with zero warnings and clean imports

SPECIALIZATION FOCUS:
1. Argument Structure Analysis - Identifying premises, conclusions, logical flow
2. Editorial Consistency - Checking for internal contradictions, coherence  
3. Logical Fallacy Detection - Identifying logical errors and weak reasoning
4. Source Credibility Assessment - Evaluating evidence quality and sourcing

NOTE: Sentiment and bias analysis have been centralized in Scout V2 Agent.
Use Scout V2 for all sentiment/bias analysis.
"""


import os
import re
import statistics
from datetime import datetime
from typing import Any

from common.observability import get_logger

# Configure logging

logger = get_logger(__name__)

# Feedback logging pattern
FEEDBACK_LOG = os.path.join(os.path.dirname(__file__), "critic_feedback.log")

def log_feedback(event: str, details: dict[str, Any]) -> None:
    """Universal feedback logging for Critic Agent V2."""
    try:
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "agent": "critic_v2",
            "details": details
        }

        with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
            f.write(f"{feedback_entry}\n")

        logger.info(f"📝 Feedback logged: {event}")
    except Exception as e:
        logger.error(f"❌ Feedback logging failed: {e}")

# =============================================================================
# SPECIALIZED LOGICAL ANALYSIS FUNCTIONS
# =============================================================================

def analyze_argument_structure(text: str, url: str = None) -> dict[str, Any]:
    """
    Analyze the logical structure of arguments in text content.
    
    Args:
        text (str): Content to analyze for argument structure
        url (str, optional): Source URL for context
        
    Returns:
        Dict containing argument analysis including premises, conclusions,
        logical flow, and argument strength assessment
    """
    try:
        logger.info(f"🧠 Analyzing argument structure for {len(text)} characters")

        # Extract logical connectors and argument indicators
        premises = _extract_premises(text)
        conclusions = _extract_conclusions(text)
        logical_flow = _analyze_logical_flow(text)
        argument_strength = _assess_argument_strength(text, premises, conclusions)

        analysis = {
            "premises": premises,
            "conclusions": conclusions,
            "logical_flow": logical_flow,
            "argument_strength": argument_strength,
            "structural_analysis": {
                "premise_conclusion_ratio": len(premises) / max(len(conclusions), 1),
                "logical_connectors_count": len(logical_flow.get("connectors", [])),
                "argument_complexity": _calculate_argument_complexity(premises, conclusions),
                "coherence_score": _calculate_coherence_score(text)
            },
            "analysis_metadata": {
                "text_length": len(text),
                "url": url,
                "analysis_time": datetime.now().isoformat(),
                "analyzer_version": "critic_v2_argument_structure"
            }
        }

        logger.info(f"✅ Argument analysis complete: {len(premises)} premises, {len(conclusions)} conclusions")
        return analysis

    except Exception as e:
        logger.error(f"❌ Error in argument structure analysis: {e}")
        return {"error": str(e)}

def assess_editorial_consistency(text: str, url: str = None) -> dict[str, Any]:
    """Assess editorial consistency and internal coherence."""
    try:
        logger.info(f"📝 Assessing editorial consistency for {len(text)} characters")
        contradictions = _detect_contradictions(text)
        coherence_score = _calculate_coherence_score(text)

        return {
            "contradictions": contradictions,
            "coherence_score": coherence_score,
            "consistency_score": max(0, 1.0 - len(contradictions) * 0.2),
            "analysis_time": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Error in editorial consistency: {e}")
        return {"error": str(e)}

def detect_logical_fallacies(text: str, url: str = None) -> dict[str, Any]:
    """Detect logical fallacies and reasoning errors."""
    try:
        logger.info(f"🕵️ Detecting logical fallacies in {len(text)} characters")
        fallacies = _detect_common_fallacies(text)

        return {
            "fallacies_detected": fallacies,
            "fallacy_count": len(fallacies),
            "logical_strength": max(0, 1.0 - len(fallacies) * 0.3),
            "analysis_time": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Error in fallacy detection: {e}")
        return {"error": str(e)}

def assess_source_credibility(text: str, url: str = None) -> dict[str, Any]:
    """Assess source credibility and evidence quality."""
    try:
        logger.info(f"📚 Assessing source credibility for {len(text)} characters")
        citations = _extract_citations(text)

        return {
            "citations": citations,
            "citation_count": len(citations),
            "credibility_score": min(1.0, len(citations) * 0.2),
            "analysis_time": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Error in credibility assessment: {e}")
        return {"error": str(e)}

# =============================================================================
# ESSENTIAL HELPER FUNCTIONS
# =============================================================================

def _extract_premises(text: str) -> list[dict[str, Any]]:
    """Extract premises from argument text."""
    premise_indicators = ['because', 'since', 'given that', 'as', 'due to']
    premises = []
    sentences = re.split(r'[.!?]+', text)

    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence:
            continue
        for indicator in premise_indicators:
            if indicator in sentence.lower():
                premises.append({
                    "text": sentence,
                    "indicator": indicator,
                    "position": i,
                    "strength": 0.7
                })
                break
    return premises

def _extract_conclusions(text: str) -> list[dict[str, Any]]:
    """Extract conclusions from argument text."""
    conclusion_indicators = ['therefore', 'thus', 'hence', 'so', 'consequently']
    conclusions = []
    sentences = re.split(r'[.!?]+', text)

    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence:
            continue
        for indicator in conclusion_indicators:
            if indicator in sentence.lower():
                conclusions.append({
                    "text": sentence,
                    "indicator": indicator,
                    "position": i,
                    "strength": 0.7
                })
                break
    return conclusions

def _analyze_logical_flow(text: str) -> dict[str, Any]:
    """Analyze logical flow and connectors."""
    connectors = ['however', 'but', 'furthermore', 'moreover', 'additionally']
    found_connectors = []

    for connector in connectors:
        if connector in text.lower():
            found_connectors.append({"connector": connector, "type": "transition"})

    return {
        "connectors": found_connectors,
        "flow_coherence": min(1.0, len(found_connectors) / 3.0)
    }

def _assess_argument_strength(text: str, premises: list[dict], conclusions: list[dict]) -> dict[str, Any]:
    """Assess overall argument strength."""
    if not premises and not conclusions:
        return {"strength_score": 0.0, "assessment": "No clear argumentative structure"}

    premise_count = len(premises)
    conclusion_count = len(conclusions)
    balance_score = 1.0 - abs(premise_count - conclusion_count) / max(premise_count + conclusion_count, 1)

    return {
        "strength_score": balance_score,
        "premise_quality": 0.7,
        "conclusion_quality": 0.7,
        "balance_score": balance_score,
        "assessment": "Moderate argumentative structure"
    }

def _calculate_argument_complexity(premises: list[dict], conclusions: list[dict]) -> float:
    """Calculate argument complexity score."""
    return min((len(premises) + len(conclusions)) / 2.0, 10.0)

def _calculate_coherence_score(text: str) -> float:
    """Calculate text coherence."""
    sentences = re.split(r'[.!?]+', text)
    valid_sentences = [s.strip() for s in sentences if s.strip()]

    if len(valid_sentences) < 2:
        return 0.5

    # Simple coherence based on sentence length consistency
    sentence_lengths = [len(s.split()) for s in valid_sentences]
    if len(sentence_lengths) > 1:
        avg_length = statistics.mean(sentence_lengths)
        variation = statistics.stdev(sentence_lengths) if len(sentence_lengths) > 1 else 0
        coherence = 1.0 - min(variation / avg_length, 1.0)
    else:
        coherence = 1.0

    return coherence

def _detect_contradictions(text: str) -> list[dict[str, Any]]:
    """Detect internal contradictions."""
    contradictions = []
    sentences = re.split(r'[.!?]+', text)

    # Simple contradiction detection
    for i, sentence1 in enumerate(sentences):
        for j, sentence2 in enumerate(sentences[i+1:], i+1):
            if 'not' in sentence1.lower() and any(word in sentence2.lower() for word in sentence1.lower().split() if word != 'not'):
                contradictions.append({
                    "sentence1": sentence1.strip(),
                    "sentence2": sentence2.strip(),
                    "confidence": 0.5
                })

    return contradictions[:3]  # Limit to top 3

def _detect_common_fallacies(text: str) -> list[dict[str, Any]]:
    """Detect common logical fallacies."""
    fallacies = []

    # Ad hominem detection
    if any(phrase in text.lower() for phrase in ['attacks', 'character assassination', 'personally']):
        fallacies.append({
            "fallacy": "ad_hominem",
            "confidence": 0.6,
            "description": "Personal attack rather than addressing argument"
        })

    # Appeal to authority
    if any(phrase in text.lower() for phrase in ['expert says', 'authority claims', 'because someone said']):
        fallacies.append({
            "fallacy": "appeal_to_authority",
            "confidence": 0.5,
            "description": "Inappropriate appeal to authority"
        })

    return fallacies

def _extract_citations(text: str) -> list[dict[str, Any]]:
    """Extract citations and references."""
    citations = []

    # Look for citation patterns
    patterns = [
        (r'according to ([A-Z][a-z]+ [A-Z][a-z]+)', 'person'),
        (r'([A-Z][a-z]+ [A-Z][a-z]+) said', 'person'),
        (r'study by ([A-Z][A-Za-z\s]+)', 'study')
    ]

    for pattern, citation_type in patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            citations.append({
                "text": match.group(1),
                "type": citation_type,
                "position": match.start()
            })

    return citations


def get_llama_model():
    """Compatibility shim for tests that expect get_llama_model to exist.

    Return a (model, tokenizer) tuple or (None, None). Tests typically
    monkeypatch this function during unit tests; a lightweight shim
    prevents AttributeError during collection.
    """
    return (None, None)
