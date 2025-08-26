
"""
Specialized Analyst Agent Tools - Production Ready
Focused on quantitative analysis, entity extraction, and statistical insights

SPECIALIZATION:
- Entity extraction and recognition using spaCy NER models
- Numerical data analysis and statistics  
- Trend analysis and pattern detection
- Performance metrics and KPIs
- Text complexity and readability analysis

NOTE: Sentiment and bias analysis has been centralized in Scout V2 Agent
Use Scout V2 endpoints for sentiment/bias analysis.
"""

import logging
import os
import json
import re
import statistics
import warnings
from datetime import datetime
from typing import List, Dict, Any
from collections import Counter, defaultdict

# Configure logging with production standards
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("analyst.tools")

# Online Training Integration (deferred)
try:
    from training_system import (
        initialize_online_training, get_training_coordinator,
        add_training_feedback
    )
    initialize_online_training = initialize_online_training
    get_training_coordinator = get_training_coordinator
    add_training_feedback = add_training_feedback
    ONLINE_TRAINING_AVAILABLE = False
except ImportError:
    initialize_online_training = None
    get_training_coordinator = None
    add_training_feedback = None
    ONLINE_TRAINING_AVAILABLE = False
    logger.warning("⚠️ Online Training not available for Analyst V2")

# runtime guard
_online_training_initialized = False

def _ensure_online_training_initialized(update_threshold: int = 35) -> None:
    """Lazily initialize the online training coordinator for Analyst."""
    global _online_training_initialized, ONLINE_TRAINING_AVAILABLE
    if _online_training_initialized:
        return
    if initialize_online_training is None:
        ONLINE_TRAINING_AVAILABLE = False
        _online_training_initialized = True
        return
    try:
        initialize_online_training(update_threshold=update_threshold)
        ONLINE_TRAINING_AVAILABLE = True
        logger.info("🎓 Online Training lazily initialized for Analyst V2")
    except Exception as e:
        ONLINE_TRAINING_AVAILABLE = False
        logger.warning(f"⚠️ Analyst online training failed to initialize at runtime: {e}")
    finally:
        _online_training_initialized = True

# Suppress specific warnings for production deployment
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# Detect availability of specialized NLP dependencies without importing them
import importlib.util

_spacy_spec = importlib.util.find_spec("spacy")
HAS_SPACY = _spacy_spec is not None

# Provide a lightweight helper to call spaCy.explain() when needed without
# importing spaCy at module import time (avoids triggering spacy CLI imports
# and their deprecation warnings during pytest collection).
def _spacy_explain(label: str) -> str:
    try:
        if not HAS_SPACY:
            return label
        import spacy as _spacy_module
        return _spacy_module.explain(label) or label
    except Exception:
        return label

# Import numerical analysis dependencies  
try:
    import numpy as np
    import scipy.stats as stats
    HAS_NUMPY = True
    logger.info("✅ NumPy/SciPy available for statistical analysis")
except ImportError as e:
    logger.warning(f"⚠️ NumPy/SciPy not available: {e}")
    HAS_NUMPY = False
    np = None
    stats = None

_transformers_spec = importlib.util.find_spec("transformers")
HAS_TRANSFORMERS = _transformers_spec is not None
_torch_spec = importlib.util.find_spec("torch")
_TORCH_AVAILABLE = _torch_spec is not None
if HAS_TRANSFORMERS:
    logger.info("✅ Transformers package available for potential NER fallback")
else:
    logger.warning("⚠️ Transformers package not available; NER fallback disabled")

# Lazy import helpers for heavy external libraries
def _import_transformers_pipeline():
    """Import transformers.pipeline and torch lazily and return (pipeline, torch)

    Returns (pipeline_fn, torch_module) or (None, None) on failure.
    """
    try:
        from transformers import pipeline as _pipeline
        import torch as _torch
        return _pipeline, _torch
    except Exception as e:
        logger.warning(f"Could not import transformers pipeline: {e}")
        return None, None

# Configuration and constants
FEEDBACK_LOG = os.environ.get("ANALYST_FEEDBACK_LOG", "feedback_analyst.log")
SPACY_MODEL = "en_core_web_sm"  # English model for NER
NER_FALLBACK_MODEL = "dbmdz/bert-large-cased-finetuned-conll03-english"

# Global model instances for efficiency
_spacy_nlp = None
_ner_pipeline = None

def ensure_spacy_model_loaded():
    """Load spaCy model lazily to avoid import-time downloads during tests."""
    global _spacy_nlp
    if _spacy_nlp is not None:
        return _spacy_nlp
    if not HAS_SPACY:
        return None
    try:
        import importlib
        spacy_module = importlib.import_module('spacy')
        _spacy_nlp = spacy_module.load(SPACY_MODEL)
        logger.info(f"✅ Loaded spaCy model: {SPACY_MODEL}")
        return _spacy_nlp
    except Exception as e:
        logger.warning(f"Could not load {SPACY_MODEL}: {e}")
        return None

def log_feedback(event: str, details: Dict[str, Any]) -> None:
    """
    Production-standard feedback logging with structured format.
    
    Args:
        event: Event type identifier
        details: Event details and metadata
    """
    try:
        timestamp = datetime.utcnow().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "event": event,
            "details": details,
            "agent": "analyst"
        }
        
        with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
            f.write(f"{json.dumps(log_entry)}\n")
            
    except Exception as e:
        logger.error(f"Failed to log feedback: {e}")

def get_spacy_model():
    """
    Load spaCy model with production error handling.
    
    Returns:
        spaCy Language model or None if unavailable
    """
    global _spacy_nlp
    
    if not HAS_SPACY:
        logger.warning("spaCy not available, using fallback NER")
        return None

    return ensure_spacy_model_loaded()

def get_mistral_model():
    """Compatibility shim for tests that expect get_mistral_model to exist.

    Return a (model, tokenizer) tuple or (None, None) when not available.
    Tests monkeypatch this during unit tests, so a lightweight shim prevents
    AttributeError during collection.
    """
    return (None, None)

def get_ner_pipeline():
    """
    Load transformer-based NER pipeline as fallback.
    
    Returns:
        HuggingFace NER pipeline or None if unavailable
    """
    global _ner_pipeline

    if not HAS_TRANSFORMERS:
        logger.warning("Transformers not available for NER fallback")
        return None

    if _ner_pipeline is None:
        pipeline_fn, torch_mod = _import_transformers_pipeline()
        if pipeline_fn is None:
            return None
        try:
            device = 0 if (torch_mod is not None and getattr(torch_mod, 'cuda', None) and torch_mod.cuda.is_available()) else -1
            _ner_pipeline = pipeline_fn(
                "ner",
                model=NER_FALLBACK_MODEL,
                aggregation_strategy="simple",
                device=device
            )
            logger.info(f"✅ Loaded NER pipeline: {NER_FALLBACK_MODEL}")
        except Exception as e:
            logger.error(f"Error loading NER pipeline: {e}")
            return None

    return _ner_pipeline

# =============================================================================
# SPECIALIZED ANALYST FUNCTIONS - PRODUCTION READY
# =============================================================================

def _extract_entities_patterns(text: str) -> List[Dict[str, Any]]:
    """
    Pattern-based entity extraction as last resort fallback.
    
    Args:
        text: Input text
        
    Returns:
        List of entity dictionaries
    """
    entities = []
    
    # Pattern for capitalized names (people, organizations, locations)
    name_pattern = r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b'
    matches = re.finditer(name_pattern, text)
    
    for match in matches:
        entity_text = match.group()
        # Skip common words that match the pattern
        if entity_text.lower() not in {'The', 'This', 'That', 'Then', 'There', 'When', 'Where'}:
            entities.append({
                "text": entity_text,
                "label": "UNKNOWN",
                "start": match.start(),
                "end": match.end(),
                "confidence": 0.5,
                "description": "Pattern-based extraction"
            })
    
    return entities[:20]  # Limit results

def _clean_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Clean and deduplicate entity list.
    
    Args:
        entities: Raw entity list
        
    Returns:
        Cleaned entity list
    """
    if not entities:
        return []
    
    # Remove duplicates by text (case-insensitive)
    seen = set()
    cleaned = []
    
    for entity in entities:
        entity_text = entity["text"].lower().strip()
        if entity_text and entity_text not in seen and len(entity_text) > 1:
            seen.add(entity_text)
            cleaned.append(entity)
    
    # Sort by confidence (highest first)
    cleaned.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    
    return cleaned[:15]  # Limit to top 15 entities

def identify_entities(text: str) -> List[Dict[str, Any]]:
    """
    Advanced entity extraction using spaCy NER with transformer fallback.
    
    Args:
        text: Input text for entity extraction
        
    Returns:
        List of entity dictionaries with type, text, start, end, confidence
    """
    if not text or not text.strip():
        return []
        
    logger.info(f"🔍 Extracting entities from {len(text)} characters")
    
    entities = []
    processing_method = "none"
    
    try:
        # Primary: spaCy NER (fastest, most accurate)
        nlp = get_spacy_model()
        if nlp:
            doc = nlp(text)
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "confidence": 0.9,  # Default confidence for spaCy
                    "description": _spacy_explain(ent.label_) if HAS_SPACY else ent.label_
                })
            processing_method = "spacy"
            
        # Fallback: Transformer NER pipeline
        elif HAS_TRANSFORMERS:
            ner_pipeline = get_ner_pipeline()
            if ner_pipeline:
                # Limit text length for transformer processing
                text_limited = text[:2000] if len(text) > 2000 else text
                ner_results = ner_pipeline(text_limited)
                
                for result in ner_results:
                    entities.append({
                        "text": result["word"],
                        "label": result["entity_group"],
                        "start": result["start"],
                        "end": result["end"], 
                        "confidence": round(result["score"], 3),
                        "description": result["entity_group"]
                    })
                processing_method = "transformer"
                
        # Last resort: Pattern-based extraction
        else:
            entities = _extract_entities_patterns(text)
            processing_method = "patterns"
            
        # Clean and deduplicate entities
        entities = _clean_entities(entities)
        
        logger.info(f"✅ Extracted {len(entities)} entities using {processing_method}")
        
        log_feedback("identify_entities", {
            "text_length": len(text),
            "entities_found": len(entities),
            "method": processing_method,
            "entities": [e["text"] for e in entities[:5]]  # Log first 5
        })
        
        return entities
        
    except Exception as e:
        logger.error(f"❌ Entity extraction failed: {e}")
        log_feedback("identify_entities_error", {
            "text_length": len(text),
            "error": str(e),
            "method": processing_method
        })
        return []

def analyze_text_statistics(text: str) -> Dict[str, Any]:
    """
    Comprehensive text statistical analysis for news content.
    
    Args:
        text: Input text for analysis
        
    Returns:
        Dictionary containing statistical metrics
    """
    if not text or not text.strip():
        return {"error": "Empty text provided"}
        
    logger.info(f"📊 Analyzing text statistics for {len(text)} characters")
    
    try:
        # Basic text metrics
        word_count = len(text.split())
        sentence_count = len([s for s in text.split('.') if s.strip()])
        paragraph_count = len([p for p in text.split('\n\n') if p.strip()])
        
        # Word length analysis
        words = text.split()
        word_lengths = [len(word.strip('.,!?;:"()[]')) for word in words if word.strip()]
        
        # Readability metrics
        avg_word_length = statistics.mean(word_lengths) if word_lengths else 0
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Complexity indicators
        complex_words = [w for w in words if len(w.strip('.,!?;:"()[]')) > 6]
        complex_word_ratio = len(complex_words) / word_count if word_count > 0 else 0
        
        # Number extraction and analysis
        numbers = _extract_numbers(text)
        
        # Calculate readability scores
        readability = _calculate_readability(text, word_count, sentence_count, word_lengths)
        
        result = {
            "basic_metrics": {
                "character_count": len(text),
                "word_count": word_count,
                "sentence_count": sentence_count,
                "paragraph_count": paragraph_count,
                "avg_word_length": round(avg_word_length, 2),
                "avg_sentence_length": round(avg_sentence_length, 2)
            },
            "complexity_metrics": {
                "complex_words": len(complex_words),
                "complex_word_ratio": round(complex_word_ratio, 3),
                "vocabulary_diversity": _calculate_vocabulary_diversity(words),
                "readability_score": readability
            },
            "numerical_content": {
                "numbers_found": len(numbers),
                "numbers": numbers[:10],  # First 10 numbers
                "numeric_density": round(len(numbers) / word_count * 100, 2) if word_count > 0 else 0
            }
        }
        
        logger.info(f"✅ Statistical analysis complete: {word_count} words, readability {readability}")
        
        log_feedback("analyze_text_statistics", {
            "word_count": word_count,
            "readability": readability,
            "complexity_ratio": complex_word_ratio
        })
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Text statistics analysis failed: {e}")
        log_feedback("analyze_text_statistics_error", {"error": str(e)})
        return {"error": str(e)}

def extract_key_metrics(text: str, url: str = None) -> Dict[str, Any]:
    """
    Extract key numerical and statistical metrics from news text.
    
    Args:
        text (str): Article text to analyze
        url (str, optional): Article URL for context
        
    Returns:
        Dict containing extracted metrics including financial data, 
        temporal references, and statistical information
    """
    try:
        logger.info(f"🔍 Extracting key metrics from text (length: {len(text)} chars)")
        
        metrics = {
            "financial_metrics": _extract_financial_metrics(text),
            "temporal_references": _extract_temporal_references(text),
            "statistical_references": _extract_statistical_references(text),
            "geographic_metrics": _extract_geographic_metrics(text),
            "numerical_data": _extract_numbers(text),
            "extraction_metadata": {
                "text_length": len(text),
                "url": url,
                "extraction_time": datetime.now().isoformat(),
                "total_metrics_found": 0  # Will be calculated
            }
        }
        
        # Calculate total metrics found
        total_metrics = (
            len(metrics["financial_metrics"]) +
            len(metrics["temporal_references"]) +
            len(metrics["statistical_references"]) +
            len(metrics["geographic_metrics"]) +
            len(metrics["numerical_data"])
        )
        
        metrics["extraction_metadata"]["total_metrics_found"] = total_metrics
        
        logger.info(f"✅ Extracted {total_metrics} total metrics")
        return metrics
        
    except Exception as e:
        logger.error(f"❌ Error extracting key metrics: {e}")
        return {"error": str(e)}

def analyze_content_trends(texts: List[str], urls: List[str] = None) -> Dict[str, Any]:
    """
    Analyze trends across multiple content pieces.
    
    Args:
        texts (List[str]): List of article texts to analyze
        urls (List[str], optional): Corresponding URLs for context
        
    Returns:
        Dict containing trend analysis including entity trends, 
        topic evolution, and statistical patterns
    """
    try:
        logger.info(f"📈 Analyzing content trends across {len(texts)} texts")
        
        # Filter out empty texts
        valid_texts = [text for text in texts if text and text.strip()]
        if not valid_texts:
            return {"error": "No valid texts provided for trend analysis"}
        
        trends = {
            "entity_trends": _analyze_entity_trends(valid_texts),
            "topic_trends": _analyze_topic_trends(valid_texts),
            "statistical_trends": _analyze_statistical_trends(valid_texts),
            "temporal_patterns": _analyze_temporal_patterns(valid_texts),
            "trend_metadata": {
                "total_texts_analyzed": len(valid_texts),
                "total_characters": sum(len(text) for text in valid_texts),
                "average_text_length": sum(len(text.split()) for text in valid_texts) / len(valid_texts),
                "urls_provided": bool(urls and len(urls) > 0),
                "analysis_time": datetime.now().isoformat()
            }
        }
        
        logger.info(f"✅ Content trends analysis completed for {len(valid_texts)} texts")
        return trends
        
    except Exception as e:
        logger.error(f"❌ Error analyzing content trends: {e}")
        return {"error": str(e)}

# =============================================================================
# HELPER FUNCTIONS FOR SPECIALIZED ANALYSIS
# =============================================================================

def _extract_numbers(text: str) -> List[Dict[str, Any]]:
    """Extract numerical data from text."""
    numbers = []
    
    # Pattern for various number formats
    number_patterns = [
        (r'\b\d{1,3}(?:,\d{3})*\.?\d*\%', 'percentage'),
        (r'\$\d{1,3}(?:,\d{3})*\.?\d*(?:[kmb]illion)?', 'currency'),
        (r'\b\d{1,3}(?:,\d{3})*\.?\d*\b', 'number')
    ]
    
    for pattern, num_type in number_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            numbers.append({
                "value": match.group(),
                "type": num_type,
                "position": match.start()
            })
    
    return numbers

def _calculate_readability(text: str, word_count: int, sentence_count: int, word_lengths: List[int]) -> float:
    """Calculate simplified readability score."""
    if sentence_count == 0 or not word_lengths:
        return 0.0
        
    avg_sentence_length = word_count / sentence_count
    avg_word_length = statistics.mean(word_lengths)
    
    # Simplified readability formula (0-100 scale)
    readability = 100 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length / 100)
    return max(0, min(100, round(readability, 1)))

def _calculate_vocabulary_diversity(words: List[str]) -> float:
    """Calculate type-token ratio for vocabulary diversity."""
    if not words:
        return 0.0
        
    unique_words = set(word.lower().strip('.,!?;:"()[]') for word in words)
    return round(len(unique_words) / len(words), 3) if words else 0.0

def _extract_financial_metrics(text: str) -> List[Dict[str, Any]]:
    """Extract financial metrics from text."""
    financial_patterns = [
        (r'\$\d+(?:,\d+)*(?:\.\d+)?(?:\s*(?:million|billion|trillion))?', 'currency'),
        (r'\d+(?:\.\d+)?%', 'percentage'),
        (r'(?:up|down|increased|decreased)\s+(?:by\s+)?\d+(?:\.\d+)?%', 'change_percentage')
    ]
    
    metrics = []
    for pattern, metric_type in financial_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            metrics.append({
                "value": match.group(),
                "type": metric_type,
                "context": text[max(0, match.start()-20):match.end()+20].strip()
            })
    
    return metrics

def _extract_temporal_references(text: str) -> List[Dict[str, Any]]:
    """Extract temporal references from text."""
    temporal_patterns = [
        (r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}', 'full_date'),
        (r'\b\d{1,2}/\d{1,2}/\d{4}', 'date_slash'),
        (r'\b(?:yesterday|today|tomorrow|last week|next week|this month)', 'relative_time')
    ]
    
    references = []
    for pattern, ref_type in temporal_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            references.append({
                "value": match.group(),
                "type": ref_type,
                "position": match.start()
            })
    
    return references

def _extract_statistical_references(text: str) -> List[Dict[str, Any]]:
    """Extract statistical references from text."""
    stat_patterns = [
        (r'(?:poll|survey|study|research)\s+(?:shows?|indicates?|finds?|suggests?)', 'study_reference'),
        (r'\d+(?:\.\d+)?%\s+of\s+(?:respondents|people|participants)', 'poll_result'),
        (r'according\s+to\s+(?:a\s+)?(?:poll|survey|study)', 'source_reference')
    ]
    
    statistics_refs = []
    for pattern, stat_type in stat_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            statistics_refs.append({
                "value": match.group(),
                "type": stat_type,
                "context": text[max(0, match.start()-30):match.end()+30].strip()
            })
    
    return statistics_refs

def _extract_geographic_metrics(text: str) -> List[Dict[str, Any]]:
    """Extract geographic and demographic metrics."""
    # This would ideally use NER, but for fallback:
    geo_patterns = [
        (r'\b(?:in|from|across)\s+(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', 'location_reference'),
        (r'\d+(?:,\d+)*\s+(?:people|residents|citizens|voters)', 'demographic_count')
    ]
    
    geo_metrics = []
    for pattern, geo_type in geo_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            geo_metrics.append({
                "value": match.group(),
                "type": geo_type,
                "position": match.start()
            })
    
    return geo_metrics[:10]  # Limit results

def _analyze_entity_trends(texts: List[str]) -> Dict[str, Any]:
    """Analyze entity trends across multiple texts."""
    if not HAS_SPACY:
        return {"error": "spaCy not available for entity trend analysis"}
    
    nlp = get_spacy_model()
    if not nlp:
        return {"error": "Could not load spaCy model"}
    
    entity_counts = Counter()
    entity_types = defaultdict(list)
    
    for text in texts:
        if not text:
            continue
            
        doc = nlp(text)
        for ent in doc.ents:
            entity_counts[ent.text] += 1
            entity_types[ent.label_].append(ent.text)
    
    return {
        "most_common_entities": dict(entity_counts.most_common(10)),
        "entity_types": {k: len(set(v)) for k, v in entity_types.items()},
        "total_unique_entities": len(entity_counts)
    }

def _analyze_topic_trends(texts: List[str]) -> Dict[str, Any]:
    """Analyze topic trends using keyword frequency."""
    # Combine all texts
    all_text = " ".join(texts).lower()
    words = re.findall(r'\b[a-z]{4,}\b', all_text)  # Words 4+ characters
    
    # Remove common stop words (simplified list)
    stop_words = {'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 'said', 'would', 'there', 'could', 'more', 'what', 'when', 'where', 'were', 'their', 'than', 'about', 'after', 'before', 'during', 'through'}
    filtered_words = [w for w in words if w not in stop_words]
    
    word_counts = Counter(filtered_words)
    
    return {
        "trending_topics": dict(word_counts.most_common(15)),
        "total_keywords": len(word_counts),
        "vocabulary_size": len(set(filtered_words))
    }

def _analyze_statistical_trends(texts: List[str]) -> Dict[str, Any]:
    """Analyze statistical trends across texts."""
    if not HAS_NUMPY:
        # Fallback without numpy
        return {"error": "NumPy not available for statistical trend analysis"}
    
    text_lengths = [len(text.split()) for text in texts if text]
    
    if not text_lengths:
        return {"error": "No valid texts for analysis"}
    
    return {
        "length_statistics": {
            "mean": float(np.mean(text_lengths)),
            "std": float(np.std(text_lengths)),
            "min": float(np.min(text_lengths)),
            "max": float(np.max(text_lengths)),
            "median": float(np.median(text_lengths))
        },
        "sample_size": len(text_lengths)
    }

def _analyze_temporal_patterns(texts: List[str]) -> Dict[str, Any]:
    """Analyze temporal patterns across texts."""
    temporal_mentions = Counter()
    
    for text in texts:
        if not text:
            continue
            
        # Count temporal references
        temporal_words = ['today', 'yesterday', 'tomorrow', 'week', 'month', 'year', 'recent', 'current', 'future', 'past']
        words = text.lower().split()
        
        for word in words:
            if any(temp in word for temp in temporal_words):
                temporal_mentions[word] += 1
    
    return {
        "temporal_frequency": dict(temporal_mentions.most_common(10)),
        "temporal_density": sum(temporal_mentions.values()) / len(texts) if texts else 0
    }
