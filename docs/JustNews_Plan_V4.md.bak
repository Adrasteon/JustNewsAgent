# JustNews V4: RTX-Enhanced Migration and Implementation Plan

## Reasoning Agent (Nucleoid) Integration

### Purpose and Role
The Reasoning Agent provides symbolic logic, fact validation, contradiction detection, and explainability for news analysis. It is built on the Nucleoid neuro-symbolic AI framework and is fully integrated into the MCP bus architecture. The agent enables:
- Fact ingestion and rule definition for news claims
- Symbolic querying and contradiction detection
- Explainable reasoning for editorial and fact-checking workflows
- Integration with other agents for hybrid neuro-symbolic + neural workflows

### Use Cases
- **Fact Validation**: Ingests facts from Scout, Analyst, or Fact Checker agents and applies logical rules to validate claims.
- **Contradiction Detection**: Detects logical inconsistencies in news articles or between multiple sources.
- **Explainability**: Provides human-readable explanations for why a claim is accepted, rejected, or flagged as contradictory.
- **Editorial Support**: Assists Chief Editor and Critic agents with logic-based recommendations and transparency.

### Technical Details
- **API Endpoints**: `/add_fact`, `/add_facts`, `/add_rule`, `/query`, `/evaluate`, `/health`
- **MCP Bus Integration**: Registers tools and responds to `/call` requests for symbolic reasoning tasks.
- **Port**: 8008 (default)
- **Resource Usage**: <1GB RAM, CPU only (no GPU required)

### Example Workflow
1. Scout Agent extracts a claim from a news article.
2. Fact Checker agent verifies the claim using neural models.
3. Reasoning Agent ingests the claim as a fact and applies logical rules.
4. Contradictions or logical gaps are detected and reported to the Chief Editor.
5. Editorial workflow uses Reasoning Agent's explanations for transparency and auditability.

---

## Overview

This document provides the detailed engineering plan for migrating JustNewsAgentic from V3 to the **Native Specialized Model V4 Architecture**. The migration leverages specialized models for each function with integrated training system for continuous improvement while maintaining zero-downtime deployment.

## 1. Migration Strategy Enhanced with Specialized Models

### Core Principles
- **Specialized Model Architecture**: Task-specific models replace general DialoGPT where appropriate
- **Native Ubuntu Deployment**: Eliminate Docker containerization overhead
- **Training System Integration**: Continuous model improvement through production feedback
- **Performance-First Design**: Achieve superior results through model specialization

### Three-Phase Specialization Approach

```
Phase 1: Native Specialized Foundation (COMPLETE)
├─ Ubuntu native deployment eliminating Docker dependencies
├─ Specialized models replacing general DialoGPT implementations
├─ Training system integration for continuous improvement  
├─ Achieve 4x+ performance improvement through specialization
└─ Validate crash-free operation with professional GPU management

Phase 2: Training System Optimization (ONGOING)
├─ Complete training system integration across all agents
├─ Multi-model training coordination and monitoring
├─ Active learning selection and performance optimization
└─ Specialized model improvement through production feedback

Phase 3: Complete Specialization (FUTURE)
├─ Deploy fully specialized models for all agents
├─ Cross-agent learning and optimization sharing
├─ Achieve complete independence from general models
└─ Domain specialization using comprehensive training data
```

## 2. Phase 1: Native Specialized Foundation (COMPLETE)

### Objectives ✅ ACHIEVED
- Eliminate Docker dependencies for Ubuntu native deployment
- Implement specialized models for task-specific optimization  
- Achieve 4x+ performance improvement through model specialization
- Integrate training system for continuous model improvement
- Establish production-validated performance with crash-free operation

### Specialized Model Architecture

#### 2.1 Agent Specialization Implementation

**Current Specialized Implementations:**

```python
# Synthesizer Agent - Article Generation Specialization
class SynthesizerV2Engine:
    """5-Model Architecture for Comprehensive Content Synthesis"""
    
    def __init__(self):
        # Model 1: BERTopic for advanced topic clustering
        self.models['bertopic'] = BERTopic(embedding_model="all-MiniLM-L6-v2")
        
        # Model 2: BART for neural abstractive summarization  
        self.models['bart'] = BartForConditionalGeneration.from_pretrained(
            "facebook/bart-large-cnn"
        )
        
        # Model 3: T5 for text-to-text generation and neutralization
        self.models['t5'] = T5ForConditionalGeneration.from_pretrained("t5-small")
        
        # Model 4: DialoGPT for conversational refinement (specialized use)
        self.models['dialogpt'] = AutoModelForCausalLM.from_pretrained(
            "microsoft/DialoGPT-medium"
        )
        
        # Model 5: SentenceTransformer for semantic embeddings
        self.models['embeddings'] = SentenceTransformer("all-MiniLM-L6-v2")

# Critic Agent - Quality Assessment Specialization  
class CriticV2Engine:
    """5-Model Architecture for Comprehensive Content Review"""
    
    def __init__(self):
        # Model 1: BERT for content quality scoring
        self.models['bert'] = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=5
        )
        
        # Model 2: RoBERTa for advanced bias detection
        self.models['roberta'] = RobertaForSequenceClassification.from_pretrained(
            "roberta-base", num_labels=3
        )
        
        # Model 3: DeBERTa for factual consistency evaluation
        self.models['deberta'] = DebertaForSequenceClassification.from_pretrained(
            "microsoft/deberta-base", num_labels=3
        )
        
        # Model 4: DistilBERT for readability assessment
        self.models['distilbert'] = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=5
        )
        
        # Model 5: SentenceTransformer for plagiarism detection
        self.models['embeddings'] = SentenceTransformer("all-MiniLM-L6-v2")
```

#### 2.2 Native Ubuntu Deployment

**Environment Setup:**
```bash
# Native Ubuntu deployment - no Docker required
source /home/adra/miniconda3/etc/profile.d/conda.sh
conda activate justnews-production

# Direct agent startup
python agents/synthesizer/main.py  # Port 8006
python agents/critic/main.py       # Port 8007
python agents/reasoning/main.py    # Port 8008
```

**Performance Validation Results:**
```python
# Validated Production Performance
PERFORMANCE_RESULTS = {
    "analyst_agent": {
        "inference_speed": "730+ articles/sec",  # Native TensorRT validated
        "method": "native_tensorrt",
        "memory_usage": "2.3GB GPU",
        "stability": "zero_crashes"
    },
    "scout_agent": {
        "processing_speed": "8.14 articles/sec",  # Ultra-fast processing
        "content_discovery": "production_validated", 
        "model": "LLaMA-3-8B + LLaVA",
        "specialization": "web_content_analysis"
    },
    "training_system": {
        "throughput": "28,800+ articles/hour",
        "update_frequency": "every_45_minutes_per_agent",
        "safety": "automatic_rollback_protection",
        "status": "production_operational"
    }
}
```
#### 2.3 Training System Integration

**Comprehensive Training Architecture:**

```python
# Training System Integration with Specialized Models
from training_system import (
    initialize_online_training,
    get_system_training_manager, 
    collect_prediction,
    submit_correction
)

class SpecializedAgentWithTraining:
    """Base class for agents with integrated training system"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.training_coordinator = get_training_coordinator()
        self.specialized_models = self._load_specialized_models()
        
    def predict_with_training_feedback(self, input_text: str, task: str):
        """Perform prediction with automatic training data collection"""
        
        # Get prediction from specialized model
        prediction = self.specialized_models[task].predict(input_text)
        confidence = self._calculate_confidence(prediction)
        
        # Automatically collect for training system
        collect_prediction(
            agent_name=self.agent_name,
            task_type=task,
            input_text=input_text,
            prediction=prediction,
            confidence=confidence
        )
        
        # Handle low-confidence predictions for active learning
        if confidence < 0.7:  # Uncertainty threshold
            self._add_to_active_learning_buffer(input_text, prediction, confidence)
        
        return prediction
    
    def handle_user_correction(self, input_text: str, incorrect_output: Any, 
                             correct_output: Any, priority: int = 2):
        """Handle user corrections with immediate updates for critical issues"""
        
        submit_correction(
            agent_name=self.agent_name,
            task_type="user_correction",
            input_text=input_text,
            incorrect_output=incorrect_output,
            correct_output=correct_output,
            priority=priority  # 3=immediate update, 2=high priority, 1=normal
        )
```

**Production Training Metrics:**
```python
TRAINING_SYSTEM_STATUS = {
    "coordinator": {
        "implementation": "850+ lines production code",
        "features": ["EWC", "active_learning", "rollback_protection"],
        "status": "operational"
    },
    "performance": {
        "throughput": "28,800+ articles/hour",
        "update_frequency": "every_45_minutes_per_agent",
        "model_updates": "82+ updates/hour_across_all_agents", 
        "safety_threshold": "5%_accuracy_drop_triggers_rollback"
    },
    "integration_status": {
        "scout_agent": "✅ integrated",
        "analyst_agent": "✅ integrated", 
        "critic_agent": "✅ integrated",
        "synthesizer_agent": "⏳ in_progress",
        "fact_checker_agent": "⏳ in_progress"
    }
}
```

#### 2.4 Specialized Model Performance Validation

**Multi-Model Architecture Validation:**

```python
def validate_specialized_models():
    """Comprehensive validation of all specialized model implementations"""
    
    validation_results = {}
    
    # Synthesizer Agent - 5 models for article generation
    synthesizer = SynthesizerV2Engine()
    synthesizer_status = synthesizer.get_model_status()
    validation_results['synthesizer'] = {
        "models_loaded": f"{synthesizer_status['total_models']}/5",
        "capabilities": ["clustering", "summarization", "generation", "refinement", "embeddings"],
        "specialization": "article_synthesis_from_verified_data"
    }
    
    # Critic Agent - 5 models for quality assessment  
    critic = CriticV2Engine()
    critic_status = critic.get_model_status()
    validation_results['critic'] = {
        "models_loaded": f"{critic_status['total_models']}/5", 
        "capabilities": ["quality", "bias_detection", "factual_consistency", "readability", "originality"],
        "specialization": "quality_neutrality_accuracy_assessment"
    }
    
    # Fact Checker Agent - 4 specialized models
    fact_checker = FactCheckerV2Engine()
    fact_checker_info = fact_checker.get_model_info()
    validation_results['fact_checker'] = {
        "models_loaded": f"{len([m for m in fact_checker_info.values() if m['loaded']])}/4",
        "capabilities": ["verification", "credibility", "claims_extraction", "evidence_analysis"],
        "specialization": "multi_model_fact_verification"
    }
    
    return validation_results

# Expected Results:
SPECIALIZATION_VALIDATION = {
    "synthesizer": {"models_loaded": "5/5", "specialization": "complete"},
    "critic": {"models_loaded": "5/5", "specialization": "complete"},
    "fact_checker": {"models_loaded": "4/4", "specialization": "complete"},
    "analyst": {"models_loaded": "native_tensorrt", "performance": "730+_art/sec"},
    "scout": {"models_loaded": "LLaMA-3-8B+LLaVA", "performance": "8.14_art/sec"}
}
```

### Phase 1 Success Criteria - ✅ ACHIEVED

**Performance Benchmarks:**
- ✅ Native deployment achieving 730+ articles/sec (Analyst TensorRT)
- ✅ Specialized models outperforming general alternatives
- ✅ Training system operational with 28,800+ articles/hour capacity
- ✅ Zero crashes with professional GPU memory management
- ✅ Ubuntu native eliminating Docker containerization overhead
- ✅ Multi-model architecture validated across Synthesizer, Critic, Fact Checker

**Specialization Validation:**
- ✅ Synthesizer: 5-model architecture for comprehensive article generation
- ✅ Critic: 5-model architecture for quality/neutrality/accuracy assessment  
- ✅ Fact Checker: 4-model pipeline for verification without inappropriate features
- ✅ Reasoning: Nucleoid symbolic logic with <1GB CPU footprint
- ✅ Scout: LLaVA visual analysis for web content discovery (8GB justified)

---

#### 2.2 RTX-Optimized Hybrid Implementation

**Enhanced `agents/analyst/hybrid_tools_v4.py` with RTX AI Toolkit:**

```python
# NVIDIA RTX AI Toolkit Integration
from nvidia_aim import InferenceManager  # AIM SDK for RTX orchestration
import tensorrt_llm  # TensorRT-LLM for 4x performance
from nvidia_workbench import ModelOptimizer  # Model compression

class RTXOptimizedHybridManager:
    """RTX 3090 optimized inference with Docker fallback."""
    
    def __init__(self):
        # Primary: TensorRT-LLM for RTX 3090 (4x performance)
        self.aim_client = InferenceManager(
            target_device="rtx_3090",
            precision="int4",  # 3x model compression
            optimization_level="max_performance"
        )
        
        # Load RTX-optimized models
        self.bias_model = self.aim_client.load_model(
            "mistral-7b-news-bias",
            backend="tensorrt-llm",
            ampere_optimizations=True  # RTX 3090 SM86 specific
        )
        
        self.sentiment_model = self.aim_client.load_model(
            "mistral-7b-sentiment",
            backend="tensorrt-llm",
            quantization="int4"  # Fits in 24GB VRAM
        )
        
        # Fallback: Docker Model Runner for stability
        self.docker_client = DockerModelClient("ai/mistral")
        self.performance_monitor = RTXPerformanceMonitor()
    
    def query_with_rtx_optimization(self, prompt: str, task: str) -> Tuple[str, str, Dict]:
        """RTX-first inference with intelligent fallback."""
        start_time = time.time()
        
        try:
            # Try RTX TensorRT-LLM first (4x faster)
            if task == "bias":
                response = self.bias_model.generate(prompt, max_tokens=10)
            elif task == "sentiment":
                response = self.sentiment_model.generate(prompt, max_tokens=10)
            else:
                response = self.aim_client.generate(prompt, max_tokens=50)
            
            elapsed = time.time() - start_time
            metrics = {
                "source": "tensorrt-llm",
                "elapsed_time": elapsed,
                "tokens_per_second": len(response) / elapsed,
                "gpu_memory_used": self.aim_client.get_memory_usage(),
                "performance_gain": "4x_optimized"
            }
            
            return response, "tensorrt-llm", metrics
            
        except Exception as e:
            # Fallback to Docker Model Runner
            logger.warning(f"RTX inference failed, using Docker fallback: {e}")
            response = self.docker_client.query_model(prompt)
            
            elapsed = time.time() - start_time
            metrics = {
                "source": "docker-fallback",
                "elapsed_time": elapsed,
                "fallback_reason": str(e),
                "performance_impact": "baseline"
            }
            
            return response, "docker-fallback", metrics

class RTXPerformanceMonitor:
    """Monitor RTX 3090 performance and optimization effectiveness."""
    
    def log_rtx_metrics(self, operation: str, metrics: Dict):
        """Log RTX-specific performance data."""
        enhanced_metrics = {
            **metrics,
            "gpu_architecture": "ampere_sm86",
            "target_hardware": "rtx_3090",
            "tensorrt_version": tensorrt.__version__,
            "optimization_backend": "tensorrt-llm"
        }
        
            f.write(f"{datetime.utcnow().isoformat()}\t{operation}\t{json.dumps(enhanced_metrics)}\n")
```

#### 2.3 RTX Performance Validation and Benchmarking

**Phase 1 Success Criteria - RTX 3090 Optimization:**

```python
# RTX Performance Validation Suite
def validate_rtx_optimization():
    """Comprehensive RTX 3090 performance validation."""
    
    test_cases = [
        {"task": "bias_scoring", "text": "This is a test news article about politics."},
        {"task": "sentiment_analysis", "text": "Breaking news creates market uncertainty."},
        {"task": "entity_extraction", "text": "President Biden met with CEO Elon Musk yesterday."}
    ]
    
    results = {
        "rtx_performance": {},
        "docker_baseline": {},
        "performance_gains": {}
    }
    
    manager = RTXOptimizedHybridManager()
    
    for test in test_cases:
        # RTX TensorRT-LLM Performance
        rtx_start = time.time()
        rtx_response, rtx_source, rtx_metrics = manager.query_with_rtx_optimization(
            test["text"], test["task"]
        )
        rtx_elapsed = time.time() - rtx_start
        
        # Docker Baseline Performance  
        docker_start = time.time()
        docker_response = manager.docker_client.query_model(test["text"])
        docker_elapsed = time.time() - docker_start
        
        # Calculate Performance Gains
        performance_multiplier = docker_elapsed / rtx_elapsed if rtx_elapsed > 0 else 0
        memory_efficiency = rtx_metrics.get("gpu_memory_used", 0)
        
        results["rtx_performance"][test["task"]] = {
            "elapsed_time": rtx_elapsed,
            "tokens_per_second": rtx_metrics.get("tokens_per_second", 0),
            "gpu_memory_mb": memory_efficiency
        }
        
        results["docker_baseline"][test["task"]] = {
            "elapsed_time": docker_elapsed,
            "subprocess_overhead": True
        }
        
        results["performance_gains"][test["task"]] = {
            "speed_multiplier": f"{performance_multiplier:.1f}x faster",
            "target_achieved": performance_multiplier >= 4.0,
            "memory_optimized": memory_efficiency < 6000  # MB for INT4 quantization
        }
    
    return results

# Expected RTX 3090 Results:
EXPECTED_RTX_PERFORMANCE = {
    "bias_scoring": {"target_time": "<0.5s", "baseline_time": "2.0s", "gain": "4x"},
    "sentiment_analysis": {"target_time": "<0.5s", "baseline_time": "2.0s", "gain": "4x"},
    "entity_extraction": {"target_time": "<1.0s", "baseline_time": "4.0s", "gain": "4x"},
    "memory_usage": {"int4_quantized": "<6GB", "baseline": "12GB+", "compression": "3x"},
    "stability": {"crash_rate": "0%", "fallback_rate": "<5%", "uptime": "99.9%"}
}
```

**RTX Validation Checklist:**
- [ ] TensorRT-LLM achieves 4x performance improvement on RTX 3090
- [ ] INT4 quantization provides 3x model compression with <2% accuracy loss
- [ ] AIM SDK successfully orchestrates between RTX and Docker backends
- [ ] Zero system crashes with professional GPU memory management
- [ ] Docker Model Runner provides reliable fallback when RTX unavailable
- [ ] Performance metrics logging captures RTX-specific data
- [ ] Memory usage stays within RTX 3090 24GB VRAM limits
            api_key="not-needed"  # Docker Model Runner doesn't require API key
        )
    
    def generate_with_fallback(self, prompt: str, model_preference: str = None) -> str:
        """Generate text with automatic fallback to backup models."""
        models_to_try = [
            model_preference or self.primary_model,
            self.fallback_model,
            self.backup_model
        ]
        
        for model in models_to_try:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=512,
                    temperature=0.1
                )
                
                # Log successful inference
                self.log_inference_success(model, prompt[:100])
                return response.choices[0].message.content
                
            except Exception as e:
                logger.warning(f"Model {model} failed: {e}")
                self.log_inference_failure(model, str(e))
                continue
        
        raise RuntimeError("All models failed for inference")
    
    def log_inference_success(self, model: str, prompt_preview: str):
        """Log successful inference for performance tracking."""
        log_feedback("inference_success", {
            "model": model,
            "prompt_preview": prompt_preview,
            "timestamp": datetime.utcnow().isoformat(),
            "inference_type": "docker_model_runner"
        })
    
    def log_inference_failure(self, model: str, error: str):
        """Log failed inference for debugging."""
        log_feedback("inference_failure", {
            "model": model,
            "error": error,
            "timestamp": datetime.utcnow().isoformat(),
            "inference_type": "docker_model_runner"
        })

# Global hybrid client
_hybrid_client = None

def get_hybrid_client():
    """Get or create hybrid inference client."""
    global _hybrid_client
    if _hybrid_client is None:
        _hybrid_client = HybridInferenceClient()
    return _hybrid_client
```

**Enhanced Feedback Collection:**

```python
def log_feedback_enhanced(event: str, details: dict, performance_metrics: dict = None):
    """Enhanced feedback logging with performance metrics and training preparation."""
    
    feedback_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "event": event,
        "details": details,
        "performance_metrics": performance_metrics or {},
        "training_metadata": {
            "suitable_for_training": is_suitable_for_training(event, details),
            "quality_score": calculate_quality_score(details),
            "data_category": categorize_training_data(event, details)
        }
    }
    
    # Log to file (existing functionality)
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(feedback_entry) + "\n")
    
    # Log to database (new functionality)
    if os.environ.get("FEEDBACK_DATABASE") == "enabled":
        store_feedback_in_database(feedback_entry)

def is_suitable_for_training(event: str, details: dict) -> bool:
    """Determine if this feedback is suitable for training data."""
    training_events = ["score_bias", "score_sentiment", "identify_entities"]
    has_valid_output = details.get("score") is not None or details.get("entities")
    return event in training_events and has_valid_output

def calculate_quality_score(details: dict) -> float:
    """Calculate quality score for training data prioritization."""
    score = 1.0
    
    # Reduce score for errors
    if "error" in details:
        score *= 0.1
    
    # Increase score for successful complex operations
    if details.get("entities") and len(details["entities"]) > 2:
        score *= 1.2
    
    # Increase score for clear bias/sentiment indicators
    if details.get("score") and abs(details["score"]) > 0.7:
        score *= 1.1
    
    return min(score, 1.0)

def categorize_training_data(event: str, details: dict) -> str:
    """Categorize feedback for specialized model training."""
    categories = {
        "score_bias": "bias_detection",
        "score_sentiment": "sentiment_analysis", 
        "identify_entities": "entity_recognition"
    }
    return categories.get(event, "general")
```

#### 2.3 Updated Analysis Functions

**Bias Scoring with Docker Model Runner:**

```python
def score_bias(text: str) -> float:
    """Score political bias using Docker Model Runner with enhanced feedback."""
    
    start_time = time.time()
    client = get_hybrid_client()
    
    prompt = f"""Analyze the political bias in this news text. Respond with only a number between -1.0 (left bias) and +1.0 (right bias), where 0.0 is neutral.

Text: {text[:2000]}

Bias score:"""
    
    try:
        result = client.generate_with_fallback(prompt, model_preference="ai/mistral:7b-instruct-v0.3")
        inference_time = time.time() - start_time
        
        # Extract score from response
        score_match = re.search(r'-?\d*\.?\d+', result.strip())
        if score_match:
            score = float(score_match.group())
            score = max(-1.0, min(1.0, score))  # Clamp to valid range
        else:
            logger.warning(f"Could not parse bias score from: {result}")
            score = 0.0
        
        # Enhanced feedback logging
        performance_metrics = {
            "inference_time": inference_time,
            "model_used": client.primary_model,
            "text_length": len(text),
            "confidence": abs(score)  # Higher absolute values indicate higher confidence
        }
        
        log_feedback_enhanced("score_bias", {
            "text": text[:200], 
            "score": score, 
            "raw_response": result,
            "method": "docker_model_runner"
        }, performance_metrics)
        
        return score
        
    except Exception as e:
        error_time = time.time() - start_time
        log_feedback_enhanced("score_bias_error", {
            "text": text[:200], 
            "error": str(e),
            "method": "docker_model_runner",
            "error_time": error_time
        })
        logger.error(f"Error in score_bias: {e}")
        return 0.0  # Neutral fallback
```

#### 2.4 Database Schema Updates

**New Training Data Tables:**

```sql
-- Enhanced training examples table
CREATE TABLE IF NOT EXISTS training_examples_v4 (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Source information
    agent_name VARCHAR(50) NOT NULL,
    tool_name VARCHAR(50) NOT NULL,
    data_category VARCHAR(50) NOT NULL,
    
    -- Input/Output data
    input_text TEXT NOT NULL,
    expected_output JSONB NOT NULL,
    actual_output JSONB,
    
    -- Quality metrics
    quality_score FLOAT DEFAULT 0.0,
    training_priority INTEGER DEFAULT 0,
    
    -- Performance data
    inference_time FLOAT,
    model_used VARCHAR(100),
    
    -- Training metadata
    suitable_for_training BOOLEAN DEFAULT FALSE,
    used_in_training BOOLEAN DEFAULT FALSE,
    training_iteration INTEGER
);

-- Performance metrics table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    agent_name VARCHAR(50) NOT NULL,
    tool_name VARCHAR(50) NOT NULL,
    model_used VARCHAR(100),
    
    inference_time FLOAT,
    success_rate FLOAT,
    error_rate FLOAT,
    
    -- Performance benchmarks
    accuracy_score FLOAT,
    precision_score FLOAT,
    recall_score FLOAT,
    
    -- Metadata
    sample_size INTEGER,
    benchmark_version VARCHAR(20)
);

-- Model evolution tracking
CREATE TABLE IF NOT EXISTS model_versions (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50), -- 'docker_runner' or 'custom'
    version VARCHAR(50) NOT NULL,
    
    -- Performance metrics
    benchmark_score FLOAT,
    training_examples_count INTEGER,
    
    -- Deployment status
    status VARCHAR(20) DEFAULT 'training', -- 'training', 'testing', 'deployed', 'deprecated'
    deployment_percentage FLOAT DEFAULT 0,
    
    -- Configuration
    model_config JSONB,
    training_config JSONB
);
```

### Phase 1 Validation Tasks

#### 2.5 Testing and Validation

**Performance Benchmarking:**

```python
# tests/test_phase1_migration.py
import pytest
import time
from agents.analyst.tools import score_bias, score_sentiment, identify_entities

class TestPhase1Migration:
    """Test suite for Phase 1 Docker Model Runner migration."""
    
    def test_docker_model_runner_connectivity(self):
        """Verify Docker Model Runner is accessible."""
        from agents.analyst.tools import get_hybrid_client
        
        client = get_hybrid_client()
        response = client.generate_with_fallback("Hello, are you working?")
        
        assert response is not None
        assert len(response) > 0
        assert "error" not in response.lower()
    
    def test_bias_scoring_performance(self):
        """Test bias scoring performance meets requirements."""
        test_text = """The new tax policy announced yesterday has sparked heated debate among politicians. Progressive lawmakers argue it unfairly benefits the wealthy, while conservative representatives claim it will stimulate economic growth."""
        
        start_time = time.time()
        score = score_bias(test_text)
        inference_time = time.time() - start_time
        
        # Performance requirements
        assert inference_time < 5.0  # Under 5 seconds
        assert -1.0 <= score <= 1.0   # Valid range
        
        # Should detect some bias in politically charged text
        assert abs(score) > 0.1
    
    def test_fallback_mechanism(self):
        """Test that fallback models work when primary fails."""
        from agents.analyst.tools import get_hybrid_client
        
        client = get_hybrid_client()
        
        # Simulate primary model failure by using non-existent model
        original_primary = client.primary_model
        client.primary_model = "ai/nonexistent:model"
        
        try:
            response = client.generate_with_fallback("Test fallback mechanism")
            assert response is not None
            assert len(response) > 0
        finally:
            client.primary_model = original_primary
    
    def test_feedback_collection_enhancement(self):
        """Verify enhanced feedback collection is working."""
        from agents.analyst.tools import log_feedback_enhanced
        
        test_details = {
            "text": "Test news article",
            "score": 0.7,
            "model_used": "ai/mistral:7b-instruct-v0.3"
        }
        
        test_metrics = {
            "inference_time": 1.5,
            "confidence": 0.7
        }
        
        # Should not raise exceptions
        log_feedback_enhanced("score_bias", test_details, test_metrics)
        
        # Verify feedback log file exists and has new entry
        import os
        assert os.path.exists(FEEDBACK_LOG)
```

**Migration Validation Script:**

```bash
#!/bin/bash
# validate_phase1.sh

echo "=== Phase 1 Migration Validation ==="

# 1. Check Docker Model Runner status
echo "Checking Docker Model Runner..."
docker model status || exit 1

# 2. Pull required models
echo "Pulling required models..."
docker model pull ai/mistral:7b-instruct-v0.3
docker model pull ai/llama3.2:7b-instruct
docker model pull ai/gemma3:7b-instruct

# 3. Test model connectivity
echo "Testing model connectivity..."
docker model run ai/mistral:7b-instruct-v0.3 "Hello, respond with 'OK' if you're working."

# 4. Start services
echo "Starting JustNews services..."
docker-compose up -d

# 5. Wait for services to be ready
echo "Waiting for services to initialize..."
sleep 30

# 6. Run integration tests
echo "Running integration tests..."
python -m pytest tests/test_phase1_migration.py -v

# 7. Performance benchmark
echo "Running performance benchmarks..."
python tests/benchmark_phase1.py

echo "=== Phase 1 Validation Complete ==="
```

### Phase 1 Success Criteria

- ✅ All Docker Model Runner models accessible and responsive
- ✅ Zero model corruption incidents
- ✅ Average inference time < 3 seconds
- ✅ 99%+ inference success rate with fallback
- ✅ Enhanced feedback collection operational
- ✅ All existing functionality preserved
- ✅ Performance meets or exceeds V3 baseline

---

## 3. Phase 2: AI Workbench Training Pipeline (Weeks 3-6)

### Objectives
- Setup NVIDIA AI Workbench for RTX-optimized model training
- Implement QLoRA fine-tuning for domain-specific news analysis
- Create TensorRT Model Optimizer integration for compression
- Train first generation RTX-optimized custom models with 4x performance

### RTX AI Workbench Setup

#### 3.1 AI Workbench Project Configuration

**JustNews RTX Training Project Setup:**

```yaml
# .workbench/project.yaml - AI Workbench Project Configuration
apiVersion: workbench/v1
kind: Project
metadata:
  name: justnews-v4-rtx-training
  description: "RTX 3090 optimized news analysis model training"
  
spec:
  base_image: "nvidia/tensorrt-llm:24.12-py3"
  
  environment:
    variables:
      - CUDA_VISIBLE_DEVICES: "0"  # RTX 3090
      - NVIDIA_VISIBLE_DEVICES: "0"
      - TENSORRT_VERSION: "10.11"
      - TARGET_GPU: "rtx_3090"
      - GPU_ARCHITECTURE: "ampere_sm86"
  
  resources:
    gpu:
      count: 1
      memory: "24GB"  # RTX 3090 VRAM
      architecture: "ampere"
  
  training:
    technique: "qlora"  # Parameter Efficient Fine-Tuning
    framework: "transformers"
    optimization:
      quantization: "int4"
      tensorrt_optimization: true
      ampere_specific: true
  
  deployment:
    target_backend: "tensorrt-llm"
    optimization_level: "max_performance"
    fallback_backend: "docker-model-runner"
```

**AI Workbench Training Notebook (`training/rtx_news_training.ipynb`):**

```python
# RTX-Optimized News Analysis Model Training
import os
from nvidia_workbench import TrainingManager, ModelOptimizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import tensorrt_llm
from datasets import Dataset
import json

# Initialize RTX-specific training environment
training_manager = TrainingManager(
    target_gpu="rtx_3090",
    precision="mixed",  # FP16 + INT4 quantization
    memory_optimization=True
)

# Load and prepare feedback data for training
def prepare_training_data():
    """Convert JustNews feedback logs to training dataset."""
    
    # Load feedback logs
    feedback_data = []
    with open("feedback_analyst.log", "r") as f:
        for line in f:
            if "score_bias" in line or "score_sentiment" in line:
                feedback_data.append(json.loads(line.split("\t")[2]))
    
    # Convert to training format
    training_examples = []
    for feedback in feedback_data:
        if feedback.get("success") and feedback.get("raw_response"):
            training_examples.append({
                "input": feedback["text"],
                "output": feedback["raw_response"],
                "task": feedback.get("task", "analysis"),
                "quality_score": feedback.get("quality_score", 1.0)
            })
    
    return Dataset.from_list(training_examples)

# QLoRA Training Configuration for RTX 3090
training_config = {
    "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
    "dataset": prepare_training_data(),
    "max_seq_length": 2048,
    "learning_rate": 1e-4,
    "batch_size": 4,  # Optimized for RTX 3090 24GB
    "gradient_accumulation_steps": 4,
    "num_epochs": 3,
    "lora_config": {
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj", "o_proj", "gate_proj"],
        "lora_dropout": 0.1
    },
    "quantization": {
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_use_double_quant": True
    }
}

# Train RTX-optimized models
def train_rtx_optimized_models():
    """Train domain-specific models optimized for RTX 3090."""
    
    models_to_train = [
        {"name": "justnews-bias-analyzer", "task": "bias_scoring"},
        {"name": "justnews-sentiment-analyzer", "task": "sentiment_analysis"},
        {"name": "justnews-entity-extractor", "task": "entity_extraction"}
    ]
    
    for model_spec in models_to_train:
        print(f"Training {model_spec['name']} for RTX 3090...")
        
        # Filter training data by task
        task_data = training_config["dataset"].filter(
            lambda x: x["task"] == model_spec["task"]
        )
        
        # Train with QLoRA
        trained_model = training_manager.train_qlora(
            base_model=training_config["model_name"],
            dataset=task_data,
            **training_config
        )
        
        # Optimize with TensorRT Model Optimizer
        optimizer = ModelOptimizer(target_gpu="rtx_3090")
        optimized_model = optimizer.optimize(
            model=trained_model,
            techniques=["quantization", "pruning", "tensorrt_conversion"],
            target_precision="int4",
            performance_target="4x_speedup"
        )
        
        # Save RTX-optimized model
        optimized_model.save(f"./models/{model_spec['name']}-rtx-optimized")
        
        print(f"✅ {model_spec['name']} training complete with RTX optimization")

# Execute training
if __name__ == "__main__":
    train_rtx_optimized_models()
```

#### 3.2 RTX Training Performance Expectations

**Expected AI Workbench Training Results on RTX 3090:**

```python
# RTX 3090 Training Performance Benchmarks
RTX_TRAINING_BENCHMARKS = {
    "qlora_training": {
        "mistral_7b": {
            "training_time": "2-4 hours",  # vs 12+ hours on CPU
            "memory_usage": "18GB VRAM",  # INT4 quantization
            "throughput": "~500 tokens/sec",
            "efficiency_gain": "6x faster than CPU"
        }
    },
    
    "tensorrt_optimization": {
        "model_compression": "3x smaller",  # 14GB -> 4.7GB
        "inference_speedup": "4x faster",
        "optimization_time": "15-30 minutes",
        "accuracy_retention": ">98%"
    },
    
    "deployment_readiness": {
        "rtx_native_format": "TensorRT-LLM engine",
        "fallback_compatibility": "Docker Model Runner",
        "cross_gpu_portable": "Ampere+ architectures",
        "production_ready": "Enterprise-grade stability"
    }
}
```

**Phase 2 AI Workbench Success Criteria:**
- [ ] QLoRA training completes successfully on RTX 3090 in <4 hours
- [ ] TensorRT Model Optimizer achieves 3x compression with <2% accuracy loss
- [ ] Custom models demonstrate domain-specific improvements over base models
- [ ] RTX-optimized models integrate seamlessly with hybrid inference system
- [ ] Training pipeline produces reproducible results with version control
- [ ] A/B testing framework validates custom model performance gains

#### 3.3 RTX Model Deployment Integration

**Enhanced Hybrid Manager with Custom RTX Models:**

```python
class RTXCustomModelManager(RTXOptimizedHybridManager):
    """Extended hybrid manager with custom RTX-trained models."""
    
    def __init__(self):
        super().__init__()
        # Load custom RTX-optimized models
        self.custom_models = self._load_custom_rtx_models()
        self.model_selector = RTXModelSelector()
    
    def _load_custom_rtx_models(self):
        """Load AI Workbench trained RTX-optimized models."""
        custom_models = {}
        
        model_paths = {
            "bias": "./models/justnews-bias-analyzer-rtx-optimized",
            "sentiment": "./models/justnews-sentiment-analyzer-rtx-optimized", 
            "entities": "./models/justnews-entity-extractor-rtx-optimized"
        }
        
        for task, path in model_paths.items():
            if os.path.exists(path):
                custom_models[task] = self.aim_client.load_model(
                    path,
                    backend="tensorrt-llm",
                    custom_optimized=True,
                    performance_profile="rtx_3090_max"
                )
                logger.info(f"✅ Loaded custom RTX model for {task}")
            else:
                logger.info(f"⏳ Custom model for {task} not yet trained")
        
        return custom_models
    
    def query_with_custom_models(self, prompt: str, task: str) -> Tuple[str, str, Dict]:
        """Use custom RTX models when available, fallback to base models."""
        
        # Try custom RTX-optimized model first
        if task in self.custom_models:
            try:
                response = self.custom_models[task].generate(
                    prompt, 
                    max_tokens=50,
                    rtx_optimized=True
                )
                
                metrics = {
                    "source": f"custom-rtx-{task}",
                    "model_type": "domain_specialized",
                    "performance_tier": "5x_optimized"  # Custom + RTX optimization
                }
                
                return response, f"custom-rtx-{task}", metrics
                
            except Exception as e:
                logger.warning(f"Custom RTX model failed for {task}: {e}")
        
        # Fallback to standard RTX optimization
        return super().query_with_rtx_optimization(prompt, task)
```
    volumes:
      - ./feedback_logs:/feedback_logs
      - ./training/data:/processed_data
    depends_on:
      - db

  # Redis for training queue
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

#### 3.2 Training Data Pipeline

**Training Data Processor:**

```python
# training/data_processor/processor.py
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import sqlite3
import psycopg2

class TrainingDataProcessor:
    """Processes feedback logs into training datasets."""
    
    def __init__(self, db_url: str, feedback_log_path: str):
        self.db_url = db_url
        self.feedback_log_path = feedback_log_path
        
    def extract_training_examples(self, days_back: int = 7) -> Dict[str, List]:
        """Extract training examples from recent feedback logs."""
        
        # Get cutoff date
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        training_data = {
            "bias_detection": [],
            "sentiment_analysis": [],
            "entity_recognition": []
        }
        
        # Process database feedback
        db_examples = self._extract_from_database(cutoff_date)
        
        # Process file-based feedback
        file_examples = self._extract_from_files(cutoff_date)
        
        # Combine and categorize
        all_examples = db_examples + file_examples
        
        for example in all_examples:
            category = example.get("training_metadata", {}).get("data_category", "general")
            if category in training_data:
                training_data[category].append(example)
        
        return training_data
    
    def _extract_from_database(self, cutoff_date: datetime) -> List[Dict]:
        """Extract training examples from database."""
        conn = psycopg2.connect(self.db_url)
        
        query = """
        SELECT * FROM training_examples_v4 
        WHERE created_at >= %s 
        AND suitable_for_training = TRUE 
        AND quality_score > 0.5
        ORDER BY quality_score DESC, created_at DESC
        """
        
        df = pd.read_sql(query, conn, params=[cutoff_date])
        conn.close()
        
        return df.to_dict('records')
    
    def _extract_from_files(self, cutoff_date: datetime) -> List[Dict]:
        """Extract training examples from feedback log files."""
        examples = []
        
        try:
            with open(self.feedback_log_path, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        entry_date = datetime.fromisoformat(entry.get("timestamp", ""))
                        
                        if entry_date >= cutoff_date:
                            if entry.get("training_metadata", {}).get("suitable_for_training", False):
                                examples.append(entry)
                    except (json.JSONDecodeError, ValueError):
                        continue
        except FileNotFoundError:
            pass
        
        return examples
    
    def prepare_bias_training_data(self, examples: List[Dict]) -> Tuple[List[str], List[float]]:
        """Prepare bias detection training data."""
        texts = []
        scores = []
        
        for example in examples:
            details = example.get("details", {})
            if "text" in details and "score" in details:
                texts.append(details["text"])
                scores.append(float(details["score"]))
        
        return texts, scores
    
    def prepare_sentiment_training_data(self, examples: List[Dict]) -> Tuple[List[str], List[float]]:
        """Prepare sentiment analysis training data."""
        texts = []
        scores = []
        
        for example in examples:
            details = example.get("details", {})
            if "text" in details and "score" in details:
                texts.append(details["text"])
                scores.append(float(details["score"]))
        
        return texts, scores
    
    def prepare_entity_training_data(self, examples: List[Dict]) -> Tuple[List[str], List[List[Dict]]]:
        """Prepare entity recognition training data."""
        texts = []
        entities = []
        
        for example in examples:
            details = example.get("details", {})
            if "text" in details and "entities" in details:
                texts.append(details["text"])
                entities.append(details["entities"])
        
        return texts, entities
```

#### 3.3 Custom Model Training

**Fine-tuning Pipeline:**

```python
# training/coordinator/trainer.py
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import wandb
import os

class NewsAnalysisTrainer:
    """Custom model trainer for news analysis tasks."""
    
    def __init__(self, base_model: str = "mistralai/Mistral-7B-Instruct-v0.3"):
        self.base_model = base_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize Weights & Biases for experiment tracking
        wandb.init(project="justnews-v4-training")
    
    def train_bias_detector(self, texts: List[str], scores: List[float], 
                          output_dir: str = "./models/bias_detector_v1"):
        """Train a custom bias detection model."""
        
        # Prepare tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model,
            num_labels=1,  # Regression for bias score
            problem_type="regression"
        )
        
        # Prepare dataset
        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)
        
        dataset = Dataset.from_dict({"text": texts, "labels": scores})
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Split dataset
        train_size = int(0.8 * len(tokenized_dataset))
        train_dataset = tokenized_dataset.select(range(train_size))
        eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=50,
            save_steps=100,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="wandb"
        )
        
        # Custom metrics
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = predictions.reshape(-1)
            
            # Calculate regression metrics
            mse = torch.nn.functional.mse_loss(torch.tensor(predictions), torch.tensor(labels))
            mae = torch.nn.functional.l1_loss(torch.tensor(predictions), torch.tensor(labels))
            
            return {
                "mse": mse.item(),
                "mae": mae.item(),
                "rmse": torch.sqrt(mse).item()
            }
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics
        )
        
        # Train model
        trainer.train()
        
        # Save model
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        # Log final metrics
        eval_results = trainer.evaluate()
        wandb.log({"final_eval": eval_results})
        
        return output_dir
    
    def train_sentiment_analyzer(self, texts: List[str], scores: List[float],
                                output_dir: str = "./models/sentiment_analyzer_v1"):
        """Train a custom sentiment analysis model."""
        # Similar to bias detector but optimized for sentiment
        # Implementation details similar to train_bias_detector
        pass
    
    def train_entity_recognizer(self, texts: List[str], entities: List[List[Dict]],
                               output_dir: str = "./models/entity_recognizer_v1"):
        """Train a custom entity recognition model."""
        # Implementation for NER training using token classification
        pass
```

#### 3.4 A/B Testing Framework

**Model Comparison System:**

```python
# training/ab_testing/comparator.py
import random
import json
import time
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ModelPerformance:
    model_id: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    avg_inference_time: float
    error_rate: float
    user_satisfaction: float

class ABTestingFramework:
    """A/B testing framework for comparing Docker models vs custom models."""
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.test_configs = {}
        
    def create_ab_test(self, test_name: str, model_a: str, model_b: str, 
                      traffic_split: float = 0.5, duration_days: int = 7):
        """Create a new A/B test configuration."""
        
        test_config = {
            "test_name": test_name,
            "model_a": model_a,  # e.g., "docker:ai/mistral:7b-instruct-v0.3"
            "model_b": model_b,  # e.g., "custom:bias_detector_v1"
            "traffic_split": traffic_split,
            "start_date": datetime.utcnow(),
            "duration_days": duration_days,
            "active": True,
            "results": {
                "model_a": {"requests": 0, "successes": 0, "total_time": 0.0},
                "model_b": {"requests": 0, "successes": 0, "total_time": 0.0}
            }
        }
        
        self.test_configs[test_name] = test_config
        self._save_test_config(test_config)
        
    def route_request(self, test_name: str, request_data: Dict) -> str:
        """Route request to appropriate model based on A/B test configuration."""
        
        if test_name not in self.test_configs:
            raise ValueError(f"Test {test_name} not found")
        
        config = self.test_configs[test_name]
        
        if not config["active"]:
            return config["model_a"]  # Default to model A if test inactive
        
        # Route based on traffic split
        if random.random() < config["traffic_split"]:
            return config["model_a"]
        else:
            return config["model_b"]
    
    def record_result(self, test_name: str, model_id: str, success: bool, 
                     inference_time: float, user_feedback: Dict = None):
        """Record the result of a model inference for A/B testing."""
        
        if test_name not in self.test_configs:
            return
        
        config = self.test_configs[test_name]
        model_key = "model_a" if model_id == config["model_a"] else "model_b"
        
        config["results"][model_key]["requests"] += 1
        if success:
            config["results"][model_key]["successes"] += 1
        config["results"][model_key]["total_time"] += inference_time
        
        # Store detailed result in database
        self._store_ab_result(test_name, model_id, success, inference_time, user_feedback)
        
    def analyze_test(self, test_name: str) -> Dict[str, Any]:
        """Analyze A/B test results and determine winner."""
        
        if test_name not in self.test_configs:
            raise ValueError(f"Test {test_name} not found")
        
        config = self.test_configs[test_name]
        results = config["results"]
        
        # Calculate metrics for each model
        model_a_metrics = self._calculate_metrics(results["model_a"])
        model_b_metrics = self._calculate_metrics(results["model_b"])
        
        # Determine statistical significance (simplified)
        significance = self._calculate_significance(model_a_metrics, model_b_metrics)
        
        # Determine winner
        winner = self._determine_winner(model_a_metrics, model_b_metrics, significance)
        
        analysis = {
            "test_name": test_name,
            "model_a": {
                "id": config["model_a"],
                "metrics": model_a_metrics
            },
            "model_b": {
                "id": config["model_b"], 
                "metrics": model_b_metrics
            },
            "statistical_significance": significance,
            "winner": winner,
            "recommendation": self._generate_recommendation(winner, significance)
        }
        
        return analysis
    
    def _calculate_metrics(self, results: Dict) -> Dict[str, float]:
        """Calculate performance metrics from raw results."""
        requests = results["requests"]
        successes = results["successes"]
        total_time = results["total_time"]
        
        if requests == 0:
            return {"success_rate": 0.0, "avg_inference_time": 0.0, "error_rate": 1.0}
        
        return {
            "success_rate": successes / requests,
            "avg_inference_time": total_time / requests,
            "error_rate": (requests - successes) / requests
        }
    
    def _determine_winner(self, metrics_a: Dict, metrics_b: Dict, significance: float) -> str:
        """Determine the winning model based on multiple criteria."""
        
        # Weighted scoring system
        score_a = (
            metrics_a["success_rate"] * 0.4 +
            (1 / max(metrics_a["avg_inference_time"], 0.1)) * 0.3 +
            (1 - metrics_a["error_rate"]) * 0.3
        )
        
        score_b = (
            metrics_b["success_rate"] * 0.4 +
            (1 / max(metrics_b["avg_inference_time"], 0.1)) * 0.3 +
            (1 - metrics_b["error_rate"]) * 0.3
        )
        
        if significance < 0.05:  # Statistically significant
            return "model_a" if score_a > score_b else "model_b"
        else:
            return "inconclusive"
```

### Phase 2 Success Criteria

- ✅ Training pipeline processes feedback into quality training data
- ✅ First custom models achieve parity with Docker Model Runner
- ✅ A/B testing framework operational and collecting metrics
- ✅ Automated model training and evaluation pipeline
- ✅ Model registry and version control system functional

---

## 4. Phase 3: Progressive Model Replacement (Months 2-6)

### Objectives
- Deploy custom models with reliable fallback mechanisms
- Gradually replace Docker models based on performance metrics
- Achieve complete AI independence
- Optimize models for news-specific analysis tasks

### Technical Tasks

#### 4.1 Custom Model Deployment

**Hybrid Model Manager:**

```python
# agents/common/hybrid_model_manager.py
import os
import json
import torch
from typing import Dict, Any, Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import requests

class HybridModelManager:
    """Manages deployment and fallback between custom and Docker models."""
    
    def __init__(self):
        self.custom_models = {}
        self.model_configs = self._load_model_configs()
        self.deployment_strategy = self._load_deployment_strategy()
        
    def _load_model_configs(self) -> Dict:
        """Load model configuration from database or config file."""
        # Query model_versions table for active models
        return {
            "bias_detection": {
                "custom_model_path": "./models/bias_detector_v2",
                "docker_fallback": "ai/mistral:7b-instruct-v0.3",
                "deployment_percentage": 50,  # Start with 50% traffic
                "performance_threshold": {"accuracy": 0.85, "inference_time": 2.0}
            },
            "sentiment_analysis": {
                "custom_model_path": "./models/sentiment_analyzer_v1",
                "docker_fallback": "ai/llama3.2:7b-instruct",
                "deployment_percentage": 25,  # More conservative rollout
                "performance_threshold": {"accuracy": 0.80, "inference_time": 2.5}
            }
        }
    
    def get_model_for_task(self, task: str, request_id: str = None) -> Dict[str, Any]:
        """Get appropriate model for task based on deployment strategy."""
        
        if task not in self.model_configs:
            raise ValueError(f"Unknown task: {task}")
        
        config = self.model_configs[task]
        
        # Determine if we should use custom model based on deployment percentage
        use_custom = self._should_use_custom_model(task, request_id)
        
        if use_custom and self._is_custom_model_available(task):
            return {
                "type": "custom",
                "model_path": config["custom_model_path"],
                "fallback": config["docker_fallback"]
            }
        else:
            return {
                "type": "docker",
                "model_name": config["docker_fallback"]
            }
    
    def _should_use_custom_model(self, task: str, request_id: str) -> bool:
        """Determine if custom model should be used based on deployment percentage."""
        import hashlib
        
        config = self.model_configs[task]
        deployment_pct = config.get("deployment_percentage", 0)
        
        if deployment_pct == 0:
            return False
        elif deployment_pct == 100:
            return True
        else:
            # Consistent routing based on request_id hash
            if request_id:
                hash_val = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
                return (hash_val % 100) < deployment_pct
            else:
                # Random routing if no request_id
                import random
                return random.random() < (deployment_pct / 100.0)
    
    def load_custom_model(self, task: str) -> bool:
        """Load custom model into memory."""
        
        if task not in self.model_configs:
            return False
        
        config = self.model_configs[task]
        model_path = config["custom_model_path"]
        
        try:
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            
            # Move to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model.eval()
            
            self.custom_models[task] = {
                "model": model,
                "tokenizer": tokenizer,
                "device": device,
                "loaded_at": datetime.utcnow()
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load custom model for {task}: {e}")
            return False
    
    def inference_with_fallback(self, task: str, text: str, request_id: str = None) -> Dict[str, Any]:
        """Perform inference with automatic fallback to Docker model."""
        
        model_info = self.get_model_for_task(task, request_id)
        
        try:
            if model_info["type"] == "custom":
                result = self._custom_inference(task, text)
                
                # Log successful custom inference
                self._log_inference_result(task, "custom", True, result.get("inference_time", 0))
                
                return result
                
            else:
                result = self._docker_inference(model_info["model_name"], text)
                
                # Log Docker inference
                self._log_inference_result(task, "docker", True, result.get("inference_time", 0))
                
                return result
                
        except Exception as e:
            logger.warning(f"Primary inference failed for {task}: {e}")
            
            # Fallback to Docker model
            try:
                fallback_model = model_info.get("fallback") or self.model_configs[task]["docker_fallback"]
                result = self._docker_inference(fallback_model, text)
                
                # Log fallback inference
                self._log_inference_result(task, "docker_fallback", True, result.get("inference_time", 0))
                
                return result
                
            except Exception as fallback_error:
                logger.error(f"Fallback inference also failed for {task}: {fallback_error}")
                
                # Log complete failure
                self._log_inference_result(task, "failed", False, 0)
                
                raise RuntimeError(f"Both primary and fallback inference failed for {task}")
```

#### 4.2 Performance Monitoring and Auto-scaling

**Model Performance Monitor:**

```python
# training/monitoring/performance_monitor.py
import time
import psycopg2
from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd

class ModelPerformanceMonitor:
    """Monitors model performance and triggers scaling decisions."""
    
    def __init__(self, db_url: str):
        self.db_url = db_url
        
    def collect_performance_metrics(self, hours_back: int = 24) -> Dict[str, Dict]:
        """Collect performance metrics for all models."""
        
        conn = psycopg2.connect(self.db_url)
        
        query = """
        SELECT 
            tool_name as task,
            model_used,
            AVG(inference_time) as avg_inference_time,
            COUNT(*) as request_count,
            SUM(CASE WHEN success_rate > 0.95 THEN 1 ELSE 0 END) / COUNT(*) as success_rate,
            AVG(accuracy_score) as avg_accuracy
        FROM performance_metrics 
        WHERE timestamp >= %s
        GROUP BY tool_name, model_used
        """
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        df = pd.read_sql(query, conn, params=[cutoff_time])
        conn.close()
        
        # Organize by task
        metrics_by_task = {}
        for _, row in df.iterrows():
            task = row['task']
            model = row['model_used']
            
            if task not in metrics_by_task:
                metrics_by_task[task] = {}
            
            metrics_by_task[task][model] = {
                "avg_inference_time": row['avg_inference_time'],
                "request_count": row['request_count'],
                "success_rate": row['success_rate'],
                "avg_accuracy": row['avg_accuracy']
            }
        
        return metrics_by_task
    
    def evaluate_deployment_readiness(self, task: str, custom_model: str) -> Dict[str, Any]:
        """Evaluate if custom model is ready for increased deployment."""
        
        metrics = self.collect_performance_metrics()
        
        if task not in metrics or custom_model not in metrics[task]:
            return {"ready": False, "reason": "Insufficient data"}
        
        custom_metrics = metrics[task][custom_model]
        
        # Find Docker model metrics for comparison
        docker_metrics = None
        for model_name, model_metrics in metrics[task].items():
            if model_name.startswith("ai/"):  # Docker Model Runner models
                docker_metrics = model_metrics
                break
        
        if not docker_metrics:
            return {"ready": False, "reason": "No Docker model baseline"}
        
        # Evaluation criteria
        evaluation = {
            "performance_better": custom_metrics["avg_accuracy"] > docker_metrics["avg_accuracy"],
            "speed_acceptable": custom_metrics["avg_inference_time"] < 3.0,
            "reliability_good": custom_metrics["success_rate"] > 0.95,
            "sufficient_volume": custom_metrics["request_count"] > 100
        }
        
        ready = all(evaluation.values())
        
        return {
            "ready": ready,
            "evaluation": evaluation,
            "custom_metrics": custom_metrics,
            "docker_metrics": docker_metrics
        }
    
    def recommend_deployment_percentage(self, task: str) -> int:
        """Recommend deployment percentage for custom model."""
        
        readiness = self.evaluate_deployment_readiness(task, f"custom_{task}_model")
        
        if not readiness["ready"]:
            return 0
        
        current_percentage = self._get_current_deployment_percentage(task)
        
        # Conservative scaling strategy
        if current_percentage == 0:
            return 10  # Start with 10%
        elif current_percentage < 50:
            return min(current_percentage + 20, 50)  # Increase by 20% up to 50%
        elif current_percentage < 90:
            return min(current_percentage + 10, 90)  # More conservative after 50%
        else:
            return 100  # Full deployment
    
    def auto_scale_deployment(self):
        """Automatically scale deployment based on performance."""
        
        tasks = ["bias_detection", "sentiment_analysis", "entity_recognition"]
        
        for task in tasks:
            try:
                recommended_pct = self.recommend_deployment_percentage(task)
                current_pct = self._get_current_deployment_percentage(task)
                
                if recommended_pct != current_pct:
                    self._update_deployment_percentage(task, recommended_pct)
                    
                    logger.info(f"Updated {task} deployment: {current_pct}% -> {recommended_pct}%")
                    
            except Exception as e:
                logger.error(f"Auto-scaling failed for {task}: {e}")
```

#### 4.3 Complete Independence Validation

**Independence Assessment Tool:**

```python
# scripts/assess_independence.py
import requests
import json
from datetime import datetime, timedelta
import psycopg2

class IndependenceAssessment:
    """Assess progress toward complete AI independence."""
    
    def __init__(self, db_url: str):
        self.db_url = db_url
        
    def assess_current_state(self) -> Dict[str, Any]:
        """Assess current state of AI independence."""
        
        # Check custom model deployment percentages
        deployment_status = self._check_deployment_status()
        
        # Check external dependencies
        external_deps = self._check_external_dependencies()
        
        # Check performance parity
        performance_status = self._check_performance_parity()
        
        # Calculate overall independence score
        independence_score = self._calculate_independence_score(
            deployment_status, external_deps, performance_status
        )
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "independence_score": independence_score,
            "deployment_status": deployment_status,
            "external_dependencies": external_deps,
            "performance_status": performance_status,
            "recommendations": self._generate_recommendations(
                deployment_status, external_deps, performance_status
            )
        }
    
    def _check_deployment_status(self) -> Dict[str, int]:
        """Check deployment percentages for all tasks."""
        conn = psycopg2.connect(self.db_url)
        
        query = """
        SELECT model_name, deployment_percentage 
        FROM model_versions 
        WHERE status = 'deployed' AND model_type = 'custom'
        """
        
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()
        
        deployment_status = {}
        for model_name, deployment_pct in results:
            task = model_name.split('_')[0]  # Extract task from model name
            deployment_status[task] = deployment_pct
        
        return deployment_status
    
    def _check_external_dependencies(self) -> Dict[str, bool]:
        """Check for remaining external dependencies."""
        
        # Check if Docker Model Runner is still being used
        docker_usage = self._check_docker_model_usage()
        
        # Check for external API calls
        external_api_usage = self._check_external_api_usage()
        
        return {
            "docker_model_runner_active": docker_usage > 0,
            "external_api_calls": external_api_usage > 0,
            "docker_usage_percentage": docker_usage,
            "external_api_percentage": external_api_usage
        }
    
    def _check_performance_parity(self) -> Dict[str, Dict[str, float]]:
        """Check if custom models meet performance requirements."""
        
        conn = psycopg2.connect(self.db_url)
        
        # Get recent performance metrics
        query = """
        SELECT 
            tool_name,
            model_used,
            AVG(accuracy_score) as avg_accuracy,
            AVG(inference_time) as avg_inference_time
        FROM performance_metrics 
        WHERE timestamp >= %s
        GROUP BY tool_name, model_used
        """
        
        cutoff_time = datetime.utcnow() - timedelta(days=7)
        cursor = conn.cursor()
        cursor.execute(query, [cutoff_time])
        results = cursor.fetchall()
        conn.close()
        
        performance_status = {}
        for tool, model, accuracy, inference_time in results:
            if tool not in performance_status:
                performance_status[tool] = {}
            
            model_type = "custom" if not model.startswith("ai/") else "docker"
            performance_status[tool][model_type] = {
                "accuracy": accuracy,
                "inference_time": inference_time
            }
       