"""
JustNews Metrics Library - Prometheus Integration
Provides standardized metrics collection for all JustNews agents
"""

import time
from typing import Optional, Dict, Any
from contextlib import contextmanager
from functools import wraps

from prometheus_client import (
    Counter, Gauge, Histogram, Summary,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)
from prometheus_client.multiprocess import MultiProcessCollector
import psutil
import GPUtil
import logging

logger = logging.getLogger(__name__)

class JustNewsMetrics:
    """
    Core metrics collection class for JustNews agents.
    Provides standardized metrics, middleware, and utilities.
    """

    def __init__(self, agent_name: str, registry: Optional[CollectorRegistry] = None):
        """
        Initialize metrics for an agent.

        Args:
            agent_name: Name of the agent (e.g., 'scout', 'analyst')
            registry: Optional custom registry (useful for testing)
        """
        self.agent_name = agent_name
        self.registry = registry or CollectorRegistry()

        # Initialize standard metrics
        self._init_standard_metrics()
        self._init_agent_specific_metrics()
        self._init_system_metrics()

        logger.info(f"Initialized metrics for agent: {agent_name}")

    def _init_standard_metrics(self):
        """Initialize standard HTTP and request metrics."""
        # Request metrics
        self.requests_total = Counter(
            'justnews_requests_total',
            'Total number of requests processed',
            ['agent', 'method', 'endpoint', 'status'],
            registry=self.registry
        )

        self.request_duration = Histogram(
            'justnews_request_duration_seconds',
            'Request duration in seconds',
            ['agent', 'method', 'endpoint'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
            registry=self.registry
        )

        # Error metrics
        self.errors_total = Counter(
            'justnews_errors_total',
            'Total number of errors',
            ['agent', 'error_type', 'endpoint'],
            registry=self.registry
        )

        # Active connections
        self.active_connections = Gauge(
            'justnews_active_connections',
            'Number of active connections',
            ['agent'],
            registry=self.registry
        )

    def _init_agent_specific_metrics(self):
        """Initialize agent-specific metrics (to be extended by subclasses)."""
        # Processing metrics
        self.processing_queue_size = Gauge(
            'justnews_processing_queue_size',
            'Current size of processing queue',
            ['agent', 'queue_type'],
            registry=self.registry
        )

        self.processing_duration = Histogram(
            'justnews_processing_duration_seconds',
            'Processing duration in seconds',
            ['agent', 'operation_type'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry
        )

        # Quality metrics
        self.quality_score = Histogram(
            'justnews_quality_score',
            'Quality score distribution',
            ['agent', 'metric_type'],
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry
        )

    def _init_system_metrics(self):
        """Initialize system-level metrics."""
        # Memory usage
        self.memory_usage_bytes = Gauge(
            'justnews_memory_usage_bytes',
            'Memory usage in bytes',
            ['agent', 'type'],
            registry=self.registry
        )

        # CPU usage
        self.cpu_usage_percent = Gauge(
            'justnews_cpu_usage_percent',
            'CPU usage percentage',
            ['agent'],
            registry=self.registry
        )

        # GPU metrics (if available)
        try:
            gpu_count = len(GPUtil.getGPUs())
            if gpu_count > 0:
                self.gpu_memory_used_bytes = Gauge(
                    'justnews_gpu_memory_used_bytes',
                    'GPU memory used in bytes',
                    ['agent', 'gpu_id'],
                    registry=self.registry
                )

                self.gpu_utilization_percent = Gauge(
                    'justnews_gpu_utilization_percent',
                    'GPU utilization percentage',
                    ['agent', 'gpu_id'],
                    registry=self.registry
                )
        except Exception as e:
            logger.warning(f"Could not initialize GPU metrics: {e}")

    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record an HTTP request."""
        self.requests_total.labels(
            agent=self.agent_name,
            method=method,
            endpoint=endpoint,
            status=str(status)
        ).inc()

        self.request_duration.labels(
            agent=self.agent_name,
            method=method,
            endpoint=endpoint
        ).observe(duration)

    def record_error(self, error_type: str, endpoint: str = ""):
        """Record an error."""
        self.errors_total.labels(
            agent=self.agent_name,
            error_type=error_type,
            endpoint=endpoint
        ).inc()

    def record_processing(self, operation_type: str, duration: float):
        """Record processing operation."""
        self.processing_duration.labels(
            agent=self.agent_name,
            operation_type=operation_type
        ).observe(duration)

    def update_queue_size(self, queue_type: str, size: int):
        """Update processing queue size."""
        self.processing_queue_size.labels(
            agent=self.agent_name,
            queue_type=queue_type
        ).set(size)

    def record_quality_score(self, metric_type: str, score: float):
        """Record quality score."""
        self.quality_score.labels(
            agent=self.agent_name,
            metric_type=metric_type
        ).observe(score)

    def update_system_metrics(self):
        """Update system resource metrics."""
        try:
            # Memory metrics
            process = psutil.Process()
            memory_info = process.memory_info()

            self.memory_usage_bytes.labels(
                agent=self.agent_name,
                type='rss'
            ).set(memory_info.rss)

            self.memory_usage_bytes.labels(
                agent=self.agent_name,
                type='vms'
            ).set(memory_info.vms)

            # CPU metrics
            cpu_percent = process.cpu_percent(interval=1.0)
            self.cpu_usage_percent.labels(
                agent=self.agent_name
            ).set(cpu_percent)

            # GPU metrics
            try:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    self.gpu_memory_used_bytes.labels(
                        agent=self.agent_name,
                        gpu_id=str(i)
                    ).set(gpu.memoryUsed * 1024 * 1024)  # Convert MB to bytes

                    self.gpu_utilization_percent.labels(
                        agent=self.agent_name,
                        gpu_id=str(i)
                    ).set(gpu.load * 100)
            except Exception as e:
                logger.debug(f"Could not update GPU metrics: {e}")

        except Exception as e:
            logger.warning(f"Could not update system metrics: {e}")

    def get_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        return generate_latest(self.registry).decode('utf-8')

    @contextmanager
    def measure_time(self, operation_type: str):
        """Context manager to measure operation duration."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record_processing(operation_type, duration)

    def request_middleware(self, request, call_next):
        """
        FastAPI middleware for automatic request metrics collection.

        Usage:
            app.middleware("http")(metrics.request_middleware)
        """
        start_time = time.time()

        # Update active connections
        self.active_connections.labels(agent=self.agent_name).inc()

        try:
            response = call_next(request)
            duration = time.time() - start_time

            # Record successful request
            self.record_request(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code,
                duration=duration
            )

            return response

        except Exception as e:
            duration = time.time() - start_time

            # Record error
            self.record_error(
                error_type=type(e).__name__,
                endpoint=request.url.path
            )

            # Record failed request
            self.record_request(
                method=request.method,
                endpoint=request.url.path,
                status=500,
                duration=duration
            )

            raise

        finally:
            # Decrement active connections
            self.active_connections.labels(agent=self.agent_name).dec()


# Global metrics instance (can be overridden per agent)
_default_metrics = None

def get_metrics(agent_name: str) -> JustNewsMetrics:
    """Get or create metrics instance for an agent."""
    global _default_metrics

    if _default_metrics is None or _default_metrics.agent_name != agent_name:
        _default_metrics = JustNewsMetrics(agent_name)

    return _default_metrics

def init_metrics_for_agent(agent_name: str) -> JustNewsMetrics:
    """Initialize metrics for a specific agent."""
    return JustNewsMetrics(agent_name)

# Utility functions for common patterns
def measure_processing_time(operation_type: str):
    """Decorator to measure processing time."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            agent_name = getattr(args[0], 'agent_name', 'unknown') if args else 'unknown'
            metrics = get_metrics(agent_name)

            with metrics.measure_time(operation_type):
                return func(*args, **kwargs)
        return wrapper
    return decorator

def record_quality_metric(metric_type: str, score: float, agent_name: str = None):
    """Record a quality metric."""
    if agent_name is None:
        # Try to infer from context
        agent_name = 'unknown'

    metrics = get_metrics(agent_name)
    metrics.record_quality_score(metric_type, score)

def update_system_metrics(agent_name: str = None):
    """Update system metrics for an agent."""
    if agent_name is None:
        agent_name = 'unknown'

    metrics = get_metrics(agent_name)
    metrics.update_system_metrics()
