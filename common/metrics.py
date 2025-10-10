"""
JustNews Metrics Library - Prometheus Integration
Provides standardized metrics collection for all JustNews agents

Preview Mode: Graceful fallbacks for optional dependencies
"""

import time
from typing import Optional, Dict, Any
from contextlib import contextmanager
from functools import wraps
import logging

logger = logging.getLogger(__name__)

# Graceful fallback for prometheus_client (optional in Preview mode)
try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
    )
    from prometheus_client.multiprocess import MultiProcessCollector
    PROMETHEUS_AVAILABLE = True
except ImportError:
    logger.warning("prometheus_client not available - using fallback stubs")
    PROMETHEUS_AVAILABLE = False
    
    # Minimal fallback stubs for Preview mode
    class _MetricStub:
        """Stub metric that does nothing"""
        def __init__(self, *args, **kwargs):
            pass
        def inc(self, *args, **kwargs):
            pass
        def dec(self, *args, **kwargs):
            pass
        def set(self, *args, **kwargs):
            pass
        def observe(self, *args, **kwargs):
            pass
        def labels(self, **kwargs):
            return self
    
    Counter = Gauge = Histogram = Summary = _MetricStub
    
    class CollectorRegistry:
        """Fallback CollectorRegistry for Preview mode"""
        def __init__(self):
            pass
    
    def generate_latest(registry=None):
        """Fallback metrics generation"""
        return b"# Prometheus client not available in Preview mode\n"
    
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"
    
    class MultiProcessCollector:
        """Fallback MultiProcessCollector"""
        def __init__(self, *args, **kwargs):
            pass

# Graceful fallback for psutil (optional)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    logger.warning("psutil not available - system metrics disabled")
    PSUTIL_AVAILABLE = False
    psutil = None

# Graceful fallback for GPUtil (optional)
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    logger.warning("GPUtil not available - GPU metrics disabled")
    GPUTIL_AVAILABLE = False
    GPUtil = None

class JustNewsMetrics:
    """
    Core metrics collection class for JustNews agents.
    Provides standardized metrics, middleware, and utilities.
    """

    # Agent display names for clearer labeling
    AGENT_DISPLAY_NAMES = {
        'scout': 'content-discovery-agent',
        'analyst': 'sentiment-analysis-agent',
        'synthesizer': 'content-synthesis-agent',
        'fact_checker': 'fact-verification-agent',
        'critic': 'quality-assessment-agent',
        'memory': 'data-storage-agent',
        'reasoning': 'logical-reasoning-agent',
        'newsreader': 'news-processing-agent',
        'balancer': 'load-balancing-agent',
        'archive': 'content-archiving-agent',
        'dashboard': 'web-dashboard-agent',
        'analytics': 'system-analytics-agent',
        'chief_editor': 'workflow-orchestration-agent',
        'crawler': 'content-crawling-agent',
        'gpu_orchestrator': 'gpu-resource-orchestrator',
        'mcp_bus': 'communication-bus'
    }

    # Endpoint display names for clearer labeling
    ENDPOINT_DISPLAY_NAMES = {
        '/health': 'health-check-endpoint',
        '/ready': 'readiness-check-endpoint',
        '/metrics': 'prometheus-metrics-endpoint',
        '/': 'root-endpoint',
        '/unified_production_crawl': 'crawling-operation-endpoint',
        '/get_crawler_info': 'crawler-info-endpoint',
        '/get_performance_metrics': 'performance-metrics-endpoint',
        '/analyze': 'sentiment-analysis-endpoint',
        '/synthesize': 'content-synthesis-endpoint',
        '/fact_check': 'fact-checking-endpoint',
        '/quality_check': 'quality-assessment-endpoint',
        '/store': 'data-storage-endpoint',
        '/retrieve': 'data-retrieval-endpoint',
        '/reason': 'logical-reasoning-endpoint',
        '/process': 'content-processing-endpoint',
        '/balance': 'load-balancing-endpoint',
        '/archive': 'content-archiving-endpoint',
        '/dashboard': 'dashboard-endpoint',
        '/analytics': 'analytics-endpoint',
        '/orchestrate': 'workflow-orchestration-endpoint',
        '/gpu_status': 'gpu-status-endpoint',
        '/register': 'agent-registration-endpoint',
        '/call': 'inter-agent-communication-endpoint'
    }

    # HTTP status code mappings
    STATUS_DISPLAY_NAMES = {
        '200': 'success',
        '201': 'created',
        '400': 'bad-request',
        '401': 'unauthorized',
        '403': 'forbidden',
        '404': 'not-found',
        '500': 'internal-server-error',
        '502': 'bad-gateway',
        '503': 'service-unavailable'
    }

    def __init__(self, agent_name: str, registry: Optional[CollectorRegistry] = None):
        """
        Initialize metrics for an agent.

        Args:
            agent_name: Name of the agent (e.g., 'scout', 'analyst')
            registry: Optional custom registry (useful for testing)
        """
        self.agent_name = agent_name
        self.display_name = self.AGENT_DISPLAY_NAMES.get(agent_name, f'{agent_name}-agent')
        self.registry = registry or CollectorRegistry()

        # Initialize standard metrics
        self._init_standard_metrics()
        self._init_agent_specific_metrics()
        self._init_system_metrics()

        logger.info(f"Initialized metrics for agent: {agent_name} (display: {self.display_name})")

    def _init_standard_metrics(self):
        """Initialize standard HTTP and request metrics."""
        # Request metrics with enhanced labels
        self.requests_total = Counter(
            'justnews_requests_total',
            'Total number of requests processed',
            ['agent', 'agent_display_name', 'method', 'endpoint', 'endpoint_display_name', 'status', 'status_display_name'],
            registry=self.registry
        )

        self.request_duration = Histogram(
            'justnews_request_duration_seconds',
            'Request duration in seconds',
            ['agent', 'agent_display_name', 'method', 'endpoint', 'endpoint_display_name'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
            registry=self.registry
        )

        # Error metrics with enhanced labels
        self.errors_total = Counter(
            'justnews_errors_total',
            'Total number of errors',
            ['agent', 'agent_display_name', 'error_type', 'endpoint', 'endpoint_display_name'],
            registry=self.registry
        )

        # Active connections with enhanced labels
        self.active_connections = Gauge(
            'justnews_active_connections',
            'Number of active connections',
            ['agent', 'agent_display_name'],
            registry=self.registry
        )

    def _init_agent_specific_metrics(self):
        """Initialize agent-specific metrics (to be extended by subclasses)."""
        # Processing metrics with enhanced labels
        self.processing_queue_size = Gauge(
            'justnews_processing_queue_size',
            'Current size of processing queue',
            ['agent', 'agent_display_name', 'queue_type', 'queue_display_name'],
            registry=self.registry
        )

        self.processing_duration = Histogram(
            'justnews_processing_duration_seconds',
            'Processing duration in seconds',
            ['agent', 'agent_display_name', 'operation_type', 'operation_display_name'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry
        )

        # Quality metrics with enhanced labels
        self.quality_score = Histogram(
            'justnews_quality_score',
            'Quality score distribution',
            ['agent', 'agent_display_name', 'metric_type', 'metric_display_name'],
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry
        )

    def _init_system_metrics(self):
        """Initialize system-level metrics."""
        # Memory usage with enhanced labels
        self.memory_usage_bytes = Gauge(
            'justnews_memory_usage_bytes',
            'Memory usage in bytes',
            ['agent', 'agent_display_name', 'type', 'memory_type_display_name'],
            registry=self.registry
        )

        # CPU usage with enhanced labels
        self.cpu_usage_percent = Gauge(
            'justnews_cpu_usage_percent',
            'CPU usage percentage',
            ['agent', 'agent_display_name'],
            registry=self.registry
        )

        # GPU metrics (if GPUtil is available)
        if GPUTIL_AVAILABLE:
            try:
                gpu_count = len(GPUtil.getGPUs())
                if gpu_count > 0:
                    self.gpu_memory_used_bytes = Gauge(
                        'justnews_gpu_memory_used_bytes',
                        'GPU memory used in bytes',
                        ['agent', 'agent_display_name', 'gpu_id', 'gpu_display_name'],
                        registry=self.registry
                    )

                    self.gpu_utilization_percent = Gauge(
                        'justnews_gpu_utilization_percent',
                        'GPU utilization percentage',
                        ['agent', 'agent_display_name', 'gpu_id', 'gpu_display_name'],
                        registry=self.registry
                    )
            except Exception as e:
                logger.warning(f"Could not initialize GPU metrics: {e}")
        else:
            logger.debug("GPUtil not available - skipping GPU metric initialization")

    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record an HTTP request."""
        endpoint_display = self.ENDPOINT_DISPLAY_NAMES.get(endpoint, f'{endpoint.replace("/", "").replace("_", "-")}-endpoint')
        status_display = self.STATUS_DISPLAY_NAMES.get(str(status), f'http-{status}')

        self.requests_total.labels(
            agent=self.agent_name,
            agent_display_name=self.display_name,
            method=method,
            endpoint=endpoint,
            endpoint_display_name=endpoint_display,
            status=str(status),
            status_display_name=status_display
        ).inc()

        self.request_duration.labels(
            agent=self.agent_name,
            agent_display_name=self.display_name,
            method=method,
            endpoint=endpoint,
            endpoint_display_name=endpoint_display
        ).observe(duration)

    def record_error(self, error_type: str, endpoint: str = ""):
        """Record an error."""
        endpoint_display = self.ENDPOINT_DISPLAY_NAMES.get(endpoint, f'{endpoint.replace("/", "").replace("_", "-")}-endpoint' if endpoint else 'unknown-endpoint')

        self.errors_total.labels(
            agent=self.agent_name,
            agent_display_name=self.display_name,
            error_type=error_type,
            endpoint=endpoint,
            endpoint_display_name=endpoint_display
        ).inc()

    def record_processing(self, operation_type: str, duration: float):
        """Record processing operation."""
        # Create a more readable operation display name
        operation_display = operation_type.replace('_', '-').replace(' ', '-')

        self.processing_duration.labels(
            agent=self.agent_name,
            agent_display_name=self.display_name,
            operation_type=operation_type,
            operation_display_name=operation_display
        ).observe(duration)

    def update_queue_size(self, queue_type: str, size: int):
        """Update processing queue size."""
        queue_display = queue_type.replace('_', '-').replace(' ', '-')

        self.processing_queue_size.labels(
            agent=self.agent_name,
            agent_display_name=self.display_name,
            queue_type=queue_type,
            queue_display_name=queue_display
        ).set(size)

    def record_quality_score(self, metric_type: str, score: float):
        """Record quality score."""
        metric_display = metric_type.replace('_', '-').replace(' ', '-')

        self.quality_score.labels(
            agent=self.agent_name,
            agent_display_name=self.display_name,
            metric_type=metric_type,
            metric_display_name=metric_display
        ).observe(score)

    def update_system_metrics(self):
        """Update system resource metrics."""
        if not PSUTIL_AVAILABLE:
            logger.debug("psutil not available - skipping system metrics")
            return
            
        try:
            # Memory metrics
            process = psutil.Process()
            memory_info = process.memory_info()

            self.memory_usage_bytes.labels(
                agent=self.agent_name,
                agent_display_name=self.display_name,
                type='rss',
                memory_type_display_name='resident-set-size'
            ).set(memory_info.rss)

            self.memory_usage_bytes.labels(
                agent=self.agent_name,
                agent_display_name=self.display_name,
                type='vms',
                memory_type_display_name='virtual-memory-size'
            ).set(memory_info.vms)

            # CPU metrics
            cpu_percent = process.cpu_percent(interval=1.0)
            self.cpu_usage_percent.labels(
                agent=self.agent_name,
                agent_display_name=self.display_name
            ).set(cpu_percent)

            # GPU metrics (only if GPUtil is available)
            if GPUTIL_AVAILABLE:
                try:
                    gpus = GPUtil.getGPUs()
                    for i, gpu in enumerate(gpus):
                        gpu_display = f'gpu-{i}'

                        self.gpu_memory_used_bytes.labels(
                            agent=self.agent_name,
                            agent_display_name=self.display_name,
                            gpu_id=str(i),
                            gpu_display_name=gpu_display
                        ).set(gpu.memoryUsed * 1024 * 1024)  # Convert MB to bytes

                        self.gpu_utilization_percent.labels(
                            agent=self.agent_name,
                            agent_display_name=self.display_name,
                            gpu_id=str(i),
                            gpu_display_name=gpu_display
                        ).set(gpu.load * 100)
                except Exception as e:
                    logger.debug(f"Could not update GPU metrics: {e}")
            else:
                logger.debug("GPUtil not available - skipping GPU metrics")

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

    async def request_middleware(self, request, call_next):
        """
        FastAPI middleware for automatic request metrics collection.

        Usage:
            app.middleware("http")(metrics.request_middleware)
        """
        start_time = time.time()

        # Update active connections
        self.active_connections.labels(
            agent=self.agent_name,
            agent_display_name=self.display_name
        ).inc()

        try:
            response = await call_next(request)
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
            self.active_connections.labels(
                agent=self.agent_name,
                agent_display_name=self.display_name
            ).dec()


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
