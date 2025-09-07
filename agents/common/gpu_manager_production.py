"""
from common.observability import get_logger
Production Multi-Agent GPU Manager for JustNewsAgent
Replaces the shim implementation with production-ready GPU resource management

Features:
- Per-agent GPU allocation with memory limits
- Automatic GPU health monitoring
- Resource pooling and optimization
- Graceful fallback to CPU
- Comprehensive error handling and recovery
- Real-time performance metrics
- Atomic allocation operations
"""


import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any

# GPU and ML imports with graceful fallbacks
try:
    import numpy as np
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    TORCH_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    TORCH_AVAILABLE = False
    torch = None
    np = None

logger = get_logger(__name__)

@dataclass
class GPUAllocation:
    """Represents a GPU allocation for an agent"""
    agent_name: str
    gpu_device: int
    allocated_memory_gb: float
    batch_size: int
    allocation_time: datetime
    status: str = "active"

@dataclass
class GPUStatus:
    """GPU device status information"""
    device_id: int
    total_memory_gb: float
    used_memory_gb: float
    free_memory_gb: float
    utilization_percent: float
    temperature_c: float
    power_draw_w: float
    is_healthy: bool

class GPUHealthMonitor:
    """Monitors GPU health and performance"""

    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.last_check = 0
        self._lock = threading.Lock()
        self._health_cache: dict[int, GPUStatus] = {}

    def get_gpu_status(self, device_id: int = 0) -> GPUStatus:
        """Get current GPU status with caching"""
        current_time = time.time()

        with self._lock:
            if current_time - self.last_check < self.check_interval:
                return self._health_cache.get(device_id)

        # Perform fresh health check
        status = self._check_gpu_health(device_id)

        with self._lock:
            self._health_cache[device_id] = status
            self.last_check = current_time

        return status

    def _check_gpu_health(self, device_id: int) -> GPUStatus:
        """Perform comprehensive GPU health check"""
        try:
            # Get nvidia-smi data
            nvidia_data = self._get_nvidia_smi_data(device_id)
            torch_data = self._get_torch_memory_data(device_id)

            # Calculate health status
            is_healthy = self._assess_health(nvidia_data, torch_data)

            return GPUStatus(
                device_id=device_id,
                total_memory_gb=nvidia_data.get('total_memory_gb', 0),
                used_memory_gb=torch_data.get('used_memory_gb', 0),
                free_memory_gb=nvidia_data.get('free_memory_gb', 0),
                utilization_percent=nvidia_data.get('utilization_percent', 0),
                temperature_c=nvidia_data.get('temperature_c', 0),
                power_draw_w=nvidia_data.get('power_draw_w', 0),
                is_healthy=is_healthy
            )

        except Exception as e:
            logger.error(f"GPU health check failed for device {device_id}: {e}")
            return GPUStatus(
                device_id=device_id,
                total_memory_gb=0,
                used_memory_gb=0,
                free_memory_gb=0,
                utilization_percent=0,
                temperature_c=0,
                power_draw_w=0,
                is_healthy=False
            )

    def _get_nvidia_smi_data(self, device_id: int) -> dict[str, float]:
        """Get GPU data from nvidia-smi"""
        try:
            cmd = [
                'nvidia-smi',
                '--query-gpu=memory.total,memory.free,utilization.gpu,temperature.gpu,power.draw',
                '--format=csv,nounits,noheader'
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

            if result.returncode != 0:
                return {}

            lines = result.stdout.strip().split('\n')
            if device_id >= len(lines):
                return {}

            parts = [p.strip() for p in lines[device_id].split(',')]

            return {
                'total_memory_gb': float(parts[0]) / 1024 if parts[0] else 0,
                'free_memory_gb': float(parts[1]) / 1024 if parts[1] else 0,
                'utilization_percent': float(parts[2]) if parts[2] else 0,
                'temperature_c': float(parts[3]) if parts[3] else 0,
                'power_draw_w': float(parts[4]) if parts[4] else 0,
            }

        except Exception:
            return {}

    def _get_torch_memory_data(self, device_id: int) -> dict[str, float]:
        """Get memory data from PyTorch"""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return {'used_memory_gb': 0}

        try:
            allocated = torch.cuda.memory_allocated(device_id) / (1024**3)
            return {'used_memory_gb': allocated}
        except Exception:
            return {'used_memory_gb': 0}

    def _assess_health(self, nvidia_data: dict, torch_data: dict) -> bool:
        """Assess overall GPU health"""
        try:
            # Check temperature (should be < 85°C)
            temp = nvidia_data.get('temperature_c', 0)
            if temp > 85:
                return False

            # Check utilization (should be responsive)
            util = nvidia_data.get('utilization_percent', 0)
            if util > 95:  # Might indicate hung GPU
                return False

            # Check memory consistency
            total = nvidia_data.get('total_memory_gb', 0)
            free = nvidia_data.get('free_memory_gb', 0)
            used = torch_data.get('used_memory_gb', 0)

            if total > 0 and (free + used) > total * 1.1:  # 10% tolerance
                return False

            return True

        except Exception:
            return False

class MultiAgentGPUManager:
    """
    Production GPU manager for multi-agent systems

    Features:
    - Atomic GPU allocation per agent
    - Memory limit enforcement
    - Health monitoring and automatic recovery
    - Resource optimization and pooling
    - Comprehensive logging and metrics
    """

    def __init__(self, max_memory_per_agent_gb: float = 8.0, health_check_interval: float = 30.0):
        self.max_memory_per_agent_gb = max_memory_per_agent_gb
        self.health_monitor = GPUHealthMonitor(health_check_interval)

        # Allocation tracking
        self._allocations: dict[str, GPUAllocation] = {}
        self._lock = threading.Lock()

        # Performance metrics
        self.metrics = {
            'total_allocations': 0,
            'successful_allocations': 0,
            'failed_allocations': 0,
            'cpu_fallbacks': 0,
            'gpu_recoveries': 0
        }

        logger.info("🤖 MultiAgentGPUManager initialized")
        logger.info(f"   Max memory per agent: {max_memory_per_agent_gb}GB")
        logger.info(f"   Health check interval: {health_check_interval}s")

    def request_gpu_allocation(
        self,
        agent_name: str,
        requested_memory_gb: float = 4.0,
        preferred_device: int | None = None,
        model_type: str = "general"
    ) -> dict[str, Any]:
        """
        Request GPU allocation for an agent

        Args:
            agent_name: Unique agent identifier
            requested_memory_gb: Requested GPU memory in GB
            preferred_device: Preferred GPU device (None for auto-selection)
            model_type: Type of model ('embedding', 'generation', 'vision', 'general')

        Returns:
            Allocation result dictionary
        """
        self.metrics['total_allocations'] += 1

        with self._lock:
            try:
                # Check if agent already has allocation
                if agent_name in self._allocations:
                    existing = self._allocations[agent_name]
                    if existing.status == 'active':
                        return {
                            'status': 'already_allocated',
                            'gpu_device': existing.gpu_device,
                            'allocated_memory_gb': existing.allocated_memory_gb,
                            'batch_size': existing.batch_size,
                            'message': f'Agent {agent_name} already has active allocation'
                        }

                # Validate memory request
                if requested_memory_gb > self.max_memory_per_agent_gb:
                    return {
                        'status': 'memory_limit_exceeded',
                        'message': f'Requested {requested_memory_gb}GB exceeds limit of {self.max_memory_per_agent_gb}GB'
                    }

                # Find available GPU
                device_id = self._find_available_gpu(requested_memory_gb, preferred_device)
                if device_id is None:
                    self.metrics['cpu_fallbacks'] += 1
                    return {
                        'status': 'cpu_fallback',
                        'reason': 'No GPU available with sufficient memory',
                        'message': 'Falling back to CPU processing'
                    }

                # Check GPU health
                gpu_status = self.health_monitor.get_gpu_status(device_id)
                if not gpu_status.is_healthy:
                    logger.warning(f"GPU {device_id} health check failed, attempting recovery")
                    if not self._attempt_gpu_recovery(device_id):
                        return {
                            'status': 'gpu_unhealthy',
                            'reason': f'GPU {device_id} is unhealthy',
                            'message': 'GPU health check failed'
                        }

                # Calculate batch size based on memory and model type
                batch_size = self._calculate_optimal_batch_size(requested_memory_gb, model_type)

                # Create allocation
                allocation = GPUAllocation(
                    agent_name=agent_name,
                    gpu_device=device_id,
                    allocated_memory_gb=requested_memory_gb,
                    batch_size=batch_size,
                    allocation_time=datetime.now(),
                    status='active'
                )

                self._allocations[agent_name] = allocation
                self.metrics['successful_allocations'] += 1

                logger.info(f"✅ GPU allocated: {agent_name} -> GPU {device_id} ({requested_memory_gb}GB, batch_size={batch_size}, model_type={model_type})")

                return {
                    'status': 'allocated',
                    'gpu_device': device_id,
                    'allocated_memory_gb': requested_memory_gb,
                    'batch_size': batch_size,
                    'model_type': model_type,
                    'message': f'Successfully allocated GPU {device_id} for {agent_name}'
                }

            except Exception as e:
                self.metrics['failed_allocations'] += 1
                logger.error(f"❌ GPU allocation failed for {agent_name}: {e}")
                return {
                    'status': 'allocation_failed',
                    'reason': str(e),
                    'message': 'GPU allocation failed due to internal error'
                }

    def release_gpu_allocation(self, agent_name: str) -> bool:
        """Release GPU allocation for an agent"""
        with self._lock:
            if agent_name not in self._allocations:
                return False

            allocation = self._allocations[agent_name]
            if allocation.status != 'active':
                return False

            # Mark as released
            allocation.status = 'released'

            # Clean up GPU memory if possible
            self._cleanup_gpu_memory(allocation.gpu_device)

            logger.info(f"✅ GPU released: {agent_name} <- GPU {allocation.gpu_device}")
            return True

    def _find_available_gpu(self, requested_memory_gb: float, preferred_device: int | None = None) -> int | None:
        """Find an available GPU with sufficient memory"""
        if not GPU_AVAILABLE:
            return None

        try:
            device_count = torch.cuda.device_count()

            # Check preferred device first
            if preferred_device is not None and 0 <= preferred_device < device_count:
                gpu_status = self.health_monitor.get_gpu_status(preferred_device)
                if (gpu_status.is_healthy and
                    gpu_status.free_memory_gb >= requested_memory_gb):
                    return preferred_device

            # Check all devices
            for device_id in range(device_count):
                gpu_status = self.health_monitor.get_gpu_status(device_id)
                if (gpu_status.is_healthy and
                    gpu_status.free_memory_gb >= requested_memory_gb):
                    return device_id

        except Exception as e:
            logger.error(f"Error finding available GPU: {e}")

        return None

    def _calculate_optimal_batch_size(self, memory_gb: float, model_type: str = "general") -> int:
        """Calculate optimal batch size using enhanced optimization with learning"""
        try:
            # Try enhanced optimization first
            try:
                from .gpu_optimizer_enhanced import optimize_gpu_allocation
                optimization = optimize_gpu_allocation(self.__class__.__name__, model_type, memory_gb)

                if optimization['confidence'] > 0.3:  # Use learned optimization if confidence is reasonable
                    logger.info(f"🎯 Using learned optimization: batch_size={optimization['batch_size']} (confidence={optimization['confidence']:.2f})")
                    return optimization['batch_size']
            except ImportError:
                logger.debug("Enhanced optimizer not available, using heuristic optimization")

            # Fall back to enhanced heuristic optimization
            return self._calculate_enhanced_heuristic_batch_size(memory_gb, model_type)

        except Exception as e:
            logger.warning(f"Error in enhanced batch calculation: {e}, using basic heuristic")
            return self._calculate_basic_heuristic_batch_size(memory_gb, model_type)

    def _calculate_enhanced_heuristic_batch_size(self, memory_gb: float, model_type: str = "general") -> int:
        """Enhanced heuristic batch size calculation with configuration awareness"""
        try:
            # Get GPU memory info for more accurate calculations
            if GPU_AVAILABLE and torch.cuda.is_available():
                gpu_status = self.health_monitor.get_gpu_status(0)  # Use first GPU as reference
                free_memory = gpu_status.free_memory_gb

                # Use available memory as constraint
                available_memory = min(memory_gb, free_memory * 0.8)  # Use 80% of available

                # Try to get model-specific configuration
                try:
                    from .gpu_config_manager import get_model_config
                    model_config = get_model_config()
                    recommended_batch = model_config.get('batch_size_recommendation')

                    if recommended_batch and available_memory >= model_config.get('max_memory_usage_gb', 0):
                        return min(recommended_batch, int(available_memory * 8))
                except ImportError:
                    pass

                # Enhanced model-specific batch size calculations
                if model_type == "embedding":
                    # Embedding models: optimize for latency with larger batches
                    if available_memory >= 4:
                        return min(32, int(available_memory * 8))  # Up to 32 for large memory
                    elif available_memory >= 2:
                        return 16
                    elif available_memory >= 1:
                        return 8
                    else:
                        return 4

                elif model_type == "generation":
                    # Generation models: balance throughput and memory
                    if available_memory >= 8:
                        return min(16, int(available_memory * 2))
                    elif available_memory >= 4:
                        return 8
                    elif available_memory >= 2:
                        return 4
                    else:
                        return 2

                elif model_type == "vision":
                    # Vision models: larger inputs, smaller batches
                    if available_memory >= 8:
                        return min(8, int(available_memory * 1))
                    elif available_memory >= 4:
                        return 4
                    else:
                        return 2

                elif model_type == "classification":
                    # Classification models: larger batches for efficiency
                    if available_memory >= 8:
                        return min(32, int(available_memory * 6))
                    elif available_memory >= 4:
                        return 16
                    elif available_memory >= 2:
                        return 8
                    else:
                        return 4

                else:  # general case
                    # Adaptive batch sizing based on memory with enhanced logic
                    if available_memory >= 8:
                        return 24
                    elif available_memory >= 4:
                        return 12
                    elif available_memory >= 2:
                        return 6
                    else:
                        return 2
            else:
                # CPU fallback - smaller batches
                return 1

        except Exception as e:
            logger.warning(f"Error in enhanced heuristic batch calculation: {e}")
            return self._calculate_basic_heuristic_batch_size(memory_gb, model_type)

    def _calculate_basic_heuristic_batch_size(self, memory_gb: float, model_type: str = "general") -> int:
        """Basic heuristic batch size calculation (fallback)"""
        try:
            # Get GPU memory info for more accurate calculations
            if GPU_AVAILABLE and torch.cuda.is_available():
                gpu_status = self.health_monitor.get_gpu_status(0)  # Use first GPU as reference
                free_memory = gpu_status.free_memory_gb

                # Reserve some memory for overhead
                available_memory = min(memory_gb, free_memory * 0.8)  # Use 80% of available

                # Model-specific batch size calculations
                if model_type == "embedding":
                    # Embedding models: smaller batches, focus on latency
                    if available_memory >= 4:
                        return min(32, int(available_memory * 8))  # Up to 32 for large memory
                    elif available_memory >= 2:
                        return 16
                    elif available_memory >= 1:
                        return 8
                    else:
                        return 4

                elif model_type == "generation":
                    # Generation models: balance throughput and memory
                    if available_memory >= 8:
                        return min(16, int(available_memory * 2))
                    elif available_memory >= 4:
                        return 8
                    elif available_memory >= 2:
                        return 4
                    else:
                        return 2

                elif model_type == "vision":
                    # Vision models: larger inputs, smaller batches
                    if available_memory >= 8:
                        return min(8, int(available_memory * 1))
                    elif available_memory >= 4:
                        return 4
                    else:
                        return 2

                else:  # general case
                    # General heuristic: more memory = larger batches
                    if available_memory >= 8:
                        return 16
                    elif available_memory >= 4:
                        return 8
                    elif available_memory >= 2:
                        return 4
                    else:
                        return 1
            else:
                # CPU fallback - smaller batches
                return 1

        except Exception as e:
            logger.warning(f"Error calculating optimal batch size: {e}, using conservative default")
            return 1

    def _attempt_gpu_recovery(self, device_id: int) -> bool:
        """Attempt to recover an unhealthy GPU"""
        try:
            # Clear GPU cache
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Wait a moment
            time.sleep(1)

            # Re-check health
            gpu_status = self.health_monitor.get_gpu_status(device_id)
            if gpu_status.is_healthy:
                self.metrics['gpu_recoveries'] += 1
                logger.info(f"✅ GPU {device_id} recovered")
                return True

        except Exception as e:
            logger.error(f"GPU recovery failed for device {device_id}: {e}")

        return False

    def _cleanup_gpu_memory(self, device_id: int):
        """Clean up GPU memory for a device"""
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                # Switch to device and clear cache
                torch.cuda.set_device(device_id)
                torch.cuda.empty_cache()
                logger.debug(f"GPU memory cleaned for device {device_id}")
        except Exception as e:
            logger.warning(f"Failed to cleanup GPU memory for device {device_id}: {e}")

    def get_allocation_status(self, agent_name: str) -> dict[str, Any] | None:
        """Get allocation status for an agent"""
        with self._lock:
            allocation = self._allocations.get(agent_name)
            if not allocation:
                return None

            return {
                'agent_name': allocation.agent_name,
                'gpu_device': allocation.gpu_device,
                'allocated_memory_gb': allocation.allocated_memory_gb,
                'batch_size': allocation.batch_size,
                'allocation_time': allocation.allocation_time.isoformat(),
                'status': allocation.status
            }

    def get_system_status(self) -> dict[str, Any]:
        """Get overall system status"""
        gpu_statuses = []
        if GPU_AVAILABLE:
            for device_id in range(torch.cuda.device_count()):
                gpu_statuses.append(self.health_monitor.get_gpu_status(device_id))

        return {
            'gpu_available': GPU_AVAILABLE,
            'torch_available': TORCH_AVAILABLE,
            'gpu_count': len(gpu_statuses) if GPU_AVAILABLE else 0,
            'gpu_statuses': [vars(status) for status in gpu_statuses],
            'active_allocations': len([a for a in self._allocations.values() if a.status == 'active']),
            'total_allocations': len(self._allocations),
            'metrics': self.metrics.copy()
        }

    @contextmanager
    def gpu_context(self, agent_name: str, memory_gb: float = 4.0):
        """Context manager for GPU allocation"""
        allocation = self.request_gpu_allocation(agent_name, memory_gb)

        if allocation['status'] == 'allocated':
            try:
                yield allocation
            finally:
                self.release_gpu_allocation(agent_name)
        else:
            # CPU fallback
            yield {
                'status': 'cpu_fallback',
                'gpu_device': -1,
                'allocated_memory_gb': 0,
                'batch_size': 1,
                'message': allocation.get('message', 'CPU fallback')
            }

# Global manager instance
_global_gpu_manager: MultiAgentGPUManager | None = None
_manager_lock = threading.Lock()

def get_gpu_manager() -> MultiAgentGPUManager:
    """Get the global GPU manager instance"""
    global _global_gpu_manager
    with _manager_lock:
        if _global_gpu_manager is None:
            _global_gpu_manager = MultiAgentGPUManager()
        return _global_gpu_manager

def request_agent_gpu(agent_name: str, memory_gb: float = 4.0, model_type: str = "general") -> dict[str, Any]:
    """Request GPU allocation for an agent (compatibility function)"""
    manager = get_gpu_manager()
    return manager.request_gpu_allocation(agent_name, memory_gb, model_type=model_type)

def release_agent_gpu(agent_name: str) -> bool:
    """Release GPU allocation for an agent (compatibility function)"""
    manager = get_gpu_manager()
    return manager.release_gpu_allocation(agent_name)

# Initialize manager on import
_gpu_manager = get_gpu_manager()
