"""
Hardware Detection and Optimization Module
Intelligently detects available hardware and optimizes parallelization accordingly
"""

import os
import psutil
import logging
import platform
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeviceType(Enum):
    """Enumeration for device types"""
    CPU = "cpu"
    GPU_NVIDIA = "gpu_nvidia"
    GPU_AMD = "gpu_amd"
    GPU_INTEL = "gpu_intel"
    TPU = "tpu"

class EnvironmentType(Enum):
    """Enumeration for environment types"""
    LOCAL_LAPTOP = "local_laptop"
    COLAB = "colab"
    SAGEMAKER = "sagemaker"
    CLUSTER = "cluster"
    UNKNOWN = "unknown"

@dataclass
class HardwareSpec:
    """Data class for hardware specifications"""
    device_type: DeviceType
    device_name: str
    memory_gb: float
    cores: int
    compute_capability: Optional[str] = None
    environment: EnvironmentType = EnvironmentType.UNKNOWN

@dataclass
class OptimizationConfig:
    """Configuration for hardware optimization"""
    max_batch_size: int
    max_workers: int
    memory_fraction: float
    use_mixed_precision: bool
    parallel_models: int
    device_placement: str

class HardwareDetector:
    """
    Advanced hardware detection and optimization system
    Supports CPU, NVIDIA/AMD/Intel GPUs, and various environments
    """
    
    def __init__(self):
        self.detected_devices = []
        self.environment = self._detect_environment()
        self.primary_device = None
        self._initialize_device_detection()
    
    def _detect_environment(self) -> EnvironmentType:
        """Detect the execution environment"""
        try:
            # Check for Google Colab
            import google.colab
            return EnvironmentType.COLAB
        except ImportError:
            pass
        
        # Check for AWS SageMaker
        if os.path.exists('/opt/ml') or 'SAGEMAKER_JOB_NAME' in os.environ:
            return EnvironmentType.SAGEMAKER
        
        # Check for cluster environments
        if any(env in os.environ for env in ['SLURM_JOB_ID', 'PBS_JOBID', 'LSB_JOBID']):
            return EnvironmentType.CLUSTER
        
        # Default to local laptop
        return EnvironmentType.LOCAL_LAPTOP
    
    def _initialize_device_detection(self):
        """Initialize and detect all available devices"""
        logger.info(f"Initializing hardware detection for {self.environment.value} environment")
        
        # Detect CPU
        self._detect_cpu()
        
        # Detect GPUs
        self._detect_nvidia_gpu()
        self._detect_amd_gpu()
        self._detect_intel_gpu()
        
        # Detect TPUs (for Colab/Cloud environments)
        if self.environment in [EnvironmentType.COLAB, EnvironmentType.CLUSTER]:
            self._detect_tpu()
        
        # Set primary device
        self._set_primary_device()
    
    def _detect_cpu(self):
        """Detect CPU specifications"""
        try:
            cpu_count = psutil.cpu_count(logical=True)
            cpu_info = platform.processor()
            memory_info = psutil.virtual_memory()
            
            cpu_spec = HardwareSpec(
                device_type=DeviceType.CPU,
                device_name=cpu_info if cpu_info else f"{cpu_count}-core CPU",
                memory_gb=memory_info.total / (1024**3),
                cores=cpu_count,
                environment=self.environment
            )
            
            self.detected_devices.append(cpu_spec)
            logger.info(f"Detected CPU: {cpu_spec.device_name} with {cpu_spec.cores} cores, {cpu_spec.memory_gb:.1f}GB RAM")
            
        except Exception as e:
            logger.warning(f"Failed to detect CPU: {e}")
    
    def _detect_nvidia_gpu(self):
        """Detect NVIDIA GPUs using multiple methods"""
        try:
            # Method 1: Try TensorFlow GPU detection
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            
            for i, gpu in enumerate(gpus):
                try:
                    # Get GPU details
                    gpu_details = tf.config.experimental.get_device_details(gpu)
                    gpu_name = gpu_details.get('device_name', f'GPU_{i}')
                    
                    # Try to get memory info
                    memory_gb = self._get_gpu_memory(gpu_name)
                    compute_capability = gpu_details.get('compute_capability', 'Unknown')
                    
                    gpu_spec = HardwareSpec(
                        device_type=DeviceType.GPU_NVIDIA,
                        device_name=gpu_name,
                        memory_gb=memory_gb,
                        cores=1,  # GPU cores are different, using 1 as placeholder
                        compute_capability=compute_capability,
                        environment=self.environment
                    )
                    
                    self.detected_devices.append(gpu_spec)
                    logger.info(f"Detected NVIDIA GPU: {gpu_name}, {memory_gb}GB, Compute Capability: {compute_capability}")
                    
                except Exception as gpu_e:
                    logger.warning(f"Failed to get details for GPU {i}: {gpu_e}")
            
        except ImportError:
            logger.info("TensorFlow not available for GPU detection")
        except Exception as e:
            logger.warning(f"Failed to detect NVIDIA GPUs with TensorFlow: {e}")
        
        # Method 2: Try nvidia-ml-py (if available)
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_gb = memory_info.total / (1024**3)
                
                # Check if already detected by TensorFlow
                if not any(gpu.device_name == name for gpu in self.detected_devices if gpu.device_type == DeviceType.GPU_NVIDIA):
                    gpu_spec = HardwareSpec(
                        device_type=DeviceType.GPU_NVIDIA,
                        device_name=name,
                        memory_gb=memory_gb,
                        cores=1,
                        environment=self.environment
                    )
                    
                    self.detected_devices.append(gpu_spec)
                    logger.info(f"Detected NVIDIA GPU via pynvml: {name}, {memory_gb:.1f}GB")
            
        except ImportError:
            pass  # pynvml not available
        except Exception as e:
            logger.warning(f"Failed to detect NVIDIA GPUs with pynvml: {e}")
    
    def _detect_amd_gpu(self):
        """Detect AMD GPUs (basic detection)"""
        try:
            # Basic AMD GPU detection using system info
            # This is a simplified version - full AMD support would require ROCm
            if platform.system() == "Linux":
                import subprocess
                result = subprocess.run(['lspci'], capture_output=True, text=True)
                if 'AMD' in result.stdout and ('VGA' in result.stdout or 'Display' in result.stdout):
                    logger.info("AMD GPU detected via lspci (basic detection)")
                    # Add basic AMD GPU spec
                    amd_spec = HardwareSpec(
                        device_type=DeviceType.GPU_AMD,
                        device_name="AMD GPU (detected)",
                        memory_gb=4.0,  # Default assumption
                        cores=1,
                        environment=self.environment
                    )
                    self.detected_devices.append(amd_spec)
        except Exception as e:
            logger.debug(f"AMD GPU detection failed: {e}")
    
    def _detect_intel_gpu(self):
        """Detect Intel GPUs (basic detection)"""
        try:
            # Basic Intel GPU detection
            if platform.system() == "Linux":
                import subprocess
                result = subprocess.run(['lspci'], capture_output=True, text=True)
                if 'Intel' in result.stdout and ('VGA' in result.stdout or 'Display' in result.stdout):
                    logger.info("Intel GPU detected via lspci (basic detection)")
                    intel_spec = HardwareSpec(
                        device_type=DeviceType.GPU_INTEL,
                        device_name="Intel GPU (detected)",
                        memory_gb=2.0,  # Default assumption
                        cores=1,
                        environment=self.environment
                    )
                    self.detected_devices.append(intel_spec)
        except Exception as e:
            logger.debug(f"Intel GPU detection failed: {e}")
    
    def _detect_tpu(self):
        """Detect TPUs in cloud environments"""
        try:
            # Check for TPU in Colab
            if self.environment == EnvironmentType.COLAB:
                import subprocess
                result = subprocess.run(['ls', '/dev/'], capture_output=True, text=True)
                if 'accel' in result.stdout:
                    logger.info("TPU detected in Colab environment")
                    tpu_spec = HardwareSpec(
                        device_type=DeviceType.TPU,
                        device_name="Cloud TPU",
                        memory_gb=8.0,  # Typical TPU memory
                        cores=8,  # TPU cores
                        environment=self.environment
                    )
                    self.detected_devices.append(tpu_spec)
        except Exception as e:
            logger.debug(f"TPU detection failed: {e}")
    
    def _get_gpu_memory(self, gpu_name: str) -> float:
        """Get GPU memory in GB"""
        try:
            import tensorflow as tf
            gpu_devices = tf.config.list_physical_devices('GPU')
            if gpu_devices:
                # Try to get memory limit (this might not work in all cases)
                return 8.0  # Default assumption for most modern GPUs
        except Exception:
            pass
        
        # GPU memory estimation based on common models
        gpu_memory_map = {
            'rtx 3050': 4.0,
            'rtx 4060': 8.0,
            'rtx 3060': 8.0,
            'rtx 3070': 8.0,
            'rtx 3080': 10.0,
            'rtx 4070': 12.0,
            'rtx 4080': 16.0,
            'rtx 4090': 24.0,
            't4': 15.0,  # Tesla T4
            'v100': 32.0,  # Tesla V100
            'a100': 40.0,  # A100
        }
        
        gpu_name_lower = gpu_name.lower()
        for key, memory in gpu_memory_map.items():
            if key in gpu_name_lower:
                return memory
        
        return 4.0  # Default fallback
    
    def _set_primary_device(self):
        """Set the primary device based on priority: GPU > TPU > CPU"""
        if not self.detected_devices:
            logger.warning("No devices detected!")
            return
        
        # Priority: High-end GPUs > TPUs > Standard GPUs > CPUs
        gpu_devices = [d for d in self.detected_devices if d.device_type in [DeviceType.GPU_NVIDIA, DeviceType.GPU_AMD, DeviceType.GPU_INTEL]]
        tpu_devices = [d for d in self.detected_devices if d.device_type == DeviceType.TPU]
        cpu_devices = [d for d in self.detected_devices if d.device_type == DeviceType.CPU]
        
        if gpu_devices:
            # Prefer GPU with more memory
            self.primary_device = max(gpu_devices, key=lambda x: x.memory_gb)
        elif tpu_devices:
            self.primary_device = tpu_devices[0]
        elif cpu_devices:
            self.primary_device = cpu_devices[0]
        
        logger.info(f"Primary device set to: {self.primary_device.device_name}")
    
    def get_optimization_config(self) -> OptimizationConfig:
        """Get optimization configuration based on detected hardware"""
        if not self.primary_device:
            # Fallback configuration for CPU-only
            return OptimizationConfig(
                max_batch_size=32,
                max_workers=2,
                memory_fraction=0.5,
                use_mixed_precision=False,
                parallel_models=1,
                device_placement="CPU"
            )
        
        # GPU optimizations
        if self.primary_device.device_type in [DeviceType.GPU_NVIDIA, DeviceType.GPU_AMD, DeviceType.GPU_INTEL]:
            return self._get_gpu_optimization_config()
        
        # TPU optimizations
        elif self.primary_device.device_type == DeviceType.TPU:
            return self._get_tpu_optimization_config()
        
        # CPU optimizations
        else:
            return self._get_cpu_optimization_config()
    
    def _get_gpu_optimization_config(self) -> OptimizationConfig:
        """Get GPU-specific optimization configuration"""
        memory_gb = self.primary_device.memory_gb
        gpu_name = self.primary_device.device_name.lower()
        
        # Specific optimizations for common GPUs
        if 'rtx 3050' in gpu_name or 'rtx 4060' in gpu_name:
            # Mid-range local laptop GPUs
            return OptimizationConfig(
                max_batch_size=64,
                max_workers=2,
                memory_fraction=0.8,
                use_mixed_precision=True,
                parallel_models=1,
                device_placement="GPU"
            )
        
        elif 'rtx 4060' in gpu_name and self.environment == EnvironmentType.LOCAL_LAPTOP:
            # RTX 4060 laptop optimization
            return OptimizationConfig(
                max_batch_size=128,
                max_workers=3,
                memory_fraction=0.85,
                use_mixed_precision=True,
                parallel_models=1,
                device_placement="GPU"
            )
        
        elif 't4' in gpu_name and self.environment in [EnvironmentType.COLAB, EnvironmentType.SAGEMAKER]:
            # Tesla T4 in cloud environments
            return OptimizationConfig(
                max_batch_size=256,
                max_workers=4,
                memory_fraction=0.9,
                use_mixed_precision=True,
                parallel_models=2,
                device_placement="GPU"
            )
        
        else:
            # General GPU configuration based on memory
            if memory_gb >= 16:
                batch_size = 512
                workers = 6
                parallel = 3
            elif memory_gb >= 8:
                batch_size = 256
                workers = 4
                parallel = 2
            else:
                batch_size = 128
                workers = 2
                parallel = 1
            
            return OptimizationConfig(
                max_batch_size=batch_size,
                max_workers=workers,
                memory_fraction=0.8,
                use_mixed_precision=True,
                parallel_models=parallel,
                device_placement="GPU"
            )
    
    def _get_tpu_optimization_config(self) -> OptimizationConfig:
        """Get TPU-specific optimization configuration"""
        return OptimizationConfig(
            max_batch_size=1024,
            max_workers=8,
            memory_fraction=0.9,
            use_mixed_precision=True,
            parallel_models=4,
            device_placement="TPU"
        )
    
    def _get_cpu_optimization_config(self) -> OptimizationConfig:
        """Get CPU-specific optimization configuration"""
        cores = self.primary_device.cores
        memory_gb = self.primary_device.memory_gb
        
        # Conservative settings for CPU
        max_workers = min(cores // 2, 4)  # Don't overload CPU
        batch_size = min(64, int(memory_gb * 8))  # Conservative batch size
        
        return OptimizationConfig(
            max_batch_size=batch_size,
            max_workers=max_workers,
            memory_fraction=0.6,  # Leave room for system
            use_mixed_precision=False,
            parallel_models=1,
            device_placement="CPU"
        )
    
    def get_device_summary(self) -> Dict:
        """Get a summary of all detected devices"""
        return {
            "environment": self.environment.value,
            "primary_device": {
                "type": self.primary_device.device_type.value if self.primary_device else None,
                "name": self.primary_device.device_name if self.primary_device else None,
                "memory_gb": self.primary_device.memory_gb if self.primary_device else None
            },
            "all_devices": [
                {
                    "type": device.device_type.value,
                    "name": device.device_name,
                    "memory_gb": device.memory_gb,
                    "cores": device.cores,
                    "compute_capability": device.compute_capability
                }
                for device in self.detected_devices
            ],
            "optimization_config": self.get_optimization_config().__dict__
        }
    
    def setup_tensorflow_gpu(self):
        """Configure TensorFlow for optimal GPU usage"""
        try:
            import tensorflow as tf
            
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                config = self.get_optimization_config()
                
                # Configure memory growth
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                # Set memory limit if needed
                if config.memory_fraction < 1.0:
                    memory_limit = int(self.primary_device.memory_gb * 1024 * config.memory_fraction)
                    tf.config.experimental.set_virtual_device_configuration(
                        gpus[0],
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)]
                    )
                
                # Enable mixed precision if supported
                if config.use_mixed_precision:
                    try:
                        policy = tf.keras.mixed_precision.Policy('mixed_float16')
                        tf.keras.mixed_precision.set_global_policy(policy)
                        logger.info("Mixed precision enabled for faster training")
                    except Exception as e:
                        logger.warning(f"Failed to enable mixed precision: {e}")
                
                logger.info(f"TensorFlow GPU configuration completed for {len(gpus)} GPU(s)")
            else:
                logger.info("No GPUs detected for TensorFlow configuration")
                
        except ImportError:
            logger.warning("TensorFlow not available for GPU configuration")
        except Exception as e:
            logger.error(f"Failed to configure TensorFlow GPU: {e}")

# Factory function for easy instantiation
def create_hardware_detector() -> HardwareDetector:
    """Factory function to create and initialize hardware detector"""
    return HardwareDetector()

# Convenience function for quick optimization config
def get_quick_optimization_config() -> OptimizationConfig:
    """Quick function to get optimization config without full detection"""
    detector = create_hardware_detector()
    return detector.get_optimization_config()