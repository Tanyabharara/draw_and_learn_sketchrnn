"""
Core Benchmarking Engine with Hardware-Aware Parallel Execution
Orchestrates model evaluation with intelligent resource management
"""

import os
import sys
import time
import json
import logging
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, Future
from multiprocessing import Manager, Queue, Value
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from pathlib import Path
import numpy as np
from datetime import datetime

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interfaces import IModelPlugin, BenchmarkMetrics, Prediction, ProcessedInput, ModelInfo
from core.model_registry import ModelRegistry, get_global_registry
from utils.hardware_detector import HardwareDetector, OptimizationConfig, create_hardware_detector

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution"""
    test_data_path: Optional[str] = None
    test_labels_path: Optional[str] = None
    batch_size: int = 32
    max_samples: Optional[int] = None
    warmup_runs: int = 3
    timing_runs: int = 10
    parallel_models: bool = True
    max_workers: Optional[int] = None
    memory_limit_mb: Optional[float] = None
    timeout_seconds: int = 300
    save_predictions: bool = False
    save_detailed_metrics: bool = True
    output_dir: str = "benchmark_results"
    enable_profiling: bool = False
    validate_predictions: bool = True
    
@dataclass
class ModelBenchmarkResult:
    """Results for a single model benchmark"""
    model_id: str
    model_name: str
    model_info: Optional[ModelInfo] = None
    metrics: Optional[BenchmarkMetrics] = None
    execution_time: float = 0.0
    memory_peak_mb: Optional[float] = None
    error: Optional[str] = None
    predictions: Optional[List[Prediction]] = None
    profiling_data: Optional[Dict[str, Any]] = None
    benchmark_timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = asdict(self)
        result['benchmark_timestamp'] = self.benchmark_timestamp.isoformat()
        if self.model_info:
            result['model_info'] = self.model_info.to_dict()
        if self.metrics:
            result['metrics'] = self.metrics.to_dict()
        if self.predictions:
            result['predictions'] = [p.to_dict() for p in self.predictions]
        return result

@dataclass
class BenchmarkSummary:
    """Summary of complete benchmark run"""
    total_models: int
    successful_models: int
    failed_models: int
    total_execution_time: float
    hardware_info: Dict[str, Any]
    config: BenchmarkConfig
    results: List[ModelBenchmarkResult]
    start_time: datetime
    end_time: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'total_models': self.total_models,
            'successful_models': self.successful_models, 
            'failed_models': self.failed_models,
            'total_execution_time': self.total_execution_time,
            'hardware_info': self.hardware_info,
            'config': asdict(self.config),
            'results': [r.to_dict() for r in self.results],
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat()
        }

class MemoryMonitor:
    """Thread-safe memory monitoring utility"""
    
    def __init__(self):
        self._peak_memory = Value('f', 0.0)
        self._monitoring = Value('b', False)
        self._monitor_thread = None
    
    def start_monitoring(self):
        """Start memory monitoring in background thread"""
        self._monitoring.value = True
        self._peak_memory.value = 0.0
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self) -> float:
        """Stop monitoring and return peak memory usage in MB"""
        self._monitoring.value = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        return self._peak_memory.value
    
    def _monitor_loop(self):
        """Memory monitoring loop"""
        try:
            import psutil
            process = psutil.Process()
            
            while self._monitoring.value:
                try:
                    memory_mb = process.memory_info().rss / (1024 * 1024)
                    if memory_mb > self._peak_memory.value:
                        self._peak_memory.value = memory_mb
                    time.sleep(0.1)  # Check every 100ms
                except psutil.NoSuchProcess:
                    break
                except Exception as e:
                    logger.debug(f"Memory monitoring error: {e}")
                    
        except ImportError:
            logger.warning("psutil not available for memory monitoring")

class BenchmarkEngine:
    """
    Core benchmarking engine with hardware-aware parallel execution
    """
    
    def __init__(self, 
                 registry: Optional[ModelRegistry] = None,
                 hardware_detector: Optional[HardwareDetector] = None):
        self.registry = registry or get_global_registry()
        self.hardware_detector = hardware_detector or create_hardware_detector()
        self.optimization_config = self.hardware_detector.get_optimization_config()
        
        # Setup TensorFlow GPU if available
        self.hardware_detector.setup_tensorflow_gpu()
        
        # Thread safety
        self._lock = threading.RLock()
        self._active_benchmarks = 0
        
        # Configure based on hardware
        self._configure_execution()
        
        logger.info(f"Benchmark engine initialized with {self.optimization_config.device_placement} optimization")
    
    def _configure_execution(self):
        """Configure execution parameters based on hardware"""
        # Set default worker counts based on hardware
        if self.optimization_config.device_placement == "GPU":
            # GPU: fewer workers to avoid memory contention
            self.default_max_workers = min(self.optimization_config.max_workers, 2)
        else:
            # CPU: can handle more parallel workers
            self.default_max_workers = self.optimization_config.max_workers
        
        # Memory management
        self.max_memory_per_model = self.optimization_config.memory_fraction * 1024  # MB
        
        logger.info(f"Execution config - Workers: {self.default_max_workers}, "
                   f"Memory per model: {self.max_memory_per_model:.1f}MB")
    
    def benchmark_models(self, 
                        model_ids: List[str],
                        test_data: np.ndarray,
                        test_labels: np.ndarray,
                        config: Optional[BenchmarkConfig] = None) -> BenchmarkSummary:
        """
        Benchmark multiple models with parallel execution
        
        Args:
            model_ids: List of model identifiers to benchmark
            test_data: Test dataset
            test_labels: Test labels
            config: Benchmark configuration
            
        Returns:
            BenchmarkSummary with results for all models
        """
        start_time = datetime.now()
        config = config or BenchmarkConfig()
        
        logger.info(f"Starting benchmark of {len(model_ids)} models")
        
        # Prepare output directory
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate and prepare test data
        test_data, test_labels = self._prepare_test_data(test_data, test_labels, config)
        
        # Determine execution strategy
        use_parallel = (
            config.parallel_models and 
            len(model_ids) > 1 and 
            self.optimization_config.parallel_models > 1
        )
        
        if use_parallel:
            results = self._benchmark_models_parallel(model_ids, test_data, test_labels, config)
        else:
            results = self._benchmark_models_sequential(model_ids, test_data, test_labels, config)
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # Create summary
        successful = len([r for r in results if r.error is None])
        failed = len(results) - successful
        
        summary = BenchmarkSummary(
            total_models=len(model_ids),
            successful_models=successful,
            failed_models=failed,
            total_execution_time=total_time,
            hardware_info=self.hardware_detector.get_device_summary(),
            config=config,
            results=results,
            start_time=start_time,
            end_time=end_time
        )
        
        # Save results
        self._save_results(summary, output_dir)
        
        logger.info(f"Benchmark completed in {total_time:.2f}s - "
                   f"{successful} successful, {failed} failed")
        
        return summary
    
    def _benchmark_models_parallel(self, 
                                  model_ids: List[str],
                                  test_data: np.ndarray,
                                  test_labels: np.ndarray,
                                  config: BenchmarkConfig) -> List[ModelBenchmarkResult]:
        """Parallel model benchmarking with resource management"""
        max_workers = min(
            config.max_workers or self.default_max_workers,
            self.optimization_config.parallel_models,
            len(model_ids)
        )
        
        logger.info(f"Running parallel benchmark with {max_workers} workers")
        
        results = []
        
        # Use ThreadPoolExecutor for I/O bound tasks (model loading)
        # GPU memory is shared between threads in same process
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all benchmark tasks
            future_to_model = {}
            for model_id in model_ids:
                future = executor.submit(
                    self._benchmark_single_model_safe,
                    model_id, test_data, test_labels, config
                )
                future_to_model[future] = model_id
            
            # Collect results as they complete
            for future in as_completed(future_to_model, timeout=config.timeout_seconds):
                model_id = future_to_model[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Completed benchmark for {model_id}")
                except Exception as e:
                    logger.error(f"Benchmark failed for {model_id}: {str(e)}")
                    results.append(ModelBenchmarkResult(
                        model_id=model_id,
                        model_name=model_id,
                        error=str(e)
                    ))
        
        return results
    
    def _benchmark_models_sequential(self, 
                                   model_ids: List[str],
                                   test_data: np.ndarray,
                                   test_labels: np.ndarray,
                                   config: BenchmarkConfig) -> List[ModelBenchmarkResult]:
        """Sequential model benchmarking"""
        logger.info("Running sequential benchmark")
        
        results = []
        for i, model_id in enumerate(model_ids):
            logger.info(f"Benchmarking model {i+1}/{len(model_ids)}: {model_id}")
            
            try:
                result = self._benchmark_single_model_safe(model_id, test_data, test_labels, config)
                results.append(result)
            except Exception as e:
                logger.error(f"Benchmark failed for {model_id}: {str(e)}")
                results.append(ModelBenchmarkResult(
                    model_id=model_id,
                    model_name=model_id,
                    error=str(e)
                ))
        
        return results
    
    def _benchmark_single_model_safe(self, 
                                   model_id: str,
                                   test_data: np.ndarray,
                                   test_labels: np.ndarray,
                                   config: BenchmarkConfig) -> ModelBenchmarkResult:
        """
        Safely benchmark a single model with error handling and resource management
        """
        memory_monitor = MemoryMonitor()
        start_time = time.time()
        
        try:
            memory_monitor.start_monitoring()
            
            # Load model
            plugin = self.registry.get_model(model_id)
            if plugin is None:
                raise RuntimeError(f"Failed to load model: {model_id}")
            
            # Get model info
            model_info = plugin.get_model_info()
            
            # Run benchmark
            metrics, predictions = self._run_model_benchmark(plugin, test_data, test_labels, config)
            
            execution_time = time.time() - start_time
            peak_memory = memory_monitor.stop_monitoring()
            
            return ModelBenchmarkResult(
                model_id=model_id,
                model_name=model_info.name,
                model_info=model_info,
                metrics=metrics,
                execution_time=execution_time,
                memory_peak_mb=peak_memory,
                predictions=predictions if config.save_predictions else None
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            peak_memory = memory_monitor.stop_monitoring()
            
            logger.error(f"Model benchmark failed for {model_id}: {str(e)}")
            logger.debug(f"Error traceback: {traceback.format_exc()}")
            
            return ModelBenchmarkResult(
                model_id=model_id,
                model_name=model_id,
                execution_time=execution_time,
                memory_peak_mb=peak_memory,
                error=str(e)
            )
        
        finally:
            # Cleanup
            try:
                if 'plugin' in locals():
                    plugin.cleanup()
            except:
                pass
    
    def _run_model_benchmark(self, 
                           plugin: IModelPlugin,
                           test_data: np.ndarray,
                           test_labels: np.ndarray,
                           config: BenchmarkConfig) -> Tuple[BenchmarkMetrics, List[Prediction]]:
        """
        Run the actual benchmark for a model
        
        Args:
            plugin: Model plugin to benchmark
            test_data: Test dataset
            test_labels: Test labels
            config: Benchmark configuration
            
        Returns:
            Tuple of (metrics, predictions)
        """
        logger.debug(f"Running benchmark for {plugin.get_model_info().name}")
        
        # Warmup runs
        if config.warmup_runs > 0:
            logger.debug(f"Running {config.warmup_runs} warmup iterations")
            warmup_data = test_data[:min(config.warmup_runs, len(test_data))]
            for data_point in warmup_data:
                try:
                    processed = plugin.preprocess_input(data_point)
                    plugin.predict(processed)
                except Exception as e:
                    logger.debug(f"Warmup iteration failed: {e}")
        
        # Actual benchmark runs
        predictions = []
        preprocessing_times = []
        inference_times = []
        
        # Limit test data if specified
        if config.max_samples:
            test_data = test_data[:config.max_samples]
            test_labels = test_labels[:config.max_samples]
        
        logger.debug(f"Running benchmark on {len(test_data)} samples")
        
        # Process in batches for memory efficiency
        batch_size = min(config.batch_size, self.optimization_config.max_batch_size)
        
        for i in range(0, len(test_data), batch_size):
            batch_data = test_data[i:i + batch_size]
            batch_predictions = []
            
            # Process each sample in the batch
            for data_point in batch_data:
                try:
                    # Preprocess
                    processed = plugin.preprocess_input(data_point)
                    preprocessing_times.append(processed.preprocessing_time)
                    
                    # Predict
                    prediction = plugin.predict(processed)
                    inference_times.append(prediction.inference_time)
                    predictions.append(prediction)
                    batch_predictions.append(prediction)
                    
                except Exception as e:
                    logger.warning(f"Prediction failed for sample {i}: {e}")
                    # Create dummy prediction to maintain alignment
                    predictions.append(Prediction(
                        raw_output=np.zeros(28),  # Assuming 28 classes
                        probabilities=np.zeros(28),
                        predicted_classes=["error"],
                        confidence_scores=np.array([0.0]),
                        inference_time=0.0,
                        metadata={"error": str(e)}
                    ))
        
        # Calculate metrics
        metrics = self._calculate_metrics(predictions, test_labels, plugin.get_model_info())
        
        # Add timing statistics to metrics
        if preprocessing_times:
            metrics.preprocessing_time_mean = np.mean(preprocessing_times)
            metrics.preprocessing_time_std = np.std(preprocessing_times)
        
        if inference_times:
            metrics.inference_time_mean = np.mean(inference_times)
            metrics.inference_time_std = np.std(inference_times)
            # Calculate throughput (samples per second)
            total_inference_time = sum(inference_times)
            metrics.throughput = len(predictions) / total_inference_time if total_inference_time > 0 else 0.0
        
        return metrics, predictions
    
    def _calculate_metrics(self, 
                          predictions: List[Prediction],
                          test_labels: np.ndarray,
                          model_info: ModelInfo) -> BenchmarkMetrics:
        """
        Calculate comprehensive benchmark metrics
        
        Args:
            predictions: List of model predictions
            test_labels: Ground truth labels
            model_info: Model information
            
        Returns:
            BenchmarkMetrics object
        """
        try:
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, f1_score,
                classification_report, confusion_matrix
            )
        except ImportError:
            logger.error("scikit-learn not available for metrics calculation")
            return BenchmarkMetrics(
                model_name=model_info.name,
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                inference_time_mean=0.0,
                inference_time_std=0.0,
                preprocessing_time_mean=0.0,
                preprocessing_time_std=0.0,
                throughput=0.0,
                metadata={"error": "scikit-learn not available"}
            )
        
        # Extract predicted classes
        predicted_classes = []
        for pred in predictions:
            if pred.predicted_classes and len(pred.predicted_classes) > 0:
                # Try to convert to integer if it's a string number
                try:
                    predicted_classes.append(int(pred.predicted_classes[0]))
                except (ValueError, TypeError):
                    # If conversion fails, use 0 as default
                    predicted_classes.append(0)
            else:
                predicted_classes.append(0)
        
        predicted_classes = np.array(predicted_classes)
        
        # Ensure we have the same number of predictions and labels
        min_length = min(len(predicted_classes), len(test_labels))
        predicted_classes = predicted_classes[:min_length]
        test_labels = test_labels[:min_length]
        
        # Calculate basic metrics
        accuracy = accuracy_score(test_labels, predicted_classes)
        precision = precision_score(test_labels, predicted_classes, average='weighted', zero_division=0)
        recall = recall_score(test_labels, predicted_classes, average='weighted', zero_division=0)
        f1 = f1_score(test_labels, predicted_classes, average='weighted', zero_division=0)
        
        # Per-class metrics
        try:
            report = classification_report(test_labels, predicted_classes, output_dict=True, zero_division=0)
            per_class_metrics = {}
            for class_label, metrics_dict in report.items():
                if isinstance(metrics_dict, dict) and class_label not in ['accuracy', 'macro avg', 'weighted avg']:
                    per_class_metrics[str(class_label)] = metrics_dict
        except Exception as e:
            logger.warning(f"Failed to calculate per-class metrics: {e}")
            per_class_metrics = None
        
        # Confusion matrix
        try:
            conf_matrix = confusion_matrix(test_labels, predicted_classes)
        except Exception as e:
            logger.warning(f"Failed to calculate confusion matrix: {e}")
            conf_matrix = None
        
        return BenchmarkMetrics(
            model_name=model_info.name,
            accuracy=float(accuracy),
            precision=float(precision),
            recall=float(recall),
            f1_score=float(f1),
            inference_time_mean=0.0,  # Will be filled by caller
            inference_time_std=0.0,   # Will be filled by caller
            preprocessing_time_mean=0.0,  # Will be filled by caller
            preprocessing_time_std=0.0,   # Will be filled by caller
            throughput=0.0,  # Will be filled by caller
            per_class_metrics=per_class_metrics,
            confusion_matrix=conf_matrix,
            metadata={
                "model_type": model_info.model_type.value,
                "total_samples": len(predictions),
                "valid_predictions": len([p for p in predictions if "error" not in p.metadata])
            }
        )
    
    def _prepare_test_data(self, 
                          test_data: np.ndarray,
                          test_labels: np.ndarray,
                          config: BenchmarkConfig) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare and validate test data
        
        Args:
            test_data: Raw test data
            test_labels: Test labels
            config: Benchmark configuration
            
        Returns:
            Tuple of (prepared_data, prepared_labels)
        """
        logger.debug(f"Preparing test data: {test_data.shape}, labels: {test_labels.shape}")
        
        # Validate data alignment
        if len(test_data) != len(test_labels):
            min_length = min(len(test_data), len(test_labels))
            test_data = test_data[:min_length]
            test_labels = test_labels[:min_length]
            logger.warning(f"Data/label length mismatch, truncated to {min_length}")
        
        # Apply max_samples limit
        if config.max_samples and len(test_data) > config.max_samples:
            test_data = test_data[:config.max_samples]
            test_labels = test_labels[:config.max_samples]
            logger.info(f"Limited test data to {config.max_samples} samples")
        
        return test_data, test_labels
    
    def _save_results(self, summary: BenchmarkSummary, output_dir: Path):
        """Save benchmark results to files"""
        try:
            # Save main summary
            summary_file = output_dir / f"benchmark_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(summary_file, 'w') as f:
                json.dump(summary.to_dict(), f, indent=2)
            
            logger.info(f"Benchmark summary saved to: {summary_file}")
            
            # Save individual model results
            for result in summary.results:
                if result.error is None:
                    result_file = output_dir / f"model_{result.model_id}_results.json"
                    with open(result_file, 'w') as f:
                        json.dump(result.to_dict(), f, indent=2)
            
            # Save hardware info
            hardware_file = output_dir / "hardware_info.json"
            with open(hardware_file, 'w') as f:
                json.dump(summary.hardware_info, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
    
    def get_optimization_recommendations(self, summary: BenchmarkSummary) -> Dict[str, Any]:
        """
        Generate optimization recommendations based on benchmark results
        
        Args:
            summary: Benchmark summary
            
        Returns:
            Dictionary with optimization recommendations
        """
        recommendations = {
            "hardware_optimization": [],
            "model_optimization": [],
            "system_optimization": []
        }
        
        # Hardware recommendations
        if self.optimization_config.device_placement == "CPU":
            recommendations["hardware_optimization"].append(
                "Consider using GPU acceleration for faster inference"
            )
        
        # Model-specific recommendations
        successful_results = [r for r in summary.results if r.error is None]
        if successful_results:
            # Find slowest models
            slow_models = sorted(successful_results, key=lambda x: x.metrics.inference_time_mean, reverse=True)[:3]
            for model in slow_models:
                recommendations["model_optimization"].append(
                    f"Model {model.model_name} has high inference time: {model.metrics.inference_time_mean:.4f}s"
                )
            
            # Find models with low accuracy
            low_accuracy_models = [r for r in successful_results if r.metrics.accuracy < 0.8]
            for model in low_accuracy_models:
                recommendations["model_optimization"].append(
                    f"Model {model.model_name} has low accuracy: {model.metrics.accuracy:.3f}"
                )
        
        # System recommendations
        if summary.failed_models > 0:
            recommendations["system_optimization"].append(
                f"{summary.failed_models} models failed - check logs for issues"
            )
        
        return recommendations

# Convenience functions
def create_benchmark_engine(registry: Optional[ModelRegistry] = None) -> BenchmarkEngine:
    """Create a benchmark engine with default configuration"""
    return BenchmarkEngine(registry)

def quick_benchmark(model_ids: List[str], 
                   test_data: np.ndarray, 
                   test_labels: np.ndarray,
                   **config_kwargs) -> BenchmarkSummary:
    """
    Quick benchmark function for simple use cases
    
    Args:
        model_ids: List of model IDs to benchmark
        test_data: Test dataset
        test_labels: Test labels
        **config_kwargs: Additional configuration parameters
        
    Returns:
        BenchmarkSummary with results
    """
    config = BenchmarkConfig(**config_kwargs)
    engine = create_benchmark_engine()
    return engine.benchmark_models(model_ids, test_data, test_labels, config)