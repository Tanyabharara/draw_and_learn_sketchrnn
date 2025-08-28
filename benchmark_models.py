#!/usr/bin/env python3
"""
SketchRNN Model Benchmarking System
Comprehensive benchmarking tool for evaluating multiple models with hardware-aware optimization
"""

import os
import sys
import argparse
import logging
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

# Add benchmark system to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'benchmark_system'))

# Import benchmark system components
from benchmark_system.core.benchmark_engine import BenchmarkEngine, BenchmarkConfig
from benchmark_system.core.model_registry import get_global_registry, register_model
from benchmark_system.core.model_factory import create_model_plugin
from benchmark_system.utils.hardware_detector import create_hardware_detector
from benchmark_system.utils.visualization import create_visualizer
from benchmark_system.core.metrics import calculate_metrics

# Configure logging
def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('benchmark.log', mode='a')
        ]
    )

logger = logging.getLogger(__name__)

def load_test_data(data_path: str, labels_path: Optional[str] = None) -> tuple:
    """
    Load test data from various formats
    
    Args:
        data_path: Path to test data file
        labels_path: Optional path to labels file
        
    Returns:
        Tuple of (test_data, test_labels)
    """
    logger.info(f"Loading test data from {data_path}")
    
    data_path = Path(data_path)
    
    try:
        if data_path.suffix == '.pkl':
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                
            if isinstance(data, dict):
                # If data is a dictionary, try to extract test data
                if 'test_data' in data and 'test_labels' in data:
                    return data['test_data'], data['test_labels']
                elif 'X_test' in data and 'y_test' in data:
                    return data['X_test'], data['y_test']
                else:
                    # Try to load from categories (QuickDraw format)
                    return load_quickdraw_test_data(data)
            else:
                # Assume it's just data array
                test_data = data
                test_labels = load_labels_separately(labels_path) if labels_path else None
                return test_data, test_labels
                
        elif data_path.suffix == '.npz':
            data = np.load(data_path)
            test_data = data['test_data'] if 'test_data' in data else data['data']
            test_labels = data['test_labels'] if 'test_labels' in data else data['labels']
            return test_data, test_labels
            
        elif data_path.suffix == '.npy':
            test_data = np.load(data_path)
            test_labels = load_labels_separately(labels_path) if labels_path else None
            return test_data, test_labels
            
        else:
            raise ValueError(f"Unsupported data format: {data_path.suffix}")
            
    except Exception as e:
        logger.error(f"Failed to load test data: {str(e)}")
        raise

def load_quickdraw_test_data(data_dict: Dict[str, Any], samples_per_category: int = 100) -> tuple:
    """
    Load test data from QuickDraw format dictionary
    
    Args:
        data_dict: Dictionary with category names as keys
        samples_per_category: Number of samples per category for testing
        
    Returns:
        Tuple of (test_data, test_labels)
    """
    logger.info("Loading QuickDraw test data")
    
    # Load categories
    categories_file = Path('data/categories.json')
    if categories_file.exists():
        with open(categories_file, 'r') as f:
            categories = json.load(f)
    else:
        categories = list(data_dict.keys())
    
    test_data = []
    test_labels = []
    
    for i, category in enumerate(categories):
        if category in data_dict:
            category_data = data_dict[category]
            
            # Take samples for testing (from the end to avoid training data overlap)
            test_samples = category_data[-samples_per_category:] if len(category_data) >= samples_per_category else category_data
            
            test_data.extend(test_samples)
            test_labels.extend([i] * len(test_samples))
            
            logger.info(f"Loaded {len(test_samples)} test samples for {category}")
    
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)
    
    # Reshape for CNN if needed
    if len(test_data.shape) == 2:
        test_data = test_data.reshape(-1, 28, 28)
    
    logger.info(f"Total test data shape: {test_data.shape}, labels: {test_labels.shape}")
    
    return test_data, test_labels

def load_labels_separately(labels_path: str) -> np.ndarray:
    """Load labels from separate file"""
    labels_path = Path(labels_path)
    
    if labels_path.suffix == '.pkl':
        with open(labels_path, 'rb') as f:
            return pickle.load(f)
    elif labels_path.suffix == '.npy':
        return np.load(labels_path)
    elif labels_path.suffix == '.json':
        with open(labels_path, 'r') as f:
            return np.array(json.load(f))
    else:
        raise ValueError(f"Unsupported label format: {labels_path.suffix}")

def discover_models(model_dir: str = "models") -> List[str]:
    """
    Automatically discover model files in directory
    
    Args:
        model_dir: Directory to search for models
        
    Returns:
        List of discovered model IDs
    """
    logger.info(f"Discovering models in {model_dir}")
    
    model_dir = Path(model_dir)
    if not model_dir.exists():
        logger.warning(f"Model directory {model_dir} does not exist")
        return []
    
    registry = get_global_registry()
    discovered_models = []
    
    # Common model file extensions
    extensions = ['.h5', '.keras', '.pb', '.pth', '.pt', '.onnx', '.pkl']
    
    for ext in extensions:
        for model_file in model_dir.glob(f'*{ext}'):
            try:
                model_id = f"auto_{model_file.stem}"
                
                # Register the model
                success = register_model(
                    model_id=model_id,
                    name=model_file.stem,
                    model_path=str(model_file),
                    description=f"Auto-discovered model from {model_file}",
                    tags=['auto-discovered'],
                    categories_path='data/categories.json'  # Assume QuickDraw categories
                )
                
                if success:
                    discovered_models.append(model_id)
                    logger.info(f"Registered model: {model_id}")
                
            except Exception as e:
                logger.warning(f"Failed to register {model_file}: {str(e)}")
    
    logger.info(f"Discovered {len(discovered_models)} models")
    return discovered_models

def benchmark_models_cli(args):
    """Main benchmarking function called from CLI"""
    
    # Setup hardware detection
    logger.info("Initializing hardware detection...")
    hardware_detector = create_hardware_detector()
    hardware_summary = hardware_detector.get_device_summary()
    
    logger.info(f"Hardware Summary:")
    logger.info(f"  Environment: {hardware_summary['environment']}")
    logger.info(f"  Primary Device: {hardware_summary['primary_device']['name']}")
    logger.info(f"  Device Type: {hardware_summary['primary_device']['type']}")
    
    # Load test data
    if args.data_path:
        test_data, test_labels = load_test_data(args.data_path, args.labels_path)
    else:
        # Try to load default QuickDraw data
        default_data_path = Path('data/quickdraw_data.pkl')
        if default_data_path.exists():
            test_data, test_labels = load_test_data(str(default_data_path))
        else:
            logger.error("No test data specified and default data not found")
            return 1
    
    logger.info(f"Loaded test data: {test_data.shape}, labels: {test_labels.shape}")
    
    # Determine models to benchmark
    if args.model_ids:
        model_ids = args.model_ids
    elif args.discover_models:
        model_ids = discover_models(args.model_dir)
        if not model_ids:
            logger.error("No models discovered")
            return 1
    else:
        # Try to use existing registered models
        registry = get_global_registry()
        registrations = registry.list_models()
        model_ids = [reg.model_id for reg in registrations]
        
        if not model_ids:
            logger.error("No models found. Use --discover-models or specify --model-ids")
            return 1
    
    logger.info(f"Benchmarking {len(model_ids)} models: {model_ids}")
    
    # Configure benchmarking
    config = BenchmarkConfig(
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        warmup_runs=args.warmup_runs,
        timing_runs=args.timing_runs,
        parallel_models=args.parallel,
        max_workers=args.max_workers,
        timeout_seconds=args.timeout,
        save_predictions=args.save_predictions,
        save_detailed_metrics=args.detailed_metrics,
        output_dir=args.output_dir
    )
    
    # Create and run benchmark
    benchmark_engine = BenchmarkEngine(hardware_detector=hardware_detector)
    
    try:
        logger.info("Starting benchmark execution...")
        summary = benchmark_engine.benchmark_models(model_ids, test_data, test_labels, config)
        
        logger.info("Benchmark completed successfully!")
        logger.info(f"Results: {summary.successful_models} successful, {summary.failed_models} failed")
        
        # Generate reports
        if args.generate_reports:
            logger.info("Generating comprehensive reports...")
            visualizer = create_visualizer(args.output_dir)
            report_dir = visualizer.generate_comprehensive_report(summary)
            logger.info(f"Reports generated in: {report_dir}")
        
        # Print summary to console
        print_summary(summary)
        
        return 0
        
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")
        return 1

def print_summary(summary):
    """Print benchmark summary to console"""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    print(f"Total Models: {summary.total_models}")
    print(f"Successful: {summary.successful_models}")
    print(f"Failed: {summary.failed_models}")
    print(f"Total Time: {summary.total_execution_time:.2f}s")
    
    successful_results = [r for r in summary.results if r.error is None and r.metrics is not None]
    
    if successful_results:
        # Sort by accuracy
        successful_results.sort(key=lambda x: x.metrics.accuracy, reverse=True)
        
        print(f"\nTOP PERFORMERS:")
        print("-" * 40)
        
        for i, result in enumerate(successful_results[:5], 1):
            print(f"{i}. {result.model_name}")
            print(f"   Accuracy: {result.metrics.accuracy:.4f}")
            print(f"   Speed: {result.metrics.inference_time_mean:.4f}s")
            print(f"   Throughput: {result.metrics.throughput:.1f} samples/sec")
    
    failed_results = [r for r in summary.results if r.error is not None]
    if failed_results:
        print(f"\nFAILED MODELS:")
        print("-" * 40)
        for result in failed_results:
            print(f"• {result.model_name}: {result.error}")
    
    print("\n" + "="*80)

def register_model_cli(args):
    """Register a model via CLI"""
    try:
        success = register_model(
            model_id=args.model_id,
            name=args.name or args.model_id,
            model_path=args.model_path,
            version=args.version,
            description=args.description or "",
            tags=args.tags or [],
            categories_path=args.categories_path
        )
        
        if success:
            logger.info(f"Successfully registered model: {args.model_id}")
            return 0
        else:
            logger.error(f"Failed to register model: {args.model_id}")
            return 1
            
    except Exception as e:
        logger.error(f"Registration failed: {str(e)}")
        return 1

def list_models_cli(args):
    """List registered models via CLI"""
    registry = get_global_registry()
    models = registry.list_models()
    
    if not models:
        print("No models registered.")
        return 0
    
    print(f"\nREGISTERED MODELS ({len(models)} total)")
    print("="*80)
    
    for model in models:
        print(f"ID: {model.model_id}")
        print(f"Name: {model.name}")
        print(f"Type: {model.model_type.value}")
        print(f"Path: {model.model_path}")
        print(f"Registered: {model.registered_at.strftime('%Y-%m-%d %H:%M:%S')}")
        if model.tags:
            print(f"Tags: {', '.join(model.tags)}")
        print("-" * 40)
    
    return 0

def test_model_cli(args):
    """Test a single model via CLI"""
    try:
        # Load model
        plugin = create_model_plugin(args.model_path, categories_path=args.categories_path)
        
        # Load test data
        if args.data_path:
            test_data, test_labels = load_test_data(args.data_path, args.labels_path)
        else:
            # Create dummy test data
            test_data = np.random.rand(10, 28, 28).astype(np.float32)
            test_labels = np.random.randint(0, 28, 10)
        
        # Test preprocessing and inference
        print(f"Testing model: {plugin.get_model_info().name}")
        
        sample_data = test_data[0]
        processed = plugin.preprocess_input(sample_data)
        prediction = plugin.predict(processed)
        
        print(f"Model Info:")
        print(f"  Type: {plugin.get_model_info().model_type.value}")
        print(f"  Input Shape: {plugin.get_model_info().input_shape}")
        print(f"  Output Shape: {plugin.get_model_info().output_shape}")
        
        print(f"Test Results:")
        print(f"  Preprocessing Time: {processed.preprocessing_time:.4f}s")
        print(f"  Inference Time: {prediction.inference_time:.4f}s")
        print(f"  Predicted Class: {prediction.predicted_classes[0] if prediction.predicted_classes else 'None'}")
        print(f"  Confidence: {prediction.confidence_scores[0] if prediction.confidence_scores is not None else 'None'}")
        
        plugin.cleanup()
        return 0
        
    except Exception as e:
        logger.error(f"Model test failed: {str(e)}")
        return 1

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="SketchRNN Model Benchmarking System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Global arguments
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Run model benchmarking')
    bench_parser.add_argument('--model-ids', nargs='+', help='Model IDs to benchmark')
    bench_parser.add_argument('--discover-models', action='store_true', 
                             help='Auto-discover models in model directory')
    bench_parser.add_argument('--model-dir', default='models', help='Directory to search for models')
    bench_parser.add_argument('--data-path', help='Path to test data file')
    bench_parser.add_argument('--labels-path', help='Path to test labels file (if separate)')
    bench_parser.add_argument('--batch-size', type=int, default=32, help='Batch size for evaluation')
    bench_parser.add_argument('--max-samples', type=int, help='Maximum samples to evaluate')
    bench_parser.add_argument('--warmup-runs', type=int, default=3, help='Number of warmup runs')
    bench_parser.add_argument('--timing-runs', type=int, default=10, help='Number of timing runs')
    bench_parser.add_argument('--parallel', action='store_true', help='Enable parallel model evaluation')
    bench_parser.add_argument('--max-workers', type=int, help='Maximum parallel workers')
    bench_parser.add_argument('--timeout', type=int, default=300, help='Timeout per model (seconds)')
    bench_parser.add_argument('--save-predictions', action='store_true', help='Save model predictions')
    bench_parser.add_argument('--detailed-metrics', action='store_true', help='Calculate detailed metrics')
    bench_parser.add_argument('--generate-reports', action='store_true', help='Generate visualization reports')
    bench_parser.add_argument('--output-dir', default='benchmark_results', help='Output directory')
    
    # Register model command
    reg_parser = subparsers.add_parser('register', help='Register a model')
    reg_parser.add_argument('model_id', help='Unique model identifier')
    reg_parser.add_argument('model_path', help='Path to model file')
    reg_parser.add_argument('--name', help='Human-readable model name')
    reg_parser.add_argument('--version', default='1.0.0', help='Model version')
    reg_parser.add_argument('--description', help='Model description')
    reg_parser.add_argument('--tags', nargs='+', help='Model tags')
    reg_parser.add_argument('--categories-path', default='data/categories.json', 
                           help='Path to categories file')
    
    # List models command
    list_parser = subparsers.add_parser('list', help='List registered models')
    
    # Test model command
    test_parser = subparsers.add_parser('test', help='Test a single model')
    test_parser.add_argument('model_path', help='Path to model file')
    test_parser.add_argument('--data-path', help='Path to test data')
    test_parser.add_argument('--labels-path', help='Path to test labels')
    test_parser.add_argument('--categories-path', default='data/categories.json', 
                            help='Path to categories file')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Execute command
    if args.command == 'benchmark':
        return benchmark_models_cli(args)
    elif args.command == 'register':
        return register_model_cli(args)
    elif args.command == 'list':
        return list_models_cli(args)
    elif args.command == 'test':
        return test_model_cli(args)
    else:
        parser.print_help()
        return 1

if __name__ == '__main__':
    sys.exit(main())