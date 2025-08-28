#!/usr/bin/env python3
"""
Validation Script for SketchRNN Benchmarking System
Tests all components to ensure they work correctly
"""

import os
import sys
import json
import pickle
import numpy as np
import logging
from pathlib import Path
import traceback

# Add benchmark system to path
sys.path.append('./benchmark_system')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        # Core components
        from benchmark_system.utils.hardware_detector import create_hardware_detector
        from benchmark_system.interfaces import IModelPlugin, ModelType, ModelInfo
        from benchmark_system.core.model_factory import create_model_plugin
        from benchmark_system.core.model_registry import get_global_registry, register_model
        from benchmark_system.plugins import CNNModelPlugin, GenericModelPlugin
        from benchmark_system.core.benchmark_engine import BenchmarkEngine, BenchmarkConfig
        from benchmark_system.core.metrics import calculate_metrics
        from benchmark_system.utils.visualization import create_visualizer
        
        print("✅ All imports successful")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {str(e)}")
        traceback.print_exc()
        return False

def test_hardware_detection():
    """Test hardware detection functionality"""
    print("\n🖥️ Testing hardware detection...")
    
    try:
        from benchmark_system.utils.hardware_detector import create_hardware_detector
        
        detector = create_hardware_detector()
        hardware_summary = detector.get_device_summary()
        opt_config = detector.get_optimization_config()
        
        print(f"✅ Hardware detected: {hardware_summary['primary_device']['name']}")
        print(f"✅ Optimization config: {opt_config.device_placement}")
        
        return True
        
    except Exception as e:
        print(f"❌ Hardware detection failed: {str(e)}")
        return False

def test_model_registration():
    """Test model registry functionality"""
    print("\n📋 Testing model registry...")
    
    try:
        from benchmark_system.core.model_registry import get_global_registry, register_model
        
        registry = get_global_registry()
        
        # Create test data
        test_categories = ["cat", "dog", "house"]
        
        # Test registration (with dummy model path)
        test_model_path = "test_model.h5"
        Path(test_model_path).touch()  # Create dummy file
        
        success = register_model(
            model_id="test_model",
            name="Test Model",
            model_path=test_model_path,
            categories=test_categories
        )
        
        if success:
            models = registry.list_models()
            print(f"✅ Model registered successfully. Total models: {len(models)}")
        else:
            print("❌ Model registration failed")
            return False
        
        # Cleanup
        Path(test_model_path).unlink(missing_ok=True)
        registry.remove_model("test_model")
        
        return True
        
    except Exception as e:
        print(f"❌ Model registry test failed: {str(e)}")
        return False

def create_test_data():
    """Create test data for validation"""
    print("\n📊 Creating test data...")
    
    try:
        # Create sample categories
        categories = ["cat", "dog", "house", "tree", "car"]
        
        # Create sample test data
        test_data = np.random.randint(0, 255, (50, 28, 28), dtype=np.uint8)
        test_labels = np.random.randint(0, len(categories), 50)
        
        # Save categories
        data_dir = Path('data')
        data_dir.mkdir(exist_ok=True)
        
        with open(data_dir / 'categories.json', 'w') as f:
            json.dump(categories, f)
        
        print(f"✅ Test data created: {test_data.shape}, labels: {test_labels.shape}")
        return test_data, test_labels, categories
        
    except Exception as e:
        print(f"❌ Test data creation failed: {str(e)}")
        return None, None, None

def create_test_model():
    """Create a test model for validation"""
    print("\n🤖 Creating test model...")
    
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
        
        model = models.Sequential([
            layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(32, activation='relu'),
            layers.Dense(5, activation='softmax')  # 5 classes
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Save model
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / 'test_cnn_model.h5'
        model.save(model_path)
        
        print(f"✅ Test model created and saved: {model_path}")
        return str(model_path)
        
    except ImportError:
        print("⚠️ TensorFlow not available, skipping model creation")
        return None
    except Exception as e:
        print(f"❌ Test model creation failed: {str(e)}")
        return None

def test_model_plugin(model_path, test_data, categories):
    """Test model plugin functionality"""
    print("\n🔌 Testing model plugin...")
    
    try:
        from benchmark_system.core.model_factory import create_model_plugin
        
        # Create plugin
        plugin = create_model_plugin(
            model_path=model_path,
            categories=categories
        )
        
        # Test model info
        model_info = plugin.get_model_info()
        print(f"✅ Model info: {model_info.name} ({model_info.model_type.value})")
        
        # Test preprocessing
        sample_data = test_data[0]
        processed = plugin.preprocess_input(sample_data)
        print(f"✅ Preprocessing: {processed.original_shape} -> {processed.processed_shape}")
        
        # Test prediction
        prediction = plugin.predict(processed)
        print(f"✅ Prediction: class={prediction.predicted_classes[0]}, confidence={prediction.confidence_scores[0]:.4f}")
        
        # Cleanup
        plugin.cleanup()
        
        return True
        
    except Exception as e:
        print(f"❌ Model plugin test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_benchmark_engine(model_path, test_data, test_labels, categories):
    """Test the complete benchmark engine"""
    print("\n🚀 Testing benchmark engine...")
    
    try:
        from benchmark_system.core.benchmark_engine import BenchmarkEngine, BenchmarkConfig
        from benchmark_system.core.model_registry import register_model
        from benchmark_system.utils.hardware_detector import create_hardware_detector
        
        # Register test model
        model_id = "validation_test_model"
        success = register_model(
            model_id=model_id,
            name="Validation Test Model",
            model_path=model_path,
            categories=categories
        )
        
        if not success:
            print("❌ Failed to register model for benchmark test")
            return False
        
        # Create benchmark config
        config = BenchmarkConfig(
            batch_size=16,
            max_samples=20,  # Small for quick test
            warmup_runs=1,
            timing_runs=2,
            parallel_models=False,  # Disable for simple test
            output_dir="validation_results"
        )
        
        # Create benchmark engine
        hardware_detector = create_hardware_detector()
        engine = BenchmarkEngine(hardware_detector=hardware_detector)
        
        # Run benchmark
        summary = engine.benchmark_models(
            model_ids=[model_id],
            test_data=test_data,
            test_labels=test_labels,
            config=config
        )
        
        print(f"✅ Benchmark completed:")
        print(f"   Total models: {summary.total_models}")
        print(f"   Successful: {summary.successful_models}")
        print(f"   Failed: {summary.failed_models}")
        print(f"   Execution time: {summary.total_execution_time:.2f}s")
        
        if summary.successful_models > 0:
            result = summary.results[0]
            if result.metrics:
                print(f"   Accuracy: {result.metrics.accuracy:.4f}")
                print(f"   Inference time: {result.metrics.inference_time_mean:.4f}s")
        
        return summary.successful_models > 0
        
    except Exception as e:
        print(f"❌ Benchmark engine test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_visualization(summary):
    """Test visualization functionality"""
    print("\n📊 Testing visualization...")
    
    try:
        from benchmark_system.utils.visualization import create_visualizer
        
        visualizer = create_visualizer("validation_reports")
        report_dir = visualizer.generate_comprehensive_report(summary)
        
        print(f"✅ Visualization reports generated: {report_dir}")
        
        # Check if files were created
        report_files = list(report_dir.glob("*"))
        print(f"✅ Generated {len(report_files)} report files")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization test failed: {str(e)}")
        return False

def test_cli_interface():
    """Test CLI interface"""
    print("\n💻 Testing CLI interface...")
    
    try:
        # Test help command
        import subprocess
        result = subprocess.run([sys.executable, 'benchmark_models.py', '--help'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and 'benchmark' in result.stdout:
            print("✅ CLI interface accessible")
            return True
        else:
            print("❌ CLI interface test failed")
            return False
            
    except Exception as e:
        print(f"❌ CLI test failed: {str(e)}")
        return False

def cleanup_test_files():
    """Clean up test files"""
    print("\n🧹 Cleaning up test files...")
    
    try:
        # Remove test model
        test_model_path = Path('models/test_cnn_model.h5')
        test_model_path.unlink(missing_ok=True)
        
        # Remove validation results
        import shutil
        for dir_name in ['validation_results', 'validation_reports']:
            if Path(dir_name).exists():
                shutil.rmtree(dir_name)
        
        print("✅ Cleanup completed")
        
    except Exception as e:
        print(f"⚠️ Cleanup warning: {str(e)}")

def main():
    """Main validation function"""
    print("🔍 VALIDATING SKETCHRNN BENCHMARKING SYSTEM")
    print("=" * 60)
    
    test_results = []
    
    # Test 1: Imports
    test_results.append(("Imports", test_imports()))
    
    # Test 2: Hardware Detection
    test_results.append(("Hardware Detection", test_hardware_detection()))
    
    # Test 3: Model Registry
    test_results.append(("Model Registry", test_model_registration()))
    
    # Test 4: Create test data
    test_data, test_labels, categories = create_test_data()
    if test_data is not None:
        test_results.append(("Test Data Creation", True))
        
        # Test 5: Create test model
        model_path = create_test_model()
        if model_path:
            test_results.append(("Test Model Creation", True))
            
            # Test 6: Model Plugin
            test_results.append(("Model Plugin", test_model_plugin(model_path, test_data, categories)))
            
            # Test 7: Benchmark Engine
            summary = None
            if test_benchmark_engine(model_path, test_data, test_labels, categories):
                test_results.append(("Benchmark Engine", True))
                
                # Get the summary for visualization test
                try:
                    from benchmark_system.core.benchmark_engine import BenchmarkEngine, BenchmarkConfig
                    from benchmark_system.utils.hardware_detector import create_hardware_detector
                    
                    config = BenchmarkConfig(max_samples=10, output_dir="validation_results")
                    engine = BenchmarkEngine(hardware_detector=create_hardware_detector())
                    summary = engine.benchmark_models(["validation_test_model"], test_data[:10], test_labels[:10], config)
                except:
                    pass
            else:
                test_results.append(("Benchmark Engine", False))
            
            # Test 8: Visualization
            if summary:
                test_results.append(("Visualization", test_visualization(summary)))
            else:
                test_results.append(("Visualization", False))
            
        else:
            test_results.append(("Test Model Creation", False))
            test_results.append(("Model Plugin", False))
            test_results.append(("Benchmark Engine", False))
            test_results.append(("Visualization", False))
    else:
        test_results.append(("Test Data Creation", False))
    
    # Test 9: CLI Interface
    test_results.append(("CLI Interface", test_cli_interface()))
    
    # Print results summary
    print("\n" + "=" * 60)
    print("🎯 VALIDATION RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print("-" * 60)
    print(f"TOTAL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! System is ready for use.")
    elif passed >= total * 0.8:
        print("⚠️ Most tests passed. System is mostly functional.")
    else:
        print("❌ Multiple test failures. System needs attention.")
    
    # Cleanup
    cleanup_test_files()
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)