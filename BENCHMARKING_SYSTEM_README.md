# SketchRNN Model Benchmarking System

A comprehensive, modular, and scalable benchmarking system for evaluating machine learning models with hardware-aware optimization and advanced metrics analysis.

## 🚀 Features

### Core Capabilities
- **Modular Plugin Architecture**: Easy to add support for any model type (CNN, Transformer, RNN, etc.)
- **Hardware-Aware Optimization**: Automatically detects and optimizes for GPU/CPU configurations
- **Parallel Processing**: Efficient evaluation of multiple models with intelligent resource management
- **Comprehensive Metrics**: Beyond accuracy - includes confidence calibration, timing, memory usage
- **Rich Visualizations**: Performance charts, accuracy analysis, timing breakdowns
- **Easy Integration**: Both CLI and Python API interfaces

### Hardware Support
- **GPU Optimization**: RTX 3050, RTX 4060, Tesla T4, and other NVIDIA GPUs
- **Environment Detection**: Local laptop, Google Colab, AWS SageMaker
- **Memory Management**: Intelligent memory allocation and cleanup
- **Mixed Precision**: Automatic FP16 support where available

## 📁 Project Structure

```
benchmark_system/
├── core/                           # Core benchmarking components
│   ├── benchmark_engine.py         # Main benchmarking orchestrator
│   ├── model_factory.py           # Universal model loading
│   ├── model_registry.py          # Model management system
│   └── metrics.py                 # Comprehensive metrics calculation
├── interfaces/                     # Abstract interfaces
│   └── __init__.py                # Base plugin interfaces
├── plugins/                        # Model-specific implementations
│   └── __init__.py                # CNN and generic plugins
├── utils/                         # Utility modules
│   ├── hardware_detector.py       # Hardware detection and optimization
│   └── visualization.py           # Reporting and visualization
├── benchmark_models.py            # CLI interface
├── validate_benchmark_system.py   # System validation
└── SketchRNN_Benchmarking_Demo.ipynb  # Interactive notebook
```

## 🛠️ Installation & Setup

### Prerequisites
```bash
# Core dependencies
pip install numpy scikit-learn matplotlib seaborn

# For TensorFlow models (recommended)
pip install tensorflow

# For PyTorch models (optional)
pip install torch

# For ONNX models (optional)
pip install onnxruntime

# For enhanced features (optional)
pip install psutil pillow
```

### Quick Setup
```bash
# Clone/navigate to the project directory
cd draw_and_learn_sketchrnn

# Validate the system
python validate_benchmark_system.py

# Run a quick test
python benchmark_models.py list
```

## 🎯 Quick Start

### 1. Using the CLI Interface

#### Discover and Benchmark Models
```bash
# Auto-discover models in the models/ directory
python benchmark_models.py benchmark --discover-models --generate-reports

# Benchmark specific models with parallel processing
python benchmark_models.py benchmark --model-ids model1 model2 --parallel --detailed-metrics

# Benchmark with custom parameters
python benchmark_models.py benchmark --discover-models \
    --batch-size 64 \
    --max-samples 1000 \
    --parallel \
    --generate-reports \
    --output-dir my_benchmark_results
```

#### Model Management
```bash
# Register a new model
python benchmark_models.py register my_cnn_model /path/to/model.h5 \
    --name "My CNN Model" \
    --tags cnn sketchrnn \
    --description "Custom CNN for sketch recognition"

# List all registered models
python benchmark_models.py list

# Test a single model
python benchmark_models.py test /path/to/model.h5 --data-path data/test_data.pkl
```

### 2. Using the Python API

```python
from benchmark_system.core.benchmark_engine import BenchmarkEngine, BenchmarkConfig
from benchmark_system.core.model_registry import register_model
from benchmark_system.utils.hardware_detector import create_hardware_detector

# Register models
register_model(
    model_id="my_model",
    name="My Model",
    model_path="models/my_model.h5",
    categories=["cat", "dog", "house"]  # Your categories
)

# Configure benchmarking
config = BenchmarkConfig(
    batch_size=64,
    max_samples=500,
    parallel_models=True,
    generate_reports=True
)

# Run benchmark
hardware_detector = create_hardware_detector()
engine = BenchmarkEngine(hardware_detector=hardware_detector)

summary = engine.benchmark_models(
    model_ids=["my_model"],
    test_data=test_data,
    test_labels=test_labels,
    config=config
)

print(f"Accuracy: {summary.results[0].metrics.accuracy:.4f}")
```

### 3. Using the Jupyter Notebook

Open `SketchRNN_Benchmarking_Demo.ipynb` for an interactive demonstration with:
- Hardware detection
- Model registration
- Benchmark execution  
- Results visualization
- Advanced analysis

## 🔧 Advanced Configuration

### Hardware Optimization

The system automatically detects and optimizes for your hardware:

```python
from benchmark_system.utils.hardware_detector import create_hardware_detector

detector = create_hardware_detector()
config = detector.get_optimization_config()

print(f"Optimized for: {config.device_placement}")
print(f"Max batch size: {config.max_batch_size}")
print(f"Parallel workers: {config.max_workers}")
```

### Custom Model Plugins

Add support for new model types by implementing the plugin interface:

```python
from benchmark_system.interfaces import BaseModelPlugin, ModelType

class MyCustomModelPlugin(BaseModelPlugin):
    def __init__(self):
        super().__init__(ModelType.CUSTOM)
    
    def load_model(self, config):
        # Load your model
        pass
    
    def preprocess_input(self, data):
        # Preprocess input data
        pass
    
    def predict(self, input_data):
        # Run inference
        pass
    
    def get_model_info(self):
        # Return model information
        pass
```

### Benchmark Configuration

Customize benchmarking behavior:

```python
config = BenchmarkConfig(
    batch_size=32,              # Batch size for inference
    max_samples=1000,           # Limit test samples
    warmup_runs=5,              # Warmup iterations
    timing_runs=10,             # Timing measurement runs
    parallel_models=True,       # Enable parallel execution
    max_workers=4,              # Override auto-detected workers
    timeout_seconds=300,        # Per-model timeout
    save_predictions=False,     # Save individual predictions
    save_detailed_metrics=True, # Calculate advanced metrics
    output_dir="results"        # Output directory
)
```

## 📊 Metrics and Analysis

### Basic Metrics
- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Per-class and weighted averages
- **Inference Time**: Mean and standard deviation
- **Throughput**: Samples per second

### Advanced Metrics
- **Confidence Calibration**: Expected Calibration Error (ECE)
- **Reliability Score**: High-confidence prediction accuracy
- **Statistical Analysis**: Confidence intervals, entropy
- **Memory Efficiency**: Accuracy per MB of memory
- **Per-Category Analysis**: Detailed breakdown by class

### Visualization
- Performance comparison charts
- Accuracy vs speed trade-offs
- Timing analysis and latency percentiles
- Confidence calibration plots
- Per-category performance heatmaps

## 🎮 Example Use Cases

### 1. Compare Multiple SketchRNN Models
```bash
python benchmark_models.py benchmark \
    --model-ids sketchrnn_v1 sketchrnn_v2 sketchrnn_v3 \
    --data-path data/quickdraw_test.pkl \
    --parallel \
    --generate-reports
```

### 2. Evaluate Model Performance on Different Hardware
```python
# The system automatically optimizes for your hardware
# RTX 4060 laptop: batch_size=128, workers=3, mixed_precision=True
# Tesla T4 cloud: batch_size=256, workers=4, parallel_models=2
# CPU only: batch_size=32, workers=2, parallel_models=1
```

### 3. A/B Test Model Variants
```python
models = ["baseline_model", "improved_model", "optimized_model"]
summary = quick_benchmark(models, test_data, test_labels, detailed_metrics=True)

for result in summary.results:
    print(f"{result.model_name}: {result.metrics.accuracy:.4f} accuracy")
```

## 🔍 System Validation

Run the validation script to ensure everything works:

```bash
python validate_benchmark_system.py
```

This tests:
- ✅ Module imports
- ✅ Hardware detection
- ✅ Model registry
- ✅ Plugin functionality
- ✅ Benchmark engine
- ✅ Visualization system
- ✅ CLI interface

## 🐛 Troubleshooting

### Common Issues

1. **"Model not found" error**
   ```bash
   # Register the model first
   python benchmark_models.py register model_id /path/to/model.h5
   ```

2. **GPU memory issues**
   ```python
   # Reduce batch size or enable memory management
   config = BenchmarkConfig(batch_size=16, memory_limit_mb=2048)
   ```

3. **Import errors**
   ```bash
   # Install missing dependencies
   pip install tensorflow scikit-learn matplotlib
   ```

4. **Slow performance**
   ```bash
   # Enable parallel processing and hardware optimization
   python benchmark_models.py benchmark --parallel --discover-models
   ```

### Debug Mode
```bash
# Enable verbose logging
python benchmark_models.py benchmark --log-level DEBUG --model-ids model1
```

## 🤝 Contributing

### Adding New Model Types

1. Create a plugin class inheriting from `BaseModelPlugin`
2. Implement required methods: `load_model`, `preprocess_input`, `predict`, `get_model_info`
3. Register the plugin in the model factory
4. Add tests for the new plugin

### Adding New Metrics

1. Extend the `MetricsCalculator` class
2. Add new metric calculations
3. Update visualization components
4. Add documentation and examples

## 📈 Performance Benchmarks

### Hardware Performance (typical results)
- **RTX 4060 Laptop**: 150-300 samples/sec, 8GB memory efficient
- **Tesla T4 (Colab)**: 200-400 samples/sec, optimal for cloud training
- **CPU Only**: 20-50 samples/sec, good for development/testing

### Model Comparison (QuickDraw 28 classes)
- **Accuracy Range**: 85-98% depending on architecture
- **Inference Time**: 0.001-0.01s per sample
- **Memory Usage**: 50-500MB depending on model size

## 📄 License

This project is part of the SketchRNN learning system. See the main project README for license information.

## 🙋‍♂️ Support

For issues, questions, or contributions:
1. Check the troubleshooting guide above
2. Run the validation script
3. Enable debug logging for detailed error information
4. Review the example notebooks for usage patterns

---

**Happy Benchmarking! 🚀**