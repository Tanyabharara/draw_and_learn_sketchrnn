"""
Universal Model Factory with Dynamic Loading
Automatically detects and loads any model type from various formats
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Type
import numpy as np

# Import base interfaces
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interfaces import (
    IModelPlugin, BaseModelPlugin, ModelType, InputFormat, OutputFormat,
    ModelInfo, ModelPluginError, ModelLoadError
)

# Configure logging
logger = logging.getLogger(__name__)

class ModelFormatDetector:
    """
    Detects model format and architecture from file analysis
    """
    
    SUPPORTED_EXTENSIONS = {
        '.h5': 'keras_hdf5',
        '.keras': 'keras_native', 
        '.pb': 'tensorflow_savedmodel',
        '.pth': 'pytorch',
        '.pt': 'pytorch',
        '.onnx': 'onnx',
        '.pkl': 'pickle',
        '.joblib': 'joblib',
        '.json': 'tensorflow_js',
        '.tflite': 'tensorflow_lite',
        '.safetensors': 'safetensors',
        '.bin': 'huggingface_bin',
        '.msgpack': 'msgpack',
        '.npz': 'numpy_archive'
    }
    
    @classmethod
    def detect_format(cls, model_path: Union[str, Path]) -> str:
        """
        Detect model format from file extension and content
        
        Args:
            model_path: Path to model file or directory
            
        Returns:
            Detected format string
            
        Raises:
            ValueError: If format cannot be detected
        """
        model_path = Path(model_path)
        
        # Check if it's a directory (SavedModel format)
        if model_path.is_dir():
            if (model_path / 'saved_model.pb').exists():
                return 'tensorflow_savedmodel'
            elif (model_path / 'pytorch_model.bin').exists():
                return 'huggingface_pytorch'
            elif (model_path / 'tf_model.h5').exists():
                return 'huggingface_tensorflow'
            else:
                raise ValueError(f"Unknown directory model format: {model_path}")
        
        # Check file extension
        extension = model_path.suffix.lower()
        if extension in cls.SUPPORTED_EXTENSIONS:
            return cls.SUPPORTED_EXTENSIONS[extension]
        
        # Try to detect from content for extensionless files
        try:
            with open(model_path, 'rb') as f:
                header = f.read(1024)
            
            # Check for various file signatures
            if b'tensorflow' in header.lower():
                return 'tensorflow_savedmodel'
            elif b'pytorch' in header.lower() or b'torch' in header.lower():
                return 'pytorch'
            elif b'onnx' in header.lower():
                return 'onnx'
            elif header.startswith(b'\x80\x03'):  # Python pickle signature
                return 'pickle'
            
        except Exception as e:
            logger.warning(f"Could not analyze file content: {e}")
        
        raise ValueError(f"Unknown model format for file: {model_path}")

class ArchitectureDetector:
    """
    Analyzes loaded models to detect their architecture type
    """
    
    @classmethod
    def detect_architecture(cls, model: Any, format_type: str, config: Dict[str, Any]) -> ModelType:
        """
        Detect model architecture from loaded model object
        
        Args:
            model: Loaded model object
            format_type: Detected format type
            config: Model configuration
            
        Returns:
            Detected ModelType
        """
        # Check config for explicit architecture specification
        if 'architecture_type' in config:
            arch_str = config['architecture_type'].lower()
            try:
                return ModelType(arch_str)
            except ValueError:
                logger.warning(f"Unknown architecture type in config: {arch_str}")
        
        # Format-specific detection
        if format_type in ['keras_hdf5', 'keras_native', 'tensorflow_savedmodel']:
            return cls._detect_tensorflow_keras_architecture(model)
        elif format_type in ['pytorch']:
            return cls._detect_pytorch_architecture(model)
        elif format_type == 'onnx':
            return cls._detect_onnx_architecture(model)
        elif format_type in ['pickle', 'joblib']:
            return cls._detect_sklearn_architecture(model)
        else:
            return ModelType.CUSTOM
    
    @classmethod
    def _detect_tensorflow_keras_architecture(cls, model) -> ModelType:
        """Detect TensorFlow/Keras model architecture"""
        try:
            if hasattr(model, 'layers'):
                layer_types = [type(layer).__name__ for layer in model.layers]
                layer_names = [layer.name for layer in model.layers]
                
                logger.info(f"Analyzing layers: {layer_types[:10]}...")  # Show first 10 layers
                
                # CNN Detection
                conv_layers = [name for name in layer_types if 'Conv' in name]
                dense_layers = [name for name in layer_types if 'Dense' in name]
                rnn_layers = [name for name in layer_types if any(rnn in name for rnn in ['LSTM', 'GRU', 'RNN', 'SimpleRNN'])]
                attention_layers = [name for name in layer_types if 'Attention' in name or 'MultiHead' in name]
                
                # Architecture priority: Transformer > CNN+Attention > RNN > CNN > MLP
                if attention_layers:
                    if conv_layers:
                        return ModelType.CNN  # CNN with attention - classify as CNN for now
                    else:
                        return ModelType.TRANSFORMER
                elif rnn_layers:
                    if 'LSTM' in str(layer_types):
                        return ModelType.LSTM
                    elif 'GRU' in str(layer_types):
                        return ModelType.GRU
                    else:
                        return ModelType.RNN
                elif conv_layers:
                    # Further CNN sub-type detection
                    if any('ResNet' in name or 'Residual' in name for name in layer_names):
                        return ModelType.RESNET
                    elif any('MobileNet' in name for name in layer_names):
                        return ModelType.MOBILENET
                    elif any('EfficientNet' in name for name in layer_names):
                        return ModelType.EFFICIENTNET
                    else:
                        return ModelType.CNN
                elif dense_layers and not conv_layers:
                    return ModelType.MLP
                else:
                    return ModelType.CUSTOM
            
            # Check model name/class for additional hints
            if hasattr(model, '__class__'):
                class_name = model.__class__.__name__.lower()
                if 'resnet' in class_name:
                    return ModelType.RESNET
                elif 'mobilenet' in class_name:
                    return ModelType.MOBILENET
                elif 'efficientnet' in class_name:
                    return ModelType.EFFICIENTNET
                elif 'transformer' in class_name or 'vit' in class_name:
                    return ModelType.TRANSFORMER
        
        except Exception as e:
            logger.warning(f"Failed to analyze TensorFlow/Keras architecture: {e}")
        
        return ModelType.CNN  # Default for TensorFlow models
    
    @classmethod
    def _detect_pytorch_architecture(cls, model) -> ModelType:
        """Detect PyTorch model architecture"""
        try:
            if hasattr(model, 'modules'):
                module_types = [type(module).__name__ for module in model.modules()]
                
                # PyTorch module detection
                if any('Conv' in name for name in module_types):
                    if any('Attention' in name or 'MultiHead' in name for name in module_types):
                        return ModelType.TRANSFORMER
                    elif any('ResNet' in name or 'BasicBlock' in name or 'Bottleneck' in name for name in module_types):
                        return ModelType.RESNET
                    else:
                        return ModelType.CNN
                elif any(rnn in name for rnn in ['LSTM', 'GRU', 'RNN'] for name in module_types):
                    if 'LSTM' in str(module_types):
                        return ModelType.LSTM
                    elif 'GRU' in str(module_types):
                        return ModelType.GRU
                    else:
                        return ModelType.RNN
                elif any('Linear' in name for name in module_types):
                    return ModelType.MLP
            
            # Check class name
            if hasattr(model, '__class__'):
                class_name = model.__class__.__name__.lower()
                if 'resnet' in class_name:
                    return ModelType.RESNET
                elif 'transformer' in class_name or 'vit' in class_name:
                    return ModelType.TRANSFORMER
        
        except Exception as e:
            logger.warning(f"Failed to analyze PyTorch architecture: {e}")
        
        return ModelType.CNN  # Default for PyTorch models
    
    @classmethod
    def _detect_onnx_architecture(cls, model) -> ModelType:
        """Detect ONNX model architecture"""
        try:
            # Basic ONNX analysis would require onnx package
            # For now, return generic CNN
            pass
        except Exception as e:
            logger.warning(f"Failed to analyze ONNX architecture: {e}")
        
        return ModelType.CNN
    
    @classmethod
    def _detect_sklearn_architecture(cls, model) -> ModelType:
        """Detect scikit-learn model architecture"""
        try:
            if hasattr(model, '__class__'):
                class_name = model.__class__.__name__.lower()
                if any(name in class_name for name in ['neural', 'mlp', 'perceptron']):
                    return ModelType.MLP
                else:
                    return ModelType.CUSTOM  # Other sklearn models
        except Exception as e:
            logger.warning(f"Failed to analyze sklearn architecture: {e}")
        
        return ModelType.CUSTOM

class UniversalModelLoader:
    """
    Universal model loader supporting multiple formats and frameworks
    """
    
    def __init__(self):
        self.loaders = {
            'keras_hdf5': self._load_keras_h5,
            'keras_native': self._load_keras_native,
            'tensorflow_savedmodel': self._load_tensorflow_savedmodel,
            'pytorch': self._load_pytorch,
            'onnx': self._load_onnx,
            'pickle': self._load_pickle,
            'joblib': self._load_joblib,
            'tensorflow_js': self._load_tensorflow_js,
            'tensorflow_lite': self._load_tensorflow_lite,
            'safetensors': self._load_safetensors,
            'huggingface_pytorch': self._load_huggingface_pytorch,
            'huggingface_tensorflow': self._load_huggingface_tensorflow,
            'numpy_archive': self._load_numpy_archive
        }
    
    def load_model(self, model_path: Union[str, Path], format_type: str) -> Any:
        """
        Load model using appropriate loader
        
        Args:
            model_path: Path to model file
            format_type: Detected format type
            
        Returns:
            Loaded model object
            
        Raises:
            ModelLoadError: If loading fails
        """
        if format_type not in self.loaders:
            raise ModelLoadError(f"Unsupported model format: {format_type}")
        
        try:
            return self.loaders[format_type](model_path)
        except Exception as e:
            raise ModelLoadError(f"Failed to load {format_type} model from {model_path}: {str(e)}")
    
    def _load_keras_h5(self, model_path: Union[str, Path]) -> Any:
        """Load Keras HDF5 model"""
        try:
            import tensorflow as tf
            return tf.keras.models.load_model(model_path, compile=False)
        except ImportError:
            raise ModelLoadError("TensorFlow not available for Keras model loading")
    
    def _load_keras_native(self, model_path: Union[str, Path]) -> Any:
        """Load Keras native format model"""
        try:
            import tensorflow as tf
            return tf.keras.models.load_model(model_path, compile=False)
        except ImportError:
            raise ModelLoadError("TensorFlow not available for Keras model loading")
    
    def _load_tensorflow_savedmodel(self, model_path: Union[str, Path]) -> Any:
        """Load TensorFlow SavedModel"""
        try:
            import tensorflow as tf
            return tf.saved_model.load(str(model_path))
        except ImportError:
            raise ModelLoadError("TensorFlow not available for SavedModel loading")
    
    def _load_pytorch(self, model_path: Union[str, Path]) -> Any:
        """Load PyTorch model"""
        try:
            import torch
            return torch.load(model_path, map_location='cpu')
        except ImportError:
            raise ModelLoadError("PyTorch not available for model loading")
    
    def _load_onnx(self, model_path: Union[str, Path]) -> Any:
        """Load ONNX model"""
        try:
            import onnxruntime as ort
            return ort.InferenceSession(str(model_path))
        except ImportError:
            raise ModelLoadError("ONNX Runtime not available for model loading")
    
    def _load_pickle(self, model_path: Union[str, Path]) -> Any:
        """Load pickled model"""
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    
    def _load_joblib(self, model_path: Union[str, Path]) -> Any:
        """Load joblib model"""
        try:
            import joblib
            return joblib.load(model_path)
        except ImportError:
            raise ModelLoadError("joblib not available for model loading")
    
    def _load_tensorflow_js(self, model_path: Union[str, Path]) -> Any:
        """Load TensorFlow.js model"""
        try:
            import tensorflow as tf
            # TensorFlow.js models are usually directories with model.json
            if Path(model_path).is_file() and model_path.suffix == '.json':
                model_dir = Path(model_path).parent
                return tf.keras.models.load_model(model_dir)
            else:
                return tf.keras.models.load_model(model_path)
        except ImportError:
            raise ModelLoadError("TensorFlow not available for TensorFlow.js model loading")
    
    def _load_tensorflow_lite(self, model_path: Union[str, Path]) -> Any:
        """Load TensorFlow Lite model"""
        try:
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path=str(model_path))
            interpreter.allocate_tensors()
            return interpreter
        except ImportError:
            raise ModelLoadError("TensorFlow not available for TensorFlow Lite model loading")
    
    def _load_safetensors(self, model_path: Union[str, Path]) -> Any:
        """Load SafeTensors model"""
        try:
            from safetensors import safe_open
            return safe_open(model_path, framework="tf")
        except ImportError:
            raise ModelLoadError("safetensors not available for model loading")
    
    def _load_huggingface_pytorch(self, model_path: Union[str, Path]) -> Any:
        """Load HuggingFace PyTorch model"""
        try:
            from transformers import AutoModel
            return AutoModel.from_pretrained(model_path)
        except ImportError:
            raise ModelLoadError("transformers not available for HuggingFace model loading")
    
    def _load_huggingface_tensorflow(self, model_path: Union[str, Path]) -> Any:
        """Load HuggingFace TensorFlow model"""
        try:
            from transformers import TFAutoModel
            return TFAutoModel.from_pretrained(model_path)
        except ImportError:
            raise ModelLoadError("transformers not available for HuggingFace model loading")
    
    def _load_numpy_archive(self, model_path: Union[str, Path]) -> Any:
        """Load NumPy archive (custom format)"""
        return np.load(model_path, allow_pickle=True)

class UniversalModelFactory:
    """
    Factory class for creating model plugins with automatic detection
    """
    
    def __init__(self):
        self.format_detector = ModelFormatDetector()
        self.architecture_detector = ArchitectureDetector()
        self.model_loader = UniversalModelLoader()
        self.plugin_registry = {}
        
        # Register default plugin mappings
        self._register_default_plugins()
    
    def _register_default_plugins(self):
        """Register default plugin classes for each model type"""
        # We'll import these when the specific plugins are created
        self.plugin_registry = {
            ModelType.CNN: 'CNNModelPlugin',
            ModelType.TRANSFORMER: 'TransformerModelPlugin', 
            ModelType.RNN: 'RNNModelPlugin',
            ModelType.LSTM: 'LSTMModelPlugin',
            ModelType.GRU: 'GRUModelPlugin',
            ModelType.MLP: 'MLPModelPlugin',
            ModelType.RESNET: 'ResNetModelPlugin',
            ModelType.MOBILENET: 'MobileNetModelPlugin',
            ModelType.EFFICIENTNET: 'EfficientNetModelPlugin',
            ModelType.CUSTOM: 'GenericModelPlugin',
            ModelType.ENSEMBLE: 'EnsembleModelPlugin',
            ModelType.API_MODEL: 'APIModelPlugin'
        }
    
    def register_plugin(self, model_type: ModelType, plugin_class: Type[IModelPlugin]):
        """
        Register a custom plugin class for a model type
        
        Args:
            model_type: ModelType enum value
            plugin_class: Plugin class implementing IModelPlugin
        """
        self.plugin_registry[model_type] = plugin_class
        logger.info(f"Registered plugin {plugin_class.__name__} for {model_type.value}")
    
    def create_plugin_from_config(self, config: Dict[str, Any]) -> IModelPlugin:
        """
        Create model plugin from configuration with automatic detection
        
        Args:
            config: Configuration dictionary containing at least 'model_path'
            
        Returns:
            Configured IModelPlugin instance
            
        Raises:
            ModelLoadError: If plugin creation fails
        """
        try:
            # Validate basic configuration
            if 'model_path' not in config:
                raise ValueError("Configuration must contain 'model_path'")
            
            model_path = Path(config['model_path'])
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Detect format
            format_type = self.format_detector.detect_format(model_path)
            logger.info(f"Detected model format: {format_type}")
            
            # Load model
            model = self.model_loader.load_model(model_path, format_type)
            logger.info(f"Successfully loaded model from {model_path}")
            
            # Detect architecture
            architecture_type = self.architecture_detector.detect_architecture(model, format_type, config)
            logger.info(f"Detected architecture: {architecture_type.value}")
            
            # Create appropriate plugin
            plugin = self._create_plugin_instance(architecture_type)
            
            # Pre-configure the plugin
            plugin.model = model
            plugin.config = config
            plugin.model_type = architecture_type
            plugin.is_loaded = True
            
            # Load categories if provided
            if 'categories_path' in config or 'categories' in config:
                if hasattr(plugin, 'load_categories'):
                    categories = config.get('categories_path', config.get('categories', []))
                    plugin.load_categories(categories)
            
            # Setup preprocessing
            if hasattr(plugin, '_setup_preprocessing'):
                plugin._setup_preprocessing(config)
            
            # Generate model info
            plugin.model_info = plugin.get_model_info()
            
            logger.info(f"Successfully created {plugin.__class__.__name__} for {architecture_type.value}")
            return plugin
            
        except Exception as e:
            logger.error(f"Failed to create plugin from config: {str(e)}")
            raise ModelLoadError(f"Plugin creation failed: {str(e)}")
    
    def _create_plugin_instance(self, model_type: ModelType) -> IModelPlugin:
        """
        Create plugin instance for the specified model type
        
        Args:
            model_type: Detected model type
            
        Returns:
            IModelPlugin instance
        """
        if model_type not in self.plugin_registry:
            logger.warning(f"No specific plugin for {model_type.value}, using generic plugin")
            model_type = ModelType.CUSTOM
        
        plugin_class_name = self.plugin_registry[model_type]
        
        # Dynamically import and create plugin
        if isinstance(plugin_class_name, str):
            plugin_class = self._import_plugin_class(plugin_class_name)
        else:
            plugin_class = plugin_class_name
        
        return plugin_class()
    
    def _import_plugin_class(self, class_name: str) -> Type[IModelPlugin]:
        """
        Dynamically import plugin class
        
        Args:
            class_name: Name of plugin class to import
            
        Returns:
            Plugin class
        """
        try:
            # Try to import specific classes from plugins module
            if class_name == 'CNNModelPlugin':
                try:
                    from ..plugins import CNNModelPlugin
                    return CNNModelPlugin
                except ImportError:
                    pass
            elif class_name == 'GenericModelPlugin':
                try:
                    from ..plugins import GenericModelPlugin
                    return GenericModelPlugin
                except ImportError:
                    pass
            elif class_name in ['TransformerModelPlugin', 'RNNModelPlugin', 'LSTMModelPlugin', 
                              'GRUModelPlugin', 'MLPModelPlugin', 'ResNetModelPlugin', 
                              'MobileNetModelPlugin', 'EfficientNetModelPlugin']:
                # These plugins don't exist yet, use GenericModelPlugin
                try:
                    from ..plugins import GenericModelPlugin
                    logger.info(f"Using GenericModelPlugin for {class_name}")
                    return GenericModelPlugin
                except ImportError:
                    pass
            
            # If all imports fail, return fallback plugin
            logger.warning(f"Plugin class {class_name} not found, using fallback plugin")
            return self._create_fallback_plugin()
            
        except Exception as e:
            logger.warning(f"Failed to import plugin {class_name}: {e}")
            return self._create_fallback_plugin()
    
    def _create_fallback_plugin(self) -> Type[IModelPlugin]:
        """
        Create a basic fallback plugin for unsupported model types
        """
        class FallbackModelPlugin(BaseModelPlugin):
            """Fallback plugin for unsupported model types"""
            
            def load_model(self, config: Dict[str, Any]) -> Any:
                # Model should already be loaded by factory
                return self.model
            
            def preprocess_input(self, data: Any):
                from interfaces import ProcessedInput
                import time
                
                start_time = time.time()
                
                # Basic preprocessing
                if isinstance(data, np.ndarray):
                    processed_data = data
                else:
                    processed_data = np.array(data)
                
                # Normalize if needed
                if processed_data.dtype == np.uint8:
                    processed_data = processed_data.astype(np.float32) / 255.0
                
                # Add batch dimension if needed
                if len(processed_data.shape) == 2:  # (H, W)
                    processed_data = processed_data.reshape(1, *processed_data.shape, 1)
                elif len(processed_data.shape) == 3:  # (H, W, C)
                    processed_data = processed_data.reshape(1, *processed_data.shape)
                
                preprocessing_time = time.time() - start_time
                
                return ProcessedInput(
                    data=processed_data,
                    original_shape=data.shape if hasattr(data, 'shape') else (0,),
                    processed_shape=processed_data.shape,
                    batch_size=1,
                    preprocessing_time=preprocessing_time
                )
            
            def predict(self, input_data):
                from interfaces import Prediction
                import time
                
                start_time = time.time()
                
                try:
                    # Try different inference methods
                    if hasattr(self.model, 'predict'):
                        # Keras/TensorFlow
                        output = self.model.predict(input_data.data, verbose=0)
                    elif hasattr(self.model, '__call__'):
                        # PyTorch or callable
                        output = self.model(input_data.data)
                        if hasattr(output, 'detach'):
                            output = output.detach().cpu().numpy()
                    elif hasattr(self.model, 'run'):
                        # ONNX
                        input_name = self.model.get_inputs()[0].name
                        output = self.model.run(None, {input_name: input_data.data})[0]
                    else:
                        raise RuntimeError("Unknown model interface")
                    
                    inference_time = time.time() - start_time
                    
                    # Handle output format
                    if len(output.shape) > 1 and output.shape[0] == 1:
                        output = output[0]
                    
                    predicted_class = int(np.argmax(output)) if len(output.shape) >= 1 else 0
                    confidence = float(np.max(output)) if len(output.shape) >= 1 else 0.0
                    
                    return Prediction(
                        raw_output=output,
                        probabilities=output,
                        predicted_classes=[predicted_class],
                        confidence_scores=np.array([confidence]),
                        inference_time=inference_time,
                        batch_size=input_data.batch_size
                    )
                    
                except Exception as e:
                    raise RuntimeError(f"Prediction failed: {str(e)}")
            
            def get_model_info(self):
                from interfaces import ModelInfo, InputFormat, OutputFormat
                
                # Try to extract info from model
                input_shape = (28, 28, 1)  # Default
                output_shape = (28,)  # Default for 28 classes
                
                try:
                    if hasattr(self.model, 'input_shape'):
                        input_shape = self.model.input_shape[1:]  # Remove batch dim
                    elif hasattr(self.model, 'inputs'):
                        input_shape = self.model.inputs[0].shape[1:]
                    
                    if hasattr(self.model, 'output_shape'):
                        output_shape = self.model.output_shape[1:]  # Remove batch dim
                    elif hasattr(self.model, 'outputs'):
                        output_shape = self.model.outputs[0].shape[1:]
                except:
                    pass
                
                return ModelInfo(
                    name=f"Fallback_{self.__class__.__name__}",
                    version="1.0.0",
                    model_type=self.model_type,
                    input_format=InputFormat.IMAGE_3D,
                    output_format=OutputFormat.CLASSIFICATION,
                    input_shape=input_shape,
                    output_shape=output_shape,
                    preprocessing_steps=['normalize', 'reshape'],
                    metadata={'plugin_type': 'fallback'}
                )
        
        return FallbackModelPlugin

# Convenience function for quick plugin creation
def create_model_plugin(model_path: Union[str, Path], **config_kwargs) -> IModelPlugin:
    """
    Convenience function to create a model plugin
    
    Args:
        model_path: Path to model file
        **config_kwargs: Additional configuration parameters
        
    Returns:
        Configured IModelPlugin instance
    """
    config = {'model_path': str(model_path)}
    config.update(config_kwargs)
    
    factory = UniversalModelFactory()
    return factory.create_plugin_from_config(config)

# Create factory instance for module-level use
_factory_instance = UniversalModelFactory()

def get_factory() -> UniversalModelFactory:
    """Get the global factory instance"""
    return _factory_instance