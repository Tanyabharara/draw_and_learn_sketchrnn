"""
Abstract Model Plugin Interfaces and Base Classes
Defines the standard interface that all model plugins must implement
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import numpy as np
import logging
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Enumeration for supported model types"""
    CNN = "cnn"
    TRANSFORMER = "transformer"
    RNN = "rnn"
    LSTM = "lstm"
    GRU = "gru"
    MLP = "mlp"
    RESNET = "resnet"
    VIT = "vit"  # Vision Transformer
    MOBILENET = "mobilenet"
    EFFICIENTNET = "efficientnet"
    CUSTOM = "custom"
    ENSEMBLE = "ensemble"
    API_MODEL = "api_model"  # For cloud APIs like Gemini, GPT-4V, etc.

class InputFormat(Enum):
    """Enumeration for input data formats"""
    IMAGE_2D = "image_2d"          # (height, width) or (height, width, channels)
    IMAGE_3D = "image_3d"          # (height, width, channels)
    IMAGE_BATCH = "image_batch"    # (batch, height, width, channels)
    SEQUENCE = "sequence"          # (sequence_length, features)
    TABULAR = "tabular"           # (features,)
    TEXT = "text"                 # String or tokenized text
    AUDIO = "audio"               # Audio waveform or spectrogram
    VIDEO = "video"               # (frames, height, width, channels)
    MULTIMODAL = "multimodal"     # Multiple input types
    RAW = "raw"                   # Raw bytes or custom format

class OutputFormat(Enum):
    """Enumeration for output formats"""
    CLASSIFICATION = "classification"      # Class probabilities
    REGRESSION = "regression"             # Continuous values
    DETECTION = "detection"               # Bounding boxes + classes
    SEGMENTATION = "segmentation"         # Pixel-wise labels
    EMBEDDING = "embedding"               # Feature vectors
    SEQUENCE = "sequence"                 # Sequential output
    MULTICLASS = "multiclass"             # Multiple class predictions
    MULTILABEL = "multilabel"             # Multiple binary predictions
    CUSTOM = "custom"                     # Custom output format

@dataclass
class ModelInfo:
    """Comprehensive model information"""
    name: str
    version: str
    model_type: ModelType
    input_format: InputFormat
    output_format: OutputFormat
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    num_parameters: Optional[int] = None
    model_size_mb: Optional[float] = None
    supported_batch_sizes: List[int] = field(default_factory=lambda: [1, 8, 16, 32, 64, 128])
    preprocessing_steps: List[str] = field(default_factory=list)
    postprocessing_steps: List[str] = field(default_factory=list)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'name': self.name,
            'version': self.version,
            'model_type': self.model_type.value,
            'input_format': self.input_format.value,
            'output_format': self.output_format.value,
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
            'num_parameters': self.num_parameters,
            'model_size_mb': self.model_size_mb,
            'supported_batch_sizes': self.supported_batch_sizes,
            'preprocessing_steps': self.preprocessing_steps,
            'postprocessing_steps': self.postprocessing_steps,
            'resource_requirements': self.resource_requirements,
            'metadata': self.metadata
        }

@dataclass
class ProcessedInput:
    """Container for processed input data"""
    data: Union[np.ndarray, Any]
    original_shape: Tuple[int, ...]
    processed_shape: Tuple[int, ...]
    batch_size: int
    preprocessing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding large data arrays)"""
        return {
            'original_shape': self.original_shape,
            'processed_shape': self.processed_shape,
            'batch_size': self.batch_size,
            'preprocessing_time': self.preprocessing_time,
            'metadata': self.metadata,
            'data_type': str(type(self.data))
        }

@dataclass
class Prediction:
    """Container for model predictions"""
    raw_output: Union[np.ndarray, Any]
    probabilities: Optional[np.ndarray] = None
    predicted_classes: Optional[Union[List[str], List[int], np.ndarray]] = None
    confidence_scores: Optional[np.ndarray] = None
    inference_time: float = 0.0
    batch_size: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_top_prediction(self, k: int = 1) -> Dict[str, Any]:
        """Get top k predictions"""
        if self.probabilities is not None and self.predicted_classes is not None:
            if len(self.probabilities.shape) == 1:  # Single sample
                top_indices = np.argsort(self.probabilities)[-k:][::-1]
                return {
                    'classes': [self.predicted_classes[i] for i in top_indices],
                    'probabilities': self.probabilities[top_indices].tolist(),
                    'confidence': float(np.max(self.probabilities))
                }
            else:  # Batch
                # Return top predictions for first sample in batch
                top_indices = np.argsort(self.probabilities[0])[-k:][::-1]
                return {
                    'classes': [self.predicted_classes[i] for i in top_indices],
                    'probabilities': self.probabilities[0][top_indices].tolist(),
                    'confidence': float(np.max(self.probabilities[0]))
                }
        return {'classes': [], 'probabilities': [], 'confidence': 0.0}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = {
            'inference_time': self.inference_time,
            'batch_size': self.batch_size,
            'metadata': self.metadata
        }
        
        # Add arrays as lists for JSON serialization
        if self.probabilities is not None:
            result['probabilities'] = self.probabilities.tolist()
        if self.predicted_classes is not None:
            if isinstance(self.predicted_classes, np.ndarray):
                result['predicted_classes'] = self.predicted_classes.tolist()
            else:
                result['predicted_classes'] = self.predicted_classes
        if self.confidence_scores is not None:
            result['confidence_scores'] = self.confidence_scores.tolist()
        
        return result

@dataclass
class BenchmarkMetrics:
    """Container for benchmark results"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    inference_time_mean: float
    inference_time_std: float
    preprocessing_time_mean: float
    preprocessing_time_std: float
    throughput: float  # samples per second
    memory_usage_mb: Optional[float] = None
    per_class_metrics: Optional[Dict[str, Dict[str, float]]] = None
    confusion_matrix: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = {
            'model_name': self.model_name,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'inference_time_mean': self.inference_time_mean,
            'inference_time_std': self.inference_time_std,
            'preprocessing_time_mean': self.preprocessing_time_mean,
            'preprocessing_time_std': self.preprocessing_time_std,
            'throughput': self.throughput,
            'memory_usage_mb': self.memory_usage_mb,
            'metadata': self.metadata
        }
        
        if self.per_class_metrics:
            result['per_class_metrics'] = self.per_class_metrics
        if self.confusion_matrix is not None:
            result['confusion_matrix'] = self.confusion_matrix.tolist()
        
        return result

class IModelPlugin(ABC):
    """
    Abstract base class for all model plugins
    All model implementations must inherit from this class and implement its methods
    """
    
    def __init__(self):
        self.model = None
        self.config = {}
        self.is_loaded = False
        self.model_info = None
    
    @abstractmethod
    def load_model(self, config: Dict[str, Any]) -> Any:
        """
        Load the model from the specified configuration
        
        Args:
            config: Dictionary containing model loading configuration
                   Must include 'model_path' at minimum
        
        Returns:
            The loaded model object
        
        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If configuration is invalid
            RuntimeError: If model loading fails
        """
        pass
    
    @abstractmethod
    def preprocess_input(self, data: Any) -> ProcessedInput:
        """
        Preprocess input data for the model
        
        Args:
            data: Raw input data (image, text, etc.)
        
        Returns:
            ProcessedInput object containing preprocessed data and metadata
        
        Raises:
            ValueError: If input data is invalid
            RuntimeError: If preprocessing fails
        """
        pass
    
    @abstractmethod
    def predict(self, input_data: ProcessedInput) -> Prediction:
        """
        Run inference on preprocessed input data
        
        Args:
            input_data: ProcessedInput object from preprocess_input
        
        Returns:
            Prediction object containing model output and timing information
        
        Raises:
            RuntimeError: If inference fails
            ValueError: If input format is incorrect
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """
        Get comprehensive information about the model
        
        Returns:
            ModelInfo object with model specifications
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """
        Clean up model resources (memory, GPU memory, etc.)
        Should be called when model is no longer needed
        """
        pass
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate model configuration
        
        Args:
            config: Configuration dictionary to validate
        
        Returns:
            True if configuration is valid, False otherwise
        """
        required_fields = ['model_path']
        return all(field in config for field in required_fields)
    
    def is_model_loaded(self) -> bool:
        """Check if model is successfully loaded"""
        return self.is_loaded and self.model is not None
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """
        Get supported input/output formats for this plugin
        
        Returns:
            Dictionary with 'input' and 'output' keys containing format lists
        """
        return {
            'input': [InputFormat.IMAGE_2D.value, InputFormat.IMAGE_3D.value],
            'output': [OutputFormat.CLASSIFICATION.value]
        }
    
    def batch_predict(self, input_data_list: List[ProcessedInput]) -> List[Prediction]:
        """
        Run batch inference on multiple inputs
        Default implementation calls predict() for each input
        Subclasses can override for more efficient batch processing
        
        Args:
            input_data_list: List of ProcessedInput objects
        
        Returns:
            List of Prediction objects
        """
        predictions = []
        for input_data in input_data_list:
            prediction = self.predict(input_data)
            predictions.append(prediction)
        return predictions
    
    def estimate_memory_usage(self, batch_size: int = 1) -> float:
        """
        Estimate memory usage in MB for given batch size
        
        Args:
            batch_size: Batch size for estimation
        
        Returns:
            Estimated memory usage in MB
        """
        if self.model_info and self.model_info.model_size_mb:
            # Simple estimation: model size + input/output tensors
            input_size_mb = np.prod(self.model_info.input_shape) * batch_size * 4 / (1024 * 1024)  # float32
            output_size_mb = np.prod(self.model_info.output_shape) * batch_size * 4 / (1024 * 1024)
            return self.model_info.model_size_mb + input_size_mb + output_size_mb
        return 100.0  # Default estimation

class BaseModelPlugin(IModelPlugin):
    """
    Base implementation of IModelPlugin with common functionality
    Provides default implementations for common operations
    """
    
    def __init__(self, model_type: ModelType = ModelType.CUSTOM):
        super().__init__()
        self.model_type = model_type
        self.categories = []
        self.preprocessing_pipeline = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def load_categories(self, categories_path: Union[str, Path, List[str]]) -> None:
        """
        Load category labels for classification models
        
        Args:
            categories_path: Path to categories file, or list of category names
        """
        if isinstance(categories_path, (str, Path)):
            import json
            with open(categories_path, 'r') as f:
                self.categories = json.load(f)
        elif isinstance(categories_path, list):
            self.categories = categories_path
        else:
            raise ValueError("categories_path must be string path or list of categories")
        
        self.logger.info(f"Loaded {len(self.categories)} categories")
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Enhanced configuration validation"""
        required_fields = ['model_path']
        optional_fields = ['categories_path', 'input_shape', 'preprocessing', 'batch_size']
        
        # Check required fields
        if not all(field in config for field in required_fields):
            missing = [f for f in required_fields if f not in config]
            self.logger.error(f"Missing required config fields: {missing}")
            return False
        
        # Validate model path
        model_path = Path(config['model_path'])
        if not model_path.exists():
            self.logger.error(f"Model file not found: {model_path}")
            return False
        
        # Validate categories if provided
        if 'categories_path' in config:
            categories_path = Path(config['categories_path'])
            if not categories_path.exists():
                self.logger.error(f"Categories file not found: {categories_path}")
                return False
        
        return True
    
    def _setup_preprocessing(self, config: Dict[str, Any]) -> None:
        """
        Setup preprocessing pipeline based on configuration
        Can be overridden by subclasses for custom preprocessing
        """
        preprocessing_config = config.get('preprocessing', {})
        
        # Default preprocessing steps
        self.preprocessing_steps = [
            'normalize',  # Normalize pixel values to [0, 1]
            'resize',     # Resize to model input size
            'reshape'     # Add batch dimension if needed
        ]
        
        # Update from config
        if 'steps' in preprocessing_config:
            self.preprocessing_steps = preprocessing_config['steps']
    
    def _apply_preprocessing_step(self, data: np.ndarray, step: str, **kwargs) -> np.ndarray:
        """
        Apply a single preprocessing step
        
        Args:
            data: Input data array
            step: Preprocessing step name
            **kwargs: Additional parameters for the step
        
        Returns:
            Processed data array
        """
        if step == 'normalize':
            # Normalize to [0, 1] range
            if data.dtype == np.uint8:
                return data.astype(np.float32) / 255.0
            elif data.max() > 1.0:
                return data.astype(np.float32) / data.max()
            return data.astype(np.float32)
        
        elif step == 'resize':
            # Resize image (requires PIL or cv2)
            target_size = kwargs.get('size', (28, 28))
            if len(data.shape) == 2:  # Grayscale
                try:
                    from PIL import Image
                    img = Image.fromarray(data)
                    img = img.resize(target_size, Image.Resampling.LANCZOS)
                    return np.array(img)
                except ImportError:
                    # Fallback: use numpy interpolation (basic)
                    self.logger.warning("PIL not available, using basic resize")
                    return data  # Return as-is if can't resize
            return data
        
        elif step == 'reshape':
            # Add batch dimension if needed
            if len(data.shape) == 2:  # (H, W) -> (1, H, W, 1)
                return data.reshape(1, *data.shape, 1)
            elif len(data.shape) == 3:  # (H, W, C) -> (1, H, W, C)
                return data.reshape(1, *data.shape)
            return data
        
        else:
            self.logger.warning(f"Unknown preprocessing step: {step}")
            return data
    
    def cleanup(self) -> None:
        """Default cleanup implementation"""
        self.model = None
        self.is_loaded = False
        self.logger.info("Model resources cleaned up")

class ModelPluginError(Exception):
    """Custom exception for model plugin errors"""
    pass

class ModelLoadError(ModelPluginError):
    """Exception raised when model loading fails"""
    pass

class InferenceError(ModelPluginError):
    """Exception raised when inference fails"""
    pass

class PreprocessingError(ModelPluginError):
    """Exception raised when preprocessing fails"""
    pass

# Utility functions for plugin development
def timing_decorator(func):
    """Decorator to measure function execution time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # If result is a Prediction object, set the timing
        if hasattr(result, 'inference_time'):
            result.inference_time = end_time - start_time
        
        return result
    return wrapper

def validate_input_shape(expected_shape: Tuple[int, ...], actual_shape: Tuple[int, ...], allow_batch: bool = True) -> bool:
    """
    Validate input shape against expected shape
    
    Args:
        expected_shape: Expected input shape
        actual_shape: Actual input shape
        allow_batch: Whether to allow batch dimension
    
    Returns:
        True if shapes are compatible
    """
    if allow_batch and len(actual_shape) == len(expected_shape) + 1:
        # Check all dimensions except batch dimension
        return actual_shape[1:] == expected_shape
    else:
        return actual_shape == expected_shape

def convert_to_numpy(data: Any) -> np.ndarray:
    """
    Convert various data types to numpy array
    
    Args:
        data: Input data (tensor, list, PIL image, etc.)
    
    Returns:
        NumPy array
    """
    if isinstance(data, np.ndarray):
        return data
    
    # Handle PIL Images
    try:
        from PIL import Image
        if isinstance(data, Image.Image):
            return np.array(data)
    except ImportError:
        pass
    
    # Handle TensorFlow tensors
    try:
        import tensorflow as tf
        if isinstance(data, tf.Tensor):
            return data.numpy()
    except ImportError:
        pass
    
    # Handle PyTorch tensors
    try:
        import torch
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
    except ImportError:
        pass
    
    # Handle lists and other sequences
    if hasattr(data, '__iter__') and not isinstance(data, str):
        return np.array(data)
    
    raise ValueError(f"Cannot convert {type(data)} to numpy array")