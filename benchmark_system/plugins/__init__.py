"""
Model Plugin Implementations
Specific implementations for different model architectures
"""

import os
import sys
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interfaces import (
    IModelPlugin, BaseModelPlugin, ModelType, InputFormat, OutputFormat,
    ModelInfo, ProcessedInput, Prediction, timing_decorator, convert_to_numpy,
    validate_input_shape, ModelLoadError, InferenceError, PreprocessingError
)

# Configure logging
logger = logging.getLogger(__name__)

class CNNModelPlugin(BaseModelPlugin):
    """
    CNN Model Plugin optimized for SketchRNN and similar CNN architectures
    Supports TensorFlow/Keras CNN models for image classification
    """
    
    def __init__(self):
        super().__init__(ModelType.CNN)
        self.input_shape = (28, 28, 1)  # Default for SketchRNN
        self.num_classes = 28  # Default for QuickDraw categories
        
    def load_model(self, config: Dict[str, Any]) -> Any:
        """
        Load CNN model from configuration
        
        Args:
            config: Configuration dictionary containing model_path and other settings
            
        Returns:
            Loaded TensorFlow/Keras model
        """
        try:
            # Validate configuration
            if not self.validate_config(config):
                raise ValueError("Invalid configuration")
            
            model_path = Path(config['model_path'])
            
            # Load TensorFlow/Keras model
            try:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(str(model_path), compile=False)
                self.logger.info(f"Successfully loaded Keras model from {model_path}")
            except ImportError:
                raise ModelLoadError("TensorFlow not available for CNN model loading")
            except Exception as e:
                raise ModelLoadError(f"Failed to load Keras model: {str(e)}")
            
            # Extract model information
            self._extract_model_info(config)
            
            # Setup preprocessing
            self._setup_preprocessing(config)
            
            # Load categories if provided
            if 'categories_path' in config:
                self.load_categories(config['categories_path'])
            elif 'categories' in config:
                self.categories = config['categories']
            
            self.config = config
            self.is_loaded = True
            
            return self.model
            
        except Exception as e:
            self.logger.error(f"Failed to load CNN model: {str(e)}")
            raise ModelLoadError(f"CNN model loading failed: {str(e)}")
    
    def _extract_model_info(self, config: Dict[str, Any]):
        """Extract information from the loaded model"""
        try:
            if hasattr(self.model, 'input_shape') and self.model.input_shape:
                self.input_shape = self.model.input_shape[1:]  # Remove batch dimension
            
            if hasattr(self.model, 'output_shape') and self.model.output_shape:
                output_shape = self.model.output_shape
                if isinstance(output_shape, tuple) and len(output_shape) >= 2:
                    self.num_classes = output_shape[-1]  # Last dimension is usually classes
            
            # Override with config if specified
            if 'input_shape' in config:
                self.input_shape = tuple(config['input_shape'])
            if 'num_classes' in config:
                self.num_classes = config['num_classes']
                
            self.logger.info(f"Model info - Input shape: {self.input_shape}, Classes: {self.num_classes}")
            
        except Exception as e:
            self.logger.warning(f"Failed to extract model info: {e}")
    
    @timing_decorator
    def preprocess_input(self, data: Any) -> ProcessedInput:
        """
        Preprocess input data for CNN inference
        
        Args:
            data: Raw input data (numpy array, PIL image, etc.)
            
        Returns:
            ProcessedInput object with preprocessed data
        """
        start_time = time.time()
        
        try:
            # Convert to numpy array
            if not isinstance(data, np.ndarray):
                data = convert_to_numpy(data)
            
            original_shape = data.shape
            
            # Handle different input formats
            processed_data = self._preprocess_image(data)
            
            preprocessing_time = time.time() - start_time
            
            return ProcessedInput(
                data=processed_data,
                original_shape=original_shape,
                processed_shape=processed_data.shape,
                batch_size=processed_data.shape[0] if len(processed_data.shape) > 3 else 1,
                preprocessing_time=preprocessing_time,
                metadata={
                    'preprocessing_steps': self.preprocessing_steps,
                    'input_format': 'image',
                    'normalized': True
                }
            )
            
        except Exception as e:
            raise PreprocessingError(f"CNN preprocessing failed: {str(e)}")
    
    def _preprocess_image(self, data: np.ndarray) -> np.ndarray:
        """
        Preprocess image data for CNN model
        
        Args:
            data: Input image data
            
        Returns:
            Preprocessed image data
        """
        # Handle different input shapes
        if len(data.shape) == 2:  # Grayscale image (H, W)
            data = data.reshape(*data.shape, 1)  # Add channel dimension
        elif len(data.shape) == 3 and data.shape[-1] == 3:  # RGB image
            # Convert RGB to grayscale if model expects grayscale
            if self.input_shape[-1] == 1:
                data = np.dot(data[...,:3], [0.2989, 0.5870, 0.1140])
                data = data.reshape(*data.shape, 1)
        
        # Resize if necessary
        target_height, target_width = self.input_shape[:2]
        if data.shape[:2] != (target_height, target_width):
            data = self._resize_image(data, (target_height, target_width))
        
        # Normalize pixel values to [0, 1]
        if data.dtype == np.uint8:
            data = data.astype(np.float32) / 255.0
        elif data.max() > 1.0:
            data = data.astype(np.float32) / data.max()
        else:
            data = data.astype(np.float32)
        
        # Add batch dimension if not present
        if len(data.shape) == 3:  # (H, W, C)
            data = np.expand_dims(data, axis=0)  # (1, H, W, C)
        
        # Validate final shape
        expected_shape = (1,) + self.input_shape
        if data.shape != expected_shape:
            self.logger.warning(f"Shape mismatch: got {data.shape}, expected {expected_shape}")
            # Try to reshape if possible
            if np.prod(data.shape) == np.prod(expected_shape):
                data = data.reshape(expected_shape)
            else:
                raise ValueError(f"Cannot reshape {data.shape} to {expected_shape}")
        
        return data
    
    def _resize_image(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image to target size
        
        Args:
            image: Input image
            target_size: Target (height, width)
            
        Returns:
            Resized image
        """
        try:
            from PIL import Image
            
            # Convert to PIL Image
            if len(image.shape) == 3 and image.shape[-1] == 1:
                # Grayscale with channel dimension
                pil_image = Image.fromarray(image.squeeze(-1))
            elif len(image.shape) == 2:
                # Grayscale without channel dimension
                pil_image = Image.fromarray(image)
            else:
                # RGB or other formats
                pil_image = Image.fromarray(image.astype(np.uint8))
            
            # Resize
            resized = pil_image.resize((target_size[1], target_size[0]), Image.Resampling.LANCZOS)
            result = np.array(resized)
            
            # Restore original shape format
            if len(image.shape) == 3:
                result = result.reshape(*result.shape, image.shape[-1])
            
            return result
            
        except ImportError:
            self.logger.warning("PIL not available, using basic resize")
            # Fallback: simple nearest neighbor resize using numpy
            from scipy import ndimage
            try:
                zoom_factors = [target_size[0] / image.shape[0], target_size[1] / image.shape[1]]
                if len(image.shape) == 3:
                    zoom_factors.append(1.0)  # Don't resize channel dimension
                return ndimage.zoom(image, zoom_factors, order=1)
            except ImportError:
                self.logger.warning("SciPy not available, returning original image")
                return image
    
    @timing_decorator
    def predict(self, input_data: ProcessedInput) -> Prediction:
        """
        Run CNN inference on preprocessed input
        
        Args:
            input_data: ProcessedInput from preprocess_input
            
        Returns:
            Prediction object with results and timing
        """
        start_time = time.time()
        
        try:
            if not self.is_model_loaded():
                raise InferenceError("Model not loaded")
            
            # Run inference
            try:
                raw_output = self.model.predict(input_data.data, verbose=0)
            except Exception as e:
                raise InferenceError(f"Model prediction failed: {str(e)}")
            
            inference_time = time.time() - start_time
            
            # Process output
            probabilities = raw_output[0] if len(raw_output.shape) > 1 else raw_output
            predicted_class_idx = int(np.argmax(probabilities))
            confidence = float(np.max(probabilities))
            
            # Map to category names if available
            if self.categories and predicted_class_idx < len(self.categories):
                predicted_class = self.categories[predicted_class_idx]
            else:
                predicted_class = str(predicted_class_idx)
            
            return Prediction(
                raw_output=raw_output,
                probabilities=probabilities,
                predicted_classes=[predicted_class],
                confidence_scores=np.array([confidence]),
                inference_time=inference_time,
                batch_size=input_data.batch_size,
                metadata={
                    'model_type': 'cnn',
                    'predicted_class_idx': predicted_class_idx,
                    'top_k_classes': self._get_top_k_predictions(probabilities, k=5)
                }
            )
            
        except Exception as e:
            if not isinstance(e, InferenceError):
                raise InferenceError(f"CNN inference failed: {str(e)}")
            raise
    
    def _get_top_k_predictions(self, probabilities: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Get top-k predictions with class names and probabilities"""
        top_indices = np.argsort(probabilities)[-k:][::-1]
        top_predictions = []
        
        for idx in top_indices:
            class_name = self.categories[idx] if self.categories and idx < len(self.categories) else str(idx)
            top_predictions.append({
                'class': class_name,
                'class_idx': int(idx),
                'probability': float(probabilities[idx])
            })
        
        return top_predictions
    
    def get_model_info(self) -> ModelInfo:
        """
        Get comprehensive CNN model information
        
        Returns:
            ModelInfo object with model specifications
        """
        # Calculate model parameters if possible
        num_parameters = None
        model_size_mb = None
        
        try:
            if hasattr(self.model, 'count_params'):
                num_parameters = self.model.count_params()
                # Rough estimate: 4 bytes per parameter (float32)
                model_size_mb = (num_parameters * 4) / (1024 * 1024)
        except Exception as e:
            self.logger.debug(f"Could not calculate model parameters: {e}")
        
        # Determine output shape
        output_shape = (self.num_classes,)
        if hasattr(self.model, 'output_shape'):
            try:
                output_shape = self.model.output_shape[1:]  # Remove batch dimension
            except:
                pass
        
        return ModelInfo(
            name=f"CNN_{self.config.get('name', 'SketchRNN')}",
            version=self.config.get('version', '1.0.0'),
            model_type=ModelType.CNN,
            input_format=InputFormat.IMAGE_3D,
            output_format=OutputFormat.CLASSIFICATION,
            input_shape=self.input_shape,
            output_shape=output_shape,
            num_parameters=num_parameters,
            model_size_mb=model_size_mb,
            supported_batch_sizes=[1, 8, 16, 32, 64, 128, 256],
            preprocessing_steps=['resize', 'normalize', 'reshape'],
            postprocessing_steps=['softmax', 'argmax'],
            resource_requirements={
                'min_memory_mb': model_size_mb * 2 if model_size_mb else 100,
                'preferred_device': 'GPU',
                'supports_batching': True,
                'supports_mixed_precision': True
            },
            metadata={
                'framework': 'tensorflow',
                'categories': self.categories,
                'input_channels': self.input_shape[-1] if len(self.input_shape) >= 3 else 1,
                'architecture_details': self._get_architecture_summary()
            }
        )
    
    def _get_architecture_summary(self) -> Dict[str, Any]:
        """Get a summary of the model architecture"""
        try:
            if hasattr(self.model, 'layers'):
                layer_info = []
                for layer in self.model.layers:
                    layer_info.append({
                        'name': layer.name,
                        'type': type(layer).__name__,
                        'output_shape': str(layer.output_shape) if hasattr(layer, 'output_shape') else 'unknown'
                    })
                
                return {
                    'total_layers': len(self.model.layers),
                    'layer_summary': layer_info[:10],  # First 10 layers
                    'trainable_params': self.model.count_params() if hasattr(self.model, 'count_params') else None
                }
        except Exception as e:
            self.logger.debug(f"Could not generate architecture summary: {e}")
        
        return {'summary': 'Architecture details not available'}
    
    def batch_predict(self, input_data_list: List[ProcessedInput]) -> List[Prediction]:
        """
        Optimized batch prediction for CNN models
        
        Args:
            input_data_list: List of ProcessedInput objects
            
        Returns:
            List of Prediction objects
        """
        if not input_data_list:
            return []
        
        try:
            # Check if all inputs have the same shape for efficient batching
            first_shape = input_data_list[0].processed_shape[1:]  # Remove batch dimension
            can_batch = all(
                inp.processed_shape[1:] == first_shape 
                for inp in input_data_list
            )
            
            if can_batch and len(input_data_list) > 1:
                # Efficient batch processing
                return self._batch_predict_optimized(input_data_list)
            else:
                # Fall back to individual predictions
                return super().batch_predict(input_data_list)
                
        except Exception as e:
            self.logger.warning(f"Batch prediction optimization failed: {e}")
            return super().batch_predict(input_data_list)
    
    def _batch_predict_optimized(self, input_data_list: List[ProcessedInput]) -> List[Prediction]:
        """Optimized batch prediction with tensor concatenation"""
        start_time = time.time()
        
        # Concatenate all inputs into a single batch
        batch_data = np.concatenate([inp.data for inp in input_data_list], axis=0)
        
        # Run batch inference
        batch_output = self.model.predict(batch_data, verbose=0)
        
        inference_time = time.time() - start_time
        
        # Split results back to individual predictions
        predictions = []
        for i, input_data in enumerate(input_data_list):
            probabilities = batch_output[i]
            predicted_class_idx = int(np.argmax(probabilities))
            confidence = float(np.max(probabilities))
            
            predicted_class = (
                self.categories[predicted_class_idx] 
                if self.categories and predicted_class_idx < len(self.categories)
                else str(predicted_class_idx)
            )
            
            prediction = Prediction(
                raw_output=batch_output[i:i+1],  # Keep batch dimension for consistency
                probabilities=probabilities,
                predicted_classes=[predicted_class],
                confidence_scores=np.array([confidence]),
                inference_time=inference_time / len(input_data_list),  # Approximate per-sample time
                batch_size=1,
                metadata={
                    'model_type': 'cnn',
                    'batch_optimized': True,
                    'predicted_class_idx': predicted_class_idx
                }
            )
            predictions.append(prediction)
        
        return predictions
    
    def cleanup(self) -> None:
        """Clean up CNN model resources"""
        try:
            if self.model is not None:
                # Clear TensorFlow session if needed
                try:
                    import tensorflow as tf
                    tf.keras.backend.clear_session()
                except:
                    pass
                
                self.model = None
            
            super().cleanup()
            self.logger.info("CNN model resources cleaned up")
            
        except Exception as e:
            self.logger.warning(f"Error during CNN cleanup: {e}")

class GenericModelPlugin(BaseModelPlugin):
    """
    Generic model plugin that can handle various model types
    Serves as fallback when specific plugins are not available
    """
    
    def __init__(self):
        super().__init__(ModelType.CUSTOM)
    
    def load_model(self, config: Dict[str, Any]) -> Any:
        """Load model using factory (should be pre-loaded)"""
        if self.model is None:
            raise ModelLoadError("Generic plugin requires pre-loaded model")
        return self.model
    
    def preprocess_input(self, data: Any) -> ProcessedInput:
        """Generic preprocessing"""
        start_time = time.time()
        
        # Convert to numpy
        if not isinstance(data, np.ndarray):
            data = convert_to_numpy(data)
        
        original_shape = data.shape
        processed_data = data
        
        # Basic normalization
        if processed_data.dtype == np.uint8:
            processed_data = processed_data.astype(np.float32) / 255.0
        
        # Add batch dimension if needed
        if len(processed_data.shape) == 2:
            processed_data = processed_data.reshape(1, *processed_data.shape, 1)
        elif len(processed_data.shape) == 3:
            processed_data = processed_data.reshape(1, *processed_data.shape)
        
        preprocessing_time = time.time() - start_time
        
        return ProcessedInput(
            data=processed_data,
            original_shape=original_shape,
            processed_shape=processed_data.shape,
            batch_size=1,
            preprocessing_time=preprocessing_time,
            metadata={'plugin_type': 'generic'}
        )
    
    def predict(self, input_data: ProcessedInput) -> Prediction:
        """Generic prediction"""
        start_time = time.time()
        
        try:
            # Try different inference methods
            if hasattr(self.model, 'predict'):
                output = self.model.predict(input_data.data, verbose=0)
            elif hasattr(self.model, '__call__'):
                output = self.model(input_data.data)
                if hasattr(output, 'numpy'):
                    output = output.numpy()
            elif hasattr(self.model, 'run'):
                # ONNX
                input_name = self.model.get_inputs()[0].name
                output = self.model.run(None, {input_name: input_data.data})[0]
            else:
                raise InferenceError("Unknown model interface")
            
            inference_time = time.time() - start_time
            
            # Process output
            if len(output.shape) > 1 and output.shape[0] == 1:
                output = output[0]
            
            predicted_class_idx = int(np.argmax(output)) if len(output.shape) >= 1 else 0
            confidence = float(np.max(output)) if len(output.shape) >= 1 else 0.0
            
            return Prediction(
                raw_output=output,
                probabilities=output,
                predicted_classes=[str(predicted_class_idx)],
                confidence_scores=np.array([confidence]),
                inference_time=inference_time,
                batch_size=input_data.batch_size,
                metadata={'plugin_type': 'generic'}
            )
            
        except Exception as e:
            raise InferenceError(f"Generic prediction failed: {str(e)}")
    
    def get_model_info(self) -> ModelInfo:
        """Get generic model information"""
        return ModelInfo(
            name="GenericModel",
            version="1.0.0",
            model_type=ModelType.CUSTOM,
            input_format=InputFormat.IMAGE_3D,
            output_format=OutputFormat.CLASSIFICATION,
            input_shape=(28, 28, 1),
            output_shape=(28,),
            preprocessing_steps=['normalize'],
            metadata={'plugin_type': 'generic'}
        )

# Additional specialized plugins can be added here
# For example: TransformerModelPlugin, RNNModelPlugin, etc.

# Export all plugins
__all__ = [
    'CNNModelPlugin',
    'GenericModelPlugin'
]