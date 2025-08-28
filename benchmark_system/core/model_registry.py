"""
Model Registry System with Plugin Management
Centralized system for registering, managing, and retrieving model plugins
"""

import json
import logging
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Type, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interfaces import IModelPlugin, ModelType, ModelInfo, ModelPluginError
from core.model_factory import UniversalModelFactory

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class ModelRegistration:
    """Information about a registered model"""
    model_id: str
    name: str
    version: str
    model_type: ModelType
    plugin_class: str
    config: Dict[str, Any]
    model_path: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    registered_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    usage_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = asdict(self)
        result['model_type'] = self.model_type.value
        result['registered_at'] = self.registered_at.isoformat()
        result['last_used'] = self.last_used.isoformat() if self.last_used else None
        return result

@dataclass
class RegistryConfig:
    """Configuration for model registry"""
    auto_discovery: bool = True
    cache_plugins: bool = True
    max_cached_plugins: int = 10
    registry_file: str = "model_registry.json"
    auto_save: bool = True
    validate_on_load: bool = True
    enable_versioning: bool = True
    backup_registry: bool = True

class ModelRegistry:
    """
    Centralized registry for managing model plugins
    Provides registration, discovery, caching, and lifecycle management
    """
    
    def __init__(self, config: Optional[RegistryConfig] = None):
        self.config = config or RegistryConfig()
        self._registered_models: Dict[str, ModelRegistration] = {}
        self._plugin_cache: Dict[str, IModelPlugin] = {}
        self._plugin_classes: Dict[str, Type[IModelPlugin]] = {}
        self._lock = threading.RLock()
        self._factory = UniversalModelFactory()
        
        # Statistics
        self._stats = {
            'total_registered': 0,
            'total_loaded': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0
        }
        
        # Load existing registry
        self._load_registry()
        
        # Auto-discovery if enabled
        if self.config.auto_discovery:
            self._auto_discover_models()
    
    def register_model(self, 
                      model_id: str,
                      name: str,
                      model_path: Union[str, Path],
                      version: str = "1.0.0",
                      description: str = "",
                      tags: Optional[List[str]] = None,
                      config: Optional[Dict[str, Any]] = None,
                      model_type: Optional[ModelType] = None,
                      **metadata) -> bool:
        """
        Register a new model in the registry
        
        Args:
            model_id: Unique identifier for the model
            name: Human-readable name
            model_path: Path to model file
            version: Model version
            description: Model description
            tags: List of tags for categorization
            config: Additional configuration
            model_type: Explicit model type (will be auto-detected if None)
            **metadata: Additional metadata
            
        Returns:
            True if registration successful, False otherwise
        """
        with self._lock:
            try:
                # Validate inputs
                if model_id in self._registered_models:
                    if self.config.enable_versioning:
                        # Create versioned ID
                        original_id = model_id
                        counter = 1
                        while model_id in self._registered_models:
                            model_id = f"{original_id}_v{counter}"
                            counter += 1
                        logger.info(f"Model ID conflict resolved, using: {model_id}")
                    else:
                        logger.warning(f"Model {model_id} already registered, skipping")
                        return False
                
                model_path = Path(model_path)
                if not model_path.exists():
                    raise FileNotFoundError(f"Model file not found: {model_path}")
                
                # Prepare configuration
                model_config = config or {}
                model_config['model_path'] = str(model_path.absolute())
                
                # Auto-detect model type if not provided
                if model_type is None:
                    try:
                        # Create temporary plugin to detect type
                        temp_plugin = self._factory.create_plugin_from_config(model_config)
                        model_type = temp_plugin.model_type
                        # Clean up temporary plugin
                        temp_plugin.cleanup()
                        del temp_plugin
                    except Exception as e:
                        logger.warning(f"Failed to auto-detect model type: {e}")
                        model_type = ModelType.CUSTOM
                
                # Create registration
                registration = ModelRegistration(
                    model_id=model_id,
                    name=name,
                    version=version,
                    model_type=model_type,
                    plugin_class=self._get_plugin_class_name(model_type),
                    config=model_config,
                    model_path=str(model_path.absolute()),
                    description=description,
                    tags=tags or [],
                    metadata=metadata
                )
                
                # Validate registration if enabled
                if self.config.validate_on_load:
                    if not self._validate_registration(registration):
                        logger.error(f"Registration validation failed for {model_id}")
                        return False
                
                # Register the model
                self._registered_models[model_id] = registration
                self._stats['total_registered'] += 1
                
                logger.info(f"Successfully registered model: {model_id} ({name})")
                
                # Auto-save if enabled
                if self.config.auto_save:
                    self._save_registry()
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to register model {model_id}: {str(e)}")
                self._stats['errors'] += 1
                return False
    
    def get_model(self, model_id: str, force_reload: bool = False) -> Optional[IModelPlugin]:
        """
        Get a model plugin by ID
        
        Args:
            model_id: Model identifier
            force_reload: Force reload even if cached
            
        Returns:
            IModelPlugin instance or None if not found
        """
        with self._lock:
            try:
                if model_id not in self._registered_models:
                    logger.warning(f"Model {model_id} not found in registry")
                    return None
                
                registration = self._registered_models[model_id]
                
                # Check cache first
                if not force_reload and self.config.cache_plugins and model_id in self._plugin_cache:
                    plugin = self._plugin_cache[model_id]
                    if plugin.is_model_loaded():
                        self._stats['cache_hits'] += 1
                        self._update_usage_stats(model_id)
                        return plugin
                
                # Load plugin
                self._stats['cache_misses'] += 1
                plugin = self._factory.create_plugin_from_config(registration.config)
                
                # Cache the plugin if caching is enabled
                if self.config.cache_plugins:
                    self._add_to_cache(model_id, plugin)
                
                self._stats['total_loaded'] += 1
                self._update_usage_stats(model_id)
                
                logger.info(f"Successfully loaded model plugin: {model_id}")
                return plugin
                
            except Exception as e:
                logger.error(f"Failed to load model {model_id}: {str(e)}")
                self._stats['errors'] += 1
                return None
    
    def list_models(self, 
                    model_type: Optional[ModelType] = None,
                    tags: Optional[List[str]] = None,
                    active_only: bool = True) -> List[ModelRegistration]:
        """
        List registered models with optional filtering
        
        Args:
            model_type: Filter by model type
            tags: Filter by tags (model must have all specified tags)
            active_only: Only return active models
            
        Returns:
            List of matching model registrations
        """
        with self._lock:
            models = list(self._registered_models.values())
            
            # Apply filters
            if active_only:
                models = [m for m in models if m.is_active]
            
            if model_type:
                models = [m for m in models if m.model_type == model_type]
            
            if tags:
                models = [m for m in models if all(tag in m.tags for tag in tags)]
            
            # Sort by usage count and last used
            models.sort(key=lambda x: (x.usage_count, x.last_used or datetime.min), reverse=True)
            
            return models
    
    def remove_model(self, model_id: str, cleanup: bool = True) -> bool:
        """
        Remove a model from the registry
        
        Args:
            model_id: Model identifier
            cleanup: Whether to cleanup cached plugin
            
        Returns:
            True if removal successful
        """
        with self._lock:
            try:
                if model_id not in self._registered_models:
                    logger.warning(f"Model {model_id} not found for removal")
                    return False
                
                # Cleanup cached plugin
                if cleanup and model_id in self._plugin_cache:
                    plugin = self._plugin_cache[model_id]
                    plugin.cleanup()
                    del self._plugin_cache[model_id]
                
                # Remove from registry
                del self._registered_models[model_id]
                
                logger.info(f"Successfully removed model: {model_id}")
                
                # Auto-save if enabled
                if self.config.auto_save:
                    self._save_registry()
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to remove model {model_id}: {str(e)}")
                return False
    
    def update_model(self, model_id: str, **updates) -> bool:
        """
        Update model registration information
        
        Args:
            model_id: Model identifier
            **updates: Fields to update
            
        Returns:
            True if update successful
        """
        with self._lock:
            try:
                if model_id not in self._registered_models:
                    logger.warning(f"Model {model_id} not found for update")
                    return False
                
                registration = self._registered_models[model_id]
                
                # Update allowed fields
                updatable_fields = ['name', 'description', 'tags', 'metadata', 'is_active', 'config']
                for field, value in updates.items():
                    if field in updatable_fields:
                        setattr(registration, field, value)
                    else:
                        logger.warning(f"Field {field} is not updatable")
                
                logger.info(f"Successfully updated model: {model_id}")
                
                # Auto-save if enabled
                if self.config.auto_save:
                    self._save_registry()
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to update model {model_id}: {str(e)}")
                return False
    
    def search_models(self, query: str) -> List[ModelRegistration]:
        """
        Search models by name, description, or tags
        
        Args:
            query: Search query
            
        Returns:
            List of matching model registrations
        """
        query_lower = query.lower()
        results = []
        
        for registration in self._registered_models.values():
            if not registration.is_active:
                continue
            
            # Search in name, description, and tags
            searchable_text = [
                registration.name.lower(),
                registration.description.lower(),
                *[tag.lower() for tag in registration.tags],
                registration.model_id.lower()
            ]
            
            if any(query_lower in text for text in searchable_text):
                results.append(registration)
        
        return results
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics
        
        Returns:
            Dictionary containing registry statistics
        """
        with self._lock:
            active_models = sum(1 for m in self._registered_models.values() if m.is_active)
            
            type_distribution = defaultdict(int)
            for registration in self._registered_models.values():
                if registration.is_active:
                    type_distribution[registration.model_type.value] += 1
            
            return {
                'total_registered': len(self._registered_models),
                'active_models': active_models,
                'cached_plugins': len(self._plugin_cache),
                'type_distribution': dict(type_distribution),
                'performance_stats': self._stats.copy(),
                'cache_hit_rate': (
                    self._stats['cache_hits'] / 
                    max(self._stats['cache_hits'] + self._stats['cache_misses'], 1)
                ) * 100
            }
    
    def export_registry(self, filepath: Union[str, Path]) -> bool:
        """
        Export registry to JSON file
        
        Args:
            filepath: Output file path
            
        Returns:
            True if export successful
        """
        try:
            registry_data = {
                'config': asdict(self.config),
                'models': {
                    model_id: registration.to_dict()
                    for model_id, registration in self._registered_models.items()
                },
                'stats': self._stats,
                'exported_at': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(registry_data, f, indent=2)
            
            logger.info(f"Registry exported to: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export registry: {str(e)}")
            return False
    
    def import_registry(self, filepath: Union[str, Path], merge: bool = True) -> bool:
        """
        Import registry from JSON file
        
        Args:
            filepath: Input file path
            merge: Whether to merge with existing registry
            
        Returns:
            True if import successful
        """
        try:
            with open(filepath, 'r') as f:
                registry_data = json.load(f)
            
            if not merge:
                self._registered_models.clear()
                self._plugin_cache.clear()
            
            models_data = registry_data.get('models', {})
            for model_id, model_data in models_data.items():
                # Convert back to ModelRegistration
                model_data['model_type'] = ModelType(model_data['model_type'])
                model_data['registered_at'] = datetime.fromisoformat(model_data['registered_at'])
                if model_data['last_used']:
                    model_data['last_used'] = datetime.fromisoformat(model_data['last_used'])
                
                registration = ModelRegistration(**model_data)
                self._registered_models[model_id] = registration
            
            logger.info(f"Registry imported from: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import registry: {str(e)}")
            return False
    
    def cleanup_cache(self):
        """Clean up plugin cache"""
        with self._lock:
            for plugin in self._plugin_cache.values():
                plugin.cleanup()
            self._plugin_cache.clear()
            logger.info("Plugin cache cleaned up")
    
    def _add_to_cache(self, model_id: str, plugin: IModelPlugin):
        """Add plugin to cache with LRU eviction"""
        if len(self._plugin_cache) >= self.config.max_cached_plugins:
            # Find least recently used plugin
            lru_model_id = min(
                self._plugin_cache.keys(),
                key=lambda x: self._registered_models[x].last_used or datetime.min
            )
            lru_plugin = self._plugin_cache.pop(lru_model_id)
            lru_plugin.cleanup()
        
        self._plugin_cache[model_id] = plugin
    
    def _update_usage_stats(self, model_id: str):
        """Update usage statistics for a model"""
        registration = self._registered_models[model_id]
        registration.last_used = datetime.now()
        registration.usage_count += 1
    
    def _validate_registration(self, registration: ModelRegistration) -> bool:
        """Validate a model registration"""
        try:
            # Check if model file exists
            if not Path(registration.model_path).exists():
                logger.error(f"Model file not found: {registration.model_path}")
                return False
            
            # Try to create plugin (quick validation)
            plugin = self._factory.create_plugin_from_config(registration.config)
            plugin.cleanup()
            
            return True
            
        except Exception as e:
            logger.error(f"Registration validation failed: {str(e)}")
            return False
    
    def _get_plugin_class_name(self, model_type: ModelType) -> str:
        """Get plugin class name for model type"""
        class_mapping = {
            ModelType.CNN: 'CNNModelPlugin',
            ModelType.TRANSFORMER: 'TransformerModelPlugin',
            ModelType.RNN: 'RNNModelPlugin',
            ModelType.LSTM: 'LSTMModelPlugin',
            ModelType.GRU: 'GRUModelPlugin',
            ModelType.MLP: 'MLPModelPlugin',
            ModelType.RESNET: 'ResNetModelPlugin',
            ModelType.MOBILENET: 'MobileNetModelPlugin',
            ModelType.EFFICIENTNET: 'EfficientNetModelPlugin',
            ModelType.CUSTOM: 'GenericModelPlugin'
        }
        return class_mapping.get(model_type, 'GenericModelPlugin')
    
    def _auto_discover_models(self):
        """Automatically discover models in common locations"""
        try:
            # Common model file extensions
            extensions = ['.h5', '.keras', '.pb', '.pth', '.pt', '.onnx', '.pkl']
            
            # Search paths
            search_paths = [
                Path.cwd() / 'models',
                Path.cwd() / 'checkpoints',
                Path.cwd(),
                Path.cwd() / 'saved_models'
            ]
            
            discovered = 0
            for search_path in search_paths:
                if search_path.exists() and search_path.is_dir():
                    for ext in extensions:
                        for model_file in search_path.glob(f'*{ext}'):
                            try:
                                model_id = f"auto_{model_file.stem}"
                                if model_id not in self._registered_models:
                                    success = self.register_model(
                                        model_id=model_id,
                                        name=model_file.stem,
                                        model_path=model_file,
                                        description=f"Auto-discovered from {model_file}",
                                        tags=['auto-discovered'],
                                        metadata={'discovery_path': str(search_path)}
                                    )
                                    if success:
                                        discovered += 1
                            except Exception as e:
                                logger.debug(f"Failed to auto-register {model_file}: {e}")
            
            if discovered > 0:
                logger.info(f"Auto-discovered {discovered} models")
                
        except Exception as e:
            logger.warning(f"Auto-discovery failed: {str(e)}")
    
    def _load_registry(self):
        """Load registry from file"""
        registry_file = Path(self.config.registry_file)
        if registry_file.exists():
            try:
                self.import_registry(registry_file, merge=False)
                logger.info(f"Registry loaded from {registry_file}")
            except Exception as e:
                logger.warning(f"Failed to load registry: {str(e)}")
    
    def _save_registry(self):
        """Save registry to file"""
        try:
            # Backup existing registry if enabled
            registry_file = Path(self.config.registry_file)
            if self.config.backup_registry and registry_file.exists():
                backup_file = registry_file.with_suffix('.bak')
                registry_file.replace(backup_file)
            
            self.export_registry(registry_file)
            
        except Exception as e:
            logger.error(f"Failed to save registry: {str(e)}")

# Global registry instance
_global_registry = None
_registry_lock = threading.Lock()

def get_global_registry(config: Optional[RegistryConfig] = None) -> ModelRegistry:
    """Get or create global registry instance"""
    global _global_registry
    
    with _registry_lock:
        if _global_registry is None:
            _global_registry = ModelRegistry(config)
        return _global_registry

def reset_global_registry():
    """Reset global registry (for testing)"""
    global _global_registry
    
    with _registry_lock:
        if _global_registry:
            _global_registry.cleanup_cache()
            _global_registry = None

# Convenience functions
def register_model(*args, **kwargs) -> bool:
    """Register model using global registry"""
    return get_global_registry().register_model(*args, **kwargs)

def get_model(model_id: str, **kwargs) -> Optional[IModelPlugin]:
    """Get model using global registry"""
    return get_global_registry().get_model(model_id, **kwargs)

def list_models(**kwargs) -> List[ModelRegistration]:
    """List models using global registry"""
    return get_global_registry().list_models(**kwargs)

def search_models(query: str) -> List[ModelRegistration]:
    """Search models using global registry"""
    return get_global_registry().search_models(query)