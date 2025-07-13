"""Model registry for managing available super-resolution models."""

from typing import Dict, Type, Optional, List

from .base_upscaler import BaseUpscaler


class ModelRegistry:
    """Central registry for all upscaler models."""
    
    _models: Dict[str, Type[BaseUpscaler]] = {}
    _aliases: Dict[str, str] = {}
    _metadata: Dict[str, Dict[str, any]] = {}
    
    @classmethod
    def register(
        cls,
        name: str,
        model_class: Type[BaseUpscaler],
        aliases: Optional[List[str]] = None,
        metadata: Optional[Dict[str, any]] = None,
    ) -> None:
        """Register a model class.
        
        Args:
            name: Primary model name
            model_class: Model class (subclass of BaseUpscaler)
            aliases: Alternative names for the model
            metadata: Additional model information (description, paper, etc.)
        """
        if not issubclass(model_class, BaseUpscaler):
            raise TypeError(f"{model_class} must be a subclass of BaseUpscaler")
        
        if name in cls._models:
            raise ValueError(f"Model '{name}' is already registered")
        
        # Register primary name
        cls._models[name] = model_class
        
        # Register aliases
        if aliases:
            for alias in aliases:
                if alias in cls._aliases:
                    raise ValueError(f"Alias '{alias}' is already registered")
                cls._aliases[alias] = name
        
        # Store metadata
        if metadata:
            cls._metadata[name] = metadata
    
    @classmethod
    def get(cls, name: str) -> Type[BaseUpscaler]:
        """Get model class by name or alias.
        
        Args:
            name: Model name or alias
            
        Returns:
            Model class
            
        Raises:
            ValueError: If model not found
        """
        # Check if it's an alias
        if name in cls._aliases:
            name = cls._aliases[name]
        
        if name not in cls._models:
            available = cls.list_models()
            raise ValueError(
                f"Unknown model: '{name}'. Available models: {', '.join(available)}"
            )
        
        return cls._models[name]
    
    @classmethod
    def list_models(cls, include_aliases: bool = False) -> List[str]:
        """List all registered models.
        
        Args:
            include_aliases: Whether to include aliases in the list
            
        Returns:
            List of model names
        """
        models = list(cls._models.keys())
        
        if include_aliases:
            models.extend(cls._aliases.keys())
        
        return sorted(models)
    
    @classmethod
    def get_metadata(cls, name: str) -> Dict[str, any]:
        """Get model metadata.
        
        Args:
            name: Model name or alias
            
        Returns:
            Model metadata dictionary
        """
        # Resolve alias
        if name in cls._aliases:
            name = cls._aliases[name]
        
        return cls._metadata.get(name, {})
    
    @classmethod
    def get_aliases(cls, name: str) -> List[str]:
        """Get all aliases for a model.
        
        Args:
            name: Primary model name
            
        Returns:
            List of aliases
        """
        return [alias for alias, primary in cls._aliases.items() if primary == name]
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registrations (mainly for testing)."""
        cls._models.clear()
        cls._aliases.clear()
        cls._metadata.clear()


# Decorator for easy registration
def register_model(
    name: str,
    aliases: Optional[List[str]] = None,
    metadata: Optional[Dict[str, any]] = None,
):
    """Decorator to register a model class.
    
    Usage:
        @register_model("my-model", aliases=["MyModel"])
        class MyModelUpscaler(BaseUpscaler):
            ...
    """
    def decorator(cls: Type[BaseUpscaler]) -> Type[BaseUpscaler]:
        ModelRegistry.register(name, cls, aliases, metadata)
        return cls
    
    return decorator