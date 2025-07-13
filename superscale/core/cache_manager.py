"""Model cache manager for efficient memory usage."""

import gc
import weakref
from collections import OrderedDict
from typing import Optional, Any, List

import torch


class CacheManager:
    """Manages loaded models with LRU eviction.
    
    This class implements a Least Recently Used (LRU) cache for loaded models,
    automatically evicting old models when the cache is full.
    """
    
    def __init__(self, max_models: int = 3):
        """Initialize cache manager.
        
        Args:
            max_models: Maximum number of models to keep in memory
        """
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._max_models = max_models
        self._weak_refs: dict[str, weakref.ref] = {}
    
    def get(self, model_name: str) -> Optional[Any]:
        """Get model from cache.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Cached model or None if not found
        """
        if model_name in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(model_name)
            return self._cache[model_name]
        
        # Check weak reference
        if model_name in self._weak_refs:
            ref = self._weak_refs[model_name]
            model = ref()
            if model is not None:
                # Restore to strong reference
                self._cache[model_name] = model
                self._cache.move_to_end(model_name)
                return model
            else:
                # Clean up dead reference
                del self._weak_refs[model_name]
        
        return None
    
    def put(self, model_name: str, model: Any) -> None:
        """Add model to cache.
        
        Args:
            model_name: Name of the model
            model: Model instance
        """
        # Check if already in cache
        if model_name in self._cache:
            self._cache.move_to_end(model_name)
            return
        
        # Check if we need to evict
        while len(self._cache) >= self._max_models:
            # Evict least recently used
            evicted_name, evicted_model = self._cache.popitem(last=False)
            self._cleanup_model(evicted_name, evicted_model)
        
        # Add to cache
        self._cache[model_name] = model
    
    def _cleanup_model(self, model_name: str, model: Any) -> None:
        """Clean up evicted model.
        
        Args:
            model_name: Name of the model
            model: Model instance to clean up
        """
        # Move to weak reference
        self._weak_refs[model_name] = weakref.ref(model)
        
        # Move model to CPU to free GPU memory
        if hasattr(model, 'to'):
            try:
                model.to('cpu')
            except:
                pass  # Some models might not support .to()
        
        # Trigger garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def clear(self, model_name: Optional[str] = None) -> None:
        """Clear specific model or all models from cache.
        
        Args:
            model_name: Name of model to clear, or None to clear all
        """
        if model_name:
            # Clear specific model
            if model_name in self._cache:
                model = self._cache.pop(model_name)
                self._cleanup_model(model_name, model)
                
                # Force deletion
                del model
                gc.collect()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        else:
            # Clear all models
            models_to_clear = list(self._cache.keys())
            for name in models_to_clear:
                self.clear(name)
            
            # Clear weak references
            self._weak_refs.clear()
    
    def list_cached(self) -> List[str]:
        """List currently cached models.
        
        Returns:
            List of model names in cache
        """
        return list(self._cache.keys())
    
    def set_max_models(self, max_models: int) -> None:
        """Update maximum number of models.
        
        Args:
            max_models: New maximum
        """
        self._max_models = max_models
        
        # Evict models if necessary
        while len(self._cache) > self._max_models:
            evicted_name, evicted_model = self._cache.popitem(last=False)
            self._cleanup_model(evicted_name, evicted_model)
    
    def get_memory_usage(self) -> dict:
        """Get memory usage statistics.
        
        Returns:
            Dictionary with memory usage info
        """
        stats = {
            "cached_models": len(self._cache),
            "max_models": self._max_models,
            "weak_refs": len(self._weak_refs),
        }
        
        if torch.cuda.is_available():
            stats["gpu_memory_allocated"] = torch.cuda.memory_allocated()
            stats["gpu_memory_reserved"] = torch.cuda.memory_reserved()
        
        return stats
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CacheManager(cached={len(self._cache)}/{self._max_models}, "
            f"weak_refs={len(self._weak_refs)})"
        )


# Global cache manager instance
_cache_manager = CacheManager()


def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance.
    
    Returns:
        Global CacheManager instance
    """
    return _cache_manager