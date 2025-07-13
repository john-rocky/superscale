"""Main pipeline for super-resolution."""

from pathlib import Path
from typing import Union, Optional, Dict, Any

import torch
from PIL import Image
import numpy as np

from ..core.registry import ModelRegistry
from ..core.cache_manager import get_cache_manager
from ..core.utils import get_device, get_optimal_dtype, ensure_rgb
from ..core.base_upscaler import BaseUpscaler


class SuperscalePipeline:
    """Main pipeline for super-resolution.
    
    This class provides the primary interface for using super-resolution models
    in Superscale. It handles model loading, caching, and inference.
    """
    
    def __init__(
        self,
        model: str,
        device: Union[str, torch.device] = "auto",
        dtype: Optional[torch.dtype] = None,
        download: bool = True,
        cache: bool = True,
        **kwargs
    ):
        """Initialize pipeline.
        
        Args:
            model: Model name or alias
            device: Device to run on ("auto", "cpu", "cuda", etc.)
            dtype: Data type for model (None for auto-detection)
            download: Whether to auto-download model weights
            cache: Whether to use model caching
            **kwargs: Additional arguments passed to model constructor
        """
        self.model_name = model
        self.device = get_device(device)
        self.dtype = dtype or get_optimal_dtype(self.device)
        self.download = download
        self.use_cache = cache
        
        # Load or create model
        self._model = self._load_or_create_model(model, **kwargs)
    
    def _load_or_create_model(self, model_name: str, **kwargs) -> BaseUpscaler:
        """Load model from cache or create new instance."""
        # Check cache first
        if self.use_cache:
            cache_manager = get_cache_manager()
            cached_model = cache_manager.get(model_name)
            if cached_model is not None:
                print(f"Loaded {model_name} from cache")
                # Ensure it's on the right device
                cached_model.to(self.device)
                return cached_model
        
        # Create new model instance
        print(f"Loading {model_name}...")
        model_class = ModelRegistry.get(model_name)
        model_instance = model_class(device=self.device, dtype=self.dtype, **kwargs)
        
        # Load weights
        checkpoint_path = self._get_checkpoint_path(model_name)
        model_instance.load_weights(checkpoint_path)
        
        # Add to cache
        if self.use_cache:
            cache_manager.put(model_name, model_instance)
        
        return model_instance
    
    def _get_checkpoint_path(self, model_name: str) -> Path:
        """Get checkpoint path for model."""
        from ..core.model_manager import get_model_manager
        
        model_manager = get_model_manager()
        
        # Check if model is already downloaded
        checkpoint_path = model_manager.get_model_path(model_name)
        
        if checkpoint_path:
            return checkpoint_path
        
        # If download is enabled, download the model
        if self.download:
            try:
                checkpoint_path = model_manager.download_model(model_name)
                return checkpoint_path
            except Exception as e:
                raise RuntimeError(f"Failed to download {model_name}: {e}")
        
        # Model not found and download disabled
        raise FileNotFoundError(
            f"Model {model_name} not found. "
            f"Enable download=True or run: superscale model download {model_name}"
        )
    
    def __call__(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        scale: int = 4,
        **kwargs
    ) -> Image.Image:
        """Run super-resolution on an image.
        
        Args:
            image: Input image (PIL Image, numpy array, or path)
            scale: Upscaling factor
            **kwargs: Additional arguments passed to model
            
        Returns:
            Super-resolved PIL Image
        """
        # Handle string/path inputs
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        
        # Ensure RGB
        if isinstance(image, Image.Image):
            image = ensure_rgb(image)
        
        # Run upscaling
        return self._model.upscale(image, scale=scale, **kwargs)
    
    def to(self, device: Union[str, torch.device]) -> "SuperscalePipeline":
        """Move pipeline to device.
        
        Args:
            device: Target device
            
        Returns:
            Self for chaining
        """
        self.device = get_device(device)
        self._model.to(self.device)
        return self
    
    @property
    def model(self) -> BaseUpscaler:
        """Get the underlying model."""
        return self._model
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SuperscalePipeline(model='{self.model_name}', "
            f"device='{self.device}', dtype={self.dtype})"
        )


# Convenience functions
def load(model: str, **kwargs) -> SuperscalePipeline:
    """Load a super-resolution model.
    
    This is the primary entry point for loading models in Superscale.
    
    Args:
        model: Model name or alias (e.g., "Hermes", "hitsr-sir-x4")
        **kwargs: Additional arguments for pipeline initialization
        
    Returns:
        SuperscalePipeline instance
        
    Examples:
        >>> pipe = load("Hermes", device="cuda")
        >>> result = pipe("image.jpg", scale=4)
    """
    return SuperscalePipeline(model, **kwargs)


def up(
    image: Union[Image.Image, np.ndarray, str, Path],
    model: str = "Hermes",
    scale: int = 4,
    device: str = "auto",
    **kwargs
) -> Image.Image:
    """One-line super-resolution.
    
    This is a convenience function for quick super-resolution without
    explicitly creating a pipeline.
    
    Args:
        image: Input image
        model: Model to use
        scale: Upscaling factor
        device: Device to run on
        **kwargs: Additional arguments
        
    Returns:
        Super-resolved PIL Image
        
    Examples:
        >>> result = up("low_res.jpg", model="Athena", scale=4)
    """
    pipe = load(model, device=device)
    return pipe(image, scale=scale, **kwargs)