"""
Superscale - Universal Super-Resolution Toolkit

A unified library for state-of-the-art image super-resolution models
with a diffusers-like API.
"""

__version__ = "0.1.0"

# Import and register models
def _register_models():
    """Register all available models."""
    # Dummy model (always available for testing)
    try:
        from .models.dummy import adapter
    except ImportError:
        pass
    
    # HiT-SR models
    try:
        from .models.hitsr import adapter
    except ImportError:
        pass  # HiT-SR not available
    
    # TSD-SR models
    try:
        from .models.tsdsr import adapter
    except ImportError:
        pass  # TSD-SR not available

_register_models()


# Lazy imports to avoid loading heavy dependencies on import
def load(model: str, **kwargs):
    """Load a super-resolution model.
    
    Args:
        model: Model name or alias (e.g., 'Hermes', 'hitsr-sir-x4')
        **kwargs: Additional arguments for model initialization
        
    Returns:
        SuperscalePipeline: Pipeline object for super-resolution
        
    Examples:
        >>> pipe = superscale.load("Hermes", device="cuda")
        >>> result = pipe("low_res.jpg", scale=4)
    """
    from .api.pipeline import SuperscalePipeline
    return SuperscalePipeline(model, **kwargs)


def up(image, model: str = "Hermes", scale: int = 4, **kwargs):
    """One-line super-resolution.
    
    Args:
        image: Input image (PIL.Image, numpy array, or path)
        model: Model to use (default: 'Hermes')
        scale: Upscaling factor (default: 4)
        **kwargs: Additional arguments
        
    Returns:
        PIL.Image: Super-resolved image
        
    Examples:
        >>> hr_image = superscale.up("low_res.jpg", scale=4)
    """
    pipe = load(model, **kwargs)
    return pipe(image, scale=scale)


def list_models():
    """List available models.
    
    Returns:
        list: List of available model names
    """
    from .core.registry import ModelRegistry
    return ModelRegistry.list_models()


def session(model: str, **kwargs):
    """Create a session for efficient batch processing.
    
    Args:
        model: Model name
        **kwargs: Additional arguments
        
    Returns:
        SuperscaleSession: Session context manager
        
    Examples:
        >>> with superscale.session("Hermes") as sess:
        ...     for img in images:
        ...         result = sess.upscale(img)
    """
    from .api.session import session as _session
    return _session(model, **kwargs)


# Aliases for different naming preferences
summon = load  # For the pixel-god metaphor
dismiss = lambda model=None: _dismiss(model)  # Clear model from cache


def _dismiss(model=None):
    """Clear model(s) from cache."""
    from .core.cache_manager import _cache_manager
    _cache_manager.clear(model)


# Model presets/aliases
MODEL_ALIASES = {
    # Mythological names (primary aliases)
    "Hermes": "tsdsr",        # Messenger god - fast communication of details
    "Athena": "hitsr-sir-x4", # Goddess of wisdom - intelligent SR
    "Apollo": "hitsr-sng-x4", # God of light - illuminating details  
    "Artemis": "hitsr-srf-x4", # Goddess of nature - natural results
    "Zeus": "varsr-d16",      # King of gods - powerful VAR model
    "Hera": "varsr-d20",      # Queen of gods - larger VAR model
    
    # Descriptive aliases
    "fast": "hitsr-sir-x4",
    "quality": "tsdsr",
    "balanced": "hitsr-sng-x4",
}


__all__ = [
    "__version__",
    "load",
    "up", 
    "summon",
    "dismiss",
    "list_models",
    "session",
    "MODEL_ALIASES",
]