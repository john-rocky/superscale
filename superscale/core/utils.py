"""Common utilities for the Superscale library."""

import os
from pathlib import Path
from typing import Union, Optional, Tuple

import torch


def get_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Get the appropriate device for computation.
    
    Args:
        device: Device specification. Can be:
            - None or "auto": Automatically select GPU if available
            - "cpu": Force CPU
            - "cuda", "cuda:0", etc.: Specific GPU
            - torch.device: Pass through
            
    Returns:
        torch.device object
    """
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if isinstance(device, torch.device):
        return device
    
    return torch.device(device)


def get_optimal_dtype(device: torch.device, prefer_fp16: bool = True) -> torch.dtype:
    """Get optimal dtype for the given device.
    
    Args:
        device: Target device
        prefer_fp16: Whether to prefer fp16 when available
        
    Returns:
        Optimal dtype
    """
    if device.type == "cuda" and prefer_fp16:
        # Check if GPU supports fp16
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability(device)
            # fp16 is well supported on compute capability >= 7.0
            if capability[0] >= 7:
                return torch.float16
    
    return torch.float32


def ensure_rgb(image) -> any:
    """Ensure image is in RGB format.
    
    Args:
        image: PIL Image
        
    Returns:
        RGB PIL Image
    """
    if image.mode == "RGB":
        return image
    
    if image.mode == "RGBA":
        # Create white background
        from PIL import Image
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        return background
    
    return image.convert("RGB")


def calculate_output_size(
    input_size: Tuple[int, int],
    scale: int,
    align_to: Optional[int] = None
) -> Tuple[int, int]:
    """Calculate output size after upscaling.
    
    Args:
        input_size: Input (width, height)
        scale: Upscaling factor
        align_to: Align output size to be divisible by this value
        
    Returns:
        Output (width, height)
    """
    out_w = input_size[0] * scale
    out_h = input_size[1] * scale
    
    if align_to:
        out_w = (out_w // align_to) * align_to
        out_h = (out_h // align_to) * align_to
    
    return (out_w, out_h)


def get_cache_dir() -> Path:
    """Get the default cache directory for models.
    
    Returns:
        Path to cache directory
    """
    # Follow HuggingFace convention
    cache_dir = os.environ.get("SUPERSCALE_CACHE", None)
    
    if cache_dir:
        return Path(cache_dir)
    
    # Default to ~/.cache/superscale
    home = Path.home()
    return home / ".cache" / "superscale"


def format_bytes(num_bytes: int) -> str:
    """Format bytes as human-readable string.
    
    Args:
        num_bytes: Number of bytes
        
    Returns:
        Formatted string
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    
    return f"{num_bytes:.1f} PB"


def is_image_file(path: Union[str, Path]) -> bool:
    """Check if a file is an image.
    
    Args:
        path: File path
        
    Returns:
        True if the file is an image
    """
    image_extensions = {
        ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".tif", ".webp"
    }
    
    path = Path(path)
    return path.suffix.lower() in image_extensions


def validate_scale(scale: int, supported_scales: list) -> int:
    """Validate and adjust scale factor.
    
    Args:
        scale: Requested scale
        supported_scales: List of supported scales
        
    Returns:
        Valid scale factor
        
    Raises:
        ValueError: If scale cannot be adjusted to a valid value
    """
    if scale in supported_scales:
        return scale
    
    # Find closest supported scale
    closest = min(supported_scales, key=lambda x: abs(x - scale))
    
    if abs(closest - scale) > 1:
        raise ValueError(
            f"Scale {scale} not supported. Supported scales: {supported_scales}"
        )
    
    return closest


class Timer:
    """Simple timer context manager."""
    
    def __init__(self, name: str = "Operation", verbose: bool = True):
        """Initialize timer.
        
        Args:
            name: Name of the operation
            verbose: Whether to print timing
        """
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.elapsed = 0.0
    
    def __enter__(self):
        """Start timing."""
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        """Stop timing and print if verbose."""
        import time
        self.elapsed = time.time() - self.start_time
        
        if self.verbose:
            print(f"{self.name} took {self.elapsed:.3f}s")


def check_dependencies(model_type: str) -> bool:
    """Check if dependencies for a model type are installed.
    
    Args:
        model_type: Model type (hitsr, tsdsr, varsr)
        
    Returns:
        True if all dependencies are available
    """
    try:
        if model_type == "hitsr":
            import cv2
            import scipy
            return True
            
        elif model_type == "tsdsr":
            import diffusers
            import transformers
            import safetensors
            return True
            
        elif model_type == "varsr":
            import einops
            return True
            
    except ImportError:
        return False
    
    return False