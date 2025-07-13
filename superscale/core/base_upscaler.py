"""Base class for all super-resolution models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Optional, Dict, Any, Tuple

import numpy as np
import torch
from PIL import Image


class BaseUpscaler(ABC):
    """Abstract base class for all upscaler models.
    
    This class defines the common interface that all super-resolution models
    must implement to work with the Superscale framework.
    """
    
    def __init__(
        self,
        model_name: str,
        device: Union[str, torch.device] = "cpu",
        dtype: Optional[torch.dtype] = None,
    ):
        """Initialize base upscaler.
        
        Args:
            model_name: Name of the model
            device: Device to run the model on
            dtype: Data type for model weights
        """
        self.model_name = model_name
        self.device = self._parse_device(device)
        self.dtype = dtype or (torch.float16 if self.device.type == "cuda" else torch.float32)
        self.model = None
        self.config = {}
        self._loaded = False
    
    def _parse_device(self, device: Union[str, torch.device]) -> torch.device:
        """Parse device string or object."""
        if isinstance(device, torch.device):
            return device
        
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        return torch.device(device)
    
    @abstractmethod
    def load_weights(self, checkpoint_path: Union[str, Path], **kwargs) -> None:
        """Load model weights from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            **kwargs: Additional loading arguments
        """
        pass
    
    @abstractmethod
    def preprocess(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        scale: int,
    ) -> Dict[str, Any]:
        """Preprocess input image for model.
        
        Args:
            image: Input image
            scale: Upscaling factor
            
        Returns:
            Dictionary containing preprocessed tensors and metadata
        """
        pass
    
    @abstractmethod
    def forward(
        self,
        preprocessed: Dict[str, Any],
        **kwargs
    ) -> torch.Tensor:
        """Run model inference.
        
        Args:
            preprocessed: Preprocessed input from preprocess()
            **kwargs: Additional inference arguments
            
        Returns:
            Output tensor
        """
        pass
    
    @abstractmethod
    def postprocess(
        self,
        output: torch.Tensor,
        original_size: Tuple[int, int],
        **kwargs
    ) -> Image.Image:
        """Postprocess model output to PIL Image.
        
        Args:
            output: Model output tensor
            original_size: Original input image size (width, height)
            **kwargs: Additional postprocessing arguments
            
        Returns:
            Postprocessed PIL Image
        """
        pass
    
    def upscale(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        scale: int = 4,
        **kwargs
    ) -> Image.Image:
        """Main upscaling interface.
        
        Args:
            image: Input image (PIL Image, numpy array, or path)
            scale: Upscaling factor
            **kwargs: Additional arguments passed to forward()
            
        Returns:
            Super-resolved PIL Image
        """
        if not self._loaded:
            raise RuntimeError(f"Model {self.model_name} not loaded. Call load_weights() first.")
        
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Store original size
        original_size = image.size
        
        # Preprocess
        preprocessed = self.preprocess(image, scale)
        
        # Inference
        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.cuda.amp.autocast(enabled=self.dtype == torch.float16):
                    output = self.forward(preprocessed, **kwargs)
            else:
                output = self.forward(preprocessed, **kwargs)
        
        # Postprocess
        return self.postprocess(output, original_size)
    
    def to(self, device: Union[str, torch.device]) -> "BaseUpscaler":
        """Move model to device.
        
        Args:
            device: Target device
            
        Returns:
            Self for chaining
        """
        self.device = self._parse_device(device)
        if self.model is not None:
            self.model = self.model.to(self.device)
        return self
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(model_name='{self.model_name}', device='{self.device}')"


class ImageProcessor:
    """Utility class for common image processing operations."""
    
    @staticmethod
    def image_to_tensor(
        image: Union[Image.Image, np.ndarray],
        dtype: torch.dtype = torch.float32,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Convert image to tensor.
        
        Args:
            image: Input image
            dtype: Output tensor dtype
            normalize: Whether to normalize to [0, 1]
            
        Returns:
            Tensor of shape (1, C, H, W)
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to float
        if image.dtype == np.uint8:
            image = image.astype(np.float32)
            if normalize:
                image = image / 255.0
        
        # Add batch dimension and convert to CHW
        if image.ndim == 2:  # Grayscale
            image = image[np.newaxis, np.newaxis, ...]
        elif image.ndim == 3:  # RGB
            image = image.transpose(2, 0, 1)[np.newaxis, ...]
        
        return torch.from_numpy(image).to(dtype)
    
    @staticmethod
    def tensor_to_image(
        tensor: torch.Tensor,
        denormalize: bool = True,
        clamp: bool = True,
    ) -> Image.Image:
        """Convert tensor to PIL Image.
        
        Args:
            tensor: Input tensor of shape (1, C, H, W) or (C, H, W)
            denormalize: Whether to denormalize from [0, 1] to [0, 255]
            clamp: Whether to clamp values to valid range
            
        Returns:
            PIL Image
        """
        # Remove batch dimension if present
        if tensor.ndim == 4:
            tensor = tensor.squeeze(0)
        
        # Move to CPU and convert to numpy
        array = tensor.detach().cpu().numpy()
        
        # Convert CHW to HWC
        if array.shape[0] in [1, 3]:
            array = array.transpose(1, 2, 0)
        
        # Denormalize
        if denormalize and array.max() <= 1.0:
            array = array * 255.0
        
        # Clamp and convert to uint8
        if clamp:
            array = np.clip(array, 0, 255)
        
        array = array.astype(np.uint8)
        
        # Handle grayscale
        if array.shape[2] == 1:
            array = array.squeeze(2)
        
        return Image.fromarray(array)
    
    @staticmethod
    def resize_to_multiple(
        image: Image.Image,
        multiple: int = 8,
        mode: str = "bicubic"
    ) -> Tuple[Image.Image, Tuple[int, int]]:
        """Resize image to be divisible by multiple.
        
        Args:
            image: Input image
            multiple: The multiple to align to
            mode: Resampling mode
            
        Returns:
            Resized image and original size
        """
        w, h = image.size
        original_size = (w, h)
        
        # Calculate new dimensions
        new_w = (w // multiple) * multiple
        new_h = (h // multiple) * multiple
        
        if new_w != w or new_h != h:
            resample = getattr(Image, mode.upper(), Image.BICUBIC)
            image = image.resize((new_w, new_h), resample)
        
        return image, original_size