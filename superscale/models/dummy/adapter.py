"""Dummy upscaler for testing the framework."""

from pathlib import Path
from typing import Dict, Any, Union, Tuple

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from ...core.base_upscaler import BaseUpscaler, ImageProcessor
from ...core.registry import register_model


class SimpleUpscaleNet(nn.Module):
    """Simple neural network that does bilinear upscaling."""
    
    def __init__(self, scale: int = 4):
        super().__init__()
        self.scale = scale
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Simple bilinear upscaling."""
        return torch.nn.functional.interpolate(
            x, 
            scale_factor=self.scale, 
            mode='bilinear', 
            align_corners=False
        )


@register_model(
    "dummy",
    aliases=["test-model", "dummy-x4"],
    metadata={
        "description": "Dummy model for testing (bilinear upscaling)",
        "speed": 5,
        "quality": 1
    }
)
class DummyUpscaler(BaseUpscaler):
    """Dummy upscaler that uses simple bilinear interpolation."""
    
    def __init__(
        self,
        model_name: str = "dummy",
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = None,
    ):
        """Initialize dummy upscaler."""
        super().__init__(model_name, device, dtype)
        self.supported_scales = [2, 3, 4, 8]
    
    def load_weights(self, checkpoint_path: Union[str, Path] = None, **kwargs) -> None:
        """Load dummy model (no real weights needed)."""
        # For dummy model, we don't need real weights
        self.model = SimpleUpscaleNet(scale=4)
        self.model = self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        print(f"Dummy model loaded on {self.device}")
    
    def preprocess(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        scale: int,
    ) -> Dict[str, Any]:
        """Preprocess input image."""
        # Validate scale
        if scale not in self.supported_scales:
            raise ValueError(
                f"Scale {scale} not supported. "
                f"Supported scales: {self.supported_scales}"
            )
        
        # Convert to tensor
        tensor = ImageProcessor.image_to_tensor(image, dtype=self.dtype, normalize=True)
        
        # Move to device
        tensor = tensor.to(self.device)
        
        return {
            "input": tensor,
            "scale": scale,
            "original_size": image.size if isinstance(image, Image.Image) else None,
        }
    
    def forward(self, preprocessed: Dict[str, Any], **kwargs) -> torch.Tensor:
        """Run dummy inference (bilinear upscaling)."""
        input_tensor = preprocessed["input"]
        scale = preprocessed["scale"]
        
        # Update model scale if needed
        if self.model.scale != scale:
            self.model.scale = scale
        
        # Simple upscaling
        with torch.no_grad():
            output = self.model(input_tensor)
        
        return output
    
    def postprocess(
        self,
        output: torch.Tensor,
        original_size: Tuple[int, int],
        **kwargs
    ) -> Image.Image:
        """Postprocess model output."""
        # Convert to image
        image = ImageProcessor.tensor_to_image(output, denormalize=True, clamp=True)
        
        # Ensure exact output size
        if original_size:
            expected_size = (
                original_size[0] * self.model.scale,
                original_size[1] * self.model.scale
            )
            if image.size != expected_size:
                image = image.resize(expected_size, Image.LANCZOS)
        
        return image


# Also register with "Hermes" alias for testing since TSD-SR is not implemented yet
@register_model(
    "dummy-hermes",
    aliases=["Hermes"],
    metadata={
        "description": "Dummy model aliased as Hermes for testing",
        "speed": 5,
        "quality": 1
    }
)
class DummyHermes(DummyUpscaler):
    """Dummy model aliased as Hermes for testing."""
    pass