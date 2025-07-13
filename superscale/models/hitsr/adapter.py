"""HiT-SR model adapter."""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple

import torch
import numpy as np
from PIL import Image

from ...core.base_upscaler import BaseUpscaler, ImageProcessor
from ...core.registry import register_model


class HiTSRAdapter(BaseUpscaler):
    """Adapter for HiT-SR (Hierarchical Transformer) models."""
    
    SUPPORTED_MODELS = {
        "hitsr-sir-x2": {"scale": 2, "variant": "HiT_SIR"},
        "hitsr-sir-x3": {"scale": 3, "variant": "HiT_SIR"},
        "hitsr-sir-x4": {"scale": 4, "variant": "HiT_SIR"},
        "hitsr-sng-x2": {"scale": 2, "variant": "HiT_SNG"},
        "hitsr-sng-x3": {"scale": 3, "variant": "HiT_SNG"},
        "hitsr-sng-x4": {"scale": 4, "variant": "HiT_SNG"},
        "hitsr-srf-x2": {"scale": 2, "variant": "HiT_SRF"},
        "hitsr-srf-x3": {"scale": 3, "variant": "HiT_SRF"},
        "hitsr-srf-x4": {"scale": 4, "variant": "HiT_SRF"},
    }
    
    def __init__(
        self,
        model_name: str,
        device: Union[str, torch.device] = "cpu",
        dtype: Optional[torch.dtype] = None,
    ):
        """Initialize HiT-SR adapter."""
        super().__init__(model_name, device, dtype)
        
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unknown HiT-SR model: {model_name}")
        
        self.model_config = self.SUPPORTED_MODELS[model_name]
        self.scale = self.model_config["scale"]
        self.variant = self.model_config["variant"]
        
        # Will be loaded by load_weights
        self._model_module = None
    
    def _load_hitsr_module(self):
        """Dynamically load HiT-SR modules."""
        # Check if we have the native backend version
        try:
            from ...backends.native.hitsr import create_model
            return create_model
        except ImportError:
            pass
        
        # Fallback to third_party if available (development mode)
        third_party_path = Path(__file__).parent.parent.parent.parent.parent / "third_party" / "HiT-SR"
        if third_party_path.exists():
            sys.path.insert(0, str(third_party_path))
            try:
                from basicsr.models import create_model
                return create_model
            finally:
                sys.path.remove(str(third_party_path))
        
        raise ImportError(
            "HiT-SR implementation not found. Please run sync_models.py or install in development mode."
        )
    
    def load_weights(self, checkpoint_path: Union[str, Path], **kwargs) -> None:
        """Load HiT-SR model weights."""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load the model creation function
        create_model = self._load_hitsr_module()
        
        # Create minimal options for HiT-SR
        opt = self._create_minimal_opt(checkpoint_path)
        
        # Create and load model
        self.model = create_model(opt)
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Set to eval mode
        self.model.eval()
        
        self._loaded = True
    
    def _create_minimal_opt(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Create minimal options dict for HiT-SR."""
        # This is a simplified version - actual implementation would need proper config
        opt = {
            "name": self.variant,
            "model_type": "HiTSRModel",
            "scale": self.scale,
            "num_gpu": 1 if self.device.type == "cuda" else 0,
            "dist": False,
            "network_g": {
                "type": self.variant.replace("HiT_", "HiT"),
                "upscale": self.scale,
                "in_chans": 3,
                "img_size": 64,
                "window_size": 16,
                "img_range": 1.0,
                "depths": [6, 6, 6, 6, 6, 6],
                "embed_dim": 180,
                "num_heads": [6, 6, 6, 6, 6, 6],
                "mlp_ratio": 2,
                "upsampler": "pixelshuffle",
                "resi_connection": "1conv",
            },
            "path": {
                "pretrain_network_g": str(checkpoint_path),
                "strict_load_g": True,
                "resume_state": None,
            },
        }
        
        return opt
    
    def preprocess(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        scale: int,
    ) -> Dict[str, Any]:
        """Preprocess input image."""
        if scale != self.scale:
            raise ValueError(f"Model supports {self.scale}x scaling, got {scale}x")
        
        # Convert to tensor
        tensor = ImageProcessor.image_to_tensor(image, dtype=self.dtype, normalize=True)
        
        # Move to device
        tensor = tensor.to(self.device)
        
        return {
            "lq": tensor,  # Low quality (input) image
            "scale": scale,
            "original_size": image.size if isinstance(image, Image.Image) else None,
        }
    
    def forward(self, preprocessed: Dict[str, Any], **kwargs) -> torch.Tensor:
        """Run HiT-SR inference."""
        lq = preprocessed["lq"]
        
        # HiT-SR expects the input in the model
        self.model.lq = lq
        
        # Run test
        self.model.test()
        
        # Get output
        output = self.model.output
        
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
        
        # Resize to exact expected size if needed
        if original_size:
            expected_size = (
                original_size[0] * self.scale,
                original_size[1] * self.scale
            )
            if image.size != expected_size:
                image = image.resize(expected_size, Image.LANCZOS)
        
        return image


# Register all HiT-SR variants
@register_model(
    "hitsr-sir-x2",
    aliases=["HiT-SIR-2x"],
    metadata={"description": "HiT-SIR 2x upscaling", "speed": 4, "quality": 4}
)
class HiTSIR2x(HiTSRAdapter):
    def __init__(self, **kwargs):
        super().__init__("hitsr-sir-x2", **kwargs)


@register_model(
    "hitsr-sir-x4",
    aliases=["HiT-SIR-4x", "Athena"],
    metadata={"description": "HiT-SIR 4x upscaling", "speed": 4, "quality": 4}
)
class HiTSIR4x(HiTSRAdapter):
    def __init__(self, **kwargs):
        super().__init__("hitsr-sir-x4", **kwargs)


@register_model(
    "hitsr-sng-x4",
    aliases=["HiT-SNG-4x", "Apollo"],
    metadata={"description": "HiT-SNG 4x upscaling", "speed": 4, "quality": 4}
)
class HiTSNG4x(HiTSRAdapter):
    def __init__(self, **kwargs):
        super().__init__("hitsr-sng-x4", **kwargs)


@register_model(
    "hitsr-srf-x4",
    aliases=["HiT-SRF-4x", "Artemis"],
    metadata={"description": "HiT-SRF 4x upscaling", "speed": 3, "quality": 5}
)
class HiTSRF4x(HiTSRAdapter):
    def __init__(self, **kwargs):
        super().__init__("hitsr-srf-x4", **kwargs)