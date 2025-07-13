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
        """Load HiT-SR architecture directly from synced code."""
        try:
            # Import HiT-SR architecture directly
            from ...backends.native.hitsr.basicsr.archs.hit_sir_arch import HiT_SIR
            
            # Create a simple model factory function
            def create_hitsr_model(opt):
                model = HiT_SIR(
                    upscale=opt['network_g']['upscale'],
                    in_chans=opt['network_g']['in_chans'],
                    img_size=opt['network_g']['img_size'],
                    base_win_size=opt['network_g']['base_win_size'],
                    img_range=opt['network_g']['img_range'],
                    depths=opt['network_g']['depths'],
                    embed_dim=opt['network_g']['embed_dim'],
                    num_heads=opt['network_g']['num_heads'],
                    expansion_factor=opt['network_g']['expansion_factor'],
                    resi_connection=opt['network_g']['resi_connection'],
                    hier_win_ratios=opt['network_g']['hier_win_ratios'],
                    upsampler=opt['network_g']['upsampler']
                )
                
                # Load checkpoint
                import torch
                checkpoint = torch.load(opt['path']['pretrain_network_g'], map_location='cpu', weights_only=False)
                model.load_state_dict(checkpoint, strict=True)
                
                return model
            
            return create_hitsr_model
            
        except ImportError as e:
            # Fallback: try to load from third_party with comprehensive mocking
            return self._load_from_third_party()
    
    def _load_from_third_party(self):
        """Fallback method to load from third_party with mocking."""
        third_party_path = Path(__file__).parent.parent.parent.parent / "third_party" / "HiT-SR"
        
        if not third_party_path.exists():
            raise ImportError("HiT-SR implementation not found. Please run sync_models.py")
        
        sys.path.insert(0, str(third_party_path))
        try:
            # Comprehensive mocking for problematic imports  
            import torch
            
            class MockModule:
                def __init__(self, name="MockModule"):
                    self._name = name
                def __getattr__(self, name):
                    return MockModule(f"{self._name}.{name}")
                def __call__(self, *args, **kwargs):
                    return None
            
            # Mock all problematic modules
            mock_modules = [
                'torchvision.utils', 'torchvision.transforms.functional',
                'torchvision.io', 'torchvision.datasets', 'utils',
                'timm.models.fx_features', 'timm.models.layers'
            ]
            
            for module_name in mock_modules:
                sys.modules[module_name] = MockModule(module_name)
            
            if not hasattr(torch.library, 'register_fake'):
                torch.library.register_fake = lambda x: lambda f: f
            
            from basicsr.models import build_model
            return build_model
        except Exception as e:
            raise ImportError(f"Could not load HiT-SR: {e}")
        finally:
            if str(third_party_path) in sys.path:
                sys.path.remove(str(third_party_path))
    
    def load_weights(self, checkpoint_path: Union[str, Path], **kwargs) -> None:
        """Load HiT-SR model weights."""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Check and install dependencies
        self._check_dependencies()
        
        # Load the model creation function
        create_model = self._load_hitsr_module()
        
        try:
            # Create minimal options for HiT-SR
            opt = self._create_minimal_opt(checkpoint_path)
            
            # Create and load model
            self.model = create_model(opt)
            
            # Move to device and set eval mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            raise RuntimeError(f"Failed to load HiT-SR model: {e}")
        
        self._loaded = True
    
    def _check_dependencies(self):
        """Check and optionally install HiT-SR dependencies."""
        required_packages = {
            "timm": "timm>=0.6.13",
            "einops": "einops", 
            "cv2": "opencv-python",
            "scipy": "scipy"
        }
        
        missing_packages = []
        
        for import_name, package_name in required_packages.items():
            try:
                __import__(import_name)
            except ImportError:
                missing_packages.append(package_name)
        
        if missing_packages:
            error_msg = f"""
Missing required dependencies for HiT-SR: {', '.join(missing_packages)}

To install HiT-SR dependencies, run:
    pip install superscale[hitsr]

Or install manually:
    pip install {' '.join(missing_packages)}
"""
            raise ImportError(error_msg)
    
    def _create_minimal_opt(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Create minimal options dict for HiT-SR."""
        # This is a simplified version - actual implementation would need proper config
        opt = {
            "name": self.variant,
            "model_type": "HITModel", 
            "scale": self.scale,
            "num_gpu": 1 if self.device.type == "cuda" else 0,
            "dist": False,
            "is_train": False,
            "path": {
                "pretrain_network_g": str(checkpoint_path),
                "strict_load_g": True,
            },
            "val": {
                "save_img": False,
                "use_chop": False,
            },
            "network_g": {
                "type": self.variant,
                "upscale": self.scale,
                "in_chans": 3,
                "img_size": 64,
                "base_win_size": [8, 8],
                "img_range": 1.0,
                "depths": [6, 6, 6, 6],
                "embed_dim": 60,
                "num_heads": [6, 6, 6, 6],
                "expansion_factor": 2,
                "resi_connection": "1conv",
                "hier_win_ratios": [0.5, 1, 2, 4, 6, 8],
                "upsampler": "pixelshuffledirect",
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
        
        with torch.no_grad():
            # Direct inference with the PyTorch model
            output = self.model(lq)
        
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