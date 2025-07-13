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
        # Go up 5 levels: hitsr/adapter.py -> hitsr -> models -> superscale -> project_root
        third_party_path = Path(__file__).parent.parent.parent.parent / "third_party" / "HiT-SR"
        
        if third_party_path.exists():
            sys.path.insert(0, str(third_party_path))
            try:
                # Comprehensive monkey patching for compatibility
                import torch
                
                # Mock problematic modules to avoid import errors
                class MockModule:
                    def __init__(self, name="MockModule"):
                        self._name = name
                    def __getattr__(self, name):
                        return MockModule(f"{self._name}.{name}")
                    def __call__(self, *args, **kwargs):
                        return None
                    def __repr__(self):
                        return f"<MockModule: {self._name}>"
                
                # Mock comprehensive list of problematic imports
                mock_modules = [
                    'torchvision.utils',
                    'torchvision.transforms.functional', 
                    'torchvision.io',
                    'torchvision.io.image',
                    'torchvision.datasets',
                    'torchvision.datasets._optical_flow',
                    'torchvision.ops.misc',
                    'timm.models.fx_features',
                    'timm.models.layers',
                    'utils'  # Prevent utils conflict
                ]
                
                for module_name in mock_modules:
                    sys.modules[module_name] = MockModule(module_name)
                
                # Add torch.library compatibility
                if not hasattr(torch.library, 'register_fake'):
                    torch.library.register_fake = lambda x: lambda f: f
                
                from basicsr.models import build_model
                return build_model
            except Exception as e:
                print(f"Warning: Failed to load HiT-SR from third_party: {e}")
                raise ImportError("Could not load HiT-SR modules")
            finally:
                if str(third_party_path) in sys.path:
                    sys.path.remove(str(third_party_path))
        
        raise ImportError(
            "HiT-SR implementation not found. Please run sync_models.py or install in development mode."
        )
    
    def load_weights(self, checkpoint_path: Union[str, Path], **kwargs) -> None:
        """Load HiT-SR model weights."""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Check and install dependencies
        self._check_dependencies()
        
        # Load the model creation function
        create_model = self._load_hitsr_module()
        
        # Patch torch.load for PyTorch 2.6 compatibility
        import torch
        original_load = torch.load
        def patched_load(*args, **kwargs):
            kwargs.setdefault('weights_only', False)
            return original_load(*args, **kwargs)
        torch.load = patched_load
        
        try:
            # Create minimal options for HiT-SR
            opt = self._create_minimal_opt(checkpoint_path)
            
            # Create and load model
            self.model = create_model(opt)
        finally:
            # Restore original torch.load
            torch.load = original_load
        
        # Move to device
        self.model.net_g = self.model.net_g.to(self.device)
        
        # Set to eval mode
        self.model.net_g.eval()
        
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