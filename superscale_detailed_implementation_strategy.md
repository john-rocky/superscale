# Superscale Detailed Implementation Strategy

## Executive Summary

This document outlines a comprehensive step-by-step implementation strategy for creating a unified super-resolution library that provides a diffusers-like API for multiple state-of-the-art upscaler models. The strategy addresses dependency conflicts, architectural differences, and provides a phased implementation approach.

## 1. Core Challenges & Solutions

### 1.1 Dependency Conflicts

**Problem**: The three models have conflicting dependencies:
- PyTorch: 2.2.1 (VARSR) vs 2.2.2 (TSD-SR) vs flexible (HiT-SR)
- Diffusers: 0.29.1 (TSD-SR) vs 0.32.2 (VARSR)
- Transformers: 4.49.0 (TSD-SR) vs 4.37.2 (VARSR)
- Pillow: 10.3.0 (TSD-SR) vs 11.1.0 (VARSR)

**Solution**: Multi-layered approach:
1. **Core dependencies**: Use compatible ranges (torch>=2.2,<2.4)
2. **Optional extras**: Model-specific dependencies as extras
3. **Dynamic loading**: Lazy import with fallback mechanisms
4. **Environment isolation**: Support for separate virtual environments via subprocess

### 1.2 Architecture Differences

**Problem**: Each model has different:
- Loading mechanisms (YAML, HuggingFace, custom checkpoints)
- Preprocessing requirements
- Inference patterns
- Output formats

**Solution**: Adapter pattern with common interface:
```python
class BaseUpscaler(ABC):
    @abstractmethod
    def load_model(self, model_path, device):
        pass
    
    @abstractmethod
    def preprocess(self, image, scale):
        pass
    
    @abstractmethod
    def upscale(self, image, scale, **kwargs):
        pass
    
    @abstractmethod
    def postprocess(self, output):
        pass
```

## 2. Implementation Phases

### Phase 1: Foundation (Week 1)

**1.1 Project Structure**
```
superscale/
├── superscale/
│   ├── __init__.py
│   ├── core/
│   │   ├── base_upscaler.py      # Abstract base class
│   │   ├── registry.py           # Model registry
│   │   ├── cache_manager.py      # Model caching (Summoner)
│   │   └── utils.py              # Common utilities
│   ├── models/
│   │   ├── __init__.py
│   │   ├── hitsr/
│   │   │   ├── __init__.py
│   │   │   ├── adapter.py        # HiT-SR adapter
│   │   │   └── utils.py
│   │   ├── tsdsr/
│   │   │   ├── __init__.py
│   │   │   ├── adapter.py        # TSD-SR adapter
│   │   │   └── utils.py
│   │   └── varsr/
│   │       ├── __init__.py
│   │       ├── adapter.py        # VARSR adapter
│   │       └── utils.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── pipeline.py           # Main API
│   │   └── presets.py            # Model presets
│   ├── cli/
│   │   ├── __init__.py
│   │   └── main.py               # CLI interface
│   └── gui/
│       ├── __init__.py
│       └── app.py                # Gradio interface
├── tests/
├── examples/
├── scripts/
│   ├── setup_models.py           # Model download/setup
│   └── convert_weights.py        # Weight conversion
├── pyproject.toml
└── README.md
```

**1.2 Core Dependencies (pyproject.toml)**
```toml
[project]
name = "superscale"
version = "0.1.0"
dependencies = [
    "torch>=2.2,<2.4",
    "pillow>=10.0",
    "numpy>=1.24,<2.0",
    "opencv-python>=4.8",
    "tqdm>=4.65",
    "pyyaml>=6.0",
    "huggingface-hub>=0.20",
]

[project.optional-dependencies]
hitsr = [
    "timm>=0.9,<1.0",
    "einops>=0.7",
    "scikit-image>=0.21",
]
tsdsr = [
    "diffusers>=0.29,<0.33",
    "transformers>=4.40,<4.50",
    "peft>=0.10",
    "accelerate>=0.25",
]
varsr = [
    "diffusers>=0.32,<0.34",
    "transformers>=4.37,<4.40",
    "xformers>=0.0.24; platform_system != 'Windows'",
]
all = ["superscale[hitsr,tsdsr,varsr]"]
dev = [
    "pytest>=7.0",
    "ruff>=0.1.0",
    "black>=23.0",
]
```

**1.3 Base Classes Implementation**

```python
# core/base_upscaler.py
from abc import ABC, abstractmethod
from typing import Union, Optional, Dict, Any
import numpy as np
from PIL import Image
import torch

class BaseUpscaler(ABC):
    """Abstract base class for all upscaler models."""
    
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.config = {}
    
    @abstractmethod
    def load_weights(self, checkpoint_path: str, **kwargs) -> None:
        """Load model weights from checkpoint."""
        pass
    
    @abstractmethod
    def preprocess(
        self, 
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        scale: int
    ) -> Dict[str, Any]:
        """Preprocess input image."""
        pass
    
    @abstractmethod
    def forward(
        self,
        preprocessed: Dict[str, Any],
        **kwargs
    ) -> torch.Tensor:
        """Run model inference."""
        pass
    
    @abstractmethod
    def postprocess(
        self,
        output: torch.Tensor,
        original_size: tuple
    ) -> Image.Image:
        """Postprocess model output."""
        pass
    
    def upscale(
        self,
        image: Union[Image.Image, np.ndarray, str],
        scale: int = 4,
        **kwargs
    ) -> Image.Image:
        """Main upscaling interface."""
        # Convert to PIL Image if needed
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Store original size
        original_size = image.size
        
        # Preprocess
        preprocessed = self.preprocess(image, scale)
        
        # Inference
        with torch.no_grad():
            output = self.forward(preprocessed, **kwargs)
        
        # Postprocess
        return self.postprocess(output, original_size)
```

### Phase 2: Model Adapters (Week 2-3)

**2.1 HiT-SR Adapter**

```python
# models/hitsr/adapter.py
import sys
import importlib.util
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from ...core.base_upscaler import BaseUpscaler

class HiTSRAdapter(BaseUpscaler):
    """Adapter for HiT-SR models."""
    
    SUPPORTED_MODELS = {
        "HiT-SIR": "HiT_SIR.pth",
        "HiT-SNG": "HiT_SNG.pth", 
        "HiT-SRF": "HiT_SRF.pth",
    }
    
    def __init__(self, model_type: str = "HiT-SIR", device: str = "cpu"):
        super().__init__(f"hitsr-{model_type.lower()}", device)
        self.model_type = model_type
        self._load_basicsr_modules()
    
    def _load_basicsr_modules(self):
        """Dynamically load BasicSR modules."""
        # Add HiT-SR path to sys.path temporarily
        hitsr_path = Path(__file__).parent.parent.parent.parent / "HiT-SR"
        sys.path.insert(0, str(hitsr_path))
        
        try:
            # Import required modules
            from basicsr.models import create_model
            from basicsr.utils import set_random_seed
            from basicsr.utils.options import parse_options
            
            self._create_model = create_model
            self._parse_options = parse_options
        finally:
            # Remove from path to avoid conflicts
            sys.path.remove(str(hitsr_path))
    
    def load_weights(self, checkpoint_path: str, config_path: Optional[str] = None):
        """Load HiT-SR model weights."""
        # Load configuration
        if config_path:
            opt = self._parse_options(config_path, is_train=False)
        else:
            # Use default config
            opt = self._get_default_config()
        
        # Create model
        self.model = self._create_model(opt)
        
        # Load checkpoint
        self.model.load_network(checkpoint_path)
        self.model.to(self.device)
    
    def preprocess(self, image, scale):
        # HiT-SR specific preprocessing
        # Convert to tensor, normalize, etc.
        pass
    
    def forward(self, preprocessed, **kwargs):
        # Run HiT-SR inference
        pass
    
    def postprocess(self, output, original_size):
        # Convert output to PIL Image
        pass
```

**2.2 TSD-SR Adapter**

```python
# models/tsdsr/adapter.py
from typing import Optional, Dict, Any
import torch
from ...core.base_upscaler import BaseUpscaler

class TSDSRAdapter(BaseUpscaler):
    """Adapter for TSD-SR (Stable Diffusion based) models."""
    
    def __init__(self, device: str = "cpu"):
        super().__init__("tsdsr", device)
        self._transformer = None
        self._vae = None
        self._scheduler = None
        
    def _check_dependencies(self):
        """Check if required dependencies are available."""
        try:
            import diffusers
            import transformers
            import peft
            return True
        except ImportError as e:
            raise ImportError(
                "TSD-SR requires additional dependencies. "
                "Install with: pip install superscale[tsdsr]"
            ) from e
    
    def load_weights(self, checkpoint_path: str, **kwargs):
        """Load TSD-SR model weights including LoRA adapters."""
        self._check_dependencies()
        
        from diffusers import (
            SD3Transformer2DModel,
            AutoencoderKL,
            FlowMatchEulerDiscreteScheduler
        )
        from peft import LoraConfig, get_peft_model
        
        # Load base models
        self._transformer = SD3Transformer2DModel.from_pretrained(
            "stabilityai/stable-diffusion-3-medium",
            subfolder="transformer",
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32
        )
        
        self._vae = AutoencoderKL.from_pretrained(
            "stabilityai/stable-diffusion-3-medium",
            subfolder="vae",
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32
        )
        
        # Load LoRA weights
        # Implementation details...
```

**2.3 VARSR Adapter**

```python
# models/varsr/adapter.py
class VARSRAdapter(BaseUpscaler):
    """Adapter for VARSR (Visual Autoregressive) models."""
    
    def __init__(self, model_size: str = "d16", device: str = "cpu"):
        super().__init__(f"varsr-{model_size}", device)
        self.model_size = model_size
        self._var_model = None
        self._vae_model = None
```

### Phase 2.5: Model Download Management (Week 3)

**2.5.1 Model Registry with Download Support**

```python
# core/model_manager.py
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import requests
from tqdm import tqdm
import torch
from huggingface_hub import hf_hub_download, snapshot_download

class ModelManager:
    """Manages model downloads and local storage."""
    
    # Model configurations with download info
    MODEL_CONFIGS = {
        # HiT-SR Models (from OneDrive and HuggingFace)
        "hitsr-sir-x2": {
            "url": "https://1drv.ms/u/c/de821e161e64ce08/EQQi-zFl1VpDhxuhXdGcfN0Bc8F_p3yxqIJPQdXVzVdYhg?e=dwdjgI",
            "filename": "HiT-SIR-2x.pth",
            "size": "~50MB",
            "type": "onedrive",
            "huggingface": "XiangZ/hit-sr",
        },
        "hitsr-sir-x4": {
            "url": "https://1drv.ms/u/c/de821e161e64ce08/EbWwGQo4c7xGh88aKZ9BNc8B8yg5E-c8Y_WKPGAqQkD_xA?e=4Imdxo",
            "filename": "HiT-SIR-4x.pth",
            "size": "~50MB",
            "type": "onedrive",
            "huggingface": "XiangZ/hit-sr",
        },
        "hitsr-sng-x4": {
            "url": "https://1drv.ms/u/c/de821e161e64ce08/EQfQfdbXo7lCvvBa_sTxMEQBzEb2n8NQbB8vhQZJA3LmUg?e=KoWEUd",
            "filename": "HiT-SNG-4x.pth",
            "size": "~55MB",
            "type": "onedrive",
            "huggingface": "XiangZ/hit-sr",
        },
        "hitsr-srf-x4": {
            "url": "https://1drv.ms/u/c/de821e161e64ce08/EVzqJK3Jh6pMrQaXdQ-WBYABRqKqxiNFatdvh5x0d9fLfg?e=C3voE6",
            "filename": "HiT-SRF-4x.pth",
            "size": "~180MB",
            "type": "onedrive",
            "huggingface": "XiangZ/hit-sr",
        },
        
        # TSD-SR Models (from Google Drive/OneDrive)
        "tsdsr": {
            "base_model": "stabilityai/stable-diffusion-3-medium-diffusers",
            "lora_weights": {
                "google_drive_id": "1XJY9Qxhz0mqjTtgDXr07oFy9eJr8jphI",
                "onedrive_url": "https://1drv.ms/f/c/d75249b59f444489/EsQQ2LLXp7pHsYMBVubgcsYBvEQXMmcNXGnz695odCGByQ",
            },
            "teacher_model": {
                "google_drive_id": "1do8pfdm_oNUhJKxTlC_x7LqY7NlE0-Q7",
            },
            "prompt_embeddings": {
                "google_drive_id": "1_kSod1CCq_xwdwDnLYUhr7iaT70eFPBD",
            },
            "size": "~5.4GB (base) + ~100MB (LoRA)",
            "type": "composite",  # Base model from HF + LoRA from Drive
        },
        
        # VARSR Models (from HuggingFace)
        "varsr-d16": {
            "repo_id": "qyp2000/VARSR",
            "filename": "varsr_d16.pth",
            "size": "~310MB",
            "type": "huggingface",
        },
        "varsr-d20": {
            "repo_id": "qyp2000/VARSR",
            "filename": "varsr_d20.pth",
            "size": "~460MB",
            "type": "huggingface",
        },
        "varsr-vqvae": {
            "repo_id": "qyp2000/VARSR",
            "filename": "vqvae.pth",
            "size": "~150MB",
            "type": "huggingface",
        },
    }
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize model manager."""
        self.cache_dir = cache_dir or self._get_default_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_file = self.cache_dir / "metadata.json"
        self._load_metadata()
    
    def _get_default_cache_dir(self) -> Path:
        """Get default cache directory."""
        # Follow HuggingFace Hub convention
        return Path.home() / ".cache" / "superscale"
    
    def _load_metadata(self):
        """Load download metadata."""
        if self._metadata_file.exists():
            with open(self._metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
    
    def _save_metadata(self):
        """Save download metadata."""
        with open(self._metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def get_model_path(self, model_name: str) -> Optional[Path]:
        """Get local path for a model if it exists."""
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}")
        
        config = self.MODEL_CONFIGS[model_name]
        
        if config["type"] == "direct":
            model_path = self.cache_dir / model_name / config["filename"]
            if model_path.exists() and self._verify_checksum(model_path, config.get("sha256")):
                return model_path
        
        elif config["type"] == "huggingface":
            model_dir = self.cache_dir / model_name
            # Check if all required files exist
            if model_dir.exists():
                all_files_exist = all(
                    (model_dir / file).exists() 
                    for file in config.get("files", [])
                )
                if all_files_exist:
                    return model_dir
        
        return None
    
    def download_model(
        self, 
        model_name: str, 
        force: bool = False,
        progress_callback: Optional[callable] = None
    ) -> Path:
        """Download a model if not already cached."""
        existing_path = self.get_model_path(model_name)
        if existing_path and not force:
            return existing_path
        
        config = self.MODEL_CONFIGS[model_name]
        
        if config["type"] == "direct":
            return self._download_direct(model_name, config, progress_callback)
        elif config["type"] == "onedrive":
            return self._download_onedrive(model_name, config, progress_callback)
        elif config["type"] == "google_drive":
            return self._download_google_drive(model_name, config, progress_callback)
        elif config["type"] == "huggingface":
            return self._download_huggingface(model_name, config, progress_callback)
        elif config["type"] == "composite":
            return self._download_composite(model_name, config, progress_callback)
        else:
            raise ValueError(f"Unknown download type: {config['type']}")
    
    def _download_direct(
        self, 
        model_name: str, 
        config: Dict, 
        progress_callback: Optional[callable] = None
    ) -> Path:
        """Download model directly from URL."""
        model_dir = self.cache_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = model_dir / config["filename"]
        temp_path = output_path.with_suffix(".tmp")
        
        try:
            # Download with progress bar
            response = requests.get(config["url"], stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(temp_path, 'wb') as f:
                with tqdm(
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    desc=f"Downloading {model_name}"
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
                        if progress_callback:
                            progress_callback(pbar.n / total_size)
            
            # Verify checksum
            if not self._verify_checksum(temp_path, config.get("sha256")):
                raise ValueError("Checksum verification failed")
            
            # Move to final location
            temp_path.rename(output_path)
            
            # Update metadata
            self.metadata[model_name] = {
                "path": str(output_path),
                "downloaded_at": str(Path.ctime(output_path)),
                "size": output_path.stat().st_size,
            }
            self._save_metadata()
            
            return output_path
            
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise e
    
    def _download_onedrive(
        self,
        model_name: str,
        config: Dict,
        progress_callback: Optional[callable] = None
    ) -> Path:
        """Download from OneDrive with proper handling."""
        # OneDrive URLs need special handling
        # Convert share URL to direct download URL
        url = config["url"]
        if "1drv.ms" in url:
            # OneDrive direct download conversion
            url = url.replace("/u/", "/download/")
        
        return self._download_direct(model_name, {**config, "url": url}, progress_callback)
    
    def _download_google_drive(
        self,
        model_name: str,
        config: Dict,
        progress_callback: Optional[callable] = None
    ) -> Path:
        """Download from Google Drive using existing utilities."""
        model_dir = self.cache_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to use existing download utilities from TSD-SR/VARSR
        try:
            # Import download utilities if available
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / "TSD-SR"))
            from download_util import download_file_from_google_drive
            
            output_path = model_dir / config["filename"]
            download_file_from_google_drive(
                config["google_drive_id"],
                str(output_path)
            )
            return output_path
        except ImportError:
            # Fallback to gdown
            try:
                import gdown
                url = f"https://drive.google.com/uc?id={config['google_drive_id']}"
                output_path = model_dir / config["filename"]
                gdown.download(url, str(output_path), quiet=False)
                return output_path
            except ImportError:
                raise ImportError(
                    "Please install gdown to download from Google Drive: "
                    "pip install gdown"
                )
        finally:
            if str(Path(__file__).parent.parent.parent / "TSD-SR") in sys.path:
                sys.path.remove(str(Path(__file__).parent.parent.parent / "TSD-SR"))
    
    def _download_composite(
        self,
        model_name: str,
        config: Dict,
        progress_callback: Optional[callable] = None
    ) -> Path:
        """Download composite models (e.g., TSD-SR with base model + LoRA)."""
        model_dir = self.cache_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Download base model from HuggingFace
        if "base_model" in config:
            print(f"Downloading base model: {config['base_model']}")
            base_dir = model_dir / "base_model"
            snapshot_download(
                repo_id=config["base_model"],
                cache_dir=str(base_dir),
                local_dir=str(base_dir),
            )
        
        # Download LoRA weights
        if "lora_weights" in config:
            print("Downloading LoRA weights...")
            lora_dir = model_dir / "lora_weights"
            lora_dir.mkdir(exist_ok=True)
            
            # Try Google Drive first, then OneDrive
            if "google_drive_id" in config["lora_weights"]:
                self._download_google_drive(
                    f"{model_name}_lora",
                    {
                        "google_drive_id": config["lora_weights"]["google_drive_id"],
                        "filename": "lora_weights.safetensors"
                    }
                )
        
        # Download additional components
        for component in ["teacher_model", "prompt_embeddings"]:
            if component in config:
                print(f"Downloading {component}...")
                component_dir = model_dir / component
                component_dir.mkdir(exist_ok=True)
                # Download logic...
        
        return model_dir
    
    def _download_huggingface(
        self, 
        model_name: str, 
        config: Dict,
        progress_callback: Optional[callable] = None
    ) -> Path:
        """Download model from HuggingFace Hub."""
        model_dir = self.cache_dir / model_name
        
        try:
            if "files" in config:
                # Download specific files
                for file in config["files"]:
                    hf_hub_download(
                        repo_id=config["repo_id"],
                        filename=file,
                        revision=config.get("revision", "main"),
                        cache_dir=str(model_dir),
                        local_dir=str(model_dir),
                    )
            else:
                # Download entire repository
                snapshot_download(
                    repo_id=config["repo_id"],
                    revision=config.get("revision", "main"),
                    cache_dir=str(model_dir),
                    local_dir=str(model_dir),
                )
            
            # Update metadata
            self.metadata[model_name] = {
                "path": str(model_dir),
                "downloaded_at": str(Path.ctime(model_dir)),
                "type": "huggingface",
            }
            self._save_metadata()
            
            return model_dir
            
        except Exception as e:
            # Clean up on failure
            if model_dir.exists():
                import shutil
                shutil.rmtree(model_dir)
            raise e
    
    def _verify_checksum(self, file_path: Path, expected_sha256: Optional[str]) -> bool:
        """Verify file checksum."""
        if not expected_sha256:
            return True  # No checksum to verify
        
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest() == expected_sha256
    
    def list_downloaded_models(self) -> Dict[str, Dict]:
        """List all downloaded models."""
        downloaded = {}
        for model_name in self.MODEL_CONFIGS:
            path = self.get_model_path(model_name)
            if path:
                downloaded[model_name] = {
                    "path": str(path),
                    "info": self.metadata.get(model_name, {})
                }
        return downloaded
    
    def get_download_size(self, model_name: str) -> str:
        """Get download size for a model."""
        if model_name in self.MODEL_CONFIGS:
            return self.MODEL_CONFIGS[model_name].get("size", "Unknown")
        return "Unknown"
    
    def clear_cache(self, model_name: Optional[str] = None):
        """Clear cached models."""
        if model_name:
            model_path = self.get_model_path(model_name)
            if model_path:
                if model_path.is_dir():
                    import shutil
                    shutil.rmtree(model_path)
                else:
                    model_path.unlink()
                
                if model_name in self.metadata:
                    del self.metadata[model_name]
                    self._save_metadata()
        else:
            # Clear all cache
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.metadata = {}
            self._save_metadata()

# Global instance
_model_manager = ModelManager()
```

**2.5.2 Integration with Pipeline**

```python
# api/pipeline.py (updated)
from ..core.model_manager import _model_manager

class SuperscalePipeline:
    """Main pipeline for super-resolution."""
    
    def __init__(
        self,
        model: str,
        device: str = "auto",
        dtype: Optional[torch.dtype] = None,
        download: bool = True,  # Auto-download if not found
        **kwargs
    ):
        self.model_name = model
        self.device = self._get_device(device)
        self.dtype = dtype or (torch.float16 if self.device != "cpu" else torch.float32)
        
        # Download model if needed
        if download:
            self._ensure_model_downloaded(model)
        
        # Get model from cache or create new
        self._model = self._load_or_create_model(model, **kwargs)
    
    def _ensure_model_downloaded(self, model_name: str):
        """Ensure model is downloaded."""
        if not _model_manager.get_model_path(model_name):
            print(f"Downloading {model_name} ({_model_manager.get_download_size(model_name)})...")
            _model_manager.download_model(model_name)
    
    def _get_checkpoint_path(self, model_name: str) -> Path:
        """Get local checkpoint path."""
        path = _model_manager.get_model_path(model_name)
        if not path:
            raise FileNotFoundError(
                f"Model {model_name} not found. "
                f"Run 'superscale download {model_name}' to download it."
            )
        return path
```

**2.5.3 CLI Commands for Model Management**

```python
# cli/main.py (updated)
@cli.group()
def model():
    """Model management commands."""
    pass

@model.command()
@click.argument('model_name', required=False)
@click.option('--all', is_flag=True, help='Download all models')
@click.option('--force', is_flag=True, help='Force re-download')
def download(model_name, all, force):
    """Download model weights."""
    from ..core.model_manager import _model_manager
    
    if all:
        models = ModelRegistry.list_models()
        for model in models:
            click.echo(f"Downloading {model}...")
            try:
                _model_manager.download_model(model, force=force)
                click.echo(f"✓ {model} downloaded successfully")
            except Exception as e:
                click.echo(f"✗ Failed to download {model}: {e}")
    elif model_name:
        try:
            path = _model_manager.download_model(model_name, force=force)
            click.echo(f"Downloaded to: {path}")
        except Exception as e:
            click.echo(f"Error: {e}")
    else:
        click.echo("Please specify a model name or use --all")

@model.command()
def list():
    """List downloaded models."""
    from ..core.model_manager import _model_manager
    
    downloaded = _model_manager.list_downloaded_models()
    
    if not downloaded:
        click.echo("No models downloaded yet.")
        return
    
    click.echo("Downloaded models:")
    for model_name, info in downloaded.items():
        click.echo(f"  - {model_name}: {info['path']}")

@model.command()
@click.argument('model_name', required=False)
@click.option('--all', is_flag=True, help='Clear all models')
def clear(model_name, all):
    """Clear downloaded models."""
    from ..core.model_manager import _model_manager
    
    if all:
        if click.confirm("Clear all downloaded models?"):
            _model_manager.clear_cache()
            click.echo("All models cleared.")
    elif model_name:
        _model_manager.clear_cache(model_name)
        click.echo(f"Cleared {model_name}")
    else:
        click.echo("Please specify a model name or use --all")

@model.command()
def info():
    """Show model information."""
    from ..core.model_manager import _model_manager
    
    click.echo("Available models:")
    for model_name, config in _model_manager.MODEL_CONFIGS.items():
        downloaded = "✓" if _model_manager.get_model_path(model_name) else "✗"
        size = config.get("size", "Unknown")
        click.echo(f"  [{downloaded}] {model_name} ({size})")
```

**2.5.4 Automatic Download with Progress in GUI**

```python
# gui/components/download_progress.py
import gradio as gr
from typing import Optional
from ..core.model_manager import _model_manager

class DownloadProgress:
    """Download progress component for Gradio."""
    
    def __init__(self):
        self.progress = 0.0
        self.status = "Ready"
    
    def create_component(self):
        """Create Gradio component."""
        with gr.Column():
            self.status_text = gr.Textbox(
                value=self.status,
                label="Status",
                interactive=False
            )
            self.progress_bar = gr.Progress()
        
        return self.status_text, self.progress_bar
    
    def download_with_progress(self, model_name: str):
        """Download model with progress updates."""
        def update_progress(progress: float):
            self.progress = progress
            self.progress_bar(progress)
        
        self.status = f"Downloading {model_name}..."
        self.status_text.update(value=self.status)
        
        try:
            path = _model_manager.download_model(
                model_name,
                progress_callback=update_progress
            )
            self.status = f"✓ {model_name} ready"
            self.status_text.update(value=self.status)
            return path
        except Exception as e:
            self.status = f"✗ Error: {str(e)}"
            self.status_text.update(value=self.status)
            raise
```

**2.5.5 Smart Caching with LRU + Disk Cache**

```python
# core/smart_cache.py
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

class SmartCache:
    """Smart caching system with disk persistence."""
    
    def __init__(self, cache_dir: Path, max_cache_size_gb: float = 50.0):
        self.cache_dir = cache_dir
        self.max_cache_size = max_cache_size_gb * 1024 * 1024 * 1024  # Convert to bytes
        self.cache_index_file = cache_dir / "cache_index.json"
        self._load_index()
    
    def _load_index(self):
        """Load cache index from disk."""
        if self.cache_index_file.exists():
            with open(self.cache_index_file, 'r') as f:
                self.index = json.load(f)
        else:
            self.index = {}
    
    def _save_index(self):
        """Save cache index to disk."""
        with open(self.cache_index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def _get_cache_size(self) -> int:
        """Get total size of cached files."""
        total = 0
        for path in self.cache_dir.rglob('*'):
            if path.is_file() and path != self.cache_index_file:
                total += path.stat().st_size
        return total
    
    def _evict_lru(self, required_space: int):
        """Evict least recently used items to make space."""
        # Sort by last access time
        items = sorted(
            self.index.items(),
            key=lambda x: x[1].get('last_access', 0)
        )
        
        freed_space = 0
        for model_name, info in items:
            if freed_space >= required_space:
                break
            
            model_path = Path(info['path'])
            if model_path.exists():
                size = info.get('size', 0)
                if model_path.is_dir():
                    shutil.rmtree(model_path)
                else:
                    model_path.unlink()
                
                del self.index[model_name]
                freed_space += size
        
        self._save_index()
    
    def ensure_space(self, required_size: int):
        """Ensure enough space is available."""
        current_size = self._get_cache_size()
        available_space = self.max_cache_size - current_size
        
        if available_space < required_size:
            self._evict_lru(required_size - available_space)
    
    def mark_accessed(self, model_name: str):
        """Mark a model as accessed."""
        if model_name in self.index:
            self.index[model_name]['last_access'] = datetime.now().isoformat()
            self._save_index()
```

### Phase 3: Unified API (Week 4)

**3.1 Model Registry**

```python
# core/registry.py
from typing import Dict, Type, Optional
from .base_upscaler import BaseUpscaler

class ModelRegistry:
    """Central registry for all upscaler models."""
    
    _models: Dict[str, Type[BaseUpscaler]] = {}
    _aliases: Dict[str, str] = {}
    
    @classmethod
    def register(
        cls,
        name: str,
        model_class: Type[BaseUpscaler],
        aliases: Optional[list] = None
    ):
        """Register a model class."""
        cls._models[name] = model_class
        
        if aliases:
            for alias in aliases:
                cls._aliases[alias] = name
    
    @classmethod
    def get(cls, name: str) -> Type[BaseUpscaler]:
        """Get model class by name or alias."""
        # Check aliases first
        if name in cls._aliases:
            name = cls._aliases[name]
        
        if name not in cls._models:
            raise ValueError(f"Unknown model: {name}")
        
        return cls._models[name]
    
    @classmethod
    def list_models(cls) -> list:
        """List all registered models."""
        return list(cls._models.keys())

# Register models
from ..models.hitsr import HiTSRAdapter
from ..models.tsdsr import TSDSRAdapter
from ..models.varsr import VARSRAdapter

ModelRegistry.register("hitsr-sir", HiTSRAdapter, aliases=["HiT-SIR", "Athena"])
ModelRegistry.register("hitsr-sng", HiTSRAdapter, aliases=["HiT-SNG", "Apollo"])
ModelRegistry.register("hitsr-srf", HiTSRAdapter, aliases=["HiT-SRF", "Artemis"])
ModelRegistry.register("tsdsr", TSDSRAdapter, aliases=["TSD-SR", "Hermes"])
ModelRegistry.register("varsr", VARSRAdapter, aliases=["VARSR", "Zeus"])
```

**3.2 Cache Manager (Summoner)**

```python
# core/cache_manager.py
from collections import OrderedDict
from typing import Optional, Dict, Any
import gc
import torch

class CacheManager:
    """Manages loaded models with LRU eviction."""
    
    def __init__(self, max_models: int = 3):
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._max_models = max_models
    
    def get(self, model_name: str) -> Optional[Any]:
        """Get model from cache."""
        if model_name in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(model_name)
            return self._cache[model_name]
        return None
    
    def put(self, model_name: str, model: Any):
        """Add model to cache."""
        # Check if we need to evict
        if len(self._cache) >= self._max_models:
            # Evict least recently used
            evicted_name, evicted_model = self._cache.popitem(last=False)
            self._cleanup_model(evicted_model)
        
        self._cache[model_name] = model
    
    def _cleanup_model(self, model):
        """Clean up evicted model."""
        if hasattr(model, 'to'):
            model.to('cpu')
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def clear(self, model_name: Optional[str] = None):
        """Clear specific model or all models."""
        if model_name:
            if model_name in self._cache:
                model = self._cache.pop(model_name)
                self._cleanup_model(model)
        else:
            for model in self._cache.values():
                self._cleanup_model(model)
            self._cache.clear()

# Global instance
_cache_manager = CacheManager()
```

**3.3 Main API**

```python
# api/pipeline.py
from typing import Union, Optional, Dict, Any
from PIL import Image
import numpy as np
from ..core.registry import ModelRegistry
from ..core.cache_manager import _cache_manager

class SuperscalePipeline:
    """Main pipeline for super-resolution."""
    
    def __init__(
        self,
        model: str,
        device: str = "auto",
        dtype: Optional[torch.dtype] = None,
        **kwargs
    ):
        self.model_name = model
        self.device = self._get_device(device)
        self.dtype = dtype or (torch.float16 if self.device != "cpu" else torch.float32)
        
        # Get model from cache or create new
        self._model = self._load_or_create_model(model, **kwargs)
    
    def _load_or_create_model(self, model_name: str, **kwargs):
        """Load model from cache or create new instance."""
        # Check cache first
        cached = _cache_manager.get(model_name)
        if cached:
            return cached
        
        # Create new model
        model_class = ModelRegistry.get(model_name)
        model = model_class(device=self.device, **kwargs)
        
        # Load weights (simplified - actual implementation would handle paths)
        checkpoint_path = self._get_checkpoint_path(model_name)
        model.load_weights(checkpoint_path)
        
        # Add to cache
        _cache_manager.put(model_name, model)
        
        return model
    
    def __call__(
        self,
        image: Union[Image.Image, np.ndarray, str],
        scale: int = 4,
        **kwargs
    ) -> Image.Image:
        """Run super-resolution."""
        return self._model.upscale(image, scale, **kwargs)

# Convenience functions
def load(model: str, **kwargs) -> SuperscalePipeline:
    """Load a super-resolution model."""
    return SuperscalePipeline(model, **kwargs)

def up(
    image: Union[Image.Image, np.ndarray, str],
    model: str = "Hermes",
    scale: int = 4,
    **kwargs
) -> Image.Image:
    """One-line super-resolution."""
    pipe = load(model, **kwargs)
    return pipe(image, scale)

# Aliases
summon = load
dismiss = lambda model: _cache_manager.clear(model)
```

### Phase 3.5: Efficient Continuous Usage Patterns (Week 4)

**3.5.1 Session-Based Processing**

```python
# api/session.py
from typing import List, Union, Optional, Iterator
from contextlib import contextmanager
import torch
from PIL import Image
import numpy as np
from .pipeline import SuperscalePipeline
from ..core.cache_manager import _cache_manager

class SuperscaleSession:
    """Efficient session management for continuous processing."""
    
    def __init__(
        self,
        model: str,
        device: str = "auto",
        persistent: bool = True,  # Keep model in VRAM
        **kwargs
    ):
        self.pipeline = SuperscalePipeline(model, device=device, **kwargs)
        self.persistent = persistent
        self._processed_count = 0
    
    def upscale(
        self,
        image: Union[Image.Image, np.ndarray, str],
        scale: int = 4,
        **kwargs
    ) -> Image.Image:
        """Upscale single image."""
        result = self.pipeline(image, scale, **kwargs)
        self._processed_count += 1
        return result
    
    def upscale_batch(
        self,
        images: List[Union[Image.Image, np.ndarray, str]],
        scale: int = 4,
        batch_size: int = 1,
        **kwargs
    ) -> List[Image.Image]:
        """Batch process multiple images efficiently."""
        results = []
        
        # Process in batches to optimize GPU usage
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            if batch_size > 1 and hasattr(self.pipeline._model, 'batch_forward'):
                # Use batch processing if supported
                batch_results = self.pipeline._model.batch_forward(batch, scale, **kwargs)
                results.extend(batch_results)
            else:
                # Fall back to sequential processing
                for img in batch:
                    result = self.upscale(img, scale, **kwargs)
                    results.append(result)
        
        return results
    
    def upscale_generator(
        self,
        images: Iterator[Union[Image.Image, np.ndarray, str]],
        scale: int = 4,
        **kwargs
    ) -> Iterator[Image.Image]:
        """Generator for memory-efficient processing of large datasets."""
        for image in images:
            yield self.upscale(image, scale, **kwargs)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup if not persistent."""
        if not self.persistent:
            # Move model to CPU to free VRAM
            if hasattr(self.pipeline._model, 'to'):
                self.pipeline._model.to('cpu')
            torch.cuda.empty_cache()

@contextmanager
def session(model: str, **kwargs) -> SuperscaleSession:
    """Context manager for efficient batch processing."""
    sess = SuperscaleSession(model, **kwargs)
    try:
        yield sess
    finally:
        # Cleanup handled by __exit__
        pass

# Usage examples:
# with session("Hermes") as sess:
#     for img in images:
#         result = sess.upscale(img)
```

**3.5.2 Pipeline Pooling for Concurrent Processing**

```python
# api/pool.py
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Union, Optional, Callable
import multiprocessing as mp
from .session import SuperscaleSession

class SuperscalePool:
    """Pool of pipelines for concurrent processing."""
    
    def __init__(
        self,
        model: str,
        num_workers: int = None,
        device_ids: Optional[List[int]] = None,
        executor_type: str = "thread",  # "thread" or "process"
        **kwargs
    ):
        self.model = model
        self.num_workers = num_workers or mp.cpu_count()
        self.device_ids = device_ids or [0]  # GPU IDs for multi-GPU
        self.executor_type = executor_type
        self.kwargs = kwargs
        
        # Create worker pool
        if executor_type == "thread":
            self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
        else:
            self.executor = ProcessPoolExecutor(max_workers=self.num_workers)
        
        # Pre-load models for each worker
        self._init_workers()
    
    def _init_workers(self):
        """Initialize model on each worker."""
        self.sessions = []
        for i in range(self.num_workers):
            # Distribute across GPUs if multiple available
            device_id = self.device_ids[i % len(self.device_ids)]
            device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
            
            session = SuperscaleSession(
                self.model,
                device=device,
                persistent=True,
                **self.kwargs
            )
            self.sessions.append(session)
    
    def map(
        self,
        images: List[Union[Image.Image, np.ndarray, str]],
        scale: int = 4,
        callback: Optional[Callable] = None,
        **kwargs
    ) -> List[Image.Image]:
        """Process images in parallel."""
        futures = []
        results = [None] * len(images)
        
        for idx, image in enumerate(images):
            # Round-robin distribution
            session = self.sessions[idx % len(self.sessions)]
            
            future = self.executor.submit(
                session.upscale, image, scale, **kwargs
            )
            futures.append((idx, future))
        
        # Collect results
        for idx, future in futures:
            result = future.result()
            results[idx] = result
            if callback:
                callback(idx, result)
        
        return results
    
    def close(self):
        """Clean up resources."""
        self.executor.shutdown(wait=True)
        for session in self.sessions:
            if hasattr(session.pipeline._model, 'to'):
                session.pipeline._model.to('cpu')
        torch.cuda.empty_cache()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
```

**3.5.3 Lazy Loading and Warm Start**

```python
# core/lazy_loader.py
from typing import Optional, Dict, Any
import torch
from functools import lru_cache

class LazyModelLoader:
    """Lazy loading with warm start optimization."""
    
    def __init__(self):
        self._model_specs: Dict[str, Dict[str, Any]] = {}
        self._warm_models: Dict[str, Any] = {}
    
    def register_model_spec(self, name: str, spec: Dict[str, Any]):
        """Register model specification for lazy loading."""
        self._model_specs[name] = spec
    
    def warm_start(self, model_names: List[str], device: str = "cpu"):
        """Pre-load models for faster first inference."""
        for name in model_names:
            if name not in self._warm_models:
                print(f"Warming up {name}...")
                model = self._load_model(name, device)
                
                # Run dummy inference to initialize CUDA kernels
                dummy_input = torch.randn(1, 3, 64, 64).to(device)
                with torch.no_grad():
                    _ = model(dummy_input)
                
                self._warm_models[name] = model
    
    @lru_cache(maxsize=None)
    def _load_model(self, name: str, device: str) -> Any:
        """Load model with caching."""
        spec = self._model_specs.get(name)
        if not spec:
            raise ValueError(f"Model {name} not registered")
        
        # Load model based on spec
        # Implementation depends on model type
        return model

# Global lazy loader
_lazy_loader = LazyModelLoader()
```

**3.5.4 Streaming Processing for Video/Large Datasets**

```python
# api/streaming.py
from typing import Iterator, Optional, Callable
import numpy as np
from PIL import Image
from queue import Queue
from threading import Thread
import cv2

class StreamProcessor:
    """Streaming processor for continuous input sources."""
    
    def __init__(
        self,
        model: str,
        buffer_size: int = 10,
        **kwargs
    ):
        self.session = SuperscaleSession(model, **kwargs)
        self.buffer_size = buffer_size
        self.input_queue = Queue(maxsize=buffer_size)
        self.output_queue = Queue(maxsize=buffer_size)
        self._processing = False
    
    def process_video(
        self,
        input_path: str,
        output_path: str,
        scale: int = 4,
        fps: Optional[float] = None,
        callback: Optional[Callable] = None
    ):
        """Process video file frame by frame."""
        cap = cv2.VideoCapture(input_path)
        
        # Get video properties
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_fps = fps or orig_fps
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            out_fps,
            (width * scale, height * scale)
        )
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)
            
            # Upscale
            upscaled = self.session.upscale(pil_frame, scale)
            
            # Convert back to OpenCV format
            upscaled_np = np.array(upscaled)
            upscaled_bgr = cv2.cvtColor(upscaled_np, cv2.COLOR_RGB2BGR)
            
            # Write frame
            out.write(upscaled_bgr)
            
            frame_count += 1
            if callback:
                callback(frame_count)
        
        cap.release()
        out.release()
    
    def process_stream(
        self,
        input_iterator: Iterator[np.ndarray],
        scale: int = 4
    ) -> Iterator[np.ndarray]:
        """Process streaming input with buffering."""
        # Start processing thread
        self._processing = True
        process_thread = Thread(
            target=self._process_worker,
            args=(scale,)
        )
        process_thread.start()
        
        # Feed input
        for frame in input_iterator:
            self.input_queue.put(frame)
            if not self.output_queue.empty():
                yield self.output_queue.get()
        
        # Signal end of stream
        self.input_queue.put(None)
        
        # Drain remaining output
        while not self.output_queue.empty():
            yield self.output_queue.get()
        
        self._processing = False
        process_thread.join()
    
    def _process_worker(self, scale: int):
        """Worker thread for processing."""
        while self._processing:
            frame = self.input_queue.get()
            if frame is None:
                break
            
            # Process frame
            pil_frame = Image.fromarray(frame)
            upscaled = self.session.upscale(pil_frame, scale)
            upscaled_np = np.array(upscaled)
            
            self.output_queue.put(upscaled_np)
```

**3.5.5 Memory-Efficient Tiled Processing**

```python
# api/tiling.py
from typing import Tuple, Optional
import numpy as np
from PIL import Image

class TiledProcessor:
    """Process large images in tiles to save memory."""
    
    def __init__(
        self,
        session: SuperscaleSession,
        tile_size: int = 512,
        overlap: int = 32
    ):
        self.session = session
        self.tile_size = tile_size
        self.overlap = overlap
    
    def process_large_image(
        self,
        image: Union[Image.Image, np.ndarray],
        scale: int = 4,
        **kwargs
    ) -> Image.Image:
        """Process large image in tiles."""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        h, w = image.shape[:2]
        output_h, output_w = h * scale, w * scale
        output = np.zeros((output_h, output_w, 3), dtype=np.uint8)
        
        # Calculate tile positions
        tiles = self._calculate_tiles(h, w)
        
        for y, x, tile_h, tile_w in tiles:
            # Extract tile with overlap
            y_start = max(0, y - self.overlap)
            x_start = max(0, x - self.overlap)
            y_end = min(h, y + tile_h + self.overlap)
            x_end = min(w, x + tile_w + self.overlap)
            
            tile = image[y_start:y_end, x_start:x_end]
            
            # Process tile
            tile_pil = Image.fromarray(tile)
            upscaled_tile = self.session.upscale(tile_pil, scale, **kwargs)
            upscaled_tile_np = np.array(upscaled_tile)
            
            # Calculate output position
            out_y_start = y * scale
            out_x_start = x * scale
            out_y_end = out_y_start + tile_h * scale
            out_x_end = out_x_start + tile_w * scale
            
            # Blend overlapping regions
            if self.overlap > 0:
                upscaled_tile_np = self._blend_tile(
                    output, upscaled_tile_np,
                    out_y_start, out_x_start,
                    scale
                )
            
            # Place tile in output
            output[out_y_start:out_y_end, out_x_start:out_x_end] = \
                upscaled_tile_np[:tile_h * scale, :tile_w * scale]
        
        return Image.fromarray(output)
    
    def _calculate_tiles(
        self,
        height: int,
        width: int
    ) -> List[Tuple[int, int, int, int]]:
        """Calculate tile positions."""
        tiles = []
        for y in range(0, height, self.tile_size - self.overlap):
            for x in range(0, width, self.tile_size - self.overlap):
                tile_h = min(self.tile_size, height - y)
                tile_w = min(self.tile_size, width - x)
                tiles.append((y, x, tile_h, tile_w))
        return tiles
```

### Phase 4: CLI & Testing (Week 5)

**4.1 CLI Implementation**

```python
# cli/main.py
import click
from pathlib import Path
from PIL import Image
from ..api import up, load, dismiss
from ..core.registry import ModelRegistry

@click.group()
def cli():
    """Superscale - Universal Super-Resolution Toolkit"""
    pass

@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('-m', '--model', default='Hermes', help='Model to use')
@click.option('-s', '--scale', default=4, type=int, help='Upscaling factor')
@click.option('-o', '--output', help='Output path')
@click.option('-d', '--device', default='auto', help='Device (cpu/cuda/auto)')
def up(input_path, model, scale, output, device):
    """Upscale an image."""
    # Load image
    image = Image.open(input_path)
    
    # Upscale
    result = up(image, model=model, scale=scale, device=device)
    
    # Save
    if not output:
        input_path = Path(input_path)
        output = input_path.parent / f"{input_path.stem}_x{scale}{input_path.suffix}"
    
    result.save(output)
    click.echo(f"Saved to: {output}")

@cli.command()
def list_models():
    """List available models."""
    models = ModelRegistry.list_models()
    click.echo("Available models:")
    for model in models:
        click.echo(f"  - {model}")

@cli.command()
@click.argument('model', required=False)
def dismiss(model):
    """Free model from memory."""
    dismiss(model)
    click.echo(f"Dismissed: {model or 'all models'}")
```

**4.2 Testing Strategy**

```python
# tests/test_api.py
import pytest
from PIL import Image
import numpy as np
from superscale import up, load, dismiss

@pytest.fixture
def sample_image():
    """Create a small test image."""
    return Image.new('RGB', (64, 64), color='red')

@pytest.mark.parametrize("model", ["Hermes", "Athena", "Zeus"])
def test_basic_upscale(sample_image, model):
    """Test basic upscaling for each model."""
    try:
        result = up(sample_image, model=model, scale=4, device="cpu")
        assert result.size == (256, 256)
    except ImportError:
        pytest.skip(f"Dependencies for {model} not installed")

def test_pipeline_interface(sample_image):
    """Test pipeline interface."""
    pipe = load("Hermes", device="cpu")
    result = pipe(sample_image, scale=2)
    assert result.size == (128, 128)

def test_cache_management():
    """Test model caching."""
    # Load multiple models
    pipe1 = load("Hermes")
    pipe2 = load("Athena")
    pipe3 = load("Zeus")
    
    # Dismiss specific model
    dismiss("Hermes")
    
    # Dismiss all
    dismiss()
```

### Phase 5: Advanced Features (Week 6-7)

**5.1 Batch Processing**

```python
# api/batch.py
from typing import List, Union
from concurrent.futures import ThreadPoolExecutor
import torch
from .pipeline import SuperscalePipeline

class BatchProcessor:
    """Batch processing for multiple images."""
    
    def __init__(self, pipeline: SuperscalePipeline):
        self.pipeline = pipeline
    
    def process_batch(
        self,
        images: List[Union[str, Image.Image]],
        scale: int = 4,
        batch_size: int = 4,
        **kwargs
    ) -> List[Image.Image]:
        """Process multiple images in batches."""
        results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            # Process batch
            batch_results = []
            for img in batch:
                result = self.pipeline(img, scale, **kwargs)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results
```

**5.2 Model Conversion Scripts**

```python
# scripts/convert_weights.py
import argparse
from pathlib import Path

def convert_hitsr_weights(input_path: Path, output_path: Path):
    """Convert HiT-SR weights to unified format."""
    # Implementation
    pass

def convert_tsdsr_weights(input_path: Path, output_path: Path):
    """Convert TSD-SR weights to unified format."""
    # Implementation
    pass

def convert_varsr_weights(input_path: Path, output_path: Path):
    """Convert VARSR weights to unified format."""
    # Implementation
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_type", choices=["hitsr", "tsdsr", "varsr"])
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    
    args = parser.parse_args()
    
    converters = {
        "hitsr": convert_hitsr_weights,
        "tsdsr": convert_tsdsr_weights,
        "varsr": convert_varsr_weights,
    }
    
    converters[args.model_type](args.input_path, args.output_path)
```

### Phase 6: GUI Development (Week 8)

```python
# gui/app.py
import gradio as gr
from ..api import up, ModelRegistry

def create_app():
    """Create Gradio interface."""
    
    def upscale_image(image, model, scale):
        """Upscale callback."""
        if image is None:
            return None
        
        result = up(image, model=model, scale=scale)
        return result
    
    # Get available models
    models = ModelRegistry.list_models()
    
    # Create interface
    with gr.Blocks(title="Superscale") as app:
        gr.Markdown("# Superscale - Super Resolution Toolkit")
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input Image", type="pil")
                model_dropdown = gr.Dropdown(
                    choices=models,
                    value=models[0],
                    label="Model"
                )
                scale_slider = gr.Slider(
                    minimum=2,
                    maximum=8,
                    value=4,
                    step=1,
                    label="Scale"
                )
                upscale_btn = gr.Button("Upscale", variant="primary")
            
            with gr.Column():
                output_image = gr.Image(label="Output Image")
        
        # Connect events
        upscale_btn.click(
            fn=upscale_image,
            inputs=[input_image, model_dropdown, scale_slider],
            outputs=output_image
        )
    
    return app

if __name__ == "__main__":
    app = create_app()
    app.launch()
```

## 3. Dependency Resolution Strategy

### 3.1 Environment Isolation

```python
# core/environment.py
import subprocess
import sys
from pathlib import Path
import venv

class ModelEnvironment:
    """Isolated environment for model-specific dependencies."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.env_path = Path.home() / ".superscale" / "envs" / model_name
    
    def setup(self):
        """Create virtual environment."""
        if not self.env_path.exists():
            venv.create(self.env_path, with_pip=True)
            self._install_dependencies()
    
    def _install_dependencies(self):
        """Install model-specific dependencies."""
        pip_path = self.env_path / "bin" / "pip"
        
        requirements = {
            "hitsr": ["torch>=2.0", "timm>=0.9", "einops"],
            "tsdsr": ["torch==2.2.2", "diffusers==0.29.1", "transformers==4.49.0"],
            "varsr": ["torch==2.2.1", "diffusers==0.32.2", "transformers==4.37.2"],
        }
        
        if self.model_name in requirements:
            subprocess.run([
                str(pip_path), "install", *requirements[self.model_name]
            ])
    
    def run_in_env(self, code: str) -> Any:
        """Run code in isolated environment."""
        python_path = self.env_path / "bin" / "python"
        # Implementation using subprocess or similar
        pass
```

### 3.2 Dynamic Import System

```python
# core/dynamic_loader.py
import importlib
import sys
from typing import Any, Optional

class DynamicLoader:
    """Dynamic module loader with version checking."""
    
    @staticmethod
    def try_import(
        module_name: str,
        min_version: Optional[str] = None,
        max_version: Optional[str] = None
    ) -> Any:
        """Try to import module with version constraints."""
        try:
            module = importlib.import_module(module_name)
            
            # Check version if specified
            if hasattr(module, "__version__"):
                version = module.__version__
                # Version checking logic
                
            return module
            
        except ImportError:
            return None
    
    @staticmethod
    def import_with_fallback(primary: str, fallback: str) -> Any:
        """Import with fallback option."""
        module = DynamicLoader.try_import(primary)
        if module is None:
            module = DynamicLoader.try_import(fallback)
            if module is None:
                raise ImportError(f"Neither {primary} nor {fallback} available")
        return module
```

## 4. Testing & Quality Assurance

### 4.1 Integration Tests

```python
# tests/integration/test_models.py
import pytest
from pathlib import Path
import numpy as np
from PIL import Image
from superscale import up

class TestModelIntegration:
    """Integration tests for all models."""
    
    @pytest.fixture
    def test_images(self):
        """Generate test images of various sizes."""
        return {
            "small": Image.new('RGB', (64, 64), 'red'),
            "medium": Image.new('RGB', (256, 256), 'green'),
            "large": Image.new('RGB', (512, 512), 'blue'),
            "grayscale": Image.new('L', (128, 128), 128),
        }
    
    @pytest.mark.parametrize("model,scale", [
        ("Hermes", 2), ("Hermes", 4),
        ("Athena", 2), ("Athena", 4),
        ("Zeus", 2), ("Zeus", 4),
    ])
    def test_upscaling_consistency(self, test_images, model, scale):
        """Test that models produce consistent output sizes."""
        for name, img in test_images.items():
            try:
                result = up(img, model=model, scale=scale)
                expected_size = (img.width * scale, img.height * scale)
                assert result.size == expected_size
            except ImportError:
                pytest.skip(f"{model} dependencies not installed")
```

### 4.2 Performance Benchmarks

```python
# tests/benchmarks/benchmark_models.py
import time
from statistics import mean, stdev
from superscale import load

def benchmark_model(model_name: str, image_sizes: list, iterations: int = 10):
    """Benchmark model performance."""
    pipe = load(model_name, device="cuda")
    results = {}
    
    for size in image_sizes:
        # Create test image
        img = Image.new('RGB', (size, size))
        times = []
        
        # Warm up
        pipe(img, scale=4)
        
        # Benchmark
        for _ in range(iterations):
            start = time.time()
            pipe(img, scale=4)
            times.append(time.time() - start)
        
        results[size] = {
            "mean": mean(times),
            "stdev": stdev(times),
            "min": min(times),
            "max": max(times),
        }
    
    return results
```

## 5. Documentation & Examples

### 5.1 API Documentation

```python
# docs/api_reference.md
"""
# Superscale API Reference

## Quick Start

```python
import superscale as ss

# One-line upscaling
hr_image = ss.up("low_res.jpg", model="Hermes", scale=4)

# Pipeline interface
pipe = ss.load("Athena", device="cuda")
hr_image = pipe("low_res.jpg", scale=4)

# Batch processing
images = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = ss.batch_up(images, model="Zeus", scale=2)
```

## Models

| Model | Alias | Type | Best For |
|-------|-------|------|----------|
| tsdsr | Hermes | Diffusion | High quality, slower |
| hitsr-sir | Athena | Transformer | Balanced |
| varsr | Zeus | Autoregressive | Fast, good quality |
"""
```

### 5.2 Migration Guide

```python
# docs/migration_guide.md
"""
# Migration Guide

## From Native Models

### HiT-SR
```python
# Before
from basicsr.models import create_model
model = create_model(opt)

# After
import superscale as ss
pipe = ss.load("Athena")
```

### TSD-SR
```python
# Before
from test_tsdsr import load_tsdsr
model = load_tsdsr()

# After
import superscale as ss
pipe = ss.load("Hermes")
```
"""
```

## 6. CI/CD Pipeline

### 6.1 GitHub Actions

```yaml
# .github/workflows/test.yml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10]
        extras: [base, hitsr, tsdsr, varsr, all]
    
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -e .[${{ matrix.extras }}]
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=superscale
```

## 7. Release Strategy

### 7.1 Version Management

```toml
# pyproject.toml
[tool.poetry]
version = "0.1.0"

[tool.poetry-dynamic-versioning]
enable = true
pattern = "^v(?P<base>\\d+\\.\\d+\\.\\d+)"
```

### 7.2 Release Checklist

1. **Pre-release**
   - [ ] All tests passing
   - [ ] Documentation updated
   - [ ] CHANGELOG.md updated
   - [ ] Version bumped

2. **Release**
   - [ ] Tag version
   - [ ] Build distributions
   - [ ] Upload to PyPI
   - [ ] Create GitHub release

3. **Post-release**
   - [ ] Update model weights on HuggingFace
   - [ ] Announce on social media
   - [ ] Update examples

## 8. Future Enhancements

### 8.1 Planned Features

1. **Model Quantization**
   - INT8/INT4 quantization support
   - Mobile deployment

2. **Additional Models**
   - Real-ESRGAN integration
   - ESRGAN variants
   - Custom model support

3. **Advanced Features**
   - Video super-resolution
   - Real-time processing
   - Web API service

### 8.2 Community Features

1. **Model Hub**
   - User-uploaded models
   - Fine-tuned variants
   - Domain-specific models

2. **Benchmarking Suite**
   - Standardized benchmarks
   - Quality metrics
   - Performance tracking

## 9. Git Management & Packaging Strategy

### 9.1 Repository Structure

```
superscale/
├── superscale/              # Main package (included in wheel)
│   ├── backends/
│   │   ├── native/         # Copied model implementations
│   │   │   ├── hitsr/     # Minimal HiT-SR code
│   │   │   │   ├── __init__.py
│   │   │   │   ├── models.py
│   │   │   │   ├── utils.py
│   │   │   │   └── configs/
│   │   │   ├── tsdsr/     # Minimal TSD-SR code
│   │   │   │   ├── __init__.py
│   │   │   │   ├── pipeline.py
│   │   │   │   ├── lora_utils.py
│   │   │   │   └── configs/
│   │   │   └── varsr/     # Minimal VARSR code
│   │   │       ├── __init__.py
│   │   │       ├── var_model.py
│   │   │       ├── vqvae.py
│   │   │       └── configs/
│   │   └── diffusers/     # Future: Diffusers-compatible versions
│   └── ...                # Core library code
├── third_party/           # Git submodules (NOT in wheel)
│   ├── HiT-SR/           # Git submodule
│   ├── TSD-SR/           # Git submodule
│   ├── VARSR/            # Git submodule
│   └── LICENSE-NOTICE/   # License files from upstream
├── scripts/
│   ├── sync_models.py    # Copy from third_party to backends
│   ├── minimize_code.py  # Strip unnecessary files
│   └── verify_licenses.py
├── LICENSE
├── LICENSE-3rdparty.md   # Third-party license summary
└── pyproject.toml
```

### 9.2 Git Submodule Setup

```bash
# Initial setup (development branch)
git checkout -b dev/integrate-models

# Add submodules
git submodule add --name hitsr https://github.com/XPixelGroup/HiT-SR.git third_party/HiT-SR
git submodule add --name tsdsr https://github.com/Iceclear/TSD-SR.git third_party/TSD-SR  
git submodule add --name varsr https://github.com/FoundationVision/VARSR.git third_party/VARSR

# Pin to specific versions
cd third_party/HiT-SR && git checkout v1.0.0 && cd ../..
cd third_party/TSD-SR && git checkout stable-v1 && cd ../..
cd third_party/VARSR && git checkout v1.0 && cd ../..

# Commit submodule references
git add .gitmodules third_party/
git commit -m "Add model submodules (dev only)"
```

### 9.3 Model Code Synchronization

```python
# scripts/sync_models.py
import shutil
from pathlib import Path
import ast
import re

class ModelSynchronizer:
    """Sync minimal code from submodules to backends."""
    
    # Files to copy for each model
    SYNC_PATTERNS = {
        "hitsr": {
            "include": [
                "basicsr/archs/hit_*.py",
                "basicsr/models/hit_*.py", 
                "options/test/*.yml",
            ],
            "exclude": [
                "*_test.py",
                "*.md",
                "__pycache__",
                ".git",
            ],
            "rename": {
                "basicsr/archs": "models",
                "basicsr/models": "trainers",
            }
        },
        "tsdsr": {
            "include": [
                "ldm_patched/modules/sd3/*.py",
                "ldm_patched/modules/model_*.py",
                "configs/*.yaml",
            ],
            "exclude": [
                "test_*.py",
                "benchmark_*.py",
            ]
        },
        "varsr": {
            "include": [
                "models/var.py",
                "models/vqvae.py",
                "models/basic_var.py",
                "utils/amp_opt.py",
            ],
            "exclude": [
                "*_test.py",
                "demo_*.py",
            ]
        }
    }
    
    def sync_model(self, model_name: str):
        """Sync one model from third_party to backends/native."""
        src_dir = Path(f"third_party/{model_name}")
        dst_dir = Path(f"superscale/backends/native/{model_name.lower()}")
        
        # Clear destination
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        dst_dir.mkdir(parents=True)
        
        # Copy files based on patterns
        patterns = self.SYNC_PATTERNS[model_name.lower()]
        for pattern in patterns["include"]:
            # Copy matching files...
            pass
        
        # Minimize imports
        self._minimize_imports(dst_dir)
        
        # Add __init__.py with proper exports
        self._create_init_file(dst_dir, model_name)
        
        # Copy license
        self._copy_license(src_dir, dst_dir)
    
    def _minimize_imports(self, path: Path):
        """Remove unnecessary imports and dependencies."""
        for py_file in path.rglob("*.py"):
            content = py_file.read_text()
            
            # Remove test imports
            content = re.sub(r'^import pytest.*$', '', content, flags=re.MULTILINE)
            content = re.sub(r'^from .* import.*test.*$', '', content, flags=re.MULTILINE)
            
            # Update relative imports
            content = content.replace('from basicsr.', 'from .')
            
            py_file.write_text(content)
    
    def _copy_license(self, src_dir: Path, dst_dir: Path):
        """Copy license file."""
        for license_file in ["LICENSE", "LICENSE.md", "LICENSE.txt"]:
            src_license = src_dir / license_file
            if src_license.exists():
                dst_license = dst_dir / "LICENSE"
                shutil.copy2(src_license, dst_license)
                break

if __name__ == "__main__":
    syncer = ModelSynchronizer()
    for model in ["HiT-SR", "TSD-SR", "VARSR"]:
        print(f"Syncing {model}...")
        syncer.sync_model(model)
```

### 9.4 License Management

```markdown
# LICENSE-3rdparty.md

# Third-Party Licenses

This project includes code from the following projects:

## HiT-SR
- **Source**: https://github.com/XPixelGroup/HiT-SR
- **License**: Apache-2.0
- **Copyright**: Copyright (c) 2024 XPixel Group
- **Files**: superscale/backends/native/hitsr/*

## TSD-SR  
- **Source**: https://github.com/Iceclear/TSD-SR
- **License**: Apache-2.0
- **Copyright**: Copyright (c) 2024 TSD-SR Authors
- **Files**: superscale/backends/native/tsdsr/*

## VARSR
- **Source**: https://github.com/FoundationVision/VARSR
- **License**: MIT
- **Copyright**: Copyright (c) 2024 Foundation Vision
- **Files**: superscale/backends/native/varsr/*

## Additional Components

### Stable Diffusion 3
- **Source**: https://huggingface.co/stabilityai/stable-diffusion-3-medium
- **License**: Stability AI Community License
- **Note**: Weights downloaded separately, not included in package

For full license texts, see the LICENSE file in each model's directory.
```

### 9.5 Package Configuration

```toml
# pyproject.toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "superscale"
version = "0.1.0"
description = "Universal super-resolution toolkit with diffusers-like API"
readme = "README.md"
license = "Apache-2.0"
authors = [
    {name = "Superscale Contributors"},
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [
    "torch>=2.2,<2.4",
    "pillow>=10.0",
    "numpy>=1.24,<2.0",
    "tqdm>=4.65",
    "pyyaml>=6.0",
    "huggingface-hub>=0.20",
    "requests>=2.28",
]

[project.optional-dependencies]
# Minimal deps for each model
hitsr = [
    "opencv-python>=4.8",
    "scipy>=1.9",
]
tsdsr = [
    "safetensors>=0.4",
    "omegaconf>=2.3",
]
varsr = [
    "einops>=0.7",
]
# Full deps for development
dev = [
    "pytest>=7.0",
    "ruff>=0.1.0",
    "black>=23.0",
    "twine>=4.0",
]
# All models
all = ["superscale[hitsr,tsdsr,varsr]"]

[project.scripts]
superscale = "superscale.cli.main:cli"

[tool.hatch.build]
include = [
    "superscale/**/*.py",
    "superscale/**/*.yaml",
    "superscale/**/*.yml", 
    "superscale/**/*.json",
    "superscale/backends/native/*/LICENSE",
]
exclude = [
    "*.pyc",
    "__pycache__",
    ".git",
    "third_party/**",  # Exclude submodules
    "**/*_test.py",
    "**/test_*.py",
    "tests/**",
    "docs/**",
    "examples/**",
    "*.md",
    "!README.md",
    "!LICENSE-3rdparty.md",
]

[tool.hatch.build.targets.wheel]
packages = ["superscale"]

# Size optimization
[tool.hatch.build.targets.wheel.shared-data]
"LICENSE-3rdparty.md" = "superscale/LICENSE-3rdparty.md"
```

### 9.6 CI/CD Workflow

```yaml
# .github/workflows/sync-models.yml
name: Sync Model Code

on:
  pull_request:
    paths:
      - 'third_party/**'
  workflow_dispatch:

jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
      with:
        submodules: recursive
        
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
        
    - name: Sync model code
      run: |
        python scripts/sync_models.py
        
    - name: Verify licenses
      run: |
        python scripts/verify_licenses.py
        
    - name: Check package size
      run: |
        pip wheel . --no-deps -w dist/
        ls -lh dist/
        # Fail if wheel > 50MB
        size=$(stat -f%z dist/*.whl 2>/dev/null || stat -c%s dist/*.whl)
        if [ $size -gt 52428800 ]; then
          echo "Wheel too large: $size bytes"
          exit 1
        fi
        
    - name: Test minimal install
      run: |
        pip install dist/*.whl
        python -c "import superscale; print(superscale.__version__)"
```

### 9.7 Development Workflow

```bash
# Development setup
git clone --recursive https://github.com/org/superscale.git
cd superscale
git checkout dev/integrate-models

# Update submodules
git submodule update --init --recursive

# Sync latest changes
python scripts/sync_models.py

# Run tests
pytest tests/

# Build package
pip install build
python -m build

# Check package contents
tar -tf dist/superscale-*.tar.gz | less
unzip -l dist/superscale-*.whl | less
```

### 9.8 Release Process

```bash
# 1. Update submodules to stable versions
cd third_party/HiT-SR && git checkout v1.0.0 && cd ../..
cd third_party/TSD-SR && git checkout stable-v1 && cd ../..
cd third_party/VARSR && git checkout v1.0 && cd ../..

# 2. Sync and minimize code
python scripts/sync_models.py --minimize

# 3. Run full test suite
pytest tests/ --cov=superscale

# 4. Update version
# Edit pyproject.toml version

# 5. Build release
python -m build

# 6. Upload to PyPI
twine upload dist/*

# 7. Tag release (main branch, no submodules)
git checkout main
git merge dev/integrate-models --no-commit
git reset HEAD third_party .gitmodules
git commit -m "Release v0.1.0"
git tag v0.1.0
git push origin main --tags
```

### 9.9 Size Optimization Techniques

```python
# scripts/minimize_code.py
def minimize_for_release():
    """Minimize code for release."""
    
    # Remove docstrings from non-public methods
    # Remove type hints in performance-critical code
    # Merge small modules
    # Remove duplicate utility functions
    
    # Target sizes:
    # - hitsr: < 5MB
    # - tsdsr: < 10MB (excluding SD3 base) 
    # - varsr: < 8MB
    # Total wheel: < 30MB
```

## Conclusion

This implementation strategy provides a robust foundation for creating a unified super-resolution library. The phased approach ensures that each component is properly tested before moving to the next phase, while the modular architecture allows for easy extension and maintenance.

Key success factors:
- Careful dependency management
- Consistent API design
- Comprehensive testing
- Clear documentation
- Active community engagement
- Proper license compliance
- Efficient packaging strategy