"""Model download and management."""

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, List, Callable, Any
from urllib.parse import urlparse

import requests
from tqdm import tqdm

from .utils import get_cache_dir, format_bytes


class ModelManager:
    """Manages model downloads and local storage."""
    
    # Model configurations with download info
    MODEL_CONFIGS = {
        # HiT-SR Models (from Google Drive)
        "hitsr-sir-x2": {
            "repo_id": "XiangZ/hit-sr",
            "filename": "hit-sir-2x",
            "size": "~3MB",
            "type": "huggingface",
        },
        "hitsr-sir-x3": {
            "repo_id": "XiangZ/hit-sr",
            "filename": "hit-sir-3x",
            "size": "~3MB",
            "type": "huggingface",
        },
        "hitsr-sir-x4": {
            "gdrive_id": "1P_b9WakJKYPVGJ1yc34ey8S_QXdDT4Ua",
            "filename": "HiT-SIR-4x.pth",
            "size": "~3MB",
            "sha256": None,
            "type": "gdrive",
        },
        "hitsr-sng-x2": {
            "repo_id": "XiangZ/hit-sr",
            "filename": "hit-sng-2x",
            "size": "~4MB",
            "type": "huggingface",
        },
        "hitsr-sng-x3": {
            "repo_id": "XiangZ/hit-sr",
            "filename": "hit-sng-3x",
            "size": "~4MB",
            "type": "huggingface",
        },
        "hitsr-sng-x4": {
            "repo_id": "XiangZ/hit-sr",
            "filename": "hit-sng-4x",
            "size": "~4MB",
            "type": "huggingface",
        },
        "hitsr-srf-x2": {
            "repo_id": "XiangZ/hit-sr",
            "filename": "hit-srf-2x",
            "size": "~3MB",
            "type": "huggingface",
        },
        "hitsr-srf-x3": {
            "repo_id": "XiangZ/hit-sr",
            "filename": "hit-srf-3x",
            "size": "~3MB",
            "type": "huggingface",
        },
        "hitsr-srf-x4": {
            "repo_id": "XiangZ/hit-sr",
            "filename": "hit-srf-4x",
            "size": "~3MB",
            "type": "huggingface",
        },
        
        # Dummy models for testing
        "dummy": {
            "url": "none",
            "filename": "dummy.pth",
            "size": "0MB",
            "type": "dummy",
        },
        "dummy-hermes": {
            "url": "none", 
            "filename": "dummy.pth",
            "size": "0MB",
            "type": "dummy",
        },
        
        # Placeholder for other models
        "tsdsr": {
            "repo_id": "stabilityai/stable-diffusion-3-medium-diffusers",
            "size": "~5.4GB",
            "type": "huggingface",
            "note": "Requires additional LoRA weights",
        },
        "varsr-d16": {
            "repo_id": "qyp2000/VARSR",
            "filename": "varsr_d16.pth",
            "size": "~310MB",
            "type": "huggingface",
        },
    }
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize model manager."""
        self.cache_dir = cache_dir or get_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_file = self.cache_dir / "metadata.json"
        self._load_metadata()
    
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
            # Check if it's an alias
            from ..core.registry import ModelRegistry
            try:
                # This might resolve an alias
                model_class = ModelRegistry.get(model_name)
                # Get the primary name
                for name, cls in ModelRegistry._models.items():
                    if cls == model_class:
                        model_name = name
                        break
            except:
                pass
            
            if model_name not in self.MODEL_CONFIGS:
                return None
        
        config = self.MODEL_CONFIGS[model_name]
        
        # Dummy models don't need real files
        if config["type"] == "dummy":
            # Return a fake path that will trigger the model to load without weights
            return Path("dummy")
        
        if config["type"] == "direct" or config["type"] == "gdrive":
            model_dir = self.cache_dir / "checkpoints" / model_name
            model_path = model_dir / config["filename"]
            if model_path.exists():
                return model_path
        
        elif config["type"] == "huggingface":
            model_dir = self.cache_dir / "checkpoints" / model_name
            if "filename" in config:
                model_path = model_dir / config["filename"]
                if model_path.exists():
                    return model_path
            elif model_dir.exists():
                # Look for any model file
                for ext in [".pth", ".pt", ".ckpt", ".safetensors", ".bin"]:
                    for file in model_dir.glob(f"*{ext}"):
                        return file
        
        return None
    
    def download_model(
        self,
        model_name: str,
        force: bool = False,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Path:
        """Download a model if not already cached."""
        # Check if already exists
        existing_path = self.get_model_path(model_name)
        if existing_path and not force:
            print(f"Model {model_name} already downloaded: {existing_path}")
            return existing_path
        
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}")
        
        config = self.MODEL_CONFIGS[model_name]
        
        print(f"Downloading {model_name} ({config.get('size', 'Unknown size')})...")
        
        if config["type"] == "dummy":
            # Dummy models don't need downloading
            return Path("dummy")
        elif config["type"] == "direct":
            return self._download_direct(model_name, config, progress_callback)
        elif config["type"] == "huggingface":
            return self._download_huggingface(model_name, config, progress_callback)
        elif config["type"] == "gdrive":
            return self._download_gdrive(model_name, config, progress_callback)
        else:
            raise ValueError(f"Unknown download type: {config['type']}")
    
    def _download_direct(
        self,
        model_name: str,
        config: Dict[str, Any],
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Path:
        """Download model directly from URL."""
        model_dir = self.cache_dir / "checkpoints" / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = model_dir / config["filename"]
        temp_path = output_path.with_suffix(".tmp")
        
        try:
            # Download with progress bar
            response = requests.get(config["url"], stream=True, allow_redirects=True)
            response.raise_for_status()
            
            # Get total size
            total_size = int(response.headers.get('content-length', 0))
            
            # Download with progress
            with open(temp_path, 'wb') as f:
                with tqdm(
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    desc=f"Downloading {model_name}"
                ) as pbar:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            pbar.update(len(chunk))
                            
                            if progress_callback:
                                progress_callback(downloaded / total_size)
            
            # Verify checksum if provided
            if config.get("sha256"):
                if not self._verify_checksum(temp_path, config["sha256"]):
                    raise ValueError("Checksum verification failed")
            
            # Move to final location
            temp_path.rename(output_path)
            
            # Update metadata
            self.metadata[model_name] = {
                "path": str(output_path),
                "downloaded_at": str(output_path.stat().st_ctime),
                "size": output_path.stat().st_size,
                "version": config.get("version", "unknown"),
            }
            self._save_metadata()
            
            print(f"Downloaded to: {output_path}")
            return output_path
            
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise RuntimeError(f"Download failed: {e}")
    
    def _download_huggingface(
        self,
        model_name: str,
        config: Dict[str, Any],
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Path:
        """Download model from HuggingFace Hub."""
        try:
            from huggingface_hub import hf_hub_download, snapshot_download
        except ImportError:
            raise ImportError(
                "huggingface-hub is required for this model. "
                "Install with: pip install huggingface-hub"
            )
        
        model_dir = self.cache_dir / "checkpoints" / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            if "filename" in config:
                # Download specific file
                local_path = hf_hub_download(
                    repo_id=config["repo_id"],
                    filename=config["filename"],
                    cache_dir=str(model_dir),
                    local_dir=str(model_dir),
                    local_dir_use_symlinks=False,
                )
                return Path(local_path)
            else:
                # Download entire repository
                local_dir = snapshot_download(
                    repo_id=config["repo_id"],
                    cache_dir=str(model_dir),
                    local_dir=str(model_dir),
                    local_dir_use_symlinks=False,
                )
                return Path(local_dir)
                
        except Exception as e:
            raise RuntimeError(f"HuggingFace download failed: {e}")
    
    def _download_gdrive(
        self,
        model_name: str,
        config: Dict[str, Any],
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Path:
        """Download model from Google Drive using gdown."""
        try:
            import gdown
        except ImportError:
            raise ImportError(
                "gdown is required for Google Drive downloads. "
                "Install with: pip install gdown"
            )
        
        model_dir = self.cache_dir / "checkpoints" / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = model_dir / config["filename"]
        
        try:
            # Download with gdown
            print(f"Downloading from Google Drive: {config['gdrive_id']}")
            success = gdown.download(
                id=config["gdrive_id"],
                output=str(output_path),
                quiet=False
            )
            
            if not success or not output_path.exists():
                raise RuntimeError("gdown download failed")
            
            # Verify checksum if provided
            if config.get("sha256"):
                if not self._verify_checksum(output_path, config["sha256"]):
                    raise ValueError("Checksum verification failed")
            
            # Update metadata
            self.metadata[model_name] = {
                "path": str(output_path),
                "downloaded_at": str(output_path.stat().st_ctime),
                "size": output_path.stat().st_size,
                "version": config.get("version", "unknown"),
            }
            self._save_metadata()
            
            print(f"Downloaded to: {output_path}")
            return output_path
            
        except Exception as e:
            if output_path.exists():
                output_path.unlink()
            raise RuntimeError(f"Google Drive download failed: {e}")
    
    def _verify_checksum(self, file_path: Path, expected_sha256: str) -> bool:
        """Verify file checksum."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest() == expected_sha256
    
    def list_downloaded_models(self) -> Dict[str, Dict[str, Any]]:
        """List all downloaded models."""
        downloaded = {}
        
        for model_name in self.MODEL_CONFIGS:
            path = self.get_model_path(model_name)
            if path:
                info = {
                    "path": str(path),
                    "size": format_bytes(path.stat().st_size),
                }
                if model_name in self.metadata:
                    info.update(self.metadata[model_name])
                downloaded[model_name] = info
        
        return downloaded
    
    def get_download_size(self, model_name: str) -> str:
        """Get download size for a model."""
        if model_name in self.MODEL_CONFIGS:
            return self.MODEL_CONFIGS[model_name].get("size", "Unknown")
        return "Unknown"
    
    def clear_cache(self, model_name: Optional[str] = None):
        """Clear cached models."""
        import shutil
        
        if model_name:
            # Clear specific model
            model_path = self.get_model_path(model_name)
            if model_path:
                # Remove the model directory
                model_dir = model_path.parent
                if model_dir.exists():
                    shutil.rmtree(model_dir)
                
                # Update metadata
                if model_name in self.metadata:
                    del self.metadata[model_name]
                    self._save_metadata()
                
                print(f"Cleared cache for {model_name}")
        else:
            # Clear all cache
            checkpoints_dir = self.cache_dir / "checkpoints"
            if checkpoints_dir.exists():
                shutil.rmtree(checkpoints_dir)
            
            self.metadata = {}
            self._save_metadata()
            
            print("Cleared all model cache")
    
    def verify_model(self, model_name: str) -> bool:
        """Verify that a model is properly downloaded."""
        path = self.get_model_path(model_name)
        if not path:
            return False
        
        # Check file exists and has reasonable size
        if not path.exists():
            return False
        
        # Minimum size check (at least 1MB)
        if path.stat().st_size < 1024 * 1024:
            return False
        
        # If checksum is available, verify it
        config = self.MODEL_CONFIGS.get(model_name, {})
        if config.get("sha256"):
            return self._verify_checksum(path, config["sha256"])
        
        return True


# Global instance
_model_manager = None


def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager