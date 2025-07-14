"""Download utilities for TSD-SR models."""

import os
import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Optional, Union
from urllib.parse import urlparse, parse_qs

import requests
from tqdm import tqdm
from huggingface_hub import snapshot_download

from ...utils.download_utils import (
    download_file,
    get_cache_dir,
    verify_checksum,
    extract_archive,
)


class TSDSRDownloader:
    """Handle downloading of TSD-SR model weights and dependencies."""
    
    # Model configurations with download info
    MODEL_CONFIGS = {
        "tsdsr": {
            "lora_weights": {
                "name": "TSD-SR LoRA weights",
                "gdrive_folder": "1XJY9Qxhz0mqjTtgDXr07oFy9eJr8jphI",
                "onedrive_folder": "EsQQ2LLXp7pHsYMBVubgcsYBvEQXMmcNXGnz695odCGByQ",
                "files": [
                    "transformer.safetensors",
                    "vae.safetensors",
                ],
                "subdir": "checkpoint/tsdsr",
            },
            "prompt_embeddings": {
                "name": "Default prompt embeddings",
                "gdrive_folder": "1XJY9Qxhz0mqjTtgDXr07oFy9eJr8jphI",
                "onedrive_folder": "EsQQ2LLXp7pHsYMBVubgcsYBvEQXMmcNXGnz695odCGByQ",
                "files": [
                    "prompt_embeds.pt",
                    "pool_embeds.pt",
                ],
                "subdir": "dataset/default",
            },
        },
        "tsdsr-mse": {
            "lora_weights": {
                "name": "TSD-SR MSE LoRA weights",
                "gdrive_folder": "1XJY9Qxhz0mqjTtgDXr07oFy9eJr8jphI",
                "onedrive_folder": "EsQQ2LLXp7pHsYMBVubgcsYBvEQXMmcNXGnz695odCGByQ",
                "files": [
                    "transformer.safetensors",
                    "vae.safetensors",
                ],
                "subdir": "checkpoint/tsdsr-mse",
            },
            "prompt_embeddings": {
                # Same as tsdsr
                "name": "Default prompt embeddings",
                "gdrive_folder": "1XJY9Qxhz0mqjTtgDXr07oFy9eJr8jphI",
                "onedrive_folder": "EsQQ2LLXp7pHsYMBVubgcsYBvEQXMmcNXGnz695odCGByQ",
                "files": [
                    "prompt_embeds.pt",
                    "pool_embeds.pt",
                ],
                "subdir": "dataset/default",
            },
        },
        "tsdsr-gan": {
            "lora_weights": {
                "name": "TSD-SR GAN LoRA weights",
                "gdrive_folder": "1XJY9Qxhz0mqjTtgDXr07oFy9eJr8jphI",
                "onedrive_folder": "EsQQ2LLXp7pHsYMBVubgcsYBvEQXMmcNXGnz695odCGByQ",
                "files": [
                    "transformer.safetensors",
                    "vae.safetensors",
                ],
                "subdir": "checkpoint/tsdsr-gan",
            },
            "prompt_embeddings": {
                # Same as tsdsr
                "name": "Default prompt embeddings",
                "gdrive_folder": "1XJY9Qxhz0mqjTtgDXr07oFy9eJr8jphI",
                "onedrive_folder": "EsQQ2LLXp7pHsYMBVubgcsYBvEQXMmcNXGnz695odCGByQ",
                "files": [
                    "prompt_embeds.pt",
                    "pool_embeds.pt",
                ],
                "subdir": "dataset/default",
            },
        },
    }
    
    # SD3 model info
    SD3_MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"
    
    # Additional downloads
    TEACHER_LORA_URL = "https://drive.google.com/file/d/1do8pfdm_oNUhJKxTlC_x7LqY7NlE0-Q7/view?usp=sharing"
    NULL_EMBEDDINGS_FOLDER = "1_kSod1CCq_xwdwDnLYUhr7iaT70eFPBD"
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize downloader."""
        self.cache_dir = Path(cache_dir) if cache_dir else get_cache_dir() / "tsdsr"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def download_model(
        self,
        model_name: str,
        include_sd3: bool = True,
        include_teacher: bool = False,
        force: bool = False,
        use_onedrive: bool = False,
    ) -> Dict[str, Path]:
        """Download TSD-SR model weights and dependencies.
        
        Args:
            model_name: Model variant (tsdsr, tsdsr-mse, tsdsr-gan)
            include_sd3: Whether to download SD3 base model
            include_teacher: Whether to download teacher LoRA weights
            force: Force re-download even if files exist
            use_onedrive: Use OneDrive instead of Google Drive
            
        Returns:
            Dictionary with paths to downloaded components
        """
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}")
            
        paths = {}
        
        # Download LoRA weights
        print(f"\nDownloading {model_name} weights...")
        lora_config = self.MODEL_CONFIGS[model_name]["lora_weights"]
        lora_path = self._download_from_cloud(
            lora_config,
            use_onedrive=use_onedrive,
            force=force
        )
        paths["lora_weights"] = lora_path
        
        # Download prompt embeddings
        print(f"\nDownloading prompt embeddings...")
        prompt_config = self.MODEL_CONFIGS[model_name]["prompt_embeddings"]
        prompt_path = self._download_from_cloud(
            prompt_config,
            use_onedrive=use_onedrive,
            force=force
        )
        paths["prompt_embeddings"] = prompt_path
        
        # Download SD3 base model if requested
        if include_sd3:
            print(f"\nDownloading Stable Diffusion 3 base model...")
            sd3_path = self._download_sd3(force=force)
            paths["sd3_model"] = sd3_path
            
        # Download teacher LoRA if requested
        if include_teacher:
            print(f"\nDownloading teacher LoRA weights...")
            teacher_path = self._download_teacher_lora(force=force)
            paths["teacher_lora"] = teacher_path
            
            print(f"\nDownloading null prompt embeddings...")
            null_path = self._download_null_embeddings(
                use_onedrive=use_onedrive,
                force=force
            )
            paths["null_embeddings"] = null_path
            
        return paths
        
    def _download_from_cloud(
        self,
        config: Dict,
        use_onedrive: bool = False,
        force: bool = False
    ) -> Path:
        """Download files from Google Drive or OneDrive."""
        # Determine destination path
        dest_dir = self.cache_dir / config["subdir"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if files already exist
        all_exist = all((dest_dir / f).exists() for f in config["files"])
        if all_exist and not force:
            print(f"  {config['name']} already downloaded")
            return dest_dir
            
        # Show download instructions
        print(f"\n📥 Downloading {config['name']}...")
        print(f"  Files: {', '.join(config['files'])}")
        
        if use_onedrive:
            folder_id = config["onedrive_folder"]
            url = f"https://1drv.ms/f/c/d75249b59f444489/{folder_id}"
            print(f"\n  OneDrive URL: {url}")
        else:
            folder_id = config["gdrive_folder"]
            url = f"https://drive.google.com/drive/folders/{folder_id}"
            print(f"\n  Google Drive URL: {url}")
            
        print("\n  ⚠️  Due to cloud storage restrictions, please:")
        print(f"  1. Visit the URL above")
        print(f"  2. Download the following files:")
        for f in config["files"]:
            print(f"     - {f}")
        print(f"  3. Place them in: {dest_dir}")
        
        # Wait for user confirmation
        input("\n  Press Enter after downloading the files...")
        
        # Verify files exist
        missing = []
        for f in config["files"]:
            if not (dest_dir / f).exists():
                missing.append(f)
                
        if missing:
            raise FileNotFoundError(
                f"Missing files in {dest_dir}: {', '.join(missing)}"
            )
            
        print(f"  ✓ {config['name']} ready")
        return dest_dir
        
    def _download_sd3(self, force: bool = False) -> Path:
        """Download SD3 base model from HuggingFace."""
        sd3_dir = self.cache_dir / "sd3-medium"
        
        # Check if already downloaded
        if sd3_dir.exists() and not force:
            # Check for key files
            key_files = ["transformer/diffusion_pytorch_model.safetensors", "vae/diffusion_pytorch_model.safetensors"]
            if all((sd3_dir / f).exists() for f in key_files):
                print("  SD3 model already downloaded")
                return sd3_dir
                
        print(f"  Downloading from HuggingFace: {self.SD3_MODEL_ID}")
        print("  This may take a while (~14GB)...")
        
        try:
            # Download from HuggingFace
            snapshot_download(
                repo_id=self.SD3_MODEL_ID,
                local_dir=sd3_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            print("  ✓ SD3 model downloaded successfully")
            
        except Exception as e:
            print(f"\n  ❌ Failed to download SD3: {e}")
            print("\n  Alternative: Download manually from:")
            print(f"  https://huggingface.co/{self.SD3_MODEL_ID}")
            print(f"  Place files in: {sd3_dir}")
            raise
            
        return sd3_dir
        
    def _download_teacher_lora(self, force: bool = False) -> Path:
        """Download teacher LoRA weights."""
        dest_dir = self.cache_dir / "checkpoint" / "teacher"
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract file ID from Google Drive URL
        file_id = "1do8pfdm_oNUhJKxTlC_x7LqY7NlE0-Q7"
        
        print(f"\n  📥 Downloading teacher LoRA weights...")
        print(f"  Google Drive URL: {self.TEACHER_LORA_URL}")
        print("\n  ⚠️  Please:")
        print(f"  1. Visit the URL above")
        print(f"  2. Download the file")
        print(f"  3. Place it in: {dest_dir}")
        
        input("\n  Press Enter after downloading...")
        
        # Check if file exists
        if not any(dest_dir.iterdir()):
            raise FileNotFoundError(f"No files found in {dest_dir}")
            
        print("  ✓ Teacher LoRA weights ready")
        return dest_dir
        
    def _download_null_embeddings(
        self,
        use_onedrive: bool = False,
        force: bool = False
    ) -> Path:
        """Download null prompt embeddings."""
        dest_dir = self.cache_dir / "dataset" / "null"
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        if dest_dir.exists() and any(dest_dir.iterdir()) and not force:
            print("  Null embeddings already downloaded")
            return dest_dir
            
        folder_id = self.NULL_EMBEDDINGS_FOLDER
        url = f"https://drive.google.com/drive/folders/{folder_id}"
        
        print(f"\n  📥 Downloading null prompt embeddings...")
        print(f"  Google Drive URL: {url}")
        print("\n  ⚠️  Please download all files from the folder")
        print(f"  Place them in: {dest_dir}")
        
        input("\n  Press Enter after downloading...")
        
        if not any(dest_dir.iterdir()):
            raise FileNotFoundError(f"No files found in {dest_dir}")
            
        print("  ✓ Null embeddings ready")
        return dest_dir
        
    def get_download_instructions(self, model_name: str) -> str:
        """Get manual download instructions."""
        if model_name not in self.MODEL_CONFIGS:
            return f"Unknown model: {model_name}"
            
        instructions = f"""
TSD-SR Model Download Instructions
==================================

Model: {model_name}

1. SD3 Base Model (Required):
   - URL: https://huggingface.co/{self.SD3_MODEL_ID}
   - Size: ~14GB
   - Download all files to: {self.cache_dir}/sd3-medium/

2. TSD-SR LoRA Weights:
   - Google Drive: https://drive.google.com/drive/folders/{self.MODEL_CONFIGS[model_name]['lora_weights']['gdrive_folder']}
   - OneDrive: https://1drv.ms/f/c/d75249b59f444489/{self.MODEL_CONFIGS[model_name]['lora_weights']['onedrive_folder']}
   - Files: {', '.join(self.MODEL_CONFIGS[model_name]['lora_weights']['files'])}
   - Download to: {self.cache_dir}/{self.MODEL_CONFIGS[model_name]['lora_weights']['subdir']}/

3. Prompt Embeddings:
   - Use same links as LoRA weights
   - Files: {', '.join(self.MODEL_CONFIGS[model_name]['prompt_embeddings']['files'])}
   - Download to: {self.cache_dir}/{self.MODEL_CONFIGS[model_name]['prompt_embeddings']['subdir']}/

Optional:
---------
4. Teacher LoRA (for training):
   - URL: {self.TEACHER_LORA_URL}
   - Download to: {self.cache_dir}/checkpoint/teacher/

5. Null Embeddings (for training):
   - URL: https://drive.google.com/drive/folders/{self.NULL_EMBEDDINGS_FOLDER}
   - Download all files to: {self.cache_dir}/dataset/null/
"""
        return instructions


def download_tsdsr_model(
    model_name: str = "tsdsr",
    cache_dir: Optional[str] = None,
    include_sd3: bool = True,
    force: bool = False,
) -> Dict[str, Path]:
    """Download TSD-SR model weights.
    
    Args:
        model_name: Model variant to download
        cache_dir: Custom cache directory
        include_sd3: Whether to download SD3 base model
        force: Force re-download
        
    Returns:
        Dictionary with paths to downloaded components
    """
    downloader = TSDSRDownloader(cache_dir)
    return downloader.download_model(
        model_name,
        include_sd3=include_sd3,
        force=force
    )


def print_download_instructions(model_name: str = "tsdsr"):
    """Print manual download instructions."""
    downloader = TSDSRDownloader()
    print(downloader.get_download_instructions(model_name))


def main():
    """Main entry point for command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download TSD-SR models")
    parser.add_argument(
        "model",
        choices=["tsdsr", "tsdsr-mse", "tsdsr-gan"],
        help="Model variant to download"
    )
    parser.add_argument(
        "--no-sd3",
        action="store_true",
        help="Skip downloading SD3 base model"
    )
    parser.add_argument(
        "--instructions-only",
        action="store_true",
        help="Print download instructions only"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download"
    )
    
    args = parser.parse_args()
    
    if args.instructions_only:
        print_download_instructions(args.model)
    else:
        paths = download_tsdsr_model(
            args.model,
            include_sd3=not args.no_sd3,
            force=args.force
        )
        print("\nDownloaded files:")
        for name, path in paths.items():
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()