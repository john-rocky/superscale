"""Automatic download implementation for TSD-SR (with workarounds)."""

import os
import re
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs

import requests
from tqdm import tqdm
from huggingface_hub import hf_hub_download, snapshot_download


class TSDSRAutoDownloader:
    """Fully automatic downloader for TSD-SR models."""
    
    # Known direct download links (if available)
    DIRECT_DOWNLOAD_URLS = {
        "tsdsr": {
            "transformer.safetensors": {
                # These would be direct download URLs if available
                # Format: "url": "https://drive.google.com/uc?export=download&id=FILE_ID",
                # "md5": "expected_md5_hash"
            },
            "vae.safetensors": {
                # Direct URL would go here
            },
            "prompt_embeds.pt": {
                # Direct URL would go here  
            },
            "pool_embeds.pt": {
                # Direct URL would go here
            },
        },
    }
    
    # Alternative: Host on HuggingFace
    HUGGINGFACE_REPOS = {
        "tsdsr": "username/tsdsr-weights",  # If weights were on HF
        "tsdsr-mse": "username/tsdsr-mse-weights",
        "tsdsr-gan": "username/tsdsr-gan-weights",
    }
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize auto downloader."""
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "superscale" / "tsdsr"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def download_from_gdrive_file(self, file_id: str, dest_path: Path) -> Path:
        """Download a single file from Google Drive using file ID.
        
        This works for individual files if we know their IDs.
        """
        url = "https://drive.google.com/uc"
        session = requests.Session()
        
        # First request
        response = session.get(url, params={"id": file_id, "export": "download"}, stream=True)
        
        # Check for virus scan warning
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                # Bypass virus scan warning
                params = {"id": file_id, "export": "download", "confirm": value}
                response = session.get(url, params=params, stream=True)
                break
        
        # Download with progress bar
        total_size = int(response.headers.get("content-length", 0))
        
        with open(dest_path, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True, desc=dest_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        return dest_path
        
    def extract_gdrive_folder_links(self, folder_url: str) -> Dict[str, str]:
        """Attempt to extract file links from Google Drive folder.
        
        Note: This is a workaround and may not always work.
        """
        # This would require web scraping or using Google Drive API with authentication
        # For now, we return empty dict
        return {}
        
    def download_from_mirror(self, model_name: str) -> Dict[str, Path]:
        """Download from mirror servers if available."""
        # Option 1: Use a mirror server maintained by the community
        mirror_base = os.environ.get("TSDSR_MIRROR_URL", "https://mirror.example.com/tsdsr")
        
        files_to_download = {
            "transformer.safetensors": f"{mirror_base}/{model_name}/transformer.safetensors",
            "vae.safetensors": f"{mirror_base}/{model_name}/vae.safetensors",
            "prompt_embeds.pt": f"{mirror_base}/embeddings/prompt_embeds.pt",
            "pool_embeds.pt": f"{mirror_base}/embeddings/pool_embeds.pt",
        }
        
        downloaded = {}
        for filename, url in files_to_download.items():
            dest_path = self.cache_dir / model_name / filename
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                # Download file
                total_size = int(response.headers.get("content-length", 0))
                with open(dest_path, "wb") as f:
                    with tqdm(total=total_size, unit="B", unit_scale=True, desc=filename) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                
                downloaded[filename] = dest_path
                
            except requests.RequestException as e:
                print(f"Failed to download {filename} from mirror: {e}")
                
        return downloaded
        
    def download_from_huggingface(self, model_name: str) -> Dict[str, Path]:
        """Download from HuggingFace if weights are hosted there."""
        if model_name not in self.HUGGINGFACE_REPOS:
            raise ValueError(f"No HuggingFace repo configured for {model_name}")
            
        repo_id = self.HUGGINGFACE_REPOS[model_name]
        local_dir = self.cache_dir / model_name
        
        try:
            # Download entire repository
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            
            # Map files
            downloaded = {}
            for filename in ["transformer.safetensors", "vae.safetensors", "prompt_embeds.pt", "pool_embeds.pt"]:
                file_path = local_dir / filename
                if file_path.exists():
                    downloaded[filename] = file_path
                    
            return downloaded
            
        except Exception as e:
            print(f"Failed to download from HuggingFace: {e}")
            return {}
            
    def create_download_script(self, model_name: str) -> Path:
        """Create a shell script for manual download."""
        script_path = self.cache_dir / f"download_{model_name}.sh"
        
        script_content = f"""#!/bin/bash
# Auto-generated download script for {model_name}

echo "TSD-SR Model Download Script"
echo "============================"
echo "This script will help you download the required files."
echo ""

# Create directories
mkdir -p {self.cache_dir}/{model_name}
mkdir -p {self.cache_dir}/embeddings

echo "Please download the following files:"
echo ""
echo "1. From Google Drive folder:"
echo "   https://drive.google.com/drive/folders/1XJY9Qxhz0mqjTtgDXr07oFy9eJr8jphI"
echo ""
echo "   - transformer.safetensors -> {self.cache_dir}/{model_name}/transformer.safetensors"
echo "   - vae.safetensors -> {self.cache_dir}/{model_name}/vae.safetensors"
echo "   - prompt_embeds.pt -> {self.cache_dir}/embeddings/prompt_embeds.pt"
echo "   - pool_embeds.pt -> {self.cache_dir}/embeddings/pool_embeds.pt"
echo ""

# Alternative: Use gdown if available
if command -v gdown &> /dev/null; then
    echo "Found gdown! Attempting automatic download..."
    # Note: These file IDs would need to be extracted from the folder
    # gdown --id FILE_ID -O {self.cache_dir}/{model_name}/transformer.safetensors
    # gdown --id FILE_ID -O {self.cache_dir}/{model_name}/vae.safetensors
    echo "gdown not configured with file IDs"
fi

# Alternative: Use rclone if configured
if command -v rclone &> /dev/null; then
    echo "Found rclone! You can use:"
    echo "rclone copy 'gdrive:path/to/folder' {self.cache_dir}/{model_name}/"
fi

echo ""
echo "After downloading, verify files exist:"
ls -la {self.cache_dir}/{model_name}/
ls -la {self.cache_dir}/embeddings/
"""
        
        script_path.write_text(script_content)
        script_path.chmod(0o755)
        
        return script_path
        
    def try_all_methods(self, model_name: str) -> Dict[str, Path]:
        """Try all available download methods."""
        print(f"Attempting to download {model_name}...")
        
        # Method 1: Try mirror server
        print("Trying mirror server...")
        downloaded = self.download_from_mirror(model_name)
        if len(downloaded) == 4:  # All files downloaded
            return downloaded
            
        # Method 2: Try HuggingFace
        print("Trying HuggingFace...")
        downloaded = self.download_from_huggingface(model_name)
        if len(downloaded) == 4:
            return downloaded
            
        # Method 3: Try known direct URLs
        if model_name in self.DIRECT_DOWNLOAD_URLS:
            print("Trying direct URLs...")
            downloaded = {}
            for filename, info in self.DIRECT_DOWNLOAD_URLS[model_name].items():
                if "url" in info:
                    # Download from direct URL
                    pass
                    
        # Method 4: Create download script
        print("Creating download script...")
        script_path = self.create_download_script(model_name)
        print(f"\nAutomatic download failed. Please run: {script_path}")
        
        return downloaded


# Integration with existing code
def download_tsdsr_auto(model_name: str = "tsdsr") -> Dict[str, Path]:
    """Attempt fully automatic download of TSD-SR models."""
    downloader = TSDSRAutoDownloader()
    
    # First, download SD3 from HuggingFace (this works)
    print("Downloading SD3 base model from HuggingFace...")
    sd3_path = downloader.cache_dir / "sd3-medium"
    if not sd3_path.exists():
        snapshot_download(
            repo_id="stabilityai/stable-diffusion-3-medium-diffusers",
            local_dir=sd3_path,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
    
    # Then try to download TSD-SR weights
    downloaded_files = downloader.try_all_methods(model_name)
    
    return {
        "sd3_model": sd3_path,
        "lora_weights": downloader.cache_dir / model_name,
        **downloaded_files
    }


if __name__ == "__main__":
    # Test automatic download
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "tsdsr"
    paths = download_tsdsr_auto(model)
    print("\nDownload summary:")
    for key, path in paths.items():
        print(f"  {key}: {path}")