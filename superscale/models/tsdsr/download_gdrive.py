"""Google Drive direct download implementation for TSD-SR."""

import os
import hashlib
from pathlib import Path
from typing import Dict, Optional
import requests
from tqdm import tqdm


class TSDSRGDriveDownloader:
    """Download TSD-SR from Google Drive with file IDs."""
    
    # Import configuration
    try:
        from .gdrive_config import TSDSR_FILE_IDS, FILE_CHECKSUMS
        FILE_IDS = TSDSR_FILE_IDS
        FILE_HASHES = FILE_CHECKSUMS
    except ImportError:
        # Default configuration (requires manual setup)
        FILE_IDS = {
            "tsdsr": {
                "transformer.safetensors": "YOUR_TRANSFORMER_FILE_ID",
                "vae.safetensors": "YOUR_VAE_FILE_ID",
                "prompt_embeds.pt": "YOUR_PROMPT_EMBEDS_FILE_ID",
                "pool_embeds.pt": "YOUR_POOL_EMBEDS_FILE_ID",
            },
            "tsdsr-mse": {
                "transformer.safetensors": "YOUR_MSE_TRANSFORMER_FILE_ID",
                "vae.safetensors": "YOUR_MSE_VAE_FILE_ID",
                "prompt_embeds.pt": "YOUR_PROMPT_EMBEDS_FILE_ID",
                "pool_embeds.pt": "YOUR_POOL_EMBEDS_FILE_ID",
            },
            "tsdsr-gan": {
                "transformer.safetensors": "YOUR_GAN_TRANSFORMER_FILE_ID",
                "vae.safetensors": "YOUR_GAN_VAE_FILE_ID",
                "prompt_embeds.pt": "YOUR_PROMPT_EMBEDS_FILE_ID",
                "pool_embeds.pt": "YOUR_POOL_EMBEDS_FILE_ID",
            },
        }
        FILE_HASHES = {}
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize downloader."""
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "superscale" / "tsdsr"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def download_file_from_gdrive(self, file_id: str, dest_path: Path) -> Path:
        """Download a file from Google Drive using its file ID."""
        if file_id.startswith("YOUR_"):
            raise ValueError(f"Please replace {file_id} with actual Google Drive file ID")
            
        url = "https://drive.google.com/uc?export=download"
        session = requests.Session()
        
        # First request
        response = session.get(url, params={"id": file_id}, stream=True)
        
        # Check for virus scan warning
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                # Bypass virus scan warning
                params = {"id": file_id, "confirm": value}
                response = session.get(url, params=params, stream=True)
                break
        
        # Get file size
        total_size = int(response.headers.get("content-length", 0))
        
        # Download with progress bar
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(dest_path, "wb") as f:
            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=dest_path.name,
                ncols=80
            ) as pbar:
                for chunk in response.iter_content(chunk_size=32768):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        # Verify checksum if available
        if dest_path.name in self.FILE_HASHES:
            expected_hash = self.FILE_HASHES[dest_path.name]
            actual_hash = self._calculate_md5(dest_path)
            if actual_hash != expected_hash:
                dest_path.unlink()
                raise ValueError(f"Checksum mismatch for {dest_path.name}")
        
        return dest_path
        
    def _calculate_md5(self, file_path: Path) -> str:
        """Calculate MD5 hash of a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
        
    def download_model(self, model_name: str, force: bool = False) -> Dict[str, Path]:
        """Download all files for a model."""
        if model_name not in self.FILE_IDS:
            raise ValueError(f"Unknown model: {model_name}")
            
        print(f"\n🚀 Downloading {model_name} from Google Drive...")
        
        downloaded = {}
        file_ids = self.FILE_IDS[model_name]
        
        # Determine paths
        model_dir = self.cache_dir / "checkpoint" / model_name
        embeddings_dir = self.cache_dir / "dataset" / "default"
        
        for filename, file_id in file_ids.items():
            # Determine destination
            if filename.endswith(".safetensors"):
                dest_path = model_dir / filename
            else:
                dest_path = embeddings_dir / filename
                
            # Check if already exists
            if dest_path.exists() and not force:
                print(f"✓ {filename} already exists")
                downloaded[filename] = dest_path
                continue
                
            # Download
            print(f"\n📥 Downloading {filename}...")
            try:
                self.download_file_from_gdrive(file_id, dest_path)
                print(f"✓ {filename} downloaded successfully")
                downloaded[filename] = dest_path
            except Exception as e:
                print(f"❌ Failed to download {filename}: {e}")
                
        return downloaded
        
    def set_file_ids(self, model_name: str, file_ids: Dict[str, str]):
        """Set file IDs for a model (for configuration)."""
        if model_name not in self.FILE_IDS:
            self.FILE_IDS[model_name] = {}
        self.FILE_IDS[model_name].update(file_ids)


def download_tsdsr_gdrive(
    model_name: str = "tsdsr",
    file_ids: Optional[Dict[str, str]] = None,
    cache_dir: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Path]:
    """Download TSD-SR model from Google Drive.
    
    Args:
        model_name: Model variant (tsdsr, tsdsr-mse, tsdsr-gan)
        file_ids: Optional dict of filename -> file_id mappings
        cache_dir: Custom cache directory
        force: Force re-download
        
    Returns:
        Dict of filename -> Path mappings
        
    Example:
        # With your file IDs
        file_ids = {
            "transformer.safetensors": "1AbC123...",
            "vae.safetensors": "1DeF456...",
            "prompt_embeds.pt": "1GhI789...",
            "pool_embeds.pt": "1JkL012...",
        }
        paths = download_tsdsr_gdrive("tsdsr", file_ids=file_ids)
    """
    downloader = TSDSRGDriveDownloader(cache_dir)
    
    # Set custom file IDs if provided
    if file_ids:
        downloader.set_file_ids(model_name, file_ids)
    
    return downloader.download_model(model_name, force=force)


# コマンドライン用
def main():
    """Command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download TSD-SR from Google Drive")
    parser.add_argument("model", choices=["tsdsr", "tsdsr-mse", "tsdsr-gan"])
    parser.add_argument("--force", action="store_true", help="Force re-download")
    
    # File ID arguments
    parser.add_argument("--transformer-id", help="Google Drive file ID for transformer.safetensors")
    parser.add_argument("--vae-id", help="Google Drive file ID for vae.safetensors")
    parser.add_argument("--prompt-embeds-id", help="Google Drive file ID for prompt_embeds.pt")
    parser.add_argument("--pool-embeds-id", help="Google Drive file ID for pool_embeds.pt")
    
    args = parser.parse_args()
    
    # Build file IDs dict if provided
    file_ids = {}
    if args.transformer_id:
        file_ids["transformer.safetensors"] = args.transformer_id
    if args.vae_id:
        file_ids["vae.safetensors"] = args.vae_id
    if args.prompt_embeds_id:
        file_ids["prompt_embeds.pt"] = args.prompt_embeds_id
    if args.pool_embeds_id:
        file_ids["pool_embeds.pt"] = args.pool_embeds_id
    
    try:
        paths = download_tsdsr_gdrive(
            args.model,
            file_ids=file_ids if file_ids else None,
            force=args.force
        )
        
        print("\n✨ Download complete!")
        print("\nDownloaded files:")
        for filename, path in paths.items():
            print(f"  {filename}: {path}")
            
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease provide Google Drive file IDs:")
        print("  python -m superscale.models.tsdsr.download_gdrive tsdsr \\")
        print("    --transformer-id YOUR_FILE_ID \\")
        print("    --vae-id YOUR_FILE_ID \\")
        print("    --prompt-embeds-id YOUR_FILE_ID \\")
        print("    --pool-embeds-id YOUR_FILE_ID")
        return 1
        
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())