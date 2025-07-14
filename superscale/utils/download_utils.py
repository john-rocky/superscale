"""Download utilities for superscale models."""

import hashlib
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Tuple, Dict
from urllib.parse import urlparse

import requests
from tqdm import tqdm


def get_cache_dir() -> Path:
    """Get the default cache directory for superscale models."""
    # Check environment variable first
    cache_dir = os.environ.get("SUPERSCALE_CACHE_DIR")
    
    if cache_dir:
        return Path(cache_dir)
    
    # Default to ~/.cache/superscale
    home = Path.home()
    default_cache = home / ".cache" / "superscale"
    default_cache.mkdir(parents=True, exist_ok=True)
    
    return default_cache


def download_file(
    url: str,
    dest_path: Path,
    chunk_size: int = 8192,
    resume: bool = True,
    headers: Optional[Dict[str, str]] = None,
) -> Path:
    """Download a file with progress bar and resume support.
    
    Args:
        url: URL to download from
        dest_path: Destination file path
        chunk_size: Download chunk size
        resume: Whether to resume partial downloads
        headers: Additional headers for the request
        
    Returns:
        Path to downloaded file
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file already exists
    if dest_path.exists() and not resume:
        return dest_path
    
    # Setup headers
    if headers is None:
        headers = {}
        
    # Handle resume
    mode = "wb"
    resume_pos = 0
    
    if resume and dest_path.exists():
        resume_pos = dest_path.stat().st_size
        headers["Range"] = f"bytes={resume_pos}-"
        mode = "ab"
    
    # Make request
    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()
    
    # Get total size
    total_size = int(response.headers.get("content-length", 0))
    if resume and resume_pos > 0:
        total_size += resume_pos
    
    # Download with progress bar
    with open(dest_path, mode) as f:
        with tqdm(
            total=total_size,
            initial=resume_pos,
            unit="B",
            unit_scale=True,
            desc=dest_path.name,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    return dest_path


def verify_checksum(
    file_path: Path,
    expected_checksum: str,
    algorithm: str = "sha256"
) -> bool:
    """Verify file checksum.
    
    Args:
        file_path: Path to file
        expected_checksum: Expected checksum value
        algorithm: Hash algorithm (sha256, md5, etc.)
        
    Returns:
        True if checksum matches
    """
    hash_func = getattr(hashlib, algorithm)()
    
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    
    actual_checksum = hash_func.hexdigest()
    return actual_checksum.lower() == expected_checksum.lower()


def extract_archive(
    archive_path: Path,
    extract_to: Path,
    remove_archive: bool = False
) -> Path:
    """Extract archive file.
    
    Args:
        archive_path: Path to archive file
        extract_to: Directory to extract to
        remove_archive: Whether to remove archive after extraction
        
    Returns:
        Path to extracted directory
    """
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)
    
    # Determine archive type and extract
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.suffix in [".tar", ".gz", ".tgz", ".bz2"]:
        import tarfile
        with tarfile.open(archive_path, "r:*") as tar_ref:
            tar_ref.extractall(extract_to)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.suffix}")
    
    # Remove archive if requested
    if remove_archive:
        archive_path.unlink()
    
    return extract_to


def download_from_google_drive(
    file_id: str,
    dest_path: Path,
    quiet: bool = False
) -> Path:
    """Download file from Google Drive.
    
    Note: This is a placeholder. For large files or files requiring
    authentication, manual download may be required.
    
    Args:
        file_id: Google Drive file ID
        dest_path: Destination path
        quiet: Suppress output
        
    Returns:
        Path to downloaded file
    """
    base_url = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    
    response = session.get(base_url, params={"id": file_id}, stream=True)
    token = _get_confirm_token(response)
    
    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(base_url, params=params, stream=True)
    
    _save_response_content(response, dest_path, quiet)
    return dest_path


def _get_confirm_token(response):
    """Extract confirmation token from Google Drive response."""
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def _save_response_content(response, destination, quiet=False):
    """Save response content with progress bar."""
    CHUNK_SIZE = 32768
    
    with open(destination, "wb") as f:
        if not quiet:
            total = int(response.headers.get("content-length", 0))
            with tqdm(total=total, unit="B", unit_scale=True) as pbar:
                for chunk in response.iter_content(CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        else:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)


def safe_download(
    url: str,
    dest_dir: Path,
    filename: Optional[str] = None,
    checksum: Optional[str] = None,
    extract: bool = False,
) -> Path:
    """Safely download and optionally extract a file.
    
    Args:
        url: URL to download
        dest_dir: Destination directory
        filename: Optional filename (extracted from URL if not provided)
        checksum: Optional checksum to verify
        extract: Whether to extract if archive
        
    Returns:
        Path to downloaded (and possibly extracted) content
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine filename
    if filename is None:
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        if not filename:
            filename = "download"
    
    dest_path = dest_dir / filename
    
    # Download file
    download_file(url, dest_path)
    
    # Verify checksum if provided
    if checksum:
        if not verify_checksum(dest_path, checksum):
            dest_path.unlink()
            raise ValueError(f"Checksum verification failed for {filename}")
    
    # Extract if requested and is archive
    if extract and dest_path.suffix in [".zip", ".tar", ".gz", ".tgz", ".bz2"]:
        extract_dir = dest_dir / dest_path.stem
        extract_archive(dest_path, extract_dir, remove_archive=True)
        return extract_dir
    
    return dest_path