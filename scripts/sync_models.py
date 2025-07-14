#!/usr/bin/env python3
"""Sync model implementations from third_party to backends/native."""

import argparse
import ast
import re
import shutil
from pathlib import Path
from typing import Dict, List, Set


class ModelSynchronizer:
    """Sync minimal code from submodules to backends."""
    
    # Files to copy for each model
    SYNC_PATTERNS = {
        "hitsr": {
            "include": [
                # Core model files
                "basicsr/archs/hit_arch.py",
                "basicsr/archs/hit_sir_arch.py",
                "basicsr/archs/hit_sng_arch.py",
                "basicsr/archs/hit_srf_arch.py",
                "basicsr/models/hit_sr_model.py",
                # Essential utilities
                "basicsr/utils/registry.py",
                "basicsr/utils/logger.py",
                "basicsr/utils/misc.py",
                # Config files
                "options/test/HiT-*/test_Hit_*.yml",
            ],
            "exclude": [
                "*_test.py",
                "**/test_*.py",
                "*.md",
                "__pycache__",
                ".git",
                "experiments/",
                "datasets/",
            ],
            "rename": {
                "basicsr/archs": "models",
                "basicsr/models": "trainers",
                "basicsr/utils": "utils",
                "options": "configs",
            },
        },
        "tsdsr": {
            "include": [
                # Core model files
                "models/autoencoder_kl.py",
                # Essential utils
                "utils/util.py",
                "utils/vaehook.py",
                "utils/wavelet_color_fix.py",
                "utils/device.py",
                # Test scripts for reference
                "test/test_tsdsr.py",
                # Configs
                "config/*.yaml",
            ],
            "exclude": [
                "train_*.py",
                "benchmark_*.py",
                "**/train/**",
                "*.ipynb",
                "basicsr/**",
                "data/**",
            ],
            "rename": {
                "models": "models",
                "utils": "utils",
                "config": "configs",
                "test": "examples",
            },
        },
        "varsr": {
            "include": [
                # Core model files
                "models/var.py",
                "models/vqvae.py",
                "models/basic_var.py",
                # Essential utils
                "utils/amp_opt.py",
                "utils/data_utils.py",
                # RoPE implementation
                "models/rope_mixed.py",
                # Config
                "cfgs/*.yaml",
            ],
            "exclude": [
                "*_test.py",
                "demo_*.py",
                "train_*.py",
                "**/test/**",
            ],
            "rename": {
                "models": "models",
                "utils": "utils",
                "cfgs": "configs",
            },
        },
    }
    
    def __init__(self, project_root: Path):
        """Initialize synchronizer."""
        self.project_root = project_root
        self.third_party_dir = project_root / "third_party"
        self.backends_dir = project_root / "superscale" / "backends" / "native"
    
    def sync_model(self, model_name: str, force: bool = False) -> None:
        """Sync one model from third_party to backends/native."""
        model_lower = model_name.lower().replace("-", "")
        
        if model_lower not in self.SYNC_PATTERNS:
            raise ValueError(f"Unknown model: {model_name}")
        
        print(f"Syncing {model_name}...")
        
        src_dir = self.third_party_dir / model_name
        dst_dir = self.backends_dir / model_lower.replace("-", "")
        
        if not src_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {src_dir}")
        
        # Clear destination if force
        if dst_dir.exists() and force:
            shutil.rmtree(dst_dir)
        
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy files based on patterns
        patterns = self.SYNC_PATTERNS[model_lower]
        copied_files = self._copy_files(src_dir, dst_dir, patterns)
        
        # Minimize imports
        self._minimize_imports(dst_dir)
        
        # Create __init__.py files
        self._create_init_files(dst_dir)
        
        # Copy license
        self._copy_license(src_dir, dst_dir)
        
        print(f"  Copied {len(copied_files)} files")
    
    def _copy_files(
        self,
        src_dir: Path,
        dst_dir: Path,
        patterns: Dict[str, any]
    ) -> List[Path]:
        """Copy files matching patterns."""
        copied = []
        
        for pattern in patterns["include"]:
            # Handle glob patterns
            if "*" in pattern:
                for src_file in src_dir.glob(pattern):
                    if src_file.is_file():
                        if not self._should_exclude(src_file, patterns["exclude"]):
                            dst_file = self._get_dst_path(
                                src_file, src_dir, dst_dir, patterns["rename"]
                            )
                            dst_file.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(src_file, dst_file)
                            copied.append(dst_file)
            else:
                # Direct file path
                src_file = src_dir / pattern
                if src_file.exists() and src_file.is_file():
                    if not self._should_exclude(src_file, patterns["exclude"]):
                        dst_file = self._get_dst_path(
                            src_file, src_dir, dst_dir, patterns["rename"]
                        )
                        dst_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src_file, dst_file)
                        copied.append(dst_file)
        
        return copied
    
    def _should_exclude(self, file_path: Path, exclude_patterns: List[str]) -> bool:
        """Check if file should be excluded."""
        for pattern in exclude_patterns:
            if file_path.match(pattern):
                return True
        return False
    
    def _get_dst_path(
        self,
        src_file: Path,
        src_root: Path,
        dst_root: Path,
        rename_map: Dict[str, str]
    ) -> Path:
        """Get destination path with renaming."""
        rel_path = src_file.relative_to(src_root)
        
        # Apply renaming
        parts = list(rel_path.parts)
        for i, part in enumerate(parts[:-1]):  # Don't rename the file itself
            if part in rename_map:
                parts[i] = rename_map[part]
        
        return dst_root / Path(*parts)
    
    def _minimize_imports(self, path: Path) -> None:
        """Remove unnecessary imports and update paths."""
        for py_file in path.rglob("*.py"):
            try:
                content = py_file.read_text(encoding="utf-8")
                original = content
                
                # Remove test imports
                content = re.sub(r'^import pytest.*$', '', content, flags=re.MULTILINE)
                content = re.sub(r'^from .* import.*test.*$', '', content, flags=re.MULTILINE)
                
                # Update relative imports for HiT-SR
                if "hitsr" in str(py_file):
                    content = content.replace('from basicsr.', 'from .')
                    content = content.replace('import basicsr.', 'import .')
                
                # Update relative imports for TSD-SR
                if "tsdsr" in str(py_file):
                    # No specific import updates needed for TSD-SR as it uses relative imports already
                    pass
                
                # Remove empty lines at the beginning
                content = content.lstrip('\n')
                
                if content != original:
                    py_file.write_text(content, encoding="utf-8")
                    
            except Exception as e:
                print(f"  Warning: Failed to process {py_file}: {e}")
    
    def _create_init_files(self, root_dir: Path) -> None:
        """Create __init__.py files in all directories."""
        for dir_path in root_dir.rglob("*"):
            if dir_path.is_dir() and not (dir_path / "__init__.py").exists():
                # Check if it has Python files
                py_files = list(dir_path.glob("*.py"))
                if py_files:
                    init_file = dir_path / "__init__.py"
                    init_file.write_text('"""Auto-generated module."""\n')
    
    def _copy_license(self, src_dir: Path, dst_dir: Path) -> None:
        """Copy license file."""
        for license_file in ["LICENSE", "LICENSE.md", "LICENSE.txt", "NOTICE"]:
            src_license = src_dir / license_file
            if src_license.exists():
                dst_license = dst_dir / "LICENSE"
                shutil.copy2(src_license, dst_license)
                print(f"  Copied license: {license_file}")
                break
    
    def create_wrapper(self, model_name: str) -> None:
        """Create a wrapper module for easier imports."""
        model_lower = model_name.lower().replace("-", "")
        wrapper_file = self.backends_dir / model_lower / "__init__.py"
        
        if model_lower == "hitsr":
            content = '''"""HiT-SR backend wrapper."""

from .models.hit_arch import *
from .models.hit_sir_arch import *
from .models.hit_sng_arch import *
from .models.hit_srf_arch import *

# Simplified model creation
def create_model(opt):
    """Create HiT-SR model from options."""
    from .trainers.hit_sr_model import HiTSRModel
    return HiTSRModel(opt)
'''
        elif model_lower == "tsdsr":
            content = '''"""TSD-SR backend wrapper."""

from .models.autoencoder_kl import AutoencoderKL
from .utils.util import load_lora_state_dict
from .utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
from .utils.vaehook import _init_tiled_vae

__all__ = [
    'AutoencoderKL',
    'load_lora_state_dict',
    'adain_color_fix',
    'wavelet_color_fix',
    '_init_tiled_vae',
]
'''
        elif model_lower == "varsr":
            content = '''"""VARSR backend wrapper."""

from .models.var import *
from .models.vqvae import *
from .models.basic_var import *
'''
        else:
            content = f'"""Auto-generated module for {model_name}."""\n'
        
        wrapper_file.write_text(content)
        print(f"  Created wrapper: {wrapper_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Sync models from third_party to backends")
    parser.add_argument(
        "models",
        nargs="*",
        help="Models to sync (default: all)",
        choices=["HiT-SR", "TSD-SR", "VARSR", "all"],
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing files",
    )
    parser.add_argument(
        "--minimize",
        action="store_true",
        help="Minimize code size for release",
    )
    
    args = parser.parse_args()
    
    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    syncer = ModelSynchronizer(project_root)
    
    # Determine which models to sync
    if not args.models or "all" in args.models:
        models = ["HiT-SR", "TSD-SR", "VARSR"]
    else:
        models = args.models
    
    # Sync each model
    for model in models:
        try:
            syncer.sync_model(model, force=args.force)
            syncer.create_wrapper(model)
        except Exception as e:
            print(f"Error syncing {model}: {e}")
            continue
    
    print("\nSync complete!")
    
    # Show size statistics
    total_size = 0
    for model in models:
        model_dir = syncer.backends_dir / model.lower()
        if model_dir.exists():
            size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
            print(f"{model}: {size / 1024 / 1024:.1f} MB")
            total_size += size
    
    print(f"Total: {total_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()