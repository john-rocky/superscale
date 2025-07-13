"""Command-line interface for Superscale."""

import sys
from pathlib import Path
from typing import Optional

import click
from PIL import Image

from .. import __version__
from ..api import load, up
from ..core.registry import ModelRegistry
from ..core.model_manager import get_model_manager
from ..core.cache_manager import get_cache_manager


@click.group()
@click.version_option(version=__version__, prog_name="superscale")
def cli():
    """Superscale - Universal Super-Resolution Toolkit.
    
    A unified library for state-of-the-art image super-resolution models
    with a diffusers-like API.
    """
    pass


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("-m", "--model", default="Hermes", help="Model to use for upscaling")
@click.option("-s", "--scale", default=4, type=int, help="Upscaling factor")
@click.option("-o", "--output", help="Output path (default: auto-generated)")
@click.option("-d", "--device", default="auto", help="Device to use (auto/cpu/cuda)")
@click.option("--no-download", is_flag=True, help="Disable automatic model download")
def up(input_path: str, model: str, scale: int, output: Optional[str], device: str, no_download: bool):
    """Upscale a single image.
    
    Examples:
        superscale up image.jpg -m Hermes -s 4
        superscale up photo.png -m Athena -o output.png
    """
    input_path = Path(input_path)
    
    if not input_path.is_file():
        click.echo(f"Error: {input_path} is not a file", err=True)
        sys.exit(1)
    
    # Determine output path
    if output:
        output_path = Path(output)
    else:
        # Auto-generate output name
        output_path = input_path.parent / f"{input_path.stem}_x{scale}{input_path.suffix}"
    
    try:
        # Load model and upscale
        click.echo(f"Loading {model}...")
        pipe = load(model, device=device, download=not no_download)
        
        click.echo(f"Upscaling {input_path} by {scale}x...")
        result = pipe(str(input_path), scale=scale)
        
        # Save result
        result.save(output_path)
        click.echo(f"✓ Saved to: {output_path}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.group()
def model():
    """Model management commands."""
    pass


@model.command("list")
@click.option("--downloaded", is_flag=True, help="Show only downloaded models")
def list_models(downloaded: bool):
    """List available models."""
    if downloaded:
        # Show downloaded models
        manager = get_model_manager()
        downloaded_models = manager.list_downloaded_models()
        
        if not downloaded_models:
            click.echo("No models downloaded yet.")
            return
        
        click.echo("Downloaded models:")
        for name, info in downloaded_models.items():
            size = info.get("size", "Unknown")
            click.echo(f"  • {name} ({size})")
    else:
        # Show all available models
        models = ModelRegistry.list_models()
        
        click.echo("Available models:")
        for name in models:
            # Get metadata
            metadata = ModelRegistry.get_metadata(name)
            desc = metadata.get("description", "")
            
            # Get aliases
            aliases = ModelRegistry.get_aliases(name)
            alias_str = f" (aliases: {', '.join(aliases)})" if aliases else ""
            
            click.echo(f"  • {name}{alias_str} - {desc}")


@model.command("download")
@click.argument("model_name")
@click.option("--force", is_flag=True, help="Force re-download")
def download_model(model_name: str, force: bool):
    """Download model weights.
    
    Examples:
        superscale model download Hermes
        superscale model download hitsr-sir-x4 --force
    """
    try:
        manager = get_model_manager()
        
        # Check if already downloaded
        if not force and manager.get_model_path(model_name):
            click.echo(f"{model_name} is already downloaded.")
            if click.confirm("Download anyway?"):
                force = True
            else:
                return
        
        # Download
        click.echo(f"Downloading {model_name}...")
        path = manager.download_model(model_name, force=force)
        click.echo(f"✓ Downloaded to: {path}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@model.command("info")
@click.argument("model_name", required=False)
def model_info(model_name: Optional[str]):
    """Show model information.
    
    Examples:
        superscale model info          # Show all models
        superscale model info Hermes   # Show specific model
    """
    manager = get_model_manager()
    
    if model_name:
        # Show specific model info
        try:
            # Resolve aliases
            model_class = ModelRegistry.get(model_name)
            
            # Find primary name
            primary_name = None
            for name, cls in ModelRegistry._models.items():
                if cls == model_class:
                    primary_name = name
                    break
            
            if not primary_name:
                primary_name = model_name
            
            # Get info
            metadata = ModelRegistry.get_metadata(primary_name)
            aliases = ModelRegistry.get_aliases(primary_name)
            
            click.echo(f"Model: {primary_name}")
            if aliases:
                click.echo(f"Aliases: {', '.join(aliases)}")
            
            if metadata:
                for key, value in metadata.items():
                    click.echo(f"{key.capitalize()}: {value}")
            
            # Download info
            if primary_name in manager.MODEL_CONFIGS:
                config = manager.MODEL_CONFIGS[primary_name]
                click.echo(f"Download size: {config.get('size', 'Unknown')}")
                
                # Check if downloaded
                path = manager.get_model_path(primary_name)
                if path:
                    click.echo(f"Status: Downloaded ({path})")
                else:
                    click.echo("Status: Not downloaded")
            
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
    else:
        # Show all models
        click.echo("Available models:\n")
        
        for name in ModelRegistry.list_models():
            metadata = ModelRegistry.get_metadata(name)
            desc = metadata.get("description", "No description")
            
            # Check download status
            downloaded = "✓" if manager.get_model_path(name) else "✗"
            size = manager.get_download_size(name)
            
            click.echo(f"[{downloaded}] {name} ({size})")
            click.echo(f"    {desc}")
            click.echo()


@model.command("clear")
@click.argument("model_name", required=False)
@click.option("--all", is_flag=True, help="Clear all models")
def clear_cache(model_name: Optional[str], all: bool):
    """Clear model cache.
    
    Examples:
        superscale model clear Hermes    # Clear specific model
        superscale model clear --all     # Clear all models
    """
    manager = get_model_manager()
    
    if all:
        if click.confirm("Clear all downloaded models?"):
            manager.clear_cache()
            click.echo("✓ Cleared all models")
    elif model_name:
        if click.confirm(f"Clear {model_name}?"):
            manager.clear_cache(model_name)
            click.echo(f"✓ Cleared {model_name}")
    else:
        click.echo("Please specify a model name or use --all")


@cli.command()
def cache():
    """Show cache information."""
    from ..core.utils import format_bytes
    
    # Model cache
    cache_manager = get_cache_manager()
    cache_info = cache_manager.get_memory_usage()
    
    click.echo("Model Cache:")
    click.echo(f"  Cached models: {cache_info['cached_models']}/{cache_info['max_models']}")
    click.echo(f"  Weak references: {cache_info['weak_refs']}")
    
    if "gpu_memory_allocated" in cache_info:
        allocated = format_bytes(cache_info["gpu_memory_allocated"])
        reserved = format_bytes(cache_info["gpu_memory_reserved"])
        click.echo(f"  GPU memory: {allocated} allocated, {reserved} reserved")
    
    # Disk cache
    model_manager = get_model_manager()
    downloaded = model_manager.list_downloaded_models()
    
    click.echo("\nDisk Cache:")
    if downloaded:
        total_size = 0
        for name, info in downloaded.items():
            click.echo(f"  • {name}: {info['size']}")
            if "path" in info:
                path = Path(info["path"])
                if path.exists():
                    total_size += path.stat().st_size
        
        click.echo(f"\nTotal disk usage: {format_bytes(total_size)}")
    else:
        click.echo("  No models downloaded")


@cli.command()
def gui():
    """Launch the web GUI (requires gradio)."""
    try:
        from ..gui.app import create_app
        
        click.echo("Launching Superscale GUI...")
        app = create_app()
        app.launch()
        
    except ImportError:
        click.echo(
            "GUI requires gradio. Install with:\n"
            "  pip install superscale[gui]",
            err=True
        )
        sys.exit(1)


if __name__ == "__main__":
    cli()