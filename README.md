# Superscale 🚀

Universal super-resolution toolkit with a diffusers-like API for state-of-the-art image upscaling models.

## Features

- 🎯 **Simple API**: One-line image super-resolution similar to diffusers
- 🏛️ **Multiple SOTA Models**: HiT-SR, TSD-SR, VARSR and more
- ⚡ **Efficient**: Smart caching and batch processing support
- 🔧 **Flexible**: Support for different scales (2x, 4x, 8x)
- 📦 **Easy Install**: Minimal dependencies with optional extras

## Quick Start

```python
import superscale as ss

# One-line super-resolution
hr_image = ss.up("low_res.jpg", model="Hermes", scale=4)

# Or use the pipeline interface
pipe = ss.load("Athena", device="cuda")
hr_image = pipe("low_res.jpg", scale=4)

# Efficient batch processing
with ss.session("Zeus") as sess:
    for img in images:
        result = sess.upscale(img, scale=4)
```

## Installation

```bash
# Basic installation
pip install superscale

# Install with specific model support
pip install superscale[tsdsr]  # For TSD-SR model
pip install superscale[hitsr]  # For HiT-SR models  
pip install superscale[varsr]  # For VARSR models
pip install superscale[all]    # All models

# Install with GUI support
pip install superscale[gui]
```

## Available Models

| Model | Alias | Description | Speed | Quality |
|-------|-------|-------------|-------|---------|
| TSD-SR | Hermes | Stable Diffusion 3 based SR | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| HiT-SIR | Athena | Hierarchical Transformer SR | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| HiT-SNG | Apollo | HiT-SR variant | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| HiT-SRF | Artemis | HiT-SR large variant | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| VARSR-d16 | Zeus | Visual Autoregressive SR | ⭐⭐⭐ | ⭐⭐⭐⭐ |

## CLI Usage

```bash
# Upscale a single image
superscale up image.jpg -m Hermes -s 4 -o output.jpg

# List available models
superscale model list

# Download model weights
superscale model download Athena

# Launch GUI
superscale gui
```

## Advanced Usage

### Custom Device and Precision

```python
# Use specific GPU
pipe = ss.load("Hermes", device="cuda:1", dtype=torch.float16)

# CPU inference
pipe = ss.load("Athena", device="cpu")
```

### Tiled Processing for Large Images

```python
from superscale.api.tiling import TiledProcessor

with ss.session("Zeus") as sess:
    tiled = TiledProcessor(sess, tile_size=512, overlap=32)
    result = tiled.process_large_image(large_image, scale=4)
```

### Video Processing

```python
from superscale.api.streaming import StreamProcessor

processor = StreamProcessor("Hermes")
processor.process_video("input.mp4", "output.mp4", scale=2)
```

## Model Details

### HiT-SR (Hierarchical Transformer SR)
- Paper: [ECCV 2024 Oral]
- Three variants: SIR, SNG, SRF
- Supports 2x, 3x, 4x scaling
- Efficient transformer architecture

### TSD-SR (Text-to-Image Diffusion SR)  
- Paper: [CVPR 2025]
- Based on Stable Diffusion 3
- Best quality, slower speed
- Supports 2x, 4x, 8x scaling

### VARSR (Visual Autoregressive SR)
- Next-scale prediction approach
- Good balance of speed and quality
- Supports arbitrary scaling factors

## Development

```bash
# Clone with submodules (development only)
git clone --recursive https://github.com/yourusername/superscale.git
cd superscale

# Install in development mode
pip install -e ".[dev,all]"

# Run tests
pytest tests/

# Format code
black superscale/
ruff check superscale/
```

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

Model implementations are adapted from their original repositories with respective licenses:
- HiT-SR: Apache-2.0
- TSD-SR: Apache-2.0  
- VARSR: MIT

See [LICENSE-3rdparty.md](LICENSE-3rdparty.md) for full third-party license information.

## Citation

If you use Superscale in your research, please cite:

```bibtex
@software{superscale2024,
  title = {Superscale: Universal Super-Resolution Toolkit},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/superscale}
}
```

And the original model papers:
- [HiT-SR](https://github.com/XPixelGroup/HiT-SR)
- [TSD-SR](https://github.com/Iceclear/TSD-SR)
- [VARSR](https://github.com/FoundationVision/VARSR)