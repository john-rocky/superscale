# TSD-SR (Target Score Distillation for Super-Resolution)

TSD-SR is a one-step diffusion model based on Stable Diffusion 3, achieving real-world image super-resolution through target score distillation.

## Features

- **One-step inference**: Fast super-resolution without iterative denoising
- **Based on SD3**: Leverages the power of Stable Diffusion 3
- **4x upscaling**: Fixed 4x scale factor
- **Multiple variants**: 
  - `tsdsr`: Standard model
  - `tsdsr-mse`: MSE-optimized variant
  - `tsdsr-gan`: GAN-optimized variant

## Usage

```python
import superscale

# Load TSD-SR model
pipe = superscale.load("tsdsr", device="cuda")

# Or use the mythological alias
pipe = superscale.summon("Hermes")  # Messenger god - fast communication

# Upscale an image
result = pipe("low_res.jpg", scale=4)
```

## Model Requirements

### Dependencies

Install TSD-SR dependencies:
```bash
pip install superscale[tsdsr]
```

This installs:
- `diffusers>=0.29,<0.33`
- `transformers>=4.40,<4.50`
- `safetensors>=0.4`
- `omegaconf>=2.3`
- `peft>=0.10`

### Model Weights

TSD-SR requires:
1. **Stable Diffusion 3 base model** from HuggingFace
2. **TSD-SR LoRA weights** (download from official sources)
3. **Prompt embeddings** for optimal results

## Automatic Download

TSD-SR supports automatic download of model weights:

```python
import superscale

# Auto-download on first use
pipe = superscale.load("tsdsr", device="cuda")
# Will prompt for manual download from Google Drive/OneDrive

# Or download manually
from superscale.models.tsdsr.download import download_tsdsr_model

# Download with interactive prompts
paths = download_tsdsr_model("tsdsr")

# Get download instructions
python -m superscale.models.tsdsr.download tsdsr --instructions-only
```

### Fully Automatic Download (with Google Drive)

To enable fully automatic download:

1. **Upload files to your Google Drive**:
   - Download files from the [official sources](#download-sources)
   - Upload to your Google Drive
   - Set sharing to "Anyone with the link"

2. **Configure file IDs**:
   Edit `superscale/models/tsdsr/gdrive_config.py`:
   ```python
   TSDSR_FILE_IDS = {
       "tsdsr": {
           "transformer.safetensors": "YOUR_FILE_ID",  # Replace
           "vae.safetensors": "YOUR_FILE_ID",  # Replace
           "prompt_embeds.pt": "YOUR_FILE_ID",  # Replace
           "pool_embeds.pt": "YOUR_FILE_ID",  # Replace
       },
   }
   ```

3. **Use automatic download**:
   ```python
   # Now works automatically!
   pipe = superscale.load("tsdsr", device="cuda")
   
   # Or via command line
   python -m superscale.models.tsdsr.download_gdrive tsdsr
   ```

### Download Sources

- **SD3 Base Model**: [HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers)
- **LoRA Weights**: 
  - [Google Drive](https://drive.google.com/drive/folders/1XJY9Qxhz0mqjTtgDXr07oFy9eJr8jphI)
  - [OneDrive](https://1drv.ms/f/c/d75249b59f444489/EsQQ2LLXp7pHsYMBVubgcsYBvEQXMmcNXGnz695odCGByQ)

## Technical Details

- **Architecture**: SD3 Transformer with LoRA adaptation
- **Training**: Target score distillation from teacher model
- **Inference**: Single-step diffusion process
- **Supported formats**: RGB images
- **Optimal input size**: 128x128 minimum recommended

## Summoning Ritual

When loading TSD-SR models, you'll see a magical summoning ritual:

```
🌌 INITIATING DIFFUSION SUMMONING RITUAL 🌌
============================================================

      ✨ ∞ ✨ ∞ ✨
   ∞     ╭─────╮     ∞
  ✨   ╭─┤ SD3 ├─╮   ✨
 ∞   ╭─┴─┴─────┴─┴─╮   ∞
✨  │  🎯 TSD-SR 🎯  │  ✨
 ∞   ╰─┬─┬─────┬─┬─╯   ∞
  ✨   ╰─┤ x4  ├─╯   ✨
   ∞     ╰─────╯     ∞
      ✨ ∞ ✨ ∞ ✨

🎯 Summoning the Target Score Distillation Entity
📜 Model: TSD-SR
⚡ Scale Factor: 4x
🔮 Base: Stable Diffusion 3

🌟 Reciting the diffusion incantation...
   「From noise to clarity, from small to grand」
   「By the power of score distillation, enhance!」
   「One step to rule them all, one step to upscale!」

⚡ The diffusion portal opens... Latent space aligns...
🌊 ≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈ 🌊

🎯 SUCCESS! TSD-SR manifests from the latent realm!
🌌 One-step diffusion magic is now at your command!
```

## References

- [TSD-SR Paper](https://arxiv.org/abs/2411.18263)
- [Official Repository](https://github.com/Microtreei/TSD-SR)
- Accepted by CVPR2025