# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview
Superscale is a plug-and-play super-resolution toolkit containing three state-of-the-art image super-resolution models:
- **HiT-SR**: Hierarchical Transformer for Efficient Image Super-Resolution
- **TSD-SR**: One-Step Diffusion with Target Score Distillation for Real-World Image Super-Resolution
- **VARSR**: Visual Autoregressive Modeling for Image Super-Resolution

## Code Modification Policy
- Do not modify any code until explicitly instructed to do so
- Review code thoroughly before making any changes
- Confirm with user before implementing major structural changes

## Git Commit Policy
- Never include Claude-related attribution or "Generated with Claude Code" in commit messages
- Write professional, standard git commit messages without AI tool references

## Build and Test Commands

### HiT-SR
```bash
# Setup
cd HiT-SR
conda create -n HiTSR python=3.8
conda activate HiTSR
pip install -r requirements.txt
python setup.py develop

# Training (Multi-GPU)
python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 basicsr/train.py -opt options/Train/train_HiT_SIR_x4.yml --launcher pytorch

# Testing
python basicsr/test.py -opt options/Test/test_HiT_SIR_x4.yml

# Test without ground-truth
python basicsr/test.py -opt options/Test/test_single_x4.yml
```

### TSD-SR
```bash
# Setup
cd TSD-SR
conda create -n tsdsr python=3.9
conda activate tsdsr
pip install -r requirements.txt

# Testing
python test/test_tsdsr.py \
--pretrained_model_name_or_path /path/to/sd3 \
-i imgs/test \
-o outputs/test \
--lora_dir checkpoint/tsdsr \
--embedding_dir dataset/default

# Training (Multi-GPU)
accelerate launch --config_file config/config.yaml --gpu_ids 0,1,2,3 --num_processes 4 --mixed_precision="fp16" train/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_batch_size=2 --rank=64 \
  --num_train_epochs=200 --checkpointing_steps=5000 \
  --learning_rate=5e-06 --output_dir=$OUTPUT_DIR

# Test metrics
python test/test_metrics.py \
--inp_imgs outputs/DrealSR \
--gt_imgs imgs/StableSR_testsets/DrealSRVal_crop128/test_HR \
--log logs/metrics
```

### VARSR
```bash
# Setup
cd VARSR
conda create -n varsr python=3.9
conda activate varsr
pip install -r requirements.txt

# Testing (512x512)
python test_varsr.py

# Testing (High-resolution)
python test_tile.py

# Training (Multi-GPU)
torchrun --nproc-per-node=8 train.py --depth=24 --batch_size=4 --ep=5 --fp16=1 --tblr=5e-5 --alng=1e-4 --wpe=0.01 --wandb_flag=True

# Class-to-Image
python test_C2I.py
```

## Code Style Guidelines
- Follow Python PEP 8 style guide
- Use type hints where appropriate
- Maintain consistent naming conventions across all models
- Keep imports organized (standard library, third-party, local imports)
- Add docstrings for all public functions and classes
- Use meaningful variable names
- Add comments for complex logic
- Avoid print statements in production code; use proper logging

## Documentation Guidelines
- All documentation, comments, and public API docs must be written in English
- Use clear, concise descriptions for all functions and classes
- Include parameter descriptions and return types in docstrings
- For complex functionality, include usage examples
- Maintain README.md files in each subdirectory with specific instructions
- Document any environment variables or configuration requirements

## Testing Guidelines
- Run tests before committing changes
- Ensure all models can run inference successfully
- Test with provided sample images before modifying core functionality
- Verify GPU memory usage stays within reasonable bounds
- Check that results are reproducible with fixed random seeds

## Model-Specific Notes

### HiT-SR
- Based on BasicSR framework
- Uses PyTorch 1.8.0 + Torchvision 0.9.0
- Configuration files in YAML format
- Supports multi-GPU training with distributed launch

### TSD-SR
- Based on Stable Diffusion 3
- Uses Diffusers library and LoRA weights
- Requires prompt embeddings for training
- Supports accelerate for multi-GPU training

### VARSR
- Visual autoregressive model
- Requires VQVAE model for inference
- Supports tile-based processing for high-resolution images
- Uses wandb for experiment tracking

## Environment Variables
- `CUDA_VISIBLE_DEVICES`: Control GPU usage
- `HF_ENDPOINT`: For Hugging Face mirror (TSD-SR)
- `BASICSR_EXT`: Whether to compile CUDA extensions (HiT-SR)

## Common Issues and Solutions
- Memory errors: Reduce batch size or use gradient accumulation
- CUDA version conflicts: Ensure PyTorch CUDA version matches system CUDA
- Missing dependencies: Check model-specific requirements.txt
- Slow training: Enable mixed precision training (fp16)