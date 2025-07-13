#!/usr/bin/env python3
"""Test if Superscale is working."""

import superscale as ss
from PIL import Image
import numpy as np

print(f"Superscale version: {ss.__version__}")

# List available models
print("\nAvailable models:")
models = ss.list_models()
for model in models:
    print(f"  - {model}")

# Create a test image
print("\nCreating test image...")
test_image = Image.new('RGB', (64, 64), color='red')
test_image.save('test_input.png')
print("Saved test_input.png")

# Test with dummy model
print("\nTesting with dummy model...")
try:
    # Method 1: One-line upscaling
    result = ss.up(test_image, model="dummy", scale=4)
    print(f"✓ One-line upscaling worked! Output size: {result.size}")
    result.save('test_output_dummy.png')
    
    # Method 2: Pipeline interface
    pipe = ss.load("dummy", device="cpu")
    print(f"✓ Loaded pipeline: {pipe}")
    
    result2 = pipe(test_image, scale=2)
    print(f"✓ Pipeline upscaling worked! Output size: {result2.size}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test with "Hermes" alias (dummy)
print("\nTesting with Hermes alias...")
try:
    result = ss.up(test_image, model="Hermes", scale=4)
    print(f"✓ Hermes alias worked! Output size: {result.size}")
    result.save('test_output_hermes.png')
except Exception as e:
    print(f"✗ Error: {e}")

print("\nDone! Check test_output_*.png files.")