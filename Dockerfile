FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    timm==0.6.13 \
    einops \
    scipy \
    opencv-python \
    gdown \
    huggingface-hub \
    tqdm \
    requests \
    pillow

# Copy superscale code
WORKDIR /workspace
COPY . /workspace/superscale/

# Install superscale
WORKDIR /workspace/superscale
RUN pip install -e .

# Set working directory
WORKDIR /workspace

CMD ["/bin/bash"]