# Base image with CUDA 12.1 - good for NVIDIA RTX 3000+ series
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set non-interactive mode for installs
ENV DEBIAN_FRONTEND=noninteractive

# Install ffmpeg, python, and other useful tools for this project
RUN apt-get update && apt-get install -y \
    ffmpeg \
    python3-pip \
    python3-venv \
    git \
    nano \
    procps \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tmux \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install the Python libraries we need for analysis
RUN pip3 install --no-cache-dir \
    opencv-python-headless \
    mediapipe \
    imagehash \
    numpy \
    tqdm \
    tensorboard \
    torch \
    torchvision \
    rembg[gpu] \
    google-genai \
    google-cloud-aiplatform

# Create a non-root user 'pipeline_user' with user ID 1000
# This helps avoid permission errors on mounted volumes
RUN useradd -m -s /bin/bash -u 1000 pipeline_user

# Switch to the new user
USER pipeline_user

# We will mount our project data here
WORKDIR /projects
