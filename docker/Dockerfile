# 1. Start from the 12.1.1 image (newer, fixes dependency issues)
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

WORKDIR /projects
ENV DEBIAN_FRONTEND=noninteractive

# 2. Install all OS-level packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    nano \
    tmux \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    ffmpeg \
    # --- FIX 1: Install the cuDNN 9 runtime library (for rembg) ---
    libcudnn9-cuda-12 \
    # --- FIX 2: Install the CUDA runtime libraries (for ffmpeg) ---
    cuda-libraries-12-1

# 3. Install all Python packages (pip)
RUN pip3 install --no-cache-dir \
    # --- The Core ML/GPU Stack ---
    torch \
    torchvision \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    \
    # --- The Conflict Resolution ---
    ultralytics \
    \
    # This will now find the correct cuDNN 9 library we just installed
    onnxruntime-gpu \
    --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ \
    \
    rembg \
    \
    # --- Other Utilities & Fixes ---

    # --- ADDED FOR STEP 03_score.py ---
    # This is the one package we're adding.
    # It's needed for sharpness/brightness checks.
    opencv-python-headless \
    # ---
    \
    # This fixes the _idat bug (and is needed for 01b_validate)
    "Pillow>=10.3.0" \
    imagehash \
    tqdm \
    tensorboard \
    google-genai \
    google-cloud-aiplatform

# 4. Set the default command to keep the container alive
CMD ["/bin/bash"]
