ARG CUDA_VARIANT=nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive
FROM ${CUDA_VARIANT}

# 1. Install system/build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      vim nano curl wget git-lfs \
      htop nvtop \
      build-essential \
      software-properties-common cmake git python3.11 python3.11-dev python3.11-distutils python3.11-venv python3-pip \
      libssl-dev libgl1 libglib2.0-0

# 2. Install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g npm@latest && \
    rm -rf /var/lib/apt/lists/*

# 3. Set python3.11 as default and install base Python tools
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    python3 --version && \
    python3 -m pip install --upgrade pip
