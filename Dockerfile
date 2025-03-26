FROM nvcr.io/nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip

# Process requirements file to fix dependency conflicts
RUN pip install -r requirements.txt

# Set CUDA environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
ENV CUDA_VISIBLE_DEVICES=0

# Set PYTHONPATH to ensure module imports work correctly
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Make scripts executable
RUN chmod +x examples/test.py

# Create required directories if they don't exist
RUN mkdir -p data/hb-kor models videos results

# Default command
CMD ["python3", "examples/test.py"]