FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

WORKDIR /usr/src/vision-app

# Install only essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev python3-pip python3-opencv \
    build-essential cmake \
    libopenblas-dev liblapack-dev \
    libgl1-mesa-glx libx11-dev \
    libsm6 libxext6 libxrender-dev \
    libgtk-3-dev \
    libpq-dev gcc git curl nano tzdata \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set pip aliases
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install Python packages
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir opencv-python-headless \
 && pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install local editable packages
RUN pip install --no-cache-dir -e modules/yolo_tracking \
 && pip install --no-cache-dir -e modules/insightface \
 && pip install --no-cache-dir -e .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose the desired port
EXPOSE 5003
