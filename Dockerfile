FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# Set timezone non-interactively
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Set working directory
WORKDIR /usr/src/vision-app

# Install system dependencies and Python3
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    python3-dev \
    python3-venv \
    python3-pip \
    git \
    curl \
    nano \
    netcat \
    # Add OpenCV dependencies
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    python3-opencv \
    # Add tzdata and configure it non-interactively
    tzdata \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python3 -m venv /opt/venv

# Activate virtual environment and set PATH
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first to leverage Docker caching
COPY requirements.txt .

# Install opencv-python explicitly before other requirements
RUN pip install --upgrade pip && \
    pip install opencv-python && \
    pip install -r requirements.txt

# Copy the rest of the project files
COPY . .

# Now install your package in editable mode
RUN pip install -e .

# Expose port (if needed)
EXPOSE 5003

# Default command to run the app
CMD ["python", "examples/test.py"]