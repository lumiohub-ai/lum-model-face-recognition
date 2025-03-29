FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

WORKDIR /usr/src/vision-app

# Install system dependencies and Python
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    python3-dev \
    python3-pip \
    python3-numpy \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python3 as default
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with GPU support
RUN pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118

# Copy the rest of the application
COPY . .

# Expose the port the app runs on
EXPOSE 5003

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tashkent

# Install utilities and tzdata without prompts
RUN apt-get update && \
    apt-get install -y \
        tzdata \
        curl \
        nano \
        netcat && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone && \
    dpkg-reconfigure -f noninteractive tzdata && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN date

# Command to run the application
# CMD ["python", "examples/test.py"]