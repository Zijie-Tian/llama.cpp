FROM ubuntu:22.04

# Install essential development tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    curl \
    wget \
    vim \
    nano \
    pkg-config \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -s /bin/bash ubuntu

# Set up user and working directory as recommended
USER ubuntu
WORKDIR /home/ubuntu

# Install PyTorch CPU version only
RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cpu

# Set default command
CMD ["/bin/bash"]
