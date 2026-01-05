# Multi-stage build for Stable Diffusion Generator
# This Dockerfile builds a containerized version with CUDA support

# ============================================
# Stage 1: Builder - Compile everything
# ============================================
FROM golang:1.25-bookworm AS builder

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive


WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

COPY . .

RUN CGO_ENABLED=0 go build -o go-ai-gen main.go





# ============================================
# Stage 2: Python Dependencies Builder
# ============================================

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04


ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# Install Python and runtime dependencies
RUN apt-get update && apt-get install -y  \
  python3 \
  python3-pip \
  python3-venv \
  ca-certificates \
  openssh-server \
  vim \
  nano \
  htop \
  && rm -rf /var/lib/apt/lists/*

# Install torch first (largest package)
RUN pip3 install --no-cache-dir torch torchvision torchaudio   && \
  rm -rf ~/.cache/pip /tmp/*

# Install Python dependencies
RUN pip3 install --no-cache-dir \
  diffusers \
  transformers \
  accelerate \
  safetensors \
  sentencepiece \
  protobuf \
  pillow \
  xformers \
  bitsandbytes \
  hf_transfer \
  gguf>=0.10.0 \
  peft  && \
  rm -rf ~/.cache/pip /tmp/*

# Install Nunchaku from pre-built wheel (v1.1.0, no compilation!)
RUN pip3 install --no-cache-dir \
  https://github.com/nunchaku-tech/nunchaku/releases/download/v1.1.0/nunchaku-1.1.0+torch2.11-cp310-cp310-linux_x86_64.whl && \
  rm -rf ~/.cache/pip /tmp/* /var/tmp/* /root/.cache/*

WORKDIR /app

COPY --from=builder /app/go-ai-gen /app/
# Copy Python script
COPY python_scripts/*.py /app/python_scripts/
# Copy startup script
COPY start.sh /app/start.sh



# Install Nunchaku separately (uses existing torch)
RUN  mkdir -p  /app/downloads && \
  mkdir -p /run/sshd && \
  echo 'root:root' | chpasswd && \
  sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
  sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
  chmod +x /app/start.sh



#   && echo "export HF_HOME=/app/models/.cache" > /etc/profile.d/hf_home.sh


# Set library path for CUDA
ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}"

# Expose SSH port
EXPOSE 22 8080

# Start SSH server in foreground for Runpod
CMD ["/bin/bash", "/app/start.sh"]

