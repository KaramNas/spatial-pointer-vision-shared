# GPU-enabled inference server for spatial-pointer-vision.
#
# Base: CUDA 12.1 runtime (Ubuntu 22.04). Requires the NVIDIA Container
# Toolkit on the host, plus `docker run --gpus all` or docker-compose's GPU
# reservation block (see ../docker-compose.yml) to pass the host GPU
# through:
#   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
#
# Build from the REPO ROOT (not this directory), since it needs
# server/requirements.txt and the rest of server/ in its build context:
#   docker build -f server/Dockerfile -t spatial-pointer-vision-server .
# (docker-compose.yml does this for you.)
#
# First build is slow: torch's CUDA 12.1 wheel alone is ~2.4GB.
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Install dependencies before copying source so `docker build` can cache this
# (usually the slowest) layer across code-only changes.
COPY server/requirements.txt server/requirements.txt
RUN python -m pip install --upgrade pip \
    && pip install -r server/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

COPY server/ server/

ENV BASE_MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct" \
    DEVICE=cuda \
    HOST=0.0.0.0 \
    PORT=8000
# MODEL_CHECKPOINT_PATH is intentionally left unset here -- pass it via
# `docker run -e MODEL_CHECKPOINT_PATH=...` or docker-compose.yml so it
# points at a checkpoint bind-mounted from the host (checkpoints are
# git-ignored and multiple GB, so they're never baked into the image).

EXPOSE 8000

CMD ["python", "server/main.py"]
