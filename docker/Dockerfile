FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV SAM2_BUILD_CUDA=0
# Prefer the most conservative SAM2 runtime path in Docker. This avoids
# several predictor-init crashes seen with lower-precision kernels.
ENV OVO_SAM2_FP32=1
ENV OVO_ENABLE_SAM_WARMUP=0

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3-pip git build-essential \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Must match WORK_DIR in docker/run_docker.sh
WORKDIR /ws/external

RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade "pip<26" setuptools wheel && \
    pip install --no-cache-dir \
    torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu121 && \
    pip install \
    pyyaml tqdm psutil wandb plyfile numpy==2.4.* matplotlib seaborn \
    opencv-python==4.11.* imageio scipy scikit-learn pandas \
    transformers==4.51.0 open_clip_torch==2.32.0 open3d==0.19.0 \
    huggingface-hub==0.30.1 einops==0.8.1 debugpy

COPY . /ws/external

RUN pip install -e /ws/external/thirdParty/segment-anything-2 --no-build-isolation && \
    pip install -e /ws/external/thirdParty/perception_models --no-dependencies

CMD ["/bin/bash"]
