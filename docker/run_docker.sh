#!/bin/bash

# OVO Docker 실행 스크립트 (이미지는 docker/build.sh 로 빌드, 경로는 Dockerfile WORKDIR 와 동일해야 함)
# 사용법: ./run_docker.sh [GPU번호] [명령어...]
# 예시: ./run_docker.sh 0 python run_eval.py --dataset_name Replica --experiment_name ovo_mapping --run --scenes office0
#       ./run_docker.sh 1 bash
#       ./run_docker.sh python run_eval.py ...   (기본값: GPU 0)

set -euo pipefail

# 기본 설정 (docker/build.sh → docker/Dockerfile 의 태그와 일치)
LOCAL_IMAGE="ovo:cuda121"
REMOTE_IMAGE=""

if docker image inspect "${LOCAL_IMAGE}" &> /dev/null; then
    IMAGE_NAME="${LOCAL_IMAGE}"
    echo "Using local Docker image: ${IMAGE_NAME}"
else
    if [ -n "${REMOTE_IMAGE}" ]; then
        IMAGE_NAME="${REMOTE_IMAGE}"
        echo "Local image not found. Will use remote image: ${IMAGE_NAME}"
    else
        echo "Error: Docker image '${LOCAL_IMAGE}' not found." >&2
        echo "Build from repo root: ./docker/build.sh" >&2
        exit 1
    fi
fi

# 컨테이너 내부 작업 디렉터리 (Dockerfile WORKDIR 와 동일)
WORK_DIR="/ws/external"

# 현재 스크립트의 상위 = OVO 프로젝트 루트
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 런타임 data 루트: 호스트 /media/gml78905/T75/wh -> 컨테이너 /ws/data
DATASET_DIR="${OVO_HOST_DATA_ROOT:-/media/gml78905/T7/wh}"
DATASET_MOUNT="/ws/data"
DATA_ROOT_IN_CONTAINER="${OVO_DATA_ROOT:-${DATASET_MOUNT}/OVO}"

# SAM2 / PyTorch 등이 기본 64MB shm에서 세그폴트 나는 경우가 있어 넉넉히 둠 (필요 시 SHM_SIZE=16g 등)
SHM_SIZE="${SHM_SIZE:-32g}"
DEBUGPY_PORT="${DEBUGPY_PORT:-}"

if [ ! -d "${DATASET_DIR}" ]; then
    echo "Warning: host data path does not exist yet: ${DATASET_DIR}" >&2
    echo "Docker will create an empty directory there on first run." >&2
fi

GPU_NUM="0"
COMMAND_ARGS=()

if [ $# -gt 0 ] && [[ "$1" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    GPU_NUM="$1"
    shift
    COMMAND_ARGS=("$@")
else
    COMMAND_ARGS=("$@")
fi

CONTAINER_NAME="ovo_${GPU_NUM//,/_}"

if command -v nvidia-smi &> /dev/null; then
    GPU_FLAG="--gpus device=${GPU_NUM}"
    echo "GPU detected. Using GPU(s): ${GPU_NUM}"
else
    GPU_FLAG=""
    echo "No GPU detected. Running in CPU mode"
fi

if [ "$(docker ps -aq -f name="${CONTAINER_NAME}")" ]; then
    echo "Stopping existing container named: ${CONTAINER_NAME}..."
    docker stop "${CONTAINER_NAME}" > /dev/null 2>&1 || true
    docker rm "${CONTAINER_NAME}" > /dev/null 2>&1 || true
fi

if [ "${IMAGE_NAME}" = "${REMOTE_IMAGE}" ] && [ -n "${REMOTE_IMAGE}" ]; then
    echo "Pulling Docker image: ${IMAGE_NAME}"
    docker pull "${IMAGE_NAME}"
else
    echo "Using local image. Skipping pull."
fi

echo "Starting Docker container: ${CONTAINER_NAME}"
echo "Project: ${PROJECT_DIR} -> ${WORK_DIR}"
echo "Data:    ${DATASET_DIR} -> ${DATASET_MOUNT}"
echo "OVO root in container: ${DATA_ROOT_IN_CONTAINER}"
echo "GPU(s):  ${GPU_NUM}"
echo "shm:     ${SHM_SIZE}"
if [ -n "${DEBUGPY_PORT}" ]; then
    echo "debugpy: ${DEBUGPY_PORT}"
fi

PORT_ARGS=()
if [ -n "${DEBUGPY_PORT}" ]; then
    PORT_ARGS=(-p "${DEBUGPY_PORT}:${DEBUGPY_PORT}")
fi

DISPLAY_ARGS=()
if [ -n "${DISPLAY:-}" ] && [ -d /tmp/.X11-unix ]; then
    echo "X11:     ${DISPLAY} (mounted /tmp/.X11-unix)"
    DISPLAY_ARGS=(
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw
        -e DISPLAY="${DISPLAY}"
    )
else
    echo "X11:     unavailable (GUI apps may not open)"
fi

DOCKER_RUN=(docker run -it --rm
    ${GPU_FLAG}
    --name "${CONTAINER_NAME}"
    --ipc=host
    --shm-size="${SHM_SIZE}"
    --privileged
    "${PORT_ARGS[@]}"
    "${DISPLAY_ARGS[@]}"
    -v "${PROJECT_DIR}:${WORK_DIR}"
    -v "${DATASET_DIR}:${DATASET_MOUNT}"
    -w "${WORK_DIR}"
    -e PYTHONUNBUFFERED=1
    -e CUDA_VISIBLE_DEVICES="${GPU_NUM}"
    -e OVO_DATA_ROOT="${DATA_ROOT_IN_CONTAINER}"
    -e OVO_SAM2_FP32="${OVO_SAM2_FP32:-1}"
    -e OVO_ENABLE_SAM_WARMUP="${OVO_ENABLE_SAM_WARMUP:-0}"
    "${IMAGE_NAME}"
)

if [ ${#COMMAND_ARGS[@]} -eq 0 ]; then
    echo "No command provided. Starting interactive bash shell..."
    "${DOCKER_RUN[@]}" bash
else
    "${DOCKER_RUN[@]}" "${COMMAND_ARGS[@]}"
fi

echo "Container stopped."
