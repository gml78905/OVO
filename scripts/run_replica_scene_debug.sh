#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOCKER_RUNNER="${PROJECT_DIR}/docker/run_docker.sh"

SCENE="${1:-office0}"
EXPERIMENT_NAME="${2:-replica_debug}"
GPU="${3:-0}"
DEBUGPY_PORT="${DEBUGPY_PORT:-5678}"

child_pid=""

cleanup() {
    if [ -n "${child_pid}" ] && kill -0 "${child_pid}" 2>/dev/null; then
        kill "${child_pid}" 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM

if [ ! -x "${DOCKER_RUNNER}" ]; then
    echo "Docker runner not found or not executable: ${DOCKER_RUNNER}" >&2
    exit 1
fi

echo "Starting Replica debug run"
echo "  scene: ${SCENE}"
echo "  experiment: ${EXPERIMENT_NAME}"
echo "  gpu: ${GPU}"
echo "  debugpy port: ${DEBUGPY_PORT}"

DEBUGPY_PORT="${DEBUGPY_PORT}" "${DOCKER_RUNNER}" "${GPU}" \
    python -Xfrozen_modules=off -m debugpy --listen "0.0.0.0:${DEBUGPY_PORT}" --wait-for-client \
    run_eval.py --dataset_name Replica --experiment_name "${EXPERIMENT_NAME}" \
    --run --segment --eval --scenes "${SCENE}" &
child_pid=$!

for _ in $(seq 1 120); do
    if bash -lc "exec 3<>/dev/tcp/127.0.0.1/${DEBUGPY_PORT}" 2>/dev/null; then
        echo "DEBUGPY_READY:${DEBUGPY_PORT}"
        wait "${child_pid}"
        exit $?
    fi
    sleep 1
done

echo "Timed out waiting for debugpy on port ${DEBUGPY_PORT}" >&2
exit 1
