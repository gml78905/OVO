#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOCKER_RUNNER="${PROJECT_DIR}/docker/run_docker.sh"

usage() {
    cat <<'EOF'
Usage:
  scripts/run_replica_scene.sh <scene> [experiment_name] [gpu]

Examples:
  scripts/run_replica_scene.sh office0
  scripts/run_replica_scene.sh office0 ovo_mapping
  scripts/run_replica_scene.sh office0 ovo_mapping 1
  scripts/run_replica_scene.sh office0 ovo_mapping 0 --no-eval
  scripts/run_replica_scene.sh office0 ovo_mapping 0 -- --segment_every 5

Behavior:
  - Runs Replica scene through docker/run_docker.sh
  - Defaults to:
      experiment_name=replica_run
      gpu=0
      actions=--run --segment --eval
  - Extra args after '--' are forwarded to run_eval.py

Flags:
  --no-run       Skip --run
  --no-segment   Skip --segment
  --no-eval      Skip --eval
  -h, --help     Show this help
EOF
}

if [ $# -lt 1 ]; then
    usage
    exit 1
fi

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

SCENE="$1"
EXPERIMENT_NAME="${2:-replica_run}"
GPU="${3:-0}"
shift $(( $# >= 3 ? 3 : $# >= 2 ? 2 : 1 ))

RUN_FLAG="--run"
SEGMENT_FLAG="--segment"
EVAL_FLAG="--eval"
EXTRA_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --no-run)
            RUN_FLAG=""
            ;;
        --no-segment)
            SEGMENT_FLAG=""
            ;;
        --no-eval)
            EVAL_FLAG=""
            ;;
        --)
            shift
            EXTRA_ARGS=("$@")
            break
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            ;;
    esac
    shift
done

if [ ! -x "${DOCKER_RUNNER}" ]; then
    echo "Docker runner not found or not executable: ${DOCKER_RUNNER}" >&2
    exit 1
fi

CMD=(
    python
    run_eval.py
    --dataset_name Replica
    --experiment_name "${EXPERIMENT_NAME}"
    --scenes "${SCENE}"
)

if [ -n "${RUN_FLAG}" ]; then
    CMD+=("${RUN_FLAG}")
fi
if [ -n "${SEGMENT_FLAG}" ]; then
    CMD+=("${SEGMENT_FLAG}")
fi
if [ -n "${EVAL_FLAG}" ]; then
    CMD+=("${EVAL_FLAG}")
fi

if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    CMD+=("${EXTRA_ARGS[@]}")
fi

echo "Running Replica scene"
echo "  scene: ${SCENE}"
echo "  experiment: ${EXPERIMENT_NAME}"
echo "  gpu: ${GPU}"
echo "  command: ${CMD[*]}"

"${DOCKER_RUNNER}" "${GPU}" "${CMD[@]}"
