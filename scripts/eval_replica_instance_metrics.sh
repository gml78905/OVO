#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PRED_DIR="${PRED_DIR:-/ws/data/OVO/output/Replica/ovo_mapping/instance_pred}"
REPLICA_ROOT="${REPLICA_ROOT:-/ws/data/replica_v1}"
EVAL_INFO="${EVAL_INFO:-/ws/data/OVO/working/configs/Replica/eval_info.yaml}"

if [ $# -eq 0 ]; then
    SCENES=(office0 office1 office2 office3 office4 room0 room1 room2)
else
    SCENES=("$@")
fi

cd "${PROJECT_DIR}"
python scripts/eval_replica_instance_metrics.py \
    --pred_dir "${PRED_DIR}" \
    --replica_root "${REPLICA_ROOT}" \
    --eval_info "${EVAL_INFO}" \
    --scenes "${SCENES[@]}"
