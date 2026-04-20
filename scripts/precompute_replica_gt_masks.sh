#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPLICA_ROOT="${REPLICA_ROOT:-/ws/data/replica_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-/ws/data/OVO/working/masks/replica_gt_instance}"
MAX_DISTANCE="${MAX_DISTANCE:-0.05}"
MIN_AREA="${MIN_AREA:-20}"

if [ $# -eq 0 ]; then
    echo "Usage: bash scripts/precompute_replica_gt_masks.sh <scene> [<scene> ...]"
    exit 1
fi

cd "${PROJECT_DIR}"
for scene in "$@"; do
    python scripts/precompute_replica_gt_masks.py \
        --scene "${scene}" \
        --replica_root "${REPLICA_ROOT}" \
        --output_dir "${OUTPUT_DIR}" \
        --max_distance "${MAX_DISTANCE}" \
        --min_area "${MIN_AREA}"
done
