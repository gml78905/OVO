#!/usr/bin/env bash
set -euo pipefail

# 어디서 실행하든 OVO 레포 루트를 빌드 컨텍스트로 사용 (docker/ 안에서 bash build.sh 해도 동작)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

# Dockerfile WORKDIR(/ws/external)는 docker/run_docker.sh 의 WORK_DIR 과 맞춰야 함
docker build -t ovo:cuda121 -f docker/Dockerfile .
