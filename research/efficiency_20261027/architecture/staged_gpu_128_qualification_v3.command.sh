#!/usr/bin/env bash
# PREPARED ONLY. ROOT admits GPU5 after the current-model diagnostic terminates.
# ROOT must provide the external resource guardian and actual GPU5 UUID.
set -euo pipefail
cd /workspace-SR003.nfs2/ARG-GNN
source scripts/project_env.sh
test "${ARGGNN_PHYSICAL_GPU:?ROOT must admit physical GPU5}" = "5"
test "${ARGGNN_GPU_MEMORY_POLICY:?ROOT must set the existing policy}" = "full_device"
test -n "${CUDA_VISIBLE_DEVICES:?ROOT must select the admitted GPU5 UUID}"
PYTHONPATH='/workspace-SR003.nfs2/ARG-GNN:/workspace-SR003.nfs2/ARG-GNN/src' \
  .runtime/env/bin/python -B -m research.efficiency_20261027.architecture.staged_gpu_128_qualification_v3 \
  --output research/efficiency_20261027/architecture/staged_transport_gpu5_v1/data \
  --max-rss-gib 32 \
  --max-gpu-reserved-gib 24 \
  --max-seconds 1200
