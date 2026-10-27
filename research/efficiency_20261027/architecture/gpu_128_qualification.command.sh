#!/usr/bin/env bash
# PREPARED ONLY. Root owns GPU admission and external resource supervision.
# The caller must already export the admitted physical GPU4 UUID and policy.
set -euo pipefail
cd /workspace-SR003.nfs2/ARG-GNN
source scripts/project_env.sh
test "${ARGGNN_PHYSICAL_GPU:?Root must admit the physical GPU}" = "4"
test "${ARGGNN_GPU_MEMORY_POLICY:?Root must set the existing policy}" = "full_device"
test -n "${CUDA_VISIBLE_DEVICES:?Root must select the admitted GPU UUID}"
PYTHONPATH='/workspace-SR003.nfs2/ARG-GNN:/workspace-SR003.nfs2/ARG-GNN/src' \
  .runtime/env/bin/python -B -m research.efficiency_20261027.architecture.gpu_128_qualification \
  --output research/efficiency_20261027/architecture/gpu_128_qualification_attempt01 \
  --max-rss-gib 32 \
  --max-gpu-reserved-gib 24 \
  --max-seconds 1200 \
  --tree-block 16 \
  --haplotype-tile 16 \
  --scan-block 32 \
  --warm-repeats 1
