#!/usr/bin/env bash
set -euo pipefail
cd /workspace-SR003.nfs2/ARG-GNN
source scripts/project_env.sh
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/workspace-SR003.nfs2/ARG-GNN/research/sliding_graph_convolution_20261028/tmp
.runtime/env/bin/python -B research/sliding_graph_convolution_20261028/test_operator.py
