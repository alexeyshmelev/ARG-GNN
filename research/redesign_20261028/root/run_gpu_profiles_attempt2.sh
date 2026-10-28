#!/usr/bin/env bash
# Sequential fixed-batch qualification on authorized physical GPU 3 only.
set -euo pipefail
cd /workspace-SR003.nfs2/ARG-GNN
source scripts/project_env.sh
export PYTHONPATH=.runtime/vendor/nvml_profile_20261028:src:.
export CUDA_VISIBLE_DEVICES=GPU-df11de7f-7a15-ebe2-085d-9c7d907f3786
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
mkdir -p research/redesign_20261028/root/gpu_profiles
mkdir research/redesign_20261028/root/gpu_profiles/series_started_attempt2
for arm in no_cross structure; do
    for batch_size in 4 16; do
        for version in v1 v2; do
            .runtime/env/bin/python3.11 -B research/redesign_20261028/root/gpu_profile.py --version "$version" --arm "$arm" --batch-size "$batch_size" > "research/redesign_20261028/root/gpu_profiles/${arm}_${version}_b${batch_size}.attempt2.log" 2>&1
        done
    done
done
printf 'All eight profiles exited zero.\n' > research/redesign_20261028/root/gpu_profiles/series_completed_attempt2.txt
