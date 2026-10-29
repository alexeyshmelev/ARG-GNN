#!/usr/bin/env bash
set -u
cd /workspace-SR003.nfs2/ARG-GNN || exit 1
source scripts/project_env.sh
export PYTHONPATH=.runtime/vendor/nvml_profile_20261028:src:.
export CUDA_VISIBLE_DEVICES=GPU-3db85505-d061-e45d-d154-e7e8c1cad497
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
for batch_sites in 16 8 32
do
    .runtime/env/bin/python3.11 -B research/joint_model_20261029/root/train.py --protocol research/joint_model_20261029/root/profile_protocol_v1.json --arm structure --stage profile --gpu 5 --run-name profile_b${batch_sites}_v1 --batch-sites "$batch_sites" > "research/joint_model_20261029/root/profile_b${batch_sites}_v1.log" 2>&1
    code=$?
    printf '%s\n' "$code" > "research/joint_model_20261029/root/profile_b${batch_sites}_v1.exit"
    if [ "$code" -ne 0 ]; then exit "$code"; fi
done
