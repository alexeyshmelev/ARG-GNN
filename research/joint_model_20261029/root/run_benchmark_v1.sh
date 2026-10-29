#!/usr/bin/env bash
set -u
cd /workspace-SR003.nfs2/ARG-GNN
source scripts/project_env.sh
export PYTHONPATH=.runtime/vendor/nvml_profile_20261028:src:.
arm="$1"
case "$arm" in no_cross|leaf_only|structure) ;; *) exit 2 ;; esac
CUDA_VISIBLE_DEVICES=GPU-3db85505-d061-e45d-d154-e7e8c1cad497 .runtime/env/bin/python3.11 -B research/joint_model_20261029/root/train.py --protocol research/joint_model_20261029/root/benchmark_protocol_v1.json --arm "$arm" --stage benchmark --gpu 5 --run-name "benchmark_${arm}_v1" > "research/joint_model_20261029/root/benchmark_${arm}_v1.log" 2>&1
code=$?
printf '%s\n' "$code" > "research/joint_model_20261029/root/benchmark_${arm}_v1.exit"
exit "$code"
