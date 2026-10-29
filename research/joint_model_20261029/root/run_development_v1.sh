#!/usr/bin/env bash
set -u
cd /workspace-SR003.nfs2/ARG-GNN
source scripts/project_env.sh
export PYTHONPATH=.runtime/vendor/nvml_profile_20261028:src:.
run_arm() {
    local arm="$1"
    local gpu="$2"
    local gpu_uuid="$3"
    CUDA_VISIBLE_DEVICES="$gpu_uuid" .runtime/env/bin/python3.11 -B research/joint_model_20261029/root/train.py --protocol research/joint_model_20261029/root/development_protocol_v1.json --arm "$arm" --stage development --gpu "$gpu" --run-name "development_${arm}_v1" > "research/joint_model_20261029/root/development_${arm}_v1.log" 2>&1
    local code=$?
    printf '%s\n' "$code" > "research/joint_model_20261029/root/development_${arm}_v1.exit"
    return "$code"
}
run_arm no_cross 4 GPU-6cf64ff3-d8c2-9201-b2b8-683029625d71 &
pid_no_cross=$!
run_arm leaf_only 4 GPU-6cf64ff3-d8c2-9201-b2b8-683029625d71 &
pid_leaf_only=$!
run_arm structure 5 GPU-3db85505-d061-e45d-d154-e7e8c1cad497 &
pid_structure=$!
printf '%s\n' "no_cross=$pid_no_cross leaf_only=$pid_leaf_only structure=$pid_structure"
wait "$pid_no_cross"
code_no_cross=$?
wait "$pid_leaf_only"
code_leaf_only=$?
wait "$pid_structure"
code_structure=$?
printf '%s\n' "exit no_cross=$code_no_cross leaf_only=$code_leaf_only structure=$code_structure"
if (( code_no_cross || code_leaf_only || code_structure )); then exit 1; fi
