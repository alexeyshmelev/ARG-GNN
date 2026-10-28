#!/usr/bin/env bash
# Two independent CPU arms of the registered bounded learning diagnostic.
# No native baseline, production queue, GPU or original scientific input changes.
set -euo pipefail
cd /workspace-SR003.nfs2/ARG-GNN
source scripts/project_env.sh
export PYTHONPATH=src:.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=''
run_dir=research/redesign_20261028/root/learning_runs
mkdir -p "$run_dir"
mkdir "$run_dir/pair_started"
manifest=research/redesign_20261028/diagnostic_packets/manifest.json
.runtime/env/bin/python3.11 -B research/redesign_20261028/root/learning_diagnostic.py --arm no_cross --manifest "$manifest" > "$run_dir/no_cross.log" 2>&1 &
local_pid=$!
printf '%s\n' "$local_pid" > "$run_dir/no_cross.pid"
.runtime/env/bin/python3.11 -B research/redesign_20261028/root/learning_diagnostic.py --arm structure --manifest "$manifest" > "$run_dir/structure.log" 2>&1 &
structure_pid=$!
printf '%s\n' "$structure_pid" > "$run_dir/structure.pid"
local_exit=0
if wait "$local_pid"; then
    local_exit=0
else
    local_exit=$?
fi
structure_exit=0
if wait "$structure_pid"; then
    structure_exit=0
else
    structure_exit=$?
fi
printf 'no_cross_exit=%s\nstructure_exit=%s\n' "$local_exit" "$structure_exit" > "$run_dir/pair_exit.txt"
if (( local_exit != 0 || structure_exit != 0 )); then
    exit 1
fi
