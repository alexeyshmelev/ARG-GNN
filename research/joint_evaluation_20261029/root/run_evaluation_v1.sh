#!/usr/bin/env bash
set -u
cd /workspace-SR003.nfs2/ARG-GNN
source scripts/project_env.sh
export PYTHONPATH=.runtime/vendor/nvml_profile_20261028:src:.
run_parent() {
    local parent="$1"
    local gpu="$2"
    local gpu_uuid="$3"
    CUDA_VISIBLE_DEVICES="$gpu_uuid" .runtime/env/bin/python3.11 -B -m research.joint_evaluation_20261029.prediction.predict --parent "$parent" --gpu "$gpu" --freeze research/joint_evaluation_20261029/prediction/freeze.json > "research/joint_evaluation_20261029/root/${parent}.log" 2>&1
    local code=$?
    printf '%s\n' "$code" > "research/joint_evaluation_20261029/root/${parent}.exit"
    return "$code"
}
lane_five() {
    run_parent ancestralpaths_european_test_000 5 GPU-3db85505-d061-e45d-d154-e7e8c1cad497 || return $?
    run_parent ancestralpaths_european_test_002 5 GPU-3db85505-d061-e45d-d154-e7e8c1cad497 || return $?
    run_parent ancestralpaths_european_test_004 5 GPU-3db85505-d061-e45d-d154-e7e8c1cad497
}
lane_four() {
    run_parent ancestralpaths_european_test_001 4 GPU-6cf64ff3-d8c2-9201-b2b8-683029625d71 || return $?
    run_parent ancestralpaths_european_test_003 4 GPU-6cf64ff3-d8c2-9201-b2b8-683029625d71
}
lane_five &
pid_five=$!
lane_four &
pid_four=$!
printf '%s\n' "lane_gpu5=$pid_five lane_gpu4=$pid_four"
wait "$pid_five"
code_five=$?
wait "$pid_four"
code_four=$?
printf '%s\n' "prediction_exit_gpu5=$code_five prediction_exit_gpu4=$code_four"
if (( code_five || code_four )); then exit 1; fi
.runtime/env/bin/python3.11 -B research/joint_evaluation_20261029/scoring/common_scoring.py seal --protocol research/joint_evaluation_20261029/scoring/scoring_protocol_v2.json --output research/joint_evaluation_20261029/scoring/prediction_barrier_v2.json || exit $?
.runtime/env/bin/python3.11 -B research/joint_evaluation_20261029/scoring/common_scoring.py score --protocol research/joint_evaluation_20261029/scoring/scoring_protocol_v2.json --seal research/joint_evaluation_20261029/scoring/prediction_barrier_v2.json --output research/joint_evaluation_20261029/scoring/comparison_v2.json
code_score=$?
printf '%s\n' "$code_score" > research/joint_evaluation_20261029/root/scoring_v2.exit
exit "$code_score"
