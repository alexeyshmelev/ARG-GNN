#!/usr/bin/env bash
# Explicit root-only VAL000 counts operation.
set -euo pipefail
if [[ $# -ne 1 || "$1" != "--execute" ]]; then
  echo "Usage: $0 --execute" >&2
  exit 2
fi
cd /workspace-SR003.nfs2/ARG-GNN
source scripts/project_env.sh
export CUDA_VISIBLE_DEVICES=''
task_source="research/reviews/C3_native_VAL000_counts_namespace_root_20261015_v1"
"$ARGGNN_PYTHON" -B "$task_source/root_native_driver_v1.py" --activate
flock -x research/reviews/American_C3_short_train2mb_wave_source_v1/scheduler/admission.lock "$ARGGNN_PYTHON" -B "$task_source/root_native_driver_v1.py" --admit
setsid --wait "$ARGGNN_PYTHON" -B "$task_source/root_native_driver_v1.py" --launch counts
