#!/usr/bin/env bash
# Root executes one reviewed explicit action. Default prints usage.
# Caller must externally flock the existing scheduler/admission.lock for admit/close.
# This wrapper and Python driver do not acquire that cooperative outer lock.
set -euo pipefail
cd /workspace-SR003.nfs2/ARG-GNN
source scripts/project_env.sh
export PYTHONDONTWRITEBYTECODE=1
packet=research/reviews/C3_GNN_prefetch_current980_root_20261015_v1
driver="$packet/root_driver_v1.py"
case "${1:-help}" in
 bind)
  [[ $# == 3 ]]
  exec "$ARGGNN_PYTHON" -B "$driver" --bind --inactive-request "$2" --expected-inactive-request-sha256 "$3"
  ;;
 admit)
  [[ $# == 2 ]]
  exec "$ARGGNN_PYTHON" -B "$driver" --admit "$2"
  ;;
 launch)
  [[ $# == 2 ]]
  exec setsid --wait "$ARGGNN_PYTHON" -B "$driver" --launch "$2"
  ;;
 close)
  [[ $# == 4 ]]
  exec "$ARGGNN_PYTHON" -B "$driver" --close "$2" --root-approval "$3" --expected-root-approval-sha256 "$4"
  ;;
 *)
  cat <<'USAGE'
The new canonical shim is installed with exact staged parity. Preserve every old source.
Copy the inactive request and caps to new actual root files. Fill all genuine refs,
bind the filled caps via caps_template_ref, and supply real2or4-worker bounds for TRAIN.
Never guess plan/wait/close refs, UUIDs or caps; inactive nulls cannot bind.
  bash ROOT_COMMANDS.sh bind ACTUAL_FILLED_REQUEST_PATH ACTUAL_SHA
  bash ROOT_COMMANDS.sh admit STAGE
  bash ROOT_COMMANDS.sh launch STAGE
Root must consume the genuine foreground terminal result before close.
  bash ROOT_COMMANDS.sh close STAGE ACTUAL_ROOT_APPROVAL_PATH ACTUAL_SHA
Stages: train; predict_val_000 through predict_val_004; matrix; score.
Optional observation-only plan stages: plan_val_000 through plan_val_004.
Native MLM is never a GNN dependency. TRAIN requires all20 actual plans.
VAL score requires all30 prediction seals and its separate private capability.
USAGE
  ;;
esac
