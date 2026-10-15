#!/usr/bin/env bash
# Root invokes one explicit action after reviewing HANDOFF.md.
set -euo pipefail
cd /workspace-SR003.nfs2/ARG-GNN
source scripts/project_env.sh
export PYTHONDONTWRITEBYTECODE=1
packet=research/reviews/C3_native_postMLM980_root_source_20261015_v1
driver="$packet/root_native_driver_v1.py"
mlm=experiments/results/American_C3_native_MLM_full30mb_mlm_fit_v8/completed.json
fit=experiments/results/American_C3_native_train000_postMLM980_fit_v1/completed.json
fit_wait="$packet/admitted_fit_v1/actual_wait.json"
case "${1:-help}" in
  source-check)
    [[ $# == 1 ]]
    exec "$ARGGNN_PYTHON" -B "$packet/verify_source_v1.py"
    ;;
  bind-fit)
    [[ $# == 6 ]] # action GPU UUID genuine_MLM_SHA actual_MLM_root_wait_path wait_SHA
    exec "$ARGGNN_PYTHON" -B "$driver" --bind fit --gpu "$2" --gpu-uuid "$3" \
      --MLM "$mlm" --MLM-sha256 "$4" --MLM-wait "$5" --MLM-wait-sha256 "$6"
    ;;
  bind-ram)
    [[ $# == 8 ]] # action GPU UUID MLM_SHA MLM_wait_path MLM_wait_SHA actual_fit_SHA fit_wait_SHA
    exec "$ARGGNN_PYTHON" -B "$driver" --bind embeddings --gpu "$2" --gpu-uuid "$3" \
      --MLM "$mlm" --MLM-sha256 "$4" --MLM-wait "$5" --MLM-wait-sha256 "$6" \
      --fit "$fit" --fit-sha256 "$7" --fit-wait "$fit_wait" --fit-wait-sha256 "$8"
    ;;
  admit-fit) [[ $# == 1 ]]; exec "$ARGGNN_PYTHON" -B "$driver" --admit fit ;;
  admit-ram) [[ $# == 1 ]]; exec "$ARGGNN_PYTHON" -B "$driver" --admit embeddings ;;
  launch-fit) [[ $# == 1 ]]; exec setsid --wait "$ARGGNN_PYTHON" -B "$driver" --launch fit ;;
  launch-ram) [[ $# == 1 ]]; exec setsid --wait "$ARGGNN_PYTHON" -B "$driver" --launch embeddings ;;
  close-fit) [[ $# == 3 ]]; exec "$ARGGNN_PYTHON" -B "$driver" --close fit --root-approval "$2" --expected-root-approval-sha256 "$3" ;;
  close-ram) [[ $# == 3 ]]; exec "$ARGGNN_PYTHON" -B "$driver" --close embeddings --root-approval "$2" --expected-root-approval-sha256 "$3" ;;
  *)
    cat <<'USAGE'
Root-only conditional sequence:
  bash ROOT_COMMANDS.sh source-check
  bash ROOT_COMMANDS.sh bind-fit GPU UUID MLM_SHA MLM_ROOT_WAIT_PATH MLM_ROOT_WAIT_SHA
  bash ROOT_COMMANDS.sh admit-fit
  bash ROOT_COMMANDS.sh launch-fit
After root personally consumes genuine supervisor terminal wait0, complete close approval metadata and close-fit.
After genuine fit completion/wait0:
  bash ROOT_COMMANDS.sh bind-ram GPU UUID MLM_SHA MLM_ROOT_WAIT_PATH MLM_ROOT_WAIT_SHA FIT_SHA FIT_ROOT_WAIT_SHA
  bash ROOT_COMMANDS.sh admit-ram
  bash ROOT_COMMANDS.sh launch-ram
After root personally consumes genuine supervisor terminal wait0, complete close approval metadata and close-ram.
All hashes and UUIDs must be actual. Existing genuine counts refs are pinned defaults.
USAGE
    ;;
esac
