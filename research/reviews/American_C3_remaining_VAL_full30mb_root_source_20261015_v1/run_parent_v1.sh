#!/usr/bin/env bash
# Root-only, fresh one-parent pipeline. Every failed command stops; no replay.
set -euo pipefail
if [[ $# -ne 2 || "$1" != "--execute-parent" ]]; then
  echo "Usage: $0 --execute-parent american_c3_val_001|002|003|004" >&2
  exit 2
fi
case "$2" in american_c3_val_001|american_c3_val_002|american_c3_val_003|american_c3_val_004) ;; *) exit 2 ;; esac
cd /workspace-SR003.nfs2/ARG-GNN
task_parent="$2"
task_source="research/reviews/American_C3_remaining_VAL_full30mb_root_source_20261015_v1"
for task_stage in simulate infer native_paths; do
  .runtime/env/bin/python -B "$task_source/root_packet_v1.py" --parent "$task_parent" --admit "$task_stage"
  .runtime/env/bin/python -B "$task_source/root_execution_v1.py" --parent "$task_parent" --launch "$task_stage"
  .runtime/env/bin/python -B "$task_source/close_completed_v1.py" --parent "$task_parent" --scope "$task_stage" --execute
done
