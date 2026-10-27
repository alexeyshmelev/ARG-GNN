#!/usr/bin/env bash
# Prepared for ROOT review and dispatch only. This file has not been launched.
set -euo pipefail
cd /workspace-SR003.nfs2/ARG-GNN
source scripts/project_env.sh
export PYTHONPATH=src:.
"$ARGGNN_PYTHON" -B -m research.efficiency_20261027.root.run_guarded_profile_with_peer \
  --directory research/efficiency_20261027/root/runs/train001_preparation_phases_v1 \
  --peer-directory research/efficiency_20261027/root/runs/european_cohort8_full_gpu5_v2 \
  --peer-directory research/efficiency_20261027/root/runs/european_short_development_val_10mb_v1 \
  --gpu 5 --host-gib 128 --device-gib 0 --wait-seconds 3600 --run-seconds 3600 \
  -- "$ARGGNN_PYTHON" -B -m research.efficiency_20261027.independent_efficiency_challenge.preparation_phase_profile.profile_train001_v2 \
  --output /workspace-SR003.nfs2/ARG-GNN/research/efficiency_20261027/root/runs/train001_preparation_phases_v1/data \
  --source-pins-sha256 4fe7fc1fe20e027c69933f9e9af8d4d0f33061094411b8b1ba922f4aaad88fca
