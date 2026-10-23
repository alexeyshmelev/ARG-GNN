#!/usr/bin/env bash
set -euo pipefail
cd /workspace-SR003.nfs2/ARG-GNN
. scripts/project_env.sh
export PYTHONPATH="$ARGGNN_ROOT/src:$ARGGNN_ROOT"
exec .runtime/env/bin/python -B -s -u -m arggnn.benchmarking.paper_campaign.scheduler --plan configs/benchmarks/paper_ap_cnn_execution_20261024.json
