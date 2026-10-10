#!/usr/bin/env bash
set -euo pipefail
cd /workspace-SR003.nfs2/ARG-GNN
export ARGMIX_NATIVE_PROFILE=cu129
source /workspace-SR003.nfs2/ARG-GNN/configs/official_argmix_native_environment_v1/containment.env.sh
/workspace-SR003.nfs2/ARG-GNN/.runtime/official_argmix_abi210_v1/cu129/env/bin/python /workspace-SR003.nfs2/ARG-GNN/experiments/results/argmix_native_abi210_execution_v1/v6/build/native_torchscript_cpu_probe.py /workspace-SR003.nfs2/ARG-GNN/.runtime/official_argmix_abi210_v1/cu129/admission/v6
