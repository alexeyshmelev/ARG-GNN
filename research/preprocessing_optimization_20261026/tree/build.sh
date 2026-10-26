#!/usr/bin/env bash
set -euo pipefail
cd /workspace-SR003.nfs2/ARG-GNN
# Atomic publication preserves any already loaded shared-library mapping.
c++ -O3 -std=c++17 -fPIC -shared -fno-fast-math -o research/preprocessing_optimization_20261026/tree/libstructural.build.so research/preprocessing_optimization_20261026/tree/kernels.cpp
mv research/preprocessing_optimization_20261026/tree/libstructural.build.so research/preprocessing_optimization_20261026/tree/libstructural.so
