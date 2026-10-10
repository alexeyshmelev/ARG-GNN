# Research log

## 2026-10-10: initialization
- Cloned the empty user repository to /workspace-SR003.nfs2/ARG-GNN.
- Confirmed seven A100-SXM4-80GB devices. Project eligibility is restricted to free GPUs 2–5, with at most two active.
- Confirmed conda 24.11.3 at /home/jovyan/miniconda3/bin/conda.
- Started independent architecture, ARG benchmark and prior-art investigations.
- Research status: hypothesis formation. No training or validated model results yet.
- Initial evidence: GST already interleaves graph encoding and set feedback. Positional encoding alone would be insufficient evidence of novelty.
- Initial reproducibility finding: current ARGMix code differs from its April 2026 manuscript, requiring pinned implementations and explicit reproduction scope.
