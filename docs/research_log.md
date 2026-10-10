# Research log

## 2026-10-10: initialization
- Cloned the empty user repository to /workspace-SR003.nfs2/ARG-GNN.
- Confirmed seven A100-SXM4-80GB devices. Project eligibility is restricted to free GPUs 2–5, with at most two active.
- Confirmed conda 24.11.3 at /home/jovyan/miniconda3/bin/conda.
- Started independent architecture, ARG benchmark and prior-art investigations.
- Research status: hypothesis formation. No training or validated model results yet.
- Initial evidence: GST already interleaves graph encoding and set feedback. Positional encoding alone would be insufficient evidence of novelty.
- Initial reproducibility finding: current ARGMix code differs from its April 2026 manuscript, requiring pinned implementations and explicit reproduction scope.

## Infrastructure and initial implementation
- Dedicated conda environment installed with CUDA PyTorch 2.6.0, msprime, tskit and NVTOP. Disabled user-site packages to prevent shared-environment leakage.
- Resource guards restrict jobs to free GPUs 2–5 and at most two concurrent leases.
- A small GPU 2 probe confirmed the requested process title via NVIDIA process reporting. NVTOP verification will be captured during a training job.
- Native PyTorch graph models, independent temporal controls and migration-derived simulation labels implemented.
- End-to-end CPU correctness tests passed.
- Versioned oracle pilot protocol frozen before model training. Its role is numerical/engineering validation and headroom assessment, not a final novelty claim.

## Completed oracle engineering pilot
- Two training seeds completed seven model/control variants each on the frozen 16/4/8 parent-replicate panel.
- NVTOP captured both GPU process names on physical GPUs 2 and 3.
- The nearest-TMRCA comparator scored 98.10% on the uniform held-out grid. The initial neural encoders were much weaker and often predicted nearly constant classes. This pilot does not establish a useful proposed-model gain.
- Next step: diagnose reference-information reachability, add stronger genealogy-conditioned local controls, and assess the separate four-source ancient-European generator.
- A fresh-context code audit led to fixes for switch diagnostics, checkpoint uniqueness, provenance checks, storage guards, and inherited GPU leases.
- Corrected full-resolution diagnostics changed 7 of 28 cases, from 749 to 759 switches in total. Original labels/tensors were preserved and their hashes verified.

## Fixed-anchor plain pilot and control integration, 2026-10-10

The nine-model plain inferred Q64 pilot completed on the reused 16 TRAIN/four VAL parents with seed101. Full depth2 selected NLL is 1.02564669, full depth1 1.03080856, scalar depth2 1.03870764, diagonal depth2 1.03976517 and gain-matched diagonal depth2 1.04072184. This is exploratory selection evidence. There is no final ancestry test or established novel superiority. The complete score and per-parent ledger is in research/reviews/anchor_plain_pilot_complete_v1.json.

Capacity/source-feedback and split-age interventions, canonicalization and direct-core Chickenpox adaptation are integrated in new independent modules. Their last combined integration verification passed 173 tests. Their training remains pending. The exhaustive label-free age-panel audit establishes non-scalar geometry and about94–95% physical-span source coverage, with no predictive implication. The fresh-development/confirmation protocol is a proposal requiring concrete execution and analysis locks. No confirmatory parents have been generated.

Relate authorization was reaffirmed by the user: “I reviewed, use it.” Credentials remain external to scientific artifacts.
