# Separate four-path AncestralPaths experiment

The new condition implements Supplementary Note 3.1 of [Irving-Pease et al.](https://www.nature.com/articles/s41586-023-06705-1): remove Bronze Age Anatolian (BAA) contributions from the genealogical nearest-neighbour distributions and remove their two path labels. The six-path experiment remains intact.

## What changes

- Recompute features from the same inferred trees. BAA leaves are ineligible as reference contributions before choosing informative ancestor shells and normalizing proportions. A shell containing only BAA leaves is skipped. Original tree topology and sample identities are preserved.
- Retain the original focal examples from the four non-BAA query cohorts. Their labels remain paths 1–4. Labels for paths 5–6 are rejected, never merged.
- Train a fresh CNN with input `1 × 5 × 8`, convolution with 32 filters and a `5 × 1` kernel, flatten, Dense64, and a four-output softmax. The model has 16,900 parameters. No six-class weights or optimizer state are reused.
- Retain the recovered author's five epochs, batch size 30 and Adam settings. With 40,000 training rows, a full fit performs 6,670 updates. Selection is the final fifth epoch.
- Score original focal IDs on five independent test simulations. The primary cohort is present-day GBR. Report ancestry accuracy and precision matrices as well as secondary metrics. Uncertainty is the sample SD across simulated test parents, not across training seeds.

## Data and remaining reconstruction choices

This experiment reuses three 200 Mb training simulations and five independent test simulations. It does not modify the simulation core or re-estimate genealogies. Supplementary Note 3.1 describes exclusion from GNN distributions and labels, not removal from the original Relate input.

Retaining the frozen non-BAA support gives 40,000 training examples, 10,000 per cohort, and 8,000 test examples per parent, including 2,000 GBR examples. Note 3.1 does not specify its own training/test counts, so these allocations are explicit reconstruction choices. The exact four-output historical source was not recovered. The layer dimensions are adapted from the recovered author CNN according to the stated four-path experiment.

The paper also compares its four-path classifier to the original six-output classifier on the non-BAA population. That comparison must retain all six possible predictions for the six-output model. Summing or renormalizing six probabilities into four is a different analysis.

## Prepared execution

[The separate plan](../../configs/benchmarks/paper_ancestralpaths_four_20261027.json) contains 15 CPU tasks: eight observed-feature preparation stages, one fresh fit, five prediction stages and one scoring stage. The existing qualified TensorFlow CPU environment is sufficient for this small CNN. Preparation uses the existing tskit runtime. Full output reservations total less than 0.5 GiB.

The plan is additive and is not inserted into the live campaign. It does not restart or change existing jobs. It schedules the AncestralPaths baseline only. A four-output TreeLoom fit on the same support is still required for a new comparative result.

To inspect the exact feature/model definition on anogena:

```bash
cd /workspace-SR003.nfs2/ARG-GNN
. scripts/project_env.sh
export CUDA_VISIBLE_DEVICES=
.runtime/env/bin/python3.11 -B -s -m arggnn.benchmarking.paper_ancestralpaths_four describe
```

The task factories are compatible with `arggnn.benchmarking.paper_campaign.runtime`. The campaign must resolve each dependency reference and apply resource reservations before dispatch. The plan command itself does not launch training.

Implementation: [four-path package](../../src/arggnn/benchmarking/paper_ancestralpaths_four). The unchanged original code and the six-output results remain separate.

## Verification

All 33 CPU checks passed: 12 feature/label tests, 15 workflow tests, one complete task-plan/factory check, and five model/runtime tests. Model tests include a synthetic update and exact saved/restored probabilities and optimizer state. No GPU or real TEST truth was used. [Verification receipt](verification.json), [workflow log](workflow_tests.log), [model log](model_tests.log).
