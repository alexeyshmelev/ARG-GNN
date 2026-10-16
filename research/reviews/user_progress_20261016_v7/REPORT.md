# Current progress

Updated UTC: 2026-10-16T19:08:22.420717+00:00.

The ten-page AAAI paper is written, compiled and visually checked, including its Overleaf bundle and architecture figure. The scientific evidence is still incomplete. We cannot yet claim overall top-1 or a submission-ready paper. There is substantial new training and data-preparation progress, but no new ancestry accuracy result since the last report.

## What moved forward

- **Native ARGformer reached 22,000 / 227,990 saved optimizer updates**, verified across all 77 AdamW parameter states. That is 15,500 updates beyond the preserved 6,500 checkpoint. Model, scheduler, callbacks and RNG state are present. The run then stopped because its controller failed to return a fresh authenticated reply before the deadline. The saved checkpoint is retained. A narrow controller-liveness repair is being developed, with no further training success assumed. [Verified checkpoint](../native_v10_stopped_checkpoint_progress_root_20261016_v2/checkpoint_progress.root_v2.json) · [Actual stopped-run evidence](../native_v10_ROOT_actual_preparation_20261016_v1/actual_tool_terminal.root_v1.json).
- **All five original American 30 Mb validation genealogies have completed inference.** The last, VAL004, completed and was closed successfully. VAL003 native-path preparation has been launched. Its packet contains a genuine launch receipt but no worker progress or completion receipt yet, so completed focal preparation is not claimed. The original VAL002 GNN plan also completed its ordinary successful close. VAL001 prediction preparation has started independently. VAL004 native paths and the VAL003/004 GNN observation plans still need completion. [VAL004 completion](../../../experiments/results/American_C3_val004_full30mb_infer_native14v3_v3/completed.json) · [VAL003 launch](../American_C3_VAL003004_downward8GB_source_20261016_v1/admitted_val003_native_paths_down8_v1/worker_aux/launch.json).
- **Our American GNN's six saved checkpoints are now accepted for evaluation.** The dedicated validator succeeded and verified all six states through 600 updates and 30 epochs. The existing acceptance gate passed. The original training invocation's recorded exit1 remains preserved. This removes the saved-model readiness obstacle, without establishing ancestry accuracy. VAL000 CPU prediction admission has now completed and the exact full30Mb stage was launched once for all six saved checkpoints. Model worker completion and ancestry scoring remain pending. [Acceptance result](../C3_saved600_validator_ROOT_actual_20261016_v1/artifact_acceptance_gate.root_actual_v1.json) · [Actual prediction admission](../C3_full30Mb_eval_ROOT_predict_val_000_root01_oct16/admitted.json) · [Prediction launch](../C3_full30Mb_eval_ROOT_v2_activation_20261016_v1/predict_val_000_execution_intent.root_v1.json).

## Actual measured comparison

European results below use common physical-span support on eight reused 2 Mb development validation simulations, with one optimizer seed. Values are mean accuracy% ± sample SD across simulations. Present182, Neo302, Yam26 and All686 are the current fixed cohorts.

| Method | Present182 | Neolithic302 | Yamnaya26 | All686 |
|---|---:|---:|---:|---:|
| TreeLoom | **97.0778 ± 0.5674** | **99.7335 ± 0.0682** | **99.9366 ± 0.1229** | **98.8935 ± 0.2345** |
| TreeLoom-NoMem | 97.0000 ± 0.4978 | 99.6468 ± 0.0595 | 99.9072 ± 0.1399 | 98.8258 ± 0.2215 |
| Current author ARGMix | 94.4461 ± 1.1491 | 99.2955 ± 0.2623 | 99.2218 ± 0.8966 | 97.6130 ± 0.4975 |
| Released Dense AncestralPaths, derived C4 output | 93.7830 ± 0.8385 | 97.5693 ± 0.2748 | 95.4303 ± 1.4096 | 95.8071 ± 0.4473 |

TreeLoom's Present182 accuracy is 2.6317 percentage points above current ARGMix on this development panel. This compares complete pipelines with disclosed input, training and selection differences. It does not isolate architecture, reproduce the original published tables, or establish an independent test gain.

The matched memory comparison is more modest: TreeLoom gains 0.0778 accuracy points over NoMem, while boundary F1 falls from **19.0488 ± 1.4275%** to **18.3807 ± 1.3141%**. Both completed 960 additional updates after the same warm start. Each training history totals 3,840 updates. Selection used 3,840 for TreeLoom and 3,520 for NoMem. The higher-order triple-support variant failed both registered utility criteria, so useful new-operator novelty remains unproved.

American native FLARE has **98.0100% focal accuracy** and **97.7151% macro F1** on one original validation parent. It has no multi-parent SD yet. Native ARGformer and our American GNN still have no ancestry score, so their ranking is unknown. European span accuracy and American focal accuracy must not be ranked together.

[Every setup, all seven cohorts and NLL/F1/boundary/calibration metrics](../../current_scoreboard.md) · [Method and label differences](../../../docs/biological_comparison_scope.md).

## What remains for the paper

1. Finish the shared American comparison across all original 30 Mb validation parents, including native ARGformer, FLARE and our GNN, with the same queries, truth and scored support. Our saved American GNN used twenty independent 2 Mb TRAIN parents. Its full30Mb evaluation will therefore be a disclosed pipeline comparison, not a matched full30Mb-training architecture ablation. All five parents and all six saved checkpoint candidates must be evaluated before fixed selection and final scoring.
2. Complete the larger 1,611-diploid ARGMix reconstruction, training and common-data scoring. Original published-table reproduction remains unproved because the original table inputs and historical code/configuration archive are unavailable. Preserve author algorithms and disclose the actual reconstruction.
3. Run the reserved independent European test and report uncertainty with its proper unit. Current headline SD measures variation across reused validation simulations, not variation across training seeds.
4. Establish a useful architectural advance. Whole-tree graph processing, tracked haplotypes and genomic context are implemented, but the current extra higher-order mechanism has not earned a novelty or utility claim. The registered same-marker alignment experiment remains pending.
5. Strengthen biological evidence. Real-data predictions and PCA exist, but have no ancestry truth. The Danish genotype contrasts did not support the prespecified positive-direction hypotheses. They cannot serve as independent confirmation of ancestry accuracy.

The paper now has a coherent biological motivation, final-model description, matched memory ablation and an editable diagram linking an actual ARG to its matching marginal trees and implemented modules. Fresh readers informed the figure corrections, and all ten pages and the extracted Overleaf build were checked. The remaining work is primarily experimental and scientific, rather than manuscript formatting.

[Current PDF](../../../paper/current_full_panel_working.pdf) · [Overleaf source bundle](../../../paper/current_full_panel_overleaf_working.zip) · [Paper build and visual verification](../paper_scientific_rewrite_20261016_v3/final_integration.root_v1.json) · [ARGformer downstream-evaluator fidelity](../ARGformer_Table1_downstream_resolution_20261016_v1/RESOLUTION.md).
