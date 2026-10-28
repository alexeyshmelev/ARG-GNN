# European scoring and historical-comparison audit

The current four-method comparison is arithmetically valid on common TEST targets. No saved-row, parent, class-order, or same-site truth mismatch was found. It is a comparison of complete pipelines; the inputs and optimizer schedules differ. The earlier 97.08% result does not establish that the current fit should beat the recovered CNN.

## Independent current scoring verification

All 20 prediction metadata and NPZ hashes were authenticated before reading truth. Each parent uses an identical immutable registry across all four methods. Every file contains canonical indices 0..9999, and the CNN/GST global example IDs match parent:index exactly. All native-six probability columns have the declared order Ana_nonBAA, CHG_nonBAA, WHG, EHG, Ana_viaBAA, CHG_viaBAA. Dense and TreeLoom NPZ files do not contain global example_ids; their authenticated parent metadata and exact canonical indices supply that binding.

The audit independently joined sample ID/haplotype identities and original physical positions to the generation-authenticated truth, without importing the production scorer or truth join. All 240 combinations of method, parent, overall/cohort and native-six/C4 endpoint reproduced exact confusion matrices, accuracy and macro F1. Equal-parent means and sample SD also match. Only 209 selected native truth interval rows per parent were extracted with bounded streaming; the duplicate source-class truth member was never opened.

| Method | Native-six accuracy % ± SD | Derived C4 accuracy % ± SD |
|---|---:|---:|
| Historical CNN architecture | 92.702 ± 0.352 | 95.844 ± 0.392 |
| Released Dense control | 92.490 ± 0.533 | 95.704 ± 0.491 |
| TreeLoom | 54.802 ± 1.024 | 70.832 ± 1.199 |
| GST LAI eight-tree adaptation | 42.448 ± 1.133 | 57.860 ± 1.040 |

The C4 calculation sums native probabilities as (p0+p4,p1+p5,p2,p3) before argmax. It is not four-class retraining. Macro F1 includes all nominal endpoint classes even when some have zero support in a cohort.

Current TRAIN receipts also agree on the three TRAIN parents, sampled registries and exact supervised target metadata hashes across all four methods. CNN and Dense share exact feature files. These TRAIN parents are disjoint from the five TEST parents. Full target-array and model-lineage checks are covered by the companion model/input audits.

There is a real degenerate prediction pattern in the saved TreeLoom probabilities: across all 50,000 TEST records, native argmax counts are [28068,8020,0,13910,2,0]. WHG and CHG_viaBAA are never selected; BAA accuracy is zero. This is not a metric-unit error. Collapsing to C4 and restricting to present-day queries still gives TreeLoom 68.17% versus CNN 93.20%, so native-six versus C4 or cohort aggregation alone does not explain the failure.

## Why the earlier wins are not the same experiment

| Condition | Earlier headline comparison | Current European comparison |
|---|---|---|
| Headline | H050 TreeLoom 97.0778%, ARGMix 94.4461%, AP derived C4 93.7830% | CNN 92.702%, Dense 92.490%, TreeLoom 54.802%, GST 42.448% |
| Evaluation parents | Eight reused development VAL simulations | Five separately generated TEST parents |
| Training parents and length | 32 TRAIN parents, 2 Mb each | Three TRAIN parents, 200 Mb each |
| Evaluation length | 2 Mb each | 200 Mb each |
| Headline support | Present182 physical-span accuracy | 10,000 sampled observed-SNP/query rows per parent; 2,000 per each of five cohorts |
| Output endpoint | Four source ancestries; AP six-class probabilities collapsed before argmax | Native six census-path classes; derived C4 reported separately |
| AP comparator identity | Released Dense workflow, not the recovered historical CNN | Both historical CNN architecture and released Dense control |
| AP training | 5,190,874 rows, five epochs, batch 32, final epoch 5 | 50,000 rows, five epochs; Dense batch 32/7,815 updates, CNN batch 30/8,335 updates |
| TreeLoom training | H045 warmstart, 2,880 earlier + 960 new updates; exact-span loss | Fresh current six-output fit; 30 passes/90 whole-parent updates, sampled targets |
| TreeLoom selection | Best reused-VAL Present182 span accuracy from passes 5/10/15/20/25/30 | Final pass 30 under the frozen current schedule |
| Tree source | Observed Relate post-size inferred trees | Observed Relate post-size inferred trees |

The historical H050 scientific completion explicitly lists ap_expanded_train_000..031 and ap_expanded_val_000..007, 2,000,000 bp, 952 haplotypes, 686 queries, exact-span loss, one optimizer seed 101 and the VAL selection rule. Its highest stable-leaf score uses 3,840 cumulative model-lineage updates. The current native-six comparison uses ancestralpaths_european_train_000..002 and ancestralpaths_european_test_000..004. Parent identities, length, supervision density, endpoint, fit lineage and checkpoint selection therefore changed together.

The historical ARGMix comparator used its native SNP inputs, 11 epochs with early stopping, selected epoch 9 and seed 50. It is absent from the current four-method European table. An assertion that the current TreeLoom now loses to ARGMix is unsupported by that table.

Both examined historical and current inference receipts bind the same Relate 1.2.4 binary SHA and post-five-iteration pipeline. Their actual trees and simulations differ. There is no evidence here for a change from true trees to inferred trees, or from Relate to tsinfer, as the explanation.

The earlier matched continuation already reached 97.0000%. Stable-leaf memory added only 0.0778 percentage points while lowering boundary F1 from 19.0488% to 18.3807%; both registered utility criteria failed. The old cross-method advantage was a descriptive development result and did not isolate an architectural contribution.

## Interpretation and limits

The common-target comparison is valid. A matched-input architectural claim is not: the input audit confirms that CNN/Dense use eight observed ancient-cohort channels, while the current TreeLoom leaf labels collapse all five admixed query cohorts into unknown (retaining age, topology and genotypes). Observed cohort identity is legitimate metadata, not local-ancestry truth leakage. This asymmetry may contribute, but it is not a proven sole cause; the missing WHG predictions also require explanation.

No evidence of TEST label mixing was found in the scoring contracts, shared truth join, or saved prediction identities. Prediction metadata declare truth-free operation, and production scoring authenticates all predictions before truth. This audit did not replay prediction computations or record operating-system file access, so it does not independently prove every internal feature-to-row operation or universal absence of training leakage.

The audit wrote only this directory. No training, prediction generation, restarts, queue edits, signals or Git actions were performed. Root was notified that the README sentence saying GST is still absent from the interim table is stale relative to the completed four-method product.

Evidence: [independent scoring](independent_scoring_evidence.json), [audit source](independent_scoring_audit.py), [training and historical source bindings](training_and_history_evidence.json). The historical identity is independently stated in research/reviews/development_confusion_20261016_v1/REPORT.md and docs/biological_comparison_scope.md; the exact H050 completion is experiments/results/H050_CPU_full192_savedfit_failedCLI_root_v1/completed.json.
