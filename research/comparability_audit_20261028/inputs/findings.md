# European TreeLoom–AncestralPaths comparability audit

The 54.802% versus 92.702% native-six result compares two pipelines with the same sampled training labels and TEST targets, but different observable input representations and optimization schedules. The CNN receives the eight ancient cohort categories explicitly in its tree-shell features. TreeLoom retains sample ages, genotypes, all sampled leaves and inferred tree structure, but maps all five query cohorts to a common unknown reference class. This is a proven input disparity. Its causal share of the accuracy gap is not established by this audit.

This read-only audit follows the actual prediction references in `experiments/results/paper_gst_lai_8_20261024/paper_gst_lai_8__all_methods_all_test__score/native/comparison.json`. No training, prediction generation, scientific-data mutation, launch, or Git operation was performed. The only writes are audit scripts and reports in this directory. Numerical evidence and authenticated references are in `inputs_audit.json` (SHA256 `87d459a55c358a019600bb51f372b6b3c07064793325b3fb484ad8a253a98aa9`).

## Actual fit and prediction lineage

TreeLoom TEST predictions reference the completed fit under `experiments/results/benchmark_campaign_20261018_workflow/release_20261022_full_vram_v2/other/AP__all_train__treeloom_fit/native/`. Its executed job is `.runtime/benchmarking/workflow_20261022_full_vram_v2/jobs/AP__all_train__treeloom_fit.json`, using `research.repairs.benchmark_full_vram_20261022.training.ap_fit_v2`. The selected checkpoint is `checkpoints/epoch_30.pt`, SHA256 `c06fc82db27d21cb7933da07d4247692c18b5c96df9cb67cb0bff71453e3f5e3`.

CNN TEST predictions reference `experiments/results/paper_ancestralpaths_cnn_20261024/paper_ap_cnn__all_train__fit/native/fit.json`, SHA256 `4888c5aef7e59b05dc08562e4bcd56895bc1b21fa83dc103c4683a93ca09c51b`. The selected checkpoint is `historical_cnn_epoch5.h5`, SHA256 `0cb84b47cfa01f002c548f83682e6601d48a48052693be75ee3278b5f3b9bff1`.

All five TEST prediction metadata files were checked: each method consistently uses its one fit/checkpoint, both methods use the same sampled registry and inferred-tree reference for each parent, and the native-six class order matches. Source hashes attached to the TreeLoom prediction wrapper, recurrent executor, and static executor were verified. The CNN fit-job source pins were verified. The original TRAIN000 feature job, `.runtime/benchmarking/workflow_20261019_v8/jobs/AP__ancestralpaths_european_train_000__ap_features.json`, pins the inspected CNN bridge and feature executor hashes.

## Sampling and label exposure

The TRAIN parents are exactly `ancestralpaths_european_train_000`, `001`, and `002`. Their registry sizes are 16,667, 16,667, and 16,666: **50,000 examples**, not 30,000. For each parent, TreeLoom's executed scientific specification and CNN's actual feature/target products point to the identical sampled registry and native-six targets. CNN feature indices equal every registry index in order, and its saved training-example ID sequence exactly matches the concatenated 50,000 `parent:index` IDs.

Each query cohort contributes 10,000 TRAIN examples. All five TEST parents, `ancestralpaths_european_test_000` through `004`, contain 10,000 sampled examples apiece, with 2,000 per query cohort. Sampling-plan references match; the plan SHA256 is `2d39761d0cc030af72ba1e313be63dfce3fa7e5a4e1f4cecbdf33a5109fc7f5b`.

Native-six class order is `Ana_nonBAA, CHG_nonBAA, WHG, EHG, Ana_viaBAA, CHG_viaBAA`.

| TRAIN cohort | Ana_nonBAA | CHG_nonBAA | WHG | EHG | Ana_viaBAA | CHG_viaBAA |
|---|---:|---:|---:|---:|---:|---:|
| present_day | 3,775 | 2,462 | 1,295 | 2,468 | 0 | 0 |
| Bronze_Age | 3,822 | 2,478 | 1,212 | 2,488 | 0 | 0 |
| BAA | 0 | 0 | 0 | 0 | 7,658 | 2,342 |
| Yam | 0 | 4,776 | 0 | 5,224 | 0 | 0 |
| Neo | 7,463 | 0 | 2,537 | 0 | 0 | 0 |
| Total | 15,060 | 9,716 | 5,044 | 10,180 | 7,658 | 2,342 |

Both methods therefore receive the same supervised native-six labels. Saved metadata states that query ancestry truth is not opened during feature extraction or prediction. The inspected feature extractor consumes inferred-tree topology, observed cohort identity, and ancestor ages; it does not consume locus-specific query ancestry truth.

## Reference and cohort information

The observed panel has 952 haplotypes. Panel membership, roles, reference classes, and sample dates are identical across the three TRAIN and five TEST parents.

| Cohort | Haplotypes | TreeLoom reference class | Sample age, generations |
|---|---:|---|---:|
| present_day | 182 | unknown | 0 |
| Bronze_Age | 162 | unknown | 100–165 |
| BAA | 14 | unknown | 136–168 |
| Yam | 26 | unknown | 153–176 |
| Neo | 302 | unknown | 139–258 |
| WHG | 102 | WHG | 210–529 |
| EHG | 86 | EHG | 155–402 |
| Ana | 50 | Ana | 217–360 |
| CHG | 28 | CHG | 184–473 |

TreeLoom has 266 pure-reference haplotypes and 686 unknown query haplotypes. Its explicit leaf metadata is four pure-reference one-hot values, an unknown flag, and `log1p(sample_age)/log(10001)`. All query sample ages remain available, including BAA ages; their ranges overlap other ancient query cohorts. The selected checkpoint's last six leaf-projection column norms are approximately `[0.75647, 0.67614, 0.70855, 0.76348, 0.54162, 0.47524]`. Age therefore has a present, nonzero input path. This does not measure its learned causal effect on predictions.

TreeLoom's eight donor-frequency channels are **four ancestry classes multiplied by two age bins**, `<350` and `>=350` generations. Counts in `[Ana, CHG, WHG, EHG]` order are `[48,18,90,76]` and `[2,10,12,10]`. They are not eight observed-cohort categories. All four pure reference classes are present, including 102 WHG haplotypes. Whole-panel allele frequencies and entropy additionally include the complete panel.

The CNN feature tensor has shape `(5,9)`: five successive nonempty upward shells of newly encountered ancient leaves, with channels `Bronze_Age, BAA, Yam, Neo, WHG, EHG, Ana, CHG, ancestor_age/1500`. This uses 770 ancient leaves, including 504 ancient query-cohort leaves and 266 pure references; 182 present-day leaves are excluded from the shell fractions. The historical extraction convention includes an ancient focal sample itself. Its observed cohort can thus contribute to its features. All eight cohort channels have nonzero values in the actual cached TRAIN tensors: counts of positive shell entries are `[138297,39639,49642,135454,39776,51422,57617,38031]` in the stated feature order.

Relevant source locations:

- `research/repairs/benchmark_multigpu_20261022/scientific/queue_adapters/european_observations.py:205` assigns observed reference classes; query cohorts become unknown.
- `research/repairs/benchmark_multigpu_20261022/scientific/queue_adapters/european_execution.py:122` constructs the four one-hots, unknown flag, and sample-age input.
- `research/reviews/H060_isolated_hold_compatibility_source_20261017_v1/observations_v1.py:18` constructs the age-by-ancestry donor channels.
- `src/arggnn/ancestralpaths_native_bridge.py:254` specifies the cohort-shell features, self-inclusion, age normalization, and padding.
- `src/arggnn/benchmarking/european_execution.py:186` invokes the CNN extractor with the original integer age convention and float64 repeated-add arithmetic before float32 storage.
- `research/reviews/author_repository_deep_audit_20261023_v1/ancestralpaths/history_source/previous_training.py:35` and `:149` corroborate the historical cohort categorization and shell representation.

## Why the native-six endpoint is sensitive to this difference

The truth-only `paths_from_census_populations` routine at `src/arggnn/ancestralpaths_native_bridge.py:393` maps the four source ancestries to six path labels by using the focal sample's observed BAA cohort: Ana and CHG for BAA become the two viaBAA classes. The actual training targets above verify that all BAA examples use those two labels and no other cohort does.

Observed cohort identity is therefore directly relevant to native-six label naming. The CNN gets this information in its feature representation; TreeLoom does not get an explicit categorical query-cohort input. This is observable sample metadata, not locus-specific ancestry truth. Whether either representation should be used is an experimental design choice that must be made consistently when claiming an architecture comparison.

## Optimization and checkpoint selection

| Item | TreeLoom | Historical CNN reconstruction |
|---|---|---|
| Sampled TRAIN rows | Same 50,000 | Same 50,000 |
| Epochs | 30 | 5 |
| Optimizer updates | 90 | 8,335 |
| Update granularity | One full-parent sampled mean CE update | Minibatch size 30 |
| Total label presentations | 1,500,000 | 250,000 |
| Optimizer | AdamW | Keras legacy Adam |
| Learning rate / betas | 0.001 / 0.9, 0.999 | 0.001 / 0.9, 0.999 |
| Epsilon / weight decay | 1e-8 / 1e-4 | 1e-7 / no decay |
| Gradient clipping | Norm 1 | No equivalent clipping recorded |
| Selected checkpoint | Fixed final epoch 30 | Fixed final epoch 5 |
| Validation-based selection | None | None |

The TreeLoom progress log contains 90 events summing to 1,500,000 sampled label presentations. The saved TreeLoom optimizer state has step 90; the saved CNN HDF5 optimizer iteration variable is 8,335. Thus the different update budgets are authenticated beyond descriptive settings. TreeLoom's loss at `research/repairs/benchmark_multigpu_20261022/scientific/queue_adapters/european_execution.py:292` sums tiled CE and divides by the complete parent label count; line 302 checks every sampled draw contributes once.

The CNN fit declares a 19,078-parameter historical convolutional architecture with equivalent portable NHWC execution, fresh initialization, and final-epoch selection. It explicitly does not claim exact reproduction of the historical runtime/random stream. More TreeLoom epochs do not imply more optimizer updates, and this run does not support a claim that TreeLoom saw fewer training labels.

## Reported endpoints

Values below are percentages from the actual comparison's equal-parent means. Derived C4 sums native-six probabilities as `[p0+p4, p1+p5, p2, p3]` before scoring. It is not a separately trained four-class CNN.

| Cohort | TreeLoom native6 | CNN native6 | TreeLoom derived C4 | CNN derived C4 |
|---|---:|---:|---:|---:|
| Overall | 54.802 | 92.702 | 70.832 | 95.844 |
| BAA | 0.000 | 95.690 | 83.470 | 97.430 |
| Bronze_Age | 68.110 | 92.050 | 67.010 | 94.930 |
| Neo | 71.580 | 93.480 | 72.050 | 97.600 |
| Yam | 65.990 | 93.970 | 63.460 | 96.060 |
| present_day | 68.330 | 88.320 | 68.170 | 93.200 |

The gap remains 25.012 percentage points after the four-class probability merge. Consequently, the native BAA naming issue alone cannot establish an explanation for the whole performance difference. Cohort-dependent admixture priors, optimization, representation, and learned decision behavior remain possible contributors.

## Supported conclusions and limits

**Proven:** exact matched sampled rows, labels, parent identities, and inferred-tree references; different explicit cohort information; age input present with nonzero checkpoint weights; all four pure donor classes present; different optimizer update count and granularity; fixed final-epoch selection; different native-six and derived-four results.

**Not established:** the causal fraction attributable to cohort categories, the amount of age information the model actually uses, whether matching optimizer budgets would close the gap, or whether a trained TreeLoom with equivalent cohort inputs would match the CNN. No ablation, retraining, new prediction run, or causal attribution was performed.

These scores are valid measurements of the two configured pipelines. They do not isolate architecture quality under equal input information and optimization. A future controlled comparison would need a declared common observed-covariate policy, aligned training/selection budgets, and both native-six and ancestry-only endpoints. Such experiments remain paused.
