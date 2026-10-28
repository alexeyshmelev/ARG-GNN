# TreeLoom accuracy audit, 28 October 2026

TreeLoom training restart is held at the owner's request. The European accuracy gap is real in the saved predictions. Exact training examples, labels and TEST targets match, but the configured methods receive different cohort information and have very different optimization schedules. Neither the input disparity nor the schedule has yet been shown to explain the entire gap. The recent American efficiency deployment did not train this European checkpoint. Earlier inference optimizations still require complete-model qualification.

## Verified current results

All values are percentages, mean ± sample SD over five independent TEST chromosomes. These are measurements of the configured pipelines on common targets, not a reproduction of a paper's numerical table or an isolated architecture comparison.

| Method | Native six-class accuracy | Native six-class macro F1 | Derived four-group accuracy |
|---|---:|---:|---:|
| AncestralPaths historical CNN, author-source port | **92.702 ± 0.352** | **91.579 ± 0.205** | **95.844 ± 0.392** |
| AncestralPaths released Dense control | 92.490 ± 0.533 | 91.396 ± 0.426 | 95.704 ± 0.491 |
| TreeLoom, current European fit | 54.802 ± 1.024 | 31.575 ± 0.571 | 70.832 ± 1.199 |
| GST-LAI-8 adaptation | 42.448 ± 1.133 | 27.882 ± 0.752 | 57.860 ± 1.040 |

Derived four-group accuracy sums the six-output probabilities before classification. It is not direct four-class training. GST-LAI-8 adapts GST to local ancestry inference. ARGMix and ARGformer are absent from this European table.

The independent scorer authenticated 20 prediction metadata/array pairs and reproduced all 240 method × parent × cohort/overall × endpoint confusion matrices, accuracies and macro F1 values. Parent aggregation and sample SD also agree. No class permutation or same-site truth join error was found.

Canonical result: [four-method comparison](../../experiments/results/paper_gst_lai_8_20261024/paper_gst_lai_8__all_methods_all_test__score/native/comparison.json). [Independent scoring audit](scoring/REPORT.md).

## Same supervision, different observed information

The actual CNN training ID sequence exactly equals TreeLoom's concatenated 50,000 sampled TRAIN IDs. Targets match for all three TRAIN chromosomes. All five TEST chromosomes have the same registered sample, haplotype and position identities and inferred-tree references. The training and test parents are disjoint.

TreeLoom retains all leaves, genotypes, inferred topology, sample age and four pure-reference ancestry labels. It converts the 686 query haplotypes across five observed cohorts to a common unknown reference class. CNN features retain eight ancient-cohort categories, including the ancient query cohorts. The eight TreeLoom donor channels instead represent four pure ancestry classes in two age bins. They do not encode the same information.

Observed cohort identity is available sample metadata, not local ancestry truth. It is particularly relevant to the native-six endpoint, whose two BAA classes occur only for BAA queries. TreeLoom's age feature is present and its input weights are nonzero. WHG reference samples are also present. This is an information-policy disparity, not evidence that TreeLoom has no ages or WHG donors.

The omission cannot yet explain the whole gap. TreeLoom still trails by 25.01 percentage points after probabilities are merged into four source groups. On present-day queries alone, derived four-group accuracy is 68.17% versus 93.20% for the CNN. [Input and target audit](inputs/findings.md).

## Training did not establish convergence

| Training quantity | TreeLoom | AncestralPaths CNN |
|---|---:|---:|
| Unique sampled TRAIN rows | 50,000 | 50,000 |
| Dataset passes | 30 | 5 |
| Optimizer updates | 90 | 8,335 |
| Label presentations | 1,500,000 | 250,000 |
| Update granularity | One complete parent, about 16,667 labels | Minibatch of 30 |
| Checkpoint selection | Final pass 30 | Final epoch 5 |

Both saved optimizer states confirm the update counts. TreeLoom did not see less data. It made far fewer, much larger parameter updates. Equal optimizer counts are not inherently required for a fair comparison, but our TreeLoom schedule lacks evidence of convergence or an adequate validation-based justification.

The mean training loss changed from about 1.657 at pass 10 to 1.544 at pass 20 and 1.095 at pass 29, then rose to 1.116 at pass 30. There was substantial late learning and instability, not a monotonic improvement at the end. Twenty-four of 90 updates were clipped. Final pre-clipping gradient norms include 21.22 and 14.80.

All six classes occur in TRAIN, with counts [15060, 9716, 5044, 10180, 7658, 2342]. All 49 learned tensors changed from the exact seeded initialization. Each classifier row changed and has nonzero optimizer moments. This rules out an entirely frozen head or absent labels as simple explanations, not every gradient bug.

Across 50,000 TEST records, the predicted class counts are [28068, 8020, 0, 13910, 2, 0]. TreeLoom never selects WHG or CHG_viaBAA and selects Ana_viaBAA only twice. Its BAA native-six accuracy is zero. This class collapse is a learning diagnostic that must be resolved before a credible performance claim.

## What execution verification establishes

The scored European checkpoint was written on 25 October at 04:00:57 UTC. Its SHA256 is `c06fc82db27d21cb7933da07d4247692c18b5c96df9cb67cb0bff71453e3f5e3`. It predates the 27 October American efficiency deployment. All 488 pinned Python source references authenticate, and the fit contains no 27 October source. Prediction restores the checkpoint with strict parameter loading.

The actual fit uses the static European executor. Auditing only the separate dynamic executor would not qualify this result. The fit uses an older sequential history implementation, while prediction uses the 24 October affine-prefix implementation. The shared tree encoder methods are structurally identical, but history reduction order differs.

A CPU decoder test with the actual checkpoint compared the original monolithic GRU with the actual tiled decoder and connected recurrent chunks. Mean loss matched exactly. Maximum context-gradient difference was 3.73e-9 and maximum parameter-gradient difference was 2.98e-8.

A separate CPU history test used the checkpoint's learned projection and decay weights over 1,031 synthetic intervals with changing groups, irregular spans and both directions. Maximum raw history differences were 3.46e-6 forward and 1.86e-7 backward. The projected memory difference was 2.38e-7. These are measured component checks, not a full-parent qualification.

No full European encoder/decoder forward and backward replay was performed in this audit. No optimizer update or new model training was run. Therefore no execution bug has been demonstrated, but the full fit/prediction implementation difference remains an open check. [Model lineage and numerical evidence](model/REPORT.md).

## Why the earlier wins do not predict this result

The earlier approximately 97.08% headline used present-day physical-span accuracy on eight reused 2 Mb validation simulations. It came from a warmstarted four-class lineage with 3,840 cumulative optimizer updates and validation-based checkpoint selection. The AncestralPaths comparator was the released Dense model, not the recovered historical CNN.

The current model is a fresh six-output fit on three 200 Mb TRAIN chromosomes, selected at the fixed final pass after 90 updates and scored on sampled SNP/query targets from five 200 Mb TEST chromosomes. Both experiments used observed Relate trees. We found no true-tree versus inferred-tree switch that explains the gap.

Those earlier results remain valid as recorded development measurements. They did not establish superiority on this benchmark. Earlier progress messages overstated what those wins established. [Detailed historical comparison](scoring/REPORT.md).

## Work required before TreeLoom training resumes

- [x] Authenticate the actual checkpoint, source lineage and prediction products.
- [x] Independently reproduce scoring and its aggregation.
- [x] Verify exact common TRAIN IDs and labels and common TEST bindings.
- [x] Inspect retained and omitted observed metadata.
- [x] Confirm parameter updates, class support and the learning trajectory.
- [x] Check the actual decoder and history implementations on isolated CPU probes.
- [ ] Qualify the full European fit/prediction path with the same checkpoint and identical input, including node/leaf ordering, global history boundaries, normalization, logits and gradients.
- [ ] Define and test one common observed-cohort information policy. Preserve the author CNN and add any missing legitimate covariates to a separately named TreeLoom condition. Never supply query ancestry truth.
- [ ] Establish a TreeLoom optimization schedule using TRAIN and independent VAL learning behavior, class coverage and gradient diagnostics. Do not choose it by the existing TEST scores or mechanically copy the CNN update count.
- [ ] Complete independent validation inference from the already generated data after its operational repair is qualified. Preserve its seeds, simulation and inference recipe.
- [ ] Register the corrected condition and acceptance criteria before training. Preserve this failed-fit evidence and the author baselines.

The existing TEST results have been inspected and informed this audit. They must be disclosed as such. Do not tune against them. Claims following iterative development will require a newly reserved untouched confirmation set in addition to reporting the existing benchmark transparently.

American TreeLoom's continuation after update 593 remains held as requested. Its logging failure is a separate operational issue. Native ARGformer may continue its unchanged healthy run. No queue repair or TreeLoom restart was submitted during this audit.
