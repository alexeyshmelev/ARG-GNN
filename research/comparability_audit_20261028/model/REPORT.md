The scored European TreeLoom is a trained six-class checkpoint with severe class collapse, a small number of optimizer updates, and no demonstrated convergence. This audit found no missing TRAIN class, unused output head, checkpoint-loading omission, class-output mask, or decoder gradient-scaling failure. Underoptimization is a supported explanation to investigate, not a proved sole cause. Full real-data fit/prediction encoder parity remains untested. TreeLoom restart remains on hold.

The current comparison at `experiments/results/paper_gst_lai_8_20261024/paper_gst_lai_8__all_methods_all_test__score/native/comparison.json` has SHA256 `eb4deea8be7ed6b9e3eacb38de89a918ae04b905bd8194c0e5d521fc021ac435`. Its five `predictions.treeloom` product references were rehashed and traced to the same final European checkpoint: `experiments/results/benchmark_campaign_20261018_workflow/release_20261022_full_vram_v2/other/AP__all_train__treeloom_fit/native/checkpoints/epoch_30.pt`, SHA256 `c06fc82db27d21cb7933da07d4247692c18b5c96df9cb67cb0bff71453e3f5e3`. The completion receipt, checkpoint, result and successful actual-wait receipt have filesystem timestamps around 2026-10-25 04:01 UTC. October 22 in these paths identifies the source/run family, not the final checkpoint date.

Confirmed output and training evidence:

| Native class | TRAIN rows | TEST argmax predictions across five parents |
|---|---:|---:|
| Ana_nonBAA | 15,060 | 28,068 |
| CHG_nonBAA | 9,716 | 8,020 |
| WHG | 5,044 | 0 |
| EHG | 10,180 | 13,910 |
| Ana_viaBAA | 7,658 | 2 |
| CHG_viaBAA | 2,342 | 0 |

TRAIN contains exactly 50,000 sampled rows across three parents: 16,667, 16,667 and 16,666. All 10,000 BAA TRAIN rows have class 4 or 5, and all six classes occur in every parent. The target metadata SHA256 values are `53094dcd6e8ab5ea72b5ac3b57284e7a1bcc7376070e1ee0d311c17cd03fbc59`, `581e4de79bd8a3a854a3ca328d399f97cb9e266165f3b3961f0fcd2d569b12e7`, and `924ef733e9a5cee6cbc1ad5150113ade0a024626b091e5aa033d2f3fc802636d`. These are the same target references independently reported for CNN/Dense/GST. Binding is in `.runtime/benchmarking/workflow_20261022_full_vram_v2/jobs/AP__all_train__treeloom_fit.json`, under `call.spec.scientific_spec.train_inputs`.

Registry records are stored in physical-position order, but both target generation and TreeLoom observation preparation use canonical `example_index` order. `european_sampling.registry_arrays` sorts by that index; target generation writes labels into `result[record["example_index"]]`. No label permutation was found. The initial audit's cohort cross-tab mistakenly used storage order; `evidence.v2.json` corrects and explicitly supersedes that part of v1. Overall class counts, weights and loss evidence were unaffected.

Every one of the 49 saved parameter tensors differs from the exact seed-101 initialization. All 49 Adam counters equal 90 and inspected state/moments are finite. Every element in the final 6-by-32 classifier weight matrix changed. Its per-class weight-change L2 norms are `[0.19380, 0.12362, 0.18420, 0.15760, 0.22284, 0.12961]`; each class has nonzero first and second Adam moments. The viaBAA bias first moments are negative (`-0.0023443`, `-0.0051878`), indicating recent optimizer pressure to increase those logits. This does not measure per-example gradient contributions, but rules out an entirely unused head. Saved probabilities for all six classes are finite and nonzero; class 4 reaches approximately 0.409 and class 5 approximately 0.199, so the omitted argmax classes are not numerically masked out.

The actual fit calls `ap_fit_v2` -> `queue_adapters.european_stage` -> `european_observations` -> `european_execution`. It does not invoke the public dynamic-AST European executor. Prediction calls the chunked-GRU wrapper and static `paper_argmix_treeloom_executor`. The actual `european_stage._validate_saved` at line 301 calls `model.load_state_dict(state, strict=True)`; the prediction wrapper calls that validator before scoring. A failure to load learned weights is ruled out.

The model has 32,391 parameters, a one-layer bidirectional GRU, and a six-output classifier. Four donor founder classes in the input encoding are distinct from the six ancestry-path outputs. Query reference labels are masked to -1 while known pure-source reference labels remain 0 through 3. Observed sample ages and genotypes are retained. This donor mask is not an output-class mask. No explicit class weighting, cohort-specific output mask, or class permutation appears in the actual loss/readout path.

Optimization is exactly 30 full passes and one AdamW step per complete parent: 90 updates. Each tile uses `cross_entropy(..., reduction="sum") / len(all_parent_labels)` at `european_execution.py:292`. The tile losses therefore sum to the parent mean, including repeated sampled rows. A temporary detached decoder input is followed by its complete context gradient and `context.backward(gradient)` at line 338. Full-sequence GRU chunks carry hidden state without detaching. All parameters must have finite connected gradients before global norm clipping to 1 and one optimizer step.

Learning rate is fixed at 0.001, weight decay 0.0001, betas 0.9/0.999 and epsilon 1e-8. No scheduler, early stopping or validation-based checkpoint selection is used. The selected checkpoint is fixed epoch 30. In comparison, ROOT independently identified CNN's 8,335 minibatch updates over five passes of the same 50,000 TRAIN rows. Thus TreeLoom has far fewer optimizer steps, but more row presentations: 1.5 million versus CNN's 250,000. Update counts alone are not equivalent optimization budgets across these architectures.

| Epoch | Mean of three recorded pre-update parent losses |
|---|---:|
| 1 | 1.78491 |
| 10 | 1.65748 |
| 20 | 1.54358 |
| 25 | 1.24961 |
| 28 | 1.10720 |
| 29 | 1.09521 |
| 30 | 1.11627 |

The last update loss is 1.01331, but the final epoch mean is above epoch 29. Twenty-four of 90 updates are clipped; the last three pre-clipping norms are 21.2221, 14.7965 and 0.71255. This is evidence of substantial learning with late instability, not proof of convergence or a monotonic final improvement. Training accuracy was not recorded. No new training accuracy or full-parent forward run was computed.

The October 27 American optimizations did not produce this scored European checkpoint. All 488 distinct Python source references collected from the European fit job match their pinned SHA256, and none contains the October 27 source family. The separate American attempt-8 job runs `research.efficiency_20261027.integration_candidate.restore_entrypoint_combined_v5`, has 175 references from that source family, and its current checkpoint SHA256 is `e32184cfe1eb6d953909814c9bc1588bf1df5b84aa45941dbabd5e2752916491`. It has 32,132 parameters, three outputs, seed 70001, 20 parents, and 593 durable updates toward a planned 600. It is a different data endpoint and checkpoint; its recovery remains held.

A material earlier implementation difference does exist between European fit and prediction. Fit uses the October 17 sequential stable-leaf recurrence with 128-hold windows. Prediction uses the October 24 affine-prefix implementation (`treeloom_speed/stable_leaf_history_v2.py` and `parallel_history_window.py`) through `paper_argmix_treeloom_history.py`, with 1024-hold windows. This changes FP32 arithmetic order. It is inaccurate to say that no American-derived speedup entered European inference. Both fit and prediction also use the same October 22 chunked-GRU adapter. Shared tree executor methods `_base`, `_block`, `_finite_own_episode`, `_first`, `_forcing` and `_second` are AST-identical; `_sites` differs only in an equivalent `bank` alias. Layout grouping and block/tile sizes differ.

Two small CPU qualifications were performed with no optimizer steps and CUDA never initialized:

- Decoder: actual scored weights, seven synthetic holds, nine synthetic haplotypes, all six labels, repeated focal coordinates, an empty decoder tile, and recurrent chunks of three holds. Original monolithic GRU/mean CE versus actual tiled decoder/manual context VJP gave identical loss, maximum context-gradient difference `3.73e-9`, and maximum parameter-gradient difference `2.98e-8`. All six output rows received nonzero synthetic gradient.
- History: actual scored `memory.U`, `memory.O`, and `memory.ell_theta` (`ell=5427.47`), 1,031 synthetic holds, 12 haplotypes, changing cut groups and irregular spans. Actual sequential/window-128 versus affine-prefix/window-1024 functions were compared in both directions and pooled/projected back through the learned memory output. Maximum directional differences were `3.46e-6` and `1.86e-7`; maximum projected memory difference was `2.38e-7`.

These tests give no evidence of a gross decoder or history implementation failure in the tested conditions. They do not prove full real-data encoder equivalence, GPU reduction equivalence, or all long-sequence numerical behavior. No complete real-parent parity test was run. A faithful real-data check needs identical globally prepared episode/cut recipes, whereas an arbitrary tree slice changes those recipes and boundary conditions. The existing TRAIN000 qualification contains 95,625 holds, 181,974,375 packed nodes and about 75 minutes of preparation; this audit did not recreate that workload or construct a separately qualified slice.

Confirmed observations are near-total failure to emit three trained classes and the mismatch in optimization granularity. Plausible mechanisms include incomplete optimization, late gradient instability and difficulty separating cohort/path labels from the available representation. Their causal shares are not established. Earlier inference optimization mismatch, other full-encoder/preprocessing defects and architecture/data effects remain unexcluded by the limited probes. No sole cause of the 54.802% versus 92.702% gap is asserted.

Primary machine-readable evidence is `evidence.v2.json`, `source_review.v1.json` and `history_probe.v2.json` in this directory. Source scripts are retained. All writes were confined to this audit directory; existing jobs, models, checkpoints and scientific prediction products were read-only. No training restart, model change, Git operation or queue action occurred.
