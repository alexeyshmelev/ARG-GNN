# ARG-GNN

<!-- joint-finish-operations:start -->
**Queue repair, 30 October 2026:** two European datasets now progress through CPU preparation concurrently. IID002 completed generation and Relate-input preparation and has started native reconstruction alongside IID001. Existing scientific workers were preserved. The benchmark is still running.

- **Why GPU 3 was idle:** both methods finished IID000 predictions; the next GPU stages require unfinished CPU genealogy reconstruction. The queue previously serialized the next dataset behind scratch cleanup.
- **Repairs active:** a rolling two-parent preparation window with full completion reservations; exact producer/consumer ownership; recovery from temporary observation and publication failures; immutable genuine completion receipts; status reports showing current blockers separately from historical failures.
- **Validation:** 64 focused tests passed across the shared scheduler, preparation lane, consumer/reporting bridge and ARGformer handoff. Independent reviews and live startup checks passed. No simulation, model, split, training budget, score rule or registered comparator condition changed.
- **ARGformer transition:** a safe handoff watcher is waiting while TEST007/008 workers remain active. It may replace only a verified childless controller. It will not interrupt their training or place a long ARGformer fit ahead of ready TreeLoom/native work on GPU 3.
- **Current capacity:** approximately 1.032 TB committed within the 1.2 TB project ceiling. This is reserved capacity, not a fresh full-directory disk-usage measurement. GPU idleness can remain while required CPU inputs are unfinished.
- **Comparator identity:** the registered comparator remains **ARGMix, equation-corrected variant**. Untouched `author_key_i` is an open comparison-fidelity item, not a completed author-code comparison.

[Repair, limitations and work status](research/joint_evaluation_20261029/queue_audit_20261030/SUSTAINABILITY.md) · [Current queue](research/joint_evaluation_20261029/root/CURRENT_STATUS.json) · [Verified live snapshot](research/joint_evaluation_20261029/queue_audit_20261030/live_status_v2.json). Final American and ARGMix comparisons, independent numerical review and manuscript completion remain pending.
<!-- joint-finish-operations:end -->

<!-- joint-model-20261029:start -->
## TreeLoom-Joint: held-out benchmark results, 29 October

**The new model is trained, integrated and scored on all five European TEST chromosomes.** Its six-class accuracy is **96.33 ± 0.16%**, compared with **92.70 ± 0.35%** for the historical AncestralPaths CNN. The paired improvement is **3.63 ± 0.28 percentage points** across chromosomes. Every method is scored on the same 50,000 canonical TEST targets. Architecture and checkpoint selection used TRAIN data only.

| Method | Accuracy (%) ↑ | Macro F1 (%) ↑ | Brier ↓ | NLL ↓ | Calibration error (%) ↓ |
|---|---:|---:|---:|---:|---:|
| TreeLoom-Joint (new) | **96.33 ± 0.16** | **95.94 ± 0.15** | **0.0629 ± 0.0039** | **0.1808 ± 0.0140** | 2.55 ± 0.19 |
| AncestralPaths historical CNN (author-source port) | 92.70 ± 0.35 | 91.58 ± 0.21 | 0.1102 ± 0.0063 | 0.2143 ± 0.0119 | **0.71 ± 0.20** |
| AncestralPaths Dense control | 92.49 ± 0.53 | 91.40 ± 0.43 | 0.1163 ± 0.0071 | 0.2317 ± 0.0128 | 1.69 ± 0.38 |
| TreeLoom (previous architecture) | 54.80 ± 1.02 | 31.58 ± 0.57 | 0.5735 ± 0.0088 | 1.0597 ± 0.0258 | 2.66 ± 0.57 |
| GST-LAI-8 (ancestry adaptation) | 42.45 ± 1.13 | 27.88 ± 0.75 | 0.6750 ± 0.0066 | 1.3657 ± 0.0135 | 2.30 ± 1.04 |

Values are mean ± sample SD across five independently simulated chromosomes, not across training seeds. Calibration error uses ten equal-width confidence bins. The new model improves accuracy, macro F1, Brier score and NLL on every chromosome, but its confidence calibration is worse than the CNN. This is a comparison on shared simulated data, not a claim to reproduce an original published numerical table.

| Simulated cohort | CNN accuracy (%) | TreeLoom-Joint accuracy (%) | Gain (percentage points) |
|---|---:|---:|---:|
| Present day | 88.32 ± 1.55 | **93.08 ± 0.87** | +4.76 |
| Bronze Age | 92.05 ± 1.13 | **95.57 ± 0.61** | +3.52 |
| BAA | 95.69 ± 0.37 | **97.49 ± 0.32** | +1.80 |
| Neolithic | 93.48 ± 0.50 | **98.16 ± 0.55** | +4.68 |
| Yamnaya | 93.97 ± 0.64 | **97.34 ± 0.29** | +3.37 |

**Execution:** the full European fit completed 50 epochs and 3,900 updates in 6.00 minutes. Warm training averaged 7,600 targets/s and 88.2% sampled GPU activity. Each TEST chromosome took about 62–66 seconds for preparation and prediction, with four CPU workers. Test preparation is streamed through RAM, with no persistent bulk cache.

**Architecture:** three joint PyTorch Geometric layers exchange messages within complete selected native trees and across trees through shared leaves and branch structure. All queries at one SNP reuse one encoding. There is no recurrent network. The general idea of graph windows has prior art, so this result does not establish a new general GNN kernel.

**Queue snapshot:** 155 original campaign stages completed, **69 pending or running** and 11 explicitly superseded after counting parallel task IDs only once. The records contain 2 running, 67 pending, 16 blocked and 1 failed stages. These are preparation, fitting, prediction and scoring stages, not that many separate experiments. The failed previous American TreeLoom and its dependent rows are retained as history. The effective view includes the active native retry. The 17 old American TreeLoom rows remain historical; current native retry failures are counted separately below.

- **ARGformer:** 14/15 dataset predictions complete. The remaining 1 parents are assigned to the original GPU3 lane and the two native lanes on GPUs4 and5. Their original training schedules remain unchanged.
- **FLARE:** 15/15 prediction datasets complete and authenticated.
- **ARGMix:** the TreeLoom execution continuation is running: joint_argmix.published_recon_iid_001.prediction. The fixed five-epoch TreeLoom fit is complete. Remaining predictions follow producer readiness. Final paired ancestry scores remain pending.
- **New American TreeLoom:** the fixed 13-epoch fit and all fifteen held-out predictions are complete. Paired ancestry scoring awaits native comparator predictions.
- **Direct four-class AncestralPaths CNN and TreeLoom-Joint:** both fresh four-output models completed training and prediction on all five TEST chromosomes. The completed comparison uses the same 40,000 supervised targets and 40,000 TEST targets. TreeLoom used its fixed 50 epochs and CNN its fixed five epochs. Present-day accuracy remains the registered primary endpoint; results are reported below.

[Detailed scores, class metrics and confusion matrices](research/joint_evaluation_20261029/scoring/comparison_v4.json) · [Independent result review](research/joint_evaluation_20261029/scoring/FINAL_RESULT_REVIEW_v1.json) · [Current queue snapshot](research/joint_evaluation_20261029/root/CURRENT_STATUS.json) · [Integration work plan](research/joint_evaluation_20261029/PLAN.md) · [Frozen model and development results](research/joint_model_20261029/PLAN.md)

### European benchmark: completed direct four-class comparison

Both models are trained on the same 40,000 supervised rows and scored on the same 40,000 TEST rows. Values are mean ± sample SD across five chromosomes. Present-day accuracy is the registered primary metric.

| Method | Present-day accuracy (%) ↑ | Overall accuracy (%) ↑ | Overall macro F1 (%) ↑ |
|---|---:|---:|---:|
| AncestralPaths CNN (direct four outputs) | **93.38 ± 1.47** | 95.55 ± 0.59 | 95.22 ± 0.62 |
| TreeLoom-Joint | 92.80 ± 1.22 | **95.82 ± 0.30** | **95.52 ± 0.28** |

The models use different observed representations. TreeLoom keeps BAA query leaves as unknown context, with no BAA target labels. The CNN uses its historical four-path features. Both have four direct output classes, with no probability merging.

[Full direct four-class scores and cohort matrices](research/joint_evaluation_20261029/european_four/outputs/paired_score/comparison.json)

- **Native ARGMix:** ARGMix, registered equation-corrected variant (author-derived Transformer) completed 13 epochs and selected epoch 4 by the registered validation rule. Training used 500,000 TRAIN examples and selection used 100,000 validation examples. These fit milestones are separate from held-out ancestry scores. Current owned stages: 2 succeeded, 0 running, 18 pending, 0 failed, 0 blocked. The original failed CUDA-initialization attempt remains historical evidence. This registered reconstruction is not a claim to reproduce an original published numerical table.
- **Native preparation:** the registered parallel producer owns forty-eight stages across six remaining parents. Current stages: 2 succeeded, 1 running, 45 pending, 0 failed, 0 blocked. Original main claim-blocked rows remain historical and are excluded from current producer counts.
<!-- joint-model-20261029:end -->

<!-- joint-four-conditions-20261029 -->
The direct four-class comparison uses one initialization per method. TreeLoom receives 50 epochs, or 2,000,000 target presentations, and the documented CNN reconstruction receives its registered five epochs, or 200,000 presentations. Both use the same 40,000 supervised rows. This is not an equal-compute comparison or a claim to reproduce the original publication runtime. The primary present-day endpoint favors the CNN. Fixed four-class macro F1 retains all four classes even in cohorts with absent classes. [Independent completed-result audit](research/joint_evaluation_20261029/review_finish/european_four_completed_checks_v1.json).


<!-- bounded-redesign-20261028:start -->
## TreeLoom redesign, 28 October

**The new bounded model completed its first learning diagnostic.** It processes small groups of complete inferred trees and exchanges information through shared leaves and groups defined at several ancestral ages. Ordinary minibatches replace full-chromosome optimizer updates. Both arms completed 400 updates with the same initialization, contexts and target order.

| New model arm | Fitting accuracy, 128 targets | Development accuracy, 64 targets | Development macro F1 | Development cross-entropy |
|---|---:|---:|---:|---:|
| Without cross-tree communication | **79.69%** | 68.75% | 60.21% | **0.5786** |
| With cross-tree communication | 77.34% | **70.31%** | **65.82%** | 0.5824 |

The accuracy difference is **one additional correct prediction out of 64**, with slightly worse cross-entropy. This establishes learnability, not superiority. These are small internal development results on a separate original TRAIN chromosome, not the five-chromosome benchmark below. One development chromosome and one seed do not support a chromosome-level SD. CPU training took 4.16 and 5.81 minutes respectively. A separate GPU profile is complete: with 16 focal targets, the cross-tree model took 280.61 ms per forward/backward batch, reduced to 263.58 ms by omitting irrelevant second-pass computations. This short fixed-batch profile excludes data loading and does not establish sustained 80% utilization. [GPU measurements and limits](research/redesign_20261028/GPU_QUALIFICATION.md).

The old full-parent continuation remains held. Native competitor methods and simulations are unchanged. The separate sliding joint-graph convolution investigation has produced a tested operator, described below. No new general GNN kernel or ancestry accuracy advantage is claimed.

[Complete diagnostic and interpretation](research/redesign_20261028/LEARNING_DIAGNOSTIC.md) · [Design and work plan](research/redesign_20261028/PLAN.md) · [Implemented architecture](research/redesign_20261028/architecture/PROTOTYPE_REPORT.md) · [Critical literature review](research/redesign_20261028/literature/REPORT.md) · [Independent technical review](research/redesign_20261028/review/REVIEW.md)

**Separate sliding graph-convolution prototype:** each PyTorch Geometric layer updates native tree edges and connections across trees simultaneously. Cross-tree branch weights use shared sampled descendants and overlapping ancestral lifetimes. The isolated layer passed 18 CPU checks, including explicit sample-identity validation. Its full ancestry encoder/readout and learning comparison are still pending. Sliding graph windows already appear in prior work, so no general novelty or accuracy claim follows from this design. [Mathematical design](research/sliding_graph_convolution_20261028/DESIGN.md) · [Prior art](research/sliding_graph_convolution_20261028/PRIOR_ART.md) · [Independent critique](research/redesign_20261028/architecture/SLIDING_BRANCH_REVIEW.md).
<!-- bounded-redesign-20261028:end -->

<!-- accuracy-audit-20261028:start -->
## TreeLoom accuracy audit, 28 October

**TreeLoom training restart is held.** Its current European six-class accuracy is **54.80 ± 1.02%**, versus **92.70 ± 0.35%** for the AncestralPaths historical CNN. Independent scoring reproduced the results. Both fits use the exact same **50,000 TRAIN examples and labels**, and their five TEST chromosomes and target identities match.

The audit found two substantial setup differences. TreeLoom omits observed query-cohort categories that the CNN retains, although it keeps sample ages and all four pure-reference groups. TreeLoom also made only **90 full-parent optimizer updates**, compared with **8,335 CNN minibatch updates**. It saw more label presentations, but convergence was not established. Its saved predictions never select two of the six classes. Neither input changes nor longer training are yet proven to recover the gap.

The scored checkpoint predates the latest American efficiency deployment. Decoder and history component checks agree closely, but the complete European fit/prediction execution path still needs qualification. The earlier approximately 97.08% result came from a different four-class development experiment, a warmstarted model and a Dense comparator. It did not establish superiority on this benchmark. Earlier progress messages overstated that result.

[Full audit, numerical evidence and restart requirements](research/comparability_audit_20261028/REPORT.md). Author baselines, simulations, labels and recorded results are preserved. The operational snapshot below predates this audit hold.
<!-- accuracy-audit-20261028:end -->

<!-- recovery-no-timeouts-20261027:start -->
## Benchmark status, 28 October

Verified **2026-10-28 12:49 MSK**. GST training, five test predictions and common scoring are complete. Native ARGformer now has **6/15** prediction datasets complete, and validation004 masked-token pretraining is active on GPU 4, epoch 4 of 5.

American TreeLoom advanced from 437 to **593/600 committed updates**, then stopped because the progress logger reached an incorrectly sized 2 MiB limit. Its checkpoint includes update 593, even though the final log entry is update 592. Seven updates remain. This operational failure blocks 15 TreeLoom predictions and the American comparison. A logging repair was being prepared. TreeLoom continuation is now held for the accuracy audit above.

Independent European validation finished generating the original **200 Mb** parent, with **1,340,560 sites**. It then stopped before native inference because its own slow resource check could make a five-second storage sample expire. The generated data are intact. The proposed repair would reuse them with the original seeds and inference settings. It has not been launched.

Current queue: **1 running, 61 succeeded, 173 pending, 1 failed and 16 blocked stages**. GPUs 3 and 5 currently have no active campaign work. One ready ARGMix Relate task still waits on stale controller storage accounting. These are operational blockers, not a completed benchmark or evidence of an accuracy gain. [Latest evidence and repairs](research/recovery_20261028/PLAN.md).
<!-- recovery-no-timeouts-20261027:end -->

<!-- storage-cleanup-20261027:start -->
## Storage cleanup, 27 October

Removed **380.6 GB** of obsolete pilot datasets, epoch prediction dumps, a superseded ARGMix training archive and downloaded installer caches. Code, design history, metrics, selected historical predictions and all current benchmark dependencies remain. Cleanup was restricted to this repository on anogena. The earlier 924.2 GB inventory minus removals gives approximately **543.6 GB**, before subsequent benchmark writes. This is not a fresh full-directory measurement.

Cleanup did not signal any process. A separate full European validation run had failed before cleanup and its failure evidence is preserved. [Cleanup report and exact deletion records](research/storage_cleanup_20261027/README.md).
<!-- storage-cleanup-20261027:end -->

<!-- european-interim-20261027:start -->
## European ancestry comparison, 28 October

**Current TreeLoom trails the CNN and Dense baselines, while exceeding the completed GST adaptation.** All four methods were scored on the same five test chromosomes and registered targets. The adapter audit found unequal population metadata: the CNN receives eight observed cohort channels, while TreeLoom collapses five admixed cohorts into an unknown reference label. These scores therefore do not isolate the architectural difference. Any corrected TreeLoom condition must qualify the common observed-input policy and full execution path before training, with development restricted to TRAIN/VAL. The values below are mean ± sample SD across chromosomes, expressed as percentages. This is a common-data comparison and does not claim exact reproduction of a published numerical table.

| Method | Six-class accuracy | Six-class macro F1 | Accuracy after merging into four ancestry groups |
|---|---:|---:|---:|
| AncestralPaths historical CNN (author-source port) | **92.70 ± 0.35** | **91.58 ± 0.21** | **95.84 ± 0.39** |
| AncestralPaths Dense control | 92.49 ± 0.53 | 91.40 ± 0.43 | 95.70 ± 0.49 |
| TreeLoom, current European condition | 54.80 ± 1.02 | 31.58 ± 0.57 | 70.83 ± 1.20 |
| GST-LAI-8, adaptation for ancestry | 42.45 ± 1.13 | 27.88 ± 0.75 | 57.86 ± 1.04 |

GST-LAI-8 is an adaptation of Graph Set Transformer to ancestry inference, not the GST paper's original benchmark. The derived four-group accuracy merges six-output probabilities and is not direct four-class training. [Complete cohort summaries](research/recovery_20261028/european_comparison_summary.json).

The CNN uses the recovered historical author architecture, ported to an equivalent CPU tensor layout. Dense is a separate released author model. Both fits use the disclosed 50,000-row common-data protocol. The historical 812,500-row sampler and original software random stream are not reproduced. [Source and scored-checkpoint verification](research/efficiency_20261027/fidelity/EUROPEAN_SCORED_CNN_PROVENANCE_LABEL_REVIEW.json).

The separately requested direct four-class CNN experiment is implemented but not yet trained. The American efficiency tests concern a different model configuration and do not resolve this European accuracy gap.

[Complete four-method scores and cohort breakdown](experiments/results/paper_gst_lai_8_20261024/paper_gst_lai_8__all_methods_all_test__score/native/comparison.json)
<!-- european-interim-20261027:end -->

<!-- efficiency-20261027:start -->
## TreeLoom efficiency work, 27 October

Historical deployment snapshot. The current training hold and update 593 checkpoint are described above.

**The requested sustained 80% useful GPU utilization remains unproven.** The combined execution changes are now deployed on GPU 3. Attempt 8 restored the complete checkpoint at **281/600**, including optimizer moments, random state and the original chromosome order. Its latest complete checkpoint is **287/600**. The first production update took **693.39 seconds** and averaged only **15.83% device activity** during its observed training window. Including cold preparation, the observation averaged **7.55%**. The deployed pipeline still falls far short of the goal. A later 64.8-minute observation, including cold preparation, averaged **11.48 percent** device activity with the TreeLoom PID as the only compute client. A complete warm cycle is still unmeasured.

| Tested change | Reference time | Candidate time | Measured reduction |
|---|---:|---:|---:|
| Eight CPU threads, explicit backward and larger tree blocks, one complete chromosome | 620.75 s | **380.42 s** | 38.7% |
| Fuse consecutive tree operations, one complete chromosome | 324.49 s | **298.47 s** | 8.0% |
| Keep upward states and their gradients on the GPU, one complete chromosome | 318.32 s | **237.01 s** | 25.5% |
| Prepare two chromosomes concurrently, four encoders per producer | 1,560.89 s | **1,238.36 s** | 20.7% |

Each row is a separate, fixed-order paired qualification. These reductions must not be multiplied or treated as measured whole-training acceleration. Full numerical tests retained all 77,472 trees, 1,600 samples, 227,474 sites and 10,000 targets. Logits, loss, all 49 gradients, 49 updated parameters and 147 Adam tensors passed the declared FP32 tolerances, with zero changed predicted classes. Concurrent preparation produced identical complete signatures and preserved the random state, with eight encoder children observed against four sequentially.

The resident executor peaked at **59.85 GB allocated** and **64.80 GB reserved** GPU memory. The deployed worker selects it from the actual chromosome size and available memory before forward propagation. When it does not fit, the worker uses the qualified lower-memory executor. It keeps the same model, data, loss and training schedule.

The earlier preparation-inclusive observation averaged **18.35% device activity**. The full resident qualification averaged **69.09%** on a shared GPU. Neither establishes the requested sustained utilization. The new observation records active compute processes, complete update boundaries and idle gaps. Its warm assessment must cover all 20 training chromosomes and every selected execution route.

The first production update spent **499.12 seconds in backward propagation**, including **387.13 seconds** in the final tree sweep, while two background producers were active. Preparation uses threads in the same Python interpreter as GPU dispatch. Contention is a plausible mechanism, but external stack sampling was denied and causation is not yet established. Resource callbacks also accumulated 109.14 seconds since startup across threads, so that number cannot be subtracted from one update. The full TRAIN001 comparison now qualifies integer episode construction: 225.35 to 199.33 seconds, an 11.55 percent reduction in this stage. All nine structural/episode arrays match the archived hashes, and newly computed original/candidate episode metadata are identical. This is one fixed-order CPU pair, not an end-to-end speedup. It remains isolated for integration with fresh CPU producer processes. The asynchronous resource-check candidate has also passed independent timing and failure-propagation tests, but is not deployed. Compact packing accounts for only about 25.5 seconds per chromosome in the paired preparation test. The optional GPU-context candidate remains deferred after exceeding its original strict context tolerance. The slower TreeLoom-Transport candidate is also deferred. Historical failures and closed hypotheses remain recorded.

The separate European cohort correction failed full-chromosome qualification: 180 of 3,899,392 context values exceeded the original tolerance. The corrected fit has not been launched. The full 200 Mb independent validation runner also failed before cleanup because its storage scanner could race with an atomic status-file replacement. The reviewed successor fixes that race and transient queue handovers, removes elapsed experiment cutoffs, and completed the same original simulation. It subsequently failed before inference on a five-second storage freshness check, as detailed in the current status above. All prior failures remain preserved. Total reserved commitments are about 1.125 TB under the authorized 1.2 TB ceiling. This is conservative reservation accounting, not current disk usage. Original TRAIN/TEST inputs remain unchanged, and no new accuracy gain is claimed.

[Work plan](research/efficiency_20261027/PLAN.md) · [Independent combined review](research/efficiency_20261027/fidelity/COMBINED_EXECUTION_V5_RESULTS_REVIEW.json) · [Deployment release](research/efficiency_20261027/root/combined_execution_release_v5.json) · [Latest status](research/efficiency_20261027/root/benchmark_snapshot_v11.json) · [First production update](research/efficiency_20261027/root/production_attempt8_first_update_v1/report.json)
<!-- efficiency-20261027:end -->

<!-- requested-corrections-20261027:start -->
## Requested benchmark additions, 27 October

**AncestralPaths four-class training is implemented as a separate condition.** It recomputes input features with Bronze Age Anatolian (BAA) references excluded and trains a fresh four-output CNN. This is direct four-class training, not merged six-class predictions. The original six-class benchmark remains intact. All 33 CPU checks passed. A separate 15-stage CPU plan is prepared, with 40,000 training examples and five independent test parents. Full four-class training has not been launched, so no new accuracy score is claimed. A matched four-output TreeLoom fit remains necessary for a new comparison. [Scientific definition and execution plan](research/benchmark_corrections_20261027/ANCESTRALPATHS_FOUR.md).

**ARGMix original metadata are partly recovered.** Public reference identities and dated records recover 13 CHG, 42 EHG, 48 WHG and 13 Yamnaya individuals, matching the paper's four cohort counts. The exact final roster is still unresolved: 3 farmer exclusions, 39 other ancient exclusions, individual farmer ancestry assignments and the selection of 660 modern Europeans remain unknown. Published date sources also disagree for some individuals. All source versions and candidate identities are preserved, with no arbitrary membership or age choices. Existing simulations have not been replaced. [Recovery report and source tables](research/argmix_roster_recovery_20261027/public_metadata/REPORT.md).

<!-- requested-corrections-20261027:end -->

<!-- owner-stop-current:start -->
## Remaining comparisons

GST, AncestralPaths CNN, Dense and current European TreeLoom have completed common scoring, shown above. The American comparison still needs TreeLoom's final seven updates and predictions, plus the remaining native ARGformer datasets. TreeLoom continuation is held for the accuracy audit. FLARE already has 15 native prediction products. ARGMix still needs its blocked preparation and downstream fits, predictions and scoring. Direct four-class AncestralPaths training is prepared but unlaunched. The European cohort correction remains unqualified after its numerical mismatch.

All computation stays on anogena using only physical GPUs 3, 4 and 5. No reliable final completion time is available while these operational blockers remain.
<!-- owner-stop-current:end -->

**Preprocessing audit, 26 October:** preparation is not yet fully parallel. At the original audit, American TreeLoom used one background preparation worker. The qualified deployment below now uses four encoding processes within that producer. GST has four loader workers, but its initial VCF joins run serially before they start. European TreeLoom uses eight CPU threads during encoding but prepares and predicts parents sequentially. Native ARGformer has eight persistent loader workers and is actively training. [Measured bottlenecks and concrete execution improvements](research/preprocessing_efficiency_20261026/REPORT.md). No model, simulation or running job was changed by this audit.

<!-- exact-preprocessing-current:start -->
**Exact preprocessing optimization, 26 October: qualification passed.** Complete American preprocessing fell from **25.4 to 12.8 minutes (1.98×)** using four CPU processes. Full European VCF parsing fell from **422 to 46 seconds (9.10×)**. All compared preprocessing arrays, shapes, dtypes and metadata match exactly. FP32 checks of the unchanged model, gradients and AdamW update pass numerical tolerances, with an unchanged-baseline repeat as control. Extensive boundary, malformed-input, process-cleanup and checkpoint tests passed independent review. These are preprocessing gains, not measured whole-training speedups or new ancestry scores.

**Deployed for American TreeLoom on GPU 3.** The optimized worker restored update **241/600**, optimizer moments and RNG state. The first complete input finished preparation and packing in **14 min 56 s**. GPU training now overlaps four-worker preparation of the next input. This is a live pipeline check, not a paired whole-training speedup or a newly completed optimizer update. The other benchmark workers remain active, and native Relate also gained admission. The queue has **5 running, 32 succeeded, 215 pending and no failed or blocked stages** at the deployment check. Native competitor implementations, simulations, model equations and scoring are unchanged. European prediction prefetch is not deployed. [Measured results and remaining scope](research/preprocessing_optimization_20261026/REPORT.md) · [Tests and deployment gates](research/preprocessing_optimization_20261026/PLAN.md).
<!-- exact-preprocessing-current:end -->

<!-- figure-redesign-current:start -->
**TreeLoom figures simplified, 26 October:** [prediction flow](paper/figures/TreeLoom_architecture.png) follows one query leaf from the observed trees to ancestry probabilities. [Encoder](paper/figures/TreeLoom_encoder.png) explains the two tree rounds, clade context and leaf memory. Routine implementation detail is removed. A fresh reader recovered the main computation from the images alone, with remaining feature definitions clarified in the captions. [Updated 10-page paper](paper/current_full_panel_working.pdf) · [Overleaf bundle](paper/current_full_panel_overleaf_working.zip) · [Full reader reports and source check](paper/figure_redesign_20261026_v6/README.md). All eight table source blocks are unchanged. Experiments stayed stopped during that figure revision.
<!-- figure-redesign-current:end -->

<!-- paper-benchmark-rebuild-current:start -->
## Last benchmark snapshot before the requested stop, 25 October

**American TreeLoom and native ARGformer training are progressing. European TreeLoom training has finished. The ARGMix memory failure is repaired, European prediction is queued for retry, and the GST loader repair is qualified. No new matched ancestry advantage is established yet.**

| Workstream | Current verified result |
|---|---|
| American TreeLoom | **237/600 training updates recorded**, on GPU 3. Each update retains the complete simulated parent, frozen model, optimizer and full genomic history. |
| European TreeLoom | **90/90 training updates finished.** Prediction repair passed CPU and small CUDA checks. Its automatic retry is queued after the full-parent profile exits, reusing the final checkpoint without retraining. |
| Corrected ARGformer | Native predictions completed for **two TEST parents and one validation parent**. MLM training for validation parent 001 continues on GPU 4. These products still require common ancestry scoring. |
| AncestralPaths | Historical CNN completed 8,335 optimizer updates and all five TEST prediction products. Common scoring awaits the paired models. |
| FLARE | All 15 native American outputs passed reuse checks. Paired held-out comparison remains pending. |
| ARGMix | TRAIN000 Relate, conversion and preprocessing completed. The RAM repair passed. TRAIN001 simulation and Relate inputs also completed. Its native Relate run currently waits for storage admission, about **443 MB** short of its required reservation under the 990 GB operating ceiling. |
| GST-LAI-8 | Previous fit timed out with three durable updates and no recoverable checkpoint. The repaired cursor and checkpoints passed window, optimizer/resume and observed-tree checks. **Fit restart is prepared but not submitted.** This is an adapted LAI baseline. |

Every paired comparison uses the same observed parent, reference/query identities and evaluation positions. Available publication settings are implemented explicitly. Unreleased details use documented reconstructions. The simulations and observed data are fixed for the new scalability work. Eligible GPUs are **3, 4 and 5** and the project storage limit is **1 TB**.

The current American TreeLoom has a measured warm median of **9.74 minutes per full-parent update** over the latest ten updates. Context forward and trunk backward dominate its measured wall time. This motivates model and execution work beyond simply allocating more GPU memory.

| Scalability result on 128 observed TRAIN trees | Interpretation |
|---|---|
| 317,897 active-node occurrences across 128 trees | Whole-tree expansion is large, even for a bounded slice. |
| Reusable layout: 12.67 MB → 6.34 MB stored indices | Exact execution prototype. Original rebuild: 25.5 ms. First construction: 32.9 ms. Subsequent reuse: 56.9 microseconds. Full-parent/GPU speedup remains unmeasured. |
| Layout first-sweep outputs and gradients agree on a bounded real-tree check | Supports correct index/reduction reuse. It is not an ancestry result. |
| Sparse remapping was slightly slower | Rejected as an American optimization. |
| Primitive-input census: 63.89% repeated first-upward state occurrences | Sharing is predominantly in leaf/input computations. This is not a 63.89% runtime saving. The stricter source-ID census missed these identical-input cases. |
| Categorical SNP encoder: 409,600 → 768 MLP evaluations in the measured chunk | Exact-equation prototype passed. Four-thread CPU warm forward was **1.77×** faster and the forward/backward probe **1.98×** faster. Outputs matched and maximum gradient error was 2.35e-7. The integrated GPU slice result is below. Full-parent gain remains unmeasured. |
| Rank-eight basis captures 98.11% of centered upward-state variation, but maximum relative vector error is 16.89% | Approximation remains deferred. Average compression quality does not establish preserved ancestry accuracy. |

**Integrated GPU result:** the exact successor passed a paired test on the real 128-tree TRAIN slice, including all 49 parameter gradients and the clipped AdamW update. One warm run took **1.26 s versus 2.06 s**, a **1.63x** ratio. Peak PyTorch reserved GPU memory fell from **1.53 GB to 0.98 GB**. Maximum gradient and updated-parameter differences were **3.28e-7** and **2.98e-8**. This uses the 22 registered focal labels within the slice. It is a restricted-window timing test, not a full-parent speedup or held-out accuracy result. The first full-parent profile was cancelled when a benchmark job needed the reserved host capacity. A fresh retry is now preparing the complete TRAIN000 observation for GPU 5. No full-parent timing result exists yet. [Full-parent retry](research/repairs/benchmark_recovery_20261025/full_profile_restart.json). [Measured receipt](research/scalability_20261024/root_integration_measurement.json) · [Paired run](research/scalability_20261024/profiling/runs/slice_gpu5_v1/data/completed.json).

Verified 2026-10-25T23:40:41.329749+03:00. Campaign stages: 32 succeeded or reused, 2 running, 2 failed, 2 blocked and 213 pending. The failed European and GST rows persist until their reviewed retries are submitted. Stage counts are not completed method comparisons. [Live-check record](research/repairs/benchmark_recovery_20261025/status_after_repairs.json) · [Repairs and remaining work](research/repairs/benchmark_recovery_20261025/REPORT.md).

The healthy frozen benchmark fits continue with their existing source. A scalable successor will be labelled separately and must earn its place through end-to-end time, memory, ancestry accuracy and boundary-resolution measurements. Generic caching or low rank alone is not a novelty claim.

[Scalability plan](research/scalability_20261024/PLAN.md) · [Categorical SNP encoder results](research/scalability_20261024/categorical_site/REPORT.md) · [Measured execution results](research/scalability_20261024/execution/REPORT.md) · [Scientific review](research/scalability_20261024/review/shared_computation_scientific_review.txt) · [Benchmark plan](research/benchmark_rebuild_20261024/PLAN.md) · [Detailed scores and historical comparisons](research/current_scoreboard.md)
<!-- paper-benchmark-rebuild-current:end -->


<!-- readable-gpu345-queue:start -->
### Historical queue snapshot, superseded by the campaign above
Updated 2026-10-23T23:08:57.823270+00:00. Status: drained_for_operational_update.
74 pending, 0 running, 259 succeeded, 1 failed, 16 blocked, 0 unknown.
Local active: waiting for dependencies or capacity.
Live training stages: bounded stage metadata pending.
Standalone work: no live standalone generations.
The earlier v9 queue block is a historical snapshot. This block describes the active successor.
Ancestry accuracy is pending the frozen prediction and scoring barriers. Training loss is not accuracy.
One optional external AP001 archive is deferred with its full retained storage charge and zero credit.
[Current queue metadata](research/benchmark_gpu345_current.json).
<!-- readable-gpu345-queue:end -->













**TreeLoom** learns local ancestry inference (LAI) from ordered collections of complete genealogical trees. Its GNN processes tree structure and follows persistent haplotypes across genomic intervals.

## Earlier benchmark campaign and development history

**The earlier LAI benchmark continuation used 350 stages, including storage recovery and fresh qualifications. The current campaign is described at the top of this README.** It covers shared American and European simulations, full training, native competitors, paired scoring and Danish data application. Current execution and matched results are published below. All computation and Git commands run on anogena, using physical GPUs 3, 4 and 5.

<!-- BEGIN CURRENT BENCHMARK DIAGNOSIS -->
**Verified 2026-10-22T23:09:15.814521+03:00: only physical GPUs 3, 4 and 5 are authorized.** American TreeLoom has moved from GPU2 to GPU4 and is running without an artificial GPU memory limit.

| Workstream | Current status |
| --- | --- |
| American TreeLoom, GPU4 | Restored 6/600 completed updates and all optimizer state. Next update: 7. Completed updates replayed: 0 |
| Native ARGformer, GPU3 | The ongoing fifth validation pretraining job is preserved. Completed fourth validation embeddings remain available for ancestry prediction |
| European TreeLoom, GPU5 | Both the existing four-class fit and the AP preparation job are preserved under their original supervisors |
| Fresh matched ancestry accuracy and F1 | Pending complete training and common scoring. This hardware move changes no scientific method or benchmark input |

The old GPU2 worker exited under its own supervisor, and its exact cancellation wait is retained. No unrelated process was signalled. Existing GPU3/5 source files, approvals and jobs remain unchanged. Legacy GPU fields in the policy exist solely for those original workers. New execution is governed by the current explicit GPU3/4/5 authority.

Conservative storage projection: 968.60 GB, below the 980 GB operational ceiling and the 1 TB project limit. Full GPU memory remains enabled.

[Verified migration](research/benchmark_current_verified_status.json) · [Current queue](research/benchmark_gpu345_current.json)
<!-- END CURRENT BENCHMARK DIAGNOSIS -->
<!-- BEGIN SCIENTIFIC COMPARISON SCOPE -->
## What the current benchmark can establish

**Audited 2026-10-23T04:15:46.474888+03:00: the active queue does not cover all four original paper benchmarks.** It implements paired comparisons on new shared simulations. Fresh seeds are acceptable, but material differences in populations, models and evaluation rules remain. Completing its 350 stages alone does not complete the original research goal.

| Comparison | Actual condition | Relation to the paper |
| --- | --- | --- |
| TreeLoom, native ARGformer and FLARE | Shared 30 Mb American simulations and identical frozen query records | Broad biological recipe reconstructed. Original sample ploidy and evaluator unresolved. Released ARGformer temperature/weighting differ from paper. Native top1 and FLARE focal projection are disclosed choices |
| TreeLoom and released AncestralPaths | 476-person panel, three 200 Mb training sequences and five independent test sequences, six ancestry paths | Paper scale and demographic scenario restored. Current queue uses the released Dense classifier. Deep history search recovered the earlier CNN and an example checkpoint. Exact paper-run data/settings and example/site sampling remain unresolved |
| TreeLoom and released ARGMix | Four-source task on the AP 476-person 200 Mb panel, with BAA, uniform map and no genotype error | Not ARGMix's 1,611-person chr22/error scenario or its IID/misspecification design. Current 15-reference/no-ancestor release differs from paper 25-reference/three-ancestor representation |
| TreeLoom and GST on original GST tasks | Absent from the active queue | Full candidate/native comparison unfinished. Earlier native 1,000-reaction validation pilot does not close this requirement |

Within each paired endpoint, code fixes common sample/site identities and requires prediction seals before held-out ancestry scoring. Native-six AP and four-source ARGMix conditions use different output labels and support, so their scores must stay separate. Training budgets and information access differ by method. These experiments assess complete pipelines, not architecture alone.

**No fresh matched ancestry scores or all-method superiority claim are established.** The source review does not certify that released implementations match their papers. Completed predictions and common scoring still require verification. Existing jobs were preserved.

[Scope audit and source references](research/reviews/benchmark_scientific_scope_20261023_v1/audit.json) · [European deviations](docs/european_native_comparison_deviations_v1.md) · [ARGformer paper audit](docs/argformer_published_benchmark_audit_v1.md) · [ARGMix paper audit](docs/argmix_published_benchmark_audit_v1.md)


**Deep repository reinspection, 23 October:** earlier missing-code claims were too broad. We recovered the historical AncestralPaths CNN and verified its example checkpoint metadata, ARGMix ancestor-extraction components, and deleted GSTv2/CIFAR source. We checked 9/12/27/18 reachable public commits for ARGMix/ARGformer/AncestralPaths/GST. These discoveries do not change running baselines or certify original-table reproduction. [Full corrections, file inventories and remaining work](research/reviews/author_repository_deep_audit_20261023_v1/REPORT.md).

<!-- BEGIN FILE AGREEMENT AUDIT V2 -->
**File-by-file paper agreement audit, 24 October:** three reviewers and the primary agent accounted for all **433 current official files** across ARGMix, ARGformer, AncestralPaths, GST and FLARE. Exact path sets and payload hashes were checked. Per-file records distinguish full source reading, structured data/lock scans, Java data-flow review and image metadata review.

- **ARGMix:** TreeLoom completed its fixed five-epoch fit and waits for TEST observations. The registered equation-corrected ARGMix comparator is training through the reviewed CUDA initialization retry. Final paired ancestry scores remain pending.
- **AncestralPaths:** historical CNN topology and Keras Flatten order are recovered, with a concrete inactive specification. The released Dense model and historical CNN are separate conditions. Native sampling and some downstream painting code differ from the publication recipe.
- **ARGformer:** the historical environment is recovered. Temperature, weighting, masking and extant pooling conflicts remain. The original PCA10 ancestry classifier is not released.
- **GST and FLARE:** GST synthetic totals and all seven BH training counts are resolved, while model/version/training differences remain. All FLARE test fixtures were scanned. They contain no ancestry truth and are not a substitute for the American benchmark.

**This audit changes no running baseline and establishes no new accuracy score.** Known corrections are prepared separately. Unreleased original settings remain explicit gaps, and the active queue still does not cover every original paper benchmark.

[Full audit and repair status](research/reviews/author_repository_file_agreement_20261023_v2/REPORT.md) · [Every agreement and gap](research/reviews/author_repository_file_agreement_20261023_v2/gap_register.json) · [Coverage validation](research/reviews/author_repository_file_agreement_20261023_v2/coverage_validation.json)
<!-- END FILE AGREEMENT AUDIT V2 -->

<!-- END SCIENTIFIC COMPARISON SCOPE -->

<!-- preserved-worker-controllers:start -->
<details>
<summary>Preserved controllers and earlier attempts</summary>

<!-- readable-full-vram-queue:start -->
### Current full-VRAM queue
Updated 2026-10-25T04:01:33.323012+00:00. Status: drained_for_operational_update.
85 pending, 0 running, 248 succeeded, 1 failed, 16 blocked, 0 unknown.
Local active: waiting for dependencies or capacity.
Live training stages: bounded stage metadata pending.
Standalone work: no live standalone generations.
The earlier v9 queue block is a historical snapshot. This block describes the active successor.
Ancestry accuracy is pending the frozen prediction and scoring barriers. Training loss is not accuracy.
One optional external AP001 archive is deferred with its full retained storage charge and zero credit.
[Current queue metadata](research/benchmark_full_vram_current.json).
<!-- readable-full-vram-queue:end -->

<!-- readable-recovery-afternoon-v2-queue:start -->
### Current recovery queue
Updated 2026-10-23T18:46:45.122508+00:00. Status: drained_for_operational_update.
83 pending, 0 running, 246 succeeded, 2 failed, 19 blocked, 0 unknown.
Local active: waiting for dependencies or capacity.
Live training stages: bounded stage metadata pending.
Standalone work: no live standalone generations.
The earlier v9 queue block is a historical snapshot. This block describes the active successor.
Ancestry accuracy is pending the frozen prediction and scoring barriers. Training loss is not accuracy.
One optional external AP001 archive is deferred with its full retained storage charge and zero credit.
[Current queue metadata](research/benchmark_recovery_afternoon_v2_current.json).
<!-- readable-recovery-afternoon-v2-queue:end -->

<!-- readable-recovery-afternoon-queue:start -->
### Current recovery queue
Updated 2026-10-22T17:45:34.770979+00:00. Status: drained_for_operational_update.
103 pending, 0 running, 238 succeeded, 3 failed, 5 blocked, 0 unknown.
Local active: waiting for dependencies or capacity.
Live training stages: bounded stage metadata pending.
Standalone work: no live standalone generations.
The earlier v9 queue block is a historical snapshot. This block describes the active successor.
Ancestry accuracy is pending the frozen prediction and scoring barriers. Training loss is not accuracy.
One optional external AP001 archive is deferred with its full retained storage charge and zero credit.
[Current queue metadata](research/benchmark_recovery_afternoon_current.json).
<!-- readable-recovery-afternoon-queue:end -->

<!-- readable-parallel-resource-overlay:start -->
### Current resource overlay queue
Updated 2026-10-22T17:21:56.765930+00:00. Status: drained_for_operational_update.
87 pending, 0 running, 238 succeeded, 3 failed, 21 blocked, 0 unknown.
Local active: waiting for dependencies or capacity.
Live training stages: bounded stage metadata pending.
Standalone work: no live standalone generations.
The earlier v9 queue block is a historical snapshot. This block describes the active successor.
Ancestry accuracy is pending the frozen prediction and scoring barriers. Training loss is not accuracy.
One optional external AP001 archive is deferred with its full retained storage charge and zero credit.
[Current queue metadata](research/benchmark_parallel_resource_overlay_current.json).
<!-- readable-parallel-resource-overlay:end -->

<!-- readable-parallel-queue:start -->
### Current parallel queue
Updated 2026-10-22T09:25:40.026116+00:00. Status: drained_for_operational_update.
116 pending, 0 running, 232 succeeded, 1 failed, 0 blocked, 0 unknown.
Local active: waiting for dependencies or capacity.
Live training stages: bounded stage metadata pending.
Standalone work: no live standalone generations.
The earlier v9 queue block is a historical snapshot. This block describes the active successor.
Ancestry accuracy is pending the frozen prediction and scoring barriers. Training loss is not accuracy.
One optional external AP001 archive is deferred with its full retained storage charge and zero credit.
[Current queue metadata](research/benchmark_parallel_current.json).
<!-- readable-parallel-queue:end -->

<!-- readable-recovery-queue:start -->
### Current recovery queue
Updated 2026-10-22T11:05:09.648817+00:00. Status: drained_for_operational_update.
76 pending, 0 running, 232 succeeded, 7 failed, 32 blocked, 0 unknown.
Local active: waiting for dependencies or capacity.
Live training stages: bounded stage metadata pending.
Standalone work: no live standalone generations.
The earlier v9 queue block is a historical snapshot. This block describes the active successor.
Ancestry accuracy is pending the frozen prediction and scoring barriers. Training loss is not accuracy.
One optional external AP001 archive is deferred with its full retained storage charge and zero credit.
[Current queue metadata](research/benchmark_readable_current.json).
<!-- readable-recovery-queue:end -->

</details>
<!-- preserved-worker-controllers:end -->


<!-- BEGIN COMPLETED WORK ACCOUNTING -->
## What the recent GPU jobs accomplished

Audited at 20 October 2026, 01:52 Moscow. These are verified training and prediction outputs, separate from queue preparation counts.

| Completed work | Output | Elapsed job time |
| --- | --- | ---: |
| Fresh ARGformer, first validation simulation | Five pretraining epochs, 226,885 optimizer updates | 7.65 h |
| Fresh ARGformer, second validation simulation | Five pretraining epochs, 224,170 optimizer updates | 7.50 h |
| Fresh ARGformer, first simulation, subsequent stages | 2,125 contrastive updates, embeddings and 10,000 ancestry predictions | 2.58 h |
| Earlier American GNN, separate from this fresh fit | 600 completed updates over 30 epochs, verified on 16 October | 9.12 h to update 600 |

The five completed fresh ARGformer stages took **17.74 hours of elapsed job time**. This includes loading and data movement, and does not measure GPU kernel time. An earlier separate ARGformer chain also completed 2,135 contrastive updates and retrieval. These receipts confirm real training, but do not account for every hour of the previous days, including failed attempts and repairs.

**The fresh matched comparison still has no ancestry scores.** FLARE predictions are complete for 15/15 American simulations and native ARGformer predictions for 2/15. American TreeLoom has no completed fit in this fresh campaign. Its failed 32 GiB attempt is preserved, and the repaired 64 GiB retry is queued. See the current diagnosis above for active training and memory admission.

### Scores that already exist, from earlier development experiments

These European accuracy scores are means and sample SDs across eight reused validation simulations for the present-day cohort. They are earlier development evidence, with method-specific input and training protocols, not results from the fresh benchmark.

| Method | Accuracy (%) |
| --- | ---: |
| TreeLoom | **97.08 ± 0.57** |
| Released ARGMix | 94.45 ± 1.15 |
| AncestralPaths, probabilities mapped to four classes | 93.78 ± 0.84 |

On a separate earlier set of five American validation simulations, FLARE achieved **98.10 ± 0.34% accuracy** and **97.92 ± 0.41% macro F1**. One simulation was previously inspected during development. These FLARE figures cannot be compared with the European table above.

[Completed-work audit and receipt references](research/reviews/benchmark_status_20261020/completed_work_audit.json) · [Full historical scores](research/current_scoreboard.json)
<!-- END COMPLETED WORK ACCOUNTING -->

<!-- BEGIN CURRENT RESOURCE BENCHMARK WORKFLOW -->
## Active benchmark continuation

Observed 2026-10-20T07:17:30.314165+00:00. 164 succeeded; 1 failed; 0 running; 155 pending; 16 blocked; 0 unknown.
Run status: `drained_for_operational_update`. Physical GPU: 0.

[Current queue details](research/benchmark_resource_current.json) · [Authenticated matched scores](research/resource_matched_benchmark_results.json)

| Stage | Succeeded | Failed | Running | Pending | Blocked | Unknown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ap_features | 2 | 0 | 0 | 6 | 0 | 0 |
| ap_fit | 0 | 0 | 0 | 1 | 0 | 0 |
| ap_generate | 9 | 0 | 0 | 0 | 0 | 0 |
| ap_infer | 3 | 0 | 0 | 5 | 0 | 0 |
| ap_predict | 0 | 0 | 0 | 5 | 0 | 0 |
| ap_registry | 2 | 0 | 0 | 6 | 0 | 0 |
| ap_train_targets | 0 | 0 | 0 | 3 | 0 | 0 |
| contrastive | 1 | 0 | 0 | 14 | 0 | 0 |
| embeddings | 1 | 0 | 0 | 14 | 0 | 0 |
| external | 9 | 1 | 0 | 0 | 0 | 0 |
| factory | 3 | 0 | 0 | 21 | 1 | 0 |
| fit_factory | 0 | 0 | 0 | 2 | 0 | 0 |
| focal_factory | 35 | 0 | 0 | 0 | 0 | 0 |
| generation_factory | 29 | 0 | 0 | 0 | 0 | 0 |
| inference_factory | 29 | 0 | 0 | 0 | 0 | 0 |
| mlm | 1 | 0 | 0 | 13 | 0 | 0 |
| native | 15 | 0 | 0 | 0 | 0 | 0 |
| prediction_factory | 0 | 0 | 0 | 2 | 15 | 0 |
| prepare | 16 | 0 | 0 | 13 | 0 | 0 |
| qualification_factory | 3 | 0 | 0 | 0 | 0 | 0 |
| release_archive | 0 | 0 | 0 | 4 | 0 | 0 |
| release_c4_registry | 2 | 0 | 0 | 6 | 0 | 0 |
| release_c4_score | 0 | 0 | 0 | 1 | 0 | 0 |
| release_c4_targets | 0 | 0 | 0 | 3 | 0 | 0 |
| release_fit | 0 | 0 | 0 | 1 | 0 | 0 |
| release_frame | 3 | 0 | 0 | 6 | 0 | 0 |
| release_infer | 0 | 0 | 0 | 5 | 0 | 0 |
| release_native6_score | 0 | 0 | 0 | 1 | 0 | 0 |
| release_native_targets | 0 | 0 | 0 | 4 | 0 | 0 |
| release_seal | 0 | 0 | 0 | 5 | 0 | 0 |
| retrieval | 1 | 0 | 0 | 14 | 0 | 0 |

Running workflow jobs: none.
Named external queue jobs running: none. External queue states are listed separately from workflow counts.

No supported authenticated matched score has completed in this workflow.
<!-- END CURRENT RESOURCE BENCHMARK WORKFLOW -->

<details>
<summary>Previous v8 queue and original failure evidence</summary>

<!-- BEGIN CURRENT BENCHMARK WORKFLOW -->
## Active benchmark continuation

Observed 2026-10-19T23:03:44.227173+00:00. 158 succeeded; 2 failed; 0 running; 139 pending; 27 blocked; 0 unknown.
Run status: `drained_for_operational_update`. Physical GPU: 0.

[Current queue details](research/benchmark_active_current.json) · [Authenticated matched scores](research/active_matched_benchmark_results.json)

| Stage | Succeeded | Failed | Running | Pending | Blocked | Unknown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ap_features | 2 | 0 | 0 | 5 | 1 | 0 |
| ap_fit | 0 | 0 | 0 | 1 | 0 | 0 |
| ap_generate | 9 | 0 | 0 | 0 | 0 | 0 |
| ap_infer | 3 | 0 | 0 | 5 | 0 | 0 |
| ap_predict | 0 | 0 | 0 | 4 | 1 | 0 |
| ap_registry | 2 | 0 | 0 | 5 | 1 | 0 |
| ap_train_targets | 0 | 0 | 0 | 3 | 0 | 0 |
| contrastive | 1 | 0 | 0 | 14 | 0 | 0 |
| embeddings | 1 | 0 | 0 | 14 | 0 | 0 |
| external | 8 | 1 | 0 | 0 | 0 | 0 |
| factory | 1 | 0 | 0 | 15 | 1 | 0 |
| fit_factory | 0 | 1 | 0 | 2 | 0 | 0 |
| focal_factory | 35 | 0 | 0 | 0 | 0 | 0 |
| generation_factory | 29 | 0 | 0 | 0 | 0 | 0 |
| inference_factory | 29 | 0 | 0 | 0 | 0 | 0 |
| mlm | 1 | 0 | 0 | 13 | 0 | 0 |
| native | 15 | 0 | 0 | 0 | 0 | 0 |
| prediction_factory | 0 | 0 | 0 | 0 | 17 | 0 |
| prepare | 16 | 0 | 0 | 13 | 0 | 0 |
| qualification_factory | 1 | 0 | 0 | 0 | 0 | 0 |
| release_archive | 0 | 0 | 0 | 4 | 0 | 0 |
| release_c4_registry | 2 | 0 | 0 | 5 | 1 | 0 |
| release_c4_score | 0 | 0 | 0 | 0 | 1 | 0 |
| release_c4_targets | 0 | 0 | 0 | 3 | 0 | 0 |
| release_fit | 0 | 0 | 0 | 1 | 0 | 0 |
| release_frame | 2 | 0 | 0 | 6 | 1 | 0 |
| release_infer | 0 | 0 | 0 | 4 | 1 | 0 |
| release_native6_score | 0 | 0 | 0 | 0 | 1 | 0 |
| release_native_targets | 0 | 0 | 0 | 4 | 0 | 0 |
| release_seal | 0 | 0 | 0 | 4 | 1 | 0 |
| retrieval | 1 | 0 | 0 | 14 | 0 | 0 |

Running workflow jobs: none.
Named external queue jobs running: none. External queue states are listed separately from workflow counts.

No supported authenticated matched score has completed in this workflow.
<!-- END CURRENT BENCHMARK WORKFLOW -->

</details>

<details>
<summary>Previous v7 queue, retained for its original jobs and failure history</summary>

<!-- BEGIN REPAIRED BENCHMARK WORKFLOW -->
## Active benchmark continuation

Observed 2026-10-19T03:08:12.269397+00:00. 150 succeeded; 9 failed; 0 running; 83 pending; 84 blocked; 0 unknown.
Run status: `drained_for_operational_update`. Physical GPU: 0.

[Current queue details](research/benchmark_repaired_current.json) · [Authenticated matched scores](research/repaired_matched_benchmark_results.json)

| Stage | Succeeded | Failed | Running | Pending | Blocked | Unknown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ap_features | 1 | 0 | 0 | 0 | 7 | 0 |
| ap_fit | 0 | 0 | 0 | 0 | 1 | 0 |
| ap_generate | 9 | 0 | 0 | 0 | 0 | 0 |
| ap_infer | 1 | 6 | 0 | 0 | 0 | 0 |
| ap_predict | 0 | 0 | 0 | 0 | 5 | 0 |
| ap_registry | 1 | 0 | 0 | 0 | 7 | 0 |
| ap_train_targets | 0 | 0 | 0 | 0 | 3 | 0 |
| contrastive | 1 | 0 | 0 | 14 | 0 | 0 |
| embeddings | 1 | 0 | 0 | 14 | 0 | 0 |
| external | 7 | 3 | 0 | 0 | 0 | 0 |
| factory | 1 | 0 | 0 | 14 | 2 | 0 |
| fit_factory | 0 | 0 | 0 | 0 | 3 | 0 |
| focal_factory | 35 | 0 | 0 | 0 | 0 | 0 |
| generation_factory | 29 | 0 | 0 | 0 | 0 | 0 |
| inference_factory | 29 | 0 | 0 | 0 | 0 | 0 |
| mlm | 0 | 0 | 0 | 14 | 0 | 0 |
| native | 15 | 0 | 0 | 0 | 0 | 0 |
| prediction_factory | 0 | 0 | 0 | 0 | 17 | 0 |
| prepare | 17 | 0 | 0 | 13 | 0 | 0 |
| release_archive | 0 | 0 | 0 | 0 | 4 | 0 |
| release_c4_registry | 1 | 0 | 0 | 0 | 7 | 0 |
| release_c4_score | 0 | 0 | 0 | 0 | 1 | 0 |
| release_c4_targets | 0 | 0 | 0 | 0 | 3 | 0 |
| release_fit | 0 | 0 | 0 | 0 | 1 | 0 |
| release_frame | 1 | 0 | 0 | 0 | 8 | 0 |
| release_infer | 0 | 0 | 0 | 0 | 5 | 0 |
| release_native6_score | 0 | 0 | 0 | 0 | 1 | 0 |
| release_native_targets | 0 | 0 | 0 | 0 | 4 | 0 |
| release_seal | 0 | 0 | 0 | 0 | 5 | 0 |
| retrieval | 1 | 0 | 0 | 14 | 0 | 0 |

Running workflow jobs: none.
Named external queue jobs running: none. External queue states are listed separately from workflow counts.

No supported authenticated matched score has completed in this workflow.
<!-- END REPAIRED BENCHMARK WORKFLOW -->

</details>


<details>
<summary>Previous continuation, retained while its original jobs finish</summary>

<!-- BEGIN CONTINUATION BENCHMARK WORKFLOW -->
## Active benchmark continuation

The repaired continuation is being prepared. The 21 CPU tests for worker supervision, live predecessor adoption and report isolation passed. No new scientific score is claimed.
<!-- END CONTINUATION BENCHMARK WORKFLOW -->

</details>

<details>
<summary>Previous workflow, retained while its last inference job finishes</summary>

<!-- BEGIN EXECUTABLE BENCHMARK WORKFLOW -->
## Executable benchmark workflow

Observed 2026-10-18T14:08:16.399441+00:00. 139 succeeded; 4 failed; 0 running; 46 pending; 136 blocked; 0 unknown.
Run status: `drained_for_operational_update`. Physical GPU: 0.

[Current queue details](research/benchmark_execution_current.json) · [Authenticated matched scores](research/current_matched_benchmark_results.json)

| Stage | Succeeded | Failed | Running | Pending | Blocked | Unknown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ap_features | 0 | 0 | 0 | 6 | 2 | 0 |
| ap_fit | 0 | 0 | 0 | 0 | 1 | 0 |
| ap_generate | 9 | 0 | 0 | 0 | 0 | 0 |
| ap_infer | 0 | 3 | 0 | 6 | 0 | 0 |
| ap_predict | 0 | 0 | 0 | 0 | 5 | 0 |
| ap_registry | 0 | 0 | 0 | 6 | 2 | 0 |
| ap_train_targets | 0 | 0 | 0 | 0 | 3 | 0 |
| contrastive | 0 | 0 | 0 | 0 | 15 | 0 |
| embeddings | 0 | 0 | 0 | 0 | 15 | 0 |
| external | 6 | 0 | 0 | 0 | 0 | 0 |
| factory | 0 | 0 | 0 | 0 | 17 | 0 |
| fit_factory | 0 | 0 | 0 | 1 | 2 | 0 |
| focal_factory | 35 | 0 | 0 | 0 | 0 | 0 |
| generation_factory | 29 | 0 | 0 | 0 | 0 | 0 |
| inference_factory | 29 | 0 | 0 | 0 | 0 | 0 |
| mlm | 0 | 1 | 0 | 0 | 14 | 0 |
| native | 15 | 0 | 0 | 0 | 0 | 0 |
| prediction_factory | 0 | 0 | 0 | 15 | 2 | 0 |
| prepare | 16 | 0 | 0 | 0 | 14 | 0 |
| release_archive | 0 | 0 | 0 | 0 | 4 | 0 |
| release_c4_registry | 0 | 0 | 0 | 6 | 2 | 0 |
| release_c4_score | 0 | 0 | 0 | 0 | 1 | 0 |
| release_c4_targets | 0 | 0 | 0 | 0 | 3 | 0 |
| release_fit | 0 | 0 | 0 | 0 | 1 | 0 |
| release_frame | 0 | 0 | 0 | 6 | 3 | 0 |
| release_infer | 0 | 0 | 0 | 0 | 5 | 0 |
| release_native6_score | 0 | 0 | 0 | 0 | 1 | 0 |
| release_native_targets | 0 | 0 | 0 | 0 | 4 | 0 |
| release_seal | 0 | 0 | 0 | 0 | 5 | 0 |
| retrieval | 0 | 0 | 0 | 0 | 15 | 0 |

Running workflow jobs: none.
Named external queue jobs running: none. External queue states are listed separately from workflow counts.

No supported authenticated matched score has completed in this workflow.
<!-- END EXECUTABLE BENCHMARK WORKFLOW -->

</details>

[Queue and current workstreams](docs/benchmark_queue.md) · [Executable dependency graph](configs/benchmarks/executable_workflow_20261018_v7.json)

The maintained interface is [src/arggnn/benchmarking](src/arggnn/benchmarking). It separates method adapters, biological protocols, observed-only data descriptors, single-GPU scheduling and common scoring. Original author code and immutable experiment records stay in place.

- [Workflow and release procedure](docs/benchmark_workflow.md)
- [Method implementations and adaptations](docs/benchmark_methods.md)
- [Paper recipes and remaining scientific gaps](docs/paper_benchmark_recipes.md)
- [Model lineage and lessons](docs/model_history.md)
- [Historical reconstruction protocols](configs/benchmarks/protocols)
- [Current exact-job execution policy](configs/benchmarks/execution_policy.json)
- [Complete prior research history](docs/research_history_before_benchmark_refactor_20261018.md)

## Inspect the plan

On anogena, inside the dedicated repository:

```bash
source scripts/project_env.sh
export PYTHONPATH=.:src:scripts
.runtime/env/bin/python -B -s -m arggnn.benchmarking methods
.runtime/env/bin/python -B -s -m arggnn.benchmarking paper-recipes
.runtime/env/bin/python -B -s -m arggnn.benchmarking paper-plan ancestralpaths_european
.runtime/env/bin/python -B -s -m arggnn.benchmarking status
```

The new paper plans generate each independent simulation once and share its observations, reference/query panel and frozen evaluation targets across methods. They preserve the native three-source, four-source and six-path tasks separately. Fresh seeds are permitted. The AP generator now supports the paper's 200 Mb length, complete age roster and 170-generation BAA event. Other material publication gaps remain listed in the plan and [recipe notes](docs/paper_benchmark_recipes.md).

These commands inspect small metadata. They do not train models or load scientific datasets. The plan reports missing data products, method constraints and unresolved publication details. The current continuation runs fresh native comparators while TreeLoom preprocessing is repaired. The complete American, AncestralPaths and ARGMix workflows are integrated, including shared scoring and current empirical application. Historical admissions do not authorize new launches.

## Model advantages and evidence

TreeLoom preserves complete local trees, genomic order and persistent sample identity. The selected model uses bounded GPU windows for learned history. Its replacement fit uses the repaired full-size CPU preparation and preserves the frozen training schedule.

On eight reused European development simulations, TreeLoom achieved 97.0778 ± 0.5674% modern span accuracy, compared with 94.4461 ± 1.1491% for the released ARGMix pipeline and 93.7830 ± 0.8385% for the derived four-source AncestralPaths endpoint. These are common-data development results with disclosed differences from the publications. TreeLoom-NoMem has higher boundary F1. No overall top-1 claim is established.

Native FLARE previously completed five American validation parents. The new independent American comparison is reported separately in the current workflow results. American three-source ancestry and European four-source ancestry are separate protocols. GST is an adapted architecture control rather than a native LAI competitor.

<!-- FROZEN_BENCHMARK_PROGRESS_START -->
## Historical launch snapshot and earlier results

Snapshot UTC: 2026-10-18T12:53:52.089475+00:00. Historical allocation: GPU 0. The current campaign and GPU assignment are reported above.

| Stage | Current verified status |
|---|---|
| Execution | Released by the user. Physical GPU0 only. Exact runnable jobs and dependent stages are in the maintained queue. |
| Native ARGformer | Fresh native ARGformer passed the repaired DataSpec check and reached trainer.fit, then our supervisor stopped it at the configured256GiB summed RSS cap. Optimizer update count is unverified. Memory accounting and group cleanup need repair. |
| Native follow-on | Earlier exposed-validation chain completed all four embedding stages and native retrieval with actual success. It was not scored and is separate from the fresh comparison. |
| Fresh American data | All35 independent30Mb simulations, shared inferred trees and fixed focal preparations completed. |
| Frozen TreeLoom | All three fresh fit tasks explicitly held for preprocessing repair. Stopped American first-parent preparation after more than four hours with0 optimizer updates. Preserve full20parents/30passes/600updates. |
| European comparisons | Three European inference stages are active on the completed simulations. The metadata-cap correction passed startup. No fresh European training or score yet. |
| Native FLARE | Native FLARE completed all15 fresh American validation/test predictions. Paired scoring waits for all methods. Earlier exposed validation scores remain historical. |

**Native FLARE, five American validation simulations:** accuracy 98.0960 ± 0.3442% and macro F1 97.9196 ± 0.4059%.

| Population | Precision % | Recall % | F1 % |
|---|---:|---:|---:|
| EUR | 97.3671 ± 0.5621 | 97.7505 ± 0.4517 | 97.5579 ± 0.4452 |
| EAS | 98.5230 ± 0.2364 | 98.6975 ± 0.2639 | 98.6101 ± 0.2463 |
| AFR | 98.2167 ± 0.5343 | 96.9760 ± 1.0077 | 97.5909 ± 0.6883 |

Equal weight per simulation, 10,000 fixed focal records each. SD is across five simulation parents, with ddof 1. VAL000 was previously exposed. This is a common-data development comparison, with no exact publication-table reproduction claimed.
[Complete aggregate](experiments/results/FLARE_original5VAL_aggregate_root02_memo_oct17/FLARE_original5VAL_descriptive_aggregate.json).

The substantial TreeLoom schedule is 20 full 30 Mb TRAIN parents and 30 passes, or 600 complete-parent updates from fresh weights. A disposable one-update check is not representative training. No overall top-1 claim is established.
<!-- FROZEN_BENCHMARK_PROGRESS_END -->

<!-- CURRENT_SCOREBOARD_START -->
## Completed comparison

These previously completed development results are preserved separately from the current matched benchmark results in the executable workflow report.

European development panel: eight reused simulation parents. Values are equal-parent means ± sample SD. These common-data results do not establish exact publication reproduction or independent test performance.

| Method | Present-day span accuracy % ↑ | NLL ↓ | Boundary F1 at 1 kb % ↑ |
|---|---:|---:|---:|
| TreeLoom | **97.0778 ± 0.5674** | **0.088115 ± 0.021993** | 18.3807 ± 1.3141 |
| TreeLoom-NoMem | 97.0000 ± 0.4978 | 0.088541 ± 0.019128 | **19.0488 ± 1.4275** |
| Released ARGMix | 94.4461 ± 1.1491 | 0.219695 ± 0.023975 | 11.7372 ± 1.7328 |
| AncestralPaths, derived C4 | 93.7830 ± 0.8385 | ∞ | 7.0281 ± 0.7779 |

TreeLoom-NoMem removes the persistent sample memory and retains the matched continued-training control. AncestralPaths C4 is an explicit collapse of six native path classes. Native hard calls and raw zero posterior probabilities are retained.

Native FLARE's earlier American three-source results appear above. The executable workflow report tracks new independent comparisons separately. GST expanded fitting is outside the current LAI queue.

[All recorded setup and cohort metrics](research/current_scoreboard.json) · [Prior research and results](docs/research_history_before_benchmark_refactor_20261018.md)
<!-- CURRENT_SCOREBOARD_END -->

## Paper and complete results

The manuscript is a working AAAI draft. Benchmarking is incomplete.

[Working PDF](paper/current_full_panel_working.pdf) · [LaTeX](paper/current_full_panel_working.tex) · [Overleaf bundle](paper/current_full_panel_overleaf_working.zip)

[Detailed current scores](research/current_scoreboard.json) · [Experiment ledger](research/experiment_ledger.md) · [All recorded metric receipts](research/all_recorded_metric_receipts.json) · [Research history](docs/research_history_before_benchmark_refactor_20261018.md)

Metrics retain their original support, population definitions and provenance. Sample SD across simulations and variation across optimizer seeds are different quantities. The benchmark interface rejects missing evaluation targets and keeps native hard calls and zero probabilities intact.



