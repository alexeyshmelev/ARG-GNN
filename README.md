# ARG-GNN

Graph neural networks for ordered collections of whole genealogical trees, with local ancestry inference from ancestral recombination graphs as the main application. The manuscript uses the AAAI template.

## Current execution and utilization

Updated 2026-10-11 at 06:50 UTC. **Three GPU workers run concurrently on physical GPUs 2 and 3.** They cover 19 model/seed fits. Each worker interleaves its controls, with three independent workers active together.

| GPU | Experiment | Complete epochs for every control |
| --- | --- | ---: |
| 2 | Seven mechanism controls, seed 202 | 19 |
| 3 | Cached seven mechanism controls, seed 303 | 9 |
| 3 | Whole-tree Census, five controls, seed 101 | 9 |

These are ongoing development fits, with no new completed accuracy comparison. The Census fit uses Relate trees. CPU numerical qualification on all 32 same-tsinfer tiles is still running.

**Both GPUs are active, but utilization remains bursty.** The latest 40-observation sample spans 45.6 seconds and averages **56.4% on GPU 2 and 26.8% on GPU 3**. GPU 3 has 15 zero-utilization observations, despite both workers remaining live. A separate initial snapshot showed 67%/100%, illustrating why a single NVTOP reading is insufficient. These short, phase-dependent windows do not establish a throughput improvement. [Current measurements](research/reviews/gpu_utilization_followup_v4_root/summary.json) and [worker progress and CPU/I/O evidence](research/reviews/gpu_utilization_followup_v4_root/progress.json).

The same window recorded 45.9, 46.3 and 85.4 CPU seconds for the three workers. The cached worker read 19.78 GB through read calls during this phase. This is not a measurement of physical disk traffic and does not by itself identify the costly function. CPU preparation, validation and small sequential graph operations are the current throughput investigation. Root is preparing successor work while these registered fits finish. Native ARGformer is already queued for GPU 2.

Measured gaps include CPU tree-bank loading/checking, geometry construction, validation and autograd work. Scheduler waiting was negligible in the completed [phase diagnosis](research/reviews/concurrent_worker_bottleneck_v1_author/README.md). Every GPU process uses 'AI Auto Research Pipeline Test'.

The cache preserves fields, logits, gradients, clipped optimizer updates and RNG behavior for all seven controls. One-tile warm loading averaged 0.993 seconds versus 2.023 seconds originally, a 50.9% latency reduction. Full-training speedup is unmeasured. Source inspection identified a remaining bottleneck: even hot cache hits rehash the complete original bank, and seven-control validation requests 112 banks per epoch. The [registered cache successor](research/reviews/biological_bank_snapshot_cache_v2_registration_root.json) will retain once-verified bank snapshots in CPU memory with the original training and validation loops. It is being implemented and has no actual throughput qualification yet. The cached worker is active alongside the independent whole-tree hypothesis. [Accepted paired launch](research/reviews/gpu3_parallel_fits_v2_root/plan_independent_review.json).

The next Census optimization reuses certified fixed tree geometry through the unchanged neural forward. All five controls, gradients, two clipped AdamW steps and raw validation outputs matched bitwise in the handmade checks. Independent review passed 170 fresh checks plus 6,688 replayed assertions. Genuine CPU timing now passed on the original TRAIN000 tile1, retaining all 132 trees and all five controls twice. It recorded one forest binding and nine reuses in ten calls, a 120.90-second whole process and a sampled peak RSS of 3.98 GB. The objective was squared observed logits, with no targets, optimizer or fit. This establishes execution of geometry reuse, without a paired speedup measurement. GPU and full-training benefit remain unmeasured. The accepted successor uses the user-authorized 1 TB limit and combined accounting. [Actual CPU timing](research/reviews/census_certified_forest_reuse_CPU_timing_v2_root/root_acceptance.json). [Source review](research/reviews/census_certified_forest_reuse_v1_independent/receipt.json).

Native ARGformer v7 completed actual CPU runtime capture, covering 12 scientific modules, 179 distributions and 40,619 executable import records. Its four full-batch probes, four native fits and four exhaustive retrieval stages remain pending. The earlier v6 probe reached CUDA and an optimizer step, then failed a temporary-scope guard. The source correction preserves the native scientific methods. The exact V3 GPU 2 successor queue passed independent review and is armed. Its live CPU supervisor waits for the current exclusive worker and launcher to exit naturally before the four full-batch probes. No new native GPU stage has started. [Queue activation](research/reviews/native_argformer_tsinfer_fit_v7_execution_root_v3/root_acceptance.json). [Runtime capture](experiments/results/native_argformer_tsinfer_fit_v7_root/compact_runtime_inventory.json).

The whole-parent continuity route reconstructed and assembled **all 955 original TRAIN000 trees**, all eight original tiles and all 256 cached anchors. It preserves 280 leaves, 184 query/context states, 96 references and 47 observed feature fields. This is observed-input qualification, with no targets, model profile or fitting. Full-support numerical, device and exact-mass loss checks precede the matched reset/carry experiment. [Actual input acceptance](research/reviews/anchor_parent_producer_actual_train000_v1_root/observed_acceptance.json).

## Main advantages and current evidence

**Current development result: our age-D2 candidate averages 84.78% present-day ancestry accuracy over all three planned optimizer seeds, 101/202/303. Current-release ARGMix reaches 85.55%.** Our candidate has slightly lower negative log likelihood. Overall superiority and distinct architectural novelty remain research objectives.

Measured advantages in this pilot:

- **Accuracy gains over three controls.** Our age-D2 candidate gains 2.72 percentage points over released AncestralPaths Dense, 4.75 over the GST-style width 32 reconstruction, and 0.71 over the matched GRU.
- **Better likelihood and calibration than ARGMix in this development run.** Present-day NLL is 0.4299 versus 0.4353. The same 15-bin calibration ECE is 0.0327 versus 0.0861. ARGMix has higher accuracy and a better present-day Brier score. These observations need independent confirmation.
- **Small learned model.** The candidate has 10,420 parameters, versus 28,308 for the matched GRU and 43,300 for GST32. Parameter count is 63.2% lower than the GRU. The compact implementation completed all 34 numerical probes but failed the fixed CUDA float32 parity gate. Profiling remained blocked, so training speed and memory improvements are unestablished.

Method features:

- **Whole-tree geometry.** The evaluated candidate derives interactions from complete connected marginal trees, including internal ancestors and all 280 sampled haplotypes. Its learned node features currently come from the sampled leaves. All 3,718 original inferred trees are represented.
- **Ordered genomic context.** It exchanges information across trees using genomic order and physical interval lengths, and tracks the same sampled haplotypes across neighboring trees. Reference evidence uses all 96 pure-source references, with 184 query/context states and 120 scored outputs.
- **Biological constraints under development.** A separate census-cut prototype adds a neural upward/downward whole-tree encoder and retention based on changing sample-clade membership. Older references remain ancestry proxies. Its observation-only native joins and all 32 development-tile CPU numerical checks completed, including 320 control/dtype checks with no failed cells. These checks validate implementation arithmetic, with no ancestry targets or predictive training. Predictive benefit remains untested. It is separate from the candidate reported below.

## Comparison with existing methods

Completed development results as of 2026-10-10. **Higher accuracy is better. Lower NLL, Brier and ECE are better.** Primary results concern 64 present-day query haplotypes.

| Method | Accuracy ↑ | NLL ↓ | Brier ↓ | ECE ↓ | Optimizer seeds |
| --- | ---: | ---: | ---: | ---: | --- |
| ARGMix, current author release | 85.55% | 0.4353 | 0.2191 | 0.0861 | 50 |
| Our candidate, age D2 | 84.78% | 0.4299 | 0.2244 | 0.0327 | 101, 202, 303 |
| Our candidate, plain D2 | 84.79% | 0.4299 | 0.2247 | 0.0325 | 101, 202, 303 |
| Matched GRU D2 | 84.07% | 0.4541 | 0.2351 | 0.0306 | 101, 202, 303 |
| AncestralPaths, released Dense | 82.06% | 0.8191† | 0.2857† | 0.0940† | 101, 202, 303 |
| GST-style reconstruction, width 32 | 80.03% | 0.5689 | 0.2920 | 0.0236 | 101 |
| GST-style reconstruction, width 64 | 79.86% | 0.5690 | 0.2928 | 0.0310 | 101 |
| ARGformer, author implementation | Pending qualified comparison | — | — | — | Four native corpora prepared, fits pending |

† AncestralPaths NLL, Brier and ECE use a disclosed auxiliary stable-logit evaluation with unchanged trained weights and identical class predictions. Its original raw probability underflow and infinite NLL are preserved in the experiment records. No probability floor is applied.

Metrics: accuracy is the fraction of correctly assigned query base pairs. NLL is negative log probability of the true class, in nats. Brier is the sum of squared errors across the four class probabilities. ECE measures the confidence-versus-accuracy gap in 15 equally spaced confidence bins. Scores weight exact physical query support and average parent-level values. GST32 has the lowest ECE in this pilot, despite its lower accuracy. Calibration alone does not establish better ancestry inference.

**Against our strongest current comparator:** the age-D2 candidate trails ARGMix by 0.77 accuracy percentage points. Age and plain candidates differ by only 0.01 points. Neither age conditioning nor overall superiority is established by this pilot.

Other query strata, using the same selected models:

| Query group | Candidate accuracy | ARGMix accuracy | Candidate NLL | ARGMix NLL |
| --- | ---: | ---: | ---: | ---: |
| Ancient queries, 56 haplotypes | 88.43% | 89.33% | 0.3281 | 0.3433 |
| All queries, 120 haplotypes | 86.48% | 87.32% | 0.3824 | 0.3924 |

### Completed mechanism ablations, seed101

Same two development VAL parents and64 present-day queries as the main biological comparison. Each row uses its own NLL-selected checkpoint. These single-seed results are separate from the three-seed means above, with independent confirmation pending.

| Model mechanism | Accuracy ↑ | NLL ↓ |
| --- | ---: | ---: |
| scalar age d2 | 84.80% | 0.4281 |
| age propagation only d2 | 84.86% | 0.4279 |
| age source only d2 | 84.85% | 0.4269 |
| age gainmatched d2 | 83.73% | 0.4531 |
| noncontext age d2 | 84.88% | 0.4262 |
| age d1 | 84.70% | 0.4258 |
| age d1 plus pointwise | 84.74% | 0.4246 |

The noncontext control has the highest selected accuracy and depth1-plus-pointwise the lowest selected NLL. These results do not establish an age or cross-query feedback benefit. [Exact metrics and original output bindings](experiments/results/biological_mechanism_seed101_summary_v1_root/summary.json).

### What these comparisons mean

The common task is four-source ancestry, ANA/CHG/WHG/EHG, at census 350 generations. Data comprise four independent 2 Mb simulation parents, split into two TRAIN and two VAL parents before extraction. Each parent has 96 pure references, 64 unlabeled mixed-context haplotypes and 120 queries, including 64 present-day and 56 ancient queries. All methods are scored on identical observed physical coverage and query identities. Accuracy, NLL and Brier weight exact class-span intersections within 100 bp bins, then average the two parent scores equally.

These are validation results used during model development and checkpoint selection. **No final TEST results or statistical superiority claim are reported.** All three planned candidate/GRU optimizer seeds, 101/202/303, are completed and included. Optimizer seeds are not independent biological replicates. Seed 101 mechanism controls are completed. Seven controls continue under seed 202 on GPU 2, while seed 303 uses the qualified cache on GPU 3 alongside the five-control whole-tree Census experiment. Seed means summarize separate fits, without ensembling probabilities. All optimizer repeats reuse the same two validation parents.

Baseline fidelity:

- **ARGMix:** current author implementation at commit `c2481248`, with its native model, objective, checkpoint selection, refinement and overlap averaging retained. This is a common-task transfer. The available release differs from the April preprint, so this table is not its numerical reproduction.
- **AncestralPaths:** released TensorFlow Dense model, preserving its native features and six path outputs, followed by a disclosed six-path-to-four-source probability mapping. The paper's unspecified CNN is not reproduced by this released Dense baseline.
- **GST:** independently reconstructed Graph Set Transformer style controls adapted to per-query ancestry outputs. These rows are architectural controls, not executions of the authors' implementation.
- **ARGformer:** native training and retrieval ran on Relate trees. That export does not preserve shared internal ancestor IDs across trees, which ARGformer requires. Its ranking is withheld and a separate current tsinfer/tsdate inference condition has now completed for all four development parents, preserving genotypes, sampling ages and connected shared internal IDs. The native tskit 0.6.4 reader loaded the exact new tree written with tskit 1.0.3, preserving all 280 sample IDs without conversion. Full native path/corpus preparation now passed on all four development parents, including the author tokenizer, genuine shared internal IDs, both path splits and full pure-reference corpus. MLM uses paths from all 280 observed samples. Pure-reference filtering to 96 samples applies to contrastive training and retrieval. The original 512-sample MLM and 4,096-sample contrastive batches are feasible in every corpus. Native fits and retrieval on this new condition are pending. Shared ancestry IDs will not be fabricated from local numbering or matching clades.

Input features, supervision, native selection rules and post-processing differ between methods and are disclosed in the [baseline fidelity policy](docs/biological_baseline_fidelity_v1.md). Matched scoring support provides a common endpoint, rather than an exact reproduction of each paper's original dataset and pipeline.

## Original paper data requirement

Final evaluation must use each supplied paper's data and task, with exact data, split, preprocessing, native method, checkpoint selection, postprocessing and scoring identities. Every change is recorded. The current development table does not fulfill this milestone. The effective requirements passed an independent source audit, which establishes documentary fidelity only. Original-paper empirical comparisons remain pending. See the [paper-dataset evaluation contract](docs/paper_dataset_evaluation_contract_v3.md) for concrete acquisition routes and unresolved original data/model details.

All 39 ancient-inclusive Wohns genealogy files cited by ARGformer and both pinned GST raw snapshots have now been acquired and checksum-verified, totaling 2.73 GB. Their original processed subsets and splits are still to be qualified. Acquisition alone does not establish a completed comparison. The GST USPTO archive has also been extracted without changing member names, order or bytes. Root independently verified CRC and SHA for all four files, totaling 12,710,870 bytes. Native parsing, graph features, split fidelity and model execution remain pending. [Exact payload receipt](research/reviews/gst_uspto_payload_copy_v2_root/root_acceptance.json).

A smaller simulation, current code release or changed ancestry endpoint is reported under its own label. None can silently stand in for the original experiment. Each comparison will publish raw-data hashes, reference/query identities, splits and an adaptation record covering inputs, model, training, postprocessing and metrics. Missing original assets remain explicit. The candidate and genuine comparator must share the same scoring queries and coverage for each paired endpoint.

**Final comparison gate:** freeze train/validation/test membership before fitting, run the genuine author method and our approach on the same retained paper-task examples, and score identical query identities, genomic intervals and metrics. Publish the complete preprocessing, training, model-selection and postprocessing adaptation record, including an explicit “none” where no change was made. A modified architecture or ancestry endpoint gets a separately named row. Missing original assets prevent an exact-reproduction claim.

## Evidence and research log

- [Full README metrics and pairwise differences](experiments/results/readme_biological_summary_v4/summary.json).
- [Concise comparison status](docs/biological_comparison_status_v1.md) and [machine-readable table](experiments/results/biological_comparison_status_v1/summary.json).
- [Completed seed 303 results](experiments/results/biological_candidate_pilot_v1_seed303_summary/summary.json), [raw acquisition receipt](experiments/results/paper_native_raw_acquisition_v1/completion.json), and [completed seed 202 results](experiments/results/biological_candidate_pilot_v1_seed202_summary/summary.json) and [four-parent observed tsinfer/tsdate qualification](experiments/results/biological_observed_tsinfer_v2_summary/summary.json).
- [Candidate results](experiments/results/biological_candidate_pilot_v1_seed101_summary/summary.json), [ARGMix results](experiments/results/official_argmix_biological_metadata_repair_v3_summary/summary.json), [AncestralPaths stable evaluation](experiments/results/ancestralpaths_stable_readout_v1/summary.json), and [GST results](experiments/results/biological_gst_control_v1_seed101_summary/summary.json).
- [Biological benchmark contract](docs/biological_benchmark_contract_v1.md), [estimand and features](docs/biological_estimand_and_features_v1.md), and [census prototype](docs/census_tree_encoder_v1.md).
- [Completed census numerical qualification](experiments/results/census_native_actual32_numerical_v1_summary/summary.json) and [matched-tree biological comparison audit](docs/matched_biological_comparison_audit_v1.md).
- [Four qualified native ARGformer corpora](experiments/results/official_argformer_tsinfer_parent_v3_summary/summary.json).
- [Experiment ledger](research/experiment_ledger.md), [exact result index](research/experiment_results.csv), and [working AAAI manuscript](paper/anchor_kernel_working.tex).

## Environment and contributions

The project runs on anogena in `/workspace-SR003.nfs2/ARG-GNN`. Read [AGENTS.md](AGENTS.md) before contributing. Source `scripts/project_env.sh` before installing packages or running Python. The main environment is `.runtime/env`. Runtime state, datasets and checkpoints stay on the server and are excluded from Git.

The user increased the project storage allowance to **1 TB on 2026-10-11**, with a 950 GB scheduling ceiling. The latest completed allocated-size scan measured **232.12 GB**. Combined conservative storage reservations project **868.51 GB**, which is an accounting projection rather than actual used space. See the [resource authorization](research/resource_authorizations/project_storage_1TB_20261011.json).

Current resource execution is summarized at the top of this README. The corrected GPU 3 pool admits the cached mechanism worker and the whole-tree Census worker together. Their conservative reservations total 64 GiB, plus 8 GiB device headroom, within the A100's 80 GiB. The earlier cache launch failed on a missing storage-snapshot command field and was stopped by root. Its partial Census artifacts and failure assessment are preserved. The corrected snapshot passed the unchanged original storage class before relaunch. See the [paired pool](research/reviews/gpu3_parallel_fits_v2_root/pool_admission.json), [original snapshot preflight](research/reviews/gpu3_parallel_fits_v2_root/original_snapshot_preflight.json), and [preserved failure assessment](research/reviews/gpu3_cache_snapshot_failure_v1_independent/assessment.json).

A new [boundary diagnostic](docs/biological_tile_boundary_diagnostic_v1.md) reuses unchanged completed predictions on the two development VAL parents. Age-D2 accuracy trails current-release ARGMix by 1.61 points within 10 kb of artificial tile boundaries, 1.82 points at 10–25 kb, and 0.57 points at 50–125 kb. The overall deficit remains 0.77 points. This exploratory association motivates testing cross-tile continuity. It does not establish a cause or a new accuracy gain.

Native ARGformer v7 completed genuine CPU-only runtime capture with 12 scientific modules and 179 distributions in the original environment. Full native fits remain pending. The census fit v2 protocol and GPU profile source passed independent review. The separate same-tsinfer census observation inspection completed on CPU: all 32 cells passed, covering 15,956 marginal trees. Root froze the 32 certificates and rehashed all 64 materialized files (2.16 GB), then rechecked 110 source/observed/metadata objects. CPU numerical qualification is running. Dedicated same-tsinfer device/fit consumers passed source and handmade independent review, while actual same-tsinfer GPU profiling and fitting remain pending. See the [native capture evidence](experiments/results/native_argformer_tsinfer_fit_v7_root/compact_runtime_inventory.json), [fit protocol](docs/census_biological_fit_v2.md), and [same-tsinfer bridge](docs/census_tsinfer_observation_v1.md). The native ARGformer plan requires four same-parent full-batch 4,096 probes, four genuine fits, and four exhaustive retrieval stages. Each corpus retains the original native batches and all pure-reference paths. The completed runtime capture does not qualify memory or predict accuracy. See the [12-stage packet](research/reviews/native_argformer_tsinfer_fit_v6_execution_root/execution_plan.json) and [same-tree inspection freeze](research/reviews/census_tsinfer_actual32_numerical_v1_root/inspection_freeze.json).

Run `.runtime/env/bin/python scripts/update_research_ledger.py` after new results. Record a hypothesis or explicit repeat reason before each substantive experiment. Preserve complete outcomes, provenance and failures, and push concise conclusions regularly.

The next scientific checks are native ARGformer fits and full-corpus retrieval, completed optimizer/mechanism comparisons, independent simulation-parent confirmation, demographic shifts and a justified real-data endpoint. Published ancestry paintings are comparator outputs, rather than ground truth.
