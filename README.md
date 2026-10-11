# ARG-GNN

Graph neural networks for ordered collections of whole genealogical trees, with local ancestry inference from ancestral recombination graphs as the main application. The manuscript uses the AAAI template.

## Current experiments

Updated 2026-10-11 17:20 UTC.

**The completed FP32 discovery comparisons below use one seed and 160 updates per arm.** All rows below use the same two repeatedly reused development VAL parents, present-day query identities and exact physical scoring support. Each trained arm uses its prespecified minimum-NLL checkpoint.

| Completed development pipeline | Present-day accuracy ↑ | NLL ↓ |
| --- | ---: | ---: |
| Frozen native ARGMix | 85.55% | 0.4353 |
| ARGMix plus ordinary smoothing, H025 | 85.94% | 0.3891 |
| Smoothing plus native-age reference context, H026 | 85.91% | 0.3824 |
| Smoothing plus census350 reference context, H026 | 85.91% | 0.3824 |
| Smoothing plus sample-only reference propagation, H027 | 85.93% | 0.3827 |
| Smoothing plus genuine ancestral-reference propagation, H027 | 85.93% | 0.3826 |
| Smoothing plus sham ancestral-reference propagation, H027 | 86.04% | 0.3837 |
| Smoothing plus fixed observed-genotype context, H028 | 85.92% | 0.3888 |
| Smoothing plus fixed genotype/graph mean fusion, H028 | 85.90% | 0.3852 |
| Smoothing plus learned static whole-tree GNN, H029 | 85.956% | 0.38318 |
| Smoothing plus carrier-GRU whole-tree GNN, H029 | 85.939% | 0.38795 |
| Smoothing plus carrier-matrix whole-tree GNN, H029 | 85.956% | 0.38316 |
| Smoothing plus shuffled-history GNN control, H029 | 85.958% | 0.38315 |

**Observed reference context improves likelihood, but genuine ancestral continuity and census conditioning have no demonstrated accuracy advantage.** The sham row is a control, not the proposed method. Its higher accuracy cannot support a genuine ancestry mechanism claim. We have not established a novel GNN contribution or final-test superiority.

[H026 selected results and parent/class/boundary diagnostics](experiments/results/reference_census_innovation_v1_summary_root/summary.json) and [H027 selected results and diagnostics](experiments/results/reference_trace_residual_v1_summary_root/summary.json). H026 native, age-free and census arms select epoch10. Its sham selects epoch6. H027 arms select epoch10. H027's positive normalization-mass check passed, but very small reference masses need care in interpreting local normalized fields.

- **H028 completed:** all three fixed genotype/graph/fusion heads finished 10 FP32 epochs and 160 updates, seed 101. The genotype-only and fixed-fusion routes do not improve smoothing accuracy. Graph-only exactly matches H026 native-reference selected metrics and is an explicitly disclosed matched-control repeat. Each head allocates 32 parameters, 28 with input signal because the older ANA channel is absent. This does not establish that raw SNPs lack signal. [Selected comparison and diagnostics](experiments/results/observed_variant_residual_v1_summary_root/summary.json).
- **H029 completed:** all four whole-tree models finished 10 FP32 epochs and 160 updates per arm, seed 101. Static/matrix accuracy exceeds smoothing by about 0.016 percentage points. Matrix does not beat static or shuffled-history controls, and H026 native reference context has lower NLL. The GRU selects epoch 7, the other arms epoch 10 by minimum VAL NLL. GRU and matrix use 16 state scalars per carrier/direction, with unequal active parameter counts. [Selected comparison, retained-file verification and diagnostics](experiments/results/edge_innovation_gnn_v1_summary_root/summary.json). The first real TRAIN tile passed finite FP32 loss/gradient checks for 359,640 augmented vertices and 550 marginal holds, with a cumulative allocator peak of 2.69 GB.
- **H030 recovery running on GPU 3:** the model and AdamW states resume from durable epochs 4/4/3/3/3. The worker has passed CUDA and NVTOP process-title checks and is preparing the graphs. It will execute 528 remaining updates to finish 160 logical updates per arm. The original partial run and 48 unsaved updates that require replay remain disclosed. Static epoch 3 NLL was 0.37976 versus smoothing 0.38914, while accuracy was 85.8835% versus 85.9397%. These are interim results. [Recovery startup](experiments/results/reference_logcavity_resume_full_storage_v1_seed101_GPU3/started.json), [restoration protocol](docs/reference_logcavity_resume_v1.md).
- **H031 completed:** direct fixed SNP evidence scores 42.34% on development VAL. On baseline-error spans it scores 26.84%/29.16%, below the unfitted observed-donor-prior ANA prediction at 28.05%/31.91%. This evidence does not justify a strong complementary-signal claim. Its oracle union is only a diagnostic bound. [Results and source-prior context](experiments/results/observed_variant_signal_diagnostic_v1_summary_root/summary.json).
- **H032 running with real FP32 updates on GPU 2:** all three donor-support Gram conditions have completed at least 112 updates per arm. Latest tree-Gram Present64 accuracy is 85.942% and NLL is 0.37549, compared with completed H025 smoothing at 85.940% and 0.38914. Likelihood improves, but accuracy and genuine tree dependence have no demonstrated gain. This is an unfinished development comparison. [First actual optimizer update](experiments/results/reference_dependence_full_storage_v1_seed101_GPU2/first_training_tile.json), [latest tree-Gram metrics](experiments/results/reference_dependence_full_storage_v1_seed101_GPU2/tree_gram.epoch_07.metrics.json), [protocol](docs/reference_dependence_gnn_v1.md).
- **Native ARGformer:** genuine author training and retrieval completed on all four adapted EuropeanC4/tsinfer parents. Present-day top1 accuracy is75.68%, fixed top5 is75.14%, and unmodified raw-vote NLL is infinite. The main comparison below explains the differing genealogy and training objective.
- **Original GST:** all four native USPTO models completed the bounded TRAIN/VALID pilot below.
- **Previous partial runs:** interrupted mechanism-seed and Census fits remain preserved. They are not final30epoch results. Large FP64 equivalence sweeps remain deferred.

Frozen ARGMix predictions derive from Relate trees. The new graph context comes from independently inferred tsinfer/tsdate trees. TRAIN emitter predictions are in-sample and the frozen smoother was development-selected. No TEST parent has been evaluated.

Novelty is assessed throughout development in the [claim-by-claim novelty log](docs/novelty_assessment_v1.md). The latest independent [design critique](research/reviews/edge_innovation_novelty_fresh_v1/critique.md) identifies TGN and Gated DeltaNet precedents. Whole-tree cavity construction plus carrier-conditioned memory remains a hypothesis about useful biological coupling, with priority and predictive benefit unresolved. Existing kernels, recurrence and message passing are discovery tools.

At most physical GPUs2 and3 are assigned to the project. GPU workers are titled 'AI Auto Research Pipeline Test'. Current storage authorization is1TB. See [development priorities](docs/development_priorities_20261011.md).

## Main advantages and current evidence

**The new static whole-tree GNN scores 85.956%, versus 85.940% for ordinary smoothing and 85.554% for frozen ARGMix.** The added GNN gain is about 0.016 percentage points on reused development data. Matrix memory does not improve over static or shuffled history, so it does not establish the requested ancestry mechanism. H026 native-age reference context has the lower NLL, 0.3824, with 85.91% accuracy. The higher86.04% sham accuracy is reported transparently as a control result. The earlier standalone age-D2 candidate averages84.78% over optimizer seeds101/202/303. A distinct, stronger GNN contribution remains a research objective.

Measured advantages of the earlier standalone candidate:

- **Accuracy gains over three controls.** Our age-D2 candidate gains 2.72 percentage points over released AncestralPaths Dense, 4.75 over the GST-style width 32 reconstruction, and 0.71 over the matched GRU.
- **Better likelihood and calibration than ARGMix in this development run.** Present-day NLL is 0.4299 versus 0.4353. The same 15-bin calibration ECE is 0.0327 versus 0.0861. ARGMix has higher accuracy and a better present-day Brier score. These observations need independent confirmation.
- **Small learned model.** The candidate has 10,420 parameters, versus 28,308 for the matched GRU and 43,300 for GST32. Parameter count is 63.2% lower than the GRU. The compact implementation completed all 34 numerical probes but failed the fixed CUDA float32 parity gate. Profiling remained blocked, so training speed and memory improvements are unestablished.

Method features:

- **Whole-tree geometry.** The evaluated candidate derives interactions from complete connected marginal trees, including internal ancestors and all 280 sampled haplotypes. Its learned node features currently come from the sampled leaves. All 3,718 original inferred trees are represented.
- **Ordered genomic context.** It exchanges information across trees using genomic order and physical interval lengths, and tracks the same sampled haplotypes across neighboring trees. Reference evidence uses all 96 pure-source references, with 184 query/context states and 120 scored outputs.
- **Biological constraints under development.** A separate census-cut prototype adds a neural upward/downward whole-tree encoder and retention based on changing sample-clade membership. Older references remain ancestry proxies. Its observation-only joins and numerical checks completed. Its five-control predictive fit was interrupted after 22 complete metric epochs. Partial observations did not justify further unchanged training. It is separate from the candidate reported below.

## Comparison with existing methods

Completed development results, including the 2026-10-11 FP32 refinement pilot. **Higher accuracy is better. Lower NLL, Brier and ECE are better.** Primary results concern 64 present-day query haplotypes.

| Method | Accuracy ↑ | NLL ↓ | Brier ↓ | ECE ↓ | Optimizer seeds |
| --- | ---: | ---: | ---: | ---: | --- |
| Smoothing plus static whole-tree GNN, H029 | 85.956% | 0.38318 | 0.1980 | 0.0206 | 101, emitter 50 |
| Smoothing plus carrier-matrix GNN, H029 | 85.956% | 0.38316 | 0.1980 | 0.0206 | 101, emitter 50 |
| Smoothing plus native-age reference context, H026 | 85.91% | 0.3824 | 0.1977 | 0.0250 | 101, emitter 50 |
| Smoothing plus genuine ancestral-reference context, H027 | 85.93% | 0.3826 | 0.1978 | 0.0237 | 101, emitter 50 |
| ARGMix plus ordinary smoothing, development refinement | 85.94% | 0.3891 | 0.2004 | 0.0260 | 101, emitter 50 |
| ARGMix, current author release | 85.55% | 0.4353 | 0.2191 | 0.0861 | 50 |
| Our candidate, age D2 | 84.78% | 0.4299 | 0.2244 | 0.0327 | 101, 202, 303 |
| Our candidate, plain D2 | 84.79% | 0.4299 | 0.2247 | 0.0325 | 101, 202, 303 |
| Matched GRU D2 | 84.07% | 0.4541 | 0.2351 | 0.0306 | 101, 202, 303 |
| AncestralPaths, released Dense | 82.06% | 0.8191† | 0.2857† | 0.0940† | 101, 202, 303 |
| GST-style reconstruction, width 32 | 80.03% | 0.5689 | 0.2920 | 0.0236 | 101 |
| GST-style reconstruction, width 64 | 79.86% | 0.5690 | 0.2928 | 0.0310 | 101 |
| ARGformer, genuine native tsinfer pilot, top1 | 75.68% | ∞ | 0.4864 | 0.2432 | Native5MLM+1contrastiveepoch perARG |

ARGformer uses unchanged author model, objectives, full native batches and retrieval. It was fitted separately on each observed ARG, including VAL observations without query ancestry labels. Primary top1 is75.68% and prespecified top5 is75.14%. Raw hard votes assign zero probability to some true classes, preserving infinite NLL. Its tsinfer/tsdate genealogy differs from the supervised Relate ARGMix pipeline. These are descriptive scores on identical query/physical support, with a reduced EuropeanC4 transfer task and brief training, rather than original-paper numerical reproduction or a matched-topology architecture comparison. [Native ancestry summary and diagnostics](experiments/results/native_argformer_tsinfer_fit_v7_summary_v2/summary.json).

† AncestralPaths NLL, Brier and ECE use a disclosed auxiliary stable-logit evaluation with unchanged trained weights and identical class predictions. Its original raw probability underflow and infinite NLL are preserved in the experiment records. No probability floor is applied.

Metrics: accuracy is the fraction of correctly assigned query base pairs. NLL is negative log probability of the true class, in nats. Brier is the sum of squared errors across the four class probabilities. ECE measures the confidence-versus-accuracy gap in 15 equally spaced confidence bins. Scores weight exact physical query support and average parent-level values. GST32 has the lowest ECE in this pilot, despite its lower accuracy. Calibration alone does not establish better ancestry inference.

**For the earlier standalone candidate:** age-D2 trails native ARGMix by 0.77 accuracy percentage points. Age and plain candidates differ by only 0.01 points. Neither age conditioning nor overall superiority is established by this pilot.

Other query strata, using the same selected models:

| Query group | Candidate accuracy | ARGMix accuracy | Candidate NLL | ARGMix NLL |
| --- | ---: | ---: | ---: | ---: |
| Ancient queries, 56 haplotypes | 88.43% | 89.33% | 0.3281 | 0.3433 |
| All queries, 120 haplotypes | 86.48% | 87.32% | 0.3824 | 0.3924 |

### Completed mechanism ablations, seeds 101 and 202

Same two development VAL parents and 64 present-day queries as the main biological comparison. Each row uses its own NLL-selected checkpoint. These are optimizer repeats on the same biological parents.

| Model mechanism | Accuracy, seed 101 ↑ | Accuracy, seed 202 ↑ | NLL, seed 101 ↓ | NLL, seed 202 ↓ |
| --- | ---: | ---: | ---: | ---: |
| Scalar age D2 | 84.80% | 84.64% | 0.4281 | 0.4343 |
| Age propagation only D2 | 84.86% | 84.74% | 0.4279 | 0.4338 |
| Age source only D2 | 84.85% | 84.65% | 0.4269 | 0.4331 |
| Age gain-matched D2 | 83.73% | 83.14% | 0.4531 | 0.4530 |
| Noncontext age D2 | 84.88% | 84.66% | 0.4262 | 0.4327 |
| Age D1 | 84.70% | 84.87% | 0.4258 | 0.4211 |
| Age D1 plus pointwise | 84.74% | 84.28% | 0.4246 | 0.4327 |

Age D1 has the lowest two-seed mean NLL, 0.4235. The pointwise addition has no consistent benefit, and the gain-matched control is weaker in both seeds. Cross-query feedback and age-conditioning advantages are not established. The seed 303 mechanism repeat was interrupted during the user-directed shift to one-seed architecture discovery. Its partial artifacts are retained. [Seed 101 metrics](experiments/results/biological_mechanism_seed101_summary_v1_root/summary.json) and [seed 202 metrics](experiments/results/biological_mechanism_seed202_summary_v1_root/summary.json).

### Original Graph Set Transformer on USPTO

Executed the authors' unchanged parser, models and training function at commit '6469d56', using the first 1,000 retained TRAIN reactions and all 1,495 retained VALID reactions, 20 epochs and native seed 0. Runtime was ported to Python 3.14.3 and CPU Torch 2.10. The validation set was supplied in both native evaluation slots to keep TEST unopened.

| Native model | Selected VALID atom AP ↑ |
| --- | ---: |
| GSC-CA | 0.6379 |
| GSC-BC | 0.6538 |
| GCN-DS | 0.6395 |
| GCN-ST | 0.6382 |

These are reaction-center scores on the original dataset's development split, separate from the GST-style ancestry controls above. Our approach has not run on this task yet. This bounded validation pilot is not a numerical reproduction of the paper or a final-test comparison. [Native results and disclosed changes](experiments/results/gst_uspto_native_TRAIN_VALID_seed0_v1/summary.json).

### What these comparisons mean

The common task is four-source ancestry, ANA/CHG/WHG/EHG, at census 350 generations. Data comprise four independent 2 Mb simulation parents, split into two TRAIN and two VAL parents before extraction. Each parent has 96 pure references, 64 unlabeled mixed-context haplotypes and 120 queries, including 64 present-day and 56 ancient queries. All methods are scored on identical observed physical coverage and query identities. Accuracy, NLL and Brier weight exact class-span intersections within 100 bp bins, then average the two parent scores equally.

These are validation results used during model development and checkpoint selection. **No final TEST results or statistical superiority claim are reported.** All three planned candidate/GRU optimizer seeds, 101/202/303, are completed and included. Optimizer seeds are not independent biological replicates. Mechanism controls are completed for seeds 101 and 202. The seed 303 mechanism repeat and the Census experiment were interrupted at saved checkpoints to prioritize the new architecture pilot. Neither interrupted run is a final 30-epoch result. Seed means summarize separate fits, without ensembling probabilities. All optimizer repeats reuse the same two validation parents.

Baseline fidelity:

- **ARGMix:** current author implementation at commit `c2481248`, with its native model, objective, checkpoint selection, refinement and overlap averaging retained. This is a common-task transfer. The available release differs from the April preprint, so this table is not its numerical reproduction.
- **AncestralPaths:** released TensorFlow Dense model, preserving its native features and six path outputs, followed by a disclosed six-path-to-four-source probability mapping. The paper's unspecified CNN is not reproduced by this released Dense baseline.
- **GST:** independently reconstructed Graph Set Transformer style controls adapted to per-query ancestry outputs. These rows are architectural controls, not executions of the authors' implementation.
- **ARGformer:** native training and retrieval ran on Relate trees. That export does not preserve shared internal ancestor IDs across trees, which ARGformer requires. Its ranking is withheld and a separate current tsinfer/tsdate inference condition has now completed for all four development parents, preserving genotypes, sampling ages and connected shared internal IDs. The native tskit 0.6.4 reader loaded the exact new tree written with tskit 1.0.3, preserving all 280 sample IDs without conversion. Full native path/corpus preparation now passed on all four development parents, including the author tokenizer, genuine shared internal IDs, both path splits and full pure-reference corpus. MLM uses paths from all 280 observed samples. Pure-reference filtering to 96 samples applies to contrastive training and retrieval. The original 512-sample MLM and 4,096-sample contrastive batches are feasible in every corpus. All four native fits and exhaustive retrievals on this condition completed. The genuine VAL ancestry comparison is reported above, with raw top1 and prespecified top5 votes. Shared ancestry IDs will not be fabricated from local numbering or matching clades.

Input features, supervision, native selection rules and post-processing differ between methods and are disclosed in the [baseline fidelity policy](docs/biological_baseline_fidelity_v1.md). Matched scoring support provides a common endpoint, rather than an exact reproduction of each paper's original dataset and pipeline.

## Original paper data requirement

Final evaluation must use each supplied paper's data and task, with exact data, split, preprocessing, native method, checkpoint selection, postprocessing and scoring identities. Every change is recorded. The current development table does not fulfill this milestone. The effective requirements passed an independent source audit, which establishes documentary fidelity only. Original-paper empirical comparisons remain pending. See the [paper-dataset evaluation contract](docs/paper_dataset_evaluation_contract_v3.md) for concrete acquisition routes and unresolved original data/model details.

All 39 ancient-inclusive Wohns genealogy files cited by ARGformer and both pinned GST raw snapshots have now been acquired and checksum-verified, totaling 2.73 GB. Their original processed subsets and splits are still to be qualified. Acquisition alone does not establish a completed comparison. The GST USPTO archive has also been extracted without changing member names, order or bytes. Root independently verified CRC and SHA for all four files, totaling 12,710,870 bytes. Native parsing and all four native models have now run on a bounded TRAIN/VALID pilot, described below. Our approach and final TEST remain unevaluated on this task. [Exact payload receipt](research/reviews/gst_uspto_payload_copy_v2_root/root_acceptance.json).

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

The user increased the project storage allowance to **1 TB on2026-10-11**, with a950GB scheduling ceiling. The latest completed whole-project measurement was about289.7GB. Historical conservative reservations project949.53GB, an accounting projection rather than physical usage. A scoped, append-only accounting successor has source and handmade checks ready for review. It has not been activated and no historical charge has been released. It is intended to close completed reservations without rescanning the whole repository or rewriting old admissions. See the [resource authorization](research/resource_authorizations/project_storage_1TB_20261011.json).

Native ARGformer v7 completed genuine fitting and retrieval in the original captured author environment. Both VAL predictions were sealed before unchanged external scoring. The same-tsinfer Census observation and CPU numerical checks completed, but this separate interrupted model has no final superiority result. [Native capture evidence](experiments/results/native_argformer_tsinfer_fit_v7_root/compact_runtime_inventory.json) and [completed ancestry comparison](experiments/results/native_argformer_tsinfer_fit_v7_summary_v2/summary.json).

Run `.runtime/env/bin/python scripts/update_research_ledger.py` after new results. Record a hypothesis or explicit repeat reason before each substantive experiment. Preserve complete outcomes, provenance and failures, and push concise conclusions regularly.

The next scientific checks are accuracy and mechanism discovery with matched controls, followed by independent simulation-parent confirmation, original-paper task comparisons, demographic shifts and a justified real-data endpoint. Published ancestry paintings are comparator outputs, rather than ground truth.
