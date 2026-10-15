# ARG-GNN

Graph neural networks for ordered collections of whole genealogical trees, with local ancestry inference from ancestral recombination graphs as the main application. The manuscript uses the AAAI template.

## Latest status

**Benchmark scope correction (15 October):** the completed comparison below uses our 476-diploid, 2 Mb development panel. The ARGMix paper reports a 1,611-diploid setup that excludes BAA. These are different experiments. The latest 2.1317 pp development gain does not demonstrate improvement over the published ARGMix tables, and ARGformer’s published American EUR/EAS/AFR benchmark is also unreplicated. Full published-setup comparisons, training-seed variation and real-data PCA validation are outstanding.

Our current best development model uses PyTorch Geometric whole-tree message passing, observed SNP features, exact local topology-change flags and a bidirectional GRU across genomic trees. It has 30,788 parameters and uses no ARGMix predictions. The original H038 backbone has 29,764 parameters.

The full-panel development experiment uses **32 independent training simulations and eight validation simulations**, with 952 observed haplotypes, 686 queries and 266 proxies. All simulations, Relate inference and observation banks completed. Eight reserved outer simulations remain ungenerated and untouched.

**Both FP32 PyG fits completed 30 passes and 960 whole-parent optimizer updates each.**

| Full-panel model | Present182 accuracy | NLL | Selected pass |
| --- | ---: | ---: | ---: |
| Our aligned whole-tree PyG GNN | **95.4891%** | **0.133497** | 30 |
| Matched zero-concordance control | 94.9252% | 0.147232 | 25 |
| Native ARGMix, current author release | 94.4461% | 0.219695 | Native smoothed VAL-loss selection |
| Native AncestralPaths, derived C4 endpoint | 93.7830% | ∞* | Fixed author epoch 5 |

The selected difference is **0.5639 percentage points** on development validation. Both completed the same total training budget, but their selected checkpoints have different update counts. These results use the AncestralPaths-derived demographic panel with fresh 2 Mb simulation streams. It is not ARGMix's published 1,611-diploid panel. They are not original-paper numerical reproduction, and cannot be compared with the older pilot table. The larger early learning gap did not persist.

**Same-data comparison with native AncestralPaths completed:** our selected model gains **1.7062 percentage points** on the same eight validation parents, 182 present-day queries and exact physical-span C4 scoring support. AncestralPaths retains the author Dense512 model and five-epoch fit on 5,190,874 sampled rows. Its six-path probabilities are collapsed to four source classes before argmax, with native six-path SNP-focal scores reported separately. Our checkpoint uses validation selection, while AP uses its fixed final fifth epoch. This is a development comparison using the full author panel and fresh 2 Mb simulations, not the original paper's numerical benchmark.

*AP's strict raw FP32 NLL is infinite because one parent assigns zero probability to a true class over 800 query-base pairs. Scores use no probability floor or recalibration.*

**Native ARGMix completed 11 author epochs and all eight validation inferences, with actual launcher exit 0.** Its native early-stopping rule ended training. Its common-support CPU score is **94.4461%** with NLL **0.219695**, giving our model a **1.0430 percentage-point accuracy gain** on the same eight development simulations and Present182 physical support. The score uses sealed native overlap-averaged probabilities without recalibration or a probability floor. The H038 full-panel pipelines use the same native Relate post-size trees. Our GNN additionally uses observed SNP features. Earlier README text incorrectly called this full-panel genealogy tsinfer/tsdate because reused helper modules carry that name. The comparison still does not isolate the GNN architecture from its SNP features. It is a current-author-release comparison, with disclosed input metadata adaptations, rather than original-paper numerical reproduction. ARGformer and independent outer evaluation remain pending.

**New completed layer experiment (15 October):** directed clade exchange improves Present182 development accuracy to **95.712 ± 0.968%**, versus **95.564 ± 1.041%** for equal-budget plain continuation and **95.496 ± 1.064%** for generic smoothing. All three use the same H038 warm start, one optimizer seed101, ten additional passes and320 full-parent updates. Each selects pass10 under the frozen pass5/10 accuracy rule. Directed gains0.1484 pp over plain continuation, positive in seven of eight parents. SD is across simulation parents, not training seeds. Native ARGMix remains94.4461% on this development endpoint. This result does not establish original-paper superiority or architectural novelty. [Completed comparison and actual wait evidence](research/reviews/H039_completed_comparison_root_20261015_v1/README.md).

**Latest completed architecture comparison (GPUs 4 and 5):** both H040 arms completed **30 additional FP32 passes and 960 updates**, starting from the same H038 checkpoint. The frozen evaluation completed with actual exit 0.

| Matched 30-pass continuation | Present182 accuracy, mean ± SD (%) | NLL | Selected additional pass |
| --- | ---: | ---: | ---: |
| Exact local-clade flags, H042 | **96.5778 ± 0.5954** | 0.100656 | 30 |
| Tensor clade coupling | 96.2674 ± 0.7493 | 0.109332 | 30 |
| Matched event MLP | 96.2007 ± 0.7048 | 0.110014 | 30 |

The tensor-minus-event-MLP difference is **0.0667 pp**, positive in **6/8** VAL simulations. SD is across simulations, with one optimizer seed. This small effect does not establish significance or novelty. Native current-release ARGMix remains 94.4461% on this development endpoint, but the 1.8212 pp pipeline gap cannot be attributed solely to the tensor layer. H040 received 960 additional updates beyond H038 and uses observed SNP features. Population advantages are mixed. [Population metrics, paired effects and evidence](research/reviews/H040_completed_comparison_root_20261015_v1/README.md).

**The simpler local-clade flag control is currently strongest:** H042 reaches 96.5778 ± 0.5954%, exceeding tensor coupling by 0.3104 pp at the same added budget and parameter count. This argues against assigning novelty or usefulness to the tensor coupling alone. H043 now tests neighboring-tree clade states while preserving the local flags, using a matched partner-blind control. [H042 population metrics and paired comparisons](research/reviews/H042_completed_comparison_root_20261015_v1/README.md).

Only physical GPUs 4 and 5 are eligible. Previous assignments to GPUs 2 and 3 are revoked.

**Next accuracy hypothesis, H044:** preserve the full H042PyG/GRU backbone and test a small SNP-resolved readout against an equal-capacity control using the same marker evidence pooled within each native tree. This targets within-tree ancestry changes that the current constant-per-tree output cannot resolve. The registered paired condition uses31,428parameters per arm and30additional FP32passes. Source integration and independent review are in progress. No H044training or accuracy result exists yet.

**Boundary error analysis:** on the fixed H038 Present182 output, boundary precision is10.71 ± 1.21% and recall17.99 ± 2.01% at1kb. A class-agnostic geometric oracle can cover72.70 ± 2.80% of true changes. About27.3% have no allowed hold edge nearby, while two-thirds of the model's missed changes do have an edge. This motivates better continuity/transition learning and finer marker readout, rather than validation-chosen smoothing. [Unchanged-output biological diagnostic](research/reviews/H038_boundary_error_diagnostic_source_v1/biological_conclusions.md). H039/H040 boundary gains are not established.

**ARGformer benchmark preparation:** the first full **30 Mb American-admixture TRAIN simulation completed**, with 1,600 haplotypes and 226,185 sites. It uses the paper’s mutation and recombination rates, with unresolved sample-unit and evaluation details explicitly disclosed. The short observed-genotype inference qualifier completed. The first full inference attempt stopped because its supervisor checked file stability while the worker was writing. The supervision-only recovery completed with actual direct wait 0, unchanged biological settings and all failed artifacts preserved. The full 30 Mb inferred genealogy is now available. Eight independent2Mb TRAIN parents completed both simulation and inference. Parents008/009 are being prepared. Complete native path preparation stopped at its preset10-million unique-path cap before corpus serialization. The complete label-blind counting pass has now finished:25,574,587unique paths,316,672,003path tokens and maximum raw path length29. These exact counts size a full-corpus retry without subsampling. Native model fits and C3 scores remain unfinished. This is a reconstructed benchmark input, not an ARGformer result or exact Table 1 reproduction. The fixed real-data region was acquired: 13,312 candidate variants, with 434 passing all prespecified filters, below the 1,000-site gate. This region fails the registered qualification. The prospectively registered10Mb extension then stopped at its decode runtime cap, with partial files retained and no successful acquisition claim. Model application and ancestry-masked genotype PCA remain unrun.

**Novelty audit:** the completed local-clade control outperforms H040 tensor coupling. A fresh independent H043 audit finds weak architectural novelty: direct cross-graph messages already exist in Graph Matching Networks, and H043 is an edge-conditioned MPNN over a deterministic clade correspondence graph. The nearest changed-clade leaf-sector construction is a narrower possible contribution whose priority remains unestablished. H043's partner-blind control tests usefulness of foreign states, not originality or optimality of the chosen partners. Its paired FP32 training completed30passes and960whole-parent updates per arm on GPU5. The frozen paired evaluation is now admitted and launched there, with no performance claim before completion. [Fresh H043 audit and primary sources](research/reviews/H043_priority_fresh_v2/audit.md) · [H040 audit](research/reviews/H040_priority_audit_fresh_v1/README.md).

**Published benchmark requirements:** [ARGMix original setup and Tables1/2](docs/argmix_published_benchmark_audit_v1.md), [ARGformer American-admixture Table1](docs/argformer_published_benchmark_audit_v1.md), and [real-data/PCA analysis contracts](docs/real_data_validation_plan_v1.md) now have primary-source audits. ARGMix uses a different panel, reference tokens and error conditions. ARGformer reports EUR/EAS/AFR precision and recall, a distinct task from our European four-source development panel. Original-simulation reconstruction, faithful native training, paired population metrics and real-data model application remain to be completed. Verified public real-data assets are available, but acquisition/audit alone is not validation.

**Which comparisons are complete?**

| Benchmark | Completed evidence | What is still needed |
| --- | --- | --- |
| European common-input development panel | Native AP, current native ARGMix and our GNN on eight shared VAL simulations, population SDs and confusion/boundary diagnostics | Independent test confirmation and training-seed variation |
| ARGMix published Tables 1 and 2 | Primary-source protocol and exact published numbers audited | Published-panel IID and demographic-misspecification experiments. Paper-era code and several recipe details are unavailable |
| ARGformer American Table 1 | Full 30 Mb simulated and inferred TRAIN input completed. Eight independent2Mb TRAIN parents prepared. Complete native path counts now verified, with a full-corpus retry in preparation. Native ARGformer and FLARE adapters in preparation | Remaining independent data, native fits and common EUR/EAS/AFR precision, recall and F1 |
| Empirical ancestry and PCA | Fixed region acquired. Only 434 of 13,312 variants pass all filters, below the registered 1,000-site gate | Complete the prospectively registered10Mb extension after resolving its decode throughput, then genealogy inference, model application and ancestry-masked genotype PCA |

Published percentages and newly measured development results are separate experiments. No current result supports a claim of superiority over every method or population.

**Population-level results on the same development simulations:** mean ± sample SD across eight VAL simulations, with one optimizer seed per model. H042 uses the H038 warm start plus30 additional passes/960updates. Native competitor training and selection remain as disclosed above.

| Cohort | Native AP accuracy (%) | Current native ARGMix (%) | H038 aligned (%) | H042 local-clade flags (%) |
|---|---:|---:|---:|---:|
| Present (182) | 93.78±0.84 | 94.45±1.15 | 95.49±0.83 | **96.58±0.60** |
| Bronze Age (162) | 94.76±1.18 | 97.71±0.75 | 98.17±0.80 | **99.07±0.32** |
| BAA (14) | 96.94±1.32 | 98.44±1.02 | 98.90±0.89 | **99.02±1.07** |
| Neo (302) | 97.57±0.27 | 99.30±0.26 | 99.41±0.13 | **99.53±0.18** |
| Yam (26) | 95.43±1.41 | 99.22±0.90 | 98.96±1.31 | **99.86±0.19** |
| Ancient (504) | 96.54±0.32 | 98.76±0.31 | 98.98±0.32 | **99.39±0.16** |
| All (686) | 95.81±0.45 | 97.61±0.50 | 98.05±0.43 | **98.64±0.26** |

H042 has higher mean span accuracy than native AP and current native ARGMix in all seven reported cohorts. This does not imply a win on every simulation or metric. The Present182 pipeline gain over current native ARGMix is2.1317pp. It includes differences in SNP inputs, training budget and checkpoint selection. Bronze_Age is an approximate correspondence to the paper's Post-Neolithic category, not a verified identical panel. These rows cannot be compared directly with the paper's Tables1/2.

Present182 boundary F1 at1kb tolerance is **18.05 ± 1.84%** for H042, **13.42 ± 1.47%** for H038, **11.74 ± 1.73%** for current native ARGMix and **7.03 ± 0.78%** for native AncestralPaths. H042 boundary precision is15.21 ± 2.01% and recall22.30 ± 1.86%. Boundary inference remains weak despite high span accuracy. [H042 completed cohort/boundary summaries and confusion matrices](research/reviews/H042_completed_cohort_diagnostics_root_20261015_v1/README.md) · [Native class metrics and comparison](research/reviews/native_common_cohort_diagnostics_20261015_v1/diagnostic_report.md).

![Development population accuracy and tract boundaries](research/reviews/H042_completed_cohort_diagnostics_root_20261015_v1/accuracy_and_boundary_mean_SD.png)

[Native ARGMix same-support CPU result and transcript](research/reviews/expanded_native_argmix_H038_score_root_v1/readonly_CPU_result_v1.json)

[Native AncestralPaths comparison and receipts](experiments/results/full_panel_H038_AP_comparison_root/summary.json)

[Final PyG results and efficiency](experiments/results/mutation_routed_full_panel_pyg_v1_summary_root/summary.json) · [Expanded protocol](research/reviews/expanded_full_panel_v1_root/registration.json)

**Success criteria:** top-1 accuracy must be demonstrated on faithful shared-data comparisons and the original-paper benchmarks where assets allow. Architectural novelty is assessed separately. GST's graph-set tasks are evaluated separately from biological ancestry. Any additional benchmark needs a documented scientific purpose and a prospectively disclosed protocol.

**Efficiency:** H038 training now aggregates exact scored spans by native tree interval instead of repeating training logits across the 100 bp grid. It retains every whole tree, query and genomic GRU state. Model arithmetic is FP32, and validation keeps the original grid. A small loss/gradient check passed. The first real whole-parent updates took about 1.8 seconds, with a first-update Torch reserved-memory peak around 24 GB. Subsequent 32-parent passes take about 44 seconds. These are training measurements, with preprocessing and evaluation reported separately. Preparation, training, validation and inference costs will accompany accuracy.

[Research success criteria](docs/research_success_criteria_v1.md)

## Current experiments

| Completed pilot pipeline | Present-day accuracy | NLL |
| --- | ---: | ---: |
| Our standalone PyG model, 400 total updates | 83.90% | 0.4915 |
| ARGMix, current author release | 85.55% | 0.4353 |
| ARGMix plus ordinary smoothing | 85.94% | 0.3891 |

These scores use two reused validation simulations and 64 present-day query haplotypes. Our standalone model uses tsinfer/tsdate trees and raw SNPs, while ARGMix uses Relate trees. Training budgets also differ. They are descriptive pilot results, with no top-1 or novel-mechanism claim.

Updated 2026-10-12 01:33 UTC. Latest full-panel progress: [all 40 development simulations completed](research/reviews/expanded_full_panel_v1_root/progress_all40_simulations_completed_v1.json).

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
| Smoothing plus kernel-only reference GNN, H030 | 85.9104% | 0.382352 |
| Smoothing plus static log-cavity whole-tree GNN, H030 | 85.9355% | 0.374655 |
| Smoothing plus log-cavity carrier-GRU GNN, H030 | 85.9102% | 0.374975 |
| Smoothing plus log-cavity carrier-matrix GNN, H030 | 85.9421% | 0.374083 |
| Smoothing plus log-cavity shuffled-history control, H030 | 85.9636% | 0.374457 |
| Smoothing plus independence-Gram GNN, H032 | 85.9385% | 0.374581 |
| Smoothing plus marginal outer-product GNN, H032 | 85.9284% | 0.374689 |
| Smoothing plus tree-Gram GNN, H032 | 85.9378% | 0.374591 |
| Smoothing plus interleaved tree/TV GNN, H033 | 85.9697% | 0.386110 |
| Smoothing plus blocked tree/TV control, H033 | 85.9695% | 0.386111 |
| Smoothing plus signed mixed-path GNN, H035 | 85.9335% | 0.374666 |
| Smoothing plus path-sum control, H035 | 85.9438% | 0.374558 |

**Observed reference context improves likelihood, but genuine ancestral continuity and census conditioning have no demonstrated accuracy advantage.** The sham row is a control, not the proposed method. Its higher accuracy cannot support a genuine ancestry mechanism claim. We have not established a novel GNN contribution or final-test superiority.

[H026 selected results and parent/class/boundary diagnostics](experiments/results/reference_census_innovation_v1_summary_root/summary.json) and [H027 selected results and diagnostics](experiments/results/reference_trace_residual_v1_summary_root/summary.json). H026 native, age-free and census arms select epoch10. Its sham selects epoch6. H027 arms select epoch10. H027's positive normalization-mass check passed, but very small reference masses need care in interpreting local normalized fields.

- **H028 completed:** all three fixed genotype/graph/fusion heads finished 10 FP32 epochs and 160 updates, seed 101. The genotype-only and fixed-fusion routes do not improve smoothing accuracy. Graph-only exactly matches H026 native-reference selected metrics and is an explicitly disclosed matched-control repeat. Each head allocates 32 parameters, 28 with input signal because the older ANA channel is absent. This does not establish that raw SNPs lack signal. [Selected comparison and diagnostics](experiments/results/observed_variant_residual_v1_summary_root/summary.json).
- **H029 completed:** all four whole-tree models finished 10 FP32 epochs and 160 updates per arm, seed 101. Static/matrix accuracy exceeds smoothing by about 0.016 percentage points. Matrix does not beat static or shuffled-history controls, and H026 native reference context has lower NLL. The GRU selects epoch 7, the other arms epoch 10 by minimum VAL NLL. GRU and matrix use 16 state scalars per carrier/direction, with unequal active parameter counts. [Selected comparison, retained-file verification and diagnostics](experiments/results/edge_innovation_gnn_v1_summary_root/summary.json). The first real TRAIN tile passed finite FP32 loss/gradient checks for 359,640 augmented vertices and 550 marginal holds, with a cumulative allocator peak of 2.69 GB.
- **H030 recovery completed:** all five arms reached 160 logical FP32 updates and selected epoch 10 by minimum development VAL NLL. Static scores 85.9355% accuracy and NLL 0.374655. Carrier matrix scores 85.9421% and 0.374083, but its accuracy does not beat shuffled history at 85.9636%. Recovery restored 272 updates, executed 528 new updates and replayed 48 unsaved original updates: 800 final logical updates and 848 historically executed updates. [Verified results and recovery lineage](experiments/results/reference_logcavity_resume_full_storage_v1_summary_root/summary.json).
- **H031 completed:** direct fixed SNP evidence scores 42.34% on development VAL. On baseline-error spans it scores 26.84%/29.16%, below the unfitted observed-donor-prior ANA prediction at 28.05%/31.91%. This evidence does not justify a strong complementary-signal claim. Its oracle union is only a diagnostic bound. [Results and source-prior context](experiments/results/observed_variant_signal_diagnostic_v1_summary_root/summary.json).
- **H032 three-arm fit completed:** all donor-support Gram conditions finished 160 FP32 updates, seed 101, and selected epoch 10 by minimum VAL NLL. Genuine tree Gram scores 85.9378% accuracy and NLL 0.37459, versus independence 85.9385% and 0.37458, marginal outer product 85.9284% and 0.37469, and ordinary smoothing 85.9397% and 0.38914. This establishes a likelihood improvement over smoothing, with no donor-dependence or accuracy advantage. H030 static first-moment context already achieves nearly the same likelihood improvement without Gram. [Matched static comparison](experiments/results/reference_logcavity_resume_full_storage_v1_summary_root/H032_static_comparison.json). [Verified selected results and class/boundary metrics](experiments/results/reference_dependence_full_storage_v1_summary_root/summary.json).
- **H033 completed:** both arms finished 160 FP32 updates. Minimum-NLL selection chooses epoch10 for both. Interleaved accuracy85.9697% and blocked85.9695% are effectively tied. The difference does not establish the interleaving mechanism. [Verified selected results](experiments/results/interleaved_tree_tv_gnn_v1_summary_root/summary.json).
- **H035 completed:** both arms finished160 FP32 updates. Selected signed difference scores85.9335%/NLL0.374666 at epoch10. Path sum scores85.9438%/0.374558 at its selected epoch7. Signed difference does not beat its matched control or smoothing accuracy. Observed roles are correspondences, not verified ancestral identities. [Verified selected results](experiments/results/observed_role_mixed_path_gnn_v1_summary_root/summary.json).
- **H034 full-panel data:** two fresh 2 Mb parents completed simulation and Relate inference with the full952-haplotype author panel,686 queries and266 pure references. Two more parents completed, giving four prepared parents in total. The second outer parent remains registered and ungenerated. The frozen downstream split uses TRAIN000/001 for training, nominalTRAIN002 for internal validation and two independent outer discovery parents. These use the author demographic/sample panel, shorter segments and new streams. No method scores or original-paper numerical reproduction are claimed. [Protocol](docs/ancestralpaths_full_panel_discovery_v1.md), [native baseline plan](research/reviews/full_panel_native_baseline_plan_v2/plan.md).
- **H036 direct classifier completed:** both aligned and shuffled arms finished160 full-parent FP32 updates. Minimum-NLL selection gives aligned74.8564% accuracy/NLL0.600348 at epoch80 and shuffled73.7807%/0.620340 at epoch76. This direct model uses original SNPs plus complete tsinfer/tsdate trees, with no ARGMix teacher. It remains below historical ARGMix and smoothing accuracy. The original placement-mechanism gate remains inconclusive. [Verified paired results](experiments/results/mutation_routed_whole_tree_gnn_total160_summary_root/summary.json).
- **PyG bridge completed:** the standalone model reached 83.8969% accuracy and NLL 0.491455 at selected epoch 170, versus 83.7871% and 0.465909 for shuffled support at epoch 130. Both completed 400 total updates. These two-parent pilots remain below ARGMix at 85.554%. The median training epoch with two full-parent updates took about 1.47 seconds, versus about 4.3 seconds in the earlier pipeline. This measures the overall training phase, rather than PyG in isolation. Training occupied only 35–36% of each 600-second run. The broader stage reduces checkpoint frequency and caches repeated authorization checks. [Verified results](experiments/results/mutation_routed_pyg_total400_summary_root/summary.json).
- **H038 full-panel PyG fits completed:** aligned95.4891%/NLL0.133497 versus zero-concordance94.9252%/0.147232 on eight independent development VAL simulations. Both completed960 updates. Selected checkpoints are pass30 and25 respectively. Native baselines retain author algorithms on the shared observations and scoring support. Original-paper benchmarks and independent outer evaluation remain outstanding. [Final results](experiments/results/mutation_routed_full_panel_pyg_v1_summary_root/summary.json).
- **Native ARGformer:** genuine author training and retrieval completed on all four adapted EuropeanC4/tsinfer parents. Present-day top1 accuracy is75.68%, fixed top5 is75.14%, and unmodified raw-vote NLL is infinite. The main comparison below explains the differing genealogy and training objective.
- **Original GST:** all four native USPTO models completed the bounded TRAIN/VALID pilot below.
- **Previous partial runs:** interrupted mechanism-seed and Census fits remain preserved. They are not final30epoch results. Large FP64 equivalence sweeps remain deferred.

Frozen ARGMix predictions derive from Relate trees. The new graph context comes from independently inferred tsinfer/tsdate trees. TRAIN emitter predictions are in-sample and the frozen smoother was development-selected. No TEST parent has been evaluated.

Novelty is assessed throughout development in the [claim-by-claim novelty log](docs/novelty_assessment_v1.md). The latest independent [design critique](research/reviews/edge_innovation_novelty_fresh_v1/critique.md) identifies TGN and Gated DeltaNet precedents. Whole-tree cavity construction plus carrier-conditioned memory remains a hypothesis about useful biological coupling, with priority and predictive benefit unresolved. Existing kernels, recurrence and message passing are discovery tools.

At most two physical GPUs from indices2–5 are assigned to the project. GPU workers are titled 'AI Auto Research Pipeline Test'. Current storage authorization is1TB. See [development priorities](docs/development_priorities_20261011.md).

## Main advantages and current evidence

**Latest full-panel evidence:** the H039 directed extension scores95.7119% Present182 development accuracy. Its equal-budget plain continuation scores95.5635% and generic smoothing scores95.4964%. The earlier H038 aligned model scores95.4891% with29,764 parameters, versus94.9252% for zero concordance,94.4461% for current native ARGMix and93.7830% for native AncestralPaths on the same development C4 endpoint. It learns directly from observed SNPs and complete inferred trees, without ARGMix predictions. A32-parent training pass takes 43.6 seconds. The 0.5639 pp validation advantage supports further development, with architectural novelty, original-benchmark confirmation and superiority over full-panel ARGformer and untouched outer simulations still unproven.

**Earlier reused-panel evidence:** the strongest genuine residual pipeline scored85.9697%, versus85.9397% for ordinary smoothing and85.554% for frozen ARGMix. Those historical pilot values are separate from the full-panel experiment above.

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
| Smoothing plus interleaved tree/TV GNN, H033 | 85.9697% | 0.386110 | 0.19855 | 0.02377 | 101, emitter 50 |
| Smoothing plus blocked tree/TV control, H033 | 85.9695% | 0.386111 | 0.19855 | 0.02376 | 101, emitter 50 |
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

The user increased the project storage allowance to **1 TB on2026-10-11**, with a950GB scheduling ceiling. The latest completed whole-project measurement was about289.7GB. Historical conservative reservations project949.53GB, an accounting projection rather than physical usage. A narrowly audited accounting successor has now retired the original failed H039 allowance, retaining all34.103MB of its artifacts and releasing9.966GB of unused allowance. It performed one targeted scope measurement, with no deletion or whole-repository scan. Current conservative charges are separate from physical usage. See the [resource authorization](research/resource_authorizations/project_storage_1TB_20261011.json).

Native ARGformer v7 completed genuine fitting and retrieval in the original captured author environment. Both VAL predictions were sealed before unchanged external scoring. The same-tsinfer Census observation and CPU numerical checks completed, but this separate interrupted model has no final superiority result. [Native capture evidence](experiments/results/native_argformer_tsinfer_fit_v7_root/compact_runtime_inventory.json) and [completed ancestry comparison](experiments/results/native_argformer_tsinfer_fit_v7_summary_v2/summary.json).

Run `.runtime/env/bin/python scripts/update_research_ledger.py` after new results. Record a hypothesis or explicit repeat reason before each substantive experiment. Preserve complete outcomes, provenance and failures, and push concise conclusions regularly.

The next scientific checks are accuracy and mechanism discovery with matched controls, followed by independent simulation-parent confirmation, original-paper task comparisons, demographic shifts and a justified real-data endpoint. Published ancestry paintings are comparator outputs, rather than ground truth.
