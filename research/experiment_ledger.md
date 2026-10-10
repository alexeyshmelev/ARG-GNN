# Experiment and decision ledger

Concise scientific decisions with complete results indexed separately. Pilot selection evidence is not independent confirmation.

Before a substantive run, register its question and check this ledger. A repeat requires an explicit reason linked to the earlier ID. Complete result values and source hashes are in [experiment_results.csv](experiment_results.csv) and [experiment_ledger.json](experiment_ledger.json). Method/configuration, dataset, source, seed and checkpoint bindings remain in the linked original receipts. Failures and partial runs are retained.

Status is local to a question. A pilot observation is not a validated paper claim.

## H001: Initial ordered lineage networks improve the easy oracle panel

**pilot_answered_negative.** Initial graph/set/GRU/Transformer/supra/lineage variants, two seeds, 16 TRAIN/4 VAL/8 historical pilot TEST parents.

**Conclusion:** The nearest-TMRCA oracle comparator reached 98.10% grid accuracy. Initial neural models were much weaker. This panel does not support an ancestry improvement claim.

**Next/repeat rule:** Do not rerun unchanged. Reopen only for a corrected encoder or a demonstrably harder inferred-tree task.

Evidence: [docs/research_log.md](../docs/research_log.md), [experiments/results/oracle_pilot_v1_seed101/results.json](../experiments/results/oracle_pilot_v1_seed101/results.json), [experiments/results/oracle_pilot_v1_seed202/results.json](../experiments/results/oracle_pilot_v1_seed202/results.json).

## H002: Reference reachability explains weak initial graph encoders

**pilot_answered.** Reference-feature and genealogy-aware encoders on ancient oracle and paired inferred Q8 panels.

**Conclusion:** Reference-aware controls improved the engineering comparison. Ordered-lineage novelty and superiority remain unsupported.

**Next/repeat rule:** Revisit only for a concrete input-reachability defect or new biological benchmark.

Evidence: [docs/reference_encoder.md](../docs/reference_encoder.md), [docs/local_encoder_diagnostics.md](../docs/local_encoder_diagnostics.md), [experiments/results/ancient_common_input_pilot_v2_seed101_recovery/results.json](../experiments/results/ancient_common_input_pilot_v2_seed101_recovery/results.json), [experiments/results/inferred_common_input_pilot_v3_seed101/results.json](../experiments/results/inferred_common_input_pilot_v3_seed101/results.json).

## H003: Boundary memory beats all-record temporal controls

**pilot_answered_negative.** Boundary full/diagonal/gain controls and GNN+GRU/Transformer, Q64 inferred pilot.

**Conclusion:** Boundary full NLL 1.038652 versus GRU 1.038243. It did not consistently beat temporal controls on reused validation parents.

**Next/repeat rule:** Do not repeat the same seed/config/cohort. A new model mechanism or independent biological confirmation needs a new entry.

Evidence: [docs/boundary_study_controls.md](../docs/boundary_study_controls.md), [experiments/results/boundary_inferred_cohort64_pilot_v1_seed101_priority/results.json](../experiments/results/boundary_inferred_cohort64_pilot_v1_seed101_priority/results.json).

## H004: Fixed-anchor cross-query geometry helps compared with diagonal decay

**pilot_observation.** Nine plain anchor variants on 16 TRAIN/4 reused VAL Q64 inferred parents, seed101.

**Conclusion:** Full depth2 NLL 1.025647 versus scalar 1.038708, diagonal 1.039765 and gain-matched diagonal 1.040722. These are exploratory selected scores.

**Next/repeat rule:** Independent parents/seeds and full-information competitors can confirm or reject this observation. No unchanged pilot rerun.

Evidence: [docs/anchor_pilot.md](../docs/anchor_pilot.md), [research/reviews/anchor_plain_pilot_complete_v1.json](../research/reviews/anchor_plain_pilot_complete_v1.json).

## H005: Query-source feedback explains the fixed-anchor result

**pilot_answered_unestablished.** Capacity-matched noncontextual/static controls, same Q64 parents/seed101.

**Conclusion:** Full depth2 NLL 1.025647 and noncontextual depth2 1.026034 are nearly equal. Static depth2 NLL 1.117582 supports a sequence-information benefit in this pilot.

**Next/repeat rule:** Do not assert a feedback contribution. Reopen only with a specific changed feedback mechanism or independent prespecified confirmation.

Evidence: [experiments/results/anchor_inferred_cohort64_capacity_controls_v1_seed101/results.json](../experiments/results/anchor_inferred_cohort64_capacity_controls_v1_seed101/results.json), [research/reviews/anchor_capacity_pilot_complete_v1.json](../research/reviews/anchor_capacity_pilot_complete_v1.json).

## H006: Age-conditioned graph geometry improves ancestry prediction

**pilot_observation_small.** Full age versus matched scalar-age, depth1/2, same reused Q64 validation parents/seed101.

**Conclusion:** Age full depth2 NLL 1.023317 versus scalar-age full depth2 1.025001. The difference is small and unconfirmed. Geometry audits alone are not prediction evidence.

**Next/repeat rule:** Independent biological evaluation can test this. Remaining scalar-age ablations are distinct unfinished cells, not reruns.

Evidence: [research/reviews/anchor_age_temporal_pilot_complete_v1.json](../research/reviews/anchor_age_temporal_pilot_complete_v1.json), [experiments/results/anchor_inferred_cohort64_age_pilot_v1_seed101_priority/results.json](../experiments/results/anchor_inferred_cohort64_age_pilot_v1_seed101_priority/results.json), [experiments/results/anchor_age_controls_scheduling_interruption_v1.json](../experiments/results/anchor_age_controls_scheduling_interruption_v1.json).

## H007: Anchor kernel improves over matched temporal layers

**pilot_observation.** Same source encoder/anchors, GRU/Transformer and reference smoothing, Q64 seed101.

**Conclusion:** Age full depth2 accuracy 53.7170% versus matched anchor GRU 53.0029% and Transformer 52.1454%. Scopes differ because the candidate propagates intervening geometry holds.

**Next/repeat rule:** Confirm with all-record competitors, independent parents and a frozen biological task. Shared data alone does not equal shared information access.

Evidence: [docs/anchor_temporal_baselines.md](../docs/anchor_temporal_baselines.md), [research/reviews/anchor_age_temporal_pilot_complete_v1.json](../research/reviews/anchor_age_temporal_pilot_complete_v1.json), [research/reviews/canonical_development_audit_v1/review.md](../research/reviews/canonical_development_audit_v1/review.md).

## H008: Paper-equation ARGMix adaptation matches the candidate pilot

**pilot_completed_native_author_benchmark_pending.** Our independent full-tree/nearest-reference equation adaptations and whole-tree Transformer, same Q64 cohort.

**Conclusion:** Full-tree ARGMix equation adaptation NLL1.142297/accuracy48.5321%, nearest-reference adaptation1.141211/48.4802%, whole-tree Transformer1.118489/50.3815%. These are our shared-data implementations and selected pilot scores, not native published benchmark reproductions.

**Next/repeat rule:** Do not rerun these unchanged completed cells. Native author execution and independent frozen biological comparisons answer distinct questions.

Evidence: [docs/argmix_development.md](../docs/argmix_development.md), [experiments/results/inferred_cohort64_reference_controls_v1_seed101/argmix_full_tree.json](../experiments/results/inferred_cohort64_reference_controls_v1_seed101/argmix_full_tree.json), [experiments/results/inferred_cohort64_reference_controls_v1_seed101/argmix_nearest_reference.json](../experiments/results/inferred_cohort64_reference_controls_v1_seed101/argmix_nearest_reference.json), [experiments/results/inferred_cohort64_reference_controls_v1_seed101/results.json](../experiments/results/inferred_cohort64_reference_controls_v1_seed101/results.json).

## H009: GST and GraphSSM explain the Q64 pilot gain

**both_pilot_controls_completed_independent_confirmation_pending.** GST cross-attention/span and GraphSSM ZOH on the same Q64 inferred parents/seed101.

**Conclusion:** GST selected NLL1.123058/accuracy49.9237%. GraphSSM selected NLL1.055522/accuracy52.8595%. Both use the same four reused Q64 VAL parents and seed101 as the candidate. Candidate age anchor D2 NLL1.023317/accuracy53.7170% is a selected exploratory comparison, not independent superiority. Q8 GST is a different task.

**Next/repeat rule:** Do not repeat these completed unchanged cells. Reopen for a documented implementation correction, matched new biological development panel or independent frozen confirmation.

Evidence: [docs/gst_baseline.md](../docs/gst_baseline.md), [docs/graphssm_adaptation.md](../docs/graphssm_adaptation.md), [configs/inferred_cohort64_gst_graphssm_controls_v1.json](../configs/inferred_cohort64_gst_graphssm_controls_v1.json), [experiments/results/inferred_cohort64_gst_graphssm_controls_v1_seed101/gst_ca_span.json](../experiments/results/inferred_cohort64_gst_graphssm_controls_v1_seed101/gst_ca_span.json), [experiments/results/inferred_cohort64_gst_graphssm_controls_v1_seed101/graphssm_zoh.json](../experiments/results/inferred_cohort64_gst_graphssm_controls_v1_seed101/graphssm_zoh.json), [experiments/results/inferred_cohort64_gst_graphssm_controls_v1_seed101/results.json](../experiments/results/inferred_cohort64_gst_graphssm_controls_v1_seed101/results.json).

## H010: The kernel shows a broad cross-domain predictive advantage

**pilot_answered_negative.** Chickenpox fixed county topology, chronological validation, seed101. Ordinary controls and direct anchor/boundary adaptations.

**Conclusion:** Ordinary GRU MAE 16.28198 is lower than anchor full depth1 16.83561/depth2 17.24698 and boundary full 18.01937. Current evidence does not support broad superiority. Remaining controls are still running.

**Next/repeat rule:** Complete pending cells. Do not repeat this unchanged pilot or claim a changing-topology result from fixed county data.

Evidence: [docs/chickenpox_protocol.md](../docs/chickenpox_protocol.md), [docs/chickenpox_anchor_pilot.md](../docs/chickenpox_anchor_pilot.md), [experiments/results/chickenpox_common_input_pilot_v1_seed101/results.json](../experiments/results/chickenpox_common_input_pilot_v1_seed101/results.json), [experiments/results/chickenpox_anchor_pilot_v1_seed101_cpu/anchor_full_d1.json](../experiments/results/chickenpox_anchor_pilot_v1_seed101_cpu/anchor_full_d1.json), [experiments/results/chickenpox_boundary_pilot_v1_seed101_cpu/boundary_full.json](../experiments/results/chickenpox_boundary_pilot_v1_seed101_cpu/boundary_full.json).

## H011: Ordered propagation responds to order and ignores identical subdivision

**structural_check_completed.** Untrained synthetic operator-order/subdivision probe, separate from ancestry training.

**Conclusion:** The synthetic probe changes under reversal and retains subdivision differences at or below 4.1e-14. This establishes the measured structural behavior only.

**Next/repeat rule:** Do not redo unchanged. Rerun if the operator implementation changes or to validate a new operator.

Evidence: [docs/anchor_kernel.md](../docs/anchor_kernel.md), [experiments/results/operator_order_probe_v1](../experiments/results/operator_order_probe_v1).

## H012: Native ARGMix reproduces published biological performance

**current_author_source_selected_coherent_native_environment_resolved.** Pinned unchanged author source c2481248eb0d, native TmrcaAttn component/TorchScript qualification.

**Conclusion:** Nine CPU native component tests passed. The user has no older archive, so c248 current author source is the available comparator, with differences disclosed. Author Python Torch2.12 conflicts with Cargo tch0.23 requiring Torch2.10. A coherent Torch2.10/cu129 environment candidate and84-package toolchain resolved and froze, installation/build pending. No biological fit ran.

**Next/repeat rule:** Build and execute the available native release on matched biological data. Do not call it April reproduction or repeat the same archive search without a new lead.

Evidence: [docs/official_argmix_execution_v1.md](../docs/official_argmix_execution_v1.md), [experiments/results/official_argmix_cpu_validation_20261010_v1.json](../experiments/results/official_argmix_cpu_validation_20261010_v1.json), [docs/argmix_paper_revision_search_v1.md](../docs/argmix_paper_revision_search_v1.md), [research/protocols/argmix_paper_revision_search_v1.json](../research/protocols/argmix_paper_revision_search_v1.json), [docs/argmix_native_environment_candidate_v1.md](../docs/argmix_native_environment_candidate_v1.md), [experiments/results/argmix_native_dependency_resolution_v1/completion.json](../experiments/results/argmix_native_dependency_resolution_v1/completion.json).

## H013: Native ARGformer is a genuine ancestry competitor

**actual_full_native_GPU_MLM_and_contrastive_qualified_biology_pending.** Pinned f7545be7bca0 node-vocabulary model, observed-only path/namespace/retrieval bridge.

**Conclusion:** After disclosed packaging repairs, both unchanged full12-layer author models passed actual CUDA native RoPE/FlashAttention forward/backward, verified title/UUID in NVTOP, with all12 layers having gradients. Handmade inputs/random weights and no optimizer update establish runtime availability, not biological training or accuracy. Biological pretraining/retrieval remain pending.

**Next/repeat rule:** Do not rerun unchanged successful full-model qualification. Repeat only after relevant source/runtime/adapter change. Proceed to source-native biological fitting with query ancestry absent from learning and same-data scoring.

Evidence: [docs/official_argformer_execution_v1.md](../docs/official_argformer_execution_v1.md), [experiments/results/official_argformer_cpu_validation_20261010_v1.json](../experiments/results/official_argformer_cpu_validation_20261010_v1.json), [docs/argformer_native_environment_candidate_v1.md](../docs/argformer_native_environment_candidate_v1.md), [experiments/results/official_argformer_environment_candidate_v1.json](../experiments/results/official_argformer_environment_candidate_v1.json), [configs/official_argformer_native_environment_v1/lock-package-manifest.json](../configs/official_argformer_native_environment_v1/lock-package-manifest.json), [docs/official_argformer_runtime_qualification_v1.md](../docs/official_argformer_runtime_qualification_v1.md), [experiments/results/argformer_native_environment_v1/completion.json](../experiments/results/argformer_native_environment_v1/completion.json), [experiments/results/argformer_base_dependency_probe_v1/receipt.json](../experiments/results/argformer_base_dependency_probe_v1/receipt.json), [experiments/results/argformer_native_environment_v2/completion.json](../experiments/results/argformer_native_environment_v2/completion.json), [experiments/results/official_argformer_actual_mlm_v1/receipt.json](../experiments/results/official_argformer_actual_mlm_v1/receipt.json), [experiments/results/official_argformer_actual_contrastive_v1/receipt.json](../experiments/results/official_argformer_actual_contrastive_v1/receipt.json), [experiments/results/official_argformer_native_qualification_sequence_v1.json](../experiments/results/official_argformer_native_qualification_sequence_v1.json).

## H014: Native AncestralPaths supplies biological path ancestry and a comparator

**native_biological_CPU_fit_registered_waiting_Relate.** All17 original files match the paper-cited Zenodo version, native5x9 features/six classes plus explicit probability collapse.

**Conclusion:** The actual TensorFlow2.15.1 source-native Dense runtime passed forward, synthetic update and exact checkpoint replay. The native biological runner preserves source TRAIN class-stratified quotas,5epochs/batch32 and six-output/C4 probability sums. Independent static audit exposed and repaired Relate manifest/local-coordinate defects, then found no remaining static blockers. Native fits are scheduled after all four observed-only inferred parents complete, no score exists yet.

**Next/repeat rule:** Execute the registered three-seed biological source-native fit once on the frozen inferred manifest. Preserve quota stalls/fit failures and correct specific evidence-directed defects before a separately recorded retry. Do not silently relax native sampler or delete difficult validation bins/queries.

Evidence: [docs/ancestralpaths_native_protocol_v1.md](../docs/ancestralpaths_native_protocol_v1.md), [experiments/results/ancestralpaths_native_cpu_validation_20261010_v2.json](../experiments/results/ancestralpaths_native_cpu_validation_20261010_v2.json), [configs/ancestralpaths_native_cpu_requirements_v1.txt](../configs/ancestralpaths_native_cpu_requirements_v1.txt), [scripts/qualify_ancestralpaths_runtime.py](../scripts/qualify_ancestralpaths_runtime.py), [experiments/results/ancestralpaths_native_runtime_v1/receipt.json](../experiments/results/ancestralpaths_native_runtime_v1/receipt.json), [experiments/results/ancestralpaths_native_environment_v1/completion.json](../experiments/results/ancestralpaths_native_environment_v1/completion.json), [scripts/train_ancestralpaths_biological_v1.py](../scripts/train_ancestralpaths_biological_v1.py), [configs/ancestralpaths_biological_pilot_v1.json](../configs/ancestralpaths_biological_pilot_v1.json), [docs/ancestralpaths_biological_pilot_v1.md](../docs/ancestralpaths_biological_pilot_v1.md), [research/reviews/ancestralpaths_biological_integration_v1/corrected_review.json](../research/reviews/ancestralpaths_biological_integration_v1/corrected_review.json).

## H015: Fresh engineering development eliminates repeated-pilot selection

**preparation_in_progress.** Frozen 64 TRAIN/16 VAL/TEST0 AncientEurope_4A21, Q64/24-reference cohort and deterministic runners.

**Conclusion:** All80 oracle parents are prepared. Relate inferred-parent preparation is still running. The cohort is an engineering benchmark, not any supplied paper's numerical reproduction. No fresh fitting or final selection has completed.

**Next/repeat rule:** Complete the existing frozen preparation without editing pinned sources. A biological benchmark amendment is required before final confirmation.

Evidence: [docs/fresh_development_preparation_v1.md](../docs/fresh_development_preparation_v1.md), [experiments/results/fresh_development_preparation_v1](../experiments/results/fresh_development_preparation_v1).

## H016: Deterministic CUDA execution is admitted on the project stack

**engineering_check_completed.** Synthetic admission v2 on actual physical GPU3, neural gradients/replay and deterministic smoothers.

**Conclusion:** All44 synthetic CUDA cases passed. Initial UUID-prefix admission rejection is retained. These checks do not establish ancestry accuracy or completed fresh training.

**Next/repeat rule:** Repeat only if the admitted dependency/model/runtime changes. Do not use CPU/mock CUDA receipts as actual GPU evidence.

Evidence: [experiments/results/synthetic_cuda_admission_v1](../experiments/results/synthetic_cuda_admission_v1), [experiments/results/synthetic_cuda_admission_v2](../experiments/results/synthetic_cuda_admission_v2), [research/reviews/resource_snapshot_cuda_admission_v2.json](../research/reviews/resource_snapshot_cuda_admission_v2.json).

## H017: Sealed confirmation workflow protects independent evaluation

**engineering_check_completed_not_scientific.** Fresh-context audit and repaired lock/prediction/custodian/temperature/joined analysis modules.

**Conclusion:** Audits exposed and repaired provenance/publication/tile-edge defects. Final CPU suites and independent probes passed. No real selection/calibration/ID confirmation or genuine competitor adequacy was certified.

**Next/repeat rule:** Repeat targeted checks only after relevant code changes. Revise the biological task/comparator contract before any final lock or held-out access.

Evidence: [research/reviews/confirmatory_workflow_audit_v1](../research/reviews/confirmatory_workflow_audit_v1), [experiments/results/custodian_cpu_validation_20261010_v7.json](../experiments/results/custodian_cpu_validation_20261010_v7.json), [docs/validation_temperature.md](../docs/validation_temperature.md).

## H018: Real-data patterns support a new biological conclusion

**not_started.** Public phased/dated ancient and modern cohort, inferred trees, fixed ancestry and hypothesis contract.

**Conclusion:** No real-data biological hypothesis has been tested here. Real-data agreement cannot establish per-site accuracy without ground truth.

**Next/repeat rule:** Prespecify a falsifiable hypothesis and public-data provenance first. Report concordance/robustness with uncertainty rather than known ancestry accuracy.

Evidence: [docs/ancestralpaths_native_protocol_v1.md](../docs/ancestralpaths_native_protocol_v1.md), [docs/benchmark_plan.md](../docs/benchmark_plan.md).

## H019: A learned low-rank whole-tree interface supplies a novel scalable kernel

**standard_operator_novelty_claim_rejected_no_training.** Analytic proposal P row-stochastic, B=P diag(P^T1)^(-1/2), grounded generator (lambda+delta)I-lambda B_Q B_Q^T, exact rank-r actions and prior-art comparison.

**Conclusion:** The unconstrained operator is standard normalized bipartite diffusion and a learned-star Kron response, with close hypergraph/DiffPool precedents. Positivity/contraction and rank-r actions are derivable, but Q64 speed, predictive benefit and distinct novelty are untested. No implementation or training ran.

**Next/repeat rule:** Do not implement/retest the bare operator as novel. Reopen only for a precise observed age/path/shared-ancestor constraint with a justified scientific hypothesis and a new comparison against the standard operator.

Evidence: [docs/low_rank_whole_tree_kernel_design_v1.md](../docs/low_rank_whole_tree_kernel_design_v1.md), [research/protocols/low_rank_whole_tree_kernel_design_v1.json](../research/protocols/low_rank_whole_tree_kernel_design_v1.json).

## H020: Native competitors and the candidate improve source/path ancestry on a matched biological panel

**four_biological_parents_simulated_observed_Relate_chain_registered.** Four new2Mb parents,2TRAIN/2VAL/TEST0,140 diploids with96 pure-source reference haplotypes,64 mixed-reference haplotypes and120 queries. Source-derived11-population model with census350 and fixed six-path/C4 labels, observation-only Relate inference before native fits.

**Conclusion:** All four development parents completed simulation with separate hidden census truth and observed phased genotypes/rosters. The140-diploid cohort differs from the authors476 cohort and200Mb setting. Next registered execution infers these observations with native Relate1.2.4 plus five population-size iterations, preserving pre/post conditions and exact SNP/coordinate/sample-ID joins. No biological fit or validation score exists yet.

**Next/repeat rule:** Finish this new observed-only Relate chain once. Preserve failures and correct specific defects under new receipts instead of rerunning an unchanged cell. Native biological fits use this exact observed manifest, with six/C4/output/supervision distinctions recorded. Power confirmation from development variation.

Evidence: [docs/biological_benchmark_contract_v1.md](../docs/biological_benchmark_contract_v1.md), [configs/biological_benchmark_contract_v1.json](../configs/biological_benchmark_contract_v1.json), [docs/biological_paths_execution_pilot_v1.md](../docs/biological_paths_execution_pilot_v1.md), [configs/biological_paths_execution_pilot_v1.json](../configs/biological_paths_execution_pilot_v1.json), [research/protocols/biological_paths_execution_pilot_v1.freeze.json](../research/protocols/biological_paths_execution_pilot_v1.freeze.json), [experiments/results/biological_paths_execution_pilot_v1/completion.json](../experiments/results/biological_paths_execution_pilot_v1/completion.json), [docs/biological_paths_relate_v1.md](../docs/biological_paths_relate_v1.md), [configs/biological_paths_relate_v1.json](../configs/biological_paths_relate_v1.json).
