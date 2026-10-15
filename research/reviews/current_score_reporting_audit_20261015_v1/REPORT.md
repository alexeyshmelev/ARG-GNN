# Current detailed score reporting audit

Audited scoreboard snapshot: **2026-10-15T17:29:55.098832+00:00**. Read-only audit of existing scored JSON and completion/wait metadata. No new inference, model/native runtime, truth arrays, GPU, admission, accounting or Git action. Only this audit directory was written.

All 17 European setups use 32 TRAIN and eight reused development VAL simulations, 2 Mb, 952 observed haplotypes and 686 scored queries. Means weight independent parents equally after common physical-span scoring. Accuracy/F1 are percentages and their sample SDs are percentage points across eight VAL simulations. NLL is in nats. One optimizer seed is used per setup. This is development evidence, with no untouched TEST or statistical superiority claim.

## All 17 European setups

| Setup | Present accuracy % ± SD pp | NLL nats ± SD | Boundary F1 % ± SD pp | Training budget | Source |
|---|---:|---:|---:|---|---|
| H045 foreign_episode | 96.8270 ± 0.5870 | 0.092980 ± 0.020423 | 18.1558 ± 1.3790 | H042 lineage 1920 + 960 new updates; 30 extra passes | S5 |
| H045 own_state | 96.8175 ± 0.6142 | 0.092976 ± 0.020230 | 18.0557 ± 1.7761 | H042 lineage 1920 + 960 new updates; 30 extra passes | S5 |
| H044 localized | 96.7919 ± 0.5967 | 0.094944 ± 0.020882 | 15.5374 ± 1.1777 | H042 lineage 1920 + 960 new updates; 30 extra passes | S4 |
| H046 foreign_flank | 96.7892 ± 0.6042 | 0.094685 ± 0.020530 | 17.4847 ± 1.7741 | H042 lineage 1920 + 960 new updates; 30 extra passes | S6 |
| H044 tree_pooled | 96.7866 ± 0.6161 | 0.094962 ± 0.021065 | 18.6750 ± 1.2687 | H042 lineage 1920 + 960 new updates; 30 extra passes | S4 |
| H047 original_risk | 96.7777 ± 0.5970 | 0.094808 ± 0.020849 | 18.2775 ± 1.7493 | H042 lineage 1920 + 960 new updates; 30 extra passes | S7 |
| H046 own_state | 96.7776 ± 0.6351 | 0.094792 ± 0.021067 | 17.3354 ± 1.6777 | H042 lineage 1920 + 960 new updates; 30 extra passes | S6 |
| H047 balanced_risk | 96.7576 ± 0.5287 | 0.096819 ± 0.021357 | 18.5574 ± 1.7800 | H042 lineage 1920 + 960 new updates; 30 extra passes | S7 |
| H042 exact local-clade flags | 96.5778 ± 0.5954 | 0.100656 ± 0.020069 | 18.0451 ± 1.8423 | H038 960 + 960 new updates; 30 extra passes | S2 |
| H043 true_partner | 96.5182 ± 0.6201 | 0.101527 ± 0.019742 | 17.4301 ± 1.3301 | H042 lineage 1920 + 960 new updates; 30 extra passes | S3 |
| H043 own_state | 96.5029 ± 0.6006 | 0.101819 ± 0.019255 | 17.1055 ± 0.7889 | H042 lineage 1920 + 960 new updates; 30 extra passes | S3 |
| H040 tensor | 96.2674 ± 0.7493 | 0.109332 ± 0.021073 | 16.1986 ± 2.0270 | H038 960 + 960 new updates; 30 extra passes | S1 |
| H040 event_mlp | 96.2007 ± 0.7048 | 0.110014 ± 0.020489 | 16.1312 ± 1.6911 | H038 960 + 960 new updates; 30 extra passes | S1 |
| H038 aligned whole-tree PyG | 95.4891 ± 0.8338 | 0.133497 ± 0.026419 | 13.4195 ± 1.4686 | Fresh 30 passes / 960 updates; selected pass 30 | S8 |
| H038 zero-concordance control | 94.9252 ± 0.7917 | 0.147232 ± 0.023024 | 11.7904 ± 1.3304 | Fresh 30 passes / 960 updates; selected pass 25 | S9 |
| Current native ARGMix | 94.4461 ± 1.1491 | 0.219695 ± 0.023975 | 11.7372 ± 1.7328 | 11 epochs; native early stop, selected epoch 9, seed 50 | S10 |
| Native AncestralPaths derived C4 | 93.7830 ± 0.8385 | ∞ | 7.0281 ± 0.7779 | 5 epochs, batch 32, 5,190,874 rows; final epoch 5, seed 101 | S10 |

GNN fits use FP32 AdamW, learning rate 0.001, weight decay 0.0001 and seed 101. Each full pass has 32 parent updates. H040/H042 include 960 H038 initialization updates; H043–H047 include 1920 H042-lineage updates before 960 new updates. These are cumulative model-lineage exposure counts, with reset continuation optimizers. H038 zero-concordance completed 960 updates but selected pass 25 (800 updates). Native objectives and schedules differ, so these are not matched training-budget or architecture-only comparisons.

## Seven cohort best mean accuracies

| Cohort | Setup | Accuracy % ± SD pp | Source |
|---|---|---:|---|
| Present182 | H045 foreign_episode | 96.8270 ± 0.5870 | S5 |
| BronzeAge162 | H045 foreign_episode | 99.2047 ± 0.3199 | S5 |
| BAA14 | H047 balanced_risk | 99.3747 ± 0.6939 | S7 |
| Neo302 | H045 own_state | 99.6424 ± 0.0859 | S5 |
| Yam26 | H045 foreign_episode | 99.9039 ± 0.1292 | S5 |
| Ancient504 | H047 original_risk | 99.4931 ± 0.1426 | S7 |
| All686 | H045 own_state | 98.7815 ± 0.2566 | S5 |

Ancient504's exact leader is H047 original_risk (99.49311244093180%), narrowly above H046 foreign_flank (99.49306819538324%); both round to 99.4931%. These are maxima among reused development rows, not independent discoveries. BronzeAge162 is not asserted identical to the paper's Post-Neolithic cohort.

All 119 setup/cohort rows, including all seven accuracies and every available scalar metric, are in [JSON](all17_allcohort_metrics.actual.json) and [CSV](all17_allcohort_metrics.actual.csv). They include balanced accuracy, macro F1, NLL, boundary precision/recall/F1, Brier, 15-bin ECE and dosage MSE/MAE, with sample SD and defined/undefined-parent counts. JSON retains exact floats.

## Metric normalization and provenance

All 17 scoreboard summary source hashes verified. Across 2066 numeric comparisons, the largest absolute difference from recomputed scored scalar receipts was 1.11e-16. See [verification receipt](verification.actual.json). The audit reaggregates existing scalar metrics only.

H040/H044 saved selected_full_diagnostics cells contain all seven cohort metrics and SDs, including metrics formerly dashed in the scoreboard. H043 also has macro F1, Brier, ECE and dosage metrics in its saved selected diagnostics. Root integrated the H040/H044 helper and expanded the detail table while this audit ran. The audit JSON supplies the full 17-setup crosswalk, including remaining available metrics.

Native AncestralPaths raw NLL is **∞ in Present, BronzeAge, Neo, Yam, Ancient and All**, proven by positive infinite_nll_query_bp. It is not missing, and finite parents must not be averaged as a substitute. Only BAA14 is finite: 0.10171249788097754 ± 0.043356347126851054. Native Yam boundary F1 is undefined for one parent; conditional n=7 means are AP 0.4655728068 ± 0.4007173896% and ARGMix 10.0349594788 ± 18.3801075877%. Root integrated the supplied native override, preserving those counts.

Native ARGMix uses seed 50 and native early stopping (11 completed epochs, selected epoch 9, cap 20, patience 2). AP is the released TensorFlow Dense model, five epochs, batch 32, 5,190,874 realized TRAIN rows, seed 101, final epoch 5 without VAL selection. Six path probabilities are collapsed to C4 before argmax. The released Dense baseline is not the paper's unspecified CNN.

H044's scoring outputs were completed and independently verified, but the original CLI exited 1 after a return-type/logging bug. Its real exit 1 and owned-process drain remain disclosed; no successful original exit is manufactured.

## Other native scores and pending work

| Condition | Actual result / budget | Status and source |
|---|---|---|
| Earlier adapted ARGformer | Top1 Present64 75.6818 ± 4.5195%, Ancient56 75.4447 ± 3.5081%, All120 75.5711 ± 4.0476%. Top5 75.1358 ± 3.7500%, 74.7885 ± 4.0220%, 74.9737 ± 3.8769%. Raw NLL ∞. | Separate two-VAL, 280-input, 2 Mb pilot; per-ARG five MLM + one reference-only contrastive epoch. Not one of the 17 full-panel rows. experiments/results/native_argformer_tsinfer_fit_v7_summary_v2/summary.json |
| GST USPTO native pilot | GSC-CA AP 0.6379193900009189 (epoch 19), GSC-BC 0.6537832932979444 (19), GCN-DS 0.6394512876723991 (17), GCN-ST 0.6382343233309435 (16). | First 1000 retained TRAIN / all 1495 VALID, seed 0, CPU, 20 epochs / 1260 updates per fit, AdamW lr 0.001, weight decay 0.0001, batch 16. No seed SD. TEST withheld. experiments/results/gst_uspto_native_TRAIN_VALID_seed0_v1/summary.json |
| GST full-data extension | Parser retained 10,439/10,500 TRAIN reactions and 364,213 atoms. Full 20-epoch batch-16 budget is 13,060 updates per fit. | Data readiness only. Full-data/five-seed fit, our molecular adapter and paired TEST comparison remain unfinished. research/reviews/GST_full_TRAIN_count_root_v1/counts.json |
| American full30Mb TRAIN000 | Corpus and genuine counts completed: contrastive 8,624,912 TRAIN + 959,065 VAL; 2105 updates at batch 4096, remainder 2832. MLM five-epoch budget 224775 updates; 10,000 query focals. | Counts completion acbb9512… and actual wait 6b2cad31…; counts are not training or accuracy. experiments/results/American_C3_native_train000_full30mb_counts980_v1/completed.json |
| Native American ARGformer | GPU5 MLM activity observed; five-epoch completion/final saved Adam verification remain pending. | No American native ancestry score. Contrastive/RAM source successor is inactive pending genuine MLM completion. Prior v11 exit 1 and unknown updates preserved. Snapshot original_American_C3 / native_ARGformer_execution. |
| Original30Mb VAL000 and FLARE | Simulation, inference and native-path/10k-focal preparation completed with real wait 0/drain; common R600/Q1000. FLARE target 228,477 sites. TRAIN000 native inference and 10k-focal call seal completed. | VAL conversion/predictions/seals and common-truth scores pending at the snapshot. TRAIN000 is engineering TRAIN, not held-out accuracy. American ARGformer, FLARE and GNN precision/recall/F1 remain pending. |
| H048 | GPU4 training; completed-pass update counts are a lower bound. 16/30 completed passes; {'graph_to_own_confidence_fixed_distance_chain': 512, 'graph_to_transition': 512} verified update lower bounds. No validation accuracy yet. | No validated cross-domain result or distinct graph-collection operator. GST reactant sets have no native chain order; PKU-MMD remains a possible future decoder study after label-overlap semantics. |
| Empirical H045 application | 355 trees, 2074 sites, 40 observed haplotypes, 24 query haplotypes / 12 subjects, 9,996,712 covered bp; probabilities 355×24×4, proportions and original-genotype PCA written; zero new optimization steps. | Completed descriptive application with genuine successful wait. No known ancestry truth or accuracy. Variable-panel I/O adaptation disclosed. Original strict mutation-export gate remains failed/preserved; original genotypes supply SNP features. experiments/results/H018_H045_empirical_application_root_v1/completed.json |

The American condition is a disclosed 1600-haploid, EUR/EAS/AFR reconstruction with a 12-generation pulse, not exact ARGformer Table 1 reproduction. The top1 downstream rule differs from unavailable paper PCA/clustering. Shared scoring identities/support are required for future paired method results.

## README correction handoff

The 17:25 README still had lower stale Table row 141 saying VAL simulation was running and inferred VAL observations were needed; all three preparation stages are complete. Row 142 still listed empirical application/proportions/PCA as needed; those descriptive outputs are complete. Line 150 still said the variable-panel adapter and ancestry/PCA result were pending. Root corrected these lower sections dynamically while this audit ran. The upper H046 running/no-result text had already been corrected. Historical language scoped to acquisition alone is not a current pending claim.

Retain the substantive limits: high span accuracy coexists with weak boundary F1. H043–H047 mechanism gains failed registered practical criteria. H045 foreign-episode gain is 0.009568 pp; H046 foreign-flank gain is 0.011573 ± 0.069124 pp paired SD; H047 balanced-risk gain is −0.020029 pp. These do not establish useful novel mechanisms. Independent TEST, seed robustness, exact paper reproduction, architecture-only attribution and cross-domain success remain unestablished.

## Exact raw-score source crosswalk

- S1: [experiments/results/H040_GPU45_frozen_evaluation_seed101_v3/completed.json](/workspace-SR003.nfs2/ARG-GNN/experiments/results/H040_GPU45_frozen_evaluation_seed101_v3/completed.json), SHA256 `db98cff6c9412b4fddef850f0e38c6c09bec769f82811d2bae6836804466cf72`.
- S2: [experiments/results/H042_GPU4_clade_flags_evaluate_seed101_memo_v1/completed.json](/workspace-SR003.nfs2/ARG-GNN/experiments/results/H042_GPU4_clade_flags_evaluate_seed101_memo_v1/completed.json), SHA256 `789600181135353f3ba87255ba2662ef467db883682acab1294e2f578f9ec271`.
- S3: [experiments/results/H043_GPU5_transport_evaluate_seed101_v1/completed.json](/workspace-SR003.nfs2/ARG-GNN/experiments/results/H043_GPU5_transport_evaluate_seed101_v1/completed.json), SHA256 `fd75c326fc4e494c71cf205963cc26c50427f60ee8bd90101c6234fa7606d7df`.
- S4: [experiments/results/H044_marker_cached_evaluate_seed101_v2/completed.json](/workspace-SR003.nfs2/ARG-GNN/experiments/results/H044_marker_cached_evaluate_seed101_v2/completed.json), SHA256 `bc46c8edfaae2ef0894cd8ae057a69ce1d6844bda8fb9262b849709e4b4524a4`.
- S5: [experiments/results/H045_GPU5_episode_evaluate_seed101_v1/completed.json](/workspace-SR003.nfs2/ARG-GNN/experiments/results/H045_GPU5_episode_evaluate_seed101_v1/completed.json), SHA256 `1f98815c0791120f512d7d472db39487a3839be96f7a7ff975a57326b3b1f14b`.
- S6: [experiments/results/H046_GPU5_flank_evaluate_seed101_eval980_v1/completed.json](/workspace-SR003.nfs2/ARG-GNN/experiments/results/H046_GPU5_flank_evaluate_seed101_eval980_v1/completed.json), SHA256 `3ba02434c6f02ee381389247dd9f16349e2a2fc4a8fb048f998e530ecea46a36`.
- S7: [experiments/results/H047_GPU4_cohort_risk_evaluate_seed101_v1/completed.json](/workspace-SR003.nfs2/ARG-GNN/experiments/results/H047_GPU4_cohort_risk_evaluate_seed101_v1/completed.json), SHA256 `08b39291332286cc0bfd9daa2105f2585156e24ccc50a97738316bd3644a0382`.
- S8: [experiments/results/mutation_routed_full_panel_pyg_v1_aligned_seed101_GPU2/final_selected_full_diagnostics.json](/workspace-SR003.nfs2/ARG-GNN/experiments/results/mutation_routed_full_panel_pyg_v1_aligned_seed101_GPU2/final_selected_full_diagnostics.json), SHA256 `5d29abd41a479b5bf9308e1536dd1031ce8b307d9de043bec54fbeef7ce0886b`.
- S9: [experiments/results/mutation_routed_full_panel_pyg_v1_zero_concordance_seed101_GPU3/final_selected_full_diagnostics.json](/workspace-SR003.nfs2/ARG-GNN/experiments/results/mutation_routed_full_panel_pyg_v1_zero_concordance_seed101_GPU3/final_selected_full_diagnostics.json), SHA256 `f4f0d9d0f3b5a5ca1317ac87213332831c4593db9205f3b01615e6cbec5dcf27`.
- S10: [research/reviews/native_common_cohort_diagnostics_20261015_v1/per_parent_metrics.json](/workspace-SR003.nfs2/ARG-GNN/research/reviews/native_common_cohort_diagnostics_20261015_v1/per_parent_metrics.json), SHA256 `f221906384c10e05803d06d0eaee8c2c40936bbb9c3063f410ad4d53501aebde`.

Frozen integration helpers: [H040/H044 normalizer](normalization_reference_v1.py) and [native overrides](native_metric_overrides_v1.py). Existing scored JSON is their only input. No further edits to these two imported source files will be made without coordination.
