# External technical evidence appendix

Repository: /workspace-SR003.nfs2/ARG-GNN on anogena; paths below are relative to that root. This appendix is external audit material, not manuscript prose.

220 source/document/compact-metadata files are inventoried in evidence.json. All124 stored source/config/execution-reference hashes match. Payload/truth/checkpoint/weight hashes quoted in metadata were not dereferenced or rechecked.

## Source line pointers

- research/reviews/ARGMix_published_benchmark_audit_20261015_v1/audit.md
  - line 34: §2.2 states: “We trained both models on a simulated demography from Irving-Pease et al. 2024 containing 500,000 examples (subgraphs), generated from 5 replicates”.
  - line 36: §4.2 says “We modified the msprime simulations to exclude Bronze Age Anatolians”, added genotype error 0.001 to all samples including present-day samples, used the chromosome 22 recombination map, and ran Relate with effective population size 30,000 and mutation rate 1.25×10−8. Misspecification perturbs population sizes by 1–50%, makes admixture timings older by 1–25%, and changes divergence times by ±1–25%. Exact perturbation distributions, joint draws and raw seeds are unspecified.
  - line 38: §4.3 says “We reimplemented AncestralPaths in PyTorch” and “To mitigate the redundancy from local correlation between trees we skipped every ten trees.” Validation uses a separate replicate, 100,000 examples and every 100 trees. ARGMix uses 25 nearest references by TMRCA and three inferred ancestors, giving 29 nodes including the query. Both models use the best validation-loss checkpoint. Testing uses three additional replicates of 100,000 samples each, and five misspecified replicates of 100,000 samples without skipped trees.
  - line 60: | Mutation and observation error | Both nominalmu1.25e−8. H038 JC69 discrete Hudson simulation writes mutations directly to VCF with no genotype-error injection; paper error0.001. |
  - line 64: | AncestralPaths model | Paper PyTorch reimplementation, bestVALloss. H038 native Keras Flatten→Dense512ReLU→Dropout0.2→Dense6softmax, nativeAdam, five epochs, finalepoch. |
  - line 96: The real-data analysis in §2.3/Fig2B masks non-Anatolian **genotype haplotypes to missing**, then applies ancestry-specific PCA. A PCA of four ancestry-fraction vectors would be a different analysis. The paper reports 3,720,746 final variants, 1,611 individuals, Beagle5.4, ancient QC/imputation and INFO>0.8, Ötzi GLIMPSE and HGDP+1KGP references, plus snputils ancestry-specific analysis without exact version/settings. Accuracy on simulations cannot establish this real-data result.

- research/reviews/AncestralPaths_biological_protocol_audit_20261016_v1/REPORT.md
  - line 5: The paper defines ancestry as a route through a demographic history, including ancient admixed queries. Its six-path task, its BAA-excluded retrained four-path task, and the project's derived four-source span endpoint are different experiments. Current European development scores cannot be compared numerically with the paper's94.6% headline as if the labels, data and metric matched.
  - line 28: | Feature definition |Traverse from the focal haplotype towards the root. Record proportions of sampled ancient groups below successive informative ancestors, excluding leaves already counted at earlier ancestors. Use the first five informative nodes, ancient-group proportions plus node age, and -15 padding when the root is reached early. |30–32 |
  - line 31: | Main train/test description |Three200Mbp TRAIN sequences with different seeds.10,000 training pairs from each of five admixed cohorts,50,000 total, across evenly spaced sites. Five MORE simulated/inferred sequence pairs are used for testing. Relate mutation1.25e-8 and initial haploid population estimate30,000 are stated. |32 |
  - line 36: The source calls the classifier a convolutional neural network on p31. The archived training.py actually defines Flatten(5,9), Dense512/ReLU, Dropout0.2, Dense6/softmax and five epochs (lines173–177). Reading this method section does not recover an exact published CNN specification. The current baseline can properly be called the archived/released Dense workflow with disclosed adaptations; it cannot silently become a numerical reproduction of the paper's stated CNN.
  - line 38: ### The94.6% headline is not today's C4 span score
  - line 42: - Remove BAA from GNN distributions AND remove path labels5/6, retrain a smaller-feature four-path model: reported94.3% +/-0.27%, over five testing tree sequences in GBR.
  - line 43: - Keep BAA and a six-path classifier, then average over the four non-BAA paths: reported94.6% +/-0.46%. Fig.S16 still allows six predicted path labels and shows predicted-row precision.
  - line 99: 1. Keep three AP endpoints labelled separately in prose/tables: native six-path; BAA-removed/retrained four-path; project-derived C4 probability collapse. Cite pp27 and43–44 for the distinction. Do not use the94.6 headline as a matched baseline for current span accuracy.
  - line 103: 5. Keep the archived Dense model, source sampler quotas and runtime/selection adaptations disclosed. Source scripts use TRAIN parents3,5,6 and TEST5–9, unlike the supplement's five MORE test sequences; current independent-parent fixes are appropriate new-study corrections, not unchanged reproduction. Exact original CNN details, raw streams, site sampling and trained weights remain missing.

- research/reviews/ARGformer_published_benchmark_audit_20261015_v1/audit.md
  - line 29: “Samples” is not explicitly resolved as haploid genomes versus diploid people in the article. This matters: stdpopsim's old positional/get_samples API counts haploids, while0.3.0 dictionary and named-population CLI counts diploid individuals. Passing1000/200/200/200 unchanged to the new API creates3,200 haplotypes. The inactive reconstruction proposal explicitly chooses1,600 haploid genomes (800 paired diploid subjects) and marks this as an assumption to confirm, not a fact proven about Table1.
  - line 31: Section3.1 evaluates10,000 unique marginal paths sampled from admixed individuals. PCA is fitted on unadmixed reference embeddings, retains10 PCs, and query ancestry is assigned by reference-cluster membership. Retrieval uses nearest reference-haplotype path embeddings and their reference labels. FLARE consumes simulated genotypes; ARGformer consumes the inferred genealogy. This is a native-method/input comparison, not proof that genotype and inferred-ARG inputs are identical information modalities.
  - line 33: ARGformer encodes ARG-specific node identifiers along leaf-to-root marginal paths. Shared internal IDs across trees are meaningful. MLM predicts masked node tokens, not ancestry; paper masking probability is0.30. Contrastive learning uses a learned pooling head, L2-normalized embeddings, supervised reference-population positives, in-batch negatives and inverse-frequency anchor weighting (Eq5). The paper states a default temperature0.05.
  - line 48: | Weighting |sequence_contrastive.py calls build_contrastive_model(cfg.model); the weighting code looks for cfg.train_loader and catches failure with a uniform-weight warning | Released configuration lacks that nested loader, so the stock call can take uniform weighting rather than paper Eq5. Record actual warning/weights. Fixing it is a disclosed implementation correction, not unchanged code. |
  - line 51: | PCA classifier |Not present as an integrated Table1 evaluator | Ten PCs are specified, but clustering/label-assignment method, fit details and ties are unresolved. Do not invent “k-means” or “nearest centroid” and call it exact. |
  - line 53: Other unresolved details are simulation/mutation seeds, tsinfer/tsdate versions/options, error-free VCF creation, original ploidy/count convention, donor/query path curation, exact checkpoint/step and pooling setting, which10,000 unique paths were chosen, and the true ancestry/FLARE call coordinate for a path recurring over multiple intervals. The latter determines the score unit: a unique inferred path can recur over several disjoint spans and cross true-ancestry changes. A hidden purity filter or majority-label convention would materially change the benchmark.

- research/reviews/ARGformer_Table1_downstream_resolution_20261016_v1/RESOLUTION.md
  - line 1: The paper has two Table 1 classifiers: PCA10/clustering and nearest-reference retrieval. The learned head_norm embedding sits upstream of either classifier. Continue the prepared native head_norm/top1 route unchanged as an author-release reconstruction on common data. The exact Table 1 PCA classifier is not recoverable from the inspected paper/supplement and pinned public release/history.
  - line 5: The pinned author release f7545be7bca0200fb39aeb7f94a8e5f0b6bf5596 implements a trained pooling head and head_norm L2 normalization (embeddings.py:98–178,162–171), with encoder/head load checks at74–90. retrieve.py:13–16,38–114 implements normalized dot-product streamingtopk; CLI default k=1 is at124. Existing RAM I/O calls those exact APIs withk1,q_batch1024,e_chunk10,0000. Unit-vector Euclidean/cosine rankings agree except numeric/tie behavior; preserve author torch.topk behavior. No PCA precedes this released retrieval route.
  - line 9: Runnable route: own-VAL000 MLM→known-reference contrastive/readout→four separate query/train,query/val,reference/train,reference/val extraction calls→train-then-val concatenation→native normalizedtop1→reference-sidecar label lookup→10,000-call seal. Use own contrastive checkpoint, head_norm,lastlayer,batch256,max_samples0; querylabels disabled/reference labels enabled. The author multi-split extractor randomly interleaves loaders, so preserve existing single-split calls/fullpath sidecars. CLI defaults first_node/max_samples10,000 are not this configured route. Current RAM outputs preserve native indices/scores and one-hot calls as hard-call t
  - line 15: Truth barrier: MLM observes paths only; contrastive train/eval and nearest-neighbor labels use known unadmixed references. Query path filenames train/val grant no ancestry labels. All10,000calls/identities seal before the separate scorer opens private pulse truth at frozen half-open interval coordinates. No query-truth cluster permutation, class mapping, k/temperature/cluster tuning or post-hoc focal filtering is legitimate. No TEST route is introduced.
  - line 17: Keep disclosed paper/release differences: paperdefaulttemperature.05 versus releaseYAML.30, and inverse-frequency weights versus unchanged config's uniform fallback (sequence_contrastive.py:598–623). Injecting cfg.model.train_loader would be a new variant. Haploid interpretation, current inference/unary processing and focal seed/midpoint support remain reconstructed. No suitable public author checkpoint/evaluator was recovered, so the own-ARG MLM→contrastive chain remains the supported initializer route.

- experiments/results/expanded_native_argmix_full32TRAIN8VAL_fit_v1/fit/training_backend.yaml
  - line 24: num_layers: 4
  - line 25: num_refs: 15
  - line 43: num_layers: 1
  - line 46: early_stop_patience: 2
  - line 51: max_train_subgraphs: null
  - line 53: num_epochs: 20
  - line 72: truth_label_mode: first_snp

- experiments/results/expanded_native_argmix_full32TRAIN8VAL_fit_v1/fit/train_joint_model.log
  - line 2: 2026-10-12 04:08:50,612 __main__ INFO: Loaded prebuilt joint training tensor chunks: train=(21952, 1024, 16) val=(5488, 1024, 16) archive_seq_len=train:16,val:16 expected_seq_len=16 num_refs=15 base_encoder=tmrca_transformer base_final_layer_dim=16 gru_input_dim=17 pad_id=0 vocab_size=9
  - line 11: 2026-10-12 05:17:49,794 __main__ INFO: Joint epoch 9/20 train_loss=0.4350 train_acc=0.9591 val_loss=0.4016 val_acc=0.9765
  - line 13: 2026-10-12 05:39:12,799 __main__ INFO: Joint epoch 11/20 train_loss=0.4325 train_acc=0.9604 val_loss=0.4023 val_acc=0.9758
  - line 14: 2026-10-12 05:39:12,800 __main__ INFO: Early stopping triggered.

- .runtime/author_sources/argmix_c2481248eb0d/src/config_loader.py
  - line 41: raise ValueError("model.num_ancs is no longer supported (ancestor features were removed)")
  - line 406: raise ValueError("model.num_ancs is no longer supported (ancestor features were removed)")

- .runtime/author_sources/argmix_c2481248eb0d/src/tree_sampler.py
  - line 152: raise ValueError("num_ancs is no longer supported (ancestor features were removed)")

- .runtime/author_sources/argmix_c2481248eb0d/src/joint_interval_model.py
  - line 777: num_ancs=0,

- .runtime/author_sources/argmix_c2481248eb0d/src/rust/ancpath/src/bin/build_training.rs
  - line 328: bail!("model.num_ancs is no longer supported (ancestor features were removed)");
  - line 330: let num_ancs: usize = 0;

- .runtime/author_sources/argmix_c2481248eb0d/src/run_inference.py
  - line 481: raise ValueError("model.num_ancs is no longer supported (ancestor features were removed)")
  - line 482: num_ancs = 0
  - line 765: f"Checkpoint num_ancs={ckpt_num_ancs} is unsupported (ancestor features were removed)"

- .runtime/author_sources/argmix_c2481248eb0d/src/rust/ancpath/src/fast_lca/engine.rs
  - line 934: for _ in 0..num_ancs {
  - line 951: buffers.nodes.extend(buffers.inferred_ancs.iter().copied());

- .runtime/author_sources/argmix_c2481248eb0d/docs/config_reference.md
  - line 90: - `model.num_refs` is how many reference haplotypes each predict haplotype is scored against, taken nearest-first by TMRCA. It sets the size of the per-interval TMRCA matrix, so it drives extraction time and memory more than anything else here. Setting this to be quite low (less than 10 even) can be useful for little reduction in accuracy depending on the demography.

- src/arggnn/expanded_native_ancestralpaths_v1.py
  - line 145: training=dict(seed=101, epochs=5, batch_size=32, shuffle=True, checkpoint="final_epoch5",
  - line 147: sampler=dict(source_num_seq=3, maximum_source_passes=128,
  - line 381: bounded_realized_sampler_departure_from_unlimited_author_loops=True, VAL_selection=False)
  - line 424: tf.keras.utils.set_random_seed(101)
  - line 429: epochs=5, batch_size=32, shuffle=True, verbose=2)
  - line 440: sampler_reports=reports, epochs=5, batch_size=32, shuffle=True, seed=101,
  - line 498: collapsed = bridge.collapse_native_probabilities(p6)
  - line 547: np.testing.assert_allclose(bridge.collapse_native_probabilities(p6), [[.60, .37, .02, .01]])
  - line 548: assert int(p6.argmax()) == 1 and int(bridge.collapse_native_probabilities(p6).argmax()) == 0

- .runtime/author_sources/ancestralpaths_77db3cfae100/training.py
  - line 20: from keras.layers import Dense, Dropout, Activation, Flatten
  - line 173: model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(5, 9)),tf.keras.layers.Dense(512, activation=tf.nn.relu),tf.keras.layers.Dropout(0.2),tf.keras.layers.Dense(6, activation=tf.nn.softmax)])
  - line 177: model.fit(X_train, lab_train_target, epochs=5)

- src/arggnn/train_official_argformer_biological.py
  - line 493: cfg.device_train_microbatch_size = 64
  - line 503: model = module.build_contrastive_model(cfg.model)
  - line 505: raise RuntimeError("Source uniform fallback and unused three-class layer required")
  - line 633: trainer.fit()
  - line 635: trainer.save_checkpoint(str(checkpoint))

- .runtime/author_sources/argformer_f7545be7bca0/sequence_contrastive.py
  - line 417: temperature: float = 0.05,
  - line 604: local=cfg.train_loader.local,
  - line 622: print(f"[WARN] Failed to compute class weights: {e}. Using uniform weighting.")
  - line 832: model = build_contrastive_model(cfg.model)

- .runtime/author_sources/argformer_f7545be7bca0/yamls/mlm.yaml
  - line 115: max_duration: 5ep
  - line 118: global_train_batch_size: 512
  - line 122: device_train_microbatch_size: 512
  - line 123: precision: amp_bf16
  - line 149: save_interval: 500ba

- .runtime/author_sources/argformer_f7545be7bca0/yamls/contrastive.yaml
  - line 46: temperature: 0.3
  - line 65: negative_sampling_temperature: 1.0
  - line 94: max_duration: 1ep
  - line 97: global_train_batch_size: 4096
  - line 101: device_train_microbatch_size: 4096
  - line 102: precision: fp32
  - line 123: save_interval: 20ba

- research/reviews/native_VAL000_GPU0_borrowed32_source_20261016_v2/native_child_v1.py
  - line 61: def _final_checkpoint(directory,*,epochs,updates,max_files,max_bytes,torch,reference):
  - line 107: # No cfg.model.train_loader injection: preserve current uniform fallback.
  - line 212: and resolved["save_num_checkpoints_to_keep"]==1,"Unchanged full author numeric budget and checkpoint retention")
  - line 229: trainer=module.main(config,return_trainer=True)
  - line 236: require(observed_trainer["epoch"]==5 and observed_trainer["batch"]==227990
  - line 237: and observed_optimizer["unique_step_values"]==[227990],
  - line 238: "Actual full epoch5/update227990 Trainer timestamp")
  - line 240: final=_final_checkpoint(directory/"checkpoints",epochs=5,updates=227990,
  - line 244: expected_training_budget=dict(epochs=5,optimizer_updates=227990),observed_counters_verified=True)

- research/reviews/American_C3_FLARE_source_v1/AdmixPar_v0.6.0.java
  - line 79: private static final float DEF_MIN_MAF = 0.005f;
  - line 80: private static final int DEF_MIN_MAC = 50;
  - line 82: static final float DEF_GEN = 10f;
  - line 83: private static final boolean DEF_EM = true;
  - line 84: static final boolean DEF_UPDATE_P = false;
  - line 88: private static final long DEF_SEED = -99999;
  - line 96: static final int DEF_EM_ITS = 20;
  - line 137: min_maf = Validate.floatArg("min-maf", argsMap, false, DEF_MIN_MAF, -FMAX, Math.nextDown(0.5f));
  - line 138: min_mac = Validate.intArg("min-mac", argsMap, false, DEF_MIN_MAC, 0, IMAX);
  - line 163: em_its = Validate.intArg("em-its", argsMap, false, DEF_EM_ITS, 0, IMAX);
  - line 224: + "  min-maf=<minimum MAF in reference VCF file>           (default: " + DEF_MIN_MAF + ")" + nl
  - line 225: + "  min-mac=<minimum MAC in reference VCF file>           (default: " + DEF_MIN_MAC + ")" + nl

- src/arggnn/american_c3_evaluate_v1.py
  - line 6: prediction_ref=None,focal_registry_ref=None,external_truth_ref=None,method=None,output_cap_bytes=10000000,
  - line 43: def verify_prediction(p,r):
  - line 57: require(cfg['external_truth_ref'] is None and cfg['prediction_ref'] is None,'Native seal accepts no private truth/prediction input')
  - line 115: prediction['probabilities']=np.eye(3,dtype=np.float32)[calls];verify_prediction(prediction,r)
  - line 127: def execute(cfg,w):
  - line 133: calls=verify_prediction(load_npz(cfg['prediction_ref']),r)
  - line 135: truth=load_npz(cfg['external_truth_ref'])
  - line 151: prediction_ref=cfg['prediction_ref'],focal_registry_ref=cfg['focal_registry_ref'],external_truth_ref=cfg['external_truth_ref'],

- src/arggnn/american_c3_flare_evaluate_v1.py
  - line 17: projection=PROJECTION, max_focal_records=10000, max_markers=1000000,
  - line 27: and cfg["projection"] == PROJECTION, "Native sealing cannot read private truth")
  - line 34: and native["projection"] == converted["projection"] == PROJECTION,
  - line 35: "Exact unmodified native inputs/common focal projection required")
  - line 139: and fields[8] == "GT:AN1:AN2:ANP1:ANP2",
  - line 218: projection=PROJECTION, edge_carried_focal_records=edge_carried,
  - line 219: common_focal_registry_unchanged=True, native_AN1_AN2_calls_used_unchanged=True,
  - line 220: one_hot_is_native_call_transport=True, rounded_ANP_renormalized=False,
  - line 236: and sealed["native_AN1_AN2_calls_used_unchanged"] is True

## Actual metadata JSON pointers

- ARGMix experiments/results/expanded_native_argmix_full32TRAIN8VAL_fit_v1/completed.json: /fit_ref, /input_ref, /native_checkpoint_selection, /TRAIN_parents, /VAL_parents. fit/native_fit.json: /native_selected_epoch, /source_patience, /source_epochs_cap, /owned_code_sha256, /native_backend. fit/train_joint_model.py.command.json: /stock_source, /source_sha256, /argv, /exit_code. native_exec.json supplies the actual outer invocation.
- AP data/expanded_native_ancestralpaths_v2/completed.json: /source_ref, /config_ref, /fit_ref, /epochs, /VAL_selection, /realized_training_rows, /sampler_shortfall_rows. run/fit/fit.json: /history, /native_model, /optimizer, /native_path_counts, /sampler_reports. Per-cohort sums and exact three deficits are extracted in evidence.json /AP_realized_sampling. Score experiments/results/expanded_native_AP_H038_VAL_score_v1/summary.json: /C4_primary_equal_parent, /native6_SNP_focal_equal_parent, /checkpoint_selection.
- Reduced-European ARGformer experiments/results/native_argformer_tsinfer_fit_v7_summary_v2/summary.json: /retrieval_receipts, /equal_parent_means/1/primary_present_day_64, /disclosures. Each retrieval receipt /prediction_seal/root_native_fit binds its own-parent fit. VAL000 and VAL001 fit/*r003/receipt.json: /pretraining, /contrastive, /encoder_initialization, /true_query_ancestry_read. Each phase/resolved_source_invocation.json gives actual settings; phase_receipt.json gives epochs/updates. configs/native_argformer_tsinfer_fit_v7.json: /native_recipe, /retrieval_rule. configs/official_argformer_tsinfer_biological_v2.json: /data_fidelity, /pretraining, /contrastive, /retrieval.
- Older adapted ARGformer experiments/results/official_argformer_biological_readout_v1/summary.json has Present64 mean0.3589723174 and explicit fixed pure-reference C4 readout scope, distinct from native top1.
- American native32 research/reviews/native_VAL000_GPU0_borrowed32_source_20261016_v2/source_handoff.v2.json: /status, /source_only, /execution_authorized, /full_native_local_ancestry_pipeline_completed. science.template.json: /scientific_protocol. v1 operational import failure is preserved via /P1_independent_review_ref; v2 is source-only, not an execution result.
- FLARE data/FLARE_val000_native_namespace_980_root_v1/completed.json: /source_commit, /native_options, /native_argv_ref, /binary_provenance_ref, /query_ancestry_truth_opened, /native_outputs_modified. native_argv.json: /argv, /no_model_or_gt_ancestries_supplied. data/native_FLARE_build_existing_JDK_root_v1/receipts/binary_provenance.json: /tag, /source_commit, /native_sources_modified. Cached commit_tag.json versus commit_head.json disambiguate tag and HEAD.
- FLARE data/FLARE_val000_convert_memo2_980_root_v2/completed.json: /input_sites, /native_expected_markers, /unsupported_sites_dropped, /recorded_recombination_rate, /query_haplotypes, /reference_haplotypes. Score experiments/results/FLARE_C3_VAL000_common_score_H020980_20261016_v1/completed.json: /records, /result, /probabilities_modified, /probability_floor, /author_Table1_exact_reproduction. Actual score terminal is retained in research/reviews/FLARE_VAL000_score_H020980_root_20261016_v1/actual_tool_terminal.root_v1.json.
- ARGMix_paper25plus3_source_search.json records the full77-file pinned source/docs/config search and hashes. Explicit reject/hardcode entrypoints coexist with low-level ancestor remnants; no supported complete25+3 author recipe was found. A25+0 configuration sensitivity is supported via model.num_refs=25 and must retain its separate name.
