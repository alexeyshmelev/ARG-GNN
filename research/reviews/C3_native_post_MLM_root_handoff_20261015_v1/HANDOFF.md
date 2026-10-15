# Source update: RAM adapter implemented

See RAM_INTEGRATION.md for the completed adapter, exact source diffs and synthetic verification. Current980 root integration and all genuine future actual refs remain unresolved. The original scientific/settings discussion below remains applicable.

# Native ARGformer after v12 MLM: source handoff

State: source preparation only. Root reports v12 launched in session48073, PID2943908, GPU5, with optimizer steps not yet verified. This task did not independently inspect that process or infer completion. Genuine success, checkpoint, downstream counts and new admissions remain unresolved. TRAIN000 remains engineering TRAIN. No corpus/checkpoint/truth payload, native/model execution, admission, guard/accounting mutation or Git operation occurred here.

## Existing narrow MLM bridge

post_mlm_bridge_v1.py supplies genuine-v12 adoption plus a contrastive-only integration hook. It has no CLI, launcher or authority. v12_mlm_phase_v1(actual_parent_completion_ref, actual_root_wait_ref, spec) requires the published v12 parent and namespace completions, exact author/corpus/producer lineage, launcher wait0/owned drain, native wait0/drain, epoch5/update224775 Trainer/live/saved Adam evidence, original corpus-authentication metadata and the actual regular checkpoint ref. Adoption opens bounded JSON only.

run_contrastive_after_v12_v1(record,spec,cfg,ctx,lease,run_child=...,mlm_completion_ref=...,mlm_root_wait_ref=...) submits exactly one existing-contract packet: name=contrastive, stage=contrastive, initializer_checkpoint_ref=<genuine v12 checkpoint>. After actual contrastive completion/wait0 it writes the generic completed_C3_fresh_native_fit envelope using the two actual phase refs. The existing fit_completion_v1 checks the necessary MLM child fields rather than its schema, so it accepts the original v12 child directly. No MLM rerun or invented compatibility child is needed.

There is no launch-ready root command yet. The old generic fit entry always reruns MLM; its root contract pins retired completed-credit-v3 admissions/launcher and v2 corpus. The actual corpus is v3. Generic consumer/child canonical main-source and launcher checks also need a narrow root successor, so substituting an environment policy is insufficient. Preserve frozen sources. Bind the bridge, adapted root entry/child/context, actual callback, all transitive sources, runtime and actual metadata into new current980 reservations.

Reuse prospective_980GB_capacity_source_v1/prospective_980gb_v1.py, activation_policy.root_active_v1.json and scripts/launch_reserved_gpu_authorized_GPU45_980_v1.py with actual marker/repair authority, original mutex, leases/title/runtime/device checks and GPUs4/5 only. Carry forward the reviewed pidfd owned-lifecycle and native-interpreter abstract-AF_UNIX helper for unchanged fork workers. The old generic supervisor uses numeric process-group cleanup and its child lacks that IPC bootstrap. V12 supervisor/launcher are MLM/amendment-specific and cannot authorize new downstream phases. root_launch_command_argv remains null.

## Exact scientific route and actual-count dependency

After actual published MLM success and both genuine wait/drain levels, take the final regular checkpoint ref from the child completion; never guess a filename, follow latest or select by validation metric. Root handles completion/close accounting.

Before contrastive admission, execute the existing admitted generic CPU count_preparation_v1 against the actual immutable v3 preparation/config. It authenticates all six raw datasets, full-path sidecars, both reference labels and the complete10000-ID join. V12 metadata-derived MLM counts do not replace that generic receipt. Exact contrastive/train rows and U=floor(N_train/4096) remain null here. Total reference rows across splits are9,583,977, but the global native shuffle/split means N_contrastive_train cannot be inferred as floor(0.9*reference_total).

Call sequence_contrastive.train(cfg), not its raw CLI, which attempts missing yamls/defaults.yaml. Load the pinned contrastive.yaml; adapt I/O paths, offline base config, device, disabled external logging, fresh load_path=None/autoresume false, genuine MLM initializer, and the retention-only change justified below.

Scientific settings: one epoch, seed36, FP32, world_size1, global/device/microbatch4096 for both train and eval; max length128; hidden768,12 layers/12 heads FlexBERT, trained pooling head, normalized dot similarity, temperature0.30. Train shuffle/drop_last true; eval false/false; workers4; DistributedSamplerPCG64DXSM default shuffle seed9176. Preserve full4096 because smaller microbatches alter the in-batch objective.

The executed supervised loss uses all same-class non-self rows as positives, a log-softmax denominator over all non-self in-batch rows, positive average per anchor, then mean over anchors with positives. Pair/hard-negative helper datasets are unused; YAML hard-negative count, negative-sampling temperature and symmetric field do not alter this executed path. Balanced sampling stays false. Keep the current uniform class-weight fallback: build_contrastive_model receives cfg.model, whose absent nested train_loader causes the optional inverse-frequency weighting to fall back. Do not inject that plumbing.

Fresh DecoupledAdamW lr2e-5, betas0.9/0.999, eps1e-6, weight_decay1e-6; linear scheduler warmup0.1dur/alpha_f0.02. Eval every20 batches, subset200. Save every20 batches and at final completion; select regular final epoch1/exact U, not best eval. The generic contrastive child checks final Composer timestamp plus model/optimizer presence; it does not independently inspect all live Adam counters as v12 MLM does. No stronger evidence was observed here.

## Checkpoint retention: source-proven I/O-only cap

The release contrastive YAML keeps all(-1). A fresh operational successor can set save_num_checkpoints_to_keep=1, preserving save_interval20ba, final-save schedule, full model/optimizer state, fresh/no-resume behavior and actual final-regular-checkpoint selection. This matches native MLM YAML retention1 and its cap2 regular-file planning, rather than setting retention2.

Pinned Composer evidence:
- trainer.py passes this field only as CheckpointSaver.num_checkpoints_to_keep.
- checkpoint_saver.py:475 saves the full checkpoint; :502-515 repoints latest to the new regular file; :607-618 appends and removes the oldest saved regular file if over retention.
- Rotation only edits its saved-path list and removes an old file. It does not update model, optimizer, scheduler, sampler or RNG. The callback state_dict records all_saved_checkpoints_to_timestamp, which rotation does not prune.
- create_interval_scheduler in utils/misc.py defaults include_end_of_training=True(:57-60) and triggers the final batch/epoch event(:120-121). That is unchanged.
- Author MaxMetricTracker only logs maximum metrics, never reloads/selects an earlier checkpoint. No adaptive early stopping or best-checkpoint selection exists in train().

With retention1, two regular checkpoint files may coexist during save-before-rotate, then one remains. Plan any postdrain latest-normalization copy separately. max_checkpoint_files=2 is therefore meaningful; save_num_checkpoints_to_keep=2 would instead allow three transient regular files. Failed-file preservation still applies. This is an explicitly disclosed release storage-retention change, not a claim that every YAML value is untouched or that wall-clock/bitwise execution is identical.

## Preferred RAM downstream route: no duplicate embedding files

The original disk route writes four split embedding dictionaries and then complete concat copies. Their raw tensor data total49,274,585,872bytes, and retained originals+concat98,549,171,744bytes, excluding file overhead, sidecars, checkpoints, cache/TMP/logs.

A faithful I/O-only route can fuse embedding, native retrieval and prediction seal into one admitted native child, keeping all complete native arrays in CPU RAM until sealing and writing only compact predictions, original top1 indices/scores and provenance. Separate child processes cannot share ordinary RAM without a new transport; keep this one supervised child and preserve its actual wait/drain.

Existing author return APIs:
- load_checkpoint_model(config_path, final_contrastive_checkpoint) returns model,cfg.
- _build_arg_split_dataloader returns a plain unmasked, unshuffled loader with workers0, no drop_last and optional split labels.
- get_embeddings(model,batch,layer_index=None,pooling_strategy='head_norm') returns native batch vectors.
- embed_arg_dataset always torch.save's a full dictionary and returns only vectors. It cannot directly supply all native arrays in RAM. No global torch.save/torch.load/filesystem monkeypatch is justified.
- retrieve.ensure_l2_normalized returns native F.normalize tensors; retrieve.topk_indices_streaming(Q,E,device,k=1,q_batch=1024,e_chunk=100000) returns original best_scores,best_idx. labels_from_metadata produces the native corpus label list.

The narrow adapter is now implemented in ram_downstream_v1.py. See RAM_INTEGRATION.md for exact root calls and RAM_synthetic_checks.json for targeted CPU synthetic parity. The following source plan is implemented:
1. For each exact single split in query_paths/train,val and contrastive/train,val, use the existing native loader and get_embeddings loop with batch256, max_samples0 (all), last layer/head_norm, is_classification true and labels only on contrastive. Retain the full same CPU arrays: embeddings, input_ids, attention_masks, extant_node_ids, labels where present, and identical metadata. Preserve original dtype and partial batch. The single-iterator random.randrange(1) calls can be retained exactly; they never reorder rows. No sample cap or numerical conversion.
2. Apply the existing concat validation and torch.cat(train,val,dim=0) to every tensor key in RAM. Metadata splits=[train,val], complete row counts, same class/sample/record sidecars. This can temporarily duplicate RAM; it needs a monitored RAM cap, not98.55GB of disk. A later preallocation optimization is unnecessary for the faithful first adapter.
3. Call unchanged native ensure_l2_normalized, topk_indices_streaming and labels_from_metadata on these combined dictionaries. Build the same native retrieval dictionary, including native k1 top1 shape, indices, scores and corpus label list; omit query_labels.
4. Introduce only an in-memory input variant of the project native_top1_seal/concat readers. Preserve every validation, exact full-path identity join, class mapping and one-hot call operation from existing source. Native retrieval/classification math must remain unchanged. No smoothing, PCA/ANN replacement, custom ties or calibration.
5. Preserve predictions.npz and native_top1_preserved.npz using the current compact writer after exact validation. Receipt binds actual final checkpoint/config, source manifests, complete counts, sidecar refs, shape/dtype metadata and hashes of actual RAM array bytes if root wants authenticated replay provenance. RAM hashes are not torch.save file hashes and must be labeled accordingly. Actual wait/drain remains outside the child. Arrays can be released only after sealing.

RAM planning must include original/combined arrays, native full-corpus normalization output (~29.44GB for reference vectors alone), model/checkpoint state, full raw loader materialization, labels, score workspace and temporary torch.cat copies. User authorizes CPU RAM, but the root cap remains an actual monitored operational constraint. No fit/RAM sufficiency claim is made here. Compact prediction+preserved-top1 array payload is 640,000bytes before NPZ/metadata overhead, with no huge embedding-file copies. Cache/TMP/checkpoint growth still needs fresh positive current980 caps.

## Exactly10000 common focal semantics and disclosures

Full observed leaf-to-root paths are enumerated in native sample/tree order, deduplicated by complete tuple before native128-token truncation, and use one per-ARG observed vocabulary plus seed42 90/10 split. MLM uses all25,574,587 unique paths. Contrastive uses known reference labels only. Exactly10000 unique admixed full paths are selected without truth, without replacement, independent seed92001. Support is one frozen unique-path focal record, which may repeat an original sample across different paths.

Each record carries full-path record ID, observed sample node and original sample index. Coordinate is the midpoint of its first observed tree interval, an explicit reconstruction assumption. Seal joins record IDs, confirms both sample fields, restores frozen focal order and maps the retrieved known reference class into float32 one-hot [EUR,EAS,AFR]. Original native scores/indices are retained. Token rows, embedding values or extant-node ID alone never identify paths. Truth is opened only by a later external scorer after sealing.

This is the pinned native release retrieval route on the current reconstruction, not exact paper Table1 reproduction. Original Table1 PCA/cluster downstream code is unavailable; normalized top1 reference retrieval is a disclosed different rule. Release temperature0.30 differs from paper default0.05. Current data use1600 haploid samples, tsinfer0.5.1/tsdate0.2.7 inferred trees with explicit unary collapse and the focal-coordinate assumption above. No accuracy, heldout score, optimizer progress or native-fit result was produced here.
