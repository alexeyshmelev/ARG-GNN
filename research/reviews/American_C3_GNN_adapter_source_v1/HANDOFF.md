# Fresh American C3 GNN source handoff

Four new modules implement generic class/sample observations, bounded whole-tree CPU blocks, a freshly initialized C3 PyG model, exact two-pass genomic inference and short-parent supervised training APIs. Original H038/H039 modules and checkpoints are unchanged. This source task performed no biological data reads, simulations, training, VAL truth access, GPU work, installs or resource admissions.

The one root-authorized handmade FP32 CPU qualifier passed with exit 0. Its actual argv, environment, source hashes and compact result are in handmade_CPU_qualifier_receipt.json. It used three handmade trees, 24 nodes including unary vertices, six haplotypes, two sites and an empty hold. No ancestry targets were used. C3 masks worked, and all 38 learned parameter tensors had finite full-parent gradients. Ordinary bidirectional GRU versus the two whole-direction scans differed by 0. Re-encoding separate whole-tree blocks differed by 1.19e-7. Two- and three-block final logits differed by 1.49e-8 at tolerance 1e-5. This is an implementation check, not model accuracy or scale qualification.

## Source and first executable stage

- american_c3_gnn_observations_v1.py: original whole-tree layout and five-field mutation-concordance equations, generic class frequencies and reference masks.
- american_c3_gnn_dataset_v1.py: guarded observation-only block plan CLI, spawn CPU Dataset/DataLoader and PyG block objects.
- american_c3_gnn_model_v1.py: new parameters throughout, generic class-dependent inputs and fresh C3 head, unchanged gated PyG cavity messages.
- american_c3_gnn_run_v1.py: root-launcher APIs for two-pass focal inference, complete short-parent TRAIN steps and strict fresh-family checkpoints.

Actual metadata-only CLI was verified under existing Torch 2.6.0+cu124 and PyG 2.7.0:

    CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .runtime/env/bin/python -B -m arggnn.american_c3_gnn_dataset_v1 --runtime-report

No flags print inactive defaults and exact source closure. To materialize the small block index, root binds actual observation_ref, native_preparation_ref and focal_registry_ref in an activated config, then supplies its existing capacity-v2 admission:

    CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .runtime/env/bin/python -B -m arggnn.american_c3_gnn_dataset_v1 --execute --admission "$C3_GNN_PLAN_ADMISSION" --expected-admission-sha256 "$C3_GNN_PLAN_ADMISSION_SHA"

Common C3 admission rules and all existing resource guards apply. The CLI writes block_index.json and completed.json only. It does not cache feature arrays or launch a model.

The native preparation must bind exactly the observation and focals used by this GNN. Only sanitized inferred-tsinfer/tsdate genealogy is accepted. A simulated-true ORACLE tree is rejected. Node population, migration, individual, provenance and metadata annotations must be absent. Sample columns, inferred sample nodes, dates and original sample indices must join exactly. Focals remain the exact native full-path IDs and coordinates.

## Class and sample transfer

Class order is EUR, EAS, AFR in the supplied inactive C3 configs. The model and frequency functions accept generic class/sample cardinality. Constructor expected_haplotypes can be fixed or None, but a change in training/evaluation sample count is a separate transfer condition that root must record. The same-cohort draft fixes 1600 haplotypes and source counts 200/200/200.

Two age cohorts preserve the original younger/older grouping rule, now with 2C donor channels. For C3 this is six donor channels and 15 site features instead of the European model's eight/19. Leaf side input has C+2 fields: three known-class indicators, unknown-label flag and sampling age. All current American samples are present-day, so older donor channels remain explicitly masked. The original alpha 0.5, cohort-common callability, panel frequencies, mean/max/count/empty pooling, branch support and node/edge geometry formulas are preserved.

input_reference_class_mask and input_reference_sample_indices can hide observed reference labels without changing actual query roles. Absent donor classes/cohorts remain masked. Hidden reference labels never become scored query identities. At least one known reference is required. Model input reference masks are explicit config fields; actual query ancestry is absent.

Every parameter initializes afresh. C4 head slicing, class relabeling, permissive state loading and European checkpoint initialization have no supported path. Only the new checkpoint schema with exact constructor contract loads strictly. This model retains the original direct GRU classifier; it does not introduce an HMM or new temporal transition estimator.

## Bounded whole-tree loading

block_plan_30mb_inactive.json defaults to at most 32 trees, 250000 native nodes and 4096 sites per block. A plan greedily packs complete chronological trees under these limits. If one complete tree exceeds a cap, planning rejects instead of pruning it. Every active marginal node, original sample, unary branch, site and interval survives.

Workers load a verified tree sequence once per process and memory-map the original genotype matrix. Only the current complete block is converted to tensors. Two spawn workers, prefetch factor 1 and batch_size None avoid automatic graph concatenation/reindexing. Workers have one CPU thread and empty CUDA visibility. Project-local scratch paths and abstract Linux IPC sockets preserve the existing worker conventions. GPU transfer happens only in the root-launched main process, using pinned tensors and nonblocking copies.

Plan byte estimates cover input tensors, not peak activations, native tree-library memory or GPU training graphs. Each worker retains its own tree-sequence copy. Actual counts and root memory/lease qualification are still required. The source contains no claim that proposed caps fit a particular GPU reservation.

## Exact 30 Mb genomic context

The original one-layer bidirectional GRU is represented by independently named forward and reverse GRUs with the same equations, jointly trained from fresh initialization. Exact inference uses complete-state carry, not a finite halo:

1. The forward loader visits every complete tree block in order. Tree/site contexts are encoded and the forward hidden state carries across block boundaries. Only predeclared focal forward vectors are retained on CPU.
2. Forward workers drain before reverse workers start. The reverse loader visits the same blocks in reverse order, re-encodes the same whole-tree contexts in eval mode and carries the reverse hidden state.
3. Each focal combines its local context, saved forward state and reverse state through the fresh C3 classifier. The output schema exactly matches american_c3_evaluate_v1.

Global previous/next interval spans are used at every block boundary, including blocks without focal records or sites. GRU state resets only at true chromosome endpoints. A finite halo is not exact for an unrestricted GRU. Two passes avoid a full-chromosome GPU graph/history or disk context spool, at the cost of encoding trees twice. Host focal state is Nfocal * hidden_dim * 4 bytes, 1.28 MB for 10000 focals at width 32, plus final posteriors and identities. The recurrence is mathematically the full bidirectional recurrence, with ordinary FP32 kernel roundoff permitted.

Root-launcher integration:

    forward_dataset = AmericanC3WholeTreeDatasetV1(plan_ref, direction='forward', worker_tmp_directory=owned_scratch)
    reverse_dataset = AmericanC3WholeTreeDatasetV1(plan_ref, direction='reverse', worker_tmp_directory=owned_scratch)
    forward_loader = make_dataloader(forward_dataset, num_workers=2, pin_memory=True)
    reverse_loader = make_dataloader(reverse_dataset, num_workers=2, pin_memory=True)
    model = AmericanC3WholeTreeGNNV1(**registered_model_kwargs).to(logical_device)
    prediction, receipt = streamed_focal_predictions(model, forward_loader, reverse_loader,
        device=logical_device, checkpoint_ref=exact_fresh_C3_checkpoint_ref)

Use these APIs only inside root's admitted GPU launcher after its required process title and lease setup. Do not pre-start the reverse iterator. The API strictly loads the bound new-family checkpoint, requires a parent independent of its trained parent IDs, and drains both loaders on success/failure. It returns arrays and metadata to the root writer; it creates no unguarded GPU output file. Root must bind actual launcher/model/helper/checkpoint/runtime/plan sources, record the checkpoint ref and save through its admitted output writer. Private truth is never opened.

## Defensible training condition

The provided condition is fresh_C3_supervised_short_independent_complete_parents. short_train_condition_inactive.json proposes independent 100 kb TRAIN simulations with the same 1600 haplotypes, while evaluation uses 30 Mb. These are source-only proposals with distinct seeds, not generated parents or admitted jobs.

train_complete_parent_step accepts separate TRAIN targets aligned to query focal rows. It requires a registered TRAIN parent, an explicit length bound below 30 Mb and one complete parent fitting a single admitted block. The full recurrent graph receives gradients. Multi-block truncated BPTT or detached-state training is rejected, not substituted. Optimizer/schedule and actual training registry must be registered by root.

block_plan_short_train_inactive.json offers larger source limits solely to allow a short complete parent to fit if actual measured counts and resources permit. Those limits have not been capacity-qualified. If a parent exceeds them or available GPU memory, root must register a different short-length/cohort condition, or a separately justified training method. The current implementation does not claim exact bounded-memory full-30Mb training.

This supervised short-parent training differs from the native paper's MLM plus contrastive full-path training. Report that condition explicitly. Any change in sample count/reference mask also belongs in the transfer condition. Checkpoint payloads record fresh initialization, class/cardinality contract, model seed, trained parent IDs and actual short training lengths.

All new scientific execution and final common scoring remain root work. No GNN accuracy, full-cohort throughput or 30 Mb memory result is asserted here.
