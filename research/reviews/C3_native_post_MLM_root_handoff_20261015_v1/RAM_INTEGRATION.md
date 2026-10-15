# RAM adapter root integration

This packet is source-ready for a root successor, not launch-ready. No actual completion, checkpoint, count result, resource fit or admission is supplied. Keep every future actual ref null until observed. All writes by this task are confined to this directory.

## Implemented files and exact differences

ram_downstream_v1.py adds narrow functions taking caller-owned modules/tensors:
- embed_single_split_ram_v1: calls the pinned author's _build_arg_split_dataloader and get_embeddings. It uses batch256, workers0, no shuffle/drop_last via that native loader, last-layer head_norm, all rows and labels only for contrastive. The loop retains the same CPU tensors and even the original one-iterator random.randrange calls. It returns the complete native dictionary instead of calling torch.save. No whole-dataset author embedding function is called because that function always writes.
- row_order_arrays_v1 and concatenate_embeddings_ram_v1: same current project sidecar/tensor checks and bit-for-bit torch.cat in train,val order, returning arrays instead of writing copies.
- retrieve_ram_v1: calls unchanged native labels_from_metadata, ensure_l2_normalized and topk_indices_streaming with topk1/q_batch1024/e_chunk100000. It builds the identical k1 native return dictionary, including shape(M,1) top1 arrays. No retrieval algorithm is reimplemented.
- tensor_manifest_v1: optional bounded-chunk zero-copy raw CPU tensor hashing. Hashes describe raw C-order tensor bytes with explicit dtype/shape, not torch.save file bytes.
- native_top1_seal_ram_v1: original project seal function with only input-read substitutions, a RAM status/provenance disclosure and a rejection of fictitious disk embedding/retrieval refs. Original full-path join, checks, native-index/score preservation and one-hot math are retained.
- verify_prediction is copied verbatim from the existing project source.

seal_IO_only_source.diff is an exact diff from original native_top1_seal to the RAM function. ram_adapter_addition.diff is the exact whole-file addition. Source hashes and line locations are in source_inventory.json.

No global filesystem, Torch save/load, model, installed-file or author-source monkeypatch is present. Retention1 is separately documented and is not implemented by the RAM module. The generic checkpoint-selection bridge remains post_mlm_bridge_v1.py.

## Actual root dependencies before using these functions

1. Genuine published v12 completion plus root/native waits0/drain, actual epoch/update/Adam counter evidence and actual regular checkpoint ref.
2. Genuine admitted generic six-dataset counts and original closed v3 preparation/config refs. Actual contrastive/train rows and U stay unresolved in this source packet.
3. Genuine contrastive-only fit with unchanged science and separately disclosed retention1; actual final regular checkpoint at epoch1/exact U and wait/drain.
4. Fresh current980 reservation/source closure for a single native RAM postprocessing child, actual CPU RAM/cache/TMP/checkpoint allowances, current GPU4/5 lease/title/device validation, runtime/author import-origin validation and the existing abstract-AF_UNIX bootstrap before imports. Preserve genuine direct waits and owned drain. Nothing here uses v12 amendment authority for a new allocation.
5. Bind exact adapter plus original project/author source hashes in the root transitive closure. Function injection is for code reuse/testing and conveys no authority. Actual execution must supply the pinned author modules after existing origin/runtime checks, not test doubles.

The old generic root/consumer/child pins and entrypoint still need the current-policy successor described in HANDOFF.md. This module is not a bypass for those checks. No runnable root launch command is claimed.

## Single-child call sequence

After root context/lease/runtime and immutable input validation, import the pinned author embeddings and retrieve modules, Torch and NumPy, and this source-bound module. Load registry through the existing project registry function and each full-path sidecar through existing row_order or the same bounded no-pickle NPZ reader. Root verifies preparation_v1/counts_v1 and exact10000 query registry first.

For view in (query_paths,contrastive), and split in (train,val):
- Derive the same per-view I/O config using the existing _io_config embedding branch, with genuine final contrastive checkpoint, local base config and no training initializer.
- Write only the small resolved config into its admitted child scope. Call author load_checkpoint_model(config_path,checkpoint).
- Call embed_single_split_ram_v1(torch=torch,embeddings_module=author_embeddings,model=model,cfg=loaded_cfg,view=view,split=split,expected_rows=counts[datasets][view/split][rows],fast_guard=root_fast_guard).
- Retain that dictionary and its original row-order sidecar. Delete the model only after that split is fully embedded; retain all produced arrays.
- Call concatenate_embeddings_ram_v1(torch=torch,view=view,parts={train:...,val:...},orders={train:...,val:...},counts=counts,parent_id=parent_id,fast_guard=root_fast_guard). Retain combined dictionaries and original sidecar refs. Release split dictionaries after successful validation/concat to reduce CPU RAM. This is deletion of RAM objects, not project files.

Then native = retrieve_ram_v1(retrieve_module=author_retrieve,query=combined_query,corpus=combined_reference,device=the_already_validated_CUDA_device,fast_guard=root_fast_guard). The adapter does not choose a device or silently initialize one. CPU is used only in the synthetic test, not a production fallback.

Optionally compute tensor_manifest_v1(torch=torch,values={query_paths:combined_query,contrastive:combined_reference,retrieval:native},fast_guard=root_fast_guard) and save its compact JSON with actual checkpoint/config/counts/source/sidecar refs through the existing qualified writer. All hashes come from actual current RAM contents; no source task has supplied them.

For native_top1_seal_ram_v1, use the existing evaluation config values with native_query_embeddings_ref=None, native_corpus_embeddings_ref=None and native_retrieval_ref=None. Set external_truth_ref=None/prediction_ref=None, expected_splits=[train,val], expected_pooling=head_norm, actual preparation/focal refs, and native_RAM_tensor_manifest_ref to the actual compact manifest ref if written. Pass actual prep, original split orders dictionary, query/corpus RAM dictionaries, native return dictionary, Torch and registry. Writer interface is the same small json(name,value)/npz(name,**arrays) interface as the existing project seal; root supplies its qualified admitted writer/adapter.

The result status is sealed_native_C3_top1_RAM_IO_v1. predictions.npz and native_top1_preserved.npz preserve the existing schema and downstream external evaluator semantics. Completion explicitly reports full_native_arrays_in_CPU_RAM_v1 and embedding_file_artifacts_written=False, with None for nonexistent embedding/retrieval file refs. It must not be relabeled as a generic file-producing embedding/retrieval phase. Root records actual child/launcher completion/waits/drain afterward; the existing external scorer uses the genuine prediction_ref.

## Capacity and evidence

The RAM route removes49.274585872GB of four split embedding output tensors and their duplicate49.274585872GB concatenated disk copy. It retains the same arrays in CPU RAM. Concat can temporarily double view arrays; native normalization adds reference vector output (~29.44GB), and raw dataset/model/optimizer/labels/cache/workspace need separate monitored allowances. This is a source plan, not an observed peak-RAM or native full4096-fit qualification. No float downcast, token omission, sampling, compressed numerical replacement or smaller training batch is allowed.

Only prediction and preserved-top1 numeric payloads total640,000bytes for10000 rows, plus NPZ strings/metadata/provenance. No huge embeddings/retrieval .pt files are written by the adapter. Positive disk caps still cover caches/TMP/config/logs and retained checkpoints.

check_ram_synthetic_v1.py passed in the shared project interpreter with CUDA_VISIBLE_DEVICES empty. It used10000 synthetic CPU query rows,3 synthetic references, four native-shaped loaders and42 batches, including final partial batches. The actual author's pure retrieval helper ASTs were compiled in an isolated namespace and run on those CPU tensors with unchanged topk1/q_batch1024/e_chunk100000.

The original project seal was executed with isolated injected in-memory readers, not global filesystem/Torch monkeypatches. Its predictions and preserved raw top1 outputs matched the RAM seal exactly under a permuted focal registry and repeated sample-node IDs across unique paths. Negative checks reject query-label leakage, duplicate full-path IDs, out-of-range top1 indices and int64-to-int32 token downcast. Raw tensor hash matched a separate byte conversion. CUDA remained uninitialized; no real model/corpus/checkpoint/truth was opened and no admission/accounting/Git operation occurred. This verifies the narrow I/O adaptation, not native science/accuracy/resource fit. Receipt: RAM_synthetic_checks.json.

To rerun this exact targeted check after a source change:
source scripts/project_env.sh
CUDA_VISIBLE_DEVICES="" "$ARGGNN_PYTHON" -B research/reviews/C3_native_post_MLM_root_handoff_20261015_v1/check_ram_synthetic_v1.py
