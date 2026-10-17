# Native saved-MLM downstream storage bound

The exact persisted embedding tensor storage is **49,979,548,096 bytes (46.5470814109GiB)**. This follows from existing authenticated count metadata and unchanged source shapes/dtypes, with no checkpoint, pickle corpus, tensor payload or truth opened. The current64GiB whole cap cannot yet be replaced by a certified smaller whole cap from this evidence: runtime AUX/cache and physical allocation remain additional terms. No cap or sealed source was changed.

## Exact dominant arrays

There are9,721,236reference rows and10,000query rows, persisted once across the original4single-split embedding files. Every row contains a768-wide FP32 embedding,128-wide int64 input_ids,128-wide int64 attention_masks and one int64 extant_node_id. Only reference rows contain one int64 label. token_type_ids are created in batches but are not retained in output_data. Short raw paths still pad to128; their maximum29nodes does not reduce saved matrix widths.

| Array across all four files | Bytes |
|---|---:|
| embeddings | 29,894,356,992 |
| input_ids | 9,964,785,664 |
| attention_masks | 9,964,785,664 |
| extant_node_ids | 77,849,888 |
| reference labels | 77,769,888 |
| **Total tensor storages** | **49,979,548,096** |

The largest contrastive/train file contributes44,932,188,144tensor bytes (41.846361GiB); contrastive/val4,996,079,952 (4.652962GiB); query/train46,167,384; query/val5,112,616. The source concatenates each key into fresh complete tensors before torch.save, so batch-list retention in RAM is not another disk copy. Original train+val combination remains RAM-only, including row sidecars. Its full49.98GB does not get persisted again.

## Checkpoints and retrieval

The1epoch/2135update contrastive run preserves106interval events and one final event. Installed Composer writes uncompressed .pt directly and only then rotates the older newly generated checkpoint; keep1 therefore gives a two-file transient output pair. The original phase RLIMIT_FSIZE and accounting checks enforce3GiB/file, giving **6GiB** for that pair. Post-drain latest normalization may retain a final+latest regular pair under the same two-file policy. The107events are not107simultaneously retained files.

Embedding tensor bytes plus that enforced checkpoint pair are **56,421,999,040 bytes (52.5470814109GiB)** before serialization, metadata and AUX/cache. This uses existing checkpoint limits, not a newly inferred model-state size and not a newly proposed cap.

Retrieval persists10000×1 int64 indices and FP32scores,120,000tensor bytes, and also the original9,721,236known-reference labels as single-character Python strings. The installed Torch default is pickle protocol2. Conservatively,11bytes/string plus2bytes per1000-element list batch and64KiB framing gives at most107,018,576bytes for that list. Two small synthetic RAM-only pickle checks confirm that opcode envelope. Actual returned top1 entries alias the same original score/index tensors at k1; there is no large similarity matrix written. The existing retrieval file limit is256MiB and its whole retrieval/seal group remains1GiB. Two prespecified10000row prediction/index NPZ files and compact JSON add further small terms; no scientific values or formats were changed.

## Why this does not prove a smaller whole reservation

The unchanged accounting measures max(st_size,st_blocks×512)+4096 for every regular file, directories4096, across original output AND all redirected TMP/cache/log/AUX. Raw tensor nbytes is a logical payload proof, not an NFS allocation or runtime-file envelope.

The existing per-group limits are8GiBcontrastive,56GiBembedding and1GiBretrieval/seal, subject to64GiBwhole. Embedding AUX is not separately capped below its share of56GiB. CUDA_CACHE_MAXSIZE256MiB does not bound TORCH_EXTENSIONS_DIR, TRITON_CACHE_DIR, NUMBA_CACHE_DIR, HF/TMP or the other redirected runtime paths. The phase source supplies no tighter combined successful-run bound for those paths. Even known scientific tensor bytes need ZIP/pickle headers and observed allocation beyond logical bytes.

A future whole-bound proof must establish the residual embedding-runtime/allocation envelope, authenticate its source/policy, and receive the normal successor review before ROOT changes a reservation. Existing source-only proof certifies64GiB as the current enforced whole envelope; it does not show64GiB is scientifically inherent. A smaller payload estimate must not be presented as a smaller charged whole bound.

If the stated909.884GBprojection excludes all pending13GBinference+14GBsimulation+64MiBqualification, then even the52.547GiB tensor-plus-checkpoint subtotal alone would project approximately993.373GB before native AUX/metadata, above980GB. Thus reducing native64 alone cannot support all of that simultaneous pending work. This arithmetic is conditional on ROOT's stated projection membership; no current accounting or capacity was queried.

## Source locations

- Original embedding save/get_embeddings/dataloader: author embeddings.py lines181–237 and302–409; ARGDataset padding in src/arg_data.py lines268–293.
- Exact dtype/shape enforcement and original disk combination reversal: src/arggnn/american_c3_native_concat_v1.py concatenate_embeddings_v1; operational RAMCombinationWriter in efficient_downstream_v1.py.
- Original retrieval output and label-string construction: author retrieve.py lines19–34 and153–168; installed Torch serialization.py DEFAULT_PROTOCOL2.
- Checkpoint save/rotate: installed Composer callbacks/checkpoint_saver.py lines461–621; utils/checkpoint.py lines1252–1274.
- Existing native worker limits: native64/worker_context_v1.py lines340–354; whole/group accounting native64/contract_v1.py lines198–215; underlying allocation accounting common_v1.py lines249–280.

All exact source/count references and arithmetic are sealed in calculation.v1.json. Actual guards/runtime/GPU/process probes, data/checkpoint/truth reads/hashes, admissions/launches/closes and cap mutations: zero.
