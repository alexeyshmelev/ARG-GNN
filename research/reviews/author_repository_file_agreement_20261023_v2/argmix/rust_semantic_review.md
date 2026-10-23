# ARGMix Rust and dependency-lock semantic supplement

This supplement completes full-body static reading of 13 assigned Rust files and one supporting reference-selection file, plus complete structured record scans of three lockfiles. The five remaining incremental-engine/iterator files have separate ownership in rust_incremental_iterator_review.json/MD. Both binary-lifting implementations are covered here.

Source: AI-sandbox/argmix, commit c2481248eb0de5fda659d68f59c51e0e27becd20, under .runtime/author_sources/argmix_c2481248eb0d. Every code file's complete declarations, function bodies, comments and embedded test bodies were read from numbered source. Test bodies were not executed. rust_semantic_review.json records exact SHA256, byte count, line coverage, purpose, decisive lines and semantic observations for each file. Lockfiles are explicitly dependency records, not semantic scientific code. No hashes, token inventories or AST inventories are substituted for body reading.

## Extraction and callable scientific conditions

fast_lca/binary_lift.rs (311 lines) builds a target-local ancestor closure and node-major binary-lifting table, using an epoch-marked node-to-local map. It checks common roots and returns no LCA for disconnected trees. ancpath_incremental/binary_lift.rs (284 lines) implements the corresponding LCA structure with u32 parent/root/up storage. Both obtain i32 node times and clear epoch stamps on wrap. This data structure review is not evidence of measured speed or numerical parity of the larger engines.

fast_lca/engine.rs (1,235 lines) was read in full. It rebuilds the local LCA index per tree, precomputes uncapped query-reference distances, applies shared nearest-reference selection and computes reference-reference TMRCA lazily with epoch caches. Its low-level public constructor retains num_ancs and directly walks the focal parent chain at lines 932-954. It pads missing nodes with the sample node and uses population special tokens <SAMPLE>, <ANC> and <PAD>. These surviving components are different from the current archive/full pipeline's rejection of ancestor features.

The linear transform at fast_lca/engine.rs:827-835 computes min(delta as u16,max_distance). Rust integer narrowing is modulo 65,536: a delta of 65,536 becomes zero before clipping, and 65,537 becomes one. The uncapped u32 ranking path at 206-210 does not perform that narrowing, so the defect affects matrix encoding rather than that reference ranking. Default log mode follows log2(1+distance), truncation and cap and must be kept separate from this raw-linear finding. The primary ARGMix reviewer owns the exact inactive clamp-before-narrow patch. No author source is changed here.

The shared ref_selection.rs (110 lines, supporting read) ranks by distance then **input reference index**, not node ID. If fewer donors remain, it reduces the requested drop and duplicates the final selected reference to achieve the requested width. build_training.rs:466-470 obtains reference IDs from unsorted HashMap iteration. Consequently equal-distance donor identities can vary across builds even with a fixed sampling seed. Its chunk groups also iterate groups.values() at 746, so chunk order can vary before a later fixed-seed shuffle. This is a source-native reproducibility limitation. It is not permission to reorder frozen inputs silently.

The fast engine emits complete trees into buffered batches and can exceed the requested max_rows or max_count; these are not exact fixed scientific-example denominators. Eligible-tree counters, clipped-coordinate handling and missing-LCA sentinels must be interpreted through the actual caller.

## Native archive builder and truth contract

bin/build_training.rs (1,217 lines) is an executable archive builder. It accepts only the fast backend and rejects the presence of model.num_ancs at 327-330, then sets num_ancs to zero. Its current path therefore cannot produce the paper's ancestor-augmented representation without a separately named reconstruction.

The builder defaults to 15 references, log-scale TMRCA cap 14, 1,024-row chunks with stride 512, first_snp truth labels, and TRAIN reference augmentation dropping the nearest one reference while VAL defaults to no drop. It resolves log_scale_tmrca from the model block only. Previous max_*_rows config keys are rejected; max_*_subgraphs counts distinct labeled intervals while keeping query rows from accepted intervals. max_haps is None in this builder, and the sample seed controls only optional capping machinery.

All feature rows are collected before recurrent chunks are allocated. Chunks are grouped by shard and persistent haplotype identity, sorted by genomic start, and carry padding labels -100, an explicit row mask, genetic span and previous-start cM distance. The output is a stored, uncompressed ZIP64 NPZ with input_ids, tmrca, labels, row_mask, cm_features and format-v2 metadata. It is not a zero-copy or constant-memory archive pipeline.

npz_truth.rs (687 lines) keeps supervised labels outside observed extraction. It joins diploid names from the Relate sample file to two truth columns, optionally intersects truth SNPs with HAPS positions, applies class-name remapping, and returns first nonnegative SNP label or majority label. Majority ties select the lowest class index. An optional purity threshold can remove intervals. Missing HAPS positions are counted rather than rejected.

Bounds use floor(left) and ceil(right), followed by half-open SNP-rank search at 320-323. This is an explicit interval-label convention, distinct from exact focal-site truth. The code assumes sorted variant_pos without checking monotonicity here. The metadata helper at 128-161 is visible literal Python passed to python -c and loads NPZ strings with allow_pickle=True. It was not dispatched in this audit. Future use needs controlled trusted artifacts and a pinned interpreter; no environment or author module was loaded here.

## Optional merge and legacy routes

signature_cache.rs (204 lines) constructs two domain-tagged 64-bit DefaultHasher values into a 128-bit change signature. IDs-only mode omits the time matrix; Full includes it. These are noncryptographic cache/change keys rather than collision-free or cross-version stable scientific fingerprints. Legacy signatures hash population strings and the flat path hashes integer tokens, so byte interchangeability is not assumed.

flat_row_merger.rs (478 lines) checks shape, same signature, same optional label and exact-bit endpoint adjacency. An IDs-only merge can retain the first matrix while branch times change. Its drained batch has labels=None, and build_training resolves truth **after** merging. Thus merged-span first-SNP supervision is materially different from per-tree supervision. The current selected release recipe has merging disabled, as documented by the primary ARGMix reviewer; this is an optional-route limitation.

interval_emitter.rs (234 lines) is the legacy cache-based emitter. It compares signature and label but omits an adjacency check, so skipped intervals can be bridged. flush_all drains a HashMap without stable ordering. The current flat merger independently guards adjacency. Surviving low-level ancestor and legacy machinery should not be conflated with current full-pipeline callability.

## Inference, coordinate storage and completion evidence

pipeline_types.rs (402 lines) defines separate observed tensor batches, output feature plans, fp16/fp32 transfer settings, HDF5 batches, per-stage heartbeat and structured error/capped-progress states. These schemas are not measured numerical or completion evidence.

rust_torchscript_inference.rs (234 lines) loads a model and vocabulary, pads input token IDs, promotes u16 matrix values to Int64, constructs the non-padding mask, passes three model inputs and returns argmax calls. This module does not explicitly call eval or disable gradients. Whether that affects behavior depends on the separately reviewed export contract, so this supplement makes no proven dropout-defect claim. The Python-exposed loader falls back to CPU when requested CUDA is unavailable; future project wrappers must enforce the authorized explicit resource policy.

inference_worker.rs (1,398 lines) accepts logits or tuple(logits,final_layer), computes optional softmax probabilities and confidence, and passes output to TSV/HDF5. Even with fp32 host transfer, copy_tensor_to_u16_bits:351-362 converts persisted features to half-precision bit patterns. The stored probability/logit/final-layer representation therefore has fp16 precision. No output was numerically read or evaluated here.

The pinned-memory allocator at 159-164 always returns an error and falls back. The first fallback's non_blocking flag at 267 can retain the pre-fallback can_use_pinned_memory value. This is an operational qualification concern, not measured inference failure. Model counters increment before successful completion, and the nonverbose text says status=complete at 1389 even when fatal_error is present at 1393. The structured stats error_message and the writer completion result must determine success, not that text alone. The final writer-result receive has no internal timeout, so an external finite supervisor remains necessary.

hdf5_writer.rs (856 lines) preserves f64 tree endpoints, sample/reference identities and typed diagnostics, validates labels/classes/shapes, and stores features as u16 fp16 bit patterns with an explicit feature_storage attribute. Compression is none or Shuffle plus Deflate(1). Finish is required before final metadata is written. Premature channel disconnection removes that writer's new output file and returns an error. Progress written_rows includes staged rows before durable flush and is not completion evidence. Historical outputs were not changed.

tsv_writer.rs (101 lines) casts fractional f64 interval endpoints to i64 before map keying, so distinct fractional intervals can collapse and later values can overwrite prior calls. It ignores confidence and writes NA for absent node predictions. The HDF5 path retains the less lossy coordinate representation. No TSV/HDF5 artifact was generated.

## Dependency-record coverage

All 8,047 lockfile lines were parsed as complete YAML/TOML documents and every package/version/source/checksum/dependency/environment field recursively traversed:

- pixi.lock: 5,157 lines, schema 6, one default linux-64 environment, 343 exact package records comprising 306 conda and 37 PyPI packages. All 343 environment references resolve to records, and every record has SHA256. It records Python 3.12.13, PyTorch/libtorch 2.12.0, CUDA 12.9, HDF5 1.14.6, h5py 3.16.0, tskit 0.6.4, NumPy 2.5.1, Rust 1.90.0 and Transformers 4.57.6, among the complete retained records.
- ancpath/Cargo.lock: 2,261 lines, Cargo lock version 4, 241 package records, 239 registry records plus the two local crates. It pins tch/torch-sys 0.23.0, pyo3 0.28.2, tskit 0.15.0-alpha.3 and hdf5-metno 0.12.4.
- ancpath_incremental/Cargo.lock: 629 lines, version 4, 73 records, 72 registry packages plus the local incremental crate. It instead pins pyo3 0.28.3 and tskit 0.15.0, with additional transitive-version differences.

Complete dependency records and their checks are retained in the JSON. These are release dependency artifacts. They do not certify an installed environment, successful build, runtime equivalence, or the exact environment behind a paper cell. No package was installed, imported from author source, or compiled.

## Integration and inactivity

Only rust_semantic_review.json and this report were written by this supplementary worker in the ARGMix directory. The primary ARGMix worker owns final coverage/matrix integration and proposed inactive diffs. The separate iterator/incremental review supplies its own exact hashes and conclusions. No production author/project source, queue, policy, process, simulation, model, checkpoint, test, benchmark or Git state was changed or executed.
