# TEST evaluation adapter readiness: metadata-only assessment

The original European AP benchmark has five held-out parents, `ancestralpaths_european_test_000` through `_004`, with exactly 10,000 canonical targets per parent (2,000 in each of five query cohorts), for 50,000 total. Native class order remains `Ana_nonBAA, CHG_nonBAA, WHG, EHG, Ana_viaBAA, CHG_viaBAA`. Frozen sampling plan: `configs/benchmarks/recipes/ap_paper_sampled_plan_v1.json`, SHA256 `2d39761d0cc030af72ba1e313be63dfce3fa7e5a4e1f4cecbdf33a5109fc7f5b`.

This is an assessment, not permission or a launched evaluation. Root's full TRAIN benchmark fit is in progress; TEST work remains a later stage after predictor freeze. No frozen source or prepared TRAIN-cache file was changed.

## What was checked, and what remains unknown

All five original generation, inferred-observation and canonical-registry tasks are marked succeeded. Their small result envelopes were authenticated. File-stat checks confirm all five post-size native trees, observed VCFs, registry files, inference metadata files and panel files exist. Each result envelope declares 10,000 canonical examples. Native tree files are approximately 8.79-8.82 GB each and VCFs 2.60 GB each; these are existing sources and need no duplicate copy. Inference metadata offsets are 342, 47, 105, 88 and 12 bp for TEST000-004.

Access was limited to source code, sampling/workflow plans, controller task metadata, authenticated small result envelopes, one bounded 16 KiB inference metadata prefix per parent, and file existence/size checks. No TEST registry rows, panel rows, tree payloads, VCF/genotype payloads, private target-schema contents, ancestry truth, predictions or scores were opened. No TEST data were exported or prepared, and no model was selected on TEST. Large source payload hashes and actual TEST tree-reader/graph-shape compatibility were therefore not revalidated. The exact paths, historical hashes, envelope hashes, source-code hashes and stat evidence are in `TEST_EVALUATION_READINESS.json`.

## Exact reusable code and required adapters

| Existing code | Reuse and boundary |
|---|---|
| `src/arggnn/benchmarking/european_sampling.py:25` (`paper_sampling_plan`) | Exact parent roster, quotas and native six-class endpoint. Do not regenerate or alter the original frozen registry. |
| `european_sampling.py:187` (`registry_arrays`) | Canonical `example_index` order and actual sample/position keys for prediction binding. Apply later to authenticated original TEST registries. |
| `research/redesign_20261028/data/window_data.py` (`ReplacementTreeReader`, `SortedVCF`, `small_json`, `exact_vcf_position`, `uniform_coordinate_context`) | Reuse native complete-tree and observed-SNP reads, metadata authentication and the existing context generator. Retain fail-closed native topology validation and all SNPs in selected complete intervals. |
| `research/joint_model_20261029/data/observations.py` | Reuse its scientific observation contract, namespace identities and complete-tree extraction logic in a separate versioned evaluation adapter. Its current constructor intentionally permits only TRAIN000-002 and split `train`; do not bypass or weaken that frozen guard. |
| `data/register_site_plan.py` | Reuse canonical grouping by native site and one context per site. A later TEST plan must freeze its own parent/RNG namespace (for example disjoint ordinals 3-7, explicitly registered), keeping K=5, four IID coordinates, 50 kb radius, source masses and full-interval SNP rules unchanged. No tuning or repeated contexts selected using TEST outcomes. |
| `model/joint_model_v1.py` (`prepare_observations`, `prepared_to_state`, `prepared_from_state`, `collate_prepared`, `JointAncestryModel`) | Reuse unchanged after full predictor freeze. Pure observation input and compact focal routing are already suitable. Use the qualified focal-only view only if the frozen selected arm is `no_cross`. |
| `data/site_dataset.py` | Reuse authentication, immutable worker-cache and collation patterns in a new prediction-only loader. The existing class hardcodes the 1,245-site TRAIN manifest and split support and requires labels; it must remain unchanged. Prediction items carry observations, site keys and canonical IDs, without labels or training loss scaling. |
| `data/loader_shutdown.py` | Reuse the now CPU- and GPU-qualified worker resource-sharer cleanup and explicit owned-loader shutdown. |
| `root/train.py:evaluate` | Reuse the identity/finite-output validation idea, not the function unchanged: it requires labels and computes cross entropy during prediction. A separate prediction function must never load labels. |
| `european_sampling.py:140` (`native_truth_labels`) | Reuse only in the external scoring phase after required sealed-prediction barriers. It joins original-position SNPs through semantic `(sample_id, haplotype)` identities to unchanged native six-path truth and returns labels indexed by canonical example ID. |
| `src/arggnn/benchmarking/european_native_models.py:367` (`_native_metrics`) | Exact native endpoint metrics: accuracy, macro-F1, per-class precision/recall/F1/counts, confusion, raw NLL and Brier. Zero true-class probability produces infinite-NLL status; no probability clipping or hidden renormalization. |
| `src/arggnn/benchmarking/metrics.py` (`Prediction`, `score_parent`, `summarize_parents`) | Generic method identity and frozen support-fingerprint checks; equal independent-parent mean and sample SD (`ddof=1`). Query identity includes haplotype as well as sample ID. |

`data/targets.py` is TRAIN-only and must not be generalized into a TEST-prediction label loader. Keep TEST truth references in the separate external scoring namespace shown in the assessment JSON.

The original `european_native_models.score_native_comparison` at line 398 hardcodes methods `ancestralpaths` and `treeloom`, requires every method on all five parents before reading any private target metadata, and restores exact canonical `example_indices == arange(10000)`. The executable frozen workflow additionally routes its score stage through `european_argmix_workflow.execute_stage` / `release_native6_score`, which invokes `european_argmix_release.verify_cross_endpoint_barrier` for native-six and C4 products first. Neither entrypoint can be repurposed unchanged for a new joint-model method ID. A later explicitly registered scorer must retain its required all-parent/all-method prediction barrier and native endpoint, using the pure joins/metrics above. Do not relabel the new model as the old TreeLoom implementation or silently change the existing frozen comparison DAG/barrier.

## Proposed later per-parent streaming evaluation

1. Freeze the final checkpoint hash, model source/configuration, selected arm, training-derived settings and TEST context seed namespace before opening TEST observations for prediction. Record exact method identity and the intended comparison/scoring barrier.
2. Authenticate the original five registries and observation bindings, and register their exact canonical row/site mappings without resampling or target dropping. The source cohort quotas imply approximately 209 focal sites per parent, but actual registries must be verified at that stage.
3. For one complete parent at a time, use four CPU workers to prepare immutable site tensors in RAM from its observed native sources. Feed the registered prediction loader; this preserves site reuse and avoids a persistent full-five `.pt` cache. Observation extraction remains label-free. Native sources and the TRAIN cache remain untouched.
4. Reassemble every output into canonical example-index order 0..9,999. Seal per-parent raw float32 six-class probabilities (and optional logits), IDs, registry/source/checkpoint hashes, and an explicit `query_ancestry_truth_opened=false` receipt. Validate finite logits, the unmodified probability simplex, every canonical ID once and exact row coverage. Finish and seal all five parents before any truth access.
5. In a separate external scoring step, satisfy the registered method/parent barrier, authenticate producer-bound generation/schema/truth references, then call `native_truth_labels`. Report per-parent and per-cohort native metrics, and equal-parent means/sample SDs across the five independent simulations. Existing scorer configuration caps truth decompression at 16 GiB per parent and scorer RSS at 32 GiB; those are legacy guards, not measured truth sizes. Process scoring parents sequentially and retain the explicit cap/checks.

## Rough resources, derived without reading TEST payloads

The frozen cohort query counts 182, 162, 14, 302 and 26 imply respective 2,000-target site grids of 11, 13, 143, 7 and 77 positions. Their exact rational rank-grid union has 209 elements; minimum spacing is 1/1001. Thus each parent has 209 unique sampled ranks if it has at least 1,001 retained SNPs, otherwise no more than 209. Five parents therefore project to 1,045 shared-site contexts. This is quota/grid mathematics, not an inspection of TEST registry rows.

| Quantity | TRAIN-based projection |
|---|---:|
| Full-five prepared state | 7.56 GB |
| Full-five raw observation packets | 78.8 MB |
| One-parent prepared state in RAM | 1.51 GB |
| Four-worker CPU preparation, all five | 257 seconds |
| Probabilities + int64 example IDs, all 50,000 targets, one model | 1.6 MB uncompressed |
| Optional float32 logits, all five | another 1.2 MB |
| Eventual selected int64 label vector, external scorer only | 0.4 MB |

These are mean-based projections from the measured TRAIN cache, not worst-case bounds or measured TEST payload sizes. A full-five prepared disk cache plus raw packets would bring the completed 9.123 GB TRAIN scope to roughly 16.766 GB, leaving only 0.414 GB below the 16 GiB cap before later training/evaluation artifacts and size variability. It is not a safe full-cache reservation. Per-parent RAM preparation with small sealed prediction outputs is the recommended next-stage design; any optional disk staging and retention policy must be explicitly registered then. No source copying, TEST extraction, scoring, cache allocation or execution was launched for this assessment.
