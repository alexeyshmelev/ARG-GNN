# European benchmark stages still required

The canonical AP recipe now generates **200,000,000 bp**, the complete released **476 diploids / 952 haplotypes**, **3 independent TRAIN + 5 evaluation parents**, and **six census-350 paths**. BAA admixture uses the paper's **170 generations**; the released code's 169 is retained as provenance. Fresh seeds are allowed. Generation is concrete; a complete paper-aligned training campaign is not ready to execute.

## Ready source and concrete preparation helpers

- `european_recipes.generate_ancestralpaths_parent` is the actual guarded msprime generation entrypoint. It derives map endpoints and all length metadata from the canonical recipe and rejects stale source pins or changed strict-mode metadata.
- New `european_execution.prepare_relate_inputs` reuses the pinned observed VCF filter with a full-length uniform map: 1 -> 0 cM and 200,000,001 -> 200 cM. Filtering still materializes retained VCF rows in RAM.
- New `european_execution.clean_converted_tree` applies the existing minus-one converter coordinate transform, actual-support clipping and annotation removal at the declared length. Native sample/node IDs and all retained sites/trees remain. Optional 100 bp diagnostic-grid metadata is explicit; native sampled training needs no grid.
- New `european_execution.iter_native_sampled_features` visits trees once for an already-frozen, position-sorted sampled registry. It preserves repeated examples and original example IDs. Native shell arithmetic uses Float64 repeated addition, then casts final 5x9 rows to Float32. No labels enter extraction.
- Existing licensed Relate 1.2.4 installation metadata and the AP CPU TensorFlow interpreter are present. Presence and prior source review do not qualify a new 200 Mb run.

These are concrete preparation functions, not a campaign launcher or training CLI. ROOT supplies actual admission, process supervision, storage/RSS/wall limits and queue dependencies.

## Required queue order and current blockers

1. **Freeze campaign and independent seeds.** Exactly three TRAIN and five evaluation parents. Freeze ancestry/mutation and native inference seeds, ordered observed sample identities, sampled-registry policy and native-six class order.
2. **Generate each complete 200 Mb parent on CPU.** Use the maintained generator and fresh complete-parent output/RSS/wall limits. Failed limits preserve failure; never shorten, crop or redraw a parent.
3. **Run native Relate on CPU.** Wire the new preparation/cleanup functions around ConvertFromVcf, canonical sample order, SNP annotations, N=30,000/mu=1.25e-8 initial inference, five size iterations/final branch re-estimation, and ConvertToTreeSequence. Reuse source-pinned native tools/process supervision. The existing `prepare_ancestralpaths_full_panel_discovery_v1.infer_parent` cannot be called unchanged: `observed_assets`, the old input-map adapter and old converted-tree cleanup all hardcode 2 Mb.
4. **Freeze paper-style sampled registries.** The supplement specifies 50,000 TRAIN examples total: 10,000 per admixed cohort across the three TRAIN sequences; evaluation uses 10,000 GNNs from each additional sequence. Define a disclosed deterministic distribution of integer quotas across parents and same-site truth joins, preserving the sampled native-six endpoint for both methods. No registry is produced by the new feature iterator.
5. **Prepare streamed native AP features and TRAIN labels separately.** The reviewed `expanded_native_ancestralpaths_v1` is not a drop-in driver: it enforces 32 TRAIN/8 validation and requests 162,500 rows per parent with the released class-cap sampler. Its dense all-tree/all-grid cache is unnecessary for a 50,000-row sampled fit. Use the source bridge's 5x9 features, native six labels and a fresh final-epoch-five Dense/Adam fit. The paper's CNN wording versus released Dense architecture remains an explicit fidelity gap; do not claim the published CNN was recovered.
6. **Prepare complete European TreeLoom observations and qualify execution.** The reviewed optimized preparer is explicitly American C3-only (1,600 haplotypes, 30 Mb and C3 classes). The European facade accepts an already-built bank and a 32,391-parameter native-six model; it does not create that bank. A European observed-only bank/episodes/cut builder and whole-parent GPU0 fit qualification are missing.
7. **Train, seal predictions, then score externally.** TreeLoom must use the sampled registry and six-class loss, not the historical bp-weighted C4 target mass. Do not enqueue an executable European training command until the preceding artifacts and qualifications exist. The C3 cold qualification cannot authorize this workload.

ARGMix's exact roster/ages, map identity/domain, error mechanism, perturbation draws and parts of the inference/representation recipe remain explicit gaps in its separate recipe. The prospective1611 reconstruction is not substituted.

## Storage and RAM: shape-derived planning estimates

Let S be generated SNP sites, T_truth raw simulated intervals and T native inferred trees. These counts are not fixed by the declared length. The current source has no practical count-based upper bound or measured 200 Mb output maximum. A reservation cap is a fail-stop limit, not a prediction of scientific output size.

| Component | Size from current code |
|---|---|
| Two uint8 truth label matrices plus Float64 interval boundaries | About 1,388 * T_truth bytes before NPZ compression |
| Truth extraction Python row lists alone | At least 5,488 * T_truth bytes of label references, plus list/interval overhead, during conversion to arrays |
| 200 Mb auxiliary grid, two Float64 arrays | About 30.52 MiB; array-expression temporaries add memory |
| Phased 476-diploid VCF | Roughly 1.9 KiB/site, about 1.8 GiB per million sites; text metadata adds overhead |
| Observed VCF + filtered VCF + converted HAPS | Roughly three such genotype-text copies before native intermediate files |
| Site-coordinate JSON | Approximately 0.1-0.2 KiB/site on disk; Python dictionaries cost more RAM |
| Historical dense AP Float32 feature cache | 123,480 * T bytes: 11.50 GiB per 100,000 trees |
| Historical 2,000,000-center x 686 int32 join alone | 5.11 GiB, excluding site joins/features |
| Streamed AP shell workspace for all 686 queries in one tree | 246,960 bytes of Float64 features, about 241 KiB; the tree/TS and temporary leaf sets are additional |
| 50,000 retained AP Float32 training rows | 9,000,000 bytes, about 8.58 MiB, plus labels/registry and fit copies |

There is **no defensible finite operational maximum estimate from 200 Mb alone** in this implementation: raw tree tables, recurrent mutation counts, retained sites, Python metadata, sanitized/mutated tree copies, NPZ buffers and native Relate intermediates all contribute. The discrete site domain bounds sites, but its worst case is not a useful resource prediction. The old 2 Mb 4 GiB output reservation must not be reused as a qualified 200 Mb bound. A simplistic 100x extrapolation is not a guarantee either. ROOT must choose and enforce a fresh cap against current storage; a first complete-parent qualification supplies actual counts and high-water marks for later parents.

The current European TreeLoom forward also retains full T x 952 tensors. For **100,000 trees**, individual Float32 allocations are: context **11.35 GiB**, GRU input **12.41 GiB**, bidirectional GRU history **22.70 GiB**, and classifier concatenation **34.05 GiB**, before the complete observation bank, tree sweeps, other activations, gradients and optimizer state. These individual allocations need not all have identical lifetimes; they are not an asserted measured peak. The classifier concatenation alone exceeds a 24 GiB cap above roughly 70,500 trees. Bounded cut-memory recurrence does not bound the genomic GRU or these complete-bank emissions. A small wrapper cannot make this an admitted full-parent GPU path.

## Source references

- Maintained generation/evidence: `src/arggnn/benchmarking/european_recipes.py`, `configs/benchmarks/recipes/ancestralpaths_european.json`.
- New preparation helpers: `src/arggnn/benchmarking/european_execution.py`; synthetic tests: `tests/test_european_execution.py`.
- Original inference algorithms: `scripts/infer_biological_paths_relate_v1.py` (`filter_vcf`, `inferred_sample_mapping`, `clean_converted_tree`); `src/arggnn/prepare_ancestralpaths_full_panel_discovery_v1.py` (`native_tools`, `native_step`, `infer_parent`).
- Native AP: `src/arggnn/ancestralpaths_native_bridge.py`; frozen historical driver `src/arggnn/expanded_native_ancestralpaths_v1.py`.
- TreeLoom: `research/reviews/frozen_benchmark_training_ready_20261017_v1/european_frozen_treeloom_v1.py`; `src/arggnn/benchmarking/treeloom.py`; American-only preparer `research/reviews/TreeLoom_preparation_reuse_source_20261017_v1/optimized_preparation_v1.py`.
- Existing source-only AP review: `research/reviews/expanded_native_ancestralpaths_v1_fresh_audit_v1/audit.md` explicitly leaves actual feature memory/runtime unqualified.
- Primary sampling evidence: AP supplement text lines 1034-1049 in `.runtime/author_sources/ancestralpaths_supplement_41586_2023_6705_MOESM1_ESM.txt`.

## Full-length Relate orchestration implemented

`src/arggnn/benchmarking/european_native_relate.py` now loads the complete 200 Mb observed whitelist and orchestrates native conversion, inference, five population-size iterations and tree conversion. It preserves 952 samples, nine cohorts and 686 query haplotypes, without reading ancestry truth. The retained-SNP membership lookup uses a set, avoiding quadratic lookups while preserving the separate ordering checks. Five synthetic tests pass. A real bounded caller, actual fresh inputs, native sampling and European TreeLoom integration remain pending. No full-length native execution is claimed.
