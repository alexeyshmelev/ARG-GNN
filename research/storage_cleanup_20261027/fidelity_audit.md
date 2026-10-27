# Historical storage dependency audit — 2026-10-27

Audit scope: current paper queue, forthcoming direct four-class AncestralPaths work, fullVAL, and TreeLoom efficiency qualification. This audit performed targeted reads and wrote only audit reports; it did not delete data, launch workloads, change Git, or perform a recursive size inventory. ROOT owns the final manifest, open-file/inode guards, and any deletion.

## Evidence and limits

The canonical queue is `configs/benchmarks/paper_campaign_resume_20261026_v1.json` (SHA-256 `5c0bd431ca0f466fb155ec12adefa4437f594eb7699c8ed592667f0bd95a762d`), with state `.runtime/benchmarking/paper_campaign_resume_20261026_v1/state.json`. The inspected snapshot contained 252 tasks: 49 succeeded, 200 pending, and 3 running. Current task call arguments contained no direct `data/*` or top-level `results/*` paths.

The reference index follows explicit metadata references from the current plan, direct-four plan, original AP TreeLoom job, FIT CLIv3 pins, fullVAL evaluator pins, preprocessing release, and TRAIN001 timing review, with up to three metadata hops and pinned Python string-literal inspection. It reached the 1,200-JSON cap; its 10,530 paths and 891 Python source paths are a bounded audit, not exhaustive absence proof. References can describe historical provenance rather than live consumers. Existing ROOT inventories supplied sizes; this audit did not repeat them.

## Preserve current dependencies

- All of `experiments/results/benchmark_campaign_20261018` and the current `experiments/results/benchmark_campaign_20261018_workflow` dependencies, especially `release_v2` and the original AP three TRAIN/five TEST generated, native, registry, and target artifacts.
- Current output/input roots: `experiments/results/paper_american_20261024`, `experiments/results/paper_ancestralpaths_cnn_20261024`, `experiments/results/paper_campaign_20261024`, `experiments/results/paper_campaign_resume_20261026`, `experiments/results/paper_gst_lai_8_20261024`, and `experiments/paper_argmix_1611_rebuild_20261024_v1`; preserve all active checkpoints and explicitly linked inputs.
- `data/ARGMix_prospective1611_HapMapII_GRCh37_chr22_v1`: current ARGMix recipe requires `chr22.rebased.map`, `map_binding.json`, and `derivation_receipt.json`; retain the complete small map/provenance scope.
- `data/ARGMix_prospective1611_first_train_H020v3_generated_train000_oct16_root_v1/operational_successor_train000_live_boundary_root_v1`: current ARGMix recipe `existing_train0.path` and published observation receipt depend on this scope, including `observed/observed.vcf`; retain observed/truth inputs.
- `data/native_FLARE_build_existing_JDK_root_v1`: preserve `flare.jar`, `receipts/binary_provenance.json`, and supporting provenance; the current American workflow uses this native implementation.
- All current `research/efficiency_20261027/root/runs/` inputs/outputs and fixed FIT/qualification checkpoints. Specifically retain both `european_short_development_val_10mb_v1` and `european_short_development_val_10mb_v2` unchanged while fullVAL runs: its guardian verifies retained hashes and exact charged-byte stability. Retain live `european_independent_validation_200mb_v1` in full.
- Preserve code, configs, design descriptions, all metrics/history JSON, text receipts, figures, `paper/`, and `docs/`.

## Four old data scopes: bulk payload candidates

The following scopes have no structured dependency references in the audited closure and no current task calls to their historical producer/consumer modules. Their bulky payloads are cleanup candidates within ROOT's exact allowlist and runtime guards; retain metadata and concise historical evidence.

| Scope | ROOT inventory allocated bytes | Main obsolete payload |
| --- | ---: | --- |
| `data/biological_candidate_pilot_v1` | 38,036,480,000 | Historical bank and whole-observed tensors |
| `data/expanded_full_panel_v1` | 47,641,731,072 | Historical pairwise arrays, trees, ancestral and VCF intermediates |
| `data/expanded_native_argmix_prepared_v3` | 37,099,175,936 | Prepared historical tensors |
| `data/ARGMix_prospective1611_Relate_accepted_train000_v4` | 31,074,267,136 | Old accepted Relate binaries and intermediates |

Text references to the biological candidate name occurred in old config-kind/provenance checks. Expanded full-panel references identify the historical 32 TRAIN/8 VAL `ap_expanded_*` experiment and source-audit evidence. No targeted hits were found for the prepared-v3 or accepted-Relate scopes. Current ARGMix train000 uses the protected live-boundary generated scope above. The direct-four/fullVAL/FIT routes use original AP artifacts, not these pilot/expanded datasets.

## Historical top-level results

The ten candidate/mechanism/official-ARGformer result scopes listed in `fidelity_selected_evidence_v1.json` are historical and have no current task-call references in this audit. Per-epoch `.epoch_XX.*.probabilities.npz` are the principal disposable candidate/mechanism bulk artifacts. Obsolete bulky model/embedding tensors in the four official-ARGformer pilot scopes are candidates; current benchmark checkpoints are in the protected experiment scopes.

Subtract the exact `keep_paths` list in `fidelity_selected_evidence_v1.json` from any cleanup manifest. It preserves 46 sealed `.selected.*.probabilities.npz` predictions plus four `downstream_C4_local_readout.npz` files (50 files, 3,077,205,838 logical bytes). Top-level JSON was read in all ten scopes with no parse errors. Preserve all JSON/text metadata and small selected/best-epoch checkpoints retained by ROOT's size threshold. Metrics explicitly reference sealed selected prediction files; retaining those files makes the historical evidence independently inspectable without retaining every epoch dump.

The four-data-plus-ten-results gross inventory is 291,835,777,024 allocated bytes. This is not a reclaimable-byte estimate: metadata and selected evidence remain, and hardlinks may make gross totals differ from actual reclamation. Do not delete entire directories or apply a repository-wide extension rule.

## Deferred and unassessed

- The ~85.84 GB C4 `release_20261022_multigpu/other/C4__ancestralpaths_european_train_000__release_archive/native/train.pt` archive is owned by the execution audit. This audit found historical archive metadata references, no direct current-plan path to that tensor, and no direct fullVAL/direct-four/FIT use; the bounded closure cannot independently authorize its deletion.
- Earlier `data/age*`, `data/ancient*`, `data/boundary*`, raw/processed, and other old roots remain unassessed in this final tranche. No disposition is inferred from their names.
- Existing processes, including old `handmade_spawn_check.py`, are not stopped or altered by this audit. ROOT must apply current process/open-file, exact-path, no-symlink, and inode guards immediately before execution.
