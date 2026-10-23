# ARGMix source census

All pinned files were read as text/data. All 85 public author files are byte-identical to freshly fetched public HEAD. The 86th file is the local source receipt. Scientific paths were traced statically, without imports, compilation, test execution or model runs. Dependency locks are provenance rather than recovered experiment configurations.

| File | Lines | SHA256 prefix | Role |
|---|---:|---|---|
| `.gitignore` | 38 | `3c28e7148271edb0` | Excluded development tests/scripts/checkpoints lead |
| `ARG_GNN_SOURCE_RECEIPT.json` | 11 | `4d91b81cdc3f30cc` | Local provenance addition, not a public author file |
| `CITATION.cff` | 6 | `f15a5e6171302bb2` | Citation metadata, no paper-era archive link |
| `LICENSE` | 21 | `dbd9c86fc2e571b7` | MIT licence |
| `README.md` | 140 | `0fb9fccdc7331cd1` | User workflow and public links |
| `docs/config_reference.md` | 170 | `a7ea710592f85a4b` | Public documentation examined for inputs, settings and links |
| `docs/forward_config.md` | 118 | `abbf99ab5efeb387` | Public documentation examined for inputs, settings and links |
| `guides/hgdp_1kgp.md` | 546 | `ae4256598a2d4a1e` | Public documentation examined for inputs, settings and links |
| `pixi.lock` | 5157 | `8639e2a46050139e` | Dependency lock metadata, not experiment inputs |
| `pixi.toml` | 67 | `2a2f26a3fd1cefdf` | Callable entrypoints/build/dependency definitions |
| `sim/README.md` | 11 | `acb2a522cd77ca1b` | Public documentation examined for inputs, settings and links |
| `sim/Snakefile` | 723 | `4519b473e868dafd` | Forward simulation/Relate/NPZ/MSP/PCA workflow |
| `sim/scripts/build_relate_inputs.py` | 66 | `a3eb23fa077bd266` | Age/population input joins |
| `sim/scripts/build_sample_manifests.py` | 373 | `1703d1e3be582d1e` | User-supplied founder metadata and simulated sample IDs |
| `sim/scripts/forward_sim_multievent.py` | 624 | `52b587b3563f5a81` | Real-founder mosaic simulator, Morgan/Poisson breakpoints |
| `sim/scripts/make_truth_msp.py` | 78 | `d2be9a849f015bcd` | NPZ ancestry-label export to MSP |
| `sim/scripts/plot_pca.py` | 104 | `c8ee15ffbf0896b6` | QC plot only |
| `src/argmix_cache.py` | 81 | `33fbd963b0f87677` | Cache fingerprints, no model recipe recovery |
| `src/argmix_cli.py` | 166 | `e29f2250e00ba369` | CLI dispatch |
| `src/argmix_execution.py` | 148 | `e995f4ac47c4e074` | Subprocess/wait/run receipts, not executed |
| `src/argmix_manifest.py` | 218 | `06a0090d378e3e7c` | Manifest and sample-order/reference joins |
| `src/argmix_project.py` | 763 | `e1b6018e45e6d6e3` | Configuration defaults/validation/backend materialization |
| `src/commands/__init__.py` | 1 | `fa82b59e47de01c7` | CLI command implementation/helper |
| `src/commands/data.py` | 443 | `d07745cffe36230c` | CLI command implementation/helper |
| `src/commands/infer.py` | 989 | `93c52237b619cc8a` | CLI command implementation/helper |
| `src/commands/model.py` | 410 | `56dfa32d72796e43` | CLI command implementation/helper |
| `src/commands/project.py` | 155 | `e511410b394b785c` | CLI command implementation/helper |
| `src/commands/runtime.py` | 78 | `83daa114fdf5c60a` | CLI command implementation/helper |
| `src/config_loader.py` | 654 | `639ce92d2be505f9` | Scientific configuration validation and ancestor rejection |
| `src/export_torchscript.py` | 453 | `5953d5b9f683fb18` | Checkpoint-to-TorchScript exporter, ancestor geometry restriction |
| `src/final_test_metrics.py` | 1463 | `b78d619f25a6e3cf` | Callable held-out SNP/dose/coverage evaluator |
| `src/forward_panel.py` | 311 | `aac9c42d488e9ebc` | Reference/source roles, founder sample split and age metadata |
| `src/jackknife.py` | 79 | `e4e0c6340582ff07` | Haplotype-count jackknife uncertainty helper |
| `src/joint_interval_model.py` | 1805 | `8e30a7ece3cf5236` | Base/weighted encoders, bidirectional GRU, extraction/chunk/refinement |
| `src/joint_support.py` | 363 | `8a27ac02497b122d` | GRU, masked loss, HDF5 rows and map interpolation |
| `src/run_inference.py` | 1098 | `41ecbc14605d7377` | Fast Rust inference wrapper and ancestor/checkpoint restrictions |
| `src/run_joint_inference.py` | 173 | `ab4eefcf2de42b44` | Joint GRU inference/HDF5 output |
| `src/rust/ancpath/Cargo.lock` | 2261 | `8af5197c132cc1a0` | Rust dependency/build metadata; no original model/data settings |
| `src/rust/ancpath/Cargo.toml` | 52 | `17ca8039ce0db2be` | Rust dependency/build metadata; no original model/data settings |
| `src/rust/ancpath/build.rs` | 34 | `b7a60b69603c7fba` | Rust dependency/build metadata; no original model/data settings |
| `src/rust/ancpath/src/bin/build_training.rs` | 1217 | `55c136ba40ccc416` | Executable archive builder; forces zero ancestors |
| `src/rust/ancpath/src/extracted_batch.rs` | 90 | `a1231a4b6215df82` | Extractor/LCA/packing/merging/IO support component and inline tests |
| `src/rust/ancpath/src/fast_benchmark.rs` | 118 | `90846a05ec39e391` | Exported raw fast extraction benchmark API accepts ancestors |
| `src/rust/ancpath/src/fast_lca/binary_lift.rs` | 311 | `be5e5d77b0c9d980` | Extractor/LCA/packing/merging/IO support component and inline tests |
| `src/rust/ancpath/src/fast_lca/engine.rs` | 1235 | `1daf49d5c4ac777a` | Fast LCA extraction with surviving ancestor-chain implementation |
| `src/rust/ancpath/src/fast_lca/mod.rs` | 2 | `81f1587cf0b47d94` | Extractor/LCA/packing/merging/IO support component and inline tests |
| `src/rust/ancpath/src/fast_lca/tmrca.rs` | 147 | `3831f5acb18a2ce4` | Extractor/LCA/packing/merging/IO support component and inline tests |
| `src/rust/ancpath/src/flat_row_merger.rs` | 478 | `804ce015a81fa4f4` | Extractor/LCA/packing/merging/IO support component and inline tests |
| `src/rust/ancpath/src/hdf5_writer.rs` | 856 | `71d0eb7ad8e89c8c` | Extractor/LCA/packing/merging/IO support component and inline tests |
| `src/rust/ancpath/src/inference_worker.rs` | 1398 | `c72b5366faa9b446` | Extractor/LCA/packing/merging/IO support component and inline tests |
| `src/rust/ancpath/src/interval_bounds.rs` | 34 | `3d1777329684a63d` | Extractor/LCA/packing/merging/IO support component and inline tests |
| `src/rust/ancpath/src/interval_emitter.rs` | 234 | `42d308fdb835b23b` | Extractor/LCA/packing/merging/IO support component and inline tests |
| `src/rust/ancpath/src/lib.rs` | 143 | `ec8614b78c931f03` | Extractor/LCA/packing/merging/IO support component and inline tests |
| `src/rust/ancpath/src/npz_truth.rs` | 687 | `845e77c18c74e3bf` | Rust first-SNP/majority target resolver with Python-oracle parity fixtures |
| `src/rust/ancpath/src/pipeline_error.rs` | 35 | `4bf2690c9580324d` | Extractor/LCA/packing/merging/IO support component and inline tests |
| `src/rust/ancpath/src/pipeline_types.rs` | 402 | `26e236b27be7658b` | Extractor/LCA/packing/merging/IO support component and inline tests |
| `src/rust/ancpath/src/rust_torchscript_inference.rs` | 234 | `604177293010c2f6` | Exported generic model inference component, requires supplied weights |
| `src/rust/ancpath/src/sample_iterator.rs` | 1453 | `df75f4278bf3a777` | Exported legacy extraction API with surviving ancestor path |
| `src/rust/ancpath/src/shard_pool_iterator.rs` | 1064 | `873299927fcb2800` | Main inference worker orchestration, accepts low-level ancestor argument |
| `src/rust/ancpath/src/signature_cache.rs` | 204 | `ab19ecc75ad18504` | Extractor/LCA/packing/merging/IO support component and inline tests |
| `src/rust/ancpath/src/subgraph_merge_pipeline.rs` | 183 | `1933163f59a6fb15` | Merge-off pass-through; merge-on requires zero ancestors |
| `src/rust/ancpath/src/tmrca.rs` | 6 | `8c0133889e354d98` | Extractor/LCA/packing/merging/IO support component and inline tests |
| `src/rust/ancpath/src/tsv_writer.rs` | 101 | `9ac73e2ec4ed0ae1` | Extractor/LCA/packing/merging/IO support component and inline tests |
| `src/rust/ancpath/tests/npz_truth_oracle.py` | 94 | `d6aa7f6b76efd2cf` | Synthetic parity-test oracle; not original benchmark dataset |
| `src/rust/ancpath_incremental/Cargo.lock` | 629 | `88785e1e67315dc4` | Rust dependency/build metadata; no original model/data settings |
| `src/rust/ancpath_incremental/Cargo.toml` | 21 | `6254873c8aebf6a2` | Rust dependency/build metadata; no original model/data settings |
| `src/rust/ancpath_incremental/src/benchmark.rs` | 98 | `5015c629829711e7` | Deprecated incremental support component and inline tests |
| `src/rust/ancpath_incremental/src/binary_lift.rs` | 284 | `7cddd7d0af63b7b6` | Deprecated incremental support component and inline tests |
| `src/rust/ancpath_incremental/src/engine.rs` | 1820 | `4c338c107e8cbb16` | Deprecated incremental extractor, reference-only matrix dimension |
| `src/rust/ancpath_incremental/src/lib.rs` | 35 | `db17b4088ec111dc` | Deprecated incremental support component and inline tests |
| `src/rust/ancpath_incremental/src/ref_selection.rs` | 110 | `0638e2ac6d8eabaa` | Shared nearest-reference selection, ties/drop/padding tests |
| `src/rust/ancpath_incremental/src/sample_batch.rs` | 231 | `825281ded7d37802` | Deprecated incremental support component and inline tests |
| `src/tmrca_attn.py` | 378 | `8cc88e2acc4a93d3` | Base Transformer attention and feature model |
| `src/train_joint_model.py` | 648 | `07f2afa904fa0063` | Actual archive loader/trainer/selection/checkpoint path |
| `src/tree_sampler.py` | 609 | `cbc5a5d4c573a636` | Legacy/compatibility Python extraction and ancestor rejection |
| `src/truth_labels.py` | 303 | `91148a4e894ffd3d` | Python per-interval NPZ target resolver |
| `src/utils/__init__.py` | 0 | `e3b0c44298fc1c14` | Small metadata/device/logging/reproducibility helper |
| `src/utils/device.py` | 27 | `f78ab683ac087aaf` | Small metadata/device/logging/reproducibility helper |
| `src/utils/logging_setup.py` | 10 | `4f832733473803b5` | Small metadata/device/logging/reproducibility helper |
| `src/utils/reproducibility.py` | 14 | `c29f327b34337a38` | Small metadata/device/logging/reproducibility helper |
| `src/utils/tsv_io.py` | 32 | `b3322c0078645835` | Small metadata/device/logging/reproducibility helper |
| `templates/forward_blank/argmix.advanced.yaml` | 2 | `adb3877f80d12159` | Only shipped template and placeholder inputs |
| `templates/forward_blank/argmix.yaml` | 89 | `ad0445a13ef2fe49` | Only shipped template and placeholder inputs |
| `templates/forward_blank/events.yaml` | 12 | `578b322897a8253f` | Only shipped template and placeholder inputs |
| `templates/forward_blank/forward_config.yaml` | 31 | `fb863abfef2cee6a` | Only shipped template and placeholder inputs |
| `templates/forward_blank/reference_panel.tsv` | 1 | `17c0268ade716983` | Only shipped template and placeholder inputs |
