# Unchanged author ARGMix execution bridge v1

## What this work establishes

The new bridge executes the author's unchanged `TmrcaAttn` component, rather than the independent `ARGMixTreeControl` adaptation. It verifies the complete pinned source before and after import, runs an explicitly synthetic CPU forward at the stock template's width 128/four layers, and checks a native TorchScript export in memory. It preserves the author's attention implementation, vocabulary embeddings, CLS readout, masks, distance handling and parameterization.

It does not establish a trained native competitor, paper reproduction, real-data accuracy or scientific readiness. No model training, optimizer step, Rust extraction, upstream simulation, installation, GPU execution, checkpoint load, real cohort label access or final-data access occurred. Existing model, preparation, lock, predictor, custody, selection, calibration and analysis files were left frozen.

The author source remains unchanged at:

- Repository: [AI-sandbox/argmix](https://github.com/AI-sandbox/argmix).
- Commit: `c2481248eb0de5fda659d68f59c51e0e27becd20`.
- Original archive SHA256: `98c2cda680ca07c62723177599f2e7f37dc5b15d3ab05d2aa8c4a8f7f075747e`.
- Complete 86-file inventory SHA256: `4fbcc41801027983a9654920d143f053d5b74ffea020b751a13757c0ae327ae8`. The inventory hashes sorted relative file paths and their byte SHA256 values using compact sorted JSON plus one newline.
- Retrieval receipt: `.runtime/author_sources/argmix_c2481248eb0d/ARG_GNN_SOURCE_RECEIPT.json`, SHA256 `4d91b81cdc3f30cc0cec6b9523cc936c9dad0947f8f44ba6d3c3c7c171fb8698`.
- Native component: `src/tmrca_attn.py`, SHA256 `8cc88e2acc4a93d3038ea0ac862f4d48e860c58dc202dc93b64c634c75fa76f4`.

The retrieval receipt's source-only assertion describes the earlier download. This new work records genuine execution of the unmodified component on synthetic inputs in a separate receipt. It does not rewrite that retrieval receipt or claim that the full author pipeline ran. The MIT licence and GRPE attribution remain in the unmodified source.

## Exact version distinction

Shanks et al., *Graph transformer for ancient ancestry inference*, [bioRxiv v1](https://www.biorxiv.org/content/10.64898/2026.04.05.714076v1.full), describes 25 nearest observed references, three immediate inferred ancestors and the query, hence 29 biological nodes plus the classifier token. Its published-task settings must be tied to an official compatible source revision and training configuration before a reproduction claim.

The inspected current commit differs materially:

| Item | Current unchanged author source | Earlier independent control |
| --- | --- | --- |
| Input | Population token IDs, full query/reference TMRCA matrix and token mask | Shared 48 observed features |
| Biological tokens | Query plus references; Python and Rust training force zero ancestors | Full tree or optional nearest references/available ancestors |
| Template references | 15; training leave-nearest-out augmentation drops one nearest reference | Existing panel has 24 total reference haplotypes |
| Distance | Integer node times, asymmetric elapsed TMRCA, optional log2(1+distance), template cap 14 | Independent paper-equation elapsed-time implementation |
| Attention relative key | Original row-gather behavior, yielding the disclosed k_i term | Printed-equation k_j implementation |
| Neural shape | Template width 128, four transformer layers, four heads, 16 base output features | Earlier primary width 32/two layers |
| Current CLI training | Transformer plus genomic bidirectional GRU | Independent supervised classifier |
| Claim established here | Unchanged native component CPU/TorchScript execution only | Independent adaptation |

`config_loader.get_model_config`, `tree_sampler.ShardDataset`, `export_torchscript.py` and Rust `bin/build_training.rs` reject `num_ancs`; the Rust training builder sets it to zero. Lower-level Rust code still contains ancestor extraction, but bypassing the supported Python/Rust training contracts to enable it would create a changed execution configuration, not demonstrate the published version. The bridge does not correct the k_i/k_j discrepancy, add missing ancestors, replace token embeddings with shared features, remove the GRU from the current CLI, or weaken the model.

A paper-compatible official revision remains a concrete prerequisite. Current stock execution may also be studied under its own revision and method label, with its differences and tuning budget disclosed.

## Native data and training flow

The stock CLI sequence is `init → validate → build-data → build-training(train,val) → train → test/infer → export`. `build-data` performs the author's forward simulation/Relate workflow. It is not run here. Native `train` invokes `src/train_joint_model.py`; the Rust `build_training` binary prepares its tensor archives. The independent project adapters do not replace those stages.

The biological input contract includes:

- An observed inferred tree sequence with node IDs, tree intervals and observed node times, plus chromosome/build-compatible recombination map and Relate preprocessing assets when that upstream workflow is used.
- A reference/sample-age panel. Native inference reads `node_id ancestry age`; full forward projects use `sample_id ancestry age cohort_role founder_split`. Ages, tree node times and generation units must agree. Reference ancestry tokens are observed reference information.
- Query identities selected independently of their census outcomes. The native query token is always `<SAMPLE>`. Query census ancestry labels must not enter vocabulary selection, reference selection, time matrices, model inputs or tuning.
- Separate external truth NPZ, Relate `.sample` haplotype correspondence and observed SNP/coordinate joins for training/evaluation. `ShardDataset` requires the truth NPZ and sample file outside inference. The author's observed-position filtering and label-purity policies must be retained and audited.
- Explicit parent/subject/founder-pool assignments before any interval extraction, windowing, augmentation or genomic chunking. The stock forward protocol splits founder panels into disjoint train/validation/test pools, with only training founders used as ARG references.
- Enough actual observed reference haplotypes for the configured nearest-reference rule. Published 25-nearest execution requires at least 25 available references. Current leave-nearest-out training at k references additionally asks for k+1 candidates. The present 24-reference panel cannot establish the published 29-node configuration. The unchanged Rust selection helper can reduce the requested nearest-drop or repeat its last reference index when the pool is too small. A full biological input preflight must count distinct available reference haplotypes rather than treating padded token width as evidence of the required panel.

Native `tree_sampler.py` defines `<PAD>, <UNK>, <SAMPLE>, <ANC>` then sorted population tokens. Even though `<ANC>` remains in the vocabulary, supported current extraction uses zero ancestor features. The current supported fast Rust backend rounds observed node times to i32, derives nearest-reference ranking and pairwise elapsed times from observed genealogies, and uses reference-index ordering for ties. The native linear distance path casts to u16 before its cap, while the log path transforms integer elapsed times first; these exact native transforms and observed time ranges need recording rather than a silent numerical rewrite. The bridge does not substitute another tie rule, continuous-time transform or feature extractor.

The current template sets width 128, four transformer layers/heads, feedforward multiplier 4, dropout 0.25, attention dropout 0.1 and base output width 16. It uses genomic chunks of 1024 rows with stride 512, a bidirectional width-128 one-layer GRU and two recombination-map features. Training defaults are batch 16, 20 epochs, AdamW learning rate 2e-4, weight decay 1e-4, label smoothing 0.1 and patience 2. It selects the lowest validation batch-mean loss and restores that checkpoint. These are source findings, not completed training observations or a substitute for matching the paper's settings.

## Original exports and rigorous evaluation

The native transformer returns original float32 logits before softmax. Its author TorchScript exporter supports logits-only and logits-plus-final-feature outputs. Those logits can support uncapped float64 log-softmax NLL if preserved at their original precision with exact interval/query/class joins.

The Rust HDF5 path stores probabilities, logits and final features as uint16 half-float bit patterns. `inference_worker.rs` converts float32 transfers to half-float bits too, so choosing float32 device-to-host transfer does not retain float32 values in the final HDF5. Reading uint16 values as numeric probabilities would also be incorrect. Quantized exports cannot silently substitute for the existing original-raw-logit primary contract.

Current joint refinement computes softmax for each overlapping chunk, then averages probabilities at each row. It returns final probabilities, not one original final logit vector. Taking log of those averaged probabilities and naming the result raw logits would change the estimand and disguise the native aggregation. The method/version and score contract must explicitly distinguish original base logits, per-chunk refined logits and native final averaged probabilities before any confirmatory freeze.

A subsequent shared-data method comparison must preserve the exact observed trees, reference/query identities, source panel and physical coordinate joins, with external outcome labels. It must disclose native preprocessing and information differences, ancestry classes, tuning trials/updates/seeds, validation selection, parameter counts, reference count, compute and missing/failing cases. Biological parents remain the independent units. Use comparable parent-level NLL/proper-score, calibration, ancestry-dose and tract/boundary metrics with physical span weighting and uncertainty; do not turn correlated windows, haplotypes or parent-seed cells into independent replicates.

Published-task reproduction is a separate branch. The existing stdpopsim four-founder census benchmark differs from the AncestralPaths Nature six path-ancestry classes/nine-cohort formulation and from ARGMix's forward/reference-panel task. No class merging, label renaming or cross-task score should be called a reproduction without a prespecified biological mapping. ARGformer's observed-only node-vocabulary adaptation/retrieval also requires explicit transductive-versus-new-ARG transfer and downstream readout definitions. These are integration requirements, not reasons to exclude those biologically relevant methods.

## Root-only environment and next steps

The current project interpreter is Python 3.11.17 / Torch 2.6.0+cu124. It lacks `ancpath`, `h5py`, `polars`, `maturin`, `snakemake`, `snputils`, `transformers` and `wandb`. Pixi, Cargo, Rust, Relate, Snakemake, bcftools and plink2 were not found in its command PATH. Installed tskit 1.0.3 exceeds the author's <0.7 requirement; pandas 2.3.3 is below the author's >=3.0.2 requirement.

The author's `pixi.toml` declares Linux-64, Python >=3.12,<3.13, CUDA 12, Rust >=1.90,<1.91, maturin >=1.9.5,<2, clang 21, compatible compilers/binutils/HDF5, tskit >=0.6.4,<0.7, pandas >=3.0.2,<4, polars >=1.40,<2, and the remaining declared scientific/VCF/workflow packages. Its actual `pixi.lock` pins Python 3.12 and Torch/libtorch 2.12.0 with CUDA 12.9. The Rust manifest uses tch 0.23 with download-libtorch. Python/Rust loaded-library versions, generated rpaths, exported TorchScript compatibility and CUDA-driver compatibility need measured build/import checks. The current environment is not an equivalent native runtime.

Concrete remaining steps owned by root:

1. Resolve the official paper-compatible revision/configuration, or explicitly retain this newer current revision as a separately named comparison. Freeze published reproduction and shared-data scope, sample-age/reference panel, ancestry classes and original score/export definitions.
2. Create an exact execution copy inside `.runtime/author_exec/`, leaving the pristine source read-only. Verify all 86 original file hashes before and after build. Stock compilation adds `.pixi`/target outputs to that copy, so the current strict pristine-tree checker is not a post-build directory inventory checker; record generated build artifacts separately.
3. Install/build an isolated contained native runtime using the author's manifest/lock after resolving any locked Torch/tch/driver compatibility issue. Keep environments, Pixi/rattler/package caches, Cargo/Rust homes and targets, temporary files, plotting/WandB/Torch caches and logs inside this repository. Do not alter the frozen project environment. Set `PYTHONDONTWRITEBYTECODE=1` when importing pristine author Python.
4. Follow the author's stock compile tasks on the execution copy: `pixi install`, then `pixi run compile`, which builds both the ancpath Python extension and the separate `build_training` binary. The optional deprecated incremental backend is not required. Verify actual library imports/build provenance and CPU stock CLI help/export on synthetic inputs before real extraction or training.
5. Prepare/review genuine development inputs and split/reference/truth manifests, and measure that native extraction returns the original token/distance/identity contracts. Real-data assets and licences, chromosome/build, sample ages, input genotypes/trees, checkpoint/source hashes and all preparation choices require an auditable receipt. No actual final targets should be prepared or opened at this stage.
6. After those checks, root may schedule reviewed training/inference with the project launcher/resource policy, matched tuning and independent parent evaluation. Preserve full original probabilities/logits and failure records. No final lock or held-out preparation is authorized by this source bridge.

These are requirements and commands for root review, not installations or data jobs performed by this subagent.

## API, commands and actual verification

New owned files:

- `src/arggnn/official_argmix_bridge.py`
- `scripts/check_official_argmix_source.py`
- `tests/test_official_argmix_bridge.py`

`verify_author_source` admits the exact full source inventory and rejects escaped/traversing/symlinked paths, missing/extra files or changed bytes. `OfficialARGMixSource.check` detects later drift. `inspect_official_source` reads source/dependency metadata without upstream execution. `native_model_smoke` loads the unmodified component by its pinned file path, suppresses bytecode writes, requires explicit hidden CUDA, restores CPU thread/RNG/module state and performs no optimization.

```bash
source scripts/project_env.sh
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 "$ARGGNN_PYTHON" \
  scripts/check_official_argmix_source.py
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 "$ARGGNN_PYTHON" \
  scripts/check_official_argmix_source.py --native-model-smoke
```

The source preflight, native component check and focused pytest run all returned exit 0. The actual component has 803,347 parameters; synthetic tokens are [2,16], asymmetric distance matrices [2,16,16], and raw float32 logits [2,3]. TorchScript matched exactly. These inputs are not biological feature extraction and weights are untrained random initialization.

The focused CPU suite passed **9 tests in 5.64 seconds**. The one warning is the unchanged author's dtype-specific Tensor annotation being treated as an ordinary Tensor by TorchScript; it was preserved. Tests cover full source/hash inventory, missing/extra/changed/symlinked files before import, containment, later source drift, explicit CPU gating, exact stock native constructor/forward/script behavior and source/RNG preservation.

Detached logs, launch PID, per-stage exit status and final receipt are stored under `.runtime/tmp/official_argmix_native_v1/`. The complete author inventory remained unchanged. These checks establish a runnable native component bridge in the current CPU environment. They do not establish native biological extraction, a complete stock training/inference pipeline, trained checkpoints, real-data results or scientific readiness.
