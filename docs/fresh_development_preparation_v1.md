# Fresh ancestry development preparation v1

This is a concrete CPU preparation plan for **64 TRAIN / 16 VAL / TEST 0** fresh independent parents. It freezes preparation identities, source implementation hashes and resource rules. It does not freeze a final confirmatory test, selected architectures, checkpoints, contrasts or analysis. No simulation, native inference, model training or GPU launch was performed while authoring or checking these files.

The dedicated files are scripts/prepare_fresh_development.py, configs/ancient_development64_v1.json, configs/fresh_development_preparation_v1.json, configs/fresh_development_execution_freeze_v1.json and tests/test_prepare_fresh_development.py. Shared implementations and existing pilot configs remain unchanged.

## Exact interfaces and the resolved gap

AncientSimulationConfig, ancient_replicate_seeds, simulate_ancient_replicate and save_ancient_replicate remain the canonical oracle APIs. ReferenceFeatureAdapter uses the unchanged observed input/serialization contract from prepare_study.py. The existing prepare_study CLI's small loop includes Git queries, so this dedicated adapter uses the same underlying preparation APIs and records exact source hashes without invoking Git.

The existing prepare_relate_study.validate_protocol, verify_loaded_panel and summary explicitly hard-code 16 TRAIN / 4 VAL. Calling that CLI directly cannot prepare the larger panel. The new adapter reuses its inference_seed, parent_binding and verify_pair functions and the unchanged prepare_relate_inputs, run_paired_relate, RelateConfig and verified_relate_installation APIs. It generalizes only the dedicated panel loop and count-based verification. It performs no runtime monkey patch.

prepare_boundary_dataset already accepts arbitrary train/validation counts and rejects TEST data. prepare_age_dataset/load_age_caches bind every sidecar to the unchanged plain bank, observed source and exact preparation. The wrapper invokes these interfaces directly with eight physical tiles, float64 preparation, resistance scale 0.0001, condition cap 1e12, age scales 350/1400 generations and strength 3. It verifies all 80 parent identities and 640 cache tiles.

The inferred protocol cannot contain the new oracle manifest SHA before generation. The wrapper resolves and writes it as data/ancient_inferred_development64_v1/protocol.json only after the completed oracle manifest has been checked. That protocol, its source config/manifest and every raw/prepared parent artifact are then hashed into the inferred manifest. Root should use the completion receipt's factual hashes when building later training configs.

## Seeds, access boundary and preflight

The masters are **20261101** for ancient ancestry/mutation generation and **20261102** for Relate. Train replicate IDs are 0–63 and VAL IDs 0–15. Splits are fixed before inference, tiling, scoring or subsampling.

The seed ledger uses the actual installed ancient SeedSequence scheme:
[master_seed, 0xAE4A21, split_code, replicate], yielding separate ancestry/mutation seeds. Relate uses the existing SHA256 ARG-GNN/Relate/v1.2.4/{master_seed}/{split}/{replicate} signed-31-bit scheme. A development-v1 name is a ledger label, not a replacement seed function.

Before any data generation, scan every repository *manifest*.json, including partial manifests, experiment copies, .runtime/cache and pytest fixtures, plus all data/**/metadata.json so orphaned interrupted parent artifacts cannot evade the audit. Recursively collect declared simulation/inference seed integers. Reject any collision across the 240 proposed stream values or with prior stream values, including cross-stream collisions. Record every scanned file's path/hash and seed locations in the preparation freeze. At execution, repeat the current audit and retain frozen historical seed uses even if old files have disappeared. Only this development cohort's own output roots are excluded from subsequent scans, because they are the planned source—not prior independent parents.

A malformed prior manifest fails closed. The safety test removes its own deliberately malformed fixture after verifying this behavior; no valid historical manifest is ignored or deleted. Symlinks are not followed. If a final-confirmatory data manifest appears under data, the development audit refuses access without opening it. The wrapper has no arbitrary output, master-seed or split arguments and rejects changes to the fixed 64/16/TEST0 layout. Final confirmatory parents cannot be produced through this interface.

Default invocation is read-only preflight. --freeze writes the initial preparation execution manifest only and requires all four dataset output directories to be fresh. --execute requires the exact freeze-file SHA256, unchanged configs/source code/seed registry and a repeat seed/storage audit. This is an execution selector for root, not an extra user permission request. Relate's already recorded human authorization (“I reviewed, use it”) covers this research use; no new licence acceptance or redistribution occurs.

## Verified commands

Run on anogena inside /workspace-SR003.nfs2/ARG-GNN:

~~~bash
cd /workspace-SR003.nfs2/ARG-GNN
source scripts/project_env.sh
CUDA_VISIBLE_DEVICES='' "$ARGGNN_PYTHON" scripts/prepare_fresh_development.py --preflight
~~~

The default invocation without flags is equivalent. To create a new initial preparation freeze only, if it does not already exist:

~~~bash
CUDA_VISIBLE_DEVICES='' "$ARGGNN_PYTHON" scripts/prepare_fresh_development.py --freeze
~~~

The freeze generated during this task is root-reviewable. Its exact SHA256 and final preflight evidence are recorded below after sealing. Root may use separate stage invocations to bound CPU memory lifetime. These are executable commands for later root use; **they have not been run by this task**:

~~~bash
CUDA_VISIBLE_DEVICES='' "$ARGGNN_PYTHON" scripts/prepare_fresh_development.py --execute --stage oracle --freeze-sha256 5e19ff1650397ec657f5d90c341e4a223cf624154736f38f62908e42a0ff7ee1
CUDA_VISIBLE_DEVICES='' "$ARGGNN_PYTHON" scripts/prepare_fresh_development.py --execute --stage inferred --freeze-sha256 5e19ff1650397ec657f5d90c341e4a223cf624154736f38f62908e42a0ff7ee1
CUDA_VISIBLE_DEVICES='' "$ARGGNN_PYTHON" scripts/prepare_fresh_development.py --execute --stage boundary --freeze-sha256 5e19ff1650397ec657f5d90c341e4a223cf624154736f38f62908e42a0ff7ee1
CUDA_VISIBLE_DEVICES='' "$ARGGNN_PYTHON" scripts/prepare_fresh_development.py --execute --stage age --freeze-sha256 5e19ff1650397ec657f5d90c341e4a223cf624154736f38f62908e42a0ff7ee1
CUDA_VISIBLE_DEVICES='' "$ARGGNN_PYTHON" scripts/prepare_fresh_development.py --execute --stage verify --freeze-sha256 5e19ff1650397ec657f5d90c341e4a223cf624154736f38f62908e42a0ff7ee1
~~~

The default freeze path is configs/fresh_development_execution_freeze_v1.json. Root can instead run one --execute --stage all invocation from a completely fresh state. Do not mix an all-stage execution with already completed output directories. All source/model selection stages remain separate from this preparation wrapper.

## Outputs, verification and admission

The four dedicated output directories are:

- data/ancient_development64_v1
- data/ancient_inferred_development64_v1
- data/boundary_inferred_development64_v1
- data/age_inferred_development64_v1

Oracle and inferred manifests follow the existing loader-compatible schema. Successful parent records bind metadata, raw tensors, prepared inputs and adapter configuration. Raw census truth is stored only in the oracle audit parent and never copied to an inferred case. Native .sample order, both query haplotypes, original physical coordinates and external targets must remain aligned. No tail extrapolation or majority-truth filtering is allowed. Original 128 midpoint coordinates remain the source grid, with only the declared inferred SNP coverage mask.

Require Relate returncodes [0,0], finite physical coverage >=0.95, 120–128 retained grid points, positive packed-node count <=1,000,000, all 64 query haplotypes and all 24 references, no ancestry annotations in observed inputs, full-parent canonicalization, and full-parent electrical admission with strictly positive resistance and condition <=1e12. Full-parent checks occur before tiling. Missing ancestry classes, source proportions, tract purity/difficulty, model predictions and NLL are not admission criteria. Electrical admission is repeated through the existing boundary preparation's own checked path.

Every real stage writes an immutable timestamped receipt and JSONL event ledger under experiments/results/fresh_development_preparation_v1. Ledgers include parent-attempt start, label-blind admission/rejection observations and reasons, phase/source IDs, cache events and failures. A failed native inference keeps its native logs and partial parent outputs. A rejected parent stops the whole development stage; the wrapper does not silently remove, substitute, rebalance, renumber or retry it. A started parent with no completed admission is visibly incomplete. Root must review failure/recovery before proceeding.

A successful verify stage performs existing hash-bound raw/prepared loading, exact paired target/query/coordinate checks, canonicalization, complete tile coverage, verified boundary cache loading and verified age sidecar loading. It writes a **tracked compact completion.json** under the receipt directory with every oracle/inferred/boundary/age manifest SHA, protocol/source/preparation SHA, frozen source hashes, exact parent/tile counts, registry hash and ledger location. Root should require this completion receipt before freezing candidate or baseline training configs. It reports no learned ancestry performance.

Existing output directories, even empty ones, fail safely. There is no automatic resume or overwrite. A later explicit failed-stage continuation adapter may be necessary after inspecting a preserved failure. The original scientific identity must survive any infrastructure retry; parent selection cannot respond to observed outcomes. The wrapper does not treat a partial manifest as completed input.

## Resource preflight and realistic limitations

The current 20-parent source artifacts total 6,039,603,598 logical bytes across oracle, inferred, plain boundary and age caches, approximately 0.302 GB per parent. The development plan reserves **twice** this measured per-parent cost across 80 parents, about **48.3 GB**, plus **10 GB** scratch and **20 GB** margin for other concurrent project writes. The storage scan conservatively uses max(logical size, allocated blocks), includes the environment, does not follow symlinks and may count hard links repeatedly. It strictly rejects projected usage at the 450 GB warning threshold; 500 GB is the absolute cap. Filesystem free space is checked separately. The full reserve is retained at every stage/parent refresh, intentionally conservative even when earlier developmental outputs already exist.

The first preflight read 609 manifests/metadata files, found 182 distinct prior declared seed values and no collision among 240 proposed values. Its conservative current storage was 24,235,394,429 bytes and projected total with all development reserves 102,552,223,213 bytes, below 450 GB. Final post-test evidence is appended below.

The existing boundary/age loaders are eager. Preparation requires at least **40 GB** MemAvailable CPU RAM, sets CUDA_VISIBLE_DEVICES empty and fixes four CPU threads. The source panel's observed native inference took about 21 s total and complete inferred preparation about 333 s over 20 parents, but full cache construction, integrity scans, canonicalization, repeated admission and larger parents can dominate. Repository-wide NFS scans may take tens of seconds each. No wall-clock guarantee is claimed. No dense oracle-boundary cohort or perturbation cache is requested here. Root schedules all later GPU training under the two-GPU project rule separately.

## Validation record

After the final admission/ledger changes, the dedicated safety suite and existing Relate protocol binding suite passed **26 tests in 3.73 s**, with CUDA disabled. The tests are synthetic safety checks and reads of existing pilot protocol manifests; they simulate no chromosomes and execute no native Relate command. They cover proposed registry determinism and uniqueness, prior/partial/orphan provenance, held-out access refusal before reading, malformed-manifest failure, storage margin and warning stop, coverage/node/native admission, fixed development counts/masters, symlink escape, explicit freeze/source drift, default no-write preflight, empty-output refusal and durable parent failure ledgers. These tests do not establish empirical ancestry benefit.

Root must review the execution freeze and completion facts. Main-paper claims still require the separate model-selection and independently frozen confirmatory evaluation design.

## Final execution-manifest seal

Preparation execution freeze: configs/fresh_development_execution_freeze_v1.json.

SHA256: 5e19ff1650397ec657f5d90c341e4a223cf624154736f38f62908e42a0ff7ee1.

Sealed at 2026-10-10T05:56:15.874349+00:00, after the final safety tests and full read-only preflight. The seal scanned 618 prior manifest/metadata files, retained 186 distinct declared prior seed values and found no collision among all 240 proposed stream values. Conservative current repository storage was 24,247,899,910 bytes and projected storage including development/reserves was 102,564,728,694 bytes. All four development output directories were absent at sealing. No development or final-confirmatory data was generated.
