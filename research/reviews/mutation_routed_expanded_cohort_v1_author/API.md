# H038 full-author-panel PyG implementation API

The new modules are source-ready and inactive. Root owns generation, preparation, source review, storage/memory admission and launch. The current model config requires `execution_guard_ready=true` and a positive root-admitted `output_cap_bytes` before fitting; both are deliberately unprovided in the source-only config. H036/H037 active or historical sources are untouched.

## Frozen condition

Root registration is `research/reviews/expanded_full_panel_v1_root/registration.json`. Generation config is `configs/expanded_full_panel_v1.json`. They bind48 independent parents:32TRAIN,8internalVAL,8reserved outer. Fitting requires all40 development observations and receives no outer target bindings.

The panel is952haplotypes/476diploids, with266pure-source proxies (Ana50/CHG28/WHG102/EHG86) and686 queries (Present182/Bronze162/BAA14/Neo302/Yam26). Original author dates/order and observed-only class eligibility survive. Queries including BAA have unknown input ancestry. Native observed Relate post-five-size-iteration trees are joined through actual sample IDs/dates and complete native holds. No true genealogy or population/ancestry annotations enter the model.

Both arms use seed101 and identical width32/29764-parameter/38-state-key PyG modules. `aligned` uses genuine branch concordance. `zero_concordance` zeros only the first3 true-support fields; callable count and empty flags (3:5), geometry, site inputs, labels, GRU and learned capacity remain identical. This is a declared feature ablation. The old unused placement shuffle is not computed in the new adapter; its dataclass slot is an explicit identity placeholder, then the ablation is prepared once on CPU. No sham-mechanism or novelty claim is made.

One update is one complete independent TRAIN parent. Each pass has32 updates. Maximum30 passes/960 updates, with selection at5/10/15/20/25/30 passes (160updates apart). Selection maximizes equal-parent Present182 exact-span accuracy, then minimizes NLL, then retains the earliest pass. Stop after3 scheduled checks without a strict primary-accuracy improvement. Lower-NLL ties may improve the selected checkpoint but do not reset accuracy patience. Completion records actual passes/steps and does not claim equal optimization exposure when arms stop differently. Historical H036/H037 NLL selection is unchanged.

## Observed preparation and cache

`mutation_routed_full_panel_observations_v1.py` exports:

- `validate_registry_v1(registry)`: exactly32/8/8 distinct IDs and96 distinct ancestry/mutation streams.
- `load_full_parent_v1(parent_row, entry_ref, reader, *, scratch_directory)`: strict generated observed entry, roster/date/native tree/VCF/grid joins.
- `prepare_full_collection_v1(registry, entry_refs, reader, *, scratch_directory, include_outer=False)`: prepares development by default without receiving any targets; returns parents/registry/entry_refs/sham_bin_edges/receipt. Reports actual observed-preparation elapsed time.
- `validate_full_bank_v1(bank)`: full952/686/266 production contract.

The generator emits `full_panel_observed_entry.json` and an observed inference wrapper in exactly this contract. Entry refs are source_roster/vcf/site_coordinates/scoring_coordinates/native_tree/derived_roster/inference_receipt/observation_manifest. Explicit dates, original2Mb coordinates, native coverage/offset and retained fixed100bp indexes must match. Numeric helpers use independent namespaces with narrowly recognized count-contract edits; production calls remain `fixture=False`. Frozen modules are not mutated.

`mutation_routed_full_panel_cache_v1.py` exports:

```python
manifest_ref = write_full_cache_v1(
    observed_bundle, directory,
    source_refs=exact_python_source_records,
    guard_exact_bytes=root_write_guard,
)
index = read_full_cache_index_v1(
    manifest_ref, reader,
    registry=registry, entry_refs=entry_refs, source_refs=source_refs,
)
item = read_full_cache_parent_v1(parent_id, index, reader)
```

The writer accepts only observed model/scoring/receipt fields. It compresses/publishes one full parent at a time, guarding its actual compressed bytes, and publishes the manifest last. It performs no budget registration. Partial publication is incomplete and must not be reused. Source provenance accepts implementation Python files only, so targets cannot be smuggled through source-ref reads. No pickle or score target cache is accepted.

`FullPanelParentDatasetV1(manifest_ref, registry=..., entry_refs=..., source_refs=..., parent_ids=..., worker_tmp_directory=..., allow_outer=False)` works with the existing twoCPUspawn `make_parent_dataloader_v1` and explicit idempotent worker-PID drain. It yields whole PyG Data items without rebasing indexes. Training reads40prepared CPU banks once; it never repeats raw VCF/tree decoding per update.

## Fit input and execution

Source: `src/arggnn/train_mutation_routed_full_panel_pyg_v1.py`.
Config: `configs/mutation_routed_full_panel_pyg_v1.json`.

Input schema `mutation_routed_full_panel_pyg_input_v1` requires root_authorizer=root_agent, fit_authorized=true, pyg_authorized=true, status=active, arm aligned/zero_concordance, seed101, device=cuda, a canonical job_id, fresh output_directory, original created/expires UTC fields and positive max_cpu_rss_bytes/max_cuda_reserved_bytes.

`refs` must contain config/runtime/storage/registration/generation_config/observation_cache/target_bindings path/SHA records. `observation_entry_refs` contains exactly the32TRAIN+8VAL entry records. `observation_cache_source_refs` matches the observed manifest's source records. `source_sha256` binds all five new source modules, the unchanged PyG model and all reused observation/model/cache/score/helpers, the expanded producer and closed-scope storage adapter. The source map must equal `ALLOWED_SOURCE_PATHS` exactly, including the base byte-reader implementation `census_tsinfer_observation_v1.py` and the exact root CPU cache-producer path `src/arggnn/prepare_expanded_pyg_observations_v1.py`; extra paths and omitted paths are rejected before any source hashing or reading. All fourteen helper names bind fixed Python paths and the same hashes. The five H038 `source_refs` must match exactly. Observed-cache source refs can only select already approved Python modules; they cannot widen that closure. The exported pure `validated_source_records_v1(spec, cfg)` supplies root with the sorted exact record set.

Before metadata reads, `validated_metadata_ref_v1` requires canonical relative nontruth JSON paths and exact path/SHA records. `validated_input_refs_v1` requires the exact seven metadata roles, with fixed config, registration and generation-config paths. The input capability and base-fit config are also checked before reading; generation receipt and observed-entry roles are checked before their reads.

The target binding JSON has `parents` for exactly40development parents. Each row contains split, truth_npz and generation_receipt. The truth path must be `generation_config.output_root/parent_id/truth/full_truth_intervals.npz`; its hash must equal the same-parent generation receipt's original truth hash. Query identities are independently checked against observed scoring rows when truth opens.

All40 observed banks are durably sealed ready before TRAIN truth access. CPU TRAIN targets are separate pinned FP32 tensors for loss only, stored as [native_hold,686query,4class]. `native_hold_train_mass_v1(parent, bank, target, helpers)` uses the unchanged exact interval/bin intersections, maps each scored bin through `bank.readout_hold`, and sums its class spans directly into native holds in FP64 before one FP32 conversion. It never constructs the dense grid target mass. Every query's mass per hold must equal the sum of original scored-bin widths assigned to that hold. Native holds without scored bins keep zero mass.

`MutationRoutedFullPanelPyGGNNV1.forward_train_holds(bank, control=arm)` calls the complete inherited tree encoder and ordered bidirectional GRU, with the same aligned/shuffled internal route, then selects the686 query columns once per native hold. The training loop uses those logits with the same `exact_span_loss`. Algebraically, each hold's log probability multiplies the sum of its original grid-bin class masses, preserving the objective and gradients; ordinary FP32 summation order can differ. Full tree/GRU history and all holds remain present. The existing `forward` and every VAL seal/scoring call remain on the original grid. At the previously observed903–995 holds and a complete20k-bin grid, expected TRAIN target storage for32parents is0.317–0.349GB rather than7.025GB; actual fresh-parent shapes/memory/timing remain root measurements. Actual GPU free/reserved bytes and root bounds set complete-parent residency capacity after workspace allowance. If all40 banks fit they stay resident; otherwise a bounded LRU uses a separate copy stream to prefetch the next complete bank without changing full-parent history. There is no per-step raw preprocessing. A complete bank that cannot fit admission stops the run. First actual update timing/peak memory counts within the registered budget, with no dummy update. The actual post-update parameter finiteness check concatenates the tiny29764-parameter tensors for one device reduction and one boolean synchronization per update. CPU physical-span bookkeeping uses FP64; model inputs, parameters, logits, loss and stored TRAIN targets remain FP32.

Execute only through root's existing admitted reserved GPU launcher:

```text
.runtime/env/bin/python -B -m arggnn.train_mutation_routed_full_panel_pyg_v1 --execute --input <path> --expected-input-sha256 <sha>
```

Default CLI prints source-only metadata and creates no CUDA context.

## Seals, metrics and timing

`RawSealOutputV1` uses root's `project_storage_closed_scopes_v1.PilotOutputV9`/budget_check_v9/validate_storage_v9 API, with journal_closed_scopes_credit_v1 and canonical-mutex fingerprint coherence. It saves raw uncompressed probability-only NPZs with original coordinates/support, fsync, exclusive inode identity and one immutable content hash. Readonly arrays from exactly that serialization and immutable bytes are retained for last/best only. Current scoring validates the sealed record, original support, full raw FP32 shape/probability values directly from those arrays, avoiding redundant NFS reads and NPZ CRC reparsing. External records use the normal verified byte reader.

At selection, all8 current VAL seals precede any VAL truth access. `FullPanelPrimaryValidationV1.score` caches separate Present182 physical mass once, after seals, and calculates only accuracy/NLL. Raw zeros with positive mass retain infinite NLL. No floor, renormalization or bin/purity change is made. `full_final` uses unchanged generic full_span_score exactly once after selection for Present182/Ancient504/all686 and ancient-cohort class, Brier, ECE, boundary and short-tract diagnostics.

TRAIN rows are buffered by pass and written only at evaluation. Evidence reports observed preparation/cache-load time, actual training time, validation time, final diagnostic time, peak CPU RSS/CUDA allocated/reserved memory, actual completed passes/updates and query-bp metrics. Selection timing separates pre-bookkeeping, VAL forward, CPU copy, raw seal, primary scoring, checkpoint and post-bookkeeping, with the first real update/selection retained as sampled phase timings. These are measurements to be produced by actual execution, not claimed performance gains.

The fit worker accesses no outer observations or truth. A separate future outer stage requires the externally supplied path/SHA ref for `research/reviews/expanded_full_panel_v1_root/outer_score_policy.json`. That policy does not currently authorize any execution. The public `VerifiedOuterProductSealsV1` dataclass carries records, exact product-to-parent mappings and observed parent metadata; its type and declared cardinality grant no authority.

The policy schema is `mutation_routed_full_panel_outer_policy_v1`, hypothesis H038, root_authorizer=root_agent, status=frozen and outer_evaluation_authorized=true. It binds the current frozen registration/generation refs, ordered `outer_parent_ids`, externally frozen ordered `required_product_names`, `outer_parent_support_refs`, `outer_target_bindings` and `product_seal_refs`. Products are not invented by this implementation. The support records must be `research/reviews/expanded_full_panel_v1_root/outer_support/<registered_parent_id>.json`; schema `mutation_routed_full_panel_outer_support_v1` binds parent_id, split=outer, observations_only=true, truth_or_oracle_read=false, positions, edges, retained_indexes, query_slots, raw_query_nodes and cohorts. These root-frozen observed support records are the authority for the same686 columns and physical support. Target bindings contain exactly split, truth_npz and generation_receipt.

`verify_outer_product_seals_v1(products, parents, reader, *, outer_policy_ref, required_product_names=None)` rereads/hash-verifies that policy, current full48 registration and generation metadata, exact registered eight outer parents, full product-by-eight record matrix, distinct prediction paths, complete observed support and all raw probability archives. It also binds each frozen target hash to same-parent generation provenance. Equal prediction hashes can be valid; reusing one archive path for multiple matrix cells is rejected. The optional caller product list must match the external policy and cannot define the authoritative list.

`FullPanelExternalTargetV1(..., purpose="OUTER_after_all_product_seals", sealed_records=proof, outer_policy_ref=external_ref, required_outer_products=None)` independently reruns that entire verifier before truth, compares the proof transport, the requested parent support and the frozen target binding, and only then opens target bytes. A manually constructed public dataclass, even with plausible product/parent cardinality, cannot skip any part of that gate. Future outer scoring should time this complete re-verification; it does not affect current fitting/VAL selection. No outer policy was created or read from project data during these source-only repairs.

## Verification and limits

`implementation_checks.json` records a handmade full952 star tree/one-site VCF, production preparation without fixture bypass, both actual PyG CPU forwards, preserved ablation fields/capacity, observed cache roundtrip with exact compressed guard bytes,40handmade full952 PyG Dataset items through twoCPUspawn workers with both original PIDs drained and repeat-confirmed, analytic Present182 NLL after all8seals, invalid query labels/seeds/missing seals/outer proofs rejected, accuracy/NLL/earliest selection ties, patience3 and immutable current-seal support/hash checks. It performs zero optimizer updates and reads no actual generated observation or truth arrays.

The follow-up `boundary_negative_check.py` and `boundary_negative_checks.json` preserve the original independent fresh-audit findings and record24 focused in-memory checks:22 meaningful rejection cases plus source/matrix sanity. They cover extra truth/unreviewed source refs, missing reader-base pins, helper/cache substitutions, role-specific metadata path substitutions, public proof forgery, missing/unauthorized policy, incomplete matrices, foreign parents/support, reused archives, a final invalid archive and wrong generation provenance. The final-archive and generation-provenance negatives verify all16 handmade product-parent seals before failing, without opening any truth bytes. No generated observation/truth arrays, optimizer updates, GPU launches or CUDA context were used.

The subsequent `native_hold_train_mass_check.py` and `native_hold_train_mass_checks.json` contain one tiny synthetic FP32 comparison for both arms, with4native holds,7scored bins,686queries, interval boundaries crossing bins and one retained unscored hold. Independent grid-mass accumulation matches direct hold accumulation. Old grid and new hold readouts share the same handmade hold logits, yielding loss difference2.384185791015625e-7 and maximum full hold-logit gradient difference1.1641532182693481e-10, within ordinary FP32 tolerance. This check performs no optimizer step, tree/GRU fit, actual array read or CUDA work. Observation/cache source bytes are unchanged by this optimization.

Actual full952 GPU forward/backward memory/time, full40 CPU-bank memory, production dataset/cache creation and broad accuracy are unqualified here. Original author date/template text was read to construct the handmade schema. The current config remains inactive until root assigns an actual raw-seal storage cap and admits complete data/runtime/source closure.
