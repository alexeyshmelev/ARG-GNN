# Census native qualification runner v1

H021 development-only source reader/qualification implementation. It does not execute original-paper data or genuine comparator experiments, and must not be presented as original-paper reproduction. The two previously reviewed prototypes, sealed biological adapter and independent review remain unchanged. Actual observation/model/target arrays, fitting, GPU work and actual storage/runtime admission were not executed by this author. A different fresh reviewer must qualify this new runner before a root admission can invoke either phase.

The runner is `src/arggnn/census_native_qualification_runner_v1.py`, its frozen method is `configs/census_native_qualification_runner_v1.json`, and its handmade suite is `tests/test_census_native_qualification_runner_v1.py`. Source and runtime receipts are in `research/reviews/census_native_qualification_runner_v1_author/`.

## Two explicit phases

`inspect` byte-verifies the original observed/native/prepared file closure, decodes observation-only inputs, joins every original tile through the unchanged adapter, and writes one small JSON evidence manifest. The default inventory is four original development parents, two TRAIN and two VAL, with eight original tiles each. Every cell is attempted. A failed join or physical event collision is preserved and prevents a complete inspection status. No model forward, biological target or fitted scaler occurs in inspection.

Successful inspection writes `inspection_complete_pending_root_freeze`, with32 per-tile certificate digests, original decoded subject/sample/readout lineage, connected/unused node counts, per-tree query-weighted group/singleton distributions, physical-span-weighted group sizes and opportunities, sample-set clade turnover and right-hold-span weights, older/equal-census proxy counts, forest height/depth and source memory estimates. Event counts distinguish anchor/boundary, anchor/readout and boundary/readout coincidences and count unequal frozen IEEE anchor coincidences. A single tree has unmeasured turnover. These are structural support observations, not predictive utility or measured memory findings.

The inspection output is evidence. Its freshly computed digests do not admit themselves for numerical use. The root must independently review and freeze the evidence file hash and expected per-tile digests in a new active numerical admission with a new reservation and CPU launcher.

`numerical` reads that externally frozen inspection, checks its source/observation/config lineage and tree-count inventory, then rebuilds the same native banks from the same byte-verified original inputs. Every rebuilt certificate must match the independently expected per-tile digest. Decoded lineage, structural summaries and collision counts must also match the frozen inspection. No bulk certified tensor artifact or duplicated geometry cache is persisted.

The unchanged metadata-only selector chooses TRAIN minimum/maximum original-tree counts, tying by parent ID then tile ID. Those distinct cells execute first. The remaining cells complete the fixed all32 mechanical qualification. VAL cells receive the same fixed target-free checks, with no outcomes, tuning or protocol adaptation. Every cell must cover float64 and float32 across proposed, eta0, one fixed global membership permutation, raw diagonal and local. Missing cell/control/dtype coverage prevents qualification. Numerical failure stops the phase and preserves completed cells and the failure.

Each dtype/control starts from the same source-seeded initial state, whose expected fingerprint is externally frozen in the new admission. No checkpoint/model array is loaded. Native tskit MRCA and census sample-set oracles independently verify donors/cuts. Numerical checks cover original scored sample/readout/target-column ordering, separate-tree versus forest outputs/VJPs, float32 versus float64 outputs/VJPs, every-layer positive matching mass and normalized source bounds without floors, whole-GNN float64 central directional derivatives, finite active VJPs, and one clipped disposable AdamW step per condition. Each model and optimizer is discarded. The synthetic objective uses fixed coefficients on mean logits and no biological targets. These steps are unit qualification, not fitting. Eta is structurally inactive in a one-tree cell with no turnover boundary, as well as in the eta0/local controls. Its absent gradient is explicitly expected in those cases.

## Root byte and lineage admission

The command accepts an externally expected admission file SHA256. That active receipt binds the shipped runner config, the exact runner and all25 source/config/protocol/prior-review dependencies, a fresh new runner review binding/full receipt, main Python/compiled Torch/NumPy/tskit runtime, root storage snapshot/reservation, CPU launcher, original native producer manifest, original prepared manifest and explicit feature lineage.

All admitted observation files are enumerated in an exact `observed_artifact_sha256` closure. The reader checks every original native tree, source/derived roster, producer completion, original grid and serialized x48 `.whole_observed.pt` file before any native/array decoder. It verifies the bytes again when consumed. Arbitrary extra files and truth/oracle/target paths are rejected. The old full-age `.bank.pt` geometry is neither decoded nor used. No supplied donor tensor can replace rebuilt native donors. Default Torch decoding uses `weights_only=True` and CPU mapping; grid decoding uses `allow_pickle=False`.

Original observed roster fields are exactly sample index, subject ID, haplotype, observed node ID, observed cohort, observed age, declared role and observed reference class. The derived roster must be the exact original row plus its declared source-to-inferred-node mapping. Original subject/haplotype pairing is checked in actual full-panel mode. The native samples exhaust the roster one-to-one and native float64 sample dates equal the declared observed dates exactly. All Q are strictly younger350 with no reference class. Older/equal350 R remain observed class proxies. Original post-five condition-specific tree/roster/coordinate/completion records and prepared sample slots, dynamics IDs, scored output indexes, original retained grid, tree IDs and reference landmarks must agree.

The sealed adapter traces every sample-ancestor union, excludes unused global internals, joins anonymous interiors by pinned native postorder and verifies all serialized fields before projecting F47 or suppressing anonymous unary rows. F47 remains exact stored columns0–6 and8–47. Producer replay is value-exact dtype/shape/stride equality, with the known signed-zero byte equivalence. Actual original artifact byte identity is separately established by this reader. Native dates/cuts/donors use direct float64 rather than encoded-age inversion.

Existing x48 and F47 use the original fixed reference transformation and no fitted scaler. A required feature-lineage record binds the prepared manifest, producer and reference-adapter hashes, exact F47 columns, original two TRAIN parent IDs, `identity_no_fitted_scaler`, and false input-label flag. Fitted, VAL-derived or arbitrary new transforms are rejected. F47 retains original age and TMRCA summaries; it is not age-free.

Physical positions remain original bp divided by2Mb. Every original100bp center, original tree and clipped250kb tile is retained. An exact boundary belongs to its right hold. Equal physical anchor coordinates with unequal frozen normalized IEEE arithmetic are counted and rejected by the sealed adapter. No epsilon, tile normalization, age change, geometry adaptation or support selection upon outcomes is allowed.

## Exact receipt contracts

The active receipt has schema `census_native_qualification_admission_v1`, status `active`, a distinct job ID, phase, UTC creation/expiry, `refs`, `source_sha256`, exact observed artifact closure, reserved output directory, expected certificate map, inspection manifest ref, initial state fingerprints by dtype and the frozen global Q permutation. All fields are required. Inspection requires the last four numerical fields to be empty/null. Numerical requires32 expected certificate entries keyed `parent_id:tile_id`, an externally hashed complete inspection, both state fingerprints and a valid complete Q permutation. Admission age and duration are bounded to24 hours; actual execution stops if its admission expires.

`refs` contains exactly `runner_config`, `runtime`, `storage`, `launcher`, `native_manifest`, `prepared_manifest`, `feature_lineage`, and `runner_review`. Every ref has project-local path and SHA256. The new review binding has schema `census_native_qualification_review_binding_v1`, status `qualified_source_handmade_CPU_only`, exact runner/config hashes, `actual_admission:false`, and `review_receipt_ref`. The full review receipt must bind those artifact hashes and have a qualified verdict. Fixture review receipts cannot admit actual mode. The current author receipt is pending independent review and cannot serve as that review.

Runtime schema `census_native_CPU_runtime_v1` binds exact `runtime_identity_v1()` plus hashes of the eight shipped interpreter/package/compiled runtime paths. The actual main interpreter is checked. Runtime symlinks are allowed only for these explicitly fixed installed tools; source/observation symlink files and paths resolving outside the project are rejected.

Storage schema `census_native_storage_reservation_v1` requires an active matching job/reserved directory, fresh UTC dates, complete snapshot ref, used project bytes, at least1GB reserved new bytes, nonnegative concurrent reserve and500GB project limit. Used+new+concurrent reservations must remain below450GB. The linked `census_native_root_storage_snapshot_v1` must be complete/successful, bind this root and used-byte count, and be at most one hour old. The runner performs no global storage scan. Evidence writes are exclusive, counted and limited to1GB. Verified native decoding uses a temporary file inside the reserved evidence directory, limited to half that cap and removed immediately after loading. Other model/input/autograd memory remains unmeasured and needs root resource admission.

The CPU launcher receipt is exactly schema `census_native_CPU_launcher_v1`, matching job ID, Python `.runtime/env/bin/python`, empty `CUDA_VISIBLE_DEVICES`, `PYTHONDONTWRITEBYTECODE=1` and `GPU_admitted:false`. One CPU thread is used. No GPU reservation is requested or created.

## Frozen command shapes

The root creates actual admission files and independently supplies their expected SHA256 values only after fresh review and resource/observation admission. Source inspection and numerical admission use separate commands:

```sh
cd /workspace-SR003.nfs2/ARG-GNN
source scripts/project_env.sh
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.runtime/env/bin/python -m arggnn.census_native_qualification_runner_v1 \
  --phase inspect --admission research/admissions/ROOT_INSPECTION_ADMISSION.json \
  --expected-admission-sha256 ROOT_FROZEN_INSPECTION_ADMISSION_SHA256
```

```sh
.runtime/env/bin/python -m arggnn.census_native_qualification_runner_v1 \
  --phase numerical --admission research/admissions/ROOT_NUMERICAL_ADMISSION.json \
  --expected-admission-sha256 ROOT_FROZEN_NUMERICAL_ADMISSION_SHA256
```

Those placeholder paths/hashes are command contracts, not executable admitted jobs. The output directory comes from its hashed active reservation. `initial_state_sha256_by_dtype` can be independently derived from `_model(shipped_config,dtype).state_dict()` through the sealed fingerprint, without opening a model checkpoint. The root must freeze the permutation before observations/outcomes are used for any tuning.

## Handmade author qualification and limits

The final author suite passed21 cases in23.09s, wrapper24.12s, using the main runtime, one CPU thread, empty CUDA visibility and author-owned pytest temporaries. The four-parent/eight-tile fixture uses eight sample leaves, two marginal trees over2Mb, one unused global node and heterogeneous Q-to-R MRCA donor weights. The full32 inspection→independent mock freeze→numerical handoff exercised the default native/NPZ/safe-Torch decoders with injected verified bytes and mock root/review/resource receipts. Numerical phase callbacks test coverage without duplicating expensive models across all toy tiles. A separate real small two-tree cell and a full280-sample one-tree cell each execute all ten dtype/control checks and disposable steps. The one-tree case qualifies structurally inactive eta.

Negative tests reject changed x48 bytes before any decoder, source/truth fields, wrong source/review/runtime/storage pins, stale admission, altered scaler lineage, missing original tile, wrong post-five condition, target/grid ordering, extra unadmitted artifacts, stale expected certificate, wrong initial weights, mutated donors, and filesystem symlinks/byte drift. One initial test assumed unrelated-parent decoders should not run after a corrupted parent. Preflight was strengthened to verify the entire global closure before any decoder, and its expected global failure was corrected. All failed/passing attempt receipts remain preserved. No tolerance was relaxed.

Mock fixture receipts are explicitly fixture-mode data, not actual root, runtime, storage or independent review admission. The default config cannot use small fixture counts, injected decoders/probes or fixture reviews to admit actual files. Actual32 support, real file/role/scaler joins, actual numerical cells, resource behavior, predictive benefit and fitted/scientific claims remain unrun and root-owned.
