# Independent review of ARGMix biological metadata repair v3

Verdict: PASS for this source-only metadata startup repair.
This does not establish successful repaired runtime fitting or any biological
performance result. The reviewer did not author the v3 repair. The reviewer
participated in the earlier v1 adapter integration, which is disclosed here.

## Conservation and source cause

The repaired module hash is
4a6c44dfd81ba3f75e74c5287a4b55776bb9fe1612732c457748643cd75d7b1f.
The config hash is
25829cdb811da918db4cb7fad5ee607a44610bd1f04bc40dbe2f5564a6d58eb2.
The unchanged runner hash is
4261fb1195d16f73370b5fa0144ba0f59afe6a67726af6567634b0bd282e223d.

Independent AST comparison against the exact preserved v1 producer snapshot
confirms only policy, stage, verify_backend, stock_train, and infer_and_seal
changed. reference_metadata_amendment is the sole new function. All remaining
original functions are unchanged. Config differences are limited to the
new compatibility amendment and the new owned-module hash. Native model,
joint transformer/GRU, objective, smoothing, augmentation, optimizer, seed,
selection, source archive preparation, external-label handling, runtime
qualification, and author execution retain their prior definitions.

All 86 immutable author files and their execution copies match source-tree
SHA256 4fbcc41801027983a9654920d143f053d5b74ffea020b751a13757c0ae327ae8
for current-native author commit c2481248eb0de5fda659d68f59c51e0e27becd20.
This remains an adaptation of that unchanged current-native release, with
no claim of exact April-paper reproduction.

The stock count function requires paths.references at
joint_interval_model.py:719-726. Stock training calls it at line329 before
archive loading. The resulting available_predict has exactly two load
sites, train_joint_model.py:347 and :599, both for logging. The log writer
records it in joint_training_data.txt at joint_interval_model.py:1193.
It does not affect model construction, sampler, archive choice, objective,
optimizer, scheduler, early stopping, or predictions.
The source config loader also checks path existence, which does not change
scientific input selection. Model-builder configuration and forward-source
metadata helpers do not consume the new pointer as a scientific input.

## First TRAIN metadata pointer and paired correspondence

The selected pointer is the already-bound observed references.tsv for
bio_paths_pilot_train_000. Its SHA256 is
341b99dd59d65fd9947b38738685aacefe58541ead9524ace7fe8b5a428639a3.
Independent checks read observed TSV and roster metadata only. Each of the
four references files has 280 distinct IDs0-279, QUERY120,
UNLABELED_REF64, and24 each ANA, CHG, WHG, and EHG. Every observed ID,
role-derived population, and age matches the respective bound roster.
The copied roster equals the source observed roster. All four references
files have identical bytes and the same count eligibility.

The stock count is120 per panel. It must not be interpreted as240 across
both TRAIN parent panels. Using the first TRAIN representative is
scientifically valid for the narrow stock count/log interface because every
panel has the same eligibility and no scientific input uses that count.
No query ancestry target contributes to the metadata count. Integer age
truncation in the stock reader cannot alter population membership.

Independent YAML comparison establishes that both original TRAIN pairs
and both original VAL pairs retain their exact bound tree, reference,
Relate sample, and native label-pointer correspondence. Truth and archive
contents were not opened or hashed in this review. The repair verifies
these bindings again at actual runtime before adding the representative
pointer. Inference still passes each actual VAL parent's explicit
references argument. The TRAIN metadata pointer cannot replace that input.

## Exact inherited product admission

The two inherited producer records are unchanged:

| Product | Path | SHA256 |
| --- | --- | --- |
| Observed frames | experiments/results/official_argmix_biological_pilot_v2/observed/observed_frames.json | b8de4b503bfebcf7eb71b6861879826f24731deb37aabe68b5283f6f07f07531 |
| Native archives | experiments/results/official_argmix_biological_pilot_v2/native/native_prepared.json | 2c15bd908ae89b81e1a8c9d07809e82c46d6b1b41a2031ac93013dc54534961c |

Both retain original config
beb49bccf0f4780d6d76a3e738e1cba85dc5a1de4c846811aea2a20962029819
and original module
cde961f2a9ed7aa98b6d24af644ef83d5657abed7ac5d8029afe46cebafc1665,
with the unchanged runner hash above. The native preparation record points
to that exact frames record.

Compatibility admits only these exact record dictionaries and original
kind/config/owned-source declarations. It rejects another path, changed
bytes, added record fields, another kind, or another original declaration.
It does not rewrite producer provenance. verify_frames and verify_prepared
are unchanged and still perform all bound-file checks during root execution.
There is no old GPU or old fitted-artifact compatibility exception.

## Mandatory root launch requirements

The source-only policy command passed at the repaired config hash.
Actual execution remains the primary agent's responsibility.

1. Register this mechanical metadata-interface correction before the run.
   Reuse the factual successful unchanged build and CPU FFI records already
   used by v2, with their exact paths and hashes. No source rebuild or
   preparation repeat is required.
2. Use the unchanged project launcher with one free allowed physical GPU2-5,
   maximum two project GPU jobs, a fresh output, and the exact process title.
   Run phase qualify-gpu under module arggnn.official_argmix_biological,
   --execute, and the repaired config hash above. The isolated ABI210/cu129
   runtime must establish actual full-source CUDA forward/backward and stock
   Rust TorchScript GPU fixture admission, own UUID/title/NVTOP, and current
   config/owned-code hashes.
3. Use that new successful GPU receipt and its factual hash for phase train.
   Supply the native_prepared path and hash in the table, with a fresh fit
   output. Do not run prepare-observed or prepare-native again.
4. Use the new current-code stock_fit.json and its factual hash for phase
   infer-and-score, together with the same new GPU admission and a fresh
   inference output. The old v2 GPU receipt or failed fit cannot substitute.

A handmade old-policy GPU receipt was rejected. A current-policy handmade
receipt was admitted only when both GPU qualification flags were present.
These tests validate admission logic without qualifying a real GPU.
The documented commands in docs/official_argmix_biological_metadata_repair_v3.md
match the runner and module interfaces. Root must fill actual existing build,
CPU, fresh GPU, and fresh fit receipt hashes at execution time.

## Evidence and boundaries

check_01 records a passed source-only policy invocation.
check_02 records five independent source/metadata/handmade checks passing.
check_03 records all19 author handmade/source checks passing when replayed
with temporary files contained only in this independent audit directory.

No biological arrays, tree tensors, sample/grid binary arrays, native
archives, target arrays, fitted checkpoints, outcome files, GPU calls,
native execution, installation, simulation, or Git operation were performed.
Only source, config, producer JSON provenance, observed TSV/roster metadata,
and paired YAML were inspected. Writes are confined to this independent
review directory on anogena.

No blocking issue was found within the requested source-only scope.
Actual new-code GPU qualification and fitting are still required. An actual
failure must be preserved and audited rather than bypassing any binding or
changing author scientific behavior.
