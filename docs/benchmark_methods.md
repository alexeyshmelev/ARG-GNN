# Benchmark method adapters

The registry in `configs/benchmarks/methods.json` describes real source-pinned
implementations and supported scientific protocols. Discovery and planning use
only the standard library. They do not import Torch/TensorFlow/PyG, inspect a
device, open a dataset, construct a model or launch a process.

Real execution is held for later explicit user approval and a fresh actual
admission. A registry job specification is an **inert pre-admission plan**.
The scheduler is fake-tested with injected validators. Production enqueue currently rejects every scientific family until its real admission adapter is integrated and reviewed. A generic normalized receipt is insufficient.
Historical configurations, source passes and leases grant no new authority.

## Methods and scientific scope

| ID | Implementation | Compatible protocol | Interpretation |
|---|---|---|---|
| `treeloom` | Actual H050 PyTorch/PyG model, corrected C3 history, original C4 model with optional bounded history | American C3, prospective/expanded European C4 | Our frozen model with explicit population/panel bindings |
| `argformer` | Official RoPE/FlashAttention2 MLM, contrastive pooling, embeddings and retrieval | American C3 | Author code on a common-data reconstruction |
| `argmix` | Official Rust extraction, TMRCA transformer, overlapping BiGRU and refinement | Prospective/expanded European C4 | Current release, distinct from unavailable April-paper architecture |
| `flare` | Qualified native FLARE 0.6.0 Java driver and existing observation/sealing adapters | American C3 | Native algorithm on common reconstructed genotypes |
| `ancestralpaths` | Released 5×9 ancestor features, Dense512/dropout/six-path softmax | Expanded European C4 | Released Dense model with separate derivedC4 endpoint |
| `gst` | Existing independent span-token graph-set architecture and feature projection | Expanded European C4 architecture control | Task/architecture adaptation, not native LAI author code |

No row claims exact published numerical reproduction. Each descriptor records
author/source pins, native settings, adaptations and blocking requirements.
Common-data comparisons measure pipeline performance. Differences in native
representations/supervision limit architecture-only attribution.

The European protocols preserve their actual independent-parent registries
before sampling. The prospective panel has5 TRAIN/1 validation/3 IID test parents
of 35,178,458 bp. Its five proposed misspecified parents are outside this IID
protocol. The expanded panel has32 TRAIN/8 validation parents of 2 Mb. Eight
reserved outer parents are deferred and `test: []` grants no TEST access.
Both protocols remain blocked pending artifacts, resource qualification and
an approved scientific recipe.

## API

```python
from arggnn.benchmarking.methods import list_methods, get_method, build_job_spec

descriptors = list_methods()
argmix = get_method("argmix")
plan = build_job_spec(
    "argmix", "fit",
    protocol_id="european_c4_prospective1611",
    artifacts={},
    output_directory=".runtime/tmp/my_future_argmix_run",
)
assert plan["execution_authorized"] is False
print(plan["missing_artifacts"])
```

`get_method()` exposes `id`, `label`, `protocol_ids`, `requirements`, `stages`,
`pipeline` and `source_refs`. A stage can have narrower compatibility than its
method. The protocol-specific ordered pipeline is in `pipeline_by_protocol`;
individual update APIs are not an additional campaign to run. The prospective ARGMix observation adapter hard-pins five TRAIN/one VAL,
so it cannot prepare expanded H038 data. Native author training accepts both
registered four-class native input recipes.

`verify_method_sources(id)` authenticates registered source/config bytes.
`resolve_stage(id, stage)` authenticates sources and lazily returns the actual
callable without invoking it. Numerical stages retain their original observed,
truth, writer and resource arguments. No generic stage invents a score.

`build_job_spec()` accepts named `{path, sha256}` artifact refs. It does not open
payloads or authenticate authority. Missing inputs leave `argv` absent.
Complete native-process inputs produce an argument list, never a shell command.
The ARGMix fit plan invokes unchanged `train_joint_model.py` with `--config`,
`--train-pt`, `--val-pt` and `--checkpoint-output`. It never emits the obsolete
GPU2/3 wrapper. A future supervisor must bind physical GPU0, process title,
qualified native Torch/Rust environment, all output/cache/TMP scopes and the
inherited audited lease before author imports.

## TreeLoom facade

`arggnn.benchmarking.treeloom` contains no copied tree/GRU equations.
`build_model()` constructs the actual C3 CPU model with seed 70001, 49 trainable
tensors, 32,132 parameters and FP32. It creates no CUDA context.
`prepare_observation()`, `train_step()` and `predict_logits()` delegate to the
immutable preparer and corrected executor.

The exact corrected C3 manifest is:

```
research/reviews/TreeLoom_GPU_history_source_20261017_v1/source_manifest.v1.json
0b0013f0907ef1b9a2b7b37dda1e59230111761d19a55821e9c6489adde5c4eb
```

Its CPU FP32 source check and independent source pass are engineering evidence.
Actual guarded cold GPU0 qualification, the full 30 Mb update and measured fit
within 24 GiB allocator / 128 GiB RSS remain pending. This C3 manifest cannot
authorize European data or C4 training.

`build_european_model(protocol_id, execution_mode="bounded_history", ...)`
loads the pinned `FrozenEuropeanTreeLoomV1` equations in a private module.
The prospective condition already binds 3,222 haplotypes / 2,972 queries.
The expanded binding adds only 952 haplotypes / 686 queries / 2 Mb, retaining the
original 32,325-parameter four-class model. No historical file or public imported
module changes; the separate AP 200 Mb condition remains intact.

European bounded-history mode imports the pinned class-independent
`_Direction` custom autograd function, including cross-window state VJPs,
and calls the original `CutMemoryV1._history` pooling body. It evaluates
FP32 coefficients on CPU and streams at most 128 holds per recurrence/VJP window.
Trainable forcing and length-scale gradients stay connected. The tree encoder
and observation bank retain full-bank execution: **this does not bound total
European GPU memory**. C4 jobs require genuine complete-parent GPU0 qualification.
The C3 qualification is insufficient. No bitwise GPU parity or speed is claimed.

`make_european_bundle()` binds complete caller-prepared bank/episode/cut tensors
and observation identity/support. `european_forward()` calls the original model.
`train_european_step()` performs one complete-parent update with the existing
exact-span loss, original FP32 AdamW and clip 1. The caller separately supplies
authorized TRAIN target mass and owns the frozen 30-pass order, initialization,
prediction barrier and fixed-final checkpoint. Bundle construction accepts no
query truth.

## Native method boundaries

ARGformer has extraction, MLM, contrastive, embedding and retrieval stages.
Its saved-MLM contrastive8 route covers one phase only; a contrastive checkpoint
does not complete the method. A fresh full pipeline preserves the ARG-specific
vocabulary, full-path deduplication, native YAMLs and genuine initializer
lineage, then seals every focal prediction before scoring. The audited release
does not provide exact Table1 data/checkpoints.
See `docs/argformer_published_benchmark_audit_v1.md`.

ARGMix preserves native 15-reference/0-ancestor settings, TRAIN nearest-drop 1,
VAL drop 0, native edge support, first-SNP supervision and overlap/tail
refinement. Primary prospective training uses all eligible native observed-SNP
context. The earlier forced shared 100k-row fit is a distinct sensitivity
condition. Common target scoring reports native coverage.
See `docs/argmix_published_benchmark_audit_v1.md`.

FLARE retains min-MAC 50 and other native options. Incompatible small fixtures
fail instead of weakening settings. Existing conversion preserves phased
reference/query pairing; focal projection and native output rounding are
disclosed.

AncestralPaths retains its released six-way Dense model and final epoch5,
without added VAL early stopping. Six probabilities remain available.
Summing paths 1+5 and 2+6 produces a separately named C4 endpoint, distinct from
the BAA-excluded paper experiment. The code/paper CNN discrepancy remains in
`docs/ancestralpaths_native_protocol_v1.md`. No American C3 mapping is supplied.

GST exposes its actual constructor and source47 feature projection as an
architecture control. Its old biological fit hard-pins a 120-query pilot and
cannot serve as the expanded 686-query trainer. Expanded fitting remains an
explicit integration gap, with no dummy successful stage or native LAI claim.

## Verification and remaining work

`tests/test_benchmark_methods.py` uses source metadata, JSON validation, fake
artifact plans and mocked delegation only. It checks lazy discovery without
numerical imports, exact source pins, real callable definitions, incompatible
protocol rejection, GPU0-only native ARGMix plans, C4-specific TreeLoom plans,
GST labelling and blocked independent-parent protocols. It starts no training,
reconstruction, download, resource admission or CUDA.

Before real runs, bind actual observation/target artifacts, native runtimes,
complete prediction barriers and fresh admitted worker receipts. Freeze raw
zero-probability NLL/reporting and coverage semantics. No probability floor,
hidden support reduction or cross-protocol score comparison is allowed.
