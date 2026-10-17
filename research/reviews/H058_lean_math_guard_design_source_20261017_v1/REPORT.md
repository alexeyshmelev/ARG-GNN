# H058 lean math guard: inactive source prototype

The concrete change is a 51-line integration diff plus a 155-line effect-gate helper. It removes **only the recursive accounting replay from the two `train_one` guard calls after genuine first-step qualification**. Each math boundary keeps the exact current authority/expiry/work cutoff, lease/title/UUID, RSS/CUDA, 24-hour timer, and **unchanged three-walk charged-scope census/cap check**. No live v7 source is changed. This packet is not a v8 admission, independent approval or performance qualification.

The earlier `H058_consumer_guard_hotpath_source_audit_20261017_v1` was read first. The H059 author supplied the recent performance design directly: a math-only guard argument, preserved other boundaries, first genuine steps for lazy initialization, and explicit file-effect analysis. ROOT then clarified that scoped census/cap detection stays at both math endpoints and that this work does not require a universal native-write sandbox. Applicable `AGENTS.md:18,88,234` protects the pinned runtime, science, original scopes and admitted live consumers. ROOT's explicit efficiency interpretation authorizes this fresh source design, with live v7 unchanged.

## Exact integration

`integration.INACTIVE.v1.patch` is against the three exact v7 files recorded in `integration_preview.v1.json`. It was applied to strings in RAM for AST/compile checks, never to live files.

1. Context adds `math_guard`, byte-for-byte the original `guard` statements except its final `g.budget_check_v9(storage,boundary=False)`. It retains `charged_census_v1(spec)` exactly. The original full guard remains untouched and stays in `ctx["guard"]`.
2. Only `scheduled_step_v1`'s argument to `h45.train_one` is routed through `MathEffectGateV1.run_step`. All five scientific arguments, `qualify=first`, `after_optimizer_step=counted`, both surrounding cache validators and first-step hooks/receipts are unchanged.
3. The worker environment disables CUDA disk cache, PyTorch kernel disk cache, Torch JIT and Torch compilation before Torch/CUDA import; bytecode remains disabled. No model architecture, objective, checkpoint/recomputation policy, tensor kernel option, optimizer, update order or runtime precision/determinism setting is changed.

The first real step of **each** arm uses the original full guard at both endpoints and keeps all genuine gradient/parameter/peak-memory checks and receipts. It counts toward the original training budget. There is no probe or extra update. Lean mode requires both first steps to have returned successfully. If the complete two-arm 1,920-update schedule succeeds, four first-step endpoints remain full and the other 3,836 math endpoints omit recursive historical replay. All endpoint censuses remain. This is a source count, not observed progress or a wall-time benefit estimate.

## Boundaries that remain full or unchanged

| Boundary | Exact v7 location | Preservation |
|---|---|---|
| Context/fit initialization, before scientific imports and model setup | `context_factory_v1.py:7–15,39–59,102–105`; producer `:84–189,330–334` | Original full validation/configure-fit checks and guard calls unchanged. |
| Reader and complete episode construction | context `:80–89`; producer `:259–278`; existing observation preparation | Existing reader, source, target phase, episode and storage checks unchanged. |
| Device parent and payload transitions | `observations_gpu_v1.py:13–62` | Existing full guard before/after device-parent transfer and all binding/preparation rules unchanged. No new full call is falsely attributed to payload/release sites that did not have one. |
| Scientific/cache checks around and inside math | producer `:217–220`; observations `:64–69`; H045 `train_one:287–328` | Both surrounding validators, F1 binding/version checks, complete-bank/FP32/finite loss/gradient/state checks, clipping, AdamW, gradient cleanup and callback unchanged. |
| Actual scientific writes | producer `:240–254,280–322`; selected `PilotOutputV9.save:499–524` | Original strict writer serializes to RAM, checks local scientific cap and full `boundary=True` accounting with exact planned bytes before exclusive open/write/fsync. Writer source/AST is not patched. |
| Operational writes | v7 common `compact_v1:400+`; root driver original write sites | Original prospective census/accounting checks and caps unchanged. |
| Supervisor polling | root driver `:245–268` | Existing owned-RSS, charged census, capacity/attachment checks and polling cadence unchanged. The original poll uses census/resource checks; no new claim that it replays full accounting every poll. Entry/final/accounting write checks remain as before. |
| Success/failure/checkpoint/pass receipts | producer `:280–326` | All existing output saves and exception behavior retained. Gate clears its active region in `finally` before any failure writer runs. |

## Actual eager Torch/PyG effect trace

`math_effect_call_index.v1.json` records exact spans and AST-derived direct calls. `input_pins.v1.json` hashes the project and installed framework source texts without importing those frameworks.

| Actual call path | Source evidence | Effect boundary |
|---|---|---|
| Forward, span-weighted objective, backward, clip, optimizer and gradient cleanup | H045 `train_one:287–328`; `biological_gst_control_v1.exact_span_loss:137–143` | Original tensors/module state/scalars and exact guard callbacks. The function has no output writer, reader, loader or subprocess call. The counted callback only increments an in-RAM integer. |
| Stage and site activation checkpoints | H058 CPU model `:33–46`; whole-tree `encode_sites:307–341`; packed same-marker `:411–439,476–500` | Actual non-reentrant `torch.utils.checkpoint` recomputation, retained as-is. Activation checkpointing is not file checkpoint serialization. The latter remains outside math behind the strict writer. |
| Real PyG sweeps | clade-flag aliases `:250–252` bind `anonymous_clade_gnn_v1:90–131`; `mutation_routed_pyg_gnn_v1:29–80` | Actual `MessagePassing.propagate`, tensor linear/gates, aggregation and recipient exclusion. This is not merely an nn.Module name. PyG `SumAggregation.forward:19–22` delegates to sum reduction; `_scatter.py:67–70` uses tensor `scatter_add_`. |
| Clade and same-marker injection, repeated sweeps, GRU/readout | integration hooks `:49–83`; ordered clade `:330–357,396–399`; clade flag `:273–304` | Tensor reductions, copies/index-add, two sweeps, genomic GRU and classifier remain exact. Binding checks occur during forward/recomputation too. |
| AdamW and lazy framework paths | installed optimizer `:71–98,472–492,959–975`; AdamW `:207–266,840–875`; checkpoint `:343–351,1460–1484` | Normal AdamW imports `torch._dynamo`; its wrappers use `graph_break`. Normal optimizer functions use inactive `record_function` contexts. These are retained, not treated as active profiling or compilation. First actual steps warm these paths. Any later unexpected audited compilation/exec or write fails after a full guard. |

## Unexpected effects and inherited descriptors

The helper checks that its process-local hook installation is observed; a silently rejected installation fails. Its Python audit hook is active only for later math calls. A write-capable open, file mutation, file-backed mmap, process launch, trace/profiler installation or dynamic compile/exec invokes the **original full guard, then rejects the effect before that audited operation**. A later import triggers a full check; a resulting compile/exec/write remains rejected. This is escalation/fail-stop, not permission for an unbounded writer. The hook is process-global during math, with a thread-local reentrancy bypass restricted to the exact original guard callbacks.

That bypass matters: the original lease validator performs an authenticated `O_RDWR` exclusion probe and resource/capacity work (`direct_gpu0_v1:209–236`), and the full accounting guard opens its original mutex. These mandatory checks remain unchanged and must not be rejected as model writes. Existing resource/title/UUID/RSS/CUDA semantics, including v7 identity corrections, are retained through the same `resources()` closure.

The actual driver uses stdout/stderr pipes (`root_driver_v1:229–230`) and passes only the launch gate plus physical/logical FDs; the gate FD closes at worker line 127. The remaining two lock descriptors are intentionally `O_RDWR` (`direct_gpu0_v1:179,190`). They stay open and retain their locks and identities. The traced model/loss/optimizer callback path does not use them for writes; the original resource callback reads/checks them and validates their named inode and contents. The helper rejects other open regular writer FDs, including regular stdout/stderr redirection, and checks those two exact identities at math boundaries. It neither reopens locks read-only nor treats arbitrary regular writers as exceptions.

Python audit events do not cover ordinary `os.write` on an already open FD or every native write. This design does not claim otherwise. Native CUDA/PyTorch cache controls are set before science; no model/data-loader/background writer or compile/profiler is introduced. Existing native computation and resource subprocesses remain trusted under the unchanged pinned runtime. Unexpected audited Python background writes are caught while the region is active. Arbitrary foreign native code, undetected native background writers and hostile callback injection are outside the stated trust scope, as they were outside the original before/after accounting mechanism's prevention boundary.

## Detection timing and remaining review work

Both math endpoints and supervisor polling retain the original charged-scope cap detection. The existing reservation, active/work/expiry checks and all source/ledger/write boundaries remain in force. Unrelated historical-ledger corruption or source-proof changes may now be noticed at the next original full boundary instead of the next math endpoint. That timing change is explicit. No syscall-complete sandbox, identical exception timing, native write-freedom proof or measured speedup is claimed.

The source design has no remaining blocker under ROOT's clarified trusted-runtime scope. It remains **unqualified and inactive**: the 23 focused AST/source assertions and compile-to-RAM check did not execute the helper, framework, guard, state check, resource probe or numerical step. A fresh independent review must inspect hook recursion/failure behavior, the original writable probe exception, exact runtime/environment compatibility and the concrete future source closure. Any runtime qualification belongs to the first scheduled real update, never an extra probe. An unexpected cold path or writer descriptor must fail rather than silently weaken a guard.

ROOT subsequently reported a new Census selector `fd05d8934ed29eed39f7e60121a33e48fbeb0106991e76690664ca96355692e4`, with child verification pending at the time of this audit. This packet does not capture state or switch v7's frozen Relate64 route. A later fresh successor must bind the genuinely verified/reviewed current selector supplied by ROOT. It must also fit the original 64-role ceiling honestly: ROOT requested a later authenticated operational-module consolidation of driver and producer roles while preserving separate namespaces/AST semantics and both functional checks. No full new namespace, role deletion, live migration, v8 admission or launch is implemented here.

All server writes are confined to this review directory. Earlier source audits, v7 sources, current records and scientific artifacts are preserved.
