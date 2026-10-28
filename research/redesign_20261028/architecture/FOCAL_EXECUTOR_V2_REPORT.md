# Separate focal executor candidate: implementation and FP32 qualification

The candidate in `window_model_focal_executor_v2.py` preserves frozen v1's
observations and learned modules, and reduces computation to focal target rows
after the first source-tree sweep. It is a numerical execution candidate, not
a new scientific model. Frozen v1 and the running pilot were not modified.
All work in this report used authorized anogena CPU execution. No GPU,
ancestry-label fitting, TEST input, or model checkpoint was used or created.

The tested candidate SHA256 is
`44af7456c82ba896c7c4c3c03708ae4536495a85a539f013ce79bfc7ebb2e786`.
The frozen reference remains
`e0528c8e0d01b69512fe588dc74ec966a3855489abf4160a8ef4b9111e5a4e12`.

## Changes and API

The model subclasses the frozen class and inherits its complete state_dict.
The exact observation whitelist and coordinate adapter are reused. Preparation
stores immutable source-to-focal node, tree, cut, level and window routes.
Each selected focal tree retains every node, leaf and SNP. Multiple focal rows
on the same tree share its execution. Coordinate source duplicates retain
separate occurrences, weights and positions. No learned activation is cached
across calls or updates.

In structure and leaf_only modes the first sweep still encodes every source
tree. Temporal kernels contain only requested focal target rows, while keeping
every source column and its quadrature weight. Interval kernels explicitly
retain the different self-integral formula. Cut transport projects all source
cut states, then pools only target leaves into the corresponding complete focal
cuts. The second sweep runs on complete focal trees. In no_cross mode only
complete focal trees and their SNPs are encoded and swept.

For K=5 with one focal tree per window, structure/leaf_only sweep work changes
from 2K complete-tree occurrences to K+1. Each temporal kernel changes from
K by K to 1 by K. No_cross sweep work changes from K to 1. These operation
counts are not wall-time guarantees; SNP encoding, routing, depth-wise memory
traffic, autograd and CPU matrix-kernel choices remain relevant.

Use one module for preparation and model construction:

```python
from research.redesign_20261028.architecture import window_model_focal_executor_v2 as wm
observations = wm.load_observations(packet_path)
batch = wm.prepare_observations(observations)
model = wm.WindowTreeLoom(width=32)
model.load_state_dict(frozen_model.state_dict(), strict=True)
logits = model(batch, mode='structure')
```

`wm.prepare_executor(frozen_batch)` constructs the plan from an existing frozen
CPU Batch. `batch.to(device)` transfers immutable routes; GPU execution was not
qualified. The candidate Batch contains `.source`, `.focal`, `.trees`, `.nodes`,
`.cut_indices` and `.windows`. Its type differs from frozen Batch, and metadata
inspection should use `batch.source`. `core_logits` and `core_focal_mask` retain
their output behavior. Empty focal lists use the frozen execution path.

## Gradient connectivity correction

A genuine first-attempt defect occurred when no_cross selected only edgeless
focal forests while omitted source trees contained edges. Frozen v1 allocated
zero gradients for edge-only modules, but pruning made their gradients None.
AdamW treats these differently for decay and existing momentum. The candidate
now adds a zero-valued dependency on exactly the omitted edge-only modules in
that case: up/down gates, up/down messages, downward update and downward norm.
This preserves parameter connectivity without constructing omitted trees'
learned states. Cut and temporal modules remain disconnected in no_cross as in
frozen v1. Direct focal SNP encoding already preserves site_mlp connectivity,
including when genotypes are missing.

`FOCAL_EXECUTOR_V2_SEQUENCE.json` records a two-step no_cross check: first an
edgeless focal forest with no pooled sites and missing focal genotype, then an
edged focal tree with sites. The first step matches bitwise. The second maximum
parameter difference is 6.1467e-8 and optimizer-state difference 1.8627e-9.
The independent review receipt `../review/FOCAL_V2_NUMERICS.json` additionally
passes the stronger edged/edgeless/edged sequence, exercising nonzero momentum
during the middle zero-gradient batch. Gradient-None patterns and optimizer
step counters match at all three steps, with maximum final parameter difference
5.9605e-8.

## FP32 results and retained failure

`FOCAL_EXECUTOR_V2_PARITY.json` covers 11 fixtures in all three modes, 33 paired
checks. Fixtures include forests, polytomies, old singleton samples, a genuine
one-leaf empty-site tree, missing focal calls, no focal rows, arbitrary node
renumbering, interval self terms, repeated focal rows, repeated coordinates and
containing trees, mixed interval/coordinate windows, one real uniform packet,
and four distinct registered fitting observation packets. Fresh identical
width-32 models are used for every case/mode; no fitting labels enter the
numerical probe. AdamW uses lr=0.001, weight_decay=0.01, eps=1e-8 and global
norm clipping to 0.5. The fixed linear-plus-squared-logit probe activates
clipping; the empty-output fixture has zero loss.

All 33 checks pass the declared logit, gradient, gradient-None status,
optimizer-moment and post-update-logit comparisons. Maximum observed errors:

| Quantity | Maximum absolute difference |
|---|---:|
| Initial logits | 4.47035e-8 |
| Unclipped parameter gradients | 5.96046e-6 |
| Parameters after one clipped AdamW step | 5.12600e-5 |
| Logits after that step | 1.01328e-6 |
| Numerical probe loss after that step | 2.98023e-6 |

The original strict parameter criterion, atol=3e-6 and rtol=2e-5, passes 28 of
33 comparisons. Five strict failures remain: interval repeated-focal leaf_only
(5.1260e-5), duplicate-coordinate structure (6.7540e-6), repeated-focal
no_cross (3.4254e-6), permuted-node structure (7.3016e-6), and real four-packet
leaf_only (5.5470e-6). The receipt deliberately says `all_passed: false`.
The initial failed log/receipt and the second full receipt are retained.
The qualification script reports failures in JSON and continues to inspect
all cases; a successful process exit is not a claim that strict parity passed.

The worst case is a near-zero component of leaf_context.0.weight. Its raw
FP32 gradient is 5.9575541e-8 in v1 versus 1.5279511e-8 in v2. Both global
gradient norms are 39.411533. Clipping produces 7.5581358e-10 versus
1.9384569e-10. AdamW's eps=1e-8 makes that tiny difference material relative
to the component's update. Reducing the evaluated matrix shapes changes FP32
roundoff and reduction order. The measured post-step output and loss errors
are small, but this does not establish bitwise or long-trajectory optimizer
state equivalence. Tolerances were not loosened to erase the failure, and no
FP64 equivalence study was undertaken.

On the four real packets, structure passes the strict parameter criterion
(maximum 4.3586e-7), as does no_cross (1.8161e-7). Their post-step logit maxima
are 4.8429e-8 and 2.9802e-8. Leaf_only's strict parameter difference of
5.5470e-6 corresponds to post-step logit difference 5.9605e-8 and probe-loss
difference 2.0862e-7. These measurements qualify a bounded FP32 numerical
behavior, not ancestry accuracy or a guarantee of identical future training.

The independent review used a separately stated 1e-5 scale-normalized
functional/VJP engineering budget. Its worst synthetic full-VJP relative L2
error is 1.0509e-7; on the four real packets the three modes have errors from
3.0651e-7 to 3.4279e-7. All reviewed functional checks pass that budget. Its
verdict is "Qualified functional FP32 equivalence for reviewed CPU cases;
strict optimizer-parameter equivalence is not established." The budget is an
engineering tolerance for accumulated arithmetic, not a mathematical bound
or a replacement for the retained strict parameter failures.

## Small real four-packet CPU measurement

The observation inputs are registered fit draws 0 through 3, loaded without
labels. Together they contain 20 full source tree occurrences, 38,060 nodes,
952 samples per tree, 360 SNP occurrences and 4 focal rows. The focal plan
contains 4 complete trees and 7,612 nodes. Packet hashes are in each
`FOCAL_EXECUTOR_PROBE_v*_*.json` receipt.

Each version/mode used a fresh process, width 32, two Torch CPU threads, one
warmup and three measured forward/backward repetitions of a squared-logit
probe. There were no optimizer updates. Timing medians exclude loading and
preparation. RSS is the process high-water mark including imports, arrays,
preparation, PyTorch allocation and the warmup; it is not isolated tensor
memory. The shared host and filesystem cache were uncontrolled, and the v1
pilot remained running. This is a small measurement, not a throughput study.

| Mode | v1 forward/backward seconds | v2 forward/backward seconds | v1/v2 | v1 peak RSS GB | v2 peak RSS GB |
|---|---:|---:|---:|---:|---:|
| structure | 0.60581 | 0.48943 | 1.238 | 3.580 | 2.164 |
| leaf_only | 0.56998 | 0.58105 | 0.981 | 3.548 | 2.305 |
| no_cross | 0.37166 | 0.12415 | 2.994 | 2.000 | 0.931 |

GB uses decimal bytes. Median forward and backward components separately are
recorded in the receipts; their sum need not equal the median total. Frozen
observation preparation took about 0.30 seconds per four-packet batch and the
extra focal plan about 0.0073-0.0082 seconds. Those routes can be reused for an
immutable packet/context. Leaf_only showed no speed improvement in this short
measurement despite reduced operation count and RSS. No GPU speed or memory
claim follows from this CPU result.

## Scope and disposition

Keep the existing pilot on frozen v1. The candidate provides exact dependency
pruning at the model-operation level and small measured FP32 output/VJP
errors, with a corrected optimizer-connectivity corner case and documented
strict AdamW failures. It should remain separately versioned; any later use
must identify the executor and numerical limits rather than claim identical
training trajectories. The sampled-coordinate underflow risk and scientific
identifiability limitations documented for v1 are inherited unchanged.

## Device preparation handoff

No GPU was launched for this work. For nonempty focal batches, inspection finds
no candidate-only CPU operation inside forward: window kinds are cached Python
booleans; source, target and output routes are device tensors; temporal code
uses no Python conversion of device scalars or runtime unique/nonzero calls.
All plan preparation is deliberately CPU-only. A profiling harness can use:

```python
from research.redesign_20261028.architecture import window_model_frozen_v1 as v1
from research.redesign_20261028.architecture import window_model_focal_executor_v2 as v2
observations = v1.collate_observations([v1.load_observations(path) for path in paths])
cpu_batch = v1.prepare_observations(observations)
cpu_plan = v2.prepare_executor(cpu_batch)
batch_v1 = cpu_batch.to(device)
batch_v2 = cpu_plan.to(device)
model_v1 = v1.WindowTreeLoom(width=32).to(device)
model_v2 = v2.WindowTreeLoom(width=32).to(device)
model_v2.load_state_dict(model_v1.state_dict(), strict=True)
```

This is device-transfer API guidance, not GPU qualification. Inherited
`bases` still uses dynamic boolean indexing of parent edges before bincount,
which can synchronize CUDA; inherited Python per-depth PyG dispatch and
full-field copies remain. Empty-focal fallback uses v1's original temporal
routing, including its device-scalar conversions. Transferring source and focal
Batch objects independently also duplicates some shared observation tensors on
a new device; this does not alter their values but belongs in memory accounting.
