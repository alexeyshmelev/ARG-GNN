# Edge innovation GNN v1 — implemented H029 discovery design

The source now implements a small learned operator: observed reference-cavity
changes update qualified child-carrier memory, and current-key memory reads enter
complete-tree messages before query gathering. It tests a census-specific coupling
of established primitives. TGN, tree cavities, recurrent graph state and Gated
DeltaNet algebra are prior art. Historical priority and predictive value remain
unresolved. No fit or actual-data GPU qualification is claimed by this document.
The original unimplemented proposal and its independent assessment are preserved
in the author/reviewer packets.

## Observation and readout contract

Retain every native vertex and original edge in every complete observed
sample-ancestor union, including context64, within each original250kb tile.
Insert one analytic point on every native edge crossing350. A crossing edge is
represented by its two metric segments with its original-child carrier provenance.
Native IDs remain qualified joins, never learned embeddings or true-ancestor
assertions. Actual sample identities join the complete observed280 roster and
the exact frozen-emitter Q120 columns.

Use only the96 observed pure-reference proxy classes and observed dates as
reference signal. Each younger reference contributes a class-balanced contrast at
its valid analytic350 point. Each older reference contributes at its actual
inferred sampling node and age. No older descendant, age350 state or ancestry
label is invented. The two age families have separate four-class channels.
Queries gather at their actual sample-to350 cut. Query/context source labels,
true ancestors, true populations and migrations never enter the model.

A bias-free projection maps eight reference contrasts to D16 values. Shared
learned upward and subtree-excluding downward sweeps process every augmented
native forest by height/depth batches. Geometry, cut relation and observed
reference support modulate gates; they do not provide standalone prediction
values. Zero reference contrasts produce zero residual in every arm.

## Atomic writes and conservative episodes

For a qualified original child active in adjacent holds, compute old/new child
and outside states from the complete old/new forests. The event is a native
parent-key change or direct reference-proxy relocation onto/off that carrier's
analytic cut, plus creation of a new episode. All simultaneous edits therefore
use complete pre/post snapshots and no arbitrary event serialization.

This implementation is intentionally narrower than all contextual innovation:
remote cavity/subtree changes may alter the current read and decoder without
triggering another write on an unchanged attachment/source carrier. No extra
retention gate is applied to unchanged boundaries. Unknown child/parent
qualification, absence/reappearance and tile endpoints start from zero.
Removed carriers have no recipient and no resurrected state. Qualified sample
joins remain separate from internal native joins.

A paired event builds the same key/value payload in the matrix and GRU arms:

`v=tanh(V([du,do,mean(u_old,u_new)*do]))`, `k=bounded_unit(K(outside_old))`.

A new episode uses fresh local `[u,outside,u*outside]` values. Keys use a fixed
norm epsilon1e-4, solely for latent-vector normalization. Both directions share
one learned physical decay length within1–100kb, initialized at10kb. Observed cavity
differences and geometry condition shared retention/write gates. Matrix state
has16 scalars and uses the established gated delta update with an outer tanh.
The matched bias-free GRU also has16 state scalars and receives the same
key/value payload, decayed/retained state and write gate. Its vector is reshaped
only for the common current-key bilinear read. Thus the GRU comparison isolates
structured delta updating, not every possible conditional-memory mechanism.

Both offline directions read with the current outside key. Memory enters its
original carrier edge before a second whole-tree decoder. The decoder has
reference-conditioned gates and linear value maps, allowing exact positive
exponential scaling within a fixed hold after gathering. Static and the two
memory streams share decoder maps. Every actual100bp readout uses its containing
native hold, with the full-grid H025 selected smoother as the frozen output skip.
Subtract the FULL actual Q mean within exact observed-age groups, then apply a
zero-initialized bias-free48→4 head. No baseline-logit input, class bias,
temperature or query-ID embedding is learned. Roster centering is unlabeled
transduction and all arms retain the same roster.

## Four matched one-seed arms and execution

| Arm | Retained history |
|---|---|
| Static | Same learned encoder/linear tree decoder; memory streams zero |
| Carrier GRU | Same atomic payload, episodes, gates, decay and tree read; vector recurrence |
| Carrier matrix | Conditional carrier matrices with established gated delta updates |
| Ancestral-history sham | Same matrix events and instantaneous cavities; inherited internal histories deranged within observed age/duration/support/cut bins; sample histories fixed |

All shared encoder/decoder/gate/head weights are copied from one seed101 template.
All arms have4755 template parameters. Active scalar counts are3072 Static,4755 GRU,
3603 matrix and3603 sham. Static's active parameter blocks contain3200 scalars,
including128 head weights multiplying identically zero memory streams. The two recurrent mechanisms share16 live-state scalars
per carrier per direction, with their parameter difference disclosed. Report
actual live-carrier counts and sham changed-lookup fraction/singletons.

Batch whole-forest neural sweeps once per tile. Carrier recurrence deliberately
loops over native boundaries, batching all live carriers at each boundary.
Existing target-free inspection metadata reports431–618 holds per tile, so each
recurrent arm uses862–1236 directional update batches per tile. This finite
implementation does not claim an associative scan, continuum invariance or
subdivision-exact behavior. GPU speed and allocator peaks await root qualification.

Fit FP32, seed101, ten epochs and160 updates per arm on the existing16 TRAIN
tiles. Keep the same parent split, exact-span loss, AdamW/gradient clipping and
earliest minimum equal-parent VAL Present64 NLL selection. Reuse the unchanged
scorer for class recall/confusion, calibration, boundary F1 within1kb and short
tracts. One seed and repeatedly used development parents are discovery evidence.
A Static or sample-history-only explanation cannot establish ancestral memory.

The BudgetV4-compatible driver requires a root-pinned active input, logical GPU
lease, runtime and separate output reservation. It uses probability-only scorer
seals, retains current/best seals and last/best checkpoints, and prunes superseded
nonbest seals before new VAL output. The output cap is1GB. Root owns registration,
qualification, launch, commits and storage. CPU checks use only hand-derived
toy trees and signal/gradient paths.

The runner times forward/backward for each arm on its first real TRAIN tile,
with CUDA synchronization only for those timings. It records finite loss and
gradient checks, peak reservation, geometry counts and active parameters. These
first real optimizer steps count toward160, avoiding duplicate preparation.
