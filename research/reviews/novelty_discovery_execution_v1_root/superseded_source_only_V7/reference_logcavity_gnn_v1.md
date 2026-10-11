# H030 reference log-cavity whole-tree GNN

H030 tests whether normalized reference evidence supplies an informative scale
for learned tree propagation and carrier history, compared with H029's attenuated
raw reference contrasts. The source has hand-derived CPU checks only. Root owns
prospective registration and actual qualification/execution.

This is a reference-mixture tree kernel with exact log-domain rerooting cavities,
followed by a learned finite whole-tree GNN. Sum-product messages, positive tree
affinities, rerooting, TGN memory and DeltaNet updates are established primitives.
It is analogous to pruning/rerooting algorithms, but it is not a Felsenstein
substitution likelihood, a population-migration model or an ancestry posterior.
The new condition is unit-scale normalized reference evidence and a learned
additive age potential, not a claimed new normalization or recurrent primitive.

## Observed source and additive edge potential

Retain all original native vertices/edges in every complete observed280
sample-ancestor union, including context64. Insert analytic350 points on every
crossing native edge. Younger/query sample paths use their valid350 points.
The96 observed pure-reference proxy labels supply signal. Older references
remain at their real inferred sampling nodes and dates, with no invented census
descendants or older ANA source. Actual sample identities and qualified internal
IDs are joins, never learned embeddings or true-ancestor assertions.

Each younger/older family has four reference-class channels. A class-c reference
contributes log weight `-log(n_family,c)` at its valid source node, balancing
represented classes. Multiple sources on one node combine by logsumexp.

For family g with fixed scale s_g=350/1400 generations, learn two shared positive
coefficients and define

`Phi_g(t)=a_g*max(t-350,0)/s_g+b_g*log1p(max(t-350,0)/s_g)`.

Initialize a=1,b=.1 in every arm. Coefficients use softplus and are shared across
classes, queries, native edges and both genomic directions. Edge cost is
`Phi(parent_age)-Phi(child_age)`, which is nonnegative and telescopes across a
serial subdivision. Below350 costs are exactly zero. All valid reference sources
are at/above350, so a path below350 supplies a reference-independent common
factor that cancels from normalized fields. Such a rate is unidentifiable here
and is deliberately omitted. Observed below350 dates still enter the finite GNN's
geometry gates; this is not a claim that the kernel learned pre350 age effects.

## Exact log cavities and masks

For each native/cut node, upward log evidence sums own sources and all child
evidence attenuated by their edge costs. Outside evidence sums parent-local
sources, parent outside evidence and sibling evidence, then applies the child
edge cost. Sibling exclusion uses prefix/suffix log sums, avoiding total-minus-child
subtraction of tiny positive masses. Entire native forests batch their upward/
downward levels, and siblings batch by fixed child rank.

Normalize each age family's available classes with relative-log logsumexp.
Subtract a detached maximum before computing the normalizer, avoiding absolute
large-log cancellation in FP32. All-empty supports are replaced before logsumexp
and return zero fields with explicit false masks and finite zero gradients.
Absent individual classes keep -inf log evidence and exactly zero unit values.
There are no pseudocounts, probability floors or small-mass FP32 divisions.

Subtract the uniform prior over represented classes from each unit field,
masked to zero for empty families. These centered reference values feed learned
bias-free projections. The upward GNN injects subtree-normalized evidence at
every node; the downward GNN injects exact outside-normalized evidence and
combines parent outside, parent-local sources and sibling learned messages.
Using a parent's subtree field in the outside base would leak the excluded child
and is specifically avoided. Geometry/support modulate gates rather than provide
standalone prediction values.

Exact kernel fields are invariant to a zero-source serial subdivision. The
nonlinear GNN has node/edge gates, repeated bases and a finite tree decoder,
so the complete architecture is explicitly not subdivision invariant.

## Five new matched arms and frozen comparator

| New arm | Condition |
|---|---|
| Log-cavity only | New36-parameter learned positive age-kernel plus bias-free8→4 head |
| Static | Unit evidence feeds the learned whole-tree encoder/linear decoder; no history |
| Carrier GRU | Same atomic cavity payload, gates,16-scalar state, decay, episodes and bilinear tree read |
| Carrier matrix | Same information/read with established gated delta update |
| Internal-history sham | Same instantaneous cavities/events; inherited internal histories deranged; sample histories fixed |

The separately labelled log-cavity-only arm tests this new learned age potential,
not a repeat fit of H026's answered fixed-kernel head. Reuse H026's selected frozen
native_reference predictions/checkpoint/metrics as the strong external comparator.
That comparator retains its original younger donor-age weighting, while H030
younger sources are valid census proxies. This information-policy distinction
must accompany the comparison.

Every new arm starts from a single seed101 template with identical shared
encoder/decoder/gates/cost coefficients/head initialization. Total template
parameters4791. Operational active block counts are36 kernel-only,3204 Static
(including128 zero-memory head scalars),4759 GRU and3607 matrix/sham.
Older ANA remains a zero kernel input, so its four kernel-head weights are
data-unidentified despite the nominal36 count. Recurrent state is16 scalars per
qualified carrier per direction; GRU has additional update parameters, disclosed.

Keep H029's conservative event trigger: attachment-key/direct-source relocation
or episode creation. Complete old/new cavities measure contextual changes,
including simultaneous distant rearrangements. They do not identify the causal
effect of recombination. No counterfactual causal attribution is claimed.
Unchanged attachments/sources have no extra write gate. Absence/unqualification/
reappearance, terminal episodes and250kb tile edges reset, without resurrection.

Use the same second complete-tree decoder, current-key read and physical
bidirectional decay. The full original100bp support and Q120 roster remain,
with the exact H025 selected smoother as frozen skip. Same-age full-roster
centering feeds zero-initialized bias-free residual heads. Query/context ancestry
labels and baseline logits do not enter context or learned heads.

Fit FP32 seed101, ten epochs/160 updates per arm, identical original TRAIN/VAL
parents, exact-span loss and minimum equal-parent VAL Present64 NLL selection.
Reuse the unchanged class/confusion/calibration/short-tract/boundary scorer.
The first actual TRAIN tile records synchronized forward/backward timing and
finite gradients per arm while preserving that real update in the160 total.

## Feasibility and remaining scope

Existing target-free metadata has431–618 holds per250kb tile and at most449840
augmented rows under a conservative cut-point bound. Carrier recurrence uses
862–1236 directional boundary batches per recurrent tile. The new CPU bank adds
about56 bytes per augmented row for log sources, age bases and sibling indexes,
at most25.2MB per maximal tile. GPU log-field/autograd overhead and wall time are
unmeasured; root must inspect its first real training tile.

Retain probability-only current/best seals and last/best checkpoints within2GB.
Five arms plus frozen H025 output give a conservative raw probability peak844.8MB
before compression and small metadata. H026 comparator files are reused.
The scientific design and exact BudgetV7 adapter/import pins are source ready.
Execution still requires root's reviewed active admission and2GB reservation.
The new guard/writer stay isolated from original V4/V6 consumers; the readiness
flag is source readiness, not execution approval. Temporary tree writes are guarded before/after their actual bounded bytes
plus8192 overhead; directory creation has8192 allowance. Existing CUDA helper
qualification is guarded before/after a conservative32MB allowance.

A kernel-only gain over H026 would support this learned kernel/source-policy
condition, not a new graph primitive. No claim of accuracy, ancestry preservation,
original primitives or generality
follows from CPU cases or a one-seed reused-development fit. Matrix versus GRU
tests structured updating under the same conditional read; it cannot establish
priority of the whole coupling.
