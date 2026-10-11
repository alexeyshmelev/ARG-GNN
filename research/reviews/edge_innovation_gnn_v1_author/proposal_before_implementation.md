# Edge innovation GNN v1 — unimplemented proposal

Propose one learned operator: **conditional branch memory written by atomic tree
reattachments and read into whole-tree messages before query pooling**. H025
found no useful persistence increment; H026 is discovering reference information
with established kernels. This proposal tests a different bottleneck: whether
pooling each tree to query contexts loses which ancestral carrier experienced a
reference-context change. No implementation, priority or predictive result is
claimed.

## Inputs and learned whole-tree context

Keep every complete observed tree, original vertex/edge, sampling age, physical
interval and qualified native identity. IDs are joins, never embeddings. Add
analytic census 350 points. Use eight observed reference-contrast channels:
younger-reference class proxies at their valid 350 cuts, older-reference cohort
proxies at their actual sampling nodes. Older nodes never receive fictitious
350 states. Q/context labels, true ancestors/populations and migrations remain
excluded. As in H026, these are soft inferred-geometry features, not ancestry
assignments.

Use hidden dimension `D=16`. A shared learned upward sweep gives child-side
reference states `u`; a downward sweep gives outside states `d[p\\c]`, excluding
the child subtree. Each edge message has a learned gate conditioned on observed
age, duration, cut relation and reference support. Reference values pass through
bias-free maps/tanh; geometry modulates gates, never supplies a standalone
prediction value. Thus zero reference contrasts imply zero context. Process all
nodes with batched forest levels, not sampled paths or one neural call per node.

At a genomic boundary, pair deleted `(c,p_old)` and inserted `(c,p_new)` edges by
qualified child identity. Compute BOTH cavities from the complete pre/post trees,
not arbitrarily ordered intermediate edge deletions. Their simultaneous-boundary
transition is independent of edge-event serialization. Pairing an inferred object
does not assert unchanged true ancestry or an unchanged descendant subtree.
Unknown parent correspondence produces an unpaired local event, not a match by
row number, date or clade resemblance.

For a paired reattachment define

```
Delta_u = u_new[c] - u_old[c]
Delta_d = d_new[p_new\\c] - d_old[p_old\\c]
v = tanh(V([Delta_u, Delta_d, mean(u_old[c],u_new[c])*Delta_d]))
k = unit_bounded(K(d_old[p_old\\c]))
```

`V` is a learned bias-free value network; `K` is a learned key network. Geometry
and cut-crossing indicators condition their gates. `v` includes the outside
change even when the moved subtree contains no reference, which is essential
for unlabeled query-subtree reattachments. Unmatched insertion tokens use fresh
local values and the current outside key; removed carriers are discarded, with no invented recipient.

## Ordered conditional state and whole-tree read

Each active qualified ancestral child carries `M[c]`, a `4x4` associative response
matrix per scan direction. It stores observed reference-context responses, not
an ancestry label/posterior. At a boundary, with physical gap `Delta_bp`,

```
Mbar = exp(-Delta_bp/ell_bp) * gamma * M_previous[c]
write = (v - Mbar @ k)[:, None] * k[None, :]
M_new[c] = tanh(Mbar + eta * write)
```

Here `k,v` have dimension 4, `eta` is learned in `[0,1]`, and `gamma` is learned
retention from observed cavity disagreement, source support and structural change.
`gamma` is applied at actual attachment/source-relocation writes, not every
serialized hold. It is not a calibrated correspondence probability and may erase
memory even when an inferred identity persists. Missing/unqualified correspondence starts
from zero; absence followed by reappearance never resurrects memory. Unedited,
unchanged-source boundaries have no write and no extra retention gate. Initialize
physical length `ell_bp=10000`, learned within `[1000,100000]` bp.

Read with the CURRENT outside key, `z_e=M[c] @ K(d_current[p\\c])`. Inject `z_e`
into its original edge before a second complete upward/downward message sweep.
This decoder uses learned reference-conditioned gates and linear value maps.
Consequently, a past branch response can affect other current queries through
the actual tree; it is not appended after query pooling. Its values depend on
which reference environment the carrier previously left and currently occupies.

Run forward/reverse scans for offline ancestry. Decode static and both memory
contexts once per hold; within a hold, scale the two memory components by their
positive physical distances from the respective boundaries. The linear memory
value decoder supports this inexpensive decay without rerunning the tree GNN at
every 100 bp point. Retain every original readout and the full-grid strong-emitter
skip. Gather queries at their observed 350 points; distinct frozen skips prevent hard
clade equality. This is a finite model, not a continuum/subdivision-exact claim.

Batch static upward/downward encoding across ALL native forests by depth. Pack
contiguous qualified-carrier episodes, resetting on absence/unqualified joins.
Process independent episodes together at each of their ordered edit indices.
The nonlinear tanh/delta update still requires an episode event loop; do not
assert an associative scan. Reconstruct each hold's matrix by physical decay
since its last write. Thus neither neural encoding nor memory needs a separate
GNN call per node/hold. Batch the linear, reference-gated second decoder over the
whole forest. No resurrection or future-target input is introduced.

A bias-free zero-initialized head consumes only these contexts after subtraction
of each position's FULL actual same-age-Q mean. No baseline-logit input, class
bias or temperature is learned. The centering is transductive unlabeled roster
context. All arms keep the same roster and frozen strongest completed skip.

## Small matched experiment, unimplemented

Use FP32, `D=16`, memory rank 4, seed 101, ten epochs/160 updates per arm, the
existing parent split/loss/selection and 250 kb tile resets. Share reference-source
policy, whole-tree encoder template/initialization, context head, observed data
and physical gaps. Do not select widths from VAL. Report actual parameter counts
and runtime rather than building a parity project.

| Arm | Information retained |
|---|---|
| Static | Learned whole-tree reference context; no history |
| Generic | Same static context, its difference, AND the same instantaneous edit information after whole-tree readout; gap-aware bidirectional query GRU with an explicit capacity budget |
| Genuine | Conditional carrier matrices and memory-conditioned whole-tree messages |
| Memory sham | Identical genuine network/events, but inherited matrices deranged among qualified carriers within observed age/duration/support bins |

For Generic, generate fresh one-boundary edit responses without inherited branch
memory, decode them through the same whole-tree read, then filter at query level.
It therefore receives instantaneous edit evidence, not merely weaker reference
kernels. The principal tests are Genuine versus Generic and Genuine versus sham.
Static measures whether gains came only from learning a better reference encoder.
Choose Generic width from TRAIN geometry before any fit: at least `D`, and
sufficient live query-state scalars to cover the median genuine carrier-state
count (`rank^2 * live_carriers` per direction). Its nonshared trainable update
capacity must match or exceed Genuine; no VAL width sweep. Both receive the
same observations, instantaneous edit payload and physical gaps; only where
history is retained differs. Report actual state/parameter counts.

Sham leaves labels, instantaneous cavities, events, dates and baseline untouched.
Keep sample-carrier histories fixed and derange only internal ancestral histories,
so a sample-only matrix-memory gain cannot establish the ancestry claim. Report
changed-lookup fraction and singleton bins.

The anticipated benefit is history-sensitive reference rerouting: equal current
query contexts can hide different past carrier associations that a later edit
exposes. This is an information-loss motivation, not a proven expressiveness or
biological statement. Preserve parent accuracy/NLL, class recalls/confusions,
calibration, short tracts and boundary F1 within 1 kb. A gain only over the plain
emitter or Static does not support this mechanism. Reused development parents
and one seed provide pilot evidence only.

## Closest prior art and failure conditions

TGN already combines timed edge events, memory and graph operators. GraphSSM
already handles graph-conditioned history and persistent/new nodes. Gated
DeltaNet already supplies the matrix update above; gated linear attention also
uses matrix-valued state. Cavity tree messages are also established in belief propagation. These are
established primitives, not novel algebra.
GST already interleaves graph processing with collection feedback. Ancestral
Paths uses reference cohort proportions/ages; ARGMix combines reference/TMRCA
attention and genomic GRU; ARGformer learns shared ancestral-path tokens.

The candidate contribution is the **atomic old/new reference-cavity binding,
uncertain carrier association and conditional whole-tree read before pooling**.
It is genuinely learned, but priority remains unresolved. A suitably equipped
TGN could implement it, and an ordinary node/edge-memory model could match it.
Novelty fails if existing work already supplies this coupling, or if Generic or
sham matches Genuine. If Static explains the gain, report improved reference
encoding rather than ordered innovation. Irrelevant recombination, inferred-ID
errors and rare informative edits can make retention collapse to zero. None of
these possibilities justifies hard ancestry preservation or a novelty claim.
