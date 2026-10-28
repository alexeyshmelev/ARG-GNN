# Independent review of the sliding branch joint convolution

Review date: 28 October 2026. Reviewer scope: biological interpretation,
operator semantics, numerical edge cases, complexity and decisive controls.
No author's files were edited. No GPU, ancestry training, labels or real-data
selection was used. Three tiny independent semantic examples and a sample-ID
contract check ran on CPU; their source and receipt are
`review_sliding_branch_counterexamples.py` and
`SLIDING_BRANCH_COUNTEREXAMPLES.json` in this directory.

The inspected numerical implementation is
`research/sliding_graph_convolution_20261028/branch_joint_conv.py`, SHA256
`f837adcd36c9a8e13d893fed5e54c5d8dcace184196d5f53498fa5213a22c3ae`.
I first inspected its predecessor ending `ca3bd2`; the author then fixed its
sample-roster validation and preserved prior evidence. The author is updating
DESIGN wording in response to this review without changing the pinned
numerical implementation. Findings below refer to that implementation.

## Verdict

Retain this as a distinct, falsifiable architecture candidate. Its joint
message scheduling and finite-branch carriers differ materially from frozen
TreeLoom's one cut exchange between complete cavity sweeps. It is not merely
a renaming of the same executable model. However, its correspondence is an
integral of shared-leaf cut incidence, not a newly discovered ancestor-matching
principle or general convolution. The evidence establishes an operator and
some useful exact identities; it establishes neither ancestry usefulness nor
a computational advantage for a complete predictor.

The scientifically useful hypothesis is narrow: a common tree initializer plus
age-aware branch correspondence may use nearby inferred genealogical
neighborhoods more effectively than same-leaf exchange or a few fixed cuts.
The main competing explanation is that it supplies additional smoothing,
capacity and inferred-age/coverage information. A direct comparison only
against the frozen model cannot distinguish these explanations.

## What the weights actually mean

For a target branch u with descendants D_u and interval I_u, the implemented
normalized row has the following exact interpretation. Choose a sampled
identity uniformly from D_u and an age uniformly from I_u. In the source
tree, find the finite branch containing that sample's lineage at that age;
return its branch state, or zero if no finite source branch exists. Average
over those choices. Genomic alpha then takes another conditional average over
source coordinates on the relevant side.

This interpretation explains both the merits and the limits. It aligns
observed descendant witnesses at overlapping dates without comparing internal
IDs. It does not establish that either inferred branch represents the same
historical ancestor, population, migration event or ancestral recombination
edge. Independently inferred trees can agree on descendants and age intervals
while having different latent genealogies or correlated reconstruction errors.
A weighted average of learned states is not a calibrated interpolation of
independently inferred ancestral individuals.

There is no penalty for extra source descendants. The independent example has
a target branch with descendants {0,1} and a source branch with descendants
{0,1,2} over the same lifetime; its normalized coupling is exactly 1. This is
correct asymmetric witness averaging. It cannot be described as a specific
ancestor match or a confidence score. Large source clades can distribute the
same broad state to many small target clades. Raw M is symmetric under pair
reversal, but target normalization and directional genomic normalization do
not imply reciprocal stochasticity or global ancestral-mass conservation.

The uniform sampled-identity measure also embeds panel composition. Unequal
cohort/reference representation can affect clade witness averages. That is an
observed design choice, not ancestry leakage, but it needs diagnostics and a
panel-composition control before giving a population interpretation.

## Ages, roots and branch edge cases

Ancient terminal samples are handled correctly under the declared model:
paths start at each sample's observed age, so no witness from that sample is
introduced at younger ages. Monotone parent dates and identical sample ages
across trees are required. A true one-node sampled root has no finite branch
and therefore zero branch-channel support, while same-leaf links can still
provide context. Forests have the corresponding per-lineage finite-root rule.
Sampled interior ancestors are explicitly unsupported and rejected; the full
ancestry-model adapter must not silently relabel them as terminal samples.

Roots have no branch above them. This avoids inventing infinitely old
ancestors, but absence above an inferred root means unavailable inferred
support, not biological absence of an ancestral lineage. If a source root is
younger than the target branch extent, its missing mass attenuates the message.
Consequently coverage can encode dating error, inferred MRCA age, forest
completeness or reconstruction quality. The implementation returns
branch_coverage but BranchJointConv does not automatically consume it as a
feature (lines 93 and 176-181). A complete model must declare how coverage is
exposed, and compare against a coverage-only contextual channel. Root states
can still influence descendants through native edges; excluding their branch
above the root does not isolate the model from root information.

Zero-duration branches contribute no positive overlap, so no division by a
zero branch mass occurs. They remain native nodes and native edges. The
native channel assigns the same hop mechanism to zero- and positive-duration
edges. Adding a unary or zero-duration node can therefore change a shallow
network's receptive field and nonlinear computation even when the metric
ancestral path conveys the same biological information. Native-node
renumbering equivariance does not imply invariance to such graph refinements.
A canonicalization policy or an explicit sensitivity test is required. An
arbitrarily short positive branch can retain a normalized branch message of
order one, while the zero-duration limit drops that channel; raw overlap
continuity does not make normalized B continuous at zero mass.

Linear ancestral measure does not make every old branch arbitrarily large in
message norm: division by its target mass keeps each B row substochastic.
The actual concern is averaging over unequal durations. The independent toy
has a target lineage lasting 1,000 generations; the source's recent 0-10
segment receives weight 0.01 and its older 10-1,000 segment weight 0.99. A
recent signal can be overwhelmed inside that average. Conversely, a very short
target branch still gets a complete normalized channel if covered. Old source
branches can also be reused across many targets, creating broad smoothing
hubs even though individual row sums are bounded.

The code uses absolute chronological generations. Simultaneously multiplying
all branch and sampling ages by the same positive constant cancels from B
under the linear measure; independently rescaling inferred trees does not.
The correspondence therefore assumes a common age calibration across local
trees. Date uncertainty, branch-order error and inconsistent age scales are
not modeled. A fixed recent-age density, finite age cap, or age-stratified
channels would change the prior and need separately declared comparisons.
Do not select a decay scale only because it improves the current diagnostic.

Each node supplies one state for its entire outgoing branch lifetime. The
method integrates piecewise-constant branch states over all overlapping ages;
it does not learn or resolve a separate state at every age. With those same
states and measures, a quadrature using all relevant branch endpoints exactly
reproduces the overlap matrix. A finite-cut approximation tests resolution and
cost, not a fundamentally different correspondence family.

## Genomic normalization and roster contract

Separate left/right kernels preserve direction, but normalize each side
conditionally. With exactly one positively weighted source on one side,
alpha=1 for every distance strictly inside the radius. The independent check
finds identical leaf weights at distances 0.1R and 0.999999R, then no edges at
R. The triangular numerator therefore does not guarantee absolute attenuation
or continuity when a sparse side loses its final source. This is particularly
relevant for small K windows. The model may still use relative distances among
several neighbors, but the complete predictor should expose side availability
and kernel mass or explicitly accept this conditional-average semantics.
Distance controls must match this normalization, directionality and radius.

The predecessor validated only equal roster lengths and sampling ages. That
would silently accept a permutation of equally dated present-day samples in
one tree. The author repaired this: explicit unique sample_ids are now
required and ordered rosters must agree before coupling (lines 20-33, 67-68,
115-116). The independent mismatched-ID example now rejects correctly. IDs
remain routing metadata, not numerical input features. Adapter correctness
still matters: copying the same IDs onto the wrong leaf order defeats any
operator-level validation.

Distinct coordinates are required by this operator. Repeated containing
trees at different coordinates are allowed, but exact repeated coordinate
draws are rejected. The frozen coordinate executor supports duplicate draws,
so these APIs do not yet cover the same full input domain. A continuous
sampler almost surely avoids exact repeats, but finite precision, deterministic
sampling and future quadrature adapters need an explicit policy. Do not
silently deduplicate or jitter positions and claim the same finite-window
predictor. Pure genomic serialization subdivision invariance, when attributed
features are preserved, says nothing about recomputed SNP pools or unary-node
refinement inside a tree.

## Implementation and worst-case cost

The synchronous update uses old-layer node states for all six relations and
nodewise normalization. It has no recurrence or depth-scheduled cavity pass.
Its L layers still traverse only L native edges from raw features; loading a
complete topology alone does not deliver distant reference evidence. A shared
complete-tree initializer is the sensible comparison, with its full cost
included. The current operator omits observation/SNP encoding, that initializer,
focal classifier, target objective and real-data loader. Its synthetic layer
timing cannot be compared to a complete frozen-model forward/backward time.

For one tree pair, the path merge is bounded by the sum of the two ancestral
path lengths across all samples. Sparse accumulation may coalesce repeated
branch pairs, but it still walks every witness path. K trees with S supported
neighbors can cost O(K S H D); for skewed genealogies D can scale with H, and
with arbitrary unary chains it can scale with native node count. E can be
quadratic in sample count per pair in unfavorable trees; no worst-case linear
promise follows from sparsity in balanced toys. Constant K alone does not
bound tree depth, native nodes, SNP count or adjacency size.

The actual validator separately follows ancestors from every native node
(lines 39-46), costing O(sum_t N_t D_t). A single-sample unary chain can make
this quadratic in its native node count. Paths are rebuilt inside masses and
again for each ordered pair; pair symmetry does not currently reuse the
reverse overlap. These are correctness-reference implementation costs, not
biological necessities. Report validation, overlap construction, edge count,
transfer and execution separately before choosing an optimization.

Each layer applies seven dense node transforms and six aggregations. The
O(N d^2 + E d) arithmetic description is reasonable for a fixed channel count.
However, ordinary edge_index PyG propagation gathers x_j and materializes
x_j*weight (lines 162-166): actual forward temporaries include O(E d), even
without autograd. O(Nd+E) describes node state plus stored adjacency, not total
forward peak memory for this implementation. Training additionally retains
activations/messages as required by autograd. The stored edge weights are
float64 and converted to feature dtype on each relation call, creating further
conversion/allocation work. A fused sparse implementation might improve this;
none is supplied here.

## Decisive controls and rejection criteria

First separate correspondence from layer scheduling. Use the same observation
adapter, full-tree initializer, focal targets, source coordinate draws,
genomic normalization, output head and optimization for: native-only joint
layers; same-leaf joint layers; selected-cut joint layers; and branch-overlap
joint layers. Match parameter budgets or explicitly measure capacity effects.
A direct frozen-v1 comparison changes both correspondence and schedule and
cannot identify which causes a difference. A factorial comparison of exchange
carrier and cavity-versus-joint schedule is stronger if resources permit.

Within that common schedule, the most decisive controls are:

1. Same-leaf plus coverage/geometry features versus branch content. This asks
   whether inferred-root availability and dates account for a gain.
2. Branch transport with correspondence shuffled only in the branch channel,
   within observed sampling-age, cohort and reference-role strata. Keep native
   trees, same-leaf identity links, features and target labels fixed; monitor
   changes in coverage so the intervention does not simply add missingness.
3. Few cuts, increasingly fine cuts, and exact endpoint integration with the
   same branch states. This distinguishes finite-resolution effects from a
   new representation claim and measures the accuracy/cost tradeoff.
4. Declared age sensitivity: common versus inconsistent date perturbations,
   recent-age weighting/caps, and diagnostics stratified by inferred root age,
   target-branch duration and source/target clade-size ratio. These must be
   independently fitted controls for mechanism claims, not post hoc inference
   changes selected on evaluation outcomes.
5. Matching genomic radius, directionality, side availability and normalization
   across arms. Include short-tract boundaries and sparse one-neighbor sides;
   average accuracy alone can reward excessive smoothing.

Record NLL, per-class recall, boundary localization, short-tract recall and
false switches, with independent parent/chromosome replication and no TEST
selection. A gain that disappears after coverage/capacity matching, survives
branch-only correspondence destruction, or mainly erases short tracts does
not support the intended genealogical-correspondence mechanism. Measure the
end-to-end preparation, memory, forward/backward and convergence cost on the
same actual bounded packets; balanced synthetic topology is insufficient.

The proposed biological question is worth testing after these controls are
specified. Present it as a specialized continuous-age transport inside a
standard relational GNN, with a separate schedule and explicit observed-data
assumptions. Do not present it as proof of matching independently inferred
ancestors, a full ARG reconstruction, or a demonstrated speed/accuracy win.
