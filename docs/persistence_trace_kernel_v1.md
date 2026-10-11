# Persistence-trace kernel v1

This isolated prototype implements the hypothesis that persistent ancestral
branch memory can add value to strong frozen ancestry unaries. It does not claim
priority, a new general kernel class, continuum exactness, subdivision exactness,
or predictive benefit. It changes no existing encoder, loader, evaluation,
resource policy, or model entry point.

The source is `src/arggnn/persistence_trace_kernel_v1.py`. No training/data/GPU
entry point is provided.

## Input and identity contract

`MetricTreeHoldV1` contains a complete connected observed sample-ancestor union,
not unused rows from a tree-sequence node table. Each original vertex and branch
is retained. Branch durations must be strictly positive observed endpoint time
differences. This version rejects zero-duration branches rather than inventing
ages, adding a duration floor, or silently contracting them.

- `parent` and `sample_nodes` are local array rows.
- `sample_ids` are actual persistent observed sample identities, bound by the
  caller's parent-wide `sample_identity_namespace`.
- `qualified_node_ids` are producer-qualified parent-wide native identities.
  Unknown identities are `-1`. `identity_namespace` names their producer/parent
  namespace. Local row numbers, local Relate node numbers, equal dates, matching
  clades, and `native_vertex_index` alone do not qualify an identity.
- Native identities and sample identities perform joins only. They are never
  numeric features or embedding indices. Known native ages and actual sample
  ages must agree when the same qualified object reappears.

The declarations are provenance obligations of the caller. This small module
cannot establish producer identity semantics from integer arrays.

`source_sample_ids` explicitly declares the samples for which the caller has
frozen strong-emitter logits at each physical hold center. Sources are flattened
in tree order and then in that supplied roster order. This can be Q120 only if
those are the available logits. Context64 and reference96 must not receive
invented ancestry truth. A readout query present in a hold must have its own
frozen source there, which supplies positive local normalization mass.

`readout_logits[P,Q,K]` is the strong emitter's output on the unchanged original
physical readout grid. It is a direct skip. Source logits and direct-skip logits
are detached inside the module. Observed reference labels may enter only under
the same original observed-input contract. Hidden query labels, true ancestor
IDs, and internal population annotations are not inputs.

## Finite carrier and operator

For each interval `[left,right)`, define width
`w = (right-left)/genomic_scale_bp` and center `(left+right)/2`. Defaults are
2,000,000 bp and 10,000 generations. These are coordinate scales, not inferred
admixture dates.

The carrier at a hold center consists of all original vertices plus one midpoint
for every original branch. A branch of observed duration `ell` becomes two
spatial edges, each with raw conductance

```
w * 2 * age_scale_generations / ell
```

Adjacent holds have positive normalized center distance `d`. Shared actual
samples receive their own temporal edge of conductance `1/d`. This continuity
exists only between adjacent holds in which the sample is observed. Absence
never causes memory to be copied from an older non-adjacent occurrence.

A genuine ancestral branch match requires equal qualified namespace, child ID
and parent ID. Matching branch endpoint ages must agree. Its trace is sampled at
child, midpoint and parent with lumped weights `1/6,4/6,1/6`, multiplied by

```
(ell / age_scale_generations) / d
```

An endpoint observed as a sample on either side is omitted from this ancestral
trace because sample continuity has its separate term. Original ancestral
vertices incident to several persistent branches receive the corresponding sum
of positive trace conductances. This is mass-lumped branch-trace quadrature,
not the full consistent finite-element mass matrix or an exact continuous field.

For each of spatial, sample and ancestral conductance graphs, form a symmetric
weighted normalized Laplacian `L = D^-1/2 (D-W) D^-1/2`, with zero diagonal for
isolated states. These components have spectrum in `[0,2]`. The channel operator
is

```
A_h = I + s_h L_spatial + t_h (L_sample + L_ancestor_control).
```

`sample_only` omits `L_ancestor_control`. The two learned rate vectors and all
projection/head parameters are identical for every control. Rates are bounded
between `1e-4` and `max_rate` (default 20) using sigmoid coordinates. The same
active temporal-rate family handles sample continuity and ancestral traces.
With the default bound, every channel's spectral condition number is at most 121
before floating-point error. Identity screening is a deliberate finite
coefficient/count measure, not an unregularized branch-mass-only system.

Component normalization changes the interpretation of physical scales. It
retains relative branch-duration and temporal-conductance ratios, but cancels a
uniform rescaling within a component. In particular, the common interval-width
factor cancels within each isolated spatial tree. Physical widths still weight
source impulses. These are bounded graph filters, not a claimed discretization
with unchanged continuum coefficients or a serial-insertion theorem.

Project frozen source logits with a shared bias-free linear map and `tanh`.
Scatter width-weighted projected sources onto their actual sample carrier rows.
Alongside them, scatter the identical width-weighted ones source for each
channel. Solve numerator and mass together and read both with the same physical
interpolation. Divide numerator by mass, without a numerical mass floor.
Positive conductances and identity screening make the generated matrices
symmetric positive definite M-matrices. Thus the exact inverse is nonnegative
and matching ones-source normalization yields a bounded aggregation. Finite
arithmetic rejects a nonpositive/nonfinite readout mass. Row normalization is
not a claim that the final message is a PSD Gram kernel.

`readout_x_bp` is interpolated linearly between adjacent hold centers for an
actual query observed at both centers. Before the first center and after the
last, the correction clamps to that endpoint. If a bracketing sample is absent,
use its containing hold's local carrier value. A query absent from its containing
tree is rejected. Direct-skip unaries remain on the original readout grid, so a
long held tree does not force a piecewise-constant output.

## Controls

The same prepared bank supports:

1. `sample_only`: all spatial trees and sources remain, with sample continuity
   but no cross-genome ancestral trace.
2. `genuine_ancestor`: qualified original branch matches above.
3. `sham_ancestor`: permute right-hand matched branches within bins of observed
   midpoint age, log duration and left/right endpoint sample-role patterns.

The default sham bins are 500 generations and `log(2)` duration width. A seeded
random ordering followed by a one-step cyclic shift deranges every bin of size
at least 2. Singleton bins remain genuine and are disclosed. This preserves the
matched-branch count, trace-component count, source-side conductances and all
spatial topology/unaries. It is **bin-matched**, not exact numerical age/duration
matching. Endpoint-degree distributions and the normalized ancestral operator
can change, as intended by the geometry control.

The bank reports `qualified_branch_matches`, `sham_changed_branch_matches`,
`sham_bins`, `sham_singleton_bins`, and sample/ancestral trace counts. If few
matches can change, that sham has little diagnostic power. Preparing shams again
after changing array serialization can draw another pairing. Treat the prepared
pairing as fixed experimental geometry. A harmless global identity rename is
not a sham.

## Solve, readout and small API

Sparse FP32 PCG uses a Jacobi preconditioner, at most 128 iterations and
relative true-equation residual tolerance `1e-4` (absolute tolerance 0).
It fails explicitly if it cannot meet the declared tolerance. No dense inverse
is formed. First-order implicit differentiation solves the same adjoint system:

```
A v = upstream_gradient
rhs_gradient = v
s_gradient = -sum(v * L_spatial solution)
t_gradient = -sum(v * L_temporal solution)
```

It differentiates the linear-system solution to the solve tolerance, not an
unrolled fixed-iteration algorithm. Higher-order derivatives are unsupported.
Preparation reads CPU float64 observed coordinates/ages, then returns FP32
operators, source weights and interpolation weights. `bank.to(device)` keeps
FP32 as the default without selecting devices or launching work. Projection and
residual-head operations may use mixed precision. The sparse solve and adjoint
disable autocast and retain the explicit FP32 system dtype. Explicit
`bank.to(device, dtype=torch.float64)` and float64 RHS are diagnostic options,
with `1e-10` solve tolerance. No FP64 parity experiment is part of the pilot. Device memory,
end-to-end cost and GPU sparse performance have not been measured.

```python
bank = prepare_persistence_trace_bank_v1(
    trees, intervals_bp, source_sample_ids,
    readout_x_bp, readout_sample_ids,
    sample_identity_namespace="producer-qualified-parent-roster",
)
layer = PersistenceTraceKernelV1(num_classes=4, channels=8)
logits = layer(bank, source_logits, readout_logits,
               control="genuine_ancestor")
```

The residual output affine layer starts with all weights and biases zero, so
finite successful forward evaluation initially reproduces the direct-skip
emitter exactly. As with other zero residual-output initializations, kernel
parameters begin receiving loss gradients after the output weights move away
from zero.

## Scientific comparison and the one sanity example

Run one development seed per candidate/arm in FP32 or mixed precision with the
FP32 solve. Start with the strongest completed frozen emitter. Keep parents, actual Q masks,
original physical grid, clipping/loss weights and existing selection criterion.
The principal contrast is genuine versus sample-only, with identical local trees,
unaries and residual capacity. Genuine versus sham tests qualified ancestral
correspondence. Compare incremental value with the strongest already declared
ordinary genomic smoother. This is an experimental plan, not a performed fit.

One necessary basic check used 10 handmade carrier states on CPU in FP32.
A fixed nonzero residual-output fixture exercised the implicit gradients after
checking the constructor's zero initialization. The single forward/backward had
cross-entropy loss 1.3903154134750366, all parameter gradients finite, maximum
spatial-rate gradient 2.5874194761854596e-05 and temporal-rate
gradient 4.537112999969395e-06. This is basic arithmetic evidence, not a fit or
biological result. No actual dataset, trained emitter, GPU job or benchmark was
opened or run for this prototype. An explicitly named optional FP64 diagnostic
is retained in source and is not part of the FP32 development pilot.

Closest primary references are the established [dynamic space-time graph
kernels](https://arxiv.org/abs/1612.03615v2), [metric-graph Whittle–Matérn
fields](https://arxiv.org/abs/2205.06163v2), [GraphSSM](https://arxiv.org/abs/2406.00943v2)
and [dynamic orthogonal sheaf transport](https://arxiv.org/abs/2606.10071v1).
Sparse solves, Schur complements, metric-graph operators and implicit
backpropagation are established tools. The substantive hypothesis here is
qualified whole-branch history retained before joint boundary condensation.
