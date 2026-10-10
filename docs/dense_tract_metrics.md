# Dense ancestry tract helpers

These CPU helpers implement descriptive dense diagnostics for a future frozen
evaluation. They do not load datasets, generate ancestry, prepare model inputs,
train models, run predictions, select cohorts, perform parent/seed inference, or
establish a custodian/prediction/analysis freeze. The proposed protocol is
`research/protocols/confirmatory_design_v1.json`, which remains a proposal unless
the responsible runner separately binds its execution artifacts.

## Public API

All APIs live in `src/arggnn/dense_tract_metrics.py`.

- `DenseGrid(grid_id, domain_bp, coverage_bp, grid_points, original_indices, positions_bp)`
  validates the identity of one global uniform physical-bp midpoint grid.
- `coverage_clipped_midpoint_grid(*, domain_bp, coverage_bp, grid_id, grid_points=4096)`
  declares its original midpoint coordinates retained within observed coverage.
  It reads neither truth nor features. It creates no source anchors.
- `DenseGrid.cell_bounds_bp()` returns the original grid cells clipped physically
  to observed coverage. `DenseGrid.identity_record()` returns the explicit grid
  fields and SHA-256 of their sorted compact JSON serialization.
- `Tract(left_bp, right_bp, class_slot)` describes a half-open exact truth tract.
- `TractPartition(query_ids, domain_bp, num_classes, tracts)` validates complete,
  contiguous, ancestry-merged full-domain truth partitions in original query order.
- `extract_census_tracts(ts, query_node_ids, *, query_ids,
  source_population_to_class, census_time, num_classes)` extracts **external
  labels only** from a census-bearing `tskit.TreeSequence`.
- `dense_tract_metrics(raw_logits, *, query_ids, grid, true_partitions,
  artificial_tile_edges_bp, boundary_tolerance_bp=1000.,
  short_tract_cutoff_bp=10000., tile_edge_radius_bp=10000.)` joins explicitly bound
  predictions and exact external truth, returning JSON-serializable descriptive
  metrics for one parent and one checkpoint.

`query_node_ids[i]` and `query_ids[i]` explicitly bind the raw truth sample node
to the original query identity. This supports a separately audited truth/observed
node mapping. IDs are unique nonempty strings or integers. The extractor preserves
their order. Metrics require prediction `query_ids` to equal partition
`query_ids` in the same order. They never sort or infer a correspondence.

The logit tensor is `[retained midpoint, ordered query, class slot]`. Class slots
are integer `0..num_classes-1`; the population map defines their meaning. The
caller must hash-bind the same class-slot meaning to its checkpoint/prediction
manifest. The helpers cannot infer a model's class names from numeric logits.
All slots remain present in reports even when ancestry is absent.

## Exact external census truth

Extraction visits every exact tskit tree interval. Every query must be a unique
sample younger than the requested census. Its full rootward path must contain
exactly one `msprime.NODE_IS_CEN_EVENT` ancestor, at the requested age within
absolute tolerance `1e-9` generations, whose population appears in the explicit
map. Missing ancestors, multiple flagged ancestors, incorrect census ages, or
invalid/unmapped populations are errors.

A new tree interval with the same ancestry class extends the existing tract.
Tree topology, census node identity, or unrelated leaf changes therefore do not
create ancestry switches. Extracted partitions cover `[0,ts.sequence_length)`
and retain exact ancestry-switch positions. No cleaned inferred-tree boundary
is used as truth. The resulting object belongs exclusively in label storage and
the joined analysis, never model feature generation.

## Grid and coverage identity

Domain and coverage are finite increasing physical intervals, with coverage
inside a nonnegative physical domain. Domain, resolution, original retained
indices, and physical midpoint coordinates are all explicit. Toy resolutions
are permitted for handmade tests; a frozen evaluation must explicitly bind the
proposed 4096-point resolution.

The original centers are
`domain_left + (i + 0.5) * (domain_right-domain_left)/grid_points`.
Retain every center in half-open `[coverage_left,coverage_right)`, in original
order. Positions must equal these centers within eight float64 machine epsilons
times the domain coordinate scale. Missing, repeated, reordered, noninteger,
gapped, or regenerated coverage-local indices/positions are rejected.
The single-interval API rejects disjoint coverage. A runner needing disjoint
coverage requires a separately specified extension rather than using this API
to bridge gaps. Empty retained grids are supported explicitly.

The original cell for index `i` is
`[domain_left+i*step,domain_left+(i+1)*step)`. Its weight is the intersection
length with coverage **only if its center was retained**. No missing-center cell
is filled. For domain `[0,100000)`, four centers, and coverage
`[20000,80000)`, retained centers are `37500,62500` and supported cells are
`[25000,50000),[50000,75000)`. The represented span is 50000 bp and the
unrepresented coverage-edge strips total 10000 bp. For coverage
`[10000,80000)`, weights are 15000, 25000, and 25000 bp. End cells can therefore
have unequal weights.

Results include coverage span, retained centers/indices, original clipped cell
bounds, represented physical bp, unrepresented physical bp, and grid hash.
Prediction tracts and quadrature can have different denominators, explicitly
reported.

## Boundary and tract metrics

Truth switches count only when strictly inside coverage. Clipping preserves
exact interior truth boundaries. Neither a coverage endpoint nor a domain
endpoint creates a switch.

The argmax tie rule is NumPy's first class slot. Predicted switches occur at the
physical midpoint between successive retained dense samples whose argmax
differs. Equal adjacent predictions merge. These coordinates are a declared
resolution-limited reconstruction; they are not an exact change point of the
model's continuous readout. The first and last predicted classes extend to
coverage edges for reconstructed tract geometry. They add no endpoint switches.
An empty prediction grid produces no prediction tracts and zero geometric
readout support.

Global boundary diagnostics reuse
`confirmatory_metrics.boundary_match`: maximum-cardinality sorted
one-to-one earliest-feasible matching, inclusive 1000-bp default tolerance.
Localization distances describe that deterministic pairing, not a globally
minimum-distance assignment. Reports retain true/predicted support, matched,
missed, extra, precision/recall/F1, absolute switch-count error, and matched
coordinates. When truth has zero switches, recall/F1 are NA, while extra
predicted switches and switch-count error remain visible.

Per-class boundaries are explicitly **incidence** diagnostics: each switch
belongs to both adjoining classes. Each class is independently matched
one-to-one, so its matching can differ from the global class-agnostic matching.
Summing class-incidence counts counts each switch twice.

Clipped truth records retain original endpoints/length, visible endpoints/length,
whether coverage truncates each original tract, and domain/coverage censor flags.
A switch exactly on a coverage edge is censored even when no tract length was
truncated, because both sides are not visible. Domain endpoints are censored
because the biological tract may continue beyond the physical domain.
`full_observed` requires both true switch endpoints strictly inside coverage
and domain. Length summaries preserve visible and fully observed lengths,
counts, domain censoring, and coverage censoring separately.

Reconstructed tract overlap uses a linear sweep of exact clipped truth against
the reconstructed argmax partitions. Its confusion/error spans are exact for
those two declared partitions, not evidence of exact continuous model behavior.
It reports per-class false/missed ancestry span and signed predicted-minus-true
class span, with explicit evaluated and visible supports. With no prediction
support, overlap accuracy and span errors are NA.

A full true tract has a matched length error only when both true switches match
consecutive predicted switches enclosing a predicted tract of the same class.
Reports expose full-true support, matched-length support, unmatched count, signed
errors, and mean absolute error. Censored tracts do not receive fabricated
full-length errors.

## Approximate span quadrature and short tracts

NLL uses the existing stable float64 raw-logit policy: subtract the maximum and
apply log-sum-exp without temperature fitting or probability floors. Nonfinite
logits or a log-softmax outside representable float64 range are rejected.
Extreme wrong-class losses, including 2000 nats, remain uncapped.

Each retained center's exact midpoint truth label and raw logits receive its
physical original-cell weight. This is **approximate midpoint truth-label
quadrature**, not exact integrated NLL or accuracy. True ancestry changes inside
a cell are not integrated. The geometric overlap metric is reported separately.
NLL and accuracy with zero evaluated bp are NA; per-class conditional NLL and
accuracy with zero true midpoint support are NA. Precision/recall use their
respective explicit denominators; F1 is NA when both true/predicted support are
zero, consistent with existing confirmatory metrics.

Short-tract eligibility is conditioned on original exact length <=10000 bp and
both true switch endpoints strictly inside coverage/domain. Domain-edge and
coverage-censored tracts are excluded from this full-tract diagnostic and remain
visible in the general tract records. The cutoff is inclusive.

Reports provide eligible true-tract count and bp, count with at least one retained
dense midpoint, zero-midpoint count, evaluated quadrature bp, unrepresented true
tract bp, and per-class support. A short tract without a retained midpoint has
zero evaluated bp and NA accuracy/NLL. For short readout only, an eligible
midpoint's original clipped cell is additionally clipped to that short tract.
Cells whose center is outside the short tract receive no short-readout weight.
This avoids scoring surrounding ancestry as support for a short tract.
Short readout remains approximate because the model is only evaluated at centers.

## Artificial tile edges and aggregation

Tile edges must be explicitly supplied, sorted, distinct, finite, and strictly
inside the original physical domain. Readout is classified by center distance
within 10000 bp inclusive of any artificial edge, otherwise interior. Each group
retains midpoint count, weighted bp, class supports, accuracy, and NLL.

Boundary groups reuse the global match. Recall is assigned by the true boundary
coordinate; precision by the predicted coordinate. A match crossing the
near/interior cutoff is explicitly counted as a cross-group match, not rematched.
Separate matched-true and matched-predicted denominators avoid an invented group
F1 for a pair crossing the cutoff.

Per-query and per-class records preserve support and NA values. Top-level
quadrature pools query-bp within one parent/checkpoint. Physical represented span
is reported once; pooled quadrature denominator is that span times query count.
Boundary totals retain per-query mean absolute switch-count error and query
support. No cell/query is treated as an independent parent or training seed, and
these helpers produce no uncertainty intervals or cohort-level inference.

## Validation

Only handmade partitions/logits and handmade tskit tables are used. Tests cover
same-ancestry topology merging, exact switches and query/node correspondence,
invalid/missing/ambiguous census paths, clipping and edge censoring, short full
tracts including zero midpoint support, tile-group denominators and cross-group
matches, unequal endpoint cells, unsampled strips, strict grid rejection, exact
truth changes inside cells, empty support, raw extreme logits, absent classes,
visible false boundaries without true switches, and one-to-one matching costs.

Run on anogena, from the repository after sourcing `scripts/project_env.sh`:

```sh
.runtime/env/bin/python -m pytest -q tests/test_dense_tract_metrics.py tests/test_confirmatory_metrics.py
```

No simulation, real dataset/truth/held-out reads, model inputs, GPU use, or package
installation is part of this validation.
