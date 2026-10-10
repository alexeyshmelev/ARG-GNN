# Experimental architecture and API

## Scientific status

The module is a first testable implementation. It contains no validated accuracy or efficiency claims. Novelty is speculative. Interleaving, gates, temporal graph attention, graph matching and continuum attention all have prior art. The narrow hypothesis is that structural correspondence between interval-bearing genealogies improves ancestry estimation beyond distance-aware smoothing, while respecting redundant interval segmentation.

Relevant sources:

- Graph Set Transformer: https://arxiv.org/html/2606.05116
- ARGMix, Graph transformer for ancient ancestry inference: https://www.biorxiv.org/content/10.64898/2026.04.05.714076v1.full
- ARGformer: https://doi.org/10.1093/bioinformatics/btag438 and https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13501305/fullTextXML
- AncestralPaths in The selection landscape and genetic legacy of ancient Eurasians: https://www.nature.com/articles/s41586-023-06705-1
- Continuum Attention for Neural Operators: https://arxiv.org/html/2406.06486v4
- DySAT: https://arxiv.org/abs/1812.09430
- TGAT: https://arxiv.org/abs/2002.07962
- TGN: https://arxiv.org/abs/2006.10637
- Graph Matching Networks: https://arxiv.org/abs/1904.12787

## Shared contract

`arggnn.data.GraphSequence` owns the dataclass. `arggnn.models.GraphSequenceInput` is a structural Protocol, with no data-module import dependency. Every model consumes one independently simulated sequence at a time. All tensors share a device.

| Field | Shape | Meaning |
| --- | --- | --- |
| `x` | `[N,F]` floating | Observed node features. No truth ancestry, migration labels or trainable ID embeddings |
| `edge_index` | `[2,E]` long | Directed local graph edges. Supply both directions for undirected tree message passing |
| `node_graph` | `[N]` long | Graph indexes in `[0,G)`, every graph nonempty |
| `node_ids` | `[N]` long | Stable leaf identity, unique inside each graph. Internal nodes use `-1` |
| `clades` | `[N,C]` | Descendant landmark indicators in `[0,1]`. A fixed shared landmark panel defines columns |
| `intervals` | `[G,2]` floating | Positive-width tree intervals forming a complete nonoverlapping tile `[a,b]` within global `[0,1]`. Storage order may be arbitrary |
| `query_node_ids` | `[Q]` long | Unique nonnegative stable leaf IDs. Each occurs exactly once in every graph |
| `positions` | `[P]` floating | Query coordinates in `[0,1)`, chosen independently of how intervals are serialized or split |
| `target_graph` | `[P]` long | Graph containing the corresponding position |

An optional `window_bounds[2]` tensor declares the tile endpoints and lets validation detect missing left or right range. If bounds are absent, the declared range defaults to the full chromosome `[0,1]`. Partial tiles therefore require explicit `window_bounds` and missing outer range is rejected. Ancestry labels `[P,Q]` remain outside this object. Outputs are logits `[P,Q,num_classes]`. Physical or genetic coordinates must be chosen and recorded consistently. The current implementation uses one normalized coordinate, not both. Coordinate scale determines the learned kernel scale. The lead initializes `tau` to 0.05 by default.

Constructors share `input_dim`, `num_classes`, `hidden_dim=64`, `num_layers=2`, `validate=True`. Use `build_model(name, input_dim, num_classes, **kwargs)` and `count_parameters(model)`. Matching local encoders does not by itself match complete model parameter counts. Report counts and tune width or budgets explicitly in controlled experiments.

## Implemented baselines

- `independent`: shared incoming-mean MPNN, query-leaf linear classification independently for each local tree
- `set_pool`: the same encoder, per-graph mean pool, span-weighted set context, then late concatenation with the query leaf
- `gru`: the same encoder, leaf-aligned bidirectional GRU in genomic order, with interval widths and midpoint gaps as explicit distance inputs

These are deliberately small baselines, not reproductions of published methods. In particular, none should be reported as ARGMix, ARGformer or GST. A GST-like graph-token attention/interleaving baseline, actual reference-conditioned ARGMix comparison and published method reproductions remain deferred.

## IntervalLineageGNN

Independent local encoder banks `b[l][N,D]` are computed at all depths. A target graph is instantiated at each query coordinate `x`. Feedback mode alternates a local MPNN step with attention from target nodes to all independent source-bank nodes. The source banks are never overwritten with context-conditioned states.

For source interval `I_j=[a_j,b_j)`, the kernel measure is

```
omega_j(x) = integral over I_j of exp(-abs(x-y)/tau) dy
```

The implementation computes `log(omega)` with `expm1` formulas, retaining finite log weights for distant or narrow intervals. The learned `tau=softplus(raw_tau)+1e-4` is positive. Each source node receives measure `omega_j / number_of_nodes_in_graph_j`, preventing graph cardinality alone from multiplying source mass.

The score is scaled query-key attention plus learned same-leaf and descendant-Jaccard biases. Identity relations are invariant under a common arbitrary renaming of stable leaf IDs. No persistent ancestor identity is assumed. For binary landmarks, Jaccard is intersection divided by union. All-zero clades have zero overlap.

The normalized source-weighted value message passes through a target-node gate and a residual LayerNorm. Classification reads the queried leaves. The first version uses one attention head per layer and dense node-to-node correspondence, with no dropout.

## Exact guarantee and limits

At deterministic evaluation, replacing a source interval by any disjoint partition carrying identical trees, features and correspondences leaves the output at every fixed query coordinate unchanged, up to floating-point arithmetic. The source node states repeat, while their interval masses sum to the original mass. Target-tree local propagation is unchanged. Induction through target attention layers proves the property because source banks stay independent.

Node ordering is equivariant. Graph serialization ordering is equivariant when interval and index metadata are carried with each graph. Query-leaf ordering is equivariant. The symmetric distance kernel also permits chromosome reversal with the associated interval and target mapping. This guarantee does not apply to changing genealogy, deleting distinct trees or resampling evaluation positions at new tree midpoints. It is a representation guarantee, not proof of predictive improvement.

The current model operates over all trees and nodes in the supplied sequence/window. It has no sparse neighborhood radius yet. It is intended for small pilot windows, not a whole chromosome in one call.

## Ablations

All lead variants retain the same parameter count.

- `no_correspondence`: removes same-leaf and descendant-clade biases
- `midpoint_weights`: uses interval width times an exponential kernel at the interval midpoint
- `equal_tree_weights`: uses the same midpoint kernel with equal mass per tree record, omitting interval widths
- `late_fusion`: completes all tree-local encoding before the same sequence of context blocks. It has no target-local propagation after context injection

The last three source weighting/fusion options can also be selected directly on `IntervalLineageGNN`. Late fusion with exact weights remains refinement invariant. Midpoint/equal variants generally do not.

## Cost and verification

Let `J` be tree count, `B` query-coordinate count, `n` average nodes per tree, `E` average edges, `d` width and `L` depth. Source banks cost `O(L J (E d+n d^2))`. Target local propagation costs `O(L B (E d+n d^2))`. Dense attention costs `O(L B J n^2 d)` for roughly equal graphs. At each target position the score tensor is about `[n,J*n]`, with source bank storage `O(L J n d)`. Autograd retains target computations across positions. This is not a linear-cost architecture. Future sparse matching/windows would require separate implementation and benchmarks.

`tests/test_models.py` uses CPU-only standard unittest. It checks variable graph sizes, finite gradients and logits, node and graph serialization permutations, arbitrary stable-ID renaming, query and landmark permutation, chromosome reversal and coordinate translation, exact interval refinement, known midpoint/equal failures, narrow-interval kernel stability and invalid ranges. Run after sourcing `scripts/project_env.sh` with the project Python interpreter. No GPU execution or installed-package modification is needed for these tests.
