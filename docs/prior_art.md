# Prior art and contribution boundaries

Status: literature and experimental design, checked 2026-10-10. No benchmark
results or architectural novelty conclusions are asserted here.

## Closest comparisons

| Source | Established capability | Consequence for this project |
| --- | --- | --- |
| [Graph Set Transformer](https://arxiv.org/abs/2606.05116), [code](https://github.com/daenuprobst/gst-conference) | Interleaves local GCN propagation, graph pooling, graph-token attention and gated feedback into nodes at every layer | Interleaving graph and collection context is already its central contribution. Positional encoding alone is a weak extension |
| [EvolveGCN, AAAI 2020](https://arxiv.org/abs/1902.10191) | Evolves GCN parameters through recurrence, including sequences with completely different node sets | Varying internal vertices is not a new capability |
| [GraphSSM, NeurIPS 2024](https://arxiv.org/abs/2406.00943), [full text](https://arxiv.org/html/2406.00943), [MIT code](https://github.com/EdisonLeeeee/GraphSSM) | Continuous-time graph state-space dynamics, irregular observation gaps and zero-order-hold discretization | Direct comparator for interval processing |
| [DySAT](https://arxiv.org/abs/1812.09430) | Structural attention and identity-aligned temporal node attention | Identity-aligned temporal attention is established |
| [STAGIN, NeurIPS 2021](https://arxiv.org/abs/2105.13495), [GPL-3 code](https://github.com/egyptdj/stagin) | GNN readouts and a Transformer produce representations of temporal brain-graph sequences | Graph-sequence classification using pooled temporal attention is established |
| [Graph Matching Networks, ICML 2019](https://arxiv.org/abs/1904.12787) | Joint graph-pair reasoning with node cross-graph attention | Learned cross-graph correspondence is established, although the prediction task differs |
| [TGNN4I, AISTATS 2023](https://arxiv.org/abs/2302.08415) | Graph-aware recurrent updates and continuous per-node dynamics for irregular, partially observed series | Irregular spacing and missing observations are established |
| [STG-NCDE, AAAI 2022](https://arxiv.org/abs/2112.03558) | Spatial and temporal neural controlled differential equations | Continuous-coordinate graph sequence modeling is established |
| [MP-SSM, 2026 revision](https://arxiv.org/abs/2505.18728) | Permutation-equivariant message-passing state-space models for static and temporal graphs | An efficient equivariant SSM-GNN is too broad a contribution claim |
| [Temporal Graph Kernels](https://arxiv.org/abs/1911.05496) | Temporal graph classification through reduced, directed line-graph and static expansion representations | A lifted spatial-plus-temporal graph is a necessary architectural control |
| [Dynamic temporal graph warping](https://arxiv.org/abs/1810.06240) | Temporal graph alignment combining temporal warping and vertex correspondence | Graph-sequence alignment is established |
| [Time-Variant Graph Classification](https://arxiv.org/abs/1609.04350) | Classifies graph sequences through discriminative graph-transformation subsequences | The graph-sequence classification problem is not new |
| [Filtration Surfaces](https://arxiv.org/abs/2309.03616), [code](https://github.com/aidos-lab/filtration_surfaces) | Dynamic graph classification with changing node sets and edge weights | Weighted dynamic graph classification has scalable alternatives |
| [Molecular set representation learning](https://doi.org/10.1038/s42256-024-00856-0) | Set representations of molecules, complexes and reaction participants | Collections of separate graphs already arise in molecular ML |
| [Distribution regression](https://arxiv.org/abs/1411.2066) | Kernel mean embeddings and learning from distributions and sets | Duration-weighted graph summaries are established baseline constructions |
| [Graph neural processes](https://arxiv.org/abs/1902.10042), [CGNP](https://arxiv.org/abs/1812.05212), [STGNP, KDD 2023](https://arxiv.org/abs/2305.18719) | Sparse context to target distributions, irregular functional data and graph-aware spatiotemporal extrapolation | Relevant to uncertainty, but not exact independent graph-collection classifier comparators |
| [DiffPool](https://arxiv.org/abs/1806.08804) | Learned hierarchical node clusters and coarsened graphs | Within-graph hierarchy must be distinguished from a collection of separate graphs |

[Graph-Mamba](https://arxiv.org/abs/2402.00789) and
[Graph Mamba Networks](https://arxiv.org/abs/2402.08678) principally sequentialize
nodes or neighborhoods within a static graph. Their graph-sequence terminology
should not be confused with a sequence of separate graph objects.

## Direct GraphSSM counterexample

GraphSSM Appendix B.2 explicitly handles a changing node set on a sequence of
intervals. Equation (21), the memory alignment procedure, retains the state of
vertices in the intersection of consecutive node sets and initializes new
vertices to zero or a neighbor-derived state. It discusses zero-order-hold
updates for newly appearing vertices. Thus stable sample identities combined
with changing ancestral vertices do not by themselves distinguish ARG-GNN.
Its continuous-time foundation also supplies semigroup behavior closely related
to exact subdivision consistency. This comparison must be addressed directly.

A future GraphSSM adaptation should implement the paper equations, supply
observed genomic gaps instead of silently learning a gap solely from features,
use Appendix B.2 alignment for persistent samples, and document how graph-level
or fixed-coordinate predictions differ from its published node-classification
experiments. No exact GraphSSM reproduction is presently claimed.

## Hypothesis to test

The project studies interval-bearing graph-valued signals with a persistent
subset of nodes and graph-local anonymous nodes. Independent tree source banks
and lineage-conditioned, interval-integrated attention may be useful. Whether
that operator is distinct and beneficial remains unvalidated.

The precise representation property is that replacing an unchanged graph on
[a,b) by identical graph copies on [a,c) and [c,b) preserves predictions at fixed
coordinate queries, while total interval mass and sample correspondence remain
unchanged. Midpoint quadrature is not generally exact under this transformation.
A nonlinear update executed once per serialized graph also generally breaks it.
Closed-form interval integration or a continuous-coordinate operator is needed
for an exact claim. The property itself must not be called newly invented.

A sequence of separate graphs generally allows independent vertex relabeling
unless correspondence is supplied. A dynamic graph usually represents changing
relations among persistent entities. Marginal ARG trees lie between these
settings, with stable samples, variable ancestors and genomic coordinates.
This motivates the formulation but does not remove temporal-graph comparisons.

## Required controls and claim gates

1. Shared whole-tree encoder followed by duration-weighted DeepSets.
2. The same observed inputs and identities with a late temporal Transformer.
3. GST-ca and an explicitly labeled coordinate/span-aware GST adaptation.
4. An independently implemented spatial-plus-leaf-correspondence supra-graph
   MPNN, including interval/coordinate features.
5. A documented GraphSSM adaptation before a strong general interval-model claim.
6. A domain baseline on the cross-domain benchmark.

The native PyTorch controls in `src/arggnn/temporal_baselines.py` are independent
baseline families, not published-method reproductions. They must be tested for
node relabeling and serialization invariance with intervals retained. Order
sensitivity means rearranging graph contents along genomic coordinates, not
merely reordering their serialized storage.

Use identical observed features, masks, coordinates, leaf identities and query
locations across models. Never feed truth-derived labels or annotations into an
encoder. Match tuning budgets and report parameter count, throughput and memory
alongside accuracy. Split replicates before generating windows. Separate
hypotheses, implementation checks, pilot observations and validated results.

GST source was inspected at main commit
`6469d56aa131463325058b897b1e435c6d2edcd2`. GitHub metadata declared no code
license at inspection time. Filtration Surfaces likewise had no declared license.
Visibility of source does not by itself establish reuse permission. Independent
paper-based implementations should state their adaptations. No source copied
from those repositories is part of the temporal controls.

## Implemented temporal-control API

`LeafTemporalTransformer(input_dim, num_classes, hidden_dim=64, num_layers=2,
temporal_layers=2, num_heads=4, dropout=0.1, validate=True)` uses the final
`SharedGraphEncoder` bank, each tree's whole-node mean and query-leaf features.
A per-query-leaf bidirectional Transformer uses interval endpoint, midpoint and
span features. Fixed-position decoder queries combine observed local leaf
features with coordinate encodings and attend to the complete temporal memory.

`LeafCorrespondenceSupraMPNN(input_dim, num_classes, hidden_dim=64, num_layers=2,
supra_layers=2, dropout=0.1, validate=True)` uses the same shared encoder and
native tree edges. It adds bidirectional stable-leaf correspondence edges between
genomic neighboring graphs, including reference leaves. Separate relation
messages use coordinate gaps and spans. Its temporal receptive field is finite,
controlled by `supra_layers`. It reads containing-tree leaves at the fixed query
positions. An absent leaf is not bridged across an intervening graph.

Both accept the shared `GraphSequence` tensor contract and return `[P,Q,K]`.
They support an optional supplied encoder with the shared bank API. Clades are
validated as shared metadata but are not used as extra learned inputs in these
controls. Integer IDs select correspondence and are not learned features.
Both sort or join using supplied intervals, so changing serialization while
retaining those intervals does not change the modeled genomic signal. Moving
graph contents to different fixed intervals is a substantive order change.
They are noncausal offline inference baselines. Neither is labeled an exact
published reproduction or an exactly subdivision-consistent model.

CPU verification was run with `CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=1` and
`MKL_NUM_THREADS=1`, after sourcing `scripts/project_env.sh`. The focused command
is `python -m pytest tests/test_temporal_baselines.py
--basetemp=/workspace-SR003.nfs2/ARG-GNN/.runtime/tmp/pytest-prior -q`.
Checks cover output shape, finite training gradients, node-index permutation,
stable-ID renaming, coordinate-preserving graph serialization, actual order
sensitivity, query/position order, one graph, and nonlocal-context dependence.
These are implementation checks, not empirical performance findings.

The focused CPU suite completed with **14 passed in 2.46 seconds** on
2026-10-10. No GPU context was created for this verification.
