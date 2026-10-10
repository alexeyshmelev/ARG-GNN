# GST graph-set baseline adaptations

## Source and scientific status

This is an independent implementation based on Jose E. Escrig Molina, Baoquan Chen and Daniel Probst, **Graph Set Transformer**, [arXiv:2606.05116v1](https://arxiv.org/html/2606.05116v1), posted 3 June 2026. The original HTML reports CC BY 4.0. The original equations were read before implementation. This is a documented baseline adaptation, not an exact reproduction of the authors' code, tuning or reported results.

Official HTML, abstract metadata and PDF were reacquired with HTTP 200 responses into `.runtime/cache/gst-baseline/`. The source manifest records URLs, version, byte sizes and locally computed SHA256 values. The HTML's mathematical `alttext` was preserved in the cached text extraction rather than dropping equations.

| Artifact | SHA256 |
| --- | --- |
| Original v1 HTML | `0f40b097d0ff5b7958cd05554886062cafdf0fff6e7ae0120aaa2ce71162af14` |
| Original v1 PDF | `6b77a1c3a5d9c569aefcd99d1e2337f913df097774bea6e8231177e03ef50b3a` |
| Abstract metadata HTML | `cb189bc9c49a5aa187d04c815e0a67f90ca17da445a239ee33bd4f9d2e6d1fe1` |

The [public GST source](https://github.com/daenuprobst/gst-conference/blob/6469d56aa131463325058b897b1e435c6d2edcd2/src/graph_set_transformer/models/gst.py) was inspected read-only at commit `6469d56aa131463325058b897b1e435c6d2edcd2`. The inspected file has SHA256 `bc08c15d55247b32bda98e4b03fcaee988ceb8f654423240a33ca632ce11ef72`. No source code was copied or installed. The repository had no declared code license in the earlier project audit. The implementation uses native tensor operations and PyTorch attention, without adding PyTorch Geometric or any other dependency.

## Paper-to-implementation mapping

One `GraphSequence` is one set of marginal graphs. Blocks have distinct parameters. Each block receives the previous block's **context-updated node states**, so feedback participates in the next local graph update.

| Paper equations | Implemented operation |
| --- | --- |
| (1)-(2) | Normalized GCN, per-graph GraphNorm, SiLU, dropout, then residual node update |
| (3)-(5) | Learned `Linear-ReLU-Linear` scalar scores, stable softmax separately inside every graph, weighted node sum to one graph token |
| (6) | Pre-LayerNorm multihead graph-token self-attention and residual |
| (7) | Pre-LayerNorm graph-token FFN and residual |
| (8), GST-bc | Broadcast the contextualized token of the node's home graph |
| (9), GST-ca | LayerNorm node queries attend to all contextualized graph tokens in the set |
| (10)-(11), paper gate | Two sigmoid channel gates from `[local node; home token; set context]`, followed by `local + home_gate*home + set_gate*context` |
| (12) | Stack blocks, then an MLP on the selected query leaf |

GraphNorm includes learned mean subtraction: for graph `g`, `mu_g=mean(h_i)`, `r_i=h_i-alpha*mu_g`, and `var_g=mean(r_i^2)`. Normalized states are `gamma*r_i/sqrt(var_g+eps)+beta`, with `eps=1e-5`. This is per-graph normalization, rather than normalizing each node independently or pooling all graphs into one normalization group.

The GCN is explicitly `D^(-1/2) (A+sI) D^(-1/2) H W + b`, with incoming adjacency `A[dst,src]`. The default `s=2` follows the inspected source's `GCNConv(improved=True)` request. It is an explicit mathematical choice, not a tested claim of bitwise equivalence to a particular PyG version. Supplied self edges are replaced by exactly one loop of this declared weight. Other supplied directed edges are retained. The genealogy packer supplies both directions for tree edges. No cross-graph edges are added. This local operator differs from the existing shared mean MPNN and must be reported in comparisons.

## Paper/source discrepancies and explicit decisions

The v1 paper refers to `x_i^(2)` in equations (9)-(11), but defines only the post-GCN residual `x_i^(1)` beforehand. The inspected source has no intervening node FFN. We therefore explicitly use `x_i^(2)=x_i^(1)`. We do not invent an extra node stage.

The paper and pinned source implement materially different gates. Both are selectable and must be separately reported:

| `gate_variant` | Mathematical feedback | Gate conditioning and initialization |
| --- | --- | --- |
| `paper`, default | `local + sigmoid(a)*home + sigmoid(b)*context` | `[local;home;context] -> Linear(3D,D) -> SiLU -> Linear(D,2D)`. Final weights and bias are zero, yielding initial gates 0.5 |
| `source` | `local + home + sigmoid(b)*context` | `[local;context] -> Linear(2D,D) -> SiLU -> Linear(D,D)`. Final bias is -1, with ordinary weight initialization |

The paper's gate-initialization sentence is ambiguous about which named matrices are initialized to zero. This implementation uses zero **output-layer** weights and bias with ordinary first-layer initialization. That choice is explicit and should not be presented as a verified reproduction of the authors' initialization. The source-gate variant independently implements the inspected mathematical rule, rather than importing the authors' code.

Additional implementation choices are explicit:

- A learned linear input projection maps the common observed feature vector to hidden width. No extra input nonlinearity is added.
- The graph-token FFN uses hidden expansion factor 2 and SiLU, matching the inspected defaults. Both attention modules use the declared head count.
- The final scalar bias in the node-pooling score is omitted because a constant cancels in each graph's softmax. This removes one redundant parameter per block.
- Stochastic depth is fixed to zero. Declared dropout applies to GCN updates, attention probabilities, graph-token FFNs and the readout. Invariant deterministic outputs are assessed in evaluation mode with dropout zero.
- One set is processed per call. The paper's multi-set padded batching/masking is unnecessary for this tensor contract. We do not claim batched throughput equivalence.
- The paper includes node-level tasks already. The task adaptation here selects stable query leaves and their containing graphs at fixed coordinates instead of returning every atom/node or a set-level label. The output is `[P,Q,K]`. Coordinates within the same marginal graph yield the same query-leaf output.
- This is an offline whole-window set model. Its graph context uses every observed graph in the supplied tile. It is not a causal forecasting model.
- Default dimensions and optimization budgets are project choices. No published data, parameter-parity sweep, optimizer schedule or result is reproduced. Equal width/depth does not imply equal parameter count or identical local graph operators.

## Separately named record and span controls

Both weighting choices use exactly the same learned parameters and state-dictionary keys. Their only difference is the measure assigned to **graph-token keys**, at every set self-attention and node-to-set cross-attention layer.

For projected query/key scores `s_ij`, attention is

```
alpha_ij = m_j * exp(s_ij) / sum_k m_k * exp(s_ik).
```

`record` sets `m_j=1`, the counting measure implicit in the paper's ordinary softmax over graph tokens. `span` sets `m_j=b_j-a_j`, the observed interval width. Implementation adds `log(m_j)-log(sum_k m_k)` to key logits. The common normalization constant cancels in softmax. Widths are computed in interval dtype before casting the log bias to feature dtype. Within-graph attentional pooling remains unchanged and normalized separately per graph.

| Control name, paper gate | Routing | Graph-token measure |
| --- | --- | --- |
| `gst_ca_record` | Node-to-set cross-attention | Counting measure |
| `gst_ca_span` | Node-to-set cross-attention | Interval width |
| `gst_bc_record` | Home-token broadcast after set attention | Counting measure |
| `gst_bc_span` | Home-token broadcast after set attention | Interval width |

Selecting `gate_variant="source"` adds `_source_gate` to these names. The paper recommends broadcast as its cheaper default. Both routes are available here. The common constructor defaults to cross-attention for the project's GST-ca comparison.

These are **order-blind set controls**. The record variant ignores interval endpoints and widths. The span variant uses width only. Neither uses endpoint order, distances, scoring-position encodings, clade correspondences or learned numerical ID embeddings. Moving equal-width source graphs to different genomic locations while holding a target graph fixed leaves its predictions unchanged. More generally, permuting graph/width pairs preserves the span model. Moving a graph to a different width changes the weighted set's measure and can change its output. IDs only locate query leaves and their externally declared column order. The common reference adapter uses within-tree TMRCA summaries, so it does not introduce genomic order into these controls.

With deterministic evaluation, splitting an unchanged graph interval into identical copies preserves `span` outputs at common fixed coordinates. Initial node states, graph-normalized updates and pooled tokens are identical for the copies. Additive key masses preserve attention sums, pointwise token FFNs and gates preserve equality, and this argument repeats at every block. Ordinary counting measure generally changes when one record becomes two. The CPU tests exercise both behaviors. This statement concerns identical input graph copies and observed widths. It is not a learned ancestry accuracy claim, and nonzero dropout draws need not produce pathwise identical outputs.

## Distinction from the existing span-set-lineage control

| Property | GST adaptations | Existing `span_set_lineage` |
| --- | --- | --- |
| Cross-graph source representation | One learned attentional token per graph | All observed node states, with graph-node count normalization |
| Set interaction | Graph tokens contextualize one another through MHSA at every block | Query-conditioned node attention reads independent local source banks |
| Feedback | All nodes in every graph update before the next GCN | Target state feeds its next local update, while source banks remain independent |
| Local graph update | Symmetric normalized GCN and GraphNorm | Shared incoming-neighbor mean MPNN and LayerNorm |
| Leaf/clade correspondence in attention | None | Stable-ID equality and clade overlap biases |
| Interval information | Counting or width measure only | Width measure in the span control, distance-integrated kernel in the interval control |

A positive GST comparison would not establish that interleaving itself is new. Both GST and the project controls already interleave local and collection information. Likewise, subdivision invariance of the span adaptation alone is not a novelty claim. The architectural and correspondence differences, and the learned-versus-independent source banks, need separate evidence.

## API and integration boundary

```python
from arggnn.gst_baseline import (
    GSTAdaptation, GSTRecordTokenAdaptation, GSTSpanTokenAdaptation,
)

record = GSTRecordTokenAdaptation(
    input_dim=48, num_classes=4, hidden_dim=32, num_layers=2,
    num_heads=4, dropout=0.0, routing="cross_attention", gate_variant="paper",
)
span = GSTSpanTokenAdaptation(
    input_dim=48, num_classes=4, hidden_dim=32, num_layers=2,
    num_heads=4, dropout=0.0, routing="cross_attention", gate_variant="paper",
)
logits = span(prepared_graph_sequence)  # [P,Q,K]
provenance = span.configuration_provenance()
```

`GSTAdaptation` also accepts `token_weighting="record"` or `"span"`. The separately named classes fix this argument. Constructors accept the existing `input_dim`, `num_classes`, `hidden_dim`, `num_layers` and `validate` arguments. `head` is an MLP whose last linear layer can be replaced by the project's regression wrapper. No registry, frozen protocol or GPU schedule was changed by this subtask.

## Recorded CPU verification

`tests/test_gst_baseline.py` completed with **27 passed in 2.94 seconds**. Tests cover both routes, both gate choices and both measures, finite outputs and all parameter gradients, node-index permutations, arbitrary stable-ID renaming, graph serialization, query-column permutation, actual source-content rearrangement at a fixed target, global reflection/translation/rescaling, absence of clade/ID embeddings, one graph with empty edges, the declared GCN/GraphNorm operator, and rejection of invalid cross-graph edges. A hook on the second block's GCN verifies that changing a distant graph already changes the local target state **before** that GCN update. Span refinement passes while record-count refinement changes outputs.

A CPU smoke check also used the actual prepared inferred v3 training parent 0, first global tile, without an optimizer step. The tile has 30 inferred marginal graphs, 1,890 nodes, 48 common input features and observed bounds `[0.000293,0.125]`. Each of the four paper-gate controls returned `[16,8,4]`, with finite outputs, loss and all parameter gradients. For width 32, depth 2 and four heads, broadcast controls have 34,756 parameters and cross-attention controls have 43,332. Record and span counts are identical within each route.

`docs/gst_baseline_provenance.json` binds that smoke input's manifest and prepared tensor, original paper hashes, the independent implementation hash, all constructor choices, and measured single-call CPU times. Those times include differing first-call overhead and are implementation diagnostics, not a comparative throughput benchmark. No GPU job, package installation, training run, commit or publication was performed by this subtask.
