# H036 PyTorch Geometric implementation v1

This is an implementation change to the existing H036 computation, with no new ancestry architecture or accuracy claim. The separate source is [mutation_routed_pyg_gnn_v1.py](../src/arggnn/mutation_routed_pyg_gnn_v1.py), public class MutationRoutedPyGGNNV1. The original source and running frozen models are unchanged.

## API and checkpoint compatibility

Constructor arguments exactly match MutationRoutedWholeTreeGNNV1:

~~~python
from arggnn.mutation_routed_pyg_gnn_v1 import MutationRoutedPyGGNNV1

model = MutationRoutedPyGGNNV1(
    hidden_dim=32, expected_haplotypes=280,
    site_chunk_size=256, checkpoint_sites=True,
)
model.load_state_dict(legacy_model_state_dict, strict=True)
logits = model(bank, control="aligned")  # original [readouts, queries, 4]
~~~

The input remains MutationRoutedWholeTreeInputsV1. The cached-parent PyG Dataset supplies that bank with data.to_model_inputs(). The complete per-parent forest is one item; this model does not require PyG auto-batching, index offsets, targets or added observation arrays. forward_holds(), forward(), encode_sites() and all site pooling/checkpointing, leaf features, support controls, normalization, full-parent bidirectional GRU and classifier retain the original implementation.

There are exactly 29,764 learned parameters at width32, 38 state-dict keys, and the same named-parameter order. Both new PyG helper modules have zero parameters/buffers. Learned linear weights are passed into propagation as tensors rather than registered again beneath helpers, preventing duplicate checkpoint aliases. The original up_message/down_message modules remain their sole owners.

The legacy checkpoint's outer dictionary format remains the runner's responsibility: extract its existing model state field before strict load. This source does not invent a checkpoint format or claim an actual trained-checkpoint/GPU comparison.

## Actual PyG graph operations

The upward helper subclasses torch_geometric.nn.MessagePassing and uses SumAggregation. For one topological level, its compact bipartite edge_index maps one source row per native child to the child's destination parent rank. Its message() computes exactly

m_child = Linear_up(u_child) * sigmoid_gate_child / degree_parent,

then PyG sums messages by parent. These exact edge messages are returned for the downward exclusion calculation. The upward residual update and LayerNorm remain unchanged.

The downward helper also uses MessagePassing/SumAggregation. Each compact source row holds the appropriate parent-side context for one native parent-to-recipient edge. Its message() subtracts that recipient's exact upward contribution, then computes

d_child = Linear_down(outside_parent + base_parent + sum_messages_parent - m_child) * gate_child.

PyG routes and sums one parent message to each child, followed by the original down_update, tanh and LayerNorm. Recipient subtree information only re-enters the final leaf context. No child information is added to the parent-side outside recursion.

Cached topological levels retain all native edges and valid children-before-parent/parent-before-children order. Compact propagation only allocates level-sized edge indices, gathered features and aggregate outputs. Whole-forest upward, message-sum, edge-message and outside storage is allocated once per encode_trees call, not once per depth. Site/hold pooling and unique leaf feature injection retain their original index_add operations; those are feature placement/reduction, while tree message propagation is performed by PyG itself. This is more than a Data wrapper around the former graph aggregation.

## Qualification and limits

The [CPU FP32 receipt](../research/reviews/mutation_routed_pyg_gnn_v1_author/toy_check.json) covers a handmade forest of41 nodes,3 trees,8 persistent leaves,7 original sites,4 queries and5 readouts. It includes binary/unbalanced trees, unary branches, a polytomy, nonidentity sample-node joins, heterochronous leaves, missing alleles, an inactive donor channel, an empty-site hold, site checkpoint recomputation and both aligned/shuffled support.

PyTorch2.6.0+cu124 / PyG2.7.0 passed strict state loading and exact parameter/key/order checks. For both controls, maximum absolute output, loss and every parameter-gradient differences were0. All parameter gradients were finite and present. No optimizer, target/data run, GPU execution, package installation or Git operation occurred in this qualification.

Reproduction:

~~~bash
source scripts/project_env.sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .runtime/env/bin/python \
  research/reviews/mutation_routed_pyg_gnn_v1_author/toy_check.py
~~~

The root owns actual cached-whole-parent FP32 forward/backward, trained-checkpoint, runtime/memory comparison and GPU launch. CUDA aggregation can differ in reduction order, so tiny CPU exact equality is not a GPU bitwise guarantee. No full fit, speedup, memory improvement, PyG compile support or predictive benefit is asserted. The donor-flux architecture proposal remains separately saved for future selection.
