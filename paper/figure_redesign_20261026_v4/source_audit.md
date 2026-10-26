# Source agreement

Read-only audit by a separate agent, 26 October 2026. No model, dataset or result was changed.

## Initial representations

A shared site MLP processes observed features separately for each site and leaf. Mean and maximum pool called sites within each tree interval, separately for each leaf. A projection also receives call count, an empty indicator and interval span. Reference ancestry with an unknown flag and sampling age join the pooled representation in the leaf projection. A separate MLP initializes every node from age, root and sampled-leaf indicators. Encoded leaf evidence is added only to leaf rows.

The diagram summarizes these operations without enumerating all 19 site features or the auxiliary pooling scalars. These remain available in the method and code. It must not imply pooling different leaves together or giving observed alleles directly to internal ancestors.

Sources: `src/arggnn/mutation_routed_whole_tree_gnn_v1.py` and `src/arggnn/anonymous_clade_gnn_v1.py`.

## Message passing

Two complete upward and downward rounds reuse parameters. Upward aggregation uses gated transformed children and the arithmetic mean over children. Downward aggregation combines parent outside state, parent base and other branches. The root outside state starts at zero. Both rounds retain every node. The final leaf embedding concatenates upward and outside states, then projects 64 to 32 with tanh.

Nine observed branch features supply separate directional gates. Four describe branch geometry and five summarize agreement of observed allele partitions with branch partitions. This is not inference or routing of simulated mutation placements. The paper phrase about inferred mutations was corrected to observed allele partitions and inferred branches.

Sources: `src/arggnn/anonymous_clade_gnn_v1.py` and `src/arggnn/mutation_routed_tree_observations_v1.py`.

## Sample memory

Ascend from each sampled leaf while the parent is no older than 350 generations. A sample older than the cut remains a singleton. Sum upward and outside states at the selected representative, project 32 to 8 and copy to the group's leaves. Track each leaf in both genomic directions using exponential memory. Average directions per leaf, then average leaves in the current group. Subtract the current group context and project 8 to 32. Add this correction only at current group representatives before the second round.

The preserved T2 example has representatives 0, 8 and 6 for groups {q}, {r1,r2} and {r3,r4}. The original hardcoded list omitted node 6. The new drawing derives representatives from the actual saved tree and ages.

Source: `research/reviews/order2_cut_memory_implementation_20261015_v1/order2_cut_memory_v1.py`.

## Structural correction

Exact descendant-leaf sets determine proper-clade correspondence. Gain means absent from the preceding tree. Loss means absent from the following tree. Persistence means a consecutive run of at least two trees. These flags can overlap. They select factored learned transforms of the current upward state. The frozen own-state model does not transfer neighboring hidden states on this branch. Correction rows belong to current clade representatives, with zeros elsewhere. The diagram's G, L and P abbreviate the exact factored maps defined in the supplement, with separate persistence parameters.

Sources: `src/arggnn/anonymous_clade_flag_control_v1.py`, `src/arggnn/ordered_clade_episode_gnn_v1.py`, and `research/reviews/frozen_benchmark_training_ready_20261017_v1/european_frozen_treeloom_v1.py`.

## Prediction

For each query leaf, stack local embeddings in tree order. Append interval length and distances to previous and next interval centers, transformed by log(1+x)/log(2000001). Missing neighbors contribute zero. A one-layer bidirectional GRU maps 35 inputs to 64 contextual features. Concatenate original local32 and context64, then apply Linear96-to-32, GELU, Linear32-to-4 and softmax for the pictured European condition. Matrices and probabilities in the figures are illustrative.
