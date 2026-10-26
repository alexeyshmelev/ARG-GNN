# Figure 2 source-conformance closure, round 2

Date: 2026-10-26. Scope: the frozen `review_round2/TreeLoom_encoder` assets and their generator, compared by reading the frozen TreeLoom implementation. This is a source audit, separate from the independent image-only reader. Readability acceptance belongs to that reader.

## Verdict and frozen identity

The displayed Figure 2 operators, operands, dimensions and worked toy agree with the frozen stable-leaf model. Round 1 ambiguities listed below have explicit replacements in round 2. The figure does not claim that structural clade flags transport hidden states from neighboring trees. Only the stable-leaf memory does that here.

- Generator SHA-256: `8b39ebd716069309e253877a3e11c09b7bb26108d3af0040a6176052462aa7f4`
- PNG SHA-256: `52a97946732fc279aceb322b6633cd5cdbda54f04da8e51213520ceb3a678e05`
- PDF SHA-256: `d27001d0832dbdcef91bb8334efb59df73ee64a6df6435690d5ffccede3404ec`

## Round 1 ambiguity closures

| Issue | Frozen round 2 repair |
|---|---|
| First-round U/O could appear serial or disconnected | M has an explicit fork into separately labelled U¹ and O¹ matrices, each 9×32, enclosed as a pair. |
| Unclear correction operands | U¹ directly supplies S(d). Separate U¹/O¹ routes enter a pair port supplying Δ(c). The pair is not an additional sum or concatenation operation. Panel c expands the actual gather, sum and projection. |
| Unclear correction and second-round shapes | S and Δ are each labelled 9×32. Original B enters B+S+Δ. The second M explicitly forks into U²/O², each 9×32, then both continue to the leaf readout. |
| Unspecified weight-sharing scope | Panel b states that the maps are shared across nodes, trees and both rounds, with distinct upward and downward maps. Panel d states that its correction maps repeat across proper clades and intervals. These are separate parameter families. |
| Leaf initialization implicit | A pictured zero32 vector now supplies the leaves' aggregate A. Upward leaves still receive the learned node update. The downward root row remains explicitly zero. |
| Worked T₂ gather could look like the only gather | Its enclosing frame is marked t=1,2,3. The V₁/V₂/V₃ stack therefore contains independently computed tree-local contexts. |
| Only one memory return appeared explicit | The Δ scatter has an attached repeat bracket for all t=1,2,3. |
| Metadata R collided with group-average R₂ | The group mean is now J₂ throughout, including J₂−F₂. |
| Structural flags lacked visible derivation | The current T₂ column is outlined, adjacent presence is 0/1/1, g=1−prev, l=1−next, and e=1 for a consecutive run of at least two. The proper-clade range 2≤|C|<5 is shown. |
| Length-scale constant was hard to parse | It is written as 100 + 99,900 × sigmoid(θ). |

## Formula and operand checks

1. **Two complete tree rounds.** The frozen `encode_trees` obtains U¹/O¹ from B, calculates the two local structural adapters and memory, scatters memory at its representatives, then calls the same message operator with B+S+Δ and unchanged gates. It finally selects leaf rows from U²/O². Thus the scaffold has the right dependencies.
2. **Upward update.** For child c of parent v, the saved message is `g_up(c) ⊙ Linear(U(c)) / degree(v)`. Their sum is A(v). The node result is `LayerNorm(B(v) + tanh(Linear(concat(B(v), A(v)))))`. The worked parent b has two children. The 2×32 messages, sum32, concat64 and update32 match source.
3. **Downward update.** The context for r1 is `O(b)+B(b)+A(b)−m(r1)`, where m is its already normalized saved message. A bias-free 32→32 map, downward gate, 32→32 update, tanh and LayerNorm follow. No extra downward residual is pictured or implemented. The all-node matrices retain intermediate nodes as well as leaves.
4. **Cut and forcing.** The real cut is 350 generations. Representatives are the highest real ancestors at or below the cut. The worked T₂ representatives q/b/a and the three assignment rows are correct. Gathered U¹/O¹ have shape3×32, their sum is projected to F₂:3×8, then copied to V₂:5×8. The r1 sequence is F₁[d], F₂[b], F₃[b], with separate tree indices.
5. **Memory.** The learned length is `100+99900 sigmoid(θ)` and α_t is `exp(−L_t/length)`. Each direction starts from its endpoint forcing. Each subsequent state is `α_t previous + (1−α_t) current_forcing`, using the current interval in both directions. Source uses the equivalent numerically stable `−expm1(−L/length)` for the second coefficient. Mean directions gives one 8-vector per leaf. Mean leaves within the current group gives J_t. The centered J_t−F_t is projected8→32 and scattered only at current representatives.
6. **Structural adapters.** Both consume only current U¹(b). Three distinct bias-free32→8 maps form x/y/z for `g(x⊙y)−l(x⊙z)`, followed by an8→32 map. A separate family forms x′/y′/z′ for `x′⊙y′+z′`, followed by its8→32 map and eligibility e. For the pictured interior T₂, C={r1,r2} has gain1, loss0 and eligibility1. The two residuals add at b. Repeating current proper-clade occurrences yields S.

The toy has five age-zero leaves and nine nodes per tree. Shapes are examples, not restrictions on production tree size. Hidden cell shades are schematic and are not measured model activations. Boundary gain/loss uses the source's zero policy for a missing adjacent interval. Older-than-cut samples use singleton groups, and duplicate unary clades use the deepest representative. Those general policies do not change the depicted interior example.

## Source and execution record

Read-only reinspection covered:

- `research/reviews/frozen_benchmark_training_ready_20261017_v1/european_frozen_treeloom_v1.py`, frozen model binding and `encode_trees`
- `src/arggnn/anonymous_clade_gnn_v1.py`, `prepare_tree_base_v1`, `sweep_tree_state_v1`, `leaf_context_v1`
- `src/arggnn/mutation_routed_whole_tree_gnn_v1.py`, learned map dimensions and message equations
- `src/arggnn/anonymous_clade_flag_control_v1.py`, structural flags and local gain/loss adapter
- `src/arggnn/ordered_clade_episode_gnn_v1.py`, consecutive presence groups and the selected `own_state` adapter
- `research/reviews/order2_cut_memory_implementation_20261015_v1/order2_cut_memory_v1.py`, cut representatives and selected `stable_leaf_memory` recurrence

This closure used source reads, a text diff of the two frozen generators, and file hashes. No model was imported or executed. No model computation, simulation, training, benchmark, GPU operation, or scientific result was changed or run. Round 1 and round 2 snapshots and canonical paper assets were not modified.

## Final label clarification for round 3

The working encoder was rerendered after two label-only clarifications: the last downward block now explicitly says Linear32→32, tanh, LayerNorm on three aligned lines, and the learned θ block explicitly shows shape1×1. The one scalar feeds the single length-scale transform that is reused for every interval. No mathematical operator, operand, dimension, dependency or model code changed. Visual inspection of the final full PNG found both labels contained within their existing blocks without overlap. The earlier review snapshots remain unchanged.

Final working assets frozen for the round 3 snapshot:

- Generator SHA-256: `35570bf9962d7576cf28eac5511570f72dd76ae8310e5620444ab43fb672a551`
- PNG SHA-256: `713dd68e7d55c01792091c2bd76995243e58e7c7c499fbbb4d0ebef3f3c01a6c`
- PDF SHA-256: `83c700836849c1cc537fa8c43cff5816da648d126526537d0dc4b443e775f90c`

Only the figure generator, its rendered working assets and this audit were updated. Rendering used CPU plotting code, with no model imports, model computation, simulation, training, benchmark, GPU activity, canonical-paper change or commit.
