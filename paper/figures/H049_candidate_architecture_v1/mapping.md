# H049 candidate architecture — source mapping

architecture.svg is editable SVG with native text, lines and shapes; no embedded raster. generate_svg.py regenerates it deterministically using Python's standard library. H049 is a candidate, not an accepted or validated method. No scientific data, checkpoint, truth or model execution was used. source_refs.json pins the exact source bytes.

Source aliases (exact repository paths):

- **core**: research/reviews/order2_cut_memory_implementation_20261015_v1/order2_cut_memory_v1.py
- **producer**: research/reviews/H049_order2_cut_memory_root_source_20261015_v1/producer_v1.py
- **tree**: src/arggnn/anonymous_clade_gnn_v1.py
- **flags**: src/arggnn/anonymous_clade_flag_control_v1.py
- **pyg**: src/arggnn/mutation_routed_pyg_gnn_v1.py
- **inputs**: src/arggnn/mutation_routed_whole_tree_gnn_v1.py
- **decoder**: src/arggnn/marker_grid_reset_chain_gnn_v1.py

| SVG block | Exact source functions / lines | Meaning |
|---|---|---|
| A1-complete-trees | inputs MutationRoutedWholeTreeInputsV1, 29–50; core prepare_cut_plan_v1, 150–251; producer load_observations_v1, 159–180 | Ordered native tree holds with full dated topology and all 952 persistent sampled-leaf columns. The drawn five-leaf trees are schematic; their local internal node IDs are not matched across positions. |
| A cut rings / line | core prepare_cut_plan_v1, 197–251 | Age ≤350: highest actual ancestor at/below 350, including a root if saturated. Age >350: actual sampled node singleton. Disjoint groups use sample membership; unchanged membership retains its cut episode. The line neither prunes trees nor creates virtual nodes. |
| B1 candidate | core CutMemoryV1._direction, 399–402; _direction_layouts, 121–147 | Transport m(A,B) to m(B,C) as sum_A [n(A∩B∩C)/n(B∩C)] m(A,B). A, B, C are consecutive distinct cut partitions in traversal order, not output class labels. |
| B2 adjacent null | core CutMemoryV1._direction, 403–407 | Pool within B using n(A∩B)/n(B), then copy into each outgoing (B,C) cell, removing dependence on C within B. |
| B3 stable leaf | core CutMemoryV1._direction, 386–393 | Retain one rank-8 state per persistent sample identity; forcing follows its current cut group. No pair-cell remapping; same final group readback. |
| B matched capacity | core CutMemoryV1.__init__, 351–362; Order2CutMarkerGridV1.__init__, 439–448; make_matched_family_v1, 503–530 | Each model has 31,933 common + 513 memory = 32,446 trainable parameters. Identical H042 pass30 warm start and fresh initialization; separate model copies and independent AdamW. Weights are not tied across arms. |
| C1-inputs | tree prepare_tree_base_v1, 90–108; inputs _site_chunk, 281–305, encode_sites, 307–341 | Categorical SNPs/callability and observed proxy frequencies feed sites; proxy labels/unknown mask and sample ages feed leaves; observed node dates/root/sample indicators feed nodes; geometry/observed branch support set gates. |
| C2-first-sweep | core _backbone_state, 460–462; tree sweep_tree_state_v1, 110–128; pyg _GatedUpwardMessagePassingV1, 22–57, _ExcludedCavityMessagePassingV1, 60–80 | Reused whole-tree PyG upward/outside states at all nodes. Downward cavity messages exclude the recipient's own upward contribution. Width 32. |
| C3-clade | core _backbone_state, 463; flags RankEightCladeFlagAdapterV1, 223–244 | Reused local clade-flag adapter on first upward states; distinct from candidate memory. It does not gather another tree's state. |
| C4-memory / C4a-three-partitions | core CutMemoryV1.forward, 421–434; _direction, 372–411; _history, 413–419; ell_bp, 365–366 | f=U(up+outside) at actual representatives. Initialize m=f; at cut changes remap first; each hold then uses m←exp(-span/ell)m+(1-exp(-span/ell))f. Unchanged partitions retain layout. Reverse traversal maps (C,B)→(B,A). |
| C4 readback / C5-sum | core _history, 414–419; forward, 428–434; _backbone_state, 464–467 | Both directions share U/O/ell within an arm. Average histories on leaves, mean within current groups, subtract current forcing, project with O, scatter-add at actual representatives. Add reused clade injection and unchanged node base. |
| C6-second-sweep | core _backbone_state, 466–468; tree sweep_tree_state_v1, 110–128, leaf_context_v1, 130–131 | A second full upward/downward PyG sweep on base+clade+memory_delta, then sampled-leaf context. Full topology, dates and all sampled leaves remain available. |
| C7-genomic-gru | core _backbone_state, 469–478; inputs __init__, 276–279 | 32-d context + three span/neighbor-gap geometry features enter reused bidirectional GRU (32 hidden units per direction). Concatenation gives [T,952,96]. Same sample column across all holds; no history reset/detach. |
| D1-tree-logits | core _decode_state, 480–494; decoder decode_block_v1, 217–224 | Classify 96-d state, select all 686 original queries, gather by grid_hold. First 32 state channels separately supply rate context. |
| D2a-SNP-bins / D2-emission | decoder disjoint_bin_features_v1, 41–85; prepare_marker_grid_reset_v1, 134–155; decode_block_v1, 221–223, 228–229 | Original disjoint bins ≤100 bp yield four centered categorical evidence means + log1p usable-SNP count. Count-masked marker-head residual adds to local tree logits. Original centers are 100 bp apart; observed tails outside scored coverage stay in the tree body. |
| D3-transition-rates | decoder decode_block_v1, 224–229; base_rate, 209–211; log_pi, 213–215 | Mean/absolute difference of neighboring 32-d contexts drive lambda=lambda0 exp(g), the common graph_to_transition arm. Four-class pi is a separate learned simplex. |
| D4-chain | decoder reset_log_transition_v1, 158–173; reset_chain_log_marginals_v1, 176–184 | Exact rank-one reset kernel on physical gaps and inherited exact chain scan with initial pi. Each query block spans the whole chromosome; query blocking does not reset positions. |
| D5-probabilities | core _decode_state, 494, forward, 496–500; producer run_evaluate_v1, 477–485 | Concatenate query blocks; producer applies softmax(-1). Output is [position, query, 4] on unchanged scoring support. Four is the class count, not memory order or arm count. |

The insertion order is exactly core lines 461–478: prepare base → first full tree sweep → clade injection plus candidate memory at representatives → second full sweep → leaf context plus full-parent genomic GRU → unchanged common H048 decoder. The cut is a grouping relation alongside the graph computation.

ell=100+99900 sigmoid(theta), initialized to 5,000 bp, has mathematical range (100,100000) bp. U has 32×8 weights, O has 8×32 weights (zero initialized), and ell adds one scalar. The figure makes no performance, acceptance, information-dominance or operator-priority claim. Root will render/inspect before manuscript inclusion.
