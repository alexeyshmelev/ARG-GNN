# Fresh figure and naming QA

Status: requires two narrow changes before final publication. This is editorial and implementation QA, not conference peer review.

Reviewed candidate: paper/_build_TreeLoom_fig_naming_20261016_v2_attempt3/current_full_panel_working.pdf, SHA256 1f89b5c44513e3162c4fd617f02312f17644cd030dbc85fce118659b02bb7fcb. Manuscript source inspected at SHA256 10e4102b642012117836b97fd8bb635709bfe1a2d9bb925c659022edf9f362a0. Figure page 3 and standalone TreeLoom_architecture_render.png inspected as pixels through SSH into NodeRepl memory, without controlling-Mac project files.

## Findings

1. **P2 — Make tensor axes and sampled-node projection explicit.** Figure 1 uses unlabeled illustrative cell grids (7×4 for B_t and 4×5 for E_t). A reader cannot identify row/feature axes or tell that full native-node states become sampled-haplotype embeddings. The arrow from Sweep 2 directly to E_t also omits the learned concatenation/projection in anonymous_clade_gnn_v1.py:130–131 and mutation_routed_whole_tree_gnn_v1.py:276. Add symbolic labels B_t in R^{N_t×d}, E_t in R^{S×d}, with rows = native nodes/sampled haplotypes and columns = features, and identify concatenate sampled upward/outside → Linear → tanh. Numeric hyperparameters need not be added. This is a clarity gap, not evidence of an incorrect implemented module.

2. **P2 — Complete the two supplemental alias bindings.** TreeLoom-MLP and TreeLoom-Tensor occur in result prose and the supplemental structural-comparison table, but neither is in model_names_and_table_bindings.json.models or completed_metric_snapshot.json.models. Add their exact existing recorded H040 condition/score bindings and a brief suffix definition. The eight currently bound aliases match the described conditions and preserve their distinct training lineages. This finding does not assert an incorrect score.

## Checks passed

- Figure fits cleanly with readable lettering, no clipping, and no unintended connector crossings at the inspected publication page. The separate known trade-table float placement issue was outside this audit and is not duplicated here.
- Toy JSON and drawing code describe an actual illustrative msprime Hudson ARG with 4 samples, 13 nodes, 14 edges, two paired recombination events (nodes 5/6 and 8/9), and 3 intervals [0,1211), [1211,3602), [3602,10000). Every recorded parent is older than its child. The three plotted marginal topologies, sample colors and interval boundaries agree with the JSON. Unary collapse is disclosed. Hypothetical mutations are placed on real branches. No experimental data or fitted predictions are represented by the schematic output.
- Actual source confirms two full-tree sweeps with shared weights and gates, the original base bypass, parallel clade/local/sample-memory residuals, one bidirectional GRU layer, and local-plus-history classification. Bottom-up averaging, top-down recipient exclusion and root outside=0 agree with the sweep inset. Observed inputs and masked query reference labels are stated, with no query ancestry truth drawn as an input.
- All seven generated cohort-table binding rows exactly equal the completed metric snapshot. The inspected primary pipeline, memory, cohort and boundary table values agree with the snapshot after displayed rounding. Column/row winners are bolded correctly, including AncestralPaths boundary recall and NoMem precision/F1. The manuscript discloses differing developmental exposure and the common final memory-comparison budget.

Scope limits: no experiments, model execution, raw data, model weights, ancestry truth, resource/guard/accounting calls, package installs or Git operations. No manuscript, figure or PDF authored/edited by this reviewer. Root owns final correction, rebuilt-pixel verification and publication.
