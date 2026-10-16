# Scientific status of the current working paper

The ten-page AAAI manuscript presents TreeLoom for local ancestry inference from complete inferred local trees. Its main narrative explains biological motivation, the representation, evaluation and informative results. Figure 1 is on page 3. The supplement starts on page 8.

European development uses 32 training simulations and eight reused validation simulations, each 2 Mb, with 476 diploids or 952 haplotypes. Modern-query span accuracy is 97.0778 ± 0.5674% for TreeLoom, 97.0000 ± 0.4978% for TreeLoom-NoMem, 94.4461 ± 1.1491% for current author ARGMix and 93.7830 ± 0.8385% for the released dense AncestralPaths classifier with derived four-source output. SD is across simulations, with one optimizer seed. Boundary F1 favors NoMem at 19.0488%, versus 18.3807% for TreeLoom.

These results do not establish publication-benchmark top-1, independent-test generalization, or useful new-operator attribution. ARGMix release15+0 differs from paper25+3 inputs, and the AncestralPaths dense archive differs from the paper's CNN. Scientific TODOs remain visible for missing simulation/evaluator details, faithful native American comparisons, independent testing, broader utility and biological corroboration.

The editorial rewrite changes no experimental values. The final model and completed comparators replace the developmental catalogue in the main results. One meaningful matched memory comparison retains both accuracy and boundary F1. Additional controls are in the scientific supplement. Figure 1 was checked by fresh image-only readers and revised in response. The v5 correction matches miniature GNN topology to T1, groups all tree inputs, shows the initial feature encoder and persistent sample rows, and makes physical geometry and argmax explicit. Its input values and output probabilities are illustrative.

Both the canonical source and extracted Overleaf bundle compile. All ten rendered pages were inspected, and all ten bundle pages are pixel-identical to the canonical render. [Final integration](../research/reviews/paper_figure_clarity_ROOT_20261016_v5/final_integration.root_v1.json).

[Current detailed scores](../research/current_scoreboard.md) · [Comparator-fidelity matrix](../research/reviews/competitor_fidelity_user_audit_20261016_v1/comparison_matrix.md) · [Provenance](current_full_panel_provenance.json).
