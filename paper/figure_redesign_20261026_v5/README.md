# TreeLoom tensor diagrams, revision 5

[Prediction flow](../figures/TreeLoom_architecture.png) · [Encoder expansion](../figures/TreeLoom_encoder.png) · [Paper](../current_full_panel_working.pdf) · [Overleaf bundle](../current_full_panel_overleaf_working.zip)

The two figures use explicit tensor rows, connected operands and repeated operations. The example follows the same leaves through three intervals and shows all ancestor rows through both graph rounds. The encoder has a marked parent and expansion. Memory copying, both recurrence directions, group reduction, subtraction and scatter are distinct. Every color has a semantic legend.

Revision 4's conceptual reading passed too weak a test. The owner still could not trace the computation. Revision 5 uses a fresh reader with no source, manuscript or conversational history, who must trace a named leaf and ancestor from the images. The root checks this reconstruction against the frozen implementation. The [review record](review.md) preserves issues and repairs. This is a communication check, not owner approval or evidence of scientific novelty.

The final fresh reader correctly reconstructed all tested key paths. [Full independent explanation](review_round3/reader_report.md) · [Acceptance and exact image hashes](reader_acceptance.json). The two figures occupy pages 3 and 4 of the eleven-page compiled draft.

[Source contract](source_audit.md) · [Source hashes](source_hashes.json) · [Integration receipt](integration.json)

Observed values come from the previously saved illustrative tree sequence. Hidden activations and probabilities are schematic. No new simulation, model training, profiling, checkpoint change or benchmark run occurred. All eight manuscript table source blocks are unchanged. Experiments remain stopped.

Rebuild the CPU-only drawings on anogena from the dedicated repository:

```bash
. scripts/project_env.sh
"$ARGGNN_PYTHON" paper/figure_redesign_20261026_v5/build_architecture.py
"$ARGGNN_PYTHON" paper/figure_redesign_20261026_v5/build_encoder.py
```

The integration script requires a reader acceptance receipt bound to the exact images. Canonical PDF, SVG and PNG figures are also stored in `paper/figures/`. The Overleaf bundle contains its main source, bibliography, AAAI style and both vector PDFs.
