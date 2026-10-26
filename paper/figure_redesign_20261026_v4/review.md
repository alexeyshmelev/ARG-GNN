# Independent figure reconstruction

Each acceptance attempt uses a newly spawned agent with no conversation history. The reader receives only the two frozen PNGs, their access instructions and the same request to explain the conceptual flow. It is forbidden to inspect the manuscript, source, earlier reviews or task history. No desired answer or verdict is supplied. Exact code reconstruction is outside scope.

The orchestrator compares the independent explanation with the implemented method. A favorable aesthetic judgment alone is insufficient. Any missing or incorrect key computation triggers revision and a new reader. This is a check of these readers and these images, not a guarantee of immediate comprehension by every scientist.

## Round 1 — not accepted

Reader: `/root/v4_blind_reader_1`, fresh context.

The reader correctly reconstructed all-node message passing, group context copied to persistent leaves, two memory directions, directional then group averaging, subtraction, group-ancestor injection, the shared second round, leaf selection and the final genomic GRU. It could not fully explain input pooling, group selection, structural correction placement, edge information or the log transformation. Its conclusion was: “I cannot explain all key steps without guessing.”

Revision: separated the node and leaf feature routes, showed site pooling within interval and leaf, specified the age-cut rule, showed structural correction of current upward states at current clade rows, and made the log(1+x) transformation explicit. A source audit independently found and corrected a missing toy group-representative marker. The plotting code now computes that selection from the saved tree.

## Round 2 — not accepted

Reader: `/root/v4_blind_reader_2`, fresh context.

The reader independently explained all eight major stages and correctly distinguished all aggregation axes and the three mechanisms carrying information across trees. Two gaps remained. The displayed gain/loss/persistence examples could be interpreted as mutually exclusive, and observed variants had no visible arrow into edge gates.

Revision: replaced the examples with a presence-to-flag truth table, including simultaneous flags and an isolated clade. Added a learned edge-gate block with separately traceable tree and observed-variant inputs. Marked its wire crossings as underpasses. Named the structural maps nonlinear, consistent with their factored products.

## Round 3 — accepted conceptual explanation

Reader: `/root/v4_blind_reader_3`, fresh context. Images are frozen in `review_round3/`. The explanation was compared with the source audit. Its result is recorded below.

### Round 3 result — conceptual reconstruction accepted

The reader independently described the full input mapping, both complete message-passing rounds, all-node state retention, group context copied to leaves, the memory recurrence, directional and group averaging, subtraction and representative injection, overlapping structural flags on current states, leaf selection and projection, ordered stacking, all three geometry features, the BiGRU, local skip connection, classifier and interval-specific ancestry outputs.

It stated: “I see no demonstrably incorrect major connection or missing conceptual bridge between the figures.” It correctly identified the three distinct mechanisms carrying information across trees. Its remaining limits concerned exact message equations, gate formulas, feature encodings and edge-of-sequence conventions, which are implementation details rather than missing conceptual stages.

The orchestrator checked this account against `source_audit.md`. All key stages and aggregation axes agreed with the source. After acceptance, final-page inspection found two labels touching bypass lines. They were moved without changing any operation or connection. A new reader with no prior context was assigned the final images to preserve the fresh-reading requirement for the delivered pair.

## Final delivery pair — accepted

Reader: `/root/v4_blind_reader_final`, fresh context. Its complete [independent explanation](final_reader.md) is retained with mathematical notation transcribed to plain text. It saw only the two images in `review_final/`, whose hashes match the canonical PNGs.

The reader correctly described every conceptual stage and aggregation axis. It distinguished the shared state connection from arithmetic addition using the explicit destination labels, and correctly used only the current upward state in structural maps. It did not infer a boundary convention for clade flags from the unrelated distance-feature sentinel. Exact gate formulas, detailed message equations and boundary conventions remain in the manuscript and source rather than being reconstructed from the picture.

The acceptance criterion is the user's requested complete conceptual explanation, not exact implementation reconstruction. The final account meets that criterion. The remaining shared-line observation is retained rather than suppressed: its labels resolved the intended computation for this reader, but the record does not claim zero possible ambiguity for every reader.

### Orchestrator comparison with source

| Key stage | Independent account agrees with implementation |
|---|---|
| Observed inputs, masked query labels and separate node/leaf initialization | Yes |
| Called-site pooling within interval and leaf | Yes |
| Both all-node traversals, zero root outside state and shared second-round parameters | Yes |
| Age-cut grouping and group context projection | Yes |
| Stable leaf identity across changing groups | Yes |
| Exponential forward/reverse memory and interval-length retention | Yes |
| Direction mean, current-group leaf mean, context subtraction and group-representative injection | Yes |
| Exact descendant-set flags with overlap and current-state nonlinear maps | Yes |
| Corrections added to the original base before round 2 | Yes |
| Leaf selection, feature concatenation and 64-to-32 projection | Yes |
| Ordered tree rows, three geometry columns and missing-neighbor zeros | Yes |
| BiGRU, local embedding bypass, classifier and interval-specific ancestry probabilities | Yes |
| Distinction among the three mechanisms carrying information across trees | Yes |
