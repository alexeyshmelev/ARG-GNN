# Figure 1 source conformance closure

Date: 2026-10-26. Scope: the final Figure 1 input, encoder interface and sequence readout. This is a source audit, not an independent image-only comprehension review. Figure 2 internals receive their own closure review.

## Verdict

The repaired Figure 1 agrees with the frozen four-class European TreeLoom implementation and the saved illustrative tree sequence. All four Figure 1 findings from the first frozen review are closed. No additional source discrepancy was found in this scope.

The audit read the final generator and inspected the final PNG. All eight source and toy SHA256 pins in `source_hashes.json` still match. The final layout record reports a minimum font size of 7 pt and zero text-to-text overlaps. Visual inspection found no text obscured by a connector in this Figure 1. The layout checker alone does not certify arrow clearance or reader comprehension.

## Closure of first-review findings

| Finding | Final evidence |
|---|---|
| Rounded site labels | Labels now come from the saved sites, 1.5k, 5k and 8.5k. The worked allele is the query at position 5000. |
| Implicit branch repetition | The example is labelled one of eight branches. Stacked feature sheets, two gate maps and shared ×8 explicitly connect the 9-entry branch example to both 8×32 gate matrices. |
| Implicit tree repetition | The leaf readout has a T1, T2, T3 repeat scope with the same projection. Its output connects to each of E1, E2 and E3. The query row from each tree has its own connection into the corresponding row of E_q. |
| Ambiguous sequence port | Both lower ports explicitly carry d:E_q. Their object is the three-tree sequence for one fixed leaf, not a five-leaf matrix for one tree. |

## Further verified details

- Node rows preserve all nine T2 nodes, including four ancestors. Normalized age, root bit and leaf bit feed Linear 3→32 and tanh. The learned leaf vectors occupy the five leaf rows and four zero ancestor rows before addition to the node matrix.
- Reference metadata contains four ancestry entries, an unknown bit and normalized sampling age. The query label is unknown. All five actual sampled ages are zero. The metadata feeds both allele lookup and leaf concatenation.
- The saved T2 interval is [3000,7000), with one called site. The pooling input explicitly says called sites and S=1. Mean and maximum strips agree for this one-site example. The count and span values are approximately 0.060206 and 0.571679. The site MLP and 67→32 and 38→32 projections match source, with their side connections entering the first Linear and leaving the final activation.
- The b→c example has child age 212.5, parent age 400, duration 187.5 and descendant fraction 2/5. Its nine observed entries, separate directional gates and sigmoid match the implementation contract. Eight real edges is correct. The code's dummy root gate carries no branch message and is intentionally absent from the eight-edge picture.
- Selecting five leaf rows from both final nine-node state matrices gives 5×64, followed by Linear 64→32 and tanh. The visual concatenate result has five rows. E_q has three rows, each copied from the corresponding selected query row, with consistent illustrative cell values.
- The depicted genomic endpoints are 0, 3000, 7000 and 10000. Their centers are 1500, 5000 and 8500. The length and two center-distance entries are attached to measured positions. The raw rows are [3000,0,3500], [4000,3500,3500] and [3000,3500,0]. Log normalization agrees with the frozen source.
- The input to the single bidirectional GRU is 3×35, consisting of local32 and geometry3. Both directions receive every input row. Each direction produces width32. The diagram explicitly collects and concatenates them into 3×64, then concatenates the original local32 to form 3×96. Linear 96→32, GELU, Linear 32→4 and softmax give 3×4 local ancestry probabilities.
- Learned Linear layers and the GRU are gold. Standalone tanh, GELU, sigmoid, softmax and fixed operations are gray. Embeddings and logits are teal, observed tensors gray, and final probabilities purple. The legend and ancestry-class key are visible. The gold ENC composite contains learned operations and is explicitly linked to its Figure 2 expansion.

## Evidence scope

The checked source paths include the frozen European class, its inherited `prepare_tree_base_v1` and `leaf_context_v1`, the site encoder and constructor in the whole-tree base, and the shared source contract. The saved tree sequence was loaded only to inspect its intervals, sites, sample ages, node parentage and genotype matrix. No model was imported or executed. No simulation, training, queue, benchmark or recorded result was changed.

Hidden tensor colors remain schematic and are not recorded model activations. The four output columns are the European worked example, not a claim that every benchmark has four classes. Missing genotypes are excluded from pooling in source. The fully called toy does not demonstrate that additional case.

## Audited artifact hashes

| Artifact | SHA256 |
|---|---|
| build_architecture.py | 9e2c0ee0c66b1a8f6202519cbe96a05a4dc5d881ca38b1f9e495284fbcc455f8 |
| TreeLoom_architecture.png | 061e47f3eff3fec64a69bc4d8f519e03a119ee1ee37232027cf50b0c2d8e2951 |
| TreeLoom_architecture.pdf | 3fdfe389676117c2a3bf6ca9077526f3fc0c552b077668cf7d1f38084fccae2a |
| TreeLoom_architecture.svg | 8fcc469d78b94c067de42d43ad152b1d01f324ead5f740684cd8097cd798f75d |
| architecture_layout_qa.json | 60e08d3974391da0e679924a81e207e0d36e24f58a3d7501df1b28a9ba88bbd5 |

The immutable `review_round1` artifacts were not altered. This report does not replace the required fresh reader review.

## Post-round2 layer-count clarification

The second image-only reader recovered the sequence flow but could not determine the GRU layer count. A label-only repair now states `forward (1 layer)` and `reverse (1 layer)` directly above the respective recurrent chains. Their width32 labels were moved upward to preserve clearance. The three cells in each chain remain three interval steps of one recurrent layer, not three stacked layers. The source specifies one bidirectional GRU layer.

The working Figure 1 was re-rendered and visually inspected after this repair. Minimum font size remains 7 pt and the updated layout QA still records zero text-to-text overlaps. No connector obscures either new layer-count label. No tensor, operation, numerical value, model source or experimental artifact changed. The immutable review_round1 and review_round2 snapshots were not modified. The artifacts below are frozen for the next reader round.

| Artifact after label repair | SHA256 |
|---|---|
| build_architecture.py | 79634b1be7068db61fbc751454a45ca46322eb5826d12b67e2223de547c28783 |
| TreeLoom_architecture.png | 8a7edf2e1ef51ea563d42fc1daf79d64caca506a5a35e5dded078109d05a9c0a |
| TreeLoom_architecture.pdf | 52b7b8b5080dc597167c8f9b7ee63e010173a7191004369571313e7806aa426e |
| TreeLoom_architecture.svg | 1e84583ebc39466e2d40d2848d5322783e3f98e6401c0a0b7e26d1bb8bd7fc9c |
| architecture_layout_qa.json | 60e08d3974391da0e679924a81e207e0d36e24f58a3d7501df1b28a9ba88bbd5 |
