# Source conformance of the first frozen figure review

Date: 2026-10-26. This audit compares `review_round1` images and generators with the frozen stable-leaf TreeLoom implementation. It is an implementation audit, not an independent image-only reading. No model computation, simulation, training, experiment or result was changed or executed.

## Verdict

The displayed learned operations and the numerical worked example agree with the frozen model, apart from the two rounded site-position labels identified below. The memory and structural equations agree with their actual source. Several repetition and continuation labels remain ambiguous. These are diagram issues, not evidence that the model or recorded comparisons were changed.

## Items to repair

1. **Incorrect observed site labels in Figure 1b.** The genotype rows are labelled `1k`, `5k`, `8k`. The preserved tree sequence contains positions 1500, 5000 and 8500 bp. Use `1.5k`, `5k`, `8.5k`, or the full values. The allele values and the worked site at 5000 bp are correct.
2. **Branch repetition is implicit in Figure 1c.** The worked input is one 9-vector for branch b to c. Applying each Linear 9 to 32 map once produces one 32-vector, not an 8 by 32 matrix. The implementation repeats the shared map for every real branch. Add an explicit repeat-over-branches scope. Eight branches is correct for the nine-node example. The implementation's unused root gate slot does not transmit a message.
3. **Tree repetition is implicit in Figure 1d.** The detailed observation example is T2, but the single drawn leaf-projection arrow enters E1, followed by independently drawn E2 and E3. Each tree must have its own observed base, two-round encoding and leaf projection. Indicate repetition over trees and connect its collection to all three E matrices. Do not imply that T2 directly produces E1 or that all three matrices are copies of one tree's features.
4. **Continuation port names should identify the selected leaf.** The lower ports named `d:E` carry E_q, with shape trees by 32, rather than E_t, with shape leaves by 32. Rename these ports consistently to `d:E_q`. The actual array concatenations in the generator use the correct three-row leaf sequence.

## Important scope notes

- Site pooling in code includes called sites only. The worked example has one called site, so its picture is numerically valid. For the general diagram, `called sites` should qualify the pooled rows. Otherwise a reader could reasonably pool missing-site MLP outputs, which would differ from the implementation.
- A missing genotype makes the 19 input entries zero and its embedding is excluded from pooling. Availability is also masked by genotype callability. These conditions are not exercised by the fully called toy.
- The 350-generation memory cut uses the highest real ancestor at or below the cut. A sample older than the cut is its own singleton group. The pictured leaves all have age zero, so the omitted older-sample case does not invalidate this example. Retain the policy in the method text.
- There is no claim of persistent inferred ancestral-node identity. Structural matches use descendant sets. Memory aligns persistent leaf identities. This distinction is correctly represented in Figure 2.
- Hidden colors are schematic, not measured activations. Arithmetic correctness should be read from the connected operators and dimensions, not inferred from color intensity.

## Verified Figure 1 operations

| Stage | Source check |
|---|---|
| Node observations | All nine toy node rows are present. Ages, root bit and leaf bit are correct. Linear 3 to 32 and tanh are correct. |
| Reference observations | Four one-hot ancestry entries, unknown bit and sampling age are correct. Query q has unknown ancestry and all sampled ages are zero. |
| Site lookup | For the query allele at site 5000, younger donor log frequencies are log(0.25), log(0.25), log(0.75), log(0.75). Older channels are unavailable. Panel log probability is log(0.6), entropy is about 0.673, and called is one. |
| Site encoder | Linear 19 to 32, GELU, Linear 32 to 32 and tanh match source. |
| Pooling | Mean32, max32, normalized count, empty bit and normalized span form 67 entries. Count about 0.060 and span about 0.572 are correct for one called site in the 4000 bp interval. |
| Leaf base | Concatenation of pooled32 and metadata6, Linear 38 to 32, tanh, then addition only at leaf rows are correct. |
| Branch observations | The displayed b to c row uses ages212.5 and400, duration187.5, descendant fraction2/5, exact and closest support1, discrepancy0, normalized split count log(3)/log(100001), and empty0. All values agree. |
| Gate functions | Separate upward and downward Linear 9 to 32 plus sigmoid are correct. |
| Final leaf context | Select all five leaf rows of second-round U and O, concatenate32+32, Linear64 to32 and tanh are correct. |
| Genomic geometry | Intervals0-3000,3000-7000,7000-10000, centers1500,5000,8500, center gaps3500, boundary zeros and normalization are correct. |
| Sequence model | Ordered leaf32 plus geometry3 produces35. One bidirectional GRU with32 units per direction produces64. Concatenation with local32 produces96. Linear96 to32, GELU, Linear32 to4 and softmax are correct. |

## Verified Figure 2 operations

| Stage | Source check |
|---|---|
| Repeated encoder | Two complete rounds use the same message parameters and the same gates. The second round starts from original base plus both corrections. |
| Upward messages | A bias-free32 to32 transform, edge gate, division by parent child count, saved per-child messages and row sum are correct. |
| Upward node update | Concatenate base32 and aggregate32, Linear64 to32, tanh, add base, then LayerNorm. Order is correct. |
| Downward message | Parent outside plus parent base plus parent aggregate minus recipient's saved normalized contribution. Bias-free transform, downward gate, learned update, tanh and LayerNorm follow. There is no extra residual. Root outside is zero. |
| Node traversal | The example's upward stages leaves,a/b,c,root and downward stages root,a/c,q/b/r3/r4,r1/r2 are valid. Both matrices retain every node row. |
| Cut table | T1 groups d,d,r2,a,a and T2/T3 groups q,b,b,a,a are correct. d is node5 at200 generations, a is node6 at175, b is node8 at212.5. |
| Memory forcing | Gather first-round U and O at representatives, add, project32 to8, then copy group rows to their leaves. |
| Persistent-leaf example | r1 follows F1[d], F2[b], F3[b]. The last two vectors are separately indexed by tree and are not assumed identical. |
| Memory decay | 100+99900 sigmoid(theta) bounds the learned length. exp(-L/length) gives approximately0.55,0.45,0.55 at the initial5000 bp length. |
| Directional recurrence | Each direction starts from the endpoint forcing. Later states combine prior directional state and current forcing using the current interval's decay. Reverse ordering is correct. |
| Memory return | Repeat all leaves, average the two directions, average leaves in each current group, subtract current group forcing, project8 to32, scatter at q,b,a. Other node rows are zero for this toy cut. |
| Change adapter | Three separate bias-free32 to8 maps form x,y,z. Gain times x elementwise y minus loss times x elementwise z is projected8 to32. |
| Persistence adapter | Three separate bias-free32 to8 maps form x',y',z'. The sum of x' elementwise y' and z' is projected8 to32 and multiplied by eligibility. |
| Structural example | C={r1,r2} is absent in T1 and present in T2/T3. At T2, gain1, loss0 and persistence eligibility1 are correct. Proper clades exclude roots and singleton leaves. |
| Structural return | The two local corrections add at b. Repeating proper clades and adding their scattered rows forms the all-node structural correction. No neighboring hidden vector is used by these maps. |

## Source evidence

- `research/reviews/frozen_benchmark_training_ready_20261017_v1/european_frozen_treeloom_v1.py`
- `src/arggnn/mutation_routed_whole_tree_gnn_v1.py`
- `src/arggnn/mutation_routed_pyg_gnn_v1.py`
- `src/arggnn/anonymous_clade_gnn_v1.py`
- `src/arggnn/anonymous_clade_flag_control_v1.py`
- `src/arggnn/ordered_clade_episode_gnn_v1.py`
- `research/reviews/order2_cut_memory_implementation_20261015_v1/order2_cut_memory_v1.py`
- `src/arggnn/mutation_routed_tree_observations_v1.py`
- `research/reviews/figure_v15_flow_revision_20261017_v1/v15r2/illustrative_recorded_ARG.trees`

The immutable first-review inputs remain under `review_round1`. Later image changes should receive a new snapshot and a new image-only reader, rather than silently replacing the first-review evidence.
