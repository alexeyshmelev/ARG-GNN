# Conditional next hypothesis: ordered marker evidence through whole trees

**Retain one conditional utility candidate, without registering or implementing it yet.** If H050 does not meet its registered criteria, test whether learned marker-order evidence becomes more useful when routed through complete local trees than when applied only to the same sampled haplotype. First run the separate TRAIN-only representability audit. A useful result is possible, but generic GNN, convolution, interleaving or cross-graph novelty is not established.

## The actual gap and the untested condition

H045 foreign_episode reaches 96.8270 ± 0.5870% Present182 span accuracy. Its own_state control reaches 96.8175 ± 0.6142%, an improvement of only 0.009568 percentage points. Boundary recall remains 22.59%, with 905 matched changes, 3,103 misses and 5,063 extra calls across eight development parents. These are existing saved results, not newly measured values.

H045 and H050 encode observed SNP tuples and mean/max pool them within each native-tree hold. The final direct readout is constant for each query within that hold. The explicit site-pooling route loses within-hold SNP order when its input tuples and other bank fields are held fixed. This is not a theorem that the complete network has no linkage information. Tree features, mutation support and its BiGRU can encode correlated observations indirectly.

H044 tested fixed SNP boxes, with negligible accuracy gain and worse boundary F1. H046 tested clade flank context, H048 a reset-chain decoder, and H049/H050 higher-order cut memory. None tests the exact condition below: a learned marker sequence feeding sampled-leaf states back through the complete genealogy before the next graph sweep. H033's frozen-emitter tree/TV alternation did test generic interleaving and was negative. The proposed condition must not be sold as rediscovering interleaving.

C4 is the simulated source population at 350 generations BP. A C4 transition is a change in that target state, not every recombination event. An inferred topology change neither identifies a C4 switch nor a shared recombination event. Matching nodes across loci uses only the same observed sampled haplotype, with no persistent internal-ancestor identity, attachment likelihood or latent shared-switch assertion.

## A precise small model

Keep the complete H045 own_state architecture, all 952 observed haplotypes, all 686 query outputs, source labels/dates, mutation support and full-parent genomic BiGRU. Use its same completed pass30 checkpoint as a fixed common initializer, regardless of H050's numerical outcome. All backbone weights remain trainable.

Let t(j) map original scored 100 bp bin j to its native hold. Use the existing coordinate sidecar to join actual observed SNPs, without editing the frozen bank. Preserve all SNPs in the original backbone. In each supported scoring bin, mean-pool the existing 32-dimensional learned site embeddings for each haplotype and retain log-count, empty-bin and clipped-width channels. No new donor posterior or independently claimed biological likelihood is introduced.

1. Run the original first complete tree sweep and gather 32-dimensional sampled-leaf context u[t,h].
2. Form a 67-channel bin input from 32 site channels, 32 gathered tree channels and the three count/geometry channels. Project to rank8.
3. Apply four residual depthwise-separable Conv1d blocks along genomic bins, with kernel3, dilations1/8/64/512, rank8 and LayerNorm. Weights are shared across sampled haplotypes. This fixed receptive field is117.1kb on interior100bp bins, not a tuned demographic tract rate. The existing BiGRU still provides full-parent history.
4. Span-average the resulting v[j,h] back to each native hold, obtaining p[t,h]. Map p through a zero-initialized bias-free8-to32 projection J and inject it only at that hold's actual sampled nodes before the second complete PyG upward/downward sweep. Every internal and unary vertex remains.
5. Retain the original genomic BiGRU and classifier. At every original scoring bin add a zero-initialized bias-free8-to4 readout of v[j,h]. Thus the predictor can emit different ancestry probabilities within an inferred-tree hold.

With bias-free67-to8 input, four blocks containing24 depthwise weights,64 pointwise weights and16 LayerNorm scalars, plus J and the8-to4 readout, this adds exactly1,240 allocated parameters by construction, for33,052 total over H045's31,812. This is an analytic design count, not an implemented-model verification or memory result.

## Controls and representative training

Freeze four arms before any outputs exist:

- **Ordered marker feedback:** the model above.
- **Own-leaf ordered readout:** identical marker computation, parameters and two complete tree sweeps. Apply Jp to final sampled-leaf context after the second sweep, before the common genomic BiGRU. Marker information never enters other haplotypes through the second graph sweep. Both arms retain the same final marker readout.
- **Hold-mean feedback:** same graph feedback and temporal layers, but replace bin-local site embeddings/counts by their corresponding hold averages before the sequence operator. This removes marker localization from that route while keeping the original whole-tree SNP inputs. It is an information ablation, not an equal-information routing control.
- **Unmodified H045 continuation:** original31,812-parameter class without unused extra weights, to expose added-training gains.

The primary graph-routing contrast is ordered feedback versus own-leaf ordered readout. The hold-mean arm separately tests localization. They must not be collapsed into one causal claim.

Proposed fixed budget: original32TRAIN/eightVAL, seed101,30 full passes and960 whole-parent FP32 AdamW updates per arm. Same parent order, reset optimizer, exact physical-span objective, six scheduled checkpoints and original accuracy selection. All192 VAL products must be sealed before truth. No outer/TEST use, tile resets, detached history, subgraphs, reference top-k or SNP thinning.

A proposed primary practical gate is at least1.00 percentage point Present182 boundary-F1 gain over EACH control, with at most0.05 point accuracy loss versus EACH control. This would support boundary utility only. Accuracy, NLL, all seven cohort rows, paired parent effects and true/false short tracts remain mandatory. No validation-selected smoothing or boundary-based checkpoint reselection.

PyG encodes complete trees twice per pass, once per required sweep, rather than once per marker. Torch Conv1d processes haplotype blocks with exact585-bin halos for this four-block receptive field. Checkpointing may recompute blocks but must retain every gradient. A single FP32[M,H,8] tensor uses609,280,000bytes atM20,000/H952. That is not peak training memory. A real scheduled training update must establish feasibility under unchanged caps. No new GPU reservation or runtime claim follows from this estimate.

## Conditions for implementation

1. H050 first completes its existing registered fit/evaluation. Do not interrupt or alter it.
2. Root's separate bounded TRAIN-only audit confirms a material support limitation: proposed gate, equal-parent grid-minus-hold oracle gap at least0.10 accuracy points OR at least10% of true C4 changes are grid-representable within1kb but not hold-representable. Freeze the gate before audit outcomes. This measures representability, not available biological signal.
3. A focused source/FP32 check confirms same initial H045 function, correct raw-sample/coordinate joins, finite gradients, exact chunk/halo behavior, all original scoring support and independent model/optimizer states. Its first optimizer step must belong to the full scheduled run.
4. A fresh bounded primary-literature review must decide whether the exact route has any substantive distinction worth pursuing. If it is an established heterogeneous spatiotemporal GNN, it may still be a useful engineering baseline. It cannot discharge the desired new general architecture contribution by renaming it.
5. Root separately registers the condition and its resource caps. This report grants no execution authority.

Reject or defer this direction if the TRAIN audit leaves little relevant headroom, the ordered marker control performs equally well, graph feedback damages short-tract recovery/calibration, or representative full-parent training is infeasible. A negative H050 result does not itself justify this candidate.

## Primary overlap and claim limits

| Re-read primary source | Established overlap and narrower distinction |
|---|---|
| Li et al., Graph Matching Networks, ICML2019, Section3.2 Eqs4–6,10–11 | Cross-graph latent messages inside graph computation already exist. The proposed route uses observed sampled-leaf joins and marker coordinates, without learned all-pairs matching. |
| Escrig Molina et al., Graph Set Transformer, arXiv2606.05116v1, Section3.1 Steps2–5 | Pool, exchange collection context and route it back into nodes already exist. Ordered marker states rather than graph tokens are an application-specific change. |
| Whitehouse et al., Tree Sequences as a General-Purpose Tool for Population Genetic Inference, MBE2024, Figure1 and GCN architecture text | Whole marginal-tree GNNs plus genomic sequence learning are established. This candidate retains node correspondence and marker-resolved feedback, rather than a graph-plus-sequence priority claim. |
| Browning et al., FLARE, AJHG2023, Subjects and methods: Hidden markov model | Marker-level ancestry and donor continuity already exist in stronger probabilistic specialist models. The proposed convolution is neither a donor-copying posterior nor a FLARE reproduction. |

These cached full primary passages were read directly for this report. No exhaustive search or exact-combination priority finding is claimed. The unrun donor-identity/remix condition already has an owner and remains distinct. Rejected shared-switch factors and unqualified attachment mixtures are not reopened.

Only source, existing concise result/decision metadata and cached primary text were read. No arrays, genotype/tree/checkpoint/probability/truth payloads, fit, numerical audit, live probes, guard/admission/runtime mutation or Git operation occurred.
