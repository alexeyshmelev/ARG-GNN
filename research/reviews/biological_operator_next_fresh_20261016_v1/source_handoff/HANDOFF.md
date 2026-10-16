# Existing same-marker proposal: source-only handoff

**Status: isolated unexecuted source, pending independent review and root registration. No new hypothesis ID, run, numerical qualification, speed or novelty claim.** This concretizes the existing `next_useful_biological_mechanism_20261016_v1/REPORT.md` Hypothesis2 refinement. Read `SPEC_FIX.md` first. The decision registry currently has no registered ID for it. The registry explicitly requires one optimizer seed per discovery candidate and defers extra seeds until a convincing predictive pilot. That supersedes the optional immediate multi-seed recommendation in the preceding assessment report.

## Files and scope

- `same_marker_allele_support_v1.py`: strict CPU observation-layout preparation, complete-diploid alignment-null construction, fixed source/age/allele seeding and query lookup, bounded blockwise Torch/PyG path evidence, and the separate optional learned gate/rank adapter.
- `integration_hooks_v1.py`: isolated subclass of the exact H045 own-state class. It adds query residuals before the second complete PyG sweep and inherits the full-parent BiGRU/native-hold classifier. It contains no loader, optimizer, training loop or evaluator.
- `SPEC_FIX.md`: source blockers and the complementary-split derivation, resolved before coding.

No live `src/` file is edited. These files are not imported by any live path. Only Python-standard-library AST parsing was performed. Imports, model construction, tensor operations, tests, scientific payload access and forwards/backwards have NOT been run. Source inspection is not a numerical correctness certificate.

## API and shapes

Use the exact verified `MutationRoutedWholeTreeInputsV1` bank. A future root-owned caller can supply its CPU observation fields to:

    layouts = prepare_parent_layouts_v1(
        bank.parent, bank.sample_index, bank.site_hold,
        bank.reference_class, bank.sample_age, bank.query_column,
        bank.genotype, route_permutation=identity_or_null,
        allele_count=original_observed_sidecar_cardinality)

The optional cardinality vector is original observation metadata, not a target. `native_node[V]` maps each complete single-hold tree back to the unmodified global bank. Every original node and sampled leaf is preserved. The preparation returns only O(N+T*H) topology/lookup information and does not write a sidecar. It does not certify reference purity, file identity or roster provenance, which root must authenticate through the existing observed-source bindings.

For one hold:

| Object | Shape / meaning |
|---|---|
| Parent/local sampled indexes | `[V]`, `[H]`, all actual original nodes/leaves |
| Original categorical SNP block | `[B,H]`, `B<=16`, allele0..3, -1 missing |
| Fixed reference seed | `[V,B*8*4]` in log domain, labelled references only |
| Upward/outside messages | same fixed32 channels per marker, distinct directions |
| Existing gate input | `[V,9]` branch geometry/support plus two `[V,32]` first-upward states |
| Per-marker query evidence | `[B,Q,24]`:8 matching-frequency contrasts,8 relative log masses,8 availability channels |
| Checkpoint block outputs | three `[Q,24]` arrays:sum,max,valid count |
| Hold summary / new residual | `[Q,48]` mean/max, then `[Q,32]` residual |

The standalone `SameMarkerPathEvidenceV1(layout, hold_genotype, log_gamma)` has **no trainable parameters**. It can consume fixed externally supplied attenuation or the optional learned gate. Its fixed bookkeeping must not be advertised as a learned operator. The gate and rank adapter use familiar operations and remain unproven. Analytic additional capacity is74 gate parameters +384 rank-input weights +256 output weights =714, yielding32,526 allocated weights over H04531,812. No instantiated count was measured.

`SameMarkerPayloadV1` holds the original episode payload plus these layouts. `SameMarkerSupportGNNV1` keeps the inherited forward signature `forward_train_holds(bank, episodes=payload, control="aligned")`. It reads exactly the existing nine edge channels, including the old unpolarized support as gate covariates only. It never constructs the new evidence by multiplying source frequencies by those pooled support scores. It gathers the new residual only at actual query sampled nodes, scatters once into the global base, then runs the original second sweep and readout.

A future caller supplies the authenticated H045 own-state pass30 model-state mapping to `initialize_supplied_h045_state_v1`. This helper reads no checkpoint file and does NOT establish its lineage. Root should reuse the existing H050 `validate_supplied_warmstart_v1` source/receipt checks before extracting those supplied weights. The new output starts exactly zero. Gate/input gradients may therefore be zero on the first scheduled update, while output-projection gradients can become nonzero. This is expected zero-output initialization, not evidence that the gate has learned.

## Alleles, orientation and exceptional cases

The original branch concordance is a symmetric split statistic. Summing source frequencies over equally scored complementary biallelic supports collapses to a constant. The new route does not do that. It keeps source×age×allele channels separate through the tree and selects the observed focal query allele before any site reduction. For opposite-allele donors, unequal path weights produce `w0/(w0+w1)-1/2` for a query carrying allele0. There is no summation with the allele1 lookup. When both weights are equal the extra contrast correctly vanishes.

Cavity directions identify evidence arriving from the descendant side versus the outside side of each edge. Their combination follows each reference-to-query unique path once. Simultaneous categorical relabelling at a site leaves the result unchanged. No reference/alternate or derived/ancestral polarity is invented.

Missing reference calls create no seed, and frequencies normalize only over called references in that source/age group. Missing queries contribute neither lookup nor mean/max evidence. Absent groups stay masked with zero features, including absent older sources. All four possible nucleotide-allele slots are retained, so multiallelic sites need no biallelic split reduction. There is no winning-allele/branch argmin, hence no tie-breaking by node ID. Tied path evidence is summed. Every unary node/edge is traversed. Extra unary segmentation can change a per-edge learned attenuation, so no unary-subdivision invariance is claimed. Sampled interiors remain unsupported, matching the existing bank contract.

Safe empty log sums and prefix/suffix cavities avoid empty-logsum gradients and subtraction of nearly equal positive masses. The explicit edge-log bounds[-16,-0.0001] are a model/numerical restriction, not an evolutionary posterior. Nonempty messages remain finite in log space even when ordinary FP32 path products would underflow. Truly empty groups are identified from call counts, not from a small numerical value. These properties need numerical review before execution and have not been demonstrated by a run.

## Memory and cost

With S original SNPs, V_t nodes in their containing hold, G=8 and A=4, forward message arithmetic is O(sum_s V_t(s)*G*A), plus gate evaluation and the unchanged backbone. Two directed sweeps visit every edge. Prefix/suffix exclusion is linear in the number of children, including multifurcations. Python node/level loops may be expensive.

At most16 sites enter one hold message tensor. One FP32 message buffer costs `4*V*B*32` bytes, or2.048MB atV1000/B16. Several seed/state/cavity buffers coexist. This is NOT peak memory. Whole-parent node×site×source tensors are never constructed. CPU preparation does retain original genotype storage and O(N) Python/tensor topology metadata.

Every differentiable block is nonreentrantly checkpointed, returning only reduced sums/max/counts. This prevents retention of all block message arrays for backward, while recomputing them and preserving gradients. Stored reduction outputs cost O(sum_t ceil(S_t/B)*Q*24) scalars, in addition to original genotypes, backbone states, gate inputs, `[T,Q,32]` residuals and optimizer state. Checkpointing does not remove full-parent BPTT or certify total device memory. No SNP thinning, partial trees, detach, tile resets or truncated gradients are allowed. The first real scheduled whole-parent update, counted in the actual960-step budget, must establish feasibility under root resource guards. There is no proposed disposable pilot update.

## One full-budget utility experiment

Retain the existing pending hypothesis, rather than registering a duplicate name. Freeze before outputs:

- Aligned support and alignment-null support use identical copies of the full H045 own-state pass30 body plus714 added parameters and identical initial weights.
- The null changes only the added route: complete reference diploid blocks are permuted within observed source×younger/older-family strata, constant across all SNPs. Use one frozen observation-only seed per parent, without reseeding. Original backbone observations/gates stay in place. Record moved/called mass and immovable strata. Exact date attachment within an age family is disrupted, so the interpretation includes branch/date alignment. Small immovable strata limit the test.
- The third arm is exact H045 continuation. **Reuse the completed seed101 H050 continuation if the equivalence conditions below hold. Do not rerun its optimizer schedule.**

Both new arms use seed101, FP32,32 original TRAIN parents and eight reused development VAL parents, full952/all686, fresh independent AdamW(lr.001, weight_decay.0001, betas.9/.999, eps1e-8), gradient norm clip1,30 complete passes/960 parent updates, checkpoints5/10/15/20/25/30, and original exact-span loss. Parent order is `random.Random(101+pass).shuffle(list(TRAIN))`, with the EXACT original ordered TRAIN list. Both new arms therefore add1,920 updates. Six checkpoints across two new arms produce96 new VAL products that must all be sealed before their scorer opens targets. Selection remains equal-parent Present182 accuracy, lower NLL, earlier pass. No boundary reselection or early stopping.

H050 continuation reuse requires authenticated SAME H045 checkpoint/completion refs, actual TRAIN observation/content/roster bindings, original source versions for H045/PyG/SNP/episode/GRU/readout/loss,32-parent update order, FP32 math/determinism/runtime settings, fresh optimizer/clip/budget, all six checkpoints, selection code and identical scoring support. The inspected H050 `fixed_protocol_v1` and `family_from_base_state_v1` show intended equality and an unmodified H045 continuation class. That is source evidence, not proof that every actual artifact binding matches today. Root must certify exact actual equivalence. If it fails, label the baseline unmatched and block a matched utility claim. Do not silently replay it, change the protocol, or infer equivalence merely from equal scores.

Reuse existing immutable continuation scores and receipts when scorer definitions match. If extra predeclared tract/transition diagnostics require predictions no longer stored from its RAM evaluation, root may produce sealed predictions from the saved fixed checkpoint without optimization under separate execution authority. That is not new training. This handoff does not execute it.

Proposed frozen gate: at least+1.00pp Present182 boundaryF1 at1kb over BOTH null and continuation, no more than0.05pp span-accuracy loss and no mean NLL regression against either. Report all seven cohorts, parent-paired effects/sampleSD, precision/recall/F1 and counts, typed-transition diagnostics, true-short-tract recovery and false short tracts/false-switched bp perMb with frozen strata/edge censoring. Reused VAL is discovery evidence. A convincing result can motivate later additional seeds and untouched outer confirmation. No generic operator novelty follows, and no unchanged failed condition should be replayed.

## Required future correctness review, not executed tests

Review or later qualify: the two-reference biallelic weighted lookup above, allele-category permutation, missing query/donor and absent source cases, multiallelic categories, unary chains and multifurcations, duplicate/equal weights without arbitrary branch selection, long paths that underflow in probability space, finite gradients for all-empty messages, block-size forward/gradient agreement up to valid max-tie subgradients, complete-tree relabelling, aligned/null identical marginal allele counts/LD, no ancestry target path, initial zero-residual equality, and full first scheduled optimizer-step feasibility. Static AST parsing establishes only syntax.

Read sources: original pending proposal and decision registry, `mutation_routed_whole_tree_gnn_v1.py` input/19-channel pool, `mutation_routed_tree_observations_v1.py` donor-group order and symmetric concordance, `mutation_routed_pyg_gnn_v1.py`, `anonymous_clade_flag_control_v1.py`, `ordered_clade_episode_gnn_v1.py`, `train_ordered_clade_episode_v1.py:287–328` and parent ordering, and H050 `direct_readout_cut_memory_v1.py:150–228`/`integration_hooks_v1.py:fixed_protocol_v1`.
