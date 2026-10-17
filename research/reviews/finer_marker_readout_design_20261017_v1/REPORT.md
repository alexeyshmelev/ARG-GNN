# Marker readout with matched tree feedback

**Recommendation: develop a marker-resolution head as a biological hypothesis; no-go for a standalone new-operator claim.** Complete the current information path: retain each SNP's recurrent state at readout, while testing identical marker evidence before versus after the second complete-tree pass. This can represent switches inside an inferred hold. It is unimplemented and untested; current H058, TreeLoom and learning v2 remain unchanged.

## One architecture, one matched contrast

Reuse the full first tree sweep, current positive same-marker source evidence and both original complete-tree passes. For each native hold and query, feed all original SNPs in order into forward/reverse width8 GRUs. Each input contains the3G evidence channels, the original query-called bit and log1p physical gap normalized by log1p(parent length). Each direction uses its own preceding gap; the hold edge supplies the first gap. Reset only at native holds. The original genomic BiGRU still crosses every hold for every sampled identity.

Preserve the common H058 pooled residual in both arms. The two marker endpoint states produce the same rank8→32 ordered residual as in v2. **Feedback** injects it at query nodes before sweep2. **Readout control** injects it at the same sampled contexts immediately after sweep2, before the genomic BiGRU. Clone identical weights; use independent optimizers. Both arms contain and use every module.

At observed SNP s, emit base-logit plus a zero-initialized residual head:
`Linear(K+16+3G+2,16) → SiLU → Linear(16,C,no_bias)`.
Inputs are the original final classifier state K of that SNP's hold/query, its two local marker states, same evidence, called bit and the forward-direction gap. The ordered endpoint projection and marker head start zero in both arms. All parameters remain trainable FP32. Thus capacity, evidence, geometry, marker resolution, loss and budget match; the intervention is ordered-state placement relative to complete-tree messages. This tests graph-feedback utility, not whether either arm receives order.

Use each dataset's original class order and declared source×age groups: European C4/G8; American C3 and its original observed age grouping. Empty groups remain empty. The design does not assert transfer of one dataset's labels, weights or group roster to the other.

## Exact observation-only endpoint mapping

For native interval `[l_t,r_t)`, retain every original ordered SNP `x_1<…<x_m`. Define cell edges `l_t,(x_1+x_2)/2,…,(x_(m-1)+x_m)/2,r_t`. A coordinate maps to its cell's SNP; equality at a midpoint selects the right SNP. Determine its native hold with the existing half-open/right-sided rule first. Extend the first/last SNP only to that hold's edge. A zero-SNP hold uses its unchanged coarse full-tree/genomic prediction, including isolated sampled roots; add no virtual edge or marker.

Freeze these maps from original observation coordinates, native intervals and endpoint geometry before any targets:

- **European:** for every unchanged original grid center `u_j`, output `p_jq=p_a(u_j),q`. Preserve all original positions, clipped bin edges, retained support, query order and seven cohorts. Bins can straddle SNP-cell/hold boundaries; the original center assignment and bin geometry still apply. Existing boundary scoring uses the original grid probabilities, not new midpoint coordinates.
- **American:** for every original registry record i, return `p_a(position_i),sample_i` in its exact record order. Preserve all10,000 record IDs, positions, sample-node/sample-column joins and class order, including repeated mapped pairs. Do not regenerate, majority-filter or relocate focals.

These SNP cells are an observation-derived output tessellation, not inferred ancestry segments. Native holds constrain tree reuse; they do not force all readouts in a hold to agree.

## Exact external TRAIN targets

European `FullPanelExternalTargetV1.target_mass` already forms `M_jqc` by intersecting the original bin edges with authorized TRAIN truth intervals. After the fixed map, aggregate `Mbar_aqc=sum_(j:a(u_j)=a) M_jqc`, including fallback anchors. Train with `-sum Mbar*log p / sum Mbar`. This is exactly the original grid-span objective under the declared mapping. Intersecting truth directly with SNP cells would change that objective and is excluded.

American targets remain the existing sealed10,000-vector from the original half-open truth join. Aggregate label counts by fixed mapped anchor/query, then compute `-sum N_aqc log p_aqc /10000`, exactly the original focal-mean CE. No SNP truth labels or extra American supervision are manufactured. Targets enter only the external loss; no truth breakpoint chooses a graph, cell, coordinate, feature or checkpoint.

Preserve full European32×30=960 updates per arm and American20×30=600 additional updates per arm if these successor conditions are later registered, with each dataset's original warm-start, seeds, optimizer and selection. All corresponding paired checkpoint/parent endpoint products precede either arm's held-out truth.

## Scientifically valid support

Known pure references alone seed source evidence. Keep reference/query routing, actual sampled dates, four called-allele slots, multiallelic/recurrent calls and explicit missing masks. Missing positions remain recurrent positions; ancestry can be predicted from neighboring observations without pretending the missing call exists.

For an internal query, eligible references must lie in its connected component and outside its descendant subtree. Compute matching mass and ordinary called-frequency background on that same eligible set; zero eligible called references means unavailable evidence. Internal sampled references seed their actual nodes, including singleton corrections. For current leaf queries this agrees with the usual all-reference background. H058's unusual zero-outside/all-group-valid edge behavior must not silently certify a general internal-sample extension. A new adapter must implement this declared support and preserve complete forests; current leaf-only packed contracts are not evidence of such support.

## Real Torch/PyG interface and costs

Prepare only immutable observation maps/support, then expose `encode_complete_parent`, `iter_marker_logits` and an external block-loss caller. Retain full forest-level PyG sweeps and genomic recurrence. Store forward/reverse marker boundary states every16 SNPs with nonreentrant checkpoints. Recompute local states/evidence within a tile for the SNP head; checkpoint feature construction, recurrence and loss consumption together. Carry boundary states without detach. One optimizer update follows the complete-parent loss, with no truncated BPTT.

Boundary state storage scales as `O((S/16)Qd)`; an active tile has evidence/kernel/GRU/head work. Dense evidence still costs `O(sum_t S_t N_t G A)`; sparse path evaluation has its real cover/path costs. Recurrence adds `O(SQd²)`, and emitting all SNP predictions requires at least `Omega(SQC)` output work. Recomputations, original site encoding, both tree passes and genomic states remain. Stream probabilities/loss; do not retain full-parent SNP activations or rerun a nonlinear full-tree GNN per SNP. Endpoint-active head pairs may be precomputed from geometry alone; all SNPs still influence recurrent states.

This is an implementable interface design, not a fit or speed result. Complete backward lifetime, original American zero-SNP forest handling and a real marker-output transport remain implementation work.

## Utility and novelty decision

European fixed-grid boundary metrics can test within-hold recovery, alongside accuracy/NLL, false switches and short-tract recall. American10,000 sparse focals test point prediction and cannot alone validate dense ancestry boundaries. Any American boundary claim needs a separately authorized sufficiently dense endpoint.

Internal H044 already implements localized grid readout; the earlier ordered-marker design already proposed marker prediction. Public primary abstracts confirm RFMix's sequence-conditioned local ancestry, FLARE's Li–Stephens model/compression/checkpoints, and Ralph et al.'s tree-sequence sample-weight framework. GRUs, residual conditioning, checkpointing and the existing tree path kernel are established. The present bounded check establishes no priority for their combination. **Reject a new-operator/novelty claim now.** A useful result would be matched evidence that graph feedback improves the unchanged biological endpoints beyond the equally informed marker-readout control. No efficacy, speed, numerical fit or ancestry result is claimed.
