# A finite cross-tile state-carry experiment

H023 motivates testing whether resetting context at 250 kb boundaries contributes to the candidate’s development error. Age-D2’s deficit versus current-release ARGMix is 1.61–1.82 percentage points within 25 kb and 0.57 points at 50–125 kb. Both exposed VAL parents show a larger nearest-band deficit. This supports a next architectural hypothesis, not a causal explanation. The following is a source-based proposal. No implementation, qualification, fit or new target access has occurred.

## Primary design: matched reset versus exact carry

Run the following finite 2×2 comparison on the original H020 Relate inputs, with seeds 101/202/303: 12 separate fits. Age versus plain retains the same age-derived features; it is an operator comparison, not an age-free feature ablation. Existing H020 results and native ARGMix seed 50 remain historical descriptive references.

| Arm | Kernel geometry | Artificial tile-boundary behavior |
| --- | --- | --- |
| Age reset | Existing age branches | Zero both directional states at each tile boundary |
| Age carry | Same age branches | Pass both directional states through every boundary |
| Plain reset | Existing plain branches | Same resets |
| Plain carry | Same plain branches | Same continuous carry |

All arms use identical original trees, sample joins, Q184/R96/Q120 roles, F47 features, source GNN, 2 kernel layers, 3 scales, hidden width 32, latent width 8, learned parameters, initial weights, raw probability endpoint and selection rule. Retain the union of the existing 32 anchors per clipped tile, hence 256 anchors across each parent, with their exact cached float64 locations and physical cell masses. Calling the existing uniform layout constructor with count 256 on the clipped parent would move anchors; do not do that. Keep every original marginal tree, including full topology at cuts through a tree. A repeated original tree ID across clipped tiles is a shared original tree, not a new independent tree or an excuse to prune nodes. Carry joins persistent sampled IDs in the original Q order. Internal ancestor IDs remain provenance.

Carry is the continuous whole-parent finite-anchor kernel evaluated in streamed blocks. In the current source, each directional accumulator is [scale, Q, latent, numerator-or-ones-mass]. Carry both numerator and matching mass; carrying only normalized hidden outputs would change normalization. Preserve chronological noncommuting graph holds, actual physical durations divided by the original 2 Mb, forward-inclusive/reverse-exclusive self terms, original Cqr reference sources, and positive-mass checks without floors. A true marginal-tree boundary switches the hold operator; an artificial tile boundary only divides computation. Keep references independently encoded and clamped.

The kernel is bidirectional. A left-to-right hidden-state handoff alone is insufficient. For each nonlinear layer, construct all fixed anchor sources, complete forward and reverse scans with the incoming prefix/suffix states, combine messages at anchors, then update anchors before constructing the next layer’s sources. Domain endpoints have the same zero exterior states. Artificial cuts do not. No averaging or smoothing of output probabilities is introduced.

## Training and memory

Use one optimizer step per complete TRAIN parent, with the exact sum of original query120 class-span losses divided by that parent’s total query-bp. Shuffle the two TRAIN parents identically across arms using the fixed seed schedule. Hold parameters unchanged until all forward scans, all chunk losses and the full parent backward pass finish. Use the original AdamW rate 0.001, weight decay 0.0001 with rate/mixture exclusion, clipping 1.0, maximum 30 epochs, patience 8, and minimum equal-parent VAL present64 raw-NLL selection.

The reset arms must use this same parent-step loss/update schedule. H020 originally stepped once per shuffled tile; comparing a parent-step carry fit only to that historical fit would confound continuity with optimization. The new matched maximum is 60 parent steps rather than 480 tile steps. Report both budgets. Do not change the budget after seeing results.

Start with ordinary PyTorch autograd and non-reentrant activation checkpointing of source-encoder blocks and scan/readout blocks. Checkpoint recomputation must use the same current parameters and fixed prepared observations. Do not detach boundary states: truncated backpropagation would be a different gradient and a separate explicitly labelled arm. A custom adjoint is unnecessary for the first implementation and would need separate qualification.

Keep unchanged fixed banks on CPU and transfer only required original operator blocks. Do not gather eigenvectors for all parent readout segments at once; the existing eigenvectors[segment_graph] pattern would create a large repeated dense allocation. Retain anchor contexts and the small prefix/suffix boundary tables. Stream original readouts in blocks, for example 128 centers, accumulating one scalar parent loss. Readout contexts do not feed new anchor sources in the current model, which permits exact recomputation of target blocks once the layer’s global anchor contexts are available. All Q184 states still propagate even when only Q120 are decoded/scored.

One directional accumulator contains 3×184×8×2 = 8,832 scalars, or 35,328 bytes in float32. Boundary snapshots for two directions, two layers and nine tile edges contain 1,271,808 bytes, about 1.21 MiB. This is an arithmetic state estimate, not measured GPU memory. Original dense caches, encoder/backward activations, anchor contexts, readout buffers, optimizer and allocator costs remain. Preserve cache fields, dtype and serialization rather than compressing them to meet the estimate.

## Gates and stopping point

Before any fit, independently qualify reset replay against the original per-tile model and streamed carry against a small dense whole-parent reference using exactly the same anchor union. Require anchor/target logits, both source/mass scans, parameter VJPs through every layer, directional derivatives and one clipped AdamW update to agree at prespecified float64 and float32 tolerances. Include noncommuting operators across a cut, a cut inside one original tree, a coincident source/readout self term, clipped terminal tiles, sample-order permutations with restored joins, and a loss in a late tile that has a nonzero gradient through an early-tile source. Failure blocks fitting; do not relax tolerances or substitute detached training.

Then profile a genuine complete TRAIN parent forward, loss, backward and optimizer step with the original cache fields before admitting the 12 fits. Root selects and registers the proposal, budgets and tolerances before implementation or execution. Root owns scheduling and storage. This proposal does not bypass the current compact CUDA parity failure with an unqualified fast path.

Freeze the existing primary overall accuracy/NLL/Brier and the four H023 boundary bands plus separate terminal accounting. Show each parent and each separate fit. Any improvement remains a repeated-VAL development observation. A new independent biological confirmation would need its own root registration and authorized data; no TEST or new-parent access is included here.

Source basis: anchor_kernel.py lines 221–275 define the bidirectional numerator/mass action, lines 332–349 define normalization, and lines 389–409 show that nonlinear sources update at anchors while targets are readouts. biological_candidate.py lines 471–525 define existing tile anchors and complete-tree preservation; lines 765–795 define the original per-tile optimizer schedule. Exact source hashes accompany this proposal.
