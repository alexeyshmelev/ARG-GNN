# H032 prospective mathematical core and protocol

This source-only hypothesis tests whether a donor-dependence proxy helps static whole-tree propagation. H029's memory comparison and H031's weak direct-genotype errors provide context, not proof of a donor bottleneck. No new primitive, posterior covariance, causal attribution or ordered-architecture claim follows.

## Field and proof

Keep H030 placements, additive family edge costs and weights omega_r=1/n_family,class. For a fixed donor subset S and node v, let k(v,r)=exp(-tree_cost(v,r)), M=sum omega*k*e_class, Z=sum M, pi_r=omega*k/Z. Retain H030 p=M/Z exactly through its unchanged first-moment implementation at fixed cost coefficients.

G=sum_rs omega_r*omega_s*k(v,r)*k(v,s)*k(r,s)*e_r*e_s^T; C=G/Z^2. Each family remains separate. For a local colocated component Y=sum omega*e_class, M=H=Y and G=Y*Y^T, including cross terms between distinct colocated donors. H=sum omega*k(v,r)^2*e_class is a doubled-distance first moment, not sum omega^2*k^2.

Across an edge of cost c, M'=exp(-c)M, H'=exp(-2c)H, G'=exp(-2c)G. For disjoint directional components at their common boundary v, every cross-component source pair passes through v, giving k(r,s)=k(v,r)k(v,s). Hence merge M=M_a+M_b, H=H_a+H_b, G=G_a+G_b+H_aH_b^T+H_bH_a^T. Repeated positive merges are associative in exact arithmetic. Upward components include local sources and children. Outside components merge local, parent-side and siblings excluding the recipient; prefix/suffix merges avoid total-minus-child cancellation. Merge upward and outside for full fields. These are exact finite-tree sums for each respective donor subset, not a tree-likelihood approximation.

For the independent donor kernel K=I, local G is diagonal sum_r omega_r^2*e_r*e_r^T, not Y*Y^T even for colocated donors. Its edge transform is still exp(-2c)G, but cross-component G terms are zero. All arms retain the same first moment and labels.

K_rs=exp(-tree_cost(r,s)) is PSD: construct a unit-variance root variable and for each edge a=exp(-c), X_child=a X_parent+sqrt(1-a^2)epsilon. Their covariance equals K; disconnected roots have independent variables. Zero-cost edges and colocated donors are allowed. Consequently C=B^T K B is symmetric, nonnegative and PSD. C is a weighted support Gram, not a posterior uncertainty of query ancestry. 1^T C 1 lies in [0,1]. For K=I its reciprocal gives weighted effective count for nonempty support. An absent class is zero and empty families have zero fields/false masks.

## Stable relative FP32 messages

Store a detached finite gauge q plus log-relative M-q, H-q and G-2q. Empty components carry q=0 and -inf logs. Transport sets q'=q-c, leaves relative M and G unchanged, and subtracts c from relative H. Merge chooses the largest active component gauge, rescales M/H by q_i-q and G by 2(q_i-q), and applies masked positive log sums/cross terms. Final C uses relative G minus twice logsumexp(relative M). Transport includes numerically zero cost.detach()-cost compensation in relative logs to preserve exact cost gradients despite detached gauges. No huge absolute second-log normalizer subtraction, tiny-mass division, pseudocount or FP64 actual-parent matrix occurs. H030's unchanged p remains the value source; the independently gauged M only normalizes moments and is mathematically the same M. Floating-point associativity differs, so numerical tolerances apply to dense Gram oracles, while p parity at equal parameters is bit-exact.

Source-free serial subdivisions preserve costs by telescoping and preserve all kernel moment fields. The finite nonlinear GNN is not serial-subdivision invariant. Dense matrices are permitted only on a tiny handmade FP32 oracle. Actual source arithmetic scales O(N*4^2) per family/forest sweep. Saved autograd and allocator overhead remain unmeasured.

## Static learned protocol and repeat policy

Use H030's static encoder/decoder/value projections and allocated template, adding four bias-free 20-to-16 Gram gate projections (encoder up/down and decoder up/down). Their zero initialization preserves H030-static initial behavior. Centered p contrasts remain the sole value source; symmetric Gram entries modify gates only. Local/outside exclusion is unchanged. No carrier scan or learned history is executed. All new arms share one copied seed-101 initialization and allocated model/head capacity.

`first_moment_only` feeds zero Gram gates and exactly reproduces H030 static for identical inherited state. It is available only for source/toy verification. Do not refit this answered unchanged condition. Reuse H030's selected static sealed probabilities/checkpoint/metrics as the historical first-moment comparator when H030 completes. The three new fitted arms have no H030-completion dependency. Bind only the immutable launched H030 input `research/reviews/novelty_discovery_execution_v1_root/H030/input.json` (SHA256 `20642248f51f54d270118c84fa2a038299ab63d1abc7bf56c85e71c6c2972414`) and its prospective output `experiments/results/reference_logcavity_gnn_v1_seed101_GPU3`. Validate identical observed parents, emitter/target support, H025 skip, H030 source/config and scorer metadata. Read no changing H030 checkpoint or interim output. Initialize all H032 models from the shared seed-101 template, with no H030 warmstart. Reuse already-completed frozen H026 native-reference seals/metrics and H025 during fitting. H032 fit completion is complete even while final H030 selected-static comparison is pending; a separate bound metadata-only summary joins final selected results after both fits complete.

The three new trained arms are `independence_gram`, `marginal_outer_gram` and `tree_gram`: one FP32 seed 101, 160 updates each, original 32 complete 250 kb TRAIN/VAL tiles and full Q120 support, frozen H025 skip, unchanged exact-span loss/scorer, earliest minimum equal-parent VAL Present64 NLL. First real TRAIN forward/backward qualification counts toward the 160 updates.

The allocated template is equal, but first-moment-only has 20 zero Gram channels, independence only eight diagonal channels before absent classes, and genuine C and marginal outer products have off-diagonal channels. Older ANA absence removes further channels. Disclose operational active counts/input ranks rather than claiming equal effective capacity. Comparison with frozen H030 includes different extra gate capacity after training, and one-seed historical selection is discovery evidence.

## Alignment-sham feasibility limit

A class/family/date/diploid-block coherent parent-wide donor permutation would keep labels, placements, p, unit diagonal, eigenvalues and PSD while breaking K alignment with actual pi. It destroys the directional factorization above: a pair's permuted path need not pass through the component boundary. No exact 4-by-4 reroot substitution implements that sham.

At maximal N=449,840 and younger R=84, a dense N-by-R-by-R field has 3,174,071,040 FP32 entries, 12.70 GB before outside/full variants or autograd. Even bounded chunks avoid that allocation but retain billions of pair operations per field/tile; donor-feature contractions also cost O(N R^2). Restricted roster movement and timing are unknown without payloads. Therefore the alignment sham is explicitly omitted from this source-only 160-step protocol, not replaced by a class shuffle or history sham. A separate prospectively reviewed bounded implementation and meaningful moved-fraction qualification would be required. Genuine versus independence and marginal-outer controls still cannot establish alignment-specific biological attribution.

## Falsification and limits

A tie with independence weakens usefulness of the dependence field. The marginal-outer control sets K to the all-ones constant kernel, C=p p^T, providing generic first-moment quadratic features through the same 20 channels and four gate projections. It is not independence. Genuine tree C must improve over both independence and marginal outer before a useful donor-dependence claim. A gain over frozen first moments without beating these controls supports extra gate/readout or generic quadratic capacity rather than donor genealogy. Require practical accuracy/NLL gains consistent across parents with class recalls, calibration, short tracts and boundary precision/F1 preserved. Proxy labels, inferred native trees and reused development VAL remain limitations. Existing tree OU covariance, second-order pooling, correlated-observation weighting and H030 rerooting are direct predecessors. Priority and scientific usefulness of this exact coupling remain unresolved.

## Equal-cost parity qualification

Bit-exact p parity holds at equal raw_cost values. The three fitted arms optimize their own age coefficients and may learn different p fields. Their comparison tests the whole learned Gram-gating/kernel condition, not a fixed-p intervention throughout fitting. No donor bottleneck or support-correlation mechanism is proved by this control design.
