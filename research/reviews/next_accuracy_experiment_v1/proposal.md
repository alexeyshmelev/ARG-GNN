# One next accuracy experiment: interleaved whole-tree and genomic evidence correction

**Proposal only.** Test whether feeding genomic confidence changes back through complete trees reduces excess ancestry switches. No implementation, fit, new seed or infrastructure change is authorized by this file.

## Demonstrated bottleneck and limits

Completed H032 selects epoch10 by NLL for all three arms. Present64 accuracy/NLL are independence85.938549%/0.374581, marginal outer85.928390%/0.374689 and genuine tree85.937846%/0.374591. Frozen H025 smoothing is85.939717%/0.389136; H026 native reference is85.914510%/0.382393. Genuine dependence supplies no accuracy or NLL advantage over independence. Earlier epoch9 accuracy is not the selected result.

A concrete failure is excessive, poorly located ancestry switches despite improved probability scores. H025 predicts415/310 boundaries on VAL000/001, matching7/12 of181/168 true boundaries. Final H032 tree predicts976/752, matching33/27: precision3.381%/3.590%, with148/141 true boundaries missed. Recall improves, but fragmentation more than doubles and aggregate accuracy does not. These counts demonstrate a prediction failure; they do not prove that every unmatched switch is a short noisy tract or that topology caused it.

Source adds a precise restriction: H032's graph residual for a query is constant within a native hold, and frozen H025 logits enter only after the reference-driven graph context is computed. H030 also excludes those logits from graph values. Thus the tree route cannot condition its correction on the query's different100bp margins inside the same hold. Missing confidence-conditioned feedback is the hypothesis, not an established bottleneck cause. H030's recovered final first-moment comparison is still pending; do not label its interrupted output completed.

## One layer, with exact forward definition

Use original Q120 bins, R96 labels, all native vertices/edges and analytic350 source insertions. Older sources stay at actual sampling nodes. Reference fields are the H030 class-balanced unit contrasts with explicit empty masks and absent older ANA zero; fix age-potential coefficients at the existing initialization1/.1 in both arms. No Gram or carrier memory is added.

Let `x0[b,q,:]=P ell_H025[b,q,:]`, where P removes the four-class mean. Crucially, lift these observed prediction contrasts onto each query's **actual native sample node**, not a shared hard census group. Let S_h select those Q nodes in hold h, and `r_h=W_R reference_contrast_h`, with W_R8→4 bias-free. r is a predictor evidence field, not population state or a posterior.

Define a symmetric weighted whole-tree Laplacian L_h on every original/cut edge. For edge(u,v), use the mean of the two fixed positive-age affinities divided by `max(degree(u),degree(v))`; hence weighted degree≤1. Query precision is `tau[b,q]=1+sigmoid(a+b_margin*margin_H025[b,q])`, bounded between1 and2 and using frozen top-two margins only. Internal/source precision remains1. For any current Q×4 field x, the whole-tree step is

`K_b = I + S_h^T diag(tau_b) S_h + lambda_T L_h`

`z_b = K_b^{-1} [r_h + S_h^T diag(tau_b) x_b]`,  `A(x)_b = S_h z_b`.

lambda_T is a shared bounded positive scalar initialized0.1. K is SPD, with eigenvalues between1 and5 for lambda_T≤1. Compute z by scalar-precision/vector-message upward elimination and downward substitution on the complete tree; never form a dense matrix or inverse. These established Gaussian/tree-elimination operations let one query's revised evidence reach other queries through the full tree.

The ordered step B is classwise1D total-variation proximal correction over consecutive physical100bp bins of each actual query:

`B(x) = prox_{lambda_G sum_(b,q,c) w_b |x[b+1,q,c]-x[b,q,c]|}(x)`,

with `w_b=min(1,100/(position[b+1]-position[b]))`. Use one fixed eight-iteration FP32 dual-projection implementation, so the implemented B is a finite approximation, not a claimed exact TV solver. lambda_G is shared/bounded positive, initialized0.1. No link crosses the existing250kb tile boundary; this test concerns within-tile fragmentation, not H024 carry.

The proposed layer is `x_joint=(B A)^3 x0`. Return frozen H025 logits plus `W_out center_same_age(x_joint-x0)`, with the original full-roster centering and a zero-initialized bias-free4→4 head. Train W_R, W_out, the two precision coefficients and two scalar strengths:52 allocated scalars, identical in both arms; disclose the older-ANA zero input column and other contrast-rank constraints. Reference dates, labels, confidence and support are observed inputs; query truth never enters features.

## One decisive matched control

The blocked control is `x_blocked=B^3 A^3 x0`. It receives exactly the same bins, whole trees, reference values, confidence weights, positive coefficients, three complete-tree solves, three ordered TV calls,52-scalar allocation, initialization and head. Both fit their own fixed ordering. Only the interleaved arm returns an ordered correction to the trees before the next full-tree solve. The blocked arm already contains whole-tree evidence and ordinary genomic smoothing, so a win tests useful feedback rather than comparison with a weak sequence baseline. Different splitting can change optimization; superiority would not prove an ancestral mechanism.

Forward state is P×120×4: at P≤2500,4.8MB per FP32 field. B uses that bounded state. Evaluate A in blocks of32 readouts with checkpoint recomputation: a working node field is32×N_h×4. At N_h=1000 this is0.512MB per field; even the crude existing449,840-row tile bound used as a per-hold ceiling gives230.3MB per field. Precision/messages add several such arrays; actual maximum N_h, GPU peak and seconds must be recorded on the first real TRAIN update. Cost is O(3 sum_b N_h(b)×4 + 24 P Q×4), without Q² matrices, FP64 matrices or saved node-by-bin history across all three solves. This is an implementable estimate, not measured feasibility.

Use the same single seed101, FP32, original16 TRAIN/16 VAL complete tiles, exact-span TRAIN NLL, fixed160 updates per arm and unchanged biological scorer. Preregister **earliest maximum equal-parent Present64 accuracy, then lower NLL for ties** for both new arms, explicitly differing from legacy NLL selection. Keep legacy selected checkpoints fixed; no H032/H025/H026 refit. Success requires an accuracy gain on both parents versus blocked control and frozen smoothing, with better boundary precision/F1, preserved true-short-tract performance and no class-recall tradeoff. NLL-only improvement, boundary erasure, or a blocked tie rejects the intended accuracy/feedback claim. First-update qualification counts toward160; no large matrix or seed sweep.

## Collisions and what could be a contribution

Whole-tree Gaussian conditioning/electrical smoothing, graph label propagation, fused-lasso/TV, unrolled inference and operator splitting are established. Interleaving noncommuting operators is not new mathematics. RFMix already combines ancestry classification with ordered CRF inference; CRFs and graph-TV denoising are direct collisions. GraphSSM/TGN cover dynamic graph state, GST covers collection feedback, and the project already contains Kron/GP/tree-cavity operators. Publication-level priority would require a focused primary-paper comparison; no exhaustive search is claimed here.

H021 already implements census-group exchange and clade-turnover retention. Its interrupted epoch22 proposed/eta0 scores are79.899%/81.410% accuracy and0.569738/0.533564 NLL; these are partial results under a different backbone, not final controls against H025. This candidate uses neither those hard groups nor overlap-retention: its new condition is per-bin emitter evidence at unique sample nodes and nonlinear ordered correction fed back through all native trees. H024's reset/carry accuracy remains untested; its single FP64 numerical cell and H023's older near-tile deficit do not validate this layer. Artificial-boundary carry is expressly not being retested.

The potentially useful contribution is a small confidence-conditioned whole-tree graph-collection **inference route** that corrects biological classification/tract fragmentation beyond a capacity- and operation-matched blocked composition. It is not yet a novel operator, ancestry posterior or validated mechanism. Inferred-tree errors, in-sample TRAIN teacher predictions and two repeatedly used development parents remain material limitations; a negative result must not trigger another pooling/memory sweep.

Evidence inspected: H032 `experiments/results/reference_dependence_full_storage_v1_seed101_GPU2/completed.json` and selected epoch10 metrics; H025/H026 selected comparison/metrics; H030 `reference_logcavity_gnn_v1.py:261` and H032 `reference_dependence_gnn_v1.py:313`; H021 epoch22 metrics/ledger and census-cut documentation; H024 ledger and tile-boundary diagnostic; existing `next_architecture_novelty_fresh_v1` and `edge_innovation_novelty_fresh_v1` critiques. Only this proposal file was written.
