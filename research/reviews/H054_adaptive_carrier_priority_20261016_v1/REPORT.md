# H054 adaptive-carrier priority audit and mathematical design boundary

2026-10-16. Source/literature-only follow-up to H051. This is not an acceptance verdict or an implementation proposal accepted for execution. All writes are confined to this new directory on anogena. No model or architecture code, model execution, GPU work, real observation/label/checkpoint access, install, guard/accounting operation, admission, Git operation, or edit to another directory occurred.

## Finding before further architecture work

As presently stated, “adapt the carrier to the hidden-state residual, retain a fixed number of slots, and bound projection loss” is a generic adaptive compression/model-reduction direction. Learned graph coarsening, multiresolution projection, largest-coefficient adaptation, certified reduced models, and compressed recurrent memory are all established. A local residual certificate and the finite-horizon inequality below are standard mathematical consequences, not sufficient architectural novelty.

The narrow unresolved capability is practical and testable: under changing observed partitions, retain an older predictive mode outside the last-two-cut carrier at a fixed, fully counted memory budget, more efficiently than a strong fixed multiscale carrier AND an adaptive Haar/coefficient-retention control, while approaching the same-weight full-leaf recurrence. A meaningful result requires the improvement to survive storage of memberships, transient leaf features, training activations, and selection time. The five papers below do not establish that this exact capability is impossible or already solved, but this short audit cannot establish priority for it either. Calling an adaptive partition a new general graph operator would be unsupported.

Root reports a real TRAIN-observation audit with22 partition changes and21 nonzero innovation operators. This audit has not read that payload or receipt. The reported counts are geometry-only evidence: nonzero J does not show that the learned state has a nonzero projected component, that the lost component is predictive, that a chosen K is feasible, or that compression improves model accuracy.

## Five closest primary papers, and exact overlap

Only these five papers are used as the priority shortlist. Other papers appearing in their reference lists are not claimed reviewed.

| Primary paper | What directly overlaps | What the proposed constrained experiment still needs to establish |
|---|---|---|
| Ying et al., Hierarchical Graph Representation Learning with Differentiable Pooling, NeurIPS2018, arXiv1806.08804v4 [P1] | DiffPool learns feature-dependent node-to-cluster assignments S, pools features as S^T Z and adjacency as S^T A S, and establishes permutation invariance under appropriate component GNN assumptions. Input-dependent equivariant pooling and a prescribed coarse width already exist. | DiffPool is a learned soft hierarchy for graph representation. Its inspected method does not give a hard leaf-aligned recurrent carrier with the specific pathwise projection certificate below. That distinction is a constraint/specialization, not proof that adaptive memory is new. Include a learned pooling/compression control if claiming advantages over adaptive pooling, rather than only comparing a deliberately weak fixed cut. |
| Wang et al., Haar Graph Pooling, ICML2020, arXiv1909.11580v3 [P2] | A graph-clustering hierarchy defines an orthonormal Haar basis. Compressive transforms retain coarse coefficients and discard fine detail. Reconstruction error as energy in discarded orthogonal detail is exactly the relevant geometry. Mean-preserving refinement and multiscale coefficient storage are established. | Current-graph low-pass pooling is different from retaining past leaf-aligned detail across changing carriers. The proposed mechanism must beat equal-slot fixed multiscale pooling and simple adaptive coefficient retention. The paper's qualitative “small information loss” statement is not a uniform downstream prediction guarantee. |
| Cohen, Dahmen and DeVore, Adaptive wavelet methods for elliptic operator equations: Convergence rates, Math. Comp.70(233),27–75 (2001) [P3] | Adaptive wavelet selection, keeping a limited number of significant coefficients, approximation error rates, and computational complexity relative to best N-term approximation already predate this proposal. Its publisher-supplied abstract explicitly benchmarks against retaining the N largest solution coefficients and claims a rate-optimal adaptive method under its elliptic assumptions. | Those PDE hypotheses are not a guarantee for a learned GRU, but residual-driven adaptation plus an error/rate claim is established model-reduction territory. A greedy split rule with no optimality guarantee is weaker than this literature's type of approximation theorem. The new work would need a genuinely sharper constraint/result for causal changing graph carriers, or convincing empirical cost/error superiority. |
| Grepl and Patera, A posteriori error bounds for reduced-basis approximations of parametrized parabolic partial differential equations, ESAIM:M2AN39(1),157–181 (2005) [P4] | Galerkin reduced spaces, a greedy adaptive construction, rigorous output-error bounds, time-dependent problems, and explicit offline/online operation counts already coexist in a primary method. “Certified adaptive reduced temporal state” is not a new principle. | Parabolic/coercive and affine-parameter assumptions do not automatically apply to a neural update. Here the update constants and finite-horizon accumulation must be stated and verified separately. A GRU's gating is not a substitute for those assumptions. |
| Rae, Potapenko, Jayakumar and Lillicrap, Compressive Transformers for Long-Range Sequence Modelling, ICLR2020, arXiv1911.05507v1 [P5] | Fixed-size recent and compressed recurrent memories, mean/max/convolutional/most-used compression, and reconstruction or attention-reconstruction objectives already target task-relevant long-range memory under a fixed slot allocation. The attention-reconstruction objective deliberately permits loss of information no longer used by attention. | It is temporal-token compression rather than a hard partition of persistent graph leaves. A named leaf-carrier specialization does not negate this overlap. Show that the geometric certificate or leaf alignment buys an actual retention/cost capability beyond an equal-budget generic memory compressor. |

Primary URLs and reading limits:

- [P1] https://arxiv.org/html/1806.08804v4 . Read section3.2, equations3–6 and permutation proposition, plus section3.3 auxiliary losses.
- [P2] https://arxiv.org/html/1909.11580v3 . Read sections3–4, Definition1, orthogonality, compression discussion and relevant transform identities. The v5 request returned404, then the actual v3 was obtained.
- [P3] https://doi.org/10.1090/S0025-5718-00-01252-7 . Read verified bibliographic metadata and the publisher-supplied primary abstract through Crossref. The AMS HTML request returned403. No full proof or theorem hypotheses were inspected, so the comparison is restricted to the abstract's explicit adaptive/N-term/error/complexity claims.
- [P4] https://doi.org/10.1051/m2an:2005006 . Read the primary abstract and bibliographic record at https://www.numdam.org/item/M2AN_2005__39_1_157_0/ . The publisher request returned403. No full theorem proof was inspected, so no assertion about a specific estimator's constants is made.
- [P5] https://arxiv.org/html/1911.05507v1 . Read sections3.1–3.3 and Algorithms1–2, including the exact slot allocation and compression losses.

The five references constrain the novelty claim even with the two older full-text limits. They are a targeted shortlist, not an exhaustive systematic search. Material hashes, exact excerpts and retrieval outcomes are recorded separately.

## One concrete rule: causal residual-based binary refinement

This is a mathematical specification only. It illustrates how little novelty follows from the generic direction. It is not model code or execution authority.

### State, inputs, order of operations

There are H fixed, authenticated persistent leaf witnesses. Feature width F, hard capacity K, tolerance epsilon, channel order and all numerical conventions are frozen before evaluation. H is fixed within a sequence; independent sequences have separate carriers. No internal-node identity is assumed.

Let R_t be the observed joint partition of the last two DISTINCT cut episodes available by time t, with the same explicit bootstrap/hold convention for every control. More generally a different observation-only base can be frozen, but changing it changes the condition. Require |R_t|<=K. If it fails, report this event/sequence as infeasible for this refinement rule. Do not silently merge base cells, enlarge K or omit the event. At most K occupied feature rows are allowed. Unused slots may remain empty.

The stored state is a hard partition S_{t-1}, its occupied means v_{t-1}, and the reconstruction h_hat_{t-1}=L_{S_{t-1}}v_{t-1}. Using only current/past observations, compute

    z_t = U_t(h_hat_{t-1}; G_{≤t})
    S_t = refine(R_t, z_t, K, epsilon)
    h_hat_t = P_{S_t} z_t.

Predictions use the compressed h_hat_t and the same current features/readout as the controls. The reference full-leaf recurrence is

    h*_t = U_t(h*_{t-1}; G_{≤t})

with the SAME weights, inputs and initialization.

This specification applies compression AFTER U_t. It does not insert the H051 P_BC P_AB carry before U_t. It is therefore a different compressed recurrence, not a claimed drop-in equivalence to H051. A comparison must include a fixed-carrier control with the same update/compression order, as well as actual H051. Otherwise a gain could come from moving the projection rather than selecting adaptive cells.

### Deterministic split and selection

Start with S=R_t. For each occupied nonsingleton cell Q:

1. Compute its feature means mu_Q,j and within-cell feature variances. Choose the channel j_Q with largest variance, breaking channel ties by the frozen channel index. This does not use a leaf index.
2. If all variances are zero, Q is unsplittable. Otherwise split it into Q_-={i in Q:z_i,jQ<=mu_Q,jQ} and Q_+={i in Q:z_i,jQ>mu_Q,jQ}. Both sets are nonempty in exact arithmetic.
3. Score the split by the exact decrease in normalized squared reconstruction error,

       g_Q = |Q_-| |Q_+| / (H |Q|)
             * ||mu_Q- - mu_Q+||_2^2.

   This score uses all channels although one channel proposes the split. One binary split consumes one additional feature row.

If ||z_t-P_S z_t||_leaf<=epsilon, stop. Otherwise choose ALL frontier cells tied at the largest positive g_Q and split them simultaneously if their total added slots fit the remaining capacity. Recompute scores only for new children. If the entire highest-score tie group does not fit, stop and return the current partition with an explicit tie-limited flag. Stop also at K, no positive split, or zero residual. This conservative tie policy can underfill K and is intentionally part of the rule.

The final residual is evaluated directly,

    delta_t^2 = (1/H) sum_Q sum_{i in Q} ||z_t(i)-mu_Q||_2^2.

The identity delta_before^2-delta_after^2=g_Q is the ordinary within/between-cell variance identity. The rule is not claimed globally optimal over partitions, minimax, rate-optimal, or computationally linear. With highly unbalanced splits, repeated scans can cost O(KHF), with additional frontier selection work. A faster implementation would need separate source review and an actual cost proof/measurement.

The rule always refines R_t and never stores more than K occupied mean vectors. It does not guarantee delta_t<=epsilon at a given K. Return the actual residual, counts and the exact stop reason. A residual threshold cannot be promised independently of capacity.

### Causality and equivariance, including the tie limitation

For any simultaneous permutation of leaves, memberships, observations and state, means/variances/gains are unchanged and each proposed child set is permuted. Selecting a complete tied group preserves this property. Renaming anonymous base or carrier labels only renames rows. Thus reconstructed h_hat_t is leaf-permutation equivariant in exact arithmetic. Channel rotations are not an intended symmetry; fixed feature-channel order is part of the specification. There is no global internal ancestor matching, leaf-ID embedding, or use of current/future labels.

Choosing the smallest leaf/sample/country ID to fill the last slot would break this claim under witness renaming. On symmetric inputs a deterministic equivariant rule generally cannot choose one among interchangeable cells when the capacity admits only one. “Exactly K occupied cells on every input” and this tie symmetry can conflict. This rule promises a fixed cap K, not exact occupancy. A stochastic alternative would need an explicitly coupled/distributional equivariance claim, rather than pretending deterministic exact symmetry.

FP32/FP64 reductions can change near-ties under a leaf permutation. A numerical implementation must declare reproducible reduction/tie conventions and test its stated tolerance. A mathematical equivariance proof is not a bitwise floating-point guarantee. Membership selection is discrete. If training is later proposed, one honest convention is to stop gradients through selection and differentiate the averages/update conditional on selected memberships. That is not an end-to-end differentiable selection rule and needs to be disclosed.

## What must actually be stored and counted

Let k_t<=K. For a concrete FP32/int32/int64 layout, a conservative inference inventory is:

| Allocation | Size and role |
|---|---|
| Persistent occupied means | 4 k_t F bytes |
| Persistent leaf-to-cell membership | 4H bytes, assuming k_t fits int32 |
| Occupied counts | 8 k_t bytes |
| Last two distinct observed cut vectors, if not already common input storage | 8H bytes for two int32 arrays; current/base joint construction may need more |
| Certificate/stop record | delta_t, measured occupancy, epsilon feasibility, tie/cap stop flag, and the bound accumulator; small per step, but complete audit logs are O(T) |
| Aligned witness/catalogue mapping | Explicit common-input allocation. “Metadata” does not mean zero cost. No dense H-by-K assignment matrix is necessary for a hard carrier. |
| Transient decoded old state | H-by-F floating values, unless fused into a separately justified implementation |
| Current leaf forcing, candidate z_t and update gates | Additional H-by-F arrays plus the encoder/update's existing intermediates. The simple specification materializes them. Do not count only the stored K-by-F means. |
| Selection/certificate scratch | Old/new membership arrays O(H), cell/child means and statistics O(KF), gains/frontier O(K), plus reduction and grouping workspace. Repeated greedy scans incur time even when scratch is reused. |
| Graph encoder, readout, logits, observations and schedules | All common allocations still count toward total peak memory. They do not disappear because temporal features are compressed. |

The simple implementation has at least an H-by-F transient even if persistent feature memory is K-by-F. With separate decode, forcing and candidate buffers, several H-by-F arrays can be live simultaneously, before GRU gates/encoder activations are counted. No factor-H/K peak-memory advantage is established.

Under full BPTT, saved updates/gates can remain O(THF), and selected memberships can cost O(TH) integers. Checkpointing/recomputation or custom fused backward may reduce this, but they must be applied fairly to controls and their runtime counted. Storing old full-leaf residuals, full hidden states, all past partitions, uncounted wavelet coefficients, or a replay cache is extra memory. Any such allocation consumes the declared budget or invalidates the comparison. A post hoc run using a full-leaf teacher to choose cells is not the causal rule above.

A feature-slot budget alone is consequently weaker than a byte budget. Report both. An equal-slot comparison is a controlled representation experiment; it is not sufficient proof of a system-memory advantage.

## Honest finite-horizon propagation bound

Use the normalized leaf norm ||x||_leaf=(H^{-1}sum_i ||x_i||_2^2)^{1/2}. Let U_t be the entire recurrent update for the fixed observed input at t. Assume, explicitly,

    ||U_t(x)-U_t(y)||_leaf <= L_t ||x-y||_leaf

for all x,y in a stated forward-invariant domain containing both the full and compressed trajectories. The constants must bound the actual complete update, not just one selected linear map. No assumption L_t<=1 is made.

Let e_t=||h*_t-h_hat_t||_leaf and delta_t=||z_t-P_St z_t||_leaf. A direct triangle inequality gives

    e_t <= L_t e_{t-1} + delta_t,

and therefore, for a finite horizon T,

    e_T <= (product_{j=1}^T L_j)e_0
           + sum_{s=1}^T [delta_s product_{j=s+1}^T L_j].

The empty product is1. Data-dependent/discontinuous partition selection does not invalidate this pathwise inequality: it compares with the projection actually selected on the compressed trajectory, rather than requiring the selection map itself to be Lipschitz. It also does not make the selection map smooth or training-stable.

If L_t<=L and delta_t<=epsilon for every step, then e_T<=L^T e_0+epsilon sum_{q=0}^{T-1}L^q. For L=1, this is e_0+T epsilon. For L>1, it can grow exponentially and become useless. The report must show actual bound values, not merely state that a theorem exists. Empty or tiny errors do not create a predictive gain.

With floating error, use an upper bound bar_delta_t>=true projection error and an additional implementation discrepancy eta_t for rounded update/reconstruction relative to the mathematical map. Replace delta_t by bar_delta_t+eta_t. FP64 accumulation alone is not a rigorous roundoff certificate. Without validated arithmetic or an explicit rounding bound, call the computed quantity a reconstruction-error diagnostic rather than a certified upper bound.

### An explicit conservative GRU constant, rather than an assumed contraction

For a leafwise reset-after-affine GRU with fixed input f_t, write

    r=sigmoid(a_r+U_r h),
    z=sigmoid(a_z+U_z h),
    n=tanh(a_n+r*(U_n h+b_n)),
    U(h)=(1-z)*n+z*h.

Products marked * are componentwise. The a terms depend only on the same observed forcing and fixed input parameters. If ||h||_infinity<=B throughout the domain, put M_n=B||U_n||_{infinity->infinity}+||b_n||_infinity. Since sigmoid is1/4-Lipschitz and tanh is1-Lipschitz,

    L_n <= ||U_n||_2 + M_n ||U_r||_2/4
    L_GRU <= 1 + L_n + (B+1)||U_z||_2/4

is a conservative valid upper bound in the per-leaf Euclidean norm, hence in the normalized leaf norm for a shared leafwise GRU. The leading1 comes from the direct state path; this bound generally does not show contraction. Tighter gate-range bounds may replace the leading factors, but must hold over the entire stated domain.

Starting at0, a conventional GRU keeps h in [-1,1] coordinatewise, and cell means preserve this range, so B=1 is available for this specific update. This range fact is not a contraction proof. A different GRU reset convention or state-dependent graph update needs its own bound. Empirical Jacobian samples are local observations, not uniform L_t certificates. The architecture cannot assert calibrated probabilities, class-accuracy improvement, or true-tract retention from this state-error bound.

If a shared readout R_t is c_t-Lipschitz on the domain, its output discrepancy is <=c_t e_t in the corresponding norm. Agreement of argmax labels additionally needs an appropriate margin bound. If models are retrained independently, their weights differ and this same-weight comparison no longer bounds their difference. Report compression fidelity and independently trained utility as distinct questions.

## A–B–A and a decisive test beyond it

### Correct positive/negative controls

Let A be a nontrivial fine partition, B a coarsening, and q an A-constant, globally mean-zero state mode. First-order projection P_B may remove q. Returning to A cannot recover it from the collapsed state without new identifying input.

However, H051 alpha1 already preserves that A-mode across the direct A–B–A triplet:

    old joint AB=A, new joint BA=A,
    carry=P_A P_A q=q.

So A–B–A is a positive control for pair transport and a loss/reappearance diagnostic for first-order projection. It cannot by itself establish a new adaptive-carrier capability. If current observations supply the old signal again, even a zero-memory method can appear to recover it; the diagnostic must distinguish memory from re-observation.

A stronger construction is A–B–C–A, with B coarse and C a distinct transverse partition whose detail is independent of A. For an A-mode q, P_C P_A q can be0, so the last-two-cut carrier loses the older mode before A returns. For four equally weighted leaves, A={1,2}|{3,4}, C={1,3}|{2,4}, q=(1,1,-1,-1) is an exact example. Use an update that preserves q during the ambiguous interval and supply no new q-identifying forcing at retrieval.

This example alone still does not prove adaptive superiority: retaining the joint A,C needs four hard cells, and an equal-slot fixed four-cell carrier can preserve everything for four leaves. A test that gives the adaptive method more slots than its control would be uninformative.

### Minimal decisive source-frozen experiment design

No experiment was executed here. Before any implementation or registration, freeze:

1. A family with H substantially larger than K, multiple possible old detail modes/partition orientations, and held-out sequences of coarsenings/transverse cuts/reappearances. Avoid choosing a single orientation that favors one basis. Begin with a SAME-WEIGHT known update/reference and measure state reconstruction/retrieval error directly. Include A–B–A as the pair-transport success control and at least one beyond-pair loss case. Report infeasible |R_t|>K and tie-limited events.
2. Actual H051 alpha0/alpha1, a fixed-carrier recurrence with the same post-update compression order, the strongest prespecified equal-slot fixed multiscale carrier chosen on TRAIN/VAL, a simple adaptive Haar/coefficient-retention compressor with the same observation/history access, and full-leaf recurrence. A full-leaf baseline is an uncompressed reference, not an equal-slot method. Freeze width, forcing, update, readout, seeds and readout timing.
3. A K/error/time/byte curve, including persistent bytes, transient peak, BPTT activations if training, membership/schedule costs, selection time, and the actual finite-horizon error bound and its tightness. Do not turn topological availability counts into evidence of model benefit.
4. Only if the mechanism survives, test identical frozen inferred-tree observations and independent parent splits with the existing false-short-burden/true-short-retention metrics. Learned selection/update variants need matched parameter/optimizer budgets and no TEST-based basis/threshold choice. Biologically useful tradeoff remains a separate result.

A substantive evidence delta would be either a sharper proven constraint/result for causal carrier selection that is absent from these known adaptive schemes, or a reproducible quality-cost advantage against BOTH strong fixed multiscale and ordinary adaptive coefficient compression under the complete allocation accounting. Merely proving the displayed triangle-inequality bound, reporting nonzero J, beating alpha0, showing a single A–B–A success, or calling stored rows “compressed memory” is insufficient.

Stop the architecture direction as a novelty claim if generic adaptive Haar/learned compression matches it at equal total cost, the transient/BPTT overhead eliminates the claimed savings, almost all events need K near H, the bound is vacuous without useful empirical fidelity, or gains only appear when hidden full-leaf history is retained outside the counted budget. It may still be a useful engineering option, but the claim would then be engineering specialization rather than a new principle.

## Evidence and authority limits

This follow-up creates a report and source/quotation receipts only. No proposed rule was implemented, numerically verified, trained or scored. The two older papers were inspected at primary-abstract level only, as explicitly stated, and no uninspected theorem was imported into the GRU analysis. No model accuracy, ancestry score or measured compression saving is claimed. Current Relate data still cannot be assumed to carry persistent internal biological identities. Root owns any next source design, registration, resource admission, implementation integration or execution. Only physical GPU0 is eligible for any future authorized experiment under the current user policy.
