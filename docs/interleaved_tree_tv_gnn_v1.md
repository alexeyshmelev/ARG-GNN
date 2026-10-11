# H033: interleaved observed-tree and genomic evidence correction

This is one utility experiment with two freshly fitted compositions:
`interleaved=(B A)^3` and `blocked=B^3 A^3`.
Both use the selected immutable H025 genomic-smoothing emissions and skip, the
same observed native trees, source policy, confidence, 52 allocated scalars,
initialization, and three calls each to A and B. Local confidence-dependent
calibration is possible in **both** arms. Only superiority over the matched
blocked arm supports useful interleaving. Tree harmonic solves, TV, graph
propagation, and operator splitting are established. This is not yet a novel
operator or an ancestry mechanism claim.

## Inputs and source roles

Preparation reuses the immutable H030 native/cut geometry and source adapter.
All native nodes and original/cut edges survive. Younger observed references
use their existing analytic 350-generation source placements. Older observed
references remain at their actual native sampling nodes. There are no
fictitious older 350 nodes or new labels.

H033 explicitly selects each query's **actual native sampling row**, rather
than H030's stored query census-cut readout. These rows are distinct, and the
code asserts their disjointness from genuine reference source placements.
Holds with no retained original readout remain in the prepared geometry but
contribute no solve copy. There is no padding or dummy query value.

At each genuine source node, obtain H030 local unit fields from the original
inverse-family-class-count log weights. Subtract the uniform prior over the
represented classes within each family, and apply the explicit family support
mask. Empty families stay zero. Older ANA has no source or pseudocount and its
contrast input column is zero. The reference RHS is
`r=W_R reference_contrast`, where W_R is bias-free 8-to-4.
The binary `m_i` source mask is one at an actual source placement, regardless
of the number of references sharing that placement, and zero elsewhere.

Four-channel states at native, internal, and analytic cut nodes are learned
predictor surrogates. They are not population states, query truth, or ancestry
truth appropriate to each node's age. Query ancestry never enters preparation
or the forward pass. Exact-span TRAIN target mass is opened only after the
target-free geometry seal.

## Anchored harmonic tree step A

Let `x0[b,q,:]=ell_H025[b,q,:]-mean_class(ell_H025[b,q,:])`.
The frozen top-two logit margin determines
`tau[b,q]=1+sigmoid(a+b_margin*margin_H025[b,q])`.
The same confidence map is held fixed across the three A calls for a forward
pass. It is recalculated from frozen H025 logits, not from query labels or the
current surrogate state.

For child i and parent j, define conductance

`c_ij = mean_f exp(-[1,.1] dot edge_basis[i,f,:]) / max(degree_i,degree_j)`.

Both age-family coefficients are fixed at H030's initialization 1/.1.
No four learned family-cost parameters are hidden in the allocation.
Conductance is nonnegative and the weighted degree is at most one.

For each retained readout's complete observed tree, S selects its actual Q
sample rows. With fixed interior absorption epsilon=0.01 and a shared bounded
`lambda_T=sigmoid(raw_tree_strength)` initialized to 0.1,

`K=epsilon I + lambda_T L + S^T diag(tau) S + diag(m)`

`z=K^{-1}[diag(m) r + S^T diag(tau) x]`, `A(x)=S z`.

Interior vertices receive only the fixed 0.01 absorption, rather than a unit
anchor at every vertex. Source and query anchors remain explicit. K is SPD.
For the asserted disjoint source/Q roles and lambda_T<=1, its eigenvalues lie
between 0.01 and `0.01+max(tau,1)+2 lambda_T<=4.01`.
This bound does not imply a measured error bound on a large FP32 tree.

The implementation never forms K or an inverse. FP32 elimination starts with
`d_i=epsilon+m_i+tau_i(on Q)+lambda_T weighted_degree_i` and RHS h.
For every child c of parent p, after eliminating descendants,

`d_p <- d_p - (lambda_T c_cp)^2/d_c`

`h_p <- h_p + (lambda_T c_cp) h_c/d_c`.

Solve each root `z=h/d`, then substitute downward:
`z_c=(h_c+lambda_T c_cp z_p)/d_c`.
Polytomies and multiple roots use the same scalar-precision/vector-message
rules.

A custom first-order backward solves `v=K^{-1}S^T grad_output` with the same
elimination. It returns
`grad_x=tau*v_Q`,
`grad_r=m*sum_readouts(v)`,
`grad_tau=sum_class(v_Q*(x-z_Q))`, and
`grad_lambda_T=-sum_edges c_cp*(v_c-v_p) dot (z_c-z_p)`.
The reference sum is implemented through the replicated-node origin gather.
Higher-order derivative support is not claimed.

## Ordered step B

B is exactly eight FP32 projected-dual iterations approximating the classwise
weighted TV proximal map. Let `D x=x[b+1]-x[b]`,
`w_b=min(1,100/(position[b+1]-position[b]))`, and
`lambda_G=sigmoid(raw_tv_strength)`, initialized to 0.1.

Each B resets `p0=0` and repeats eight times:

`y_k=x-D^T p_k`

`p_{k+1}=clip(p_k+0.25 D y_k, -lambda_G*w, +lambda_G*w)`.

Return `x-D^T p8`. The transpose is
`[-p0,p0-p1,...,p_last]`. The unweighted D has squared operator norm at most
four, so step 0.25 is safe for the dual projected-gradient update. Physical
distance weights enter its box bounds, not D. The finite algorithm is the
definition of the implemented B, not a claim of exact TV convergence.
No link crosses an existing 250kb tile boundary.

## Output, allocation and execution

Return frozen H025 logits plus the bias-free zero-initialized 4-to-4 W_out
applied to `center_same_age(x_final-x0)`. Centering uses the original complete
Q roster and exact observed age groups at every original readout.

The 52 allocated scalars are W_R32, W_out16, confidence2 and strengths2.
The older-ANA zero column removes four usable reference weights. Reference
contrast rank is at most five because each represented age-family contrast
sums to zero. Realized effective capacity is unmeasured, so only allocated
capacity is matched.

Each 32-readout block packs one genuine complete tree copy per retained
readout as a ragged block-diagonal forest. Elimination levels run across those
copies together. Origin indexes gather the shared reference projection and
sum its gradients back to original node rows. This avoids a CUDA solve loop
per native hold. RHS gathers occur inside a nonreentrant checkpoint, and the
custom solve avoids storing per-level autodiff histories.

At P<=2500, a P-by-120-by-4 FP32 query field uses 4.8 MB. For illustrative mean
hold size 700, replicated geometry has about 1.75 million nodes and a
four-channel field would use 28 MB. The active solve field is at most the
actual 32-readout packed node count times four FP32 values. Metadata and
precision/message arrays add memory. Actual maximum hold/packed sizes are
recorded during preparation. The first real TRAIN optimizer update records
forward/backward time and per-arm CUDA allocated/reserved peaks and counts
toward the 160-update budget. No measured real-tile feasibility is claimed
from the tiny CPU check.

## Frozen scientific and execution protocol

Use only seed101, FP32, original 16 TRAIN and 16 VAL complete tiles, and ten
epochs/160 optimizer updates per arm. Projection learning rate is 0.01 with
weight decay 0.0001. Confidence/strength scalar learning rate is 0.002 with
zero decay. Clip gradient norm at one. Both arms share these prospectively
fixed choices and start bit-identically.

Retain **earliest minimum equal-parent Present64 VAL NLL** selection.
Accuracy is reported from that selected checkpoint. There is no hidden
accuracy reselection. The original biological scorer, exact-span loss,
100bp support, masks, offsets, target joins and per-tile frozen H025
computation are unchanged. Reuse selected H025 and H026 comparisons without
refitting either baseline. There is no dependency on live H030 output.

Input schema is `interleaved_tree_tv_gnn_input_v1`. It uses the same active
root authorization, seed, CUDA lease, resource caps and observed-parent
transport as H032. Required refs are config, runtime, storage, the four frozen
baseline refs (source/config/completed/checkpoint), emitter_bindings,
target_bindings, and the three strong_reference refs
(completed/config/checkpoint). Both new implementations must be pinned in
source_sha256. Config helper refs pin the reused scientific sources,
`project_storage_full_reservation_v1.py`, the actual
`launch_reserved_gpu_full_storage_v1.py`, immutable V6 and the original V9
core. The actual launcher must be a reservation consumer binding.
No retry-v3 adapter or new guard infrastructure is used. Root owns admission,
registration, reservation and GPU launch.

A useful result needs accuracy improvement on both development parents
against blocked and frozen smoothing, with boundary precision/F1 and
short-tract/class-recall checks. NLL alone or a blocked tie cannot support the
interleaving accuracy claim. Inferred-tree error, in-sample TRAIN teacher
predictions and reuse of the two development parents limit interpretation.

The author receipt records a tiny synthetic CPU FP32 dense/residual/VJP check,
TV checks, exact initial skip and finite checkpointed toy updates. It opens
no real observation or target arrays and uses no GPU.
