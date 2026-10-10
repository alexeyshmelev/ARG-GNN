# Fixed-anchor nonlinear boundary kernel GNN

Status: CPU-verified experimental implementation. No training, scoring, GPU job or performance assessment was performed for this prototype. The motivating geometry/depth hypotheses and scientific limitations are recorded in `research/reviews/age_conditioned_boundary_design_v1.md`. The source implements its Cached multiscale boundary flows and Fixed-anchor nonlinear kernel GNN sections. It changes no frozen boundary memory, Kron primitive, dataset, registry or running worker.

## Interface and common observed features

`prepare_anchor_kernel_bank(plain_bank, age_cache=None, num_anchors=32, num_scales=None)` returns an `AnchorKernelBank`. Preparation happens on CPU and solves no eigenproblem. It reuses already cached fixed operators/eigenpairs, verifies their dimensions, normalization, partition, PSD/orthogonality/reconstruction, and builds fixed physical sweep metadata. Without an age cache only plain geometry is available, with S=1 by default. An explicit positive `num_scales` repeats plain geometry into S independent branches.

A supplied `AgeInterfaceCache` must expose `cqq`, `cqr`, `eigenvalues`, `eigenvectors` and parallel `scalar_cqq`, `scalar_cqr`, `scalar_eigenvalues`, `scalar_eigenvectors` arrays shaped `[G,S,...]`. The adapter checks the Q/R identity/index order against the plain cache, requires `normalization="shared_plain_mean_boundary_diagonal"`, and checks the shared plain scale and first plain branch exactly. The architecture-owned `AgeInterfaceBank` supplies `.plain_bank` and `.operator_cache`. Its default S ordering is plain, age350, age1400. The consumer does not create or choose age operators from labels.

The model's ordinary independent source GNN is the unchanged `BoundaryResponseGNN`. It encodes every observed graph once using the **plain** normalized full-boundary response and the same selected observed leaf fields in every control. The ancestry protocol supplies 47 fields with raw parent-branch excluded. No position, record count, anchor index, numerical identity embedding, census vertex, query target, horizon-root class, or extra candidate-only reference summary is added. References keep their independently encoded observed local embeddings throughout all kernel layers. Only query embeddings receive anchor context.

Move the prepared pack with `.to(device, dtype)` once to match the model. The pack's feature/operator tensors cast to the model dtype. Physical coordinate differences are prepared in float64 from the original interval/position values, which preserves each original value exactly. Layout coordinates retain float64 during transfer. External scoring positions and their original order remain in the unmodified plain bank.

## Fixed measure, chronology and sources

For an exact tile [a,b], choose A=32 predeclared equal-cell midpoints

\[
y_p=a+(p+1/2)(b-a)/A,\qquad \omega_p=(b-a)/A,
\quad p=0,\ldots,A-1.
\]

Anchors and masses depend on tile endpoints and the declared anchor count, not tree midpoints or tree-record count. The tile must be an exact positive contiguous interval partition. Anchor local embeddings come from their half-open containing graph. Exact breakpoint local features therefore follow the same `[left,right)` convention as the ordinary bank. A different convention or moving a breakpoint across a fixed anchor can change local features.

The coordinate layout contains the sorted union of interval endpoints, anchors and requested readouts. Every positive subsegment records its original containing graph. Thus every propagation action uses all intersected graph holds and exact partial durations, in chronological order. Adding another readout can split homogeneous holds numerically but adds no source impulse or nonlinear update to the anchor bank. Predictions are consequently independent of which other readout coordinates are requested in exact arithmetic, with floating-point tolerance tested.

For each layer, branch and latent channel,

\[
K_{\ell,s,h,g}=\lambda_{\ell,s,h} C_{QQ,g,s}
+d_{\ell,s,h}I.
\]

Rates are strictly positive and global across holds and query nodes. Fixed eigenvectors remain valid. Between anchors/readouts, each cached homogeneous action is

\[
V_{g,s}\operatorname{diag}
\bigl(e^{-\Delta(\lambda_{\ell,s,h}\mu_{g,s}+d_{\ell,s,h})}\bigr)V_{g,s}^{\top}.
\]

Actual chronological products are retained across noncommuting holds. The implementation does not replace them by an exponential of the averaged generator, average transitions across scales, or evolve one state under a mixture of operators. Separate branch states are required.

With query anchor embedding U and fixed original reference embedding R, each layer evaluates its bounded shared-across-scale source projections once at each physical anchor:

\[
S_{\ell,s,h,p}
=\tanh q_{\ell,h}(U^\ell_{Q,p})
-\lambda_{\ell,s,h} C_{QR,g(y_p),s}
\tanh r_{\ell,h}(U^0_{R,p}).
\]

There is no source or nonlinear recomputation when only a tree record is duplicated. Later layers' query sources depend on previous **anchor** context. Reference sources remain clamped observed graph-local embeddings and may vary with the anchor's containing graph.

## Exact finite kernel action and row mass

Let T(x,y) be the forward chronological propagator for x>=y. Define the symmetric matrix-valued space-time kernel

\[
H(x,y)=T(x,y)\ (x\ge y),\qquad
H(x,y)=T(y,x)^\top\ (x<y).
\]

The positive-duration reverse sweep reverses the order of symmetric homogeneous factors and computes the transposed forward product. It never takes inverse flow. At each fixed anchor the inclusive forward state adds its own weighted impulse. For gain-matched diagonal propagation this is the transformed impulse defined below. The reverse state is read before adding that anchor's impulse. The resulting sum counts the self term exactly once:

\[
a_p=T(y_p,y_{p-1})a_{p-1}+\omega_p S_p,
\qquad
b_p=T(y_{p+1},y_p)^\top(b_{p+1}+\omega_{p+1}S_{p+1}).
\]

With global nonnegative `softmax` branch weights pi, the numerator is

\[
M_{\ell,h}(x)=\sum_s\pi_{\ell,s,h}\sum_p\omega_p
H_{\ell,s,h}(x,y_p)S_{\ell,s,h,p}.
\]

The primary normalization uses the corresponding **query and reference** ones source,

\[
m_{\ell,s,h,p}=\mathbf1_Q-
\lambda_{\ell,s,h}C_{QR,g(y_p),s}\mathbf1_R,
\quad
D_{\ell,h}(x)=\sum_s\pi_{\ell,s,h}\sum_p\omega_p
H_{\ell,s,h}(x,y_p)m_{\ell,s,h,p}.
\]

Numerator and mass are bundled into the same sweeps. The gain-matched control transforms both by the same positive cached equilibrium map before propagation, as detailed below. Nonpositive C_QR gives m>=1 in exact arithmetic. The anchor self term gives a positive omega lower bound there. Non-anchor readouts have no self impulse, so that omega lower bound does not apply. The code explicitly rejects nonfinite/nonpositive mass, including possible numerical underflow. It adds no denominator floor, rate clamp, spectral clip or silent repair. Rates and numerical decay must also remain finite and positive. The message passed to the nonlinear layer is elementwise M/D.

Because the source projections use tanh, every projected query/reference signal lies in [-1,1]. A conductance query block plus positive damping is a nonsingular M-matrix, so its homogeneous exponential is entrywise nonnegative. Chronological products, positive anchor masses and nonnegative mixtures preserve nonnegativity. The reference coefficients -lambda*C_QR are also nonnegative. The gain-matched diagonal source map D*K_full^-1 is entrywise nonnegative because K_full is a nonsingular M-matrix and D has positive entries, so it preserves the same coefficient interpretation. The denominator is exactly the sum of all corresponding query and reference coefficients. Therefore each normalized message entry is a convex combination of anchor query/reference signals and satisfies |M/D|<=1 in exact arithmetic, even for noncommuting holds and arbitrary finite positive rates. CPU tests check this across all geometry/coupling controls with saturated source projections, varied rates and mixtures, while unnormalized source magnitudes exceed one. This bound applies to the normalized latent message, not to final gated embeddings or logits.

For symmetric positive definite K(x), the complete unnormalized H is PSD even when interval operators do not commute. One proof uses the linear SDE with diffusion sqrt(2K) and initial covariance I: the covariance remains I, and its cross-covariance is H. Global nonnegative mixtures preserve PSD. The tests materialize small complete block kernels and verify their symmetry/PSD. **Row-mass-normalized actions are generally asymmetric and are not PSD Gram matrices.** Scale-specific C_QR sources may also differ, so the implemented numerator is then a sum of kernel actions on different source fields rather than a single mixed kernel acting on a common source. The rectangular reference map is not a PSD Gram matrix. The gain-matched diagonal source map is also an additional source weighting, and the effective composed map H_diagonal(x,y)*D(y)*K_full(y)^-1 must not be presented as a symmetric PSD Gram matrix. The PSD statement applies to its underlying diagonal-propagation H and global nonnegative mixtures of that H. These conditional within-sequence kernels are not GP predictions or similarities between complete graph collections.

## Nonlinear layers and exact decoder

Each layer updates only the fixed query anchor embeddings:

\[
U^{\ell+1}_{Q,p}=\operatorname{LayerNorm}\left[
U^\ell_{Q,p}+\operatorname{sigmoid}
G_\ell[U^\ell_{Q,p},M^\ell_p/D^\ell_p]
\odot W_\ell(M^\ell_p/D^\ell_p)\right].
\]

The same layer's anchor-source kernel action is evaluated at every requested physical readout coordinate, using exact partial hold products. Its readout embedding starts from the independently encoded containing-tree embedding and receives the same gate/update in each layer. Only the anchors become later-layer source nodes. A depth-two model therefore has exactly two kernel layers, with neither a hidden third kernel layer nor nearest-anchor interpolation. The final linear head reads the last-layer readout embeddings and returns `[P,Q,num_classes]`.

`encode_context(prepared)` exposes final anchors/readouts and per-layer normalized messages, numerator, denominator and source banks for mathematical inspection. `anchor_kernel_action` is the standalone finite-action primitive. It takes sources `[A,S,Q,H,V]`, where V=2 can bundle numerator/mass, and returns `[M,S,Q,H,V]` at unique anchor/readout coordinates. Ordinary forward computation does not materialize a dense anchor-by-query Gram matrix. Spectral decay/factors are batched before sequential sweeps, with no tensor-to-Python validation inside individual homogeneous hold actions.

## Matched controls and rates

The model constructor is:

```python
AnchorKernelModel(
    input_dim=47, num_classes=4, hidden_dim=32, num_layers=2,
    latent_dim=8, kernel_depth=2, num_scales=1,
    geometry="plain", coupling="full", num_anchors=32,
    initial_coupling=20., initial_damping=1.,
    rate_parameterization="log",
)
```

| Choice | Meaning |
|---|---|
| `geometry="plain"` | Repeat the ordinary plain Q/R operators across the same S branches |
| `geometry="age"` | Supplied fixed age operators under their shared plain scale |
| `geometry="matched_scalar_age"` | Supplied age-mean-diagonal-matched scalar copies of plain relative geometry |
| `coupling="full"` | Coupled query propagation under lambda*Cqq+d*I |
| `coupling="scalar"` | Ordinary scalar temporal family, with Cqq replaced by I and K=(lambda+d)*I |
| `coupling="diagonal"` | Independent query propagation with D=lambda*diag(Cqq)+d*I and original sources |
| `coupling="gain_matched_diagonal"` | Transform numerator and ones-mass impulses by D*K_full^-1, then propagate independently under D |
| `kernel_depth=1` or `2` | One or two nonlinear anchor/context layers with the same anchor measure |
| `rate_parameterization="log"` or `"softplus"` | Matched initial positive rates with different optimization parameterization |

The scalar independent-node time kernel retains the exact same source GNN, C_QR reference source map, q/r projections, gates, layer schema and reference-aware ones denominator. It removes spatial state propagation, while retaining graph dependence in the shared source encoder and reference forcing. It changes self decay and equilibrium gain as well as spatial transfer, so it is an ordinary scalar temporal family control rather than an isolated cross-query-memory mechanism control. It is not a removal of every cross-query or geographic information route.

The raw diagonal control keeps each query's original self degree in D=lambda*diag(C_QQ)+d*I and removes off-diagonal state propagation. It keeps the original source and reference-aware mass impulses. This also changes gain and modal decay relative to the full system.

The gain-matched diagonal control uses that same independent D, but at each anchor it applies the local fixed-operator source transform

\[
B_p=D_{g(y_p)}K_{full,g(y_p)}^{-1},\qquad
S'_p=B_p S_p,\qquad m'_p=B_p m_p.
\]

`gain_matched_anchor_sources` applies the cached full query eigenpairs separately per branch/channel, without an online solve or eigendecomposition. The production primitive bundles numerator and ones mass so that both receive exactly the same transform. References remain clamped observed sources. Propagation between anchors uses diagonal factors and contains no cross-query prior-state transfer. The B_p map deliberately retains instantaneous cross-query source mixing.

For one homogeneous operator and continuous constant forcing, D^-1 S'=K_full^-1 S, and the same equality holds for the mass source. Equivalently, the unnormalized infinite-domain integrated impulse gains match: the full two-sided integral is 2*K_full^-1 S and the diagonal integral after transformation is 2*D^-1 B S. This is the quantity called gain-matched. A finite anchor measure does **not** make the pointwise messages equal. At an anchor's self block the full impulse is omega*S, while the gain-matched impulse is omega*B*S, which generally differs. Changing operators also provides no single global equilibrium to match. The tests verify both local homogeneous equilibrium equality and unequal finite-anchor self impulses. Do not attribute finite-anchor differences solely to removal of memory exchange.

The positivity of B_p preserves the normalized convex-combination bound when the denominator receives the same transform. It does not imply that the effective transformed-source kernel is a PSD Gram matrix.

For fixed S/depth/widths, all geometry and coupling choices have identical trainable parameter schemas and counts. The source maps are shared across branches within each layer. Default width32/source-depth2/latent8/class4 counts are 10,324 for S1/depth2, 8,188 for S3/depth1, and 10,420 for S3/depth2. Depth comparisons change capacity and require a separate matched local/source-capacity control for attribution. S independent branches also add hidden memory capacity even when their output channel width is held fixed.

Initial rates accept one positive scalar or an S-length list. For the planned S3 comparison, use the same predeclared `[5,20,80]` coupling schedule in plain, age and matched-scalar-age branches. Identical repeated-plain initial branches can retain a symmetry that makes a multibranch control unnecessarily weak. No distinct initialization is silently inferred from operator labels.

Log rates use lambda=lambda_init*exp(raw), with raw initially zero, and likewise for damping. The calibrated softplus control uses

\[
\lambda=\lambda_{init}\,
\frac{\operatorname{softplus}(\operatorname{invsoftplus}(\lambda_{init})+raw)}
{\operatorname{softplus}(\operatorname{invsoftplus}(\lambda_{init}))}.
\]

It is an additive softplus shift with explicit finite-precision calibration. Both modes have zero-initialized raw parameters, the same buffers/schema, and exactly equal initial rates and outputs after dtype conversion. The parameterization changes optimizer behavior, not initial forward behavior or expressiveness. This implementation supplies no optimizer or convergence claim. `model.rate_and_mixture_parameters()` yields coupling, damping and mixture parameters for exclusion from weight decay in both matched optimization modes. The configuration manifest records current per-layer `[S,H]` rates, damping and mixture weights, plus raw coupling/damping parameters and mixture logits. The integration owner should additionally record rate/damping/mixture gradient norms, relative rate movement and modal decay during training.

Example:

```python
prepared = prepare_anchor_kernel_bank(
    age_bank.plain_bank, age_bank.operator_cache, num_anchors=32,
).to(device, dtype)
model = AnchorKernelModel(
    num_scales=3, geometry="plain", kernel_depth=2,
    initial_coupling=[5.,20.,80.], rate_parameterization="log",
).to(device=device, dtype=dtype)
logits = model(prepared)
```

S1 plain is the default prototype until the architecture-owned label-free operator diversity/coverage audit establishes whether age channels warrant assessment. No label-dependent operator selection is implemented.

## Contracts, limits and verification

In exact arithmetic, unchanged fixed anchors, tile endpoints, interface, Q/R partition, observed selected features and operators give unchanged anchor/target messages after identical interval subdivision. Homogeneous factors compose by the semigroup law, and no record-based nonlinear update is inserted. Serial unary insertion preserves the static boundary operators under the primitive's resistance/feature contract. Integrated age resistance preserves the same contract by telescoping. Induction through the nonlinear anchor layers then preserves later sources, gates and readouts. CPU fixtures verify both contracts through depth two.

This is an **exact finite anchor-kernel action**, not exact integration of a continuously varying nonlinear source. Anchor masses are a predeclared discrete measure. Changing A, anchors, tile endpoints, sample set, source roles or age geometry changes the model. No anchor-refinement invariance, arbitrary age-perturbation robustness, full-tree dynamical projection, novelty or predictive advantage is claimed. Inferred age/topology error remains an empirical robustness question. Fixed geometry/depth/rate choices need factorial controls, and the current negative coupled-flow pilot remains negative evidence.

Verification command:

```bash
. scripts/project_env.sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$ARGGNN_PYTHON" -m pytest -q tests/test_anchor_kernel.py
```

**79 tests passed in 4.65 seconds**, with no warnings. The tests cover dense SciPy `expm` chronological products on explicitly noncommuting operators, scalar analytic kernels, raw/gain-matched diagonal dense oracles, local homogeneous gain equality with unequal finite-anchor self impulses, reverse orientation, full block PSD/global mixtures, asymmetric row normalization, self-once impulses, reference-aware mass, bounded normalized messages with saturated sources and varied rates, every geometry/coupling/depth shape and gradient, equal parameter schemas, equal log/softplus initial outputs in every coupling control, current/raw rate/damping/mixture manifest values, observed47/default32-anchor dimensions, actual age-cache interoperability, unary/interval/serialization contracts, node-storage/numerical-ID/query-order equivariance, extra/duplicated/permuted readout independence, empty R/zero response, no online eigensolve or dense matrix exponential, age-cache contract rejection, and explicit mass-underflow/rate-overflow rejection. The dense SciPy oracle avoids the inspected PyTorch2.6 float64 small-norm `matrix_exp` approximation discrepancy, while production actions use the fixed cached spectral exponential.

Source SHA256 at freeze: `2b8cc1c1726f746f961c0d63a4870e806a1a9d3d6e1fa6724dc0bda39967b083`.
Test SHA256 at freeze: `092c5d99c299bee4fed44abe3e80bf096b6696784a88e427d704669b347cd9fd`.
