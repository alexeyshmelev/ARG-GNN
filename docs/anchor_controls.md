# Fixed-anchor capacity and source-feedback controls

Status: CPU-verified attribution controls. No training, score, GPU job, installation, Git operation or shared implementation edit was performed. These classes live in the new `anchor_controls.py` module and import the frozen `AnchorKernelModel`. The running job's `anchor_kernel.py`, runner, configurations, geometry/cache preparation and dependencies are unchanged.

## Common API and parameter contract

Both controls inherit the candidate's constructor, layer modules, rate parameterization and `forward(prepared)` interface. They use the same `AnchorKernelBank` prepared by `prepare_anchor_kernel_bank` and return `[P,Q,num_classes]` in original external scoring/query order.

```python
from arggnn.anchor_controls import (
    NoncontextualAnchorKernelModel, StaticLocalAnchorModel,
)

model = NoncontextualAnchorKernelModel(
    input_dim=47, num_classes=4, hidden_dim=32, num_layers=2,
    latent_dim=8, kernel_depth=2, num_scales=3, geometry="plain",
    coupling="full", num_anchors=32, initial_coupling=[5.,20.,80.],
    initial_damping=1., rate_parameterization="log",
)
logits = model(prepared)
```

Geometry choices remain `plain`, `age` and `matched_scalar_age`. The shared source GNN always uses the original independent **plain** full-boundary response and the same observed selected leaf fields. The ancestry feature contract is unchanged: 47 observed fields with raw parent-branch omitted. Each layer retains the candidate's q/r projections, message projection, gate, normalization, global coupling/damping and branch-mixture parameters. Both log and calibrated softplus rates are supported. References remain original independently encoded observed graph-local embeddings.

The controls add no learned fields. Their `state_dict` keys and trainable tensor shapes exactly match the candidate for fixed width, source depth, kernel depth and branch count. Thus candidate/noncontextual/static parameter counts are respectively identical: S1/depth2 **10,324**, S3/depth1 **8,188**, S3/depth2 **10,420** for the default dimensions above. Empty-reference projections remain in the common schema and are disclosed as inactive, as in the candidate.

The inherited `rate_and_mixture_parameters()` method supports the same exclusion from weight decay. Control manifests preserve initial/current rates, damping, mixtures and raw parameters. They add a control interpretation and `controls_implementation_sha256`, while retaining the frozen core's `implementation_sha256` and configuration snapshot. No optimizer, runner or configuration is supplied by this module.

## Noncontextual query-source depth control

`NoncontextualAnchorKernelModel` keeps every layer's chronological kernels, clamped R source maps, reference-inclusive ones normalization, fixed anchor grid and exact target-coordinate kernel readouts. It supports the candidate's `full`, `scalar`, `diagonal` and `gain_matched_diagonal` coupling choices.

Let U^0_Q(y_p) be the original independently encoded query embedding at a fixed anchor. Every layer forms its query source from **that original embedding**:

\[
S_{\ell,s,h,p}=\tanh q_{\ell,h}(U^0_Q(y_p))
-\lambda_{\ell,s,h}C_{QR,g(y_p),s}
\tanh r_{\ell,h}(U^0_R(y_p)).
\]

The candidate instead projects its current contextual anchor embedding U^ell_Q(y_p). The control's per-layer q/r maps are still different learned maps. Sources can therefore differ across layers, but they cannot depend on earlier anchor messages. Each layer retains its exact kernel action and matching ones-source denominator, including the gain-matched numerator/mass source transform where selected.

Target readout embeddings still receive the same gated residual update at every depth and retain their previous decoder context. This preserves nonlinear decoder depth while removing contextual query-source feedback into the global anchor bank. Anchor residual updates remain as an audit output from `encode_context`; those updated anchors are never supplied to subsequent source projections in this control. No additional parameters are attached merely to match a count.

The control still uses temporal/global kernel context and may rely on chronology. It is not order blind, a local-only control, a guaranteed null, or a predicted performance failure. Its comparison with the candidate addresses source feedback beyond layer-specific global messages and repeated decoder updates.

The intervention test changes the first layer's gate/message-update weights. Later noncontextual source banks remain exactly unchanged, while target residual outputs change. Applying the same intervention to the contextual candidate changes its second-layer source bank. Depth-one noncontextual and contextual models with matched initial weights produce exactly identical outputs.

## Same-capacity static local depth control

`StaticLocalAnchorModel` currently requires `coupling="full"` and rejects other coupling modes explicitly. It uses the same plain/age/matched-scalar-age branch operators, but processes every observed graph independently. No anchor impulse, homogeneous hold, chronological product, graph average or cross-coordinate message is evaluated. The anchor-count field is retained for common configuration and is unused by its computation.

At each graph and depth, query embeddings receive graph-local context. Reference embeddings remain original. Define

\[
F_{\ell,g,s,h}=\tanh q_{\ell,h}(U^\ell_{Q,g})
-\lambda_{\ell,s,h}C_{QR,g,s}
\tanh r_{\ell,h}(U^0_{R,g}),
\qquad
m_{\ell,g,s,h}=\mathbf1_Q-
\lambda_{\ell,s,h}C_{QR,g,s}\mathbf1_R.
\]

For the full fixed query operator K=lambda*C_QQ+d*I, the control applies the cached eigenpairs to **both** source and mass:

\[
N_{\ell,g,h}=\sum_s\pi_{\ell,s,h}K_{\ell,g,s,h}^{-1}F_{\ell,g,s,h},
\qquad
D_{\ell,g,h}=\sum_s\pi_{\ell,s,h}K_{\ell,g,s,h}^{-1}m_{\ell,g,s,h},
\qquad
M_{\ell,g,h}=N_{\ell,g,h}/D_{\ell,g,h}.
\]

`static_local_messages` bundles numerator and mass through the same cached spectral inverse, mixes branches globally and divides elementwise. It performs no online solve, inverse or eigendecomposition. Finite positive rates/decay and finite positive mass are required, with no floor or silent clipping.

The query embedding receives the same learned update as the candidate:

\[
U^{\ell+1}_{Q,g}=\operatorname{LayerNorm}
\left[U^\ell_{Q,g}
+\operatorname{sigmoid}G_\ell[U^\ell_{Q,g},M_{\ell,g}]
\odot W_\ell M_{\ell,g}\right].
\]

Later static sources can depend on this prior **within-graph** query context. Final embeddings are gathered by the original `target_graph` and passed to the same head. Coordinates inside the same graph consequently have exactly the same static prediction. Changing another graph's observed features leaves those predictions unchanged. Shared trained weights do not constitute inference-time cross-coordinate context.

All coupling/damping/mixture parameters remain in the computation, together with the same projections/gates/norms. Nonzero gradients for each rate/mixture parameter are verified on a nondegenerate Q/R S3 fixture. This is not a guarantee of identifiability or nonzero gradients for every input. In a query-only branch, common scalar scaling of K cancels after numerator/mass normalization. S1 mixture logits are necessarily redundant, and empty reference projections are inactive. Matching a parameter count does not eliminate differences in effective capacity, hidden states or computation.

The static K^-1 operator is symmetric PSD and entrywise nonnegative for the positive-damped conductance query block. With bounded q/r and the matching reference-inclusive mass, normalized messages retain the convex-combination bound |M|<=1. Branch-dependent source maps and normalization do not turn the implemented complete normalized action into a symmetric PSD Gram matrix. This control measures graph-local depth/capacity, not a GP or a temporal equilibrium across changing graphs. It retains geographic query mixing and is not a feature-only or independent-node null.

`encode_context` returns `StaticLocalContext(local, targets, layers)`. Each `StaticLocalMessages` contains the normalized graph-local message, numerator, denominator and source/mass banks. These are graph-indexed audit fields rather than fixed-anchor source fields.

## Observation and representation contracts

Both controls consume only the common observed feature bank, fixed geometry, Q/R partition and existing scoring metadata. They consult no external labels. County/sample identities remain correspondence keys, not learned numerical features. The frozen source GNN and static operators preserve the established unary/feature contract. Identical graph-record splitting preserves the noncontextual kernels through homogeneous semigroup composition and preserves the static local outputs because no record-level aggregation occurs. Reordering storage, queries or readouts preserves the corresponding predictions. Adding a readout does not add a source. The static control is also independent of anchor-grid choice because it uses no anchors.

These are conditional attribution controls, not validated scientific gains. They do not establish novelty, a predictive benefit, biological generalization, inferred-tree robustness or a guaranteed null. The noncontextual model still has global context. The static model still has local nonlinear query mixing. Paired settings, seeds, preprocessing, optimization and validation budgets remain the integration owner's responsibility.

## CPU verification snapshot

```bash
. scripts/project_env.sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$ARGGNN_PYTHON" -m pytest -q tests/test_anchor_controls.py
```

**40 tests passed in 3.69 seconds**, with no warnings. Tests reuse the frozen interface fixtures read-only and cover exact schema/count matching, all parameter gradients, all noncontextual coupling modes, source-feedback intervention, depth-one identity, per-layer dense SciPy static solves on both numerator/mass, graph-local independence, same-graph coordinate equality, unary/interval/serialization invariance, numerical-ID/node-storage/query/readout equivariance, readout-set independence, poisoned external labels, no online solve/inverse/eigensolve/matrix exponential, log/softplus initialization equality, bounded messages, unused static anchors, explicit unsupported coupling rejection, empty references and control manifest fields.

Frozen core dependency SHA256 remains `2b8cc1c1726f746f961c0d63a4870e806a1a9d3d6e1fa6724dc0bda39967b083`.
Control source SHA256: `f4939b4515bafb7ba77373980d2be44f7c9be1a5b6881f2166ca0a2d6265063e`.
Control test SHA256: `55f21a8670ee01782bdb33f35ec06752e38c953771a8668e50878771dd3ccb2c`.
