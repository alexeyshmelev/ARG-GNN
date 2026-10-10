# Split age query/reference attribution interventions

Status: CPU-verified operator-block attribution controls. No training, score, GPU job, Git operation, installation, runner/configuration edit or existing-file modification was performed. `age_intervention_controls.py` reuses the frozen `AnchorKernelModel` and its observed feature/cache contract. The peer question is whether any age effect arises through query propagation, reference forcing, or their interaction.

## API and selection

```python
from arggnn.age_intervention_controls import SplitAgeAnchorKernelModel

model = SplitAgeAnchorKernelModel(
    input_dim=47, num_classes=4, hidden_dim=32, num_layers=2,
    latent_dim=8, kernel_depth=2, num_scales=3, geometry="age",
    coupling="full", num_anchors=32, initial_coupling=[5.,20.,80.],
    initial_damping=1., rate_parameterization="log",
    age_intervention="source_only",
)
logits = model(prepared)
```

The prepared `AnchorKernelBank` must contain matched S3 `plain` and `age` operators with the same graph/query/reference order. The default architecture-owned age cache orders branches plain, age350, age1400, under the shared plain full-boundary normalization. `num_scales=3` and `geometry="age"` are required. The class supports `full`, `diagonal` and `gain_matched_diagonal` coupling. Scalar coupling is explicitly excluded from this initial intervention interface.

| Intervention | Query C_QQ and its eigenpairs | Reference C_QR |
|---|---|---|
| `source_only` | Repeated plain | Age |
| `propagation_only` | Age | Repeated plain |

`selected_operators(prepared)` returns an `AnchorOperators` view whose fields alias the chosen fixed cache tensors. `intervention_bank(prepared)` copies only the small operator dictionary and replaces its `age` entry with that view. It shares the original observed bank and coordinate layout. This wrapper is constructed per forward without tensor copying, mutation, geometry preparation or online factorization. Original operators remain intact.

The common independent source GNN, plain normalized full-boundary response, 47 observed branch-excluded fields, Q/R identities, anchor measure, scoring coordinates and all learned modules remain unchanged. No extra reference summary, query target or truth-tree input is introduced. Numerator and ones mass always consult the **same chosen C_QR**.

## Meaning and limits

At each layer and anchor, the raw conditional source and its corresponding mass are

\[
F_{\ell,s,h,p}=\tanh q_{\ell,h}(U^\ell_Q(y_p))
-\lambda_{\ell,s,h}C^{selected}_{QR,g(y_p),s}
\tanh r_{\ell,h}(U^0_R(y_p)),
\]

\[
m_{\ell,s,h,p}=\mathbf1_Q-
\lambda_{\ell,s,h}C^{selected}_{QR,g(y_p),s}\mathbf1_R.
\]

Full query propagation uses

\[
K_{\ell,s,h,g}=\lambda_{\ell,s,h}C^{selected}_{QQ,g,s}+d_{\ell,s,h}I.
\]

The names refer to selection of the operator blocks, not complete isolation of a biological pathway. At nonlinear depth, changing either block can change contextual query embeddings and therefore later learned q sources. Under `gain_matched_diagonal`, the selected query block also determines the instantaneous D*K_full^-1 source/mass transform, in addition to diagonal propagation. A propagation-only gain-matched intervention therefore changes that transform while keeping the raw C_QR map plain. Full coupling is the clearest first comparison for separating these block roles.

The selected QQ/QR pair may **not** be blocks of a single physically reduced full tree. Their joint row sums need not satisfy the original full-boundary Laplacian relation. No mixed full-boundary response is fabricated, no physical current/Dirichlet interpretation is asserted for the joined blocks, and no physical Kron geometry claim is made. These are conditional learned-filter attribution interventions on independently cached observed operators.

The mathematical action remains valid. Selected C_QQ is symmetric PSD with nonpositive off-diagonal entries, and positive damping gives a nonsingular M-matrix generator. Selected C_QR is nonpositive. Thus homogeneous query actions and their chronological products remain entrywise nonnegative. Reference coefficients and anchor/mixture weights are nonnegative. The matching ones denominator gives a convex combination of bounded q/r source signals, preserving |normalized message|<=1 in exact arithmetic. The gain-matched transform is also positive and is applied identically to numerator and mass. Numerical positivity/finiteness checks, without floors or clamps, remain inherited.

The unnormalized query-propagation block kernel remains PSD under the inherited conditions, irrespective of the reference map. This does not make the complete reference-source-weighted or row-normalized message a symmetric PSD Gram matrix. No GP prediction, physical-tree response, full-tree dynamical projection, novelty, null effect or predictive benefit is claimed.

## Matched capacity and representation

No learned parameter is added. The candidate and both split interventions have exactly the same parameter/state schemas and counts at fixed dimensions. Default S3/source-depth2/hidden32/latent8/class4 counts are **8,188** at kernel depth one and **10,420** at depth two. The same log/softplus parameterization and initial branch schedules apply. `rate_and_mixture_parameters()` and initial/current rate/damping/mixture manifest fields are inherited. When age and repeated plain arrays are identical, either intervention gives exactly the original model output with the same weights.

Each selected block inherits its own observed geometry and feature invariance. Consequently serial unary insertion, identical interval subdivision, graph serialization changes and query/readout permutations preserve the split control under the same fixed interface, anchors, tile and deterministic-evaluation contract. Additional readouts create no source anchors. These contracts do not imply anchor-refinement invariance or robustness to inferred-age errors.

The configuration manifest identifies `age_intervention`, the QQ/eigensystem source, the reference forcing and mass source, `physical_full_boundary_Kron_response=False`, the frozen core hash and the new intervention module hash. The original `geometry="age"` constructor field remains the bank selection key, while the explicit operator-status tag describes the mixed attribution intervention. Integration, validation budgets and later experiment scheduling remain root-owned.

## CPU verification snapshot

```bash
. scripts/project_env.sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$ARGGNN_PYTHON" -m pytest -q tests/test_age_intervention_controls.py
```

**43 tests passed in 5.37 seconds**, with no warnings. Frozen fixtures and the independent dense SciPy oracle are reused read-only. Checks cover selected tensor aliases and nonmutation, exact identical-age/plain limiting identity at both depths and all supported couplings, unchanged schemas/counts, finite gradients, every layer's chosen C_QR source/ones mass and selected C_QQ dense action, bounded messages under saturation, unary/interval/serialization invariance, query/readout permutations and added readouts, observed-only poisoned labels, no online eigensolve/solve/inverse/matrix exponential, matched log/softplus initialization, explicit manifest attribution and invalid settings/cache rejection.

Intervention source SHA256: `e0ebce68121af8c9bea250d90e28b40df84747ad25c8ce8f13cb71ddcbc41e28`.
Intervention test SHA256: `efe336f703f6534e2b576a4777f3ab5923829ce348d338f5824eac754a56d4b5`.
Existing `anchor_kernel.py` remains `2b8cc1c1726f746f961c0d63a4870e806a1a9d3d6e1fa6724dc0bda39967b083`.
Existing `anchor_controls.py` remains `f4939b4515bafb7ba77373980d2be44f7c9be1a5b6881f2166ca0a2d6265063e`.
