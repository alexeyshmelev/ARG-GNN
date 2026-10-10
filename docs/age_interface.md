# Fixed observed age-interface bank

Implemented experiment, 2026-10-10. This CPU preparation module follows
`research/reviews/age_conditioned_boundary_design_v1.md`. It establishes
geometry and representation contracts. It does not establish a predictive
advantage or a new mathematical construction.

## Preparation API

```python
from arggnn.age_interface import precompute_age_interfaces

age_bank = precompute_age_interfaces(
    observed_sequence,
    plain_bank=existing_plain_bank,  # optional, checked against observed source
    time_column=6, parent_branch_column=7,
    time_scale=10000.0, resistance_scale=1e-4,
    age_scales=(350.0, 1400.0), strength=3.0,
)
plain_bank = age_bank.plain_bank
age_cache = age_bank.operator_cache
```

Preparation is CPU float64. Encoded ages are first decoded by
`expm1(x[:, time_column] * log1p(time_scale))`. Edge resistance is then

$
r_s(c,p)=10^{-4}[\Phi_s(p)-\Phi_s(c)],
\qquad \Phi_s(t)=4t-3s\arctan(t/s).
$

`age_potential_difference` evaluates the same endpoint difference using
the scaled identity
`atan(p/s)-atan(c/s)=atan2(s*(p-c),s*s+p*c)`, avoiding subtraction of
nearly equal old-age arctangents and overflow in the two products. It uses
no quadrature, learned edge weight, epsilon floor, jitter, or spectral clipping.
Equal-age branches are rejected under the existing explicit positive-resistance
policy. Nonfinite decoded ages or arithmetic are rejected.

The returned frozen `AgeInterfaceBank` contains the ordinary
`LeafInterfaceBank` as `plain_bank` and a separate `AgeInterfaceCache` as
`operator_cache`. An existing plain bank is checked for source geometry,
features, identities, coordinate metadata, physical scales, partition, and
fixed eigensystem before reuse. Its ordinary plain local response and selected
leaf fields remain the local-encoder input. For the prepared F48 cohort input,
the persistent leaf bank has the same 47 fields, excluding parent-branch column 7.

References default to the sorted observed sample complement of the original
query IDs. A supplied plain query cache retains its declared reference order.
An explicit conflicting order is rejected. `age_scales=()` yields S1 plain.
A consumer can also repeat the ordinary plain cache for a capacity-matched
multibranch control, independently of whether age geometry is useful.

## Cache and controls

The scale dimension is axis 1, ordered
`operator_scales=(0.0,350.0,1400.0)`,
`operator_labels=('plain','age350','age1400')` by default.

| Fields | Shape and meaning |
|---|---|
| `raw_response` | [G,S,B,B], separately retained raw boundary responses |
| `normalized_response` | [G,S,B,B], all raw responses divided by the same plain scale |
| `normalization_scale` | [G], ordinary plain full-boundary mean diagonal |
| `cqq`, `cqr` | [G,S,Q,Q] and [G,S,Q,R], unchanged query/reference order |
| `eigenvalues`, `eigenvectors` | [G,S,Q] and [G,S,Q,Q], fixed query-block spectral cache |
| `scalar_cqq`, `scalar_cqr` | Parallel matched scalar-age controls |
| `scalar_eigenvalues`, `scalar_eigenvectors` | Scaled plain eigenvalues and repeated plain basis |
| `mean_diagonal_ratios` | [G,S], age attenuation relative to the raw plain mean diagonal |
| `diagnostics` | [G][S], raw Schur condition/residual/row-sum/PSD audit |
| `spectral_reconstruction_relative`, `spectral_orthogonality_max` | [G,S], numerical fixed-basis checks |

The cache also retains sample IDs, packed boundary nodes, query/reference IDs
and indices, physical interval weights, physical parameters, and schema version.
Its normalization tag is `shared_plain_mean_boundary_diagonal`.

For nonzero plain response,
$
\sigma_g=\operatorname{mean}\operatorname{diag}(C_{g,0}),\qquad
\widehat C_{g,s}=C_{g,s}/\sigma_g,\qquad
\alpha_{g,s}=\operatorname{mean}\operatorname{diag}(C_{g,s})/\sigma_g.
$
The scalar control uses $\alpha_{g,s}\widehat C_{g,0}$. Its full-boundary
mean diagonal matches the age response and its relative geometry remains plain.
The implementation stores its query/reference blocks and spectral action, with
no additional eigensolve. It does not independently normalize each age operator.
For an analytically zero plain response, all age responses are also exactly zero,
the shared scale is one, and the benign scalar ratio convention is one.

Both bank and cache provide `.to(device,dtype)`, `.as_dict()`, and
`.from_dict()`. Tensor transfer preserves original coordinate precision
through `plain_bank.to`. `torch.save(bank.as_dict())` and
`torch.load(...,weights_only=True)` round-trip without custom pickle classes.
The cache can be serialized separately as a sidecar bound to an independently
verified plain bank. Schema, shape, finite-value, device, identity, and
normalization checks reject incompatible serialized caches.

Every eigendecomposition occurs during fixed CPU preparation. The consumer
receives existing matrices and eigenpairs. There is no online eigendecomposition,
learned matrix, learned-matrix cache, or input target field in this module.

## Representation and numerical verification

For strength 3, $1\le\Phi_s'(t)\le4$. Consequently every edge satisfies
$r_0\le r_s\le4r_0$, and the common-boundary Schur responses satisfy
$$
\frac14 C_0\preceq C_s\preceq C_0.
$$
Endpoint potentials telescope under a legal non-sample unary insertion.
Series reduction therefore preserves the response in exact arithmetic.
Identical interval subdivision changes only the record serialization and leaves
each operator and its physical-width average unchanged.

The combined CPU suite passed **89 tests in 8.12 s**:
`test_age_interface.py`, `test_kron_interface.py`,
`test_boundary_dataset.py`, and `test_anchor_kernel.py`.
The age module contributes 15 tests covering potential telescoping and bounds,
close old ages, unary and interval subdivision, node permutation, shared
normalization, matched scalar geometry, eigensystems, Loewner bounds, source
mismatch rejection, S1 and analytically zero geometry, coordinate transfer,
weights-only serialization, and independence from external supervision tensors.
The anchor consumer separately passed real-cache integration and blocked
online-eigensolve checks. These are implementation checks, not model-performance
tests.

## Bounded label-free cohort64 qualification

The complete raw record is
`research/reviews/age_geometry_qualification_v1.json`
(SHA256 `3dc63211eca87a43fc6fb8a7e74d9ffa0947f69e2ef16f9a0a02841935a9b430`).

Only the **first five serialized graphs of first train parent rep000** were
examined for each observed source. The prepared tensor hashes were verified.
All ten graphs were subsequently rebuilt with the implemented module. Each has
**Q64/R24/B88**, 175 packed nodes, and the same 47 persistent leaf fields.
The oracle parent has 1,534 graphs and the inferred parent has 183.

| Observed source | Age scale | Relative departure from best scalar plain response | Shared-scale mean diagonal | Mean reference grounding row sum |
|---|---:|---:|---:|---:|
| Oracle | 350 | 0.114456–0.114511 | 0.509053–0.509738 | 0.051287–0.053139 |
| Oracle | 1400 | 0.022557–0.022702 | 0.862416–0.863334 | 0.095488–0.099293 |
| Inferred | 350 | 0.151619–0.266044 | 0.490872–0.517089 | 0.065789–0.072504 |
| Inferred | 1400 | 0.037664–0.060141 | 0.849607–0.864448 | 0.130136–0.141067 |

Departure is $\|C_s-\beta C_0\|_F/\|C_s\|_F$, with
$\beta=\langle C_s,C_0\rangle_F/\|C_0\|_F^2$. Reference grounding is the
query mean of $-\widehat C_{QR,s}\mathbf{1}_R$. Positive query-block eigenvalue
ranges are recorded per graph in the raw report. Worst fixed-basis reconstruction
relative error was **2.9281e-15**, largest interior condition number was
**13,928.1**, and the smallest normalized Loewner slack eigenvalue was
**-9.7842e-16**, consistent with roundoff near the constant null mode.

Strict observed horizon coverage uses decoded age **<350**. Below-cut edges
connect only when both endpoints are strictly younger. An edge crossing the
horizon terminates at its own unlabeled virtual root and cannot merge components.
This is a connectivity diagnostic, not a learned forest operator.

| Observed source | Queries in multiquery components | Queries in nonsingle sample components | Queries in reference-grounded components |
|---|---:|---:|---:|
| Oracle, each sampled graph | 15/64 (23.44%) | 16/64 (25.00%) | 1/64 (1.56%) |
| Inferred, sampled range | 14–20/64 (21.88–31.25%) | 14–20/64 (21.88–31.25%) | 0–1/64 (0–1.56%) |

No older/equal-horizon references or conflicting observed reference-category
components appeared in these ten graphs. Oracle graphs had 21 singleton
reference components and inferred graphs had 20–22. Full component-size records
are retained in the JSON.

These adjacent graphs from one parent per source are a bounded diagnostic sample,
not a panel estimate. The oracle and inferred serialized intervals are not
coordinate-paired. Their observed geometries show nonzero relative age changes,
especially at scale 350, while hard-horizon reference grounding is sparse.
No labels, truth ancestry, inferred label agreement, accuracy, or predictive gain
were measured. These observations support testing a soft age geometry hypothesis,
not claiming biological correctness for inferred young-coalescence components.

## Provenance snapshot

| Dependency | SHA256 |
|---|---|
| `src/arggnn/age_interface.py` | `15f9615a84a230386d8f71f5a317a46051c395f75408c79a6213f1b07f101f3d` |
| `src/arggnn/kron_interface.py` | `09ec6286c185e7aa2b13d76dcbe8fea24b3a1c785a6619ac9bc5464e0de66d69` |
| `src/arggnn/models.py` | `8429bd02a40950a3dc4d76046ab4b31b366871499d95e295a16ade730f567df5` |
| `src/arggnn/data.py` | `fb4e4b3ad8744731e6d2c5966cc2679f9f2d59dfd079facc280d69fbc61ef654` |
| `tests/test_age_interface.py` | `5e1226b6547b59bb95b344e683cfdd33a3ed19ba504d7877fa0cc4f76b489367` |
| Oracle source manifest | `c9b3af5c1ebbc57a9b04d58902d7d13a12dfd240a81c879deb74f82cd2ad0385` |
| Inferred source manifest | `611c7e32973c22bd35115b835279884f7b0c866136e91ae2c5f2a95daefd6fe7` |

Prepared/raw tensor, protocol, and reference-adapter hashes are in the raw report.
Dataset sidecars must bind this source and its direct computation dependencies
to the previously verified plain preparation and exact observed clipped inputs.
No GPU jobs, package installations, Git changes, or label scoring were performed
by this module task.
