# Exhaustive inferred age-panel geometry audit

Completed 2026-10-10. This label-free audit replaces the earlier first-five-graph
sample with an exhaustive summary of the sealed inferred cohort64 age panel.
It measures operator geometry, numerical quality, observed horizon components,
and fixed-anchor source sampling. It does not establish ancestry agreement,
inferred biological correctness, or predictive benefit.

## Scope and aggregation

The verified panel contains **20 parents (16 train, 4 validation), 160 exact
clipped tiles, 3,719 original observed trees, and 3,859 clipped graph holds**.
Every parent has **Q64/R24/B88**. The extra 140 holds are original trees split
at physical tile boundaries. Each tile uses 32 physical midpoint anchors,
giving 256 anchors per parent.

Both load_boundary_dataset and load_age_caches verified source identities,
frozen implementation dependencies, preparation settings, partitions, sidecar
hashes, byte counts, and plain-bank binding before calculations. Mandatory
loader checks read external supervision for integrity verification. The audit
calculations use only fixed operator caches, observed topology/ages/reference
roles, and physical coordinates. Labels, baseline probabilities, oracle trees,
population truth, and query-label agreement do not enter the metrics.

Graph metric m is averaged within a parent as sum(width * m) / sum(width).
The denominator is that parent's observed physical span, including exact clipped
end holds. Parent means are then averaged equally. Tables separate train and
validation parents; tiles and graph records are not treated as independent
replicates. The JSON also retains every parent mean/range, tile coverage and
original graph mapping, graph extrema, thresholds, and a pooled descriptive
summary. This is an exhaustive description of this finite pilot panel.

## Non-scalar geometry and attenuation

For each raw full-boundary response, the best scalar is
beta = <Cs,C0>_F / ||C0||_F^2. The best-scalar departure is
||Cs - beta*C0||_F / ||Cs||_F.

The mean-matched full-boundary comparison uses alpha*C0, where alpha is the
cached age/plain full-boundary mean-diagonal ratio. The direct cached scalar
comparison concatenates Cqq and Cqr into a Q-by-B query-row matrix and compares
it with the concatenated scalar_cqq/scalar_cqr cache. This checks the actual
matched_scalar_age consumer blocks as well as the full boundary construction.
All operators share the ordinary plain normalization.

Values below are equal-parent means of physical-span-weighted parent metrics.
Bracketed ranges are **parent means**, not individual graph extrema.

| Split | Age scale | Best-scalar full-B departure [parent range] | Mean-matched full-B departure | Direct cached scalar query-row departure | Mean-diagonal attenuation |
|---|---:|---:|---:|---:|---:|
| Train, 16 parents | 350 | 0.283298 [0.226263–0.307821] | 0.323291 | 0.351091 | 0.540561 |
| Train, 16 parents | 1400 | 0.082279 [0.050426–0.099930] | 0.108712 | 0.126365 | 0.853833 |
| Validation, 4 parents | 350 | 0.291471 [0.284681–0.304802] | 0.339247 | 0.347477 | 0.545024 |
| Validation, 4 parents | 1400 | 0.090593 [0.087844–0.092827] | 0.118047 | 0.127845 | 0.852472 |

The age responses have measurable relative geometry beyond their best scalar
plain approximation across this panel. This resolves the limited geometric
question raised by the earlier adjacent-graph sample. Whether that geometry
helps prediction remains an independent factorial-comparison question.

## Spectral and Schur quality

Fixed query eigenpairs were reconstructed against the cached Cqq matrices
afresh, including the matched scalar caches. Full-boundary raw minimum
eigenvalues and interior condition numbers below are retained preparation-time
Schur diagnostics, rather than newly fitted quantities.

| Operator | Minimum query eigenvalue over all holds | Maximum query reconstruction relative error | Maximum matched-scalar reconstruction relative error | Maximum interior condition | Minimum raw full-B response eigenvalue |
|---|---:|---:|---:|---:|---:|
| Plain | 5.45644e-4 | 4.12402e-15 | 4.12401e-15 | 11,347.5 | -9.62308e-14 |
| Age 350 | 1.36588e-4 | 3.99752e-15 | 4.17732e-15 | 22,352.6 | -2.92166e-14 |
| Age 1400 | 1.37886e-4 | 4.06397e-15 | 4.11172e-15 | 40,254.8 | -3.01553e-14 |

All query minima are positive. The tiny signed raw-response values occur around
the full-boundary constant null mode and are preserved as numerical diagnostics.
Maximum interior solve relative residual was 6.24580e-17, and maximum query-basis
orthogonality error was 5.88419e-15. No clipping, jitter, epsilon normalization,
or learned-matrix eigendecomposition was introduced by the audit.

## Anchor32 source coverage

Coverage uses the actual consumer's fixed physical midpoint layout. A graph
hold is touched if it contains at least one anchor. Covered physical span is
the union of complete touched holds, not merely the point measure of the anchors.
Original-tree coverage maps each clipped hold back to its observed parent tree
and counts that tree once.

| Split | Touched clipped records [parent range] | Covered clipped span [parent range] | Touched original records | Covered original span |
|---|---:|---:|---:|---:|
| Train | 82.430% [77.273–87.261%] | 94.618% [93.246–96.845%] | 83.866% | 94.969% |
| Validation | 81.780% [79.000–85.149%] | 94.072% [92.741–94.973%] | 82.672% | 94.254% |

About 5–6% of physical span is in clipped holds with no anchor source evaluation
at this resolution. **Every hold still participates in chronological generator
propagation.** These fractions measure source sampling coverage, not omitted
operator propagation, integration error, or predictive loss. They do not
establish an anchor-refinement contract.

## Strict observed horizon components

The component diagnostic uses decoded observed age strictly below 350 or 600.
Edges connect only if both endpoints are below the horizon. An exact horizon
vertex cannot merge below-cut children, and crossing edges terminate at distinct
unlabeled virtual roots. Reference roles are the observed one-hot input fields.
A conflicting component contains references with more than one observed role;
no query ancestry label is consulted.

Percentages are again equal-parent means after physical-span weighting.

| Split | Horizon | Queries in multiquery components | Queries grounded by any reference | Queries exposed to conflicting reference roles | Singleton references |
|---|---:|---:|---:|---:|---:|
| Train | 350 | 22.561% | 1.244% | 0.00298% | 94.302% |
| Validation | 350 | 22.967% | 1.472% | 0.00604% | 95.546% |
| Train | 600 | 43.596% | 14.989% | 0.74563% | 57.205% |
| Validation | 600 | 42.953% | 14.904% | 1.08665% | 57.558% |

No references are older than or exactly on these horizons in the observed panel.
The JSON additionally records nonsingle sample coverage, query-component size,
all conflicting-reference component counts, and parent/graph ranges.
Reference grounding remains sparse at 350. Coverage expands at 600, accompanied
by more mixed observed reference roles. These are inferred-input connectivity
observations, not biological agreement guarantees or inferred error rates.

## Reproduction and provenance

Run on anogena inside the repository, using the existing project environment:

~~~sh
source scripts/project_env.sh
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
"$ARGGNN_PYTHON" scripts/audit_age_panel.py --self-test
"$ARGGNN_PYTHON" scripts/audit_age_panel.py --output NEW_REPO_OUTPUT.json
~~~

The script binds the version-1 manifests explicitly and refuses an existing
output. Self-checks cover physical-span weighting, a known scalar residual,
spectral reconstruction, and strict horizon ties. Separate report checks
confirmed parent aggregation, tile/span accounting, Q/R counts, plain-control
zero departure, scalar-fit inequalities, positive query spectra, source hashes,
and valid coverage bounds.

The CPU run used four threads and completed in **42.298 seconds**, including
verified loading. The completed raw report is
research/reviews/age_panel_geometry_v1.json (668,037 bytes). The fresh run log
is .runtime/logs/age_panel_geometry_v1.log, PID 636423.

| Binding | SHA256 |
|---|---|
| Audit script | 3cc57dcb1493c90807a0ce6476771cf3d0843da7abad860d02d115f98c52c82b |
| Raw audit report | 25686672572877ae00fbdd567cb13ce058a7896d79321202795e5e929ebfb0c7 |
| Run log | b81d18f9b459aff5bd25f9b8a0c5bdb7db79890500d931f21b2dab433157cd70 |
| Age manifest | 23a0903548c09f0a8f803e67140815c2497b487f25470745cad1e8e0b8b04871 |
| Age preparation | ec50f55989271ee93a6fe3174ecd90e4899c82de5cdac2f55d6ae87f8b0a71d7 |
| Boundary manifest | 7f25874806535897d819b99d20c8f0da1b47f4fe10dde04280e7c42f6e1b9e09 |
| Boundary preparation | c1749a89c2776debadb16a814516a1998057f380fada0c9efe8acafaed284642 |
| Observed inferred source manifest | 611c7e32973c22bd35115b835279884f7b0c866136e91ae2c5f2a95daefd6fe7 |
| Observed inferred source protocol | 66c6bb30191013d5c2a16a40ca870d313c858eb5f05cb3afc9aa88ed50514cb5 |

All direct/dependency source hashes, environment versions, each parent's
prepared/raw observed tensor and adapter hashes, and exact tile mappings are
retained in the JSON. Source and sealed manifest hashes were checked again
before exclusive output creation. No GPU job, model fitting, package install,
Git operation, shared edit, or prediction scoring was performed.
