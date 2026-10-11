# H033 inherited metadata interpretation

The bank's `kernel_subdivision_invariant=True` flag is inherited from H030. It describes the reused additive-age affinity kernel, whose affinities multiply across serial edge subdivisions. It must not be interpreted as serial-subdivision invariance of H033's active operator.

H033 solves `K z = diag(m) r + S^T diag(tau) x`, with `K=0.01 I + lambda_T L + diag(m) + S^T diag(tau) S`. Its Laplacian uses the mean of the two fixed family affinities divided by the maximum endpoint degree. Inserting an interior vertex adds a 0.01 absorption term, and the resulting degree-normalized conductances need not produce the same harmonic transfer. H033 therefore makes no serial-subdivision invariance claim. On the fixed prepared geometry, K is symmetric positive definite with the stated spectral bounds; both arms use exactly that same geometry and operator.

This note clarifies inherited metadata only. It changes no running scientific source, configuration, prepared geometry, model state, or execution.
