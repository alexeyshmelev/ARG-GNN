# Independent graph Gaussian-process comparator

This comparator supplies an independently implemented covariance-model family
for the fixed-topology Chickenpox pilot. It is a stationary spectral
graph Ornstein–Uhlenbeck (OU) Gaussian process, inspired by the graph-SPDE
framework of Nikitin, John, Solin and Kaski (2022). It is **not an exact
reproduction** of their stochastic heat equation kernel (SHEK), source code,
or experiments. No accuracy result or comparative claim is established by
this implementation document.

## Source and attribution

Primary reference: Alexander V. Nikitin, St John, Arno Solin and Samuel Kaski.
*Non-separable Spatio-temporal Graph Kernels via SPDEs*. AISTATS 2022,
PMLR 151:10640–10660.

- Published record: https://proceedings.mlr.press/v151/nikitin22a.html
- Published PDF: https://proceedings.mlr.press/v151/nikitin22a/nikitin22a.pdf
- Inspected manuscript: https://arxiv.org/html/2111.08524v3
- Author software identified in that manuscript:
  https://github.com/AaltoPML/spatiotemporal-graph-kernels

The inspected arXiv version is v3, dated 27 December 2024, so it is not
assumed identical to the 2022 published PDF. Sections 4.1 and 4.4 motivate
Laplacian spectral covariances and heat-type graph SDEs, including a
matrix-scaled driving-noise extension. Section 5 says that the paper uses
Gaussian likelihoods and marginal-likelihood hyperparameter optimization
with L-BFGS in GPflow. Sections 5.3 and E.2 discuss Hungarian Chickenpox
and sliding-window experiments, including four- and six-week extrapolation.
The official repository was identified but its implementation was not
copied or independently verified for this comparator.

This implementation follows the *family-level* ideas of Laplacian spectral
covariance, graph-dependent temporal decay, Gaussian regression and
marginal-likelihood fitting. The specific stationary covariance below,
its normalization, parameterization, bounds, optimizer initialization,
analytic gradients, exact mode-wise filtering, and current forecasting
protocol are independent choices. Paper experiments, kernels or numerical
results must not be presented as reproduced.

Source retrieval URLs, byte lengths, content hashes and the inspection date
are recorded in [graph_gp_provenance.json](graph_gp_provenance.json).

## Model definition

Let A be binary undirected county adjacency after omitting self loops, and
L = D − A = U diag(ell_m) U^T be its combinatorial Laplacian decomposition.
The covariance uses the geographic graph in edge-ID county order, not the
CSV column order. Duplicate edge records collapse to one adjacency entry.
Asymmetric edge sets are rejected. Supplied self loops would cancel in
D − A and their omission does not change this Laplacian.

Every county is standardized using its training-only empirical mean and
standard deviation. The latent standardized process has zero mean.
There is no extra fitted intercept, calendar covariate, trend or seasonal
term. For the primary nonseparable variant, define

\[
w_m = (\rho+\ell_m)^{-p},\qquad
q_m = s^2\,\frac{w_m}{N^{-1}\sum_j w_j},\qquad
\alpha_m = \delta+\lambda\ell_m.
\]

The six positive hyperparameters are latent signal variance s²,
observation noise variance r, baseline decay δ, graph diffusion λ, spatial
shift ρ and spatial power p. The normalized q_m have mean s², so the
average node latent variance is s². Individual node variances need not be
equal.

Independent latent modes z_m(t) satisfy the stationary SDE

\[
dz_m(t)=-\alpha_m z_m(t)\,dt+\sqrt{2\alpha_m q_m}\,dW_m(t),
\qquad z_m(0)\sim\mathcal N(0,q_m).
\]

Thus, with x(t) = U z(t),

\[
K((i,t),(j,u))
=\sum_m U_{im}U_{jm}\,q_m e^{-\alpha_m|t-u|}.
\]

Observation noise is independent Gaussian with variance r in standardized
node space. U is orthogonal and scalar node noise remains scalar,
independent noise in mode space. This property is essential to the exact
factorization. Missing observations, unequal node noise, nonorthogonal
directed-graph bases and changing graph topology are not supported.

Each q_m is positive, and each exponential time kernel is positive
semidefinite. Their nonnegative spectral sum gives a valid covariance.
For λ>0 and differing Laplacian eigenvalues, temporal decay depends on the
spatial mode, so the kernel generally does not factor into one spatial
kernel and one common time kernel.

In node space this is a stable heat-type graph SDE with generator
δI + λL and driving-noise covariance U diag(2α_m q_m) U^T. The positive
shift δ makes stationary initialization well-defined even for the constant
Laplacian mode. The paper's SHEK includes finite-start-time accumulation
and uses a fractional Laplacian for the drift. Here the drift is shifted
linear in L, the stationary spatial covariance has independently fitted
shift and power, and the driving noise is correspondingly matrix-scaled.
These are material adaptation differences.

## Matched variants

All variants use exactly the same scaler, observed history, one-week
forecast horizon, Gaussian likelihood, optimizer budget policy, raw-unit
readout and absence of clipping.

| Variant | Latent covariance | Active fitted parameters |
|---|---|---|
| nonseparable | Mode-dependent rate δ + λ ell_m and spatial q_m above | s², r, δ, λ, ρ, p (6) |
| separable | Same spatial q_m, common rate δ | s², r, δ, ρ, p (5) |
| temporal | q_m = s², common rate δ, identity node basis | s², r, δ (3) |

The graph-free variant comprises independent identically parameterized
temporal OU GPs per standardized county. Counties share hyperparameters,
but their latent states and observation noise are independent. This
controls access to adjacency, rather than supplying a separately tuned
GP per county. Parameter objects retain defaults for inactive fields,
which do not affect the selected variant. Metadata lists active fields
and their correct active count.

The separable temporal kernel is exponential, not the paper's RBF time
kernel. These variants isolate graph dependence and mode-dependent
decay within this independent family. They do not establish reproduction
of the paper's separable controls.

## Training and chronological prediction contract

The protocol is read from the root-owned frozen runner configuration.
For the current Chickenpox pilot it specifies 522 raw observations,
history 52, train_end 365 and validation_end 443.

1. The scaler is fitted on raw weeks [0,365), per county. Any supplied
   scaler must exactly match these moments and its declared fit boundary.
2. GP covariance hyperparameters are fitted by full Gaussian marginal
   likelihood on those same 365 standardized raw training weeks. Initial
   weeks [0,52) are training observations, not evaluation targets.
3. The optimizer and its restart selection receive no validation or test
   labels. Candidate selection minimizes training negative log marginal
   likelihood. The likelihood is in standardized units, with no raw-unit
   Jacobian term. That Jacobian is constant with respect to GP parameters.
4. For a forecast target t, callers supply only raw counts[t−52:t]. Each
   call initializes a fresh stationary prior, conditions on exactly these
   52 observations and predicts one week beyond the last observation.
   No stored state carries information from earlier calls or weeks.
5. Validation targets [365,443) may use observed earlier validation weeks
   in their rolling histories, as required by the existing one-week
   protocol. Hyperparameters and scaler remain fixed.
6. Predictions are transformed back to raw case units and are never
   clipped. The model may therefore predict negative counts.

The module has no split-scoring routine. It does not inspect
post-training values during fitting and does not perform final-test
scoring. Test assessment belongs to a separately authorized frozen
evaluation run, not this implementation.

Training moments themselves are estimated from all training weeks,
including weeks later than an early training forecast. The implemented
method is intended for holdout forecasts after train_end. Using it to
claim causal in-sample training backtests would require a separate
expanding-window refit protocol.

## Exact and efficient calculation

For unit weekly spacing, the mode transition is

\[
\phi_m=e^{-\alpha_m},\qquad
Q_m=q_m(1-e^{-2\alpha_m}).
\]

Starting with mean zero and variance q_m, scalar Kalman innovations
provide the exact GP marginal likelihood. The forecast mean and
covariance are the exact Gaussian conditional given the supplied window.
The implementation uses expm1 for Q_m to avoid cancellation at small
rates. It propagates analytic derivatives of the innovations likelihood
with respect to log hyperparameters. This is an exact state-space
representation of the specified covariance, with no rank truncation,
variational approximation, inducing points or Monte Carlo inference.

For N counties, T training weeks, P active parameters and E optimizer
evaluations, fitting costs O(N³ + T N² + E T N P). Training storage is
O(T N + N² + N P). Forecasting costs O(H N² + H N + N³) with H history
weeks. The final O(N³) term forms a complete returned N×N predictive
covariance, using dense matrix multiplication. N=20 and H=52 are small.
The dense covariance function is supplied only as an oracle/debugging
interface and is not called by fitting or prediction.

CPU jobs should set CUDA_VISIBLE_DEVICES empty and cap numerical-library
threads for repeatable resource accounting. No GPU operation is used.

## Optimization policy and diagnostics

Default policy is three starts, seed 101, up to 100 L-BFGS-B iterations per
start. The first start uses (s²,r,δ,λ,ρ,p) = (1,0.1,0.2,0.05,1,1).
Other starts add independent N(0,1) perturbations in active log parameter
space, then clip to bounds. The root-owned runner may override seed,
number of starts and iteration budget before results are observed.
These are optimizer starts, not independent statistical data replicates.

| Parameter | Lower bound | Upper bound |
|---|---:|---:|
| s² | 0.00001 | 100 |
| r | 0.000001 | 100 |
| δ | 0.0001 | 10 |
| λ | 0.000001 | 10 |
| ρ | 0.001 | 1000 |
| p | 0.1 | 4 |

Optimization is performed in log units with ftol 1e−10 and gtol 1e−6.
The minimum training NLL among finite optimized candidates and their
initializations is retained, including a finite initialization if an
optimizer fails to improve it. This does not assert global optimality
or convergence. Metadata records every initial/final objective,
termination status/message, iteration/evaluation count, gradient
magnitude, selected restart/candidate kind, bound contact and fit time.
A raw gradient can remain nonzero at a constrained optimum.

Marginal-likelihood fitting can prefer weak geography or parameters at
bounds. Such outcomes are results to report, not reasons to alter the
search after viewing validation. A parameter at a bound indicates a
limitation of the specified family/bounds, and should be disclosed.
No accuracy target or desired ranking guides optimization.

## API

Implementation: src/arggnn/graph_gp.py.

Example integration, leaving scoring to the root-owned runner:

~~~python
fitted = GraphOUForecaster.fit_training(
    data, protocol, scaler, variant="nonseparable",
    seed=101, restarts=3, maxiter=100,
)
prediction = np.stack([
    fitted.predict(data.counts[t - protocol.history:t])
    for t in protocol.targets("validation")
])
record = fitted.metadata()
~~~

predict_distribution(raw_history) returns raw mean [N] and raw observation
covariance [N,N]. Set observation_noise=False to return latent predictive
covariance. Both condition on noisy historical observations. Raw
covariance rescales both axes by county standard deviations, so scalar
standardized observation noise becomes county-scaled raw noise.
Uncertainty conditions on fitted hyperparameters and training moments.
It omits their estimation uncertainty and carries no calibration claim.

## Verification and scientific limits

tests/test_graph_gp.py verifies all three variants against independently
constructed node-space covariances using fractional matrix powers and
matrix exponentials. It verifies full dense Gaussian likelihoods and
one-step conditional means/covariances, analytic likelihood gradients
against finite differences, positive semidefiniteness, permutation
equivariance, graph-free invariance, scaler leakage rejection, exact
window length, stationary reset, and invariance of fitting to future NaN
values. It also checks the pinned archive's geographic column mapping:
the CSV starts with BUDAPEST, while edge-ID county 0 is BACS and county
4 is BUDAPEST. The archive has 41 distinct undirected nonloop borders,
82 directed nonloop entries, plus 20 loops. This differs from the
paper's text describing 61 edges, and is not silently equated.

The implementation qualification on 10 October 2026 used NumPy 2.4.6
and SciPy 1.17.1 with CPU numerical-library thread counts set to one.
Its 27 tests passed in 3.38 seconds, excluding SSH startup. This is
numerical/integrity qualification, not forecast-accuracy evidence.
Wall times for full training fits and rolling forecasts must be measured
by the actual root-owned run and retained in run artifacts.

This is a useful independent GP-family comparator, but it is deliberately
limited. Gaussian observation noise approximates nonnegative count data.
The model is stationary, has only OU temporal dependence, lacks annual
periodicity and nonstationary epidemic dynamics, and assumes complete
observations on a fixed graph. It is not a seasonal GP, a wave kernel,
a changing-topology GP or evidence of epidemiological causation.
Its predictive uncertainty must be empirically assessed before a
calibration claim. It cannot support a claim of reproducing Nikitin
et al., surpassing published state of the art, or demonstrating that
nonseparability is generally preferable. Comparisons must identify the
independent adaptation and the exact shared pilot protocol.

