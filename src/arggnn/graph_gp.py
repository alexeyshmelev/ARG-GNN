"""Independent stationary spectral graph-OU Gaussian-process comparator.

Inspired by Nikitin et al. (AISTATS 2022), not a reproduction of their SHEK.
See docs/graph_gp.md for the covariance, adaptations, and evaluation contract.
NumPy/SciPy CPU implementation. This module never evaluates a data split.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import time
from typing import Literal

import numpy as np
from scipy.optimize import minimize

from .chickenpox import ChickenpoxData, ForecastProtocol, TrainingScaler

Variant = Literal["nonseparable", "separable", "temporal"]
VARIANTS = ("nonseparable", "separable", "temporal")
# Positive bounds are specified in natural units, optimized in log units.
PARAMETER_BOUNDS = {
    "signal_variance": (1e-5, 100.0), "noise_variance": (1e-6, 100.0),
    "decay": (1e-4, 10.0), "diffusion": (1e-6, 10.0),
    "spatial_shift": (1e-3, 1000.0), "spatial_power": (0.1, 4.0),
}


@dataclass(frozen=True)
class GraphOUParameters:
    signal_variance: float = 1.0
    noise_variance: float = 0.1
    decay: float = 0.2
    diffusion: float = 0.05
    spatial_shift: float = 1.0
    spatial_power: float = 1.0

    def validate(self) -> None:
        if any(not np.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("Every covariance parameter must be finite and positive")


def _names(variant: Variant) -> tuple[str, ...]:
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    common = ("signal_variance", "noise_variance", "decay")
    if variant == "temporal":
        return common
    return common + (("diffusion",) if variant == "nonseparable" else ()) + (
        "spatial_shift", "spatial_power")


def graph_laplacian(edge_index: np.ndarray, n: int) -> np.ndarray:
    """Binary undirected combinatorial L=D-A in supplied node-ID order.

    Ignore self loops and repeated entries. Reject asymmetric adjacency instead
    of silently changing a directed graph into an undirected one.
    """
    edge = np.asarray(edge_index)
    if (not isinstance(n, (int, np.integer)) or n < 1 or edge.ndim != 2
            or edge.shape[0] != 2 or not np.issubdtype(edge.dtype, np.integer)):
        raise ValueError("Expected positive node count and integer [2,E] edges")
    if edge.size and (edge.min() < 0 or edge.max() >= n):
        raise ValueError("Edge node ID is outside supplied county order")
    adjacency = np.zeros((n, n), dtype=np.float64)
    adjacency[edge[0], edge[1]] = 1.0
    np.fill_diagonal(adjacency, 0.0)
    if not np.array_equal(adjacency, adjacency.T):
        raise ValueError("Graph OU covariance requires symmetric undirected adjacency")
    return np.diag(adjacency.sum(axis=1)) - adjacency


def _eigendecomposition(laplacian: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(laplacian, dtype=np.float64)
    if (matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[0] != matrix.shape[1]
            or not np.isfinite(matrix).all()
            or not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0)):
        raise ValueError("Laplacian must be a finite symmetric square matrix")
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if eigenvalues.min() < -1e-10:
        raise ValueError("Laplacian must be positive semidefinite")
    return np.maximum(eigenvalues, 0.0), eigenvectors


def modal_covariance(eigenvalues: np.ndarray, parameters: GraphOUParameters,
                     variant: Variant = "nonseparable") -> tuple[np.ndarray, np.ndarray]:
    """Return stationary mode variances q and decay rates alpha, in week units."""
    _names(variant)
    parameters.validate()
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    if (eigenvalues.ndim != 1 or eigenvalues.size < 1
            or not np.isfinite(eigenvalues).all() or (eigenvalues < 0).any()):
        raise ValueError("Expected nonnegative finite Laplacian eigenvalues")
    rates = np.full_like(eigenvalues, parameters.decay)
    if variant == "temporal":
        return np.full_like(eigenvalues, parameters.signal_variance), rates
    log_weights = -parameters.spatial_power * np.log(parameters.spatial_shift + eigenvalues)
    weights = np.exp(log_weights - log_weights.max())
    variances = parameters.signal_variance * weights / weights.mean()
    if variant == "nonseparable":
        rates += parameters.diffusion * eigenvalues
    return variances, rates


def spectral_covariance(times: np.ndarray, laplacian: np.ndarray,
                        parameters: GraphOUParameters, variant: Variant = "nonseparable",
                        *, observation_noise: bool = True) -> np.ndarray:
    """Dense covariance oracle in time-major [t0/node0,...,t1/node0,...] order.

    Production fitting/forecasting does not allocate this dense matrix. Separate
    observations at repeated timestamps retain independent observation noise.
    """
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or times.size < 1 or not np.isfinite(times).all():
        raise ValueError("Expected nonempty finite one-dimensional times")
    eigenvalues, basis = _eigendecomposition(laplacian)
    variances, rates = modal_covariance(eigenvalues, parameters, variant)
    mode_kernel = variances[:, None, None] * np.exp(
        -rates[:, None, None] * np.abs(times[:, None] - times[None, :])[None])
    covariance = np.einsum("im,mtu,jm->tiuj", basis, mode_kernel, basis).reshape(
        times.size * basis.shape[0], times.size * basis.shape[0])
    if observation_noise:
        covariance += parameters.noise_variance * np.eye(covariance.shape[0])
    return covariance


def _filter(modal_values: np.ndarray, variances: np.ndarray, rates: np.ndarray,
            noise: float) -> tuple[float, np.ndarray, np.ndarray]:
    """Stationary scalar OU innovations filter, posterior at last observation."""
    mean = np.zeros_like(variances)
    variance = variances.copy()
    phi = np.exp(-rates)
    process_variance = variances * -np.expm1(-2 * rates)
    nll = 0.0
    for observation in modal_values:
        innovation = observation - mean
        total = variance + noise
        nll += 0.5 * np.sum(np.log(2 * np.pi * total) + innovation ** 2 / total)
        mean += variance / total * innovation
        variance = variance * noise / total
        # Carry the posterior separately so the caller can request any horizon.
        posterior_mean, posterior_variance = mean, variance
        mean = phi * posterior_mean
        variance = phi ** 2 * posterior_variance + process_variance
    return float(nll), posterior_mean, posterior_variance


def marginal_negative_log_likelihood(standardized_values: np.ndarray,
                                     laplacian: np.ndarray,
                                     parameters: GraphOUParameters,
                                     variant: Variant = "nonseparable") -> float:
    """Exact standardized-data GP NLL for complete, equally spaced observations."""
    eigenvalues, basis = _eigendecomposition(laplacian)
    values = np.asarray(standardized_values, dtype=np.float64)
    if (values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != basis.shape[0]
            or not np.isfinite(values).all()):
        raise ValueError("Expected finite nonempty [weeks,N] standardized observations")
    variances, rates = modal_covariance(eigenvalues, parameters, variant)
    return _filter(values @ basis, variances, rates, parameters.noise_variance)[0]


def _decode(log_parameters: np.ndarray, variant: Variant) -> GraphOUParameters:
    return GraphOUParameters(**dict(zip(_names(variant), np.exp(log_parameters), strict=True)))


def _nll_gradient(log_parameters: np.ndarray, modal_values: np.ndarray,
                  eigenvalues: np.ndarray, variant: Variant) -> tuple[float, np.ndarray]:
    """Forward derivatives of the exact innovations likelihood in log parameters."""
    names = _names(variant)
    p = _decode(log_parameters, variant)
    q, rates = modal_covariance(eigenvalues, p, variant)
    k, n = len(names), eigenvalues.size
    dq, drates = np.zeros((k, n)), np.zeros((k, n))
    dnoise = np.zeros((k, 1))
    dq[names.index("signal_variance")] = q
    dnoise[names.index("noise_variance")] = p.noise_variance
    drates[names.index("decay")] = p.decay
    if variant != "temporal":
        for name, dlog_weight in (
            ("spatial_shift", -p.spatial_power * p.spatial_shift / (p.spatial_shift + eigenvalues)),
            ("spatial_power", -p.spatial_power * np.log(p.spatial_shift + eigenvalues)),
        ):
            dq[names.index(name)] = q * (dlog_weight - np.sum(q * dlog_weight) / np.sum(q))
    if variant == "nonseparable":
        drates[names.index("diffusion")] = p.diffusion * eigenvalues
    phi = np.exp(-rates)
    dphi = -phi * drates
    one_minus_phi2 = -np.expm1(-2 * rates)
    mean, variance = np.zeros(n), q.copy()
    dmean, dvariance = np.zeros((k, n)), dq.copy()
    nll, gradient = 0.0, np.zeros(k)
    for observation in modal_values:
        innovation = observation - mean
        total = variance + p.noise_variance
        dtotal = dvariance + dnoise
        dinnovation = -dmean
        nll += 0.5 * np.sum(np.log(2 * np.pi * total) + innovation ** 2 / total)
        gradient += 0.5 * np.sum(
            (1 / total - innovation ** 2 / total ** 2) * dtotal
            + 2 * innovation / total * dinnovation, axis=1)
        gain = variance / total
        dgain = (dvariance - gain * dtotal) / total
        posterior_mean = mean + gain * innovation
        dposterior_mean = dmean + dgain * innovation + gain * dinnovation
        posterior_variance = variance * p.noise_variance / total
        dposterior_variance = (
            dvariance * p.noise_variance / total + variance * dnoise / total
            - posterior_variance * dtotal / total)
        mean = phi * posterior_mean
        dmean = dphi * posterior_mean + phi * dposterior_mean
        variance = phi ** 2 * posterior_variance + q * one_minus_phi2
        dvariance = (2 * phi * dphi * (posterior_variance - q)
                     + phi ** 2 * dposterior_variance + one_minus_phi2 * dq)
    return float(nll), gradient


@dataclass(frozen=True)
class GraphOUForecaster:
    parameters: GraphOUParameters
    variant: Variant
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    scaler: TrainingScaler
    history: int
    counties: tuple[str, ...]
    fit_end: int
    fit_diagnostics: dict

    @classmethod
    def fit_training(cls, data: ChickenpoxData, protocol: ForecastProtocol,
                     scaler: TrainingScaler | None = None,
                     variant: Variant = "nonseparable", *, seed: int = 101,
                     restarts: int = 3, maxiter: int = 100) -> GraphOUForecaster:
        """Fit zero-mean stationary GP hyperparameters on counts[:train_end] only.

        Multistart selection uses training marginal likelihood, never validation.
        Supplied scaler must exactly equal the scaler fitted on training weeks.
        """
        names = _names(variant)
        protocol.targets("train")  # Validate all chronological boundaries.
        counts = np.asarray(data.counts)
        if (counts.ndim != 2 or counts.shape[0] != protocol.observations
                or counts.shape[1] < 1 or len(data.counties) != counts.shape[1]
                or len(set(data.counties)) != len(data.counties)):
            raise ValueError("Protocol dimensions and unique county order must match the data")
        if (not isinstance(restarts, (int, np.integer)) or restarts < 1
                or not isinstance(maxiter, (int, np.integer)) or maxiter < 1
                or not isinstance(seed, (int, np.integer)) or seed < 0):
            raise ValueError("Positive integer optimizer budgets and nonnegative integer seed required")
        expected_scaler = TrainingScaler.fit(counts, protocol.train_end)
        if scaler is None:
            scaler = expected_scaler
        if (scaler.fit_end != protocol.train_end
                or not np.array_equal(scaler.mean, expected_scaler.mean)
                or not np.array_equal(scaler.scale, expected_scaler.scale)):
            raise ValueError("Scaler must use exactly the prescribed training observations")
        started = time.monotonic()
        laplacian = graph_laplacian(data.edge_index, counts.shape[1])
        eigenvalues, basis = _eigendecomposition(laplacian)
        # Identity basis makes the graph-free control explicitly node independent.
        if variant == "temporal":
            basis, eigenvalues = np.eye(counts.shape[1]), np.zeros(counts.shape[1])
        training = scaler.transform(counts[:protocol.train_end]).astype(np.float64)
        modal_training = training @ basis
        bounds = np.log(np.asarray([PARAMETER_BOUNDS[name] for name in names]))
        initial_parameters = GraphOUParameters()
        initial = np.log([getattr(initial_parameters, name) for name in names])
        rng = np.random.default_rng(seed)
        candidates, attempts = [], []
        for restart in range(restarts):
            start = initial if restart == 0 else np.clip(
                initial + rng.normal(0.0, 1.0, len(names)), bounds[:, 0], bounds[:, 1])
            initial_nll = _nll_gradient(start, modal_training, eigenvalues, variant)[0]
            candidates.append((initial_nll, start.copy(), restart, "initial"))
            optimized = minimize(_nll_gradient, start, args=(modal_training, eigenvalues, variant),
                                 method="L-BFGS-B", jac=True, bounds=bounds,
                                 options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-6})
            if np.isfinite(optimized.fun) and np.isfinite(optimized.x).all():
                candidates.append((float(optimized.fun), optimized.x.copy(), restart, "optimized"))
            attempts.append({
                "restart": restart, "initial_parameters": asdict(_decode(start, variant)),
                "initial_nll": float(initial_nll), "final_nll": float(optimized.fun),
                "success": bool(optimized.success), "status": int(optimized.status),
                "message": str(optimized.message), "iterations": int(optimized.nit),
                "evaluations": int(optimized.nfev),
                "gradient_max_abs": float(np.max(np.abs(optimized.jac))),
            })
        selected_nll, fitted_log, selected_restart, selected_kind = min(candidates, key=lambda x: x[0])
        fitted_parameters = _decode(fitted_log, variant)
        at_bound = [name for name, value, bound in zip(names, fitted_log, bounds, strict=True)
                    if np.min(np.abs(value - bound)) < 1e-5]
        diagnostics = {
            "training_nll_standardized": float(selected_nll), "fit_seconds": time.monotonic() - started,
            "optimizer": "L-BFGS-B with exact forward likelihood gradients",
            "seed": int(seed), "restarts": int(restarts), "maxiter_per_restart": int(maxiter),
            "selected_restart": selected_restart, "selected_candidate": selected_kind,
            "attempts": attempts, "active_parameters": list(names),
            "active_parameter_count": len(names),
            "natural_parameter_bounds": {name: list(PARAMETER_BOUNDS[name]) for name in names},
            "parameters_at_bound": at_bound,
            "training_observations_sha256": hashlib.sha256(
                np.ascontiguousarray(counts[:protocol.train_end], dtype="<f8").tobytes()).hexdigest(),
            "laplacian_sha256": hashlib.sha256(laplacian.astype("<f8").tobytes()).hexdigest(),
            "archive_sha256": data.archive_sha256,
        }
        # Detached copies prevent later mutations of the caller's scaler leaking in.
        owned_scaler = TrainingScaler(scaler.mean.copy(), scaler.scale.copy(), scaler.fit_end)
        for array in (eigenvalues, basis, owned_scaler.mean, owned_scaler.scale):
            array.setflags(write=False)
        return cls(fitted_parameters, variant, eigenvalues, basis, owned_scaler,
                   protocol.history, tuple(data.counties), protocol.train_end, diagnostics)

    def predict_distribution(self, raw_history: np.ndarray, *,
                             observation_noise: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """One-week forecast conditioned on exactly history supplied observed weeks.

        Return raw-count mean [N] and covariance [N,N]. Reinitialize the stationary
        prior for every call. The observation covariance includes scalar noise
        in standardized units, which becomes per-county scaled in raw units.
        """
        raw_history = np.asarray(raw_history, dtype=np.float64)
        if (raw_history.shape != (self.history, len(self.counties))
                or not np.isfinite(raw_history).all()):
            raise ValueError("Expected finite raw history in fitted county order with exactly history weeks")
        values = self.scaler.transform(raw_history) @ self.eigenvectors
        variances, rates = modal_covariance(self.eigenvalues, self.parameters, self.variant)
        _, posterior_mean, posterior_variance = _filter(
            values, variances, rates, self.parameters.noise_variance)
        phi = np.exp(-rates)
        forecast_mean = (phi * posterior_mean) @ self.eigenvectors.T
        forecast_variance = phi ** 2 * posterior_variance + variances * -np.expm1(-2 * rates)
        if observation_noise:
            forecast_variance += self.parameters.noise_variance
        covariance = (self.eigenvectors * forecast_variance) @ self.eigenvectors.T
        covariance *= self.scaler.scale[:, None] * self.scaler.scale[None, :]
        return self.scaler.inverse(forecast_mean), covariance

    def predict(self, raw_history: np.ndarray) -> np.ndarray:
        """Unclipped conditional mean in raw case units, shape [N]."""
        return self.predict_distribution(raw_history)[0]

    def metadata(self) -> dict:
        """Serializable provenance and fit diagnostics, no model scoring."""
        return {
            "method": "independent_stationary_spectral_graph_ou_gp", "variant": self.variant,
            "exact_paper_reproduction": False,
            "reference": "Nikitin, John, Solin, Kaski. AISTATS 2022, PMLR 151:10640-10660",
            "reference_url": "https://proceedings.mlr.press/v151/nikitin22a.html",
            "implementation_documentation": "docs/graph_gp.md",
            "graph": "binary undirected combinatorial Laplacian, loops omitted",
            "time_unit": "week", "stationary_initialization": True,
            "likelihood": "complete Gaussian observations, scalar standardized observation noise",
            "hyperparameter_fit_range": [0, self.fit_end],
            "scaler_fit_range": [0, self.scaler.fit_end],
            "training_mean": self.scaler.mean.tolist(), "training_scale": self.scaler.scale.tolist(),
            "history_weeks": self.history, "forecast_horizon_weeks": 1,
            "county_order": list(self.counties), "parameters": asdict(self.parameters),
            "prediction_clipping": "none", "inference": "exact independent-mode Kalman filter",
            "uncertainty": "Gaussian plug-in hyperparameters, no calibration claim",
            "training_complexity": "O(N^3 + T*N^2 + evaluations*T*N*P), P<=6",
            "evaluation_split": "none in this module", "final_test_evaluated": False,
            **self.fit_diagnostics,
        }

