# Fixed geographic boundary memory for Chickenpox

This is a CPU-verified forecast adapter for the existing frozen Hungarian Chickenpox protocol in `docs/chickenpox_protocol.md`. It uses the unchanged `BoundaryMemoryModel`, including its learned graph-local source encoder, exact query-memory actions and comparison controls. No model training or forecast scores were computed while implementing this adapter. Fixed geographic topology cannot establish changing-topology or ARG benefits.

`precompute_chickenpox_geometry(data, history=52, no_geography=False)` prepares a `ChickenpoxBoundaryGeometry` once. All 20 county nodes are persistent boundary and query nodes, with an empty reference set and no interior nodes, so the generic response is exactly C=L. Each unique undirected geographic neighbor link has unit conductance. The 82 native directed non-self records represent 41 links and are deduplicated rather than doubled. The 20 native self loops contribute zero in L=D-A. The original 102 adjacency records remain in the geometry metadata. The source GNN's self/residual update supplies its own node contribution and does not add loop conductance.

The full-boundary mean diagonal of native L is 4.1. The adapter caches the normalized L eigensystem once and shares it across all 52 holds, the homogeneous bag and every forecast history. `no_geography=True` removes the neighbor links, yielding exact C=0 with normalization scale one. The same learned source encoder and memory controls still apply. Geometry preparation stays on CPU. Move geometry with `.to(device, dtype)` once to match the model before binding histories.

`bind_chickenpox_history(geometry, data, scaler, target, permutation=None)` replaces only the observed feature tensor in the cached template. It shares response, coordinates, query partition and eigensystem tensors, without rebuilding graph topology or decomposing matrices. It checks county order, source archive identity and native adjacency against the prepared geometry. The single feature is the standardized observed count. County IDs are correspondence keys and never become learned numeric features.

The input is exactly `counts[target-52:target]`. Fit `TrainingScaler` on raw observations `[0:365]`, as required by the frozen protocol, and keep it fixed during validation and test. No target week or future count enters the bank. Standardized observations first cast to float32, exactly as in `forecast_sequence`, even if a consumer later uses float64. The intervals and final midpoint anchor also match `forecast_sequence` exactly. There are 52 uniform weekly holds on [0,1] in the existing float32 coordinate precision, with a one-sided readout halfway through the final observed hold. The following week's label remains external. This midpoint is a forecast-head convention, not a claim that the future target is part of the observed domain.

`ChickenpoxBoundaryModel(hidden_dim=32, num_layers=2, latent_dim=8, coupling="full", aggregation="ordered")` subclasses the same memory model with `input_dim=1`, scalar `num_classes=1`, and `bidirectional=False`. It returns `[1,20,1]` without replacing a classification head. All four coupling choices are available: `full`, `diagonal`, `gain_matched_diagonal`, and `gain_leak_diagonal`. All three aggregation choices are available: `ordered`, `homogeneous_bag`, and `equilibrium`. The default one-sided configuration has 4,937 parameters. Its common reference projection contains 264 inactive parameters because R is empty, disclosed in the manifest. The remaining 4,673 parameters are active. The same parameter schema is retained across coupling controls.

For R empty, the exact query row sums of L vanish, so the gain-leak control's ideal independent decay is damping alone. It still retains full instantaneous equilibrium mixing of sources, while removing cross-query initial-state memory transfer. Finite-precision row-sum roundoff follows the unchanged memory model's explicit roundoff policy. Raw diagonal and gain-matched diagonal controls use the geographic self-degree in their independent decay. None of these controls changes the source GNN.

The homogeneous bag uses all 52 already observed weeks plus the explicit last-week local readout. It is an offline historical pooling control. For the chronology intervention, use the protocol's `history_permutation(52, seed, fix_last=True)` and pass that permutation to the binding function. County identities, observed multiset, targets, interval locations and last graph remain fixed. The existing float32 linspace intervals have tiny differences in their numerical widths. Their width-weighted bag is consequently nearly, rather than bitwise, permutation-invariant. This precision detail is retained to match common inputs. Reordering graph storage together with its intervals is a separate serialization check.

Train with standardized squared error and inverse-scale outputs before reporting MAE, RMSE and per-county MAE in original cases. Preserve the existing no-clipping policy, chronological splits, rolling-origin evaluation and validation-only configuration selection. The integration owner freezes configuration and schedules training. This module computes no scores and changes no training runner.

Example integration:

```python
from arggnn.boundary_chickenpox import (
    ChickenpoxBoundaryModel, bind_chickenpox_history,
    precompute_chickenpox_geometry,
)
from arggnn.chickenpox import TrainingScaler

scaler = TrainingScaler.fit(data.counts, train_end=365)
model = ChickenpoxBoundaryModel()
geometry = precompute_chickenpox_geometry(data).to("cpu", model.head.weight.dtype)
bank = bind_chickenpox_history(geometry, data, scaler, target=52)
standardized_prediction = model(bank)[0, :, 0]
# The caller supplies the following-week target and owns scoring/training.
```

CPU verification: **42 tests passed in 2.79 seconds**, comprising 14 existing protocol tests and 28 adapter tests. These check every coupling/aggregation shape and gradient, equality with the generic all-boundary primitive and unchanged model computation, empty references, cached geometry sharing, no online eigendecomposition, target/future exclusion, fixed-last permutation, zero-geography equivalence, county permutation equivariance, source/topology mismatch rejection and a finite untrained forward on the pinned archive's first training forecast. The actual-data smoke test reads no forecast labels and computes no score.
