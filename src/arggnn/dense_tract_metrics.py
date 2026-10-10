"""CPU-only, label-separated dense ancestry tract diagnostics.

These are pure helpers, not a custodian, prediction runner, analysis freeze, or
cohort admission rule. No truth-bearing input features are produced.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Integral, Real
from collections.abc import Mapping

import msprime
import numpy as np
import tskit
from scipy.special import logsumexp

from .confirmatory_metrics import boundary_match


def _integer(value, name, *, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _query_order(values):
    values = tuple(values)
    if (not values or any(isinstance(v, (bool, np.bool_)) or
            not isinstance(v, (str, Integral)) or (isinstance(v, str) and not v)
            for v in values)):
        raise ValueError("Explicit nonempty ordered string/integer query IDs required")
    values = tuple(int(v) if isinstance(v, Integral) else v for v in values)
    if len(set(values)) != len(values):
        raise ValueError("Query IDs must be unique")
    return values


def _interval(values, name):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (2,) or not np.isfinite(values).all() or values[0] >= values[1]:
        raise ValueError(f"{name} must be a finite increasing physical-bp interval")
    return tuple(float(v) for v in values)


def _physical_domain(values):
    domain = _interval(values, "domain_bp")
    if domain[0] < 0 or not np.isfinite(domain[1]-domain[0]):
        raise ValueError("Physical domain must be nonnegative with finite representable span")
    return domain


def _physical_parameter(value, name, *, positive=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or
            not np.isfinite(value) or (value <= 0 if positive else value < 0)):
        raise ValueError(f"{name} must be finite {'positive' if positive else 'nonnegative'} physical units")
    return float(value)


def _global_midpoints(domain, n):
    full = domain[0] + (np.arange(n, dtype=np.float64)+.5)*((domain[1]-domain[0])/n)
    if (not np.isfinite(full).all() or np.any(np.diff(full) <= 0) or
            np.any(full <= domain[0]) or np.any(full >= domain[1])):
        raise ValueError("Global midpoint coordinates must be finite, internal, and strictly increasing in float64")
    return full


@dataclass(frozen=True)
class DenseGrid:
    """Identity of retained centers of one global uniform midpoint grid.

    Every original center inside half-open coverage must be retained, in its
    original order. Disjoint coverage or missing interior centers is unsupported
    and rejected rather than bridged. Toy resolutions are allowed explicitly;
    the proposed evaluation resolution is 4096.
    """
    grid_id: str
    domain_bp: tuple[float, float]
    coverage_bp: tuple[float, float]
    grid_points: int
    original_indices: tuple[int, ...]
    positions_bp: tuple[float, ...]

    def __post_init__(self):
        if not isinstance(self.grid_id, str) or not self.grid_id:
            raise ValueError("An explicit nonempty grid_id is required")
        domain = _physical_domain(self.domain_bp)
        coverage = _interval(self.coverage_bp, "coverage_bp")
        n = _integer(self.grid_points, "grid_points", minimum=1)
        if coverage[0] < domain[0] or coverage[1] > domain[1]:
            raise ValueError("Coverage must be contained in the declared physical domain")
        indices = np.asarray(self.original_indices)
        positions = np.asarray(self.positions_bp, dtype=np.float64)
        if indices.ndim != 1 or positions.ndim != 1 or len(indices) != len(positions):
            raise ValueError("Aligned one-dimensional original indices and positions required")
        # The empty vector still has a valid integer identity.
        if len(indices) and (not np.issubdtype(indices.dtype, np.integer) or
                np.issubdtype(indices.dtype, np.bool_)):
            raise ValueError("Original indices must be integer slots")
        full = _global_midpoints(domain, n)
        expected = np.flatnonzero((full >= coverage[0]) & (full < coverage[1]))
        if not np.array_equal(indices, expected):
            raise ValueError("Retained original indices must exactly match ordered coverage-clipped grid")
        tolerance = 8 * np.finfo(np.float64).eps * max(1., abs(domain[0]), abs(domain[1]))
        if not np.isfinite(positions).all() or not np.allclose(
                positions, full[expected], rtol=0., atol=tolerance):
            raise ValueError("Positions must preserve the original global uniform midpoint grid")
        object.__setattr__(self, "domain_bp", domain)
        object.__setattr__(self, "coverage_bp", coverage)
        object.__setattr__(self, "grid_points", n)
        object.__setattr__(self, "original_indices", tuple(int(i) for i in indices))
        object.__setattr__(self, "positions_bp", tuple(float(p) for p in positions))

    def cell_bounds_bp(self):
        """Original cells clipped physically to coverage; no missing-center fill."""
        indices = np.asarray(self.original_indices, dtype=np.float64)
        step = (self.domain_bp[1]-self.domain_bp[0])/self.grid_points
        left = np.maximum(self.coverage_bp[0], self.domain_bp[0]+indices*step)
        right = np.minimum(self.coverage_bp[1], self.domain_bp[0]+(indices+1)*step)
        return np.column_stack((left, right))

    def identity_record(self):
        record = dict(grid_id=self.grid_id, domain_bp=list(self.domain_bp),
            coverage_bp=list(self.coverage_bp), grid_points=self.grid_points,
            original_indices=list(self.original_indices), positions_bp=list(self.positions_bp))
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        return dict(**record, sha256=hashlib.sha256(encoded).hexdigest())


def coverage_clipped_midpoint_grid(*, domain_bp, coverage_bp, grid_id, grid_points=4096):
    """Declare observed readout coordinates only, without features or truth."""
    domain = _physical_domain(domain_bp)
    coverage = _interval(coverage_bp, "coverage_bp")
    n = _integer(grid_points, "grid_points", minimum=1)
    full = _global_midpoints(domain, n)
    indices = np.flatnonzero((full >= coverage[0]) & (full < coverage[1]))
    return DenseGrid(grid_id, domain, coverage, n, tuple(indices), tuple(full[indices]))


@dataclass(frozen=True)
class Tract:
    left_bp: float
    right_bp: float
    class_slot: int

    def __post_init__(self):
        interval = _interval((self.left_bp, self.right_bp), "tract")
        slot = _integer(self.class_slot, "class_slot")
        object.__setattr__(self, "left_bp", interval[0])
        object.__setattr__(self, "right_bp", interval[1])
        object.__setattr__(self, "class_slot", slot)


@dataclass(frozen=True)
class TractPartition:
    """Exact full-domain partitions in explicitly bound original query order."""
    query_ids: tuple
    domain_bp: tuple[float, float]
    num_classes: int
    tracts: tuple[tuple[Tract, ...], ...]

    def __post_init__(self):
        queries = _query_order(self.query_ids)
        domain = _physical_domain(self.domain_bp)
        k = _integer(self.num_classes, "num_classes", minimum=2)
        rows = tuple(tuple(t if isinstance(t, Tract) else Tract(*t) for t in row)
                     for row in self.tracts)
        if len(rows) != len(queries):
            raise ValueError("Partition rows must preserve explicit query correspondence")
        for row in rows:
            if not row or row[0].left_bp != domain[0] or row[-1].right_bp != domain[1]:
                raise ValueError("Each truth partition must cover the complete declared domain")
            for i, tract in enumerate(row):
                if tract.class_slot >= k:
                    raise ValueError("Truth class slots must lie in [0,num_classes)")
                if i and (row[i-1].right_bp != tract.left_bp or
                          row[i-1].class_slot == tract.class_slot):
                    raise ValueError("Truth partitions must be contiguous and merge adjacent equal ancestry")
        object.__setattr__(self, "query_ids", queries)
        object.__setattr__(self, "domain_bp", domain)
        object.__setattr__(self, "num_classes", k)
        object.__setattr__(self, "tracts", rows)


def extract_census_tracts(ts, query_node_ids, *, query_ids,
                          source_population_to_class, census_time, num_classes):
    """Label-only extraction from an external census-bearing tskit sequence.

    query_node_ids[i] is explicitly bound to query_ids[i], even when truth and
    observed node numbers differ. Exactly one census-flagged ancestor is
    required on every query path on every exact tree interval, at census_time
    (absolute tolerance 1e-9 generations). Topology changes with equal ancestry
    are merged. Nothing from this helper belongs in model inputs.
    """
    if not isinstance(ts, tskit.TreeSequence):
        raise ValueError("An external tskit TreeSequence is required")
    queries = _query_order(query_ids)
    nodes = tuple(_integer(q, "query_node_id") for q in query_node_ids)
    if len(nodes) != len(queries) or len(set(nodes)) != len(nodes):
        raise ValueError("Unique query nodes aligned to explicit ordered query IDs required")
    if not set(nodes).issubset(set(int(q) for q in ts.samples())):
        raise ValueError("Query nodes must be sample nodes")
    k = _integer(num_classes, "num_classes", minimum=2)
    census_time = _physical_parameter(census_time, "census_time", positive=True)
    if any(ts.node(q).time >= census_time for q in nodes):
        raise ValueError("Query sampling dates must be younger than census")
    if not isinstance(source_population_to_class, Mapping):
        raise ValueError("Explicit population-to-class mapping required")
    mapping = {}
    for population, slot in source_population_to_class.items():
        population = _integer(population, "source population")
        slot = _integer(slot, "source class slot")
        if population >= ts.num_populations or slot >= k:
            raise ValueError("Population map requires existing populations and valid class slots")
        mapping[population] = slot
    if not mapping:
        raise ValueError("Explicit nonempty population-to-class map required")
    rows = [[] for _ in queries]
    for tree in ts.trees():
        left, right = map(float, tree.interval)
        for i, q in enumerate(nodes):
            u = q
            census = []
            while u != tskit.NULL:
                if ts.node(u).flags & msprime.NODE_IS_CEN_EVENT:
                    census.append(u)
                u = tree.parent(u)
            if len(census) != 1:
                kind = "missing" if not census else "ambiguous"
                raise ValueError(f"{kind} census ancestry for query {queries[i]!r} in [{left},{right})")
            ancestor = ts.node(census[0])
            if not np.isclose(ancestor.time, census_time, rtol=0., atol=1e-9):
                raise ValueError("Invalid census ancestor age")
            if ancestor.population not in mapping:
                raise ValueError("Census ancestor population missing from explicit class map")
            slot = mapping[ancestor.population]
            if rows[i] and rows[i][-1].class_slot == slot:
                previous = rows[i][-1]
                rows[i][-1] = Tract(previous.left_bp, right, slot)
            else:
                rows[i].append(Tract(left, right, slot))
    return TractPartition(queries, (0., float(ts.sequence_length)), k, tuple(map(tuple, rows)))


def _stable_log_softmax(logits):
    # Same raw-logit policy as confirmatory_metrics.logit_metrics; no floor.
    with np.errstate(over="ignore", invalid="ignore"):
        shifted = logits - logits.max(axis=-1, keepdims=True)
        logp = shifted - logsumexp(shifted, axis=-1, keepdims=True)
    if not np.isfinite(logp).all():
        raise ValueError("Log-softmax exceeds float64 representable range")
    return logp


def _labels_at(row, positions):
    ends = np.array([t.right_bp for t in row], dtype=np.float64)
    slots = np.array([t.class_slot for t in row], dtype=np.int64)
    return slots[np.searchsorted(ends, positions, side="right")]


def _span_readout(logp, labels, predicted, weights, k):
    """Midpoint-label quadrature, never an exact integrated likelihood."""
    weights = np.asarray(weights, dtype=np.float64)
    denominator = float(weights.sum())
    active = weights > 0
    chosen_loss = -logp[np.arange(len(labels)), labels]
    confusion = np.bincount(labels*k+predicted, weights=weights, minlength=k*k).reshape(k, k)
    classes = []
    for c in range(k):
        mask = active & (labels == c)
        support = float(weights[mask].sum())
        predicted_support = float(weights[predicted == c].sum())
        tp = float(confusion[c, c])
        total = support+predicted_support
        classes.append(dict(class_slot=c, midpoint_support=int(mask.sum()),
            true_support_bp=support, predicted_support_bp=predicted_support,
            nll=float(np.sum(chosen_loss[mask]*weights[mask])/support) if support else None,
            accuracy=float(tp/support) if support else None,
            precision=float(tp/predicted_support) if predicted_support else None,
            recall=float(tp/support) if support else None,
            f1=float(2*tp/total) if total else None))
    return dict(evaluated_midpoints=int(active.sum()), denominator_bp=denominator,
        nll=float(np.sum(chosen_loss*weights)/denominator) if denominator else None,
        accuracy=float(np.sum((labels == predicted)*weights)/denominator) if denominator else None,
        confusion_bp=confusion.tolist(), per_class=classes)


def _censor_flags(left, right, grid):
    dl, dr = grid.domain_bp
    cl, cr = grid.coverage_bp
    domain_left = left == dl
    domain_right = right == dr
    coverage_left = left == cl and cl > dl
    coverage_right = right == cr and cr < dr
    return dict(domain_censored_left=domain_left, domain_censored_right=domain_right,
        coverage_censored_left=coverage_left, coverage_censored_right=coverage_right,
        left_censored=domain_left or coverage_left,
        right_censored=domain_right or coverage_right,
        full_observed=not (domain_left or domain_right or coverage_left or coverage_right))


def _clip_truth(row, grid):
    cl, cr = grid.coverage_bp
    records = []
    for i, tract in enumerate(row):
        left, right = max(cl, tract.left_bp), min(cr, tract.right_bp)
        if left < right:
            records.append(dict(tract_index=i, class_slot=tract.class_slot,
                left_bp=left, right_bp=right, length_bp=right-left,
                original_left_bp=tract.left_bp, original_right_bp=tract.right_bp,
                original_length_bp=tract.right_bp-tract.left_bp,
                coverage_truncated_left=left > tract.left_bp,
                coverage_truncated_right=right < tract.right_bp,
                **_censor_flags(left, right, grid)))
    return records


def _predicted_tracts(predicted, positions, grid):
    if not len(positions):
        return []
    changes = np.flatnonzero(predicted[1:] != predicted[:-1])
    boundaries = (positions[changes]+positions[changes+1])/2
    endpoints = np.r_[grid.coverage_bp[0], boundaries, grid.coverage_bp[1]]
    slots = np.r_[predicted[0], predicted[changes+1]]
    return [dict(tract_index=i, class_slot=int(slot), left_bp=float(endpoints[i]),
        right_bp=float(endpoints[i+1]), length_bp=float(endpoints[i+1]-endpoints[i]),
        **_censor_flags(float(endpoints[i]), float(endpoints[i+1]), grid))
        for i, slot in enumerate(slots)]


def _length_summary(records):
    lengths = [float(t["length_bp"]) for t in records]
    full_lengths = [float(t["length_bp"]) for t in records if t["full_observed"]]
    def summary(values):
        return dict(count=len(values), total_bp=float(sum(values)),
            mean_bp=float(np.mean(values)) if values else None,
            median_bp=float(np.median(values)) if values else None,
            min_bp=float(min(values)) if values else None,
            max_bp=float(max(values)) if values else None, lengths_bp=values)
    return dict(visible=summary(lengths), full_observed=summary(full_lengths),
        censored_tracts=len(lengths)-len(full_lengths),
        domain_censored_tracts=sum(t["domain_censored_left"] or t["domain_censored_right"] for t in records),
        coverage_censored_tracts=sum(t["coverage_censored_left"] or t["coverage_censored_right"] for t in records))


def _near_edges(positions, edges, radius):
    if not len(edges):
        return np.zeros(len(positions), dtype=bool)
    return np.min(np.abs(np.asarray(positions)[:, None]-edges[None, :]), axis=1) <= radius


def _boundary_groups(match, true_positions, predicted_positions, edges, radius):
    true_near = _near_edges(true_positions, edges, radius)
    predicted_near = _near_edges(predicted_positions, edges, radius)
    result = {}
    for name, use_near in (("near_tile_edges", True), ("interior", False)):
        tm = true_near if use_near else ~true_near
        pm = predicted_near if use_near else ~predicted_near
        nt, npred = int(tm.sum()), int(pm.sum())
        mt = sum(bool(tm[m["true_index"]]) for m in match["matches"])
        mp = sum(bool(pm[m["predicted_index"]]) for m in match["matches"])
        result[name] = dict(true_boundary_support=nt, predicted_boundary_support=npred,
            matched_true_boundaries=mt, matched_predicted_boundaries=mp,
            missed=nt-mt, false=npred-mp, recall=mt/nt if nt else None,
            precision=mp/npred if npred else None)
    result["cross_group_matches"] = sum(bool(true_near[m["true_index"]]) !=
        bool(predicted_near[m["predicted_index"]]) for m in match["matches"])
    return result


def _overlap_geometry(true_records, predicted_records, k):
    confusion = np.zeros((k, k), dtype=np.float64)
    overlaps = [np.zeros(k, dtype=np.float64) for _ in true_records]
    i = j = 0
    # Both partitions are ordered; a linear sweep avoids a Cartesian tract join.
    while i < len(true_records) and j < len(predicted_records):
        true, predicted = true_records[i], predicted_records[j]
        span = max(0., min(true["right_bp"], predicted["right_bp"]) -
                   max(true["left_bp"], predicted["left_bp"]))
        overlaps[i][predicted["class_slot"]] += span
        confusion[true["class_slot"], predicted["class_slot"]] += span
        true_end, predicted_end = true["right_bp"], predicted["right_bp"]
        if true_end <= predicted_end:
            i += 1
        if predicted_end <= true_end:
            j += 1
    for true, overlap in zip(true_records, overlaps):
        support = float(overlap.sum())
        true["reconstructed_prediction_overlap_bp_per_class"] = overlap.tolist()
        true["reconstructed_prediction_overlap_support_bp"] = support
        true["reconstructed_prediction_overlap_accuracy"] = (
            float(overlap[true["class_slot"]]/support) if support else None)
    total = float(confusion.sum())
    classes = []
    for c in range(k):
        truth = float(confusion[c].sum())
        prediction = float(confusion[:, c].sum())
        visible_truth = float(sum(t["length_bp"] for t in true_records if t["class_slot"] == c))
        visible_prediction = float(sum(t["length_bp"] for t in predicted_records if t["class_slot"] == c))
        classes.append(dict(class_slot=c, true_support_bp=truth,
            predicted_support_bp=prediction, true_visible_support_bp=visible_truth,
            predicted_visible_support_bp=visible_prediction,
            signed_class_span_error_bp=prediction-truth if total else None,
            missed_class_span_bp=float(truth-confusion[c, c]) if total else None,
            false_class_span_bp=float(prediction-confusion[c, c]) if total else None,
            recall=float(confusion[c, c]/truth) if truth else None,
            precision=float(confusion[c, c]/prediction) if prediction else None))
    return dict(denominator_bp=total, confusion_bp=confusion.tolist(),
        accuracy=float(np.trace(confusion)/total) if total else None, per_class=classes,
        policy="Exact overlap of exact clipped truth with resolution-limited reconstructed argmax tracts")


def _length_errors(true_records, predicted_records, match):
    matched = {m["true_bp"]: m["predicted_index"] for m in match["matches"]}
    for true in true_records:
        true["matched_predicted_tract_index"] = None
        true["matched_length_error_bp"] = None
        if not true["full_observed"]:
            continue
        left = matched.get(true["left_bp"])
        right = matched.get(true["right_bp"])
        if left is not None and right == left+1:
            predicted = predicted_records[left+1]
            if predicted["class_slot"] == true["class_slot"]:
                true["matched_predicted_tract_index"] = left+1
                true["matched_length_error_bp"] = predicted["length_bp"]-true["original_length_bp"]


def _length_error_summary(records):
    eligible = [t for t in records if t["full_observed"]]
    errors = [float(t["matched_length_error_bp"]) for t in eligible
              if t["matched_length_error_bp"] is not None]
    return dict(full_true_tract_support=len(eligible), matched_length_support=len(errors),
        unmatched_full_true_tracts=len(eligible)-len(errors), signed_errors_bp=errors,
        mean_signed_error_bp=float(np.mean(errors)) if errors else None,
        mean_absolute_error_bp=float(np.mean(np.abs(errors))) if errors else None,
        policy="Both true switches matched to consecutive predicted switches enclosing the same class")


def _short_tract_readout(true_records, logp, labels, predicted, positions, cells, k, cutoff):
    eligible = [t for t in true_records if t["full_observed"] and t["original_length_bp"] <= cutoff]
    weights = np.zeros(len(positions), dtype=np.float64)
    records = []
    for true in eligible:
        mask = (positions >= true["original_left_bp"]) & (positions < true["original_right_bp"])
        clipped = np.maximum(0., np.minimum(cells[:, 1], true["original_right_bp"]) -
                             np.maximum(cells[:, 0], true["original_left_bp"]))
        clipped[~mask] = 0.
        weights += clipped
        readout = _span_readout(logp, labels, predicted, clipped, k)
        records.append(dict(tract_index=true["tract_index"], class_slot=true["class_slot"],
            length_bp=true["original_length_bp"], retained_midpoints=int(mask.sum()),
            evaluated_quadrature_bp=readout["denominator_bp"], nll=readout["nll"],
            accuracy=readout["accuracy"]))
    def support(selected):
        total = float(sum(t["length_bp"] for t in selected))
        represented = float(sum(t["evaluated_quadrature_bp"] for t in selected))
        sampled = sum(t["retained_midpoints"] > 0 for t in selected)
        return dict(true_tract_count=len(selected), true_tract_bp=total,
            midpoint_supported_tracts=sampled, zero_midpoint_tracts=len(selected)-sampled,
            evaluated_quadrature_bp=represented, unrepresented_true_tract_bp=max(0., total-represented))
    readout = _span_readout(logp, labels, predicted, weights, k)
    return dict(cutoff_bp=float(cutoff), **support(records), readout=readout, tracts=records,
        per_class=[dict(class_slot=c, **support([t for t in records if t["class_slot"] == c]),
                       readout=readout["per_class"][c]) for c in range(k)],
        policy="Original length <= cutoff, with both true switch endpoints strictly inside coverage/domain")


def _aggregate_boundaries(reports):
    totals = {key: sum(r[key] for r in reports)
              for key in ("true_boundaries", "predicted_boundaries", "matched", "missed", "extra")}
    nt, npred, matched = totals["true_boundaries"], totals["predicted_boundaries"], totals["matched"]
    errors = [r["absolute_switch_count_error"] for r in reports]
    return dict(**totals, precision=matched/npred if npred else None,
        recall=matched/nt if nt else None, f1=2*matched/(nt+npred) if nt else None,
        absolute_switch_count_error_total=sum(errors),
        mean_absolute_switch_count_error_per_query=float(np.mean(errors)),
        query_support=len(reports))


def dense_tract_metrics(raw_logits, *, query_ids, grid, true_partitions,
                        artificial_tile_edges_bp, boundary_tolerance_bp=1000.,
                        short_tract_cutoff_bp=10000., tile_edge_radius_bp=10000.):
    """Descriptive one-parent/one-checkpoint dense metrics; no inference.

    Logits are [retained original midpoint, explicit ordered query, class].
    Truth is joined only here, after a caller has sealed observed predictions.
    All thresholds are physical bp. Defaults follow confirmatory_design_v1;
    callers must freeze any changes outside this module.
    """
    if not isinstance(grid, DenseGrid) or not isinstance(true_partitions, TractPartition):
        raise ValueError("Validated DenseGrid and TractPartition objects required")
    queries = _query_order(query_ids)
    if queries != true_partitions.query_ids:
        raise ValueError("Prediction and truth query IDs must have identical explicit original order")
    if grid.domain_bp != true_partitions.domain_bp:
        raise ValueError("Truth and prediction physical domains must agree exactly")
    k = true_partitions.num_classes
    z = np.asarray(raw_logits, dtype=np.float64)
    if z.shape != (len(grid.positions_bp), len(queries), k) or not np.isfinite(z).all():
        raise ValueError("Finite raw logits must align [retained original grid,query,class]")
    for value, name in ((boundary_tolerance_bp, "boundary_tolerance_bp"),
                        (short_tract_cutoff_bp, "short_tract_cutoff_bp"),
                        (tile_edge_radius_bp, "tile_edge_radius_bp")):
        _physical_parameter(value, name)
    edges = np.asarray(artificial_tile_edges_bp, dtype=np.float64)
    if (edges.ndim != 1 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0) or
            np.any(edges <= grid.domain_bp[0]) or np.any(edges >= grid.domain_bp[1])):
        raise ValueError("Tile edges must be explicit sorted distinct internal physical domain coordinates")
    logp = _stable_log_softmax(z)
    positions = np.asarray(grid.positions_bp, dtype=np.float64)
    predicted = z.argmax(axis=-1)
    labels = np.column_stack([_labels_at(row, positions) for row in true_partitions.tracts])
    cells = grid.cell_bounds_bp()
    weights = cells[:, 1]-cells[:, 0]
    near = _near_edges(positions, edges, tile_edge_radius_bp)
    query_records = []
    for i, query in enumerate(queries):
        truth = _clip_truth(true_partitions.tracts[i], grid)
        prediction = _predicted_tracts(predicted[:, i], positions, grid)
        true_positions = np.array([t.right_bp for t in true_partitions.tracts[i][:-1]
            if grid.coverage_bp[0] < t.right_bp < grid.coverage_bp[1]])
        predicted_positions = np.array([t["right_bp"] for t in prediction[:-1]])
        match = boundary_match(true_positions, predicted_positions, tolerance_bp=boundary_tolerance_bp)
        geometry = _overlap_geometry(truth, prediction, k)
        _length_errors(truth, prediction, match)
        class_records = []
        for c in range(k):
            true_class_positions = [t.right_bp for j, t in enumerate(true_partitions.tracts[i][:-1])
                if grid.coverage_bp[0] < t.right_bp < grid.coverage_bp[1] and
                c in (t.class_slot, true_partitions.tracts[i][j+1].class_slot)]
            predicted_class_positions = [t["right_bp"] for j, t in enumerate(prediction[:-1])
                if c in (t["class_slot"], prediction[j+1]["class_slot"])]
            true_class = [t for t in truth if t["class_slot"] == c]
            prediction_class = [t for t in prediction if t["class_slot"] == c]
            class_records.append(dict(class_slot=c,
                boundary_incidence=boundary_match(true_class_positions, predicted_class_positions,
                                                  tolerance_bp=boundary_tolerance_bp),
                true_tract_lengths=_length_summary(true_class),
                predicted_tract_lengths=_length_summary(prediction_class),
                tract_length_errors=_length_error_summary(true_class),
                reconstructed_class_span_errors=geometry["per_class"][c]))
        query_records.append(dict(query_id=query,
            visible_true_boundary_positions_bp=true_positions.tolist(),
            predicted_boundary_positions_bp=predicted_positions.tolist(), boundaries=match,
            boundary_location_groups=_boundary_groups(match, true_positions, predicted_positions,
                                                      edges, tile_edge_radius_bp),
            true_tracts=truth, predicted_tracts=prediction,
            true_tract_lengths=_length_summary(truth), predicted_tract_lengths=_length_summary(prediction),
            tract_length_errors=_length_error_summary(truth), reconstructed_tract_overlap=geometry,
            span_quadrature=_span_readout(logp[:, i], labels[:, i], predicted[:, i], weights, k),
            readout_location_groups={name: _span_readout(logp[:, i], labels[:, i], predicted[:, i],
                weights*mask, k) for name, mask in (("near_tile_edges", near), ("interior", ~near))},
            short_true_tracts=_short_tract_readout(truth, logp[:, i], labels[:, i], predicted[:, i],
                positions, cells, k, short_tract_cutoff_bp),
            per_class=class_records))
    aggregate_readout = _span_readout(logp.reshape(-1, k), labels.ravel(), predicted.ravel(),
                                     np.repeat(weights, len(queries)), k)
    represented = float(weights.sum())
    coverage_span = grid.coverage_bp[1]-grid.coverage_bp[0]
    return dict(schema_version=1, grid=grid.identity_record(), query_ids=list(queries),
        num_classes=k, artificial_tile_edges_bp=edges.tolist(),
        boundary_tolerance_bp=float(boundary_tolerance_bp),
        short_tract_cutoff_bp=float(short_tract_cutoff_bp), tile_edge_radius_bp=float(tile_edge_radius_bp),
        retained_midpoints=len(positions), physical_coverage_bp=coverage_span,
        quadrature_represented_physical_bp=represented,
        quadrature_unrepresented_physical_bp=max(0., coverage_span-represented),
        quadrature_cell_bounds_bp=cells.tolist(),
        boundaries=_aggregate_boundaries([q["boundaries"] for q in query_records]),
        span_quadrature=aggregate_readout,
        per_class=[dict(class_slot=c, span_quadrature=aggregate_readout["per_class"][c],
            boundary_incidence=_aggregate_boundaries([q["per_class"][c]["boundary_incidence"]
                                                      for q in query_records]))
            for c in range(k)],
        per_query=query_records,
        policies=dict(nll="Float64 stable log-softmax of original uncalibrated raw logits; no probability floor",
            quadrature="Approximate midpoint truth-label quadrature over physical clipped original cells only; "
                "truth changes inside cells are not integrated and cells with missing centers receive no support",
            prediction="Resolution-limited argmax tracts; switches are midpoints between successive retained "
                "dense samples; endpoint classes extend to coverage only for reconstructed tract geometry",
            boundaries="Only switches strictly inside coverage; coverage/domain endpoints are never switches",
            short_tracts="Only <= cutoff original true tracts with two visible true switch endpoints; "
                "short cells additionally clipped to each eligible tract and requiring their center inside it",
            tile_groups="Within radius inclusive of any artificial internal tile edge, assigned by midpoint "
                "for readout and each boundary coordinate for global-match group counts",
            class_boundaries="Class-incidence diagnostic, independently one-to-one matched per class; "
                "each switch belongs to its two adjoining classes; global matching is class-agnostic",
            units="All spans and coordinates are physical bp; aggregate quadrature denominator is query-bp",
            inference="Descriptive within one parent/checkpoint only; queries/cells are not independent parents "
                "or training seeds; no confidence intervals or scientific-phase promotion"))
