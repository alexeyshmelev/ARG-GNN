"""Observed-only CPU preparation for direct mutation-routed whole-tree models.

No loader, ancestry target, population annotation, census flag, oracle tree or
inferred ancestral allele is accepted. Native anonymous internals and unary
branches survive. Sample IDs are joins only. All original holds survive.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Sequence
import math

import numpy as np

CLASSES = ("ANA", "CHG", "WHG", "EHG")
SUPPORT_FIELDS = ("exact_split_support", "closest_weighted_support",
                  "mean_discrepancy", "callable_split_count", "empty")
ROSTER_FIELDS = {"sample_index", "sample_id", "haplotype", "observed_node_id",
                 "observed_cohort", "age_generations", "role", "observed_reference_class"}


@dataclass(frozen=True)
class NativeObservedTreeV1:
    parent: np.ndarray
    node_time: np.ndarray
    sample_node: np.ndarray
    sample_id: np.ndarray


@dataclass(frozen=True)
class ObservedSampleJoinV1:
    sample_id: np.ndarray
    native_sample_node: np.ndarray
    reference_class: np.ndarray
    sample_age: np.ndarray
    query_id: np.ndarray
    haplotype_keys: tuple


@dataclass(frozen=True)
class ShamBinEdgesV1:
    """Quantile interior edges fixed from TRAIN observed geometry only."""
    log_duration_edges: tuple
    descendant_count_edges: tuple
    training_branch_count: int
    bins_requested: int
    provenance: str = "TRAIN_observed_native_geometry_before_targets"


@dataclass(frozen=True)
class MutationRoutedTreeObservationsV1:
    parent: np.ndarray
    node_time: np.ndarray
    sample_index: np.ndarray
    upward_levels: tuple
    upward_children: tuple
    upward_parent_rank: tuple
    downward_levels: tuple
    edge_geometry: np.ndarray
    branch_support_true: np.ndarray
    branch_support_sham: np.ndarray
    genotype: np.ndarray
    log_donor_allele_frequency: np.ndarray
    donor_call_mask: np.ndarray
    panel_allele_frequency: np.ndarray
    site_entropy: np.ndarray
    site_hold: np.ndarray
    spans: np.ndarray
    readout_hold: np.ndarray
    query_column: np.ndarray
    reference_class: np.ndarray
    sample_age: np.ndarray
    metadata: dict

    def model_arrays(self):
        """Exact network dataclass fields; no provenance or target extras."""
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != "metadata"}


def _integer(value, name, ndim=1):
    a = np.asarray(value)
    if a.ndim != ndim or a.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer array of rank {ndim}")
    return a


def validate_observed_sample_join_v1(original, derived, mapping,
        native_sample_nodes, native_sample_times, *, fixture=False):
    """H028/H035-qualified original-to-tsinfer observed roster identity join.

    original and derived are observation manifests, not tree-node metadata.
    This helper never asks for tree populations or simulated mutation records.
    """
    count = len(original)
    native = _integer(native_sample_nodes, "native_sample_nodes")
    times = np.asarray(native_sample_times, dtype=np.float64)
    if (not count or len(derived) != count or len(mapping) != count or len(native) != count
            or len(set(native.tolist())) != count or np.any(native < 0) or times.shape != (count,)
            or not np.isfinite(times).all() or np.any(times < 0)):
        raise ValueError("Complete unique native sampled roster and observed dates required")
    native_age = dict(zip(native.tolist(), times.tolist()))
    keys, ids, ages, ref, query = [], [], [], [], []
    role_counts = {k: 0 for k in ("query", "pure_source_reference", "mixed_reference_only")}
    for i, (raw, row, join) in enumerate(zip(original, derived, mapping)):
        if (set(raw) != ROSTER_FIELDS or type(raw["sample_index"]) is not int
                or raw["sample_index"] != i or set(join) != {"sample_index", "inferred_tree_node_id"}
                or join["sample_index"] != i or type(join["inferred_tree_node_id"]) is not int
                or type(raw["observed_node_id"]) is not int):
            raise ValueError("Exact original field whitelist and explicit native sample mapping required")
        node = join["inferred_tree_node_id"]
        expected = dict(raw)
        expected["source_observed_node_id"] = expected.pop("observed_node_id")
        expected["observed_node_id"] = expected["inferred_tree_node_id"] = node
        if row != expected:
            raise ValueError("Derived observed identity, date or native mapping changed")
        sid, hap = raw["sample_id"], raw["haplotype"]
        age, role, cls = raw["age_generations"], raw["role"], raw["observed_reference_class"]
        if (not isinstance(sid, str) or not sid or type(hap) is not int or hap not in (0, 1)
                or type(age) not in (int, float) or not math.isfinite(age) or age < 0
                or node not in native_age or native_age[node] != age or role not in role_counts):
            raise ValueError("Observed sampled haplotype, role and exact native date required")
        if role == "pure_source_reference":
            if type(cls) is not int or cls not in range(4) or raw["observed_cohort"] != CLASSES[cls]:
                raise ValueError("Only observed pure-source proxy classes may be input labels")
            ref.append(cls)
        else:
            if cls is not None:
                raise ValueError("Unknown query and mixed-reference input ancestry required")
            ref.append(-1)
        if role == "query":
            query.append(i)
        keys.append((sid, hap)); ids.append(node); ages.append(age); role_counts[role] += 1
    if len(set(keys)) != count or len(set(ids)) != count or set(ids) != set(native.tolist()):
        raise ValueError("Sample ID/haplotype and native sample joins must be bijections")
    if not fixture:
        if count != 280 or role_counts != {"query": 120, "pure_source_reference": 96, "mixed_reference_only": 64}:
            raise ValueError("All280 observed haplotypes, pureR96, mixed64 and actualQ120 required")
        if [ref.count(c) for c in range(4)] != [24] * 4:
            raise ValueError("Observed pureR96 must have24 per class")
        for i in range(0, count, 2):
            a, b = original[i:i+2]
            if (a["sample_id"] != b["sample_id"] or [a["haplotype"], b["haplotype"]] != [0, 1]
                    or any(a[k] != b[k] for k in ("role", "age_generations", "observed_cohort", "observed_reference_class"))):
                raise ValueError("Original phased diploid sample pairing changed")
    return ObservedSampleJoinV1(np.arange(count, dtype=np.int64), np.asarray(ids, dtype=np.int64),
        np.asarray(ref, dtype=np.int64), np.asarray(ages, dtype=np.float64),
        np.asarray(query, dtype=np.int64), tuple(keys))


def align_observed_genotype_columns_v1(genotype, source_keys, desired_keys):
    """Unique exact (actual sample_id, haplotype) join, never positional guessing."""
    g = _integer(genotype, "genotype", 2)
    source, desired = [tuple(k) for k in source_keys], [tuple(k) for k in desired_keys]
    if (any(len(key) != 2 or not isinstance(key[0], str) or not key[0]
            or type(key[1]) is not int or key[1] not in (0, 1) for key in source + desired)
            or len(source) != g.shape[1] or len(desired) != len(source)
            or len(set(source)) != len(source) or len(set(desired)) != len(desired)
            or set(source) != set(desired)):
        raise ValueError("Unique complete matching observed sample ID/haplotype keys required")
    at = {key: col for col, key in enumerate(source)}
    return g[:, [at[key] for key in desired]].copy()


def _layout(tree):
    p = _integer(tree.parent, "parent").astype(np.int64, copy=False)
    t = np.asarray(tree.node_time, dtype=np.float64)
    sn = _integer(tree.sample_node, "sample_node").astype(np.int64, copy=False)
    sid = _integer(tree.sample_id, "sample_id").astype(np.int64, copy=False)
    n = len(p)
    if (not n or t.shape != (n,) or not np.isfinite(t).all() or np.any(t < 0)
            or np.any(p < -1) or np.any(p >= n) or np.count_nonzero(p == -1) != 1
            or not len(sn) or sid.shape != sn.shape or np.any(sn < 0) or np.any(sn >= n)
            or np.any(sid < 0) or len(set(sn.tolist())) != len(sn) or len(set(sid.tolist())) != len(sid)):
        raise ValueError("One complete observed native tree and unique sampled identities required")
    edge = np.flatnonzero(p >= 0)
    duration = np.zeros(n, dtype=np.float64)
    duration[edge] = t[p[edge]] - t[edge]
    if np.any(duration[edge] <= 0):
        raise ValueError("Strictly positive native endpoint durations required")
    children = [[] for _ in range(n)]
    for child in edge:
        children[p[child]].append(int(child))
    root = int(np.flatnonzero(p == -1)[0])
    preorder, stack = [], [root]
    depth = np.zeros(n, dtype=np.int64)
    while stack:
        node = stack.pop(); preorder.append(node)
        for child in children[node]:
            depth[child] = depth[node] + 1
        stack.extend(children[node])
    if len(preorder) != n:
        raise ValueError("Disconnected or cyclic administrative tree rows are forbidden")
    postorder = np.asarray(preorder[::-1], dtype=np.int64)
    descendant = np.zeros(n, dtype=np.int64); descendant[sn] = 1
    height = np.zeros(n, dtype=np.int64)
    for node in postorder:
        if p[node] >= 0:
            descendant[p[node]] += descendant[node]
            height[p[node]] = max(height[p[node]], height[node] + 1)
    if np.any(descendant == 0):
        raise ValueError("Only the complete sampled-ancestor union may be retained")
    return p, t, sn, sid, postorder, depth, height, descendant, duration


def _validate_alleles(genotype, allele_count):
    g = _integer(genotype, "genotype", 2)
    k = _integer(allele_count, "allele_count")
    if (k.shape != (len(g),) or np.any(k < 1) or np.any(k > 4)
            or np.any(g < -1) or np.any(g >= k[:, None])):
        raise ValueError("Original categorical nucleotide alleles1..4 and -1 missing calls required")
    return g, k


def observed_allele_frequency_arrays_v1(genotype, allele_count, reference_class,
        sample_age, *, alpha=0.5, age_split=350.):
    """Compact raw frequencies with H028 common-cohort donor callability.

    Eight channels are younger ANA..EHG then older ANA..EHG. A globally absent
    group remains masked. If any globally present class in a cohort has no
    calls at a site, all present classes of that cohort are unavailable there.
    Missing leaf calls are masked by the model before its tuple MLP.
    """
    g, k = _validate_alleles(genotype, allele_count)
    ref = _integer(reference_class, "reference_class")
    age = np.asarray(sample_age, dtype=np.float64)
    if (ref.shape != (g.shape[1],) or np.any(ref < -1) or np.any(ref > 3)
            or age.shape != ref.shape or not np.isfinite(age).all() or np.any(age < 0)
            or alpha != 0.5 or age_split != 350.):
        raise ValueError("Observed pure proxy labels/ages, alpha0.5 and350 age groups required")
    s, amax = len(g), int(k.max(initial=1))
    logfreq = np.zeros((s, amax, 8), dtype=np.float32)
    mask = np.zeros((s, 8), dtype=bool)
    class_counts = np.zeros(8, dtype=np.int64)
    allele_valid = np.arange(amax)[None, :] < k[:, None]
    for cohort in range(2):
        age_group = age < age_split if cohort == 0 else age >= age_split
        donors, calls = {}, {}
        for cls in range(4):
            channel = 4 * cohort + cls
            columns = np.flatnonzero(age_group & (ref == cls))
            class_counts[channel] = len(columns)
            if len(columns):
                donors[channel] = g[:, columns]
                calls[channel] = np.count_nonzero(donors[channel] >= 0, axis=1)
        common = np.ones(s, dtype=bool)
        for called in calls.values():
            common &= called > 0
        for channel, donor in donors.items():
            count = np.stack([np.count_nonzero(donor == a, axis=1) for a in range(amax)], axis=1)
            prob = (count + alpha) / (calls[channel] + k * alpha)[:, None]
            active = common[:, None] & allele_valid
            logfreq[:, :, channel] = np.where(active, np.log(prob), 0.)
            mask[:, channel] = common
    called = np.count_nonzero(g >= 0, axis=1)
    count = np.stack([np.count_nonzero(g == a, axis=1) for a in range(amax)], axis=1)
    panel = np.zeros((s, amax), dtype=np.float32)
    np.divide(count, called[:, None], out=panel, where=called[:, None] > 0)
    entropy = -(panel * np.log(np.where(panel > 0, panel, 1.))).sum(1).astype(np.float32)
    return logfreq, mask, panel, entropy, class_counts


def branch_concordance_support_v1(tree, genotype, allele_count, sample_ids):
    """Five branch summaries from descendant allele-count sweeps, no pair matrix.

    Every observed nonempty proper allele split is counted, including both
    complementary biallelic splits. The symmetric discrepancy has no ancestral
    polarization. Minimum-discrepancy ties include native unary branches.
    """
    g, k = _validate_alleles(genotype, allele_count)
    sample_ids = _integer(sample_ids, "sample_ids")
    p, _, sn, sid, postorder, _, _, _, _ = _layout(tree)
    if (g.shape[1] != len(sample_ids) or len(set(sample_ids.tolist())) != len(sample_ids)
            or set(sample_ids.tolist()) != set(sid.tolist())):
        raise ValueError("All and only actual sampled haplotypes must join genotype columns")
    col = {identity: at for at, identity in enumerate(sample_ids.tolist())}
    sample_column = np.asarray([col[identity] for identity in sid], dtype=np.int64)
    n = len(p); branches = np.flatnonzero(p >= 0)
    out = np.zeros((n, 5), dtype=np.float64)
    splits = 0
    for site in range(len(g)):
        value = g[site, sample_column]
        called = value >= 0; ncalled = int(np.count_nonzero(called))
        if ncalled < 2:
            continue
        total = np.bincount(value[called], minlength=int(k[site]))
        proper = np.flatnonzero((total > 0) & (total < ncalled))
        if not len(proper):
            continue
        descendant_alleles = np.zeros((n, int(k[site])), dtype=np.int64)
        descendant_alleles[sn[called], value[called]] = 1
        for node in postorder:
            if p[node] >= 0:
                descendant_alleles[p[node]] += descendant_alleles[node]
        d = descendant_alleles.sum(1)[branches]
        for allele in proper:
            difference = d + int(total[allele]) - 2 * descendant_alleles[branches, allele]
            mismatch = np.minimum(difference, ncalled - difference)
            discrepancy = mismatch.astype(np.float64) / ncalled
            minimum = int(mismatch.min())
            winners = branches[mismatch == minimum]
            if minimum == 0:
                out[winners, 0] += 1. / len(winners)
            out[winners, 1] += (1. - minimum / ncalled) / len(winners)
            out[branches, 2] += discrepancy
            splits += 1
    if splits:
        out[branches, :3] /= splits
        out[branches, 3] = splits
    else:
        out[branches, 4] = 1.
    out[p < 0, 4] = 1.
    return out.astype(np.float32)


def observed_sham_geometry_v1(trees):
    """Observed nonroot duration/count arrays for freezing TRAIN bins."""
    durations, descendants = [], []
    for tree in trees:
        p, _, _, _, _, _, _, count, duration = _layout(tree)
        durations.append(duration[p >= 0]); descendants.append(count[p >= 0])
    if not durations:
        raise ValueError("Observed TRAIN tree geometry is required")
    return np.concatenate(durations), np.concatenate(descendants)


def fit_sham_bin_edges_v1(duration, descendant_count, *, split, bins=4):
    """Fit one immutable stratification from target-free TRAIN geometry."""
    d, c = np.asarray(duration, dtype=np.float64), _integer(descendant_count, "descendant_count")
    if (split != "TRAIN" or d.ndim != 1 or not len(d) or c.shape != d.shape
            or not np.isfinite(d).all() or np.any(d <= 0) or np.any(c < 1)
            or type(bins) is not int or bins < 1):
        raise ValueError("Nonempty observed TRAIN branch geometry and positive bin count required")
    q = np.arange(1, bins, dtype=np.float64) / bins
    duration_edges = tuple(np.unique(np.quantile(np.log1p(d), q)).tolist())
    count_edges = tuple(np.unique(np.quantile(c, q)).tolist())
    return ShamBinEdgesV1(duration_edges, count_edges, len(d), bins)


def _validate_sham_bins(edges):
    if (not isinstance(edges, ShamBinEdgesV1) or edges.training_branch_count < 1
            or edges.bins_requested < 1 or edges.provenance != "TRAIN_observed_native_geometry_before_targets"):
        raise ValueError("Frozen TRAIN-observation bin edges are required")
    for values in (edges.log_duration_edges, edges.descendant_count_edges):
        a = np.asarray(values, dtype=np.float64)
        if a.ndim != 1 or not np.isfinite(a).all() or np.any(np.diff(a) <= 0):
            raise ValueError("Finite strictly increasing frozen sham bin edges required")


def shuffle_branch_support_v1(parent, support, duration, descendant_count,
        node_hold, spans, readout_hold, bin_edges, *, seed=101):
    """One fixed within-tree stratified whole-vector derangement; no reseeding."""
    _validate_sham_bins(bin_edges)
    p, count, hold = (_integer(value, name) for value, name in
                      ((parent, "parent"), (descendant_count, "descendant_count"), (node_hold, "node_hold")))
    value = np.asarray(support, dtype=np.float32)
    duration = np.asarray(duration, dtype=np.float64); span = np.asarray(spans, dtype=np.float64)
    readout = _integer(readout_hold, "readout_hold")
    n = len(p)
    if (value.shape != (n, 5) or duration.shape != (n,) or count.shape != (n,)
            or hold.shape != (n,) or np.any(count < 1) or not np.isfinite(value).all() or np.any(value < 0)
            or not np.isfinite(duration).all() or np.any(duration[p >= 0] <= 0)
            or span.ndim != 1 or not len(span) or np.any(span <= 0) or not np.isfinite(span).all()
            or np.any(hold < 0) or np.any(hold >= len(span)) or np.any(readout < 0)
            or np.any(readout >= len(span)) or type(seed) is not int or seed != 101):
        raise ValueError("Fixed seed101 valid observed forest/support geometry required")
    rng = np.random.default_rng(seed)
    permutation = np.arange(n, dtype=np.int64)
    strata = {}
    for node in np.flatnonzero(p >= 0):
        db = int(np.searchsorted(bin_edges.log_duration_edges, np.log1p(duration[node]), side="right"))
        cb = int(np.searchsorted(bin_edges.descendant_count_edges, count[node], side="right"))
        strata.setdefault((int(hold[node]), db, cb), []).append(int(node))
    eligible_strata = 0
    for key in sorted(strata):
        members = np.asarray(strata[key], dtype=np.int64)
        if len(members) >= 4:
            order = rng.permutation(members)
            permutation[order] = np.roll(order, 1)
            eligible_strata += 1
    sham = value[permutation].copy()
    branches = p >= 0; assignment_moved = branches & (permutation != np.arange(n))
    changed = value != sham
    nonzero = branches[:, None] & (value != 0)
    supported = branches & np.any(value[:, :3] != 0, axis=1)
    supported_moved = supported & np.any(changed[:, :3], axis=1)
    physical = span[hold]
    readout_weight = np.bincount(readout, minlength=len(span)).astype(np.float64)[hold]
    def fraction(numerator, denominator, weight=None):
        w = np.ones(n, dtype=np.float64) if weight is None else weight
        den = float(w[denominator].sum())
        return float(w[numerator].sum() / den) if den else None
    def field_fraction(weight):
        w = weight[:, None]
        den = float((nonzero * w).sum())
        return float((nonzero * changed * w).sum() / den) if den else None
    constant_fields = [SUPPORT_FIELDS[f] for f in range(5)
        if all(len(np.unique(value[(hold == h) & branches, f])) <= 1 for h in range(len(span)))]
    moved_fraction = fraction(supported_moved, supported)
    metadata = dict(seed=seed, favorable_reseeding=False, minimum_stratum_size=4,
        eligible_strata=eligible_strata, total_strata=len(strata),
        branch_count=int(branches.sum()), assignment_moved_branches=int(assignment_moved.sum()),
        unmoved_branches=int((branches & ~assignment_moved).sum()),
        assignment_moved_fraction=fraction(assignment_moved, branches),
        supported_vectors=int(supported.sum()), supported_vectors_numerically_moved=int(supported_moved.sum()),
        supported_vector_moved_fraction=moved_fraction,
        physical_weighted_supported_vector_moved_fraction=fraction(supported_moved, supported, physical),
        readout_query_weighted_supported_vector_moved_fraction=fraction(supported_moved, supported, readout_weight),
        nonzero_field_moved_fraction=field_fraction(np.ones(n)),
        physical_weighted_nonzero_field_moved_fraction=field_fraction(physical),
        readout_query_weighted_nonzero_field_moved_fraction=field_fraction(readout_weight),
        per_field_nonzero_count=nonzero.sum(0).tolist(),
        per_field_moved_nonzero_count=(nonzero & changed).sum(0).tolist(),
        null_constant_fields=constant_fields,
        mechanism_inconclusive=moved_fraction is None or moved_fraction < 0.5,
        bin_edges=dict(log_duration_edges=list(bin_edges.log_duration_edges),
            descendant_count_edges=list(bin_edges.descendant_count_edges),
            training_branch_count=bin_edges.training_branch_count,
            bins_requested=bin_edges.bins_requested, provenance=bin_edges.provenance))
    return sham, metadata


def _level_arrays(level):
    buckets = [[] for _ in range(int(level.max(initial=0)) + 1)]
    for node, rank in enumerate(level):
        buckets[int(rank)].append(node)
    return tuple(np.asarray(nodes, dtype=np.int64) for nodes in buckets)


def prepare_mutation_routed_tree_observations_v1(trees: Sequence[NativeObservedTreeV1],
        intervals_bp, positions_bp0, genotype, allele_count, sample_ids, reference_class,
        sample_age, query_ids, readout_x_bp, *, sham_bin_edges: ShamBinEdgesV1,
        readout_bin_edges_bp=None, fixture=False):
    """Prepare one complete parent from observed input arrays, before targets.

    TRAIN bins must already be frozen independently, using observed geometry
    from all designated TRAIN parents. This function applies them unchanged.
    Original positions and clipped scoring bin edges are receipt metadata.
    """
    _validate_sham_bins(sham_bin_edges)
    g, k = _validate_alleles(genotype, allele_count)
    sid = _integer(sample_ids, "sample_ids").astype(np.int64, copy=False)
    ref = _integer(reference_class, "reference_class").astype(np.int64, copy=False)
    qi = _integer(query_ids, "query_ids").astype(np.int64, copy=False)
    age = np.asarray(sample_age, dtype=np.float64)
    x = np.asarray(positions_bp0, dtype=np.float64)
    intervals = np.asarray(intervals_bp, dtype=np.float64)
    readout = np.asarray(readout_x_bp, dtype=np.float64)
    h = g.shape[1]
    if (not trees or not h or not len(qi) or intervals.shape != (len(trees), 2) or not np.isfinite(intervals).all()
            or np.any(intervals[:, 1] <= intervals[:, 0]) or intervals[0, 0] < 0
            or not np.array_equal(intervals[:-1, 1], intervals[1:, 0])
            or sid.shape != (h,) or np.any(sid < 0) or len(set(sid.tolist())) != h
            or ref.shape != (h,) or np.any(ref < -1) or np.any(ref > 3)
            or age.shape != (h,) or not np.isfinite(age).all() or np.any(age < 0)
            or len(set(qi.tolist())) != len(qi) or not set(qi.tolist()).issubset(set(sid.tolist()))
            or x.shape != (len(g),) or not np.isfinite(x).all() or np.any(x < 0) or np.any(np.diff(x) <= 0)
            or readout.ndim != 1 or not len(readout) or not np.isfinite(readout).all()
            or np.any(np.diff(readout) <= 0) or np.any(readout < intervals[0, 0])
            or np.any(readout >= intervals[-1, 1])):
        raise ValueError("Complete ordered observed forest, persistent roster and physical readouts required")
    column = {identity: at for at, identity in enumerate(sid.tolist())}
    qc = np.asarray([column[identity] for identity in qi], dtype=np.int64)
    if np.any(ref[qc] != -1):
        raise ValueError("Actual query columns must have unknown input ancestry")
    if not fixture and (h != 280 or len(qi) != 120 or np.count_nonzero(ref >= 0) != 96
            or [np.count_nonzero(ref == c) for c in range(4)] != [24] * 4):
        raise ValueError("Production requires280 observed haps, actualQ120 and only pureR96 labels")
    if readout_bin_edges_bp is not None:
        edges = np.asarray(readout_bin_edges_bp, dtype=np.float64)
        if (edges.shape != (len(readout) + 1,) or not np.isfinite(edges).all()
                or np.any(np.diff(edges) <= 0) or edges[0] != intervals[0, 0]
                or edges[-1] != intervals[-1, 1] or np.any(readout < edges[:-1])
                or np.any(readout >= edges[1:])):
            raise ValueError("Preserve all original clipped readout bins and containing centers")
    inside = (x >= intervals[0, 0]) & (x < intervals[-1, 1])
    covered_g, covered_k, covered_x = g[inside], k[inside], x[inside]
    site_hold = np.searchsorted(intervals[:, 1], covered_x, side="right").astype(np.int64)
    readout_hold = np.searchsorted(intervals[:, 1], readout, side="right").astype(np.int64)
    parents, times, sample_rows, support, heights, depths, holds, counts, durations = [], [], [], [], [], [], [], [], []
    offset = 0
    for hold, tree in enumerate(trees):
        p, t, sn, tsid, _, depth, height, descendant, duration = _layout(tree)
        if set(tsid.tolist()) != set(sid.tolist()):
            raise ValueError("Each native hold must retain all280 persistent observed sampled haps")
        native_row = dict(zip(tsid.tolist(), sn.tolist()))
        rows = np.asarray([native_row[identity] for identity in sid], dtype=np.int64)
        if not np.array_equal(t[rows], age):
            raise ValueError("Observed sample dates changed in a native hold")
        global_parent = p.copy(); global_parent[p >= 0] += offset
        parents.append(global_parent); times.append(t); sample_rows.append(rows + offset)
        lo, hi = np.searchsorted(site_hold, [hold, hold + 1])
        support.append(branch_concordance_support_v1(tree, covered_g[lo:hi], covered_k[lo:hi], sid))
        heights.append(height); depths.append(depth); holds.append(np.full(len(p), hold, dtype=np.int64))
        counts.append(descendant); durations.append(duration); offset += len(p)
    parent = np.concatenate(parents); node_time = np.concatenate(times)
    height, depth = np.concatenate(heights), np.concatenate(depths)
    node_hold, descendant, duration = np.concatenate(holds), np.concatenate(counts), np.concatenate(durations)
    true = np.concatenate(support)
    spans = intervals[:, 1] - intervals[:, 0]
    geometry = np.zeros((len(parent), 4), dtype=np.float32)
    child = np.flatnonzero(parent >= 0)
    geometry[child, :3] = np.stack((np.log1p(node_time[child]),
        np.log1p(node_time[parent[child]]), np.log1p(duration[child])), axis=1) / math.log(10001.)
    geometry[child, 3] = descendant[child] / h
    levels = _level_arrays(height)
    rank = np.empty(len(parent), dtype=np.int64)
    for nodes in levels:
        rank[nodes] = np.arange(len(nodes))
    child_groups = [[] for _ in levels]
    for node in child:
        child_groups[int(height[parent[node]])].append(int(node))
    upward_children = tuple(np.asarray(nodes, dtype=np.int64) for nodes in child_groups)
    upward_parent_rank = tuple(rank[parent[nodes]] for nodes in upward_children)
    downward_levels = _level_arrays(depth)[1:]
    sham, sham_report = shuffle_branch_support_v1(parent, true, duration, descendant,
        node_hold, spans, readout_hold, sham_bin_edges)
    logfreq, callmask, panel, entropy, group_count = observed_allele_frequency_arrays_v1(
        covered_g, covered_k, ref, age)
    metadata = dict(schema="mutation_routed_tree_observations_v1", input_only=True,
        targets_read=False, native_unary_branches_preserved=True,
        raw_chronology_validated_cpu_float64=True,
        raw_chronology_policy="strict_native_parent_age_greater_than_child_age_before_cast",
        edge_duration_source="CPU_float64_native_endpoint_difference_before_float32_feature_cast",
        rounded_node_clock_policy="FP32_parent_child_ties_allowed_only_with_positive_raw_duration_feature",
        all_observed_samples=h, known_pure_source_samples=int(np.count_nonzero(ref >= 0)),
        unknown_input_ancestry_samples=int(np.count_nonzero(ref < 0)), actual_query_readout_columns=len(qc),
        genuine_holds=len(trees), vertices=len(parent), native_nonroot_branches=len(child),
        original_sites=len(g), sites_in_observed_coverage=len(covered_g),
        multiallelic_sites_in_coverage=int(np.count_nonzero(covered_k > 2)),
        sites_excluded_only_outside_coverage=int(np.count_nonzero(~inside)),
        empty_original_holds=int(np.count_nonzero(np.bincount(site_hold, minlength=len(trees)) == 0)),
        holds_without_scoring_centers=int(np.count_nonzero(np.bincount(readout_hold, minlength=len(trees)) == 0)),
        coverage_bp0=intervals[[0, -1], [0, 1]].tolist(),
        intervals_bp0=intervals.tolist(), readout_positions_bp0=readout.tolist(),
        readout_bin_edges_bp0=None if readout_bin_edges_bp is None else edges.tolist(),
        support_fields=list(SUPPORT_FIELDS), donor_group_counts=group_count.tolist(),
        allele_index_embedding=False, ancestral_polarization=False,
        internal_node_identity_feature=False, query_role_feature=False,
        sample_ids_policy="unique_actual_observed_join_keys_only", sham=sham_report)
    return MutationRoutedTreeObservationsV1(parent, node_time, np.stack(sample_rows), levels,
        upward_children, upward_parent_rank, downward_levels, geometry, true, sham,
        covered_g.astype(np.int16, copy=True), logfreq, callmask, panel, entropy, site_hold,
        spans, readout_hold, qc, ref.copy(), age.copy(), metadata)
