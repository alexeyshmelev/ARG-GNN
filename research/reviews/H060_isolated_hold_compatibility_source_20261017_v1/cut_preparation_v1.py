"""Original 350-generation cut algebra; only exact isolated hold ownership differs."""
from research.reviews.order2_cut_memory_implementation_20261015_v1 import order2_cut_memory_v1 as _original
from .structural_index_v1 import classify_sample_roots,validate_root_partition
np=_original.np
torch=_original.torch
math=_original.math
replace=_original.replace
_require=_original._require
_BIND_FIELDS=_original._BIND_FIELDS
HORIZON=_original.HORIZON
_direction_layouts=_original._direction_layouts
_fingerprint=_original._fingerprint
_stamps=_original._stamps
_bank_stamps=_original._bank_stamps
CutPlanV1=_original.CutPlanV1

def prepare_cut_plan_v1(bank, *, expected_haplotypes=952,
                        max_nodes=10000000, max_holds=10000,isolated_hold_indices=()):
    """CPU observation-only preparation. Small H is allowed for handmade checks.

    Production bundle preparation separately requires all952/686 original rows.
    No scientific file is opened here. Fingerprints cover only supplied tensors.
    """
    if int((bank.parent<0).sum())==len(bank.sample_index):
        _require(tuple(isolated_hold_indices)==(),"Connected input has no isolated-hold declaration")
        return _original.prepare_cut_plan_v1(bank,expected_haplotypes=expected_haplotypes,
            max_nodes=max_nodes,max_holds=max_holds)
    xs = tuple(getattr(bank, n) for n in _BIND_FIELDS)
    _require(all(isinstance(x, torch.Tensor) and x.device.type == "cpu"
                 and not x.requires_grad for x in xs), "CPU fixed observed cut inputs")
    _require(bank.parent.dtype == bank.sample_index.dtype == torch.long
             and bank.node_time.dtype == bank.sample_age.dtype == bank.spans.dtype == torch.float32,
             "Original native index and FP32 date/span types")
    p = bank.parent.detach().numpy()
    times = bank.node_time.detach().numpy()
    samples = bank.sample_index.detach().numpy()
    ages = bank.sample_age.detach().numpy()
    spans = bank.spans.detach().numpy()
    _require(p.ndim == 1 and samples.ndim == 2, "Packed native forest and sample table")
    n, (t, h) = len(p), samples.shape
    _require(0 < n <= max_nodes and 0 < t <= max_holds and h == expected_haplotypes
             and times.shape == (n,) and ages.shape == (h,) and spans.shape == (t,),
             "Explicit bounded complete native dimensions")
    _require(np.isfinite(times).all() and np.isfinite(ages).all()
             and np.isfinite(spans).all() and (times >= 0).all() and (ages >= 0).all()
             and (spans > 0).all() and (p >= -1).all() and (p < n).all()
             and (samples >= 0).all() and (samples < n).all()
             and np.unique(samples).size == t*h, "Finite dates, positive spans and unique native leaves")
    nonroot = p >= 0
    _require((times[p[nonroot]] >= times[nonroot]).all(), "Monotone observed parent dates")
    roots = np.flatnonzero(~nonroot)
    _require(roots.size >= t, "Each native hold has its real rooted components")
    degree = np.bincount(p[nonroot], minlength=n)
    leaf_mask = np.zeros(n, dtype=bool)
    leaf_mask[samples.reshape(-1)] = True
    _require(np.array_equal(leaf_mask, degree == 0), "All and only declared sampled leaves")
    _require(np.array_equal(times[samples], np.broadcast_to(ages, (t, h))),
             "Actual sampled dates persist identically across holds")

    head = np.full(n, -1, dtype=np.int64)
    sibling = np.full(n, -1, dtype=np.int64)
    for child in np.flatnonzero(nonroot):
        parent = int(p[child])
        sibling[child] = head[parent]
        head[parent] = child
    owner = np.full(n, -1, dtype=np.int32)
    cut = np.full(n, -1, dtype=np.int64)
    for tree_id, root in enumerate(roots.tolist()):
        stack = [root]
        while stack:
            u = stack.pop()
            _require(owner[u] < 0, "No repeated/cyclic native vertex")
            owner[u] = tree_id
            parent = int(p[u])
            if times[u] <= HORIZON:
                cut[u] = u if parent < 0 or times[parent] > HORIZON else cut[parent]
                _require(cut[u] >= 0, "Highest real at-or-below-cut ancestor exists")
            child = int(head[u])
            while child >= 0:
                stack.append(child)
                child = int(sibling[child])
    _require((owner >= 0).all(), "No rootless cycle or disconnected hidden component")
    ordered=[classify_sample_roots(p,head,roots[owner],row)for row in samples]
    validate_root_partition(ordered,roots,isolated_hold_indices)
    for at in isolated_hold_indices:
        _require(not bool((bank.site_hold==at).any()),"Isolated hold must have zero original SNPs")
    representatives, group_counts, root_flags = [], [], []
    offsets = [0]
    local_groups = np.empty((t, h), dtype=np.int64)
    global_groups = np.empty((t, h), dtype=np.int64)
    above = ages > HORIZON
    for at in range(t):
        lookup, reps = {}, []
        for i, node in enumerate(samples[at].tolist()):
            representative = int(node if above[i] else cut[node])
            if representative not in lookup:
                lookup[representative] = len(reps)
                reps.append(representative)
            local_groups[at, i] = lookup[representative]
        k = len(reps)
        global_groups[at] = local_groups[at] + offsets[-1]
        representatives.extend(reps)
        group_counts.extend(np.bincount(local_groups[at], minlength=k).tolist())
        root_flags.extend(bool(p[u] < 0 and times[u] <= HORIZON) for u in reps)
        offsets.append(offsets[-1] + k)
    _require(len(set(representatives)) == len(representatives),
             "One unique actual representative per native cut group")
    episode_groups, episode_of_hold, starts = [], [], []
    for at in range(t):
        if not episode_groups or not np.array_equal(local_groups[at], episode_groups[-1]):
            episode_groups.append(local_groups[at].copy())
            starts.append(at)
        episode_of_hold.append(len(episode_groups) - 1)
    g = len(episode_groups)
    ep = np.stack(episode_groups)
    ep_counts = np.zeros((g, h), dtype=np.float32)
    for e, row in enumerate(ep):
        k = int(row.max()) + 1
        ep_counts[e, :k] = np.bincount(row, minlength=k)
    forward = _direction_layouts(episode_groups, range(g), h)
    reverse = _direction_layouts(episode_groups, range(g-1, -1, -1), h)
    fingerprints = tuple((name, _fingerprint(getattr(bank, name))) for name in _BIND_FIELDS)
    def tensor(a, dtype=None):
        return torch.as_tensor(np.asarray(a), dtype=dtype).contiguous()
    plan = CutPlanV1(
        tensor(representatives, torch.long), tensor(global_groups, torch.long),
        tensor(group_counts, torch.float32), tensor(ep, torch.long), tensor(ep_counts),
        *(tensor(x) for x in forward), *(tensor(x) for x in reverse),
        tensor(above, torch.bool), tensor(root_flags, torch.bool),
        tuple(offsets), tuple(episode_of_hold), t, h, n, fingerprints, _bank_stamps(bank), ())
    plan = replace(plan, tensor_stamps=_stamps(plan.tensors()))
    plan.validate(bank)
    # Truth-free opportunity metadata only; no outcome or ancestry metric.
    dependence = []
    for e in range(2, g):
        a, b, c = episode_groups[e-2:e+1]
        triples, ab, bc, nb = {}, {}, {}, {}
        for x, y, z in zip(a.tolist(), b.tolist(), c.tolist()):
            triples[(x, y, z)] = triples.get((x, y, z), 0) + 1
            ab[(x, y)] = ab.get((x, y), 0) + 1
            bc[(y, z)] = bc.get((y, z), 0) + 1
            nb[y] = nb.get(y, 0) + 1
        information = sum(k/h * math.log(k*nb[y]/(ab[(x,y)]*bc[(y,z)]))
                          for (x,y,z), k in triples.items())
        dependence.append(float(information))
    metadata = dict(
        schema="order2_observed_cut_plan_v1", horizon_generations=HORIZON,
        native_holds=t, haplotypes=h, native_nodes=n, cut_episodes=g, cut_change_events=g-1,
        native_hold_cut_group_counts=np.diff(offsets).tolist(), episode_starts=starts,
        above_horizon_actual_samples=int(above.sum()),
        root_saturated_group_occurrences=int(sum(root_flags)),
        max_forward_pair_cells=int((forward[3] > 0).sum(1).max()),
        max_reverse_pair_cells=int((reverse[3] > 0).sum(1).max()),
        conditional_information_nats_per_leaf=dependence,
        cut_plan_tensor_bytes=plan.tensor_bytes(), source_fingerprints=dict(fingerprints),
        grouping_uses_query_ancestry=False, ancestral_identity_claim=False,
        sample_age_policy="<=350 real cut; >350 actual-sample singleton",
        unchanged_cut_native_hold_reprojection=False)
    return plan, metadata
