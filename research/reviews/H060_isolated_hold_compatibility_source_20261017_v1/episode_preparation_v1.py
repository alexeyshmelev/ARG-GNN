"""Original proper-clade episode recipe with exact isolated native-root bookkeeping."""
from arggnn import ordered_clade_episode_gnn_v1 as _episodes
from arggnn.anonymous_exact_clade_transport_v1 import _forest_index as _connected_forest_index
from arggnn.anonymous_clade_flag_control_v1 import _tree_clades as _connected_tree_clades
from .structural_index_v1 import classify_sample_roots,validate_root_partition,isolated_clades
np=_episodes.np
_require=_episodes._require
_sha=_episodes._sha
_span_denominators=_episodes._span_denominators
DEFAULT_PAYLOAD_CAP=_episodes.DEFAULT_PAYLOAD_CAP
FLAG_SCHEMA=_episodes.FLAG_SCHEMA
CHANNELS=_episodes.CHANNELS
SCHEMA=_episodes.SCHEMA
def _tree_clades(root,row,parent,head,sibling):
    if isinstance(root,tuple):return isolated_clades(root,row,parent,head)
    return _connected_tree_clades(root,row,parent,head,sibling)

def _forest_index(parent, samples, *, isolated_hold_indices):
    """Memoized root discovery: every parent link is followed at most once."""
    count = parent.size
    head = np.full(count, -1, dtype=np.int32)
    sibling = np.full(count, -1, dtype=np.int32)
    for node, ancestor in enumerate(parent):
        if ancestor >= 0:
            sibling[node] = head[ancestor]
            head[ancestor] = node
    roots = np.full(count, -2, dtype=np.int32)
    parent_steps = 0
    for start in range(count):
        if roots[start] != -2:
            continue
        trail, node = [], start
        while roots[node] == -2 and parent[node] >= 0:
            roots[node] = -3
            trail.append(node)
            node = int(parent[node])
            parent_steps += 1
        if parent[node] < 0:
            root = node
            roots[node] = root
        else:
            _require(roots[node] >= 0, "Acyclic observed parent forest required")
            root = int(roots[node])
        for node in reversed(trail):
            roots[node] = root
    ordered=[classify_sample_roots(parent,head,roots,row)for row in samples]
    validate_root_partition(ordered,np.flatnonzero(parent<0),isolated_hold_indices)
    return head,sibling,ordered,parent_steps

def build_clade_episode_arrays_v1(
        parent, sample_index, spans, *, max_payload_bytes=DEFAULT_PAYLOAD_CAP,isolated_hold_indices=()):
    """Bounded CPU topology/span preparation, permitting small synthetic panels.

    Stable sampled-leaf columns are an upstream input contract. Every proper
    clade occurrence is represented, but only episodes of length >=2 inject.
    Arrays are bounded by 18*N+4*T bytes before construction. This is a compact
    array allocation bound, not a total Python/process-memory resource proof.
    """
    if np.count_nonzero(np.asarray(parent)<0)==len(sample_index):
        _require(tuple(isolated_hold_indices)==(),"Connected input has no isolated-hold declaration")
        return _episodes.build_clade_episode_arrays_v1(parent,sample_index,spans,max_payload_bytes=max_payload_bytes)
    parent, samples, spans = np.asarray(parent), np.asarray(sample_index), np.asarray(spans)
    _require(parent.ndim == 1 and 0 < parent.size < 2**31 and parent.dtype.kind == "i"
             and not np.any(parent < -1) and not np.any(parent >= parent.size),
             "Bounded signed-integer observed parent forest required")
    _require(samples.ndim == 2 and samples.dtype.kind == "i"
             and 0 < samples.shape[0] < 2**31 and 3 <= samples.shape[1] < 2**31
             and not np.any(samples < 0) and not np.any(samples >= parent.size),
             "Complete stable sampled-leaf rows required")
    n, (t, h) = parent.size, samples.shape
    _require(spans.dtype == np.float32 and spans.shape == (t,)
             and np.isfinite(spans).all() and np.all(spans > 0),
             "One finite positive FP32 physical span per native tree required")
    allocation_bound = 18 * n + 4 * t
    _require(type(max_payload_bytes) is int and allocation_bound <= max_payload_bytes,
             "Explicit payload cap must cover conservative 18*N+4*T array allocation")
    head, sibling, roots, parent_steps = _forest_index(parent, samples,isolated_hold_indices=isolated_hold_indices)
    flags = np.zeros((n, 2), dtype=np.uint8)
    # Every member and every episode owns at least one distinct native vertex.
    # Fixed bounded buffers avoid an unbounded Python collection of episodes.
    nodes = np.empty(n, dtype=np.int32)
    holds = np.empty(n, dtype=np.int32)
    episodes = np.empty(n, dtype=np.int32)
    sizes = np.zeros(n, dtype=np.int32)
    previous, previous_episode = None, {}
    members = episode_count = visited = 0
    clade_counts, node_counts, unary_counts = [], [], []
    for hold, (root, row) in enumerate(zip(roots, samples)):
        current, count, unary = _tree_clades(root, row, parent, head, sibling)
        visited += count
        clade_counts.append(len(current))
        node_counts.append(count)
        unary_counts.append(unary)
        if previous is not None:
            for mask, node in previous.items():
                if mask not in current:
                    flags[node, 1] = 1
            for mask, node in current.items():
                if mask not in previous:
                    flags[node, 0] = 1
        current_episode = {}
        for mask, node in current.items():
            _require(members < n, "At most one proper-clade membership per native vertex")
            episode = previous_episode.get(mask)
            if episode is None:
                _require(episode_count < n, "Episode buffer bound exceeded")
                episode = episode_count
                episode_count += 1
            nodes[members], holds[members], episodes[members] = node, hold, episode
            sizes[episode] += 1
            current_episode[mask] = episode
            members += 1
        previous, previous_episode = current, current_episode
    _require(visited == n, "Every original vertex, including unaries, must be retained")
    del previous, previous_episode, current, current_episode
    # Shrinking owned buffers does not create full-size duplicate payloads.
    for array in (nodes, holds, episodes):
        array.resize((members,), refcheck=False)
    sizes.resize((episode_count,), refcheck=False)
    arrays = dict(
        node_flags=flags, member_node=nodes, member_hold=holds,
        member_episode=episodes, episode_size=sizes, hold_span=spans.copy(),
    )
    eligible = _span_denominators(holds, episodes, sizes, spans)
    flag_meta = dict(
        schema=FLAG_SCHEMA, channels=list(CHANNELS), native_holds=int(t),
        haplotypes=int(h), native_nodes=int(n), node_flags_sha256=_sha(flags),
        exact_bitsets_retained=False, internal_ancestor_correspondence_claimed=False,
        labels_times_targets_or_outcomes_used=False,
    )
    meta = dict(
        schema=SCHEMA, native_holds=int(t), haplotypes=int(h), native_nodes=int(n),
        member_count=int(members), episode_count=int(episode_count),
        eligible_member_count=int(eligible.sum()),
        eligible_episode_count=int(np.count_nonzero(sizes >= 2)),
        singleton_episode_count=int(np.count_nonzero(sizes == 1)),
        array_sha256={key: _sha(value) for key, value in arrays.items()},
        topology_sha256=_sha(parent), sample_index_sha256=_sha(samples),
        span_sha256=_sha(spans), payload_bytes=sum(a.nbytes for a in arrays.values()),
        array_allocation_bound_bytes=int(allocation_bound),
        max_payload_bytes=max_payload_bytes, parent_links_followed=int(parent_steps),
        proper_clades_per_tree=clade_counts, native_nodes_per_tree=node_counts,
        unary_equivalences_per_tree=unary_counts,
        episode_policy="maximal_consecutive_trees_exact_sampled_descendant_set",
        representative_policy="deepest_structural_node_in_equal_clade_unary_chain",
        roots_and_singletons_excluded=True, all_original_vertices_preserved=True,
        exact_bitsets_retained=False, internal_ancestor_correspondence_claimed=False,
        observation_only=True, labels_times_targets_or_outcomes_used=False,
        flag_metadata=flag_meta,
    )
    return arrays, meta
