"""Original proper-clade episode recipe with exact isolated native-root bookkeeping."""
from arggnn import ordered_clade_episode_gnn_v1 as _episodes
from arggnn.anonymous_exact_clade_transport_v1 import _forest_index as _connected_forest_index
from arggnn.anonymous_clade_flag_control_v1 import _tree_clades as _connected_tree_clades
from research.reviews.H060_isolated_hold_compatibility_source_20261017_v1.structural_index_v1 import classify_sample_roots,validate_root_partition,isolated_clades
np=_episodes.np
_require=_episodes._require
_sha=_episodes._sha
_span_denominators=_episodes._span_denominators
DEFAULT_PAYLOAD_CAP=_episodes.DEFAULT_PAYLOAD_CAP
FLAG_SCHEMA=_episodes.FLAG_SCHEMA
CHANNELS=_episodes.CHANNELS
SCHEMA=_episodes.SCHEMA
from .kernels import _forest_index, _tree_clades

def build_clade_episode_arrays_v1(
        parent, sample_index, spans, *, max_payload_bytes=DEFAULT_PAYLOAD_CAP,isolated_hold_indices=()):
    """Bounded CPU topology/span preparation, permitting small synthetic panels.

    Stable sampled-leaf columns are an upstream input contract. Every proper
    clade occurrence is represented, but only episodes of length >=2 inject.
    Arrays are bounded by 18*N+4*T bytes before construction. This is a compact
    array allocation bound, not a total Python/process-memory resource proof.
    """
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


def build_episodes(bank, *, max_payload_bytes, isolated_hold_indices=()):
    """Return the same verified runtime episode object, with unchanged loader."""
    from research.repairs.benchmark_recovery_20261020.runtime.episodes import load_verified_arrays
    from .kernels import validate_bank_levels
    validate_bank_levels(bank, bank.parent.numpy())
    arrays, metadata = build_clade_episode_arrays_v1(bank.parent.numpy(), bank.sample_index.numpy(),
        bank.spans.numpy(), max_payload_bytes=max_payload_bytes, isolated_hold_indices=isolated_hold_indices)
    return load_verified_arrays(_episodes.OrderedCladeEpisodesV1, arrays, metadata), metadata
