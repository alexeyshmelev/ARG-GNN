"""Static forms of the already qualified episode preparation functions."""
from collections.abc import Mapping
import numpy as np
import torch
from arggnn.ordered_clade_episode_gnn_v1 import (
    OrderedCladeEpisodesV1, AnonymousCladeFlagsV1, ARRAY_FIELDS, DEFAULT_PAYLOAD_CAP,
    FLAG_SCHEMA, CHANNELS, SCHEMA, _require, _sha, _span_denominators,
)
from research.reviews.H060_isolated_hold_compatibility_source_20261017_v1.episode_preparation_v1 import _tree_clades
from research.reviews.workflow_progress_diagnosis_20261018.treeloom import indices_v1 as ix

def build_clade_episode_arrays_v1(
        parent, sample_index, spans, *, max_payload_bytes=DEFAULT_PAYLOAD_CAP, _forest_index):
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
    head, sibling, roots, parent_steps = _forest_index(parent, samples)
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

def load_verified_arrays(cls, arrays, meta):
    """Check compact structure and hashes, not external observation lineage.

    Exact-clade membership is established by the topology builder, not by
    hashes supplied by an arbitrary caller. Root owns original-input lineage.
    """
    _require(isinstance(arrays, Mapping) and set(arrays) == set(ARRAY_FIELDS)
             and isinstance(meta, Mapping) and meta.get("schema") == SCHEMA,
             "Exact compact episode arrays and metadata required")
    n, t, h = (meta.get(k) for k in ("native_nodes", "native_holds", "haplotypes"))
    m, e = (meta.get(k) for k in ("member_count", "episode_count"))
    _require(all(type(v) is int for v in (n, t, h, m, e))
             and 0 < n < 2**31 and 0 < t < 2**31 and 3 <= h < 2**31
             and 0 <= e <= m <= n and ((m == 0) == (e == 0)),
             "Bounded episode dimensions required")
    expected = dict(
        node_flags=(np.uint8, (n, 2)), member_node=(np.int32, (m,)),
        member_hold=(np.int32, (m,)), member_episode=(np.int32, (m,)),
        episode_size=(np.int32, (e,)), hold_span=(np.float32, (t,)),
    )
    hashes = meta.get("array_sha256", {})
    for key, (dtype, shape) in expected.items():
        array = arrays[key]
        _require(isinstance(array, np.ndarray) and array.dtype == dtype
                 and array.shape == shape and array.flags.c_contiguous
                 and _sha(array) == hashes.get(key),
                 "Episode array dtype, shape, contiguity or hash differs: " + key)
    _require(meta.get("exact_bitsets_retained") is False
             and meta.get("internal_ancestor_correspondence_claimed") is False
             and meta.get("labels_times_targets_or_outcomes_used") is False
             and meta.get("observation_only") is True
             and meta.get("roots_and_singletons_excluded") is True
             and meta.get("all_original_vertices_preserved") is True
             and meta.get("episode_policy")
             == "maximal_consecutive_trees_exact_sampled_descendant_set"
             and meta.get("representative_policy")
             == "deepest_structural_node_in_equal_clade_unary_chain",
             "Registered observation-only episode policies required")
    nodes, holds, episodes, sizes, spans = (
        arrays[k] for k in ARRAY_FIELDS[1:]
    )
    _require(np.all((nodes >= 0) & (nodes < n))
             and np.unique(nodes).size == m
             and np.all((holds >= 0) & (holds < t))
             and np.all((episodes >= 0) & (episodes < e))
             and np.all(sizes >= 1) and np.isfinite(spans).all()
             and np.all(spans > 0) and np.all(holds[1:] >= holds[:-1])
             and _sha(spans) == meta.get("span_sha256"),
             "Unique bounded nodes, ordered holds and positive spans required")
    _require(np.array_equal(np.bincount(episodes, minlength=e), sizes),
             "Episode sizes must equal exact membership counts")
    ix.consecutive_members(holds, episodes)
    eligible = _span_denominators(holds, episodes, sizes, spans)
    payload_bytes = sum(a.nbytes for a in arrays.values())
    cap = meta.get("max_payload_bytes")
    _require(type(cap) is int and payload_bytes == meta.get("payload_bytes")
             and payload_bytes <= 18*n + 4*t <= cap
             and meta.get("array_allocation_bound_bytes") == 18*n + 4*t
             and int(eligible.sum()) == meta.get("eligible_member_count")
             and int(np.count_nonzero(sizes >= 2)) == meta.get("eligible_episode_count")
             and int(np.count_nonzero(sizes == 1)) == meta.get("singleton_episode_count"),
             "Payload cap, allocation bound and eligibility counts differ")
    flag_meta = meta.get("flag_metadata", {})
    _require(all(flag_meta.get(k) == meta[k]
                 for k in ("native_nodes", "native_holds", "haplotypes")),
             "H042 flags and episode dimensions must match")
    flags = AnonymousCladeFlagsV1.from_verified_arrays(arrays["node_flags"], flag_meta)
    selected = np.zeros(n, dtype=np.bool_)
    selected[nodes] = True
    _require(not np.any(arrays["node_flags"][~selected]),
             "Only proper-clade representatives may have H042 flags")
    # Copies give the prepared payload independent ownership of supplied arrays.
    return cls(
        AnonymousCladeFlagsV1(flags.node_flags.clone(), t, h, n),
        *(torch.from_numpy(arrays[k].copy()) for k in ARRAY_FIELDS[1:]),
    )


def build_episodes(bank, *, max_payload_bytes, isolated_hold_indices=()):
    def forest(parent, samples):
        return ix.forest_for_bank(bank, parent, samples, isolated_hold_indices)
    arrays, metadata = build_clade_episode_arrays_v1(
        bank.parent.numpy(), bank.sample_index.numpy(), bank.spans.numpy(),
        max_payload_bytes=max_payload_bytes, _forest_index=forest)
    return load_verified_arrays(OrderedCladeEpisodesV1, arrays, metadata), metadata
