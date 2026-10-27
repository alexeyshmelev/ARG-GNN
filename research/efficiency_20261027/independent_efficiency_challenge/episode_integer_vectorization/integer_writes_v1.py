"""Exact per-hold integer stores; no learned-state reuse or floating reductions."""
import numpy as np
from arggnn.ordered_clade_episode_gnn_v1 import _require


def scalar_integer_writes(current, previous_episode, hold, nodes, holds, episodes,
                          sizes, members, episode_count, n):
    """Original loop, retained for exceptional bounds and integer warning parity."""
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
    return current_episode, members, episode_count


def write_episode_integers(current, previous_episode, hold, nodes, holds, episodes,
                           sizes, members, episode_count, n):
    """Batch only integer writes, preserving the original insertion and ID order.

    Inputs are the validated internal buffers and clade dictionaries of the
    original episode builder. Temporary storage is bounded by one native hold.
    Conservative bounds route exceptional states through the unchanged loop,
    retaining original guard order and partial mutations on rejected states.
    """
    count = len(current)
    if members + count > n or episode_count + count > n:
        return scalar_integer_writes(current, previous_episode, hold, nodes,
                                     holds, episodes, sizes, members, episode_count, n)
    if not count:
        return {}, members, episode_count

    first_episode_count = episode_count
    current_episode = {}
    node_values, episode_values = [], []
    for mask, node in current.items():
        episode = previous_episode.get(mask)
        if episode is None:
            episode = episode_count
            episode_count += 1
        node_values.append(node)
        episode_values.append(episode)
        current_episode[mask] = episode
    episode_ids = np.asarray(episode_values, dtype=np.int32)
    # Valid episode counts cannot overflow: each is at most T < 2**31.
    # Preserve NumPy scalar overflow warnings for synthetic boundary states.
    if np.any(sizes[episode_ids] > np.iinfo(np.int32).max - count):
        return scalar_integer_writes(current, previous_episode, hold, nodes,
                                     holds, episodes, sizes, members, first_episode_count, n)
    stop = members + count
    nodes[members:stop] = node_values
    holds[members:stop] = hold
    episodes[members:stop] = episode_ids
    # add.at also preserves repeated-ID counts, without relying on uniqueness.
    np.add.at(sizes, episode_ids, np.int32(1))
    return current_episode, stop, episode_count
