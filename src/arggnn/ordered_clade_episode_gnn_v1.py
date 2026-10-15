"""H045 contiguous exact proper-clade episodes and matched Torch/PyG model.

Source-only preparation/model API. Exact sampled-descendant bitsets exist only
for the previous/current trees during preparation. No target, dataset,
checkpoint-file reader, launcher, admission or execution policy is provided.
The supplied H042 pass-30 state is retained, including its learned flag adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import hashlib

import numpy as np
import torch
from torch import nn
from torch_geometric.nn.aggr import SumAggregation

from .anonymous_clade_flag_control_v1 import (
    AnonymousCladeFlagsV1, AnonymousCladeFlagControlGNNV1,
    build_clade_flag_arrays_v1, validate_clade_flags_v1,
    _cpu_index_array, _tree_clades, SCHEMA as FLAG_SCHEMA, CHANNELS,
)
# Reuse only H043's linear-time forest indexing, not its transfer operator.
from .anonymous_exact_clade_transport_v1 import _forest_index
from .mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeInputsV1

SOURCE = "src/arggnn/ordered_clade_episode_gnn_v1.py"
SCHEMA = "ordered_exact_proper_clade_episodes_v1"
ARMS = ("foreign_episode", "own_state")
ARRAY_FIELDS = (
    "node_flags", "member_node", "member_hold", "member_episode",
    "episode_size", "hold_span",
)
NEW_KEYS = frozenset("episode_coupling.%s.weight" % name for name in ("U", "V", "W", "O"))
DEFAULT_PAYLOAD_CAP = 1073741824 - 67108864
MODEL_KWARGS = dict(
    hidden_dim=32, expected_haplotypes=952,
    site_chunk_size=256, checkpoint_sites=True,
)


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _span_denominators(member_hold, member_episode, episode_size, spans):
    """FP32 accumulation in native-tree order, with safe singleton handling."""
    weights = spans[member_hold]
    totals = np.zeros(len(episode_size), dtype=np.float32)
    np.add.at(totals, member_episode, weights)
    eligible = episode_size[member_episode] >= 2
    denominator = totals[member_episode] - weights
    _require(np.isfinite(totals).all()
             and np.isfinite(denominator).all()
             and np.all(denominator[eligible] > 0),
             "Finite positive FP32 leave-one-out spans required for eligible episodes")
    return eligible


def build_clade_episode_arrays_v1(
        parent, sample_index, spans, *, max_payload_bytes=DEFAULT_PAYLOAD_CAP):
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


@dataclass(frozen=True)
class OrderedCladeEpisodesV1:
    flags: AnonymousCladeFlagsV1
    member_node: torch.Tensor
    member_hold: torch.Tensor
    member_episode: torch.Tensor
    episode_size: torch.Tensor
    hold_span: torch.Tensor

    @property
    def native_holds(self):
        return self.flags.native_holds

    @property
    def haplotypes(self):
        return self.flags.haplotypes

    @property
    def native_nodes(self):
        return self.flags.native_nodes

    @classmethod
    def from_verified_arrays(cls, arrays, meta):
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
        last = np.full(e, -1, dtype=np.int32)
        for hold, episode in zip(holds, episodes):
            _require(last[episode] < 0 or hold == last[episode] + 1,
                     "Each episode must occur once per consecutive tree")
            last[episode] = hold
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

    def to(self, device, *, non_blocking=False):
        return type(self)(
            self.flags.to(device, non_blocking=non_blocking),
            *(getattr(self, k).to(device, non_blocking=non_blocking)
              for k in ARRAY_FIELDS[1:]),
        )

    def tensor_bytes(self):
        return self.flags.tensor_bytes() + sum(
            getattr(self, k).numel() * getattr(self, k).element_size()
            for k in ARRAY_FIELDS[1:]
        )


def build_clade_episodes_v1(bank, *, max_payload_bytes=DEFAULT_PAYLOAD_CAP):
    """Root supplies the original CPU bank; no observation/checkpoint reader."""
    if type(bank) is not MutationRoutedWholeTreeInputsV1:
        raise TypeError("Exact original observation-only bank required")
    samples = _cpu_index_array(bank.sample_index)
    _require(samples.shape[1] == 952 and bank.query_column.shape == (686,),
             "Original full952/all686query native bank required")
    _require(bank.spans.device.type == "cpu" and bank.spans.dtype == torch.float32
             and not bank.spans.requires_grad,
             "Read-only original CPU FP32 spans required")
    arrays, meta = build_clade_episode_arrays_v1(
        _cpu_index_array(bank.parent), samples, bank.spans.numpy(),
        max_payload_bytes=max_payload_bytes,
    )
    return OrderedCladeEpisodesV1.from_verified_arrays(arrays, meta), meta


def validate_clade_episodes_v1(episodes, upward):
    if not isinstance(episodes, OrderedCladeEpisodesV1):
        raise TypeError("Verified compact clade-episode payload required")
    validate_clade_flags_v1(episodes.flags, upward)
    m, e = episodes.member_node.numel(), episodes.episode_size.numel()
    for name, size in (("member_node", m), ("member_hold", m),
                       ("member_episode", m), ("episode_size", e)):
        value = getattr(episodes, name)
        _require(value.shape == (size,) and value.dtype == torch.int32
                 and value.device == upward.device and not value.requires_grad,
                 "Compact same-device int32 episode vectors required")
    _require(episodes.hold_span.shape == (episodes.native_holds,)
             and episodes.hold_span.dtype == torch.float32
             and episodes.hold_span.device == upward.device
             and not episodes.hold_span.requires_grad,
             "Same-device read-only FP32 physical spans required")


class RankEightCladeEpisodeAdapterV1(nn.Module):
    """Separate 1,024-weight residual, with the same PyG pooling in both arms."""
    def __init__(self):
        super().__init__()
        self.U = nn.Linear(32, 8, bias=False, dtype=torch.float32)
        self.V = nn.Linear(32, 8, bias=False, dtype=torch.float32)
        self.W = nn.Linear(32, 8, bias=False, dtype=torch.float32)
        self.O = nn.Linear(8, 32, bias=False, dtype=torch.float32)
        nn.init.zeros_(self.O.weight)
        self.aggregation = SumAggregation()

    def pooled_context(self, upward, episodes):
        validate_clade_episodes_v1(episodes, upward)
        nodes = episodes.member_node.long()
        member = upward.index_select(0, nodes)
        weights = episodes.hold_span.index_select(0, episodes.member_hold.long())[:, None]
        index = episodes.member_episode.long()
        count = episodes.episode_size.numel()
        total = self.aggregation(member * weights, index=index, dim_size=count, dim=0)
        total_weight = self.aggregation(weights, index=index, dim_size=count, dim=0)
        eligible = episodes.episode_size.index_select(0, index) >= 2
        denominator = total_weight.index_select(0, index) - weights
        safe_denominator = torch.where(eligible[:, None], denominator, torch.ones_like(denominator))
        foreign = (total.index_select(0, index) - weights * member) / safe_denominator
        foreign = torch.where(eligible[:, None], foreign, torch.zeros_like(foreign))
        return member, foreign, eligible

    def forward(self, upward, episodes, *, arm):
        _require(arm in ARMS, "Registered foreign_episode or own_state arm required")
        _require(all(p.dtype == torch.float32 and p.device == upward.device
                     for p in self.parameters()),
                 "FP32 same-device episode adapter parameters required")
        member, foreign, eligible = self.pooled_context(upward, episodes)
        context = foreign if arm == "foreign_episode" else member + 0.0 * foreign
        residual = self.O(self.U(member) * self.V(context) + self.W(member))
        residual = residual * eligible[:, None].to(dtype=torch.float32)
        return upward.new_zeros(upward.shape).index_add(
            0, episodes.member_node.long(), residual,
        )


class OrderedCladeEpisodeGNNV1(AnonymousCladeFlagControlGNNV1):
    """Complete learned H042 body + separate 1,024-weight episode adapter."""
    def __init__(self, hidden_dim=32, *, expected_haplotypes=952,
                 site_chunk_size=256, checkpoint_sites=True, arm="foreign_episode"):
        _require(arm in ARMS, "Registered H045 arm required")
        super().__init__(
            hidden_dim=hidden_dim, expected_haplotypes=expected_haplotypes,
            site_chunk_size=site_chunk_size, checkpoint_sites=checkpoint_sites,
        )
        self.episode_coupling = RankEightCladeEpisodeAdapterV1()
        self.arm = arm
        # Preserve FP32 even when a caller has changed torch's default dtype.
        self.float()

    def encode_trees(self, bank, *, flags, control="aligned"):
        # Inherited H042 forward_holds passes our richer payload as 'flags'.
        episodes = flags
        _require(isinstance(episodes, OrderedCladeEpisodesV1),
                 "H045 episode payload required")
        _require(bank.spans.shape == episodes.hold_span.shape
                 and torch.equal(bank.spans, episodes.hold_span),
                 "Episode physical spans must equal this bank's original spans")
        base, ug, dg, degree = self.prepare_tree_base_v1(bank, control=control)
        upward, _outside = self.sweep_tree_state_v1(bank, base, ug, dg, degree)
        injection = (
            self.clade_coupling(upward, episodes.flags)
            + self.episode_coupling(upward, episodes, arm=self.arm)
        )
        second_upward, second_outside = self.sweep_tree_state_v1(
            bank, base + injection, ug, dg, degree,
        )
        return self.leaf_context_v1(bank, second_upward, second_outside)

    def forward_holds(self, bank, *, episodes, control="aligned"):
        return super().forward_holds(bank, flags=episodes, control=control)

    def forward_train_holds(self, bank, *, episodes, control="aligned"):
        return self.forward_holds(bank, episodes=episodes, control=control).index_select(
            1, bank.query_column,
        )

    def forward(self, bank, *, episodes, control="aligned"):
        return self.forward_train_holds(bank, episodes=episodes, control=control).index_select(
            0, bank.readout_hold,
        )


def load_h042_episode_warmstart_v1(model, payload):
    """Accept a caller-supplied H042 pass30 payload; never read a checkpoint."""
    _require(isinstance(model, OrderedCladeEpisodeGNNV1)
             and isinstance(payload, Mapping)
             and payload.get("schema") == "anonymous_clade_flag_checkpoint_v1"
             and payload.get("hypothesis_id") == "H042"
             and payload.get("phase") == "train"
             and payload.get("arm") == "exact_clade_flags"
             and payload.get("seed") == 101 and payload.get("pass_index") == 30
             and payload.get("additional_optimizer_steps") == 960
             and payload.get("parameter_count") == 30788
             and payload.get("teacher_used") is False
             and payload.get("model_kwargs") == MODEL_KWARGS,
             "Original FP32 H042 seed101 pass30/960-step teacher-free payload required")
    state = payload.get("model_state")
    expected = model.state_dict()
    _require(isinstance(state, Mapping) and set(state) == set(expected) - NEW_KEYS,
             "Complete learned H042 body and flag adapter keys required")
    for key, value in state.items():
        _require(isinstance(value, torch.Tensor) and value.device.type == "cpu"
                 and value.dtype == torch.float32 and value.shape == expected[key].shape
                 and bool(torch.isfinite(value).all()),
                 "Finite original CPU FP32 H042 learned state required: " + key)
    _require(all(p.dtype == torch.float32 for p in model.parameters())
             and sum(p.numel() for p in model.parameters()) == 31812
             and not bool(model.episode_coupling.O.weight.count_nonzero()),
             "FP32 31,812-weight model with new O exactly zero required")
    # All keys/shapes/values were validated before any learned state is modified.
    result = model.load_state_dict(state, strict=False)
    _require(set(result.missing_keys) == NEW_KEYS and not result.unexpected_keys,
             "Only the four new episode projections may be missing")
    return dict(
        h042_parameter_count=30788, added_parameter_count=1024,
        total_parameter_count=31812, complete_learned_H042_flag_adapter_preserved=True,
        output_projection_initialized_zero=True, full_body_trainable=True,
        source_only=True, checkpoint_provenance_verified=False,
    )


def build_matched_episode_family_v1(payload, model_kwargs=None):
    """CPU construction only: identical states and independent reset AdamW.

    A future root-owned runner must perform its own admission, transfer and
    execution checks. This helper neither reads a checkpoint nor launches work.
    """
    model_kwargs = dict(MODEL_KWARGS) if model_kwargs is None else model_kwargs
    _require(model_kwargs == MODEL_KWARGS, "Registered H042 model settings required")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(101)
        prototype = OrderedCladeEpisodeGNNV1(**model_kwargs)
        proof = load_h042_episode_warmstart_v1(prototype, payload)
        initial = {k: v.clone() for k, v in prototype.state_dict().items()}
        family = {}
        for arm in ARMS:
            model = OrderedCladeEpisodeGNNV1(**model_kwargs, arm=arm)
            model.load_state_dict(initial, strict=True)
            family[arm] = model
    optimizers = {
        arm: torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001,
                              betas=(.9, .999), eps=1e-8)
        for arm, model in family.items()
    }
    return family, optimizers, dict(
        proof, arms=list(ARMS), seed=101, dtype="float32", gradient_clip=1.0,
        same_initial_weights=True, independent_reset_AdamW=True,
        training_updates_executed=0,
    )
