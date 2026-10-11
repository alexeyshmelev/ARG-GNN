"""H029 learned reference-cavity carrier memory on complete observed trees.

TGN, cavity messages and gated DeltaNet primitives are established. This module
tests their census-specific coupling, not new algebra or true ancestor tracking.
Every native vertex survives. Analytic 350 cuts add nodes, older proxies remain
on real sampling nodes, and qualified identities only join adjacent occurrences.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import random
from typing import Sequence

import torch
from torch import Tensor, nn

from .persistence_trace_kernel_v1 import MetricTreeHoldV1, _tree

ARMS = ("static", "carrier_gru", "carrier_matrix", "sham_ancestor")


@dataclass(frozen=True)
class CarrierFrameV1:
    nodes: Tensor
    previous_nodes: Tensor
    previous_index: Tensor
    sham_previous_index: Tensor
    event: Tensor
    gap_bp: float

    def to(self, device):
        return replace(self, **{name: getattr(self, name).to(device) for name in
            ("nodes", "previous_nodes", "previous_index", "sham_previous_index", "event")})


@dataclass(frozen=True)
class EdgeInnovationBankV1:
    parent: Tensor
    source: Tensor
    geometry: Tensor
    child_count: Tensor
    upward_levels: tuple
    upward_children: tuple
    upward_ranks: tuple
    downward_levels: tuple
    query_nodes: Tensor
    readout_hold: Tensor
    forward_distance: Tensor
    reverse_distance: Tensor
    age_groups: tuple
    forward_frames: tuple
    reverse_frames: tuple
    metadata: dict

    def to(self, device):
        simple = ("parent", "source", "geometry", "child_count", "query_nodes",
                  "readout_hold", "forward_distance", "reverse_distance")
        tuples = ("upward_levels", "upward_children", "upward_ranks", "downward_levels", "age_groups")
        return replace(self, **{name: getattr(self, name).to(device) for name in simple},
            **{name: tuple(x.to(device) for x in getattr(self, name)) for name in tuples},
            forward_frames=tuple(x.to(device) for x in self.forward_frames),
            reverse_frames=tuple(x.to(device) for x in self.reverse_frames))


def prepare_edge_innovation_bank_v1(
        trees: Sequence[MetricTreeHoldV1], intervals_bp: Tensor, readout_x_bp: Tensor,
        query_ids: Tensor, reference_ids: Tensor, reference_classes: Tensor, *,
        census: float = 350., sample_identity_namespace: str,
        sham_seed: int = 101, age_scale: float = 10000.,
        sham_age_bin: float = 500., sham_log_duration_bin: float = math.log(2.)):
    """Target-free CPU preparation of native forests plus exact census insertions.

    Neural operations batch whole forest levels. Only the recurrent carrier
    update loops over ordered boundaries, all live carriers together per boundary.
    This deliberately finite execution plan is not an associative scan claim.
    """
    fixed = (intervals_bp, readout_x_bp, query_ids, reference_ids, reference_classes)
    if (not trees or any(x.device.type != "cpu" or x.requires_grad for x in fixed)
            or intervals_bp.dtype != torch.float64 or intervals_bp.shape != (len(trees), 2)
            or readout_x_bp.dtype != torch.float64 or readout_x_bp.ndim != 1
            or any(x.dtype != torch.long or x.ndim != 1 for x in fixed[2:])
            or reference_ids.shape != reference_classes.shape or not len(query_ids)
            or not len(reference_ids) or query_ids.unique().numel() != len(query_ids)
            or reference_ids.unique().numel() != len(reference_ids)
            or bool(torch.isin(query_ids, reference_ids).any())
            or bool(((reference_classes < 0) | (reference_classes > 3)).any())
            or not bool(torch.isfinite(intervals_bp).all())
            or bool((intervals_bp[:, 1] <= intervals_bp[:, 0]).any())
            or not torch.equal(intervals_bp[:-1, 1], intervals_bp[1:, 0])
            or not len(readout_x_bp) or not bool(torch.isfinite(readout_x_bp).all())
            or bool(((readout_x_bp < intervals_bp[0, 0]) |
                     (readout_x_bp >= intervals_bp[-1, 1])).any())
            or not sample_identity_namespace
            or any(not math.isfinite(x) or x <= 0 for x in
                   (census, age_scale, sham_age_bin, sham_log_duration_bin))):
        raise ValueError("Fixed contiguous complete observed geometry and distinct Q/R joins required")
    parent, sources, geometries, children, heights, depths, queries = [], [], [], [], [], [], []
    offsets = [0]
    layouts = []
    known_age, sample_age, sample_native = {}, {}, {}
    query_ages = reference_ages = counts = None
    older_rows, younger_rows, retained = 0, 0, 0
    for hold, tree in enumerate(trees):
        samples, _, _ = _tree(tree)
        native_n = tree.parent.numel()
        p = tree.parent.tolist()
        t = tree.node_times.tolist()
        ids = tree.qualified_node_ids.tolist()
        sample_by_node = {node: identity for identity, node in samples.items()}
        if any(i not in samples for i in query_ids.tolist() + reference_ids.tolist()):
            raise ValueError("Every hold must retain all actual query/reference samples")
        for identity, node in samples.items():
            join = (tree.identity_namespace, ids[node]) if ids[node] >= 0 else None
            if identity in sample_age and (sample_age[identity] != t[node] or sample_native[identity] != join):
                raise ValueError("Actual sample date/native qualified join changed")
            sample_age[identity], sample_native[identity] = t[node], join
        for node, identity in enumerate(ids):
            if identity >= 0:
                key = (tree.identity_namespace, identity)
                if key in known_age and known_age[key] != t[node]:
                    raise ValueError("Qualified inferred object has inconsistent observed dates")
                known_age[key] = t[node]
        qa = torch.tensor([t[samples[i]] for i in query_ids.tolist()], dtype=torch.float64)
        ra = torch.tensor([t[samples[i]] for i in reference_ids.tolist()], dtype=torch.float64)
        if bool((qa >= census).any()) or bool((ra == census).any()):
            raise ValueError("This discovery requires actual younger Q and no reference exactly at census")
        if query_ages is None:
            query_ages, reference_ages = qa, ra
            counts = [torch.bincount(reference_classes[ra < census], minlength=4),
                      torch.bincount(reference_classes[ra > census], minlength=4)]
            if any(not bool((c > 0).any()) for c in counts):
                raise ValueError("Both observed younger/older reference families required")
        elif not torch.equal(qa, query_ages) or not torch.equal(ra, reference_ages):
            raise ValueError("Observed Q/R ages changed across the native tree collection")

        # Every crossing original edge receives one analytic cut. No native node
        # or original branch provenance is removed, and no older descendant exists.
        cut_for_child = {}
        original_parent = p.copy()
        original_time = t.copy()
        for child, ancestor in enumerate(original_parent):
            if ancestor >= 0 and original_time[child] < census < original_time[ancestor]:
                cut = len(p)
                cut_for_child[child] = cut
                p.append(ancestor); t.append(census)
                p[child] = cut
        local_children = [[] for _ in p]
        for child, ancestor in enumerate(p):
            if ancestor >= 0:
                local_children[ancestor].append(child)
        upward = sorted(range(len(p)), key=lambda node: t[node])
        downward = list(reversed(upward))
        source = torch.zeros(len(p), 8, dtype=torch.float32)
        support = [[0, 0] for _ in p]
        direct_proxy_ids = [[] for _ in p]

        def census_node(sample):
            node = sample
            while t[node] < census:
                node = p[node]
                if node < 0:
                    raise ValueError("Observed sample path must actually cross the census date")
            if t[node] != census:
                raise ValueError("Analytic native-edge census cut is missing")
            return node

        for identity, cls, age in zip(reference_ids.tolist(), reference_classes.tolist(), ra.tolist()):
            node = samples[identity]
            cohort = 0 if age < census else 1
            if cohort == 0:
                node = census_node(node); younger_rows += 1
            else:
                if t[node] != age:
                    raise ValueError("Older proxy must remain at its actual observed sampling node")
                older_rows += 1
            c = counts[cohort]
            present = (c > 0).float()
            contrast = torch.nn.functional.one_hot(torch.tensor(cls), 4).float() - present / present.sum()
            source[node, cohort * 4:cohort * 4 + 4] += contrast / (c[cls] * present.sum())
            support[node][cohort] += 1
            direct_proxy_ids[node].append(identity)
        descendant_support = [values.copy() for values in support]
        for node in upward:
            if p[node] >= 0:
                for cohort in (0, 1):
                    descendant_support[p[node]][cohort] += descendant_support[node][cohort]
        totals = [int(c.sum()) for c in counts]
        height = [0] * len(p); depth = [0] * len(p)
        for node in upward:
            if local_children[node]:
                height[node] = 1 + max(height[c] for c in local_children[node])
        for node in downward:
            if p[node] >= 0:
                depth[node] = depth[p[node]] + 1
        geometry = []
        scale = math.log1p(age_scale)
        for node, ancestor in enumerate(p):
            duration = 0. if ancestor < 0 else t[ancestor] - t[node]
            sub = descendant_support[node]
            geometry.append([math.log1p(t[node]) / scale,
                math.log1p(t[node] if ancestor < 0 else t[ancestor]) / scale,
                math.log1p(duration) / scale, float(t[node] == census),
                sub[0] / totals[0], sub[1] / totals[1],
                (totals[0] - sub[0]) / totals[0], (totals[1] - sub[1]) / totals[1]])
        offset = offsets[-1]
        carriers = []
        for child, ancestor in enumerate(original_parent):
            if ancestor < 0:
                continue
            anchor = cut_for_child.get(child, child)
            sample = sample_by_node.get(child)
            qualified = ids[child] >= 0 and ids[ancestor] >= 0 and bool(tree.identity_namespace)
            # Parent qualification is required for atomic paired reattachments.
            key = (("sample", sample_identity_namespace, sample) if sample is not None
                   else ("native", tree.identity_namespace, ids[child])) if qualified else None
            parent_key = (tree.identity_namespace, ids[ancestor]) if qualified else None
            duration = original_time[ancestor] - original_time[child]
            sub = descendant_support[anchor]
            bin_key = (math.floor(original_time[child] / sham_age_bin),
                math.floor(math.log(duration) / sham_log_duration_bin),
                tuple(sub), anchor in cut_for_child.values())
            carriers.append(dict(key=key, parent=parent_key, anchor=offset + anchor,
                source=tuple(sorted(direct_proxy_ids[anchor])), sample=sample is not None, bin=bin_key))
        queries.append(torch.tensor([offset + census_node(samples[i]) for i in query_ids.tolist()]))
        parent.extend(-1 if x < 0 else offset + x for x in p)
        sources.append(source); geometries.append(torch.tensor(geometry, dtype=torch.float32))
        children.extend([[offset + c for c in values] for values in local_children])
        heights.extend(height); depths.extend(depth)
        offsets.append(offset + len(p))
        layouts.append(carriers); retained += native_n

    up_levels, up_children, up_ranks = [], [], []
    for level in range(max(heights) + 1):
        members = [node for node, value in enumerate(heights) if value == level]
        gathered, ranks = [], []
        for rank, node in enumerate(members):
            gathered.extend(children[node]); ranks.extend([rank] * len(children[node]))
        up_levels.append(torch.tensor(members)); up_children.append(torch.tensor(gathered, dtype=torch.long))
        up_ranks.append(torch.tensor(ranks, dtype=torch.long))
    down_levels = tuple(torch.tensor([n for n, value in enumerate(depths) if value == level])
                        for level in range(1, max(depths) + 1))
    sham_changed = sham_eligible = singleton = events = new_episodes = 0

    def frames(reverse):
        nonlocal sham_changed, sham_eligible, singleton, events, new_episodes
        result, previous, previous_boundary = [], [], None
        order = list(range(len(trees)))
        if reverse:
            order.reverse()
        for hold in order:
            current = layouts[hold]
            lookup = {c["key"]: i for i, c in enumerate(previous) if c["key"] is not None}
            inherited, old_nodes, is_event = [], [], []
            bins = {}
            for rank, carrier in enumerate(current):
                old = lookup.get(carrier["key"], -1) if carrier["key"] is not None else -1
                inherited.append(old)
                old_nodes.append(previous[old]["anchor"] if old >= 0 else carrier["anchor"])
                changed = old < 0 or any(carrier[k] != previous[old][k] for k in ("parent", "source"))
                is_event.append(changed)
                if old < 0:
                    new_episodes += 1
                if old >= 0 and not carrier["sample"]:
                    bins.setdefault(carrier["bin"], []).append(rank)
            sham = inherited.copy()
            rng = random.Random(sham_seed + hold * 2 + int(reverse))
            for members in bins.values():
                sham_eligible += len(members)
                if len(members) < 2:
                    singleton += 1
                    continue
                shuffled = members.copy(); rng.shuffle(shuffled)
                for j, rank in enumerate(shuffled):
                    sham[rank] = inherited[shuffled[(j + 1) % len(shuffled)]]
                    sham_changed += 1
            boundary = float(intervals_bp[hold, 1 if reverse else 0])
            gap = 0. if previous_boundary is None else abs(boundary - previous_boundary)
            result.append(CarrierFrameV1(torch.tensor([c["anchor"] for c in current]),
                torch.tensor(old_nodes), torch.tensor(inherited), torch.tensor(sham),
                torch.tensor(is_event, dtype=torch.bool), gap))
            previous, previous_boundary = current, boundary
            events += sum(is_event)
        return tuple(result)

    forward, reverse = frames(False), frames(True)
    hold = torch.searchsorted(intervals_bp[:, 0].contiguous(), readout_x_bp, right=True) - 1
    if bool((readout_x_bp >= intervals_bp[hold, 1]).any()):
        raise ValueError("Original readouts must use their actual containing native hold")
    groups = tuple((query_ages == age).nonzero().flatten() for age in torch.unique(query_ages))
    metadata = dict(kind="edge_innovation_whole_native_forest_with_analytic_census_points",
        native_vertices=retained, analytic_cut_vertices=offsets[-1] - retained,
        total_vertices=offsets[-1], original_edges=retained - len(trees),
        augmented_edges=offsets[-1] - len(trees), holds=len(trees),
        max_height=max(heights), max_depth=max(depths),
        max_live_carriers=max(len(x) for x in layouts), total_carrier_occurrences=sum(len(x) for x in layouts),
        state_scalars_per_carrier_per_direction=16, bidirectional_events_including_creation=events,
        bidirectional_new_episodes=new_episodes, bidirectional_terminal_episodes=new_episodes,
        sham_changed_lookups=sham_changed, sham_eligible_internal_lookups=sham_eligible,
        sham_changed_lookup_fraction=sham_changed / sham_eligible if sham_eligible else 0.,
        sham_singleton_bins=singleton, query_count=len(query_ids), reference_count=len(reference_ids),
        younger_reference_count=int((reference_ages < census).sum()),
        older_reference_count=int((reference_ages > census).sum()),
        younger_class_counts=counts[0].tolist(), older_class_counts=counts[1].tolist(),
        older_proxy_placements_at_actual_sampling_nodes=older_rows,
        younger_proxy_placements_at_analytic_350=younger_rows,
        native_identity="qualified_adjacent_child_only;absence_resets;not_true_ancestor",
        query_readout="actual_sample_to_analytic350_cut;older_references_not_queries",
        recurrent_execution="all_live_carriers_batched_at_each_boundary;no_GNN_call_per_hold",
        retained_native_geometry=True, population_truth_inputs=False)
    return EdgeInnovationBankV1(torch.tensor(parent), torch.cat(sources), torch.cat(geometries),
        torch.tensor([len(x) for x in children], dtype=torch.float32),
        tuple(up_levels), tuple(up_children), tuple(up_ranks), down_levels, torch.stack(queries),
        hold, (readout_x_bp - intervals_bp[hold, 0]).float(),
        (intervals_bp[hold, 1] - readout_x_bp).float(), groups, forward, reverse, metadata)


class EdgeInnovationGNNV1(nn.Module):
    """Small FP32 reference-only encoder, carrier updater and linear tree read.

    Matrix and GRU arms both have16 state scalars per carrier per direction.
    The GRU gets the same key/value payload, external retention/write gates and
    bilinear current-key read, so their contrast isolates structured delta updates.
    It does not isolate conditional lookup from all forms of recurrent memory.
    """
    def __init__(self, hidden_dim=16, rank=4, initial_ell_bp=10000.,
                 min_ell_bp=1000., max_ell_bp=100000., key_norm_epsilon=1e-4):
        super().__init__()
        if hidden_dim != 16 or rank != 4 or not min_ell_bp < initial_ell_bp < max_ell_bp:
            raise ValueError("Fixed D16/rank4 FP32 discovery architecture required")
        self.hidden_dim, self.rank = hidden_dim, rank
        self.min_ell_bp, self.max_ell_bp = min_ell_bp, max_ell_bp
        if not 0 < key_norm_epsilon <= 1e-3:
            raise ValueError("Positive small latent key norm epsilon required")
        self.key_norm_epsilon = key_norm_epsilon
        raw = math.log((initial_ell_bp - min_ell_bp) / (max_ell_bp - initial_ell_bp))
        self.raw_ell = nn.Parameter(torch.tensor(raw, dtype=torch.float32))
        self.reference_projection = nn.Linear(8, hidden_dim, bias=False)
        self.up_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.down_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.up_gate = nn.Linear(8, hidden_dim)
        self.down_gate = nn.Linear(8, hidden_dim)
        self.value = nn.Linear(3 * hidden_dim, rank, bias=False)
        self.key = nn.Linear(hidden_dim, rank, bias=False)
        self.retention = nn.Linear(2 * hidden_dim + 8, 1)
        self.write_gate = nn.Linear(2 * hidden_dim + 8, 1)
        self.carrier_gru = nn.GRUCell(2 * rank, rank * rank, bias=False)
        self.decoder_base = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.decoder_up = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.decoder_down = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.decoder_edge = nn.Linear(rank, hidden_dim, bias=False)
        self.decoder_up_gate = nn.Linear(2 * hidden_dim + 8, hidden_dim)
        self.decoder_down_gate = nn.Linear(2 * hidden_dim + 8, hidden_dim)
        self.context_head = nn.Linear(3 * hidden_dim, 4, bias=False)
        nn.init.zeros_(self.context_head.weight)
        self.last_diagnostics = {}

    def ell_bp(self):
        return self.min_ell_bp + (self.max_ell_bp - self.min_ell_bp) * self.raw_ell.sigmoid()

    def parameter_counts(self, control):
        def active(name):
            if control != "carrier_gru" and name.startswith("carrier_gru."):
                return False
            if control == "static" and (name == "raw_ell" or
                    name.startswith(("value.", "key.", "retention.", "write_gate.", "decoder_edge."))):
                return False
            return True
        blocks = sum(p.numel() for name, p in self.named_parameters() if active(name))
        zero_stream_head = 2 * self.hidden_dim * 4 if control == "static" else 0
        return dict(total=sum(p.numel() for p in self.parameters()),
                    active=blocks - zero_stream_head, active_parameter_blocks=blocks,
                    zero_memory_stream_head_scalars=zero_stream_head)

    def encode(self, bank):
        n, d = len(bank.parent), self.hidden_dim
        base = self.reference_projection(bank.source)
        up = base.new_zeros(n, d); total = base.new_zeros(n, d)
        messages = base.new_zeros(n, d)
        ug = self.up_gate(bank.geometry).sigmoid()
        dg = self.down_gate(bank.geometry).sigmoid()
        for nodes, children, ranks in zip(bank.upward_levels, bank.upward_children, bank.upward_ranks):
            acc = base.new_zeros(len(nodes), d)
            if children.numel():
                value = self.up_value(up[children]) * ug[children]
                value = value / bank.child_count[bank.parent[children]][:, None]
                messages.index_copy_(0, children, value)
                acc.index_add_(0, ranks, value)
            value = base[nodes] + acc
            total.index_copy_(0, nodes, value)
            up.index_copy_(0, nodes, value.tanh())
        outside = base.new_zeros(n, d)
        for nodes in bank.downward_levels:
            parents = bank.parent[nodes]
            cavity = outside[parents] + total[parents] - messages[nodes]
            outside.index_copy_(0, nodes, (self.down_value(cavity) * dg[nodes]).tanh())
        return up, outside

    def bounded_key(self, outside):
        key = self.key(outside).tanh()
        return key / key.norm(dim=-1, keepdim=True).clamp_min(self.key_norm_epsilon)

    def scan(self, bank, up, outside, frames, control):
        edge_read = up.new_zeros(len(up), self.rank)
        previous = up.new_zeros(0, self.rank, self.rank)
        # This is a boundary loop, not a node-wise neural loop. In sham, internal
        # inherited state lookups are deranged, but instantaneous cavity inputs,
        # active carriers and event masks stay exactly those of the genuine arm.
        for frame in frames:
            nodes = frame.nodes
            linked = frame.previous_index >= 0
            inherited = frame.sham_previous_index if control == "sham_ancestor" else frame.previous_index
            if previous.numel():
                old = previous[inherited.clamp_min(0)] * linked[:, None, None]
            else:
                old = up.new_zeros(len(nodes), self.rank, self.rank)
            u, o = up[nodes], outside[nodes]
            old_u, old_o = up[frame.previous_nodes], outside[frame.previous_nodes]
            du, do = u - old_u, o - old_o
            old_key = self.bounded_key(torch.where(linked[:, None], old_o, o))
            value_input = torch.cat((du, do, (u + old_u) * .5 * do), -1)
            fresh_input = torch.cat((u, o, u * o), -1)
            value = self.value(torch.where(linked[:, None], value_input, fresh_input)).tanh()
            gate_input = torch.cat((du, do, bank.geometry[nodes]), -1)
            gamma = self.retention(gate_input).sigmoid()
            eta = self.write_gate(gate_input).sigmoid()
            decayed = old * torch.exp(-frame.gap_bp / self.ell_bp())
            mbar = decayed * torch.where(frame.event[:, None], gamma, torch.ones_like(gamma))[:, :, None]
            if control == "carrier_gru":
                candidate = self.carrier_gru(torch.cat((value, old_key), -1), mbar.flatten(1))
                written = ((1 - eta) * mbar.flatten(1) + eta * candidate).reshape_as(mbar)
            else:
                residual = value - (mbar @ old_key[:, :, None]).squeeze(-1)
                written = (mbar + eta[:, :, None] * residual[:, :, None] * old_key[:, None, :]).tanh()
            previous = torch.where(frame.event[:, None, None], written, decayed)
            z = (previous @ self.bounded_key(o)[:, :, None]).squeeze(-1)
            edge_read.index_copy_(0, nodes, z)
        return edge_read

    def decode(self, bank, up, outside, forward, reverse):
        n, d = len(up), self.hidden_dim
        # Three independent linear value streams share maps and reference gates.
        # This permits exact positive within-hold exponential scaling after gather.
        zero = up.new_zeros(n, d)
        base = torch.stack((self.decoder_base(up + outside), zero, zero), 1)
        edge = torch.stack((zero, self.decoder_edge(forward), self.decoder_edge(reverse)), 1)
        conditions = torch.cat((up, outside, bank.geometry), -1)
        ug = self.decoder_up_gate(conditions).sigmoid()[:, None]
        dg = self.decoder_down_gate(conditions).sigmoid()[:, None]
        node_value = up.new_zeros(n, 3, d)
        total = up.new_zeros(n, 3, d); messages = up.new_zeros(n, 3, d)
        for nodes, children, ranks in zip(bank.upward_levels, bank.upward_children, bank.upward_ranks):
            acc = up.new_zeros(len(nodes), 3, d)
            if children.numel():
                value = (self.decoder_up(node_value[children]) + edge[children]) * ug[children]
                value = value / bank.child_count[bank.parent[children]][:, None, None]
                messages.index_copy_(0, children, value); acc.index_add_(0, ranks, value)
            value = base[nodes] + acc
            total.index_copy_(0, nodes, value); node_value.index_copy_(0, nodes, value)
        down = up.new_zeros(n, 3, d)
        for nodes in bank.downward_levels:
            parents = bank.parent[nodes]
            cavity = down[parents] + total[parents] - messages[nodes]
            value = (self.decoder_down(cavity) + edge[nodes]) * dg[nodes]
            down.index_copy_(0, nodes, value)
        return node_value + down

    def forward(self, bank, frozen_baseline_logits, *, control):
        if (control not in ARMS or bank.source.dtype != torch.float32
                or frozen_baseline_logits.dtype != torch.float32
                or frozen_baseline_logits.shape != (len(bank.readout_hold), bank.query_nodes.shape[1], 4)
                or frozen_baseline_logits.device != self.raw_ell.device
                or bank.source.device != self.raw_ell.device
                or not bool(torch.isfinite(frozen_baseline_logits).all())):
            raise ValueError("Matched FP32 full-roster bank and finite frozen skip required")
        up, outside = self.encode(bank)
        empty = up.new_zeros(len(up), self.rank)
        forward = empty if control == "static" else self.scan(bank, up, outside, bank.forward_frames, control)
        reverse = empty if control == "static" else self.scan(bank, up, outside, bank.reverse_frames, control)
        decoded = self.decode(bank, up, outside, forward, reverse)
        context = decoded[bank.query_nodes[bank.readout_hold]]
        if control != "static":
            context = context * torch.stack((torch.ones_like(bank.forward_distance),
                torch.exp(-bank.forward_distance / self.ell_bp()),
                torch.exp(-bank.reverse_distance / self.ell_bp())), -1)[:, None, :, None]
        context = context.flatten(2)
        centered = context.clone()
        for group in bank.age_groups:
            centered[:, group] = context[:, group] - context[:, group].mean(1, keepdim=True)
        logits = frozen_baseline_logits.detach() + self.context_head(centered)
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("Nonfinite learned whole-tree FP32 context readout")
        self.last_diagnostics = dict(ell_bp=float(self.ell_bp().detach()),
            forward_read_l2=float(forward.detach().norm()),
            reverse_read_l2=float(reverse.detach().norm()),
            centered_context_l2=float(centered.detach().norm()))
        return logits
