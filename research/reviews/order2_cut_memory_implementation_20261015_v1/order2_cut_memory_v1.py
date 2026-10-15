"""Three-arm sample-witness order-2 cut memory, source-only.

Known incidence/memory-network machinery; domain utility is unproved.
No data reader, checkpoint reader, writer, launcher, admission or CLI.
The caller owns authenticated observations, resources, warm start and truth gates.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Mapping
import copy
import hashlib
import math

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn.aggr import SumAggregation

from arggnn.marker_grid_reset_chain_gnn_v1 import (
    MarkerGridResetChainGNNV1, MarkerGridResetBundleV1,
    load_h042_grid_reset_warmstart_v1, MODEL,
)
from arggnn.marker_resolution_readout_v1 import H042_PASS30_REF, H042_COMPLETION_REF

ARMS = ("triple_support_pair_memory", "adjacent_marginal_null", "stable_leaf_memory")
HORIZON = 350.0
RANK = 8
ADDED_PARAMETERS = 513
TOTAL_PARAMETERS = 32446
_BIND_FIELDS = ("parent", "node_time", "sample_index", "sample_age", "spans")


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _stamps(tensors):
    return tuple((id(x), x.data_ptr(), x._version, tuple(x.shape), x.dtype, str(x.device))
                 for x in tensors)


def _bank_stamps(bank):
    return _stamps(tuple(getattr(bank, n) for n in _BIND_FIELDS))


def _fingerprint(x):
    a = x.detach().contiguous().numpy()
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class CutPlanV1:
    """Compact disjoint cuts and event layouts; no targets or node-ID matching."""
    representatives: torch.Tensor       # [K_total], packed native node indexes
    leaf_groups: torch.Tensor           # [T,H], globally unique native-hold group rows
    group_counts: torch.Tensor          # [K_total], positive FP32
    episode_groups: torch.Tensor        # [G,H], local cut labels
    episode_group_counts: torch.Tensor  # [G,H], padded FP32
    forward_cells: torch.Tensor         # [G,H], leaf -> active pair cell
    forward_current_group: torch.Tensor # [G,H], cell -> current local group
    forward_old_group: torch.Tensor     # [G,H], cell -> preceding distinct cut group
    forward_counts: torch.Tensor        # [G,H], padded FP32
    reverse_cells: torch.Tensor
    reverse_current_group: torch.Tensor
    reverse_old_group: torch.Tensor
    reverse_counts: torch.Tensor
    above_horizon: torch.Tensor         # [H], actual sample mask
    root_saturated: torch.Tensor        # [K_total], actual root mask
    group_offsets: tuple
    episode_of_hold: tuple
    native_holds: int
    haplotypes: int
    native_nodes: int
    source_fingerprints: tuple
    source_stamps: tuple
    tensor_stamps: tuple

    def tensors(self):
        return (self.representatives, self.leaf_groups, self.group_counts,
                self.episode_groups, self.episode_group_counts,
                self.forward_cells, self.forward_current_group, self.forward_old_group,
                self.forward_counts, self.reverse_cells, self.reverse_current_group,
                self.reverse_old_group, self.reverse_counts, self.above_horizon,
                self.root_saturated)

    def validate(self, bank):
        _require(self.source_stamps == _bank_stamps(bank),
                 "Cut plan belongs to these immutable native topology/date/span tensors")
        _require(self.tensor_stamps == _stamps(self.tensors()), "Cut-plan tensors changed")
        _require(tuple(bank.sample_index.shape) == (self.native_holds, self.haplotypes)
                 and bank.parent.numel() == self.native_nodes
                 and self.representatives.device == bank.parent.device,
                 "Cut plan and complete native bank dimensions/device agree")
        return self

    def _to_for_bank(self, device, source_bank, target_bank):
        """Only called by the bound bundle's explicit observation transfer."""
        self.validate(source_bank)
        names = ("representatives", "leaf_groups", "group_counts", "episode_groups",
                 "episode_group_counts", "forward_cells", "forward_current_group",
                 "forward_old_group", "forward_counts", "reverse_cells",
                 "reverse_current_group", "reverse_old_group", "reverse_counts",
                 "above_horizon", "root_saturated")
        moved = {name: getattr(self, name).to(device) for name in names}
        result = replace(self, **moved, source_stamps=_bank_stamps(target_bank), tensor_stamps=())
        result = replace(result, tensor_stamps=_stamps(result.tensors()))
        result.validate(target_bank)
        return result

    def tensor_bytes(self):
        return sum(x.numel() * x.element_size() for x in self.tensors())


def _direction_layouts(groups, order, h):
    g = len(groups)
    cells = np.empty((g, h), dtype=np.int64)
    current = np.zeros((g, h), dtype=np.int64)
    old = np.zeros((g, h), dtype=np.int64)
    counts = np.zeros((g, h), dtype=np.float32)
    previous = None
    for e in order:
        now = groups[e]
        if previous is None:
            cells[e] = now
            k = int(now.max()) + 1
            current[e, :k] = np.arange(k)
            old[e, :k] = np.arange(k)
            counts[e, :k] = np.bincount(now, minlength=k)
        else:
            lookup = {}
            for i, key in enumerate(zip(groups[previous].tolist(), now.tolist())):
                if key not in lookup:
                    j = len(lookup)
                    lookup[key] = j
                    old[e, j], current[e, j] = key
                cells[e, i] = lookup[key]
            k = len(lookup)
            counts[e, :k] = np.bincount(cells[e], minlength=k)
        previous = e
    return cells, current, old, counts


def prepare_cut_plan_v1(bank, *, expected_haplotypes=952,
                        max_nodes=10000000, max_holds=10000):
    """CPU observation-only preparation. Small H is allowed for handmade checks.

    Production bundle preparation separately requires all952/686 original rows.
    No scientific file is opened here. Fingerprints cover only supplied tensors.
    """
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
    _require(roots.size == t, "One native tree root per hold")
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
    tree_of_hold = owner[samples[:, 0]]
    _require(np.unique(tree_of_hold).size == t
             and (owner[samples] == tree_of_hold[:, None]).all(),
             "Original sample-row order identifies every complete tree exactly once")

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


@dataclass(frozen=True)
class Order2CutBundleV1:
    grid: MarkerGridResetBundleV1
    cut: CutPlanV1

    @property
    def bank(self):
        return self.grid.bank

    @property
    def source_bank(self):
        return self.grid.source_bank

    @property
    def markers(self):
        return self.grid.markers

    @property
    def features_cpu(self):
        return self.grid.features_cpu

    @property
    def flags(self):
        return self.grid.flags

    def tensors(self):
        return self.grid.tensors() + self.cut.tensors()

    def tensor_bytes(self):
        return sum(x.numel() * x.element_size() for x in self.tensors())

    def validate(self):
        _require(isinstance(self.grid, MarkerGridResetBundleV1), "Unchanged prepared H048 grid bundle")
        bank, markers = self.grid.validate()
        _require(tuple(bank.sample_index.shape)[1:] == (952,)
                 and tuple(bank.query_column.shape) == (686,), "Complete original952/686 support")
        self.cut.validate(bank)
        return bank, markers

    def to(self, device):
        self.validate()
        moved = self.grid.to(device)
        plan = self.cut._to_for_bank(device, self.grid.bank, moved.bank)
        return type(self)(moved, plan)


def prepare_order2_bundle_v1(grid, **bounds):
    _require(isinstance(grid, MarkerGridResetBundleV1), "Root supplies original H048 preparation")
    bank, _ = grid.validate()
    _require(bank.parent.device.type == "cpu" and bank.sample_index.shape[1] == 952
             and bank.query_column.shape == (686,), "Unchanged complete CPU952/686 grid support")
    plan, metadata = prepare_cut_plan_v1(bank, expected_haplotypes=952, **bounds)
    result = Order2CutBundleV1(grid, plan)
    result.validate()
    return result, metadata


class CutMemoryV1(nn.Module):
    """Known incidence aggregation on cut-pair cells or matched stable leaves."""
    def __init__(self, *, arm=ARMS[0], checkpoint_memory=True):
        super().__init__()
        _require(arm in ARMS, "One fixed matched memory arm")
        self.arm = arm
        self.checkpoint_memory = bool(checkpoint_memory)
        self.U = nn.Linear(32, RANK, bias=False)
        self.O = nn.Linear(RANK, 32, bias=False)
        theta = math.log((5000.-100.)/(100000.-5000.))
        self.ell_theta = nn.Parameter(torch.tensor(theta, dtype=torch.float32))
        self.sum = SumAggregation()
        nn.init.zeros_(self.O.weight)
        _require(sum(p.numel() for p in self.parameters()) == ADDED_PARAMETERS, "Exactly513 memory scalars")

    @property
    def ell_bp(self):
        return 100.+(100000.-100.)*self.ell_theta.sigmoid()

    def _mean(self, values, index, counts, rows):
        total = self.sum(values, index=index, dim_size=rows, dim=0)
        return total / counts.clamp_min(1.)[:, None]

    def _direction(self, forcing, spans, plan, reverse):
        h, t = plan.haplotypes, plan.native_holds
        cells = plan.reverse_cells if reverse else plan.forward_cells
        cur = plan.reverse_current_group if reverse else plan.forward_current_group
        old = plan.reverse_old_group if reverse else plan.forward_old_group
        counts = plan.reverse_counts if reverse else plan.forward_counts
        order = range(t-1, -1, -1) if reverse else range(t)
        result = [None]*t
        state = None
        previous_episode = None
        rho = torch.exp(-spans/self.ell_bp)
        source_factor = -torch.expm1(-spans/self.ell_bp)
        for at in order:
            e = plan.episode_of_hold[at]
            if self.arm == ARMS[2]:
                force = forcing.index_select(0, plan.leaf_groups[at])
                if state is None:
                    state = force
                else:
                    state = rho[at]*state + source_factor[at]*force
                result[at] = state
                continue
            active = (counts[e] > 0).to(forcing.dtype)[:, None]
            force = forcing.index_select(0, cur[e]+plan.group_offsets[at])*active
            if state is None:
                state = force
            else:
                if e != previous_episode:
                    lifted = state.index_select(0, cells[previous_episode])
                    if self.arm == ARMS[0]:
                        state = self._mean(lifted, cells[e], counts[e], h)
                    else:
                        # n_AB/n_B null, independent of the outgoing C inside B.
                        pooled = self._mean(lifted, plan.episode_groups[previous_episode],
                                            plan.episode_group_counts[previous_episode], h)
                        state = pooled.index_select(0, old[e])*active
                state = (rho[at]*state + source_factor[at]*force)*active
            result[at] = state.index_select(0, cells[e])
            previous_episode = e
        return torch.stack(result, 0)

    def _history(self, forcing, spans, plan):
        left = self._direction(forcing, spans, plan, False)
        right = self._direction(forcing, spans, plan, True)
        leaf = .5*(left+right)
        read = self._mean(leaf.flatten(0, 1), plan.leaf_groups.flatten(),
                          plan.group_counts, forcing.shape[0])
        return read-forcing

    def forward(self, upward, outside, spans, plan):
        _require(upward.dtype == outside.dtype == spans.dtype == torch.float32
                 and upward.shape == outside.shape == (plan.native_nodes, 32)
                 and spans.shape == (plan.native_holds,)
                 and upward.device == plan.representatives.device,
                 "Same-device FP32 complete first-sweep fields")
        representatives = plan.representatives
        force = self.U(upward.index_select(0, representatives)
                       + outside.index_select(0, representatives))
        if self.checkpoint_memory and self.training and torch.is_grad_enabled():
            value = checkpoint(self._history, force, spans, plan, use_reentrant=False)
        else:
            value = self._history(force, spans, plan)
        return self.O(value)  # Compact[K_total,32], no dense node-pair tensor.


class Order2CutMarkerGridV1(nn.Module):
    """Actual full H042/PyG + memory + unchanged common H048 grid decoder."""
    def __init__(self, *, arm=ARMS[0], query_block=32,
                 checkpoint_chain=True, checkpoint_memory=True):
        super().__init__()
        self.common = MarkerGridResetChainGNNV1(
            arm="graph_to_transition", query_block=query_block,
            checkpoint_chain=checkpoint_chain)
        self.memory = CutMemoryV1(arm=arm, checkpoint_memory=checkpoint_memory)
        self.float()
        _require(sum(p.numel() for p in self.parameters()) == TOTAL_PARAMETERS,
                 "Exactly31933+513 weights in every arm")

    @property
    def arm(self):
        return self.memory.arm

    def _backbone_state(self, bank, flags, cut):
        cut.validate(bank)
        _require(bank.sample_index.shape[1] == 952 and bank.query_column.shape == (686,)
                 and flags.native_holds == len(bank.spans)
                 and all(p.dtype == torch.float32 for p in self.parameters()),
                 "Full original FP32 support and fixed flags")
        body = self.common.backbone
        base, ug, dg, degree = body.prepare_tree_base_v1(bank, control="aligned")
        upward, outside = body.sweep_tree_state_v1(bank, base, ug, dg, degree)
        injection = body.clade_coupling(upward, flags)
        delta = self.memory(upward, outside, bank.spans, cut)
        injection = injection.index_add(0, cut.representatives, delta)
        second_up, second_out = body.sweep_tree_state_v1(
            bank, base+injection, ug, dg, degree)
        context = body.leaf_context_v1(bank, second_up, second_out)
        # Exact inherited H042/H048 full-parent genomic geometry/GRU calculation.
        t, h, _ = context.shape
        span = bank.spans.log1p()/math.log(2000001.)
        prev = torch.cat((bank.spans.new_zeros(1), .5*(bank.spans[1:]+bank.spans[:-1])))
        nxt = torch.cat((.5*(bank.spans[1:]+bank.spans[:-1]), bank.spans.new_zeros(1)))
        geometry = torch.stack((span, prev.log1p()/math.log(2000001.),
                                nxt.log1p()/math.log(2000001.)), -1)
        sequence = torch.cat((context, geometry[:, None].expand(t, h, 3)), -1)
        history, _ = body.genomic_gru(sequence)
        return torch.cat((context, history), -1)

    def _decode_state(self, state, bank, grid):
        # Reuse the existing decoder method itself, not a reimplementation.
        base = self.common.backbone.classifier(state).index_select(1, bank.query_column)
        context = state[..., :32].index_select(1, bank.query_column)
        blocks = []
        for first in range(0, base.shape[1], self.common.query_block):
            last = min(first+self.common.query_block, base.shape[1])
            args = (base[:, first:last], context[:, first:last], grid.features_cpu[:, first:last],
                    grid.grid_hold, grid.delta_bp, grid.pair_left, grid.pair_right, grid.edge_pair)
            if self.common.checkpoint_chain and self.training and torch.is_grad_enabled():
                value = checkpoint(self.common.decode_block_v1, *args, use_reentrant=False)
            else:
                value = self.common.decode_block_v1(*args)
            blocks.append(value)
        return torch.cat(blocks, 1)

    def forward(self, bundle):
        _require(isinstance(bundle, Order2CutBundleV1), "Explicit bound grid/cut bundle")
        bank, _ = bundle.validate()
        state = self._backbone_state(bank, bundle.grid.flags, bundle.cut)
        return self._decode_state(state, bank, bundle.grid)


def make_matched_family_v1(payload, *, checkpoint_ref, completion_ref, device="cpu",
                            query_block=32, checkpoint_chain=True, checkpoint_memory=True):
    """Root supplies an authenticated actual H042 payload; this function reads no file."""
    _require(isinstance(payload, Mapping) and checkpoint_ref == H042_PASS30_REF
             and completion_ref == H042_COMPLETION_REF, "Bound original H042 pass30 references")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(101)
        original = Order2CutMarkerGridV1(
            arm=ARMS[0], query_block=query_block, checkpoint_chain=checkpoint_chain,
            checkpoint_memory=checkpoint_memory)
        inherited = load_h042_grid_reset_warmstart_v1(
            original.common, payload, checkpoint_ref=checkpoint_ref, completion_ref=completion_ref)
        family = {ARMS[0]: original}
        for arm in ARMS[1:]:
            family[arm] = copy.deepcopy(original)
            family[arm].memory.arm = arm
    for model in family.values():
        model.to(device)
    optimizers = {arm: torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001,
                    betas=(.9, .999), eps=1e-8) for arm, model in family.items()}
    proof = dict(schema="order2_cut_memory_matched_family_v1", arms=list(ARMS),
                 common_decoder="graph_to_transition", parameters_per_arm=TOTAL_PARAMETERS,
                 inherited_grid_parameters=31933, added_memory_parameters=ADDED_PARAMETERS,
                 all_parameters_trainable=all(p.requires_grad for m in family.values() for p in m.parameters()),
                 fresh_AdamW_per_arm=True, optimizer_updates_per_arm_required=960,
                 passes_required=30, inherited_warmstart_check=inherited,
                 checkpoint_bytes_and_runtime_authentication="external_root_responsibility")
    return family, optimizers, proof


def logical_bounds_v1(native_nodes, native_holds, *, haplotypes=952):
    """Named upper tables only, not a total allocator/admission certificate."""
    _require(all(type(x) is int and x > 0 for x in
                 (native_nodes, native_holds, haplotypes)), "Positive metadata counts")
    th = native_holds*haplotypes
    return dict(
        native_nodes=native_nodes, native_holds=native_holds, haplotypes=haplotypes,
        complete_cut_plan_tensor_upper_bytes=89*th+haplotypes,
        one_rank8_FP32_hold_leaf_bank_bytes=4*RANK*th,
        both_direction_rank8_banks_bytes=8*RANK*th,
        rank8_compact_forcing_upper_bytes=4*RANK*th,
        compact32channel_injection_upper_bytes=4*32*th,
        preparation_work_estimate_bytes=128*native_nodes+512*th+512*haplotypes+64*1024**2,
        preparation_work_estimate_is_not_an_RSS_bound=True,
        backbone_marker_chain_optimizer_autograd_and_workspaces_included=False,
        recurrence_parallel_speedup_claimed=False)
