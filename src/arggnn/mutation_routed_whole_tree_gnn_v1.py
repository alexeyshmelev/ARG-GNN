"""Directly supervised mutation-routed whole-tree prototype (source-only).

The only control difference is placement of supplied branch-support vectors.
Observed categorical alleles are lookup indexes, never learned embeddings.
No ancestry targets, frozen emitters, teachers, or residual logits are inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import math
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

CONTROL_NAMES = ("aligned", "shuffled")
_FLOAT_FIELDS = (
    "node_time", "edge_geometry", "branch_support_true", "branch_support_sham",
    "log_donor_allele_frequency", "panel_allele_frequency", "site_entropy",
    "spans", "sample_age",
)
_INDEX_FIELDS = ("parent", "sample_index", "site_hold", "readout_hold", "query_column",
                 "reference_class")
_LEVEL_FIELDS = ("upward_levels", "upward_children", "upward_parent_rank", "downward_levels")


@dataclass(frozen=True)
class MutationRoutedWholeTreeInputsV1:
    parent: Tensor                       # [N], root=-1, no cross-tree edges
    node_time: Tensor                    # [N], observed generations
    sample_index: Tensor                 # [T,H], persistent haplotype column order
    upward_levels: tuple[Tensor, ...]     # all N nodes, children before parents
    upward_children: tuple[Tensor, ...]   # children gathered at each upward level
    upward_parent_rank: tuple[Tensor, ...]  # child -> rank in that level
    downward_levels: tuple[Tensor, ...]   # nonroots, parents before children
    edge_geometry: Tensor                # [N,4], documented encoded geometry
    branch_support_true: Tensor          # [N,5]
    branch_support_sham: Tensor           # [N,5], placement-only control
    genotype: Tensor                     # [S,H], -1 missing, categorical lookup only
    log_donor_allele_frequency: Tensor    # [S,A,8], observed proxy panel only
    donor_call_mask: Tensor               # [S,8], bool
    panel_allele_frequency: Tensor        # [S,A]
    site_entropy: Tensor                 # [S]
    site_hold: Tensor                    # [S], original interval membership
    spans: Tensor                        # [T], positive physical interval widths
    readout_hold: Tensor                 # [R], exact scoring-center membership
    query_column: Tensor                 # [Q], actual-query columns only
    reference_class: Tensor               # [H], -1 unknown or 0..3 observed proxy
    sample_age: Tensor                    # [H], observed sampling date

    @classmethod
    def from_arrays(cls, arrays: Mapping, *, device="cpu"):
        """Convert an explicit observation-only mapping, rejecting extra keys.

        Shape/topology validation is a separate preparation-time CPU action.
        A loader must select these exact keys rather than pass a truth payload.
        """
        expected = {f.name for f in fields(cls)}
        if set(arrays) != expected:
            raise ValueError(f"Exact observation keys required; missing={sorted(expected-set(arrays))}, "
                             f"extra={sorted(set(arrays)-expected)}")
        converted = {}
        for name in expected:
            value = arrays[name]
            if name in _LEVEL_FIELDS:
                converted[name] = tuple(torch.as_tensor(x, dtype=torch.long, device=device) for x in value)
            elif name in _INDEX_FIELDS:
                converted[name] = torch.as_tensor(value, dtype=torch.long, device=device)
            elif name == "genotype":
                value = torch.as_tensor(value, device=device)
                if value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
                    raise ValueError("Observed genotype must have integer dtype")
                converted[name] = value
            elif name == "donor_call_mask":
                value = torch.as_tensor(value, device=device)
                if value.dtype != torch.bool:
                    raise ValueError("Donor callability must have bool dtype")
                converted[name] = value
            else:
                converted[name] = torch.as_tensor(value, dtype=torch.float32, device=device)
        return cls(**converted)

    def to(self, device):
        """Move the compact bank, preserving FP32 and categorical integer types."""
        values = {}
        for f in fields(self):
            value = getattr(self, f.name)
            values[f.name] = (tuple(x.to(device) for x in value) if f.name in _LEVEL_FIELDS
                              else value.to(device))
        return replace(self, **values)


def validate_inputs_v1(bank: MutationRoutedWholeTreeInputsV1, *, expected_haplotypes=280):
    """Structural/missingness checks for a CPU observation bank, without targets.

    Invoke once before transfer, not on every optimizer step. This validates
    representation, not the source provenance or biological purity of proxies.
    Raw CPU chronology is certified by preprocessing. Equal rounded FP32
    node clocks are valid when raw-derived duration geometry remains positive.
    """
    if not isinstance(bank, MutationRoutedWholeTreeInputsV1):
        raise TypeError("Expected the observation-only dataclass")
    for f in fields(bank):
        value = getattr(bank, f.name)
        tensors = value if f.name in _LEVEL_FIELDS else (value,)
        if any(not isinstance(x, Tensor) or x.device.type != "cpu" for x in tensors):
            raise ValueError("Preparation validation requires CPU tensors")
    if any(getattr(bank, k).dtype != torch.float32 for k in _FLOAT_FIELDS):
        raise ValueError("Prototype floating observations must be FP32")
    if any(getattr(bank, k).dtype != torch.long for k in _INDEX_FIELDS):
        raise ValueError("Topology/identity lookup tensors must be int64")
    if bank.genotype.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("Categorical integer genotypes required")
    if bank.donor_call_mask.dtype != torch.bool:
        raise ValueError("Call mask must be boolean")
    n = bank.parent.numel()
    if bank.parent.ndim != 1 or not n or bank.node_time.shape != (n,):
        raise ValueError("Nonempty node geometry required")
    if bank.sample_index.ndim != 2:
        raise ValueError("sample_index must be [T,H]")
    t, h = bank.sample_index.shape
    if not t or h != expected_haplotypes:
        raise ValueError("Every original interval must retain the expected full sampled panel")
    if (bank.edge_geometry.shape != (n, 4) or bank.branch_support_true.shape != (n, 5)
            or bank.branch_support_sham.shape != (n, 5) or bank.spans.shape != (t,)
            or bank.reference_class.shape != (h,) or bank.sample_age.shape != (h,)):
        raise ValueError("Geometry/support/sample field shapes differ")
    if bank.genotype.ndim != 2 or bank.genotype.shape[1] != h:
        raise ValueError("Genotype columns must follow persistent sample order")
    s = bank.genotype.shape[0]
    if bank.log_donor_allele_frequency.ndim != 3:
        raise ValueError("Donor log frequencies must be [S,A,8]")
    a = bank.log_donor_allele_frequency.shape[1]
    if (not 1 <= a <= 4 or bank.log_donor_allele_frequency.shape != (s, a, 8)
            or bank.panel_allele_frequency.shape != (s, a)
            or bank.donor_call_mask.shape != (s, 8) or bank.site_entropy.shape != (s,)
            or bank.site_hold.shape != (s,)):
        raise ValueError("All original categorical site arrays must agree")
    if (bank.readout_hold.ndim != 1 or bank.query_column.ndim != 1
            or not bank.readout_hold.numel() or not bank.query_column.numel()):
        raise ValueError("Exact nonempty readout/query index vectors required")
    def finite(x):
        return bool(torch.isfinite(x).all())
    for name in _FLOAT_FIELDS:
        if name != "log_donor_allele_frequency" and not finite(getattr(bank, name)):
            raise ValueError(f"Nonfinite {name}")
    if (bool((bank.node_time < 0).any()) or bool((bank.sample_age < 0).any())
            or bool((bank.spans <= 0).any()) or bool((bank.site_entropy < 0).any())):
        raise ValueError("Observed ages/entropy must be nonnegative and spans positive")
    if (bool((bank.parent < -1).any()) or bool((bank.parent >= n).any())
            or bool((bank.sample_index < 0).any()) or bool((bank.sample_index >= n).any())
            or bank.sample_index.unique().numel() != t*h):
        raise ValueError("Every sample must have a unique per-tree native vertex")
    if (bool((bank.genotype < -1).any()) or bool((bank.genotype >= a).any())
            or bool((bank.site_hold < 0).any()) or bool((bank.site_hold >= t).any())
            or bool((bank.site_hold[1:] < bank.site_hold[:-1]).any())
            or bool((bank.readout_hold < 0).any()) or bool((bank.readout_hold >= t).any())
            or bool((bank.query_column < 0).any()) or bool((bank.query_column >= h).any())
            or bank.query_column.unique().numel() != bank.query_column.numel()
            or bool((bank.reference_class < -1).any()) or bool((bank.reference_class > 3).any())):
        raise ValueError("Invalid categorical, interval, query or proxy-class lookup")
    if h == 280 and (bank.query_column.numel() != 120
            or any(int((bank.reference_class == c).sum()) != 24 for c in range(4))):
        raise ValueError("The declared 280-panel requires actual query120 and pure R96, 24 per class")
    if bool((bank.reference_class[bank.query_column] != -1).any()):
        raise ValueError("Scored queries may not carry input reference labels")
    if bool((bank.panel_allele_frequency < 0).any()) or bool((bank.panel_allele_frequency > 1).any()):
        raise ValueError("Panel frequencies must be probabilities")
    if s:
        chosen = bank.genotype.long().clamp_min(0)
        called = bank.genotype >= 0
        donor = bank.log_donor_allele_frequency.gather(1, chosen[..., None].expand(-1, -1, 8))
        active = called[..., None] & bank.donor_call_mask[:, None, :]
        if not bool(torch.isfinite(donor[active]).all()) or bool((donor[active] > 1e-6).any()):
            raise ValueError("Called/available donor log frequencies must be finite and <=0")
        panel = bank.panel_allele_frequency.gather(1, chosen)
        if bool((panel[called] <= 0).any()):
            raise ValueError("Called allele must have positive observed panel frequency")
    parent = bank.parent.tolist()
    children = [[] for _ in range(n)]
    for child, p in enumerate(parent):
        if p >= 0:
            if p == child or bank.node_time[p] < bank.node_time[child]:
                raise ValueError("Rounded FP32 node clocks must not reverse parent chronology")
            if bank.edge_geometry[child, 2] <= 0:
                raise ValueError("Each retained nonroot requires positive raw-derived duration geometry")
            children[p].append(child)
    roots = [v for v, p in enumerate(parent) if p == -1]
    if len(roots) != t:
        raise ValueError("One complete connected tree is required per interval")
    if (len(bank.upward_levels) != len(bank.upward_children)
            or len(bank.upward_levels) != len(bank.upward_parent_rank)):
        raise ValueError("Upward level/child/rank tuples must agree")
    up_order = [int(v) for level in bank.upward_levels for v in level.tolist()]
    down_order = [int(v) for level in bank.downward_levels for v in level.tolist()]
    if sorted(up_order) != list(range(n)) or sorted(down_order) != sorted(v for v in range(n) if parent[v] >= 0):
        raise ValueError("Sweeps must retain every node and every nonroot")
    up_rank = {v: level for level, nodes in enumerate(bank.upward_levels) for v in nodes.tolist()}
    down_rank = {v: level for level, nodes in enumerate(bank.downward_levels) for v in nodes.tolist()}
    for level, ch, ranks in zip(bank.upward_levels, bank.upward_children, bank.upward_parent_rank):
        if any(x.dtype != torch.long or x.ndim != 1 for x in (level, ch, ranks)) or ch.shape != ranks.shape:
            raise ValueError("Level arrays must be one-dimensional int64")
        expected = [(child, rank) for rank, v in enumerate(level.tolist()) for child in children[v]]
        if sorted(zip(ch.tolist(), ranks.tolist())) != sorted(expected):
            raise ValueError("Upward children/ranks must enumerate exactly the original edges")
        for child in ch.tolist():
            if up_rank[child] >= up_rank[parent[child]]:
                raise ValueError("Upward order must put each child before its parent")
    for level in bank.downward_levels:
        if level.dtype != torch.long or level.ndim != 1:
            raise ValueError("Downward levels must be int64 vectors")
        for v in level.tolist():
            p = parent[v]
            if p >= 0 and parent[p] >= 0 and down_rank[p] >= down_rank[v]:
                raise ValueError("Downward order must put each parent before its child")
    root_of = [-1]*n
    for v in reversed(up_order):
        root_of[v] = v if parent[v] < 0 else root_of[parent[v]]
    if any(r < 0 for r in root_of):
        raise ValueError("Parent structure is cyclic or sweep order invalid")
    owned_roots = set()
    sampled = set(bank.sample_index.flatten().tolist())
    for hold, row in enumerate(bank.sample_index.tolist()):
        rr = {root_of[v] for v in row}
        if len(rr) != 1 or next(iter(rr)) in owned_roots:
            raise ValueError("Sample columns must belong to their own complete interval tree")
        owned_roots.update(rr)
        if any(children[v] for v in row):
            raise ValueError("This prototype requires sampled leaves, not sampled interiors")
        if not torch.equal(bank.node_time[bank.sample_index[hold]], bank.sample_age):
            raise ValueError("Persistent sampled ages changed across intervals")
    if {v for v in range(n) if not children[v]} != sampled:
        raise ValueError("Every tree leaf must be one of the retained sampled haplotypes")
    if bool((bank.edge_geometry[roots] != 0).any()):
        raise ValueError("Root edge geometry must be zero")
    for name in ("branch_support_true", "branch_support_sham"):
        support = getattr(bank, name)
        if bool((support < 0).any()) or bool((support[:, :3] > 1+1e-6).any()):
            raise ValueError("Nonnegative support/counts and unit-scaled summaries required")
        if bool(((support[:, 4] != 0) & (support[:, 4] != 1)).any()):
            raise ValueError("Empty support flag must be binary")
        if not torch.equal(support[roots], torch.tensor([0., 0., 0., 0., 1.]).expand(t, 5)):
            raise ValueError("Root support must be [0,0,0,0,1] in both arms")
        empty = support[:, 4] == 1
        if bool((support[empty, :4] != 0).any()):
            raise ValueError("Empty branch evidence must have zero support/count fields")
    return {"nodes": n, "trees": t, "haplotypes": h, "sites": s,
            "queries": bank.query_column.numel(), "readouts": bank.readout_hold.numel()}


class MutationRoutedWholeTreeGNNV1(nn.Module):
    """Width-32 direct ancestry classifier over complete observed tree collections."""
    def __init__(self, hidden_dim=32, *, expected_haplotypes=280,
                 site_chunk_size=256, checkpoint_sites=True):
        super().__init__()
        if any(type(x) is not int or x < 1 for x in (hidden_dim, expected_haplotypes, site_chunk_size)):
            raise ValueError("Positive integer widths/panel/chunk size required")
        self.hidden_dim = hidden_dim
        self.expected_haplotypes = expected_haplotypes
        self.site_chunk_size = site_chunk_size
        self.checkpoint_sites = bool(checkpoint_sites)
        self.site_mlp = nn.Sequential(nn.Linear(19, hidden_dim), nn.GELU(),
                                      nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.site_pool_projection = nn.Sequential(nn.Linear(2*hidden_dim+3, hidden_dim), nn.Tanh())
        self.leaf_projection = nn.Sequential(nn.Linear(hidden_dim+6, hidden_dim), nn.Tanh())
        self.node_projection = nn.Sequential(nn.Linear(3, hidden_dim), nn.Tanh())
        self.up_gate = nn.Linear(9, hidden_dim)
        self.down_gate = nn.Linear(9, hidden_dim)
        self.up_message = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.up_update = nn.Linear(2*hidden_dim, hidden_dim)
        self.up_norm = nn.LayerNorm(hidden_dim)
        self.down_message = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.down_update = nn.Linear(hidden_dim, hidden_dim)
        self.down_norm = nn.LayerNorm(hidden_dim)
        self.leaf_context = nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.Tanh())
        self.genomic_gru = nn.GRU(hidden_dim+3, hidden_dim, num_layers=1, bidirectional=True)
        self.classifier = nn.Sequential(nn.Linear(3*hidden_dim, hidden_dim), nn.GELU(),
                                        nn.Linear(hidden_dim, 4))

    def _site_chunk(self, genotype, donor, donor_mask, panel, entropy, inverse, count_holds):
        """Lookup and locally reduce one chunk without a full S*H*19 feature cache."""
        h, d = genotype.shape[1], self.hidden_dim
        called = genotype >= 0
        chosen = genotype.long().clamp_min(0)
        logfreq = donor.gather(1, chosen[..., None].expand(-1, -1, 8))
        available = donor_mask[:, None, :].expand(-1, h, -1) & called[..., None]
        logfreq = torch.where(available, logfreq, torch.zeros_like(logfreq))
        panel_chosen = panel.gather(1, chosen)
        panel_log = torch.where(called, panel_chosen, torch.ones_like(panel_chosen)).log()
        ent = entropy[:, None].expand(-1, h)
        tuple19 = torch.cat((logfreq, available.to(logfreq.dtype), panel_log[..., None],
                             ent[..., None], called[..., None].to(logfreq.dtype)), -1)
        tuple19 = torch.where(called[..., None], tuple19, torch.zeros_like(tuple19))
        encoded = self.site_mlp(tuple19)
        index = (inverse[:, None]*h + torch.arange(h, device=genotype.device)[None]).flatten()
        values = torch.where(called[..., None], encoded, torch.zeros_like(encoded)).reshape(-1, d)
        sums = values.new_zeros(count_holds*h, d).index_add(0, index, values)
        counts = values.new_zeros(count_holds*h, 1).index_add(
            0, index, called.to(values.dtype).reshape(-1, 1))
        max_values = torch.where(called[..., None], encoded, torch.full_like(encoded, -torch.inf)).reshape(-1, d)
        maxima = values.new_full((count_holds*h, d), -torch.inf).scatter_reduce(
            0, index[:, None].expand_as(max_values), max_values, reduce="amax", include_self=True)
        maxima = torch.where(counts > 0, maxima, torch.zeros_like(maxima))
        return sums.reshape(count_holds, h, d), maxima.reshape(count_holds, h, d), counts.reshape(count_holds, h, 1)

    def encode_sites(self, bank):
        t, h = bank.sample_index.shape
        d = self.hidden_dim
        pool_holds, pool_sums, pool_max, pool_count = [], [], [], []
        for start in range(0, len(bank.genotype), self.site_chunk_size):
            end = min(start+self.site_chunk_size, len(bank.genotype))
            holds, inverse = torch.unique_consecutive(bank.site_hold[start:end], return_inverse=True)
            args = (bank.genotype[start:end], bank.log_donor_allele_frequency[start:end],
                    bank.donor_call_mask[start:end], bank.panel_allele_frequency[start:end],
                    bank.site_entropy[start:end], inverse, len(holds))
            if self.checkpoint_sites and self.training and torch.is_grad_enabled():
                sums, maximum, count = checkpoint(self._site_chunk, *args, use_reentrant=False)
            else:
                sums, maximum, count = self._site_chunk(*args)
            pool_holds.append(holds); pool_sums.append(sums); pool_max.append(maximum); pool_count.append(count)
        sums = bank.spans.new_zeros(t, h, d)
        maxima = bank.spans.new_full((t, h, d), -torch.inf)
        counts = bank.spans.new_zeros(t, h, 1)
        if pool_holds:
            index = torch.cat(pool_holds)
            sums = sums.index_add(0, index, torch.cat(pool_sums))
            counts = counts.index_add(0, index, torch.cat(pool_count))
            chunk_max = torch.cat(pool_max)
            chunk_count = torch.cat(pool_count)
            chunk_max = torch.where(chunk_count > 0, chunk_max, torch.full_like(chunk_max, -torch.inf))
            maxima = maxima.scatter_reduce(
                0, index[:, None, None].expand_as(chunk_max), chunk_max, reduce="amax", include_self=True)
        present = counts > 0
        means = sums / counts.clamp_min(1)
        maxima = torch.where(present, maxima, torch.zeros_like(maxima))
        logcount = counts.log1p()/math.log(100001.)
        empty = (~present).to(sums.dtype)
        span = (bank.spans.log1p()/math.log(2000001.))[:, None, None].expand(t, h, 1)
        pooled = torch.cat((means, maxima, logcount, empty, span), -1)
        return self.site_pool_projection(pooled)

    def encode_trees(self, bank, *, control):
        t, h = bank.sample_index.shape
        n, d = len(bank.parent), self.hidden_dim
        site_leaf = self.encode_sites(bank)
        labels = bank.reference_class
        onehot = torch.nn.functional.one_hot(labels.clamp_min(0), 4).to(site_leaf.dtype)
        onehot = onehot * (labels >= 0)[:, None]
        unknown = (labels < 0).to(site_leaf.dtype)[:, None]
        age = (bank.sample_age.log1p()/math.log(10001.))[:, None]
        observed_leaf = torch.cat((onehot, unknown, age), -1)[None].expand(t, h, 6)
        leaf_base = self.leaf_projection(torch.cat((site_leaf, observed_leaf), -1))
        sampled = bank.node_time.new_zeros(n)
        sampled.index_fill_(0, bank.sample_index.flatten(), 1.)
        root = (bank.parent < 0).to(sampled.dtype)
        node_age = bank.node_time.log1p()/math.log(10001.)
        base = self.node_projection(torch.stack((node_age, root, sampled), -1))
        base = base.index_add(0, bank.sample_index.flatten(), leaf_base.reshape(-1, d))
        raw = bank.branch_support_true if control == "aligned" else bank.branch_support_sham
        support = torch.cat((raw[:, :3], raw[:, 3:4].log1p()/math.log(100001.), raw[:, 4:5]), -1)
        gate_input = torch.cat((bank.edge_geometry, support), -1)
        ug, dg = self.up_gate(gate_input).sigmoid(), self.down_gate(gate_input).sigmoid()
        nonroot = bank.parent >= 0
        degree = torch.bincount(bank.parent[nonroot], minlength=n).to(base.dtype).clamp_min(1)
        upward = base.new_zeros(n, d)
        message_sum = base.new_zeros(n, d)
        child_message = base.new_zeros(n, d)
        for nodes, children, ranks in zip(bank.upward_levels, bank.upward_children, bank.upward_parent_rank):
            aggregate = base.new_zeros(len(nodes), d)
            if children.numel():
                message = self.up_message(upward[children])*ug[children]
                message = message/degree[bank.parent[children], None]
                child_message.index_copy_(0, children, message)
                aggregate.index_add_(0, ranks, message)
            message_sum.index_copy_(0, nodes, aggregate)
            value = self.up_norm(base[nodes] + self.up_update(torch.cat((base[nodes], aggregate), -1)).tanh())
            upward.index_copy_(0, nodes, value)
        outside = base.new_zeros(n, d)
        for nodes in bank.downward_levels:
            parents = bank.parent[nodes]
            cavity = outside[parents] + base[parents] + message_sum[parents] - child_message[nodes]
            value = self.down_message(cavity)*dg[nodes]
            value = self.down_norm(self.down_update(value).tanh())
            outside.index_copy_(0, nodes, value)
        # Recipient subtree is introduced only at final readout, not into outside
        # recursion. Hence the outside message excludes the receiving child.
        context = self.leaf_context(torch.cat((upward[bank.sample_index], outside[bank.sample_index]), -1))
        return context

    def forward_holds(self, bank: MutationRoutedWholeTreeInputsV1, *, control="aligned"):
        """Return [T,H,4] logits, decoding once per original tree/sample hold."""
        if control not in CONTROL_NAMES:
            raise ValueError(f"Control must be one of {CONTROL_NAMES}")
        parameter = next(self.parameters())
        if parameter.dtype != torch.float32:
            raise ValueError("This discovery prototype requires FP32 parameters")
        t, h = bank.sample_index.shape
        if h != self.expected_haplotypes or not t or bank.parent.device != parameter.device:
            raise ValueError("Full sampled panel and bank/model device must agree")
        for name in _FLOAT_FIELDS:
            value = getattr(bank, name)
            if value.dtype != torch.float32 or value.device != parameter.device:
                raise ValueError("All floating bank fields must share FP32 model device")
        for f in fields(bank):
            value = getattr(bank, f.name)
            tensors = value if f.name in _LEVEL_FIELDS else (value,)
            if any(x.device != parameter.device for x in tensors):
                raise ValueError("Complete bank must share model device")
        context = self.encode_trees(bank, control=control)
        span = bank.spans.log1p()/math.log(2000001.)
        prev = torch.cat((bank.spans.new_zeros(1), .5*(bank.spans[1:]+bank.spans[:-1])))
        nxt = torch.cat((.5*(bank.spans[1:]+bank.spans[:-1]), bank.spans.new_zeros(1)))
        geometry = torch.stack((span, prev.log1p()/math.log(2000001.),
                                nxt.log1p()/math.log(2000001.)), -1)
        sequence = torch.cat((context, geometry[:, None].expand(t, h, 3)), -1)
        # One vectorized bidirectional scan. No reset at artificial tile edges,
        # no detached history, and no repeated tree encoding per readout.
        history, _ = self.genomic_gru(sequence)
        return self.classifier(torch.cat((context, history), -1))

    def forward(self, bank: MutationRoutedWholeTreeInputsV1, *, control="aligned"):
        """Return direct raw logits [R,Q,4], with no target/teacher argument."""
        holds = self.forward_holds(bank, control=control)
        return holds[bank.readout_hold[:, None], bank.query_column[None, :]]
