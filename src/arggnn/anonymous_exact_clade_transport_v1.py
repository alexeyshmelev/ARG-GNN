"""H043 exact changed-clade pairing and matched Torch/PyG layer.

Source validation only. No reader, training runner, targets, disk mask cache,
resource admission or internal-ancestor correspondence is provided.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .anonymous_clade_flag_control_v1 import (
    AnonymousCladeFlagsV1, AnonymousCladeFlagControlGNNV1,
    RankEightCladeFlagAdapterV1, validate_clade_flags_v1,
    load_h038_flag_warmstart_v1, _cpu_index_array,
    SCHEMA as FLAG_SCHEMA, CHANNELS,
)
from .mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeInputsV1

SOURCE = "src/arggnn/anonymous_exact_clade_transport_v1.py"
SCHEMA = "anonymous_exact_changed_clade_pairs_v1"
ARMS = ("true_partner", "own_state")
INDEX_FIELDS = ("boundary_index", "old_node", "new_node", "leaf_support", "intersection_count")
EVENT_FIELDS = (*INDEX_FIELDS, "weight")
ARRAY_FIELDS = ("node_flags", "node_size", *EVENT_FIELDS)
DEFAULT_PAYLOAD_CAP = 1073741824 - 67108864


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _forest_index(parent, samples):
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
    ordered = []
    for row in samples:
        root = int(roots[int(row[0])])
        _require(np.all(roots[row] == root), "Each complete sampled panel must lie in one tree")
        ordered.append(root)
    _require(len(set(ordered)) == samples.shape[0] == int(np.count_nonzero(parent < 0)),
             "One distinct full sampled tree per native hold required")
    return head, sibling, ordered, parent_steps


def _tree_clades(root, row, head, sibling, node_size):
    """One postorder; exact integer masks never leave the caller's tree pair."""
    sampled = {int(node): 1 << column for column, node in enumerate(row)}
    _require(len(sampled) == len(row), "Distinct sampled leaves required")
    preorder, stack = [], [root]
    while stack:
        node = stack.pop()
        preorder.append(node)
        child = int(head[node])
        while child >= 0:
            stack.append(child)
            child = int(sibling[child])
    masks, clades, unary, leaves = {}, {}, 0, 0
    for node in reversed(preorder):
        child = int(head[node])
        if child < 0:
            _require(node in sampled, "Every native terminal must be in the sampled panel")
            mask, child_count = sampled[node], 0
            leaves += 1
        else:
            _require(node not in sampled, "Sampled-interior cases are unsupported")
            mask, child_count = 0, 0
            while child >= 0:
                mask |= masks[child]
                child_count += 1
                child = int(sibling[child])
        masks[node] = mask
        size = mask.bit_count()
        node_size[node] = size
        if 1 < size < len(row):
            if mask in clades:
                _require(child_count == 1 and masks[int(head[node])] == mask,
                         "Equal proper clades must form a unary chain")
                unary += 1
            else:
                clades[mask] = node
    _require(leaves == len(row) and masks[root] == (1 << len(row)) - 1,
             "Exactly the full sampled-leaf panel required")
    return clades, len(preorder), unary


def _nearest_changed(root, row, changed, head, sibling):
    """Carry the deepest changed representative in one root-to-leaf traversal."""
    result = np.full(len(row), -1, dtype=np.int32)
    if not changed:
        return result, 0
    columns = {int(node): column for column, node in enumerate(row)}
    stack, visits = [(root, -1)], 0
    while stack:
        node, nearest = stack.pop()
        visits += 1
        if node in changed:
            nearest = node
        child = int(head[node])
        if child < 0:
            result[columns[node]] = nearest
        while child >= 0:
            stack.append((child, nearest))
            child = int(sibling[child])
    return result, visits


def build_clade_transport_arrays_v1(parent, sample_index, *, max_payload_bytes=DEFAULT_PAYLOAD_CAP):
    """Joint exact H042 flags + <=n nearest pairs per boundary, CPU topology only.

    Stable sample-column identity is an upstream observation-reader contract.
    Adjacent exact masks are ephemeral. No per-leaf ancestor climb is used.
    """
    parent, samples = np.asarray(parent), np.asarray(sample_index)
    _require(parent.ndim == 1 and 0 < parent.size < 2**31 and parent.dtype.kind == "i"
             and not np.any(parent < -1) and not np.any(parent >= parent.size),
             "Bounded signed-integer parent forest required")
    _require(samples.ndim == 2 and samples.dtype.kind == "i"
             and samples.shape[0] >= 1 and samples.shape[1] >= 3
             and not np.any(samples < 0) and not np.any(samples >= parent.size),
             "Complete stable sampled-leaf rows required")
    N, (T, n) = parent.size, samples.shape
    _require(n < 2**31 and type(max_payload_bytes) is int and 6*N <= max_payload_bytes,
             "Explicit positive compact-payload cap must include all flags and sizes")
    head, sibling, roots, parent_steps = _forest_index(parent, samples)
    flags = np.zeros((N, 2), dtype=np.uint8)
    sizes = np.zeros(N, dtype=np.int32)
    chunks = {name: [] for name in EVENT_FIELDS}
    previous = None
    clade_counts, node_counts, unary_counts, audits = [], [], [], []
    total_events = nearest_visits = total_leaf_events = visited = 0
    for hold, (root, row) in enumerate(zip(roots, samples)):
        current, count, unary = _tree_clades(root, row, head, sibling, sizes)
        visited += count
        clade_counts.append(len(current));node_counts.append(count);unary_counts.append(unary)
        if previous is not None:
            previous_clades, old_root, old_row = previous
            removed = {previous_clades[mask]: mask for mask in previous_clades if mask not in current}
            added = {current[mask]: mask for mask in current if mask not in previous_clades}
            for node in removed:
                flags[node, 1] = 1
            for node in added:
                flags[node, 0] = 1
            old, visits = _nearest_changed(old_root, old_row, removed, head, sibling)
            nearest_visits += visits
            new, visits = _nearest_changed(root, row, added, head, sibling)
            nearest_visits += visits
            pairs, row_mass, col_mass = {}, {}, {}
            leaf_events = 0
            for a, b in zip(old, new):
                a, b = int(a), int(b)
                if a == b == -1:
                    continue
                leaf_events += 1
                pairs[a, b] = pairs.get((a, b), 0) + 1
                row_mass[a] = row_mass.get(a, 0) + 1
                col_mass[b] = col_mass.get(b, 0) + 1
            total_events += len(pairs);total_leaf_events += leaf_events
            _require(6*N + 24*total_events <= max_payload_bytes,
                     "Exact complete pair payload exceeds cap; never subsample")
            records = {name: [] for name in EVENT_FIELDS}
            for (a, b), support in sorted(pairs.items()):
                intersection = ((removed[a] & added[b]).bit_count()
                                if a >= 0 and b >= 0 else 0)
                values = (hold-1, a, b, support, intersection,
                          support/math.sqrt(row_mass[a]*col_mass[b]))
                for name, value in zip(EVENT_FIELDS, values):
                    records[name].append(value)
            for name in EVENT_FIELDS:
                dtype = np.float32 if name == "weight" else np.int32
                chunks[name].append(np.asarray(records[name], dtype=dtype))
            covered_old = {a for a, _ in pairs if a >= 0}
            covered_new = {b for _, b in pairs if b >= 0}
            audits.append(dict(boundary_index=hold-1, removed_clades=len(removed),
                added_clades=len(added), raw_leaf_events=leaf_events, pairs=len(pairs),
                paired_leaf_support=sum(k for (a,b),k in pairs.items() if a >= 0 and b >= 0),
                one_sided_leaf_support=sum(k for (a,b),k in pairs.items() if a < 0 or b < 0),
                removed_without_nearest_support=len(removed)-len(covered_old),
                added_without_nearest_support=len(added)-len(covered_new)))
        previous = current, root, row
    _require(visited == N, "No omitted or unused forest vertices")
    del previous
    arrays = dict(node_flags=flags, node_size=sizes)
    for name in EVENT_FIELDS:
        arrays[name] = (np.concatenate(chunks[name]) if chunks[name] else
                        np.empty(0, dtype=np.float32 if name == "weight" else np.int32))
    flag_meta = dict(schema=FLAG_SCHEMA, channels=list(CHANNELS), native_holds=int(T),
        haplotypes=int(n), native_nodes=int(N), node_flag_shape=[int(N),2],
        node_flag_dtype="uint8", node_flag_bytes=int(flags.nbytes),
        topology_sha256=_sha(parent), sample_index_sha256=_sha(samples),
        node_flags_sha256=_sha(flags), proper_clades_per_tree=clade_counts,
        native_nodes_per_tree=node_counts, unary_equivalences_per_tree=unary_counts,
        gained_from_left_count=int(flags[:,0].sum()), lost_to_right_count=int(flags[:,1].sum()),
        representative_policy="deepest_structural_node_in_equal_clade_unary_chain",
        exact_bitsets_retained=False, internal_ancestor_correspondence_claimed=False,
        observation_only=True, labels_times_targets_or_outcomes_used=False)
    meta = dict(schema=SCHEMA, native_holds=int(T), haplotypes=int(n), native_nodes=int(N),
        event_count=total_events, raw_leaf_events=total_leaf_events,
        event_bound_per_boundary=int(n), raw_array_bytes=sum(a.nbytes for a in arrays.values()),
        max_payload_bytes=max_payload_bytes, array_sha256={k:_sha(v) for k,v in arrays.items()},
        flag_meta=flag_meta, boundary_audit=audits,
        root_parent_link_steps=parent_steps, clade_postorder_node_visits=visited,
        nearest_traversal_node_visits=nearest_visits, exact_bitsets_retained=False,
        internal_ancestor_correspondence_claimed=False, all_H042_flags_retained=True,
        changed_clade_endpoint_coverage_claimed=False,
        sample_column_identity_requires_upstream_verification=True,
        labels_times_targets_or_outcomes_used=False)
    return arrays, meta


@dataclass(frozen=True)
class AnonymousExactCladeTransportV1:
    flags: AnonymousCladeFlagsV1
    node_size: torch.Tensor
    boundary_index: torch.Tensor
    old_node: torch.Tensor
    new_node: torch.Tensor
    leaf_support: torch.Tensor
    intersection_count: torch.Tensor
    weight: torch.Tensor

    @property
    def native_nodes(self):return self.flags.native_nodes
    @property
    def native_holds(self):return self.flags.native_holds
    @property
    def haplotypes(self):return self.flags.haplotypes

    @classmethod
    def from_verified_arrays(cls, arrays, meta):
        _require(set(arrays) == set(ARRAY_FIELDS) and meta.get("schema") == SCHEMA
                 and meta.get("exact_bitsets_retained") is False
                 and meta.get("all_H042_flags_retained") is True
                 and meta.get("internal_ancestor_correspondence_claimed") is False
                 and meta.get("labels_times_targets_or_outcomes_used") is False,
                 "Explicit verified compact observation-only payload required")
        N,T,n = (meta[k] for k in ("native_nodes","native_holds","haplotypes"))
        _require(all(type(v) is int for v in (N,T,n)) and N > 0 and T >= 1 and n >= 3,
                 "Exact bounded payload dimensions")
        flags = AnonymousCladeFlagsV1.from_verified_arrays(arrays["node_flags"],meta["flag_meta"])
        _require((flags.native_nodes,flags.native_holds,flags.haplotypes) == (N,T,n),
                 "H042 flag dimensions differ")
        _require(all(isinstance(v,np.ndarray) and _sha(v) == meta["array_sha256"].get(k)
                     for k,v in arrays.items()), "Payload array SHA drift")
        size = arrays["node_size"]
        _require(size.dtype == np.int32 and size.shape == (N,)
                 and np.all((1 <= size) & (size <= n)), "Complete node descendant sizes")
        P = arrays["weight"].size
        _require(type(meta["event_count"]) is int and meta["event_count"] == P
                 and P <= n*(T-1), "At most n deduplicated pairs per boundary in aggregate")
        for name in INDEX_FIELDS:
            _require(arrays[name].dtype == np.int32 and arrays[name].shape == (P,),
                     "Compact int32 event vectors")
        weights = arrays["weight"]
        _require(weights.dtype == np.float32 and weights.shape == (P,)
                 and np.isfinite(weights).all() and np.all((weights > 0) & (weights <= 1.000001)),
                 "Finite positive normalized FP32 support")
        a,b,k,inter,boundary = (arrays[name] for name in
            ("old_node","new_node","leaf_support","intersection_count","boundary_index"))
        _require(np.all((0 <= boundary) & (boundary < T-1))
                 and np.all((-1 <= a) & (a < N)) and np.all((-1 <= b) & (b < N))
                 and np.all((a >= 0) | (b >= 0))
                 and np.all((1 <= k) & (k <= n)) and np.all((0 <= inter) & (inter <= n)),
                 "Bounded nonempty exact pair records")
        if P:
            old_real,new_real = a >= 0,b >= 0
            _require(np.all(arrays["node_flags"][a[old_real],1] == 1)
                     and np.all(arrays["node_flags"][b[new_real],0] == 1),
                     "Every endpoint has its retained lost/gained H042 flag")
            both = old_real & new_real
            _require(np.all(inter[~both] == 0) and np.all(inter[both] >= k[both])
                     and np.all(k[old_real] <= size[a[old_real]])
                     and np.all(k[new_real] <= size[b[new_real]])
                     and np.all(inter[both] <= np.minimum(size[a[both]],size[b[both]])),
                     "Exact support/intersection cardinality bounds")
            if P > 1:
                ordered = ((boundary[1:] > boundary[:-1]) |
                    ((boundary[1:] == boundary[:-1]) &
                     ((a[1:] > a[:-1]) | ((a[1:] == a[:-1]) & (b[1:] > b[:-1])))))
                _require(np.all(ordered), "Strict boundary/local-pair ordering and deduplication")
            per_boundary = np.bincount(boundary,weights=k,minlength=max(0,T-1))
            _require(np.all(per_boundary <= n), "At most n raw leaf events per boundary")
        _require(sum(v.nbytes for v in arrays.values()) == meta["raw_array_bytes"]
                 and meta["raw_array_bytes"] <= meta["max_payload_bytes"],
                 "Exact compact payload byte bound")
        return cls(flags,*(torch.from_numpy(arrays[name]) for name in ("node_size",*EVENT_FIELDS)))

    def to(self, device, *, non_blocking=False):
        return type(self)(self.flags.to(device,non_blocking=non_blocking),
            *(getattr(self,name).to(device,non_blocking=non_blocking)
              for name in ("node_size",*EVENT_FIELDS)))

    def tensor_bytes(self):
        return self.flags.tensor_bytes()+sum(getattr(self,name).numel()*getattr(self,name).element_size()
            for name in ("node_size",*EVENT_FIELDS))


def build_clade_transport_v1(bank, *, max_payload_bytes=DEFAULT_PAYLOAD_CAP):
    """Production preparation entry: same original full CPU observation bank."""
    if type(bank) is not MutationRoutedWholeTreeInputsV1:
        raise TypeError("Exact original observation-only bank required")
    parent,samples = _cpu_index_array(bank.parent),_cpu_index_array(bank.sample_index)
    _require(samples.ndim == 2 and samples.shape[1] == 952 and bank.query_column.shape == (686,),
             "Original full952/all686 query bank required")
    arrays,meta = build_clade_transport_arrays_v1(parent,samples,max_payload_bytes=max_payload_bytes)
    meta["query_column_sha256"] = _sha(_cpu_index_array(bank.query_column))
    return AnonymousExactCladeTransportV1.from_verified_arrays(arrays,meta),meta


def validate_transport_v1(transport, upward):
    if not isinstance(transport,AnonymousExactCladeTransportV1):
        raise TypeError("Verified explicit exact-clade payload required")
    validate_clade_flags_v1(transport.flags,upward)
    P = transport.weight.numel()
    _require(transport.node_size.dtype == torch.int32
             and transport.node_size.shape == (transport.native_nodes,),
             "Compact full node sizes required")
    for name in ("node_size",*EVENT_FIELDS):
        value = getattr(transport,name)
        dtype = torch.float32 if name == "weight" else torch.int32
        _require(value.dtype == dtype and value.device == upward.device and not value.requires_grad
                 and (name == "node_size" or value.shape == (P,)),
                 "Read-only same-device exact transport vectors required")


class RankEightExactCladeTransportV1(RankEightCladeFlagAdapterV1):
    """Exactly the H042 four bias-free projections; shared local + cross branch."""
    def __init__(self, *, event_block_size=32768, checkpoint_event_blocks=True, _projections=None):
        if _projections is None:
            super().__init__()
        else:
            _require(type(_projections) is RankEightCladeFlagAdapterV1,
                     "Reuse exactly the already initialized H042 projections")
            nn.Module.__init__(self)
            self.U,self.V,self.W,self.O = (_projections.U,_projections.V,_projections.W,_projections.O)
        _require(type(event_block_size) is int and event_block_size > 0
                 and type(checkpoint_event_blocks) is bool, "Bounded event block/checkpoint policy")
        self.event_block_size = event_block_size
        self.checkpoint_event_blocks = checkpoint_event_blocks

    @staticmethod
    def local_rank_v1(pu,pv,pw,node_flags):
        channels = node_flags.to(dtype=pu.dtype)
        return channels[:,0:1]*pu*pv-channels[:,1:2]*pu*pw

    @staticmethod
    def _event_block(pu,pv,pw,node_size,old_node,new_node,support,intersection,weight,n,arm):
        old_present,new_present = old_node >= 0,new_node >= 0
        a,b = old_node.clamp_min(0).long(),new_node.clamp_min(0).long()
        def state(value,index,present):
            return value[index].masked_fill(~present[:,None],0.)
        ua,ub = state(pu,a,old_present),state(pu,b,new_present)
        if arm == "true_partner":
            old_partner_index,new_partner_index = b,a
        elif arm == "own_state":
            old_partner_index,new_partner_index = a,b
        else:
            raise ValueError("Prospectively fixed true_partner or own_state arm required")
        both_present = old_present & new_present
        old_partner = state(pv,old_partner_index,both_present)
        new_partner = state(pw,new_partner_index,both_present)
        sa = node_size[a].to(pu.dtype).masked_fill(~old_present,0.)
        sb = node_size[b].to(pu.dtype).masked_fill(~new_present,0.)
        inter,k = intersection.to(pu.dtype),support.to(pu.dtype)
        scale = math.log1p(n)
        one = torch.ones_like(k)
        geometry = torch.stack((one,sa.log1p()/scale,sb.log1p()/scale,inter/n,
            (sa+sb-2*inter)/n,k/n,(~old_present).to(pu.dtype),(~new_present).to(pu.dtype)),-1)
        reversed_geometry = geometry[:,[0,2,1,3,4,5,7,6]]
        old = F.gelu(ua+old_partner+geometry)
        new = F.gelu(ub+new_partner+reversed_geometry)
        old = old*weight[:,None]*old_present[:,None].to(pu.dtype)
        new = new*weight[:,None]*new_present[:,None].to(pu.dtype)
        return old,new

    def rank_residual(self, upward, transport, *, arm="true_partner"):
        validate_transport_v1(transport,upward)
        _require(arm in ARMS and all(p.dtype == torch.float32 and p.device == upward.device
                                   for p in self.parameters()), "Fixed arm and FP32 same-device weights")
        pu,pv,pw = self.U(upward),self.V(upward),self.W(upward)
        local = self.local_rank_v1(pu,pv,pw,transport.flags.node_flags)
        cross = upward.new_zeros(transport.native_nodes,8)
        for start in range(0,transport.weight.numel(),self.event_block_size):
            part = slice(start,min(start+self.event_block_size,transport.weight.numel()))
            a,b = transport.old_node[part],transport.new_node[part]
            args = (pu,pv,pw,transport.node_size,a,b,transport.leaf_support[part],
                    transport.intersection_count[part],transport.weight[part],transport.haplotypes,arm)
            if self.checkpoint_event_blocks and self.training and torch.is_grad_enabled():
                old,new = checkpoint(self._event_block,*args,use_reentrant=False)
            else:
                old,new = self._event_block(*args)
            cross.index_add_(0,a.clamp_min(0).long(),-old)
            cross.index_add_(0,b.clamp_min(0).long(),new)
        return local+cross

    def forward(self, upward, transport, *, arm="true_partner"):
        return self.O(self.rank_residual(upward,transport,arm=arm))


class AnonymousExactCladeTransportGNNV1(AnonymousCladeFlagControlGNNV1):
    """Inherited Torch/PyG H042 body/readout, no full training or target API."""
    def __init__(self, hidden_dim=32, *, expected_haplotypes=952,site_chunk_size=256,
                 checkpoint_sites=True,arm="true_partner",event_block_size=32768,
                 checkpoint_event_blocks=True):
        _require(arm in ARMS,"Fixed matched transport arm required")
        super().__init__(hidden_dim=hidden_dim,expected_haplotypes=expected_haplotypes,
                         site_chunk_size=site_chunk_size,checkpoint_sites=checkpoint_sites)
        self.arm = arm
        self.clade_coupling = RankEightExactCladeTransportV1(
            event_block_size=event_block_size,checkpoint_event_blocks=checkpoint_event_blocks,
            _projections=self.clade_coupling)

    def encode_trees(self, bank, *, flags, control="aligned"):
        transport = flags  # Internal inherited readout name; public API is explicit.
        base,ug,dg,degree = self.prepare_tree_base_v1(bank,control=control)
        upward,_outside = self.sweep_tree_state_v1(bank,base,ug,dg,degree)
        injection = self.clade_coupling(upward,transport,arm=self.arm)
        second_upward,second_outside = self.sweep_tree_state_v1(bank,base+injection,ug,dg,degree)
        return self.leaf_context_v1(bank,second_upward,second_outside)

    def forward_holds(self, bank, *, transport, control="aligned"):
        return super().forward_holds(bank,flags=transport,control=control)

    def forward_train_holds(self, bank, *, transport, control="aligned"):
        return self.forward_holds(bank,transport=transport,control=control).index_select(1,bank.query_column)

    def forward(self, bank, *, transport, control="aligned"):
        return self.forward_train_holds(bank,transport=transport,control=control).index_select(0,bank.readout_hold)


def load_h038_transport_warmstart_v1(model, payload):
    if not isinstance(model,AnonymousExactCladeTransportGNNV1):
        raise TypeError("Exact H043 source model required")
    proof = load_h038_flag_warmstart_v1(model,payload)
    return dict(proof,arm=model.arm,operator=SCHEMA,all_H042_flags_retained=True,
                only_foreign_vs_own_state_gather_changes_between_arms=True)
