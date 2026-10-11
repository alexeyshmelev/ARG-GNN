"""H032 static whole-tree reference gates with an observed donor-support Gram.

Tree OU covariance, positive rerooting and second-order neural statistics are
established. This candidate coupling has unresolved usefulness and priority.
No carrier recurrence, ancestry posterior or causal donor claim is made.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch
from torch import Tensor, nn

from .persistence_trace_kernel_v1 import _tree
from .reference_logcavity_gnn_v1 import (
    ReferenceLogCavityGNNV1, ReferenceLogCavityBankV1,
    prepare_reference_logcavity_bank_v1, masked_logaddexp_v1,
    unit_class_fields_v1,
)

ARMS = ("independence_gram", "marginal_outer_gram", "tree_gram")
SOURCE_CONTROLS = ("first_moment_only",) + ARMS


@dataclass(frozen=True)
class ReferenceDependenceBankV1:
    reference: ReferenceLogCavityBankV1
    log_self_source: Tensor       # sum individual omega^2, NOT squared class sum
    upward_components: tuple     # per level: (children, parent row) at child rank
    metadata: dict

    @property
    def forest(self):
        return self.reference.forest

    def to(self, device):
        return replace(self, reference=self.reference.to(device),
            log_self_source=self.log_self_source.to(device),
            upward_components=tuple(tuple((c.to(device), r.to(device)) for c,r in level)
                                    for level in self.upward_components))


def prepare_reference_dependence_bank_v1(trees, intervals_bp, readout_x_bp,
        query_ids, reference_ids, reference_classes, *, census=350.,
        sample_identity_namespace, **geometry):
    """Unchanged H030 p geometry; extra independent-donor self-weight sources."""
    bank = prepare_reference_logcavity_bank_v1(trees, intervals_bp, readout_x_bp,
        query_ids, reference_ids, reference_classes, census=census,
        sample_identity_namespace=sample_identity_namespace, **geometry)
    n = len(bank.forest.parent)
    log_self = torch.full_like(bank.log_source, -torch.inf)
    counts = (bank.metadata["younger_class_counts"], bank.metadata["older_class_counts"])
    offset = 0
    for tree in trees:
        samples, _, _ = _tree(tree)
        p, t = tree.parent.tolist(), tree.node_times.tolist()
        for child, ancestor in enumerate(p.copy()):
            if ancestor >= 0 and t[child] < census < t[ancestor]:
                point = len(p); p.append(ancestor); t.append(census); p[child] = point
        for identity, cls in zip(reference_ids.tolist(), reference_classes.tolist()):
            node = samples[identity]
            family = int(t[node] > census)
            if family == 0:
                while t[node] < census:
                    node = p[node]
                    if node < 0:
                        raise ValueError("Younger reference requires actual census crossing")
                if t[node] != census:
                    raise ValueError("Analytic census source placement differs from H030")
            log_weight2 = torch.tensor(-2 * math.log(counts[family][cls]), dtype=torch.float32)
            column = 4 * family + cls
            log_self[offset+node,column] = torch.logaddexp(log_self[offset+node,column], log_weight2)
        offset += len(p)
    if offset != n:
        raise ValueError("Independent self-source rows differ from H030 forest")
    children = [[] for _ in range(n)]
    for child, parent in enumerate(bank.forest.parent.tolist()):
        if parent >= 0:
            children[parent].append(child)
    components = []
    for nodes in bank.forest.upward_levels:
        rows = nodes.tolist()
        ranks = []
        for rank in range(max((len(children[node]) for node in rows), default=0)):
            pairs = [(children[node][rank], row) for row,node in enumerate(rows)
                     if len(children[node]) > rank]
            ranks.append((torch.tensor([c for c,_ in pairs], dtype=torch.long),
                          torch.tensor([r for _,r in pairs], dtype=torch.long)))
        components.append(tuple(ranks))
    metadata = dict(bank.metadata, kind="reference_dependence_static_whole_native_forest",
        donor_weight="exact_H030_omega_1_per_family_class_count",
        Gram="query_conditioned_weighted_tree_OU_support_not_posterior_covariance",
        independent_local_Gram="sum_individual_omega_squared_diagonal",
        tree_local_Gram="coincident_class_weight_outer_product_including_distinct_donors",
        moment_normalization="detached_relative_M_H_G_gauges_FP32",
        recurrent_paths_executed=False, alignment_sham="omitted_unqualified_dense_cost",
        fitted_arms=list(ARMS), historical_first_moment="frozen_selected_H030_static_no_refit")
    return ReferenceDependenceBankV1(bank, log_self, tuple(components), metadata)


# Each component stores (q, log(M)-q, log(H)-q, log(G)-2q).
# q is a detached numerical gauge, never a learned value or lost gradient route.
def empty_moment_v1(rows, reference):
    return (reference.new_zeros(rows,2),
            reference.new_full((rows,2,4),-torch.inf),
            reference.new_full((rows,2,4),-torch.inf),
            reference.new_full((rows,2,4,4),-torch.inf))


def take_moment_v1(moment, rows):
    return tuple(value[rows] for value in moment)


def put_moment_v1(moment, rows, values):
    for target, value in zip(moment, values):
        target.index_copy_(0, rows, value)


def transport_moment_v1(moment, cost):
    q,m,h,g = moment
    active = torch.isfinite(m).any(-1)
    # Compensation retains cost VJPs despite detaching the transported gauge.
    correction = cost.detach() - cost
    new_q = torch.where(active, q-cost.detach(), torch.zeros_like(q))
    return (new_q, m+correction[...,None],
            h-cost[...,None]+correction[...,None],
            g+2*correction[...,None,None])


def merge_moment_v1(a, b, *, independent=False):
    qa,ma,ha,ga = a; qb,mb,hb,gb = b
    available_a = torch.isfinite(ma).any(-1)
    available_b = torch.isfinite(mb).any(-1)
    masked_a = torch.where(available_a,qa,torch.full_like(qa,-torch.inf))
    masked_b = torch.where(available_b,qb,torch.full_like(qb,-torch.inf))
    q = torch.maximum(masked_a,masked_b)
    q = torch.where(available_a|available_b,q,torch.zeros_like(q)).detach()
    da = torch.where(available_a,qa-q,torch.zeros_like(q))
    db = torch.where(available_b,qb-q,torch.zeros_like(q))
    ma,ha,ga = ma+da[...,None],ha+da[...,None],ga+2*da[...,None,None]
    mb,hb,gb = mb+db[...,None],hb+db[...,None],gb+2*db[...,None,None]
    g = masked_logaddexp_v1(ga,gb)
    if not independent:
        cross = masked_logaddexp_v1(ha[..., :,None]+hb[...,None,:],
                                   hb[..., :,None]+ha[...,None,:])
        g = masked_logaddexp_v1(g,cross)
    return q,masked_logaddexp_v1(ma,mb),masked_logaddexp_v1(ha,hb),g


def normalize_moment_v1(moment):
    """Unit first moment and C with no absolute large-log subtraction."""
    _,m,_,g = moment
    active = torch.isfinite(m).any(-1)
    safe = torch.where(active[...,None],m,torch.zeros_like(m))
    log_z = torch.logsumexp(safe,-1)
    p = torch.where(torch.isfinite(m),(m-log_z[...,None]).exp(),torch.zeros_like(m))
    value = (g-2*log_z[...,None,None]).exp()
    c = torch.where(active[...,None,None],value,torch.zeros_like(value))
    return p,c,active


class ReferenceDependenceGNNV1(ReferenceLogCavityGNNV1):
    """H030 static value route; Gram affects four gates and no standalone value."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gram_up_gate = nn.Linear(20,self.hidden_dim,bias=False)
        self.gram_down_gate = nn.Linear(20,self.hidden_dim,bias=False)
        self.gram_decoder_up_gate = nn.Linear(20,self.hidden_dim,bias=False)
        self.gram_decoder_down_gate = nn.Linear(20,self.hidden_dim,bias=False)
        for layer in (self.gram_up_gate,self.gram_down_gate,
                      self.gram_decoder_up_gate,self.gram_decoder_down_gate):
            nn.init.zeros_(layer.weight)
        self.register_buffer("triangle",torch.triu_indices(4,4),persistent=False)

    def local_moment(self, bank, *, independent=False):
        source = bank.reference.log_source.reshape(-1,2,4)
        active = torch.isfinite(source).any(-1)
        q = source.detach().max(-1).values
        q = torch.where(active,q,torch.zeros_like(q))
        m = source-q[...,None]
        if independent:
            g = source.new_full((len(source),2,4,4),-torch.inf)
            diagonal = bank.log_self_source.reshape(-1,2,4)-2*q[...,None]
            for cls in range(4):
                g[:,:,cls,cls] = diagonal[:,:,cls]
        else:
            g = m[..., :,None]+m[...,None,:]
        return q,m,m,g

    def moment_cavities(self, bank, *, independent=False):
        base = bank.reference; forest = base.forest
        cost = (base.edge_basis*torch.nn.functional.softplus(self.raw_cost)[None]).sum(-1)
        if not bool(torch.isfinite(cost).all()):
            raise ValueError("Nonfinite shared positive age costs")
        n = len(forest.parent)
        local = self.local_moment(bank,independent=independent)
        up = empty_moment_v1(n,base.log_source)
        for nodes,components in zip(forest.upward_levels,bank.upward_components):
            value = take_moment_v1(local,nodes)
            for children,rows in components:
                incoming = transport_moment_v1(take_moment_v1(up,children),cost[children])
                merged = merge_moment_v1(take_moment_v1(value,rows),incoming,independent=independent)
                put_moment_v1(value,rows,merged)
            put_moment_v1(up,nodes,value)
        incoming = transport_moment_v1(up,cost)
        prefix = empty_moment_v1(n,base.log_source)
        accumulator = empty_moment_v1(n,base.log_source)
        for children in base.children_by_rank:
            parents = forest.parent[children]
            old = take_moment_v1(accumulator,parents)
            put_moment_v1(prefix,children,old)
            merged = merge_moment_v1(old,take_moment_v1(incoming,children),independent=independent)
            put_moment_v1(accumulator,parents,merged)
        siblings = empty_moment_v1(n,base.log_source)
        accumulator = empty_moment_v1(n,base.log_source)
        for children in reversed(base.children_by_rank):
            parents = forest.parent[children]
            old = take_moment_v1(accumulator,parents)
            merged = merge_moment_v1(take_moment_v1(prefix,children),old,independent=independent)
            put_moment_v1(siblings,children,merged)
            merged = merge_moment_v1(old,take_moment_v1(incoming,children),independent=independent)
            put_moment_v1(accumulator,parents,merged)
        outside = empty_moment_v1(n,base.log_source)
        for nodes in forest.downward_levels:
            parents = forest.parent[nodes]
            value = merge_moment_v1(take_moment_v1(outside,parents),
                                    take_moment_v1(local,parents),independent=independent)
            value = merge_moment_v1(value,take_moment_v1(siblings,nodes),independent=independent)
            put_moment_v1(outside,nodes,transport_moment_v1(value,cost[nodes]))
        return up,outside

    def gram_channels(self, c):
        return c[...,self.triangle[0],self.triangle[1]].reshape(-1,20)

    def grams(self, bank, log_up, log_out, control):
        if control == "first_moment_only":
            zero = log_up.new_zeros(len(log_up),20)
            return zero,zero,zero
        if control == "marginal_outer_gram":
            def outer(log):
                p,_ = unit_class_fields_v1(log)
                p = p.reshape(-1,2,4)
                return self.gram_channels(p[..., :,None]*p[...,None,:])
            return outer(log_up),outer(log_out),outer(masked_logaddexp_v1(log_up,log_out))
        independent = control == "independence_gram"
        up,out = self.moment_cavities(bank,independent=independent)
        full = merge_moment_v1(up,out,independent=independent)
        return tuple(self.gram_channels(normalize_moment_v1(value)[1]) for value in (up,out,full))

    def encode_dependence(self, bank, log_up, log_out, gram_up, gram_out):
        forest = bank.forest; reference = bank.reference
        up_input,_,_ = self.unit_contrasts(reference,log_up)
        out_input,_,_ = self.unit_contrasts(reference,log_out)
        local_input,_,_ = self.unit_contrasts(reference,reference.log_source)
        base = self.reference_projection(up_input); local = self.reference_projection(local_input)
        n,d = len(forest.parent),self.hidden_dim
        up = base.new_zeros(n,d); message_sum = base.new_zeros(n,d); messages = base.new_zeros(n,d)
        ug = (self.up_gate(forest.geometry)+self.gram_up_gate(gram_up)).sigmoid()
        dg = (self.down_gate(forest.geometry)+self.gram_down_gate(gram_out)).sigmoid()
        for nodes,children,ranks in zip(forest.upward_levels,forest.upward_children,forest.upward_ranks):
            acc = base.new_zeros(len(nodes),d)
            if children.numel():
                value = self.up_value(up[children])*ug[children]
                value = value/forest.child_count[forest.parent[children]][:,None]
                messages.index_copy_(0,children,value); acc.index_add_(0,ranks,value)
            message_sum.index_copy_(0,nodes,acc)
            up.index_copy_(0,nodes,(base[nodes]+acc).tanh())
        outside = base.new_zeros(n,d)
        for nodes in forest.downward_levels:
            parents = forest.parent[nodes]
            cavity = outside[parents]+local[parents]+message_sum[parents]-messages[nodes]
            value = self.reference_projection(out_input[nodes])+self.down_value(cavity)*dg[nodes]
            outside.index_copy_(0,nodes,value.tanh())
        return up,outside

    def decode_dependence(self, bank, up, outside, gram):
        n,d = len(up),self.hidden_dim
        zero = up.new_zeros(n,d)
        base = torch.stack((self.decoder_base(up+outside),zero,zero),1)
        conditions = torch.cat((up,outside,bank.geometry),-1)
        ug = (self.decoder_up_gate(conditions)+self.gram_decoder_up_gate(gram)).sigmoid()[:,None]
        dg = (self.decoder_down_gate(conditions)+self.gram_decoder_down_gate(gram)).sigmoid()[:,None]
        node_value = up.new_zeros(n,3,d); total = up.new_zeros(n,3,d); messages = up.new_zeros(n,3,d)
        for nodes,children,ranks in zip(bank.upward_levels,bank.upward_children,bank.upward_ranks):
            acc = up.new_zeros(len(nodes),3,d)
            if children.numel():
                value = self.decoder_up(node_value[children])*ug[children]
                value = value/bank.child_count[bank.parent[children]][:,None,None]
                messages.index_copy_(0,children,value); acc.index_add_(0,ranks,value)
            value = base[nodes]+acc
            total.index_copy_(0,nodes,value); node_value.index_copy_(0,nodes,value)
        down = up.new_zeros(n,3,d)
        for nodes in bank.downward_levels:
            parents = bank.parent[nodes]
            cavity = down[parents]+total[parents]-messages[nodes]
            down.index_copy_(0,nodes,self.decoder_down(cavity)*dg[nodes])
        return node_value+down

    def parameter_counts(self, control):
        if control not in SOURCE_CONTROLS:
            raise ValueError("Unknown H032 condition")
        counts = super().parameter_counts("static")
        zero_gram = 20 if control == "first_moment_only" else (13 if control == "independence_gram" else 4)
        counts.update(Gram_projection_parameters=1280,
            known_zero_Gram_input_channels_of20=zero_gram,
            known_zero_Gram_projection_weights=zero_gram*4*self.hidden_dim,
            nominal_active_after_known_zero_Gram_columns=counts["active"]-zero_gram*4*self.hidden_dim,
            recurrent_paths_executed=False,
            input_rank_note="p contrast rank at most5; symmetric Gram slots at most16, independence7; realized rank unmeasured")
        return counts

    def forward(self, bank, frozen_baseline_logits, *, control):
        if control not in SOURCE_CONTROLS:
            raise ValueError("Unknown H032 static Gram condition")
        base = bank.reference; forest = bank.forest
        if (base.log_source.dtype != torch.float32 or frozen_baseline_logits.dtype != torch.float32
                or frozen_baseline_logits.shape != (len(forest.readout_hold),forest.query_nodes.shape[1],4)
                or frozen_baseline_logits.device != self.raw_cost.device
                or base.log_source.device != self.raw_cost.device
                or not bool(torch.isfinite(frozen_baseline_logits).all())):
            raise ValueError("Original finite FP32 H025 support and source bank required")
        # This immutable H030 first-moment calculation preserves p bit-for-bit.
        log_up,log_out = self.log_cavities(base)
        gu,go,gf = self.grams(bank,log_up,log_out,control)
        if control == "first_moment_only":
            up,outside = self.encode_units(base,log_up,log_out)
            empty = up.new_zeros(len(up),self.rank)
            decoded = self.decode(forest,up,outside,empty,empty)
        else:
            up,outside = self.encode_dependence(bank,log_up,log_out,gu,go)
            decoded = self.decode_dependence(forest,up,outside,gf)
        context = decoded[forest.query_nodes[forest.readout_hold]].flatten(2)
        centered = context.clone()
        for group in forest.age_groups:
            centered[:,group] = context[:,group]-context[:,group].mean(1,keepdim=True)
        result = frozen_baseline_logits.detach()+self.context_head(centered)
        if not bool(torch.isfinite(result).all()):
            raise ValueError("Nonfinite static donor-support residual")
        self.last_diagnostics = dict(positive_cost_coefficients=torch.nn.functional.softplus(
            self.raw_cost).detach().cpu().tolist(),centered_context_l2=float(centered.detach().norm()),
            Gram_full_l2=float(gf.detach().norm()),recurrent_paths_executed=False,condition=control)
        return result
