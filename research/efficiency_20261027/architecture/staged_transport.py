"""Explicit stage VJPs for the unchanged Transport research candidate.

Forward returns independent focal logits. Call loss.backward(), then
backward_focals(logits.grad) before any optimizer update. Parameter gradients
accumulate normally. No global checkpoint/copy DAG crosses compute blocks.
Optional CPU first-node caching eliminates duplicate first-tree evaluations.
No data reader, optimizer, target resolver, admission or launcher is included.
"""
import math
import torch
from .bounded_transport import BoundedTransport, need
from .treeloom_transport import affine_scan


class StagedTransport(BoundedTransport):
    def __init__(self, model, bank, node_flags, native_nodes_per_tree, *,
                 holds_per_block=128, haplotypes_per_tile=128,
                 scan_holds_per_block=1024, cache_first_nodes=False,
                 resource_check=lambda stage: None):
        super().__init__(model, bank, node_flags, native_nodes_per_tree,
                         holds_per_block=holds_per_block,
                         haplotypes_per_tile=haplotypes_per_tile,
                         scan_holds_per_block=scan_holds_per_block,
                         resource_check=resource_check)
        self.cache_first_nodes = bool(cache_first_nodes)
        self.state = None
        self.chunks = [(a, min(a + self.scan_block, len(bank.spans)))
                       for a in range(0, len(bank.spans), self.scan_block)]

    def _decay(self, bank):
        raw = bank.branch_support_true
        support = torch.cat((raw[:, :3], raw[:, 3:4].log1p() / math.log(100001.),
                             raw[:, 4:5]), dim=-1)
        features = torch.cat((bank.edge_geometry, support), dim=-1)
        duration = torch.expm1(bank.edge_geometry[:, 2] * math.log(10001.))
        return torch.exp(-duration[:, None] * torch.exp(
            self.model.log_tree_rate + self.model.edge_rate(features).tanh()))

    def _scan_device(self, layer, a, b, reverse, leaf, carry):
        geometry = self.geometry[a:b].to(self.device)
        features = torch.cat((leaf, geometry[:, None].expand(-1, leaf.shape[1], -1)), -1)
        exponent = -self.bank.spans[a:b, None, None].to(self.device) * torch.exp(
            layer.log_rate + layer.rate(features).tanh())
        alpha = torch.exp(exponent)
        beta = -torch.expm1(exponent) * layer.force(features).tanh()
        if reverse:
            alpha, beta = alpha.flip(0), beta.flip(0)
        value = affine_scan(alpha, beta, initial=carry)
        return value.flip(0) if reverse else value

    @torch.no_grad()
    def _direction(self, layer, context, reverse):
        output = torch.empty_like(context)
        carry = torch.zeros(context.shape[1:], dtype=context.dtype, device=self.device)
        incoming = [None] * len(self.chunks)
        order = range(len(self.chunks) - 1, -1, -1) if reverse else range(len(self.chunks))
        for i in order:
            a, b = self.chunks[i]
            incoming[i] = carry.cpu().clone()
            value = self._scan_device(layer, a, b, reverse, context[a:b].to(self.device), carry)
            output[a:b].copy_(value.cpu())
            carry = value[0 if reverse else -1].clone()
        return output, incoming

    def _merge_device(self, layer, leaf, left, right):
        return layer.leaf_norm(leaf + layer.merge(torch.cat((leaf, left, right), -1)))

    @torch.no_grad()
    def _history_forward(self, layer, context):
        left, _ = self._direction(layer, context, False)
        right, _ = self._direction(layer, context, True)
        output = torch.empty_like(context)
        for a, b in self.chunks:
            value = self._merge_device(layer, context[a:b].to(self.device),
                                       left[a:b].to(self.device), right[a:b].to(self.device))
            output[a:b].copy_(value.cpu())
        return output

    def _direction_vjp(self, layer, context, output_grad, incoming, reverse, input_grad):
        # Reverse the block traversal of the matching forward direction.
        order = range(len(self.chunks)) if reverse else range(len(self.chunks) - 1, -1, -1)
        carry_grad = torch.zeros(context.shape[1:], dtype=context.dtype, device=self.device)
        for i in order:
            a, b = self.chunks[i]
            leaf = context[a:b].to(self.device).detach().requires_grad_(True)
            carry = incoming[i].to(self.device).detach().requires_grad_(True)
            value = self._scan_device(layer, a, b, reverse, leaf, carry)
            vjp = output_grad[a:b].to(self.device).clone()
            vjp[0 if reverse else -1].add_(carry_grad)
            torch.autograd.backward(value, vjp)
            input_grad[a:b].add_(leaf.grad.cpu())
            carry_grad = carry.grad.detach().clone()
            del leaf, carry, value, vjp
        # Initial boundary is fixed zero, so it has no upstream predecessor.

    def _history_vjp(self, layer, context, output_grad):
        with torch.no_grad():
            left, left_in = self._direction(layer, context, False)
            right, right_in = self._direction(layer, context, True)
        input_grad = torch.empty_like(context)
        for a, b in self.chunks:
            leaf = context[a:b].to(self.device).detach().requires_grad_(True)
            lv = left[a:b].to(self.device).detach().requires_grad_(True)
            rv = right[a:b].to(self.device).detach().requires_grad_(True)
            value = self._merge_device(layer, leaf, lv, rv)
            torch.autograd.backward(value, output_grad[a:b].to(self.device))
            input_grad[a:b].copy_(leaf.grad.cpu())
            # Incoming boundaries were copied before directional states are
            # overwritten by their adjoints, keeping host scratch tile bounded.
            left[a:b].copy_(lv.grad.cpu())
            right[a:b].copy_(rv.grad.cpu())
            del leaf, lv, rv, value
        self._direction_vjp(layer, context, left, left_in, False, input_grad)
        self._direction_vjp(layer, context, right, right_in, True, input_grad)
        return input_grad

    def _cached_second(self, z, first_nodes, history):
        bank, plan = self._block(z)
        decay = self._decay(bank)
        leaf = first_nodes[bank.sample_index]
        nodes = first_nodes.index_add(0, bank.sample_index.reshape(-1),
                                     (history - leaf).reshape(-1, self.model.hidden_dim))
        nodes = self._tree(self.model.rounds[1], nodes, decay, plan)
        return nodes[bank.sample_index]

    @torch.no_grad()
    def forward_focals(self, focal_hold, focal_sample_column):
        need(self.state is None, 'Consume or release the preceding staged forward first')
        need(focal_hold.device.type == focal_sample_column.device.type == 'cpu'
             and focal_hold.dtype == focal_sample_column.dtype == torch.long
             and focal_hold.shape == focal_sample_column.shape and focal_hold.ndim == 1
             and len(focal_hold) > 0, 'Nonempty original CPU focal coordinates required')
        t, h = self.bank.sample_index.shape
        need(bool(((focal_hold >= 0) & (focal_hold < t)).all())
             and bool(((focal_sample_column >= 0) & (focal_sample_column < h)).all()),
             'Every focal must lie in the supplied observation bank')
        first = torch.empty((t, h, self.model.hidden_dim), dtype=torch.float32)
        nodes0 = torch.empty((len(self.bank.parent), self.model.hidden_dim)) if self.cache_first_nodes else None
        self.tick('staged_transport_first_tree_begin')
        for z in self.blocks:
            a, b, lo, hi = z
            if nodes0 is None:
                first[a:b].copy_(self._first(z))
            else:
                bank, plan, nodes, decay = self._initial(z)
                nodes = self._tree(self.model.rounds[0], nodes, decay, plan)
                first[a:b].copy_(nodes[bank.sample_index].cpu())
                nodes0[lo:hi].copy_(nodes.cpu())
                del bank, plan, nodes, decay
        history0 = torch.empty_like(first)
        self.tick('staged_transport_first_history_begin')
        for a in range(0, h, self.hap_tile):
            history0[:, a:a+self.hap_tile].copy_(self._history_forward(
                self.model.rounds[0], first[:, a:a+self.hap_tile]))
        second = torch.empty_like(first)
        self.tick('staged_transport_second_tree_begin')
        for z in self.blocks:
            a, b, lo, hi = z
            if nodes0 is None:
                second[a:b].copy_(self._second(z, history0[a:b]))
            else:
                second[a:b].copy_(self._cached_second(
                    z, nodes0[lo:hi].to(self.device), history0[a:b].to(self.device)).cpu())
        logits = torch.empty((len(focal_hold), 3), device=self.device)
        self.tick('staged_transport_final_history_begin')
        for a in range(0, h, self.hap_tile):
            b = min(a + self.hap_tile, h)
            rows = ((focal_sample_column >= a) & (focal_sample_column < b)).nonzero(as_tuple=True)[0]
            if not len(rows):
                continue
            history = self._history_forward(self.model.rounds[1], second[:, a:b])
            hold, col = focal_hold[rows], focal_sample_column[rows] - a
            value = self.model.classifier(torch.cat((second[:, a:b][hold, col], history[hold, col]), -1).to(self.device))
            logits.index_copy_(0, rows.to(self.device), value)
            del history, value
        self.state = dict(first=first, history0=history0, second=second, nodes0=nodes0,
                          hold=focal_hold.clone(), column=focal_sample_column.clone(),
                          versions=tuple(p._version for p in self.model.parameters()))
        self.tick('staged_transport_forward_complete')
        return logits.detach().requires_grad_(True)

    def backward_focals(self, logit_vjp):
        need(self.state is not None, 'A staged forward must precede explicit backward')
        state = self.state
        need(tuple(p._version for p in self.model.parameters()) == state['versions'],
             'No weight update may occur between staged forward and backward')
        hold, column = state['hold'], state['column']
        need(logit_vjp is not None and logit_vjp.shape == (len(hold), 3),
             'Supply the complete focal-logit cotangent')
        first, history0, second = state['first'], state['history0'], state['second']
        nodes0 = state['nodes0']
        second_grad = torch.zeros_like(second)
        self.tick('staged_transport_final_history_vjp_begin')
        with torch.enable_grad():
            for a in range(0, 1600, self.hap_tile):
                b = min(a + self.hap_tile, 1600)
                rows = ((column >= a) & (column < b)).nonzero(as_tuple=True)[0]
                if not len(rows):
                    continue
                context = second[:, a:b]
                history = self._history_forward(self.model.rounds[1], context)
                hh, cc = hold[rows], column[rows] - a
                tree_focal = context[hh, cc].to(self.device).detach().requires_grad_(True)
                history_focal = history[hh, cc].to(self.device).detach().requires_grad_(True)
                value = self.model.classifier(torch.cat((tree_focal, history_focal), -1))
                torch.autograd.backward(value, logit_vjp[rows.to(logit_vjp.device)].to(self.device))
                history_grad = torch.zeros_like(context)
                history_grad.index_put_((hh, cc), history_focal.grad.cpu(), accumulate=True)
                direct = torch.zeros_like(context)
                direct.index_put_((hh, cc), tree_focal.grad.cpu(), accumulate=True)
                del history, tree_focal, history_focal, value
                indirect = self._history_vjp(self.model.rounds[1], context, history_grad)
                second_grad[:, a:b].copy_(direct + indirect)
                del context, history_grad, direct, indirect
            del second
            state.pop('second')
            history0_grad = torch.empty_like(history0)
            node0_grad = torch.empty_like(nodes0) if nodes0 is not None else None
            self.tick('staged_transport_second_tree_vjp_begin')
            for z in self.blocks:
                a, b, lo, hi = z
                history = history0[a:b].to(self.device).detach().requires_grad_(True)
                if nodes0 is None:
                    value = self._second(z, history)
                    torch.autograd.backward(value, second_grad[a:b])
                else:
                    nodes = nodes0[lo:hi].to(self.device).detach().requires_grad_(True)
                    value = self._cached_second(z, nodes, history)
                    torch.autograd.backward(value, second_grad[a:b].to(self.device))
                    node0_grad[lo:hi].copy_(nodes.grad.cpu())
                    del nodes
                history0_grad[a:b].copy_(history.grad.cpu())
                del history, value
            del second_grad, history0, nodes0
            state.pop('history0'); state.pop('nodes0')
            first_grad = torch.empty_like(first)
            self.tick('staged_transport_first_history_vjp_begin')
            for a in range(0, 1600, self.hap_tile):
                first_grad[:, a:a+self.hap_tile].copy_(self._history_vjp(
                    self.model.rounds[0], first[:, a:a+self.hap_tile],
                    history0_grad[:, a:a+self.hap_tile]))
            del history0_grad, first
            state.pop('first')
            self.tick('staged_transport_first_tree_vjp_begin')
            for z in self.blocks:
                a, b, lo, hi = z
                if node0_grad is None:
                    value = self._first(z)
                    torch.autograd.backward(value, first_grad[a:b])
                else:
                    bank, plan, nodes, decay = self._initial(z)
                    value = self._tree(self.model.rounds[0], nodes, decay, plan)
                    vjp = node0_grad[lo:hi].to(self.device).clone()
                    vjp.index_add_(0, bank.sample_index.reshape(-1),
                                   first_grad[a:b].reshape(-1, self.model.hidden_dim).to(self.device))
                    torch.autograd.backward(value, vjp)
                    del bank, plan, nodes, decay, vjp
                del value
        self.state = None
        self.tick('staged_transport_backward_complete')

    def release(self):
        self.state = None

    def named_host_array_accounting(self):
        t, h = self.bank.sample_index.shape
        d = self.model.hidden_dim
        history = t * h * d * 4
        tile = t * min(h, self.hap_tile) * d * 4
        node = len(self.bank.parent) * d * 4
        return dict(history_bytes=history, tile_bytes=tile,
                    full_history_and_adjoint_bound_bytes=4 * history,
                    optional_node_cache_and_adjoint_bytes=2 * node if self.cache_first_nodes else 0,
                    conservative_tile_scratch_bytes=6 * tile,
                    excludes='Observation bank, original episodes/flags, jump indexes, framework allocator and device workspace; not an RSS guarantee')
