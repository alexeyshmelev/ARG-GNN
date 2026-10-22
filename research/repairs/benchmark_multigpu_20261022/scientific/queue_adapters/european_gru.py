"""Bounded European bidirectional GRU calls with complete connected BPTT.

Unidirectional execution views share the original parameter objects. They are
not registered as new model children and do not alter saved state keys. Chunk
boundaries do not detach recurrent states. FP32 reduction order may differ.
"""
from types import MethodType
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def chunked_bidirectional_forward(self, sequence, hx=None):
    if not torch.is_tensor(sequence) or sequence.ndim != 3 or len(sequence) == 0:
        raise ValueError("A complete nonempty time by haplotype sequence is required")
    if sequence.shape[-1] != self.input_size:
        raise ValueError("Original recurrent input width is required")
    if hx is not None and hx.shape != (2, sequence.shape[1], self.hidden_size):
        raise ValueError("Original forward and reverse hidden states are required")
    outputs, final = [], []
    for direction, module in enumerate((self._recovery_forward_view, self._recovery_reverse_view)):
        module.train(self.training)
        values = sequence if direction == 0 else sequence.flip(0)
        state = None if hx is None else hx[direction:direction+1]
        pieces = []
        def recurrent_chunk(chunk, incoming, view=module):
            return view(chunk.contiguous(), None if incoming is None else incoming.contiguous())
        for first in range(0, len(values), self._recovery_chunk_holds):
            chunk = values[first:first+self._recovery_chunk_holds]
            if torch.is_grad_enabled():
                output, state = checkpoint(recurrent_chunk, chunk, state,
                    use_reentrant=False, preserve_rng_state=True)
            else:
                output, state = recurrent_chunk(chunk, state)
            pieces.append(output)
        result = torch.cat(pieces, dim=0)
        outputs.append(result if direction == 0 else result.flip(0))
        final.append(state)
    return torch.cat(outputs, -1), torch.cat(final, 0)


def install_chunked_gru(model, *, chunk_holds=1024):
    module = model.genomic_gru
    if type(module) is not nn.GRU or module.num_layers != 1 or not module.bidirectional \
            or module.batch_first or module.dropout != 0 or module.input_size != 35 \
            or module.hidden_size != 32 or not module.bias:
        raise ValueError("Exact original one-layer European bidirectional GRU is required")
    if type(chunk_holds) is not int or not 1 <= chunk_holds <= 4096:
        raise ValueError("Explicit bounded recurrent execution chunk size is required")
    before = [(name, id(parameter), tuple(parameter.shape)) for name, parameter in model.named_parameters()]
    keys = tuple(model.state_dict())
    views = []
    with torch.random.fork_rng(devices=[]):
        for suffix in ("", "_reverse"):
            view = nn.GRU(35, 32, num_layers=1, bidirectional=False, dropout=0)
            for name in ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"):
                setattr(view, name, getattr(module, name+suffix))
            views.append(view)
    object.__setattr__(module, "_recovery_forward_view", views[0])
    object.__setattr__(module, "_recovery_reverse_view", views[1])
    module._recovery_chunk_holds = chunk_holds
    module.forward = MethodType(chunked_bidirectional_forward, module)
    after = [(name, id(parameter), tuple(parameter.shape)) for name, parameter in model.named_parameters()]
    if before != after or keys != tuple(model.state_dict()):
        raise RuntimeError("Recurrent execution adapter changed model parameter identities or saved keys")
    return model
