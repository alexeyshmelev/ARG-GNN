"""Contiguous bounded cuDNN calls with uninterrupted full-sequence BPTT."""
from types import MethodType
import torch
from torch.utils.checkpoint import checkpoint


def chunked_scan_context(self, context, geometry, *, direction, hidden=None):
    if direction not in ("forward", "reverse") or context.ndim != 3 \
            or geometry.shape != (context.shape[0], 3):
        raise ValueError("Complete global sequence and declared direction required")
    sequence = torch.cat((context, geometry[:, None, :].expand(-1, context.shape[1], -1)), -1)
    if direction == "reverse":
        sequence = sequence.flip(0)
    sequence = sequence.contiguous()
    module = self.forward_gru if direction == "forward" else self.reverse_gru
    size = self._recovery_gru_chunk_holds
    pieces, state = [], hidden
    def recurrent_chunk(chunk, incoming):
        return module(chunk.contiguous(), None if incoming is None else incoming.contiguous())
    for first in range(0, len(sequence), size):
        chunk = sequence[first:first+size]
        if torch.is_grad_enabled():
            output, state = checkpoint(recurrent_chunk, chunk, state,
                use_reentrant=False, preserve_rng_state=True)
        else:
            output, state = recurrent_chunk(chunk, state)
        # Keep state in the graph. A detach here would change the objective.
        pieces.append(output)
    history = torch.cat(pieces, dim=0)
    return (history.flip(0).contiguous() if direction == "reverse" else history), state


def install_chunked_gru(model, *, chunk_holds=1024):
    if type(chunk_holds) is not int or not 1 <= chunk_holds <= 4096:
        raise ValueError("Explicit bounded recurrent chunk size required")
    before = [(name, id(parameter), tuple(parameter.shape)) for name, parameter in model.named_parameters()]
    model._recovery_gru_chunk_holds = chunk_holds
    model.scan_context = MethodType(chunked_scan_context, model)
    after = [(name, id(parameter), tuple(parameter.shape)) for name, parameter in model.named_parameters()]
    if before != after:
        raise RuntimeError("Recurrent execution adapter changed model parameters")
    return model
