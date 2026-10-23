"""Parallel affine prefix for the frozen stable-leaf history window.

This replaces only the arithmetic schedule of _window. There is no recurrent
state detach, local reset, division by a cumulative decay, model change or I/O.
The initial state and both traversal directions match the frozen recurrence.
"""
import torch

def parallel_window(forcing, rho, source, groups, boundary, reverse):
    n, haplotypes = groups.shape
    rank = forcing.shape[1]
    if n == 0 or rho.shape != (n,) or source.shape != (n,):
        raise ValueError("Nonempty original scalar-decay window required")
    leaves = forcing.index_select(0, groups.reshape(-1)).reshape(n, haplotypes, rank)
    if reverse:
        leaves, rho, source = leaves.flip(0), rho.flip(0), source.flip(0)
    alpha = rho
    beta = source[:, None, None] * leaves
    if boundary is None:
        # The original first traversal state is exactly its forcing, not beta*f.
        alpha = torch.cat((rho.new_zeros(1), rho[1:]))
        beta = torch.cat((leaves[:1], beta[1:]), dim=0)
    offset = 1
    while offset < n:
        previous_alpha, previous_beta = alpha, beta
        alpha = torch.cat((previous_alpha[:offset],
                           previous_alpha[offset:] * previous_alpha[:-offset]), dim=0)
        beta = torch.cat((previous_beta[:offset],
                          previous_beta[offset:] +
                          previous_alpha[offset:, None, None] * previous_beta[:-offset]), dim=0)
        offset *= 2
    states = beta if boundary is None else beta + alpha[:, None, None] * boundary
    final = states[-1]
    return (states.flip(0) if reverse else states), final
