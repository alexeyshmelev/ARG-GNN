"""Capacity and source-feedback attribution controls for fixed-anchor kernels.

These classes reuse the frozen AnchorKernelModel parameter schema without
changing its implementation. Neither is a guaranteed null or a performance
claim. Static queries exchange graph-local context but no coordinate history.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .anchor_kernel import (AnchorKernelBank, AnchorKernelContext,
    AnchorKernelLayer, AnchorKernelModel, AnchorOperators)


def _observed_local_embeddings(model: AnchorKernelModel, prepared: AnchorKernelBank,
                               *, require_anchors: bool) -> tuple[Tensor, Tensor]:
    bank = prepared.bank
    if (prepared.num_scales != model.num_scales or model.geometry not in prepared.operators
            or (require_anchors and prepared.layout.anchors.numel() != model.num_anchors)):
        raise ValueError("Prepared branch/anchor count or geometry differs from control")
    weight = model.encoder.input_linear.weight
    if bank.leaf_x.shape[-1] != model.input_dim or bank.leaf_x.dtype != weight.dtype or bank.leaf_x.device != weight.device:
        raise ValueError("Prepare observed features with matching model dimension/device/dtype")
    if not bool(torch.isfinite(bank.leaf_x).all()):
        raise ValueError("Observed feature bank must be finite")
    h = model.encoder(bank.leaf_x,bank.query_cache.normalized_response)
    return h[:,bank.query_cache.query_indices],h[:,bank.query_cache.reference_indices]


def _manifest(model: AnchorKernelModel, prepared: AnchorKernelBank | None,
              changes: dict[str, Any]) -> dict[str, Any]:
    result = AnchorKernelModel.configuration_manifest(model,prepared)
    result["core_configuration_sha256"] = result.pop("configuration_sha256")
    result.update(changes)
    result["controls_implementation_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["configuration_sha256"] = hashlib.sha256(json.dumps(result,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return result


class NoncontextualAnchorKernelModel(AnchorKernelModel):
    """Every layer projects the original local anchor queries into sources.

    Kernel messages, reference projections, normalization, gates and target
    residual updates are unchanged. Later target contexts do not become sources.
    Anchor residuals are retained as an audit output but never feed back into
    source projections in this control.
    """
    def encode_context(self, prepared: AnchorKernelBank) -> AnchorKernelContext:
        query,reference = _observed_local_embeddings(self,prepared,require_anchors=True)
        layout,bank = prepared.layout,prepared.bank
        original_anchor_query = query[layout.anchor_graph]
        fixed_reference = reference[layout.anchor_graph]
        anchors = original_anchor_query
        targets = query[bank.target_graph]
        messages = []
        for layer in self.layers:
            message = layer.messages(prepared,original_anchor_query,fixed_reference,self.geometry,self.coupling)
            anchors = layer.update(anchors,message.anchors)
            targets = layer.update(targets,message.targets)
            messages.append(message)
        return AnchorKernelContext(anchors,targets,tuple(messages))

    def configuration_manifest(self,prepared: AnchorKernelBank | None = None) -> dict[str,Any]:
        return _manifest(self,prepared,{
            "kind":"noncontextual_source_fixed_anchor_control",
            "source_update":"original independent graph-local query anchor embedding at every layer",
            "decoder":"same layer-wise exact kernel readouts and residual updates; target context retained",
            "anchor_source_feedback":False,
            "control_interpretation":"source-feedback attribution; nonlinear decoder depth, kernels and global context remain",
            "limitations":"not a guaranteed null or performance claim; inherited finite-anchor and feature/geometry contracts",
        })


@dataclass(frozen=True)
class StaticLocalMessages:
    message: Tensor              # [G,Q,H], normalized local equilibrium
    numerator: Tensor            # [G,Q,H], mixed K^-1 source
    mass: Tensor                 # [G,Q,H], mixed K^-1 matching ones source
    sources: Tensor              # [G,S,Q,H]
    mass_sources: Tensor         # [G,S,Q,H]


@dataclass(frozen=True)
class StaticLocalContext:
    local: Tensor                # [G,Q,D], final graph-local contextual queries
    targets: Tensor              # [P,Q,D], original target_graph gather
    layers: tuple[StaticLocalMessages,...]


def static_local_messages(layer: AnchorKernelLayer, operators: AnchorOperators,
                          query: Tensor, reference: Tensor) -> StaticLocalMessages:
    """Full cached K^-1 on numerator AND reference-inclusive ones mass.

    Each graph is processed independently. Sources share layer q/r maps across
    branches, references stay original, and only graph-local queries update.
    No coordinate scan, graph average, online solve or eigendecomposition occurs.
    """
    rates,damping = layer.rates,layer.damping
    if not bool(torch.isfinite(rates).all()) or not bool(torch.isfinite(damping).all()) or bool((rates<=0).any()) or bool((damping<=0).any()):
        raise ValueError("Local equilibrium rates must be finite and positive")
    query_signal = layer.query_projection(query).tanh()
    reference_signal = layer.reference_projection(reference).tanh()
    sources = query_signal[:,None]-(operators.cqr@reference_signal[:,None])*rates[None,:,None,:]
    mass_sources = torch.ones_like(sources)-operators.cqr.sum(-1)[...,None]*rates[None,:,None,:]
    decay = operators.eigenvalues[...,None]*rates[None,:,None,:]+damping[None,:,None,:]
    if not bool(torch.isfinite(decay).all()) or bool((decay<=0).any()):
        raise ValueError("Local full equilibrium decay must be finite and positive")
    bundled = torch.stack((sources,mass_sources),dim=-1)
    shape = bundled.shape
    vectors = operators.eigenvectors
    projected = (vectors.transpose(-1,-2)@bundled.flatten(-2)).reshape(shape)
    equilibrium = (vectors@(projected/decay[...,None]).flatten(-2)).reshape(shape)
    mixed = (equilibrium*layer.mixture[None,:,None,:,None]).sum(dim=1)
    numerator,mass = mixed.unbind(-1)
    if not bool(torch.isfinite(mixed).all()) or bool((mass<=0).any()):
        raise ValueError("Local normalized equilibrium mass must be finite and positive")
    return StaticLocalMessages(numerator/mass,numerator,mass,sources,mass_sources)


class StaticLocalAnchorModel(AnchorKernelModel):
    """Same learned schemas, independently repeated full local equilibrium GNN.

    Full query eigenpairs are used at each graph. Positive rates/mixtures and all
    maps/gates/norms remain in the computation, but no coordinate kernel is used.
    The anchor count is retained as a common configuration field and is unused.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.coupling != "full":
            raise ValueError("Static local control currently requires coupling='full'")

    def encode_context(self, prepared: AnchorKernelBank) -> StaticLocalContext:
        query,fixed_reference = _observed_local_embeddings(self,prepared,require_anchors=False)
        operators = prepared.operators[self.geometry]
        messages = []
        for layer in self.layers:
            message = static_local_messages(layer,operators,query,fixed_reference)
            query = layer.update(query,message.message)
            messages.append(message)
        return StaticLocalContext(query,query[prepared.bank.target_graph],tuple(messages))

    def configuration_manifest(self,prepared: AnchorKernelBank | None = None) -> dict[str,Any]:
        return _manifest(self,prepared,{
            "kind":"same_capacity_static_local_equilibrium_control",
            "source_update":"per-layer graph-local contextual query embedding; original local reference embeddings fixed",
            "memory_interface":"query context within each graph, no cross-coordinate context",
            "normalization":"full K^-1 on numerator and reference-inclusive ones mass per graph, mix then divide; no floor",
            "decoder":"graph-local residual updates followed by target_graph gather",
            "anchor_grid":"unused by static local control",
            "anchor_coordinates_used":False,"self_term":None,
            "initial_state_cross_query_exchange":False,
            "instantaneous_anchor_source_mixing":False,"static_query_source_mixing":True,
            "underlying_symmetric_block_kernel_is_PSD":None,
            "static_unormalized_equilibrium_operator_is_PSD":True,
            "control_interpretation":"same-capacity local depth attribution; geographic query mixing remains",
            "limitations":"no cross-coordinate context or time-scale interpretation; parameter degeneracies can remain; not a guaranteed null/performance claim",
        })
