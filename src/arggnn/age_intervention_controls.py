"""Split age query-propagation/reference-forcing attribution interventions.

Mixed QQ/QR blocks need not come from one physical reduced tree. The classes
reuse frozen fixed-anchor math and parameters, without modifying its caches.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from .anchor_kernel import (AnchorKernelBank, AnchorKernelContext,
                           AnchorKernelModel, AnchorOperators)


class SplitAgeAnchorKernelModel(AnchorKernelModel):
    """Hold either query geometry or reference forcing at repeated plain.

    Numerator and matching ones mass always use the same chosen CQR. This is a
    three-branch attribution intervention, not a physical full Kron response.
    """
    def __init__(self,input_dim: int = 47,num_classes: int = 4,hidden_dim: int = 32,
                 num_layers: int = 2,latent_dim: int = 8,kernel_depth: int = 2,
                 num_scales: int = 3,geometry: str = "age",coupling: str = "full",
                 num_anchors: int = 32,initial_coupling: float | Sequence[float] = 20.,
                 initial_damping: float | Sequence[float] = 1.,rate_parameterization: str = "log",
                 age_intervention: str = "source_only"):
        if age_intervention not in {"source_only","propagation_only"}:
            raise ValueError("age_intervention must be source_only or propagation_only")
        if geometry != "age" or num_scales != 3:
            raise ValueError("Split age intervention requires geometry='age' and three fixed age branches")
        if coupling not in {"full","diagonal","gain_matched_diagonal"}:
            raise ValueError("Split age intervention supports full, diagonal or gain_matched_diagonal")
        super().__init__(input_dim=input_dim,num_classes=num_classes,hidden_dim=hidden_dim,
            num_layers=num_layers,latent_dim=latent_dim,kernel_depth=kernel_depth,
            num_scales=num_scales,geometry=geometry,coupling=coupling,num_anchors=num_anchors,
            initial_coupling=initial_coupling,initial_damping=initial_damping,
            rate_parameterization=rate_parameterization)
        self.age_intervention = age_intervention

    def selected_operators(self,prepared: AnchorKernelBank) -> AnchorOperators:
        """Select fixed tensor aliases, with no copying, mutation or factorization."""
        if prepared.num_scales != 3 or not {"plain","age"}.issubset(prepared.operators):
            raise ValueError("Prepare matched three-branch plain and age operators first")
        plain,age = prepared.operators["plain"],prepared.operators["age"]
        if plain.cqq.shape != age.cqq.shape or plain.cqr.shape != age.cqr.shape:
            raise ValueError("Plain and age blocks must share graph/query/reference shape and order")
        query = plain if self.age_intervention=="source_only" else age
        reference = age if self.age_intervention=="source_only" else plain
        return AnchorOperators(query.cqq,reference.cqr,query.eigenvalues,query.eigenvectors)

    def intervention_bank(self,prepared: AnchorKernelBank) -> AnchorKernelBank:
        """Return a lightweight wrapper sharing the observed bank and layout."""
        selected = self.selected_operators(prepared)
        operators = dict(prepared.operators)
        operators["age"] = selected
        return replace(prepared,operators=operators)

    def encode_context(self,prepared: AnchorKernelBank) -> AnchorKernelContext:
        return super().encode_context(self.intervention_bank(prepared))

    def configuration_manifest(self,prepared: AnchorKernelBank | None = None) -> dict[str,Any]:
        result = super().configuration_manifest(prepared)
        result["core_configuration_sha256"] = result.pop("configuration_sha256")
        query_source = "repeated_plain" if self.age_intervention=="source_only" else "age"
        reference_source = "age" if self.age_intervention=="source_only" else "repeated_plain"
        result.update(kind="split_age_QQ_QR_anchor_attribution_intervention",
            age_intervention=self.age_intervention,
            query_operator_source=query_source,
            query_eigensystem_source=query_source,
            reference_forcing_operator_source=reference_source,
            reference_mass_operator_source=reference_source,
            physical_full_boundary_Kron_response=False,
            operator_status="QQ and QR selected independently; may not be blocks of one reduced tree response",
            control_interpretation="separates age reference forcing from age query propagation; same source encoder/maps/capacity",
            limitations="conditional attribution intervention; not physical tree geometry, a guaranteed null, performance result or isolated pathway at nonlinear depth",
            intervention_implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        result["configuration_sha256"] = hashlib.sha256(json.dumps(result,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        return result
