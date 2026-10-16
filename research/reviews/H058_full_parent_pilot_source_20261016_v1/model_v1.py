"""H058 full model. Frozen packedv2/F1 integration plus memory-only stage checkpoints.

No reader, target, optimizer, launcher or resource admission is defined here.
"""
from __future__ import annotations
import copy
from collections.abc import Mapping
import torch
from torch.utils.checkpoint import checkpoint
from arggnn import train_ordered_clade_episode_v1 as h45
from arggnn.ordered_clade_episode_gnn_v1 import OrderedCladeEpisodeGNNV1
from research.reviews.same_marker_full_parent_binding_successor_source_20261016_v1.integration_hooks_v3 import (
    PackedSameMarkerSupportGNNV3, initialize_supplied_h045_state_v3)
from .protocol_v1 import (ARMS, CONTINUATION, MODEL, PARAMETERS, WARM_CHECKPOINT_REF,
    WARM_COMPLETION_REF, require)

MODEL_KWARGS = MODEL


class H058FullParentGNNV1(PackedSameMarkerSupportGNNV3):
    """Same32526 weights and F1 forward; recompute original whole-tree stages.

    Checkpoint closures capture only the immutable bank and shared stage inputs.
    They do not detach, truncate genomic history, crop a tree or alter kernels.
    Nested original whole-hold/marker-block checkpoints remain active.
    Actual backward peak remains a required scheduled-step measurement.
    """
    def __init__(self, *, checkpoint_backbone=True):
        super().__init__(**MODEL, marker_block_size=16)
        require(type(checkpoint_backbone) is bool, "Explicit stage checkpoint flag")
        self.checkpoint_backbone = checkpoint_backbone

    def _checkpoint_stage(self, function, *args, **kwargs):
        if self.checkpoint_backbone and self.training and torch.is_grad_enabled():
            return checkpoint(function, *args, use_reentrant=False,
                preserve_rng_state=True, **kwargs)
        return function(*args, **kwargs)

    def prepare_tree_base_v1(self, bank, *, control="aligned"):
        return self._checkpoint_stage(super().prepare_tree_base_v1, bank, control=control)

    def sweep_tree_state_v1(self, bank, base, ug, dg, degree):
        return self._checkpoint_stage(super().sweep_tree_state_v1, bank, base, ug, dg, degree)

    def leaf_context_v1(self, bank, upward, outside):
        return self._checkpoint_stage(super().leaf_context_v1, bank, upward, outside)


def validate_supplied_warmstart_v1(payload, completion, *, checkpoint_ref, completion_ref):
    """Copied exact H050 strong-initializer metadata check; external reader hashes bytes."""
    require(checkpoint_ref == WARM_CHECKPOINT_REF and completion_ref == WARM_COMPLETION_REF,
        "Exact completed H045 own-state pass30 refs")
    require(isinstance(completion, Mapping)
        and completion.get("schema") == "H045_episode_train_completion_v1"
        and completion.get("status") == "completed_fixed_30_pass_paired_TRAIN"
        and completion.get("hypothesis_id") == "H045"
        and completion.get("steps") == {"foreign_episode": 960, "own_state": 960}
        and len(completion.get("history", [])) == 30
        and completion.get("checkpoint_groups", {}).get("30", {}).get("own_state") == checkpoint_ref
        and completion.get("VAL_truth_read") is False and completion.get("outer_truth_read") is False
        and completion.get("conditioned_on_H040_VAL") is False
        and completion.get("scientific_refs") == list(h45.SCIENTIFIC_REFS),
        "Actual complete original H045 metadata")
    require(isinstance(payload, Mapping) and payload.get("schema") == "H045_episode_checkpoint_v1"
        and payload.get("hypothesis_id") == "H045" and payload.get("phase") == "train"
        and payload.get("arm") == "own_state" and payload.get("pass_index") == 30
        and payload.get("additional_optimizer_steps") == 960 and payload.get("seed") == 101
        and payload.get("parameter_count") == 31812 and payload.get("model_kwargs") == MODEL_KWARGS
        and payload.get("teacher_used") is False and payload.get("optimizer_state_reset") is True
        and payload.get("conditioned_on_H040_VAL") is False
        and payload.get("scientific_refs") == list(h45.SCIENTIFIC_REFS)
        and payload.get("warmstart_checkpoint_ref") == h45.WARM_CHECKPOINT
        and payload.get("initial_state_sha256") == completion["initial_state_sha256"]["own_state"]
        and all(payload.get(k) == completion[v] for k,v in
            (("input_ref", "input_ref"), ("config_ref", "config_ref"), ("observation_seal", "observations_seal"))),
        "Exact original H045 checkpoint metadata and lineage")
    return payload["model_state"]


def build_family_v1(payload, completion, *, checkpoint_ref, completion_ref,
                    device, arms=ARMS, checkpoint_backbone=True):
    require(tuple(arms) in (ARMS, (CONTINUATION,)), "Two new arms or explicitly separate continuation")
    state = validate_supplied_warmstart_v1(payload, completion,
        checkpoint_ref=checkpoint_ref, completion_ref=completion_ref)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(101)
        if tuple(arms) == ARMS:
            first = H058FullParentGNNV1(checkpoint_backbone=checkpoint_backbone)
            initialize_supplied_h045_state_v3(first, state)
            models = {ARMS[0]: first, ARMS[1]: copy.deepcopy(first)}
        else:
            first = OrderedCladeEpisodeGNNV1(**MODEL, arm="own_state")
            first.load_state_dict(state, strict=True)
            models = {CONTINUATION: first}
    for arm, value in models.items():
        require(sum(p.numel() for p in value.parameters()) == PARAMETERS[arm]
            and all(p.dtype == torch.float32 and p.requires_grad for p in value.parameters()),
            "Exact full-model FP32 trainability and capacity")
        require(all(torch.equal(value.state_dict()[k], v) for k,v in state.items()),
            "Every original H045 tensor is retained")
        value.to(device=device, dtype=torch.float32)
    pointers = [{p.data_ptr() for p in value.parameters()} for value in models.values()]
    if len(pointers) == 2:
        require(not pointers[0] & pointers[1], "Independent aligned/null learned storage")
    optimizers = {arm: torch.optim.AdamW(value.parameters(), lr=.001, weight_decay=.0001,
        betas=(.9,.999), eps=1e-8) for arm,value in models.items()}
    require(all(not value.state for value in optimizers.values()), "Fresh independent optimizer moments")
    return models, optimizers, dict(parameters={a: PARAMETERS[a] for a in arms},
        original_H045_weights_retained=True, new_output_zero=tuple(arms)==ARMS,
        checkpoint_backbone=checkpoint_backbone if tuple(arms)==ARMS else False,
        optimizer_state_reset=True, initial_predictor_equality_requires_scheduled_forward=True)
