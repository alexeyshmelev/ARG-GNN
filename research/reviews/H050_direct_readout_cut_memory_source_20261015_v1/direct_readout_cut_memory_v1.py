"""H050 source-only: H045 own-state predictor plus reused H049 internal cut memory.

No data/checkpoint reader, target resolver, guard, writer, launcher or CLI.
Root supplies authenticated CPU warm-start payloads and future corrected recipes.
"""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass, fields
from itertools import combinations
import copy
import torch
from arggnn.ordered_clade_episode_gnn_v1 import (
    OrderedCladeEpisodeGNNV1, OrderedCladeEpisodesV1, MODEL_KWARGS, _sha)
from arggnn.mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeInputsV1
from arggnn import train_ordered_clade_episode_v1 as h45
from research.reviews.order2_cut_memory_implementation_20261015_v1 import order2_cut_memory_v1 as memory_core
from research.reviews.H049_order2_cut_memory_prepare_source_20261015_v2 import cut_recipe_codec_v2 as codec

MEMORY_ARMS = memory_core.ARMS
CONTINUATION = "h045_continued_training"
ARMS = MEMORY_ARMS + (CONTINUATION,)
PARAMETERS = {a: 32325 for a in MEMORY_ARMS} | {CONTINUATION: 31812}
WARM_CHECKPOINT_REF = {"path": "experiments/results/H045_GPU5_episode_train_seed101_v1/own_state.pass_30.pt", "sha256": "11dc132ac1958c452fa1d2651bd261258aac6c036ebbdbe7175f235148905331"}
WARM_COMPLETION_REF = {"path": "experiments/results/H045_GPU5_episode_train_seed101_v1/completed.json", "sha256": "258c7da76aa1f693c684094569100e19a3430566c46fe9452e841eb5127f955e"}
MEMORY_KEYS = frozenset(("memory.U.weight", "memory.O.weight", "memory.ell_theta"))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _bank_tensors(bank):
    values = []
    for field in fields(bank):
        value = getattr(bank, field.name)
        values.extend(value if isinstance(value, tuple) else (value,))
    require(all(isinstance(x, torch.Tensor) for x in values), "Native bank tensor fields")
    return tuple(values)


def _episode_tensors(episodes):
    return (episodes.flags.node_flags,) + tuple(getattr(episodes, name) for name in
        ("member_node", "member_hold", "member_episode", "episode_size", "hold_span"))


@dataclass(frozen=True)
class H050BundleV1:
    bank: MutationRoutedWholeTreeInputsV1
    episodes: OrderedCladeEpisodesV1
    cut: memory_core.CutPlanV1
    bank_stamps: tuple
    episode_stamps: tuple

    @property
    def native_holds(self):
        return self.episodes.native_holds

    @property
    def haplotypes(self):
        return self.episodes.haplotypes

    def validate(self, bank=None):
        require(type(self.bank) is MutationRoutedWholeTreeInputsV1
            and (bank is None or bank is self.bank), "Exact same original native bank")
        require(self.bank_stamps == memory_core._stamps(_bank_tensors(self.bank))
            and self.episode_stamps == memory_core._stamps(_episode_tensors(self.episodes)),
            "Immutable bound observation and episode tensors")
        require(self.episodes.native_nodes == self.bank.parent.numel()
            and tuple(self.bank.sample_index.shape) == (self.native_holds, 952)
            and self.haplotypes == 952 and tuple(self.bank.query_column.shape) == (686,)
            and torch.equal(self.episodes.hold_span, self.bank.spans),
            "Original full952/all686 support and identical physical spans")
        self.cut.validate(self.bank)
        return self.bank

    def to(self, device):
        self.validate()
        bank = self.bank.to(device)
        episodes = self.episodes.to(device)
        cut = self.cut._to_for_bank(device, self.bank, bank)
        result = type(self)(bank, episodes, cut,
            memory_core._stamps(_bank_tensors(bank)),
            memory_core._stamps(_episode_tensors(episodes)))
        result.validate()
        return result

    def tensor_bytes(self):
        return self.episodes.tensor_bytes() + self.cut.tensor_bytes()


def bind_prepared_v1(bank, episodes, episode_metadata, cut):
    """Bind caller-supplied observation-only H045 episode and decoded cut recipes."""
    require(type(bank) is MutationRoutedWholeTreeInputsV1
        and bank.parent.device.type == "cpu"
        and isinstance(episodes, OrderedCladeEpisodesV1)
        and isinstance(cut, memory_core.CutPlanV1), "Verified original CPU recipes")
    require(episode_metadata.get("schema") == "ordered_exact_proper_clade_episodes_v1"
        and episode_metadata.get("observation_only") is True
        and episode_metadata.get("labels_times_targets_or_outcomes_used") is False
        and episode_metadata["topology_sha256"] == _sha(bank.parent.numpy())
        and episode_metadata["sample_index_sha256"] == _sha(bank.sample_index.numpy())
        and episode_metadata["span_sha256"] == _sha(bank.spans.numpy()),
        "Episode metadata matches exact original topology/sample/spans")
    # Payload hashes belong to the verified H045 builder; no ancestry labels enter.
    names = ("node_flags", "member_node", "member_hold", "member_episode", "episode_size", "hold_span")
    for name, tensor in zip(names, _episode_tensors(episodes)):
        require(tensor.device.type == "cpu" and _sha(tensor.numpy())
            == episode_metadata["array_sha256"][name], "Exact prepared H045 episode content")
    result = H050BundleV1(bank, episodes, cut,
        memory_core._stamps(_bank_tensors(bank)),
        memory_core._stamps(_episode_tensors(episodes)))
    result.validate()
    return result


def bind_compact_recipe_v1(bank, episodes, episode_metadata, payload, receipt):
    """Reuse corrected codec_v2; root must authenticate genuine all40 completion/wait."""
    return bind_prepared_v1(bank, episodes, episode_metadata,
        codec.decode_v1(payload, receipt, bank))


class DirectReadoutCutMemoryGNNV1(OrderedCladeEpisodeGNNV1):
    """Only the encoder splice is new; H045 GRU/classifier/readout stay inherited."""
    def __init__(self, *, memory_arm=MEMORY_ARMS[0], checkpoint_memory=True):
        require(memory_arm in MEMORY_ARMS, "Registered H050 memory rule")
        super().__init__(**MODEL_KWARGS, arm="own_state")
        self.memory = memory_core.CutMemoryV1(
            arm=memory_arm, checkpoint_memory=checkpoint_memory)
        self.float()
        require(sum(p.numel() for p in self.parameters()) == 32325,
            "Unchanged31812 plus exactly513 memory parameters")

    def encode_trees(self, bank, *, flags, control="aligned"):
        bundle = flags
        require(isinstance(bundle, H050BundleV1), "Observation-bound H050 bundle")
        bundle.validate(bank)
        episodes = bundle.episodes
        base, ug, dg, degree = self.prepare_tree_base_v1(bank, control=control)
        upward, outside = self.sweep_tree_state_v1(bank, base, ug, dg, degree)
        injection = (self.clade_coupling(upward, episodes.flags)
            + self.episode_coupling(upward, episodes, arm="own_state"))
        delta = self.memory(upward, outside, bank.spans, bundle.cut)
        injection = injection.index_add(0, bundle.cut.representatives, delta)
        second_up, second_out = self.sweep_tree_state_v1(
            bank, base + injection, ug, dg, degree)
        return self.leaf_context_v1(bank, second_up, second_out)


def validate_supplied_warmstart_v1(payload, completion, *, checkpoint_ref, completion_ref):
    """Validate supplied metadata/tensor structure; caller authenticates file bytes."""
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


def family_from_base_state_v1(state, *, checkpoint_memory=True):
    """Pure tensor construction. Production callers use make_matched_family_v1."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(101)
        base = OrderedCladeEpisodeGNNV1(**MODEL_KWARGS, arm="own_state")
        expected = base.state_dict()
        require(isinstance(state, Mapping) and set(state) == set(expected), "Complete H045 state keys")
        for key, value in state.items():
            require(isinstance(value, torch.Tensor) and value.device.type == "cpu"
                and value.dtype == torch.float32 and value.shape == expected[key].shape
                and bool(torch.isfinite(value).all()), "Finite original FP32 H045 state: " + key)
        base.load_state_dict(state, strict=True)
        first = DirectReadoutCutMemoryGNNV1(checkpoint_memory=checkpoint_memory)
        copied = first.load_state_dict(state, strict=False)
        require(set(copied.missing_keys) == MEMORY_KEYS and not copied.unexpected_keys,
            "Only the513 new memory scalars may be absent")
        family = {}
        for arm in MEMORY_ARMS:
            family[arm] = copy.deepcopy(first)
            family[arm].memory.arm = arm
        family[CONTINUATION] = base
    for arm, model in family.items():
        require(sum(p.numel() for p in model.parameters()) == PARAMETERS[arm]
            and all(p.requires_grad and p.dtype == torch.float32 for p in model.parameters()),
            "Exact disclosed capacity and full FP32 trainability")
        require(all(torch.equal(model.state_dict()[k], value) for k,value in state.items()),
            "All H045 trained weights retained in every arm")
    require(type(family[CONTINUATION]) is OrderedCladeEpisodeGNNV1
        and not hasattr(family[CONTINUATION], "memory"), "Unmodified continuation, no unused parameters")
    for left, right in combinations(ARMS, 2):
        require(not ({p.data_ptr() for p in family[left].parameters()}
            & {p.data_ptr() for p in family[right].parameters()}), "Independent learned storage")
    return family


def make_matched_family_v1(payload, completion, *, checkpoint_ref, completion_ref,
                           checkpoint_memory=True):
    state = validate_supplied_warmstart_v1(payload, completion,
        checkpoint_ref=checkpoint_ref, completion_ref=completion_ref)
    family = family_from_base_state_v1(state, checkpoint_memory=checkpoint_memory)
    return family, dict(arms=list(ARMS), parameters_per_arm=PARAMETERS,
        original_H045_own_state_weights_preserved=True,
        memory_output_zero=all(not bool(family[a].memory.O.weight.count_nonzero()) for a in MEMORY_ARMS),
        common_initial_function_required=True, no_memory_control_has_no_extra_parameters=True,
        optimizer_state_must_reset=True, checkpoint_bytes_authentication="external_root")


def episode_argument_v1(arm, bundle):
    require(arm in ARMS and isinstance(bundle, H050BundleV1), "Registered bound arm")
    bundle.validate()
    return bundle.episodes if arm == CONTINUATION else bundle


def validate_arm_model_v1(arm, model):
    require(arm in ARMS and model.arm == "own_state", "Common H045 own-state model")
    if arm == CONTINUATION:
        require(type(model) is OrderedCladeEpisodeGNNV1 and not hasattr(model,"memory"),
            "Unmodified H045 continuation control")
    else:
        require(type(model) is DirectReadoutCutMemoryGNNV1 and model.memory.arm == arm,
            "Exact H050 memory rule for this arm")
    return model
