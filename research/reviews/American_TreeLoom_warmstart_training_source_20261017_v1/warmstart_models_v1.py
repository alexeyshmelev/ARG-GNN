"""Matched final-mechanism TreeLoom transfer from accepted C3 saved600 weights.

Scientific source only. No files, private target resolver, writer or launcher.
Both arms retain structural conditioning, two shared sweeps and whole-parent GRUs.
"""
from __future__ import annotations
import copy, resource, time
import torch
from torch import nn
from torch.nn import functional as F
from collections.abc import Mapping
from arggnn.american_c3_common_v1 import require, CLASSES
from arggnn.american_c3_gnn_model_v1 import AmericanC3WholeTreeGNNV1
from arggnn.american_c3_gnn_targets_v1 import TRAIN, TrainFocalTargetVectorV1
from arggnn.american_c3_gnn_stream_train_v1 import _parameter_sha
from arggnn.anonymous_clade_flag_control_v1 import RankEightCladeFlagAdapterV1
from arggnn.ordered_clade_episode_gnn_v1 import RankEightCladeEpisodeAdapterV1
from research.reviews.American_TreeLoom_full_mechanism_source_20261017_v2 import american_treeloom_full_c3_v2 as full

FAMILY = "American_TreeLoom_saved600_warmstart_matched_v1"
CONDITION = "accepted_C3_saved600_backbone_plus600_new_native_focal_updates_per_arm"
ARMS = ("stable_leaf_memory", "no_added_memory")
PARAMETERS = {ARMS[0]: 32132, ARMS[1]: 31619}
TENSORS = {ARMS[0]: 49, ARMS[1]: 46}
NEW_PREFIXES = ("clade_coupling.", "episode_coupling.", "memory.")
prepare_from_loaded_observation_v1 = full.prepare_from_loaded_observation_v1
AmericanTreeLoomWholeParentV1 = full.AmericanTreeLoomWholeParentV1


def _encode_full(model, bundle, *, memory):
    expected = WarmstartTreeLoomFullV1 if memory else WarmstartTreeLoomNoMemV1
    require(type(model) is expected and type(bundle) is AmericanTreeLoomWholeParentV1,
        "Exact new matched arm and complete globally prepared parent")
    if memory:
        require(model.memory.arm == "stable_leaf_memory", "Fixed final stable-leaf memory")
    else:
        require(not hasattr(model, "memory"), "NoMem has no added memory module or parameters")
    bank = bundle.validate()
    p = next(model.parameters())
    require(bank.parent.device == p.device
        and all(q.dtype == torch.float32 and q.device == p.device for q in model.parameters()),
        "Complete FP32 observation parent and matched model on one admitted device")
    base, ug, dg, degree = model.prepare_tree_base_v1(bank)
    upward, outside = model.sweep_tree_state_v1(bank, base, ug, dg, degree)
    injection = (model.clade_coupling(upward, bundle.episodes.flags)
        + model.episode_coupling(upward, bundle.episodes, arm="own_state"))
    if memory:
        delta = model.memory(upward, outside, bank.spans, bundle.cut)
        injection = injection.index_add(0, bundle.cut.representatives, delta)
    second_up, second_out = model.sweep_tree_state_v1(
        bank, base + injection, ug, dg, degree)
    return model.leaf_context_v1(bank, second_up, second_out)


def _contract(model, arm):
    contract = full.AmericanTreeLoomC3FullV1.checkpoint_contract(model)
    contract.update(schema=FAMILY+"_checkpoint_contract", family=FAMILY,
        condition=CONDITION, arm=arm,
        memory_arm="stable_leaf_memory" if arm == ARMS[0] else "absent",
        memory_parameters=513 if arm == ARMS[0] else 0,
        memory_rank=8 if arm == ARMS[0] else 0,
        memory_length_scale_initial_bp=5000. if arm == ARMS[0] else None,
        memory_length_scale_bounds_bp=[100.,100000.] if arm == ARMS[0] else None,
        expected_parameters=PARAMETERS[arm], expected_trainable_tensors=TENSORS[arm],
        warmstart="accepted_saved600_final_backbone_weights_only",
        inherited_saved_logical_updates=600, fresh_optimizer=True,
        added_structural_output_projections_zero_at_initialization=True,
        legacy_saved600_is_final_TreeLoom=False)
    return contract


class WarmstartTreeLoomFullV1(full.AmericanTreeLoomC3FullV1):
    def encode_full_parent_v1(self, bundle):
        return _encode_full(self, bundle, memory=True)
    def checkpoint_contract(self):
        return _contract(self, ARMS[0])


class WarmstartTreeLoomNoMemV1(full.AmericanTreeLoomC3FullV1):
    def __init__(self):
        # Original C3 body plus both original local residuals. No memory is created.
        AmericanC3WholeTreeGNNV1.__init__(self, hidden_dim=32, class_order=CLASSES,
            expected_haplotypes=1600, donor_cohorts=2,
            site_chunk_size=256, checkpoint_sites=True)
        self.clade_coupling = RankEightCladeFlagAdapterV1()
        self.episode_coupling = RankEightCladeEpisodeAdapterV1()
        self.float()
        require(sum(p.numel() for p in self.parameters()) == PARAMETERS[ARMS[1]]
            and len(tuple(self.parameters())) == TENSORS[ARMS[1]]
            and all(p.requires_grad and p.dtype == torch.float32 for p in self.parameters()),
            "Actual future NoMem capacity and full trainability")
    def encode_full_parent_v1(self, bundle):
        return _encode_full(self, bundle, memory=False)
    def checkpoint_contract(self):
        return _contract(self, ARMS[1])


def validate_arm_model_v1(arm, model):
    require(arm in ARMS and type(model) is
        (WarmstartTreeLoomFullV1 if arm == ARMS[0] else WarmstartTreeLoomNoMemV1),
        "Distinct matched final-method arm family")
    require(sum(p.numel() for p in model.parameters()) == PARAMETERS[arm]
        and len(tuple(model.parameters())) == TENSORS[arm]
        and all(p.requires_grad and p.dtype == torch.float32 for p in model.parameters()),
        "All current matched-arm parameters trainable in FP32")
    return model


def make_matched_family_v1(shared_state, warmstart_binding, *, device):
    """Copy exact29571 accepted backbone scalars; initialize new residuals once.

    Caller has already authenticated the supplied final epoch30 saved600 state.
    Neither old optimizer moments nor old RNG enter this fresh transfer condition.
    """
    require(isinstance(shared_state, Mapping) and isinstance(warmstart_binding, Mapping)
        and warmstart_binding.get("schema") == "TreeLoom_accepted_saved600_warmstart_binding_v1"
        and warmstart_binding.get("accepted_saved_artifacts_only") is True
        and warmstart_binding.get("epoch") == 30
        and warmstart_binding.get("saved_logical_updates") == 600
        and warmstart_binding.get("model_parameters") == 29571
        and warmstart_binding.get("original_producer_success_claimed") is False,
        "Explicit accepted-artifact final-backbone warmstart, no producer relabeling")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(70001)
        with torch.device("cpu"):
            memory = WarmstartTreeLoomFullV1()
        torch.random.default_generator.manual_seed(70001)
        with torch.device("cpu"):
            nomem = WarmstartTreeLoomNoMemV1()
    family = {ARMS[0]: memory, ARMS[1]: nomem}
    baseline_keys = {k for k in memory.state_dict() if not k.startswith(NEW_PREFIXES)}
    require(set(shared_state) == baseline_keys and len(baseline_keys) == 38,
        "Exact shared backbone keys, no C4 slicing or omitted backbone parameters")
    require(sum(x.numel() for x in shared_state.values()) == 29571,
        "All accepted29571 backbone scalars supplied")
    for arm, model in family.items():
        expected = model.state_dict()
        for key, value in shared_state.items():
            require(isinstance(value, torch.Tensor) and value.device.type == "cpu"
                and value.dtype == torch.float32 and value.shape == expected[key].shape
                and bool(torch.isfinite(value).all()), "Finite accepted C3 backbone tensor: "+key)
        copied = model.load_state_dict(shared_state, strict=False)
        require(set(copied.missing_keys) == set(expected)-baseline_keys
            and all(k.startswith(NEW_PREFIXES) for k in copied.missing_keys)
            and not copied.unexpected_keys
            and all(torch.equal(model.state_dict()[k], v) for k,v in shared_state.items()),
            "Every shared backbone weight copied exactly; only new residual keys initialize")
        require(not bool(model.clade_coupling.O.weight.count_nonzero())
            and not bool(model.episode_coupling.O.weight.count_nonzero())
            and (arm != ARMS[0] or not bool(model.memory.O.weight.count_nonzero())),
            "Both structural outputs and full-arm memory output initialize at zero")
        validate_arm_model_v1(arm, model)
    common = nomem.state_dict()
    require(all(torch.equal(memory.state_dict()[k], v) for k,v in common.items()),
        "Identical shared backbone AND structural initialization across matched arms")
    require(not ({p.data_ptr() for p in memory.parameters()} & {p.data_ptr() for p in nomem.parameters()}),
        "Independent parameter storage")
    optimizers = {}
    for arm, model in family.items():
        model.to(device=device, dtype=torch.float32)
        optimizers[arm] = torch.optim.AdamW(model.parameters(),
            lr=.001, weight_decay=.0001, betas=(.9,.999), eps=1e-8,
            maximize=False, amsgrad=False, foreach=False, fused=False,
            capturable=False, differentiable=False)
        require(not optimizers[arm].state, "Fresh independent AdamW; no inherited moments")
    return family, optimizers, dict(family=FAMILY, condition=CONDITION,
        warmstart_binding=copy.deepcopy(dict(warmstart_binding)),
        shared_backbone_scalars_copied=29571, new_outputs_zero=True,
        common_initial_function_expected=True, old_optimizer_and_rng_restored=False,
        parameters_per_arm=PARAMETERS, tensors_per_arm=TENSORS)


def train_complete_parent_step_v1(arm, model, bundle, targets, optimizer, *,
        resource_check, first_update=False):
    """One exact complete-parent focal-mean CE and optimizer step for one arm."""
    validate_arm_model_v1(arm, model)
    require(type(bundle) is AmericanTreeLoomWholeParentV1
        and bundle.split == "train" and bundle.parent_id in TRAIN
        and isinstance(targets, TrainFocalTargetVectorV1)
        and targets.parent_id == bundle.parent_id
        and targets.focal_registry_ref == bundle.focal_registry_ref
        and targets.focal_identity_sha256 == bundle.focal_identity_sha256
        and targets.labels_cpu.device.type == "cpu"
        and targets.labels_cpu.dtype == torch.long and targets.labels_cpu.shape == (10000,)
        and bool(((targets.labels_cpu >= 0) & (targets.labels_cpu < 3)).all())
        and callable(resource_check), "Separate sealed TRAIN-only native focal targets")
    require(type(optimizer) is torch.optim.AdamW and len(optimizer.param_groups) == 1,
        "Original minimizing default AdamW")
    group = optimizer.param_groups[0]
    require(group["lr"] == .001 and group["weight_decay"] == .0001
        and group["betas"] == (.9,.999) and group["eps"] == 1e-8
        and group["maximize"] is False and group["amsgrad"] is False
        and len(group["params"]) == TENSORS[arm]
        and len({id(p) for p in group["params"]}) == TENSORS[arm]
        and {id(p) for p in group["params"]} == {id(p) for p in model.parameters()}
        and all(p.requires_grad for p in model.parameters()),
        "Every live structural/memory/body parameter in the exact optimizer")
    bank = bundle.validate()
    started = time.monotonic()
    resource_check("TreeLoom_"+arm+"_complete_parent_begin")
    model.train();optimizer.zero_grad(set_to_none=True)
    if first_update and bank.parent.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(bank.parent.device)
    before = _parameter_sha(model) if first_update else None
    added = {k:p.detach().cpu().clone() for k,p in model.named_parameters()
        if first_update and k.startswith(NEW_PREFIXES)}
    with torch.autocast(device_type=bank.parent.device.type, enabled=False):
        logits = model(bundle)
        require(logits.shape == (10000,3), "All original native focal logits")
        loss = F.cross_entropy(logits, targets.labels_cpu.to(logits.device), reduction="mean")
    require(bool(torch.isfinite(loss)), "Finite complete-parent native-focal CE")
    resource_check("TreeLoom_"+arm+"_complete_parent_backward")
    loss.backward()
    grads = [p.grad for p in model.parameters()]
    require(all(g is not None and bool(torch.isfinite(g).all()) for g in grads)
        and any(bool((g != 0).any()) for g in grads),
        "All parameter gradients connected and finite; global gradient nonzero")
    added_gradient = {k:float(p.grad.detach().abs().sum()) for k,p in model.named_parameters()
        if first_update and k.startswith(NEW_PREFIXES)}
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    require(bool(torch.isfinite(norm)), "Finite gradient norm")
    resource_check("TreeLoom_"+arm+"_complete_parent_optimizer")
    optimizer.step()
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), "Finite updated weights")
    qualification = None
    if first_update:
        after = _parameter_sha(model)
        require(before != after, "First scheduled actual update changes parameters")
        qualification = dict(parameter_checksum_changed=True, all_gradients_finite=True,
            global_gradient_nonzero=True,
            added_parameter_gradient_L1=added_gradient,
            added_parameter_changed={k:not torch.equal(p.detach().cpu(),added[k])
                for k,p in model.named_parameters() if k in added},
            no_all_scalars_nonzero_gradient_claim=True,
            zero_output_delays_input_projection_loss_gradient=True)
    optimizer.zero_grad(set_to_none=True)
    resource_check("TreeLoom_"+arm+"_complete_parent_end")
    return dict(arm=arm,parent_id=bundle.parent_id,records=10000,
        loss=float(loss.detach()),gradient_norm=float(norm),
        whole_parent_optimizer_updates=1,block_reset_or_truncated_BPTT=False,
        detached_recurrent_history=False,all_trees_and_samples_retained=True,
        first_real_update_checks=qualification,
        complete_parent_step_wall_seconds_including_checks=time.monotonic()-started,
        timing_is_inside_declared600_update_run=True,
        observed_resources=dict(process_peak_RSS_bytes=int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,
            CUDA_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(bank.parent.device))
                if bank.parent.device.type=="cuda" else None,
            CUDA_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(bank.parent.device))
                if bank.parent.device.type=="cuda" else None,
            CUDA_peak_reset_at_first_scheduled_update=first_update),
        ancestry_accuracy=None)
