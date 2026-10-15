"""Small H050 scientific integration API; root owns orchestration and resources."""
from __future__ import annotations
from collections.abc import Mapping
import numpy as np
import torch
from arggnn import train_ordered_clade_episode_v1 as h45
from arggnn.H040_latent_clade_execution_v1 import RAMMatrix, array_identity
from .direct_readout_cut_memory_v1 import (
    ARMS, MEMORY_ARMS, CONTINUATION, PARAMETERS, MODEL_KWARGS,
    make_matched_family_v1, episode_argument_v1, validate_arm_model_v1, require)

TRAIN, VAL, PASSES = h45.TRAIN, h45.VAL, h45.PASSES
SELECTION = h45.SELECTION


def fixed_protocol_v1():
    return dict(schema="H050_direct_readout_cut_memory_protocol_v1", hypothesis_id="H050",
        arms=list(ARMS), parameters_per_arm=PARAMETERS, MODEL=MODEL_KWARGS,
        TRAIN_parent_ids=list(TRAIN), VAL_parent_ids=list(VAL), seed=101, dtype="float32",
        passes=30, updates_per_pass=32, updates_per_arm=960, updates_all_arms=3840,
        checkpoint_passes=list(PASSES), scheduled_checkpoint_files=24,
        VAL_prediction_products_required_before_truth=192, original_grid_bp=100,
        observed_haplotypes=952, query_haplotypes=686, region_length_bp=2000000,
        learning_rate=.001, weight_decay=.0001, AdamW_betas=[.9,.999], AdamW_epsilon=1e-8,
        gradient_clip_norm=1., parent_order="shuffle_seed101_plus_pass",
        common_decoder="unchanged_H045_own_state_full_tree_BiGRU_direct_classifier",
        warmstart="completed_H045_own_state_pass30", all_parameters_trainable=True,
        independent_reset_AdamW=True, first_step_qualification="first_scheduled_step_no_extra_updates",
        loss="unchanged_H045_complete_native_hold_original_exact_span_loss",
        selection=SELECTION, early_stopping=False, VAL_influences_optimization=False,
        same_VAL_is_development_evidence=True, outer_truth_read=False,
        no_generic_operator_priority_claim=True)


def build_family_v1(payload, completion, *, checkpoint_ref, completion_ref, device="cpu"):
    family, proof = make_matched_family_v1(payload, completion,
        checkpoint_ref=checkpoint_ref, completion_ref=completion_ref)
    for model in family.values():
        model.to(device=device, dtype=torch.float32)
    optimizers = {arm: torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001,
        betas=(.9,.999), eps=1e-8) for arm,model in family.items()}
    require(all(not optimizer.state for optimizer in optimizers.values()), "Fresh independent AdamW states")
    return family, optimizers, proof


def train_step_v1(arm, model, optimizer, bundle, mass, guard, *, completed_steps,
                  after_optimizer_step=None):
    """Reuse H045's real whole-parent FP32 step; qualification is scheduled step1."""
    require(arm in ARMS and type(completed_steps) is int and 0 <= completed_steps < 960,
        "One scheduled H050 parent update")
    validate_arm_model_v1(arm, model)
    h45.science()
    return h45.train_one(model, optimizer, bundle.bank, mass,
        episode_argument_v1(arm, bundle), guard, qualify=(completed_steps == 0),
        after_optimizer_step=after_optimizer_step)


def predict_holds_v1(arm, model, bundle):
    validate_arm_model_v1(arm, model)
    model.eval()
    with torch.no_grad(), torch.autocast(device_type=bundle.bank.parent.device.type, enabled=False):
        return model.forward_train_holds(bundle.bank,
            episodes=episode_argument_v1(arm, bundle), control="aligned")


def expected_product_keys_v1():
    return {(at, arm, pid) for at in PASSES for arm in ARMS for pid in VAL}


class H050RAMMatrixV1(RAMMatrix):
    """Original native-hold RAM representation and evaluator grid mapping,192seals."""
    def __init__(self, output, reader, parents, banks, cfg, *, checkpoint_groups):
        h45.science()
        super().__init__(output, reader, parents, banks, cfg)
        require(set(parents) == set(banks) == set(VAL)
            and set(checkpoint_groups) == {str(p) for p in PASSES}
            and all(set(checkpoint_groups[str(p)]) == set(ARMS) for p in PASSES),
            "All24 actual prescribed checkpoints and original8VAL")
        self.checkpoint_groups = checkpoint_groups
        self.barrier_ref = None

    def seal(self, at, arm, pid, value, checkpoint):
        key = (at, arm, pid)
        require(key in expected_product_keys_v1() and key not in self.values and not self.verified
            and checkpoint == self.checkpoint_groups[str(at)][arm], "Exact unique H050 frozen cell/checkpoint")
        bank, parent = self.banks[pid], self.parents[pid]
        planned = len(bank.spans)*686*4*4
        require(self.bytes + planned <= self.cfg["RAM_prediction_cap_bytes"], "Root complete RAM cap preflight")
        value = np.array(value, dtype=np.float32, copy=True, order="C")
        require(value.shape == (len(bank.spans),686,4) and np.isfinite(value).all()
            and np.all(value >= 0) and np.allclose(value.sum(-1),1,rtol=1e-5,atol=1e-6),
            "Complete original native-hold FP32 probabilities")
        value.setflags(write=False)
        geometry = {k:array_identity(parent[k]) for k in
            ("positions","edges","retained_indexes","query_slots","raw_query_nodes","cohorts")}
        geometry["readout_hold"] = array_identity(bank.readout_hold.numpy())
        packet = dict(schema="H050_native_hold_RAM_seal_v1", hypothesis_id="H050",
            pass_index=at, arm=arm, parent_id=pid, checkpoint_ref=checkpoint,
            probability=array_identity(value), geometry=geometry, parameters=PARAMETERS[arm],
            prediction_array_durable=False, targets_opened_before_seal=False)
        record = self.output.save("%s.pass_%02d.%s.RAM_seal.json" % (arm,at,pid), packet)
        self.bytes += int(value.nbytes)
        self.values[key],self.records[key],self.packets[key] = value,record,packet
        self.lookup[record["path"]] = key

    def verify(self):
        require(not self.verified and set(self.values) == expected_product_keys_v1(),
            "All4arms x6checkpoints x8VAL products before any VAL truth")
        for key,value in self.values.items():
            require(not value.flags.writeable and array_identity(value) == self.packets[key]["probability"]
                and self.reader.decode_json(self.records[key]) == self.packets[key],
                "Frozen original probabilities and actual seal JSON unchanged")
        self.barrier_ref = self.output.save("all192_VAL_RAM_sealed_before_truth.json", dict(
            schema="H050_all192_original_VAL_barrier_v1", hypothesis_id="H050",
            checkpoint_passes=list(PASSES), arms=list(ARMS), VAL_parent_ids=list(VAL),
            complete_native_RAM_prediction_count=192, native_probability_bytes=self.bytes,
            dense_prediction_arrays_saved=False, all192_sealed_before_any_VAL_truth=True,
            seals=[self.records[k] for k in sorted(self.records)]))
        self.verified = True
        return self.barrier_ref


def practical_criteria_v1(primary_selected):
    require(isinstance(primary_selected, Mapping) and set(primary_selected) == set(ARMS),
        "Selected equal-parent Present182 metrics for all4arms")
    require(all(all(np.isfinite(row[k]) and 0 <= row[k] <= 1
        for k in ("accuracy","boundary_f1")) for row in primary_selected.values()),
        "Defined probability-scale Present182 accuracy and boundaryF1")
    true = primary_selected[MEMORY_ARMS[0]]
    contrasts = {}
    for control in ARMS[1:]:
        other = primary_selected[control]
        da = true["accuracy"] - other["accuracy"]
        df = true["boundary_f1"] - other["boundary_f1"]
        contrasts[control] = dict(accuracy_delta=da, boundary_F1_delta=df,
            primary=(da >= .001 and df >= 0), secondary=(df >= .01 and da >= -.0005))
    return dict(contrasts=contrasts,
        meets_primary=all(x["primary"] for x in contrasts.values()),
        meets_secondary_boundary=all(x["secondary"] for x in contrasts.values()),
        no_memory_control_parameter_count=31812, memory_parameter_count=32325,
        historical_H045_comparison="report_separately_not_a_matched_causal_contrast",
        reused8VAL_one_seed_development_evidence=True)
