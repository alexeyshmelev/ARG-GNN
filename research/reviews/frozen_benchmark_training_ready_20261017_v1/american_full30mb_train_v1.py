"""Inactive frozen-architecture American30Mb training callable.

ROOT must independently review and bind this callable to its existing guarded
worker, authenticated complete-parent observations, TRAIN-only target joiner,
writer, device and genuine OS wait. This file has no admission or launch path.
The archived H060 invocation is not resumed, edited or replaced.
"""
from __future__ import annotations
import copy
import random
from collections.abc import Mapping

TRAIN = tuple("american_c3_train_%03d" % i for i in range(20))
VAL = tuple("american_c3_val_%03d" % i for i in range(5))
PASSES = (5, 10, 15, 20, 25, 30)
CONFIG = dict(
    schema="TreeLoom_frozen_H050_American_full30Mb_training_v1",
    execution_enabled=False, class_order=["EUR", "EAS", "AFR"],
    TRAIN_parent_ids=list(TRAIN), VAL_parent_ids=list(VAL),
    region_length_bp=30000000, haplotypes=1600, queries=1000,
    focals_per_parent=10000, epochs=30, updates_per_epoch=20,
    total_optimizer_updates=600, checkpoint_epochs=list(PASSES),
    model_seed=70001, parent_order_seed=70002, precision="fp32",
    initialization="fresh_C3_full_TreeLoom_no_C4_or_short2Mb_weights",
    optimizer=dict(name="AdamW", lr=.001, weight_decay=.0001,
                   betas=[.9, .999], eps=1e-8),
    gradient_clip_norm=1., memory_arm="stable_leaf_memory",
    early_stopping=False, purity_filter=False,
    validation_used_for_optimization=False, TEST_authorized=False,
    checkpoint_selection="fixed_final_epoch30_before_scored_benchmark_truth",
    scored_benchmark_parent_ids=list(VAL), benchmark_checkpoint_tuning=False,
    focal_exposures=6000000, physical_geometry_normalizer=2000001.,
)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def epoch_order_v1(epoch):
    need(type(epoch) is int and 1 <= epoch <= 30, "Fixed one-based epoch")
    rows = list(TRAIN)
    random.Random(70002 + epoch).shuffle(rows)
    return rows


def run_train_v1(config, *, device, observation_barrier,
                 load_parent, join_targets, checkpoint, event, resource_check):
    """Train exactly600 complete30Mb parents under ROOT-owned callbacks.

    observation_barrier is an externally authenticated immutable JSON value.
    load_parent(p) -> (new full30Mb bundle, source-bound parent seal).
    join_targets(p, bundle, seal) -> original TrainFocalTargetVectorV1.
    checkpoint(epoch, payload) -> durable ROOT writer path/SHA256 ref.
    event(name, metadata) uses ROOT's bounded writer. No direct I/O occurs here.

    The first update belongs to this fixed600-update condition. No short trial,
    extra qualification update, VAL target access, retry or fallback is present.
    """
    need(isinstance(config, Mapping) and set(config) == set(CONFIG)
         and config["execution_enabled"] is True
         and all(config[k] == v for k, v in CONFIG.items()
                 if k != "execution_enabled"), "Exact frozen inactive config activated only by ROOT")
    need(all(callable(f) for f in
             (load_parent, join_targets, checkpoint, event, resource_check)),
         "Actual admitted ROOT callbacks are mandatory")
    need(isinstance(observation_barrier, Mapping)
         and observation_barrier.get("schema") == "frozen_American_full30Mb_all20_observation_barrier_v1"
         and observation_barrier.get("TRAIN_parent_ids") == list(TRAIN)
         and observation_barrier.get("region_length_bp") == 30000000
         and observation_barrier.get("all_native_trees_nodes_sites_samples_retained") is True
         and observation_barrier.get("all20_observations_sealed_before_any_TRAIN_truth") is True
         and observation_barrier.get("private_truth_opened") is False
         and set(observation_barrier.get("parents", {})) == set(TRAIN),
         "Genuine externally authenticated all20 complete30Mb observation barrier required")
    # Imports and all numerical work are future admitted operations only.
    import torch
    from arggnn.american_c3_gnn_targets_v1 import TrainFocalTargetVectorV1
    from arggnn.american_c3_gnn_train_v1 import _cpu_snapshot
    from . import american_full30mb_frozen_v1 as frozen
    resource_check("before_frozen_full30Mb_model")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(70001)
        model = frozen.AmericanTreeLoomC3FullV1()
    model = model.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=.001, weight_decay=.0001,
        betas=(.9, .999), eps=1e-8, maximize=False, amsgrad=False,
        foreach=False, fused=False, capturable=False, differentiable=False)
    need(not optimizer.state, "Fresh independent optimizer state")
    refs, steps = {}, 0
    event("frozen_full30Mb_training_started",
          dict(config=dict(config), architecture=model.checkpoint_contract(),
               ancestry_accuracy=None, observation_barrier=copy.deepcopy(observation_barrier)))
    for epoch in range(1, 31):
        for parent_id in epoch_order_v1(epoch):
            resource_check("before_complete30Mb_parent_load")
            bundle, seal = load_parent(parent_id)
            need(type(bundle) is frozen.AmericanTreeLoomWholeParentV1
                 and bundle.parent_id == parent_id and bundle.split == "train"
                 and bundle.sequence_length_bp == 30000000
                 and isinstance(seal, Mapping)
                 and dict(seal) == observation_barrier["parents"][parent_id]
                 and seal.get("observation_only") is True
                 and seal.get("all_native_trees_nodes_sites_samples_retained") is True
                 and seal.get("observation_ref") == bundle.observation_ref
                 and seal.get("focal_registry_ref") == bundle.focal_registry_ref
                 and seal.get("native_preparation_ref") == bundle.native_preparation_ref
                 and seal.get("focal_identity_sha256") == bundle.focal_identity_sha256,
                 "Exact complete30Mb observation and original focal identity barrier")
            bundle.validate()
            targets = join_targets(parent_id, bundle, seal)
            need(type(targets) is TrainFocalTargetVectorV1
                 and targets.parent_id == parent_id
                 and targets.labels_cpu.device.type == "cpu"
                 and targets.labels_cpu.dtype == torch.long
                 and targets.labels_cpu.shape == (10000,)
                 and bool(((targets.labels_cpu >= 0) & (targets.labels_cpu < 3)).all())
                 and targets.receipt.get("split") == "train"
                 and targets.receipt.get("labels_in_observation_bank") is False,
                 "TRAIN-only full native focal targets after observation barrier")
            bundle = bundle.to(device)
            result = frozen.train_complete_parent_step_v1(
                model, bundle, targets, optimizer, resource_check=resource_check)
            steps += 1
            counts = [int(v["step"].item()) for v in optimizer.state.values()]
            need(len(counts) == 49 and set(counts) == {steps},
                 "Every frozen-model tensor has the genuine scheduled AdamW step count")
            event("complete30Mb_parent_update",
                  dict(epoch=epoch, optimizer_updates=steps, **result))
            del bundle, targets
        if epoch in PASSES:
            resource_check("before_frozen_full30Mb_checkpoint")
            payload = dict(
                schema="TreeLoom_frozen_H050_American_full30Mb_checkpoint_v1",
                epoch=epoch, complete_parent_updates=steps, seed=70001,
                config=copy.deepcopy(dict(config)),
                observation_barrier=copy.deepcopy(observation_barrier),
                contract=model.checkpoint_contract(),
                model_state=_cpu_snapshot(model.state_dict()),
                optimizer_state=_cpu_snapshot(optimizer.state_dict()),
                VAL_truth_read=False, TEST_truth_read=False,
                ancestry_accuracy=None)
            ref = checkpoint(epoch, payload)
            need(isinstance(ref, Mapping) and set(ref) == {"path", "sha256"}
                 and isinstance(ref["path"], str) and len(ref["sha256"]) == 64,
                 "Actual durable ROOT checkpoint ref")
            refs[str(epoch)] = dict(ref)
            del payload
            resource_check("after_frozen_full30Mb_checkpoint")
    need(steps == 600 and set(refs) == {str(p) for p in PASSES},
         "Fixed30 epochs, all20 full parents, all six checkpoints")
    return dict(
        schema="TreeLoom_frozen_H050_American_full30Mb_training_completion_v1",
        status="completed_scientific_callable_pending_genuine_ROOT_wait",
        config=copy.deepcopy(dict(config)), complete_parent_updates=steps,
        checkpoint_refs=refs, VAL_truth_read=False, TEST_truth_read=False,
        ancestry_accuracy=None, root_admission_or_outer_wait_claimed=False)
