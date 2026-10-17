"""Inactive future VAL000-002 predictor, using the existing VAL003/004 interleave.

ROOT must bind this callable and its sibling scan in a separately reviewed fresh
worker/source closure. This file supplies no launch, authority, accounting,
lifecycle, matrix, truth resolver or scorer. Scientific imports belong after the
guarded CPU worker entry, exactly as for the existing interleaved bridge."""
from __future__ import annotations
import torch
from arggnn.american_c3_common_v1 import require
from arggnn.american_c3_evaluate_v1 import registry, verify_prediction
from arggnn.american_c3_gnn_targets_v1 import identity_sha
from arggnn.american_c3_gnn_model_v1 import AmericanC3WholeTreeGNNV1
from arggnn.american_c3_gnn_dataset_isolated_v2 import (
    AmericanC3WholeTreeDatasetV1, make_dataloader)
from arggnn.american_c3_gnn_shared30mb_isolated_v2 import (
    MODEL_KWARGS, DEVELOPMENT_BRANCH, validate_shared_plan_v1,
    validate_shared_metadata_v1)
from arggnn.american_c3_gnn_predict_isolated_v2 import (
    completed_training_v1, eligible_checkpoint_v1)
from arggnn.american_c3_gnn_run_isolated_v2 import load_c3_checkpoint
from research.reviews.C3_interleaved_prediction_successor_source_20261017_v1.six_candidate_scan_v1 import interleaved_six_focal_predictions_v1
from arggnn.american_c3_gnn_train_v1 import scheduled_checkpoint_metadata_v1


def new_candidate_models_v1():
    # Original producer constructed one discarded random initialization before
    # strict loading. Retain that exact advancement, and isolate only the five
    # additional constructor draws introduced by this scheduling optimization.
    with torch.device("cpu"):
        first = AmericanC3WholeTreeGNNV1(**MODEL_KWARGS)
        with torch.random.fork_rng(devices=[]):
            extra = [AmericanC3WholeTreeGNNV1(**MODEL_KWARGS) for _ in range(5)]
    return [first, *extra]


def predict_v1(req, training_bundle, training, writer, fast, *, contract):
    """Original six-candidate VAL output support, with observed features reused."""
    require(req["phase"] in tuple("predict_val_%03d" % i for i in range(3)),
            "Only future original VAL000-002 successors; TEST remains deferred")
    require(tuple(contract.VAL) == tuple("american_c3_val_%03d" % i for i in range(5)),
            "Original five-parent contract order required")
    parent = contract.VAL[int(req["phase"][-3:])]
    plan_ref = req["plan_evidence"]["plan_ref"]
    plan = contract.read(plan_ref)
    validate_shared_plan_v1(plan, parent_id=parent)
    validate_shared_metadata_v1(plan["config"], parent_id=parent)
    training_ref = training_bundle["training_completion_ref"]
    require(completed_training_v1(training_ref) == training,
            "Same immutable six-candidate 600-update training family")
    require(parent not in training["trained_parent_ids"]
            and training["train_sequence_lengths_bp"] == [2000000] * 20,
            "Same disclosed 20x2Mb TRAIN to original30Mb VAL transfer")
    candidates = training["checkpoint_candidates"]
    require(len(candidates) == 6
            and [c["epoch"] for c in candidates] == [5, 10, 15, 20, 25, 30],
            "Every original epoch, in the original candidate order")
    # Strict loading replaces every random parameter before inference.
    models = new_candidate_models_v1()
    metadata = []
    for candidate, model in zip(candidates, models):
        fast("before_original_six_candidate_cell")
        same, checked = eligible_checkpoint_v1(candidate["checkpoint_ref"], training_ref)
        require(same == training, "Each candidate has unchanged original provenance")
        loaded = load_c3_checkpoint(candidate["checkpoint_ref"], model)
        # The original loader preserves nested trainer_state/candidate metadata.
        # Use the original strict scheduled-checkpoint validator, as the older
        # successful loading route does, rather than assuming flattened fields.
        loaded_metadata = scheduled_checkpoint_metadata_v1(
            loaded, protocol_sha256=training["protocol_sha256"])
        require(loaded_metadata == checked
                and loaded_metadata["epoch"] == candidate["epoch"]
                and loaded_metadata["global_step"] == candidate["global_step"],
                "Strict all-parameter loading of the exact scheduled candidate")
        metadata.append(checked)
    records = registry(plan["focal_registry_ref"])
    require(len(records["record_ids"]) == 10000
            and plan["haplotypes"] == 1600
            and plan["sequence_length_bp"] == 30000000,
            "Original full panel, length and focal support")
    datasets = [AmericanC3WholeTreeDatasetV1(
        plan_ref, direction=direction, worker_tmp_directory=writer.tmp)
        for direction in ("forward", "reverse")]
    loaders = [make_dataloader(dataset, num_workers=0, pin_memory=False)
               for dataset in datasets]
    try:
        results = interleaved_six_focal_predictions_v1(
            models, *loaders, records=records, runtime_guard=fast)
    finally:
        for dataset in datasets:
            dataset._state = None
            dataset._pid = None
    seals = []
    for candidate, checked, (prediction, receipt) in zip(candidates, metadata, results):
        fast("before_prediction_seal")
        verify_prediction(prediction, records)
        epoch = checked["epoch"]
        prediction_ref = writer.save(
            "%s_epoch%03d_predictions.npz" % (parent, epoch), prediction, kind="npz")
        receipt.update(
            status="sealed_C3_original_heldout_predictions", split="val", epoch=epoch,
            global_step=checked["global_step"], prediction_ref=prediction_ref,
            training_completion_ref=training_ref, checkpoint_ref=candidate["checkpoint_ref"],
            protocol_sha256=training["protocol_sha256"],
            focal_identity_sha256=identity_sha(records), records=10000,
            class_order=list(models[0].class_order), selected_checkpoint_ref=None,
            private_truth_opened=False, probabilities_modified=False, purity_filter=False,
            evaluation_length_bp=30000000, training_condition_differs_from_paper_native=True,
            training_branch=DEVELOPMENT_BRANCH,
            trained_parent_ids=list(training["trained_parent_ids"]),
            train_sequence_lengths_bp=[2000000]*20, training_2Mb_to_30Mb_length_transfer=True,
            original_full30Mb_matched_training_claimed=False,
            same_observation_and_native_focals=True,
            original_C3_scorer="src/arggnn/american_c3_evaluate_v1.py",
            complete_original_5VAL_selection_required_before_TEST=True,
            CPU_observation_scheduling="six_candidate_block_interleaving_v1",
            checkpoint_independent_features_shared=True,
            learned_contexts_shared=False, per_candidate_tree_encoding_passes=2)
        seals.append(writer.save("%s_epoch%03d_prediction_seal.json" % (parent, epoch), receipt))
    return writer.save(parent + "_all6_prediction_seals.json", dict(
        status="sealed_C3_one_original_VAL_all6_candidates", parent_id=parent,
        training_completion_ref=training_ref, prediction_seal_refs=seals,
        common_focal_records=10000, private_truth_opened=False, TEST_deferred=True,
        complete_original5VAL_barrier_still_required=True, exact_CPU_FP32_source_math=True,
        architecture="AmericanC3WholeTreeGNNV1", full_TreeLoom_history_pair_model_claimed=False,
        all1600_graph_and_recurrent_columns_preserved=True,
        no_context_cache_or_protocol_reduction=True))
