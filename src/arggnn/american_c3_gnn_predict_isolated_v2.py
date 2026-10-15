"""Prediction sealing for original held-out C3 parents; accepts no truth.

Root supplies admitted loaders, transferred model, writer and fast resource
callback. Existing strict checkpoint loading and exact two-pass recurrent
inference are reused. This module owns no launch/accounting/data preparation.
"""
from __future__ import annotations
import torch
from .american_c3_common_v1 import require,read,bound,CLASSES
from .american_c3_evaluate_v1 import registry,verify_prediction
from .american_c3_gnn_targets_v1 import REGISTRY_REF,TRAIN,VAL,TEST,identity_sha
from .american_c3_gnn_stream_train_isolated_v2 import validate_plan_v1,require_baseline_v1
from .american_c3_gnn_train_v1 import CHECKPOINT_EPOCHS,SCHEMA,scheduled_checkpoint_metadata_v1
from .american_c3_gnn_run_isolated_v2 import streamed_focal_predictions
from .american_c3_gnn_dataset_isolated_v2 import shutdown_dataloader

def completed_training_v1(record):
    value=read(record)
    require(value.get("status")=="completed_C3_checkpointed_complete_parent_training"
        and value.get("schema")==SCHEMA and value.get("registry_ref")==REGISTRY_REF
        and value.get("trained_parent_ids")==list(TRAIN) and value.get("train_sequence_lengths_bp")==[2000000]*20
        and value.get("epochs")==30 and value.get("complete_parent_updates")==600
        and value.get("focal_exposures")==6000000 and value.get("VAL_used_for_optimization") is False
        and value.get("H043_extension") is None and value.get("purity_filter") is False
        and [x["epoch"] for x in value["checkpoint_candidates"]]==list(CHECKPOINT_EPOCHS)
        and all(x["global_step"]==x["epoch"]*20 for x in value["checkpoint_candidates"])
        and len({x["checkpoint_ref"]["path"] for x in value["checkpoint_candidates"]})==6,
        "Full primary training completion and all six frozen checkpoint refs required")
    return value

def eligible_checkpoint_v1(checkpoint_ref,training_completion_ref):
    training=completed_training_v1(training_completion_ref)
    candidates=[c for c in training["checkpoint_candidates"] if c["checkpoint_ref"]==checkpoint_ref]
    require(len(candidates)==1,"Checkpoint must be a frozen candidate in completed600-update primary run")
    value=torch.load(bound(checkpoint_ref),map_location="cpu",weights_only=True,mmap=True)
    metadata=scheduled_checkpoint_metadata_v1(value,protocol_sha256=training["protocol_sha256"])
    require(metadata["epoch"]==candidates[0]["epoch"]
        and metadata["observation_seal_sha256"]==training["observation_seal_sha256"]
        and metadata["target_join_sha256"]==training["target_join_sha256"],
        "Candidate binds this completed run and exact TRAIN data seals")
    return training,metadata

def seal_prediction_v1(model,forward_loader,reverse_loader,*,split,parent_id,checkpoint_ref,
        training_completion_ref,writer,device,runtime_guard,selected_checkpoint_ref=None):
    """Seal exact original NPZ identities/probabilities before any scoring.

    VAL may use each of the six frozen epochs. TEST is allowed only after a
    separately frozen original-five-VAL selection. No private truth capability,
    path or labels are arguments to this prediction callable.
    """
    require((split=="val" and parent_id in VAL) or (split=="test" and parent_id in TEST),
        "Only original5VAL or original10TEST; original30Mb TRAIN remains TRAIN")
    require(callable(runtime_guard) and callable(getattr(writer,"save",None)),"Root-supplied resource/writer capabilities")
    require_baseline_v1(model);runtime_guard("before_heldout_prediction")
    require(all(getattr(loader,"_iterator",None) is None for loader in (forward_loader,reverse_loader)),
        "Root creates both loaders without starting workers before prediction guards")
    training,metadata=eligible_checkpoint_v1(checkpoint_ref,training_completion_ref)
    if split=="test":
        require(selected_checkpoint_ref is not None,"TEST remains deferred until original5VAL selection is sealed")
        selected=read(selected_checkpoint_ref)
        require(selected.get("status")=="selected_C3_original5VAL_checkpoint"
            and selected.get("training_completion_ref")==training_completion_ref
            and selected.get("checkpoint_ref")==checkpoint_ref and selected.get("TEST_used_for_selection") is False
            and selected.get("validation_parent_ids")==list(VAL),
            "Exact separately sealed original5VAL checkpoint selection before TEST")
    else:require(selected_checkpoint_ref is None,"VAL candidates precede checkpoint selection")
    fd,rd=forward_loader.dataset,reverse_loader.dataset
    require(fd.plan_ref==rd.plan_ref and fd.plan["parent_id"]==parent_id
        and fd.direction=="forward" and rd.direction=="reverse","Same original held-out observation in opposite orders")
    validate_plan_v1(fd.plan,parent_id=parent_id,length_bp=30000000)
    require(parent_id not in training["trained_parent_ids"],"Independent held-out lineage")
    failure=None
    try:
        predictions,receipt=streamed_focal_predictions(model,forward_loader,reverse_loader,
            device=device,checkpoint_ref=checkpoint_ref)
        runtime_guard("before_prediction_seal")
        records=registry(fd.plan["focal_registry_ref"]);verify_prediction(predictions,records)
        prediction_ref=writer.save("%s_epoch%03d_predictions.npz"%(parent_id,metadata["epoch"]),predictions,kind="npz")
        # The old inference API returns checkpoint metadata including trainer
        # extras. Do not copy optimizer/RNG/history tensors into a JSON receipt.
        receipt.pop("training_metadata",None)
        receipt.update(status="sealed_C3_original_heldout_predictions",split=split,epoch=metadata["epoch"],
            global_step=metadata["global_step"],prediction_ref=prediction_ref,
            training_completion_ref=training_completion_ref,checkpoint_ref=checkpoint_ref,
            protocol_sha256=training["protocol_sha256"],focal_identity_sha256=identity_sha(records),
            records=10000,class_order=list(CLASSES),selected_checkpoint_ref=selected_checkpoint_ref,
            private_truth_opened=False,probabilities_modified=False,purity_filter=False)
        return writer.save("%s_epoch%03d_prediction_seal.json"%(parent_id,metadata["epoch"]),receipt)
    except BaseException as error:
        failure=error;raise
    finally:
        try:
            drains=[shutdown_dataloader(x) for x in (forward_loader,reverse_loader)]
            require(all(d["drained"] for d in drains),"Owned inference workers must drain on every exit")
        except BaseException as drain_error:
            if failure is None:raise
            if hasattr(failure,"add_note"):failure.add_note("Inference drain failed: "+str(drain_error)[:500])

def seal_validation_matrix_v1(prediction_seal_refs,*,training_completion_ref,writer):
    """Metadata-only30-cell barrier. Pass this ref to external scoring next."""
    training=completed_training_v1(training_completion_ref)
    require(isinstance(prediction_seal_refs,list) and len(prediction_seal_refs)==30,"Six epochs x original5VAL complete matrix")
    cells={}
    for record in prediction_seal_refs:
        seal=read(record);key=(seal["epoch"],seal["parent_id"])
        require(seal.get("status")=="sealed_C3_original_heldout_predictions"
            and seal.get("split")=="val" and key[0] in CHECKPOINT_EPOCHS and key[1] in VAL
            and key not in cells and seal.get("training_completion_ref")==training_completion_ref
            and seal.get("protocol_sha256")==training["protocol_sha256"]
            and seal.get("private_truth_opened") is False and seal.get("records")==10000
            and seal.get("purity_filter") is False,
            "Each original VAL cell sealed independently without private truth")
        candidate=next(c for c in training["checkpoint_candidates"] if c["epoch"]==key[0])
        require(seal["checkpoint_ref"]==candidate["checkpoint_ref"],"Exact frozen epoch checkpoint")
        # bind the sealed prediction and common registry now, before resolver
        # access in the separate external-score callable.
        bound(seal["prediction_ref"]);bound(seal["focal_registry_ref"])
        cells[key]=dict(epoch=key[0],parent_id=key[1],prediction_seal_ref=record,
            prediction_ref=seal["prediction_ref"],focal_registry_ref=seal["focal_registry_ref"],
            focal_identity_sha256=seal["focal_identity_sha256"],checkpoint_ref=seal["checkpoint_ref"])
    require(set(cells)=={(e,p) for e in CHECKPOINT_EPOCHS for p in VAL},"No missing/extra VAL cells")
    return writer.save("VAL_all30_predictions_sealed.json",dict(status="sealed_C3_all30_original_VAL_predictions",
        training_completion_ref=training_completion_ref,protocol_sha256=training["protocol_sha256"],
        validation_parent_ids=list(VAL),checkpoint_epochs=list(CHECKPOINT_EPOCHS),cells=[cells[k] for k in sorted(cells)],
        all_predictions_sealed_before_VAL_truth=True,TEST_deferred=True,metrics="original_C3_external_evaluator_unchanged"))
