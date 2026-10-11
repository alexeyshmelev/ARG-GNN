"""H036 source-only exact continuation: epoch10/20updates -> epoch80/160updates.

Fresh root continuation authorization is mandatory. No restart, new seed,
new arm, altered model, detached history, teacher, or changed scoring.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import io
import json
from pathlib import Path
import random
import time

from . import census_tsinfer_fit_v1 as fit
from . import train_mutation_routed_whole_tree_gnn_v1 as initial
from . import train_persistence_trace_pilot_v1 as h025

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "src/arggnn/continue_mutation_routed_whole_tree_gnn_v1.py"
CONFIG = "configs/mutation_routed_whole_tree_gnn_continue_v1.json"
EFFICIENT = "src/arggnn/mutation_routed_observation_efficiency_v1.py"
ARMS = initial.ARMS


def inputs_v1(record):
    from .native_argformer_exec import validate_lease, apply_and_validate_title
    lease = validate_lease(); apply_and_validate_title()
    reader = fit.ConsumerReaderV1()
    spec = reader.decode_json(record)
    if (spec.get("schema") != "mutation_routed_whole_tree_gnn_continue_input_v1"
            or spec.get("root_authorizer") != "root_agent" or spec.get("fit_authorized") is not True
            or spec.get("continuation_authorized") is not True or spec.get("status") != "active"
            or spec.get("arm") not in ARMS or spec.get("seed") != 101
            or spec.get("device") != "cuda" or spec.get("epochs", 70) != 70
            or not fit.canonical_job_v1(spec.get("job_id"))):
        raise ValueError("Fresh root decision/authorization for exact H036 continuation required")
    fit._check_active_v1(spec); fit.require_logical_job_v1(lease, spec)
    refs = spec["refs"]
    cfg = reader.decode_json(refs["config"])
    old_cfg = reader.decode_json(cfg["initial_fit_config_ref"])
    if (refs["config"]["path"] != CONFIG or cfg.get("hypothesis_id") != "H036"
            or cfg["kind"] != "mutation_routed_whole_tree_gnn_continue_v1"
            or cfg["arms"] != list(ARMS) or cfg["seeds"] != [101] or cfg["dtype"] != "float32"
            or cfg["initial_epochs"] != 10 or cfg["additional_epochs"] != 70
            or cfg["initial_updates_per_arm"] != 20 or cfg["additional_updates_per_arm"] != 140
            or cfg["total_updates_per_arm"] != 160 or cfg["output_cap_bytes"] != 750000000
            or cfg["evaluation_epochs"] != list(range(12,81,4))
            or cfg.get("execution_guard_ready") is not True
            or any(cfg[k] != old_cfg[k] for k in (
                "model", "parameter_count", "metrics", "selection", "learning_rate",
                "weight_decay", "gradient_clip_norm", "prototype_ref", "preprocessor_ref"))):
        raise ValueError("Exact frozen model/data/loss/scoring and initial+140update protocol required")
    if cfg["initial_fit_config_ref"]["path"] != initial.CONFIG:
        raise ValueError("Continuation must bind the original initial-fit config")
    for name in ("max_cpu_rss_bytes", "max_cuda_reserved_bytes"):
        if type(spec.get(name)) is not int or spec[name] <= 0:
            raise ValueError("Positive root CPU/CUDA bounds required")
    for path in (SOURCE, initial.SOURCE, initial.MODULE, initial.PREPROCESSOR, initial.LOADER, EFFICIENT):
        if spec["source_sha256"].get(path) != fit._hash(path):
            raise ValueError("Complete original and continuation source closure required")
    for path, digest in spec["source_sha256"].items():
        reader.read({"path":path, "sha256":digest})
    for ref in cfg["helper_refs"].values():
        reader.read(ref)
    for ref in (cfg["initial_driver_ref"], cfg["prototype_ref"], cfg["preprocessor_ref"], cfg["efficiency_ref"]):
        reader.read(ref)
    reader.read(refs["runtime"])
    prior = reader.decode_json(refs["initial_completed"])
    prior_input = reader.decode_json(refs["initial_input"])
    if (prior.get("status") != "completed_one_seed_FP32_direct_full_parent_initial20updates"
            or prior.get("initial_budget_complete") is not True
            or prior.get("arm") != spec["arm"] or prior.get("seed") != 101
            or prior.get("optimizer_steps_per_arm") != 20 or prior.get("epochs") != 10
            or prior.get("dtype") != "float32" or prior.get("teacher_used") is not False
            or prior["input_ref"] != refs["initial_input"]
            or prior_input["schema"] != "mutation_routed_whole_tree_gnn_input_v1"
            or prior_input["arm"] != spec["arm"] or prior_input["seed"] != 101
            or prior_input["refs"]["config"] != cfg["initial_fit_config_ref"]
            or prior_input["observed_parents"] != spec["observed_parents"]
            or prior_input["refs"]["target_bindings"] != refs["target_bindings"]):
        raise ValueError("Completed same-arm initial20-update data/seed/provenance required")
    for path, old_hash in prior_input["source_sha256"].items():
        reader.read({"path":path,"sha256":old_hash})
    model_record = prior["models"][spec["arm"]]
    history = model_record["history"]
    if ([r["epoch"] for r in history] != list(range(1,11))
            or [r["optimizer_steps"] for r in history] != list(range(2,21,2))
            or model_record["steps"] != 20 or history[-1]["checkpoint"] != refs["resume_checkpoint"]):
        raise ValueError("Resume exactly the LAST epoch10 checkpoint, never a selected earlier epoch")
    if set(spec["observed_parents"]) != set(fit.PARENTS):
        raise ValueError("Exactly original four TRAIN/VAL parents required")
    base = reader.decode_json(cfg["base_fit_config"])
    bindings = reader.decode_json(refs["target_bindings"])
    manifest = reader.decode_json(base["external_target_manifest"])
    rows = {r["parent_id"]:r for r in manifest["replicates"]}
    if set(bindings["parents"]) != set(fit.PARENTS):
        raise ValueError("Original target bindings required")
    for parent_id, binding in bindings["parents"].items():
        expected = {"path":f"data/biological_paths_execution_pilot_v1/{parent_id}/truth/full_truth_intervals.npz",
                    "sha256":rows[parent_id]["files"]["truth/full_truth_intervals.npz"]["sha256"]}
        if binding["truth_npz"] != expected or binding["split"] != rows[parent_id]["split"]:
            raise ValueError("Original external target binding changed")
    storage = reader.decode_json(refs["storage"])
    from .project_storage_full_reservation_v1 import validate_storage_v9
    if cfg["helper_refs"]["actual_full_storage_launcher"] not in storage["reservation"]["consumer_bindings"]:
        raise ValueError("Actual full-storage launcher must be reserved")
    validate_storage_v9(spec,cfg,storage,source_ref={"path":SOURCE,"sha256":spec["source_sha256"][SOURCE]},
                        config_ref=refs["config"],runtime_ref=refs["runtime"])
    initial.science_v1()
    return spec,cfg,base,bindings,reader,storage,prior


def recursive_equal_v1(a,b):
    """Exact checkpoint-state equality, ignoring only tensor storage device."""
    torch = fit.torch
    if isinstance(a,torch.Tensor) or isinstance(b,torch.Tensor):
        return (isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor)
                and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.detach().cpu(),b.detach().cpu()))
    if isinstance(a,dict) and isinstance(b,dict):
        return a.keys()==b.keys() and all(recursive_equal_v1(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)) and isinstance(b,(tuple,list)):
        return type(a)==type(b) and len(a)==len(b) and all(recursive_equal_v1(x,y) for x,y in zip(a,b))
    return type(a)==type(b) and a==b


def restore_v1(spec,cfg,prior,reader,device):
    torch = fit.torch
    saved = torch.load(io.BytesIO(reader.read(spec["refs"]["resume_checkpoint"])),
                       map_location="cpu",weights_only=True)
    if (saved["arm"] != spec["arm"] or saved["seed"] != 101
            or saved["epoch"] != 10 or saved["optimizer_steps"] != 20
            or saved["teacher_used"] is not False or saved["input_ref"] != spec["refs"]["initial_input"]
            or saved["initial_state_sha256"] != prior["initial_state_sha256"]):
        raise ValueError("Exact same-arm last epoch10 model/optimizer checkpoint required")
    # The original factory is used only to instantiate unchanged modules/AdamW.
    # All trained parameters and every optimizer state are then overwritten.
    state = initial.model_state_v1(cfg,device)
    if state["initial_sha256"] != prior["initial_state_sha256"]:
        raise ValueError("Original seed101 initialization fingerprint changed")
    state["model"].load_state_dict(saved["model_state"],strict=True)
    state["optimizer"].load_state_dict(saved["optimizer_state"])
    if not recursive_equal_v1(saved["model_state"],state["model"].state_dict()):
        raise ValueError("Restored model bytes differ")
    if not recursive_equal_v1(saved["optimizer_state"],state["optimizer"].state_dict()):
        raise ValueError("Restored AdamW moments/counters/groups differ")
    history = deepcopy(prior["models"][spec["arm"]]["history"])
    earliest = min(history,key=lambda row:initial.selection_nll_v1(row["equal_parent_metrics"]))
    selected = deepcopy(prior["models"][spec["arm"]]["selected"])
    if (selected["epoch"] != earliest["epoch"] or selected["checkpoint"] != earliest["checkpoint"]
            or initial.selection_nll_v1(selected["equal_parent_metrics"]) != initial.selection_nll_v1(earliest["equal_parent_metrics"])):
        raise ValueError("Initial selection was not the strict earliest minimum NLL")
    external_best = [selected["checkpoint"]]+[row["prediction_seal"] for row in selected["cells"]]
    retained = prior["models"][spec["arm"]]["retained_records"]
    if any(retained.get(r["path"]) != r for r in external_best):
        raise ValueError("Selected initial checkpoint/probability seals must remain retained")
    # Imported records are references only. Never pass them to the new writer's
    # discard(), never copy them into new output, and never delete old results.
    state.update(steps=20,best=initial.selection_nll_v1(selected["equal_parent_metrics"]),
        best_epoch=selected["epoch"],selected=selected,history=history,
        current=[],retained_best=[],external_initial_best=external_best,TRAIN_epoch_logs=[])
    return state


def prepare_v1(spec,bindings,reader,output,helpers):
    from .mutation_routed_observation_efficiency_v1 import prepare_parents_v1
    from .mutation_routed_whole_tree_gnn_v1 import validate_inputs_v1
    bundle = prepare_parents_v1(spec["observed_parents"],reader,
                              scratch_directory=output.path/"observation_tmp")
    items=bundle["parents"]
    if set(items)!=set(fit.PARENTS):
        raise ValueError("Exact four-parent observation collection required")
    banks,parents,receipts,train_ids,val_ids={},{},{},[],[]
    for parent_id in fit.PARENTS:
        item=items[parent_id];bank,parent=item["model_inputs"],item["scoring"]
        validate_inputs_v1(bank)
        binding=bindings["parents"][parent_id]
        if (parent["parent_id"]!=parent_id or parent["split"]!=binding["split"]
                or parent["query_slots"].tolist()!=binding["query_sample_indexes"]
                or parent["raw_query_nodes"].tolist()!=binding["query_raw_node_ids"]
                or not fit.np.array_equal(bank.query_column.numpy(),parent["query_slots"])
                or len(bank.readout_hold)!=len(parent["positions"])
                or not fit.np.array_equal(parent["positions"],50.+100.*parent["retained_indexes"])
                or parent["positions"].shape!=(len(parent["edges"])-1,)):
            raise ValueError("Original query/physical scoring support changed")
        banks[parent_id],parents[parent_id],receipts[parent_id]=bank,parent,item["observations_receipt"]
        (train_ids if parent["split"]=="train" else val_ids).append(parent_id)
    if len(train_ids)!=2 or len(val_ids)!=2:
        raise ValueError("Original two TRAIN/two VAL split required")
    seal=output.save("observations_features_ready.json",dict(
        schema="mutation_routed_continue_all_parent_observation_seal_v1",
        all_four_parents_complete=True,targets_opened=False,teacher_used=False,
        parents=receipts,loader_receipt=bundle["receipt"],sham_bin_edges=bundle["sham_bin_edges"],
        efficiency_provenance=bundle["efficiency_provenance"],
        full_parent_history_detached=False,original_native_holds_preserved=True))
    targets={}
    for parent_id in train_ids:
        target=fit.ExternalTargetV1(parents[parent_id],bindings["parents"][parent_id],"TRAIN_loss",reader)
        targets[parent_id]=target.target_mass(parents[parent_id],helpers)
    output.save("TRAIN_targets_ready.json",dict(observations_features_seal=seal,
        target_purpose="TRAIN_loss_only",query_truth_features=False,
        parents={p:{"target_ref":bindings["parents"][p]["truth_npz"],"shape":list(targets[p].shape)} for p in train_ids}))
    return banks,parents,train_ids,val_ids,targets,seal


def run_v1(record):
    spec,cfg,base,bindings,reader,storage,prior=inputs_v1(record)
    from .biological_candidate import configure_fit
    from .project_storage_full_reservation_v1 import PilotOutputV9,budget_check_v9
    torch,np=fit.torch,fit.np
    cfg=dict(cfg,_arm=spec["arm"]);arm=spec["arm"]
    output=PilotOutputV9(spec["output_directory"],storage,reader)
    budget_check_v9(storage,boundary=True,required_write_bytes=32000000)
    device,runtime,lease=configure_fit("cuda",base,output.path)
    output.used=sum(max(p.stat().st_size,p.stat().st_blocks*512)+8192
                    for p in output.path.rglob("*") if p.is_file())
    if output.used>32000000 or output.used>output.limit:
        raise ValueError("Runtime qualification exceeded own allowance")
    helpers=fit.metric_helpers_v1()
    receipt=dict(schema="mutation_routed_whole_tree_gnn_continuation_evidence_v1",
        hypothesis_id="H036",status="running",arm=arm,seed=101,dtype="float32",
        input_ref=record,config_ref=spec["refs"]["config"],runtime=runtime,
        initial_completed_ref=spec["refs"]["initial_completed"],
        resume_checkpoint=spec["refs"]["resume_checkpoint"],initial_updates=20,
        additional_updates=140,total_updates=160,initial_epoch=10,final_epoch=80,
        evaluation_epochs=cfg["evaluation_epochs"],
        selection_history_scope="all_initial10_epochs_and_scheduled_continued_epochs_only",
        exact_optimizer_resume=True,restarted=False,new_arm=False,
        teacher_used=False,whole_parent_history_detached=False,artificial_tile_reset=False,
        checkpoint_selection_across_combined_history=initial.SELECTION,
        source_sha256=spec["source_sha256"],novelty_claim=False,development_only=True,TEST_parents=0,
        comparison_limitation=cfg["comparison_limitation"],models={})
    output.save("started.json",receipt);started=time.monotonic()
    state=None;last_durable_epoch=10;last_durable_checkpoint=spec["refs"]["resume_checkpoint"]
    try:
        initial.resources_v1(spec,storage,lease)
        banks,parents,train_ids,val_ids,targets,observation_seal=prepare_v1(
            spec,bindings,reader,output,helpers)
        state=restore_v1(spec,cfg,prior,reader,device)
        receipt["initial_state_sha256"]=state["initial_sha256"]
        receipt["observation_features_seal"]=observation_seal
        output.save("resume_verified.json",dict(
            resume_checkpoint=spec["refs"]["resume_checkpoint"],
            initial_state_sha256=state["initial_sha256"],restored_steps=20,restored_epoch=10,
            model_state_exact=True,optimizer_state_exact=True,
            combined_selection_starts_with_initial_selected_epoch=state["best_epoch"],
            external_initial_best=state["external_initial_best"]))
        for epoch in range(11,81):
            epoch_start=time.monotonic()
            if epoch in cfg["evaluation_epochs"]:
                initial.prune_nonbest_probability_seals_v1(output,state)
            initial.resources_v1(spec,storage,lease)
            order=list(train_ids);random.Random(101+epoch).shuffle(order)
            losses=[]
            for parent_id in order:
                initial.lightweight_resources_v1(spec,lease)
                bank=banks[parent_id].to(device)
                row=initial.train_parent_v1(state,bank,targets[parent_id],cfg,device,qualifying=False)
                losses.append(row["TRAIN_exact_span_query120_NLL"])
                del bank
                initial.lightweight_resources_v1(spec,lease)
            train_row=dict(epoch=epoch,arm=arm,total_optimizer_steps=state["steps"],
                additional_optimizer_steps=state["steps"]-20,
                TRAIN_mean_full_parent_query120_NLL=float(np.mean(losses)),
                evaluation_due=epoch in cfg["evaluation_epochs"],
                last_durable_epoch_before_evaluation=last_durable_epoch,
                last_durable_checkpoint_before_evaluation=last_durable_checkpoint,
                updates_since_last_durable_checkpoint=state["steps"]-2*last_durable_epoch,
                elapsed_TRAIN_seconds=time.monotonic()-epoch_start)
            train_record=output.save(f"{arm}.epoch_{epoch:02d}.TRAIN_updates.json",train_row)
            state["TRAIN_epoch_logs"].append(train_record)
            if epoch not in cfg["evaluation_epochs"]:
                initial.resources_v1(spec,storage,lease)
                print(json.dumps(dict(stage="H036_continuation_TRAIN_epoch",
                    VAL_truth_opened_this_epoch=False,checkpoint_saved_this_epoch=False,**train_row)),flush=True)
                continue
            seals=[]
            for parent_id in val_ids:
                initial.lightweight_resources_v1(spec,lease)
                bank=banks[parent_id].to(device);state["model"].eval()
                with torch.no_grad():
                    logits=state["model"](bank,control=arm)
                    seal=initial.seal_probabilities_v1(output,arm,epoch,parents[parent_id],logits)
                seals.append((parent_id,seal));del bank,logits
            manifest=output.save(f"{arm}.epoch_{epoch:02d}.VAL_predictions_sealed.json",dict(
                epoch=epoch,arm=arm,all_VAL_predictions_sealed_before_scoring=True,
                targets_opened_in_this_prediction_phase=False,seals=dict(seals)))
            cells,means=h025.score_v1(seals,parents,bindings,reader,helpers,cfg)
            nll=initial.selection_nll_v1(means)
            previous={r["path"]:r for r in state["current"]+state["retained_best"]}
            checkpoint=output.save(f"{arm}.epoch_{epoch:02d}.pt",dict(
                model_state={k:v.detach().cpu() for k,v in state["model"].state_dict().items()},
                optimizer_state=state["optimizer"].state_dict(),epoch=epoch,arm=arm,
                optimizer_steps=state["steps"],seed=101,input_ref=record,
                original_input_ref=spec["refs"]["initial_input"],
                initial_state_sha256=state["initial_sha256"],teacher_used=False,
                initial_completed_ref=spec["refs"]["initial_completed"]), "torch")
            last_durable_epoch,last_durable_checkpoint=epoch,checkpoint
            state["current"]=[r for _,r in seals]+[checkpoint]
            row=dict(epoch=epoch,arm=arm,optimizer_steps=state["steps"],
                additional_optimizer_steps=state["steps"]-20,
                TRAIN_mean_full_parent_query120_NLL=float(np.mean(losses)),
                cells=cells,equal_parent_metrics=means,checkpoint=checkpoint,
                prediction_manifest=manifest)
            state["history"].append(row)
            if nll<state["best"]:
                state.update(best=nll,best_epoch=epoch,selected=row,
                             retained_best=list(state["current"]))
            row["selected_epoch"]=state["best_epoch"]
            output.save(f"{arm}.epoch_{epoch:02d}.metrics.json",row)
            keep={r["path"] for r in state["current"]+state["retained_best"]}
            for path,product in previous.items():
                if path not in keep:output.discard(product)
            initial.resources_v1(spec,storage,lease)
            print(json.dumps(dict(stage="H036_continuation_epoch",arm=arm,epoch=epoch,
                total_optimizer_steps=state["steps"],additional_optimizer_steps=state["steps"]-20,
                elapsed_seconds=time.monotonic()-epoch_start,
                selected_epoch_across_complete_history=state["best_epoch"],
                primary_present64=means["primary_present_day_64"])),flush=True)
        if (state["steps"]!=160 or [r["epoch"] for r in state["history"]]
                !=list(range(1,11))+cfg["evaluation_epochs"] or len(state["TRAIN_epoch_logs"])!=70):
            raise ValueError("Exact20+140updates and prespecified evaluation history incomplete")
        receipt["models"][arm]=dict(selected=state["selected"],history=state["history"],
            steps=160,additional_steps=140,parameter_count=cfg["parameter_count"],
            TRAIN_epoch_logs=state["TRAIN_epoch_logs"],last_checkpoint=last_durable_checkpoint,
            last_checkpoint_epoch=last_durable_epoch,
            selected_origin="initial" if state["best_epoch"]<=10 else "continued",
            external_initial_best=state["external_initial_best"],
            retained_continuation_records={r["path"]:r for r in state["current"]+state["retained_best"]})
        reader.read(record);initial.resources_v1(spec,storage,lease)
        receipt.update(status="completed_same_seed_exact_H036_continuation_total160updates",
            elapsed_seconds=time.monotonic()-started,output_accounted_bytes=output.used,
            initial_best_record_never_modified=True,joint_paired_comparison_pending_other_worker=True)
    except Exception as error:
        receipt.update(status="failed_partial_preserved",error_type=type(error).__name__,error=str(error),
            elapsed_seconds=time.monotonic()-started,output_accounted_bytes=output.used,
            last_durable_epoch=last_durable_epoch,last_durable_checkpoint=last_durable_checkpoint,
            logical_steps_reached=20 if state is None else state["steps"],
            updates_not_checkpointed=0 if state is None else state["steps"]-2*last_durable_epoch,
            automatic_replay_or_resume_executed=False,
            future_recovery_requires_separately_registered_exact_checkpoint_continuation=True)
        output.save("failed.json",receipt);raise
    output.save("completed.json",receipt)
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--input");parser.add_argument("--expected-input-sha256")
    args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="source_only_continuation_not_authorized_or_executed",
            hypothesis_id="H036",implementation={"path":SOURCE,"sha256":fit._hash(SOURCE)},
            config={"path":CONFIG,"sha256":fit._hash(CONFIG)},arms=list(ARMS),
            resume_epoch=10,resume_steps=20,additional_steps=140,total_steps=160,
            fresh_root_continuation_decision_required=True),indent=2));return 0
    if not args.input or not args.expected_input_sha256:
        raise ValueError("Fresh pinned root continuation capability required")
    run_v1({"path":args.input,"sha256":args.expected_input_sha256});return 0


if __name__=="__main__":
    raise SystemExit(main())
