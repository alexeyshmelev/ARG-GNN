"""Admitted GST-LAI-8 fitting and canonical truth-free prediction products."""
from __future__ import annotations
import gc
import copy
import importlib.metadata
import math
import sys
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from . import CONDITION
from .dataset import CompleteTreeSets,loader,load_parents
from .model import GstLai8,specification,AUTHOR
from . import checkpoint as recovery
from arggnn.benchmarking.paper_gst_lai_runtime_v2 import workflow as baseline
from arggnn.benchmarking import european_native_models as io
from arggnn.benchmarking import european_sampling as sampling
from arggnn.benchmarking.paper_ancestralpaths.data import checked_plan

MODULE="research.repairs.benchmark_recovery_20261025.gst.workflow"
SETTINGS=dict(epochs=30,learning_rate=.001,weight_decay=.0001,betas=[.9,.999],
              epsilon=1e-8,clip_grad_norm=1.,workers=4,prefetch_factor=1,
              checkpoint_selection="fixed_final_epoch30",seed=20261024)


def _device(name):
    io._need(name in ("cpu","cuda"),"Explicit ROOT-admitted device required")
    if name == "cuda":
        io._need(os.environ.get("CUDA_VISIBLE_DEVICES") not in (None,"","-1")
            and os.environ.get("ARGGNN_PHYSICAL_GPU") in ("3","4","5")
            and torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "Exactly the currently admitted physical GPU must be exposed")
        return torch.device("cuda:0")
    io._need(os.environ.get("CUDA_VISIBLE_DEVICES") == "","Explicit zero-CUDA CPU condition required")
    return torch.device("cpu")


def _labels(inputs,plan,records):
    values={}
    for parent in plan["train_parent_ids"]:
        target=io._read(inputs[parent]["training_targets_ref"])
        io._need(target["schema"] == "ap_paper_TRAIN_native_targets_v1"
            and target["parent_id"] == parent and target["split"] == "train"
            and target["registry_ref"] == inputs[parent]["sampled_registry_ref"]
            and target["no_label_dependent_selection"] is True
            and target["native_path_names"] == list(sampling.NATIVE_PATHS)
            and target["rows"] == len(records[parent]["record_ids"]),
            "Only exact prepared TRAIN native-six targets may enter adapted fit")
        y=io._arrays(target["arrays_ref"],cap=2*2**20)["native_labels_0based"]
        io._need(y.dtype == np.int64 and y.shape == (len(records[parent]["record_ids"]),)
            and ((y>=0)&(y<6)).all(),"Complete canonical TRAIN labels required")
        values[parent]=y
    return values


def _shutdown(data_loader):
    iterator=getattr(data_loader,"_iterator",None)
    if iterator is not None:
        iterator._shutdown_workers()
    gc.collect()


def fit(plan_ref,train_inputs,output_directory,*,device,threads,max_sites,max_metadata_bytes,
        resource_check,execution_check,write_headroom_bytes,resume_checkpoint_ref=None):
    """Original fit with exact cursor windows and complete-parent recovery only."""
    io._ready(resource_check,execution_check,write_headroom_bytes)
    plan=checked_plan(plan_ref)
    torch.set_num_threads(threads)
    states,records,proof=load_parents(train_inputs,plan,split="train",
        protocol_id="ancestralpaths_paper200mb_native6",max_sites=max_sites,
        max_metadata_bytes=max_metadata_bytes,check=resource_check)
    output=io._new_output(output_directory)
    barrier_ref=io._json(output/"observation_barrier.json",dict(parents=proof,
        all_TRAIN_observed_inputs_sealed_before_targets=True,private_truth_opened=False,
        plan_ref=plan_ref),resource_check)
    labels=_labels(train_inputs,plan,records)
    data=CompleteTreeSets(states,records)
    data_loader,sampler=loader(data,workers=4,pin_memory=device=="cuda")
    torch.manual_seed(SETTINGS["seed"])
    np.random.seed(SETTINGS["seed"]);random.seed(SETTINGS["seed"])
    target_device=_device(device)
    model=GstLai8(sampling.NATIVE_PATHS).to(target_device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001,
                                betas=(.9,.999),eps=1e-8)
    source_refs=repair_source_refs(python=sys.executable)
    updates=0
    history=[]
    total_trees=total_edges=total_sites=0
    first_epoch=1
    restored_progress=None
    pending_rng=None
    elapsed_before=0.
    latest_ref=None
    if resume_checkpoint_ref is not None:
        saved=torch.load(io._checked(resume_checkpoint_ref),map_location="cpu",weights_only=True)
        io._need(saved["plan_ref"] == plan_ref
            and saved["class_order"] == list(sampling.NATIVE_PATHS),
            "Exact original plan and six native classes required for recovery")
        restored_progress=recovery.validate_progress(saved,plan,SETTINGS,train_inputs,source_refs,proof)
        model.load_state_dict(saved["model_state"],strict=True)
        optimizer.load_state_dict(saved["optimizer_state"])
        updates=saved["optimizer_updates"]
        history=copy.deepcopy(saved["history"])
        first_epoch=restored_progress["epoch"]
        total_trees=restored_progress["completed_epoch_totals"]["trees"]
        total_edges=restored_progress["completed_epoch_totals"]["edges"]
        total_sites=restored_progress["completed_epoch_totals"]["sites"]
        elapsed_before=restored_progress["training_elapsed_seconds"]
        pending_rng=saved["random_state"]
        latest_ref=resume_checkpoint_ref
        del saved
    started=time.monotonic()
    try:
        for epoch in range(first_epoch,31):
            order=list(plan["train_parent_ids"])
            random.Random(SETTINGS["seed"]+epoch).shuffle(order)
            completed=[]
            counts={p:dict(trees=0,sites=0,edges=0) for p in order}
            if restored_progress is not None and epoch == first_epoch:
                completed=list(restored_progress["completed_parents"])
                counts=copy.deepcopy(restored_progress["counts"])
            sampler.order=order[len(completed):]
            current=None
            seen={}
            parent_loss=0.
            def finish_parent(parent):
                nonlocal updates,parent_loss,latest_ref
                io._need(bool(seen[parent].all()),"Every canonical TRAIN focal must occur exactly once")
                io._need(counts[parent]["trees"] == states[parent]["ts"].num_trees
                    and counts[parent]["sites"] == states[parent]["ts"].num_sites,
                    "Every original tree and site must precede the parent optimizer step")
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                updates+=1
                history.append(dict(epoch=epoch,parent_id=parent,optimizer_updates=updates,
                    loss=parent_loss,gradient_norm=float(norm),
                    elapsed_seconds=elapsed_before+time.monotonic()-started))
                optimizer.zero_grad(set_to_none=True)
                parent_loss=0.
                completed.append(parent)
                progress=dict(epoch=epoch,completed_parents=list(completed),counts=counts,
                    completed_epoch_totals=dict(trees=total_trees,edges=total_edges,sites=total_sites),
                    training_elapsed_seconds=elapsed_before+time.monotonic()-started)
                payload=dict(schema=recovery.SCHEMA,model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict(),optimizer_updates=updates,
                    class_order=list(sampling.NATIVE_PATHS),settings=SETTINGS,plan_ref=plan_ref,
                    train_inputs=train_inputs,source_refs=source_refs,observation_proof=proof,
                    observation_barrier_ref=barrier_ref,history=history,progress=progress,
                    random_state=recovery.random_state())
                latest=recovery.save_latest(output,payload,resource_check)
                latest_ref=io._ref(latest)
                print("GST completed parent",epoch,parent,"optimizer_updates",updates,flush=True)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            if sampler.order:
                iterator=iter(data_loader)
                if pending_rng is not None:
                    # A newly spawned loader draws a worker base seed. Roll back
                    # that extra main-process draw before any resumed dropout.
                    # The observed-only encoder itself uses no randomness.
                    recovery.restore_random_state(pending_rng)
                    pending_rng=None
                for cpu_window in iterator:
                    execution_check();resource_check("GST_LAI_before_complete_tree_set")
                    parent=cpu_window.parent_id
                    if parent != current:
                        if current is not None:
                            finish_parent(current)
                        current=parent
                        seen[parent]=np.zeros(len(labels[parent]),dtype=bool)
                    counts[parent]["trees"]+=len(cpu_window.tree_node_counts)
                    counts[parent]["sites"]+=cpu_window.sites
                    counts[parent]["edges"]+=cpu_window.native_edge_count
                    window=cpu_window.to(target_device)
                    index=cpu_window.focal_indices.numpy()
                    if len(index):
                        io._need(not seen[parent][index].any(),"No repeated canonical output row")
                        y=torch.as_tensor(labels[parent][index],device=target_device)
                        logits=model.sampled_logits(window)
                        loss=torch.nn.functional.cross_entropy(logits,y,reduction="sum")/len(labels[parent])
                        io._need(bool(torch.isfinite(loss)),"Finite full-parent weighted sampled loss required")
                        loss.backward()
                        parent_loss+=float(loss.detach())
                        seen[parent][index]=True
                    else:
                        # Preserve the original no-focal forward and its dropout draws.
                        with torch.no_grad():
                            model(window)
                    del window,cpu_window
                if current is not None:
                    finish_parent(current)
            for parent in order:
                io._need(counts[parent]["trees"] == states[parent]["ts"].num_trees
                    and counts[parent]["sites"] == states[parent]["ts"].num_sites,
                    "All native trees/sites must traverse each complete training pass")
            io._need(completed == order,"No omitted or repeated parent optimizer step")
            total_trees+=sum(v["trees"] for v in counts.values())
            total_sites+=sum(v["sites"] for v in counts.values())
            total_edges+=sum(v["edges"] for v in counts.values())
            io._json(output/("epoch_"+str(epoch)+".json"),dict(epoch=epoch,
                optimizer_updates=updates,counts=counts,history=history[-3:]),resource_check)
            restored_progress=None
        io._need(updates == 90,"Thirty complete passes over all three TRAIN parents required")
        optimizer_steps=[int(s["step"].item()) for s in optimizer.state.values()]
        io._need(len(optimizer_steps) == len(list(model.parameters())) and set(optimizer_steps) == {90},
            "Every learned tensor must have all 90 actual AdamW updates")
        io._need(repair_source_refs(python=sys.executable) == source_refs,
            "Isolated GST repair sources changed during fit")
        state={k:v.detach().cpu() for k,v in model.state_dict().items()}
        checkpoint=output/"gst_lai_8_epoch30.pt"
        resource_check(16*2**20)
        torch.save(dict(schema="paper_gst_lai_8_checkpoint_v1",model_state=state,
            optimizer_state=optimizer.state_dict(),optimizer_updates=updates,
            class_order=list(sampling.NATIVE_PATHS),settings=SETTINGS,
            plan_ref=plan_ref,observation_barrier_ref=barrier_ref),checkpoint)
        return io._json(output/"fit.json",dict(schema="paper_gst_lai_8_fit_v1",condition=CONDITION,
            plan_ref=plan_ref,train_inputs=train_inputs,checkpoint_ref=io._ref(checkpoint),
            observation_barrier_ref=barrier_ref,settings=SETTINGS,optimizer_updates=updates,
            parameters=sum(p.numel() for p in model.parameters()),model_spec=specification(),
            every_optimizer_tensor_updates=optimizer_steps,
            total_native_tree_visits=total_trees,total_native_edge_visits=total_edges,
            total_native_site_visits=total_sites,history=history,
            native_path_names=list(sampling.NATIVE_PATHS),TEST_truth_read=False,
            recovery_checkpoint_ref=latest_ref,resumed_from_ref=resume_checkpoint_ref,
            recovery_peak_checkpoint_bytes=recovery.PEAK_CHECKPOINT_BYTES),resource_check)
    finally:
        _shutdown(data_loader)


def predict(fit_ref,heldout_inputs,output_directory,*,device,threads,max_sites,max_metadata_bytes,
            resource_check,execution_check,write_headroom_bytes):
    io._ready(resource_check,execution_check,write_headroom_bytes)
    torch.set_num_threads(threads)
    fitted=io._read(fit_ref,observed=True)
    io._need(fitted["schema"] == "paper_gst_lai_8_fit_v1" and fitted["condition"] == CONDITION
        and fitted["optimizer_updates"] == 90 and fitted["settings"] == SETTINGS,
        "Full fixed-budget GST-LAI-8 final checkpoint required")
    plan=checked_plan(fitted["plan_ref"])
    states,records,proof=load_parents(heldout_inputs,plan,split="test",
        protocol_id="ancestralpaths_paper200mb_native6",max_sites=max_sites,
        max_metadata_bytes=max_metadata_bytes,check=resource_check)
    data=CompleteTreeSets(states,records)
    data_loader,sampler=loader(data,workers=4,pin_memory=device=="cuda")
    model=GstLai8(sampling.NATIVE_PATHS)
    saved=torch.load(io._checked(fitted["checkpoint_ref"],observed=True),
                     map_location="cpu",weights_only=True)
    io._need(saved["schema"] == "paper_gst_lai_8_checkpoint_v1"
        and saved["class_order"] == list(sampling.NATIVE_PATHS)
        and saved["settings"] == SETTINGS and saved["optimizer_updates"] == 90,
        "Exact adapted model checkpoint and class order required")
    model.load_state_dict(saved["model_state"],strict=True)
    del saved
    target_device=_device(device)
    model=model.to(target_device).eval()
    probabilities={p:np.empty((len(records[p]["record_ids"]),6),np.float32) for p in states}
    seen={p:np.zeros(len(records[p]["record_ids"]),bool) for p in states}
    counts={p:dict(trees=0,sites=0,edges=0) for p in states}
    try:
        with torch.no_grad():
            for cpu_window in data_loader:
                execution_check();resource_check("GST_LAI_TEST_observed_tree_set")
                p=cpu_window.parent_id
                counts[p]["trees"]+=len(cpu_window.tree_node_counts)
                counts[p]["sites"]+=cpu_window.sites
                counts[p]["edges"]+=cpu_window.native_edge_count
                window=cpu_window.to(target_device)
                logits=model.sampled_logits(window)
                index=cpu_window.focal_indices.numpy()
                io._need(not seen[p][index].any(),"Every canonical TEST row must be emitted once")
                probabilities[p][index]=logits.softmax(-1).cpu().numpy()
                seen[p][index]=True
        output=io._new_output(output_directory)
        predictions={}
        for parent in plan["test_parent_ids"]:
            io._need(seen[parent].all() and counts[parent]["trees"] == states[parent]["ts"].num_trees
                and counts[parent]["sites"] == states[parent]["ts"].num_sites,
                "All canonical TEST rows and every native tree/site required")
            p = probabilities[parent]
            io._need(p.shape == (10000,6) and p.dtype == np.float32
                and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
                and np.allclose(p.sum(1),1.,atol=1e-6,rtol=1e-5),
                "Every raw native-six probability must be finite and normalized before seal")
            array_ref=io._npz(output/(parent+".npz"),dict(
                probabilities=probabilities[parent],example_indices=records[parent]["record_ids"],
                example_ids=np.asarray([parent+":"+str(i) for i in records[parent]["record_ids"]])),
                resource_check)
            predictions[parent]=io._json(output/(parent+".json"),dict(
                schema="ap_paper_native_predictions_v1",condition=CONDITION,parent_id=parent,
                split="test",rows=len(probabilities[parent]),
                registry_ref=heldout_inputs[parent]["sampled_registry_ref"],
                predictions_ref=array_ref,fit_ref=fit_ref,native_path_names=list(sampling.NATIVE_PATHS),
                query_truth_read=False,counts=counts[parent],model_spec=specification()),
                resource_check)
        return dict(schema="paper_gst_lai_8_prediction_completion_v1",predictions=predictions,
            all_five_TEST_parent_predictions_sealed=True,query_truth_read=False,
            observed_parent_proof=proof)
    finally:
        _shutdown(data_loader)


def execute_stage(spec,*,resource_check,execution_check):
    args=dict(spec["arguments"])
    io._ready(resource_check,execution_check,args["write_headroom_bytes"])
    io._need(Path(sys.prefix).resolve() == io._path(spec["runtime_prefix"])
        and torch.__version__ == "2.6.0+cu124"
        and importlib.metadata.version("torch-geometric") == "2.7.0"
        and importlib.metadata.version("tskit") == "1.0.3",
        "Explicit qualified project Torch2.6/PyG2.7/tskit1.0.3 runtime required")
    fn={"fit":fit,"predict":predict}[spec["stage"]]
    return fn(output_directory=spec["output_directory"],resource_check=resource_check,
              execution_check=execution_check,**args)


def repair_source_refs(*,python):
    # Original factory supplies the complete released/native dependency pins.
    canonical_python=str(Path(python).resolve())
    original=baseline.make_call("fit",{},str(io.ROOT/"research/repairs/benchmark_recovery_20261025/gst"),python=canonical_python)
    names=("__init__.py","model.py","dataset.py","checkpoint.py","workflow.py")
    return original["source_refs"]+[io._ref(Path(__file__).parent/name) for name in names]


def make_call(stage,arguments,output_directory,*,python):
    result=baseline.make_call(stage,arguments,output_directory,python=python)
    result["module"]=MODULE
    result["source_refs"]=repair_source_refs(python=python)
    return result
