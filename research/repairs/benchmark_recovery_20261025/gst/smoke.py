"""Small synthetic CPU comparisons; no benchmark data or CUDA is opened."""
from __future__ import annotations
import argparse
import copy
from dataclasses import fields, is_dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
import tskit
from torch.utils.data import DataLoader

from arggnn.benchmarking.paper_gst_lai_runtime_v2 import dataset as original
from arggnn.benchmarking.paper_gst_lai_runtime_v2.workflow import SETTINGS, _shutdown
from arggnn.benchmarking.european_sampling import NATIVE_PATHS
from . import checkpoint as recovery
from .dataset import CompleteTreeSets, loader
from .model import GstLai8
from .workflow import repair_source_refs

TENSORS_COMPARED = 0


def same(left, right):
    global TENSORS_COMPARED
    if torch.is_tensor(left):
        assert left.dtype == right.dtype and left.shape == right.shape
        assert torch.equal(left, right), "Tensor values differ"
        TENSORS_COMPARED += 1
    elif is_dataclass(left):
        assert type(left) is type(right)
        for field in fields(left):same(getattr(left,field.name),getattr(right,field.name))
    elif isinstance(left,dict):
        assert set(left) == set(right)
        for key in left:same(left[key],right[key])
    elif isinstance(left,(tuple,list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a,b in zip(left,right):same(a,b)
    else:
        assert left == right


def synthetic_state(connected):
    tables=tskit.TableCollection(float(connected+1))
    for _ in range(6):tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE,time=0.)
    genotypes=[];positions=[];alleles=[]
    for hold in range(connected):
        root=tables.nodes.add_row(time=20.+hold*.01)
        unary=tables.nodes.add_row(time=5.+hold*.001)
        tables.edges.add_row(hold,hold+1,root,unary)
        tables.edges.add_row(hold,hold+1,unary,0)
        for leaf in range(1,6):tables.edges.add_row(hold,hold+1,root,leaf)
        for offset in ([.25,.75] if hold%5 == 0 else [.5]):
            count=3 if hold%3 == 0 else 2
            calls=[(hold+leaf)%count for leaf in range(6)]
            site=tables.sites.add_row(hold+offset,"A")
            for leaf,value in enumerate(calls):
                if value:tables.mutations.add_row(site,leaf,"CG"[value-1])
            if hold%7 == 0:calls[5]=-1
            genotypes.append(calls);positions.append(hold+offset);alleles.append(count)
    tables.sort()
    ts=tables.tree_sequence()
    state=dict(ts=ts,nodes=ts.samples(),node_time=np.asarray(ts.nodes_time),
        genotype=np.asarray(genotypes,np.int8),allele_count=np.asarray(alleles,np.int8),
        positions=np.asarray(positions),labels=np.asarray([0,1,2,3,-1,-1]),
        ages=np.zeros(6),query_columns=np.asarray([4,5]))
    focal_holds=np.arange(0,connected,11)
    records=dict(record_ids=np.arange(2*len(focal_holds),dtype=np.int64),
        sample_indices=np.tile(np.asarray([4,5]),len(focal_holds)),
        hold=np.repeat(focal_holds,2))
    return state,records


def originals():
    states={};records={}
    for number,connected in enumerate((66,74,82)):
        parent="synthetic_"+str(number)
        states[parent],records[parent]=synthetic_state(connected)
    return states,records


def ordered_indices(data,epoch):
    order=list(data.parent_order)
    random.Random(SETTINGS["seed"]+epoch).shuffle(order)
    return order,[index for parent in order for index in data.parent_windows[parent]]


def compare_windows(states,records):
    baseline=original.CompleteTreeSets(states,records)
    workers=[CompleteTreeSets(states,records) for _ in range(4)]
    visits=0
    for epoch in (1,2):
        _,indices=ordered_indices(baseline,epoch)
        for worker in range(4):
            for index in indices[worker::4]:
                same(baseline[index],workers[worker][index])
                visits+=1
    random_access=CompleteTreeSets(states,records)
    for index in (len(baseline)-1,0,3,1,1,0):
        same(baseline[index],random_access[index])
    state=random_access.__getstate__()
    assert state["_cursor"] is None and state["workspaces"] == {}
    return dict(exact_window_visits=visits,random_access_checks=6,
        worker_cursor_counts=[worker.cursor_counts for worker in workers],
        original_new_Tree_seeks=visits,pickle_excludes_cursor_and_workspace=True)


def compare_loader(states,records):
    data=CompleteTreeSets(states,records)
    baseline=original.CompleteTreeSets(states,records)
    data_loader,sampler=loader(data,workers=4,pin_memory=False)
    pids=[];processes=[];visits=0
    try:
        for epoch in (1,2):
            order,indices=ordered_indices(data,epoch)
            sampler.order=order
            emitted=[]
            for part in data_loader:
                index=indices[len(emitted)]
                same(part,baseline[index])
                emitted.append((part.parent_id,part.start_hold))
                visits+=1
            assert emitted == [data.windows[index] for index in indices]
            current=[process.pid for process in data_loader._iterator._workers]
            if pids:assert current == pids
            pids=current
        processes=list(data_loader._iterator._workers)
    finally:
        _shutdown(data_loader)
    assert len(processes) == 4 and all(not process.is_alive() and process.exitcode == 0 for process in processes)
    return dict(workers=4,persistent_across_two_epochs=True,worker_pids=pids,
        worker_exitcodes=[process.exitcode for process in processes],exact_window_visits=visits)


def update(model,optimizer,data,parent,records):
    model.train();optimizer.zero_grad(set_to_none=True)
    seen=np.zeros(len(records[parent]["record_ids"]),bool)
    outputs=[];loss_value=0.
    counts=dict(trees=0,sites=0,edges=0)
    for index in data.parent_windows[parent]:
        window=data[index]
        counts["trees"]+=len(window.tree_node_counts)
        counts["sites"]+=window.sites
        counts["edges"]+=window.native_edge_count
        rows=window.focal_indices.numpy()
        if len(rows):
            target=torch.as_tensor(rows%6,dtype=torch.long)
            logits=model.sampled_logits(window)
            outputs.append(logits.detach().clone())
            loss=torch.nn.functional.cross_entropy(logits,target,reduction="sum")/len(seen)
            loss.backward();loss_value+=float(loss.detach());seen[rows]=True
        else:
            with torch.no_grad():model(window)
    assert seen.all()
    gradients={name:parameter.grad.detach().clone() for name,parameter in model.named_parameters()}
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    return dict(logits=outputs,gradients=gradients,loss=loss_value,norm=float(norm),counts=counts)


def new_model(initial=None):
    model=GstLai8(NATIVE_PATHS)
    if initial is not None:model.load_state_dict(initial)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001,betas=(.9,.999),eps=1e-8)
    return model,optimizer


def compare_updates_and_resume(states,records,output):
    torch.manual_seed(SETTINGS["seed"]);np.random.seed(SETTINGS["seed"]);random.seed(SETTINGS["seed"])
    baseline,optimizer=new_model()
    initial=recovery.cpu_copy(baseline.state_dict());initial_rng=recovery.random_state()
    original_data=original.CompleteTreeSets(states,records)
    order,_=ordered_indices(original_data,1)
    first=update(baseline,optimizer,original_data,order[0],records)
    boundary_model=recovery.cpu_copy(baseline.state_dict())
    boundary_optimizer=recovery.cpu_copy(optimizer.state_dict())
    boundary_rng=recovery.random_state()
    candidate,candidate_optimizer=new_model(initial)
    recovery.restore_random_state(initial_rng)
    candidate_first=update(candidate,candidate_optimizer,CompleteTreeSets(states,records),order[0],records)
    same(first,candidate_first)
    same(boundary_model,candidate.state_dict());same(boundary_optimizer,candidate_optimizer.state_dict())
    same(boundary_rng,recovery.random_state())
    recovery.restore_random_state(boundary_rng)

    plan=dict(train_parent_ids=list(states))
    proof={parent:dict(trees=state["ts"].num_trees,sites=state["ts"].num_sites) for parent,state in states.items()}
    counts={parent:(first["counts"] if parent == order[0] else dict(trees=0,sites=0,edges=0)) for parent in states}
    sources=repair_source_refs(python=os.sys.executable)
    inputs={parent:dict(synthetic_only=True) for parent in states}
    history=[dict(epoch=1,parent_id=order[0],optimizer_updates=1)]
    payload=dict(schema=recovery.SCHEMA,model_state=boundary_model,optimizer_state=boundary_optimizer,
        optimizer_updates=1,settings=SETTINGS,train_inputs=inputs,source_refs=sources,
        observation_proof=proof,history=history,random_state=boundary_rng,
        progress=dict(epoch=1,completed_parents=[order[0]],counts=counts,
            completed_epoch_totals=dict(trees=0,sites=0,edges=0),training_elapsed_seconds=0.))
    checkpoints=output/"checkpoint_test";checkpoints.mkdir()
    latest=recovery.save_latest(checkpoints,payload,lambda stage:None)
    saved=torch.load(latest,map_location="cpu",weights_only=True)
    recovery.validate_progress(saved,plan,SETTINGS,inputs,sources,proof)
    second=update(baseline,optimizer,original_data,order[1],records)
    expected_model=recovery.cpu_copy(baseline.state_dict())
    expected_optimizer=recovery.cpu_copy(optimizer.state_dict())
    expected_rng=recovery.random_state()

    resumed,resumed_optimizer=new_model()
    resumed.load_state_dict(saved["model_state"])
    resumed_optimizer.load_state_dict(saved["optimizer_state"])
    # Exercise the same extra base-seed draw made by a newly constructed loader.
    iterator=iter(DataLoader([0],batch_size=None,num_workers=0))
    assert next(iterator) == 0
    recovery.restore_random_state(saved["random_state"])
    actual=update(resumed,resumed_optimizer,CompleteTreeSets(states,records),order[1],records)
    same(second,actual)
    same(expected_model,resumed.state_dict());same(expected_optimizer,resumed_optimizer.state_dict())
    same(expected_rng,recovery.random_state())
    assert set(int(value["step"]) for value in resumed_optimizer.state.values()) == {2}
    rejected=0
    for field,value in (("source_refs",[]),("optimizer_updates",2)):
        invalid=copy.deepcopy(saved);invalid[field]=value
        try:recovery.validate_progress(invalid,plan,SETTINGS,inputs,sources,proof)
        except ValueError:rejected+=1
    assert rejected == 2
    cap_rejected=False
    try:recovery.CappedWriter(io.BytesIO()).write(bytes(recovery.MAX_CHECKPOINT_BYTES+1))
    except ValueError:cap_rejected=True
    assert cap_rejected
    payload["optimizer_updates"]=2
    payload["model_state"]=expected_model;payload["optimizer_state"]=expected_optimizer
    payload["history"]=history+[dict(epoch=1,parent_id=order[1],optimizer_updates=2)]
    payload["progress"]["completed_parents"]=order[:2]
    payload["progress"]["counts"][order[1]]=second["counts"]
    payload["random_state"]=expected_rng
    recovery.save_latest(checkpoints,payload,lambda stage:None)
    final=torch.load(latest,map_location="cpu",weights_only=True)
    recovery.validate_progress(final,plan,SETTINGS,inputs,sources,proof)
    assert not (checkpoints/"recovery_latest.staging.pt").exists()
    assert len(list(checkpoints.iterdir())) == 1
    return dict(original_vs_cursor_logits_gradients_parameters_optimizer_exact=True,
        dropout_training_enabled=True,all_learned_tensors=len(resumed_optimizer.state),
        resumed_second_update_exact=True,all_Adam_counters=2,
        post_update_python_numpy_torch_RNG_exact=True,new_loader_seed_draw_rolled_back=True,
        recovery_bytes=latest.stat().st_size,checkpoint_cap_bytes=recovery.MAX_CHECKPOINT_BYTES,
        checkpoint_peak_cap_bytes=recovery.PEAK_CHECKPOINT_BYTES,
        invalid_binding_or_counter_rejected=rejected,oversize_write_rejected=cap_rejected,
        atomic_latest_replaced_with_no_staging=True)


def seek_microbenchmark():
    state,_=synthetic_state(2048)
    ts=state["ts"]
    starts=list(range(0,ts.num_trees,8))
    original_hash=hashlib.sha256();cursor_hash=hashlib.sha256()
    started=time.perf_counter()
    for worker in range(4):
        for start in starts[worker::4]:
            tree=ts.at_index(start)
            original_hash.update(tree.parent_array.tobytes())
    original_seconds=time.perf_counter()-started
    started=time.perf_counter()
    advances=0
    for worker in range(4):
        tree=ts.first()
        for start in starts[worker::4]:
            while tree.index < start:
                assert tree.next();advances+=1
            cursor_hash.update(tree.parent_array.tobytes())
    cursor_seconds=time.perf_counter()-started
    assert original_hash.digest() == cursor_hash.digest()
    return dict(synthetic_native_trees=ts.num_trees,windows=len(starts),
        original_new_Tree_seeks=len(starts),cursor_new_Tree_starts=4,cursor_next_calls=advances,
        selected_parent_arrays_exact=True,original_seek_seconds=original_seconds,
        cursor_seconds=cursor_seconds,not_end_to_end_training_speed=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    output=Path(args.output)
    assert not output.exists()
    output.mkdir(parents=True)
    started=time.monotonic()
    states,records=originals()
    result=dict(schema="GST_cursor_recovery_synthetic_CPU_qualification_v1",status="PASS",
        synthetic_only=True,actual_TRAIN_or_TEST_data_opened=False,CUDA_used=False,
        window_comparison=compare_windows(states,records),
        four_worker_loader=compare_loader(states,records),
        optimizer_and_recovery=compare_updates_and_resume(states,records,output),
        seek_microbenchmark=seek_microbenchmark())
    result["tensor_comparisons"]=TENSORS_COMPARED
    result["elapsed_seconds"]=time.monotonic()-started
    result["source_refs"]=repair_source_refs(python=os.sys.executable)
    assert not torch.cuda.is_initialized()
    (output/"receipt.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__ == "__main__":
    main()
