"""CPU-only cache control, exact codec, resume schedule and admission qualification."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import copy
import json
import random
import threading
import time
import numpy as np
import torch
import tskit
from arggnn.benchmarking import contracts as c
from . import training
from .pool import MemoryGovernor, PipelinedCompactParents
from research.reviews.workflow_progress_diagnosis_20261018.treeloom.resident_pool_v2 import exact_signature

BASE=c.ROOT/"research/repairs/TreeLoom_lazy_compact_prefetch_20261024_v1"

def need(ok,message):
    if not ok:raise AssertionError(message)

def fixture(base,registry,runtime):
    parent=registry.ids("train")[0]
    ages=np.asarray(([0.]*100+[1000.]*100)*3+[0.]*1000)
    tables=tskit.TableCollection(sequence_length=30000000)
    current=[tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE,time=age) for age in ages]
    clock=5000.
    while len(current)>1:
        parents=[]
        for first in range(0,len(current),2):
            up=tables.nodes.add_row(time=clock)
            for child in current[first:first+2]:tables.edges.add_row(0,30000000,up,child)
            parents.append(up)
        current=parents;clock*=2
    edge_parents=tuple(tables.edges.parent);edge_children=tuple(tables.edges.child)
    tables.edges.clear()
    for left,right in ((0,500),(500,30000000)):
        for up,down in zip(edge_parents,edge_children):tables.edges.add_row(left,right,int(up),int(down))
    positions=np.asarray([10.,30.,15000060.,15000090.])
    for position in positions:tables.sites.add_row(position,"A")
    tables.sort();ts=tables.tree_sequence();nodes=np.asarray(ts.samples(),dtype=np.int64)
    focal_indices=600+np.arange(10000,dtype=np.int64)%1000
    focal_positions=np.concatenate((np.linspace(.05,499.95,5000),np.linspace(1000.,29999900.,5000)))
    focals=dict(record_ids=np.arange(10000,dtype=np.int64),sample_nodes=nodes[focal_indices],
        sample_indices=focal_indices,positions_bp0=focal_positions,parent_id=np.asarray(parent),
        class_order=np.asarray(training.fresh.CLASSES))
    np.savez(base/"focals.npz",**focals)
    manifest=dict(schema="american_c3_observed_v1",genealogy_kind=runtime.frozen.GENEALOGY,
        query_ancestry_truth_in_inputs=False,class_order=list(training.fresh.CLASSES),parent_id=parent,
        sequence_length_bp=30000000,synthetic_CPU_smoke=True)
    (base/"observation.json").write_text(json.dumps(manifest)+"\n")
    (base/"preparation.json").write_text(json.dumps(dict(synthetic_CPU_smoke=True))+"\n")
    refs=dict(observation_ref=c.file_ref(base/"observation.json"),
        focal_registry_ref=c.file_ref(base/"focals.npz"),native_preparation_ref=c.file_ref(base/"preparation.json"))
    state=dict(manifest=manifest,ts=ts,nodes=nodes,labels=np.asarray([2]*200+[0]*200+[1]*200+[-1]*1000),
        query_columns=np.arange(600,1600),focals=focals,
        genotype=np.random.default_rng(71).binomial(1,.2,size=(4,1600)).astype(np.int8),
        positions=positions,allele_count=np.full(4,2,dtype=np.int8),ages=ages,
        spans=np.asarray([500.,29999500.]),focal_tree=(focal_positions>=500).astype(np.int64))
    return runtime.prepare_observation(state,split="train",resource_check=lambda stage:None,
        **refs,max_nodes=100000,max_holds=4,max_episode_payload_bytes=16*2**20),refs

def state_equal(a,b):
    return (a[0]==b[0] and a[1][0]==b[1][0] and np.array_equal(a[1][1],b[1][1])
            and a[1][2:]==b[1][2:] and torch.equal(a[2],b[2]))

def rng_state():return random.getstate(),np.random.get_state(),torch.get_rng_state().clone()

def codec_check(base,registry):
    runtime=training.fresh.FreshTreeLoom(registry)
    before=rng_state()
    (bundle,metadata),refs=fixture(base,registry,runtime)
    need(state_equal(before,rng_state()),"Observed CPU fixture preparation must not consume global RNG")
    parent=bundle.parent_id;signature=exact_signature(bundle,metadata)
    prepared=[]
    def prepare(p,*,resource_check):
        need(p==parent,"Synthetic same-parent identity");prepared.append(p)
        resource_check("synthetic_prepare");return bundle,metadata
    pool=PipelinedCompactParents(parent_bounds={parent:128*2**20},input_refs={parent:refs},external_arrays={},
        max_cache_bytes=128*2**20,max_rss_bytes=2**30,phase_reserve_bytes=128*2**20,
        rss_bytes=lambda:0,resource_check=lambda stage:None,event=lambda name,value:None,prepare=prepare)
    try:
        for _ in range(3):
            restored,meta=pool.load(parent)
            need(exact_signature(restored,meta)==signature,"Qualified codec must restore every tensor and scalar exactly")
            need(restored.bank.parent.data_ptr()!=bundle.bank.parent.data_ptr(),"Restored expanded arrays must be independent")
            del restored,meta
        receipt=pool.receipt()
        need(prepared==[parent] and receipt["restorations"]=={parent:3},"Only one preparation across repeated restores")
        need(state_equal(before,rng_state()),"Background CPU preparation/codec must preserve global RNG")
        return dict(passed=True,native_holds=bundle.native_holds,haplotypes=1600,focals=10000,
            observed_sites=4,exact_restores=3,global_RNG_unchanged=True,pool=receipt)
    finally:pool.close()

def schedule_check(registry,original,barrier):
    schedule=[(e,p) for e in range(1,31) for p in registry.epoch_order(e)]
    need(schedule[58][1]=="argformer_american_train_013","Saved58 must continue with original parent013")
    inputs=original["train_inputs"];prepared=[];updates=[];targets=[];events=[];cursor=[58];active=[False]
    threads=set();producing=[0];max_producing=[0];lock=threading.Lock()
    def prepare(parent,*,resource_check):
        with lock:
            producing[0]+=1;max_producing[0]=max(max_producing[0],producing[0]);threads.add(threading.get_ident())
        resource_check("fake_observed_prepare");time.sleep(.001)
        seal=barrier["parents"][parent]
        bundle=SimpleNamespace(parent_id=parent,split="train",sequence_length_bp=30000000,
            focal_identity_sha256=seal["focal_identity_sha256"],validate=lambda:None,
            **{key:inputs[parent][key] for key in training.fresh.INPUT_REFS})
        meta=dict(parent_id=parent,declared_split="train",parent_registry_ref=registry.reference,
            all_native_trees_nodes_sites_samples_retained=True,private_truth_opened=False)
        with lock:prepared.append(parent);producing[0]-=1
        return bundle,meta
    def pack(bundle,meta,**kwargs):return dict(bundle=bundle,meta=meta,charged_bytes=10,read_peak_upper_bound_bytes=10)
    def restore(record,**kwargs):return record["bundle"],record["meta"]
    pool=PipelinedCompactParents(parent_bounds={p:10 for p in registry.ids("train")},
        input_refs=inputs,external_arrays={},max_cache_bytes=200,max_rss_bytes=1000000,
        phase_reserve_bytes=1000,rss_bytes=lambda:0,resource_check=lambda stage:None,
        event=lambda name,value:events.append((name,value)),prepare=prepare,pack=pack,restore=restore)
    def load(parent):
        need(parent==schedule[cursor[0]][1],"No update reorder")
        return pool.load(parent)
    @contextmanager
    def phase():
        with pool.training_phase():
            active[0]=True
            pool.prefetch_next([p for _,p in schedule[cursor[0]+1:]])
            try:yield
            finally:active[0]=False
    def join(parent,bundle,meta):
        need(active[0] and parent in prepared,"Complete actual parent plus RAM admission before its labels")
        targets.append(parent)
        receipt=dict(parent_id=parent,split="train",parent_registry_ref=registry.reference,
            truth_ref=inputs[parent]["truth_ref"],focal_registry_ref=bundle.focal_registry_ref,
            focal_identity_sha256=bundle.focal_identity_sha256,records=10000,labels_in_observation_bank=False)
        return SimpleNamespace(parent_id=parent,focal_registry_ref=bundle.focal_registry_ref,
            focal_identity_sha256=bundle.focal_identity_sha256,receipt=receipt)
    optimizer=SimpleNamespace(steps=58)
    def step(model,bundle,target,opt,meta,**kwargs):
        need(active[0],"Training phase must remain reserved");updates.append(bundle.parent_id);opt.steps+=1
        return dict(parent_id=bundle.parent_id,one_complete_parent_optimizer_update=True,native_focals=10000)
    def checkpoint(epoch,steps,model,opt,latest=False):
        if latest:cursor[0]=steps
        return dict(path="synthetic",sha256=str(steps))
    try:
        result=training.train_loop(registry=registry,runtime=SimpleNamespace(train_step=step),model=None,
            optimizer=optimizer,train_inputs=inputs,observation_barrier=barrier,load_parent=load,
            join_targets=join,checkpoint=checkpoint,event=lambda *args:None,resource_check=lambda stage:None,
            optimizer_counts=lambda opt:[opt.steps]*49,start_steps=58,training_phase=phase)
        receipt=pool.receipt()
        need(updates==targets==[p for _,p in schedule[58:]],"Every remaining update must preserve original order/objective")
        need(len(prepared)==20 and set(prepared)==set(registry.ids("train")),"All20 prepare exactly once")
        need(max_producing==[1] and len(threads)==1,"Exactly one producer thread")
        need(receipt["parents"]==20 and receipt["compact_charged_bytes"]==200,"All20 compact records retained")
        return dict(passed=True,start_steps=58,next_parent=schedule[58][1],remaining_updates=len(updates),
            fixed_final_updates=result["complete_parent_updates"],one_producer_thread=True,
            labels_only_after_full_parent_and_memory_admission=True,original_schedule_identity=training.schedule_identity(schedule),pool=receipt)
    finally:pool.close()

def admission_check():
    admitted=threading.Event();started=threading.Event();errors=[]
    governor=MemoryGovernor(maximum_bytes=100,rss_bytes=lambda:0,resource_check=lambda stage:None,reserve_bytes=60)
    def other():
        started.set()
        with governor.phase("producer"):admitted.set()
    with governor.phase("training"):
        thread=threading.Thread(target=other);thread.start();need(started.wait(1),"Waiter starts")
        need(not admitted.wait(.03),"Concurrent complete phases must wait when both reserves cannot fit")
    thread.join(2);need(not thread.is_alive() and admitted.is_set(),"Waiter resumes after phase release")
    governor.close()
    blocked=MemoryGovernor(maximum_bytes=100,rss_bytes=lambda:90,resource_check=lambda stage:None,reserve_bytes=20)
    try:
        with blocked.phase("impossible"):raise AssertionError("Unadmitted phase entered")
    except ValueError:pass
    else:raise AssertionError("Impossible RSS must fail without eviction")
    blocked.close()
    kwargs=dict(parent_bounds={"p":10},input_refs={"p":{}},external_arrays={},max_cache_bytes=10,
        max_rss_bytes=1000,phase_reserve_bytes=100,rss_bytes=lambda:0,resource_check=lambda stage:None,
        event=lambda *args:None)
    calls=[]
    def failed(parent,**kwargs):calls.append(parent);raise RuntimeError("Expected synthetic producer failure")
    pool=PipelinedCompactParents(**kwargs,prepare=failed)
    try:
        for _ in range(2):
            try:pool.load("p")
            except RuntimeError:pass
            else:raise AssertionError("Producer error must propagate")
        need(calls==["p"],"Failed parent must not silently retry")
    finally:pool.close()
    entering=threading.Event()
    def cancellable(parent,*,resource_check):
        entering.set()
        while True:resource_check("cooperative_close");time.sleep(.001)
    pool=PipelinedCompactParents(**kwargs,prepare=cancellable);pool.prefetch("p")
    need(entering.wait(1),"Producer entered before cooperative close")
    closer=threading.Thread(target=pool.close);closer.start();closer.join(2)
    need(not closer.is_alive(),"Cooperative close must not deadlock")
    return dict(passed=True,overlap_denied_until_capacity=True,impossible_RSS_rejected=True,
        producer_failure_propagates_without_retry=True,cooperative_close_no_deadlock=True)

def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    base=BASE/"cpu_smoke_v2";base.mkdir()
    started=time.monotonic()
    job=c.read_json(".runtime/benchmarking/workflow_20261019_v8/jobs/AM_treeloom_fit.json")
    original=job["call"]["spec"]["training_spec"]
    registry=training.fresh.FreshParentRegistry.from_ref(original["parent_registry_ref"])
    captured=c.read_json("experiments/results/paper_campaign_20261024/treeloom_migration_capture/completed.json")
    checkpoint_ref=captured["checkpoint_ref"]
    saved=torch.load(c.contained_path(checkpoint_ref["path"]),map_location="cpu",weights_only=True,mmap=True)
    model=training.fresh.FreshTreeLoom(registry).build_model()
    evidence_ref=c.file_ref("experiments/results/paper_campaign_20261024/treeloom_migration_capture/evidence.json")
    spec=dict(original,resume_checkpoint_ref=checkpoint_ref,resume_evidence_ref=evidence_ref)
    need(training.validate_resume_checkpoint(saved,model,registry,spec)==58,"Actual captured49-state Adam checkpoint must validate at58")
    barrier=c.read_ref(saved["observation_barrier_ref"])
    result=dict(schema="TreeLoom_lazy_compact_prefetch_CPU_qualification_v1",passed=True,CPU_only=True,
        no_GPU_initialization=not torch.cuda.is_initialized(),actual_saved_updates=58,
        actual_saved_optimizer_states=len(saved["optimizer_state"]["state"]),checkpoint_ref=checkpoint_ref,
        codec=codec_check(base,registry),schedule=schedule_check(registry,original,barrier),
        admission=admission_check(),seconds=time.monotonic()-started,
        limitations="CPU synthetic control and two-hold1600hap10000focal exact codec checks; no full-parent GPU run or speed measured")
    (base/"completed.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
