"""Actual guarded frozen TreeLoom qualification/full-training worker, inactive by default."""
from pathlib import Path
import argparse,hashlib,json,os,resource,sys,time
from . import contract_v1 as c
from . import borrowed_gpu0_v1 as borrowed
def run_v1(input_ref,ticket_ref,gate_fd):
    c.require(os.read(gate_fd,1)==b"G","Genuine parent launch receipt before worker");os.close(gate_fd)
    q,_,_,storage=c.validate_input(input_ref);os.sched_setaffinity(0,q["cpu_affinity"]);g=c.guard_v1(q,storage)
    runtime=c.read(q["runtime_ref"])
    c.require(Path.cwd()==c.ROOT and Path(sys.executable).resolve()==c.local(runtime["python"]).resolve()
        and sys.dont_write_bytecode and sys.flags.no_user_site and sys.flags.optimize==0,
        "Actual ROOT-qualified nonoptimized scientific interpreter")
    ticket=c.read(ticket_ref)
    c.require(ticket["job_id"]==q["job_id"]and ticket["physical_gpu"]==0 and ticket["gpu_uuid"]==c.GPU0_UUID
        and ticket["direct_attachment_authority_ref"]==q["direct_attachment_authority_ref"],"Exact parent borrowed lease ticket")
    native=borrowed.install_before_torch_v1(q,q,ticket)
    # Original helper verifies setproctitle and full /proc/self/cmdline before any CUDA.
    title=native.apply_and_validate_title()
    import torch
    from arggnn.american_c3_gnn_dataset_v1 import load_observation
    from arggnn.american_c3_gnn_train_v1 import _cpu_snapshot
    from research.reviews.frozen_benchmark_training_ready_20261017_v1 import american_full30mb_frozen_v1 as frozen
    from research.reviews.frozen_benchmark_training_ready_20261017_v1 import american_full30mb_prepare_v1 as preparation
    from research.reviews.frozen_benchmark_training_ready_20261017_v1.american_full30mb_train_v1 import epoch_order_v1
    from .targets_v1 import TrainTargetJoinV1
    from .bounded_execution_v1 import train_step_v1
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    total=torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(q["caps"]["max_cuda_reserved_bytes"]/total,device=0)
    torch.cuda.reset_peak_memory_stats(0)
    caps=q["caps"];start=time.monotonic();clock_start=__import__("datetime").datetime.fromisoformat(q["created_at_utc"]).timestamp()
    deadline=min(clock_start+caps["max_seconds"],q["drain_expires_unix"])-caps["work_cutoff_margin_seconds"]
    last=[0.];latest=[None]
    def tick(label):
        c.active(q);c.require(time.time()<deadline,"Original finite work cutoff")
        c.require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=caps["max_cpu_rss_bytes"],"Own384GiB CPU RSS cap")
        c.require(torch.cuda.memory_reserved(0)<=64*2**30,"Own64GiB allocator cap")
        if time.monotonic()-last[0]>=30:
            g.budget_check_v9(storage,boundary=True,required_write_bytes=65536)
            borrowed.check_attachment_v1(q,ticket,int(os.environ["ARGGNN_GPU_LEASE_FD"]),int(os.environ["downstream_LOGICAL_LOCK_FD"]))
            latest[0]=borrowed.capacity_v1(q,borrowed.base_v1(),worker_pid=os.getpid())
            last[0]=time.monotonic()
    output=c.local(q["output_directory"])
    def event(name,value):
        data=c.encoded(dict(event=name,**value))+b"\n";c.require(len(data)<=65536,"Bounded event")
        tick("before_progress_write")
        path=output/"progress.jsonl"
        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_APPEND|os.O_NOFOLLOW,0o600)
        try:
            c.require(os.fstat(fd).st_size+len(data)<=2**20,"Bounded600step progress file")
            c.require(os.write(fd,data)==len(data),"Complete progress write");os.fsync(fd)
        finally:os.close(fd)
    def checkpoint(epoch,payload):
        tick("before_checkpoint");g.budget_check_v9(storage,boundary=True,required_write_bytes=8*2**20)
        c.require(4096+sum(p.stat().st_size+4096 for p in output.iterdir())+8*2**20+4096<=caps["output_cap_bytes"],"Prospective primary32MiB writer bound")
        path=output/("qualification.checkpoint.pt"if q["phase"]=="qualification"else"epoch_%02d.pt"%epoch)
        with path.open("xb")as stream:torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
        c.require(path.stat().st_size<=8*2**20,"Finite exact model/optimizer checkpoint")
        # Scientific checkpoint writer SHA is scoped and authorized; never a review-time payload read.
        sha=hashlib.sha256()
        with path.open("rb")as stream:
            while True:
                block=stream.read(2**20)
                if not block:break
                sha.update(block)
        return dict(path=str(path.relative_to(c.ROOT)),sha256=sha.hexdigest())
    # Metadata qualification of every registered observation precedes target resolver construction.
    for parent_id,value in q["parents"].items():
        observed=c.read(value["observation_ref"]);prepared=c.read(value["native_preparation_ref"])
        c.require(observed["parent_id"]==prepared["parent_id"]==parent_id
            and observed["sequence_length_bp"]==30000000 and observed["split"]=="train"
            and observed["query_ancestry_truth_in_inputs"]is False
            and prepared["observation_ref"]==value["observation_ref"]and prepared["focals_ref"]==value["focal_registry_ref"]
            and prepared["query_ancestry_truth_opened"]is False,"Every original full30Mb TRAIN observation before truth")
    barrier=c.read(q["observation_barrier_ref"])if q["phase"]=="full_training"else None
    target_join=TrainTargetJoinV1(lambda p:q["train_truth_refs"][p],
        max_truth_uncompressed_bytes=8*2**30,train_truth_refs=q["train_truth_refs"],allowed_parent_ids=tuple(q["parents"]))
    seals={};plans={}
    def load_parent(parent_id):
        tick("before_full_parent_observations")
        item=q["parents"][parent_id]
        cfg=dict(native_preparation_ref=item["native_preparation_ref"],class_order=["EUR","EAS","AFR"],
            expected_haplotypes=1600,reference_counts=[200,200,200],input_reference_class_mask=None,
            input_reference_sample_indices=None)
        state=load_observation(item["observation_ref"],item["focal_registry_ref"],cfg)
        bundle,meta=preparation.prepare_from_loaded_observation_v1(state,split="train",
            observation_ref=item["observation_ref"],focal_registry_ref=item["focal_registry_ref"],
            native_preparation_ref=item["native_preparation_ref"],max_nodes=400000000,max_holds=100000,max_episode_payload_bytes=8*2**30)
        del state
        c.require(bundle.sequence_length_bp==30000000 and bundle.native_holds<=100000
            and bundle.parent_id==parent_id and bundle.original_sites>0,"Whole original30Mb parent; bounds never crop")
        if barrier is not None:
            seal=barrier["parents"][parent_id]
            c.require(seal["observation_only"]is True and seal["all_native_trees_nodes_sites_samples_retained"]is True
                and seal["observation_ref"]==bundle.observation_ref and seal["focal_registry_ref"]==bundle.focal_registry_ref
                and seal["native_preparation_ref"]==bundle.native_preparation_ref
                and seal["focal_identity_sha256"]==bundle.focal_identity_sha256,"Actual all20 barrier exact bundle identity")
        if parent_id not in plans:
            plan=dict(parent_id=parent_id,sequence_length_bp=30000000,focal_registry_ref=bundle.focal_registry_ref,
                observation_ref=bundle.observation_ref,native_preparation_ref=bundle.native_preparation_ref,
                original_sites=bundle.original_sites,native_holds=bundle.native_holds,
                full_global_recipe_metadata_sha256=hashlib.sha256(c.encoded(meta)).hexdigest(),
                all_native_trees_nodes_sites_samples_retained=True,observation_only=True,private_truth_opened=False)
            pr=c.put(q,"plan."+parent_id+".json",plan);plan["plan_ref"]=pr;plans[parent_id]=plan
            seals[parent_id]=dict(parent_id=parent_id,observation_only=True,all_blocks_sealed_before_truth=True,
                plan_ref=pr,focal_identity_sha256=bundle.focal_identity_sha256)
        return bundle,meta
    if q["phase"]=="qualification":
        from .parity_v1 import run_parity_v1
        parity=run_parity_v1(resource_check=tick)
        c.put(q,"synthetic_parity.json",parity,directory=output)
        torch.cuda.empty_cache()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(70001);model=frozen.AmericanTreeLoomC3FullV1()
    model=model.to(device="cuda:0",dtype=torch.float32)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001,betas=(.9,.999),eps=1e-8,
        maximize=False,amsgrad=False,foreach=False,fused=False,capturable=False,differentiable=False)
    c.require(not optimizer.state,"Fresh disposable qualification or fresh independent600update optimizer")
    steps=0;refs={};epochs=(1,)if q["phase"]=="qualification"else range(1,31)
    for epoch in epochs:
        order=["american_c3_train_000"]if q["phase"]=="qualification"else epoch_order_v1(epoch)
        for parent_id in order:
            bundle,meta=load_parent(parent_id)
            targets=target_join.join(parent_id,split="train",plan=plans[parent_id],observation_seal=seals[parent_id])
            result=train_step_v1(model,bundle,targets,optimizer,meta,resource_check=tick)
            steps+=1
            counts=[int(row["step"].item())for row in optimizer.state.values()]
            c.require(len(counts)==49 and set(counts)=={steps},"All49genuine scheduled AdamW counters")
            event("technical_check_update"if q["phase"]=="qualification"else"complete_parent_update",
                dict(epoch=epoch,optimizer_updates=steps,**result))
            del bundle,meta,targets
        if q["phase"]=="qualification"or epoch in(5,10,15,20,25,30):
            payload=dict(schema="TreeLoom_bounded_frozen_checkpoint_v1",phase=q["phase"],epoch=epoch,complete_parent_updates=steps,
                seed=70001,contract=model.checkpoint_contract(),config=c.read(q["config_ref"]),
                model_state=_cpu_snapshot(model.state_dict()),optimizer_state=_cpu_snapshot(optimizer.state_dict()),
                VAL_truth_read=False,TEST_truth_read=False,checkpoint_selection="fixed_final_epoch30_before_scored_benchmark_truth")
            refs[str(epoch)]=checkpoint(epoch,payload);del payload
    c.require(steps==(1 if q["phase"]=="qualification"else 600),"Exact disposable1step or fixed600updates")
    torch.cuda.synchronize(0);tick("before_completion")
    outcome=dict(schema="TreeLoom_bounded_frozen_worker_completion_v1",
        status="completed_disposable_TRAIN000_technical_optimizer_check"if q["phase"]=="qualification"else"completed_frozen_20TRAIN_30passes_600updates",
        phase=q["phase"],complete_parent_updates=steps,checkpoints=refs,input_ref=input_ref,
        native_focals_per_parent=10000,sequence_length_bp=30000000,haplotypes=1600,
        frozen_architecture="H050_stable_leaf_memory",trainable_tensors=49,parameters=32132,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(0),
        peak_cpu_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,elapsed_seconds=time.monotonic()-start,
        actual_full_process_title=title,latest_physical_GPU0_capacity_and_title=latest[0],
        qualification_weights_are_not_final_training_weights=q["phase"]=="qualification",
        disposable_qualification_uses_genuine_TRAIN000_labels=q["phase"]=="qualification",
        prior_disposable_TRAIN000_label_access_disclosed=True,
        all20_observations_before_this_full_training_truth=q["phase"]=="full_training",
        qualification_warmstart_or_optimizer_or_rng_reuse=False,
        benchmark_result=False,ancestry_accuracy=None,VAL_truth_read=False,TEST_truth_read=False,
        genuine_parent_wait_and_ROOT_outer_consumption_still_required=True,automatic_close_or_credit=False)
    c.put(q,"completed.json",outcome,directory=output)
    return outcome
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--input-ref");p.add_argument("--ticket-ref");p.add_argument("--gate-fd",type=int)
    a=p.parse_args(argv)
    if a.input_ref is None:print(json.dumps(dict(status="SOURCE_ONLY_INACTIVE",operations=0)));return 0
    c.require(a.ticket_ref is not None and a.gate_fd is not None,"Actual parent ticket/gate required")
    print(json.dumps(run_v1(json.loads(a.input_ref),json.loads(a.ticket_ref),a.gate_fd),sort_keys=True),flush=True);return 0
if __name__=="__main__":raise SystemExit(main())
