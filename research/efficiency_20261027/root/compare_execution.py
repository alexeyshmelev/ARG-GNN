"""ROOT-admitted paired GPU diagnostic; no launcher or production checkpoint writer."""
from pathlib import Path
from contextlib import contextmanager,nullcontext
from dataclasses import dataclass,fields
from types import SimpleNamespace
import argparse,copy,gc,hashlib,json,os,resource,time

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
CHECKPOINT="research/preprocessing_optimization_20261026/integration/migration/update241.pt"
JOB=".runtime/benchmarking/paper_campaign_20261024_v2/jobs/paper_american_treeloom_fit.attempt2.json"
PACKET="research/scalability_20261024/execution/representative_slice.npz"
GIB=2**30

def need(ok,message):
    if not ok:raise ValueError(message)

def ref(path):
    p=Path(path);p=p if p.is_absolute() else ROOT/p
    need(p.is_file() and not p.is_symlink(),"Ordinary immutable source required")
    h=hashlib.sha256()
    with p.open("rb") as stream:
        for block in iter(lambda:stream.read(2**20),b""):h.update(block)
    return dict(path=str(p.relative_to(ROOT)),sha256=h.hexdigest())

def rss():return int(Path("/proc/self/statm").read_text().split()[1])*os.sysconf("SC_PAGE_SIZE")

def write(path,value):
    with path.open("x") as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write("\n")

class Profile:
    def __init__(self,torch,check):self.torch=torch;self.check=check;self.timings={};self.marks={}
    @contextmanager
    def stage(self,name):
        top=name.startswith("profile_")
        if top and self.torch.cuda.is_initialized():self.torch.cuda.synchronize()
        tick=time.perf_counter()
        if top:print(json.dumps(dict(event="stage_begin",stage=name)),flush=True)
        try:yield
        finally:
            if top and self.torch.cuda.is_initialized():self.torch.cuda.synchronize()
            elapsed=time.perf_counter()-tick
            row=self.timings.setdefault(name,dict(calls=0,total_seconds=0.,max_seconds=0.))
            row["calls"]+=1;row["total_seconds"]+=elapsed;row["max_seconds"]=max(row["max_seconds"],elapsed)
            if top:print(json.dumps(dict(event="stage_complete",stage=name,seconds=elapsed)),flush=True)
    def mark(self,name,**values):self.marks[name]=self.marks.get(name,0)+1
    def snapshot(self):return dict(stage_timings=self.timings,mark_counts=self.marks)

@dataclass
class DiagnosticSliceBundle:
    bank:object;episodes:object;cut:object;parent_id:str;split:str;sequence_length_bp:int
    native_holds:int;original_sites:int;focal_hold:object;focal_sample_column:object
    focal_identity_sha256:str;observation_ref:dict;focal_registry_ref:dict;native_preparation_ref:dict
    bank_stamps:tuple;episode_stamps:tuple;focal_stamps:tuple
    def validate(self):
        from research.benchmark_rebuild_20261024.american.treeloom_speed import static_model_v2 as f
        from research.reviews.order2_cut_memory_implementation_20261015_v1 import order2_cut_memory_v1 as core
        need(self.native_holds==128 and self.bank.sample_index.shape==(128,1600),"Exact diagnostic128x1600 bank")
        need(self.bank_stamps==core._stamps(f._bank_tensors(self.bank)),"Immutable diagnostic bank")
        need(self.episode_stamps==core._stamps(f._episode_tensors(self.episodes)),"Immutable diagnostic episodes")
        need(self.focal_stamps==core._stamps((self.focal_hold,self.focal_sample_column)),"Immutable actual focal subset")
        self.cut.validate(self.bank)
        return self.bank

def prepare(args,rt,saved,check):
    import numpy as np
    import torch
    from arggnn.benchmarking import contracts as c
    from research.benchmark_rebuild_20261024.american.treeloom_speed import runtime_v2 as fresh
    job=json.loads((ROOT/JOB).read_text());original=job["call"]["spec"]["arguments"]
    parent="argformer_american_train_000";item=original["train_inputs"][parent]
    need(rt.registry.split_for(parent)=="train","Only the actual registered TRAIN parent")
    barrier=c.read_ref(saved["observation_barrier_ref"]);seal=barrier["parents"][parent]
    plan=dict(c.read_ref(seal["plan_ref"]),plan_ref=seal["plan_ref"])
    for key in fresh.INPUT_REFS:need(item[key]==seal[key],"Checkpoint barrier binds exact observed parent")
    started=time.perf_counter()
    print(json.dumps(dict(event="observation_preparation_begin",mode=args.mode)),flush=True)
    if args.mode=="full":
        from arggnn.american_c3_gnn_dataset_v1 import load_observation
        state=load_observation(item["observation_ref"],item["focal_registry_ref"],dict(
            native_preparation_ref=item["native_preparation_ref"],class_order=list(fresh.CLASSES),
            expected_haplotypes=1600,reference_counts=[200,200,200],
            input_reference_class_mask=None,input_reference_sample_indices=None))
        bundle,metadata=rt.prepare_observation(state,split="train",resource_check=check,
            **{k:item[k] for k in fresh.INPUT_REFS},max_nodes=400000000,max_holds=100000,max_episode_payload_bytes=8*GIB)
        del state;gc.collect();frozen=rt.frozen;selected=None
    else:
        from arggnn.mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeInputsV1
        from research.repairs.benchmark_recovery_20261020.runtime.episodes import build_episodes
        from research.repairs.benchmark_recovery_20261020.runtime.cuts import prepare_cut_plan_v1
        from research.reviews.order2_cut_memory_implementation_20261015_v1 import order2_cut_memory_v1 as core
        meta=json.loads((ROOT/Path(PACKET).with_suffix(".json")).read_text())
        need(ref(PACKET)==meta["packet_ref"] and meta["truth_opened"] is False,"Authenticated observed-only packet")
        need(meta["native_hold_range"]==[0,128] and meta["source_manifest_ref"]==item["observation_ref"],"Exact original source slice")
        z=np.load(ROOT/PACKET,allow_pickle=False);arrays={}
        for field in fields(MutationRoutedWholeTreeInputsV1):
            name=field.name
            if name in ("upward_levels","upward_children","upward_parent_rank","downward_levels"):
                keys=sorted((k for k in z.files if k.startswith(name+"_")),key=lambda k:int(k.rsplit("_",1)[1]))
                arrays[name]=tuple(z[k] for k in keys)
            else:arrays[name]=z[name]
        bank=MutationRoutedWholeTreeInputsV1.from_arrays(arrays)
        offsets=z["node_offsets"];counts=np.diff(offsets)
        isolated=tuple(i for i,(a,b) in enumerate(zip(offsets,offsets[1:])) if b-a==1600 and bool((bank.parent[int(a):int(b)]<0).all()))
        episodes,episode_metadata=build_episodes(bank,max_payload_bytes=GIB,isolated_hold_indices=isolated)
        cut,cut_metadata=prepare_cut_plan_v1(bank,expected_haplotypes=1600,max_nodes=400000,max_holds=128,isolated_hold_indices=isolated)
        need(ref(item["focal_registry_ref"]["path"])==item["focal_registry_ref"],"Exact original focal file before subset selection")
        f=np.load(ROOT/item["focal_registry_ref"]["path"],allow_pickle=False)
        positions=f["positions_bp0"];intervals=z["intervals"]
        selected=(positions>=intervals[0,0])&(positions<intervals[-1,1])
        need(bool(selected.any()),"No true fixed focals in slice; do not substitute fabricated labels")
        hold=torch.from_numpy(np.searchsorted(intervals[:,1],positions[selected],side="right").astype(np.int64))
        column=torch.from_numpy(f["sample_indices"][selected].astype(np.int64))
        bundle=DiagnosticSliceBundle(bank,episodes,cut,parent,"train",int(intervals[-1,1]-intervals[0,0]),128,len(bank.genotype),
            hold,column,seal["focal_identity_sha256"],item["observation_ref"],item["focal_registry_ref"],item["native_preparation_ref"],
            core._stamps(rt.frozen._bank_tensors(bank)),core._stamps(rt.frozen._episode_tensors(episodes)),core._stamps((hold,column)))
        metadata=dict(parent_id=parent,declared_split="train",parent_registry_ref=rt.registry.reference,
            episode_metadata=dict(episode_metadata,native_nodes_per_tree=counts.tolist()),
            cut_metadata=cut_metadata,layout_cache_scope=dict(kind="diagnostic_slice",parent_id=parent,holds=[0,128],source_site_range=[0,681]),
            observation_ref=item["observation_ref"],focal_registry_ref=item["focal_registry_ref"],native_preparation_ref=item["native_preparation_ref"],
            diagnostic_slice=True,private_truth_opened=False)
        frozen=SimpleNamespace(AmericanTreeLoomC3FullV1=rt.frozen.AmericanTreeLoomC3FullV1,
            AmericanTreeLoomWholeParentV1=DiagnosticSliceBundle,_bank_tensors=rt.frozen._bank_tensors,_episode_tensors=rt.frozen._episode_tensors)
        bundle.validate();del z,arrays
    preparation_seconds=time.perf_counter()-started;check("observations_prepared_before_truth")
    print(json.dumps(dict(event="observation_preparation_complete_before_truth",seconds=preparation_seconds)),flush=True)
    tick=time.perf_counter()
    join=fresh.FreshTrainTargetJoin(rt.registry,lambda p:original["train_inputs"][p]["truth_ref"],
        train_inputs=original["train_inputs"],max_truth_uncompressed_bytes=original["max_truth_uncompressed_bytes"])
    target=join.join(parent,split="train",plan=plan,observation_seal=seal)
    labels=target.labels_cpu if selected is None else target.labels_cpu[torch.from_numpy(selected)].clone()
    scope=dict(mode=args.mode,parent_id=parent,native_holds=bundle.native_holds,haplotypes=1600,
        sites=bundle.original_sites,supervised_original_focals=len(labels),full_original_focals=10000,
        observation_ref=item["observation_ref"],focal_registry_ref=item["focal_registry_ref"],truth_ref=item["truth_ref"],
        focal_identity_sha256=bundle.focal_identity_sha256,preparation_seconds=preparation_seconds,
        true_TRAIN_label_join_seconds=time.perf_counter()-tick,heldout_truth_opened=False,
        full_parent_claim=args.mode=="full",history_window_boundary="original full parent" if args.mode=="full" else "restricted128hold window; future history outside window absent",
        physical_support_bp=[0,bundle.sequence_length_bp],packet_ref=ref(PACKET) if args.mode=="slice" else None)
    return bundle,metadata,labels,frozen,scope

def run_one(kind,phase,repeat,args,rt,saved,bundle,metadata,labels,frozen,cache,check):
    import torch
    from research.efficiency_20261027.execution.executors import build_executor
    from research.benchmark_rebuild_20261024.american.treeloom_speed.decoder_v2 import decode_parent_backward,geometry_from_spans
    from research.repairs.TreeLoom_lazy_compact_prefetch_20261024_v1.training import restore_random_state
    profile=Profile(torch,check);torch.cuda.reset_peak_memory_stats();started=time.perf_counter();logits=[];executor=None
    with profile.stage("profile_model_optimizer_transfer"):
        model=rt.build_model();model.load_state_dict(saved["model_state"],strict=True);model=model.to(device="cuda:0",dtype=torch.float32)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001,betas=(.9,.999),eps=1e-8,
            maximize=False,amsgrad=False,foreach=False,fused=False,capturable=False,differentiable=False)
        optimizer.load_state_dict(copy.deepcopy(saved["optimizer_state"]));restore_random_state(saved["random_state"])
    hook=model.classifier.register_forward_hook(lambda module,inputs,output:logits.append(output.detach().cpu()))
    trace=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,profile_memory=False,with_stack=False) if args.trace_transfers else nullcontext()
    try:
        with trace as captured:
            model.train();optimizer.zero_grad(set_to_none=True)
            with profile.stage("profile_layout"):
                executor=build_executor(kind,model,bundle,metadata,execution="staged",layout_cache=cache,
                    frozen_module=frozen,holds_per_block=args.holds_per_block,haplotypes_per_gru=args.haplotypes_per_gru,
                    history_holds_per_window=1024,resource_check=check,profile=profile,
                    **({"max_sweep_plan_bytes":(64*2**20 if args.mode=="slice" else 28*GIB),
                        "sweep_plan_admit":lambda amount,stage:need(rss()+amount<=args.max_rss_gib*GIB,"Sweep plan RAM unavailable at "+stage)} if kind=="level_local" else {}))
            with torch.autocast(device_type="cuda",enabled=False):
                with profile.stage("profile_context_forward"):context=executor.context_forward()
                with profile.stage("profile_decoder_forward_backward"):
                    loss,vjp,decoder=decode_parent_backward(model,context,geometry_from_spans(bundle.bank.spans),
                        bundle.focal_hold,bundle.focal_sample_column,labels,tile_haplotypes=args.haplotypes_per_gru,profile=profile,resource_check=check)
                with profile.stage("profile_trunk_backward"):context.backward(vjp)
            gradients={name:p.grad.detach().cpu().clone() for name,p in model.named_parameters()}
            need(len(gradients)==49 and all(bool(torch.isfinite(g).all()) for g in gradients.values()),"All49 finite connected actual gradients")
            with profile.stage("profile_clip_optimizer"):
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            check("profile_update_complete")
        parameters={name:p.detach().cpu().clone() for name,p in model.named_parameters()}
        moments={str(i):{k:v.detach().cpu().clone() if torch.is_tensor(v) else v for k,v in row.items()} for i,row in optimizer.state_dict()["state"].items()}
        need(len(moments)==49 and all(int(row["step"].item())==242 for row in moments.values()),"Diagnostic241to242 update preserves all49 Adam counters")
        copies=[];operators=[]
        if args.trace_transfers:
            averages=captured.key_averages()
            for row in sorted(averages,key=lambda item:getattr(item,"device_time_total",0.),reverse=True)[:25]:
                operators.append(dict(name=row.key,calls=row.count,cpu_self_us=row.self_cpu_time_total,cpu_total_us=row.cpu_time_total,device_self_us=getattr(row,"self_device_time_total",0.),device_total_us=getattr(row,"device_time_total",0.)))
            for row in averages:
                if "memcpy" in row.key.lower() or row.key in ("aten::_to_copy","aten::copy_"):
                    copies.append(dict(name=row.key,calls=row.count,cpu_total_us=row.cpu_time_total,
                        device_total_us=getattr(row,"device_time_total",0.)))
        receipt=dict(kind=kind,phase=phase,repeat=repeat,seconds=time.perf_counter()-started,
            loss=float(loss),gradient_norm=float(norm),decoder=decoder,stages=profile.snapshot(),
            torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            RSS_bytes=rss(),process_lifetime_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            copy_events=copies,top_operator_events=operators,transfer_trace_enabled=args.trace_transfers,
            cache_snapshot=cache.snapshot() if cache is not None else None,
            sweep_plan_snapshot=executor.sweep_plan_snapshot() if hasattr(executor,"sweep_plan_snapshot") else None)
        numerical=dict(logits=torch.cat(logits),gradients=gradients,parameters=parameters,moments=moments)
        return receipt,numerical
    finally:
        hook.remove()
        if "context" in locals():del context
        if "vjp" in locals():del vjp
        optimizer.zero_grad(set_to_none=True)
        if executor is not None and hasattr(executor,"release_layout"):executor.release_layout()
        del executor,model,optimizer;gc.collect();torch.cuda.empty_cache()

def compare(reference,candidate,atol=1e-4,rtol=1e-4):
    import torch
    errors={}
    for category in ("logits","gradients","parameters","moments"):
        maximum=0.
        def visit(a,b):
            nonlocal maximum
            if isinstance(a,dict):
                need(set(a)==set(b),"Identical numerical state keys")
                for key in a:visit(a[key],b[key])
            elif torch.is_tensor(a):
                torch.testing.assert_close(a,b,atol=atol,rtol=rtol)
                if a.numel():maximum=max(maximum,float((a-b).abs().max()))
            else:need(a==b,"Identical scalar numerical state")
        visit(reference[category],candidate[category]);errors[category+"_max_abs_error"]=maximum
    return dict(passed=True,atol=atol,rtol=rtol,**errors)

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument("--mode",choices=("slice","full"),required=True);p.add_argument("--output",required=True)
    p.add_argument("--holds-per-block",type=int,default=32);p.add_argument("--haplotypes-per-gru",type=int,default=4)
    p.add_argument("--gpu-memory-gib",type=float,default=4);p.add_argument("--max-rss-gib",type=float,default=16)
    p.add_argument("--warm-repeats",type=int,default=1);p.add_argument("--trace-transfers",action="store_true")
    args=p.parse_args();need(0<=args.warm_repeats<=3,"Bounded explicit warm repeats")
    need(os.environ.get("ARGGNN_PHYSICAL_GPU") in ("3","4","5") and os.environ.get("CUDA_VISIBLE_DEVICES","").startswith("GPU-")
        and "," not in os.environ["CUDA_VISIBLE_DEVICES"] and os.environ.get("ARGGNN_GPU_MEMORY_POLICY")=="full_device","ROOT-selected one GPU3/4/5 UUID and full-device policy required")
    import setproctitle
    setproctitle.setproctitle("AI Auto Research Pipeline Test")
    import torch
    from research.benchmark_rebuild_20261024.american.treeloom_speed import runtime_v2 as fresh,training_v2 as validation
    from research.scalability_20261024.integration.cache import LayoutCache
    torch.set_num_threads(4 if args.mode=="slice" else 1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    output=Path(args.output).resolve();need(output.is_relative_to(ROOT) and not output.exists(),"Fresh project-contained profile output required")
    output.mkdir(parents=True)
    def check(stage):
        need(rss()<=args.max_rss_gib*GIB,"Profile RSS guard exceeded at "+stage)
        if torch.cuda.is_initialized():need(torch.cuda.memory_reserved()<=args.gpu_memory_gib*GIB,"Soft GPU reserved-memory budget exceeded at "+stage)
    source_paths=[Path(__file__),ROOT/"research/benchmark_rebuild_20261024/american/treeloom_speed/source_pins.json",ROOT/"research/scalability_20261024/categorical_site/categorical_site.py",ROOT/"research/scalability_20261024/execution/layout_plan.py"]
    source_paths+=sorted((ROOT/"research/scalability_20261024/integration").glob("*.py"))
    source_paths+=sorted((ROOT/"research/efficiency_20261027/execution").glob("*.py"))
    source_refs=[ref(path) for path in source_paths]
    upstream=ROOT/"research/scalability_20261024/integration/artifact_manifest.json"
    upstream_manifest_ref=ref(upstream) if upstream.exists() else None
    started=time.perf_counter()
    need(ref(CHECKPOINT)["sha256"]=="12b8fbdd4bfa164ca3aab28d967d1154a3ce389cfc49aefca0fbdf9dc143de24","Exact immutable241 checkpoint")
    saved=torch.load(ROOT/CHECKPOINT,map_location="cpu",weights_only=True,mmap=True)
    registry=fresh.FreshParentRegistry.from_ref(saved["parent_registry_ref"]);rt=fresh.FreshTreeLoom(registry)
    validation.validate_saved_state(saved,rt.build_model(),registry,expected_steps=241)
    if args.mode=="full":
        from research.preprocessing_optimization_20261026.integration.hooks import install_tree
        install_tree(preparation_workers=4,max_rss_bytes=int(args.max_rss_gib*GIB))
    bundle,metadata,labels,frozen,scope=prepare(args,rt,saved,check)
    cache_bytes=64*2**20 if args.mode=="slice" else 8*GIB
    def admit(working_bytes,stage):need(rss()+working_bytes<=args.max_rss_gib*GIB,"Cache working reserve unavailable at "+stage)
    cache=LayoutCache(max_bytes=cache_bytes,admit=admit)
    records=[];baseline=None
    for phase,repeat,kinds in [("same_process_first_call",0,("original","integrated","level_local"))]+[("warm_layout_and_libraries",r,(("level_local","integrated","original") if r%2 else ("original","integrated","level_local"))) for r in range(1,args.warm_repeats+1)]:
        for kind in kinds:
            row,numerical=run_one(kind,phase,repeat,args,rt,saved,bundle,metadata,labels,frozen,cache if kind!="original" else None,check)
            if baseline is None:baseline=numerical;row["comparison"]=dict(reference=True)
            else:row["comparison"]=compare(baseline,numerical)
            records.append(row);write(output/(kind+"_"+phase+"_"+str(repeat)+".json"),row)
    need([ref(path) for path in source_paths]==source_refs,"Profile source changed during run")
    result=dict(schema="TreeLoom_three_arm_efficiency_profile_v1",status="completed",scope=scope,
        checkpoint_ref=ref(CHECKPOINT),source_ref=ref(__file__),source_refs=source_refs,upstream_integration_manifest_ref=upstream_manifest_ref,execution_partitions=dict(holds_per_block=args.holds_per_block,haplotypes_per_gru=args.haplotypes_per_gru,history_window=1024),
        FP32=True,physical_gpu=os.environ["ARGGNN_PHYSICAL_GPU"],visible_UUID=os.environ["CUDA_VISIBLE_DEVICES"],
        max_rss_bytes=args.max_rss_gib*GIB,soft_GPU_reserved_budget_bytes=args.gpu_memory_gib*GIB,hard_allocator_cap=False,
        external_guardian_required=True,records=records,total_seconds=time.perf_counter()-started,
        caveats=["Same-process first-call timings are not independent process-cold GPU measurements; common CPU observation preparation and label join are recorded once.",
            "Warm runs restore the same241 weights/moments/RNG and reuse the same immutable prepared bank; diagnostic updates are not committed to training.",
            "Copy event timings are overlapping nested diagnostics, not additive phases. Optional torch profiler introduces overhead.",
            "Slice mode retains actual22 fixed focal labels on restricted128hold support and has truncated outside-window history; it is not a full-parent training/update or accuracy result.",
            "No production checkpoint, simulation, input, held-out truth or live source is changed."])
    write(output/"completed.json",result);print(json.dumps(dict(status="completed",completion_ref=ref(output/"completed.json"),scope=scope),indent=2))

if __name__=="__main__":main()
