"""H047 paired loss-only H042 continuation. Inert without root context."""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import fields
import hashlib,io,json,math,random,statistics,time
from pathlib import Path
from . import train_anonymous_clade_flag_control_v1 as h42

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE="src/arggnn/train_cohort_risk_continuation_v1.py"
ARMS=("original_risk","balanced_risk")
TRAIN,VAL,PASSES=h42.TRAIN,h42.VAL,h42.PASSES
MODEL,SELECTION,METRICS=h42.MODEL,h42.SELECTION,h42.METRICS
CORE_REF=dict(path="src/arggnn/cohort_risk_continuation_v1.py",sha256="d76094d9363ef2c4452b497b0949ea56a8c731f82d0f02aa32d607d889ffb9ef")
PROPOSAL_REF=dict(path="research/reviews/H047_biology_novelty_proposal_v1/proposal_spec.json",
    sha256="46190af8de6ce9eb6c1244e35756c53c68cd7599498b47814002aaf9b95c67eb")
REGISTRATION_PATH="research/reviews/H047_GPU45_cohort_risk_root_source_v1/registration_root_v1.json"
WARM_CHECKPOINT=dict(path="experiments/results/H042_GPU4_clade_flags_train_seed101_v1/exact_clade_flags.pass_30.pt",
    sha256="520fcffa1d5e85603620a7f4688b7fdf46669686fea7960fc311852ff2f7e366")
WARM_COMPLETION=dict(path="experiments/results/H042_GPU4_clade_flags_train_seed101_v1/completed.json",
    sha256="5b7d7adab0de14c18016db86bfc18ddc2b1d652ff530fadd101aaa73c90a6984")
SCIENTIFIC_REFS=(CORE_REF,dict(path=h42.SOURCE,sha256="cb58ec3f81d81cdcd88637f3ade695a3e3ef97a2d0c10b7713e9bddb43fbb2ed"),
    *h42.SCIENTIFIC_REFS,
    dict(path="src/arggnn/mutation_routed_full_panel_primary_v1.py",sha256="a94fd5af1cd10b1605f75621c36a130faa3f9d5ad71068f8b187d8e73c7e5374"),
    dict(path="src/arggnn/mutation_routed_full_panel_fast_reader_v1.py",sha256="d6b963dc14e1f6752e310e627faf2e4f9a2e408fcb7dab17e3439771ab227e03"))
OBJECTIVES={"original_risk":"(182/686)*LP+(504/686)*LA","balanced_risk":"0.5*LP+0.5*LA"}
RESOURCE_CAPS=("CPU_bank_cap_bytes","CPU_target_cap_bytes","flag_cache_cap_bytes",
    "flag_preparation_cap_bytes","GPU_residency_cap_bytes","GPU_flag_cap_bytes","RAM_prediction_cap_bytes")
require=h42.require


def science():
    global np,torch,h038,fit,core
    h42.science()
    np,torch,h038,fit=h42.np,h42.torch,h42.h038,h42.fit
    from . import cohort_risk_continuation_v1 as core


def validate_protocol(cfg,*,require_registration=True):
    require(cfg.get("schema")=="H047_cohort_risk_continuation_execution_config_v1"
        and cfg.get("hypothesis_id")=="H047" and cfg.get("arms")==list(ARMS)
        and cfg.get("linked_hypothesis_ids")==["H042","H043","H044","H045"]
        and cfg.get("seed")==101 and cfg.get("dtype")=="float32"
        and cfg.get("passes")==30 and cfg.get("updates_per_pass")==32 and cfg.get("updates_per_arm")==960
        and cfg.get("checkpoint_passes")==list(PASSES)
        and cfg.get("TRAIN_parent_ids")==list(TRAIN) and cfg.get("VAL_parent_ids")==list(VAL)
        and cfg.get("model")==MODEL and cfg.get("parameters_per_arm")==30788
        and cfg.get("base_parameters")==30788 and cfg.get("added_parameters")==0
        and cfg.get("all_H042_weights_trainable") is True and cfg.get("objectives")==OBJECTIVES
        and cfg.get("objective_group_normalization")=="actual_original_represented_group_query_bp"
        and cfg.get("training_target_policy")=="original_grid_mass_aggregated_native_holds_once_shared_all32_retained"
        and cfg.get("execution_mode")=="paired_sequential_one_parent"
        and cfg.get("parent_order")=="shuffle_seed101_plus_pass"
        and cfg.get("learning_rate")==.001 and cfg.get("weight_decay")==.0001
        and cfg.get("AdamW_betas")==[.9,.999] and cfg.get("AdamW_epsilon")==1e-8
        and cfg.get("gradient_clip_norm")==1. and cfg.get("early_stopping") is False
        and cfg.get("first_step_qualification")=="within_continuous_training_no_extra_updates"
        and cfg.get("VAL_influences_optimization") is False and cfg.get("H046_scores_used") is False
        and cfg.get("same_VAL_is_development_evidence") is True
        and cfg.get("selection")==SELECTION and cfg.get("metrics")==METRICS
        and cfg.get("warmstart_checkpoint_ref")==WARM_CHECKPOINT
        and cfg.get("warmstart_completion_ref")==WARM_COMPLETION
        and cfg.get("core_ref")==CORE_REF and cfg.get("scientific_refs")==list(SCIENTIFIC_REFS)
        and cfg.get("proposal_ref")==PROPOSAL_REF
        and cfg.get("practical_accuracy_gain_threshold")==.001
        and cfg.get("practical_boundary_F1_loss_tolerance")==.01
        and cfg.get("practical_criteria_aggregation")=="equal_parent_mean_Present182_across8VAL"
        and cfg.get("flag_arrays_saved_to_disk") is False and cfg.get("dense_prediction_arrays_saved") is False
        and cfg.get("outer_truth_read") is False and cfg.get("novel_architecture_claim") is False,
        "Exact zero-new-parameter H047 matched loss-only protocol")
    for name in RESOURCE_CAPS:
        require(type(cfg.get(name)) is int and cfg[name]>0,"Positive root-reviewed cap: "+name)
    r=cfg.get("registration_ref")
    if require_registration:
        require(isinstance(r,dict) and set(r)=={"path","sha256"} and r["path"]==REGISTRATION_PATH
            and isinstance(r["sha256"],str) and len(r["sha256"])==64
            and all(c in "0123456789abcdef" for c in r["sha256"]),"Exact root registration before execution")


def tensor_stamps_v1(bank):
    rows=[]
    for f in fields(bank):
        value=getattr(bank,f.name)
        for t in (value if isinstance(value,tuple) else (value,)):
            rows.append((f.name,id(t),t.data_ptr(),t._version,tuple(t.shape),t.dtype,str(t.device)))
    return tuple(rows)


def flag_stamp_v1(flag):
    t=flag.node_flags
    return id(t),t.data_ptr(),t._version,tuple(t.shape),t.dtype,str(t.device)


class FlagCacheV1:
    """All content hashes once per preparation; O(tensor-count) stamps on hits."""
    def __init__(self,ctx):
        self.ctx,self.cfg,self.guard=ctx,ctx["cfg"],ctx["guard"]
        self.values,self.proofs,self.bank_stamps,self.flag_stamps=OrderedDict(),{},{},{}
        self.bytes=self.peak=self.builds=self.hits=0

    def get(self,pid,bank,keys):
        self.guard()
        stamp=tensor_stamps_v1(bank)
        if pid in self.bank_stamps:require(stamp==self.bank_stamps[pid],"Immutable original CPU bank changed")
        else:self.bank_stamps[pid]=stamp
        if pid in self.values:
            flag=self.values[pid]
            require(flag_stamp_v1(flag)==self.flag_stamps[pid],"Immutable CPU flags changed")
            self.values.move_to_end(pid);self.hits+=1;return flag
        declared=h42.flag_preparation_plan(int(bank.parent.numel()),int(bank.sample_index.shape[1]))
        self.ctx["before_flag_build"](pid,bank,declared)
        require(declared["total_planning_allowance_bytes"]<=self.cfg["flag_preparation_cap_bytes"],
            "Flag preparation planning cap")
        from .anonymous_clade_flag_control_v1 import build_clade_flags_v1
        flag,meta=build_clade_flags_v1(bank)
        proof=dict(metadata=meta,runtime_array_identity=h42.shared.array_identity(flag.node_flags.numpy()),
            stable_sample_keys_sha256=hashlib.sha256(json.dumps(keys,separators=(",",":")).encode()).hexdigest(),
            preparation_plan=declared)
        if pid in self.proofs:require(proof==self.proofs[pid],"Rebuilt observation recipe changed")
        else:self.proofs[pid]=proof
        size=flag.tensor_bytes();require(size<=self.cfg["flag_cache_cap_bytes"],"Complete flags exceed LRU cap")
        while self.values and self.bytes+size>self.cfg["flag_cache_cap_bytes"]:
            _,old=self.values.popitem(last=False);self.bytes-=old.tensor_bytes()
        self.values[pid]=flag;self.flag_stamps[pid]=flag_stamp_v1(flag)
        self.bytes+=size;self.peak=max(self.peak,self.bytes);self.builds+=1
        return flag


def load_observations_v1(ctx,reader,ids,cache):
    from .mutation_routed_full_panel_fast_reader_v1 import read_full_cache_index_v1,read_full_cache_parent_v1
    index=read_full_cache_index_v1(ctx["observation_cache_ref"],reader,registry=ctx["registry"],
        entry_refs=ctx["entry_refs"],source_refs=ctx["observation_source_refs"])
    banks,parents,keys,columns,total={},{},{},{},0
    for pid in ids:
        ctx["guard"]();item=read_full_cache_parent_v1(pid,index,reader)
        bank,parent=item["model_inputs"],item["scoring"]
        require(parent["split"]==("train" if pid in TRAIN else "val"),"Original observed phase split")
        total+=h038.tensor_bytes_v1(bank)
        require(total<=ctx["cfg"]["CPU_bank_cap_bytes"],"All phase CPU banks exceed cap")
        banks[pid],parents[pid]=bank,parent
        keys[pid]=h42.shared.stable_keys(bank,ctx["entry_refs"][pid],reader)
        p,a=core.cohort_columns_v1(parent["cohorts"]);columns[pid]=torch.from_numpy(p)
        cache.get(pid,bank,keys[pid])
        del item
    require(set(cache.proofs)==set(ids) and set(cache.values)==set(ids),
        "All phase recipes and retained flags before every target; no per-update rebuild")
    seal=ctx["output"].save("all_%d_H047_observation_recipes.json"%len(ids),dict(
        schema="H047_observation_barrier_v1",hypothesis_id="H047",parent_ids=list(ids),recipes=cache.proofs,
        observation_cache_ref=ctx["observation_cache_ref"],entry_refs={p:ctx["entry_refs"][p] for p in ids},
        cohort_counts=dict(core.COHORT_COUNTS),observed_cohort_masks_only=True,CPU_bank_tensor_bytes=total,
        all_observations_precede_phase_truth=True,flag_arrays_saved_to_disk=False,
        content_hashes_once_per_preparation_then_tensor_stamps=True))
    return banks,parents,keys,columns,seal


def load_warmstart_v1(reader):
    completed=reader.decode_json(WARM_COMPLETION)
    require(completed.get("schema")=="anonymous_clade_flag_train_completion_v1"
        and completed.get("hypothesis_id")=="H042" and completed.get("arm")=="exact_clade_flags"
        and completed.get("status")=="completed_fixed_30_pass_flag_TRAIN"
        and completed.get("steps")=={"exact_clade_flags":960} and len(completed.get("history",[]))==30
        and completed.get("checkpoint_groups",{}).get("30",{}).get("exact_clade_flags")==WARM_CHECKPOINT
        and completed.get("VAL_truth_read") is False and completed.get("outer_truth_read") is False,
        "Actual completed original H042 pass30 warmstart")
    return torch.load(io.BytesIO(reader.read(WARM_CHECKPOINT)),map_location="cpu",weights_only=False)


def build_family_v1(payload,device,guard):
    guard();family,proof=core.make_matched_family_v1(payload,**MODEL)
    hashes={a:h42.shared.state_identity(m.state_dict()) for a,m in family.items()}
    require(len(set(hashes.values()))==1,"Identical pure H042 initial states")
    for arm in ARMS:family[arm]=family[arm].to(device=device,dtype=torch.float32)
    optimizers={a:torch.optim.AdamW(family[a].parameters(),lr=.001,weight_decay=.0001,
        betas=(.9,.999),eps=1e-8) for a in ARMS}
    require(all(not o.state for o in optimizers.values()),"Independent reset AdamW")
    require(not ({p.data_ptr() for p in family[ARMS[0]].parameters()}&
        {p.data_ptr() for p in family[ARMS[1]].parameters()}),"Independent learned device storage")
    return family,optimizers,proof,hashes


def update_v1(model,optimizer,bank,mass,flags,present,arm,guard,*,qualify=False,after_optimizer_step=None):
    guard();model.train();optimizer.zero_grad(set_to_none=True)
    device=bank.parent.device
    if qualify and device.type=="cuda":
        torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
    started=time.monotonic()
    try:
        with torch.autocast(device_type=device.type,enabled=False):
            logits=model.forward_train_holds(bank,flags=flags,control="aligned")
            loss,metrics=core.cohort_objective_v1(logits,mass,present,arm=arm)
        values={k:float(v.detach()) if isinstance(v,torch.Tensor) else v for k,v in metrics.items()}
        del metrics
        loss.backward()
        require(all(p.grad is None or p.grad.dtype==torch.float32 and bool(torch.isfinite(p.grad).all())
            for p in model.parameters()),"Finite FP32 full-body gradients")
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        if after_optimizer_step is not None:after_optimizer_step()
        require(all(bool(torch.isfinite(p).all()) for p in model.parameters()),"Finite updated FP32 weights")
        require(all(not isinstance(v,torch.Tensor) or v.dtype==torch.float32 and bool(torch.isfinite(v).all())
            for state in optimizer.state.values() for v in state.values()),"Finite independent FP32 AdamW")
        row=dict(optimization_objective=float(loss.detach()),gradient_norm=float(norm),seconds=time.monotonic()-started,
            finite_FP32=True,**values)
        if qualify:
            if device.type=="cuda":
                torch.cuda.synchronize(device)
                row["memory"]=dict(real_GPU_qualification=True,allocated_bytes=int(torch.cuda.memory_allocated(device)),
                    reserved_bytes=int(torch.cuda.memory_reserved(device)),
                    peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                    peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)))
            else:row["memory"]=dict(real_GPU_qualification=False,device_type=device.type)
        del logits,loss,norm
    finally:optimizer.zero_grad(set_to_none=True)
    require(all(p.grad is None for p in model.parameters()),"Gradients end before next independent arm")
    guard();return row


def run_train_v1(ctx,input_ref):
    from .mutation_routed_full_panel_primary_v1 import FullPanelExternalTargetV1
    cfg,output,guard,device=ctx["cfg"],ctx["output"],ctx["guard"],ctx["device"]
    reader=h42.PhaseReader(ctx["reader"],"train",ctx["bindings"]);cache=FlagCacheV1(ctx)
    steps={a:0 for a in ARMS};history=[];checkpoints={};qualifications={}
    def record_step(arm):steps[arm]+=1
    try:
        banks,parents,keys,columns,observation_seal=load_observations_v1(ctx,reader,TRAIN,cache)
        reader.truth_gate=True
        targets,target_bytes,helpers={},0,fit.metric_helpers_v1()
        for pid in TRAIN:
            guard()
            target=FullPanelExternalTargetV1(parents[pid],ctx["bindings"][pid],"TRAIN_loss",reader)
            # Existing helper intersects original100bp bins with true intervals,
            # then aggregates exact physical class-span mass by native hold.
            hold_count=int(banks[pid].spans.numel())
            require(target_bytes+12*hold_count*686*4<=cfg["CPU_target_cap_bytes"],
                "Retained masses plus one-parent FP64/FP32 construction cap")
            mass=h038.native_hold_train_mass_v1(parents[pid],banks[pid],target,helpers)
            target_bytes+=int(mass.nbytes)
            require(target_bytes<=cfg["CPU_target_cap_bytes"],"All32 once-built external target masses cap")
            targets[pid]=torch.from_numpy(mass);del target,mass
        warm=load_warmstart_v1(reader)
        family,optimizers,initialization,hashes=build_family_v1(warm,device,guard);del warm
        residency=h038.FullParentResidencyV1(banks,targets,device,cfg["GPU_residency_cap_bytes"])
        output.save("initialization.json",dict(hypothesis_id="H047",initialization=initialization,
            initial_state_sha256=hashes,observations_seal=observation_seal,external_TRAIN_mass_bytes=target_bytes,
            original_risk_is_matched_extra_optimization_control=True,added_parameters=0,objectives=OBJECTIVES,
            same_VAL_is_development_evidence=True,H046_scores_used=False))
        for at in range(1,31):
            order=list(TRAIN);random.Random(101+at).shuffle(order)
            sums={a:{k:0. for k in ("optimization_objective","original_all_query_NLL","Present_NLL","ancient_NLL")} for a in ARMS}
            started=time.monotonic()
            for number,pid in enumerate(order,1):
                guard();flags_cpu=cache.get(pid,banks[pid],keys[pid])
                require(flags_cpu.tensor_bytes()<=cfg["GPU_flag_cap_bytes"],"Complete GPU flags cap")
                require(residency.needed(pid)+flags_cpu.tensor_bytes()<=cfg["GPU_residency_cap_bytes"],
                    "Shared complete bank,target and flags input residency cap")
                bank,mass=residency.get(pid);flags=flags_cpu.to(device);present=columns[pid].to(device)
                first=at==1 and number==1
                for arm in ARMS:
                    require(all(p.grad is None for m in family.values() for p in m.parameters()),
                        "No gradients across independent arms")
                    row=update_v1(family[arm],optimizers[arm],bank,mass,flags,present,arm,guard,
                        qualify=first,after_optimizer_step=lambda arm=arm:record_step(arm))
                    for k in sums[arm]:sums[arm][k]+=row[k]/32
                    if first:qualifications[arm]=dict(row,parent_id=pid,pass_index=1,optimizer_step=1,
                        part_of_continuous_960_update_training=True,no_extra_qualification_updates=True)
                if first:output.save("first_real_parent_step_qualification.json",dict(
                    hypothesis_id="H047",arms=qualifications,continuous_training=True,no_extra_updates=True))
                del bank,mass,flags,flags_cpu,present,row
                residency.resident.clear();residency.bytes=0
                if number%8==0:print(json.dumps(dict(stage="H047_TRAIN",pass_index=at,parents=number,steps=steps)),flush=True)
            require(steps=={a:32*at for a in ARMS},"Exactly32 actual complete-parent updates per pass/arm")
            row=dict(pass_index=at,steps=steps.copy(),mean_TRAIN_metrics=sums,parent_order=order,
                full_parent_updates_per_arm=32,seconds=time.monotonic()-started)
            history.append(row);output.save("TRAIN_pass_%02d.json"%at,row)
            if at in PASSES:
                groups={}
                for arm in ARMS:
                    cp=dict(schema="H047_cohort_risk_checkpoint_v1",hypothesis_id="H047",phase="train",arm=arm,
                        pass_index=at,additional_optimizer_steps=steps[arm],seed=101,model_kwargs=MODEL,
                        parameter_count=30788,added_parameters=0,model_state={k:v.detach().cpu()
                            for k,v in family[arm].state_dict().items()},optimizer_state=optimizers[arm].state_dict(),
                        optimizer_state_reset=True,complete_H042_body_trainable=True,teacher_used=False,
                        initial_state_sha256=hashes[arm],input_ref=input_ref,config_ref=ctx["config_ref"],
                        observation_seal=observation_seal,warmstart_checkpoint_ref=WARM_CHECKPOINT,
                        warmstart_completion_ref=WARM_COMPLETION,core_ref=CORE_REF,scientific_refs=list(SCIENTIFIC_REFS),
                        proposal_ref=PROPOSAL_REF,registration_ref=cfg["registration_ref"],objective=OBJECTIVES[arm])
                    groups[arm]=output.save("%s.pass_%02d.pt"%(arm,at),cp,"torch")
                checkpoints[str(at)]=groups
        require(steps=={a:960 for a in ARMS} and set(checkpoints)=={str(p) for p in PASSES},
            "Complete paired30pass960update/six-checkpoint training")
        result=dict(schema="H047_paired_train_completion_v1",hypothesis_id="H047",
            status="completed_fixed30pass_paired_TRAIN",input_ref=input_ref,config_ref=ctx["config_ref"],
            arms=list(ARMS),steps=steps,history=history,checkpoint_groups=checkpoints,
            initial_state_sha256=hashes,initialization=initialization,observations_seal=observation_seal,
            first_step_qualification=qualifications,warmstart_checkpoint_ref=WARM_CHECKPOINT,
            warmstart_completion_ref=WARM_COMPLETION,core_ref=CORE_REF,scientific_refs=list(SCIENTIFIC_REFS),
            proposal_ref=PROPOSAL_REF,registration_ref=cfg["registration_ref"],objectives=OBJECTIVES,
            added_parameters=0,VAL_truth_read=False,outer_truth_read=False,H046_scores_used=False,
            same_VAL_is_development_evidence=True,selection_deferred=True,
            flag_cache_peak_bytes=cache.peak,flag_builds=cache.builds,flag_cache_hits=cache.hits,
            external_TRAIN_mass_bytes=target_bytes,target_policy=cfg["training_target_policy"])
        output.save("completed.json",result);return result
    except Exception as error:
        output.save("failed.json",dict(schema="H047_paired_failure_v1",hypothesis_id="H047",phase="train",
            status="failed_partial_preserved",error_type=type(error).__name__,error=str(error),
            logical_steps=steps.copy(),completed_checkpoint_groups=checkpoints,automatic_replay=False,
            VAL_truth_read=False,outer_truth_read=False,H046_scores_used=False))
        raise

class H047PhaseReaderV1(h42.PhaseReader):
    """Exact known RAM packet bytes only; inherited target allowlist/gate unchanged."""
    def __init__(self,base,phase,bindings,*,virtual_prefix):
        super().__init__(base,phase,bindings)
        prefix=Path(virtual_prefix)
        require(not prefix.is_absolute() and ".." not in prefix.parts and "truth" not in prefix.parts,
            "Owned canonical RAM packet namespace")
        self.virtual_prefix=prefix;self.packet_bytes={};self.packet_refs={}
    def register_packet_v1(self,key,packet):
        require(self.phase=="evaluate" and not self.truth_gate and self.ram is not None
            and not self.ram.verified and key in {(a,at,p) for a in ARMS for at in PASSES for p in VAL},
            "Exact observation-only probability packet producer")
        arm,at,pid=key
        require(packet["schema"]=="H047_marker_original_grid_RAM_seal_v1"
            and packet["hypothesis_id"]=="H047" and packet["arm"]==arm
            and packet["pass_index"]==at and packet["parent_id"]==pid,"Packet/key identity")
        path=str(self.virtual_prefix/("%s.pass_%02d.%s.json"%(arm,at,pid)))
        require(path not in self.packet_bytes,"RAM packet replacement denied")
        payload=(json.dumps(packet,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
        ref=dict(path=path,sha256=hashlib.sha256(payload).hexdigest())
        self.packet_bytes[path]=payload;self.packet_refs[path]=dict(ref)
        return ref
    def read(self,record,**kwargs):
        path=record["path"]
        if path in self.packet_bytes:
            payload=self.packet_bytes[path]
            require(self.phase=="evaluate" and record==self.packet_refs[path]
                and hashlib.sha256(payload).hexdigest()==record["sha256"],
                "Exact registered immutable RAM packet bytes")
            return payload
        return super().read(record,**kwargs)

class DenseRAMMatrixV1:
    """All2arms x6checkpoints x8 original dense grids before any VAL truth."""
    def __init__(self,ctx,reader,parents,banks):
        self.ctx,self.reader,self.parents,self.banks=ctx,reader,parents,banks
        self.values,self.records,self.lookup,self.packets={},{},{},{}
        self.bytes,self.verified,self.barrier_ref=0,False,None
    def seal(self,arm,at,pid,value,checkpoint):
        key=(arm,at,pid);require(key not in self.values,"Immutable product replacement denied")
        parent=self.parents[pid]
        require(self.bytes+4*len(parent["positions"])*686*4<=self.ctx["cfg"]["RAM_prediction_cap_bytes"],
            "Preflight complete dense prediction matrix cap")
        value=np.array(value,dtype=np.float32,copy=True,order="C")
        require(value.shape==(len(parent["positions"]),686,4) and np.isfinite(value).all()
            and np.all(value>=0) and np.allclose(value.sum(-1),1,rtol=1e-5,atol=1e-6),"Complete original dense FP32 probabilities")
        self.bytes+=int(value.nbytes);require(self.bytes<=self.ctx["cfg"]["RAM_prediction_cap_bytes"],"All96 dense RAM cap")
        value.setflags(write=False)
        geometry={k:h42.shared.array_identity(parent[k]) for k in
            ("positions","edges","retained_indexes","query_slots","raw_query_nodes","cohorts")}
        geometry["readout_hold"]=h42.shared.array_identity(self.banks[pid].readout_hold.numpy())
        packet=dict(schema="H047_marker_original_grid_RAM_seal_v1",hypothesis_id="H047",arm=arm,pass_index=at,
            parent_id=pid,checkpoint_ref=checkpoint,probability=h42.shared.array_identity(value),geometry=geometry,
            core_ref=CORE_REF,prediction_array_durable=False,targets_opened_before_seal=False)
        ref=self.reader.register_packet_v1(key,packet)
        self.values[key],self.records[key],self.packets[key]=value,ref,packet;self.lookup[ref["path"]]=key
    def verify(self):
        require(set(self.values)=={(a,at,p) for a in ARMS for at in PASSES for p in VAL},"All96 products before truth")
        for key,value in self.values.items():
            self.ctx["guard"]()
            require(not value.flags.writeable and h42.shared.array_identity(value)==self.packets[key]["probability"]
                and self.reader.decode_json(self.records[key])==self.packets[key],"Frozen probability content/record changed")
        self.barrier_ref=self.ctx["output"].save("all96_original_grid_VAL_RAM_sealed_before_truth.json",dict(
            schema="H047_marker_all96_VAL_barrier_v1",hypothesis_id="H047",arms=list(ARMS),
            checkpoint_passes=list(PASSES),VAL_parent_ids=list(VAL),RAM_prediction_bytes=self.bytes,
            complete_dense_prediction_products=96,all96_sealed_before_any_VAL_truth=True,
            seals=[self.records[k] for k in sorted(self.records)],
            packets=[dict(record=self.records[k],packet=self.packets[k]) for k in sorted(self.packets)],
            virtual_packet_bytes_resolved_only_by_known_RAM_reader=True,outer_truth_read=False))
        self.verified=True
    def grid(self,record,parent):
        require(self.verified and record["path"] in self.lookup,"Complete verified96-product barrier")
        key=self.lookup[record["path"]];value=self.values[key]
        require(record==self.records[key] and key[2]==parent["parent_id"] and not value.flags.writeable
            and h42.shared.array_identity(value)==self.packets[key]["probability"],"Current exact original-grid same-parent product")
        return value


def load_checkpoints_v1(reader,train,cfg):
    require(train.get("schema")=="H047_paired_train_completion_v1"
        and train.get("hypothesis_id")=="H047" and train.get("status")=="completed_fixed30pass_paired_TRAIN"
        and train.get("arms")==list(ARMS) and train.get("steps")=={a:960 for a in ARMS}
        and train.get("core_ref")==CORE_REF and train.get("scientific_refs")==list(SCIENTIFIC_REFS)
        and train.get("registration_ref")==cfg["registration_ref"] and train.get("proposal_ref")==PROPOSAL_REF
        and train.get("objectives")==OBJECTIVES and train.get("added_parameters")==0
        and train.get("VAL_truth_read") is False and train.get("outer_truth_read") is False
        and train.get("H046_scores_used") is False and len(set(train["initial_state_sha256"].values()))==1
        and set(train["checkpoint_groups"])=={str(p) for p in PASSES},
        "Actual matched original-risk/balanced-risk fixed H047 TRAIN")
    train_cfg=reader.decode_json(train["config_ref"]);validate_protocol(train_cfg);validate_protocol(cfg)
    require({k:v for k,v in train_cfg.items() if k not in RESOURCE_CAPS}
        =={k:v for k,v in cfg.items() if k not in RESOURCE_CAPS},"Only resource caps may differ across phases")
    states,refs={},{}
    for at in PASSES:
        require(set(train["checkpoint_groups"][str(at)])==set(ARMS),"Exactly2 arms per checkpoint")
        for arm in ARMS:
            r=train["checkpoint_groups"][str(at)][arm]
            cp=torch.load(io.BytesIO(reader.read(r)),map_location="cpu",weights_only=False)
            require(cp.get("schema")=="H047_cohort_risk_checkpoint_v1" and cp.get("hypothesis_id")=="H047"
                and cp.get("phase")=="train" and cp.get("arm")==arm and cp.get("pass_index")==at
                and cp.get("additional_optimizer_steps")==32*at and cp.get("seed")==101
                and cp.get("parameter_count")==30788 and cp.get("added_parameters")==0
                and cp.get("model_kwargs")==MODEL and cp.get("complete_H042_body_trainable") is True
                and cp.get("teacher_used") is False and cp.get("optimizer_state_reset") is True
                and cp.get("initial_state_sha256")==train["initial_state_sha256"][arm]
                and cp.get("input_ref")==train["input_ref"] and cp.get("config_ref")==train["config_ref"]
                and cp.get("observation_seal")==train["observations_seal"]
                and cp.get("warmstart_checkpoint_ref")==WARM_CHECKPOINT
                and cp.get("warmstart_completion_ref")==WARM_COMPLETION and cp.get("core_ref")==CORE_REF
                and cp.get("scientific_refs")==list(SCIENTIFIC_REFS) and cp.get("proposal_ref")==PROPOSAL_REF
                and cp.get("registration_ref")==cfg["registration_ref"] and cp.get("objective")==OBJECTIVES[arm]
                and all(isinstance(v,torch.Tensor) and v.dtype==torch.float32 and bool(torch.isfinite(v).all())
                    for v in cp["model_state"].values()),"Exact finite pure-H042 checkpoint provenance")
            states[arm,at],refs[arm,at]=cp["model_state"],r
    return states,refs


def paired_effects_v1(full,*,group="primary_present_day_182"):
    left={r["parent_id"]:r["groups"][group] for r in full["original_risk"]["cells"]}
    right={r["parent_id"]:r["groups"][group] for r in full["balanced_risk"]["cells"]}
    require(set(left)==set(right)==set(VAL),"Exact8 matched original VAL parents")
    cells=[]
    for pid in VAL:
        a,b=left[pid],right[pid]
        fa,fb=a.get("boundaries",{}).get("f1"),b.get("boundaries",{}).get("f1")
        cells.append(dict(parent_id=pid,accuracy_delta_balanced_minus_original=b["accuracy"]-a["accuracy"],
            NLL_improvement_original_minus_balanced=None if a["nll"] is None or b["nll"] is None else a["nll"]-b["nll"],
            boundary_F1_delta_balanced_minus_original=None if fa is None or fb is None else fb-fa))
    gain=statistics.mean(r["accuracy_delta_balanced_minus_original"] for r in cells)
    bd=[r["boundary_F1_delta_balanced_minus_original"] for r in cells]
    boundary=None if any(x is None for x in bd) else statistics.mean(bd)
    return dict(cells=cells,population=group,mean_accuracy_gain=gain,mean_boundary_F1_delta=boundary,
        sample_SD_accuracy_gain=statistics.stdev(r["accuracy_delta_balanced_minus_original"] for r in cells),
        positive_parents=sum(r["accuracy_delta_balanced_minus_original"]>0 for r in cells),
        negative_parents=sum(r["accuracy_delta_balanced_minus_original"]<0 for r in cells),
        meets_registered_practical_criteria=gain>=.001 and boundary is not None and boundary>=-.01,
        accuracy_threshold=.001,boundary_F1_loss_tolerance=.01,
        aggregation="equal_parent_mean_Present182_across8VAL" if group=="primary_present_day_182" else "equal_parent_mean_across8VAL",
        statistical_unit="independent_ARG_parent",same_VAL_is_development_evidence=True,
        significance_or_novelty_claim=False)


def run_evaluate_v1(ctx,input_ref):
    from .mutation_routed_full_panel_primary_v1 import FullPanelPrimaryValidationV1
    cfg,output,guard,device=ctx["cfg"],ctx["output"],ctx["guard"],ctx["device"]
    reader=H047PhaseReaderV1(ctx["reader"],"evaluate",ctx["bindings"],
        virtual_prefix=str(output.path.relative_to(ROOT)/"H047_RAM_probability_packets"))
    train=reader.decode_json(ctx["train_completion_ref"]);states,checkpoints=load_checkpoints_v1(reader,train,cfg)
    cache=FlagCacheV1(ctx)
    banks,parents,keys,columns,observation_seal=load_observations_v1(ctx,reader,VAL,cache)
    ram=DenseRAMMatrixV1(ctx,reader,parents,banks);reader.ram=ram
    residency=h038.FullParentResidencyV1(banks,{},device,cfg["GPU_residency_cap_bytes"])
    models={a:core.AnonymousCladeFlagControlGNNV1(**MODEL).to(device=device,dtype=torch.float32)
        .eval().requires_grad_(False) for a in ARMS}
    for pid in VAL:
        guard();flags_cpu=cache.get(pid,banks[pid],keys[pid])
        require(flags_cpu.tensor_bytes()<=cfg["GPU_flag_cap_bytes"]
            and residency.needed(pid)+flags_cpu.tensor_bytes()<=cfg["GPU_residency_cap_bytes"],
            "Complete evaluation parent and flags residency caps")
        bank,_=residency.get(pid);flags=flags_cpu.to(device)
        for arm in ARMS:
            for at in PASSES:
                guard();models[arm].load_state_dict(states[arm,at],strict=True)
                with torch.no_grad(),torch.autocast(device_type=device.type,enabled=False):
                    holds=models[arm].forward_train_holds(bank,flags=flags,control="aligned").softmax(-1)
                    probability=holds.index_select(0,bank.readout_hold)
                ram.seal(arm,at,pid,probability.cpu().numpy(),checkpoints[arm,at]);del holds,probability
                print(json.dumps(dict(stage="H047_VAL_prediction",sealed_products=len(ram.values),total_products=96)),flush=True)
        del bank,flags,flags_cpu
        residency.resident.clear();residency.bytes=0
    del models,residency,states
    ram.verify();reader.truth_gate=True;ctx["VAL_truth_gate_opened"]=True
    selected,histories,full,receipts={},{},{},{}
    for arm in ARMS:
        validation=FullPanelPrimaryValidationV1(parents,ctx["bindings"],reader,fit.metric_helpers_v1(),cfg)
        rows=[]
        for at in PASSES:
            guard();cells,means=validation.score([(p,ram.records[arm,at,p]) for p in VAL])
            rows.append(dict(pass_index=at,cells=cells,equal_parent_metrics=means,
                checkpoint=checkpoints[arm,at],selection_rule=SELECTION))
        best=min(rows,key=lambda r:h038.selection_key_v1(r["equal_parent_metrics"],r["pass_index"]))
        cells,means=validation.full_final([(p,ram.records[arm,best["pass_index"],p]) for p in VAL])
        full[arm]=dict(cells=cells,equal_parent_metrics=means,selection_revised=False)
        selected[arm],histories[arm],receipts[arm]=best,rows,validation.receipt()
    groups=set(full[ARMS[0]]["cells"][0]["groups"])
    require(all(set(c["groups"])==groups for a in ARMS for c in full[a]["cells"]),"Complete matched cohort diagnostics")
    result=dict(schema="H047_paired_VAL_completion_v1",hypothesis_id="H047",
        status="completed_original8VAL_paired_development_selection",input_ref=input_ref,
        config_ref=ctx["config_ref"],train_completion_ref=ctx["train_completion_ref"],
        observations_seal=observation_seal,registration_ref=cfg["registration_ref"],proposal_ref=PROPOSAL_REF,
        core_ref=CORE_REF,selection_rule=SELECTION,selected=selected,history=histories,
        selected_full_diagnostics=full,metric_receipts=receipts,registered_practical_criteria=paired_effects_v1(full),
        all_selected_population_paired_effects={g:paired_effects_v1(full,group=g) for g in sorted(groups)},
        optimization_performed=False,all96_RAM_seals_before_any_VAL_truth=True,
        RAM_dense_probability_bytes=ram.bytes,complete_probability_matrix_manifest_ref=ram.barrier_ref,
        virtual_seals_durable_in_matrix_manifest=True,dense_prediction_arrays_saved=False,
        outer_truth_read=False,same_VAL_is_development_evidence=True,H046_scores_used=False,
        original_risk_is_matched_extra_optimization_control=True,added_parameters=0,novel_architecture_claim=False)
    output.save("completed.json",result);return result


def run_context_v1(ctx,input_ref):
    require(ctx.get("root_authorizer")=="root_agent" and ctx.get("admitted") is True
        and ctx.get("phase") in ("train","evaluate") and ctx.get("arm")=="joint_pair"
        and callable(ctx.get("guard")) and callable(ctx.get("before_flag_build")),"Admitted paired root context")
    validate_protocol(ctx["cfg"])
    for r in (*SCIENTIFIC_REFS,PROPOSAL_REF,ctx["cfg"]["registration_ref"]):h42.bound(r)
    require(ctx["reader"].root.resolve()==ROOT and ctx["device"].type=="cuda"
        and ctx["reader"].decode_json(ctx["config_ref"])==ctx["cfg"],"Exact root reader/config and leased CUDA")
    ids=TRAIN if ctx["phase"]=="train" else VAL
    require(set(ctx["bindings"])==set(ids) and set(ctx["entry_refs"])==set(TRAIN+VAL),
        "Exact phase targets and all40 original observations")
    science();ctx["guard"]()
    if ctx["phase"]=="train":return run_train_v1(ctx,input_ref)
    try:return run_evaluate_v1(ctx,input_ref)
    except Exception as error:
        ctx["output"].save("failed.json",dict(schema="H047_paired_failure_v1",hypothesis_id="H047",phase="evaluate",
            status="failed_partial_preserved",error_type=type(error).__name__,error=str(error),automatic_replay=False,
            VAL_truth_gate_opened=ctx.get("VAL_truth_gate_opened",False),outer_truth_read=False))
        raise
