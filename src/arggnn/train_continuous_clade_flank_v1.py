"""H046 paired original-grid TRAIN/evaluation API. Inert source-only import.

Root owns every resource admission, process title, guard, launcher and device.
Uses unchanged H044 coordinate/allele preparation and exact-span target/scoring.
"""
from __future__ import annotations
from collections import OrderedDict
import hashlib
import io
import json
import math
from pathlib import Path
import random
import time

from . import train_anonymous_clade_flag_control_v1 as h42
from . import train_marker_resolution_readout_v1 as h44

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE = "src/arggnn/train_continuous_clade_flank_v1.py"
ARMS = ("foreign_flank","own_state")
TRAIN,VAL,PASSES = h42.TRAIN,h42.VAL,h42.PASSES
MODEL,SELECTION,METRICS = h42.MODEL,h42.SELECTION,h42.METRICS
READOUT = dict(query_block=32,center_block=1024,checkpoint_readout=True,checkpoint_scan=True)
REGISTRATION_REF = dict(path="research/reviews/H046_continuous_clade_context_proposal_v1/registration_root_v1.json",
    sha256="fbff43201332463eb8a5651aec541705002b17325482773c546032ebf7d22198")
DESIGN_REF = dict(path="research/reviews/H046_continuous_clade_context_proposal_v1/proposal_spec.json",
    sha256="0d933a7ef7fb65a9d11e5421ecb622631b674ac497d302a7d307a47a834e685a")
CORE_REF = {'path': 'src/arggnn/continuous_clade_flank_gnn_v1.py', 'sha256': '2c81b74c59a532b46ed75f7fd656ea526ec06fd9bf62af35855ab291f5451e20'}
WARM_CHECKPOINT,WARM_COMPLETION = h44.WARM_CHECKPOINT,h44.WARM_COMPLETION
SCIENTIFIC_REFS = ({'path': 'src/arggnn/continuous_clade_flank_gnn_v1.py', 'sha256': '2c81b74c59a532b46ed75f7fd656ea526ec06fd9bf62af35855ab291f5451e20'}, {'path': 'src/arggnn/train_marker_resolution_readout_v1.py', 'sha256': '386814feb14a64310b8c51c2cfbaf681a59b0273f8be0f72ffde5a222eb08c3c'}, {'path': 'src/arggnn/marker_resolution_readout_v1.py', 'sha256': '27ecd30683024def2d408bf16250102b3252d0ae112687b9664fca086b141310'}, {'path': 'src/arggnn/train_anonymous_clade_flag_control_v1.py', 'sha256': 'cb58ec3f81d81cdcd88637f3ade695a3e3ef97a2d0c10b7713e9bddb43fbb2ed'}, {'path': 'src/arggnn/anonymous_clade_flag_control_v1.py', 'sha256': 'd88a4639a53e63e2bff91e20d98e2af1f5cf8abc71e5fde2e02ac35e831eb8ce'}, {'path': 'src/arggnn/anonymous_exact_clade_transport_v1.py', 'sha256': '441751a252f111fdf0d8af4e70e93e5a923b7bad7829f3aed7d5060d65727304'}, {'path': 'src/arggnn/anonymous_clade_gnn_v1.py', 'sha256': '2f878e6334079cff083f67b6da8eda129dd30e7beebc62ae2a93d51796c3f137'}, {'path': 'src/arggnn/mutation_routed_whole_tree_gnn_v1.py', 'sha256': '254f5155e31f81ae675e2bd08d38c97c9bf130d47e98efa2deea4c7344d9e073'}, {'path': 'src/arggnn/mutation_routed_pyg_gnn_v1.py', 'sha256': '53866aa4616d12e7d6cbdd34160ee7ab8fb00999444c348ee4d251b8fdfd4337'}, {'path': 'src/arggnn/mutation_routed_full_panel_fast_reader_v1.py', 'sha256': 'd6b963dc14e1f6752e310e627faf2e4f9a2e408fcb7dab17e3439771ab227e03'}, {'path': 'src/arggnn/mutation_routed_full_panel_primary_v1.py', 'sha256': 'a94fd5af1cd10b1605f75621c36a130faa3f9d5ad71068f8b187d8e73c7e5374'})
RESOURCE_CAPS = (
    "CPU_bank_cap_bytes","CPU_target_cap_bytes","marker_cache_cap_bytes",
    "marker_preparation_cap_bytes","CPU_coordinate_bulk_cap_bytes","CPU_binding_bulk_cap_bytes",
    "flank_cache_cap_bytes","flank_payload_cap_bytes","flank_preparation_cap_bytes",
    "GPU_residency_cap_bytes","GPU_flank_cap_bytes","RAM_prediction_cap_bytes",
)
require,bound = h42.require,h42.bound
LOCAL_CALLABILITY_STRATA = (("0",0,1),("1-2",1,3),("3-5",3,6),(">=6",6,None))


def science():
    global np,torch,h038,fit,core
    h44.science()
    np,torch,h038,fit = h44.np,h44.torch,h44.h038,h44.fit
    from . import continuous_clade_flank_gnn_v1 as core


def validate_protocol(cfg):
    require(cfg.get("schema")=="H046_continuous_clade_flank_execution_config_v1"
        and cfg.get("hypothesis_id")=="H046" and cfg.get("arms")==list(ARMS)
        and cfg.get("linked_hypothesis_ids")==["H042","H044","H045"]
        and cfg.get("seed")==101 and cfg.get("dtype")=="float32"
        and cfg.get("passes")==30 and cfg.get("updates_per_pass")==32
        and cfg.get("updates_per_arm")==960 and cfg.get("checkpoint_passes")==list(PASSES)
        and cfg.get("TRAIN_parent_ids")==list(TRAIN) and cfg.get("VAL_parent_ids")==list(VAL)
        and cfg.get("execution_mode")=="paired_sequential_one_parent"
        and cfg.get("parent_order")=="shuffle_seed101_plus_pass"
        and cfg.get("model")==MODEL and cfg.get("readout")==READOUT
        and cfg.get("parameters_per_arm")==31165 and cfg.get("base_parameters")==30788
        and cfg.get("added_parameters")==377
        and cfg.get("learning_rate")==.001 and cfg.get("weight_decay")==.0001
        and cfg.get("AdamW_betas")==[.9,.999] and cfg.get("AdamW_epsilon")==1e-8
        and cfg.get("gradient_clip_norm")==1. and cfg.get("early_stopping") is False
        and cfg.get("first_step_qualification")=="within_continuous_training_no_extra_updates"
        and cfg.get("VAL_influences_optimization") is False
        and cfg.get("same_VAL_is_development_evidence") is True
        and cfg.get("selection")==SELECTION and cfg.get("metrics")==METRICS
        and cfg.get("core_ref")==CORE_REF and cfg.get("scientific_refs")==list(SCIENTIFIC_REFS)
        and cfg.get("registration_ref")==REGISTRATION_REF and cfg.get("design_ref")==DESIGN_REF
        and cfg.get("warmstart_checkpoint_ref")==WARM_CHECKPOINT
        and cfg.get("warmstart_completion_ref")==WARM_COMPLETION
        and cfg.get("TRAIN_target_policy")=="one_complete_original100bp_parent_mass_streamed_once_shared_between_arms"
        and cfg.get("invariance_scope")=="fixed_field_original_focal_support_episode_membership_and_positions_only"
        and cfg.get("dense_prediction_arrays_saved") is False
        and cfg.get("flank_arrays_saved_to_disk") is False
        and cfg.get("outer_truth_read") is False
        and cfg.get("practical_accuracy_gain_threshold")==.001
        and cfg.get("practical_boundary_F1_loss_tolerance")==.01
        and cfg.get("practical_criteria_aggregation")=="equal_parent_mean_Present182_across8VAL"
        and cfg.get("local_callability_strata")==["0","1-2","3-5",">=6"]
        and cfg.get("local_callability_definition")=="min_usable_SNPs_left_right_within_original_focal_hold"
        and cfg.get("local_callability_diagnostics")=="selected_models_only_after_complete96_seals_truth_barrier",
        "Exact registered H046 paired original-grid protocol")
    for name in RESOURCE_CAPS:
        require(type(cfg.get(name)) is int and cfg[name]>0,"Positive root-reviewed cap: "+name)
    require(cfg["flank_payload_cap_bytes"]<=1006632960,"Source compact-payload limit")


def flank_upper_bounds_v1(bank, grid_centers):
    s,t,n = int(bank.genotype.shape[0]),int(bank.sample_index.shape[0]),int(bank.parent.numel())
    # Actual M,E are observed only after preparation; both are at mostN.
    value = core.logical_flank_bounds_v1(s,t,n,n,n,int(grid_centers),
        query_block=READOUT["query_block"],center_block=READOUT["center_block"])
    return dict(value,native_nodes=n,native_holds=t,haplotypes=int(bank.sample_index.shape[1]),
        sites=s,grid_centers=int(grid_centers),member_upper_bound=n,episode_upper_bound=n,
        preparation_named_array_allowance_bytes=128*n+64*int(grid_centers)+16*t)


class ContinuousFlankCacheV1:
    """RAM-only LRU; unchanged H044 evidence and bound source-native intervals."""
    def __init__(self,ctx,reader):
        self.ctx,self.cfg,self.guard = ctx,ctx["cfg"],ctx["guard"]
        self.markers = h44.MarkerCacheV1(ctx,reader)
        self.values,self.proofs = OrderedDict(),{}
        self.bytes=self.peak=self.builds=self.hits=0

    def get(self,pid,bank,parent):
        self.guard()
        if pid in self.values:
            value=self.values[pid];value.validate()
            self.values.move_to_end(pid);self.hits+=1
            return value
        marker_bundle=self.markers.get(pid,bank,parent)
        limits=flank_upper_bounds_v1(bank,len(parent["positions"]))
        require(limits["preparation_named_array_allowance_bytes"]<=self.cfg["flank_preparation_cap_bytes"],
                "Named clade preparation allowance exceeds root cap")
        self.ctx["before_flank_build"](pid,bank,limits)
        entry=self.ctx["preparation"]["parents"][pid]
        intervals=self.markers.coordinates[entry["coordinates_member_selector"]["intervals_bp"]]
        value,meta=core.prepare_continuous_clade_bank_v1(marker_bundle,intervals,
            max_payload_bytes=self.cfg["flank_payload_cap_bytes"])
        counts=core.logical_flank_bounds_v1(meta["sites"],meta["native_holds"],meta["native_nodes"],
            meta["member_count"],meta["episode_count"],meta["grid_centers"],
            query_block=READOUT["query_block"],center_block=READOUT["center_block"])
        require(value.flanks.tensor_bytes()==counts["compact_flank_payload_bytes"],
                "Exact compact payload tensor count")
        proof=dict(metadata=meta,logical_bounds=counts,marker_recipe=self.markers.proofs[pid],
            original_observed_entry_ref=self.ctx["entry_refs"][pid])
        if pid in self.proofs:require(proof==self.proofs[pid],"Immutable rebuilt observation recipe")
        else:self.proofs[pid]=proof
        # Conservatively charge borrowed marker tensors too. Separate H044 marker
        # cache has its own cap; aliases never increase these union upper bounds.
        size=value.flanks.tensor_bytes()+value.markers.tensor_bytes()
        require(size<=self.cfg["flank_cache_cap_bytes"],"One complete flank+marker parent exceeds CPU LRU cap")
        while self.values and self.bytes+size>self.cfg["flank_cache_cap_bytes"]:
            _,old=self.values.popitem(last=False)
            self.bytes-=old.flanks.tensor_bytes()+old.markers.tensor_bytes()
        self.values[pid]=value;self.bytes+=size;self.peak=max(self.peak,self.bytes);self.builds+=1
        self.guard()
        return value


def load_observations_v1(ctx,reader,ids,cache):
    from .mutation_routed_full_panel_fast_reader_v1 import read_full_cache_index_v1,read_full_cache_parent_v1
    index=read_full_cache_index_v1(ctx["observation_cache_ref"],reader,registry=ctx["registry"],
        entry_refs=ctx["entry_refs"],source_refs=ctx["observation_source_refs"])
    banks,parents,total={},{},0
    for pid in ids:
        ctx["guard"]()
        item=read_full_cache_parent_v1(pid,index,reader)
        bank,parent=item["model_inputs"],item["scoring"]
        require(parent["split"]==("train" if pid in TRAIN else "val"),"Original same-parent observed split")
        total+=h038.tensor_bytes_v1(bank)
        require(total<=ctx["cfg"]["CPU_bank_cap_bytes"],"Complete phase CPU-bank cap")
        banks[pid],parents[pid]=bank,parent
        cache.get(pid,bank,parent)
        del item
    require(set(cache.proofs)==set(ids) and set(cache.markers.proofs)==set(ids),
            "All phase observation recipes before any phase truth")
    seal=ctx["output"].save("all_%d_H046_observation_recipes.json"%len(ids),dict(
        schema="H046_complete_observation_barrier_v1",hypothesis_id="H046",parent_ids=list(ids),
        preparation_completion_ref=ctx["preparation_completion_ref"],
        preparation_actual_wait_ref=ctx["preparation_actual_wait_ref"],
        CPU_bank_tensor_bytes=total,flank_recipes=cache.proofs,
        truth_read=False,full_backbone_trees_retained=True,flank_arrays_durable=False))
    return banks,parents,seal


def load_warmstart_v1(reader):
    completed=reader.decode_json(WARM_COMPLETION)
    require(completed.get("schema")=="anonymous_clade_flag_train_completion_v1"
        and completed.get("status")=="completed_fixed_30_pass_flag_TRAIN"
        and completed.get("hypothesis_id")=="H042" and completed.get("arm")=="exact_clade_flags"
        and completed.get("steps")=={"exact_clade_flags":960} and len(completed.get("history",[]))==30
        and completed.get("checkpoint_groups",{}).get("30",{}).get("exact_clade_flags")==WARM_CHECKPOINT
        and completed.get("VAL_truth_read") is False and completed.get("outer_truth_read") is False,
        "Actual completed original H042 pass30 warmstart")
    return torch.load(io.BytesIO(reader.read(WARM_CHECKPOINT)),map_location="cpu",weights_only=False)


def build_family_v1(payload,device,guard):
    guard()
    models,old_optimizers,proof=core.make_matched_flank_family_v1(payload,**READOUT)
    del old_optimizers
    hashes={a:h42.shared.state_identity(m.state_dict()) for a,m in models.items()}
    require(len(set(hashes.values()))==1,"All initial model weights matched")
    for a in ARMS:
        guard();models[a]=models[a].to(device=device,dtype=torch.float32)
    optimizers={a:torch.optim.AdamW(models[a].parameters(),lr=.001,weight_decay=.0001,
        betas=(.9,.999),eps=1e-8) for a in ARMS}
    require(all(not o.state for o in optimizers.values()),"Independent reset AdamW")
    require(not ({p.data_ptr() for p in models[ARMS[0]].parameters()}
                &{p.data_ptr() for p in models[ARMS[1]].parameters()}),"Independent learned tensor storage")
    return models,optimizers,proof,hashes


def device_parent_v1(ctx,bundle,mass_cpu=None):
    bundle.validate()
    require(bundle.flanks.device_bytes()<=ctx["cfg"]["GPU_flank_cap_bytes"],"Complete GPU flank payload cap")
    needed=h038.tensor_bytes_v1(bundle.source_bank)+bundle.flanks.device_bytes()
    if mass_cpu is not None:needed+=mass_cpu.numel()*4
    require(needed<=ctx["cfg"]["GPU_residency_cap_bytes"],"One shared complete parent input residency cap")
    moved=bundle.to(ctx["device"])
    return moved,None if mass_cpu is None else mass_cpu.to(ctx["device"])


def update_v1(model,optimizer,bundle,mass,guard,*,qualify=False,after_optimizer_step=None):
    guard();model.train();optimizer.zero_grad(set_to_none=True)
    device=bundle.bank.parent.device
    if qualify and device.type=="cuda":
        torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
    started=time.monotonic()
    try:
        with torch.autocast(device_type=device.type,enabled=False):
            logits=model(bundle)
            loss=core.original_grid_span_nll_v1(logits,mass,bundle.markers.bin_widths_bp.to(device))
        require(loss.dtype==torch.float32 and bool(torch.isfinite(loss)),"Finite complete-parent FP32 grid loss")
        loss.backward()
        require(all(p.grad is None or (p.grad.dtype==torch.float32 and bool(torch.isfinite(p.grad).all()))
                    for p in model.parameters()),"Finite complete-body FP32 gradients")
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        if after_optimizer_step is not None:after_optimizer_step()
        require(all(bool(torch.isfinite(p).all()) for p in model.parameters()),"Finite updated learned state")
        require(all(not isinstance(v,torch.Tensor) or (v.dtype==torch.float32 and bool(torch.isfinite(v).all()))
                    for state in optimizer.state.values() for v in state.values()),"Finite FP32 independent AdamW")
        row=dict(loss=float(loss.detach()),gradient_norm=float(norm),seconds=time.monotonic()-started,
                 finite_FP32=True)
        if qualify:
            if device.type=="cuda":
                torch.cuda.synchronize(device)
                row["memory"]=dict(real_GPU_qualification=True,
                    allocated_bytes=int(torch.cuda.memory_allocated(device)),
                    reserved_bytes=int(torch.cuda.memory_reserved(device)),
                    peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                    peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)))
            else:row["memory"]=dict(real_GPU_qualification=False,device_type=device.type)
        del logits,loss,norm
    finally:
        optimizer.zero_grad(set_to_none=True)
    require(all(p.grad is None for p in model.parameters()),"End gradient lifetime before next arm")
    guard()
    return row


def run_train_v1(ctx,input_ref):
    cfg,output,guard,device=ctx["cfg"],ctx["output"],ctx["guard"],ctx["device"]
    reader=h42.PhaseReader(ctx["reader"],"train",ctx["bindings"])
    cache=ContinuousFlankCacheV1(ctx,reader)
    steps={a:0 for a in ARMS};history=[];checkpoints={};qualifications={}
    def record_step(arm):steps[arm]+=1
    try:
        banks,parents,observation_seal=load_observations_v1(ctx,reader,TRAIN,cache)
        payload=load_warmstart_v1(reader)
        models,optimizers,initialization,hashes=build_family_v1(payload,device,guard)
        del payload
        output.save("initialization.json",dict(schema="H046_paired_initialization_v1",hypothesis_id="H046",
            initialization=initialization,initial_state_sha256=hashes,observation_seal=observation_seal,
            complete_H042_body_trainable=True,independent_model_and_optimizer_states=True,
            optimizer_moments_and_step_reset=True,VAL_influences_optimization=False,
            scientific_refs=list(SCIENTIFIC_REFS),registration_ref=REGISTRATION_REF))
        reader.truth_gate=True
        for at in range(1,31):
            order=list(TRAIN);random.Random(101+at).shuffle(order)
            means={a:0. for a in ARMS};started=time.monotonic()
            for number,pid in enumerate(order,1):
                guard()
                cpu_bundle=cache.get(pid,banks[pid],parents[pid])
                mass_cpu=h44.original_mass_v1(ctx,reader,pid,parents[pid])
                bundle,mass=device_parent_v1(ctx,cpu_bundle,mass_cpu)
                first=at==1 and number==1
                for arm in ARMS:
                    require(all(p.grad is None for m in models.values() for p in m.parameters()),
                            "No gradient lifetime may cross an independent arm step")
                    row=update_v1(models[arm],optimizers[arm],bundle,mass,guard,qualify=first,
                        after_optimizer_step=lambda a=arm:record_step(a))
                    means[arm]+=row["loss"]/32.
                    if first:qualifications[arm]=dict(row,parent_id=pid,optimizer_step=steps[arm],
                        native_nodes=int(bundle.bank.parent.numel()),native_holds=len(bundle.bank.spans),
                        clade_payload_bytes=cpu_bundle.flanks.device_bytes(),
                        actual_counts=cache.proofs[pid]["logical_bounds"],
                        part_of_continuous_960_updates=True,no_extra_qualification_updates=True)
                if first:output.save("first_real_parent_step_qualification.json",dict(hypothesis_id="H046",
                    arms=qualifications,continuous_training=True,no_extra_updates=True))
                del bundle,mass,mass_cpu,cpu_bundle
                if number%8==0:print(json.dumps(dict(stage="H046_paired_TRAIN",pass_index=at,
                    complete_parents=number,steps=steps.copy(),seconds=time.monotonic()-started)),flush=True)
            require(steps=={a:32*at for a in ARMS},"Exactly32 actual whole-parent optimizer steps per arm/pass")
            row=dict(pass_index=at,steps=steps.copy(),mean_TRAIN_NLL=means,full_parent_updates_per_arm=32,
                     seconds=time.monotonic()-started)
            history.append(row);output.save("TRAIN_pass_%02d.json"%at,row)
            if at in PASSES:
                groups={}
                for arm in ARMS:
                    guard()
                    cp=dict(schema="H046_continuous_clade_flank_checkpoint_v1",hypothesis_id="H046",
                        arm=arm,phase="train",pass_index=at,additional_optimizer_steps=steps[arm],
                        seed=101,parameter_count=31165,model_state={k:v.detach().cpu() for k,v in models[arm].state_dict().items()},
                        optimizer_state=optimizers[arm].state_dict(),optimizer_state_reset=True,
                        complete_H042_body_trainable=True,teacher_used=False,initial_state_sha256=hashes[arm],
                        input_ref=input_ref,config_ref=ctx["config_ref"],core_ref=CORE_REF,readout=READOUT,
                        observation_seal=observation_seal,preparation_completion_ref=ctx["preparation_completion_ref"],
                        warmstart_checkpoint_ref=WARM_CHECKPOINT,warmstart_completion_ref=WARM_COMPLETION,
                        scientific_refs=list(SCIENTIFIC_REFS),registration_ref=REGISTRATION_REF)
                    groups[arm]=output.save("%s.pass_%02d.pt"%(arm,at),cp,"torch")
                checkpoints[str(at)]=groups
        require(steps=={a:960 for a in ARMS} and set(checkpoints)=={str(p) for p in PASSES},
                "Fixed paired30pass/960update/six-checkpoint schedule")
        result=dict(schema="H046_paired_train_completion_v1",hypothesis_id="H046",
            status="completed_fixed30pass_paired_TRAIN",input_ref=input_ref,config_ref=ctx["config_ref"],
            arms=list(ARMS),steps=steps,history=history,checkpoint_groups=checkpoints,
            initial_state_sha256=hashes,initialization=initialization,observations_seal=observation_seal,
            preparation_completion_ref=ctx["preparation_completion_ref"],warmstart_checkpoint_ref=WARM_CHECKPOINT,
            warmstart_completion_ref=WARM_COMPLETION,core_ref=CORE_REF,scientific_refs=list(SCIENTIFIC_REFS),
            registration_ref=REGISTRATION_REF,VAL_truth_read=False,outer_truth_read=False,selection_deferred=True,
            first_step_qualification=qualifications,same_VAL_is_development_evidence=True,
            flank_cache_peak_bytes=cache.peak,marker_cache_peak_bytes=cache.markers.peak,
            flank_builds=cache.builds,flank_cache_hits=cache.hits,flank_arrays_saved_to_disk=False)
        output.save("completed.json",result);return result
    except Exception as error:
        output.save("failed.json",dict(schema="H046_paired_failure_v1",hypothesis_id="H046",phase="train",
            status="failed_partial_preserved",error_type=type(error).__name__,error=str(error),
            logical_steps=steps.copy(),completed_checkpoint_groups=checkpoints,automatic_replay=False,
            VAL_truth_read=False,outer_truth_read=False))
        raise

class H046PhaseReaderV1(h42.PhaseReader):
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
        require(packet["schema"]=="H046_marker_original_grid_RAM_seal_v1"
            and packet["hypothesis_id"]=="H046" and packet["arm"]==arm
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
        packet=dict(schema="H046_marker_original_grid_RAM_seal_v1",hypothesis_id="H046",arm=arm,pass_index=at,
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
            schema="H046_marker_all96_VAL_barrier_v1",hypothesis_id="H046",arms=list(ARMS),
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
    require(train.get("schema")=="H046_paired_train_completion_v1"
        and train.get("hypothesis_id")=="H046" and train.get("status")=="completed_fixed30pass_paired_TRAIN"
        and train.get("arms")==list(ARMS) and train.get("steps")=={a:960 for a in ARMS}
        and train.get("core_ref")==CORE_REF and train.get("scientific_refs")==list(SCIENTIFIC_REFS)
        and train.get("registration_ref")==REGISTRATION_REF
        and train.get("VAL_truth_read") is False and train.get("outer_truth_read") is False
        and train.get("preparation_completion_ref")==cfg["preparation_completion_ref"]
        and set(train.get("checkpoint_groups",{}))=={str(p) for p in PASSES}
        and len(set(train["initial_state_sha256"].values()))==1,
        "Actual full matched H046 TRAIN completion and immutable original preparation")
    train_cfg=reader.decode_json(train["config_ref"]);validate_protocol(train_cfg);validate_protocol(cfg)
    require({k:v for k,v in train_cfg.items() if k not in RESOURCE_CAPS}
            =={k:v for k,v in cfg.items() if k not in RESOURCE_CAPS},
            "Only reviewed resource caps may differ between TRAIN/evaluation")
    states,refs={},{}
    for arm in ARMS:
        for at in PASSES:
            require(set(train["checkpoint_groups"][str(at)])==set(ARMS),"Exactly2 arms per checkpoint group")
            ref=train["checkpoint_groups"][str(at)][arm]
            cp=torch.load(io.BytesIO(reader.read(ref)),map_location="cpu",weights_only=False)
            require(cp.get("schema")=="H046_continuous_clade_flank_checkpoint_v1"
                and cp.get("hypothesis_id")=="H046" and cp.get("arm")==arm and cp.get("phase")=="train"
                and cp.get("pass_index")==at and cp.get("additional_optimizer_steps")==32*at
                and cp.get("seed")==101 and cp.get("parameter_count")==31165
                and cp.get("optimizer_state_reset") is True and cp.get("teacher_used") is False
                and cp.get("complete_H042_body_trainable") is True
                and cp.get("core_ref")==CORE_REF and cp.get("readout")==READOUT
                and cp.get("initial_state_sha256")==train["initial_state_sha256"][arm]
                and cp.get("input_ref")==train["input_ref"] and cp.get("config_ref")==train["config_ref"]
                and cp.get("preparation_completion_ref")==cfg["preparation_completion_ref"]
                and cp.get("warmstart_checkpoint_ref")==WARM_CHECKPOINT
                and cp.get("warmstart_completion_ref")==WARM_COMPLETION
                and cp.get("scientific_refs")==list(SCIENTIFIC_REFS)
                and cp.get("registration_ref")==REGISTRATION_REF
                and all(isinstance(v,torch.Tensor) and v.dtype==torch.float32
                        and bool(torch.isfinite(v).all()) for v in cp["model_state"].values()),
                "Exact complete FP32 H046 checkpoint provenance")
            states[arm,at],refs[arm,at]=cp["model_state"],ref
    return states,refs


def practical_criteria_v1(full):
    rows=[]
    for pid in VAL:
        c={a:next(r for r in full[a]["cells"] if r["parent_id"]==pid)
           ["groups"]["primary_present_day_182"] for a in ARMS}
        foreign,own=(c[a] for a in ARMS)
        ff,of=foreign["boundaries"]["f1"],own["boundaries"]["f1"]
        rows.append(dict(parent_id=pid,accuracy_gain=foreign["accuracy"]-own["accuracy"],
            boundary_F1_delta=None if ff is None or of is None else ff-of))
    accuracy=sum(r["accuracy_gain"] for r in rows)/8
    defined=[r["boundary_F1_delta"] for r in rows if r["boundary_F1_delta"] is not None]
    boundary=sum(defined)/8 if len(defined)==8 else None
    return dict(cells=rows,mean_Present182_accuracy_gain=accuracy,mean_boundary_F1_delta=boundary,
        meets_registered_practical_criteria=accuracy>=.001 and boundary is not None and boundary>=-.01,
        accuracy_threshold=.001,boundary_F1_loss_tolerance=.01,
        aggregation="equal_parent_mean_Present182_across8VAL",
        worst_parent_boundary_loss_bound=False,development_not_significance_or_novelty=True)



def callability_block_sums_v1(probability,mass,minimum_counts):
    """Fixed observation-only masks; target mass is score-only after the barrier."""
    p,m,c=np.asarray(probability),np.asarray(mass),np.asarray(minimum_counts)
    require(p.dtype==m.dtype==np.float32 and p.shape==m.shape and p.ndim==3 and p.shape[-1]==4
        and c.shape==p.shape[:2] and np.isfinite(c).all() and np.all(c>=0)
        and np.all(c==np.floor(c)) and np.isfinite(p).all() and np.isfinite(m).all()
        and np.all(p>=0) and np.all(m>=0) and np.allclose(p.sum(-1),1.,rtol=1e-5,atol=1e-6),
        "Complete FP32 score block and integer observed callability counts")
    correct=np.take_along_axis(m,p.argmax(-1)[...,None],axis=-1)[...,0]
    logs=np.log(np.where(p>0,p,np.float32(1)))
    finite=(-m*logs).sum(-1,dtype=np.float32)
    infinite=np.where((p==0)&(m>0),m,np.float32(0)).sum(-1,dtype=np.float32)
    support=m.sum(-1,dtype=np.float32)
    return {name:dict(denominator_query_bp=float(np.sum(support,where=mask,dtype=np.float32)),
        correct_query_bp=float(np.sum(correct,where=mask,dtype=np.float32)),
        finite_NLL_sum=float(np.sum(finite,where=mask,dtype=np.float32)),
        infinite_NLL_query_bp=float(np.sum(infinite,where=mask,dtype=np.float32)),
        original_grid_query_cells=int(mask.sum()))
        for name,lo,hi in LOCAL_CALLABILITY_STRATA
        for mask in [(c>=lo) if hi is None else ((c>=lo)&(c<hi))]}


def local_callability_diagnostics_v1(bundle,parent,probability,primary_cache,guard):
    """Selected Present182 only; all membership derives from observed SNP calls."""
    bundle.validate()
    require(primary_cache.parent_id==parent["parent_id"] and probability.shape[0]==len(parent["positions"]),
        "Exact selected same-parent original-grid score cache")
    totals={name:{k:0 for k in ("denominator_query_bp","correct_query_bp","finite_NLL_sum",
        "infinite_NLL_query_bp","original_grid_query_cells")} for name,_,_ in LOCAL_CALLABILITY_STRATA}
    columns=primary_cache.columns
    for q0 in range(0,686,READOUT["query_block"]):
        guard();q1=min(q0+READOUT["query_block"],686)
        selected=np.flatnonzero((columns>=q0)&(columns<q1))
        if not len(selected):continue
        prefix=core.LocalReferencePrefixV1(bundle.markers,q0,q1)
        windows=bundle.flanks.window_index
        left=prefix.ranges(windows[:,0],windows[:,1])[...,4]
        right=prefix.ranges(windows[:,1],windows[:,2])[...,4]
        counts=torch.minimum(left,right).numpy()[:,columns[selected]-q0]
        partial=callability_block_sums_v1(probability[:,columns[selected]],primary_cache.mass[:,selected],counts)
        for name,values in partial.items():
            for key,value in values.items():totals[name][key]+=value
        del prefix,left,right,counts,partial
    strata={}
    for name,lo,hi in LOCAL_CALLABILITY_STRATA:
        value=totals[name];den=value["denominator_query_bp"];inf=value["infinite_NLL_query_bp"]
        strata[name]=dict(value,minimum_usable_sites_inclusive=lo,maximum_usable_sites_exclusive=hi,
            support_missing=den==0,accuracy=None if den==0 else value["correct_query_bp"]/den,
            nll=None if den==0 or inf else value["finite_NLL_sum"]/den,NLL_is_infinite=bool(inf))
    return dict(parent_id=parent["parent_id"],group="primary_present_day_182",strata=strata,
        membership_observations_only=True,selection_revised=False,probability_floor_used=False,
        definition="min_usable_SNPs_left_right_within_original_focal_hold")


def aggregate_callability_diagnostics_v1(cells):
    """Equal-parent summaries expose missing support and parent variability."""
    require(len(cells)==8 and {c["parent_id"] for c in cells}==set(VAL),"All8 selected VAL diagnostics")
    result={}
    for name,_,_ in LOCAL_CALLABILITY_STRATA:
        supported=[c["strata"][name] for c in cells if not c["strata"][name]["support_missing"]]
        value=dict(parents_with_support=len(supported),parents_without_support=8-len(supported),
            total_span_support_query_bp=sum(c["strata"][name]["denominator_query_bp"] for c in cells))
        for metric in ("accuracy","nll"):
            finite=[row[metric] for row in supported if row[metric] is not None]
            value[metric]=dict(equal_parent_mean=float(np.mean(finite)) if finite else None,
                parent_population_std=float(np.std(finite)) if finite else None,
                parent_min=min(finite) if finite else None,parent_max=max(finite) if finite else None,
                finite_parent_count=len(finite),
                infinite_parent_count=sum(row["NLL_is_infinite"] for row in supported) if metric=="nll" else 0)
            if metric=="nll" and any(row["NLL_is_infinite"] for row in supported):
                value[metric]["equal_parent_mean"]=None
        result[name]=value
    return dict(cells=cells,strata=result,aggregation="equal_parent_mean_over_parents_with_support",
        missing_support_excluded_and_counted=True,selected_models_only=True,selection_revised=False)


def run_evaluate_v1(ctx,input_ref):
    from .mutation_routed_full_panel_primary_v1 import FullPanelPrimaryValidationV1
    cfg,output,guard=ctx["cfg"],ctx["output"],ctx["guard"]
    reader=H046PhaseReaderV1(ctx["reader"],"evaluate",ctx["bindings"],
        virtual_prefix=str(output.path.relative_to(ROOT)/"H046_RAM_probability_packets"))
    train=reader.decode_json(ctx["train_completion_ref"])
    states,checkpoints=load_checkpoints_v1(reader,train,cfg)
    cache=ContinuousFlankCacheV1(ctx,reader)
    banks,parents,observation_seal=load_observations_v1(ctx,reader,VAL,cache)
    ram=DenseRAMMatrixV1(ctx,reader,parents,banks);reader.ram=ram
    models={a:core.ContinuousCladeFlankGNNV1(arm=a,**READOUT).to(ctx["device"],dtype=torch.float32)
            .eval().requires_grad_(False) for a in ARMS}
    for pid in VAL:
        cpu_bundle=cache.get(pid,banks[pid],parents[pid])
        bundle,_=device_parent_v1(ctx,cpu_bundle)
        for arm in ARMS:
            for at in PASSES:
                guard();models[arm].load_state_dict(states[arm,at],strict=True)
                with torch.no_grad(),torch.autocast(device_type=ctx["device"].type,enabled=False):
                    probability=models[arm](bundle).softmax(-1)
                ram.seal(arm,at,pid,probability.cpu().numpy(),checkpoints[arm,at]);del probability
                print(json.dumps(dict(stage="H046_VAL_prediction",sealed_products=len(ram.values),total_products=96)),flush=True)
        del bundle,cpu_bundle
    del models,states
    ram.verify();reader.truth_gate=True;ctx["VAL_truth_gate_opened"]=True
    selected,histories,full,metric_receipts,callability={},{},{},{},{}
    for arm in ARMS:
        validation=FullPanelPrimaryValidationV1(parents,ctx["bindings"],reader,fit.metric_helpers_v1(),cfg)
        rows=[]
        for at in PASSES:
            guard();cells,means=validation.score([(p,ram.records[arm,at,p]) for p in VAL])
            rows.append(dict(pass_index=at,cells=cells,equal_parent_metrics=means,
                checkpoint=checkpoints[arm,at],selection_rule=SELECTION))
        best=min(rows,key=lambda row:h038.selection_key_v1(row["equal_parent_metrics"],row["pass_index"]))
        cells,means=validation.full_final([(p,ram.records[arm,best["pass_index"],p]) for p in VAL])
        full[arm]=dict(cells=cells,equal_parent_metrics=means,selection_revised=False)
        selected[arm],histories[arm],metric_receipts[arm]=best,rows,validation.receipt()
        callability_cells=[]
        for pid in VAL:
            observed=cache.get(pid,banks[pid],parents[pid])
            callability_cells.append(local_callability_diagnostics_v1(observed,parents[pid],
                ram.grid(ram.records[arm,best["pass_index"],pid],parents[pid]),validation.caches[pid],guard))
            del observed
        callability[arm]=aggregate_callability_diagnostics_v1(callability_cells)
    result=dict(schema="H046_paired_VAL_completion_v1",hypothesis_id="H046",
        status="completed_original8VAL_paired_development_selection",input_ref=input_ref,
        train_completion_ref=ctx["train_completion_ref"],observations_seal=observation_seal,
        selection_rule=SELECTION,selected=selected,history=histories,selected_full_diagnostics=full,
        metric_receipts=metric_receipts,registered_practical_criteria=practical_criteria_v1(full),
        selected_local_callability_diagnostics=callability,
        optimization_performed=False,all96_RAM_seals_before_any_VAL_truth=True,
        RAM_dense_probability_bytes=ram.bytes,complete_probability_matrix_manifest_ref=ram.barrier_ref,
        virtual_seals_durable_in_matrix_manifest=True,dense_prediction_arrays_saved=False,
        outer_truth_read=False,same_VAL_is_development_evidence=True)
    output.save("completed.json",result);return result


def run_context_v1(ctx,input_ref):
    require(ctx.get("root_authorizer")=="root_agent" and ctx.get("admitted") is True
        and ctx.get("phase") in ("train","evaluate") and ctx.get("arm")=="joint_pair"
        and callable(ctx.get("guard")) and callable(ctx.get("before_flank_build")),
        "Independently admitted paired root context")
    validate_protocol(ctx["cfg"])
    for ref in (*SCIENTIFIC_REFS,REGISTRATION_REF,DESIGN_REF):bound(ref)
    require(ctx["reader"].root.resolve()==ROOT and ctx["device"].type=="cuda"
        and ctx["reader"].decode_json(ctx["config_ref"])==ctx["cfg"],
        "Qualified root-leased CUDA reader and exact config")
    require(ctx["cfg"]["preparation_completion_ref"]==ctx["preparation_completion_ref"],
            "Exact original immutable H044 preparation")
    ids=TRAIN if ctx["phase"]=="train" else VAL
    require(set(ctx["bindings"])==set(ids) and set(ctx["entry_refs"])==set(TRAIN+VAL),
            "Exact phase targets and40 original observations")
    science();ctx["guard"]()
    if ctx["phase"]=="train":return run_train_v1(ctx,input_ref)
    try:return run_evaluate_v1(ctx,input_ref)
    except Exception as error:
        ctx["output"].save("failed.json",dict(schema="H046_paired_failure_v1",hypothesis_id="H046",
            phase="evaluate",status="failed_partial_preserved",error_type=type(error).__name__,error=str(error),
            VAL_truth_gate_opened=ctx.get("VAL_truth_gate_opened",False),outer_truth_read=False,automatic_replay=False))
        raise
