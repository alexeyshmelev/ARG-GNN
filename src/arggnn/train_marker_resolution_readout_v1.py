"""H044 matched localized/tree-pooled FP32 TRAIN and all96-sealed original VAL.

Inert by default. Existing root context/guard/launcher owns every admission.
Targets stay outside the model, with one complete original-grid parent mass
streamed once and shared by both independently optimized arms.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,io,json,random,time
from collections import OrderedDict
from pathlib import Path
from . import train_anonymous_clade_flag_control_v1 as h42

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE="src/arggnn/train_marker_resolution_readout_v1.py"
CORE_REF=dict(path="src/arggnn/marker_resolution_readout_v1.py",
    sha256="27ecd30683024def2d408bf16250102b3252d0ae112687b9664fca086b141310")
WARM_CHECKPOINT=dict(path="experiments/results/H042_GPU4_clade_flags_train_seed101_v1/exact_clade_flags.pass_30.pt",
    sha256="520fcffa1d5e85603620a7f4688b7fdf46669686fea7960fc311852ff2f7e366")
WARM_COMPLETION=dict(path="experiments/results/H042_GPU4_clade_flags_train_seed101_v1/completed.json",
    sha256="5b7d7adab0de14c18016db86bfc18ddc2b1d652ff530fadd101aaa73c90a6984")
ARMS=("localized","tree_pooled")
TRAIN,VAL,PASSES=h42.TRAIN,h42.VAL,h42.PASSES
SELECTION=h42.SELECTION
require,bound=h42.require,h42.bound

def science():
    global np,torch,h038,fit,core,memory_bounds
    h42.science();np,torch,h038,fit=h42.np,h42.torch,h42.h038,h42.fit
    from . import marker_resolution_readout_v1 as core
    from . import marker_resolution_memory_bounds_v2 as memory_bounds

def validate_protocol_v1(cfg):
    require(cfg.get("schema")=="H044_marker_execution_config_v1" and cfg.get("hypothesis_id")=="H044"
        and cfg.get("arms")==list(ARMS) and cfg.get("seed")==101 and cfg.get("dtype")=="float32"
        and cfg.get("passes")==30 and cfg.get("updates_per_pass")==32 and cfg.get("updates_per_arm")==960
        and cfg.get("checkpoint_passes")==list(PASSES)
        and cfg.get("TRAIN_parent_ids")==list(TRAIN) and cfg.get("VAL_parent_ids")==list(VAL)
        and cfg.get("parameters_per_arm")==31428 and cfg.get("base_parameters")==30788
        and cfg.get("added_parameters")==640 and cfg.get("model")==h42.MODEL
        and cfg.get("learning_rate")==.001 and cfg.get("weight_decay")==.0001
        and cfg.get("AdamW_betas")==[.9,.999] and cfg.get("AdamW_epsilon")==1e-8
        and cfg.get("gradient_clip_norm")==1. and cfg.get("early_stopping") is False
        and cfg.get("VAL_influences_optimization") is False and cfg.get("selection")==SELECTION
        and cfg.get("metrics")==h42.METRICS and cfg.get("core_ref")==CORE_REF
        and cfg.get("warmstart_checkpoint_ref")==WARM_CHECKPOINT and cfg.get("warmstart_completion_ref")==WARM_COMPLETION
        and cfg.get("readout")==dict(query_block=64,center_block=2048,checkpoint_readout=True)
        and cfg.get("TRAIN_target_policy")=="one_complete_original100bp_parent_mass_streamed_once_shared_between_arms"
        and cfg.get("dense_prediction_arrays_saved") is False and cfg.get("outer_truth_read") is False
        and cfg.get("same_VAL_is_development_evidence") is True,
        "Exact fixed H044 matched30pass/original-grid protocol")
    for name in ("CPU_bank_cap_bytes","CPU_target_cap_bytes","flag_cache_cap_bytes",
        "flag_preparation_cap_bytes","GPU_residency_cap_bytes","GPU_flag_cap_bytes",
        "RAM_prediction_cap_bytes","marker_cache_cap_bytes","marker_preparation_cap_bytes",
        "CPU_coordinate_bulk_cap_bytes","CPU_binding_bulk_cap_bytes"):
        require(type(cfg.get(name)) is int and cfg[name]>0,"Positive root-reviewed cap: "+name)

def marker_preparation_allowance_v1(s,h,p,*,b=64,c=4):
    """Conservative named CPU tensor allowance; allocator/native Python peak unmeasured."""
    evidence=5*s*h+512*s+16*h
    query_features=40*s*b*c+128*p*b*c+64*b*c
    return dict(evidence_construction_named_allowance_bytes=evidence,
        query_prefix_window_feature_named_allowance_bytes=query_features,
        maximum_named_allowance_bytes=max(evidence,query_features),
        allocator_and_native_backbone_peak_unmeasured=True)

class H044PhaseReaderV1(h42.PhaseReader):
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
        require(packet["schema"]=="H044_marker_original_grid_RAM_seal_v1"
            and packet["hypothesis_id"]=="H044" and packet["arm"]==arm
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

class MarkerCacheV1:
    """Bounded evidence LRU; all observation recipes sealed before any phase truth."""
    def __init__(self,ctx,reader):
        self.ctx,self.reader,self.cfg=ctx,reader,ctx["cfg"]
        self.values,self.proofs=OrderedDict(),{}
        self.bytes,self.peak,self.builds,self.hits=0,0,0,0
        self.cache_metadata=reader.decode_json(ctx["observation_cache_ref"])
        preparation=ctx["preparation"];ids=tuple(TRAIN)+tuple(VAL)
        require(preparation["artifact_format"]=="H044_all40_batched_coordinate_binding_v2"
            and set(preparation["parents"])==set(ids),"Exact all40 batched observation preparation")
        coordinates_ref=preparation["coordinates_ref"];bindings_ref=preparation["bindings_ref"]
        expected={p+"__"+k for p in ids for k in
            ("positions_bp0","centers_bp","bin_edges_bp","intervals_bp","allele_cardinality")}
        with np.load(io.BytesIO(reader.read(coordinates_ref)),allow_pickle=False) as archive:
            require(set(archive.files)==expected,"Exact200 observation-only archive members")
            self.coordinates={}
            coordinate_bytes=0
            for member in archive.files:
                value=np.array(archive[member],copy=True,order="C")
                coordinate_bytes+=int(value.nbytes)
                require(coordinate_bytes<=self.cfg["CPU_coordinate_bulk_cap_bytes"],"Decoded all40 coordinate CPU cap")
                self.coordinates[member]=torch.from_numpy(value)
        require(coordinate_bytes==preparation["retained_coordinate_bytes"],"Exact prepared bulk coordinate bytes")
        bulk=reader.decode_json(bindings_ref)
        require(bulk["schema"]=="H044_all40_original_observed_marker_bindings_v2"
            and set(bulk["parents"])==set(ids) and bulk["observations_only"] is True
            and bulk["targets_read"] is False,"Exact all40 original marker binding container")
        binding_bytes=len(json.dumps(bulk["parents"],sort_keys=True,separators=(",",":"),allow_nan=False).encode())
        require(binding_bytes==preparation["retained_binding_JSON_bytes"]
            and binding_bytes<=self.cfg["CPU_binding_bulk_cap_bytes"],"Exact bounded all40 binding metadata bytes")
        self.bindings=bulk["parents"]
        for pid in ids:
            entry=preparation["parents"][pid]
            require(entry["coordinates_ref"]==coordinates_ref and entry["binding_ref"]==bindings_ref
                and entry["binding_member_selector"]==pid
                and entry["coordinates_member_selector"]=={k:pid+"__"+k for k in
                    ("positions_bp0","centers_bp","bin_edges_bp","intervals_bp","allele_cardinality")},
                "Exact parent selectors into shared immutable coordinate/binding products")
    def get(self,pid,bank,parent):
        self.ctx["guard"]()
        if pid in self.values:
            bundle=self.values[pid];bundle.validate();self.values.move_to_end(pid);self.hits+=1
            return bundle
        entry=self.ctx["preparation"]["parents"][pid]
        binding=self.bindings[entry["binding_member_selector"]]
        require(binding["parent_id"]==pid and binding["observations_only"] is True and binding["targets_read"] is False
            and binding["support_rule"]=="original_allele_cardinality"
            and binding["original_observed_entry_ref"]==self.ctx["entry_refs"][pid]
            and binding["original_cache_manifest_ref"]==self.ctx["observation_cache_ref"]
            and binding["original_cache_array_ref"]==entry["original_cache_array_ref"]
                ==self.cache_metadata["parents"][pid]["array_ref"]
            and binding["original_loader_source_ref"]==core.ORIGINAL_LOADER_REF
            and binding["original_loader_source_closure"]==self.ctx["observation_source_refs"]
            and all(binding["original_loader_cache_checks"].values())
            and binding["original_loader_receipt"]["targets_read"] is False
            and binding["original_loader_receipt"]["entry_ref"]==self.ctx["entry_refs"][pid],
            "Actual original loader/cache/entry and prepared sidecar provenance")
        arrays={key:self.coordinates[member] for key,member in entry["coordinates_member_selector"].items()}
        require(np.array_equal(arrays["centers_bp"].numpy(),parent["positions"])
            and np.array_equal(arrays["bin_edges_bp"].numpy(),parent["edges"]),
            "Original H042 cached scoring coordinates equal prepared physical sidecar")
        # A private decoded mapping/array copy is retained; consumers never mutate
        # NumPy aliases, .data, metadata, the original bank or prepared tensors.
        sidecar=core.MarkerCoordinatesV1(**arrays,binding=binding)
        limits=memory_bounds.logical_bounds_v2(bank.genotype.shape[0],len(bank.spans),len(parent["positions"]),
            genotype_bytes=bank.genotype.element_size())
        # Retained evidence plus CPU query-block prefix/features/temporaries.
        allowance=marker_preparation_allowance_v1(bank.genotype.shape[0],bank.genotype.shape[1],len(parent["positions"]))
        require(allowance["maximum_named_allowance_bytes"]<=self.cfg["marker_preparation_cap_bytes"],
            "Count-derived named marker construction/feature allowance exceeds root cap")
        markers=core.prepare_marker_evidence_v1(bank,sidecar)
        bundle=core.bind_marker_bank_v1(bank,markers);size=markers.tensor_bytes()
        require(size==entry["marker_tensor_bytes"] and size<=self.cfg["marker_cache_cap_bytes"],
            "Exact one-parent marker bytes and bounded LRU cap")
        proof=dict(coordinates_ref=entry["coordinates_ref"],binding_ref=entry["binding_ref"],
            coordinates_member_selector=entry["coordinates_member_selector"],binding_member_selector=entry["binding_member_selector"],
            binding_sha256=hashlib.sha256(json.dumps(binding,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
            marker_tensor_bytes=size,counts=entry["counts"],logical_bounds=limits,preparation_allowance=allowance)
        if pid in self.proofs:require(self.proofs[pid]==proof,"Rebuilt observation recipe changed after truth barrier")
        else:self.proofs[pid]=proof
        while self.values and self.bytes+size>self.cfg["marker_cache_cap_bytes"]:
            _,old=self.values.popitem(last=False);self.bytes-=old.markers.tensor_bytes()
        self.values[pid]=bundle;self.bytes+=size;self.peak=max(self.peak,self.bytes);self.builds+=1
        self.ctx["guard"]()
        return bundle

def load_observations_v1(ctx,reader,ids,flags):
    banks,parents,keys,flag_seal=h42.load_banks(ctx,reader,ids,flags)
    markers=MarkerCacheV1(ctx,reader)
    for pid in ids:markers.get(pid,banks[pid],parents[pid])
    require(set(markers.proofs)==set(ids) and set(flags.proofs)==set(ids),"Complete phase observation recipe barrier")
    seal=ctx["output"].save("all_%d_H044_observation_recipes.json"%len(ids),dict(
        schema="H044_marker_complete_observation_barrier_v1",hypothesis_id="H044",parent_ids=list(ids),
        preparation_completion_ref=ctx["preparation_completion_ref"],
        preparation_actual_wait_ref=ctx["preparation_actual_wait_ref"],flag_seal=flag_seal,
        marker_recipes=markers.proofs,all_observations_precede_phase_truth=True,
        truth_read=False,full_backbone_trees_retained=True,marker_arrays_durable=False))
    return banks,parents,keys,markers,seal

def original_mass_v1(ctx,reader,pid,parent):
    from .mutation_routed_full_panel_primary_v1 import FullPanelExternalTargetV1
    require(12*len(parent["positions"])*686*4<=ctx["cfg"]["CPU_target_cap_bytes"],
        "Preflight one-parent FP64 plus FP32 physical target construction cap")
    target=FullPanelExternalTargetV1(parent,ctx["bindings"][pid],"TRAIN_loss",reader)
    raw=target.target_mass(parent,fit.metric_helpers_v1())
    require(raw.dtype==np.float64 and raw.shape==(len(parent["positions"]),686,4),
        "Complete original100bp target mass, no native-hold aggregation")
    # Bound simultaneous FP64 construction and final FP32 array, not32 retained parents.
    require(raw.nbytes+raw.size*4<=ctx["cfg"]["CPU_target_cap_bytes"],"One-parent target construction cap")
    mass=np.array(raw,dtype=np.float32,copy=True,order="C")
    del raw,target
    return torch.from_numpy(mass)

def update_v1(model,optimizer,bundle,mass,flags,guard):
    guard();model.train();optimizer.zero_grad(set_to_none=True);start=time.monotonic()
    with torch.autocast(device_type="cuda",enabled=False):
        logits=model(bundle,flags=flags)
        loss=core.original_grid_span_nll_v1(logits,mass,bundle.markers.bin_widths_bp.to(mass.device))
    require(bool(torch.isfinite(loss)),"Finite complete-parent FP32 loss")
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    optimizer.step()
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()),"Finite trained weights")
    row=dict(loss=float(loss.detach()),gradient_norm=float(norm),seconds=time.monotonic()-start)
    optimizer.zero_grad(set_to_none=True);del logits,loss,norm;guard()
    return row

def device_parent_v1(ctx,bundle,flags_cpu,mass_cpu=None):
    flag_bytes=flags_cpu.tensor_bytes()
    require(flag_bytes<=ctx["cfg"]["GPU_flag_cap_bytes"],"Complete original GPU flags cap")
    needed=h038.tensor_bytes_v1(bundle.source_bank)+flag_bytes+(0 if mass_cpu is None else mass_cpu.numel()*4)
    require(needed<=ctx["cfg"]["GPU_residency_cap_bytes"],"One complete bank+flags+grid mass input residency cap")
    moved=bundle.to(ctx["device"]);flags=flags_cpu.to(ctx["device"])
    mass=None if mass_cpu is None else mass_cpu.to(ctx["device"])
    return moved,flags,mass

def run_train_v1(ctx,input_ref):
    cfg,output,guard,device=ctx["cfg"],ctx["output"],ctx["guard"],ctx["device"]
    reader=h42.PhaseReader(ctx["reader"],"train",ctx["bindings"])
    flag_cache=h42.FlagCache(cfg,guard);steps={a:0 for a in ARMS};history=[];checkpoints={}
    try:
        banks,parents,keys,marker_cache,observation_seal=load_observations_v1(ctx,reader,TRAIN,flag_cache)
        completion=reader.decode_json(core.H042_COMPLETION_REF)
        require(completion["schema"]=="anonymous_clade_flag_train_completion_v1"
            and completion["status"]=="completed_fixed_30_pass_flag_TRAIN"
            and completion["hypothesis_id"]=="H042" and completion["arm"]=="exact_clade_flags"
            and completion["steps"]=={"exact_clade_flags":960} and len(completion["history"])==30
            and completion["VAL_truth_read"] is False and completion["outer_truth_read"] is False,
            "Exact completed original H042 warmstart metadata")
        payload=torch.load(io.BytesIO(reader.read(core.H042_PASS30_REF)),map_location="cpu",weights_only=False)
        models,initialization=core.make_matched_pair_v1(payload,
            checkpoint_ref=core.H042_PASS30_REF,completion_ref=core.H042_COMPLETION_REF)
        del payload
        initial_hash=h42.shared.state_identity(models["localized"].state_dict())
        require(initial_hash==h42.shared.state_identity(models["tree_pooled"].state_dict()),"Exactly identical full initialization")
        models={a:m.to(device=device,dtype=torch.float32) for a,m in models.items()}
        optimizers={a:torch.optim.AdamW(models[a].parameters(),lr=.001,weight_decay=.0001,
            betas=(.9,.999),eps=1e-8) for a in ARMS}
        require(all(len(o.state)==0 for o in optimizers.values()),"All AdamW moments and steps reset")
        output.save("initialization.json",dict(schema="H044_marker_paired_initialization_v1",hypothesis_id="H044",
            initialization=initialization,initial_state_sha256=initial_hash,observation_seal=observation_seal,
            optimizer_moments_and_step_reset=True,all_H042_weights_trainable=True,
            one_complete_grid_mass_shared_between_arms=True,VAL_influences_optimization=False,
            same_VAL_is_development_evidence=True))
        reader.truth_gate=True
        for at in range(1,31):
            order=list(TRAIN);random.Random(101+at).shuffle(order)
            means={a:0. for a in ARMS};start=time.monotonic()
            for number,pid in enumerate(order,1):
                guard()
                cpu_bundle=marker_cache.get(pid,banks[pid],parents[pid])
                flags_cpu=flag_cache.get(pid,banks[pid],keys[pid])
                mass_cpu=original_mass_v1(ctx,reader,pid,parents[pid])
                bundle,flags,mass=device_parent_v1(ctx,cpu_bundle,flags_cpu,mass_cpu)
                for arm in ARMS:
                    row=update_v1(models[arm],optimizers[arm],bundle,mass,flags,guard)
                    steps[arm]+=1;means[arm]+=row["loss"]/32.
                # Release one shared complete-parent target before the next parent.
                del bundle,flags,mass,mass_cpu,cpu_bundle,flags_cpu
                if number%8==0:print(json.dumps(dict(stage="H044_paired_TRAIN",pass_index=at,
                    complete_parents=number,steps=steps,elapsed_seconds=time.monotonic()-start)),flush=True)
            require(steps=={a:32*at for a in ARMS},"Exactly one complete-parent update per arm")
            row=dict(pass_index=at,steps=dict(steps),mean_TRAIN_NLL=means,
                full_parent_updates_per_arm=32,seconds=time.monotonic()-start)
            history.append(row);print(json.dumps(dict(stage="H044_TRAIN_pass_completed",**row)),flush=True)
            if at in PASSES:
                checkpoints[str(at)]={}
                for arm in ARMS:
                    checkpoint=dict(schema="H044_marker_checkpoint_v1",hypothesis_id="H044",arm=arm,
                        pass_index=at,additional_optimizer_steps=steps[arm],seed=101,parameter_count=31428,
                        model_state={k:v.detach().cpu() for k,v in models[arm].state_dict().items()},
                        optimizer_state=optimizers[arm].state_dict(),optimizer_state_reset=True,
                        all_H042_weights_trainable=True,teacher_used=False,initial_state_sha256=initial_hash,
                        input_ref=input_ref,config_ref=ctx["config_ref"],core_ref=CORE_REF,
                        observation_seal=observation_seal,preparation_completion_ref=ctx["preparation_completion_ref"],
                        warmstart_checkpoint_ref=core.H042_PASS30_REF,warmstart_completion_ref=core.H042_COMPLETION_REF,
                        readout=cfg["readout"],same_VAL_is_development_evidence=True)
                    checkpoints[str(at)][arm]=output.save("%s.pass_%02d.pt"%(arm,at),checkpoint,"torch")
        require(steps=={a:960 for a in ARMS} and set(checkpoints)=={str(p) for p in PASSES},"Fixed paired schedule completed")
        result=dict(schema="H044_marker_paired_train_completion_v1",hypothesis_id="H044",
            status="completed_fixed30pass_paired_TRAIN",input_ref=input_ref,config_ref=ctx["config_ref"],
            arms=list(ARMS),steps=steps,history=history,checkpoint_groups=checkpoints,core_ref=CORE_REF,
            initial_state_sha256=initial_hash,initialization=initialization,observations_seal=observation_seal,
            preparation_completion_ref=ctx["preparation_completion_ref"],warmstart_checkpoint_ref=core.H042_PASS30_REF,
            warmstart_completion_ref=core.H042_COMPLETION_REF,VAL_truth_read=False,outer_truth_read=False,
            selection_deferred=True,same_VAL_is_development_evidence=True,
            marker_cache_peak_bytes=marker_cache.peak,marker_builds=marker_cache.builds,
            marker_cache_hits=marker_cache.hits,flag_cache_peak_bytes=flag_cache.peak_runtime_bytes,
            target_residency_policy=cfg["TRAIN_target_policy"])
        output.save("completed.json",result);return result
    except Exception as error:
        output.save("failed.json",dict(schema="H044_marker_paired_failure_v1",hypothesis_id="H044",phase="train",
            status="failed_partial_preserved",error_type=type(error).__name__,error=str(error),
            logical_steps=steps,completed_checkpoint_groups=checkpoints,automatic_replay=False,
            VAL_truth_read=False,outer_truth_read=False))
        raise

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
        packet=dict(schema="H044_marker_original_grid_RAM_seal_v1",hypothesis_id="H044",arm=arm,pass_index=at,
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
            schema="H044_marker_all96_VAL_barrier_v1",hypothesis_id="H044",arms=list(ARMS),
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

def run_evaluate_v1(ctx,input_ref):
    from .mutation_routed_full_panel_primary_v1 import FullPanelPrimaryValidationV1
    reader=H044PhaseReaderV1(ctx["reader"],"evaluate",ctx["bindings"],
        virtual_prefix=str(ctx["output"].path.relative_to(ROOT)/"H044_RAM_probability_packets"))
    cfg,output,guard=ctx["cfg"],ctx["output"],ctx["guard"]
    train=reader.decode_json(ctx["train_completion_ref"])
    require(train["schema"]=="H044_marker_paired_train_completion_v1" and train["hypothesis_id"]=="H044"
        and train["status"]=="completed_fixed30pass_paired_TRAIN" and train["arms"]==list(ARMS)
        and train["steps"]=={a:960 for a in ARMS} and train["core_ref"]==CORE_REF
        and train["VAL_truth_read"] is False and train["outer_truth_read"] is False
        and train["preparation_completion_ref"]==ctx["preparation_completion_ref"],
        "Actual complete paired TRAIN with original preparation")
    states,checkpoints={},{}
    for arm in ARMS:
        for at in PASSES:
            guard();ref=train["checkpoint_groups"][str(at)][arm]
            cp=torch.load(io.BytesIO(reader.read(ref)),map_location="cpu",weights_only=False)
            require(cp["schema"]=="H044_marker_checkpoint_v1" and cp["hypothesis_id"]=="H044"
                and cp["arm"]==arm and cp["pass_index"]==at and cp["additional_optimizer_steps"]==32*at
                and cp["parameter_count"]==31428 and cp["seed"]==101 and cp["teacher_used"] is False
                and cp["optimizer_state_reset"] is True and cp["all_H042_weights_trainable"] is True
                and cp["core_ref"]==CORE_REF and cp["readout"]==cfg["readout"]
                and cp["initial_state_sha256"]==train["initial_state_sha256"]
                and cp["input_ref"]==train["input_ref"] and cp["config_ref"]==train["config_ref"]
                and cp["preparation_completion_ref"]==ctx["preparation_completion_ref"]
                and cp["warmstart_checkpoint_ref"]==core.H042_PASS30_REF
                and cp["warmstart_completion_ref"]==core.H042_COMPLETION_REF
                and all(isinstance(v,torch.Tensor) and v.dtype==torch.float32 for v in cp["model_state"].values()),
                "Exact paired FP32 checkpoint provenance")
            states[arm,at],checkpoints[arm,at]=cp["model_state"],ref;del cp
    flags=h42.FlagCache(cfg,guard)
    banks,parents,keys,markers,observation_seal=load_observations_v1(ctx,reader,VAL,flags)
    ram=DenseRAMMatrixV1(ctx,reader,parents,banks);reader.ram=ram
    models={a:core.MarkerResolutionReadoutV1(arm=a,**cfg["readout"]).to(ctx["device"],dtype=torch.float32)
        .eval().requires_grad_(False) for a in ARMS}
    for pid in VAL:
        cpu_bundle=markers.get(pid,banks[pid],parents[pid]);flags_cpu=flags.get(pid,banks[pid],keys[pid])
        bundle,device_flags,_=device_parent_v1(ctx,cpu_bundle,flags_cpu)
        for arm in ARMS:
            for at in PASSES:
                guard();models[arm].load_state_dict(states[arm,at],strict=True)
                with torch.no_grad(),torch.autocast(device_type="cuda",enabled=False):
                    probabilities=models[arm](bundle,flags=device_flags).softmax(-1)
                ram.seal(arm,at,pid,probabilities.cpu().numpy(),checkpoints[arm,at]);del probabilities
                print(json.dumps(dict(stage="H044_VAL_prediction",sealed_products=len(ram.values),total_products=96)),flush=True)
        del bundle,device_flags,cpu_bundle,flags_cpu
    del models,states
    ram.verify();reader.truth_gate=True;ctx["VAL_truth_gate_opened"]=True
    selected,histories,full,metric_receipts={},{},{},{}
    for arm in ARMS:
        validation=FullPanelPrimaryValidationV1(parents,ctx["bindings"],reader,fit.metric_helpers_v1(),cfg)
        rows=[]
        for at in PASSES:
            guard();cells,means=validation.score([(p,ram.records[arm,at,p]) for p in VAL])
            row=dict(pass_index=at,cells=cells,equal_parent_metrics=means,
                checkpoint=checkpoints[arm,at],selection_rule=SELECTION)
            rows.append(row)
        best=min(rows,key=lambda row:h038.selection_key_v1(row["equal_parent_metrics"],row["pass_index"]))
        cells,means=validation.full_final([(p,ram.records[arm,best["pass_index"],p]) for p in VAL])
        full[arm]=dict(cells=cells,equal_parent_metrics=means,selection_revised=False)
        selected[arm],histories[arm],metric_receipts[arm]=best,rows,validation.receipt()
    result=dict(schema="H044_marker_paired_VAL_completion_v1",hypothesis_id="H044",
        status="completed_original8VAL_paired_development_selection",input_ref=input_ref,
        train_completion_ref=ctx["train_completion_ref"],observations_seal=observation_seal,
        selection_rule=SELECTION,selected=selected,history=histories,selected_full_diagnostics=full,
        metric_receipts=metric_receipts,optimization_performed=False,
        all96_RAM_seals_before_any_VAL_truth=True,RAM_dense_probability_bytes=ram.bytes,
        complete_probability_matrix_manifest_ref=ram.barrier_ref,virtual_seals_durable_in_matrix_manifest=True,
        dense_prediction_arrays_saved=False,outer_truth_read=False,same_VAL_is_development_evidence=True)
    output.save("completed.json",result);return result

def run_context_v1(ctx,input_ref):
    require(ctx.get("root_authorizer")=="root_agent" and ctx.get("admitted") is True
        and ctx.get("phase") in ("train","evaluate") and ctx.get("arm")=="joint_pair"
        and callable(ctx.get("guard")),"Admitted paired root context")
    validate_protocol_v1(ctx["cfg"]);bound(CORE_REF)
    require(ctx["reader"].root.resolve()==ROOT and ctx["device"].type=="cuda"
        and ctx["reader"].decode_json(ctx["config_ref"])==ctx["cfg"],"Qualified leased CUDA reader and exact config")
    ids=TRAIN if ctx["phase"]=="train" else VAL
    require(set(ctx["bindings"])==set(ids) and set(ctx["entry_refs"])==set(TRAIN+VAL),"Exact phase-only targets and40observations")
    science();ctx["guard"]()
    if ctx["phase"]=="train":return run_train_v1(ctx,input_ref)
    try:return run_evaluate_v1(ctx,input_ref)
    except Exception as error:
        ctx["output"].save("failed.json",dict(schema="H044_marker_paired_failure_v1",hypothesis_id="H044",
            phase="evaluate",status="failed_partial_preserved",error_type=type(error).__name__,error=str(error),
            VAL_truth_gate_opened=ctx.get("VAL_truth_gate_opened",False),outer_truth_read=False,automatic_replay=False))
        raise

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true");parser.add_argument("--input")
    parser.add_argument("--expected-input-sha256");args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="source_only_not_executed",source=SOURCE,arms=list(ARMS),
            passes=30,updates_per_arm=960,checkpoint_passes=list(PASSES),VAL_products=96,
            root_context_required=True,outer_truth_read=False)));return 0
    require(args.input and args.expected_input_sha256,"Pinned active root paired capability")
    record=dict(path=args.input,sha256=args.expected_input_sha256);spec=json.loads(bound(record))
    require(spec.get("schema")=="H044_marker_root_execution_input_v1" and spec.get("hypothesis_id")=="H044"
        and spec.get("root_authorizer")=="root_agent" and spec.get("status")=="active"
        and spec.get("phase") in ("train","evaluate"),"Active root paired phase capability")
    ref=spec["root_context_module_ref"];bound(ref)
    require(ref["path"].startswith("research/reviews/") and ref["path"].endswith(".py"),"Separate pinned root context")
    module_spec=importlib.util.spec_from_file_location("_H044_paired_context",ROOT/ref["path"])
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    run_context_v1(module.build_context_v1(record,spec),record);return 0

if __name__=="__main__":raise SystemExit(main())
