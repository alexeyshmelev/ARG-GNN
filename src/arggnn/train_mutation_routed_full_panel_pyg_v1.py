"""Inactive by default H038 full952/32TRAIN/8VAL FP32 PyG worker."""
from __future__ import annotations
import argparse
from collections import OrderedDict
from dataclasses import fields,replace
import hashlib
import io
import json
import os
from pathlib import Path
import random
import resource
import time
import numpy as np
import torch
from . import census_tsinfer_fit_v1 as fit
from . import train_mutation_routed_whole_tree_gnn_v1 as original
from .project_storage_closed_scopes_v1 import PilotOutputV9,budget_check_v9,validate_storage_v9
from .mutation_routed_full_panel_observations_v1 import validate_registry_v1
from .mutation_routed_full_panel_cache_v1 import (
    FullPanelParentDatasetV1,make_parent_dataloader_v1,shutdown_parent_dataloader_v1,read_full_cache_index_v1)
from .mutation_routed_full_panel_primary_v1 import FullPanelExternalTargetV1,FullPanelPrimaryValidationV1,PRIMARY
from .mutation_routed_full_panel_pyg_model_v1 import MutationRoutedFullPanelPyGGNNV1,prepare_control_bank_v1,ARMS
SOURCE="src/arggnn/train_mutation_routed_full_panel_pyg_v1.py"
CONFIG="configs/mutation_routed_full_panel_pyg_v1.json"
NEW_SOURCES=tuple("src/arggnn/"+name for name in (
    "mutation_routed_full_panel_observations_v1.py","mutation_routed_full_panel_cache_v1.py",
    "mutation_routed_full_panel_primary_v1.py","mutation_routed_full_panel_pyg_model_v1.py",
    "train_mutation_routed_full_panel_pyg_v1.py"))
ROOT=Path(__file__).resolve().parents[2]


IMPLEMENTATION_SOURCES=frozenset(NEW_SOURCES)|frozenset({
    "src/arggnn/mutation_routed_pyg_gnn_v1.py","src/arggnn/mutation_routed_whole_tree_gnn_v1.py",
    "src/arggnn/mutation_routed_tree_observations_v1.py","src/arggnn/mutation_routed_parent_loader_v1.py",
    "src/arggnn/mutation_routed_observation_efficiency_v1.py","src/arggnn/mutation_routed_observation_cache_v1.py",
    "src/arggnn/mutation_routed_pyg_dataset_v1.py","src/arggnn/mutation_routed_fast_primary_v1.py",
    "src/arggnn/observed_variant_residual_v1.py","src/arggnn/full_panel_native_baseline_contracts_v1.py",
    "src/arggnn/census_tsinfer_observation_v1.py",original.SOURCE,
    "src/arggnn/prepare_expanded_full_panel_v1.py","src/arggnn/prepare_expanded_pyg_observations_v1.py",
    "src/arggnn/project_storage_closed_scopes_v1.py"})
HELPER_SOURCE_PATHS={
    "H025_score_helpers":"src/arggnn/train_persistence_trace_pilot_v1.py",
    "actual_full_storage_launcher":"scripts/launch_reserved_gpu_full_storage_v1.py",
    "configure_fit":"src/arggnn/biological_candidate.py",
    "exact_observation_efficiency":"src/arggnn/mutation_routed_observation_efficiency_v1.py",
    "full_reservation_adapter":"src/arggnn/project_storage_closed_scopes_v1.py",
    "immutable_base_launcher":"scripts/launch_reserved_gpu_v6.py",
    "immutable_budget_core":"scripts/project_budget_v9.py",
    "lease_title":"src/arggnn/native_argformer_exec.py",
    "metrics":"scripts/train_ancestralpaths_biological_v1.py",
    "original_initial_driver":original.SOURCE,
    "original_observed_loader":"src/arggnn/mutation_routed_parent_loader_v1.py",
    "span_loss":"src/arggnn/biological_gst_control_v1.py",
    "targets_scores":"src/arggnn/census_tsinfer_fit_v1.py",
    "original_full_reservation_adapter":"src/arggnn/project_storage_full_reservation_v1.py",
}
ALLOWED_SOURCE_PATHS=IMPLEMENTATION_SOURCES|frozenset(HELPER_SOURCE_PATHS.values())


def validated_source_records_v1(spec,cfg):
    """Validate the whole source boundary before any implementation/helper read."""
    sources=spec.get("source_sha256")
    if not isinstance(sources,dict) or set(sources)!=ALLOWED_SOURCE_PATHS:
        raise ValueError("Exact reviewed Python implementation/helper source closure required")
    for path,digest in sources.items():
        if (not isinstance(path,str) or not path.endswith(".py")
                or not path.startswith(("src/arggnn/","scripts/"))
                or Path(path).as_posix()!=path or ".." in Path(path).parts
                or not isinstance(digest,str) or len(digest)!=64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("Canonical Python source path and SHA256 required")
    helpers=cfg.get("helper_refs")
    if not isinstance(helpers,dict) or set(helpers)!=set(HELPER_SOURCE_PATHS):
        raise ValueError("Exact reviewed helper names required")
    for name,path in HELPER_SOURCE_PATHS.items():
        if helpers[name]!={"path":path,"sha256":sources[path]}:
            raise ValueError("Helper refs must match their fixed implementation source paths")
    own=cfg.get("source_refs")
    if (not isinstance(own,list) or len(own)!=len(NEW_SOURCES)
            or {r.get("path") for r in own if isinstance(r,dict)}!=set(NEW_SOURCES)
            or any(r!={"path":r.get("path"),"sha256":sources.get(r.get("path"))} for r in own)):
        raise ValueError("Config must bind exactly the five new implementation sources")
    cache=spec.get("observation_cache_source_refs")
    if not isinstance(cache,dict) or not cache:
        raise ValueError("Reviewed observed cache source refs required")
    for ref in cache.values():
        if (not isinstance(ref,dict) or set(ref)!={"path","sha256"}
                or ref.get("path") not in ALLOWED_SOURCE_PATHS
                or not ref["path"].startswith("src/arggnn/")
                or not ref["path"].endswith(".py")
                or sources[ref["path"]]!=ref["sha256"]):
            raise ValueError("Cache source refs cannot widen the reviewed implementation closure")
    return tuple({"path":path,"sha256":sources[path]} for path in sorted(ALLOWED_SOURCE_PATHS))


METADATA_REF_PATHS={
    "config":CONFIG,
    "registration":"research/reviews/expanded_full_panel_v1_root/registration.json",
    "generation_config":"configs/expanded_full_panel_v1.json",
}
INPUT_REF_NAMES=frozenset(METADATA_REF_PATHS)|frozenset({
    "runtime","storage","observation_cache","target_bindings"})


def validated_metadata_ref_v1(record,*,expected_path=None):
    """Reject archive/truth/path substitutions before metadata bytes are read."""
    if not isinstance(record,dict) or set(record)!={"path","sha256"}:
        raise ValueError("Exact metadata path/SHA record required")
    path,digest=record["path"],record["sha256"]
    if (not isinstance(path,str) or Path(path).is_absolute() or Path(path).as_posix()!=path
            or ".." in Path(path).parts or not path.endswith(".json")
            or any(part in {"truth","oracle"} for part in Path(path).parts)
            or not isinstance(digest,str) or len(digest)!=64
            or any(c not in "0123456789abcdef" for c in digest)
            or expected_path is not None and path!=expected_path):
        raise ValueError("Canonical role-specific nontruth JSON metadata required")
    return record


def validated_input_refs_v1(spec):
    refs=spec.get("refs")
    if not isinstance(refs,dict) or set(refs)!=INPUT_REF_NAMES:
        raise ValueError("Exact H038 input metadata roles required")
    for name,ref in refs.items():
        validated_metadata_ref_v1(ref,expected_path=METADATA_REF_PATHS.get(name))
    return refs


def resources_v1(spec,storage,lease):
    original.lightweight_resources_v1(spec,lease)
    return budget_check_v9(storage,boundary=True)


def validate_protocol_v1(cfg):
    if (cfg.get("hypothesis_id")!="H038" or cfg.get("execution_guard_ready") is not True
            or cfg.get("arms")!=list(ARMS) or cfg.get("seed")!=101 or cfg.get("dtype")!="float32"
            or cfg.get("passes")!=30 or cfg.get("updates_per_pass")!=32 or cfg.get("total_updates")!=960
            or cfg.get("evaluation_passes")!=[5,10,15,20,25,30]
            or cfg.get("selection")!="earliest_maximum_equal_parent_Present182_exact_span_accuracy_then_lower_NLL_then_earlier_pass"
            or cfg.get("early_stop_nonimproving_accuracy_checks")!=3
            or cfg.get("model")!={"hidden_dim":32,"expected_haplotypes":952,"site_chunk_size":256,"checkpoint_sites":True}
            or cfg.get("parameter_count")!=29764 or cfg.get("learning_rate")!=.001
            or cfg.get("weight_decay")!=.0001 or cfg.get("gradient_clip_norm")!=1.
            or type(cfg.get("output_cap_bytes")) is not int or cfg["output_cap_bytes"]<=0
            or type(cfg.get("workspace_reserve_bytes")) is not int or cfg["workspace_reserve_bytes"]<=0):
        raise ValueError("Fresh admitted H038 thirty-pass/960-complete-parent-update protocol required")


def inputs_v1(record):
    from .native_argformer_exec import validate_lease,apply_and_validate_title
    lease=validate_lease();apply_and_validate_title()
    validated_metadata_ref_v1(record)
    reader=fit.ConsumerReaderV1();spec=reader.decode_json(record)
    if (spec.get("schema")!="mutation_routed_full_panel_pyg_input_v1"
            or spec.get("root_authorizer")!="root_agent" or spec.get("fit_authorized") is not True
            or spec.get("pyg_authorized") is not True or spec.get("status")!="active"
            or spec.get("arm") not in ARMS or spec.get("seed")!=101 or spec.get("device")!="cuda"
            or not fit.canonical_job_v1(spec.get("job_id"))):
        raise ValueError("Active root H038 single-arm PyG capability required")
    fit._check_active_v1(spec);fit.require_logical_job_v1(lease,spec)
    refs=validated_input_refs_v1(spec)
    cfg=reader.decode_json(refs["config"]);validate_protocol_v1(cfg)
    validated_metadata_ref_v1(cfg["base_fit_config"],expected_path="configs/census_tsinfer_fit_v1.json")
    source_records=validated_source_records_v1(spec,cfg)
    for ref in source_records:
        if ref["sha256"]!=fit._hash(ref["path"]):raise ValueError("Reviewed implementation source changed")
    for ref in source_records:reader.read(ref)
    import torch_geometric
    if (str(torch.__version__)!=cfg["runtime"]["torch"] or torch.version.cuda!=cfg["runtime"]["torch_cuda"]
            or torch_geometric.__version__!=cfg["runtime"]["torch_geometric"]):
        raise ValueError("Pinned Torch/CUDA/PyG runtime required")
    generation=reader.decode_json(refs["generation_config"]);registry=generation["parent_registry"]
    expected_cache_sources={ref["path"]:ref["sha256"] for ref in spec["observation_cache_source_refs"].values()}
    if any(spec["source_sha256"].get(path)!=digest for path,digest in expected_cache_sources.items()):
        raise ValueError("Observed-cache and fit source closure pins differ")
    roles=validate_registry_v1(registry)
    selected=set(roles["train"]+roles["val"])
    for p,ref in spec["observation_entry_refs"].items():
        validated_metadata_ref_v1(ref,expected_path=generation["output_root"]+"/"+p+"/full_panel_observed_entry.json")
    if set(spec["observation_entry_refs"])!=selected:
        raise ValueError("Fit may consume only32TRAIN and8VAL observed entries")
    bindings=reader.decode_json(refs["target_bindings"])["parents"]
    if set(bindings)!=selected:raise ValueError("Fit receives no outer target bindings")
    for p,binding in bindings.items():
        expected_path=generation["output_root"]+"/"+p+"/truth/full_truth_intervals.npz"
        if binding["truth_npz"]["path"]!=expected_path:
            raise ValueError("Target record must name this registered parent's separate truth")
        validated_metadata_ref_v1(binding["generation_receipt"])
        generation_record=reader.decode_json(binding["generation_receipt"])
        if (generation_record["parent_id"]!=p
                or generation_record["files"]["truth/full_truth_intervals.npz"]["sha256"]!=binding["truth_npz"]["sha256"]):
            raise ValueError("Target hash must bind original same-parent generation receipt")
        split=next(r["split"] for r in registry if r["parent_id"]==p)
        if binding["split"]!=split:raise ValueError("Root target split binding differs")
    registration=reader.decode_json(refs["registration"])
    if (registration["hypothesis_id"]!="H038" or registration["maximum_passes"]!=30
            or registration["maximum_full_parent_updates"]!=960 or registration["arms"]!=list(ARMS)
            or registration["evaluation_passes"]!=cfg["evaluation_passes"]
            or registration["checkpoint_selection"]!=cfg["selection"]
            or registration["early_stopping"]!="3_scheduled_validation_checks_without_primary_accuracy_improvement"
            or registration["config_ref"]!=refs["generation_config"]
            or registration["parent_registry"]!=registry or cfg["registration_ref"]!=refs["registration"]
            or cfg["generation_config_ref"]!=refs["generation_config"]):
        raise ValueError("Prospective root H038 generation/budget/ablation registration required")
    for name in ("max_cpu_rss_bytes","max_cuda_reserved_bytes"):
        if type(spec.get(name)) is not int or spec[name]<=0:raise ValueError("Positive root resource bounds required")
    reader.read(refs["runtime"])
    storage=reader.decode_json(refs["storage"])
    if cfg["helper_refs"]["actual_full_storage_launcher"] not in storage["reservation"]["consumer_bindings"]:
        raise ValueError("Real storage launcher must be reserved")
    validate_storage_v9(spec,cfg,storage,source_ref={"path":SOURCE,"sha256":spec["source_sha256"][SOURCE]},
        config_ref=refs["config"],runtime_ref=refs["runtime"])
    original.science_v1()
    return spec,cfg,registry,roles,bindings,reader,storage


class RawSealOutputV1(PilotOutputV9):
    """Raw immutable NPZ seals, exactly guarded writes and one content hash."""
    def __init__(self,*args):
        super().__init__(*args);self.sealed_payloads={}
    def raw_npz(self,name,value):
        if Path(name).name!=name or name in ("",".",".."):raise ValueError("Flat seal name required")
        expected={"C4_probabilities","positions_bp","bin_edges_bp","retained_fixed100bp_indexes","query_sample_indexes"}
        if set(value)!=expected:raise ValueError("Probability-only original-support raw seal schema required")
        arrays={key:np.array(array,copy=True,order="C") for key,array in value.items()}
        for array in arrays.values():array.setflags(write=False)
        buffer=io.BytesIO();np.savez(buffer,**arrays);payload=buffer.getvalue()
        planned=len(payload)+8192
        if self.used+planned>self.limit:raise ValueError("Raw seal exceeds admitted own output")
        budget_check_v9(self.storage,boundary=True,required_write_bytes=planned)
        free=os.statvfs(self.path)
        if planned>free.f_bavail*free.f_frsize:raise ValueError("Filesystem free bytes insufficient")
        path=self.path/name
        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,"wb") as stream:
            stream.write(payload);stream.flush();os.fsync(stream.fileno());os.fchmod(stream.fileno(),0o444)
            identity=os.fstat(stream.fileno())
        named=path.lstat()
        if (identity.st_dev,identity.st_ino)!=(named.st_dev,named.st_ino):raise ValueError("Seal identity changed")
        record={"path":str(path.relative_to(ROOT)),"sha256":hashlib.sha256(payload).hexdigest()}
        self.used+=planned;self.owned[record["path"]]=(dict(record),planned,identity.st_dev,identity.st_ino)
        self.sealed_payloads[record["path"]]=(dict(record),payload,arrays)
        return record
    def discard(self,record):
        # Release immutable in-memory seal bytes when the corresponding owned file is pruned.
        super().discard(record);self.sealed_payloads.pop(record["path"],None)


class CurrentSealReaderV1:
    """Already-hashed immutable sealed bytes during same worker scoring phase."""
    def __init__(self,base,output):self.base=base;self.output=output
    def read(self,record):
        known=self.output.sealed_payloads.get(record["path"])
        if known is not None:
            if known[0]!=record:raise ValueError("Current seal record differs from immutable written bytes")
            return known[1]
        return self.base.read(record)
    def current_probability_seal_v1(self,record,parent):
        known=self.output.sealed_payloads.get(record["path"])
        if known is None:return None
        if known[0]!=record:raise ValueError("Current probability seal identity differs")
        arrays=known[2]
        checks={"positions_bp":parent["positions"],"bin_edges_bp":parent["edges"],
            "retained_fixed100bp_indexes":parent["retained_indexes"],"query_sample_indexes":parent["query_slots"]}
        if any(not np.array_equal(arrays[k],value) for k,value in checks.items()):
            raise ValueError("Immutable current seal original support differs")
        probability=arrays["C4_probabilities"]
        if (probability.dtype!=np.float32 or probability.shape!=(len(parent["positions"]),686,4)
                or not np.isfinite(probability).all() or np.any(probability<0)
                or not np.allclose(probability.sum(-1),1,rtol=1e-5,atol=1e-6)):
            raise ValueError("Raw complete immutable FP32 probabilities required")
        return probability


def tensor_bytes_v1(bank):
    return sum(t.numel()*t.element_size() for f in fields(bank)
        for t in (getattr(bank,f.name) if isinstance(getattr(bank,f.name),tuple) else (getattr(bank,f.name),)))


class FullParentResidencyV1:
    """Bounded complete-parent GPU banks; pinned CPU inputs are never decoded per step."""
    def __init__(self,banks,targets,device,cap):
        self.banks,self.targets,self.device,self.cap=banks,targets,device,cap
        self.resident=OrderedDict();self.bytes=0;self.stream=torch.cuda.Stream(device=device)
        self.transfers=0;self.transfer_bytes=0
    def needed(self,p):
        return tensor_bytes_v1(self.banks[p])+(0 if p not in self.targets else self.targets[p].numel()*4)
    def prefetch(self,p,protected=()):
        if p in self.resident:return True
        needed=self.needed(p)
        if needed>self.cap:raise ValueError("One complete parent exceeds observed residency admission")
        for old in list(self.resident):
            if self.bytes+needed<=self.cap:break
            if old not in protected:
                row=self.resident.pop(old);self.bytes-=row[3];del row
        if self.bytes+needed>self.cap:return False
        with torch.cuda.stream(self.stream):
            values={}
            for f in fields(self.banks[p]):
                value=getattr(self.banks[p],f.name)
                values[f.name]=tuple(t.to(self.device,non_blocking=True) for t in value) if isinstance(value,tuple) else value.to(self.device,non_blocking=True)
            bank=replace(self.banks[p],**values)
            target=None if p not in self.targets else self.targets[p].to(self.device,non_blocking=True)
            ready=torch.cuda.Event();ready.record(self.stream)
        self.resident[p]=(bank,target,ready,needed);self.bytes+=needed
        self.transfers+=1;self.transfer_bytes+=needed
        return True
    def get(self,p):
        if p not in self.resident:self.prefetch(p)
        bank,target,event,_=self.resident[p];self.resident.move_to_end(p)
        current=torch.cuda.current_stream(self.device);current.wait_event(event)
        for f in fields(bank):
            value=getattr(bank,f.name)
            for tensor in (value if isinstance(value,tuple) else (value,)):tensor.record_stream(current)
        if target is not None:target.record_stream(current)
        return bank,target


def native_hold_train_mass_v1(parent,bank,target,helpers):
    """Sum original scored-bin class spans by readout hold without a dense grid mass."""
    if bank.readout_hold.device.type!="cpu" or bank.query_column.numel()!=686:
        raise ValueError("Complete CPU observed TRAIN readout map required")
    readout=bank.readout_hold.numpy()
    holds=int(bank.spans.numel())
    if (readout.shape!=(len(parent["positions"]),) or readout.dtype.kind not in "iu"
            or not holds or np.any((readout<0)|(readout>=holds))):
        raise ValueError("Exact original scored-bin to native-hold membership required")
    bins,rows,spans=helpers.intersections(parent["edges"],target.intervals)
    mass=np.zeros((holds,686,4),dtype=np.float64)
    np.add.at(mass,(readout[bins,None],np.arange(686)[None,:],target.c4[rows]),spans[:,None])
    expected=np.bincount(readout,weights=np.diff(parent["edges"]),minlength=holds)
    if not np.allclose(mass.sum(-1),expected[:,None],atol=1e-8,rtol=0):
        raise ValueError("Aggregated TRAIN mass must preserve every original scored-bin physical span")
    return np.asarray(mass,dtype=np.float32,order="C")


def _train_parent_v1(model,optimizer,bank,target,cfg,arm):
    from .biological_gst_control_v1 import exact_span_loss
    model.train();optimizer.zero_grad(set_to_none=True)
    loss=exact_span_loss(model.forward_train_holds(bank,control=arm),target)
    if not bool(torch.isfinite(loss)):raise ValueError("Nonfinite complete-parent TRAIN loss")
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["gradient_clip_norm"],error_if_nonfinite=True)
    optimizer.step()
    if not bool(torch.isfinite(torch.cat([p.detach().reshape(-1) for p in model.parameters()])).all()):raise ValueError("Nonfinite trained parameter")
    return float(loss.detach())


def _nll(means):
    group=means[PRIMARY]
    return float("inf") if group["NLL_is_infinite"] else group["nll"]


def selection_key_v1(means,pass_index):
    return (-means[PRIMARY]["accuracy"],_nll(means),pass_index)


def run_v1(record):
    spec,cfg,registry,roles,bindings,reader,storage=inputs_v1(record)
    from .biological_candidate import configure_fit
    arm=spec["arm"];output=RawSealOutputV1(spec["output_directory"],storage,reader)
    budget_check_v9(storage,boundary=True,required_write_bytes=32000000)
    base=reader.decode_json(cfg["base_fit_config"])
    device,runtime,lease=configure_fit("cuda",base,output.path)
    output.used=sum(max(p.stat().st_size,p.stat().st_blocks*512)+8192 for p in output.path.rglob("*") if p.is_file())
    if output.used>32000000:raise ValueError("Actual CUDA qualification exceeds admitted32MB")
    helpers=fit.metric_helpers_v1();started=time.monotonic();loader=None
    receipt=dict(schema="mutation_routed_full_panel_pyg_evidence_v1",hypothesis_id="H038",
        status="running",arm=arm,seed=101,input_ref=record,config_ref=spec["refs"]["config"],
        registration_ref=spec["refs"]["registration"],runtime=runtime,source_sha256=spec["source_sha256"],
        TRAIN_parents=32,VAL_parents=8,outer_parents_accessed=0,observed_haplotypes=952,
        query_haplotypes=686,primary_present_queries=182,novelty_claim=False,
        scientific_control="first3_branch_concordance_zero_only;count_empty_geometry_preserved")
    output.save("started.json",receipt)
    steps=0;durable_steps=0;last_checkpoint=None;pending_rows=[]
    try:
        prepare_start=time.monotonic()
        ids=roles["train"]+roles["val"]
        index=read_full_cache_index_v1(spec["refs"]["observation_cache"],reader,registry=registry,
            entry_refs=spec["observation_entry_refs"],source_refs=spec["observation_cache_source_refs"])
        dataset=FullPanelParentDatasetV1(spec["refs"]["observation_cache"],registry=registry,
            entry_refs=spec["observation_entry_refs"],source_refs=spec["observation_cache_source_refs"],
            parent_ids=ids,worker_tmp_directory=output.path/"data_workers_tmp")
        loader=make_parent_dataloader_v1(dataset,pin_memory=True)
        banks,parents,worker_rows={},{},[]
        for graph in loader:
            p=graph.parent_id
            if p in banks:raise ValueError("Duplicate complete parent")
            if graph.worker_cuda_visible_devices!="" or graph.worker_cpu_threads!=1:
                raise ValueError("Strict CPU-only spawn preparation workers required")
            banks[p]=prepare_control_bank_v1(graph.to_model_inputs(),arm);parents[p]=graph.scoring
            worker_rows.append(dict(parent_id=p,worker_pid=graph.worker_pid))
            original.lightweight_resources_v1(spec,lease)
        if set(banks)!=set(ids) or len({r["worker_pid"] for r in worker_rows})!=2:
            raise ValueError("Both CPU workers must finish exactly40development parents")
        drain=shutdown_parent_dataloader_v1(loader)
        if drain["drained"] is not True:raise ValueError("Owned data workers did not drain")
        observation_seal=output.save("all_development_observations_ready.json",dict(
            observation_cache_ref=spec["refs"]["observation_cache"],parent_ids=ids,
            all40_observed_banks_complete=True,targets_opened=False,outer_parents_accessed=0,
            workers=worker_rows,worker_drain=drain,sham_bin_edges=index["sham_bin_edges"],
            control=arm,only_first3_branch_support_fields_zeroed=arm=="zero_concordance"))
        targets={}
        for p in roles["train"]:
            target=FullPanelExternalTargetV1(parents[p],bindings[p],"TRAIN_loss",reader)
            mass=native_hold_train_mass_v1(parents[p],banks[p],target,helpers)
            targets[p]=torch.from_numpy(mass).pin_memory()
            del mass,target
            original.lightweight_resources_v1(spec,lease)
        data_prepare_seconds=time.monotonic()-prepare_start
        random.seed(101);np.random.seed(101);torch.manual_seed(101)
        model=MutationRoutedFullPanelPyGGNNV1(**cfg["model"])
        initial_hash=original.state_fingerprint_v1(model)
        if sum(p.numel() for p in model.parameters())!=29764:raise ValueError("Unchanged learned capacity required")
        model.to(device=device,dtype=torch.float32)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001,betas=(.9,.999),eps=1e-8)
        free,_=torch.cuda.mem_get_info(device);reserved=torch.cuda.memory_reserved(device)
        residency_cap=min(int(free),spec["max_cuda_reserved_bytes"]-int(reserved))-cfg["workspace_reserve_bytes"]
        if residency_cap<=0:raise ValueError("No admitted complete-parent residency headroom")
        residency=FullParentResidencyV1(banks,targets,device,residency_cap)
        all_bytes=sum(residency.needed(p) for p in ids)
        all_resident=all_bytes<=residency_cap
        if all_resident:
            for p in ids:residency.prefetch(p)
        current=[];retained_best=[];best_key=(float("inf"),float("inf"),float("inf"));selected=None;history=[]
        highest_accuracy=-float("inf");stale_accuracy_checks=0;stop_reason="maximum30passes"
        validation=FullPanelPrimaryValidationV1(parents,bindings,CurrentSealReaderV1(reader,output),helpers,cfg)
        train_seconds=0.;validation_seconds=0.;phase_samples=[]
        output.save("initialization_and_admission.json",dict(initial_state_sha256=initial_hash,
            data_prepare_seconds=data_prepare_seconds,observation_features_seal=observation_seal,
            all_observed_bank_bytes=sum(tensor_bytes_v1(b) for b in banks.values()),
            all_bank_plus_TRAIN_mass_bytes=all_bytes,resident_bank_cap_bytes=residency_cap,
            TRAIN_mass_layout="native_hold_by686query_by4class_exact_scored_bin_sum",
            TRAIN_mass_bytes=sum(t.numel()*t.element_size() for t in targets.values()),
            TRAIN_logits_on_native_holds=True,VAL_predictions_on_original_grid=True,
            workspace_reserve_bytes=cfg["workspace_reserve_bytes"],all40_banks_resident=all_resident,
            strict_complete_parent_prefetch=True,first_actual_update_counts_within960=True))
        for pass_index in range(1,31):
            pass_start=time.monotonic();order=list(roles["train"])
            random.Random(101+pass_index).shuffle(order);losses=[]
            for at,p in enumerate(order):
                original.lightweight_resources_v1(spec,lease)
                bank,target=residency.get(p)
                if at+1<len(order):residency.prefetch(order[at+1],protected=(p,))
                update_start=time.monotonic()
                loss=_train_parent_v1(model,optimizer,bank,target,cfg,arm)
                steps+=1;losses.append(loss)
                if steps==1:
                    phase_samples.append(dict(stage="first_actual_complete_parent_update",parent_id=p,
                        elapsed_seconds=time.monotonic()-update_start,counts_in_frozen_training_budget=True,
                        max_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                        max_cuda_reserved_bytes=int(torch.cuda.max_memory_reserved(device))))
                original.lightweight_resources_v1(spec,lease)
            elapsed=time.monotonic()-pass_start;train_seconds+=elapsed
            pending_rows.append(dict(pass_index=pass_index,optimizer_steps=steps,distinct_TRAIN_parents=32,
                complete_parent_updates_this_pass=32,TRAIN_equal_parent_query686_NLL=float(np.mean(losses)),
                elapsed_training_seconds=elapsed))
            print(json.dumps(dict(stage="H038_TRAIN_pass",arm=arm,**pending_rows[-1])),flush=True)
            if pass_index not in cfg["evaluation_passes"]:continue
            evaluation_start=time.monotonic();phases={}
            bookkeeping_start=time.monotonic();resources_v1(spec,storage,lease)
            best_paths={r["path"] for r in retained_best}
            for product in list(current):
                if product["path"].endswith(".npz") and product["path"] not in best_paths:
                    output.discard(product);current.remove(product)
            output.save(f"TRAIN_passes_{pass_index-4:02d}through{pass_index:02d}.json",pending_rows)
            pending_rows=[];phases["pre_prediction_bookkeeping_seconds"]=time.monotonic()-bookkeeping_start
            forward_seconds=copy_seconds=seal_seconds=0.;seals=[]
            model.eval()
            for p in roles["val"]:
                bank,_=residency.get(p);torch.cuda.synchronize(device);phase=time.monotonic()
                with torch.no_grad():logits=model(bank,control=arm);probability=logits.softmax(-1)
                torch.cuda.synchronize(device);forward_seconds+=time.monotonic()-phase;phase=time.monotonic()
                raw=probability.cpu().numpy();copy_seconds+=time.monotonic()-phase;phase=time.monotonic()
                if raw.shape!=(len(parents[p]["positions"]),686,4) or not np.isfinite(raw).all():
                    raise ValueError("Complete finite raw FP32 probabilities required")
                seal=output.raw_npz(f"{arm}.pass_{pass_index:02d}.{p}.npz",dict(
                    C4_probabilities=raw,positions_bp=parents[p]["positions"],bin_edges_bp=parents[p]["edges"],
                    retained_fixed100bp_indexes=parents[p]["retained_indexes"],query_sample_indexes=parents[p]["query_slots"]))
                seals.append((p,seal));seal_seconds+=time.monotonic()-phase
                del logits,probability,raw
            phases.update(VAL_forward_seconds=forward_seconds,VAL_CPU_copy_seconds=copy_seconds,
                          raw_seal_seconds=seal_seconds)
            output.save(f"pass_{pass_index:02d}.all8_VAL_sealed.json",dict(
                pass_index=pass_index,all8_predictions_sealed_before_truth=True,seals=dict(seals)))
            phase=time.monotonic();cells,means=validation.score(seals)
            phases["primary_score_seconds"]=time.monotonic()-phase
            key=selection_key_v1(means,pass_index)
            if means[PRIMARY]["accuracy"]>highest_accuracy:
                highest_accuracy=means[PRIMARY]["accuracy"];stale_accuracy_checks=0
            else:stale_accuracy_checks+=1
            phase=time.monotonic()
            previous={r["path"]:r for r in current+retained_best}
            checkpoint=output.save(f"{arm}.pass_{pass_index:02d}.pt",dict(
                model_state={k:v.detach().cpu() for k,v in model.state_dict().items()},
                optimizer_state=optimizer.state_dict(),pass_index=pass_index,optimizer_steps=steps,arm=arm,
                seed=101,input_ref=record,initial_state_sha256=initial_hash,teacher_used=False),"torch")
            phases["checkpoint_seconds"]=time.monotonic()-phase
            durable_steps=steps;last_checkpoint=checkpoint;current=[r for _,r in seals]+[checkpoint]
            row=dict(pass_index=pass_index,optimizer_steps=steps,cells=cells,equal_parent_metrics=means,
                     checkpoint=checkpoint,phase_seconds=phases,primary_selection_only=True)
            history.append(row)
            if key<best_key:best_key=key;selected=row;retained_best=list(current)
            row["selected_pass"]=selected["pass_index"]
            row["stale_accuracy_checks"]=stale_accuracy_checks
            row["selection_rule"]=cfg["selection"]
            phase=time.monotonic()
            keep={r["path"] for r in current+retained_best}
            for path,product in previous.items():
                if path not in keep:output.discard(product)
            output.save(f"pass_{pass_index:02d}.primary_metrics.json",row)
            phases["post_score_bookkeeping_seconds"]=time.monotonic()-phase
            validation_seconds+=time.monotonic()-evaluation_start
            if pass_index==5:phase_samples.append(dict(stage="first_selection_phase",**phases))
            resources_v1(spec,storage,lease)
            print(json.dumps(dict(stage="H038_VAL_selection",arm=arm,pass_index=pass_index,
                optimizer_steps=steps,selected_pass=selected["pass_index"],primary=means[PRIMARY],phase_seconds=phases)),flush=True)
            if pass_index<30 and stale_accuracy_checks>=3:
                stop_reason="three_scheduled_checks_without_primary_accuracy_improvement"
                break
        if (steps!=32*pass_index or [r["pass_index"] for r in history]
                !=[p for p in cfg["evaluation_passes"] if p<=pass_index]
                or (pass_index<30 and stale_accuracy_checks<3)):
            raise ValueError("Registered complete-parent budget/evaluation/early-stop history incomplete")
        phase=time.monotonic()
        final_cells,final_means=validation.full_final([(c["parent_id"],c["prediction_seal"]) for c in selected["cells"]])
        final_full_seconds=time.monotonic()-phase
        full_record=output.save("final_selected_full_diagnostics.json",dict(
            selected_pass=selected["pass_index"],selection_revised=False,cells=final_cells,equal_parent_metrics=final_means))
        final_drain=shutdown_parent_dataloader_v1(loader)
        if final_drain["drained"] is not True:raise ValueError("Worker PIDs must drain before completion")
        receipt.update(status="completed_H038_seed101_accuracy_selected_full_parent_run",optimizer_steps=steps,
            maximum_optimizer_steps=960,completed_passes=pass_index,stop_reason=stop_reason,
            early_stop_nonimproving_accuracy_checks=3,selection_rule=cfg["selection"],
            equal_optimizer_exposure_across_arms_claimed=False,
            initial_state_sha256=initial_hash,selected=selected,history=history,last_checkpoint=last_checkpoint,
            retained_records={r["path"]:r for r in current+retained_best},final_full_diagnostics=full_record,
            data_prepare_seconds=data_prepare_seconds,training_seconds=train_seconds,
            validation_seconds=validation_seconds,final_full_diagnostic_seconds=final_full_seconds,
            elapsed_seconds=time.monotonic()-started,phase_samples=phase_samples,
            max_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
            max_cuda_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
            max_cpu_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
            input_transfers=residency.transfers,input_transfer_bytes=residency.transfer_bytes,
            all40_banks_resident=all_resident,worker_drain=final_drain,primary_scoring=validation.receipt(),
            outer_truth_read=False,outer_predictions_executed=False)
        reader.read(record);resources_v1(spec,storage,lease)
    except Exception as error:
        drain_error=None
        if loader is not None:
            try:receipt["worker_drain"]=shutdown_parent_dataloader_v1(loader)
            except Exception as worker_error:drain_error=str(worker_error)
        receipt.update(status="failed_partial_preserved",error_type=type(error).__name__,error=str(error),
            logical_steps_reached=steps,last_checkpoint=last_checkpoint,updates_not_checkpointed=steps-durable_steps,
            pending_TRAIN_rows=pending_rows,worker_drain_error=drain_error,automatic_replay=False,
            elapsed_seconds=time.monotonic()-started)
        output.save("failed.json",receipt);raise
    output.save("completed.json",receipt)
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--execute",action="store_true")
    parser.add_argument("--input");parser.add_argument("--expected-input-sha256");args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="source_only_H038_not_executed",hypothesis_id="H038",
            source={"path":SOURCE,"sha256":fit._hash(SOURCE)},config={"path":CONFIG,"sha256":fit._hash(CONFIG)},
            arms=list(ARMS),TRAIN=32,VAL=8,outer=8,passes=30,total_updates_per_arm=960),indent=2));return 0
    if not args.input or not args.expected_input_sha256:raise ValueError("Pinned root execution capability required")
    run_v1({"path":args.input,"sha256":args.expected_input_sha256});return 0


if __name__=="__main__":raise SystemExit(main())
