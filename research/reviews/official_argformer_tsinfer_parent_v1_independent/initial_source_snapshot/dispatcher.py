#!/usr/bin/env python
"""Root-admitted single-parent CPU ARGformer path preparation; source-only default."""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import copy
import datetime
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import random
import stat
import sys
import traceback
from types import SimpleNamespace

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
CONFIG=ROOT/"configs/official_argformer_tsinfer_parent_v1.json"
SCRIPT=ROOT/"scripts/prepare_official_argformer_tsinfer_parent_v1.py"
FIXED_CONFIG={'kind': 'official_argformer_tsinfer_parent_v1', 'status': 'source_only_authoring_review_then_root_single_parent_CPU_admission', 'condition': 'native_ARGformer_current_tsinfer_tsdate_observed_heterochronous_development_transfer_v2', 'observation_condition': 'current_tsinfer_tsdate_observed_heterochronous_task_transfer_v2', 'parent_ids': ['bio_paths_pilot_train_000', 'bio_paths_pilot_train_001', 'bio_paths_pilot_val_000', 'bio_paths_pilot_val_001'], 'default_parent': 'bio_paths_pilot_train_000', 'bridge': {'implementation': {'path': 'src/arggnn/official_argformer_tsinfer_biological_v2.py', 'sha256': '24fbf9c4f06e532fc87fe9ee570ca875e0cad56ff5c719cfd64230930eac760e'}, 'config': {'path': 'configs/official_argformer_tsinfer_biological_v2.json', 'sha256': '502d1de0bc548b936753e84a9e1e1807474768d528bdf68fa90cbb8c921ba0a8'}}, 'combined_observed_manifest': {'path': 'data/biological_paths_execution_pilot_v1/tsinfer_tsdate_observed_v2/combined_observed_manifest.json', 'sha256': 'cbcabd3346802ad87a5b4aa0ea584b1ace8cfcf5ec06a04da4c827f51c3fb44c'}, 'output_root': 'data/official_argformer_tsinfer_parent_v1', 'launch_log_root': 'research/reviews/official_argformer_tsinfer_parent_v1_author', 'native_recipe_config_sha256': '502d1de0bc548b936753e84a9e1e1807474768d528bdf68fa90cbb8c921ba0a8', 'frozen_files_sha256': {'AGENTS.md': 'ade3524279fc686ddb961790252ea4aa5513ec2a12013f082a023b949702d652', 'src/arggnn/official_argformer_tsinfer_biological_v2.py': '24fbf9c4f06e532fc87fe9ee570ca875e0cad56ff5c719cfd64230930eac760e', 'configs/official_argformer_tsinfer_biological_v2.json': '502d1de0bc548b936753e84a9e1e1807474768d528bdf68fa90cbb8c921ba0a8', 'src/arggnn/train_official_argformer_biological.py': 'edc290b9fbce093f4b844f7d07c4963c23bf24257f43c7d502c43a2753170b94', 'src/arggnn/official_argformer_bridge.py': 'd6d81a078cff1bb132bc5040a62db7e728c439472b8093eba41b24a4c1edab80', 'scripts/infer_biological_paths_tsinfer_observed_v2.py': '9904dbbed05d38d2e1c3e03c7bc07f84376df309fff8fe4852ddeec229aac7b8', 'configs/biological_paths_tsinfer_observed_v2.json': '450a502066b74231b81f364ce47f4f927c1c28bf22be20f6d7a9abf945bd0953', 'src/arggnn/gpu_pool_v1.py': 'ae1e0494ab672cfa3017759460eb06a4b4789eaef54928496c3c80314adcc0e1', '.runtime/author_sources/argformer_f7545be7bca0/src/arg_data.py': '2e7df2c955ece89ced3f05c609a4450f997250df5a7cb06ff4a2ca957e49ffdf'}, 'runtime': {'prefix': '.runtime/conda/envs/argformer_native_v1', 'interpreter': '.runtime/conda/envs/argformer_native_v1/bin/python', 'python_version': [3, 11, 9], 'distributions': {'numpy': '1.26.4', 'tskit': '0.6.4', 'torch': '2.4.0'}, 'CPU_only': True, 'threads': 1, 'native_model_not_imported': True}, 'storage': {'ledger_path': '.runtime/gpup/storage_reservations.json', 'scheduler_lock': '.runtime/gpup/scheduler.lock', 'snapshot': {'path': 'experiments/results/biological_shared_pool_storage_admission_v1/snapshot.json', 'sha256': '0826fcedefa40dd261999c4916b85ae11f28628e5cee9046b2e4f6bf868fa10d'}, 'external_reserve_bytes': 130000000000, 'warning_bytes': 450000000000, 'hard_limit_bytes': 500000000000, 'reservation_key_prefix': 'official_argformer_tsinfer_parent_v1', 'reserved_bytes': 2000000000, 'local_write_cap_bytes': 2000000000, 'failure_receipt_max_bytes': 65536, 'existing_observed20GB_slot_not_borrowed': True, 'stage_boundary_cap_not_OS_quota': True}, 'preparation': {'native_tail_AST_unchanged': True, 'full_unique_path_dedup_before_leaf_skip_truncation': True, 'max_seq_len': 128, 'skip_extant_tokens': True, 'path_split_seed': 42, 'train_fraction': 0.9, 'all280_observed_leaves': True, 'pure_reference_count': 96, 'mixed_count': 64, 'query_count': 120, 'only_pure_observed_labels_supervise_contrastive': True, 'query_truth_never_accessed_including_TRAIN': True, 'full_reference_corpus_train_plus_VAL': True, 'validate_all_tokenized_rows_and_readout_mappings': True}, 'fitting': {'MLM_epochs': 5, 'MLM_logical_batch': 512, 'MLM_microbatch': 64, 'contrastive_epochs': 1, 'contrastive_logical_and_microbatch': 4096, 'fit_permitted_by_dispatcher': False, 'query_truth_authorized': False, 'scientific_fallback': False}, 'claims': {'paper_reproduction': False, 'original_AmericanAdmixture_archives_recovered': False, 'current_heterochronous_European_C4_development_task_transfer': True}, 'input_tree_format_compatibility': {'written_tskit_version': '1.0.3', 'admitted_reader_tskit_version': '0.6.4', 'first_actual_array_operation': 'immutable_hash_then_tskit.load_of_exact_selected_observed.trees', 'unsupported': 'preserve_failure_no_conversion_no_upgrade_no_fallback'}}
THREAD_VARS=("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS")
CACHE_VARS=("TMPDIR","TMP","TEMP","XDG_CACHE_HOME","XDG_CONFIG_HOME","XDG_DATA_HOME","TORCH_HOME","HF_HOME","NUMBA_CACHE_DIR","MPLCONFIGDIR","TORCHINDUCTOR_CACHE_DIR","TRITON_CACHE_DIR","CUDA_CACHE_PATH","WANDB_DIR","WANDB_CACHE_DIR","WANDB_CONFIG_DIR")

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()

def exact(a,b):
    return canonical(a)==canonical(b)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def confined(value):
    p=Path(value);p=p if p.is_absolute() else ROOT/p
    if p==ROOT or ROOT not in p.parents or ".." in p.parts or p.resolve()!=p:
        raise ValueError("canonical confined non-symlink project path required")
    if any("truth" in x.lower() or "oracle" in x.lower() for x in p.parts):
        raise ValueError("observed-only paths required; no query truth")
    return p

def ref(path):
    p=confined(path)
    return {"path":str(p.relative_to(ROOT)),"sha256":sha(p)}

def bound(reference,expected=None,json_only=False):
    if not isinstance(reference,dict) or set(reference)!={"path","sha256"}:
        raise ValueError("exact immutable path/SHA256 reference required")
    h=reference["sha256"]
    if not isinstance(h,str) or len(h)!=64 or any(x not in "0123456789abcdef" for x in h):
        raise ValueError("lowercase SHA256 required")
    p=confined(reference["path"])
    if expected is not None:
        expected=confined(expected)
        if p!=expected or reference["path"] not in {str(expected),str(expected.relative_to(ROOT))}:
            raise ValueError("exact expected artifact path required")
    if not p.is_file() or p.is_symlink() or (json_only and p.suffix!=".json"):
        raise ValueError("regular admitted metadata/artifact file required")
    if sha(p)!=h:
        raise ValueError("immutable artifact bytes changed")
    return p

def read_json(reference,expected=None):
    p=bound(reference,expected,json_only=True)
    if p.stat().st_size>32*1024*1024:
        raise ValueError("JSON metadata limit exceeded")
    return json.loads(p.read_text())

def policy():
    c=json.loads(CONFIG.read_text())
    if not exact(c,FIXED_CONFIG):
        raise ValueError("exact one-parent CPU policy required")
    return c

def verify_source(c):
    for name,h in c["frozen_files_sha256"].items():
        if sha(confined(name))!=h:
            raise ValueError("frozen authority changed: "+name)

def load_bridge(c):
    reference=c["bridge"]["implementation"]
    p=bound(reference,"src/arggnn/official_argformer_tsinfer_biological_v2.py")
    spec=importlib.util.spec_from_file_location("_frozen_inactive_argformer_tsinfer_bridge",p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    bc=module.policy()
    bound(c["bridge"]["config"],module.CONFIG,json_only=True)
    module.verify_source(bc)
    return module,bc

def selected_scopes(c,parent):
    if parent not in c["parent_ids"]:
        raise ValueError("one selected development parent required")
    output=confined(Path(c["output_root"])/parent)
    log=confined(Path(c["launch_log_root"])/(parent+".launch.log"))
    return output,log

def admission_bindings(c,parent):
    output,log=selected_scopes(c,parent)
    return {"kind":"official_argformer_tsinfer_parent_v1_root_admission","authorizer":"root_agent",
        "status":"root_authorized_CPU_single_parent_preparation","active":True,
        "condition":c["condition"],"scope_parents":[parent],"implementation":ref(SCRIPT),"config":ref(CONFIG),
        "bridge":c["bridge"],"combined_observed_manifest":c["combined_observed_manifest"],
        "native_recipe_config_sha256":c["native_recipe_config_sha256"],
        "output":str(output.relative_to(ROOT)),"launch_log":str(log.relative_to(ROOT)),
        "CPU_only":True,"threads":1,"preparation_only":True,"native_fit_or_retrieval_authorized":False,
        "query_truth_authorized":False,"scientific_fallback":False,"paper_reproduction_claimed":False}

def validate_admission(c,parent,a,review,runtime):
    expected=admission_bindings(c,parent)
    if any(not exact(a.get(k),v) for k,v in expected.items()):
        raise ValueError("root single-parent exact CPU source/config/input/output scope required")
    review_expected={"kind":"official_argformer_tsinfer_parent_v1_independent_source_review",
        "review_role":"independent_reviewer","verdict":"PASS","implementation":expected["implementation"],
        "config":expected["config"],"bridge":c["bridge"]}
    if any(not exact(review.get(k),v) for k,v in review_expected.items()):
        raise ValueError("exact fresh independent dispatcher source PASS required")
    runtime_expected={"kind":"official_argformer_tsinfer_parent_v1_root_CPU_runtime",
        "authorizer":"root_agent","CPU_only":True,"threads":1,**c["runtime"]}
    if any(not exact(runtime.get(k),v) for k,v in runtime_expected.items()):
        raise ValueError("root-issued exact CPU runtime authority required")
    pyhash=runtime.get("python_executable_sha256")
    if not isinstance(pyhash,str) or len(pyhash)!=64 or any(x not in "0123456789abcdef" for x in pyhash):
        raise ValueError("root-issued exact runtime executable SHA256 required")
    if a.get("actual_input_tree_roster_grid_validation_authorized") is not True:
        raise ValueError("root actual observed tree/roster/grid preparation authority required")
    return expected

def validate_experiment_registration(c,parent,registration):
    expected={"kind":"official_argformer_tsinfer_parent_v1_root_experiment_registration",
        "authorizer":"root_agent","status":"registered_CPU_preparation_and_feasibility_only",
        "condition":c["condition"],"scope_parents":[parent],"implementation":ref(SCRIPT),"config":ref(CONFIG),
        "combined_observed_manifest":c["combined_observed_manifest"],"query_truth_authorized":False}
    if any(not exact(registration.get(k),v) for k,v in expected.items()) or not isinstance(registration.get("hypothesis_id"),str) or not registration["hypothesis_id"]:
        raise ValueError("explicit registered parent CPU feasibility experiment required")

def runtime_preflight(c,runtime):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise ValueError("CPU child must explicitly hide every CUDA device")
    prefix=confined(c["runtime"]["prefix"])
    executable=ROOT/c["runtime"]["interpreter"]
    if (Path(sys.prefix).resolve()!=prefix or list(sys.version_info[:3])!=c["runtime"]["python_version"]
        or not executable.is_file() or executable.resolve().parent!=prefix/"bin"
        or Path(sys.executable).resolve()!=executable.resolve()
        or sha(executable)!=runtime["python_executable_sha256"]):
        raise ValueError("actual fixed admitted CPU interpreter differs")
    import importlib.metadata
    versions={}
    for name,wanted in c["runtime"]["distributions"].items():
        actual=importlib.metadata.version(name)
        if actual.split("+")[0]!=wanted:
            raise ValueError("actual admitted CPU distribution differs: "+name)
        versions[name]=actual
    return {"prefix":str(prefix),"executable_sha256":runtime["python_executable_sha256"],"versions":versions,"CPU_only":True}

def local_bytes(path):
    path=confined(path)
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total=0
    def error(e):raise e
    for parent,dirs,files in os.walk(path,onerror=error,followlinks=False):
        for name in dirs+files:
            p=Path(parent)/name
            if p.is_symlink():raise ValueError("owned output/log symlink forbidden")
        for name in files:
            p=Path(parent)/name
            if not p.is_file():raise ValueError("nonregular owned output file forbidden")
            total+=p.stat().st_size
    return total

def load_pool(c):
    p=confined("src/arggnn/gpu_pool_v1.py")
    if sha(p)!=c["frozen_files_sha256"]["src/arggnn/gpu_pool_v1.py"]:
        raise ValueError("frozen ledger source changed")
    spec=importlib.util.spec_from_file_location("_CPU_preparation_readonly_ledger",p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

class Storage:
    def __init__(self,c,parent,receipt,module):
        self.c=c;self.parent=parent;self.receipt=receipt;self.module=module
        self.output,self.log=selected_scopes(c,parent);self.high_water=0;self.first=True
        p=c["storage"]
        expected={"kind":"official_argformer_tsinfer_parent_v1_root_storage","authorizer":"root_agent",
          "implementation":ref(SCRIPT),"config":ref(CONFIG),"scope_parents":[parent],
          "reservation_key":p["reservation_key_prefix"]+"/"+parent,
          "reserved_bytes":p["reserved_bytes"],"local_write_cap_bytes":p["local_write_cap_bytes"],
          "output":str(self.output),"write_scopes":[str(self.output),str(self.log)],"snapshot":p["snapshot"],
          "external_reserve_bytes":p["external_reserve_bytes"],"warning_bytes":p["warning_bytes"],
          "hard_limit_bytes":p["hard_limit_bytes"],"stage_boundary_cap_not_OS_quota":True}
        if any(not exact(receipt.get(k),v) for k,v in expected.items()):
            raise ValueError("exact fresh parent2GB output/log storage authority required")
        h=receipt.get("ledger_sha256_at_admission")
        if not isinstance(h,str) or len(h)!=64:
            raise ValueError("fresh root ledger SHA256 required")
        self.last=self.check()
    def check(self):
        p=self.c["storage"];ledger_path=confined(p["ledger_path"]);lock=confined(p["scheduler_lock"])
        if not lock.is_file() or not ledger_path.is_file():
            raise ValueError("existing canonical scheduler lock and live ledger required")
        # NFS exclusive flock requires a writable descriptor. Open existing only,
        # without truncate/create or byte writes; ledger/controller state stays readonly.
        fd=os.open(lock,os.O_RDWR|os.O_NOFOLLOW)
        with os.fdopen(fd,"r+") as locked:
            descriptor=os.fstat(locked.fileno());current_lock=lock.stat()
            if (not stat.S_ISREG(descriptor.st_mode) or
                (descriptor.st_dev,descriptor.st_ino)!=(current_lock.st_dev,current_lock.st_ino)):
                raise ValueError("canonical existing scheduler mutex changed")
            fcntl.flock(locked,fcntl.LOCK_EX)
            before=sha(ledger_path)
            if self.first and before!=self.receipt["ledger_sha256_at_admission"]:
                raise ValueError("ledger changed before CPU admission; fresh root receipt required")
            snapshot=self.module.snapshot_record(ROOT,str(confined(p["snapshot"]["path"])),p["snapshot"]["sha256"])
            def available():
                s=os.statvfs(ROOT);return s.f_bavail*s.f_frsize
            ledger=self.module.StorageLedger(ROOT,snapshot,p["external_reserve_bytes"],p["warning_bytes"],available)
            if ledger.path!=ledger_path:
                raise ValueError("existing exact persistent ledger required")
            state=ledger._load()
            expected={"reserved_bytes":p["reserved_bytes"],"output":str(self.output)}
            if not exact(state["jobs"].get(self.receipt["reservation_key"]),expected):
                raise ValueError("exact new parent preparation reservation required; no borrowed slot")
            result=ledger._check(state)
            if sha(ledger_path)!=before:
                raise ValueError("ledger changed during atomic readonly check")
        if result["projected_bytes"]>=p["hard_limit_bytes"]:
            raise ValueError("500GB guard exceeded")
        current=local_bytes(self.output)+local_bytes(self.log)
        self.high_water=max(self.high_water,current)
        if self.high_water>p["local_write_cap_bytes"]:
            raise ValueError("combined parent output/cache/temp/log2GB sampled cap exceeded")
        self.first=False
        self.last={**result,"ledger_sha256":before,"local_bytes":current,"high_water_bytes":self.high_water,
                   "readonly":True,"stage_boundary_cap_not_OS_quota":True}
        return self.last

class Authority:
    def __init__(self,c,parent,runtime_proof,storage):
        self.c=c;self.parent=parent;self.runtime=runtime_proof;self.storage=storage
        self.output,_=selected_scopes(c,parent)
        self.owns_output=False
        self.ready=(runtime_proof.get("CPU_only") is True and bool(runtime_proof.get("executable_sha256")))
    def require(self):
        if not self.ready:
            raise RuntimeError("root runtime/CPU/storage authority required before import or write")
    def json_new(self,path,value):
        self.require();p=confined(path)
        if not self.owns_output:
            raise RuntimeError("fresh admitted stage must own output before write")
        if self.output not in p.parents:
            raise ValueError("write must stay inside this one admitted output")
        self.storage.check()
        payload=json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n"
        with p.open("x") as stream:stream.write(payload)
        self.storage.check()
    def failure_evidence(self,value):
        self.require()
        if not self.owns_output:
            raise RuntimeError("failure evidence requires freshly owned output")
        p=self.output/"dispatcher_receipt.json"
        payload=json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n"
        if len(payload.encode())>self.c["storage"]["failure_receipt_max_bytes"]:
            raise ValueError("failure receipt diagnostic allowance exceeded")
        with p.open("x") as stream:stream.write(payload)

def contain(authority):
    authority.require()
    if authority.output.exists():
        raise ValueError("fresh parent-scoped stage output required")
    authority.storage.check();authority.output.parent.mkdir(parents=True,exist_ok=True)
    authority.output.mkdir()
    authority.owns_output=True
    for name in CACHE_VARS:
        p=authority.output/"write_scopes"/name.lower();p.mkdir(parents=True)
        os.environ[name]=str(p)
    for name in THREAD_VARS:os.environ[name]="1"
    os.environ.update(PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1",WANDB_MODE="disabled",HF_HUB_DISABLE_TELEMETRY="1")
    sys.dont_write_bytecode=True
    authority.storage.check()

def CPU_libraries(authority):
    authority.require()
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="" or any(os.environ.get(n)!="1" for n in THREAD_VARS):
        raise ValueError("one-thread CPU environment required before imports")
    import numpy as np
    import tskit
    import torch
    prefix=confined(authority.c["runtime"]["prefix"])
    for module in [np,tskit,torch]:
        if prefix not in Path(module.__file__).resolve().parents:
            raise ValueError("actual CPU library origin outside admitted prefix")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from arggnn import official_argformer_bridge as actual_bridge
    expected=confined("src/arggnn/official_argformer_bridge.py")
    if Path(actual_bridge.__file__).resolve()!=expected or actual_bridge.ROOT!=ROOT or sha(expected)!=authority.c["frozen_files_sha256"]["src/arggnn/official_argformer_bridge.py"]:
        raise ValueError("actual observed extraction/vocabulary module origin differs")
    return np,tskit

def preparation_definition(c,bridge,bc):
    node=bridge.function_nodes(bridge.FIT,bc,["prepare_paths"])["prepare_paths"]
    start=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="tree_path" for t in n.targets))
    return copy.deepcopy(node.body[start:])

def preparation_callable(c,bridge,bc,authority):
    authority.require()
    body=preparation_definition(c,bridge,bc)
    args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=n) for n in ["entry","parent_id","output","expected_manifest_sha","config"]],
                      vararg=None,kwonlyargs=[],kw_defaults=[],kwarg=None,defaults=[])
    fn=ast.FunctionDef(name="prepare_native_paths",args=args,body=body,decorator_list=[],returns=None,type_comment=None)
    module=ast.Module(body=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0),fn],type_ignores=[])
    ast.fix_missing_locations(module)
    ns={"Path":Path,"ROOT":ROOT,"CONFIG":bridge.CONFIG,"sha":sha,"observed_path":confined,
        "json_new":authority.json_new,"math":math,"random":random,"pickle":pickle}
    exec(compile(module,str(confined(bridge.FIT)),"exec"),ns)
    core=ns["prepare_native_paths"]
    def admitted(entry,parent,output,manifest_sha,config):
        authority.require()
        if parent!=authority.parent or confined(output)!=authority.output/"prepared" or not exact(config,bc):
            raise ValueError("native preparation call must match one admitted parent/scientific policy")
        authority.storage.check()
        return core(entry,parent,confined(output),manifest_sha,config)
    return admitted

class PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        raise ValueError("only trusted primitive corpus serialization allowed")

def primitive_load(path):
    with Path(path).open("rb") as stream:return PrimitiveUnpickler(stream).load()

def budget(bridge,bc,role_counts,query_needed,declared_counts=None,unique_full_paths=None):
    if set(role_counts)!={"train","val"}:
        raise ValueError("both exact native path splits required")
    roles={"pure_source_reference","mixed_reference_only","query"}
    for split,counts in role_counts.items():
        if set(counts)!=roles or any(type(x) is not int or x<0 for x in counts.values()):
            raise ValueError("typed actual per-split role counts required")
    mlm={s:sum(role_counts[s].values()) for s in role_counts}
    refs={s:role_counts[s]["pure_source_reference"] for s in role_counts}
    queries=sum(role_counts[s]["query"] for s in role_counts)
    total=sum(mlm.values());pure=sum(refs.values())
    if unique_full_paths is not None and (type(unique_full_paths) is not int or unique_full_paths!=total):
        raise ValueError("actual unique path total does not match native record")
    if declared_counts is not None:
        if not exact(declared_counts,{"mlm":mlm,"contrastive":refs}):
            raise ValueError("actual per-split pure-reference/MLM counts do not match native record")
    if any(refs[s]>mlm[s] for s in refs) or type(query_needed) is not int or not 0<query_needed<=queries:
        raise ValueError("pure-ref/query/MLM split counts inconsistent")
    if pure+queries>total:
        raise ValueError("full query plus reference paths exceed unique corpus")
    counts={"unique_full_paths":total,"MLM_train":mlm["train"],"MLM_val":mlm["val"],
       "pure_reference_train":refs["train"],"pure_reference_val":refs["val"],"query_full_paths_scored":query_needed}
    feasible=mlm["train"]>=512 and mlm["val"]>0 and refs["train"]>=4096 and refs["val"]>0
    if feasible:
        result=bridge.corpus_budget(bc,counts)
    else:
        result={"actual_counts":counts,"MLM_optimizer_updates":5*(mlm["train"]//512),
                "contrastive_optimizer_updates":refs["train"]//4096,
                "pure_reference_corpus_paths_all_train_plus_val":pure,"exhaustive_query_corpus_pairs":pure*query_needed,
                "MLM_epochs":5,"contrastive_epochs":1,"MLM_logical_batch":512,"MLM_microbatch":64,
                "contrastive_full_logical_and_microbatch":4096,"query_truth_trains_any_model":False}
    result.update(actual_per_split_role_counts=role_counts,full_query_unique_paths=queries,
                  native_MLM512_contrastive4096_feasible=feasible,no_budget_or_corpus_reduction=True)
    return result

def author_tokenizer(c,authority):
    authority.require()
    p=confined(".runtime/author_sources/argformer_f7545be7bca0/src/arg_data.py")
    if sha(p)!=c["frozen_files_sha256"][str(p.relative_to(ROOT))]:
        raise ValueError("genuine author tokenizer source changed")
    tree=ast.parse(p.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="ARGDataset")
    nodes=[]
    for name in ["_tokenize_raw_sequence_from_vocab","__getitem__"]:
        node=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name))
        nodes.append(node)
    module=ast.Module(body=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0),*nodes],type_ignores=[])
    ast.fix_missing_locations(module);ns={}
    exec(compile(module,str(p),"exec"),ns)
    return ns,{n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest() for n in nodes}

def token_rows(paths,vocab,methods):
    obj=SimpleNamespace(sequences=paths,_data_is_pretokenized=False,return_metadata=True,
       skip_extant_tokens=True,max_seq_len=128,pad_sequences=True,pad_token_id=0,cls_token_id=2,sep_token_id=3,
       _node_id_token_map=vocab["node_id_tokens"])
    obj._tokenize_raw_sequence_from_vocab=lambda row:methods["_tokenize_raw_sequence_from_vocab"](obj,row)
    return lambda index:methods["__getitem__"](obj,index)

def validate_tokens(paths,vocab,methods):
    row_at=token_rows(paths,vocab,methods);fingerprints={};stream=hashlib.sha256()
    truncated=0;duplicates=0
    for index,path in enumerate(paths):
        row=row_at(index)
        if row["extant_node_id_raw"]!=path[0] or len(row["input_ids"])!=128 or len(row["attention_mask"])!=128:
            raise ValueError("native tokenizer row/full path mapping differs")
        model_input={"input_ids":row["input_ids"],"attention_mask":row["attention_mask"],"token_type_ids":row["token_type_ids"]}
        h=hashlib.sha256(canonical(model_input)).hexdigest()
        if h in fingerprints:
            first=row_at(fingerprints[h])
            if any(first[k]!=model_input[k] for k in model_input):
                raise ValueError("token fingerprint collision; exact source equality fails")
            duplicates+=1
        else:fingerprints[h]=index
        stream.update(canonical({"path_id":index,**row})+b"\n")
        truncated+=len(path)-1>126
    return {"tokenized_rows":len(paths),"distinct_tokenized_inputs":len(fingerprints),
            "duplicate_tokenized_inputs_retained":duplicates,"full_paths_truncated_by_author128_policy":truncated,
            "tokenized_ordered_stream_sha256":stream.hexdigest(),"full_path_ID_row_mapping_preserved":True}

def qualify_corpus(c,bridge,bc,authority,prepared,np):
    authority.require();directory=authority.output/"prepared"
    if prepared["policy_sha256"]!=sha(bridge.CONFIG) or prepared["true_query_ancestry_read"] is not False:
        raise ValueError("prepared native recipe/truth policy differs")
    for relative,h in prepared["files_sha256"].items():
        p=confined(directory/relative)
        if directory not in p.parents or sha(p)!=h:
            raise ValueError("immutable native preparation artifact differs")
    corpus=primitive_load(directory/"corpus.pkl")
    if set(corpus)!={"paths","occurrences","namespace"}:
        raise ValueError("exact native full corpus schema required")
    paths=corpus["paths"];occurrences=corpus["occurrences"]
    if not isinstance(paths,tuple) or len(paths)!=prepared["unique_full_paths"] or len(set(paths))!=len(paths):
        raise ValueError("full paths must be unique before truncation")
    rows=json.loads((directory/"observed_roster.json").read_text())
    nodes={r["inferred_tree_node_id"]:r for r in rows}
    if len(nodes)!=280 or {p[0] for p in paths}!=set(nodes):
        raise ValueError("full corpus must retain all280 native observed leaves")
    ref_nodes=[r["inferred_tree_node_id"] for r in rows if r["role"]=="pure_source_reference"]
    query_nodes=[r["inferred_tree_node_id"] for r in rows if r["role"]=="query"]
    labels={n:nodes[n]["observed_reference_class"] for n in ref_nodes}
    if len(ref_nodes)!=96 or len(query_nodes)!=120:
        raise ValueError("complete pure96/query120 role corpus required")
    if any(not isinstance(p,tuple) or len(p)<2 or len(set(p))!=len(p) or any(type(n) is not int or n<0 for n in p) for p in paths):
        raise ValueError("genuine acyclic ancestor-bearing native ID full paths required")
    expected_vocab={"special":{"[PAD]":0,"[MASK]":1,"[CLS]":2,"[SEP]":3},
       "node_id_tokens":{f"node_{node}":i+4 for i,node in enumerate(sorted({n for p in paths for n in p}))}}
    all_sequences=primitive_load(directory/"embedding/all/raw_all_sequences_and_vocab.pkl")
    if set(all_sequences)!={"raw_sequences","vocab"} or [tuple(x) for x in all_sequences["raw_sequences"]]!=list(paths) or all_sequences["vocab"]!=expected_vocab:
        raise ValueError("exact source node vocabulary and full-path embedding order required")
    ids=list(range(len(paths)));random.Random(42).shuffle(ids);cut=int(.9*len(ids))
    split_ids={"train":ids[:cut],"val":ids[cut:]};role_counts={}
    for split,selected in split_ids.items():
        role_counts[split]={role:sum(nodes[paths[i][0]]["role"]==role for i in selected)
                           for role in ["pure_source_reference","mixed_reference_only","query"]}
        for task in ["mlm","contrastive"]:
            actual_ids=json.loads((directory/task/split/"path_ids.json").read_text())
            expected=selected if task=="mlm" else [i for i in selected if paths[i][0] in labels]
            if not exact(actual_ids,expected):
                raise ValueError("native seeded full-path split or pure-ref selection differs")
            payload=primitive_load(directory/task/split/("raw_"+split+"_sequences_and_vocab.pkl"))
            if set(payload)!={"raw_sequences","vocab"} or payload["vocab"]!=expected_vocab or payload["raw_sequences"]!=[list(paths[i]) for i in expected]:
                raise ValueError("exact native task corpus row/full-path mapping differs")
            if task=="contrastive" and primitive_load(directory/task/split/"labels.pkl")!=[labels[paths[i][0]] for i in expected]:
                raise ValueError("contrastive labels must be pure observed population labels only")
            if prepared["path_counts"][task][split]!=len(expected):
                raise ValueError("recorded actual per-split count differs")
    with np.load(directory/"coordinate_path_joins.npz",allow_pickle=False) as archive:
        joins={k:archive[k].copy() for k in archive.files}
    required={"grid_indexes","original_positions_bp","local_positions_bp","query_nodes","reference_nodes","reference_classes","query_path_ids","reference_path_ids"}
    if set(joins)!=required or not np.array_equal(joins["query_nodes"],query_nodes) or not np.array_equal(joins["reference_nodes"],ref_nodes) or not np.array_equal(joins["reference_classes"],[labels[n] for n in ref_nodes]):
        raise ValueError("exact native query/reference/label readout mappings required")
    coord=prepared["coordinate_contract"];offset=coord["local_to_original_offset_bp"]
    if (not np.array_equal(joins["grid_indexes"],coord["retained_fixed100bp_indexes"])
        or not np.array_equal(joins["original_positions_bp"],50+100*joins["grid_indexes"])
        or not np.array_equal(joins["local_positions_bp"],joins["original_positions_bp"]-offset)):
        raise ValueError("entire retained original/local coordinate support required")
    kernels=bridge.retrieval_core(bc)
    kernels["validate_support"](paths,occurrences,corpus["namespace"],prepared["namespace"],joins,
                                coord["coverage_original_bp0"][1]-offset)
    pure_paths,pure_labels=kernels["corpus_roles"](paths,joins["query_nodes"],joins["reference_nodes"],joins["reference_classes"])
    if set(paths[int(i)][0] for i in pure_paths)!=set(ref_nodes):
        raise ValueError("all96 pure-reference leaves must be present in full train-plus-VAL corpus")
    query_needed=len(np.unique(joins["query_path_ids"]))
    measured_budget=budget(bridge,bc,role_counts,query_needed,prepared["path_counts"],prepared["unique_full_paths"])
    if len(pure_paths)!=measured_budget["pure_reference_corpus_paths_all_train_plus_val"]:
        raise ValueError("full reference corpus split counts differ")
    methods,method_hashes=author_tokenizer(c,authority)
    tokens=validate_tokens(paths,expected_vocab,methods)
    internal=Counter(n for path in paths for n in path[1:])
    return {"status":"qualified_full_native_preparation" if measured_budget["native_MLM512_contrastive4096_feasible"] else "prepared_native_source_budget_infeasible",
      "budget":measured_budget,"tokenization":tokens,"native_author_tokenizer_ASTs":method_hashes,
      "native_original_node_IDs_used":True,"ancestor_node_IDs_seen_in_multiple_full_paths":sum(v>1 for v in internal.values()),
      "ancestor_ID_sharing_full_path_occurrences":sum(v-1 for v in internal.values() if v>1),
      "source_node_vocabulary_size":len(expected_vocab["node_id_tokens"])+4,
      "all_unique_paths_before_truncate":len(paths),"full_pure_reference_corpus_paths":len(pure_paths),
      "full_readout_occurrence_partitions_validated":True,"query_truth_used":False,"model_fit_or_retrieval_executed":False}

def execute(c,bridge,bc,args):
    if not all([args.expected_source_sha256,args.expected_config_sha256,args.admission,args.expected_admission_sha256]):
        raise ValueError("root exact source/config/admission opt-in required")
    if sha(SCRIPT)!=args.expected_source_sha256 or sha(CONFIG)!=args.expected_config_sha256:
        raise ValueError("reviewed dispatcher source/config changed")
    a=read_json({"path":args.admission,"sha256":args.expected_admission_sha256})
    review=read_json(a["independent_review"])
    runtime=read_json(a["runtime"])
    validate_admission(c,args.parent,a,review,runtime)
    validate_experiment_registration(c,args.parent,read_json(a["experiment_registration"]))
    runtime_proof=runtime_preflight(c,runtime)
    storage=Storage(c,args.parent,read_json(a["storage"]),load_pool(c))
    authority=Authority(c,args.parent,runtime_proof,storage)
    # Metadata source is read before native arrays, but all active authority is fixed.
    metadata=bridge.metadata_report(bc)
    manifest=read_json(c["combined_observed_manifest"],bc["manifest_contract"]["path"])
    row=next(r for r in manifest["replicates"] if r["parent_id"]==args.parent)
    native=row["frozen_helper_result"]
    rows=read_json({k:native["observed_sample_manifest"][k] for k in ["path","sha256"]})
    sources=read_json(native["original_observed_sample_manifest"])
    record=bridge.native_metadata(bc,row,rows,sources)
    entry=bridge.preparation_entry(record)
    result={"kind":"official_argformer_tsinfer_parent_v1_receipt","status":"failed",
      "condition":c["condition"],"parent_id":args.parent,"namespace":record["namespace"],
      "implementation":ref(SCRIPT),"config":ref(CONFIG),"bridge":c["bridge"],
      "combined_observed_manifest":c["combined_observed_manifest"],"admission":ref(confined(args.admission)),
      "CPU_runtime":runtime_proof,"CPU_only":True,"threads":1,"actual_native_preparation_completed":False,
      "query_truth_used":False,"model_fit_or_retrieval_executed":False,"scientific_fallback":False,
      "paper_reproduction_claimed":False,"exit_code":1}
    try:
        contain(authority)
        authority.json_new(authority.output/"admission_metadata.json",{"root_admission":a,"metadata_binding":metadata["parents"][args.parent]})
        np,tskit=CPU_libraries(authority)
        result["original_tree_format_compatibility_attempted"]=True
        phase=bridge.phase_for(bc,args.parent)
        expected_tree=Path(bc["manifest_contract"]["path"]).parent/phase/args.parent/"observed.trees"
        tree_path=bound({k:native["inferred_tree"][k] for k in ["path","sha256"]},expected_tree)
        # First actual admitted array read: test writer1.0.3 -> reader0.6.4 without conversion.
        compatibility_tree=tskit.load(tree_path)
        if (compatibility_tree.num_samples!=280 or compatibility_tree.num_sites!=native["sites_in_scored_support"]
            or set(compatibility_tree.samples().tolist())!={r["inferred_tree_node_id"] for r in rows}):
            raise ValueError("original observed tree format/sample/site compatibility differs")
        compatibility={"status":"original_observed_tree_loaded_without_conversion","reader_tskit_version":tskit.__version__,
            "input_writer_tskit_version":c["input_tree_format_compatibility"]["written_tskit_version"],
            "input_tree":{k:native["inferred_tree"][k] for k in ["path","sha256"]},
            "num_samples":compatibility_tree.num_samples,"num_sites":compatibility_tree.num_sites,
            "num_nodes":compatibility_tree.num_nodes,"native_IDs_preserved":True,"conversion_upgrade_fallback":False}
        authority.json_new(authority.output/"input_tree_compatibility.json",compatibility)
        result["input_tree_compatibility"]=ref(authority.output/"input_tree_compatibility.json")
        del compatibility_tree
        core=preparation_callable(c,bridge,bc,authority)
        prepared=core(entry,args.parent,authority.output/"prepared",c["combined_observed_manifest"]["sha256"],bc)
        result["actual_native_preparation_completed"]=True
        qualification=qualify_corpus(c,bridge,bc,authority,prepared,np)
        authority.json_new(authority.output/"corpus_qualification.json",qualification)
        result["corpus_qualification"]=ref(authority.output/"corpus_qualification.json")
        result["prepared_manifest"]=ref(authority.output/"prepared/prepared_manifest.json")
        result["status"]=qualification["status"]
        result["exit_code"]=0 if qualification["budget"]["native_MLM512_contrastive4096_feasible"] else 2
        for field in ["inferred_tree","completion_provenance","original_observed_sample_manifest"]:
            bound({k:native[field][k] for k in ["path","sha256"]})
        bound({k:native["observed_sample_manifest"][k] for k in ["path","sha256"]})
        bound({k:native["coordinate_contract"]["fixed_grid"][k] for k in ["path","sha256"]})
        bridge.verify_source(bc);verify_source(c)
        result["storage"]=storage.check()
    except BaseException as error:
        result.update(status="failed_partial_preserved",error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),exit_code=1)
    result["created_utc"]=datetime.datetime.now(datetime.timezone.utc).isoformat()
    if not authority.owns_output:
        raise RuntimeError("failure before owned output; preserve root outer launch log: "+result.get("error",result["status"]))
    authority.failure_evidence(result)  # bounded small evidence even after sampled cap failure
    try:storage.check()
    except Exception as error:
        if result["exit_code"]==0:
            result.update(status="failed_storage_after_receipt",error=str(error),exit_code=1)
            p=authority.output/"dispatcher_receipt.json"
            # One bounded terminal evidence amendment, never scientific source overwrite.
            payload=json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n"
            if len(payload.encode())>c["storage"]["failure_receipt_max_bytes"]:raise
            p.write_text(payload)
    return result

def source_report(c,bridge,bc):
    body=preparation_definition(c,bridge,bc)
    return {"kind":"official_argformer_tsinfer_parent_v1_source_report",
      "status":"source_only_executable_CPU_dispatcher_unexecuted","implementation":ref(SCRIPT),"config":ref(CONFIG),
      "bridge":c["bridge"],"combined_observed_manifest":c["combined_observed_manifest"],
      "native_preparation_tail_AST_sha256":hashlib.sha256(ast.dump(ast.Module(body=body,type_ignores=[]),include_attributes=False).encode()).hexdigest(),
      "native20_function_ASTs":bridge.function_proof(bc)[bridge.FIT],
      "original60_author_tree":bridge.verify_author_source(bc),"CPU_runtime_inspected":False,
      "live_ledger_read":False,"actual_arrays_opened":False,"preparation_fit_retrieval_executed":False,
      "query_truth_used":False,"paper_reproduction_claimed":False}

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--parent",default="bio_paths_pilot_train_000")
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--expected-config-sha256")
    parser.add_argument("--admission")
    parser.add_argument("--expected-admission-sha256")
    args=parser.parse_args(argv)
    sys.dont_write_bytecode=True
    c=policy();verify_source(c);bridge,bc=load_bridge(c)
    if args.parent not in c["parent_ids"]:parser.error("one development parent only")
    if not args.execute:
        print(json.dumps(source_report(c,bridge,bc),indent=2));return 0
    result=execute(c,bridge,bc,args)
    print(json.dumps(result,indent=2));return result["exit_code"]

if __name__=="__main__":
    raise SystemExit(main())
