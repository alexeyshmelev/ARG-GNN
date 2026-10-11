"""Inactive same-tsinfer Census consumer. Source report is standard-library only.

Original fit2 scientific functions are copied verbatim. Only certified observation
transport, authority schemas and target/support lineage are adapted.
"""
from __future__ import annotations
import argparse
import ast
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from . import census_tsinfer_observation_v1 as bridge
ROOT=Path(__file__).resolve().parents[2]
SOURCE="src/arggnn/census_tsinfer_fit_v1.py"
CONFIG="configs/census_tsinfer_fit_v1.json"
CONTROLS=("proposed","eta0","permuted_groups","raw_diagonal","local")
PARENTS=tuple(f"bio_paths_pilot_{s}_{i:03d}" for s in ("train","val") for i in range(2))
GATE_REFS={"numerical_evidence","numerical_admission","inspection_evidence","inspection_freeze","inspection_lineage","numerical_terminal"}
REFS=GATE_REFS|{"profile","config","review","runtime","storage","launcher","memory","registration","target_manifest","target_bindings"}
np=torch=CensusTreeGNNV1=fingerprint_sha256_v1=None
_CAPSULE=None

class ConsumerReaderV1(bridge.VerifiedObservationReaderV1):
    def read(self,record,*,runtime_file=False,runtime=False):
        return super().read(record,runtime=runtime_file or runtime)

def require_logical_job_v1(lease,admission):
    pool_job=lease["launcher_metadata"]["job"]
    actual=os.environ.get("ARGGNN_LOGICAL_JOB_NAME") or os.environ.get("ARGGNN_JOB_NAME")
    if actual!=admission["job_id"] or os.environ.get("ARGGNN_JOB_NAME")!=pool_job or not actual:
        raise ValueError("Admitted fit logical job differs from actual held lease worker")

def canonical_job_v1(value):
    return isinstance(value,str) and bool(value) and all(c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in value)
def _utc(value):return bridge._utc(value)
def _hash(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
def load_config_v1():
    cfg=json.loads((ROOT/CONFIG).read_text())
    old=json.loads((ROOT/"configs/census_biological_fit_v2.json").read_text())
    if cfg["kind"]!="census_tsinfer_fit_v1" or any(cfg[k]!=old[k] for k in ("training","model","controls","metrics","seeds","readout_chunk_size","external_target_manifest")):
        raise ValueError("Original fit2 scientific policy changed")
    return cfg

def source_report_v1():
    cfg=load_config_v1()
    for path,digest in cfg["source_sha256"].items():
        if _hash(path)!=digest:raise ValueError("Pinned consumer dependency changed: "+path)
    original=ast.parse((ROOT/"src/arggnn/census_biological_fit_v2.py").read_text())
    current=ast.parse((ROOT/SOURCE).read_text())
    maps=[{n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))} for tree in (original,current)]
    for name in cfg["unchanged_fit_AST"]:
        if maps[0][name]!=maps[1][name]:raise ValueError("Original fit scientific AST changed: "+name)
    def normalized_run(tree,old=False):
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="run_fit_v1")
        if old:
            for child in ast.walk(node):
                if isinstance(child,ast.Constant) and child.value=="census_biological_fit_evidence_v2":child.value="census_tsinfer_fit_evidence_v1"
        return ast.dump(node,include_attributes=False)
    if normalized_run(original,True)!=normalized_run(current):raise ValueError("Original fit2 entire training/selection loop changed")
    bridge.source_report_v1()
    return {"schema":"census_tsinfer_fit_source_report_v1","status":"inactive_source_only",
        "implementation":{"path":SOURCE,"sha256":_hash(SOURCE)},"config":{"path":CONFIG,"sha256":_hash(CONFIG)},
        "unchanged_fit_AST":cfg["unchanged_fit_AST"],"actual_arrays_opened":False,
        "models_created":False,"GPU_used":False,"column7":"original observed encoded parent-branch duration dropped by F47",
        "query_ancestry_input":False}

def require_terminal_numerical_v1(evidence,native,inspection,freeze,terminal,refs,lineage=None):
    """JSON-only real bridge-schema gate; never reads arrays or constructs models."""
    cfg=bridge.load_config_v1()
    expected={f"{p}:{i}" for p in PARENTS for i in range(8)}
    if terminal!={"schema":"census_tsinfer_numerical_root_terminal_binding_v1","status":"complete","returncode":0,
        "evidence_ref":refs["numerical_evidence"],"admission_ref":refs["numerical_admission"],
        "inspection_ref":refs["inspection_evidence"],"freeze_ref":refs["inspection_freeze"],"job_id":native["job_id"]}:
        raise ValueError("Exact terminal exit0 same-tsinfer numerical binding required")
    sources=dict(cfg["source_sha256"],**{bridge.SOURCE:_hash(bridge.SOURCE)})
    for item,phase,status in ((inspection,"inspect","inspection_complete_pending_root_freeze"),(evidence,"numerical","numerical_complete_no_fitting_or_predictive_claim")):
        if (item.get("schema")!="census_tsinfer_observation_evidence_v1" or item.get("phase")!=phase
            or item.get("status")!=status or item.get("fixture_mode") is not False or item.get("expected_cells")!=32
            or item.get("failed_cells")!=[] or len(item.get("cells",[]))!=32
            or any(item.get(k) is not False for k in ("biological_targets_opened","fit_performed","GPU_used","feature_semantic_adaptation"))
            or item.get("old_Relate_arm_unchanged") is not True
            or item.get("source_sha256")!=sources or item.get("observed_artifact_sha256")!=cfg["observed_artifact_sha256"]
            or item.get("metadata_input_sha256")!=cfg["metadata_input_sha256"]):
            raise ValueError("Complete32 unchanged target-free actual tsinfer evidence required")
    if (native.get("schema")!="census_tsinfer_observation_admission_v1" or native.get("status")!="active"
        or native.get("authorizer")!="root_agent" or native.get("phase")!="numerical"
        or native.get("query_truth_authorized") is not False or native.get("fit_authorized") is not False
        or native.get("source_sha256")!=sources or native.get("observed_artifact_sha256")!=cfg["observed_artifact_sha256"]
        or native.get("inspection_manifest")!=refs["inspection_evidence"]
        or native.get("refs",{}).get("config")!={"path":bridge.CONFIG,"sha256":_hash(bridge.CONFIG)}
        or evidence.get("admission_ref")!=refs["numerical_admission"] or evidence.get("job_id")!=native["job_id"]):
        raise ValueError("Genuine same-tsinfer numerical admission/source/inspection binding required")
    if (freeze.get("schema")!="root_census_tsinfer_inspection_freeze_v1" or freeze.get("status")!="complete"
        or freeze.get("authorizer")!="root_agent" or freeze.get("inspection_evidence")!=refs["inspection_evidence"]
        or freeze.get("actual_observation_cells")!=32 or freeze.get("raw_and_producer_objects")!=64
        or freeze.get("source_sha256")!=sources or freeze.get("observed_artifact_sha256")!=cfg["observed_artifact_sha256"]
        or freeze.get("initialization_model_seed_dtype_runtime_unchanged") is not True
        or any(freeze.get(k) is not False for k in ("raw_payload_decoded","models_created","targets_accessed","GPU_used","numerical_or_fit_authorized_by_this_freeze"))):
        raise ValueError("Exact sealed root same-tsinfer inspection freeze required")
    if (not lineage or lineage.get("schema")!="root_census_tsinfer_source_lineage_v1" or lineage.get("status")!="complete"
        or lineage.get("authorizer")!="root_agent" or lineage.get("inspection_ref")!=refs["inspection_evidence"]
        or lineage.get("freeze_ref")!=refs["inspection_freeze"]
        or lineage.get("source_sha256")!=sources or lineage.get("observed_artifact_sha256")!=cfg["observed_artifact_sha256"]
        or lineage.get("metadata_input_sha256")!=cfg["metadata_input_sha256"]
        or lineage.get("source_and_original_count_lineage_rechecked") is not True
        or any(lineage.get(k) is not False for k in ("model_or_label_outcomes_used_for_selection","arrays_decoded","models_created","targets_accessed","GPU_used"))):
        raise ValueError("Separate root full source/observation/metadata lineage recheck required")
    frozen={f"{c['parent_id']}:{c['tile_id']}":c for c in inspection["cells"]}
    if len(frozen)!=32 or set(frozen)!=expected:raise ValueError("Exact unique32 frozen cells required")
    hashes={key:c["certificate_sha256"] for key,c in frozen.items()}
    closure={r["path"]:r["sha256"] for c in frozen.values() for r in (c["raw_ref"],c["producer_ref"])}
    if (len(closure)!=64 or any(not p.startswith(cfg["inspection_output_prefix"]+inspection["job_id"]+"/") for p in closure)
        or hashes!=freeze.get("expected_certificates") or hashes!=native.get("expected_certificates")
        or closure!=freeze.get("derived_artifact_sha256") or closure!=native.get("derived_artifact_sha256")
        or native.get("initial_state_sha256_by_dtype")!=freeze.get("initial_state_sha256_by_dtype")
        or native.get("group_permutation")!=freeze.get("group_permutation")
        or sorted(native.get("group_permutation",[]))!=list(range(184))):
        raise ValueError("Frozen exact32 certificates/64 materializations/init/permutation required")
    wanted=[[d,c] for d in ("float64","float32") for c in CONTROLS]
    seen=set()
    for cell in evidence["cells"]:
        key=f"{cell['parent_id']}:{cell['tile_id']}"
        if key in seen or key not in frozen:raise ValueError("Duplicate/unknown numerical cell")
        seen.add(key);before=frozen[key];q=cell.get("qualification",{})
        if (cell.get("status")!="numerically_qualified" or before.get("status")!="inspected_pending_root_freeze"
            or any(cell.get(k)!=before.get(k) for k in ("split","num_trees","certificate_sha256","inner_certificate_sha256","lineage","structural","event_collision_counts","raw_ref","producer_ref"))
            or q.get("status")!="qualified" or q.get("performed_controls")!=wanted or len(q.get("checks",[]))!=10
            or cell.get("native_oracles",{}).get("direct_native_census_groups") is not True
            or cell.get("native_oracles",{}).get("float64_native_dates") is not True):
            raise ValueError("Every actual cell must retain exact certified geometry and native oracles")
        for check,(dtype,control) in zip(q["checks"],wanted):
            if (check.get("dtype")!=dtype or check.get("control")!=control
                or check.get("initial_state_sha256")!=native["initial_state_sha256_by_dtype"][dtype]
                or any(check.get(k) is not True for k in ("finite_disposable_AdamW","forest_output_VJP_parity","sample_readout_target_column_order"))):
                raise ValueError("Unchanged complete numerical checks/state/control order required")
    selection=inspection["prespecified_TRAIN_numerical_cells"]
    train=[c for c in frozen.values() if c["split"]=="train"]
    ordered=sorted(train,key=lambda c:(c["num_trees"],c["parent_id"],c["tile_id"]))
    expected_selection={"minimum":{k:ordered[0][k] for k in ("num_trees","parent_id","split","tile_id")},
        "maximum":{k:sorted(train,key=lambda c:(-c["num_trees"],c["parent_id"],c["tile_id"]))[0][k] for k in ("num_trees","parent_id","split","tile_id")}}
    if selection!=expected_selection or evidence.get("prespecified_TRAIN_numerical_cells")!=selection or freeze.get("prespecified_TRAIN_numerical_cells")!=selection:
        raise ValueError("Frozen target-free TRAIN extrema selection differs")
    if [(c["parent_id"],c["tile_id"]) for c in evidence["cells"][:2]]!=[(selection[k]["parent_id"],selection[k]["tile_id"]) for k in ("minimum","maximum")]:
        raise ValueError("Actual numerical TRAIN extrema must run first")
    return native

def gate_from_reader_v1(admission,reader):
    gate={k:reader.decode_json(admission["refs"][k]) for k in GATE_REFS}
    native=require_terminal_numerical_v1(gate["numerical_evidence"],gate["numerical_admission"],gate["inspection_evidence"],
        gate["inspection_freeze"],gate["numerical_terminal"],admission["refs"],gate["inspection_lineage"])
    inspect_terminal=reader.decode_json(gate["inspection_freeze"]["inspection_terminal"])
    if inspect_terminal.get("returncode")!=0 or inspect_terminal.get("watchdog_triggered") is not False:raise ValueError("Actual inspection terminal exit0 required")
    cfg=bridge.load_config_v1()
    for closure in (native["source_sha256"],cfg["metadata_input_sha256"]):
        for p,h in closure.items():reader.read({"path":p,"sha256":h})
    native=dict(native,_inspection=gate["inspection_evidence"])
    return native,cfg["numerical_policy"]

def verify_observation_closure_v1(native,reader):
    """Called only after fresh root storage and memory capabilities pass."""
    for closure in (native["observed_artifact_sha256"],native["derived_artifact_sha256"]):
        for p,h in closure.items():reader.read({"path":p,"sha256":h})

def _dependencies_v1(cfg=None):
    """Source-owned transport supplier; all bridge scientific APIs remain exact ASTs."""
    cfg=cfg or bridge.load_config_v1();bridge.source_report_v1(cfg)
    import numpy as numpy
    import torch as pytorch
    import tskit
    from .data import GraphSequence
    from .reference_encoder import ReferenceFeatureAdapter
    from . import census_biological_bank_v1 as bank
    from . import census_native_qualification_runner_v1 as numerical
    for module,path in ((numpy,".runtime/env/lib/python3.11/site-packages/numpy/__init__.py"),
                        (pytorch,".runtime/env/lib/python3.11/site-packages/torch/__init__.py"),
                        (bank,"src/arggnn/census_biological_bank_v1.py"),(numerical,"src/arggnn/census_native_qualification_runner_v1.py")):
        if Path(module.__file__).resolve()!=(ROOT/path).resolve():raise ValueError("Exact consumer scientific module origin required")
    pytorch.set_num_threads(1)
    tree=ast.parse((ROOT/cfg["extractor"]["source"]).read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==cfg["extractor"]["function"])
    scope={"np":numpy,"torch":pytorch,"math":math,"GraphSequence":GraphSequence,"ReferenceFeatureAdapter":ReferenceFeatureAdapter}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(ROOT/cfg["extractor"]["source"]),"exec"),scope)
    return numpy,pytorch,tskit,bank,numerical,scope["pack_observed_tile"]

def bridge_capsule_v1():
    global _CAPSULE
    if _CAPSULE is None:
        bridge.source_report_v1()
        names=("_load_parent_v1","role_join_v1","physical_contract_v1","extract_native_tsinfer_tile_v1",
            "raw_sequence_bytes_v1","certify_native_tile_v1","validate_tsinfer_certificate_v1",
            "forward_certified_tsinfer_v1","_producer_payload_v1")
        tree=ast.parse((ROOT/bridge.SOURCE).read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        if {n.name for n in nodes}!=set(names):raise ValueError("Exact bridge API AST closure required")
        capsule=dict(vars(bridge));capsule["_dependencies_v1"]=_dependencies_v1
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(ROOT/bridge.SOURCE),"exec"),capsule)
        _CAPSULE=capsule
    return _CAPSULE

@dataclass(frozen=True)
class ConsumerCertificateV1:
    outer:object
    @property
    def provenance(self):return self.outer.native_certificate.provenance
    @property
    def native_trees(self):return self.outer.native_certificate.native_trees

def validate_census_biological_certificate_v1(cert,expected):
    if not isinstance(cert,ConsumerCertificateV1):raise ValueError("Same-tsinfer outer certificate required")
    bridge_capsule_v1()["validate_tsinfer_certificate_v1"](cert.outer,expected)
    return cert
def forward_certified_census_biological_v1(model,cert,**kwargs):
    validate_census_biological_certificate_v1(cert,kwargs["expected_certificate_sha256"])
    return bridge_capsule_v1()["forward_certified_tsinfer_v1"](model,cert.outer,**kwargs)

def _science_v1():
    global np,torch,CensusTreeGNNV1,fingerprint_sha256_v1
    np,torch,_,bank,_,_=_dependencies_v1()
    from .census_tree_encoder_v1 import CensusTreeGNNV1 as Model
    CensusTreeGNNV1=Model;fingerprint_sha256_v1=bank.fingerprint_sha256_v1

def read_manifests_v1(native,qcfg,reader):
    cfg=bridge.load_config_v1();helpers=bridge.metadata_inputs_v1(cfg,reader)
    cells=native["_inspection"]["cells"];paired=[]
    for h,pin in helpers:
        own=[c for c in cells if c["parent_id"]==h["parent_id"]]
        retained=h["coordinate_contract"]["retained_fixed100bp_indexes"];lookup={g:i for i,g in enumerate(retained)}
        tiles=[]
        for cell in sorted(own,key=lambda c:c["tile_id"]):
            grid=cell["lineage"]["original_grid_indexes"]
            expected=[g for g in retained if cell["tile_id"]*250000<=50+100*g<(cell["tile_id"]+1)*250000]
            if grid!=expected:raise ValueError("Frozen exact physical tile output indexes differ")
            tiles.append({"tile_id":cell["tile_id"],"position_indexes":[lookup[g] for g in grid],"original_tree_ids":cell["lineage"]["original_tree_ids"],"frozen":cell})
        if len(tiles)!=8 or sorted(i for t in tiles for i in t["position_indexes"])!=list(range(len(retained))):raise ValueError("Complete original retained bins required")
        paired.append((h,{"pin":pin,"tiles":tiles}))
    return paired

def load_parent_v1(parent,bank,native,qcfg,reader,tmp_directory):
    return bridge_capsule_v1()["_load_parent_v1"](parent,bank["pin"],bridge.load_config_v1(),reader,tmp_directory)

def prepare_tile_v1(parent,bank,tile,loaded,native,qcfg,reader):
    cfg=bridge.load_config_v1();capsule=bridge_capsule_v1()
    ts,roles,positions,offset,coverage,refs,lineage=loaded;previous=tile["frozen"]
    fresh,*_=capsule["extract_native_tsinfer_tile_v1"](ts,roles,positions,coverage,tile["tile_id"],cfg)
    producer=reader.decode_json(previous["producer_ref"])
    if producer!=capsule["_producer_payload_v1"](fresh,lineage,tile["tile_id"],cfg,previous["raw_ref"]):raise ValueError("Fresh native x48 producer differs from frozen same-tsinfer materialization")
    raw=torch.load(io.BytesIO(reader.read(previous["raw_ref"])),map_location="cpu",weights_only=True)
    outer,_=capsule["certify_native_tile_v1"](ts,roles,positions,coverage,tile["tile_id"],
        {"native_observed_ts":refs["native"],"serialized_whole_observed":previous["raw_ref"],"observed_roster":refs["derived"],
         "observed_grid":refs["grid"],"producer_receipt":previous["producer_ref"]},
        dict(lineage,raw_ref=previous["raw_ref"],producer_ref=previous["producer_ref"]),cfg,cached_raw=raw)
    capsule["validate_tsinfer_certificate_v1"](outer,native["expected_certificates"][f"{parent['parent_id']}:{tile['tile_id']}"],cfg)
    if outer.lineage!=previous["lineage"] or len(outer.native_certificate.native_trees)!=previous["num_trees"] or outer.native_certificate.certificate_sha256!=previous["inner_certificate_sha256"]:raise ValueError("Frozen outer/inner native geometry differs")
    return ConsumerCertificateV1(outer),None,None

def review_v1(review,source,config):
    if (review.get("schema")!="census_tsinfer_consumer_independent_review_v1" or review.get("review_role")!="independent_reviewer"
        or review.get("verdict")!="PASS" or review.get("actual_admission") is not False or review.get("fixture_mode") is not False
        or review.get("artifact_sha256",{}).get(source)!=_hash(source)
        or review.get("artifact_sha256",{}).get(config)!=_hash(config)):
        raise ValueError("Fresh independent exact same-tsinfer consumer review required")

def registration_v1(reg,a,cfg,stage):
    if (reg.get("schema")!="census_tsinfer_consumer_root_registration_v1" or reg.get("authorizer")!="root_agent"
        or reg.get("status")!="registered" or reg.get("job_id")!=a["job_id"] or reg.get("stage")!=stage
        or reg.get("condition")!=cfg["condition"] or reg.get("hypothesis_id")!="H021" or not reg.get("repeat_reason")
        or reg.get("query_truth_authorized")!=(stage=="fit") or reg.get("fit_authorized")!=(stage=="fit")):
        raise ValueError("Distinct root registered condition and exact truth purpose required")

def budget_check_v1(storage,first=False,required_write_bytes=0):
    from . import project_budget_v4 as budget
    cfg=load_config_v1()
    return budget.check_reservation(ROOT,storage["reservation"],
        expected_source_sha256=cfg["budget_source"]["sha256"],expected_config_sha256=cfg["budget_config"]["sha256"],
        initial_ledger_pins=storage["initial_ledger_pins"] if first else None,boundary=True,required_write_bytes=required_write_bytes)

def storage_v1(storage,a,cfg,source,config,first=True):
    prefix=cfg["output_prefix"]+a["job_id"]
    reservation=storage.get("reservation",{})
    if (storage.get("schema")!="census_tsinfer_consumer_storage_v1" or storage.get("status")!="active"
        or storage.get("job_id")!=a["job_id"] or storage.get("output_directory")!=prefix or a["output_directory"]!=prefix
        or storage.get("reserved_new_bytes")!=cfg["output_reservation_bytes"] or storage.get("mode")!="project_budget_v4"
        or reservation.get("output")!=str(ROOT/prefix) or reservation.get("auxiliary_scopes")!=[]
        or reservation.get("reserved_bytes")!=cfg["output_reservation_bytes"]
        or any(ref not in reservation.get("consumer_bindings",[]) for ref in ({"path":source,"sha256":_hash(source)},{"path":config,"sha256":_hash(config)},a["refs"]["runtime"]))
        or set(storage.get("initial_ledger_pins",{}))!={"legacy_ledger_sha256","supplemental_ledger_sha256"}
        or any(not bridge._sha(h) for h in storage["initial_ledger_pins"].values())
        or not _utc(storage["created_at_utc"])<=datetime.now(timezone.utc)<_utc(storage["expires_at_utc"])):
        raise ValueError("Fresh exact reviewed v4 consumer reservation required")
    from . import project_budget_v4 as budget
    account=budget.check_reservation(ROOT,reservation,expected_source_sha256=cfg["budget_source"]["sha256"],
        expected_config_sha256=cfg["budget_config"]["sha256"],initial_ledger_pins=storage["initial_ledger_pins"] if first else None,boundary=True)
    if account.get("legacy_ledger_mutated") is not False or account.get("supplemental_ledger_mutated") is not False:raise ValueError("Readonly reservation guard required")
    return storage

def memory_v1(memory,a,cfg):
    if (memory.get("schema")!="census_tsinfer_consumer_memory_v1" or memory.get("status")!="active"
        or memory.get("job_id")!=a["job_id"] or memory.get("external_watchdog") is not True
        or memory.get("GPU_total_reservation_mib")!=cfg["GPU_total_reservation_mib"]
        or memory.get("CPU_total_reservation_bytes")!=cfg["CPU_total_reservation_bytes"]
        or not _utc(memory["created_at_utc"])<=datetime.now(timezone.utc)<_utc(memory["expires_at_utc"])
        or _utc(memory["created_at_utc"])<_utc(a["created_at_utc"]) or _utc(memory["expires_at_utc"])>_utc(a["expires_at_utc"])):
        raise ValueError("Fresh exact full-backward memory cap/watchdog required")

def runtime_v1(runtime,native,reader,schema):
    numerical=reader.decode_json(native["refs"]["runtime"]);cfg=bridge.load_config_v1()
    if (runtime.get("schema")!=schema or runtime.get("python")!=str(ROOT/".runtime/env/bin/python")
        or runtime.get("numerical_identity")!=numerical["identity"]
        or runtime.get("runtime_file_sha256")!=numerical["runtime_file_sha256"]
        or set(runtime.get("runtime_file_sha256",{}))!=set(cfg["runtime_files"])):
        raise ValueError("Exact completed numerical runtime identity and bytes required")
    if Path(sys.executable).resolve()!=(ROOT/".runtime/env/bin/python").resolve():raise ValueError("Main project interpreter required")
    for p,h in runtime["runtime_file_sha256"].items():reader.read({"path":p,"sha256":h},runtime_file=True)
    _science_v1()
    import tskit
    actual={"python":sys.version.split()[0],"torch":str(torch.__version__),"numpy":str(np.__version__),"tskit":str(tskit.__version__)}
    if actual!={k:numerical["identity"][k] for k in actual}:raise ValueError("Live package identity changed")
    return numerical

def target_bindings_v1(bindings,metadata,a,cfg):
    if (bindings.get("schema")!="census_tsinfer_targets_root_binding_v1" or bindings.get("observed_artifact_sha256")!=a["observed_artifact_sha256"]
        or bindings.get("manifest_ref")!=a["refs"]["target_manifest"] or set(bindings.get("parents",{}))!=set(PARENTS)
        or a["refs"]["target_manifest"]!=cfg["external_target_manifest"]):
        raise ValueError("Separate same-tsinfer target bindings required")
    byid={row["parent_id"]:row for row in metadata["replicates"]}
    for parent,record in bindings["parents"].items():
        split="train" if "_train_" in parent else "val";wanted={"path":f"data/biological_paths_execution_pilot_v1/{parent}/truth/full_truth_intervals.npz","sha256":byid[parent]["files"]["truth/full_truth_intervals.npz"]["sha256"]}
        if (set(record)!={"split","truth_npz","query_sample_indexes","query_raw_node_ids"} or record["split"]!=split or byid[parent]["split"]!=split
            or record["truth_npz"]!=wanted or len(record["query_sample_indexes"])!=120 or len(set(record["query_sample_indexes"]))!=120
            or len(record["query_raw_node_ids"])!=120 or len(set(record["query_raw_node_ids"]))!=120):
            raise ValueError("Original simulation target/source sample join required")

def admit_fit_v1(record):
    reader=ConsumerReaderV1();a=reader.decode_json(record)
    fields={"schema","status","root_authorizer","fit_authorized","job_id","created_at_utc","expires_at_utc","refs","source_sha256",
        "seed","device","output_directory","expected_certificates","observed_artifact_sha256","group_permutation"}
    if set(a)!=fields or not canonical_job_v1(a["job_id"]) or a["schema"]!="census_tsinfer_fit_admission_v1" or a["status"]!="active" or a["root_authorizer"]!="root_agent" or a["fit_authorized"] is not True or set(a["refs"])!=REFS:
        raise ValueError("Fresh exact root same-tsinfer fit capability required")
    _check_active_v1(a)
    from .native_argformer_exec import validate_lease
    require_logical_job_v1(validate_lease(),a)
    cfg=load_config_v1();source_report_v1()
    if reader.decode_json(a["refs"]["config"])!=cfg or a["refs"]["config"]!={"path":CONFIG,"sha256":_hash(CONFIG)}:raise ValueError("Exact new consumer configuration required")
    sources=dict(cfg["source_sha256"],**{SOURCE:_hash(SOURCE)})
    if a["source_sha256"]!=sources:raise ValueError("Exact complete new consumer source closure required")
    for p,h in sources.items():reader.read({"path":p,"sha256":h})
    review_v1(reader.decode_json(a["refs"]["review"]),SOURCE,CONFIG)
    registration_v1(reader.decode_json(a["refs"]["registration"]),a,cfg,"fit")
    native,qcfg=gate_from_reader_v1(a,reader)
    if (a["job_id"]==native["job_id"] or a["expected_certificates"]!=native["expected_certificates"]
        or a["observed_artifact_sha256"]!=native["observed_artifact_sha256"] or a["group_permutation"]!=native["group_permutation"]
        or a["seed"] not in cfg["seeds"] or a["device"]!="cuda" or cfg["model"]!=qcfg["model"]):
        raise ValueError("Exact qualified same-tsinfer scientific condition required")
    storage=storage_v1(reader.decode_json(a["refs"]["storage"]),a,cfg,SOURCE,CONFIG)
    memory_v1(reader.decode_json(a["refs"]["memory"]),a,cfg)
    verify_observation_closure_v1(native,reader)
    runtime_v1(reader.decode_json(a["refs"]["runtime"]),native,reader,"census_tsinfer_fit_runtime_v1")
    launcher=reader.decode_json(a["refs"]["launcher"])
    if (launcher.get("schema")!="census_tsinfer_fit_launcher_v1" or launcher.get("job_id")!=a["job_id"] or launcher.get("device")!=a["device"]
        or launcher.get("GPU_admitted")!=(a["device"]=="cuda") or launcher.get("python")!=".runtime/env/bin/python"
        or launcher.get("external_watchdog") is not True or launcher.get("profile_gate_passed") is not True
        or launcher.get("PYTHONDONTWRITEBYTECODE")!="1" or os.environ.get("PYTHONDONTWRITEBYTECODE")!="1"
        or (a["device"]=="cpu" and os.environ.get("CUDA_VISIBLE_DEVICES")!="")):raise ValueError("Exact profile-qualified launcher/watchdog required")
    profile=reader.decode_json(a["refs"]["profile"])
    if (profile.get("schema")!="census_tsinfer_fit_profile_gate_v1" or profile.get("status")!="qualified_for_fixed_protocol"
        or profile.get("fit_config_ref")!=a["refs"]["config"] or profile.get("source_sha256")!=a["source_sha256"]
        or profile.get("numerical_ref")!=a["refs"]["numerical_evidence"] or profile.get("device")!=a["device"]
        or profile.get("controls")!=list(CONTROLS) or profile.get("model")!=cfg["model"]
        or profile.get("full_whole_tree_forward_backward_measured") is not True
        or profile.get("fresh_root_resource_reservation") is not True or not profile.get("profile_evidence_ref")
        or (a["device"]=="cuda" and profile.get("cuda_float32_output_VJP_parity_qualified") is not True)):
        raise ValueError("Actual same-tsinfer full-backward/parity profile required")
    profile_source="src/arggnn/census_tsinfer_device_profile_v1.py"
    profile_config="configs/census_tsinfer_device_profile_v1.json"
    pconfig=json.loads((ROOT/profile_config).read_text())
    psource=dict(pconfig["source_sha256"],**{profile_source:_hash(profile_source)})
    if profile.get("profile_implementation_ref")!={"path":profile_source,"sha256":cfg["source_sha256"][profile_source]} or profile.get("profile_config_ref")!={"path":profile_config,"sha256":_hash(profile_config)}:
        raise ValueError("Exact dedicated same-tsinfer profile implementation/config required")
    measured=reader.decode_json(profile["profile_evidence_ref"])
    if (measured.get("schema")!="census_tsinfer_device_profile_evidence_v1" or measured.get("status")!="completed_target_free_device_parity_and_backward_profile"
        or measured.get("numerical_ref")!=a["refs"]["numerical_evidence"] or measured.get("query_truth_accessed") is not False
        or measured.get("scientific_fit_performed") is not False
        or measured.get("all_five_controls_CPU_float32_CUDA_float32_output_VJP_parity") is not True
        or measured.get("full_whole_tree_forward_backward_measured") is not True
        or measured.get("config_ref")!=profile["profile_config_ref"] or measured.get("source_sha256")!=psource
        or len(measured.get("selected_cells",[]))!=2):
        raise ValueError("Genuine completed same-tsinfer profile receipt required")
    terminal=reader.decode_json(profile["profile_terminal_ref"])
    profile_admission=reader.decode_json(measured["admission_ref"])
    if terminal!={"schema":"census_tsinfer_device_profile_root_terminal_binding_v1","status":"complete","returncode":0,
        "evidence_ref":profile["profile_evidence_ref"],"admission_ref":measured["admission_ref"],"job_id":profile_admission["job_id"]}:
        raise ValueError("Actual dedicated profile terminal exit0 required")
    if profile_admission.get("source_sha256")!=psource or profile_admission.get("refs",{}).get("numerical_evidence")!=a["refs"]["numerical_evidence"]:
        raise ValueError("Dedicated profile execution lineage differs")
    selection=native["_inspection"]["prespecified_TRAIN_numerical_cells"]
    for row,key in zip(measured["selected_cells"],("minimum","maximum")):
        chosen=selection[key]
        if (any(row.get(k)!=chosen[k] for k in ("parent_id","tile_id")) or row.get("original_tree_count")!=chosen["num_trees"]
            or row.get("certificate_sha256")!=a["expected_certificates"][f"{chosen['parent_id']}:{chosen['tile_id']}"]
            or [c.get("control") for c in row.get("checks",[])]!=list(CONTROLS)):
            raise ValueError("Exact same-tsinfer extrema/certified five-control measured profile required")
        for check in row["checks"]:
            peak=check.get("selected_device",{}).get("peak_CUDA_reserved_bytes")
            if (check.get("CPU_float32_selected_device_float32_output_VJP_parity") is not True or check.get("finite_disposable_AdamW") is not True
                or type(peak) is not int or peak<=0 or peak>cfg["GPU_total_reservation_mib"]*1024*1024):
                raise ValueError("Genuine bounded CUDA full-backward parity measurements required")
    bindings=reader.decode_json(a["refs"]["target_bindings"]);metadata=reader.decode_json(a["refs"]["target_manifest"])
    target_bindings_v1(bindings,metadata,a,cfg)
    return a,cfg,native,qcfg,bindings,reader,storage

def _check_resources_v1(a,storage,lease):
    _check_active_v1(a)
    if datetime.now(timezone.utc)>=_utc(storage["expires_at_utc"]):raise ValueError("Fit storage authority expired")
    from .biological_candidate import check_fit_lease
    check_fit_lease(lease);require_logical_job_v1(lease,a);budget_check_v1(storage)
    memory_v1(ConsumerReaderV1().decode_json(a["refs"]["memory"]),a,load_config_v1())
    if a["device"]=="cuda" and torch.cuda.max_memory_reserved()>load_config_v1()["GPU_total_reservation_mib"]*1024*1024:
        raise ValueError("Measured allocator peak exceeds exact full-backward reservation")


def _jsonable_v1(value):
    if isinstance(value,torch.Tensor):return _jsonable_v1(value.detach().cpu().numpy())
    if isinstance(value,np.ndarray):return _jsonable_v1(value.tolist())
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):_jsonable_v1(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_jsonable_v1(v) for v in value]
    return value

class FitOutputV1:
    """Exclusive bounded receipt, prediction and checkpoint writes, no overwrite."""
    def __init__(self,directory,reservation,reader):
        self.path=(reader.root/reader.relative(directory)).resolve()
        try:self.path.relative_to(reader.root.resolve())
        except ValueError:raise ValueError('Fit output resolves outside repository') from None
        self.reservation=reservation
        if self.path.exists():raise ValueError('Exclusive new fit output required')
        self.path.mkdir(parents=True);self.limit=reservation['reserved_new_bytes'];self.used=0
    def save(self,name,value,kind='json'):
        if Path(name).name!=name:raise ValueError('Named fit files only')
        if self.reservation.get('expires_at_utc') and datetime.now(timezone.utc)>=_utc(self.reservation['expires_at_utc']):raise ValueError('Fit output reservation expired')
        budget_check_v1(self.reservation)
        buffer=io.BytesIO()
        if kind=='json':data=(json.dumps(_jsonable_v1(value),sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
        else:
            if kind=='npz':np.savez_compressed(buffer,**value)
            elif kind=='torch':torch.save(value,buffer)
            else:raise ValueError('Unknown bounded output type')
            data=buffer.getvalue()
        if self.used+len(data)+8192>self.limit:raise ValueError('Reserved fit output cap exceeded')
        stat=os.statvfs(self.path)
        if len(data)+8192>stat.f_bavail*stat.f_frsize:raise ValueError('Insufficient free bytes')
        budget_check_v1(self.reservation,required_write_bytes=len(data)+8192)
        with (self.path/name).open('xb') as stream:stream.write(data)
        self.used+=len(data)+8192
        budget_check_v1(self.reservation)
        return {'path':str((self.path/name).relative_to(ROOT)),'sha256':hashlib.sha256(data).hexdigest()}

def _check_active_v1(admission):
    now=datetime.now(timezone.utc)
    if (not _utc(admission['created_at_utc'])<=now<_utc(admission['expires_at_utc'])
        or (_utc(admission['expires_at_utc'])-_utc(admission['created_at_utc'])).total_seconds()>86400):raise ValueError('Fit admission expired or exceeds24 hours')

def metric_helpers_v1():
    spec=importlib.util.spec_from_file_location('census_biological_span_helpers_v1',ROOT/'scripts/train_ancestralpaths_biological_v1.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    return helper

def scoring_parent_v1(parent,bank,loaded,reader,root_binding):
    _,roster,positions,_,coverage,refs,_=loaded
    source=json.loads(reader.read(refs['source']));query=np.array([i for i,r in enumerate(source) if r['role']=='query'],dtype=np.int64)
    raw=np.array([source[i]['observed_node_id'] for i in query],dtype=np.int64)
    if query.tolist()!=root_binding['query_sample_indexes'] or raw.tolist()!=root_binding['query_raw_node_ids']:raise ValueError('External target columns must equal original observed query identities')
    retained=np.flatnonzero((positions>=coverage[0])&(positions<coverage[1]));edges=np.r_[100.*retained,100.*(retained[-1]+1)]
    edges[0]=max(edges[0],coverage[0]);edges[-1]=min(edges[-1],coverage[1])
    cohorts=np.array([source[i]['observed_cohort'] for i in query])
    required={'present_day':64,'Bronze_Age':16,'BAA':8,'Neo':16,'Yam':16}
    if {key:int((cohorts==key).sum()) for key in required}!=required or len(query)!=120:raise ValueError('Frozen query strata changed')
    return {'parent_id':parent['parent_id'],'split':parent['split'],'samples':source,'query_slots':query,'raw_query_nodes':raw,
        'cohorts':cohorts,'positions':positions[retained],'edges':edges,'retained_indexes':retained,'coverage_original':coverage,
        'query_ids':np.array([r.actual_id for r in roster if r.role=='query'],dtype=np.int64)}

class ExternalTargetV1:
    """Only C4 interval targets. Target graph/source ancestry identities never load."""
    def __init__(self,parent,binding,purpose,reader,prediction_record=None):
        if purpose not in ('TRAIN_loss','VAL_after_sealed_prediction') or parent['split']!=('train' if purpose=='TRAIN_loss' else 'val'):raise ValueError('Target purpose/split mismatch')
        if purpose=='VAL_after_sealed_prediction':
            if prediction_record is None:raise ValueError('VAL target opening requires sealed prediction')
            reader.read(prediction_record)
        data=reader.read(binding['truth_npz'])
        with np.load(io.BytesIO(data),allow_pickle=False) as archive:
            if set(archive.files)!={'intervals_bp','query_raw_node_ids','native_paths_1based','source_classes_0based'}:raise ValueError('Exact full-interval target schema required')
            self.intervals=archive['intervals_bp'].copy();self.c4=archive['source_classes_0based'].copy();ids=archive['query_raw_node_ids'].copy()
        # Six-path labels are deliberately never decoded or used.
        if (self.intervals.dtype!=np.float64 or self.intervals.ndim!=2 or self.intervals.shape[1]!=2
            or self.intervals[0,0]!=0 or self.intervals[-1,1]!=2000000 or not np.array_equal(self.intervals[:-1,1],self.intervals[1:,0])
            or not np.isfinite(self.intervals).all() or np.any(self.intervals[:,1]<=self.intervals[:,0])
            or self.c4.shape!=(len(self.intervals),120) or self.c4.dtype.kind not in 'iu' or np.any((self.c4<0)|(self.c4>3))
            or not np.array_equal(ids,parent['raw_query_nodes'])):raise ValueError('Full C4 interval/query identity target join invalid')
        self.parent_id=parent['parent_id'];self.split=parent['split'];self.purpose=purpose
        self.provenance={'record':binding['truth_npz'],'purpose':purpose,'prediction_record':prediction_record}
    def target_mass(self,parent,helpers):
        bins,rows,spans=helpers.intersections(parent['edges'],self.intervals)
        mass=np.zeros((len(parent['positions']),120,4),dtype=np.float64)
        np.add.at(mass,(bins[:,None],np.arange(120)[None,:],self.c4[rows]),spans[:,None])
        if not np.allclose(mass.sum(-1),np.diff(parent['edges'])[:,None],rtol=0,atol=1e-8):raise ValueError('Exact class-span mass incomplete')
        return mass

def optimizer_groups_v1(model,cfg):
    rate=[p for name,p in model.named_parameters() if name.endswith(('raw_exchange','raw_damping','raw_eta','mixture_logits'))]
    ids={id(p) for p in rate}
    return [{'params':[p for p in model.parameters() if id(p) not in ids],'weight_decay':cfg['training']['weight_decay']},
            {'params':rate,'weight_decay':0.}]

def matched_states_v1(cfg,seed,device):
    if seed not in cfg['seeds']:raise ValueError('Frozen optimizer seed required')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    initial=CensusTreeGNNV1(**cfg['model']).to(device=device,dtype=torch.float32)
    initial_hash=fingerprint_sha256_v1({k:v.detach().cpu() for k,v in initial.state_dict().items()})
    states={}
    for control in CONTROLS:
        model=deepcopy(initial)
        states[control]={'model':model,'optimizer':torch.optim.AdamW(optimizer_groups_v1(model,cfg),lr=cfg['training']['learning_rate'],betas=(.9,.999),eps=1e-8),
            'initial':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'initial_sha256':initial_hash,'steps':0,'best':float('inf'),'best_epoch':None,'history':[]}
    return states

def train_step_v1(state,cert,expected_hash,mass,control,permutation,cfg,device):
    if cert.provenance['split']!='train':raise ValueError('TRAIN-only backward requires TRAIN observed parent')
    model=state['model'];model.train();state['optimizer'].zero_grad(set_to_none=True)
    kwargs={'group_permutation':permutation} if control=='permuted_groups' else {}
    result=forward_certified_census_biological_v1(model,cert,expected_certificate_sha256=expected_hash,control=control,readout_chunk_size=cfg['readout_chunk_size'],**kwargs)
    target=torch.as_tensor(mass,dtype=torch.float64,device=device)
    if result.logits.shape!=target.shape or not bool(torch.isfinite(target).all() & (target>=0).all()) or not target.sum()>0:raise ValueError('Actual query120 exact TRAIN mass shape/values differ')
    loss=-(torch.log_softmax(result.logits,dim=-1).double()*target).sum()/target.sum()
    if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite exact TRAIN query-bp loss')
    loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Nonfinite TRAIN gradient')
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['training']['gradient_clip_norm'],error_if_nonfinite=True)
    state['optimizer'].step();state['steps']+=1
    if not all(torch.isfinite(p).all() for p in model.parameters()):raise ValueError('Nonfinite fitted parameter')
    validate_census_biological_certificate_v1(cert,expected_hash)
    return {'loss':float(loss.detach()),'unclipped_gradient_norm':float(norm),'steps':state['steps']}

def raw_prediction_v1(state,cert,expected_hash,control,permutation,cfg):
    state['model'].eval();kwargs={'group_permutation':permutation} if control=='permuted_groups' else {}
    with torch.no_grad():
        result=forward_certified_census_biological_v1(state['model'],cert,expected_certificate_sha256=expected_hash,control=control,readout_chunk_size=cfg['readout_chunk_size'],**kwargs)
        logits=result.logits.detach().cpu();p=logits.softmax(-1).numpy()
    if not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(-1),1.,rtol=1e-5,atol=1e-6):raise ValueError('Raw C4 probabilities invalid without floors/renormalization')
    return logits.numpy(),p,result.scored_query_ids.numpy()

def strata_v1(parent):
    groups={'primary_present_day_64':np.flatnonzero(parent['cohorts']=='present_day'),
        'secondary_ancient_56':np.flatnonzero(parent['cohorts']!='present_day'),'all_queries_120':np.arange(120)}
    for cohort in ('Bronze_Age','BAA','Neo','Yam'):groups['age_stratum_'+cohort]=np.flatnonzero(parent['cohorts']==cohort)
    return groups

def score_sealed_v1(prediction_record,parent,binding,reader,helpers,cfg):
    data=reader.read(prediction_record)
    with np.load(io.BytesIO(data),allow_pickle=False) as archive:
        p=archive['C4_probabilities'].copy()
        checks={'positions_bp':parent['positions'],'bin_edges_bp':parent['edges'],'retained_fixed100bp_indexes':parent['retained_indexes'],'query_sample_indexes':parent['query_slots']}
        if any(not np.array_equal(archive[k],v) for k,v in checks.items()) or p.shape!=(len(parent['positions']),120,4):raise ValueError('Sealed prediction output/physical/identity join differs')
    if not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(-1),1,atol=1e-6,rtol=1e-5):raise ValueError('Invalid raw probabilities')
    target=ExternalTargetV1(parent,binding,'VAL_after_sealed_prediction',reader,prediction_record)
    groups={name:helpers.full_span_score(p,parent,target,columns,cfg['metrics']) for name,columns in strata_v1(parent).items()}
    return {'parent_id':parent['parent_id'],'split':'val','prediction_seal':prediction_record,'target_provenance':target.provenance,'groups':groups,'development_only':True,'TEST_parents':0}

def _nll_v1(value):return float('inf') if value['infinite_nll_query_bp'] else value['nll']

def run_fit_v1(admission_ref):
    """Actual use is root-admitted only. Author tests call handmade components."""
    admission,cfg,native_admission,qcfg,bindings,reader,storage=admit_fit_v1(admission_ref)
    helpers=metric_helpers_v1();paired=read_manifests_v1(native_admission,qcfg,reader)
    output=FitOutputV1(admission['output_directory'],storage,reader);tmp=output.path/'tmp';tmp.mkdir()
    from .biological_candidate import configure_fit,check_fit_lease
    device,runtime,lease=configure_fit(admission['device'],cfg,output.path)
    output.used=sum(max(p.stat().st_size,p.stat().st_blocks*512)+8192 for p in output.path.rglob('*') if p.is_file())
    if output.used>output.limit:raise ValueError('Runtime qualification writes exceeded fit reservation')
    states=matched_states_v1(cfg,admission['seed'],device);permutation=torch.tensor(admission['group_permutation'],dtype=torch.long)
    receipt={'schema':'census_tsinfer_fit_evidence_v1','status':'running','hypothesis_id':'H021','job_id':admission['job_id'],'admission_ref':admission_ref,
        'seed':admission['seed'],'source_sha256':admission['source_sha256'],'config_ref':admission['refs']['config'],'numerical_ref':admission['refs']['numerical_evidence'],
        'runtime':runtime,'same_initial_weights_sha256':states['proposed']['initial_sha256'],'models':{},'development_VAL_parents':2,'TEST_parents':0,
        'handmade_fixture':bool(runtime.get('handmade_CPU',False)),'full_whole_tree_architecture':True,'target_inputs':False,'older_reference_proxy_evidence':True,'fitting_claim':False}
    output.save('started.json',receipt);start=time.monotonic()
    try:
        for epoch in range(1,cfg['training']['epochs']+1):
            _check_resources_v1(admission,storage,lease)
            order=list(range(cfg['training']['TRAIN_tiles_per_epoch']));random.Random(admission['seed']+epoch).shuffle(order)
            # Load one qualified full TRAIN parent at a time, but honor frozen global tile order.
            loaded_by_parent={};parent_scores={};targets={};train_tiles=[]
            for parent,bank in paired:
                if parent['split']!='train':continue
                loaded=load_parent_v1(parent,bank,native_admission,qcfg,reader,tmp_directory=tmp)
                loaded_by_parent[parent['parent_id']]=loaded
                score_parent=scoring_parent_v1(parent,bank,loaded,reader,bindings['parents'][parent['parent_id']]);parent_scores[parent['parent_id']]=score_parent
                target=ExternalTargetV1(score_parent,bindings['parents'][parent['parent_id']],'TRAIN_loss',reader)
                targets[parent['parent_id']]=target.target_mass(score_parent,helpers)
                train_tiles.extend((parent,bank,tile) for tile in sorted(bank['tiles'],key=lambda t:t['tile_id']))
            losses={c:[] for c in CONTROLS}
            for index in order:
                _check_resources_v1(admission,storage,lease)
                parent,bank,tile=train_tiles[index];cert,_,_=prepare_tile_v1(parent,bank,tile,loaded_by_parent[parent['parent_id']],native_admission,qcfg,reader)
                identity=f"{parent['parent_id']}:{tile['tile_id']}";expected=admission['expected_certificates'][identity]
                validate_census_biological_certificate_v1(cert,expected)
                indexes=np.asarray(tile['position_indexes'],dtype=int)
                for control in CONTROLS:
                    _check_resources_v1(admission,storage,lease)
                    losses[control].append(train_step_v1(states[control],cert,expected,targets[parent['parent_id']][indexes],control,permutation,cfg,device)['loss'])
                del cert
            del loaded_by_parent,targets
            # All five models' VAL predictions are sealed before any epoch VAL target read.
            seals={c:[] for c in CONTROLS};val_parents={}
            for parent,bank in paired:
                if parent['split']!='val':continue
                loaded=load_parent_v1(parent,bank,native_admission,qcfg,reader,tmp_directory=tmp)
                score_parent=scoring_parent_v1(parent,bank,loaded,reader,bindings['parents'][parent['parent_id']]);val_parents[parent['parent_id']]=score_parent
                logits={c:np.empty((len(score_parent['positions']),120,4),dtype=np.float32) for c in CONTROLS};seen=np.zeros(len(score_parent['positions']),dtype=bool)
                for tile in sorted(bank['tiles'],key=lambda t:t['tile_id']):
                    _check_resources_v1(admission,storage,lease)
                    cert,_,_=prepare_tile_v1(parent,bank,tile,loaded,native_admission,qcfg,reader);expected=admission['expected_certificates'][f"{parent['parent_id']}:{tile['tile_id']}"]
                    indexes=np.asarray(tile['position_indexes'],dtype=int)
                    if seen[indexes].any():raise ValueError('Repeated VAL physical output bins')
                    for control in CONTROLS:
                        _check_resources_v1(admission,storage,lease)
                        values,_,ids=raw_prediction_v1(states[control],cert,expected,control,permutation,cfg)
                        if not np.array_equal(ids,score_parent['query_ids']):raise ValueError('VAL query output identity differs')
                        logits[control][indexes]=values
                    seen[indexes]=True;del cert
                if not seen.all():raise ValueError('Missing VAL bins, no denominator reduction')
                for control in CONTROLS:
                    values=logits[control];p=torch.from_numpy(values).softmax(-1).numpy()
                    seal=output.save(f'{control}.epoch_{epoch:02d}.{parent["parent_id"]}.npz',{'logits':values,'C4_probabilities':p,
                        'positions_bp':score_parent['positions'],'bin_edges_bp':score_parent['edges'],'retained_fixed100bp_indexes':score_parent['retained_indexes'],
                        'query_sample_indexes':score_parent['query_slots'],'normalization_length_bp':np.array(2000000)},'npz')
                    seals[control].append((parent['parent_id'],seal))
                del loaded,logits
            _check_active_v1(admission)
            for control in CONTROLS:
                _check_resources_v1(admission,storage,lease)
                cells=[score_sealed_v1(seal,val_parents[parent_id],bindings['parents'][parent_id],reader,helpers,cfg) for parent_id,seal in seals[control]]
                nll=float(np.mean([_nll_v1(cell['groups']['primary_present_day_64']) for cell in cells]));state=states[control]
                row={'epoch':epoch,'TRAIN_mean_tile_query120_NLL':float(np.mean(losses[control])),'VAL_equal_parent_present64_NLL':None if not np.isfinite(nll) else nll,'VAL_NLL_is_infinite':not np.isfinite(nll),'optimizer_steps':state['steps'],'cells':cells}
                state['history'].append(row);output.save(f'{control}.epoch_{epoch:02d}.metrics.json',row)
                # Every epoch runs. Selection is descriptive development VAL minimum, earliest tie.
                if state['best_epoch'] is None or nll<state['best']:
                    checkpoint=output.save(f'{control}.best_epoch_{epoch:02d}.pt',{'model_state':{k:v.detach().cpu() for k,v in state['model'].state_dict().items()},
                        'optimizer_state':state['optimizer'].state_dict(),'epoch':epoch,'control':control,'seed':admission['seed'],'admission_ref':admission_ref},'torch')
                    state.update(best=nll,best_epoch=epoch,checkpoint=checkpoint)
            print(json.dumps({'stage':'census_fit_epoch','seed':admission['seed'],'epoch':epoch,'all_five_controls_steps':epoch*cfg['training']['TRAIN_tiles_per_epoch']}),flush=True)
        for control,state in states.items():
            if state['steps']!=cfg['training']['optimizer_steps_per_control_seed'] or state['best_epoch'] is None:raise ValueError('Incomplete fixed-budget/finite-selected-checkpoint fit')
            saved=torch.load(io.BytesIO(reader.read(state['checkpoint'])),map_location='cpu',weights_only=True)
            delta=max(float((v-state['initial'][k]).abs().max()) for k,v in saved['model_state'].items() if v.is_floating_point())
            if not delta>0:raise ValueError('Selected checkpoint has no genuine trained weight change')
            selected=state['history'][state['best_epoch']-1]
            means={}
            for group in strata_v1(next(iter(val_parents.values()))):
                means[group]={key:float(np.mean([cell['groups'][group][key] for cell in selected['cells']])) for key in ('accuracy','brier','calibration_ece')}
                cells=[cell['groups'][group] for cell in selected['cells']]
                means[group]['nll']=None if any(c['infinite_nll_query_bp'] for c in cells) else float(np.mean([c['nll'] for c in cells]))
                means[group]['NLL_is_infinite']=any(c['infinite_nll_query_bp'] for c in cells)
            receipt['models'][control]={'status':f"fixed{cfg['training']['epochs']}epochs_{state['steps']}steps_complete",'initial_sha256':state['initial_sha256'],'selected_epoch':state['best_epoch'],
                'checkpoint':state['checkpoint'],'optimizer_steps':state['steps'],'selected_weight_max_delta':delta,'selected_metrics':selected,'equal_parent_metrics':means}
        _check_resources_v1(admission,storage,lease)
        # Recheck all pinned sources and admission bytes at terminal completion.
        reader.read(admission_ref)
        for path,sha in admission['source_sha256'].items():reader.read({'path':path,'sha256':sha})
        receipt.update(status='completed_five_fixed_budget_development_controls_one_seed',fitting_claim=not receipt['handmade_fixture'],optimizer_fit_completed=True,elapsed_seconds=time.monotonic()-start,output_accounted_bytes=output.used)
    except Exception as error:
        receipt.update(status='failed_partial_preserved',error_type=type(error).__name__,error=str(error),elapsed_seconds=time.monotonic()-start,output_accounted_bytes=output.used)
        output.save('failed.json',receipt);raise
    output.save('completed.json',receipt);return receipt
def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--execute",action="store_true");parser.add_argument("--admission");parser.add_argument("--expected-admission-sha256")
    args=parser.parse_args(argv)
    if not args.execute:print(json.dumps(source_report_v1(),indent=2));return 0
    if not args.admission or not args.expected_admission_sha256:raise ValueError("Fresh root admission required")
    from .native_argformer_exec import validate_lease,apply_and_validate_title
    validate_lease();apply_and_validate_title()
    result=run_fit_v1({"path":args.admission,"sha256":args.expected_admission_sha256});print(json.dumps({"status":result["status"]}));return 0
if __name__=="__main__":raise SystemExit(main())
