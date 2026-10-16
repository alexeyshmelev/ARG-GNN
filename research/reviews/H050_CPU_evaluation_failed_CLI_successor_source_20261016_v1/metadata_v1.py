"""H050-only CPU EVALUATE metadata successor. Standard-library import, no actions.

Root owns every real proof, qualification, admission and launch. These APIs only
assemble fresh metadata after an actual H050 logical terminal, without requiring
the C3 job or its shared controller to retire.
"""
from pathlib import Path
from datetime import datetime, timezone
import ast, copy, hashlib, importlib.util, json, os, re, stat, sys

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
HERE = "research/reviews/H050_CPU_evaluation_failed_CLI_successor_source_20261016_v1"
SOURCE = HERE + "/metadata_v1.py"
RUNTIME = HERE + "/runtime_v1.py"
GPU_META = "research/reviews/H050_GPU0_shared_eval_lineage_source_20261016_v1/runtime_metadata_adapter_v1.py"
GPU_META_SHA = "f74efc5c2bb1effdc13a791d7354a0db041db16051e2f607e5a1fb99902b9bae"
TERMINAL = HERE + "/failed_cli_terminal_v1.py"
TERMINAL_SHA = "db29f245b4c37d85404592bdbe8809c6b15bfc8c17557d9819185b4f11ed536f"
PRODUCER = "research/reviews/H050_direct_readout_cut_memory_source_20261015_v1/producer_integration_v1/producer_v1.py"
CONTEXT = "research/reviews/H050_direct_readout_cut_memory_source_20261015_v1/producer_integration_v1/root_bridge_v1/context_factory_v1.py"
TRAIN_INPUT = "research/reviews/GPU0_shared_root_integration_20261016_v1/admitted_H050_GPU0_train_v1/input.json"
TRAIN_INPUT_SHA = "1100941aaf8b1431cb0c4ea963daf568843ed1b76caa397c98bedb0fe3e57576"
TRAIN_CONFIG = "research/reviews/GPU0_shared_admission_draft_20261016_v1/bound_metadata_v1/H050_config.json"
TRAIN_CONFIG_SHA = "926856f74b0bf9f083baa128beedfcdbebbe7fdb9ffac3557b002397ec8f22a3"
GUARD = {"path":"research/reviews/prospective_980GB_capacity_source_v1/prospective_980gb_v1.py","sha256":"fc71030865e0a3d5da31120d4daa2caa0b2d39ec7dd212b8c46c476b5a866d56"}
POLICY = {"path":"research/reviews/prospective_980GB_capacity_source_v1/activation_policy.root_active_v1.json","sha256":"eb3408004a508136aa434c0a2e02a9d14a5598a96b8483c264c93026f9326018"}
FIXED = {
 "guard":GUARD, "policy":POLICY,
 "repair":{"path":"research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json","sha256":"f6f2d628199e8be73db8b5d90d428570d8afdefeab9dc5cd94bb5eafb39247db"},
 "renewal":{"path":"research/reviews/H039_marker_repair_renewal_source_20261016_v1/marker_repair_renewal_v1.py","sha256":"491236877e8d71cae8bd7619880858ac501c57dd93e59479ca32dacacab20468"},
 "renewal_certificate":{"path":"research/reviews/H039_marker_repair_renewal_source_20261016_v1/renewal_certificate.root_active_v1.json","sha256":"66006202094d7a3ef2b29cbec28eb93f334505b72589f09d86e4b57498162db7"},
 "memo":{"path":"research/reviews/resource_guard_hotpath_source_audit_v1/transaction_memo_adapter_v2.py","sha256":"c972e9a779d424ceafdb258848614bf40994d1004e2c73fc5cd225ae99621364"},
 "v4":{"path":"src/arggnn/project_budget_v4.py","sha256":"a093c7d86f57fee569cb99f4712250e9505a1119f3950fa44282c7ba8c57df47"},
 "monitor":{"path":"research/reviews/FLARE_existing_JDK_source_v1/root_build_driver.py","sha256":"d7afaf5c6c6976a5c4804fe41d320f8609fd346d3f4ed0bca29235001d7fe561"}
}
OUTPUT_CAP = 105 * 2**20
RSS_CAP = 64 * 2**30
DEADLINE = 86400
AUX_CAP = 16 * 2**20
LOG_CAP = 2 * 2**20
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")

def require(ok, message):
    if not ok: raise ValueError(message)

def path(name):
    require(isinstance(name,str) and name and not Path(name).is_absolute() and ".." not in Path(name).parts,
        "Contained relative project pathname")
    p=ROOT/name
    require(p!=ROOT and p.resolve()==p and p.is_relative_to(ROOT) and not p.is_symlink(),
        "Canonical non-symlink in-project pathname")
    return p

def raw(record, cap=32*2**20):
    require(isinstance(record,dict) and set(record)=={"path","sha256"}
        and isinstance(record["sha256"],str) and re.fullmatch("[0-9a-f]{64}",record["sha256"]),
        "Exact relative pathname/SHA256")
    p=path(record["path"]);before=p.lstat()
    signature=lambda s:(s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    require(stat.S_ISREG(before.st_mode) and before.st_nlink==1 and before.st_size<=cap,
        "Bounded singly linked regular immutable input")
    with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW),"rb") as f:
        require(signature(os.fstat(f.fileno()))==signature(before),"Unchanged opened identity")
        data=f.read(cap+1)
        require(signature(os.fstat(f.fileno()))==signature(before),"Unchanged read identity")
    require(len(data)<=cap and signature(p.lstat())==signature(before)
        and hashlib.sha256(data).hexdigest()==record["sha256"],"Exact immutable bytes")
    return data

def read(record):
    require(Path(record["path"]).suffix==".json","Metadata JSON only")
    return json.loads(raw(record))

def ref(name):
    p=path(name);require(p.is_file() and p.stat().st_size<=32*2**20,"Bounded source/metadata only")
    return {"path":name,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}

def unique(rows):
    out={}
    for r in rows:
        require(set(r)=={"path","sha256"} and Path(r["path"]).suffix in (".py",".sh",".json"),
            "Metadata/source-only original reservation")
        require(r["path"] not in out or out[r["path"]]==r,"No contradictory pins")
        out[r["path"]]=r
    return list(out.values())

def load(record,label):
    code=raw(record)
    spec=importlib.util.spec_from_file_location(label,path(record["path"]))
    m=importlib.util.module_from_spec(spec);sys.modules[label]=m
    exec(compile(code,str(path(record["path"])),"exec"),m.__dict__)
    return m

PREVIOUS_METADATA_REF={"path":"research/reviews/H050_CPU_evaluation_successor_source_20261016_v1/metadata_v2.py","sha256":"27e9054ccf65d57dffd2c5cfd68b52c355dbcee3aac697bfe3f1bb72cc4c41de"}
PREVIOUS_RUNTIME_REF={"path":"research/reviews/H050_CPU_evaluation_successor_source_20261016_v1/runtime_v2.py","sha256":"a271c83c50ea8651a71f8de609897a246642a82d013c794f1d033fcccfe3bbe6"}
ORIGINAL_TERMINAL_REF={"path":"research/reviews/H050_GPU0_shared_eval_lineage_source_20261016_v1/shared_terminal_v1.py","sha256":"b20169f3244ff973e6d4db36a54004e0235b6c112a458b93ace56a94deec0b79"}
ROOT_OUTCOME_REF={"path":"research/reviews/H050_full_budget_failed_CLI_root_20261016_v1/actual_outcome.root_v1.json","sha256":"554e56ed22ee96910128b36580177e4c355ea51e0ad65115f4a5e3d8f6875819"}
QUALIFICATION_REF={"path":"research/reviews/H050_one_FP32_CPU_forward_qualification_source_20261016_v1/qualification_receipt.root_passed_v1.json","sha256":"0a87bbefa3a17596b0b5f579004cc5ce09dca6d0d24558f5d11fd4671c739e48"}

def failed_cli_terminal():
    return load({"path":TERMINAL,"sha256":TERMINAL_SHA},"_H050_exact_failed_CLI_terminal_only")

def gpu_metadata():
    """Private operational namespace, no modification of original success validator."""
    original=load({"path":GPU_META,"sha256":GPU_META_SHA},"_H050_original_lineage_CPU_failed_CLI")
    namespace=dict(original.__dict__);namespace["terminal"]=failed_cli_terminal
    source=raw({"path":GPU_META,"sha256":GPU_META_SHA}).decode()
    names={"_validate_shared_train_v1","metadata_namespace_v1","metadata_train_lineage_v1"}
    nodes=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes}==names,"Only original scientific-lineage operational glue")
    exec(compile(ast.Module(body=nodes,type_ignores=[]),"<unchanged_H050_scientific_lineage_explicit_failed_CLI_provenance>","exec"),namespace)
    from types import SimpleNamespace
    result=SimpleNamespace(**original.__dict__)
    result.terminal=failed_cli_terminal
    result.metadata_namespace_v1=namespace["metadata_namespace_v1"]
    result.metadata_train_lineage_v1=namespace["metadata_train_lineage_v1"]
    return result


def immutable_scientific_config(cfg):
    require(ref(TRAIN_CONFIG)=={"path":TRAIN_CONFIG,"sha256":TRAIN_CONFIG_SHA},"Exact original TRAIN config")
    old=read(ref(TRAIN_CONFIG));expected=copy.deepcopy(old);expected["output_cap_bytes"]=OUTPUT_CAP
    require(cfg==expected,"Exact TRAIN science/provenance/caps, only original EVAL105MiB output cap differs")
    return cfg

def count_plan(cfg):
    g=gpu_metadata();immutable_scientific_config(cfg)
    plan=g.pure_count_plan_v1(cfg)
    # Same complete eight-parent plan, all24 checkpoint reads and192 seals.
    require(plan["parent_count"]==8 and plan["checkpoint_files_written"]==0
        and plan["VAL_probability_seals_before_truth"]==192
        and plan["all192_native_hold_probability_bytes"]<=cfg["RAM_prediction_cap_bytes"]
        and plan["total_output_scope_upper_bound_bytes"]<=OUTPUT_CAP,
        "Unchanged complete EVAL count-derived RAM/storage plan")
    working=(cfg["CPU_bank_cap_bytes"]+cfg["episode_cache_cap_bytes"]
        +cfg["bundle_cache_cap_bytes"]+plan["all192_native_hold_probability_bytes"]
        +cfg["grid_preparation_cap_bytes"]+24*cfg["checkpoint_file_cap_bytes"])
    require(working<RSS_CAP,"Finite retained tensors and original preparation allowance fit original64GiB envelope")
    return dict(original_count_plan=plan,CPU_count_planned_bytes=working,
        peak_RSS_and_wall_runtime_unqualified=True,max_host_RSS_bytes=RSS_CAP,
        output_cap_bytes=OUTPUT_CAP,auxiliary_cap_bytes=AUX_CAP,log_cap_bytes=LOG_CAP,
        actual_GPU_cap_bytes=0,no_GPU_lease_required=True)

def payload_plan(cfg,spec):
    """Read JSON provenance only. Never stat/open the named scientific payloads."""
    original=read(cfg["original_H038_input_ref"]);targets=read(original["refs"]["target_bindings"])
    cache=read(spec["refs"]["observation_cache"]);cut=read(cfg["cut_preparation_completion_ref"])
    train=read(spec["train_completion_ref"]);protocol=cfg["scientific_protocol"]
    val=protocol["VAL_parent_ids"];arms=protocol["arms"];passes=protocol["checkpoint_passes"]
    require(len(val)==8 and len(arms)==4 and len(passes)==6 and set(train["checkpoint_groups"])=={str(v) for v in passes},
        "All original eight VAL parents and24 scheduled checkpoints")
    rows=[]
    for pid in val:
        o=cache["parents"][pid];c=cut["parents"][pid]
        require(o["array_ref"]==read(cfg["preparation_completion_ref"])["parents"][pid]["original_cache_array_ref"]
            and c["lossless_roundtrip_verified"] is True,"Original same-parent cache and corrected cut")
        truth_binding=targets["parents"][pid]
        truth_provenance=read(truth_binding["generation_receipt"])["files"]["truth/full_truth_intervals.npz"]
        require(truth_provenance["sha256"]==truth_binding["truth_npz"]["sha256"],
            "Original generation metadata binds exact truth byte cap without reading truth")
        for role,r,cap in (("observed_cache",o["array_ref"],o["compressed_bytes"]),
            ("cut_recipe",c["arrays_ref"],c["compressed_archive_bytes"]),
            ("VAL_truth",truth_binding["truth_npz"],truth_provenance["bytes"])):
            require(type(cap) is int and 0<cap<=8_000_000_000,"Original finite immutable reader limit")
            rows.append(dict(role=role,parent_id=pid,ref=r,max_file_bytes=cap))
    for at in passes:
        require(set(train["checkpoint_groups"][str(at)])==set(arms),"Exactly all four arms each scheduled group")
        for arm in arms:
            rows.append(dict(role="scheduled_checkpoint",arm=arm,pass_index=at,
                ref=train["checkpoint_groups"][str(at)][arm],max_file_bytes=cfg["checkpoint_file_cap_bytes"]))
    require(len(rows)==48 and len({r["ref"]["path"] for r in rows})==48,"Exactly48 distinct scientific payload roles")
    return dict(schema="H050_CPU_EVAL_exact_payload_plan_v1",rows=rows,
        checkpoint_payloads_read_during_binding=False,truth_payloads_read_during_binding=False,
        TRAIN_payloads_permitted=False,all_eight_VAL=True)

def validate_qualification(record):
    require(record==QUALIFICATION_REF,"Reuse exact genuine existing CPU FP32 receipt, never replay")
    original=load(PREVIOUS_METADATA_REF,"_H050_qualified_original_CPU_metadata")
    qualification=original.validate_qualification(record)
    old=ast.parse(raw(PREVIOUS_RUNTIME_REF))
    new=ast.parse(raw(ref(RUNTIME)))
    # Runtime has only new file routing pins. Every function and all scientific,
    # monitor, timeout, payload, writer and numerical execution AST stays equal.
    routes={"HERE","SOURCE","METADATA","METADATA_SHA"}
    def frozen(tree):
        return ast.dump(ast.Module(body=[n for n in tree.body if not (
            isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in routes for t in n.targets))],
            type_ignores=[]),include_attributes=False)
    require(frozen(old)==frozen(new),"Exact already-qualified CPU execution AST, only metadata routing constants differ")
    return qualification


def required_bindings(cfg,spec):
    # Original direct refs, same three read-only JSON transitivities and same
    # exact scientific warm-checkpoint JSON proxy. No payload is a direct ref.
    g=gpu_metadata();ns=g.metadata_namespace_v1(cfg)
    src=g.exact(g.COMMON)
    node=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=="direct_required_refs_v1")
    exec(compile(ast.Module(body=[node],type_ignores=[]),"<unchanged_H050_direct_metadata_refs>","exec"),ns)
    rows=ns["direct_required_refs_v1"](cfg,dict(spec,phase="train"))
    indirect=[cfg[k] for k in g.INDIRECT_JSON_FIELDS]
    require(all(r in rows and Path(r["path"]).suffix==".json" for r in indirect),"Only same three indirect JSON roles")
    rows=[r for r in rows if r not in indirect]
    proxy=g.load(g.METADATA,"_H050_original_warm_metadata_proxy_CPU")
    rows=proxy.source_metadata_refs_v1(rows,cfg)
    rows=unique(rows+[ref(SOURCE),ref(RUNTIME),ref(PRODUCER),
        {"path":GPU_META,"sha256":GPU_META_SHA},{"path":TERMINAL,"sha256":TERMINAL_SHA},
        spec["CPU_caps_ref"],spec["CPU_payload_plan_ref"],spec["CPU_closure_ref"],
        *FIXED.values(),*[ref(p) for p in g.BUDGET_PATHS],PREVIOUS_METADATA_REF,PREVIOUS_RUNTIME_REF,
        ORIGINAL_TERMINAL_REF,ROOT_OUTCOME_REF,spec["train_actual_wait_ref"]])
    if spec.get("CPU_FP32_qualification_ref") is not None:rows=unique(rows+[spec["CPU_FP32_qualification_ref"]])
    require(len(rows)<=64,"Complete original CPU source/metadata union at most64 direct bindings")
    return rows

def closure_for(cfg,spec,request_ref):
    return dict(schema="H050_CPU_EVAL_exact_closure_v1",root_authorizer="root_agent",
        request_ref=request_ref,CPU_runtime_ref=ref(RUNTIME),CPU_metadata_ref=ref(SOURCE),
        original_producer_ref=ref(PRODUCER),original_context_ref=ref(CONTEXT),
        original_scientific_source_map=gpu_metadata().inherited_sources_v1(gpu_metadata().MetadataReaderV1(),cfg),
        original_config_ref=spec["refs"]["config"],runtime_ref=spec["refs"]["runtime"],
        exact_phase_refs={k:v for k,v in spec["refs"].items() if k!="storage"},
        CPU_caps_ref=spec["CPU_caps_ref"],CPU_payload_plan_ref=spec["CPU_payload_plan_ref"],
        CPU_FP32_qualification_ref=spec.get("CPU_FP32_qualification_ref"),
        shared_train_terminal_certificate_ref=spec["train_actual_wait_ref"],
        actual_TRAIN_leader_exit_code=1,actual_TRAIN_client_exit_code=1,
        TRAIN_full_scientific_budget_complete=True,TRAIN_completed_success_storage_credit=False,
        actual_failed_CLI_root_observation_ref=ROOT_OUTCOME_REF,
        train_completion_ref=spec["train_completion_ref"],train_launch_ref=spec["train_launch_ref"],
        train_io_lineage=spec["train_io_lineage"],fixed_refs=FIXED,
        exact_indirect_JSON_refs={k:cfg[k] for k in gpu_metadata().INDIRECT_JSON_FIELDS},
        no_GPU_pool_retirement_dependency=True,unchanged_all4arms6checkpoints8VAL192barrier=True,
        original_count_plan=count_plan(cfg),CPU_actual_device="cpu",actual_cuda_cap_bytes=0)

def validate_spec(spec,storage=None,require_qualification=True):
    require(spec.get("schema")=="H050_CPU_EVAL_root_input_v1" and spec.get("status")=="active"
        and spec.get("root_authorizer")=="root_agent" and spec.get("hypothesis_id")=="H050"
        and spec.get("phase")=="evaluate" and spec.get("arm")=="joint_four_arm"
        and spec.get("execution_authorized") is True and spec.get("optimization_authorized") is False
        and spec.get("GPU_authorized") is False and spec.get("device")=="cpu"
        and spec.get("physical_gpu") is None and spec.get("max_cuda_reserved_bytes")==0
        and spec.get("max_cpu_rss_bytes")==RSS_CAP and spec.get("runtime_entry_ref")==ref(RUNTIME)
        and all(spec.get(k)==ref(RUNTIME) for k in ("main_source_ref","worker_cli_ref",
            "root_context_module_ref","root_driver_ref","canonical_dispatcher_ref"))
        and spec.get("producer_ref")==ref(PRODUCER),
        "Exact H050 CPU-only EVALUATE authority, zero optimizer updates")
    require(spec.get("TRAIN_actual_leader_exit_code")==spec.get("TRAIN_actual_tool_exit_code")==1
        and spec.get("H050_completed_scope_credit_authorized") is False,
        "No failed TRAIN CLI outcome may be relabelled success0 or storage credit")
    issued=datetime.fromisoformat(spec["created_at_utc"]);end=datetime.fromisoformat(spec["expires_at_utc"])
    require(issued.tzinfo is not None and end.tzinfo is not None
        and issued<=datetime.now(timezone.utc)<end and 0<(end-issued).total_seconds()<=DEADLINE,
        "Fresh absolute CPU authority, at most24 hours")
    renewal=read(FIXED["renewal_certificate"])
    renewal_end=datetime.fromisoformat(renewal["renewed_expires_at_utc"].replace("Z","+00:00"))
    require(end<renewal_end,"Entire CPU authority and final guard stay inside original authentic renewal")
    require(NAME.fullmatch(spec["job_id"]) and spec["job_id"]!=read(ref(TRAIN_INPUT))["job_id"]
        and spec["output_directory"].startswith("experiments/results/H050_CPU_")
        and path(spec["output_directory"]).parent.is_dir(),"Distinct finite H050 CPU output identity")
    cfg=immutable_scientific_config(read(spec["refs"]["config"]))
    old=read({"path":TRAIN_INPUT,"sha256":TRAIN_INPUT_SHA})
    require(all(spec["refs"][k]==old["refs"][k] for k in ("registration","generation_config","observation_cache"))
        and spec["source_sha256"]==old["source_sha256"]
        and spec["observation_entry_refs"]==old["observation_entry_refs"]
        and spec["observation_cache_source_refs"]==old["observation_cache_source_refs"],
        "Same original40 observed entries/cache/sources and frozen provenance")
    g=gpu_metadata()
    require(spec["train_io_lineage"]==g.metadata_train_lineage_v1(spec,cfg),"Original full3840-update TRAIN proof")
    require(read(spec["CPU_payload_plan_ref"])==payload_plan(cfg,spec),"Exact all48 selected payload authorities")
    require(read(spec["CPU_closure_ref"])==closure_for(cfg,spec,spec["CPU_root_request_ref"]),
        "Direct CPU closure authenticates every frozen science/runtime/lineage ref")
    caps=read(spec["CPU_caps_ref"])
    require(caps==dict(schema="H050_CPU_EVAL_caps_v1",max_host_RSS_bytes=RSS_CAP,
        wall_deadline_seconds=DEADLINE,output_cap_bytes=OUTPUT_CAP,auxiliary_cap_bytes=AUX_CAP,
        log_cap_bytes=LOG_CAP,CPU_threads=spec["CPU_threads"],actual_cuda_cap_bytes=0)
        and type(spec["CPU_threads"]) is int and 1<=spec["CPU_threads"]<=32,"Explicit original finite CPU envelope")
    runtime=read(spec["refs"]["runtime"])
    require(runtime==dict(schema="H050_CPU_EVAL_runtime_v1",device="cpu",physical_gpu=None,
        job_id=spec["job_id"],output_directory=spec["output_directory"],created_at_utc=spec["created_at_utc"],
        expires_at_utc=spec["expires_at_utc"],executable=str(ROOT/".runtime/env/bin/python"),
        process_title="AI Auto Research Pipeline Test",CPU_threads=spec["CPU_threads"],CUDA_VISIBLE_DEVICES=""),
        "Exact CPU process/runtime identity")
    if require_qualification:validate_qualification(spec["CPU_FP32_qualification_ref"])
    if storage is not None:
        own=storage["reservation"];required=required_bindings(cfg,spec)
        require(storage.get("kind")=="project_budget_v9_storage_record" and storage.get("active") is True
            and storage.get("consumer_uses_v9_guard") is True and storage.get("accounting_mode")=="prospective_980gb_v1"
            and storage["job_id"]==spec["job_id"] and storage["output_directory"]==spec["output_directory"]
            and storage["expires_at_utc"]==spec["expires_at_utc"] and storage["reserved_new_bytes"]==OUTPUT_CAP
            and storage["completed_overlay_policy_ref"]==POLICY and own["reserved_bytes"]==OUTPUT_CAP
            and own["output"]==str(path(spec["output_directory"]))
            and own["auxiliary_scopes"]==[str(path(spec["worker_auxiliary_scope"]))]
            and len(own["consumer_bindings"])==len(required)<=64
            and all(r in own["consumer_bindings"] for r in required),
            "Genuine unchanged prospective guard record, exact105MiB complete charged output+auxiliary and source union")
    return cfg

def save(folder,name,value):
    p=path(folder+"/"+name);require(not p.exists(),"Fresh metadata only, preserve existing artifacts")
    raw_bytes=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    require(len(raw_bytes)<=32*2**20,"Bounded metadata serialization")
    with os.fdopen(os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),"wb") as f:
        f.write(raw_bytes);f.flush();os.fsync(f.fileno())
    return ref(folder+"/"+name)

def prepare_metadata(request_ref,folder):
    """Root metadata only, after genuine failed CLI with authenticated full science. No admission or science."""
    req=read(request_ref)
    require(req.get("schema")=="H050_CPU_EVAL_metadata_request_v1" and req.get("status")=="active"
        and req.get("root_authorizer")=="root_agent" and req.get("output_cap_bytes")==OUTPUT_CAP
        and req.get("GPU_authorized") is False and req.get("optimization_authorized") is False,
        "Explicit root105MiB CPU EVALUATE metadata request")
    require(req.get("TRAIN_actual_leader_exit_code")==1 and req.get("TRAIN_actual_tool_exit_code")==1
        and req.get("H050_completed_scope_credit_authorized") is False,
        "Root explicitly acknowledges genuine failed CLI, no H050 success0 storage credit")
    g=gpu_metadata();proof=g.terminal().validate_certificate(req["train_shared_terminal_certificate_ref"])
    require(proof["certificate"]["input_ref"]=={"path":TRAIN_INPUT,"sha256":TRAIN_INPUT_SHA},
        "Only actual H050 original GPU0 full TRAIN logical terminal")
    old=proof["input"];cfg=copy.deepcopy(read(old["refs"]["config"]));cfg["output_cap_bytes"]=OUTPUT_CAP
    immutable_scientific_config(cfg);count_plan(cfg)
    target=path(folder);require(not target.exists() and target.parent.is_dir(),"Fresh CPU metadata directory")
    require(NAME.fullmatch(req["job_id"]) and req["output_directory"].startswith("experiments/results/H050_CPU_")
        and not path(req["output_directory"]).exists() and not path(folder+"/worker_aux").exists(),
        "Fresh science output and auxiliary, no replay")
    spec=copy.deepcopy(old)
    spec.update(schema="H050_CPU_EVAL_root_input_v1",phase="evaluate",device="cpu",physical_gpu=None,
        max_cuda_reserved_bytes=0,max_cpu_rss_bytes=RSS_CAP,job_id=req["job_id"],output_directory=req["output_directory"],
        created_at_utc=req["created_at_utc"],expires_at_utc=req["expires_at_utc"],
        worker_auxiliary_scope=folder+"/worker_aux",GPU_authorized=False,optimization_authorized=False,
        execution_authorized=True,CPU_threads=req["CPU_threads"],runtime_entry_ref=ref(RUNTIME),
        main_source_ref=ref(RUNTIME),worker_cli_ref=ref(RUNTIME),root_driver_ref=ref(RUNTIME),
        root_context_module_ref=ref(RUNTIME),canonical_dispatcher_ref=ref(RUNTIME),
        train_completion_ref=proof["certificate"]["completed_ref"],
        train_launch_ref=proof["certificate"]["capability_ref"],
        train_actual_wait_ref=req["train_shared_terminal_certificate_ref"],
        CPU_FP32_qualification_ref=req.get("CPU_FP32_qualification_ref"),CPU_root_request_ref=request_ref,
        TRAIN_actual_leader_exit_code=1,TRAIN_actual_tool_exit_code=1,H050_completed_scope_credit_authorized=False)
    spec["refs"].pop("storage",None)
    spec["train_io_lineage"]=g.metadata_train_lineage_v1(spec,cfg)
    # Historical CUDA input operational fields cannot masquerade as CPU routing.
    spec["inherited_GPU_TRAIN_operational_annotations"]={k:spec.pop(k) for k in list(spec)
        if k.startswith("operational_") or k in ("execution_gpu_uuid","root_host_resource_enforcement_active")}
    original=read(cfg["original_H038_input_ref"]);targets=read(original["refs"]["target_bindings"])
    val=cfg["scientific_protocol"]["VAL_parent_ids"]
    target.mkdir()
    spec["refs"]["config"]=save(folder,"scientific_config.json",cfg)
    spec["refs"]["runtime"]=save(folder,"runtime.json",dict(schema="H050_CPU_EVAL_runtime_v1",device="cpu",
        physical_gpu=None,job_id=spec["job_id"],output_directory=spec["output_directory"],
        created_at_utc=spec["created_at_utc"],expires_at_utc=spec["expires_at_utc"],
        executable=str(ROOT/".runtime/env/bin/python"),process_title="AI Auto Research Pipeline Test",
        CPU_threads=spec["CPU_threads"],CUDA_VISIBLE_DEVICES=""))
    spec["refs"]["target_bindings"]=save(folder,"VAL_target_bindings.json",dict(
        schema="H050_original_phase_target_bindings_v1",phase="evaluate",outer_included=False,
        TRAIN_included=False,VAL_included=True,parents={p:targets["parents"][p] for p in val}))
    spec["CPU_caps_ref"]=save(folder,"caps.json",dict(schema="H050_CPU_EVAL_caps_v1",max_host_RSS_bytes=RSS_CAP,
        wall_deadline_seconds=DEADLINE,output_cap_bytes=OUTPUT_CAP,auxiliary_cap_bytes=AUX_CAP,
        log_cap_bytes=LOG_CAP,CPU_threads=spec["CPU_threads"],actual_cuda_cap_bytes=0))
    spec["CPU_payload_plan_ref"]=save(folder,"payload_plan.json",payload_plan(cfg,spec))
    spec["CPU_closure_ref"]=save(folder,"closure.json",closure_for(cfg,spec,request_ref))
    spec["consumer_manifest_ref"]=spec["CPU_closure_ref"]
    validate_spec(spec,require_qualification=False)
    rows=required_bindings(cfg,spec)
    return dict(schema="H050_CPU_EVAL_metadata_ready_not_admitted_v1",
        draft_ref=save(folder,"input_draft_without_storage.json",spec),consumer_bindings=rows,
        output_cap_bytes=OUTPUT_CAP,worker_auxiliary_scope=spec["worker_auxiliary_scope"],
        output_plan=count_plan(cfg),qualification_required_before_finalize=True,
        no_admission_guard_or_scientific_execution=True)

def finalize_input(draft_ref,storage_ref,folder):
    """Bind genuine original admission only. Never reserve, extend or launch."""
    spec=read(draft_ref);require("storage" not in spec["refs"],"One fresh draft without storage")
    spec["refs"]["storage"]=storage_ref;validate_spec(spec,read(storage_ref),require_qualification=True)
    return save(folder,"input.json",spec)
