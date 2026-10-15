"""Specific default-inert root contracts; no guard/state/runtime work at import."""
from pathlib import Path
from datetime import datetime,timezone
import copy,hashlib,importlib.util,json,os,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/C3_native_postMLM980_root_source_20261015_v1"
SOURCE=REL+"/root_native_contract_v1.py"
DRIVER=REL+"/root_native_driver_v1.py"
CONTEXT=REL+"/context_factory_v1.py"
RUNTIME_HELPER=REL+"/native_runtime_v1.py"
PARENT="american_c3_train_000"
GPU_PHASES=("fit","embeddings")
CPU_PHASES=()
PHASES=GPU_PHASES
GPU_MAIN=REL+"/entry_postMLM_v1.py"
CPU_MAIN=None
CORE="483d78bf2ef3c917d65ecf1dde5a7809bfadc21194c9f9a90394ce5c532fdd5e"
BASE="a8049bc103afb1cbbbce6f8fc3707a21de9613b23034599641b688994fc5caa3"
GUARD_REF={'path': 'research/reviews/prospective_980GB_capacity_source_v1/prospective_980gb_v1.py', 'sha256': 'fc71030865e0a3d5da31120d4daa2caa0b2d39ec7dd212b8c46c476b5a866d56'}
POLICY_REF={'path': 'research/reviews/prospective_980GB_capacity_source_v1/activation_policy.root_active_v1.json', 'sha256': 'eb3408004a508136aa434c0a2e02a9d14a5598a96b8483c264c93026f9326018'}
REPAIR_REF=dict(path="research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json",sha256="f6f2d628199e8be73db8b5d90d428570d8afdefeab9dc5cd94bb5eafb39247db")
MEMO_REF=dict(path="research/reviews/resource_guard_hotpath_source_audit_v1/transaction_memo_adapter_v2.py",sha256="c972e9a779d424ceafdb258848614bf40994d1004e2c73fc5cd225ae99621364")
V4_REF=dict(path="src/arggnn/project_budget_v4.py",sha256="a093c7d86f57fee569cb99f4712250e9505a1119f3950fa44282c7ba8c57df47")
LAUNCHER_REF=dict(path=REL+"/launcher_postMLM980_v1.py",sha256=None)
GPU_AUTH_REF=dict(path="research/reviews/H040_GPU45_migration_root_20261015_v1/GPU45_authorization.json",sha256="0231ae8edb05f8dd34ce127a0de0b97c1dd64fa461ad081e7a83bb92c558d26c")
BASE_CONFIG_REF=dict(path="research/reviews/C3_bert_base_configuration_root_v1/config.json",sha256="7160e1553ad2ca51d8c1cb066be533db31826e12d173824c1bb0cb1a4f187d20")
NATIVE_CPU_QUAL_REF=dict(path="research/reviews/C3_bert_base_configuration_root_v1/native_CPU_qualification.json",sha256="56ed247a09484753a64e9ed3bee6c80eea4e5bfb156ab5acdb09c24ce828e515")
NATIVE_INVENTORY_REF=dict(path="experiments/results/native_argformer_tsinfer_fit_v7_root/environment_inventory.json",sha256="077d3e950c3bb3efdecc7292e06c8157506663293b4efe4723ccf7b877973461")
LEGACY_RUNTIME_REF=dict(path="src/arggnn/native_argformer_tsinfer_fit_v7.py",sha256="b94a4e16a6a9dc230256974b9a0e45a6edd1b41b9ecc8ac4171c4461ec514712")
PROC_HELPER="research/reviews/American_C3_native_paths_root_v1/root_native_paths_driver.py"
OBS_REF=dict(path="experiments/results/American_C3_observed_full30mb_supervisor_repair_20261015_v1/observed/manifest.json",sha256="c3d58a8d10d901a162f406e2e218a10e17e5f1617b3c626503d72bcbd1b4ccee")
PREP_OUTPUT="experiments/results/American_C3_native_paths_train000_full30mb_seed92001_v3"
PREP_REF={'path': 'experiments/results/American_C3_native_paths_train000_full30mb_seed92001_v3/completed.json', 'sha256': '3cc49fbf55bd4b4b3859a93ce1123e9218f407f7c7299ad5029be81f59c7c6fa'}
PREP_CONFIG_REF={'path': 'research/reviews/American_C3_native_paths_root_v3/admitted_native_paths_train000_full30mb_v3/config.json', 'sha256': '9dfd5eb24274933816e9b5af122c1279aa132bc37afa2e7e3563aea04201e01c'}
PREP_CLOSE_REF={'path': 'research/reviews/American_C3_native_paths_root_v3/completed_scope_close_root_v1/close_receipt.json', 'sha256': '46d972e3ea36a41a147b81fa310b739b487efc1d2808d72594a89ec0f854b533'}
PREP_WAIT_REF={'path': 'research/reviews/American_C3_native_paths_root_v3/admitted_native_paths_train000_full30mb_v3/actual_wait.json', 'sha256': 'd4f8fd549a8b3610d21b9cebbf4e7690fd52dc95d24e11253aea29860520b336'}
COUNTS_REF={'path': 'experiments/results/American_C3_native_train000_full30mb_counts980_v1/completed.json', 'sha256': 'acbb95120d554532205a7e2647861c87be9a5de063df30c1d85bd702a8b28cb9'}
COUNTS_WAIT_REF={'path': 'research/reviews/C3_native_counts980_root_handoff_20261015_v1/admitted_counts980_v1/actual_wait.json', 'sha256': '6b2cad31b77b9351fdfeff0e21bea52051bb974d54f792cbde668df71ced3deb'}
CONTRASTIVE_ROWS={"train":8624912,"val":959065}
CONTRASTIVE_UPDATES=2105
CONTRASTIVE_DROPPED_REMAINDER=2832
def require(ok,message):
    if not ok:raise ValueError(message)
def local(value):
    p=Path(value);p=p if p.is_absolute() else ROOT/p
    require(p!=ROOT and p.resolve()==p and ".." not in p.parts and p.is_relative_to(ROOT),"Canonical project path")
    return p
def ref(value):
    p=local(value);h=hashlib.sha256()
    with p.open("rb") as stream:
        for b in iter(lambda:stream.read(8388608),b""):h.update(b)
    return dict(path=str(p.relative_to(ROOT)),sha256=h.hexdigest())
def read(record,cap=1048576):
    require(isinstance(record,dict) and set(record)=={"path","sha256"},"Exact actual path/SHA ref")
    p=local(record["path"]);require(p.stat().st_size<=cap and ref(p)==record,"Bounded exact metadata SHA")
    return json.loads(p.read_text())
def save(directory,name,value):
    p=local(directory/name);raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    require(len(raw)<=65536,"Root authority/evidence file <=64KiB")
    with p.open("xb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return ref(p)
def load(record,name):
    require(ref(record["path"])==record,"Exact helper source")
    spec=importlib.util.spec_from_file_location(name,local(record["path"]))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def active(spec):
    now=datetime.now(timezone.utc)
    start=datetime.fromisoformat(spec["created_at_utc"]);end=datetime.fromisoformat(spec["expires_at_utc"])
    require(start.tzinfo and end.tzinfo and start<=now<end and (end-start).total_seconds()<=86400,
        "Actual unexpired root capability <=24hours")
def original_guard():
    # Only future root actions call this. Never install memo2 in this process.
    for record in (GUARD_REF,POLICY_REF,REPAIR_REF,V4_REF):require(ref(record["path"])==record,"Fixed current authority source")
    sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"scripts")]
    guard=load(GUARD_REF,"_C3_native_counts_current980")
    guard.configure_v2(REPAIR_REF,policy_ref=POLICY_REF)
    require(all(getattr(guard.budget.v4,n).__module__==guard.budget.v4.__name__
        for n in ("canonical","file_bytes","mutex")),"Root reserve/close must use unwrapped originalv4")
    return guard
def consumer_guard():
    guard=original_guard();memo=load(MEMO_REF,"_C3_native_consumer_memo2")
    adapter=memo.install(guard.budget.v4,expected_source_path=ROOT/V4_REF["path"],
        expected_source_sha256=V4_REF["sha256"])
    return guard,adapter
def phase_paths(phase):
    require(phase in PHASES,"Specific registered native stage")
    return (ROOT/REL/("admitted_"+phase+"_v1"),
        "experiments/results/American_C3_native_train000_postMLM980_"+phase+"_v1")
def validate_upstream(completed_ref,wait_ref,*,preparation=False):
    completed=read(completed_ref);wait=read(wait_ref)
    if preparation:
        require(wait.get("actual_direct_worker_wait") is True and wait.get("owned_worker_group_ESRCH") is True
            and wait.get("operational_failure") is None and wait.get("cleanup_errors")==[]
            and completed.get("status")=="completed_native_C3_path_preparation"
            and completed.get("parent_id")==PARENT and completed.get("observation_ref")==OBS_REF
            and completed_ref["path"]==PREP_OUTPUT+"/completed.json","Actual original complete native corpus/wait/drain")
    else:
        require(wait.get("actual_direct_launcher_wait") is True and wait.get("all_owned_processes_and_groups_drained") is True
            and wait.get("operational_failure") is None,"Actual completed native root stage wait/drain")
    require(wait.get("returncode")==0 and wait.get("completed_ref")==completed_ref,"Exact actual completion/directwait0")
    return completed,wait
def validate_caps(caps,request):
    phase=request["phase"];refs=request["refs"]
    require(caps.get("schema")=="american_c3_native_actual_stage_caps_v1"
        and caps.get("root_authorizer")=="root_agent" and caps.get("active") is True
        and caps.get("parent_id")==PARENT and caps.get("phase")==phase
        and caps.get("runtime_ref")==refs["runtime"] and caps.get("preparation_ref")==refs["preparation"]
        and caps.get("counts_ref")==refs.get("counts") and caps.get("driver_source_ref")==ref(DRIVER),
        "Root must bind actual same-stage corpus/counts/runtime/caps and exact driver")
    for key in ("reservation_cap_bytes","phase_write_budget_bytes","max_cpu_rss_bytes",
        "operational_deadline_seconds","max_preparation_file_bytes","max_checkpoint_files",
        "max_checkpoint_bytes","max_embedding_input_bytes","outer_log_cap_bytes","outer_auxiliary_cap_bytes",
        "root_packet_overhead_cap_bytes","child_log_ram_bytes"):
        require(type(caps.get(key)) is int and caps[key]>0,"Root must fill positive actual source/count-derived bounds")
    require(caps["phase_write_budget_bytes"]+caps["outer_auxiliary_cap_bytes"]+caps["root_packet_overhead_cap_bytes"]<=caps["reservation_cap_bytes"]
        and caps["root_packet_overhead_cap_bytes"]>=12*(65536+8192)
        and caps["outer_auxiliary_cap_bytes"]>=6*caps["outer_log_cap_bytes"]+65536
        and caps["operational_deadline_seconds"]<=86400
        and caps.get("bound_method")=="actual_source_and_corpus_counts_no_fit_or_timing_guarantee",
        "Whole phase/temp/log/metadata growth must fit actual reserved cap")
    if phase in GPU_PHASES:
        require(type(caps.get("max_cuda_reserved_bytes")) is int and 0<caps["max_cuda_reserved_bytes"]<=50000000000
            and request.get("physical_gpu") in (4,5) and caps.get("physical_gpu")==request["physical_gpu"],
            "Exact root actual CUDA cap and eligible device, no invented fit guarantee")
    return caps
def make_config(req,caps):
    DEFAULTS=native_contract().DEFAULTS
    cfg=copy.deepcopy(DEFAULTS);phase=req["phase"]
    cfg.update(execution_enabled=True,parent_id=PARENT,phase=phase,
        phase_write_budget_bytes=caps["phase_write_budget_bytes"],
        short_tmp_directory=".runtime/tmp/c3post980/"+("f" if phase=="fit" else "r"),
        child_log_ram_bytes=caps["child_log_ram_bytes"])
    for key in ("max_preparation_file_bytes","max_checkpoint_files","max_checkpoint_bytes","max_embedding_input_bytes"):
        cfg[key]=caps[key]
    return cfg

BRIDGE="research/reviews/C3_native_post_MLM_root_handoff_20261015_v1/post_mlm_bridge_v1.py"
RAM="research/reviews/C3_native_post_MLM_root_handoff_20261015_v1/ram_downstream_v1.py"
NVML_ACCEPTANCE_REF=dict(path="research/reviews/American_C3_native_MLM_full_fit_v8/nvml_abi_acceptance.root_active_v1.json",
 sha256="69b13084c5275629e194651b71a9db2b233f0ef924e6c12394b2971aa70e195b")
def native_contract():
    return load(ref(REL+"/consumer_contract_v1.py"),"_postMLM980_native_contract")
def source_paths():
    native=native_contract()
    return tuple(dict.fromkeys((*native.PROJECT_SOURCES,SOURCE,DRIVER,CONTEXT,RUNTIME_HELPER,
        "src/arggnn/census_tsinfer_fit_v1.py","src/arggnn/census_tsinfer_observation_v1.py",
        "src/arggnn/biological_candidate.py","src/arggnn/project_budget_v9_consumer.py",
        "scripts/launch_reserved_gpu_full_storage_v1.py","scripts/launch_reserved_gpu_terminal_retired_scopes_v1.py",
        PROC_HELPER,"research/reviews/American_C3_native_MLM_amendment_v12/owned_lifecycle_v2.py",
        "research/reviews/American_C3_native_MLM_full_source_v5/pidfd_backend_v1.py",
        "research/reviews/American_C3_native_MLM_amendment_v12/abstract_unix_bootstrap_v1.py",
        "research/reviews/American_C3_native_MLM_amendment_v12/terminal_namespace_v12.py",
        "research/reviews/American_C3_native_MLM_amendment_v12/terminal_bounds_v1.py",
        "research/reviews/American_C3_native_MLM_full_fit_v8/gpu_monitor_v8.py",
        "research/reviews/American_C3_native_MLM_full_fit_v8/progress_v8.py",
        "research/reviews/American_C3_native_MLM_full_fit_v8/nvml_abi_provenance.json",
        REL+"/launcher_postMLM980_v1.py","src/arggnn/resources.py","scripts/project_budget_v9.py")))
def counts_metadata_v1(record,wait_ref,refs):
    require(record==COUNTS_REF and wait_ref==COUNTS_WAIT_REF,"Exact completed current980 counts and actual root wait pins")
    value,wait=validate_upstream(record,wait_ref)
    native=native_contract();prep=read(PREP_REF)
    require(value.get("schema")=="american_c3_native_actual_counts_v1"
        and value.get("status")=="completed_C3_native_observation_only_counts"
        and value.get("source_ref")==ref("src/arggnn/american_c3_native_concat_v1.py")
        and value.get("preparation_ref")==PREP_REF and value.get("preparation_config_ref")==PREP_CONFIG_REF
        and value.get("parent_id")==PARENT and value.get("class_order")==["EUR","EAS","AFR"]
        and value.get("native_training_eligible") is True and value.get("query_ancestry_truth_opened") is False
        and value.get("query_focal_count")==10000 and wait.get("phase")=="counts",
        "Genuine exact generic six-dataset counts plus actual root wait0/drain")
    require(set(value["datasets"])==set(prep["native_dataset_refs"]),"All six native count datasets")
    for key,item in value["datasets"].items():
        require(item["dataset_ref"]==prep["native_dataset_refs"][key]
            and item["row_order_ref"]==prep["full_path_id_order_refs"][key]
            and type(item["rows"]) is int and item["rows"]>0,"Exact complete raw-path count identities")
    require(value["MLM_optimizer_updates"]==5*(value["datasets"]["mlm/train"]["rows"]//512)==224775
        and value["contrastive_optimizer_updates"]==value["datasets"]["contrastive/train"]["rows"]//4096==CONTRASTIVE_UPDATES
        and value["datasets"]["contrastive/train"]["rows"]%4096==CONTRASTIVE_DROPPED_REMAINDER
        and all(value["datasets"]["contrastive/"+sp]["rows"]==n for sp,n in CONTRASTIVE_ROWS.items())
        and sum(value["datasets"]["query_paths/"+sp]["rows"] for sp in ("train","val"))==10000
        and sum(value["datasets"]["contrastive/"+sp]["rows"] for sp in ("train","val"))==9583977,
        "Actual full-corpus release budgets, no metadata-estimated reference split")
    return value
def validate_fit_metadata_v1(record,wait_ref,*,counts_ref,mlm_ref):
    packet,output=phase_paths("fit")
    require(record["path"]==output+"/completed.json"
        and wait_ref["path"]==str((packet/"actual_wait.json").relative_to(ROOT)),
        "Actual own postMLM980 fit completion and terminal root wait paths")
    fit,wait=validate_upstream(record,wait_ref)
    require(fit.get("status")=="completed_C3_fresh_native_fit" and fit.get("MLM_reexecuted") is False
        and fit["parent_id"]==PARENT and fit["counts_ref"]==counts_ref
        and fit["preparation_ref"]==PREP_REF and fit.get("mlm_completion_ref")==mlm_ref
        and fit.get("post_MLM_bridge_source_ref")==ref(BRIDGE)
        and wait.get("phase")=="fit" and wait.get("parent_id")==PARENT
        and wait.get("driver_ref")==ref(DRIVER),
        "Actual own contrastive-only fresh fit and root wait0/drain")
    return fit

def validate_request(record):
    req=read(record);native=native_contract()
    require(req.get("schema")=="american_c3_native_specific_stage_request_v1"
        and req.get("root_authorizer")=="root_agent" and req.get("execution_authorized") is True
        and req.get("parent_id")==PARENT and req.get("phase") in PHASES
        and req.get("physical_gpu") in (4,5),"Explicit fresh root GPU4/5 postMLM phase")
    native.no_private_refs(req);refs=req["refs"];phase=req["phase"]
    required={"runtime","preparation","preparation_config","preparation_close","counts","MLM_completion"}
    if phase=="embeddings":required.add("fit_completion")
    require(set(refs)==required and refs["preparation"]==PREP_REF
        and refs["preparation_config"]==PREP_CONFIG_REF and refs["preparation_close"]==PREP_CLOSE_REF,
        "Exact actual observation-only preparation and phase-specific late-bound receipts")
    require(req["upstream_wait_refs"]["preparation"]==PREP_WAIT_REF,"Genuine original preparation wait")
    prep,_=validate_upstream(PREP_REF,PREP_WAIT_REF,preparation=True)
    close=read(PREP_CLOSE_REF)
    require(close["value"]["future_writes_and_relaunch_revoked"] is True
        and close["value"]["request"]["completed_ref"]==PREP_REF,"Original corpus permanently closed READONLY")
    counts=counts_metadata_v1(refs["counts"],req["upstream_wait_refs"]["counts"],refs)
    bridge=load(ref(BRIDGE),"_postMLM980_actual_MLM_bridge")
    bridge.v12_mlm_phase_v1(refs["MLM_completion"],req["upstream_wait_refs"]["MLM_completion"],
        dict(parent_id=PARENT,refs=dict(preparation=PREP_REF)))
    if phase=="embeddings":
        validate_fit_metadata_v1(refs["fit_completion"],req["upstream_wait_refs"]["fit_completion"],
            counts_ref=refs["counts"],mlm_ref=refs["MLM_completion"])
    caps=validate_caps(read(req["caps_ref"]),req);runtime=read(refs["runtime"])
    require(runtime.get("schema")=="american_c3_native_postMLM_runtime_binding_v1"
        and runtime.get("status")=="existing_inventory_bound_actual_GPU_at_execution"
        and runtime.get("root_authorizer")=="root_agent"
        and runtime["target_physical_gpu"]==req["physical_gpu"]
        and runtime["target_GPU_UUID"]==req["GPU_UUID"]
        and runtime["bert_base_config_ref"]==BASE_CONFIG_REF
        and runtime["native_CPU_qualification_ref"]==NATIVE_CPU_QUAL_REF
        and runtime["native_inventory_ref"]==NATIVE_INVENTORY_REF,
        "Existing exact inventory/local-base binding, actual device checked in held lease")
    import re
    require(re.fullmatch(r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",req["GPU_UUID"]),
        "Actual root-selected GPU UUID, never a guessed device")
    read(runtime["shared_cpu_runtime_ref"]);read(NATIVE_CPU_QUAL_REF)
    acceptance=read(NVML_ACCEPTANCE_REF)
    require(acceptance.get("active") is True and acceptance.get("root_accepts_actual_UUID_memory_comparison") is True,
        "Existing reviewed NVML ABI/device memory monitor acceptance")
    require(caps["checkpoint_live_allowance_bytes"]==(8*2**30+65536 if phase=="fit" else 0)
        and caps["cache_allowance_bytes"]==4*2**30 and caps["temp_allowance_bytes"]==256*2**20
        and caps["child_artifact_allowance_bytes"]==256*2**20
        and caps["max_checkpoint_files"]==2 and caps["max_checkpoint_bytes"]==4*2**30,
        "Fixed retention1/max2-file operational checkpoint/cache/TMP caps")
    return req,caps,runtime

def launcher_ref_v1():return ref(LAUNCHER_REF["path"])
