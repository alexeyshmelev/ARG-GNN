"""Specific default-inert root contracts; no guard/state/runtime work at import."""
from pathlib import Path
from datetime import datetime,timezone
import copy,hashlib,importlib.util,json,os,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/American_C3_native_MLM_full_source_v2"
SOURCE=REL+"/root_native_contract_v1.py"
DRIVER=REL+"/root_native_driver_v1.py"
CONTEXT=REL+"/context_factory_v1.py"
RUNTIME_HELPER=REL+"/native_runtime_v1.py"
PARENT="american_c3_train_000"
CORE="483d78bf2ef3c917d65ecf1dde5a7809bfadc21194c9f9a90394ce5c532fdd5e"
BASE="a8049bc103afb1cbbbce6f8fc3707a21de9613b23034599641b688994fc5caa3"
GUARD_REF=dict(path="research/reviews/concurrent_completed_scope_credits_v3/completed_scope_credits_v3.py",sha256="ea14c12c41b9385a86182aa9c24058f7b1276107fc5706c70e47828e8b2c31ef")
POLICY_REF=dict(path="research/reviews/completed_scope_credits_root_activation_20261015_v1/activation_policy.json",sha256="a0885b5209172dd79a835d442e0ad3495f5c1a0eac61a1c5c08154ec877ecf85")
REPAIR_REF=dict(path="research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json",sha256="f6f2d628199e8be73db8b5d90d428570d8afdefeab9dc5cd94bb5eafb39247db")
MEMO_REF=dict(path="research/reviews/resource_guard_hotpath_source_audit_v1/transaction_memo_adapter_v2.py",sha256="c972e9a779d424ceafdb258848614bf40994d1004e2c73fc5cd225ae99621364")
V4_REF=dict(path="src/arggnn/project_budget_v4.py",sha256="a093c7d86f57fee569cb99f4712250e9505a1119f3950fa44282c7ba8c57df47")
LAUNCHER_REF=dict(path="scripts/launch_reserved_gpu_authorized_GPU45_completed_v1.py",sha256="d4cc990e430625e9cc7d02739e7e8012f5c8b3bdc78f048f801febacea64583d")
GPU_AUTH_REF=dict(path="research/reviews/H040_GPU45_migration_root_20261015_v1/GPU45_authorization.json",sha256="0231ae8edb05f8dd34ce127a0de0b97c1dd64fa461ad081e7a83bb92c558d26c")
BASE_CONFIG_REF=dict(path="research/reviews/C3_bert_base_configuration_root_v1/config.json",sha256="7160e1553ad2ca51d8c1cb066be533db31826e12d173824c1bb0cb1a4f187d20")
NATIVE_CPU_QUAL_REF=dict(path="research/reviews/C3_bert_base_configuration_root_v1/native_CPU_qualification.json",sha256="56ed247a09484753a64e9ed3bee6c80eea4e5bfb156ab5acdb09c24ce828e515")
NATIVE_INVENTORY_REF=dict(path="experiments/results/native_argformer_tsinfer_fit_v7_root/environment_inventory.json",sha256="077d3e950c3bb3efdecc7292e06c8157506663293b4efe4723ccf7b877973461")
LEGACY_RUNTIME_REF=dict(path="src/arggnn/native_argformer_tsinfer_fit_v7.py",sha256="b94a4e16a6a9dc230256974b9a0e45a6edd1b41b9ecc8ac4171c4461ec514712")
PROC_HELPER="research/reviews/American_C3_native_paths_root_v1/root_native_paths_driver.py"
OBS_REF=dict(path="experiments/results/American_C3_observed_full30mb_supervisor_repair_20261015_v1/observed/manifest.json",sha256="c3d58a8d10d901a162f406e2e218a10e17e5f1617b3c626503d72bcbd1b4ccee")
PREP_OUTPUT="experiments/results/American_C3_native_paths_train000_full30mb_seed92001_v3"
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
    guard=load(GUARD_REF,"_C3_native_current_credit_v3")
    guard.configure_v2(REPAIR_REF,policy_ref=POLICY_REF)
    require(all(getattr(guard.budget.v4,n).__module__==guard.budget.v4.__name__
        for n in ("canonical","file_bytes","mutex")),"Root reserve/close must use unwrapped originalv4")
    return guard

def consumer_guard():
    guard=original_guard();memo=load(MEMO_REF,"_C3_native_consumer_memo2")
    adapter=memo.install(guard.budget.v4,expected_source_path=ROOT/V4_REF["path"],
        expected_source_sha256=V4_REF["sha256"])
    return guard,adapter

PREP_REF=dict(path=PREP_OUTPUT+"/completed.json",sha256="3cc49fbf55bd4b4b3859a93ce1123e9218f407f7c7299ad5029be81f59c7c6fa")
PHASES=GPU_PHASES=("qualify_mlm","mlm")
CPU_PHASES=()
GPU_MAIN="src/arggnn/american_c3_native_mlm_full_v2.py"
CPU_MAIN=None
def native_contract():
    sys.path.insert(0,str(ROOT/REL))
    import consumer_contract_v1 as native
    return native
def phase_paths(phase):
    require(phase in PHASES,"Specific MLM-only stage")
    return ROOT/REL/("admitted_"+phase+"_v1"),"experiments/results/American_C3_native_MLM_full30mb_"+phase+"_v2"
def source_paths():
    native=native_contract()
    return tuple(dict.fromkeys((*native.PROJECT_SOURCES,SOURCE,DRIVER,CONTEXT,RUNTIME_HELPER,PROC_HELPER,
        "src/arggnn/census_tsinfer_fit_v1.py","src/arggnn/census_tsinfer_observation_v1.py",
        "src/arggnn/biological_candidate.py","src/arggnn/project_budget_v9_consumer.py",
        "scripts/launch_reserved_gpu_full_storage_v1.py","scripts/launch_reserved_gpu_terminal_retired_scopes_v1.py")))
def validate_upstream(completed_ref,wait_ref,*,preparation=False):
    completed=read(completed_ref);wait=read(wait_ref)
    require(wait.get("returncode")==0 and wait.get("completed_ref")==completed_ref,"Actual same completion/directwait0")
    if preparation:
        require(completed_ref==PREP_REF and completed.get("status")=="completed_native_C3_path_preparation"
            and completed.get("parent_id")==PARENT and completed.get("observation_ref")==OBS_REF
            and wait.get("actual_direct_worker_wait") is True and wait.get("owned_worker_group_ESRCH") is True
            and wait.get("cleanup_errors")==[] and wait.get("operational_failure") is None,
            "Actual complete v3 native corpus and directwait/drain")
    else:
        require(wait.get("actual_direct_launcher_wait") is True
            and wait.get("all_owned_processes_and_groups_drained") is True
            and wait.get("operational_failure") is None,"Actual completed qualifier wait/drain")
    return completed,wait
def validate_caps(caps,request):
    require(caps.get("schema")=="american_c3_native_mlm_caps_v1" and caps.get("root_authorizer")=="root_agent"
        and caps.get("active") is True and caps.get("phase")==request["phase"]
        and caps.get("parent_id")==PARENT and caps.get("physical_gpu")==request["physical_gpu"]
        and caps.get("driver_source_ref")==ref(DRIVER) and caps.get("runtime_ref")==request["refs"]["runtime"]
        and caps.get("preparation_ref")==PREP_REF,"Root actual same-stage source/corpus/runtime caps")
    exact=dict(reservation_cap_bytes=18*2**30,phase_write_budget_bytes=16*2**30,
        checkpoint_live_allowance_bytes=6*2**30,cache_allowance_bytes=8*2**30,
        temp_allowance_bytes=256*2**20,child_artifact_allowance_bytes=256*2**20,
        phase_metadata_allowance_bytes=64*2**20,outer_auxiliary_cap_bytes=128*2**20,
        root_packet_overhead_cap_bytes=2**20,child_log_ram_bytes=4*2**20,outer_log_cap_bytes=16*2**20,
        max_checkpoint_files=2,max_checkpoint_bytes=2950918528,operational_deadline_seconds=86400)
    require(all(type(caps.get(k)) is int and caps[k]==v for k,v in exact.items()),"Exact reviewed 18GiB/16GiB planning caps")
    require(type(caps.get("max_cpu_rss_bytes")) is int and 0<caps["max_cpu_rss_bytes"]<=384*2**30
        and type(caps.get("max_cuda_reserved_bytes")) is int and 0<caps["max_cuda_reserved_bytes"]<=50000000000,
        "Root actual RAM/GPU caps, no fit guarantee")
    require(caps.get("bound_method")=="source_counts_with_unmeasured_cache_workspace_allowance"
        and caps.get("aggregate_cache_OS_quota_claimed") is False,"Honest operational cache allowance")
    return caps
def make_config(req,caps):
    native=native_contract();cfg=copy.deepcopy(native.DEFAULTS)
    cfg.update(execution_enabled=True,phase=req["phase"],phase_write_budget_bytes=caps["phase_write_budget_bytes"],
        short_tmp_directory=".runtime/tmp/c3mlm2/"+("q" if req["phase"]=="qualify_mlm" else "f"),
        child_log_ram_bytes=caps["child_log_ram_bytes"],max_checkpoint_files=caps["max_checkpoint_files"],
        max_checkpoint_bytes=caps["max_checkpoint_bytes"])
    return cfg
def validate_request(record):
    req=read(record);native=native_contract()
    require(req.get("schema")=="american_c3_native_mlm_stage_request_v1" and req.get("root_authorizer")=="root_agent"
        and req.get("execution_authorized") is True and req.get("parent_id")==PARENT
        and req.get("phase") in PHASES and req.get("physical_gpu") in (4,5),"Root specific MLM-only stage request")
    native.no_private_refs(req);refs=req["refs"];phase=req["phase"]
    require(all(isinstance(refs.get(k),dict) for k in ("runtime","preparation","preparation_config","preparation_close")),
        "Root must bind actual corpus/config/closed READONLY receipt/runtime")
    require(refs["preparation"]==PREP_REF,"Exact completed v3 corpus")
    prep,_=validate_upstream(PREP_REF,req["upstream_wait_refs"]["preparation"],preparation=True)
    pcfg=read(refs["preparation_config"])
    require(pcfg["execution_enabled"] is True and pcfg["path_sampling_seed"]==92001 and pcfg["observation_ref"]==OBS_REF,
        "Original full TRAIN000 corpus recipe")
    closed=read(refs["preparation_close"]);value=closed.get("value",{})
    require(closed.get("kind")=="concurrent_completed_scope_close_receipt_v1"
        and closed.get("key")=="H041/American-C3-native-paths-train000-full30mb-92001-v3"
        and value.get("future_writes_and_relaunch_revoked") is True
        and value.get("request",{}).get("completed_ref")==PREP_REF,"Actual permanently closed READONLY corpus credit receipt")
    require(prep["unique_paths"]==25574587 and prep["path_tokens"]==316672003 and prep["vocabulary_nodes"]==106769
        and prep["max_raw_path_nodes"]==29 and prep["query_ancestry_truth_opened"] is False,"Exact full native corpus counts")
    caps=validate_caps(read(req["caps_ref"]),req);runtime=read(refs["runtime"])
    require(runtime.get("schema")=="american_c3_native_mlm_runtime_candidate_v1"
        and runtime.get("root_authorizer")=="root_agent" and runtime.get("status")=="candidate_no_GPU_fit_claim"
        and runtime.get("target_physical_gpu")==req["physical_gpu"] and runtime.get("target_GPU_UUID")
        and runtime.get("native_inventory_ref")==NATIVE_INVENTORY_REF
        and runtime.get("bert_base_config_ref")==BASE_CONFIG_REF
        and runtime.get("native_CPU_qualification_ref")==NATIVE_CPU_QUAL_REF,"Exact pinned runtime candidate; no invented GPU qualification")
    for k in ("shared_cpu_runtime_ref","legacy_inventory_policy_ref","legacy_runtime_bindings_ref"):read(runtime[k],cap=33554432)
    cpu=read(NATIVE_CPU_QUAL_REF);read(BASE_CONFIG_REF);read(NATIVE_INVENTORY_REF,cap=33554432)
    require(ref(runtime["native_python_ref"]["path"])==runtime["native_python_ref"]
        and runtime["native_python_ref"]["sha256"]==cpu["executable_sha256"]
        and local(runtime["native_prefix"])==local(cpu["prefix"])
        and cpu["GPU_context_created"] is False and cpu["model_instantiated"] is False,"Exact existing native interpreter/CPU qualification")
    if phase=="mlm":
        require(isinstance(refs.get("qualification"),dict) and isinstance(refs.get("timing_review"),dict),
            "Full fit requires actual 512/big-vocab qualifier and root timing review")
        q,_=validate_upstream(refs["qualification"],req["upstream_wait_refs"]["qualification"])
        child=read(q["child_completion_ref"]);proof=child["qualification"]
        require(q.get("status")=="completed_C3_native_MLM_qualification_v1"
            and q.get("parent_id")==PARENT and q.get("preparation_ref")==PREP_REF
            and q.get("runtime_ref")==refs["runtime"] and child.get("physical_gpu")==req["physical_gpu"]
            and proof.get("status")=="passed_synthetic_MLM_resource_qualification"
            and proof.get("batch")==512 and proof.get("microbatch")==512 and proof.get("padded_length")==128
            and proof.get("true_tokens_per_path")==30 and proof.get("vocabulary")==106776
            and proof.get("parameters")==167655192 and proof.get("optimizer_steps")==3
            and proof.get("precision")=="amp_bf16" and proof.get("losses_finite") is True
            and proof.get("actual_context_uuid")==runtime["target_GPU_UUID"].removeprefix("GPU-")
            and proof.get("scientific_training_or_checkpoint_claimed") is False,"Actual same-GPU stage-specific full512/native big-vocab qualification")
        qspec=read(child["active_input_ref"])
        require(qspec["source_sha256"].get(REL+"/qualifier_v1.py")==ref(REL+"/qualifier_v1.py")["sha256"]
            and proof.get("seed")==17 and proof.get("layers")==12 and proof.get("heads")==12
            and proof.get("hidden_size")==768 and proof.get("intermediate_size")==3072
            and 0<proof["peak_reserved_bytes"]<=caps["max_cuda_reserved_bytes"]
            and len(proof["step_seconds"])==3 and all(type(x) in (int,float) and x>0 for x in proof["step_seconds"]),
            "Actual current-source full architecture and peak/timing qualifier evidence")
        timing=read(refs["timing_review"])
        require(timing.get("root_authorizer")=="root_agent" and timing.get("active") is True
            and timing.get("qualification_ref")==refs["qualification"]
            and timing.get("actual_qualifier_step_seconds")==proof["step_seconds"]
            and timing.get("full_optimizer_updates")==224775 and timing.get("operational_deadline_seconds")==86400
            and timing.get("root_accepts_unproven_24h_completion") is True
            and timing.get("full_budget_ETA_claimed") is False,"Actual timing evidence and explicit root24h risk review; no invented ETA")
    return req,caps,runtime
