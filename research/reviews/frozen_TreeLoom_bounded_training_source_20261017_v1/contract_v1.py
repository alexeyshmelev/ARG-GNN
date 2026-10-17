"""Finite ROOT input/source/storage contract; standard library, inert on import."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib.util,json,os,stat,sys,time
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/frozen_TreeLoom_bounded_training_source_20261017_v1"
MANIFEST=HERE+"/source_manifest.v1.json";DIRECT=HERE+"/borrowed_gpu0_v1.py"
SOURCE=HERE+"/worker_v1.py"
GPU0_UUID="GPU-ff06ca1f-403a-f2d0-b584-71e177452667"
BASE_POOL=dict(path="src/arggnn/gpu_pool_v1.py",sha256="ae1e0494ab672cfa3017759460eb06a4b4789eaef54928496c3c80314adcc0e1")
GPU0_GUARD=dict(path="research/reviews/GPU0_shared_execution_source_20261016_v1/gpu0_native_guard_v1.py",sha256="3a08d3bd84adc95aafd332bc224fdb7aa38730d6c2f48dee8108944251b7ee69")
PIDFD=dict(path="research/reviews/American_C3_native_MLM_full_source_v5/pidfd_backend_v1.py",sha256="61be5d89357d1b95611e0ba26dd39d381ececf3ce6a36c347507cdc062373eca")
OWNED=dict(path="research/reviews/American_C3_native_MLM_amendment_v12/owned_lifecycle_v2.py",sha256="93d71acc40f1037d0b37acb2ca0859300ea4bd9741409e9bd00ca21bf100d633")
GUARD=dict(path="research/reviews/Census_observation_two_scope_retention_source_20261017_v1/census_observation_retention_v1.py",sha256="72f6dc51844b2a261b581f5e69bd2a22982e6244c059f24549747a7b851faeeb")
GPU_PHASES=("qualification","full_training")
CURRENT_INPUT_REF=None
_SELECTED_GUARD=None
_CONSUMER_MEMO=None
def require(ok,message):
    if not ok:raise ValueError(message)
need=require
def local(path):
    p=Path(path);p=p if p.is_absolute()else ROOT/p
    require(p!=ROOT and p.is_relative_to(ROOT)and".."not in p.parts and p.resolve()==p,"Canonical in-project path")
    return p
def ref(path):
    p=local(path);s=p.lstat()
    require(stat.S_ISREG(s.st_mode)and s.st_nlink==1 and s.st_size<=32*2**20,"Compact single-link source/metadata")
    return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def raw(record):
    require(type(record)is dict and set(record)=={"path","sha256"}and ref(record["path"])==record,"Exact immutable source/metadata SHA")
    return local(record["path"]).read_bytes()
def read(record):return json.loads(raw(record))
def encoded(value):return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def load(record,name):
    raw(record);sp=importlib.util.spec_from_file_location(name,local(record["path"]))
    m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def active(value):
    now=datetime.now(timezone.utc);a=datetime.fromisoformat(value["created_at_utc"]);b=datetime.fromisoformat(value["expires_at_utc"])
    require(a.tzinfo and b.tzinfo and a<=now<b and value.get("root_authorizer")=="root_agent"
        and value.get("status")=="active"and value.get("execution_authorized")is True
        and value.get("template_only")is not True,"Finite explicit ROOT authority")
def validate_input(record):
    global CURRENT_INPUT_REF
    require(CURRENT_INPUT_REF in(None,record),"One immutable input per interpreter")
    q=read(record);active(q)
    require(q["schema"]=="ROOT_frozen_TreeLoom_bounded_training_input_v1"and q["phase"]in GPU_PHASES
        and q["source_manifest_ref"]==ref(MANIFEST)and q["physical_gpu"]==0 and q["gpu_uuid"]==GPU0_UUID
        and q["process_title"]=="AI Auto Research Pipeline Test"and q["query_VAL_TEST_truth_access_authorized"]is False,
        "Exact frozen qualification or600update training input, GPU0only")
    m=read(q["source_manifest_ref"])
    for path,sha in m["source_sha256"].items():raw(dict(path=path,sha256=sha))
    review=read(q["source_review_ref"])
    require(review.get("independent")is True and review.get("verdict")=="SOURCE_PASS"and review.get("blockers")==[]
        and review.get("candidate_manifest_ref")==q["source_manifest_ref"],"Independent whole-route source pass")
    cfg=read(q["config_ref"]);expected=read(ref(HERE+"/scientific_config.FROZEN.v1.json"))
    require(cfg==expected and cfg["epochs"]==30 and cfg["total_optimizer_updates"]==600
        and cfg["checkpoint_selection"]=="fixed_final_epoch30_before_scored_benchmark_truth"
        and cfg["memory_arm"]=="stable_leaf_memory"and cfg["execution_enabled"]is True,"Exact600update scientific freeze")
    caps=q["caps"]
    require(caps["max_cuda_reserved_bytes"]==64*2**30 and caps["non_Torch_margin_bytes"]==2*2**30
        and caps["headroom_bytes"]==4*2**30 and caps["max_cpu_rss_bytes"]==384*2**30
        and caps["cleanup_seconds"]==30 and caps["terminal_evidence_margin_seconds"]==150
        and caps["work_cutoff_margin_seconds"]==300 and caps["log_bytes"]==4*2**20
        and 0<caps["max_seconds"]<=7*86400 and caps["output_cap_bytes"]==32*2**20
        and caps["reservation_bytes"]==64*2**20,"Finite explicit original resource bounds, no fit claim")
    start=datetime.fromisoformat(q["created_at_utc"]).timestamp();expiry=datetime.fromisoformat(q["expires_at_utc"]).timestamp()
    require(start+caps["max_seconds"]<=expiry and q["drain_expires_unix"]==expiry
        and q["ROOT_record_directory"]==q["worker_auxiliary_scope"],"Single immutable work and terminal clock")
    require(q["base_pool_ref"]==BASE_POOL and q["all_other_GPU0_borrower_routes_revoked"]is True
        and q["preserve_controller_until_owned_drain"]is True and q["automatic_retry_or_resume"]is False,"Exact empty-owner exclusive borrower")
    required=["american_c3_train_000"]if q["phase"]=="qualification"else["american_c3_train_%03d"%i for i in range(20)]
    require(list(q["parents"])==required and set(q["train_truth_refs"])==set(required)
        and q["qualification_is_disposable_and_not_benchmark_result"]is(q["phase"]=="qualification"),"Separate1parent check or exact20TRAIN")
    if q["phase"]=="full_training":
        acceptance=read(q["qualification_acceptance_ref"])
        require(acceptance.get("schema")=="ROOT_frozen_TreeLoom_actual_qualification_acceptance_v1"
            and acceptance.get("root_authorizer")=="root_agent"and acceptance.get("actual_outer_terminal_consumed")is True
            and type(acceptance.get("actual_exit_code"))is int and acceptance["actual_exit_code"]==0
            and acceptance.get("all_owned_processes_drained")is True
            and acceptance.get("source_manifest_ref")==q["source_manifest_ref"]
            and acceptance.get("synthetic_all49_parity_passed")is True
            and acceptance.get("full30Mb_TRAIN000_disposable_optimizer_update_verified")is True
            and acceptance.get("qualification_weights_reused")is False,
            "Real consumed parity+wholeTRAIN000 qualification0 before substantial training")
        for key in("qualification_input_ref","completed_ref","actual_parent_wait_ref","actual_outer_terminal_ref"):raw(acceptance[key])
        barrier=read(q["observation_barrier_ref"])
        require(barrier["schema"]=="frozen_American_full30Mb_all20_observation_barrier_v2"
            and barrier["TRAIN_parent_ids"]==required and barrier["region_length_bp"]==30000000
            and barrier["all_native_trees_nodes_sites_samples_retained"]is True
            and barrier["all20_observations_sealed_before_this_full_training_truth"]is True
            and barrier["this_full_training_truth_opened"]is False
            and barrier["prior_disposable_TRAIN000_qualification_genuine_truth_opened"]is True
            and barrier["qualification_weights_optimizer_or_rng_reused"]is False and set(barrier["parents"])==set(required),
            "Real all20observation barrier before any full-training truth")
    runtime=read(q["runtime_ref"]);qualification=read(q["runtime_qualification_ref"])
    require(isinstance(q["cpu_affinity"],list)and 8<=len(q["cpu_affinity"])<=32
        and len(set(q["cpu_affinity"]))==len(q["cpu_affinity"])and all(type(n)is int and n>=0 for n in q["cpu_affinity"])
        and qualification["cpu_affinity"]==q["cpu_affinity"],"Actual ROOT-assigned bounded CPU affinity")
    require(runtime["python"]==qualification["python"]and qualification["ROOT_qualified_existing_Torch_PyG_runtime"]is True
        and runtime["source_ref"]==ref(SOURCE)and runtime["CUDA_VISIBLE_DEVICES"]==GPU0_UUID
        and local(runtime["python"]).is_file(),"Existing ROOT-qualified scientific interpreter, no installation")
    CURRENT_INPUT_REF=record
    return q,None,None,read(q["storage_ref"])
def active_amendment_v1():
    require(CURRENT_INPUT_REF is not None,"Actual authenticated new frozen input")
    q=read(CURRENT_INPUT_REF);active(q);return q
def guard_v1(q,storage):
    global _SELECTED_GUARD,_CONSUMER_MEMO
    if _SELECTED_GUARD is None:
        g=load(GUARD,"_frozen_TreeLoom_exact_Census")
        selected=g.load_selected_guard_v1(q["storage_policy_selector_ref"])
        require(selected is g and g.MODE=="Census_observation_two_scope_retention_v1","Existing exact current Census guard")
        namespace=load(dict(path="research/reviews/current_mutex_contention_20261015_v1/consumer_namespace_memo_v1.py",
            sha256="17a1377320f61c3c0cc8d3816d0259f9dae3121ed91bf7206a7ee63fcab8eb7f"),"_frozen_TreeLoom_readonly_namespace")
        _CONSUMER_MEMO=namespace.install_consumer_v1(g.budget.v4,g.budget,consumer_role="read_only_consumer",
            expected_source_path=ROOT/"src/arggnn/project_budget_v4.py",
            expected_source_sha256="a093c7d86f57fee569cb99f4712250e9505a1119f3950fa44282c7ba8c57df47")
        _SELECTED_GUARD=g
    g=_SELECTED_GUARD
    require(g._SELECTOR_REF==q["storage_policy_selector_ref"],"No read-only guard lifetime selector swap")
    g.validate_storage_v9(q,dict(output_cap_bytes=q["caps"]["reservation_bytes"]),storage,
        source_ref=ref(SOURCE),config_ref=q["config_ref"],runtime_ref=q["runtime_ref"])
    required=[ref(SOURCE),ref(MANIFEST),q["config_ref"],q["runtime_ref"]]
    own=storage["reservation"]
    require(own["reserved_bytes"]==64*2**20 and own["output"]==str(local(q["output_directory"]))
        and own["auxiliary_scopes"]==[str(local(q["worker_auxiliary_scope"]))]
        and all(r in own["consumer_bindings"]for r in required)and len(own["consumer_bindings"])<=64,
        "Actual original whole primary+AUX64MiB reservation")
    return g
def validate_cap_v1(record):return validate_input(record)
def put(q,name,value,*,directory=None):
    folder=local(q["worker_auxiliary_scope"])if directory is None else directory
    data=encoded(value)+b"\n";require(len(data)<=65536,"Bounded64KiB metadata")
    p=folder/name
    rows=list(folder.iterdir());limit=q["caps"]["output_cap_bytes"]if folder==local(q["output_directory"])else 16*2**20
    require(all(x.is_file()and not x.is_symlink()for x in rows)and len(rows)<64
        and 4096+sum(x.stat().st_size+4096 for x in rows)+len(data)+4096<=limit,"Bounded flat own metadata scope")
    with p.open("xb")as stream:require(stream.write(data)==len(data),"Complete exclusive write");stream.flush();os.fsync(stream.fileno())
    return ref(str(p.relative_to(ROOT)))


def work_deadline_monotonic_v1():
    q=active_amendment_v1()
    end=min(datetime.fromisoformat(q["created_at_utc"]).timestamp()+q["caps"]["max_seconds"],
        q["drain_expires_unix"])-q["caps"]["work_cutoff_margin_seconds"]
    return time.monotonic()+end-time.time()
