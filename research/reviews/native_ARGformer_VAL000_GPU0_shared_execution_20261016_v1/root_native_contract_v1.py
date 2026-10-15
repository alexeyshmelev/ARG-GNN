"""Specific default-inert root contracts; no guard/state/runtime work at import."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import copy,hashlib,importlib.util,json,os,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/native_ARGformer_VAL000_GPU0_shared_execution_20261016_v1"
SOURCE=REL+"/root_native_contract_v1.py"
DRIVER=REL+"/root_native_driver_v1.py"
CONTEXT=REL+"/context_factory_v1.py"
RUNTIME_HELPER=REL+"/native_runtime_v1.py"
PARENT='american_c3_val_000'
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
GPU_AUTH_REF={'path': 'research/reviews/GPU0_only_user_transition_root_20261016_v1/GPU0_only_policy.root_active_v1.json', 'sha256': 'bba84cecf416b2d284126bd31b72cff26ef9c46ee24dae375721d3f57240ed24'}
BASE_CONFIG_REF=dict(path="research/reviews/C3_bert_base_configuration_root_v1/config.json",sha256="7160e1553ad2ca51d8c1cb066be533db31826e12d173824c1bb0cb1a4f187d20")
NATIVE_CPU_QUAL_REF=dict(path="research/reviews/C3_bert_base_configuration_root_v1/native_CPU_qualification.json",sha256="56ed247a09484753a64e9ed3bee6c80eea4e5bfb156ab5acdb09c24ce828e515")
NATIVE_INVENTORY_REF=dict(path="experiments/results/native_argformer_tsinfer_fit_v7_root/environment_inventory.json",sha256="077d3e950c3bb3efdecc7292e06c8157506663293b4efe4723ccf7b877973461")
LEGACY_RUNTIME_REF=dict(path="src/arggnn/native_argformer_tsinfer_fit_v7.py",sha256="b94a4e16a6a9dc230256974b9a0e45a6edd1b41b9ecc8ac4171c4461ec514712")
PROC_HELPER="research/reviews/American_C3_native_paths_root_v1/root_native_paths_driver.py"
OBS_REF={'path': 'experiments/results/American_C3_val000_full30mb_shared_v1/inference/observed/manifest.json', 'sha256': '899e00e64880a0f8c8c34897e6c6d5d6e6c8be886557add343d65d0f70c90c44'}
PREP_OUTPUT='experiments/results/American_C3_val000_full30mb_shared_v1/native_paths'
PREP_REF={'path': 'experiments/results/American_C3_val000_full30mb_shared_v1/native_paths/completed.json', 'sha256': '40a3b588d0aa2b474c3f4aab492e34feb0a753a0f8cb73c20d9c0059d8a8c16b'}
PREP_CONFIG_REF={'path': 'research/reviews/prospective_980GB_capacity_source_v1/admitted_native_paths_v1/config.json', 'sha256': 'c7c95390e12daa2216babc8ccf82d04227a2c96f4e9af8a04dfb626443fa3146'}
PREP_CLOSE_REF={'path': 'research/reviews/VAL000_native_paths_completed_close_20261015_v1/closed_VAL000_native_paths_v1/close_receipt.json', 'sha256': '6fbf88b2ddabf0450a24f91b1d6499f11aa9f5a25a77acb86535821bfa28ac94'}
PREP_WAIT_REF={'path': 'research/reviews/prospective_980GB_capacity_source_v1/admitted_native_paths_v1/actual_wait.json', 'sha256': '3b7e0d9a2e0b83c968c8e2340b850d44e27db1783990a07e933cfa09b58975b8'}
COUNTS_REF={'path': 'experiments/results/American_C3_native_val000_full30mb_counts980_namespace_v1/completed.json', 'sha256': '502c111b443c46eea789dc03ee3ff8b6454546ce6e86e3f9fcddba25e82489a7'}
COUNTS_WAIT_REF={'path': 'research/reviews/C3_native_VAL000_counts_namespace_root_20261015_v1/admitted_counts980_v1/actual_wait.json', 'sha256': 'ab2d1f7a7e09756009556ae6f793899c9e4fd5a7cc202e25d877104d3b2f9c91'}
PREP_IO_REF={'path': 'research/reviews/prospective_980GB_capacity_source_v1/root_VAL_cpu_execution_v2.py', 'sha256': '9f684478aca4d7f4867dc58cc3542bb651559e137a168bf35a9f61570ecc5847'}
CONTRASTIVE_ROWS={'train': 8748479, 'val': 972757}
CONTRASTIVE_UPDATES=2135
CONTRASTIVE_DROPPED_REMAINDER=3519
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
RENEWAL_SOURCE_REF={'path': 'research/reviews/H039_marker_repair_renewal_source_20261016_v1/marker_repair_renewal_v1.py', 'sha256': '491236877e8d71cae8bd7619880858ac501c57dd93e59479ca32dacacab20468'}
RENEWAL_CERTIFICATE_REF={'path': 'research/reviews/H039_marker_repair_renewal_source_20261016_v1/renewal_certificate.root_active_v1.json', 'sha256': '66006202094d7a3ef2b29cbec28eb93f334505b72589f09d86e4b57498162db7'}
def install_marker_renewal_v1():
    require(ref(RENEWAL_SOURCE_REF["path"])==RENEWAL_SOURCE_REF,"Exact reviewed additive marker renewal")
    module=load(RENEWAL_SOURCE_REF,"_VAL000_prediction_route_sharedGPU0_marker_renewal_v1")
    return module.install_renewal_v1(RENEWAL_CERTIFICATE_REF)


GPU0_UUID='GPU-ff06ca1f-403a-f2d0-b584-71e177452667'
GPU0_POLICY_REF={'path': 'research/reviews/GPU0_only_user_transition_root_20261016_v1/GPU0_only_policy.root_active_v1.json', 'sha256': 'bba84cecf416b2d284126bd31b72cff26ef9c46ee24dae375721d3f57240ed24'}
GPU0_STOP_REF={'path': 'research/reviews/GPU0_only_user_transition_root_20261016_v1/actual_stop_and_absence.root_v1.json', 'sha256': 'dd47aa8483c6d03ce5c45213c9ead33e0faee999103ba2db926b0f5b25384de1'}
GPU0_NATIVE_GUARD_REF={"path":'research/reviews/GPU0_shared_execution_source_20261016_v1/gpu0_native_guard_v1.py',"sha256":'3a08d3bd84adc95aafd332bc224fdb7aa38730d6c2f48dee8108944251b7ee69'}
SHARED_POOL_SOURCE="research/reviews/GPU0_shared_execution_source_20261016_v1/gpu0_shared_pool_v1.py"
SHARED_POOL_LAUNCHER="research/reviews/GPU0_shared_execution_source_20261016_v1/launch_gpu0_shared_v1.py"
SHARED_POOL_BASE="src/arggnn/gpu_pool_v1.py"
SHARED_NATIVE_RUNTIME='research/reviews/native_ARGformer_VAL000_GPU0_shared_execution_20261016_v1/shared_native_runtime_entry_v1.py'
STORAGE_MONITOR='research/reviews/native_ARGformer_VAL000_GPU0_shared_execution_20261016_v1/owned_storage_profile_v1.py'
GPU0_SHARED_SOURCE_REFS={'research/reviews/GPU0_only_user_transition_root_20261016_v1/GPU0_only_policy.root_active_v1.json': {'path': 'research/reviews/GPU0_only_user_transition_root_20261016_v1/GPU0_only_policy.root_active_v1.json', 'sha256': 'bba84cecf416b2d284126bd31b72cff26ef9c46ee24dae375721d3f57240ed24'}, 'research/reviews/GPU0_only_user_transition_root_20261016_v1/actual_stop_and_absence.root_v1.json': {'path': 'research/reviews/GPU0_only_user_transition_root_20261016_v1/actual_stop_and_absence.root_v1.json', 'sha256': 'dd47aa8483c6d03ce5c45213c9ead33e0faee999103ba2db926b0f5b25384de1'}, 'research/reviews/GPU0_shared_execution_source_20261016_v1/gpu0_bootstrap_renewal_v1.py': {'path': 'research/reviews/GPU0_shared_execution_source_20261016_v1/gpu0_bootstrap_renewal_v1.py', 'sha256': '58fc5e27f5cdc02f70463314068160299981778cf0196d1e77c0577ca27cfcc1'}, 'research/reviews/GPU0_shared_execution_source_20261016_v1/gpu0_native_guard_v1.py': {'path': 'research/reviews/GPU0_shared_execution_source_20261016_v1/gpu0_native_guard_v1.py', 'sha256': '3a08d3bd84adc95aafd332bc224fdb7aa38730d6c2f48dee8108944251b7ee69'}, 'research/reviews/GPU0_shared_execution_source_20261016_v1/gpu0_runtime_adapter_v1.py': {'path': 'research/reviews/GPU0_shared_execution_source_20261016_v1/gpu0_runtime_adapter_v1.py', 'sha256': '957ec724bb345a76932d5ea43b0df527bc9d12b068815db91cf17da1b0c85896'}, 'research/reviews/GPU0_shared_execution_source_20261016_v1/gpu0_shared_pool_v1.py': {'path': 'research/reviews/GPU0_shared_execution_source_20261016_v1/gpu0_shared_pool_v1.py', 'sha256': '28bb690f33c2384d04f88dbacff1f910694c398c81874f9b9751336f4fb6429c'}, 'research/reviews/GPU0_shared_execution_source_20261016_v1/launch_gpu0_shared_v1.py': {'path': 'research/reviews/GPU0_shared_execution_source_20261016_v1/launch_gpu0_shared_v1.py', 'sha256': '9bf5df4b964dfb537bd5e534de2a5a4b4c19e2d3e7b5857c92c64db376ee5b40'}, 'research/reviews/H039_marker_repair_renewal_source_20261016_v1/marker_repair_renewal_v1.py': {'path': 'research/reviews/H039_marker_repair_renewal_source_20261016_v1/marker_repair_renewal_v1.py', 'sha256': '491236877e8d71cae8bd7619880858ac501c57dd93e59479ca32dacacab20468'}, 'research/reviews/H039_marker_repair_renewal_source_20261016_v1/renewal_certificate.root_active_v1.json': {'path': 'research/reviews/H039_marker_repair_renewal_source_20261016_v1/renewal_certificate.root_active_v1.json', 'sha256': '66006202094d7a3ef2b29cbec28eb93f334505b72589f09d86e4b57498162db7'}, 'src/arggnn/gpu45_c3cache_h050_pool_v1.py': {'path': 'src/arggnn/gpu45_c3cache_h050_pool_v1.py', 'sha256': 'd3279c6208737371031330f4bef9325199ba20c86f0f7d8e930c134b128d29fb'}, 'src/arggnn/gpu_pool_v1.py': {'path': 'src/arggnn/gpu_pool_v1.py', 'sha256': 'ae1e0494ab672cfa3017759460eb06a4b4789eaef54928496c3c80314adcc0e1'}, 'research/reviews/GPU45_shared_downstream_lineage_source_20261016_v1/pool.staged.py': {'path': 'research/reviews/GPU45_shared_downstream_lineage_source_20261016_v1/pool.staged.py', 'sha256': '07d930aba589c16be8bd8c97a7948e519e108def97bc7538bb061bf634c6a5ce'}}
GPU0_SOURCE_ROOT="research/reviews/native_ARGformer_VAL000_GPU0_shared_execution_20261016_v1"
GPU0_REVIEWED_SOURCE_PATHS=('consumer_contract_v1.py', 'context_factory_v1.py', 'entry_postMLM_v1.py', 'launcher_postMLM980_v1.py', 'mlm_v1/consumer_contract_v1.py', 'mlm_v1/context_factory_v1.py', 'mlm_v1/entry_mlm980_v1.py', 'mlm_v1/launcher_mlm980_v1.py', 'mlm_v1/native_child_v1.py', 'mlm_v1/native_runtime_v1.py', 'mlm_v1/root_native_contract_v1.py', 'mlm_v1/root_native_driver_v1.py', 'mlm_v1/supervisor_v1.py', 'native_child_v1.py', 'native_runtime_v1.py', 'owned_storage_profile_v1.py', 'post_mlm_bridge_v1.py', 'root_native_contract_v1.py', 'root_native_driver_v1.py', 'shared_native_runtime_entry_v1.py', 'supervisor_v1.py')
GPU0_EXECUTION_APPROVAL_PATH=GPU0_SOURCE_ROOT+"/GPU0_execution_approval.root_active_v1.json"
def require_gpu0_execution_route_v1():
    # Explicit bounded metadata only; no guard configure, mutation or probe here.
    policy=read(GPU0_POLICY_REF);absence=read(GPU0_STOP_REF)
    require(policy.get("schema")=="root_latest_human_GPU_policy_v1"
        and policy.get("status")=="active" and policy.get("root_authorizer")=="root_agent"
        and policy.get("allowed_physical_GPUs")==[0] and policy.get("maximum_distinct_project_GPUs")==1
        and policy.get("all_nonzero_GPUs_excluded") is True and policy.get("GPU0_uuid")==GPU0_UUID
        and policy.get("preserve_external_jobs") is True and policy.get("stop_evidence_ref")==GPU0_STOP_REF,
        "Exact current human GPU0-only device policy")
    require(absence.get("schema")=="root_GPU0_only_actual_stop_and_absence_v1"
        and absence.get("NVML_project_title_absent_all_devices") is True
        and absence.get("external_processes_untouched") is True
        and absence.get("native_tool",{}).get("native_full_budget_completed") is False
        and absence.get("known_owned_proc_absence") and all(absence["known_owned_proc_absence"].values()),
        "Actual root old-owned-context absence, no invented engineering completion")
    approval=read(ref(GPU0_EXECUTION_APPROVAL_PATH))
    now=datetime.now(timezone.utc)
    start=datetime.fromisoformat(approval["created_at_utc"]);end=datetime.fromisoformat(approval["expires_at_utc"])
    require(start.tzinfo and end.tzinfo and start<=now<end and (end-start).total_seconds()<=604800,
        "Actual unexpired root source/resource review <=7days; stage capability remains separately bounded")
    require(approval.get("schema")=="native_VAL000_GPU0_execution_approval_v1"
        and approval.get("active") is True and approval.get("root_authorizer")=="root_agent"
        and approval.get("physical_gpu")==0 and approval.get("GPU_UUID")==GPU0_UUID
        and approval.get("latest_human_device_policy_ref")==GPU0_POLICY_REF
        and approval.get("actual_old_owned_context_absence_ref")==GPU0_STOP_REF
        and approval.get("new_GPU0_native_guard_ref")==GPU0_NATIVE_GUARD_REF
        and approval.get("GPU0_source_route_reviewed") is True
        and approval.get("root_resource_review_completed") is True
        and approval.get("shared_pool_inherited_lease_route_reviewed") is True
        and approval.get("smaller_7GiB_operational_profile_reviewed") is True
        and approval.get("periodic_overshoot_and_unmeasured_cache_risk_accepted") is True
        and approval.get("reviewed_shared_pool_source_refs")==GPU0_SHARED_SOURCE_REFS
        and approval.get("preserve_external_jobs") is True
        and approval.get("reuse_old_GPU45_authority") is False,
        "Root must issue NEW exact-source GPU0 route/resource approval before bind/admit/launch")
    expected={GPU0_SOURCE_ROOT+"/"+r:ref(GPU0_SOURCE_ROOT+"/"+r)["sha256"] for r in GPU0_REVIEWED_SOURCE_PATHS}
    require(all(ref(r["path"])==r for r in GPU0_SHARED_SOURCE_REFS.values()),"Exact sealed shared pool/guard/bootstrap/policy/stop/renewal refs")
    require(approval.get("reviewed_source_sha256")==expected
        and ref(GPU0_NATIVE_GUARD_REF["path"])==GPU0_NATIVE_GUARD_REF,
        "Exact reviewed native source route and dedicated GPU0 adapter")
    return policy

def original_guard():
    require_gpu0_execution_route_v1()
    install_marker_renewal_v1()
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
        "experiments/results/American_C3_native_val000_postMLM_prediction_route_sharedGPU0_"+phase+"_v1")
def validate_upstream(completed_ref,wait_ref,*,preparation=False):
    completed=read(completed_ref);wait=read(wait_ref)
    if preparation:
        require(wait.get("actual_direct_worker_wait") is True and wait.get("owned_worker_group_ESRCH") is True
            and wait.get("operational_failure") is None
            and wait.get("parent_id")==PARENT and wait.get("stage")=="native_paths"
            and wait.get("root_IO_source_ref")==PREP_IO_REF
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
            and request.get("physical_gpu") in (0,) and caps.get("physical_gpu")==request["physical_gpu"],
            "Exact root actual CUDA cap and eligible physicalGPU0, no invented fit guarantee")
    renewal=read(RENEWAL_CERTIFICATE_REF)
    expiry=datetime.fromisoformat(renewal["renewed_expires_at_utc"].replace("Z","+00:00"))
    require(expiry>datetime.now(timezone.utc)+timedelta(seconds=caps["operational_deadline_seconds"]+600),
        "Genuine renewed marker authority must cover entire planned phase and drain")
    return caps
def make_config(req,caps):
    DEFAULTS=native_contract().DEFAULTS
    cfg=copy.deepcopy(DEFAULTS);phase=req["phase"]
    cfg.update(execution_enabled=True,parent_id=PARENT,phase=phase,
        phase_write_budget_bytes=caps["phase_write_budget_bytes"],
        short_tmp_directory=".runtime/tmp/c3val0shpost/"+("f" if phase=="fit" else "r"),
        child_log_ram_bytes=caps["child_log_ram_bytes"])
    for key in ("max_preparation_file_bytes","max_checkpoint_files","max_checkpoint_bytes","max_embedding_input_bytes"):
        cfg[key]=caps[key]
    return cfg

BRIDGE="research/reviews/native_ARGformer_VAL000_GPU0_shared_execution_20261016_v1/post_mlm_bridge_v1.py"
RAM="research/reviews/C3_native_post_MLM_root_handoff_20261015_v1/ram_downstream_v1.py"
NVML_ACCEPTANCE_REF=dict(path="research/reviews/American_C3_native_MLM_full_fit_v8/nvml_abi_acceptance.root_active_v1.json",
 sha256="69b13084c5275629e194651b71a9db2b233f0ef924e6c12394b2971aa70e195b")
def native_contract():
    return load(ref(REL+"/consumer_contract_v1.py"),"_postMLM980_native_contract")
def source_paths():
    native=native_contract()
    return tuple(dict.fromkeys(('research/reviews/GPU45_shared_downstream_lineage_source_20261016_v1/pool.staged.py',SHARED_NATIVE_RUNTIME,STORAGE_MONITOR,SHARED_POOL_SOURCE,SHARED_POOL_LAUNCHER,SHARED_POOL_BASE,'research/reviews/GPU0_shared_execution_source_20261016_v1/gpu0_runtime_adapter_v1.py','src/arggnn/gpu45_c3cache_h050_pool_v1.py','research/reviews/GPU0_shared_execution_source_20261016_v1/gpu0_bootstrap_renewal_v1.py',GPU0_POLICY_REF['path'],GPU0_STOP_REF['path'],GPU0_EXECUTION_APPROVAL_PATH,RENEWAL_SOURCE_REF['path'],RENEWAL_CERTIFICATE_REF['path'],*native.PROJECT_SOURCES,SOURCE,DRIVER,CONTEXT,RUNTIME_HELPER,
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
    require(value["MLM_optimizer_updates"]==5*(value["datasets"]["mlm/train"]["rows"]//512)==227990
        and value["contrastive_optimizer_updates"]==value["datasets"]["contrastive/train"]["rows"]//4096==CONTRASTIVE_UPDATES
        and value["datasets"]["contrastive/train"]["rows"]%4096==CONTRASTIVE_DROPPED_REMAINDER
        and all(value["datasets"]["contrastive/"+sp]["rows"]==n for sp,n in CONTRASTIVE_ROWS.items())
        and sum(value["datasets"]["query_paths/"+sp]["rows"] for sp in ("train","val"))==10000
        and sum(value["datasets"]["contrastive/"+sp]["rows"] for sp in ("train","val"))==9721236,
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
    require_gpu0_execution_route_v1()
    req=read(record);native=native_contract()
    require(req.get("schema")=="american_c3_native_specific_stage_request_v1"
        and req.get("root_authorizer")=="root_agent" and req.get("execution_authorized") is True
        and req.get("parent_id")==PARENT and req.get("phase") in PHASES
        and req.get("physical_gpu") in (0,) and req.get("GPU_UUID")==GPU0_UUID,"Explicit fresh root GPU0 postMLM phase")
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
    bridge.val000_mlm_phase_v1(refs["MLM_completion"],req["upstream_wait_refs"]["MLM_completion"],
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
    require(caps["checkpoint_live_allowance_bytes"]==(6*2**30+65536 if phase=="fit" else 0)
        and caps["cache_allowance_bytes"]==512*2**20 and caps["temp_allowance_bytes"]==128*2**20
        and caps["child_artifact_allowance_bytes"]==128*2**20
        and caps["max_checkpoint_files"]==2 and caps["max_checkpoint_bytes"]==3*2**30,
        "Explicit smaller operational profile; full science unchanged; monitored failstop without fit/OS quota claim")
    return req,caps,runtime

def launcher_ref_v1():return ref(LAUNCHER_REF["path"])
