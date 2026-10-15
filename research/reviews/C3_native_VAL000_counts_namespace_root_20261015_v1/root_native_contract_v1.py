"""Specific default-inert root contracts; no guard/state/runtime work at import."""
from pathlib import Path
from datetime import datetime,timezone
import copy,hashlib,importlib.util,json,os,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL='research/reviews/C3_native_VAL000_counts_namespace_root_20261015_v1'
SOURCE=REL+"/root_native_contract_v1.py"
DRIVER=REL+"/root_native_driver_v1.py"
CONTEXT=REL+"/context_factory_v1.py"
RUNTIME_HELPER="research/reviews/American_C3_native_model_root_v1/native_runtime_v1.py"
PARENT='american_c3_val_000'
GPU_PHASES=()
CPU_PHASES=("counts",)
PHASES=("counts",)
GPU_MAIN="src/arggnn/american_c3_native_run_v1.py"
CPU_MAIN="src/arggnn/american_c3_native_concat_v1.py"
CORE="483d78bf2ef3c917d65ecf1dde5a7809bfadc21194c9f9a90394ce5c532fdd5e"
BASE="a8049bc103afb1cbbbce6f8fc3707a21de9613b23034599641b688994fc5caa3"
GUARD_REF={'path': 'research/reviews/prospective_980GB_capacity_source_v1/prospective_980gb_v1.py', 'sha256': 'fc71030865e0a3d5da31120d4daa2caa0b2d39ec7dd212b8c46c476b5a866d56'}
POLICY_REF={'path': 'research/reviews/prospective_980GB_capacity_source_v1/activation_policy.root_active_v1.json', 'sha256': 'eb3408004a508136aa434c0a2e02a9d14a5598a96b8483c264c93026f9326018'}
REPAIR_REF=dict(path="research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json",sha256="f6f2d628199e8be73db8b5d90d428570d8afdefeab9dc5cd94bb5eafb39247db")
MEMO_REF=dict(path="research/reviews/resource_guard_hotpath_source_audit_v1/transaction_memo_adapter_v2.py",sha256="c972e9a779d424ceafdb258848614bf40994d1004e2c73fc5cd225ae99621364")
V4_REF=dict(path="src/arggnn/project_budget_v4.py",sha256="a093c7d86f57fee569cb99f4712250e9505a1119f3950fa44282c7ba8c57df47")
LAUNCHER_REF={'path': 'scripts/launch_reserved_gpu_authorized_GPU45_980_v1.py', 'sha256': '0ca3c59e49f5fd55f5a798c99a4a8a6613e71a90bcb3f89a04eb66e7555046ca'}
GPU_AUTH_REF=dict(path="research/reviews/H040_GPU45_migration_root_20261015_v1/GPU45_authorization.json",sha256="0231ae8edb05f8dd34ce127a0de0b97c1dd64fa461ad081e7a83bb92c558d26c")
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
NAMESPACE_MEMO_REF={'path': 'research/reviews/current_mutex_contention_20261015_v1/consumer_namespace_memo_v1.py', 'sha256': '17a1377320f61c3c0cc8d3816d0259f9dae3121ed91bf7206a7ee63fcab8eb7f'}
PREP_IO_REF={'path': 'research/reviews/prospective_980GB_capacity_source_v1/root_VAL_cpu_execution_v2.py', 'sha256': '9f684478aca4d7f4867dc58cc3542bb651559e137a168bf35a9f61570ecc5847'}
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
    guard=original_guard();memo=load(NAMESPACE_MEMO_REF,"_C3_VAL000_counts_consumer_namespace_memo")
    adapter=memo.install_consumer_v1(guard.budget.v4,guard.budget,consumer_role="read_only_consumer",
        expected_source_path=ROOT/V4_REF["path"],expected_source_sha256=V4_REF["sha256"])
    require(adapter.receipt["namespace_extension_schema"]=="resource_guard_consumer_namespace_memo_v1"
        and adapter.receipt["cross_operation_cache"] is False,"Exact pinned namespace consumer memo")
    return guard,adapter
def phase_paths(phase):
    require(phase in PHASES,"Specific registered native stage")
    return (ROOT/REL/"admitted_counts980_v1",
        "experiments/results/American_C3_native_val000_full30mb_counts980_namespace_v1")
def source_paths():
    from arggnn.american_c3_native_contract_v1 import PROJECT_SOURCES
    extras=(SOURCE,DRIVER,CONTEXT,RUNTIME_HELPER,PROC_HELPER,
        "src/arggnn/census_tsinfer_fit_v1.py","src/arggnn/census_tsinfer_observation_v1.py",
        "src/arggnn/biological_candidate.py","src/arggnn/project_budget_v9_consumer.py",
        "scripts/launch_reserved_gpu_full_storage_v1.py","scripts/launch_reserved_gpu_terminal_retired_scopes_v1.py")
    return tuple(dict.fromkeys((*PROJECT_SOURCES,*extras)))
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
            and request.get("physical_gpu") in (4,5) and caps.get("physical_gpu")==request["physical_gpu"],
            "Exact root actual CUDA cap and eligible device, no invented fit guarantee")
    return caps
def make_config(req,caps):
    from arggnn.american_c3_native_contract_v1 import DEFAULTS
    cfg=copy.deepcopy(DEFAULTS);phase=req["phase"]
    cfg.update(execution_enabled=True,parent_id=PARENT,phase=phase,
        phase_write_budget_bytes=caps["phase_write_budget_bytes"],
        short_tmp_directory=".runtime/tmp/c3val0cntns/v1",
        child_log_ram_bytes=caps["child_log_ram_bytes"])
    for key in ("max_preparation_file_bytes","max_checkpoint_files","max_checkpoint_bytes","max_embedding_input_bytes"):
        cfg[key]=caps[key]
    return cfg
def validate_request(record):
    from arggnn import american_c3_native_contract_v1 as native
    req=read(record);require(req.get("schema")=="american_c3_native_specific_stage_request_v1"
        and req.get("root_authorizer")=="root_agent" and req.get("execution_authorized") is True
        and req.get("parent_id")==PARENT and req.get("phase") in PHASES,"Explicit one-parent root stage request")
    native.no_private_refs(req)
    phase=req["phase"];refs=req["refs"]
    required={"counts":(),"fit":("counts",),"embeddings":("counts","fit_completion"),
        "concat":("counts","fit_completion","embedding_completion"),
        "retrieve":("counts","fit_completion","combined_completion"),
        "prediction_seal":("counts","fit_completion","combined_completion","retrieval_completion")}[phase]
    require(all(isinstance(refs.get(k),dict) for k in ("runtime","preparation","preparation_config",*required)),
        "Every actual required upstream reference must be filled before admission")
    caps=validate_caps(read(req["caps_ref"]),req)
    prep,_=validate_upstream(refs["preparation"],req["upstream_wait_refs"]["preparation"],preparation=True)
    pcfg=read(refs["preparation_config"])
    require(pcfg["observation_ref"]==OBS_REF and pcfg["path_sampling_seed"]==93001
        and pcfg["execution_enabled"] is True and prep["query_ancestry_truth_opened"] is False,
        "Frozen original held-out VAL000 per-ARG corpus recipe")
    runtime=read(refs["runtime"])
    require(runtime.get("schema")=="american_c3_native_counts_CPU_runtime_binding_v1" and runtime.get("root_authorizer")=="root_agent"
        and runtime.get("status")=="existing_CPU_metadata_bound_compare_at_execution" and runtime.get("shared_cpu_runtime_ref") is not None,
        "Existing shared CPU metadata binding, actual runtime comparison occurs in consumer")
    read(runtime["shared_cpu_runtime_ref"])
    if phase!="counts":
        counts,_=validate_upstream(refs["counts"],req["upstream_wait_refs"]["counts"])
        require(counts["source_ref"]==ref(CPU_MAIN) and counts["preparation_ref"]==refs["preparation"]
            and counts["preparation_config_ref"]==refs["preparation_config"] and counts["parent_id"]==PARENT
            and counts["query_ancestry_truth_opened"] is False,"Exact actual source-bound native counts")
        if phase in GPU_PHASES or phase=="concat":require(counts["native_training_eligible"] is True,"Complete release batches required")
    for key in ("fit_completion","embedding_completion","combined_completion","retrieval_completion"):
        if refs.get(key) is not None:validate_upstream(refs[key],req["upstream_wait_refs"][key])
    if phase in GPU_PHASES:
        require(ref(runtime["native_python_ref"]["path"])==runtime["native_python_ref"]
            and runtime.get("native_prefix") and runtime.get("legacy_inventory_policy_ref")
            and runtime.get("legacy_runtime_bindings_ref") and runtime.get("native_inventory_ref")
            and runtime.get("bert_base_config_ref"),"Actual executable/prefix/inventory/local base config required")
        read(runtime["bert_base_config_ref"]);read(runtime["legacy_inventory_policy_ref"])
        read(runtime["legacy_runtime_bindings_ref"]);read(runtime["native_inventory_ref"],cap=33554432)
        require(runtime["qualified_physical_gpu"]==req["physical_gpu"] and runtime["qualified_GPU_UUID"]
            and runtime["bert_base_config_ref"]==BASE_CONFIG_REF
            and runtime["native_CPU_qualification_ref"]==NATIVE_CPU_QUAL_REF
            and runtime["native_inventory_ref"]==NATIVE_INVENTORY_REF,
            "Exact existing native inventory and official local BERT qualification")
        cpu=read(NATIVE_CPU_QUAL_REF);gpu=read(runtime["native_GPU_qualification_ref"])
        require(cpu["bert_base_config_ref"]==BASE_CONFIG_REF
            and cpu["executable_sha256"]==runtime["native_python_ref"]["sha256"]
            and local(cpu["executable"])==local(runtime["native_python_ref"]["path"])
            and local(cpu["prefix"])==local(runtime["native_prefix"])
            and cpu["GPU_context_created"] is False and cpu["model_instantiated"] is False,
            "Actual CPU-only offline local BERT qualification, no fit claim")
        require(gpu.get("schema")=="american_c3_native_GPU_runtime_qualification_v1"
            and gpu.get("root_authorizer")=="root_agent" and gpu.get("status")=="qualified"
            and gpu.get("physical_gpu")==req["physical_gpu"]
            and gpu.get("GPU_UUID")==runtime["qualified_GPU_UUID"]
            and gpu.get("native_python_ref")==runtime["native_python_ref"]
            and gpu.get("native_inventory_ref")==runtime["native_inventory_ref"]
            and gpu.get("bert_base_config_ref")==BASE_CONFIG_REF
            and gpu.get("actual_native_runtime_and_GPU_context_checked") is True,
            "Actual separately SHA-bound same-GPU runtime/context qualification required; no invented model fit")
    return req,caps,runtime


# Extra closed-input requirement is observation-only and has no MLM dependency.
_original_validate_request=validate_request
def validate_request(record):
    req,caps,runtime=_original_validate_request(record)
    require(req["phase"]=="counts" and req["refs"]["preparation"]==PREP_REF
        and req["refs"]["preparation_config"]==PREP_CONFIG_REF
        and req["refs"]["preparation_close"]==PREP_CLOSE_REF
        and req["upstream_wait_refs"]=={"preparation":PREP_WAIT_REF},
        "Only the actual closed VAL000 observation-only input; no TRAIN MLM/weights dependency")
    close=read(PREP_CLOSE_REF);proof=close.get("value",{})
    require(close.get("kind")=="concurrent_completed_scope_close_receipt_v1"
        and close.get("key")=="H041/American-C3-val000-full30mb-native_paths-shared-980-v1"
        and proof.get("future_writes_and_relaunch_revoked") is True
        and proof.get("request",{}).get("completed_ref")==PREP_REF
        and proof.get("request",{}).get("direct_wait_ref")==PREP_WAIT_REF,
        "Actual completed preparation permanently closed READONLY")
    observation=read(OBS_REF)
    require(observation["parent_id"]==PARENT and observation["split"]=="val"
        and observation["sequence_length_bp"]==30000000 and observation["query_ancestry_truth_in_inputs"] is False,
        "Original held-out VAL000 observed input, never relabelled TRAIN")
    require(caps["max_preparation_file_bytes"]==close["retained_bytes"]
        and caps["source_input_bound_ref"]==PREP_CLOSE_REF,
        "Per-file bound comes from actual closed total retained-scope bytes")
    require(set(req["refs"])=={"runtime","preparation","preparation_config","preparation_close"},
        "CPU counts accepts no model/checkpoint/completion inputs")
    return req,caps,runtime

def source_paths():
    from arggnn.american_c3_native_contract_v1 import PROJECT_SOURCES
    return tuple(dict.fromkeys((*PROJECT_SOURCES,SOURCE,DRIVER,CONTEXT,RUNTIME_HELPER,
        "src/arggnn/census_tsinfer_fit_v1.py","src/arggnn/census_tsinfer_observation_v1.py",
        "src/arggnn/biological_candidate.py","src/arggnn/project_budget_v9_consumer.py",
        "scripts/launch_reserved_gpu_full_storage_v1.py","scripts/launch_reserved_gpu_terminal_retired_scopes_v1.py",
        "research/reviews/American_C3_native_MLM_amendment_v12/owned_lifecycle_v2.py",
        "research/reviews/American_C3_native_MLM_full_source_v5/pidfd_backend_v1.py")))
