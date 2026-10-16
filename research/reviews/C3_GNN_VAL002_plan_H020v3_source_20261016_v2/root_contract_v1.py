"""Current980 GNN root contract. Metadata only; all actions require explicit root modes."""
from pathlib import Path
import copy,hashlib,importlib.util,json,os,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/C3_GNN_VAL002_plan_H020v3_source_20261016_v2"
SOURCE=REL+"/root_contract_v1.py";DRIVER=REL+"/root_driver_v1.py";ENTRY=REL+"/worker_entry_v1.py"
CANONICAL="src/arggnn/american_c3_gnn_current980_exec_v1.py"
GPU_MODULE="arggnn.american_c3_gnn_current980_exec_v1"
GPU_MAIN=CANONICAL;CPU_MAIN=ENTRY
BASE_HELPER=dict(path="research/reviews/C3_native_postMLM980_root_source_20261015_v1/root_native_contract_v1.py",
 sha256="aeb992d49d632003cbd971432afa607cc8e8ba2b0658f57501fcc435c8d74b69")
def _load_base():
 p=ROOT/BASE_HELPER["path"]
 if hashlib.sha256(p.read_bytes()).hexdigest()!=BASE_HELPER["sha256"]:raise ValueError("Exact reviewed common guard/metadata utilities")
 sp=importlib.util.spec_from_file_location("_C3GNN_current980_common",p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
_base=_load_base()
require,local,ref,read,save,load,active=(_base.require,_base.local,_base.ref,_base.read,_base.save,_base.load,_base.active)
CORE,BASE,GUARD_REF,POLICY_REF,REPAIR_REF,MEMO_REF,V4_REF,GPU_AUTH_REF=(_base.CORE,_base.BASE,_base.GUARD_REF,_base.POLICY_REF,_base.REPAIR_REF,_base.MEMO_REF,_base.V4_REF,_base.GPU_AUTH_REF)

GUARD_REF={'path': 'research/reviews/prospective980_H020_legacy_source_20261016_v3/prospective_980gb_H020_v3.py', 'sha256': 'ec475ecf59f0335397ee8b310cd0cb406ce4e03161e7c791861c6583799e6635'}
POLICY_REF={'path': 'research/reviews/prospective980_H020_root_activation_20261016_v3/activation_policy.root_active_v1.json', 'sha256': '7360bd28f57c8eaef19336ccd77eba193afae554e98d2ef0726e8ec879d7e582'}
RENEWAL_REF={'path': 'research/reviews/H039_marker_repair_renewal_source_20261016_v1/marker_repair_renewal_v1.py', 'sha256': '491236877e8d71cae8bd7619880858ac501c57dd93e59479ca32dacacab20468'}
CERT_REF={'path': 'research/reviews/H039_marker_repair_renewal_source_20261016_v1/renewal_certificate.root_active_v1.json', 'sha256': '66006202094d7a3ef2b29cbec28eb93f334505b72589f09d86e4b57498162db7'}
SUPERVISOR_HELPER_REF={'path': 'src/arggnn/american_c3_native_run_v1.py', 'sha256': '65653b2522978d20341e35ab9e36610a75024a29931026961a5bc4b759fab2cc'}
PIDFD_REF={'path': 'research/reviews/American_C3_native_MLM_full_source_v5/pidfd_backend_v1.py', 'sha256': '61be5d89357d1b95611e0ba26dd39d381ececf3ce6a36c347507cdc062373eca'}
OWNED_REF={'path': 'research/reviews/American_C3_native_MLM_amendment_v12/owned_lifecycle_v2.py', 'sha256': '93d71acc40f1037d0b37acb2ca0859300ea4bd9741409e9bd00ca21bf100d633'}

PREP_REF={'path': 'experiments/results/American_C3_val002_full30mb_native_paths_H020v3_v1/completed.json', 'sha256': 'aa5afa541e495ea4eda2b39fe522673a277c9759278bc40fa91f29206fcf559a'}
WAIT_REF={'path': 'research/reviews/American_C3_VAL002_native_paths_H020v3_source_20261016_v1/admitted_native_paths_H020v3_v1/actual_wait.json', 'sha256': '91fabc93dcd8692bd6a9d6b46bd34fd42d1c334a3a8fefbd42b10f2d9f3a5004'}
TERMINAL_REF={'path': 'research/reviews/American_C3_VAL002_native_paths_H020v3_source_20261016_v1/actual_tool_terminal.root_v1.json', 'sha256': '902fef18f521f7aa5f5a7899abba3ae944d6ddda253934ecc0f101494cf9545b'}

PHASE="plan_val_002"
PARENT="american_c3_val_002"
PHASES=CPU_PHASES=(PHASE,)
GPU_PHASES=()
GPU_MODULE=GPU_MAIN=None
CPU_MAIN=ENTRY
CONTEXT=RUNTIME_HELPER=ENTRY
STATUSES={PHASE:"completed_C3_GNN_current980_plan_val_002"}
LAUNCHER_REF=None
GPU_AUTH_REF=None
CANONICAL=None
SOURCE_PINS=REL+"/source_pins.v2.json"
FIXED_CONFIG=REL+"/plan_config.fixed.v1.json"
BASE_CONFIG_REF=dict(path="configs/census_tsinfer_fit_v1.json",sha256="7c2436d5479e68beb1a8f3edb04ca0ee729a23be35ca2d7893827c0db68e3b00")
RUNTIME_REF=dict(path="research/reviews/ARGformer_American_C3_implementation_20261015_v1/shared_cpu_runtime_metadata.json",sha256="e0944a1c2c4b2a26d2edfe68fad11db691b4b6853ebeddd8e3b79692b43fa602")
HOOK_REF=dict(path="research/reviews/American_C3_PyG_execution_source_20261015_v1/execution_hooks_v1.py",sha256="6546cc4498a8bcbac4919ae08ce75b23a0ac8e5b350da25e3fa31a967286bfa1")
TRAIN_SOURCE_REF=dict(path="src/arggnn/american_c3_gnn_train_v1.py",sha256="cce52c9f1776e46c7005509544bfb3743acaf860f256e546e3ac5fc7cb518df2")
PROC_HELPER=_base.PROC_HELPER

def phase_paths(stage):
    require(stage==PHASE,"Only original VAL002 observation-plan stage")
    return ROOT/REL/("admitted_"+PHASE+"_v1"),"experiments/results/C3_GNN_VAL002_plan_H020v3_v2"

def hooks():return load(HOOK_REF,"_C3_VAL002_unchanged_plan_hook")

def source_paths():
    pins=read(ref(SOURCE_PINS),cap=33554432)
    for r in pins["source_refs"]:require(ref(r["path"])==r,"Exact complete plan/runtime/source metadata closure")
    return tuple(r["path"] for r in pins["source_refs"])

def validate_bindings(stage,b):
    require(stage==PHASE and set(b)=={"parent_id","plan_config","native_preparation_ref","preparation_wait_ref"}
        and b["parent_id"]==PARENT and b["native_preparation_ref"]==PREP_REF
        and b["preparation_wait_ref"]==WAIT_REF,"Only actual new VAL002 preparation and original parent")
    prep=read(PREP_REF);wait=read(WAIT_REF);terminal=read(TERMINAL_REF)
    require(prep["status"]=="completed_native_C3_path_preparation" and prep["parent_id"]==PARENT
        and prep["query_focal_count"]==10000 and prep["query_ancestry_truth_opened"] is False
        and prep["class_order"]==["EUR","EAS","AFR"]
        and prep["genealogy_kind"]=="inferred_tsinfer_tsdate_CURRENT_RECONSTRUCTION_not_simulated_true_ARG",
        "Actual same-parent native focal preparation, no native fit or label dependency")
    require(wait["completed_ref"]==PREP_REF and wait["returncode"]==0
        and wait["actual_direct_worker_wait"] is True and wait["owned_worker_group_ESRCH"] is True
        and wait["operational_failure"] is None,"Actual new preparation direct wait0 and drain")
    require(terminal["schema"]=="root_actual_external_tool_terminal_v1"
        and terminal["actual_terminal_consumed"] is True and terminal["actual_exit_code"]==0
        and terminal["completed_ref"]==PREP_REF and terminal["direct_wait_ref"]==WAIT_REF,
        "Genuinely consumed preparation outer terminal0")
    cfg=b["plan_config"]
    require(cfg==read(ref(FIXED_CONFIG)) and cfg["execution_enabled"] is True
        and cfg["native_preparation_ref"]==PREP_REF and cfg["focal_registry_ref"]==prep["focals_ref"]
        and cfg["observation_ref"]==prep["observation_ref"],"Exact unchanged original VAL plan recipe using actual new refs")
    obs=read(prep["observation_ref"])
    require(obs["parent_id"]==PARENT and obs["split"]=="val" and obs["sequence_length_bp"]==30000000
        and obs["sample_unit"]=="haploid" and obs["counts_haploid"]=={"AFR":200,"EUR":200,"ASIA":200,"ADMIX":1000}
        and obs["class_order"]==["EUR","EAS","AFR"] and obs["query_ancestry_truth_in_inputs"] is False
        and obs["genealogy_kind"]==prep["genealogy_kind"],"Genuine sanitized original full30Mb observation metadata")
    return b

def make_config(req,caps):
    return dict(schema="C3_GNN_current980_operational_config_v1",execution_enabled=True,phase=PHASE,
        output_cap_bytes=caps["reservation_cap_bytes"],short_tmp_directory=REL+"/admitted_"+PHASE+"_v1/worker_tmp",
        child_log_ram_bytes=caps["child_log_ram_bytes"])

def validate_request(record):
    req=read(record,cap=33554432)
    require(req.get("schema")=="C3_GNN_VAL002_plan_H020v3_request_v1" and req.get("active") is True
        and req.get("root_authorizer")=="root_agent" and req.get("phase")==PHASE
        and req.get("native_MLM_dependency") is False and req.get("TRAIN_dependency") is False
        and req.get("physical_gpu") is None and req.get("GPU_UUID") is None,"Fresh CPU-only observation-plan authority")
    require(req["runtime_ref"]==RUNTIME_REF and req["base_fit_config_ref"]==BASE_CONFIG_REF
        and req["scientific_train_source_ref"]==TRAIN_SOURCE_REF and req["hook_ref"]==HOOK_REF
        and req["source_pins_ref"]==ref(SOURCE_PINS)
        and req["operational_adapter_ref"]==ref(DRIVER)
        and req["preparation_outer_terminal_ref"]==TERMINAL_REF,
        "Exact qualified runtime/science and actual operational/prerequisite sources")
    source_paths()
    validate_bindings(PHASE,req["bindings"])
    caps=read(req["caps_ref"]);expected=read(ref(REL+"/caps.inactive.json"));expected.update(active=True,phase=PHASE)
    require(caps==expected,"All original32MiB/20MiB/3600s/RSS operational cap values unchanged")
    from datetime import datetime,timezone,timedelta
    end=datetime.fromisoformat(read(CERT_REF)["renewed_expires_at_utc"])
    require(datetime.now(timezone.utc)+timedelta(seconds=caps["operational_deadline_seconds"]+600)<end,
        "Existing renewal covers actual prospective stage and drain; no expiry extension")
    return req,caps,read(RUNTIME_REF)

def original_guard():
    # Install exact marker renewal before every actual marker/guard consumer.
    renewal=load(RENEWAL_REF,"_VAL002_plan_exact_marker_renewal")
    require(renewal.install_renewal_v1(CERT_REF)==CERT_REF,"Exact existing renewal")
    for r in (GUARD_REF,POLICY_REF,REPAIR_REF,V4_REF):require(ref(r["path"])==r,"Exact active H020v3 authorities")
    sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"scripts")]
    guard=load(GUARD_REF,"_VAL002_plan_actual_H020v3_guard")
    guard.configure_v2(REPAIR_REF,policy_ref=POLICY_REF)
    require(guard.MODE=="prospective_980gb_H020_v3"
        and all(getattr(guard.budget.v4,n).__module__==guard.budget.v4.__name__ for n in ("canonical","file_bytes","mutex")),
        "Active H020v3; root mutations retain original unwrapped v4")
    return guard

def consumer_guard():
    guard=original_guard();memo=load(MEMO_REF,"_VAL002_plan_original_consumer_memo2")
    adapter=memo.install(guard.budget.v4,expected_source_path=ROOT/V4_REF["path"],expected_source_sha256=V4_REF["sha256"])
    return guard,adapter

def bind(template_ref):
    req=read(template_ref,cap=33554432)
    require(req.get("active") is False and req.get("phase")==PHASE,"Inactive explicit ROOT request")
    validate_bindings(PHASE,req["bindings"])
    folder=ROOT/REL
    require(not (folder/("request_"+PHASE+".root_active_v1.json")).exists(),"No bind replay")
    caps=read(ref(REL+"/caps.inactive.json"));caps.update(active=True,phase=PHASE)
    caps_ref=save(folder,"caps_"+PHASE+".root_active_v1.json",caps)
    req.update(active=True,caps_ref=caps_ref)
    request_ref=save(folder,"request_"+PHASE+".root_active_v1.json",req)
    validate_request(request_ref)
    return dict(request_ref=request_ref,actual_admission=False,actual_execution=False)
