"""Exact pending full30Mb VAL002 native-path H020v3 operational successor. Import/default plan is source-only.

Root alone admits, launches and closes. One phase at a time, after predecessor
close; frozen original producer CLIs/science and original mutation API retained.
"""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,fcntl,hashlib,importlib.util,json
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/American_C3_VAL002_native_paths_H020v3_source_20261016_v1"
HERE=ROOT/REL
SOURCE=REL+"/root_packet_v1.py"
BASIS="research/reviews/prospective_980GB_capacity_source_v1"
BASE_REFS_REF=dict(path=BASIS+"/frozen_refs.json",sha256="a345ed40e95c3d343ee2e2c96b5b57346be3ba4408105fc5c7ed65805908c8dc")
CONTRACT_REF=dict(path=BASIS+"/shared_contract.inactive.json",sha256="50af87a85ad1f231faa846d0a21ee2017e05594dee32bf29d73a9c79013b5f89")
FORECAST_REF=dict(path=BASIS+"/storage_forecast.json",sha256="bb379b4530b29fb82c4a6aea565750ec56e4dd55b7b2a20f03cc5a878aa134c4")
REGISTRY_REF=dict(path="research/reviews/ARGformer_American_C3_implementation_20261015_v1/parent_seed_registry_inactive.json",sha256="608da56508757646a0584cb653f7f6af4c910f05d560d6da1b99c0c1ea691f9a")
MEMO_REF=dict(path="research/reviews/resource_guard_hotpath_source_audit_v1/transaction_memo_adapter_v2.py",sha256="c972e9a779d424ceafdb258848614bf40994d1004e2c73fc5cd225ae99621364")
V4_REF=dict(path="src/arggnn/project_budget_v4.py",sha256="a093c7d86f57fee569cb99f4712250e9505a1119f3950fa44282c7ba8c57df47")
NAMESPACE_MEMO_REF={'path': 'research/reviews/current_mutex_contention_20261015_v1/consumer_namespace_memo_v1.py', 'sha256': '17a1377320f61c3c0cc8d3816d0259f9dae3121ed91bf7206a7ee63fcab8eb7f'}
BASE_CONFIG_REFS={
 "simulate":dict(path=BASIS+"/admitted_simulate_v1/config.json",sha256="3d83bdfc00634d83ee53efedfc9392d6fd195ff6ecc097cd71437ccba1c44e39"),
 "infer":dict(path=BASIS+"/admitted_infer_v1/config.json",sha256="e36a23293e255c755934016b57f97145ed790541eafa23e5ac0c68c924f5d71a"),
 "native_paths":dict(path=BASIS+"/admitted_native_paths_v1/config.json",sha256="c7c95390e12daa2216babc8ccf82d04227a2c96f4e9af8a04dfb626443fa3146")}
STAGES=("native_paths",)
PARENTS=("american_c3_val_002",)
PARENT=OUTPUT=SEED_ROW=None
PRODUCERS={"simulate":"src/arggnn/american_c3_simulate_v1.py","infer":"src/arggnn/american_c3_observe_v3.py","native_paths":"src/arggnn/american_c3_paths_v1.py"}
DIRS={"simulate":"simulation","infer":"inference","native_paths":"native_paths"}
COMMON="src/arggnn/american_c3_common_v1.py"
AUTHOR=".runtime/author_sources/argformer_f7545be7bca0/src/data/prepare_data_pretrain.py"
SIM_RUNTIME="research/reviews/American_C3_root_execution_20261015_v1/admitted_full30mb_v2/runtime.json"
INFER_RUNTIME="research/reviews/American_C3_observation_root_execution_20261015_v3/inference_cpu_runtime_metadata.json"
OVERLAY="research/reviews/prospective980_H020_legacy_source_20261016_v3/prospective_980gb_H020_v3.py"
POLICY="research/reviews/prospective980_H020_root_activation_20261016_v3/activation_policy.root_active_v1.json"
REPAIR="research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json"
BUDGET="scripts/project_budget_v9.py"
LAUNCHER="scripts/launch_reserved_gpu_v6.py"
ADMISSION_LOCK=HERE/"root_admission.lock"
SIMULATION_COMPLETION_REF={'path': 'experiments/results/American_C3_val002_full30mb_shared_v1/simulation/completed.json', 'sha256': 'fd929fb58d3816af39d938aac4423b2687082158261f8baf40d3d02be6e681c5'}
SIMULATION_OBSERVATION_REF={'path': 'experiments/results/American_C3_val002_full30mb_shared_v1/simulation/observed/manifest.json', 'sha256': 'f564935f49ff2eb2d3717b32bad4269183add2fa69c972fd55a0ae1ce49b54bb'}
INFERENCE_COMPLETION_REF={'path': 'experiments/results/American_C3_val002_full30mb_shared_v1/inference/completed.json', 'sha256': 'ea8172b2774e66b6bc4761feec1b86045da0cf7b27219a02d28cb351e3b201ee'}
INFERENCE_OBSERVATION_REF={'path': 'experiments/results/American_C3_val002_full30mb_shared_v1/inference/observed/manifest.json', 'sha256': 'a36eb6faad850b963c252ad4ac0efd57f9d95df60a626ef31dc41fcc55779ba0'}
INFERENCE_WAIT_REF={'path': 'research/reviews/American_C3_remaining_VAL_namespace_memo_root_20261015_v1/admitted_val002_infer_v1/actual_wait.json', 'sha256': 'e6d1b8c29084d7992465b0037b1ef1d7ef8e4ebab0492d9aa0f35ba4fea504d3'}
INFERENCE_CLOSE_REF={'path': 'research/reviews/American_C3_remaining_VAL_namespace_memo_root_20261015_v1/closed_val002_infer_v1/close_receipt.json', 'sha256': '29e41e92e48bec76439deb559171f5a91e3ff18dc6f9e5f2f7a499e93a2dd036'}
PROTOCOL_REF={'path': 'research/reviews/American_C3_VAL30mb_shared_source_v1/native_per_ARG_protocol_audit.json', 'sha256': '16c624deeb1af091312691ea319e2a74739ac7bb6ff92dd483890185f70c4688'}
GUARD3_REF={'path': 'research/reviews/prospective980_H020_legacy_source_20261016_v3/prospective_980gb_H020_v3.py', 'sha256': 'ec475ecf59f0335397ee8b310cd0cb406ce4e03161e7c791861c6583799e6635'}
POLICY3_REF={'path': 'research/reviews/prospective980_H020_root_activation_20261016_v3/activation_policy.root_active_v1.json', 'sha256': '7360bd28f57c8eaef19336ccd77eba193afae554e98d2ef0726e8ec879d7e582'}
RENEWAL_REF={'path': 'research/reviews/H039_marker_repair_renewal_source_20261016_v1/marker_repair_renewal_v1.py', 'sha256': '491236877e8d71cae8bd7619880858ac501c57dd93e59479ca32dacacab20468'}
RENEWAL_CERT_REF={'path': 'research/reviews/H039_marker_repair_renewal_source_20261016_v1/renewal_certificate.root_active_v1.json', 'sha256': '66006202094d7a3ef2b29cbec28eb93f334505b72589f09d86e4b57498162db7'}
READINESS_REF={'path': 'research/reviews/American_C3_VAL002_native_paths_H020v3_source_20261016_v1/readiness.v1.json', 'sha256': '4e27143e0037220b1aa9c7d45e5c4484fd81bc54668fac7802fd08195dc96109'}
FRESH_OUTPUT="experiments/results/American_C3_val002_full30mb_native_paths_H020v3_v1"

def require(ok,message):
    if not ok:raise ValueError(message)

def local(value):
    p=Path(value);p=p if p.is_absolute() else ROOT/p
    require(".." not in p.parts and p.resolve()==p,"Canonical own metadata path")
    p.relative_to(ROOT)
    return p

def ref(path):
    p=local(path)
    return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def read(record):
    require(isinstance(record,dict) and set(record)=={"path","sha256"} and ref(record["path"])==record,"Exact source/metadata SHA")
    raw=local(record["path"]).read_bytes()
    require(len(raw)<=1048576,"Bounded source/metadata JSON")
    return json.loads(raw)

def save(folder,name,value):
    raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    require(len(raw)<=65536,"Compact root metadata")
    with (folder/name).open("xb") as stream:stream.write(raw)
    return ref(folder/name)

def upstream_observation(stage,completion_ref,wait_ref):
    require(stage=="native_paths" and completion_ref==INFERENCE_COMPLETION_REF and wait_ref==INFERENCE_WAIT_REF,
        "Exact original VAL002 inferred predecessor only")
    ready=read(READINESS_REF)
    require(ready["parent_id"]==PARENT=="american_c3_val_002" and ready["source_only"] is True,"Exact source readiness record")
    simulation=read(SIMULATION_COMPLETION_REF)
    require(simulation["status"]=="completed_C3_simulation_reconstruction" and simulation["parent_id"]==PARENT
        and simulation["sample_unit"]=="haploid" and simulation["class_order"]==["EUR","EAS","AFR"]
        and simulation["sites"]==229909 and simulation["observation_ref"]==SIMULATION_OBSERVATION_REF,
        "Original full VAL002 completed simulation metadata")
    simulated=read(SIMULATION_OBSERVATION_REF)
    require(simulated["parent_id"]==PARENT and simulated["sequence_length_bp"]==30000000
        and simulated["query_ancestry_truth_in_inputs"] is False,"Original observed simulation lineage")
    completed=read(completion_ref);wait=read(wait_ref);close=read(INFERENCE_CLOSE_REF)
    require(wait.get("returncode")==0 and wait.get("actual_direct_worker_wait") is True
        and wait.get("completed_ref")==completion_ref and wait.get("owned_worker_group_ESRCH") is True
        and wait.get("operational_failure") is None and wait.get("parent_id")==PARENT,
        "Original inferred predecessor actual wait0 and owned drain")
    require(close["key"]==predecessor_key(stage) and close["value"]["future_writes_and_relaunch_revoked"] is True
        and close["value"]["proof"]["completed_ref"]==completion_ref,"Original inferred predecessor closed, never reused")
    require(completed["status"]=="completed_current_C3_inference" and completed["observation_ref"]==INFERENCE_OBSERVATION_REF,
        "Successful exact inferred predecessor metadata")
    m=read(INFERENCE_OBSERVATION_REF)
    require(m["parent_id"]==PARENT and m["split"]=="val" and m["sequence_length_bp"]==30000000
        and m["sample_unit"]=="haploid" and m["class_order"]==["EUR","EAS","AFR"]
        and m["counts_haploid"]==dict(AFR=200,EUR=200,ASIA=200,ADMIX=1000)
        and m["query_ancestry_truth_in_inputs"] is False and m["sites_and_sample_identities_verified"] is True
        and m["source_observation_ref"]==SIMULATION_OBSERVATION_REF
        and m["genealogy_kind"]=="inferred_tsinfer_tsdate_CURRENT_RECONSTRUCTION_not_simulated_true_ARG",
        "Exact original all1600 observed full30Mb inferred VAL002 identity")
    # No truth, genotype, tree, NPZ, path or vocabulary payload is followed here.
    return INFERENCE_OBSERVATION_REF

def source_closure(stage,pins):
    paths=[COMMON,PRODUCERS[stage]]
    if stage=="infer":paths.append(PRODUCERS["simulate"])
    if stage=="native_paths":paths.append(AUTHOR)
    return [pins[p] for p in paths]

def _admit_locked(stage,policy_ref,*,completion_ref=None,wait_ref=None):
    """Root-only explicit current-overlay admission; never called by source checks."""
    require(Path.cwd().resolve()==ROOT,"Repository execution only")
    pins=frozen();policy_ref=policy_ref or pins[POLICY];require(POLICY in pins and policy_ref==pins[POLICY],"Exact active root prospective980 policy")
    guard=load_guard_v1(consumer=False)
    with guard.budget.v4.mutex(ROOT):state=guard._state_locked(pins[BUDGET]["sha256"],pins[LAUNCHER]["sha256"],fresh=True)
    require(predecessor_key(stage) in state[1],"Previous original phase/parent native_paths must be genuinely CLOSED before a new reservation")
    if stage!="simulate" and completion_ref is None and wait_ref is None:completion_ref,wait_ref=upstream_refs(stage)
    cfg=configuration(stage,completion_ref=completion_ref,wait_ref=wait_ref,active=True)
    packet,output,job=identity(stage)
    require(not packet.exists() and not local(output).exists(),"Fresh stage, no replay/resume")
    packet.mkdir()
    config_ref=save(packet,"config.json",cfg)
    contract_ref=save(packet,"shared_contract.json",contract())
    closure=source_closure(stage,pins);runtime=pins[INFER_RUNTIME if stage=="infer" else SIM_RUNTIME]
    consumers=[*closure,config_ref,runtime,ref(SOURCE),BASE_REFS_REF,contract_ref,
        FORECAST_REF,BASE_CONFIG_REFS[stage],\
        pins["research/reviews/ARGformer_American_C3_implementation_20261015_v1/parent_seed_registry_inactive.json"],*state[3].values(),
        *[pins[p] for p in (OVERLAY,POLICY,REPAIR,"research/reviews/H039_missing_marker_repair_v1/marker_repair_v1.py",
          "research/reviews/H039_failed_scope_capacity_v1/failed_scope_capacity_v1.py",
          "research/reviews/H039_failed_scope_capacity_v2/failed_scope_capacity_v2.py","src/arggnn/project_budget_v9_consumer.py")]]
    consumers.extend([REGISTRY_REF,MEMO_REF,V4_REF,NAMESPACE_MEMO_REF,ref(HERE/"source_manifest.json"),*operational_refs()])
    consumers.extend([*readiness_refs(),RENEWAL_REF,RENEWAL_CERT_REF])
    if completion_ref is not None:consumers.extend([completion_ref,wait_ref])
    consumers=list({(r["path"],r["sha256"]):r for r in consumers}.values())
    require(len(consumers)<=64,"Existing64 direct source/JSON bound")
    now=datetime.now(timezone.utc);expiry=(now+timedelta(hours=22)).isoformat()
    request=dict(key="H041/"+job,output=str(ROOT/output),auxiliary_scopes=[str(packet/"worker_aux")],
        reserved_bytes=cfg["output_cap_bytes"],consumer_bindings=consumers)
    authority=guard.operation_draft_v1("reserve",request,expected_before=guard._pins(state[2]),bindings=state[3])
    authority.update(active=True,expires_at_utc=expiry)
    authority_ref=save(packet,"reserve_authority.json",authority)
    receipt=guard.apply_reserve_v1(authority_ref,expected_source_sha256=pins[BUDGET]["sha256"],expected_launcher_sha256=pins[LAUNCHER]["sha256"])
    save(packet,"reserve_receipt.json",receipt)
    storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,accounting_mode=guard.MODE,
        job_id=job,output_directory=output,reserved_new_bytes=cfg["output_cap_bytes"],reservation=receipt["value"],
        expires_at_utc=expiry,completed_overlay_policy_ref=policy_ref,initial_ledger_pins=receipt["initial_ledger_pins"],
        budget_components=dict(implementation=pins[BUDGET],launcher=pins[LAUNCHER],consumer_adapter=pins[OVERLAY]))
    spec=dict(root_authorizer="root_agent",hypothesis_id="H041",job_id=job,parent_id=PARENT,split="val",stage=stage,
        execution_authorized=True,GPU_authorized=False,source_ref=pins[PRODUCERS[stage]],source_closure_refs=closure,
        config_ref=config_ref,runtime_ref=runtime,storage_adapter_ref=pins[OVERLAY],marker_repair_authority_ref=pins[REPAIR],
        completed_overlay_policy_ref=policy_ref,shared_evaluation_contract_ref=contract_ref,
        driver_ref=ref(SOURCE),storage=storage,output_directory=output,created_at_utc=now.isoformat(),expires_at_utc=expiry,
        upstream_completion_ref=completion_ref,upstream_actual_wait_ref=wait_ref,
        query_ancestry_truth_payload_access_authorized=False,TEST_access_authorized=False,GNN_optimization_authorized=False,
        consumer_memo_ref=MEMO_REF,consumer_namespace_memo_ref=NAMESPACE_MEMO_REF,original_v4_ref=V4_REF,registry_ref=REGISTRY_REF,parent_seed_row=dict(SEED_ROW),
        worker_bootstrap_ref=ref(HERE/"worker_bootstrap_v1.py"),IO_supervisor_ref=ref(HERE/"root_execution_v1.py"),
        operational_source_manifest_ref=ref(HERE/"source_manifest.json"))
    admission_ref=save(packet,"admission.json",spec)
    admitted_ref=save(packet,"admitted.json",dict(stage=stage,parent_id=PARENT,admission_ref=admission_ref,driver_ref=ref(SOURCE)))
    return dict(admitted_ref=admitted_ref,admission_ref=admission_ref,projected_bytes=receipt["projected_bytes"],
        warning_active=receipt["warning_active"],warning_message=receipt["warning_message"],
        remaining_to_human_limit_bytes=receipt["remaining_to_human_limit_bytes"])

def select_parent(parent):
    global PARENT,OUTPUT,SEED_ROW
    require(parent!="american_c3_val_001","VAL001 is excluded: preserve its original live/admitted route without replay")
    require(parent in PARENTS,"Only frozen remaining VAL002–004 parents")
    registry=read(REGISTRY_REF)
    rows=[r for r in registry["parents"] if r["parent_id"]==parent and r["split"]=="val"]
    require(len(rows)==1,"Unique original independent VAL registry row")
    PARENT=parent;SEED_ROW=rows[0]
    OUTPUT="experiments/results/American_C3_val"+parent[-3:]+"_full30mb_shared_v1"
    return dict(SEED_ROW)

def operational_refs():
    manifest=read(ref(HERE/"source_manifest.json"))
    require(manifest["source_only"] is True,"Inert root source manifest")
    require(all(ref(p)==dict(path=p,sha256=h) for p,h in manifest["source_sha256"].items()),
            "Every frozen operational source changed")
    return [dict(path=p,sha256=h) for p,h in manifest["source_sha256"].items()]

def frozen():
    value=read(BASE_REFS_REF)["refs"]
    require(all(ref(p)==r for p,r in value.items()),"Original frozen scientific/source/runtime pins")
    for r in [REGISTRY_REF,CONTRACT_REF,FORECAST_REF,MEMO_REF,V4_REF,NAMESPACE_MEMO_REF,*BASE_CONFIG_REFS.values(),
              GUARD3_REF,POLICY3_REF,RENEWAL_REF,RENEWAL_CERT_REF,PROTOCOL_REF,READINESS_REF]:
        require(ref(r["path"])==r,"Exact original science or new operational authority")
    operational_refs()
    policy=read(POLICY3_REF)
    require(policy["schema"]=="prospective_980GB_H020_credit_policy_v3" and policy["active"] is True
        and policy["source_ref"]==GUARD3_REF and policy["scheduling_ceiling_bytes"]==980000000000
        and policy["warning_threshold_bytes"]==950000000000 and policy["human_limit_bytes"]==1000000000000,
        "Exact active H020v3 policy, no old future admission")
    value[OVERLAY]=GUARD3_REF
    value[POLICY]=POLICY3_REF
    return value

def identity(stage):
    require(PARENT=="american_c3_val_002" and stage=="native_paths","Only one exact pending preparation stage")
    return HERE/"admitted_native_paths_H020v3_v1",FRESH_OUTPUT,"American-C3-val002-full30mb-native_paths-H020v3-v1"

def close_folder(stage):
    identity(stage)
    return HERE/("closed_val"+PARENT[-3:]+"_"+stage+"_v1")

def predecessor_key(stage):
    require(stage=="native_paths","No simulation, inference or other-parent admission")
    return "H041/American-C3-val002-full30mb-infer-shared-980-v1"

def upstream_refs(stage):
    require(stage=="native_paths","Only original VAL002 inferred predecessor")
    return INFERENCE_COMPLETION_REF,INFERENCE_WAIT_REF

def configuration(stage,*,completion_ref=None,wait_ref=None,active=False):
    frozen();require(PARENT in PARENTS and stage in STAGES,"Exact selected original VAL row")
    original=read(BASE_CONFIG_REFS[stage]);cfg=dict(original)
    if stage=="simulate":
        require(completion_ref is None and wait_ref is None,"No previous simulation input")
        cfg.update(parent_id=PARENT,ancestry_seed=SEED_ROW["ancestry_seed"],mutation_seed=SEED_ROW["mutation_seed"])
        allowed={"execution_enabled","parent_id","ancestry_seed","mutation_seed"}
    else:
        if completion_ref is None and wait_ref is None:completion_ref,wait_ref=upstream_refs(stage)
        cfg["observation_ref"]=upstream_observation(stage,completion_ref,wait_ref)
        allowed={"execution_enabled","observation_ref"}
        if stage=="native_paths":
            cfg["path_sampling_seed"]=SEED_ROW["path_sampling_seed"];allowed.add("path_sampling_seed")
    cfg["execution_enabled"]=active
    require({k for k in cfg if cfg[k]!=original[k]}<=allowed and set(cfg)==set(original),
            "No silent change from actual VAL000 scientific configuration")
    return cfg

def contract():
    value=read(CONTRACT_REF)
    value.update(parent_id=PARENT,parent_seed_row=dict(SEED_ROW),focal_sampling_seed=SEED_ROW["path_sampling_seed"])
    return value

def load_guard_v1(*,consumer=False):
    install_renewal_v1()
    pins=frozen()
    loader=importlib.util.spec_from_file_location("_VAL002_actual_H020v3_guard",ROOT/OVERLAY)
    guard=importlib.util.module_from_spec(loader);loader.loader.exec_module(guard)
    guard.configure_v2(pins[REPAIR],policy_ref=pins[POLICY])
    require(guard.MODE=="prospective_980gb_H020_v3" and all(getattr(guard.budget.v4,n).__module__==guard.budget.v4.__name__
        for n in ("canonical","file_bytes","mutex")),"Original unwrapped guard functions before role split")
    if consumer:
        ms=importlib.util.spec_from_file_location("_VAL002_readonly_namespace_memo",ROOT/NAMESPACE_MEMO_REF["path"])
        memo=importlib.util.module_from_spec(ms);ms.loader.exec_module(memo)
        guard._remaining_VAL_memo2=memo.install_consumer_v1(guard.budget.v4,guard.budget,
            consumer_role="read_only_consumer",expected_source_path=ROOT/V4_REF["path"],expected_source_sha256=V4_REF["sha256"])
        require(guard._remaining_VAL_memo2.receipt["installed"] is True
            and guard._remaining_VAL_memo2.receipt["cross_operation_cache"] is False
            and guard._remaining_VAL_memo2.receipt["namespace_extension_schema"]=="resource_guard_consumer_namespace_memo_v1",
            "Original operation-local namespace memo only in read-only consumers")
    return guard

def stage_plan(stage):
    pins=frozen();folder,output,job=identity(stage)
    return dict(stage=stage,parent_id=PARENT,seed_row=SEED_ROW,registry_ref=REGISTRY_REF,split="val",
        sequence_length_bp=30000000,counts_haploid=dict(AFR=200,EUR=200,ASIA=200,ADMIX=1000),
        reference_sample_indices=[0,600],query_sample_indices=[600,1600],query_focal_count=10000,
        job_id=job,packet_directory=str(folder.relative_to(ROOT)),output_directory=output,
        actual_VAL000_config_basis_ref=BASE_CONFIG_REFS[stage],source_ref=pins[PRODUCERS[stage]],
        source_closure_refs=source_closure(stage,pins),runtime_ref=pins[INFER_RUNTIME if stage=="infer" else SIM_RUNTIME],
        required_predecessor_closed_key=predecessor_key(stage),original_caps_bytes=dict(simulate=6500000000,infer=12500000000,native_paths=16000000000),
        consumer_memo_ref=MEMO_REF,consumer_namespace_memo_ref=NAMESPACE_MEMO_REF,original_v4_ref=V4_REF,root_mutations_unwrapped=True,
        no_bulk_prereservation=True,all_method_prediction_seals_before_truth=True,
        native_model_transfer_from_other_ARG=False,GNN_VAL_optimization=False,TEST_access=False)

def admit(stage,policy_ref=None,*,completion_ref=None,wait_ref=None):
    with ADMISSION_LOCK.open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _admit_locked(stage,policy_ref,completion_ref=completion_ref,wait_ref=wait_ref)

def command(stage,admission_ref):
    pins=frozen();spec=read(admission_ref);folder,output,job=identity(stage)
    require(admission_ref["path"]==str((folder/"admission.json").relative_to(ROOT))
        and spec["parent_id"]==PARENT and spec["stage"]==stage and spec["job_id"]==job
        and spec["source_ref"]==pins[PRODUCERS[stage]] and spec["driver_ref"]==ref(SOURCE)
        and spec["execution_authorized"] is True and spec["GPU_authorized"] is False
        and spec["consumer_memo_ref"]==MEMO_REF and spec["consumer_namespace_memo_ref"]==NAMESPACE_MEMO_REF
        and spec["original_v4_ref"]==V4_REF
        and spec["worker_bootstrap_ref"]==ref(HERE/"worker_bootstrap_v1.py"),"Exact newly admitted remaining-VAL CPU stage")
    python=ROOT/(".runtime/envs/biological_tsinfer_v1/bin/python" if stage=="infer" else ".runtime/env/bin/python")
    original_argv=[str(python),"-B","-m","arggnn."+Path(PRODUCERS[stage]).stem,"--execute","--admission",admission_ref["path"],"--expected-admission-sha256",admission_ref["sha256"]]
    return dict(argv=[str(python),"-B",str(HERE/"worker_bootstrap_v1.py"),*original_argv[3:]],original_scientific_argv=original_argv,
        cwd=str(ROOT),environment=dict(CUDA_VISIBLE_DEVICES="",PYTHONPATH=str(ROOT)+":"+str(ROOT/"src"),PYTHONDONTWRITEBYTECODE="1",
          TMPDIR=str(folder/"worker_aux/work"),TMP=str(folder/"worker_aux/work"),TEMP=str(folder/"worker_aux/work"),
          OMP_NUM_THREADS="8",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMBA_NUM_THREADS="8",
          ARGGNN_COMPLETED_CREDITS_POLICY_REF=json.dumps(spec["completed_overlay_policy_ref"],separators=(",",":"))),
        create_before_launch=[str((ROOT/output).parent),str(folder/"worker_aux/work")],
        supervisor_requirements="Unchanged original v2 owned-session CPU monitor, wait/drain and resource bounds.",
        deadlines_seconds=dict(simulate=3600,infer=10800,native_paths=7200)[stage],
        native_RSS_ceiling_bytes=68719476736 if stage=="native_paths" else None)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--parent",choices=PARENTS,required=True)
    m=p.add_mutually_exclusive_group();m.add_argument("--plan",choices=STAGES);m.add_argument("--admit",choices=STAGES);m.add_argument("--command",choices=STAGES)
    p.add_argument("--policy");p.add_argument("--policy-sha256");p.add_argument("--admission");p.add_argument("--admission-sha256")
    a=p.parse_args();select_parent(a.parent)
    if a.admit:
        require(bool(a.policy)==bool(a.policy_sha256),"Policy path and hash together")
        value=admit(a.admit,dict(path=a.policy,sha256=a.policy_sha256) if a.policy else None)
    elif a.command:value=command(a.command,dict(path=a.admission,sha256=a.admission_sha256))
    else:value=stage_plan(a.plan) if a.plan else dict(status="source_only_plan_no_guards_or_jobs",stages=[stage_plan(s) for s in STAGES])
    print(json.dumps(value,sort_keys=True,indent=2))


_RENEWAL_READY=False
def install_renewal_v1():
    global _RENEWAL_READY
    if _RENEWAL_READY:return
    require(ref(RENEWAL_REF["path"])==RENEWAL_REF and ref(RENEWAL_CERT_REF["path"])==RENEWAL_CERT_REF,
        "Approved exact renewal before accounting/project imports")
    sp=importlib.util.spec_from_file_location("_VAL002_approved_marker_renewal",ROOT/RENEWAL_REF["path"])
    renewal=importlib.util.module_from_spec(sp);sp.loader.exec_module(renewal)
    renewal.install_renewal_v1(RENEWAL_CERT_REF)
    _RENEWAL_READY=True

def readiness_refs():
    return [SIMULATION_COMPLETION_REF,SIMULATION_OBSERVATION_REF,INFERENCE_COMPLETION_REF,
        INFERENCE_OBSERVATION_REF,INFERENCE_WAIT_REF,INFERENCE_CLOSE_REF,PROTOCOL_REF,READINESS_REF]

if __name__=="__main__":main()
