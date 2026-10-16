"""Thin ROOT metadata/guard adapter for original VAL003/004 preparation. Inert import."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import contextlib,fcntl,hashlib,importlib.util,json,os,stat,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/American_C3_VAL004_fresh7500MB_authority_source_20261016_v1"
SOURCE=REL+"/root_contract_v1.py";DRIVER=REL+"/root_driver_v1.py";WORKER=REL+"/worker_entry_v1.py"
MANIFEST=REL+"/source_manifest.v1.json";SOURCE_PINS=MANIFEST
READINESS=REL+"/readiness.v1.json";BASIS=REL+"/scientific_basis.v1.json";CAPFILE=REL+"/caps.template.inactive.v1.json"
GUARD_REF=POLICY_REF=SELECTOR_REF=None
_SELECTOR_MODULE=_SELECTED_GUARD=None
SELECTOR_SOURCE_REF={'path': 'research/reviews/native7_H050_stopped_retention_successor_source_20261016_v1/explicit_policy_selector_v1.py', 'sha256': '2a991ca6e2da3f4ba05888bc44e4cd53025f929abf88760aa4f93571e13bc87b'}
APPROVED_GUARDS=({'source_ref': {'path': 'research/reviews/native14_stopped_retention_successor_source_20261016_v3/native14_stopped_retention_v1.py', 'sha256': '60d010e393bda25f662381051b3fd3d4e4decfa034b10c3baad3c3f30a6a196c'}, 'source_manifest_ref': {'path': 'research/reviews/native14_stopped_retention_successor_source_20261016_v3/source_manifest.v1.json', 'sha256': 'c2bae5575ae0e8e68e152b0edde0b7b2d53e4fc7398a3a8501ce278e23a84fa6'}, 'measurement_source_ref': {'path': 'research/reviews/native14_stopped_retention_successor_source_20261016_v3/retained_nodes_v1.py', 'sha256': '83f544e45eba5190174b70d43bf84ed4f0b58f80159be593b4e8392c3e4afb92'}}, {'source_ref': {'path': 'research/reviews/native7_H050_stopped_retention_successor_source_20261016_v1/stopped_retention_v1.py', 'sha256': 'c9392ed88b76a5a6c1394d13ff78b55df83c1b16d8186d23b4d3110b46008e8f'}, 'source_manifest_ref': {'path': 'research/reviews/native7_H050_stopped_retention_successor_source_20261016_v1/source_manifest.v1.json', 'sha256': 'cb6c3af2858d007d74edb08742f8633ddef5f2a9cb4c6cd5b3682205fee32d88'}, 'measurement_source_ref': {'path': 'research/reviews/native7_H050_stopped_retention_successor_source_20261016_v1/retained_nodes_v1.py', 'sha256': '1da50bff13c94cc3003aab9ce186746300641d9ec8ef6c49f4c443d1ab179f91'}})
ORIGINAL_EXPIRY_UTC='2026-10-16T20:04:16.988073+00:00'
SLOT_IDENTITY_REF={'path': 'research/reviews/American_C3_VAL003004_native14v3_preparation_source_20261016_v3/slot_resource_identity.v3.json', 'sha256': '8174fa2a8044e0c200530f92ef6496239f4a7722ee5e35cace0662fe7147a3b2'}
PARTIAL_ADMISSION_AUDIT_REF={'path': 'research/reviews/VAL003_admission_partial_audit_20261016_v1/conclusion.v1.json', 'sha256': '69936ff6f13ae820453c21e026f3fa6e5adeca749845ae805a14cc7c468bb140'}
SLOT_REL='research/reviews/American_C3_VAL003004_current_Census_preparation_source_20261016_v1'
REPAIR_REF=dict(path="research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json",sha256="f6f2d628199e8be73db8b5d90d428570d8afdefeab9dc5cd94bb5eafb39247db")
RENEWAL_REF=dict(path="research/reviews/H039_marker_repair_renewal_source_20261016_v1/marker_repair_renewal_v1.py",sha256="491236877e8d71cae8bd7619880858ac501c57dd93e59479ca32dacacab20468")
CERT_REF=dict(path="research/reviews/H039_marker_repair_renewal_source_20261016_v1/renewal_certificate.root_active_v1.json",sha256="66006202094d7a3ef2b29cbec28eb93f334505b72589f09d86e4b57498162db7")
MEMO_REF=dict(path="research/reviews/resource_guard_hotpath_source_audit_v1/transaction_memo_adapter_v2.py",sha256="c972e9a779d424ceafdb258848614bf40994d1004e2c73fc5cd225ae99621364")
NAMESPACE_REF=dict(path="research/reviews/current_mutex_contention_20261015_v1/consumer_namespace_memo_v1.py",sha256="17a1377320f61c3c0cc8d3816d0259f9dae3121ed91bf7206a7ee63fcab8eb7f")
V4_REF=dict(path="src/arggnn/project_budget_v4.py",sha256="a093c7d86f57fee569cb99f4712250e9505a1119f3950fa44282c7ba8c57df47")
PIDFD=dict(path="research/reviews/American_C3_native_MLM_full_source_v5/pidfd_backend_v1.py",sha256="61be5d89357d1b95611e0ba26dd39d381ececf3ce6a36c347507cdc062373eca")
OWNED=dict(path="research/reviews/American_C3_native_MLM_amendment_v12/owned_lifecycle_v2.py",sha256="93d71acc40f1037d0b37acb2ca0859300ea4bd9741409e9bd00ca21bf100d633")
BASE_CONFIG_REF=dict(path="configs/census_tsinfer_fit_v1.json",sha256="7c2436d5479e68beb1a8f3edb04ca0ee729a23be35ca2d7893827c0db68e3b00")
PARENTS=("american_c3_val_004",);STAGES=("native_paths","plan")
PARENT=STAGE=PHASE=None;CAPS={};PHASES=CPU_PHASES=();GPU_PHASES=()
SLOT_FD=None;GPU_MODULE=GPU_MAIN=GPU_AUTH_REF=LAUNCHER_REF=None
STATUSES={};ENTRY=WORKER
def require(ok,message):
    if not ok:raise ValueError(message)
def need(ok,message):require(ok,message)
def local(path):
    p=Path(path);p=p if p.is_absolute() else ROOT/p
    require(".." not in p.parts and p.resolve()==p and p.is_relative_to(ROOT),"Canonical contained source/metadata")
    return p
def ref(path):
    p=local(path);require((p.suffix in (".py",".json",".sh",".lock") or p==ROOT/"research/reviews/old_native14GiB_retention_evidence_20261016_v1/AUDIT.md") and p.is_file() and not p.is_symlink()
        and p.stat().st_size<=32*2**20,"Bounded source/JSON only; no scientific payload authentication here")
    return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def raw(record):
    require(isinstance(record,dict) and set(record)=={"path","sha256"} and ref(record["path"])==record,"Exact actual source/metadata ref")
    return local(record["path"]).read_bytes()
def read(record,cap=32*2**20):
    data=raw(record);require(len(data)<=cap,"Bounded compact JSON");return json.loads(data)
def load(record,name):
    raw(record);sp=importlib.util.spec_from_file_location(name,local(record["path"]))
    m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def save(folder,name,value,cap=2**20):
    require(Path(name).name==name,"Flat immutable metadata")
    data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    require(len(data)<=cap,"Bounded metadata receipt")
    fd=os.open(folder/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,"wb") as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
    return ref(folder/name)
def put(path,value,cap=2**20):return save(path.parent,path.name,value,cap=cap)
def active(value,margin=0):
    now=datetime.now(timezone.utc);start=datetime.fromisoformat(value["created_at_utc"]);end=datetime.fromisoformat(value["expires_at_utc"])
    require(start.tzinfo and end.tzinfo and start<=now and now+timedelta(seconds=margin)<end,"Actual current ROOT authority and finite evidence margin")
def current(value,margin=0):active(value,margin)
def select(parent,stage):
    global PARENT,STAGE,PHASE,CAPS,PHASES,CPU_PHASES,STATUSES,ENTRY,RUNTIME_REF,HOOK_REF,TRAIN_SOURCE_REF
    require(parent in PARENTS and stage in STAGES,"Only future original VAL003/004 native-path/plan")
    PARENT=parent;STAGE=stage;PHASE=stage+"_val_"+parent[-3:];PHASES=CPU_PHASES=(PHASE,)
    CAPS=read(ref(CAPFILE))["stages"][stage]
    status={"infer":"completed_current_C3_inference","native_paths":"completed_native_C3_path_preparation","plan":"completed_C3_GNN_current980_"+PHASE}
    STATUSES={PHASE:status[stage]};ENTRY=REL+"/plan_worker_v1.py" if stage=="plan" else WORKER
    basis=read(ref(BASIS));RUNTIME_REF=basis[{"infer":"inference_runtime_ref","native_paths":"native_runtime_ref","plan":"plan_runtime_ref"}[stage]]
    HOOK_REF=basis["plan_hook_ref"]
    TRAIN_SOURCE_REF=next(r for r in read(ref(MANIFEST))["refs"] if r["path"]=="src/arggnn/american_c3_gnn_train_v1.py")
def phase_paths(phase):
    require(phase==PHASE,"Exact selected phase")
    return local(REL+"/admitted_val"+PARENT[-3:]+"_"+STAGE+"_fresh7500_v1"),"experiments/results/American_C3_val"+PARENT[-3:]+"_full30mb_"+STAGE+"_fresh7500_v1"
def validate_slot_resources(rows):
    require(SLOT_IDENTITY_REF in rows,"Exact JSON slot identity receipt stays in sealed source closure")
    value=read(SLOT_IDENTITY_REF)
    require(set(value)=={"schema","same_original_pool","maximum_CPU_producers","slots"}
        and value["schema"]=="American_C3_original_two_slot_identity_v3"
        and value["same_original_pool"] is True and value["maximum_CPU_producers"]==2
        and len(value["slots"])==2,"Only exact original two-slot resource receipt")
    for index,row in enumerate(value["slots"]):
        expected=SLOT_REL+"/cpu_slot_"+str(index)+".lock"
        require(set(row)=={"ref","st_dev","st_ino","st_size","st_nlink","file_kind"}
            and row["ref"] in rows and row["ref"]["path"]==expected
            and row["file_kind"]=="regular" and row["st_size"]==0 and row["st_nlink"]==1,
            "Same exact original slot paths and fixed empty single-link regular identities")
        raw(row["ref"]);p=local(expected);st=p.lstat()
        require(stat.S_ISREG(st.st_mode) and not p.is_symlink()
            and (st.st_dev,st.st_ino,st.st_size,st.st_nlink)==
                (row["st_dev"],row["st_ino"],row["st_size"],row["st_nlink"]),
            "Original slot inode/content identity must remain unchanged")
def source_paths():
    rows=read(ref(MANIFEST))["refs"]
    for r in rows:raw(r)
    validate_slot_resources(rows)
    return tuple(r["path"] for r in rows)
def selector_value(record):
    value=read(record)
    require(value.get("active") is True and value.get("root_authorizer")=="root_agent"
        and value.get("selector_source_ref")==SELECTOR_SOURCE_REF
        and value.get("repair_authority_ref")==REPAIR_REF,"Exact active immutable ROOT selector and known selector source")
    require(any(all(value.get(k)==v for k,v in choice.items()) for choice in APPROVED_GUARDS),
        "Reject unknown selected source/manifest/measurement digest")
    for k in ("source_ref","source_manifest_ref","measurement_source_ref","policy_ref","source_review_ref"):raw(value[k])
    p=read(value["policy_ref"])
    require(p.get("source_ref")==value["source_ref"] and p.get("source_manifest_ref")==value["source_manifest_ref"]
        and p.get("measurement_source_ref")==value["measurement_source_ref"] and p.get("source_review_ref")==value["source_review_ref"],
        "Exact selected actual policy digest and source lineage")
    return value
def bind_selector(record):
    global SELECTOR_REF,POLICY_REF,GUARD_REF,_SELECTOR_MODULE,_SELECTED_GUARD
    require(SELECTOR_REF is None or SELECTOR_REF==record,"No operational selector swap")
    value=selector_value(record)
    if _SELECTOR_MODULE is None:_SELECTOR_MODULE=load(SELECTOR_SOURCE_REF,"_VAL004_fresh7500_explicit_selector")
    g=_SELECTOR_MODULE.load_selected_guard_v1(record)
    require(g.MODE==value["accounting_mode"] and g.JOURNAL==value["journal"],"Exact returned selected guard mode/journal")
    SELECTOR_REF=dict(record);POLICY_REF=dict(value["policy_ref"]);GUARD_REF=dict(value["source_ref"]);_SELECTED_GUARD=g
    return value
def original_guard():
    require(SELECTOR_REF is not None,"Explicit immutable selected ROOT guard required")
    bind_selector(SELECTOR_REF)
    require(_SELECTED_GUARD is not None,"Exactly once configured selected guard")
    return _SELECTED_GUARD
def consumer_guard():
    g=original_guard();m=load(NAMESPACE_REF,"_American_VAL003004_readonly_namespace_memo")
    handle=m.install_consumer_v1(g.budget.v4,g.budget,consumer_role="read_only_consumer",
        expected_source_path=ROOT/V4_REF["path"],expected_source_sha256=V4_REF["sha256"])
    return g,handle
def guard(consumer=True):return consumer_guard()[0] if consumer else original_guard()
def observation(record,parent):
    m=read(record)
    require(m["parent_id"]==parent and m["split"]=="val" and m["sequence_length_bp"]==30000000
        and m["sample_unit"]=="haploid" and m["class_order"]==["EUR","EAS","AFR"]
        and m["counts_haploid"]==dict(AFR=200,EUR=200,ASIA=200,ADMIX=1000)
        and m["query_ancestry_truth_in_inputs"] is False,"Original full30Mb all1600 sanitized observed identity")
    return m
def existing_simulation(parent):
    ready=read(ref(READINESS));row=ready["parents"][parent]
    completed=read(row["simulation_completion_ref"]);wait=read(row["simulation_wait_ref"])
    terminal=read(row["simulation_outer_terminal_ref"]);closed=read(row["simulation_close_ref"])
    cfg=read(row["simulation_config_ref"]);m=observation(row["simulation_observation_ref"],parent)
    require(completed["status"]=="completed_C3_simulation_reconstruction" and completed["parent_id"]==parent
        and completed["sites"]==row["sites"] and completed["observation_ref"]==row["simulation_observation_ref"],
        "Exact existing completed original simulation metadata")
    require(wait["actual_direct_worker_wait"] is True and wait["returncode"]==0 and wait["owned_worker_group_ESRCH"] is True
        and wait["operational_failure"] is None and wait["completed_ref"]==row["simulation_completion_ref"],
        "Genuine saved simulation direct wait0 and drain")
    require(terminal["actual_terminal_consumed"] is True and terminal["actual_exit_code"]==0
        and terminal["completed_ref"]==row["simulation_completion_ref"] and terminal["actual_wait_ref"]==row["simulation_wait_ref"],
        "Genuinely consumed original simulation outer terminal0")
    require(closed["key"]==row["simulation_closed_key"] and closed["value"]["future_writes_and_relaunch_revoked"] is True
        and closed["value"]["proof"]["completed_ref"]==row["simulation_completion_ref"]
        and closed["value"]["proof"]["direct_wait_ref"]==row["simulation_wait_ref"],"Existing completed simulation closed; never replay or credit again")
    require(cfg["ancestry_seed"]==row["seed_row"]["ancestry_seed"] and cfg["mutation_seed"]==row["seed_row"]["mutation_seed"]
        and cfg["sequence_length_bp"]==30000000,"Original independent simulation seeds/full extent")
    return row
def predecessor_paths_v1(parent,stage):
    require(parent in PARENTS and stage in STAGES,"Only exact future parent/stage prerequisites")
    suffix=parent[-3:]
    if stage=="native_paths":
        return ("infer","research/reviews/American_C3_VAL003004_native14v3_preparation_source_20261016_v3/admitted_val"+suffix+"_infer_v3",
            "experiments/results/American_C3_val"+suffix+"_full30mb_infer_native14v3_v3",
            "H041/American-C3-val"+suffix+"-full30mb-infer-native14v3-v3")
    return ("native_paths",REL+"/admitted_val"+suffix+"_native_paths_fresh7500_v1",
        "experiments/results/American_C3_val"+suffix+"_full30mb_native_paths_fresh7500_v1",
        "H041/American-C3-val"+suffix+"-full30mb-native_paths-fresh7500-v1")
def actual_prior(parent,stage):
    row=existing_simulation(parent)
    if stage=="infer":
        return dict(completion_ref=row["simulation_completion_ref"],wait_ref=row["simulation_wait_ref"],
            outer_ref=row["simulation_outer_terminal_ref"],close_ref=row["simulation_close_ref"],
            observation_ref=row["simulation_observation_ref"],closed_key=row["simulation_closed_key"])
    previous,folder_rel,output,previous_key=predecessor_paths_v1(parent,stage)
    folder=local(folder_rel)
    cp=ref(output+"/completed.json");wr=ref(folder/"worker_aux/actual_wait.json")
    tr=ref(folder/"outer_terminal.root_actual_v1.json");cr=ref(folder/"closed/close_receipt.json")
    done=read(cp);wait=read(wr);terminal=read(tr);close=read(cr)
    require(wait["completed_ref"]==cp and wait["actual_direct_worker_wait"] is True and wait["returncode"]==0
        and wait["parent_id"]==parent and wait["phase"]==previous+"_val_"+parent[-3:]
        and wait["input_ref"]["path"]==str((folder/"input.json").relative_to(ROOT))
        and wait["all_owned_processes_and_groups_drained"] is True and wait["operational_failure"] is None,
        "Actual same-parent predecessor success/wait/drain")
    require(terminal["root_authorizer"]=="root_agent" and terminal["actual_terminal_consumed"] is True
        and terminal["actual_exit_code"]==0 and terminal["completed_ref"]==cp and terminal["direct_wait_ref"]==wr,
        "ROOT genuinely consumed exact predecessor outer terminal")
    require(close["key"]==previous_key
        and close["value"]["future_writes_and_relaunch_revoked"] is True and close["value"]["proof"]["completed_ref"]==cp
        and close["value"]["proof"]["direct_wait_ref"]==wr,"Actual predecessor ordinary completed-scope close")
    require(done["status"]=={"infer":"completed_current_C3_inference","native_paths":"completed_native_C3_path_preparation"}[previous],
        "Exact unchanged producer completion")
    obs=observation(done["observation_ref"],parent)
    require(obs["genealogy_kind"]=="inferred_tsinfer_tsdate_CURRENT_RECONSTRUCTION_not_simulated_true_ARG"
        and obs["sites_and_sample_identities_verified"] is True and obs["source_observation_ref"]==row["simulation_observation_ref"],
        "All original inferred sites and sample identities; no trimming/subsampling")
    result=dict(completion_ref=cp,wait_ref=wr,outer_ref=tr,close_ref=cr,observation_ref=done["observation_ref"],closed_key=close["key"])
    if previous=="infer":
        require(result==read(ref(REL+"/frozen_infer_prerequisite.v1.json")),"Exact sealed original VAL004 inferred prerequisite remains immutable")
    if previous=="native_paths":
        require(done["parent_id"]==parent and done["query_focal_count"]==10000 and done["query_ancestry_truth_opened"] is False
            and done["native_split_seed"]==42 and done["native_train_fraction"]==.9
            and done["class_order"]==["EUR","EAS","AFR"],"Actual complete author paths and original10000 label-blind focals")
        result["focal_registry_ref"]=done["focals_ref"]
    return result
def scientific_config(parent,stage,prior,active_flag):
    basis=read(ref(BASIS));row=read(ref(READINESS))["parents"][parent]
    template=read(ref(REL+"/"+stage+".scientific_config.template.inactive.v1.json"))
    if stage in ("infer","native_paths"):
        original=read(basis[stage+"_config_ref"]);cfg=dict(original)
        cfg.update(execution_enabled=active_flag,observation_ref=prior["observation_ref"])
        allowed={"execution_enabled","observation_ref"}
        if stage=="native_paths":cfg["path_sampling_seed"]=row["seed_row"]["path_sampling_seed"];allowed.add("path_sampling_seed")
    else:
        original=read(basis["plan_config_ref"]);cfg=dict(original)
        cfg.update(execution_enabled=active_flag,observation_ref=prior["observation_ref"],
            native_preparation_ref=prior["completion_ref"],focal_registry_ref=prior["focal_registry_ref"])
        allowed={"execution_enabled","observation_ref","native_preparation_ref","focal_registry_ref"}
    require(set(cfg)==set(original) and {k for k in cfg if cfg[k]!=original[k]}<=allowed,"Only declared original parent/prerequisite operational rebindings")
    return cfg
def worker_caps(stage):
    if stage=="plan":
        value=read(read(ref(CAPFILE))["plan_original_caps_ref"]);value.update(active=True,phase=PHASE);return value
    return dict(CAPS)
def make_config(req,caps):
    if STAGE!="plan":return req["bindings"]["scientific_config"]
    packet,_=phase_paths(PHASE)
    return dict(schema="C3_GNN_current980_operational_config_v1",execution_enabled=True,phase=PHASE,
        output_cap_bytes=caps["reservation_cap_bytes"],short_tmp_directory=str((packet/"worker_aux/tmp").relative_to(ROOT)),
        child_log_ram_bytes=caps["child_log_ram_bytes"])
def hooks():return load(HOOK_REF,"_American_VAL003004_unchanged_full_block_plan_hook")
def reviewed(record):
    value=read(record)
    require(value.get("verdict")=="SOURCE_PASS" and value.get("blockers")==[],"Actual independent SOURCE_PASS")
    handoff=value.get("candidate_handoff_ref",value.get("source_handoff_ref"))
    if handoff is not None:
        seal=read(handoff)
        require(handoff["path"]==REL+"/source_handoff.v1.json" and seal["source_manifest_ref"]==ref(MANIFEST)
            and seal["source_only"] is True,"Review binds this exact sealed continuation handoff/manifest")
    else:require(value.get("source_manifest_ref")==ref(MANIFEST),"Review binds this exact continuation manifest")
    return value
def authority_metadata(value,review_ref,selector_ref,stage,now=None):
    require(stage in STAGES,"Exact never-admitted original VAL004 stage")
    require(value.get("schema")=="VAL004_fresh7500_explicit_ROOT_authority_v1"
        and value.get("active") is True and value.get("root_authorizer")=="root_agent"
        and value.get("parent_id")=="american_c3_val_004" and value.get("allowed_stages")==list(STAGES)
        and value.get("adapter_manifest_ref")==ref(MANIFEST) and value.get("source_review_ref")==review_ref
        and value.get("storage_policy_selector_ref")==selector_ref
        and value.get("historical_expired_authority_utc")==ORIGINAL_EXPIRY_UTC
        and value.get("frozen_infer_prerequisite_ref")==ref(REL+"/frozen_infer_prerequisite.v1.json")
        and value.get("old_v3_and_down8_future_routes_revoked") is True
        and value.get("only_never_admitted_original_VAL004_stages") is True
        and value.get("no_replay_resume_or_old_clock_reset") is True
        and value.get("native_accounting_failstop_bytes")==7500000000
        and value.get("native_scientific_output_cap_bytes")==16000000000
        and value.get("native_scientific_wall_seconds")==7200
        and value.get("maximum_CPU_producers")==2
        and value.get("all30_C3_prediction_seals_before_VAL_truth") is True
        and value.get("query_truth_access_authorized") is False
        and value.get("GNN_optimization_authorized") is False
        and value.get("TEST_access_authorized") is False,"Explicit fresh ROOT authority preserves all original science and barriers")
    start=datetime.fromisoformat(value["created_at_utc"]);end=datetime.fromisoformat(value["expires_at_utc"])
    old=datetime.fromisoformat(ORIGINAL_EXPIRY_UTC)
    require(start.tzinfo and end.tzinfo and start>=old and end-start==timedelta(hours=24),
        "Separately authorized fresh24h interval after immutable historical expiry")
    actual=datetime.now(timezone.utc) if now is None else now
    require(start<=actual and actual+timedelta(seconds=CAPS["wall_seconds"]+300)<end,
        "Fresh full unchanged scientific runtime plus drain/evidence preparation margin")
    require(end<datetime.fromisoformat(read(CERT_REF)["renewed_expires_at_utc"]),
        "Existing marker renewal covers new finite authority")
    return value
def validate_root_authority(record,review_ref,selector_ref,stage):
    return authority_metadata(read(record),review_ref,selector_ref,stage)
def validate_request(record):
    q=read(record);select(q["parent_id"],q["stage"]);bind_selector(q["storage_policy_selector_ref"]);active(q,300)
    authority=validate_root_authority(q["root_authority_ref"],q["source_review_ref"],SELECTOR_REF,STAGE)
    require(q["created_at_utc"]==authority["created_at_utc"] and q["expires_at_utc"]==authority["expires_at_utc"],
        "New request uses exact separate ROOT authority clocks, never old clock replay")
    require(q["schema"]=="American_C3_VAL004_fresh7500_request_v1" and q["active"] is True
        and q["root_authorizer"]=="root_agent" and q["phase"]==PHASE and q["execution_authorized"] is True
        and q["source_review_passed"] is True and q["adapter_manifest_ref"]==ref(MANIFEST)
        and q["guard_ref"]==GUARD_REF and q["policy_ref"]==POLICY_REF and q["maximum_CPU_producers"]==2
        and q["query_truth_access_authorized"] is False and q["GNN_optimization_authorized"] is False
        and q["TEST_access_authorized"] is False and q["all30_C3_prediction_seals_before_VAL_truth"] is True,
        "Exact fresh independently reviewed VAL004-only CPU preparation")
    require(q.get("reservation_only_downward_failstop_authorized") is True
        and q.get("accounting_reservation_bytes")==CAPS["reservation_bytes"],"Explicit reservation-only failstop")
    if STAGE=="native_paths":
        require(CAPS["reservation_bytes"]==7500000000 and q["scientific_config"]["output_cap_bytes"]==16000000000,
            "7.5B accounting failstop, unchanged16GB scientific bound")
    reviewed(q["source_review_ref"]);source_paths()
    prior=actual_prior(PARENT,STAGE)
    require(q["prior"]==prior and q["scientific_config"]==scientific_config(PARENT,STAGE,prior,True),
        "Genuine complete same-parent predecessor and original scientific recipe")
    require(q["bindings"]==(dict(parent_id=PARENT,plan_config=q["scientific_config"]) if STAGE=="plan"
        else dict(parent_id=PARENT,scientific_config=q["scientific_config"])),"Original inputs and no truth payload")
    require(read(q["caps_ref"])==worker_caps(STAGE) and q["runtime_ref"]==RUNTIME_REF,"Original runtime and operational limits")
    return q,read(q["caps_ref"]),read(RUNTIME_REF)
def bind(parent,stage,review_ref,selector_ref,root_authority_ref):
    select(parent,stage);reviewed(review_ref);source_paths()
    authority=validate_root_authority(root_authority_ref,review_ref,selector_ref,stage)
    bind_selector(selector_ref);original_guard()
    prior=actual_prior(parent,stage);cfg=scientific_config(parent,stage,prior,True)
    caps_ref=save(local(REL),"caps_val004_"+stage+".root_active_v1.json",worker_caps(stage))
    q=dict(schema="American_C3_VAL004_fresh7500_request_v1",active=True,root_authorizer="root_agent",
        execution_authorized=True,parent_id=parent,stage=stage,phase=PHASE,
        created_at_utc=authority["created_at_utc"],expires_at_utc=authority["expires_at_utc"],
        root_authority_ref=root_authority_ref,storage_policy_selector_ref=selector_ref,
        adapter_manifest_ref=ref(MANIFEST),guard_ref=GUARD_REF,policy_ref=POLICY_REF,source_review_ref=review_ref,source_review_passed=True,
        prior=prior,scientific_config=cfg,caps_ref=caps_ref,runtime_ref=RUNTIME_REF,maximum_CPU_producers=2,
        query_truth_access_authorized=False,GNN_optimization_authorized=False,TEST_access_authorized=False,
        all30_C3_prediction_seals_before_VAL_truth=True,
        reservation_only_downward_failstop_authorized=True,accounting_reservation_bytes=CAPS["reservation_bytes"],
        bindings=dict(parent_id=parent,plan_config=cfg) if stage=="plan" else dict(parent_id=parent,scientific_config=cfg))
    r=save(local(REL),"request_val004_"+stage+".root_active_v1.json",q)
    validate_request(r);return r
def never_admitted_stage_keys_v1():
    require(PARENT=="american_c3_val_004" and STAGE in STAGES,"Only original VAL004 future stages")
    prefix="H041/American-C3-val004-full30mb-"+STAGE
    return (prefix+"-native14v3-v3",prefix+"-native14v3-down8-v1",prefix+"-fresh7500-v1")
def require_never_admitted_v1(reservations):
    require(all(key not in reservations for key in never_admitted_stage_keys_v1()),
        "No old or new allocation/key replay, including closed historical allocations")
def unique(rows):
    d={}
    for r in rows:raw(r);require(r["path"] not in d or d[r["path"]]==r,"Exact unique role");d[r["path"]]=r
    require(len(d)<=64,"Original direct64 ceiling");return list(d.values())
def direct_roles(request_ref,q,cfgref,closure,runtime):
    # All current returned guard roles stay direct through state[3]. Historical
    # source/evidence are transitively authenticated by this directly bound manifest.
    basis=read(ref(BASIS))
    return [ref(SOURCE),ref(DRIVER),ref(WORKER),ref(REL+"/plan_worker_v1.py"),ref(MANIFEST),
        request_ref,q["caps_ref"],cfgref,closure,runtime,q["source_review_ref"],
        q["root_authority_ref"],q["storage_policy_selector_ref"],SELECTOR_SOURCE_REF,
        basis["common_producer_ref"],basis["native_producer_ref"],basis["author_path_helper_ref"]]
def input_spec(record):
    s=read(record);q,caps,_=validate_request(s["root_stage_request_ref"]);active(s,300)
    packet,output=phase_paths(PHASE);storage=read(s["refs"]["storage"])
    require(record["path"]==str((packet/"input.json").relative_to(ROOT)) and s["phase"]==PHASE and s["parent_id"]==PARENT
        and s["root_driver_ref"]==ref(DRIVER) and s["output_directory"]==output and s["storage"]==storage
        and s["created_at_utc"]==q["created_at_utc"] and s["expires_at_utc"]==q["expires_at_utc"]
        and s["runtime_ref"]==RUNTIME_REF and s["selected_policy_ref"]==POLICY_REF
        and s["storage_policy_selector_ref"]==SELECTOR_REF and s["root_authority_ref"]==q["root_authority_ref"] and s["GPU_authorized"] is False,"Exact admitted original CPU stage")
    require(storage["reservation"]["reserved_bytes"]==CAPS["reservation_bytes"]
        and storage["reserved_new_bytes"]==CAPS["reservation_bytes"]
        and storage["completed_overlay_policy_ref"]==POLICY_REF and storage["budget_components"]["consumer_adapter"]==GUARD_REF,
        "Actual current Census full charged stage reservation")
    require(storage.get("storage_policy_selector_ref")==SELECTOR_REF,"Storage binds exact immutable ROOT selector")
    require(s["source_sha256"]=={path:ref(path)["sha256"] for path in source_paths()},"Complete exact authenticated transitive source map")
    for path,sha in s["source_sha256"].items():raw(dict(path=path,sha256=sha))
    closure=read(s["transitive_source_closure_ref"])
    require(closure["source_sha256"]==s["source_sha256"] and closure["root_stage_request_ref"]==s["root_stage_request_ref"]
        and closure["bindings"]==q["bindings"] and closure["native_MLM_dependency"] is False,"Exact actual input/source closure")
    needed=direct_roles(s["root_stage_request_ref"],q,s["refs"]["config"],s["transitive_source_closure_ref"],s["refs"]["runtime"])
    require(len(storage["reservation"]["consumer_bindings"])<=64 and all(r in storage["reservation"]["consumer_bindings"] for r in needed),
        "Every complete actual phase role bound under original64 limit")
    require(read(s["refs"]["config"])==make_config(q,caps),"Original admitted scientific/operational config")
    return s,q,read(s["refs"]["config"])
@contextlib.contextmanager
def cpu_slot():
    selected=None
    for index in range(2):
        path=local(SLOT_REL+"/cpu_slot_"+str(index)+".lock");raw(ref(path))
        fd=os.open(path,os.O_RDWR|os.O_NOFOLLOW)
        try:
            require(os.fstat(fd).st_ino==path.lstat().st_ino and os.fstat(fd).st_size==0,"Exact immutable source slot inode")
            try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:os.close(fd);continue
            selected=fd;break
        except BaseException:os.close(fd);raise
    require(selected is not None,"Two CPU producers already attached; no third launch/queue")
    try:yield selected
    finally:os.close(selected)
