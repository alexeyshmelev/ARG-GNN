"""Thin ROOT metadata/guard adapter for original VAL003/004 preparation. Inert import."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import contextlib,fcntl,hashlib,importlib.util,json,os,stat,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/American_C3_VAL003004_native14v3_preparation_source_20261016_v2"
SOURCE=REL+"/root_contract_v1.py";DRIVER=REL+"/root_driver_v1.py";WORKER=REL+"/worker_entry_v1.py"
MANIFEST=REL+"/source_manifest.v1.json";SOURCE_PINS=MANIFEST
READINESS=REL+"/readiness.v1.json";BASIS=REL+"/scientific_basis.v1.json";CAPFILE=REL+"/caps.template.inactive.v1.json"
GUARD_REF={'path': 'research/reviews/native14_stopped_retention_successor_source_20261016_v3/native14_stopped_retention_v1.py', 'sha256': '60d010e393bda25f662381051b3fd3d4e4decfa034b10c3baad3c3f30a6a196c'}
NATIVE_MANIFEST_REF={'path': 'research/reviews/native14_stopped_retention_successor_source_20261016_v3/source_manifest.v1.json', 'sha256': 'c2bae5575ae0e8e68e152b0edde0b7b2d53e4fc7398a3a8501ce278e23a84fa6'}
NATIVE_HELPER_REF={'path': 'research/reviews/native14_stopped_retention_successor_source_20261016_v3/retained_nodes_v1.py', 'sha256': '83f544e45eba5190174b70d43bf84ed4f0b58f80159be593b4e8392c3e4afb92'}
POLICY_PATH="research/reviews/native14_stopped_retention_ROOT_activation_20261016_v3/activation_policy.root_active_v1.json"
POLICY_REF=None
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
PARENTS=("american_c3_val_003","american_c3_val_004");STAGES=("infer","native_paths","plan")
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
    require(parent in PARENTS and stage in STAGES,"Only original VAL003/004 inference/native-path/plan")
    PARENT=parent;STAGE=stage;PHASE=stage+"_val_"+parent[-3:];PHASES=CPU_PHASES=(PHASE,)
    CAPS=read(ref(CAPFILE))["stages"][stage]
    status={"infer":"completed_current_C3_inference","native_paths":"completed_native_C3_path_preparation","plan":"completed_C3_GNN_current980_"+PHASE}
    STATUSES={PHASE:status[stage]};ENTRY=REL+"/plan_worker_v1.py" if stage=="plan" else WORKER
    basis=read(ref(BASIS));RUNTIME_REF=basis[{"infer":"inference_runtime_ref","native_paths":"native_runtime_ref","plan":"plan_runtime_ref"}[stage]]
    HOOK_REF=basis["plan_hook_ref"]
    TRAIN_SOURCE_REF=next(r for r in read(ref(MANIFEST))["refs"] if r["path"]=="src/arggnn/american_c3_gnn_train_v1.py")
def phase_paths(phase):
    require(phase==PHASE,"Exact selected phase")
    return local(REL+"/admitted_val"+PARENT[-3:]+"_"+STAGE+"_v2"),"experiments/results/American_C3_val"+PARENT[-3:]+"_full30mb_"+STAGE+"_native14v3_v2"
def source_paths():
    rows=read(ref(MANIFEST))["refs"]
    for r in rows:raw(r)
    return tuple(r["path"] for r in rows)
def bind_policy(record):
    global POLICY_REF
    require(isinstance(record,dict) and record.get("path")==POLICY_PATH,"Explicit exact actual native14v3 ROOT policy ref")
    raw(record)
    require(POLICY_REF is None or POLICY_REF==record,"Immutable native14v3 policy binding; no policy swap")
    p=read(record)
    require(p.get("active") is True and p.get("schema")=="native14_exact_stopped_retention_policy_v1"
        and p.get("source_ref")==GUARD_REF and p.get("source_manifest_ref")==NATIVE_MANIFEST_REF
        and p.get("measurement_source_ref")==NATIVE_HELPER_REF and p.get("journal")==".runtime/storage/native14_stopped_retention_v3"
        and p.get("root_authorizer")=="root_agent","Exact active reviewed native14v3 guard/source/manifest policy")
    for key in ("root_activation_ref","permanent_revocation_ref","source_review_ref"):raw(p[key])
    activation=read(p["root_activation_ref"])
    require(activation.get("active") is True and activation.get("root_authorizer")=="root_agent"
        and activation.get("source_ref")==GUARD_REF and activation.get("source_manifest_ref")==NATIVE_MANIFEST_REF
        and activation.get("measurement_source_ref")==NATIVE_HELPER_REF
        and activation.get("new_journal")==p["journal"] and activation.get("permanent_revocation_ref")==p["permanent_revocation_ref"]
        and activation.get("source_review_ref")==p["source_review_ref"]
        and activation.get("frozen_parent_pins")==p["base_pins"]
        and activation.get("frozen_parent_projected_bytes")==p["base_projected_bytes"],
        "Actual exact frozen-parent ROOT activation agrees with immutable policy")
    review=read(p["source_review_ref"])
    require(review.get("verdict") in ("PASS","SOURCE_PASS") and review.get("source_ref")==GUARD_REF
        and review.get("source_manifest_ref")==NATIVE_MANIFEST_REF and review.get("measurement_source_ref")==NATIVE_HELPER_REF,
        "Exact native14v3 independent source/manifest/helper review")
    require(read(p["permanent_revocation_ref"]).get("active") is True,"Actual permanent exact-key ROOT revocation")
    POLICY_REF=dict(record)
    return p
def original_guard():
    require(POLICY_REF is not None,"Explicit actual native14v3 policy required before guard configuration")
    bind_policy(POLICY_REF)
    for r in (GUARD_REF,POLICY_REF,REPAIR_REF,RENEWAL_REF,CERT_REF,MEMO_REF,NAMESPACE_REF,V4_REF):raw(r)
    sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"scripts")]
    g=load(GUARD_REF,"_American_VAL003004_actual_native14v3")
    g.configure_with_renewal_v1(REPAIR_REF,policy_ref=POLICY_REF)
    require(g.MODE=="native14_stopped_retention_v3" and all(getattr(g.budget.v4,n).__module__==g.budget.v4.__name__
        for n in ("canonical","file_bytes","mutex")),"Fresh original unwrapped current Census mutation guard")
    return g
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
def actual_prior(parent,stage):
    row=existing_simulation(parent)
    if stage=="infer":
        return dict(completion_ref=row["simulation_completion_ref"],wait_ref=row["simulation_wait_ref"],
            outer_ref=row["simulation_outer_terminal_ref"],close_ref=row["simulation_close_ref"],
            observation_ref=row["simulation_observation_ref"],closed_key=row["simulation_closed_key"])
    previous="infer" if stage=="native_paths" else "native_paths"
    folder=local(REL+"/admitted_val"+parent[-3:]+"_"+previous+"_v2")
    output="experiments/results/American_C3_val"+parent[-3:]+"_full30mb_"+previous+"_native14v3_v2"
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
    require(close["key"]=="H041/American-C3-val"+parent[-3:]+"-full30mb-"+previous+"-native14v3-v2"
        and close["value"]["future_writes_and_relaunch_revoked"] is True and close["value"]["proof"]["completed_ref"]==cp
        and close["value"]["proof"]["direct_wait_ref"]==wr,"Actual predecessor ordinary completed-scope close")
    require(done["status"]=={"infer":"completed_current_C3_inference","native_paths":"completed_native_C3_path_preparation"}[previous],
        "Exact unchanged producer completion")
    obs=observation(done["observation_ref"],parent)
    require(obs["genealogy_kind"]=="inferred_tsinfer_tsdate_CURRENT_RECONSTRUCTION_not_simulated_true_ARG"
        and obs["sites_and_sample_identities_verified"] is True and obs["source_observation_ref"]==row["simulation_observation_ref"],
        "All original inferred sites and sample identities; no trimming/subsampling")
    result=dict(completion_ref=cp,wait_ref=wr,outer_ref=tr,close_ref=cr,observation_ref=done["observation_ref"],closed_key=close["key"])
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
def validate_request(record):
    q=read(record);bind_policy(q["policy_ref"]);select(q["parent_id"],q["stage"]);active(q,300)
    require(q["schema"]=="American_C3_VAL003004_native14v3_request_v2" and q["active"] is True
        and q["root_authorizer"]=="root_agent" and q["phase"]==PHASE and q["execution_authorized"] is True
        and q["source_review_passed"] is True and q["adapter_manifest_ref"]==ref(MANIFEST)
        and q["guard_ref"]==GUARD_REF and q["policy_ref"]==POLICY_REF and q["maximum_CPU_producers"]==2
        and q["query_truth_access_authorized"] is False and q["GNN_optimization_authorized"] is False
        and q["TEST_access_authorized"] is False and q["all30_C3_prediction_seals_before_VAL_truth"] is True,
        "Exact fresh independently reviewed CPU preparation authority")
    reviewed(q["source_review_ref"]);source_paths()
    prior=actual_prior(PARENT,STAGE)
    require(q["prior"]==prior and q["scientific_config"]==scientific_config(PARENT,STAGE,prior,True),"Actual complete predecessor and unchanged original scientific recipe")
    require(q["bindings"]==dict(parent_id=PARENT,plan_config=q["scientific_config"]) if STAGE=="plan"
        else q["bindings"]==dict(parent_id=PARENT,scientific_config=q["scientific_config"]),"Exact method inputs only; no truth payload inputs")
    require(read(q["caps_ref"])==worker_caps(STAGE) and q["runtime_ref"]==RUNTIME_REF,"Exact original chosen runtime and caps")
    return q,read(q["caps_ref"]),read(RUNTIME_REF)
def bind(parent,stage,review_ref,policy_ref):
    bind_policy(policy_ref);original_guard()
    select(parent,stage);reviewed(review_ref);source_paths()
    prior=actual_prior(parent,stage);cfg=scientific_config(parent,stage,prior,True);now=datetime.now(timezone.utc)
    end=now+timedelta(seconds=CAPS["wall_seconds"]+600)
    require(end<datetime.fromisoformat(read(CERT_REF)["renewed_expires_at_utc"]),"Original renewal covers whole wall/drain/evidence authority")
    caps_ref=save(local(REL),"caps_val"+parent[-3:]+"_"+stage+".root_active_v1.json",worker_caps(stage))
    q=dict(schema="American_C3_VAL003004_native14v3_request_v2",active=True,root_authorizer="root_agent",
        execution_authorized=True,parent_id=parent,stage=stage,phase=PHASE,created_at_utc=now.isoformat(),expires_at_utc=end.isoformat(),
        adapter_manifest_ref=ref(MANIFEST),guard_ref=GUARD_REF,policy_ref=POLICY_REF,source_review_ref=review_ref,source_review_passed=True,
        prior=prior,scientific_config=cfg,caps_ref=caps_ref,runtime_ref=RUNTIME_REF,maximum_CPU_producers=2,
        query_truth_access_authorized=False,GNN_optimization_authorized=False,TEST_access_authorized=False,
        all30_C3_prediction_seals_before_VAL_truth=True,
        bindings=dict(parent_id=parent,plan_config=cfg) if stage=="plan" else dict(parent_id=parent,scientific_config=cfg))
    r=save(local(REL),"request_val"+parent[-3:]+"_"+stage+".root_active_v1.json",q)
    validate_request(r);return r
def unique(rows):
    d={}
    for r in rows:raw(r);require(r["path"] not in d or d[r["path"]]==r,"Exact unique role");d[r["path"]]=r
    require(len(d)<=64,"Original direct64 ceiling");return list(d.values())
def direct_roles(request_ref,q,cfgref,closure,runtime):
    basis=read(ref(BASIS))
    result=[ref(SOURCE),ref(DRIVER),ref(WORKER),ref(REL+"/plan_worker_v1.py"),ref(MANIFEST),
        ref(READINESS),ref(BASIS),ref(CAPFILE),request_ref,q["caps_ref"],cfgref,closure,runtime,q["source_review_ref"],
        GUARD_REF,POLICY_REF,REPAIR_REF,RENEWAL_REF,CERT_REF,MEMO_REF,NAMESPACE_REF,V4_REF,PIDFD,OWNED,
        ref(SLOT_REL+"/cpu_slot_0.lock"),ref(SLOT_REL+"/cpu_slot_1.lock"),
        basis["common_producer_ref"],basis["inference_producer_ref"],basis["simulation_sanitize_helper_ref"],
        basis["native_producer_ref"],basis["author_path_helper_ref"],basis["plan_hook_ref"],basis["plan_dataset_ref"],
        basis[STAGE+"_config_ref"] if STAGE in ("infer","native_paths") else basis["plan_config_ref"]]
    result.extend(q["prior"][k] for k in ("completion_ref","wait_ref","outer_ref","close_ref","observation_ref"))
    return result
def input_spec(record):
    s=read(record);q,caps,_=validate_request(s["root_stage_request_ref"]);active(s,300)
    packet,output=phase_paths(PHASE);storage=read(s["refs"]["storage"])
    require(record["path"]==str((packet/"input.json").relative_to(ROOT)) and s["phase"]==PHASE and s["parent_id"]==PARENT
        and s["root_driver_ref"]==ref(DRIVER) and s["output_directory"]==output and s["storage"]==storage
        and s["created_at_utc"]==q["created_at_utc"] and s["expires_at_utc"]==q["expires_at_utc"]
        and s["runtime_ref"]==RUNTIME_REF and s["native14_policy_ref"]==POLICY_REF and s["GPU_authorized"] is False,"Exact admitted original CPU stage")
    require(storage["reservation"]["reserved_bytes"]==CAPS["reservation_bytes"]
        and storage["completed_overlay_policy_ref"]==POLICY_REF and storage["budget_components"]["consumer_adapter"]==GUARD_REF,
        "Actual current Census full charged stage reservation")
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
