"""Inactive H058 GPU0 cached TRAIN metadata using the existing current selector API. Inert import."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,importlib,importlib.util,io,json,os,stat,sys,time,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/GPU0_postdrain_two_slot_pool_source_20261017_v2"
SOURCE=HERE+"/common_v1.py"
DRIVER=HERE+"/root_driver_v1.py"
CONTEXT=HERE+"/context_factory_v1.py"
DIRECT=HERE+"/direct_gpu0_v1.py"
MANIFEST=HERE+"/source_manifest.v1.json"
FIXED_PHASES=("train",)
CAPS=dict(reservation_bytes=256*2**20,scientific_output_bytes=192*2**20,
    auxiliary_bytes=60*2**20,directory_and_accounting_allowance_bytes=4*2**20,
    log_bytes=128*2**10,configure_fit_bytes=8*2**20,checkpoint_bytes=1572864,
    max_seconds=86400,max_cpu_rss_bytes=128*2**30,max_cuda_reserved_bytes=28*2**30,
    C3_obsolete_compute_envelope_bytes=0,peer_native_envelope_bytes=34*2**30,headroom_bytes=4*2**30,
    cleanup_seconds=30,terminal_evidence_margin_seconds=150,work_cutoff_margin_seconds=300)

BASE_POOL={"path":"src/arggnn/gpu_pool_v1.py","sha256":"ae1e0494ab672cfa3017759460eb06a4b4789eaef54928496c3c80314adcc0e1"}
GPU0_GUARD={"path":"research/reviews/GPU0_shared_execution_source_20261016_v1/gpu0_native_guard_v1.py","sha256":"3a08d3bd84adc95aafd332bc224fdb7aa38730d6c2f48dee8108944251b7ee69"}
GPU0_UUID="GPU-ff06ca1f-403a-f2d0-b584-71e177452667"
NVTOP_HELPER={"path":"scripts/qualify_official_argformer_runtime.py","sha256":"8fe652f87848f4a296e3d88555998fb388f11ae5c04d0858095e70c34b10b31c"}
CONFIGURE_FIT_SOURCE={"path":"src/arggnn/biological_candidate.py","sha256":"6b3a25d0e59026cab7189032eb41221feb7d1bdf69e15f6c737a672dfb711dd3"}
DEVELOPMENT_SOURCE={"path":"src/arggnn/development_execution.py","sha256":"4a2f87320615eb655a7700b6bbff4a7d293b1b16749468afa46f0de2d12e276f"}
BASEFIT={"path":"configs/census_tsinfer_fit_v1.json","sha256":"7c2436d5479e68beb1a8f3edb04ca0ee729a23be35ca2d7893827c0db68e3b00"}
PIDFD={'path': 'research/reviews/American_C3_native_MLM_full_source_v5/pidfd_backend_v1.py', 'sha256': '61be5d89357d1b95611e0ba26dd39d381ececf3ce6a36c347507cdc062373eca'}
OWNED={'path': 'research/reviews/American_C3_native_MLM_amendment_v12/owned_lifecycle_v2.py', 'sha256': '93d71acc40f1037d0b37acb2ca0859300ea4bd9741409e9bd00ca21bf100d633'}
RENEWAL={'path': 'research/reviews/H039_marker_repair_renewal_source_20261016_v1/marker_repair_renewal_v1.py', 'sha256': '491236877e8d71cae8bd7619880858ac501c57dd93e59479ca32dacacab20468'}
CERT={'path': 'research/reviews/H039_marker_repair_renewal_source_20261016_v1/renewal_certificate.root_active_v1.json', 'sha256': '66006202094d7a3ef2b29cbec28eb93f334505b72589f09d86e4b57498162db7'}
GUARD={'path':'research/reviews/Relate_failed64GiB_retention_successor_source_20261017_v1/relate_failed64_retention_v1.py','sha256':'b8b696a0df45c12b931e563600c2b52e89e73dd19c7d160351063687dcfb5164'}
POLICY={'path':'research/reviews/Relate_failed64GiB_retention_ROOT_activation_20261017_v1/activation_policy.root_active_v1.json','sha256':'fdec761f8cae04022093299955655f8ffa27166829653817165126d7ada6a65c'}
REPAIR={'path': 'research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json', 'sha256': 'f6f2d628199e8be73db8b5d90d428570d8afdefeab9dc5cd94bb5eafb39247db'}
MEMO={'path': 'research/reviews/resource_guard_hotpath_source_audit_v1/transaction_memo_adapter_v2.py', 'sha256': 'c972e9a779d424ceafdb258848614bf40994d1004e2c73fc5cd225ae99621364'}
NAMESPACE_MEMO={'path': 'research/reviews/current_mutex_contention_20261015_v1/consumer_namespace_memo_v1.py', 'sha256': '17a1377320f61c3c0cc8d3816d0259f9dae3121ed91bf7206a7ee63fcab8eb7f'}
V4={'path': 'src/arggnn/project_budget_v4.py', 'sha256': 'a093c7d86f57fee569cb99f4712250e9505a1119f3950fa44282c7ba8c57df47'}

SELECTOR_REF={'path':'research/reviews/Relate_failed64GiB_retention_ROOT_activation_20261017_v1/explicit_child_selector.root_active_v1.json','sha256':'e27d47a22d0da1b2e8f0bdf0818cb2c6c473d8c9ddb5424618bf97e704928642'}
SELECTOR_API_REF={'path':'research/reviews/H061_postselection_execution_source_20261017_v1/root_contract_v1.py','sha256':'c01479a37ad9725f5a3ab7a1759f31db48b4d4d8678ce2b723f1226fb2e74d98'}
_SELECTOR_API=None

def require(ok,message):
    if not ok:raise ValueError(message)

def local(path):
    require(type(path) is str and not Path(path).is_absolute() and '..' not in Path(path).parts,'Contained project path')
    p=ROOT/path
    require(p.resolve()==p and p.is_relative_to(ROOT),'Canonical contained path')
    return p

def ref(path):
    p=local(path)
    require(p.is_file() and not p.is_symlink() and p.stat().st_size<=32*2**20,'Bounded source/metadata')
    return dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def read(r):
    require(isinstance(r,dict) and set(r)=={'path','sha256'} and ref(r['path'])==r,'Exact actual source/JSON ref')
    return json.loads(local(r['path']).read_bytes())

def load(r,name):
    require(ref(r['path'])==r,'Exact source before private import')
    sp=importlib.util.spec_from_file_location(name,local(r['path']))
    m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m

def active(q):
    now=datetime.now(timezone.utc);a=datetime.fromisoformat(q['created_at_utc']);b=datetime.fromisoformat(q['expires_at_utc'])
    require(a.tzinfo and b.tzinfo and a<=now<b,'Current actual ROOT authority')

def unique(rows):
    d={}
    for r in rows:
        require(ref(r['path'])==r and (r['path'] not in d or d[r['path']]==r),'Complete exact unique bindings')
        d[r['path']]=r
    require(len(d)<=64,'Original64 direct-consumer ceiling')
    return list(d.values())

def guard_v1(*,consumer=True):
    """Reuse reviewed immutable selector/singleton/cache-before-memo API."""
    global _SELECTOR_API
    require(type(consumer) is bool,"Explicit ROOT mutation or read-only consumer")
    sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"scripts")]
    if _SELECTOR_API is None:
        api=load(SELECTOR_API_REF,"_H058_GPU0_existing_selector_contract")
        chosen=api.bind_selector(SELECTOR_REF)
        require(chosen["source_ref"]==GUARD and chosen["policy_ref"]==POLICY
            and chosen["accounting_mode"]=="Relate_failed64GiB_retention_v1",
            "Only exact current reviewed Relate child; no fallback or in-run swap")
        _SELECTOR_API=api
    return _SELECTOR_API.guard(consumer)

def raw(record):
    require(ref(record["path"])==record,"Exact source/metadata bytes")
    return local(record["path"]).read_bytes()

def save(directory,name,value):
    require(Path(name).name==name,"Flat admission metadata name")
    data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    require(len(data)<=8*2**20,"Compact root admission metadata")
    target=directory/name
    fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,"wb") as f:f.write(data);f.flush();os.fsync(f.fileno())
    return ref(str(target.relative_to(ROOT)))

def source_refs_v1():
    manifest=read(ref(MANIFEST));rows=manifest["refs"]
    require(all(ref(r["path"])==r for r in rows),"Exact sealed ROOT source closure")
    return rows

def science_v1(q):
    seal=read(q["science_handoff_ref"])
    require(seal.get("source_only") is True or seal.get("no_runtime_execution") is True
        or seal.get("execution_authorized") is False,"Actual source-only scientific handoff")
    cfg=read(q["config_ref"])
    producer=cfg["producer_ref"]
    require(seal.get("producer_ref")==producer and seal.get("source_manifest_ref")==cfg["source_manifest_ref"],
        "Scientific producer/manifest exactly match the actual source handoff reviewed by ROOT")
    for record in seal["files"]:
        require(ref(record["path"])==dict(path=record["path"],sha256=record["sha256"]),
            "Every sealed scientific source/handoff file unchanged")
    require(producer["path"].startswith(HERE+"/")
        and producer["path"].endswith("/producer_v1.py"),"H058 scientific successor family only")
    require(ref(producer["path"])==producer,"Actual exact producer")
    manifest=read(cfg["source_manifest_ref"])
    for r in manifest["refs"]:raw(r)
    require(producer in manifest["refs"],"Actual producer in scientific closure")
    protocol_path="research/reviews/H058_full_parent_CPU_execution_source_20261017_v1/protocol_v1.py"
    protocol=load(ref(protocol_path),"_H058_bound_frozen_protocol")
    require(cfg["scientific_protocol"]==protocol.fixed_protocol_v1(),"Frozen full-budget protocol")
    return cfg,producer,protocol,manifest

def validate_gate_v1(cfg,protocol,phase,caps):
    require(cfg["schema"]=="H058_full_parent_GPU0_cached_execution_config_v1" and cfg["execution_enabled"] is True
        and cfg["root_resource_caps_reviewed"] is True
        and cfg["max_cuda_reserved_bytes"]==caps["max_cuda_reserved_bytes"]
        and cfg["max_cpu_rss_bytes"]==CAPS["max_cpu_rss_bytes"]
        and cfg["output_cap_bytes"]==CAPS["scientific_output_bytes"]
        and cfg["checkpoint_file_cap_bytes"]==CAPS["checkpoint_bytes"]
        and cfg["evaluation_backend"]=="cpu","Exact28GiB CUDA/128GiB RSS/current192MiB output; CPU evaluation remains separately gated")
    completed=read(cfg["one_hold_completion_ref"]);wait=read(cfg["one_hold_direct_wait_ref"])
    terminal=read(cfg["one_hold_outer_terminal_ref"]);receipt=read(completed["qualification_ref"])
    require(completed.get("status")=="completed_H058_one_observed_hold_FP32_qualification"
        and receipt.get("status")=="PASS" and receipt.get("optimizer_updates")==0
        and receipt.get("schema")=="same_marker_one_observed_TRAIN000_FP32_qualification_v1"
        and receipt.get("parent_id")=="ap_expanded_train_000"
        and completed.get("actual_wait_ref")==cfg["one_hold_direct_wait_ref"]
        and wait.get("completed_ref")==completed["qualification_ref"] and wait.get("returncode")==0
        and wait.get("actual_direct_worker_wait") is True
        and wait.get("all_owned_processes_and_groups_drained") is True
        and wait.get("operational_failure") is None
        and terminal.get("actual_exit_code")==0 and terminal.get("actual_terminal_consumed") is True
        and terminal.get("completed_ref")==cfg["one_hold_completion_ref"]
        and terminal.get("schema")=="actual_root_tool_terminal_v1"
        and terminal.get("actual_wait_ref")==cfg["one_hold_direct_wait_ref"],"Actual consumed one-hold PASS gate")
    layout=read(cfg["layout_resource_proof_ref"])
    require(layout.get("root_authorizer")=="root_agent"
        and layout.get("status")=="admitted_first_scheduled_full_parent_step"
        and layout.get("source_count_ref")==cfg["source_count_ref"]
        and layout.get("physical_gpu")==0 and layout.get("reduced_data") is False
        and layout.get("counted_in_960_updates") is True,"Actual full-parent resource admission")
    if phase=="train":
        directive=read(cfg["full_training_directive_ref"])
        require(directive.get("root_authorizer")=="root_agent"
            and directive.get("full30pass_training_authorized") is True
            and directive.get("scientific_protocol")==protocol.fixed_protocol_v1(),"Actual full30pass authority")

def validate_native_noncoexistence_v3(q,*,fresh):
    """Future successful native completion and actual absence; never stop native."""
    v=read(q["native_noncoexistence_qualification_ref"]);active(v)
    require(v.get("schema")=="H058_ROOT_after_native_success_qualification_v1"
        and v.get("root_authorizer")=="root_agent" and v.get("status")=="qualified"
        and v.get("adapter_manifest_ref")==ref(MANIFEST) and v.get("H058_job_id")==q["job_id"]
        and v.get("physical_gpu")==0 and v.get("full_native_optimizer_updates_verified")==227990
        and v.get("actual_native_outer_terminal_consumed") is True
        and v.get("native_outer_exit_code")==0
        and v.get("no_native_GPU0_attachment_alive") is True
        and v.get("all_pre_handover_native_launch_resume_execution_routes_revoked") is True
        and v.get("original_controller_preserved_until_all_old_borrowers_drained") is True
        and v.get("new_pool_native_slot_is_distinct_from_completed_native_authority") is True,
        "Only genuine future completed native, never current live authority or stopped/unknown history")
    wait=read(v["native_actual_wait_ref"]);outer=read(v["native_actual_outer_terminal_ref"])
    require(wait.get("schema")=="native32_fixed_full_parent_actual_wait_v2"
        and wait.get("returncode")==0 and wait.get("actual_direct_worker_wait") is True
        and wait.get("all_owned_processes_and_groups_drained") is True
        and wait.get("operational_failure") is None
        and wait.get("completed_ref")==v["native_completed_ref"],
        "Actual native successful direct wait/drain/completion join")
    raw(v["native_completed_ref"])
    require(outer.get("root_authorizer")=="root_agent"
        and outer.get("actual_terminal_consumed") is True
        and outer.get("actual_exit_code")==0
        and outer.get("completed_ref")==v["native_completed_ref"]
        and outer.get("actual_wait_ref")==v["native_actual_wait_ref"],
        "Future ROOT-consumed actual native outer0 joined to its direct wait")
    absence=read(v["fresh_targeted_native_absence_ref"])
    require(absence.get("root_authorizer")=="root_agent"
        and absence.get("native_known_roots_and_contexts_absent") is True
        and absence.get("no_native_GPU0_attachment_alive") is True
        and absence.get("native_actual_wait_ref")==v["native_actual_wait_ref"]
        and absence.get("historical_unknowns_preserved") is True,
        "Future targeted absence after genuine native drain, no historical success rewrite")
    stamp=datetime.fromisoformat(v["observed_at_utc"]);now=datetime.now(timezone.utc)
    require(stamp.tzinfo and stamp<=now,"Nonfuture ROOT native absence qualification")
    if fresh:require((now-stamp).total_seconds()<=300,"Fresh native absence before admission/launch")
    require(isinstance(v.get("native_process_identities"),list) and v["native_process_identities"],
        "Exact known native PID/start identities to recheck throughout attachment")
    return v

def validate_request_v1(record):
    q=read(record);active(q)
    require(time.time()+CAPS["work_cutoff_margin_seconds"]<
        datetime.fromisoformat(q["expires_at_utc"]).timestamp(),
        "Fresh work authority includes finite cleanup and terminal-evidence margin")
    require(q.get("schema")=="H058_GPU0_cached_full_parent_ROOT_request_v1" and q.get("status")=="active"
        and q.get("root_authorizer")=="root_agent" and q.get("execution_authorized") is True
        and q.get("phase") in FIXED_PHASES and q.get("source_review_passed") is True
        and q["adapter_manifest_ref"]==ref(MANIFEST),"Explicit fresh reviewed H058 ROOT request")
    require(q["packet_directory"].startswith("research/reviews/H058_postdrain_ROOT_actual_")
        and q["output_directory"].startswith("experiments/results/H058_postdrain_full_parent_")
        and q["job_id"]=="H058-postdrain-fulltrain-"+q["instance_id"]
        and q["instance_id"] and all(ch in "abcdefghijklmnopqrstuvwxyz0123456789_-" for ch in q["instance_id"]),
        "Separate fresh finite GPU0 namespaces")
    preparation=local(q["preparation_directory"])
    require(preparation==local(record["path"]).parent and preparation.is_dir()
        and q["preparation_directory"].startswith("research/reviews/H058_postdrain_ROOT_preparation_")
        and all(preparation!=local(q[key]) and not preparation.is_relative_to(local(q[key]))
            and not local(q[key]).is_relative_to(preparation) for key in ("packet_directory","output_directory")),
        "Staging outside never-reserved packet/output")
    require(q.get("storage_policy_selector_ref")==SELECTOR_REF
        and q.get("TRAIN_truth_only") is True and q.get("VAL_truth_authorized") is False
        and q.get("TEST_authorized") is False and q.get("automatic_retry_resume_or_fallback") is False,
        "Exact current storage selector and TRAIN-only one-shot request")
    require(datetime.fromisoformat(q["expires_at_utc"]).timestamp()-time.time()>CAPS["max_seconds"]+600,
        "Fresh admission has full24h science plus cleanup/evidence margins")
    prior=read(q["prior_H058_CPU_disposition_ref"])
    require(prior.get("root_authorizer")=="root_agent" and prior.get("actual_terminal_consumed") is True
        and prior.get("all_owned_processes_and_groups_drained") is True
        and prior.get("old_launch_resume_write_routes_revoked") is True
        and prior.get("old_files_charges_clocks_and_counters_preserved") is True
        and prior.get("new_full_GPU_budget_explicitly_authorized") is True
        and prior.get("new_gpu_request_job_id")==q["job_id"]
        and isinstance(prior.get("actual_optimizer_steps"),dict)
        and prior.get("repeat_reason"),"ROOT must account for actual CPU outcome before new complete GPU fit")
    raw(prior["actual_wait_ref"]);raw(prior["actual_outer_terminal_ref"])
    require({k:v for k,v in q["caps"].items() if k!="max_cuda_reserved_bytes"}
        =={k:v for k,v in CAPS.items() if k!="max_cuda_reserved_bytes"}
        and type(q["caps"]["max_cuda_reserved_bytes"]) is int
        and q["caps"]["max_cuda_reserved_bytes"]==28*2**30,
        "Exact narrow caps; ROOT may choose positive CUDA cap up to28GiB")
    if q["phase"]=="train":
        validate_native_noncoexistence_v3(q,fresh=True)
        from . import pool_contract_v1 as pc
        pc.validate_H058_request_v1(q)
    local(q["packet_directory"]);local(q["output_directory"]);source_refs_v1()
    review=read(q["source_review_ref"]);raw(q["science_handoff_ref"])
    require(review.get("independent") is True and review.get("verdict")=="SOURCE_PASS"
        and review.get("source_ref")==ref(DRIVER)
        and review.get("source_manifest_ref")==ref(MANIFEST)
        and review.get("blockers")==[],"Actual complete independent source review before admission")
    cfg,producer,protocol,manifest=science_v1(q);validate_gate_v1(cfg,protocol,q["phase"],q["caps"])
    if q["phase"]=="evaluate":
        completion=read(q["train_completion_ref"]);train_wait=read(q["train_actual_wait_ref"])
        require(completion.get("schema")=="H058_full_parent_train_completion_v1"
            and completion.get("status")=="completed_fixed30pass_full_TRAIN"
            and completion.get("arms")==list(protocol.ARMS)
            and completion.get("steps")=={a:960 for a in protocol.ARMS}
            and completion.get("scientific_protocol")==protocol.fixed_protocol_v1()
            and completion.get("VAL_truth_read") is False and completion.get("outer_truth_read") is False
            and len(completion.get("history",[]))==30,"Actual fixed full30pass TRAIN completion")
        outer=read(q["train_outer_terminal_ref"])
        require(train_wait.get("returncode")==0 and train_wait.get("actual_direct_worker_wait") is True
            and train_wait.get("all_owned_processes_and_groups_drained") is True
            and train_wait.get("completed_ref")==q["train_completion_ref"]
            and outer.get("actual_exit_code")==0 and outer.get("actual_terminal_consumed") is True
            and outer.get("completed_ref")==q["train_completion_ref"]
            and outer.get("direct_wait_ref")==q["train_actual_wait_ref"],"Actual full TRAIN/outer wait before EVAL")
    return q,cfg,producer,protocol,manifest

def validate_input_v1(record):
    s=read(record);active(s);q,cfg,producer,protocol,manifest=validate_request_v1(s["root_stage_request_ref"])
    require(record["path"]==q["packet_directory"]+"/input.json"
        and s["schema"]=="H058_full_parent_ROOT_input_v1" and s["status"]=="active"
        and s["root_authorizer"]=="root_agent" and s["execution_authorized"] is True
        and s["phase"]==q["phase"] and s["job_id"]==q["job_id"]
        and s["output_directory"]==q["output_directory"] and s["producer_ref"]==producer
        and s["refs"]["config"]==q["config_ref"]
        and s["root_context_module_ref"]==ref(CONTEXT)
        and s["created_at_utc"]==q["created_at_utc"] and s["expires_at_utc"]==q["expires_at_utc"],
        "Exact admitted H058 input identity")
    storage=read(s["refs"]["storage"])
    require(storage["job_id"]==s["job_id"] and storage["output_directory"]==s["output_directory"]
        and storage["reserved_new_bytes"]==CAPS["reservation_bytes"]
        and storage["reservation"]["auxiliary_scopes"]==[str(local(q["packet_directory"]))]
        and storage["completed_overlay_policy_ref"]==POLICY
        and storage["budget_components"]["consumer_adapter"]==GUARD,"Actual current child whole256MiB output+packet reservation")
    needed=[SELECTOR_REF,ref(MANIFEST),ref(DRIVER),ref(CONTEXT),ref(DIRECT),s["root_stage_request_ref"],
        q["config_ref"],q["science_handoff_ref"],q["source_review_ref"],cfg["source_manifest_ref"],producer,s["refs"]["runtime"]]
    if q["phase"]=="train":needed.append(q["native_noncoexistence_qualification_ref"])
    require(len(storage["reservation"]["consumer_bindings"])<=64
        and all(r in storage["reservation"]["consumer_bindings"] for r in needed),"Original64 direct limit and full transitive closure")
    for path,sha in s["source_sha256"].items():raw(dict(path=path,sha256=sha))
    return s,q,cfg,storage,protocol

def worker_env_paths_v1(spec):
    base=str(local(spec["worker_auxiliary_scope"]))
    return dict(TMPDIR=base+"/tmp",TMP=base+"/tmp",TEMP=base+"/tmp",
        XDG_CACHE_HOME=base+"/cache",XDG_CONFIG_HOME=base+"/config",
        XDG_DATA_HOME=base+"/data",XDG_STATE_HOME=base+"/state",TORCH_HOME=base+"/cache/torch",
        TORCH_EXTENSIONS_DIR=base+"/cache/torch_extensions",PYTORCH_KERNEL_CACHE_PATH=base+"/cache/torch_kernels",
        TRITON_CACHE_DIR=base+"/cache/triton",NUMBA_CACHE_DIR=base+"/cache/numba",
        MPLCONFIGDIR=base+"/cache/matplotlib",PYTHONPYCACHEPREFIX=base+"/cache/pycache",
        CUDA_CACHE_PATH=base+"/cache/cuda",PIP_CACHE_DIR=base+"/cache/pip",
        HF_HOME=base+"/cache/huggingface",HF_HUB_CACHE=base+"/cache/huggingface/hub")

def charged_census_v1(spec,*,required_write_bytes=0):
    """Original no-follow file measurement, with current256MiB output+entire-packet scope."""
    def measured(path):
        total=entries=0
        for directory,dirs,files in os.walk(local(path),followlinks=False):
            dp=Path(directory);st=dp.lstat();require(stat.S_ISDIR(st.st_mode),"Canonical charged directory")
            total+=st.st_blocks*512
            for name in dirs+files:
                p=dp/name
                try:st=p.lstat()
                except FileNotFoundError:continue
                entries+=1;require(entries<=8192,"Bounded charged scope entries")
                require(stat.S_ISDIR(st.st_mode) or stat.S_ISREG(st.st_mode),"No charged links/special nodes")
                if stat.S_ISREG(st.st_mode):
                    require(st.st_nlink==1,"No charged hardlinks")
                    total+=max(st.st_size,st.st_blocks*512)
        return total
    packet=str(Path(spec["worker_auxiliary_scope"]).parent)
    output=measured(spec["output_directory"]);whole_packet=measured(packet)
    auxiliary=measured(spec["worker_auxiliary_scope"])
    require(output<=CAPS["scientific_output_bytes"]
        and auxiliary+required_write_bytes<=CAPS["auxiliary_bytes"]
        and whole_packet-auxiliary<=CAPS["directory_and_accounting_allowance_bytes"]
        and output+whole_packet+required_write_bytes<=CAPS["reservation_bytes"],
        "Current192MiB scientific+60MiB AUX+4MiB packet and256MiB whole scope")
    return {spec["output_directory"]:output,packet:whole_packet}

def compact_v1(spec,name,value,g,storage,*,cap=2**20):
    directory=local(spec["worker_auxiliary_scope"])
    raw_bytes=(json.dumps(value,sort_keys=True,allow_nan=False)+"\n").encode()
    require(len(raw_bytes)<=cap,"Fixed compact operational receipt cap")
    charged_census_v1(spec,required_write_bytes=len(raw_bytes)+8192)
    g.budget_check_v9(storage,boundary=True,required_write_bytes=len(raw_bytes)+8192)
    active(spec)  # A contended original guard may have consumed the time margin.
    return save(directory,name,value)

def require_work_window_v2(spec):
    require(time.time()+CAPS["work_cutoff_margin_seconds"]<
        datetime.fromisoformat(spec["expires_at_utc"]).timestamp(),
        "Scientific work stops before reserved cleanup and terminal-evidence margin")

def authenticated_base_fit_v2(reader):
    # Only dynamically executed runtime sources are selected from this historic
    # nested map. Unused old AGENTS/config metadata is not execution authority.
    value=reader.decode_json(BASEFIT)
    refs=read(ref(MANIFEST))["refs"]
    require(value["gpu_runtime"]["nvtop_helper"]==NVTOP_HELPER["path"],
        "Exact executed original NVTOP helper path")
    for record in (NVTOP_HELPER,CONFIGURE_FIT_SOURCE,DEVELOPMENT_SOURCE):
        require(value["source_sha256"].get(record["path"])==record["sha256"]
            and record in refs,"Executed BASEFIT source agrees with reviewed aggregate ROOT manifest")
        raw(record)
    return value
