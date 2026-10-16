"""First reserved European outer parent. Default inert; ROOT-only future admission."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, ctypes, hashlib, importlib, importlib.util, json, os, selectors, signal, stat, subprocess, sys, time
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/European_outer8_current_policy_successor_source_20261016_v1"
BASE_HERE="research/reviews/European_outer8_preparation_source_20261016_v3"
SOURCE=HERE+"/prepare_outer000_v1.py"
MANIFEST=HERE+"/source_manifest.v1.json"
REGISTRY=BASE_HERE+"/outer8_registry.v1.json"
METHODS=BASE_HERE+"/frozen_method_ids.v1.json"
PROTOCOL=BASE_HERE+"/protocol.v1.json"
CONFIG=dict(path="configs/expanded_full_panel_v1.json",sha256="662ba8e4613ce85b9aaec9909e873c9cddf9c62672b895e7d9e02f851d518d08")
PRODUCER=dict(path="src/arggnn/prepare_expanded_full_panel_v1.py",sha256="2ff091677986d5cfec2936c2acf53eeed71c957192f9f048cb47f59d3351913e")
ORIGINAL=dict(path="src/arggnn/prepare_ancestralpaths_full_panel_discovery_v1.py",sha256="ead823158a001dabdbf2224d812e7ef8079b6af4382f6e8e10ca0e0894760e71")
SELECTOR=dict(path="research/reviews/native7_H050_stopped_retention_successor_source_20261016_v1/explicit_policy_selector_v1.py",sha256="2a991ca6e2da3f4ba05888bc44e4cd53025f929abf88760aa4f93571e13bc87b")
APPROVED={
 "native14_stopped_retention_v3":dict(source_ref={'path': 'research/reviews/native14_stopped_retention_successor_source_20261016_v3/native14_stopped_retention_v1.py', 'sha256': '60d010e393bda25f662381051b3fd3d4e4decfa034b10c3baad3c3f30a6a196c'},manifest_ref={'path': 'research/reviews/native14_stopped_retention_successor_source_20261016_v3/source_manifest.v1.json', 'sha256': 'c2bae5575ae0e8e68e152b0edde0b7b2d53e4fc7398a3a8501ce278e23a84fa6'},policy_ref={'path': 'research/reviews/native14_stopped_retention_ROOT_activation_20261016_v3/activation_policy.root_active_v1.json', 'sha256': '72ed727feb1a824fa8f9c2b31ce374dca6728ca4e655e9562b4dedc84122f417'}),
 "native7_H050_stopped_retention_v1":dict(source_ref={'path': 'research/reviews/native7_H050_stopped_retention_successor_source_20261016_v1/stopped_retention_v1.py', 'sha256': 'c9392ed88b76a5a6c1394d13ff78b55df83c1b16d8186d23b4d3110b46008e8f'},manifest_ref={'path': 'research/reviews/native7_H050_stopped_retention_successor_source_20261016_v1/source_manifest.v1.json', 'sha256': 'cb6c3af2858d007d74edb08742f8633ddef5f2a9cb4c6cd5b3682205fee32d88'},policy_ref=None)}
_SELECTED_REF=_SELECTED_VALUE=_SELECTOR_MODULE=_GUARD=None
REPAIR=dict(path="research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json",sha256="f6f2d628199e8be73db8b5d90d428570d8afdefeab9dc5cd94bb5eafb39247db")
NAMESPACE=dict(path="research/reviews/current_mutex_contention_20261015_v1/consumer_namespace_memo_v1.py",sha256="17a1377320f61c3c0cc8d3816d0259f9dae3121ed91bf7206a7ee63fcab8eb7f")
V4=dict(path="src/arggnn/project_budget_v4.py",sha256="a093c7d86f57fee569cb99f4712250e9505a1119f3950fa44282c7ba8c57df47")
PIDFD=dict(path="research/reviews/American_C3_native_MLM_full_source_v5/pidfd_backend_v1.py",sha256="61be5d89357d1b95611e0ba26dd39d381ececf3ce6a36c347507cdc062373eca")
OWNED=dict(path="research/reviews/American_C3_native_MLM_amendment_v12/owned_lifecycle_v2.py",sha256="93d71acc40f1037d0b37acb2ca0859300ea4bd9741409e9bd00ca21bf100d633")
PARENT="ap_expanded_outer_000"
OUTPUT="data/expanded_full_panel_v1/"+PARENT
JOB="European-outer000-original-full-panel-v1"
CAPS=dict(reservation_bytes=4*2**30,output_bytes=4000000000,auxiliary_bytes=128*2**20,
          cpu_threads=8,rss_bytes=32*2**30,wall_seconds=43200,cleanup_seconds=30,
          work_reserve_seconds=300,evidence_reserve_seconds=150,log_bytes=2*2**20)
OBSERVED=["genotypes.vcf","sample_manifest.json","sample_order.txt","sample_ages.generations",
          "population_labels.txt","site_coordinate_manifest.json","uniform_recombination_map.txt",
          "scoring_coordinates.npz","observation_manifest.json"]
PRIVATE=["census_ancestry.trees","full_truth_intervals.npz","target_schema.json"]

def need(ok,msg):
    if not ok:raise ValueError(msg)
def local(name):
    p=Path(name);p=p if p.is_absolute() else ROOT/p
    need(p!=ROOT and p.is_relative_to(ROOT) and ".." not in p.parts and p.resolve()==p,"Canonical contained path")
    return p
def ref(name):
    p=local(name);s=p.lstat()
    need(stat.S_ISREG(s.st_mode) and s.st_nlink==1 and s.st_size<=32*2**20,"Compact regular source/metadata only")
    return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def raw(r):
    need(isinstance(r,dict) and set(r)=={"path","sha256"} and ref(r["path"])==r,"Exact source/metadata SHA")
    return local(r["path"]).read_bytes()
def read(r):return json.loads(raw(r))
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def load(r,name):
    raw(r);sp=importlib.util.spec_from_file_location(name,local(r["path"]))
    m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def current(q,margin=0):
    now=datetime.now(timezone.utc);created=datetime.fromisoformat(q["created_at_utc"]);expires=datetime.fromisoformat(q["expires_at_utc"])
    need(created.tzinfo and expires.tzinfo and created<=now and now.timestamp()+margin<expires.timestamp(),"Current actual ROOT authority/margin")
def sources():
    m=read(ref(MANIFEST))
    for r in m["refs"]+m.get("expected_original_metadata_and_assets",[]):raw(r)
    need(ref(SOURCE) in m["refs"],"Actual wrapper in sealed closure")
    return m["refs"]
def freeze(r):
    f=read(r);inventory=read(ref(METHODS))
    need(f.get("schema")=="European_outer8_method_freeze_v1" and f.get("active") is True
         and f.get("root_authorizer")=="root_agent" and f.get("identity_inventory_ref")==ref(METHODS)
         and f.get("method_ids")==inventory["primary_method_ids"]
         and f.get("all_methods_fixed_before_first_outer_generation") is True
         and f.get("H058_automatically_included") is False
         and f.get("outer_data_or_truth_used_for_selection") is False,"Actual fixed methods before TEST generation")
    stamp=datetime.fromisoformat(f["frozen_at_utc"])
    need(stamp.tzinfo and stamp<=datetime.now(timezone.utc),"Actual nonfuture freeze")
    return f
def request(r):
    q=read(r);current(q,CAPS["wall_seconds"]+600);sources();f=freeze(q["method_freeze_ref"])
    need(q.get("schema")=="European_outer000_preparation_ROOT_request_v1" and q.get("status")=="active"
         and q.get("root_authorizer")=="root_agent" and q.get("generation_and_native_Relate_authorized") is True
         and q.get("GPU_authorized") is False and q.get("outer_scoring_authorized") is False
         and q.get("source_manifest_ref")==ref(MANIFEST) and q.get("caps")==CAPS
         and q.get("parent_id")==PARENT and q.get("output_directory")==OUTPUT
         and q.get("packet_directory","").startswith("research/reviews/European_outer000_ROOT_actual_")
         and q.get("automatic_retry_or_reseed") is False,"Exact first-parent one-shot ROOT request")
    cfg=read(CONFIG);registry=read(ref(REGISTRY))
    rows=[x for x in cfg["parent_registry"] if x["split"]=="outer"]
    need(rows==registry["outer_rows"] and len(rows)==8 and rows[0]["parent_id"]==PARENT
         and all(x["downstream_role"]=="UNTOUCHED_OUTER" and x["evaluation_authorized"] is False for x in rows)
         and cfg["observation_contract"]["Relate_input_whitelist"]==OBSERVED,"Original reserved registry/whitelist")
    need(datetime.fromisoformat(f["frozen_at_utc"])<=datetime.fromisoformat(q["created_at_utc"]),"Freeze precedes actual request")
    cpus=q["cpu_ids"];need(isinstance(cpus,list) and len(cpus)==8 and len(set(cpus))==8
         and all(type(x) is int for x in cpus) and set(cpus)<=os.sched_getaffinity(0),"Eight actually available CPU identities")
    review=read(q["source_review_ref"])
    need(review.get("verdict") in ("PASS","SOURCE_PASS") and review.get("source_manifest_ref")==ref(MANIFEST),"Actual independent source review")
    prospective(q,cfg)
    return q,cfg

def selection(selector_ref):
    """Read exact ROOT choice only. Does not configure or probe a guard."""
    value=read(selector_ref)
    need(value.get("schema")=="explicit_ROOT_storage_policy_selector_v1"
         and value.get("active") is True and value.get("root_authorizer")=="root_agent"
         and value.get("selector_source_ref")==SELECTOR
         and value.get("accounting_mode") in APPROVED
         and value.get("no_fallback_discovery_upgrade_or_in_run_swap") is True
         and value.get("future_ROOT_adapters_only") is True
         and value.get("scheduling_ceiling_bytes")==980000000000
         and value.get("human_limit_bytes")==1000000000000
         and value.get("warning_threshold_bytes")==950000000000,
         "Exact explicit reviewed ROOT policy choice; no discovery or fallback")
    approved=APPROVED[value["accounting_mode"]]
    need(value.get("source_ref")==approved["source_ref"]
         and value.get("source_manifest_ref")==approved["manifest_ref"]
         and (approved["policy_ref"] is None or value.get("policy_ref")==approved["policy_ref"]),
         "Exact sealed current native14v3 or separately activated reviewed child")
    for key in ("source_ref","source_manifest_ref","measurement_source_ref","source_review_ref","repair_authority_ref"):
        raw(value[key])
    return value

def registration(q):
    """Authenticate the real prospective registration snapshot, not readiness flags."""
    receipt=read(q["registration_ref"])
    need(receipt.get("schema")=="European_outer8_prospective_registration_v1"
         and receipt.get("root_authorizer")=="root_agent" and receipt.get("status")=="registered"
         and receipt.get("condition")=="European_outer8_frozen_predictor_independent_confirmation_v1"
         and receipt.get("linked_hypothesis_ids")==["H038","H050"]
         and receipt.get("method_freeze_ref")==q["method_freeze_ref"]
         and receipt.get("protocol_ref")==ref(PROTOCOL) and receipt.get("registry_ref")==ref(REGISTRY)
         and receipt.get("decisions_snapshot_ref")==q["registration_decisions_ref"]
         and receipt.get("ledger_snapshot_ref")==q["registration_ledger_ref"],
         "Actual exact independent-confirmation registration and immutable snapshots")
    decisions=read(q["registration_decisions_ref"])
    entries=[d for d in decisions["decisions"] if d.get("id")==receipt["hypothesis_id"]]
    need(len(entries)==1,"Exactly one actual registered hypothesis in decisions snapshot")
    decision=entries[0]
    for key in ("condition","linked_hypothesis_ids","method_freeze_ref","protocol_ref","registry_ref"):
        need(decision.get(key)==receipt[key],"Registered independent condition "+key)
    need(decision.get("outer_tuning_authorized") is False
         and isinstance(decision.get("repeat_reason"),str) and decision["repeat_reason"].strip(),
         "Explicit independent confirmation reason and no outer tuning")
    ledger=raw(q["registration_ledger_ref"]).decode("utf-8")
    need(receipt["hypothesis_id"] in ledger and "independent" in ledger.lower(),
         "Actual ledger snapshot contains the independent registration")
    stamp=datetime.fromisoformat(receipt["registered_at_utc"])
    freeze_time=datetime.fromisoformat(read(q["method_freeze_ref"])["frozen_at_utc"])
    need(stamp.tzinfo and freeze_time<=stamp<=datetime.fromisoformat(q["created_at_utc"]),
         "Actual freeze and registration precede new request")
    return receipt

def prospective(q,cfg):
    registration(q);selection(q["policy_selector_ref"])
    runtime=read(q["runtime_ref"]);qualification=read(q["runtime_qualification_ref"])
    need(runtime.get("schema")=="European_outer000_CPU_runtime_v1"
         and runtime.get("device")=="cpu" and runtime.get("CUDA_VISIBLE_DEVICES")==""
         and runtime.get("interpreter")==str(ROOT/".runtime/env/bin/python")
         and runtime.get("config_runtime")==cfg["runtime"] and runtime.get("native_threads")==8
         and runtime.get("cpu_ids")==q["cpu_ids"],"Actual complete original generation runtime")
    need(qualification.get("schema")=="European_outer000_runtime_qualification_v1"
         and qualification.get("root_authorizer")=="root_agent" and qualification.get("status")=="qualified"
         and qualification.get("runtime_ref")==q["runtime_ref"]
         and qualification.get("operational_manifest_ref")==ref(MANIFEST)
         and qualification.get("config_ref")==CONFIG and qualification.get("producer_ref")==PRODUCER
         and qualification.get("unresolved_items")==[],"Actual reviewed generation runtime qualification")

def generation_direct_refs_v1(q,request_ref):
    """Keep every returned guard separately. This is only the additional role set."""
    return [ref(MANIFEST),ref(SOURCE),CONFIG,PRODUCER,ORIGINAL,ref(REGISTRY),ref(METHODS),ref(PROTOCOL),
        request_ref,q["runtime_ref"],q["method_freeze_ref"],q["source_review_ref"],PIDFD,OWNED,NAMESPACE,V4,
        SELECTOR,q["policy_selector_ref"],q["registration_ref"],q["registration_decisions_ref"],
        q["registration_ledger_ref"],q["runtime_qualification_ref"]]

def consumer_direct_refs_v1(q,stage):
    need(stage in ("validate","score"),"Exact complete matrix consumer stage")
    refs=[ref(MANIFEST),ref(HERE+"/check_prediction_barrier_v1.py"),ref(SOURCE),
        ref(BASE_HERE+"/source_manifest.v1.json"),ref(PROTOCOL),
        ref(BASE_HERE+"/outer_prediction_validator_v2.py"),ref(BASE_HERE+"/outer_adapters_v2.py"),
        ref(BASE_HERE+"/h050_observed_bridge_v3.py"),ref(REGISTRY),ref(METHODS),NAMESPACE,V4,
        q["root_request_ref"],q["runtime_ref"],q["method_freeze_ref"],q["entry_review_ref"],
        q["runtime_qualification_ref"],q["bundle_ref"],SELECTOR,q["policy_selector_ref"],
        q["registration_ref"],q["registration_decisions_ref"],q["registration_ledger_ref"]]
    if stage=="score":
        refs.extend([ref(BASE_HERE+"/score_outer_v2.py"),q["scorer_review_ref"],q["actual_integration_qualification_ref"]])
    return refs


def guard(selector_ref,consumer):
    """Configure exact selected module once before memo. Never reselect after memo."""
    global _SELECTED_REF,_SELECTED_VALUE,_SELECTOR_MODULE,_GUARD
    chosen=selection(selector_ref)
    if _GUARD is None:
        selector=load(SELECTOR,"_European_outer_explicit_selector")
        g=selector.load_selected_guard_v1(selector_ref)
        need(g.MODE==chosen["accounting_mode"] and g._POLICY_REF==chosen["policy_ref"],
             "Actual selected guard/policy identity")
        _SELECTED_REF=selector_ref.copy();_SELECTED_VALUE=chosen;_SELECTOR_MODULE=selector;_GUARD=g
    else:
        need(selector_ref==_SELECTED_REF and chosen==_SELECTED_VALUE
             and _GUARD.MODE==chosen["accounting_mode"] and _GUARD._POLICY_REF==chosen["policy_ref"],
             "Same immutable configured selection for lifetime; no reconfigure")
    g=_GUARD
    if consumer and not hasattr(g,"_outer_memo"):
        need(all(getattr(g.budget.v4,n).__module__==g.budget.v4.__name__
                 for n in ("canonical","file_bytes","mutex")),"Fresh original guard before memo")
        ns=load(NAMESPACE,"_European_outer_original_memo")
        g._outer_memo=ns.install_consumer_v1(g.budget.v4,g.budget,consumer_role="read_only_consumer",
            expected_source_path=local(V4["path"]),expected_source_sha256=V4["sha256"])
    return g
def put(path,value,cap=2**20):
    p=local(path);data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    need(len(data)<=cap,"Compact immutable metadata")
    fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,"wb") as f:f.write(data);f.flush();os.fsync(f.fileno())
    return ref(p)
def boundary(g,spec,planned=0,target="output"):
    current(spec)
    state=g.budget_check_v9(spec["storage"],boundary=True,required_write_bytes=int(planned))
    m=g.budget.v4.measure_scopes(ROOT,[local(OUTPUT)])
    a=g.budget.v4.measure_scopes(ROOT,[local(spec["worker_auxiliary_scope"])])
    need(m["conservative_scope_bytes"]+(planned if target=="output" else 0)<=CAPS["output_bytes"]
         and a["conservative_scope_bytes"]+(planned if target=="aux" else 0)<=CAPS["auxiliary_bytes"],"Separate original output/AUX caps")
    current(spec)
    return state
def sample_scope(spec):
    """Descriptor-relative no-follow periodic census; not an atomic quota."""
    totals=[]
    for name,limit in ((OUTPUT,CAPS["output_bytes"]),(spec["worker_auxiliary_scope"],CAPS["auxiliary_bytes"])):
        total=0;count=0
        try:rootfd=os.open(local(name),os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        except FileNotFoundError:totals.append(0);continue
        def visit(fd,depth):
            nonlocal total,count
            need(depth<=24,"Bounded owned scope depth")
            st=os.fstat(fd);total+=max(st.st_size,st.st_blocks*512)
            with os.scandir(fd) as entries:names=[e.name for e in entries]
            need(len(names)<=16384,"Bounded directory entries")
            for child in names:
                count+=1;need(count<=16384,"Bounded owned namespace")
                try:st=os.stat(child,dir_fd=fd,follow_symlinks=False)
                except FileNotFoundError:continue
                if stat.S_ISDIR(st.st_mode):
                    try:nextfd=os.open(child,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
                    except FileNotFoundError:continue
                    try:visit(nextfd,depth+1)
                    finally:os.close(nextfd)
                else:
                    need(stat.S_ISREG(st.st_mode) and st.st_nlink==1,"No links or special owned nodes")
                    total+=max(st.st_size,st.st_blocks*512)
                need(total<=limit,"Owned scope polling cap")
        try:visit(rootfd,0)
        finally:os.close(rootfd)
        totals.append(total)
    need(sum(totals)<=CAPS["reservation_bytes"],"Whole owned polling cap")
    return totals
def input_spec(r):
    s=read(r);current(s);q,cfg=request(s["root_request_ref"]);chosen=selection(q["policy_selector_ref"])
    need(r["path"]==q["packet_directory"]+"/input.json" and s["schema"]=="expanded_full_panel_v1_root_admission"
         and s["parent_id"]==PARENT and s["allowed_stages"]==["all"] and s["output_directory"]==OUTPUT
         and s["outer_evaluation_authorized"] is False and s["source_ref"]==PRODUCER
         and s["config_ref"]==CONFIG and s["config_content_sha256"]==digest(cfg)
         and s["parent_registry_sha256"]==digest(cfg["parent_registry"])
         and s["storage_adapter_ref"]==chosen["source_ref"] and s["worker_auxiliary_scope"]==q["packet_directory"]+"/worker_aux"
         and s["created_at_utc"]==q["created_at_utc"] and s["expires_at_utc"]==q["expires_at_utc"]
         and s["storage"]["completed_overlay_policy_ref"]==chosen["policy_ref"]
         and s["policy_selector_ref"]==q["policy_selector_ref"]
         and s["registration_ref"]==q["registration_ref"]
         and s["storage"]["reserved_new_bytes"]==CAPS["reservation_bytes"],"Exact unchanged producer admission/storage")
    return s,q,cfg
def admit(r):
    need(Path.cwd()==ROOT and sys.dont_write_bytecode,"ROOT cwd/no bytecode")
    q,cfg=request(r);packet=local(q["packet_directory"])
    need(not packet.exists() and not local(OUTPUT).exists(),"Fresh first-parent scopes; no resume")
    chosen=selection(q["policy_selector_ref"]);g=guard(q["policy_selector_ref"],False);packet.mkdir()
    rt=q["runtime_ref"]
    impl,launcher=ref(g.budget.SOURCE_REL),ref(g.budget.LAUNCHER_REL)
    with g.budget.v4.mutex(ROOT):state=g._state_locked(impl["sha256"],launcher["sha256"],fresh=True)
    consumers=g.consumer_bindings_v1(list(state[3].values()),generation_direct_refs_v1(q,r))
    auxiliary=q["packet_directory"]+"/worker_aux"
    own=dict(key="European_outer8/"+PARENT,output=str(local(OUTPUT)),auxiliary_scopes=[str(local(auxiliary))],
        reserved_bytes=CAPS["reservation_bytes"],consumer_bindings=consumers)
    auth=g.operation_draft_v1("reserve",own,expected_before=g._pins(state[2]),bindings=state[3])
    auth.update(active=True,expires_at_utc=q["expires_at_utc"])
    ar=put(packet/"reserve_authority.json",auth)
    receipt=g.apply_reserve_v1(ar,expected_source_sha256=impl["sha256"],expected_launcher_sha256=launcher["sha256"])
    rr=put(packet/"reserve_receipt.json",receipt)
    storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,accounting_mode=g.MODE,
        job_id=JOB,output_directory=OUTPUT,reserved_new_bytes=CAPS["reservation_bytes"],reservation=receipt["value"],
        initial_ledger_pins=receipt["initial_ledger_pins"],created_at_utc=q["created_at_utc"],expires_at_utc=q["expires_at_utc"],
        completed_overlay_policy_ref=chosen["policy_ref"],budget_components=dict(implementation=impl,launcher=launcher,consumer_adapter=chosen["source_ref"]))
    sr=put(packet/"storage.json",storage)
    spec=dict(schema="expanded_full_panel_v1_root_admission",root_authorizer="root_agent",execution_authorized=True,status="active",
        parent_id=PARENT,allowed_stages=["all"],job_id=JOB,output_directory=OUTPUT,source_ref=PRODUCER,config_ref=CONFIG,
        config_content_sha256=digest(cfg),parent_registry_sha256=digest(cfg["parent_registry"]),runtime_ref=rt,
        created_at_utc=q["created_at_utc"],expires_at_utc=q["expires_at_utc"],storage=storage,storage_adapter_ref=chosen["source_ref"],
        outer_evaluation_authorized=False,worker_auxiliary_scope=auxiliary,root_request_ref=r,
        registration_ref=q["registration_ref"],policy_selector_ref=q["policy_selector_ref"],no_scientific_modifications_to_original_generation=True)
    ir=put(packet/"input.json",spec)
    put(packet/"admitted.json",dict(input_ref=ir,storage_ref=sr,reserve_receipt_ref=rr,consumer_count=len(consumers),execution=False))
    return dict(input_ref=ir,admission_only=True)
def worker(r,gate_fd):
    need(os.read(gate_fd,1)==b"G","Owned launch gate")
    os.close(gate_fd);s,q,cfg=input_spec(r)
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="" and os.getpid()==os.getpgrp()==os.getsid(0),"Owned CPU-only session")
    os.sched_setaffinity(0,set(q["cpu_ids"]))
    for name,expected in cfg["runtime"].items():
        if name=="python":need(sys.version_info[:2]==(3,11),"Original Python3.11")
        else:need(importlib.import_module(name).__version__==expected,"Original runtime "+name)
    g=guard(q["policy_selector_ref"],True)
    g.validate_storage_v9(s,dict(output_cap_bytes=CAPS["reservation_bytes"]),s["storage"],
        source_ref=ref(SOURCE),config_ref=CONFIG,runtime_ref=s["runtime_ref"])
    raw(PRODUCER);raw(ORIGINAL)
    producer=importlib.import_module("arggnn.prepare_expanded_full_panel_v1")
    need(Path(producer.__file__).resolve()==local(PRODUCER["path"]),"Original producer origin")
    result=producer.execute_parent_v1(cfg,PARENT,"all",s,guard=lambda n=0:boundary(g,s,int(n)))
    need(result["parent_id"]==PARENT and result["Relate_completed"] is True and result["outer_truth_evaluated"] is False,"Complete generation/native inference only")
    for directory,expected in (("observed",OBSERVED),("truth",PRIVATE)):
        names=sorted(p.name for p in (local(OUTPUT)/directory).iterdir())
        need(names==sorted(expected),"Exact original output/private-label whitelist")
    boundary(g,s);print(json.dumps(dict(status="prepared_original_outer000",outer_truth_scored=False)),flush=True);return 0
def execute(r):
    started=time.monotonic();s,q,cfg=input_spec(r);g=guard(q["policy_selector_ref"],True)
    need(Path.cwd()==ROOT and Path(sys.executable).resolve()==(ROOT/".runtime/env/bin/python").resolve(),"Original CPU supervisor runtime")
    need(not local(OUTPUT).exists() and not local(s["worker_auxiliary_scope"]).exists(),"Fresh one-shot execution only")
    g.validate_storage_v9(s,dict(output_cap_bytes=CAPS["reservation_bytes"]),s["storage"],source_ref=ref(SOURCE),config_ref=CONFIG,runtime_ref=s["runtime_ref"])
    aux=local(s["worker_auxiliary_scope"]);g.budget_check_v9(s["storage"],boundary=True,required_write_bytes=32768)
    aux.mkdir();(aux/"tmp").mkdir()
    put(aux/"attempt_consumed.json",dict(input_ref=r,consumed_utc=datetime.now(timezone.utc).isoformat(),one_shot=True))
    load(PIDFD,"pidfd_backend_v1");owned=load(OWNED,"_European_outer_owned")
    sys.modules["pidfd_backend_v1"].self_probe_v1()
    need(not owned._children(os.getpid()),"No unrelated children")
    need(ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)==0,"Owned subreaper")
    gate_read,gate_write=os.pipe();proc=tree=seed=None;status=None;failure=None;tail=bytearray();eof=False;nonblocking=False;cancel=[None]
    old={sig:signal.signal(sig,lambda n,f:cancel.__setitem__(0,n)) for sig in (signal.SIGINT,signal.SIGTERM)}
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",
        PYTHONPATH=os.pathsep.join([str(ROOT),str(ROOT/"src"),str(ROOT/"scripts")]),
        TMPDIR=str(aux/"tmp"),TMP=str(aux/"tmp"),TEMP=str(aux/"tmp"),OMP_NUM_THREADS="8",MKL_NUM_THREADS="8")
    deadline=min(started+CAPS["wall_seconds"]-300,time.monotonic()+datetime.fromisoformat(s["expires_at_utc"]).timestamp()-time.time()-300)
    try:
        proc=subprocess.Popen([str(ROOT/".runtime/env/bin/python"),"-B","-m",HERE.replace("/",".")+".prepare_outer000_v1",
            "--worker-ref",json.dumps(r),"--gate-fd",str(gate_read)],cwd=ROOT,env=env,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(gate_read,),bufsize=0)
        os.close(gate_read);gate_read=None;os.set_blocking(proc.stdout.fileno(),False);nonblocking=True
        seed=owned._stat(proc.pid);tree=owned.OwnedTree(proc.pid)
        put(aux/"launch.json",dict(pid=proc.pid,input_ref=r,ownership=tree.evidence(),gate_closed=True))
        need(time.monotonic()<deadline and cancel[0] is None,"Work window before gate")
        os.write(gate_write,b"G");os.close(gate_write);gate_write=None
        while time.monotonic()<deadline and cancel[0] is None:
            event=os.waitid(os.P_PID,proc.pid,os.WEXITED|os.WNOHANG|os.WNOWAIT)
            tree.refresh();need(len(tree.known)<=64,"Bounded own process roster")
            rss=0
            for row in tree.known.values():
                if owned._same(owned._stat(row["pid"]),row):
                    try:rss+=int(Path("/proc/%d/statm"%row["pid"]).read_text().split()[1])*os.sysconf("SC_PAGE_SIZE")
                    except (FileNotFoundError,ProcessLookupError):pass
            need(rss<=CAPS["rss_bytes"],"Owned RSS fail-stop")
            current(s,300);sample_scope(s)
            try:data=os.read(proc.stdout.fileno(),65536)
            except BlockingIOError:data=None
            if data==b"":eof=True
            elif data:tail.extend(data);del tail[:-CAPS["log_bytes"]]
            if event is not None:break
            time.sleep(.2)
        else:failure="Cancellation or scientific work cutoff"
    except BaseException as exc:failure=type(exc).__name__+": "+str(exc)[:2000]
    finally:
        for fd in (gate_read,gate_write):
            if fd is not None:
                try:os.close(fd)
                except OSError:pass
        stop=time.monotonic();end=stop+CAPS["cleanup_seconds"];absence=False;groups=[];adopted=[];errors=[]
        while proc is not None and time.monotonic()<end:
            try:
                if status is None:
                    pid,actual=os.waitpid(proc.pid,os.WNOHANG)
                    if pid==proc.pid:status=actual;proc.returncode=os.waitstatus_to_exitcode(actual)
                if tree is not None:tree.refresh();need(len(tree.known)<=64,"Bounded cleanup roster")
                rows=list(tree.known.values())[:64] if tree is not None else ([seed] if seed else [])
                for row in rows:
                    if time.monotonic()>=end:break
                    owned.signal_identity(row,signal.SIGTERM if time.monotonic()-stop<10 else signal.SIGKILL)
                if status is not None:
                    for _ in range(64):
                        if time.monotonic()>=end:break
                        try:pid,actual=os.waitpid(-1,os.WNOHANG)
                        except ChildProcessError:break
                        if not pid:break
                        if len(adopted)<64:adopted.append(dict(pid=pid,status=actual,returncode=os.waitstatus_to_exitcode(actual)))
                        else:failure=failure or "Adopted roster overflow"
                for _ in range(4):
                    if time.monotonic()>=end or not nonblocking:break
                    try:data=os.read(proc.stdout.fileno(),65536)
                    except BlockingIOError:break
                    if not data:eof=True;break
                    tail.extend(data);del tail[:-CAPS["log_bytes"]]
                if status is not None and tree is None:break
                if status is not None and tree is not None and time.monotonic()<end:
                    absence=tree.identities_absent() and not owned._children(os.getpid())
                    if absence and eof:
                        groups=[]
                        for pgid in sorted({x["group"] for x in tree.known.values()}):
                            gone=None
                            if time.monotonic()<end:
                                try:os.killpg(pgid,0)
                                except ProcessLookupError:gone=True
                                else:gone=False
                            groups.append(dict(pgid=pgid,absent=gone))
                        if groups and all(x["absent"] is True for x in groups):break
            except BaseException as exc:
                failure=failure or "Cleanup observation failed"
                if len(errors)<16:errors.append(type(exc).__name__+": "+str(exc)[:500])
            time.sleep(min(.05,max(0,end-time.monotonic())))
        if proc is not None and proc.stdout is not None:proc.stdout.close()
        for sig,handler in old.items():signal.signal(sig,handler)
    rc=None if status is None else os.waitstatus_to_exitcode(status)
    drained=status is not None and absence and eof and bool(groups) and all(x["absent"] is True for x in groups)
    if not drained:failure=failure or "Unresolved owned drain/wait"
    if rc!=0 or any(x["returncode"]!=0 for x in adopted):failure=failure or "Genuine nonzero/unknown child wait"
    observation=dict(schema="European_outer000_actual_wait_v1",input_ref=r,pid=None if proc is None else proc.pid,
        actual_direct_worker_wait=status is not None,actual_OS_wait_status=status,returncode=rc,
        all_owned_processes_and_groups_drained=drained,group_observations=groups,adopted_waits=adopted,
        operational_failure=failure,cleanup_errors=errors,outer_truth_scored=False,automatic_close_or_credit=False)
    print(json.dumps(observation,sort_keys=True),flush=True)
    if not drained or time.time()+150>=datetime.fromisoformat(s["expires_at_utc"]).timestamp():
        print(json.dumps(dict(retained_worker_tail=bytes(tail[-8192:]).decode(errors="replace"),no_future_stage_authorized=True)),flush=True)
        return dict(status="FAILED_UNRESOLVED_OR_EVIDENCE_MARGIN",returncode=1,actual_wait_ref=None,observation=observation)
    try:
        boundary(g,s,32768,target="aux");wr=put(aux/"actual_wait.json",observation)
        boundary(g,s,len(tail)*6+8192,target="aux");put(aux/"worker_log.json",dict(tail=bytes(tail).decode(errors="replace"),retained_raw_bytes=len(tail)),cap=16*2**20)
        cp=ref(OUTPUT+"/expanded_parent_completed.json")
        completed=read(cp)
        need(completed["parent_id"]==PARENT and completed["Relate_completed"] is True and completed["outer_truth_evaluated"] is False,"Actual complete original pipeline")
        return dict(status="PREPARED" if failure is None else "FAILED",returncode=0 if failure is None else 1,
            actual_wait_ref=wr,completed_ref=cp,outer_tool_terminal_consumed=False,outer_truth_scored=False)
    except BaseException as exc:
        print(json.dumps(dict(actual_wait_observation=observation,evidence_failure=str(exc)[:2000])),flush=True)
        return dict(status="FAILED_EVIDENCE",returncode=1)
def main():
    p=argparse.ArgumentParser(description=__doc__);a=p.add_mutually_exclusive_group()
    a.add_argument("--admit-ref");a.add_argument("--execute-ref");a.add_argument("--worker-ref")
    p.add_argument("--gate-fd",type=int);args=p.parse_args()
    if not any((args.admit_ref,args.execute_ref,args.worker_ref)):
        print(json.dumps(dict(status="SOURCE_ONLY_INACTIVE",admission=False,execution=False)));return 0
    sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"scripts")]
    need(sys.platform.startswith("linux") and sys.dont_write_bytecode,"Fresh Linux/no-bytecode ROOT entry")
    if args.worker_ref:return worker(json.loads(args.worker_ref),args.gate_fd)
    result=admit(json.loads(args.admit_ref)) if args.admit_ref else execute(json.loads(args.execute_ref))
    print(json.dumps(result,sort_keys=True),flush=True);return result.get("returncode",0)
if __name__=="__main__":raise SystemExit(main())
