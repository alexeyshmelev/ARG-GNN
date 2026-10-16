"""H056-only existing monitored CPU IO integration, source only on import.
Root separately admits actual counts/data, then chooses preparation or frozen
application. No admission, optimizer, tuning, recovery or generic dispatch.
"""
from pathlib import Path
from dataclasses import fields
import argparse,datetime,hashlib,importlib,importlib.util,io,json,os,resource,selectors,signal,subprocess,sys,time

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/H056_bank_application_runtime_source_20261016_v1"
SOURCE=HERE+"/runtime_v2.py"
IMPLEMENTATION="research/reviews/H056_H36_implementation_20261016_v1"
PROTOCOL_REF={'path': 'research/reviews/H056_frozen_H045_Danish_application_registration_20261016_v1/protocol.root_frozen_v1.json', 'sha256': '9544bdbd499bddafeb0774c67e4e86ef25535d41dd2409e710c29b679619cd6e'}
CHECKPOINT_REF={'path': 'experiments/results/H045_GPU5_episode_train_seed101_v1/foreign_episode.pass_30.pt', 'sha256': 'bfd8a87632015a900d9da6009d4b0ee037958460038d7949d0998e80b770796b'}
ROSTER_REF={'path': 'research/reviews/Danish_genotype_corroboration_source_20261016_v2/candidate_roster.metadata.v1.tsv', 'sha256': 'eefcb18bb4c59d959b26d1e6b8a4a62c4eb0c9bbcfa89a7602797716e6d3cf46'}
CHECKPOINT_DESCRIPTOR_REF={'path': 'research/reviews/H018_H045_empirical_application_source_v1/checkpoint_descriptor.json', 'sha256': 'fd8bbd7e61af5d93316e3f4c1d31275d1ee991f6f734991f0091ec069e5a1f12'}
FIXED_REFS={'guard': {'path': 'research/reviews/prospective_980GB_capacity_source_v1/prospective_980gb_v1.py', 'sha256': 'fc71030865e0a3d5da31120d4daa2caa0b2d39ec7dd212b8c46c476b5a866d56'}, 'policy': {'path': 'research/reviews/prospective_980GB_capacity_source_v1/activation_policy.root_active_v1.json', 'sha256': 'eb3408004a508136aa434c0a2e02a9d14a5598a96b8483c264c93026f9326018'}, 'repair_authority': {'path': 'research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json', 'sha256': 'f6f2d628199e8be73db8b5d90d428570d8afdefeab9dc5cd94bb5eafb39247db'}, 'renewal': {'path': 'research/reviews/H039_marker_repair_renewal_source_20261016_v1/marker_repair_renewal_v1.py', 'sha256': '491236877e8d71cae8bd7619880858ac501c57dd93e59479ca32dacacab20468'}, 'renewal_certificate': {'path': 'research/reviews/H039_marker_repair_renewal_source_20261016_v1/renewal_certificate.root_active_v1.json', 'sha256': '66006202094d7a3ef2b29cbec28eb93f334505b72589f09d86e4b57498162db7'}, 'memo': {'path': 'research/reviews/resource_guard_hotpath_source_audit_v1/transaction_memo_adapter_v2.py', 'sha256': 'c972e9a779d424ceafdb258848614bf40994d1004e2c73fc5cd225ae99621364'}, 'v4': {'path': 'src/arggnn/project_budget_v4.py', 'sha256': 'a093c7d86f57fee569cb99f4712250e9505a1119f3950fa44282c7ba8c57df47'}, 'monitor': {'path': 'research/reviews/FLARE_existing_JDK_source_v1/root_build_driver.py', 'sha256': 'd7afaf5c6c6976a5c4804fe41d320f8609fd346d3f4ed0bca29235001d7fe561'}, 'contract': {'path': 'research/reviews/H056_H36_implementation_20261016_v1/contract_v1.py', 'sha256': '4c7727e27be2dddeb698dc78ff6d7ca143d85b47187a1fdd23dc7c17217d8662'}}
MONITOR_REF=FIXED_REFS["monitor"]
GUARD_REF=FIXED_REFS["guard"]
POLICY_REF=FIXED_REFS["policy"]
REPAIR_REF=FIXED_REFS["repair_authority"]
RSS,DEADLINE=2_000_000_000,1800
AUX,LOG_CAP=524288,262144
ENV=dict(CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",OMP_NUM_THREADS="4",
 OPENBLAS_NUM_THREADS="4",MKL_NUM_THREADS="4",NUMEXPR_NUM_THREADS="4")
SCIENCE_REF=CFG_REF=None
_GUARD=None

def need(ok,message):
    if not ok:raise ValueError(message)

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def local(value):
    p=Path(value);p=p if p.is_absolute() else ROOT/p
    need(p.resolve()==p and p.is_relative_to(ROOT) and p!=ROOT,"Canonical project-only path")
    return p

def ref(value):
    p=local(value);return dict(path=p.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def immutable_bytes(record,cap):
    need(isinstance(record,dict) and set(record)=={"path","sha256"},"Exact path/SHA reference")
    p=local(record["path"]);before=p.lstat()
    signature=lambda st:(st.st_dev,st.st_ino,st.st_mode,st.st_nlink,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
    need(p.is_file() and not p.is_symlink() and before.st_nlink==1 and before.st_size<=cap,"Bounded singly linked regular original file")
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,"rb") as stream:
        need(signature(os.fstat(stream.fileno()))==signature(before),"Original read identity")
        raw=stream.read(cap+1);after=os.fstat(stream.fileno())
    need(len(raw)<=cap and signature(before)==signature(after)==signature(p.lstat())
        and hashlib.sha256(raw).hexdigest()==record["sha256"],"Exact immutable read")
    return raw

def put(path,value,cap=65536):
    raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    need(len(raw)<=cap,"Compact precharged root metadata")
    fd=os.open(local(path),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,"wb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return ref(path)

def unique(records):
    values={}
    for r in records:
        need(set(r)=={"path","sha256"} and Path(r["path"]).suffix in (".py",".sh",".json"),"Only directly admissible metadata/source refs")
        need(r["path"] not in values or values[r["path"]]==r,"No conflicting path hash")
        values[r["path"]]=r
    result=list(values.values());need(len(result)<=64,"At most64 complete actual consumer bindings");return result

def expiry(value):
    stamp=datetime.datetime.fromisoformat(value.replace("Z","+00:00"))
    need(stamp.tzinfo is not None and stamp>datetime.datetime.now(datetime.timezone.utc),"Unexpired root authority")

def cpu():
    need(Path.cwd().resolve()==ROOT and os.environ.get("CUDA_VISIBLE_DEVICES")=="","Project cwd and explicit empty CUDA visibility")

def monitor_helpers():
    need(ref(MONITOR_REF["path"])==MONITOR_REF,"Frozen reused owned-process monitor helpers")
    return importlib.import_module("research.reviews.FLARE_existing_JDK_source_v1.root_build_driver")

def metadata(record,cap=32*2**20):
    need(Path(record["path"]).suffix==".json","Exact JSON metadata")
    return json.loads(immutable_bytes(record,cap))

def source_load(record,name):
    code=immutable_bytes(record,2**20)
    spec=importlib.util.spec_from_file_location(name,local(record["path"]))
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod
    exec(compile(code,str(local(record["path"])),"exec"),mod.__dict__)
    return mod

def ceil_mib(n):return ((n+2**20-1)//2**20)*2**20

def valid_counts(counts):
    need(set(counts)=={"candidate_sites","sites","constants","native_trees","native_nodes","native_edges","vertex_occurrences"}
      and all(type(v)is int for v in counts.values()) and counts["candidate_sites"]>=counts["sites"]>=1000
      and 0<=counts["constants"]<=counts["sites"] and 0<counts["native_trees"]<=counts["sites"]
      and counts["native_nodes"]>=36 and counts["native_edges"]>0
      and counts["vertex_occurrences"]>=36*counts["native_trees"],"Actual seven native counts, not historical or forecast")
    return counts

def preparation_caps_v1(counts):
    """Inactive until actual counts exist. Abort ceilings are not measured fits."""
    c=valid_counts(counts);V,S,T=c["vertex_occurrences"],c["sites"],c["native_trees"]
    tensor=256*V+640*S+65536
    flags=2*V
    meta=2**20+1024*T
    bank=ceil_mib(tensor+flags+meta+2*2**20)
    complete=ceil_mib(meta+2**20)
    science=bank+complete
    total=ceil_mib(science+2**20+AUX+2**20)
    roles=dict(original_VCF=24*2**20,bridge_haps=16*2**20,bridge_sample=65536,bridge_ages=65536,
      frozen_sites=16*2**20,roster=65536,serialized_readback=2**20,graph_contract=2**20,native_mapping=2**20,
      native_graph=ceil_mib(8*2**20+8192*S+128*(c["native_nodes"]+c["native_edges"])))
    return dict(bank_serialization_cap_bytes=bank,completion_serialization_cap_bytes=complete,
      scientific_output_cap_bytes=science,output_cap_bytes=total,root_worker_receipt_cap_bytes=2**20,
      native_facts_read_cap_bytes=ceil_mib(2**20+3072*S),observation_role_caps=roles,
      input_read_cap_bytes=sum(roles.values()),VCF_decompressed_cap_bytes=200_000_000,
      CPU_tensor_cap_bytes=tensor,flag_cap_bytes=flags,CPU_preparation_cap_bytes=tensor+2058*V+5120)

def application_caps_v1(chromosomes):
    need(set(chromosomes)=={"20","21","22"},"All three actual chromosomes")
    cs={k:valid_counts(v) for k,v in chromosomes.items()}
    probability=sum(ceil_mib(384*c["native_trees"]+65536) for c in cs.values())
    provenance=sum(2**20+512*c["native_trees"] for c in cs.values())+2**20
    payload=ceil_mib(probability+provenance)
    total=ceil_mib(payload+2**20+AUX+2**20)
    return dict(max_serialized_artifact_payload_bytes=payload,output_cap_bytes=total,
      bank_read_cap_bytes=max(preparation_caps_v1(c)["bank_serialization_cap_bytes"] for c in cs.values()),
      checkpoint_read_cap_bytes=1048576,episode_cap_bytes=max(18*c["vertex_occurrences"]+4*c["native_trees"] for c in cs.values()),
      root_worker_receipt_cap_bytes=2**20)

def configured_guard():
    """Original guard + reviewed expiry renewal/memo2, never new accounting."""
    global _GUARD
    if _GUARD is not None:return _GUARD
    for r in FIXED_REFS.values():immutable_bytes(r,2**20)
    renewal=source_load(FIXED_REFS["renewal"],"_H056_bankapp_repair_renewal")
    renewal.install_renewal_v1(FIXED_REFS["renewal_certificate"])
    m=source_load(GUARD_REF,"_H056_bankapp_original980")
    m.configure_v2(REPAIR_REF,policy_ref=POLICY_REF)
    memo=source_load(FIXED_REFS["memo"],"_H056_bankapp_transaction_memo2")
    memo.install(m.budget.v4,expected_source_path=local(FIXED_REFS["v4"]["path"]),
      expected_source_sha256=FIXED_REFS["v4"]["sha256"])
    need(m.MODE=="prospective_980gb_v1","Unchanged original active980guard")
    _GUARD=m;return m

def runtime_check(runtime,spec):
    import importlib.metadata
    need(runtime["prefix"]==str(ROOT/".runtime/env") and runtime["CUDA_VISIBLE_DEVICES"]==""
      and runtime["executable"]==str(Path(sys.executable).resolve()) and runtime["python"]==sys.version
      and runtime["CPU_threads"]==4 and runtime["packages"]
      and all(isinstance(v,str) and v and importlib.metadata.version(k)==v for k,v in runtime["packages"].items())
      and runtime["expires_at_utc"]==spec["expires_at_utc"],"Actual existing non-null CPU package/runtime subset")
    need(set(runtime["packages"])=={"numpy","torch","torch_geometric","tskit"},"Exact required CPU package subset")
    expiry(spec["expires_at_utc"])
    issued=datetime.datetime.fromisoformat(runtime["created_at_utc"].replace("Z","+00:00"))
    end=datetime.datetime.fromisoformat(spec["expires_at_utc"].replace("Z","+00:00"))
    renewed=datetime.datetime.fromisoformat(metadata(FIXED_REFS["renewal_certificate"])["renewed_expires_at_utc"].replace("Z","+00:00"))
    need(issued.tzinfo is not None and issued<=datetime.datetime.now(datetime.timezone.utc)
      and 0<(end-issued).total_seconds()<=DEADLINE and end<renewed,
      "Original at-most1800s absolute authority fits authentic renewal, never reset")


_FINAL_TIMER_V2=None

def arm_final_timer_v2(started,spec):
    """Retain a deadline flag through final receipts without interrupting waits."""
    global _FINAL_TIMER_V2
    need(_FINAL_TIMER_V2 is None and signal.getitimer(signal.ITIMER_REAL)==(0.,0.),
      "Fresh original process deadline timer")
    expiry(spec["expires_at_utc"])
    absolute=datetime.datetime.fromisoformat(spec["expires_at_utc"].replace("Z","+00:00")).timestamp()
    remaining=min(DEADLINE-(time.monotonic()-started),absolute-time.time())
    need(remaining>0,"Original elapsed and absolute authority remain current")
    state=dict(started=started,absolute=absolute,expired=False,previous=signal.getsignal(signal.SIGALRM))
    def mark_expired(signum,frame):
        # Polling termination and final checks act on this flag. Never break wait/receipt cleanup.
        state["expired"]=True
    signal.signal(signal.SIGALRM,mark_expired)
    _FINAL_TIMER_V2=state
    signal.setitimer(signal.ITIMER_REAL,remaining)
    check_final_deadline_v2()

def check_final_deadline_v2():
    need(_FINAL_TIMER_V2 is not None,"Original process deadline remains installed")
    state=_FINAL_TIMER_V2
    need(not state["expired"] and time.monotonic()-state["started"]<DEADLINE
      and time.time()<state["absolute"],"Original deadline includes final writes and guard")
    need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=RSS,"Final own process peakRSS remains in original envelope")
    cpu()

def restore_final_timer_v2():
    global _FINAL_TIMER_V2
    if _FINAL_TIMER_V2 is not None:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,_FINAL_TIMER_V2["previous"])
        _FINAL_TIMER_V2=None

def complete_source_map(scientific,io):
    need(isinstance(scientific["source_sha256"],dict) and isinstance(io["source_sha256"],dict),"Explicit full source maps")
    for p,h in scientific["source_sha256"].items():
        need(io["source_sha256"].get(p)==h,"All frozen scientific sources remain directly bound")
    for p,h in io["source_sha256"].items():
        need(Path(p).suffix==".py" and ref(p)==dict(path=p,sha256=h),"Actual direct source authentication")
    contract_raw=immutable_bytes(FIXED_REFS["contract"],2**20)
    # Literal source metadata only, enforced again by the scientific consumer.
    need(hashlib.sha256(contract_raw).hexdigest()==FIXED_REFS["contract"]["sha256"],"Pinned fixed H36 contract")

def load_input(args):
    global SCIENCE_REF,CFG_REF
    cpu();need(args.input and args.expected_sha256,"Actual root input path/SHA")
    record=dict(path=local(args.input).relative_to(ROOT).as_posix(),sha256=args.expected_sha256)
    spec=metadata(record);cfg=metadata(spec["config_ref"]);storage=metadata(spec["storage_ref"]);expiry(spec["expires_at_utc"])
    mode=spec["operation"]
    need(mode in ("prepare","apply") and spec["root_authorizer"]=="root_agent"
      and spec["hypothesis_id"]=="H056" and spec["registered"] is True and spec["execution_authorized"] is True
      and spec["protocol_ref"]==PROTOCOL_REF and spec["GPU_authorized"] is False
      and spec["optimization_authorized"] is False and spec["root_host_resource_enforcement_active"] is True
      and spec["io_context_ref"]==ref(SOURCE) and spec["storage_adapter_ref"]==GUARD_REF,
      "One of exactly two explicit H056 CPU scientific operations")
    expected_schema="H056_H36_CPU_preparation_root_v1" if mode=="prepare" else "H056_H36_CPU_application_root_v1"
    producer_path=IMPLEMENTATION+("/reader_h36_v1.py" if mode=="prepare" else "/apply_h36_v1.py")
    need(spec["schema"]==expected_schema and spec["producer_ref"]==ref(producer_path),"Exact H056 producer/root schema")
    need(cfg["protocol_ref"]==PROTOCOL_REF and cfg["CPU_threads"]==4 and cfg["max_host_RSS_bytes"]==RSS
      and cfg["wall_deadline_seconds"]==DEADLINE and cfg["output_cap_bytes"]==storage["reserved_new_bytes"],
      "Original2GB/1800s/CPU4 envelope and count-derived charged cap")
    if mode=="prepare":
        need(cfg["schema"]=="H056_H36_reader_config_v1" and cfg["status"]=="root_admitted_observation_preparation"
          and cfg["preparation_authorized"] is True and spec["preparation_authorized"] is True
          and spec["model_or_checkpoint_or_GPU_authorized"] is False,"Observation-only preparation authority")
        caps=preparation_caps_v1(cfg["expected_counts"])
        need(all(cfg[k]==caps[k] for k in ("output_cap_bytes","input_read_cap_bytes","VCF_decompressed_cap_bytes",
          "CPU_tensor_cap_bytes","flag_cap_bytes","CPU_preparation_cap_bytes"))
          and spec["caps"]==caps,"Exact actual-count preparation/serialization caps")
        need(set(cfg["observation_refs"])==set(caps["observation_role_caps"]),"Exact originalGT/native-graph observation roles")
    else:
        need(cfg["schema"]=="H056_H36_CPU_application_config_v1" and cfg["execution_enabled"] is True
          and cfg["optimization_steps"]==0 and cfg["checkpoint_ref"]==CHECKPOINT_REF
          and cfg["checkpoint_descriptor_ref"]==CHECKPOINT_DESCRIPTOR_REF and set(cfg["chromosomes"])=={"20","21","22"},
          "Fixed H045 model, no optimization, all three new banks")
        caps=application_caps_v1({k:v["counts"] for k,v in cfg["chromosomes"].items()})
        need(all(cfg[k]==caps[k] for k in ("output_cap_bytes","max_serialized_artifact_payload_bytes",
          "bank_read_cap_bytes","checkpoint_read_cap_bytes","episode_cap_bytes")) and spec["caps"]==caps,
          "Exact actual-count application/output caps")
        need(len({json.dumps(v["roster_ref"],sort_keys=True) for v in cfg["chromosomes"].values()})==1,"One shared exact36-row roster")
    need(spec["source_manifest_ref"]==cfg["source_manifest_ref"],"Exact scientific manifest")
    scientific=metadata(cfg["source_manifest_ref"]);io_manifest=metadata(spec["io_source_manifest_ref"])
    need(io_manifest["schema"]=="H056_specific_CPU_IO_closure_v1" and io_manifest["operation"]==mode
      and io_manifest["protocol_ref"]==PROTOCOL_REF and scientific["protocol_ref"]==PROTOCOL_REF
      and io_manifest["scientific_manifest_ref"]==cfg["source_manifest_ref"],"Specific complete IO/scientific closure")
    complete_source_map(scientific,io_manifest)
    required=unique([spec["io_context_ref"],spec["producer_ref"],spec["config_ref"],cfg["source_manifest_ref"],spec["runtime_ref"],
      spec["io_source_manifest_ref"],PROTOCOL_REF,*spec["resource_binding_refs"],
      *[dict(path=p,sha256=h) for p,h in io_manifest["source_sha256"].items()],*io_manifest["required_metadata_refs"]])
    required_stage_metadata=[PROTOCOL_REF]
    if mode=="prepare":
        required_stage_metadata.extend([spec["native_completion_ref"],spec["native_wait_ref"],
          cfg["observation_refs"]["roster"],cfg["observation_refs"]["serialized_readback"],
          cfg["observation_refs"]["graph_contract"],cfg["observation_refs"]["native_mapping"]])
    else:
        required_stage_metadata.extend([CHECKPOINT_DESCRIPTOR_REF,
          *scientific["required_metadata_refs"],
          *[x[k] for x in cfg["chromosomes"].values() for k in ("bank_completion_ref","preparation_wait_ref","roster_ref")]])
    need(all(r in required for r in required_stage_metadata),"All actual stage completion/wait/roster/graph metadata direct")
    own=storage["reservation"]
    need(storage["kind"]=="project_budget_v9_storage_record" and storage["active"] is True and storage["consumer_uses_v9_guard"] is True
      and storage["accounting_mode"]=="prospective_980gb_v1" and storage["completed_overlay_policy_ref"]==POLICY_REF
      and storage["output_directory"]==spec["output_directory"] and storage["job_id"]==spec["job_id"]
      and storage["expires_at_utc"]==spec["expires_at_utc"] and own["key"].startswith("H056/")
      and own["output"]==str(local(spec["output_directory"])) and own["reserved_bytes"]==storage["reserved_new_bytes"]
      and own["auxiliary_scopes"]==[str(local(spec["auxiliary_directory"]))]
      and len(own["consumer_bindings"])<=64 and len(own["consumer_bindings"])==len(required)
      and all(r in own["consumer_bindings"] and ref(r["path"])==r for r in required),
      "Exact existing prospective980GB storage, source/runtime/output/expiry and full direct union<=64")
    out,aux=local(spec["output_directory"]),local(spec["auxiliary_directory"])
    need(out!=aux and not out.is_relative_to(aux) and not aux.is_relative_to(out),"Disjoint charged output/aux")
    need(not out.exists(),"No replay or overwrite of existing scientific output")
    runtime=metadata(spec["runtime_ref"]);runtime_check(runtime,spec)
    for r in FIXED_REFS.values():
        need(Path(r["path"]).suffix not in (".py",".sh",".json") or r in own["consumer_bindings"],"Actual fixed source/resource/renewal refs direct")
    SCIENCE_REF,CFG_REF=spec["producer_ref"],spec["config_ref"]
    return record,spec,cfg,storage

class Reader:
    """Original immutable reader with explicit H056-only role/cap allowlists."""
    root=ROOT
    def __init__(self,records,payloads):
        self.allowed={r["path"]:(r,32*2**20) for r in records}
        for r,cap in payloads:
            need(set(r)=={"path","sha256"} and type(cap)is int and cap>0,"Exact finite selected payload authority")
            need(r["path"] not in self.allowed or self.allowed[r["path"]][0]==r,"No conflicting payload identity")
            self.allowed[r["path"]]=(r,cap)
    def read(self,record):
        item=self.allowed.get(record.get("path"))
        need(item is not None and item[0]==record,"Explicit admitted metadata/source or exact selected payload")
        return immutable_bytes(record,item[1])
    def decode_json(self,record):
        need(Path(record["path"]).suffix==".json","Only named JSON metadata decode")
        return json.loads(self.read(record))

def worker(args):
    # The original startup barrier remains before configuration or tensor imports.
    need(args.monitor_parent_pid==os.getppid() and args.monitor_read_fd is not None,"Actual outer monitor parent")
    with os.fdopen(args.monitor_read_fd,"rb") as stream:ticket=stream.read(65)
    need(ticket==(args.expected_sha256+"\n").encode(),"Parent seeded owned PID/RSS before release")
    b=monitor_helpers()
    def stop(signum,frame):raise b.WorkerStop("Root outer monitor requested termination")
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    started=time.monotonic();record,spec,cfg,storage=load_input(args)
    arm_final_timer_v2(started,spec);m=configured_guard()
    startup=m.validate_storage_v9(spec,cfg,storage,source_ref=spec["producer_ref"],config_ref=spec["config_ref"],runtime_ref=spec["runtime_ref"])
    io_manifest=metadata(spec["io_source_manifest_ref"])
    need(startup["bindings"]==io_manifest["original_bindings"],"Original full current guard binding map")
    caps=spec["caps"];mode=spec["operation"]
    metadata_records=[*storage["reservation"]["consumer_bindings"],record,spec["storage_ref"]]
    payloads=[]
    if mode=="prepare":
        roles=caps["observation_role_caps"]
        payloads=[(r,roles[name]) for name,r in cfg["observation_refs"].items()]
        contract=metadata(cfg["observation_refs"]["graph_contract"])
        payloads.append((contract["native_facts_ref"],caps["native_facts_read_cap_bytes"]))
        need(contract["counts"]==cfg["expected_counts"] and contract["graph_only_ready"] is True
          and contract["immutable_original_GT_features"] is True,"New actual graph-only contract, no old recovery facts")
        native_done=metadata(spec["native_completion_ref"]);native_wait=metadata(spec["native_wait_ref"])
        need(native_done["schema"]=="H056_native_Relate_completed_v1" and native_done["graph_only_ready"] is True
          and native_done["counts"]==cfg["expected_counts"]
          and native_done["graph_contract_ref"]==cfg["observation_refs"]["graph_contract"]
          and native_done["native_graph_ref"]==cfg["observation_refs"]["native_graph"]
          and native_done["native_mapping_ref"]==cfg["observation_refs"]["native_mapping"]
          and native_wait["schema"]=="H056_root_actual_native_tool_wait_v1"
          and native_wait["root_authorizer"]=="root_agent" and native_wait["actual_tool_call_id"]
          and native_wait["actual_tool_terminal_consumed"] is True
          and native_wait["actual_tool_exit_code"]==0 and native_wait["actual_owned_group_absent"] is True
          and native_wait["native_completion_ref"]==spec["native_completion_ref"]
          and native_wait["native_root_input_ref"]==native_done["root_input_ref"],
          "Actual native completion plus genuine root outer wait0/drain")
    else:
        payloads=[(item["bank_ref"],cfg["bank_read_cap_bytes"]) for item in cfg["chromosomes"].values()]
        payloads.extend([(CHECKPOINT_REF,cfg["checkpoint_read_cap_bytes"]),(ROSTER_REF,65536)])
    reader=Reader(metadata_records,payloads)
    out=local(spec["output_directory"]);need(not out.exists(),"Fresh exclusive H056 output")
    m.budget_check_v9(storage,boundary=True,required_write_bytes=8192);out.mkdir()
    checks=[];used=0
    def before(stage,plan):
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        need(peak<=RSS and time.monotonic()-started<=DEADLINE,"Original worker peakRSS/deadline")
        expiry(spec["expires_at_utc"])
        if mode=="prepare":
            current=int(Path("/proc/self/statm").read_text().split()[1])*os.sysconf("SC_PAGE_SIZE")
            need(current+cfg["CPU_preparation_cap_bytes"]<=RSS,"Original RAM preparation allowance fits remaining RSS envelope")
        growth=plan.get("required_write_bytes",0)
        need(type(growth)is int and growth>=0,"Finite planned original-scope growth")
        checked=m.budget_check_v9(storage,boundary=True,required_write_bytes=growth)
        need(checked["reservation"]==storage["reservation"] and len(checks)<64,"Exact original charged reservation, bounded resource log")
        checks.append(dict(stage=stage,peak_worker_RSS_bytes=peak,elapsed_seconds=time.monotonic()-started,planned_growth_bytes=growth))
    science_cap=caps["scientific_output_cap_bytes"] if mode=="prepare" else caps["max_serialized_artifact_payload_bytes"]
    def writer(name,data,root_receipt=False):
        nonlocal used
        limit=science_cap+(caps["root_worker_receipt_cap_bytes"] if root_receipt else 0)
        need(isinstance(name,str) and Path(name).name==name and name not in (".","..")
          and isinstance(data,bytes) and used+len(data)<=limit,"Exact basename-only bounded complete H056 outputs")
        before("root_writer_boundary",dict(required_write_bytes=len(data)+8192))
        fd=os.open(out/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,"wb") as f:f.write(data);f.flush();os.fsync(f.fileno())
        used+=len(data);actual=ref(out/name)
        need(actual["sha256"]==hashlib.sha256(data).hexdigest(),"Actual exclusive fsynced output SHA")
        return actual
    ctx=dict(root_authorizer="root_agent",admitted=True,input_ref=record,source_ref=spec["producer_ref"],
      config_ref=spec["config_ref"],runtime_ref=spec["runtime_ref"],cfg=cfg,consumer_bindings=storage["reservation"]["consumer_bindings"],
      io_context_ref=spec["io_context_ref"],resource_binding_refs=spec["resource_binding_refs"],reader=reader,writer=writer,before_stage=before)
    # All source/ref/resource gates and original outer monitor precede imports.
    sys.path[:0]=[str(ROOT),str(ROOT/"src")]
    source_load(FIXED_REFS["contract"],"research.reviews.H056_H36_implementation_20261016_v1.contract_v1")
    if mode=="prepare":
        before("prepare_original_GT_native_graph_RAM",{})
        science=importlib.import_module("research.reviews.H056_H36_implementation_20261016_v1.reader_h36_v1")
        result=science.prepare_from_root_context_v1(ctx)
        before("serialize_complete_original_bank",dict(required_write_bytes=caps["bank_serialization_cap_bytes"]))
        import torch
        bank=result["bank"]
        payload=dict(schema="H056_empirical_CPU_observation_bank_v1",protocol_ref=PROTOCOL_REF,chromosome=cfg["chromosome"],
          input_ref=record,bank_tensors={f.name:getattr(bank,f.name) for f in fields(bank)},
          flag_node_flags=result["flags"].node_flags,metadata=result["metadata"])
        stream=io.BytesIO();torch.save(payload,stream)
        bank_raw=stream.getvalue();need(len(bank_raw)<=caps["bank_serialization_cap_bytes"],"Actual original bank serialization cap")
        bank_ref=writer("observation_bank.pt",bank_raw)
        del bank_raw,stream,payload
        completion=dict(schema="H056_empirical_CPU_preparation_completed_v1",status="completed_observation_preparation",
          protocol_ref=PROTOCOL_REF,chromosome=cfg["chromosome"],counts=cfg["expected_counts"],input_ref=record,
          producer_ref=spec["producer_ref"],bank_ref=bank_ref,metadata=result["metadata"],
          no_query_ancestry_truth=True,no_model_or_checkpoint_executed=True,no_GPU_used=True,
          completed_utc=utc(),root_actual_outer_wait0_and_drain_still_required=True)
        data=(json.dumps(completion,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
        need(len(data)<=caps["completion_serialization_cap_bytes"],"Bounded actual preparation completion metadata")
        done=writer("completed.json",data)
    else:
        # Calls unchanged strict checkpoint load and FP32 inference equations.
        science=importlib.import_module("research.reviews.H056_H36_implementation_20261016_v1.apply_h36_v1")
        done=science.execute_v1(ctx)
    need(done==ref(out/"completed.json"),"Actual original scientific completion returned")
    receipt=(json.dumps(dict(schema="H056_actual980_CPU_IO_worker_receipt_v1",operation=mode,scientific_completion_ref=done,
      io_context_ref=spec["io_context_ref"],actual_guard_ref=GUARD_REF,actual_policy_ref=POLICY_REF,
      resource_checks=checks,elapsed_seconds=time.monotonic()-started,optimization_steps=0),
      sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    need(len(receipt)<=caps["root_worker_receipt_cap_bytes"],"Bounded root worker receipt")
    root_receipt=writer("root_IO_worker_receipt.json",receipt,True)
    m.budget_check_v9(storage,boundary=True)
    check_final_deadline_v2()
    print(json.dumps(dict(scientific_completion_ref=done,IO_worker_receipt_ref=root_receipt)),flush=True)
    return 0

def launch(args):
    launch_started=time.monotonic();record,spec,cfg,storage=load_input(args)
    arm_final_timer_v2(launch_started,spec);b=monitor_helpers();m=configured_guard()
    m.validate_storage_v9(spec,cfg,storage,source_ref=SCIENCE_REF,config_ref=CFG_REF,runtime_ref=spec["runtime_ref"])
    aux=local(spec["auxiliary_directory"])
    need(not aux.exists() and not local(spec["output_directory"]).exists(),"Single launch; fresh output and auxiliary")
    m.budget_check_v9(storage,boundary=True,required_write_bytes=AUX);aux.mkdir()
    runtime=metadata(spec["runtime_ref"]);env=os.environ.copy();env.update(ENV)
    env.update(TMPDIR=str(aux),TMP=str(aux),TEMP=str(aux))
    rfd,wfd=os.pipe()
    argv=[runtime["executable"],"-B",str(ROOT/SOURCE),"--worker",
        "--input",record["path"],"--expected-sha256",record["sha256"],"--monitor-parent-pid",str(os.getpid()),"--monitor-read-fd",str(rfd)]
    proc=None;selector=None;owned={};groups={};peak=races=discarded=0;log=bytearray();error=None
    started=utc();clock=time.monotonic();term_at=None;killed=False;rc=None;minimal_ref=None;seed_ref=None;released=False
    def retain(chunk):
        nonlocal discarded
        keep=min(len(chunk),LOG_CAP-len(log));log.extend(chunk[:keep]);discarded+=len(chunk)-keep
    try:
        proc=subprocess.Popen(argv,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            start_new_session=True,pass_fds=(rfd,))
        row=b.process_info(proc.pid)
        need(row is not None and row["pgid"]==proc.pid and row["session"]==proc.pid,"Seed actual owned PID identity before child release")
        owned[proc.pid]=row["start_ticks"];groups[proc.pid]={proc.pid:row["start_ticks"]}
        initial,misses=b.observe(owned,groups);peak=initial;races+=misses
        need(initial<=RSS,"Initial aggregate owned+monitor RSS before any child compute")
        seed_ref=put(aux/"launch_monitor_seed.json",dict(schema="rootH018_precompute_owned_monitor_seed_v1",worker_pid=proc.pid,
            worker_start_ticks=row["start_ticks"],worker_pgid=row["pgid"],worker_session=row["session"],supervisor_pid=os.getpid(),
            initial_aggregate_RSS_bytes=initial,actual_owned_roster_seeded=True,child_compute_released=False,input_ref=record,checked_utc=utc()))
        selector=selectors.DefaultSelector();selector.register(proc.stdout,selectors.EVENT_READ)
        # Only now may worker configure guards/import tensor libraries/read payloads.
        os.write(wfd,(record["sha256"]+"\n").encode());released=True
        os.close(wfd);wfd=None;os.close(rfd);rfd=None
        while proc.poll() is None:
            now=time.monotonic()
            try:
                rss,misses=b.observe(owned,groups);peak=max(peak,rss);races+=misses
                need(rss<=RSS and now-clock<=DEADLINE,"Aggregate2GB RSS soft bound or1800s deadline exceeded")
                expiry(spec["expires_at_utc"])
                check_final_deadline_v2()
            except BaseException as exc:
                if error is None:error=dict(type=type(exc).__name__,message=str(exc)[:2048])
            if error is not None and term_at is None:proc.terminate();term_at=now
            if term_at is not None and now-term_at>=30 and not killed:
                b.signal_owned(groups,signal.SIGKILL);proc.kill();killed=True
            for key,_ in selector.select(.25):
                chunk=os.read(key.fd,65536)
                if chunk:retain(chunk)
                else:selector.unregister(key.fileobj)
    except BaseException as exc:
        error=error or dict(type=type(exc).__name__,message=str(exc)[:2048])
        if proc is not None:
            try:proc.terminate()
            except ProcessLookupError:pass
    finally:
        for fd in (rfd,wfd):
            if fd is not None:
                try:os.close(fd)
                except OSError as exc:error=error or dict(type=type(exc).__name__,message=str(exc)[:2048])
        if proc is not None:
            try:rc=proc.wait(timeout=30 if error else 5)
            except subprocess.TimeoutExpired:
                try:b.signal_owned(groups,signal.SIGKILL)
                except BaseException as exc:error=error or dict(type=type(exc).__name__,message=str(exc)[:2048])
                try:proc.kill()
                except ProcessLookupError:pass
                rc=proc.wait()
            # First postwait write; no guard, drain certificate or log precedes this.
            minimal_ref=put(aux/"actual_wait_minimal.json",dict(schema="rootH018_actual_direct_wait_minimal_v1",actual_direct_wait=True,
                returncode=rc,pid=proc.pid,worker_pgid=proc.pid,input_ref=record,io_context_ref=spec["io_context_ref"],
                started_utc=started,waited_utc=utc(),elapsed_seconds=time.monotonic()-clock),cap=8192)
    if proc is None:
        put(aux/"spawn_failed.json",dict(actual_direct_wait=False,error=error));raise RuntimeError("Popen failed; no actual wait exists")
    live=b.absent_groups(groups)
    if live:
        b.signal_owned(groups,signal.SIGKILL);until=time.monotonic()+10
        while live and time.monotonic()<until:time.sleep(.1);live=b.absent_groups(groups)
    if selector is not None:
        for _ in range(16):
            ready=selector.select(0)
            if not ready:break
            for key,_ in ready:
                chunk=os.read(key.fd,65536)
                if chunk:retain(chunk)
                else:selector.unregister(key.fileobj)
        selector.close()
    proc.stdout.close();need(not live,"Observed owned group unresolved; minimal wait retained, no final guard/log")
    m.budget_check_v9(storage,boundary=True,required_write_bytes=len(log)+65536)
    fd=os.open(aux/"worker_stdout.txt",os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,"wb") as f:f.write(log);f.flush();os.fsync(f.fileno())
    terminal=put(aux/"outer_wait.json",dict(schema="rootH018_actual980_outer_monitor_wait_v1",actual_wait_ref=minimal_ref,
        input_ref=record,returncode=rc,error=error,argv=argv,launch_monitor_seed_ref=seed_ref,child_compute_released=released,
        observed_owned_groups_ESRCH=True,observed_owned_groups=sorted(groups),observed_process_identities=owned,
        observed_peak_aggregate_RSS_bytes=peak,RSS_limit_bytes=RSS,RSS_is_polling_soft_bound_not_kernel_quota=True,
        aggregate_includes_outer_monitor=True,poll_seconds=.25,proc_exit_or_task_read_races=races,
        unobserved_between_poll_processes_possible=True,wall_deadline_seconds=DEADLINE,termination_grace_seconds=30,
        elapsed_seconds=time.monotonic()-clock,retained_stdout_ref=ref(aux/"worker_stdout.txt"),discarded_stdout_bytes=discarded))
    m.budget_check_v9(storage,boundary=True)
    check_final_deadline_v2()
    need(rc==0 and error is None,"Actual empirical CPU worker failed; preserve partials/direct wait, no fallback")
    return dict(status="completed_H056_"+spec["operation"]+"_worker_wait0_outer_root_terminal_still_required",scientific_completion_ref=ref(local(spec["output_directory"])/"completed.json"),
        actual_wait_ref=minimal_ref,outer_wait_ref=terminal,accuracy_available=False,automatic_completed_scope_credit=False)

def seal_preparation_wait_v1(root_input_ref,root_outer_wait,root_outer_wait_ref):
    """Pure root-owned evidence join after genuine external tool terminal0.
    Returns metadata only. Root persists it in its precharged metadata packet.
    """
    spec=metadata(root_input_ref)
    need(spec["operation"]=="prepare" and spec["schema"]=="H056_H36_CPU_preparation_root_v1","Only H056 bank preparation")
    aux=local(spec["auxiliary_directory"])
    minimal_ref=ref(aux/"actual_wait_minimal.json");outer_ref=ref(aux/"outer_wait.json")
    minimal,outer=metadata(minimal_ref),metadata(outer_ref)
    completed_ref=ref(local(spec["output_directory"])/"completed.json");done=metadata(completed_ref)
    need(minimal["actual_direct_wait"] is True and minimal["returncode"]==0 and minimal["input_ref"]==root_input_ref
      and outer["actual_wait_ref"]==minimal_ref and outer["returncode"]==0 and outer["error"] is None
      and outer["observed_owned_groups_ESRCH"] is True and outer["input_ref"]==root_input_ref,
      "Original actual direct worker wait0 and owned-group drain")
    need(root_outer_wait["schema"]=="H056_root_actual_bank_tool_wait_v1" and root_outer_wait["root_authorizer"]=="root_agent"
      and root_outer_wait["actual_tool_call_id"] and root_outer_wait["actual_tool_terminal_consumed"] is True
      and root_outer_wait["actual_tool_exit_code"]==0 and root_outer_wait["actual_owned_group_absent"] is True
      and root_outer_wait["input_ref"]==root_input_ref and root_outer_wait["worker_outer_wait_ref"]==outer_ref
      and root_outer_wait["bank_completion_ref"]==completed_ref,"Root genuine external tool terminal and current owned drain")
    need(metadata(root_outer_wait_ref)==root_outer_wait and done["schema"]=="H056_empirical_CPU_preparation_completed_v1"
      and done["input_ref"]==root_input_ref and done["status"]=="completed_observation_preparation","Persisted actual root evidence and new bank")
    return dict(schema="H056_root_genuine_preparation_wait_v1",protocol_ref=PROTOCOL_REF,actual_exit_code=0,
      actual_direct_wait_observed=True,process_group_drained=True,completed_ref=completed_ref,input_ref=root_input_ref,
      original_worker_minimal_wait_ref=minimal_ref,original_worker_outer_wait_ref=outer_ref,
      actual_root_tool_wait_ref=root_outer_wait_ref,source_ref=ref(SOURCE),sealed_utc=utc())

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group()
    g.add_argument("--launch",action="store_true");g.add_argument("--worker",action="store_true")
    p.add_argument("--input");p.add_argument("--expected-sha256")
    p.add_argument("--monitor-parent-pid",type=int);p.add_argument("--monitor-read-fd",type=int)
    args=p.parse_args(argv)
    if not(args.launch or args.worker):
        print(json.dumps(dict(status="inactive_source_only",admissions=0,model_calls=0)));return 0
    sys.dont_write_bytecode=True;sys.path[:0]=[str(ROOT),str(ROOT/"src")]
    os.environ.update(ENV);cpu()
    try:
        result=worker(args) if args.worker else launch(args)
        check_final_deadline_v2()
    finally:
        restore_final_timer_v2()
    if not args.worker:print(json.dumps(result,sort_keys=True),flush=True)
    return result if args.worker else 0

if __name__=="__main__":raise SystemExit(main())
