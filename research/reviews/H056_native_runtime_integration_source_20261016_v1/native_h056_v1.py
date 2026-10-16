"""H056-only root native caller. Import is inert and uses stdlib only.
Root supplies one genuine prospective980GB admission after actual acquisition.
This file does not admit, install, acquire, resume, train or apply a model.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, importlib.util, json, math, os, re, signal, stat, struct, subprocess, sys, time

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/H056_native_runtime_integration_source_20261016_v1"
SOURCE=HERE+"/native_h056_v1.py"
IMPLEMENTATION="research/reviews/H056_H36_implementation_20261016_v1"
PROTOCOL_REF={"path":"research/reviews/H056_frozen_H045_Danish_application_registration_20261016_v1/protocol.root_frozen_v1.json","sha256":"9544bdbd499bddafeb0774c67e4e86ef25535d41dd2409e710c29b679619cd6e"}
PROFILE={"effective_haploid_N":30000,"mutation_rate":"1.25e-8","years_per_generation":28,"population_size_iterations_executed":0,"pre_size_only":True,"seed":414855340,"native_memory_GB":"2"}
DATA_NAMES={"baseline.haps","baseline.sample","sample_ages.generations","qualified.regional.map","frozen_site_identities.json","polarization_equivalence.json","bridge_preparation.json"}
DIRECT_SUFFIXES={".py",".sh",".json"}
FIXED_REFS={'guard': {'path': 'research/reviews/prospective_980GB_capacity_source_v1/prospective_980gb_v1.py', 'sha256': 'fc71030865e0a3d5da31120d4daa2caa0b2d39ec7dd212b8c46c476b5a866d56'}, 'policy': {'path': 'research/reviews/prospective_980GB_capacity_source_v1/activation_policy.root_active_v1.json', 'sha256': 'eb3408004a508136aa434c0a2e02a9d14a5598a96b8483c264c93026f9326018'}, 'renewal': {'path': 'research/reviews/H039_marker_repair_renewal_source_20261016_v1/marker_repair_renewal_v1.py', 'sha256': '491236877e8d71cae8bd7619880858ac501c57dd93e59479ca32dacacab20468'}, 'renewal_certificate': {'path': 'research/reviews/H039_marker_repair_renewal_source_20261016_v1/renewal_certificate.root_active_v1.json', 'sha256': '66006202094d7a3ef2b29cbec28eb93f334505b72589f09d86e4b57498162db7'}, 'repair_authority': {'path': 'research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json', 'sha256': 'f6f2d628199e8be73db8b5d90d428570d8afdefeab9dc5cd94bb5eafb39247db'}, 'installation_metadata': {'path': 'research/reviews/H056_native_runtime_integration_source_20261016_v1/pinned_original_installation.metadata.v1.json', 'sha256': '578e5c04ce297c80c742376297f51412270421958085b2bb28ca86bb9d7e47c8'}, 'map_source_metadata': {'path': 'research/reviews/H056_map_source_metadata_20261016_v1/source_metadata.root_v1.json', 'sha256': '0745c2a210d855607abb710da840b1b513d53c96adfde7a67d6f4c75700a767e'}, 'memo': {'path': 'research/reviews/resource_guard_hotpath_source_audit_v1/transaction_memo_adapter_v2.py', 'sha256': 'c972e9a779d424ceafdb258848614bf40994d1004e2c73fc5cd225ae99621364'}, 'v4': {'path': 'src/arggnn/project_budget_v4.py', 'sha256': 'a093c7d86f57fee569cb99f4712250e9505a1119f3950fa44282c7ba8c57df47'}}
SOURCE_PATHS={"driver":SOURCE,"owned_helpers":HERE+"/native_owned_helpers_v1.py",
 "native_core":IMPLEMENTATION+"/native_h36_core_v1.py","contract":IMPLEMENTATION+"/contract_v1.py",
 "serialized_readback":"research/reviews/H056_acquisition_runtime_source_20261016_v1/serialized_readback_v1.py",
 "repaired_bridge":"research/reviews/H056_acquisition_runtime_source_20261016_v1/bridge_all_candidates_v1.py"}
_RAN=False

def require(ok,message):
    if not ok:raise ValueError(message)

def local(name):
    require(isinstance(name,str) and name and not Path(name).is_absolute() and ".." not in Path(name).parts,"Contained project-relative path")
    p=ROOT/name
    require(p!=ROOT and p.resolve()==p and not p.is_symlink(),"Canonical nonsymlink project path")
    return p

def exact_ref(ref):
    require(isinstance(ref,dict) and set(ref) in ({"path","sha256"},{"path","sha256","bytes"})
      and isinstance(ref["sha256"],str) and re.fullmatch("[0-9a-f]{64}",ref["sha256"]),"Exact persisted SHA256 reference")
    local(ref["path"])
    return ref

def plain_ref(record):
    exact_ref(record)
    return {k:record[k] for k in ("path","sha256")}

def ref(path,cap=64_000_000):
    p=path if isinstance(path,Path) else local(path)
    require(p.resolve()==p and ROOT in p.parents and not p.is_symlink(),"Canonical contained reference")
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        before=os.fstat(fd);require(stat.S_ISREG(before.st_mode) and before.st_size<=cap,"Bounded regular reference")
        digest=hashlib.sha256();total=0
        while True:
            part=os.read(fd,min(1048576,cap-total+1))
            if not part:break
            total+=len(part);require(total<=cap,"Bounded reference bytes");digest.update(part)
        after=os.fstat(fd)
    finally:os.close(fd)
    require(identity(before)==identity(after)==identity(p.lstat()) and total==before.st_size,"Stable reference read")
    return dict(path=p.relative_to(ROOT).as_posix(),sha256=digest.hexdigest())

def identity(s):
    return s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns

def raw(record,cap=16*2**20):
    exact_ref(record);p=local(record["path"]);fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        before=os.fstat(fd);require(stat.S_ISREG(before.st_mode) and before.st_size<=cap,"Bounded regular metadata/source")
        parts=[];total=0
        while True:
            part=os.read(fd,min(1048576,cap-total+1))
            if not part:break
            total+=len(part);require(total<=cap,"Bounded immutable read");parts.append(part)
        after=os.fstat(fd)
    finally:os.close(fd)
    data=b"".join(parts)
    require(identity(before)==identity(after)==identity(p.lstat()) and hashlib.sha256(data).hexdigest()==record["sha256"]
      and ("bytes" not in record or len(data)==record["bytes"]),"Unchanged pinned metadata/source")
    return data

def read(record,cap=16*2**20):
    def pairs(items):
        result={}
        for key,value in items:
            require(key not in result,"Unique JSON keys");result[key]=value
        return result
    def invalid(value):raise ValueError("No nonfinite JSON "+value)
    return json.loads(raw(record,cap),object_pairs_hook=pairs,parse_constant=invalid)

def load(record,name):
    code=raw(record);spec=importlib.util.spec_from_file_location(name,local(record["path"]))
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    exec(compile(code,str(local(record["path"])),"exec"),module.__dict__)
    return module

def encoded(value):
    return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()

def unique(refs):
    out={}
    for r in refs:
        exact_ref(r);require(set(r)=={"path","sha256"},"Direct bindings omit byte annotations")
        require(r["path"] not in out or out[r["path"]]==r["sha256"],"No conflicting direct bindings")
        out[r["path"]]=r["sha256"]
    return [dict(path=p,sha256=h) for p,h in sorted(out.items())]

def ceil_mib(n):return ((n+2**20-1)//2**20)*2**20

def derive_caps_v1(counts,max_chunks):
    """Finite abort ceilings, not predictions or established memory/usage fits."""
    require(set(counts)=={"candidate_sites","sites","constants"} and all(type(v)is int for v in counts.values())
      and counts["candidate_sites"]>=counts["sites"]>=1000 and 0<=counts["constants"]<=counts["sites"],"Actual C/S/K")
    S=counts["sites"]
    require(type(max_chunks)is int and 1<=max_chunks<=S,"Root finite chunk ceiling")
    children=5*max_chunks+3
    workspace=ceil_mib(8*2**20+8192*S+(20+8*max_chunks)+max_chunks*(16+4*S))
    facts=ceil_mib(2**20+3072*S)
    packet=ceil_mib(2**20+32768*children)
    output=workspace+facts+packet+2*2**20
    logs=min(16_000_000,1_000_000*children)
    total=ceil_mib(output+logs+2**20)
    return dict(native_workspace_abort_cap_bytes=workspace,native_facts_cap_bytes=facts,
      packet_cap_bytes=packet,native_output_cap_bytes=output,combined_log_cap_bytes=logs,
      output_cap_bytes=total,native_file_cap_bytes=max(workspace,facts,packet),
      native_output_file_count_cap=1024+32*children,input_bundle_cap_bytes=16*2**20,
      max_chunks=max_chunks,maximum_native_children=children)

class BindingIO:
    """Only existing inherited H018 immutable/helper API, no new dispatcher."""
    local=staticmethod(local)
    read=staticmethod(read)
    ref=staticmethod(ref)
    encoded=staticmethod(encoded)
    def __init__(self,helper,packet,caps):
        self.helper,self.packet,self.caps=helper,packet,caps
        self.packet_serialized_bytes=0
    def save(self,folder,name,value):
        require(folder==self.packet,"Only the fixed owned packet directory")
        payload=encoded(value)
        require(self.packet_serialized_bytes+len(payload)<=self.caps["packet_cap_bytes"],"Bounded packet cumulative bytes")
        self.helper.write_bytes(folder,name,payload)
        self.packet_serialized_bytes+=len(payload)
        return ref(folder/name,self.caps["native_file_cap_bytes"])

def actual_counts_v1(ts,base):
    """Independent topology census. No variants, mutation/ancestry labels or old facts."""
    samples=list(map(int,ts.samples()))
    require(samples==list(range(36)) and ts.num_samples==36,"Actual native36 sample insertion")
    total=0;ntrees=0
    for tree in ts.trees():
        reached=set()
        for sample in samples:
            node=sample;steps=0
            while node!=-1:
                require(0<=node<ts.num_nodes,"Actual bounded node reference")
                reached.add(node);steps+=1;require(steps<=ts.num_nodes,"Acyclic sampled ancestor walk")
                node=int(tree.parent(node))
        total+=len(reached);ntrees+=1
    result=dict(base,native_trees=int(ts.num_trees),native_nodes=int(ts.num_nodes),
      native_edges=int(ts.num_edges),vertex_occurrences=total)
    require(ntrees==result["native_trees"] and 0<ntrees<=base["sites"]
      and result["native_nodes"]>=36 and result["native_edges"]>0 and total>=36*ntrees,"Actual seven-count census")
    return result

def read_headers_v1(work,core,counts,J):
    def take(name,cap):
        p=work/"native"/name;fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
        try:
            before=os.fstat(fd);require(stat.S_ISREG(before.st_mode) and 0<before.st_size<=cap,"Bounded actual native header")
            data=os.read(fd,cap+1);after=os.fstat(fd)
        finally:os.close(fd)
        require(len(data)==before.st_size and identity(before)==identity(after)==identity(p.lstat()),"Stable actual native header")
        return data
    global_raw=take("parameters.bin",20+8*J)
    require(len(global_raw)>=20,"Actual complete header")
    n,S,c=struct.unpack("<iii",global_raw[:12])
    require(n==36 and S==counts["sites"] and 1<=c<=J,"Actual H36/site/chunk header")
    chunks={k:take("parameters_c"+str(k)+".bin",16+4*S) for k in range(c)}
    return core.parse_headers_v1(global_raw,chunks,counts=counts,max_chunks=J)

def validate_acquisition_v1(cfg,manifest):
    chromosome,counts=cfg["chromosome"],cfg["counts"]
    expected=read(cfg["counts_ref"])
    require(expected["schema"]=="H056_actual_acquisition_counts_v1" and expected["protocol_ref"]==PROTOCOL_REF
      and expected["chromosome"]==chromosome and expected["counts"]==counts,"Actual acquisition count seal")
    completed=read(cfg["bridge_completed_ref"]);readback=read(cfg["serialized_readback_ref"])
    require(completed["schema"]=="H056_acquisition_runtime_completed_v1"
      and completed["status"]=="completed_acquisition_bridge_and_serialized_readback_root_wait_required"
      and completed["chromosome"]==chromosome and completed["protocol_ref"]==PROTOCOL_REF
      and completed["counts"]==counts and completed["no_native_or_model_execution"] is True
      and plain_ref(completed["artifacts"]["counts.json"])==cfg["counts_ref"],"Exact genuine completed acquisition/count asset")
    require(set(manifest["data_refs"])==DATA_NAMES,"Exact original bridge assets only")
    for name,r in manifest["data_refs"].items():
        require(completed["artifacts"][name]==r and r["path"]==cfg["bridge_directory"]+"/"+name,"Exact actual bridge asset scope/ref")
    require(readback["schema"]=="H056_acquisition_serialized_readback_root_sealed_v1"
      and readback["status"]=="root_actual_wait0_drain_and_serialized_readback_sealed"
      and readback["protocol_ref"]==PROTOCOL_REF and readback["chromosome"]==chromosome and readback["counts"]==counts
      and readback["root_preparation_wait_verified"] is True and readback["root_owned_group_absent"] is True
      and readback["worker_completion_ref"]==cfg["bridge_completed_ref"]
      and readback["worker_readback_ref"]==plain_ref(completed["worker_readback_ref"])
      and readback["preparation_actual_wait_ref"]==cfg["bridge_actual_wait_ref"]
      and readback["retained_sites"]==counts["sites"],"New independently sealed serialized inputs")
    for key in ("actual_serialized_site_nucleotide_phase_order_verified","actual_serialized_sample_haplotype_date_join_verified",
      "actual_qualified_map_bytes_and_local_site_join_verified"):
        require(readback[key] is True,"New actual serialization/map proof")
    wait=read(cfg["bridge_actual_wait_ref"])
    require(wait["schema"]=="H056_root_actual_preparation_wait_v1" and wait["root_authorizer"]=="root_agent"
      and wait["actual_tool_call_id"] and wait["actual_tool_terminal_consumed"] is True and wait["actual_tool_exit_code"]==0
      and type(wait["worker_pid"])is int and wait["worker_pid"]>1 and wait["process_group"]==wait["worker_pid"]
      and wait["actual_worker_PID_absent"] is True and wait["actual_owned_group_absent"] is True
      and wait["worker_readback_ref"]==plain_ref(completed["worker_readback_ref"])
      and wait["worker_completion_ref"]==cfg["bridge_completed_ref"],"Root actual external terminal wait0 and exact owned drain")
    require(all(readback["actual_output_sha256"][name]==r["sha256"] for name,r in manifest["data_refs"].items()),"Readback binds every actual input byte")
    qualification=read(cfg["map_qualification_ref"])
    require(qualification["schema"]=="H056_map_qualification_v1" and completed["map_qualification_ref"]==cfg["map_qualification_ref"]
      and qualification["chromosome"]==chromosome and qualification["root_authorizer"]=="root_agent"
      and qualification["map_sha256"]==manifest["data_refs"]["qualified.regional.map"]["sha256"]
      and qualification["byte_identity_verified"] is True and qualification["source_semantics_qualified"] is True
      and qualification["source_map_brackets_full_requested_interval"] is True
      and qualification["uniform_map_substitution"] is False and qualification["source_positions_or_distances_rescaled"] is False,
      "Independent actual chromosome-specific original family map")
    source_map=read(FIXED_REFS["map_source_metadata"])
    asset=next(a for a in source_map["assets"] if a["chromosome"]==chromosome)
    require(qualification["source_repository"]==source_map["source_repository"]
      and qualification["source_commit"]==source_map["source_commit"]
      and qualification["source_path"]==asset["source_path"]
      and qualification["git_blob_sha1"]==asset["git_blob_sha1"]
      and completed["original_map_source_refs"]==dict(raw_map=qualification["raw_map_ref"],
        map=qualification["regional_map_ref"],map_semantics=qualification["source_semantics_ref"],
        map_qualification=cfg["map_qualification_ref"])
      and qualification["regional_map_ref"]["sha256"]==manifest["data_refs"]["qualified.regional.map"]["sha256"],
      "Exact fixed original map family source identity and actual four-reference acquisition join")
    return readback,qualification

def validate_root_v1(root_input_ref):
    spec=read(root_input_ref,65536)
    require(set(spec)=={"schema","status","root_authorizer","hypothesis_id","registered","execution_authorized",
      "driver_ref","config_ref","runtime_ref","source_manifest_ref","storage_ref"}
      and spec["schema"]=="H056_native_Relate_root_input_v1" and spec["status"]=="active"
      and spec["root_authorizer"]=="root_agent" and spec["hypothesis_id"]=="H056" and spec["registered"] is True
      and spec["execution_authorized"] is True and spec["driver_ref"]==ref(SOURCE),"One explicit authenticated H056 native execution input")
    cfg=read(spec["config_ref"]);caps=derive_caps_v1(cfg["counts"],cfg["caps"]["max_chunks"])
    require(cfg["schema"]=="H056_native_Relate_config_v1" and cfg["protocol_ref"]==PROTOCOL_REF
      and cfg["chromosome"] in ("20","21","22") and cfg["scientific_options"]==PROFILE and cfg["caps"]==caps
      and cfg["device"]=="cpu" and cfg["physical_gpu"] is None and cfg["cuda_visible_devices"]==""
      and cfg["no_replay_or_phase_resume"] is True and cfg["model_execution_authorized"] is False
      and cfg["fixed_refs"]==FIXED_REFS,"Exact native H056 pre-size science/caps/fixed refs")
    require(isinstance(cfg["job_id"],str) and re.fullmatch("H056-native-chr(20|21|22)-[A-Za-z0-9_.-]+",cfg["job_id"]),"Specific H056 chromosome job")
    require(cfg["job_id"].startswith("H056-native-chr"+cfg["chromosome"]+"-"),"Exact chromosome/job join")
    paths=[local(cfg[k]) for k in ("output_directory","auxiliary_directory","bridge_directory","packet_directory")]
    out,aux,bridge,packet=paths
    require(packet==out/"runtime" and out!=aux and out not in aux.parents and aux not in out.parents
      and out!=bridge and out not in bridge.parents and bridge not in out.parents
      and aux!=bridge and aux not in bridge.parents and bridge not in aux.parents,"Fresh disjoint native/aux and immutable bridge scopes")
    manifest=read(spec["source_manifest_ref"])
    require(set(manifest)=={"schema","protocol_ref","source_refs","metadata_refs","data_refs","original_bindings"}
      and manifest["schema"]=="H056_native_Relate_source_manifest_v1" and manifest["protocol_ref"]==PROTOCOL_REF
      and set(manifest["source_refs"])==set(SOURCE_PATHS)
      and all(manifest["source_refs"][k]["path"]==p for k,p in SOURCE_PATHS.items()),"Specific complete caller/core/repaired bridge source family")
    require(manifest["source_refs"]["driver"]==spec["driver_ref"],"Actual native driver source")
    for r in [*manifest["source_refs"].values(),*manifest["metadata_refs"].values(),*FIXED_REFS.values()]:raw(r)
    expected_meta={k:cfg[k] for k in ("counts_ref","bridge_completed_ref","serialized_readback_ref","bridge_actual_wait_ref","map_qualification_ref","roster_ref")}
    require(manifest["metadata_refs"]==dict(expected_meta,protocol_ref=PROTOCOL_REF),"Exact complete source metadata closure")
    require(len(manifest["original_bindings"])>0,"Actual original guard/state binding closure")
    direct=unique([*manifest["original_bindings"].values(),*manifest["source_refs"].values(),*manifest["metadata_refs"].values(),
      *(r for r in FIXED_REFS.values() if Path(r["path"]).suffix in DIRECT_SUFFIXES),spec["config_ref"],spec["runtime_ref"],spec["source_manifest_ref"]])
    require(len(direct)<=64 and all(Path(r["path"]).suffix in DIRECT_SUFFIXES for r in direct),"Full unchanged64 direct metadata/source bound")
    runtime=read(spec["runtime_ref"]);expires=datetime.fromisoformat(runtime["expires_at_utc"].replace("Z","+00:00"))
    issued=datetime.fromisoformat(runtime["created_at_utc"].replace("Z","+00:00"))
    installation=read(FIXED_REFS["installation_metadata"])
    require(runtime==dict(schema="H056_native_Relate_runtime_v1",root_authorizer="root_agent",job_id=cfg["job_id"],
      output_directory=cfg["output_directory"],auxiliary_directory=cfg["auxiliary_directory"],created_at_utc=runtime["created_at_utc"],
      expires_at_utc=runtime["expires_at_utc"],python_binary_ref=installation["python_binary_ref"],python_version=installation["python_version"],
      device="cpu",physical_gpu=None,CPU_threads=4,global_wall_seconds=2400,
      native_binary_refs=installation["native_binary_refs"],license_ref=installation["license_ref"]),"Actual exact native runtime")
    require(issued.tzinfo is not None and expires.tzinfo is not None and issued<=datetime.now(timezone.utc)<expires
      and 0<(expires-issued).total_seconds()<=2400,"Current bounded native authority")
    require(Path(sys.executable).resolve()==local(installation["python_binary_ref"]["path"]) and sys.version==installation["python_version"],"Original exact project Python")
    raw(installation["python_binary_ref"],32_000_000)
    storage=read(spec["storage_ref"]);own=storage["reservation"]
    require(storage["kind"]=="project_budget_v9_storage_record" and storage["active"] is True and storage["consumer_uses_v9_guard"] is True
      and storage["accounting_mode"]=="prospective_980gb_v1" and storage["job_id"]==cfg["job_id"]
      and storage["output_directory"]==cfg["output_directory"] and storage["expires_at_utc"]==runtime["expires_at_utc"]
      and own["reserved_bytes"]==storage["reserved_new_bytes"]==caps["output_cap_bytes"] and own["key"].startswith("H056/")
      and own["output"]==str(out) and own["auxiliary_scopes"]==[str(aux)] and unique(own["consumer_bindings"])==direct
      and storage["completed_overlay_policy_ref"]==FIXED_REFS["policy"],"Exact already admitted prospective980GB storage and complete direct closure")
    readback,qualification=validate_acquisition_v1(cfg,manifest)
    require(not out.exists() and not aux.exists() and bridge.is_dir(),"Native attempt is fresh and cannot resume")
    return spec,cfg,manifest,runtime,storage,readback,qualification,installation

def run_v1(root_input_ref):
    global _RAN
    require(not _RAN and Path.cwd().resolve()==ROOT and sys.platform.startswith("linux"),"One fresh root Linux native caller")
    _RAN=True;sys.dont_write_bytecode=True
    sys.path[:0]=[str(ROOT),str(ROOT/"src")]
    os.environ.update(CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1")
    require(not any(n in sys.modules for n in ("torch","torch_geometric","numpy","tskit")),"No scientific imports before source/authority guards")
    spec,cfg,manifest,runtime,storage,readback,qualification,installation=validate_root_v1(root_input_ref)
    renewal=load(FIXED_REFS["renewal"],"_H056_original_marker_renewal")
    renewal.install_renewal_v1(FIXED_REFS["renewal_certificate"])
    certificate,_=renewal.validate_certificate_v1(FIXED_REFS["renewal_certificate"])
    end=datetime.fromisoformat(runtime["expires_at_utc"].replace("Z","+00:00")).timestamp()
    require(end<datetime.fromisoformat(certificate["renewed_expires_at_utc"].replace("Z","+00:00")).timestamp(),"Expiry within authentic original renewal")
    guard=load(FIXED_REFS["guard"],"_H056_original_prospective980")
    guard.configure_v2(FIXED_REFS["repair_authority"],policy_ref=FIXED_REFS["policy"])
    memo=load(FIXED_REFS["memo"],"_H056_original_transaction_memo2")
    memo.install(guard.budget.v4,expected_source_path=local(FIXED_REFS["v4"]["path"]),
      expected_source_sha256=FIXED_REFS["v4"]["sha256"])
    preflight=guard.budget_check_v9(storage,first=True,boundary=True,required_write_bytes=0)
    require(preflight["bindings"]==manifest["original_bindings"],"Actual unchanged original guard bindings")
    h=load(manifest["source_refs"]["owned_helpers"],"_H056_owned_helpers")
    h.bind_h056_v1(cfg,cfg["caps"]);h.limit_process()
    require(signal.getsignal(signal.SIGCHLD)!=signal.SIG_IGN and signal.getitimer(signal.ITIMER_REAL)==(0.,0.),"Fresh direct-wait process without other timer")
    out,aux,packet=(local(cfg[k]) for k in ("output_directory","auxiliary_directory","packet_directory"))
    caps=cfg["caps"];b=BindingIO(h,packet,caps)
    def boundary(required=0):
        require(time.time()<end,"Original absolute native expiry")
        require(os.environ.get("CUDA_VISIBLE_DEVICES")=="" and not any(n in sys.modules for n in ("torch","torch_geometric")),"CPU-only no model")
        return guard.budget_check_v9(storage,boundary=True,required_write_bytes=required)
    started=time.monotonic();observations=[];terminal_inventories={}
    error=None;success=False;facts_ref=completed_ref=headers_ref=counts_ref=mapping_ref=contract_ref=None
    prior=signal.getsignal(signal.SIGALRM)
    def deadline(signum,frame):raise TimeoutError("H056 original2400s/absolute native deadline")
    signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,min(2400,end-time.time()))
    try:
        boundary(caps["output_cap_bytes"])
        out.mkdir();aux.mkdir();packet.mkdir();h.syncdir(out.parent);h.syncdir(aux.parent);h.syncdir(out)
        h.save(b,packet,"launch_claim.json",dict(schema="H056_native_launch_claim_v1",root_input_ref=root_input_ref,
          controller_pid=os.getpid(),started_utc=h.utc(),no_replay_or_phase_resume=True))
        work=out/"work";work.mkdir();h.syncdir(out)
        os.environ.update(TMPDIR=str(aux),TMP=str(aux),TEMP=str(aux))
        load(manifest["source_refs"]["contract"],"research.reviews.H056_H36_implementation_20261016_v1.contract_v1")
        core=load(manifest["source_refs"]["native_core"],"_H056_native_pure_core")
        payloads={name:h.bounded_file(b,r,caps["input_bundle_cap_bytes"]) for name,r in manifest["data_refs"].items()}
        require(sum(map(len,payloads.values()))<=caps["input_bundle_cap_bytes"],"Bounded original complete bridge")
        independent_readback=load(manifest["source_refs"]["serialized_readback"],"_H056_independent_map_readback")
        independent_readback.map_v1(payloads["qualified.regional.map"],qualification,cfg["chromosome"])
        roster_raw=h.bounded_file(b,cfg["roster_ref"],64000)
        observed=core.expected_observations_v1(payloads,roster_raw,readback,chromosome=cfg["chromosome"],counts=cfg["counts"])
        provenance=dict(schema="H056_native_original_GT_provenance_v1",protocol_ref=PROTOCOL_REF,root_input_ref=root_input_ref,
          config_ref=spec["config_ref"],source_manifest_ref=spec["source_manifest_ref"],runtime_ref=spec["runtime_ref"],
          original_bridge_refs=manifest["data_refs"],bridge_completed_ref=cfg["bridge_completed_ref"],
          readback_ref=cfg["serialized_readback_ref"],actual_acquisition_wait_ref=cfg["bridge_actual_wait_ref"],
          map_qualification_ref=cfg["map_qualification_ref"],scientific_options=PROFILE,installation_metadata_ref=FIXED_REFS["installation_metadata"],
          no_size_iterations=True,exact_source_build_identity_proven=False,immutable_original_GT_features=True,
          no_genotype_time_or_native_table_repair=True,no_query_truth_or_model=True)
        h.write_bytes(out,"provenance.json",encoded(provenance));h.syncdir(out)
        h.verify_runtime_packages(b)
        binaries={}
        for name,r in installation["native_binary_refs"].items():
            h.bounded_file(b,r,64_000_000,retain=False);binaries[name]=str(local(r["path"]))
        stages=core.cli_stages_v1(binaries,local(cfg["bridge_directory"]),PROFILE)
        expected_stages=None;aggregate_log_bytes=0
        relate_deadline=time.monotonic()+h.SETTINGS["Relate_seconds"]
        for name,argv,kind in stages:
            require(name=="make_chunks" or headers_ref is not None,"No later stage before exact actual headers")
            boundary(0)
            for r in installation["native_binary_refs"].values():h.bounded_file(b,r,64_000_000,retain=False)
            for key in ("baseline.haps","baseline.sample","sample_ages.generations","qualified.regional.map"):
                h.bounded_file(b,manifest["data_refs"][key],caps["input_bundle_cap_bytes"],retain=False)
            seconds=(relate_deadline-time.monotonic()) if kind=="relate" else h.SETTINGS["converter_seconds"]
            seconds=min(seconds,end-time.time(),2400-(time.monotonic()-started))
            log_remaining=caps["combined_log_cap_bytes"]-aggregate_log_bytes
            require(seconds>0 and log_remaining>0,"Original aggregate native time/log ceiling")
            h.save(b,packet,name+"_prelaunch.json",dict(schema="H056_native_prelaunch_v1",root_input_ref=root_input_ref,argv=argv,cwd=str(work),
              controller_pid=os.getpid(),started_utc=h.utc(),seconds=seconds,new_owned_process_group=True,stage_kind=kind,
              aggregate_log_bytes_before_stage=aggregate_log_bytes,remaining_aggregate_log_cap_bytes=log_remaining,native_headers_ref=headers_ref))
            require(len(os.listdir("/proc/self/task"))==1,"Original single-thread preexec native limits")
            proc=None;observation=None;log=b""
            try:
                proc=subprocess.Popen(argv,cwd=work,env=dict(os.environ),stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
                  stderr=subprocess.STDOUT,start_new_session=True,close_fds=True,preexec_fn=h.native_limits)
            finally:
                if proc is not None:observation,log=h.supervise(proc,seconds,out,aux,log_remaining)
            require(observation is not None,"Native spawn failed before direct wait")
            observations.append(dict(stage=name,**observation))
            # First postwait persistence precedes guards, logs, inventory and next stage.
            wait_ref=h.save(b,packet,name+"_direct_wait.json",dict(root_input_ref=root_input_ref,stage=name,**observation))
            aggregate_log_bytes+=observation["log_total_bytes"]
            h.write_bytes(aux,name+".log",log);h.syncdir(aux)
            require(aggregate_log_bytes<=caps["combined_log_cap_bytes"],"Original combined log ceiling")
            require(observation["actual_direct_wait_observed"] is True and observation["actual_exit_code"]==0
              and observation["process_group_drained"] is True and observation["control_error"] is None,"Stop after failed/unknown wait or non-drain")
            total,files=h.file_inventory(out,stable=True);logs,logfiles=h.file_inventory(aux,stable=True)
            require(total<=caps["native_output_cap_bytes"] and total+logs+131072<=caps["output_cap_bytes"],"Original terminal stable scope ceiling")
            terminal_inventories[name]=h.save(b,packet,name+"_terminal_inventory.json",dict(root_input_ref=root_input_ref,native_direct_wait_ref=wait_ref,
              stage=name,after_actual_wait0_and_owned_group_drain=True,two_consecutive_complete_equal_scans=True,
              output_bytes=total,auxiliary_bytes=logs,output_file_count=len(files),auxiliary_file_count=len(logfiles),
              bounded_attempts=h.SETTINGS["inventory_max_attempts"],active_inventory_is_sampled=True))
            if name=="make_chunks":
                headers=read_headers_v1(work,core,cfg["counts"],caps["max_chunks"])
                headers_ref=h.save(b,packet,"native_headers.json",dict(root_input_ref=root_input_ref,**headers))
                stages.extend(core.cli_stages_v1(binaries,local(cfg["bridge_directory"]),PROFILE,headers))
                require(len(stages)<=caps["maximum_native_children"],"Original declared5J+3 child ceiling")
                expected_stages=[s[0] for s in stages]
            boundary(0)
        require([o["stage"] for o in observations]==expected_stages and headers_ref is not None,"All exact declared native waits before graph facts")
        artifacts={}
        for name in ("native.anc","native.mut","native.trees"):
            artifacts[name]=ref(work/name,caps["native_file_cap_bytes"])
            h.bounded_file(b,artifacts[name],caps["native_file_cap_bytes"],retain=False)
        boundary(0)
        import tskit
        require(tskit.__version__=="1.0.3","Exact inherited tskit version")
        ts=tskit.load(str(work/"native.trees"))
        require(ref(work/"native.trees",caps["native_file_cap_bytes"])==artifacts["native.trees"],"Tree bytes immutable across native load")
        counts=actual_counts_v1(ts,cfg["counts"])
        count_doc=dict(schema="H056_actual_native_topology_counts_v1",protocol_ref=PROTOCOL_REF,chromosome=cfg["chromosome"],
          counts=counts,base_counts_ref=cfg["counts_ref"],native_graph_ref=artifacts["native.trees"],
          census="ts.num_trees,num_nodes,num_edges;sum explicit36-sample ancestor-union size over every marginal tree",
          variants_mutations_ancestry_labels_read_for_census=False,after_all_actual_native_wait0_and_owned_group_drain=True)
        counts_ref=h.save(b,packet,"actual_graph_counts.json",count_doc)
        # The verifier requires the full seven-count observation identity.
        observed_full=core.expected_observations_v1(payloads,roster_raw,readback,chromosome=cfg["chromosome"],counts=counts)
        anc_raw=h.bounded_file(b,artifacts["native.anc"],caps["native_file_cap_bytes"])
        mut_raw=h.bounded_file(b,artifacts["native.mut"],caps["native_file_cap_bytes"])
        facts=core.verify_native_v1(payloads,observed_full,anc_raw,mut_raw,ts,chromosome=cfg["chromosome"],counts=counts)
        facts.update(actual_graph_counts_ref=counts_ref,native_graph_ref=artifacts["native.trees"],
          native_direct_wait_refs={o["stage"]:ref(packet/(o["stage"]+"_direct_wait.json")) for o in observations},
          native_parameter_headers_ref=headers_ref,actual_acquisition_wait_ref=cfg["bridge_actual_wait_ref"],
          serialized_readback_ref=cfg["serialized_readback_ref"],map_qualification_ref=cfg["map_qualification_ref"],
          terminal_inventory_refs=terminal_inventories,aggregate_native_log_bytes=aggregate_log_bytes,
          actual_H056_native_executed=True,exact_source_build_identity_proven=False)
        facts_raw=encoded(facts);require(len(facts_raw)<=caps["native_facts_cap_bytes"],"Bounded actual strict and graph-only facts")
        boundary(0);h.write_bytes(out,"native_facts.json",facts_raw);h.syncdir(out)
        facts_ref=ref(out/"native_facts.json",caps["native_file_cap_bytes"])
        require(facts["schema"]=="H056_H36_actual_native_facts_v1" and facts["graph_only_ready"] is True
          and facts["counts"]==counts and facts["strict_GT_actual_status"] in
          ("verified_original_GT_pre_size_native_genealogy","unsupported_native_facts_preserved"),"New independent graph readiness; strictGT status stays actual")
        mapping=dict(schema="H056_actual_native_sample_mapping_v1",protocol_ref=PROTOCOL_REF,chromosome=cfg["chromosome"],
          counts=counts,source_native_facts_ref=facts_ref,rows=facts["native_sample_haplotype_date_mapping"])
        mapping_ref=h.save(b,packet,"native_sample_mapping.json",mapping)
        contract=dict(schema="H056_graph_structure_plus_original_GT_v1",protocol_ref=PROTOCOL_REF,chromosome=cfg["chromosome"],
          counts=counts,actual_graph_counts_ref=counts_ref,native_graph_ref=artifacts["native.trees"],native_facts_ref=facts_ref,
          graph_only_ready=True,immutable_original_GT_features=True,strict_GT_actual_status=facts["strict_GT_actual_status"],
          coverage=facts["coverage"],map_qualification_ref=cfg["map_qualification_ref"],serialized_readback_ref=cfg["serialized_readback_ref"],
          original_bridge_refs=manifest["data_refs"],no_historical_failure_or_recovery_reused=True)
        contract_ref=h.save(b,packet,"graph_contract.json",contract)
        for retained in out.rglob("*"):
            if retained.is_file():
                fd=os.open(retained,os.O_RDONLY|os.O_NOFOLLOW)
                try:os.fsync(fd)
                finally:os.close(fd)
        total,assets=h.file_inventory(out,True,stable=True);logs,_=h.file_inventory(aux,stable=True)
        require(total<=caps["native_output_cap_bytes"] and total+logs+131072<=caps["output_cap_bytes"],"Final original stable scope cap")
        boundary(0)
        terminal=dict(schema="H056_native_Relate_completed_v1",status="new_native_graph_qualified_original_GT_features_required",
          protocol_ref=PROTOCOL_REF,chromosome=cfg["chromosome"],job_id=cfg["job_id"],root_input_ref=root_input_ref,
          counts=counts,actual_graph_counts_ref=counts_ref,native_graph_ref=artifacts["native.trees"],native_facts_ref=facts_ref,
          native_mapping_ref=mapping_ref,graph_contract_ref=contract_ref,strict_GT_actual_status=facts["strict_GT_actual_status"],
          graph_only_ready=True,immutable_original_GT_features=True,actual_native_direct_wait_refs=facts["native_direct_wait_refs"],
          assets=assets,no_internal_controller_OS_exit_claim=True,root_actual_outer_wait0_and_drain_still_required=True,
          elapsed_seconds=time.monotonic()-started,reservation_remains_fully_charged=True,biological_or_accuracy_claim=False)
        h.write_bytes(out,"completed.json",encoded(terminal));h.syncdir(out)
        completed_ref=ref(out/"completed.json");success=True
    except BaseException as exc:
        error=h.error_record(exc);success=False
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,prior)
    # Never claim the current controller's own OS exit or complete old-writer absence.
    final=dict(schema="H056_native_root_controller_result_v1",root_input_ref=root_input_ref,success=success,
      status="completed_native_graph_qualification_root_outer_wait_required" if success else "failed_or_unsupported_preserved",
      completed_ref=completed_ref,facts_ref=facts_ref,actual_graph_counts_ref=counts_ref,native_mapping_ref=mapping_ref,
      graph_contract_ref=contract_ref,error=error,actual_native_wait_observations=observations,
      controller_OS_exit_not_self_certified=True,reservation_remains_fully_charged=True,
      all_partial_outputs_preserved=True,replay_or_phase_resume_authorized=False,finished_utc=h.utc())
    if packet.is_dir():
        result=h.save(b,packet,"controller_result.json",final)
        print(json.dumps(dict(controller_result_ref=result,success=success)),flush=True)
    else:print(json.dumps(final),flush=True)
    return 0 if success else 1

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--execute",action="store_true")
    p.add_argument("--root-input");p.add_argument("--root-input-sha256");a=p.parse_args(argv)
    if not a.execute:
        print(json.dumps(dict(status="source_only",native_executed=False,root_owns_admission=True)));return 0
    require(a.root_input and a.root_input_sha256,"Actual root input path/SHA required")
    return run_v1(dict(path=a.root_input,sha256=a.root_input_sha256))

if __name__=="__main__":raise SystemExit(main())
