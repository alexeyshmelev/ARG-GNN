"""H056-only guarded acquisition caller reusing the actual H055 guard/checker.
Inert on import. Root owns admission, invocation, genuine external waits and seals.
The first admitted acquisition qualifies BGZF/index transport before any data request.
Acquisition never runs a model or Relate.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, gzip, hashlib, importlib.util, io, json, os, re, resource, signal, stat, subprocess, sys, time
from urllib.request import build_opener,ProxyHandler,HTTPRedirectHandler
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/H056_acquisition_runtime_source_20261016_v1"
SOURCE=HERE+"/runtime_v2.py"
BASE_REF={'path': 'research/reviews/H055_CPU_runtime_source_20261016_v1/runtime_v1.py', 'sha256': '00ca41a2f5b4efb0a9d1b3c6812352c03a9f14c6dd6b4b76f6d88c738007a565'}
FIXED_REFS={'guard': {'path': 'research/reviews/prospective_980GB_capacity_source_v1/prospective_980gb_v1.py', 'sha256': 'fc71030865e0a3d5da31120d4daa2caa0b2d39ec7dd212b8c46c476b5a866d56'}, 'policy': {'path': 'research/reviews/prospective_980GB_capacity_source_v1/activation_policy.root_active_v1.json', 'sha256': 'eb3408004a508136aa434c0a2e02a9d14a5598a96b8483c264c93026f9326018'}, 'checker': {'path': 'research/reviews/GPU45_TRAIN_controller_execution_override_20261016_v1/controller_metadata_scope_check_v1.py', 'sha256': '2621353de0dc56f328f7f308ba17a341425ef2fc02d54739b6ccfc7d1a6ce2cf'}, 'renewal': {'path': 'research/reviews/H039_marker_repair_renewal_source_20261016_v1/marker_repair_renewal_v1.py', 'sha256': '491236877e8d71cae8bd7619880858ac501c57dd93e59479ca32dacacab20468'}, 'renewal_certificate': {'path': 'research/reviews/H039_marker_repair_renewal_source_20261016_v1/renewal_certificate.root_active_v1.json', 'sha256': '66006202094d7a3ef2b29cbec28eb93f334505b72589f09d86e4b57498162db7'}, 'repair_authority': {'path': 'research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json', 'sha256': 'f6f2d628199e8be73db8b5d90d428570d8afdefeab9dc5cd94bb5eafb39247db'}, 'memo': {'path': 'research/reviews/resource_guard_hotpath_source_audit_v1/transaction_memo_adapter_v2.py', 'sha256': 'c972e9a779d424ceafdb258848614bf40994d1004e2c73fc5cd225ae99621364'}, 'v4': {'path': 'src/arggnn/project_budget_v4.py', 'sha256': 'a093c7d86f57fee569cb99f4712250e9505a1119f3950fa44282c7ba8c57df47'}, 'base': {'path': 'research/reviews/H055_CPU_runtime_source_20261016_v1/runtime_v1.py', 'sha256': '00ca41a2f5b4efb0a9d1b3c6812352c03a9f14c6dd6b4b76f6d88c738007a565'}, 'protocol': {'path': 'research/reviews/H056_frozen_H045_Danish_application_registration_20261016_v1/protocol.root_frozen_v1.json', 'sha256': '9544bdbd499bddafeb0774c67e4e86ef25535d41dd2409e710c29b679619cd6e'}, 'contract': {'path': 'research/reviews/H056_H36_implementation_20261016_v1/contract_v1.py', 'sha256': '4c7727e27be2dddeb698dc78ff6d7ca143d85b47187a1fdd23dc7c17217d8662'}, 'extractor': {'path': 'research/reviews/H056_H36_implementation_20261016_v1/extract_h36_v1.py', 'sha256': '52b3a3e72f6ef641bbc6b77dd5941a9343eb8e81752ce874d463ba820419c749'}, 'bridge': {'path': 'research/reviews/H056_acquisition_runtime_source_20261016_v1/bridge_all_candidates_v1.py', 'sha256': 'dbb13e97dc3fb9051abd1ec0289f0f383d85eadb36df0a893703f81460cdb53a'}, 'bgzf': {'path': 'research/reviews/H056_acquisition_runtime_source_20261016_v1/bgzf_stream_v1.py', 'sha256': 'e660043a06cc7978b16fc049cfafb9f8cf05d1be962cd7439d64b878297c9c91'}, 'readback': {'path': 'research/reviews/H056_acquisition_runtime_source_20261016_v1/serialized_readback_v1.py', 'sha256': '0deb54f25d1fb1e9bd85b2c93e642d2f12044313519ce03b5e4dfd11dec666e4'}, 'container_fixture': {'path': 'research/reviews/H056_acquisition_runtime_source_20261016_v1/container_fixture_source_v1.py', 'sha256': '14504b08dd2ddb1275b99d4678989e942a483ad1133291f9cad8edae1585650d'}, 'owned_helpers': {'path': 'research/reviews/H056_acquisition_runtime_source_20261016_v1/acquisition_owned_helpers_v1.py', 'sha256': 'e3e3349a52a9ea621efd6cac2dd108cec99e08420611d5e412625ec47773bd21'}, 'original_supervisor': {'path': 'research/reviews/H018_observed2074_native_inference_source_v2/root_inference.py', 'sha256': '3c4f0228788d6d455b2b7bc84868ac3fed5854a0cf9afe7c63e417f75949b163'}, 'map_source_metadata': {'path': 'research/reviews/H056_map_source_metadata_20261016_v1/source_metadata.root_v1.json', 'sha256': '0745c2a210d855607abb710da840b1b513d53c96adfde7a67d6f4c75700a767e'}, 'bcftools': {'path': '.runtime/official_argmix_abi210_v1/cu129/env/bin/bcftools', 'sha256': '39c08aca9485dc1e9760c5c3cc9cf56d6900a92dc6696f814a40b38a09431d9c'}, 'bgzip': {'path': '.runtime/official_argmix_abi210_v1/cu129/env/bin/bgzip', 'sha256': 'f397a01076e1c7f2d047bce51cfb687f5c270287d36b72eb4e4758d7433da786'}, 'tabix': {'path': '.runtime/official_argmix_abi210_v1/cu129/env/bin/tabix', 'sha256': 'a3fc8998fde40931fdb26ed270f6e77b9965c9fa56f54f3790f4485d7d423962'}}
RSS_CAP, DEADLINE, ROOT_METADATA_CAP=2*2**30,4*3600,512*1024
LOGICAL_CAP,BRIDGE_CAP=200000000,16*2**20
ACQUISITION_CAP=64*2**20
NAME=re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
_RAN=False

def require(ok,message):
    if not ok:
        raise ValueError(message)

class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        raise ValueError("H056 refuses redirects before another source request")

def bootstrap_base_v1():
    p=ROOT/BASE_REF["path"]
    require(p.resolve()==p and not p.is_symlink(),"Canonical preserved H055 caller")
    before=p.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink==1 and before.st_size<65536,"Bounded original H055 source")
    data=p.read_bytes()
    require(hashlib.sha256(data).hexdigest()==BASE_REF["sha256"]
            and (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==
                (p.lstat().st_dev,p.lstat().st_ino,p.lstat().st_size,p.lstat().st_mtime_ns,p.lstat().st_ctime_ns),
            "Unchanged actual H055 caller source")
    spec=importlib.util.spec_from_file_location("_H056_original_H055_caller",p)
    base=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=base
    exec(compile(data,str(p),"exec"),base.__dict__)
    return base

def run_v1(root_input_ref):
    global _RAN
    require(not _RAN and sys.platform.startswith("linux"),"One fresh Linux CPU caller")
    _RAN=True
    started=time.monotonic()
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    sys.dont_write_bytecode=True
    require(os.getpgrp()==os.getpid(),"Root must invoke this caller in its own new session/group")
    require(not any(n in sys.modules for n in ("torch","torch_geometric","tensorflow","jax","cupy")),"CPU-only fresh interpreter")
    soft,hard=resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS,(min([RSS_CAP]+[x for x in (soft,hard) if x!=resource.RLIM_INFINITY]),hard))
    base=bootstrap_base_v1()
    spec=base.read(root_input_ref,65536)
    keys={"schema","status","root_authorizer","hypothesis_id","registered","execution_authorized","job_id",
          "output_directory","created_at_utc","expires_at_utc","runtime_entry_ref","refs"}
    source_ref={"path":SOURCE,"sha256":hashlib.sha256(base.path(SOURCE).read_bytes()).hexdigest()}
    require(Path(__file__).resolve()==base.path(SOURCE) and set(spec)==keys
            and spec["schema"]=="H056_acquisition_CPU_root_input_v1" and spec["status"]=="active"
            and spec["root_authorizer"]=="root_agent" and spec["hypothesis_id"]=="H056"
            and spec["registered"] is True and spec["execution_authorized"] is True
            and isinstance(spec["job_id"],str) and NAME.fullmatch(spec["job_id"])
            and spec["runtime_entry_ref"]==source_ref
            and set(spec["refs"])=={"config","runtime","source_closure","storage"},"Exact root H056 acquisition input")
    issued=datetime.fromisoformat(spec["created_at_utc"].replace("Z","+00:00"))
    expires=datetime.fromisoformat(spec["expires_at_utc"].replace("Z","+00:00"))
    require(issued.tzinfo is not None and expires.tzinfo is not None and issued<=datetime.now(timezone.utc)<expires
            and 0<(expires-issued).total_seconds()<=DEADLINE,"Current original absolute authority at most4h")
    end=expires.timestamp()
    out=base.path(spec["output_directory"])
    require(spec["output_directory"].startswith("experiments/results/") and out.parent.is_dir() and not out.exists(),
            "Fresh scientific scope under an existing project parent")
    metadata=base.path(spec["refs"]["config"]["path"]).parent
    require(metadata!=out and metadata not in out.parents and out not in metadata.parents
            and all(base.path(r["path"]).parent==metadata for r in spec["refs"].values()),"One disjoint root metadata folder")
    for ref in FIXED_REFS.values():
        base.raw(ref)
    cfg=base.read(spec["refs"]["config"])
    ck={"schema","root_authorizer","hypothesis_id","mode","chromosome","fixed_refs",
        "asset_ref","roster_ref","map_ref","map_qualification_ref","raw_map_ref","map_semantics_ref",
        "container_qualification_ref","device","physical_gpu","cuda_visible_devices","max_cpu_rss_bytes",
        "global_wall_seconds","logical_extraction_cap_bytes","charged_scope_cap_bytes",
        "max_bridge_serialized_bytes","original_scope_checker_reuse_root_authorized"}
    mode=cfg.get("mode")
    cap=ACQUISITION_CAP
    require(set(cfg)==ck and cfg["schema"]=="H056_acquisition_CPU_config_v1"
            and cfg["root_authorizer"]=="root_agent" and cfg["hypothesis_id"]=="H056"
            and mode=="acquisition" and cfg["fixed_refs"]==FIXED_REFS
            and cfg["device"]=="cpu" and cfg["physical_gpu"] is None and cfg["cuda_visible_devices"]==""
            and cfg["max_cpu_rss_bytes"]==RSS_CAP and cfg["global_wall_seconds"]==DEADLINE
            and cfg["logical_extraction_cap_bytes"]==LOGICAL_CAP and cfg["charged_scope_cap_bytes"]==cap
            and cfg["max_bridge_serialized_bytes"]==BRIDGE_CAP
            and cfg["original_scope_checker_reuse_root_authorized"] is True,"Exact fixed CPU/logical/durable resource config")
    dynamic_names=("asset","roster","map","map_qualification","raw_map","map_semantics")
    require(cfg["chromosome"] in ("20","21","22"),"Prespecified H056 chromosome")
    dynamic={n:base.exact_ref(cfg[n+"_ref"]) for n in dynamic_names}
    first_container=cfg["container_qualification_ref"] is None
    if not first_container:
        dynamic["container_qualification"]=base.exact_ref(cfg["container_qualification_ref"])
    runtime={k:spec[k] for k in ("job_id","output_directory","created_at_utc","expires_at_utc")}
    runtime.update(schema="H056_acquisition_CPU_runtime_v1",root_authorizer="root_agent",mode=mode,
                   device="cpu",physical_gpu=None,cuda_visible_devices="",global_wall_seconds=DEADLINE)
    require(base.read(spec["refs"]["runtime"])==runtime,"Exact actual runtime/output/expiry")
    closure=base.read(spec["refs"]["source_closure"])
    bound=dict(FIXED_REFS,runtime_entry=source_ref,config=spec["refs"]["config"],runtime=spec["refs"]["runtime"],**dynamic)
    require(set(closure)=={"schema","root_authorizer","bound_refs","original_bindings"}
            and closure["schema"]=="H056_acquisition_CPU_source_closure_v1" and closure["root_authorizer"]=="root_agent"
            and closure["bound_refs"]==bound and isinstance(closure["original_bindings"],dict)
            and closure["original_bindings"],"Exact inherited original full-state closure")
    direct=base.unique([*closure["original_bindings"].values(),
        *(r for r in bound.values() if Path(r["path"]).suffix in {".py",".sh",".json"}),spec["refs"]["source_closure"]])
    require(len(direct)<=64 and all(Path(r["path"]).suffix in {".py",".sh",".json"} for r in direct),"Original64 direct binding limit")
    storage=base.read(spec["refs"]["storage"])
    own=storage["reservation"]
    require(storage.get("kind")=="project_budget_v9_storage_record" and storage.get("active") is True
            and storage.get("consumer_uses_v9_guard") is True and storage.get("accounting_mode")=="prospective_980gb_v1"
            and storage["job_id"]==spec["job_id"] and storage["output_directory"]==spec["output_directory"]
            and storage["expires_at_utc"]==spec["expires_at_utc"]
            and type(storage["reserved_new_bytes"]) is int and storage["reserved_new_bytes"]==own["reserved_bytes"]==cap
            and own["output"]==str(out) and own["auxiliary_scopes"]==[]
            and isinstance(own["key"],str) and own["key"].startswith("H056/")
            and isinstance(own["consumer_bindings"],list) and len(own["consumer_bindings"])<=64
            and base.unique(own["consumer_bindings"])==direct
            and storage["completed_overlay_policy_ref"]==FIXED_REFS["policy"],"Original real open fully charged scope and bindings")
    filesystem=os.statvfs(out.parent)
    unit=max(4096,filesystem.f_frsize,filesystem.f_bsize)
    overhead=64*unit
    require(unit<=2**20 and cap<=filesystem.f_bavail*filesystem.f_frsize
            and overhead+ROOT_METADATA_CAP<cap,"Conservative finite directory/file allocation slack")
    checker=None
    previous=signal.getsignal(signal.SIGALRM)
    require(signal.getitimer(signal.ITIMER_REAL)==(0.,0.),"Fresh original wall timer")
    def alarm(signum,frame):
        raise TimeoutError("H056 original4h/absolute authority deadline")
    signal.signal(signal.SIGALRM,alarm)
    signal.setitimer(signal.ITIMER_REAL,min(DEADLINE-(time.monotonic()-started),end-time.time()))
    counters=dict(serialized_bytes=0,logical_extraction_bytes=0,root_serialized_bytes=0,request_callbacks=0)
    latest=None
    scope_checks=0
    last_charge,last_check=0,time.monotonic()
    def runtime_check():
        require(time.monotonic()-started<DEADLINE and time.time()<end,"Original global/absolute deadline")
        require(os.environ.get("CUDA_VISIBLE_DEVICES")=="" and not any(n in sys.modules for n in
            ("torch","torch_geometric","tensorflow","jax","cupy")),"No model/GPU imports")
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=RSS_CAP,"CPU peak self RSS<=2GiB")
    def boundary(pending=0,force=False):
        nonlocal latest,scope_checks,last_charge,last_check
        runtime_check()
        require(type(pending) is int and 0<=pending and counters["serialized_bytes"]+pending+overhead+ROOT_METADATA_CAP<=cap,
                "Durable serialized content plus allocation/receipt slack fits charged scope")
        if force or counters["serialized_bytes"]+pending-last_charge>=2**20 or time.monotonic()-last_check>=60:
            latest=checker.check(required=pending+overhead+ROOT_METADATA_CAP)
            scope_checks+=1
            last_charge,last_check=counters["serialized_bytes"]+pending,time.monotonic()
            runtime_check()
    payload=out/"payload"
    aux=out/"tool_aux"
    artifacts={}
    def artifact(p):
        require(p.resolve()==p and p.is_file() and not p.is_symlink(),"Canonical completed artifact")
        before=p.stat()
        digest=hashlib.sha256()
        with p.open("rb") as f:
            for b in iter(lambda:f.read(2**20),b""):
                digest.update(b)
        require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==
                (p.stat().st_dev,p.stat().st_ino,p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_ctime_ns),"Stable completed artifact bytes")
        return dict(path=str(p.relative_to(ROOT)),sha256=digest.hexdigest(),bytes=before.st_size)
    def save(folder,name,data,root=False):
        require(Path(name).name==name and isinstance(data,bytes),"Exclusive basename-only immutable write")
        if root:
            require(counters["root_serialized_bytes"]+len(data)<=ROOT_METADATA_CAP,"Root metadata total512KiB")
        boundary(len(data),force=root)
        p=folder/name
        with os.fdopen(os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),"wb") as f:
            f.write(data);f.flush();os.fsync(f.fileno())
        counters["serialized_bytes"]+=len(data)
        if root:counters["root_serialized_bytes"]+=len(data)
        return artifact(p)
    def ref2(value):
        return {k:value[k] for k in ("path","sha256")}
    def save_json(folder,name,value,root=False):
        return save(folder,name,(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode(),root)
    try:
        runtime_check()
        sys.path[:0]=[str(ROOT),str(ROOT/"src")]
        renewal=base.load(FIXED_REFS["renewal"],"_H056_original_marker_renewal")
        renewal.install_renewal_v1(FIXED_REFS["renewal_certificate"])
        cert,_=renewal.validate_certificate_v1(FIXED_REFS["renewal_certificate"])
        require(end<datetime.fromisoformat(cert["renewed_expires_at_utc"].replace("Z","+00:00")).timestamp(),"Expiry fits authentic renewal")
        guard=base.load(FIXED_REFS["guard"],"_H056_original980_guard")
        guard.configure_v2(FIXED_REFS["repair_authority"],policy_ref=FIXED_REFS["policy"])
        check_module=base.load(FIXED_REFS["checker"],"_H056_original_scope_checker")
        checker=check_module.ControllerMetadataScopeV1(guard,storage)
        require(checker.bindings==closure["original_bindings"],"Unchanged complete original startup bindings")
        boundary(force=True)
        require(not out.exists(),"Fresh output before original guarded writes")
        out.mkdir(mode=0o700);payload.mkdir(mode=0o700);aux.mkdir(mode=0o700)
        start_ticks=int(Path("/proc/self/stat").read_text().split(") ",1)[1].split()[19])
        save_json(out,"started.json",dict(schema="H056_acquisition_started_v1",mode=mode,input_ref=root_input_ref,
            worker_pid=os.getpid(),worker_start_ticks=start_ticks,process_group=os.getpgrp(),
            source_ref=source_ref,storage_ref=spec["refs"]["storage"],original_full_startup_guard_passed=True),True)
        helpers=base.load(FIXED_REFS["owned_helpers"],"_H056_original_owned_tool_helpers")
        helpers.bind_acquisition_v1(str(payload.relative_to(ROOT)),str(aux.relative_to(ROOT)),cap)
        bgzf=base.load(FIXED_REFS["bgzf"],"_H056_BGZF_encoder")
        readback=base.load(FIXED_REFS["readback"],"_H056_independent_serialized_readback")
        waits=[]
        log_bytes=0
        def supervised_tool(label,argv):
            nonlocal log_bytes
            require(label in {"integrity","index","samples","count","contigs","VCF_integrity","VCF_index","VCF_samples",
                              "VCF_count","VCF_contigs","QC_integrity"},"Finite exact H056 container operation")
            executable=base.path(FIXED_REFS[("bgzip" if label in ("integrity","VCF_integrity","QC_integrity") else
                 "tabix" if label in ("index","contigs","VCF_index","VCF_contigs") else "bcftools")]["path"])
            require(argv[0]==str(executable) and all(isinstance(x,str) for x in argv),"Pinned original tool executable")
            tool_file_cap=2**20
            boundary(tool_file_cap,force=True)
            def limits():
                os.environ.update(CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
                resource.setrlimit(resource.RLIMIT_AS,(RSS_CAP,RSS_CAP))
                resource.setrlimit(resource.RLIMIT_FSIZE,(tool_file_cap,tool_file_cap))
                resource.setrlimit(resource.RLIMIT_CORE,(0,0))
                resource.setrlimit(resource.RLIMIT_CPU,(120,130))
            proc=subprocess.Popen(argv,cwd=aux,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                  start_new_session=True,preexec_fn=limits)
            observed,captured=helpers.supervise(proc,min(120,end-time.time()),payload,aux,65536)
            waits.append(dict(stage=label,argv=argv,observation=observed,stdout_sha256=hashlib.sha256(captured).hexdigest()))
            log_bytes+=len(captured)
            save(aux,label+".stdout.txt",captured)
            save_json(aux,label+".wait.json",observed)
            require(observed["actual_direct_wait_observed"] is True and observed["actual_exit_code"]==0
                    and observed["owned_group_absent_after_wait"] is True and observed["control_error"] is None,
                    "Actual successful original tool wait and exact owned group drain")
            # Index writes bypass our serializer, but original supervisor observes full owned scopes.
            total,_=helpers.file_inventory(payload,stable=True)
            aux_total,_=helpers.file_inventory(aux,stable=True)
            root_total=sum(p.stat().st_size for p in out.iterdir() if p.is_file())
            counters["serialized_bytes"]=total+aux_total+root_total
            boundary(force=True)
            return captured
        tools={n:str(base.path(FIXED_REFS[n]["path"])) for n in ("bgzip","tabix","bcftools")}
        container_worker_ref=None
        if first_container:
            fixture=base.load(FIXED_REFS["container_fixture"],"_H056_tiny_container_only_fixture")
            def fixture_save(name,data):
                save(payload,name,data)
                return str(payload/name)
            transport=fixture.qualify_v1(bgzf.BGZFEncoderV1,fixture_save,supervised_tool,**tools)
            transport.update(schema="H056_container_worker_qualification_v1",root_authorizer="root_agent",
                actual_tool_waits=list(waits),root_preparation_wait_verified=False,root_owned_group_absent=False,
                source_refs={k:FIXED_REFS[k] for k in ("bgzf","container_fixture","owned_helpers","bgzip","tabix","bcftools")},
                scientific_QC_or_model_qualification=False)
            container_worker_ref=ref2(save_json(out,"container_qualification.worker.json",transport,True))
        else:
            # Authenticate the root-sealed previous one-time container qualification.
            seal=base.read(dynamic["container_qualification"])
            require(seal["schema"]=="H056_container_root_actual_wait_seal_v1" and seal["root_authorizer"]=="root_agent"
                    and seal["passed"] is True and seal["root_preparation_wait_verified"] is True
                    and seal["root_owned_group_absent"] is True,"Genuine root-sealed completed container qualification")
            worker=base.read(seal["worker_qualification_ref"])
            wcomplete=base.read(seal["worker_completion_ref"])
            wwait=base.read(seal["root_actual_wait_ref"])
            require(seal==readback.seal_container_root_postwait_v1(worker,seal["worker_qualification_ref"],
                seal["worker_completion_ref"],wwait,seal["root_actual_wait_ref"])
                and wcomplete["worker_qualification_ref"]==seal["worker_qualification_ref"]
                and worker["source_refs"]=={k:FIXED_REFS[k] for k in ("bgzf","container_fixture","owned_helpers","bgzip","tabix","bcftools")},
                "Exact current container/tools source qualification and actual persisted wait chain")
        asset=base.read(dynamic["asset"])
        roster=base.read(dynamic["roster"])
        maps=base.raw(dynamic["map"],2**20)
        qualification=base.read(dynamic["map_qualification"])
        semantics=base.read(dynamic["map_semantics"])
        raw_map=base.raw(dynamic["raw_map"],2**20)
        require(qualification["regional_map_ref"]==dynamic["map"] and qualification["raw_map_ref"]==dynamic["raw_map"]
                and qualification["source_semantics_ref"]==dynamic["map_semantics"]
                and qualification["sites_and_haps_map_join_qualified"] is False,
                "Exact same original-family map/raw/semantic qualifications; empirical joins still pending")
        map_metadata=base.read(FIXED_REFS["map_source_metadata"])
        map_assets=[a for a in map_metadata["assets"] if a["chromosome"]==cfg["chromosome"]]
        require(len(map_assets)==1,"Exact frozen chromosome map source metadata")
        map_asset=map_assets[0]
        require(qualification["schema"]=="H056_map_qualification_v1"
                and qualification["source_repository"]==map_metadata["source_repository"]=="https://github.com/odelaneau/GLIMPSE"
                and qualification["source_commit"]==map_metadata["source_commit"]=="103ca071ea9e4e648192916142e5ed353ac94e27"
                and qualification["source_path"]==map_asset["source_path"]
                and qualification["git_blob_sha1"]==map_asset["git_blob_sha1"]
                and len(raw_map)==map_asset["bytes"]
                and hashlib.sha1(b"blob "+str(len(raw_map)).encode()+b"\0"+raw_map).hexdigest()==map_asset["git_blob_sha1"]
                and semantics["schema"]=="H056_map_source_semantics_v1"
                and semantics["root_authorizer"]=="root_agent" and semantics["status"]=="root_reviewed_source_semantics"
                and semantics["chromosome"]==cfg["chromosome"],"Exact original GLIMPSE commit/blob and active root map semantics")
        require(base.read(qualification["root_source_semantics_ref"])==semantics,
                "Copied actual map semantic bytes equal exact original root semantic authority")
        readback.map_v1(maps,qualification,cfg["chromosome"])
        extractor=base.load(FIXED_REFS["extractor"],"_H056_exact_extraction_core")
        bridge=base.load(FIXED_REFS["bridge"],"_H056_QC0_all_candidate_bridge")
        extractor.validate_asset_v1(asset)
        require(asset["chromosome"]==cfg["chromosome"],"One exact configured chromosome")
        extractor.validate_roster_identity_v1(roster)
        streams={}
        class Writer:
            def append(self,name,data):
                require(name in {"H056.chr"+cfg["chromosome"]+".noAP.vcf","regional_site_QC.tsv"} and isinstance(data,bytes),
                        "Only original AP-free source streams")
                require(counters["logical_extraction_bytes"]+len(data)<=LOGICAL_CAP,"Original200MB logical extraction cap, no thinning")
                if name not in streams:
                    p=payload/(name+".gz")
                    f=os.fdopen(os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),"wb",buffering=0)
                    def emit(raw,f=f):
                        boundary(len(raw))
                        written=f.write(raw)
                        require(written==len(raw),"Full guarded BGZF block write")
                        counters["serialized_bytes"]+=len(raw)
                    streams[name]=(f,bgzf.BGZFEncoderV1(emit))
                streams[name][1].append(data)
                counters["logical_extraction_bytes"]+=len(data)
        def before_request():
            boundary()
            counters["request_callbacks"]+=1
        try:
            result=extractor.execute_v1(dict(root_authorizer="root_agent",admitted=True,
                protocol_ref=FIXED_REFS["protocol"],roster=roster,asset=asset,opener=build_opener(ProxyHandler({}),NoRedirect()).open,
                before_request=before_request,writer=Writer(),output_cap_bytes=LOGICAL_CAP))
            compressed={}
            for name,(f,encoder) in streams.items():
                compressed[name]=encoder.close()
                f.flush();os.fsync(f.fileno());f.close()
                artifacts[name+".gz"]=artifact(payload/(name+".gz"))
        finally:
            for f,encoder in streams.values():
                if not f.closed:f.close()
        require(result["status"]=="acquired_QC_qualified_root_completion_and_readback_required",
                "Under1000 qualified sites remains inconclusive, no bridge/native success")
        counts=result["counts"]
        require(result["serialized_bytes"]==counters["logical_extraction_bytes"],"Independent logical extraction counter equality")
        result.update(serialized_stream_encoding="BGZF",stream_container_metadata=compressed,
                      logical_extraction_cap_bytes=LOGICAL_CAP,charged_scope_cap_bytes=cap)
        artifacts["extraction.json"]=save_json(payload,"extraction.json",result)
        artifacts["counts.json"]=save_json(payload,"counts.json",dict(schema="H056_actual_acquisition_counts_v1",protocol_ref=FIXED_REFS["protocol"],chromosome=cfg["chromosome"],counts=counts))
        vcf_name="H056.chr"+cfg["chromosome"]+".noAP.vcf.gz"
        vcf=str(payload/vcf_name)
        supervised_tool("VCF_integrity",[tools["bgzip"],"-t",vcf])
        supervised_tool("VCF_index",[tools["tabix"],"-p","vcf",vcf])
        samples=supervised_tool("VCF_samples",[tools["bcftools"],"query","-l",vcf])
        observed_count=supervised_tool("VCF_count",[tools["bcftools"],"index","-n",vcf])
        contigs=supervised_tool("VCF_contigs",[tools["tabix"],"-l",vcf])
        supervised_tool("QC_integrity",[tools["bgzip"],"-t",str(payload/"regional_site_QC.tsv.gz")])
        require(samples.decode().splitlines()==list(extractor.SUBJECTS)
                and observed_count.strip()==str(counts["candidate_sites"]).encode()
                and contigs.strip()==cfg["chromosome"].encode(),"Actual tools18-sample/total-candidate/chromosome join")
        artifacts[vcf_name+".tbi"]=artifact(payload/(vcf_name+".tbi"))
        vcf_raw=base.raw({k:artifacts[vcf_name][k] for k in ("path","sha256")},24*2**20)
        qc_compressed=base.raw({k:artifacts["regional_site_QC.tsv.gz"][k] for k in ("path","sha256")},cap)
        with gzip.GzipFile(fileobj=io.BytesIO(qc_compressed)) as f:
            qc_raw=f.read(80000001)
        require(len(qc_raw)<=80000000,"QC logical readback bound, no truncation accepted")
        payloads,bridge_report=bridge.assemble_baseline_v1(vcf_raw,qc_raw,roster,chromosome=cfg["chromosome"],counts=counts,
            expected_vcf_sha256=hashlib.sha256(vcf_raw).hexdigest(),expected_qc_sha256=hashlib.sha256(qc_raw).hexdigest(),
            max_output_bytes=BRIDGE_CAP)
        for name,data in payloads.items():
            artifacts[name]=save(payload,name,data)
        artifacts["qualified.regional.map"]=save(payload,"qualified.regional.map",maps)
        # Reopen bytes from durable output for an independent serialized readback.
        actual_payloads={name:base.raw({k:artifacts[name][k] for k in ("path","sha256")},BRIDGE_CAP)
                         for name in readback.CORE_NAMES}
        actual_map=base.raw({k:artifacts["qualified.regional.map"][k] for k in ("path","sha256")},2**20)
        verified=readback.verify_v1(vcf_raw,qc_raw,roster,actual_payloads,actual_map,qualification,
            chromosome=cfg["chromosome"],counts=counts,extraction=result,protocol_ref=FIXED_REFS["protocol"])
        verified.update(actual_source_stream_refs={n:artifacts[n] for n in (vcf_name,"regional_site_QC.tsv.gz")},
            count_seal_ref=ref2(artifacts["counts.json"]),original_map_source_refs=dynamic,
            container_tool_actual_waits=waits,source_ref=source_ref)
        result_ref=ref2(save_json(out,"serialized_readback.worker.json",verified,True))
        completed=ref2(save_json(out,"completed.json",dict(schema="H056_acquisition_runtime_completed_v1",
            status="completed_acquisition_bridge_and_serialized_readback_root_wait_required",
            chromosome=cfg["chromosome"],protocol_ref=FIXED_REFS["protocol"],counts=counts,artifacts=artifacts,
            worker_readback_ref=result_ref,worker_qualification_ref=container_worker_ref,
            extraction_ref=ref2(artifacts["extraction.json"]),input_ref=root_input_ref,
            storage_ref=spec["refs"]["storage"],map_qualification_ref=dynamic["map_qualification"],
            original_map_source_refs={n:dynamic[n] for n in ("raw_map","map","map_semantics","map_qualification")},
            no_native_or_model_execution=True,root_preparation_wait_verified=False,
            peak_self_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            elapsed_seconds=time.monotonic()-started,counters_before_completion=dict(counters)),True))
        boundary(force=True)
        final_scope=checker.check(required=0)
        runtime_check()
        return dict(status="completed_H056_acquisition_worker_root_wait_required",worker_readback_ref=result_ref,
            worker_completion_ref=completed,worker_qualification_ref=container_worker_ref,
            worker_pid=os.getpid(),process_group=os.getpgrp(),final_own_scope_check=final_scope)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous)
        if checker is not None:checker.memo.uninstall()

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute",action="store_true")
    p.add_argument("--input")
    p.add_argument("--expected-input-sha256")
    a=p.parse_args(argv)
    if not a.execute:
        print(json.dumps(dict(status="source_only_inactive",hypothesis_id="H056",device="cpu")))
        return 0
    require(a.input and a.expected_input_sha256,"Exact root input and externally supplied digest")
    print(json.dumps(run_v1(dict(path=a.input,sha256=a.expected_input_sha256)),sort_keys=True,allow_nan=False),flush=True)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
