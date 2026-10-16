"""Root-only H056 exact-three-map CPU acquisition and inherited regional crop.
Default CLI is inert. Root owns admission, explicit authority and external wait.
"""
from pathlib import Path
from datetime import datetime,timezone
from urllib.request import Request,build_opener,HTTPRedirectHandler,ProxyHandler
import argparse,hashlib,importlib.util,json,os,re,resource,signal,stat,sys,time
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/H056_map_acquisition_root_source_20261016_v1"
SOURCE=HERE+"/runtime_v1.py"
CHARGED_CAP=32*2**20
SCIENCE_CAP=8*2**20
RSS_CAP=2*2**30
WALL_SECONDS=900
NETWORK_SECONDS=120
ROOT_METADATA_CAP=512*1024
JOB="H056-H36-maps-CPU-v1"
OUTPUT="experiments/results/H056_H36_qualified_maps_CPU_v1"
COMMIT="103ca071ea9e4e648192916142e5ed353ac94e27"
NOTICE="Copyright © 2018, University of Oxford"
NAME=re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
_RAN=False
FIXED_REFS={'checker': {'path': 'research/reviews/GPU45_TRAIN_controller_execution_override_20261016_v1/controller_metadata_scope_check_v1.py', 'sha256': '2621353de0dc56f328f7f308ba17a341425ef2fc02d54739b6ccfc7d1a6ce2cf'}, 'guard': {'path': 'research/reviews/prospective_980GB_capacity_source_v1/prospective_980gb_v1.py', 'sha256': 'fc71030865e0a3d5da31120d4daa2caa0b2d39ec7dd212b8c46c476b5a866d56'}, 'memo': {'path': 'research/reviews/resource_guard_hotpath_source_audit_v1/transaction_memo_adapter_v2.py', 'sha256': 'c972e9a779d424ceafdb258848614bf40994d1004e2c73fc5cd225ae99621364'}, 'metadata': {'path': 'research/reviews/H056_map_source_metadata_20261016_v1/source_metadata.root_v1.json', 'sha256': '0745c2a210d855607abb710da840b1b513d53c96adfde7a67d6f4c75700a767e'}, 'original_qualifier': {'path': 'research/reviews/H018_genealogy_next_source_v1/qualification_source.py', 'sha256': '8ca31c108ebc99b2eb1cd363e3ebb29e3173697521076edf0ef4c7c4304f82b0'}, 'original_runtime': {'path': 'research/reviews/H055_CPU_runtime_source_20261016_v1/runtime_v1.py', 'sha256': '00ca41a2f5b4efb0a9d1b3c6812352c03a9f14c6dd6b4b76f6d88c738007a565'}, 'policy': {'path': 'research/reviews/prospective_980GB_capacity_source_v1/activation_policy.root_active_v1.json', 'sha256': 'eb3408004a508136aa434c0a2e02a9d14a5598a96b8483c264c93026f9326018'}, 'protocol': {'path': 'research/reviews/H056_frozen_H045_Danish_application_registration_20261016_v1/protocol.root_frozen_v1.json', 'sha256': '9544bdbd499bddafeb0774c67e4e86ef25535d41dd2409e710c29b679619cd6e'}, 'renewal': {'path': 'research/reviews/H039_marker_repair_renewal_source_20261016_v1/marker_repair_renewal_v1.py', 'sha256': '491236877e8d71cae8bd7619880858ac501c57dd93e59479ca32dacacab20468'}, 'renewal_certificate': {'path': 'research/reviews/H039_marker_repair_renewal_source_20261016_v1/renewal_certificate.root_active_v1.json', 'sha256': '66006202094d7a3ef2b29cbec28eb93f334505b72589f09d86e4b57498162db7'}, 'repair_authority': {'path': 'research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json', 'sha256': 'f6f2d628199e8be73db8b5d90d428570d8afdefeab9dc5cd94bb5eafb39247db'}, 'science': {'path': 'research/reviews/H056_map_acquisition_root_source_20261016_v1/map_science_v1.py', 'sha256': '81bc1dc09378ca2ee38a9c1faab14e5782124563b67bec5c879ef565a9729de2'}, 'semantic_evidence_0': {'path': 'research/reviews/H018_map_coordinate_basis_source_20261015_v1/genotype_reader.cpp', 'sha256': '02ee2237a35b5495a29a8f2c345aa488bac13275ee418381c1647078fb7c56dd'}, 'semantic_evidence_1': {'path': 'research/reviews/H018_map_coordinate_basis_source_20261015_v1/variant.cpp', 'sha256': '72d88a596fc5739ec73ce79cbe6bb6fd8c536f4ad19eff39507df976402e5fee'}, 'semantic_evidence_2': {'path': 'research/reviews/H018_map_acquisition_root_source_v1/acquired_v1/gmap_reader.cpp', 'sha256': '95a3032f219fc997d40f766e70ae5c356f4692f6c8454528bdfbee58626b0c63'}, 'semantic_evidence_3': {'path': 'research/reviews/H018_map_acquisition_root_source_v1/acquired_v1/variant_map.cpp', 'sha256': '544e35c9e76a2fb2885f71b494b73b222d725c296963840cbd0284ddf3516a91'}, 'semantic_evidence_4': {'path': '.runtime/author_sources/public_realdata_metadata_join_v1/GLIMPSE_v1_tree.json', 'sha256': 'b27b3729108b32e5be99d43f26dfa7e6486a3ebf237bb58b92346771d3b4e5f7'}, 'semantic_evidence_5': {'path': 'research/reviews/H018_map_coordinate_basis_source_20261015_v1/coordinate_evidence.json', 'sha256': 'cdb9bcbf5c2df6d3aaaa3fbdc4d2e31c271ffc2553ab5f2cd7a6a47ef2dc3991'}, 'semantic_evidence_6': {'path': 'research/reviews/H018_map_coordinate_basis_source_20261015_v1/retained_map_source_inspection.json', 'sha256': '51132703233c6b98e31f2f2c3bdcdc4d83108423db42d8ee146b39a92c7a3cb6'}, 'v4': {'path': 'src/arggnn/project_budget_v4.py', 'sha256': 'a093c7d86f57fee569cb99f4712250e9505a1119f3950fa44282c7ba8c57df47'}}
ASSETS=[{'URL': 'https://raw.githubusercontent.com/odelaneau/GLIMPSE/103ca071ea9e4e648192916142e5ed353ac94e27/maps/genetic_maps.b37/chr20.b37.gmap.gz', 'bytes': 583035, 'chromosome': '20', 'git_blob_sha1': '2d28bfb7f3825930f77802d9513e96cd5b5ba8c4', 'source_path': 'maps/genetic_maps.b37/chr20.b37.gmap.gz'}, {'URL': 'https://raw.githubusercontent.com/odelaneau/GLIMPSE/103ca071ea9e4e648192916142e5ed353ac94e27/maps/genetic_maps.b37/chr21.b37.gmap.gz', 'bytes': 317881, 'chromosome': '21', 'git_blob_sha1': '7166a765adafd891b35859241c45f4f931b8d25e', 'source_path': 'maps/genetic_maps.b37/chr21.b37.gmap.gz'}, {'URL': 'https://raw.githubusercontent.com/odelaneau/GLIMPSE/103ca071ea9e4e648192916142e5ed353ac94e27/maps/genetic_maps.b37/chr22.b37.gmap.gz', 'bytes': 322186, 'chromosome': '22', 'git_blob_sha1': 'eec8ceff2fdff0d70badcb09c18e5c6cc5b04854', 'source_path': 'maps/genetic_maps.b37/chr22.b37.gmap.gz'}]
def require(ok, message):
    if not ok:
        raise ValueError(message)

def path(name):
    require(isinstance(name, str) and name and not Path(name).is_absolute()
            and ".." not in Path(name).parts, "Relative contained project pathname")
    p = ROOT / name
    require(p != ROOT and p.resolve() == p and not p.is_symlink(), "Canonical non-symlink project path")
    return p

def exact_ref(ref):
    require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}
            and isinstance(ref["sha256"], str) and re.fullmatch("[0-9a-f]{64}", ref["sha256"]),
            "Exact relative path/SHA256 reference")
    path(ref["path"])
    return ref

def raw(ref, limit=32 * 2**20):
    exact_ref(ref)
    p = path(ref["path"])
    before = p.lstat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_nlink, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size <= limit,
            "Bounded singly linked metadata/source")
    with os.fdopen(os.open(p, os.O_RDONLY | os.O_NOFOLLOW), "rb") as f:
        require(identity(os.fstat(f.fileno())) == identity(before), "Unchanged bound open identity")
        data = f.read(limit + 1)
        require(identity(os.fstat(f.fileno())) == identity(before), "Unchanged bound read identity")
    require(len(data) <= limit and identity(p.lstat()) == identity(before)
            and hashlib.sha256(data).hexdigest() == ref["sha256"], "Exact unchanged bound bytes")
    return data

def read(ref, limit=32 * 2**20):
    def pairs(items):
        result = {}
        for k, v in items:
            require(k not in result, "No duplicate JSON keys")
            result[k] = v
        return result
    def invalid(value):
        raise ValueError("No nonfinite JSON constant: " + value)
    return json.loads(raw(ref, limit), object_pairs_hook=pairs, parse_constant=invalid)

def source_ref():
    require(Path(__file__).resolve() == path(SOURCE), "Actual runtime source path")
    return {"path": SOURCE, "sha256": hashlib.sha256(path(SOURCE).read_bytes()).hexdigest()}

def unique(refs):
    result = {}
    for ref in refs:
        exact_ref(ref)
        require(ref["path"] not in result or result[ref["path"]] == ref["sha256"], "No conflicting source pins")
        result[ref["path"]] = ref["sha256"]
    return [{"path": p, "sha256": h} for p, h in sorted(result.items())]

def load(ref, label):
    data = raw(ref)
    spec = importlib.util.spec_from_file_location(label, path(ref["path"]))
    module = importlib.util.module_from_spec(spec)
    sys.modules[label] = module
    # Execute the authenticated source bytes, without producer/guard pyc output.
    exec(compile(data, str(path(ref["path"])), "exec"), module.__dict__)
    return module

class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        raise ValueError("Fixed immutable URLs forbid redirects, fallback and mirrors")


def encode(value):
    return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()


def run_v1(input_ref):
    global _RAN
    require(not _RAN and sys.platform.startswith("linux"),"One fresh Linux CPU caller")
    _RAN=True
    started=time.monotonic()
    os.environ.update(CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
                      MKL_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1")
    sys.dont_write_bytecode=True
    require(not any(n in sys.modules for n in ("torch","torch_geometric","tensorflow","jax","cupy")),
            "No model/GPU imports in map caller")
    available=os.sched_getaffinity(0)
    require(available,"One available CPU")
    os.sched_setaffinity(0,{min(available)})
    soft,hard=resource.getrlimit(resource.RLIMIT_AS)
    ceiling=min([RSS_CAP]+[x for x in (soft,hard) if x!=resource.RLIM_INFINITY])
    resource.setrlimit(resource.RLIMIT_AS,(ceiling,hard))
    spec=read(input_ref,65536)
    require(set(spec)=={"schema","status","root_authorizer","hypothesis_id","registered","execution_authorized",
                       "job_id","output_directory","created_at_utc","expires_at_utc","runtime_entry_ref","refs"}
            and spec["schema"]=="H056_map_root_input_v1" and spec["status"]=="active"
            and spec["root_authorizer"]=="root_agent" and spec["hypothesis_id"]=="H056"
            and spec["registered"] is True and spec["execution_authorized"] is True
            and spec["job_id"]==JOB and spec["output_directory"]==OUTPUT
            and spec["runtime_entry_ref"]==source_ref()
            and set(spec["refs"])=={"config","runtime","storage","source_closure"},
            "Exact explicit root map job and actual runtime binding")
    issued=datetime.fromisoformat(spec["created_at_utc"].replace("Z","+00:00"))
    expires=datetime.fromisoformat(spec["expires_at_utc"].replace("Z","+00:00"))
    require(issued.tzinfo is not None and expires.tzinfo is not None
            and issued<=datetime.now(timezone.utc)<expires
            and 0<(expires-issued).total_seconds()<=WALL_SECONDS,"Current at-most900second authority")
    end=expires.timestamp()
    out=path(OUTPUT)
    require(out.parent.is_dir() and not out.exists(),"Fresh single-use scientific output")
    folder=path(spec["refs"]["config"]["path"]).parent
    require(folder!=out and folder not in out.parents and out not in folder.parents
            and all(path(r["path"]).parent==folder for r in spec["refs"].values()),
            "Separate exact root metadata folder")
    for ref in FIXED_REFS.values():raw(ref)
    meta=read(FIXED_REFS["metadata"])
    require(meta["assets"]==ASSETS and meta["source_commit"]==COMMIT
            and meta["genome_build"]=="GRCh37"
            and meta["nominal_region_absolute_bp0"]==[20000000,30000000],
            "Exact source metadata and fixed intervals")
    cfg=read(spec["refs"]["config"])
    semantic_refs=cfg.get("root_semantic_refs")
    expected=dict(schema="H056_map_CPU_config_v1",root_authorizer="root_agent",hypothesis_id="H056",
                  fixed_refs=FIXED_REFS,assets=ASSETS,root_semantic_refs=semantic_refs,device="cpu",
                  physical_gpu=None,cuda_visible_devices="",charged_scope_cap_bytes=CHARGED_CAP,
                  scientific_output_cap_bytes=SCIENCE_CAP,max_cpu_rss_bytes=RSS_CAP,
                  global_wall_seconds=WALL_SECONDS,network_seconds=NETWORK_SECONDS,network_retries=0,
                  network_body_cap_bytes=sum(a["bytes"]+1 for a in ASSETS),
                  source_expanded_per_map_cap_bytes=8000000,map_data_RAM_planning_bytes=128*2**20,
                  map_data_RAM_forecast_measured=False,auxiliary_scopes=[],
                  original_scope_checker_reuse_root_authorized=True)
    require(cfg==expected and isinstance(semantic_refs,dict) and set(semantic_refs)=={"20","21","22"},
            "Exact three maps and CPU/network/RAM/storage caps")
    semantics={}
    evidence=[v for k,v in FIXED_REFS.items() if k.startswith("semantic_evidence_")]
    for c,r in semantic_refs.items():
        s=read(r)
        require(s.get("schema")=="H056_map_source_semantics_v1"
                and s.get("status")=="root_reviewed_source_semantics" and s.get("root_authorizer")=="root_agent"
                and s.get("chromosome")==c and s.get("assembly")=="GRCh37"
                and s.get("physical_units")=="bp" and s.get("cumulative_distance_units")=="cM"
                and s.get("position_basis")=="absolute_bp1" and s.get("source_format")=="pos_chr_cM"
                and s.get("exact_header_tokens")==["pos","chr","cM"]
                and s.get("semantic_evidence_refs")==evidence,
                "Separate root-reviewed original consumer semantics for exact chromosome")
        semantics[c]=s
    runtime={k:spec[k] for k in ("job_id","output_directory","created_at_utc","expires_at_utc")}
    runtime.update(schema="H056_map_CPU_runtime_v1",root_authorizer="root_agent",device="cpu",
                   physical_gpu=None,cuda_visible_devices="",global_wall_seconds=WALL_SECONDS)
    require(read(spec["refs"]["runtime"])==runtime,"Exact original runtime/output/expiry")
    closure=read(spec["refs"]["source_closure"])
    bound=dict(FIXED_REFS,runtime_entry=spec["runtime_entry_ref"],config=spec["refs"]["config"],
               runtime=spec["refs"]["runtime"],**{"semantics_"+c:r for c,r in semantic_refs.items()})
    require(set(closure)=={"schema","root_authorizer","bound_refs","original_bindings"}
            and closure["schema"]=="H056_map_CPU_source_closure_v1" and closure["root_authorizer"]=="root_agent"
            and closure["bound_refs"]==bound and isinstance(closure["original_bindings"],dict)
            and closure["original_bindings"],"Exact original guard/source closure")
    direct=unique([*closure["original_bindings"].values(),
                   *(r for r in bound.values() if Path(r["path"]).suffix in {".py",".sh",".json"}),
                   spec["refs"]["source_closure"]])
    require(len(direct)<=64,"Unchanged original64 direct bindings")
    storage=read(spec["refs"]["storage"])
    own=storage["reservation"]
    require(storage.get("kind")=="project_budget_v9_storage_record" and storage.get("active") is True
            and storage.get("consumer_uses_v9_guard") is True
            and storage.get("accounting_mode")=="prospective_980gb_v1"
            and storage["job_id"]==JOB and storage["output_directory"]==OUTPUT
            and storage["expires_at_utc"]==spec["expires_at_utc"]
            and storage["reserved_new_bytes"]==own["reserved_bytes"]==CHARGED_CAP
            and own["output"]==str(out) and own["auxiliary_scopes"]==[]
            and own["key"]=="H056/H36_qualified_maps_CPU_v1"
            and unique(own["consumer_bindings"])==direct
            and storage["completed_overlay_policy_ref"]==FIXED_REFS["policy"],
            "Actual fully charged32MiB original reservation, no auxiliary scopes")
    fs=os.statvfs(out.parent)
    unit=max(4096,fs.f_frsize,fs.f_bsize)
    # 3 raw maps +3 crops +3 semantics +3 quals +NOTICE +acquisition +terminal,
    # one directory, plus conservative extra inode/block slack.
    allocation=26*unit
    require(unit<=32768 and SCIENCE_CAP+ROOT_METADATA_CAP+allocation<=CHARGED_CAP
            and CHARGED_CAP<=fs.f_bavail*fs.f_frsize,"Block/directory slack fits32MiB")
    checker=None
    previous=signal.getsignal(signal.SIGALRM)
    require(signal.getitimer(signal.ITIMER_REAL)==(0.0,0.0),"Fresh wall timer")
    def timeout(signum,frame):raise TimeoutError("H056 map wall/network/absolute expiry")
    signal.signal(signal.SIGALRM,timeout)
    def remaining():return min(WALL_SECONDS-(time.monotonic()-started),end-time.time())
    def check_runtime():
        require(remaining()>0 and os.environ.get("CUDA_VISIBLE_DEVICES")==""
                and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=RSS_CAP,
                "Current CPU-only root wall and RSS caps")
    def arm(seconds=None):
        t=remaining()
        if seconds is not None:t=min(t,seconds)
        require(t>0,"Positive timer boundary")
        signal.setitimer(signal.ITIMER_REAL,t)
    arm()
    try:
        sys.path[:0]=[str(ROOT),str(ROOT/"src")]
        renewal=load(FIXED_REFS["renewal"],"_H056_map_original_renewal")
        renewal.install_renewal_v1(FIXED_REFS["renewal_certificate"])
        cert,_=renewal.validate_certificate_v1(FIXED_REFS["renewal_certificate"])
        require(end<datetime.fromisoformat(cert["renewed_expires_at_utc"].replace("Z","+00:00")).timestamp(),
                "Map expiry within original marker renewal")
        guard=load(FIXED_REFS["guard"],"_H056_map_original_980_guard")
        guard.configure_v2(FIXED_REFS["repair_authority"],policy_ref=FIXED_REFS["policy"])
        cm=load(FIXED_REFS["checker"],"_H056_map_original_scope_checker")
        checker=cm.ControllerMetadataScopeV1(guard,storage)
        require(checker.bindings==closure["original_bindings"],"All actual original state bindings unchanged")
        checker.check(required=CHARGED_CAP)
        check_runtime()
        require(not out.exists(),"Still fresh before exact single-use claim")
        claim=encode(dict(schema="H056_map_single_attempt_claim_v1",input_ref=input_ref,
            job_id=JOB,worker_PID=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),
            network_retries_authorized=False,no_internal_OS_exit_or_wait0_claim=True))
        require(len(claim)<=65536,"Compact root single-use claim")
        out.mkdir()
        with os.fdopen(os.open(out/"run_claim.json",os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),"wb") as f:
            f.write(claim)
            f.flush()
            os.fsync(f.fileno())
        claim_ref=dict(path=OUTPUT+"/run_claim.json",sha256=hashlib.sha256(claim).hexdigest())
        directory_fd=os.open(out,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(directory_fd)
        finally:os.close(directory_fd)
        science=load(FIXED_REFS["science"],"_H056_exact_map_science")
        opener=build_opener(ProxyHandler({}),NoRedirect())
        payloads={}
        requests=[]
        qualifications={}
        error=None
        total=0
        network_start=time.monotonic()
        try:
            for asset in ASSETS:
                c=asset["chromosome"]
                require(asset["URL"]=="https://raw.githubusercontent.com/odelaneau/GLIMPSE/"+COMMIT+"/"+asset["source_path"],
                        "Fixed commit URL")
                left=NETWORK_SECONDS-(time.monotonic()-network_start)
                arm(left)
                row=dict(chromosome=c,url=asset["URL"],expected_bytes=asset["bytes"],response_body_bytes=0,
                         started_utc=datetime.now(timezone.utc).isoformat(),verified=False)
                requests.append(row)
                req=Request(asset["URL"],headers={"User-Agent":"ARG-GNN-H056-map-source/1","Accept-Encoding":"identity"})
                with opener.open(req,timeout=min(30,left)) as response:
                    require(response.status==200 and response.url==asset["URL"],"Exact HTTP200 fixed URL")
                    require(response.headers.get("Content-Encoding") in (None,"identity"),"Identity response encoding")
                    require(response.headers.get("Content-Length") in (None,str(asset["bytes"])),"Exact source Content-Length")
                    row.update(status=response.status,final_url=response.url,Content_Length=response.headers.get("Content-Length"),
                               Content_Encoding=response.headers.get("Content-Encoding"),ETag=response.headers.get("ETag"))
                    data=response.read(asset["bytes"]+1)
                total+=len(data)
                row["response_body_bytes"]=len(data)
                payloads["chr"+c+".b37.gmap.gz"]=data
                require(total<=cfg["network_body_cap_bytes"] and len(data)==asset["bytes"],"Exact bounded source map body")
                blob=hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()
                require(blob==asset["git_blob_sha1"],"Exact Git blob source identity")
                regional,qualification=science.qualify_candidate_map_bytes_v1(data,semantics[c],asset)
                def output_ref(name,value):
                    return dict(path=OUTPUT+"/"+name,sha256=hashlib.sha256(value).hexdigest())
                rawname="chr"+c+".b37.gmap.gz"
                mapname="chr"+c+".qualified.regional.map"
                semname="chr"+c+".map_semantics.json"
                semdata=encode(semantics[c])
                qualification.update(schema="H056_map_qualification_v1",root_authorizer="root_agent",chromosome=c,
                    assembly="GRCh37",physical_unit="bp",genetic_unit="cM",
                    raw_map_ref=output_ref(rawname,data),regional_map_ref=output_ref(mapname,regional),
                    source_semantics_ref=output_ref(semname,semdata),root_source_semantics_ref=semantic_refs[c],
                    source_repository="https://github.com/odelaneau/GLIMPSE",source_commit=COMMIT,
                    source_path=asset["source_path"],raw_map_source_asset=asset,
                    original_qualifier_ref=FIXED_REFS["original_qualifier"],
                    H056_science_source_ref=FIXED_REFS["science"])
                payloads[mapname]=regional
                payloads[semname]=semdata
                payloads["chr"+c+".map_qualification.json"]=encode(qualification)
                qualifications[c]=qualification
                row.update(verified=True,sha256=hashlib.sha256(data).hexdigest(),git_blob_sha1=blob,
                           completed_utc=datetime.now(timezone.utc).isoformat())
                require(sum(map(len,payloads.values()))<=SCIENCE_CAP,"All scientific payloads fit8MiB")
            require(total==sum(a["bytes"] for a in ASSETS) and len(qualifications)==3,"Exactly3 maps acquired and qualified")
        except Exception as exc:
            error=dict(type=type(exc).__name__,message=str(exc)[:2048])
        finally:
            arm()
        check_runtime()
        payloads["NOTICE.txt"]=(NOTICE+"\nOriginal GLIMPSE source commit "+COMMIT+
            "\nSource maps retained unchanged. Regional maps preserve the original piecewise-linear curve.\n").encode()
        payloads["acquisition.json"]=encode(dict(schema="H056_map_acquisition_v1",requests=requests,body_bytes=total,
            expected_body_bytes=sum(a["bytes"] for a in ASSETS),network_retries=0,network_seconds=NETWORK_SECONDS,
            elapsed_network_and_qualification_seconds=time.monotonic()-network_start,
            input_ref=input_ref,source_metadata_ref=FIXED_REFS["metadata"],error=error,
            all_source_qualifications_passed=error is None,sites_and_haps_map_join_qualified=False,
            no_genotype_tree_model_or_ancestry_truth_reads=True,no_child_processes=True,CPU_only=True))
        terminal_name="completed.json" if error is None else "failed.json"
        payloads[terminal_name]=encode(dict(schema="H056_map_terminal_v1",status="qualified_exact3_maps" if error is None else "failed_preserved",
            job_id=JOB,input_ref=input_ref,run_claim_ref=claim_ref,storage_ref=spec["refs"]["storage"],runtime_entry_ref=spec["runtime_entry_ref"],
            source_closure_ref=spec["refs"]["source_closure"],error=error,output_cap_bytes=CHARGED_CAP,
            asset_refs={n:dict(path=OUTPUT+"/"+n,sha256=hashlib.sha256(b).hexdigest(),bytes=len(b)) for n,b in payloads.items()},
            no_internal_OS_exit_or_wait0_claim=True,reservation_remains_fully_charged=True,
            peak_self_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            elapsed_seconds=time.monotonic()-started,sites_and_haps_map_join_qualified=False))
        batch=sum(map(len,payloads.values()))
        require(batch<=SCIENCE_CAP and batch+len(claim)+allocation<=CHARGED_CAP,"Complete durable batch fits original32MiB cap")
        checker.check(required=batch+allocation)
        check_runtime()
        require(out.is_dir() and out.resolve()==out and raw(claim_ref,65536)==claim,
                "Exact current single-use output claim, no replay or replacement")
        for name,data in payloads.items():
            with os.fdopen(os.open(out/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),"wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
        fd=os.open(out,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
        final_scope=checker.check(required=0)
        check_runtime()
        result=dict(status="completed_H056_maps" if error is None else "failed_H056_maps_preserved",
                    terminal_ref=dict(path=OUTPUT+"/"+terminal_name,sha256=hashlib.sha256(payloads[terminal_name]).hexdigest()),
                    qualified_chromosomes=list(qualifications),retained_serialized_bytes=batch,
                    final_own_scope_check=final_scope,no_internal_OS_exit_or_wait0_claim=True)
        print(json.dumps(result,sort_keys=True,allow_nan=False),flush=True)
        return 0 if error is None else 1
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous)
        if checker is not None:checker.memo.uninstall()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute",action="store_true")
    p.add_argument("--input")
    p.add_argument("--expected-input-sha256")
    args=p.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="source_only_inactive",hypothesis_id="H056",CPU_only=True)))
        return 0
    require(args.input and args.expected_input_sha256,"Explicit root input and externally supplied exact SHA256")
    return run_v1(dict(path=args.input,sha256=args.expected_input_sha256))


if __name__=="__main__":
    raise SystemExit(main())
