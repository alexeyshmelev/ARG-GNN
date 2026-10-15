"""Independent frozen successor review. Only source/metadata reads and mocked helpers."""
from pathlib import Path
from types import SimpleNamespace as NS
from datetime import datetime, timezone
import ast, copy, hashlib, json

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
BASE=ROOT/"research/reviews/American_C3_native_MLM_amendment_v9"
HERE=BASE/"native_v9_independent_review"
checks=[]
def check(name,value):
    if not value:raise AssertionError(name)
    checks.append(name)
def expect(name,kind,call):
    try:call()
    except kind:checks.append(name);return
    raise AssertionError(name+" did not reject")
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fn(src,name):return next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name==name)
def part(src,name):return ast.get_source_segment(src,fn(src,name))
def take(src,names,env):
    nodes=[n for n in ast.parse(src).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),"<independent-source-extraction>","exec"),env)
    return env
def require(ok,msg):
    if not ok:raise ValueError(msg)

handoff=read(BASE/"source_handoff.json");closure=read(BASE/"execution_source_closure.json")
check("frozen successor handoff exact",sha(BASE/"source_handoff.json")=="a89af75d833a8b6cb4c7b805017c5693b07fe33b2b0b16b4750c3ae5fc0a1d9b")
check("frozen successor closure exact",sha(BASE/"execution_source_closure.json")=="61c36c6b7fba57526ceb135a7cfe3222849e17ee69fd0f50210beede89ba4a43")
check("repository instructions unchanged",sha(ROOT/"AGENTS.md")=="7a6cd4107dd419df0c6e50baaac192fbbd04563e651f415f04ef5ad93176895b")
check("all48 frozen execution sources exact",len(closure["source_sha256"])==48 and all(sha(ROOT/p)==h for p,h in closure["source_sha256"].items()))
check("handoff closure source maps identical",handoff["source_sha256"]==closure["source_sha256"])
oldinput=read(ROOT/"research/reviews/American_C3_native_MLM_full_fit_v8/admitted_mlm_v1/input.json")
check("all31 original source pins preserved exact",len(oldinput["source_sha256"])==31 and all(closure["source_sha256"].get(p)==h for p,h in oldinput["source_sha256"].items()))
check("17 actual successor sources correctly counted",len(set(closure["source_sha256"])-set(oldinput["source_sha256"]))==17)
preserved=[]
for directory,expected,count in [
    ("pre_current_writer_fix_v1","1454ddf963af43386b9e8f5a821195d547e8a4235b21b2397a1838ef9407b6b1",29),
    ("pre_self_fd_fix_v2","91eda86e9e1f7c7f18011c583892a97596eb1fe0d6175fb5cf4b3c7a0b31e660",32)]:
    p=BASE/directory/"preservation_manifest.json";m=read(p)
    check(directory+" manifest exact",sha(p)==expected and len(m["files"])==count)
    check(directory+" all preserved bytes exact",all(sha(ROOT/x["preserved_path"])==x["sha256"] and (ROOT/x["preserved_path"]).stat().st_size==x["bytes"] for x in m["files"]))
    preserved.append({"path":str(p.relative_to(ROOT)),"sha256":sha(p),"files":count})
v2=read(BASE/"pre_self_fd_fix_v2/execution_source_closure.json")
changed=[p for p,h in closure["source_sha256"].items() if v2["source_sha256"].get(p)!=h]
check("only descriptor helper production source changed",changed==[str((BASE/"current_writer_proof_v9.py").relative_to(ROOT))])
src=(BASE/"current_writer_proof_v9.py").read_text()
oldsrc=(BASE/"pre_self_fd_fix_v2/current_writer_proof_v9.py").read_text()
newtree=ast.parse(src);oldtree=ast.parse(oldsrc)
def without_fd(tree):
    return ast.dump(ast.Module(body=[n for n in tree.body if not isinstance(n,ast.FunctionDef) or n.name!="_fd_targets"],type_ignores=[]),include_attributes=False)
check("only fd target function AST changed",without_fd(newtree)==without_fd(oldtree))
check("exact reviewed helper hash",sha(BASE/"current_writer_proof_v9.py")=="8c96e50a4f6566cf5b6321b3c5f549f9d0b19ec079de56261241d0f9af3998f3")

# No real /proc or os/process call: the extracted helper receives only these fakes.
class FakePath:
    def __init__(self,path,state):self.path=path;self.state=state
    def __truediv__(self,name):return FakePath(self.path+"/"+str(name),self.state)
    @property
    def name(self):return self.path.rsplit("/",1)[-1]
    def __str__(self):return self.path
    def iterdir(self):
        # listdir-shaped old inspection fd closes before the returned names are used.
        self.state["open"]=True
        names=[FakePath(self.path+"/"+name,self.state) for name in self.state["names"]]
        self.state["open"]=False
        return iter(names)
class Entries:
    def __init__(self,path,state):self.path=path;self.state=state
    def __enter__(self):self.state["open"]=True;return self
    def __iter__(self):
        for name in self.state["names"]:yield NS(name=name,path=str(self.path)+"/"+name)
    def __exit__(self,*args):self.state["open"]=False;self.state["closed"]+=1
def fixture(text,pid=123):
    state={"open":False,"closed":0,"names":["0","1","99"],"missing":None,"denied":None,"read":[]}
    def target(path):
        name=str(path).rsplit("/",1)[-1];state["read"].append((name,state["open"]))
        if name==state["denied"]:raise PermissionError("mock permission denied")
        if name==state["missing"] or (name=="99" and not state["open"]):raise FileNotFoundError("mock gone descriptor")
        return "/mock-proc/"+str(pid)+"/fd" if name=="99" else "/unrelated/"+name
    def time_limit(deadline):require(deadline>0,"mock deadline exceeded")
    env={"PROC":FakePath("/mock-proc",state),"os":NS(scandir=lambda path:Entries(path,state),readlink=target),
        "_time_limit":time_limit,"require":require,"MAX_FDS":3}
    take(text,{"Changed","_fd_targets"},env)
    return env,state
old,s=fixture(oldsrc)
expect("mock reproduces blocked listdir self-fd Changed",old["Changed"],lambda:old["_fd_targets"](123,10))
new,s=fixture(src)
first=new["_fd_targets"](123,10);second=new["_fd_targets"](123,10)
check("new live-iterator result includes own inspection descriptor",first["99"]=="/mock-proc/123/fd" and len(first)==3)
check("repeated mock self metadata stable",first==second)
check("all mock readlinks occur before iterator closes",all(active for _,active in s["read"]) and not s["open"] and s["closed"]==2)
for pid in (123,456):
    env,state=fixture(src,pid);state["missing"]="1"
    expect("missing descriptor still Changed for mock pid "+str(pid),env["Changed"],lambda:env["_fd_targets"](pid,10))
    check("iterator closes on missing descriptor "+str(pid),not state["open"] and state["closed"]==1)
env,state=fixture(src);state["names"]=["0","1","2","99"]
expect("fd cap rejects extra descriptor",ValueError,lambda:env["_fd_targets"](123,10))
check("fd cap fail closes iterator",not state["open"] and state["closed"]==1)
env,state=fixture(src);state["names"]=[]
check("empty fd table accepted without fabrication",env["_fd_targets"](123,10)=={})
env,state=fixture(src);state["denied"]="0"
expect("permission failure remains fail closed",PermissionError,lambda:env["_fd_targets"](123,10))
check("permission failure closes iterator",not state["open"] and state["closed"]==1)
env,state=fixture(src)
expect("deadline remains fail closed",ValueError,lambda:env["_fd_targets"](123,0))
check("deadline failure closes iterator",not state["open"] and state["closed"]==1)
check("exact max descriptor count accepted",len(fixture(src)[0]["_fd_targets"](123,10))==3)

# Check current proof output/order by replacing all observation operations.
roots=[{"pid":p,"absent":True,"method":"targeted os.kill(pid,0) ESRCH"} for p in (2817234,2817625,2817931,2818200)]
groups=[{"group":x["pid"],"absent":True,"method":"targeted os.killpg(group,0) ESRCH"} for x in roots]
scan={"complete":True,"stable_same_uid_membership":True,"current_matching_processes":[],"unrelated_process_contents_recorded":False}
events=[]
def fake_roots():events.append("roots_groups");return copy.deepcopy((roots,groups))
def fake_scan():events.append("scan");return copy.deepcopy(scan)
env={"observe_known_roots_and_groups":fake_roots,"scan_current_same_uid":fake_scan,
    "require":require,"datetime":datetime,"timezone":timezone}
take(src,{"prove_current_absence"},env)
proof=env["prove_current_absence"]()
check("proof repeats known roots groups before and after scan",events==["roots_groups","scan","roots_groups"])
check("proof explicitly retains historical incompleteness and UNKNOWN result",proof["historical_ids_complete"] is False and proof["historical_descendant_ids_lost"] is True and proof["native_wait_exit"]=="UNKNOWN" and proof["actual_optimizer_updates"]=="UNKNOWN")
check("proof discloses limited observation and cooperative revocation",proof["external_NFS_writers_outside_observation"] is True and proof["other_UID_and_remote_host_writers_outside_observation"] is True and proof["future_relaunch_prevented_by_cooperative_root_revocation"] is True)
scan["current_matching_processes"]=[{"pid":456,"match_reasons":["exact_old_scope_path"]}]
expect("nonempty exact current writer match blocks absence",ValueError,lambda:env["prove_current_absence"]())
scan["current_matching_processes"]=[]
cert=read(BASE/"old_writer_absence.inactive.json")
cert.update(active=True,all_v8_future_launch_resume_write_capabilities_cooperatively_revoked=True,
            current_writer_observation=proof,exact_old_kernel_listener_matches=[])
cenv={"require":require,"KEY":cert["key"],"KNOWN_ROOTS":tuple(x["pid"] for x in roots),"KNOWN_GROUPS":tuple(x["group"] for x in groups)}
take(src,{"validate_static_certificate"},cenv)
check("valid mock limited certificate accepted",cenv["validate_static_certificate"](cert) is cert)
for key,value in [("historical_ids_complete",True),("native_wait_exit",0),("actual_optimizer_updates",1)]:
    bad=copy.deepcopy(cert);bad[key]=value
    expect("invented certificate "+key+" rejected",ValueError,lambda bad=bad:cenv["validate_static_certificate"](bad))
bad=copy.deepcopy(cert);bad["current_writer_observation"]["known_old_roots"].pop()
expect("partial known roots rejected",ValueError,lambda:cenv["validate_static_certificate"](bad))
bad=copy.deepcopy(cert);bad["current_writer_observation"]["known_original_groups"].pop()
expect("partial known groups rejected",ValueError,lambda:cenv["validate_static_certificate"](bad))
bad=copy.deepcopy(cert);bad["current_writer_observation"]["current_same_uid_scan"]["stable_same_uid_membership"]=False
expect("unstable mock certificate rejected",ValueError,lambda:cenv["validate_static_certificate"](bad))

# Independent source checks of the unchanged v2 activation/archive/close contracts.
amend=(BASE/"amendment_v9.py").read_text();archive=(BASE/"root_archive_v9.py").read_text();driver=(BASE/"root_native_driver_v1.py").read_text()
activation=part(amend,"activate");validation=part(amend,"validate_authority");closing=part(driver,"close")
check("actual proof twice before activation and lock/token creation",activation.count("preserve_prestart_proof(auth,")==2 and activation.index("preactivation_current_writer")<activation.index("preactivation_locked_current_writer")<activation.index("fd=os.open(LOCK"))
check("live authority validates static certificate without fresh scan","validate_static_certificate" in validation and "prove_current_absence(" not in validation)
a=fn(archive,"preserve_failed_v8")
check("archive exposes no injected absence proof callback",[x.arg for x in a.args.args+a.args.kwonlyargs]==["archive","authority"] and "prove_current_absence()" in archive)
check("archive actual proof before mutation and per rename",archive.index("initial_proof=prove_current_absence()")<archive.index("archive.mkdir()") and archive.index("current_proof=prove_current_absence()")<archive.index("os.rename(root,dest)") and "def actual_current_guard():" in archive)
check("close approval explicitly covers complete current attempt","all_current_v9_attempt_owned_PIDs_and_groups_listed" in closing and 'approval.get("historical_v8_ids_complete") is False' in closing)
check("close proof postdrain and explicit lost historical roster",'current_close_proof=prove_current_absence()' in closing and 'roster_scope="complete actual current v9 attempt plus known v8 roots only"' in closing and "historical_v8_ids_complete=False,historical_v8_descendant_ids_lost=True" in closing)

refs=read(BASE/"original_refs.json")
check("all original metadata references exact",all(sha(ROOT/v["path"])==v["sha256"] for v in refs.values()))
storage=read(ROOT/refs["ORIGINAL_STORAGE_REF"]["path"]);auth=read(BASE/"execution_authority.inactive.json")
check("same14GiB existing reservation job key and scopes",storage["reserved_new_bytes"]==15032385536 and storage["reservation"]["reserved_bytes"]==15032385536 and storage["job_id"]==auth["job_id"]=="American-C3-native-MLM-full30mb-mlm-fit-v8" and storage["reservation"]["key"]==auth["key"]=="H041/American-C3-native-MLM-full30mb-mlm-fit-v8")
check("original absolute creation expiry unchanged",auth["created_at_utc"]=="2026-10-15T11:47:27.217781+00:00" and auth["expires_at_utc"]==storage["expires_at_utc"]=="2026-10-18T11:47:27.217781+00:00")
check("all three authority templates inactive",all(read(BASE/name)["active"] is False for name in ["execution_authority.inactive.json","old_writer_absence.inactive.json","root_archive_authority.inactive.json"]))
check("explicit exclusiveGPU4 after H047 release",handoff["exclusive_scheduling"]=="GPU4 only after H047 actual release; H046 may remain GPU5; no same-device concurrency" and auth["physical_gpu"]==4)
check("science capture normalization lifecycle checkpoint logic unchanged",all((BASE/n).read_bytes()==(BASE/"pre_self_fd_fix_v2"/n).read_bytes() for n in ["native_child_v1.py","native_runtime_v1.py","supervisor_v1.py","terminal_capture_v9.py","terminal_namespace_v9.py","owned_lifecycle_v2.py","terminal_bounds_v1.py","consumer_contract_v1.py"]))
check("execution source closure no hidden pin drift",all(sha(ROOT/p)==h for p,h in closure["source_sha256"].items()))

# Verify every complete metadata reference in the frozen handoff, closure,
# authorities and original-ref table; only metadata/source paths are read.
verified=set()
def refs_exact(value):
    if isinstance(value,dict):
        if set(value)=={"path","sha256"}:
            p=ROOT/value["path"]
            require(sha(p)==value["sha256"],"exact metadata reference: "+str(p))
            verified.add((value["path"],value["sha256"]))
        for child in value.values():refs_exact(child)
    elif isinstance(value,list):
        for child in value:refs_exact(child)
for obj in [handoff,closure,refs,auth,read(BASE/"root_archive_authority.inactive.json"),read(BASE/"old_writer_absence.inactive.json")]:refs_exact(obj)
check("all nested reviewed source and metadata refs exact",len(verified)>20)
result={"schema":"native_v9_independent_successor_checks_v3","status":"PASS","checks_passed":len(checks),"checks":checks,
        "verified_distinct_metadata_references":len(verified),"changed_execution_sources":changed,
        "preservation_manifests":preserved,"frozen_handoff_sha256":sha(BASE/"source_handoff.json"),
        "frozen_closure_sha256":sha(BASE/"execution_source_closure.json"),
        "no_actual_proc_process_kernel_GPU_archive_namespace_guard_or_scientific_operations":True,
        "only_source_AST_extracted_helpers_executed_with_mock_dependencies":True,
        "earlier46_checks_preserved_not_falsely_claimed_as_rerun":True,
        "recorded_utc":datetime.now(timezone.utc).isoformat()}
(HERE/"checks_v3.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
print(json.dumps({"status":"PASS","checks_passed":len(checks),"metadata_refs":len(verified),"checks_sha256":sha(HERE/"checks_v3.json")}))
