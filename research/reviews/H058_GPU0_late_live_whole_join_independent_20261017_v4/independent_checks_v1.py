"""Independent whole-join source review: source metadata + isolated AST/RAM only."""
from pathlib import Path, PurePosixPath
from datetime import datetime,timezone,timedelta
from types import SimpleNamespace
import ast,copy,errno,hashlib,json,stat
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
D="research/reviews/H058_GPU0_full_training_successor_source_20261017_v4"
B="research/reviews/H058_GPU0_full_training_successor_source_20261017_v3"
OUT="research/reviews/H058_GPU0_late_live_whole_join_independent_20261017_v4"
NAMES=("__init__.py","common_v1.py","context_factory_v1.py","direct_gpu0_v1.py","observations_gpu_v1.py","producer_v1.py","root_driver_v1.py")
checks=[]
def ck(name,ok=True):
    if not ok:raise AssertionError(name)
    checks.append(dict(name=name,status="PASS"))
def reject(name,call,kind=Exception,message=None):
    try:call()
    except kind as e:
        if message is not None:assert message in str(e),(name,type(e),str(e))
        ck(name)
    else:raise AssertionError("Accepted forbidden case: "+name)
def data(p):
    p=ROOT/p
    assert p.resolve()==p and p.is_file() and not p.is_symlink() and p.stat().st_size<=32*2**20, str(p)
    assert p.suffix not in (".pt",".npz",".npy",".trees",".vcf",".gz")
    return p.read_bytes()
def text(p):return data(p).decode()
def j(p):return json.loads(data(p))
def ref(p):return dict(path=p,sha256=hashlib.sha256(data(p)).hexdigest())
def fn(t,n):return next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name==n)
def dump(x):return ast.dump(x,include_attributes=False)
def norm(x):
    return dump(x).replace("H058_GPU0_full_training_successor_source_20261017_v4","H058_GPU0_full_training_successor_source_20261017_v3").replace("H058_GPU0_v4_","H058_GPU0_v3_").replace("H058-GPU0-v4-","H058-GPU0-v3-")
def emit(nodes,ns,label):
    z=ast.Module(body=copy.deepcopy(nodes),type_ignores=[]);ast.fix_missing_locations(z)
    exec(compile(z,"<independent_RAM_"+label+">","exec"),ns)
def allrefs(x):
    if isinstance(x,dict):
        if set(x)=={"path","sha256"}:yield x
        else:
            for v in x.values():yield from allrefs(v)
    elif isinstance(x,list):
        for v in x:yield from allrefs(v)
expected={D+"/source_handoff.v4.json":"81e22ab3069877c37b15b92cd79f5f198dde62547d5980147d8477c9a11f328f",
D+"/source_manifest.v1.json":"dfd5d28cec6cdd00828e6d59a34ac080e6e3698db58db9db896ed7f65e9e7996",
D+"/root_driver_v1.py":"695499bd2bb3077808dd979012ab35e1f04e7f0a667cd209dd108e0ce90c82a8",
"research/reviews/H058_GPU0_direct_resource_gate_independent_20261017_v3/review.v3.json":"9657bad42b643ae5a74c7415cd3dcd33c2ca8d4ecf0739f446709f5c9136cb1b",
"research/reviews/H058_GPU0_v3_ROOT_terminal_20261017_v1/prior_CPU_and_GPU_disposition_for_root02.root_active_v1.json":"e0073925916dfe933fbc75b66bce4f00901ff080b18d31becf10be238a2ba821"}
for p,s in expected.items():ck("exact_assigned_seal:"+p,ref(p)["sha256"]==s)
agg=j(D+"/source_manifest.v1.json");baseagg=j(B+"/source_manifest.v1.json")
ck("aggregate_271_distinct",len(agg["refs"])==len({r["path"] for r in agg["refs"]})==271)
ck("aggregate_exact_map",agg["source_sha256"]=={r["path"]:r["sha256"] for r in agg["refs"]})
for r in agg["refs"]:ck("aggregate_ref:"+r["path"],ref(r["path"])==r)
for r in baseagg["refs"]:ck("preserved_v3_aggregate_ref:"+r["path"],r in agg["refs"])
for version in (("v3",B),("v4",D)):
    seal=j(version[1]+"/MANIFEST."+version[0]+".json")
    for r in seal["files"]:
        ck("artifact:"+r["path"],ref(r["path"])["sha256"]==r["sha256"] and len(data(r["path"]))==r["bytes"])
for name in ("source_handoff.v4.json","SOURCE_MAP.v4.json","dependencies.v1.json","producer_handoff.v1.json","producer_manifest.v1.json"):
    for r in allrefs(j(D+"/"+name)):ck("joined:"+name+":"+r["path"],ref(r["path"])==r)
for p in sorted((ROOT/D).glob("*INACTIVE*.json")):
    for r in allrefs(j(str(p.relative_to(ROOT)))):
        if Path(r["path"]).suffix==".pt":
            protocol_tree=ast.parse(text("research/reviews/H058_full_parent_CPU_execution_source_20261017_v1/protocol_v1.py"))
            declared=next(eval(compile(ast.Expression(n.value),"<frozen_source_ref_constant>","eval"),{"__builtins__":{},"dict":dict}) for n in protocol_tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="WARM_CHECKPOINT_REF" for t in n.targets))
            ck("template_checkpoint_ref_equals_frozen_protocol_without_payload_read",r==declared)
        else:ck("template_ref:"+p.name+":"+r["path"],ref(r["path"])==r)
trees={n:ast.parse(text(D+"/"+n)) for n in NAMES}
old={n:ast.parse(text(B+"/"+n)) for n in NAMES}
for n in NAMES:
    compile(trees[n],"<parse_only_"+n+">","exec")
    ck("all7_parse_compile:"+n)
    ck("production_in_aggregate:"+n,ref(D+"/"+n) in agg["refs"])
    if n not in ("common_v1.py","direct_gpu0_v1.py"):ck("whole_AST_namespace_only:"+n,norm(trees[n])==norm(old[n]))
c=trees["common_v1.py"];d=trees["direct_gpu0_v1.py"];driver=trees["root_driver_v1.py"]
# Exact restoration of the intended new request-mode semantics.
z=copy.deepcopy(c);req=fn(z,"validate_request_v1")
ck("new_keyword_only_boolean_mode",len(req.args.kwonlyargs)==1 and req.args.kwonlyargs[0].arg=="admitted_input" and dump(req.args.kw_defaults[0])==dump(ast.Constant(False)))
ck("first_mode_type_predicate",ast.unparse(req.body[0])=="require(type(admitted_input) is bool, 'Explicit fresh admission or authenticated admitted input')")
req.args.kwonlyargs=[];req.args.kw_defaults=[];req.body.pop(0)
calls=[n for n in ast.walk(req) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="validate_native_resource_disposition_v1"]
ck("single_admission_native_gate",len(calls)==1 and len(calls[0].keywords)==1 and calls[0].keywords[0].arg=="fresh" and ast.unparse(calls[0].keywords[0].value)=="not admitted_input")
calls[0].keywords[0].value=ast.Constant(True)
inputcalls=[n for n in ast.walk(fn(z,"validate_input_v1")) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="validate_request_v1"]
ck("authenticated_input_explicit_mode",len(inputcalls)==1 and ast.unparse(inputcalls[0].keywords[0])=="admitted_input=True")
inputcalls[0].keywords=[]
ck("common_entire_restored_AST_exact_v3",norm(z)==norm(old["common_v1.py"]))
# Exact restoration of all direct live-check changes.
z=copy.deepcopy(d);ctrl=fn(z,"controller_v1")
groups=[n for n in ctrl.body if isinstance(n,ast.For) and ast.unparse(n.target)=="group"]
ck("single_targeted_known_group_loop",len(groups)==1 and ast.unparse(groups[0].iter)=="native['native_process_groups']")
ctrl.body.remove(groups[0])
capfn=fn(z,"capacity_v1");trybody=next(n for n in capfn.body if isinstance(n,ast.Try)).body
ix=next(i for i,n in enumerate(trybody) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=="native")
added=trybody[ix:ix+3]
ck("direct_native_context_exact_three_statements",[type(x).__name__ for x in added]==["Assign","Assign","Expr"] and "gpu['compute']" in ast.unparse(added[2]) and "native_process_identities" in ast.unparse(added[1]))
del trybody[ix:ix+3]
inner=next(n for n in fn(z,"install_before_torch_v1").body if isinstance(n,ast.FunctionDef) and n.name=="validate_lease")
newcalls=[n for n in inner.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=="capacity_v1"]
ck("single_live_capacity_in_wrapped_lease",len(newcalls)==1 and ast.unparse(newcalls[0])=="capacity_v1(q, base, worker_pid=os.getpid())")
inner.body.remove(newcalls[0])
ck("direct_entire_restored_AST_exact_v3",norm(z)==norm(old["direct_gpu0_v1.py"]))
# Isolate complete actual pure validator, with every external observation replaced by RAM.
now=datetime.now(timezone.utc)
class FixedDT(datetime):
    @classmethod
    def now(cls,tz=None):return now if tz else now.replace(tzinfo=None)
ns=dict(Path=Path,datetime=FixedDT,timezone=timezone)
emit([n for n in c.body if isinstance(n,ast.Assign)],ns,"source_constants")
emit([fn(c,n) for n in ("require","active","validate_native_resource_disposition_v1","validate_request_v1","validate_input_v1","require_work_window_v2","unique")],ns,"full_validators")
vr=dict(path="RAM/native_resource.json",sha256="a"*64);ar=dict(path="RAM/native_absence.json",sha256="b"*64)
records={}
for key in ("NATIVE_RESOURCE_WAIT_REF","NATIVE_RESOURCE_OUTER_REF","NATIVE_RESOURCE_AMENDMENT_REF","NATIVE_ROOT_DISPOSITION_REF"):
    records[ns[key]["path"]]=j(ns[key]["path"])
wait=records[ns["NATIVE_RESOURCE_WAIT_REF"]["path"]]
uid=wait["last_capacity_sample"]["actual_worker_start_identity"]["uid"]
ns["os"]=SimpleNamespace(getuid=lambda:uid)
v=j(D+"/native_resource_disposition.INACTIVE.template.v1.json")
absence=j(D+"/native_targeted_absence.INACTIVE.template.v1.json")
v.update(created_at_utc=(now-timedelta(minutes=10)).isoformat(),expires_at_utc=(now+timedelta(hours=26)).isoformat(),status="resources_released",H058_job_id="H058-GPU0-v4-fulltrain-root02_oct17",observed_at_utc=(now-timedelta(seconds=443.077285)).isoformat(),fresh_targeted_native_absence_ref=ar,no_native_GPU0_attachment_alive=True,all_old_native_launch_resume_write_routes_permanently_revoked=True)
absence.update(observed_at_utc=v["observed_at_utc"],native_known_roots_and_contexts_absent=True,no_native_GPU0_attachment_alive=True,all_exact_native_identities_and_groups_absent=True,all_old_native_launch_resume_write_routes_permanently_revoked=True)
records[vr["path"]]=v;records[ar["path"]]=absence
def RAMread(r):return records[r["path"]] if r["path"] in records else j(r["path"])
ns.update(read=RAMread,ref=ref,time=SimpleNamespace(time=lambda:now.timestamp()),raw=lambda r:data(r["path"]),source_refs_v1=lambda:agg["refs"])
class RamPath(PurePosixPath):
    def is_dir(self):return True
ns["local"]=lambda p:RamPath(str(ROOT/p))
q=j(D+"/ROOT_request.INACTIVE.template.v1.json")
qr=dict(path=q["preparation_directory"]+"/ROOT_request.json",sha256="c"*64)
reviewref=dict(path="RAM/independent_review.json",sha256="d"*64)
q.update(created_at_utc=(now-timedelta(minutes=5)).isoformat(),expires_at_utc=(now+timedelta(hours=26)).isoformat(),execution_authorized=True,status="active",source_review_passed=True,source_review_ref=reviewref,native_resource_disposition_ref=vr,config_ref=dict(path="RAM/config.json",sha256="e"*64))
cfg=dict(source_manifest_ref=ref(D+"/producer_manifest.v1.json"))
producer=ref(D+"/producer_v1.py")
records[qr["path"]]=q
records[reviewref["path"]]=dict(independent=True,verdict="SOURCE_PASS",source_ref=ref(D+"/root_driver_v1.py"),source_manifest_ref=ref(D+"/source_manifest.v1.json"),blockers=[])
ns["science_v1"]=lambda request:(cfg,producer,SimpleNamespace(),j(D+"/producer_manifest.v1.json"))
ns["validate_gate_v1"]=lambda *args:None
native=ns["validate_native_resource_disposition_v1"]
ck("entire_native_validator_all50_and_exact_group",len(native(q,fresh=False)["native_process_identities"])==50 and native(q,fresh=False)["native_process_groups"]==[3752065])
reject("ordinary_request_rejects_443second_proof",lambda:ns["validate_request_v1"](qr),ValueError,"Fresh native absence")
ck("entire_request_admitted_accepts_aged_initial_snapshot",ns["validate_request_v1"](qr,admitted_input=True)[0] is q)
oldstamp=v["observed_at_utc"]
for seconds,accepted in ((299,True),(300,True),(301,False)):
    v["observed_at_utc"]=absence["observed_at_utc"]=(now-timedelta(seconds=seconds)).isoformat()
    if accepted:ck("fresh_admission_accepts_age_"+str(seconds),ns["validate_request_v1"](qr)[0] is q)
    else:reject("fresh_admission_rejects_age_"+str(seconds),lambda:ns["validate_request_v1"](qr),ValueError,"Fresh native absence")
v["observed_at_utc"]=absence["observed_at_utc"]=oldstamp
for bad in (None,0,1,"true"):reject("nonboolean_admitted_mode_"+repr(bad),lambda bad=bad:ns["validate_request_v1"](qr,admitted_input=bad),ValueError)
def mutated(mapping,key,value,call,name,kind=ValueError):
    before=copy.deepcopy(mapping.get(key));present=key in mapping
    mapping[key]=value
    try:reject(name,call,kind)
    finally:
        if present:mapping[key]=before
        else:mapping.pop(key,None)
for key,val in (("native_success_claimed",True),("native_scientific_acceptance_authorized",True),("native_outer_exit_code",0),("native_direct_worker_returncode",1),("all_old_native_launch_resume_write_routes_permanently_revoked",False),("native_process_identities",v["native_process_identities"][:-1]),("native_process_groups",[]),("historical_unknowns_preserved",False),("expires_at_utc",now.isoformat())):
    if key=="historical_unknowns_preserved":
        mutated(absence,key,val,lambda:ns["validate_request_v1"](qr,admitted_input=True),"admitted_preserves_absence_"+key)
    else:mutated(v,key,val,lambda:ns["validate_request_v1"](qr,admitted_input=True),"admitted_preserves_native_"+key)
for key,val in (("TRAIN_truth_only",False),("VAL_truth_authorized",True),("TEST_authorized",True),("automatic_retry_resume_or_fallback",True),("storage_policy_selector_ref",None),("adapter_manifest_ref",None),("source_review_passed",False),("phase","evaluate"),("expires_at_utc",(now+timedelta(hours=24)).isoformat()),("job_id","H058-GPU0-v3-fulltrain-root01_oct17")):
    mutated(q,key,val,lambda:ns["validate_request_v1"](qr,admitted_input=True),"full_request_guard_"+key)
for key,val in (("independent",False),("verdict","FAIL"),("source_ref",None),("source_manifest_ref",None),("blockers",["blocked"])):
    mutated(records[reviewref["path"]],key,val,lambda:ns["validate_request_v1"](qr,admitted_input=True),"full_independent_review_guard_"+key)
mutated(q["caps"],"max_cuda_reserved_bytes",27*2**30,lambda:ns["validate_request_v1"](qr,admitted_input=True),"full_exact28GiB_guard")
# Entire admitted-input validator authenticates storage scope/child/full role ceiling.
sr=dict(path="RAM/storage.json",sha256="f"*64);rr=dict(path="RAM/runtime.json",sha256="1"*64)
s=dict(schema="H058_full_parent_ROOT_input_v1",status="active",root_authorizer="root_agent",execution_authorized=True,root_stage_request_ref=qr,phase=q["phase"],job_id=q["job_id"],output_directory=q["output_directory"],producer_ref=producer,refs=dict(config=q["config_ref"],runtime=rr,storage=sr),root_context_module_ref=ref(D+"/context_factory_v1.py"),created_at_utc=q["created_at_utc"],expires_at_utc=q["expires_at_utc"],source_sha256={})
ir=dict(path=q["packet_directory"]+"/input.json",sha256="2"*64);records[ir["path"]]=s
needed=[ns["SELECTOR_REF"],ref(D+"/source_manifest.v1.json"),ref(D+"/root_driver_v1.py"),ref(D+"/context_factory_v1.py"),ref(D+"/direct_gpu0_v1.py"),qr,q["config_ref"],q["science_handoff_ref"],reviewref,cfg["source_manifest_ref"],producer,rr,vr]
storage=dict(job_id=q["job_id"],output_directory=q["output_directory"],reserved_new_bytes=256*2**20,reservation=dict(auxiliary_scopes=[str(ROOT/q["packet_directory"])],consumer_bindings=needed),completed_overlay_policy_ref=ns["POLICY"],budget_components=dict(consumer_adapter=ns["GUARD"]))
records[sr["path"]]=storage
ck("entire_admitted_input_accepts_aged_snapshot",ns["validate_input_v1"](ir)[0] is s)
for key,val in (("reserved_new_bytes",32*2**20),("completed_overlay_policy_ref",None),("output_directory","RAM/wrong")):
    mutated(storage,key,val,lambda:ns["validate_input_v1"](ir),"admitted_storage_guard_"+key)
mutated(storage["reservation"],"auxiliary_scopes",[],lambda:ns["validate_input_v1"](ir),"admitted_entire_packet_scope_required")
mutated(storage["reservation"],"consumer_bindings",needed+[dict(path="RAM/%d"%i,sha256="0"*64) for i in range(65-len(needed))],lambda:ns["validate_input_v1"](ir),"admitted65roles_rejected")
mutated(storage["reservation"],"consumer_bindings",[r for r in needed if r!=vr],lambda:ns["validate_input_v1"](ir),"admitted_native_disposition_role_required")
# Full controller and capacity functions, RAM PID/group/registry/inventory only.
dn=dict(ns,json=json)
emit([n for n in d.body if isinstance(n,ast.Assign)],dn,"direct_constants")
emit([fn(d,n) for n in ("controller_v1","bounded_capacity_failure_v3","capacity_v1")]+[next(n for n in d.body if isinstance(n,ast.ClassDef))],dn,"full_controller_capacity")
authorityref=dict(path="RAM/direct_authority.json",sha256="3"*64);terminalref=dict(path="RAM/terminal.json",sha256="4"*64);snapshotref=dict(path="RAM/registry_snapshot.json",sha256="5"*64);csref=dict(path="RAM/controller_storage.json",sha256="6"*64)
q.update(direct_attachment_authority_ref=authorityref,controller_terminal_qualification_ref=terminalref)
authority=j(D+"/direct_attachment_authority.INACTIVE.template.v1.json")
authority.update(created_at_utc=q["created_at_utc"],expires_at_utc=q["expires_at_utc"],status="active",H058_job_id=q["job_id"],borrowed_physical_inode_attachment_authorized=True,additional_old_controller_submissions_revoked_while_H058_attached=True,old_controller_preserved_until_H058_drained=True,all_old_native_launch_resume_write_routes_permanently_revoked=True,native_resource_disposition_ref=vr)
terminal=j(D+"/controller_terminal.INACTIVE.template.v1.json")
terminal.update(created_at_utc=q["created_at_utc"],expires_at_utc=q["expires_at_utc"],registry_snapshot_ref=snapshotref,all_old_controller_submissions_revoked=True,unknown_original_outer74136_preserved=True,controller_preserved=True)
cap=dict(expires_unix=now.timestamp()+30*3600,gpu_uuid=ns["GPU0_UUID"],physical_gpu=0,jobs={dn["C3_JOB"]:dict(cuda_cap_bytes=24*2**30,margin_mib=2048,peak_mib=26*1024)},controller_storage_ref=csref)
c3=dict(logical_job=dn["C3_JOB"],requested=cap["jobs"][dn["C3_JOB"]],state="failed",exit_code=1,leader_wait_status_known=True,owned_drain_observed=True,process_identities=[])
registry=dict(controller=dict(pid=dn["CONTROLLER_PID"],start_ticks=dn["CONTROLLER_START"]),physical_gpu=0,lifecycle="running",safety_error=None,ambiguous_descendants_or_lease_holders=False,capability_ref=dn["CURRENT_CAP"],jobs=[c3],updated_unix=now.timestamp()-1000)
records.update({authorityref["path"]:authority,terminalref["path"]:terminal,snapshotref["path"]:dict(jobs=copy.deepcopy(registry["jobs"])),csref["path"]:dict(output_directory="RAM/controller"),dn["CURRENT_CAP"]["path"]:cap})
class RegistryFile:
    def is_file(self):return True
    def is_symlink(self):return False
    def stat(self):return SimpleNamespace(st_size=1000)
    def read_bytes(self):return json.dumps(registry).encode()
dn["local"]=lambda p:RegistryFile()
observed={}
pidcalls=[];groupcalls=[]
controllerrow=dict(pid=dn["CONTROLLER_PID"],uid=uid,start_ticks=dn["CONTROLLER_START"],state="S")
def proc_identity(pid):
    pidcalls.append(pid)
    if pid==dn["CONTROLLER_PID"]:return controllerrow
    value=observed.get(pid,FileNotFoundError())
    if isinstance(value,Exception):raise value
    return value
def group_absent(group,signal):
    groupcalls.append((group,signal))
    raise ProcessLookupError(errno.ESRCH,"RAM absent")
gpu=dict(index=0,uuid=ns["GPU0_UUID"],used_mib=6,total_mib=81920,compute=[])
base=SimpleNamespace(proc_identity=proc_identity,real_inventory=lambda:[copy.deepcopy(gpu)])
dn["os"]=SimpleNamespace(getuid=lambda:uid,killpg=group_absent)
ck("full_controller_pass_with_exact_absence",dn["controller_v1"](q,base)[2]==c3)
ck("full50_actual_PID_queries",set(pidcalls)=={r["pid"] for r in v["native_process_identities"]}|{dn["CONTROLLER_PID"]})
ck("only_exact_group_zero_probe",set(groupcalls)=={(3752065,0)})
for index,saved in enumerate(v["native_process_identities"]):
    observed[saved["pid"]]=dict(saved,state="S")
    reject("live_native_identity_%02d"%index,lambda:dn["controller_v1"](q,base),ValueError,"Known native identity still live")
    observed[saved["pid"]]=dict(saved,state="S",start_ticks=saved["start_ticks"]+1)
    ck("reused_PID_different_start_%02d"%index,dn["controller_v1"](q,base)[2]==c3)
    observed.clear()
for error in (PermissionError(errno.EACCES,"RAM denied"),OSError(errno.EIO,"RAM unreadable")):
    observed[v["native_process_identities"][0]["pid"]]=error
    reject("native_unknown_identity_blocks_"+type(error).__name__,lambda:dn["controller_v1"](q,base),type(error));observed.clear()
dn["os"].killpg=lambda *args:None
reject("live_native_group_blocks",lambda:dn["controller_v1"](q,base),ValueError,"Known prior native process group")
def denied_group(*args):raise PermissionError(errno.EPERM,"RAM denied")
dn["os"].killpg=denied_group
reject("denied_native_group_blocks",lambda:dn["controller_v1"](q,base),PermissionError)
dn["os"].killpg=group_absent
for key,val in (("start_ticks",dn["CONTROLLER_START"]+1),("uid",uid+1),("state","Z")):
    mutated(controllerrow,key,val,lambda:dn["controller_v1"](q,base),"controller_exact_"+key)
mutated(authority,"borrowed_physical_inode_attachment_authorized",False,lambda:dn["controller_v1"](q,base),"borrowed_authority_required")
mutated(cap,"physical_gpu",1,lambda:dn["controller_v1"](q,base),"controller_GPU0_required")
ck("full_capacity_conservative_envelopes",dn["capacity_v1"](q,base)["projected_mib"]==61446)
for index,saved in enumerate(v["native_process_identities"]):
    gpu["compute"]=[dict(pid=saved["pid"],name="AI Auto Research Pipeline Test",used_mib=1)]
    reject("native_GPU_context_%02d"%index,lambda:dn["capacity_v1"](q,base),ValueError,"Prior native roster")
gpu["compute"]=[dict(pid=900000+i,name="external",used_mib=1) for i in range(10)]+[dict(pid=v["native_process_identities"][-1]["pid"],name="native",used_mib=1)]
reject("native_context_after_first8_still_blocks",lambda:dn["capacity_v1"](q,base),ValueError,"Prior native roster")
gpu["compute"]=[dict(pid=999999,name="external unchanged",used_mib=1024)]
ck("external_context_retained",dn["capacity_v1"](q,base)["external_compute"]==gpu["compute"])
gpu["compute"]=[dict(pid=888888,name="AI Auto Research Pipeline Test",used_mib=512)]
ck("own_context_classified_without_lease_result_change",dn["capacity_v1"](q,base,worker_pid=888888)["H058_observed_mib"]==512)
gpu["compute"][0]["name"]="wrong"
reject("own_cuda_title_required",lambda:dn["capacity_v1"](q,base,worker_pid=888888),dn["GPUCapacityFailureV3"],"H058 actual CUDA title")
gpu["compute"]=[];gpu["used_mib"]=30000
reject("complete_external_and_driver_occupancy_blocks",lambda:dn["capacity_v1"](q,base),dn["GPUCapacityFailureV3"],"fitGPU0")
gpu["used_mib"]=6
# Actual wrapped installer, fake OS/FD/native guard only, varying dynamic samples.
ins=dict(ns);emit([fn(d,"install_before_torch_v1")],ins,"full_installer")
lease=dict(lease_fd=10,lease_inode=100,lease_device=1,lease_metadata_sha256="x",launcher_metadata=dict(job="controller"))
ticket=dict(physical_inode=100,physical_device=1,physical_lease_sha256="x",physical_lease_metadata=lease["launcher_metadata"],logical_device=1,logical_inode=200,logical_lock_path="RAM/logical")
events=[]
nativeguard=SimpleNamespace(validate_lease=lambda environ=None:dict(lease),apply_and_validate_title=lambda:events.append("title"))
ins.update(sys=SimpleNamespace(modules={}),load=lambda *args:SimpleNamespace(install_before_torch_v1=lambda:nativeguard),base_v1=lambda:base,controller_v1=lambda *args:events.append("controller"),held_writable_probe_v1=lambda p:events.append("probe"),ref=ref,Path=lambda p:SimpleNamespace(lstat=lambda:SimpleNamespace(st_dev=1,st_ino=200)),stat=stat,os=SimpleNamespace(environ={"H058_LOGICAL_LOCK_FD":"11","ARGGNN_GPU_LEASE_FD":"10"},getpid=lambda:888888,fstat=lambda fd:SimpleNamespace(st_mode=stat.S_IFREG,st_dev=1,st_ino=200)))
samples=[]
def RAMcapacity(q,base,worker_pid=None):
    events.append("capacity");samples.append(worker_pid)
    return dict(observed_unix=len(samples),GPU0=dict(used_mib=len(samples)))
ins["capacity_v1"]=RAMcapacity
installed=ins["install_before_torch_v1"]({},q,ticket)
first=installed.validate_lease();second=installed.validate_lease()
ck("dynamic_capacity_samples_do_not_change_lease_equality",first==second and len(samples)==3 and samples==[888888]*3)
ck("installer_capacity_precedes_title",events.index("capacity")<events.index("title"))
ck("wrapped_result_has_truthful_borrowed_semantics",first["borrowed_physical_inode"] is True and first["physical_exclusive_lock_owned_by_current_controller"] is True and first["H058_logical_lock_inode"]==200)
def failing_capacity(*args,**kw):raise ValueError("RAM full current capacity failure")
ins["capacity_v1"]=failing_capacity
reject("preCUDA_live_capacity_failure_blocks_wrapped_lease",installed.validate_lease,ValueError,"RAM full")
# Reachability joins and preserved first-CUDA ordering, source only.
def calls(fnnode,name):return [n for n in ast.walk(fnnode) if isinstance(n,ast.Call) and ast.unparse(n.func)==name]
ck("admit_keeps_fresh_default",len(calls(fn(driver,"admit_v1"),"validate_request_v1"))==1 and calls(fn(driver,"admit_v1"),"validate_request_v1")[0].keywords==[])
worker=fn(driver,"worker_v1")
ck("worker_authentication_before_install_context_producer",calls(worker,"validate_input_v1")[0].lineno<calls(worker,"direct.install_before_torch_v1")[0].lineno<calls(worker,"context_factory_v1.build_context_v1")[0].lineno<calls(worker,"producer.run_context_v1")[0].lineno)
context=fn(trees["context_factory_v1.py"],"build_context_v1")
ck("context_current_lease_then_guard_then_configure_fit",calls(context,"validate_lease")[0].lineno<calls(context,"g.budget_check_v9")[0].lineno<calls(context,"configure_fit")[0].lineno)
configure=fn(ast.parse(text("src/arggnn/biological_candidate.py")),"configure_fit")
cuda=[n for n in ast.walk(configure) if isinstance(n,ast.Call) and ast.unparse(n.func).startswith("torch.cuda.")]
ck("pinned_configure_fit_lease_before_all_direct_cuda_calls",calls(configure,"validate_lease")[0].lineno<min(n.lineno for n in cuda))
ck("configure_fit_imports_wrapped_canonical_lease",any(isinstance(n,ast.ImportFrom) and n.module=="native_argformer_exec" and any(a.name=="validate_lease" for a in n.names) for n in ast.walk(configure)))
actualinner=next(n for n in fn(d,"install_before_torch_v1").body if isinstance(n,ast.FunctionDef))
oldinner=next(n for n in fn(old["direct_gpu0_v1.py"],"install_before_torch_v1").body if isinstance(n,ast.FunctionDef))
ck("exact_lease_return_AST_preserved",norm(next(n for n in actualinner.body if isinstance(n,ast.Return)))==norm(next(n for n in oldinner.body if isinstance(n,ast.Return))))
ck("no_capacity_validate_lease_recursion",not calls(fn(d,"capacity_v1"),"validate_lease"))
ck("attach_requires_current_capacity",bool(calls(fn(d,"attach_v1"),"capacity_v1")))
ck("supervisor_requires_current_capacity_and_attachment",bool(calls(fn(driver,"execute_v1"),"direct.capacity_v1")) and bool(calls(fn(driver,"execute_v1"),"direct.check_attachment_v1")))
# Actual source-derived direct role union; no state/guard/memo execution.
role_ns=dict(ns,q=q,cfg=cfg,producer=producer,request_ref=qr,runtime=rr,closure=dict(path="RAM/closure.json",sha256="7"*64),direct=SimpleNamespace(CURRENT_CAP=dn["CURRENT_CAP"]))
for n in fn(driver,"admit_v1").body:
    nodes=[]
    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="needed" for t in n.targets):nodes=[n]
    elif isinstance(n,ast.AugAssign) and ast.unparse(n.target)=="needed":nodes=[n]
    elif isinstance(n,ast.If):nodes=[m for m in n.body if isinstance(m,ast.AugAssign) and ast.unparse(m.target)=="needed"]
    if nodes:emit(nodes,role_ns,"exact_role_expression")
capturepath="research/reviews/Relate_failed64GiB_retention_ROOT_activation_20261017_v1/actual_parent_capture.root_v1.json"
capture=j(capturepath);named=capture["all_actual_parent_bindings"];frozen={r["path"]:r for r in named.values()}
selected=dict(frozen);selected[ns["SELECTOR_REF"]["path"]]=ns["SELECTOR_REF"]
union=dict(selected)
for r in role_ns["needed"]:
    assert r["path"] not in union or union[r["path"]]==r
    union[r["path"]]=r
ck("actual_source_role_union_45named38frozen39selected_plus25_equals64",len(named)==45 and len(frozen)==38 and len(selected)==39 and len(union)==64 and len(set(union)-set(selected))==25)
child=ast.parse(text(ns["GUARD"]["path"]))
bind=ast.unparse(fn(child,"_bindings"))
ck("selected_child_preserves_all_actual_parent_roles_plus_one_capsule","copy.deepcopy(base_bindings)" in bind and "Relate64_explicit_selector_capsule" in bind and "result.update(additions)" in bind)
# Native history and H058 failure provenance, source/metadata only.
prior=j(q["prior_H058_CPU_disposition_ref"]["path"])
ck("actual_prior_budget_and_unknowns",prior["new_gpu_request_job_id"]==q["job_id"] and prior["new_GPU_budget"]==1920 and prior["new_GPU_budget_per_arm"]==960 and all(v is None for v in prior["actual_optimizer_steps"].values()) and set(prior["verified_optimizer_step_lower_bounds"].values())=={1} and set(prior["prior_GPU_actual_optimizer_steps"].values())=={0} and prior["no_CPU_or_GPU_optimizer_state_transfer"] is True)
v3terminal=j(prior["prior_GPU_attempt_terminal_ref"]["path"])
v3log=j(v3terminal["worker_log_ref"]["path"]);v3wait=j(v3terminal["actual_wait_ref"]["path"])
ck("actual_v3_terminal_worker1_outer1_drained",v3terminal["actual_exit_code"]==v3wait["returncode"]==1 and v3terminal["actual_terminal_consumed"] is True and v3wait["all_owned_processes_and_groups_drained"] is True and v3wait["completed_ref"] is None)
ck("actual_v3_log_freshness_failure_before_install_and_producer","ValueError: Fresh native absence before admission/launch" in v3log["tail"] and 's,q,cfg,storage,protocol=validate_input_v1(input_ref)' in v3log["tail"] and "context_factory_v1" not in v3log["tail"] and "producer.run_context_v1" not in v3log["tail"])
protocol=ast.parse(text("research/reviews/H058_full_parent_CPU_execution_source_20261017_v1/protocol_v1.py"))
ck("science32parents30passes_two960_updates_source","range(1, 31)" in ast.unparse(fn(trees["producer_v1.py"],"run_train_v1")) and "{a: 960 for a in arms}" in ast.unparse(fn(trees["producer_v1.py"],"run_train_v1")))
roles=dict(frozen_named=45,frozen_distinct=38,current_selected_distinct=39,additional_distinct=25,total=64,actual_live_state_evidence=False,direct_role_paths=sorted(union),capture_ref=ref(capturepath))
result=dict(schema="H058_GPU0_v4_independent_whole_join_source_checks_v1",status="PASS",count=len(checks),checks=checks,role_arithmetic=roles,production_refs=[ref(D+"/"+n) for n in NAMES],source_ref=ref(D+"/root_driver_v1.py"),source_manifest_ref=ref(D+"/source_manifest.v1.json"),source_handoff_ref=ref(D+"/source_handoff.v4.json"),five_namespace_only_files=[n for n in NAMES if n not in ("common_v1.py","direct_gpu0_v1.py")],semantic_change_files=["common_v1.py","direct_gpu0_v1.py"],production_imports=False,actual_process_group_GPU_inspection=False,guard_accounting_mutex_calls=False,model_runtime_data_truth_checkpoint_reads=False,admission_launch_Git=False)
with (ROOT/OUT/"checks.result.v1.json").open("x") as f:json.dump(result,f,sort_keys=True,indent=2);f.write("\n")
print(json.dumps(dict(status="PASS",count=len(checks),role_arithmetic={k:v for k,v in roles.items() if k!="direct_role_paths"},result_ref=ref(OUT+"/checks.result.v1.json")),sort_keys=True))

