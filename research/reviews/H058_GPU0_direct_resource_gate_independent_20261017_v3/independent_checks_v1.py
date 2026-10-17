"""Independent source-only AST/RAM review. Never imports production modules."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
import ast,copy,difflib,errno,hashlib,json
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
TARGET="research/reviews/H058_GPU0_full_training_successor_source_20261017_v3"
PREV="research/reviews/H058_GPU0_full_training_successor_source_20261017_v2"
OUT=ROOT/"research/reviews/H058_GPU0_direct_resource_gate_independent_20261017_v3"
FILES=("__init__.py","common_v1.py","context_factory_v1.py","direct_gpu0_v1.py","observations_gpu_v1.py","producer_v1.py","root_driver_v1.py")
results=[]
def check(name,ok):
    if not ok:raise AssertionError(name)
    results.append(dict(name=name,status="PASS"))
def meta(path):
    p=ROOT/path
    assert p.resolve()==p and p.is_file() and not p.is_symlink()
    assert p.suffix in (".py",".json",".md",".patch") and p.stat().st_size<=32*2**20
    return p.read_bytes()
def jread(path):return json.loads(meta(path))
def exact(path):return dict(path=path,sha256=hashlib.sha256(meta(path)).hexdigest())
expected={
 "source_handoff.v3.json":"d945b953e49d80fdd98ec30329e5cd42122687506292b5f690609400e5bd78b6",
 "source_manifest.v1.json":"1c8e0efc286d98cf9fffeceaac640709091f9aef98a60114a512ff4c305b9354",
 "MANIFEST.v3.json":"f780b502bed93c90a77ca97c2e47d8457846999395b200bb6d350bf5d19ab62a",
 "SOURCE_MAP.v3.json":"ff49a12718d1741583540354bf4ca254b4ff8b29c7c637c06e10f8a58ebc8a3f",
 "REPORT.md":"8d4bbf3a47f63179df0b8a0f471908e58eb68e83b0673433d7c3c7f4c71c1a7c",
 "ROOT_RECIPE.md":"c86ed60ca96aa961967ae385edc2b0c4bd7d81286595d60c7238586adf2da6cc"}
for n,h in expected.items():check("handoff_seal_"+n,exact(TARGET+"/"+n)["sha256"]==h)
artifact=jread(TARGET+"/MANIFEST.v3.json")
for row in artifact["files"]:
    p=row["path"];b=meta(p)
    check("artifact_"+p,hashlib.sha256(b).hexdigest()==row["sha256"] and len(b)==row["bytes"])
check("artifact_39_file_total",len(artifact["files"])==39 and sum(r["bytes"] for r in artifact["files"])==artifact["total_bytes"]==351744)
check("artifact_directory_has_no_unsealed_files",{str(p.relative_to(ROOT)) for p in (ROOT/TARGET).iterdir() if p.is_file()}=={x["path"] for x in artifact["files"]}|{TARGET+"/MANIFEST.v3.json"})
manifest=jread(TARGET+"/source_manifest.v1.json")
check("aggregate_unique_254",len(manifest["refs"])==254==len({r["path"] for r in manifest["refs"]}))
for row in manifest["refs"]:check("aggregate_"+row["path"],exact(row["path"])==row)
check("source_sha_map_exact",manifest["source_sha256"]=={r["path"]:r["sha256"] for r in manifest["refs"]})
for n in ("producer_manifest.v1.json","producer_handoff.v1.json","source_handoff.v3.json","SOURCE_MAP.v3.json","ROOT_request.INACTIVE.template.v1.json"):
    def refs(q):
        if isinstance(q,dict):
            if set(q)=={"path","sha256"}:yield q
            else:
                for v in q.values():yield from refs(v)
        elif isinstance(q,list):
            for v in q:yield from refs(v)
    for row in refs(jread(TARGET+"/"+n)):check("joined_ref_"+n+"_"+row["path"],exact(row["path"])==row)
source_map=jread(TARGET+"/SOURCE_MAP.v3.json")
check("seven_production_source_map",set(Path(x["successor_ref"]["path"]).name for x in source_map["files"])==set(FILES))
for row in jread(PREV+"/MANIFEST.v2.json")["files"]:
    b=meta(row["path"])
    check("frozen_v2_"+row["path"],hashlib.sha256(b).hexdigest()==row["sha256"] and len(b)==row["bytes"])
old={n:meta(PREV+"/"+n).decode() for n in FILES}
new={n:meta(TARGET+"/"+n).decode() for n in FILES}
trees={n:ast.parse(new[n]) for n in FILES}
def normalize(text):
    return text.replace("_20261017_v3","_20261017_v2").replace("H058_GPU0_v3_","H058_GPU0_v2_").replace("H058-GPU0-v3-","H058-GPU0-v2-").replace("native_resource_disposition_ref","native_noncoexistence_qualification_ref").replace("validate_native_resource_disposition_v1","validate_native_noncoexistence_v3").replace("all_old_native_launch_resume_write_routes_permanently_revoked","all_native_launch_resume_execution_routes_revoked_while_H058_attached").replace("Fresh exact ROOT borrowed attachment after genuine native terminal and owned drain","Fresh exact ROOT borrowed attachment after native successful completion")
native_names={"NATIVE_ROOT_DISPOSITION_REF","NATIVE_RESOURCE_WAIT_REF","NATIVE_RESOURCE_OUTER_REF","NATIVE_RESOURCE_AMENDMENT_REF"}
def normalized_tree(text,common=False):
    a=ast.parse(normalize(text))
    if common:
        a.body=[x for x in a.body if not(isinstance(x,ast.FunctionDef) and x.name=="validate_native_noncoexistence_v3") and not(isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id in native_names for t in x.targets))]
    return ast.dump(a,include_attributes=False)
for n in FILES:
    check("exact_allowed_AST_delta_"+n,normalized_tree(old[n],n=="common_v1.py")==normalized_tree(new[n],n=="common_v1.py"))
now=datetime.now(timezone.utc)
class FixedDateTime(datetime):
    @classmethod
    def now(cls,tz=None):return now if tz else now.replace(tzinfo=None)
ns=dict(Path=Path,datetime=FixedDateTime,timezone=timezone,os=SimpleNamespace(getuid=lambda:1000),time=SimpleNamespace(time=lambda:now.timestamp()))
for v in trees["common_v1.py"].body:
    if isinstance(v,ast.Assign):
        exec(compile(ast.Module(body=[v],type_ignores=[]),"<isolated_constants>","exec"),ns)
def isolate(tree,names,namespace):
    nodes=[x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name in names]
    assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),"<isolated_reviewed_functions>","exec"),namespace)
isolate(trees["common_v1.py"],{"require","active","validate_native_resource_disposition_v1","unique","validate_request_v1"},ns)
check("caps_exact",ns["CAPS"]==jread(TARGET+"/caps.INACTIVE.v1.json")["caps"] if "caps" in jread(TARGET+"/caps.INACTIVE.v1.json") else ns["CAPS"]["reservation_bytes"]==256*2**20)
v=jread(TARGET+"/native_resource_disposition.INACTIVE.template.v1.json")
absence=jread(TARGET+"/native_targeted_absence.INACTIVE.template.v1.json")
vr=dict(path="RAM/native_resource.json",sha256="a"*64)
ar=dict(path="RAM/native_absence.json",sha256="b"*64)
v.update(created_at_utc=(now-timedelta(seconds=1)).isoformat(),expires_at_utc=(now+timedelta(hours=26)).isoformat(),H058_job_id="H058-GPU0-v3-fulltrain-root01_oct17",status="resources_released",observed_at_utc=now.isoformat(),fresh_targeted_native_absence_ref=ar,no_native_GPU0_attachment_alive=True,all_old_native_launch_resume_write_routes_permanently_revoked=True)
absence.update(observed_at_utc=now.isoformat(),native_known_roots_and_contexts_absent=True,no_native_GPU0_attachment_alive=True,all_exact_native_identities_and_groups_absent=True,all_old_native_launch_resume_write_routes_permanently_revoked=True)
baseline={vr["path"]:v,ar["path"]:absence}
for k in native_names:
    row=ns[k];baseline[row["path"]]=jread(row["path"])
q=dict(job_id=v["H058_job_id"],native_resource_disposition_ref=vr)
current=copy.deepcopy(baseline)
ns["read"]=lambda r:current[r["path"]]
ns["ref"]=lambda p:exact(p)
fn=ns["validate_native_resource_disposition_v1"]
check("actual_direct0_outer1_resource_gate_accepts_RAM_new_cert",fn(q,fresh=True)==v)
def reject(name,path,key,value,fresh=True):
    global current
    current=copy.deepcopy(baseline)
    row=current[path]
    keys=key if isinstance(key,tuple) else (key,)
    for k in keys[:-1]:row=row[k]
    row[keys[-1]]=value
    try:fn(q,fresh=fresh)
    except (ValueError,KeyError,TypeError):check(name,True)
    else:raise AssertionError("accepted invalid case "+name)
for key,value in {
 "status":"qualified","schema":"H058_ROOT_after_native_success_qualification_v1","root_authorizer":"other","H058_job_id":"other",
 "physical_gpu":1,"actual_native_outer_terminal_consumed":False,"native_outer_exit_code":0,"native_direct_worker_returncode":-15,
 "native_operational_failure":None,"native_success_claimed":True,"native_scientific_acceptance_authorized":True,
 "no_native_GPU0_attachment_alive":False,"all_old_native_launch_resume_write_routes_permanently_revoked":False,
 "C3_controller_preserved_until_H058_drained":False,"native_process_identities":v["native_process_identities"][:-1],
 "native_process_groups":[],"created_at_utc":(now+timedelta(seconds=1)).isoformat(),"expires_at_utc":now.isoformat()
}.items():reject("resource_reject_"+key,vr["path"],key,value)
for key in ("native_actual_wait_ref","native_actual_outer_terminal_ref","native_execution_amendment_ref","native_ROOT_terminal_disposition_ref","adapter_manifest_ref"):
    x=copy.deepcopy(v[key]);x["sha256"]="0"*64
    reject("resource_reject_wrong_pin_"+key,vr["path"],key,x)
wp=ns["NATIVE_RESOURCE_WAIT_REF"]["path"];op=ns["NATIVE_RESOURCE_OUTER_REF"]["path"];dp=ns["NATIVE_ROOT_DISPOSITION_REF"]["path"];mp=ns["NATIVE_RESOURCE_AMENDMENT_REF"]["path"]
for key,value in {
 "schema":"wrong","returncode":1,"actual_direct_worker_wait":False,"all_owned_processes_and_groups_drained":False,
 "owned_identity_absence":False,"pipe_EOF_observed":False,"cleanup_errors":["error"],"complete_owned_roster_known":False,
 "actual_execution_amendment_ref":None,"input_ref":None,"adapter_manifest_ref":None,"operational_failure":None
}.items():reject("wait_reject_"+key,wp,key,value)
for key,value in {"complete_owned_roster_known":False,"owned_roster_truncated":True,"known_identity_count":49,"owned_identities":baseline[wp]["ownership"]["owned_identities"][:-1]}.items():
    reject("roster_reject_"+key,wp,("ownership",key),value)
dupe=copy.deepcopy(baseline[wp]["ownership"]["owned_identities"]);dupe[-1]=dupe[0]
reject("roster_reject_duplicate",wp,("ownership","owned_identities"),dupe)
reject("wait_reject_current_UID_mismatch",wp,("last_capacity_sample","actual_worker_start_identity","uid"),1001)
reject("wait_reject_group_unobserved",wp,"group_observations",[dict(pgid=3752065,absent=None)])
reject("wait_reject_group_live",wp,"group_observations",[dict(pgid=3752065,absent=False)])
for key,value in {"tool_session_id":70446,"terminal_consumed":False,"actual_tool_terminal_exit_code":0,
 "actual_direct_worker_returncode_in_tool_output":-15,"all_owned_processes_and_groups_drained_in_tool_output":False,
 "outer_failure_preserved":False,"successful_outer_terminal_claimed":True,"automatic_retry_authorized":True}.items():
    reject("outer_reject_"+key,op,key,value)
for key,value in {"schema":"wrong","actual_wait_ref":None,"actual_outer_terminal_ref":None,"amendment_ref":None,
 "actual_wait_input_ref":None,"actual_direct_worker_returncode":1,"actual_outer_returncode":0,"outer_terminal_consumed":False,
 "all_owned_processes_and_groups_drained":False,"current_v12_known_identity_count":49,
 "all_future_old_VAL000_MLM_launch_resume_write_routes_revoked":False,"operational_failure":None,
 "historical_unknowns_preserved":False,"all_old_files_scopes_charges_clocks_preserved":False,"new_native_MLM_replay_authorized":True}.items():
    reject("ROOT_disposition_reject_"+key,dp,key,value)
for key,value in {"original_input_ref":None,"adapter_manifest_ref":None}.items():reject("amendment_reject_"+key,mp,key,value)
for key,value in {"root_authorizer":"other","native_known_roots_and_contexts_absent":False,"no_native_GPU0_attachment_alive":False,
 "native_actual_wait_ref":None,"native_actual_outer_terminal_ref":None,"native_process_identities":v["native_process_identities"][:-1],
 "native_process_groups":[],"all_exact_native_identities_and_groups_absent":False,
 "all_old_native_launch_resume_write_routes_permanently_revoked":False,"historical_unknowns_preserved":False,
 "observed_at_utc":(now-timedelta(seconds=1)).isoformat()}.items():reject("absence_reject_"+key,ar["path"],key,value)
for delta,name in ((-301,"stale"),(1,"future")):
    current=copy.deepcopy(baseline);stamp=(now+timedelta(seconds=delta)).isoformat()
    current[vr["path"]]["observed_at_utc"]=current[ar["path"]]["observed_at_utc"]=stamp
    try:fn(q,fresh=True)
    except ValueError:check("absence_reject_"+name,True)
    else:raise AssertionError(name)
current=copy.deepcopy(baseline);stamp=(now-timedelta(seconds=1000)).isoformat()
current[vr["path"]]["observed_at_utc"]=current[ar["path"]]["observed_at_utc"]=stamp
check("continuing_attachment_keeps_original_observation_without_fake_refresh",fn(q,fresh=False)["observed_at_utc"]==stamp)
# Exercise the real controller prefix through the exact prior-native identity loop only.
controller=next(n for n in trees["direct_gpu0_v1.py"].body if isinstance(n,ast.FunctionDef) and n.name=="controller_v1")
prefix=copy.deepcopy(controller)
cut=next(i for i,n in enumerate(prefix.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="cap" for t in n.targets))
prefix.body=prefix.body[:cut]+[ast.Return(value=ast.Constant(value=True))]
ast.fix_missing_locations(prefix)
directns=dict(ns);directns.update(CONTROLLER_PID=3349405,CONTROLLER_START=481281638)
capref=exact("research/reviews/C3_module_route_repair_root_20261016_v1/controller/capability.root_active_v1.json")
directns["CURRENT_CAP"]=capref
authority=jread(TARGET+"/direct_attachment_authority.INACTIVE.template.v1.json")
authority.update(created_at_utc=(now-timedelta(seconds=1)).isoformat(),expires_at_utc=(now+timedelta(hours=26)).isoformat(),status="active",borrowed_physical_inode_attachment_authorized=True,additional_old_controller_submissions_revoked_while_H058_attached=True,H058_job_id=q["job_id"],native_resource_disposition_ref=vr,all_old_native_launch_resume_write_routes_permanently_revoked=True)
dr=dict(path="RAM/direct_authority.json",sha256="c"*64)
cq=dict(q,direct_attachment_authority_ref=dr)
current=copy.deepcopy(baseline);current[dr["path"]]=authority
directns["read"]=lambda r:current[r["path"]]
directns["validate_native_resource_disposition_v1"]=lambda q,fresh:current[vr["path"]]
exec(compile(ast.Module(body=[prefix],type_ignores=[]),"<isolated_exact_controller_prefix>","exec"),directns)
def missing(pid):raise ProcessLookupError()
check("controller_all_exact_native_identities_absent",directns["controller_v1"](cq,SimpleNamespace(proc_identity=missing)))
saved=v["native_process_identities"][0]
for state,start,uid,allowed in (("R",saved["start_ticks"],1000,False),("Z",saved["start_ticks"],1000,True),("X",saved["start_ticks"],1000,True),("R",saved["start_ticks"]+1,1000,True),("R",saved["start_ticks"],1001,True)):
    def identity(pid,state=state,start=start,uid=uid):
        if pid!=saved["pid"]:raise ProcessLookupError()
        return dict(pid=pid,uid=uid,start_ticks=start,state=state)
    try:result=directns["controller_v1"](cq,SimpleNamespace(proc_identity=identity))
    except ValueError:check("native_identity_"+state+"_"+str(start)+"_"+str(uid),not allowed)
    else:check("native_identity_"+state+"_"+str(start)+"_"+str(uid),allowed and result)
for key,value in {"borrowed_physical_inode_attachment_authorized":False,"controller_pid":3349406,"controller_start_ticks":1,"controller_capability_ref":None,"H058_max_cuda_reserved_bytes":27*2**30,"all_old_native_launch_resume_write_routes_permanently_revoked":False,"native_resource_disposition_ref":None}.items():
    current=copy.deepcopy(baseline);current[dr["path"]]=copy.deepcopy(authority);current[dr["path"]][key]=value
    try:directns["controller_v1"](cq,SimpleNamespace(proc_identity=missing))
    except ValueError:check("attachment_authority_reject_"+key,True)
    else:raise AssertionError(key)
# Full unchanged capacity function with only RAM controller/inventory stubs.
capns=dict(ns);capns["CURRENT_CAP"]=capref;capns["controller_v1"]=lambda q,b:({},{"updated_unix":123},{"process_identities":[]},{})
isolate(trees["direct_gpu0_v1.py"],{"GPUCapacityFailureV3","bounded_capacity_failure_v3","capacity_v1"},capns)
capns.update(json=json,time=SimpleNamespace(time=lambda:now.timestamp()),GPU0_UUID=ns["GPU0_UUID"])
gpu=dict(index=0,uuid=ns["GPU0_UUID"],total_mib=81920,used_mib=6,compute=[])
inventory=SimpleNamespace(real_inventory=lambda:[gpu],proc_identity=missing)
cQ=dict(caps=ns["CAPS"])
r=capns["capacity_v1"](cQ,inventory)
check("capacity_26plus30plus4_GiB",r["projected_mib"]==61446 and r["actual_free_required_mib"]==61440 and r["headroom_mib"]==4096)
for total,used,apps,name in ((61439,0,[],"reservation_sum"),(81920,20481,[],"external_driver"),(81920,0,[dict(pid=123,used_mib=20481,name="external")],"external_full_inventory")):
    gpu.update(total_mib=total,used_mib=used,compute=apps)
    try:capns["capacity_v1"](cQ,inventory)
    except capns["GPUCapacityFailureV3"] as e:check("capacity_reject_"+name,e.capacity_sample["raw_capacity_decisions_used_complete_inventory"] is True and len(json.dumps(e.capacity_sample))<4096)
    else:raise AssertionError(name)
gpu.update(total_mib=81920,used_mib=100,compute=[dict(pid=123,used_mib=100,name="AI Auto Research Pipeline Test")])
r=capns["capacity_v1"](cQ,inventory,worker_pid=123)
check("capacity_owned_H058_reservation_not_double_counted",r["H058_observed_mib"]==100 and r["projected_mib"]==61440)
gpu["compute"][0]["name"]="wrong"
try:capns["capacity_v1"](cQ,inventory,worker_pid=123)
except capns["GPUCapacityFailureV3"]:check("capacity_reject_wrong_owned_GPU_title",True)
else:raise AssertionError("title")
# Writable exclusion behavior, using a fake descriptor and fake flock only.
probe_ns=dict(ns);isolate(trees["direct_gpu0_v1.py"],{"held_writable_probe_v1"},probe_ns)
class Opened:
    def __enter__(self):return self
    def __exit__(self,*args):return False
class ProbePath:
    def open(self,mode):
        check("probe_opens_writable_"+str(len(results)),mode=="r+")
        return Opened()
def held(*args):raise BlockingIOError(errno.EAGAIN,"held")
probe_ns["fcntl"]=SimpleNamespace(flock=held,LOCK_EX=2,LOCK_NB=4,LOCK_UN=8)
check("writable_probe_held_accepted",probe_ns["held_writable_probe_v1"](ProbePath()) is True)
probe_ns["fcntl"].flock=lambda *args:None
try:probe_ns["held_writable_probe_v1"](ProbePath())
except RuntimeError:check("writable_probe_unheld_rejected",True)
else:raise AssertionError("unheld")
for err in (errno.EBADF,errno.EACCES,errno.ENOLCK):
    def broken(*args,err=err):raise OSError(err,"error")
    probe_ns["fcntl"].flock=broken
    try:probe_ns["held_writable_probe_v1"](ProbePath())
    except OSError as e:check("writable_probe_error_propagates_"+str(err),e.errno==err)
    else:raise AssertionError(err)
# Native scientific predicate remains separately strict; no real completion/checkpoint read.
science_path="research/reviews/native_ARGformer_postdrain_pool_source_20261017_v2/downstream_context_v1.py"
science_tree=ast.parse(meta(science_path).decode())
complete_ref=dict(path="RAM/science_complete.json",sha256="d"*64)
science_wait=copy.deepcopy(baseline[wp]);science_wait["completed_ref"]=complete_ref
science_complete=dict(parent_id="american_c3_val_000",query_ancestry_truth_opened=False,child_input_ref=science_wait["input_ref"])
science_outer=dict(actual_exit_code=0,actual_terminal_consumed=True,actual_direct_worker_returncode=0,all_owned_processes_and_groups_drained=True,actual_wait_ref=ns["NATIVE_RESOURCE_WAIT_REF"],actual_tool_session_id=70445)
science_records={complete_ref["path"]:science_complete,wp:science_wait,op:science_outer}
c=SimpleNamespace(require=ns["require"],read=lambda r:science_records[r["path"]],PARENT="american_c3_val_000",MANIFEST=science_wait["adapter_manifest_ref"]["path"],ref=lambda p:science_wait["adapter_manifest_ref"])
sci_ns=dict(c=c);isolate(science_tree,{"_bundle_v1"},sci_ns)
bundle=dict(completion_ref=complete_ref,wait_ref=ns["NATIVE_RESOURCE_WAIT_REF"],outer_terminal_ref=ns["NATIVE_RESOURCE_OUTER_REF"])
try:sci_ns["_bundle_v1"](bundle,"contrastive")
except ValueError:check("scientific_gate_still_rejects_actual_operational_failure",True)
else:raise AssertionError("science failure")
science_wait["operational_failure"]=None;science_outer["actual_exit_code"]=1
try:sci_ns["_bundle_v1"](bundle,"contrastive")
except ValueError:check("scientific_gate_still_rejects_outer1",True)
else:raise AssertionError("science outer")
science_outer["actual_exit_code"]=0
check("synthetic_success_scientific_predicate_control",sci_ns["_bundle_v1"](bundle,"contrastive")[1] is science_wait)
# Original64 role validator and genuine CPU UNKNOWN dictionary regressions.
ns["ref"]=lambda p:dict(path=p,sha256="f"*64)
rows=[dict(path="RAM/ref_%02d.json"%i,sha256="f"*64) for i in range(64)]
check("exact64_roles_accepted",len(ns["unique"](rows))==64)
try:ns["unique"](rows+[dict(path="RAM/extra.json",sha256="f"*64)])
except ValueError:check("role65_rejected",True)
else:raise AssertionError("role65")
cpu=jread("research/reviews/H058_CPU_ROOT_stop_for_GPU_20261017_v1/prior_CPU_disposition.root_active_v1.json")
cpu_require=next(x for x in trees["common_v1.py"].body if isinstance(x,ast.FunctionDef) and x.name=="validate_request_v1")
cpu_expr=next(x.value.args[0] for x in cpu_require.body if isinstance(x,ast.Expr) and isinstance(x.value,ast.Call) and len(x.value.args)>1 and isinstance(x.value.args[1],ast.Constant) and x.value.args[1].value=="ROOT must account for actual CPU outcome before new complete GPU fit")
cpu_predicate_value=eval(compile(ast.Expression(cpu_expr),"<isolated_CPU_disposition_predicate>","eval"),dict(prior=cpu,q=dict(job_id=cpu["new_gpu_request_job_id"])))
check("CPU_UNKNOWN_final_totals_accepted_unchanged_predicate",bool(cpu_predicate_value))
check("CPU_real_lowerbounds_remain_1plus1",cpu["actual_optimizer_steps"]=={"same_marker_aligned":None,"same_marker_alignment_null":None} and cpu["verified_optimizer_step_lower_bounds"]=={"same_marker_aligned":1,"same_marker_alignment_null":1})
for n,h in expected.items():check("end_recheck_seal_"+n,exact(TARGET+"/"+n)["sha256"]==h)
result=dict(schema="H058_direct_GPU0_independent_AST_RAM_checks_v1",status="PASS",count=len(results),checks=results,
 target_source_ref=exact(TARGET+"/root_driver_v1.py"),target_manifest_ref=exact(TARGET+"/source_manifest.v1.json"),
 target_handoff_ref=exact(TARGET+"/source_handoff.v3.json"),scope="Seven-file source/metadata hashes and isolated AST/RAM predicates only",
 production_module_imports=False,real_resource_or_guard_operations=False,numerical_or_data_payload_reads=False,
 actual_admission_or_execution=False,semantic_mutations_in_RAM_ignore_authentication_for_predicate_coverage=True)
with (OUT/"checks.result.v1.json").open("x") as f:json.dump(result,f,sort_keys=True,indent=2);f.write("\n")
print(json.dumps(dict(status=result["status"],count=result["count"],result_ref=exact(str((OUT/"checks.result.v1.json").relative_to(ROOT)))),sort_keys=True))
