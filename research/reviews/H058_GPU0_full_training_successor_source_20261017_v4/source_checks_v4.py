"""Conditional patch checks using AST and RAM stubs only."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from types import SimpleNamespace
import ast,copy,hashlib,json,errno
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
D="research/reviews/H058_GPU0_full_training_successor_source_20261017_v4"
B="research/reviews/H058_GPU0_full_training_successor_source_20261017_v3"
OUT=ROOT/D
rows=[]
def check(name,ok):
 if not ok:raise AssertionError(name)
 rows.append(dict(name=name,status="PASS"))
def text(p):return (ROOT/p).read_text()
def jread(p):return json.loads(text(p))
def ref(p):return dict(path=p,sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest())
def normalized(node):
 return ast.dump(node,include_attributes=False).replace("_20261017_v4","_20261017_v3").replace("H058_GPU0_v4_","H058_GPU0_v3_").replace("H058-GPU0-v4-","H058-GPU0-v3-")
def fn(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
common=ast.parse(text(D+"/common_v1.py"));direct=ast.parse(text(D+"/direct_gpu0_v1.py"))
basecommon=ast.parse(text(B+"/common_v1.py"));basedirect=ast.parse(text(B+"/direct_gpu0_v1.py"))
check("base_manifest_sealed",ref(B+"/source_manifest.v1.json")["sha256"]=="1c8e0efc286d98cf9fffeceaac640709091f9aef98a60114a512ff4c305b9354")
for candidate,base,names in ((common,basecommon,{"validate_request_v1","validate_input_v1"}),(direct,basedirect,{"controller_v1","capacity_v1","install_before_torch_v1"})):
 left=copy.deepcopy(candidate);right=copy.deepcopy(base)
 left.body=[x for x in left.body if not(isinstance(x,ast.FunctionDef) and x.name in names)]
 right.body=[x for x in right.body if not(isinstance(x,ast.FunctionDef) and x.name in names)]
 check("other_functions_unchanged_"+str(sorted(names)),normalized(left)==normalized(right))
# Restore only the six intended AST edits, then compare complete source ASTs.
x=copy.deepcopy(common)
r=fn(x,"validate_request_v1");r.args.kwonlyargs=[];r.args.kw_defaults=[];r.body.pop(0)
gate=next(n for n in r.body if isinstance(n,ast.If) and any(isinstance(z,ast.Call) and isinstance(z.func,ast.Name) and z.func.id=="validate_native_resource_disposition_v1" for z in ast.walk(n)))
next(z for z in ast.walk(gate) if isinstance(z,ast.keyword) and z.arg=="fresh").value=ast.Constant(value=True)
inp=fn(x,"validate_input_v1")
for z in ast.walk(inp):
 if isinstance(z,ast.Call) and isinstance(z.func,ast.Name) and z.func.id=="validate_request_v1":z.keywords=[]
check("common_only_intended_mode_and_forwarding_change",normalized(x)==normalized(basecommon))
x=copy.deepcopy(direct)
controller=fn(x,"controller_v1")
controller.body=[n for n in controller.body if not(isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=="group")]
capacity=fn(x,"capacity_v1")
t=capacity.body[1] if isinstance(capacity.body[1],ast.Try) else next(n for n in capacity.body if isinstance(n,ast.Try))
t.body=[n for n in t.body if not(isinstance(n,ast.Assign) and any(isinstance(z,ast.Name) and z.id in ("native","native_pids") for z in n.targets)) and not(isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and len(n.value.args)>1 and isinstance(n.value.args[1],ast.Constant) and str(n.value.args[1].value).startswith("Prior native roster GPU0 PID contexts"))]
installer=fn(x,"install_before_torch_v1")
wrapped=next(n for n in installer.body if isinstance(n,ast.FunctionDef) and n.name=="validate_lease")
wrapped.body=[n for n in wrapped.body if not(isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=="capacity_v1")]
check("direct_only_group_context_and_existing_capacity_boundary_change",normalized(x)==normalized(basedirect))
# Isolated actual admission native branch plus unchanged actual resource validator.
now=datetime.now(timezone.utc)
class FixedDateTime(datetime):
 @classmethod
 def now(cls,tz=None):return now if tz else now.replace(tzinfo=None)
ns=dict(Path=Path,datetime=FixedDateTime,timezone=timezone,os=SimpleNamespace(getuid=lambda:1000))
for n in common.body:
 if isinstance(n,ast.Assign):exec(compile(ast.Module(body=[n],type_ignores=[]),"<constant>","exec"),ns)
for name in ("require","active","validate_native_resource_disposition_v1"):
 exec(compile(ast.Module(body=[fn(common,name)],type_ignores=[]),"<isolated_source>","exec"),ns)
vr=dict(path="RAM/resource.json",sha256="a"*64);ar=dict(path="RAM/absence.json",sha256="b"*64)
v=jread(D+"/native_resource_disposition.INACTIVE.template.v1.json");absence=jread(D+"/native_targeted_absence.INACTIVE.template.v1.json")
oldstamp=(now-timedelta(seconds=443.077285)).isoformat()
v.update(created_at_utc=(now-timedelta(minutes=10)).isoformat(),expires_at_utc=(now+timedelta(hours=26)).isoformat(),status="resources_released",H058_job_id="H058-GPU0-v4-fulltrain-root02_oct17",observed_at_utc=oldstamp,fresh_targeted_native_absence_ref=ar,no_native_GPU0_attachment_alive=True,all_old_native_launch_resume_write_routes_permanently_revoked=True)
absence.update(observed_at_utc=oldstamp,native_known_roots_and_contexts_absent=True,no_native_GPU0_attachment_alive=True,all_exact_native_identities_and_groups_absent=True,all_old_native_launch_resume_write_routes_permanently_revoked=True)
records={vr["path"]:v,ar["path"]:absence}
for key in ("NATIVE_ROOT_DISPOSITION_REF","NATIVE_RESOURCE_WAIT_REF","NATIVE_RESOURCE_OUTER_REF","NATIVE_RESOURCE_AMENDMENT_REF"):records[ns[key]["path"]]=jread(ns[key]["path"])
ns["read"]=lambda r:records[r["path"]];ns["ref"]=ref
request=copy.deepcopy(fn(common,"validate_request_v1"))
nativebranch=next(n for n in request.body if isinstance(n,ast.If) and any(isinstance(z,ast.Call) and isinstance(z.func,ast.Name) and z.func.id=="validate_native_resource_disposition_v1" for z in ast.walk(n)))
request.body=[request.body[0],ast.Assign(targets=[ast.Name(id="q",ctx=ast.Store())],value=ast.Name(id="record",ctx=ast.Load())),nativebranch,ast.Return(value=ast.Constant(value=True))]
ast.fix_missing_locations(request);exec(compile(ast.Module(body=[request],type_ignores=[]),"<exact_admission_native_branch>","exec"),ns)
q=dict(phase="train",job_id=v["H058_job_id"],native_resource_disposition_ref=vr)
try:ns["validate_request_v1"](q)
except ValueError:check("fresh_admission_still_rejects_443second_static_proof",True)
else:raise AssertionError("fresh admission")
check("admitted_input_accepts_aged_original_static_proof",ns["validate_request_v1"](q,admitted_input=True))
try:ns["validate_request_v1"](q,admitted_input=None)
except ValueError:check("nonboolean_mode_rejected",True)
else:raise AssertionError("mode")
v["all_old_native_launch_resume_write_routes_permanently_revoked"]=False
try:ns["validate_request_v1"](q,admitted_input=True)
except ValueError:check("admitted_input_keeps_permanent_revocation_requirement",True)
else:raise AssertionError("revocation")
v["all_old_native_launch_resume_write_routes_permanently_revoked"]=True;v["expires_at_utc"]=now.isoformat()
try:ns["validate_request_v1"](q,admitted_input=True)
except ValueError:check("admitted_input_keeps_finite_native_authority",True)
else:raise AssertionError("expiry")
v["expires_at_utc"]=(now+timedelta(hours=26)).isoformat()
# Exact input call must select admitted mode; actual admission call stays default.
inputcall=next(n for n in ast.walk(fn(common,"validate_input_v1")) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="validate_request_v1")
check("input_forwards_explicit_admitted_mode",len(inputcall.keywords)==1 and inputcall.keywords[0].arg=="admitted_input" and inputcall.keywords[0].value.value is True)
driver=ast.parse(text(B+"/root_driver_v1.py"));admitcall=next(n for n in ast.walk(fn(driver,"admit_v1")) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="validate_request_v1")
check("actual_admission_keeps_default_fresh_mode",admitcall.keywords==[])
# Exact new group loop exercised with RAM OS stub only.
group_loop=next(n for n in fn(direct,"controller_v1").body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=="group")
groupfn=ast.FunctionDef(name="group_boundary",args=ast.arguments(posonlyargs=[],args=[ast.arg(arg="native")],kwonlyargs=[],kw_defaults=[],defaults=[]),body=[copy.deepcopy(group_loop),ast.Return(value=ast.Constant(value=True))],decorator_list=[])
ast.fix_missing_locations(groupfn);groupns={}
exec(compile(ast.Module(body=[groupfn],type_ignores=[]),"<exact_targeted_group_loop>","exec"),groupns)
calls=[]
def absent(group,signal):
 calls.append((group,signal));raise ProcessLookupError(errno.ESRCH,"absent")
groupns["os"]=SimpleNamespace(killpg=absent)
check("exact_known_native_group_ESRCH_accepted",groupns["group_boundary"](v))
check("only_exact_group_zero_observation_requested",calls==[(3752065,0)])
groupns["os"].killpg=lambda group,signal:None
try:groupns["group_boundary"](v)
except ValueError:check("live_exact_native_group_rejected",True)
else:raise AssertionError("live group")
def denied(group,signal):raise PermissionError(errno.EACCES,"denied")
groupns["os"].killpg=denied
try:groupns["group_boundary"](v)
except PermissionError:check("unknown_group_permission_error_blocks",True)
else:raise AssertionError("unknown group")
# Exact new context branch reuses the already obtained full GPU inventory.
body=next(n for n in fn(direct,"capacity_v1").body if isinstance(n,ast.Try)).body
start=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and any(isinstance(z,ast.Name) and z.id=="native" for z in n.targets))
contextfn=ast.FunctionDef(name="context_boundary",args=ast.arguments(posonlyargs=[],args=[ast.arg(arg="gpu"),ast.arg(arg="q")],kwonlyargs=[],kw_defaults=[],defaults=[]),body=copy.deepcopy(body[start:start+3])+[ast.Return(value=ast.Constant(value=True))],decorator_list=[])
ast.fix_missing_locations(contextfn)
contextns=dict(require=ns["require"],validate_native_resource_disposition_v1=lambda q,fresh:v)
exec(compile(ast.Module(body=[contextfn],type_ignores=[]),"<exact_existing_inventory_native_context_branch>","exec"),contextns)
check("empty_current_GPU0_contexts_accepted",contextns["context_boundary"](dict(compute=[]),q))
check("unrelated_GPU0_context_preserved_as_external",contextns["context_boundary"](dict(compute=[dict(pid=999999)]),q))
try:contextns["context_boundary"](dict(compute=[dict(pid=v["native_process_identities"][0]["pid"])]),q)
except ValueError:check("any_prior_native_roster_GPU_context_blocks",True)
else:raise AssertionError("native context")
# Call order and no equality-changing dynamic lease result.
inner=next(n for n in fn(direct,"install_before_torch_v1").body if isinstance(n,ast.FunctionDef) and n.name=="validate_lease")
capcall=next(n for n in inner.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=="capacity_v1")
ret=next(n for n in inner.body if isinstance(n,ast.Return))
baseinner=next(n for n in fn(basedirect,"install_before_torch_v1").body if isinstance(n,ast.FunctionDef) and n.name=="validate_lease")
baseret=next(n for n in baseinner.body if isinstance(n,ast.Return))
check("live_capacity_precedes_wrapped_lease_return",capcall.lineno<ret.lineno)
check("lease_return_dict_unchanged_for_configure_fit_equality",normalized(ret)==normalized(baseret))
check("current_worker_pid_classifies_own_H058_context",any(k.arg=="worker_pid" and ast.unparse(k.value)=="os.getpid()" for k in capcall.value.keywords))
configure=fn(ast.parse(text("src/arggnn/biological_candidate.py")),"configure_fit")
leasecall=next(n for n in ast.walk(configure) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="validate_lease")
cuda_calls=[n for n in ast.walk(configure) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and ast.unparse(n.func).startswith("torch.cuda.")]
check("pinned_configure_fit_lease_boundary_precedes_firstCUDA",leasecall.lineno<min(n.lineno for n in cuda_calls))
attach=fn(direct,"attach_v1")
check("attach_uses_existing_capacity_function",any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="capacity_v1" for n in ast.walk(attach)))
check("capacity_calls_controller_without_recursive_validate_lease",not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="validate_lease" for n in ast.walk(fn(direct,"capacity_v1"))))
# Every unchanged source/resource/science/timing predicate is retained by AST restoration above.
check("actual_v3_source_manifest_still_unchanged",ref(B+"/source_manifest.v1.json")["sha256"]=="1c8e0efc286d98cf9fffeceaac640709091f9aef98a60114a512ff4c305b9354")

# Complete fresh seven-file closure and namespace checks.
NAMES=("__init__.py","common_v1.py","context_factory_v1.py","direct_gpu0_v1.py","observations_gpu_v1.py","producer_v1.py","root_driver_v1.py")
for n in NAMES:
 tree=ast.parse(text(D+"/"+n))
 if n not in ("common_v1.py","direct_gpu0_v1.py"):
  check("five_namespace_only_"+n,normalized(tree)==normalized(ast.parse(text(B+"/"+n))))
manifest=jread(D+"/source_manifest.v1.json")
check("aggregate_ref_count_271",len(manifest["refs"])==271 and len({r["path"] for r in manifest["refs"]})==271)
for r in manifest["refs"]:check("aggregate_hash_"+r["path"],ref(r["path"])==r)
check("aggregate_exact_source_sha_map",manifest["source_sha256"]=={r["path"]:r["sha256"] for r in manifest["refs"]})
for r in jread(B+"/source_manifest.v1.json")["refs"]:check("all_frozen_v3_refs_preserved_"+r["path"],r in manifest["refs"])
for r in jread(B+"/MANIFEST.v3.json")["files"]:check("frozen_v3_artifact_"+r["path"],ref(r["path"])["sha256"]==r["sha256"] and (ROOT/r["path"]).stat().st_size==r["bytes"])
def refs(value):
 if isinstance(value,dict):
  if set(value)=={"path","sha256"}:yield value
  else:
   for x in value.values():yield from refs(x)
 elif isinstance(value,list):
  for x in value:yield from refs(x)
for p in sorted((ROOT/D).glob("*INACTIVE*.json")):
 value=json.loads(p.read_bytes())
 for r in refs(value):check("template_hash_"+p.name+"_"+r["path"],ref(r["path"])==r)
for n in ("dependencies.v1.json","producer_manifest.v1.json","producer_handoff.v1.json","SOURCE_MAP.v4.json"):
 for r in refs(jread(D+"/"+n)):check("joined_ref_"+n+"_"+r["path"],ref(r["path"])==r)
qtemplate=jread(D+"/ROOT_request.INACTIVE.template.v1.json")
check("fresh_exact_root02_inactive_identity",qtemplate["job_id"]=="H058-GPU0-v4-fulltrain-root02_oct17" and qtemplate["instance_id"]=="root02_oct17" and qtemplate["status"]=="inactive" and qtemplate["execution_authorized"] is False and qtemplate["prior_H058_CPU_disposition_ref"]==ref("research/reviews/H058_GPU0_v3_ROOT_terminal_20261017_v1/prior_CPU_and_GPU_disposition_for_root02.root_active_v1.json"))
check("fresh_exact_root02_paths",qtemplate["preparation_directory"]=="research/reviews/H058_GPU0_v4_ROOT_preparation_root02_oct17" and qtemplate["packet_directory"]=="research/reviews/H058_GPU0_v4_ROOT_actual_root02_oct17" and qtemplate["output_directory"]=="experiments/results/H058_GPU0_v4_full_parent_root02_oct17")
prior=jread(D+"/prior_CPU_disposition.INACTIVE.template.v1.json")
check("prior_actual_GPU_cause_counters_not_invented",prior["prior_GPU_v3_actual_wait_ref"] is None and prior["prior_GPU_v3_actual_outer_terminal_ref"] is None and prior["prior_GPU_v3_actual_direct_worker_returncode"] is None and prior["prior_GPU_v3_actual_optimizer_steps"] is None and prior["new_full_GPU_budget_explicitly_authorized"] is False)
check("CPU_null_and_1plus1_preserved",prior["actual_optimizer_steps"]=={"same_marker_aligned":None,"same_marker_alignment_null":None} and prior["verified_optimizer_step_lower_bounds"]=={"same_marker_aligned":1,"same_marker_alignment_null":1})
check("original_full_budget_and_caps",qtemplate["caps"]["reservation_bytes"]==256*2**20 and qtemplate["caps"]["scientific_output_bytes"]==192*2**20 and qtemplate["caps"]["auxiliary_bytes"]==60*2**20 and qtemplate["caps"]["directory_and_accounting_allowance_bytes"]==4*2**20 and qtemplate["caps"]["max_cuda_reserved_bytes"]==28*2**30 and qtemplate["caps"]["C3_fixed_envelope_bytes"]==26*2**30 and qtemplate["caps"]["headroom_bytes"]==4*2**30 and prior["new_GPU_budget"]==1920 and prior["new_GPU_budget_per_arm"]==960)


actual_prior=jread(qtemplate["prior_H058_CPU_disposition_ref"]["path"])
check("actual_ROOT_prior_v4_job_budget_and_no_state_transfer",actual_prior["new_gpu_request_job_id"]==qtemplate["job_id"] and actual_prior["new_GPU_budget"]==1920 and actual_prior["new_GPU_budget_per_arm"]==960 and actual_prior["no_CPU_or_GPU_optimizer_state_transfer"] is True)
check("actual_ROOT_prior_preserves_CPU_unknown_and_verified_GPU_zero",actual_prior["actual_optimizer_steps"]=={"same_marker_aligned":None,"same_marker_alignment_null":None} and actual_prior["verified_optimizer_step_lower_bounds"]=={"same_marker_aligned":1,"same_marker_alignment_null":1} and actual_prior["prior_GPU_actual_optimizer_steps"]=={"same_marker_aligned":0,"same_marker_alignment_null":0})
cpu_expr=next(x.value.args[0] for x in fn(common,"validate_request_v1").body if isinstance(x,ast.Expr) and isinstance(x.value,ast.Call) and len(x.value.args)>1 and isinstance(x.value.args[1],ast.Constant) and x.value.args[1].value=="ROOT must account for actual CPU outcome before new complete GPU fit")
check("actual_new_ROOT_disposition_passes_unchanged_prior_predicate",bool(eval(compile(ast.Expression(cpu_expr),"<actual_ROOT_prior_predicate>","eval"),dict(prior=actual_prior,q=qtemplate))))

result=dict(schema="H058_GPU0_v4_complete_source_AST_RAM_tests_v1",status="PASS",count=len(rows),checks=rows,
 conditional_on_actual_terminal=True,actual_terminal_observed=False,production_imports=False,live_process_or_GPU_observations=False,
 actual_guard_mutex_or_resource_calls=False,source_only=True,source_manifest_ref=ref(D+"/source_manifest.v1.json"))
with (OUT/"checks.result.v1.json").open("x") as f:json.dump(result,f,sort_keys=True,indent=2);f.write("\n")
print(json.dumps(dict(status="PASS",count=len(rows),result_ref=ref(D+"/checks.result.v1.json")),sort_keys=True))
