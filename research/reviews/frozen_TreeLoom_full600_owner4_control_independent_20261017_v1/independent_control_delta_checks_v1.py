"""Narrow independent owner4 control delta review. No science or runtime tests."""
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone,timedelta
import ast,copy,hashlib,json,math,re
BASE=Path("research/reviews/frozen_TreeLoom_bounded_training_source_20261017_v1")
OLD=BASE/"full600_ROOT_continuation_v1.py";NEW=BASE/"full600_ROOT_continuation_owner4_v1.py"
MANIFEST=BASE/"full600_ROOT_continuation_owner4_manifest.v1.json"
checks=[]
def check(ok,label):
 if not ok:raise AssertionError(label)
 checks.append(label)
def need(ok,label):
 if not ok:raise ValueError(label)
def reject(fn,label):
 try:fn()
 except(ValueError,KeyError,TypeError):checks.append(label);return
 raise AssertionError("Invalid control metadata accepted: "+label)
def ref(p):
 p=Path(p);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def rr(p):return dict(path=p,sha256=hashlib.sha256(p.encode()).hexdigest())
check(ref(NEW)["sha256"]=="31a893670cf041798564d96caad20f58bedc12ee08b6e8bc818bd1cb12249021","Exact sealed new control source")
check(ref(MANIFEST)["sha256"]=="9ea9e98c22854d7cbcf29ab99e5f14203dd21eda4cb12c7ddb4eeb50b723bca7","Exact sealed new control manifest")
check(ref(OLD)["sha256"]=="237a1d99d6ac18bcc4972644a0b5eaa9f30a3ebcbd2ab65640efb27c97a189ac","Exact previously passed continuation source")
old_ast=ast.parse(OLD.read_text());new_ast=ast.parse(NEW.read_text())
old_funcs={n.name:ast.dump(n,include_attributes=False)for n in old_ast.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
new_funcs={n.name:ast.dump(n,include_attributes=False)for n in new_ast.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
changed=[k for k in old_funcs if old_funcs[k]!=new_funcs.get(k)]
check(changed==["_request","_launch","_controls"],"Only three prior control functions changed")
check(set(new_funcs)-set(old_funcs)=={"_owner4"}and not(set(old_funcs)-set(new_funcs)),"Only owner4 metadata validator added")
for name in["_qualification","_all20","_wait","_admit","_remaining_fits","continue_v1","main"]:
 check(old_funcs[name]==new_funcs[name],"Previously passed core unchanged "+name)
old_imports=[ast.dump(n)for n in old_ast.body if isinstance(n,(ast.Import,ast.ImportFrom))]
new_imports=[ast.dump(n)for n in new_ast.body if isinstance(n,(ast.Import,ast.ImportFrom))]
check(old_imports==new_imports,"No scientific/runtime imports added")
m=json.loads(MANIFEST.read_text())
prior=m["original_continuation_independent_review_ref"];review=json.loads(Path(prior["path"]).read_text())
check(ref(prior["path"])==prior and review["independent"]is True and review["verdict"]=="SOURCE_PASS"and review["blockers"]==[],
 "Inherited actual independent SOURCE_PASS authenticated")
oldm=json.loads(Path(m["original_continuation_manifest_ref"]["path"]).read_text())
check(m["trainer_manifest_ref"]==oldm["trainer_manifest_ref"],"Same exact scientific manifest")
# Extract only changed metadata functions/constants into an explicit RAM boundary.
ns=dict(datetime=datetime,timezone=timezone,timedelta=timedelta,math=math,re=re)
ns["HERE"]=str(BASE)
for n in new_ast.body:
 if isinstance(n,ast.Assign)and all(isinstance(v,ast.Name)for v in n.targets):
  keys=[v.id for v in n.targets]
  if any(k in["SOURCE","MANIFEST","REVIEW","POLICY","SELECTOR","ENDPOINT","COORDINATOR","OWNER4_SOURCE","OWNER4_MANIFEST","OWNER4_REVIEW","FAILED_QUALIFICATION"]for k in keys):
   exec(compile(ast.Module(body=[n],type_ignores=[]),"control-constants#RAM","exec"),ns)
ram={};raw_calls=[]
def read(record):
 if record["path"].startswith("ram/"):return copy.deepcopy(ram[record["path"]])
 p=Path(record["path"]);need(ref(p)==record,"Real pinned source/metadata ref");return json.loads(p.read_text())
c=SimpleNamespace(read=read,ref=lambda p:rr(p)if str(p).startswith("ram/")else ref(p),raw=lambda r:raw_calls.append(r),
 active=lambda v:None,require=need,local=lambda p:Path("/workspace-SR003.nfs2/ARG-GNN")/p,
 ROOT=Path("/workspace-SR003.nfs2/ARG-GNN"),MANIFEST=str(BASE/"source_manifest.v1.json"),
 SOURCE="ram/original_worker.py",GPU0_UUID="GPU-ff06ca1f-403a-f2d0-b584-71e177452667")
ns["c"]=c;ns["ReadyOnly"]=type("ReadyOnly",(ValueError,),{})
for name in["_request","_owner4","_launch","_controls"]:
 node=next(n for n in new_ast.body if isinstance(n,ast.FunctionDef)and n.name==name)
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(NEW)+"#narrow-RAM","exec"),ns)
now=datetime.now(timezone.utc);end=now+timedelta(days=2)
q=json.loads((BASE/"ROOT_full600_owner4_continuation_request.INACTIVE.v1.json").read_text())
q.update(created_at_utc=now.isoformat(),expires_at_utc=end.isoformat(),wait_deadline_at_utc=(now+timedelta(days=1)).isoformat(),
 instance="ram_owner4",automatic_single_parent_launch_authorized=True,poll_seconds=30,
 qualification_input_ref=rr("ram/fresh_qualification_input"),accounting_batch_input_ref=rr("ram/current_simcontrol"))
for key in["future_qualification_acceptance_path","future_all20_readiness_path","future_all20_truth_handles_path","future_launch_authority_path"]:
 q[key]="research/reviews/RAM_only/"+key+".json"
for key,value in[("continuation_manifest_ref",ref(MANIFEST)),("source_manifest_ref",m["trainer_manifest_ref"]),
 ("source_review_ref",ns["REVIEW"]),("storage_policy_selector_ref",ns["SELECTOR"]),
 ("accounting_coordinator_source_ref",ns["COORDINATOR"]),("owner4_service_manifest_ref",ns["OWNER4_MANIFEST"])]:
 q[key]=value
q["automatic_retry_or_resume"]=False;q["all_other_GPU0_borrower_routes_revoked"]=True;q["preserve_controller_until_owned_drain"]=True
bindingref=rr("ram/qualification_binding")
ram["ram/fresh_qualification_input"]=dict(phase="qualification",source_manifest_ref=q["source_manifest_ref"],controller_binding_ref=bindingref)
ram[bindingref["path"]]=dict(owner4_service_manifest_ref=ns["OWNER4_MANIFEST"],owner4_service_source_ref=ns["OWNER4_SOURCE"])
qr=rr("ram/request");ram[qr["path"]]=q
check(ns["_request"](qr)[0]==q,"Fresh owner4 qualification accepted by actual request metadata function")
for label,where,key,value in[
 ("Reject failed original qualification",q,"qualification_input_ref",ns["FAILED_QUALIFICATION"]),
 ("Reject v3 owner manifest in request",q,"owner4_service_manifest_ref",rr("ram/oldmanifest")),
 ("Reject qualification wrong phase",ram["ram/fresh_qualification_input"],"phase","full_training"),
 ("Reject qualification wrong trainer",ram["ram/fresh_qualification_input"],"source_manifest_ref",rr("ram/wrongtrainer")),
 ("Reject qualification wrong service",ram[bindingref["path"]],"owner4_service_source_ref",ns["ENDPOINT"])]:
 old=copy.deepcopy(where[key]);where[key]=value;ram[qr["path"]]=q
 reject(lambda:ns["_request"](qr),label);where[key]=old
ram[qr["path"]]=q;q["_request_ref"]=qr
ownerref=rr("ram/owner4");readyref=rr("ram/ready4")
owner=dict(source_manifest_ref=ns["OWNER4_MANIFEST"],source_review_ref=ns["OWNER4_REVIEW"],
 source_refs={ns["OWNER4_SOURCE"]["path"]:ns["OWNER4_SOURCE"],ns["ENDPOINT"]["path"]:ns["ENDPOINT"]},
 jobs={},physical_gpu=0,gpu_uuid=c.GPU0_UUID,protected_envelope_bytes=0,storage_policy_selector_ref=ns["SELECTOR"],
 capacity_policy_ref=ns["POLICY"],expires_unix=(now+timedelta(hours=6)).timestamp())
status=dict(controller=dict(pid=1234,start_ticks=100),capability_ref=ownerref,jobs=[],lifecycle="running",safety_error=None,
 ambiguous_descendants_or_lease_holders=[],owner_service_revision=4,periodic_registry_writes=False,periodic_accounting_transactions=False)
ready=dict(root_authorizer="root_agent",actual_status_terminal_consumed=True,actual_status_tool_exit_code=0,
 controller_capability_ref=ownerref,controller_pid=1234,controller_start_ticks=100,source_ref=ns["OWNER4_SOURCE"],
 owner_parent_still_live=True,actual_status=status)
runtime_ref=rr("ram/runtime");rtq_ref=rr("ram/runtimequal")
ram[runtime_ref["path"]]=dict(source_ref=rr(c.SOURCE),CUDA_VISIBLE_DEVICES=c.GPU0_UUID,python="ram/python")
ram[rtq_ref["path"]]=dict(cpu_affinity=list(range(8)),python="ram/python",ROOT_qualified_existing_Torch_PyG_runtime=True,runtime_ref=runtime_ref)
qualification=dict(acceptance_ref=rr("ram/accepted_qualification"),acceptance=dict(completed_ref=rr("ram/completed")),
 completed=dict(elapsed_seconds=50.,peak_cuda_reserved_bytes=2**30,peak_cpu_RSS_bytes=2**30))
data=dict(readiness_ref=rr("ram/all20"),truth_handles_ref=rr("ram/truth_handles"))
launch=dict(schema="ROOT_TreeLoom_full600_launch_authority_v1",continuation_request_ref=qr,source_manifest_ref=q["source_manifest_ref"],
 qualification_acceptance_ref=qualification["acceptance_ref"],all20_readiness_ref=data["readiness_ref"],all20_truth_handles_ref=data["truth_handles_ref"],
 automatic_single_parent_launch_authorized=True,all_other_GPU0_borrower_routes_revoked=True,preserve_controller_until_owned_drain=True,
 qualification_weights_optimizer_or_rng_reused=False,controller_capability_ref=ownerref,controller_ready_ref=readyref,
 runtime_ref=runtime_ref,runtime_qualification_ref=rtq_ref,max_seconds=2000,expires_at_utc=(now+timedelta(hours=1)).isoformat(),
 measured_runtime_resource_plan=dict(schema="ROOT_TreeLoom_full600_measured_runtime_resource_plan_v1",ROOT_approved=True,
 qualification_completed_ref=qualification["acceptance"]["completed_ref"],actual_qualification_elapsed_seconds=50.,
 actual_qualification_peak_cuda_reserved_bytes=2**30,actual_qualification_peak_cpu_RSS_bytes=2**30,
 all20_parent_dimensions_reviewed=True,length_only_extrapolation=False,persistent_preparation_or_tensor_cache_assumed=False,
 full_optimizer_updates=600,basis="RAM metadata fixture only",estimated_full600_work_seconds=1000.))
ram[ownerref["path"]]=owner;ram[readyref["path"]]=ready
got=ns["_launch"](rr("ram/launch"),launch,q,qualification,data)
check(got["owner"]==owner and got["ready"]==ready,"Owner4 genuine-ready shape accepted without changing finite runtime gates")
for label,where,key,value in[
 ("Reject old ready source pin",ready,"source_ref",ns["ENDPOINT"]),
 ("Reject old ready revision",status,"owner_service_revision",3),
 ("Reject periodic accounting owner",status,"periodic_accounting_transactions",True),
 ("Reject periodic registry owner",status,"periodic_registry_writes",True),
 ("Reject nonzero actual status terminal",ready,"actual_status_tool_exit_code",1),
 ("Reject different owner identity",ready,"controller_start_ticks",101),
 ("Reject missing v3 compatibility pin",owner,"source_refs",{ns["OWNER4_SOURCE"]["path"]:ns["OWNER4_SOURCE"]}),
 ("Reject different owner review",owner,"source_review_ref",rr("ram/wrongreview"))]:
 old=copy.deepcopy(where[key]);where[key]=value;ram[ownerref["path"]]=owner;ram[readyref["path"]]=ready
 reject(lambda:ns["_launch"](rr("ram/launch"),launch,q,qualification,data),label);where[key]=old
ram[ownerref["path"]]=owner;ram[readyref["path"]]=ready
got=ns["_launch"](rr("ram/launch"),launch,q,qualification,data)
binding,attachment=ns["_controls"](got,"RAM_fixed_job")
check(binding["endpoint_source_ref"]==ns["ENDPOINT"]and binding["owner4_service_source_ref"]==ns["OWNER4_SOURCE"],
 "Frozen borrower protocol pin retained alongside actual service pin")
check(binding["independent_source_review_ref"]==ns["OWNER4_REVIEW"],"Binding attributes review to exact executed owner4 service")
check(binding["owner4_service_manifest_ref"]==ns["OWNER4_MANIFEST"],"Binding exposes exact owner4 manifest to fresh qualification/full600")
check(attachment["downstream_max_cuda_reserved_bytes"]==64*2**30 and attachment["downstream_non_Torch_margin_bytes"]==2*2**30
 and attachment["headroom_bytes"]==4*2**30,"Original borrowed64+2+4GiB limits unchanged")
result=dict(schema="TreeLoom_full600_owner4_independent_control_delta_checks_v1",passed=True,check_count=len(checks),checks=checks,
 narrow_control_delta_only=True,author_fixtures_reused=False,scientific_core_retested=False,actual_guard_imported=False,
 actual_resource_process_socket_lock_GPU_probes=False,actual_operations=False,scientific_execution=False,
 future_qualification_owner_ready_resource_values_are_explicit_RAM_metadata=True)
print(json.dumps(result,sort_keys=True,indent=2))
