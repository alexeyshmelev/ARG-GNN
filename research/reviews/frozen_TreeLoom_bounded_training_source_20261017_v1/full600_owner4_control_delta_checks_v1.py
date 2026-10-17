"""Narrow source/RAM owner4 control-delta checks; no operations or broad suite."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import ast,copy,hashlib,importlib,json,traceback
def main():
 b=importlib.import_module("research.reviews.frozen_TreeLoom_bounded_training_source_20261017_v1.full600_ROOT_continuation_owner4_v1")
 c=b.c;p=Path(c.HERE);checks=[];assertions=[0]
 def check(ok,name):
  assertions[0]+=1
  if not ok:raise AssertionError(name)
  if name not in checks:checks.append(name)
 def ref(path):return dict(path=str(path),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
 source=(p/"full600_ROOT_continuation_owner4_v1.py").read_text();old=(p/"full600_ROOT_continuation_v1.py").read_text()
 oldtree=ast.parse(old);tree=ast.parse(source)
 oldfn={n.name:n for n in oldtree.body if isinstance(n,ast.FunctionDef)}
 newfn={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
 changed={name for name in oldfn if ast.dump(oldfn[name],include_attributes=False)!=ast.dump(newfn[name],include_attributes=False)}
 check(changed=={"_request","_launch","_controls"}and set(newfn)-set(oldfn)=={"_owner4"},"only-three-control-functions-and-one-owner-helper-changed")
 for name in ("_qualification","_all20","_wait","_admit","_remaining_fits","continue_v1","main"):
  check(ast.get_source_segment(old,oldfn[name])==ast.get_source_segment(source,newfn[name]),"original-qualification-all20-finite-runtime-accounting-and-single-exec-gates-byte-identical")
 check(b.main([])==0,"actual-noargs-inactive0")
 manifest=json.loads((p/"source_manifest.v1.json").read_text())
 check(ref(p/"source_manifest.v1.json")["sha256"]=="35760763a7f0608ed56cd62565ec555d4fcae7f622ea6bfb9dd5280257d20fa0","original-trainer-manifest-unchanged")
 for path,sha in manifest["source_sha256"].items():check(ref(path)["sha256"]==sha,"all85-original-scientific-source-metadata-pins-unchanged")
 check(ref(p/"full600_ROOT_continuation_v1.py")["sha256"]=="237a1d99d6ac18bcc4972644a0b5eaa9f30a3ebcbd2ab65640efb27c97a189ac","original-reviewed-continuation-source-unchanged")
 records={}
 def add(path,value):
  r=dict(path=path,sha256=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest());records[path]=(r,copy.deepcopy(value));return r
 realread=c.read;realraw=c.raw
 def read(r):
  if r["path"]not in records:return realread(r)
  rr,value=records[r["path"]]
  if rr!=r:raise ValueError("RAM immutable pin")
  return copy.deepcopy(value)
 def raw(r):
  if r["path"]in records:read(r);return b"RAM"
  return realraw(r)
 c.read=read;c.raw=raw
 now=datetime.now(timezone.utc)
 fresh_input=add("research/reviews/RAM/fresh_owner4_qualification_input.json",dict(phase="qualification",
  source_manifest_ref=ref(p/"source_manifest.v1.json"),controller_binding_ref=None))
 binding=add("research/reviews/RAM/fresh_owner4_binding.json",dict(owner4_service_manifest_ref=b.OWNER4_MANIFEST,owner4_service_source_ref=b.OWNER4_SOURCE))
 qi=read(fresh_input);qi["controller_binding_ref"]=binding;fresh_input=add(fresh_input["path"],qi)
 batch=add("research/reviews/RAM/batch.json",dict(RAM_only=True))
 q=json.loads((p/"ROOT_full600_owner4_continuation_request.INACTIVE.v1.json").read_text())
 q.update(status="active",template_only=False,execution_authorized=True,created_at_utc=now.isoformat(),
  expires_at_utc=(now+timedelta(hours=40)).isoformat(),wait_deadline_at_utc=(now+timedelta(minutes=5)).isoformat(),
  qualification_input_ref=fresh_input,accounting_batch_input_ref=batch,instance="ram_owner4",
  future_qualification_acceptance_path="research/reviews/RAM/fresh_acceptance.json",
  future_all20_readiness_path="research/reviews/RAM/all20_ready.json",future_all20_truth_handles_path="research/reviews/RAM/all20_truth.json",
  future_launch_authority_path="research/reviews/RAM/launch.json",automatic_single_parent_launch_authorized=True)
 qref=add("research/reviews/RAM/full600_request.json",q);parsed,_=b._request(qref)
 check(parsed["qualification_input_ref"]==fresh_input,"new-owner4-qualification-input-is-dynamic-and-accepted")
 def reject(call,name,kind=ValueError):
  try:call()
  except kind:check(True,name)
  else:raise AssertionError(name)
 bad=copy.deepcopy(q);bad["qualification_input_ref"]=b.FAILED_QUALIFICATION
 badref=add("research/reviews/RAM/bad_failed_request.json",bad)
 reject(lambda:b._request(badref),"old-failed-qualification-input-cannot-replay")
 badbinding=add("research/reviews/RAM/bad_binding.json",dict(owner4_service_manifest_ref=b.OWNER4_MANIFEST,owner4_service_source_ref=b.ENDPOINT))
 badinput=add("research/reviews/RAM/bad_input.json",dict(qi,controller_binding_ref=badbinding))
 bad=copy.deepcopy(q);bad["qualification_input_ref"]=badinput
 badref=add("research/reviews/RAM/bad_service_request.json",bad)
 reject(lambda:b._request(badref),"qualification-input-must-bind-exact-owner4-service")
 owner=dict(root_authorizer="root_agent",status="active",template_only=False,execution_authorized=True,
  created_at_utc=now.isoformat(),expires_at_utc=(now+timedelta(hours=12)).isoformat(),expires_unix=(now+timedelta(hours=11)).timestamp(),
  source_manifest_ref=b.OWNER4_MANIFEST,source_review_ref=b.OWNER4_REVIEW,
  source_refs={b.OWNER4_SOURCE["path"]:b.OWNER4_SOURCE,b.ENDPOINT["path"]:b.ENDPOINT},jobs={},physical_gpu=0,
  gpu_uuid=c.GPU0_UUID,protected_envelope_bytes=0,storage_policy_selector_ref=b.SELECTOR,capacity_policy_ref=b.POLICY)
 ownerref=add("research/reviews/RAM/owner4_capability.json",owner)
 status=dict(controller=dict(pid=123,start_ticks=456),capability_ref=ownerref,jobs=[],lifecycle="running",safety_error=None,
  ambiguous_descendants_or_lease_holders=[],owner_service_revision=4,periodic_registry_writes=False,periodic_accounting_transactions=False)
 ready=dict(root_authorizer="root_agent",actual_status_terminal_consumed=True,actual_status_tool_exit_code=0,
  controller_capability_ref=ownerref,controller_pid=123,controller_start_ticks=456,source_ref=b.OWNER4_SOURCE,
  owner_parent_still_live=True,actual_status=status)
 readyref=add("research/reviews/RAM/owner4_ready.json",ready)
 runtime=dict(source_ref=ref(c.SOURCE),CUDA_VISIBLE_DEVICES=c.GPU0_UUID,python=".runtime/env/bin/python3.11")
 runtimeref=add("research/reviews/RAM/runtime.json",runtime)
 rqual=add("research/reviews/RAM/runtime_qualification.json",dict(cpu_affinity=list(range(176,184)),python=runtime["python"],
  ROOT_qualified_existing_Torch_PyG_runtime=True,runtime_ref=runtimeref))
 done_ref=add("research/reviews/RAM/qual_done.json",dict(elapsed_seconds=123.,peak_cuda_reserved_bytes=32*2**30,peak_cpu_RSS_bytes=128*2**30))
 acceptance=add("research/reviews/RAM/qual_accepted.json",dict(completed_ref=done_ref))
 qualified=dict(acceptance_ref=acceptance,acceptance=read(acceptance),completed=read(done_ref))
 data=dict(readiness_ref=add("research/reviews/RAM/all20_ready.json",dict(RAM_only=True)),
  truth_handles_ref=add("research/reviews/RAM/all20_truth.json",dict(RAM_only=True)))
 q["_request_ref"]=qref
 launch=json.loads((p/"ROOT_full600_owner4_launch_authority.INACTIVE.v1.json").read_text())
 launch.update(status="active",template_only=False,execution_authorized=True,created_at_utc=now.isoformat(),
  expires_at_utc=(now+timedelta(hours=10)).isoformat(),continuation_request_ref=qref,qualification_acceptance_ref=acceptance,
  all20_readiness_ref=data["readiness_ref"],all20_truth_handles_ref=data["truth_handles_ref"],
  controller_capability_ref=ownerref,controller_ready_ref=readyref,runtime_ref=runtimeref,runtime_qualification_ref=rqual,
  max_seconds=28800,automatic_single_parent_launch_authorized=True)
 launch["measured_runtime_resource_plan"].update(ROOT_approved=True,qualification_completed_ref=done_ref,actual_qualification_elapsed_seconds=123.,
  actual_qualification_peak_cuda_reserved_bytes=32*2**30,actual_qualification_peak_cpu_RSS_bytes=128*2**30,
  all20_parent_dimensions_reviewed=True,estimated_full600_work_seconds=6000.,basis="RAM only")
 lr=add("research/reviews/RAM/launch.json",launch)
 out=b._launch(lr,launch,q,qualified,data)
 check(out["ready"]["source_ref"]==b.OWNER4_SOURCE,"genuine-owner4-ready-accepted-with-original-measured-window")
 bind,attach=b._controls(out,"RAM-full600")
 check(bind["endpoint_source_ref"]==b.ENDPOINT and bind["independent_source_review_ref"]==b.OWNER4_REVIEW
  and bind["owner4_service_manifest_ref"]==b.OWNER4_MANIFEST and bind["owner4_service_source_ref"]==b.OWNER4_SOURCE,
  "borrower-binds-executed-v3-protocol-and-independent-v4-service-provenance")
 bad=copy.deepcopy(ready);bad["source_ref"]=b.ENDPOINT;badready=add("research/reviews/RAM/v3_ready.json",bad)
 badlaunch=dict(launch,controller_ready_ref=badready)
 reject(lambda:b._launch(lr,badlaunch,q,qualified,data),"v3-ready-label-is-not-substituted-for-actual-v4-service")
 bad=copy.deepcopy(ready);bad["actual_status"]["periodic_accounting_transactions"]=True
 badready=add("research/reviews/RAM/periodic_ready.json",bad)
 reject(lambda:b._launch(lr,dict(launch,controller_ready_ref=badready),q,qualified,data),"periodic-accounting-service-rejected")
 bad=copy.deepcopy(owner);bad["source_refs"].pop(b.ENDPOINT["path"])
 reject(lambda:b._owner4(bad,q),"owner-without-actually-executed-v3-pin-rejected")
 bad=copy.deepcopy(launch);bad["measured_runtime_resource_plan"]["estimated_full600_work_seconds"]=28800.
 reject(lambda:b._launch(lr,bad,q,qualified,data),"insufficient-measured-window-still-returns-ready-only",b.ReadyOnly)
 c.read=realread;c.raw=realraw
 result=dict(schema="full600_owner4_control_delta_source_RAM_checks_v1",source_only=True,checks_passed=len(checks),assertions_passed=assertions[0],
  checks=checks,source_ref=ref(p/"full600_ROOT_continuation_owner4_v1.py"),manifest_ref=ref(p/"full600_ROOT_continuation_owner4_manifest.v1.json"),
  no_actual_guard_import_or_resource_probe_or_wait_or_admission_or_launch=True,scientific_modules_unchanged=True,
  limitations="Narrow changed-control checks only; unchanged inherited original independent full600 source pass. Actual qualification0, all20 readiness, measured feasibility and600updates remain ROOT evidence.")
 return result
if __name__=="__main__":
 try:print(json.dumps(main(),sort_keys=True))
 except BaseException:print(traceback.format_exc());raise SystemExit(1)
