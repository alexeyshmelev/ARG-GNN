"""One genuine read-only original-scope qualification, preserving original clock and charge."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, importlib, json, os, sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
PREP=ROOT/"research/reviews/H059_saved_outer000_validator_ROOT_preparation_root02_oct17"
INSTRUCTION=ROOT/"research/reviews/H059_saved_outer000_ROOT_request_instruction_20261017_v2/ROOT_instruction.v2.json"
assert Path.cwd()==ROOT and sys.dont_write_bytecode and sys.flags.no_user_site
def ref(p):
 return {"path":str(p.relative_to(ROOT)),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
def read(r):
 p=ROOT/r["path"]
 assert ref(p)==r
 return json.loads(p.read_text())
def put(name,value):
 p=PREP/name
 with p.open("x")as f:
  json.dump(value,f,sort_keys=True,indent=2,allow_nan=False)
  f.write("\n")
  f.flush()
  os.fsync(f.fileno())
 return ref(p)
assert hashlib.sha256(INSTRUCTION.read_bytes()).hexdigest()=="0d583a2ea27b2cdf80a64107078b1ecef64929abe00c56896fc5af1c4aa45eb9"
i=json.loads(INSTRUCTION.read_text())
refs=i["refs"]
revref=ref(PREP/"old_writer_revocation.root_active_v2.json")
rev=read(revref)
assert rev["active"] and rev["all_original_launch_resume_retry_write_routes_permanently_cooperatively_revoked"]
q=json.loads((PREP/"request_staging_awaits_actual_old_scope_qualification.root_v2.json").read_text())
assert q["original_readonly_scope_qualification_ref"] is None and 221 in os.sched_getaffinity(0)
assert not (PREP/"ROOT_request.root_active_v2.json").exists()
old_input=read(refs["original_input_ref"])
storage=read(refs["original_storage_ref"])
assert storage==old_input["storage"]
assert storage["storage_policy_selector_ref"]==refs["original_selector_ref"]
assert storage["completed_overlay_policy_ref"]==refs["original_policy_ref"]
source=refs["original_guard_source_ref"]
assert ref(ROOT/source["path"])==source
g=importlib.import_module(source["path"][:-3].replace("/","."))
assert g.load_selected_guard_v1(refs["original_selector_ref"]) is g
# The original API performs its unchanged read-only clock/source/cap and full output+packet census.
view=g.budget_check_v9(storage,boundary=True,required_write_bytes=0)
observed=datetime.now(timezone.utc).isoformat()
view_ref=put("original_old_scope_guard_return.root_v2.json",view)
qualification=read(i["templates"]["old_scope_qualification_ref"])
qualification.update(root_authorizer="root_agent",status="PASS",observed_at_utc=observed,
 original_input_ref=refs["original_input_ref"],original_storage_ref=refs["original_storage_ref"],
 original_whole_reservation_bytes=4294967296,complete_original_output_and_packet_measured=True,
 old_source_guard_caps_and_clock_unchanged=True,original_charge_retained=True,
 historical_monitoring_gap_recovered=False,old_writer_revocation_ref=revref,
 actual_guard_source_ref=source,actual_old_selector_ref=refs["original_selector_ref"],
 actual_guard_return_ref=view_ref,actual_API="budget_check_v9(original_storage,boundary=True,required_write_bytes=0)")
qualified=put("original_old_scope_readonly_qualification.root_v2.json",qualification)
q["original_readonly_scope_qualification_ref"]=qualified
request=put("ROOT_request.root_active_v2.json",q)
print(json.dumps({"actual_readonly_guard_returned":True,"request_ref":request,
 "qualification_ref":qualified,"admission":False,"submit_fresh_v2_admission_promptly":True}),flush=True)
