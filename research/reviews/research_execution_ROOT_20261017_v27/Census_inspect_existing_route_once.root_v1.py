"""ROOT one-shot orchestration of the existing reviewed exact Census inspect retention API."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib, importlib, json, os, sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
OUT=ROOT/"research/reviews/Census_observation_ROOT_actual_inspect_retention_20261017_v1"
SOURCE="research/reviews/Census_observation_two_scope_retention_source_20261017_v1/census_observation_retention_v1.py"
SHA="72f6dc51844b2a261b581f5e69bd2a22982e6244c059f24549747a7b851faeeb"
SELECTOR={"path":"research/reviews/Census_observation_ROOT_activation_20261017_v1/explicit_child_selector.root_active_v1.json","sha256":"fd05d8934ed29eed39f7e60121a33e48fbeb0106991e76690664ca96355692e4"}
KEY="census_tsinfer_observation_v1/census-tsinfer-inspect-v1"
def ref(p):
 p=Path(p)
 return {"path":str(p.relative_to(ROOT)),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
def put(name,value):
 p=OUT/name
 with p.open("x") as f:
  json.dump(value,f,sort_keys=True,indent=2,allow_nan=False)
  f.write("\n")
  f.flush()
  os.fsync(f.fileno())
 return ref(p)
assert Path.cwd()==ROOT and sys.dont_write_bytecode and sys.flags.no_user_site
assert hashlib.sha256((ROOT/SOURCE).read_bytes()).hexdigest()==SHA
assert ref(ROOT/SELECTOR["path"])==SELECTOR
assert not OUT.exists()
OUT.mkdir()
g=importlib.import_module(SOURCE[:-3].replace("/","."))
assert g.load_selected_guard_v1(SELECTOR) is g
policy=g._policy()
impl=policy["implementation_ref"]["sha256"]
launcher=policy["launcher_ref"]["sha256"]
target=g._target(KEY)
put("ROOT_instruction.root_active_v1.json",{
 "schema":"ROOT_one_exact_Census_inspect_retention_instruction_v1","active":True,"root_authorizer":"root_agent",
 "key":KEY,"source_ref":{"path":SOURCE,"sha256":SHA},"selector_ref":SELECTOR,
 "independent_source_review_ref":policy["source_review_ref"],"exact_scopes":target["scope_paths"],
 "use_existing_reviewed_APIs_only":True,"source_scientific_and_clock_changes":False,
 "delete_rename_normalize_authorized":False,"original_waits_and_historical_drain_unknown_preserved":True,
 "all_other_jobs_unchanged":True,"ROOT_accepts_exact_cooperative_revocation_residual":True})
# The source's original mutex is released before its apply function takes it.
with g.budget.v4.mutex(ROOT):
 state=g._state_locked(impl,launcher,fresh=True)
 assert KEY in state[0] and KEY not in state[1]
 assert state[0][KEY]==target["original_reservation"]
 put("fresh_capture.root_v1.json",{"schema":"ROOT_Census_exact_pre_retention_capture_v1",
  "key":KEY,"pins":g._pins(state[2]),"projected_bytes":state[2]["projected_bytes"],
  "guard_roles":state[3],"utc":datetime.now(timezone.utc).isoformat()})
 # Root observations are intentionally made after potentially slow fresh-state setup.
 rev=g.stopped_revocation_draft_v1(KEY)
 rev.update(active=True,revoked_at_utc=datetime.now(timezone.utc).isoformat())
 revref=put("permanent_writer_revocation.root_active_v1.json",rev)
 facts=g._observe_targets(target)
 obs=g.stopped_observation_draft_v1(KEY,revocation_ref=revref)
 obs.update(active=True,observed_at_utc=datetime.now(timezone.utc).isoformat(),**facts)
 obsref=put("known_target_observation.root_active_v1.json",obs)
 request={"key":KEY,"reservation_sha256":target["original_reservation_sha256"],
  "revocation_ref":revref,"observation_ref":obsref}
 authority=g.operation_draft_v1(g.RETIRE_OPERATION,request,expected_before=g._pins(state[2]),bindings=state[3])
 authority.update(active=True,expires_at_utc=(datetime.now(timezone.utc)+timedelta(minutes=30)).isoformat())
 authority_ref=put("authority.root_active_v1.json",authority)
 roles=g.consumer_bindings_v1(list(state[3].values()),[revref,obsref,authority_ref])
 assert len(roles)<=64
result=g.apply_retire_stopped_v1(authority_ref,expected_source_sha256=impl,expected_launcher_sha256=launcher)
assert result["key"]==KEY and result["artifacts_deleted"]==0 and result["scientific_success_asserted"] is False
receipt=put("actual_retention_receipt.root_v1.json",result)
print(json.dumps({"receipt_ref":receipt,"retained_bytes":result["retained_bytes"],"credit_bytes":result["credit_bytes"],
 "projected_bytes":result["projected_bytes"],"actual_outer_terminal_still_requires_ROOT_consumption":True}),flush=True)
