from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,traceback,os
from research.reviews.American_C3_VAL003004_native14v3_preparation_source_20261016_v2 import root_contract_v1 as c
root=c.ROOT
out=root/"research/reviews/VAL003_admission_partial_audit_20261016_v1"
report={"schema":"VAL003_partial_admission_readonly_audit_v1","started_utc":datetime.now(timezone.utc).isoformat(),"accounting_mutated":False,"scientific_execution":False,"steps":[]}
def mark(step,**kw):
 report["steps"].append(dict(step=step,utc=datetime.now(timezone.utc).isoformat(),**kw))
 print(json.dumps(report["steps"][-1],sort_keys=True),flush=True)
def rf(path):
 p=root/path;b=p.read_bytes();return dict(path=path,sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
packet=c.REL+"/admitted_val003_infer_v2"
authority_ref=rf(packet+"/reserve_authority.json");authority_ref.pop("bytes")
request_path=c.REL+"/request_val003_infer.root_active_v1.json"
q=json.loads((root/request_path).read_bytes())
journal=root/".runtime/storage/native14_stopped_retention_v3"
def journal_snapshot():return [rf(str(p.relative_to(root))) for p in sorted(journal.iterdir())]
report["journal_before"]=journal_snapshot()
try:
 c.bind_policy(q["policy_ref"]);mark("actual_policy_bound")
 g=c.original_guard();implementation=c.ref(g.budget.SOURCE_REL);launcher=c.ref(g.budget.LAUNCHER_REL)
 mark("original_guard_configured")
 with g.budget.v4.mutex(root):
  state=g._state_locked(implementation["sha256"],launcher["sha256"],fresh=True)
  report["state"]=dict(projected_bytes=state[2]["projected_bytes"],new_reserved_bytes=state[2]["new_reserved_bytes"],completed_credit_bytes=state[2]["completed_credit_bytes"],pins=g._pins(state[2]),direct_binding_count=len(state[3]),key_reserved=authority_ref is not None and json.loads((root/authority_ref["path"]).read_bytes())["request"]["key"] in state[0],key_closed=json.loads((root/authority_ref["path"]).read_bytes())["request"]["key"] in state[1])
  mark("single_exact_locked_current_state_read",**report["state"])
  auth=g._authority(authority_ref,"reserve",state[3],fresh=True);mark("same_authority_check_passed")
  g.require(auth["expected_before"]==g._pins(state[2]),"Fresh exact admission prefix required");mark("same_expected_prefix_check_passed")
  own=auth["request"];paths=g._paths(own);mark("same_paths_check_passed",paths=[str(p) for p in paths])
  g.require(all(r in own["consumer_bindings"] for r in state[3].values()),"Every actual guard/policy/marker source required");mark("same_bindings_check_passed")
  g.require(own["key"] not in state[0] and own["key"] not in g._policy()["never_credit_keys"] and not any(p.exists() for p in paths),"New key and fresh exclusive paths");mark("same_key_and_existing_path_check_passed")
  legacy=g.budget.v4.read_json(root,g.budget.v4.LEGACY)[0]
  occupied=[g.budget.v4.canonical(root,r["output"]) for r in legacy["jobs"].values()]
  for reservation in state[0].values():occupied.extend(g._paths(reservation))
  g.require(own["key"] not in legacy["jobs"] and not any(g.budget.v4.paths_overlap(a,b) for a in paths for b in occupied),"Never overlap/reuse old, closed or new reservations");mark("same_overlap_check_passed")
  projected=g.charge(g._policy()["base_projected_bytes"],state[2]["new_reserved_bytes"]+own["reserved_bytes"],state[2]["completed_credit_bytes"],len(state[4])+1)
  mark("all_preappend_checks_passed",proposed_projected_bytes=projected)
 report["verdict"]="ALL_CURRENT_PREAPPEND_CHECKS_PASS"
except BaseException as e:
 report["verdict"]="CURRENT_PREAPPEND_REJECTION"
 report["exception"]=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc())
 print(report["exception"]["traceback"],flush=True)
finally:
 report["journal_after"]=journal_snapshot();report["journal_unchanged"]=report["journal_before"]==report["journal_after"]
 report["finished_utc"]=datetime.now(timezone.utc).isoformat()
 (out/"read_only_preappend_audit.v1.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
print(json.dumps(dict(verdict=report["verdict"],journal_unchanged=report["journal_unchanged"]),sort_keys=True),flush=True)
