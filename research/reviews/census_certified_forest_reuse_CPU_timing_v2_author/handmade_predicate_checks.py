
from pathlib import Path
import ast,copy,hashlib,json
from datetime import datetime,timezone
from functools import wraps
from types import SimpleNamespace
ROOT=Path.cwd()
DEST=ROOT/"research/reviews/census_certified_forest_reuse_CPU_timing_v2_author"
REL=DEST.relative_to(ROOT).as_posix()
OLD=ROOT/"research/reviews/census_certified_forest_reuse_v1_timing_author"
B=DEST/"cpu_resource_bridge_v2.py";C=DEST/"root_cpu_timing_recipe_v2.py";P=DEST/"protocol.json"
cfg=json.loads(P.read_bytes());cfgsha=hashlib.sha256(P.read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(value).hexdigest()
def dump(node):return ast.dump(node,include_attributes=False)
def fn(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
bt=ast.parse(B.read_text());ct=ast.parse(C.read_text())
nt=ast.parse((ROOT/"src/arggnn/census_native_qualification_runner_v1.py").read_text())
oldct=ast.parse((OLD/"root_cpu_timing_recipe.py").read_text())
oldf=fn(nt,"admit_runner_v1");newf=fn(bt,"admit_cpu_timing_v2")
def assignment(stmt,name):
 return isinstance(stmt,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in stmt.targets)
oi=next(i for i,n in enumerate(oldf.body) if assignment(n,"storage"))
ni=next(i for i,n in enumerate(newf.body) if assignment(n,"storage"))
oe=next(i for i,n in enumerate(oldf.body) if isinstance(n,ast.If) and "if launcher!=" in ast.get_source_segment((ROOT/"src/arggnn/census_native_qualification_runner_v1.py").read_text(),n))
ne=next(i for i,n in enumerate(newf.body) if isinstance(n,ast.If) and "if launcher!=" in ast.get_source_segment(B.read_text(),n))
assert [dump(n) for n in oldf.body[:oi]]==[dump(n) for n in newf.body[:ni]]
assert [dump(n) for n in oldf.body[oe:]]==[dump(n) for n in newf.body[ne:]]
oldlauncher=next(n for n in oldf.body if assignment(n,"launcher"))
newlauncher=next(n for n in newf.body if assignment(n,"launcher"))
assert dump(oldlauncher)==dump(newlauncher)
oldtry=next(n for n in fn(oldct,"main").body if isinstance(n,ast.Try))
newtry=next(n for n in fn(ct,"main").body if isinstance(n,ast.Try))
assert [dump(n) for n in oldtry.body]==[dump(n) for n in newtry.body]
assert [dump(n) for n in oldtry.handlers]==[dump(n) for n in newtry.handlers]
assert (DEST/"pins.json").read_bytes()==(OLD/"pins.json").read_bytes()
assert digest((OLD/"root_cpu_timing_recipe.py").read_bytes())==cfg["preserved_v1_caller_ref"]["sha256"]=="86a2febedd7973a9258b9b9674d388a673187aebbf96e1c32da35489d733be25"
assert digest((ROOT/"src/arggnn/census_native_qualification_runner_v1.py").read_bytes())==cfg["original_CPU_runner_ref"]["sha256"]=="a63c9481c934247fabe077aed8dabe02bb1cc831f7bbfe441f5974b397346140"
calls=[n for n in ast.walk(fn(bt,"require_storage_v2")) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=="check_reservation"]
assert len(calls)==1
call=calls[0];keywords={k.arg:dump(k.value) for k in call.keywords}
assert set(keywords)=={"expected_source_sha256","expected_config_sha256","initial_ledger_pins","boundary","required_write_bytes"}
assert keywords["boundary"]=="Constant(value=True)"
for forbidden in ("reserve","current","measure_scopes","legacy_state","run_fit_v1","observed_timing_v1"):
 assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr==forbidden for n in ast.walk(bt))
functions=[fn(bt,n) for n in ("load_bridge_config_v2","validate_account_v2","validate_storage_metadata_v2","sticky_authority_v2")]
ns={"Path":Path,"ROOT":ROOT,"datetime":datetime,"timezone":timezone,"hashlib":hashlib,"wraps":wraps,
 "CONFIG_PATH":P.relative_to(ROOT).as_posix(),"CONFIG_SHA256":cfgsha,
 "native":SimpleNamespace(_sha=lambda x:isinstance(x,str) and len(x)==64 and all(c in "0123456789abcdef" for c in x)),
 "_utc":lambda x:datetime.fromisoformat(x.replace("Z","+00:00")).astimezone(timezone.utc)}
exec(compile(ast.Module(body=functions,type_ignores=[]),"<stdlib handmade bridge predicates>","exec"),ns)
records=[]
def check(name,callback,deny=False):
 try:callback()
 except Exception as error:
  if not deny:raise
  records.append({"case":name,"status":"PASS_expected_rejection","error_type":type(error).__name__})
 else:
  if deny:raise AssertionError(name+" unexpectedly accepted")
  records.append({"case":name,"status":"PASS"})
class Reader:
 def __init__(self,data):self.data=data
 def relative(self,value):
  p=Path(value)
  if p.is_absolute():p=p.relative_to(ROOT)
  if ".." in p.parts:raise ValueError("Traversal")
  return p.as_posix()
 def read(self,ref):
  if set(ref)!={"path","sha256"}:raise ValueError("Exact ref")
  data=self.data[self.relative(ref["path"])]
  if digest(data)!=ref["sha256"]:raise ValueError("Frozen ref changed")
  return data
 def decode_json(self,ref):return json.loads(self.read(ref))
def encoded(v):return (json.dumps(v,sort_keys=True,indent=2)+"\n").encode()
def reference(path,data):return {"path":path,"sha256":digest(data)}
data={cfgkey["path"]:(ROOT/cfgkey["path"]).read_bytes() for cfgkey in
 [cfg[k] for k in ("root_source_registration_ref","human_authorization_ref","budget_source_ref","budget_config_ref",
 "original_CPU_runner_ref","preserved_v1_caller_ref","preserved_v1_pins_ref","accepted_consumer_pattern_source_ref","accepted_consumer_pattern_review_ref")]}
data[P.relative_to(ROOT).as_posix()]=P.read_bytes()
for path in cfg["executed_bridge_source_paths"]:data[path]=(ROOT/path).read_bytes()
reader=Reader(data)
check("exact_registered_source_config_and_user1TB_metadata",lambda:ns["load_bridge_config_v2"](reader))
for label,key in [("registration_SHA_changed","root_source_registration_ref"),("human_authorization_SHA_changed","human_authorization_ref"),
 ("budget_source_SHA_changed","budget_source_ref"),("budget_config_SHA_changed","budget_config_ref"),
 ("original_runner_SHA_changed","original_CPU_runner_ref"),("preserved_v1_caller_SHA_changed","preserved_v1_caller_ref")]:
 broken=dict(data);broken[cfg[key]["path"]]+=b" "
 check(label,lambda broken=broken:ns["load_bridge_config_v2"](Reader(broken)),True)
now=datetime(2026,10,11,6,0,tzinfo=timezone.utc)
runtime_ref={"path":"fixture/runtime.json","sha256":"a"*64}
admission={"job_id":"fixture-cpu-v2","created_at_utc":"2026-10-11T05:55:00+00:00","expires_at_utc":"2026-10-11T07:00:00+00:00",
 "output_directory":"fixture/output","refs":{"runtime":runtime_ref}}
sources={path:digest(data[path]) for path in cfg["executed_bridge_source_paths"]}
needed=[{"path":path,"sha256":value} for path,value in sources.items()]
needed+=[reference(P.relative_to(ROOT).as_posix(),P.read_bytes()),runtime_ref,cfg["budget_source_ref"],cfg["budget_config_ref"],cfg["human_authorization_ref"]]
review={"status":"PASS","review_role":"independent_reviewer","actual_admission":False,"fixture_mode":False,
 "artifact_sha256":dict(sources,**{P.relative_to(ROOT).as_posix():cfgsha})}
data["fixture/review.json"]=encoded(review)
storage={"schema":"census_CPU_timing_storage_v2","status":"active","job_id":"fixture-cpu-v2",
 "created_at_utc":"2026-10-11T05:55:00+00:00","expires_at_utc":"2026-10-11T07:00:00+00:00",
 "output_directory":"fixture/output","reserved_new_bytes":1_000_000_000,"mode":"project_budget_v4",
 "reservation":{"key":"fixture-cpu-v2","output":str(ROOT/"fixture/output"),"auxiliary_scopes":[],
 "reserved_bytes":1_000_000_000,"consumer_bindings":needed,
 "root_admission":{"path":"fixture/root.json","sha256":"c"*64}},
 "initial_ledger_pins":{"legacy_ledger_sha256":"a"*64,"supplemental_ledger_sha256":"b"*64},
 "bridge_config_ref":reference(P.relative_to(ROOT).as_posix(),P.read_bytes()),
 "review_ref":reference("fixture/review.json",data["fixture/review.json"])}
check("truthful_v4_storage_consumer_metadata_acceptance",lambda:ns["validate_storage_metadata_v2"](storage,admission,{"output_cap_bytes":1_000_000_000},cfg,Reader(data),now))
mutations=[
 ("inactive",lambda x:x.update(status="inactive")),
 ("wrong_job",lambda x:x.update(job_id="wrong")),
 ("wrong_output",lambda x:x.update(output_directory="fixture/wrong")),
 ("wrong_mode",lambda x:x.update(mode="legacy450")),
 ("underreserved_new",lambda x:x.update(reserved_new_bytes=999_999_999)),
 ("wrong_config_SHA",lambda x:x["bridge_config_ref"].update(sha256="0"*64)),
 ("wrong_reservation_key",lambda x:x["reservation"].update(key="wrong")),
 ("wrong_canonical_reservation_output",lambda x:x["reservation"].update(output=str(ROOT/"fixture/wrong"))),
 ("unexpected_auxiliary_scope",lambda x:x["reservation"].update(auxiliary_scopes=[str(ROOT/"fixture/extra")])),
 ("underreserved_scope",lambda x:x["reservation"].update(reserved_bytes=999_999_999)),
 ("missing_caller_binding",lambda x:x["reservation"]["consumer_bindings"].pop(1)),
 ("missing_bridge_binding",lambda x:x["reservation"]["consumer_bindings"].pop(0)),
 ("missing_runtime_binding",lambda x:x["reservation"]["consumer_bindings"].remove(runtime_ref)),
 ("missing_budget_source_binding",lambda x:x["reservation"]["consumer_bindings"].remove(cfg["budget_source_ref"])),
 ("missing_human_binding",lambda x:x["reservation"]["consumer_bindings"].remove(cfg["human_authorization_ref"])),
 ("malformed_initial_ledger_SHA",lambda x:x["initial_ledger_pins"].update(legacy_ledger_sha256="bad")),
 ("extra_initial_ledger_key",lambda x:x["initial_ledger_pins"].update(fake=0)),
 ("expired",lambda x:x.update(expires_at_utc="2026-10-11T06:00:00+00:00")),
 ("future",lambda x:x.update(created_at_utc="2026-10-11T06:01:00+00:00")),
 ("storage_older_than_base",lambda x:x.update(created_at_utc="2026-10-11T05:54:59+00:00")),
 ("storage_outlives_base",lambda x:x.update(expires_at_utc="2026-10-11T07:00:01+00:00")),
 ("extra_storage_field",lambda x:x.update(concurrent_reserve_bytes=0)),
 ("review_SHA_changed",lambda x:x["review_ref"].update(sha256="0"*64))]
for label,mutate in mutations:
 altered=copy.deepcopy(storage);mutate(altered)
 check("storage_"+label,lambda altered=altered:ns["validate_storage_metadata_v2"](altered,admission,{"output_cap_bytes":1_000_000_000},cfg,Reader(data),now),True)
for label,mutate in [
 ("not_PASS",lambda x:x.update(status="REJECT")),
 ("author_instead_of_independent",lambda x:x.update(review_role="author")),
 ("actual_admission_claim",lambda x:x.update(actual_admission=True)),
 ("fixture_mode_true",lambda x:x.update(fixture_mode=True)),
 ("source_hash_drift",lambda x:x["artifact_sha256"].update({cfg["executed_bridge_source_paths"][0]:"0"*64}))]:
 alteredreview=copy.deepcopy(review);mutate(alteredreview);altereddata=dict(data);altereddata["fixture/review.json"]=encoded(alteredreview)
 alteredstorage=copy.deepcopy(storage);alteredstorage["review_ref"]=reference("fixture/review.json",altereddata["fixture/review.json"])
 check("review_"+label,lambda:ns["validate_storage_metadata_v2"](alteredstorage,admission,{"output_cap_bytes":1_000_000_000},cfg,Reader(altereddata),now),True)
account={"baseline_bytes":167_336_640_512,"external_reserve_bytes":130_000_000_000,"legacy_reserved_bytes":150_000_000_000,
 "supplemental_reserved_bytes":420_153_359_488,"projected_bytes":867_490_000_000,
 "limit_bytes":950_000_000_000,"hard_limit_bytes":1_000_000_000_000,
 "reservation":copy.deepcopy(storage["reservation"]),"scope_measurement":{"conservative_scope_bytes":123_456},
 "legacy_ledger_mutated":False,"supplemental_ledger_mutated":False,
 "projection_is_historical_accounting_not_global_remeasurement":True}
check("genuine_simulated_combined867_49GB_accepted",lambda:ns["validate_account_v2"](account,storage))
check("same_forecast_would_reject_old450GB",lambda:(_ for _ in ()).throw(ValueError("old450GB gate")) if account["projected_bytes"]>=450_000_000_000 else None,True)
for label,mutate in [
 ("combined950GB_boundary",lambda x:x.update(supplemental_reserved_bytes=x["supplemental_reserved_bytes"]+82_510_000_000,projected_bytes=950_000_000_000)),
 ("combined1TB_boundary",lambda x:x.update(supplemental_reserved_bytes=x["supplemental_reserved_bytes"]+132_510_000_000,projected_bytes=1_000_000_000_000)),
 ("omitted_supplemental_charge",lambda x:x.update(supplemental_reserved_bytes=0)),
 ("missing_external_charge",lambda x:x.pop("external_reserve_bytes")),
 ("false_legacy_readonly",lambda x:x.update(legacy_ledger_mutated=True)),
 ("false_supplemental_readonly",lambda x:x.update(supplemental_ledger_mutated=True)),
 ("fake_ceiling",lambda x:x.update(limit_bytes=1_000_000_000_000)),
 ("fake_hard_limit",lambda x:x.update(hard_limit_bytes=2_000_000_000_000)),
 ("wrong_own_reservation",lambda x:x["reservation"].update(key="wrong")),
 ("negative_component",lambda x:x.update(legacy_reserved_bytes=-1)),
 ("noninteger_component",lambda x:x.update(projected_bytes=867_490_000_000.)),
 ("scope_overflow",lambda x:x["scope_measurement"].update(conservative_scope_bytes=1_000_000_001)),
 ("false_global_remeasurement_label",lambda x:x.update(projection_is_historical_accounting_not_global_remeasurement=False))]:
 altered=copy.deepcopy(account);mutate(altered)
 check("account_"+label,lambda altered=altered:ns["validate_account_v2"](altered,storage),True)
check("planned_boundary_write_fits",lambda:ns["validate_account_v2"](account,storage,1_000_000_000-123_456))
check("planned_boundary_write_exceeds",lambda:ns["validate_account_v2"](account,storage,1_000_000_000-123_456+1),True)
check("negative_planned_write_denied",lambda:ns["validate_account_v2"](account,storage,-1),True)
state={"allow":True,"calls":0,"writes":[]}
def raw():
 state["calls"]+=1
 if not state["allow"]:raise ValueError("simulated fresh admission denial")
 return True
guard=ns["sticky_authority_v2"](raw)
check("sticky_initial_authority_pass",guard)
state["allow"]=False
check("sticky_denial_latches",guard,True)
state["allow"]=True
before=state["calls"]
def late_write():
 guard();state["writes"].append("forbidden evidence")
check("sticky_restored_authority_cannot_write",late_write,True)
check("sticky_recovery_does_not_reenter_authority",lambda:None if state["calls"]==before and state["writes"]==[] else (_ for _ in ()).throw(AssertionError()))
proof={"schema":"census_CPU_timing_v2_storage_only_source_delta_proof",
 "root_registration_ref":cfg["root_source_registration_ref"],
 "original_nonstorage_prefix_statements_AST_equal":len(oldf.body[:oi]),
 "original_nonstorage_suffix_statements_AST_equal":len(oldf.body[oe:]),
 "original_launcher_decode_AST_equal":True,
 "complete_scientific_caller_try_body_statements_AST_equal":len(oldtry.body),
 "scientific_caller_try_body_AST_sha256":digest("\n".join(dump(n) for n in oldtry.body).encode()),
 "failure_handlers_AST_equal":True,"original_v1_caller_bytes_preserved":True,
 "all_original_scientific_pins_bytes_identical":True,
 "delta":"Only old native storage/snapshot admission statements replaced by pinned v4 own reservation guard; first ledger pins/sticky/planned-write plumbing added outside scientific body",
 "v4_checker_source_call_count":1,"v4_checker_boundary_true":True,
 "live_mutating_budget_calls_in_bridge":False,"whole_project_scan_calls_in_bridge":False,
 "actual_arrays_models_budget_calls_or_profile":False}
for name,obj in [("source_delta_proof.json",proof),("handmade_checks.json",{"schema":"census_CPU_timing_v2_handmade_predicate_checks",
 "status":"PASS","total":len(records),"records":records,"scope":"stdlib extracted pure metadata predicates only; no original module imports or live budget calls",
 "simulated86749GB_forecast_is_handmade_not_live_measurement":True})]:
 (DEST/name).write_text(json.dumps(obj,sort_keys=True,indent=2)+"\n")
files={str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in sorted(DEST.iterdir()) if p.is_file()}
receipt={"schema":"census_CPU_timing_v2_author_receipt","review_role":"author",
 "status":"source_and_handmade_storage_bridge_only","root_registration_ref":cfg["root_source_registration_ref"],
 "artifact_sha256":files,"handmade_predicate_checks":len(records),"source_delta_proof":str((DEST/"source_delta_proof.json").relative_to(ROOT)),
 "original_gate_AST_preserved_except_declared_storage_block":True,"scientific_caller_body_AST_identical":True,
 "all_scientific_pins_bytes_identical":True,"sticky_denied_evidence_checks_passed":True,
 "actual_inputs_opened":False,"models_instantiated":False,"live_budget_or_reservation_operations":False,
 "GPU_used":False,"timing_performed":False,"actual_admission":False,"Git_operations":False,
 "independent_review_still_required":True}
(DEST/"author_receipt.json").write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
print(json.dumps({"status":"PASS","handmade_predicate_checks":len(records),"source_delta_proof":proof,
 "author_receipt_sha256":digest((DEST/"author_receipt.json").read_bytes()),
 "caller_sha256":digest(C.read_bytes()),"bridge_sha256":digest(B.read_bytes()),"config_sha256":cfgsha},indent=2))
