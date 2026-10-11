"""Finite independent pure-metadata bridge review; all budget IO is in memory."""
from pathlib import Path
from copy import deepcopy
from datetime import datetime,timezone
from functools import wraps
from contextlib import contextmanager
from types import ModuleType,SimpleNamespace
from unittest.mock import patch
import ast,json,hashlib,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
AUTHOR=ROOT/"research/reviews/census_certified_forest_reuse_CPU_timing_v2_author"
OUT=ROOT/"research/reviews/census_certified_forest_reuse_CPU_timing_v2_independent"
counts={};cases=[]
def sha(data):return hashlib.sha256(data).hexdigest()
def check(value,category):
 counts[category]=counts.get(category,0)+1
 assert value,category
def dump(node):return ast.dump(node,include_attributes=False)
def fn(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
def ref(path):return {"path":str(path.relative_to(ROOT)),"sha256":sha(path.read_bytes())}
for name,expected in {
 "root_cpu_timing_recipe_v2.py":"b89fa076bead973de9193f29eccfcc814836cc82e6c00a2adf0a6cb484be0a1f",
 "cpu_resource_bridge_v2.py":"11c791e06b37d50a4fa25f237d49841cb2d274fcb4e945a1ff24984404bda766",
 "protocol.json":"c7a72c73b601f8a4bdbd4d5d4f8a9583a4a0d5d0069e5468c22e9964497ad50e",
 "author_receipt.json":"22d37c8d8e30a9177337a39e68e9fcded992768e407f250c21f286c61d60c174"
}.items():check(ref(AUTHOR/name)["sha256"]==expected,"exact_assigned_pin")
# Replay precisely the 58 author predicates and AST checks, stopping before every author write.
author_text=(AUTHOR/"handmade_predicate_checks.py").read_text();author_tree=ast.parse(author_text)
cut=next(i for i,n in enumerate(author_tree.body) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple) and [x.id for x in n.target.elts if isinstance(x,ast.Name)]==["name","obj"])
replay={}
exec(compile(ast.Module(body=author_tree.body[:cut],type_ignores=[]),"<read-only author predicate replay>","exec"),replay)
check(len(replay["records"])==58 and all(r["status"].startswith("PASS") for r in replay["records"]),"58_author_predicates_replayed_read_only")
proof=replay["proof"]
check(proof["original_nonstorage_prefix_statements_AST_equal"]==32 and proof["original_nonstorage_suffix_statements_AST_equal"]==3 and proof["original_launcher_decode_AST_equal"],"32_prefix_3_suffix_launcher_AST")
check(proof["complete_scientific_caller_try_body_statements_AST_equal"]==28 and proof["failure_handlers_AST_equal"],"28_scientific_body_and_handlers_AST")
cfg=replay["cfg"];ns=replay["ns"]
bridge_tree=replay["bt"];caller_tree=replay["ct"]
exec(compile(ast.Module(body=[fn(bridge_tree,"require_storage_v2")],type_ignores=[]),"<bridge storage adapter>","exec"),ns)
budget_source=(ROOT/"src/arggnn/project_budget_v4.py").read_text();budget_tree=ast.parse(budget_source)
policy=json.loads((ROOT/cfg["budget_config_ref"]["path"]).read_text())
budget_names={"bound","policy","scope_spec","current","legacy_state","legacy_key","empty_ledger","encoded","integer","paths_overlap","accounting_paths","check_reservation"}
BUDGET_CODE=compile(ast.Module(body=[n for n in budget_tree.body if isinstance(n,ast.FunctionDef) and n.name in budget_names],type_ignores=[]),"<extracted accepted v4 metadata>","exec")
class Fixture:
 def __init__(self):
  self.data=deepcopy(replay["data"]);self.storage=deepcopy(replay["storage"]);self.admission=deepcopy(replay["admission"])
  self.data[policy["legacy_pool_source"]["path"]]=(ROOT/policy["legacy_pool_source"]["path"]).read_bytes()
  self.events=[];self.mutations=[];self.scope_bytes=1500;self.now=replay["now"]
  runtime_bytes=b'{"handmade_runtime_metadata":true}\n'
  runtime_ref=self.put("fixture/runtime.json",runtime_bytes)
  oldruntime=self.admission["refs"]["runtime"]
  self.admission["refs"]["runtime"]=runtime_ref
  self.storage["reservation"]["consumer_bindings"]=[runtime_ref if x==oldruntime else x for x in self.storage["reservation"]["consumer_bindings"]]
  self.v4_review={
   "kind":"project_budget_v4_independent_review","review_role":"independent_reviewer","verdict":"PASS",
   "implementation":cfg["budget_source_ref"],"config":cfg["budget_config_ref"],
   "human_authorization":cfg["human_authorization_ref"],"actual_apply_performed":False}
  self.v4_review_ref=self.put("fixture/v4_source_review.json",self.v4_review)
  self.own=self.storage["reservation"]
  self.bindings={"implementation":cfg["budget_source_ref"],"config":cfg["budget_config_ref"],"human_authorization":cfg["human_authorization_ref"]}
  self.attach_root(self.own,"fixture/own_root_receipt.json")
  self.other={"key":"fixture-other","output":str(ROOT/"fixture/other-output"),"auxiliary_scopes":[],
   "reserved_bytes":450_000_000_000,"consumer_bindings":[cfg["budget_config_ref"],cfg["human_authorization_ref"]]}
  self.attach_root(self.other,"fixture/other_root_receipt.json")
  self.legacy={**policy["legacy_invariants"],"root":str(ROOT),"controllers":{"fake-controller":{"reserved_bytes":1_000_000_000,"pid":123}},
   "jobs":{"fake-a":{"reserved_bytes":40_000_000_000,"output":str(ROOT/"fixture/legacy-a")},
           "fake-b":{"reserved_bytes":60_000_000_000,"output":str(ROOT/"fixture/legacy-b")}}}
  self.ledger={"schema":"project_budget_v4_supplemental_ledger","root":str(ROOT),**self.bindings,
   "legacy_invariants":policy["legacy_invariants"],"scheduling_ceiling_bytes":950_000_000_000,"hard_limit_bytes":1_000_000_000_000,
   "reservations":{self.own["key"]:deepcopy(self.own),self.other["key"]:deepcopy(self.other)}}
  self.seal_ledgers()
  self.initial=deepcopy(self.storage["initial_ledger_pins"])
  self.bn={"Path":Path,"copy":SimpleNamespace(deepcopy=deepcopy),"json":json,"hashlib":hashlib,
   "CONFIG":cfg["budget_config_ref"]["path"],"SOURCE":cfg["budget_source_ref"]["path"],
   "FIXED_POLICY":deepcopy(policy),"LEGACY":policy["legacy_ledger"],"SUPPLEMENTAL":policy["supplemental_ledger"],"LOCK":policy["scheduler_lock"],
   "canonical":self.canonical,"file_bytes":self.file_bytes,"read_json":self.read_json,"parse":json.loads,
   "reference":lambda root,path:{"path":str(self.canonical(root,path).relative_to(root)),"sha256":self.file_bytes(root,path)[1]},
   "mutex":self.mutex,"measure_scopes":self.measure}
  exec(BUDGET_CODE,self.bn)
 def raw(self,value):return value if isinstance(value,bytes) else json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
 def put(self,path,value):
  raw=self.raw(value);self.data[path]=raw;return {"path":path,"sha256":sha(raw)}
 def attach_root(self,reservation,path):
  request={k:deepcopy(v) for k,v in reservation.items() if k!="root_admission"}
  record={"kind":"project_budget_v4_root_reserve_admission","root_authorizer":"root_agent","active":True,
   "implementation":cfg["budget_source_ref"],"config":cfg["budget_config_ref"],"human_authorization":cfg["human_authorization_ref"],
   "all_budget_writers_use_scheduler_mutex":True,"no_legacy_mutation_authorized":True,
   "GPU_authorized":False,"scientific_execution_authorized":False,"global_rebase_authorized":False,
   "reservations":[request],"independent_review":self.v4_review_ref}
  reservation["root_admission"]=self.put(path,record)
 def seal_ledgers(self):
  l=self.put(policy["legacy_ledger"],self.legacy);s=self.put(policy["supplemental_ledger"],self.ledger)
  self.storage["initial_ledger_pins"]={"legacy_ledger_sha256":l["sha256"],"supplemental_ledger_sha256":s["sha256"]}
 def canonical(self,root,value):
  p=Path(value);p=p if p.is_absolute() else root/p
  if p==root or root not in p.parents or ".." in p.parts:raise ValueError("handmade canonical path denial")
  return p
 def file_bytes(self,root,value):
  p=self.canonical(root,value);relative=str(p.relative_to(root));self.events.append("read:"+relative)
  data=self.data[relative];return data,sha(data)
 def read_json(self,root,path):
  data,digest=self.file_bytes(root,path);return json.loads(data),digest
 @contextmanager
 def mutex(self,root):
  self.events.append("fake_mutex_enter");yield;self.events.append("fake_mutex_exit")
 def measure(self,root,paths):
  self.events.append(("fake_measure",list(paths)))
  return {"conservative_scope_bytes":self.scope_bytes,"scope_paths":[str(p) for p in paths],"stat_only":True,"entries":1}
 def reader(self):return replay["Reader"](self.data)
 def run(self,*,first=True,write=0):
  parent=ModuleType("arggnn");parent.__path__=[]
  module=ModuleType("arggnn.project_budget_v4");module.check_reservation=self.bn["check_reservation"]
  before=deepcopy(self.data)
  with patch.dict(sys.modules,{"arggnn":parent,"arggnn.project_budget_v4":module}),patch.object(Path,"exists",lambda p:str(p.relative_to(ROOT)) in self.data):
   account=ns["require_storage_v2"](self.storage,self.admission,{"output_cap_bytes":1_000_000_000},self.reader(),self.now,first=first,required_write_bytes=write)
  check(self.data==before and self.mutations==[],"read_only_no_mutation")
  return account
def accepted(f,name,**kwargs):
 a=f.run(**kwargs);cases.append({"case":name,"status":"PASS"});return a
def rejected(f,name,**kwargs):
 before=deepcopy(f.data)
 try:f.run(**kwargs)
 except (ValueError,KeyError,AssertionError) as error:cases.append({"case":name,"status":"PASS_expected_rejection","error":type(error).__name__})
 else:raise AssertionError(name+" unexpectedly accepted")
 check(f.data==before and f.mutations==[],"rejection_before_mutation")
f=Fixture();account=accepted(f,"full_combined_accounting")
check(account["projected_bytes"]==167_336_640_512+130_000_000_000+101_000_000_000+451_000_000_000,"full_baseline_external_jobs_controller_and_all_supplemental")
check(account["legacy_reserved_bytes"]==101_000_000_000 and account["supplemental_reserved_bytes"]==451_000_000_000,"no_lowered_concurrency")
check([e[1] for e in f.events if isinstance(e,tuple)]==[[ROOT/"fixture/output"]],"only_own_output_tmp_scope_measured")
check(account["legacy_ledger_mutated"] is False and account["supplemental_ledger_mutated"] is False,"readonly_flags_preserved")
f=Fixture();accepted(f,"exact_planned_write_boundary",write=1_000_000_000-f.scope_bytes)
f=Fixture();rejected(f,"planned_write_overflow",write=1_000_000_000-f.scope_bytes+1)
for write in (-1,1.0,True):
 f=Fixture();rejected(f,"malformed_planned_write_"+repr(write),write=write)
for key in ("legacy_ledger_sha256","supplemental_ledger_sha256"):
 f=Fixture();f.storage["initial_ledger_pins"][key]="0"*64;rejected(f,"first_"+key+"_drift")
f=Fixture();f.ledger["reservations"].pop(f.own["key"]);f.seal_ledgers();rejected(f,"own_reservation_absent")
f=Fixture();f.ledger["reservations"][f.own["key"]]["reserved_bytes"]+=1;f.seal_ledgers();rejected(f,"stored_own_reservation_differs")
f=Fixture();f.data[f.own["root_admission"]["path"]]+=b" ";rejected(f,"root_receipt_stale_bytes")
for key,value in (("active",False),("root_authorizer","other"),("GPU_authorized",True),("scientific_execution_authorized",True),
 ("global_rebase_authorized",True),("no_legacy_mutation_authorized",False),("all_budget_writers_use_scheduler_mutex",False),
 ("reservations",[])):
 f=Fixture();path=f.own["root_admission"]["path"];record=json.loads(f.data[path]);record[key]=value
 f.own["root_admission"]=f.put(path,record);f.ledger["reservations"][f.own["key"]]=deepcopy(f.own);f.seal_ledgers()
 rejected(f,"root_receipt_"+key)
for key,value in (("verdict","REJECT"),("review_role","author"),("actual_apply_performed",True)):
 f=Fixture();review=deepcopy(f.v4_review);review[key]=value;f.v4_review_ref=f.put("fixture/v4_source_review.json",review)
 f.attach_root(f.own,"fixture/own_root_receipt.json");f.ledger["reservations"][f.own["key"]]=deepcopy(f.own);f.seal_ledgers()
 rejected(f,"root_v4_review_"+key)
for key in ("baseline_bytes","external_reserve_bytes","snapshot_sha256","limit_bytes"):
 f=Fixture();f.legacy[key]=0 if key!="snapshot_sha256" else "0"*64;f.seal_ledgers();rejected(f,"legacy_fixed_"+key)
f=Fixture();f.scope_bytes=1_000_000_001;rejected(f,"scope_overflow")
f=Fixture();f.other["output"]=f.own["output"];f.attach_root(f.other,"fixture/other_root_receipt.json");f.ledger["reservations"][f.other["key"]]=deepcopy(f.other);f.seal_ledgers();rejected(f,"overlapping_product_scope")
f=Fixture();f.own["output"]=str(ROOT/".runtime/storage/fake-product");f.storage["output_directory"]=" .runtime/storage/fake-product".strip();f.admission["output_directory"]=f.storage["output_directory"]
f.attach_root(f.own,"fixture/own_root_receipt.json");f.ledger["reservations"][f.own["key"]]=deepcopy(f.own);f.seal_ledgers();rejected(f,"scope_overlaps_accounting")
for delta in (100_663_359_488,150_663_359_488):
 f=Fixture();f.other["reserved_bytes"]+=delta;f.attach_root(f.other,"fixture/other_root_receipt.json")
 f.ledger["reservations"][f.other["key"]]=deepcopy(f.other);f.seal_ledgers();rejected(f,"full_combined_ceiling_"+str(delta))
# Subsequent reads permit truthful unrelated charges, but keep own reservation exact.
f=Fixture();accepted(f,"initial_read");initial=deepcopy(f.storage["initial_ledger_pins"])
f.other["reserved_bytes"]+=1_000_000_000;f.attach_root(f.other,"fixture/other_root_receipt.json");f.ledger["reservations"][f.other["key"]]=deepcopy(f.other);f.seal_ledgers()
f.storage["initial_ledger_pins"]=initial
account=accepted(f,"legitimate_unrelated_charge_after_first",first=False)
check(account["supplemental_reserved_bytes"]==452_000_000_000,"subsequent_full_charge_preserved")
rejected(f,"first_pin_still_rejects_changed_ledger",first=True)
# Every required consumer binding, including protocol/budget config, fails before entering fake budget mutex.
for index in range(7):
 f=Fixture();f.own["consumer_bindings"].pop(index)
 rejected(f,"required_consumer_missing_"+str(index))
 check("fake_mutex_enter" not in f.events,"consumer_denial_before_budget_IO")
for key in ("budget_source_ref","budget_config_ref","human_authorization_ref"):
 f=Fixture();f.data[cfg[key]["path"]]+=b" "
 rejected(f,"source_guard_"+key)
 check("fake_mutex_enter" not in f.events,"source_denial_before_budget_IO")
for path in cfg["executed_bridge_source_paths"]+[ns["CONFIG_PATH"]]:
 f=Fixture();record=json.loads(f.data["fixture/review.json"]);record["artifact_sha256"][path]="0"*64
 f.storage["review_ref"]=f.put("fixture/review.json",record)
 rejected(f,"independent_review_artifact_drift_"+Path(path).name)
 check("fake_mutex_enter" not in f.events,"review_denial_before_budget_IO")
# Independent proof of actual nested authority plumbing and sticky denial, without whole caller/model runs.
nested=next(n for n in fn(caller_tree,"main").body if isinstance(n,ast.FunctionDef) and n.name=="authority")
nested_old=next(n for n in fn(ast.parse((ROOT/"research/reviews/census_certified_forest_reuse_v1_timing_author/root_cpu_timing_recipe.py").read_text()),"main").body if isinstance(n,ast.FunctionDef) and n.name=="authority")
restored=deepcopy(nested);restored.decorator_list=[];restored.args=deepcopy(nested_old.args)
restored.body=[n for n in restored.body if not isinstance(n,ast.Nonlocal) and not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="first_storage_read" for t in n.targets))]
for n in ast.walk(restored):
 if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="admit_cpu_timing_v2":
  n.func.id="admit_runner_v1";n.keywords=[k for k in n.keywords if k.arg not in {"first_storage_read","required_write_bytes"}]
check(dump(restored)==dump(nested_old),"all_nonstorage_caller_authority_AST_restored")
wrapper=ast.FunctionDef(name="make_authority",args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[]),
 body=[ast.Assign(targets=[ast.Name(id="first_storage_read",ctx=ast.Store())],value=ast.Constant(value=True)),deepcopy(nested),ast.Return(value=ast.Name(id="authority",ctx=ast.Load()))],decorator_list=[])
authority_code=compile(ast.fix_missing_locations(ast.Module(body=[wrapper],type_ignores=[])),"<new authority plumbing>","exec")
def authority_fixture():
 p=json.loads((AUTHOR/"pins.json").read_text());old=json.loads((AUTHOR/"base_admission.template.json").read_text())
 current=deepcopy(old);current.update(job_id="handmade-authority",output_directory="fixture/output",created_at_utc="2026-10-11T05:55:00+00:00",expires_at_utc="2026-10-11T07:00:00+00:00")
 calls=[];state={"allow":True}
 recipe=ref(AUTHOR/"root_cpu_timing_recipe_v2.py");pins_ref=ref(AUTHOR/"pins.json")
 memory={"schema":"census_forest_reuse_CPU_profile_memory_v1","status":"active","root_authorizer":"root_agent","job_id":current["job_id"],
  "created_at_utc":current["created_at_utc"],"expires_at_utc":current["expires_at_utc"],"CPU_total_reservation_bytes":64_000_000_000,
  "GPU_admitted":False,"external_watchdog":True,"recipe_ref":recipe,"pins_ref":pins_ref,"output_directory":current["output_directory"]}
 reader=SimpleNamespace(decode_json=lambda r:deepcopy(memory))
 def gate(*args,**kwargs):
  calls.append(kwargs)
  if not state["allow"]:raise ValueError("handmade denial")
  return deepcopy(current),{"seed":41,"model":p["model"],"readout_chunk_size":128},reader
 g={"sticky_authority_v2":ns["sticky_authority_v2"],"admit_cpu_timing_v2":gate,"refs":{"base":{},"reuse":{},"memory":{}},
  "reader":reader,"old":old,"pins":p,"self_ref":recipe,"pins_ref":pins_ref,"datetime":SimpleNamespace(now=lambda tz:replay["now"]),
  "timezone":timezone,"utc":lambda x:datetime.fromisoformat(x),"admit_reuse_v1":lambda *a,**k:{"fit_authorized":False,
  "source_ref":p["reuse_source_ref"],"config_ref":p["reuse_config_ref"],"review_ref":p["reuse_review_ref"]}}
 exec(authority_code,g);return g["make_authority"](),calls,state
authority,calls,state=authority_fixture();authority();authority();authority(required_write_bytes=734)
check([c["first_storage_read"] for c in calls]==[True,False,False] and [c["required_write_bytes"] for c in calls]==[0,0,734],"first_pin_once_and_exact_planned_bytes")
state["allow"]=False
try:authority()
except ValueError:pass
else:raise AssertionError("denial failed")
state["allow"]=True;before=len(calls)
try:authority(required_write_bytes=734)
except ValueError:pass
else:raise AssertionError("sticky restoration accepted")
check(len(calls)==before,"sticky_denial_prevents_reentry_after_restore")
newtry=next(n for n in fn(caller_tree,"main").body if isinstance(n,ast.Try))
check(len(newtry.finalbody)==2 and dump(newtry.finalbody[1])==dump(next(n for n in fn(ast.parse((ROOT/"research/reviews/census_certified_forest_reuse_v1_timing_author/root_cpu_timing_recipe.py").read_text()),"main").body if isinstance(n,ast.Try)).finalbody[1]),"unchanged_evidence_write_AST")
events=[];result={"handmade":True,"records":[{"value":1}]}
g={"result":result,"json_bytes_v1":lambda value:json.dumps(value,sort_keys=True,separators=(",",":")).encode(),
 "authority":lambda **kwargs:events.append(("gate",kwargs["required_write_bytes"])),
 "output":SimpleNamespace(write=lambda name,value:events.append(("write",name)))}
exec(compile(ast.fix_missing_locations(ast.Module(body=deepcopy(newtry.finalbody),type_ignores=[])),"<new final write boundary>","exec"),g)
check(events==[("gate",len(g["json_bytes_v1"](result))),("write","evidence.json")],"planned_serialized_write_before_output")
def deny(**kwargs):raise ValueError("handmade final denial")
g["authority"]=deny;events.clear()
try:exec(compile(ast.fix_missing_locations(ast.Module(body=deepcopy(newtry.finalbody),type_ignores=[])),"<denied final boundary>","exec"),g)
except ValueError:pass
else:raise AssertionError("denied final boundary accepted")
check(events==[],"denied_final_boundary_zero_write")
proof.update({"independent_nested_nonstorage_authority_AST_restored":True,"exact_final_serialized_bytes_plumbing":True,
 "first_ledger_pin_once_then_full_genuine_unrelated_charge_checks":True,"independent_resource_integration_cases":len(cases)})
report={"schema":"census_CPU_timing_v2_independent_finite_checks","status":"PASS","author_predicate_replays":58,
 "fresh_assertions":sum(counts.values()),"categories":counts,"fresh_integration_cases":cases,
 "proof":proof,"all_budget_IO_and_mutex_scope_measurement_callbacks_handmade":True,
 "no_real_ledger_runtime_input_model_GPU_process_or_budget_API_access":True,
 "prior1770_caller_or6888_model_checks_repeated":False}
with (OUT/"finite_checks.json").open("x") as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write("\n")
print(json.dumps({"status":"PASS","author_predicate_replays":58,"fresh_assertions":sum(counts.values()),"fresh_integration_cases":len(cases),"AST_prefix_suffix_body":[32,3,28]},indent=2))
