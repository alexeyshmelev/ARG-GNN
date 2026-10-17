"""Independent bounded RAM reproduction of combined ROOT-record/journal allowance."""
from pathlib import Path
import ast,copy,hashlib,json,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
OUT=ROOT/"research/reviews/native_saved_MLM_current_Census64_independent_20261017_v2"
s=(OUT/"independent_checks_v1.py").read_text().split("# Full worker entry",1)[0]
e={"__name__":"RAM_accounting_support"};exec(compile(s,"<audited_RAM_support_before_worker>","exec"),e)
for k in ("c","guard","builder","read","put","files","dirs","local","roles","q","new_master","phase_request"):
 globals()[k]=e[k]
gpath=ROOT/c.GUARD["path"];gtree=ast.parse(gpath.read_text())
v4tree=ast.parse((ROOT/c.original.V4["path"]).read_text())
enc=next(n for n in v4tree.body if isinstance(n,ast.FunctionDef)and n.name=="encoded")
encenv=dict(json=json);exec(compile(ast.Module(body=[enc],type_ignores=[]),"<exact_v4_encoded_RAM>","exec"),encenv)
g_env=dict(copy=copy,MODE=c.CENSUS_MODE,require=c.require,RETIRE_OPERATION="retire_exact_Census_observation_retained_allowance_v1")
op=next(n for n in gtree.body if isinstance(n,ast.FunctionDef)and n.name=="operation_draft_v1")
exec(compile(ast.Module(body=[op],type_ignores=[]),"<exact_selected_guard_draft_RAM>","exec"),g_env)
guard.operation_draft_v1=g_env["operation_draft_v1"]
event_records=[]
def reserve(authority_ref,**kwargs):
 authority=read(authority_ref);own=copy.deepcopy(authority["request"]);own["root_admission"]=authority_ref
 # Exact _append event schema and exact v4 serializer; no filesystem call.
 event=dict(schema="Census_observation_exact_retention_event_v1",sequence=1,previous_sha256=None,bindings=copy.deepcopy(roles),operation="reserve",authority=authority_ref,value=own)
 data=encenv["encoded"](event)+b"\n"
 if len(data)>65536:raise ValueError("Actual guard event cap")
 event_records.append(data);guard.reserved=own
 return dict(kind="concurrent_completed_scope_reserve_receipt_v1",value=own,projected_bytes=100000000000,initial_ledger_pins=dict(base_pins={},completed_overlay_prefix=dict(events=1,tail_sha256=hashlib.sha256(data).hexdigest())),old_journal_mutated=False)
guard.apply_reserve_v1=reserve
def budget(path):
 prefix=str(path.relative_to(ROOT)).rstrip("/")+"/"
 payloads=[data for key,data in files.items()if key.startswith(prefix)]
 nodes=[d for d in dirs if d==str(path)or d.startswith(str(path)+"/")]
 return sum(map(len,payloads))+4096*(len(payloads)+len(nodes)),len(payloads),len(nodes)
attempts=[]
def attempt(n):
 instance="accounting_"+str(n)+"_"+str(len(attempts));q,a,m=new_master(instance)
 for phase in c.PHASES:
  p=phase_request(q,a,phase);p["prerequisites"]["ROOT_operational_annotation"]="x"*n
  c.ROOT_INPUT_REF=None
  try:builder.prepare_phase_v1(put("RAM/accounting/"+instance+"/"+phase+".json",p))
  except ValueError as ex:
   attempts.append(dict(padding=n,accepted_all_seven=False,reason=str(ex)[:300]));return None
 root_bytes,count,nodes=budget(local(m["ROOT_record_directory"]))
 journal_bytes=len(event_records[-1])+4096
 row=dict(padding=n,accepted_all_seven=True,own_ROOT_conservative_bytes=root_bytes,own_files=count,own_directories=nodes,reserve_journal_event_bytes=len(event_records[-1]),reserve_journal_event_conservative_bytes=journal_bytes,combined_conservative_bytes=root_bytes+journal_bytes,charged_event_allowance_bytes=524288)
 attempts.append(row);return row
low=35000;high=35000;found=None
while low<=high:
 n=(low+high)//2
 row=attempt(n)
 if row is None:high=n-1;continue
 if row["combined_conservative_bytes"]>524288:
  found=row;break
 low=n+1
assert found is None and attempts and attempts[0]["accepted_all_seven"]is False
assert "unchanged512KiB event allowance"in attempts[0]["reason"]
bound_checks=[];bound_negative=[]
empty=local("RAM/independent_event_bound_empty");dirs.add(str(empty))
builder._record_budget_v1(empty,builder.MAX_ROOT_RECORD_BYTES-4096);bound_checks.append("Exactly454656 ROOT bytes including directory accepted")
try:builder._record_budget_v1(empty,builder.MAX_ROOT_RECORD_BYTES-4096+1)
except ValueError:bound_negative.append("454657 ROOT bytes rejected")
else:raise AssertionError("ROOT allowance one-byte overflow accepted")
assert builder.MAX_ROOT_RECORD_BYTES+builder.RESERVE_EVENT_JSON_MAX_BYTES+builder.RESERVE_EVENT_NODE_CHARGE_BYTES==524288
bound_checks.append("454656+65536+4096 exactly524288")
for field,bad in(("MAX_JSON",65537),("META_CHARGE",524289),("META_CHARGE",524287)):
 prior=getattr(guard,field);setattr(guard,field,bad)
 try:
  new_master("wrong_"+field.lower()+"_"+str(bad))
 except ValueError as ex:
  assert "Exact original serialized-event maximum"in str(ex);bound_negative.append(field+" mismatch rejects before records/reserve")
 else:raise AssertionError("Wrong selected child event bound accepted")
 finally:setattr(guard,field,prior)
# Exercise the exact original _append's byte guard to first forbidden journal boundary.
class BeforeJournal(Exception):pass
journal_boundary=[]
def forbidden_journal(*args):
 journal_boundary.append(True);raise BeforeJournal()
benv=dict(json=json,require=c.require,MAX_EVENTS=4096,MAX_JSON=65536,JOURNAL="RAM/never_journal",ROOT=ROOT,budget=types.SimpleNamespace(v4=types.SimpleNamespace(encoded=encenv["encoded"],canonical=forbidden_journal)))
appendfn=next(n for n in gtree.body if isinstance(n,ast.FunctionDef)and n.name=="_append")
exec(compile(ast.Module(body=[appendfn],type_ignores=[]),"<exact_original_append_byte_bound_RAM>","exec"),benv)
state=({},set(),{},roles,[],[],{})
authority=dict(path="RAM/authority.json",sha256="0"*64)
def event_bytes(value):
 return encenv["encoded"](dict(schema="Census_observation_exact_retention_event_v1",sequence=1,previous_sha256=None,bindings=roles,operation="reserve",authority=authority,value=value))+b"\n"
base=len(event_bytes(dict(padding="")))
for target in(65536,65537):
 value=dict(padding="x"*(target-base));assert len(event_bytes(value))==target
 before=len(journal_boundary)
 try:benv["_append"]("reserve",authority,value,state)
 except BeforeJournal:
  assert target==65536 and len(journal_boundary)==before+1;bound_checks.append("Exact original65536B final event reaches first forbidden journal boundary")
 except ValueError as ex:
  assert target==65537 and len(journal_boundary)==before and "Event exceeds charged metadata bound"in str(ex);bound_negative.append("Exact original65537B event rejects before journal boundary")
 else:raise AssertionError("Original append must stop at RAM boundary")
result=dict(schema="native_Census64_independent_combined_metadata_join_v2",candidate_manifest_ref=e["source_ref"](e["CAND"]/"source_manifest.v1.json"),v1_overlap_case_rejected=True,boundary_positive_count=len(bound_checks),boundary_negative_count=len(bound_negative),boundary_checks=bound_checks,boundary_negative_cases=bound_negative,reproduction=found,attempts=attempts,actual_guard_or_journal_calls=False,actual_admission_or_filesystem_records=False,all_operations_RAM_only=True)
(OUT/"accounting_join.result.v1.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
print(json.dumps(result,sort_keys=True))
