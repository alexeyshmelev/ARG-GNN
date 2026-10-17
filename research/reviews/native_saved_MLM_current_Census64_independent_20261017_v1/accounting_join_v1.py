"""Independent bounded RAM reproduction of combined ROOT-record/journal allowance."""
from pathlib import Path
import ast,copy,hashlib,json,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
OUT=ROOT/"research/reviews/native_saved_MLM_current_Census64_independent_20261017_v1"
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
low=30000;high=40000;found=None
while low<=high:
 n=(low+high)//2
 row=attempt(n)
 if row is None:high=n-1;continue
 if row["combined_conservative_bytes"]>524288:
  found=row;break
 low=n+1
result=dict(schema="native_Census64_independent_combined_metadata_join_v1",candidate_manifest_ref=e["source_ref"](e["CAND"]/"source_manifest.v1.json"),finding_reproduced=found is not None,reproduction=found,attempts=attempts,actual_guard_or_journal_calls=False,actual_admission_or_filesystem_records=False,all_operations_RAM_only=True)
(OUT/"accounting_join.result.v1.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
print(json.dumps(result,sort_keys=True))
