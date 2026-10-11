"""Full-reservation accounting, fake locked core view and own-scope measurements."""
import ast
from copy import deepcopy
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
from types import SimpleNamespace
root=Path.cwd();out=root/"research/reviews/full_reservation_storage_v1_independent"
sha=lambda p:hashlib.sha256((root/p).read_bytes()).hexdigest()
new_path="src/arggnn/project_storage_full_reservation_v1.py"
old_path="src/arggnn/project_budget_v9_consumer.py"
assert sha(new_path)=="c0fdace58e26020c8227606e9b77325bc7308e7ec6691fe3a7e7c18cd4846105"
assert sha(old_path)=="c11a5543746899c8a387508e9877ce6ae2bf80fe4d3076c1dbe1cb8f7a43b859"
new=ast.parse((root/new_path).read_text());old=ast.parse((root/old_path).read_text())
dump=lambda n:ast.dump(n,include_attributes=False)
def node(tree,name):return next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
for name in ("_source_ref","validate_storage_v9","PilotOutputV9","source_report_v9"):
    assert dump(node(new,name))==dump(node(old,name))
refs={key:{"path":path,"sha256":key} for key,path in {"implementation":"scripts/project_budget_v9.py","launcher":"scripts/launch_reserved_gpu_v6.py","consumer_adapter":new_path}.items()}
hashes={r["path"]:r["sha256"] for r in refs.values()}
ledger_pins={k:k for k in ("legacy_ledger_sha256","supplemental_ledger_sha256","journal_tail_sha256")}
own={"reserved_bytes":2000000000,"output":str(root/"fake-output"),"consumer_bindings":list(refs.values())}
base={"kind":"project_budget_v9_storage_record","active":True,"consumer_uses_v9_guard":True,"accounting_mode":"full_reserved_no_closed_scope_credit","budget_components":refs,"reservation":own,"job_id":"fake","reserved_new_bytes":2000000000,"output_directory":"fake-output","initial_ledger_pins":ledger_pins}
view={"v4_projected_bytes":940000000000,"v9_reserved_bytes":9000000000,"projected_bytes":900000000000,"hard_limit_bytes":1000000000000,"limit_bytes":950000000000,"closed_allowance_credit_bytes":49000000000,**ledger_pins}
subset=ast.Module(body=[node(new,"_source_ref"),node(new,"budget_check_v9")],type_ignores=[])
def run(v=None,used=1000000,planned=8192,boundary=True,first=False,storage=None,core_error=None,measure_error=None):
    calls=[];events=[];lock={"held":False}
    class Mutex:
        def __enter__(self):assert not lock["held"];lock["held"]=True;events.append("lock")
        def __exit__(self,*args):assert lock["held"];lock["held"]=False;events.append("unlock")
    def integer(value,label,minimum=0):
        if type(value) is not int or value<minimum:raise ValueError("exact integer required: "+label)
        return value
    def core(*args,**kwargs):
        assert lock["held"];events.append("core");calls.append((args,kwargs))
        if core_error:raise core_error
        return deepcopy(view if v is None else v)
    paths=[root/"fake-output",root/"fake-log-scope"]
    def scope_paths(r,res,stored):assert lock["held"] and r==root and res is own and stored is True;return paths
    def measure(r,p):
        assert lock["held"] and r==root and p==paths;events.append("measure")
        if measure_error:raise measure_error
        return {"conservative_scope_bytes":used,"scope_paths":[str(x) for x in p],"stat_only":True}
    fake_v4=SimpleNamespace(file_bytes=lambda r,p:(b"",hashes.get(p,"unknown")),canonical=lambda r,p:r/p,integer=integer,mutex=lambda r:Mutex(),measure_scopes=measure)
    env={"ROOT":root,"SOURCE":new_path,"datetime":datetime,"timezone":timezone,"budget":SimpleNamespace(v4=fake_v4,check_reservation_locked=core,successor_paths=scope_paths)}
    exec(compile(subset,new_path,"exec"),env)
    result=error=None
    try:result=env["budget_check_v9"](base if storage is None else storage,first=first,boundary=boundary,required_write_bytes=planned)
    except BaseException as caught:error=caught
    assert not lock["held"]
    for args,kw in calls:
        assert args==(root,own)
        assert kw=={"expected_source_sha256":"implementation","expected_launcher_sha256":"launcher","boundary":False,"required_write_bytes":0}
    return result,error,calls,events
r,e,c,events=run(first=True)
assert e is None and r["projected_bytes"]==949000000000 and r["credited_v9_projection_bytes"]==900000000000
assert r["closed_allowance_credit_applied_to_consumer_bytes"]==0 and r["remaining_scope_bytes"]==1999000000
assert events==["lock","core","measure","unlock"] and len(c)==1
for full in (1000000000000,1000000000001):
    v=dict(view,v4_projected_bytes=full-9000000000,projected_bytes=1)
    r,e,c,events=run(v=v);assert r is None and isinstance(e,ValueError) and "Full original" in str(e) and "measure" not in events
v=dict(view,v4_projected_bytes=990000000000,v9_reserved_bytes=9000000000,projected_bytes=1)
r,e,c,events=run(v=v,used=1999991808,planned=8192)
assert e is None and r["projected_bytes"]==999000000000
r,e,c,events=run(used=1999991809,planned=8192)
assert r is None and isinstance(e,ValueError) and "original allowance" in str(e)
r,e,c,events=run(used=2000000001,planned=0);assert isinstance(e,ValueError)
r,e,c,events=run(planned=1,boundary=False);assert isinstance(e,ValueError) and not c
r,e,c,events=run(planned=0,boundary=False);assert e is None and "measure" not in events and r["scope_measurement"] is None
for key,bad in (("hard_limit_bytes",999999999999),("v4_projected_bytes",-1),("v9_reserved_bytes",True),("v4_projected_bytes",1.0)):
    r,e,c,events=run(v=dict(view,**{key:bad}));assert r is None and isinstance(e,ValueError)
for bad in (-1,True,1.0):
    r,e,c,events=run(planned=bad);assert isinstance(e,ValueError) and not c
for st in (dict(base,active=False),dict(base,accounting_mode="credited"),dict(base,expires_at_utc="2000-01-01T00:00:00Z")):
    r,e,c,events=run(storage=st);assert isinstance(e,ValueError) and not c
bad=dict(view,legacy_ledger_sha256="changed")
r,e,c,events=run(v=bad,first=True);assert isinstance(e,ValueError) and "Initial exact" in str(e)
for origin in ("core","measurement"):
    hard=ValueError(origin+" failure")
    r,e,c,events=run(core_error=hard if origin=="core" else None,measure_error=hard if origin=="measurement" else None)
    assert r is None and e is hard and events[-1]=="unlock" and len(c)==1
report={"verdict":"PASS","source_ref":{"path":new_path,"sha256":sha(new_path)},"unchanged_original_writer_and_source_storage_validation_AST":True,"formula":"v4_projected_bytes+v9_reserved_bytes;all closure credits ignored","full_charge_examples":{"normal":949000000000,"accepted_below_1TB":999000000000,"rejected_at_or_above":1000000000000},"core_called_locked_boundary_false_required_zero":True,"all_own_declared_scopes_measured_under_same_mutex":True,"own_original_2GB_cap_including_planned_writes_enforced":True,"planned_writes_require_measurement":True,"exact_1TB_human_limit_integer_authority_first_pins_preserved":True,"errors_immediate_no_retry_or_fail_open":True,"scope":{"fake_view_and_fake_scope_only":True,"live_proc_or_scope_or_ledger_reads":False,"actual_data_checkpoint_GPU_optimizer":False,"admission_or_launch":False}}
(out/"adapter_checks.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
