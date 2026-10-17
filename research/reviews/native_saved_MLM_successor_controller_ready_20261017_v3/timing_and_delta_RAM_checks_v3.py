"""Exact v3 timing predicate and unchanged complete namespace checks in RAM."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from types import SimpleNamespace
import ast,copy,json,hashlib
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
B=Path("research/reviews/native_saved_MLM_successor_controller_ready_20261017_v2")
D=Path("research/reviews/native_saved_MLM_successor_controller_ready_20261017_v3")
CHECKS=[]
def require(ok,label):
    if not ok:raise ValueError(label)
def ref(p):return dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def check(label,fn):fn();CHECKS.append(label)
def rejects(fn):
    try:fn()
    except(ValueError,KeyError,TypeError):return
    raise AssertionError("Expected rejection")
def method(source,name,scope):
    n=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name==name)
    exec(compile(ast.Module(body=[n],type_ignores=[]),"RAM_EXACT_SOURCE_AST","exec"),scope)
    return scope[name]
source=(D/"empty_gpu0_controller_v1.py").read_text();old=(B/"empty_gpu0_controller_v1.py").read_text()
tree=ast.parse(source);serve=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=="serve_v1")
oldserve=next(n for n in ast.parse(old).body if isinstance(n,ast.FunctionDef)and n.name=="serve_v1")
def timing_node(fn):
    return next(n for n in fn.body if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)
        and isinstance(n.value.func,ast.Name)and n.value.func.id=="need"
        and any(isinstance(x,ast.Subscript)and isinstance(x.value,ast.Name)and x.value.id=="cert"for x in ast.walk(n)))
predicate=timing_node(serve);oldpredicate=timing_node(oldserve)
clock=[datetime(2026,10,17,17,0,tzinfo=timezone.utc).timestamp()]
def iso(seconds):return datetime.fromtimestamp(seconds,timezone.utc).isoformat()
scope=dict(datetime=datetime,need=require)
stamp=method(source,"stamp",scope)
windowstart=clock[0]-1000;windowend=clock[0]+3000
def values(age=300):
    return dict(cert=dict(observed_at_utc=iso(clock[0]-age),known_identities=[dict(pid=1,start_ticks=2)]),
        h=dict(created_at_utc=iso(windowstart),expires_at_utc=iso(windowend)),
        cap=dict(created_at_utc=iso(windowstart),expires_unix=windowend),
        time=SimpleNamespace(time=lambda:clock[0]),stamp=stamp,need=require)
def evaluate(node,v):exec(compile(ast.Module(body=[node],type_ignores=[]),"RAM_EXACT_TIMING_PREDICATE","exec"),v)
for age in [0,119,121,300,600,999]:
    check("new_finite_roster_accepts_age_"+str(age)+"s_inside_actual_window",lambda age=age:evaluate(predicate,values(age)))
check("preserved_v2_snapshot_rejects300s_after_valid_slow_admission",lambda:rejects(lambda:evaluate(oldpredicate,values(300))))
for title,change in [
    ("future_observation",lambda v:v["cert"].update(observed_at_utc=iso(clock[0]+1))),
    ("timezone_naive_observation",lambda v:v["cert"].update(observed_at_utc="2026-10-17T16:55:00")),
    ("before_handover_start",lambda v:v["h"].update(created_at_utc=iso(clock[0]-100))),
    ("before_plan_start",lambda v:v["cap"].update(created_at_utc=iso(clock[0]-100))),
    ("expired_handover_window",lambda v:v["h"].update(expires_at_utc=iso(clock[0]-1))),
    ("expired_plan_work_window",lambda v:v["cap"].update(expires_unix=clock[0]-1)),
    ("naive_handover_start",lambda v:v["h"].update(created_at_utc="2026-10-17T16:00:00")),
]:
    v=values();change(v)
    check("new_roster_rejects_"+title,lambda v=v:rejects(lambda:evaluate(predicate,v)))
# Actual original entry prefix, including gate, slow guard and first live absent().
first=next(i for i,n in enumerate(serve.body)if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)
    and isinstance(n.value.func,ast.Name)and n.value.func.id=="absent")
prefix=copy.deepcopy(serve);prefix.body=prefix.body[:first+1]
def slow_entry():
    initial=clock[0];v=values(10);events=[]
    def guard(cap,storage):clock[0]+=300;events.append("slow_original_guard_300s");return SimpleNamespace()
    v.update(os=SimpleNamespace(read=lambda fd,n:b"G",close=lambda fd:events.append("gate_close"),environ={"CUDA_VISIBLE_DEVICES":""}),
        sys=SimpleNamespace(modules={}),validate_cap_v1=lambda r:(v["cap"],v["h"],v["cert"],{}),
        guard_v1=guard,absent=lambda rows:events.append(("actual_targeted_absent_RAM",rows)))
    exec(compile(ast.Module(body=[prefix],type_ignores=[]),"RAM_ACTUAL_SERVE_ENTRY_PREFIX","exec"),v)
    try:v["serve_v1"]({},10)
    finally:clock[0]=initial
    assert events[-1][0]=="actual_targeted_absent_RAM"and events.index("slow_original_guard_300s")<len(events)-1
check("actual_entry_prefix_survives300s_guard_and_still_rechecks_current_known_absence",slow_entry)
# Reverse only the exact timing node and namespace; all remaining owner AST is exact.
normalized=source.replace(str(D),str(B))
nt=ast.parse(normalized);ns=next(n for n in nt.body if isinstance(n,ast.FunctionDef)and n.name=="serve_v1")
idx=next(i for i,n in enumerate(ns.body)if ast.dump(n,include_attributes=False)==ast.dump(predicate,include_attributes=False))
ns.body[idx]=copy.deepcopy(oldpredicate)
assert ast.dump(nt,include_attributes=False)==ast.dump(ast.parse(old),include_attributes=False)
CHECKS.append("whole_owner_AST_one_timing_predicate_and_namespace_only")
calls=[n for n in ast.walk(serve)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=="absent"]
assert len(calls)==2
lockline=next(n.lineno for n in ast.walk(serve)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=="flock"
    and any(isinstance(x,ast.Name)and x.id=="scheduler"for x in ast.walk(n)))
assert calls[0].lineno<lockline<calls[1].lineno
CHECKS.append("both_unchanged_live_absent_calls_preserved_second_under_scheduler_lock")
parent=(D/"owner_parent_v1.py").read_text().replace(str(D),str(B))
assert parent==(B/"owner_parent_v1.py").read_text()
CHECKS.append("complete_parent_source_byte_identical_after_namespace_normalization")
newendpoint=ref(D/"empty_gpu0_controller_v1.py");oldendpoint=ref(B/"empty_gpu0_controller_v1.py")
native=json.loads((D/"native64/source_manifest.v1.json").read_text())
for r in native["source_authentication_refs"]+native["executed_source_refs"]:
    assert ref(r["path"])==r
CHECKS.append("all217_native_authenticated_dependencies_and7_executed_sources_exact")
for r in native["executed_source_refs"]:
    name=Path(r["path"]).name
    current=(D/"native64"/name).read_text().replace(str(D),str(B)).replace(str(D).replace("/","."),str(B).replace("/",".")).replace(newendpoint["sha256"],oldendpoint["sha256"])
    assert ast.dump(ast.parse(current),include_attributes=False)==ast.dump(ast.parse((B/"native64"/name).read_text()),include_attributes=False)
    CHECKS.append("complete_native_whole_AST_namespace_endpoint_only_"+name)
assert (D/"native64/scientific_config.FROZEN.v1.json").read_bytes()==(B/"native64/scientific_config.FROZEN.v1.json").read_bytes()
CHECKS.append("native_scientific_config_byte_identical")
assert ref(B/"source_manifest.v1.json")["sha256"]=="933b2a7dc5c4f2a7604e36f0d1b7bb059b9692ed5014095a9579e0141d167ec4"
assert ref(B/"native64/source_manifest.v1.json")["sha256"]=="2c70c1d6e1d1e3db87eceab3b9dc93a407dbc6f6a7938bc7b4ed94ec18e70538"
CHECKS.append("both_sealed_v2_manifests_and_all_old_failure_review_lineage_preserved")
print(json.dumps(dict(schema="empty_GPU0_owner_native_v3_author_timing_AST_RAM_checks_v1",status="AUTHOR_AST_RAM_PASS",checks=CHECKS,count=len(CHECKS),
    endpoint_source_ref=newendpoint,owner_manifest_ref=ref(D/"source_manifest.v1.json"),native_manifest_ref=ref(D/"native64/source_manifest.v1.json"),
    actual_guard_import_or_process_resource_inventory_GPU_socket_lock_operations=False,
    actual_admission_execution_or_scientific_payload_reads=False,independent_review_claimed=False),indent=2,sort_keys=True))
