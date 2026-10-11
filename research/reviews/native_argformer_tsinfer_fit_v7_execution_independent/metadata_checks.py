"""Finite metadata-only runner harness; no queue/process/runtime or budget calls."""
import ast,copy,datetime,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace
ROOT_REAL=Path.cwd();OUT=Path(__file__).resolve().parent
BASE_REAL=ROOT_REAL/"research/reviews/native_argformer_tsinfer_fit_v7_execution_root"
FROOT=OUT/"fixtures_attempt03/root";FBASE=FROOT/BASE_REAL.relative_to(ROOT_REAL);FBASE.mkdir(parents=True)
RUNNER=OUT/"bound_candidate_v1/run_four_parent_queue.py"
PLAN=json.loads((OUT/"bound_candidate_v1/execution_plan.json").read_text())
checks=[];findings=[];forbidden=[]
def ck(name,value=True):
    if not value:raise AssertionError(name)
    checks.append(name)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def audit(event,args):
    if event=="subprocess.Popen" or event in ("os.kill","os.killpg"):
        forbidden.append(event);raise PermissionError("actual launch/signal forbidden")
    if event=="open" and isinstance(args[0],(str,bytes,os.PathLike)):
        name=os.fsdecode(args[0]);p=Path(name)
        if p.is_absolute() and (str(p).startswith("/proc/") or str(p).startswith(str(ROOT_REAL/"data")) or str(p).startswith(str(ROOT_REAL/".runtime/conda/envs/argformer_native_v1")) or p in (ROOT_REAL/".runtime/gpup/storage_reservations.json",ROOT_REAL/".runtime/storage/project_budget_v4.json")):
            forbidden.append(name);raise PermissionError("actual runtime/data/process/ledger read forbidden")
sys.addaudithook(audit)
for ref in (PLAN["queue_source"],PLAN["config"],PLAN["implementation"],PLAN["budget_receipt"],PLAN["release_barrier"]):
    source=ROOT_REAL/ref["path"];target=FROOT/ref["path"];target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
for item in PLAN["stage_proposals"]:
    for key in ("admission_proposal","experiment_registration"):
        ref=item[key];target=FROOT/ref["path"];target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT_REAL/ref["path"]).read_bytes())
review_path=FROOT/"handmade_review.json"
review={"verdict":"PASS","review_role":"independent_reviewer","execution_plan":{"path":str((OUT/"bound_candidate_v1/execution_plan.json").relative_to(OUT/"bound_candidate_v1")),"sha256":PLAN["queue_source"]["sha256"]}}
plan_path=FROOT/"research/reviews/native_argformer_tsinfer_fit_v7_execution_root/execution_plan.json"
plan_path.write_bytes((OUT/"bound_candidate_v1/execution_plan.json").read_bytes())
plan_ref={"path":str(plan_path.relative_to(FROOT)),"sha256":sha(plan_path)}
review={"verdict":"PASS","review_role":"independent_reviewer","execution_plan":plan_ref,"queue_source":PLAN["queue_source"]}
review_path.write_text(json.dumps(review))
activation={"root_authorizer":"root_agent","active":True,"queue_source":PLAN["queue_source"],"execution_plan":plan_ref,"independent_review":{"path":str(review_path.relative_to(FROOT)),"sha256":sha(review_path)},"budget_receipt":copy.deepcopy(PLAN["budget_receipt"])}
activation_path=FBASE/"queue_activation.json";activation_path.write_text(json.dumps(activation))
tree=ast.parse(RUNNER.read_text())
prefix=[copy.deepcopy(n) for n in tree.body if n.lineno<44]
for n in prefix:
    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="ROOT" for t in n.targets):
        n.value=ast.Call(func=ast.Name(id="Path",ctx=ast.Load()),args=[ast.Constant(str(FROOT))],keywords=[])
module=ast.fix_missing_locations(ast.Module(body=prefix,type_ignores=[]))
def prefix_exec():
    ns={"__file__":str(FROOT/PLAN["queue_source"]["path"]),"__name__":"handmade_meta_only"}
    exec(compile(module,str(RUNNER),"exec"),ns);return ns
ns=prefix_exec();ck("unmodified_metadata_prefix_accepts_exact_fixture_refs")
barrier_path=FBASE/"release_barrier.json";original_barrier=barrier_path.read_bytes()
bad=json.loads(original_barrier);bad["waiters"]=[]
barrier_path.write_text(json.dumps(bad))
ns=prefix_exec()
ck("changed_barrier_not_plan_hash",sha(barrier_path)!=PLAN["release_barrier"]["sha256"])
ck("changed_empty_barrier_accepted_by_runner",ns["barrier"]["waiters"]==[])
findings.append({"contract":"exact_barrier_pin","changed_sha256":sha(barrier_path),"expected_sha256":PLAN["release_barrier"]["sha256"],"accepted":True})
barrier_path.write_bytes(original_barrier)
# A different activated receipt passes although the plan pins another receipt.
budget_path=FROOT/PLAN["budget_receipt"]["path"];budget_original=budget_path.read_bytes()
changed=json.loads(budget_original);changed["handmade_unreviewed_marker"]=True
budget_path.write_text(json.dumps(changed))
activation["budget_receipt"]["sha256"]=sha(budget_path);activation_path.write_text(json.dumps(activation))
ns=prefix_exec()
ck("activated_different_budget_ref_accepted",ns["reserve"].get("handmade_unreviewed_marker") is True)
ck("activated_budget_not_exact_plan_ref",activation["budget_receipt"]!=PLAN["budget_receipt"])
findings.append({"contract":"exact_budget_receipt_pin","expected_ref":PLAN["budget_receipt"],"activated_ref":copy.deepcopy(activation["budget_receipt"]),"accepted":True})
budget_path.write_bytes(budget_original);activation["budget_receipt"]=copy.deepcopy(PLAN["budget_receipt"]);activation_path.write_text(json.dumps(activation))
ns=prefix_exec()
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
exec(compile(ast.fix_missing_locations(ast.Module(body=functions,type_ignores=[])),str(RUNNER),"exec"),ns)
# Own fake ledgers provide hashes only; no actual V4 call or ledger access.
for key in ("legacy","supp"):
    path=ns[key];path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{"handmade_hash_only":true}\n')
item=PLAN["stage_proposals"][0];prop_path=FROOT/item["admission_proposal"]["path"];old=prop_path.read_bytes();prop=json.loads(old)
prop["scientific_fallback"]=True;prop_path.write_text(json.dumps(prop))
a,ap=ns["admission"](item["parent"],item["stage"],"handmade_changed_proposal_r001")
ck("mutated_proposal_hash_differs_from_plan",sha(prop_path)!=item["admission_proposal"]["sha256"])
ck("runner_activates_mutated_unreviewed_proposal",a["active"] is True and a["scientific_fallback"] is True)
findings.append({"contract":"exact_stage_proposal_pin","expected_ref":item["admission_proposal"],"changed_sha256":sha(prop_path),"active_record_ref":ns["ref"](ap),"accepted":True,"native_core_executed":False,"limit":"Existing native admission should reject fallback; this demonstrates missing root plan pin, not bypass of science guards."})
prop_path.write_bytes(old)
# Identity function only sees a handmade /proc facade; no real process reads.
identity=next(n for n in functions if n.name=="identity_alive")
for name,exists,uid,ticks,expected in (("old_alive",True,1000,123,True),("old_gone",False,1000,123,False),("PID_reused",True,1000,124,False)):
    class Proc:
        def __init__(self,*a):pass
        def exists(self):return exists
        def stat(self):return SimpleNamespace(st_uid=uid)
    context={"Path":Proc,"ticks":lambda _:ticks}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[identity],type_ignores=[])),str(RUNNER),"exec"),context)
    ck("barrier_identity_"+name,context["identity_alive"]({"pid":42,"uid":1000,"start_ticks":123}) is expected)
class WrongUID:
    def __init__(self,*a):pass
    def exists(self):return True
    def stat(self):return SimpleNamespace(st_uid=1001)
context={"Path":WrongUID,"ticks":lambda _:123}
exec(compile(ast.fix_missing_locations(ast.Module(body=[identity],type_ignores=[])),str(RUNNER),"exec"),context)
try:context["identity_alive"]({"pid":42,"uid":1000,"start_ticks":123})
except ValueError:ck("barrier_UID_change_fails_closed")
else:raise AssertionError("UID mismatch accepted")
# Execute only the unchanged scheduling tail with mocks, no actual stage runs.
tail=ast.fix_missing_locations(ast.Module(body=[n for n in tree.body if n.lineno>=112],type_ignores=[]))
parents=["bio_paths_pilot_train_000","bio_paths_pilot_train_001","bio_paths_pilot_val_000","bio_paths_pilot_val_001"]
for failure in (None,*parents):
    events=[];saved=[]
    def start(parent,stage,run,probe=None,fit=None):
        record={"parent":parent,"stage":stage,"run_id":run,"probe":probe,"fit":fit};events.append(record)
        return (None,None,record,FROOT/(parent+"_"+stage+"_receipt.json"),1)
    def finish(item):
        rec=item[2];ok=not (rec["stage"]=="probe-memory" and rec["parent"]==failure)
        return ok,item[3]
    c={"retrievals":[],"BASE":FBASE,"save":lambda path,obj:saved.append((path.name,obj)),"utc":lambda:"handmade",
       "os":SimpleNamespace(getpid=lambda:42),"ticks":lambda _:123,"ref":lambda p:{"path":str(p),"sha256":"0"*64},"start":start,"finish":finish}
    try:exec(compile(tail,str(RUNNER),"exec"),c);stopped=False
    except SystemExit as e:stopped=e.code==1
    if failure:
        count=parents.index(failure)+1
        ck("probe_failure_"+failure+"_stops",stopped and len(events)==count and all(x["stage"]=="probe-memory" for x in events))
        ck("probe_failure_"+failure+"_preserved_terminal",saved[-1][1]["stopped_on_first_failed_probe"] is True)
    else:
        ck("all_four_probes_before_any_fit",[x["stage"] for x in events[:4]]==["probe-memory"]*4)
        ck("exact_twelve_stage_runs",len(events)==12)
        ck("fit_uses_own_probe",all(x["probe"]==FROOT/(x["parent"]+"_probe-memory_receipt.json") for x in events if x["stage"]=="fit"))
        ck("retrieve_uses_own_fit",all(x["fit"]==FROOT/(x["parent"]+"_fit_receipt.json") for x in events if x["stage"]=="retrieve"))
        ck("CPU_retrieval_starts_before_next_GPU_fit",events[5]["stage"]=="retrieve" and events[6]["stage"]=="fit")
ck("no_actual_process_runtime_budget_or_GPU_operations",forbidden==[])
report={"kind":"native_v7_queue_finite_metadata_harness","checks_passed":len(checks),"checks":checks,"binding_counterexamples":findings,"forbidden_events":forbidden,"source_science_run":False,"original_runner_sha256":sha(RUNNER),"scope":"AST prefix ROOT rebased to owned fake filesystem; unchanged functions/tail mocked only at external process/budget boundaries. No actual queue was run."}
(OUT/"metadata_checks.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({"checks_passed":len(checks),"binding_counterexamples":len(findings),"forbidden_events":forbidden}))
