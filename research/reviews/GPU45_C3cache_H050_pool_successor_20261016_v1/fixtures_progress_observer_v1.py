"""Pure parent-step observer stub fixture. No Torch/trainer import/hook/job."""
from pathlib import Path
import ast,functools,hashlib,json,types
HERE=Path(__file__).resolve().parent
SOURCE=HERE/"c3_progress_observer_v1.py"
tree=ast.parse(SOURCE.read_text())
ns=dict(Path=Path,functools=functools,hashlib=hashlib,os=types.SimpleNamespace(getpid=lambda:1234))
nodes=[n for n in tree.body if isinstance(n,ast.Assign) or isinstance(n,ast.FunctionDef) and n.name in ("require","wrap_trainer_v1")]
mod=ast.Module(body=nodes,type_ignores=[]);ast.fix_missing_locations(mod);exec(compile(mod,"<isolated_C3_observer>","exec"),ns)
calls=[];events=[];returned=dict(whole_parent_optimizer_updates=1)
arg=object();kw=object();failure=RuntimeError("original fixture failure");mode=["success"]
def original(*args,**kwargs):
    calls.append((args,kwargs))
    if mode[0]=="raise":raise failure
    return returned
module=types.SimpleNamespace(__file__=str(ns["ROOT"]/ns["TRAINER_REF"]["path"]),checkpointed_complete_parent_step_v1=original)
ns["wrap_trainer_v1"](module,emit=events.append)
results=[]
def passed(name,ok):assert ok,name;results.append(dict(case=name,passed=True))
for count in range(25):
    value=module.checkpointed_complete_parent_step_v1(arg,device=kw)
    assert value is returned
passed("one_original_call_per_invocation_same_args_and_return_identity",
    len(calls)==25 and all(a==(arg,) and k=={"device":kw} for a,k in calls))
passed("successful_events_only_at1_and_every10",
    [e["successful_whole_parent_optimizer_updates"] for e in events]==[1,10,20]
    and all(e["pid"]==1234 and e["job"]==ns["JOB"] for e in events))
mode[0]="raise"
try:module.checkpointed_complete_parent_step_v1(arg,device=kw)
except RuntimeError as error:passed("same_original_exception_propagates",error is failure and len(calls)==26 and len(events)==3)
else:raise AssertionError("Original exception swallowed")
mode[0]="success";returned["whole_parent_optimizer_updates"]=0
for count in range(10):module.checkpointed_complete_parent_step_v1()
passed("unsuccessful_update_returns_do_not_increment",len(events)==3)
returned["whole_parent_optimizer_updates"]=1
for count in range(5):module.checkpointed_complete_parent_step_v1()
passed("success_count_resumes_without_failed_or_zero_steps",events[-1]["successful_whole_parent_optimizer_updates"]==30)
before=module.checkpointed_complete_parent_step_v1;ns["wrap_trainer_v1"](module,emit=events.append)
passed("private_module_wrap_is_idempotent",module.checkpointed_complete_parent_step_v1 is before)
def broken_emit(value):raise BrokenPipeError("fixture closed log")
module2=types.SimpleNamespace(__file__=module.__file__,checkpointed_complete_parent_step_v1=original)
ns["wrap_trainer_v1"](module2,emit=broken_emit)
passed("logging_failure_does_not_change_scientific_return",module2.checkpointed_complete_parent_step_v1() is returned)
passed("bounded_events_and_no_scientific_object_reads",all(len(json.dumps(e))<256 for e in events))
print(json.dumps(dict(schema="C3_shared_progress_observer_pure_fixtures_v1",source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    cases=results,cases_passed=len(results),actual_Torch_trainer_imports_hooks_model_steps_jobs_or_GPU_probes=0),indent=2,sort_keys=True))
