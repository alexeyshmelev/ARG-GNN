"""Mocked v3 process-observation classifier/full-guard paths; no /proc reads."""
import ast
from copy import deepcopy
from datetime import datetime,timezone
import hashlib,json,re
from pathlib import Path
from types import SimpleNamespace
root=Path.cwd();out=root/"research/reviews/process_snapshot_retry_v3_independent"
sha=lambda p:hashlib.sha256((root/p).read_bytes()).hexdigest()
old_path="src/arggnn/project_budget_v9_consumer_retry_v2.py"
new_path="src/arggnn/project_budget_v9_consumer_retry_v3.py"
assert sha(old_path)=="75fd586fc8e0733027451a6d9485b7e49f10c3d8d7cf69c09cdec4b9f1b7ba1e"
assert sha(new_path)=="dc735ee914dcddca9cd6078948225fa9652b10c4decef6f42df508798649de2b"
old=ast.parse((root/old_path).read_text());new=ast.parse((root/new_path).read_text())
get=lambda tree,name:next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
normalized=deepcopy(new)
normalized.body=[n for n in normalized.body if not (isinstance(n,ast.Import) and [a.name for a in n.names]==["re"]) and not(isinstance(n,ast.FunctionDef) and n.name=="transient_process_observation")]
for n in normalized.body:
    if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=="SOURCE":n.value.value=old_path
    if isinstance(n,ast.FunctionDef) and n.name=="budget_check_v9":
        h=next(x for x in ast.walk(n) if isinstance(x,ast.ExceptHandler))
        assert ast.unparse(h.body[0].test)=="not transient_process_observation(error) or attempt == 15"
        old_h=next(x for x in ast.walk(get(old,"budget_check_v9")) if isinstance(x,ast.ExceptHandler))
        h.body[0].test=deepcopy(old_h.body[0].test)
assert ast.dump(normalized,include_attributes=False)==ast.dump(old,include_attributes=False)
outer_message="Cannot inspect an own-user or recorded project process"
def permission(path,kind=PermissionError,message=outer_message,context_only=False):
    error=ValueError(message);cause=kind(13,"mock denied",path)
    if context_only:error.__context__=cause
    else:error.__cause__=cause
    return error
classifier_env={"re":re}
exec(compile(ast.Module(body=[get(new,"transient_process_observation")],type_ignores=[]),new_path,"exec"),classifier_env)
classify=classifier_env["transient_process_observation"]
valid_paths=["/proc/7/fd","/proc/7/fd/0","/proc/7/fd/19","/proc/7/fdinfo/0","/proc/7/fdinfo/19","/proc/7/cwd","/proc/7/maps"]
invalid_paths=["/proc/0/fd","/proc/07/fd","/proc/-7/fd","/proc/٧/fd","/proc/7/status","/proc/7/fdinfo","/proc/7/fd/x","/proc/7/fd/1/x","/proc/7/fd/../maps","/proc/7/maps/","/tmp/proc/7/fd","/proc/7/cwd\n",None,b"/proc/7/fd"]
assert all(classify(permission(p)) for p in valid_paths)
assert not any(classify(permission(p)) for p in invalid_paths)
rejected_errors=[ValueError(outer_message),permission("/proc/7/fd",context_only=True),permission("/proc/7/fd",kind=FileNotFoundError),permission("/proc/7/fd",message=outer_message+" ")]
assert not any(classify(e) for e in rejected_errors)
transients=["Process inventory changed during writer inspection","Process descriptors changed during writer inspection"]
assert all(classify(ValueError(s)) for s in transients)
refs={k:{"path":p,"sha256":k} for k,p in {"implementation":"scripts/project_budget_v9.py","launcher":"scripts/launch_reserved_gpu_v6.py","consumer_adapter":new_path}.items()}
expected={r["path"]:r["sha256"] for r in refs.values()}
pins={k:k for k in ("legacy_ledger_sha256","supplemental_ledger_sha256","journal_tail_sha256")}
own={"reserved_bytes":10000,"output":str(root/"mock-output"),"consumer_bindings":list(refs.values())}
storage={"kind":"project_budget_v9_storage_record","active":True,"consumer_uses_v9_guard":True,"budget_components":refs,"reservation":own,"job_id":"mock","reserved_new_bytes":10000,"output_directory":"mock-output","initial_ledger_pins":pins}
subset=ast.Module(body=[get(new,n) for n in ("_source_ref","transient_process_observation","budget_check_v9")],type_ignores=[])
def run(sequence,first=False):
    calls=[];sleeps=[]
    def guard(*args,**kwargs):
        calls.append((args,kwargs));value=sequence[min(len(calls)-1,len(sequence)-1)]
        if isinstance(value,BaseException):raise value
        return value
    env={"ROOT":root,"SOURCE":new_path,"re":re,"datetime":datetime,"timezone":timezone,"time":SimpleNamespace(sleep=sleeps.append),"budget":SimpleNamespace(v4=SimpleNamespace(file_bytes=lambda r,p:(b"",expected.get(p,"unknown")),canonical=lambda r,p:r/p),check_reservation=guard)}
    exec(compile(subset,new_path,"exec"),env)
    result=error=None
    try:result=env["budget_check_v9"](storage,first=first,boundary=True,required_write_bytes=1234)
    except BaseException as caught:error=caught
    assert all(args==(root,own) and kw=={"expected_source_sha256":"implementation","expected_launcher_sha256":"launcher","boundary":True,"required_write_bytes":1234} for args,kw in calls)
    return result,error,calls,sleeps
success={"full_guard_passed":True,**pins}
r,e,c,s=run([permission("/proc/7/fd"),ValueError(transients[0]),success],first=True)
assert e is None and r["full_guard_passed"] and r["transient_process_snapshot_retries"]==2 and len(c)==3 and s==[.25,.5]
last=permission("/proc/7/fd")
r,e,c,s=run([last])
assert r is None and e is last and len(c)==16 and len(s)==15 and sum(s)==25.75
for error in rejected_errors+[permission(p) for p in invalid_paths]+[RuntimeError(transients[0]),ValueError("unrelated guard failure")]:
    r,e,c,s=run([error]);assert r is None and e is error and len(c)==1 and not s
hard=ValueError("hard final failure")
r,e,c,s=run([permission("/proc/7/maps"),hard])
assert e is hard and len(c)==2 and s==[.25]
bad=dict(success,legacy_ledger_sha256="mismatch")
r,e,c,s=run([permission("/proc/7/fd/3"),bad],first=True)
assert r is None and isinstance(e,ValueError) and "Initial exact" in str(e) and len(c)==2
report={"verdict":"PASS","source_ref":{"path":new_path,"sha256":sha(new_path)},"v2_ref":{"path":old_path,"sha256":sha(old_path)},"all_other_AST_identical":True,"qualified_canonical_inspection_examples":valid_paths,"direct_explicit_PermissionError_cause_required":True,"nonmatching_paths_messages_causes_immediate_failure":True,"full_guard_and_write_arguments_retried_unchanged":True,"success_requires_final_complete_guard":True,"persistent_qualified_failure":{"attempts":16,"intentional_sleep_seconds":25.75,"final_error_preserved":True},"post_success_initial_ledger_guard_preserved":True,"scope":{"actual_proc_scans":False,"real_sleep":False,"GPU_or_data_or_optimizer":False,"live_budget_or_ledger":False,"admission_or_launch":False}}
(out/"adapter_checks.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
