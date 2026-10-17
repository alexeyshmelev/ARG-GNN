"""Narrow source/AST/RAM verification for H058 RSS absence-only successor."""
from pathlib import Path
from types import SimpleNamespace
import ast,copy,hashlib,json
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
D="research/reviews/H058_GPU0_full_training_successor_source_20261017_v7"
B="research/reviews/H058_GPU0_full_training_successor_source_20261017_v5"
rows=[]
def ck(n,x=True):
    if not x:raise AssertionError(n)
    rows.append(dict(name=n,status="PASS"))
def reject(n,f,kind=Exception):
    try:f()
    except kind:ck(n)
    else:raise AssertionError(n)
def ref(p):return dict(path=p,sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest())
def j(p):return json.loads((ROOT/p).read_bytes())
def tree(p):return ast.parse((ROOT/p).read_text())
def fn(t,n):return next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name==n)
def dump(n):return ast.dump(n,include_attributes=False)
def norm(n):return dump(n).replace(D,B).replace(D.replace("/","."),B.replace("/",".")).replace("H058_GPU0_v7_","H058_GPU0_v5_").replace("H058-GPU0-v7-","H058-GPU0-v5-")
def emit(nodes,ns):
    z=ast.Module(body=copy.deepcopy(nodes),type_ignores=[]);ast.fix_missing_locations(z);exec(compile(z,"<isolated_RSS_source_only>","exec"),ns)
names=("__init__.py","common_v1.py","context_factory_v1.py","direct_gpu0_v1.py","observations_gpu_v1.py","producer_v1.py","root_driver_v1.py")
old={n:tree(B+"/"+n) for n in names};new={n:tree(D+"/"+n) for n in names}
for n in names:
    compile(new[n],"<parse_only>","exec")
    if n not in ("common_v1.py","root_driver_v1.py"):assert norm(new[n])==dump(old[n])
ck("five_whole_production_ASTs_namespace_only")
z=copy.deepcopy(new["common_v1.py"]);z.body=z.body[:len(old["common_v1.py"].body)]
r=fn(z,"validate_request_v1");calls=[n for n in r.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=="validate_prior_v5_terminal_v6"]
assert len(calls)==1;r.body.remove(calls[0]);ck("whole_common_v5_recovered_except_new_exact_failure_gate",norm(z)==dump(old["common_v1.py"]))
a=copy.deepcopy(new["root_driver_v1.py"]);b=copy.deepcopy(old["root_driver_v1.py"])
a.body=[n for n in a.body if not(isinstance(n,ast.FunctionDef) and n.name=="rss_identity_v1")]
b.body=[n for n in b.body if not(isinstance(n,ast.FunctionDef) and n.name=="rss_identity_v1")]
ck("whole_driver_v5_recovered_except_RSS_monitor",norm(a)==dump(b))
owned=tree("research/reviews/American_C3_native_MLM_amendment_v12/owned_lifecycle_v2.py")
same_ns={};emit([fn(owned,"_same")],same_ns)
def run(function,sequence,statm="0 7"):
    seq=list(sequence)
    def observe(pid):
        x=seq.pop(0)
        if isinstance(x,BaseException):raise x
        return x
    def readstatm():
        if isinstance(statm,BaseException):raise statm
        return statm
    ns=dict(Path=lambda p:SimpleNamespace(read_text=readstatm),os=SimpleNamespace(sysconf=lambda k:4096))
    emit([fn(new["common_v1.py"],"require"),function],ns)
    return ns["rss_identity_v1"](SimpleNamespace(_same=same_ns["_same"],_stat=observe),dict(pid=10,start=100))
identity=dict(pid=10,start=100);other=dict(pid=10,start=101)
before=fn(old["root_driver_v1.py"],"rss_identity_v1");after=fn(new["root_driver_v1.py"],"rss_identity_v1")
reject("old_monitor_reproduces_harmless_post_read_disappearance_failure",lambda:run(before,[identity,None]),ValueError)
ck("new_post_read_confirmed_absence_returns0",run(after,[identity,None])==0)
ck("new_pre_read_confirmed_absence_returns0",run(after,[None])==0)
ck("stable_same_owned_identity_keeps_exact_page_accounting",run(after,[identity,identity])==run(before,[identity,identity])==7*4096)
for error in (FileNotFoundError(),ProcessLookupError()):
    ck("statm_disappearance_returns0:"+type(error).__name__,run(after,[identity],error)==0)
reject("positive_pre_read_PID_start_mismatch_fails",lambda:run(after,[other]),ValueError)
reject("positive_post_read_PID_start_mismatch_fails",lambda:run(after,[identity,other]),ValueError)
for label,sequence,statm in (("pre_stat_permission",[PermissionError()],"0 7"),("post_stat_permission",[identity,PermissionError()],"0 7"),("statm_permission",[identity],PermissionError()),("pre_stat_unknown_IO",[OSError(5,"RAM IO")],"0 7"),("post_stat_unknown_IO",[identity,OSError(5,"RAM IO")],"0 7"),("statm_unknown_IO",[identity],OSError(5,"RAM IO"))):
    reject("all_other_errors_propagate:"+label,lambda sequence=sequence,statm=statm:run(after,sequence,statm),OSError)
reject("malformed_statm_remains_failstop",lambda:run(after,[identity],"malformed"),IndexError)
# Execute only the new metadata prior-failure predicate with genuine pinned small JSON.
ns=dict(Path=Path)
emit([n for n in new["common_v1.py"].body if isinstance(n,ast.Assign)],ns)
emit([fn(new["common_v1.py"],n) for n in ("require","local","ref","read","raw","validate_prior_v5_terminal_v6")],ns)
ns.update(hashlib=hashlib,json=json)
prior=j(D+"/prior_CPU_disposition.INACTIVE.template.v1.json")
prior.update(prior_GPU_v5_files_charges_clocks_sources_and_failure_preserved=True,prior_GPU_v5_future_launch_resume_write_routes_permanently_revoked=True,no_CPU_or_GPU_optimizer_state_transfer=True)
ns["validate_prior_v5_terminal_v6"](prior);ck("new_prior_gate_accepts_genuine_47121outer241_worker_minus15_drain_UNKNOWN_totals")
bad=copy.deepcopy(prior);bad["prior_GPU_v5_future_launch_resume_write_routes_permanently_revoked"]=False
reject("new_prior_gate_requires_permanent_old_route_revocation",lambda:ns["validate_prior_v5_terminal_v6"](bad),ValueError)
bad=copy.deepcopy(prior);bad["prior_GPU_v5_actual_optimizer_steps"]={k:0 for k in bad["prior_GPU_v5_actual_optimizer_steps"]}
reject("zero_verified_updates_cannot_be_relabeled_zero_actual_totals",lambda:ns["validate_prior_v5_terminal_v6"](bad),ValueError)
m=j(D+"/source_manifest.v1.json");oldm=j(B+"/source_manifest.v1.json")
assert all(r in m["refs"] for r in oldm["refs"])
for r in m["refs"]:assert ref(r["path"])==r
assert m["source_sha256"]=={r["path"]:r["sha256"] for r in m["refs"]}
ck("complete355_source_ref_closure_preserves_all320_v5pins",len(m["refs"])==355)
for p,h in j(D+"/sealed_v6_baseline.v7.json").items():assert ref(p)["sha256"]==h
ck("all_original_sealed_v5_files_unchanged")
q=j(D+"/ROOT_request.INACTIVE.template.v1.json")
ck("all_ROOT_execution_review_authority_remains_inactive",q["execution_authorized"] is False and q["source_review_passed"] is False and q["prior_H058_CPU_disposition_ref"] is None)
ck("genuine_original_ROOT_clocks_and_caps_unchanged",q["created_at_utc"]=="2026-10-17T07:19:39.628628+00:00" and q["expires_at_utc"]=="2026-10-18T10:05:49.430020+00:00" and q["caps"]==j(B+"/ROOT_request.INACTIVE.template.v1.json")["caps"])

# Actual inert imports are essential: path text alone did not catch the rejected v6 binding error.
import importlib,sys
package=D.replace("/",".")
mods={n:importlib.import_module(package+"."+n) for n in ("common_v1","root_driver_v1","direct_gpu0_v1","context_factory_v1","producer_v1")}
c7=mods["common_v1"];driver7=mods["root_driver_v1"];direct7=mods["direct_gpu0_v1"];context7=mods["context_factory_v1"]
ck("actual_imported_common_and_driver_HERE_DRIVER_MANIFEST_v7",driver7.HERE==c7.HERE==D and driver7.DRIVER==D+"/root_driver_v1.py" and driver7.MANIFEST==D+"/source_manifest.v1.json")
ck("actual_imported_direct_and_context_bind_same_v7_common",direct7.HERE==context7.HERE==D and direct7.validate_native_resource_disposition_v1 is c7.validate_native_resource_disposition_v1 and context7.validate_input_v1 is c7.validate_input_v1)
ck("actual_imported_producer_own_namespace_v7",mods["producer_v1"].HERE==D)
for n in names:
 for node in ast.walk(new[n]):
  if isinstance(node,ast.ImportFrom) and node.module:
   assert not node.module.startswith(B.replace("/",".")) and not node.module.startswith("research.reviews.H058_GPU0_full_training_successor_source_20261017_v6")
ck("no_stale_operational_dotted_imports")
command_node=next(n for n in ast.walk(fn(new["root_driver_v1.py"],"execute_v1")) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="command" for t in n.targets))
commandns=dict(vars(driver7));commandns.update(input_ref=dict(path="RAM_ONLY/input.json",sha256="0"*64),gate_read=17)
emit([command_node],commandns)
ck("actual_emitted_child_argv_module_is_v7",commandns["command"][3]==package+".root_driver_v1")
return_node=next(n for n in reversed(fn(new["root_driver_v1.py"],"admit_v1").body) if isinstance(n,ast.Return))
returnns=dict(vars(driver7));returnns.update(input_ref=dict(path="RAM_ONLY/input.json",sha256="0"*64),admitted_ref=dict(path="RAM_ONLY/admitted.json",sha256="1"*64))
answer=eval(compile(ast.Expression(return_node.value),"<actual_admission_return_only>","eval"),returnns)
ck("actual_emitted_admission_one_shot_module_is_v7",answer["one_shot_exec_argv"][4]==package+".root_driver_v1")
ck("no_Torch_PyG_numerical_import_from_inert_actual_entry",not any(n=="torch" or n.startswith(("torch.","torch_geometric","numpy")) for n in sys.modules))

result=dict(schema="H058_GPU0_v6_narrow_RSS_source_AST_RAM_checks_v1",status="PASS",checks=rows,count=len(rows),source_ref=ref(D+"/root_driver_v1.py"),source_manifest_ref=ref(D+"/source_manifest.v1.json"),source_only=True,independent=False,no_actual_runtime_guard_process_GPU_payload_or_numerical_execution=True)
with (ROOT/D/"checks.result.v7.json").open("x") as f:json.dump(result,f,sort_keys=True,indent=2);f.write("\n")
print(json.dumps(dict(status="PASS",targeted_checks=len(rows),checks_ref=ref(D+"/checks.result.v7.json")),sort_keys=True))

