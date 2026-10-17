"""Independent numerical-specific adversarial cases using disclosed partial RAM scaffold only."""
from pathlib import Path
import ast,copy,json
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN");OUT=ROOT/"research/reviews/Census_numerical_inline_independent_20261017_v1"
tree=ast.parse((OUT/"independent_checks_v1.py").read_bytes())
cut=next(i for i,n in enumerate(tree.body)if isinstance(n,ast.Assign)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Name)and n.value.func.id=="seam_case")
ns={"__name__":"independent_numeric_partial_RAM"}
exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(tree.body[:cut]),type_ignores=[])),"independent-numeric-partial-RAM","exec"),ns)
fix=ns["fix"];cases=[]
mutations=[
("original_supervisor_PID",lambda t:t["known_PID_start_rows"][0].update(pid=1306273)),
("original_supervisor_start",lambda t:t["known_PID_start_rows"][0].update(start_ticks=436412147)),
("original_worker_PID",lambda t:t["known_PID_start_rows"][1].update(pid=1306274)),
("original_worker_start",lambda t:t["known_PID_start_rows"][1].update(start_ticks=436412155)),
("original_worker_group",lambda t:t.update(known_process_groups=[1306271])),
("original_scope",lambda t:t["original_reservation"].update(output="RAM/other_scope")),
("original_expiry",lambda t:t.update(original_expiry_unchanged="2026-10-11T10:28:47.092270+00:00")),
("unexpected_auxiliary",lambda t:t["original_reservation"].update(auxiliary_scopes=["RAM/aux"]))]
for label,mutate in mutations:
 z=fix["callback_case"](label,target_mutate=mutate);failed=False
 try:z["cbns"]["authorize_inline_ROOT_retirement_v1"](guard=z["g"],context=z["ctx"],finite=lambda:None)
 except ValueError:failed=True
 assert failed and not z["fs"].records
 cases.append({"case":label,"rejected_before_metadata_write":True})
for label,observe in [("stale_targets","old"),("live_or_reused_known_group","live_group")]:
 z=fix["callback_case"](label,observe=observe);failed=False
 try:z["cbns"]["authorize_inline_ROOT_retirement_v1"](guard=z["g"],context=z["ctx"],finite=lambda:None)
 except ValueError:failed=True
 assert failed and not z["fs"].records
 cases.append({"case":label,"rejected_before_metadata_write":True})
z=fix["callback_case"]("unknown_target_permission")
def denied(pgid,sig):raise PermissionError("RAM unknown group")
z["g"]._observe_targets.__globals__["os"].killpg=denied
failed=False
try:z["cbns"]["authorize_inline_ROOT_retirement_v1"](guard=z["g"],context=z["ctx"],finite=lambda:None)
except PermissionError:failed=True
assert failed and not z["fs"].records
cases.append({"case":"unknown_target_permission","rejected_before_metadata_write":True})
result={"schema":"Census_numerical_independent_specialization_RAM_checks_v1","status":"PASS","source_only":True,"independent":True,"cases":cases,"assertions":len(cases),"partial_author_RAM_setup_classes_reused":True,"actual_original_pure_target_observer_and_callback_executed":True,"operational_target_or_IO_layers_are_RAM_only":True,"actual_process_operations":False,"actual_files_written_by_callback":False,"actual_whole_capability_supplied":False}
with(OUT/"specialization_RAM.result.v1.json").open("x")as f:json.dump(result,f,sort_keys=True,indent=2);f.write("\n")
print(json.dumps({k:v for k,v in result.items()if k!="cases"},sort_keys=True))
