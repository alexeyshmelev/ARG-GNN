"""Source-only H061 reporting checks; never execute the updater or scientific code."""
from pathlib import Path
import ast,copy,hashlib,json,math
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
OUT=ROOT/"research/reviews/H061_scoreboard_reporting_source_20261017_v1"
p=ROOT/"scripts/update_current_scoreboard.py"
before=(OUT/"update_current_scoreboard.before.py").read_text();after=p.read_text()
compile(after,str(p),"exec")
oldtree=ast.parse(before);tree=ast.parse(after)
def need(v,msg):
 if not v:raise AssertionError(msg)
def fn(nodes,name):return next(n for n in nodes if isinstance(n,ast.FunctionDef)and n.name==name)
oldfunctions={n.name:n for n in oldtree.body if isinstance(n,ast.FunctionDef)}
functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
for name,node in oldfunctions.items():
 if name!="main":need(ast.dump(node,include_attributes=False)==ast.dump(functions[name],include_attributes=False),"unrelated function changed "+name)
added=set(functions)-set(oldfunctions)
need(added=={"h061_accepted_test","h061_test_summary","h061_test_lines"},"only three narrow H061 helpers")
need(after.count('h061_current_summary+" |"')==1 and after.count('h061_status+" "+h061_current_summary+')==1,"current table and README use TEST when accepted")
need('h061_lines=h061_test_lines(h061_test)+'in after and 'h061-chickenpox-accepted-test-outcome'in after,"new current TEST subsection, historical VAL retained")
names=("fmt","h061_forecast_summary","h061_accepted_test","h061_test_summary","h061_test_lines")
nodes=[copy.deepcopy(functions[n])for n in names]
constant=next(n for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=="H061_SCORE_ACCEPTANCE"for t in n.targets))
env=dict(Path=Path,ROOT=ROOT,hashlib=hashlib,json=json,math=math)
exec(compile(ast.Module(body=[constant]+nodes,type_ignores=[]),str(p)+"#reporting-functions-only","exec"),env)
accepted_ref=env["H061_SCORE_ACCEPTANCE"]
# Real exact acceptance and scored JSON lineage, without main(), runtime status, payload or resource access.
observed={}
class MetadataPath:
 def __init__(self,path):self.path=path
 def resolve(self):return self if self.path.resolve()==self.path else self.path.resolve()
 def is_file(self):return self.path.is_file()
 def is_symlink(self):return self.path.is_symlink()
 def stat(self):return self.path.stat()
 def read_bytes(self):
  relative=str(self.path.relative_to(ROOT))
  need(self.path.suffix==".json"and relative.startswith(("research/","experiments/results/")),"no scientific payload")
  data=self.path.read_bytes();observed[relative]=data;return data
class MetadataRoot:
 def __truediv__(self,path):return MetadataPath(ROOT/path)
env["ROOT"]=MetadataRoot()
real=env["h061_accepted_test"](dict(score_acceptance_ref=accepted_ref))
need(real["ready"],"actual accepted TEST lineage: "+str(real))
need(len(observed)==13,"exact bounded metadata lineage")
pending=env["h061_accepted_test"]({});need(pending["ready"]is False,"absent acceptance stays pending")
bad=dict(accepted_ref,sha256="0"*64)
need(env["h061_accepted_test"](dict(score_acceptance_ref=bad))["ready"]is False,"wrong acceptance seal stays pending")
need(len(observed)==13,"absent/wrong ref reads nothing")
lines=env["h061_test_lines"](real);summary=env["h061_test_summary"](real)
need("| Full | 13.907918 ± 0.242561 | **24.268971 ± 0.094496** |"in lines,"Full RMSE winner bold")
need("| NoMem | **13.648097 ± 0.050932** | 24.408159 ± 0.106380 |"in lines,"NoMem MAE winner bold")
need("TEST" in summary and "Registered memory utility failed"in summary,"README TEST summary")
need(any("conditional 95%"in l and "+0.105412"in l and "+0.427353"in l for l in lines),"conditional interval explicit")
need(env["h061_test_lines"](pending)==[]and env["h061_test_summary"](pending)is None,"pending hides TEST values")
need(env["h061_forecast_summary"]({"training_milestone":None})=="Selected VAL forecast metrics pending; TEST withheld","old pending behavior")
(OUT/"accepted_TEST_preview.md").write_text("\n".join(lines)+"\n\nREADME/current-row summary:\n\n"+summary+"\n")
# Reject corrupt real bytes before any scored-value opening, without filesystem mutation.
class CorruptPath(MetadataPath):
 def read_bytes(self):
  raw=super().read_bytes()
  return raw+b" "if self.path.name=="actual_wait.json"else raw
class CorruptRoot:
 def __truediv__(self,path):return CorruptPath(ROOT/path)
env["ROOT"]=CorruptRoot();need(env["h061_accepted_test"](dict(score_acceptance_ref=accepted_ref))["ready"]is False,"worker SHA mismatch")
# In-memory semantic mutations use the same extracted function with only checked()
# replaced by a fabricated metadata reader. No checksum bypass is installed in source.
records={k:json.loads(v)for k,v in observed.items()}
fake_env=dict(env,ROOT=ROOT)
def fixture_read(record):return copy.deepcopy(records[record["path"]])
fake_env["_fixture_read"]=fixture_read
node=copy.deepcopy(functions["h061_accepted_test"])
for child in node.body:
 if isinstance(child,ast.FunctionDef)and child.name=="checked":
  child.body=ast.parse("return _fixture_read(record)").body
ast.fix_missing_locations(node)
exec(compile(ast.Module(body=[node],type_ignores=[]),str(p)+"#fabricated-metadata-only","exec"),fake_env)
need(fake_env["h061_accepted_test"](dict(score_acceptance_ref=accepted_ref))["ready"],"complete fabricated metadata")
accepted=records[accepted_ref["path"]]
terminal=records[accepted["actual_outer_terminal_ref"]["path"]]
scores=records[accepted["scores_ref"]["path"]]
worker_path=accepted["actual_wait_ref"]["path"];score_path=accepted["scores_ref"]["path"]
prediction_path=scores["prediction_barrier_ref"]["path"]
negative=[]
def reject(name,path,mutator):
 original=copy.deepcopy(records[path]);mutator(records[path])
 try:need(fake_env["h061_accepted_test"](dict(score_acceptance_ref=accepted_ref))["ready"]is False,name)
 finally:records[path]=original
 negative.append(name)
reject("outer_unconsumed",accepted["actual_outer_terminal_ref"]["path"],lambda v:v.update(actual_terminal_consumed=False))
reject("outer_nonzero",terminal["subprocess_terminal_ref"]["path"],lambda v:v.update(actual_returncode=1))
reject("worker_nonzero",worker_path,lambda v:v.update(returncode=1))
reject("worker_wait_unknown",worker_path,lambda v:v.update(actual_direct_worker_wait=False))
reject("worker_not_drained",worker_path,lambda v:v.update(all_owned_processes_and_groups_drained=False))
reject("worker_cleanup_error",worker_path,lambda v:v.update(cleanup_errors=["retained failure"]))
reject("worker_group_present",worker_path,lambda v:v["group_observations"][0].update(absent=False))
reject("score_completion_new_fit",accepted["score_completion_ref"]["path"],lambda v:v.update(new_core_fits=1))
reject("wrong_score_barrier_join",accepted["score_barrier_ref"]["path"],lambda v:v.update(scores_ref={}))
reject("missing_product_authentication",score_path,lambda v:v.update(all15products_authenticated_before_scoring=False))
reject("trimmed_TEST_week",score_path,lambda v:v["target_indices"].pop())
reject("trimmed_county",score_path,lambda v:v["county_order"].pop())
reject("missing_core_fit",score_path,lambda v:v["core_scores"].pop())
reject("wrong_seed_SD",score_path,lambda v:v["aggregate"]["full"]["mae_cases"].update(seeds=[101,202]))
reject("nonfinite_metric",score_path,lambda v:v["aggregate"]["full"]["mae_cases"].update(mean=float("nan")))
reject("invented_utility_pass",score_path,lambda v:v.update(memory_utility_gate_passed=True))
reject("legacy_GRU_accepted",score_path,lambda v:v.update(legacy_GRU_status="accepted"))
reject("missing_comparator_prediction",prediction_path,lambda v:v["comparator_predictions"].pop())
reject("duplicate_prediction_product",prediction_path,lambda v:v["chronology_predictions"][0].update(path=v["predictions"][0]["path"]))
reject("wrong_selected_checkpoint",prediction_path,lambda v:v["predictions"][0].update(selected_checkpoint_ref={}))
reject("scored_prediction_barrier",prediction_path,lambda v:v.update(TEST_scored=True))
reject("wrong_registered_budget",scores["registration_ref"]["path"],lambda v:v.update(new_optimizer_updates=4799))
reject("reduced_frozen_TEST_support",scores["source_protocol_ref"]["path"],lambda v:v["splits"].update(test_target_indices=[444,522]))
result=dict(schema="H061_scoreboard_reporting_source_checks_v1",status="PASS",source_ref=dict(path="scripts/update_current_scoreboard.py",sha256=hashlib.sha256(p.read_bytes()).hexdigest()),base_source_sha256=hashlib.sha256(before.encode()).hexdigest(),compiled_in_memory=True,updater_main_not_executed=True,unchanged_existing_functions_except_main=list(n for n in oldfunctions if n!="main"),added_H061_helpers=sorted(added),actual_accepted_JSON_metadata_files_read=len(observed),absent_or_invalid_acceptance_remains_pending=True,real_worker_SHA_rejection=True,negative_fabricated_metadata_count=len(negative),negative_fixtures=negative,bold_winners={"TEST_MAE":"NoMem","TEST_RMSE":"Full"},historical_selected_VAL_preserved=True,no_runtime_status_or_output_report_edits_no_resources_no_Git=True)
(OUT/"checks.v1.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
print(json.dumps(result,sort_keys=True))

