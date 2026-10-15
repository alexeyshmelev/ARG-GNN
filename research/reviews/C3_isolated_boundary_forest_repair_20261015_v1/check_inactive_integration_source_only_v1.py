"""Static inactive integration audit; imports no project modules."""
import ast,builtins,datetime,hashlib,json,symtable
from pathlib import Path
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
OUT=ROOT/"research/reviews/C3_isolated_boundary_forest_repair_20261015_v1"
TREE=OUT/"staged_tree"
OLD="research/reviews/C3_GNN_prefetch_current980_root_20261015_v1"
NEW="research/reviews/C3_GNN_prefetch_isolated980_root_20261015_v2"
OLDP="research/reviews/C3_GNN_prefetch_provider_20261015_v1/prefetch_provider_v1.py"
NEWP=NEW+"/prefetch_provider_isolated_v2.py"
NAMES=("american_c3_gnn_observations","american_c3_gnn_dataset","american_c3_gnn_stream_train","american_c3_gnn_run","american_c3_gnn_predict","american_c3_gnn_shared30mb","american_c3_gnn_selection")
checks=[];definitions=[]
def check(name,value):
 if not value:raise AssertionError(name)
 checks.append(name)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(p.relative_to(ROOT)),sha256=sha(p))
def reverse(s):
 for a,b in [(NEW,OLD),(NEW.replace("/","."),OLD.replace("/",".")),(OLD+"/prefetch_provider_isolated_v2.py",OLDP),((OLD+"/prefetch_provider_isolated_v2").replace("/","."),OLDP[:-3].replace("/",".")),("american_c3_gnn_prefetch_isolated980_exec_v2","american_c3_gnn_prefetch_current980_exec_v1"),("C3_GNN_prefetch_isolated980","C3_GNN_prefetch_current980"),("prefetch-isolated980","prefetch-current980"),("prefetch_isolated980","prefetch_current980"),("c3gnnprefetchisolated980","c3gnnprefetch980")]:
  s=s.replace(a,b)
 for n in NAMES:s=s.replace(n+"_isolated_v2",n+"_v1")
 return s
def defs(s):
 return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
def literal(s,name):
 return next(ast.literal_eval(n.value) for n in ast.parse(s).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
manifest=json.loads((OUT/"staged_source_install_manifest.v1.json").read_text())
targets={r["intended_ref"]["path"]:TREE/r["intended_ref"]["path"] for r in manifest["files"]}
for row in manifest["files"]:
 old=row["original_ref"];target=row["intended_ref"];stage=row["staged_ref"];p=ROOT/stage["path"]
 check("staged_hash:"+target["path"],sha(p)==stage["sha256"]==target["sha256"])
 check("original_unchanged:"+old["path"],sha(ROOT/old["path"])==old["sha256"])
 check("additive_target_absent:"+target["path"],not (ROOT/target["path"]).exists())
 if p.suffix!=".py":continue
 s=p.read_text();compile(s,str(p),"exec");check("compiles:"+target["path"],True)
 if row["change_kind"]!="isolated_sample_forest_encoding":
  a,b=defs((ROOT/old["path"]).read_text()),defs(reverse(s))
  exceptions={"source_paths","validate_closed_plan_stage"} if target["path"].endswith("/root_contract_v1.py") else set()
  check("same_definitions:"+target["path"],set(a)==set(b))
  for name,value in a.items():
   if name not in exceptions:check("definition_AST_unchanged:"+target["path"]+":"+name,b[name]==value)
  definitions.append(dict(target=target["path"],unchanged=len(a)-len(exceptions),exceptions=sorted(exceptions)))
 for n in ast.walk(ast.parse(s)):
  if isinstance(n,ast.ImportFrom) and n.module:
   module=n.module.split(".")[-1]
   if module in {x+"_isolated_v2" for x in NAMES}:check("successor_import:"+target["path"]+":"+module,"src/arggnn/"+module+".py" in targets)
 table=symtable.symtable(s,str(p),"exec")
 defined=set(table.get_identifiers())|set(dir(builtins))|{"__file__","__name__","__package__","__spec__","__loader__","__builtins__","__doc__","__annotations__","__cached__"}
 def missing(t):
  result={x.get_name() for x in t.get_symbols() if x.is_referenced() and x.is_global() and x.get_name() not in defined}
  for child in t.get_children():result.update(missing(child))
  return result
 check("no_undefined_globals:"+target["path"],not missing(table))
cfg=json.loads(targets[NEW+"/training_config.inactive.json"].read_text())
check("scientific_config_bytes_identical",sha(targets[NEW+"/training_config.inactive.json"])==sha(ROOT/OLD/"training_config.inactive.json"))
check("600updates30epochs20parents_fp32",cfg["epochs"]==30 and cfg["parents_per_epoch"]==20 and cfg["total_updates"]==600 and cfg["precision"]=="fp32" and cfg["checkpoint_epochs"]==[5,10,15,20,25,30] and cfg["execution_enabled"] is False)
oldreq=json.loads((ROOT/OLD/"request_train.root_active_v1.json").read_text())
req=json.loads(targets[NEW+"/request_train.proposed.inactive.json"].read_text())
caps=json.loads(targets[NEW+"/caps_train.proposed.inactive.json"].read_text())
check("inactive_request_caps",req["active"] is False and caps["active"] is False and "caps_ref" not in req)
check("all20_bindings_identical",req["bindings"]==oldreq["bindings"] and len(req["bindings"]["actual_train_bindings"])==20)
check("fresh_caps_ref",req["caps_template_ref"]==dict(path=NEW+"/caps_train.proposed.inactive.json",sha256=sha(targets[NEW+"/caps_train.proposed.inactive.json"])))
contract=targets[NEW+"/root_contract_v1.py"].read_text()
for key,field in [("HOOK_REF","hook_ref"),("TRAIN_SOURCE_REF","scientific_train_source_ref")]:
 check("request_source:"+key,literal(contract,key)==req[field])
 check("staged_source:"+key,sha(targets[req[field]["path"]])==req[field]["sha256"])
check("dispatcher_bytes_exact",targets["src/arggnn/american_c3_gnn_prefetch_isolated980_exec_v2.py"].read_bytes()==targets[NEW+"/canonical_dispatch.staged.py"].read_bytes())
check("fresh_namespace",literal(contract,"REL")==NEW and manifest["future_admission_path"]==NEW+"/admitted_train_v1/" and manifest["future_output"]=="experiments/results/C3_GNN_prefetch_isolated980_train_v1")
check("new_dispatch_only",literal(contract,"CANONICAL")=="src/arggnn/american_c3_gnn_prefetch_isolated980_exec_v2.py" and literal(contract,"GPU_MODULE")=="arggnn.american_c3_gnn_prefetch_isolated980_exec_v2")
check("own_training_terminal_unchanged",defs(reverse(contract))["validate_terminal"]==defs((ROOT/OLD/"root_contract_v1.py").read_text())["validate_terminal"])
check("separate_prior_observation_plan_route",'"prefetch_current980","prefetch_isolated980"' in contract and "PRIOR_PREFETCH_DRIVER_REF" in contract)
for path,h in literal(targets[NEWP].read_text(),"PINS").items():check("provider_pin:"+path,sha(targets[path])==h)
for path,h in literal(contract,"PREFETCH_SOURCE_PINS").items():check("root_pin:"+path,sha(targets.get(path,ROOT/path))==h)
trainer=targets[NEW+"/trainer_prefetch_v1.py"].read_text()
check("all20_observation_barrier_unchanged",defs(reverse(trainer))["run_train_v1"]==defs((ROOT/OLD/"trainer_prefetch_v1.py").read_text())["run_train_v1"])
predict=targets["src/arggnn/american_c3_gnn_predict_isolated_v2.py"].read_text()
check("all30_VAL_seal_barrier_unchanged",defs(reverse(predict))["seal_validation_matrix_v1"]==defs((ROOT/"src/arggnn/american_c3_gnn_predict_v1.py").read_text())["seal_validation_matrix_v1"])
selection=targets["src/arggnn/american_c3_gnn_selection_isolated_v2.py"].read_text()
check("VAL_truth_scoring_unchanged",defs(reverse(selection))==defs((ROOT/"src/arggnn/american_c3_gnn_selection_v1.py").read_text()))
preserved=[ref(ROOT/("src/arggnn/"+n+".py")) for n in ("mutation_routed_tree_observations_v1","mutation_routed_whole_tree_gnn_v1","mutation_routed_pyg_gnn_v1","american_c3_gnn_model_v1","american_c3_gnn_targets_v1")]
receipt=dict(schema="C3_isolated_inactive_integration_static_v1",observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status="PASS",check_count=len(checks),checks=checks,definition_equivalence=definitions,staged_files=len(manifest["files"]),generic_helpers_model_targets_preserved=preserved,canonical_install_performed=False,actual_admission=False,actual_execution=False,guard_or_accounting_mutation=False,GPU_probe=False,ancestry_truth_opened=False)
p=OUT/"integration_static_receipt.v1.json";p.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
print(json.dumps(dict(status="PASS",check_count=len(checks),receipt_ref=ref(p)),sort_keys=True),flush=True)
