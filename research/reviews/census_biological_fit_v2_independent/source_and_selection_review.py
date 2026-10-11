import ast,hashlib,json,random
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
OUT=ROOT/"research/reviews/census_biological_fit_v2_independent"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,value):
    with (OUT/name).open("x") as f:json.dump(value,f,indent=2,allow_nan=False);f.write("\n")
checks=[]
def check(value,label):
    if not value:raise AssertionError(label)
    checks.append(label)
oldp=ROOT/"src/arggnn/census_biological_fit_v1.py";newp=ROOT/"src/arggnn/census_biological_fit_v2.py"
old=oldp.read_text();new=newp.read_text()
check(sha(newp)=="c21884ad90a2c940b49968353545edc52d345c0c7bc10fd87c492c54d4b38c34","exact reviewed successor source")
check(sha(ROOT/"configs/census_biological_fit_v2.json")=="a512816bbab61e80052e7466613d2de4c861b77ef61d0d35cd50fbf81905c621","exact successor config")
normalized=new.replace("census_biological_fit_v2","census_biological_fit_v1")
for token in ["admission","runtime","storage","launcher","profile_gate","evidence"]:
    normalized=normalized.replace("census_biological_fit_"+token+"_v2","census_biological_fit_"+token+"_v1")
normalized=normalized.replace("if state['best_epoch'] is None or nll<state['best']:","if nll<state['best']:")
check(normalized==old,"complete byte equality after enumerated version strings and checkpoint predicate")
cfg=json.loads((ROOT/"configs/census_biological_fit_v2.json").read_text())
original=json.loads((ROOT/"configs/census_biological_fit_v1.json").read_text())
norm=json.loads(json.dumps(cfg));norm["kind"]="census_biological_fit_v1";norm["gpu_runtime"]["launcher_module"]="arggnn.census_biological_fit_v1"
check(norm==original,"entire config only kind and launcher module changed")
for p,h in cfg["source_sha256"].items():check(sha(ROOT/p)==h,"preserved dependency "+p)
author=json.loads((ROOT/"research/reviews/census_biological_fit_v2_author/author_receipt.json").read_text())
for p,h in author["baseline_pins"].items():check(sha(ROOT/p)==h,"preserved predecessor "+p)
for p,h in author["artifact_sha256"].items():check(sha(ROOT/p)==h,"sealed author artifact "+p)
def predicate(source):
    run=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=="run_fit_v1")
    found=[n for n in ast.walk(run) if isinstance(n,ast.If) and "nll" in ast.unparse(n.test) and "state['best']" in ast.unparse(n.test)]
    check(len(found)==1,"one exact production checkpoint predicate")
    return compile(ast.fix_missing_locations(ast.Module(body=[found[0]],type_ignores=[])),"actual_selection_block","exec")
codes=[predicate(old),predicate(new)]
class Token:
    def __init__(self,value):self.value=value
    def detach(self):return self
    def cpu(self):return self
class Model:
    def __init__(self,epoch):self.epoch=epoch
    def state_dict(self):return {"weight":Token(self.epoch)}
class Optimizer:
    def __init__(self,epoch):self.epoch=epoch
    def state_dict(self):return {"steps":16*self.epoch}
class Output:
    def __init__(self):self.saved=[]
    def save(self,name,value,kind):
        record={"epoch":value["epoch"],"optimizer_steps":value["optimizer_state"]["steps"],"name":name}
        self.saved.append(record);return record
rng=random.Random(113)
curves=[[float("inf")]*30,[float("inf")]*10+[.8]*20,[1.]*30,[.1]+[float("inf")]*29,
        [1.,.5,.5]+[.8]*27,[float("inf")]*29+[.001],list(reversed([i/30 for i in range(1,31)]))]
curves += [[rng.choice([float("inf"),.1,.4,.8,1.]) for _ in range(30)] for _ in range(12)]
records=[]
for i,curve in enumerate(curves):
    expected=min(range(30),key=lambda j:(curve[j],j))+1
    for control in ("proposed","eta0","permuted_groups","raw_diagonal","local"):
        paths=[]
        for code in codes:
            state={"best":float("inf"),"best_epoch":None};output=Output()
            for epoch,nll in enumerate(curve,1):
                state.update(model=Model(epoch),optimizer=Optimizer(epoch))
                exec(code,{"state":state,"nll":nll,"epoch":epoch,"control":control,"output":output,"admission":{"seed":101},"admission_ref":{"path":"handmade","sha256":"0"*64}})
            paths.append((state,output))
        state,output=paths[1]
        check(state["best_epoch"]==expected,f"reference earliest raw minimum curve{i} control{control}")
        check(state["checkpoint"]["optimizer_steps"]==16*expected,"selected optimizer state is selected epoch")
        if all(x==float("inf") for x in curve):
            check(paths[0][0]["best_epoch"] is None,"original all-infinite defect reproduced")
            check(state["best_epoch"]==1,"successor all-infinite first epoch selected")
        else:check(paths[0][0]["best_epoch"]==state["best_epoch"],"valid finite trajectory selection unchanged")
        records.append({"curve_id":i,"control":control,"selected_epoch":state["best_epoch"],"selected_optimizer_steps":state["checkpoint"]["optimizer_steps"],"synthetic_trajectory_budget_steps":480,"actual_optimization_steps":0})
write("checks.json",{"checks":checks,"count":len(checks),"selection_cases":len(records),"records":records,
    "scientific_arrays_models_optimizers_targets_GPU_live_ledgers_accessed":False})
artifacts={p:sha(ROOT/p) for p in ["src/arggnn/census_biological_fit_v2.py","configs/census_biological_fit_v2.json","docs/census_biological_fit_v2.md"]}
receipt={"schema":"census_biological_fit_v2_independent_review","review_role":"independent_reviewer","verdict":"PASS",
    "status":"qualified_source_protocol_and_exact_selection_delta","actual_admission":False,"fixture_mode":False,
    "created_at_utc":datetime.now(timezone.utc).isoformat(),"artifact_sha256":artifacts,
    "author_receipt":{"path":"research/reviews/census_biological_fit_v2_author/author_receipt.json","sha256":sha(ROOT/"research/reviews/census_biological_fit_v2_author/author_receipt.json")},
    "checks_ref":{"path":str((OUT/"checks.json").relative_to(ROOT)),"sha256":sha(OUT/"checks.json")},
    "check_count":len(checks),"synthetic_selection_cases":len(records),"old_counterexample_reproduced":True,
    "original_science_and_model_unchanged_except_registered_checkpoint_predicate":True,
    "actual_biological_all_infinite_occurrence_observed":False,"actual_model_fitting_or_resource_admission":False,
    "limits":["Source/metadata and scalar production-selection-block fixtures only. No scientific fitting or target data.",
              "Actual original numerical/runtime/profile/storage/launcher/target gates remain required.",
              "Relate development arm only. Same-tsinfer fitting consumer remains separate and unqualified."]}
write("review_receipt.json",receipt)
print(json.dumps({"verdict":"PASS","check_count":len(checks),"selection_cases":len(records),"receipt_sha256":sha(OUT/"review_receipt.json")}))
