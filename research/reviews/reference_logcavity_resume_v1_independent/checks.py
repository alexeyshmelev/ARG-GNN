"""Narrow source/metadata and synthetic restoration tests; standard library only."""
import ast
from copy import deepcopy
from datetime import datetime,timezone
import hashlib,io,json
from pathlib import Path
from types import SimpleNamespace
root=Path.cwd();out=root/"research/reviews/reference_logcavity_resume_v1_independent"
sha=lambda p:hashlib.sha256((root/p).read_bytes()).hexdigest()
src="src/arggnn/train_reference_logcavity_resume_v1.py";cfg_path="configs/reference_logcavity_resume_v1.json"
pins={src:"4f7c9b55eb58282eb974ea86580bd37b79492b0e0f9f60dc035b34f6768371ec",cfg_path:"8b5d9e95f2a954f636a62f659e58a23035450ae2fd798b1aee869abb8f97b573","docs/reference_logcavity_resume_v1.md":"4a94d788364b3fdd253769d5b63c780ca92965ec6a2385f87914d382892ffd88","src/arggnn/reference_logcavity_gnn_v1.py":"8970aa2e82a43b45d1ec10b2eaae6e2203dab12356a4b968bea3c65b2e21fc0b","src/arggnn/project_budget_v9_consumer_retry_v2.py":"75fd586fc8e0733027451a6d9485b7e49f10c3d8d7cf69c09cdec4b9f1b7ba1e"}
assert all(sha(p)==v for p,v in pins.items())
cfg=json.loads((root/cfg_path).read_text())
old_cfg=json.loads((root/cfg["scientific_config_ref"]["path"]).read_text())
allowed={"kind","helper_refs","budget_guard","resource_boundaries","execution_guard_ready"}
assert all(cfg[k]==v for k,v in old_cfg.items() if k not in allowed)
for name,ref in old_cfg["helper_refs"].items():
    if name!="budget_V9_consumer":assert cfg["helper_refs"][name]==ref
assert all(sha(ref["path"])==ref["sha256"] for ref in cfg["helper_refs"].values())
old=ast.parse((root/"src/arggnn/train_reference_logcavity_gnn_v1.py").read_text())
new=ast.parse((root/src).read_text())
old_funcs={n.name:n for n in old.body if isinstance(n,ast.FunctionDef)}
new_funcs={n.name:n for n in new.body if isinstance(n,ast.FunctionDef)}
dump=lambda n:ast.dump(n,include_attributes=False)
for name in ("seal_v1","strong_reference_v1","lightweight_resources_v1"):
    assert dump(old_funcs[name])==dump(new_funcs[name])
prepared=deepcopy(new_funcs["prepare_v1"])
for n in ast.walk(prepared):
    if isinstance(n,ast.ImportFrom) and n.module=="project_budget_v9_consumer_retry_v2":n.module="project_budget_v9_consumer"
idx=next(i for i,n in enumerate(prepared.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=="geometry")
old_save=next(n for n in old_funcs["prepare_v1"].body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=="output.save")
assert dump(prepared.body[idx].value)==dump(old_save.value.args[1])
assert isinstance(prepared.body[idx+2],ast.If) and "geometry" in ast.unparse(prepared.body[idx+2].test)
prepared.body[idx:idx+4]=[deepcopy(old_save)]
assert dump(prepared)==dump(old_funcs["prepare_v1"])
def optimizer_loop(fn):
    return next(n for n in ast.walk(fn) if isinstance(n,ast.For) and ast.unparse(n.iter)=="states.values()")
assert dump(optimizer_loop(new_funcs["run_v1"]))==dump(optimizer_loop(old_funcs["run_v1"]))
def selection_if(fn):
    return next(n for n in ast.walk(fn) if isinstance(n,ast.If) and "best_epoch" in ast.unparse(n.test) and "nll <" in ast.unparse(n.test))
assert dump(selection_if(new_funcs["run_v1"]))==dump(selection_if(old_funcs["run_v1"]))
# Read only the manifest and JSON metrics metadata, never .pt/.npz payloads.
manifest_ref=cfg["resume_state_manifest_ref"];assert sha(manifest_ref["path"])==manifest_ref["sha256"]
manifest=json.loads((root/manifest_ref["path"]).read_text())
arms=("logcavity_only","static","carrier_gru","carrier_matrix","sham_ancestor")
expected_epochs=dict(zip(arms,(4,4,3,3,3)))
assert set(manifest["arms"])==set(arms)
for arm,anchor in manifest["arms"].items():
    assert anchor["last_complete_epoch"]==expected_epochs[arm] and anchor["restored_optimizer_steps"]==16*expected_epochs[arm]
    history=[]
    for ref in anchor["history_metrics_refs"]:
        assert ref["path"].endswith(".json") and sha(ref["path"])==ref["sha256"]
        history.append(json.loads((root/ref["path"]).read_text()))
    assert [r["epoch"] for r in history]==list(range(1,expected_epochs[arm]+1))
    assert all(r["arm"]==arm and r["optimizer_steps"]==16*r["epoch"] for r in history)
    nll=lambda r:float("inf") if r["equal_parent_metrics"]["primary_present_day_64"]["NLL_is_infinite"] else float(r["equal_parent_metrics"]["primary_present_day_64"]["nll"])
    selected=min(range(len(history)),key=lambda i:(nll(history[i]),i))
    assert anchor["prior_selected_epoch"]==history[selected]["epoch"]
    assert history[-1]["checkpoint"]==anchor["latest_checkpoint_ref"]
    assert history[-1]["selected_epoch"]==history[selected]["epoch"]
    assert anchor["prior_selected_metrics_ref"]==anchor["history_metrics_refs"][selected]
# Synthetic model/optimizer objects exercise the real restoration function.
def ref(p):return {"path":"mock/"+p,"sha256":"mock"}
class FakeModel:
    def load_state_dict(self,state,strict=False):assert strict;self.loaded=state
class FakeOptimizer:
    def load_state_dict(self,state):self.loaded=state
class ScalarStep:
    def __init__(self,value):self.value=value
    def item(self):return self.value
def fixture(infinite=False):
    objects={};payloads={};m={"arms":{}};states={}
    fake_cfg={"resume_state_manifest_ref":ref("manifest"),"original_input_ref":ref("original-input"),"original_frozen_baseline_checkpoint":ref("teacher")}
    for arm in arms:
        epoch=expected_epochs[arm];history=[]
        for e in range(1,epoch+1):
            probs=[ref(f"{arm}-{e}-val-{v}") for v in range(2)]
            nlls=(.8,.6,.6,.7)
            selected_epoch=1 if infinite or e==1 else 2
            row={"epoch":e,"arm":arm,"optimizer_steps":16*e,"checkpoint":ref(f"{arm}-{e}-checkpoint"),"cells":[{"prediction_seal":p} for p in probs],"equal_parent_metrics":{"primary_present_day_64":{"NLL_is_infinite":infinite,"nll":None if infinite else nlls[e-1]}},"selected_epoch":selected_epoch}
            history.append(row);objects[ref(f"{arm}-{e}-metrics")["path"]]=row
        selected=history[0 if infinite else 1]
        anchor={"last_complete_epoch":epoch,"history_metrics_refs":[ref(f"{arm}-{e}-metrics") for e in range(1,epoch+1)],"latest_metrics_ref":ref(f"{arm}-{epoch}-metrics"),"latest_checkpoint_ref":history[-1]["checkpoint"],"latest_probability_refs":[c["prediction_seal"] for c in history[-1]["cells"]],"prior_selected_epoch":selected["epoch"],"prior_selected_metrics_ref":ref(f"{arm}-{selected['epoch']}-metrics"),"prior_selected_probability_refs":[c["prediction_seal"] for c in selected["cells"]],"restored_optimizer_steps":16*epoch}
        m["arms"][arm]=anchor
        payloads[anchor["latest_checkpoint_ref"]["path"]]={"epoch":epoch,"arm":arm,"seed":101,"input_ref":fake_cfg["original_input_ref"],"frozen_baseline_checkpoint":fake_cfg["original_frozen_baseline_checkpoint"],"model_state":f"model-{arm}-{epoch}","optimizer_state":{"state":{0:{"step":ScalarStep(16*epoch)}},"param_groups":[{"params":[0]}]}}
        states[arm]={"model":FakeModel(),"optimizer":FakeOptimizer()}
    objects[fake_cfg["resume_state_manifest_ref"]["path"]]=m
    class Reader:
        def __init__(self):self.reads=[]
        def decode_json(self,r):return objects[r["path"]]
        def read(self,r):self.reads.append(r);return r["path"].encode()
    reader=Reader()
    fake_torch=SimpleNamespace(load=lambda b,**kw:payloads[b.getvalue().decode()])
    env={"fit":SimpleNamespace(torch=fake_torch),"io":io}
    exec(compile(ast.Module(body=[new_funcs["restore_states_v1"]],type_ignores=[]),src,"exec"),env)
    return states,fake_cfg,reader,objects,payloads,env["restore_states_v1"]
states,fc,reader,objects,payloads,restore=fixture()
restore(states,fc,reader)
for arm,state in states.items():
    e=expected_epochs[arm]
    assert state["model"].loaded==f"model-{arm}-{e}" and state["optimizer"].loaded["state"][0]["step"].item()==16*e
    assert state["steps"]==16*e and state["new_steps"]==0 and state["best_epoch"]==2
    assert all(p["path"].startswith("mock/") for p in state["retained_best"])
# Strict future tie preserves historical selection; strict improvement replaces it.
selection=selection_if(new_funcs["run_v1"])
state=states[arms[0]];state["current"]=[ref("new-product")]
selected_before=state["selected"]
exec(compile(ast.Module(body=[selection],type_ignores=[]),src,"exec"),{"state":state,"nll":.6,"epoch":5,"row":{"epoch":5}})
assert state["selected"] is selected_before and state["best_epoch"]==2
exec(compile(ast.Module(body=[selection],type_ignores=[]),src,"exec"),{"state":state,"nll":.59,"epoch":5,"row":{"epoch":5}})
assert state["best_epoch"]==5
states,fc,reader,objects,payloads,restore=fixture(infinite=True)
restore(states,fc,reader);assert all(s["best_epoch"]==1 for s in states.values())
for kind in ("bad_optimizer_step","bad_arm","bad_seed","bad_teacher","bad_latest_pair","history_gap"):
    states,fc,reader,objects,payloads,restore=fixture()
    arm=arms[0];a=objects[fc["resume_state_manifest_ref"]["path"]]["arms"][arm];saved=payloads[a["latest_checkpoint_ref"]["path"]]
    if kind=="bad_optimizer_step":saved["optimizer_state"]["state"][0]["step"]=ScalarStep(63)
    if kind=="bad_arm":saved["arm"]="wrong"
    if kind=="bad_seed":saved["seed"]=102
    if kind=="bad_teacher":saved["frozen_baseline_checkpoint"]=ref("wrong-teacher")
    if kind=="bad_latest_pair":a["latest_metrics_ref"]=ref("wrong-metrics")
    if kind=="history_gap":a["history_metrics_refs"].pop(1)
    error=None
    try:restore(states,fc,reader)
    except ValueError as caught:error=caught
    assert error is not None,kind
# Execute only the real schedule expressions; no model or optimizer call.
states={a:{"completed_epoch":e,"steps":16*e,"new_steps":0} for a,e in expected_epochs.items()}
epoch_loop=next(n for n in ast.walk(new_funcs["run_v1"]) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=="epoch")
epochs=eval(compile(ast.Expression(epoch_loop.iter),src,"eval"),{"states":states})
active_node=epoch_loop.body[0];schedule=[]
for epoch in epochs:
    env={"states":states,"epoch":epoch}
    exec(compile(ast.Module(body=[active_node],type_ignores=[]),src,"exec"),env)
    active=env["active_arms"];schedule.append({"epoch":epoch,"arms":list(active)})
    for arm in active:
        states[arm]["steps"]+=16;states[arm]["new_steps"]+=16;states[arm]["completed_epoch"]=epoch
assert schedule[0]["arms"]==list(arms[2:])
assert all(s["steps"]==160 for s in states.values())
new_steps={a:s["new_steps"] for a,s in states.items()};assert sum(new_steps.values())==528
report={"frozen_source_sha256":pins,"unchanged_geometry_loss_scoring_helper_AST":True,"original_AdamW_parameter_group_AST":True,"historical_and_new_earliest_minimum_NLL_selection":True,"actual_JSON_history_metadata_verified":True,"synthetic_restore_model_optimizer_and_error_checks":True,"per_arm_new_steps":new_steps,"total_new_updates":528,"final_logical_updates":800,"discarded_replayed_updates":48,"historical_plus_recovery_updates":848,"schedule":schedule,"actual_checkpoint_or_probability_or_target_payloads_read":False,"actual_model_or_optimizer_execution":False,"GPU_used":False}
(out/"checks.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
