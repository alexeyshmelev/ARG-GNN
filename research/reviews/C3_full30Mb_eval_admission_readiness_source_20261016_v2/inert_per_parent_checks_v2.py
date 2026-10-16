"""Source-only tests; every prerequisite and stage document is an explicit in-memory fixture."""
from pathlib import Path
from datetime import datetime,timezone
import ast,copy,hashlib,json,subprocess,sys,traceback,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/C3_full30Mb_eval_admission_readiness_source_20261016_v2"
OLD="research/reviews/C3_full30Mb_eval_admission_readiness_source_20261016_v1"
SOURCE=HERE+"/prepare_evaluation_root_v2.py"
checks=[]
def require(ok,message):
    if not ok:raise ValueError(message)
def good(name,value=True):
    assert value,name
    checks.append(dict(name=name,status="PASS"))
def rejects(name,fn):
    try:fn()
    except (ValueError,KeyError):checks.append(dict(name=name,status="PASS_REJECTED"));return
    raise AssertionError("Invalid fixture accepted: "+name)
def fixture(label):
    return dict(path="IN_MEMORY_SOURCE_FIXTURE/"+label+".json",sha256=hashlib.sha256(("fixture:"+label).encode()).hexdigest())
def ref(path):return dict(path=path,sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest())
def run():
    source=(ROOT/SOURCE).read_text();tree=ast.parse(source)
    prior=(ROOT/OLD/"prepare_evaluation_root_v1.py").read_text();oldtree=ast.parse(prior)
    oldfunc={n.name:ast.dump(n,include_attributes=False) for n in oldtree.body if isinstance(n,ast.FunctionDef)}
    newfunc={n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)}
    good("AST_parse_and_only_one_evidence_gate_body_changed",[name for name in newfunc if newfunc[name]!=oldfunc[name]]==["evidence_request_fields_v1"] and set(newfunc)==set(oldfunc))
    good("all_predict_matrix_score_metadata_and_admission_bodies_unchanged",all(oldfunc[n]==newfunc[n] for n in ("actual_stage_v1","prediction_metadata_v1","matrix_metadata_v1","prepare_root_v1","builder_review_v1","main")))
    oldgate=next(n for n in oldtree.body if isinstance(n,ast.FunctionDef) and n.name=="evidence_request_fields_v1")
    newgate=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="evidence_request_fields_v1")
    oldgate=copy.deepcopy(oldgate);newgate=copy.deepcopy(newgate)
    # Exclude only required-parent construction/check/lookup from comparison.
    oldgate.body=[n for n in oldgate.body if not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="plans" for t in n.targets)) and not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and any(isinstance(a,ast.Constant) and a.value=="All5 actual closed full30Mb plans required before this evaluation admission" for a in n.value.args))]
    newgate.body=[n for n in newgate.body if not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ("plans","required_parents") for t in n.targets)) and not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and any(isinstance(a,ast.Constant) and a.value=="Prediction requires only its own actual closed plan;matrix/score require all5 actual closed full30Mb plans" for a in n.value.args))]
    good("every_other_evidence_gate_statement_AST_unchanged",ast.dump(oldgate,include_attributes=False)==ast.dump(newgate,include_attributes=False))
    for args,rc in [([],0),(["--help"],0),(["--prepare-root"],1)]:
        p=subprocess.run([sys.executable,"-B","-s",str(ROOT/SOURCE),*args],cwd=ROOT,capture_output=True,text=True,timeout=10)
        good("inert_CLI_"+("_".join(args) if args else "default"),p.returncode==rc)
    good("no_scientific_imports",not any(n in sys.modules for n in ("torch","torch_geometric","numpy","tskit")))
    imports=[n for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]
    good("builder_imports_standard_library_only",all((all(a.name.split(".")[0] in {"argparse","copy","hashlib","importlib","json","re","sys"} for a in n.names) if isinstance(n,ast.Import) else n.module in ("pathlib","datetime")) for n in imports))
    forbidden={"Popen","run","system","execv","execve","spawn","posix_spawn","launch"}
    good("builder_has_no_execution_call",not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in forbidden for n in ast.walk(tree)))
    # Only these pure functions are extracted; active preparation/load/bound paths are never called.
    namespace=dict(copy=copy,require=require)
    names={"evidence_request_fields_v1","actual_stage_v1","prediction_metadata_v1","matrix_metadata_v1"}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),"<source_only_evidence_fixtures>","exec"),namespace)
    VAL=tuple("american_c3_val_%03d"%i for i in range(5));EPOCHS=(5,10,15,20,25,30)
    phases=tuple("predict_val_%03d"%i for i in range(5))+("matrix","score")
    training_ref=fixture("saved600_training");acceptance_ref=fixture("actual_acceptance_placeholder_only_in_test")
    candidates={epoch:fixture("checkpoint_"+str(epoch)) for epoch in EPOCHS}
    training=dict(checkpoint_candidates=[dict(epoch=e,checkpoint_ref=r) for e,r in candidates.items()])
    accepted=dict(training_completion_ref=training_ref,source_review_ref=fixture("main_review"))
    baseline_docs={};plan_evidence={};plans={};prediction_evidence={};simulation_refs={};plan_calls=[];read_calls=[];accept_calls=[]
    def put(label,value):
        r=fixture(label);baseline_docs[r["path"]]=value;return r
    driver_ref=fixture("unchanged_driver")
    statuses={p:"completed_C3_full30Mb_CPU_evaluation_"+p for p in phases}
    for parent in VAL:
        simobs=fixture(parent+"_simulation_observed")
        obs=put(parent+"_observed",dict(source_observation_ref=simobs))
        sim=put(parent+"_simulation_completed",dict(status="completed_C3_simulation_reconstruction",parent_id=parent,class_order=["EUR","EAS","AFR"],observation_ref=simobs))
        simulation_refs[parent]=sim
        focal=fixture(parent+"_focals")
        plans[parent]=dict(focal_registry_ref=focal,observation_ref=obs)
        plan_evidence[parent]=dict(parent_id=parent,actually_closed=True,genuine_wait_outer=True)
        sealrefs=[]
        for epoch in EPOCHS:
            prediction=fixture(parent+"_epoch_"+str(epoch)+"_prediction")
            sealrefs.append(put(parent+"_epoch_"+str(epoch)+"_seal",dict(status="sealed_C3_original_heldout_predictions",parent_id=parent,epoch=epoch,split="val",records=10000,private_truth_opened=False,training_completion_ref=training_ref,checkpoint_ref=candidates[epoch],probabilities_modified=False,prediction_ref=prediction,focal_registry_ref=focal)))
        six=put(parent+"_all6",dict(status="sealed_C3_one_original_VAL_all6_candidates",parent_id=parent,training_completion_ref=training_ref,private_truth_opened=False,prediction_seal_refs=sealrefs))
        phase="predict_val_"+parent[-3:]
        completion=put(phase+"_completed",dict(stage=phase,status=statuses[phase],root_driver_ref=driver_ref,training_completion_ref=training_ref,result_ref=six))
        wait=put(phase+"_wait",dict(returncode=0,all_owned_processes_and_groups_drained=True,completed_ref=completion))
        outer=put(phase+"_outer",dict(actual_exit_code=0,actual_terminal_consumed=True,completed_ref=completion,direct_wait_ref=wait))
        prediction_evidence[parent]=dict(completion_ref=completion,actual_wait_ref=wait,outer_terminal_ref=outer)
    cells=[]
    for parent in VAL:
        six=baseline_docs[baseline_docs[prediction_evidence[parent]["completion_ref"]["path"]]["result_ref"]["path"]]
        for sr in six["prediction_seal_refs"]:
            seal=baseline_docs[sr["path"]]
            cells.append(dict(epoch=seal["epoch"],parent_id=parent,checkpoint_ref=seal["checkpoint_ref"],prediction_seal_ref=sr,prediction_ref=seal["prediction_ref"],focal_registry_ref=seal["focal_registry_ref"]))
    matrix_result=put("matrix_result",dict(status="sealed_C3_all30_original_VAL_predictions",training_completion_ref=training_ref,all_predictions_sealed_before_VAL_truth=True,TEST_deferred=True,checkpoint_epochs=list(EPOCHS),validation_parent_ids=list(VAL),cells=cells))
    matrix_completion=put("matrix_completed",dict(stage="matrix",status=statuses["matrix"],root_driver_ref=driver_ref,training_completion_ref=training_ref,result_ref=matrix_result))
    matrix_wait=put("matrix_wait",dict(returncode=0,all_owned_processes_and_groups_drained=True,completed_ref=matrix_completion))
    matrix_outer=put("matrix_outer",dict(actual_exit_code=0,actual_terminal_consumed=True,completed_ref=matrix_completion,direct_wait_ref=matrix_wait))
    matrix_evidence=dict(completion_ref=matrix_completion,actual_wait_ref=matrix_wait,outer_terminal_ref=matrix_outer)
    docs=copy.deepcopy(baseline_docs)
    def read(record):
        read_calls.append(record["path"])
        assert record["path"].startswith("IN_MEMORY_SOURCE_FIXTURE/")
        return copy.deepcopy(docs[record["path"]])
    def training_ready(record):
        accept_calls.append(record)
        require(record==acceptance_ref,"Real saved-model acceptance required")
        return copy.deepcopy(accepted),copy.deepcopy(training)
    def plan_ready(e,parent):
        plan_calls.append(parent)
        require(e==plan_evidence[parent] and e["actually_closed"] is True and e["genuine_wait_outer"] is True,"Actual same-parent closed plan required")
        return copy.deepcopy(plans[parent])
    def terminal0(e,completion):
        wait=read(e["actual_wait_ref"]);outer=read(e["outer_terminal_ref"])
        require(type(wait["returncode"])is int and wait["returncode"]==0 and wait["all_owned_processes_and_groups_drained"] is True and wait["completed_ref"]==completion and outer["actual_terminal_consumed"] is True and outer["completed_ref"]==completion and outer["direct_wait_ref"]==e["actual_wait_ref"],"Genuine joined wait/drain/outer")
        return wait
    c=types.SimpleNamespace(VAL=VAL,EPOCHS=EPOCHS,STATUSES=statuses,DRIVER="UNCHANGED_DRIVER",ref=lambda p:driver_ref,read=read,training_ready_v1=training_ready,plan_ready_v1=plan_ready,terminal0_v1=terminal0)
    gate=namespace["evidence_request_fields_v1"]
    def evidence(phase):
        needed={VAL[int(phase[-3:])]} if phase.startswith("predict_val_") else set(VAL)
        e=dict(schema="C3_full30Mb_eval_actual_stage_evidence_v1",phase=phase,ROOT_artifact_acceptance_ref=acceptance_ref,closed_plan_evidence_by_parent={p:copy.deepcopy(plan_evidence[p]) for p in VAL if p in needed})
        if phase=="matrix":e["VAL_predictions"]=copy.deepcopy(prediction_evidence)
        if phase=="score":e.update(matrix=copy.deepcopy(matrix_evidence),VAL_simulation_completion_refs=copy.deepcopy(simulation_refs))
        return e
    for phase in phases[:5]:
        parent=VAL[int(phase[-3:])]
        plan_calls.clear();read_calls.clear()
        fields,b,t=gate(c,evidence(phase),phase)
        good(phase+"_own_plan_only_accepts_other_plans_absent",plan_calls==[parent] and fields==dict(training_evidence_ref=acceptance_ref,plan_evidence=plan_evidence[parent]) and not read_calls)
        for name,change in [
            ("no_plan",lambda e:e.update(closed_plan_evidence_by_parent={})),
            ("wrong_parent",lambda e:e.update(closed_plan_evidence_by_parent={VAL[(int(phase[-3:])+1)%5]:copy.deepcopy(plan_evidence[VAL[(int(phase[-3:])+1)%5]])})),
            ("unrelated_plans_extra",lambda e:e.update(closed_plan_evidence_by_parent=copy.deepcopy(plan_evidence))),
            ("truth_capability_extra",lambda e:e.update(VAL_truth_capability=simulation_refs)),
            ("predictions_extra",lambda e:e.update(VAL_predictions=prediction_evidence)),
            ("missing_acceptance",lambda e:e.update(ROOT_artifact_acceptance_ref=fixture("wrong_acceptance"))),
            ("own_plan_not_closed",lambda e:e["closed_plan_evidence_by_parent"][parent].update(actually_closed=False)),
            ("own_plan_no_genuine_terminal",lambda e:e["closed_plan_evidence_by_parent"][parent].update(genuine_wait_outer=False))]:
            e=evidence(phase);change(e)
            rejects(phase+"_"+name,lambda:gate(c,e,phase))
    for phase in ("matrix","score"):
        plan_calls.clear()
        fields,b,t=gate(c,evidence(phase),phase)
        good(phase+"_all5_real_plans_required",plan_calls==list(VAL))
        if phase=="matrix":good("matrix_has_no_truth_capability",set(fields)=={"training_evidence_ref","VAL_predictions"})
        else:good("score_exact5_existing_simulation_refs_copied",fields["VAL_truth_capability"]==simulation_refs and fields["plan_evidence_by_parent"]==plan_evidence)
        for parent in VAL:
            e=evidence(phase);del e["closed_plan_evidence_by_parent"][parent]
            rejects(phase+"_missing_plan_"+parent,lambda:gate(c,e,phase))
        e=evidence(phase);e["ROOT_artifact_acceptance_ref"]=fixture("not_accepted")
        rejects(phase+"_genuine_model_acceptance_required",lambda:gate(c,e,phase))
        e=evidence(phase);e["TEST_access"]=True
        rejects(phase+"_TEST_capability_extra_rejected",lambda:gate(c,e,phase))
    for parent in VAL:
        e=evidence("matrix");del e["VAL_predictions"][parent]
        rejects("matrix_missing_parent_"+parent,lambda:gate(c,e,"matrix"))
        cp=prediction_evidence[parent]["completion_ref"]
        result=baseline_docs[cp["path"]]["result_ref"];six=baseline_docs[result["path"]]
        docs=copy.deepcopy(baseline_docs);docs[result["path"]]["prediction_seal_refs"]=six["prediction_seal_refs"][:-1]
        rejects("matrix_all6_required_"+parent,lambda:gate(c,evidence("matrix"),"matrix"))
        docs=copy.deepcopy(baseline_docs);docs[six["prediction_seal_refs"][0]["path"]]["private_truth_opened"]=True
        rejects("matrix_pretruth_seal_required_"+parent,lambda:gate(c,evidence("matrix"),"matrix"))
        docs=copy.deepcopy(baseline_docs);docs[prediction_evidence[parent]["outer_terminal_ref"]["path"]]["actual_terminal_consumed"]=False
        rejects("matrix_consumed_outer_required_"+parent,lambda:gate(c,evidence("matrix"),"matrix"))
    for field,bad in [("cells",cells[:-1]),("cells",[cells[0],*cells[1:-1],cells[0]]),("all_predictions_sealed_before_VAL_truth",False),("TEST_deferred",False),("checkpoint_epochs",[30]),("validation_parent_ids",list(VAL[:-1]))]:
        docs=copy.deepcopy(baseline_docs);docs[matrix_result["path"]][field]=bad
        rejects("score_full30_barrier_"+field+"_"+str(len(checks)),lambda:gate(c,evidence("score"),"score"))
    for parent in VAL:
        first=next(r for r in cells if r["parent_id"]==parent)
        docs=copy.deepcopy(baseline_docs);docs[first["prediction_seal_ref"]["path"]]["focal_registry_ref"]=fixture("wrong_focal")
        rejects("score_closed_plan_focal_join_"+parent,lambda:gate(c,evidence("score"),"score"))
        docs=copy.deepcopy(baseline_docs);docs[simulation_refs[parent]["path"]]["parent_id"]="wrong_parent"
        rejects("score_same_parent_truth_capability_join_"+parent,lambda:gate(c,evidence("score"),"score"))
    docs=copy.deepcopy(baseline_docs);docs[matrix_wait["path"]]["returncode"]=1
    rejects("score_actual_matrix_wait0_required",lambda:gate(c,evidence("score"),"score"))
    docs=copy.deepcopy(baseline_docs);docs[matrix_outer["path"]]["actual_exit_code"]=1
    rejects("score_consumed_matrix_outer0_required",lambda:gate(c,evidence("score"),"score"))
    good("no_real_project_import_or_scientific_payload",not any(n in sys.modules for n in ("torch","torch_geometric","numpy","tskit")))
    return dict(schema="C3_eval_preparation_per_parent_v2_source_checks_v1",status="PASS_SOURCE_ONLY",source_ref=ref(SOURCE),checks=checks,check_count=len(checks),predecessor_source_ref=ref(OLD+"/prepare_evaluation_root_v1.py"),own_closed_plan_before_each_prediction=True,all5_closed_plans_before_matrix_and_score=True,all5x6_pretruth_barrier_unchanged=True,main_scientific_sources_modified=False,actual_acceptance_or_guard_resource_probe_accounting_authority_admission_execution=False,only_in_memory_explicit_fixtures=True,finished_utc=datetime.now(timezone.utc).isoformat())
def save(name,value):
    with (ROOT/HERE/name).open("x") as f:json.dump(value,f,sort_keys=True,indent=2);f.write("\n")
if __name__=="__main__":
    try:
        result=run();save("source_static_inert_checks.v2.json",result)
        print(json.dumps(dict(status=result["status"],checks=result["check_count"]),indent=2))
    except BaseException as error:
        save("source_checks.attempt1.FAILURE.json",dict(error=repr(error),traceback=traceback.format_exc(),completed_checks=checks));raise
