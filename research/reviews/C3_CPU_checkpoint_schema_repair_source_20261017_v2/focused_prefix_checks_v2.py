"""Focused synthetic primitive acceptance after namespace/directive repair.

No models, checkpoints, observations, truth, guards or accounting are opened.
The original full validate_request, stage_source, actual_stage and matrix/score
evidence function ASTs run against an explicitly synthetic in-memory VFS.
"""
from pathlib import Path
from types import SimpleNamespace
import ast,copy,hashlib,importlib.util,json,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/C3_CPU_checkpoint_schema_repair_source_20261017_v2"
V1="research/reviews/C3_CPU_checkpoint_schema_repair_source_20261017_v1"
P=ROOT/HERE
checks=[];docs={};read_paths=[]
def require(ok,message):
    if not ok:raise ValueError(message)
def good(name,ok=True):
    require(ok,name);checks.append(dict(name=name,status="PASS"))
def reject(name,fn,expected_message=None):
    try:fn()
    except (ValueError,KeyError,TypeError) as e:
        require(expected_message is None or expected_message in str(e),"Wrong rejection boundary: "+str(e))
        checks.append(dict(name=name,status="PASS_REJECTED",message=str(e)));return
    raise AssertionError(name)
def ref(path):
    p=ROOT/path
    return dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def F(label):
    return dict(path="SYNTHETIC_VFS_ONLY/"+label+".json",sha256=hashlib.sha256(("SYNTHETIC_ONLY:"+label).encode()).hexdigest())
def read(record):
    read_paths.append(record["path"])
    require(record["path"] in docs,"Synthetic VFS has no fallback filesystem reads")
    return copy.deepcopy(docs[record["path"]])
def put(label,value,record=None):
    record=F(label) if record is None else record
    docs[record["path"]]=copy.deepcopy(value);return record
def source_nodes(path):
    return {x.name:x for x in ast.parse((ROOT/path).read_text()).body if isinstance(x,ast.FunctionDef)}
def extract_constant(path,key):
    tree=ast.parse((ROOT/path).read_text())
    node=next(x for x in tree.body if isinstance(x,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id==key for t in x.targets))
    return ast.literal_eval(node.value)
def builder_paths(phase):
    # Execute the exact actual builder assignments that compose namespace paths.
    fn=source_nodes(HERE+"/prepare_evaluation_root_v1.py")["prepare_root_v1"]
    chosen=[]
    for node in fn.body:
        if isinstance(node,ast.Assign):
            ids={x.id for x in ast.walk(node) if isinstance(x,ast.Name) and isinstance(x.ctx,ast.Store)}
            if ids & {"expiry_text","tag","packet_rel","output"}:chosen.append(node)
    ns=dict(args=SimpleNamespace(phase=phase,instance="SYNTHETIC01"),
        expiry=SimpleNamespace(isoformat=lambda:"2099-01-01T00:00:00+00:00"))
    exec(compile(ast.fix_missing_locations(ast.Module(body=chosen,type_ignores=[])),"actual_builder_path_assignments","exec"),ns)
    return ns["packet_rel"],ns["output"]
def main():
    sp=importlib.util.spec_from_file_location("_C3_v2_focused_source_only",P/"acceptance_contract_v1.py")
    c=importlib.util.module_from_spec(sp);sp.loader.exec_module(c)
    manifest=c.manifest_v1()
    fns=source_nodes(HERE+"/acceptance_contract_v1.py");old=source_nodes(V1+"/acceptance_contract_v1.py")
    names=("stage_source_v1","actual_stage_v1","prediction_metadata_v1","matrix_metadata_v1","evidence_request_fields_v1","validate_request")
    for name in names:
        good("Actual contract "+name+" AST unchanged",
            ast.dump(fns[name],include_attributes=False)==ast.dump(old[name],include_attributes=False))
    directive=extract_constant(HERE+"/prepare_evaluation_root_v1.py","DIRECTIVE")
    good("Builder directive matches unchanged contract exactly",directive==c.DIRECTIVE)
    good("Immutable repaired bridge reused",ref(c.BRIDGE_REF["path"])==c.BRIDGE_REF and c.BRIDGE_REF["path"]==V1+"/prediction_bridge_v1.py")
    oldhandoff=json.loads((ROOT/V1/"source_handoff.v1.json").read_text())
    checkref=oldhandoff["source_check_ref"]
    oldchecks=json.loads((ROOT/checkref["path"]).read_text())
    good("Prior98metadata receipt hash retained without reopening checkpoints",ref(checkref["path"])==checkref and oldchecks["check_count"]==98)
    epochrefs={epoch:F("checkpoint_epoch%03d"%epoch)for epoch in c.EPOCHS}
    training_ref=F("completed_training")
    training=dict(checkpoint_candidates=[dict(epoch=e,checkpoint_ref=epochrefs[e])for e in c.EPOCHS])
    accepted=dict(training_completion_ref=training_ref)
    plans={}
    plan_evidence={}
    for parent in c.VAL:
        plans[parent]=dict(focal_registry_ref=F(parent+"_focal_registry"),observation_ref=F(parent+"_observed"))
        plan_evidence[parent]=dict(parent_id=parent,plan_ref=F(parent+"_closed_plan"))
    def training_ready(record):
        require(record==c.SAVED_ACCEPTANCE_REF,"Same synthetic accepted training family")
        return copy.deepcopy(accepted),copy.deepcopy(training)
    def plan_ready(e,parent):
        require(e==plan_evidence[parent],"Same synthetic closed parent plan")
        return copy.deepcopy(plans[parent])
    def terminal(e,completion):
        wait=read(e["actual_wait_ref"]);outer=read(e["outer_terminal_ref"])
        require(wait["returncode"]==0 and wait["owned_drain"] is True
            and outer["actual_exit_code"]==0 and outer["actual_terminal_consumed"] is True,
            "Synthetic genuine-shaped wait0/drain/consumed outer0")
        return wait
    ns=dict(vars(c))
    ns.update(read=read,ref=ref,local=lambda p:ROOT/p,require=require,
        manifest_v1=lambda:manifest,training_ready_v1=training_ready,plan_ready_v1=plan_ready,terminal0_v1=terminal,
        active=lambda value:require(value["active"] is True,"Synthetic active field"),
        integration_review_v1=lambda record:require(record==F("review"),"Synthetic reviewed ref"),
        select_storage_metadata_v1=lambda record:require(record==F("active_child_selector"),"Synthetic fixed child selector"),
        SELECTOR_REF=F("active_child_selector"),SimpleNamespace=SimpleNamespace)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fns[n] for n in names],type_ignores=[])),
        str(P/"acceptance_contract_v1.py"),"exec"),ns)
    access=lambda:SimpleNamespace(**ns)
    # Populate all30 explicitly synthetic prediction seals and their matrix.
    cells=[];prediction_evidence={}
    def make_stage(phase,result_ref,*,old_original=False):
        packet,output=builder_paths(phase)
        request_ref=put(phase+"_producer_request",dict(phase=phase,
            source_manifest_ref=c.LEGACY_MANIFEST_REF if old_original else ref(c.MANIFEST),
            training_evidence_ref=c.SAVED_ACCEPTANCE_REF))
        spec=dict(phase=phase,packet_directory=packet,output_directory=output,
            root_driver_ref=c.LEGACY_DRIVER_REF if old_original else ref(c.DRIVER),
            main_source_ref=c.LEGACY_ENTRY_REF if old_original else ref(c.ENTRY),
            source_sha256=c.legacy.manifest_v1()["source_sha256"] if old_original else manifest["source_sha256"],
            refs=dict(runtime=c.RUNTIME_REF),root_stage_request_ref=request_ref)
        input_ref=put(phase+"_producer_input",spec,record=c.OLD_INPUT_REFS[phase] if old_original else None)
        env=dict(stage=phase,status=c.legacy.STATUSES[phase],training_completion_ref=training_ref,
            input_ref=input_ref,root_driver_ref=spec["root_driver_ref"],result_ref=result_ref)
        completion_ref=put(phase+"_completion",env)
        waitref=put(phase+"_wait",dict(returncode=0,owned_drain=True))
        outerref=put(phase+"_outer",dict(actual_exit_code=0,actual_terminal_consumed=True))
        return dict(completion_ref=completion_ref,actual_wait_ref=waitref,outer_terminal_ref=outerref),input_ref
    stage_inputs={}
    for parent in c.VAL:
        seals=[]
        for epoch in c.EPOCHS:
            pred=F(parent+"_epoch%03d_prediction"%epoch)
            seal=dict(status="sealed_C3_original_heldout_predictions",split="val",parent_id=parent,
                epoch=epoch,records=10000,private_truth_opened=False,
                probabilities_modified=False,training_completion_ref=training_ref,checkpoint_ref=epochrefs[epoch],
                prediction_ref=pred,focal_registry_ref=plans[parent]["focal_registry_ref"])
            sealref=put(parent+"_epoch%03d_seal"%epoch,seal);seals.append(sealref)
            cells.append(dict(epoch=epoch,parent_id=parent,checkpoint_ref=epochrefs[epoch],
                prediction_ref=pred,prediction_seal_ref=sealref,
                focal_registry_ref=plans[parent]["focal_registry_ref"]))
        sixref=put(parent+"_all6",dict(status="sealed_C3_one_original_VAL_all6_candidates",parent_id=parent,
            training_completion_ref=training_ref,private_truth_opened=False,prediction_seal_refs=seals))
        phase="predict_val_"+parent[-3:]
        evidence,inputref=make_stage(phase,sixref,old_original=phase in c.OLD_INPUT_REFS)
        prediction_evidence[parent]=evidence;stage_inputs[phase]=inputref
    matrix=dict(status="sealed_C3_all30_original_VAL_predictions",training_completion_ref=training_ref,
        all_predictions_sealed_before_VAL_truth=True,TEST_deferred=True,checkpoint_epochs=list(c.EPOCHS),
        validation_parent_ids=list(c.VAL),cells=cells)
    matrixref=put("all30_matrix",matrix)
    matrixe,matrixinput=make_stage("matrix",matrixref);stage_inputs["matrix"]=matrixinput
    scoreref=put("score_result",dict(status="SYNTHETIC_ONLY_NOT_A_SCIENTIFIC_SCORE"))
    scoree,scoreinput=make_stage("score",scoreref);stage_inputs["score"]=scoreinput
    # Score evidence metadata mentions only synthetic same-parent simulations.
    simulations={}
    for parent in c.VAL:
        simulation_observed=F(parent+"_simulation_observed")
        put("observed_"+parent,dict(source_observation_ref=simulation_observed),plans[parent]["observation_ref"])
        simulations[parent]=put(parent+"_simulation_completed",
            dict(status="completed_C3_simulation_reconstruction",parent_id=parent,
                class_order=["EUR","EAS","AFR"],observation_ref=simulation_observed))
    def evidence_for(phase):
        ev=dict(schema="C3_full30Mb_eval_actual_stage_evidence_v1",phase=phase,
            ROOT_artifact_acceptance_ref=c.SAVED_ACCEPTANCE_REF,
            closed_plan_evidence_by_parent=plan_evidence if not phase.startswith("predict_val_") else {
                c.VAL[int(phase[-3:])]:plan_evidence[c.VAL[int(phase[-3:])]]})
        if phase=="matrix":ev["VAL_predictions"]=prediction_evidence
        elif phase=="score":ev.update(matrix=matrixe,VAL_simulation_completion_refs=simulations)
        return ev
    requests={}
    for phase in c.PHASES:
        ev=evidence_for(phase);evref=put(phase+"_request_evidence",ev)
        fields,_,_=ns["evidence_request_fields_v1"](access(),ev,phase)
        packet,output=builder_paths(phase)
        caps=c.caps_v1(phase);capsref=put(phase+"_caps",caps)
        expiry="2099-01-01T00:00:00+00:00"
        directive_ref=put(phase+"_ROOT_directive",dict(root_authorizer="root_agent",active=True,
            execution_authorized=True,phase=phase,directive=directive,builder_review_ref=F("review"),
            actual_stage_evidence_ref=evref,expires_at_utc=expiry,
            actual_ROOT_artifact_acceptance_ref=c.SAVED_ACCEPTANCE_REF,main_review_ref=c.LEGACY_MAIN_REVIEW_REF,
            original_history=c.ORIGINAL_HISTORY,all5VALx6checkpoint_barrier_preserved=True,
            storage_policy_selector_ref=F("active_child_selector")))
        req=dict(schema="C3_full30Mb_evaluation_ROOT_request_v1",active=True,root_authorizer="root_agent",
            execution_authorized=True,created_at_utc="SYNTHETIC_ONLY",expires_at_utc=expiry,phase=phase,
            device="cpu",GPU_authorized=False,physical_gpu=None,runtime_ref=c.RUNTIME_REF,
            source_manifest_ref=ref(c.MANIFEST),packet_directory=packet,output_directory=output,caps_ref=capsref,
            integration_review_ref=F("review"),ROOT_directive_ref=directive_ref,
            actual_stage_evidence_ref=evref,storage_policy_selector_ref=F("active_child_selector"),**fields)
        rr=put(phase+"_full_request",req);requests[phase]=(rr,req)
        # runtime metadata is synthetic and cannot confer real execution authority.
        put("synthetic_runtime",dict(SYNTHETIC_ONLY=True),c.RUNTIME_REF)
        returned=ns["validate_request"](rr)
        good("Complete unchanged validate_request accepts actual builder paths/directive "+phase,returned[0]==req)
        stage_evidence=(prediction_evidence[c.VAL[int(phase[-3:])]] if phase.startswith("predict_val_")
            else matrixe if phase=="matrix" else scoree)
        st=ns["stage_source_v1"](read(stage_evidence["completion_ref"]),phase)
        good("Unchanged producer stage_source accepts fresh namespace "+phase,st==read(stage_inputs[phase]))
        for field in ("packet_directory","output_directory"):
            altered=copy.deepcopy(req);altered[field]=altered[field].replace("actual_plan_lineage_checkpoint_schema_","checkpoint_schema_")
            badref=put(phase+"_bad_"+field,altered)
            reject("Rejected v1 incompatible "+field+" "+phase,lambda:ns["validate_request"](badref),
                "Fresh separated future output and administrative scopes")
        altered=read(directive_ref);altered["directive"]="ROOT authorizes one fresh nested-checkpoint-schema C3 CPU stage with unchanged observed data numerical science and all30 pretruth barrier"
        docs[directive_ref["path"]]=altered
        reject("Rejected v1 mismatched directive "+phase,lambda:ns["validate_request"](rr),
            "Exact separate future ROOT stage directive and reviewed evidence")
        docs[directive_ref["path"]]["directive"]=directive
    # Exercise the actual downstream matrix/score caller paths, not startswith alone.
    rr,_=requests["matrix"]
    good("Matrix full request accepts all5 producer routes including repaired003004",ns["validate_request"](rr)[0]["phase"]=="matrix")
    rr,_=requests["score"]
    good("Score full request accepts all30 matrix producer route without truth",ns["validate_request"](rr)[0]["phase"]=="score")
    spec=read(stage_inputs["predict_val_003"]);changed=copy.deepcopy(spec)
    changed["packet_directory"]=spec["packet_directory"].replace("actual_plan_lineage_checkpoint_schema_","checkpoint_schema_")
    docs[stage_inputs["predict_val_003"]["path"]]=changed
    reject("Matrix catches old v1 prediction producer prefix",lambda:ns["validate_request"](requests["matrix"][0]),
        "Exact new guarded future producer")
    docs[stage_inputs["predict_val_003"]["path"]]=spec
    spec=read(matrixinput);changed=copy.deepcopy(spec)
    changed["packet_directory"]=spec["packet_directory"].replace("actual_plan_lineage_checkpoint_schema_","checkpoint_schema_")
    docs[matrixinput["path"]]=changed
    reject("Score catches old v1 matrix producer prefix",lambda:ns["validate_request"](requests["score"][0]),
        "Exact new guarded future producer")
    docs[matrixinput["path"]]=spec
    evref=requests["matrix"][1]["actual_stage_evidence_ref"];ev=read(evref)
    changed=copy.deepcopy(ev);changed["closed_plan_evidence_by_parent"].pop(c.VAL[-1]);docs[evref["path"]]=changed
    reject("All5 closed-plan matrix barrier remains",lambda:ns["validate_request"](requests["matrix"][0]),
        "Prediction requires only its own actual closed plan")
    docs[evref["path"]]=ev
    m=read(matrixref);changed=copy.deepcopy(m);changed["cells"].pop();docs[matrixref["path"]]=changed
    reject("All30 pretruth score barrier remains",lambda:ns["validate_request"](requests["score"][0]),
        "Actual complete original5VALx6checkpoint pretruth matrix")
    docs[matrixref["path"]]=m
    good("No model scientific data checkpoint or guard import/read",
        not any(k in sys.modules for k in ("torch","torch_geometric","numpy","tskit"))
        and not any(p.endswith((".pt",".npz",".trees",".npy"))for p in read_paths))
    return dict(schema="C3_namespace_directive_focused_primitive_checks_v2",status="PASS_SOURCE_ONLY",
        check_count=len(checks),checks=checks,all_execution_evidence_and_runtime_policy_fixtures_synthetic=True,
        actual_validate_request_stage_source_actual_stage_evidence_matrix_score_ASTs_exercised=True,
        original98real_checkpoint_checks_not_repeated=True,read_calls=len(read_paths),
        actual_guard_model_data_truth_prediction_training_or_accounting=False,
        builder_stem="actual_plan_lineage_checkpoint_schema_",builder_contract_directive=directive)
if __name__=="__main__":
    print(json.dumps(main(),indent=2,sort_keys=True))
