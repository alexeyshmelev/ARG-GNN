"""New integration questions only. All active inputs/guards/evidence are in memory."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from types import SimpleNamespace
import ast,copy,hashlib,importlib.util,json,sys,traceback
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/C3_CPU_actual_plan_lineage_integration_source_20261016_v1"
checks=[]
def require(ok,msg):
    if not ok:raise ValueError(msg)
def good(name,ok=True):
    require(ok,name);checks.append(dict(name=name,status="PASS"))
def rejects(name,fn):
    try:fn()
    except (ValueError,KeyError):checks.append(dict(name=name,status="PASS_REJECTED"));return
    raise AssertionError("Accepted invalid fixture: "+name)
def F(label):
    return dict(path="IN_MEMORY_INTEGRATION_FIXTURE/"+label+".json",sha256=hashlib.sha256(label.encode()).hexdigest())
def segment(path,names):
    t=(ROOT/path).read_text();nodes={x.name:x for x in ast.parse(t).body if isinstance(x,(ast.FunctionDef,ast.ClassDef))}
    return t,nodes
def run():
    cp=HERE+"/acceptance_contract_v1.py"
    spec=importlib.util.spec_from_file_location("_new_inactive_C3_integration_source_only",ROOT/cp)
    actual=importlib.util.module_from_spec(spec);spec.loader.exec_module(actual)
    good("old_saved600_gate_is_delegated_without_rebinding",
        actual.legacy.SOURCE==actual.LEGACY_CONTRACT_REF["path"]
        and actual.plan_ready_v1 is not actual.legacy.plan_ready_v1
        and actual.terminal0_v1 is actual.legacy.terminal0_v1)
    source,nodes=segment(cp,[])
    legacy_source,oldnodes=segment(actual.LEGACY_CONTRACT_REF["path"],[])
    for name in ("set_active_spec_v1","phase_paths"):
        good(name+"_original_AST_unchanged",
            ast.dump(nodes[name],include_attributes=False)==ast.dump(oldnodes[name],include_attributes=False))
    oldbuilder,oldb=segment(actual.BUILDER_V2_REF["path"],[])
    for name in ("prediction_metadata_v1","matrix_metadata_v1"):
        good(name+"_reviewed_builder_v2_AST_unchanged",
            ast.dump(nodes[name],include_attributes=False)==ast.dump(oldb[name],include_attributes=False))
    worker,wn=segment(HERE+"/worker_entry_v1.py",[])
    oldworker,ow=segment(actual.LEGACY_ENTRY_REF["path"],[])
    for name in ("execute","matrix_v1","score_v1"):
        good("guarded_worker_"+name+"_original_AST_unchanged",
            ast.dump(wn[name],include_attributes=False)==ast.dump(ow[name],include_attributes=False))
    good("future_phase_set_is_narrow",actual.PHASES==("predict_val_003","predict_val_004","matrix","score"))
    good("fixed_existing_input_SHAs_preserved",[x["sha256"]for x in actual.OLD_INPUT_REFS.values()]==[
        "9318a6c9a0f01487803632894ed92f4287600a5ad808bc88bb2d65a36bead869",
        "8ac934c50f764358aa1f61347d6832416731d6772fa927503535016280aba2c1",
        "6b3d657757268b1fb40f5196afe37e781fc47fdad316d6b5085f7e4375f404df"])
    base=json.loads((ROOT/actual.LEGACY_MANIFEST_REF["path"]).read_text())
    good("base_source_manifest_hash_authenticated",
        hashlib.sha256((ROOT/actual.LEGACY_MANIFEST_REF["path"]).read_bytes()).hexdigest()==actual.LEGACY_MANIFEST_REF["sha256"])
    refs={}
    for x in [actual.LEGACY_CONTRACT_REF,actual.LEGACY_MANIFEST_REF,actual.LEGACY_DRIVER_REF,
        actual.LEGACY_ENTRY_REF,actual.LEGACY_HANDOFF_REF,actual.LEGACY_MAIN_REVIEW_REF,
        actual.BUILDER_V2_REF,actual.BUILDER_V2_HANDOFF_REF,actual.BUILDER_V2_REVIEW_REF,
        actual.PREP_HELPER_REF,actual.SCANNER_REF,actual.BRIDGE_REF,actual.SCANNER_INIT_REF,
        actual.SCANNER_MANIFEST_REF,actual.SCANNER_HANDOFF_REF,actual.SCANNER_REVIEW_REF]:
        refs[x["path"]]=x
    def ref(path):
        path=str(path)
        if path.startswith(str(ROOT)+"/"):path=path[len(str(ROOT))+1:]
        if path not in refs:
            p=ROOT/path
            refs[path]=dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) if path.endswith(".py") and p.is_file() else F("SOURCE_REF_"+path)
            refs[path]["path"]=path
        return copy.deepcopy(refs[path])
    for x in actual.effective_extra_refs_v1():refs[x["path"]]=x
    docs={};read_calls=[];plan_calls=[];legacy_accept_calls=[]
    def put(label,value,record=None):
        rec=record or F(label);docs[rec["path"]]=copy.deepcopy(value);return rec
    def read(record):
        require(not record["path"].endswith((".npz",".npy",".pt",".trees")),
            "No synthetic or actual payload opening permitted in integration tests")
        read_calls.append(record["path"]);return copy.deepcopy(docs[record["path"]])
    for path in (actual.PLAN_ROUTE_REGISTRY,actual.APPROVED_SELECTOR_SOURCES):
        put("real_source_metadata",json.loads((ROOT/path).read_text()),actual.ref(path))
    selector_ref=F("exact_explicit_ROOT_selector")
    put("runtime",dict(prefix="IN_MEMORY_QUALIFIED_RUNTIME_ONLY"),actual.RUNTIME_REF)
    def active(value):
        require(value.get("active") is True and value.get("execution_authorized") is True
            and value.get("root_authorizer")=="root_agent","Explicit fake ROOT activity")
        require(datetime.fromisoformat(value["expires_at_utc"])>datetime.now(timezone.utc),
            "Unexpired fake authority")
    oldns=dict(vars(actual.legacy))
    oldns.update(ref=ref,manifest_v1=lambda:copy.deepcopy(base),require=require)
    exec(compile(ast.Module(body=[oldnodes["required_direct_refs_v1"]],type_ignores=[]),
        "<unchanged_old_required_roles>","exec"),oldns)
    training_ref=F("training")
    candidates={e:F("checkpoint_"+str(e))for e in actual.EPOCHS}
    training=dict(checkpoint_candidates=[dict(epoch=e,checkpoint_ref=r)for e,r in candidates.items()])
    accepted=dict(training_completion_ref=training_ref,source_review_ref=actual.LEGACY_MAIN_REVIEW_REF)
    def old_accept(record):
        legacy_accept_calls.append(record);require(record==actual.SAVED_ACCEPTANCE_REF,"Exact old saved600 acceptance")
        return copy.deepcopy(accepted),copy.deepcopy(training)
    legacy=SimpleNamespace(**vars(actual.legacy))
    legacy.manifest_v1=lambda:copy.deepcopy(base)
    legacy.required_direct_refs_v1=oldns["required_direct_refs_v1"]
    legacy.training_ready_v1=old_accept
    def local(path):
        return ROOT/path if not Path(str(path)).is_absolute() else Path(path)
    plans={};plan_evidence={}
    for parent in actual.VAL:
        obs=put(parent+"_observed",dict(source_observation_ref=F(parent+"_sim_observation")))
        plans[parent]=dict(focal_registry_ref=F(parent+"_focals"),observation_ref=obs)
        plan_evidence[parent]=dict(parent_id=parent,closed=True,genuine_wait_and_outer=True)
    def plan_ready(e,parent):
        plan_calls.append(parent)
        require(e==plan_evidence[parent] and e["closed"] is True and e["genuine_wait_and_outer"] is True,"Same actual closed-plan prerequisite")
        return copy.deepcopy(plans[parent])
    def terminal(e,completion):
        wait=read(e["actual_wait_ref"]);outer=read(e["outer_terminal_ref"]);env=read(completion)
        require(wait["completed_ref"]==completion and wait["input_ref"]==env["input_ref"]
            and wait["driver_ref"]==env["root_driver_ref"] and type(wait["returncode"])is int
            and wait["returncode"]==0 and wait["all_owned_processes_and_groups_drained"] is True
            and wait["operational_failure"] is None and outer["actual_terminal_consumed"] is True
            and outer["completed_ref"]==completion and outer["actual_wait_ref"]==e["actual_wait_ref"],
            "Joined exact fake wait/drain/consumed outer facts")
        return wait
    ns=dict(vars(actual))
    ns.update(require=require,read=read,ref=ref,local=local,active=active,legacy=legacy,
        plan_ready_v1=plan_ready,terminal0_v1=terminal,_ACTIVE_SPEC=None)
    ns["SELECTOR_REF"]=selector_ref;ns["SELECTED_MODE"]="native14_stopped_retention_v3"
    def fake_select(record):
        require(record==selector_ref,"Exact fake immutable ROOT selector")
        return {}
    ns["select_storage_metadata_v1"]=fake_select
    selected=[nodes[n]for n in ["caps_v1","training_ready_v1","effective_extra_refs_v1","manifest_v1",
        "integration_review_v1","stage_source_v1","actual_stage_v1","prediction_metadata_v1",
        "matrix_metadata_v1","evidence_request_fields_v1","validate_request","validate_input_v1",
        "set_active_spec_v1","phase_paths","required_direct_refs_v1"]]
    exec(compile(ast.Module(body=selected,type_ignores=[]),"<actual_integration_functions_fake_IO_only>","exec"),ns)
    effective=copy.deepcopy(base)
    effective.update(schema="C3_CPU_actual_plan_lineage_source_manifest_v1",
        legacy_manifest_ref=actual.LEGACY_MANIFEST_REF)
    mapping=dict(base["source_sha256"])
    for x in ns["effective_extra_refs_v1"]():mapping[x["path"]]=x["sha256"]
    effective["source_sha256"]=mapping
    put("effective_manifest",effective,ref(actual.MANIFEST))
    put("new_handoff",{},ref(actual.HANDOFF))
    review_ref=F("future_complete_independent_review_ONLY_FIXTURE")
    review=dict(schema="C3_full30Mb_eval_preparation_independent_review_v1",
        independent=True,verdict="SOURCE_PASS",source_ref=ref(actual.BUILDER),
        source_handoff_ref=ref(actual.HANDOFF),main_manifest_ref=actual.LEGACY_MANIFEST_REF,
        validator_preparation_source_ref=actual.PREP_HELPER_REF,
        integration_contract_ref=ref(actual.SOURCE),integration_worker_ref=ref(actual.ENTRY),
        integration_driver_ref=ref(actual.DRIVER),integration_manifest_ref=ref(actual.MANIFEST),
        reused_builder_v2_ref=actual.BUILDER_V2_REF,scanner_handoff_ref=actual.SCANNER_HANDOFF_REF,
        scanner_review_ref=actual.SCANNER_REVIEW_REF,fixed_old_prediction_input_refs=actual.OLD_INPUT_REFS,
        plan_lineage_routes_ref=ref(actual.PLAN_ROUTE_REGISTRY),selector_source_ref=actual.SELECTOR_SOURCE_REF,
        predecessor_integration_handoff_ref=actual.PRIOR_INTEGRATION_HANDOFF_REF)
    put("new_review",review,review_ref)
    good("complete_effective_source_closure_accepts",ns["manifest_v1"]()["source_sha256"]==mapping)
    for path in (actual.SCANNER_REF["path"],actual.BRIDGE_REF["path"],"src/arggnn/american_c3_gnn_model_v1.py"):
        old=docs[ref(actual.MANIFEST)["path"]]["source_sha256"].pop(path)
        rejects("missing_or_replaced_source_"+Path(path).stem,ns["manifest_v1"])
        docs[ref(actual.MANIFEST)["path"]]["source_sha256"][path]=old
    good("complete_review_gate_accepts",ns["integration_review_v1"](review_ref)==review)
    for key in ("scanner_review_ref","integration_worker_ref","fixed_old_prediction_input_refs"):
        saved=docs[review_ref["path"]][key];docs[review_ref["path"]][key]=None
        rejects("incomplete_integration_review_"+key,lambda:ns["integration_review_v1"](review_ref))
        docs[review_ref["path"]][key]=saved
    expir=(datetime.now(timezone.utc)+timedelta(days=2)).isoformat()
    def make_request(phase,evidence):
        parent="american_c3_val_"+phase[-3:] if phase.startswith("predict")else None
        fields,_,_=ns["evidence_request_fields_v1"](SimpleNamespace(**ns),evidence,phase)
        evidence_ref=put("stage_evidence_"+phase,evidence)
        directive_ref=put("ROOT_directive_"+phase,dict(root_authorizer="root_agent",active=True,
            execution_authorized=True,phase=phase,directive=actual.DIRECTIVE,
            builder_review_ref=review_ref,actual_stage_evidence_ref=evidence_ref,
            expires_at_utc=expir,actual_ROOT_artifact_acceptance_ref=actual.SAVED_ACCEPTANCE_REF,
            main_review_ref=actual.LEGACY_MAIN_REVIEW_REF,original_history=actual.ORIGINAL_HISTORY,
            all5VALx6checkpoint_barrier_preserved=True,storage_policy_selector_ref=selector_ref))
        capref=put("caps_"+phase,actual.legacy.caps_v1(phase))
        req=dict(schema="C3_full30Mb_evaluation_ROOT_request_v1",active=True,root_authorizer="root_agent",
            execution_authorized=True,created_at_utc=datetime.now(timezone.utc).isoformat(),
            expires_at_utc=expir,phase=phase,device="cpu",GPU_authorized=False,physical_gpu=None,
            runtime_ref=actual.RUNTIME_REF,source_manifest_ref=ref(actual.MANIFEST),
            packet_directory="research/reviews/C3_full30Mb_eval_ROOT_actual_plan_lineage_"+phase+"_FIXTURE",
            output_directory="experiments/results/C3_full30Mb_eval_actual_plan_lineage_"+phase+"_FIXTURE",
            caps_ref=capref,training_evidence_ref=actual.SAVED_ACCEPTANCE_REF,
            integration_review_ref=review_ref,ROOT_directive_ref=directive_ref,
            actual_stage_evidence_ref=evidence_ref,storage_policy_selector_ref=selector_ref,**{k:v for k,v in fields.items()if k!="training_evidence_ref"})
        return req
    def evidence(phase):
        needed={actual.VAL[int(phase[-3:])]}if phase.startswith("predict")else set(actual.VAL)
        value=dict(schema="C3_full30Mb_eval_actual_stage_evidence_v1",phase=phase,
            ROOT_artifact_acceptance_ref=actual.SAVED_ACCEPTANCE_REF,
            closed_plan_evidence_by_parent={p:copy.deepcopy(plan_evidence[p])for p in actual.VAL if p in needed})
        if phase=="matrix":value["VAL_predictions"]=copy.deepcopy(predictions)
        if phase=="score":value.update(matrix=copy.deepcopy(matrix_evidence),VAL_simulation_completion_refs=copy.deepcopy(simulations))
        return value
    # Every stage's source documents below are mocked in memory, including the
    # source-frozen existing input refs. They are never read from actual files.
    predictions={};simulations={};cells=[]
    for index,parent in enumerate(actual.VAL):
        phase="predict_val_%03d"%index
        reqref=F("producer_request_"+phase)
        inp=actual.OLD_INPUT_REFS[phase]if index<3 else F("future_input_"+phase)
        old=index<3
        driver=actual.LEGACY_DRIVER_REF if old else ref(actual.DRIVER)
        entry=actual.LEGACY_ENTRY_REF if old else ref(actual.ENTRY)
        source_map=base["source_sha256"]if old else mapping
        put("producer_req_"+phase,dict(phase=phase,source_manifest_ref=actual.LEGACY_MANIFEST_REF if old else ref(actual.MANIFEST),
            training_evidence_ref=actual.SAVED_ACCEPTANCE_REF),reqref)
        put("producer_input_"+phase,dict(phase=phase,root_driver_ref=driver,main_source_ref=entry,
            source_sha256=source_map,refs=dict(runtime=actual.RUNTIME_REF),
            root_stage_request_ref=reqref,packet_directory="research/reviews/C3_full30Mb_eval_ROOT_actual_plan_lineage_"+phase+"_FIXTURE"),inp)
        seals=[]
        for epoch in actual.EPOCHS:
            pr=F(parent+"_"+str(epoch)+"_predictions")
            sr=put(parent+"_"+str(epoch)+"_seal",dict(status="sealed_C3_original_heldout_predictions",
                parent_id=parent,epoch=epoch,split="val",records=10000,private_truth_opened=False,
                training_completion_ref=training_ref,checkpoint_ref=candidates[epoch],
                probabilities_modified=False,prediction_ref=pr,focal_registry_ref=plans[parent]["focal_registry_ref"]))
            seals.append(sr);cells.append(dict(epoch=epoch,parent_id=parent,checkpoint_ref=candidates[epoch],
                prediction_seal_ref=sr,prediction_ref=pr,focal_registry_ref=plans[parent]["focal_registry_ref"]))
        six=put(parent+"_all6",dict(status="sealed_C3_one_original_VAL_all6_candidates",
            parent_id=parent,training_completion_ref=training_ref,private_truth_opened=False,prediction_seal_refs=seals))
        completed=put(phase+"_completion",dict(stage=phase,status=actual.legacy.STATUSES[phase],
            root_driver_ref=driver,input_ref=inp,training_completion_ref=training_ref,result_ref=six))
        wait=put(phase+"_wait",dict(returncode=0,driver_ref=driver,input_ref=inp,completed_ref=completed,
            all_owned_processes_and_groups_drained=True,operational_failure=None))
        outer=put(phase+"_outer",dict(actual_exit_code=0,actual_terminal_consumed=True,
            completed_ref=completed,actual_wait_ref=wait))
        predictions[parent]=dict(completion_ref=completed,actual_wait_ref=wait,outer_terminal_ref=outer)
        sim_obs=docs[plans[parent]["observation_ref"]["path"]]["source_observation_ref"]
        simulations[parent]=put(parent+"_simulation",dict(status="completed_C3_simulation_reconstruction",
            parent_id=parent,class_order=["EUR","EAS","AFR"],observation_ref=sim_obs))
    matrix_result=put("matrix_result",dict(status="sealed_C3_all30_original_VAL_predictions",
        training_completion_ref=training_ref,all_predictions_sealed_before_VAL_truth=True,
        TEST_deferred=True,checkpoint_epochs=list(actual.EPOCHS),validation_parent_ids=list(actual.VAL),cells=cells))
    matrix_req=put("matrix_producer_request",dict(phase="matrix",source_manifest_ref=ref(actual.MANIFEST),
        training_evidence_ref=actual.SAVED_ACCEPTANCE_REF))
    matrix_input=put("matrix_input",dict(phase="matrix",root_driver_ref=ref(actual.DRIVER),
        main_source_ref=ref(actual.ENTRY),source_sha256=mapping,refs=dict(runtime=actual.RUNTIME_REF),
        root_stage_request_ref=matrix_req,packet_directory="research/reviews/C3_full30Mb_eval_ROOT_actual_plan_lineage_matrix_FIXTURE"))
    matrix_cp=put("matrix_completion",dict(stage="matrix",status=actual.STATUSES["matrix"],
        input_ref=matrix_input,root_driver_ref=ref(actual.DRIVER),training_completion_ref=training_ref,result_ref=matrix_result))
    matrix_wait=put("matrix_wait",dict(returncode=0,driver_ref=ref(actual.DRIVER),input_ref=matrix_input,
        completed_ref=matrix_cp,all_owned_processes_and_groups_drained=True,operational_failure=None))
    matrix_outer=put("matrix_outer",dict(actual_exit_code=0,actual_terminal_consumed=True,
        completed_ref=matrix_cp,actual_wait_ref=matrix_wait))
    matrix_evidence=dict(completion_ref=matrix_cp,actual_wait_ref=matrix_wait,outer_terminal_ref=matrix_outer)
    access=SimpleNamespace(**ns)
    for i,parent in enumerate(actual.VAL):
        env=read(predictions[parent]["completion_ref"])
        good("mixed_route_accepts_exact_phase_"+str(i),ns["stage_source_v1"](env,"predict_val_%03d"%i)["phase"]=="predict_val_%03d"%i)
        bad=copy.deepcopy(env);bad["root_driver_ref"]=ref(actual.DRIVER)if i<3 else actual.LEGACY_DRIVER_REF
        rejects("no_old_new_driver_fallback_"+str(i),lambda:ns["stage_source_v1"](bad,"predict_val_%03d"%i))
    env=read(predictions[actual.VAL[0]]["completion_ref"]);env["input_ref"]=F("wrong_existing_input")
    rejects("old_original_input_ref_cannot_be_substituted",lambda:ns["stage_source_v1"](env,"predict_val_000"))
    for phase in ("predict_val_003","predict_val_004"):
        plan_calls.clear();req=make_request(phase,evidence(phase));rr=put("request_"+phase,req)
        good(phase+"_own_closed_plan_only",ns["validate_request"](rr)[0]==req and set(plan_calls)=={actual.VAL[int(phase[-3:])]})
        bad=copy.deepcopy(req);bad["VAL_truth_capability"]={}
        rejects(phase+"_truth_rejected",lambda:ns["validate_request"](put("bad_truth_"+phase,bad)))
        bad=copy.deepcopy(req);bad["phase"]="predict_val_000"
        rejects(phase+"_old_phase_relaunch_rejected",lambda:ns["validate_request"](put("bad_phase_"+phase,bad)))
        e=evidence(phase);e["closed_plan_evidence_by_parent"]={}
        rejects(phase+"_missing_own_plan_rejected",lambda:ns["evidence_request_fields_v1"](access,e,phase))
    for phase in ("matrix","score"):
        plan_calls.clear();req=make_request(phase,evidence(phase));rr=put("request_"+phase,req)
        good(phase+"_request_accepts_fixed_mixed_producers_and_all5plans",ns["validate_request"](rr)[0]==req and set(plan_calls)==set(actual.VAL))
        for parent in actual.VAL:
            e=evidence(phase);del e["closed_plan_evidence_by_parent"][parent]
            rejects(phase+"_cannot_omit_plan_"+parent,lambda:ns["evidence_request_fields_v1"](access,e,phase))
    saved=docs[matrix_result["path"]]["cells"]
    docs[matrix_result["path"]]["cells"]=saved[:-1]
    rejects("score_cannot_copy_truth_capabilities_before_all30cells",lambda:ns["evidence_request_fields_v1"](access,evidence("score"),"score"))
    docs[matrix_result["path"]]["cells"]=saved
    first=saved[0]["prediction_seal_ref"];old=docs[first["path"]]["focal_registry_ref"]
    docs[first["path"]]["focal_registry_ref"]=F("foreign_focal")
    rejects("score_requires_same_closed_plan_focal_support",lambda:ns["evidence_request_fields_v1"](access,evidence("score"),"score"))
    docs[first["path"]]["focal_registry_ref"]=old
    # Run original role union helper with a pure fake binding guard.
    helper_path=actual.PREP_HELPER_REF["path"];hs,hn=segment(helper_path,[])
    hns=dict(require=require)
    exec(compile(ast.Module(body=[hn["role_union_v1"]],type_ignores=[]),"<unchanged_role_union_fake_guard>","exec"),hns)
    class FakeGuard:
        def consumer_bindings_v1(self,*groups):
            indexed={}
            for group in groups:
                for x in group:
                    require(x["path"]not in indexed or indexed[x["path"]]==x,"Conflicting fake role")
                    indexed[x["path"]]=x
            require(len(indexed)<=64,"Hard64fake guard")
            return list(indexed.values())
    phase="predict_val_003";req=make_request(phase,evidence(phase))
    request_ref=put("request_for_input",req);cfg=actual.make_config(req,actual.caps_v1(phase));cfgref=put("config",cfg)
    inputdoc=dict(schema="C3_full30Mb_evaluation_active_input_v1",active=True,root_authorizer="root_agent",
        execution_authorized=True,expires_at_utc=expir,phase=phase,root_stage_request_ref=request_ref,
        root_driver_ref=ref(actual.DRIVER),main_source_ref=ref(actual.ENTRY),GPU_authorized=False,
        physical_gpu=None,output_directory=req["output_directory"],packet_directory=req["packet_directory"],
        storage_policy_selector_ref=selector_ref,source_sha256=mapping,refs=dict(config=cfgref,runtime=actual.RUNTIME_REF,storage=None),job_id="FAKE_JOB")
    required=ns["required_direct_refs_v1"](inputdoc,req)
    view=dict(bindings=base["current_guard_roles"])
    union=hns["role_union_v1"](SimpleNamespace(**ns),FakeGuard(),view,inputdoc,req,[])
    good("all_original_and_current_new_roles_preserved_within64",len(union)<=64 and all(x in union for x in base["current_guard_roles"].values()))
    badview=dict(bindings=dict(view["bindings"]))
    for i in range(65-len(union)):badview["bindings"]["unexpected"+str(i)]=F("unexpected65th_role_"+str(i))
    rejects("actual_returned65th_role_not_dropped",lambda:hns["role_union_v1"](SimpleNamespace(**ns),FakeGuard(),badview,inputdoc,req,[]))
    storage=dict(accounting_mode="native14_stopped_retention_v3",completed_overlay_policy_ref=actual.POLICY_REF,storage_policy_selector_ref=selector_ref,
        job_id="FAKE_JOB",output_directory=req["output_directory"],reserved_new_bytes=33554432,
        reservation=dict(output=str(local(req["output_directory"])),auxiliary_scopes=[
            str(local(req["packet_directory"]+"/worker_aux")),str(local(req["packet_directory"]+"/worker_tmp"))],
            consumer_bindings=union))
    sr=put("storage",storage);inputdoc["refs"]["storage"]=sr;ir=put("active_input",inputdoc)
    good("complete_new_input_gate_accepts_fake_guarded_input",ns["validate_input_v1"](ir)[0]==inputdoc)
    for missing in (actual.SCANNER_REF,actual.BRIDGE_REF,ref(actual.SOURCE),ref(actual.ENTRY)):
        prior=docs[sr["path"]]["reservation"]["consumer_bindings"]
        docs[sr["path"]]["reservation"]["consumer_bindings"]=[x for x in prior if x!=missing]
        rejects("input_missing_direct_role_"+Path(missing["path"]).stem,lambda:ns["validate_input_v1"](ir))
        docs[sr["path"]]["reservation"]["consumer_bindings"]=prior
    good("no_scientific_library_imported",not any(x in sys.modules for x in ("torch","torch_geometric","numpy","tskit")))
    good("no_payload_ref_read",not any(p.endswith((".npz",".npy",".pt",".trees"))for p in read_calls))
    return dict(schema="C3_interleaved_guarded_integration_inert_checks_v1",status="PASS_SOURCE_ONLY",
        checks=checks,check_count=len(checks),prospective_role_union=len(union),
        only_mocked_inputs_guards_reviews_and_predecessors=True,
        actual_guard_or_resource_probe_or_admission_or_job=False,
        numerical_arithmetic_tests_not_repeated=True,completed_utc=datetime.now(timezone.utc).isoformat())

if __name__=="__main__":
    try:
        result=run()
        p=ROOT/HERE/"inert_integration_checks.v1.json"
        with p.open("x")as f:json.dump(result,f,sort_keys=True,indent=2)
        print(json.dumps(dict(status=result["status"],checks=result["check_count"],roles=result["prospective_role_union"])))
    except BaseException as error:
        number=1
        while (ROOT/HERE/("inert_integration_checks.attempt%d.FAILURE.json"%number)).exists():number+=1
        p=ROOT/HERE/("inert_integration_checks.attempt%d.FAILURE.json"%number)
        with p.open("x")as f:json.dump(dict(error=repr(error),traceback=traceback.format_exc(),checks=checks),f,indent=2)
        raise
