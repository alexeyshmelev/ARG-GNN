"""ROOT-only later C3 CPU prediction/matrix/score admission. Default inert; never launches."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,copy,hashlib,importlib.util,json,re,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/C3_full30Mb_eval_admission_readiness_source_20261016_v2"
SOURCE=HERE+"/prepare_evaluation_root_v2.py"
PACKAGE="research/reviews/C3_saved600_artifact_acceptance_source_20261016_v1"
MAIN_HANDOFF_REF=dict(path=PACKAGE+"/source_handoff.v1.json",sha256="d57045ec59223ebbea48efcff9f67550717be84090efa4560855d110b31ee010")
MAIN_MANIFEST_REF=dict(path=PACKAGE+"/source_manifest.v1.json",sha256="a76e6eb58fef091dd80ca8ba918c4873b1a31b7f28f28b2e6e84104ec910af35")
PREP_HELPER_REF=dict(path="research/reviews/C3_saved600_validator_ROOT_preparation_source_20261016_v1/prepare_root_v1.py",
    sha256="810177e841b7bc2e3ca3ea29332826442122bd28ad0dd917661f13c7cab33781")
DIRECTIVE="ROOT authorizes one separately admitted unchanged C3 CPU evaluation stage from genuine saved600 artifact acceptance and actual closed plans and predecessor evidence"
PHASES=tuple("predict_val_%03d"%i for i in range(5))+("matrix","score")
GUARD_REF=dict(path="research/reviews/native14_stopped_retention_successor_source_20261016_v3/native14_stopped_retention_v1.py",
    sha256="60d010e393bda25f662381051b3fd3d4e4decfa034b10c3baad3c3f30a6a196c")
POLICY_REF=dict(path="research/reviews/native14_stopped_retention_ROOT_activation_20261016_v3/activation_policy.root_active_v1.json",
    sha256="72ed727feb1a824fa8f9c2b31ce374dca6728ca4e655e9562b4dedc84122f417")
def require(ok,message):
    if not ok:raise ValueError(message)
def reference(path):
    p=ROOT/path
    require(not Path(path).is_absolute() and ".." not in p.parts and p.resolve()==p
        and p.is_relative_to(ROOT) and p.is_file() and not p.is_symlink()
        and p.suffix in (".py",".json",".md",".sh") and p.stat().st_size<=8*2**20,
        "Bounded canonical actual source/JSON metadata only")
    return dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def bound(record):
    require(isinstance(record,dict) and set(record)=={"path","sha256"}
        and reference(record["path"])==record,"Actual explicit immutable source/JSON reference required")
    return (ROOT/record["path"]).read_bytes()
def load(record,label):
    bound(record);sp=importlib.util.spec_from_file_location(label,ROOT/record["path"])
    m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def builder_review_v1(review):
    require(review.get("schema")=="C3_full30Mb_eval_preparation_independent_review_v1"
        and review.get("independent") is True and review.get("verdict") in ("PASS","SOURCE_PASS")
        and review.get("source_ref")==reference(SOURCE)
        and review.get("source_handoff_ref")==reference(HERE+"/source_handoff.v1.json")
        and review.get("main_manifest_ref")==MAIN_MANIFEST_REF
        and review.get("validator_preparation_source_ref")==PREP_HELPER_REF,
        "Actual independent complete builder/source-handoff review, never a fixture or guessed future ref")
def actual_stage_v1(c,e,phase,training_ref):
    require(isinstance(e,dict) and e.get("outer_terminal_ref") is not None,
        "Actual stage completion/direct wait/consumed outer terminal refs required")
    env=c.read(e["completion_ref"]);wait=c.terminal0_v1(e,e["completion_ref"])
    terminal=c.read(e["outer_terminal_ref"])
    require(type(wait.get("returncode"))is int and wait["returncode"]==0
        and type(terminal.get("actual_exit_code"))is int and terminal["actual_exit_code"]==0,
        "Genuine integer stage wait0 and consumed outer0")
    require(env.get("stage")==phase and env.get("status")==c.STATUSES[phase]
        and env.get("root_driver_ref")==c.ref(c.DRIVER)
        and env.get("training_completion_ref")==training_ref,"Same reviewed saved600 evaluation producer/training lineage")
    return env,c.read(env["result_ref"])
def prediction_metadata_v1(c,parent,six,training_ref,candidates):
    require(six.get("status")=="sealed_C3_one_original_VAL_all6_candidates" and six.get("parent_id")==parent
        and six.get("training_completion_ref")==training_ref and six.get("private_truth_opened") is False
        and len(six["prediction_seal_refs"])==6,"One actual original VAL parent/all6 candidates before truth")
    seen=set()
    for record in six["prediction_seal_refs"]:
        seal=c.read(record);epoch=seal["epoch"]
        require(epoch in candidates and epoch not in seen and seal.get("parent_id")==parent
            and seal.get("status")=="sealed_C3_original_heldout_predictions" and seal.get("split")=="val"
            and seal.get("records")==10000 and seal.get("private_truth_opened") is False
            and seal.get("training_completion_ref")==training_ref
            and seal.get("checkpoint_ref")==candidates[epoch] and seal.get("probabilities_modified") is False,
            "All6 unique original metadata seals, raw probabilities and exact checkpoint lineage")
        seen.add(epoch)
    require(seen==set(c.EPOCHS),"No final-only or missing-checkpoint prediction packet")
def matrix_metadata_v1(c,matrix,training_ref,candidates,plans):
    expected={(e,p) for e in c.EPOCHS for p in c.VAL}
    require(matrix.get("status")=="sealed_C3_all30_original_VAL_predictions"
        and matrix.get("training_completion_ref")==training_ref and matrix.get("all_predictions_sealed_before_VAL_truth") is True
        and matrix.get("TEST_deferred") is True and matrix.get("checkpoint_epochs")==list(c.EPOCHS)
        and matrix.get("validation_parent_ids")==list(c.VAL) and len(matrix["cells"])==30,
        "Actual complete original5VALx6checkpoint pretruth matrix")
    seen=set()
    for cell in matrix["cells"]:
        key=cell["epoch"],cell["parent_id"]
        require(key in expected and key not in seen and cell["checkpoint_ref"]==candidates[key[0]],
            "Unique original30 Cartesian cells and exact checkpoint refs")
        seal=c.read(cell["prediction_seal_ref"])
        require(seal.get("status")=="sealed_C3_original_heldout_predictions" and seal.get("split")=="val"
            and seal.get("epoch")==key[0] and seal.get("parent_id")==key[1] and seal.get("records")==10000
            and seal.get("private_truth_opened") is False and seal.get("probabilities_modified") is False
            and seal.get("training_completion_ref")==training_ref
            and seal.get("checkpoint_ref")==cell["checkpoint_ref"]
            and seal.get("prediction_ref")==cell["prediction_ref"]
            and seal.get("focal_registry_ref")==cell["focal_registry_ref"]==plans[key[1]]["focal_registry_ref"],
            "All30 actual seals and same closed-plan focal registries before score capability")
        seen.add(key)
    require(seen==expected,"Full original30cells required before truth capability")
def evidence_request_fields_v1(c,evidence,phase):
    common={"schema","phase","ROOT_artifact_acceptance_ref","closed_plan_evidence_by_parent"}
    additional=({"VAL_predictions"} if phase=="matrix" else
        {"matrix","VAL_simulation_completion_refs"} if phase=="score" else set())
    require(set(evidence)==common|additional
        and evidence.get("schema")=="C3_full30Mb_eval_actual_stage_evidence_v1"
        and evidence.get("phase")==phase,"Exact phase-specific actual evidence;no extra truth/stage/future capability")
    b,training=c.training_ready_v1(evidence["ROOT_artifact_acceptance_ref"])
    training_ref=b["training_completion_ref"];candidates={x["epoch"]:x["checkpoint_ref"] for x in training["checkpoint_candidates"]}
    supplied=evidence["closed_plan_evidence_by_parent"]
    required_parents=({c.VAL[int(phase[-3:])]} if phase.startswith("predict_val_") else set(c.VAL))
    require(set(supplied)==required_parents,
        "Prediction requires only its own actual closed plan;matrix/score require all5 actual closed full30Mb plans")
    plans={p:c.plan_ready_v1(supplied[p],p) for p in c.VAL if p in required_parents}
    fields=dict(training_evidence_ref=evidence["ROOT_artifact_acceptance_ref"])
    if phase.startswith("predict_val_"):
        parent=c.VAL[int(phase[-3:])];fields["plan_evidence"]=copy.deepcopy(supplied[parent])
    elif phase=="matrix":
        require(set(evidence["VAL_predictions"])==set(c.VAL),"All5 actually completed prediction stages")
        for p,e in evidence["VAL_predictions"].items():
            _,six=actual_stage_v1(c,e,"predict_val_"+p[-3:],training_ref)
            prediction_metadata_v1(c,p,six,training_ref,candidates)
        fields["VAL_predictions"]=copy.deepcopy(evidence["VAL_predictions"])
    else:
        _,matrix=actual_stage_v1(c,evidence["matrix"],"matrix",training_ref)
        matrix_metadata_v1(c,matrix,training_ref,candidates,plans)
        simulations=evidence["VAL_simulation_completion_refs"]
        require(set(simulations)==set(c.VAL),"Only5 actual same-parent simulation completion refs supplied by ROOT")
        for p,r in simulations.items():
            sim=c.read(r);obs=c.read(plans[p]["observation_ref"])
            require(sim.get("status")=="completed_C3_simulation_reconstruction" and sim.get("parent_id")==p
                and sim.get("class_order")==["EUR","EAS","AFR"] and obs["source_observation_ref"]==sim["observation_ref"],
                "Actual same-parent simulation metadata;private truth payload is never opened by admission")
        fields.update(matrix=copy.deepcopy(evidence["matrix"]),plan_evidence_by_parent=copy.deepcopy(supplied),
            VAL_truth_capability=copy.deepcopy(simulations))
    return fields,b,training
def prepare_root_v1(args):
    require(args.root_directive==DIRECTIVE and args.phase in PHASES,"Explicit one-stage ROOT directive;validator excluded")
    require(Path.cwd()==ROOT and sys.dont_write_bytecode and sys.flags.no_user_site
        and Path(sys.executable).resolve()==(ROOT/".runtime/env/bin/python").resolve(),
        "Existing pinned fresh no-user-site/-B ROOT metadata interpreter")
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,47}",args.instance) is not None,"Fresh bounded ROOT instance")
    helper=load(PREP_HELPER_REF,"_C3_eval_exact_sealed_validator_prep_helpers")
    main=json.loads(bound(MAIN_HANDOFF_REF));require(main["source_manifest_ref"]==MAIN_MANIFEST_REF,"Exact sealed main handoff")
    manifest=json.loads(bound(MAIN_MANIFEST_REF));contract_path=PACKAGE+"/acceptance_contract_v1.py"
    c=load(dict(path=contract_path,sha256=manifest["source_sha256"][contract_path]),"_C3_eval_exact_saved600_contract")
    require(c.GUARD_REF==GUARD_REF and c.POLICY_REF==POLICY_REF and c.ref(c.MANIFEST)==MAIN_MANIFEST_REF,
        "Exact actual current native14v3 child/policy and unchanged saved600 package")
    c.manifest_v1()
    builder_review_ref=dict(path=args.builder_review_path,sha256=args.builder_review_sha256)
    builder_review_v1(json.loads(bound(builder_review_ref)))
    main_review_ref=dict(path=args.main_review_path,sha256=args.main_review_sha256)
    c.review_values_v1(c.read(main_review_ref),c.ref(c.SOURCE),MAIN_MANIFEST_REF)
    evidence_ref=dict(path=args.evidence_path,sha256=args.evidence_sha256)
    evidence=json.loads(bound(evidence_ref))
    fields,accepted,training=evidence_request_fields_v1(c,evidence,args.phase)
    require(accepted["source_review_ref"]==main_review_ref,"Same actual main review used by ROOT artifact acceptance")
    caps=c.caps_v1(args.phase)
    require(caps==c.read(c.ref(PACKAGE+"/caps_"+args.phase+".INACTIVE.v1.json"))
        and caps["reservation_cap_bytes"]==33554432 and caps["CPU_threads"]==32
        and caps["dataloader_workers"]==caps["max_cuda_reserved_bytes"]==0
        and caps["max_cpu_rss_bytes"]==137438953472 and caps["operational_deadline_seconds"]==86400,
        "Unchanged original CPU full-stage caps/runtime budget;no cropping or shorter phase")
    expiry=datetime.fromisoformat(args.expires_at_utc.replace("Z","+00:00"))
    require(expiry.tzinfo is not None and (expiry-datetime.now(timezone.utc)).total_seconds()>caps["operational_deadline_seconds"]+600,
        "ROOT supplies fresh expiry covering full original stage budget and preparation/drain reserve")
    expiry_text=expiry.isoformat();tag=args.phase+"_"+args.instance
    packet_rel="research/reviews/C3_full30Mb_eval_ROOT_"+tag;output="experiments/results/C3_full30Mb_eval_"+tag
    packet=c.local(packet_rel)
    require(not packet.exists() and not packet.is_symlink() and not c.local(output).exists()
        and not c.local(output).is_symlink(),"One fresh exclusive packet/output;no reuse or retry")
    job="C3-full30Mb-CPU-"+tag
    request=dict(schema="C3_full30Mb_evaluation_ROOT_request_v1",active=True,root_authorizer="root_agent",
        execution_authorized=True,created_at_utc=datetime.now(timezone.utc).isoformat(),expires_at_utc=expiry_text,
        phase=args.phase,device="cpu",GPU_authorized=False,physical_gpu=None,runtime_ref=c.RUNTIME_REF,
        source_manifest_ref=MAIN_MANIFEST_REF,packet_directory=packet_rel,output_directory=output,**fields)
    packet.mkdir()
    caps_ref=c.save(packet,"caps.json",caps);request["caps_ref"]=caps_ref
    directive_ref=c.save(packet,"ROOT_evaluation_stage_directive.json",dict(
        schema="ROOT_C3_full30Mb_one_CPU_evaluation_stage_directive_v1",root_authorizer="root_agent",active=True,
        execution_authorized=True,expires_at_utc=expiry_text,phase=args.phase,directive=args.root_directive,
        actual_recorded_utc=request["created_at_utc"],actual_stage_evidence_ref=evidence_ref,
        actual_ROOT_artifact_acceptance_ref=evidence["ROOT_artifact_acceptance_ref"],
        main_review_ref=main_review_ref,builder_review_ref=builder_review_ref,
        original_history=copy.deepcopy(c.ORIGINAL_HISTORY),all5VALx6checkpoint_barrier_preserved=True,
        original_producer_success_or_outer_wait_claimed=False,controller_shutdown_or_edit_authorized=False,
        close_credit_retire_retraining_or_automatic_retry_authorized=False))
    request_ref=c.save(packet,"request.root_active_v1.json",request)
    config=c.make_config(request,caps);config_ref=c.save(packet,"config.json",config)
    req,actual_caps,_=c.validate_request(request_ref)
    require(req==request and actual_caps==caps,"Actual unchanged phase contract before current-child mutation")
    spec=c.read(c.ref(PACKAGE+"/input.INACTIVE.template.v1.json"));spec.pop("INACTIVE",None)
    spec.update(status="active",active=True,root_authorizer="root_agent",execution_authorized=True,
        expires_at_utc=expiry_text,created_at_utc=request["created_at_utc"],job_id=job,phase=args.phase,
        output_directory=output,packet_directory=packet_rel,root_stage_request_ref=request_ref,caps_ref=caps_ref,
        parent_id=c.PARENT,source_sha256=copy.deepcopy(manifest["source_sha256"]),
        refs=dict(config=config_ref,runtime=c.RUNTIME_REF,storage=None),
        max_cpu_rss_bytes=caps["max_cpu_rss_bytes"],max_cuda_reserved_bytes=0,
        root_auxiliary_directory=packet_rel+"/worker_aux",completed_overlay_policy_ref=c.POLICY_REF)
    guard=c.original_guard()
    require(guard.MODE=="native14_stopped_retention_v3","Actual current child,never frozen ancestor mutation")
    require(all(getattr(guard.budget.v4,n).__module__==guard.budget.v4.__name__
        for n in ("canonical","file_bytes","mutex")),"Fresh original unwrapped ROOT mutation role")
    view=guard.inspect_v1(expected_source_sha256=c.CORE,expected_launcher_sha256=c.BASE)
    extra=[directive_ref,evidence_ref,reference(SOURCE),reference(HERE+"/source_handoff.v1.json"),
        builder_review_ref,main_review_ref,PREP_HELPER_REF,MAIN_HANDOFF_REF]
    roles=helper.role_union_v1(c,guard,view,spec,req,extra)
    reserve_request=dict(key="H041/"+job,output=str(c.local(output)),
        auxiliary_scopes=[str(c.local(packet_rel+"/worker_aux")),str(c.local(packet_rel+"/worker_tmp"))],
        reserved_bytes=caps["reservation_cap_bytes"],consumer_bindings=roles)
    authority=guard.operation_draft_v1("reserve",reserve_request,expected_before=guard._pins(view),bindings=view["bindings"])
    authority.update(active=True,expires_at_utc=expiry_text)
    authority_ref=c.save(packet,"reserve_authority.json",authority)
    reserved=guard.apply_reserve_v1(authority_ref,expected_source_sha256=c.CORE,expected_launcher_sha256=c.BASE)
    reserve_ref=c.save(packet,"reserve_receipt.json",reserved)
    storage=helper.storage_v1(c,guard,job,output,reserved,expiry_text)
    guard.validate_storage_v9(spec,config,storage,source_ref=c.ref(c.ENTRY),config_ref=config_ref,runtime_ref=c.RUNTIME_REF)
    storage_ref=c.save(packet,"storage.json",storage);spec["refs"]["storage"]=storage_ref
    input_ref=c.save(packet,"input.json",spec)
    actual_spec,actual_req,actual_caps=c.validate_input_v1(input_ref)
    require(actual_spec==spec and actual_req==req and actual_caps==caps,"Actual immutable admitted source/input/storage/role joins")
    admitted_ref=c.save(packet,"admitted.json",dict(input_ref=input_ref,driver_ref=c.ref(c.DRIVER),
        phase=args.phase,parent_id=c.PARENT,reserve_receipt_ref=reserve_ref,storage=storage,
        config_ref=config_ref,runtime_ref=c.RUNTIME_REF,caps_ref=caps_ref,consumer_binding_count=len(roles),
        execution_started=False,actual_stage_evidence_ref=evidence_ref,original_history=copy.deepcopy(c.ORIGINAL_HISTORY)))
    require((expiry-datetime.now(timezone.utc)).total_seconds()>caps["operational_deadline_seconds"]+300,
        "Full unchanged phase budget and terminal reserve remain;no expiry reset")
    command=["setsid",str(ROOT/".runtime/env/bin/python"),"-B",str(ROOT/c.DRIVER),
        "--launch","--input",input_ref["path"],"--expected-input-sha256",input_ref["sha256"]]
    return dict(status="ADMITTED_C3_CPU_EVALUATION_STAGE_NOT_EXECUTED",phase=args.phase,input_ref=input_ref,
        admitted_ref=admitted_ref,reserve_receipt_ref=reserve_ref,storage_ref=storage_ref,consumer_binding_count=len(roles),
        reserved_bytes=caps["reservation_cap_bytes"],exact_one_ROOT_stage_execution_argv=command,
        execution_command_emitted_not_invoked=True,no_automatic_retry=True,original_history=copy.deepcopy(c.ORIGINAL_HISTORY))
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--prepare-root",action="store_true")
    p.add_argument("--phase",choices=PHASES)
    for name in ("main-review-path","main-review-sha256","builder-review-path","builder-review-sha256",
        "evidence-path","evidence-sha256","instance","expires-at-utc","root-directive"):p.add_argument("--"+name)
    a=p.parse_args(argv)
    if not a.prepare_root:
        print(json.dumps(dict(status="SOURCE_ONLY_INACTIVE",phases=list(PHASES),admission=False,evaluation=False)));return 0
    require(all(getattr(a,n) for n in ("phase","main_review_path","main_review_sha256","builder_review_path",
        "builder_review_sha256","evidence_path","evidence_sha256","instance","expires_at_utc"))
        and a.root_directive==DIRECTIVE,"Explicit actual ROOT reviews,acceptance/stage evidence,phase,instance,expiry and directive required")
    print(json.dumps(prepare_root_v1(a),sort_keys=True),flush=True);return 0
if __name__=="__main__":raise SystemExit(main())
