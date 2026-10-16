"""ROOT-only admission for the dedicated saved600 CPU validator. Default inert."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,copy,hashlib,importlib.util,json,os,re,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
PACKAGE="research/reviews/C3_saved600_artifact_acceptance_source_20261016_v1"
HERE="research/reviews/C3_saved600_validator_ROOT_preparation_source_20261016_v1"
HANDOFF_REF=dict(path=PACKAGE+"/source_handoff.v1.json",sha256="d57045ec59223ebbea48efcff9f67550717be84090efa4560855d110b31ee010")
MANIFEST_REF=dict(path=PACKAGE+"/source_manifest.v1.json",sha256="a76e6eb58fef091dd80ca8ba918c4873b1a31b7f28f28b2e6e84104ec910af35")
DIRECTIVE="ROOT authorizes one dedicated CPU validation of saved C3 artifacts despite producer1 and outerUNKNOWN and permanently revokes original producer writes and retries"

def require(value,message):
    if not value:raise ValueError(message)

def source_bytes_v1(record):
    require(isinstance(record,dict) and set(record)=={"path","sha256"},"Exact source/metadata ref")
    path=ROOT/record["path"]
    require(not Path(record["path"]).is_absolute() and ".." not in path.parts
        and path.resolve()==path and path.is_relative_to(ROOT) and path.is_file() and not path.is_symlink()
        and path.suffix in (".py",".json",".md",".sh") and path.stat().st_size<=8*2**20,
        "Bounded canonical public source/metadata only")
    raw=path.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==record["sha256"],"Actual source/metadata SHA")
    return raw

def role_union_v1(c,guard,view,spec,req,extra):
    returned=list(view["bindings"].values())
    required=c.required_direct_refs_v1(spec,req)
    roles=guard.consumer_bindings_v1(returned,required,list(extra))
    require(1<=len(roles)<=64,"Every actual returned/current/operational role within original64 cap")
    for rows in (returned,required,list(extra)):
        require(all(r in roles for r in rows),"Never drop a returned or mandatory role")
    indexed={}
    for record in roles:
        require(record["path"] not in indexed or indexed[record["path"]]==record,"Conflicting exact source role")
        indexed[record["path"]]=record
    return roles

def storage_v1(c,guard,job,output,reserved,expiry):
    return dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,
        accounting_mode=guard.MODE,job_id=job,output_directory=output,reserved_new_bytes=33554432,
        reservation=copy.deepcopy(reserved["value"]),initial_ledger_pins=copy.deepcopy(reserved["initial_ledger_pins"]),
        expires_at_utc=expiry,completed_overlay_policy_ref=c.POLICY_REF,
        budget_components=dict(implementation=c.ref(guard.budget.SOURCE_REL),
            launcher=c.ref(guard.budget.LAUNCHER_REL),consumer_adapter=c.GUARD_REF))

def prepare_root_v1(args):
    require(args.root_directive==DIRECTIVE,"Explicit one-shot ROOT saved-artifact directive")
    require(Path.cwd()==ROOT and sys.dont_write_bytecode and sys.flags.no_user_site,
        "Exact repository cwd, no bytecode or user site")
    require(Path(sys.executable).resolve()==(ROOT/".runtime/env/bin/python").resolve(),"Existing pinned ROOT metadata interpreter")
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,47}",args.instance) is not None,"Unique bounded ROOT instance name")
    handoff=json.loads(source_bytes_v1(HANDOFF_REF))
    require(handoff["source_manifest_ref"]==MANIFEST_REF,"Exact sealed saved600 source packet")
    manifest=json.loads(source_bytes_v1(MANIFEST_REF))
    source_path=PACKAGE+"/acceptance_contract_v1.py"
    source_ref=dict(path=source_path,sha256=manifest["source_sha256"][source_path])
    source_bytes_v1(source_ref)
    loader=importlib.util.spec_from_file_location("_ROOT_saved600_exact_contract",ROOT/source_path)
    c=importlib.util.module_from_spec(loader);loader.loader.exec_module(c)
    require(c.GUARD_REF==dict(path="research/reviews/native14_stopped_retention_successor_source_20261016_v3/native14_stopped_retention_v1.py",
        sha256="60d010e393bda25f662381051b3fd3d4e4decfa034b10c3baad3c3f30a6a196c")
        and c.POLICY_REF==dict(path="research/reviews/native14_stopped_retention_ROOT_activation_20261016_v3/activation_policy.root_active_v1.json",
        sha256="72ed727feb1a824fa8f9c2b31ce374dca6728ca4e655e9562b4dedc84122f417"),
        "Exact actual native14v3 source and active policy")
    require(c.ref(c.MANIFEST)==MANIFEST_REF,"Actual reviewed manifest unchanged")
    c.manifest_v1()
    review_ref=dict(path=args.review_path,sha256=args.review_sha256)
    review=c.read(review_ref)
    c.review_values_v1(review,c.ref(c.SOURCE),MANIFEST_REF)
    caps=c.caps_v1(c.ACCEPT_PHASE)
    require(caps==c.read(c.ref(PACKAGE+"/caps_validate_saved600.INACTIVE.v1.json"))
        and caps["reservation_cap_bytes"]==33554432 and caps["operational_deadline_seconds"]==900
        and caps["max_cpu_rss_bytes"]==2147483648 and caps["CPU_threads"]==1
        and caps["dataloader_workers"]==caps["max_cuda_reserved_bytes"]==0,"Exact CPU validator failstop caps")
    expiry=datetime.fromisoformat(args.expires_at_utc.replace("Z","+00:00"))
    require(expiry.tzinfo is not None and (expiry-datetime.now(timezone.utc)).total_seconds()>caps["operational_deadline_seconds"]+600,
        "Explicit future validator expiry covers original full900s plus preparation/drain reserve")
    expiry_text=expiry.isoformat()
    tag="saved600_validator_"+args.instance
    packet_rel="research/reviews/C3_full30Mb_eval_ROOT_"+tag
    output="experiments/results/C3_full30Mb_eval_"+tag
    packet=c.local(packet_rel)
    require(not packet.exists() and not packet.is_symlink() and not c.local(output).exists()
        and not c.local(output).is_symlink(),"One fresh exclusive validator packet/output, no retry/replacement")
    job="C3-saved600-CPU-validator-"+args.instance
    request=c.read(c.ref(PACKAGE+"/ROOT_validation_request.INACTIVE.template.v1.json"))
    request.update(active=True,execution_authorized=True,root_authorizer="root_agent",
        expires_at_utc=expiry_text,created_at_utc=datetime.now(timezone.utc).isoformat(),
        packet_directory=packet_rel,output_directory=output,source_manifest_ref=MANIFEST_REF,
        source_review_ref=review_ref,original_history=copy.deepcopy(c.ORIGINAL_HISTORY),
        ROOT_accepts_original_producer1_and_outerUNKNOWN_for_artifact_validation_only=True,
        future_original_C3_training_writes_cooperatively_revoked=True,
        controller_preserved_for_borrowed_native_work=True)
    # The current child explicitly charges <=512KiB per-event packet metadata.
    # This mirrors its reviewed ROOT admission workflow. No old scope is written.
    packet.mkdir()
    caps_ref=c.save(packet,"caps.json",caps)
    request["caps_ref"]=caps_ref
    directive=dict(schema="ROOT_C3_saved600_one_CPU_validator_directive_v1",root_authorizer="root_agent",
        directive=args.root_directive,active=True,execution_authorized=True,expires_at_utc=expiry_text,
        actual_recorded_utc=datetime.now(timezone.utc).isoformat(),source_handoff_ref=HANDOFF_REF,
        source_manifest_ref=MANIFEST_REF,source_review_ref=review_ref,purpose="validate_saved_artifacts_only",
        original_history=copy.deepcopy(c.ORIGINAL_HISTORY),controller_preserved_for_borrowed_native_work=True,
        original_producer_success_claimed=False,original_outer_terminal_consumed_claimed=False,
        acceptance_prediction_scoring_retraining_or_close_authorized=False)
    directive_ref=c.save(packet,"ROOT_validator_directive.json",directive)
    revocation=dict(schema="ROOT_C3_saved600_permanent_original_producer_write_revocation_v1",
        root_authorizer="root_agent",status="permanently_cooperatively_revoked",
        actual_recorded_utc=datetime.now(timezone.utc).isoformat(),ROOT_directive_ref=directive_ref,
        original_training_input_ref=c.TRAIN_INPUT_REF,original_training_envelope_ref=c.TRAIN_ENVELOPE_REF,
        original_training_job=c.TRAIN_JOB,original_training_output=c.TRAIN_OUTPUT,
        future_original_C3_training_writes_cooperatively_revoked=True,
        future_original_C3_training_resume_retry_or_replay_revoked=True,
        original_files_caps_accounting_and_failed_history_preserved=True,
        original_history=copy.deepcopy(c.ORIGINAL_HISTORY),complete_historical_writer_absence_claimed=False,
        controller_preserved_for_borrowed_native_work=True,controller_shutdown_or_edit_authorized=False,
        new_exception="one exact independently reviewed dedicated saved600 CPU validator only")
    revoke_ref=c.save(packet,"ROOT_original_producer_write_revocation.json",revocation)
    request.update(ROOT_directive_ref=directive_ref,ROOT_original_producer_write_revocation_ref=revoke_ref)
    request_ref=c.save(packet,"request.root_active_v1.json",request)
    config=c.make_config(request,caps)
    config_ref=c.save(packet,"config.json",config)
    req,actual_caps,_=c.validate_request(request_ref)
    require(req==request and actual_caps==caps,"Actual exact reviewed validator request before current-child admission")
    spec=c.read(c.ref(PACKAGE+"/input.INACTIVE.template.v1.json"));spec.pop("INACTIVE",None)
    spec.update(status="active",active=True,root_authorizer="root_agent",execution_authorized=True,
        expires_at_utc=expiry_text,created_at_utc=request["created_at_utc"],
        job_id=job,phase=c.ACCEPT_PHASE,output_directory=output,packet_directory=packet_rel,
        root_stage_request_ref=request_ref,caps_ref=caps_ref,parent_id=c.PARENT,
        source_sha256=copy.deepcopy(manifest["source_sha256"]),
        refs=dict(config=config_ref,runtime=c.RUNTIME_REF,storage=None),
        max_cpu_rss_bytes=caps["max_cpu_rss_bytes"],max_cuda_reserved_bytes=0,
        root_auxiliary_directory=packet_rel+"/worker_aux",completed_overlay_policy_ref=c.POLICY_REF)
    guard=c.original_guard()
    require(guard.MODE=="native14_stopped_retention_v3","Actual current child, never frozen Census mutation")
    require(all(getattr(guard.budget.v4,n).__module__==guard.budget.v4.__name__
        for n in ("canonical","file_bytes","mutex")),"Fresh original unwrapped ROOT mutation role")
    view=guard.inspect_v1(expected_source_sha256=c.CORE,expected_launcher_sha256=c.BASE)
    roles=role_union_v1(c,guard,view,spec,req,[directive_ref,revoke_ref])
    reserve_request=dict(key="H041/"+job,output=str(c.local(output)),
        auxiliary_scopes=[str(c.local(packet_rel+"/worker_aux")),str(c.local(packet_rel+"/worker_tmp"))],
        reserved_bytes=caps["reservation_cap_bytes"],consumer_bindings=roles)
    authority=guard.operation_draft_v1("reserve",reserve_request,expected_before=guard._pins(view),bindings=view["bindings"])
    authority.update(active=True,expires_at_utc=expiry_text)
    authority_ref=c.save(packet,"reserve_authority.json",authority)
    # Any prefix race/uncertain append is terminal. Do not refresh/retry.
    reserved=guard.apply_reserve_v1(authority_ref,expected_source_sha256=c.CORE,expected_launcher_sha256=c.BASE)
    reserve_ref=c.save(packet,"reserve_receipt.json",reserved)
    storage=storage_v1(c,guard,job,output,reserved,expiry_text)
    guard.validate_storage_v9(spec,config,storage,source_ref=c.ref(c.ENTRY),config_ref=config_ref,runtime_ref=c.RUNTIME_REF)
    storage_ref=c.save(packet,"storage.json",storage)
    spec["refs"]["storage"]=storage_ref
    input_ref=c.save(packet,"input.json",spec)
    actual_spec,actual_req,actual_caps=c.validate_input_v1(input_ref)
    require(actual_spec==spec and actual_req==req and actual_caps==caps,"Actual admitted input/source/current-role/cap joins")
    admitted_ref=c.save(packet,"admitted.json",dict(input_ref=input_ref,driver_ref=c.ref(c.DRIVER),
        phase=c.ACCEPT_PHASE,parent_id=c.PARENT,reserve_receipt_ref=reserve_ref,storage=storage,
        config_ref=config_ref,runtime_ref=c.RUNTIME_REF,caps_ref=caps_ref,
        consumer_binding_count=len(roles),execution_started=False,original_history=copy.deepcopy(c.ORIGINAL_HISTORY)))
    require((expiry-datetime.now(timezone.utc)).total_seconds()>caps["operational_deadline_seconds"]+300,
        "Full original900s validator plus terminal reserve must still fit, no expiry reset")
    command=["setsid",str(ROOT/".runtime/env/bin/python"),"-B",str(ROOT/c.DRIVER),
        "--launch","--input",input_ref["path"],"--expected-input-sha256",input_ref["sha256"]]
    return dict(status="ADMITTED_SAVED600_CPU_VALIDATOR_NOT_EXECUTED",input_ref=input_ref,admitted_ref=admitted_ref,
        reserve_receipt_ref=reserve_ref,storage_ref=storage_ref,source_review_ref=review_ref,
        consumer_binding_count=len(roles),reserved_bytes=33554432,
        exact_one_ROOT_validator_execution_argv=command,execution_command_emitted_not_invoked=True,
        original_history=copy.deepcopy(c.ORIGINAL_HISTORY),ROOT_artifact_acceptance_still_required_after_genuine_validator_wait0_drain_outer0=True,
        controller_preserved_for_borrowed_native_work=True,no_automatic_retry=True)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-root",action="store_true")
    parser.add_argument("--review-path");parser.add_argument("--review-sha256")
    parser.add_argument("--instance");parser.add_argument("--expires-at-utc");parser.add_argument("--root-directive")
    args=parser.parse_args(argv)
    if not args.prepare_root:
        print(json.dumps(dict(status="SOURCE_ONLY_INACTIVE",admission=False,execution=False)));return 0
    if not all((args.review_path,args.review_sha256,args.instance,args.expires_at_utc)) or args.root_directive!=DIRECTIVE:
        parser.error("Active ROOT mode requires actual review path/SHA, unique instance, explicit expiry and exact ROOT directive")
    print(json.dumps(prepare_root_v1(args),sort_keys=True),flush=True)
    return 0
if __name__=="__main__":raise SystemExit(main())
