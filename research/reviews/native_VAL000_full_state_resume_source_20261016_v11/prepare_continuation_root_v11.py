"""ROOT-only fresh fiveGiB admission and one execution amendment. Default inert; never launches."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,copy,hashlib,importlib.util,json,re,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/native_VAL000_full_state_resume_source_20261016_v11"
DIRECTIVE="ROOT authorizes one exact verified22000 native continuation in a separately charged fresh5GiB envelope with explicitly selected accounting failstop and historical terminal-controller state and permanently revoked old submissions"
def require(ok,message):
    if not ok:raise ValueError(message)
def ref(path):
    p=ROOT/path
    require(not Path(path).is_absolute() and ".." not in p.parts and p.resolve()==p and p.is_file()
        and not p.is_symlink() and p.stat().st_size<=32*2**20 and p.suffix in (".py",".json",".md"),
        "Canonical bounded source/metadata only")
    return dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(record):
    require(ref(record["path"])==record,"Exact existing metadata ref")
    return json.loads((ROOT/record["path"]).read_text())
def load_contract():
    sp=importlib.util.spec_from_file_location("_native11_actual_ROOT_preparation_contract",ROOT/HERE/"amendment_contract_v1.py")
    c=importlib.util.module_from_spec(sp);sys.modules[sp.name]=c;sp.loader.exec_module(c);return c
def review_v11(c,record):
    review=read(record)
    candidate=review.get("candidate_manifest_ref",review.get("candidate_source_manifest_ref",review.get("source_manifest_ref")))
    require(review.get("verdict")=="PASS_SOURCE_ONLY" and candidate==ref(c.MANIFEST),
        "Actual independent final package PASS_SOURCE_ONLY, component review alone insufficient")
    return review
def role_union_v11(c,guard,view,request_ref,request,closure_ref):
    returned=list(view["bindings"].values());needed=c.fresh_required_roles_v11(request_ref,request,closure_ref)
    union=guard.consumer_bindings_v1(returned,needed)
    require(1<=len(union)<=64 and all(r in union for r in [*returned,*needed]),"All actual returned and needed roles under64")
    bypath={}
    for r in union:
        require(r["path"] not in bypath or bypath[r["path"]]==r,"No role hash conflict")
        bypath[r["path"]]=r
    return union
def save_root_v11(c,packet,name,value):
    data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    require(len(data)+8192<=512*1024,"One bounded ROOT admission receipt")
    existing=sum(p.lstat().st_size+8192 for p in packet.iterdir() if p.is_file())
    require(existing+len(data)+8192<=512*1024,"All ROOT packet metadata within actual current-child event allowance")
    return c.save(packet,name,value)
def prepare_v11(args):
    require(args.root_directive==DIRECTIVE and Path.cwd()==ROOT and sys.dont_write_bytecode and sys.flags.no_user_site,
        "Explicit ROOT directive and fresh -B/-s metadata interpreter")
    require(Path(sys.executable).resolve()==(ROOT/".runtime/env/bin/python").resolve()
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,47}",args.instance) is not None,"Existing runtime and fresh bounded instance")
    c=load_contract();selected=getattr(args,"reservation_bytes",5*2**30)
    require(type(selected) is int and selected in c.ACCOUNTING_CHOICES_V11,"Only explicit5GiB or17/4GiB reservation choices")
    review_ref=dict(path=args.review_path,sha256=args.review_sha256);review_v11(c,review_ref)
    manifest=read(ref(c.MANIFEST))
    require(manifest["source_only"] is True and manifest["new_executed_source_refs"]==
        [ref(HERE+"/"+n) for n in ("__init__.py","amendment_contract_v1.py","direct_gpu0_v1.py","execution_only_v1.py",
        "corpus_metadata_bridge_v1.py","resume_scientific_bridge_v5.py","alias_namespace_v7.py","fresh_entry_v9.py")],
        "Exact independently reviewed final source set")
    component_review_ref=dict(path=args.component_review_path,sha256=args.component_review_sha256)
    component_review=read(component_review_ref)
    require(component_review.get("verdict") in ("PASS","SOURCE_PASS","PASS_SOURCE_ONLY")
        and component_review.get("candidate_manifest_ref",component_review.get("source_manifest_ref"))==c.LIVENESS_COMPONENT_MANIFEST_REF,
        "Independent sealed narrow component review also required")
    old=read(c.ORIGINAL_INPUT);oldq=read(old["root_stage_request_ref"]);oldstorage=read(old["refs"]["storage"])
    require(oldstorage["reserved_new_bytes"]==7*2**30 and old["expires_at_utc"]==c.original.ORIGINAL_ABSOLUTE_EXPIRY,
        "Preserved original7GiB admission and absolute deadline")
    audit=read(c.V11_SAVED22000_AUDIT_REF);outer=read(c.V10_OUTER_TERMINAL_REF)
    require(audit["checkpoint_ref"]==c.V11_RESUME_CHECKPOINT_REF and audit["actual_saved_optimizer_updates"]==22000
        and outer["actual_terminal_consumed"] is True and outer["actual_tool_session_id"]==10747
        and outer["actual_exit_code"]==241 and outer["actual_direct_worker_returncode"]==-15,
        "Actual exact22000 verified saved progress and separate consumed10747 failure")
    now=datetime.now(timezone.utc);end=datetime.fromisoformat(old["expires_at_utc"])
    require(now.timestamp()+600<min(end.timestamp(),datetime.fromisoformat(oldq["created_at_utc"]).timestamp()+172800,
        datetime.fromisoformat("2026-10-16T23:31:32.476768+00:00").timestamp()),"Original work/entry deadlines and preparation margin; no reset")
    packet_rel="research/reviews/native_VAL000_fresh_continuation_ROOT_"+args.instance
    output="experiments/results/native_VAL000_verified22000_continuation_"+args.instance
    packet=c.local(packet_rel)
    require(not packet.exists() and not c.local(output).exists(),"Fresh exclusive ROOT packet/output; no replay")
    guard=c.fresh_guard_runtime_v11(consumer=False)
    view=guard.inspect_v1(expected_source_sha256=c.CORE,expected_launcher_sha256=c.BASE)
    # The exact current global guard decides actual capacity. No future close credit is assumed.
    require(view["projected_bytes"]+selected+2*1048576<980000000000,
        "Actual current projection plus selected full reservation and bounded accounting/admin allowance must fit980GB")
    packet.mkdir();created=now.isoformat();expiry=end.isoformat()
    directive=save_root_v11(c,packet,"ROOT_directive.json",dict(schema="ROOT_native11_verified22000_fresh_scope_directive_v1",
        root_authorizer="root_agent",template_only=False,status="ROOT_DIRECTIVE",directive=DIRECTIVE,
        source_manifest_ref=ref(c.MANIFEST),independent_source_review_ref=review_ref,
        component_review_ref=component_review_ref,controller_capability_ref=c.CONTROLLER_CAPABILITY_REF,
        controller_pid=3349405,controller_start_ticks=481281638,
        terminal_failed_C3_historical_snapshot_authorized=True,all_old_controller_submissions_permanently_revoked=True,
        v10_future_launch_write_capabilities_permanently_revoked=True,fresh5GiB_continuation_execution_authorized=True,
        fresh_accounting_reservation_bytes=selected,reservation_only_downward_failstop_authorized=selected<5*2**30,
        original7GiB_scope_is_retained_readonly=True,original_input_ref=c.ORIGINAL_INPUT,
        checkpoint_audit_ref=c.V11_SAVED22000_AUDIT_REF,resume_checkpoint_ref=c.V11_RESUME_CHECKPOINT_REF,
        no_checkpoint_payload_copy_move_delete_or_alias_change=True,all_old_charges_and_failed_artifacts_retained=True,
        original48h_window_anchor_utc=oldq["created_at_utc"],expires_at_utc=expiry,created_at_utc=created))
    request=dict(schema="native_VAL000_verified22000_fresh5GiB_ROOT_request_v11",root_authorizer="root_agent",
        execution_authorized=True,created_at_utc=created,expires_at_utc=expiry,phase="mlm",
        job_id="native-VAL000-verified22000-continuation-"+args.instance,packet_directory=packet_rel,output_directory=output,
        source_manifest_ref=ref(c.MANIFEST),original_input_ref=c.ORIGINAL_INPUT,checkpoint_audit_ref=c.V11_SAVED22000_AUDIT_REF,
        resume_checkpoint_ref=c.V11_RESUME_CHECKPOINT_REF,config_ref=old["refs"]["config"],runtime_ref=old["refs"]["runtime"],
        ROOT_directive_ref=directive,independent_source_review_ref=review_ref,direct_attachment_authority_ref=oldq["direct_attachment_authority_ref"],
        original_caps=copy.deepcopy(c.ORIGINAL_CAPS),fresh_caps=copy.deepcopy(c.CAPS),
        accounting_reservation_bytes=selected,reservation_only_downward_failstop_authorized=selected<5*2**30)
    request_ref=save_root_v11(c,packet,"request.json",request)
    closure=save_root_v11(c,packet,"consumer_manifest.json",dict(source_manifest_ref=ref(c.MANIFEST),root_request_ref=request_ref,
        original_input_ref=c.ORIGINAL_INPUT,checkpoint_audit_ref=c.V11_SAVED22000_AUDIT_REF,
        config_ref=request["config_ref"],runtime_ref=request["runtime_ref"]))
    roles=role_union_v11(c,guard,view,request_ref,request,closure)
    reserve=dict(key="H041/"+request["job_id"],output=str(c.local(output)),
        auxiliary_scopes=[str(c.local(packet_rel+"/worker_aux"))],reserved_bytes=selected,consumer_bindings=roles)
    authority=guard.operation_draft_v1("reserve",reserve,expected_before=guard._pins(view),bindings=view["bindings"])
    authority.update(active=True,expires_at_utc=expiry)
    ar=save_root_v11(c,packet,"reserve_authority.json",authority)
    rr=guard.apply_reserve_v1(ar,expected_source_sha256=c.CORE,expected_launcher_sha256=c.BASE)
    receipt=save_root_v11(c,packet,"reserve_receipt.json",rr)
    storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,
        accounting_mode=guard.MODE,job_id=request["job_id"],output_directory=output,reserved_new_bytes=selected,
        reservation=copy.deepcopy(rr["value"]),initial_ledger_pins=copy.deepcopy(rr["initial_ledger_pins"]),
        expires_at_utc=expiry,completed_overlay_policy_ref=c.FRESH_POLICY_REF,
        budget_components=dict(implementation=c.ref(guard.budget.SOURCE_REL),launcher=c.ref(guard.budget.LAUNCHER_REL),consumer_adapter=c.FRESH_GUARD_REF))
    storage_ref=save_root_v11(c,packet,"storage.json",storage)
    spec=dict(schema="native_VAL000_fresh5GiB_continuation_input_v11",root_authorizer="root_agent",
        execution_authorized=True,GPU_authorized=True,status="active",phase="mlm",job_id=request["job_id"],
        created_at_utc=created,expires_at_utc=expiry,root_stage_request_ref=request_ref,output_directory=output,
        worker_auxiliary_scope=packet_rel+"/worker_aux",refs=dict(config=request["config_ref"],runtime=request["runtime_ref"],storage=storage_ref),
        source_manifest_ref=ref(c.MANIFEST),consumer_manifest_ref=closure,parent_id=c.PARENT)
    input_ref=save_root_v11(c,packet,"input.json",spec)
    a=read(ref(HERE+"/ROOT_amendment.INACTIVE.template.v11.json"))
    a.update(schema="native32_verified22000_fresh5GiB_ROOT_amendment_v11",root_authorizer="root_agent",status="active",
        execution_authorized=True,template_only=False,no_ROOT_execution_authority_in_this_source_packet=False,
        independent_source_review_passed=True,independent_source_review_ref=review_ref,adapter_manifest_ref=ref(c.MANIFEST),
        ROOT_directive_ref=directive,created_at_utc=created,expires_at_utc=expiry,
        fresh_continuation_input_ref=input_ref,fresh_accounting_reservation_bytes=selected,
        reservation_only_downward_failstop_authorized=selected<5*2**30,fresh_science_directory=output,fresh_admin_directory=packet_rel+"/worker_aux")
    amendment_ref=save_root_v11(c,packet,"amendment.root_active_v11.json",a)
    c.set_amendment_v1(amendment_ref)
    actual_spec,actual_q,actual_cfg,actual_storage,_=c.validate_input_v1(c.ORIGINAL_INPUT)
    require(actual_storage==storage and actual_spec["output_directory"]==output
        and c.RESUME_STATE["actual_saved_progress"]["batch"]==22000,"Actual fresh input/storage and exact current22000 before execution")
    g=c.guard_v1(actual_q)
    g.validate_storage_v9(actual_spec,dict(output_cap_bytes=5*2**30),storage,source_ref=c.ref(c.ACTUAL_DRIVER),
        config_ref=request["config_ref"],runtime_ref=request["runtime_ref"])
    admitted=save_root_v11(c,packet,"admitted.json",dict(input_ref=input_ref,amendment_ref=amendment_ref,
        reserve_receipt_ref=receipt,storage_ref=storage_ref,consumer_binding_count=len(roles),execution_started=False,
        original7GiB_scope_retained=True,fresh_reserved_bytes=selected,operational_envelope_bytes=5*2**30,downward_fit_proven=False,verified_start_updates=22000))
    argv=["setsid",str(ROOT/".runtime/env/bin/python"),"-B","-s","-m",HERE.replace("/",".")+".execution_only_v1",
        "--execute","--input-ref",json.dumps(c.ORIGINAL_INPUT,separators=(",",":")),
        "--amendment-ref",json.dumps(amendment_ref,separators=(",",":"))]
    return dict(status="FRESH_NATIVE_CONTINUATION_ADMITTED_NOT_EXECUTED",input_ref=input_ref,amendment_ref=amendment_ref,
        admitted_ref=admitted,exact_one_ROOT_execution_argv=argv,execution_command_emitted_not_invoked=True,
        no_automatic_retry=True,consumer_binding_count=len(roles),reserved_bytes=selected,operational_envelope_bytes=5*2**30,fit_proven=False)
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--prepare-root",action="store_true")
    p.add_argument("--reservation-bytes",type=int,choices=(5*2**30,17*2**28),default=5*2**30)
    for n in ("review-path","review-sha256","component-review-path","component-review-sha256","instance","root-directive"):p.add_argument("--"+n)
    a=p.parse_args(argv)
    if not a.prepare_root:
        print(json.dumps(dict(status="SOURCE_ONLY_INACTIVE",admission=False,execution=False)));return 0
    require(all(getattr(a,n) for n in ("review_path","review_sha256","component_review_path","component_review_sha256","instance"))
        and a.root_directive==DIRECTIVE,"Explicit actual independent reviews, instance and ROOT directive")
    print(json.dumps(prepare_v11(a),sort_keys=True),flush=True);return 0
if __name__=="__main__":raise SystemExit(main())
