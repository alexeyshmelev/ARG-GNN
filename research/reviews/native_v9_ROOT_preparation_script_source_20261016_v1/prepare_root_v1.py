"""ROOT-only v9 authority preparation. Default is inert. Never launches training."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,sys,time

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE_DIR="research/reviews/native_v9_ROOT_preparation_script_source_20261016_v1"
V9="research/reviews/native_VAL000_full_state_resume_source_20261016_v9"
DIRECTIVE="ROOT authorizes one exact reviewed v9 attempt and permanently revokes old launch and write capabilities"
STRICT_ENTRY="2026-10-16T23:31:32.476768+00:00"

def prepare_root_v1(args):
    # Only ROOT invokes this explicit active branch after inspecting this script.
    assert args.root_directive==DIRECTIVE
    assert Path.cwd()==ROOT and sys.dont_write_bytecode and sys.flags.no_user_site
    assert Path(sys.executable).resolve()==(ROOT/".runtime/env/bin/python").resolve()
    from research.reviews.native_VAL000_full_state_resume_source_20261016_v9 import amendment_contract_v1 as c
    require=c.require
    review_ref=dict(path=args.review_path,sha256=args.review_sha256)
    require(c.ref(args.review_path)==review_ref,"Actual supplied independent review path and SHA")
    review=c.read(review_ref)
    manifest_ref=c.ref(c.MANIFEST)
    require(review.get("verdict")=="PASS_SOURCE_ONLY" and review.get("candidate_manifest_ref")==manifest_ref
        and review.get("blocking_findings")==[],"Actual exact v9 independent source PASS with no blockers")
    manifest=c.read(manifest_ref)
    for record in manifest["new_executed_source_refs"]+manifest["source_authentication_refs"]:c.raw(record)
    template_ref=c.ref(V9+"/ROOT_amendment.INACTIVE.template.v5.json")
    authority=c.read(template_ref)
    require(authority["adapter_manifest_ref"]==manifest_ref and authority["schema"]=="native32_full_state_resume_same_charge_ROOT_amendment_v9",
        "Unchanged reviewed v9 ROOT template and exact source manifest")
    c.validate_v8_failure_lineage_v9()
    old=c.read(c.V8_AMENDMENT_REF)
    require(old["ROOT_resume_qualification_ref"]==authority["ROOT_resume_qualification_ref"]==c.V8_QUALIFICATION_REF
        and old["ROOT_known_owner_observation_bundle_ref"]==authority["ROOT_known_owner_observation_bundle_ref"]==c.V8_KNOWN_BUNDLE_REF,
        "Preserve actual historical v8 refs and their old timestamps and hashes")
    for key in ("original_caps","original48h_window_anchor_utc","expires_at_utc","original_input_ref",
        "original_storage_ref","original_root_request_ref","original_author_scientific_config_ref",
        "resume_checkpoint_ref","preserved_actual_v6_amendment_ref","prior_v6_preparation_failure_facts"):
        require(authority[key]==old[key],"Unchanged original runtime/science/resource/history field "+key)
    extra=[json.loads(v) for v in args.additional_known_source_ref]
    require(len(extra)<=64,"At most64 explicit old ROOT launcher metadata refs")
    for record in extra:c.read(record)
    authority["additional_known_old_ROOT_launcher_source_refs"]=extra
    original_spec,q,cfg,storage,_=c.original.validate_input_v1(c.ORIGINAL_INPUT)
    require(storage["reserved_new_bytes"]==7*2**30 and q["caps"]==c.CAPS
        and authority["original_caps"]==c.CAPS,"Original complete7GiB charge and unchanged caps")
    aux=c.local(original_spec["worker_auxiliary_scope"])
    names=("ROOT_full_state_resume_v9_ROOT_directive.root_v1.json",
        "ROOT_full_state_resume_v9_permanent_old_revocation.root_v1.json",
        "ROOT_same_charge_full_state_resume_v9.root_active_v1.json")
    require(all(not (aux/name).exists() and not (aux/name).is_symlink() for name in names),"One fresh preparation only, no replacement")
    require(not c.local(authority["fresh_science_directory"]).exists()
        and not c.local(authority["fresh_admin_directory"]).exists(),"Fresh exact v9 child directories, created later by reviewed executor")
    def tick():
        now=time.time()
        anchored=datetime.fromisoformat(authority["original48h_window_anchor_utc"]).timestamp()+172800
        end=min(anchored,datetime.fromisoformat(authority["expires_at_utc"]).timestamp())-c.CAPS["work_cutoff_margin_seconds"]
        require(now<min(end,datetime.fromisoformat(STRICT_ENTRY).timestamp()),"Original anchored work and strict entry cutoff")
    tick()
    g=c.original.guard_v1(q,consumer=True)
    g.validate_storage_v9(original_spec,cfg,storage,source_ref=c.original.ref(c.original.DRIVER),
        config_ref=original_spec["refs"]["config"],runtime_ref=original_spec["refs"]["runtime"])
    def save(name,value):
        raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
        require(len(raw)<=65536,"Compact immutable ROOT preparation metadata")
        tick()
        g.budget_check_v9(storage,boundary=True,required_write_bytes=len(raw)+8192)
        tick()
        record=c.original.save(aux,name,value)
        fd=os.open(aux,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
        return record
    now=datetime.now(timezone.utc).isoformat()
    directive=dict(schema="ROOT_native_VAL000_one_exact_v9_directive_v1",root_authorizer="root_agent",
        status="ROOT_DIRECTIVE",template_only=False,actual_recorded_utc=now,directive=args.root_directive,
        source_template_ref=template_ref,source_manifest_ref=manifest_ref,independent_source_review_ref=review_ref,
        preserved_actual_v8_amendment_ref=c.V8_AMENDMENT_REF,genuine_v8_failed_terminal_ref=c.V8_TERMINAL_REF,
        original_input_ref=c.ORIGINAL_INPUT,original48h_window_anchor_utc=authority["original48h_window_anchor_utc"],
        strict_entry_cutoff_utc=STRICT_ENTRY,expires_at_utc=authority["expires_at_utc"],
        old_ROOT_rows_are_historical_not_current_entry_evidence=True,new_actual_entry_observations_mandatory=True,
        all_prior_charges_and_artifacts_retained=True,one_exact_descendant_attempt_authorized=True,
        deadline_reset_or_automatic_retry_authorized=False,new_reservation_or_close_or_credit_authorized=False)
    directive_ref=save(names[0],directive)
    bundle=c.read(c.V8_KNOWN_BUNDLE_REF)
    old_revocation_ref=bundle["permanent_cooperative_revocation_ref"]
    revoked=c.read(old_revocation_ref)["actual_cooperative_capability_refs"]+[c.V8_AMENDMENT_REF]
    unique=[]
    for record in revoked:
        c.raw(record)
        if record not in unique:unique.append(record)
    revocation=dict(schema="ROOT_native_VAL000_permanent_cooperative_old_capability_revocation_v9",
        root_authorizer="root_agent",status="permanently_cooperatively_revoked",template_only=False,
        actual_recorded_utc=datetime.now(timezone.utc).isoformat(),ROOT_directive_ref=directive_ref,
        source_manifest_ref=manifest_ref,independent_source_review_ref=review_ref,
        preserved_historical_revocation_ref=old_revocation_ref,actual_cooperative_capability_refs=unique,
        all_source_referenced_old_launch_capabilities_permanently_revoked=True,
        old_original_resume_entrypoints_and_automatic_retries_revoked=True,
        v8_future_launch_write_capabilities_permanently_revoked=True,new_exception_is_exact_reviewed_v9_ROOT_amendment_only=True,
        historical_files_or_failures_mutated=False,unknown_old_processes_absent_asserted=False)
    revocation_ref=save(names[1],revocation)
    authority.update(status="active",template_only=False,execution_authorized=True,
        created_at_utc=datetime.now(timezone.utc).isoformat(),independent_source_review_passed=True,
        independent_source_review_ref=review_ref,no_ROOT_execution_authority_in_this_source_packet=False,
        ROOT_directive_ref=directive_ref,ROOT_v9_permanent_cooperative_revocation_ref=revocation_ref)
    amendment_ref=save(names[2],authority)
    c.set_amendment_v1(amendment_ref)
    spec,_,_,_,_=c.validate_input_v1(c.ORIGINAL_INPUT)
    c.charged_census_v1(spec)
    registry=c.entry_known_registry_v9()
    tick()
    return dict(status="ROOT_V9_AUTHORITY_PREPARED_AND_VALIDATED_NOT_LAUNCHED",amendment_ref=amendment_ref,
        input_ref=c.ORIGINAL_INPUT,ROOT_directive_ref=directive_ref,cooperative_revocation_ref=revocation_ref,
        source_review_ref=review_ref,known_PID_rows=len(registry["identities"]),known_groups=len(registry["process_groups"]),
        new_entry_absence_established=False,preliminary_GPU_capacity_sample_performed=False,
        execution_still_requires_actual_guard_GPU0_capacity_lease_and_separate_fresh_supervisor_worker_entry=True,
        launch_performed=False,new_training_updates_verified=0,no_automatic_retry=True)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-root-v9",action="store_true")
    parser.add_argument("--review-path")
    parser.add_argument("--review-sha256")
    parser.add_argument("--root-directive")
    parser.add_argument("--additional-known-source-ref",action="append",default=[],help="Exact JSON path+SHA of explicit old ROOT launcher metadata")
    args=parser.parse_args(argv)
    if not args.prepare_root_v9:
        print(json.dumps(dict(status="SOURCE_ONLY_INACTIVE",authority_written=False,execution=False)))
        return 0
    if not args.review_path or not args.review_sha256 or args.root_directive!=DIRECTIVE:
        parser.error("ROOT active mode requires actual review path+SHA and exact --root-directive shown in RECIPE.md")
    print(json.dumps(prepare_root_v1(args),sort_keys=True),flush=True)
    return 0

if __name__=="__main__":raise SystemExit(main())
