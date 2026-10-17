"""Inactive ROOT owner4 admission builder; exact original Census API, no launch."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,re,sys
from . import empty_gpu0_controller_v1 as o
SOURCE=o.HERE+"/owner4_admission_ROOT_v1.py"
GUARD=dict(path="research/reviews/Census_observation_two_scope_retention_source_20261017_v1/census_observation_retention_v1.py",sha256="72f6dc51844b2a261b581f5e69bd2a22982e6244c059f24549747a7b851faeeb")
POLICY=dict(path="research/reviews/Census_observation_ROOT_activation_20261017_v1/activation_policy.root_active_v1.json",sha256="5e82c0895c43b77d7e69d440fa9b09dbe19f7f2565b3ae9f866d0075e6f9fad5")
SELECTOR=dict(path="research/reviews/Census_observation_ROOT_activation_20261017_v1/explicit_child_selector.root_active_v1.json",sha256="fd05d8934ed29eed39f7e60121a33e48fbeb0106991e76690664ca96355692e4")
COORDINATOR=dict(path="research/reviews/frozen_TRAIN002_019_batch_continuation_source_20261017_v3/worker_entry_v1.py",sha256="2c4a976949fe147238ba775fa6aee58bbb782a9cef2c76d65945a472eae6be73")
ADMIN_CAP=256*1024
PARENT_CAP=128*1024
def save(directory,name,value):
    raw=o.encoded(value)+b"\n";rows=list(directory.iterdir())
    o.need(len(raw)<=65536 and len(rows)<20 and all(p.is_file()and not p.is_symlink()for p in rows)
        and 4096+sum(p.stat().st_size+4096 for p in rows)+len(raw)+4096<=ADMIN_CAP
        and ADMIN_CAP+PARENT_CAP+65536+4096<=524288,"Bounded administrative/parent records inside original512KiB event allowance")
    with(directory/name).open("xb")as f:o.need(f.write(raw)==len(raw),"Complete exclusive metadata");f.flush();os.fsync(f.fileno())
    return o.ref(str((directory/name).relative_to(o.ROOT)))
def prepare_v4(record):
    q=o.read(record);o.active(q)
    o.need(q["schema"]=="ROOT_empty_GPU0_owner4_admission_request_v1"
        and q["source_manifest_ref"]==o.ref(o.MANIFEST)and q["storage_policy_selector_ref"]==SELECTOR
        and q["accounting_coordinator_source_ref"]==COORDINATOR
        and q["all_previous_owner_and_borrower_routes_revoked"]is True
        and q["old_files_charges_failures_and_clocks_preserved"]is True
        and q["automatic_retry_or_launch"]is False
        and type(q["instance"])is str and re.fullmatch("[a-z][a-z0-9_]{0,39}",q["instance"])is not None,
        "Finite explicit fresh owner4 ROOT admission only")
    o.need(type(q["pool"])is str and 1<=len(q["pool"])<=32
        and all(ch in "abcdefghijklmnopqrstuvwxyz0123456789_-"for ch in q["pool"]),"Bounded declared empty pool name")
    now=o.stamp(q["created_at_utc"]);end=o.stamp(q["expires_at_utc"])
    o.need(now<=__import__("time").time()<q["owner_expires_unix"]
        and q["owner_expires_unix"]+35<=q["drain_expires_unix"]<=end
        and 0<end-now<=7*86400,"Finite distinct owner work/drain window, no old clock extension")
    m=o.read(q["source_manifest_ref"]);review=o.read(q["source_review_ref"])
    o.need(review["independent"]is True and review["verdict"]=="SOURCE_PASS"and review["blockers"]==[]
        and review["candidate_manifest_ref"]==q["source_manifest_ref"],"Exact independent owner4 whole-source pass")
    for path,sha in m["source_sha256"].items():o.raw(dict(path=path,sha256=sha))
    h=o.read(q["handover_ref"]);o.active(h);cert=o.read(h["known_root_absence_ref"])
    o.need(h["old_controller_capability_ref"]==o.PREDECESSOR_CAP and h["old_controller_identity"]==o.PREDECESSOR_ID
        and h["predecessor_owner_capability_ref"]==o.PREDECESSOR_CAP and h["predecessor_owner_identity"]==o.PREDECESSOR_ID
        and h["predecessor_termination_outcome_preserved"]is True and h["predecessor_actual_outer_exit_code"]==1
        and h["failed_TreeLoom_qualification_input_ref"]==o.FAILED_QUALIFICATION
        and h["failed_TreeLoom_qualification_namespace_replay_authorized"]is False
        and cert["all_listed_known_roots_absent"]is True and o.PREDECESSOR_ID in cert["known_identities"]
        and max(now,o.stamp(h["created_at_utc"]))<=o.stamp(cert["observed_at_utc"])<=__import__("time").time(),
        "Actual controlled predecessor/failed qualification handover and fresh targeted roster metadata only")
    for k in("predecessor_owner_actual_wait_ref","predecessor_owner_actual_outer_ref","failed_TreeLoom_qualification_actual_outer_ref"):o.raw(h[k])
    instance=q["instance"];admin=o.local(o.HERE+"/ROOT_admission_records_"+instance)
    parent_rel="research/reviews/native_savedMLM_owner_ROOT_actual_oct17_v4_"+instance
    parent=o.local(parent_rel);output="experiments/results/GPU0_savedMLM_empty_owner_oct17_v4_"+instance
    o.need(not admin.exists()and not parent.exists()and not o.local(output).exists(),"One-shot absent control/parent/output namespaces")
    admin.mkdir(mode=0o700)
    save(admin,"attempt.json",dict(request_ref=record,automatic_retry=False,execution_started=False))
    config=o.read(o.ref(o.HERE+"/ROOT_owner_config.FROZEN.v4.json"))
    runtime=o.read(o.ref(o.HERE+"/ROOT_owner_runtime.FROZEN.v4.json"))
    config_ref=save(admin,"config.json",config);runtime_ref=save(admin,"runtime.json",runtime)
    plan=o.read(o.ref(o.HERE+"/ROOT_owner_plan.INACTIVE.v4.json"))
    plan.update(root_authorizer="root_agent",status="active",template_only=False,execution_authorized=True,
        created_at_utc=q["created_at_utc"],expires_at_utc=q["expires_at_utc"],
        expires_unix=q["owner_expires_unix"],drain_expires_unix=q["drain_expires_unix"],
        source_manifest_ref=q["source_manifest_ref"],source_review_ref=q["source_review_ref"],
        source_refs={o.SOURCE:o.ref(o.SOURCE),o.PROTOCOL_REF["path"]:o.PROTOCOL_REF},
        ROOT_record_directory=parent_rel,ROOT_downstream_release_path=parent_rel+"/ROOT_downstream_release.root_active_v1.json",
        controller_job_id="GPU0-savedMLM-empty-owner-oct17-v4-"+instance,output_directory=output,pool=q["pool"],
        capacity_guard_ref=GUARD,capacity_policy_ref=POLICY,storage_policy_selector_ref=SELECTOR,
        config_ref=config_ref,runtime_ref=runtime_ref,handover_ref=q["handover_ref"],
        marker_repair_renewal_ref=q["marker_repair_renewal_ref"],human_GPU0_policy_ref=q["human_GPU0_policy_ref"])
    plan_ref=save(admin,"plan.json",plan)
    coordinator=o.load(COORDINATOR,"_owner4_ROOT_original_v3_coordinator")
    entered=False
    try:
        with coordinator.accounting_transaction(q["accounting_batch_input_ref"]):
            entered=True;o.active(q)
            g=o.load(GUARD,"_owner4_ROOT_current_Census_mutation")
            o.need(g.load_selected_guard_v1(SELECTOR)is g and g.MODE=="Census_observation_two_scope_retention_v1",
                "Exact existing selected Census mutation route, no memo")
            policy=o.read(POLICY);implementation=policy["implementation_ref"];launcher=policy["launcher_ref"]
            reservation=dict(key="H056/"+plan["controller_job_id"],output=str(o.local(output)),auxiliary_scopes=[],reserved_bytes=32*2**20)
            direct=[GUARD,o.ref(o.SOURCE),o.ref(o.HERE+"/owner_parent_v1.py"),o.PROTOCOL_REF,
                q["source_manifest_ref"],config_ref,runtime_ref,record,q["source_review_ref"],q["handover_ref"],plan_ref,
                COORDINATOR,q["accounting_batch_input_ref"]]
            with g.budget.v4.mutex(o.ROOT):
                state=g._state_locked(implementation["sha256"],launcher["sha256"],fresh=True)
                reservation["consumer_bindings"]=g.consumer_bindings_v1(list(state[3].values()),direct)
                g.budget.v4.scope_spec(o.ROOT,reservation)
                authority=g.operation_draft_v1("reserve",reservation,expected_before=g._pins(state[2]),bindings=state[3])
                authority.update(active=True,expires_at_utc=q["expires_at_utc"])
                authority_ref=save(admin,"reserve_authority.json",authority)
                event=dict(schema="Census_observation_exact_retention_event_v1",sequence=len(state[4])+1,
                    previous_sha256=state[5][-1]if state[5]else None,bindings=state[3],operation="reserve",
                    authority=authority_ref,value=dict(reservation,root_admission=authority_ref))
                o.need(len(g.budget.v4.encoded(event)+b"\n")<=65536,"Original event64KiB ceiling unchanged")
            receipt=g.apply_reserve_v1(authority_ref,expected_source_sha256=implementation["sha256"],expected_launcher_sha256=launcher["sha256"])
            receipt_ref=save(admin,"reserve_receipt.json",receipt)
    except ValueError as exc:
        if not entered and str(exc)=="Finite shared ROOT accounting coordinator wait":
            r=save(admin,"ready_no_admission.json",dict(status="READY_NO_ADMISSION_COORDINATOR_TIMEOUT",
                admission_started=False,automatic_retry=False))
            return dict(status="READY_NO_ADMISSION_COORDINATOR_TIMEOUT",record_ref=r)
        raise
    o.need(receipt["value"]==dict(reservation,root_admission=authority_ref),"Original exact reserve return")
    storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,accounting_mode=g.MODE,
        job_id=plan["controller_job_id"],output_directory=output,reserved_new_bytes=32*2**20,reservation=receipt["value"],
        created_at_utc=q["created_at_utc"],expires_at_utc=q["expires_at_utc"],
        budget_components=dict(implementation=implementation,launcher=launcher,consumer_adapter=GUARD),
        completed_overlay_policy_ref=POLICY,storage_policy_selector_ref=SELECTOR,initial_ledger_pins=receipt["initial_ledger_pins"])
    storage_ref=save(admin,"storage.json",storage)
    cap=dict(plan,schema="ROOT_empty_GPU0_owner_capability_v1",plan_ref=plan_ref,controller_storage_ref=storage_ref)
    cap_ref=save(admin,"capability.json",cap)
    o.validate_cap_v1(cap_ref) # Full metadata/source contract only; never guard_v1 or resource operations.
    o.need(not parent.exists()and not o.local(output).exists(),"Owner scientific output absent; fresh parent records only")
    parent.mkdir(mode=0o700)
    argv=[".runtime/env/bin/python","-B","-s","-m",o.HERE.replace("/",".")+".owner_parent_v1",
        "--execute","--capability-ref",json.dumps(cap_ref,sort_keys=True)]
    result=dict(status="OWNER4_ADMITTED_ONLY",capability_ref=cap_ref,reserve_receipt_ref=receipt_ref,
        execute_argv=argv,execute_environment={"PYTHONPATH":".:src:scripts"},
        consumer_binding_count=len(receipt["value"]["consumer_bindings"]),no_automatic_launch=True,
        genuine_builder_outer0_required_before_single_parent_launch=True)
    save(admin,"admitted.json",result);return result
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--prepare-ref");a=p.parse_args(argv)
    if not a.prepare_ref:print(json.dumps(dict(status="SOURCE_ONLY_INACTIVE",admission=False,execution=False)));return 0
    o.need(Path.cwd()==o.ROOT and sys.dont_write_bytecode and sys.flags.no_user_site and sys.flags.optimize==0
        and not any(n in sys.modules for n in("torch","torch_geometric")),"Fresh ROOT -B -s stdlib mutation interpreter")
    print(json.dumps(prepare_v4(json.loads(a.prepare_ref)),sort_keys=True),flush=True);return 0
if __name__=="__main__":raise SystemExit(main())
