"""Root-only exact sixteen failed-v11 IPC nodes; no archive or process census."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,stat,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/American_C3_native_MLM_amendment_v12"
sys.path[:0]=[str(ROOT/REL),str(ROOT/"research/reviews/American_C3_native_MLM_full_fit_v8"),
    str(ROOT/"research/reviews/American_C3_native_MLM_full_source_v5"),str(ROOT),str(ROOT/"src"),str(ROOT/"scripts")]
import amendment_v12 as a
SOURCE=REL+"/root_namespace_v12.py"
ADMIN_STORAGE_REF=dict(path="research/reviews/American_C3_native_MLM_amendment_v10/root_execution_admin_v2/storage.json",
    sha256="50056bcfc7f1b2fc927951bedd9a3ca9c4965aefc97b32887c1c00ddd8fc9290")
ADMIN_KEY="H041/native-v10-preserve-failed-v8-admin-v2"
ADMIN_EXPIRY="2026-10-16T07:54:34.759077+00:00"
WRITES=ROOT/"research/reviews/American_C3_native_MLM_amendment_v10/root_execution_admin_v2/worker_metadata"
ATTEMPT=WRITES/"v12_namespace_attempt1"
ARCHIVE=ROOT/"research/reviews/American_C3_native_MLM_full_fit_v8/v8_failed_root_archive_v10"
PRESERVED_REFS=[
 dict(path=str((ARCHIVE/"current_writer_proof_before_archive_v10.json").relative_to(ROOT)),
      sha256="cb24ff9cdf166c873ae454cc04299a82ba712e0a07566b6129e8e89adfac005d"),
 dict(path=str((ARCHIVE/"before_rename.json").relative_to(ROOT)),
      sha256="c175eca4f7cdb918490a2553aa5940ab99797b41451c9c2fa854730569707436")]
OLD_V11_OBSERVATION_REF=dict(path="research/reviews/native_MLM_v12_root_preparation_20261015_v1/old_v11_socket_observation.root_v1.json",sha256="ec77101dfee9811b4f34baf6b01b5c0b19efaf3d86b835ee6d6fcd1c2225f9a5")
V11_FRAMED_DIAGNOSIS_REF=dict(path="research/reviews/native_MLM_owned_ESRCH_abstract_IPC_source_v1/diagnosis_evidence.json",sha256="02509df64d511e2d2bc27c8ac23bc050c5dd150f75d59870ceb9419fe44d6fc1")
OLD_V11_TMP=ROOT/".runtime/tmp/c3full8/f/v11a/mlm"
KNOWN_ROOTS=(2889131,2889553,2891969,2893282)
KNOWN_GROUPS=KNOWN_ROOTS
def stamp(s):
    return [s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
def known_roots_groups_absent(*,kill=os.kill,killpg=os.killpg):
    rows=[]
    for kind,ids,probe in (("pid",KNOWN_ROOTS,kill),("process_group",KNOWN_GROUPS,killpg)):
        for value in ids:
            try:probe(value,0)
            except ProcessLookupError:rows.append(dict(kind=kind,id=value,ESRCH=True))
            else:raise ValueError("Known old root/group remains observable")
    return dict(schema="native_MLM_v12_targeted_known_roots_only",checks=rows,
        observed_utc=datetime.now(timezone.utc).isoformat(),historical_ids_complete=False,
        historical_descendant_ids_lost=True,complete_old_writer_absence_claimed=False,
        native_wait_exit=-9,actual_optimizer_updates="UNKNOWN",unrelated_process_metadata_scanned=False)
def validate_directive(record):
    value=a.read(record)
    needed=dict(schema="native_MLM_v12_namespace_root_directive_v1",root_authorizer="root_agent",active=True,
        key=a.KEY,original_storage_ref=a.ORIGINAL_STORAGE_REF,reserved_bytes=a.CAP,expires_at_utc=a.EXPIRES,
        original_parent_scopes_fully_measured=True,all_regular_scientific_files_preserved_in_place=True,
        old_v11_namespace_unlinks_exact_sixteen_recorded_sockets_only=True,
        old_v11_observation_ref=OLD_V11_OBSERVATION_REF,
        allowed_operational_physical_GPUs=[4,5],actual_GPU_selection_requires_root_execution_authority=True,
        current_attempt_dead_IPC_cleanup_after_genuine_owned_drain_authorized=True,known_roots_groups_ESRCH_before_each_unlink=True,
        exact_lstat_and_kernel_no_listener_before_each_unlink=True,
        no_global_unrelated_FD_stability_prerequisite=True,
        all_prior_launch_resume_write_capabilities_cooperatively_revoked=True,
        complete_old_writer_absence_claimed=False,historical_ids_complete=False,
        historical_descendant_ids_lost=True,prior_native_wait_exit=-9,prior_optimizer_updates="UNKNOWN",
        no_new_reservation=True,science_unchanged=True,
        output_child=a.OUTPUT,tmp_child=str(a.TMP.relative_to(ROOT)),aux_child=str(a.ATTEMPT.relative_to(ROOT)),
        parent_completion_envelope_after_actual_wait0_authorized=True,
        original_admin_storage_ref=ADMIN_STORAGE_REF,original_admin_cap_bytes=5242880,
        original_admin_expires_at_utc=ADMIN_EXPIRY)
    a.require(all(value.get(k)==v for k,v in needed.items()),"Exact narrow actual root namespace directive")
    a.check_expiry(dict(created_at_utc=a.CREATED,expires_at_utc=a.EXPIRES))
    return value
def original_socket_rows():
    observation=a.read(OLD_V11_OBSERVATION_REF)
    a.require(observation["schema"]=="native_v12_root_targeted_old_v11_IPC_observation_v1"
        and observation["root_authorizer"]=="root_agent" and observation["exact_kernel_listener_matches"]==[]
        and observation["actual_unlink_performed"] is False
        and observation["complete_historical_writer_absence_claimed"] is False
        and observation["native_optimizer_updates"]=="UNKNOWN"
        and observation["all_regular_files_preserved"] is True,"Exact actual root v11 observation")
    wait=a.read(observation["actual_v11_root_wait_ref"])
    a.require(wait["root_authorizer"]=="root_agent" and wait["actual_direct_Popen_wait"] is True
        and wait["pid"]==2889131 and wait["returncode"]==1,"Genuine v11 root direct wait1")
    required={(kind,pid) for kind in ("pid","process_group") for pid in KNOWN_ROOTS}
    a.require({(r["kind"],r["id"]) for r in observation["known_roots_groups_ESRCH"]}==required
        and all(r["ESRCH"] is True for r in observation["known_roots_groups_ESRCH"]),"Bound eight root ESRCH observations")
    diagnosis=a.read(V11_FRAMED_DIAGNOSIS_REF)
    records={(r["role"],r["slot"]):r for r in diagnosis["framed_wait_records"]}
    a.require(records[("native",0)]["committed"] and records[("native",0)]["value"]["returncode"]==-9
        and records[("outer",1)]["committed"]
        and records[("outer",1)]["value"]["all_owned_processes_and_groups_drained"] is True
        and records[("outer",1)]["value"]["no_adopted_live_writer"] is True,
        "Preserved actual framed v11 failed wait and full owned drain; no optimizer claim")
    rows=observation["old_v11_exact_socket_nodes"]
    a.require(len(rows)==16 and len({r["path"] for r in rows})==16,"Exactly sixteen root-observed v11 sockets")
    for row in rows:
        p=a.local(row["path"]);s=row["lstat"]
        a.require(set(row)=={"path","lstat"} and len(s)==7 and stat.S_ISSOCK(s[2]) and s[3]==1
            and OLD_V11_TMP in p.parents and p.parent.parent==OLD_V11_TMP
            and p.parent.name.startswith("pymp-") and p.name.startswith("listener-"),
            "Only exact recorded v11 multiprocessing socket nodes")
    return rows
def socket_ancestors(path):
    p=a.local(path)
    a.require(a.PARENT_TMP in p.parents,"Only original TMP descendants")
    for parent in [p.parent,*p.parent.parents]:
        s=parent.lstat()
        a.require(stat.S_ISDIR(s.st_mode) and parent.resolve()==parent,"Canonical socket ancestors")
        if parent==a.PARENT_TMP:break
    else:raise ValueError("Original TMP ancestor missing")
def listeners(paths,*,read_text=None):
    if read_text is None:read_text=Path("/proc/net/unix").read_text()
    from terminal_namespace_v12 import kernel_listeners
    return kernel_listeners([str(a.local(path)) for path in paths],read_text)
def validate_cleanup_receipt(record,directive_ref):
    value=a.read(record)
    a.require(value["schema"]=="native_MLM_v12_exact_namespace_cleanup_receipt"
        and value["root_authorizer"]=="root_agent" and value["active"] is True
        and value["helper_ref"]==a.ref(SOURCE) and value["root_directive_ref"]==directive_ref
        and value["original_admin_storage_ref"]==ADMIN_STORAGE_REF
        and value["preserved_metadata_refs"]==PRESERVED_REFS
        and value["old_v11_observation_ref"]==OLD_V11_OBSERVATION_REF
        and value["v11_framed_diagnosis_ref"]==V11_FRAMED_DIAGNOSIS_REF
        and value["removed_socket_metadata"]==original_socket_rows()
        and len(value["per_node_events"])==16 and all(r["actual_unlink"] is True for r in value["per_node_events"])
        and value["all_v11_regular_files_and_waits_preserved"] is True
        and value["cleanup_failure"] is None
        and value["all_regular_scientific_files_preserved_in_place"] is True
        and value["original_archive_two_files_preserved_byte_exact"] is True
        and value["new_reservation_created"] is False and value["global_unrelated_metadata_census_used"] is False
        and value["complete_old_writer_absence_claimed"] is False
        and value["historical_ids_complete"] is False and value["prior_native_wait_exit"]==-9
        and value["prior_optimizer_updates"]=="UNKNOWN","Exact original-node-only root cleanup receipt")
    before=a.read(value["before_cleanup_ref"])
    a.require(before["nodes"]==original_socket_rows()
        and before["authority_ref"]==value["authority_ref"]
        and before["root_directive_ref"]==directive_ref
        and [r["index"] for r in value["per_node_events"]]==list(range(16))
        and [r["node"] for r in value["per_node_events"]]==original_socket_rows(),
        "Persisted pre-unlink manifest and exact actual event batch")
    for row in value["removed_socket_metadata"]:
        p=a.local(row["path"]);a.require(not p.exists() and not p.is_symlink(),"Old exact socket pathname must remain absent")
    return value
def admin_guard():
    import root_native_contract_v1 as c
    m,_memo=c.consumer_guard();storage=a.read(ADMIN_STORAGE_REF)
    a.require(storage["reservation"]["key"]==ADMIN_KEY and storage["reserved_new_bytes"]==5242880
        and storage["expires_at_utc"]==ADMIN_EXPIRY
        and storage["reservation"]["auxiliary_scopes"]==[str(WRITES)]
        and storage["reservation"]["output"]==str(ARCHIVE),"Unchanged existing original5MiB admin charge")
    return m,storage
def admin_save(m,storage,name,value):
    a.require(Path(name).name==name and name.endswith(".json"),"Flat exclusive administrative receipt")
    m.budget_check_v9(storage,boundary=True,required_write_bytes=73728)
    return a.save(ATTEMPT/name,value)
def cleanup(authority_ref):
    auth=a.read(authority_ref);directive_ref=auth["root_directive_ref"]
    a.require(auth["schema"]=="native_MLM_v12_exact_namespace_cleanup_authority"
        and auth["root_authorizer"]=="root_agent" and auth["active"] is True
        and auth["helper_ref"]==a.ref(SOURCE) and auth["original_admin_storage_ref"]==ADMIN_STORAGE_REF
        and auth["preserved_metadata_refs"]==PRESERVED_REFS
        and auth["old_v11_observation_ref"]==OLD_V11_OBSERVATION_REF
        and auth["v11_framed_diagnosis_ref"]==V11_FRAMED_DIAGNOSIS_REF
        and auth["no_new_reservation"] is True and auth["all_prior_capabilities_revoked"] is True,
        "Separately authenticated fresh root exact16socket cleanup authority")
    validate_directive(directive_ref)
    closure=a.read(auth["source_closure_ref"])
    a.require(closure["schema"]=="native_MLM_fullfit_execution_source_closure_v12"
        and closure["original_storage_ref"]==a.ORIGINAL_STORAGE_REF,"Actual new source closure")
    for path,sha in closure["source_sha256"].items():
        a.require(a.ref(path)["sha256"]==sha,"Exact actual source bytes")
    a.require(closure["source_sha256"].get(SOURCE)==a.ref(SOURCE)["sha256"],"Actual helper authenticated")
    rows=original_socket_rows()
    for row in rows:
        p=a.local(row["path"]);socket_ancestors(p)
        a.require(stamp(p.lstat())==row["lstat"],"Actual root-observed node lstat unchanged")
    a.require(not listeners([r["path"] for r in rows]),"No exact old v11 kernel endpoint")
    known_roots_groups_absent()
    a.require(not ATTEMPT.exists(),"One fresh cleanup attempt, never reset/replay")
    m,storage=admin_guard()
    m.budget_check_v9(storage,boundary=True,required_write_bytes=1048576)
    ATTEMPT.mkdir()
    before_ref=a.save(ATTEMPT/"before_cleanup.json",dict(authority_ref=authority_ref,
        root_directive_ref=directive_ref,nodes=rows,old_v11_observation_ref=OLD_V11_OBSERVATION_REF,
        v11_framed_diagnosis_ref=V11_FRAMED_DIAGNOSIS_REF,preserved_metadata_refs=PRESERVED_REFS,
        known_state=known_roots_groups_absent(),all_v11_regular_files_and_waits_preserved=True))
    events=[];failure=None
    try:
        for index,row in enumerate(rows):
            # Only fresh targeted process/group ESRCH, canonical ancestry/lstat and
            # exact kernel endpoint checks. No accounting replay/write per node.
            a.check_expiry()
            state=known_roots_groups_absent()
            p=a.local(row["path"]);socket_ancestors(p)
            a.require(stamp(p.lstat())==row["lstat"] and not listeners([str(p)]),
                "Immediate exact dead-v11-socket identity/endpoint check")
            p.unlink()
            events.append(dict(index=index,node=row,known_state=state,
                exact_kernel_listener_matches=[],actual_unlink=True,
                unlinked_utc=datetime.now(timezone.utc).isoformat()))
        # Original native parent scopes are still fully measured, with no exclusions.
        native=a.read(a.ORIGINAL_STORAGE_REF)
        m.budget_check_v9(native,boundary=True,required_write_bytes=0)
        for record in PRESERVED_REFS:a.read(record)
    except BaseException as error:
        failure=error
    receipt=dict(schema="native_MLM_v12_exact_namespace_cleanup_receipt",
        root_authorizer="root_agent",active=failure is None,helper_ref=a.ref(SOURCE),
        authority_ref=authority_ref,root_directive_ref=directive_ref,
        original_admin_storage_ref=ADMIN_STORAGE_REF,preserved_metadata_refs=PRESERVED_REFS,
        old_v11_observation_ref=OLD_V11_OBSERVATION_REF,v11_framed_diagnosis_ref=V11_FRAMED_DIAGNOSIS_REF,
        before_cleanup_ref=before_ref,removed_socket_metadata=[r["node"] for r in events],
        per_node_events=events,accounting_guard_inside_unlink_loop=False,
        all_regular_scientific_files_preserved_in_place=True,all_v11_regular_files_and_waits_preserved=True,
        original_archive_two_files_preserved_byte_exact=True,new_reservation_created=False,
        global_unrelated_metadata_census_used=False,complete_old_writer_absence_claimed=False,
        historical_ids_complete=False,historical_descendant_ids_lost=True,prior_native_wait_exit=-9,
        actual_v11_native_wait=-9,prior_optimizer_updates="UNKNOWN",
        cleanup_failure=None if failure is None else dict(type=type(failure).__name__,message=str(failure)[:2000]),
        completed_utc=datetime.now(timezone.utc).isoformat())
    # Persist actual completed-node events as one batch, including a partial
    # failure. Only a complete successful batch can authorize future activation.
    m.budget_check_v9(storage,boundary=True,required_write_bytes=73728)
    result=a.save(ATTEMPT/"cleanup_receipt.json",receipt)
    m.budget_check_v9(storage,boundary=True)
    if failure is not None:raise failure
    print(json.dumps(dict(namespace_cleanup_receipt_ref=result)),flush=True)
    return result
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--cleanup",action="store_true")
    p.add_argument("--authority");p.add_argument("--expected-authority-sha256");args=p.parse_args(argv)
    if not args.cleanup:
        print(json.dumps(dict(status="source_only_inert_v12",actual_cleanup=False)));return 0
    a.require(args.authority and args.expected_authority_sha256,"Exact actual root cleanup authority")
    cleanup(dict(path=args.authority,sha256=args.expected_authority_sha256));return 0
if __name__=="__main__":raise SystemExit(main())
