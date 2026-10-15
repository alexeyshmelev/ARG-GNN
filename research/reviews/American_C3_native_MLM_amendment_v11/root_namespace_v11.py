"""Root-only exact eight old IPC nodes; no archive or process census."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,stat,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/American_C3_native_MLM_amendment_v11"
sys.path[:0]=[str(ROOT/REL),str(ROOT/"research/reviews/American_C3_native_MLM_full_fit_v8"),
    str(ROOT/"research/reviews/American_C3_native_MLM_full_source_v5"),str(ROOT),str(ROOT/"src"),str(ROOT/"scripts")]
import amendment_v11 as a
SOURCE=REL+"/root_namespace_v11.py"
ADMIN_STORAGE_REF=dict(path="research/reviews/American_C3_native_MLM_amendment_v10/root_execution_admin_v2/storage.json",
    sha256="50056bcfc7f1b2fc927951bedd9a3ca9c4965aefc97b32887c1c00ddd8fc9290")
ADMIN_KEY="H041/native-v10-preserve-failed-v8-admin-v2"
ADMIN_EXPIRY="2026-10-16T07:54:34.759077+00:00"
WRITES=ROOT/"research/reviews/American_C3_native_MLM_amendment_v10/root_execution_admin_v2/worker_metadata"
ATTEMPT=WRITES/"v11_namespace_attempt1"
ARCHIVE=ROOT/"research/reviews/American_C3_native_MLM_full_fit_v8/v8_failed_root_archive_v10"
PRESERVED_REFS=[
 dict(path=str((ARCHIVE/"current_writer_proof_before_archive_v10.json").relative_to(ROOT)),
      sha256="cb24ff9cdf166c873ae454cc04299a82ba712e0a07566b6129e8e89adfac005d"),
 dict(path=str((ARCHIVE/"before_rename.json").relative_to(ROOT)),
      sha256="c175eca4f7cdb918490a2553aa5940ab99797b41451c9c2fa854730569707436")]
KNOWN_ROOTS=(2817234,2817625,2817931,2818200)
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
    return dict(schema="native_MLM_v11_targeted_known_roots_only",checks=rows,
        observed_utc=datetime.now(timezone.utc).isoformat(),historical_ids_complete=False,
        historical_descendant_ids_lost=True,complete_old_writer_absence_claimed=False,
        native_wait_exit="UNKNOWN",actual_optimizer_updates="UNKNOWN",unrelated_process_metadata_scanned=False)
def validate_directive(record):
    value=a.read(record)
    needed=dict(schema="native_MLM_v11_namespace_root_directive_v1",root_authorizer="root_agent",active=True,
        key=a.KEY,original_storage_ref=a.ORIGINAL_STORAGE_REF,reserved_bytes=a.CAP,expires_at_utc=a.EXPIRES,
        original_parent_scopes_fully_measured=True,all_regular_scientific_files_preserved_in_place=True,
        original_failed_namespace_unlinks_exact_eight_recorded_sockets_only=True,
        current_attempt_dead_IPC_cleanup_after_genuine_owned_drain_authorized=True,known_roots_groups_ESRCH_before_each_unlink=True,
        exact_lstat_and_kernel_no_listener_before_each_unlink=True,
        no_global_unrelated_FD_stability_prerequisite=True,
        all_prior_launch_resume_write_capabilities_cooperatively_revoked=True,
        complete_old_writer_absence_claimed=False,historical_ids_complete=False,
        historical_descendant_ids_lost=True,prior_native_wait_exit="UNKNOWN",prior_optimizer_updates="UNKNOWN",
        no_new_reservation=True,science_unchanged=True,
        output_child=a.OUTPUT,tmp_child=str(a.TMP.relative_to(ROOT)),aux_child=str(a.ATTEMPT.relative_to(ROOT)),
        parent_completion_envelope_after_actual_wait0_authorized=True,
        original_admin_storage_ref=ADMIN_STORAGE_REF,original_admin_cap_bytes=5242880,
        original_admin_expires_at_utc=ADMIN_EXPIRY)
    a.require(all(value.get(k)==v for k,v in needed.items()),"Exact narrow actual root namespace directive")
    a.check_expiry(dict(created_at_utc=a.CREATED,expires_at_utc=a.EXPIRES))
    return value
def original_socket_rows():
    for record in PRESERVED_REFS:a.read(record)
    before=a.read(PRESERVED_REFS[1])
    a.require(before["historical_ids_complete"] is False and before["native_wait_exit"]=="UNKNOWN"
        and before["actual_optimizer_updates"]=="UNKNOWN","Preserved historical uncertainty")
    rows=before["dead_socket_metadata"]
    a.require(len(rows)==8 and len({row["path"] for row in rows})==8,"Exactly eight original recorded sockets")
    for row in rows:
        p=a.local(row["path"]);s=row["lstat"]
        a.require(set(row)=={"path","lstat"} and len(s)==7 and stat.S_ISSOCK(s[2]) and s[3]==1
            and a.PARENT_TMP in p.parents and p.parent.name.startswith("pymp-") and p.name.startswith("listener-"),
            "Exact original multiprocessing socket metadata")
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
    from terminal_namespace_v11 import kernel_listeners
    return kernel_listeners(paths,read_text)
def validate_cleanup_receipt(record,directive_ref):
    value=a.read(record)
    a.require(value["schema"]=="native_MLM_v11_exact_namespace_cleanup_receipt"
        and value["root_authorizer"]=="root_agent" and value["active"] is True
        and value["helper_ref"]==a.ref(SOURCE) and value["root_directive_ref"]==directive_ref
        and value["original_admin_storage_ref"]==ADMIN_STORAGE_REF
        and value["preserved_metadata_refs"]==PRESERVED_REFS
        and value["removed_socket_metadata"]==original_socket_rows()
        and value["all_regular_scientific_files_preserved_in_place"] is True
        and value["original_archive_two_files_preserved_byte_exact"] is True
        and value["new_reservation_created"] is False and value["global_unrelated_metadata_census_used"] is False
        and value["complete_old_writer_absence_claimed"] is False
        and value["historical_ids_complete"] is False and value["prior_native_wait_exit"]=="UNKNOWN"
        and value["prior_optimizer_updates"]=="UNKNOWN","Exact original-node-only root cleanup receipt")
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
    a.require(auth["schema"]=="native_MLM_v11_exact_namespace_cleanup_authority"
        and auth["root_authorizer"]=="root_agent" and auth["active"] is True
        and auth["helper_ref"]==a.ref(SOURCE) and auth["original_admin_storage_ref"]==ADMIN_STORAGE_REF
        and auth["preserved_metadata_refs"]==PRESERVED_REFS
        and auth["no_new_reservation"] is True and auth["all_prior_capabilities_revoked"] is True,
        "Separately authenticated actual root helper; never a false original consumer binding")
    validate_directive(directive_ref)
    closure=a.read(auth["source_closure_ref"])
    a.require(closure["schema"]=="native_MLM_fullfit_execution_source_closure_v11"
        and closure["original_storage_ref"]==a.ORIGINAL_STORAGE_REF,"Actual new source closure")
    for path,sha in closure["source_sha256"].items():a.require(a.ref(path)["sha256"]==sha,"Exact new actual source bytes")
    a.require(closure["source_sha256"].get(SOURCE)==a.ref(SOURCE)["sha256"],"Actual helper source authenticated")
    a.require(set(p.name for p in ARCHIVE.iterdir())=={Path(x["path"]).name for x in PRESERVED_REFS},
        "Preserve exact original two-file partial archive")
    rows=original_socket_rows()
    for row in rows:
        p=a.local(row["path"]);socket_ancestors(p)
        a.require(stamp(p.lstat())==row["lstat"],"Original exact socket lstat unchanged")
    a.require(not listeners([row["path"] for row in rows]),"No exact old socket kernel endpoint")
    known_roots_groups_absent()
    a.require(not ATTEMPT.exists(),"Single cleanup attempt, no reset or replay")
    m,storage=admin_guard();m.budget_check_v9(storage,boundary=True,required_write_bytes=1048576)
    ATTEMPT.mkdir()
    admin_save(m,storage,"before_cleanup.json",dict(authority_ref=authority_ref,root_directive_ref=directive_ref,
        nodes=rows,preserved_metadata_refs=PRESERVED_REFS,known_state=known_roots_groups_absent(),
        historical_ids_complete=False,complete_old_writer_absence_claimed=False))
    events=[]
    for index,row in enumerate(rows):
        # No unrelated process state is consulted. Recheck known IDs, ancestors,
        # inode/type/metadata and exact kernel endpoint immediately before unlink.
        validate_directive(directive_ref)
        state=known_roots_groups_absent()
        p=a.local(row["path"]);socket_ancestors(p)
        a.require(stamp(p.lstat())==row["lstat"] and not listeners([str(p)]),"Fresh exact dead-socket pre-unlink state")
        event=dict(index=index,node=row,known_state=state,exact_kernel_listener_matches=[])
        events.append(admin_save(m,storage,"before_unlink_%02d.json"%index,event))
        # Administrative guard writes are outside original TMP. Recheck after
        # their accounting boundary, before the only original-namespace mutation.
        known_roots_groups_absent();socket_ancestors(p)
        a.require(stamp(p.lstat())==row["lstat"] and not listeners([str(p)]),"Immediate unchanged exact-node check")
        p.unlink()
    # Native guard still scans all original parent scopes without exclusions.
    native=a.read(a.ORIGINAL_STORAGE_REF);m.budget_check_v9(native,boundary=True,required_write_bytes=0)
    for record in PRESERVED_REFS:a.read(record)
    receipt=dict(schema="native_MLM_v11_exact_namespace_cleanup_receipt",root_authorizer="root_agent",active=True,
        helper_ref=a.ref(SOURCE),authority_ref=authority_ref,root_directive_ref=directive_ref,
        original_admin_storage_ref=ADMIN_STORAGE_REF,preserved_metadata_refs=PRESERVED_REFS,
        removed_socket_metadata=rows,pre_unlink_receipt_refs=events,
        all_regular_scientific_files_preserved_in_place=True,original_archive_two_files_preserved_byte_exact=True,
        new_reservation_created=False,global_unrelated_metadata_census_used=False,complete_old_writer_absence_claimed=False,
        historical_ids_complete=False,historical_descendant_ids_lost=True,prior_native_wait_exit="UNKNOWN",
        prior_optimizer_updates="UNKNOWN",completed_utc=datetime.now(timezone.utc).isoformat())
    result=admin_save(m,storage,"cleanup_receipt.json",receipt)
    print(json.dumps(dict(namespace_cleanup_receipt_ref=result)),flush=True)
    return result
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--cleanup",action="store_true")
    p.add_argument("--authority");p.add_argument("--expected-authority-sha256");args=p.parse_args(argv)
    if not args.cleanup:
        print(json.dumps(dict(status="source_only_inert_v11",actual_cleanup=False)));return 0
    a.require(args.authority and args.expected_authority_sha256,"Exact actual root cleanup authority")
    cleanup(dict(path=args.authority,sha256=args.expected_authority_sha256));return 0
if __name__=="__main__":raise SystemExit(main())
