"""One handmade cap/atomicity fixture. Never constructs a production phase."""
from pathlib import Path
import hashlib,io,json,os,sys,zipfile
from bounded_phase_writer_v1 import LocalWriteEngineV1,publish_exclusive_v1,NPZ_MEMBERS,charge_cap
HERE=Path(__file__).resolve().parent
def main():
    root=HERE/"handmade_fixture_v3"
    if root.exists():raise RuntimeError("Fixture is one-shot; preserve previous evidence")
    root.mkdir();aux=root/"aux";aux.mkdir();(aux/"ro").mkdir();(aux/"ro").chmod(0o555)
    rules={n:{"kind":"json","max_payload_bytes":64,"group":"metadata"} for n in ["one.json","two.json","oversize.json"]}
    rules["bad.npz"]={"kind":"npz","max_payload_bytes":4096,"members":list(NPZ_MEMBERS),"group":"metadata"}
    plan={"output":"out","auxiliary":"aux","files":rules,"readonly_aux_directories":["ro"],
        "per_file_overhead_bytes":8192,"allocation_round_bytes":4096,"per_directory_charge_cap_bytes":16384,
        "peak_phase_charge_bound_bytes":200000,"reservation_bytes":200000,"group_charge_caps":{"metadata":charge_cap(64)},
        "reserved_terminal_tail_bytes":16000,"scratch_charge_bound_bytes":charge_cap(4096)+8192,"nonce":"0"*32}
    checks=[0]
    def local():checks[0]+=1
    engine=LocalWriteEngineV1(root=root,plan=plan,auxiliary_existing={},check_local=local)
    data=b'{"x":1}\n'
    rec=engine.write_bytes("one.json",data,"json")
    assert (root/"out/one.json").read_bytes()==data and rec["sha256"]==hashlib.sha256(data).hexdigest()
    results={"successful_exclusive_bytes_and_hash":True}
    for label,name,payload,kind in [
        ("payload_cap_rejected_without_file","oversize.json",b"x"*65,"json"),
        ("aggregate_cap_rejected_without_file","two.json",b"{}","json"),
        ("duplicate_rejected_without_overwrite","one.json",b"changed","json")]:
        before=set(p.name for p in (root/"out").iterdir())
        try:engine.write_bytes(name,payload,kind)
        except ValueError:pass
        else:raise AssertionError(label)
        assert set(p.name for p in (root/"out").iterdir())==before
        assert (root/"out/one.json").read_bytes()==data
        results[label]=True
    z=io.BytesIO()
    with zipfile.ZipFile(z,"w",compression=zipfile.ZIP_DEFLATED) as f:f.writestr("wrong.npy",b"wrong")
    try:engine.write_bytes("bad.npz",z.getvalue(),"npz")
    except ValueError:pass
    else:raise AssertionError("NPZ member reject")
    assert not (root/"out/bad.npz").exists()
    results["unexpected_npz_member_rejected_before_filesystem_write"]=True
    # Exercise the actual publication primitive against a pre-existing final.
    pub=root/"publication";pub.mkdir();fd=os.open(pub,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    payload=b"complete and fsynced payload"
    tempfd=os.open("scratch",os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=fd)
    with os.fdopen(tempfd,"wb") as f:f.write(payload);f.flush();os.fsync(f.fileno());inode=(os.fstat(f.fileno()).st_dev,os.fstat(f.fileno()).st_ino)
    (pub/"occupied").write_bytes(b"sentinel")
    try:publish_exclusive_v1(fd,"scratch","occupied")
    except FileExistsError:pass
    else:raise AssertionError("No-replace publication")
    assert (pub/"occupied").read_bytes()==b"sentinel" and (pub/"scratch").read_bytes()==payload
    results["exclusive_placeholder_preserves_existing_final_and_scratch"]=True
    publish_exclusive_v1(fd,"scratch","published")
    s=(pub/"published").stat()
    assert (pub/"published").read_bytes()==payload and (s.st_dev,s.st_ino)==inode and s.st_nlink==1 and not (pub/"scratch").exists()
    results["owned_placeholder_rename_exact_fsynced_single_link_inode_complete_bytes"]=True
    os.close(fd);summary=engine.summary();engine.close()
    assert not any("project_budget" in name or "prospective_980gb" in name for name in sys.modules)
    receipt={"schema":"bounded_phase_one_handmade_cap_atomicity_fixture_v1","status":"PASS_focused_fixture_only","checks":results,
        "local_checks_called":checks[0],"actual_local_counter":summary,
        "production_phase_constructed":False,"production_guards_imported_or_called":False,
        "jobs_or_admissions_or_GPU_probes":False,"scope":"small owned fixture only; no crash recovery, scientific .max fix, full40 preparation or performance qualification"}
    (HERE/"handmade_test_receipt_v3.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":receipt["status"],"checks":len(results),"local_charge_bytes":summary["scope_charge_bytes"]},sort_keys=True))
if __name__=="__main__":main()
