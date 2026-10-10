"""Root-specific completed-GST storage settlement, preserving retained output."""
from pathlib import Path
import argparse, copy, datetime, fcntl, hashlib, json, os, stat
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
OLD=23521048576
NEW=2000000000
KEY="biological_gpu2_v1/biological-gst-seed101"
OUTPUT=ROOT/"experiments/results/biological_gst_control_v1_seed101"
AUDIT=ROOT/"research/reviews/biological_gst_completed_storage_settlement_v1/storage_revision_v2"
INSPECTED=ROOT/"research/reviews/biological_gst_completed_storage_settlement_v1/inspection.json"
INSPECTED_SHA="c420d0ea2690aed015dc895da09ca3619e2dd9699942190ca9575ded0043817d"
SOURCE=ROOT/"src/arggnn/gpu_pool_v1.py"
SOURCE_SHA="ae1e0494ab672cfa3017759460eb06a4b4789eaef54928496c3c80314adcc0e1"
SNAPSHOT_SHA="0826fcedefa40dd261999c4916b85ae11f28628e5cee9046b2e4f6bf868fa10d"
EXPECTED_LEDGER="ba58c003960dcba0fe320472edfbe5edb28adbff64bcc8840edcb1016079ab59"
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""): h.update(b)
    return h.hexdigest()
def write(name,obj):
    with (AUDIT/name).open("x") as f: json.dump(obj,f,indent=2,allow_nan=False); f.write("\n")
def snapshot_job(module):
    registry=json.loads((ROOT/".runtime/gpup/biological_gpu2_v1/registry.json").read_text())
    job=next(j for j in registry["jobs"] if j["logical_job"]=="biological-gst-seed101")
    if registry["safety_error"] is not None or job["state"]!="succeeded" or job["exit_code"]!=0:
        raise ValueError("GST job must have succeeded without pool safety error")
    if job["requested"]["module"]!="arggnn.biological_gst_control_v1" or job["requested"]["output"]!=str(OUTPUT) or job["requested"]["storage_bytes"]!=OLD:
        raise ValueError("Exact original job identity/reservation required")
    for saved in job["process_identities"]:
        try:
            current=module.proc_identity(saved["pid"])
        except (FileNotFoundError, ProcessLookupError):
            current=None
        if current is not None and current["start_ticks"]==saved["start_ticks"]:
            raise ValueError("A known GST worker/descendant remains present")
    receipt=json.loads((OUTPUT/"receipt.json").read_text())
    if receipt["status"]!="completed_two_GST_conditions_one_seed" or any(m["status"]!="completed" for m in receipt["models"].values()):
        raise ValueError("Exact completed fit receipt required")
    return {k:job[k] for k in ["logical_job","state","exit_code","pid","leader_start_ticks","requested","module_source_sha256","process_identities"]}
def inventory():
    if OUTPUT.is_symlink() or OUTPUT.resolve()!=OUTPUT or not OUTPUT.is_dir(): raise ValueError("Canonical completed output required")
    records={}; charge=0
    def walk_error(error):
        raise error
    for base, dirs, files in os.walk(OUTPUT,followlinks=False,onerror=walk_error):
        for p in [Path(base)]+[Path(base)/n for n in dirs+files]:
            if p.is_symlink(): raise ValueError("Symlink in completed output")
        info=Path(base).stat()
        if not stat.S_ISDIR(info.st_mode): raise ValueError("Non-directory scope")
        charge+=max(info.st_size,info.st_blocks*512)
        for name in sorted(files):
            p=Path(base)/name; before=p.stat()
            if not stat.S_ISREG(before.st_mode): raise ValueError("Nonregular output")
            digest=sha(p); after=p.stat()
            if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino): raise ValueError("Output changed while hashing")
            charge+=max(after.st_size,after.st_blocks*512)
            records[str(p.relative_to(OUTPUT))]={"bytes":after.st_size,"allocated_bytes":after.st_blocks*512,"sha256":digest}
    log=ROOT/".runtime/gpup/biological_gpu2_v1/jobs/biological-gst-seed101.log"
    if log.is_symlink() or not log.is_file(): raise ValueError("Canonical retained GST log required")
    info=log.stat(); charge+=max(info.st_size,info.st_blocks*512)
    records["retained_pool_log"]={"path":str(log.relative_to(ROOT)),"bytes":info.st_size,"allocated_bytes":info.st_blocks*512,"sha256":sha(log)}
    if charge>=NEW: raise ValueError("Retained output/log does not fit conservative2GB charge")
    return charge,records
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--apply",action="store_true")
    args=parser.parse_args()
    if sha(SOURCE)!=SOURCE_SHA: raise ValueError("Audited pool source changed")
    from arggnn import gpu_pool_v1 as module
    job=snapshot_job(module)
    if sha(INSPECTED)!=INSPECTED_SHA: raise ValueError("Root inspected scope receipt changed")
    inspected=json.loads(INSPECTED.read_text())
    if inspected["applied"] is not False or inspected["retained_charge_bytes"]!=NEW:
        raise ValueError("Exact prior inspect-only conservative charge required")
    with module.lock_file(ROOT,"scheduler.lock") as locked:
        fcntl.flock(locked,fcntl.LOCK_EX)
        snap=module.snapshot_record(ROOT,str(ROOT/"experiments/results/biological_shared_pool_storage_admission_v1/snapshot.json"),SNAPSHOT_SHA)
        def available():
            value=os.statvfs(ROOT); return value.f_bavail*value.f_frsize
        ledger=module.StorageLedger(ROOT,snap,130000000000,450000000000,available)
        before=ledger.path.read_bytes()
        if hashlib.sha256(before).hexdigest()!=EXPECTED_LEDGER: raise ValueError("Existing ledger changed. No settlement.")
        state=ledger._load(); old=module.encoded(state)
        if state["jobs"][KEY]!={"output":str(OUTPUT),"reserved_bytes":OLD}: raise ValueError("Exact old reservation required")
        before_account=ledger._check(state)
        snapshot_job(module)
        charged,files=inventory()
        if (charged!=inspected["retained_conservative_measured_bytes"] or files!=inspected["inventory_files"]):
            raise ValueError("Completed scope differs from frozen root inspection. No reduction.")
        # Recheck full local membership/size/hash under the scheduler lock.
        if inventory()!=(charged,files): raise ValueError("Completed output changed before reduction")
        snapshot_job(module)
        proposed=copy.deepcopy(state); proposed["jobs"][KEY]["reserved_bytes"]=NEW
        after_account=ledger._check(proposed)
        invariant=copy.deepcopy(proposed); invariant["jobs"][KEY]["reserved_bytes"]=OLD
        if module.encoded(invariant)!=old: raise ValueError("Unrelated ledger change")
        audit={"kind":"completed_GST_conservative_retained_output_settlement_v1","utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"root_authorizer":"root_agent","applied":args.apply,"source_sha256":SOURCE_SHA,"procedure_sha256":sha(__file__),"expected_before_sha256":EXPECTED_LEDGER,"terminal_job":job,"charged_scope":"finished GST output and retained worker log, never global remeasurement","prior_inspected_receipt_sha256":INSPECTED_SHA,"inventory_under_scheduler_lock_twice":True,"walk_errors_fail_closed":True,"retained_conservative_measured_bytes":charged,"retained_charge_bytes":NEW,"unmeasured_allowance_bytes":NEW-charged,"no_files_deleted":True,"all_other_reservations_and130GB_external_reserve_unchanged":True,"future_product_write_policy":"Finished GST product is frozen. New fits use fresh versioned output directories. New side effects remain covered by the unchanged external reserve.","before_account":before_account,"after_account":after_account,"inventory_files":files}
        if args.apply:
            write("ledger_before.json",json.loads(before))
            write("preapply_audit.json",audit)
            module.atomic_json(ledger.path,proposed)
            audit["after_ledger_sha256"]=sha(ledger.path)
            write("settlement_receipt.json",audit)
        else:
            write("inspection.json",audit)
        print(json.dumps({"applied":args.apply,"retained_measured_bytes":charged,"reserved_bytes":NEW,"before_projected_bytes":before_account["projected_bytes"],"after_projected_bytes":after_account["projected_bytes"],"file_records":len(files)}))
if __name__=="__main__": main()
