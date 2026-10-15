"""One focused actual-started-metadata allocation fixture; no production phase."""
from pathlib import Path
import hashlib,json,os,sys
from bounded_phase_writer_v1 import LocalWriteEngineV1,charge_cap,actual_charge
HERE=Path(__file__).resolve().parent
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
FAILED=ROOT/"experiments/results/H049_CPU_order2_cut_memory_prepare_bounded_v3/.c086d29d402a4f408776f8e4b315f056.write_000.tmp"
FAILED_SHA="34a61e98cbffbce228120466637910e66fcdecd0107d14b87893c3c4e603ef33"
def stamp(s):return [s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_size,s.st_blocks,s.st_mtime_ns,s.st_ctime_ns]
def main():
    before=FAILED.lstat();raw=FAILED.read_bytes();value=json.loads(raw)
    assert hashlib.sha256(raw).hexdigest()==FAILED_SHA and len(raw)==59702
    assert value["schema"]=="H049_order2_started_v1" and value["phase"]=="prepare"
    assert len(value["output_plan"]["parents"])==40 and value["outer_truth_read"] is False
    old_cap=((len(raw)+4095)//4096)*4096+8192
    assert actual_charge(before)>old_cap
    fixture=HERE/"actual_metadata_fixture_v2"
    fixture.mkdir(exist_ok=False);aux=fixture/"aux";aux.mkdir();(aux/"ro").mkdir(mode=0o555);(aux/"ro").chmod(0o555)
    payloads={"started.json":raw}
    for target in (65535,65536,65537,131071,131072,131073,507904):
        variant=dict(value,allocation_fixture_padding="")
        base=(json.dumps(variant,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
        assert len(base)<=target
        variant["allocation_fixture_padding"]="x"*(target-len(base))
        data=(json.dumps(variant,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
        assert len(data)==target and json.loads(data)["output_plan"]==value["output_plan"]
        payloads["metadata_%d.json"%target]=data
    rules={n:dict(kind="json",max_payload_bytes=len(b),group="metadata") for n,b in payloads.items()}
    total=sum(charge_cap(len(b)) for b in payloads.values());calls=[0]
    def local():calls[0]+=1
    plan=dict(output="out",auxiliary="aux",files=rules,readonly_aux_directories=["ro"],
        per_file_overhead_bytes=8192,allocation_round_bytes=131072,allocation_slack_bytes=131072,per_directory_charge_cap_bytes=16384,
        peak_phase_charge_bound_bytes=total+3*16384+8192,reservation_bytes=8*2**20,
        group_charge_caps=dict(metadata=total),reserved_terminal_tail_bytes=0,
        scratch_charge_bound_bytes=8192,nonce="a"*32)
    engine=LocalWriteEngineV1(root=fixture,plan=plan,auxiliary_existing={},check_local=local)
    records=[]
    for name,data in payloads.items():
        rec=engine.write_bytes(name,data,"json");p=fixture/"out"/name;s=p.lstat()
        assert p.read_bytes()==data and rec["sha256"]==hashlib.sha256(data).hexdigest()
        assert s.st_nlink==1 and actual_charge(s)<=charge_cap(len(data))
        records.append(dict(name=name,bytes=len(data),blocks512=s.st_blocks,allocated_bytes=s.st_blocks*512,
            actual_charge_bytes=actual_charge(s),new_charge_cap_bytes=charge_cap(len(data)),sha256=rec["sha256"]))
    summary=engine.summary();engine.close()
    assert stamp(FAILED.lstat())==stamp(before) and hashlib.sha256(FAILED.read_bytes()).hexdigest()==FAILED_SHA
    assert not any("project_budget" in n or "prospective_980" in n for n in sys.modules)
    receipt=dict(schema="H049_actual_metadata_allocation_fixture_v1",status="PASS_focused_actual_metadata_only",
        failed_original_ref=dict(path=str(FAILED.relative_to(ROOT)),sha256=FAILED_SHA),
        original_st_size=before.st_size,original_st_blocks=before.st_blocks,original_allocated_bytes=before.st_blocks*512,
        original_actual_charge_bytes=actual_charge(before),original_old_cap_bytes=old_cap,
        failed_original_preserved=True,byte_identical_actual_started_tested=True,records=records,
        local_counter=summary,local_checks_called=calls[0],production_phase_constructed=False,
        production_guards_or_jobs_or_GPU_or_scientific_payloads=False,
        allocation_policy="128KiB rounding plus128KiB allocation slack for nonempty files, actual boundary excess4KiB; actual st_blocks fail-stop retained; no universal filesystem allocation guarantee")
    (HERE/"actual_metadata_fixture_receipt_v2.json").write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    print(json.dumps(dict(status=receipt["status"],actual_metadata_cases=len(records),original_old_cap_bytes=old_cap,
        original_actual_charge_bytes=actual_charge(before),new_actual_started_cap_bytes=charge_cap(len(raw))),sort_keys=True))
if __name__=="__main__":
    try:main()
    except BaseException as error:
        import traceback
        (HERE/"actual_metadata_fixture_failure_v2.json").write_text(json.dumps(dict(
            type=type(error).__name__,message=str(error),traceback=traceback.format_exc()),sort_keys=True,indent=2)+"\\n")
        raise
