"""Root-only atomic preservation of the exact never-admitted014 inference packet.
Default is inert. Does not admit, launch, configure guard or mutate accounting.
"""
from pathlib import Path
import hashlib
_common=Path(__file__).with_name("recovery_common_v1.py")
if hashlib.sha256(_common.read_bytes()).hexdigest()!="20f3709aad0a51fda4f9c77c1a736ee4be590530ad25382477f1e6c7c6fd69e8":raise ValueError("Frozen recovery common source changed")
from recovery_common_v1 import *
SOURCE=HERE+"/preserve_failed_packet_v1.py"
def inspect_v1():
    m=admin();launch,terminal=failed_terminal_v1(m)
    wave=load_pinned(WAVE_REF,"_wave014_original_science_IO");_,pins=wave.frozen()
    stages={(p,s):wave.successful_stage(p,s,pins) for p,s in
            [("american_c3_short_train_014","simulate"),("american_c3_short_train_015","simulate"),("american_c3_short_train_015","infer")]}
    need(not (ROOT/"experiments/results/American_C3_short_train2mb_wave_v2/american_c3_short_train_014/inference").exists(),"Never-spawned inference output must be absent")
    need(not (PARTIAL.parent/"parent_preparation_completed.json").exists(),"014 parent must remain incomplete")
    need(PARTIAL.is_dir() and not PARTIAL.is_symlink() and sorted(p.name for p in PARTIAL.iterdir())==sorted(PARTIAL_HASHES),"Exact two-file failed-admission packet only")
    for name,digest in PARTIAL_HASHES.items():need(m.ref(PARTIAL/name)["sha256"]==digest,"Exact preserved failed admission bytes")
    need(not ARCHIVE.exists() and not (ROOT/HERE/"failed_admission_preservation.root_v1.json").exists(),"Fresh preservation destination; no replay")
    return m,launch,stages

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--execute",action="store_true");a=p.parse_args()
    if not a.execute:
        print(json.dumps(dict(status="source_only_no_move_guard_admission_or_launch",only_stage="american_c3_short_train_014/infer")));return 0
    need(Path.cwd().resolve()==ROOT,"Exact project cwd required")
    m,launch,stages=inspect_v1()
    # Root-only exact old supervisor absence. Worker authoring this helper never probes.
    observations=[m.absence(launch["remote_supervisor_pid"]),m.absence(launch["remote_supervisor_process_group"],True)]
    for pid in launch["additional_root_owned_remote_pids"]:observations.append(m.absence(pid))
    m.save(ROOT/HERE,"failed_authority_revocation.root_v1.json",dict(root_authorizer="root_agent",
        exact_failed_reserve_authority_SHA256=PARTIAL_HASHES["reserve_authority.json"],
        failed_draft_apply_resume_retry_write_capabilities_revoked=True,preserve_all_bytes=True,
        original_launch_ref=FAILED_LAUNCH_REF,actual_failed_terminal_ref=FAILED_TERMINAL_REF,
        actual_capacity_rejection_precedes_guard_reserve_append=True,old_root_absence=observations,
        new_original_driver_admission_required=True,read_only_reuse_preserved=True))
    os.rename(PARTIAL,ARCHIVE)
    for name,digest in PARTIAL_HASHES.items():need(m.ref(ARCHIVE/name)["sha256"]==digest,"Atomic preservation changed bytes")
    result=m.save(ROOT/HERE,"failed_admission_preservation.root_v1.json",dict(root_authorizer="root_agent",
        status="preserved_exact_unadmitted014_infer_packet",failed_exec_terminal_ref=FAILED_TERMINAL_REF,
        source_ref=m.ref(SOURCE),source_common_ref=m.ref(HERE+"/recovery_common_v1.py"),
        original_packet=str(PARTIAL.relative_to(ROOT)),archive_packet=str(ARCHIVE.relative_to(ROOT)),
        preserved_file_SHA256=PARTIAL_HASHES,scientific_outputs_modified=False,old_draft_replay_revoked=True,
        actual_new_admission_or_launch=False,accounting_mutated=False,
        retained_stage_refs={p+"/"+s:{k:v[k] for k in ("wait_ref","completion_ref")} for (p,s),v in stages.items()}))
    print(json.dumps(dict(preservation_receipt_ref=result,next_mode="Unchanged original driver --run-parents american_c3_short_train_014 only")));return 0
if __name__=="__main__":raise SystemExit(main())
