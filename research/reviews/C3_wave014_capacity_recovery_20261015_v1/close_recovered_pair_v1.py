"""Narrow two-root close IO for recovered014/015, using unchanged original guard/run.
Default is inert. Actual roots/terminals and four real waits/completions required.
"""
from pathlib import Path
import hashlib
_common=Path(__file__).with_name("recovery_common_v1.py")
if hashlib.sha256(_common.read_bytes()).hexdigest()!="20f3709aad0a51fda4f9c77c1a736ee4be590530ad25382477f1e6c7c6fd69e8":raise ValueError("Frozen recovery common source changed")
from recovery_common_v1 import *
SOURCE=HERE+"/close_recovered_pair_v1.py"
PRESERVE_HELPER_REF=dict(path=HERE+"/preserve_failed_packet_v1.py",sha256="9f4081d9d57cf9ef5b87d451e239a9f9f00d9b5c3fb78982ba6473909463c2f6")
STAGES=("simulate","infer")
BASE=ROOT/WAVE/"admitted"
def recovery_terminal_v1(m,launch_ref,terminal_ref):
    launch,terminal=m.read(launch_ref),m.read(terminal_ref)
    need(launch["kind"]=="actual_exec_session_launch_identity" and launch["root_authorizer"]=="root_agent"
         and launch["driver_ref"]==WAVE_REF and launch["parent_ids"]==["american_c3_short_train_014"]
         and type(launch["exec_session_id"]) is int and launch["exec_session_id"]>0 and launch["exec_session_id"]!=50286
         and type(launch["remote_supervisor_start_ticks"]) is int and launch["remote_supervisor_start_ticks"]>0
         and launch["all_additional_root_owned_remote_processes_listed"] is True,"Actual014-only recovery root identity")
    need(terminal["schema"]=="root_actual_exec_session_terminal_v1" and terminal["root_authorizer"]=="root_agent"
         and terminal["actual_exec_session_terminal"] is True and terminal["exec_session_id"]==launch["exec_session_id"]
         and terminal["actual_exec_session_exit_code"]==terminal["actual_SSH_propagated_returncode"]==0
         and terminal["remote_direct_Popen_wait_observed"] is False and terminal["remote_direct_Popen_wait_claimed"] is False
         and terminal["actual_root_pid"]==launch["remote_supervisor_pid"]
         and terminal["actual_root_start_ticks"]==launch["remote_supervisor_start_ticks"]
         and terminal["actual_root_process_group"]==launch["remote_supervisor_process_group"]
         and terminal["exact_original_root_source"]==WAVE_REF,"Genuine014-only recovery exec0, no remote Popen.wait claim")
    return launch

def collect_recovered_pair_v1(parent,launch_ref,terminal_ref,pins,m,recovery_launch_ref,recovery_terminal_ref):
    need(parent in ("american_c3_short_train_014","american_c3_short_train_015"),"Only recovered014/015 pair")
    need(launch_ref==FAILED_LAUNCH_REF and terminal_ref==FAILED_TERMINAL_REF,"Exact preserved failed pair execution")
    old,_=failed_terminal_v1(m);new=recovery_terminal_v1(m,recovery_launch_ref,recovery_terminal_ref)
    preserve_ref=m.ref(ROOT/HERE/"failed_admission_preservation.root_v1.json");preserve=m.read(preserve_ref)
    need(preserve["root_authorizer"]=="root_agent" and preserve["status"]=="preserved_exact_unadmitted014_infer_packet"
         and preserve["failed_exec_terminal_ref"]==FAILED_TERMINAL_REF and preserve["old_draft_replay_revoked"] is True
         and preserve["preserved_file_SHA256"]==PARTIAL_HASHES and preserve["source_ref"]==PRESERVE_HELPER_REF
         and m.ref(PRESERVE_HELPER_REF["path"])==PRESERVE_HELPER_REF
         and preserve["source_common_ref"]==m.ref(HERE+"/recovery_common_v1.py"),"Actual atomic preservation receipt required")
    for name,digest in PARTIAL_HASHES.items():need(m.ref(ARCHIVE/name)["sha256"]==digest,"Failed bytes remain preserved")
    pair=["american_c3_short_train_014","american_c3_short_train_015"]
    old_supervisor=old["remote_supervisor_pid"];recovery_supervisor=new["remote_supervisor_pid"]
    pids=set();groups=set();rows=[]
    for launch in (old,new):
        extra=launch["additional_root_owned_remote_pids"]
        need(isinstance(extra,list) and all(type(x) is int and x>=2 for x in extra),"Complete additional root PID roster")
        pids.update([launch["remote_supervisor_pid"],*extra]);groups.add(launch["remote_supervisor_process_group"])
    read=m.read;ref=m.ref
    for p in pair:
        parent_done_ref=ref(BASE/p/"parent_preparation_completed.json");parent_done=read(parent_done_ref)
        need(parent_done["status"]=="completed_C3_2Mb_TRAIN_parent_preparation" and parent_done["parent_id"]==p
            and parent_done["truth_kept_out_of_model_inputs"] is True,"Both actual complete unchanged parents required")
        need(parent_done["root_supervisor_pid"]==(recovery_supervisor if p.endswith("014") else old_supervisor),"Exact parent terminal publishing root")
        for stage in STAGES:
            supervisor=recovery_supervisor if p.endswith("014") and stage=="infer" else old_supervisor
            folder=BASE/p/stage
            ar,lr,wr=ref(folder/"admission.json"),ref(folder/"launch.json"),ref(folder/"actual_wait.json")
            spec,child,wait=read(ar),read(lr),read(wr);own=spec["storage"]["reservation"]
            completed_ref=wait["completed_ref"];completed=read(completed_ref)
            expected_output="experiments/results/American_C3_short_train2mb_wave_v2/"+p+("/simulation" if stage=="simulate" else "/inference")
            expected_key="H041/American-C3-short2mb-"+p+"-"+stage+"-v2"
            need(spec["driver_ref"]==pins["wave_driver"] and spec["storage_adapter_ref"]==pins["guard"]
                and spec["completed_overlay_policy_ref"]==pins["policy"] and spec["parent_id"]==p and spec["stage"]==stage
                and spec["output_directory"]==expected_output and own["key"]==expected_key
                and spec["storage"]["accounting_mode"]=="prospective_980gb_v1"
                and own["reserved_bytes"]==(1000000000 if stage=="simulate" else 2000000000)
                and child["admission_ref"]==wait["admission_ref"]==ar and child["no_helper_launcher"] is True,
                "Exact unchanged future980 stage source/reservation/output/cap")
            need(child["pid"]==wait["pid"]==wait["worker_pid"]==wait["worker_process_group"]==child["worker_process_group"]
                and child["supervisor_pid"]==wait["supervisor_pid"]==supervisor
                and wait["driver_ref"]==pins["wave_driver"] and wait["parent_id"]==p and wait["stage"]==stage
                and wait["returncode"]==0 and wait["actual_direct_worker_wait"] is True
                and wait["worker_process_group_drained"] is True and wait["operational_failure"] is None
                and wait["supervisor_capture_error"] is None and wait["terminal_log_materialization_error"] is None,
                "Genuine per-stage direct worker wait0/drain and same remote supervisor")
            need(parent_done["stages"][stage]==dict(wait_ref=wr,completion_ref=completed_ref)
                and completed_ref["path"]==expected_output+"/completed.json"
                and child["owned_process_roster"]==wait["owned_process_roster"]
                and {r["pid"] for r in wait["owned_process_roster"]}=={supervisor,wait["pid"]},
                "Exact same complete stage lineage and original source-owned roster")
            obs=read(completed["observation_ref"])
            need(obs["parent_id"]==p and obs["split"]=="train" and obs["sequence_length_bp"]==2000000
                and obs["counts_haploid"]==dict(AFR=200,EUR=200,ASIA=200,ADMIX=1000)
                and obs["query_ancestry_truth_in_inputs"] is False,"Original2Mb1600-haplotype TRAIN observation only")
            if stage=="simulate":
                need(completed["status"]=="completed_C3_simulation_reconstruction" and completed["parent_id"]==p
                    and completed["sample_unit"]=="haploid","Actual unchanged simulation completion")
            else:
                need(completed["status"]=="completed_current_C3_inference" and completed["tsinfer_num_threads"]==8
                    and obs["sites_and_sample_identities_verified"] is True,"Actual unchanged verified inference completion")
            pids.update(r["pid"] for r in wait["owned_process_roster"]);groups.add(wait["worker_process_group"])
            rows.append((p,stage,ar,lr,wr,completed_ref,spec,child,wait,completed,own))
    need(len(pids)<=256 and all(type(x) is int and x>=2 for x in groups),"Bounded complete root PID/group roster")
    return rows,sorted(pids),sorted(groups)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--execute",action="store_true")
    p.add_argument("--parent",choices=("american_c3_short_train_014","american_c3_short_train_015"));p.add_argument("--stage",choices=STAGES);p.add_argument("--output-packet")
    for name in ("recovery-exec-launch","recovery-exec-terminal"):
        p.add_argument("--"+name);p.add_argument("--expected-"+name+"-sha256")
    args=p.parse_args()
    if not args.execute:
        print(json.dumps(dict(status="source_only_no_guard_PID_observation_or_close",original_failed_exec=50286,required_recovery_parent="american_c3_short_train_014")));return 0
    m=admin();rl=m.input_ref(args,"recovery_exec_launch");rt=m.input_ref(args,"recovery_exec_terminal")
    args.exec_launch=FAILED_LAUNCH_REF["path"];args.expected_exec_launch_sha256=FAILED_LAUNCH_REF["sha256"]
    args.exec_terminal=FAILED_TERMINAL_REF["path"];args.expected_exec_terminal_sha256=FAILED_TERMINAL_REF["sha256"]
    def collect(parent,launch_ref,terminal_ref,pins):return collect_recovered_pair_v1(parent,launch_ref,terminal_ref,pins,m,rl,rt)
    def save(folder,name,value):
        if name in ("scope_write_resume_revocation.json","root_observation.json"):
            value=dict(value,failed_pair_exec_launch_ref=FAILED_LAUNCH_REF,failed_pair_exec_terminal_ref=FAILED_TERMINAL_REF,
                recovery_exec_launch_ref=rl,recovery_exec_terminal_ref=rt,
                failed_admission_preservation_ref=m.ref(ROOT/HERE/"failed_admission_preservation.root_v1.json"),
                pair_completed_under_two_actual_supervisors=True)
        return m.save(folder,name,value)
    namespace=dict(m.run.__globals__,HERE=HERE,SOURCE=SOURCE,collect_pair_v1=collect,save=save)
    # Exact original close run bytecode: same locks, fresh actual state, proof
    # request, PID checks, scope measurement and apply_close_completed_v1.
    execute=types.FunctionType(m.run.__code__,namespace,"run_recovered_pair_v1",m.run.__defaults__,m.run.__closure__)
    return execute(args)
if __name__=="__main__":raise SystemExit(main())
