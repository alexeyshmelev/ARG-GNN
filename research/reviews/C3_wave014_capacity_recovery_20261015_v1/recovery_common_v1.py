"""Pinned read-only metadata helpers; no import-time live operations."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,sys,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/C3_wave014_capacity_recovery_20261015_v1"
WAVE="research/reviews/American_C3_short_train2mb_wave_980_source_v1"
WAVE_REF=dict(path=WAVE+"/root_wave_980_driver.py",sha256="f55f1162e12302c712ab96c810dcee81f8063dd170fd54d3d19e98feb82c0df6")
CLOSE_REF=dict(path="research/reviews/American_C3_short_train2mb_wave_980_close_source_v1/close_stage_v1.py",sha256="854799fed7c3bee12acf4c025de5cee9cd59742730f905aaca3ece6d696965c7")
FAILED_LAUNCH_REF=dict(path=WAVE+"/root_pair014_015_exec_launch_identity_v1.json",sha256="2c85c37c4211bff26b85da3e3038becd2a8041b2941388cebd50252858802c09")
FAILED_TERMINAL_REF=dict(path=WAVE+"/root_exec_session50286_terminal_actual_v1.json",sha256="66566bf7350639926c99567816c3eb443cf823ac9ddb2090652d5c5801f6528b")
PARTIAL=ROOT/WAVE/"admitted/american_c3_short_train_014/infer"
ARCHIVE=ROOT/HERE/"preserved_failed_infer014_v1"
PARTIAL_HASHES={"config.json":"bcf85071b001b3b8e602041732b266992eff84378ab4dc496d99746ee7558630","reserve_authority.json":"2e8c7684731e1c19105aa5457e6e6c1e05fe56aaca0705ff1b84a2eee6d01954"}
def need(ok,message):
    if not ok:raise ValueError(message)
def load_pinned(record,name):
    p=ROOT/record["path"];need(hashlib.sha256(p.read_bytes()).hexdigest()==record["sha256"],"Original source changed")
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def admin():return load_pinned(CLOSE_REF,"_wave014_original_close_IO")
def failed_terminal_v1(m):
    launch,terminal=m.read(FAILED_LAUNCH_REF),m.read(FAILED_TERMINAL_REF)
    need(launch["kind"]=="actual_exec_session_launch_identity" and launch["root_authorizer"]=="root_agent"
         and launch["driver_ref"]==WAVE_REF and launch["parent_ids"]==["american_c3_short_train_014","american_c3_short_train_015"]
         and launch["exec_session_id"]==50286 and launch["remote_supervisor_pid"]==2903577
         and launch["remote_supervisor_start_ticks"]==475059625 and launch["remote_supervisor_process_group"]==2903577
         and launch["all_additional_root_owned_remote_processes_listed"] is True,"Exact actual original pair launch")
    need(terminal["schema"]=="root_actual_exec_session_terminal_v1" and terminal["root_authorizer"]=="root_agent"
         and terminal["actual_exec_session_terminal"] is True and terminal["exec_session_id"]==50286
         and terminal["actual_exec_session_exit_code"]==terminal["actual_SSH_propagated_returncode"]==1
         and terminal["remote_direct_Popen_wait_observed"] is False and terminal["remote_direct_Popen_wait_claimed"] is False
         and terminal["actual_root_pid"]==2903577 and terminal["actual_root_start_ticks"]==475059625
         and terminal["actual_root_process_group"]==2903577 and terminal["exact_original_root_source"]==WAVE_REF
         and terminal["root_terminal_time_unknown"] is True,"Exact actual failed exec1, no manufactured remote wait/time")
    failure=terminal["observed_failure"]
    need(failure==dict(type="ValueError",message="Prospective980GB ceiling/1TB human limit exceeded",
         failure_boundary="admit_stage_v1 -> guard.apply_reserve_v1 -> charge",stage="american_c3_short_train_014/infer",
         failed_scientific_child_launch_claimed=False),"Only exact never-spawned014 infer capacity rejection")
    return launch,terminal
