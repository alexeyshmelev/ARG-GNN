#!/usr/bin/env python3
"""Launch with full reservation accounting, preserving canonical GPU leases and runtime guards."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time

from pathlib import Path
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
sys.path[:0]=[str(ROOT/'research/reviews/native_ARGformer_VAL000_GPU0_shared_execution_20261016_v1'),str(ROOT/"research/reviews/American_C3_native_MLM_full_fit_v8"),str(ROOT/"research/reviews/American_C3_native_MLM_full_source_v5"),str(ROOT/"research/reviews/American_C3_native_MLM_amendment_v12"),str(ROOT),str(ROOT/"src"),str(ROOT/"scripts")]
ALLOWED_GPUS=(0,)
import root_native_contract_v1 as contract
from owned_lifecycle_v2 import OwnedTree
from pidfd_backend_v1 import self_probe_v1
from supervisor_v1 import _subreaper,_children,_Terminal
_STORAGE_GUARD=None
_SPEC=None
def eligible_free(record):
    available=(record["memory_total_mib"]-record["memory_used_mib"])*1048576
    physical,uuid=_SPEC["physical_gpu"],_SPEC["execution_gpu_uuid"]
    return record["index"]==physical and record["uuid"]==uuid and uuid==contract.GPU0_UUID and available>=54000000000

import project_budget_v9 as budget

SCHEDULABLE_GPUS = (0,)
SOURCE = 'research/reviews/native_ARGformer_VAL000_GPU0_shared_execution_20261016_v1/launcher_postMLM980_v1.py'
BASE_LAUNCHER = "scripts/launch_reserved_gpu_v6.py"


def full_storage_check_locked(record,source_sha,launcher_sha,*,required_write_bytes=0):
    contract.active(_SPEC)
    if source_sha!=contract.CORE or launcher_sha!=contract.BASE:raise ValueError("Original core/base source pins")
    if contract.ref(SOURCE) not in record["reservation"]["consumer_bindings"]:raise ValueError("Actual root launcher reserved")
    account=_STORAGE_GUARD.check_storage_locked_v1(record,boundary=True,required_write_bytes=required_write_bytes)
    return dict(account,actual_launcher_ref=contract.ref(SOURCE))

def admitted_storage_locked(path, expected_sha, job, source_sha, launcher_sha):
    record = budget.v4.bound(ROOT, {"path": path, "sha256": expected_sha}, True)
    if (record.get("kind") != "project_budget_v9_storage_record"
            or record.get("job_id") != job or record.get("active") is not True
            or record.get("consumer_uses_v9_guard") is not True):
        raise ValueError("Exact active root-issued v9 consumer storage authority required")
    account = full_storage_check_locked(record, source_sha, launcher_sha)
    return record, account

def save_wait_evidence_locked(record, account, child, returncode, source_sha, launcher_sha):
    output = contract.local(_SPEC["output_directory"])
    if not output.is_dir():
        # A failed-before-output run remains charged; root may record terminal evidence separately.
        return
    path = output / "launcher_exit_full_storage_v1.json"
    data = budget.v4.encoded({"kind": "native_postMLM980_GPU_launcher_actual_wait_v1",
        "actual_launcher_ref": contract.ref(SOURCE), "root_stage_request_ref": _SPEC["root_stage_request_ref"],
        "job_id": record["job_id"], "launcher_pid": os.getpid(), "worker_pid": child.pid,
        "returncode": returncode, "wait_status_known": True,
        "completed_unix": time.time(), "reservation_sha256": budget.digest(record["reservation"]),
        "descendant_drain_or_no_live_writer_claim": False,
        "storage_accounting": "prospective_980gb_v1"}) + b"\n"
    contract.active(_SPEC)
    _STORAGE_GUARD.budget_check_v9(record,boundary=True,required_write_bytes=len(data)+8192)
    # The reviewed read-only memo transaction has ended before this product write.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())

def main():
    global _STORAGE_GUARD,_SPEC,budget
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, choices=SCHEDULABLE_GPUS)
    parser.add_argument("--job", required=True)
    parser.add_argument("--storage-record", required=True)
    parser.add_argument("--expected-storage-sha256", required=True)
    parser.add_argument("--expected-budget-source-sha256", required=True)
    parser.add_argument("--expected-launcher-sha256", required=True)
    parser.add_argument("module")
    parser.add_argument("module_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    def module_arg(name):
        i=args.module_args.index(name)
        return args.module_args[i+1]
    record=dict(path=module_arg("--input"),sha256=module_arg("--expected-input-sha256"))
    _SPEC=contract.read(record)
    contract.active(_SPEC)
    req,caps,runtime=contract.validate_request(_SPEC["root_stage_request_ref"])
    if args.module!=contract.GPU_MAIN or args.gpu!=_SPEC["physical_gpu"] or args.job!=_SPEC["job_id"]:
        parser.error("Only exact current980 postMLM entry/job/root-selected GPU0")
    contract.native_contract().validate_active_input(record,_SPEC,contract.GPU_MAIN,GPU=True)
    self_probe_v1()
    _STORAGE_GUARD,_launch_memo=contract.consumer_guard()
    budget=_STORAGE_GUARD.budget
    sharing=contract.require_gpu0_execution_route_v1()
    if args.gpu!=0 or _SPEC["execution_gpu_uuid"]!=contract.GPU0_UUID:
        raise ValueError("Only current policy-bound GPU0 stage is eligible")
    # Never grant scientific/runtime authority: the bound consumer still validates those original guards.
    lease = None
    child = None
    record = account = None
    selected = None
    lock_dir = ROOT / ".runtime/locks"
    try:
        # Root-owned shared pool already holds gpu_0.lock; never acquire or rewrite it.
        guard=contract.load(contract.GPU0_NATIVE_GUARD_REF,"_shared_native_inherited_GPU0_guard")
        guard.install_before_torch_v1()
        inherited=guard.validate_lease();guard.apply_and_validate_title()
        if (inherited["physical_gpu"]!=0 or inherited["visible_gpu_uuid"]!=contract.GPU0_UUID
                or not inherited["launcher_metadata"]["job"].startswith("gpu0_shared_v1:")
                or os.environ.get("ARGGNN_LOGICAL_JOB_NAME")!=args.job):
            raise ValueError("Exact inherited canonical pool0 lease and separate native logical job required")
        with budget.v4.mutex(ROOT):
            record,account=admitted_storage_locked(args.storage_record,args.expected_storage_sha256,
                args.job,args.expected_budget_source_sha256,args.expected_launcher_sha256)
            if _SPEC["main_source_ref"]!=contract.ref(args.module):
                raise ValueError("Exact separately admitted native producer")
        selected=0
        candidates={0:dict(uuid=contract.GPU0_UUID)}
        lease=os.fdopen(os.dup(inherited["lease_fd"]),"r+")
        env = os.environ.copy()
        env.update({"CUDA_VISIBLE_DEVICES": candidates[selected]["uuid"],
            "ARGGNN_PHYSICAL_GPU": str(selected), "ARGGNN_JOB_NAME": inherited["launcher_metadata"]["job"],
            "ARGGNN_LOGICAL_JOB_NAME": args.job,
            "ARGGNN_GPU_LEASE": str(lock_dir / f"gpu_{selected}.lock"),
            "ARGGNN_GPU_LEASE_FD": str(lease.fileno()),
            "ARGGNN_BUDGET_V9_SOURCE_SHA256": args.expected_budget_source_sha256,
            "ARGGNN_BUDGET_V9_LAUNCHER_SHA256": args.expected_launcher_sha256})
        env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "4")
        env["MKL_NUM_THREADS"] = env.get("MKL_NUM_THREADS", "4")
        command = [str(ROOT / ".runtime/env/bin/python"), "-B", str(ROOT/args.module), *args.module_args]
        print(json.dumps({"event": "launch", "gpu": selected, "uuid": candidates[selected]["uuid"],
                          "job": args.job, "command_module": args.module}), flush=True)
        if _children():raise ValueError("No unrelated launcher children")
        _subreaper();contract.active(_SPEC)
        terminal=_Terminal(_SPEC,"launcher")
        child = subprocess.Popen(command,cwd=ROOT,env=env,start_new_session=True,
            pass_fds=(lease.fileno(),))
        ownership=OwnedTree(child.pid);interrupted=[]
        def stop(signum,_frame):
            interrupted.append(signum)
            ownership.signal_all(signum)
        original_handlers={sig:signal.signal(sig,stop) for sig in (signal.SIGTERM,signal.SIGINT)}
        control_error=None
        try:
            while child.poll() is None:
                ownership.refresh()
                try:contract.active(_SPEC)
                except BaseException as error:control_error=error;break
                if interrupted:control_error=ValueError("Launcher interrupted");break
                time.sleep(.2)
        finally:
            if child.poll() is None:ownership.signal_all(signal.SIGTERM)
            until=time.monotonic()+20.
            while child.poll() is None and time.monotonic()<until:
                ownership.refresh();time.sleep(.1)
            if child.poll() is None:ownership.signal_all(signal.SIGKILL)
            returncode=child.wait()  # actual direct wait first, durable before drain/postprocessing
            try:
                print(json.dumps(dict(event="actual_direct_entry_wait_v9",pid=child.pid,returncode=returncode,
                    actual_direct_supervisor_wait=True,drain_state="not_yet_certified"),separators=(",",":")),flush=True)
            except BaseException as error:control_error=control_error or error
            try:
                terminal.event(0,dict(schema="native_MLM_genuine_entry_supervisor_wait_v9",pid=child.pid,
                    returncode=returncode,actual_direct_supervisor_wait=True,drain_state="not_yet_certified"))
            except BaseException as error:
                control_error=control_error or error  # Continue real drain even if evidence storage fails.
            while ownership.live() or _children():
                ownership.signal_all(signal.SIGKILL)
                while True:
                    try:pid,status=os.waitpid(-1,os.WNOHANG)
                    except ChildProcessError:break
                    if pid==0:break
                if ownership.live() or _children():time.sleep(.1)
            terminal.event(1,dict(schema="native_MLM_actual_launcher_drain_v9",pid=child.pid,returncode=returncode,
                owned_identities_drained=ownership.identities_absent(),no_adopted_live_writer=not _children(),
                owned_identity_drain_proof=ownership.evidence()))
            terminal.close()
            if not ownership.identities_absent():raise ValueError("Owned launcher descendants not drained")
            for sig,old in original_handlers.items():signal.signal(sig,old)
        if control_error is not None:raise control_error
        print(json.dumps({"event": "supervisor_wait", "worker_pid": child.pid,
                          "returncode": returncode, "wait_status_known": True}), flush=True)
        save_wait_evidence_locked(record, account, child, returncode,
            args.expected_budget_source_sha256, args.expected_launcher_sha256)
        return returncode
    finally:
        if lease is not None:
            lease.close()

if __name__ == "__main__":
    raise SystemExit(main())
