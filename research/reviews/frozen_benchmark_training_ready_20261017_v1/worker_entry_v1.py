"""Concrete inherited-gate TRAIN001 worker using unchanged simulator and Writer."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import resource
import sys
import time
from . import train001_simulate_context_v1 as context


def worker(record,gate_fd):
    c,_=context.initialize()
    c.require(os.read(gate_fd,1)==b"G","No scientific work before actual owned launch gate")
    os.close(gate_fd)
    s,q,cfg=c.input_spec(record)
    c.require(os.environ.get("CUDA_VISIBLE_DEVICES")==""and os.getpid()==os.getpgrp()==os.getsid(0),
        "Actual owned CPU-only session")
    c.require(sys.dont_write_bytecode and sys.flags.no_user_site
        and Path(sys.prefix)==c.ROOT/".runtime/env"
        and set(os.sched_getaffinity(0))==set(q["cpu_ids"]),"Exact actual CPU interpreter/affinity")
    fd=int(os.environ["ARGGNN_C3_PREPARATION_SLOT_FD"]);identity=os.fstat(fd)
    locks=[c.ROOT/context.OLD/("cpu_slot_"+str(i)+".lock")for i in range(2)]
    paths=[p for p in locks if(p.lstat().st_dev,p.lstat().st_ino)==(identity.st_dev,identity.st_ino)]
    c.require(len(paths)==1 and identity.st_size==0,"Exact inherited original shared two-producer slot")
    with paths[0].open("r+")as probe:
        try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:pass
        else:
            fcntl.flock(probe,fcntl.LOCK_UN)
            raise ValueError("Original shared slot is not genuinely held")
    g=c.guard(True)
    started=time.monotonic()
    def resource_check(event):
        c.current(s,150)
        c.require(time.monotonic()-started<c.CAPS["wall_seconds"]-300,
            "Original finite scientific cutoff")
        c.require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=c.CAPS["rss_bytes"],
            "Actual worker RSS; unchanged outer additionally enforces all owned RSS")
        c.require(os.environ.get("CUDA_VISIBLE_DEVICES")==""and set(os.sched_getaffinity(0))==set(q["cpu_ids"]),
            "Actual CPU-only affinity remains")
        return dict(event=event)
    # The worker is the complete actual context. No absent factory is required.
    from . import american_full30mb_simulator_bridge_v1 as bridge
    adapted=dict(s,source_ref=c.ref(context.BRIDGE),config_ref=s["refs"]["config"])
    return bridge.execute_in_admitted_context_v1(s["refs"]["config"],
        spec=adapted,current_guard=g,current_guard_source_ref=c.GUARD_REF,
        resource_check=resource_check)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--worker-ref");p.add_argument("--gate-fd",type=int)
    a=p.parse_args(argv)
    if not a.worker_ref:
        print(json.dumps(dict(status="SOURCE_ONLY_INACTIVE",execution=False)));return 0
    context.initialize()[0].require(a.gate_fd is not None,"Actual inherited gate required")
    result=worker(json.loads(a.worker_ref),a.gate_fd)
    print(json.dumps(dict(completed_ref=result)),flush=True);return 0
if __name__=="__main__":raise SystemExit(main())
