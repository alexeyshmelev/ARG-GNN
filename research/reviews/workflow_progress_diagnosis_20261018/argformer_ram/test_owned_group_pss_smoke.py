"""Tiny CPU-only fork fixture for the unpinned proportional RAM candidate."""
from pathlib import Path
import json
import mmap
import os
import subprocess
import sys
import unittest
from unittest.mock import patch
import owned_group_pss_candidate as candidate

BASE=Path(__file__).resolve().parent
SHARED=16*2**20
PRIVATE=8*2**20
CHILDREN=2


def payload():
    # This is a newly created test session, never any production process.
    shared=mmap.mmap(-1,SHARED,flags=mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS)
    for offset in range(0,SHARED,4096):shared[offset]=1
    commands=[os.pipe() for _ in range(CHILDREN)];ack_read,ack_write=os.pipe();pids=[]
    for own_read,_ in commands:
        pid=os.fork()
        if pid==0:
            os.close(ack_read)
            for read,write in commands:
                os.close(write)
                if read!=own_read:os.close(read)
            os.write(ack_write,b"R")
            private=None
            while True:
                command=os.read(own_read,1)
                if command==b"A":
                    private=mmap.mmap(-1,PRIVATE,flags=mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS)
                    for offset in range(0,PRIVATE,4096):private[offset]=2
                    os.write(ack_write,b"A")
                elif command in (b"S",b""):break
            os._exit(0)
        pids.append(pid)
    os.close(ack_write)
    for read,_ in commands:os.close(read)
    def acknowledgements():
        value=b""
        while len(value)<CHILDREN:
            part=os.read(ack_read,CHILDREN-len(value))
            if not part:raise RuntimeError("Synthetic child exited without acknowledgement")
            value+=part
    acknowledgements()
    print(json.dumps(dict(pgid=os.getpid(),pids=[os.getpid(),*pids])),flush=True)
    if sys.stdin.readline().strip()=="ALLOCATE":
        for _,write in commands:os.write(write,b"A")
        acknowledgements();print("ALLOCATED",flush=True)
    if sys.stdin.readline().strip()!="STOP":raise RuntimeError("Expected synthetic stop protocol")
    for _,write in commands:os.write(write,b"S");os.close(write)
    waits=[dict(pid=pid,wait_status=os.waitpid(pid,0)[1]) for pid in pids]
    print(json.dumps(dict(actual_child_waits=waits)),flush=True)


class ProportionalForkSmoke(unittest.TestCase):
    def test_shared_memory_private_growth_unreadable_and_identity_failclosed(self):
        environment=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1")
        process=subprocess.Popen([sys.executable,"-B","-s",str(Path(__file__).resolve()),"--payload"],
            start_new_session=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
            text=True,env=environment)
        try:
            identity=json.loads(process.stdout.readline())
            self.assertEqual(identity["pgid"],process.pid)
            inventory=candidate.capture_owned_group(identity["pgid"],identity["pids"])
            before=candidate.measure_owned_group(inventory)
            # Three mappings share the same original16MiB physical pages.
            self.assertGreater(before["summed_RSS_bytes"]-before["accounted_bytes"],
                               int((CHILDREN*SHARED)*.75))
            process.stdin.write("ALLOCATE\n");process.stdin.flush()
            self.assertEqual(process.stdout.readline().strip(),"ALLOCATED")
            after=candidate.measure_owned_group(inventory)
            growth=after["accounted_bytes"]-before["accounted_bytes"]
            self.assertGreater(growth,int(CHILDREN*PRIVATE*.90))
            self.assertLess(growth,CHILDREN*PRIVATE+8*2**20)
            self.assertEqual(after["owned_inventory"],inventory)
            with patch.object(candidate,"_pss_bytes",side_effect=PermissionError("fixture denied")):
                with self.assertRaisesRegex(candidate.AccountingError,"fails closed"):
                    candidate.measure_owned_group(inventory,unreadable="fail")
                fallback=candidate.measure_owned_group(inventory,unreadable="rss")
            self.assertTrue(fallback["conservative_fallback_used"])
            self.assertEqual(fallback["accounted_bytes"],fallback["summed_RSS_bytes"])
            self.assertGreaterEqual(fallback["accounted_bytes"],after["accounted_bytes"])
            broken=json.loads(json.dumps(inventory));broken["members"][str(process.pid)]["start_ticks"]+=1
            with self.assertRaisesRegex(candidate.AccountingError,"PID/start/session"):
                candidate.measure_owned_group(broken)
            incomplete=json.loads(json.dumps(inventory));incomplete["members"].pop(str(identity["pids"][-1]))
            with self.assertRaisesRegex(candidate.AccountingError,"membership changed"):
                candidate.measure_owned_group(incomplete)
            process.stdin.write("STOP\n");process.stdin.flush()
            drained=json.loads(process.stdout.readline())
            self.assertTrue(all(row["wait_status"]==0 for row in drained["actual_child_waits"]))
            self.assertEqual(process.wait(timeout=5),0)
            result=dict(schema="tiny_owned_fork_PSS_smoke_v1",successful=True,CUDA_visible_devices="",
                imported_torch=False,scientific_functions_called=0,shared_bytes=SHARED,
                private_bytes_per_child=PRIVATE,children=CHILDREN,owned_inventory=inventory,
                before=before,after=after,actual_proportional_growth_bytes=growth,
                expected_private_growth_bytes=CHILDREN*PRIVATE,
                RSS_fallback_checked=True,unreadable_failclosed_checked=True,
                PID_start_mismatch_rejected=True,untracked_member_rejected=True,
                synthetic_actual_waits=drained,synthetic_parent_wait=process.returncode,
                production_memory_or_command_metadata_read=False,production_processes_signaled=False,
                kernel_PGID_lookups_only_for_group_completeness=True)
            (BASE/"smoke_result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
            print(json.dumps({k:result[k] for k in ("successful","actual_proportional_growth_bytes",
                "expected_private_growth_bytes","shared_bytes","children")},sort_keys=True),flush=True)
        finally:
            if process.poll() is None:
                try:process.stdin.write("STOP\n");process.stdin.flush();process.wait(timeout=5)
                except (BrokenPipeError,OSError,subprocess.TimeoutExpired):
                    process.terminate();process.wait(timeout=5)
            for stream in (process.stdin,process.stdout,process.stderr):stream.close()


if __name__=="__main__":
    if "--payload" in sys.argv:payload()
    else:unittest.main(verbosity=2)
