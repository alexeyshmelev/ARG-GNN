"""Separate continuation controller: proportional RAM and owned-child reaping.

The v3 files stay immutable. Bindings below affect only this new process. GPU,
lease, storage, resource limits and scientific callables are inherited exactly.
"""
from __future__ import annotations
import copy
import ctypes
import errno
import os
from pathlib import Path
import signal
import stat
import subprocess
import time
import types

from . import contracts as c
from . import workflow as base

MODULE="arggnn.benchmarking.workflow_v4"
_BASE_POLICY=base.current_policy
_BASE_EXECUTE=base.execute
_BASE_SAVE=base.save
_BASE_CHILD=base.child
_BASE_RUN=base.run
_BASE_EXTERNAL_RESULT=base.external_result
_BASE_EXTERNAL_RESOURCES=base.external_resources
_BASE_STORAGE=base.storage_projection
_BASE_AVAILABLE=base.available_ram_bytes
_RUN_CONTEXT=None
_CHILD_MEMORY_PATH=None
_GROUPS={}
LIMIT=1024*1024


def need(ok,message):
    if not ok:raise RuntimeError(message)


def current_policy():
    canonical=_BASE_POLICY()
    approved=canonical.get("approved_continuation_ref")
    need(isinstance(approved,dict) and set(approved)=={"path","sha256"},
         "Exact canonical approved_continuation_ref required for v4")
    policy=dict(canonical)
    policy["canonical_parent_workflow_ref"]=canonical.get("approved_workflow_ref")
    policy["approved_workflow_ref"]=approved
    if "continuation_drain_ref" in canonical:
        policy["workflow_drain_ref"]=canonical["continuation_drain_ref"]
    return policy


def _read(path):
    with Path(path).open("rb") as stream:raw=stream.read(LIMIT+1)
    need(len(raw)<=LIMIT,"Owned process metadata exceeds bounded read")
    return raw.decode()


def identity(pid):
    fields=_read(Path("/proc")/str(pid)/"stat").rsplit(")",1)[1].split()
    return dict(pid=pid,start_ticks=int(fields[19]),pgid=int(fields[2]),session=int(fields[3]),
                ppid=int(fields[1]),state=fields[0])


def same(actual,expected):
    return all(actual[key]==expected[key] for key in ("pid","start_ticks","pgid","session"))


def group_pids(pgid):
    members=set()
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():continue
        pid=int(path.name)
        try:group=os.getpgid(pid)
        except ProcessLookupError:continue
        if group==pgid:members.add(pid)
    return members


def scope_bytes(output):
    """Same charged-byte rule, tolerating only genuinely vanished native entries."""
    output=c.contained_path(output)
    need(output.resolve()==output and not output.is_symlink(),"Canonical owned output scope required")
    total=0
    def walk_error(exc):
        if not isinstance(exc,(FileNotFoundError,ProcessLookupError)):raise exc
    for directory,dirs,files in os.walk(output,followlinks=False,onerror=walk_error):
        for name in dirs+files:
            path=Path(directory)/name
            try:
                info=path.lstat()
                if stat.S_ISLNK(info.st_mode):
                    need(path.resolve().is_relative_to(output),"Native output symlink escaped assigned scope")
            except (FileNotFoundError,ProcessLookupError):continue
            total+=max(info.st_size,info.st_blocks*512)+4096
    return total


def enable_subreaper():
    libc=ctypes.CDLL(None,use_errno=True)
    need(libc.prctl(36,1,0,0,0)==0,"Linux PR_SET_CHILD_SUBREAPER failed")
    value=ctypes.c_int()
    need(libc.prctl(37,ctypes.byref(value),0,0,0)==0 and value.value==1,
         "Linux subreaper must be actually enabled")
    return True


class OwnedGroup:
    def __init__(self,pid):
        root=identity(pid)
        need(root["pgid"]==root["session"]==pid,"Actual Popen must own its fresh session/group")
        self.root=root
        self.history={pid:root}
        self.latest=None
        self.peak_accounted_bytes=0
        self.peak_summed_RSS_bytes=0
        self.samples=0
        self.fallback_samples=0
        self.adopted_waits=[]
        self.direct_wait=None
        self.cleanup_errors=[]

    def capture(self):
        members={}
        for pid in group_pids(self.root["pgid"]):
            try:row=identity(pid)
            except (FileNotFoundError,ProcessLookupError):continue
            need(row["pgid"]==self.root["pgid"] and row["session"]==self.root["session"],
                 "Owned session/group changed")
            need(row["start_ticks"]>=self.root["start_ticks"],"Group member predates the actual owned root")
            if pid in self.history:need(same(row,self.history[pid]),"Owned PID was reused")
            else:self.history[pid]=row
            members[pid]=row
        return members

    def measure(self):
        rows={}
        for attempt in range(8):
            members=self.capture()
            for pid,before in members.items():
                try:
                    if before["state"]=="Z":
                        row=dict(accounted_bytes=0,rss_bytes=0,method="zombie_zero_resident")
                    else:
                        try:
                            memory={}
                            for line in _read(Path("/proc")/str(pid)/"smaps_rollup").splitlines():
                                fields=line.split()
                                if fields and fields[0] in ("Pss:","Rss:"):
                                    need(len(fields)==3 and fields[2]=="kB","Unknown proportional memory unit")
                                    memory[fields[0][:-1]]=int(fields[1])*1024
                            need(set(memory)=={"Pss","Rss"} and 0<=memory["Pss"]<=memory["Rss"],
                                 "Incomplete proportional memory receipt")
                            row=dict(accounted_bytes=memory["Pss"],rss_bytes=memory["Rss"],method="smaps_rollup_PSS")
                        except (OSError,ValueError,RuntimeError) as exc:
                            rss=int(_read(Path("/proc")/str(pid)/"statm").split()[1])*os.sysconf("SC_PAGE_SIZE")
                            row=dict(accounted_bytes=rss,rss_bytes=rss,method="conservative_RSS_fallback",
                                     fallback_reason=type(exc).__name__)
                    after=identity(pid)
                    need(same(after,before),"Owned PID/start changed during memory measurement")
                    rows[pid]=dict(before,**row,state_after=after["state"])
                except (FileNotFoundError,ProcessLookupError):
                    rows.pop(pid,None)  # Actual vanished PID, never an unreadable live zero.
            after_members=self.capture()
            if set(after_members)<=set(rows):break
        else:raise RuntimeError("Owned group membership did not stabilize for proportional accounting")
        rows={pid:row for pid,row in rows.items() if pid in after_members}
        total=sum(row["accounted_bytes"] for row in rows.values())
        rss=sum(row["rss_bytes"] for row in rows.values())
        fallback=any(row["method"]=="conservative_RSS_fallback" for row in rows.values())
        self.samples+=1;self.fallback_samples+=int(fallback)
        self.peak_accounted_bytes=max(self.peak_accounted_bytes,total)
        self.peak_summed_RSS_bytes=max(self.peak_summed_RSS_bytes,rss)
        self.latest=dict(accounted_bytes=total,summed_RSS_bytes=rss,conservative_RSS_fallback=fallback,
                         members=list(rows.values()))
        return total

    def _signal(self,sig):
        members=self.capture()
        if not members:return
        # The original Popen session identity, not a historical/unidentified PGID.
        for row in members.values():
            try:current=identity(row["pid"])
            except (FileNotFoundError,ProcessLookupError):continue
            need(same(current,row),"Owned identity changed before group signal")
        try:os.killpg(self.root["pgid"],sig)
        except ProcessLookupError:pass

    def _reap_adopted(self,direct_pid):
        # Never waitpid(-1), and never steal the Popen child's actual wait.
        for pid,row in self.capture().items():
            if pid==direct_pid or row["ppid"]!=os.getpid():continue
            need(same(identity(pid),row),"Owned adopted-child identity changed before wait")
            try:waited,status=os.waitpid(pid,os.WNOHANG)
            except ChildProcessError:continue
            if waited:
                self.adopted_waits.append(dict(identity=row,actual_waitpid=True,
                    raw_wait_status=status,actual_returncode=os.waitstatus_to_exitcode(status)))

    def stop_and_wait(self,process):
        need(process.pid==self.root["pid"],"Cleanup must retain the actual direct Popen")
        for sig,timeout in ((signal.SIGTERM,5),(signal.SIGKILL,5)):
            try:self._signal(sig)
            except Exception as exc:
                self.cleanup_errors.append(type(exc).__name__+": "+str(exc))
                # Actual Popen methods retain the direct-child identity/returncode.
                if process.poll() is None:
                    (process.terminate if sig==signal.SIGTERM else process.kill)()
            try:code=process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:continue
            self.direct_wait=dict(pid=process.pid,actual_Popen_wait=True,actual_returncode=code)
            end=time.monotonic()+timeout
            while time.monotonic()<end:
                try:self._reap_adopted(process.pid)
                except Exception as exc:
                    self.cleanup_errors.append(type(exc).__name__+": "+str(exc));break
                if not group_pids(self.root["pgid"]):return
                time.sleep(.02)
        # Preserve real direct wait and cleanup evidence even when a group fails
        # to drain. The inherited supervisor records groupabsent=false honestly.
        if self.direct_wait is None:
            code=process.wait()
            self.direct_wait=dict(pid=process.pid,actual_Popen_wait=True,actual_returncode=code)
        try:self._reap_adopted(process.pid)
        except Exception as exc:self.cleanup_errors.append(type(exc).__name__+": "+str(exc))

    def receipt(self):
        return dict(schema="workflow_v4_owned_group_supervision_v1",root_identity=self.root,
            memory_metric="sum_owned_smaps_rollup_PSS_with_conservative_RSS_fallback",
            memory_limit_field="max_rss_bytes",peak_accounted_bytes=self.peak_accounted_bytes,
            peak_summed_RSS_bytes=self.peak_summed_RSS_bytes,memory_samples=self.samples,
            fallback_samples=self.fallback_samples,last_memory_sample=self.latest,
            owned_identity_history=list(self.history.values()),direct_Popen_wait=self.direct_wait,
            actual_adopted_child_waits=self.adopted_waits,cleanup_errors=self.cleanup_errors)


def group_pss(pgid):
    tracker=_GROUPS.get(pgid)
    if tracker is None:
        # This boundary is called inside the actual newly created scientific group.
        need(pgid==os.getpgrp() and os.getsid(0)==pgid,"Worker must measure only its own fresh session")
        tracker=_GROUPS[pgid]=OwnedGroup(pgid)
    measured=tracker.measure()
    if _CHILD_MEMORY_PATH is not None:
        _BASE_SAVE(_CHILD_MEMORY_PATH,tracker.receipt())
    return measured


def external_result(task):
    if "workflow_predecessor" in task:
        from . import workflow_predecessor_v4 as predecessor
        return predecessor.external_result(task)
    return _BASE_EXTERNAL_RESULT(task)


def external_resources(plan):
    cpu,rss=_BASE_EXTERNAL_RESOURCES(plan)
    if plan.get("workflow_external_predecessors"):
        from . import workflow_predecessor_v4 as predecessor
        extra_cpu,extra_rss=predecessor.external_resources(plan)
        cpu+=extra_cpu;rss+=extra_rss
    return cpu,rss+plan.get("prior_unknown_rss_bytes",0)


def storage_projection(plan,state,new_cap=0):
    total=_BASE_STORAGE(plan,state,new_cap)
    if plan.get("workflow_external_predecessors"):
        from . import workflow_predecessor_v4 as predecessor
        total+=predecessor.storage_reservation_bytes(plan)
    return total


def _known_credit(plan,state_path):
    bindings=[]
    if plan.get("workflow_external_predecessors"):
        from . import workflow_predecessor_v4 as predecessor
        bindings.extend(predecessor.live_group_bindings(plan))
    if state_path.exists():
        state=c.read_json(state_path)
        for task in plan["tasks"]:
            entry=state["tasks"][task["id"]]
            if task["resource"]=="external" or entry["status"]!="running":continue
            path=c.contained_path(task["output_directory"])/"supervision_identity.json"
            if not path.exists():continue  # Not yet spawned: reserve its full growth.
            receipt=c.read_json(path)
            need(receipt.get("job_ref")==entry.get("job_ref"),"Headroom identity must match actual current job")
            bindings.append(dict(receipt["child_identity"],max_rss_bytes=task["max_rss_bytes"]))
    credit=0;seen=set()
    for binding in bindings:
        try:
            pid=binding["pid"]
            need(pid not in seen,"Never credit one live group twice");seen.add(pid)
            root=identity(pid)
            need(same(root,binding) and root["state"]!="Z","Live headroom identity changed")
            tracker=OwnedGroup(pid)
            need(same(tracker.root,binding),"Live leader changed before PSS admission sample")
            tracker.measure()
            # RSS fallback may overcount shared pages. It earns ZERO credit.
            known=sum(row["accounted_bytes"] for row in tracker.latest["members"]
                      if row["method"]=="smaps_rollup_PSS")
            credit+=min(binding["max_rss_bytes"],known)
        except (OSError,RuntimeError,KeyError,ValueError):
            continue  # Unknown/unreadable or finished groups retain full reservation.
    return credit


def available_ram_bytes():
    if _RUN_CONTEXT is None:return _BASE_AVAILABLE()
    plan,state_path=_RUN_CONTEXT
    first=_known_credit(plan,state_path)
    available=_BASE_AVAILABLE()
    second=_known_credit(plan,state_path)
    credit=min(first,second)
    _BASE_SAVE(state_path.parent/"host_headroom_observation.json",dict(
        schema="workflow_v4_outstanding_RAM_growth_admission_v1",kernel_MemAvailable_bytes=available,
        known_live_PSS_credit_bytes=credit,first_known_PSS_credit_bytes=first,
        second_known_PSS_credit_bytes=second,adjusted_admission_available_bytes=available+credit,
        full_aggregate_reservations_unchanged=True,unknown_groups_receive_zero_credit=True))
    return available+credit


def child(ref):
    global _CHILD_MEMORY_PATH
    job=c.read_ref(ref)
    _CHILD_MEMORY_PATH=c.contained_path(job["task"]["output_directory"])/"scientific_memory_observation.json"
    return _BASE_CHILD(ref)


def install_bindings():
    base.MODULE=MODULE
    base.current_policy=current_policy
    base._own_group_rss=group_pss
    base.scope_bytes=scope_bytes
    base.execute=execute
    base.child=child
    base.run=run
    base.external_result=external_result
    base.external_resources=external_resources
    base.storage_projection=storage_projection
    base.available_ram_bytes=available_ram_bytes


def execute(ref):
    install_bindings()
    enable_subreaper()
    holder={}
    def spawn(*args,**kwargs):
        need(kwargs.get("start_new_session") is True,"Scientific process must own a new group")
        read_fd,write_fd=os.pipe()
        process=None
        try:
            kwargs["env"]=dict(kwargs.get("env",os.environ),ARGGNN_V4_STARTUP_FD=str(read_fd))
            kwargs["pass_fds"]=tuple(kwargs.get("pass_fds",()))+(read_fd,)
            process=subprocess.Popen(*args,**kwargs)
            os.close(read_fd);read_fd=None
            holder["process"]=process
            tracker=holder["tracker"]=OwnedGroup(process.pid)
            job=c.read_ref(ref)
            _BASE_SAVE(c.contained_path(job["task"]["output_directory"])/"supervision_identity.json",
                dict(schema="workflow_v4_Popen_identity_v1",job_ref=ref,child_identity=tracker.root,
                     scientific_entry_released_only_after_durable_identity=True),exclusive=True)
            os.write(write_fd,b"R")
            return process
        except BaseException as exc:
            # EOF closes the pre-entry barrier. No scientific callable may start
            # before its actual Popen identity has been durably recorded.
            if write_fd is not None:os.close(write_fd);write_fd=None
            if process is not None:
                tracker=holder.get("tracker")
                if tracker is not None:tracker.stop_and_wait(process)
                else:
                    if process.poll() is None:process.terminate()
                    try:process.wait(timeout=5)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
                code=process.wait()
                try:os.killpg(process.pid,0);absent=False
                except ProcessLookupError:absent=True
                job=c.read_ref(ref)
                _BASE_SAVE(c.contained_path(job["task"]["output_directory"])/"actual_wait.json",dict(
                    actual_direct_wait=True,actual_direct_returncode=code,owned_process_group_absent=absent,
                    supervisor_error="Pre-entry lifecycle receipt failed: "+type(exc).__name__+": "+str(exc),
                    output_bytes=base.scope_bytes(c.contained_path(job["task"]["output_directory"])),
                    job_ref=ref,automatic_retry=False,scientific_entry_barrier_not_released=True,
                    supervision=tracker.receipt() if tracker is not None else dict(child_pid=process.pid)))
            raise
        finally:
            if read_fd is not None:os.close(read_fd)
            if write_fd is not None:os.close(write_fd)
    def memory(pgid):
        tracker=holder["tracker"]
        need(pgid==tracker.root["pgid"],"Supervisor may measure only its own Popen group")
        return tracker.measure()
    def stop(process):
        need(process is holder["process"],"Actual owned Popen object required")
        holder["tracker"].stop_and_wait(process)
    def save(path,value,**kwargs):
        if Path(path).name=="actual_wait.json":
            supervision=holder["tracker"].receipt()
            supervision["linux_subreaper_enabled"]=True
            value=dict(value,supervision=supervision)
        return _BASE_SAVE(path,value,**kwargs)
    namespace=dict(_BASE_EXECUTE.__globals__,MODULE=MODULE,current_policy=current_policy,
                   subprocess=types.SimpleNamespace(Popen=spawn,STDOUT=subprocess.STDOUT),_own_group_rss=memory,
                   _stop_group=stop,save=save,scope_bytes=scope_bytes)
    owned=types.FunctionType(_BASE_EXECUTE.__code__,namespace,_BASE_EXECUTE.__name__,
                             _BASE_EXECUTE.__defaults__,_BASE_EXECUTE.__closure__)
    return owned(ref)


def run(ref,**kwargs):
    global _RUN_CONTEXT
    install_bindings()
    plan=c.read_ref(ref)
    if kwargs.get("execute_jobs",False):
        from .workflow_reporting_v4 import ProgressReporter
        base.ProgressReporter=ProgressReporter
    _RUN_CONTEXT=(plan,c.contained_path(plan["records_directory"])/"state.json")
    return _BASE_RUN(ref,**kwargs)


def await_startup():
    value=os.environ.pop("ARGGNN_V4_STARTUP_FD",None)
    if value is None:return
    fd=int(value)
    try:released=os.read(fd,1)
    finally:os.close(fd)
    need(released==b"R","Scientific entry blocked: parent lifecycle identity was not recorded")


def main():
    await_startup()
    install_bindings()
    base.main()


if __name__=="__main__":main()
