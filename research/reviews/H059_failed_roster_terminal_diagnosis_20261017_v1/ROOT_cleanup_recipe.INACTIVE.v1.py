"""INACTIVE ROOT-only cleanup function. No action on import or script invocation.
Uses pinned existing pidfd backend. ROOT must review, invoke, and record its result.
Original tool34504 direct wait and historical drain remain UNKNOWN.
"""
import hashlib,importlib.util,json,os,select,signal,time
from pathlib import Path
BASE=Path("/workspace-SR003.nfs2/ARG-GNN")
INPUT="research/reviews/European_outer000_ROOT_actual_Relate64_20261017_v1/input.json"
INPUT_SHA="3561b5898cb6abc553b2a8bff4528baf02f6a2a02c416d9c696f7355376b7010"
LAUNCH="research/reviews/European_outer000_ROOT_actual_Relate64_20261017_v1/worker_aux/launch.json"
LAUNCH_SHA="a63b32e86b796355adcf2234f0bc90748cb8ec186ebd2383c58c53633e935c04"
BACKEND="research/reviews/American_C3_native_MLM_full_source_v5/pidfd_backend_v1.py"
BACKEND_SHA="61be5d89357d1b95611e0ba26dd39d381ececf3ce6a36c347507cdc062373eca"
ROOT_PID=3998053
ROOT_START=488962157
UID=1000
MAX_IDENTITIES=4096

def _read(pid):
    try:
        proc=Path(f"/proc/{pid}")
        s=(proc/"stat").read_text().rsplit(")",1)[1].split()
        uids=tuple(map(int,next(v for v in (proc/"status").read_text().splitlines() if v.startswith("Uid:")).split()[1:]))
        return dict(pid=pid,state=s[0],ppid=int(s[1]),group=int(s[2]),session=int(s[3]),start=int(s[19]),uid=proc.stat().st_uid,uids=uids)
    except (FileNotFoundError,ProcessLookupError):return None

def _same(a,b):
    return a is not None and b is not None and all(a[k]==b[k] for k in ("pid","start","uid","uids"))

def cleanup_root_only(*,explicit_root_authorization,cleanup_seconds=30):
    if explicit_root_authorization != "ROOT_H059_exact3998053_488962157_current_cleanup_only":
        raise ValueError("Explicit ROOT current-cleanup directive required")
    if cleanup_seconds != 30:raise ValueError("Exact finite30second cleanup envelope")
    if os.getuid()!=UID:raise ValueError("Exact UID1000 required")
    if not __import__("sys").dont_write_bytecode:raise ValueError("Use Python -B")
    began=time.monotonic();end=began+cleanup_seconds;freeze_end=began+10
    stamp=lambda:__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
    for p,sha in ((INPUT,INPUT_SHA),(LAUNCH,LAUNCH_SHA),(BACKEND,BACKEND_SHA)):
        if hashlib.sha256((BASE/p).read_bytes()).hexdigest()!=sha:
            raise ValueError("Changed exact operational pin: "+p)
    launch=json.loads((BASE/LAUNCH).read_text())
    if launch["pid"]!=ROOT_PID or launch["ownership"]["owned_identities"]!=[dict(pid=ROOT_PID,start=ROOT_START)]:
        raise ValueError("Changed launch root identity")
    spec=importlib.util.spec_from_file_location("_H059_cleanup_pinned_pidfd",BASE/BACKEND)
    backend=importlib.util.module_from_spec(spec);spec.loader.exec_module(backend)
    held=[];by_pid={};edges=[];events=[];errors=[];groups={ROOT_PID};freeze_complete=False

    def event(kind,**kw):events.append(dict(utc=stamp(),kind=kind,**kw))
    def check_clock(deadline):
        if time.monotonic()>=deadline:raise TimeoutError("Finite cleanup deadline")

    def send(row,sig):
        # pidfd cannot retarget PID reuse; identity and UID rechecked before send.
        actual=_read(row["pid"])
        if not _same(actual,row):
            event("identity_absent_or_changed_no_signal",pid=row["pid"],start=row["start"],actual=actual);return False
        try:backend.pidfd_send_signal_v1(row["_fd"],sig,None,0)
        except ProcessLookupError:
            event("pidfd_esrch",pid=row["pid"],start=row["start"],signal=int(sig));return False
        event("pidfd_signal",pid=row["pid"],start=row["start"],signal=int(sig));return True

    def hold_and_freeze(expected,depth,parent=None):
        check_clock(freeze_end)
        if len(held)>=MAX_IDENTITIES:raise ValueError("4096current cleanup identities exceeded")
        before=_read(expected["pid"])
        if not _same(before,expected):return None
        if before["uid"]!=UID or before["uids"]!=(UID,)*4:raise ValueError("UID mismatch")
        if parent is not None:
            if not _same(_read(parent["pid"]),parent) or before["ppid"]!=parent["pid"]:
                raise ValueError("Current frozen-parent ancestry mismatch")
        try:fd=backend.pidfd_open_v1(before["pid"],0)
        except ProcessLookupError:return None
        after=_read(before["pid"])
        if not _same(after,before):
            os.close(fd);return None
        if parent is not None and (after["ppid"]!=parent["pid"] or not _same(_read(parent["pid"]),parent)):
            os.close(fd);raise ValueError("Ancestry changed across pidfd open")
        row=dict(after,_fd=fd,depth=depth)
        held.append(row);by_pid[row["pid"]]=row;groups.add(row["group"])
        event("authenticated_open_pidfd",**{k:v for k,v in row.items() if not k.startswith("_")})
        if parent is not None:
            edges.append(dict(parent=dict(pid=parent["pid"],start=parent["start"],uid=UID),child=dict(pid=row["pid"],start=row["start"],uid=UID),pidfd_open_revalidated=True))
        send(row,signal.SIGSTOP)
        while True:
            check_clock(freeze_end)
            current=_read(row["pid"])
            if not _same(current,row) or current["state"] in ("Z","X"):return row
            states=[]
            try:
                for task in Path(f"/proc/{row['pid']}/task").iterdir():
                    try:states.append((task/"stat").read_text().rsplit(")",1)[1].split()[0])
                    except (FileNotFoundError,ProcessLookupError):pass
            except (FileNotFoundError,ProcessLookupError):return row
            if states and all(s in ("T","t","Z","X") for s in states):
                event("all_current_tasks_stopped",pid=row["pid"],start=row["start"],task_count=len(states));return row
            time.sleep(.005)

    try:
        root=_read(ROOT_PID)
        if root is None:raise ValueError("Root already absent; historical descendant completeness not established by root absence")
        if root["start"]!=ROOT_START or root["uid"]!=UID or root["uids"]!=(UID,)*4 or root["group"]!=ROOT_PID or root["session"]!=ROOT_PID:
            raise ValueError("Exact original root identity/group/UID mismatch")
        command=Path(f"/proc/{ROOT_PID}/cmdline").read_bytes().replace(b"\0",b" ").decode()
        if INPUT not in command or INPUT_SHA not in command or "--worker-ref" not in command:
            raise ValueError("Exact original worker command mismatch")
        root=hold_and_freeze(root,0)
        if root is None:raise ValueError("Root vanished before freeze")
        queue=[root]
        while queue:
            check_clock(freeze_end)
            parent=queue.pop()
            before=_read(parent["pid"])
            if not _same(before,parent):
                raise ValueError("Owned identity vanished while building freeze tree; do not claim completeness")
            children=set()
            for task in Path(f"/proc/{parent['pid']}/task").iterdir():
                try:children.update(map(int,(task/"children").read_text().split()))
                except (FileNotFoundError,ProcessLookupError):pass
            if not _same(_read(parent["pid"]),parent):
                raise ValueError("Parent changed around children enumeration")
            for pid in sorted(children):
                if pid in by_pid:continue
                current=_read(pid)
                if current is None:continue
                if current["ppid"]!=parent["pid"] or current["uid"]!=UID or current["uids"]!=(UID,)*4:
                    raise ValueError("Unauthenticated current child edge")
                child=hold_and_freeze(current,parent["depth"]+1,parent)
                if child is not None:queue.append(child)
        freeze_complete=True
    except BaseException as exc:
        errors.append(dict(phase="freeze",error=type(exc).__name__+": "+str(exc)))
    finally:
        # Even a bounded freeze failure must not strand authenticated stopped jobs.
        ordered=sorted(held,key=lambda r:r["depth"],reverse=True)
        for row in ordered:
            for sig in (signal.SIGTERM,signal.SIGCONT):
                try:send(row,sig)
                except BaseException as exc:errors.append(dict(phase="term_cont",pid=row["pid"],error=type(exc).__name__+": "+str(exc)))
        grace_end=min(time.monotonic()+10,end-2)
        while time.monotonic()<grace_end:
            try:live=[r for r in held if _same(_read(r["pid"]),r) and _read(r["pid"])["state"] not in ("Z","X")]
            except BaseException as exc:
                errors.append(dict(phase="grace",error=type(exc).__name__+": "+str(exc)));break
            if not live:break
            time.sleep(.02)
        for row in ordered:
            try:
                actual=_read(row["pid"])
                if _same(actual,row) and actual["state"] not in ("Z","X"):send(row,signal.SIGKILL)
            except BaseException as exc:errors.append(dict(phase="kill",pid=row["pid"],error=type(exc).__name__+": "+str(exc)))
        while time.monotonic()<end:
            try:
                if all(not _same(_read(r["pid"]),r) for r in held):break
            except BaseException as exc:
                errors.append(dict(phase="absence",error=type(exc).__name__+": "+str(exc)));break
            time.sleep(.02)
        final=[]
        for row in held:
            try:
                actual=_read(row["pid"]);poller=select.poll();poller.register(row["_fd"],select.POLLIN)
                final.append(dict(expected={k:v for k,v in row.items() if not k.startswith("_")},actual=actual,exact_identity_absent=not _same(actual,row),pidfd_exit_ready=bool(poller.poll(0))))
            except BaseException as exc:errors.append(dict(phase="final",pid=row["pid"],error=type(exc).__name__+": "+str(exc)))
            finally:os.close(row["_fd"])
    return dict(schema="H059_ROOT_current_targeted_cleanup_v1",began_seconds_monotonic=began,ended_utc=stamp(),elapsed_seconds=time.monotonic()-began,freeze_tree_complete=freeze_complete,observed_edges=edges,known_groups=sorted(groups),events=events,errors=errors,final_identity_observations=final,original_direct_worker_wait="UNKNOWN",original_OS_wait_status=None,original_owned_drain="UNKNOWN",historical_success=False,close_or_credit=False,truth_scored=False,numeric_PID_or_group_signals=False,post_cleanup_group_absence_observation_required=True)

if __name__=="__main__":
    print(json.dumps(dict(status="INACTIVE_ROOT_RECIPE_ONLY",signals_sent=False,automatic_execution=False)))
