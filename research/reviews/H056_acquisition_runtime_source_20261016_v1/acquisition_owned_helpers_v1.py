"""H056 acquisition-only binding of unchanged original H018 owned supervision.
Import is inert. The ten inherited functions are AST-identical to root_inference.
No native/controller/admission implementation is added.
"""
from pathlib import Path
from datetime import datetime, timezone
import errno, hashlib, json, os, resource, selectors, signal, stat, sys, time
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SETTINGS=None
OUTPUT=AUX=None
CAP=OUTPUT_CAP=COMBINED_RSS=GRACE_SECONDS=None

def bind_acquisition_v1(output,aux,cap):
    global SETTINGS,OUTPUT,AUX,CAP,OUTPUT_CAP,COMBINED_RSS,GRACE_SECONDS
    if SETTINGS is not None:
        raise ValueError("Single root-owned acquisition helper binding")
    if not isinstance(output,str) or not isinstance(aux,str) or output==aux or type(cap)is not int:
        raise ValueError("Exact distinct contained payload/aux scopes and integer cap")
    OUTPUT,AUX,CAP,OUTPUT_CAP=output,aux,cap,cap
    COMBINED_RSS,GRACE_SECONDS=2*2**30,10
    SETTINGS=dict(native_file_cap_bytes=cap,native_output_file_count_cap=64,
        per_stage_log_cap_bytes=65536,inventory_max_attempts=3,inventory_retry_delay_seconds=.01)

def require(ok, text):
    if not ok:
        raise ValueError(text)

def utc():
    return datetime.now(timezone.utc).isoformat()

def identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)

def error_record(exc):
    return dict(type=type(exc).__name__, message=str(exc)[:2048])

def group_absent(pgid):
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False

def direct_wait_owned(proc, started, control_error, seconds):
    """Reap this exact child using its actual OS wait status, never a guessed rc.

    WNOHANG leaves a live leader unreaped while deadline/group signals may be
    needed. Once waitpid returns this child, no group signal is sent again.
    """
    terminate_at = None
    killed = False
    while True:
        try:
            now = time.monotonic()
            if control_error is not None or now-started > seconds:
                if control_error is None:
                    control_error = dict(type="TimeoutError", message="Owned child outlived supervisor deadline")
                if terminate_at is None:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    terminate_at = now
                elif now-terminate_at >= GRACE_SECONDS and not killed:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    killed = True
            waited_pid, status = os.waitpid(proc.pid, os.WNOHANG)
            if waited_pid == proc.pid:
                require(os.WIFEXITED(status) or os.WIFSIGNALED(status), "Terminal genuine child OS wait status")
                rc = os.waitstatus_to_exitcode(status)
                proc.returncode = rc  # Prevent Popen destructor from attempting a second reap.
                return dict(actual_exit_code=rc, actual_OS_wait_status=status,
                            actual_direct_wait_observed=True, control_error=control_error)
            require(waited_pid == 0, "waitpid returned an unrelated child")
            time.sleep(0.05)
        except ChildProcessError as exc:
            # External reaping cannot be converted to wait0 or a drain claim.
            return dict(actual_exit_code=None, actual_OS_wait_status=None,
                        actual_direct_wait_observed=False, control_error=error_record(exc))
        except (InterruptedError, KeyboardInterrupt, SystemExit) as exc:
            control_error = error_record(exc)
            continue
        except BaseException as exc:
            # Preserve ownership and continue to the exact OS wait after control
            # errors. Repeated inability to signal cannot justify a fake exit.
            control_error = error_record(exc)
            if terminate_at is None:
                terminate_at = time.monotonic()-GRACE_SECONDS

def _inventory_once_v4(folder, retain_refs=False):
    """One complete owned-root scan; transient errors invalidate the whole scan."""
    root_info=folder.lstat()
    require(stat.S_ISDIR(root_info.st_mode) and not stat.S_ISLNK(root_info.st_mode),
      "Owned inventory root must remain a real directory")
    total=0;refs={};signature=[(".",identity(root_info))]
    for path in folder.rglob("*"):
        info=path.lstat()
        require(not stat.S_ISLNK(info.st_mode),"No output symlinks")
        relative=path.relative_to(folder).as_posix()
        signature.append((relative,identity(info)))
        if stat.S_ISREG(info.st_mode):
            require(info.st_size<=SETTINGS["native_file_cap_bytes"],"Native per-file cap")
            total+=info.st_size
            require(len(refs)<SETTINGS["native_output_file_count_cap"],"Native file-count cap")
            refs[relative]=dict(bytes=info.st_size)
            if retain_refs:
                raw=path.read_bytes()
                require(len(raw)==info.st_size and identity(info)==identity(path.lstat()),
                  "Terminal inventory file changed while hashing")
                refs[relative].update(path=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(raw).hexdigest())
        else:
            require(stat.S_ISDIR(info.st_mode),"Only regular files/directories")
    require(identity(root_info)==identity(folder.lstat()),"Owned inventory root changed during scan")
    return total,refs,tuple(sorted(signature))

def file_inventory(folder, retain_refs=False, *, stable=False):
    """At most three fresh scans; only owned ENOENT/ESTALE may retry.

    No partial scan is returned. Terminal mode needs two consecutive complete
    equal signatures after actual child wait/drain. Active scans remain sampled.
    """
    require(folder in (ROOT/OUTPUT,ROOT/AUX),"Exact owned output or auxiliary inventory scope")
    require(not retain_refs or stable,"Hash inventories require terminal stability")
    previous=None
    for attempt in range(SETTINGS["inventory_max_attempts"]):
        try:
            total,refs,signature=_inventory_once_v4(folder,retain_refs)
        except OSError as exc:
            affected=Path(exc.filename) if isinstance(exc.filename,str) else None
            transient=(exc.errno in (errno.ENOENT,errno.ESTALE) and affected is not None
              and (affected==folder or folder in affected.parents))
            if not transient or attempt+1==SETTINGS["inventory_max_attempts"]:
                raise
            previous=None
        else:
            if not stable or signature==previous:
                return total,refs
            previous=signature
        if attempt+1<SETTINGS["inventory_max_attempts"]:
            time.sleep(SETTINGS["inventory_retry_delay_seconds"])
    raise ValueError("No two consecutive complete stable terminal inventories within bounded attempts")

def owned_rss(pid):
    # Descend only from this exact owned child; never scan unrelated processes.
    total=0;pending=[pid];seen=set()
    while pending:
        child=pending.pop()
        if child in seen:continue
        seen.add(child)
        require(len(seen)<=32,"Owned descendant count cap")
        try:
            text=Path("/proc/"+str(child)+"/status").read_text()
            total+=next((int(line.split()[1])*1024 for line in text.splitlines() if line.startswith("VmRSS:")),0)
            pending += [int(x) for x in Path("/proc/"+str(child)+"/task/"+str(child)+"/children").read_text().split()]
        except FileNotFoundError:
            pass
    return total

def supervise(proc,seconds,output,aux,log_remaining):
    captured,total,peak,error=bytearray(),0,0,None
    log_cap=min(SETTINGS["per_stage_log_cap_bytes"],log_remaining)
    started=time.monotonic();selector=None
    try:
        selector=selectors.DefaultSelector()
        os.set_blocking(proc.stdout.fileno(),False)
        selector.register(proc.stdout,selectors.EVENT_READ)
        eof=False
        while not eof:
            require(time.monotonic()-started<=seconds,"Native stage wall deadline")
            a,_=file_inventory(output);z,_=file_inventory(aux)
            require(a<=OUTPUT_CAP and a+z+131072<=CAP,"Monitored output/aux byte caps")
            current=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024+owned_rss(proc.pid)
            peak=max(peak,current)
            require(current<=COMBINED_RSS,"Owned controller/native descendant RSS cap")
            for key,events in selector.select(timeout=0.1):
                chunk=os.read(proc.stdout.fileno(),65536)
                if not chunk:eof=True;break
                total+=len(chunk)
                captured.extend(chunk[:max(0,log_cap-len(captured))])
                require(total<=log_cap,"Native combined stdout/stderr cap")
    except BaseException as exc:
        error=error_record(exc)
    finally:
        waited=direct_wait_owned(proc,started,error,seconds)
        drained=group_absent(proc.pid) if waited["actual_direct_wait_observed"] else False
        observation=dict(schema="H018_Relate_qualifier_native_direct_wait_v1",native_pid=proc.pid,process_group=proc.pid,
            supervisor_pid=os.getpid(),**waited,process_group_drained=drained,owned_group_absent_after_wait=drained,
            observed_before_risky_postprocessing=True,observed_utc=utc(),log_total_bytes=total,peak_combined_RSS_bytes=peak)
        cleanup=[]
        for item in (selector,proc.stdout):
            if item is not None:
                try:item.close()
                except BaseException as exc:cleanup.append(error_record(exc))
        if cleanup:
            observation["cleanup_errors"]=cleanup
            observation["control_error"]=observation["control_error"] or cleanup[0]
        try:
            print(json.dumps(dict(native_direct_wait_observation=observation)),flush=True)
        except BaseException as exc:
            observation["stdout_receipt_error"]=error_record(exc)
    return observation,bytes(captured)
