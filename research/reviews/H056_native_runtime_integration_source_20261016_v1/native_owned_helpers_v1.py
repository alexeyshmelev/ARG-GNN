"""H056-specific reuse of reviewed original owned native helpers.
No I/O, source/numerical import, guard or native execution at import.
Every inherited function below is AST-identical to its H018 predecessor.
Root-only caller binds validated H056 scopes and count-derived caps once.
"""
from pathlib import Path
from datetime import datetime,timezone
import base64,csv,errno,hashlib,importlib.util,json,os,resource,selectors,signal,stat,struct,sys,time

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SETTINGS=None
OUTPUT=AUX=None
CAP=OUTPUT_CAP=MEMORY=COMBINED_RSS=SUPERVISOR_SECONDS=GRACE_SECONDS=None
PINNED_RUNTIME_SETTINGS={'CPU_seconds_per_process': 1200, 'CPU_threads': 4, 'Relate_seconds': 1200, 'address_space_per_process_bytes': 3000000000, 'combined_RSS_cap_bytes': 4000000000, 'converter_seconds': 300, 'inventory_max_attempts': 3, 'inventory_retry_delay_seconds': 0.01, 'total_controller_seconds': 2400, 'termination_grace_seconds': 10, 'per_stage_log_cap_bytes': 1000000, 'runtime_package_RECORD_refs': [{'path': '.runtime/env/lib/python3.11/site-packages/numpy-2.4.6.dist-info/RECORD', 'sha256': '14394d8bc753425db7ec827e38d8d221d5a322f98b27db1ac3b195aef0a24c4b'}, {'path': '.runtime/env/lib/python3.11/site-packages/tskit-1.0.3.dist-info/RECORD', 'sha256': 'fd3029af41d28cebdc0a9a6399662cef45171a15ef6f0cc0626454a7bd8fb5a3'}], 'qualification_bundle_ref': {'path': 'research/reviews/H018_observed2074_native_inference_source_v2/qualification_bundle.json', 'sha256': '0754ac9184ce872c430f7febc2f8b1c39124cf4b23f21a9303238cf615e9e371'}, 'qualification_gate_source_ref': {'path': 'research/reviews/H018_observed2074_native_inference_source_v2/qualification_gate.py', 'sha256': '3f581166b3a0d2aed6dc1b6cb731a96f270b87d6851802f0723b26ed60f705b0'}, 'original_qualifier_driver_ref': {'path': 'research/reviews/H018_native_Relate_qualification_source_v1/root_qualifier_v4.py', 'sha256': 'c827912e2f0629fc683c8d3c392c2e48574701e611697eb2dd932f88c2ef1207'}, 'original_qualifier_manifest_ref': {'path': 'research/reviews/H018_native_Relate_qualification_source_v1/source_manifest_v4.json', 'sha256': '671134f22a9a64dbd77304e81661a4191990459f92b80bbe845c213d314061a6'}, 'corrected_verifier_ref': {'path': 'research/reviews/H018_native_Relate_qualification_source_v1/v4_internal_coordinate_verifier_audit_v1/fixture_and_verify_corrected_v1.py', 'sha256': '00c2deba87564fe2beee7e599d3e143f56eb669d21edc867f8127f2b731d67be'}}


def bind_h056_v1(cfg,caps):
    global SETTINGS,OUTPUT,AUX,CAP,OUTPUT_CAP,MEMORY,COMBINED_RSS,SUPERVISOR_SECONDS,GRACE_SECONDS
    if SETTINGS is not None:
        raise ValueError("Single H056 helper binding; no scope/cap swap")
    if cfg["schema"]!="H056_native_Relate_config_v1":
        raise ValueError("Specific H056 native config only")
    SETTINGS=dict(PINNED_RUNTIME_SETTINGS)
    SETTINGS.update(native_file_cap_bytes=caps["native_file_cap_bytes"],
        native_output_file_count_cap=caps["native_output_file_count_cap"],
        combined_log_cap_bytes=caps["combined_log_cap_bytes"])
    OUTPUT,AUX=cfg["output_directory"],cfg["auxiliary_directory"]
    CAP,OUTPUT_CAP=caps["output_cap_bytes"],caps["native_output_cap_bytes"]
    MEMORY=SETTINGS["address_space_per_process_bytes"]
    COMBINED_RSS=SETTINGS["combined_RSS_cap_bytes"]
    SUPERVISOR_SECONDS=SETTINGS["total_controller_seconds"]
    GRACE_SECONDS=SETTINGS["termination_grace_seconds"]


def require(ok, text):
    if not ok:
        raise ValueError(text)


def utc():
    return datetime.now(timezone.utc).isoformat()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def syncdir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save(b, folder, name, value):
    record = b.save(folder, name, value)
    syncdir(folder)
    return record


def identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def bounded_file(b, record, cap, *, retain=True):
    """Canonical non-symlink bounded bytes or streaming output/source reference.

    Unlike common.ref this accepts bounded artifacts exceeding four MB. Never
    call this on observed VCF/QC before the worker's original guard boundary.
    """
    require(set(record) in ({"path", "sha256"}, {"path", "sha256", "bytes"}), "Exact bounded reference shape")
    path = b.local(record["path"])
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    parts, digest, total = [], hashlib.sha256(), 0
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_size <= cap, "Bounded regular input/artifact")
        while True:
            chunk = os.read(fd, min(1_048_576, cap - total + 1))
            if not chunk:
                break
            total += len(chunk)
            require(total <= cap, "Input/artifact byte cap")
            digest.update(chunk)
            if retain:
                parts.append(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    require(identity(before) == identity(after) == identity(path.lstat()), "Input/artifact changed during read")
    require(total == before.st_size and digest.hexdigest() == record["sha256"], "Pinned input/artifact SHA and size")
    require("bytes" not in record or total == record["bytes"], "Completion artifact byte count")
    return b"".join(parts) if retain else dict(path=record["path"], sha256=digest.hexdigest(), bytes=total)


def error_record(exc):
    return dict(type=type(exc).__name__, message=str(exc)[:2048])


def write_bytes(folder, name, raw):
    require(Path(name).name == name and name not in ("", ".", ".."), "Flat exclusive artifact name")
    fd = os.open(folder / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


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


def limit_process():
    sys.dont_write_bytecode=True
    os.environ.update(CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",MKL_NUM_THREADS="4",
      PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/"src"),LC_ALL="C",LANG="C")
    affinity=os.sched_getaffinity(0)
    require(len(affinity)>=SETTINGS["CPU_threads"],"Four authorized CPU cores available")
    os.sched_setaffinity(0,set(sorted(affinity)[:SETTINGS["CPU_threads"]]))
    resource.setrlimit(resource.RLIMIT_AS,(MEMORY,MEMORY))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    resource.setrlimit(resource.RLIMIT_CPU,(SETTINGS["CPU_seconds_per_process"],SETTINGS["CPU_seconds_per_process"]+10))


def native_limits():
    # Called in the pre-exec child before native code. The controller imports
    # tskit/numpy only after all declared native waits. No extra launcher/process exists.
    resource.setrlimit(resource.RLIMIT_FSIZE,(SETTINGS["native_file_cap_bytes"],SETTINGS["native_file_cap_bytes"]))


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


def verify_qualification_v1(b, refs):
    bundle=b.read(SETTINGS["qualification_bundle_ref"])
    require(refs==bundle,"Exact pinned original and corrected qualification evidence bundle")
    required={"controller_result","controller_wait","admitted","request","original_facts",
      "corrected_registration","corrected_result","facts","waits"}
    require(isinstance(refs,dict) and set(refs)==required,"Complete preserved-output qualification refs")
    require(isinstance(refs["waits"],dict) and set(refs["waits"])=={
      "make_chunks","paint_c0","build_topology_c0","find_equivalent_branches_c0",
      "infer_branch_lengths_c0","combine_sections_c0","finalize","convert"},
      "Exactly eight actual v4 native wait refs before any evidence reads")
    records={k:b.read(refs[k]) for k in required if k!="waits"}
    records["waits"]={k:b.read(rec) for k,rec in refs["waits"].items()}
    gate=load("_H018_observed_corrected_v4_gate",b.local(SETTINGS["qualification_gate_source_ref"]["path"]))
    return gate.verify_v4_corrected(records,refs,SETTINGS["original_qualifier_driver_ref"],
      SETTINGS["original_qualifier_manifest_ref"],SETTINGS["corrected_verifier_ref"])


def generated_project_pycache_path_v2(path):
    """Lexical test of the canonical resolved Path supplied by the verifier.

    No arbitrary __pycache__ substring exception. Generated .pyc content is not
    authenticated by RECORD; only hashed entries are authenticated below.
    """
    cache_root=ROOT/".runtime/cache/pycache"
    return (isinstance(path,Path) and path.is_absolute() and ".." not in path.parts
        and path.suffix==".pyc" and cache_root in path.parents)


def verify_runtime_packages(b):
    site=ROOT/".runtime/env/lib/python3.11/site-packages"
    hashed=generated=blank_record=0
    for record in SETTINGS["runtime_package_RECORD_refs"]:
        raw=bounded_file(b,record,2_000_000)
        for name,digest,size in csv.reader(raw.decode().splitlines()):
            path=(site/name).resolve()
            require(ROOT in path.parents and path.is_file() and not path.is_symlink(),"Project-contained installed wheel file")
            if not digest:
                if name.endswith("/RECORD"):
                    blank_record+=1
                else:
                    require(generated_project_pycache_path_v2(path),"Only wheel RECORD or generated project pycache .pyc entries may omit hash")
                    generated+=1
                continue
            algorithm,token=digest.split("=",1)
            require(algorithm=="sha256","Pinned wheel SHA256 entries")
            expected=base64.urlsafe_b64decode(token+"="*((-len(token))%4)).hex()
            checked=bounded_file(b,dict(path=path.relative_to(ROOT).as_posix(),sha256=expected),64_000_000,retain=False)
            require(checked["bytes"]==int(size),"Installed wheel RECORD size")
            hashed+=1
    return dict(schema="H018_wheel_RECORD_runtime_preflight_v2",runtime_package_RECORD_refs=SETTINGS["runtime_package_RECORD_refs"],
        all_hashed_RECORD_entries_SHA256_and_size_verified=True,hashed_file_entries=hashed,
        generated_pyc_entries_without_RECORD_hash=generated,blank_RECORD_entries=blank_record,
        generated_pyc_contents_authenticated=False,wheel_all_files_authenticated_claimed=False,
        native_executed=False,guard_or_admission_executed=False)
