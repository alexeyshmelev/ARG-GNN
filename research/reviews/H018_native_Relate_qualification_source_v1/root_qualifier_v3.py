"""Inactive root-only tiny CPU Relate qualifier. No biological input or claim.
Root owns activation/admission/execution. Exact unchanged storage guard. No credit,
replay, phase resume, filtering, genotype reattachment or chronology repair.
"""
from pathlib import Path
from datetime import datetime, timedelta, timezone
import argparse, base64, csv, fcntl, hashlib, importlib.util, json, math, os, resource, selectors, signal, stat, struct, subprocess, sys, time

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/H018_native_Relate_qualification_source_v1"
SOURCE=REL+"/root_qualifier_v3.py"
CONFIG=REL+"/config_v3.json"
MANIFEST=REL+"/source_manifest_v3.json"
SETTINGS={'CPU_seconds_per_process': 300, 'CPU_threads': 1, 'GPU_authorized': False, 'HS_execution_authorized': False, 'Relate_seconds': 300, 'address_space_per_process_bytes': 1000000000, 'auxiliary_directory': 'research/reviews/H018_native_Relate_qualification_source_v1/worker_aux_v3', 'auxiliary_guard_growth_bytes': 3000000, 'biological_native_execution_authorized': False, 'combined_RSS_cap_bytes': 2000000000, 'combined_log_cap_bytes': 2000000, 'common_ref': {'path': 'research/reviews/H018_map_acquisition_root_source_v1/root_map_driver.py', 'sha256': '4c957db44a1bf6c9cd8e76fd6cd5e31b979b7c618554c6a73c57cce8099af65d'}, 'condition_id': 'H018_Relate_fullsite_native_support_synthetic_v1', 'converter_seconds': 120, 'fixture_expected_ref': {'path': 'research/reviews/H018_native_Relate_qualification_source_v1/fixture_expected.json', 'sha256': 'a42a6ec13f6f1c23f36fc8b5147af5f93f9e78a9608ecab72b7a6c0bcf32f1e9'}, 'fixture_profile_ref': {'path': 'research/reviews/H018_native_Relate_qualification_source_v1/fixture_profile.json', 'sha256': 'fb5321923376aa8da4a937e567419378f7d0fdb026b7d736ca2349392e829e3a'}, 'fresh_single_attempt': True, 'guard_ref': {'path': 'research/reviews/concurrent_completed_scope_credits_v3/completed_scope_credits_v3.py', 'sha256': 'ea14c12c41b9385a86182aa9c24058f7b1276107fc5706c70e47828e8b2c31ef'}, 'helper_ref': {'path': 'research/reviews/H018_native_Relate_qualification_source_v1/fixture_and_verify.py', 'sha256': '3a149c61a33af754ecf1bea8848668d44856620ba74e5c61f55439f9e5181bad'}, 'input_fixture_cap_bytes': 1000000, 'installation_ref': {'path': '.runtime/tools/relate/installation.json', 'sha256': 'c7cf58b9b366bc7a90e1320b17be781a0ad257c9b6cbe464b41451888e391b94'}, 'job_id': 'H018-native-Relate-synthetic-qualification-v3', 'license_ref': {'path': '.runtime/tools/relate/relate_v1.2.4_x86_64_static/20180411_15670_RELATE_Academic_Usage_Licence.pdf', 'sha256': 'c55aec267c4f8987f1d67c91cc2fb44a695c3f06244e3bf7be91230e61f2dbe2'}, 'native_binary_refs': {'Relate': {'path': '.runtime/tools/relate/relate_v1.2.4_x86_64_static/bin/Relate', 'sha256': '71ce82f469dd99749a3eb1847e0704be8785a26be94eae2cb8e1ed80881c7ce8'}, 'RelateFileFormats': {'path': '.runtime/tools/relate/relate_v1.2.4_x86_64_static/bin/RelateFileFormats', 'sha256': 'cb19cb6e277c27704de506f0e8155f8f53d9c3dc8d0f52615a059caa253332ee'}}, 'native_facts_cap_bytes': 3000000, 'native_file_cap_bytes': 4000000, 'native_output_file_count_cap': 256, 'native_qualification_capability': True, 'network_authorized': False, 'output_cap_bytes': 32000000, 'output_directory': 'data/H018_native_Relate_synthetic_qualification_v3', 'packet_directory': 'research/reviews/H018_native_Relate_qualification_source_v1/admission_v3', 'per_stage_log_cap_bytes': 1000000, 'physical_gpu': None, 'policy_ref': {'path': 'research/reviews/completed_scope_credits_root_activation_20261015_v1/activation_policy.json', 'sha256': 'a0885b5209172dd79a835d442e0ad3495f5c1a0eac61a1c5c08154ec877ecf85'}, 'python_binary_ref': {'path': '.runtime/env/bin/python3.11', 'sha256': '463a65c4055c18d88f6ed863d227383593395c2f7f70976e38df24f17a836289'}, 'python_version': '3.11.17 (main, Oct  7 2026, 00:02:24) [GCC 15.3.0]', 'real_GT_reads_authorized': False, 'repair_authority_ref': {'path': 'research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json', 'sha256': 'f6f2d628199e8be73db8b5d90d428570d8afdefeab9dc5cd94bb5eafb39247db'}, 'reserved_bytes': 50000000, 'runtime_package_RECORD_refs': [{'path': '.runtime/env/lib/python3.11/site-packages/numpy-2.4.6.dist-info/RECORD', 'sha256': '14394d8bc753425db7ec827e38d8d221d5a322f98b27db1ac3b195aef0a24c4b'}, {'path': '.runtime/env/lib/python3.11/site-packages/tskit-1.0.3.dist-info/RECORD', 'sha256': 'fd3029af41d28cebdc0a9a6399662cef45171a15ef6f0cc0626454a7bd8fb5a3'}], 'schema': 'H018_Relate_synthetic_qualification_config_v1', 'termination_grace_seconds': 10, 'total_controller_seconds': 600, 'operational_condition': 'separate_supported_CLI_stages_full_section_range_v3', 'author_split_CLI_ref': {'path': '.runtime/tools/relate/relate_v1.2.4_x86_64_static/scripts/RelateParallel/RelateParallel.sh', 'sha256': 'a4558aaffbcb73899bc5a3d87fbb253c80d06160995e10c40e013a7122644315'}, 'cleanup_source_audit_ref': {'path': 'research/reviews/H018_Relate_finalizer_cleanup_source_audit_v1/AUDIT.md', 'sha256': 'a836ac012628f6538b6d7d8bd683db7f35bbda72ee359e80dcbeb8bd17e3394f'}, 'expected_native_headers': {'N': 40, 'L': 512, 'num_chunks': 1, 'chunk_index': 0, 'boundary_count': 2, 'section_boundaries': [0, 512]}, 'Relate_seconds_scope': 'aggregate_all_seven_Relate_stages_including_interstage_overhead', 'expected_native_direct_wait_stages': ['make_chunks', 'paint_c0', 'build_topology_c0', 'find_equivalent_branches_c0', 'infer_branch_lengths_c0', 'combine_sections_c0', 'finalize', 'convert']}
PACKET=SETTINGS["packet_directory"]
OUTPUT=SETTINGS["output_directory"]
AUX=SETTINGS["auxiliary_directory"]
JOB=SETTINGS["job_id"]
KEY="H018/native_Relate_synthetic_qualification_v3"
CONDITION=SETTINGS["condition_id"]
CAP=SETTINGS["reserved_bytes"]
OUTPUT_CAP=SETTINGS["output_cap_bytes"]
AUX_WRITE=SETTINGS["auxiliary_guard_growth_bytes"]
MEMORY=SETTINGS["address_space_per_process_bytes"]
COMBINED_RSS=SETTINGS["combined_RSS_cap_bytes"]
SUPERVISOR_SECONDS=SETTINGS["total_controller_seconds"]
GRACE_SECONDS=SETTINGS["termination_grace_seconds"]
DIRECT_BINDING_SUFFIXES=(".json",".py",".sh")
COMMON=SETTINGS["common_ref"]
PYTHON=SETTINGS["python_binary_ref"]
PYTHON_VERSION=SETTINGS["python_version"]


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


def common():
    raw = (ROOT / COMMON["path"]).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == COMMON["sha256"], "Original admission primitive source drift")
    return load("_H018_bridge_unchanged_root_primitives", ROOT / COMMON["path"])


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


def direct_authority_refs(manifest, closure=None):
    """Retain every ref transitively; only original allowed suffixes bind directly."""
    groups = ("source_refs", "metadata_refs")
    excluded = {group: {name: rec for name, rec in manifest[group].items()
                       if Path(rec["path"]).suffix not in DIRECT_BINDING_SUFFIXES}
                for group in groups}
    require(manifest["direct_binding_suffixes"] == list(DIRECT_BINDING_SUFFIXES)
            and manifest["transitive_only_refs"] == excluded,
            "Exact original allowed direct suffixes and full excluded-reference manifest required")
    if closure is not None:
        require(all(closure[group] == manifest[group] for group in groups)
                and closure["direct_binding_suffixes"] == manifest["direct_binding_suffixes"]
                and closure["transitive_only_refs"] == excluded,
                "Actual direct JSON closure must preserve every complete source/metadata/excluded ref")
    return [rec for group in groups for rec in manifest[group].values()
            if Path(rec["path"]).suffix in DIRECT_BINDING_SUFFIXES]


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
    os.environ.update(CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",
      PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/"src"),LC_ALL="C",LANG="C")
    affinity=os.sched_getaffinity(0)
    require(affinity,"CPU affinity available")
    os.sched_setaffinity(0,{min(affinity)})
    resource.setrlimit(resource.RLIMIT_AS,(MEMORY,MEMORY))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    resource.setrlimit(resource.RLIMIT_CPU,(300,310))



def native_limits():
    # Called in the pre-exec child before native code. The controller imports
    # tskit/numpy only after all eight native waits. No extra launcher/process exists.
    resource.setrlimit(resource.RLIMIT_FSIZE,(SETTINGS["native_file_cap_bytes"],SETTINGS["native_file_cap_bytes"]))



def file_inventory(folder, retain_refs=False):
    total=0;refs={}
    for path in folder.rglob("*"):
        try:
            info=path.lstat()
        except FileNotFoundError:
            continue
        require(not stat.S_ISLNK(info.st_mode),"No output symlinks")
        if stat.S_ISREG(info.st_mode):
            require(info.st_size<=SETTINGS["native_file_cap_bytes"],"Native per-file cap")
            total+=info.st_size
            require(len(refs)<SETTINGS["native_output_file_count_cap"],"Native file-count cap")
            refs[path.relative_to(folder).as_posix()]=dict(bytes=info.st_size)
            if retain_refs:
                refs[path.relative_to(folder).as_posix()].update(path=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        else:
            require(stat.S_ISDIR(info.st_mode),"Only regular files/directories")
    return total,refs


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


def verify_request(b,record):
    r=b.read(record)
    require(r["schema"]=="H018_Relate_synthetic_qualification_request_v1" and r["status"]=="active"
        and r["root_authorizer"]=="root_agent" and r["registered_before_execution"] is True,"Root activation/registration required")
    expected=dict(job_id=JOB,condition_id=CONDITION,output_directory=OUTPUT,packet_directory=PACKET,
      auxiliary_directory=AUX,reserved_bytes=CAP,native_execution_authorized=True,
      biological_native_execution_authorized=False,real_GT_reads_authorized=False,GPU_authorized=False,
      HS_execution_authorized=False,replay_authorized=False)
    require(all(r.get(k)==v for k,v in expected.items()),"Exact tiny synthetic-only authority")
    require(r["driver_ref"]==b.ref(SOURCE) and r["config_ref"]==b.ref(CONFIG) and r["source_manifest_ref"]==b.ref(MANIFEST),"Actual source pins")
    cfg,manifest=b.read(r["config_ref"]),b.read(r["source_manifest_ref"])
    require(cfg==SETTINGS,"Exact fixed qualifier resource/tool/fixture configuration")
    require(manifest["schema"]=="H018_Relate_synthetic_qualification_source_manifest_v1"
        and manifest["source_only"] is True and manifest["source_refs"]["driver"]==r["driver_ref"]
        and manifest["source_refs"]["config"]==r["config_ref"],"Exact source closure")
    direct_authority_refs(manifest)
    for group in ("source_refs","metadata_refs"):
        for rec in manifest[group].values():bounded_file(b,rec,64_000_000,retain=False)
    registration=b.read(r["registration_ref"])
    entry=next((x for x in registration["decisions"] if x.get("id")=="H018"),None)
    require(entry is not None and JOB in json.dumps(entry,sort_keys=True),"Root H018 registration names fresh synthetic qualifier")
    require(Path(sys.executable).resolve()==ROOT/PYTHON["path"] and sys.version==PYTHON_VERSION,"Exact project runtime")
    bounded_file(b,PYTHON,32_000_000,retain=False)
    return r,cfg,manifest


def admit(record):
    b=common()
    verify_runtime_packages(b)  # ROOT pre-admission gate before packet/guard/accounting
    r,cfg,manifest=verify_request(b,record)
    packet,output,aux=(b.local(x) for x in (PACKET,OUTPUT,AUX))
    require(not any(p.exists() for p in (packet,output,aux)),"Fresh single-use scope")
    packet.mkdir();syncdir(packet.parent)
    expiry=(datetime.now(timezone.utc)+timedelta(hours=2)).isoformat()
    runtime=save(b,packet,"runtime.json",dict(schema="H018_Relate_synthetic_CPU_runtime_v1",job_id=JOB,
      python_binary_ref=PYTHON,python_version=sys.version,CPU_threads=1,device="cpu",physical_gpu=None,
      native_binary_refs=SETTINGS["native_binary_refs"],license_ref=SETTINGS["license_ref"],
      address_space_per_process_bytes=MEMORY,combined_RSS_cap_bytes=COMBINED_RSS,
      expires_at_utc=expiry,synthetic_only=True,biological_native_execution_authorized=False))
    closure=save(b,packet,"consumer_manifest.json",dict(schema="H018_Relate_synthetic_consumer_v1",request_ref=record,
      source_manifest_ref=r["source_manifest_ref"],runtime_ref=runtime,registration_ref=r["registration_ref"],
      source_refs=manifest["source_refs"],metadata_refs=manifest["metadata_refs"],
      direct_binding_suffixes=manifest["direct_binding_suffixes"],transitive_only_refs=manifest["transitive_only_refs"],
      synthetic_only=True,real_GT_read=False))
    closure_direct=direct_authority_refs(manifest,b.read(closure))
    m=b.guard()
    lock=b.local("research/reviews/American_C3_short_train2mb_wave_source_v1/scheduler/admission.lock")
    with lock.open("a+") as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        view=m.inspect_v1(expected_source_sha256=b.CORE,expected_launcher_sha256=b.BASE)
        candidates=list({x["path"]:x for x in [record,r["config_ref"],r["source_manifest_ref"],r["registration_ref"],
          runtime,closure,*manifest["source_refs"].values(),*manifest["metadata_refs"].values(),
          b.GUARD,b.POLICY,b.REPAIR,*view["bindings"].values()]}.values())
        consumers=[x for x in candidates if Path(x["path"]).suffix in DIRECT_BINDING_SUFFIXES]
        authorities=[record,r["config_ref"],r["source_manifest_ref"],r["registration_ref"],runtime,closure,
          *closure_direct,b.GUARD,b.POLICY,b.REPAIR,*view["bindings"].values()]
        require(all(Path(x["path"]).suffix in DIRECT_BINDING_SUFFIXES and x in consumers for x in authorities),
          "All actual runtime/source/guard authority direct refs retained")
        require(all(x not in consumers for g in manifest["transitive_only_refs"].values() for x in g.values()),"Full excluded refs retained transitively")
        require(len(consumers)<=64,"Original direct consumer reference bound")
        operation=m.operation_draft_v1("reserve",dict(key=KEY,output=str(output),auxiliary_scopes=[str(aux)],
          reserved_bytes=CAP,consumer_bindings=consumers),expected_before=m._pins(view),bindings=view["bindings"])
        operation.update(active=True,expires_at_utc=expiry)
        receipt=m.apply_reserve_v1(save(b,packet,"reserve_authority.json",operation),expected_source_sha256=b.CORE,expected_launcher_sha256=b.BASE)
    save(b,packet,"reserve_receipt.json",receipt)
    storage=save(b,packet,"storage.json",dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,
      accounting_mode=m.MODE,job_id=JOB,output_directory=OUTPUT,reserved_new_bytes=CAP,reservation=receipt["value"],
      expires_at_utc=expiry,initial_ledger_pins=receipt["initial_ledger_pins"],completed_overlay_policy_ref=b.POLICY,
      budget_components=dict(implementation=b.ref(m.budget.SOURCE_REL),launcher=b.ref(m.budget.LAUNCHER_REL),consumer_adapter=b.GUARD)))
    admitted=save(b,packet,"admitted.json",dict(schema="H018_Relate_synthetic_admitted_v1",request_ref=record,
      driver_ref=b.ref(SOURCE),runtime_ref=runtime,storage_ref=storage,consumer_manifest_ref=closure,
      source_manifest_ref=r["source_manifest_ref"],job_id=JOB,output_directory=OUTPUT,auxiliary_directory=AUX,
      reserved_bytes=CAP,expires_at_utc=expiry,synthetic_only=True,execution_started=False,
      consumer_binding_count=len(consumers),reservation_remains_fully_charged=True))
    print(json.dumps(dict(admitted_ref=admitted,reserved_bytes=CAP)),flush=True)
    return admitted


def admitted_inputs(record):
    b=common();ad=b.read(record)
    require(record["path"]==PACKET+"/admitted.json" and ad["schema"]=="H018_Relate_synthetic_admitted_v1"
      and ad["driver_ref"]==b.ref(SOURCE),"Exact qualifier admission")
    r,cfg,manifest=verify_request(b,ad["request_ref"])
    require(ad["job_id"]==JOB and ad["source_manifest_ref"]==r["source_manifest_ref"] and ad["output_directory"]==OUTPUT
      and ad["auxiliary_directory"]==AUX and ad["reserved_bytes"]==CAP and ad["synthetic_only"] is True,"Exact admitted scope")
    expiry=datetime.fromisoformat(ad["expires_at_utc"])
    require(expiry.tzinfo is not None and expiry>datetime.now(timezone.utc),"Unexpired admission")
    runtime,storage,closure=(b.read(ad[k]) for k in ("runtime_ref","storage_ref","consumer_manifest_ref"))
    require(runtime["job_id"]==JOB and runtime["python_binary_ref"]==PYTHON and runtime["python_version"]==sys.version
      and runtime["native_binary_refs"]==SETTINGS["native_binary_refs"] and runtime["license_ref"]==SETTINGS["license_ref"]
      and runtime["expires_at_utc"]==ad["expires_at_utc"] and runtime["synthetic_only"] is True,"Exact actual runtime")
    own=storage["reservation"]
    require(storage["job_id"]==JOB and storage["output_directory"]==OUTPUT and storage["reserved_new_bytes"]==CAP
      and own["key"]==KEY and own["output"]==str(b.local(OUTPUT)) and own["auxiliary_scopes"]==[str(b.local(AUX))]
      and own["reserved_bytes"]==CAP and storage["expires_at_utc"]==ad["expires_at_utc"],"Exact original reservation")
    direct=direct_authority_refs(manifest,closure)
    require(closure["request_ref"]==ad["request_ref"] and closure["runtime_ref"]==ad["runtime_ref"]
      and closure["source_manifest_ref"]==r["source_manifest_ref"],"Exact bound consumer closure")
    required=[ad["request_ref"],ad["runtime_ref"],ad["consumer_manifest_ref"],r["config_ref"],r["source_manifest_ref"],
      r["registration_ref"],*direct,b.GUARD,b.POLICY,b.REPAIR,*storage["budget_components"].values()]
    require(all(Path(x["path"]).suffix in DIRECT_BINDING_SUFFIXES and x in own["consumer_bindings"] for x in required),"Actual authority bindings")
    require(all(x not in own["consumer_bindings"] for g in manifest["transitive_only_refs"].values() for x in g.values()),"Excluded refs only transitive")
    return b,ad,r,storage


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


def runtime_preflight_v2():
    """Explicit ROOT-only stdlib runtime-file validation, without guard/admission."""
    return verify_runtime_packages(common())



def parse_split_headers_v3(parameters_raw, chunk_raw):
    """Pure exact little-endian x86_64 native metadata decoding for this fixture."""
    require(isinstance(parameters_raw, bytes) and len(parameters_raw)==28,
      "Exact one-chunk parameters.bin byte length")
    require(isinstance(chunk_raw, bytes) and len(chunk_raw)==20,
      "Exact one-section parameters_c0.bin byte length")
    n,l,chunks,memory,start,end=struct.unpack("<iiidii",parameters_raw)
    cn,cl,boundary_count,first,last=struct.unpack("<iiiii",chunk_raw)
    require((n,l,chunks,start,end)==(40,512,1,0,512),
      "Exact synthetic N40/L512/one-chunk/global span")
    require((cn,cl,boundary_count,first,last)==(40,512,2,0,512),
      "Exact synthetic N40/L512/two boundaries/one full section")
    require(math.isfinite(memory) and 0.0<memory<1.0,"Finite bounded native memory metadata")
    return dict(schema="H018_Relate_split_native_headers_v3",encoding="little_endian_int32_float64_x86_64",
      N=n,L=l,num_chunks=chunks,chunk_index=0,boundary_count=boundary_count,
      section_boundaries=[first,last],first_section=0,last_section=0,
      native_reported_memory_GB=memory,parameters_sha256=hashlib.sha256(parameters_raw).hexdigest(),
      chunk_parameters_sha256=hashlib.sha256(chunk_raw).hexdigest(),
      parameters_hex=parameters_raw.hex(),chunk_parameters_hex=chunk_raw.hex())


def read_split_headers_v3(work):
    """Read only newly generated small headers after genuine MakeChunks wait0."""
    values=[]
    for name,size in (("parameters.bin",28),("parameters_c0.bin",20)):
        path=work/"native"/name
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
        try:
            before=os.fstat(fd)
            require(stat.S_ISREG(before.st_mode) and before.st_size==size,"Exact bounded native header")
            raw=os.read(fd,size+1)
            after=os.fstat(fd)
        finally:
            os.close(fd)
        require(len(raw)==size and identity(before)==identity(after)==identity(path.lstat()),
          "Native header changed during bounded read")
        values.append(raw)
    return parse_split_headers_v3(*values)


def split_cli_stages_v3(binaries, inputs, profile):
    """Author-supported separate CLI modes, one full section range per chunk."""
    common_args=["--haps",str(inputs/"input.haps"),"--sample",str(inputs/"input.sample"),
      "--sample_ages",str(inputs/"sample_ages.generations"),"--map",str(inputs/"input.map"),
      "--mutation_rate",profile["mutation_rate"],"--effectiveN",str(profile["effective_haploid_N"]),
      "--memory",profile["native_memory_GB"],"--seed",str(profile["seed"]),"--output","native"]
    stages=[]
    for name,mode,chunk,full_range in (
      ("make_chunks","MakeChunks",False,False),
      ("paint_c0","Paint",True,False),
      ("build_topology_c0","BuildTopology",True,True),
      ("find_equivalent_branches_c0","FindEquivalentBranches",True,False),
      ("infer_branch_lengths_c0","InferBranchLengths",True,True),
      ("combine_sections_c0","CombineSections",True,False),
      ("finalize","Finalize",False,False)):
        argv=[binaries["Relate"],"--mode",mode,*common_args]
        if chunk:argv.extend(["--chunk_index","0"])
        if full_range:argv.extend(["--first_section","0","--last_section","0"])
        stages.append((name,argv,"relate"))
    stages.append(("convert",[binaries["RelateFileFormats"],"--mode","ConvertToTreeSequence",
      "--input","native","--output","native"],"convert"))
    return stages


def deadline(signum,frame):
    raise TimeoutError("600-second total qualifier controller deadline")


def launch(record):
    b,ad,r,storage=admitted_inputs(record)
    packet,output,aux=(b.local(x) for x in (PACKET,OUTPUT,AUX))
    require(not output.exists() and not aux.exists() and not (packet/"launch_claim.json").exists(),"No replay, continuation or phase resume")
    limit_process()
    require(signal.getsignal(signal.SIGCHLD)!=signal.SIG_IGN,"Direct child wait available")
    m=b.guard();observations=[];qualified=False;facts_ref=None;completed_ref=None;error=None;parameter_headers_ref=None
    started=time.monotonic();prior=signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,SUPERVISOR_SECONDS)
    try:
        m.budget_check_v9(storage,first=True,boundary=True,required_write_bytes=OUTPUT_CAP+AUX_WRITE)
        save(b,packet,"launch_claim.json",dict(admitted_ref=record,controller_pid=os.getpid(),started_utc=utc(),
          synthetic_only=True,retries_or_phase_resume_authorized=False))
        output.mkdir();aux.mkdir();syncdir(output.parent);syncdir(aux.parent)
        inputs=output/"inputs";work=output/"work";inputs.mkdir();work.mkdir()
        helper=load("_H018_exact_synthetic_qualifier_helper",b.local(SETTINGS["helper_ref"]["path"]))
        profile=b.read(SETTINGS["fixture_profile_ref"])
        payloads,truth=helper.make_fixture(profile)
        expected=b.read(SETTINGS["fixture_expected_ref"])
        require(set(payloads)==set(expected["assets"]) and all(
          expected["assets"][name]==dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
          for name,raw in payloads.items()),"Exact frozen tiny synthetic fixture bytes")
        for name,raw in payloads.items():write_bytes(inputs,name,raw)
        syncdir(inputs)
        write_bytes(output,"provenance.json",b.encoded(dict(schema="H018_Relate_synthetic_provenance_v1",admitted_ref=record,
          config_ref=r["config_ref"],source_manifest_ref=r["source_manifest_ref"],fixture_profile_ref=SETTINGS["fixture_profile_ref"],
          helper_ref=SETTINGS["helper_ref"],native_binary_refs=SETTINGS["native_binary_refs"],license_ref=SETTINGS["license_ref"],
          operational_condition=SETTINGS["operational_condition"],author_split_CLI_ref=SETTINGS["author_split_CLI_ref"],
          cleanup_source_audit_ref=SETTINGS["cleanup_source_audit_ref"],exact_source_build_identity_proven=False,
          synthetic_only=True,real_GT_read=False,biological_native_authorized=False,sites_dropped=0,
          qualifier_modified_native_genotypes_or_times=False)))
        verify_runtime_packages(b)
        binaries={key:str(b.local(rec["path"])) for key,rec in SETTINGS["native_binary_refs"].items()}
        stages=split_cli_stages_v3(binaries,inputs,profile)
        require([x[0] for x in stages]==SETTINGS["expected_native_direct_wait_stages"],
          "Exact eight supervised stages")
        relate_deadline=time.monotonic()+SETTINGS["Relate_seconds"]
        aggregate_log_bytes=0
        for name,argv,kind in stages:
            if name=="paint_c0":
                headers=read_split_headers_v3(work)
                parameter_headers_ref=save(b,packet,"native_headers.json",dict(admitted_ref=record,**headers))
            require(name=="make_chunks" or parameter_headers_ref is not None,
              "No later stage before exact fresh native header validation")
            m.budget_check_v9(storage,boundary=True,required_write_bytes=OUTPUT_CAP)
            for rec in SETTINGS["native_binary_refs"].values():bounded_file(b,rec,64_000_000,retain=False)
            seconds=(relate_deadline-time.monotonic()) if kind=="relate" else SETTINGS["converter_seconds"]
            log_remaining=SETTINGS["combined_log_cap_bytes"]-aggregate_log_bytes
            require(seconds>0 and log_remaining>0,"Aggregate native time and log budget available")
            save(b,packet,name+"_prelaunch.json",dict(admitted_ref=record,argv=argv,cwd=str(work),
              controller_pid=os.getpid(),started_utc=utc(),seconds=seconds,new_owned_process_group=True,
              stage_kind=kind,aggregate_log_bytes_before_stage=aggregate_log_bytes,
              remaining_aggregate_log_cap_bytes=log_remaining,parameter_headers_ref=parameter_headers_ref))
            require(len(os.listdir("/proc/self/task"))==1,"Single-threaded controller before pre-exec limits")
            proc=None;observation=None;log=b""
            try:
                proc=subprocess.Popen(argv,cwd=work,env=dict(os.environ),stdin=subprocess.DEVNULL,
                  stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True,preexec_fn=native_limits)
            finally:
                if proc is not None:observation,log=supervise(proc,seconds,output,aux,log_remaining)
            require(observation is not None,"Native spawn failed before direct wait")
            observations.append(dict(stage=name,**observation))
            # Persist genuine wait/drain before guards, logs or another stage.
            saved=save(b,packet,name+"_direct_wait.json",dict(admitted_ref=record,stage=name,**observation))
            aggregate_log_bytes+=observation["log_total_bytes"]
            write_bytes(aux,name+".log",log);syncdir(aux)
            require(aggregate_log_bytes<=SETTINGS["combined_log_cap_bytes"],"Aggregate native log cap")
            require(observation["actual_direct_wait_observed"] is True and observation["actual_exit_code"]==0
              and observation["process_group_drained"] is True and observation["control_error"] is None,
              "No next phase after failed/unknown wait or non-drain")
            m.budget_check_v9(storage,boundary=True,required_write_bytes=AUX_WRITE)
        require([o["stage"] for o in observations]==SETTINGS["expected_native_direct_wait_stages"]
          and parameter_headers_ref is not None,"All exact eight waits before native support verification")
        raw={}
        for name in ("native.anc","native.mut","native.trees"):
            path=work/name
            require(path.is_file() and path.stat().st_size<=SETTINGS["native_file_cap_bytes"],"Bounded native output")
            rec=b.ref(path)
            raw[name]=bounded_file(b,rec,SETTINGS["native_file_cap_bytes"])
        import tskit
        require(tskit.__version__=="1.0.3","Pinned existing tskit version")
        ts=tskit.load(str(work/"native.trees"))
        facts=helper.verify_native(payloads,truth,raw["native.anc"],raw["native.mut"],ts)
        facts.update(native_direct_wait_refs={o["stage"]:b.ref(packet/(o["stage"]+"_direct_wait.json")) for o in observations},
          exact_source_build_identity_proven=False,actual_observed2074_baseline_executed=False,
          native_parameter_headers_ref=parameter_headers_ref,operational_condition=SETTINGS["operational_condition"],
          aggregate_native_log_bytes=aggregate_log_bytes)
        facts_bytes=(json.dumps(facts,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
        require(len(facts_bytes)<=SETTINGS["native_facts_cap_bytes"],"Bounded full unsupported/detail facts")
        m.budget_check_v9(storage,boundary=True,required_write_bytes=AUX_WRITE)
        write_bytes(output,"native_facts.json",facts_bytes);syncdir(output)
        facts_ref=b.ref(output/"native_facts.json")
        qualified=facts["status"]=="qualified_tiny_synthetic_native_path_only"
        require(qualified,"Native support/date/edge facts failed; preserve full native output and details")
        # Native programs have exited and groups drained. Make every retained
        # native/input/facts file durable without changing any bytes.
        for retained in output.rglob("*"):
            if retained.is_file():
                fd=os.open(retained,os.O_RDONLY|os.O_NOFOLLOW)
                try:os.fsync(fd)
                finally:os.close(fd)
        total,assets=file_inventory(output,True)
        terminal=dict(schema="H018_Relate_synthetic_qualification_completed_v1",job_id=JOB,condition_id=CONDITION,
          admitted_ref=record,status="qualified_tiny_synthetic_native_path_only",synthetic_only=True,
          fixture_sites=512,haplotypes=40,constant_sites=6,biological_or_accuracy_claim=False,
          actual_observed2074_baseline_authorized=False,sites_dropped=0,qualifier_modified_native_genotypes_or_times=False,
          facts_ref=facts_ref,assets=assets,actual_native_direct_wait_refs=facts["native_direct_wait_refs"],
          reservation_remains_fully_charged=True,no_internal_controller_OS_exit_claim=True,elapsed_seconds=time.monotonic()-started)
        write_bytes(output,"completed.json",b.encoded(terminal));syncdir(output)
        completed_ref=b.ref(output/"completed.json")
    except BaseException as exc:
        error=error_record(exc);qualified=False
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,prior)
    final=dict(schema="H018_Relate_synthetic_root_controller_result_v1",admitted_ref=record,success=qualified and completed_ref is not None,
      status="qualified_tiny_synthetic_native_path_only" if qualified and completed_ref else "failed_or_unsupported_preserved",
      completed_ref=completed_ref,facts_ref=facts_ref,error=error,actual_native_wait_observations=observations,
      controller_OS_exit_not_self_certified=True,reservation_remains_fully_charged=True,all_partial_outputs_preserved=True,
      replay_or_phase_resume_authorized=False,actual_observed2074_baseline_authorized=False,finished_utc=utc())
    result=save(b,packet,"controller_result.json",final)
    print(json.dumps(dict(controller_result_ref=result,success=final["success"])),flush=True)
    return 0 if final["success"] else 1


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("admit","launch","preflight-runtime"),nargs="?")
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--record");parser.add_argument("--sha256")
    args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="inactive_source_only",root_owns_execution=True,synthetic_only=True,native_executed=False)))
        return 0
    require(args.action is not None and Path.cwd().resolve()==ROOT,"Explicit ROOT action and repository cwd required")
    if args.action=="preflight-runtime":
        print(json.dumps(runtime_preflight_v2(),sort_keys=True));return 0
    require(args.record and args.sha256,"Explicit exact root authority ref")
    record=dict(path=args.record,sha256=args.sha256)
    if args.action=="admit":admit(record);return 0
    return launch(record)


if __name__=="__main__":
    raise SystemExit(main())
