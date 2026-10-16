"""Root-only single-parent CPU wrapper; default is inert source metadata."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse, errno, hashlib, importlib.util, json, os, re, resource
import selectors, signal, stat, subprocess, sys, time, ctypes
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/ARGMix_prospective1611_first_train_H020v3_source_20261016_v1"
SOURCE=REL+"/root_generate_v1.py"
CHILD="research/reviews/ARGMix_prospective1611_generation_execution_20261016_v1/generation_child_v1.py"
CAPS=REL+"/caps.proposed.inactive.v1.json"
MANIFEST=REL+"/source_manifest.v1.json"
GEN="research/reviews/ARGMix_prospective1611_generation_20261016_v1"
GEN_SHA="8acad46c4fc1012ddd350431495ce039059f40141aa9b09382516d7a74216fd7"
GEN_HANDOFF_SHA="561ab8e99e048f73b6039a9e19e83163ee87a5c2f96486721f101a7d90cf3ec9"
PLAN="research/reviews/ARGMix_published_panel_fair_comparison_source_plan_20261016_v1/prospective_panel_config.proposed.inactive.json"
PLAN_SHA="762e790658176a8f3c8edb507023c99d8fef93e538a7a53e9cbf5404e4c4e141"
SEEDS="research/reviews/ARGMix_published_panel_next_source_20261015_v1/published_panel_plan.v1.json"
MAP="data/ARGMix_prospective1611_HapMapII_GRCh37_chr22_v1/derivation_receipt.json"
MAP_SHA="72453ecd3fc5940d4d3fd560f7897342b10e608cc81d6155b24f21f4d57e88e6"
GUARD={'path': 'research/reviews/prospective980_H020_legacy_source_20261016_v3/prospective_980gb_H020_v3.py', 'sha256': 'ec475ecf59f0335397ee8b310cd0cb406ce4e03161e7c791861c6583799e6635'}
POLICY={'path': 'research/reviews/prospective980_H020_root_activation_20261016_v3/activation_policy.root_active_v1.json', 'sha256': '7360bd28f57c8eaef19336ccd77eba193afae554e98d2ef0726e8ec879d7e582'}
HISTORY={'path': 'research/reviews/GPU0_only_user_transition_root_20261016_v1/actual_stop_and_absence.root_v1.json', 'sha256': 'dd47aa8483c6d03ce5c45213c9ead33e0faee999103ba2db926b0f5b25384de1'}
HISTORICAL_STOP={'path': 'research/reviews/GPU0_only_user_transition_root_20261016_v1/native_GPU5_stop_signal_actual_v1.json', 'sha256': '54a2250be3a5a4facc7d6e4f6d18f95293d53063d1aeb27cca8690ac84e8922e'}
REPAIR={"path":"research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json","sha256":"f6f2d628199e8be73db8b5d90d428570d8afdefeab9dc5cd94bb5eafb39247db"}
RENEWAL={"path":"research/reviews/H039_marker_repair_renewal_source_20261016_v1/marker_repair_renewal_v1.py","sha256":"491236877e8d71cae8bd7619880858ac501c57dd93e59479ca32dacacab20468"}
BOOTSTRAP={"path":"research/reviews/H039_marker_repair_renewal_source_20261016_v1/bootstrap_renewal_v1.py","sha256":"f4ae97eeaf244373d1fcd7922e2235d08225468fe5006449d189cc461317fff2"}
CERT={"path":"research/reviews/H039_marker_repair_renewal_source_20261016_v1/renewal_certificate.root_active_v1.json","sha256":"66006202094d7a3ef2b29cbec28eb93f334505b72589f09d86e4b57498162db7"}
CORE="483d78bf2ef3c917d65ecf1dde5a7809bfadc21194c9f9a90394ce5c532fdd5e"
BASE="a8049bc103afb1cbbbce6f8fc3707a21de9613b23034599641b688994fc5caa3"
PIDFD={"path":"research/reviews/American_C3_native_MLM_full_source_v5/pidfd_backend_v1.py","sha256":"61be5d89357d1b95611e0ba26dd39d381ececf3ce6a36c347507cdc062373eca"}
LIFECYCLE={"path":"research/reviews/American_C3_native_MLM_amendment_v12/owned_lifecycle_v2.py","sha256":"93d71acc40f1037d0b37acb2ca0859300ea4bd9741409e9bd00ca21bf100d633"}
GIB=1024**3
LIMITS=dict(reservation_bytes=16*GIB,observed_bytes=8*GIB,truth_bytes=6*GIB,
    receipt_bytes=64*1024**2,truth_metadata_margin_bytes=16*1024**2,
    log_bytes=16*1024**2,auxiliary_bytes=64*1024**2,root_packet_bytes=4*1024**2,
    rss_bytes=32*GIB,cpu_threads=1,wall_seconds=21600,termination_grace_seconds=10,
    drain_seconds=30,monitor_interval_seconds=0.25,guard_interval_seconds=60)
PARENTS=("published_recon_train_000",)

def need(ok,message):
    if not ok:raise ValueError(message)

def utc():return datetime.now(timezone.utc).isoformat()

def stamp(value):
    d=datetime.fromisoformat(value.replace("Z","+00:00"))
    need(d.tzinfo is not None,"Timezone-aware real UTC timestamp required")
    return d

def local(value):
    p=Path(value);p=p if p.is_absolute() else ROOT/p
    need(p!=ROOT and p.resolve()==p and p.is_relative_to(ROOT),"Exact canonical project path")
    return p

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()

def ref(path):
    p=local(path);need(p.is_file() and not p.is_symlink(),"Regular bound file required")
    return dict(path=str(p.relative_to(ROOT)),sha256=sha(p))

def checked(record):
    need(isinstance(record,dict) and set(record)=={"path","sha256"},"Exact path/SHA reference")
    need(ref(record["path"])==record,"Bound file changed")
    return local(record["path"])

def read(record,cap=2*1024**2):
    p=checked(record);need(p.stat().st_size<=cap,"Bounded metadata only")
    return json.loads(p.read_text())

def save(path,value):
    p=local(path);raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
    need(len(raw)<=1024**2,"Bounded immutable root receipt")
    with p.open("xb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return ref(p)

def load(record,name):
    p=checked(record);s=importlib.util.spec_from_file_location(name,p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def error(exc):return dict(type=type(exc).__name__,message=str(exc)[:2048])

def inventory(directory,allowed=None):
    p=local(directory)
    if not p.exists():return 0,{}
    need(not p.is_symlink() and p.is_dir(),"Owned regular directory")
    found={};pending=[p];count=0
    while pending:
        for item in pending.pop().iterdir():
            count+=1;need(count<=64,"Owned scope entry bound")
            st=item.lstat();need(not stat.S_ISLNK(st.st_mode),"No output symlinks")
            if stat.S_ISDIR(st.st_mode):pending.append(item)
            else:
                need(stat.S_ISREG(st.st_mode) and st.st_nlink==1,"Singly linked regular owned output")
                found[str(item.relative_to(p))]=st.st_size
    if allowed is not None:need(set(found)<=set(allowed),"Unexpected owned output file")
    return sum(found.values()),found

def verify_scopes(paths):
    observed,_=inventory(paths["observed"])
    truth,tf=inventory(paths["truth"],("simulation.trees","simulation_node_mapping.json","sample_allocation.json","population_mapping.json","demography.json"))
    receipts,_=inventory(paths["receipts"],("generation_receipt.json",))
    aux,_=inventory(paths["aux"]);packet,_=inventory(paths["packet"])
    packet-=aux  # aux is the one nested monitored subtree of the root packet.
    output,all_output=inventory(paths["output"])
    need(all(k.startswith(("observed/","private_truth/","private_receipts/")) or k=="completed.json" for k in all_output),"Exact generation output layout")
    packet+=output-observed-truth-receipts  # Include output-root metadata in the metadata margin.
    need(observed<=LIMITS["observed_bytes"] and truth<=LIMITS["truth_bytes"] and receipts<=LIMITS["receipt_bytes"],"Owned observed/truth/receipt cap")
    need(sum(v for k,v in tf.items() if k!="simulation.trees")<=LIMITS["truth_metadata_margin_bytes"],"Truth metadata margin")
    need(aux<=LIMITS["auxiliary_bytes"] and packet<=LIMITS["root_packet_bytes"],"Owned auxiliary/root metadata cap")
    need(observed+truth+receipts+aux+packet<=LIMITS["reservation_bytes"],"Combined16GiB scope accounting")
    return dict(observed=observed,truth=truth,receipts=receipts,auxiliary=aux,root_packet=packet)

def validate(record):
    r=read(record)
    need(r.get("schema")=="prospective1611_first_train_H020v3_root_request_v1" and r.get("active") is True
        and r.get("root_authorizer")=="root_agent" and r.get("registered_before_execution") is True,"Actual root request required")
    need(r.get("parent_id") in PARENTS and re.fullmatch("[A-Za-z0-9_-]{1,80}",r.get("scope_id","")),"One registered parent/fresh scope")
    need(r.get("automatic_retry") is False and r.get("downscale_or_seed_fallback") is False,"No automatic reruns/downscale")
    now=datetime.now(timezone.utc);expires=stamp(r["expires_at_utc"])
    need(stamp(r["created_at_utc"])<=now and now+timedelta(seconds=LIMITS["wall_seconds"]+600)<expires
        and expires-now<=timedelta(hours=8),"Current root authority covers6h plus owned drain")
    need(r["wrapper_ref"]==ref(SOURCE) and r["caps_ref"]==ref(CAPS) and r["source_manifest_ref"]==ref(MANIFEST),"Exact wrapper/cap/manifest")
    caps=read(r["caps_ref"]);need(caps["active"] is False and caps["limits"]==LIMITS,"Exact proposed operational limits selected by root request")
    manifest=read(r["source_manifest_ref"])
    need(manifest["wrapper_ref"]==r["wrapper_ref"] and manifest["caps_ref"]==r["caps_ref"],"Exact source closure")
    pins=[dict(path=x["path"],sha256=x["sha256"]) for x in manifest["upstream_refs"]]
    for x in pins:checked(x)
    need(ref(GEN+"/generation.py")["sha256"]==GEN_SHA and ref(GEN+"/source_handoff.v1.json")["sha256"]==GEN_HANDOFF_SHA
        and ref(PLAN)["sha256"]==PLAN_SHA and ref(MAP)["sha256"]==MAP_SHA,"Sealed generator/science/full-map")
    gp=json.loads(local(GEN+"/source_pins.v1.json").read_text())
    for x in gp["sources"]:checked(dict(path=x["path"],sha256=x["sha256"]))
    need(all(dict(path=x["path"],sha256=x["sha256"]) in pins for x in gp["sources"]),"Every sealed upstream pin in closure")
    need(all(x in pins for x in (ref(CHILD),PIDFD,LIFECYCLE,GUARD,POLICY,REPAIR,RENEWAL,BOOTSTRAP,CERT,HISTORY,HISTORICAL_STOP)),"Exact launcher/lifecycle/guard/renewal closure")
    md=read(dict(path=MAP,sha256=MAP_SHA))
    need(md["complete"] is True and md["length_bp"]==35178458 and md["source_interval_bp"]==[16051347,51229805],"Exact full primary map")
    runtime=read(r["runtime_binding_ref"]);qualification=read(r["runtime_qualification_ref"])
    need(runtime.get("schema")=="prospective1611_cpu_runtime_binding_v1" and runtime.get("root_qualified") is True
        and runtime.get("cpu_only") is True and set(runtime.get("modules",{}))=={"msprime","numpy","tskit"},"Actual qualified CPU runtime binding")
    need(runtime["qualification_receipt_sha256"]==r["runtime_qualification_ref"]["sha256"]
        and qualification.get("root_authorizer")=="root_agent"
        and qualification.get("schema")=="root_prospective1611_CPU_module_qualification_v1"
        and qualification.get("modules")==runtime["modules"]
        and qualification.get("generator_source_ref")==dict(path=GEN+"/generation.py",sha256=GEN_SHA)
        and qualification.get("GPU_context_created") is False and qualification.get("full_simulation_executed") is False
        and qualification.get("model_executed") is False,"Actual root CPU import/source qualification; no full-simulation claim")
    python=checked(r["python_executable_ref"])
    need(sha(Path(sys.executable).resolve())==runtime["python_executable_sha256"]==r["python_executable_ref"]["sha256"]
        and Path(sys.executable).resolve()==python
        and qualification.get("python_executable_sha256")==runtime["python_executable_sha256"]
        and local(qualification["python_executable"])==python,"Wrapper and generator use exact qualified executable")
    runtime_files=[r["python_executable_ref"]]
    for module in runtime["modules"].values():
        x=ref(module["file"]);need(x["sha256"]==module["sha256"] and bool(module["version"]),"Qualified module source identity")
        runtime_files.append(x)
    for x in r["runtime_support_refs"]:checked(x)
    need(r.get("historical_native_lineage_ref")==HISTORY
        and r.get("historical_native_stop_ref")==HISTORICAL_STOP
        and r.get("historical_native_success_claimed") is False
        and r.get("native_success_serial_prerequisite") is False,"Frozen original TRAIN000 lineage disclosure; no success prerequisite")
    history=read(HISTORY);stopped=read(HISTORICAL_STOP)
    need(history.get("schema")=="root_GPU0_only_actual_stop_and_absence_v1"
        and history["native_tool"]["terminal_consumed"] is True
        and history["native_tool"]["actual_exit_code"]==1
        and history["native_tool"]["actual_native_wait"]==-15
        and history["native_tool"]["actual_entry_wait"]==-15
        and history["native_tool"]["actual_launcher_wait"]==-15
        and history["native_tool"]["native_full_budget_completed"] is False
        and history["files_preserved"] is True and history["scope_credit_claimed"] is False,
        "Immutable stopped original TRAIN000 record, never a successful full fit or credit")
    need(stopped.get("schema")=="root_GPU0_only_native_supervisor_stop_v1"
        and stopped.get("signal")=="SIGTERM" and stopped.get("signal_sent_through_pidfd") is True
        and stopped.get("checkpoint_files_preserved") is True
        and stopped.get("actual_wait_pending") is True
        and stamp(stopped["observed_utc"])<=stamp(history["observed_utc"])<=now,
        "Earlier stop/checkpoint metadata and later consumed terminal remain distinct")
    capacity=read(r["capacity_verification_ref"])
    need(capacity.get("schema")=="prospective1611_first_train_current_capacity_root_verified_v1"
        and capacity.get("root_authorizer")=="root_agent"
        and capacity.get("root_verified_for_independent_CPU_generation") is True
        and capacity.get("current_CPU_and_memory_observation_actual") is True
        and capacity.get("current_storage_capacity_observation_actual") is True
        and capacity.get("native_success_required") is False
        and capacity.get("guard_mode")=="prospective_980gb_H020_v3"
        and capacity.get("guard_ref")==GUARD and capacity.get("policy_ref")==POLICY
        and capacity.get("parent_id")==r["parent_id"] and capacity.get("scope_id")==r["scope_id"]
        and capacity.get("wrapper_ref")==r["wrapper_ref"]
        and capacity.get("reservation_bytes")==LIMITS["reservation_bytes"] and capacity.get("rss_bytes")==LIMITS["rss_bytes"]
        and capacity.get("cpu_threads")==1 and capacity.get("available_scope_bytes",0)>=LIMITS["reservation_bytes"]
        and capacity.get("available_cpu_rss_bytes",0)>=LIMITS["rss_bytes"] and capacity.get("single_cpu_available") is True
        and capacity.get("source_only_estimate") is False,"Actual current root CPU/memory/storage capacity; live guard admission remains mandatory")
    need(type(r.get("cpu_id")) is int and r["cpu_id"]>=0 and capacity.get("cpu_id")==r["cpu_id"]
        and r["cpu_id"] in os.sched_getaffinity(0),"Root-verified explicit CPU within own allowed affinity")
    need(stamp(capacity["verified_at_utc"])<=now
        and (now-stamp(capacity["verified_at_utc"])).total_seconds()<=900,"Fresh actual current capacity, <=15minutes")
    need(expires+timedelta(seconds=600)<stamp(read(CERT)["renewed_expires_at_utc"]),"Renewal remains live through allocation expiry+drain")
    prefix="data/ARGMix_prospective1611_first_train_H020v3_generated_"+r["scope_id"]
    packet=local(REL+"/admitted_"+r["scope_id"])
    paths=dict(output=local(prefix),observed=local(prefix+"/observed"),truth=local(prefix+"/private_truth"),
        receipts=local(prefix+"/private_receipts"),packet=packet,aux=packet/"worker_aux")
    need(not any(p.exists() for p in (paths["output"],packet)),"Fresh exclusive one-parent paths; no replay/resume")
    return r,pins,runtime_files,paths

def owned_rss(pid):
    total=0
    for member in (os.getpid(),pid):
        try:
            lines=Path("/proc/%d/status"%member).read_text().splitlines()
            values=[int(x.split()[1])*1024 for x in lines if x.startswith("VmRSS:")]
            zombie=member==pid and any(x.startswith("State:") and x.split()[1] in ("Z","X") for x in lines)
            need(len(values)==1 or zombie,"Actual Linux owned-process RSS available")
            total+=values[0] if values else 0
        except FileNotFoundError:
            need(member==pid,"Controller RSS must remain available")
    return total

def child_limits(cpu):
    os.sched_setaffinity(0,{cpu})
    resource.setrlimit(resource.RLIMIT_FSIZE,(LIMITS["truth_bytes"]-LIMITS["truth_metadata_margin_bytes"],)*2)
    resource.setrlimit(resource.RLIMIT_CPU,(LIMITS["wall_seconds"],)*2)
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))

def supervise(proc,paths,guard,storage,started,cancel,lifecycle,backend,gate_fd):
    selector=owned=leader_fd=None;failure=None;peak=total=0;waited=False;status=None;captured=bytearray();adopted=[]
    try:
        # This exact direct Popen child remains unreaped while its pidfd is opened.
        leader_fd=backend.pidfd_open_v1(proc.pid,0);need(not os.get_inheritable(leader_fd),"CLOEXEC owned pidfd")
        owned=lifecycle.OwnedTree(proc.pid)
        need(os.write(gate_fd,b"GO\n")==3,"Release exactly one owned numerical child")
        os.close(gate_fd);gate_fd=None
        selector=selectors.DefaultSelector();os.set_blocking(proc.stdout.fileno(),False)
        selector.register(proc.stdout,selectors.EVENT_READ);eof=False;last_guard=started;terminal_seen=None
        while True:
            need(not cancel["signal"],"Root cancellation signal %s"%cancel["signal"])
            need(time.monotonic()-started<=LIMITS["wall_seconds"],"21600s wall ceiling")
            owned.refresh();need(len(owned.known)==1,"Unexpected owned descendant; generation has no worker capability")
            current=owned_rss(proc.pid);peak=max(peak,current)
            need(current<=LIMITS["rss_bytes"],"Monitored combined32GiB RSS ceiling")
            verify_scopes(paths)
            if time.monotonic()-last_guard>=LIMITS["guard_interval_seconds"]:
                guard.budget_check_v9(storage,boundary=True);last_guard=time.monotonic()
            for key,events in selector.select(LIMITS["monitor_interval_seconds"]):
                chunk=os.read(proc.stdout.fileno(),65536)
                if not chunk:
                    eof=True;selector.unregister(proc.stdout)
                else:
                    total+=len(chunk);captured.extend(chunk[:max(0,LIMITS["log_bytes"]-len(captured))])
                    need(total<=LIMITS["log_bytes"],"Combined stdout/stderr16MiB ceiling")
            terminal=os.waitid(os.P_PID,proc.pid,os.WEXITED|os.WNOHANG|os.WNOWAIT)
            if terminal is not None:
                if eof:break
                terminal_seen=terminal_seen or time.monotonic()
                need(time.monotonic()-terminal_seen<=LIMITS["drain_seconds"],"Terminal output pipe drain deadline")
    except BaseException as exc:failure=error(exc)
    finally:
        if gate_fd is not None:
            try:os.close(gate_fd)
            except BaseException as exc:failure=failure or error(exc)
        terminate_at=None;killed=False
        # Reviewed pidfd backend supplies the existing syscall fallback. No positive
        # numeric PID/group signal or unavailable native Python pidfd API is used.
        while True:
            try:
                terminal=os.waitid(os.P_PID,proc.pid,os.WEXITED|os.WNOHANG|os.WNOWAIT)
                if failure is not None:
                    if terminate_at is None:
                        if owned is not None:owned.signal_all(signal.SIGTERM)
                        elif leader_fd is not None:
                            try:backend.pidfd_send_signal_v1(leader_fd,signal.SIGTERM,None,0)
                            except ProcessLookupError:pass
                        terminate_at=time.monotonic()
                    if time.monotonic()-terminate_at>=LIMITS["termination_grace_seconds"] and not killed:
                        if owned is not None:owned.signal_all(signal.SIGKILL)
                        elif leader_fd is not None:
                            try:backend.pidfd_send_signal_v1(leader_fd,signal.SIGKILL,None,0)
                            except ProcessLookupError:pass
                        killed=True
                if terminal is not None:
                    pid,status=os.waitpid(proc.pid,0);need(pid==proc.pid,"Exact owned direct wait")
                    proc.returncode=os.waitstatus_to_exitcode(status);waited=True;break
                time.sleep(0.05)
            except ChildProcessError as exc:
                failure=failure or error(exc);break
            except BaseException as exc:
                failure=failure or error(exc)
        end=time.monotonic()+LIMITS["drain_seconds"];drained=False
        while waited and owned is not None and time.monotonic()<end:
            try:
                owned.refresh()
                if owned.live():
                    failure=failure or dict(type="UnexpectedDescendant",message="Child left a live owned process")
                    owned.signal_all(signal.SIGKILL)
                for pid,row in list(owned.known.items()):
                    if pid==proc.pid:continue
                    actual=lifecycle._stat(pid)
                    if not lifecycle._same(actual,row) or actual["ppid"]!=os.getpid():continue
                    try:
                        waited_pid,extra_status=os.waitpid(pid,os.WNOHANG)
                        if waited_pid==pid:adopted.append(dict(pid=pid,actual_OS_wait_status=extra_status,returncode=os.waitstatus_to_exitcode(extra_status)))
                    except ChildProcessError:pass
                drained=owned.identities_absent() and not lifecycle._children(os.getpid())
                if drained:break
                time.sleep(0.05)
            except BaseException as exc:failure=failure or error(exc)
        # Drain immediately available diagnostics after actual waits; retain at most16MiB.
        try:
            os.set_blocking(proc.stdout.fileno(),False)
            while time.monotonic()<end:
                try:chunk=os.read(proc.stdout.fileno(),65536)
                except BlockingIOError:
                    if drained:break
                    time.sleep(0.05);continue
                if not chunk:break
                total+=len(chunk);captured.extend(chunk[:max(0,LIMITS["log_bytes"]-len(captured))])
        except BaseException as exc:failure=failure or error(exc)
        for item in (selector,proc.stdout):
            if item is not None:
                try:item.close()
                except BaseException as exc:failure=failure or error(exc)
        if leader_fd is not None:
            try:os.close(leader_fd)
            except BaseException as exc:failure=failure or error(exc)
        if total>LIMITS["log_bytes"]:failure=failure or dict(type="LogLimit",message="Combined output cap")
        if not drained:failure=failure or dict(type="DrainFailure",message="Owned identities not proved absent; no completion claim")
        observation=dict(schema="prospective1611_generation_actual_direct_wait_v1",root_pid=os.getpid(),child_pid=proc.pid,
            process_group=proc.pid,actual_direct_worker_wait=waited,actual_OS_wait_status=status,
            returncode=proc.returncode if waited else None,all_owned_processes_and_groups_drained=drained,
            drain_method="subreaper_owned_identity_roster_absence_no_global_scan",adopted_child_actual_waits=adopted,
            owned_lifecycle_evidence=owned.evidence() if owned is not None else None,
            operational_failure=failure,peak_combined_RSS_bytes=peak,elapsed_seconds=time.monotonic()-started,
            captured_log_bytes=len(captured),log_total_bytes=total,observed_utc=utc())
        try:print(json.dumps(dict(actual_direct_wait_observation=observation)),flush=True)
        except BaseException as exc:observation["receipt_stdout_error"]=error(exc)
    return observation,bytes(captured)


def execute(record):
    r,pins,runtime_files,paths=validate(record)
    sys.dont_write_bytecode=True
    sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"scripts")]
    need("pidfd_backend_v1" not in sys.modules,"Fresh pidfd helper import identity")
    backend=load(PIDFD,"pidfd_backend_v1");sys.modules["pidfd_backend_v1"]=backend
    lifecycle=load(LIFECYCLE,"_prospective1611_owned_lifecycle")
    renewal=load(RENEWAL,"_prospective1611_fresh_renewal")
    renewal.install_renewal_v1(CERT)  # Before any original guard/marker consumer.
    guard=load(GUARD,"_prospective1611_first_train_H020v3_guard");guard.configure_v2(REPAIR,policy_ref=POLICY)
    view=guard.inspect_v1(expected_source_sha256=CORE,expected_launcher_sha256=BASE)
    paths["packet"].mkdir()
    closure=save(paths["packet"]/"pre_reserve_closure.json",dict(schema="prospective1611_generation_consumer_closure_v1",
        request_ref=record,wrapper_ref=r["wrapper_ref"],caps_ref=r["caps_ref"],source_manifest_ref=r["source_manifest_ref"],
        upstream_refs=pins,runtime_files=runtime_files,runtime_support_refs=r["runtime_support_refs"],
        runtime_binding_ref=r["runtime_binding_ref"],runtime_qualification_ref=r["runtime_qualification_ref"],
        capacity_verification_ref=r["capacity_verification_ref"],historical_native_lineage_ref=HISTORY,
        historical_native_stop_ref=HISTORICAL_STOP,historical_native_success_claimed=False,native_success_serial_prerequisite=False,renewal_refs=[RENEWAL,BOOTSTRAP,CERT],owned_lifecycle_refs=[PIDFD,LIFECYCLE],
        scientific_change=False,automatic_retry=False))
    candidates=[record,r["wrapper_ref"],r["caps_ref"],r["source_manifest_ref"],closure,*pins,
        r["runtime_binding_ref"],r["runtime_qualification_ref"],r["capacity_verification_ref"],
        HISTORY,HISTORICAL_STOP,*runtime_files,*r["runtime_support_refs"],
        GUARD,POLICY,REPAIR,RENEWAL,BOOTSTRAP,CERT,PIDFD,LIFECYCLE,*view["bindings"].values()]
    bindings=list({(x["path"],x["sha256"]):x for x in candidates if Path(x["path"]).suffix in (".py",".json",".sh")}.values())
    need(len(bindings)<=64,"Original guard64-ref bound; never silently drop eligible source/runtime authorities")
    auth=guard.operation_draft_v1("reserve",dict(key="ARGMix/prospective1611/first_train_H020v3/generate/"+r["scope_id"],
        output=str(paths["output"]),auxiliary_scopes=[str(paths["aux"])],
        reserved_bytes=LIMITS["reservation_bytes"],consumer_bindings=bindings),
        expected_before=guard._pins(view),bindings=view["bindings"])
    auth.update(active=True,expires_at_utc=r["expires_at_utc"])
    admitted=guard.apply_reserve_v1(save(paths["packet"]/"reserve_authority.json",auth),expected_source_sha256=CORE,expected_launcher_sha256=BASE)
    admission_ref=save(paths["packet"]/"reserve_receipt.json",admitted)
    storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,accounting_mode=guard.MODE,
        job_id="prospective1611_generate_"+r["scope_id"],output_directory=str(paths["output"].relative_to(ROOT)),
        reserved_new_bytes=LIMITS["reservation_bytes"],reservation=admitted["value"],initial_ledger_pins=admitted["initial_ledger_pins"],
        expires_at_utc=r["expires_at_utc"],completed_overlay_policy_ref=POLICY,
        budget_components=dict(implementation=ref(guard.budget.SOURCE_REL),launcher=ref(guard.budget.LAUNCHER_REL),consumer_adapter=GUARD))
    save(paths["packet"]/"storage.json",storage)
    resource_ref=save(paths["packet"]/"generator_resource_binding.json",dict(schema="prospective1611_generation_resource_binding_v1",
        root_approved=True,enforced_by_root_wrapper=True,cpu_threads=1,rss_bytes=LIMITS["rss_bytes"],
        wall_seconds=LIMITS["wall_seconds"],observed_output_bytes=LIMITS["observed_bytes"],private_truth_bytes=LIMITS["truth_bytes"],
        private_receipt_bytes=LIMITS["receipt_bytes"],wrapper_source_sha256=r["wrapper_ref"]["sha256"],admission_receipt_sha256=admission_ref["sha256"]))
    generation_request=save(paths["packet"]/"generator_request.json",dict(schema="prospective1611_generation_request_v1",
        root_approved=True,parent_id=r["parent_id"],science_plan_sha256=PLAN_SHA,generator_source_sha256=GEN_SHA,
        map_derivation_receipt=str(local(MAP)),map_derivation_receipt_sha256=MAP_SHA,
        runtime_binding=str(local(r["runtime_binding_ref"]["path"])),runtime_binding_sha256=r["runtime_binding_ref"]["sha256"],
        resource_binding=str(local(resource_ref["path"])),resource_binding_sha256=resource_ref["sha256"],
        observed_output_dir=str(paths["observed"]),private_truth_output_dir=str(paths["truth"]),private_receipt_dir=str(paths["receipts"])))
    guard.budget_check_v9(storage,boundary=True)
    paths["output"].mkdir();paths["aux"].mkdir()
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",VECLIB_MAXIMUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",
        PYTHONPATH=str(ROOT/"src"),TMPDIR=str(paths["aux"]),LC_ALL="C",LANG="C")
    cpu=r["cpu_id"];os.sched_setaffinity(0,{cpu})
    gate_read,gate_write=os.pipe()
    argv=[str(local(r["python_executable_ref"]["path"])),"-B",str(local(CHILD)),"--gate-fd",str(gate_read),
        "--request",str(local(generation_request["path"])),"--request-sha256",generation_request["sha256"]]
    save(paths["packet"]/"launch.json",dict(argv=argv,CPU_affinity=[cpu],limits=LIMITS,RLIMIT_FSIZE_bytes=LIMITS["truth_bytes"]-LIMITS["truth_metadata_margin_bytes"],
        RSS_enforcement="monitored combined owned wrapper+child /proc VmRSS; terminate on exceed; no RLIMIT_RSS",root_request_ref=record))
    need(not lifecycle._children(os.getpid()),"Fresh controller has no unrelated child processes")
    libc=ctypes.CDLL(None,use_errno=True)
    need(libc.prctl(36,1,0,0,0)==0,"Enable Linux child subreaper for owned descendant drain")
    cancel={"signal":None}
    def cancellation(signum,frame):cancel["signal"]=signum
    previous={sig:signal.signal(sig,cancellation) for sig in (signal.SIGTERM,signal.SIGINT)}
    proc=None;started=time.monotonic()
    try:
        proc=subprocess.Popen(argv,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            start_new_session=True,close_fds=True,pass_fds=(gate_read,),preexec_fn=lambda:child_limits(cpu))
        os.close(gate_read);gate_read=None
        owned_gate=gate_write;gate_write=None
        observation,log=supervise(proc,paths,guard,storage,started,cancel,lifecycle,backend,owned_gate)
    finally:
        for fd in (gate_read,gate_write):
            if fd is not None:
                try:os.close(fd)
                except OSError:pass
        for sig,handler in previous.items():signal.signal(sig,handler)
    wait_ref=save(paths["packet"]/"actual_wait.json",observation)
    with (paths["aux"]/"child.log").open("xb") as f:f.write(log)
    need(not cancel["signal"],"Root cancellation after direct wait; preserve incomplete scope")
    need(observation["returncode"]==0 and observation["operational_failure"] is None
        and observation["actual_direct_worker_wait"] and observation["all_owned_processes_and_groups_drained"],"Failed/incomplete parent; no retry")
    sizes=verify_scopes(paths);guard.budget_check_v9(storage,boundary=True)
    observed_ref=ref(paths["observed"]/"observed_generation_receipt.json")
    private_ref=ref(paths["receipts"]/"generation_receipt.json")
    observed,private=read(observed_ref),read(private_ref)
    need(observed["complete"] is True and private["complete"] is True and observed["parent_id"]==private["parent_id"]==r["parent_id"]
        and observed["haplotypes"]==3222 and private["request"]["sha256"]==generation_request["sha256"]
        and private["resource_binding_sha256"]==resource_ref["sha256"],"Actual exact full-parent completion metadata")
    completed=save(paths["output"]/"completed.json",dict(schema="prospective1611_generation_root_completed_v1",
        status="completed_prospective1611_full_parent_generation",parent_id=r["parent_id"],length_bp=35178458,
        haplotypes=3222,root_request_ref=record,wrapper_ref=r["wrapper_ref"],generation_request_ref=generation_request,
        actual_wait_ref=wait_ref,observed_generation_receipt_ref=observed_ref,private_generation_receipt_ref=private_ref,
        storage_ref=ref(paths["packet"]/"storage.json"),measured_bytes=sizes,completed_utc=utc(),inference_performed=False))
    guard.budget_check_v9(storage,boundary=True)
    print(json.dumps(dict(completed_ref=completed,actual_wait_ref=wait_ref)),flush=True)
    return 0

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--execute",action="store_true")
    parser.add_argument("--request");parser.add_argument("--request-sha256");a=parser.parse_args()
    if not a.execute:
        print(json.dumps(dict(status="source_only_no_guard_no_launch",limits=LIMITS,parent_ids=PARENTS)));return 0
    need(Path.cwd()==ROOT and a.request and a.request_sha256,"Exact repository cwd and explicit actual request/hash")
    return execute(dict(path=a.request,sha256=a.request_sha256))

if __name__=="__main__":raise SystemExit(main())
