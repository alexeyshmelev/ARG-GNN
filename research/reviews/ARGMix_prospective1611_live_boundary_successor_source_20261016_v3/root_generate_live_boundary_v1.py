"""Root-only single-parent CPU wrapper; default is inert source metadata."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse, errno, hashlib, importlib.util, json, os, re, resource
import selectors, signal, stat, subprocess, sys, time, ctypes
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/ARGMix_prospective1611_live_boundary_successor_source_20261016_v3"
SOURCE=REL+"/root_generate_live_boundary_v1.py"
CHILD="research/reviews/ARGMix_prospective1611_generation_execution_20261016_v1/generation_child_v1.py"
CAPS=REL+"/caps.source_only.v1.json"
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
OLD_WRAPPER_REF={'path': 'research/reviews/ARGMix_prospective1611_first_train_H020v3_source_20261016_v1/root_generate_v1.py', 'sha256': '302c4216e2f95272a83903ec476342e6d0b8999b6bbacb8a01a339d18c2693ed'}
OLD_STORAGE_REF={'path': 'research/reviews/ARGMix_prospective1611_first_train_H020v3_source_20261016_v1/admitted_train000_oct16_root_v1/storage.json', 'sha256': '76c8e8a49a1330937298d059633676ec7eb1f06bab5f16d134b94a95d8b70b37'}
OLD_WAIT_REF={'path': 'research/reviews/ARGMix_prospective1611_first_train_H020v3_source_20261016_v1/admitted_train000_oct16_root_v1/actual_wait.json', 'sha256': '560c4f13b1e06fd954af00c8345ae725741ab106f40e45a9b8180d974680c84f'}
OLD_REQUEST_REF={'path': 'research/reviews/ARGMix_prospective1611_first_train_H020v3_source_20261016_v1/root_actual_request_train000_v1.json', 'sha256': 'c1a61dc347deff47c5731040af7895c99f8c6fe896afbdd4de13836a16ac9b47'}
OLD_LAUNCH_REF={'path': 'research/reviews/ARGMix_prospective1611_first_train_H020v3_source_20261016_v1/admitted_train000_oct16_root_v1/launch.json', 'sha256': '5149ed02f476f9c9c7a5904f0fda28760a7b8b6b04abcfa5e6e2f4fe830cf7e7'}
OLD_ADMISSION_REF={'path': 'research/reviews/ARGMix_prospective1611_first_train_H020v3_source_20261016_v1/admitted_train000_oct16_root_v1/reserve_receipt.json', 'sha256': 'b03972e23ea66082601636ae0f486a47a0f7935c7625744368a42269a00b31ee'}
OLD_OUTPUT='/workspace-SR003.nfs2/ARG-GNN/data/ARGMix_prospective1611_first_train_H020v3_generated_train000_oct16_root_v1'
OLD_AUX='/workspace-SR003.nfs2/ARG-GNN/research/reviews/ARGMix_prospective1611_first_train_H020v3_source_20261016_v1/admitted_train000_oct16_root_v1/worker_aux'


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
                guard.budget_check_v9(storage,boundary=False);last_guard=time.monotonic()
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

OLD_REL="research/reviews/ARGMix_prospective1611_first_train_H020v3_source_20261016_v1"
SCOPE_ID="train000_live_boundary_root_v1"
_LIVE_IDS={}
_RETAINED={}
_PATHS=None

def identity(st):
    return (st.st_dev,st.st_ino,st.st_mode,st.st_nlink if stat.S_ISREG(st.st_mode) else None)
def frozen_stat(st):
    return identity(st)+(st.st_nlink,st.st_size,st.st_blocks,st.st_mtime_ns,st.st_ctime_ns)
def census(directory):
    """Descriptor-relative metadata only. Growth is allowed; identity swaps are not."""
    directory=local(directory)
    if not directory.exists():return {}
    rows={};count=0
    def walk(fd,path,depth):
        nonlocal count
        need(depth<=12,"Bounded own namespace depth")
        before=os.fstat(fd);need(stat.S_ISDIR(before.st_mode),"Own directory type")
        rows[str(path)]=dict(identity=identity(before),frozen=frozen_stat(before),bytes=max(4096,before.st_size,before.st_blocks*512),directory=True)
        count+=1;need(count<=64,"Original64-entry per-scope live bound")
        with os.scandir(fd) as entries:names=sorted(e.name for e in entries)
        need(len(names)<=64,"Bounded own directory membership")
        for name in names:
            need(name not in (".","..") and "/" not in name,"Flat child name")
            observed=os.stat(name,dir_fd=fd,follow_symlinks=False)
            need(stat.S_ISDIR(observed.st_mode) or (stat.S_ISREG(observed.st_mode) and observed.st_nlink==1),"No symlink, special file or hardlink")
            flags=os.O_RDONLY|os.O_NOFOLLOW
            if stat.S_ISDIR(observed.st_mode):flags|=os.O_DIRECTORY
            child=os.open(name,flags,dir_fd=fd)
            try:
                actual=os.fstat(child)
                need(identity(actual)==identity(observed),"Opened owned identity changed")
                target=path/name
                if stat.S_ISDIR(actual.st_mode):walk(child,target,depth+1)
                else:
                    count+=1;need(count<=64,"Original64-entry per-scope live bound")
                    later=os.fstat(child)
                    need(identity(later)==identity(actual),"Owned file identity changed")
                    rows[str(target)]=dict(identity=identity(actual),frozen=frozen_stat(later),bytes=max(actual.st_size,actual.st_blocks*512,later.st_size,later.st_blocks*512),directory=False)
                named=os.stat(name,dir_fd=fd,follow_symlinks=False)
                need(identity(named)==identity(actual),"Owned pathname replacement")
            finally:os.close(child)
        need(identity(os.fstat(fd))==identity(before),"Owned directory identity changed")
    fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:walk(fd,directory,0)
    finally:os.close(fd)
    return rows

def under(name,root):
    return Path(name).is_relative_to(root)

def measured(paths):
    rows={}
    for root in (paths["charged_output"],paths["charged_aux"]):
        now=census(root)
        need(not set(rows)&set(now),"Disjoint original charge roots")
        rows.update(now)
    # Original packet metadata was budgeted separately outside its aux scope.
    packet={k:v for k,v in census(paths["old_packet"]).items() if not under(k,paths["charged_aux"])}
    return rows,packet

def verify_scopes(paths):
    global _LIVE_IDS
    rows,old_packet=measured(paths)
    for name,row in rows.items():
        need(name not in _LIVE_IDS or row["identity"]==_LIVE_IDS[name],"Owned identity changed: "+name)
    need(set(_LIVE_IDS)<=set(rows),"Owned file disappeared; generator has no delete/rename capability")
    _LIVE_IDS.update({k:v["identity"] for k,v in rows.items()})
    for name,before in _RETAINED.items():
        current=rows.get(name) or old_packet.get(name)
        need(current is not None and current["frozen"]==before,"Retained failed artifact changed: "+name)
    allowed_observed={"observed.vcf","observed.haps","observed.sample","variants.tsv","haplotypes.tsv","roster.unbound.json","map_binding.json","observed_generation_receipt.json"}
    allowed_truth={"simulation.trees","simulation_node_mapping.json","sample_allocation.json","population_mapping.json","demography.json"}
    def group(root,allowed=None):
        group_rows={k:v for k,v in rows.items() if under(k,root)}
        if allowed is not None:
            need(all(v["directory"] or str(Path(k).relative_to(root)) in allowed for k,v in group_rows.items()),"Unexpected successor output name")
        return sum(v["bytes"] for v in group_rows.values() if not v["directory"])
    observed=group(paths["observed"],allowed_observed)
    truth=group(paths["truth"],allowed_truth)
    receipts=group(paths["receipts"],{"generation_receipt.json"})
    truth_meta=sum(v["bytes"] for k,v in rows.items() if under(k,paths["truth"]) and not v["directory"] and Path(k).name!="simulation.trees")
    auxiliary=group(paths["aux"])
    packet=sum(v["bytes"] for k,v in rows.items() if under(k,paths["packet"]) and not under(k,paths["aux"]) and not v["directory"])+sum(v["bytes"] for v in old_packet.values())
    output_metadata=sum(v["bytes"] for k,v in rows.items() if under(k,paths["output"]) and not any(under(k,paths[x]) for x in ("observed","truth","receipts")) and not v["directory"])
    packet+=output_metadata
    for name,row in rows.items():
        if under(name,paths["output"]):
            rel=Path(name).relative_to(paths["output"])
            need(not rel.parts or rel.parts[0] in ("observed","private_truth","private_receipts","completed.json"),"Exact fresh generation layout")
    need(observed<=LIMITS["observed_bytes"] and truth<=LIMITS["truth_bytes"] and receipts<=LIMITS["receipt_bytes"],"Successor observed/truth/receipt cap")
    need(truth_meta<=LIMITS["truth_metadata_margin_bytes"],"Truth metadata margin")
    need(auxiliary<=LIMITS["auxiliary_bytes"] and packet<=LIMITS["root_packet_bytes"],"Successor auxiliary/original+new packet cap")
    total=sum(v["bytes"] for v in rows.values())+sum(v["bytes"] for v in old_packet.values())
    need(total<=LIMITS["reservation_bytes"],"Retained failed plus successor allocation exceeds original16GiB charge")
    return dict(observed=observed,truth=truth,receipts=receipts,auxiliary=auxiliary,root_packet=packet,combined_original_charge_bytes=total,retained_failed_artifacts_preserved=True,measurement="sampled_conservative_allocated_bytes_not_OS_quota")

def direct_refs(record,r,manifest):
    refs=[record,ref(SOURCE),ref(CAPS),ref(MANIFEST),OLD_STORAGE_REF,OLD_WAIT_REF,OLD_REQUEST_REF,OLD_LAUNCH_REF,OLD_ADMISSION_REF,
          r["old_outer_terminal_ref"],r["old_future_write_revocation_ref"],r["capacity_verification_ref"],
          r["runtime_binding_ref"],r["runtime_qualification_ref"]]
    refs+=manifest["direct_operational_refs"]
    refs+=r["runtime_support_refs"]
    refs=list({(x["path"],x["sha256"]):x for x in refs if Path(x["path"]).suffix in (".py",".json",".sh")}.values())
    need(1<=len(refs)<=64,"Original64 direct operational-ref ceiling; never omit a required role")
    for x in refs:checked(x)
    return refs

def validate_successor(record):
    r=read(record);now=datetime.now(timezone.utc)
    need(r.get("schema")=="prospective1611_live_boundary_operational_successor_root_request_v1" and r.get("active") is True and r.get("root_authorizer")=="root_agent" and r.get("registered_before_execution") is True,"Actual ROOT-only operational successor authority")
    need(r.get("manual_operational_successor_authorized") is True and r.get("automatic_retry") is False and r.get("downscale_or_seed_fallback") is False and r.get("new_storage_admission_claimed") is False and r.get("historical_binding_mutation_authorized") is False,"Explicit manual operational successor, no retry/admission/rewrite fiction")
    need(r.get("scope_id")==SCOPE_ID and r.get("parent_id")=="published_recon_train_000" and r.get("cpu_id")==12,"One fresh full original parent on CPU12")
    need(r["wrapper_ref"]==ref(SOURCE) and r["caps_ref"]==ref(CAPS) and r["source_manifest_ref"]==ref(MANIFEST),"Exact reviewed successor source")
    caps=read(r["caps_ref"]);need(caps["active"] is False and caps["limits"]==LIMITS,"Unchanged full operational caps")
    old=read(OLD_REQUEST_REF);storage=read(OLD_STORAGE_REF);wait=read(OLD_WAIT_REF);launch=read(OLD_LAUNCH_REF)
    need(storage["reserved_new_bytes"]==storage["reservation"]["reserved_bytes"]==16*GIB and len(storage["reservation"]["consumer_bindings"])==64,"Full original16GiB charge and exact64 bindings retained")
    need(storage["reservation"]["output"]==OLD_OUTPUT and storage["reservation"]["auxiliary_scopes"]==[OLD_AUX] and storage["accounting_mode"]=="prospective_980gb_H020_v3","Exact measured original output+aux roots")
    need(wait["schema"]=="prospective1611_generation_actual_direct_wait_v1" and wait["actual_direct_worker_wait"] is True and wait["returncode"]==-15 and os.waitstatus_to_exitcode(wait["actual_OS_wait_status"])==-15 and wait["all_owned_processes_and_groups_drained"] is True and wait["operational_failure"]=={"message":"scope stat/membership changed during boundary","type":"ValueError"},"Honest failed wait-15 and owned drain, never success")
    need(wait["root_pid"]==3447891 and wait["child_pid"]==3449313 and wait["owned_lifecycle_evidence"]["owned_identities"]==[{"pid":3449313,"start":482094439}],"Exact failed attempt identities")
    outer=read(r["old_outer_terminal_ref"])
    need(outer.get("schema")=="prospective1611_failed_actual_outer_wait_root_v1" and outer.get("root_authorizer")=="root_agent" and outer.get("actual_direct_outer_tool_wait") is True and outer.get("actual_exec_session_id")==42406 and outer.get("actual_exit_code")==1 and outer.get("actual_tool_chunk_id")=="848541" and outer.get("worker_actual_wait_ref")==OLD_WAIT_REF and outer.get("wrapper_ref")==OLD_WRAPPER_REF and outer.get("wrapper_pid")==wait["root_pid"],"Actual separately consumed failed outer terminal1")
    revoked=read(r["old_future_write_revocation_ref"])
    need(revoked.get("schema")=="prospective1611_failed_attempt_root_future_write_revocation_v1" and revoked.get("root_authorizer")=="root_agent" and revoked.get("active") is True and revoked.get("old_root_request_ref")==OLD_REQUEST_REF and revoked.get("old_storage_ref")==OLD_STORAGE_REF and revoked.get("old_actual_wait_ref")==OLD_WAIT_REF and revoked.get("old_outer_terminal_ref")==r["old_outer_terminal_ref"] and revoked.get("all_old_attempt_launch_resume_write_capabilities_revoked") is True and revoked.get("all_failed_existing_artifacts_frozen") is True and revoked.get("all_failed_files_preserved") is True and revoked.get("scope_credit_claimed") is False and revoked.get("storage_record_or_charge_mutation_authorized") is False,"Truthful old-attempt revocation; no charge or artifact mutation")
    need(revoked.get("all_remote_allocation_pids")==[3447891,3449313] and [x.get("pid") for x in revoked.get("observations",[])]==[3447891,3449313] and all(x.get("absent") is True and x.get("method")=="targeted os.kill(pid,0) ESRCH" for x in revoked["observations"]),"Fresh ROOT-targeted old-identity absence attestation")
    need(all(stamp(outer["observed_utc"])<=stamp(x["checked_utc"])<=stamp(revoked["issued_at_utc"]) for x in revoked["observations"]),"Actual absence observations follow consumed outer terminal")
    need(stamp(wait["observed_utc"])<=stamp(outer["observed_utc"])<=stamp(revoked["issued_at_utc"])<=now and (now-stamp(revoked["issued_at_utc"])).total_seconds()<=900,"Failed outer consumed before fresh permanent revocation")
    expiry=stamp(r["expires_at_utc"])
    need(stamp(r["created_at_utc"])<=now and now+timedelta(seconds=LIMITS["wall_seconds"]+600)<expiry<=stamp(storage["expires_at_utc"]),"Original charge expiry unchanged and covers full6h+drain")
    need(expiry+timedelta(seconds=600)<stamp(read(CERT)["renewed_expires_at_utc"]),"Existing renewal covers authority and drain")
    need(all(r[k]==old[k] for k in ("runtime_binding_ref","runtime_qualification_ref","python_executable_ref","runtime_support_refs")),"Exact original runtime, no implicit substitute")
    runtime=read(r["runtime_binding_ref"]);qual=read(r["runtime_qualification_ref"])
    need(runtime.get("root_qualified") is True and runtime.get("cpu_only") is True and set(runtime["modules"])=={"msprime","numpy","tskit"} and runtime["qualification_receipt_sha256"]==r["runtime_qualification_ref"]["sha256"] and qual["modules"]==runtime["modules"] and qual["generator_source_ref"]=={"path":GEN+"/generation.py","sha256":GEN_SHA} and qual["full_simulation_executed"] is False and qual["GPU_context_created"] is False,"Original saved actual CPU import qualification, no simulation qualification invented")
    need(Path(sys.executable).resolve()==checked(r["python_executable_ref"]) and sha(Path(sys.executable).resolve())==runtime["python_executable_sha256"]==r["python_executable_ref"]["sha256"],"Exact original runtime executable")
    for module in runtime["modules"].values():need(ref(module["file"])["sha256"]==module["sha256"],"Exact qualified module source")
    manifest=read(r["source_manifest_ref"])
    need(manifest["wrapper_ref"]==r["wrapper_ref"] and manifest["caps_ref"]==r["caps_ref"] and manifest["old_storage_ref"]==OLD_STORAGE_REF,"Actual wrapper/caps/old charge lineage")
    for pin in manifest["transitive_library_and_science_refs"]:checked(pin)
    need(ref(GEN+"/generation.py")["sha256"]==GEN_SHA and ref(PLAN)["sha256"]==PLAN_SHA and ref(MAP)["sha256"]==MAP_SHA,"Unchanged full scientific generator/map")
    need(12 in os.sched_getaffinity(0),"Original CPU12 remains in actual allowed affinity")
    capacity=read(r["capacity_verification_ref"])
    need(capacity.get("schema")=="prospective1611_operational_successor_capacity_root_verified_v1" and capacity.get("root_authorizer")=="root_agent" and capacity.get("source_only_estimate") is False and capacity.get("wrapper_ref")==r["wrapper_ref"] and capacity.get("old_storage_ref")==OLD_STORAGE_REF and capacity.get("cpu_id")==12 and capacity.get("single_cpu_available") is True and capacity.get("available_cpu_rss_bytes",0)>=LIMITS["rss_bytes"] and capacity.get("current_CPU_memory_and_scope_observation_actual") is True and capacity.get("current_storage_capacity_observation_actual") is True and capacity.get("available_filesystem_bytes",0)>=16*GIB and stamp(capacity["verified_at_utc"])<=now and (now-stamp(capacity["verified_at_utc"])).total_seconds()<=900,"Fresh ROOT-verified CPU12/memory/retained-scope capacity")
    output=local(OLD_OUTPUT)/("operational_successor_"+SCOPE_ID)
    packet=local(OLD_AUX)/("operational_successor_"+SCOPE_ID)
    paths=dict(output=output,observed=output/"observed",truth=output/"private_truth",receipts=output/"private_receipts",packet=packet,aux=packet/"worker_aux",charged_output=local(OLD_OUTPUT),charged_aux=local(OLD_AUX),old_packet=local(OLD_REL+"/admitted_train000_oct16_root_v1"))
    need(not output.exists() and not packet.exists(),"Fixed exclusive successor namespaces; never replay")
    refs=direct_refs(record,r,manifest)
    return r,storage,paths,refs

def execute(record):
    global _PATHS,_RETAINED,_LIVE_IDS
    r,storage,paths,refs=validate_successor(record);_PATHS=paths
    sys.dont_write_bytecode=True;sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"scripts")]
    need("pidfd_backend_v1" not in sys.modules,"Fresh owned pidfd backend")
    backend=load(PIDFD,"pidfd_backend_v1");sys.modules["pidfd_backend_v1"]=backend
    lifecycle=load(LIFECYCLE,"_prospective1611_live_boundary_owned_lifecycle")
    load(RENEWAL,"_prospective1611_live_boundary_renewal").install_renewal_v1(CERT)
    guard=load(GUARD,"_prospective1611_live_boundary_unchanged_guard");guard.configure_v2(REPAIR,policy_ref=POLICY)
    # No reserve/append/close or storage-source substitution. Both roots are quiescent.
    pre=guard.budget_check_v9(storage,boundary=True)
    need(all(x in refs for x in pre["bindings"].values()),"Every actual active guard authority directly bound")
    old_rows,old_packet=measured(paths)
    retained=sum(v["bytes"] for v in old_rows.values())+sum(v["bytes"] for v in old_packet.values())
    forecast=sum(LIMITS[x] for x in ("observed_bytes","truth_bytes","receipt_bytes","auxiliary_bytes","root_packet_bytes"))+192*4096
    capacity=read(r["capacity_verification_ref"])
    need(capacity.get("retained_original_charge_bytes")==retained,"Fresh ROOT retained-byte observation matches actual quiescent census")
    need(retained+forecast<=LIMITS["reservation_bytes"],"Retained files leave room for full unchanged attempt caps; no downscale")
    paths["packet"].mkdir();paths["aux"].mkdir();paths["output"].mkdir()
    resource_ref=save(paths["packet"]/"generator_resource_binding.json",dict(schema="prospective1611_generation_resource_binding_v1",root_approved=True,enforced_by_root_wrapper=True,cpu_threads=1,rss_bytes=LIMITS["rss_bytes"],wall_seconds=LIMITS["wall_seconds"],observed_output_bytes=LIMITS["observed_bytes"],private_truth_bytes=LIMITS["truth_bytes"],private_receipt_bytes=LIMITS["receipt_bytes"],wrapper_source_sha256=ref(SOURCE)["sha256"],admission_receipt_sha256=OLD_ADMISSION_REF["sha256"],operational_successor_authority_ref=record,original_storage_ref=OLD_STORAGE_REF,new_storage_admission_claimed=False))
    generation_request=save(paths["packet"]/"generator_request.json",dict(schema="prospective1611_generation_request_v1",root_approved=True,parent_id=r["parent_id"],science_plan_sha256=PLAN_SHA,generator_source_sha256=GEN_SHA,map_derivation_receipt=str(local(MAP)),map_derivation_receipt_sha256=MAP_SHA,runtime_binding=str(local(r["runtime_binding_ref"]["path"])),runtime_binding_sha256=r["runtime_binding_ref"]["sha256"],resource_binding=str(local(resource_ref["path"])),resource_binding_sha256=resource_ref["sha256"],observed_output_dir=str(paths["observed"]),private_truth_output_dir=str(paths["truth"]),private_receipt_dir=str(paths["receipts"])))
    refs=list({(x["path"],x["sha256"]):x for x in [*refs,resource_ref,generation_request]}.values())
    need(len(refs)<=64 and resource_ref in refs and generation_request in refs,"Actual generated resource/request roles directly bound before child")
    for x in refs:checked(x)
    lineage=save(paths["packet"]/"operational_authority_and_lineage.json",dict(schema="prospective1611_manual_operational_successor_lineage_v1",root_request_ref=record,wrapper_ref=ref(SOURCE),old_storage_ref=OLD_STORAGE_REF,original_charge_bytes=16*GIB,new_storage_admission=False,old_64_consumer_bindings_unchanged=True,direct_operational_refs=refs,old_actual_wait_ref=OLD_WAIT_REF,old_outer_terminal_ref=r["old_outer_terminal_ref"],old_future_write_revocation_ref=r["old_future_write_revocation_ref"],retained_preflight_bytes=retained,full_original_attempt_forecast_bytes=forecast,automatic_retry=False,scientific_change=False))
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1",VECLIB_MAXIMUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",PYTHONPATH=str(ROOT/"src"),TMPDIR=str(paths["aux"]),LC_ALL="C",LANG="C")
    os.sched_setaffinity(0,{12})
    gate_read,gate_write=os.pipe()
    argv=[str(checked(r["python_executable_ref"])),"-B",str(local(CHILD)),"--gate-fd",str(gate_read),"--request",str(local(generation_request["path"])),"--request-sha256",generation_request["sha256"]]
    save(paths["packet"]/"launch.json",dict(argv=argv,CPU_affinity=[12],limits=LIMITS,RLIMIT_FSIZE_bytes=LIMITS["truth_bytes"]-LIMITS["truth_metadata_margin_bytes"],root_request_ref=record,operational_lineage_ref=lineage,storage_ref=OLD_STORAGE_REF,periodic_guard_boundary=False,strict_boundaries="before_gate_and_after_owned_drain"))
    # Baseline excludes only the two new authorized subtrees. Failed files stay frozen.
    rows,packet_rows=measured(paths)
    _RETAINED={k:v["frozen"] for k,v in {**rows,**packet_rows}.items() if not under(k,paths["output"]) and not under(k,paths["packet"])}
    verify_scopes(paths)
    guard.budget_check_v9(storage,boundary=True)  # Strict quiescent pre-gate check.
    need(not lifecycle._children(os.getpid()),"Fresh controller has no unrelated child")
    libc=ctypes.CDLL(None,use_errno=True);need(libc.prctl(36,1,0,0,0)==0,"Enable original owned subreaper")
    cancel={"signal":None}
    def cancellation(signum,frame):cancel["signal"]=signum
    previous={sig:signal.signal(sig,cancellation) for sig in (signal.SIGTERM,signal.SIGINT)}
    proc=None;started=time.monotonic()
    try:
        proc=subprocess.Popen(argv,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True,pass_fds=(gate_read,),preexec_fn=lambda:child_limits(12))
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
    need(observation["all_owned_processes_and_groups_drained"],"No strict boundary until genuine owned drain")
    guard.budget_check_v9(storage,boundary=True)  # Strict post-drain even on failure.
    verify_scopes(paths)
    need(not cancel["signal"] and observation["returncode"]==0 and observation["operational_failure"] is None and observation["actual_direct_worker_wait"],"Failed successor preserved; no automatic retry")
    observed_ref=ref(paths["observed"]/"observed_generation_receipt.json")
    private_ref=ref(paths["receipts"]/"generation_receipt.json")
    observed,private=read(observed_ref),read(private_ref)
    need(observed["complete"] is True and private["complete"] is True and observed["parent_id"]==private["parent_id"]==r["parent_id"] and observed["diploids"]==1611 and observed["haplotypes"]==3222 and private["request"]["sha256"]==generation_request["sha256"] and private["resource_binding_sha256"]==resource_ref["sha256"],"Actual unchanged complete full-parent result")
    completed=save(paths["output"]/"completed.json",dict(schema="prospective1611_operational_successor_completed_v1",status="completed_prospective1611_full_parent_generation_operational_successor",parent_id=r["parent_id"],length_bp=35178458,diploids=1611,haplotypes=3222,root_request_ref=record,wrapper_ref=ref(SOURCE),generation_request_ref=generation_request,actual_wait_ref=wait_ref,observed_generation_receipt_ref=observed_ref,private_generation_receipt_ref=private_ref,original_storage_ref=OLD_STORAGE_REF,operational_lineage_ref=lineage,original_charge_retained=True,new_storage_admission=False,scope_credit_claimed=False,completed_utc=utc(),inference_performed=False))
    verify_scopes(paths);guard.budget_check_v9(storage,boundary=True)
    print(json.dumps(dict(completed_ref=completed,actual_wait_ref=wait_ref,old_charge_unchanged=True,scope_credit_claimed=False)),flush=True)
    return 0

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--execute",action="store_true");parser.add_argument("--request");parser.add_argument("--request-sha256");a=parser.parse_args()
    if not a.execute:
        print(json.dumps(dict(status="source_only_manual_operational_successor",execution=False,reservation_mutation=False)));return 0
    need(Path.cwd()==ROOT and a.request and a.request_sha256,"ROOT-only exact cwd and explicit actual authority")
    return execute(dict(path=a.request,sha256=a.request_sha256))
if __name__=="__main__":raise SystemExit(main())
