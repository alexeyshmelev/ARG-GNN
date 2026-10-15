"""Pure source fixtures. No pool import, CUDA, /proc, PIDFD, guard, socket or job."""
import ast, contextlib, hashlib, json, signal, types
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE=HERE/"gpu45_c3cache_h050_pool_v2.staged.py"
tree=ast.parse(SOURCE.read_text())
controller=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="Shared45Controller")
methods={n.name:n for n in controller.body if isinstance(n,ast.FunctionDef)}
class Clock:
    wall=1000.;mono=0.
    def time(self):return self.wall
    def monotonic(self):return self.mono
clock=Clock()
signals=[];opened=[];closed=[];identities={};denied=set()
def require(ok,message):
    if not ok:raise ValueError(message)
def identity(pid):
    if pid not in identities:raise ProcessLookupError(pid)
    return dict(identities[pid])
def open_pidfd(pid):
    opened.append(pid)
    if pid in denied:raise PermissionError("fixture denied PIDFD")
    if pid not in identities:raise ProcessLookupError(pid)
    return pid+10000
def send_pidfd(fd,sig):
    signals.append(dict(pid=fd-10000,signal=int(sig),mono=clock.mono))
base=types.SimpleNamespace(proc_identity=identity,open_pidfd=open_pidfd,send_pidfd=send_pidfd)
fake_os=types.SimpleNamespace(getuid=lambda:77,close=lambda fd:closed.append(fd))
namespace=dict(time=clock,base=base,os=fake_os,signal=signal,require=require,
    ACTIVE=frozenset({"launching","running","draining"}))
for name in ["_fail_owned_supervision","_service_termination","tick","cancel_declared","_signal_job","_signal_own"]:
    module=ast.Module(body=[methods[name]],type_ignores=[]);ast.fix_missing_locations(module)
    exec(compile(module,"<isolated_"+name+">","exec"),namespace)
def row(pid,start,uid=77):
    return dict(pid=pid,start_ticks=start,uid=uid,state="S",ppid=1,pgid=pid,sid=pid)
def fixture(scan_error=False,persist_error=False):
    signals.clear();opened.clear();closed.clear();denied.clear();identities.clear()
    clock.wall=1000.;clock.mono=0.
    records={"A":[row(1001,11),row(1004,44)],"B":[row(1002,22)]}
    identities.update({r["pid"]:dict(r) for rs in records.values() for r in rs})
    # Same title and UID do not establish ownership.
    identities[1003]=dict(row(1003,33),title="AI Auto Research Pipeline Test")
    jobs={name:dict(state="running",process_identities=rs,pid=rs[0]["pid"],
        leader_start_ticks=rs[0]["start_ticks"],descendant_waits=[])
        for name,rs in records.items()}
    obj=types.SimpleNamespace(lock=contextlib.nullcontext(),lifecycle="running",
        cap=dict(expires_unix=1500.,drain_expires_unix=2000.,termination_grace_seconds=30,
            owned_termination_escalation_authorized=True),jobs=jobs,
        children={name:types.SimpleNamespace(pid=rs[0]["pid"],poll=lambda:None) for name,rs in records.items()},
        processes={r["pid"]:dict(r) for rs in records.values() for r in rs},
        pid_jobs={r["pid"]:name for name,rs in records.items() for r in rs},ambiguous=[],
        logs={},terminate_requested=False,_lease_check=lambda:None,_drain_logs=lambda:None,
        _reap_owned_adopted=lambda:None,_memory=lambda:dict(overrun_jobs=[],owned_compute_jobs=["A","B"]),
        _launch_queue=lambda:None)
    def scan():
        if scan_error:raise RuntimeError("fixture opaque unrelated PID metadata")
    def persist():
        if persist_error:raise RuntimeError("fixture persistence/guard failure")
    obj._process_scan=scan;obj._persist=persist
    for name in ["_fail_owned_supervision","_service_termination","tick","cancel_declared","_signal_job","_signal_own"]:
        setattr(obj,name,types.MethodType(namespace[name],obj))
    return obj
results=[]
def passed(name,condition,detail=None):
    assert condition,(name,detail)
    results.append(dict(case=name,passed=True,detail=detail))
x=fixture();x.jobs["A"]["cancel_requested"]=True;x._service_termination()
passed("per_job_cancel_targets_only_owned_A_and_known_descendant",
    {r["pid"] for r in signals}=={1001,1004} and all(r["signal"]==signal.SIGTERM for r in signals),list(signals))
clock.mono=29.;x._service_termination();passed("no_KILL_before30seconds",len(signals)==2)
clock.mono=30.;x._service_termination();passed("KILL_after30seconds_only_exact_owned",
    [(r["pid"],r["signal"]) for r in signals[-2:]]==[(1001,signal.SIGKILL),(1004,signal.SIGKILL)],list(signals))
x._service_termination();passed("KILL_not_repeated",len(signals)==4)
passed("TERM_or_KILL_does_not_release_children_or_logical_state",
    len(x.children)==2 and all(j["state"]=="running" for j in x.jobs.values()))
x=fixture(scan_error=True);x.jobs["A"]["cancel_requested"]=True;x.tick(False)
passed("opaque_scan_still_services_cancel_and_fail_stop_owned_drain",
    {r["pid"] for r in signals}=={1001,1002,1004} and x.lifecycle=="draining"
    and all(j["state"]=="draining" for j in x.jobs.values()),list(signals))
clock.mono=30.;x.tick(False)
passed("opaque_scan_still_services_bounded_owned_KILL",
    {r["pid"] for r in signals if r["signal"]==signal.SIGKILL}=={1001,1002,1004})
passed("external_same_title_never_opened_or_signalled",1003 not in opened and all(r["pid"]!=1003 for r in signals))
x=fixture(scan_error=True);x.terminate_requested=True;x.tick(False)
passed("explicit_shutdown_terminate_with_opaque_scan",
    {r["pid"] for r in signals}=={1001,1002,1004})
x=fixture(persist_error=True);x.jobs["A"]["cancel_requested"]=True;x.tick(False)
passed("persistence_failure_does_not_suppress_owned_signal",
    {r["pid"] for r in signals}=={1001,1002,1004} and x.lifecycle=="draining")
clock.mono=30.;x.tick(False)
passed("persistence_failure_does_not_suppress_escalation",
    {r["pid"] for r in signals if r["signal"]==signal.SIGKILL}=={1001,1002,1004})
x=fixture();identities[1004]["start_ticks"]=4444;x.jobs["A"]["cancel_requested"]=True;x._service_termination()
passed("reused_historical_PID_not_signalled",all(r["pid"]!=1004 for r in signals)
    and x.jobs["A"]["owned_termination"]["1004:44"]["identity_mismatch_observed"] is True)
x=fixture();identities[1004]["uid"]=88;x.jobs["A"]["cancel_requested"]=True;x._service_termination()
passed("changed_UID_not_signalled",all(r["pid"]!=1004 for r in signals))
x=fixture();denied.add(1001);x.terminate_requested=True;x._service_termination()
passed("one_PIDFD_error_does_not_block_other_owned_targets",
    {r["pid"] for r in signals}=={1002,1004} and "last_signal_error" in x.jobs["A"]["owned_termination"]["1001:11"])
x=fixture();clock.wall=2000.;x.terminate_requested=True;x._service_termination()
passed("no_signals_after_bounded_drain_authority_expiry",signals==[]
    and x.termination_authority_expired is True and len(x.children)==2)
x=fixture(scan_error=True)
# Execute only the real source block immediately AFTER its Popen result, with
# fake already-owned leader data. No Popen call or process is executed.
launch=methods["_launch_queue"]
block=next(n for n in ast.walk(launch) if isinstance(n,ast.Try)
    and any(isinstance(a,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="leader" for t in a.targets) for a in n.body))
first=next(i for i,n in enumerate(block.body) if isinstance(n,ast.Assign)
    and any(isinstance(t,ast.Name) and t.id=="leader" for t in n.targets))
captured=ast.Module(body=block.body[first:],type_ignores=[]);ast.fix_missing_locations(captured)
x.jobs["A"]["process_identities"]=[];x.jobs["A"]["leader_start_ticks"]=None
x.pid_jobs.clear();x.processes.clear()
local=dict(namespace,self=x,child=types.SimpleNamespace(pid=1001),name="A",job=x.jobs["A"])
try:exec(compile(captured,"<exact_post_Popen_capture_block>","exec"),local)
except RuntimeError as error:assert "opaque" in str(error)
else:raise AssertionError("Expected fixture host-scan failure")
passed("direct_leader_captured_before_opaque_host_scan",
    x.jobs["A"]["leader_start_ticks"]==11 and x.pid_jobs[1001]=="A"
    and x.jobs["A"]["process_identities"][0]["uid"]==77)
x.jobs["A"]["cancel_requested"]=True;x._service_termination()
passed("new_leader_can_cancel_after_first_host_scan_failed",
    [(r["pid"],r["signal"]) for r in signals]==[(1001,signal.SIGTERM)])


# Additional pure owned-tree reader and wait fixtures.
reader_class=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="OwnedProcReader")
reader_ns=dict(base=base,os=fake_os,Path=Path,require=require)
mod=ast.Module(body=[reader_class],type_ignores=[]);ast.fix_missing_locations(mod)
exec(compile(mod,"<isolated_OwnedProcReader>","exec"),reader_ns)
OwnedProcReader=reader_ns["OwnedProcReader"]
controller_pid=900
records={900:row(900,1),1001:dict(row(1001,11),ppid=900),1004:dict(row(1004,44),ppid=1001,sid=700,pgid=700),
    2000:dict(row(2000,100),ppid=1,title="AI Auto Research Pipeline Test")}
edges={900:{1001},1001:{1004},1004:set()}
reads=[];child_reads=[]
def own_identity(pid):
    reads.append(pid)
    if pid not in records:raise ProcessLookupError(pid)
    return dict(records[pid])
def own_children(pid):
    child_reads.append(pid)
    if pid==2000:raise PermissionError("Unrelated metadata must never be read")
    return set(edges.get(pid,set()))
reader=OwnedProcReader(identity=own_identity,children=own_children,uid=77)
table=reader(None,900,{1001:dict(start_ticks=11,leader=True)})
passed("owned_reader_discovers_recursive_setsid_descendant",
    table[1004]["controller_descendant"] and set(table)=={900,1001,1004})
passed("foreign_same_UID_title_and_FD_opacity_never_read",
    2000 not in reads and 2000 not in child_reads and reader.uninspected_unrelated==[])
# Child exits and is adopted after its former parent's children read.
edges[1001]=set();edges[900]={1001,1004};records[1004]["ppid"]=900
table=reader(None,900,{1001:dict(start_ticks=11,leader=True),1004:dict(start_ticks=44)})
passed("recorded_setsid_orphan_retained_after_subreaper_adoption",
    table[1004]["controller_descendant"] and table[1004]["ppid"]==900)
records[1005]=dict(row(1005,55),ppid=900,state="Z");edges[900].add(1005)
table=reader(None,900,{1001:dict(start_ticks=11,leader=True)})
passed("unknown_adopted_zombie_retained_for_ambiguity",
    table[1005]["controller_descendant"] and table[1005]["state"]=="Z")
records[1010]=dict(row(1010,999),ppid=1);child_reads.clear()
table=reader(None,900,{1010:dict(start_ticks=10)})
passed("reused_known_PID_not_traversed_or_adopted",
    table[1010]["controller_descendant"] is False and 1010 not in child_reads)
def denied_own(pid):
    if pid==1001:raise PermissionError("own fixture metadata denied")
    return own_identity(pid)
try:OwnedProcReader(identity=denied_own,children=own_children,uid=77)(None,900,{1001:dict(start_ticks=11)})
except RuntimeError as e:passed("unreadable_owned_identity_fails_closed","owned PID" in str(e))
else:raise AssertionError("Owned identity opacity must fail")
def denied_own_children(pid):
    if pid==1001:raise PermissionError("own task children denied")
    return own_children(pid)
try:OwnedProcReader(identity=own_identity,children=denied_own_children,uid=77)(None,900,{1001:dict(start_ticks=11)})
except RuntimeError as e:passed("unreadable_owned_task_children_fails_closed","task children" in str(e))
else:raise AssertionError("Owned task opacity must fail")
saved=records[1001]["uid"];records[1001]["uid"]=88
try:reader(None,900,{1001:dict(start_ticks=11)})
except ValueError as e:passed("owned_UID_change_fails_closed","UID" in str(e))
else:raise AssertionError("Own UID change must fail")
records[1001]["uid"]=saved
# New controller orphan between first and second controller-children observations.
records[1006]=dict(row(1006,66),ppid=900);calls=[0]
def adopting_children(pid):
    if pid==900:
        calls[0]+=1
        return {1001} if calls[0]==1 else {1001,1006}
    return set()
table=OwnedProcReader(identity=own_identity,children=adopting_children,uid=77)(None,900,{1001:dict(start_ticks=11)})
passed("new_subreaped_orphan_forces_merged_stable_pass",
    1006 in table and table[1006]["controller_descendant"] and calls[0]>=4)
def absent_identity(pid):
    if pid==1004:raise FileNotFoundError(pid)
    return own_identity(pid)
edges[900]={1001};edges[1001]={1004}
table=OwnedProcReader(identity=absent_identity,children=own_children,uid=77)(None,900,{1001:dict(start_ticks=11)})
passed("ESRCH_ENOENT_owned_exit_is_handled_without_adoption",1004 not in table)
# A controller task list that keeps an unreadable/disappearing direct child
# cannot be used as a complete tree. ENOENT retries; persistent inconsistency fails.
def unresolved_children(pid):
    return {99999} if pid==900 else set()
try:OwnedProcReader(identity=own_identity,children=unresolved_children,uid=77)(None,900,{})
except RuntimeError as e:passed("unresolved_direct_controller_child_never_certifies_drain","stabilise" in str(e))
else:raise AssertionError("Unresolved own direct child must prevent complete scan")
# Controller _process_scan adds ambiguity for unknown owned zombies/live orphans.
base_scan_ns=dict(os=types.SimpleNamespace(getpid=lambda:900,getuid=lambda:77))
original_base=Path("/workspace-SR003.nfs2/ARG-GNN/src/arggnn/gpu_pool_v1.py")
bc=next(n for n in ast.parse(original_base.read_text()).body if isinstance(n,ast.ClassDef) and n.name=="Controller")
bn=next(n for n in bc.body if isinstance(n,ast.FunctionDef) and n.name=="_process_scan")
mod=ast.Module(body=[bn],type_ignores=[]);ast.fix_missing_locations(mod)
exec(compile(mod,"<isolated_original_base_process_scan>","exec"),base_scan_ns)
x=types.SimpleNamespace(children={"A":types.SimpleNamespace(pid=1001,returncode=None)},
    jobs={"A":dict(leader_start_ticks=11,process_identities=[dict(records[1001])])},lease=None,_owned_resource_check=lambda:None,
    proc_reader=lambda lease,pid,known:dict(table,**{}) )
# The actual table used here is a deterministic fixture, no process inventory.
x.proc_reader=lambda lease,pid,known:{
 900:dict(records[900],controller_descendant=False,lease_fd_present=False),
 1001:dict(records[1001],controller_descendant=True,lease_fd_present=False),
 1004:dict(records[1004],ppid=1001,controller_descendant=True,lease_fd_present=False),
 1005:dict(records[1005],controller_descendant=True,lease_fd_present=False)}
base_scan_ns["_process_scan"](x)
passed("escaped_known_descendant_attributed_by_verified_parent",
    x.pid_jobs.get(1004)=="A")
# Execute exact override body with only its super-call replaced by the original scan.
override=ast.get_source_segment(SOURCE.read_text(),methods["_process_scan"])
override=override.replace("super()._process_scan()","base_scan(self)")
scan_ns=dict(require=require,base_scan=base_scan_ns["_process_scan"])
exec(compile(override,"<isolated_shared_scan_override>","exec"),scan_ns)
scan_ns["_process_scan"](x)
passed("unattributed_owned_zombie_blocks_logical_drain",1005 in x.ambiguous)
# Actual production wait code executed with fake waitpid only. No wait inference.
waits=[]
def fake_waitpid(pid,flags):
    waits.append(pid);return pid,0
wait_os=types.SimpleNamespace(getpid=lambda:900,WNOHANG=1,waitpid=fake_waitpid,waitstatus_to_exitcode=lambda status:0)
reap_ns=dict(os=wait_os,require=require)
mod=ast.Module(body=[methods["_reap_owned_adopted"]],type_ignores=[]);ast.fix_missing_locations(mod)
exec(compile(mod,"<isolated_actual_wait_recording>","exec"),reap_ns)
z=dict(records[1004],ppid=900,state="Z")
x=types.SimpleNamespace(children={"A":types.SimpleNamespace(pid=1001)},processes={1004:z},orphan_waits=[],
    jobs={"A":dict(process_identities=[dict(z)],descendant_waits=[])})
reap_ns["_reap_owned_adopted"](x)
passed("owned_adopted_wait_records_only_actual_mock_waitpid_result",
    waits==[1004] and x.orphan_waits[0]["wait_status_known"] and x.jobs["A"]["descendant_waits"]==x.orphan_waits)
x.processes={1005:dict(records[1005])};waits.clear()
try:reap_ns["_reap_owned_adopted"](x)
except ValueError:passed("unknown_adopted_zombie_never_waited_or_claimed",waits==[])
else:raise AssertionError("Unknown adoption must block")



# Original outer resource limits are enforced from owned rows, never foreign rows.
resource_ns=dict(os=types.SimpleNamespace(getpid=lambda:900),time=clock,require=require,ACTIVE=namespace["ACTIVE"])
mod=ast.Module(body=[methods["_owned_resource_check"]],type_ignores=[]);ast.fix_missing_locations(mod)
exec(compile(mod,"<isolated_owned_resource_check>","exec"),resource_ns)
def resource_fixture():
    clock.wall=1000.;clock.mono=10.
    return types.SimpleNamespace(cap=dict(expires_unix=1500.),jobs={
        "A":dict(state="running",started_monotonic=0.,requested=dict(cpu_rss_cap_bytes=1000,operational_deadline_seconds=86400)),
        "B":dict(state="running",started_monotonic=0.,requested=dict(cpu_rss_cap_bytes=1000,operational_deadline_seconds=86400))},
        processes={900:dict(rss_bytes=100,state="S",controller_descendant=False),
            1001:dict(rss_bytes=400,state="S",controller_descendant=True),
            1002:dict(rss_bytes=200,state="S",controller_descendant=True),
            1005:dict(rss_bytes=50,state="S",controller_descendant=True)},
        pid_jobs={1001:"A",1002:"B"},violations=[])
# Threshold breaches record fail-owned-supervision without throwing away waits.
original_resource_fixture=resource_fixture
def resource_fixture():
    x=original_resource_fixture()
    x._fail_owned_supervision=lambda message:x.violations.append(message)
    return x
x=resource_fixture();resource_ns["_owned_resource_check"](x)
passed("per_job_CPU_RSS_includes_controller_tree_and_ambiguous_owned",
    x.jobs["A"]["observed_owned_RSS_bytes"]==550 and x.jobs["B"]["observed_owned_RSS_bytes"]==350)
x.processes[1001]["rss_bytes"]=900
resource_ns["_owned_resource_check"](x)
passed("aggregate_owned_CPU_RSS_cap_overrun_requests_drain",any("RSS cap" in v for v in x.violations))
x=resource_fixture();clock.mono=86400.
resource_ns["_owned_resource_check"](x)
passed("original_monotonic_86400s_deadline_requests_drain",any("deadline" in v for v in x.violations))
x=resource_fixture();clock.wall=1500.
resource_ns["_owned_resource_check"](x)
passed("earlier_pool_expiry_requests_owned_drain_before_authority_expiry",any("authority expired" in v for v in x.violations))
# Check frozen H050's one expiry variable is clipped for runtime/spec/storage.
adapter_path=HERE/"root_shared_admission_v2.py"
ad=ast.parse(adapter_path.read_text())
fn=next(n for n in ad.body if isinstance(n,ast.FunctionDef) and n.name=="admit_h050")
rep=next(n for n in fn.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="replacements" for t in n.targets))
replacements=ast.literal_eval(rep.value)
h050=Path("/workspace-SR003.nfs2/ARG-GNN/research/reviews/H050_direct_readout_cut_memory_source_20261015_v1/producer_integration_v1/root_bridge_v1/root_driver_v1.py")
htext=h050.read_text();hn=next(n for n in ast.parse(htext).body if isinstance(n,ast.FunctionDef) and n.name=="admit")
adapted=ast.get_source_segment(htext,hn)
for before,after in replacements.items():
    assert adapted.count(before)==1
    adapted=adapted.replace(before,after)
compile(adapted,"<frozen_H050_shared_admission_compile_only>","exec")
passed("H050_runtime_spec_and_storage_share_pool_clipped_expiry",
    "datetime.fromtimestamp(SHARED_EXPIRES_UNIX" in adapted
    and adapted.count("expires_at_utc=expiry")>=3)


# Integrated real tick path: expiry -> exact-owned TERM/KILL -> actual mocked
# adopted wait + direct Popen.poll -> owned drain. No OS process or wait occurs.
x=fixture();del x.jobs["B"];del x.children["B"]
x.jobs["A"].update(logical_job="A",started_monotonic=0.,
    requested=dict(cpu_rss_cap_bytes=1000,operational_deadline_seconds=86400))
phase={"finished":False,"reaped":False};direct_polls=[];adopted_polls=[]
x.cap["expires_unix"]=1500.;clock.wall=1500.;clock.mono=0.
x.orphan_waits=[];x.logs={"A":types.SimpleNamespace(close=lambda:None)}
child=x.children["A"];child.stdout=None
def direct_poll():
    direct_polls.append(phase["finished"])
    return 0 if phase["finished"] else None
child.poll=direct_poll
x._owned_resource_check=types.MethodType(resource_ns["_owned_resource_check"],x)
def integrated_scan():
    x.processes={900:dict(row(900,1),rss_bytes=100,controller_descendant=False)}
    x.pid_jobs={};x.ambiguous=[]
    if not phase["finished"]:
        x.processes.update({1001:dict(row(1001,11),rss_bytes=400,controller_descendant=True),
            1004:dict(row(1004,44),rss_bytes=50,controller_descendant=True)})
        x.pid_jobs={1001:"A",1004:"A"}
    elif not phase["reaped"]:
        x.processes[1004]=dict(row(1004,44),ppid=900,state="Z",rss_bytes=0,controller_descendant=True)
    x._owned_resource_check()
def integrated_waitpid(pid,flags):
    adopted_polls.append(pid);phase["reaped"]=True;return pid,0
integrated_os=types.SimpleNamespace(getpid=lambda:900,WNOHANG=1,
    waitpid=integrated_waitpid,waitstatus_to_exitcode=lambda status:0)
integrated_ns=dict(os=integrated_os,require=require)
mod=ast.Module(body=[methods["_reap_owned_adopted"]],type_ignores=[]);ast.fix_missing_locations(mod)
exec(compile(mod,"<integrated_mocked_adopted_wait>","exec"),integrated_ns)
x._process_scan=integrated_scan;x._reap_owned_adopted=types.MethodType(integrated_ns["_reap_owned_adopted"],x)
x._memory=lambda:dict(overrun_jobs=[],owned_compute_jobs=[] if phase["finished"] else ["A"])
x.tick(False)
passed("expired_job_requests_owned_TERM_without_blocking_direct_poll",
    any(r["signal"]==signal.SIGTERM for r in signals) and direct_polls==[False])
clock.mono=30.;x.tick(False)
passed("expired_job_escalates_owned_KILL_while_wait_service_continues",
    {r["pid"] for r in signals if r["signal"]==signal.SIGKILL}=={1001,1004} and len(direct_polls)==2)
phase["finished"]=True;x.tick(False)
passed("expired_job_reaches_actual_mocked_adopted_wait_and_direct_wait",
    adopted_polls==[1004] and direct_polls[-1] is True
    and x.jobs["A"]["leader_wait_status_known"] is True and len(x.jobs["A"]["descendant_waits"])==1)
passed("expiry_drain_closes_only_after_actual_waits_and_no_owned_compute",
    x.lifecycle=="closed" and not x.children and x.jobs["A"]["state"]=="failed"
    and x.jobs["A"]["owned_drain_observed"] is True)

print(json.dumps(dict(schema="GPU45_C3cache_H050_pool_successor_pure_fixtures_v1",
    source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),cases=results,cases_passed=len(results),
    actual_pool_imports_guard_calls_PIDFDs_processes_jobs_GPU_queries_signals_or_Git=0),indent=2,sort_keys=True))
