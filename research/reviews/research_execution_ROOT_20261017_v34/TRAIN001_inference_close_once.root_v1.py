"""ROOT one-shot TRAIN001 inference ordinary close; inactive by default.

ROOT supplies exact finite authority, genuine consumed observation tool94085
terminal0, and permanent revocation. This helper never claims its own UI terminal.
Use unchanged original Census capture/apply under the existing v3 coordinator.
No retry, scientific read, deletion, quota change or consumer memo.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,copy,hashlib,json,os,re,stat,sys,types

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/research_execution_ROOT_20261017_v34"
SOURCE=HERE+"/TRAIN001_inference_close_once.root_v1.py"
PACKET="research/reviews/frozen_TRAIN001_observe_ready_20261017_v2/ROOT_actual_TRAIN001_root01_oct17_v2"
def R(path,sha):return dict(path=path,sha256=sha)
INPUT=R(PACKET+"/input.json","3f7399e8b26d1c8177b23e1cf6008c77c6e66d82256b54f304fa766312068776")
STORAGE=R(PACKET+"/storage.json","3e6fcd2afd73f3103a26238d78b8f56346daf6423cab10890356bc2c07076fd0")
LAUNCH=R(PACKET+"/worker_aux/launch.json","01800ddce08fd48f0f3d8e1abe45dab2fecefb431ae46faf0b4ed6c65e77fff1")
WAIT=R(PACKET+"/worker_aux/actual_wait.json","cb04e466d1e28fa7fad57d7d0f89b5c4513f741fd4897a71210e1899fdbddca7")
DONE=R("experiments/results/frozen_TreeLoom_full30Mb_TRAIN001_infer_root01_oct17_v2/completed.json","54763f046092b361647722bd09fb5a332484e004108c81fa4ad9713fb235e408")
SUPERVISOR_WAIT=R(HERE+"/TRAIN001_observation_v2_execution_actual_wait.root_v1.json","88af5a2c743e6dbc90a97b431929ae93f53f4e6eed73c414e37e0ded0618e6b8")
GUARD=R("research/reviews/Census_observation_two_scope_retention_source_20261017_v1/census_observation_retention_v1.py","72f6dc51844b2a261b581f5e69bd2a22982e6244c059f24549747a7b851faeeb")
SELECTOR=R("research/reviews/Census_observation_ROOT_activation_20261017_v1/explicit_child_selector.root_active_v1.json","fd05d8934ed29eed39f7e60121a33e48fbeb0106991e76690664ca96355692e4")
POLICY=R("research/reviews/Census_observation_ROOT_activation_20261017_v1/activation_policy.root_active_v1.json","5e82c0895c43b77d7e69d440fa9b09dbe19f7f2565b3ae9f866d0075e6f9fad5")
COORDINATOR=R("research/reviews/frozen_TRAIN002_019_batch_continuation_source_20261017_v3/worker_entry_v1.py","2c4a976949fe147238ba775fa6aee58bbb782a9cef2c76d65945a472eae6be73")
BATCH=R("research/reviews/frozen_TRAIN002_019_batch_continuation_source_20261017_v3/ROOT_batch_admission_root01_oct17/batch_input.json","dfd6ced48bf7ccfe8ebf9d58e3ffb2ab5193e8a586882d873e3bd4eb9279c092")
KEY="H041/TreeLoom-frozen-full30Mb-TRAIN001-infer-root01_oct17_v2"
OWN_SHA="bab450c10a3a4e1aed55eb50ce7046ff7ce7e10e9b5a4c8875822593769ab582"
WORKER=258730
SUPERVISOR=258023
RESERVED=13000000000
ADMIN_CAP=384*1024
MAX_JSON=65536

def need(ok,message):
    if not ok:raise ValueError(message)

def local(value):
    p=Path(value);p=p if p.is_absolute()else ROOT/p
    need(p!=ROOT and p.is_relative_to(ROOT)and ".."not in p.parts and p.resolve()==p,"Canonical contained path")
    return p

def raw(record,limit=262144):
    need(type(record)is dict and set(record)=={"path","sha256"}and type(record["path"])is str
        and type(record["sha256"])is str and re.fullmatch("[0-9a-f]{64}",record["sha256"])is not None,
        "Exact bounded immutable source/metadata reference")
    p=local(record["path"]);before=p.lstat()
    need(stat.S_ISREG(before.st_mode)and before.st_nlink==1 and 0<before.st_size<=limit,"Bounded singly linked original metadata/source")
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        opened=os.fstat(fd)
        identity=lambda s:(s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        need(identity(opened)==identity(before),"Opened exact original metadata inode")
        chunks=[];total=0
        while True:
            data=os.read(fd,min(65536,limit+1-total))
            if not data:break
            chunks.append(data);total+=len(data)
            need(total<=limit,"Bounded source/metadata bytes")
        data=b"".join(chunks)
        need(identity(os.fstat(fd))==identity(before)==identity(p.lstat())and len(data)==before.st_size,
            "Immutable complete source/metadata read")
    finally:os.close(fd)
    need(hashlib.sha256(data).hexdigest()==record["sha256"],"Original immutable SHA")
    return data

def ref(path):
    p=local(path);s=p.lstat()
    need(stat.S_ISREG(s.st_mode)and s.st_nlink==1 and s.st_size<=262144,"Bounded own source/metadata")
    r=dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest());raw(r);return r

def read(record):return json.loads(raw(record,MAX_JSON))

def load(record,name):
    data=raw(record);module=types.ModuleType(name);module.__file__=str(local(record["path"]));module.__package__=""
    sys.modules[name]=module
    exec(compile(data,module.__file__,"exec",dont_inherit=True,optimize=0),module.__dict__)
    return module

def save(directory,name,value):
    data=json.dumps(value,sort_keys=True,indent=2,allow_nan=False).encode()+b"\n"
    need(len(data)<=MAX_JSON,"Each ROOT administrative JSON<=64KiB")
    rows=list(directory.iterdir());total=4096
    need(len(rows)<8,"Bounded flat one-shot administrative records")
    for p in rows:
        s=p.lstat();need(stat.S_ISREG(s.st_mode)and s.st_nlink==1 and p.suffix==".json"and s.st_size<=MAX_JSON,
            "Own flat singly linked JSON only")
        total+=s.st_size+4096
    need(total+len(data)+4096<=ADMIN_CAP and ADMIN_CAP+65536+4096<=524288,
        "Bounded administrative records plus original event within512KiB allowance")
    fd=os.open(directory/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
    with os.fdopen(fd,"wb")as stream:
        need(stream.write(data)==len(data),"Complete exclusive administrative write");stream.flush();os.fsync(stream.fileno())
    fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.fsync(fd)
    finally:os.close(fd)
    return ref(str((directory/name).relative_to(ROOT)))

def current(q):
    start=datetime.fromisoformat(q["created_at_utc"].replace("Z","+00:00"))
    end=datetime.fromisoformat(q["expires_at_utc"].replace("Z","+00:00"))
    need(start.tzinfo is not None and end.tzinfo is not None and start<=datetime.now(timezone.utc)<end
        and 0<(end-start).total_seconds()<=3600,"Finite current<=1h ROOT one-shot authority")
    return end

def metadata(q):
    inp=read(INPUT);storage=read(STORAGE);launch=read(LAUNCH);wait=read(WAIT);done=read(DONE);supervisor=read(SUPERVISOR_WAIT)
    need(inp["schema"]=="frozen_TRAIN001_current_Census_observation_input_v1"
        and inp["parent_id"]==wait["parent_id"]=="american_c3_train_001"
        and inp["stage"]=="infer"and inp["phase"]==wait["phase"]=="infer_train_001"
        and inp["GPU_authorized"]is False and inp["query_truth_access_authorized"]is False
        and inp["TEST_access_authorized"]is False and inp["GNN_optimization_authorized"]is False,
        "Exact original CPU-only TRAIN001 inference, no scientific payload access")
    own=storage["reservation"]
    need(inp["refs"]["storage"]==STORAGE and inp["storage"]==storage
        and own["key"]==KEY and own["reserved_bytes"]==storage["reserved_new_bytes"]==RESERVED
        and own["output"]==str(local(inp["output_directory"]))
        and own["auxiliary_scopes"]==[str(local(inp["worker_auxiliary_scope"]))]
        and len(own["consumer_bindings"])==64 and storage["accounting_mode"]=="Census_observation_two_scope_retention_v1"
        and storage["completed_overlay_policy_ref"]==POLICY and storage["budget_components"]["consumer_adapter"]==GUARD,
        "Exact unchanged original13GB primary+AUX reservation and all64 consumer bindings")
    need(launch["input_ref"]==wait["input_ref"]==supervisor["input_ref"]==INPUT
        and launch["pid"]==wait["pid"]==WORKER and wait["supervisor_pid"]==SUPERVISOR
        and wait["completed_ref"]==DONE and wait["schema"]=="American_C3_full_original_CPU_actual_wait_v1"
        and type(wait["returncode"])is int and wait["returncode"]==0
        and type(wait["actual_OS_wait_status"])is int and wait["actual_OS_wait_status"]==0
        and wait["actual_direct_worker_wait"]is True and wait["actual_direct_launcher_wait"]is True
        and wait["all_owned_processes_and_groups_drained"]is True and wait["cleanup_errors"]==[]
        and wait["operational_failure"]is None
        and wait["known_owned_pids"]==[WORKER]and wait["known_owned_process_groups"]==[WORKER]
        and wait["supervisor_process_group"]==SUPERVISOR and wait["adopted_waits"]==[]
        and wait["group_observations"]==[dict(pgid=WORKER,absent=True)],
        "Genuine exact original workerOS0 and complete owned drain")
    need(launch["ownership"]["signals_use_pidfd_only"]is True
        and launch["ownership"]["no_numeric_group_signals"]is True
        and launch["ownership"]["owned_identities"]==[dict(pid=WORKER,start=493399669)]
        and supervisor["actual_direct_wait"]is True and type(supervisor["actual_returncode"])is int
        and supervisor["actual_returncode"]==0 and supervisor["pid"]==SUPERVISOR,
        "Original launch identity and separately consumed direct supervisor0; no invented wrapper fields")
    need(done["status"]=="completed_current_C3_inference"
        and local(DONE["path"])==local(inp["output_directory"])/"completed.json",
        "Exact original inference completion status; no parent_id invented in completed.json")
    outer=read(q["actual_observation_outer_terminal_ref"])
    need(outer["root_authorizer"]=="root_agent"and outer["actual_terminal_consumed"]is True
        and type(outer["actual_exit_code"])is int and outer["actual_exit_code"]==0
        and type(outer["actual_tool_session_id"])is int and outer["actual_tool_session_id"]==94085
        and outer["input_ref"]==INPUT and outer["completed_ref"]==DONE
        and outer["actual_worker_wait_ref"]==WAIT and outer["actual_supervisor_wait_ref"]==SUPERVISOR_WAIT
        and outer["all_owned_processes_and_groups_drained"]is True,
        "ROOT-supplied genuinely consumed observation94085 outer0, separate from this helper's future terminal")
    rev=read(q["permanent_route_revocation_ref"])
    need(rev["root_authorizer"]=="root_agent"and rev["active"]is True
        and rev["input_ref"]==INPUT and rev["direct_wait_ref"]==WAIT and rev["completed_ref"]==DONE
        and rev["actual_outer_terminal_ref"]==q["actual_observation_outer_terminal_ref"]
        and rev["reservation_key"]==KEY and rev["reservation_sha256"]==OWN_SHA
        and rev["output_directory"]==inp["output_directory"]and rev["worker_auxiliary_scope"]==inp["worker_auxiliary_scope"]
        and rev["all_original_launch_resume_retry_write_routes_permanently_revoked"]is True
        and rev["preserve_all_files_original_clocks_failures_and_reservations"]is True,
        "Separate exact permanent ROOT original-route revocation, preserving files and history")
    stamp=datetime.fromisoformat(rev["revoked_at_utc"].replace("Z","+00:00"))
    completed_at=datetime.fromisoformat(supervisor["utc"].replace("Z","+00:00"))
    need(stamp.tzinfo is not None and completed_at<=stamp<=datetime.now(timezone.utc),"Actual post-completion nonfuture revocation time")
    return inp,storage,launch,wait,done,own

def close_once(authorization_ref):
    need(Path.cwd()==ROOT and sys.dont_write_bytecode and sys.flags.no_user_site and sys.flags.optimize==0
        and not any(n in sys.modules for n in("torch","torch_geometric")),"Fresh ROOT -B -s nonoptimized stdlib mutation interpreter")
    q=read(authorization_ref);current(q)
    need(q["schema"]=="ROOT_TRAIN001_inference_ordinary_close_once_authorization_v1"
        and q["root_authorizer"]=="root_agent"and q["active"]is True and q["once_authorized"]is True
        and q["operation"]=="close_completed"and q["source_ref"]==ref(SOURCE)
        and q["accounting_batch_input_ref"]==BATCH and q["storage_policy_selector_ref"]==SELECTOR
        and q["ROOT_has_serialized_all_mutations_and_resolved_prior_inflight_actions"]is True
        and q["preserve_all_files_and_original_reservations"]is True and q["automatic_retry"]is False,
        "Exact finite ROOT once authority and unchanged original accounting route")
    instance=q["instance"]
    need(type(instance)is str and re.fullmatch("[a-z][a-z0-9_]{0,39}",instance)is not None,"Fresh bounded ROOT instance")
    directory=local(HERE+"/TRAIN001_inference_close_actual_"+instance)
    need(not directory.exists(),"One-shot fresh attempt; uncertainty consumes namespace without replay")
    inp,storage,launch,wait,done,own=metadata(q)
    directory.mkdir(mode=0o700)
    save(directory,"invocation.json",dict(authorization_ref=authorization_ref,source_ref=ref(SOURCE),
        automatic_retry=False,actual_close_tool_terminal_consumed=False))
    entered=False;apply_entered=False
    try:
        coordinator=load(COORDINATOR,"_ROOT_TRAIN001_inference_original_v3_coordinator")
        with coordinator.accounting_transaction(q["accounting_batch_input_ref"]):
            entered=True;current(q)
            g=load(GUARD,"_ROOT_TRAIN001_inference_exact_current_Census")
            need(g.load_selected_guard_v1(SELECTOR)is g and g.MODE=="Census_observation_two_scope_retention_v1"
                and g._POLICY_REF==POLICY and g.MAX_JSON==65536 and g.META_CHARGE==524288,
                "Unchanged selected original Census mutation route")
            need(all(getattr(g.budget.v4,n).__module__==g.budget.v4.__name__ for n in("canonical","file_bytes","mutex")),
                "Unwrapped original ROOT mutation interpreter; never consumer memo")
            need(g.budget.digest(own)==OWN_SHA,"Exact original reservation digest")
            policy=read(POLICY);implementation=policy["implementation_ref"];launcher=policy["launcher_ref"]
            need(storage["budget_components"]["implementation"]==implementation
                and storage["budget_components"]["launcher"]==launcher,"Exact original implementation/launcher pins")
            raw(implementation);raw(launcher)
            pids=[SUPERVISOR,WORKER];observations=[g._absent(pid)for pid in pids];groups=[SUPERVISOR,WORKER];group_rows=[]
            for group in groups:
                try:os.killpg(group,0)
                except ProcessLookupError:group_rows.append(dict(pgid=group,absent=True))
                else:raise ValueError("Original recorded group present/reused/unknown; preserve reservation")
            observation=dict(schema="concurrent_completed_scope_targeted_root_observation_v1",root_authorizer="root_agent",
                key=KEY,reservation_sha256=OWN_SHA,storage_ref=STORAGE,launch_ref=LAUNCH,direct_wait_ref=WAIT,completed_ref=DONE,
                launcher_pid=WORKER,all_root_owned_launcher_supervisor_worker_pids_listed=True,direct_wait_observed=True,
                completed_scope_all_old_original_resume_write_capabilities_revoked=True,preserve_all_files=True,
                external_NFS_writers_outside_observation=True,pids=pids,observations=observations,groups=groups,group_observations=group_rows,
                actual_outer_terminal_ref=q["actual_observation_outer_terminal_ref"],actual_supervisor_wait_ref=SUPERVISOR_WAIT,
                permanent_route_revocation_ref=q["permanent_route_revocation_ref"],original_failures_and_clocks_preserved=True)
            observation_ref=save(directory,"targeted_completion_observation.json",observation)
            request=dict(key=KEY,reservation_sha256=OWN_SHA,storage_ref=STORAGE,storage_locator=[],
                launch_ref=LAUNCH,launch_pid_locator=["pid"],launch_identity=dict(pid=WORKER,input_ref=INPUT),
                direct_wait_ref=WAIT,wait_pid_locator=["pid"],wait_exit_locator=["returncode"],
                wait_identity=dict(pid=WORKER,input_ref=INPUT,returncode=0,actual_OS_wait_status=0,
                    actual_direct_worker_wait=True,all_owned_processes_and_groups_drained=True,completed_ref=DONE),
                completed_ref=DONE,completed_identity=dict(status="completed_current_C3_inference"),observation_ref=observation_ref)
            current(q)
            with g.budget.v4.mutex(ROOT):
                state=g._state_locked(implementation["sha256"],launcher["sha256"],fresh=True)
                need(len(state[3])==47 and state[0][KEY]==own and KEY not in state[1],"Exact original open reservation and all47 current roles")
                need(g.consumer_bindings_v1(own["consumer_bindings"],list(state[3].values()))==own["consumer_bindings"],
                    "Retain all64 original consumer bindings; no omission/rebind/ceiling change")
                current(q)
                authority=g.operation_draft_v1("close_completed",request,expected_before=g._pins(state[2]),bindings=state[3])
                authority.update(active=True,expires_at_utc=q["expires_at_utc"])
                authority_ref=save(directory,"ordinary_close_authority.json",authority)
                before_events=len(state[4]);before_tail=state[5][-1]if state[5]else None;bindings=copy.deepcopy(state[3])
            print(json.dumps(dict(status="ORIGINAL_ORDINARY_CLOSE_APPLY_NEXT",authority_ref=authority_ref,automatic_retry=False)),flush=True)
            current(q);apply_entered=True
            receipt=g.apply_close_completed_v1(authority_ref,expected_source_sha256=implementation["sha256"],
                expected_launcher_sha256=launcher["sha256"])
            receipt_ref=save(directory,"ordinary_close_receipt.json",receipt)
            # Read/parse through the ORIGINAL authenticated guard/v4 API under its
            # original mutex. No hand-parsed filename or completion-only shortcut.
            with g.budget.v4.mutex(ROOT):
                events,hashes=g._events(bindings)
                prefix=receipt["completed_overlay_prefix"];number=prefix["events"]
                need(type(number)is int and number==before_events+1==len(events)
                    and prefix["tail_sha256"]==hashes[number-1],"Receipt exact appended original journal position/hash")
                event=events[number-1]
                need(event==dict(schema="Census_observation_exact_retention_event_v1",sequence=number,
                    previous_sha256=before_tail,bindings=bindings,operation="close_completed",authority=authority_ref,value=receipt["value"]),
                    "Receipt exact original parsed journal event, authority, complete value and previous chain")
                need(receipt["kind"]=="concurrent_completed_scope_close_receipt_v1"and receipt["key"]==KEY
                    and receipt["value"]["request"]==request and receipt["value"]["reservation_sha256"]==OWN_SHA
                    and receipt["value"]["future_writes_and_relaunch_revoked"]is True
                    and receipt["scope_measurements"]==1 and receipt["old_journal_mutated"]is False and receipt["artifacts_deleted"]==0
                    and type(receipt["retained_bytes"])is int and 0<=receipt["retained_bytes"]<=RESERVED
                    and receipt["credit_bytes"]==RESERVED-receipt["retained_bytes"]
                    and receipt["value"]["measurement"]["stat_only"]is True
                    and receipt["value"]["measurement"]["conservative_scope_bytes"]==receipt["retained_bytes"],
                    "Unchanged original single stat measurement, exact retained/credit and preserved files")
                event_ref=dict(path=g.JOURNAL+"/%06d.json"%number,sha256=hashes[number-1])
                need(g._json(event_ref)==event,"Original source-authenticated exact event reference parse")
            verified_ref=save(directory,"journal_event_verified.json",dict(schema="ROOT_TRAIN001_inference_close_original_journal_verification_v1",
                receipt_ref=receipt_ref,authority_ref=authority_ref,journal_event_ref=event_ref,
                completed_overlay_prefix=prefix,source_authenticated_original_event_parse=True,
                complete_receipt_event_value_equal=True,exact_original_reservation_key=KEY,
                ROOT_actual_close_tool_outer_terminal_consumed=False,ROOT_actual_close_tool_outer_exit_code=None))
            result=dict(status="ORDINARY_CLOSE_APPLIED",receipt_ref=receipt_ref,authority_ref=authority_ref,
                journal_event_ref=event_ref,journal_verification_ref=verified_ref,receipt_matches_exact_original_journal_event=True,
                retained_bytes=receipt["retained_bytes"],credit_bytes=receipt["credit_bytes"],projected_bytes=receipt["projected_bytes"],
                parent_id="american_c3_train_001",stage="infer",all_files_preserved=True,automatic_retry=False,
                scientific_payload_opened=False,new_ancestry_score=False,
                ROOT_actual_close_tool_outer_terminal_consumed=False,ROOT_actual_close_tool_outer_exit_code=None,
                ROOT_must_consume_genuine_close_outer0_before_publishing_accepted_capsule=True)
            save(directory,"result.json",result)
            return result,0
    except BaseException as exc:
        timeout=not entered and isinstance(exc,ValueError)and str(exc)=="Finite shared ROOT accounting coordinator wait"
        failure=dict(status="READY_NO_ACCOUNTING_COORDINATOR_TIMEOUT"if timeout else"FAILED_PRESERVE_NO_RETRY",
            error=type(exc).__name__+": "+str(exc)[:2000],coordinator_entered=entered,apply_entered=apply_entered,
            accounting_outcome="uncertain_after_apply_entry"if apply_entered else"no_apply_called",
            consumed_attempt=True,automatic_retry=False,ROOT_actual_close_tool_outer_terminal_consumed=False,
            ROOT_actual_close_tool_outer_exit_code=None)
        save(directory,"failure.json",failure)
        print(json.dumps(failure,sort_keys=True),flush=True)
        if timeout:return failure,2
        raise

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--authorization-ref");args=parser.parse_args(argv)
    if args.authorization_ref is None:
        print(json.dumps(dict(status="SOURCE_ONLY_INACTIVE",operation=False,guard_imported=False,probes=False)));return 0
    result,code=close_once(json.loads(args.authorization_ref));print(json.dumps(result,sort_keys=True),flush=True);return code

if __name__=="__main__":raise SystemExit(main())
