"""Inactive two-slot facade over the existing Shared0 lifecycle. Default inert."""
from pathlib import Path
import argparse,ast,fcntl,hashlib,json,os,resource,signal,stat,sys,time,types
from . import common_v1 as h
from . import pool_contract_v1 as c
ROOT=h.ROOT;SOURCE=c.POOL;MODULE=SOURCE[:-3].replace("/",".")
require=h.require

def _open_old_inode_v1(app):
    """Strict existing-inode open: a missing old lock is never recreated."""
    p=h.local(".runtime/locks/gpu_0.lock");expected=app._old_handover["old_physical_lease"]
    fd=os.open(p,os.O_RDWR|os.O_NOFOLLOW)
    try:
        st=os.fstat(fd);named=p.lstat()
        require(stat.S_ISREG(st.st_mode)and st.st_nlink==1 and(st.st_dev,st.st_ino)==(named.st_dev,named.st_ino)==(expected["device"],expected["inode"]),"Exact existing physical lease inode before new acquisition")
        return os.fdopen(fd,"r+")
    except BaseException:os.close(fd);raise


def _verify_new_exclusive_v1(app,lease):
    """After our real exclusive flock; no guard/accounting call under scheduler lock."""
    expected=app._old_handover["old_physical_lease"]
    p=h.local(".runtime/locks/gpu_0.lock");opened=os.fstat(lease.fileno());named=p.lstat()
    require(stat.S_ISREG(opened.st_mode)and opened.st_nlink==1
        and(opened.st_dev,opened.st_ino)==(named.st_dev,named.st_ino)==(expected["device"],expected["inode"]),"New exclusive ownership uses exact old canonical inode")
    old=os.pread(lease.fileno(),65536,0)
    require(len(old)==opened.st_size and hashlib.sha256(old).hexdigest()==expected["sha256"],"Old lease generation preserved before our new write")
    with p.open("r+")as probe:
        try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:pass
        else:
            fcntl.flock(probe,fcntl.LOCK_UN);raise RuntimeError("New owner exclusion proof failed")
    app.new_exclusive_acquisition=dict(schema="GPU0_actual_new_exclusive_acquisition_v1",pid=os.getpid(),device=opened.st_dev,inode=opened.st_ino,old_lease_sha256=expected["sha256"],writable_independent_exclusion_observed=True,acquired_new_exclusive_flock=True,inherited_or_adopted_ownership=False)


def build_controller_class_v1():
    """Future explicit operation only; imports original source without modifying files."""
    h.source_refs_v1();reviewed=h.load(c.SHARED,"_postdrain_original_Shared0")
    legacy=reviewed.legacy;base=reviewed.base
    # Adapt only finite roster ceilings in the private original base class.
    # All UID/start/session, ambiguity, discovery and assignment bodies remain.
    base_source=h.raw(h.BASE_POOL).decode();base_tree=ast.parse(base_source)
    base_class=next(n for n in base_tree.body if isinstance(n,ast.ClassDef)and n.name=="Controller")
    process_fn=next(n for n in base_class.body if isinstance(n,ast.FunctionDef)and n.name=="_process_scan")
    process_body=ast.get_source_segment(base_source,process_fn)
    require(process_body.count("len(known) > 512")==1 and process_body.count("> 8192:")==1,"Exact two original finite roster ceilings")
    process_body=process_body.replace("len(known) > 512","len(known) > (4097 if job[\"requested\"][\"worker_kind\"]==\"native\" else 512)")
    process_body=process_body.replace("> 8192:","> sum(4097 if j[\"worker_kind\"]==\"native\" else 512 for j in self.cap[\"jobs\"].values()):")
    process_ns=dict(vars(base));exec(compile(process_body,"<private_original_scan_finite_native_roster>","exec"),process_ns)
    base.Controller._process_scan=process_ns["_process_scan"]
    legacy.load_cap=lambda r,allow_drain=False:c.validate_cap_v1(r,allow_drain=allow_drain)
    legacy.current_guard=c.guard_v1;legacy.guarded_check=c.guarded_check_v1
    class PostdrainController(reviewed.Shared0Controller):
        def __init__(self,cap_ref):
            super().__init__(cap_ref)
            self.modules=frozenset({MODULE});self._old_handover=None;self.new_exclusive_acquisition=None
        def _check(self,required=0):
            require(time.time()<self.cap["drain_expires_unix"],"Finite controller drain lifetime")
            require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=2*2**30,"Controller own2GiB RSS cap")
            h.source_refs_v1();self.storage_account=c.guarded_check_v1(self.guard,self.admin,required)
            return self.storage_account
        def status(self):
            value=super().status();value.update(schema="GPU0_postdrain_two_slot_registry_v1",new_exclusive_acquisition=self.new_exclusive_acquisition,C3_compute_slots=0)
            return value
        def start(self):
            H058_job=next(n for n,j in self.cap["jobs"].items()if j["worker_kind"]=="H058")
            proof,identities=c.validate_handover_v1(self.cap["handover_ref"],H058_job,fresh=True)
            for saved in identities:
                try:now=base.proc_identity(saved["pid"])
                except(FileNotFoundError,ProcessLookupError):continue
                require(not(now["uid"]==saved["uid"]and now["start_ticks"]==saved["start_ticks"]and now["state"]not in("Z","X")),"Old owner/borrower still live; never replace it")
            self._old_handover=proof
            return _new_start(self)
        def _memory(self,incoming=0):
            value=super()._memory(incoming)
            # Fail active owned work without throwing across inherited poll/wait.
            # External occupancy may persist after an owned child has exited.
            if not incoming and any(j["state"]in legacy.ACTIVE for j in self.jobs.values())and value["blocked_by"]:
                self._fail_owned_supervision("Periodic active pool capacity failed: "+",".join(value["blocked_by"]))
            return value
        def _owned_resource_check(self):
            super()._owned_resource_check()
            if time.time()>=self.cap["expires_unix"]:self._fail_owned_supervision("New finite pool work authority expired")
            for j in self.jobs.values():
                if j["state"]in legacy.ACTIVE and time.time()>=j["requested"]["absolute_expires_unix"]:
                    self._fail_owned_supervision("Original per-job absolute authority expired")
        def submit_postdrain(self,submission_ref):
            with self.lock:
                require(self.lifecycle=="running","Pool accepting only fresh declared work")
                self.tick(False);require(not self.safety_error,"Owned supervision error blocks submission")
                v,s,q,cfg,storage=c.validate_submission_v1(self.cap_ref,submission_ref);name=v["job_id"]
                require(name not in self.jobs,"Single-use actual job; no replay")
                j=dict(self.cap["jobs"][name],submission_ref=submission_ref,input_ref=v["input_ref"],storage_ref=s["refs"]["storage"],env={},log_limit_bytes=128*2**10,absolute_expires_unix=min(c.expires(q),c.expires(v)),module=MODULE)
                c.guarded_check_v1(self.guard,storage)
                memory=self._memory(j["peak_mib"])
                require(not memory["blocked_by"],"Actual admission capacity rejected: "+",".join(memory["blocked_by"]))
                self.jobs[name]=dict(logical_job=name,requested=j,state="queued",submitted_unix=time.time(),admission=memory,pid=None,leader_start_ticks=None,process_identities=[],exit_code=None,log_bytes=0,log_dropped_bytes=0,descendant_waits=[])
                self._persist();self._launch_queue();self._persist();return dict(self.jobs[name])
        def dispatch(self,value):
            if isinstance(value,dict)and set(value)=={"v","op","submission_ref"}and value["v"]==1 and value["op"]=="submit_postdrain":
                return self.submit_postdrain(value["submission_ref"])
            require(not(isinstance(value,dict)and value.get("op")=="submit"),"Old generic/controller submissions revoked")
            return super().dispatch(value)
    # Original start, with only obsolete nonzero-GPU probes removed and an exact
    # old-generation/new-owner exclusion proof added before metadata overwrite.
    text=h.raw(c.DOWN).decode();node=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef)and n.name=="Shared45Controller")
    fn=next(n for n in node.body if isinstance(n,ast.FunctionDef)and n.name=="start");body=ast.get_source_segment(text,fn)
    obsolete="for idx in (2,3):\n                        with base.lock_file(ROOT,f\"gpu_{idx}.lock\") as h:\n                            try:fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)\n                            except BlockingIOError:raise RuntimeError(\"Forbidden legacy GPU2/3 lease remains active\")\n                    "
    require(body.count(obsolete)==1 and body.count("self.lease=lease")==1,"Exact original startup adaptation sites")
    require(body.count("lease=base.lock_file(ROOT,f\"gpu_{self.gpu}.lock\")")==1,"One original physical-open site")
    body=body.replace("lease=base.lock_file(ROOT,f\"gpu_{self.gpu}.lock\")","lease=_open_old_inode_v1(self)")
    body=body.replace(obsolete,"").replace("self.lease=lease","self.lease=lease\n                    _verify_new_exclusive_v1(self,lease)")
    ns=dict(vars(legacy),base=base,_verify_new_exclusive_v1=_verify_new_exclusive_v1,_open_old_inode_v1=_open_old_inode_v1);exec(compile(body,"<postdrain_exact_original_start>","exec"),ns)
    _new_start=ns["start"]
    # Keep original launch/wait/log/descendant/termination bodies; replace only
    # fixed old runtime bootstrap argv by this exact finite worker and recheck the
    # actual admitted input/expiry immediately before Popen.
    text=h.raw(c.OLD).decode();node=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef)and n.name=="Shared45Controller")
    fn=next(n for n in node.body if isinstance(n,ast.FunctionDef)and n.name=="_launch_queue");body=ast.get_source_segment(text,fn)
    lines=body.splitlines();matches=[i for i,line in enumerate(lines)if line.strip().startswith("argv=[")]
    require(len(matches)==1,"One exact old worker argv")
    lines[matches[0]]="            argv=[self.python,\"-B\",\"-s\",\"-m\",MODULE,\"worker\",\"--execute\",\"--capability\",self.cap_ref[\"path\"],\"--sha256\",self.cap_ref[\"sha256\"],\"--job\",name,\"--submission-ref\",json.dumps(j[\"submission_ref\"],separators=(\",\",\":\"))]"
    body="\n".join(lines)
    queue_anchor="            if job[\"state\"]!=\"queued\":continue"
    require(body.count(queue_anchor)==1,"One original queue state boundary")
    body=body.replace(queue_anchor,queue_anchor+"\n            if any(other[\"state\"]in ACTIVE and other[\"requested\"][\"worker_kind\"]==job[\"requested\"][\"worker_kind\"]for other in self.jobs.values()):continue")
    anchor="                child=subprocess.Popen("
    require(body.count(anchor)==1,"One exact Popen boundary")
    body=body.replace(anchor,"                c.validate_submission_v1(self.cap_ref,j[\"submission_ref\"])\n                require(time.time()<j[\"absolute_expires_unix\"],\"Per-job expiry before Popen\")\n"+anchor)
    ns=dict(vars(legacy),MODULE=MODULE,c=c,load_cap=legacy.load_cap,guarded_check=c.guarded_check_v1,POLICY=h.POLICY)
    exec(compile(body,"<postdrain_original_queue_exact_worker_route>","exec"),ns)
    PostdrainController._launch_queue=ns["_launch_queue"]
    # Preserve every original tick operation and exception boundary. A failure
    # already requested by _memory/_process_scan must remain visible, while the
    # original unconditional owned polls, adopted waits and drain checks proceed.
    fn=next(n for n in node.body if isinstance(n,ast.FunctionDef)and n.name=="tick")
    body=ast.get_source_segment(text,fn)
    require(body.count("                self.safety_error=None")==1,"One exact inherited transient-error reset")
    body=body.replace("                self.safety_error=None","                if self.lifecycle==\"running\":self.safety_error=None")
    ns=dict(vars(legacy),time=time)
    exec(compile(body,"<postdrain_original_tick_preserve_failure_drain>","exec"),ns)
    PostdrainController.tick=ns["tick"]
    return PostdrainController,reviewed


def worker_v1(cap_ref,submission_ref,job):
    v,s,q,cfg,storage=c.validate_submission_v1(cap_ref,submission_ref)
    require(job==s["job_id"] and os.environ.get("ARGGNN_LOGICAL_JOB_NAME")==job,"Exact pool launch identity")
    # Original H058 supervisor creates its own gated session and logical lock.
    # It truthfully borrows this new controller inode and retains all science.
    cap=c.validate_cap_v1(cap_ref)
    if cap["jobs"][job]["worker_kind"]=="native":
        if cap["jobs"][job].get("native_phase","mlm")!="mlm":
            from . import downstream_contract_v1 as downstream
            downstream.set_amendment_v1(v["input_ref"])
            from . import downstream_execution_v1
            return downstream_execution_v1.execute_v1(v["input_ref"])["returncode"]
        from . import native_root_driver_v1
        return native_root_driver_v1.execute_v1(v["input_ref"])["returncode"]
    from . import root_driver_v1
    return root_driver_v1.execute_v1(v["input_ref"])["returncode"]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("command",nargs="?",choices=("start","submit","status","shutdown","worker"));p.add_argument("--execute",action="store_true")
    p.add_argument("--capability");p.add_argument("--sha256");p.add_argument("--submission-ref");p.add_argument("--job")
    a=p.parse_args(argv)
    if not a.execute:print(json.dumps(dict(status="SOURCE_ONLY_INACTIVE",operations=False)));return 0
    require(Path.cwd()==ROOT and sys.dont_write_bytecode and sys.flags.no_user_site,"Exact fresh ROOT -B -s entry")
    require(a.command and a.capability and a.sha256,"Exact finite active capability")
    record=dict(path=a.capability,sha256=a.sha256);cap=c.validate_cap_v1(record,allow_drain=a.command in("status","shutdown"))
    if a.command=="worker":return worker_v1(record,json.loads(a.submission_ref),a.job)
    cls,reviewed=build_controller_class_v1()
    if a.command=="start":
        app=cls(record).start()
        def stop(sig,frame):app.shutdown_requested=True
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        print(json.dumps(app.status(),sort_keys=True),flush=True);app.run();return 0
    socket_path=reviewed.socket_name(cap)
    value=dict(v=1,op="status")if a.command=="status"else dict(v=1,op="shutdown",terminate=False)
    if a.command=="submit":value=dict(v=1,op="submit_postdrain",submission_ref=json.loads(a.submission_ref))
    answer=reviewed.controller_request(socket_path,value);print(json.dumps(answer,sort_keys=True),flush=True);return 0
if __name__=="__main__":raise SystemExit(main())
