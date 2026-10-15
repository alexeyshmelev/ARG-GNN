"""One persistent spawn pool; context-authenticated ordered bounded tasks."""
from __future__ import annotations
import copy,hashlib,json,multiprocessing as mp,time,traceback
from research.reviews.C3_GNN_prefetch_provider_20261015_v1.bounded_workers_v1 import _start_ticks

def context_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def _child(connection,opener,kwargs):
    reader=None;context=None
    try:
        reader=opener(**kwargs);connection.send(("ready",))
        while True:
            command=connection.recv()
            if command==("stop",):
                connection.send(("stopped",));break
            if command[0]=="bind":
                value=command[1]
                if context is not None:raise ValueError("Release previous parent before binding")
                reader.bind(value);context=copy.deepcopy(value)
                connection.send(("bound",value["context_sha256"]));continue
            if command[0]=="release":
                if context is None or command[1]!=context["context_sha256"]:raise ValueError("Exact active context release")
                reader.release(command[1]);context=None
                connection.send(("released",command[1]));continue
            try:
                tag,serial,index,digest,parent,plan_ref=command
                if tag!="read" or context is None or digest!=context["context_sha256"] or parent!=context["parent_id"] or plan_ref!=context["plan_ref"]:
                    raise ValueError("Exact parent/plan/context/task required")
                value=reader(index,context_sha256=digest,parent_id=parent,plan_ref=plan_ref)
                connection.send(("ok",serial,index,digest,parent,plan_ref,value));del value
            except BaseException as error:
                connection.send(("error",type(error).__name__,str(error)[:2000],traceback.format_exc()[-6000:]));break
    except (EOFError,BrokenPipeError):pass
    except BaseException as error:
        try:connection.send(("startup_or_context_error",type(error).__name__,str(error)[:2000]))
        except (EOFError,BrokenPipeError,OSError):pass
    finally:
        if reader is not None:reader.close()
        connection.close()

class PersistentOrderedWorkersV1:
    def __init__(self,*,workers,opener,opener_kwargs,check,task_timeout_seconds,drain_timeout_seconds):
        if not(type(workers)is int and workers in (2,4) and callable(opener) and callable(check)
            and 0<task_timeout_seconds<=3600 and 0<drain_timeout_seconds<=30):raise ValueError("Explicit2or4-worker bounds")
        self.limit=workers;self.opener=opener;self.kwargs=opener_kwargs;self.check=check
        self.task_timeout=task_timeout_seconds;self.drain_timeout=drain_timeout_seconds
        self.slots=[];self.pending={};self.serial=0;self.context=None;self.context_serial=0
        self.closed=False;self.last_drain=None;self.release_receipts=[]
        self.dispatch_count=self.delivered_count=self.discarded_count=0
    def pids(self):return [s["process"].pid for s in self.slots]
    def live_identities(self):return [(s["process"].pid,s["start_ticks"]) for s in self.slots if s["process"].is_alive()]
    def _receive(self,slot,deadline):
        while time.monotonic()<deadline:
            self.check("persistent_prefetch_wait",self.live_identities())
            if slot["connection"].poll(min(.1,max(0.,deadline-time.monotonic()))):
                try:return slot["connection"].recv()
                except (EOFError,OSError) as error:raise RuntimeError("Persistent CPU transport closed") from error
            if not slot["process"].is_alive():raise RuntimeError("Persistent worker exited before ordered reply")
        raise TimeoutError("Persistent task/context deadline exceeded")
    def start(self):
        if self.closed:raise RuntimeError("No persistent pool replay/fallback")
        if self.slots:return
        ctx=mp.get_context("spawn")
        try:
            for _ in range(self.limit):
                parent,child=ctx.Pipe(duplex=True)
                process=ctx.Process(target=_child,args=(child,self.opener,self.kwargs),daemon=False)
                try:process.start()
                except BaseException:parent.close();child.close();raise
                child.close();self.slots.append(dict(connection=parent,process=process,task=None,start_ticks=None))
                self.slots[-1]["start_ticks"]=_start_ticks(process.pid)
            deadline=time.monotonic()+self.task_timeout
            for slot in self.slots:
                if self._receive(slot,deadline)!=("ready",):raise RuntimeError("Persistent worker startup failed")
        except BaseException:self.close(failed=True);raise
    def bind(self,value):
        if self.context is not None or self.pending:raise RuntimeError("Previous physical parent remains active")
        self.start();self.context_serial+=1
        ctx=copy.deepcopy(value);ctx["context_serial"]=self.context_serial
        ctx["context_sha256"]=context_sha(ctx)
        if not(type(ctx["blocks"])is int and ctx["blocks"]>0):raise ValueError("Complete nonempty parent plan")
        self.context=ctx
        try:
            for slot in self.slots:slot["connection"].send(("bind",ctx))
            deadline=time.monotonic()+self.task_timeout
            for slot in self.slots:
                if self._receive(slot,deadline)!=("bound",ctx["context_sha256"]):raise RuntimeError("Exact parent context acknowledgement required")
        except BaseException:self.close(failed=True);raise
        return copy.deepcopy(ctx)
    def prefetch(self,index,direction,skip=lambda i:False):
        if self.context is None or direction not in (-1,1):raise ValueError("Explicit active physical context/direction")
        for slot in self.slots:
            if slot["task"] is not None:continue
            while 0<=index<self.context["blocks"] and (index in self.pending or skip(index)):index+=direction
            if not 0<=index<self.context["blocks"]:return
            self.check("persistent_prefetch_dispatch",self.live_identities());self.serial+=1
            task=(self.serial,index,time.monotonic()+self.task_timeout)
            c=self.context
            slot["connection"].send(("read",task[0],index,c["context_sha256"],c["parent_id"],c["plan_ref"]))
            slot["task"]=task;self.pending[index]=slot;self.dispatch_count+=1;index+=direction
    def _take(self,index,*,discarded=False,deadline=None):
        slot=self.pending[index];serial,physical,task_deadline=slot["task"];c=self.context
        reply=self._receive(slot,min(task_deadline,deadline) if deadline is not None else task_deadline)
        if reply[:6]!=("ok",serial,physical,c["context_sha256"],c["parent_id"],c["plan_ref"]):
            raise RuntimeError("Persistent worker/context/physical identity failure: "+repr(reply[:6])[:3000])
        del self.pending[index];slot["task"]=None
        if discarded:self.discarded_count+=1
        else:self.delivered_count+=1
        return reply[6]
    def take(self,index,direction,skip=lambda i:False):
        if self.context is None or type(index)is not int or not 0<=index<self.context["blocks"]:raise ValueError("Physical block index")
        try:
            if index not in self.pending:
                for old in list(self.pending):self._take(old,discarded=True)
            self.prefetch(index,direction,lambda i:i!=index and skip(i))
            return self._take(index)
        except BaseException:self.close(failed=True);raise
    def release_parent(self):
        if self.context is None:return None
        c=copy.deepcopy(self.context)
        try:
            deadline=time.monotonic()+self.drain_timeout
            for index in list(self.pending):self._take(index,discarded=True,deadline=deadline)
            for slot in self.slots:slot["connection"].send(("release",c["context_sha256"]))
            for slot in self.slots:
                if self._receive(slot,deadline)!=("released",c["context_sha256"]):raise RuntimeError("Exact state-clear acknowledgement required")
            row=dict(parent_id=c["parent_id"],plan_ref=c["plan_ref"],context_serial=c["context_serial"],
                context_sha256=c["context_sha256"],all_pending_replies_checked=True,all_worker_parent_states_cleared=True,
                worker_identities=[dict(pid=p,start_ticks=s) for p,s in self.live_identities()],
                workers_remain_persistent=True,worker_process_drain=False)
            if len(row["worker_identities"])!=self.limit:raise RuntimeError("Every persistent worker must acknowledge")
            self.release_receipts.append(row);self.context=None;return copy.deepcopy(row)
        except BaseException:self.close(failed=True);raise
    def close(self,*,failed=False):
        if self.closed:return copy.deepcopy(self.last_drain)
        # Normal callers release the active parent first. Failure never retries a task.
        if not failed and self.context is not None:self.release_parent()
        self.closed=True;forced=[];errors=[];deadline=time.monotonic()+self.drain_timeout
        identities=[dict(pid=s["process"].pid,start_ticks=s["start_ticks"]) for s in self.slots]
        if not failed:
            try:
                for slot in self.slots:slot["connection"].send(("stop",))
                for slot in self.slots:
                    if self._receive(slot,deadline)!=("stopped",):raise RuntimeError("Persistent stop acknowledgement missing")
            except BaseException as error:errors.append(type(error).__name__+": "+str(error)[:1000])
        for slot in self.slots:
            p=slot["process"];p.join(timeout=max(0.,deadline-time.monotonic()) if not failed and not errors else 0.)
            if p.is_alive():p.terminate();forced.append(p.pid)
        for slot in self.slots:
            p=slot["process"];p.join(timeout=2.)
            if p.is_alive():p.kill();p.join(timeout=2.)
            slot["connection"].close()
        self.pending.clear()
        self.last_drain=dict(drained=not any(s["process"].is_alive() for s in self.slots),
            failed=bool(failed or errors or forced or any(s["process"].exitcode!=0 for s in self.slots)),
            forced_worker_pids=forced,worker_pids=[x["pid"] for x in identities],worker_identities=identities,
            exitcodes=[s["process"].exitcode for s in self.slots],errors=errors,max_inflight=self.limit,
            dispatch_count=self.dispatch_count,delivered_count=self.delivered_count,discarded_prefetch_count=self.discarded_count)
        if not self.last_drain["drained"]:raise RuntimeError("Persistent workers failed to join")
        for slot in self.slots:slot["process"].close()
        self.slots.clear();self.context=None;return copy.deepcopy(self.last_drain)
