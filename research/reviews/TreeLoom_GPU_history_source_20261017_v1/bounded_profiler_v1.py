"""Compact owned main-thread stack/timing sampler; stdlib only, no Torch trace."""
from contextlib import contextmanager
from collections import Counter
from pathlib import Path
import json,os,resource,sys,threading,time

class BoundedProfilerV1:
    def __init__(self,*,interval_seconds=5):
        if interval_seconds!=5:raise ValueError("Frozen five-second sample period")
        self.interval=interval_seconds;self.main=threading.get_ident()
        self.started=time.monotonic();self.lock=threading.RLock();self.stop=threading.Event()
        self.stages={};self.stack=[];self.stacks=Counter();self.recent=[]
        self.samples=0;self.dropped_stacks=0;self.error=None;self.folder=None
        self.thread=threading.Thread(target=self._loop,name="bounded_main_stack_sampler",daemon=True)
        self.thread.start()
    def configure(self,folder,max_bytes):
        with self.lock:self.max_bytes=max_bytes;self.folder=Path(folder)
    def mark(self,label):
        with self.lock:
            self.last_mark=str(label)[:96]
    @contextmanager
    def stage(self,label):
        label=str(label)[:96];start=time.monotonic()
        with self.lock:
            if label not in self.stages:
                if len(self.stages)>=48:raise ValueError("At most48 fixed stage names")
                self.stages[label]=dict(calls=0,seconds=0.,max_seconds=0.)
            self.stack.append(label)
            self.recent=(self.recent+[dict(stage=label,entry_seconds=start-self.started)])[-16:]
        try:yield
        finally:
            elapsed=time.monotonic()-start
            with self.lock:
                row=self.stages[label];row["calls"]+=1;row["seconds"]+=elapsed
                row["max_seconds"]=max(row["max_seconds"],elapsed);self.stack.pop()
                self.recent=(self.recent+[dict(stage=label,exit_seconds=time.monotonic()-self.started,
                                            elapsed_seconds=elapsed)])[-16:]
    def _sample(self):
        frame=sys._current_frames().get(self.main);rows=[]
        while frame is not None and len(rows)<8:
            rows.append("%s:%d:%s"%(frame.f_code.co_filename[-120:],frame.f_lineno,frame.f_code.co_name[:64]))
            frame=frame.f_back
        key=tuple(rows)
        with self.lock:
            self.samples+=1
            if key in self.stacks or len(self.stacks)<24:self.stacks[key]+=1
            else:self.dropped_stacks+=1
    def snapshot(self):
        with self.lock:
            return dict(schema="TreeLoom_bounded_stage_mainstack_profile_v1",
                elapsed_seconds=time.monotonic()-self.started,interval_seconds=self.interval,
                samples=self.samples,stack_signatures_dropped=self.dropped_stacks,
                stage_timings={k:dict(v)for k,v in self.stages.items()},
                active_stages=list(self.stack),last_resource_mark=getattr(self,"last_mark",None),
                recent_stage_entry_exit=list(self.recent),
                sampled_main_stacks=[dict(count=n,frames=list(k))for k,n in self.stacks.most_common()],
                peak_cpu_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                sampler_error=self.error)
    def flush(self):
        with self.lock:
            if self.folder is None:return
            data=json.dumps(self.snapshot(),sort_keys=True,separators=(",",":"),allow_nan=False).encode()+b"\n"
            if len(data)>49152:raise ValueError("Compact profile at most48KiB")
            rows=list(self.folder.iterdir())
            if len(rows)>=64 or any(not p.is_file()or p.is_symlink()for p in rows):
                raise ValueError("Profile writes only its admitted flat scope")
            # Reserve two complete profile records before writing the replacement.
            charged=4096+sum(p.stat().st_size+4096 for p in rows)
            if charged+2*(65536+4096)>self.max_bytes:raise ValueError("Prospective profile output cap")
            tmp=self.folder/"profile.next.json";dst=self.folder/"profile.latest.json"
            fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
            try:
                with os.fdopen(fd,"wb")as stream:
                    if stream.write(data)!=len(data):raise OSError("Incomplete compact profile")
                    stream.flush();os.fsync(stream.fileno())
                os.replace(tmp,dst)
            finally:
                if tmp.exists():tmp.unlink()
    def _loop(self):
        while not self.stop.wait(self.interval):
            try:self._sample();self.flush()
            except BaseException as error:
                with self.lock:self.error=type(error).__name__+": "+str(error)[:300]
                return
    def close(self):
        self.stop.set();self.thread.join(timeout=2.)
        self._sample()
        try:self.flush()
        except BaseException as error:
            self.error=type(error).__name__+": "+str(error)[:300]
        if self.thread.is_alive():raise RuntimeError("Bounded sampler did not stop")
