"""Exact FLARE scientific call after owned bootstrap and genuine parent scientific release."""
from pathlib import Path
import argparse,importlib,json,os,resource,signal,sys,time
from datetime import datetime,timezone
from . import root_driver_v1 as d

def compact_status(fd,value):
 data=(json.dumps(value,sort_keys=True)+'\n').encode();d.need(len(data)<=4096,'Bounded owned worker status')
 d.need(os.write(fd,data)==len(data),'Whole owned status-pipe frame')
def grant_from_parent(fd):
 data=bytearray()
 while b'\n'not in data:
  chunk=os.read(fd,4096);d.need(chunk,'Parent scientific gate did not authorize work');data.extend(chunk);d.need(len(data)<=4096,'Bounded scientific release')
 d.need(data.endswith(b'\n')and data.count(b'\n')==1,'One exact scientific release frame');return json.loads(data)
def worker(record,gate_fd,ready_fd):
 d.activate();c=d.c;s,q,cfg=c.input_spec(record)
 c.require(type(gate_fd)is int and gate_fd>=0 and type(ready_fd)is int and ready_fd>=0 and ready_fd!=gate_fd,'Two distinct actual owned startup pipes')
 try:
  c.require(os.read(gate_fd,1)==b'G','Original owned pidfd bootstrap gate')
  c.require(Path.cwd()==c.ROOT and sys.dont_write_bytecode and sys.flags.no_user_site and sys.prefix==str(c.ROOT/'.runtime/env')and os.environ.get('CUDA_VISIBLE_DEVICES')=='','Exact qualified CPU interpreter')
  private=c.local(s['worker_auxiliary_scope'])/'tmp'
  for key in('TMPDIR','TMP','TEMP'):c.require(os.environ.get(key)==str(private),'Charged private TMP')
  c.require(set(q['cpu_ids'])<=os.sched_getaffinity(0),'Four actual CPU identities');os.sched_setaffinity(0,set(q['cpu_ids']))
  for key in('OMP_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='4'
  for key in('MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
  g=c.guard(True);view=d.initial_storage_transaction(s,q,g)
  # Initial package metadata and unchanged writer-class import belong to setup.
  from arggnn.american_c3_common_v1 import runtime_report
  c.require(runtime_report()==c.read(c.BASE_RUNTIME),'Exact prequalified package report');c.raw(c.WAIT_WRITER)
  context=importlib.import_module('research.reviews.FLARE_consumer_namespace_root_20261015_v1.root_context_980_v1')
  c.current(s,c.CAPS['wall_seconds']+600)
  ready=dict(schema='FLARE_worker_bootstrap_ready_v2',input_ref=record,pid=os.getpid(),monotonic=time.monotonic(),utc=datetime.now(timezone.utc).isoformat(),worker_source_ref=c.ref(c.ENTRY),initial_metadata_storage_and_all_roles_validated=True,actual_role_count=len(g.consumer_bindings_v1(list(view['bindings'].values()),c.direct_roles(s['root_stage_request_ref'],q,s['refs']['config'],s['transitive_source_closure_ref']))))
  compact_status(ready_fd,ready);grant=grant_from_parent(gate_fd)
  c.require(grant.get('schema')=='FLARE_scientific_release_v2'and grant.get('input_ref')==record and grant.get('worker_pid')==os.getpid(),'Actual parent releases this exact ready worker')
  origin=grant['phase_origin_monotonic'];deadline=grant['scientific_deadline_monotonic'];now=time.monotonic()
  c.require(type(origin)in(int,float)and type(deadline)in(int,float)and deadline-origin==c.CAPS['scientific_seconds']and origin<=now<deadline,'Original full scientific cap begins only at parent ready boundary')
  entry=dict(schema='FLARE_worker_scientific_entry_v2',input_ref=record,pid=os.getpid(),monotonic=now,utc=datetime.now(timezone.utc).isoformat(),phase_origin_monotonic=origin,scientific_deadline_monotonic=deadline)
  compact_status(ready_fd,entry)
 finally:
  os.close(gate_fd);os.close(ready_fd)
 def scientific_timeout(number,frame):raise d.PhaseStop('WORKER_SCIENTIFIC_CUTOFF')
 previous=signal.getsignal(signal.SIGALRM);c.require(signal.getitimer(signal.ITIMER_REAL)==(0.0,0.0),'Fresh worker scientific timer')
 signal.signal(signal.SIGALRM,scientific_timeout);signal.setitimer(signal.ITIMER_REAL,max(0.000001,deadline-time.monotonic()))
 count=[0]
 def progress(event,**metadata):
  if count[0]<512:
   count[0]+=1;print(json.dumps(dict(FLARE_worker_progress=dict(event=event,parent_id=q['parent_id'],stage=q['stage'],monotonic=time.monotonic(),utc=datetime.now(timezone.utc).isoformat(),**metadata)),sort_keys=True),flush=True)
 def fast():
  c.current(s,150);c.require(time.monotonic()<deadline,'Original stage scientific deadline')
  c.require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=c.CAPS['rss_bytes'],'Original CPU RSS failstop')
  c.require(os.environ.get('CUDA_VISIBLE_DEVICES')==''and set(os.sched_getaffinity(0))==set(q['cpu_ids']),'CPU-only exact4CPU affinity')
 class Adapter:
  def __getattr__(self,name):return getattr(g,name)
  def budget_check_v9(self,*args,**kwargs):
   fast();progress('original_guard_boundary_begin')
   wall_start=time.monotonic();cpu_start=time.process_time()
   try:value=g.budget_check_v9(*args,**kwargs)
   except BaseException:
    progress('original_guard_boundary_failed',boundary_wall_seconds=time.monotonic()-wall_start,boundary_process_cpu_seconds=time.process_time()-cpu_start)
    raise
   wall_seconds=time.monotonic()-wall_start;process_cpu_seconds=time.process_time()-cpu_start
   progress('original_guard_boundary_end',boundary_wall_seconds=wall_seconds,boundary_process_cpu_seconds=process_cpu_seconds)
   fast();return value
 try:
  progress('scientific_entry',phase_origin_monotonic=origin,scientific_deadline_monotonic=deadline)
  class ProgressWriter(context.WaitWriter):
   def prepare(self,name,budget_bytes):
    progress('writer_prepare_begin',product=name);p=super().prepare(name,budget_bytes);progress('writer_prepare_end',product=name);return p
   def finish(self,p,planned):
    progress('writer_finish_begin',product=p.name);r=super().finish(p,planned);progress('writer_finish_end',product=p.name);return r
  writer=ProgressWriter(s,Adapter());module=importlib.import_module(c.MODULES[q['stage']])
  c.require(c.ref(module.SOURCE)==c.ref(c.PRODUCERS[q['stage']]),'Exact unchanged original stage callable')
  fast();result=module.execute(cfg,writer);fast();progress('scientific_callable_completed');return result
 except BaseException as error:
  progress('worker_failed',exception_type=type(error).__name__,reason=str(error)[:1000]);raise
 finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--worker-ref');p.add_argument('--gate-fd',type=int);p.add_argument('--ready-fd',type=int);a=p.parse_args(argv)
 if not a.worker_ref:print(json.dumps(dict(status='SOURCE_ONLY_INACTIVE',guard_imported=False,models_imported=False)));return 0
 d.need(a.gate_fd is not None and a.ready_fd is not None,'Actual two-phase owned worker gate')
 result=worker(json.loads(a.worker_ref),a.gate_fd,a.ready_fd);print(json.dumps(dict(completion_ref=result),sort_keys=True),flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
