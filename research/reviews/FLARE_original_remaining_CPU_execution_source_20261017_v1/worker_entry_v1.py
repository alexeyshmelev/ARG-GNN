"""Actual original stage entry. Scientific imports follow admitted owned startup gate."""
from pathlib import Path
import argparse,importlib,json,os,resource,sys,time
from research.reviews.FLARE_original_remaining_CPU_execution_source_20261017_v1 import root_contract_v1 as c
class Adapter:
 def __init__(self,g,fast):self.g=g;self.fast=fast
 def __getattr__(self,name):return getattr(self.g,name)
 def budget_check_v9(self,*a,**kw):self.fast();return self.g.budget_check_v9(*a,**kw)
def worker(record,gate_fd):
 s,q,cfg=c.input_spec(record)
 c.require(type(gate_fd)is int and gate_fd>=0,"Actual parent-owned gate")
 try:c.require(os.read(gate_fd,1)==b"G","Actual pidfd ownership registered before scientific work")
 finally:os.close(gate_fd)
 c.require(Path.cwd()==c.ROOT and sys.dont_write_bytecode and sys.flags.no_user_site
  and sys.prefix==str(c.ROOT/".runtime/env")and os.environ.get("CUDA_VISIBLE_DEVICES")=="","Exact existing CPU-only interpreter")
 private=c.local(s["worker_auxiliary_scope"])/"tmp"
 for key in ("TMPDIR","TMP","TEMP"):c.require(os.environ.get(key)==str(private),"Charged private TMP")
 c.require(set(q["cpu_ids"])<=os.sched_getaffinity(0),"Four actual CPU identities");os.sched_setaffinity(0,set(q["cpu_ids"]))
 # Existing framework has32-thread defaults for TreeLoom. Override before any scientific import.
 for key in ("OMP_NUM_THREADS","NUMBA_NUM_THREADS"):os.environ[key]="4"
 for key in ("MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[key]="1"
 g=c.guard(True);c.verify_actual_roles(s,q,g)
 g.validate_storage_v9(s,dict(output_cap_bytes=c.CAPS["reservation_bytes"]),s["storage"],
  source_ref=c.ref(c.DRIVER),config_ref=s["refs"]["config"],runtime_ref=s["runtime_ref"])
 started=time.monotonic()
 def fast():
  c.current(s,150)
  c.require(time.monotonic()-started<c.CAPS["scientific_seconds"],"Original stage scientific deadline")
  c.require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=c.CAPS["rss_bytes"],"Original CPU RSS failstop")
  c.require(os.environ.get("CUDA_VISIBLE_DEVICES")==""and set(os.sched_getaffinity(0))==set(q["cpu_ids"]),"CPU-only exact4CPU affinity")
 # Source closure and ROOT gate precede NumPy, observed arrays, Java and sealer imports.
 from arggnn.american_c3_common_v1 import runtime_report
 c.require(runtime_report()==c.read(c.BASE_RUNTIME),"Exact prequalified package report")
 c.raw(c.WAIT_WRITER)
 context=importlib.import_module("research.reviews.FLARE_consumer_namespace_root_20261015_v1.root_context_980_v1")
 writer=context.WaitWriter(s,Adapter(g,fast))
 module=importlib.import_module(c.MODULES[q["stage"]])
 c.require(c.ref(module.SOURCE)==c.ref(c.PRODUCERS[q["stage"]]),"Exact unchanged original stage callable")
 fast();result=module.execute(cfg,writer);fast()
 return result
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--worker-ref",required=True);p.add_argument("--gate-fd",type=int,required=True);a=p.parse_args(argv)
 print(json.dumps(dict(completion_ref=worker(json.loads(a.worker_ref),a.gate_fd)),sort_keys=True),flush=True);return 0
if __name__=="__main__":raise SystemExit(main())
