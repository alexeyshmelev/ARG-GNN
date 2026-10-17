"""Gated readonly authentication, then exact original author binary exec; never resumes."""
from pathlib import Path
import argparse,hashlib,json,os,resource,stat,sys,time
from research.reviews.Relate_exactALL_native_only_source_20261017_v1 import root_contract_v1 as c
def payload(record,deadline,expected_bytes=None,verify_sha=True,keep_fd=False):
 """Read only regular canonical no-hardlink payload; stable fd/name identity; finite streaming."""
 c.require(time.monotonic()<deadline,"Finite payload authentication window")
 p=c.local(record["path"]);before=p.lstat()
 c.require(stat.S_ISREG(before.st_mode)and before.st_nlink==1,"Canonical single-link original payload")
 if expected_bytes is not None:c.require(before.st_size==expected_bytes,"Exact authenticated input bytes")
 fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW);h=hashlib.sha256();size=0
 with os.fdopen(os.dup(fd)if keep_fd else fd,"rb")as f:
  opened=os.fstat(f.fileno())
  c.require((opened.st_dev,opened.st_ino)==(before.st_dev,before.st_ino),"Exact readonly opened object")
  while True:
   c.require(time.monotonic()<deadline,"Finite payload authentication; fail without science")
   data=f.read(2**20)
   if not data:break
   h.update(data);size+=len(data)
  after=os.fstat(f.fileno())
 named=p.lstat()
 identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink)
 c.require(identity(before)==identity(opened)==identity(after)==identity(named),"No payload mutation or replacement during authentication")
 c.require((not verify_sha or h.hexdigest()==record["sha256"])and size==before.st_size,"Actual exact readonly payload SHA256/size")
 result=dict(path=record["path"],sha256=h.hexdigest(),bytes=size)
 return(result,fd)if keep_fd else result
def worker(record,gate_fd):
 s,q,cfg=c.input_spec(record)
 c.require(type(gate_fd)is int and gate_fd>=0,"Actual parent owned gate")
 try:c.require(os.read(gate_fd,1)==b"G","Actual pidfd ownership recorded before payload access/native work")
 finally:os.close(gate_fd)
 c.require(Path.cwd()==c.ROOT and sys.dont_write_bytecode and sys.flags.no_user_site
  and Path(sys.executable).resolve()==(c.ROOT/".runtime/env/bin/python").resolve()
  and os.environ.get("CUDA_VISIBLE_DEVICES")=="","Exact original shared CPU interpreter")
 for k in("TMPDIR","TMP","TEMP"):c.require(os.environ.get(k)==str(c.local(cfg["short_tmp_directory"])),"Charged private TMP")
 c.require(set(q["cpu_ids"])<=os.sched_getaffinity(0),"ROOT one actual available CPU")
 os.sched_setaffinity(0,set(q["cpu_ids"]))
 g=c.guard(True);c.verify_actual_roles(s,q,g)
 g.validate_storage_v9(s,dict(output_cap_bytes=c.CAPS["reservation_bytes"]),s["storage"],
  source_ref=c.ref(c.DRIVER),config_ref=s["refs"]["config"],runtime_ref=s["runtime_ref"])
 c.sources();b=c.binding();closure=c.authenticate_closure(q["readonly_input_authentication_ref"])
 deadline=time.monotonic()+240
 actual=[payload(x,deadline,x["bytes"])for x in closure["files"]]
 binary,native_fd=payload(b["binary_ref"],deadline,keep_fd=True)
 c.require(os.execve in os.supports_fd,"Linux exact authenticated binary-fd exec required")
 c.require(not c.local(q["output_directory"]).exists(),"Fresh author cwd only; no old tmp reuse")
 g.budget_check_v9(s["storage"],boundary=True,required_write_bytes=32768)
 c.local(q["output_directory"]).mkdir()
 aux=c.local(s["worker_auxiliary_scope"])
 c.put(aux/"exec_intent.json",dict(schema="Relate_exactALL_native_exec_intent_v1",input_ref=record,
  pid=os.getpid(),argv=b["exact_argv"],cwd=q["output_directory"],actual_inputs=actual,
  actual_binary=binary,CPU_threads=1,cpu_ids=q["cpu_ids"],exec_preserves_PID_and_start_identity=True,
  execution_success_asserted=False,old_files_opened_readonly=True,conversion_or_export_authorized=False))
 resource.setrlimit(resource.RLIMIT_CORE,(0,0))
 resource.setrlimit(resource.RLIMIT_FSIZE,(c.CAPS["native_bytes"],)*2)
 resource.setrlimit(resource.RLIMIT_CPU,(c.CAPS["scientific_seconds"],)*2)
 os.chdir(c.local(q["output_directory"]))
 c.current(s,300)
 # Success has no Python return path: outer direct wait0 is this exact binary after exec.
 os.execve(native_fd,b["exact_argv"],dict(os.environ))
 raise RuntimeError("execve unexpectedly returned")
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--worker-ref");p.add_argument("--gate-fd",type=int);a=p.parse_args(argv)
 if not a.worker_ref:
  print(json.dumps(dict(status="SOURCE_ONLY_INACTIVE",execution=False)));return 0
 worker(json.loads(a.worker_ref),a.gate_fd)
 return 1
if __name__=="__main__":raise SystemExit(main())
