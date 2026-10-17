"""ROOT actual one-shot external monitor for the passed FLARE aggregate stages."""
from pathlib import Path
import ctypes,datetime,hashlib,json,os,signal,subprocess,sys,time,traceback
R=Path("/workspace-SR003.nfs2/ARG-GNN");D=R/"research/reviews/research_execution_ROOT_20261017_v34"
def ref(p):
 p=Path(p);return dict(path=str(p.relative_to(R)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def put(p,value):
 with p.open("x") as f:json.dump(value,f,indent=2,sort_keys=True)
 return ref(p)
def read(r):
 p=R/r["path"];assert ref(p)==r;return json.loads(p.read_text())
def main():
 assert Path.cwd()==R and sys.dont_write_bytecode and sys.flags.no_user_site
 stage=sys.argv[1];assert stage in ("freeze","score")
 request_ref=json.loads(sys.argv[2]);request=read(request_ref);master=read(request["master_contract_ref"])
 assert request["stage"]==stage
 deadline=datetime.datetime.fromisoformat(master["expires_at_utc"]).timestamp()
 assert time.time()<deadline
 sys.path[:0]=[str(R/"research/reviews/American_C3_native_MLM_full_source_v5"),str(R/"research/reviews/American_C3_native_MLM_amendment_v12")]
 from owned_lifecycle_v2 import OwnedTree,_stat,signal_identity
 libc=ctypes.CDLL(None,use_errno=True)
 if libc.prctl(36,1,0,0,0)!=0:raise OSError(ctypes.get_errno(),"subreaper")
 env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONPATH=".:src:scripts",PYTHONDONTWRITEBYTECODE="1")
 for n in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):env[n]="1"
 argv=["setsid","--wait",str(R/".runtime/env/bin/python"),"-B","-s","-m","research.reviews.native_benchmark_readiness_20261017_v1.aggregate_cli_v1","--"+stage+"-ref",json.dumps(request_ref,separators=(",",":"))]
 put(D/("FLARE_all5_"+stage+"_exact_once_invocation.root_v1.json"),dict(argv=argv,once_only=True,source_ref=ref(Path(__file__)),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 failure=None;adopted=[];peak=0;tree=None;rc=None
 with (D/("FLARE_all5_"+stage+"_execution.stdout.root_v1.txt")).open("xb") as log:
  p=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=False)
  for _ in range(200):
   row=_stat(p.pid)
   if row is not None and row["group"]==p.pid and row["session"]==p.pid:break
   if p.poll() is not None:break
   time.sleep(.01)
  if p.poll() is not None:
   rc=p.wait()
  else:
   tree=OwnedTree(p.pid)
   put(D/("FLARE_all5_"+stage+"_launch.root_v1.json"),dict(pid=p.pid,start_ticks=tree.known[p.pid]["start"],input_ref=request_ref))
   print(json.dumps(dict(stage=stage,pid=p.pid,monitoring=True)),flush=True)
   try:
    while p.poll() is None:
     tree.refresh();assert len(tree.known)<=64
     rss=0
     for pid in tree.live():
      try:
       for line in Path("/proc/%d/status"%pid).read_text().splitlines():
        if line.startswith("VmRSS:"):rss+=int(line.split()[1])*1024
      except (FileNotFoundError,ProcessLookupError):pass
     peak=max(peak,rss)
     if rss>4*2**30:raise RuntimeError("Actual aggregate RSS exceeds4GiB")
     if time.time()>=deadline:raise RuntimeError("Original finite master expired")
     time.sleep(1)
    rc=p.wait()
   except BaseException as error:
    failure=type(error).__name__+": "+str(error)
    for sig,seconds in [(signal.SIGTERM,5),(signal.SIGKILL,25)]:
     tree.signal_all(sig);until=time.monotonic()+seconds
     while time.monotonic()<until and tree.live():time.sleep(.1)
    rc=p.wait(timeout=2)
   finally:
    until=time.monotonic()+30
    while time.monotonic()<until:
     tree.refresh()
     for pid,row in list(tree.known.items()):
      if pid==p.pid:continue
      try:
       wp,st=os.waitpid(pid,os.WNOHANG)
       if wp:adopted.append(dict(pid=wp,actual_wait_status=st))
      except ChildProcessError:pass
     if not tree.live() and tree.identities_absent():break
     for pid in tree.live():signal_identity(tree.known[pid],signal.SIGKILL)
     time.sleep(.1)
 drained=tree.identities_absent() if tree is not None else _stat(p.pid) is None
 group_absent=False
 try:os.killpg(p.pid,0)
 except ProcessLookupError:group_absent=True
 except PermissionError:pass
 done=put(D/("FLARE_all5_"+stage+"_actual_wait.root_v1.json"),dict(pid=p.pid,actual_direct_wait=True,actual_returncode=rc,all_owned_processes_and_groups_drained=bool(drained and group_absent),group_absent=group_absent,operational_failure=failure,peak_owned_RSS_bytes=peak,adopted_waits=adopted,ownership=tree.evidence() if tree else None,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 print(json.dumps(dict(actual_wait_ref=done,actual_returncode=rc,drained=bool(drained and group_absent),operational_failure=failure)),flush=True)
 return 0 if rc==0 and drained and group_absent and failure is None else 1
if __name__=="__main__":raise SystemExit(main())
