import pathlib,json,hashlib,os,sys,time,datetime,subprocess,signal
ROOT=pathlib.Path('/workspace-SR003.nfs2/ARG-GNN');BASE=ROOT/'research/reviews/census_certified_forest_reuse_CPU_timing_v2_root';OUT=ROOT/'experiments/results/census_certified_forest_reuse_CPU_timing_v2'
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(name,obj):
 p=BASE/name
 with p.open('x') as f:json.dump(obj,f,indent=2);f.write('\n')
def identity(pid):
 p=pathlib.Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(') ',1)[1].split()
 return {'pid':pid,'start_ticks':int(s[19]),'uid':p.stat().st_uid,'state':s[0]}
save('watchdog_ready.json',{'supervisor_identity':identity(os.getpid()),'started_utc':utc(),'CPU_limit_bytes':64000000000,'output_limit_bytes':1000000000,'child_walltime_limit_seconds':10800,'GPU_used':False,'status':'watchdog_established_waiting_exact_root_launch_packet'})
end=time.monotonic()+120
activation=BASE/'launch_recipe.json'
while not activation.exists():
 if time.monotonic()>end:save('exit.json',{'status':'failed_no_root_launch_packet','no_child_started':True});raise SystemExit(2)
 time.sleep(.2)
recipe=json.loads(activation.read_text())
for r in recipe['bindings']:
 assert hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['sha256']
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTHONPATH=str(ROOT/'src'),CUDA_VISIBLE_DEVICES='')
for key in ['TMPDIR','TMP','TEMP','XDG_CACHE_HOME','XDG_CONFIG_HOME','XDG_DATA_HOME','TORCH_HOME','HF_HOME','TRITON_CACHE_DIR','CUDA_CACHE_PATH','MPLCONFIGDIR']:
 path=BASE/'watchdog_write_scopes'/key.lower();path.mkdir(parents=True,exist_ok=True);env[key]=str(path)
start=time.monotonic();peak=0;failure=None
with (BASE/'child.stdout').open('xb') as stdout,(BASE/'child.stderr').open('xb') as stderr:
 child=subprocess.Popen(recipe['argv'],cwd=ROOT,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
 owned=identity(child.pid);save('child_started.json',{'identity':owned,'started_utc':utc(),'argv':recipe['argv'],'CUDA_VISIBLE_DEVICES':''})
 while child.poll() is None:
  p=pathlib.Path('/proc')/str(child.pid)
  try:rss=next((int(l.split()[1])*1024 for l in (p/'status').read_text().splitlines() if l.startswith('VmRSS:')),0)
  except FileNotFoundError:rss=0
  peak=max(peak,rss)
  used=sum(max(x.stat().st_size,x.stat().st_blocks*512)+8192 for x in OUT.rglob('*') if x.is_file()) if OUT.exists() else 0
  logs=sum((BASE/n).stat().st_size for n in ['child.stdout','child.stderr'])
  if rss>64000000000 or used>1000000000 or logs>10000000 or time.monotonic()-start>10800:
   failure={'reason':'owned_RSS_output_log_or_walltime_bound','RSS_bytes':rss,'output_bytes':used,'log_bytes':logs,'elapsed_seconds':time.monotonic()-start}
   if identity(child.pid)['start_ticks']==owned['start_ticks'] and identity(child.pid)['uid']==owned['uid']:
    fd=os.pidfd_open(child.pid)
    try:signal.pidfd_send_signal(fd,signal.SIGTERM)
    finally:os.close(fd)
   try:child.wait(timeout=10)
   except subprocess.TimeoutExpired:
    if identity(child.pid)['start_ticks']==owned['start_ticks'] and identity(child.pid)['uid']==owned['uid']:
     fd=os.pidfd_open(child.pid)
     try:signal.pidfd_send_signal(fd,signal.SIGKILL)
     finally:os.close(fd)
    child.wait()
   break
  time.sleep(.5)
 rc=child.wait()
save('exit.json',{'status':'completed_CPU_timing_process' if rc==0 and failure is None else 'failed_partial_preserved','returncode':rc,'watchdog_failure':failure,'maximum_observed_child_RSS_bytes':peak,'observed_peak_sampling_interval_seconds':.5,'walltime_seconds':time.monotonic()-start,'completed_utc':utc(),'GPU_used':False,'fit_or_optimizer_or_truth_authorized':False})
