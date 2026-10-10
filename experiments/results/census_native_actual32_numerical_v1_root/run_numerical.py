#!/usr/bin/env python3
import datetime as dt,hashlib,json,os,signal,subprocess,time
from pathlib import Path
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');os.chdir(ROOT)
OUT=ROOT/'experiments/results/census_native_actual32_numerical_v1_root'
admission=OUT/'admission.json'
expected='54e86d59796031cf6e52f2efec436b2d1e65a9f7a00da6ad6e4757a1503e9ad3'
assert hashlib.sha256(admission.read_bytes()).hexdigest()==expected
args=[str(ROOT/'.runtime/env/bin/python'),'-B','-m','arggnn.census_native_qualification_runner_v1','--phase','numerical','--admission',str(admission.relative_to(ROOT)),'--expected-admission-sha256',expected]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
receipt={'args':args,'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'GPU_requested':False,'biological_target_access':False,'model_forward':True,'watchdog_timeout_seconds':81000,'watchdog_supervisor_pid':os.getpid(),'watchdog_timeout_before_admission_expiry':True}
timed_out=False
with (OUT/'worker.log').open('xb') as out:
 child=subprocess.Popen(args,stdout=out,stderr=subprocess.STDOUT,env=env,start_new_session=True)
 receipt['pid']=child.pid;receipt['start_ticks']=int((Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()[19])
 with (OUT/'launch.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
 print(json.dumps({'pid':child.pid,'start_ticks':receipt['start_ticks']}),flush=True)
 try:exit_code=child.wait(timeout=81000)
 except subprocess.TimeoutExpired:
  timed_out=True;os.killpg(child.pid,signal.SIGTERM)
  try:exit_code=child.wait(timeout=30)
  except subprocess.TimeoutExpired:
   os.killpg(child.pid,signal.SIGKILL);exit_code=child.wait(timeout=30)
receipt.update(exit_code=exit_code,watchdog_timeout=timed_out,completed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
with (OUT/'exit.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps({'exit_code':exit_code,'watchdog_timeout':timed_out}),flush=True)
