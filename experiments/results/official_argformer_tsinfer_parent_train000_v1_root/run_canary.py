#!/usr/bin/env python3
import datetime as dt,hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');os.chdir(ROOT)
OUT=ROOT/'experiments/results/official_argformer_tsinfer_parent_train000_v1_root'
admission=OUT/'admission.json'
expected='c7774951df5b500dcacbcefe0f7bdd0191bca40f11b806bda9b93222b7eadb3e'
assert hashlib.sha256(admission.read_bytes()).hexdigest()==expected
log=ROOT/'research/reviews/official_argformer_tsinfer_parent_v1_author/bio_paths_pilot_train_000.launch.log'
args=[str(ROOT/'.runtime/conda/envs/argformer_native_v1/bin/python'),'scripts/prepare_official_argformer_tsinfer_parent_v1.py','--execute','--parent','bio_paths_pilot_train_000','--expected-source-sha256','9ac0ca01b214e86841808fee951fb55e74bf57bac7227656d6ee7410d9c8a8ed','--expected-config-sha256','9e5212c6de07bb6c28e04f644b336faade04f1e06d6d488f5af49739e2529c55','--admission',str(admission.relative_to(ROOT)),'--expected-admission-sha256',expected]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
receipt={'args':args,'log':str(log.relative_to(ROOT)),'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'GPU_requested':False,'scientific_fallback':False}
with log.open('xb') as out:
 child=subprocess.Popen(args,stdout=out,stderr=subprocess.STDOUT,env=env,start_new_session=True)
 receipt['pid']=child.pid
 receipt['start_ticks']=int((Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()[19])
 with (OUT/'launch.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
 print(json.dumps({'pid':child.pid,'start_ticks':receipt['start_ticks'],'log':receipt['log']}),flush=True)
 exit_code=child.wait()
receipt.update(exit_code=exit_code,completed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
with (OUT/'exit.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps({'exit_code':exit_code,'receipt':str((OUT/'exit.json').relative_to(ROOT))}),flush=True)
