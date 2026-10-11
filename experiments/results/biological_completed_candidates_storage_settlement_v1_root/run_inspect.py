#!/usr/bin/env python3
from pathlib import Path
import datetime as dt,json,os,subprocess
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');os.chdir(ROOT)
OUT=ROOT/'experiments/results/biological_completed_candidates_storage_settlement_v1_root'
args=[str(ROOT/'.runtime/env/bin/python'),'-B','research/reviews/biological_completed_candidates_storage_settlement_v1_author/audit_and_settle.py','--inspect','--expected-source-sha256','7ae1e458d15ab390d4e12e274f669344892af86d87ad6865e6dc865d9f068d75','--expected-config-sha256','f8600dd977db80ff2b17796cae54e9ef83a58916303957cab35bb8752abe07e3','--admission','experiments/results/biological_completed_candidates_storage_settlement_v1_root/inspect_admission.json','--expected-admission-sha256','c38a534dfd2fc1f9c87f3735f97adcdaeac7a9c1bf6b81bf6498bb801c8a80fe']
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
record={'args':args,'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'ledger_mutation':False,'scientific_array_decode':False,'GPU':False}
with (OUT/'worker.log').open('xb') as log:
 child=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
 record['pid']=child.pid;record['start_ticks']=int((Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()[19])
 with (OUT/'launch.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
 print(json.dumps({'pid':child.pid,'start_ticks':record['start_ticks']}),flush=True)
 code=child.wait()
record.update(exit_code=code,completed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
with (OUT/'exit.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
print(json.dumps({'exit_code':code}),flush=True)
