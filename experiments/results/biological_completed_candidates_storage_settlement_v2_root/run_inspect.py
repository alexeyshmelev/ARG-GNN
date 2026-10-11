#!/usr/bin/env python3
from pathlib import Path
import datetime as dt,json,os,subprocess
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');os.chdir(ROOT)
OUT=ROOT/'experiments/results/biological_completed_candidates_storage_settlement_v2_root'
args=[str(ROOT/'.runtime/env/bin/python'),'-B','research/reviews/biological_completed_candidates_storage_settlement_v1_author/storage_revision_v2/audit_and_settle.py','--inspect','--expected-source-sha256','0892242146949b46005337cd583f659867c6e108b415d864152c6f76bd8cef40','--expected-config-sha256','bee8263c0fad30d385bb5bf17324ebaac71ea0a4b1367365b6c4cd0f1f3a994d','--admission','experiments/results/biological_completed_candidates_storage_settlement_v2_root/inspect_admission.json','--expected-admission-sha256','ce22c80f6dffeba5cfa693b72eddd07893542b0da4795c2ff33dbc2fe9c80873']
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
