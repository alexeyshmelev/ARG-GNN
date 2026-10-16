"""Actual root controller client with an honest nested exit status."""
from pathlib import Path
import argparse,json,os,subprocess,sys,datetime
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
BASE=Path('research/reviews/C3_module_route_repair_root_20261016_v1')
p=argparse.ArgumentParser()
p.add_argument('operation',choices=['submit','wait','status','shutdown','cancel'])
p.add_argument('--job');p.add_argument('--record-name',required=True)
a=p.parse_args()
assert Path.cwd()==ROOT and all(c.isalnum() or c in '_.-' for c in a.record_name)
summary=json.loads((BASE/'actual_admission_summary_v1.json').read_text())
cap=summary['actual_capability_ref']
cmd=['.runtime/env/bin/python','-B','research/reviews/C3_native_worker_module_route_fix_source_20261016_v3/launch_declared_gpu0_v1.py','--execute','--renewal-certificate','research/reviews/H039_marker_repair_renewal_source_20261016_v1/renewal_certificate.root_active_v1.json','--renewal-sha256','66006202094d7a3ef2b29cbec28eb93f334505b72589f09d86e4b57498162db7','--renewal-source-sha256','491236877e8d71cae8bd7619880858ac501c57dd93e59479ca32dacacab20468','--module','research.reviews.C3_native_worker_module_route_fix_source_20261016_v3.gpu0_declared_pool_v1','--',a.operation,'--execute','--capability',cap['path'],'--sha256',cap['sha256']]
if a.job:cmd+=['--job',a.job]
r=subprocess.run(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True)
record={'schema':'root_actual_routefixed_controller_client_v1','observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'operation':a.operation,'job':a.job,'command':cmd,'actual_client_returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'external_tool_terminal_consumed':False,'controller_actual_tool_session_id':74136}
with (BASE/(a.record_name+'.json')).open('x') as f:json.dump(record,f,indent=2);f.write('\n')
try:
 result=json.loads(r.stdout)
 report={k:result.get(k) for k in ['logical_job','state','pid','leader_start_ticks','exit_code','owned_drain_observed','argv'] if k in result}
except (ValueError,TypeError):report={'stdout_tail':r.stdout[-2000:]}
print(json.dumps({'actual_client_returncode':r.returncode,'result':report,'stderr_tail':r.stderr[-2000:]}),flush=True)
sys.exit(r.returncode)
