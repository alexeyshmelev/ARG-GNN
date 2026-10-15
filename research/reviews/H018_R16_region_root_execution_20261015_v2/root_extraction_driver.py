"""Root bounded real-data extraction admission and actual worker wait."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,hashlib,importlib.util,json,os,subprocess,sys
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'src')]
HERE=ROOT/'research/reviews/H018_R16_region_root_execution_20261015_v2'
SOURCE=str(Path(__file__).relative_to(ROOT))
GUARD='research/reviews/H039_failed_scope_capacity_v2/failed_scope_capacity_v2.py'
REPAIR='research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json'
WORKER='src/arggnn/prepare_h018_r16_regional_extraction_v2.py'
DRAFT='research/reviews/H018_R16_regional_extraction_source_v1/admission.draft.json'
CORE='483d78bf2ef3c917d65ecf1dde5a7809bfadc21194c9f9a90394ce5c532fdd5e'
BASE='a8049bc103afb1cbbbce6f8fc3707a21de9613b23034599641b688994fc5caa3'
def ref(p):
 p=Path(p);p=p if p.is_absolute() else ROOT/p
 return dict(path=p.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(r):
 assert ref(r['path'])==r
 return json.loads((ROOT/r['path']).read_text())
def save(name,value):
 p=HERE/name
 with p.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
 return ref(p)
def guard():
 s=importlib.util.spec_from_file_location('root_H018_guard',ROOT/GUARD);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 m.configure_v2(ref(REPAIR));return m
def admit():
 spec=read(ref(DRAFT));spec['source_ref']=ref(WORKER);cfg=read(spec['config_ref']);m=guard()
 spec['consumer_manifest_ref']=save('exact_consumer_closure.json',dict(schema='H018_R16_exact_transitive_consumer_closure_v1',source_ref=spec['source_ref'],config_ref=spec['config_ref'],runtime_ref=spec['runtime_ref'],tool_refs=cfg['tools'],frozen_refs=cfg['frozen_refs']))
 assert not (ROOT/spec['output_directory']).exists()
 with m.budget.v4.mutex(ROOT):_,_,view,bindings,_=m._state_locked(CORE,BASE,fresh=True)
 closure=[spec['source_ref'],spec['config_ref'],spec['runtime_ref'],spec['consumer_manifest_ref'],ref(SOURCE),*bindings.values(),ref(GUARD),ref(REPAIR),ref('research/reviews/H039_missing_marker_repair_v1/marker_repair_v1.py'),ref('research/reviews/H039_failed_scope_capacity_v1/failed_scope_capacity_v1.py'),ref('src/arggnn/project_budget_v9_consumer.py')]
 closure=list({(r['path'],r['sha256']):r for r in closure}.values())
 expiry=(datetime.now(timezone.utc)+timedelta(hours=20)).isoformat()
 auth=dict(kind='H039_failed_scope_capacity_root_admission_v1',root_authorizer='root_agent',active=True,operation='reserve',accounting_mode=m.MODE,bindings=bindings,expected_before=m._pins(view),activation_barrier_ref=ref('research/reviews/H039_failed_scope_capacity_v2/terminal_activation_barrier.json'),expires_at_utc=expiry,all_budget_writers_use_scheduler_mutex=True,all_consumers_and_launchers_use_H039_failed_scope_guard=True,all_old_guard_consumers_and_supervisors_terminal_before_activation=True,historical_records_mutation_authorized=False,global_rebase_authorized=False,scientific_execution_authorized=False,GPU_authorized=False,reservation=dict(key='H018/region_acquisition_20261015',output=str(ROOT/spec['output_directory']),auxiliary_scopes=[],reserved_bytes=200_000_000,consumer_bindings=closure))
 a=save('reserve_authority.json',auth);receipt=m.apply_reserve_v1(a,expected_source_sha256=CORE,expected_launcher_sha256=BASE);save('reserve_receipt.json',receipt)
 storage=dict(kind='project_budget_v9_storage_record',active=True,consumer_uses_v9_guard=True,accounting_mode=m.MODE,job_id=spec['job_id'],output_directory=spec['output_directory'],reserved_new_bytes=200_000_000,reservation=receipt['value'],expires_at_utc=expiry,initial_ledger_pins=m._pins(m.inspect_v1(expected_source_sha256=CORE,expected_launcher_sha256=BASE)),budget_components=dict(implementation=ref(m.budget.SOURCE_REL),launcher=ref(m.budget.LAUNCHER_REL),consumer_adapter=ref(GUARD)))
 sr=save('storage.json',storage);spec.update(execution_authorized=True,expires_at_utc=expiry,storage_ref=sr)
 a=save('admission.json',spec);save('admitted.json',dict(admission_ref=a,driver_ref=ref(SOURCE)))
 print(json.dumps(dict(admitted=True,admission_ref=a)),flush=True)
def execute():
 admitted=read(ref(HERE/'admitted.json'));assert admitted['driver_ref']==ref(SOURCE)
 spec=read(admitted['admission_ref']);storage=read(spec['storage_ref']);m=guard()
 m.budget_check_v9(storage,first=True,boundary=True)
 s=importlib.util.spec_from_file_location('root_H018_acquisition',ROOT/WORKER);w=importlib.util.module_from_spec(s);s.loader.exec_module(w)
 result=w.execute_v1(admitted['admission_ref'],guard=lambda planned:m.budget_check_v9(storage,boundary=True,required_write_bytes=planned))
 print(json.dumps(dict(completed_ref=result)),flush=True)
def launch():
 cmd=[str(ROOT/'.runtime/env/bin/python'),'-B',str(ROOT/SOURCE),'--worker']
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 with (HERE/'launch.log').open('x') as f:
  proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=f)
  save('launch.json',dict(pid=proc.pid,command=cmd,actual_wait_pending=True));print(json.dumps(dict(started=True,pid=proc.pid)),flush=True);rc=proc.wait()
 save('actual_wait.json',dict(pid=proc.pid,returncode=rc,actual_direct_worker_wait=True,finished_utc=datetime.now(timezone.utc).isoformat(),log_ref=ref(HERE/'launch.log')))
 print(json.dumps(dict(returncode=rc,actual_wait=True)),flush=True);return rc
def main():
 p=argparse.ArgumentParser();p.add_argument('--admit',action='store_true');p.add_argument('--launch',action='store_true');p.add_argument('--worker',action='store_true');a=p.parse_args()
 if a.admit:admit();return 0
 if a.launch:return launch()
 if a.worker:execute();return 0
 return 0
if __name__=='__main__':raise SystemExit(main())
