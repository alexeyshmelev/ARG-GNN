from pathlib import Path
from datetime import datetime, timezone, timedelta
import importlib.util, json, os, subprocess, sys, time, traceback
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
sys.path.insert(0,str(ROOT))
from scripts import project_budget_v9 as b
P=ROOT/'research/reviews/mutation_routed_whole_tree_gnn_continue_v1_root'
CORE='483d78bf2ef3c917d65ecf1dde5a7809bfadc21194c9f9a90394ce5c532fdd5e'
BASE='a8049bc103afb1cbbbce6f8fc3707a21de9613b23034599641b688994fc5caa3'
SOURCE='src/arggnn/continue_mutation_routed_whole_tree_gnn_v1.py'
INITIAL_DRIVER='src/arggnn/train_mutation_routed_whole_tree_gnn_v1.py'
EFF='src/arggnn/mutation_routed_observation_efficiency_v1.py'
MODEL='src/arggnn/mutation_routed_whole_tree_gnn_v1.py'
PREPROCESSOR='src/arggnn/mutation_routed_tree_observations_v1.py'
LOADER='src/arggnn/mutation_routed_parent_loader_v1.py'
CONFIG='configs/mutation_routed_whole_tree_gnn_continue_v1.json'
ADAPTER='src/arggnn/project_storage_full_reservation_v1.py'
LAUNCHER='scripts/launch_reserved_gpu_full_storage_v1.py'
HELPER='research/reviews/full_reservation_admission_v1_root/admission.py'
def ref(p):return b.v4.reference(ROOT,p)
def read(p):return json.loads((ROOT/p).read_text())
def save(packet,name,obj):
 p=packet/name
 with p.open('x') as f:
  json.dump(obj,f,indent=2,sort_keys=True)
  f.write(chr(10))
 return ref(p)
def view():
 with b.v4.mutex(ROOT):
  c,bindings=b.reviewed(ROOT,CORE,BASE)
  return b.state(ROOT,c,bindings,verify_closed=False)[4],bindings
def run(arm,gpu):
 review=read(str((P/'source_review.json').relative_to(ROOT)))
 assert review['verdict']=='PASS'
 for r in review['source_refs']:
  assert ref(r['path'])==r, 'Reviewed source changed'
 cfg=read(CONFIG)
 packet=P/(arm+'_GPU'+str(gpu))
 packet.mkdir(exist_ok=False)
 now=datetime.now(timezone.utc)
 expiry=(now+timedelta(hours=23)).isoformat()
 output='experiments/results/mutation_routed_whole_tree_gnn_continue_v1_'+arm+'_seed101_GPU'+str(gpu)
 assert not (ROOT/output).exists()
 job='mutation-routed-whole-tree-gnn-continue-v1-'+arm
 sources={p:ref(p)['sha256'] for p in (SOURCE,INITIAL_DRIVER,MODEL,PREPROCESSOR,LOADER,EFF,ADAPTER,LAUNCHER,'scripts/project_budget_v9.py','scripts/launch_reserved_gpu_v6.py','src/arggnn/project_budget_v9_consumer.py')}
 for r in cfg['helper_refs'].values():sources[r['path']]=r['sha256']
 runtime=dict(schema='H036_continuation_runtime_v1',status='active',job_id=job,device='cuda',physical_gpu=gpu,
  output_directory=output,created_at_utc=now.isoformat(),expires_at_utc=expiry,
  scientific_source=ref(SOURCE),scientific_model=ref(MODEL),observation_loader=ref(LOADER),
  config=ref(CONFIG),source_review=ref(P/'source_review.json'),
  launcher=ref(LAUNCHER),base_budget_launcher=ref('scripts/launch_reserved_gpu_v6.py'),
  maximum_project_physical_GPUs=2,process_title='AI Auto Research Pipeline Test',
  training_dtype='float32',max_cpu_rss_bytes=128000000000,
  max_cuda_reserved_bytes=68719476736,saved20updates_preserved_with_optimizer_state=True)
 rt=save(packet,'runtime.json',runtime)
 authority=read('research/reviews/interleaved_tree_tv_gnn_v1_root/budget_admission.json')
 v,bindings=view()
 authority['expected_before']=b.pins(v)
 consumer=[ref(p) for p in sources]+[ref(CONFIG),rt]
 authority['reservation']=dict(key='H036/continue_'+arm+'_seed101_GPU'+str(gpu),output=str(ROOT/output),
  auxiliary_scopes=[],reserved_bytes=750000000,consumer_bindings=consumer)
 ar=save(packet,'budget_admission.json',authority)
 loader=importlib.util.spec_from_file_location('root_reserve_h036',ROOT/HELPER)
 helper=importlib.util.module_from_spec(loader)
 loader.loader.exec_module(helper)
 result=helper.apply(ROOT,ar,expected_source_sha256=CORE,expected_launcher_sha256=BASE)
 save(packet,'budget_receipt.json',result)
 v,_=view()
 storage=dict(kind='project_budget_v9_storage_record',active=True,consumer_uses_v9_guard=True,
  accounting_mode='full_reserved_no_closed_scope_credit',job_id=job,output_directory=output,
  actual_launcher=ref(LAUNCHER),reserved_new_bytes=750000000,expires_at_utc=expiry,
  reservation=result['value'],initial_ledger_pins=b.pins(v),
  budget_components=dict(implementation=bindings['implementation'],launcher=bindings['launcher'],consumer_adapter=ref(ADAPTER)))
 st=save(packet,'storage.json',storage)
 old_path='research/reviews/mutation_routed_whole_tree_gnn_v1_root/'+arm+'_GPU'+str(gpu)+'/input.json'
 old=read(old_path)
 initial_done_path='experiments/results/mutation_routed_whole_tree_gnn_v1_'+arm+'_seed101_GPU'+str(gpu)+'/completed.json'
 prior=read(initial_done_path)
 spec=dict(schema='mutation_routed_whole_tree_gnn_continue_input_v1',status='active',root_authorizer='root_agent',
  fit_authorized=True,continuation_authorized=True,job_id=job,arm=arm,device='cuda',seed=101,epochs=70,
  created_at_utc=now.isoformat(),expires_at_utc=expiry,output_directory=output,
  max_cpu_rss_bytes=128000000000,max_cuda_reserved_bytes=68719476736,
  refs=dict(config=ref(CONFIG),runtime=rt,storage=st,target_bindings=old['refs']['target_bindings'],
   registration=ref(P/'registration.json'),initial_completed=ref(initial_done_path),initial_input=ref(old_path),resume_checkpoint=prior['models'][arm]['history'][-1]['checkpoint']),
  observed_parents=old['observed_parents'],source_sha256=sources)
 inp=save(packet,'input.json',spec)
 cmd=[str(ROOT/'.runtime/env/bin/python'),'-B',str(ROOT/LAUNCHER),'--gpu',str(gpu),'--job',job,
  '--storage-record',st['path'],'--expected-storage-sha256',st['sha256'],
  '--expected-budget-source-sha256',CORE,'--expected-launcher-sha256',BASE,
  'arggnn.continue_mutation_routed_whole_tree_gnn_v1','--execute','--input',inp['path'],
  '--expected-input-sha256',inp['sha256']]
 env=dict(os.environ)
 env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
 with (packet/'launch.log').open('xb') as log:
  child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 save(packet,'launch.json',dict(arm=arm,physical_gpu=gpu,pid=child.pid,command=cmd,
  launched_utc=now.isoformat(),input_ref=inp,storage_ref=st,exit_status_known=False))
 deadline=time.monotonic()+60
 while not (ROOT/output/'started.json').exists():
  if child.poll() is not None:
   raise ValueError('Launcher terminated before validated startup '+str(child.returncode))
  if time.monotonic()>deadline:
   raise TimeoutError('Observe the same recorded launcher before any further dispatch')
  time.sleep(.25)
 started=read(output+'/started.json')
 save(packet,'validated_startup.json',dict(observed_utc=datetime.now(timezone.utc).isoformat(),
  launcher_pid=child.pid,worker_pid=started['runtime']['pid'],physical_gpu=gpu,
  input_ref=inp,started_ref=ref(output+'/started.json'),physical_GPUs_max=2))
 print(json.dumps(dict(arm=arm,physical_gpu=gpu,launcher_pid=child.pid,
  worker_pid=started['runtime']['pid'],validated_startup=True,
  full_reserved_bytes=result['full_reserved_bytes'])),flush=True)
if __name__=='__main__':
 try:
  for arm,gpu in [('aligned',2),('shuffled',3)]:
   run(arm,gpu)
 except Exception:
  traceback.print_exc(file=sys.stdout)
  raise SystemExit(1)
