"""Prepare a sealed exact-scope parallel queue; never claim or launch tasks."""
from pathlib import Path
from hashlib import sha256
import copy,json,difflib,time

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE=ROOT/'research/joint_evaluation_20261029/queue'

def ref(p):
 p=ROOT/p if not Path(p).is_absolute() else Path(p)
 return {'path':str(p.relative_to(ROOT)),'sha256':sha256(p.read_bytes()).hexdigest()}

def read(reference):
 assert ref(reference['path'])==reference,reference['path']
 return json.loads((ROOT/reference['path']).read_text())

def save(name,value):
 p=HERE/name;p.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');return ref(p)

def without_time(value):
 if isinstance(value,dict):return {k:without_time(v) for k,v in value.items() if k!='max_seconds'}
 if isinstance(value,list):return [without_time(v) for v in value]
 return value

def remove_time(value):
 if isinstance(value,dict):
  for k,v in value.items():
   if k=='max_seconds':value[k]=9223372036854775807
   else:remove_time(v)
 elif isinstance(value,list):
  for v in value:remove_time(v)

def main():
 state_path=ROOT/'.runtime/benchmarking/paper_campaign_resume_20261026_v1/state.json'
 state=json.loads(state_path.read_text());plan=read(state['plan_ref'])
 receipts=[read(x) for x in state['applied_repairs'].values()]
 for receipt in sorted(receipts,key=lambda x:x['prior_plan_revision']):
  request=read(receipt['request_ref']);replacement={x['id']:x for x in request['replacement_tasks']}
  plan['tasks']=[replacement.get(t['id'],t) for t in plan['tasks']]
 save('original_effective_plan.json',plan);save('original_state_snapshot.json',state)
 names=[f'paper_american_argformer_american_test_{i:03d}_{stage}' for i in range(3,10) for stage in ('prepare','mlm','contrastive','predict','retire')]
 old={t['id']:t for t in plan['tasks']};tasks=[copy.deepcopy(old[n]) for n in names]
 delta=[]
 for t in tasks:
  assert state['tasks'][t['id']]['status']=='pending'
  assert not (ROOT/t['output_directory']).exists(),t['output_directory']
  if t['id'].endswith('_prepare'):
   assert len(t['dependencies'])==1 and t['dependencies'][0].endswith('_retire')
   delta.append({'task_id':t['id'],'removed_storage_order_only_dependencies':t['dependencies']})
   parent=int(t['id'].split('_')[-2])
   t['dependencies']=[] if parent in (3,4) else [f'paper_american_argformer_american_test_{parent-2:03d}_retire']
  remove_time(t)
  a=without_time(old[t['id']]);b=without_time(t);a['dependencies']=b['dependencies']
  assert a==b,'Only storage-order dependencies and arbitrary wall limits may change'
 assert len(tasks)==35
 cleanup=read(ref('research/storage_cleanup_20261027/summary.json'))
 base=cleanup['prior_measured_project_allocated_bytes']-cleanup['allocated_bytes_removed']
 assert base==cleanup['prior_measurement_minus_removed_bytes']==543570411520
 # Charge the ENTIRE old controller capacity above its frozen baseline, even
 # though much of that output already existed at the inventory. Never subtract
 # cleanup bytes from that older unrelated frozen baseline.
 allowance=plan['storage_ceiling_bytes']-plan['reserved_existing_project_bytes']
 ledger={'schema':'parallel_queue_storage_v1','writer_closure_confirmed':False,
  'root_action_required':'Confirm all post-inventory project writers are represented; set true only after review. Rebuild plan binding after editing this ledger.',
  'ceiling_bytes':1_200_000_000_000,
  'base_allowances':[
   {'name':'Authenticated inventory minus exact allocated cleanup','reserved_bytes':base},
   {'name':'Entire original campaign post-baseline growth envelope, conservative double count','reserved_bytes':allowance},
   {'name':'Separate full European validation, entire cap','reserved_bytes':160*2**30},
   {'name':'Joint model entire cache/runs allowance','reserved_bytes':16*2**30},
   {'name':'Separate joint evaluation allowance','reserved_bytes':2**30}],
  'evidence_refs':[ref('research/storage_cleanup_20261027/summary.json'),ref('research/storage_cleanup_20261027/inventory.json'),ref('research/storage_cleanup_20261027/phase1_reviewed.completed.json'),ref('research/storage_cleanup_20261027/phase2_reviewed.completed.json'),state['plan_ref']],
  'limitations':['Inventory-minus-removals is not a fresh total. Full writer closure is required.','Original scheduler ceiling and baseline remain unchanged.','New parent caps are charged IN ADDITION to the complete old campaign growth envelope.','Directory/allocation overhead and new audit records use a separate 1 GiB metadata allowance.']}
 ledger_path=HERE/'storage_ledger.json'
 if ledger_path.exists():
  prior=json.loads(ledger_path.read_text())
  assert prior['base_allowances']==ledger['base_allowances'] and prior['evidence_refs']==ledger['evidence_refs']
  ledger=prior
 ledger_ref=save('storage_ledger.json',ledger)
 source=ROOT/'src/arggnn/benchmarking/paper_campaign/scheduler.py';original=source.read_text();new=original
 replacements={
  'from . import runtime as r':'from arggnn.benchmarking.paper_campaign import runtime as r\nfrom research.joint_evaluation_20261029.queue import claims',
  'from . import processes':'from arggnn.benchmarking.paper_campaign import processes\nfrom research.joint_evaluation_20261029.queue import coordination',
  'from . import repairs':'from arggnn.benchmarking.paper_campaign import repairs',
  'from . import continuation':'from arggnn.benchmarking.paper_campaign import continuation',
  "if plan['storage_ceiling_bytes'] > 1_000_000_000_000:":"if plan['storage_ceiling_bytes'] > 1_200_000_000_000:",
  "lock = r.path('.runtime/benchmarking/paper_campaign_global.lock').open('a+')":"lock = (directory / 'sidecar.lock').open('a+')",
  "seeded = continuation.initial_rows(plan)":"claims.claim_all(plan)\n    seeded = None",
  "candidates = [i for i,g in snapshot.items() if g['free_bytes'] >= task['minimum_free_gpu_bytes']":"candidates = [i for i,g in snapshot.items() if i in plan['allowed_sidecar_gpus'] and g['free_bytes'] >= task['minimum_free_gpu_bytes']",
  "if output.exists() and not continuation.permit_existing_output(task, row):":"if output.exists() and not claims.permit_claimed_output(plan, task, row):",
  "                remaining_growth = sum(":"                remaining_growth = plan.get('foreign_ram_reserve_bytes', 0) + sum(",
  "                    candidates = [i for i,g in snapshot.items()":"                    if not coordination.gpu_stage_allowed(plan, task):\n                        row['waiting_reason'] = 'Original parent002 large-GPU transition guard'\n                        continue\n                    candidates = [i for i,g in snapshot.items()"}
 for a,b in replacements.items():
  assert new.count(a)==1,a
  new=new.replace(a,b)
 (HERE/'sidecar_scheduler.py').write_text(new)
 (HERE/'sidecar_scheduler.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),new.splitlines(True),fromfile=str(source.relative_to(ROOT)),tofile='research/joint_evaluation_20261029/queue/sidecar_scheduler.py')))
 pid=state['controller_pid'];guard_id='paper_american_argformer_american_test_002_contrastive';guard=state['tasks'][guard_id]
 assert guard['status']=='running'
 proposal={'schema':'paper_campaign_v1','physical_gpus':[3,4,5],'allowed_sidecar_gpus':[4,5],
  'claim_id':'argformer_exact_parent003_009_20261029_v1','claim_task_ids':[n for n in names if n.endswith('_prepare')],
  'original_state_path':str(state_path.relative_to(ROOT)),
  'original_identity':{'plan_ref':state['plan_ref'],'plan_revision':state['plan_revision'],'pid':pid,'start_ticks':(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[19]},
  'guard_task_id':guard_id,'guard_pid':guard['pid'],'guard_start_ticks':guard['process_start_ticks'],
  'records_directory':str((HERE/'records_v1').relative_to(ROOT)),
  'storage_ceiling_bytes':ledger['ceiling_bytes'],'reserved_existing_project_bytes':sum(x['reserved_bytes'] for x in ledger['base_allowances'])+25_769_803_776+276_824_064+2**30,
  'other_queue_disk_reserved_bytes':25_769_803_776+276_824_064+2**30,
  'foreign_ram_reserve_bytes':960*2**30,
  'storage_ledger_ref':ledger_ref,'controller_metadata_allowance_bytes':2**30,
  'max_parallel_tasks':4,'max_cpu_tasks':2,'max_jobs_per_gpu':1,
  'existing_workers':[],
  'foreign_ram_reservation_description':'512 GiB for original CPU lanes, 128 GiB for original parent002 GPU chain, 256 GiB ARGMix sidecar, 64 GiB other reviewed writers; counted in addition to observed memory use.',
  'pipeline_storage_floors':[{'scope':'experiments/results/paper_american_20261024','task_ids':names,'retirement_ids':[n for n in names if n.endswith('_retire')],'retired_parent_scopes':{},'next_parent_transient_bytes':sum(t['max_output_bytes'] for t in tasks)}],
  'tasks':tasks,'controller_source_refs':[ref(HERE/'sidecar_scheduler.py'),ref(HERE/'claims.py'),ref(HERE/'coordination.py'),ref('src/arggnn/benchmarking/paper_campaign/runtime.py'),ref('src/arggnn/benchmarking/paper_campaign/processes.py')],
  'completion_merge_policy':'Prepare future merge only after original and sidecar own no running children. Preserve original blocked/failed history; adopt genuine successful sidecar rows with actual_wait and exact native results. Never infer a wait from artifact existence.'}
 save('sidecar_plan.json',proposal)
 save('proposal_receipt.json',{'created_unix':time.time(),'original_state_snapshot_ref':ref(HERE/'original_state_snapshot.json'),'effective_plan_ref':ref(HERE/'original_effective_plan.json'),'sidecar_plan_ref':ref(HERE/'sidecar_plan.json'),'changes':delta,'native_call_specs_equal_except_wall_limits':True,'selected_full_disk_caps':sum(t['max_output_bytes'] for t in tasks),'global_committed_bytes':proposal['reserved_existing_project_bytes']+sum(t['max_output_bytes'] for t in tasks)+2**30,'claim_or_launch_performed':False,'guard_worker_untouched':guard['pid'],'source_ref':ref(__file__)})
 print(json.dumps({'prepared':True,'tasks':len(tasks),'committed_bytes':proposal['reserved_existing_project_bytes']+sum(t['max_output_bytes'] for t in tasks)+2**30,'writer_closure_confirmed':ledger['writer_closure_confirmed'],'launched':False}))

if __name__=='__main__':main()
