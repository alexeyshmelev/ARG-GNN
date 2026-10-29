"""Focused CPU tests: no live claims, workers, GPU, signals or queue mutation."""
from pathlib import Path
from unittest.mock import patch
import copy,json,tempfile,hashlib
from research.joint_evaluation_20261029.queue import claims,sidecar_scheduler as scheduler
from research.joint_evaluation_20261029.queue.build_proposal import without_time

HERE=Path(__file__).resolve().parent
checks=[]
def check(name,value):
 assert value,name
 checks.append(name)
def rejects(function):
 try:function()
 except ValueError:return True
 return False

def main():
 plan=json.loads((HERE/'sidecar_plan.json').read_text());old=json.loads((HERE/'original_effective_plan.json').read_text())
 by_id={t['id']:t for t in old['tasks']}
 check('Complete DAG validates without execution',scheduler.validate(plan) is plan)
 check('GPU selection excludes live physical GPU3',plan['allowed_sidecar_gpus']==[4,5] and plan['max_jobs_per_gpu']==1)
 for t in plan['tasks']:
  original=without_time(by_id[t['id']]);candidate=without_time(t)
  if t['id'].endswith('_prepare'):
   parent=int(t['id'].split('_')[-2]);original['dependencies']=[] if parent in (3,4) else [f'paper_american_argformer_american_test_{parent-2:03d}_retire']
  check('Scientific and disk/RAM source preservation '+t['id'],original==candidate)
  check('No arbitrary task wall limit '+t['id'],t['max_seconds']==9223372036854775807)
 check('Full-stage caps charged before work',plan['pipeline_storage_floors'][0]['next_parent_transient_bytes']==sum(t['max_output_bytes'] for t in plan['tasks']))
 ledger=json.loads((HERE/'storage_ledger.json').read_text())
 unconfirmed=copy.deepcopy(ledger);unconfirmed['writer_closure_confirmed']=False
 with patch.object(claims.r,'read',return_value=unconfirmed):
  check('Unconfirmed writer closure rejects launch',rejects(lambda:claims.verify_storage(plan)))
 confirmed=copy.deepcopy(ledger);confirmed['writer_closure_confirmed']=True
 with patch.object(claims.r,'read',side_effect=lambda ref:confirmed if ref==plan['storage_ledger_ref'] else {}):
  check('Bounded confirmed ledger accepted',claims.verify_storage(plan)==confirmed)
 for status in ('pending','running'):
  with tempfile.TemporaryDirectory(prefix='claim_fixture_',dir=HERE) as raw:
   root=Path(raw);task=copy.deepcopy(plan['tasks'][0]);task['output_directory']=str(root)
   marker=root/claims.MARKER;marker.write_text(json.dumps(claims.marker_value(plan,task)))
   result=claims.permit_claimed_output(plan,task,{'status':status})
   check('Only pending owned empty claim is launchable '+status,result==(status=='pending'))
   if status=='pending':
    (root/'foreign.txt').write_text('foreign')
    check('Foreign file in claimed root rejected',rejects(lambda:claims.permit_claimed_output(plan,task,{'status':'pending'})))
 with tempfile.TemporaryDirectory(prefix='claim_fixture_',dir=HERE) as raw:
  root=Path(raw);task=copy.deepcopy(plan['tasks'][0]);task['output_directory']=str(root)
  value=claims.marker_value(plan,task);value['owner']='another-owner';(root/claims.MARKER).write_text(json.dumps(value))
  check('Foreign claim owner rejected',rejects(lambda:claims.permit_claimed_output(plan,task,{'status':'pending'})))
 check('All real original output scopes remain absent',all(not claims.r.path(t['output_directory']).exists() for t in plan['tasks']))
 state=claims.verify_original(plan)
 check('Live native worker remains running and untouched',state['tasks'][plan['guard_task_id']]['pid']==plan['guard_pid']==1976765)
 # Empty roots make the legacy no-overwrite guard reject, even though its state
 # remains pending. It does not depend on the contents of the sidecar claim.
 from arggnn.benchmarking.paper_campaign import continuation
 check('Legacy scheduler cannot authorize sidecar output overwrite',continuation.permit_existing_output(plan['tasks'][0],{'status':'pending'}) is False)
 source=(HERE/'sidecar_scheduler.py').read_text();original_source=claims.r.path('src/arggnn/benchmarking/paper_campaign/scheduler.py').read_text()
 for start,end in [('            for task_id, (process, log, start, start_ticks) in list(running.items()):','            for task in plan[\'tasks\']:\n                row = state[\'tasks\'][task[\'id\']]\n                if row[\'status\'] != \'pending\' or task[\'kind\'] != \'external\':'),('                process = subprocess.Popen(command,','            state[\'updated_unix\'] = time.time()')]:
  left=source[source.index(start):source.index(end,source.index(start))]
  right=original_source[original_source.index(start):original_source.index(end,original_source.index(start))]
  check('Genuine Popen/poll/session drain/completion path unchanged '+start.strip()[:25],left==right)
 from research.joint_evaluation_20261029.queue import coordination
 big=next(t for t in plan['tasks'] if t['id'].endswith('_contrastive'))
 small=next(t for t in plan['tasks'] if t['id'].endswith('_mlm'))
 check('24 GiB sidecar stage permits authorized sharing',coordination.gpu_stage_allowed(plan,small))
 check('52 GiB sidecar stage waits across original transition',not coordination.gpu_stage_allowed(plan,big))
 check('Future original and separate queue RAM growth reserved',plan['foreign_ram_reserve_bytes']==960*2**30)
 check('All seven parent preparations are atomically gated',len(plan['claim_task_ids'])==7 and len(plan['tasks'])==35)
 check('ARGMix full output and auxiliary cap reserved in parallel',plan['other_queue_disk_reserved_bytes']==25_769_803_776+276_824_064+2**30)
 from research.joint_evaluation_20261029.queue.future_merge import merge_rows
 fixture_tasks={'adopt':{'dependencies':[]},'next':{'dependencies':['adopt']},'old_failure':{'dependencies':[]},'old_block':{'dependencies':['old_failure']}}
 historical={'tasks':{'adopt':{'status':'blocked','error':'Output already exists, no overwrite'},'next':{'status':'blocked','error':'Required predecessor failed'},'old_failure':{'status':'failed','pid':999,'error':'historical'},'old_block':{'status':'blocked','error':'Required predecessor failed'}}}
 success={'tasks':{'adopt':{'status':'succeeded','pid':1000,'actual_wait':True,'actual_returncode':0,'owned_process_group_absent':True}}}
 rows,prior,reset=merge_rows(historical,success,fixture_tasks)
 check('Future merge keeps failed history unchanged',rows['old_failure']==historical['tasks']['old_failure'] and rows['old_block']==historical['tasks']['old_block'])
 check('Future merge resets only claim-induced pending descendant',reset==['next'] and rows['adopt']['adopted_from_sidecar'])
 check('Future merge preserves all replaced prior rows',set(prior)=={'adopt','next'})
 bad=copy.deepcopy(historical);bad['tasks']['old_failure']['status']='running'
 check('Future merge refuses live original work',rejects(lambda:merge_rows(bad,success,fixture_tasks)))
 bad=copy.deepcopy(success);bad['tasks']['adopt']['actual_wait']=False
 check('Future merge refuses inferred or missing process wait',rejects(lambda:merge_rows(historical,bad,fixture_tasks)))
 bad=copy.deepcopy(historical);bad['tasks']['adopt']['pid']=123
 check('Future merge never replaces an original launched attempt',rejects(lambda:merge_rows(bad,success,fixture_tasks)))
 receipt={'passed':True,'checks':checks,'check_count':len(checks),'GPU_jobs':0,'worker_processes_started':0,'live_claims_created':0,'signals_sent':0,
  'test_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'sidecar_sha256':hashlib.sha256((HERE/'sidecar_scheduler.py').read_bytes()).hexdigest(),'claims_sha256':hashlib.sha256((HERE/'claims.py').read_bytes()).hexdigest()}
 (HERE/'focused_tests.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

if __name__=='__main__':main()
