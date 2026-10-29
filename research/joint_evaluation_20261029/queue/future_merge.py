"""Prepare verified completion adoption only after both queues have drained.

No controller is signalled. No live state or historical row is overwritten.
"""
import argparse,copy,json,time
from pathlib import Path
from arggnn.benchmarking.paper_campaign import runtime as r,processes

HERE=Path(__file__).resolve().parent

def need(ok,message):
 if not ok:raise ValueError(message)

def merge_rows(original,sidecar,tasks):
 need(not any(x['status']=='running' for x in original['tasks'].values()),'Original queue still owns running work')
 need(set(sidecar['tasks'])<=set(tasks),'Exact selected sidecar task namespace')
 need(all(x['status']=='succeeded' and x.get('actual_wait') is True and x.get('actual_returncode')==0 and x.get('owned_process_group_absent') is True for x in sidecar['tasks'].values()),'Every adopted row requires a genuine completed successful wait')
 rows=copy.deepcopy(original['tasks']);prior={}
 for name,new in sidecar['tasks'].items():
  old=rows[name]
  need(old['status'] in ('pending','blocked') and 'pid' not in old,'Never replace a launched historical attempt')
  if old['status']=='blocked':need(old.get('error') in ('Output already exists, no overwrite','Required predecessor failed'),'Unexpected old failure cannot be silently adopted')
  prior[name]=copy.deepcopy(old);rows[name]=copy.deepcopy(new)
  rows[name]['adopted_from_sidecar']=True
 # Reset only never-launched descendants whose blockers are now absent. Genuine
 # failed history and branches still depending on it remain unchanged.
 reset=[];changed=True;affected=set(sidecar['tasks'])
 while changed:
  changed=False
  for name,row in list(rows.items()):
   if row['status']!='blocked' or row.get('error')!='Required predecessor failed' or 'pid' in row:continue
   if not affected.intersection(tasks.get(name,{}).get('dependencies',[])):continue
   if any(rows[d]['status'] in ('failed','blocked') for d in tasks.get(name,{}).get('dependencies',[])):continue
   if name not in tasks:continue
   prior.setdefault(name,copy.deepcopy(row));rows[name]=dict(status='pending',attempt=row.get('attempt',1),claim_block_released=True)
   reset.append(name);affected.add(name);changed=True
 return rows,prior,reset

def validate_wait(name,row,task):
 need(row.get('actual_wait') is True and row.get('actual_returncode')==0 and row.get('owned_process_group_absent') is True,'Genuine direct wait required')
 wait_path=r.path(task['output_directory'])/'actual_wait.json'
 wait=r.read(r.ref(wait_path))
 need(wait==row,'Immutable actual_wait differs from sidecar row')
 r.read(row['result_ref'])
 expected=r.path(task['output_directory'])/task.get('completion_file','result.json')
 need(r.path(row['result_ref']['path'])==expected,'Completion outside original exact output scope')
 need(not processes.session_members(row['pid']),'Successful sidecar process group still has live members')
 return r.ref(wait_path)

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--original-state',required=True)
 parser.add_argument('--sidecar-state',required=True,action='append')
 args=parser.parse_args()
 original_ref=r.ref(args.original_state);original=r.read(original_ref)
 effective=json.loads((HERE/'original_effective_plan.json').read_text())
 all_tasks={t['id']:t for t in effective['tasks']}
 need(set(original['tasks'])==set(all_tasks),'Exact original effective task namespace')
 need(not any(x['status']=='running' for x in original['tasks'].values()),'Original queue still owns running work')
 rows=copy.deepcopy(original['tasks']);all_prior={};all_reset=[];waits={};states=[];adopted=set()
 for path in args.sidecar_state:
  side_ref=r.ref(path);sidecar=r.read(side_ref);sideplan=r.read(sidecar['plan_ref'])
  need(original['plan_ref']==sideplan['original_identity']['plan_ref'] and original['plan_revision']==sideplan['original_identity']['plan_revision'],'Original revision changed; re-audit effective task definitions')
  definitions={t['id']:t for t in sideplan['tasks']}
  need(set(sidecar['tasks'])==set(definitions),'Complete selected sidecar task namespace required')
  selected={name:task for name,task in definitions.items() if task['kind']!='external'}
  need(not adopted.intersection(selected),'Overlapping sidecar ownership')
  need(set(selected)<=set(all_tasks),'Sidecar may adopt only original exact tasks')
  for name,task in definitions.items():
   if task['kind']=='external':
    need(original['tasks'][name]['status']=='succeeded' and sidecar['tasks'][name]['status']=='succeeded' and original['tasks'][name]['result_ref']==sidecar['tasks'][name]['result_ref']==task['result_ref'],'External input remains original evidence, never a new process wait')
  owned={'tasks':{name:sidecar['tasks'][name] for name in selected}}
  for name,row in owned['tasks'].items():
   waits[name]=validate_wait(name,row,selected[name])
  rows,prior,reset=merge_rows({'tasks':rows},owned,{**all_tasks,**selected})
  all_prior.update(prior);all_reset.extend(reset);all_tasks.update(selected);adopted.update(selected)
  states.append(side_ref)
 need(r.ref(args.original_state)==original_ref,'Original state changed while preparing proposal')
 result=dict(schema='exact_scope_future_merge_proposal_v2',created_unix=time.time(),
  original_state_ref=original_ref,sidecar_state_refs=states,
  sidecar_wait_refs=waits,rows=rows,effective_tasks=list(all_tasks.values()),
  preserved_prior_rows=all_prior,reset_claim_block_descendants=sorted(set(all_reset)),
  live_mutation_performed=False,launch_allowed=False,
  next_step='ROOT verifies both controllers own no running children, stops only then, seals snapshots, and launches a reviewed continuation using exactly these rows. Existing continuation.py cannot do this adoption and must not be silently reused.',
  failed_history_preserved=[name for name,row in original['tasks'].items() if row['status']=='failed'])
 r.save(HERE/'future_merge_proposal.json',result)
 print(json.dumps({'prepared':True,'adopted':len(adopted),'reset_only_claim_blocked':len(set(all_reset)),'live_state_changed':False}))

if __name__=='__main__':main()
