"""Controller-only resource view. Original plans, workers and sources stay unchanged."""
import argparse
import copy
import json
from pathlib import Path
import time
from research.repairs.benchmark_multigpu_20261022.runtime import readable_workflow_v2 as queue
from arggnn.benchmarking import workflow_predecessor_v5 as predecessor

c=queue.c
GIB=2**30
SCHEMA='parallel_controller_resource_overlay_v1'


def need(value,message):
    if not value:raise ValueError(message)


def memtotal_bytes():
    values=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    return int(values['MemTotal'].split()[0])*1024


def check_config(reference):
    config=c.read_ref(reference)
    need(config.get('schema')==SCHEMA and config.get('max_parallel_rss_bytes')==1536*GIB,'Exact1536GiB controller overlay required')
    required={c.file_ref(Path(__file__).parent/name)['path'] for name in ['controller_overlay_v1.py','prepare_overlay_v1.py','queue_reporting_overlay_v1.py']}
    need(required<={ref['path'] for ref in config['source_refs']},'All three explicit overlay sources must be pinned')
    for ref in config['source_refs']:
        need(c.file_ref(ref['path'])==ref,'Overlay plaintext source changed')
    base=c.read_ref(config['base_plan_ref'])
    need(base['physical_gpus']==[2,3,5] and base['host_ram_margin_bytes']>=64*GIB,'Original explicit device and RAM margin guards required')
    need(config['max_parallel_rss_bytes']+base['host_ram_margin_bytes']<=memtotal_bytes(),'Overlay exceeds physical RAM minus original margin')
    snapshot=c.read_ref(config['state_snapshot_ref'])
    need(snapshot['plan_ref']==config['base_plan_ref'],'Exact original controller snapshot required')
    need(c.contained_path(config['records_directory'])!=c.contained_path(base['records_directory']),'Separate overlay records required')
    return config,base,snapshot


def plan_view(config,base,snapshot):
    view=copy.deepcopy(base)
    bindings=config['active_predecessors']
    by_id={task['id']:task for task in base['tasks']}
    replaced={row['source_task_id']:row for row in bindings}
    need(len(replaced)==len(bindings) and 1<=len(bindings)<=2,'One or two exact original live tasks required')
    for identifier,binding in replaced.items():
        need(binding['source_plan_ref']==config['base_plan_ref'] and binding['source_task_fingerprint']==c.fingerprint(by_id[identifier]),'Exact unchanged original active task required')
        entry=snapshot['tasks'][identifier]
        need(entry['status']=='running' and entry['job_ref']==binding['original_job_ref'] and entry['pid']==binding['original_supervisor_pid'],'Exact captured active job required')
    view['tasks']=[dict(copy.deepcopy(task),resource='external',workflow_predecessor=copy.deepcopy(replaced[task['id']]))
        if task['id'] in replaced else copy.deepcopy(task) for task in base['tasks']]
    historical=copy.deepcopy(base.get('workflow_external_predecessors',[]))
    for carry in config.get('historical_retention_carries',[]):
        original=carry['original_binding']
        need(original in historical and original['source_task_id'] in replaced,'Only the duplicate historical predecessor may be carried')
        _,entry=predecessor._source(original)
        need(predecessor._drained(original,entry) is not None,'Historical carry must already be genuinely drained')
        current_charge=entry['retained_bytes']
        need(carry['original_current_charge_bytes']==current_charge and carry['cap_bytes']>=max(original['max_output_bytes'],current_charge),'Historical full charge cannot decrease')
        historical.remove(original)
    need(not ({row['source_task_id'] for row in historical}&set(replaced)),'Duplicate predecessor accounting must be explicitly retained')
    view['workflow_external_predecessors']=historical+copy.deepcopy(bindings)
    view['external_storage']=copy.deepcopy(base.get('external_storage',[]))+[
        dict(cap_bytes=row['cap_bytes'],scopes=[row['original_binding']['output_directory']],completion_evidence=[],
             reason='Conservative fixed historical retention carry with zero new credit')
        for row in config.get('historical_retention_carries',[])]
    view['records_directory']=config['records_directory']
    view['max_parallel_rss_bytes']=config['max_parallel_rss_bytes']
    view['adopted_state_snapshot_ref']=config['state_snapshot_ref']
    view['adopted_completed_task_fingerprints']={identifier:c.fingerprint(by_id[identifier])
        for identifier,entry in snapshot['tasks'].items() if entry['status']=='succeeded'}
    queue.authenticated_adopted_completed(view)
    return view


def install_controller_view(reference):
    config,base,snapshot=check_config(reference)
    original_load=queue.load_plan
    original_policy=queue.current_policy
    view=plan_view(config,base,snapshot)
    worker_module,worker_source=queue.MODULE,queue.SOURCE
    def controller_policy():
        policy=original_policy()
        need(policy['approved_workflow_ref']==config['base_plan_ref'],'Original parallel workflow approval must remain unchanged')
        need(policy.get('approved_parallel_resource_overlay_ref')==reference,'Separate exact resource overlay approval required')
        check_config(reference)
        result=dict(policy)
        if result.get('workflow_drain_ref')==config['base_plan_ref']:result.pop('workflow_drain_ref')
        if result.get('parallel_resource_overlay_drain_ref')==reference:result['workflow_drain_ref']=config['base_plan_ref']
        return result
    def controller_load(plan_ref,*,approved=False):
        need(plan_ref==config['base_plan_ref'],'Overlay must retain the original worker plan reference')
        original_load(plan_ref,approved=approved)
        if approved:controller_policy()
        return copy.deepcopy(view)
    class OverlayReporter(queue.ProgressReporter):
        def submit(self,state,final=False):
            if not final and time.monotonic()-self.last<60:return
            from .queue_reporting_overlay_v1 import publish
            publish(self.plan,self.ref,state,queue.save)
            self.last=time.monotonic()
    queue.current_policy=controller_policy
    queue.load_plan=controller_load
    queue.ProgressReporter=OverlayReporter
    need((queue.MODULE,queue.SOURCE)==(worker_module,worker_source),'Original worker module/source must remain unchanged')
    return config,view


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-ref',required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    reference=json.loads(args.config_ref)
    config,view=install_controller_view(reference)
    result=queue.run(config['base_plan_ref'],execute_jobs=args.execute)
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__=='__main__':main()
