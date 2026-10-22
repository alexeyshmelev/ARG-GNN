"""Prepare explicit overlay config and immutable current-state snapshot only."""
import argparse
import copy
import json
from pathlib import Path
from . import controller_overlay_v1 as overlay

c=overlay.c
HERE=Path(__file__).parent


def prepare(base_plan_ref,output_directory,records_directory,active_task_ids):
    base=c.read_ref(base_plan_ref)
    state_path=c.contained_path(base['records_directory'])/'state.json'
    state=c.read_json(state_path)
    overlay.need(state['plan_ref']==base_plan_ref,'Exact original controller state required')
    actual={identifier for identifier,entry in state['tasks'].items() if entry['status']=='running'}
    overlay.need(actual==set(active_task_ids),'Exact configured current active task set required')
    by_id={task['id']:task for task in base['tasks']}
    directory=c.contained_path(output_directory);directory.mkdir(parents=True,exist_ok=False)
    snapshot_ref=overlay.queue.save(directory/'source_state_snapshot.json',state,exclusive=True)
    bindings=[];carries=[]
    for identifier in active_task_ids:
        task=by_id[identifier];entry=state['tasks'][identifier]
        identity_ref=c.file_ref(c.contained_path(task['output_directory'])/'supervision_identity.json')
        identity=c.read_ref(identity_ref)['child_identity']
        supervisor=overlay.queue.shared_gpu.identity(entry['pid'])
        binding=dict(source_plan_ref=base_plan_ref,source_state_path=str(state_path.relative_to(c.ROOT)),
            source_state_snapshot_ref=snapshot_ref,source_task_id=identifier,source_task_fingerprint=c.fingerprint(task),
            original_job_ref=entry['job_ref'],original_supervision_identity_ref=identity_ref,
            original_supervisor_pid=entry['pid'],original_supervisor_start_ticks=supervisor['start_ticks'],
            owned_process_group_id=identity['pid'],owned_group_leader_start_ticks=identity['start_ticks'],
            owned_group_leader_session=identity['session'],max_rss_bytes=task['max_rss_bytes'],
            max_output_bytes=task['max_output_bytes'],resource=task['resource'],output_directory=task['output_directory'])
        overlay.predecessor._source(binding)
        bindings.append(binding)
        for historical in base.get('workflow_external_predecessors',[]):
            if historical['source_task_id']!=identifier:continue
            _,old=overlay.predecessor._source(historical)
            overlay.need(overlay.predecessor._drained(historical,old) is not None,'Only already-drained duplicate may be carried')
            carries.append(dict(original_binding=copy.deepcopy(historical),original_current_charge_bytes=old['retained_bytes'],
                original_source_state_ref=c.file_ref(historical['source_state_path']),
                cap_bytes=max(historical['max_output_bytes'],old['retained_bytes'])))
    config=dict(schema=overlay.SCHEMA,base_plan_ref=base_plan_ref,state_snapshot_ref=snapshot_ref,
        active_predecessors=bindings,historical_retention_carries=carries,records_directory=records_directory,
        max_parallel_rss_bytes=1536*overlay.GIB,observed_physical_RAM_bytes=overlay.memtotal_bytes(),
        source_refs=[c.file_ref(HERE/name) for name in ['controller_overlay_v1.py','prepare_overlay_v1.py','queue_reporting_overlay_v1.py']],
        scientific_sources_or_settings_changed=False,old_worker_module=overlay.queue.MODULE,old_worker_source=overlay.queue.SOURCE)
    overlay.plan_view(config,base,state)
    reference=overlay.queue.save(directory/'overlay_config.json',config,exclusive=True)
    overlay.check_config(reference)
    return reference


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-plan-ref',required=True)
    parser.add_argument('--output-directory',required=True)
    parser.add_argument('--records-directory',required=True)
    parser.add_argument('--active-task-id',action='append',required=True)
    args=parser.parse_args()
    reference=prepare(json.loads(args.base_plan_ref),args.output_directory,args.records_directory,args.active_task_id)
    print(json.dumps(dict(config_ref=reference),sort_keys=True),flush=True)


if __name__=='__main__':main()
