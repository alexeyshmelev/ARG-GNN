"""Plan-only same-parent output repair. Original factories and science are retained.

No launch, policy mutation, state reset, output creation or model import occurs.
ROOT decides which never-launched stages receive fresh output scopes in a new plan.
"""
from copy import deepcopy
from pathlib import Path
from arggnn.benchmarking import contracts as c
from arggnn.benchmarking import american_competitors as original

COMPETITORS='arggnn.benchmarking.american_competitors'
RETENTION='arggnn.benchmarking.retention'
AP_ARCHIVE='arggnn.benchmarking.european_retention_v9'


def need(ok,message):
    if not ok:raise ValueError(message)


def result_value(expression,state):
    """Resolve the original sealed metadata envelope, without opening model arrays."""
    need(isinstance(expression,dict) and '$task' in expression
         and set(expression)<={'$task','read','path'},'Explicit producer task reference required')
    entry=state['tasks'][expression['$task']]
    need(entry['status']=='succeeded','Actual completed input metadata required: '+expression['$task'])
    value=c.read_ref(entry['result_ref'])['value']
    if expression.get('read',False):value=c.read_ref(value)
    for component in expression.get('path',[]):value=value[component]
    return value


def owner_scope(task):
    if 'output_directory' in task:return task['output_directory']
    for key in ('workflow_predecessor','standalone_predecessor'):
        if key in task:return task[key]['output_directory']
    raise ValueError('No authenticated producer scope for '+task['id'])


def parent_ids(plan,state):
    """Use the unchanged inference/focal join, including its truth-free identity checks."""
    parents={};cache={}
    for task in plan['tasks']:
        if task.get('factory_module')!=COMPETITORS:continue
        arguments=task['factory_arguments']
        inference=result_value(arguments['inference_ref'],state)
        focal=result_value(arguments['focal_preparation_ref'],state)
        key=c.fingerprint([inference,focal])
        if key not in cache:
            source,_,_=original.inputs(inference,focal)
            parent=source['parent_id']
            need(isinstance(parent,str) and parent not in ('','.','..')
                 and Path(parent).parts==(parent,),'One canonical same-parent directory component required')
            cache[key]=parent
        parents[task['id']]=cache[key]
    return parents


def rebind_owner_paths(plan,state):
    """Run after every final output rebase, preserving successful historical descriptors."""
    by_id={task['id']:task for task in plan['tasks']};changes=[]
    for task in plan['tasks']:
        if state['tasks'][task['id']]['status']=='succeeded' or task['resource']=='external':continue
        arguments=task.get('factory_arguments',{})
        if task.get('factory_module')==RETENTION:
            identifiers=[arguments[key]['$task'] for key in ('preparation_ref','embedding_ref','prediction_ref')]
            scopes=[owner_scope(by_id[identifier]) for identifier in identifiers]
            before=deepcopy(arguments['owner_scopes']);arguments['owner_scopes']=scopes
            changes.append(dict(task_id=task['id'],field='owner_scopes',before=before,after=scopes))
        elif task.get('factory_module')==AP_ARCHIVE:
            scope=owner_scope(by_id[arguments['owner_task_id']])
            before=arguments['owner_scope'];arguments['owner_scope']=scope
            changes.append(dict(task_id=task['id'],field='owner_scope',before=before,after=scope))
    return changes


def repair_plan_paths(plan,state,output_root):
    """Return a copied plan and audit evidence; never change the original state.

    Successful outputs stay exact. Running, reserved or already materialized
    competitor attempts require ROOT reconciliation and are never rebound here.
    """
    root=c.contained_path(output_root)
    need(root.is_relative_to(c.ROOT/'experiments/results') and root.resolve()==root
         and 'benchmark_campaign_20261018_workflow' in root.parts,
         'Canonical fresh project result namespace required')
    revised=deepcopy(plan);parents=parent_ids(plan,state);changes=[]
    for task in revised['tasks']:
        if task['id'] not in parents:continue
        entry=state['tasks'][task['id']]
        if entry['status']=='succeeded':continue
        need(entry['status'] in ('pending','failed','blocked')
             and not entry.get('job_ref') and not entry.get('reservation_bytes'),
             'Only never-launched competitor stages may receive new output scopes: '+task['id'])
        old=c.contained_path(task['output_directory'])
        need(not old.exists(),'Existing competitor attempt output must be preserved and reconciled: '+str(old))
        new=root/'american'/parents[task['id']]/task['id']
        need(not new.exists(),'Fresh competitor output scope already exists: '+str(new))
        original._out(str(new/'native'),parents[task['id']],fresh=True)
        before=task['output_directory'];task['output_directory']=str(new.relative_to(c.ROOT))
        changes.append(dict(task_id=task['id'],parent_id=parents[task['id']],before=before,
            after=task['output_directory'],factory_arguments_unchanged=True,scientific_settings_unchanged=True))
    ownership=rebind_owner_paths(revised,state)
    for old,new in zip(plan['tasks'],revised['tasks']):
        if state['tasks'][old['id']]['status']=='succeeded':
            need(old==new,'Completed stage descriptor changed: '+old['id'])
        if old.get('factory_module')==COMPETITORS:
            need({key:value for key,value in old.items() if key!='output_directory'}==
                 {key:value for key,value in new.items() if key!='output_directory'},
                 'The repair may change only a competitor output scope')
    return revised,dict(schema='same_parent_competitor_plan_paths_v1',output_root=str(root.relative_to(c.ROOT)),
        scope_changes=changes,owner_path_changes=ownership,completed_stage_descriptors_unchanged=True,
        original_factories_unchanged=True,original_author_settings_unchanged=True,state_mutated=False,
        scientific_launches=0,checkpoint_payloads_opened=0,heldout_truth_opened=False)
