"""Credit one genuinely retired external VAL003 embedding scope, without rewriting it.

The old successful controller entry remains the charged accounting authority.
Only its exact charged bytes minus the measured remaining scope may be credited,
only after the new current-plan retirement has genuinely succeeded and drained.
"""
from pathlib import Path
import os
import stat
from arggnn.benchmarking import contracts as c
from arggnn.benchmarking import workflow_predecessor_v5 as predecessor
from arggnn.benchmarking.workflow_v4 import scope_bytes

SOURCE='research/repairs/benchmark_gpu345_20261022/storage/external_retirement_v1.py'
FIELD='external_native_retirement_credit'
SCHEMA='external_native_retirement_credit_v1'
OWNER='AM_argformer_american_validation_003_argformer_embeddings'
RETIREMENT='AM_argformer_american_validation_003_retire_native_scratch'
PREPARATION='AM_argformer_american_validation_003_argformer_prepare'
PREDICTION='AM_argformer_american_validation_003_argformer_retrieval'
OWNER_SCOPE='experiments/results/benchmark_campaign_20261018_workflow/release_20261022_afternoon_v3/american/argformer_american_validation_003/'+OWNER
CHARGED_BYTES=49814945793


def need(ok,message):
    if not ok:raise ValueError(message)


def _task(plan,identifier):
    matches=[task for task in plan['tasks'] if task['id']==identifier]
    need(len(matches)==1,'One exact named task required: '+identifier)
    return matches[0]


def _scope(task):
    return c.contained_path(task.get('output_directory') or task['workflow_predecessor']['output_directory'])


def protected_owner_refs():
    """Prepare data-only pins for all original regular files outside disposable scratch."""
    owner=c.contained_path(OWNER_SCOPE)
    scratch=owner/'native/scratch'
    refs=[]
    for directory,subdirectories,files in os.walk(owner,followlinks=False):
        subdirectories[:]=[name for name in subdirectories if Path(directory)/name!=scratch]
        for name in subdirectories+files:
            path=Path(directory)/name
            need(not path.is_symlink(),'External owner must contain no unreviewed symbolic links')
        for name in files:
            path=Path(directory)/name
            need(stat.S_ISREG(path.lstat().st_mode),'Protected owner entries must be ordinary regular files')
            refs.append(c.file_ref(path))
    return sorted(refs,key=lambda ref:ref['path'])


def _success_wait(wait,job_ref):
    need(wait.get('job_ref')==job_ref and wait.get('actual_direct_wait') is True
         and type(wait.get('actual_direct_returncode')) is int and wait['actual_direct_returncode']==0
         and wait.get('owned_process_group_absent') is True and wait.get('supervisor_error') is None,
         'Actual exact-job successful direct wait and absent process group required')


def validate_plan(plan):
    field=plan.get(FIELD)
    need(isinstance(field,dict) and field.get('schema')==SCHEMA
         and field.get('owner_task_id')==OWNER and field.get('retirement_task_id')==RETIREMENT,
         'Only one explicitly named VAL003 external retirement credit is supported')
    need(field.get('helper_source_ref')==c.file_ref(SOURCE)
         and field['helper_source_ref'] in plan['sources'],'The exact external retirement helper must be source-pinned')
    binding=field['original_binding']
    bindings=[row for row in plan.get('workflow_external_predecessors',[]) if row['source_task_id']==OWNER]
    owner=_task(plan,OWNER)
    need(bindings==[binding] and owner.get('resource')=='external'
         and owner.get('workflow_predecessor')==binding
         and c.contained_path(binding['output_directory'])==c.contained_path(OWNER_SCOPE),
         'The unchanged external owner binding must remain separately charged exactly once')
    original_task,current_entry=predecessor._source(binding)
    frozen=c.read_ref(field['owner_success_snapshot_ref'])
    need(frozen.get('plan_ref')==binding['source_plan_ref']
         and frozen['tasks'].get(OWNER)==current_entry,
         'Original successful owner state and charged entry must remain immutable')
    need(current_entry.get('status')=='succeeded' and type(current_entry.get('actual_controller_wait')) is int
         and current_entry['actual_controller_wait']==0 and current_entry.get('owned_process_group_absent') is True
         and current_entry.get('job_ref')==binding['original_job_ref']
         and type(current_entry.get('retained_bytes')) is int
         and current_entry['retained_bytes']==field.get('original_retained_bytes')==CHARGED_BYTES
         and current_entry['retained_bytes']<=binding['max_output_bytes'],
         'Only the exact original genuinely completed 49814945793-byte charge is eligible')
    need(field.get('original_result_ref')==current_entry['result_ref']
         and c.contained_path(field['original_result_ref']['path'])==c.contained_path(OWNER_SCOPE)/'result.json'
         and c.contained_path(field['original_wait_ref']['path'])==c.contained_path(OWNER_SCOPE)/'actual_wait.json',
         'Original owner result and wait references must remain in their exact scope')
    _success_wait(c.read_ref(field['original_wait_ref']),binding['original_job_ref'])
    envelope=c.read_ref(field['original_result_ref'])
    need(envelope.get('job_ref')==binding['original_job_ref']
         and c.contained_path(envelope['value']['path'])==c.contained_path(OWNER_SCOPE)/'native/stage_completed.json',
         'Original owner envelope must retain its exact native metadata')
    embedding=c.read_ref(envelope['value'])
    need(embedding.get('stage')=='embeddings' and embedding.get('parent_id')=='argformer_american_validation_003'
         and c.contained_path(embedding['disposable_scratch_directory'])==c.contained_path(OWNER_SCOPE)/'native/scratch',
         'Only the original completed native embedding scratch is eligible')
    pins=field.get('protected_owner_refs')
    need(isinstance(pins,list) and pins and pins==protected_owner_refs(),
         'Every protected original owner file outside disposable scratch must remain byte-identical')
    mandatory=[field['original_result_ref'],field['original_wait_ref'],envelope['value'],binding['original_supervision_identity_ref']]
    need(all(ref in pins for ref in mandatory),'Protected owner provenance and completion refs must be retained')
    retirement=_task(plan,RETIREMENT)
    need(retirement.get('resource')=='cpu' and retirement.get('factory_module')=='arggnn.benchmarking.retention'
         and retirement.get('factory_function')=='factory' and retirement.get('dependencies')==[PREDICTION]
         and RETIREMENT not in plan.get('adopted_completed_task_fingerprints',{}),
         'Only a fresh exact native scratch-retirement task may release the external charge')
    need(all(OWNER not in task.get('remeasure_succeeded_tasks',[]) for task in plan['tasks']),
         'The external owner must never also receive local scope remeasurement credit')
    scopes=[_scope(_task(plan,PREPARATION)),c.contained_path(OWNER_SCOPE),_scope(_task(plan,PREDICTION))]
    arguments=retirement['factory_arguments']
    need([c.contained_path(value) for value in arguments['owner_scopes']]==scopes,
         'Retirement must name the exact current preparation, external embedding and prediction owners')
    for key,identifier in [('preparation_ref',PREPARATION),('embedding_ref',OWNER),('prediction_ref',PREDICTION)]:
        need(arguments.get(key)=={'$task':identifier,'path':[],'read':False},
             'Retirement inputs must come only from the exact declared successful producer')
    need(c.contained_path(embedding['preparation_ref']['path']).is_relative_to(scopes[0]/'native'),
         'Original embeddings must bind this exact preparation owner')
    return field


def completed_storage_reduction(plan,state):
    field=validate_plan(plan)
    entry=state['tasks'][RETIREMENT]
    if entry.get('status')!='succeeded':return 0
    need(c.read_ref(state['plan_ref'])==plan,'The retirement state must bind this exact active plan')
    task=_task(plan,RETIREMENT)
    need(entry.get('external') is not True and type(entry.get('actual_controller_wait')) is int
         and entry['actual_controller_wait']==0 and entry.get('owned_process_group_absent') is True,
         'Retirement credit requires the actual successful controller wait and group drain')
    job=c.read_ref(entry['job_ref'])
    need(job.get('schema')=='arggnn_workflow_job_v1' and job.get('plan_ref')==state['plan_ref']
         and job.get('task')==task and job.get('task_fingerprint')==c.fingerprint(task)
         and field['helper_source_ref'] in job.get('source_refs',[])
         and job.get('call',{}).get('module')=='arggnn.benchmarking.retention'
         and job['call'].get('function')=='run','Retirement job must bind the exact current plan, task and source')
    output=c.contained_path(task['output_directory'])
    spec=job['call']['spec']
    need(spec.get('schema')=='successful_native_scratch_retirement_v1'
         and c.contained_path(spec['output_directory'])==output/'native'
         and spec.get('owner_scopes')==task['factory_arguments']['owner_scopes'],
         'Exact current scratch-retirement owner scopes and output required')
    owner_entry=state['tasks'][OWNER]
    need(owner_entry.get('status')=='succeeded' and owner_entry.get('external') is True
         and not owner_entry.get('reservation_bytes') and not owner_entry.get('retained_bytes'),
         'The original embedding owner must remain a successfully adopted external producer')
    external=c.read_ref(owner_entry['result_ref'])
    binding=field['original_binding']
    need(external.get('status')=='succeeded' and external.get('external_original_job_ref')==binding['original_job_ref']
         and external.get('external_source_result_ref')==field['original_result_ref']
         and external.get('external_actual_wait_ref')==field['original_wait_ref']
         and external.get('value')==spec.get('embedding_ref'),
         'Current external adoption must preserve the exact original embedding evidence')
    for key,identifier in [('preparation_ref',PREPARATION),('embedding_ref',OWNER),('prediction_ref',PREDICTION)]:
        producer=state['tasks'][identifier]
        need(producer.get('status')=='succeeded','Every retirement producer must genuinely succeed')
        value=c.read_ref(producer['result_ref']).get('value')
        need(spec.get(key)==value,'Actual retirement spec must use the exact successful producer result')
    _success_wait(c.read_ref(c.file_ref(output/'actual_wait.json')),entry['job_ref'])
    envelope=c.read_ref(entry['result_ref'])
    need(c.contained_path(entry['result_ref']['path'])==output/'result.json'
         and envelope.get('job_ref')==entry['job_ref']
         and c.contained_path(envelope['value']['path'])==output/'native/retirement_completed.json',
         'Retirement completion must be the exact current worker result')
    receipt=c.read_ref(envelope['value'])
    directories=[str(c.contained_path(value)/'native/scratch') for value in spec['owner_scopes'][:2]]
    expected=[str(c.contained_path(value).relative_to(c.ROOT)) for value in directories]
    need(receipt.get('successful_new_scratch_only') is True and receipt.get('original_sources_unchanged') is True
         and receipt.get('original_failures_preserved') is True
         and receipt.get('retired_directories')==expected
         and c.contained_path(receipt['inventory_ref']['path'])==output/'native/retirement_inventory.json',
         'Only the committed exact two-scope native scratch retirement is eligible')
    inventory=c.read_ref(receipt['inventory_ref'])
    need(inventory.get('schema')==spec['schema'] and inventory.get('directories')==expected
         and all(inventory.get(key)==spec.get(key) for key in ('preparation_ref','embedding_ref','prediction_ref')),
         'Committed retirement inventory must retain the exact current producer evidence')
    roots=[c.contained_path(value) for value in expected]
    need(all(not root.exists() and not root.is_symlink() for root in roots),'All exact declared scratch directories must actually be absent')
    rows=inventory.get('files');seen=set()
    need(isinstance(rows,list) and rows,'Actual committed scratch inventory required')
    for row in rows:
        path=c.contained_path(row['path'])
        need(path not in seen and any(path.is_relative_to(root) and path!=root for root in roots)
             and type(row.get('bytes')) is int and row['bytes']>=0
             and type(row.get('mtime_ns')) is int and row['mtime_ns']>=0
             and not path.exists() and not path.is_symlink(),'Inventory may name only distinct actually retired scratch files')
        seen.add(path)
    prediction=c.read_ref(spec['prediction_ref'])
    seal=c.read_ref(prediction['prediction_seal_ref'])
    need(prediction.get('stage')=='retrieval' and prediction.get('parent_id')=='argformer_american_validation_003'
         and prediction.get('records')==10000 and prediction.get('scoring_performed') is False
         and (seal.get('query_ancestry_truth_opened') is False or seal.get('private_truth_opened') is False),
         'Protected final native prediction and truth-sealed metadata must remain present')
    # Reauthenticate the original byte charge and all protected owner files after
    # reading retirement proofs. Never count logical inventory sizes as credit.
    validate_plan(plan)
    retained=scope_bytes(c.contained_path(OWNER_SCOPE))
    need(type(retained) is int and 0<=retained<=field['original_retained_bytes'],
         'Only measured remaining bytes within the original immutable charge can be credited')
    return field['original_retained_bytes']-retained
