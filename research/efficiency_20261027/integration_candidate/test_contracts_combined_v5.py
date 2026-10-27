"""CPU-only configuration, real terminal metadata, release rejection and source contracts."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from arggnn.benchmarking import contracts as c
from . import qualification_combined_v5 as gate
from . import envelope_combined_v5 as envelope
from . import restore_entrypoint_combined_v5 as restore
from . import training_combined_v5 as training
from . import step_combined_v2 as step
from .configuration_combined_v5 import configuration, EXPECTED, PINS, FACTORY, GIB
from .monitor_timing_combined_v1 import ResourceTiming

BASE = c.ROOT/'research/efficiency_20261027/integration_candidate'


def fails(call):
    try:
        call()
    except (ValueError, KeyError, TypeError, AssertionError, RuntimeError):
        return
    raise AssertionError('Expected fail-closed contract rejection')


def function(path, name):
    tree = ast.parse(Path(path).read_text())
    return ast.dump(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name))


def run_record(directory):
    root = Path('research/efficiency_20261027/root/runs')/directory
    return {key: c.file_ref(root/name) for key, name in (
        ('completed_ref', 'data/completed.json'), ('launch_ref', 'launch.json'),
        ('request_ref', 'request.json'), ('actual_wait_ref', 'actual_wait.json'))}


def main():
    assert not torch.cuda.is_initialized()
    selected = configuration(dict(EXPECTED))
    for key in selected:
        altered = dict(selected)
        del altered[key]
        fails(lambda: configuration(altered))
    for key, value in (('max_rss_bytes', 592*GIB), ('resident_scratch_bytes', 15*GIB),
                       ('holds_per_block', 128), ('preparation_producers', 1),
                       ('boundary_fusion', 1), ('executor_factory', 'unreviewed_factory')):
        fails(lambda key=key,value=value: configuration(dict(selected, **{key:value})))
    prior = c.read_json(envelope.PREDECESSOR_JOB)
    prior_ref = c.file_ref(envelope.PREDECESSOR_JOB)
    declaration = envelope.resource_declaration(prior, prior_ref)
    assert envelope.validate_declaration(declaration) == prior
    release = dict(configuration=selected, resource_declaration=declaration)
    arguments = copy.deepcopy(envelope.predecessor_request(prior)['arguments'])
    arguments.update(max_rss_bytes=768*GIB, execution_configuration=selected)
    envelope.validate_arguments(prior, arguments, release)
    for key in ('parent_registry_ref', 'train_inputs', 'preparation_headroom_bytes',
                'training_headroom_bytes', 'observation_cache_bytes', 'max_truth_uncompressed_bytes',
                'output_directory', 'max_rss_bytes'):
        altered=copy.deepcopy(arguments)
        altered[key] = None
        fails(lambda: envelope.validate_arguments(prior, altered, release))
    request={'synthetic_callback_request':True}
    task=copy.deepcopy(prior)
    task['max_rss_bytes']=768*GIB
    task['minimum_free_gpu_bytes']=76*GIB
    assert task['preferred_gpus']==[3]
    task['call']=dict(module='research.efficiency_20261027.integration_candidate.restore_entrypoint_combined_v5',
        function='run_task', spec=request)
    class Callback:
        __module__='__main__'
        def __init__(self,task):self.task=task
        def __call__(self,*args):pass
    restore.validate_resource_callback(Callback(task),prior,release,request)
    insufficient=copy.deepcopy(task);insufficient['minimum_free_gpu_bytes']=56*GIB
    fails(lambda:restore.validate_resource_callback(Callback(insufficient),prior,release,request))
    assert set(declaration['changed_fields']) == {'max_rss_bytes',
        'call.spec.successor.arguments.max_rss_bytes','minimum_free_gpu_bytes'}
    assert declaration['changed_fields']['minimum_free_gpu_bytes']==dict(before=56*GIB,after=76*GIB)
    for key in envelope.ENVELOPE_KEYS:
        altered=copy.deepcopy(task)
        altered[key]=None
        fails(lambda: restore.validate_resource_callback(Callback(altered),prior,release,request))
    fails(lambda: restore.validate_resource_callback(Callback(task),prior,release,{}))
    for name in ('train_loop','validate_saved_state','validate_resume_checkpoint',
                 'validate_resume_metadata','schedule_identity','random_state','restore_random_state'):
        assert function(BASE/'training_adaptive_v2.py',name)==function(training.__file__,name), name
    assert step.build_executor.__module__+':'+step.build_executor.__name__ == FACTORY
    entry_source=(BASE/'entrypoint_combined_v5.py').read_text()
    trainer_source=Path(training.__file__).read_text()
    assert 'install_tree(' not in entry_source
    assert trainer_source.count('preprocessing_bindings = install_tree(child_credits)')==1
    assert 'worker_resource_check(*args, **kwargs)' in trainer_source
    assert '_current_rss_bytes() <= spec["memory_envelope"]["max_rss_bytes"]' in trainer_source
    timing=ResourceTiming()
    timing.record('a',.1,.2,.3);timing.record('a',.4,.5,.6);timing.record(12,0.,.1,0.)
    measured=timing.snapshot()
    assert measured['calls']==3 and measured['checks_skipped']==0
    assert measured['by_stage_label']['a']['calls']==2
    assert measured['maximum_outer_callback_seconds']==.5
    assert 'by_stage_label' not in timing.snapshot(labels=False)
    boundary_ref=run_record('boundary_full_gpu5_v1')
    relative_command=c.read_ref(boundary_ref['request_ref'])['command']
    directory,output=gate.qualification_paths(boundary_ref,relative_command)
    absolute_command=list(relative_command)
    absolute_command[absolute_command.index('--output')+1]=str(output)
    assert gate.qualification_paths(boundary_ref,absolute_command)==(directory,output)
    absolute_record={key:dict(reference,path=str(c.ROOT/reference['path'])) for key,reference in boundary_ref.items()}
    assert gate.qualification_paths(absolute_record,absolute_command)==(directory,output)
    for path in ('/tmp/unrelated-output',str(output/'..'/'data'),
                 'research/efficiency_20261027/root/runs/../runs/boundary_full_gpu5_v1/data'):
        changed=list(relative_command);changed[changed.index('--output')+1]=path
        fails(lambda:gate.qualification_paths(boundary_ref,changed))
    changed_record=copy.deepcopy(boundary_ref)
    changed_record['actual_wait_ref']['path']='/tmp/actual_wait.json'
    fails(lambda:gate.qualification_paths(changed_record,relative_command))
    boundary=gate.terminal_run(boundary_ref,'research.efficiency_20261027.root.compare_boundary_full_v1',
        gate.BOUNDARY_MANIFEST)
    gate.validate_numerical_release(boundary,'boundary_full',checkpoint_ref=boundary['checkpoint_ref'],
        source_manifest_ref=c.file_ref(gate.BOUNDARY_MANIFEST))
    resident_ref=run_record('upward_residency_full_gpu5_v2')
    resident=gate.terminal_run(resident_ref,
        'research.efficiency_20261027.architecture.exact_upward_residency.compare_full_v2',gate.RESIDENT_MANIFEST)
    gate.validate_resident(resident,boundary_ref['completed_ref'])
    for key in ('status','all147_Adam_tensors_checked','all49_gradient_and_parameter_tensors_checked'):
        altered=copy.deepcopy(resident);altered[key]=None
        fails(lambda: gate.validate_resident(altered,boundary_ref['completed_ref']))
    altered=copy.deepcopy(resident)
    altered['records'][1]['torch_peak_reserved_bytes']=75*GIB
    fails(lambda:gate.validate_resident(altered,boundary_ref['completed_ref']))
    altered=copy.deepcopy(resident)
    altered['records'][1]['explicit_vjp']['residency_lifetime']['direct_gradient_bytes']=1
    fails(lambda:gate.validate_resident(altered,boundary_ref['completed_ref']))
    spec=c.read_json(gate.PREPARATION_SPEC)
    from research.efficiency_20261027.execution.preparation_evidence_v2 import bounded_signature, SCHEMA
    synthetic_signature=bounded_signature(dict(arrays={'a':'a'*64},structures={},outer={},metadata={}),
        array_metadata={'a':dict(shape=[1],dtype='<i8',count=1)},resource_check=lambda stage:None)
    def validate_prepared(value,spec):
        with patch.object(gate,'read',return_value=synthetic_signature):
            gate.validate_preparation(value,spec)
    rows=[]
    for arm,seconds,children in [('current_serial_pair',200.,4),('adaptive_concurrent_pair',150.,8)]:
        rows.append(dict(arm=arm,parents=spec['parents'],signature_schema=SCHEMA,
            signature_refs={p:dict(path='synthetic/'+arm+'/'+p,sha256='a'*64) for p in spec['parents']},
            pair_prepare_pack_seconds=seconds,
            supply_seconds_per_parent=seconds/2,family_monitor=dict(samples=20,peaks=dict(
                main_RSS_bytes=100*GIB,family_PSS_or_RSS_bytes=120*GIB,encoder_children=children))))
    prepared=dict(schema='TreeLoom_two_parent_preparation_comparison_v2',status='passed',
        signature_encoding='canonical_complete_JSON_SHA256_v2',
        exact_serial_concurrent_signatures=True,saved_CPU_RNG_unchanged=True,
        actual_waits_and_all_tracked_families_absent=True,training_executed=False,truth_opened=False,
        spec_ref=c.file_ref(gate.PREPARATION_SPEC),arms=rows,pair_speed_ratio=200/150)
    validate_prepared(prepared,spec)
    for key in ('status','exact_serial_concurrent_signatures','saved_CPU_RNG_unchanged',
                'actual_waits_and_all_tracked_families_absent','pair_speed_ratio','signature_encoding'):
        altered=copy.deepcopy(prepared);altered[key]=None
        fails(lambda:validate_prepared(altered,spec))
    altered=copy.deepcopy(prepared);altered['arms'][1]['family_monitor']['peaks']['encoder_children']=4
    fails(lambda:validate_prepared(altered,spec))
    altered=copy.deepcopy(prepared);altered['arms'][1]['signature_refs']={}
    fails(lambda:validate_prepared(altered,spec))
    altered=copy.deepcopy(prepared);altered['schema']='TreeLoom_two_parent_preparation_comparison_v1'
    fails(lambda:validate_prepared(altered,spec))
    pending=dict(schema='TreeLoom_combined_execution_release_v1',status='pending_qualification',
        configuration=selected,source_pins_ref=c.file_ref(PINS))
    with patch.object(gate,'read',return_value=pending):
        fails(lambda:gate.validate_release(dict(synthetic=True)))
    probe=copy.deepcopy(arguments)
    probe['output_directory']='experiments/results/paper_american_20261024/combined_CPU_readonly_probe_v5'
    assert not c.contained_path(probe['output_directory']).exists()
    built=training.build_spec(**probe)
    assert built['memory_envelope']['max_rss_bytes']==768*GIB
    assert built['requirements']['executor_factory']==FACTORY
    assert not c.contained_path(probe['output_directory']).exists()
    assert not torch.cuda.is_initialized()
    receipt=dict(status='passed',candidate_source_pins_ref=c.file_ref(PINS),CPU_only=True,
        actual_attempt7_nested_spec_built_read_only=True,checkpoint268_only_used_for_CPU_contract_validation=True,
        exact_592_to_768GiB_envelope_validated=True,
        scheduler_minimum_free_GPU_56_to_76GiB_validated=True,preferred_physical_GPUs=[3],all_undeclared_scientific_and_resource_deltas_rejected=True,
        actual_full_residency_and_boundary_terminal_metadata_validated=True,
        absolute_and_relative_run_paths_accepted_traversal_and_outside_rejected=True,
        synthetic_preparation_rejection_tests=True,real_preparation_acceptance_still_pending=True,
        exact_schedule_checkpoint_RNG_functions_AST_equal=True,explicit_pre_forward_router_factory_bound=True,actual_route_receipt_per_complete_parent=True,
        callback_check_frequency_unchanged=True,callback_timing_covered=True,
        CUDA_initialized=False,jobs_started=0,production_modified=False)
    with (BASE/'CONTRACTS_COMBINED_CPU_v5.json').open('x') as stream:json.dump(receipt,stream,indent=2)
    print(json.dumps(receipt))


if __name__=='__main__':main()
