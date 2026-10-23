"""Assemble published worker deliveries into one explicit executable campaign.

Building a plan does not run a model or open ancestry truth. Existing observations
and predictions are reused through authenticated references.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
from . import runtime as r
from . import scheduler

RESEARCH = 'research/benchmark_rebuild_20261024'
LEGACY_STATE = '.runtime/benchmarking/workflow_20261022_gpu345_v2/state.json'


def load(name):
    return json.loads(r.path(name).read_text())


def rewrite_paths(value, replacements):
    if isinstance(value, dict):
        return {k: rewrite_paths(v, replacements) for k, v in value.items()}
    if isinstance(value, list):
        return [rewrite_paths(v, replacements) for v in value]
    if isinstance(value, str):
        for source, target in sorted(replacements.items(), key=lambda row: -len(row[0])):
            if value == source or value.startswith(source + '/'):
                return target + value[len(source):]
    return value


def legacy_value(identifier):
    row = load(LEGACY_STATE)['tasks'][identifier]
    if row['status'] != 'succeeded':
        raise ValueError('Required reusable predecessor is unfinished: ' + identifier)
    return r.read(row['result_ref']).get('value', row['result_ref'])


def native_american():
    delivery = load(RESEARCH + '/american/expanded_delivery/task_dag.json')
    replacements = {t['output_directory']: t['output_directory'] + '/native' for t in delivery['tasks']}
    tasks = []
    package = r.ROOT / 'src/arggnn/benchmarking/paper_american'
    sources = [r.ref(p) for p in sorted(package.glob('*.py'))]
    for source in delivery['tasks']:
        spec = rewrite_paths(source['scientific_spec'], replacements)
        if spec.get('stage') == 'prepare':
            source = dict(source, callable_module='arggnn.benchmarking.paper_american.preparation', callable_name='run_prepare')
        task = dict(id=source['id'], dependencies=source['dependencies'], kind=source['resource'],
                    python=source['python_executable'], output_directory=source['output_directory'],
                    max_output_bytes=source['expected_output_bytes'], max_rss_bytes=source['max_rss_bytes'],
                    max_seconds=source['max_seconds'], cpu_threads=source.get('CPU_threads', 8),
                    call=dict(module=source['callable_module'], function=source['callable_name'], spec=spec),
                    source_refs=sources + spec.get('requirements', {}).get('source_refs', []))
        if task['kind'] == 'gpu':
            task['minimum_free_gpu_bytes'] = (52 if spec['stage'] == 'contrastive' else 24) * 2**30
        tasks.append(task)
    return tasks


def completed_ap():
    state = load('.runtime/benchmarking/paper_ap_cnn_20261024/state.json')
    tasks = []
    for name, row in state['tasks'].items():
        if row['status'] != 'succeeded' or row.get('actual_returncode') != 0:
            raise ValueError('AP completion missing: ' + name)
        r.read(row['result_ref'])
        tasks.append(dict(id=name, kind='external', dependencies=[], result_ref=row['result_ref']))
    return tasks


def american_score(treeloom_task_by_parent):
    from arggnn.benchmarking import american_scoring
    source = next(t for t in load('configs/benchmarks/executable_workflow_20261022_gpu345_v2.json')['tasks']
                  if t['id'] == 'AM_score')
    args = source['factory_arguments']
    preds = args['prediction_refs_by_method']
    dependencies = []
    for parent in preds['treeloom']:
        new_treeloom = treeloom_task_by_parent[parent]
        native = 'paper_american_' + parent + '_predict'
        preds['treeloom'][parent] = {'$task': new_treeloom, 'path': ['completion_ref']}
        preds['argformer'][parent] = {'$task': native, 'path': ['completion_ref']}
        preds['flare'][parent] = legacy_value('AM_' + parent + '_flare_native')
        args['generation_ready_refs_by_parent'][parent] = legacy_value('AM_' + parent + '_generation')
        args['focal_preparation_refs_by_parent'][parent] = legacy_value('AM_' + parent + '_focals')
        dependencies.extend([new_treeloom, native])
    output = 'experiments/results/paper_campaign_20261024/american_score'
    args['output_directory'] = output + '/native'
    return dict(id='paper_american_all_methods_score', dependencies=sorted(set(dependencies)),
                kind='cpu', python='.runtime/env/bin/python', output_directory=output,
                max_output_bytes=32*2**20, max_rss_bytes=32*2**30, max_seconds=86400, cpu_threads=4,
                call={'factory': dict(module='arggnn.benchmarking.paper_american.scoring',
                                      function='score_factory', kwargs=args)},
                source_refs=american_scoring.requirements()['source_refs'] + [r.ref(p) for p in sorted((r.ROOT / 'src/arggnn/benchmarking/paper_american').glob('*.py'))])


def write_american_draft():
    tasks = native_american()
    return r.save(RESEARCH + '/integration/american_adapted_tasks.json', {'tasks': tasks,
                   'inert': True, 'native_output_subdirectory': 'native',
                   'not_a_complete_launch_plan': True})



def _argmix_source_refs(extra_refs=()):
    """Authenticate the new package, qualified history and original closure."""
    from arggnn.benchmarking.paper_argmix import paper_argmix_ap_prediction
    manifest_ref = r.ref(RESEARCH + '/argmix/paper_argmix_source_manifest.json')
    manifest = r.read(manifest_ref)
    requirements = paper_argmix_ap_prediction.requirements()
    candidates = (manifest['refs'] + manifest['external_qualified_history_refs']
                  + requirements['original']['source_refs'] + requirements['source_refs']
                  + [manifest_ref] + list(extra_refs))
    sources = {}
    for item in candidates:
        reference = {key: item[key] for key in ('path', 'sha256')}
        previous = sources.get(reference['path'])
        if previous is not None and previous != reference:
            raise ValueError('Conflicting ARGMix/AP source pins: ' + reference['path'])
        if r.ref(reference['path']) != reference:
            raise ValueError('ARGMix/AP source changed: ' + reference['path'])
        sources[reference['path']] = reference
    return [sources[name] for name in sorted(sources)]


def ap_treeloom():
    reuse = load(RESEARCH + '/ancestralpaths/frozen476_reuse.json')
    fit_id = 'paper_ap_treeloom_fit_existing'
    output = 'experiments/results/paper_campaign_20261024/ap_treeloom_predictions'
    args = dict(protocol_id='ancestralpaths_paper200mb_native6',
                sampling_plan_ref=reuse['sampling_plan_ref'],
                fit_completion_ref={'$task': fit_id, 'path': ['completion_ref']},
                heldout_inputs={p: x for p, x in reuse['treeloom_observation_focal_interfaces'].items() if '_test_' in p},
                output_directory=output + '/native', max_rss_bytes=512*2**30,
                preparation_headroom_bytes=128*2**30, prediction_headroom_bytes=64*2**30,
                observation_cache_bytes=0, max_sites=5000000, max_metadata_bytes=512*2**20,
                preparation_limits=dict(max_episode_payload_bytes=64*2**30, max_holds=500000, max_nodes=1000000000),
                max_output_bytes=64*2**20, gpu_uuid_map={str(k): v for k, v in r.GPU_UUIDS.items()})
    return [dict(id=fit_id, dependencies=[], kind='external',
                 state_path='.runtime/benchmarking/workflow_20261022_full_vram_v2/state.json',
                 source_task_id='AP__all_train__treeloom_fit'),
            dict(id='paper_ap_treeloom__all_test__predict', dependencies=[fit_id], kind='gpu',
                 python='.runtime/env/bin/python', output_directory=output, max_rss_bytes=512*2**30,
                 max_output_bytes=128*2**20, max_seconds=604800, minimum_free_gpu_bytes=56*2**30,
                 cpu_threads=8, call={'factory': dict(module='arggnn.benchmarking.paper_argmix.paper_argmix_ap_prediction',
                                                      function='prediction_factory', kwargs=args)},
                 source_refs=_argmix_source_refs())]


def argmix():
    delivery = load(RESEARCH + '/argmix/paper_argmix_jobs.json')
    sources = _argmix_source_refs([delivery['config_ref']])
    tasks = []
    for source in delivery['jobs']:
        gpu = source['resources']['gpu'] is not None
        output = r.path(source['output_directory'])
        completion = str(r.path(source['completion_file']).relative_to(output))
        task = dict(id='paper_argmix_' + source['id'],
                    dependencies=['paper_argmix_' + d for d in source['dependencies']],
                    kind='gpu' if gpu else 'cpu', python=source['argv'][0],
                    output_directory=str(output), argv=source['argv'], completion_file=completion,
                    max_output_bytes=source['output_cap_bytes']+8*2**20,
                    max_rss_bytes=source['resources']['max_rss_bytes'], max_seconds=source['timeout_seconds'],
                    cpu_threads=source['resources']['cpu_threads'], source_refs=sources,
                    additional_mutable_scopes=source.get('additional_mutable_scopes', []))
        if gpu:
            task['minimum_free_gpu_bytes'] = (56 if source['stage'].startswith('treeloom-') else 16) * 2**30
        tasks.append(task)
    return tasks


def american_retirement(tasks):
    mapping = load(RESEARCH + '/american/retirement_task_mapping.json')
    by_id = {t['id']: t for t in tasks}
    # The two reused corpora can fit concurrently without creating raw path data.
    # Every newly prepared corpus follows both reuse chains, then one predecessor.
    initial = ['argformer_american_test_000', 'argformer_american_test_001']
    fresh = ['argformer_american_validation_' + str(i).zfill(3) for i in range(5)]
    fresh += ['argformer_american_test_' + str(i).zfill(3) for i in range(2, 10)]
    previous = ['paper_american_' + parent + '_retire' for parent in initial]
    for parent in fresh:
        task = by_id['paper_american_' + parent + '_prepare']
        task['dependencies'] = sorted(set(task['dependencies'] + previous))
        previous = ['paper_american_' + parent + '_retire']
    sources = [r.ref(p) for p in sorted((r.ROOT / 'src/arggnn/benchmarking/paper_american').glob('*.py'))]
    retired_scopes = {}
    for source in mapping['tasks']:
        spec = source['scientific_spec']
        task = dict(id=source['id'], dependencies=source['dependencies'], kind='cpu',
                    python=source['python_executable'], output_directory=source['output_directory'],
                    max_output_bytes=source['expected_output_bytes'], max_rss_bytes=source['max_rss_bytes'],
                    max_seconds=source['max_seconds'], cpu_threads=source.get('CPU_threads', 8),
                    call=dict(module=source['callable_module'], function=source['callable_name'], spec=spec),
                    source_refs=sources,
                    additional_mutable_scopes=[dict(item, growth_allowed=False) for item in source['additional_mutable_scopes']])
        tasks.append(task)
        retired_scopes[task['id']] = str(r.path(task['output_directory']).parent.relative_to(r.ROOT))
    transient = max(sum(t['max_output_bytes'] for t in tasks if r.path(t['output_directory']).is_relative_to(r.path(parent_scope)))
                    for parent_scope in retired_scopes.values())
    return dict(scope='experiments/results/paper_american_20261024', task_ids=[t['id'] for t in tasks], retirement_ids=list(retired_scopes),
                retired_parent_scopes=retired_scopes, next_parent_transient_bytes=transient)


def american_treeloom(checkpoint_ref, evidence_ref):
    from research.benchmark_rebuild_20261024.american.treeloom_speed import task_builder
    delivery = task_builder.build(resume_checkpoint_ref=checkpoint_ref, resume_evidence_ref=evidence_ref)
    replacements = {t['output_directory']: t['output_directory'] + '/native' for t in delivery['tasks']}
    tasks, predictions = [], {}
    for source in delivery['tasks']:
        spec = rewrite_paths(source['scientific_spec'], replacements)
        task = dict(id=source['id'], dependencies=source['dependencies'], kind=source['resource'],
                    python=source['python_executable'], output_directory=source['output_directory'],
                    max_output_bytes=source['expected_output_bytes']+16*2**20,
                    max_rss_bytes=source['max_rss_bytes'], max_seconds=source['max_seconds'],
                    cpu_threads=source['CPU_threads'], minimum_free_gpu_bytes=56*2**30,
                    call=dict(module=source['callable_module'], function=source['callable_name'], spec=spec),
                    source_refs=source['source_refs'])
        if spec['stage'] == 'predict':
            for parent in spec['arguments']['heldout_inputs']:
                predictions[parent] = task['id']
        tasks.append(task)
    return tasks, predictions


def assemble(checkpoint_ref, evidence_ref):
    gst = load('configs/benchmarks/paper_gst_lai_8_20261024_final.json')['tasks']
    for task in gst:
        if task['kind'] == 'gpu':
            task['preferred_gpus'] = [5, 4, 3]
    native = native_american()
    floor = american_retirement(native)
    treeloom, prediction_ids = american_treeloom(checkpoint_ref, evidence_ref)
    score = american_score(prediction_ids)
    score['dependencies'] += floor['retirement_ids']
    capacity = load(RESEARCH + '/integration/capacity_reservation.json')
    tasks = gst + completed_ap() + ap_treeloom() + native + treeloom + [score] + argmix()
    plan = dict(schema='paper_campaign_v1', physical_gpus=[3,4,5], tasks=tasks,
                records_directory='.runtime/benchmarking/paper_campaign_20261024_v1',
                storage_ceiling_bytes=capacity['operational_ceiling_bytes'],
                reserved_existing_project_bytes=capacity['baseline_bytes'],
                existing_workers=[w for w in capacity['existing_workers'] if w['task_id'] != 'AM_treeloom_fit'],
                capacity_ref=r.ref(RESEARCH + '/integration/capacity_reservation.json'),
                pipeline_storage_floors=[floor], max_parallel_tasks=6, max_cpu_tasks=2, max_jobs_per_gpu=2,
                scope_limits=[dict(path='experiments/paper_argmix_1611_rebuild_20261024_v1', max_bytes=40*2**30)],
                queue_coverage=dict(AP='historical CNN, retained Dense control, TreeLoom, GST-LAI-8',
                                    American='paper-oriented ARGformer, unchanged FLARE, frozen TreeLoom',
                                    ARGMix='paper-oriented ARGMix and frozen TreeLoom on shared1611-person data'),
                exact_published_number_replication_claimed=False,
                reconstruction_notes='See source-bound condition reports for paper/release differences and unreleased settings.',
                resume_checkpoint_ref=checkpoint_ref, resume_evidence_ref=evidence_ref)
    safety = load(RESEARCH + '/gst_lai/active_source_safety_review.json')
    closure = [dict(path=v['path'], sha256=v['sha256']) for v in safety['modules'].values()]
    for task in tasks:
        if task['id'].startswith('paper_gst_') or task['id'] == 'paper_ap_treeloom__all_test__predict':
            combined = task.get('source_refs', []) + closure
            task['source_refs'] = list({v['path']: v for v in combined}.values())
    scheduler.validate(plan)
    return plan
