"""Read-only effective campaign state and authenticated comparator disclosure."""
from collections import Counter
from pathlib import Path
import hashlib
import json
from arggnn.benchmarking.paper_campaign import runtime as r

BASE = 'research/joint_evaluation_20261029'
CONFIG = BASE + '/root/reporting_20261030_v1/config_v1.json'


def need(value, message):
    if not value:
        raise ValueError(message)


def read_ref(reference):
    path = r.path(reference['path'])
    payload = path.read_bytes()
    need(hashlib.sha256(payload).hexdigest() == reference['sha256'], 'Reporting reference changed: ' + str(path))
    if 'bytes' in reference:
        need(len(payload) == reference['bytes'], 'Reporting reference size changed')
    return json.loads(payload)


def optional(path):
    path = r.path(path)
    return json.loads(path.read_text()) if path.exists() else None


def successful(row):
    return (row.get('actual_wait') is True and row.get('actual_returncode') == 0
            and row.get('owned_process_group_absent') is True and row.get('products_complete', True) is True)


def condition_label(condition):
    labels = {
        'equation_corrected': 'ARGMix, registered equation-corrected variant (author-derived Transformer)',
        'author_key_i': 'ARGMix, author key-i implementation',
    }
    need(condition in labels, 'Unknown ARGMix condition must not receive a generic label')
    return labels[condition]


def selected_metadata(reference):
    selected = read_ref(reference)
    need(selected.get('schema') == 'paper_argmix_selected_model_v1', 'Selected ARGMix model required')
    wait_path = r.path(reference['path']).parent / 'actual_wait.json'
    wait = optional(wait_path)
    need(wait is not None and successful(wait), 'Selected model requires genuine successful wait')
    need(wait.get('result_ref', {}).get('sha256') == reference['sha256'] and
         r.path(wait['result_ref']['path']) == r.path(reference['path']), 'Selected model wait binding differs')
    command = wait['command']
    need(command[command.index('--condition') + 1] == selected['condition'], 'Executed condition differs')
    condition_label(selected['condition'])
    return dict(condition=selected['condition'], label=condition_label(selected['condition']),
                selected_model_ref=reference, actual_wait_ref=r.ref(wait_path),
                epochs_completed=selected['epochs_completed'], selected_epoch=selected['selected_epoch'],
                training_examples=selected['training_examples'], validation_examples=selected['validation_examples'],
                selection=selected['selection'], validation_accuracy=selected['validation_accuracy'],
                published_table_reproduction=False,
                fidelity='Registered reconstructed benchmark; the condition is disclosed separately from published numerical-table reproduction.')


def comparator_metadata(scores):
    references = []
    for row in scores['results']:
        prediction = read_ref(row['prediction_ref'])
        need(prediction.get('schema') == 'paper_argmix_predictions_v1', 'Native ARGMix predictions required')
        references.append(prediction['selected_model_ref'])
    need(len(references) == 8 and all(value == references[0] for value in references), 'All eight comparator predictions must use one selected model')
    return selected_metadata(references[0])


def merge_retry(effective, current, manifest, plan, state, plan_ref):
    """Overlay owned task IDs only; external rows never become second owners."""
    owned = set(manifest['preserved_original_rows'])
    planned = {task['id'] for task in plan['tasks'] if task['kind'] != 'external'}
    need(owned == planned and len(owned) == 20, 'Exact twenty native retry owners required')
    need(state['plan_ref'] == plan_ref, 'Active retry state belongs to another plan')
    need(owned <= set(effective) and owned <= set(state['tasks']), 'Retry cannot add or omit campaign task identities')
    need(all(current['tasks'][name] == manifest['preserved_original_rows'][name] for name in owned), 'Historical failed/blocked original rows changed')
    merged = dict(effective)
    merged.update({name: state['tasks'][name] for name in owned})
    history = {name: {key: row[key] for key in ('status', 'attempt', 'error', 'actual_returncode') if key in row}
               for name, row in manifest['preserved_original_rows'].items()}
    return merged, history, {name: state['tasks'][name] for name in sorted(owned)}


def snapshot(config_path=CONFIG):
    config = optional(config_path)
    need(config is not None and config['schema'] == 'joint_reporting_effective_state_config_v1', 'Explicit reporting config required')
    original = optional(config['original_state_path'])
    successor = optional(config['successor_state_path'])
    effective = dict(original['tasks'])
    extra = {}
    if successor:
        effective.update({name: row for name, row in successor['tasks'].items() if name in effective})
        extra = {name: row for name, row in successor['tasks'].items() if name not in effective}
    overlays = {}
    for path in config['overlay_state_paths']:
        state = optional(path)
        if state:
            need(set(state['tasks']) <= set(effective), 'Campaign overlay cannot add identities')
            effective.update(state['tasks'])
            overlays[path.removeprefix(BASE + '/')] = state.get('counts', {})
    manifest_ref = config['native_retry_manifest_ref']
    manifest = read_ref(manifest_ref)
    plan_ref = config['default_native_retry_plan_ref']
    registry = optional(config['native_retry_active_owner_path'])
    if registry is not None:
        need(registry['schema'] == 'native_argmix_retry_active_owner_v1' and registry['root_confirmed'] is True,
             'Reporting active-owner registry needs explicit root activation')
        active_manifest = read_ref(registry['active_manifest_ref'])
        need(active_manifest['original_retry_manifest_ref'] == manifest_ref, 'Registry retry lineage differs')
        plan_ref = registry['active_plan_ref']
    plan = read_ref(plan_ref)
    need(plan['retry_manifest_ref'] == manifest_ref, 'Reporting retry plan lineage differs')
    if registry is not None:
        need(plan['operational_continuation_ref'] == registry['active_manifest_ref'], 'Active plan and registry manifest differ')
    state_path = str(Path(plan['records_directory']) / 'state.json')
    if registry is not None:
        need(r.path(registry['state_path']) == r.path(state_path), 'Active state path differs from active plan')
    state = optional(state_path)
    if registry is not None:
        need(state is not None, 'Root-activated reporting owner must already have its bound state')
    history, owned_rows = {}, {}
    if state:
        effective, history, owned_rows = merge_retry(effective, successor, manifest, plan, state, plan_ref)
        overlays[state_path.removeprefix(BASE + '/')] = dict(Counter(row['status'] for row in owned_rows.values()))
    selected = None
    fit = owned_rows.get(manifest['retry_task_id'])
    if fit and fit['status'] == 'succeeded':
        need(successful(fit), 'Successful retry state requires real wait')
        selected = selected_metadata(fit['result_ref'])
    phase_rows = {path: optional(path) for path in config['phase_state_paths']}
    stages = {}
    for path in config['stage_wait_paths']:
        row = optional(path)
        if row is not None:
            stages[path] = {key: row[key] for key in ('actual_wait', 'actual_returncode', 'owned_process_group_absent', 'result_ref') if key in row}
    return dict(effective=effective, original=original, successor=successor, extra_tasks=extra, overlays=overlays,
                native_retry_owned_tasks=owned_rows, native_retry_plan_ref=plan_ref, native_retry_state_path=state_path,
                native_retry_manifest_ref=manifest_ref, native_retry_active_owner_registry=registry,
                native_retry_monitor_health=(state or {}).get('monitor_health'),
                historical_native_retry_rows=history, historical_failed_wait_ref=manifest['failed_actual_wait_ref'],
                selected_native_ARGMix=selected, phase_rows=phase_rows, stage_waits=stages,
                historical_old_tree_task_ids=config['historical_old_tree_task_ids'])


def fingerprint(value):
    """Ignore timestamps, utilization, byte counters and per-epoch progress."""
    task_keys = ('status', 'attempt', 'pid', 'result_ref', 'actual_returncode', 'error')
    phase_keys = ('status', 'reason', 'detail', 'missing_dependencies', 'operational_continuation_ref',
                  'atomic_publication_continuation_ref')
    compact = dict(tasks={name: {key: row[key] for key in task_keys if key in row} for name, row in value['effective'].items()},
                   phases={path: {key: row[key] for key in phase_keys if key in row} if row else None
                           for path, row in value['phase_rows'].items()},
                   stage_waits=value['stage_waits'], native_retry_plan_ref=value['native_retry_plan_ref'],
                   selected_native_ARGMix=value['selected_native_ARGMix'],
                   native_retry_monitor_health={key: item for key, item in (value.get('native_retry_monitor_health') or {}).items() if key in ('status', 'reason', 'degraded', 'GPU_probe_status', 'process_probe_status')})
    return hashlib.sha256(json.dumps(compact, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
