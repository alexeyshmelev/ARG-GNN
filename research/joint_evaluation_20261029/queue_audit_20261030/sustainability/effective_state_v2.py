"""Owner-aware reporting adapter with stable blocker and monitor causes."""
from collections import Counter
import copy
import hashlib
import json
from research.joint_evaluation_20261029.root.reporting_20261030_v1 import effective_state_v1 as prior
from research.joint_evaluation_20261029.queue_audit_20261030.sustainability import producer_owner_v1 as producer


def merge_producer(value, owner):
    result = dict(value)
    result['effective'] = dict(value['effective'])
    owned = owner['owned_rows'] if owner else {}
    producer.need(set(owned) <= set(value['effective']), 'Producer overlay cannot add campaign identities')
    producer.need(not set(owned).intersection(value['native_retry_owned_tasks']), 'Producer and native consumer owners overlap')
    result['original_main_producer_rows'] = {name: copy.deepcopy(value['successor']['tasks'][name]) for name in owned}
    result['effective'].update(owned)
    result['producer_owned_tasks'] = owned
    result['producer_active_registry'] = owner['registry'] if owner else None
    result['producer_monitor_health'] = owner['state'].get('monitor_health') if owner else None
    result['overlays'] = dict(value['overlays'])
    if owner:
        result['overlays'][owner['registry']['state_path']] = dict(Counter(row['status'] for row in owned.values()))
    return result


def snapshot(config_path=prior.CONFIG):
    value = prior.snapshot(config_path)
    owner = producer.load_active_owner()
    if owner:
        producer.need(value['successor']['plan_ref'] == owner['registry']['original_plan_ref'], 'Reporting original main producer identity differs')
        for name, row in owner['owned_rows'].items():
            if row['status'] == 'succeeded':
                producer.validated_wait(owner, name)
    return merge_producer(value, owner)


def stable_monitor(value):
    value = value or {}
    result = {key: value[key] for key in ('status', 'reason', 'degraded', 'GPU_probe_status', 'process_probe_status') if key in value}
    unavailable = value.get('unavailable', {})
    result['unavailable'] = {name: {key: row[key] for key in ('reason', 'error', 'error_type') if key in row}
                             if isinstance(row, dict) else row for name, row in unavailable.items()}
    return result


def observability(value):
    return dict(waiting={name: row.get('waiting_reason') for name, row in value['effective'].items() if row['status'] == 'pending'},
                native_monitor=stable_monitor(value.get('native_retry_monitor_health')),
                producer_monitor=stable_monitor(value.get('producer_monitor_health')))


def fingerprint(value):
    registry = value.get('producer_active_registry') or {}
    compact = dict(stage_fingerprint=prior.fingerprint(value), observability=observability(value),
                   producer_plan_ref=registry.get('producer_plan_ref'), producer_manifest_ref=registry.get('producer_manifest_ref'))
    return hashlib.sha256(json.dumps(compact, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
