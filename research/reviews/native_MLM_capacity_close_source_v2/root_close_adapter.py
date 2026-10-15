"""Inert source-only adapters. Only ROOT may explicitly prepare/apply a close.

No payload access, scope inventory, credit arithmetic, deletion or guard import
at module import or default CLI entry. Unchanged credit-v3 owns all accounting.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
SOURCE = 'research/reviews/research/reviews/native_MLM_capacity_close_source_v2/root_close_adapter.py'
AUDIT_REF = {'path': 'research/reviews/native_MLM_capacity_close_source_v2/candidate_audit.json', 'sha256': 'cdb63add032796fc741f2ed17a05471af8fd1c724252615e4a719e42d40b5669'}
MAX_JSON = 65536
MAX_ROOT_OBSERVATION_AGE_SECONDS = 120
REQUEST_FIELDS = frozenset(('key','reservation_sha256','storage_ref','storage_locator',
    'launch_ref','launch_pid_locator','launch_identity','direct_wait_ref','wait_pid_locator',
    'wait_exit_locator','wait_identity','completed_ref','completed_identity','observation_ref'))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def encoded(value):
    # Exact unchanged v4 encoding; only used to bind the original object.
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def path_v1(name):
    require(isinstance(name, str) and not Path(name).is_absolute(), 'Repository-relative ref required')
    p = ROOT / name
    require(p.resolve() == p and p.is_relative_to(ROOT), 'No escaping or symlink ref')
    return p


def bound_v1(ref):
    require(isinstance(ref, dict) and set(ref) == {'path','sha256'}, 'Exact source/JSON ref')
    p = path_v1(ref['path'])
    require(p.suffix in ('.py','.sh','.json') and len(p.read_bytes()) <= 1048576,
            'Bounded source/JSON evidence only')
    raw = p.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'Bound evidence changed: '+ref['path'])
    return raw


def json_v1(ref):
    raw = bound_v1(ref)
    require(len(raw) <= MAX_JSON, 'Compact authority/evidence JSON required')
    return json.loads(raw)


def source_ref_v1():
    return dict(path=SOURCE, sha256=hashlib.sha256(path_v1(SOURCE).read_bytes()).hexdigest())


def audit_v1():
    audit = json_v1(AUDIT_REF)
    json_v1(audit['candidate_source_ref'])
    for ref in audit['source_refs']:
        bound_v1(ref)
    for k in ('primitive_ref','policy_ref','repair_authority_ref'):
        bound_v1(audit[k])
    policy = json_v1(audit['policy_ref'])
    require(policy['source_ref'] == audit['primitive_ref'] and policy['marker_repair_authority_ref'] == audit['repair_authority_ref'],
            'Unchanged active policy/repair/primitive binding required')
    for k in ('implementation_ref','launcher_ref','base_ref','repair_ref','capacity_v1_ref','marker_ref'):
        bound_v1(policy[k])
    return audit, policy


def original_evidence_v1(item):
    c = item['candidate']
    storage = json_v1(c['storage_ref'])
    for key in c['storage_locator']:
        storage = storage[key]
    require(storage['reservation'] == item['original_reservation'] and storage['initial_ledger_pins'] == item['initial_ledger_pins'],
            'Preserve exact original reservation and initial pins')
    require(hashlib.sha256(encoded(storage['reservation'])).hexdigest() == item['reservation_sha256'],
            'Original reservation digest drift')
    for k in ('launch_ref','direct_wait_ref','completed_ref'):
        raw = bound_v1(c[k])
        require(len(raw) <= (1048576 if k == 'completed_ref' else MAX_JSON), 'Original metadata bound')
    launch = json_v1(c['launch_ref']); wait = json_v1(c['direct_wait_ref'])
    for v, identity in ((launch,c['launch_identity']),(wait,c['wait_identity'])):
        require(all(k in v and encoded(v[k]) == encoded(x) for k,x in identity.items()), 'Original launch/wait identity drift')
    def pointer(value, keys):
        for key in keys:
            value = value[key]
        return value
    pid = pointer(launch,c['launch_pid_locator']); code = pointer(wait,c['wait_exit_locator'])
    require(pointer(wait,c['wait_pid_locator']) == pid and type(code) is int and code == 0,
            'Keep exact original successful owned-child status')
    bound_v1(item['wait_provenance']['writer_ref'])
    provenance = item['wait_provenance']
    require(provenance['explicit_flag_added'] is False, 'Never add an original wait flag')
    if provenance['status'] == 'original_explicit_owned_direct_child_status':
        require(provenance['writer_identity_proven'] is False
                and provenance['original_explicit_wait_flag'] in ('actual_direct_worker_wait','actual_direct_launcher_wait','actual_direct_wait_observed')
                and wait[provenance['original_explicit_wait_flag']] is True
                and c['wait_identity'][provenance['original_explicit_wait_flag']] is True,
                'Exact original explicit direct wait flag; no invented writer identity')
    else:
        require(provenance['status'] == 'source_backed_owned_direct_child_status',
                'Source-backed wait provenance required')
    if item['inner_direct_worker_wait_ref'] is not None:
        inner = json_v1(item['inner_direct_worker_wait_ref'])
        require(inner['launcher_pid'] == pid and inner['returncode'] == 0 and inner['wait_status_known'] is True and
                inner['reservation_sha256'] == item['reservation_sha256'], 'Bound inner actual worker wait drift')
    for row in item['historical_failed_launches_preserved']:
        require(json_v1(row['launch_ref'])['pid'] == row['pid'] and json_v1(row['wait_ref'])['returncode'] == row['returncode'] == 1,
                'Preserve prior genuine failures')
    for ref in item['revocation_target_refs']:
        bound_v1(ref)


def request_v1(item, observation_ref):
    c = item['candidate']
    value = {k: copy.deepcopy(c[k]) for k in REQUEST_FIELDS if k in c}
    value.update(reservation_sha256=item['reservation_sha256'], observation_ref=copy.deepcopy(observation_ref))
    require(set(value) == REQUEST_FIELDS, 'Exact unchanged credit-v3 proof fields required')
    return value


def fresh_stamp_v1(value, now):
    stamp = datetime.fromisoformat(value.replace('Z','+00:00'))
    require(stamp.tzinfo is not None and 0 <= (now-stamp).total_seconds() <= MAX_ROOT_OBSERVATION_AGE_SECONDS,
            'Root targeted absence must be fresh and nonfuture')


def observation_v1(item, observation, adapter_ref, now):
    c = item['candidate']
    launcher_pid = json_v1(c['launch_ref'])
    for field in c['launch_pid_locator']:
        launcher_pid = launcher_pid[field]
    expected = dict(schema='concurrent_completed_scope_targeted_root_observation_v1',root_authorizer='root_agent',active=True,
        key=c['key'],reservation_sha256=item['reservation_sha256'],storage_ref=c['storage_ref'],
        launch_ref=c['launch_ref'],direct_wait_ref=c['direct_wait_ref'],completed_ref=c['completed_ref'],
        launcher_pid=launcher_pid,all_root_owned_launcher_supervisor_worker_pids_listed=True,
        direct_wait_observed=True,completed_scope_all_old_original_resume_write_capabilities_revoked=True,
        preserve_all_files=True,external_NFS_writers_outside_observation=True,
        source_roster_audit_ref=AUDIT_REF,root_close_adapter_ref=adapter_ref,
        roster_reconstruction_complete=True,unresolved_roster_items=[],
        revoked_original_entrypoint_refs=item['revocation_target_refs'])
    require(all(observation.get(k) == v for k,v in expected.items()), 'Exact resolved root roster/revocation certificate required')
    pids = observation.get('pids'); groups = observation.get('process_group_ids')
    unrecorded_c3 = (c['key'] in (
        'H041/American-C3-observe-qualification100kb-20261015-v5',
        'H041/American-C3-qualification100kb-20261015')
        and item.get('historical_group_ids_unrecorded') is True
        and item['known_process_group_ids'] == [])
    if unrecorded_c3:
        require(observation.get('historical_group_ids_unrecorded') is True,
                'Exact C3 historical group IDs remain unrecorded; no invented PGID')
    for values, known, label in ((pids,item['known_pids'],'PID'),(groups,item['known_process_group_ids'],'group')):
        require(isinstance(values,list) and (0 if label == 'group' and unrecorded_c3 else 1) <= len(values) <= 256 and len(set(values)) == len(values) and
                all(type(x) is int and x >= 2 for x in values) and set(known) <= set(values),
                'Complete explicit known/recovered '+label+' roster required')
    for values, rows, key, method in ((pids,observation.get('observations'),'pid','targeted os.kill(pid,0) ESRCH'),
            (groups,observation.get('process_group_observations'),'pgid','targeted os.killpg(pgid,0) ESRCH')):
        require(isinstance(rows,list) and len(rows) == len(values), 'Root observation roster length')
        for number,row in zip(values,rows):
            require(set(row) == {key,'absent','method','checked_utc'} and row[key] == number and
                    row['absent'] is True and row['method'] == method, 'Honest targeted root ESRCH evidence required')
            fresh_stamp_v1(row['checked_utc'], now)
    refs = observation.get('roster_resolution_refs')
    require(isinstance(refs,list) and 1 <= len(refs) <= 16, 'Bound root roster reconstruction evidence required')
    return refs


def absent_groups_root_v1(groups):
    # ROOT-only targeted signal-0 check, no signalling or process enumeration.
    for pgid in groups:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            continue
        except PermissionError:
            raise ValueError('Cannot certify targeted owned group absence') from None
        raise ValueError('Owned group present or number reused; closure refused')


def load_primitive_root_v1(audit):
    for path in (ROOT,ROOT/'src',ROOT/'scripts'):
        if str(path) not in sys.path:
            sys.path.insert(0,str(path))
    bound_v1(audit['primitive_ref'])
    spec = importlib.util.spec_from_file_location('_root_unchanged_completed_credits_v3',path_v1(audit['primitive_ref']['path']))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.configure_v2(audit['repair_authority_ref'],policy_ref=audit['policy_ref'])
    return module


def write_fresh_v1(path, value):
    p = path_v1(path)
    require(p.parent == ROOT / Path(SOURCE).parent and p.suffix == '.json', 'Fresh compact receipt in owned source package required')
    raw = json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
    require(len(raw.encode()) <= MAX_JSON, 'Compact charged metadata only')
    with p.open('x') as stream:
        stream.write(raw)
    return dict(path=path,sha256=hashlib.sha256(raw.encode()).hexdigest())


def root_operation_v1(tag, observation_ref, *, output_path, authority_ref=None):
    output = path_v1(output_path)
    require(output.parent == ROOT / Path(SOURCE).parent and output.suffix == '.json' and not output.exists(),
            'Receipt/draft output must be a fresh owned-package JSON before any root action')
    audit, policy = audit_v1()
    require(tag in audit['candidates'], 'Explicit audited candidate required')
    item = audit['candidates'][tag]
    original_evidence_v1(item)
    observation = json_v1(observation_ref)
    refs = observation_v1(item,observation,source_ref_v1(),datetime.now(timezone.utc))
    for ref in refs:
        value = json_v1(ref)
        require(value.get('root_authorizer') == 'root_agent' and value.get('key') == item['candidate']['key'] and
                value.get('roster_reconstruction_complete') is True and value.get('unresolved_roster_items') == [],
                'Actual root full-roster resolution required')
        if item.get('historical_group_ids_unrecorded') is True:
            require(item['candidate']['key'] in (
                'H041/American-C3-observe-qualification100kb-20261015-v5',
                'H041/American-C3-qualification100kb-20261015')
                and value.get('historical_group_ids_unrecorded') is True
                and value.get('current_cooperative_writer_roster_complete') is True
                and type(value.get('actual_current_process_observation')) is dict
                and value['actual_current_process_observation'].get('matched_live_completed_scope_writers') == []
                and type(value['actual_current_process_observation'].get('project_processes')) is list
                and bool(value['actual_current_process_observation'].get('targeting_rules')),
                'C3 empty historical groups require actual complete current-writer certificate')
    request = request_v1(item,observation_ref)
    absent_groups_root_v1(observation['process_group_ids'])
    primitive = load_primitive_root_v1(audit)
    pins = dict(expected_source_sha256=policy['implementation_ref']['sha256'],expected_launcher_sha256=policy['launcher_ref']['sha256'])
    if authority_ref is None:
        view = primitive.inspect_v1(**pins)
        result = primitive.operation_draft_v1('close_completed',request,expected_before=primitive._pins(view),bindings=view['bindings'])
        return dict(status='inactive_close_operation_prepared',operation_ref=write_fresh_v1(output_path,result))
    authority = json_v1(authority_ref)
    require(authority.get('request') == request and authority.get('operation') == 'close_completed' and
            authority.get('active') is True and authority.get('root_authorizer') == 'root_agent',
            'Exact separately activated root operation required')
    # The unchanged primitive validates authority/expiry/current exact prefix under
    # original mutex, repeats PID ESRCH, inventories only original scopes, repeats
    # PID ESRCH, then appends its immutable close event. It owns all credit math.
    receipt = primitive.apply_close_completed_v1(authority_ref,**pins)
    # Absence after the append is diagnostic only; a failure must never replay.
    try:
        absent_groups_root_v1(observation['process_group_ids'])
        group_postcheck = 'ESRCH_for_every_bound_group'
    except ValueError as error:
        group_postcheck = str(error)
    return dict(status='closed_by_unchanged_primitive_do_not_replay',receipt_ref=write_fresh_v1(output_path,
        dict(receipt=receipt,post_append_group_check=group_postcheck)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare-root',action='store_true')
    mode.add_argument('--apply-root',action='store_true')
    parser.add_argument('--candidate')
    parser.add_argument('--observation')
    parser.add_argument('--expected-observation-sha256')
    parser.add_argument('--authority')
    parser.add_argument('--expected-authority-sha256')
    parser.add_argument('--output')
    args = parser.parse_args()
    if not (args.prepare_root or args.apply_root):
        print(json.dumps(dict(status='source_only_inactive',accounting_operations=0,process_checks=0,scope_inventory=False)))
        return 0
    require(args.candidate and args.observation and args.expected_observation_sha256 and args.output,
            'ROOT must supply exact candidate/fresh observation and fresh output path')
    require(bool(args.authority) == bool(args.expected_authority_sha256) and
            bool(args.apply_root) == bool(args.authority), 'Apply requires separately activated exact primitive authority')
    result = root_operation_v1(args.candidate,dict(path=args.observation,sha256=args.expected_observation_sha256),
        output_path=args.output,authority_ref=dict(path=args.authority,sha256=args.expected_authority_sha256) if args.apply_root else None)
    print(json.dumps(result,sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
