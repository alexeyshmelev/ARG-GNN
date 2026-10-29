"""Seal bounded, independent evidence for a prospective European reserve credit."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import time

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
os.chdir(ROOT)
OUTPUT = Path('research/joint_evaluation_20261029/argmix/european_reservation_release_evidence_v1.json')
assert not OUTPUT.exists(), 'Evidence is write-once'
GIB = 2**30

def reference(path, expected=None):
    path = Path(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected is not None:
        assert digest == expected, str(path)
    return {'path': str(path), 'sha256': digest}

def read_ref(ref):
    assert reference(ref['path']) == ref
    return json.loads(Path(ref['path']).read_text())

config_ref = reference('research/joint_evaluation_20261029/queue/successor_v2_config.json', 'e3cf7982043993957774d8f6b7c4c1d72047a1f214de67782f564ff4885721ff')
ledger_ref = reference('research/joint_evaluation_20261029/queue/storage_ledger.json', '1f4f8558bd4d83f5b667189c8a1317cedfdff88e433ec716a28e6d3dc7e779ee')
basis_refs = [
    reference('research/efficiency_20261027/root/european_full_val_execution_release_v2.json', '521fecb097347d5e0fc09e250acdec4b5b3bd36ee0124f0be2ae6523077fd928'),
    reference('research/efficiency_20261027/execution/european_full_val_runner/RETAINED_SHORT_ATTEMPTS_v1.json', '131f1062bbba59d087b8306ea00c18598b598fd7def1f98840121306a2fd41eb'),
    reference('research/resume_no_timeouts_20261027/fullval/ROOT_EXECUTION_RELEASE.json', '3e824e1c635f6cd4b07e4cc829606a27a5069b885d64fd7e761a2c3ffbe486ff'),
    reference('research/resume_no_timeouts_20261027/fullval/RETAINED_ATTEMPTS.json', '1a06300aabbe07682d14ffea2b2e7e85e2e5c64167051346d7c00946fc5ef497'),
    reference('research/joint_evaluation_20261029/root/storage_writer_review_v1.json', '99f05be4eecfc71a9a55565b263ac3f1c44e69fbccc927210e160920c8d57810'),
]
config, ledger = read_ref(config_ref), read_ref(ledger_ref)
release, later_release = read_ref(basis_refs[0]), read_ref(basis_refs[2])
assert config['storage_ledger_ref'] == ledger_ref
allowance_name = 'Separate full European validation, entire cap'
allowances = [v for v in ledger['base_allowances'] if v['name'] == allowance_name]
assert len(allowances) == 1 and allowances[0]['reserved_bytes'] == 160 * GIB
assert ledger['ceiling_bytes'] == 1200000000000
full_future = release['reservation_bytes']
assert full_future == later_release['reservation_bytes'] == 128983236608
fit = release['other_reservation_breakdown']['corrected_European_fit_bytes']
small = release['other_reservation_breakdown']['efficiency_sources_and_completed_or_active_small_outputs_bytes']
assert fit == 2 * GIB and small == GIB // 2
later_residual = later_release['other_project_reserved_bytes'] - later_release['retained_attempts_reservation_in_other_bytes']
additional = later_residual - fit - small
assert additional == GIB

expected_waits = [
    ('european_independent_validation_200mb_v1', '216b762a15b956c3cda5ad3ed515d71a187dba53a428d035442d39ec0af78ac2'),
    ('european_independent_validation_200mb_v2_no_timeouts', 'e592b7df63080503a43a4492a754eae8f6a44deea87c6814004e523b3a413c84'),
    ('european_short_development_val_10mb_v1', '9042cbdecdb8970697cd2d01b8dfeb185c416ae93e538d8b2976e135044481dd'),
    ('european_short_development_val_10mb_v2', '0d4c74684d55187c5b0282f785e2916df81d232f94579c438bd2c32d5a110f9a'),
]
terminal_runs = []
tracked_pids, tracked_groups, tracked_sessions = set(), set(), set()
for name, digest in expected_waits:
    scope = Path('research/efficiency_20261027/root/runs') / name
    assert scope.is_dir() and not scope.is_symlink()
    wait_ref = reference(scope / 'actual_wait.json', digest)
    wait = read_ref(wait_ref)
    request_ref, launch_ref = reference(scope / 'request.json'), reference(scope / 'launch.json')
    request, launch = read_ref(request_ref), read_ref(launch_ref)
    assert request['aggregate_directory'] == str(scope)
    assert request['command'] == launch['actual_argv']
    assert wait['actual_wait'] is True and wait['actual_returncode'] == 1
    assert wait['owned_family_absent'] is True and wait['owned_group_absent'] is True
    assert wait['pid'] == launch['pid'] and wait['process_start_ticks'] == launch['process_start_ticks']
    worker = {'pid': wait['pid'], 'start_ticks': wait['process_start_ticks']}
    supervisor = {'pid': request['supervisor_pid'], 'start_ticks': request['supervisor_start_ticks']}
    tracked_pids.update([worker['pid'], supervisor['pid']])
    for member in wait['tracked_family']:
        tracked_pids.add(member['pid'])
        tracked_groups.add(member['pgrp'])
        tracked_sessions.add(member['session'])
    assert any(v['pid'] == worker['pid'] and str(v['start']) == worker['start_ticks'] and v['ppid'] == supervisor['pid'] for v in wait['tracked_family'])
    terminal_runs.append({'scope': str(scope), 'actual_wait_ref': wait_ref, 'request_ref': request_ref, 'launch_ref': launch_ref,
        'worker_identity': worker, 'supervisor_identity': supervisor, 'tracked_family': wait['tracked_family'],
        'actual_returncode': 1, 'actual_wait': True, 'owned_family_absent': True, 'owned_group_absent': True,
        'finished_unix': wait['finished_unix']})

def absence_snapshot():
    matches = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / 'stat').read_text()
        except (FileNotFoundError, ProcessLookupError):
            continue
        fields = raw[raw.rfind(')') + 2:].split()
        identity = {'pid': int(entry.name), 'state': fields[0], 'ppid': int(fields[1]), 'pgrp': int(fields[2]), 'session': int(fields[3]), 'start_ticks': fields[19]}
        if identity['pid'] in tracked_pids or identity['pgrp'] in tracked_groups or identity['session'] in tracked_sessions:
            matches.append(identity)
    assert not matches, matches
    return {'observed_unix': time.time(), 'tracked_pids': sorted(tracked_pids), 'tracked_pgroups': sorted(tracked_groups), 'tracked_sessions': sorted(tracked_sessions), 'matching_processes': matches, 'all_owned_identities_absent': True}

def sizes(scope):
    allocated = int(subprocess.check_output(['du', '-sk', '--', scope], text=True).split()[0]) * 1024
    logical = int(subprocess.check_output(['du', '-sb', '--', scope], text=True).split()[0])
    return allocated, logical

before = absence_snapshot()
for row in terminal_runs:
    row['allocated_bytes'], row['logical_bytes'] = sizes(row['scope'])
for row in terminal_runs:
    assert sizes(row['scope']) == (row['allocated_bytes'], row['logical_bytes']), row['scope']
after = absence_snapshot()
retained = sum(row['allocated_bytes'] for row in terminal_runs)
assert retained == 3672723456
required = full_future + retained + fit + small + additional
assert required == 136414056448 and required < 128 * GIB
for ref in [config_ref, ledger_ref, *basis_refs]:
    read_ref(ref)
for row in terminal_runs:
    for key in ['actual_wait_ref', 'request_ref', 'launch_ref']:
        read_ref(row[key])
evidence = {
    'schema': 'european_reservation_release_evidence_v1',
    'created_utc': datetime.now(timezone.utc).isoformat(), 'created_unix': time.time(),
    'author': '/root/american_finish_oct29', 'independent_of_consumer_implementation': True,
    'root_authorized': True,
    'root_authorization': {'source': 'Parent /root direct task instruction preserved in conversation', 'scope': 'Prepare immutable independent reservation-release evidence and coordinate a prospective credit in additive revised support, scheduler and coordinator; no live source changes, launch, deletion, or ceiling increase.'},
    'original_successor_config_ref': config_ref, 'original_storage_ledger_ref': ledger_ref,
    'allowance_name': allowance_name, 'prior_allowance_bytes': 160 * GIB, 'revised_allowance_bytes': 128 * GIB, 'credit_bytes': 32 * GIB,
    'full_future_attempt_bytes': full_future, 'fit_allowance_bytes': fit, 'small_work_allowance_bytes': small,
    'additional_other_work_allowance_bytes': additional,
    'additional_other_work_rationale': 'Preserve the later no-timeouts release residual allowance of 3.5 GiB after retained-attempt reservation, including the 1 GiB beyond the original 2 GiB fit and 0.5 GiB small-work allowances.',
    'terminal_runs': terminal_runs, 'retained_allocated_bytes': retained,
    'retained_logical_bytes': sum(row['logical_bytes'] for row in terminal_runs),
    'required_revised_allowance_bytes': required, 'revised_allowance_headroom_bytes': 128 * GIB - required,
    'reservation_basis_refs': basis_refs, 'process_absence_checks': [before, after],
    'measurement_method': {'allocated': 'Bounded du -sk per exact scope, multiplied by 1024; repeated and required unchanged.', 'logical': 'Bounded du -sb per exact scope; apparent bytes including directory metadata; repeated and required unchanged.', 'processes': 'Read-only /proc stat inventory before and after measurements; reject any historical tracked PID, process group or session member.', 'scientific_payloads_read': False},
    'consumer_requirements': ['Validate this immutable evidence hash and every original binding.', 'Apply exactly one prospective 32 GiB credit to the named existing 160 GiB allowance.', 'Keep the original ledger, original successor configuration, task ownership and scientific sources unchanged.', 'Recheck all terminal wait bindings, process absence and current bounded allocated sizes; fail closed if any sealed scope grows or any owned process reappears.', 'Keep the full prospective attempt reservation and all auxiliary allowances charged; this record does not authorize that future launch.'],
    'global_project_ceiling_bytes': 1200000000000, 'ceiling_unchanged': True,
    'historical_outputs_preserved': True, 'no_deletion': True, 'no_process_signals': True, 'no_scientific_success_claim': True,
    'no_fresh_whole_project_inventory_claim': True,
    'evidence_source_ref': reference(Path(__file__).relative_to(ROOT) if Path(__file__).is_absolute() else Path(__file__)),
}
with OUTPUT.open('x') as stream:
    json.dump(evidence, stream, indent=2, sort_keys=True)
    stream.write('\n')
OUTPUT.chmod(0o444)
print(json.dumps({'evidence_ref': reference(OUTPUT), 'retained_allocated_bytes': retained, 'required_revised_allowance_bytes': required, 'revised_allowance_headroom_bytes': 128 * GIB - required, 'terminal_scopes': [{k: row[k] for k in ['scope', 'allocated_bytes', 'logical_bytes']} for row in terminal_runs]}, indent=2))
