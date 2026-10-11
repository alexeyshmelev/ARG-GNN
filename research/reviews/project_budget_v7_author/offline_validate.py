"""Bounded source/schema checks: synthetic proc fixture only, no live apply."""
from pathlib import Path
import ast, copy, hashlib, json, os, sys, tempfile
from unittest.mock import patch
ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import project_budget_v7 as b
results = []
def passed(name):
    results.append({'case': name, 'result': 'PASS'})
def rejected(name, call):
    try:
        call()
    except (ValueError, PermissionError):
        passed(name)
    else:
        raise AssertionError(name)
for name in ['scripts/project_budget_v7.py', 'scripts/launch_reserved_gpu_v4.py']:
    ast.parse((ROOT / name).read_text())
passed('both source ASTs')
s = (ROOT / 'scripts/project_budget_v7.py').read_text()
assert 'all_new_consumers_use_v6_guard' not in s
assert 'v6_reserved_bytes' not in s
assert 'consumer_uses_v6_guard' not in (ROOT / 'scripts/launch_reserved_gpu_v4.py').read_text()
passed('V7 namespace consistency')
auth = {'kind': 'project_budget_v7_root_admission', 'root_authorizer': 'root_agent',
        'active': True, 'operation': 'reserve', 'all_budget_writers_use_scheduler_mutex': True,
        'all_new_consumers_use_v7_guard': True, 'v6_never_activated_root_attested': True,
        'legacy_and_v4_new_admissions_frozen': True, 'no_historical_record_mutation_authorized': True,
        'global_rebase_authorized': False, 'GPU_authorized': False,
        'scientific_execution_authorized': False, 'independent_review': 'review'}
review = {'kind': 'project_budget_v7_independent_review', 'review_role': 'independent_reviewer',
          'verdict': 'PASS', 'actual_apply_performed': False}
def authority_check(value):
    with patch.object(b.v4, 'bound', side_effect=lambda root, ref, required: review if ref == 'review' else value):
        return b.check_authority(ROOT, 'authority', 'reserve', {})
authority_check(auth)
passed('V7 guard and explicit V6 nonactivation accepted')
changed = dict(auth); changed.pop('v6_never_activated_root_attested')
rejected('missing V6 nonactivation rejected', lambda: authority_check(changed))
changed = dict(auth); changed.pop('all_new_consumers_use_v7_guard'); changed['all_new_consumers_use_v6_guard'] = True
rejected('stale V6 guard rejected', lambda: authority_check(changed))
policy_path = ROOT / 'research/reviews/project_budget_v7_H026_close_root/opaque_credential_daemons.json'
actual = json.loads(policy_path.read_text())
policy_ref = {'path': str(policy_path.relative_to(ROOT)), 'sha256': hashlib.sha256(policy_path.read_bytes()).hexdigest()}
def parse_policy(value):
    with patch.object(b.v4, 'bound', return_value=value), patch.object(b, 'credential_bindings'), patch.object(b.os, 'getuid', return_value=1000):
        return b.opaque_credential_policy(ROOT, policy_ref, [1531600,1531605])
accepted = parse_policy(actual)
assert len(accepted['entries']) == 2
passed('actual root record exact schema metadata only')
changed = copy.deepcopy(actual); changed['daemons'][0]['identity']['pid'] = 1531605
rejected('recorded scientific PID cannot be exempt', lambda: parse_policy(changed))
changed = copy.deepcopy(actual); changed['daemons'].append(copy.deepcopy(changed['daemons'][0]))
rejected('more than two opaque entries rejected', lambda: parse_policy(changed))
changed = copy.deepcopy(actual); changed['daemons'][0]['root_attested_no_scientific_scope_write_or_relaunch'] = False
rejected('no-delegation attestation required', lambda: parse_policy(changed))
os_scandir = os.scandir
with tempfile.TemporaryDirectory(prefix='offline-proc-', dir=Path(__file__).parent) as tmp:
    proc = Path(tmp).resolve()
    (proc / '101').mkdir()
    fields = ['S'] + ['0'] * 18 + ['200']
    (proc / '101' / 'stat').write_text('101 (ssh-agent) ' + ' '.join(fields))
    (proc / '101' / 'status').write_text('Uid:\t1000\t1000\t1000\t1000\n')
    (proc / 'stat').write_text('btime 1\n')
    entry = {'role': 'external_preexisting',
             'identity': {'pid': 101, 'start_ticks': 200, 'comm': 'ssh-agent', 'uid': 1000}}
    synthetic = {'entries': {101:entry}, 'recorded_worker_pids': {901},
                 'root': ROOT, 'reference': {'path': 'synthetic', 'sha256': '0'*64},
                 'project_start_unix': 100000}
    with patch.object(b.os, 'readlink', side_effect=PermissionError), patch.object(b.os, 'getuid', return_value=1000):
        receipt = b.no_live_writers([proc / 'closed'], [901], proc_root=proc, opaque_policy=synthetic)
        assert receipt['local_proc_inspected'] is False
        assert len(receipt['opaque_credential_daemons']) == 1
        assert receipt['opaque_credential_daemons'][0]['descriptor_and_maps_inspected'] is False
        passed('exact cwd denial yields transparent opaque receipt')
        rejected('unlisted own-user cwd denial remains closed', lambda: b.no_live_writers([proc / 'closed'], [901], proc_root=proc))
        changed = copy.deepcopy(synthetic); changed['entries'][101]['identity']['start_ticks'] = 201
        rejected('changed start identity rejected', lambda: b.no_live_writers([proc / 'closed'], [901], proc_root=proc, opaque_policy=changed))
        rejected('present terminal PID always rejected', lambda: b.no_live_writers([proc / 'closed'], [101], proc_root=proc, opaque_policy=synthetic))
    with patch.object(b.os, 'readlink', return_value='/outside'), patch.object(b.os, 'scandir', side_effect=lambda path: (_ for _ in ()).throw(PermissionError()) if Path(path).name == 'fd' else os_scandir(path)), patch.object(b.os, 'getuid', return_value=1000):
        rejected('descriptor denial is not an opaque exception', lambda: b.no_live_writers([proc / 'closed'], [901], proc_root=proc, opaque_policy=synthetic))
print(json.dumps({'kind': 'project_budget_v7_author_offline_validation', 'cases': results,
                  'actual_proc_inspected': False, 'actual_apply_performed': False,
                  'actual_environment_or_socket_inspected': False,
                  'GPU_used': False, 'global_scan_performed': False}, indent=2, sort_keys=True))
