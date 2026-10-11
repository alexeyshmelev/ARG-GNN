"""One synthetic six-daemon fixture plus narrow V8 schema capacity cases."""
from pathlib import Path
import ast, copy, hashlib, json, os, sys, tempfile
from unittest.mock import patch
ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import project_budget_v8 as b
for name in ['scripts/project_budget_v8.py', 'scripts/launch_reserved_gpu_v5.py']:
    ast.parse((ROOT / name).read_text())
inventory_path = ROOT / 'research/reviews/project_budget_v7_H026_close_root/remaining_opaque_process_inventory.json'
inventory = json.loads(inventory_path.read_text())
assert len(inventory['records']) == 6
old_path = ROOT / 'research/reviews/project_budget_v7_H026_close_root/opaque_credential_daemons.json'
fixture = json.loads(old_path.read_text())
fixture['kind'] = 'project_budget_v8_opaque_credential_daemons'
fixture['observation_scope'] = 'cooperative_accounting_bounded_opaque_credential_interfaces_not_global_proof'
project = next(e for e in fixture['daemons'] if e['role'] == 'project_credential_only')
fixture['daemons'] = []
for record in inventory['records']:
    identity = {k:record[k] for k in ['pid','start_ticks','comm','uid']}
    if identity['pid'] == project['identity']['pid']:
        assert identity == project['identity']
        fixture['daemons'].append(project)
    else:
        fixture['daemons'].append({'role':'external_preexisting','identity':identity,
            'root_attested_nondumpable':True,
            'root_attested_no_scientific_scope_write_or_relaunch':True,
            'root_attested_external_ownership':True,'root_attested_start_before_project':True})
ref = {'path':'synthetic_six_daemon_fixture','sha256':'0'*64}
results = []
def parse(value):
    with patch.object(b.v4, 'bound', return_value=value), patch.object(b, 'credential_bindings'), patch.object(b.os, 'getuid', return_value=1000):
        return b.opaque_credential_policy(ROOT, ref, [1531600,1531605])
policy = parse(fixture)
assert len(policy['entries']) == 6
with tempfile.TemporaryDirectory(prefix='six-daemon-fixture-', dir=Path(__file__).parent) as tmp:
    proc = Path(tmp).resolve()
    (proc / 'stat').write_text('btime 1\n')
    for entry in fixture['daemons']:
        identity = entry['identity']; p = proc / str(identity['pid']); p.mkdir()
        fields = ['S'] + ['0'] * 18 + [str(identity['start_ticks'])]
        (p / 'stat').write_text(str(identity['pid'])+' (ssh-agent) '+' '.join(fields))
        (p / 'status').write_text('Uid:\t1000\t1000\t1000\t1000\n')
    with patch.object(b.os, 'readlink', side_effect=PermissionError), patch.object(b, 'credential_bindings'), patch.object(b.os, 'getuid', return_value=1000):
        result = b.no_live_writers([proc / 'closed'], [1531600,1531605], proc_root=proc, opaque_policy=policy)
        assert result['local_proc_inspected'] is False
        assert len(result['opaque_credential_daemons']) == 6
        assert sum(e['role']=='external_preexisting' for e in result['opaque_credential_daemons']) == 5
        assert sum(e['role']=='project_credential_only' for e in result['opaque_credential_daemons']) == 1
        assert all(e['descriptor_and_maps_inspected'] is False and e['credential_contents_read'] is False for e in result['opaque_credential_daemons'])
results.append({'case':'six exact identities, five external and one project, transparent cwd-only skip','result':'PASS'})
def rejected(name, value):
    try: parse(value)
    except ValueError: results.append({'case':name,'result':'PASS'})
    else: raise AssertionError(name)
changed = copy.deepcopy(fixture); changed['daemons'].append(copy.deepcopy(project))
rejected('second project credential role rejected', changed)
changed = copy.deepcopy(fixture); changed['daemons'].append(copy.deepcopy(changed['daemons'][0]))
rejected('duplicate external PID rejected', changed)
changed = copy.deepcopy(fixture)
while len(changed['daemons']) < 17:
    entry = copy.deepcopy(changed['daemons'][0]); entry['identity']['pid'] = 9000000+len(changed['daemons']); changed['daemons'].append(entry)
rejected('seventeenth identity rejected', changed)
core = (ROOT / 'scripts/project_budget_v8.py').read_text()
launcher = (ROOT / 'scripts/launch_reserved_gpu_v5.py').read_text()
assert 'all_new_consumers_use_v8_guard' in core and 'consumer_uses_v8_guard' in launcher
assert 'v6_never_activated_root_attested' in core and 'v7_never_activated_root_attested' in core
assert '("project_budget_v6", "project_budget_v7")' in core
assert 'v8_reserved_bytes' in core and 'launcher_exit_v5.json' in core and 'reserved_gpu_v5_supervisor_wait' in core
print(json.dumps({'kind':'project_budget_v8_author_offline_validation','cases':results,
    'source_ASTs_and_V8_namespace_checked':True,
    'root_public_inventory_metadata_sha256':hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
    'actual_proc_inspected':False,'actual_environment_or_socket_inspected':False,
    'actual_apply_performed':False,'GPU_used':False,'inherited_matrix_rerun':False},indent=2,sort_keys=True))
