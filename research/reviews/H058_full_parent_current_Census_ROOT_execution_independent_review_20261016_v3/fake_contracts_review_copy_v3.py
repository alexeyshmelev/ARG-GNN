"""Small source/API-only fixtures. Never import project/guard/GPU/scientific modules."""
import ast, copy, hashlib, json, types
from pathlib import Path
from datetime import datetime, timezone, timedelta
HERE = Path(__file__).parent
OLD = HERE.parent / 'H058_full_parent_ROOT_execution_source_20261016_v2'
RESULTS = []

def check(name, condition):
    if not condition:
        raise AssertionError(name)
    RESULTS.append(dict(name=name, status='PASS'))

def require(ok, message):
    if not ok:
        raise ValueError(message)

def tree(path):
    return ast.parse(path.read_text())

def node(t, name):
    return next((n for n in t.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name))

def extract(t, names, ns):
    exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(node(t, n)) for n in names], type_ignores=[])), '<source-only-fixture>', 'exec'), ns)

def norm(value):
    value = copy.deepcopy(value)
    for n in ast.walk(value):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            n.value = n.value.replace('H058_full_parent_current_Census_ROOT_execution_source_20261016_v3', 'H058_full_parent_ROOT_execution_source_20261016_v2')
        if isinstance(n, ast.ImportFrom) and n.module:
            n.module = n.module.replace('H058_full_parent_current_Census_ROOT_execution_source_20261016_v3', 'H058_full_parent_ROOT_execution_source_20261016_v2')
        if isinstance(n, ast.alias):
            n.name = n.name.replace('H058_full_parent_current_Census_ROOT_execution_source_20261016_v3', 'H058_full_parent_ROOT_execution_source_20261016_v2')
    return ast.dump(value, include_attributes=False)
CT = tree(HERE / 'common_v1.py')
DT = tree(HERE / 'direct_gpu0_v1.py')
RT = tree(HERE / 'root_driver_v1.py')
ODT = tree(OLD / 'direct_gpu0_v1.py')
ORT = tree(OLD / 'root_driver_v1.py')
constants = {}
for n in CT.body:
    if isinstance(n, ast.Assign):
        try:
            value = ast.literal_eval(n.value)
            for target in n.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = value
        except (ValueError, TypeError):
            pass
direct_constants = {}
for n in DT.body:
    if isinstance(n, ast.Assign):
        try:
            value = ast.literal_eval(n.value)
            for target in n.targets:
                if isinstance(target, ast.Name):
                    direct_constants[target.id] = value
        except (ValueError, TypeError):
            pass
clock = types.SimpleNamespace(time=lambda : 1234.0)
uid = 1000
gpu_uuid = constants['GPU0_UUID']

def controller(q, base):
    if getattr(base, 'controller_error', False):
        raise ValueError('fake pre-inventory controller failure')
    return ({}, {'updated_unix': 1200.0}, {'process_identities': [dict(pid=10, uid=uid, start_ticks=7)]}, {})

def make_ns():
    return dict(require=require, json=json, time=clock, os=types.SimpleNamespace(getuid=lambda : uid), controller_v1=controller, GPU0_UUID=gpu_uuid, CURRENT_CAP=direct_constants['CURRENT_CAP'])
old = make_ns()
new = make_ns()
extract(ODT, ['capacity_v1'], old)
extract(DT, ['GPUCapacityFailureV3', 'bounded_capacity_failure_v3', 'capacity_v1'], new)

def make_base(compute, used=1024, total=65536, wrong_uuid=False, controller_error=False):
    gpu = dict(index=0, uuid='wrong' if wrong_uuid else gpu_uuid, total_mib=total, used_mib=used, compute=compute)
    return types.SimpleNamespace(real_inventory=lambda : [copy.deepcopy(gpu)], proc_identity=lambda pid: dict(pid=pid, uid=uid, start_ticks=7, state='S'), controller_error=controller_error)
title = 'AI Auto Research Pipeline Test'
app = lambda pid, mib, name=title: dict(pid=pid, used_mib=mib, name=name)
q = lambda cap: dict(caps=dict(max_cuda_reserved_bytes=cap * 2 ** 30))
cases = [('success_empty_28', make_base([], used=1000), q(28), None, True), ('success_C3_H', make_base([app(10, 5000), app(20, 10000), app(30, 1500)], used=17500), q(20), 20, True), ('success_chosen_1', make_base([app(10, 26000)], used=26000, total=65536), q(1), 20, True), ('failure_C3_limit', make_base([app(10, 26625)], used=26625), q(20), 20, False), ('failure_H_limit', make_base([app(20, 22529)], used=22529), q(20), 20, False), ('failure_C3_title', make_base([app(10, 100, 'wrong')], used=100), q(20), 20, False), ('failure_H_title', make_base([app(20, 100, 'wrong')], used=100), q(20), 20, False), ('failure_external_capacity', make_base([app(30, 15000)], used=15000), q(28), 20, False), ('failure_free_capacity', make_base([], used=32000), q(28), 20, False)]
for (name, base, request, pid, expected) in cases:
    outcomes = []
    for ns in [old, new]:
        try:
            outcomes.append((True, ns['capacity_v1'](request, base, worker_pid=pid)))
        except ValueError as exc:
            outcomes.append((False, str(exc)))
    check(name + '_same_decision', outcomes[0][0] == outcomes[1][0] == expected)
    check(name + '_same_result_or_reason', outcomes[0][1] == outcomes[1][1])
    if not expected:
        try:
            new['capacity_v1'](request, base, worker_pid=pid)
        except new['GPUCapacityFailureV3'] as exc:
            sample = exc.capacity_sample
            check(name + '_bounded_sample', len(json.dumps(sample, sort_keys=True, ensure_ascii=True).encode()) <= 4096 and sample['capacity_checks_passed'] is False)
for (name, base) in [('pre_inventory_controller', make_base([], controller_error=True)), ('pre_UUID', make_base([], wrong_uuid=True))]:
    try:
        new['capacity_v1'](q(28), base)
    except ValueError as exc:
        check(name + '_no_fake_sample', not hasattr(exc, 'capacity_sample'))
    else:
        raise AssertionError(name)
huge = dict(index=0, uuid=gpu_uuid, total_mib=65536, used_mib=65536, compute=[app(i, 100, '界' * 400) for i in range(80)])
sample = new['bounded_capacity_failure_v3'](huge, 1234.0, dict(classification_complete=False), ValueError('界' * 1000))
check('unicode_sample_bound', len(json.dumps(sample, sort_keys=True, ensure_ascii=True, allow_nan=False).encode()) <= 4096)
check('sample_explicit_truncation', sample['compute_count'] == 80 and sample['compute_rows_truncated'] is True and all((r['name_truncated'] for r in sample['retained_compute'])))
check('sample_complete_inventory_decisions', sample['raw_capacity_decisions_used_complete_inventory'] is True)
newbody = copy.deepcopy(next((n for n in node(DT, 'capacity_v1').body if isinstance(n, ast.Try))).body)
newbody = [n for n in newbody if not (isinstance(n, ast.Assign) and any((isinstance(t, ast.Name) and t.id in ('observed_gpu', 'observed_unix') for t in n.targets)) or (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute) and isinstance(n.value.func.value, ast.Name) and (n.value.func.value.id == 'metrics')))]
check('capacity_decision_AST_unchanged', norm(ast.Module(body=newbody, type_ignores=[])) == norm(ast.Module(body=node(ODT, 'capacity_v1').body, type_ignores=[])))
execute = node(RT, 'execute_v1')
outer = next((n for n in execute.body if isinstance(n, ast.Try) and n.finalbody))
oldouter = next((n for n in node(ORT, 'execute_v1').body if isinstance(n, ast.Try) and n.finalbody))
check('original_owned_cleanup_finally_AST_identical', norm(ast.Module(body=outer.finalbody, type_ignores=[])) == norm(ast.Module(body=oldouter.finalbody, type_ignores=[])))
check('RAM_log_AST_identical', norm(node(RT, '_RAMLog')) == norm(node(ORT, '_RAMLog')))
for fail_emit in [False, True]:
    events = []

    def emit_print(*args, **kwargs):
        events.append('stdout_attempt')
        require(kwargs.get('flush') is True, 'actual flush')
        if fail_emit:
            raise OSError('fake stdout failure')
        check('emitter_payload_' + str(fail_emit), json.loads(args[0])['H058_capacity_failure_before_owned_stop'] == sample)
        events.append('stdout_flushed')
    ns = dict(require=require, json=json, print=emit_print, failure=None, failed_capacity_sample=None, pre_stop_capacity_emitted=False, capacity_emit_error=None, exc_to_raise=new['GPUCapacityFailureV3']('fake failure', sample), owned_stop=lambda : events.append('owned_stop'))
    extract(RT, ['emit_capacity_failure_v3'], ns)
    fake = ast.Try(body=[ast.Raise(exc=ast.Name(id='exc_to_raise', ctx=ast.Load()), cause=None)], handlers=copy.deepcopy(outer.handlers), orelse=[], finalbody=[ast.Expr(value=ast.Call(func=ast.Name(id='owned_stop', ctx=ast.Load()), args=[], keywords=[]))])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fake], type_ignores=[])), '<actual-handler-fake-cleanup>', 'exec'), ns)
    check('sample_retained_before_cleanup_' + str(fail_emit), ns['failed_capacity_sample'] == sample and events[-1] == 'owned_stop' and (events[0] == 'stdout_attempt'))
    check('emitter_failure_cleanup_' + str(fail_emit), bool(ns['capacity_emit_error']) == fail_emit and ns['pre_stop_capacity_emitted'] is (not fail_emit))
for consumer in [False, True]:
    calls = []
    v4 = types.SimpleNamespace(__name__='fake_original_v4')
    for name in ('canonical', 'file_bytes', 'mutex'):
        fn = lambda : None
        fn.__module__ = v4.__name__
        setattr(v4, name, fn)
    g = types.SimpleNamespace(budget=types.SimpleNamespace(v4=v4))
    g.configure_with_renewal_v1 = lambda *a, **kw: calls.append(('configure', a, kw))
    ns = dict(require=require, ROOT=Path('/fake/project'), sys=types.SimpleNamespace(path=[]), **{k: constants[k] for k in ['GUARD', 'POLICY', 'REPAIR', 'MEMO', 'NAMESPACE_MEMO', 'V4']})
    memo = types.SimpleNamespace(install_consumer_v1=lambda *a, **kw: calls.append(('memo', a, kw)) or 'fake_handle')
    ns['load'] = lambda r, name: g if r == constants['GUARD'] else memo
    ns['ref'] = lambda p: next((constants[k] for k in ['MEMO', 'V4'] if constants[k]['path'] == p))
    extract(CT, ['guard_v1'], ns)
    got = ns['guard_v1'](consumer=consumer)
    check('actual_Census_API_binding_' + str(consumer), got is g and calls[0] == ('configure', (constants['REPAIR'],), dict(policy_ref=constants['POLICY'])))
    check('original_mutation_or_consumer_memo_' + str(consumer), len(calls) == (2 if consumer else 1) and (not consumer or g._H058_readonly_memo_v1 == 'fake_handle'))
now = datetime.now(timezone.utc)
record = dict(path='fake/qualification.json', sha256='1' * 64)
manifest = dict(path='fake/manifest.json', sha256='2' * 64)
request = dict(native_noncoexistence_qualification_ref=record, job_id='H058-full-parent-train-seed101-v1')
qual = dict(schema='H058_ROOT_actual_native_noncoexistence_qualification_v3', root_authorizer='root_agent', status='qualified', adapter_manifest_ref=manifest, H058_job_id=request['job_id'], physical_gpu=0, latest_native_direct_worker_returncode=-15, latest_native_owned_processes_and_groups_drained=True, actual_native_outer_terminal_consumed=True, no_native_GPU0_attachment_alive=True, all_native_launch_resume_execution_routes_revoked_while_H058_attached=True, C3_controller_preserved_until_H058_drained=True, native_actual_wait_ref=dict(path='fake/wait', sha256='3' * 64), native_actual_outer_terminal_ref=dict(path='fake/outer', sha256='4' * 64), fresh_targeted_native_absence_ref=dict(path='fake/absence', sha256='5' * 64), created_at_utc=(now - timedelta(minutes=1)).isoformat(), expires_at_utc=(now + timedelta(hours=24)).isoformat(), observed_at_utc=(now - timedelta(seconds=1)).isoformat())
qn = dict(require=require, datetime=datetime, timezone=timezone, MANIFEST='fake/manifest', read=lambda r: current, ref=lambda p: manifest, raw=lambda r: None)
extract(CT, ['active', 'validate_native_noncoexistence_v3'], qn)
current = copy.deepcopy(qual)
check('qualification_valid', qn['validate_native_noncoexistence_v3'](request, fresh=True) == qual)
for (key, value) in [('latest_native_direct_worker_returncode', 0), ('latest_native_owned_processes_and_groups_drained', False), ('actual_native_outer_terminal_consumed', False), ('no_native_GPU0_attachment_alive', False), ('all_native_launch_resume_execution_routes_revoked_while_H058_attached', False), ('C3_controller_preserved_until_H058_drained', False), ('adapter_manifest_ref', None), ('expires_at_utc', (now - timedelta(seconds=1)).isoformat()), ('observed_at_utc', (now + timedelta(hours=1)).isoformat()), ('observed_at_utc', (now - timedelta(minutes=6)).isoformat())]:
    current = copy.deepcopy(qual)
    current[key] = value
    try:
        qn['validate_native_noncoexistence_v3'](request, fresh=True)
    except (ValueError, KeyError):
        check('qualification_reject_' + key + '_' + str(value)[:20], True)
    else:
        raise AssertionError(key)
current = copy.deepcopy(qual)
current['observed_at_utc'] = (now - timedelta(hours=12)).isoformat()
check('continuing_active_revocation_historical_observation_allowed', qn['validate_native_noncoexistence_v3'](request, fresh=False) == current)
changes = {'common_v1.py': {'guard_v1', 'validate_request_v1', 'validate_input_v1'}, 'direct_gpu0_v1.py': {'controller_v1', 'capacity_v1'}, 'root_driver_v1.py': {'admit_v1', 'execute_v1'}, 'context_factory_v1.py': {'build_context_v1'}}
unchanged = 0
for (filename, allowed) in changes.items():
    nt = tree(HERE / filename)
    ot = tree(OLD / filename)
    for oldnode in ot.body:
        if isinstance(oldnode, (ast.FunctionDef, ast.ClassDef)) and oldnode.name not in allowed:
            check('unchanged_' + filename + '_' + oldnode.name, norm(oldnode) == norm(node(nt, oldnode.name)))
            unchanged += 1
oldcontext = (OLD / 'context_factory_v1.py').read_text().replace('H058_full_parent_ROOT_execution_source_20261016_v2', 'H058_full_parent_current_Census_ROOT_execution_source_20261016_v3').replace('<original_H020v3_writer_with_H058_256MiB_local_cap>', '<original_Census_fit2_writer_with_H058_256MiB_local_cap>')
check('context_only_package_and_diagnostic_change', oldcontext == (HERE / 'context_factory_v1.py').read_text())
result = dict(schema='H058_current_Census_source_fake_contract_results_v3', status='PASS', checks=RESULTS, passed=len(RESULTS), unchanged_protected_function_count=unchanged, actual_guard_calls=False, actual_device_or_process_probes=False, scientific_imports_or_payload_reads=False, accounting_mutations_or_admission=False)
print(json.dumps(dict(status='PASS', checks=len(RESULTS), unchanged_functions=unchanged), sort_keys=True))
