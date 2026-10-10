"""CPU-only tests: handmade fixtures, case coverage, guards, and exact replay."""
from dataclasses import replace
import fcntl
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('synthetic_cuda_admission', ROOT/'scripts/check_development_cuda_admission.py')
smoke = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = smoke
SPEC.loader.exec_module(smoke)


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    torch.set_num_threads(1)
    smoke.configure_deterministic_development()
    yield
    assert not torch.cuda.is_initialized()


@pytest.fixture(scope='module')
def small_fixture():
    return smoke.prepare_fixture(4, 4, 3)


def test_closed_production_case_matrix():
    cases = smoke.admission_cases()
    assert len(cases) == 44
    assert len({case.identifier for case in cases}) == 44
    candidates = [case for case in cases if case.family == 'candidate']
    assert {(case.geometry, case.kernel_depth, case.hidden_dim, case.dropout) for case in candidates} == {
        (geometry, depth, 32, 0.) for geometry in ('plain', 'age') for depth in (1, 2)}
    for family in smoke.NEURAL_FAMILIES[1:]:
        assert {(case.hidden_dim, case.dropout) for case in cases if case.family == family} == {
            (width, dropout) for width in (32, 64) for dropout in (0., .1)}
    assert {(case.bandwidth_bp, case.pseudocount) for case in cases if case.family == 'reference_smoother'} == {
        (bandwidth, pseudocount) for bandwidth in (10000., 30000., 100000., 300000.) for pseudocount in (.001, .01)}


def test_default_handmade_fixture_has_observed_roles_and_changing_trees():
    sequence = smoke.handmade_observed_sequence()
    sequence.validate()
    assert sequence.x.shape == (3*175, 8)
    assert sequence.num_queries == 64
    assert sequence.num_graphs == 3
    assert torch.equal(sequence.intervals, torch.tensor([[0., .31], [.31, .72], [.72, 1.]], dtype=torch.float64))
    assert set(sequence.target_graph.tolist()) == {0, 1, 2}
    topologies = []
    for graph in range(3):
        nodes = (sequence.node_graph == graph).nonzero().flatten()
        ids = sequence.node_ids[nodes]
        query = torch.isin(ids, sequence.query_node_ids)
        reference = (ids >= 0) & ~query
        assert query.sum() == 64 and reference.sum() == 24
        assert torch.equal(sequence.x[nodes[query], :4], torch.zeros(64, 4, dtype=torch.float64))
        assert torch.equal(sequence.x[nodes[reference], :4].sum(0), torch.full((4,), 6., dtype=torch.float64))
        # Clades contain only descendant observed reference presence indicators.
        assert torch.equal(sequence.clades[nodes[query]], torch.zeros(64, 4, dtype=torch.float64))
        edge = sequence.edge_index[:, sequence.node_graph[sequence.edge_index[0]] == graph] - int(nodes.min())
        assert edge.shape == (2, 348)
        topologies.append(smoke.tensor_digest(edge))
    assert len(set(topologies)) == 3
    assert torch.equal(sequence.query_node_ids, torch.tensor([1000+11*i for i in reversed(range(64))]))
    assert not hasattr(sequence, 'targets')


def test_prepared_f47_f48_and_plain_age_spectra(small_fixture):
    fixture = small_fixture
    assert fixture.raw.x.dtype == torch.float32
    assert fixture.raw.x.shape[-1] == 48
    assert fixture.boundary.leaf_x.shape == (3, 8, 47)
    assert fixture.boundary.feature_columns == tuple(i for i in range(48) if i != 7)
    assert fixture.anchors.num_scales == 3
    assert set(fixture.anchors.operators) == {'plain', 'age', 'matched_scalar_age'}
    assert not torch.equal(fixture.anchors.operators['age'].cqq[:, 1], fixture.anchors.operators['plain'].cqq[:, 1])
    assert fixture.manifest['age_spectral_reconstruction_relative_max'] < 1e-12
    assert fixture.manifest['canonical_holds'] == 3
    assert fixture.targets.shape == (8, 4)
    assert fixture.targets.dtype == torch.long
    assert not hasattr(fixture.raw, 'targets')
    assert not hasattr(fixture.boundary, 'targets')
    assert not hasattr(fixture.anchors, 'targets')


@pytest.mark.parametrize('queries,references', [(1, 4), (4, 3), (True, 4), (4, 0)])
def test_fixture_rejects_invalid_shape(queries, references):
    with pytest.raises(ValueError):
        smoke.handmade_observed_sequence(queries, references)


REPLAY_CASES = [smoke.Case('candidate', hidden_dim=8, geometry='plain', kernel_depth=1),
                smoke.Case('candidate', hidden_dim=8, geometry='age', kernel_depth=2)]
REPLAY_CASES += [smoke.Case(family, hidden_dim=8, dropout=.1) for family in smoke.NEURAL_FAMILIES[1:]]
REPLAY_CASES += [smoke.Case('reference_smoother', bandwidth_bp=30000., pseudocount=.001)]


@pytest.mark.parametrize('case', REPLAY_CASES, ids=lambda case: case.identifier)
def test_small_all_family_cpu_backward_and_exact_checkpoint_replay(case, small_fixture, tmp_path):
    result = smoke.run_case(case, small_fixture, 'cpu', tmp_path/case.identifier)
    assert result['status'] == 'success', result.get('error')
    assert result['optimizer_steps'] == 0
    assert result['replay']['logits_bitwise_equal']
    assert result['replay']['raw_nll_exact_equal']
    assert result['replay']['reload_max_absolute_logit_delta'] == 0
    assert result['replay']['repeat_max_absolute_logit_delta'] == 0
    assert (tmp_path/case.identifier/'state.pt').is_file()
    assert (tmp_path/case.identifier/'eval_logits.pt').is_file()
    persisted = json.loads((tmp_path/case.identifier/'result.json').read_text())
    assert persisted['status'] == 'success'
    if case.family == 'reference_smoother':
        assert result['parameters'] == 0 and result['backward']['applicable'] is False
    else:
        assert result['backward']['all_required_groups_active']
        assert result['observed_feature_gradient_absolute_max'] > 0
        expected = ['blocks.0.gate.0', 'blocks.1.gate.0'] if case.family == 'gst_ca_span' else []
        assert result['backward']['expected_initial_zero_groups'] == expected


def test_failure_is_persisted_without_erasing_started_receipt(monkeypatch, small_fixture, tmp_path):
    monkeypatch.setattr(smoke, 'make_case_model', lambda *_: (_ for _ in ()).throw(RuntimeError('synthetic injected failure')))
    directory = tmp_path/'failure'
    result = smoke.run_case(smoke.Case('candidate'), small_fixture, 'cpu', directory)
    assert result['status'] == 'failure'
    assert result['stage'] == 'construct'
    assert result['error']['message'] == 'synthetic injected failure'
    assert json.loads((directory/'started.json').read_text())['stage'] == 'source_guard'
    assert json.loads((directory/'result.json').read_text())['error']['message'] == 'synthetic injected failure'
    before = (directory/'result.json').read_bytes()
    with pytest.raises(FileExistsError):
        smoke.run_case(smoke.Case('candidate'), small_fixture, 'cpu', directory)
    assert (directory/'result.json').read_bytes() == before


def test_missing_parameter_gradients_are_rejected():
    with pytest.raises(RuntimeError, match='Missing or nonfinite gradient'):
        smoke.gradient_evidence(torch.nn.Linear(2, 2), 'candidate')


def test_nonfinite_and_inactive_parameter_groups_are_rejected():
    model = torch.nn.Linear(2, 2)
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    with pytest.raises(RuntimeError, match='Structurally inactive'):
        smoke.gradient_evidence(model, 'candidate')
    model.weight.grad[0, 0] = float('nan')
    with pytest.raises(RuntimeError, match='nonfinite'):
        smoke.gradient_evidence(model, 'candidate')


@pytest.fixture
def leased_env(monkeypatch, tmp_path):
    root = tmp_path/'project'; leases = root/'.runtime/locks'; leases.mkdir(parents=True)
    path = leases/'gpu_3.lock'
    handle = path.open('w+')
    json.dump({'gpu': 3, 'uuid': 'GPU-synthetic-test', 'job': 'synthetic-admission', 'launcher_pid': 123}, handle)
    handle.flush()
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    monkeypatch.setattr(smoke, 'ROOT', root)
    monkeypatch.setattr(smoke, 'project_path', lambda value: Path(value).resolve())
    env = {'ARGGNN_PHYSICAL_GPU': '3', 'ARGGNN_GPU_LEASE': str(path), 'ARGGNN_GPU_LEASE_FD': str(handle.fileno()),
           'CUDA_VISIBLE_DEVICES': 'GPU-synthetic-test', 'ARGGNN_JOB_NAME': 'synthetic-admission'}
    yield env, handle
    handle.close()


def test_inherited_single_device_lease_passes_without_cuda_probe(leased_env):
    env, _ = leased_env
    result = smoke.validate_launcher_lease(env)
    assert result['physical_gpu'] == 3
    assert result['visible_gpu_uuid'] == 'GPU-synthetic-test'


@pytest.mark.parametrize('key,value,match', [
    ('ARGGNN_PHYSICAL_GPU', '0', 'Use the project GPU launcher'),
    ('ARGGNN_PHYSICAL_GPU', '6', 'Use the project GPU launcher'),
    ('CUDA_VISIBLE_DEVICES', 'GPU-synthetic-test,GPU-second', 'single GPU'),
    ('CUDA_VISIBLE_DEVICES', '3', 'single GPU'),
    ('ARGGNN_JOB_NAME', 'different-job', 'single GPU'),
    ('ARGGNN_GPU_LEASE_FD', '-1', 'inherited launcher lease'),
])
def test_bad_launcher_env_rejected(leased_env, key, value, match):
    env, _ = leased_env
    with pytest.raises(RuntimeError, match=match):
        smoke.validate_launcher_lease(env | {key: value})


def test_unheld_lease_rejected(leased_env):
    env, handle = leased_env
    fcntl.flock(handle, fcntl.LOCK_UN)
    with pytest.raises(RuntimeError, match='not held'):
        smoke.validate_launcher_lease(env)


def test_wrong_inherited_descriptor_rejected(leased_env, tmp_path):
    env, _ = leased_env
    with (tmp_path/'different').open('w+') as other:
        with pytest.raises(RuntimeError, match='descriptor/path mismatch'):
            smoke.validate_launcher_lease(env | {'ARGGNN_GPU_LEASE_FD': str(other.fileno())})


def patch_execution_probe(monkeypatch, events, count=1):
    import setproctitle
    monkeypatch.setattr(smoke, 'set_research_title', lambda: events.append('title'))
    monkeypatch.setattr(setproctitle, 'getproctitle', lambda: smoke.PROCESS_TITLE)
    monkeypatch.setattr(smoke, 'configure_deterministic_development', lambda: events.append('policy') or {'strict': True})
    monkeypatch.setattr(smoke, 'validate_launcher_lease', lambda: events.append('lease') or {'visible_gpu_uuid': 'GPU-synthetic-test'})
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: events.append('count') or count)
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda *_: events.append('properties') or SimpleNamespace(
        name='synthetic CUDA mock', total_memory=1024, major=8, minor=0, uuid='GPU-synthetic-test'))
    original_run = smoke.subprocess.run
    def subprocess_run(command, *args, **kwargs):
        if command[0] == 'nvidia-smi':
            events.append('driver'); return SimpleNamespace(stdout='synthetic-driver')
        return original_run(command, *args, **kwargs)
    monkeypatch.setattr(smoke.subprocess, 'run', subprocess_run)


def test_title_policy_lease_precede_every_device_inspection(monkeypatch):
    events = []
    patch_execution_probe(monkeypatch, events)
    runtime = smoke.initialize_execution('cuda')
    assert events == ['title', 'policy', 'lease', 'count', 'properties', 'driver']
    assert runtime['process_title'] == smoke.PROCESS_TITLE


def test_policy_rejection_stops_before_lease_or_device(monkeypatch):
    events = []
    patch_execution_probe(monkeypatch, events)
    monkeypatch.setattr(smoke, 'configure_deterministic_development', lambda: (_ for _ in ()).throw(RuntimeError('policy rejection')))
    with pytest.raises(RuntimeError, match='policy rejection'):
        smoke.initialize_execution('cuda')
    assert events == ['title']


def test_lease_rejection_stops_before_device_inspection(monkeypatch):
    events = []
    patch_execution_probe(monkeypatch, events)
    monkeypatch.setattr(smoke, 'validate_launcher_lease', lambda: (_ for _ in ()).throw(RuntimeError('lease rejection')))
    with pytest.raises(RuntimeError, match='lease rejection'):
        smoke.initialize_execution('cuda')
    assert events == ['title', 'policy']


def test_multiple_logical_devices_rejected_before_context(monkeypatch):
    events = []
    patch_execution_probe(monkeypatch, events, count=2)
    with pytest.raises(RuntimeError, match='exactly one reserved'):
        smoke.initialize_execution('cuda')
    assert events == ['title', 'policy', 'lease', 'count']


def test_cpu_initialization_requires_explicit_invisibility_and_never_probes_device(monkeypatch):
    events = []
    patch_execution_probe(monkeypatch, events)
    runtime = smoke.initialize_execution('cpu')
    assert runtime['device_kind'] == 'cpu'
    assert events == ['title', 'policy']
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-synthetic-test')
    with pytest.raises(RuntimeError, match='explicitly empty'):
        smoke.initialize_execution('cpu')
    assert events == ['title', 'policy', 'title', 'policy']


def test_source_drift_and_fresh_json_rejection(monkeypatch, tmp_path):
    path = tmp_path/'owned.py'; path.write_text('before\n')
    pins = {'owned.py': smoke.sha(path)}
    monkeypatch.setattr(smoke, 'ROOT', tmp_path)
    smoke.verify_sources(pins)
    path.write_text('after\n')
    with pytest.raises(RuntimeError, match='Source drift'):
        smoke.verify_sources(pins)
    receipt = tmp_path/'receipt.json'
    smoke.write_json_fresh(receipt, {'status': 'success'})
    with pytest.raises(FileExistsError):
        smoke.write_json_fresh(receipt, {'status': 'failure'})
    assert json.loads(receipt.read_text()) == {'status': 'success'}


def test_source_drift_case_is_recorded_before_model_construct(monkeypatch, small_fixture, tmp_path):
    monkeypatch.setattr(smoke, 'verify_sources', lambda *_: (_ for _ in ()).throw(RuntimeError('Source drift: injected.py')))
    monkeypatch.setattr(smoke, 'make_case_model', lambda *_: pytest.fail('Construction must not run after drift'))
    result = smoke.run_case(smoke.Case('candidate'), small_fixture, 'cpu', tmp_path/'drift', source_pins={'injected.py': '0'*64})
    assert result['status'] == 'failure' and result['stage'] == 'source_guard'
    assert result['error']['message'] == 'Source drift: injected.py'


def test_main_preserves_failure_and_complete_case_accounting(monkeypatch, tmp_path):
    output = tmp_path/'run'
    monkeypatch.setattr(smoke, 'project_path', lambda value: Path(value))
    monkeypatch.setattr(smoke, 'verify_sources', lambda *_: None)
    monkeypatch.setattr(smoke, 'require_storage', lambda **_: 0)
    monkeypatch.setattr(smoke, 'initialize_execution', lambda *_, **__: (_ for _ in ()).throw(RuntimeError('synthetic startup rejection')))
    assert smoke.main(['--output', str(output), '--device', 'cpu']) == 1
    failure = json.loads((output/'run_failure.json').read_text())
    completion = json.loads((output/'completion.json').read_text())
    assert failure['message'] == 'synthetic startup rejection'
    assert completion['status'] == 'failure' and completion['cuda_admitted'] is False
    assert len(completion['uncompleted_cases']) == 44
    with pytest.raises(FileExistsError):
        smoke.main(['--output', str(output), '--device', 'cpu'])


def test_default_full_boundary_and_age_preprocessing():
    fixture = smoke.prepare_fixture()
    assert fixture.boundary.leaf_x.shape == (3, 88, 47)
    assert fixture.anchors.operators['age'].cqq.shape == (3, 3, 64, 64)
    assert fixture.anchors.operators['age'].cqr.shape == (3, 3, 64, 24)
    assert fixture.anchors.layout.anchors.numel() == 32
    assert fixture.manifest['age_spectral_reconstruction_relative_max'] < 1e-12
    assert fixture.manifest['age_spectral_orthogonality_max'] < 1e-12
    assert fixture.manifest['references_per_class'] == 6


def test_partial_runtime_receipt_preserves_admitted_steps_on_lease_rejection(monkeypatch):
    events = []; receipt = {}
    patch_execution_probe(monkeypatch, events)
    monkeypatch.setattr(smoke, 'validate_launcher_lease', lambda: (_ for _ in ()).throw(RuntimeError('lease rejection')))
    with pytest.raises(RuntimeError, match='lease rejection'):
        smoke.initialize_execution('cuda', receipt=receipt)
    assert receipt['process_title'] == smoke.PROCESS_TITLE
    assert receipt['deterministic_execution_policy'] == {'strict': True}
    assert receipt['software']['torch'] == torch.__version__
    assert receipt['initialization_stage'] == 'launcher_lease'
    assert receipt['device'] == {'scope': 'not inspected'}
    assert events == ['title', 'policy']


def test_runtime_uuid_mismatch_rejected_before_driver_query(monkeypatch):
    events = []
    patch_execution_probe(monkeypatch, events)
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda *_: events.append('properties') or SimpleNamespace(
        name='synthetic CUDA mock', total_memory=1024, major=8, minor=0, uuid='GPU-wrong-device'))
    with pytest.raises(RuntimeError, match='UUID does not match'):
        smoke.initialize_execution('cuda')
    assert events == ['title', 'policy', 'lease', 'count', 'properties']


def test_case_policy_drift_is_preserved_before_construction(monkeypatch, small_fixture, tmp_path):
    monkeypatch.setattr(smoke, 'deterministic_development_manifest', lambda: (_ for _ in ()).throw(RuntimeError('policy drift')))
    monkeypatch.setattr(smoke, 'make_case_model', lambda *_: pytest.fail('Constructed after policy drift'))
    result = smoke.run_case(smoke.Case('candidate'), small_fixture, 'cpu', tmp_path/'policy_drift')
    assert result['status'] == 'failure' and result['stage'] == 'source_guard'
    assert result['error']['message'] == 'policy drift'


def test_main_keeps_all_case_failures_and_continues(monkeypatch, small_fixture, tmp_path):
    output = tmp_path/'case_failure_run'
    cases = (smoke.Case('candidate', kernel_depth=1), smoke.Case('candidate', kernel_depth=2))
    monkeypatch.setattr(smoke, 'project_path', lambda value: Path(value))
    monkeypatch.setattr(smoke, 'admission_cases', lambda: cases)
    monkeypatch.setattr(smoke, 'verify_sources', lambda *_: None)
    monkeypatch.setattr(smoke, 'require_storage', lambda **_: 0)
    monkeypatch.setattr(smoke, 'initialize_execution', lambda _, receipt: receipt.update({'device_kind': 'cpu'}))
    monkeypatch.setattr(smoke, 'prepare_fixture', lambda: small_fixture)
    attempts = []
    def run(case, fixture, device, directory, source_pins):
        attempts.append(case.identifier)
        return {'status': 'failure' if case.kernel_depth == 1 else 'success', 'stage': 'synthetic_test_stage'}
    monkeypatch.setattr(smoke, 'run_case', run)
    assert smoke.main(['--output', str(output), '--device', 'cpu']) == 1
    completion = json.loads((output/'completion.json').read_text())
    assert attempts == [case.identifier for case in cases]
    assert completion['failed_cases'] == [cases[0].identifier]
    assert completion['successful_cases'] == [cases[1].identifier]
    assert completion['uncompleted_cases'] == []
    assert completion['cuda_admitted'] is False
