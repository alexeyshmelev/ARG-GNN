"""One explicit real tiny-filesystem fixture in this fresh owned source folder.

No project ledger, device, process, model, corpus or checkpoint payload is read.
The actual unchanged original canonical/measure_scopes and new normalizer execute
on tiny synthetic files. All fixture files and failed cases remain preserved.
"""
from pathlib import Path
import ast, hashlib, importlib.util, json, os, stat, sys
ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
HERE = 'research/reviews/native_VAL000_full_state_resume_independent_review_20261016_v7'

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def run():
    owned = ROOT / HERE
    work = owned / 'tiny_filesystem_fixture_v7'
    work.mkdir(exist_ok=False)
    original = load(ROOT / 'src/arggnn/project_budget_v4.py', '_fixture_actual_original_budget_v4')
    module = load(owned / 'alias_namespace_v7.py', '_fixture_actual_alias_v7')
    original_sha = hashlib.sha256((ROOT / 'src/arggnn/project_budget_v4.py').read_bytes()).hexdigest()
    cases = []

    def create(name, count=1, target_text=None):
        root = work / name
        root.mkdir()
        checkpoint = root / 'output' / 'checkpoints'
        checkpoint.mkdir(parents=True)
        aux = root / 'aux'
        aux.mkdir()
        for i in range(count):
            (checkpoint / ('ep0-ba%d-rank0.pt' % (6500 + i))).write_bytes(('TINY_SYNTHETIC_ONLY_' + str(i)).encode())
        target = checkpoint / ('ep0-ba%d-rank0.pt' % (6500 + count - 1))
        os.symlink(target_text if target_text is not None else target.name, checkpoint / 'latest-rank0.pt')
        return (root, checkpoint, aux, target)

    def regular_bytes(checkpoint):
        return {p.name: (hashlib.sha256(p.read_bytes()).hexdigest(), module.stamp(p.lstat())) for p in checkpoint.iterdir() if stat.S_ISREG(p.lstat().st_mode)}

    def persist(root, aux, value):
        data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
        p = aux / 'metadata.before.json'
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 384)
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        dfd = os.open(aux, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
        return dict(path=str(p.relative_to(root)), sha256=hashlib.sha256(data).hexdigest())

    def run_case(name, count=1, change=None, drained=True, preserve_change=None, expected_failure=None, extra=None, strict_failure=False):
        (root, checkpoint, aux, target) = create(name, count)
        initial = regular_bytes(checkpoint)
        events = []
        try:
            original.measure_scopes(root, [checkpoint.parent, aux])
        except ValueError as e:
            assert str(e) == 'canonical project path without symlinks required'
        else:
            raise AssertionError('Actual original strict boundary failed to reject genuine symlink')
        if extra:
            extra(root, checkpoint, aux, target)
        expected_ref = dict(path=str(target.relative_to(root)), sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        expected_identity = module.identity(target.lstat())
        if change:
            change(root, checkpoint, aux, target)
        before = regular_bytes(checkpoint)

        def prove():
            events.append('drain')
            return drained

        def preserve(value):
            events.append('durable_metadata')
            if preserve_change:
                return preserve_change(root, checkpoint, aux, target, value)
            return persist(root, aux, value)

        def strict():
            events.append('unchanged_original_strict')
            assert not (checkpoint / 'latest-rank0.pt').exists()
            answer = original.measure_scopes(root, [checkpoint.parent, aux])
            if strict_failure:
                raise ValueError('synthetic strict failure retained')
            return answer
        try:
            answer = module.normalize_latest_v7(root, checkpoint, prove_drained=prove, preserve=preserve, revalidate_pre_unlink=lambda : events.append('readonly_preunlink'), strict_boundary=strict, max_bytes=1024, max_regular_files=2, expected_target_ref=expected_ref if count == 1 else None, expected_target_identity=expected_identity if count == 1 else None, context=dict(synthetic=True, no_live_project_access=True))
        except (ValueError, RuntimeError) as e:
            assert expected_failure is not None, (name, str(e))
            assert expected_failure in str(e), (name, str(e), expected_failure)
            status = 'PASS_REJECTED_EXPECTED'
        else:
            assert expected_failure is None, (name, answer)
            assert answer['alias_absent'] is True
            assert regular_bytes(checkpoint) == initial
            assert 'durable_metadata' in events and events.index('durable_metadata') < events.index('readonly_preunlink') < events.index('unchanged_original_strict')
            status = 'PASS_NORMALIZED_EXPECTED'
        assert set(regular_bytes(checkpoint)) == set(before)
        cases.append(dict(name=name, status=status, events=events, regular_count=len(before), preserved_case_directory=str(root.relative_to(ROOT))))
    run_case('original_exact_SHAqualified', 1)
    run_case('future_two_regular_no_copy', 2)
    run_case('reject_unproven_drain', drained=False, expected_failure='Actual complete checkpoint writer drain')
    run_case('reject_absolute_target', change=lambda r, c, a, t: ((c / 'latest-rank0.pt').unlink(), os.symlink(str(t), c / 'latest-rank0.pt')), expected_failure='Relative single-component')
    run_case('reject_parent_target', change=lambda r, c, a, t: ((c / 'latest-rank0.pt').unlink(), os.symlink('../' + t.name, c / 'latest-rank0.pt')), expected_failure='Relative single-component')
    run_case('reject_changed_qualified_identity', change=lambda r, c, a, t: t.write_bytes(b'TINY_CHANGED_SYNTHETIC'), expected_failure='Exact ROOT-qualified')
    run_case('reject_three_regular_roster', 3, expected_failure='Exact bounded retained')
    run_case('reject_regular_latest', change=lambda r, c, a, t: ((c / 'latest-rank0.pt').unlink(), (c / 'latest-rank0.pt').write_bytes(b'TINY_BAD_REGULAR')), expected_failure='Only Composer own latest')

    def preserve_error(r, c, a, t, value):
        raise RuntimeError('synthetic durable metadata failed')
    run_case('reject_failed_preservation', preserve_change=preserve_error, expected_failure='synthetic durable metadata failed')

    def tamper(r, c, a, t, value):
        receipt = persist(r, a, value)
        (c / 'latest-rank0.pt').unlink()
        os.symlink('./' + t.name, c / 'latest-rank0.pt')
        return receipt
    run_case('reject_alias_changed_after_metadata', preserve_change=tamper, expected_failure='Exact alias, target and every retained regular unchanged')

    def target_tamper(r, c, a, t, value):
        receipt = persist(r, a, value)
        t.write_bytes(b'TINY_CHANGED_AFTER_DURABLE')
        return receipt
    run_case('reject_target_changed_after_metadata', preserve_change=target_tamper, expected_failure='Exact ROOT-qualified original target identity')
    run_case('strict_failure_preserves_regular_and_before_receipt', strict_failure=True, expected_failure='synthetic strict failure retained')

    def unrelated(r, c, a, t):
        os.symlink(t.name, c.parent / 'unrelated_alias')
    run_case('other_symlink_not_excluded', extra=unrelated, expected_failure='canonical project path without symlinks required')
    reviewed = ast.parse((ROOT / 'research/reviews/American_C3_native_MLM_amendment_v12/terminal_namespace_v12.py').read_text())
    candidate = ast.parse((owned / 'alias_namespace_v7.py').read_text())
    for name in ('stamp', 'own_ancestors'):
        a = next((n for n in reviewed.body if isinstance(n, ast.FunctionDef) and n.name == name))
        b = next((n for n in candidate.body if isinstance(n, ast.FunctionDef) and n.name == name))
        assert ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False)
    old = ROOT / 'research/reviews/native_VAL000_full_state_resume_source_20261016_v6'
    for name in ('direct_gpu0_v1.py', 'corpus_metadata_bridge_v1.py', 'resume_scientific_bridge_v5.py'):
        assert (old / name).read_text().replace(str(old.relative_to(ROOT)), HERE).replace('same_charge_full_state_resume_v5', 'same_charge_full_state_resume_v7') == (owned / name).read_text()
    blocks = []
    for p in [ROOT / 'research/reviews/native_VAL000_GPU0_owned_roster_source_20261016_v4/execution_only_v1.py', owned / 'execution_only_v1.py']:
        tree = ast.parse(p.read_text())
        f = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'execute_v1'))
        blocks.append(next((n.finalbody for n in f.body if isinstance(n, ast.Try) and n.finalbody)))
    assert ast.dump(ast.Module(body=blocks[0], type_ignores=[]), include_attributes=False) == ast.dump(ast.Module(body=blocks[1], type_ignores=[]), include_attributes=False)
    execution = (owned / 'execution_only_v1.py').read_text()
    assert execution.index('normalize_original_alias_v7(s,q,g,storage)') < execution.index('g.validate_storage_v9(')
    assert execution.index('native32_actual_minimal_wait_facts=minimal_facts') < execution.index('normalize_resumed_alias_v7(') < execution.index('minimal=compact_v1(s,"actual_wait_minimal.json"')
    assert hashlib.sha256((ROOT / 'src/arggnn/project_budget_v4.py').read_bytes()).hexdigest() == original_sha
    return dict(schema='native_VAL000_v7_actual_tiny_filesystem_normalizer_fixture', status='PASS_SOFTWARE_ONLY', case_count=len(cases), cases=cases, fixture_files_preserved=True, original_canonical_measure_scopes_executed_on_real_tiny_files=True, full_original_v4_cleanup_finally_AST_equal=True, scientific_and_GPU_capacity_sources_equal_to_v6=True, no_live_checkpoint_process_device_ledger_corpus_or_scientific_access=True, no_actual_resume_or_resource_fit_or_training_claim=True)
if __name__ == '__main__':
    if sys.argv[1:] != ['--run']:
        raise SystemExit('Explicit tiny-filesystem-only --run required')
    print(json.dumps(run(), sort_keys=True, indent=2))
