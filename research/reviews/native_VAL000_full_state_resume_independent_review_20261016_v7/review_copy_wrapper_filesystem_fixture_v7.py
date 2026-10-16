"""Actual v7 wrapper ordering in a fresh owned tiny synthetic filesystem."""
from pathlib import Path
import ast, hashlib, importlib.util, json, os, stat, sys, types
ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
HERE = 'research/reviews/native_VAL000_full_state_resume_independent_review_20261016_v7'

def load(p, name):
    s = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m

def run():
    folder = ROOT / HERE / 'wrapper_filesystem_fixture_v7'
    folder.mkdir(exist_ok=False)
    output = folder / 'output'
    aux = folder / 'aux'
    output.mkdir()
    aux.mkdir()
    child = 'same_charge_full_state_resume_v7'
    original_dir = output / 'same_charge_owned_roster_v4' / 'checkpoints'
    original_dir.mkdir(parents=True)
    original_checkpoint = original_dir / 'ep0-ba6500-rank0.pt'
    original_checkpoint.write_bytes(b'TINY_SYNTHETIC_ORIGINAL')
    os.symlink(original_checkpoint.name, original_dir / 'latest-rank0.pt')
    module = load(ROOT / HERE / 'alias_namespace_v7.py', '_wrapper_actual_alias')
    v4 = load(ROOT / 'src/arggnn/project_budget_v4.py', '_wrapper_actual_original_v4')
    events = []

    def require(ok, msg):
        if not ok:
            raise ValueError(msg)

    def local(value):
        p = Path(value)
        p = p if p.is_absolute() else folder / p
        require(p != folder and folder in p.parents and (p.resolve() == p), 'Fixture canonical path')
        return p

    def ref(value):
        p = local(value)
        data = p.read_bytes()
        return dict(path=str(p.relative_to(folder)), sha256=hashlib.sha256(data).hexdigest())

    def read(record):
        p = local(record['path'])
        data = p.read_bytes()
        require(hashlib.sha256(data).hexdigest() == record['sha256'], 'Fixture exact receipt')
        return json.loads(data)

    def save(directory, name, value):
        p = directory / name
        data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 384)
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        return ref(p)
    manifest = folder / 'manifest.json'
    manifest.write_text('{"synthetic":true}\n')
    checkpoint_ref = dict(path=str(original_checkpoint.relative_to(folder)), sha256=hashlib.sha256(original_checkpoint.read_bytes()).hexdigest())
    qualified = dict(current_v4_complete64_identity_and_group_drain_verified=True, checkpoint_identity=module.identity(original_checkpoint.lstat()))
    ns = dict(Path=Path, json=json, os=os, stat=stat, ROOT=folder, CHILD=child, require=require, ref=ref, read=read, save=save, local=local, MANIFEST=str(manifest.relative_to(folder)), RESUME_CHECKPOINT_REF=checkpoint_ref, ROOT_PRESERVED_METADATA_REF=dict(path='SYNTHETIC_metadata', sha256='1' * 64), V4_STDOUT_PROJECTION_REF=dict(path='SYNTHETIC_drain', sha256='2' * 64), ROOT_V6_AMENDMENT_REF=dict(path='SYNTHETIC_v6', sha256='3' * 64), V6_PREPARATION_FAILURE_FACTS=[dict(synthetic=True)], ALIAS_METADATA_CAP=65536, ALIAS_ORIGINAL_BEFORE='ROOT_full_state_resume_v7_original_Composer_alias.before_unlink.json', ALIAS_ORIGINAL_AFTER='ROOT_full_state_resume_v7_original_Composer_alias.strict_after.json', ALIAS_TERMINAL_BEFORE='ROOT_full_state_resume_v7_terminal_Composer_alias.before_unlink.json', alias_module_v7=lambda : module, current_amendment_ref_v1=lambda : dict(path='SYNTHETIC_v7', sha256='4' * 64), ORIGINAL_STATE=(True,), RESUME_STATE=qualified, active_amendment_v1=lambda : ({}, None, None), validate_resume_prerequisites_v5=lambda a: qualified)
    common = ast.parse((ROOT / 'research/reviews/native_VAL000_GPU0_borrowed32_source_20261016_v2/common_v1.py').read_text())
    sizes = next((n for n in common.body if isinstance(n, ast.FunctionDef) and n.name == '_sizes'))
    original_ns = dict(Path=Path, ROOT=folder, os=os, stat=stat, require=require, MAX_DEPTH=64, MAX_NODES=100000)
    exec(compile(ast.Module(body=[sizes], type_ignores=[]), '<actual-original-no-follow-_sizes-on-tiny-files>', 'exec'), original_ns)

    def census(spec, required_write_bytes=0):
        a = original_ns['_sizes'](output, allow_checkpoint_alias=True)
        b = original_ns['_sizes'](aux)
        total = sum((n for (_, n, _) in a + b))
        assert total + required_write_bytes <= 7 * 2 ** 30
        events.append(dict(kind='actual_original_sizes_projection', required_write_bytes=required_write_bytes))
        return {str(output): sum((n for (_, n, _) in a)), str(aux): sum((n for (_, n, _) in b))}
    ns['charged_census_v1'] = census

    class Guard:

        def budget_check_v9(self, storage, *, boundary=False, required_write_bytes=0):
            require(not required_write_bytes or boundary, 'Planned writes require exact owned-scope boundary')
            events.append(dict(kind='original_guard_API', boundary=boundary, required_write_bytes=required_write_bytes))
            return v4.measure_scopes(folder, [output, aux]) if boundary else dict(readonly=True)
    source = ast.parse((ROOT / HERE / 'amendment_contract_v1.py').read_text())
    names = {'alias_readonly_guard_v7', 'alias_strict_boundary_v7', 'alias_durable_metadata_v7', 'alias_original_context_v7', 'normalize_original_alias_v7', 'normalize_resumed_alias_v7'}
    defs = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=defs, type_ignores=[]), '<actual-v7-wrapper-functions-on-real-tiny-files>', 'exec'), ns)
    spec = dict(original_charged_output_directory=str(output.relative_to(folder)), original_charged_auxiliary_scope=str(aux.relative_to(folder)), output_directory=str((output / child).relative_to(folder)), worker_auxiliary_scope=str((aux / child).relative_to(folder)))
    original_bytes = original_checkpoint.read_bytes()
    g = Guard()
    result = ns['normalize_original_alias_v7'](spec, {}, g, {})
    assert result['alias_absent'] is True and original_checkpoint.read_bytes() == original_bytes
    repeated = ns['normalize_original_alias_v7'](spec, {}, g, {})
    assert repeated['unlink_replayed'] is False
    (aux / child).mkdir()
    future = output / child / 'checkpoints'
    future.mkdir(parents=True)
    for i in range(2):
        (future / ('ep%d-rank0.pt' % i)).write_bytes(('TINY_FUTURE_' + str(i)).encode())
    os.symlink('ep1-rank0.pt', future / 'latest-rank0.pt')
    future_bytes = {p.name: p.read_bytes() for p in future.iterdir() if not p.is_symlink()}
    result = ns['normalize_resumed_alias_v7'](spec, {}, g, {}, prove_drained=lambda : True, minimal_wait_facts=dict(synthetic_actual_wait=True))
    assert result['alias_absent'] is True and {p.name: p.read_bytes() for p in future.iterdir()} == future_bytes
    assert original_checkpoint.read_bytes() == original_bytes
    false = [r for r in events if r.get('kind') == 'original_guard_API' and r['boundary'] is False]
    assert false and all((r['required_write_bytes'] == 0 for r in false))
    return dict(schema='native_VAL000_v7_actual_wrapper_tiny_filesystem_fixture', status='PASS_SOFTWARE_ONLY', startup_prep_then_same_amendment_supervisor_no_unlink_replay=True, terminal_two_regular_files_kept_byte_identical=True, actual_original_sizes_canonical_measure_scopes_used=True, preboundary_guard_calls_readonly_with_zero_planned_write=True, exact_premetadata_files_directory_fsynced=True, events=events, no_live_checkpoint_ledger_process_device_or_scientific_access=True)
if __name__ == '__main__':
    if sys.argv[1:] != ['--run']:
        raise SystemExit('Explicit tiny-filesystem-only --run required')
    print(json.dumps(run(), sort_keys=True, indent=2))
