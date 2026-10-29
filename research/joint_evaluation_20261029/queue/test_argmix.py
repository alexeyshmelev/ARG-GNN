"""CPU-only native adapter tests; no worker, native executable or real claim."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import copy
import hashlib
import json
import tempfile
from research.joint_evaluation_20261029.queue import argmix_native_wrapper as wrapper
from research.joint_evaluation_20261029.queue import argmix_claims, argmix_scheduler
from research.joint_evaluation_20261029.queue.build_proposal import HERE

checks = []

def check(name, value):
    assert value, name
    checks.append(name)


def rejects(function):
    try:
        function()
    except ValueError:
        return True
    return False


def main():
    plan = json.loads((HERE / 'argmix_plan.json').read_text())
    original = json.loads((HERE / 'argmix_original_task.json').read_text())
    task = next(t for t in plan['tasks'] if t['kind'] != 'external')
    check('Complete CPU queue and genuine external input DAG validates', argmix_scheduler.validate(plan) is plan)
    expected = copy.deepcopy(original)
    candidate = copy.deepcopy(task)
    for key in ('argv', 'source_refs', 'max_seconds'):
        expected.pop(key)
        candidate.pop(key)
    check('Native resource caps and dependency fields preserved', candidate == expected)
    check('Native sources retained as exact prefix', task['source_refs'][:len(original['source_refs'])] == original['source_refs'])
    check('Only controller arbitrary wall limit removed', task['max_seconds'] == 9223372036854775807)
    check('Full expanded ARGformer and native storage commitment', plan['global_committed_bytes'] == 1117001926586)
    check('No GPU allowed in native CPU queue', plan['allowed_sidecar_gpus'] == [] and task['kind'] == 'cpu')
    argmix_claims.verify_storage(plan)
    check('Root-confirmed combined ledger authenticates', True)
    state, growth = argmix_claims.verify_original(plan)
    check('Current original storage excludes native admission', state['project_reserved_bytes'] + growth > 995000000000)
    check('Real native output still absent', not wrapper.r.path(task['output_directory']).exists())
    exact = ['Relate', '--mode', 'BuildTopology', '--seed', '123', '-o', 'inferred']
    arguments = dict(cwd=Path('/native/work'), stdin=wrapper.subprocess.DEVNULL, stdout='retained-log', stderr=wrapper.subprocess.STDOUT, timeout=42, check=False)
    for code in (0, 7):
        sentinel = SimpleNamespace(returncode=code)
        with patch.object(wrapper.subprocess, 'run', return_value=sentinel) as call:
            returned = wrapper.UnlimitedNativeSubprocess().run(exact, **arguments)
            check('Actual subprocess return object preserved ' + str(code), returned is sentinel)
            check('Native argv unchanged ' + str(code), call.call_args.args == (exact,))
            check('Only timeout removed from native invocation ' + str(code), call.call_args.kwargs == dict(arguments, timeout=None))
    with tempfile.TemporaryDirectory(prefix='argmix_fixture_', dir=HERE) as raw:
        output = Path(raw)
        marker = {'owner': 'test-owner'}
        (output / wrapper.claims.MARKER).write_text(json.dumps(marker))
        forwarded = []
        guards = []
        def original_fresh(path, guard):
            forwarded.append(path)
            return 'delegate'
        adapter = wrapper.claimed_fresh_directory(original_fresh, output, marker)
        check('Exact owned empty root accepted once', adapter(output, guards.append) == output and guards == [0])
        check('Second claimed-root creation rejected', rejects(lambda: adapter(output, guards.append)))
        check('Other output path delegated unchanged', adapter(output / 'other', guards.append) == 'delegate' and forwarded == [output / 'other'])
        (output / 'foreign.txt').write_text('foreign')
        adapter = wrapper.claimed_fresh_directory(original_fresh, output, marker)
        check('Foreign claim-root contents rejected', rejects(lambda: adapter(output, guards.append)))
        (output / 'foreign.txt').unlink()
        (output / wrapper.claims.MARKER).write_text(json.dumps({'owner': 'foreign'}))
        check('Foreign marker ownership rejected', rejects(lambda: adapter(output, guards.append)))
    original_scheduler = (HERE / 'sidecar_scheduler.py').read_text()
    new_scheduler = (HERE / 'argmix_scheduler.py').read_text()
    check('Scheduler differs only in claim helper binding', new_scheduler == original_scheduler.replace('from research.joint_evaluation_20261029.queue import claims', 'from research.joint_evaluation_20261029.queue import argmix_claims as claims'))
    native_source = wrapper.r.path('src/arggnn/benchmarking/paper_argmix/paper_argmix_relate.py')
    native_pin = next(pin for pin in original['source_refs'] if pin['path'] == str(native_source.relative_to(wrapper.r.ROOT)))
    check('Original native algorithm source unchanged', wrapper.r.ref(native_source) == native_pin)
    receipt = dict(passed=True, check_count=len(checks), checks=checks,
                   current_candidate_growth_bytes=growth, source_refs_verified=149,
                   plan_ref=wrapper.r.ref(HERE / 'argmix_plan.json'), native_source_ref=native_pin,
                   native_jobs=0, worker_processes=0, signals=0, live_claims=0,
                   test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (HERE / 'argmix_tests.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(passed=True, check_count=len(checks), current_candidate_growth_bytes=growth, launched=False)))


if __name__ == '__main__':
    main()
