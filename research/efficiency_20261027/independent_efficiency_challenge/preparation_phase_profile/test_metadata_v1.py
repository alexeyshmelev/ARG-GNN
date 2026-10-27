"""Metadata-only tests: no project preparation import, arrays, Torch or children."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from .timing_v1 import PhaseTimers

BASE = Path(__file__).parent


class BoundaryTests(unittest.TestCase):
    def test_return_and_argument_identity_preserved(self):
        token, result = object(), object()
        seen = []
        def original(value, *, option):
            seen.append((value, option))
            return result
        owner = SimpleNamespace(function=original)
        timer = PhaseTimers()
        with timer.wrap(owner, 'function', 'phase'):
            self.assertIs(owner.function(token, option=token), result)
        self.assertIs(owner.function, original)
        self.assertEqual(seen, [(token, token)])
        timer.require_once(('phase',))

    def test_bound_method_and_exception_restoration(self):
        sentinel = RuntimeError('original failure')
        class Bundle:
            def validate(self):
                raise sentinel
        original = Bundle.validate
        timer = PhaseTimers()
        with self.assertRaises(RuntimeError) as caught:
            with timer.wrap(Bundle, 'validate', 'validation'):
                Bundle().validate()
        self.assertIs(caught.exception, sentinel)
        self.assertIs(Bundle.validate, original)
        self.assertFalse(timer.rows[0]['success'])
        self.assertEqual(timer.rows[0]['error_type'], 'RuntimeError')

    def test_nested_times_and_events_are_explicit(self):
        events = []
        timer = PhaseTimers(events.append)
        timer.call('outer', lambda: timer.call('inner', lambda: 7))
        self.assertEqual([(row['event'], row['phase']) for row in events],
                         [('begin','outer'), ('begin','inner'), ('end','inner'), ('end','outer')])
        child, outer = timer.rows
        self.assertEqual(child['parent_phase'], 'outer')
        self.assertIsNone(outer['parent_phase'])
        self.assertGreaterEqual(outer['wall_seconds'], child['wall_seconds'])
        for row in timer.rows:
            self.assertGreaterEqual(row['calling_thread_CPU_seconds'], 0)
            self.assertGreaterEqual(row['main_process_CPU_seconds'], 0)
        timer.require_once(('outer', 'inner'))

    def test_missing_or_extra_phase_rejected(self):
        timer = PhaseTimers()
        timer.call('one', lambda: None)
        with self.assertRaises(RuntimeError):
            timer.require_once(('one', 'two'))
        timer.call('one', lambda: None)
        with self.assertRaises(RuntimeError):
            timer.require_once(('one',))

    def test_instrumentation_binding_change_is_rejected_and_restored(self):
        original = lambda: None
        owner = SimpleNamespace(function=original)
        timer = PhaseTimers()
        with self.assertRaisesRegex(RuntimeError, 'binding changed'):
            with timer.wrap(owner, 'function', 'phase'):
                owner.function = lambda: 2
        self.assertIs(owner.function, original)


class SourceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (BASE / 'profile_train001_v1.py').read_text()
        cls.tree = ast.parse(cls.source)

    def test_import_is_metadata_only_and_native_call_is_guarded(self):
        for node in self.tree.body:
            if isinstance(node, ast.Import):
                self.assertNotIn('torch', [alias.name for alias in node.names])
            if isinstance(node, ast.ImportFrom):
                self.assertTrue(node.module == 'timing_v1' or not node.module.startswith(('arggnn', 'research')))
        native = [node for node in self.tree.body if isinstance(node, ast.If)]
        self.assertEqual(len(native), 1)
        self.assertEqual(ast.unparse(native[0].test), "__name__ == '__main__'")

    def test_one_observation_and_pack_and_no_model_or_truth_loader(self):
        calls = [ast.unparse(node.func) for node in ast.walk(self.tree) if isinstance(node, ast.Call)]
        self.assertNotIn('torch.load', calls)
        self.assertFalse(any(name.endswith(('build_model', 'train_step', 'load_targets')) for name in calls))
        timer_calls = [node for node in ast.walk(self.tree) if isinstance(node, ast.Call)
                       and ast.unparse(node.func) == 'timers.call']
        timer_calls.sort(key=lambda node: node.lineno)
        self.assertEqual([node.args[0].value for node in timer_calls],
                         ['load_observation', 'prepare_observation_total', 'pack_bundle'])
        self.assertEqual([ast.unparse(node.args[1]) for node in timer_calls],
                         ['load_observation', 'runtime.prepare_observation', 'pack_bundle'])
        self.assertIn("os.environ['CUDA_VISIBLE_DEVICES'] = ''", self.source)
        self.assertIn('CAP = 128 * 2**30', self.source)
        self.assertIn('SECONDS = 3600', self.source)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__, fromlist=['*']))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    receipt = dict(schema='TreeLoom_preparation_phase_metadata_tests_v1',
        tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        passed=result.wasSuccessful(), scientific_arrays_opened=False,
        torch_imported=False, native_preparation_executed=False, child_processes_started=False)
    (BASE / 'CPU_TEST_RESULTS_v1.json').write_text(json.dumps(receipt, indent=2) + '\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
