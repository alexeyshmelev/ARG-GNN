"""Verify all saved four-path tasks can be constructed without execution."""
import json
import unittest
from arggnn.benchmarking import european_native_models as io
from arggnn.benchmarking.paper_ancestralpaths_four import CONDITION
from arggnn.benchmarking.paper_ancestralpaths_four.plan import build
from arggnn.benchmarking.paper_ancestralpaths_four.workflow import make_call


class FourPathPlanTest(unittest.TestCase):
    def test_saved_plan_and_all_readable_factories(self):
        saved = json.loads((io.ROOT / 'configs/benchmarks/paper_ancestralpaths_four_20261027.json').read_text())
        rebuilt = build(saved['source_config_ref'], output_root='experiments/results/paper_ancestralpaths_four_20261027')
        self.assertEqual(saved, rebuilt)
        self.assertFalse(saved['admitted'])
        self.assertEqual(saved['full_fit_optimizer_updates'], 6670)
        seen = set()
        for task in saved['tasks']:
            self.assertTrue(set(task['dependencies']) <= seen)
            self.assertNotIn(task['id'], seen)
            self.assertEqual(task['resource'], 'cpu')
            call = make_call(output_directory=task['output_directory'], **task['factory_arguments'])
            self.assertEqual(call['spec']['stage'], task['factory_arguments']['stage'])
            self.assertEqual(call['python'], str(io._path(task['factory_arguments']['python'])))
            self.assertGreater(len(call['source_refs']), 5)
            seen.add(task['id'])
        self.assertEqual(len(seen), 15)
        score = saved['tasks'][-1]['factory_arguments']['arguments']
        self.assertEqual(score['expected_methods'], [CONDITION])
        self.assertEqual(len(score['test_targets']), 5)


if __name__ == '__main__':
    unittest.main()
