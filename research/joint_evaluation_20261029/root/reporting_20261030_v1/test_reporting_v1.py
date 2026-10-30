"""CPU-only reporting tests; no launches, Git, CUDA or held-out truth."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import json
import unittest
from unittest.mock import patch
from research.joint_evaluation_20261029.root.reporting_20261030_v1 import effective_state_v1 as e
from research.joint_evaluation_20261029.root.reporting_20261030_v1 import finish_coordinator_v1 as c


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.names = ['task_' + str(i) for i in range(20)]
        self.old = {name: dict(status='failed' if i == 0 else 'blocked', attempt=4) for i, name in enumerate(self.names)}
        self.manifest = dict(preserved_original_rows=deepcopy(self.old))
        self.plan = dict(tasks=[dict(id=name, kind='gpu') for name in self.names] + [dict(id='external', kind='external')])
        self.plan_ref = dict(path='test-plan.json', sha256='test')

    def test_owned_pending_failed_successful_and_external_ignored(self):
        for status in ('pending', 'failed', 'succeeded'):
            effective = dict(self.old, external=dict(status='running', attempt=2))
            state = dict(plan_ref=self.plan_ref, tasks={name: dict(status=status, attempt=5) for name in self.names})
            state['tasks']['external'] = dict(status='blocked', attempt=1)
            actual, history, owned = e.merge_retry(effective, dict(tasks=self.old), self.manifest, self.plan, state, self.plan_ref)
            self.assertEqual(len(actual), 21)
            self.assertEqual(Counter(v['status'] for v in owned.values()), {status: 20})
            self.assertEqual(actual['external'], effective['external'])
            self.assertEqual(history[self.names[0]]['status'], 'failed')
            self.assertEqual(effective[self.names[0]]['attempt'], 4)

    def test_wrong_owner_or_missing_task_rejected(self):
        state = dict(plan_ref=self.plan_ref, tasks=deepcopy(self.old))
        state['tasks'].pop(self.names[0])
        with self.assertRaises(ValueError):
            e.merge_retry(self.old, dict(tasks=self.old), self.manifest, self.plan, state, self.plan_ref)
        state['tasks'][self.names[0]] = self.old[self.names[0]]
        state['plan_ref'] = dict(path='wrong', sha256='wrong')
        with self.assertRaises(ValueError):
            e.merge_retry(self.old, dict(tasks=self.old), self.manifest, self.plan, state, self.plan_ref)

    def test_conditions_remain_distinct_and_unknown_fails(self):
        self.assertIn('equation-corrected', e.condition_label('equation_corrected'))
        self.assertIn('author key-i', e.condition_label('author_key_i'))
        self.assertNotEqual(e.condition_label('equation_corrected'), e.condition_label('author_key_i'))
        with self.assertRaises(ValueError):
            e.condition_label('unknown')

    def test_mixed_selected_models_rejected(self):
        scores = dict(results=[dict(prediction_ref=dict(path=str(i))) for i in range(8)])
        def prediction(ref):
            return dict(schema='paper_argmix_predictions_v1', selected_model_ref=dict(path='fit'+str(int(ref['path'])//4), sha256='test'))
        with patch.object(e, 'read_ref', side_effect=prediction):
            with self.assertRaises(ValueError):
                e.comparator_metadata(scores)

    def test_fingerprint_ignores_poll_epoch_but_tracks_stage(self):
        value = dict(effective={'fit': dict(status='running', attempt=5, updated_unix=1, epochs_completed=1)},
                     phase_rows={'phase':dict(status='waiting', updated_utc='one')}, stage_waits={},
                     native_retry_plan_ref=self.plan_ref, selected_native_ARGMix=None)
        original = e.fingerprint(value)
        value['effective']['fit'].update(updated_unix=2, epochs_completed=2)
        value['phase_rows']['phase']['updated_utc'] = 'two'
        self.assertEqual(original, e.fingerprint(value))
        value['effective']['fit']['status'] = 'succeeded'
        self.assertNotEqual(original, e.fingerprint(value))

    def test_explicit_active_registry_follows_new_owner(self):
        manifest_ref = dict(path='manifest', sha256='m')
        active_manifest_ref = dict(path='active_manifest', sha256='am')
        new_plan_ref = dict(path='active_plan', sha256='ap')
        config = dict(schema='joint_reporting_effective_state_config_v1', original_state_path='original',
                      successor_state_path='successor', overlay_state_paths=[], native_retry_manifest_ref=manifest_ref,
                      default_native_retry_plan_ref=self.plan_ref, native_retry_active_owner_path='registry',
                      phase_state_paths=[], stage_wait_paths=[], historical_old_tree_task_ids=[])
        manifest = dict(self.manifest, retry_task_id=self.names[0], failed_actual_wait_ref=dict(path='old_wait', sha256='w'))
        plan = dict(self.plan, retry_manifest_ref=manifest_ref, operational_continuation_ref=active_manifest_ref,
                    records_directory='new_records')
        state = dict(plan_ref=new_plan_ref, tasks={name:dict(status='pending',attempt=5) for name in self.names})
        registry = dict(schema='native_argmix_retry_active_owner_v1',root_confirmed=True,
                        active_plan_ref=new_plan_ref,active_manifest_ref=active_manifest_ref,state_path='new_records/state.json')
        data = dict(config=config, original=dict(tasks=self.old), successor=dict(tasks=self.old),
                    registry=registry, **{'new_records/state.json':state})
        refs = dict(manifest=manifest, active_manifest=dict(original_retry_manifest_ref=manifest_ref), active_plan=plan)
        with patch.object(e, 'optional', side_effect=lambda path:data.get(str(path))), patch.object(e, 'read_ref', side_effect=lambda ref:refs[ref['path']]):
            result = e.snapshot('config')
            self.assertEqual(result['native_retry_plan_ref'], new_plan_ref)
            self.assertEqual(result['native_retry_state_path'], 'new_records/state.json')
            self.assertTrue(all(row['status']=='pending' for row in result['native_retry_owned_tasks'].values()))
            registry['root_confirmed'] = False
            with self.assertRaises(ValueError):
                e.snapshot('config')

    def test_status_transition_refresh_independent_of_scores(self):
        prior = dict(result_refs=[], pushed=True, status_fingerprint='old')
        self.assertTrue(c.publication_unchanged(prior, [], 'old'))
        self.assertFalse(c.publication_unchanged(prior, [], 'new'))
        self.assertFalse(c.publication_unchanged(dict(result_refs=[], pushed=True), [], 'old'))
        self.assertFalse(c.publication_unchanged(dict(prior, pushed=False), [], 'old'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
