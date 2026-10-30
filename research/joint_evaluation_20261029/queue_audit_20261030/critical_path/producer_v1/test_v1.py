"""Producer admission and claim tests; fixture files only, no launches or signals."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from arggnn.benchmarking.paper_campaign import runtime as r
from research.joint_evaluation_20261029.queue_audit_20261030.critical_path.producer_v1 import support_v1 as s, stage_v1 as wrapper

HERE = Path(__file__).parent


class ProducerTests(unittest.TestCase):
    def setUp(self):
        self.plan = json.loads((HERE/'plan_v1.json').read_text())
        self.spec = r.read(self.plan['producer_manifest_ref'])
        self.main_state = json.loads(r.path(self.spec['original_state_path']).read_text())
        self.rows = {task['id']:dict(status='pending',task_id=task['id']) for task in self.plan['tasks']}
        self.first = next(task for task in self.plan['tasks'] if task['id'].endswith('iid_002.generated'))

    def test_live_preclaim_has_no_scientific_side_effects(self):
        with mock.patch.object(s, 'validate_contract', return_value=(self.spec,self.main_state)):
            spec, state = s.validate_preclaim(self.plan)
        self.assertEqual(spec['owned_task_ids'], self.spec['owned_task_ids'])
        self.assertEqual(state['tasks'][self.spec['guard_task_id']]['status'], 'running')

    def test_guard_exit_or_original_release_forbids_claim(self):
        state = copy.deepcopy(self.main_state)
        state['tasks'][self.spec['original_release_task_id']]['status'] = 'succeeded'
        with mock.patch.object(s, 'validate_contract', return_value=(self.spec,state)):
            with self.assertRaisesRegex(ValueError,'release gate'):
                s.validate_preclaim(self.plan)
        state = copy.deepcopy(self.main_state)
        state['tasks'][self.spec['guard_task_id']]['pid'] += 1
        with mock.patch.object(s, 'validate_contract', return_value=(self.spec,state)):
            with self.assertRaisesRegex(ValueError,'guard task'):
                s.validate_preclaim(self.plan)

    def test_pending_row_drift_forbids_claim(self):
        state = copy.deepcopy(self.main_state)
        state['tasks'][self.first['id']]['unexpected'] = True
        with mock.patch.object(s, 'validate_contract', return_value=(self.spec,state)):
            with self.assertRaisesRegex(ValueError,'pending row changed'):
                s.validate_preclaim(self.plan)

    def test_exact_marker_only_claim_is_required(self):
        with tempfile.TemporaryDirectory(prefix='claim_fixture_',dir=HERE) as temporary:
            task = copy.deepcopy(self.first)
            task['output_directory'] = temporary
            marker = Path(temporary)/s.MARKER
            marker.write_text(json.dumps(s.marker_value(self.plan,task)))
            self.assertTrue(s.permit_existing_output(self.plan,task,{'status':'pending'}))
            (Path(temporary)/'unexpected_data').write_text('fixture')
            with self.assertRaisesRegex(ValueError,'Never overwrite'):
                s.permit_existing_output(self.plan,task,{'status':'pending'})

    def test_foreign_marker_and_wrong_task_never_authorize_overwrite(self):
        with tempfile.TemporaryDirectory(prefix='claim_fixture_',dir=HERE) as temporary:
            task = copy.deepcopy(self.first);task['output_directory']=temporary
            marker=s.marker_value(self.plan,task);marker['owner']='wrong-owner'
            (Path(temporary)/s.MARKER).write_text(json.dumps(marker))
            with self.assertRaisesRegex(ValueError,'Exact generation gate'):
                s.permit_existing_output(self.plan,task,{'status':'pending'})
            self.assertFalse(s.permit_existing_output(self.plan,task,{'status':'running'}))

    def test_contract_accepts_exact_candidate(self):
        spec,state=s.validate_contract(self.plan)
        self.assertEqual(len(spec['owned_task_ids']),48)
        self.assertEqual(state['plan_ref'],spec['original_plan_ref'])

    def test_contract_rejects_scientific_argument_and_cap_changes(self):
        changed=copy.deepcopy(self.plan)
        next(task for task in changed['tasks'] if task['id']==self.first['id'])['argv'].append('--alter-science')
        with self.assertRaises(ValueError):s.validate_contract(changed)
        changed=copy.deepcopy(self.plan)
        next(task for task in changed['tasks'] if task['id']==self.first['id'])['max_rss_bytes']+=1
        with self.assertRaises(ValueError):s.validate_contract(changed)

    def test_root_confirmed_receipt_required_before_initial_rows(self):
        with tempfile.TemporaryDirectory(prefix='adoption_fixture_',dir=HERE) as temporary:
            receipt_path=Path(temporary)/'claims.json'
            receipt_path.write_text(json.dumps({'schema':'native_relate_producer_claims_v1','root_confirmed':False}))
            spec=copy.deepcopy(self.spec);spec['claims_receipt_path']=str(receipt_path)
            with mock.patch.object(s,'validate_preclaim',return_value=(spec,self.main_state)),mock.patch.object(s,'verify_claims',return_value=[]),mock.patch.object(s.r,'save') as save:
                with self.assertRaisesRegex(ValueError,'Root-confirmed'):
                    s.initial_rows(self.plan)
                save.assert_not_called()

    def test_inventory_scandir_failure_is_not_zero_usage(self):
        def broken_walk(scope, *, followlinks, onerror):
            onerror(PermissionError('temporary inaccessible subtree'))
            return []
        with mock.patch.object(s.os,'walk',side_effect=broken_walk):
            with self.assertRaises(PermissionError):s.scope_inventory(r.path(self.spec['scope_path']))

    def test_inventory_rejects_symlink_scope_entries(self):
        with tempfile.TemporaryDirectory(prefix='inventory_fixture_',dir=HERE) as temporary:
            root=Path(temporary);(root/'link').symlink_to(root/'absent')
            with self.assertRaisesRegex(ValueError,'symlink'):
                s.scope_inventory(root)

    def test_rolling_two_parent_floor_accepts_current_live_capacity(self):
        proof=s.completion_reservation(self.plan,{'tasks':self.rows},self.first)
        self.assertEqual(proof['open_parents'],['published_recon_iid_001','published_recon_iid_002'])
        self.assertTrue(proof['admitted'])
        self.assertLessEqual(proof['complete_remaining_commitment_bytes']+proof['guard_bytes'],proof['scope_cap_bytes'])

    def test_third_open_parent_is_rejected_even_with_empty_inventory(self):
        rows=copy.deepcopy(self.rows);rows[self.first['id']]['status']='succeeded'
        task=next(t for t in self.plan['tasks'] if t['id'].endswith('misspec_000.generated'))
        proof=s.completion_reservation(self.plan,{'tasks':rows},task,sizes={})
        self.assertEqual(len(proof['open_parents']),3)
        self.assertFalse(proof['admitted'])

    def test_future_conversion_floor_can_reject_cheap_generator(self):
        scope=r.path(self.spec['scope_path'])
        sizes={scope/'fixture_retained_blob':80*2**30}
        proof=s.completion_reservation(self.plan,{'tasks':self.rows},self.first,sizes=sizes)
        self.assertFalse(proof['admitted'])
        self.assertGreater(proof['complete_remaining_commitment_bytes'],proof['observed_bytes']+self.first['max_output_bytes'])

    def test_wrapper_rejects_any_scientific_argument_change_before_cli(self):
        argv=['wrapper']+self.first['argv'][4:]+['--unauthorized']
        with mock.patch.object(wrapper.sys,'argv',argv),mock.patch.object(wrapper.setproctitle,'setproctitle'):
            with self.assertRaisesRegex(ValueError,'Exact reviewed generation arguments'):
                wrapper.main()

    def test_wrapper_preserves_cli_and_accepts_claim_only_once(self):
        from arggnn.benchmarking.paper_argmix import paper_argmix_cli as cli, paper_argmix_generator as generator
        original=generator._fresh_directory
        events=[]
        def scientific_cli():
            path=generator._fresh_directory(self.first['output_directory'],lambda n:events.append(('guard',n)))
            self.assertEqual(path,r.path(self.first['output_directory']))
            with self.assertRaisesRegex(ValueError,'only once'):
                generator._fresh_directory(self.first['output_directory'],lambda n:None)
            events.append(('unchanged_cli',))
        with mock.patch.object(wrapper.sys,'argv',['wrapper']+self.first['argv'][4:]),mock.patch.object(wrapper.setproctitle,'setproctitle',side_effect=lambda name:events.append(('title',name))),mock.patch.object(s,'permit_existing_output',return_value=True),mock.patch.object(cli,'main',side_effect=scientific_cli):
            wrapper.main()
        self.assertIs(generator._fresh_directory,original)
        self.assertEqual(events,[('title','AI Auto Research Pipeline Test'),('guard',16384),('unchanged_cli',)])


if __name__=='__main__':
    unittest.main()
