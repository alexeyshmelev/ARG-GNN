"""No-dispatch regression checks for CNNv3 continuation and exact idle handoff."""
import copy
import signal
import unittest
from unittest.mock import patch
from research.joint_evaluation_20261029.queue import cnn4_v3_handoff as h
from research.joint_evaluation_20261029.queue import cnn4_v3_admission as a
from arggnn.benchmarking.paper_campaign import runtime as r

BASE = 'research/joint_evaluation_20261029/queue/'


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(prior_controller=dict(pid=42, start_ticks='100'), original_state_path='fake_state')
        self.live = dict(pid=42, start_ticks='100', state='S')
        self.frozen = dict(self.live, state='T')
        self.exited = dict(self.live, state='Z')

    def test_busy_controller_never_signalled(self):
        with patch.object(h, 'identity', return_value=self.live), patch.object(h, 'state_for', return_value={}), patch.object(h, 'descendants', return_value=[43]), patch.object(h, 'boundary_reason', return_value='busy'), patch.object(h.processes, '_pidfd_open') as opened:
            self.assertFalse(h.attempt_idle_stop(self.config, {})['stopped'])
            opened.assert_not_called()

    def test_race_resumes_controller_without_termination(self):
        with patch.object(h, 'identity', side_effect=[self.live, self.live, self.frozen]), patch.object(h, 'state_for', return_value={}), patch.object(h, 'descendants', return_value=[]), patch.object(h, 'boundary_reason', side_effect=[None, 'raced worker']), patch.object(h.processes, '_pidfd_open', return_value=123), patch.object(h.processes, '_pidfd_send') as sent, patch.object(h.os, 'close'):
            self.assertFalse(h.attempt_idle_stop(self.config, {})['stopped'])
            self.assertEqual([call.args[1] for call in sent.call_args_list], [signal.SIGSTOP, signal.SIGCONT])

    def test_exact_idle_stop_preserves_state_without_fake_wait(self):
        with patch.object(h, 'identity', side_effect=[self.live, self.live, self.frozen, self.exited]), patch.object(h, 'state_for', return_value={}), patch.object(h, 'descendants', return_value=[]), patch.object(h, 'boundary_reason', return_value=None), patch.object(h.processes, '_pidfd_open', return_value=123), patch.object(h.processes, '_pidfd_send') as sent, patch.object(h.os, 'close'), patch.object(h.r, 'ref', return_value={'path':'state','sha256':'same'}):
            result=h.attempt_idle_stop(self.config, {})
            self.assertTrue(result['stopped'])
            self.assertFalse(result['old_controller_wait_consumed'])
            self.assertEqual([call.args[1] for call in sent.call_args_list], [signal.SIGSTOP, signal.SIGTERM, signal.SIGCONT])

    def test_pid_reuse_is_rejected_before_signal(self):
        with patch.object(h, 'identity', return_value=dict(self.live,start_ticks='different')), patch.object(h.processes, '_pidfd_open') as opened, self.assertRaises(ValueError):
            h.attempt_idle_stop(self.config,{})
        opened.assert_not_called()

    def test_plan_preserves_all_science_and_authenticates_aliases(self):
        config = r.read(r.ref(BASE+'cnn4_v3_config.json'))
        plan = h.check_config(config)
        self.assertEqual(len(plan['tasks']),15)
        self.assertTrue(all(t['kind'] in ('external','cpu') for t in plan['tasks']))
        self.assertEqual(plan['foreign_ram_reserve_bytes'],1408*2**30)
        for task in plan['tasks']:
            if task['kind']=='external':
                a.verify_alias(task)

    def test_science_and_resource_changes_rejected(self):
        config=r.read(r.ref(BASE+'cnn4_v3_config.json'))
        plan=r.read(config['template_ref'])
        for field,value in [('foreign_ram_reserve_bytes',1407*2**30),('max_parallel_tasks',2)]:
            changed=copy.deepcopy(plan);changed[field]=value
            with self.assertRaises(ValueError):a.validate_continuation(changed)
        changed=copy.deepcopy(plan)
        task=next(t for t in changed['tasks'] if t['kind']=='cpu')
        task['max_rss_bytes']+=1
        with self.assertRaises(ValueError):a.validate_continuation(changed)


if __name__ == '__main__':
    unittest.main()
