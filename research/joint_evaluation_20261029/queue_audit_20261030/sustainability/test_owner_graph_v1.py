"""Synthetic owner and blocker tests; no production subprocesses or writes."""
import unittest
from research.joint_evaluation_20261029.queue_audit_20261030.sustainability import owner_graph_v1 as g


class SustainabilityTests(unittest.TestCase):
    def test_current_owner_overrides_external_alias_dependency(self):
        nodes = {}
        g.merge_owner(nodes,dict(tasks=[dict(id='p9',kind='external',dependencies=['retire8'])]),dict(tasks={'p9':dict(status='pending')}),'alias')
        g.merge_owner(nodes,dict(tasks=[dict(id='p9',kind='cpu',dependencies=['retire7'])]),dict(tasks={'p9':dict(status='pending')}),'actual',True)
        self.assertEqual(nodes['p9']['dependencies'],['retire7'])
        self.assertEqual(nodes['p9']['owner'],'actual')

    def test_gpu_wait_resolves_to_running_cpu_producer(self):
        nodes = {name:dict(owner='fixture',kind=kind,status=status,dependencies=deps) for name,kind,status,deps in [
            ('relate','cpu','running',[]),('features','cpu','pending',['relate']),('predict','gpu','pending',['features'])]}
        self.assertEqual(g.blockers(nodes,'predict')[0]['task'],'relate')
        self.assertEqual(g.cycles(nodes),[])

    def test_real_cycle_is_separate_from_waiting_work(self):
        nodes={name:dict(owner='fixture',kind='cpu',status='pending',dependencies=[dep]) for name,dep in [('a','b'),('b','a')]}
        self.assertTrue(g.cycles(nodes))
        self.assertEqual(g.blockers(nodes,'a')[0]['status'],'dependency_cycle')

    def test_same_status_blocker_change_is_observable_without_epoch_churn(self):
        value=dict(tasks={'fit':dict(status='pending',waiting_reason='Storage reservation',epoch=1)},
                   monitor_health=dict(status='healthy',updated_unix=1,unavailable={}))
        first=g.observability_projection(value)
        value['tasks']['fit']['epoch']=2;value['monitor_health']['updated_unix']=2
        self.assertEqual(first,g.observability_projection(value))
        value['tasks']['fit']['waiting_reason']='Host RAM headroom'
        self.assertNotEqual(first,g.observability_projection(value))

    def test_degraded_reason_change_is_observable_without_timestamp_churn(self):
        value=dict(tasks={},monitor_health=dict(status='degraded',unavailable={'gpu_inventory':dict(reason='timeout',observed_unix=1)}))
        first=g.observability_projection(value)
        value['monitor_health']['unavailable']['gpu_inventory']['observed_unix']=2
        self.assertEqual(first,g.observability_projection(value))
        value['monitor_health']['unavailable']['gpu_inventory']['reason']='invalid inventory'
        self.assertNotEqual(first,g.observability_projection(value))


if __name__=='__main__':unittest.main(verbosity=2)
