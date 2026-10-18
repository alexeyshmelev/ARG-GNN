"""Dispatch controls preserve active producers and real completion records."""
import unittest
from unittest.mock import patch
from test_workflow_supervision import WorkflowSupervisionTest
from arggnn.benchmarking import workflow as w

class WorkflowControlsTest(unittest.TestCase):
    def setUp(self):
        self.fixture=WorkflowSupervisionTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_drain_waits_for_active_producer_without_starting_successor(self):
        f=self.fixture
        a=f.task("active");b=f.task("later",deps=("active",))
        plan=f.plan([a,b])
        original=f.tiny_script
        def script(*args,**kwargs):
            value=original(*args,**kwargs)
            return value.replace("from arggnn.benchmarking import contracts as c", "time.sleep(.2)\nfrom arggnn.benchmarking import contracts as c")
        f.tiny_script=script
        def policy(_):
            out=dict(approved_workflow_ref=f.ref)
            if (f.root/"active/ready").exists():out["workflow_drain_ref"]=f.ref
            return out
        with f.controller_context(plan,release=policy):
            state=w.run(f.ref,execute_jobs=True,poll_seconds=.01)
        self.assertEqual(state["run_status"],"drained_for_operational_update")
        self.assertEqual(state["tasks"]["active"]["status"],"succeeded")
        self.assertEqual(state["tasks"]["active"]["actual_controller_wait"],0)
        self.assertTrue(state["tasks"]["active"]["owned_process_group_absent"])
        self.assertEqual(state["tasks"]["later"]["status"],"pending")
        self.assertEqual(len(f.processes),1)
        self.assertFalse((f.root/"later").exists())

    def test_hold_preserves_pending_and_runs_independent_task(self):
        f=self.fixture
        a=f.task("held");b=f.task("independent")
        plan=f.plan([a,b]);plan.update(dispatch_hold_task_ids=["held"],dispatch_hold_reason="Preparation repair")
        def policy(_):
            out=dict(approved_workflow_ref=f.ref)
            if (f.root/"independent/actual_wait.json").exists():out["workflow_drain_ref"]=f.ref
            return out
        with f.controller_context(plan,release=policy):
            state=w.run(f.ref,execute_jobs=True,poll_seconds=.01)
        self.assertEqual(state["tasks"]["held"]["status"],"pending")
        self.assertEqual(state["tasks"]["independent"]["status"],"succeeded")
        self.assertFalse((f.root/"held").exists())
        self.assertEqual(len(f.processes),1)

    def test_drain_is_scoped_to_exact_plan(self):
        self.assertIsNone(w.dispatch_block_reason({}, {"workflow_drain_ref":{"path":"other","sha256":"other"}}, self.fixture.ref,"x"))

if __name__=="__main__":unittest.main()
