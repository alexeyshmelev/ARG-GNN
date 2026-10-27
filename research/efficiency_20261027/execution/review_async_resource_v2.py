"""Independent v2 timing/lifecycle checks; no scientific work or real scans."""
import hashlib
import io
import json
from pathlib import Path
import threading
import time
import unittest
from research.efficiency_20261027.root.async_resource_checks.check_v2 import AsyncResourceCheck
from research.efficiency_20261027.root.async_resource_checks import test_v2

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')

def ref(path):
    path=Path(path)
    return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

class Extra(unittest.TestCase):
    def fake(self):
        now=[0.]
        class Callback:
            task={'max_seconds':1000.}
            started=0.
            next_check=0.
            advance=0.
            def __call__(self,stage):now[0]+=self.advance
        callback=Callback()
        check=AsyncResourceCheck(callback,interval_seconds=30.,maximum_scan_age_seconds=120.,clock=lambda:now[0])
        check.start()
        return now,callback,check
    def test_completed_scan_cannot_cross_existing_lease(self):
        now,callback,check=self.fake()
        now[0]=80.;callback.advance=50.
        with self.assertRaisesRegex(RuntimeError,'freshness'):
            check._scan('independent_periodic',force=True)
        self.assertEqual(check._last_success,0.)
        self.assertIsNotNone(check._error)
        with self.assertRaisesRegex(RuntimeError,'failed'):check.close()
        self.assertFalse(check._thread.is_alive())
    def test_final_scan_duration_fails_completion(self):
        now,callback,check=self.fake()
        callback.advance=130.
        with self.assertRaisesRegex(RuntimeError,'duration'):check.close()
        self.assertIsNotNone(check._error)
        self.assertFalse(check._thread.is_alive())
        with self.assertRaisesRegex(RuntimeError,'failed'):check('after_failed_final_scan')
    def test_waiting_write_expires_on_wall_budget(self):
        callback=test_v2.Original();callback.task['max_seconds']=.15;callback.do_block=True
        check=AsyncResourceCheck(callback,interval_seconds=.01,maximum_scan_age_seconds=1.)
        check.start();errors=[];done=threading.Event()
        def write():
            try:check(23)
            except BaseException as error:errors.append(error)
            finally:done.set()
        self.assertTrue(callback.block.wait(1.))
        writer=threading.Thread(target=write);writer.start()
        try:
            self.assertTrue(done.wait(.5))
            self.assertEqual(len(errors),1)
            self.assertIn('wall time',str(errors[0]))
            self.assertFalse(callback.release.is_set())
            self.assertNotIn(23,callback.calls)
        finally:
            callback.release.set();writer.join(timeout=1.)
            with self.assertRaisesRegex(RuntimeError,'failed'):check.close()
        self.assertFalse(writer.is_alive() or check._thread.is_alive())

def main():
    module=Path(test_v2.__file__).with_name('check_v2.py')
    assert ref(module)['sha256']=='7904ce00d5b4d1ce405a7ebceeb29ce49220eaf9d4218a3a50c6c0bc499f8069'
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(test_v2),unittest.defaultTestLoader.loadTestsFromTestCase(Extra)])
    capture=io.StringIO();result=unittest.TextTestRunner(stream=capture,verbosity=2).run(suite)
    assert result.wasSuccessful() and result.testsRun==16,capture.getvalue()
    report=dict(schema='TreeLoom_async_resource_independent_review_v2',passed=True,
        source_refs=[ref(module),ref(Path(test_v2.__file__)),ref(Path(__file__))],
        independent_CPU_tests_passed=16,original_tests_rerun=13,additional_timing_checks=3,
        closed_findings=['Scan duration is bounded before successful lease renewal, and an existing expired lease cannot be erased at completion','Timed lock acquisition rechecks wall/freshness/error before and after acquisition; waiting writes fail before executing after deadline'],
        extra_checks=['Sub-lease scan duration that nevertheless crosses existing lease fails and latches','Overlong final scan prevents successful completion','Waiting prospective write fails its wall budget before periodic scan releases'],
        source_integration_conditions=[
            'Source-only approval of this wrapper; an actual integration patch must bind the exact original ResourceCheck and give it exclusive callback ownership.',
            'Keep existing trainer own-RSS and phase checks; preserve all numeric caps and integer prospective-write semantics.',
            'Start before dispatch; close and propagate final-scan/join failure before declaring completion.',
            'Retain ROOT external guardian for a callback that itself cannot return or a Python computation that does not reach cooperative checks.',
            '30second delay begins after scan completion; report actual cadence and sampled protection without claiming a hard resource quota.'],
        no_deployment_authorized=True,production_modified=False,real_resource_scans=0,scientific_payloads_opened=False,model_or_GPU_calls=0,
        prior_failed_review_ref=ref(ROOT/'research/efficiency_20261027/execution/ASYNC_RESOURCE_V1_SOURCE_REVIEW.json'))
    out=ROOT/'research/efficiency_20261027/execution/ASYNC_RESOURCE_V2_SOURCE_REVIEW.json'
    with out.open('x') as stream:json.dump(report,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps(dict(review=ref(out),passed=True,independent_tests=16)))

if __name__=='__main__':main()
