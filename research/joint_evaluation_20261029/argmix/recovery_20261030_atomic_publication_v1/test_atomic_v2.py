"""Deterministic publication race and failure tests; no production subprocesses."""
import inspect
import os
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from research.joint_evaluation_20261029.argmix.recovery_20261030_atomic_publication_v1 import atomic_v1 as atomic, stage_v2 as stage_v1
from research.joint_evaluation_20261029.argmix.recovery_20261030_queue import controller_v1 as original
from research.joint_evaluation_20261029.queue import successor_v2_support as support
from arggnn.benchmarking.paper_campaign import runtime as r
from arggnn.benchmarking.paper_argmix import paper_argmix_score

HERE=Path(__file__).resolve().parent


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='atomic-test-',dir=HERE)
        self.folder=Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_concurrent_reader_sees_absent_or_complete_receipt(self):
        target=self.folder/'actual_wait.json'
        value=dict(actual_wait=True,actual_returncode=0,payload='complete'*30000)
        staged,release=threading.Event(),threading.Event()
        actual_link=atomic.os.link
        errors=[]
        def paused_link(source,destination,**kwargs):
            staged.set()
            if not release.wait(5):raise TimeoutError('Test publication release missing')
            return actual_link(source,destination,**kwargs)
        def writer():
            try:atomic.atomic_record(target,value,self.folder,immutable=True)
            except BaseException as error:errors.append(error)
        with patch.object(atomic.os,'link',paused_link):
            thread=threading.Thread(target=writer);thread.start()
            self.assertTrue(staged.wait(5))
            for _ in range(100):self.assertFalse(target.exists())
            release.set();thread.join(5)
            self.assertFalse(thread.is_alive())
        self.assertEqual(errors,[])
        for _ in range(100):self.assertEqual(json.loads(target.read_text()),value)
        self.assertEqual(list(self.folder.glob('.*.pending.*')),[])

    def test_failure_before_link_publishes_nothing(self):
        target=self.folder/'gpu_chain_completed.json'
        with patch.object(atomic.os,'link',side_effect=OSError('Injected before publication')):
            with self.assertRaisesRegex(OSError,'Injected'):
                atomic.atomic_record(target,dict(complete=True),self.folder,immutable=True)
        self.assertFalse(target.exists())
        self.assertEqual(list(self.folder.iterdir()),[])

    def test_existing_receipt_never_overwritten(self):
        target=self.folder/'actual_wait.json'
        original_bytes=b'{"historical":true}\n';target.write_bytes(original_bytes)
        with self.assertRaises(FileExistsError):
            atomic.atomic_record(target,dict(historical=False),self.folder,immutable=True)
        self.assertEqual(target.read_bytes(),original_bytes)
        self.assertEqual(list(self.folder.glob('.*.pending.*')),[])

    def test_actual_future_consumer_gates_partial_product_until_atomic_wait(self):
        product=self.folder/'barrier.json';wait_path=self.folder/'actual_wait.json';config=self.folder/'config.json'
        product.write_text('{"incomplete":');config.write_text('{}\n')
        task=dict(future_wait_path=str(wait_path),source_task_id='joint_argmix.prediction_barrier',
                  producer_model_id='treeloom_joint_argmix_structure_v1',expected_result_path=str(product),config_ref=r.ref(config))
        # Product existence alone must never trigger a read or declare readiness.
        self.assertIsNone(support.future_result(task))
        product.write_text('{"complete":true}\n')
        result_ref=r.ref(product)
        value=dict(task_id=task['source_task_id'],model_id=task['producer_model_id'],actual_wait=True,
                   actual_returncode=0,owned_process_group_absent=True,pid=987654321,result_ref=result_ref)
        staged,release=threading.Event(),threading.Event()
        actual_link=atomic.os.link;errors=[]
        def paused_link(source,destination,**kwargs):
            staged.set()
            if not release.wait(5):raise TimeoutError('Test publication release missing')
            return actual_link(source,destination,**kwargs)
        def writer():
            try:atomic.atomic_record(wait_path,value,self.folder,immutable=True)
            except BaseException as error:errors.append(error)
        with patch.object(atomic.os,'link',paused_link), patch.object(support.processes,'session_members',return_value=[]), \
             patch.object(paper_argmix_score,'verify_barrier',return_value={'verified':True}) as verifier:
            thread=threading.Thread(target=writer);thread.start();self.assertTrue(staged.wait(5))
            for _ in range(100):self.assertIsNone(support.future_result(task))
            verifier.assert_not_called()
            release.set();thread.join(5)
            self.assertEqual(errors,[])
            self.assertFalse(thread.is_alive())
            for _ in range(10):self.assertEqual(support.future_result(task),result_ref)
            self.assertEqual(verifier.call_count,10)

    def test_full_successful_stage_publishes_both_identical_waits(self):
        records=self.folder/'records';results=self.folder/'results';cache=self.folder/'cache'
        records.mkdir();results.mkdir();cache.mkdir()
        output=results/'prediction';product=output/'completed.json'
        command=[str(original.ROOT/'.runtime/env/bin/python'),'-B','-m',original.WORKER_MODULE,'test-stage-not-executed']
        spec=dict(id='test.synthetic_stage',kind='cpu',argv=command,output_directory=str(output),completion_file='completed.json')
        class FakeChild:
            pid=os.getpid()
            def wait(self,timeout=None):return 0
        def fake_popen(*args,**kwargs):
            output.mkdir();product.write_text('{"complete":true}\n')
            return FakeChild()
        with patch.object(stage_v1.subprocess,'Popen',side_effect=fake_popen) as launched, \
             patch.object(stage_v1,'RESULTS',results),patch.object(stage_v1,'CACHE',cache), \
             patch.object(stage_v1,'frozen',return_value=None),patch.object(stage_v1.processes,'drain',return_value=True):
            reference=stage_v1.stage(spec,command,{},records,{'test_plan':True},{'path':'test.freeze'},1024,
                                     recovery_binding={'test_only':True})
        self.assertEqual(launched.call_count,1)
        record_path=records/spec['id']/'actual_wait.json'
        colocated=output/'actual_wait.json'
        self.assertEqual(record_path.read_bytes(),colocated.read_bytes())
        saved=json.loads(record_path.read_text())
        self.assertEqual(reference,original.ref(record_path))
        self.assertEqual(saved['result_ref'],original.ref(product))
        self.assertEqual(saved['actual_returncode'],0)
        self.assertTrue(saved['actual_wait'] and saved['products_complete'])
        self.assertEqual(list(self.folder.rglob('*.pending.*')),[])

    def test_stage_change_is_only_colocated_publication(self):
        old=inspect.getsource(original.stage)
        expected=old.replace("        with target.open('xb') as stream:\n            stream.write(payload); stream.flush(); os.fsync(stream.fileno())",
                             "        publish_bytes(target, payload, RESULTS, immutable=True, max_bytes=2**30)")
        self.assertNotEqual(old,expected)
        self.assertEqual(inspect.getsource(stage_v1.stage),expected)


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(PublicationTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    receipt=dict(schema='argmix_atomic_publication_focused_tests_v1',tests_run=result.testsRun,
                 passed=result.wasSuccessful(),failures=len(result.failures),errors=len(result.errors),
                 source_ref=r.ref(__file__),tested_source_refs=[r.ref(atomic.__file__),r.ref(stage_v1.__file__)],
                 production_launches=0,signals=0,scientific_inputs_opened=False)
    with (HERE/'test_results_v2.json').open('x') as stream:
        json.dump(receipt,stream,indent=2,sort_keys=True);stream.write('\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
