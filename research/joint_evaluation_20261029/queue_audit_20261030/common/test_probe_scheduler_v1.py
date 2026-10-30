"""Fault injection for queue I/O, without launching scientific workers or CUDA."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from research.joint_evaluation_20261029.queue_audit_20261030.common import probe_scheduler_v1 as q
from research.joint_evaluation_20261029.queue_audit_20261030.resources import probe_guard_v1 as probe


class QueueIOTests(unittest.TestCase):
    def test_io_unavailable_is_not_zero(self):
        monitor = probe.ProbeMonitor(Path('unused'))
        def unavailable():
            raise OSError('temporary filesystem failure')
        self.assertIsNone(q.observe_io(monitor, 'disk', unavailable))
        self.assertTrue(monitor.degraded)
        self.assertEqual(q.observe_io(monitor, 'disk', lambda: 7), 7)
        self.assertFalse(monitor.degraded)

    def test_contract_error_is_not_hidden(self):
        monitor = probe.ProbeMonitor(Path('unused'))
        with self.assertRaises(ValueError):
            q.observe_io(monitor, 'contract', lambda: (_ for _ in ()).throw(ValueError('changed source')))

    def test_two_task_local_errors_do_not_deadlock(self):
        monitor = probe.ProbeMonitor(Path('unused'))
        monitor.unavailable('prepare:a', OSError('a'))
        monitor.unavailable('source_read:b', OSError('b'))
        self.assertFalse(q.global_admission_blocked(monitor))
        monitor.unavailable('disk_inventory', OSError('unknown disk'))
        self.assertTrue(q.global_admission_blocked(monitor))

    def test_existing_wait_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory(prefix='queue-wait-test-', dir=q.r.ROOT/'.runtime/tmp') as temp:
            path=Path(temp)/'actual_wait.json'
            path.write_text(json.dumps({'actual_returncode':0,'finished_unix':100}))
            with self.assertRaises(ValueError):
                q.publish_wait(path,{'actual_returncode':1,'finished_unix':101})
            self.assertEqual(json.loads(path.read_text())['actual_returncode'],0)

    def test_scope_walk_does_not_silently_undercount(self):
        with tempfile.TemporaryDirectory(prefix='queue-scope-test-', dir=q.r.ROOT/'.runtime/tmp') as temp:
            def failed_walk(*args,**kwargs):
                kwargs['onerror'](PermissionError('directory cannot be inspected'))
            with patch.object(q.os,'walk',failed_walk):
                with self.assertRaises(PermissionError): q.strict_scope_bytes(Path(temp))

    def test_scope_rejects_symlink(self):
        with tempfile.TemporaryDirectory(prefix='queue-scope-test-', dir=q.r.ROOT/'.runtime/tmp') as temp:
            path=Path(temp); (path/'data').write_text('abc'); (path/'link').symlink_to(path/'data')
            with self.assertRaises(ValueError):q.strict_scope_bytes(path)

    def test_native_checkpoint_alias_is_counted_without_traversal(self):
        with tempfile.TemporaryDirectory(prefix='queue-alias-test-', dir=q.r.ROOT/'.runtime/tmp') as temp:
            root=Path(temp)
            folder=root/'experiments/results/paper_american_20261024/argformer_american_test_007/mlm/native/checkpoints'
            folder.mkdir(parents=True)
            target=folder/'ep3-ba162500-rank0.pt';target.write_bytes(b'abc')
            alias=folder/'latest-rank0.pt';alias.symlink_to(target.name)
            with patch.object(q.r,'ROOT',root):
                self.assertEqual(q.strict_scope_bytes(folder),6+len(target.name))
                target.unlink()
                self.assertEqual(q.strict_scope_bytes(folder),len(target.name))
                alias.unlink();alias.symlink_to('../../outside.pt')
                with self.assertRaises(ValueError):q.strict_scope_bytes(folder)

    def exercise(self, fault):
        parent = q.r.ROOT / '.runtime/tmp'
        with tempfile.TemporaryDirectory(prefix='queue-io-test-', dir=parent) as temp:
            root = Path(temp)
            records = root / 'records'
            records.mkdir()
            tasks = [dict(id=name, kind='cpu', dependencies=[], output_directory=str(root/'experiments'/name),
                          python='/usr/bin/python3', argv=['/usr/bin/python3', 'fake-worker', name],
                          max_output_bytes=100000, max_rss_bytes=100000, max_seconds=1000)
                     for name in ('a','b')]
            plan = dict(tasks=tasks,records_directory=str(records),scope_limits=[],foreign_ram_reserve_bytes=0,
                        max_parallel_tasks=2,max_cpu_tasks=2,existing_workers=[])
            (root/'plan.json').write_text(json.dumps(plan))
            support = SimpleNamespace(initial_rows=lambda p:None,state_metadata=lambda p,pp:{},
                    external_result=lambda t:None, external_evidence=lambda t:{},
                    metadata_within_bound=lambda p:True,admission_accounting=lambda p:{'admitted':True,'total_bytes':0},
                    foreign_owner_growth=lambda p:0,task_fits_scope=lambda *a:True,permit_existing_output=lambda *a:False)
            gpu = SimpleNamespace(allowed=lambda p:[],acquire=lambda g:None,release=lambda l:None)
            launched=[]; stops=[]; injected=[]; sleeps=[]
            original_save=q.r.save; original_scope=q.strict_scope_bytes
            class FakeProcess:
                def __init__(self, pid): self.pid=pid; self.calls=0
                def poll(self):
                    self.calls+=1
                    return 0 if self.calls>=4 else None
                def wait(self,timeout=None): return 0
            def popen(command,**kwargs):
                name=command[-1]; launched.append(name)
                output=root/'experiments'/name;output.mkdir(parents=True)
                (output/'result.json').write_text(json.dumps({'task':name}))
                return FakeProcess(70000+len(launched))
            def save(path,payload):
                path=Path(path)
                is_target=(fault=='prepare' and path.name=='a.json') or (fault=='state' and path.name=='state.json' and launched) or (fault in ('receipt','after_replace') and path.name=='actual_wait.json')
                if is_target and not injected:
                    injected.append(str(path))
                    if fault=='after_replace': original_save(path,payload)
                    raise OSError('injected write outage')
                return original_save(path,payload)
            def scope(path):
                if fault=='scope' and launched and not injected:
                    injected.append(str(path));raise OSError('injected inventory outage')
                return original_scope(path)
            def sleep(seconds):
                sleeps.append(seconds)
                if len(sleeps)>25: raise AssertionError('Queue did not recover')
            with patch.object(q.r,'ROOT',root), patch.object(q,'validate',lambda p:p), \
                 patch.object(q.subprocess,'Popen',popen),patch.object(q,'process_start_ticks',lambda p:str(p)), \
                 patch.object(q,'gpu_state',lambda:{}),patch.object(q.time,'sleep',sleep), \
                 patch.object(q.probe,'install_process_probes',lambda:None), \
                 patch.object(q.probe,'host_available_bytes',lambda:10**15), \
                 patch.object(q.processes,'pss',lambda p:0),patch.object(q.processes,'drain',lambda *a,**kw:True), \
                 patch.object(q,'stop_owned',lambda *a:stops.append(a) or 0), \
                 patch.object(q.r,'save',save),patch.object(q,'strict_scope_bytes',scope):
                result=q.run(root/'plan.json', support_module=support,owner_lock_path=records/'owner.lock',gpu_module=gpu)
                self.assertCountEqual(launched,['a','b'])
                self.assertEqual(len(launched),2)
                self.assertEqual(stops,[])
                self.assertEqual(result['counts']['succeeded'],2)
                self.assertEqual(result['counts']['failed'],0)
                self.assertEqual(len(injected),1)
                for task in tasks:
                    wait=json.loads((Path(task['output_directory'])/'actual_wait.json').read_text())
                    self.assertTrue(wait['actual_wait'])
                    self.assertEqual(wait['actual_returncode'],0)
                    self.assertEqual(wait,result['tasks'][task['id']])

    def test_scope_failure_preserves_workers(self): self.exercise('scope')
    def test_state_failure_preserves_workers(self): self.exercise('state')
    def test_receipt_failure_is_retried_without_replay(self): self.exercise('receipt')
    def test_atomic_replace_then_failed_read_preserves_wait(self): self.exercise('after_replace')
    def test_launch_preparation_retries_without_duplicate(self): self.exercise('prepare')

if __name__=='__main__': unittest.main()
