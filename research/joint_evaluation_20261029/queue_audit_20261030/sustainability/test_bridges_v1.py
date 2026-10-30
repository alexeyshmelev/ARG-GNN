"""Synthetic ownership, receipt, and reporting checks; no production dispatch."""
from copy import deepcopy
from pathlib import Path
import json
import os
import tempfile
import unittest
from unittest.mock import patch
from arggnn.benchmarking.paper_campaign import runtime as r
from research.joint_evaluation_20261029.queue_audit_20261030.sustainability import producer_owner_v1 as p
from research.joint_evaluation_20261029.queue_audit_20261030.sustainability import effective_state_v2 as e
from research.joint_evaluation_20261029.queue_audit_20261030.sustainability import native_support_v1 as n
from research.joint_evaluation_20261029.queue_audit_20261030.sustainability import publish_results_v2 as publication

HERE = Path(__file__).resolve().parent


def fixture():
    original_tasks = [dict(id=name, kind='cpu', dependencies=[], argv=['python', '-B', '-m', 'science', name],
                          output_directory='/fixture/' + name, max_rss_bytes=100) for name in sorted(p.OWNED_IDS)]
    originals = {task['id']: task for task in original_tasks}
    changes = {}
    for parent, previous in zip(p.PARENTS, [None, 'published_recon_iid_001', *p.PARENTS[:4]]):
        name = 'paper_argmix_' + parent + '.generated'
        changes[name] = dict(before=[], after=[] if previous is None else ['paper_argmix_' + previous + '.release_new_scratch'])
    original_ref = dict(path='original', sha256='original')
    manifest_ref = dict(path='manifest', sha256='manifest')
    plan_ref = dict(path='plan', sha256='plan')
    manifest = dict(schema='native_relate_parallel_producer_v1', original_plan_ref=original_ref,
        original_state_path='original-state', owned_task_ids=sorted(p.OWNED_IDS), parents=p.PARENTS,
        claim_task_ids=sorted(changes), dependency_changes=changes, generation_wrapper_module='wrapper')
    tasks = deepcopy(original_tasks)
    for task in tasks:
        if task['id'] in changes:
            task['dependencies'] = changes[task['id']]['after']
            task['argv'][3] = 'wrapper'
    tasks.append(dict(id=p.ORIGINAL_RELEASE, kind='external', dependencies=[], sidecar_plan_ref=original_ref,
                      state_path='original-state', source_task_id=p.ORIGINAL_RELEASE))
    plan = dict(tasks=tasks, records_directory='fixture-records', producer_manifest_ref=manifest_ref)
    registry = dict(schema='native_relate_producer_active_owner_v1', root_confirmed=True, original_plan_ref=original_ref,
        producer_plan_ref=plan_ref, producer_manifest_ref=manifest_ref, owned_task_ids=sorted(p.OWNED_IDS), state_path='fixture-records/state.json')
    state = dict(plan_ref=plan_ref, tasks={task['id']: dict(status='pending', task_id=task['id']) for task in tasks})
    return registry, manifest, plan, dict(tasks=original_tasks), state


def reporting_value():
    return dict(effective={'consumer': dict(status='pending', waiting_reason='Dependency')},
        native_retry_owned_tasks={'consumer': dict(status='pending')}, successor=dict(tasks={}), overlays={},
        phase_rows={}, stage_waits={}, native_retry_plan_ref={'path':'p','sha256':'h'}, selected_native_ARGMix=None,
        native_retry_monitor_health=dict(status='degraded', unavailable={'gpu':dict(reason='timeout', observed_unix=1)}))


class BridgeTests(unittest.TestCase):
    def test_atomic_readme_replace_preserves_prior_on_failure(self):
        with tempfile.TemporaryDirectory(prefix='publication_fixture_',dir=HERE) as folder:
            target=Path(folder)/'README.md';target.write_text('original')
            with patch.object(publication.os,'replace',side_effect=OSError('temporary publication failure')):
                with self.assertRaises(OSError):publication.atomic_text(target,'replacement')
            self.assertEqual(target.read_text(),'original')
            self.assertEqual(list(Path(folder).iterdir()),[target])
            publication.atomic_text(target,'replacement')
            self.assertEqual(target.read_text(),'replacement')

    def test_exact_contract_and_nonexpanded_namespace(self):
        values = fixture()
        self.assertEqual(len(p.validate_contract(*values)), 49)
        values[0]['owned_task_ids'].append('unregistered')
        with self.assertRaises(ValueError): p.validate_contract(*values)

    def test_activation_binds_root_claims_and_actual_controller(self):
        registry,manifest,plan,original,state=fixture()
        state['controller_pid']=12345
        activation=dict(schema='native_relate_producer_root_activation_v1',root_confirmed=True,pid=12345,
            process_start_ticks='9876',plan_ref=registry['producer_plan_ref'],manifest_ref=registry['producer_manifest_ref'],
            command=['python','-m','research.joint_evaluation_20261029.queue_audit_20261030.critical_path.producer_v1.scheduler_v1','--plan','plan'])
        claims=dict(schema='native_relate_producer_claims_v1',root_confirmed=True,plan_ref=registry['producer_plan_ref'],
                    manifest_ref=registry['producer_manifest_ref'],claims=[dict(path='marker'+str(i),sha256='h') for i in range(6)])
        p.validate_activation(registry,state,activation,claims)
        activation['pid']=12346
        with self.assertRaises(ValueError):p.validate_activation(registry,state,activation,claims)
        activation['pid']=12345;claims['root_confirmed']=False
        with self.assertRaises(ValueError):p.validate_activation(registry,state,activation,claims)
        claims['root_confirmed']=True;claims['manifest_ref']=dict(path='foreign',sha256='foreign')
        with self.assertRaises(ValueError):p.validate_activation(registry,state,activation,claims)

    def test_registry_requires_root_and_bound_state(self):
        values = fixture(); values[0]['root_confirmed'] = False
        with self.assertRaises(ValueError): p.validate_contract(*values)
        values = fixture(); values[4]['plan_ref'] = dict(path='other',sha256='other')
        with self.assertRaises(ValueError): p.validate_contract(*values)

    def test_science_and_dependency_changes_rejected(self):
        values = fixture(); values[2]['tasks'][1]['argv'].append('--changed-seed')
        with self.assertRaises(ValueError): p.validate_contract(*values)
        values = fixture(); name = sorted(values[1]['dependency_changes'])[0]
        values[1]['dependency_changes'][name]['after'] = ['later-label-truth']
        with self.assertRaises(ValueError): p.validate_contract(*values)

    def test_success_requires_real_unchanged_wait_and_command(self):
        name = sorted(p.NATIVE_ALIASES)[0]
        with tempfile.TemporaryDirectory(prefix='receipt_fixture_', dir=HERE) as folder:
            output = Path(folder)
            completed = output / 'completed.json'
            completed.write_text('{"synthetic":true}\n')
            task = dict(id=name, argv=['python','synthetic.py'], output_directory=str(output), completion_file='completed.json')
            row = dict(status='succeeded', task_id=name, command=task['argv'], actual_wait=True,
                       actual_returncode=0, owned_process_group_absent=True, pid=99999999, result_ref=r.ref(completed))
            wait = output / 'actual_wait.json';wait.write_text(json.dumps(row))
            owner = dict(tasks={name:task}, state=dict(tasks={name:row}),
                registry=dict(producer_plan_ref={'path':'plan','sha256':'plan'},state_path='state'),registry_ref={'path':'registry','sha256':'registry'})
            with patch('research.joint_evaluation_20261029.queue.future_merge.processes.session_members', return_value=[]):
                self.assertEqual(p.validated_wait(owner,name)['result_ref'],row['result_ref'])
                row['actual_wait']=False
                with self.assertRaises(ValueError):p.validated_wait(owner,name)
                row['actual_wait']=True;row['command']=['different.py']
                with self.assertRaises(ValueError):p.validated_wait(owner,name)
                row['command']=task['argv'];completed.write_text('{"changed":true}\n')
                with self.assertRaises(ValueError):p.validated_wait(owner,name)

    def test_claim_blocked_original_is_not_consumer_failure(self):
        name = sorted(p.NATIVE_ALIASES)[0]
        registry,manifest,plan,original,state=fixture()
        owner=dict(registry=registry,manifest=manifest,state=state)
        task=dict(id=name,source_task_id=name,kind='external',sidecar_plan_ref=registry['original_plan_ref'],state_path=manifest['original_state_path'])
        self.assertIsNone(p.resolve_alias(task,owner))
        state['tasks'][name]['status']='failed'
        with self.assertRaises(RuntimeError):p.resolve_alias(task,owner)

    def test_only_support_features_route_to_new_owner(self):
        task=dict(id='old',kind='external',source_task_id='old')
        with patch.object(n.prior,'external_result',return_value='original') as fallback:
            self.assertEqual(n.external_result(task),'original')
            fallback.assert_called_once_with(task)
        task=dict(id=sorted(p.NATIVE_ALIASES)[0],kind='external',source_task_id=sorted(p.NATIVE_ALIASES)[0])
        with patch.object(n,'ACTIVE_SPEC',{}),patch.object(n,'active_owner',return_value={}),patch.object(p,'resolve_alias',return_value=None):
            self.assertIsNone(n.external_result(task))

    def test_reporting_overlay_preserves_original_failure_history(self):
        value=reporting_value();name='producer'
        old=dict(status='blocked',error='Output already exists, no overwrite')
        value['effective'][name]=old;value['successor']['tasks'][name]=old
        owner=dict(owned_rows={name:dict(status='running',pid=123)},registry=dict(state_path='producer-state'),state={})
        result=e.merge_producer(value,owner)
        self.assertEqual(result['effective'][name]['status'],'running')
        self.assertEqual(result['original_main_producer_rows'][name],old)
        self.assertEqual(value['effective'][name],old)
        owner['owned_rows']['consumer']=dict(status='pending')
        with self.assertRaises(ValueError):e.merge_producer(value,owner)

    def test_waiting_reason_reaches_fingerprint_without_progress_churn(self):
        value=reporting_value();first=e.fingerprint(value)
        value['effective']['consumer']['epoch']=99
        value['native_retry_monitor_health']['unavailable']['gpu']['observed_unix']=2
        self.assertEqual(e.fingerprint(value),first)
        value['effective']['consumer']['waiting_reason']='Storage reservation'
        self.assertNotEqual(e.fingerprint(value),first)

    def test_changed_degraded_cause_reaches_fingerprint(self):
        value=reporting_value();first=e.fingerprint(value)
        value['native_retry_monitor_health']['unavailable']['gpu']['reason']='Permission denied'
        self.assertNotEqual(e.fingerprint(value),first)
        value=reporting_value();value['producer_monitor_health']=dict(status='degraded',unavailable={'process':dict(reason='timeout')})
        first=e.fingerprint(value)
        value['producer_monitor_health']['unavailable']['process']['reason']='Process denied'
        self.assertNotEqual(e.fingerprint(value),first)


if __name__ == '__main__':
    unittest.main(verbosity=2)
