"""Small synthetic partial-failure completion and narrow vanished-entry checks."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import closed_archives_v1 as old
from . import closed_archives_v2 as new
from .test_closed_archives_v1 import Fixture, save, HERE

class PartialFixture:
    def __init__(self,root):
        self.root = root
        original = Fixture(root)
        second = root/original.scope/'native/pre_size/second.bin'
        second.write_bytes(b'synthetic remaining original\n'*20000)
        original.target['candidates'].append(dict(path=str(second.relative_to(root)),**old.record(second)))
        before = old.scope_bytes(root/original.scope)
        original.target['before_scope_bytes'] = before
        original.target['accounting']['group_cap_bytes'] = before
        original.entry['retained_bytes'] = before
        save(root,'source_records/state.json',dict(plan_ref=original.target['producer']['source_plan_ref'],tasks={'AP__synthetic__ap_infer':original.entry}))
        inv = old.read(root,original.inventory_ref)
        inv['targets'] = [original.target]
        original.inventory_ref = save(root,old.INVENTORY,inv)
        base = dict(storage_ceiling_bytes=980000000000,external_storage=[dict(scopes=[original.scope],cap_bytes=before)],
                    workflow_external_predecessors=[],tasks=[])
        first_plan,_ = old.add_recovery_tasks(base,'failed_outputs',repo=root)
        first_task = first_plan['tasks'][0]
        failed_output = root/first_task['output_directory']
        (failed_output/'native').mkdir(parents=True)
        config_ref = save(root,'first_config.json',dict(recovery_plan=first_plan))
        job_ref = save(root,str(failed_output.relative_to(root))+'/job.json',dict(plan_ref=config_ref,task=first_task))
        row = original.target['candidates'][0]
        sidecar = failed_output/'native/file_00000.json'
        entry = old.codec.archive_one(root/row['path'],sidecar,old.MAX_TEMP,lambda label:None,lambda:None)
        wait_ref = save(root,str(failed_output.relative_to(root))+'/actual_wait.json',dict(job_ref=job_ref,
            actual_direct_wait=True,actual_direct_returncode=-15,owned_process_group_absent=True))
        failed_entry = dict(status='failed',actual_controller_wait=-15,owned_process_group_absent=True,
            job_ref=job_ref,retained_bytes=old.scope_bytes(failed_output))
        state_ref = save(root,'failed_records/state.json',dict(plan_ref=config_ref,tasks={first_task['id']:failed_entry}))
        attempt = dict(config_ref=config_ref,state_ref=state_ref,task_id=first_task['id'],entry=failed_entry,
            job_ref=job_ref,wait_ref=wait_ref,output_directory=first_task['output_directory'],
            retained_bytes=failed_entry['retained_bytes'],supervisor_identity=dict(pid=999999999,start_ticks=1))
        prior = dict(original_path=row['path'],sidecar_ref=old.reference(root,sidecar),entry=entry,
            archive_identity=old.record(entry['archive_path'])['identity'],attempt_output_directory=attempt['output_directory'])
        target = dict(key='synthetic',original=original.target,start_scope_bytes=old.scope_bytes(root/original.scope),prior_archives=[prior])
        inventory = dict(schema=new.SCHEMA+'_inventory',original_inventory_ref=original.inventory_ref,
            initial_credit_bytes=0,preserved_failed_attempts=[attempt],targets=[target])
        self.inventory_ref = save(root,new.INVENTORY,inventory)
        self.plan,_ = new.add_recovery_tasks(base,'new_versioned_outputs',repo=root)
        self.task = self.plan['tasks'][0]
        self.spec = dict(schema=new.SCHEMA,inventory_ref=self.inventory_ref,target_key='synthetic',
            output_directory=str(root/self.task['output_directory']/'native'))
        self.original,self.prior,self.attempt = original,prior,attempt
        save(root,'configs/benchmarks/execution_policy.json',dict(approved_closed_storage_recovery_refs=[]))

    def run(self):
        loader = new.load_target
        with patch.object(new,'ROOT',self.root),patch.object(new,'load_target',side_effect=lambda ref,key:loader(ref,key,repo=self.root)),patch.dict(os.environ,CUDA_VISIBLE_DEVICES=''):
            manifest_ref = new.run(self.spec,resource_check=lambda event:None,execution_check=lambda:None)
        job_ref = save(self.root,'versioned_job.json',dict(task=self.task,call=dict(module=new.MODULE,function='run',spec=self.spec)))
        output = self.task['output_directory']
        save(self.root,output+'/actual_wait.json',dict(job_ref=job_ref,actual_direct_wait=True,actual_direct_returncode=0,owned_process_group_absent=True))
        result_ref = save(self.root,output+'/result.json',dict(job_ref=job_ref,value=manifest_ref))
        state = dict(tasks={self.task['id']:dict(status='succeeded',actual_controller_wait=0,owned_process_group_absent=True,
            job_ref=job_ref,result_ref=result_ref,reservation_bytes=self.task['max_output_bytes'],retained_bytes=old.scope_bytes(self.root/output))})
        return manifest_ref,state

class PartialRecoveryTests(unittest.TestCase):
    def test_lossless_partial_completion_and_exact_once_accounting(self):
        with tempfile.TemporaryDirectory(dir=HERE) as value:
            fixture = PartialFixture(Path(value))
            manifest_ref,state = fixture.run()
            manifest = new.read(fixture.root,manifest_ref)
            self.assertEqual(len(manifest['entries']),2)
            self.assertTrue(manifest['entries'][0]['fresh_repair_gzip_verification'])
            self.assertEqual(manifest['entries'][0]['archive_path'],fixture.prior['entry']['archive_path'])
            self.assertTrue(Path(fixture.prior['entry']['archive_path']).exists())
            self.assertEqual(new.read(fixture.root,fixture.attempt['wait_ref'])['actual_direct_returncode'],-15)
            reduction = new.completed_storage_reduction(fixture.plan,state,repo=fixture.root)
            current = old.scope_bytes(fixture.root/fixture.original.scope)
            new_retained = state['tasks'][fixture.task['id']]['retained_bytes']
            old_retained = fixture.attempt['retained_bytes']
            projection = sum(row['cap_bytes'] for row in fixture.plan['external_storage'])+new_retained-reduction
            self.assertEqual(projection,current+old_retained+new_retained)

    def test_failed_old_task_never_receives_success_credit(self):
        with tempfile.TemporaryDirectory(dir=HERE) as value:
            fixture = PartialFixture(Path(value))
            state = dict(tasks={fixture.task['id']:dict(status='pending')})
            self.assertEqual(new.completed_storage_reduction(fixture.plan,state,repo=fixture.root),0)
            self.assertEqual(fixture.attempt['entry']['status'],'failed')

    def test_preserved_archive_cannot_escape_its_permanent_charge(self):
        with tempfile.TemporaryDirectory(dir=HERE) as value:
            fixture = PartialFixture(Path(value))
            _,state = fixture.run()
            fixture.plan['external_storage'].pop()
            with self.assertRaises(ValueError):new.completed_storage_reduction(fixture.plan,state,repo=fixture.root)

    def test_new_failed_wait_and_duplicate_scope_cannot_credit(self):
        with tempfile.TemporaryDirectory(dir=HERE) as value:
            fixture = PartialFixture(Path(value))
            _,state = fixture.run()
            state['tasks'][fixture.task['id']]['actual_controller_wait'] = -15
            with self.assertRaises(ValueError):new.completed_storage_reduction(fixture.plan,state,repo=fixture.root)
            state['tasks'][fixture.task['id']]['actual_controller_wait'] = 0
            duplicate = copy.deepcopy(fixture.task);duplicate['id'] += '_duplicate'
            fixture.plan['tasks'].append(duplicate)
            state['tasks'][duplicate['id']] = copy.deepcopy(state['tasks'][fixture.task['id']])
            with self.assertRaises(ValueError):new.completed_storage_reduction(fixture.plan,state,repo=fixture.root)

    def test_vanished_active_names_skip_but_permission_errors_propagate(self):
        with tempfile.TemporaryDirectory(dir=HERE) as value:
            root = Path(value);output = root/'output';output.mkdir()
            with patch.object(new,'ROOT',root),patch.object(new.os,'walk',return_value=iter([(str(output),[],['vanished.pending'])])):
                self.assertEqual(new.live_scope_bytes(output),0)
            target = output/'blocked';target.write_text('fixture')
            original_lstat = Path.lstat
            def lstat(item,*args,**kwargs):
                if item==target:raise PermissionError('synthetic access failure')
                return original_lstat(item,*args,**kwargs)
            with patch.object(new,'ROOT',root),patch.object(Path,'lstat',lstat):
                with self.assertRaises(PermissionError):new.live_scope_bytes(output)

if __name__=='__main__':unittest.main()
