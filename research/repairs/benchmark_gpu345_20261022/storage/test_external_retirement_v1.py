"""Three CPU proof tests using ordinary temporary metadata, without job launches."""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from arggnn.benchmarking import contracts as c
from . import external_retirement_v1 as helper


class RetirementProofTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.root=Path(self.temp.name).resolve()
        self.addCleanup(self.temp.cleanup)
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(c,'ROOT',self.root))
        self._fixture()

    def write(self,name,value):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,sort_keys=True)+'\n')
        return c.file_ref(path)

    def _fixture(self):
        source=self.root/helper.SOURCE;source.parent.mkdir(parents=True,exist_ok=True);source.write_text('Fixture source pin\n')
        owner_scope=helper.OWNER_SCOPE
        prep_scope='experiments/results/preparation003'
        pred_scope='experiments/results/rebased_prediction003'
        retire_scope='experiments/results/rebased_retirement003'
        self.retire_scope=retire_scope
        prep=self.write(prep_scope+'/native/preparation_completed.json',dict(schema='american_fresh_ARGformer_preparation_v1',parent_id='argformer_american_validation_003'))
        embedding=self.write(owner_scope+'/native/stage_completed.json',dict(stage='embeddings',parent_id='argformer_american_validation_003',
            preparation_ref=prep,disposable_scratch_directory=owner_scope+'/native/scratch'))
        original_task=dict(id=helper.OWNER,resource='gpu0',output_directory=owner_scope,max_output_bytes=69793218560,max_rss_bytes=274877906944)
        original_plan_ref=self.write('configs/original_plan.json',dict(tasks=[original_task],records_directory='.runtime/original'))
        job_ref=self.write('.runtime/original/jobs/owner.json',dict(plan_ref=original_plan_ref,task=original_task,task_fingerprint=c.fingerprint(original_task)))
        identity_ref=self.write(owner_scope+'/supervision_identity.json',dict(job_ref=job_ref,
            child_identity=dict(pid=124,pgid=124,ppid=123,session=124,start_ticks=500)))
        wait=dict(job_ref=job_ref,actual_direct_wait=True,actual_direct_returncode=0,owned_process_group_absent=True,supervisor_error=None)
        wait_ref=self.write(owner_scope+'/actual_wait.json',wait)
        result_ref=self.write(owner_scope+'/result.json',dict(job_ref=job_ref,value=embedding))
        entry=dict(status='succeeded',external=False,job_ref=job_ref,pid=123,actual_controller_wait=0,
            owned_process_group_absent=True,retained_bytes=helper.CHARGED_BYTES,result_ref=result_ref)
        original_state=dict(plan_ref=original_plan_ref,tasks={helper.OWNER:entry})
        self.original_state=original_state
        self.write('.runtime/original/state.json',original_state)
        snapshot=self.write('proof/original_success_snapshot.json',original_state)
        binding=dict(source_task_id=helper.OWNER,source_plan_ref=original_plan_ref,source_task_fingerprint=c.fingerprint(original_task),
            resource='gpu0',output_directory=owner_scope,max_output_bytes=original_task['max_output_bytes'],max_rss_bytes=original_task['max_rss_bytes'],
            original_job_ref=job_ref,original_supervision_identity_ref=identity_ref,original_supervisor_pid=123,
            owned_process_group_id=124,owned_group_leader_session=124,owned_group_leader_start_ticks=500,
            source_state_path='.runtime/original/state.json',source_state_snapshot_ref=snapshot)
        scopes=[prep_scope,owner_scope,pred_scope]
        retirement=dict(id=helper.RETIREMENT,resource='cpu',factory_module='arggnn.benchmarking.retention',factory_function='factory',
            dependencies=[helper.PREDICTION],output_directory=retire_scope,remeasure_succeeded_tasks=[helper.PREPARATION],
            factory_arguments=dict(owner_scopes=scopes,preparation_ref={'$task':helper.PREPARATION,'path':[],'read':False},
                embedding_ref={'$task':helper.OWNER,'path':[],'read':False},prediction_ref={'$task':helper.PREDICTION,'path':[],'read':False}))
        source_ref=c.file_ref(helper.SOURCE)
        field=dict(schema=helper.SCHEMA,owner_task_id=helper.OWNER,retirement_task_id=helper.RETIREMENT,original_binding=binding,
            owner_success_snapshot_ref=snapshot,original_retained_bytes=helper.CHARGED_BYTES,original_wait_ref=wait_ref,
            original_result_ref=result_ref,protected_owner_refs=helper.protected_owner_refs(),helper_source_ref=source_ref)
        self.plan=dict(tasks=[dict(id=helper.OWNER,resource='external',workflow_predecessor=binding),
            dict(id=helper.PREPARATION,resource='cpu',output_directory=prep_scope),
            dict(id=helper.PREDICTION,resource='cpu',output_directory=pred_scope),retirement],
            workflow_external_predecessors=[binding],sources=[source_ref],external_native_retirement_credit=field)
        plan_ref=self.write('configs/current_rebased_plan.json',self.plan)
        seal=self.write(pred_scope+'/native/seal.json',dict(query_ancestry_truth_opened=False))
        prediction=self.write(pred_scope+'/native/prediction.json',dict(stage='retrieval',parent_id='argformer_american_validation_003',
            records=10000,scoring_performed=False,prediction_seal_ref=seal))
        spec=dict(schema='successful_native_scratch_retirement_v1',owner_scopes=scopes,output_directory=retire_scope+'/native',
            preparation_ref=prep,embedding_ref=embedding,prediction_ref=prediction)
        current_job=self.write('.runtime/current/jobs/retirement.json',dict(schema='arggnn_workflow_job_v1',plan_ref=plan_ref,
            task=retirement,task_fingerprint=c.fingerprint(retirement),source_refs=[source_ref],
            call=dict(module='arggnn.benchmarking.retention',function='run',spec=spec)))
        directories=[prep_scope+'/native/scratch',owner_scope+'/native/scratch']
        inventory=self.write(retire_scope+'/native/retirement_inventory.json',dict(schema=spec['schema'],directories=directories,
            preparation_ref=prep,embedding_ref=embedding,prediction_ref=prediction,
            files=[dict(path=directories[0]+'/discarded.bin',bytes=10,mtime_ns=1),dict(path=directories[1]+'/embeddings.pt',bytes=20,mtime_ns=1)]))
        completed=self.write(retire_scope+'/native/retirement_completed.json',dict(inventory_ref=inventory,retired_directories=directories,
            successful_new_scratch_only=True,original_sources_unchanged=True,original_failures_preserved=True))
        retirement_result=self.write(retire_scope+'/result.json',dict(job_ref=current_job,value=completed))
        self.write(retire_scope+'/actual_wait.json',dict(wait,job_ref=current_job))
        external=self.write('.runtime/current/external_owner.json',dict(status='succeeded',value=embedding,external_original_job_ref=job_ref,
            external_source_result_ref=result_ref,external_actual_wait_ref=wait_ref))
        self.state=dict(plan_ref=plan_ref,tasks={helper.OWNER:dict(status='succeeded',external=True,result_ref=external),
            helper.PREPARATION:dict(status='succeeded',result_ref=self.write(prep_scope+'/result.json',dict(value=prep))),
            helper.PREDICTION:dict(status='succeeded',result_ref=self.write(pred_scope+'/result.json',dict(value=prediction))),
            helper.RETIREMENT:dict(status='succeeded',external=False,job_ref=current_job,result_ref=retirement_result,
                actual_controller_wait=0,owned_process_group_absent=True)})

    def test_no_credit_before_actual_completed_retirement(self):
        self.state['tasks'][helper.RETIREMENT]['status']='running'
        self.assertEqual(helper.completed_storage_reduction(self.plan,self.state),0)

    def test_exact_rebased_retirement_credits_only_measured_external_scope(self):
        remaining=helper.scope_bytes(c.contained_path(helper.OWNER_SCOPE))
        actual=helper.completed_storage_reduction(self.plan,self.state)
        self.assertEqual(actual,helper.CHARGED_BYTES-remaining)
        self.assertNotEqual(actual,30)
        self.assertEqual(self.original_state['tasks'][helper.OWNER]['retained_bytes'],helper.CHARGED_BYTES)

    def test_wrong_wait_or_duplicate_local_remeasurement_never_receives_credit(self):
        path=self.root/self.retire_scope/'actual_wait.json'
        wait=json.loads(path.read_text());wait['actual_direct_returncode']=1
        path.write_text(json.dumps(wait))
        with self.assertRaises(ValueError):helper.completed_storage_reduction(self.plan,self.state)
        wait['actual_direct_returncode']=0;path.write_text(json.dumps(wait))
        task=next(item for item in self.plan['tasks'] if item['id']==helper.RETIREMENT)
        task['remeasure_succeeded_tasks'].append(helper.OWNER)
        with self.assertRaises(ValueError):helper.completed_storage_reduction(self.plan,self.state)


if __name__=='__main__':unittest.main()
