"""No-dispatch admission, artifact, release and publication boundary checks."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock
from arggnn.benchmarking.paper_campaign import runtime as r
from research.joint_evaluation_20261029.root import finish_coordinator as f

BASE = 'research/joint_evaluation_20261029'


class FinishTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=r.path(BASE+'/review_finish'))
        self.folder = Path(self.temp.name)
        plan_path = BASE+'/root/finish_plan_v2.json'
        if not r.path(plan_path).exists():
            plan_path = BASE+'/root/finish_plan_v1.json'
        self.plan = r.read(r.ref(plan_path))
        config = r.read(self.plan['successor_config_ref'])
        self.template = dict(r.read(config['successor_template_ref']), handoff_config_ref=self.plan['successor_config_ref'])

    def tearDown(self):
        self.temp.cleanup()

    def test_one_byte_over_ceiling_cannot_authorize_admission(self):
        accounting = dict(admitted=False, total_bytes=1200000000001, ceiling_bytes=1200000000000)
        with patch.object(f.support, 'commitment', return_value=accounting) as ledger, patch.object(f, 'immutable') as writer:
            result = f.maybe_admit_cache(self.plan, self.template, self.folder)
            self.assertFalse(result['admitted'])
            self.assertEqual(ledger.call_args.kwargs['argmix_override_bytes'],25*2**30)
            writer.assert_not_called()

    def test_incomplete_model_set_never_releases_GPU(self):
        template = dict(self.template,root_gpu_priority_release_path=str(self.folder/'release.json'))
        with patch.object(f, 'immutable') as writer:
            result = f.maybe_release(self.plan,template,{'american_joint':dict(path='incomplete',sha256='x')})
            self.assertIsNone(result)
            writer.assert_not_called()

    def test_binary_products_are_authenticated_without_JSON_decode(self):
        binary = self.folder/'model.pt';binary.write_bytes(b'\xff\xfe\x00binary weights')
        metadata = self.folder/'product.json';metadata.write_text('{"complete":true}')
        row = dict(actual_wait=True,actual_returncode=0,owned_process_group_absent=True,products_complete=True,
            pid=123456,products=[r.ref(binary)],result_ref=r.ref(metadata))
        reference=r.save(self.folder/'wait.json',row)
        with patch.object(f.processes,'session_members',return_value=[]):
            self.assertEqual(f.waited(reference),row)

    def test_metadata_ref_with_bytes_uses_authenticated_normalized_reference(self):
        metadata = self.folder/'product.json';metadata.write_text('{"complete":true}')
        reference=dict(r.ref(metadata),bytes=metadata.stat().st_size)
        self.assertEqual(f.verify_reference(reference),{'complete':True})
        changed=dict(reference,bytes=reference['bytes']+1)
        with self.assertRaises(ValueError):f.verify_reference(changed)

    def test_failed_wait_rejected(self):
        reference=r.save(self.folder/'wait.json',dict(actual_wait=True,actual_returncode=1,owned_process_group_absent=True,pid=123456))
        with self.assertRaises(ValueError):f.waited(reference)

    def test_live_session_rejects_success_claim(self):
        reference=r.save(self.folder/'wait.json',dict(actual_wait=True,actual_returncode=0,owned_process_group_absent=True,pid=123456))
        with patch.object(f.processes,'session_members',return_value=[{'pid':123457}]),self.assertRaises(ValueError):f.waited(reference)

    def test_repeated_stage_wait_cannot_count_as_entire_GPU_chain(self):
        spec=copy.deepcopy(self.plan['model_chains']['american_joint'])
        stage = dict(actual_wait=True,actual_returncode=0,owned_process_group_absent=True,pid=123456,
                     plan_ref=spec['binding_ref'],command=['only qualification'],products=[])
        stage_ref=r.save(self.folder/'single_stage.json',stage)
        spec['stages'] = [dict(wait_path=str(self.folder/f'expected_{index}.json'),plan_ref=spec['binding_ref'],command=['expected']) for index in range(spec['GPU_stage_count'])]
        chain=dict(stage,stage_actual_wait_refs=[stage_ref]*spec['GPU_stage_count'])
        spec['wait_path']=str(self.folder/'chain.json');r.save(spec['wait_path'],chain)
        plan=dict(self.plan,model_chains={'american_joint':spec})
        with patch.object(f.processes,'session_members',return_value=[]),self.assertRaises(ValueError):
            f.chain_completions(plan)

    def test_unlisted_small_report_rejected_before_git_add(self):
        artifact=self.folder/'unlisted.json';artifact.write_text('{"innocuous":true}')
        calls=[]
        def run(command,**kwargs):
            calls.append(command)
            return Mock(returncode=0,stdout='')
        with patch.object(f.subprocess,'run',side_effect=run),patch.object(f.subprocess,'check_output',return_value='abc123'),self.assertRaises(ValueError):
            f.publish(dict(self.plan,publication_artifact_paths=[]),self.folder,[str(artifact)])
        self.assertFalse(any(command[:2]==['git','add'] for command in calls))

    def test_concurrent_staged_foreign_path_prevents_commit_and_push(self):
        artifact=self.folder/'report.json';artifact.write_text('{}')
        plan=dict(self.plan,publication_artifact_paths=[str(artifact.relative_to(r.ROOT))])
        calls=[]
        def run(command,**kwargs):
            calls.append(command)
            return Mock(returncode=0,stdout='')
        with patch.object(f.subprocess,'run',side_effect=run),patch.object(f.subprocess,'check_output',return_value=b'foreign_file.txt\0'):
            result=f.publish(plan,self.folder,[str(artifact)])
        self.assertFalse(result['pushed'])
        self.assertFalse(any(command[:2] in (['git','commit'],['git','push']) for command in calls))

    def test_complete_unique_chain_and_wrong_stage_command(self):
        spec=copy.deepcopy(self.plan['model_chains']['american_joint'])
        stages=[];wait_refs=[]
        for index in range(spec['GPU_stage_count']):
            path=self.folder/f'stage_{index}.json'
            command=['python','-m','registered_stage',str(index)]
            stages.append(dict(wait_path=str(path),plan_ref=spec['binding_ref'],command=command))
            row=dict(actual_wait=True,actual_returncode=0,owned_process_group_absent=True,pid=123456,
                     plan_ref=spec['binding_ref'],command=command,products=[])
            wait_refs.append(r.save(path,row))
        spec['stages']=stages
        chain=dict(actual_wait=True,actual_returncode=0,owned_process_group_absent=True,pid=123456,
                   plan_ref=spec['binding_ref'],stage_actual_wait_refs=wait_refs)
        spec['wait_path']=str(self.folder/'chain.json');r.save(spec['wait_path'],chain)
        plan=dict(self.plan,model_chains={'american_joint':spec})
        with patch.object(f.processes,'session_members',return_value=[]):
            self.assertEqual(set(f.chain_completions(plan)),{'american_joint'})
            wrong=copy.deepcopy(plan)
            wrong['model_chains']['american_joint']['stages'][0]['command']=['wrong command']
            with self.assertRaises(ValueError):f.chain_completions(wrong)

    def test_report_reference_cannot_traverse_outside_project(self):
        from research.joint_evaluation_20261029.root import result_sections
        payload=b'{}'
        reference=dict(path='../outside.json',sha256=hashlib.sha256(payload).hexdigest())
        with patch.object(Path,'read_bytes',return_value=payload),self.assertRaises((ValueError,AssertionError)):
            result_sections.read_ref(reference)

    def test_existing_staged_work_is_deferred_without_publisher_or_git_add(self):
        artifact=self.folder/'report.json';artifact.write_text('{}')
        with patch.object(f.subprocess,'run',return_value=Mock(returncode=1)) as command:
            plan=dict(self.plan,publication_artifact_paths=[str(artifact.relative_to(r.ROOT))])
            result=f.publish(plan,self.folder,[str(artifact)])
            self.assertFalse(result['pushed'])
            command.assert_called_once()


if __name__ == '__main__':
    unittest.main()
