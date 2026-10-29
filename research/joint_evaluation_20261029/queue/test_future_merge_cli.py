"""Exercise two-sidecar adoption and external evidence with isolated fixtures."""
from pathlib import Path
from unittest.mock import patch
import json
import tempfile
from arggnn.benchmarking.paper_campaign import runtime as r
from research.joint_evaluation_20261029.queue import future_merge

HERE=Path(__file__).resolve().parent

def main():
    with tempfile.TemporaryDirectory(prefix='merge_fixture_',dir=HERE) as raw:
        root=Path(raw)
        source_plan_ref=r.save(root/'source_plan.json',{'schema':'fixture'})
        input_ref=r.save(root/'input.json',{'prepared':True})
        tasks={'input':dict(id='input',kind='cpu',dependencies=[]),
               'a':dict(id='a',kind='cpu',dependencies=[],output_directory=str(root/'a')),
               'b':dict(id='b',kind='cpu',dependencies=['input'],output_directory=str(root/'b')),
               'next':dict(id='next',kind='cpu',dependencies=['a','b']),
               'failed':dict(id='failed',kind='cpu',dependencies=[])}
        r.save(root/'original_effective_plan.json',{'tasks':list(tasks.values())})
        original={'plan_ref':source_plan_ref,'plan_revision':0,'tasks':{
            'input':dict(status='succeeded',actual_wait=True,actual_returncode=0,result_ref=input_ref),
            'a':dict(status='blocked',error='Output already exists, no overwrite'),
            'b':dict(status='blocked',error='Output already exists, no overwrite'),
            'next':dict(status='blocked',error='Required predecessor failed'),
            'failed':dict(status='failed',pid=99999996,actual_wait=True,actual_returncode=1)}}
        original_path=root/'original.json';r.save(original_path,original)
        states=[]
        for name in ('a','b'):
            completion=r.save(root/name/'result.json',{'native':'fixture'})
            row=dict(status='succeeded',pid=99999997 if name=='a' else 99999998,actual_wait=True,
                     actual_returncode=0,owned_process_group_absent=True,result_ref=completion)
            r.save(root/name/'actual_wait.json',row)
            definitions=[tasks[name]];rows={name:row}
            if name=='b':
                definitions.insert(0,dict(id='input',kind='external',dependencies=[],result_ref=input_ref))
                rows['input']=dict(status='succeeded',result_ref=input_ref,reused=True)
            plan_ref=r.save(root/(name+'_plan.json'),dict(tasks=definitions,
                        original_identity=dict(plan_ref=source_plan_ref,plan_revision=0)))
            state_path=root/(name+'_state.json');r.save(state_path,dict(plan_ref=plan_ref,tasks=rows));states.append(state_path)
        argv=['future_merge','--original-state',str(original_path)]
        for path in states:argv+=['--sidecar-state',str(path)]
        with patch.object(future_merge,'HERE',root),patch('sys.argv',argv),patch.object(future_merge.processes,'session_members',return_value=[]):
            future_merge.main()
        result=json.loads((root/'future_merge_proposal.json').read_text())
        assert set(result['sidecar_wait_refs'])=={'a','b'}
        assert result['rows']['input']==original['tasks']['input']
        assert result['rows']['failed']==original['tasks']['failed']
        assert result['rows']['next']['status']=='pending'
        assert result['reset_claim_block_descendants']==['next']
        assert result['launch_allowed'] is False and result['live_mutation_performed'] is False
        assert json.loads(original_path.read_text())==original
        assert len(result['sidecar_state_refs'])==2
        receipt=dict(passed=True,check_count=8,multiple_sidecars=True,external_evidence_not_adopted=True,
                     actual_wait_paths_authenticated=True,original_fixture_unchanged=True,
                     test_source_ref=r.ref(__file__),merge_source_ref=r.ref(future_merge.__file__),live_mutation=False)
    r.save(HERE/'future_merge_cli_tests.json',receipt)
    print(json.dumps(receipt))

if __name__=='__main__':main()
