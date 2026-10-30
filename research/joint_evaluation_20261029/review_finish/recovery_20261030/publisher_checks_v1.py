"""Independent publisher state tests; all product writes are intercepted in memory."""
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch
import contextlib
import hashlib
import io
import json
from research.joint_evaluation_20261029.root import publish_results_recovery_v1 as subject

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
REVIEW = ROOT/'research/joint_evaluation_20261029/review_finish/recovery_20261030'
MANIFEST = dict(effective_state_path='recovery/state.json', GPU_chain_completed_path='recovery/gpu_chain.json', records_directory='recovery/records')
README = '<!-- joint-model-20261029:start -->\n- **ARGMix:** old status\n- **New American TreeLoom:** old American status\n| Method | Accuracy |\n| unchanged | 0.123456 |\n<!-- joint-model-20261029:end -->\nUnrelated footer\n'
STATUS = ROOT/'research/joint_evaluation_20261029/root/CURRENT_STATUS.json'
MD = ROOT/'README.md'
checks = []

def run_case(name, state, *, paired=False, raw_chain=None, fail_original=False):
    writes = {}
    recovery = None if state is None else dict(status=state, reason='synthetic detail')
    original_status = dict(recorded_utc='old', root_model_chain_receipts=dict(american_joint=None), notes=[],
        completed_comparison_artifacts=['experiments/results/joint_argmix_20261029/scores/completed.json'] if paired else [],
        new_ARGMix_joint_pipeline=dict(status='completed',scope='paired_comparison',comparison_artifacts=['authenticated']))
    files = {STATUS:json.dumps(original_status), MD:README}
    original_called = []
    def original():
        original_called.append(True)
        if fail_original: raise ValueError('synthetic source authentication failure')
    def optional(path):
        name = str(path)
        if name == MANIFEST['effective_state_path']: return recovery
        if name == str(Path(MANIFEST['records_directory'])/'completed.json'): return raw_chain
        if name.endswith('records_v2/joint_argmix.train_cache/actual_wait.json'): return dict(actual_wait=True,actual_returncode=-15,status='failed')
        return None
    def read_text(path, *args, **kwargs):
        assert path in files, 'Unexpected file read: '+str(path)
        return files[path]
    def write_text(path, value, *args, **kwargs):
        assert path in (STATUS, MD), 'Unexpected write: '+str(path)
        writes[path]=value;files[path]=value
        return len(value)
    with patch.object(subject.original,'main',original), patch.object(subject,'optional',optional), \
         patch.object(subject.r,'ref',lambda path:dict(path=str(path),sha256='fixture')), \
         patch.object(subject.r,'read',lambda ref:MANIFEST), \
         patch.object(Path,'read_text',read_text), patch.object(Path,'write_text',write_text), \
         patch('sys.argv',['publisher','--manifest','synthetic.json']), contextlib.redirect_stdout(io.StringIO()):
        if fail_original:
            try:subject.main()
            except ValueError:pass
            else:raise AssertionError('Authentication failure accepted')
            assert not writes
            checks.append(name);return
        subject.main()
    assert original_called == [True]
    out=json.loads(writes[STATUS]);readme=writes[MD]
    assert '| unchanged | 0.123456 |' in readme and readme.endswith('Unrelated footer\n')
    assert out['ARGMix_previous_failed_attempt']['actual_returncode']==-15
    assert out['ARGMix_recorded_chain_receipt']==raw_chain
    if paired:
        assert out['new_ARGMix_joint_pipeline']==original_status['new_ARGMix_joint_pipeline']
        assert 'paired ancestry comparison is complete and authenticated' in readme
        assert 'scores remain pending' not in readme and 'comparison remains incomplete' not in readme
    else:
        assert out['new_ARGMix_joint_pipeline']==out['ARGMix_operational_recovery']
        assert 'is complete and authenticated' not in readme
        if state=='failed':assert 'recovery reported a failure' in readme and 'comparison remains incomplete' in readme
        else:assert 'scores remain pending' in readme
    checks.append(name)

run_case('not_launched_is_pending',None)
run_case('running_is_pending','running')
run_case('failed_without_scores_reports_failure','failed')
run_case('paired_complete_with_validation_running_keeps_authenticated_comparison','running',paired=True)
run_case('paired_complete_with_operational_failure_keeps_separate_states','failed',paired=True)
run_case('unverified_products_complete_flag_cannot_publish_completion','running',raw_chain=dict(products_complete=True))
run_case('original_authentication_failure_prevents_all_wrapper_writes','running',fail_original=True)

def ref(path):
    data=path.read_bytes()
    return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
receipt=dict(schema='independent_argmix_recovery_publisher_checks_v1',passed=True,recorded_utc=datetime.now(timezone.utc).isoformat(),
             source_ref=ref(Path(subject.__file__)),original_publisher_ref=ref(Path(subject.original.__file__)),test_source_ref=ref(Path(__file__)),
             checks=checks,all_publisher_writes_captured_in_memory=True,actual_publication=False,production_or_GPU_launched=False,truth_opened=False)
output=REVIEW/'publisher_checks_result_v1.json'
with output.open('x') as stream:json.dump(receipt,stream,sort_keys=True,indent=2);stream.write('\n')
print(json.dumps(dict(passed=True,checks=len(checks),receipt_ref=ref(output)),sort_keys=True))
