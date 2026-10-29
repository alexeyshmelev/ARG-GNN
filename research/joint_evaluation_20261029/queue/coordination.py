"""Narrow cross-controller GPU guard; no process or original-state mutation."""
import json
from arggnn.benchmarking.paper_campaign import runtime as r
from research.joint_evaluation_20261029.queue import claims


def gpu_stage_allowed(plan,task):
    if task['kind']!='gpu' or task['minimum_free_gpu_bytes']<=24*2**30:
        return True
    original=json.loads(r.path(plan['original_state_path']).read_text())
    expected=plan['original_identity']
    if original['plan_ref']!=expected['plan_ref'] or original.get('plan_revision',0)!=expected['plan_revision']:
        return False
    prediction=original['tasks']['paper_american_argformer_american_test_002_predict']
    if prediction['status']!='succeeded' or prediction.get('actual_wait') is not True or prediction.get('actual_returncode')!=0:
        return False
    gate=original['tasks']['paper_american_argformer_american_test_003_prepare']
    if gate['status']!='blocked' or gate.get('error')!='Output already exists, no overwrite' or 'pid' in gate:
        return False
    task003=next(t for t in plan['tasks'] if t['id']=='paper_american_argformer_american_test_003_prepare')
    marker=r.path(task003['output_directory'])/claims.MARKER
    return marker.is_file() and json.loads(marker.read_text())==claims.marker_value(plan,task003)
