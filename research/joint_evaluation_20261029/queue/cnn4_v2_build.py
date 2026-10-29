"""Prepare an isolated CNN4 metadata-reader correction without launching jobs."""
import copy
import difflib
import json
from pathlib import Path
from arggnn.benchmarking.paper_campaign import runtime as r

HERE = Path(__file__).resolve().parent
ORIGINAL_WORKFLOW = r.ROOT / 'src/arggnn/benchmarking/paper_ancestralpaths_four/workflow.py'
MODULE = 'research.joint_evaluation_20261029.queue.cnn4_v2_workflow'


def main():
    old_path = HERE / 'cnn4_plan.json'
    old = json.loads(old_path.read_text())
    assert r.ref(old_path)['sha256'] == '89969f1fd185c6ef03434d51a6d66e53df40f4e687f2323a7c689c8e5672cfda'
    caps, measurements = {}, []
    for task in old['tasks']:
        if task['call']['spec']['stage'] != 'prepare':
            continue
        reference = task['call']['spec']['arguments']['inference_ref']
        assert r.ref(reference['path']) == reference
        size = r.path(reference['path']).stat().st_size
        caps[reference['sha256']] = size
        measurements.append(dict(inference_ref=reference, bytes=size))
    original = ORIGINAL_WORKFLOW.read_text()
    body = original[original.index('def prepare('):original.index('\ndef load_features(')]
    old_line = 'inferred = io._read(inference_ref, observed=True)'
    new_line = 'inferred = io._read(inference_ref, observed=True, cap=INFERENCE_METADATA_BYTES[inference_ref["sha256"]])'
    assert body.count(old_line) == 1
    repaired = body.replace(old_line, new_line)
    imports = '''"""CNN4 preparation with exact measured bounds for the eight pinned manifests.

All original feature extraction, target selection and model code is reused.
The original workflow and the failed original queue attempt remain unchanged.
"""
from collections import Counter
from arggnn.benchmarking.paper_ancestralpaths_four import CONDITION, features
from arggnn.benchmarking.paper_ancestralpaths_four import workflow as original
from arggnn.benchmarking.paper_ancestralpaths_four.workflow import selected_records, extract_rows
from arggnn.benchmarking import european_native_models as io
from arggnn.benchmarking.paper_ancestralpaths import data as original_data
from arggnn.ancestralpaths_native_bridge import ObservedPanel

'''
    entry = '''\n\ndef execute_stage(spec, *, resource_check, execution_check, bounded_runner=None):
    if spec["stage"] != "prepare":
        return original.execute_stage(spec, resource_check=resource_check,
                                      execution_check=execution_check, bounded_runner=bounded_runner)
    return prepare(output_directory=spec["output_directory"], **spec["arguments"],
                   guard=resource_check, execution_check=execution_check)
'''
    code = imports + 'INFERENCE_METADATA_BYTES = ' + repr(caps) + '\n\n\n' + repaired + entry
    workflow = HERE / 'cnn4_v2_workflow.py'
    with workflow.open('x') as output:
        output.write(code)
    with (HERE / 'cnn4_v2_prepare.diff').open('x') as output:
        output.write(''.join(difflib.unified_diff(body.splitlines(True), repaired.splitlines(True),
                       fromfile='original.prepare', tofile='cnn4_v2.prepare')))
    plan = copy.deepcopy(old)
    plan['records_directory'] = str((HERE / 'cnn4_v2_records').relative_to(r.ROOT))
    plan['previous_failed_plan_ref'] = r.ref(old_path)
    plan['metadata_read_correction'] = measurements
    for task in plan['tasks']:
        location = 'experiments/results/paper_ancestralpaths_four_20261027/cnn4_v2/' + task['id']
        task['output_directory'] = location
        task['call']['spec']['output_directory'] = str(r.path(location) / 'native')
        task['call']['module'] = MODULE
        task['source_refs'].append(r.ref(workflow))
    with (HERE / 'cnn4_v2_plan.json').open('x') as output:
        json.dump(plan, output, indent=2, sort_keys=True)
        output.write('\n')
    print(json.dumps(dict(plan_ref=r.ref(HERE / 'cnn4_v2_plan.json'), workflow_ref=r.ref(workflow),
                         corrected_manifests=len(caps), max_manifest_bytes=max(caps.values()), launched=False)))


if __name__ == '__main__':
    main()
