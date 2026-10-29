"""Verify the isolated metadata repair and an actual full-parent CPU preparation."""
import inspect
import json
from pathlib import Path
from unittest.mock import patch
from arggnn.benchmarking.paper_campaign import runtime as r, processes
from arggnn.benchmarking.paper_ancestralpaths_four import workflow as original
from research.joint_evaluation_20261029.queue import cnn4_scheduler as scheduler, cnn4_admission as admission
from research.joint_evaluation_20261029.queue import cnn4_v2_workflow as corrected

HERE = Path(__file__).resolve().parent
QUEUE = HERE.parent / 'queue'


def main():
    old = json.loads((QUEUE / 'cnn4_plan.json').read_text())
    new = json.loads((QUEUE / 'cnn4_v2_plan.json').read_text())
    checks = []
    def check(name, value):
        assert value, name
        checks.append(name)
    check('Exact fifteen-task DAG validates', scheduler.validate(new) is new and len(new['tasks']) == 15)
    check('Only preparation metadata read changes in the copied function',
          inspect.getsource(corrected.prepare) == inspect.getsource(original.prepare).replace(
          'inferred = io._read(inference_ref, observed=True)',
          'inferred = io._read(inference_ref, observed=True, cap=INFERENCE_METADATA_BYTES[inference_ref["sha256"]])'))
    for before, after in zip(old['tasks'], new['tasks']):
        check('Original scientific arguments and resource bounds: ' + before['id'],
              all(before[k] == after[k] for k in ('id', 'dependencies', 'kind', 'python', 'cpu_threads',
                  'max_seconds', 'max_output_bytes', 'max_rss_bytes'))
              and before['call']['spec']['arguments'] == after['call']['spec']['arguments']
              and before['call']['spec']['stage'] == after['call']['spec']['stage']
              and before['call']['spec']['runtime_prefix'] == after['call']['spec']['runtime_prefix'])
        check('Fresh contained output: ' + before['id'],
              not r.path(after['output_directory']).exists()
              and r.path(after['output_directory']).is_relative_to(r.path('experiments/results/paper_ancestralpaths_four_20261027')))
        check('Original sources preserved plus narrow successor source: ' + before['id'],
              after['source_refs'][:-1] == before['source_refs'] and after['source_refs'][-1] == r.ref(Path(corrected.__file__)))
        for pin in after['source_refs']:
            assert r.ref(pin['path']) == pin
    for measurement in new['metadata_read_correction']:
        reference = measurement['inference_ref']
        check('Exact bounded authenticated manifest: ' + reference['sha256'],
              r.ref(reference['path']) == reference and r.path(reference['path']).stat().st_size ==
              measurement['bytes'] == corrected.INFERENCE_METADATA_BYTES[reference['sha256']])
    with patch.object(admission.r, 'save') as save:
        admission.admit(new)
        check('Fresh successor admission without a write', save.call_count == 1)
    state = json.loads((QUEUE / 'cnn4_records_v1/state.json').read_text())
    check('Original attempt has no live tasks', all(row['status'] in ('failed', 'blocked') for row in state['tasks'].values()))
    for row in state['tasks'].values():
        if 'pid' in row:
            check('Failed original direct wait preserved: ' + str(row['pid']),
                  row['actual_wait'] is True and row['actual_returncode'] == 1 and row['owned_process_group_absent'] is True
                  and not processes.session_members(row['pid']))
    check('Original controller is absent', not (Path('/proc') / str(state['controller_pid'])).exists())
    result_path = r.path('experiments/results/paper_ancestralpaths_four_20261027/cnn4_v2_prepare_smoke_train000/result.json')
    result = r.read(r.ref(result_path))
    check('Actual full-parent prepare completed', result['scientific_success'] is True)
    feature, registry, records, indices, x = original.load_features(result['value'],
                     new['tasks'][0]['call']['spec']['arguments']['source_plan_ref'], split='train')
    check('Original reader accepts complete canonical output', len(x) == len(records) == feature['rows']
          and x.shape[1:] == (5, 8) and feature['query_truth_read'] is False)
    progress = json.loads((result_path.parent / 'resource_progress.json').read_text())
    receipt = dict(passed=True, checks=checks, plan_ref=r.ref(QUEUE / 'cnn4_v2_plan.json'),
        workflow_ref=r.ref(Path(corrected.__file__)), preparation_result_ref=r.ref(result_path),
        original_state_ref=r.ref(QUEUE / 'cnn4_records_v1/state.json'),
        actual_smoke_ssh_command_exit_code=0, full_parent_rows=len(x),
        elapsed_seconds=progress['elapsed_seconds'], output_bytes=r.scope_bytes(result_path.parent),
        preparation_cpu_only=True, original_attempt_preserved=True, production_fits_launched=0, signals_sent=0)
    with (HERE / 'cnn4_v2_checks.json').open('x') as output:
        json.dump(receipt, output, indent=2, sort_keys=True)
        output.write('\n')
    print(json.dumps(dict(passed=True, checks=len(checks), rows=len(x), receipt_ref=r.ref(HERE / 'cnn4_v2_checks.json'),
                         plan_ref=receipt['plan_ref'], elapsed_seconds=receipt['elapsed_seconds'])))


if __name__ == '__main__':
    main()
