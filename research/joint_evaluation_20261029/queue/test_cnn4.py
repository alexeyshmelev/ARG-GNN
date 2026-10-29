"""Validate CPU adapter parity and combined allowance without training or claims."""
from pathlib import Path
from unittest.mock import patch
import json
from arggnn.benchmarking.paper_ancestralpaths_four.workflow import make_call
from research.joint_evaluation_20261029.queue import cnn4_scheduler as scheduler, cnn4_admission as admission
from research.joint_evaluation_20261029.queue.build_proposal import HERE


def main():
    checks = []
    def check(name, value):
        assert value, name
        checks.append(name)
    plan = json.loads((HERE/'cnn4_plan.json').read_text())
    source = admission.r.read(plan['original_factory_plan_ref'])
    check('Fifteen-task CPU dependency and artifact-reference DAG validates', scheduler.validate(plan) is plan)
    check('Exactly one CPU lane, no GPU task', plan['max_parallel_tasks'] == plan['max_cpu_tasks'] == 1 and all(t['kind'] == 'cpu' for t in plan['tasks']))
    check('Exact original task ordering', [t['id'] for t in plan['tasks']] == [t['id'] for t in source['tasks']])
    for old, new in zip(source['tasks'], plan['tasks']):
        concrete = make_call(output_directory=old['output_directory'], **old['factory_arguments'])
        check('Exact original scientific execution spec '+old['id'], new['call'] == {k:concrete[k] for k in ('module','function','spec')} and new['python'] == concrete['python'] and new['source_refs'] == concrete['source_refs'])
        check('Exact original output, bounds and dependencies '+old['id'], all(new[k] == old[k] for k in ('output_directory','max_output_bytes','max_rss_bytes','dependencies')))
        check('Arbitrary controller wall limit removed '+old['id'], new['max_seconds'] == 9223372036854775807)
    with patch.object(admission.r, 'save') as save:
        admission.admit(plan)
        check('Root allocation and full existing commitment admitted without a live write', save.call_count == 1)
    check('Both evaluation paths share original 1 GiB cap', plan['combined_evaluation_allowance'] == dict(paths=['research/joint_evaluation_20261029','experiments/results/paper_ancestralpaths_four_20261027'],max_bytes=2**30))
    with patch.object(admission.r, 'scope_bytes', side_effect=[2**30,1]):
        check('Combined allowance rejects even one excess byte', not admission.within_shared_allowance(plan))
    check('Native model counts and train policy remain frozen', source['training_rows'] == 40000 and source['model_spec']['parameters'] == 16900 and source['model_spec']['epochs'] == 5 and source['full_fit_optimizer_updates'] == 6670)
    check('No simulation or genealogy inference added', source['new_simulation_required'] is False and source['new_genealogy_inference_required'] is False)
    receipt = dict(passed=True,check_count=len(checks),checks=checks,plan_ref=admission.r.ref(HERE/'cnn4_plan.json'),
                   training_jobs=0,worker_processes=0,claims=0,signals=0,live_state_mutations=0)
    (HERE/'cnn4_tests.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(passed=True,check_count=len(checks),plan_ref=receipt['plan_ref'],launched=False)))


if __name__ == '__main__':
    main()
