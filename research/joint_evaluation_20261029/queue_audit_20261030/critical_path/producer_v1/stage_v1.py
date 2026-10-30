"""Accept one exact empty generation claim, then run the unchanged CLI."""
import json
import sys
from pathlib import Path
import setproctitle
from arggnn.benchmarking.paper_campaign import runtime as r
from research.joint_evaluation_20261029.queue_audit_20261030.critical_path.producer_v1 import support_v1 as support


def main():
    if len(sys.argv) < 2 or sys.argv[1] != 'generate':
        raise ValueError('This wrapper accepts generation only')
    setproctitle.setproctitle('AI Auto Research Pipeline Test')
    plan = json.loads((Path(__file__).parent/'plan_v1.json').read_text())
    output = sys.argv[sys.argv.index('--output')+1]
    task = next(t for t in plan['tasks'] if t.get('output_directory') == output)
    support.need(task['argv'][4:] == sys.argv[1:], 'Exact reviewed generation arguments required')
    support.permit_existing_output(plan, task, {'status':'pending'})
    from arggnn.benchmarking.paper_argmix import paper_argmix_cli as cli, paper_argmix_generator as generator
    original = generator._fresh_directory
    used = False
    def claimed(path, guard):
        nonlocal used
        if r.path(path) != r.path(output):
            return original(path, guard)
        support.need(not used, 'Claim may be consumed only once')
        support.permit_existing_output(plan, task, {'status':'pending'})
        guard(16384)
        used = True
        return r.path(output)
    generator._fresh_directory = claimed
    try:
        cli.main()
    finally:
        generator._fresh_directory = original


if __name__ == '__main__':
    main()
