"""Run only root-claimed future Relate producer stages under a dedicated lock."""
import argparse
import json
from pathlib import Path
import setproctitle
from research.joint_evaluation_20261029.queue_audit_20261030.common import probe_scheduler_v1 as engine
from research.joint_evaluation_20261029.queue_audit_20261030.critical_path.producer_v1 import support_v1 as support


def main():
    setproctitle.setproctitle('AI Auto Research Pipeline Test')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.validate_only:
        plan = engine.validate(json.loads(Path(args.plan).read_text()))
        support.validate_preclaim(plan)
        print(json.dumps({'valid': True, 'tasks': len(plan['tasks'])}))
        return
    result = engine.run(args.plan, support_module=support,
        owner_lock_path='research/joint_evaluation_20261029/queue_audit_20261030/critical_path/producer_v1/owner.lock')
    print(json.dumps(result['counts']))
    if result['counts']['failed'] or result['counts']['blocked']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
