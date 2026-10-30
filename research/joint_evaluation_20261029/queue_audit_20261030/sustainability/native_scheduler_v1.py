"""Reuse the common observed-I/O scheduler with the exact native producer bridge."""
import argparse
import json
import setproctitle
from arggnn.benchmarking.paper_campaign import runtime as r
from research.joint_evaluation_20261029.queue_audit_20261030.common import probe_scheduler_v1 as engine
from research.joint_evaluation_20261029.queue_audit_20261030.sustainability import native_support_v1 as support


def main():
    setproctitle.setproctitle('AI Auto Research Pipeline Test')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.validate_only:
        plan = engine.validate(json.loads(r.path(args.plan).read_text()))
        support.validate_continuation(plan, require_owner_absent=False, require_producer_active=False)
        print(json.dumps(dict(valid=True, tasks=len(plan['tasks']))))
        return
    result = engine.run(args.plan, support_module=support,
        owner_lock_path='research/joint_evaluation_20261029/queue/native_argmix_cuda_retry_20261030_v1/sidecar_owner.lock')
    print(json.dumps(result['counts']))
    if result['counts']['failed'] or result['counts']['blocked']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
