"""Explicit ARGformer adapter for the reviewed shared probe-safe queue."""
import argparse
from arggnn.benchmarking.paper_campaign import runtime as r
from research.joint_evaluation_20261029.queue_audit_20261030.common import probe_scheduler_v1 as engine
from research.joint_evaluation_20261029.queue_audit_20261030.backfill import support_v1, gpu_v1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    plan = r.read(r.ref(args.plan))
    spec = support_v1.boundary.validate_plan(plan)
    engine.run(args.plan, support_module=support_v1, owner_lock_path=spec['owner_lock_path'],
               gpu_module=gpu_v1, allow_existing_state=True)


if __name__ == '__main__':
    main()
