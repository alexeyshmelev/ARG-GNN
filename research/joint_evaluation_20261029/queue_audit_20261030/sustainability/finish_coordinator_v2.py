"""Reuse the existing finish coordinator with owner-aware semantic signals."""
from research.joint_evaluation_20261029.root.reporting_20261030_v1 import finish_coordinator_v1 as prior
from research.joint_evaluation_20261029.queue_audit_20261030.sustainability import effective_state_v2 as effective


def main():
    prior.snapshot, prior.fingerprint = effective.snapshot, effective.fingerprint
    prior.main()


if __name__ == '__main__':
    main()
