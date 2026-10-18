"""Run explicitly approved independent American preparation jobs on CPUs.

Metadata-only plan by default. The original supervised workers enforce runtime,
space and science constraints and directly wait on generation and inference.
"""
import argparse
import json
from arggnn.benchmarking import contracts as c
from arggnn.benchmarking.cpu_execution import validate_admission
from arggnn.benchmarking.scheduler import PersistentScheduler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-ref", required=True)
    parser.add_argument("--policy-ref")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    batch = c.read_ref(json.loads(args.batch_ref))
    if batch.get("schema") != "american_cpu_parallel_batch_v1":
        raise ValueError("Explicit CPU batch required")
    if type(batch.get("cpu_parallelism")) is not int or not 1 <= batch["cpu_parallelism"] <= 5:
        raise ValueError("At most five additional CPU pipelines")
    if len(batch["admission_refs"]) != batch["cpu_parallelism"]:
        raise ValueError("One complete admitted parent per CPU lane")
    scheduler = PersistentScheduler(
        c.contained_path(batch["queue_path"]), c.contained_path(batch["controller_lock_path"]),
        validate_admission=validate_admission,
        validate_execution_policy=c.validate_execution_policy,
        cpu_parallelism=batch["cpu_parallelism"])
    if not args.execute:
        print(json.dumps({"mode":"plan", "batch":batch, "queue":scheduler.plan()}, indent=2))
        return
    if not args.policy_ref:
        raise PermissionError("Exact current ROOT release required")
    policy_ref = json.loads(args.policy_ref)
    c.check_execution_release(policy_ref)
    for receipt in batch["admission_refs"]:
        scheduler.enqueue(receipt)
    result = scheduler.run(execute=True, execution_policy_ref=policy_ref)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
