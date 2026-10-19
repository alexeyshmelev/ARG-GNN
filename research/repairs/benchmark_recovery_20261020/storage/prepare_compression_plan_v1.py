"""Prepare twelve explicit CPU archive tasks; never activate or launch them."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from research.repairs.benchmark_recovery_20261020.storage import closed_archives_v1 as recovery
from research.repairs.benchmark_recovery_20261020.storage import archive_supervisor_v1 as supervisor

ROOT = Path(__file__).resolve().parents[4]
DIRECTORY = ROOT / "research/repairs/benchmark_recovery_20261020/storage"

def main():
    source_ref=recovery.reference(ROOT,"configs/benchmarks/executable_workflow_20261019_v9.json")
    source=recovery.read(ROOT,source_ref)
    merged,evidence=recovery.add_recovery_tasks(source,"experiments/results/benchmark_campaign_20261018_workflow/recovery_20261020/storage_v1")
    tasks=[task for task in merged["tasks"] if task.get("factory_module")==recovery.MODULE]
    plan=dict(tasks=tasks,external_storage=source["external_storage"],
              workflow_external_predecessors=source["workflow_external_predecessors"],
              closed_storage_recovery=merged["closed_storage_recovery"])
    native_ref=recovery.reference(ROOT,"research/repairs/benchmark_recovery_20261020/scheduler/validation001_request.draft_v3.json")
    failed_request_ref=recovery.reference(ROOT,"research/repairs/benchmark_recovery_20261020/scheduler/validation001_request.draft_v2.json")
    failed_task=recovery.read(ROOT,failed_request_ref)["task"]
    failed_scope=recovery.path(ROOT,failed_task["output_directory"])
    failed_attempt=dict(request_ref=failed_request_ref,scope=failed_task["output_directory"],
                        job_ref=recovery.reference(ROOT,failed_scope/"job.json"),
                        wait_ref=recovery.reference(ROOT,failed_scope/"actual_wait.json"),
                        retained_bytes=recovery.scope_bytes(failed_scope))
    config=dict(schema="sequential_closed_storage_recovery_20261020_v1",
                created_at_utc=datetime.now(timezone.utc).isoformat(),
                expires_at_utc=(datetime.now(timezone.utc)+timedelta(hours=48)).isoformat(),
                source_plan_ref=source_ref,native_request_ref=native_ref,closed_native_attempts=[failed_attempt],
                inventory_ref=plan["closed_storage_recovery"]["inventory_ref"],
                recovery_plan=plan,
                records_directory=".runtime/benchmarking/closed_storage_recovery_20261020_v1",
                storage_ceiling_bytes=980_000_000_000,
                host_ram_margin_bytes=64*2**30,max_parallel_rss_bytes=1280*2**30,
                concurrent_ROOT_preparation=dict(reservation_id="ROOT_CPU128GiB_checkpoint64MiB",
                    max_output_bytes=64*2**20,max_rss_bytes=128*2**30,
                    execution_authority_granted=False,exact_later_request_binding_required=True),
                source_refs=[recovery.reference(ROOT,name) for name in recovery.NEW_SOURCES+(supervisor.SOURCE,)],
                automatic_retry=False,initial_credit_bytes=0,
                standalone_default_batch_tasks=1,
                admission_lock=supervisor.LOCK,controller_lock=supervisor.CONTROLLER_LOCK)
    config_path=DIRECTORY/"compression_plan_v1.json"
    with config_path.open("x") as handle:
        json.dump(config,handle,indent=2,sort_keys=True)
        handle.write("\n")
    inventory=recovery.read(ROOT,config["inventory_ref"])
    required=dict(config_ref=recovery.reference(ROOT,config_path),
                  approved_closed_storage_recovery_refs_add=[recovery.reference(ROOT,config_path)],
                  closed_storage_recovery_writer_revocations_add=[
                    dict(scope=target["scope"],original_job_ref=target["producer"]["job_ref"],
                         source_plan_ref=target["producer"]["source_plan_ref"]) for target in inventory["targets"]],
                  preserve_existing_policy_fields=True,
                  required_existing_source_drain_ref=source_ref,
                  first_command=".runtime/env/bin/python -s -B -m research.repairs.benchmark_recovery_20261020.storage.archive_supervisor_v1 --config research/repairs/benchmark_recovery_20261020/storage/compression_plan_v1.json --execute --max-tasks 1",
                  no_actual_activation_performed=True)
    with (DIRECTORY/"activation_requirements_v1.json").open("x") as handle:
        json.dump(required,handle,indent=2,sort_keys=True)
        handle.write("\n")
    print("CONFIG_REF",required["config_ref"])
    print("TASK_COUNT",len(tasks),"FIRST_CAP",tasks[0]["max_output_bytes"])
    print("SOURCE_REFS",config["source_refs"])
    print("PREVIEW",supervisor.capacity(config,dict(tasks={task["id"]:dict(status="pending") for task in tasks}),tasks[0]))

if __name__=="__main__":
    main()
