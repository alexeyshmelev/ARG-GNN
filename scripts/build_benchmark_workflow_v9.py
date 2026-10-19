"""Prepare V9 resource/preparation continuation without launching scientific work."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
from arggnn.benchmarking import contracts as c
from arggnn.benchmarking import workflow_predecessor_v5 as predecessor
import build_benchmark_workflow_v4 as prior_builder

PRIOR_PLAN="configs/benchmarks/executable_workflow_20261019_v8.json"
PRIOR_STATE=".runtime/benchmarking/workflow_20261019_v8/state.json"
PLAN="configs/benchmarks/executable_workflow_20261019_v9.json"
RECORDS=".runtime/benchmarking/workflow_20261019_v9"
PRIOR_OUTPUT="experiments/results/benchmark_campaign_20261018_workflow/release_v8"
OUTPUT="experiments/results/benchmark_campaign_20261018_workflow/release_v9"
SELF="scripts/build_benchmark_workflow_v9.py"
NEW_SOURCES=(SELF, "src/arggnn/benchmarking/workflow_v9.py",
             "src/arggnn/benchmarking/deferred_gpu_v9.py",
             "src/arggnn/benchmarking/retention_scheduler_v9.py",
             "src/arggnn/benchmarking/workflow_reporting_v9.py",
             "src/arggnn/benchmarking/european_preparation_v9.py",
             "research/reviews/benchmark_repairs_20261019/european_v9/integration_v1.py",
             "research/reviews/benchmark_repairs_20261019/european_v9/qualification_v1.py",
             "research/reviews/benchmark_repairs_20261019/resources_v9/resource_accounting_v1.py",
             "research/reviews/benchmark_repairs_20261019/resources_v9/native_chain_settlement_v1.json",
             "research/reviews/benchmark_repairs_20261019/resources_v9/european_archive_v1.py",
             "src/arggnn/benchmarking/european_retention_v9.py")
need=prior_builder.need
write_new=prior_builder.write_new

def verify_success(task, entry, prior_ref):
    if task["resource"]=="external":
        if "workflow_predecessor" not in task:
            return prior_builder.verify_success(task,entry,prior_ref)
        original=predecessor.external_result(task)
        need(original is not None and original["status"]=="succeeded",
             "Original workflow predecessor must actually succeed and drain")
        envelope=c.read_ref(entry["result_ref"])
        need(envelope.get("status")=="succeeded" and envelope["value"]==original["value"]
             and envelope["external_actual_wait_ref"]==original["external_actual_wait_ref"],
             "Preserved external wrapper must expose the exact original result value")
        return dict(result_ref=entry["result_ref"],actual_wait_ref=original["external_actual_wait_ref"])
    job=c.read_ref(entry["job_ref"])
    need(job["task"]==task and job["task_fingerprint"]==c.fingerprint(task),
         "Successful original scientific descriptor must remain exact")
    wait_ref,wait=prior_builder._wait(task,entry)
    if job["plan_ref"]!=prior_ref:
        evidence=c.read_ref(c.read_ref(prior_ref)["continuation_evidence_ref"])
        adopted=evidence.get("adopted_succeeded",evidence.get("adopted",{}))[task["id"]]
        need(adopted["result_ref"]==entry["result_ref"] and adopted["actual_wait_ref"]==wait_ref,
             "Exact immutable inherited-success adoption evidence required")
        original=c.read_ref(job["plan_ref"])
        need(next(t for t in original["tasks"] if t["id"]==task["id"])==task,
             "Original adopted task must remain exact")
    envelope=c.read_ref(entry["result_ref"])
    need(envelope.get("job_ref")==entry["job_ref"] and isinstance(envelope.get("value"),dict)
         and c.contained_path(entry["result_ref"]["path"])==
             c.contained_path(task["output_directory"])/"result.json"
         and entry.get("actual_controller_wait")==0 and entry.get("owned_process_group_absent") is True
         and wait["actual_direct_returncode"]==0 and wait.get("owned_process_group_absent") is True
         and wait.get("supervisor_error") is None,
         "Genuine original successful result and waits required")
    need(type(entry.get("retained_bytes")) is int and type(entry.get("reservation_bytes")) is int
         and 0<=entry["retained_bytes"]<=entry["reservation_bytes"],
         "Preserve successful original retained accounting")
    # Never open the scientific value FileRef. Large native receipts have their
    # own bounded readers at downstream scientific stages.
    return dict(result_ref=entry["result_ref"],job_ref=entry["job_ref"],actual_wait_ref=wait_ref)

def bind_live(task,entry,prior_ref,snapshot_ref,state_path):
    prior_builder._job(task,entry,prior_ref)
    need(task["resource"] in ("cpu","gpu0"),"Named original CPU/GPU producer required")
    supervisor=predecessor.proc_identity(entry["pid"])
    need(supervisor is not None and supervisor["state"]!="Z","Original supervisor must remain live")
    identity_ref=c.file_ref(c.contained_path(task["output_directory"])/"supervision_identity.json")
    receipt=c.read_ref(identity_ref)
    group=receipt["child_identity"]
    actual=predecessor.proc_identity(group["pid"])
    need(receipt["job_ref"]==entry["job_ref"] and actual is not None
         and all(actual[k]==group[k] for k in ("pid","start_ticks","pgid","session"))
         and actual["ppid"]==supervisor["pid"] and actual["state"]!="Z"
         and actual["pid"]==actual["pgid"]==actual["session"],
         "Exact durable original Popen identity must remain live")
    need(entry.get("reservation_bytes")==task["max_output_bytes"],"Keep live original storage reserve")
    return dict(source_plan_ref=prior_ref,source_state_snapshot_ref=snapshot_ref,
        source_state_path=state_path,source_task_id=task["id"],source_task_fingerprint=c.fingerprint(task),
        original_job_ref=entry["job_ref"],original_supervisor_pid=supervisor["pid"],
        original_supervisor_start_ticks=supervisor["start_ticks"],
        owned_process_group_id=actual["pid"],owned_group_leader_start_ticks=actual["start_ticks"],
        owned_group_leader_session=actual["session"],original_supervision_identity_ref=identity_ref,
        resource=task["resource"],output_directory=task["output_directory"],
        max_output_bytes=task["max_output_bytes"],max_rss_bytes=task["max_rss_bytes"])

def prepare_continuation(prior_ref,snapshot_ref,*,records=RECORDS,state_path=PRIOR_STATE,
                         prior_output=PRIOR_OUTPUT,output=OUTPUT,expected_successes=157):
    prior,state=c.read_ref(prior_ref),c.read_ref(snapshot_ref)
    need(prior.get("schema")=="arggnn_executable_workflow_v1" and prior.get("physical_gpu")==0
         and state["plan_ref"]==prior_ref,"Exact original one-GPU workflow and immutable snapshot required")
    tasks=prior["tasks"];byid={t["id"]:t for t in tasks}
    need(len(byid)==len(tasks) and set(byid)==set(state["tasks"]),"Exact original task set required")
    statuses={k:e["status"] for k,e in state["tasks"].items()}
    need(set(statuses.values())<={"succeeded","failed","blocked","pending","running"},
         "Unknown source work cannot be reinterpreted")
    need(sum(s=="succeeded" for s in statuses.values())>=expected_successes,
         "All completed original stages must remain succeeded")
    old_root,new_root=c.contained_path(prior_output),c.contained_path(output)
    need(not new_root.exists() and old_root!=new_root and not old_root.is_relative_to(new_root)
         and not new_root.is_relative_to(old_root),"Fresh disjoint v9 namespace required")
    adopted={};live={};completed_predecessors={};failed=[];replacements=[];retry_templates={}
    for task in tasks:
        key=task["id"];entry=state["tasks"][key]
        if entry["status"]=="succeeded":
            need(all(statuses[d]=="succeeded" for d in task["dependencies"]),
                 "Succeeded source cannot depend on unfinished predecessor")
            adopted[key]=verify_success(task,entry,prior_ref)
            if task["resource"] == "external" and "workflow_predecessor" in task:
                # External state entries intentionally have no local retained
                # reservation. Keep their original completed binding charged.
                completed_predecessors[key] = copy.deepcopy(task["workflow_predecessor"])
            continue
        if task["resource"]=="external" and "workflow_predecessor" in task:
            binding=copy.deepcopy(task["workflow_predecessor"])
            original_task,original_entry=predecessor._source(binding)
            if original_entry["status"] not in ("failed","blocked"):
                live[key]=binding  # Flatten v3 inheritance, never double reserve.
                continue
            drained=predecessor._drained(binding,original_entry)
            need(drained is not None,"Failed inherited producer needs its genuine drained wait")
            wait_ref,wait=drained
            charge=original_entry.get("retained_bytes")
            need(type(charge) is int and 0<charge<=binding["max_output_bytes"]
                 and wait.get("output_bytes",0)<=charge,"Exact drained inherited retained charge required")
            failed.append(dict(task_id=key,scope=str(c.contained_path(binding["output_directory"]).relative_to(c.ROOT)),
                charged_bytes=charge,actual_wait_ref=wait_ref,original_entry=copy.deepcopy(original_entry)))
            retry_templates[key]=copy.deepcopy(original_task)
            retry_templates[key]["dependencies"]=copy.deepcopy(task["dependencies"])
            new_scope=new_root/"inherited"/key
            replacements.append((c.contained_path(binding["output_directory"]),new_scope))
            continue
        need(task["resource"]!="external","Original non-workflow external inputs must already succeed")
        if entry["status"]=="running":
            live[key]=bind_live(task,entry,prior_ref,snapshot_ref,state_path)
            continue
        old_scope=c.contained_path(task["output_directory"])
        need(old_scope.is_relative_to(old_root),"Unfinished current source scope must belong to v8")
        replacements.append((old_scope,new_root/old_scope.relative_to(old_root)))
        attempted=any(k in entry for k in ("job_ref","pid","started_at","reservation_bytes"))
        if attempted or old_scope.exists():
            need(entry["status"]=="failed" and type(entry.get("actual_controller_wait")) is int,
                 "No retry without original actual controller wait")
            prior_builder._job(task,entry,prior_ref)
            wait_ref,wait=prior_builder._wait(task,entry)
            need(entry.get("owned_process_group_absent") is True
                 and wait.get("owned_process_group_absent") is True,
                 "A new unknown failed producer requires separate ROOT acceptance")
            charge=entry.get("retained_bytes")
            need(type(charge) is int and 0<charge<=entry["reservation_bytes"]
                 and wait.get("output_bytes",0)<=charge,"Preserve original drained failed retained bytes")
            failed.append(dict(task_id=key,scope=str(old_scope.relative_to(c.ROOT)),
                charged_bytes=charge,actual_wait_ref=wait_ref,original_entry=copy.deepcopy(entry)))
        else:
            need(not any(k in entry for k in ("result_ref","retained_bytes")),
                 "Unattempted materialization may not claim result or storage")
    plan=copy.deepcopy(prior);entries={}
    for index,task in enumerate(plan["tasks"]):
        key=task["id"]
        if statuses[key]=="succeeded":
            entries[key]=copy.deepcopy(state["tasks"][key])
            entries[key]["adopted_from_v8"]=dict(plan_ref=prior_ref,state_snapshot_ref=snapshot_ref,**adopted[key])
        elif key in live:
            plan["tasks"][index]=dict(id=key,resource="external",dependencies=task["dependencies"],
                                      workflow_predecessor=live[key])
            entries[key]=dict(status="pending",external=True,
                preserved_original_job_ref=live[key]["original_job_ref"])
        else:
            if key in retry_templates:
                task=plan["tasks"][index]=copy.deepcopy(retry_templates[key])
            selected_scope=c.contained_path(task["output_directory"])
            own=[(before,after) for before,after in replacements if before==selected_scope]
            need(len(own)==1,"Restored original factory must have one explicit fresh retry scope")
            mapping=own+[pair for pair in replacements if pair not in own]
            task["output_directory"]=prior_builder._map_paths(task["output_directory"],mapping)
            task["factory_arguments"]=prior_builder._map_paths(task.get("factory_arguments",{}),mapping)
            need(c.contained_path(task["output_directory"])==own[0][1],
                 "Original inherited factory output must map to its fresh retry scope")
            entries[key]=dict(status="pending",external=False,prior_status=statuses[key],
                               prior_state_snapshot_ref=snapshot_ref)
    if failed:
        plan["external_storage"].append(dict(cap_bytes=sum(v["charged_bytes"] for v in failed),
            scopes=[v["scope"] for v in failed],completion_evidence=[],
            retention="Exact permanently retained genuinely drained original producer bytes"))
    plan.update(records_directory=records,operational_successor_of=prior_ref,
        adopted_prior_state_snapshot_ref=snapshot_ref,workflow_external_predecessors=[*completed_predecessors.values(), *live.values()],
        continuation_approval_key="approved_resource_continuation_ref",
        continuation_scope="Evidence-based resource admission and exact European preparation; healthy original producers remain unchanged",
        scientific_factories_called=0,scientific_payloads_rehashed=0)
    # Previous unknown storage/RAM remains inherited unchanged. Live bindings are
    # charged dynamically only, never in per-task or duplicate static reserves.
    evidence=dict(schema="arggnn_v9_preserved_v8_work_v1",source_plan_ref=prior_ref,
        source_state_snapshot_ref=snapshot_ref,adopted_succeeded=adopted,
        live_predecessors=list(live.values()),
        retained_completed_predecessors=list(completed_predecessors.values()),retained_failed_attempts=failed,
        total_failed_charge_bytes=sum(v["charged_bytes"] for v in failed),
        inherited_unknown_rss_bytes=plan.get("prior_unknown_rss_bytes",0),
        no_scientific_factory_calls=True,no_scientific_payload_hashes=True,old_sources_and_scopes_unmodified=True)
    evidence["European_preparation_routes"] = add_european_qualification_tasks(plan, entries, output=output)
    plan, evidence["resource_accounting_corrections"] = resource_corrections(plan)
    prior_ids = set(entries)
    plan, evidence["lossless_retention_routes"] = archive_routes(plan, output)
    for task in plan["tasks"]:
        if task["id"] not in prior_ids:
            need(task["id"] not in entries and task["resource"] == "cpu",
                 "Only new separately accounted CPU retention tasks may be appended")
            entries[task["id"]] = dict(status="pending", external=False, added_by_v9=True)
    return plan,entries,evidence


EUROPEAN_FACTORY = "arggnn.benchmarking.european_preparation_v9"
EUROPEAN_PARENT = "ancestralpaths_european_train_000"
EUROPEAN_ROUTES = (
    ("AP", "AP__all_train__treeloom_fit", "AP__all_test__treeloom_predict",
     "arggnn.benchmarking.european_training"),
    ("C4", "C4__all_train__c4_treeloom_fit", "C4__all_test__c4_treeloom_predict",
     "arggnn.benchmarking.european_c4_training"),
)


def archive_routes(plan, output_root):
    from research.reviews.benchmark_repairs_20261019.resources_v9.european_archive_v1 import add_archive_tasks
    return add_archive_tasks(plan, output_root)


def resource_corrections(plan):
    from research.reviews.benchmark_repairs_20261019.resources_v9.resource_accounting_v1 import apply_resource_corrections
    return apply_resource_corrections(plan)


def add_european_qualification_tasks(plan, entries, *, output=OUTPUT):
    """Add truth-free CPU gates without calling a scientific factory.

    Existing fit schedules, observed data references, sampled support and target
    bindings are copied verbatim. Only fit stages defer their actual GPU lease.
    """
    byid = {task["id"]: task for task in plan["tasks"]}
    new_tasks = {}
    changes = []
    for prefix, fit_id, predict_id, old_module in EUROPEAN_ROUTES:
        need(fit_id in byid and predict_id in byid, "Both original European fit/prediction routes required")
        fit, predict = byid[fit_id], byid[predict_id]
        need(all(entries[key]["status"] == "pending" and not entries[key].get("external")
                 for key in (fit_id, predict_id)),
             "Never replace an already completed or live European numerical route")
        need(fit["factory_module"] == predict["factory_module"] == old_module
             and fit["factory_function"] == "fit_factory"
             and predict["factory_function"] == "prediction_factory",
             "Exact original European numerical factories required")
        qual_id = prefix + "__train_000__treeloom_preparation_qualification"
        need(qual_id not in byid and qual_id not in entries, "Fresh qualification task ID required")
        arguments = fit["factory_arguments"]
        observed = {name: copy.deepcopy(arguments["train_inputs"][EUROPEAN_PARENT][name])
                    for name in ("inference_ref", "observed_vcf_ref", "sampled_registry_ref")}
        dependencies = sorted({value["$task"] for value in observed.values()})
        need(len(dependencies) == 3 and set(dependencies) <= set(byid),
             "Exact inference/generation/protocol-registry dependencies required")
        scope = c.contained_path(output) / "european_qualification" / prefix
        qualification = dict(
            id=qual_id, resource="cpu", dependencies=dependencies,
            factory_module=EUROPEAN_FACTORY, factory_function="qualification_factory",
            factory_arguments=dict(
                protocol_id=arguments["protocol_id"],
                sampling_plan_ref=copy.deepcopy(arguments["sampling_plan_ref"]),
                parent_id=EUROPEAN_PARENT, observed_inputs=observed,
                max_sites=arguments["max_sites"],
                max_metadata_bytes=arguments["max_metadata_bytes"],
                preparation_limits=copy.deepcopy(arguments["preparation_limits"]),
                max_rss_bytes=128*2**30, max_seconds=7200, max_output_bytes=16*2**20),
            max_rss_bytes=128*2**30, max_seconds=7200, max_output_bytes=16*2**20,
            output_directory=str(scope.relative_to(c.ROOT)))
        new_tasks[fit_id] = qualification
        entries[qual_id] = dict(status="pending", external=False, added_by_v9=True)
        for task in (fit, predict):
            task["factory_module"] = EUROPEAN_FACTORY
            task["factory_arguments"]["qualification_ref"] = {"$task": qual_id}
            task["dependencies"] = [*task["dependencies"], qual_id]
        need(fit["resource"] == predict["resource"] == "gpu0",
             "Original European numerical routes must be GPU0-only")
        # The original fit process starts with complete observed CPU preparation.
        # Its actual model-creation barrier acquires GPU0 in the V9 runtime.
        fit["resource"] = "cpu"
        fit["deferred_gpu0"] = True
        changes.append(dict(protocol_id=arguments["protocol_id"], qualification_task_id=qual_id,
                            fit_task_id=fit_id, prediction_task_id=predict_id,
                            original_factory_module=old_module,
                            preserved_all_original_factory_arguments=True,
                            fit_cpu_preparation_then_actual_GPU0_lease=True))
    tasks = []
    for task in plan["tasks"]:
        if task["id"] in new_tasks:
            tasks.append(new_tasks[task["id"]])
        tasks.append(task)
    plan["tasks"] = tasks
    fit_ids = {route[1] for route in EUROPEAN_ROUTES}
    old_holds = list(plan.get("dispatch_hold_task_ids", []))
    plan["dispatch_hold_task_ids"] = [key for key in old_holds if key not in fit_ids]
    if not plan["dispatch_hold_task_ids"]:
        plan.pop("dispatch_hold_reason", None)
    return dict(routes=changes,
                removed_fit_holds_replaced_by_actual_qualification=[key for key in old_holds if key in fit_ids],
                only_observed_refs_in_qualifications=True,
                scientific_factories_called=0)


def merge_source_refs(inherited, additions):
    """Retain every old path/hash verbatim; only append genuinely new pins."""
    refs={}
    for ref in inherited:
        need(ref["path"] not in refs or refs[ref["path"]]==ref,
             "Inherited source path has conflicting frozen pins")
        refs[ref["path"]]=copy.deepcopy(ref)
    for path in additions:
        actual=c.file_ref(path)
        need(path not in refs or actual==refs[path],"Never overwrite an inherited frozen source pin")
        refs.setdefault(path,actual)
    return [refs[path] for path in sorted(refs)]

def build():
    targets=(PLAN,RECORDS+"/source_v8_state_snapshot.json",RECORDS+"/preserved_v8_work.json",
             RECORDS+"/initial_state.json",RECORDS+"/state.json",RECORDS+"/freeze_result.json")
    need(all(not c.contained_path(p).exists() for p in targets),"Never overwrite frozen v9 metadata")
    prior_ref=c.file_ref(PRIOR_PLAN);policy=c.read_json(c.POLICY_PATH)
    need(policy.get("approved_cohort_continuation_ref")==prior_ref and policy.get("cohort_continuation_drain_ref")==prior_ref,
         "Keep exact v8 cohort approval and drain new dispatch before preparing v9")
    snapshot=write_new(RECORDS+"/source_v8_state_snapshot.json",c.read_json(PRIOR_STATE))
    plan,entries,evidence=prepare_continuation(prior_ref,snapshot)
    evidence_ref=write_new(RECORDS+"/preserved_v8_work.json",evidence)
    from arggnn.benchmarking import workflow
    workflow.source_check(plan["sources"])
    plan["sources"]=merge_source_refs(plan["sources"],NEW_SOURCES)
    plan["continuation_evidence_ref"]=evidence_ref
    plan_ref=write_new(PLAN,plan);workflow.load_plan(plan_ref,approved=False)
    initial=dict(plan_ref=plan_ref,tasks=entries,complete=False,run_status="prepared_not_released",
        updated_utc=datetime.now(timezone.utc).isoformat(),waiting_reasons={},undrained_failed_jobs=[])
    initial_ref=write_new(RECORDS+"/initial_state.json",initial)
    state_ref=write_new(RECORDS+"/state.json",initial)
    result=dict(schema="arggnn_v9_frozen_continuation_v1",plan_ref=plan_ref,initial_state_ref=initial_ref,
        state_ref=state_ref,source_state_snapshot_ref=snapshot,continuation_evidence_ref=evidence_ref,
        task_counts=dict(Counter(e["status"] for e in entries.values())),
        live_predecessor_resources=dict(Counter(b.get("resource","cpu") for b in evidence["live_predecessors"])),
        inherited_unknown_rss_bytes=plan.get("prior_unknown_rss_bytes",0),
        execution_released=False,no_scientific_factory_calls=True,no_scientific_payload_hashes=True)
    freeze=write_new(RECORDS+"/freeze_result.json",result)
    print(json.dumps(dict(result,freeze_result_ref=freeze),indent=2,sort_keys=True))
    return result

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze",action="store_true")
    args=parser.parse_args()
    if args.freeze:build()
    else:parser.error("Explicit --freeze required; this script never releases or launches work")
