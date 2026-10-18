"""Freeze a v5 factory correction while exact v3/v4 producers remain owned and live."""
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

PRIOR_PLAN="configs/benchmarks/executable_workflow_20261018_v4.json"
PRIOR_STATE=".runtime/benchmarking/workflow_20261018_v4/state.json"
PLAN="configs/benchmarks/executable_workflow_20261018_v5.json"
RECORDS=".runtime/benchmarking/workflow_20261018_v5"
PRIOR_OUTPUT="experiments/results/benchmark_campaign_20261018_workflow/release_v4"
OUTPUT="experiments/results/benchmark_campaign_20261018_workflow/release_v5"
SELF="scripts/build_benchmark_workflow_v5.py"
NEW_SOURCES=(SELF, "src/arggnn/benchmarking/workflow_predecessor_v5.py",
             "src/arggnn/benchmarking/workflow_v5.py",
             "src/arggnn/benchmarking/workflow_reporting_v5.py",
             "src/arggnn/benchmarking/treeloom_preparation_v5.py")
QUALIFICATION="AM_treeloom_full_parent_qualification"
FACTORY_MODULE="arggnn.benchmarking.treeloom_preparation_v5"
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
                         prior_output=PRIOR_OUTPUT,output=OUTPUT,expected_successes=139):
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
         and not new_root.is_relative_to(old_root),"Fresh disjoint v5 namespace required")
    adopted={};live={};failed=[];replacements=[];retry_templates={}
    for task in tasks:
        key=task["id"];entry=state["tasks"][key]
        if entry["status"]=="succeeded":
            need(all(statuses[d]=="succeeded" for d in task["dependencies"]),
                 "Succeeded source cannot depend on unfinished predecessor")
            adopted[key]=verify_success(task,entry,prior_ref)
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
            new_scope=new_root/"inherited"/key
            replacements.append((c.contained_path(binding["output_directory"]),new_scope))
            continue
        need(task["resource"]!="external","Original non-workflow external inputs must already succeed")
        if entry["status"]=="running":
            live[key]=bind_live(task,entry,prior_ref,snapshot_ref,state_path)
            continue
        old_scope=c.contained_path(task["output_directory"])
        need(old_scope.is_relative_to(old_root),"Unfinished current source scope must belong to v4")
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
            entries[key]["adopted_from_v4"]=dict(plan_ref=prior_ref,state_snapshot_ref=snapshot_ref,**adopted[key])
        elif key in live:
            plan["tasks"][index]=dict(id=key,resource="external",dependencies=task["dependencies"],
                                      workflow_predecessor=live[key])
            entries[key]=dict(status="pending",external=True,
                preserved_original_job_ref=live[key]["original_job_ref"])
        else:
            if key in retry_templates:
                task=plan["tasks"][index]=copy.deepcopy(retry_templates[key])
            task["output_directory"]=prior_builder._map_paths(task["output_directory"],replacements)
            task["factory_arguments"]=prior_builder._map_paths(task.get("factory_arguments",{}),replacements)
            if key in (QUALIFICATION,"AM_treeloom_fit"):
                task["factory_module"]=FACTORY_MODULE
            entries[key]=dict(status="pending",external=False,prior_status=statuses[key],
                               prior_state_snapshot_ref=snapshot_ref)
    if failed:
        plan["external_storage"].append(dict(cap_bytes=sum(v["charged_bytes"] for v in failed),
            scopes=[v["scope"] for v in failed],completion_evidence=[],
            retention="Exact permanently retained genuinely drained original producer bytes"))
    plan.update(records_directory=records,operational_successor_of=prior_ref,
        adopted_prior_state_snapshot_ref=snapshot_ref,workflow_external_predecessors=list(live.values()),
        continuation_approval_key="approved_repair_continuation_ref",
        continuation_scope="Qualification factory lineage correction; healthy v3/v4 producers remain unchanged",
        scientific_factories_called=0,scientific_payloads_rehashed=0)
    # Previous unknown storage/RAM remains inherited unchanged. Live bindings are
    # charged dynamically only, never in per-task or duplicate static reserves.
    evidence=dict(schema="arggnn_v5_preserved_v4_work_v1",source_plan_ref=prior_ref,
        source_state_snapshot_ref=snapshot_ref,adopted_succeeded=adopted,
        live_predecessors=list(live.values()),retained_failed_attempts=failed,
        total_failed_charge_bytes=sum(v["charged_bytes"] for v in failed),
        inherited_unknown_rss_bytes=plan.get("prior_unknown_rss_bytes",0),
        no_scientific_factory_calls=True,no_scientific_payload_hashes=True,old_sources_and_scopes_unmodified=True)
    return plan,entries,evidence

def build():
    targets=(PLAN,RECORDS+"/source_v4_state_snapshot.json",RECORDS+"/preserved_v4_work.json",
             RECORDS+"/initial_state.json",RECORDS+"/state.json",RECORDS+"/freeze_result.json")
    need(all(not c.contained_path(p).exists() for p in targets),"Never overwrite frozen v5 metadata")
    prior_ref=c.file_ref(PRIOR_PLAN);policy=c.read_json(c.POLICY_PATH)
    need(policy.get("approved_continuation_ref")==prior_ref and policy.get("continuation_drain_ref")==prior_ref,
         "Keep v4 approval unchanged and drain new dispatch before preparing v5")
    snapshot=write_new(RECORDS+"/source_v4_state_snapshot.json",c.read_json(PRIOR_STATE))
    plan,entries,evidence=prepare_continuation(prior_ref,snapshot)
    evidence_ref=write_new(RECORDS+"/preserved_v4_work.json",evidence)
    from arggnn.benchmarking import workflow
    workflow.source_check(plan["sources"])
    paths={r["path"] for r in plan["sources"]}|set(NEW_SOURCES)
    plan["sources"]=[c.file_ref(p) for p in sorted(paths)]
    plan["continuation_evidence_ref"]=evidence_ref
    plan_ref=write_new(PLAN,plan);workflow.load_plan(plan_ref,approved=False)
    initial=dict(plan_ref=plan_ref,tasks=entries,complete=False,run_status="prepared_not_released",
        updated_utc=datetime.now(timezone.utc).isoformat(),waiting_reasons={},undrained_failed_jobs=[])
    initial_ref=write_new(RECORDS+"/initial_state.json",initial)
    state_ref=write_new(RECORDS+"/state.json",initial)
    result=dict(schema="arggnn_v5_frozen_continuation_v1",plan_ref=plan_ref,initial_state_ref=initial_ref,
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
