"""Default-read-only AP prediction repair preparation and explicit ROOT apply.

Run without --apply to inspect. ROOT may use --apply only after review and the
combined remaining-host-growth/GPU admission check. This helper never starts a
worker, changes the completed fit, signals a process, or changes a live source.
Publishing the repair pointer lets the existing controller schedule normally.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
BASE=ROOT/"research/repairs/benchmark_recovery_20261025/european"
DIRECTORY=ROOT/".runtime/benchmarking/paper_campaign_20261024_v2"
TARGET="paper_ap_treeloom__all_test__predict"
SCORE="paper_gst_lai_8__all_methods_all_test__score"
FIT="paper_ap_treeloom_fit_existing"
REQUEST_ID="ap_prediction_bounded_gru_v1"
MODULE="research.repairs.benchmark_recovery_20261025.european.ap_prediction_chunked_gru"


def path(value):
    p=Path(value)
    p=p if p.is_absolute() else ROOT/p
    p=p.resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError("Project-contained path required")
    return p


def ref(value):
    p=path(value)
    return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def read(reference):
    if ref(reference["path"])!=reference:
        raise ValueError("Pinned file changed: "+reference["path"])
    return json.loads(path(reference["path"]).read_text())


def save_exact(filename,value):
    p=path(filename)
    text=json.dumps(value,indent=2)+"\n"
    if p.exists():
        if p.read_text()!=text:
            raise ValueError("Existing immutable repair file differs: "+str(p))
        return ref(p)
    p.parent.mkdir(parents=True,exist_ok=True)
    temporary=p.with_name(p.name+".prepared."+str(os.getpid()))
    with temporary.open("x") as stream:
        stream.write(text);stream.flush();os.fsync(stream.fileno())
    try:
        os.link(temporary,p)
    except FileExistsError:
        if p.read_text()!=text:
            raise ValueError("Concurrent immutable repair file differs: "+str(p))
    finally:
        temporary.unlink()
    return ref(p)


def inventory(directory):
    directory=path(directory)
    if directory.is_symlink():
        raise ValueError("Archive scope cannot be a symbolic link")
    result=[]
    for p in sorted(directory.rglob("*")):
        if p.is_symlink():
            raise ValueError("Unexpected symbolic link in exact archive scope")
        if p.is_file():
            result.append(dict(relative_path=str(p.relative_to(directory)),
                bytes=p.stat().st_size,sha256=ref(p)["sha256"]))
    return result


def effective_plan(state):
    """Replay immutable applied request replacements without changing state."""
    from arggnn.benchmarking.paper_campaign import repairs
    plan=read(state["plan_ref"])
    receipts=[read(value) for value in state.get("applied_repairs",{}).values()]
    receipts.sort(key=lambda r:r["resulting_plan_revision"])
    revision=0
    for receipt in receipts:
        if (receipt["status"]!="applied" or receipt["plan_ref"]!=state["plan_ref"]
                or receipt["prior_plan_revision"]!=revision
                or receipt["prior_plan_sha256"]!=repairs._digest(plan)):
            raise ValueError("Applied repair history is not an exact ordered plan replay")
        request=read(receipt["request_ref"])
        replacements={task["id"]:task for task in request["replacement_tasks"]}
        plan["tasks"]=[copy.deepcopy(replacements.get(task["id"],task))
                       for task in plan["tasks"]]
        revision=receipt["resulting_plan_revision"]
        if receipt["resulting_plan_sha256"]!=repairs._digest(plan):
            raise ValueError("Applied repair resulting plan digest differs")
    if revision!=state.get("plan_revision",0):
        raise ValueError("Effective plan replay does not reach current revision")
    return plan


def prepare(expected_revision):
    manifest=json.loads((BASE/"patch_manifest.json").read_text())
    for pinned in manifest["pins"]:
        if ref(pinned["path"])!=pinned:
            raise ValueError("Repair source/evidence pin changed: "+pinned["path"])
    archive=read(manifest["archive_plan_ref"])
    validation=read(manifest["checkpoint_validation_ref"])
    if validation["status"]!="usable_final_checkpoint" or validation["optimizer_updates"]!=90:
        raise ValueError("The completed90-update fit must remain usable")
    if ref(validation["checkpoint_ref"]["path"])!=validation["checkpoint_ref"]:
        raise ValueError("Final fit checkpoint changed")
    state=json.loads((DIRECTORY/"state.json").read_text())
    if state["plan_ref"]!=manifest["plan_ref"]:
        raise ValueError("Campaign immutable plan changed")
    if state["tasks"][FIT]["status"]!="succeeded":
        raise ValueError("Completed AP fit must remain succeeded")
    if REQUEST_ID in state.get("applied_repairs",{}):
        return None,dict(status="already_applied",receipt_ref=state["applied_repairs"][REQUEST_ID])
    if state["tasks"][TARGET]["status"]!="failed" or state["tasks"][SCORE]["status"]!="blocked":
        raise ValueError("Only this failed prediction and blocked score may reset")
    row=state["tasks"][TARGET]
    if not row.get("actual_wait") or not row.get("owned_process_group_absent"):
        raise ValueError("Actual failed-worker wait/cleanup evidence required")
    for key in ("pid","process_start_ticks","finished_unix","actual_returncode"):
        if row.get(key)!=archive["original_task_row"].get(key):
            raise ValueError("Failed attempt identity changed")
    from arggnn.benchmarking.paper_campaign import processes
    if processes.session_members(row["pid"]):
        raise ValueError("Failed task session still has live members")
    old_scope=path(archive["source_directory"])
    destination=path(archive["destination_directory"])
    if old_scope.exists() and destination.exists():
        raise ValueError("Archive destination already exists beside old scope")
    current_scope=old_scope if old_scope.exists() else destination
    if not current_scope.exists() or inventory(current_scope)!=archive["files"]:
        raise ValueError("Exact failed-attempt inventory changed")
    plan=effective_plan(state)
    old=next(t for t in plan["tasks"] if t["id"]==TARGET)
    job=read(manifest["failed_job_ref"])
    kwargs=copy.deepcopy(job["call"]["factory"]["kwargs"])
    from research.repairs.benchmark_recovery_20261025.european import ap_prediction_chunked_gru as repair
    # Fresh-output guard is intentional. A hypothetical probe creates no path;
    # restore its output field before comparing with the immutable failed call.
    probe=copy.deepcopy(kwargs)
    probe["output_directory"]="experiments/results/paper_campaign_20261024/ap_treeloom_readonly_factory_probe/native"
    if path(probe["output_directory"]).exists():
        raise ValueError("Read-only factory probe must be absent")
    concrete=repair.prediction_factory(**probe)
    prediction=copy.deepcopy(concrete["spec"]["prediction_spec"])
    prediction["output_directory"]=kwargs["output_directory"]
    prior=json.loads((current_scope/"concrete_call.json").read_text())
    if prediction!=prior["spec"]["prediction_spec"]:
        raise ValueError("Prediction scientific/input/resource specification changed")
    replacement=copy.deepcopy(old)
    replacement["call"]=dict(factory=dict(module=MODULE,function="prediction_factory",kwargs=kwargs))
    sources=[*old["source_refs"],*repair.requirements()["source_refs"]]
    replacement["source_refs"]=list({r["path"]:r for r in sources}.values())
    if any(old[k]!=replacement[k] for k in old if k not in ("call","source_refs")):
        raise ValueError("Task resources, dependencies or scopes changed")
    for pinned in replacement["source_refs"]:
        if ref(pinned["path"])!=pinned:
            raise ValueError("Replacement source changed")
    from arggnn.benchmarking.paper_campaign.scheduler import validate
    candidate=copy.deepcopy(plan)
    candidate["tasks"]=[replacement if t["id"]==TARGET else t for t in candidate["tasks"]]
    validate(candidate)
    for t in candidate["tasks"]:
        if t["id"]==SCORE and path(t["output_directory"]).exists():
            raise ValueError("Blocked score scope unexpectedly exists")
    archive_actual_path=BASE/"archive_actual.json"
    archive_ref=ref(archive_actual_path) if archive_actual_path.exists() else None
    request=dict(schema="paper_campaign_repair_request_v1",request_id=REQUEST_ID,
        expected_plan_ref=state["plan_ref"],expected_plan_revision=expected_revision,
        expected_attempts={name:state["tasks"][name].get("attempt",1) for name in (TARGET,SCORE)},
        reset_task_ids=[TARGET,SCORE],replacement_tasks=[replacement],
        archived_bytes=archive["archived_bytes"],archive_manifest_ref=archive_ref,
        reason="Prediction-only bounded1024-hold bidirectional GRU execution, matching completed AP fit. Reuse authenticated epoch30/90-update checkpoint; preserve every native observation, sampled row, full history, class and resource cap. No retraining. Reset only failed prediction and blocked score.")
    summary=dict(status="read_only_prepared",current_plan_revision=state.get("plan_revision",0),
        expected_plan_revision=expected_revision,revision_ready=state.get("plan_revision",0)==expected_revision,
        failed_scope=str(old_scope.relative_to(ROOT)),archive_destination=str(destination.relative_to(ROOT)),
        failed_scope_bytes=archive["archived_bytes"],archive_already_moved=not old_scope.exists(),
        failed_scope_files=len(archive["files"]),task_ids=request["reset_task_ids"],
        all_prediction_spec_fields_unchanged=True,all_task_resource_scope_fields_unchanged=True,
        scheduler_candidate_valid=True,checkpoint_ref=validation["checkpoint_ref"],
        max_rss_bytes=replacement["max_rss_bytes"],GPU_authority=[3,4,5],
        source_pins_verified=len(replacement["source_refs"]),
        effective_plan_replayed_through_revision=state.get("plan_revision",0),
        failed_session_members_currently_absent=True,
        repairs_prepare_check="Performed after exact archive and before live pointer publication on --apply",
        apply_requires="ROOT independent review and verified combined remaining-growth/GPU admission")
    return request,summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-plan-revision",type=int,default=6)
    parser.add_argument("--apply",action="store_true",
        help="ROOT only: archive exact failed scope and publish repair after review/headroom verification")
    args=parser.parse_args()
    request,summary=prepare(args.expected_plan_revision)
    if request is None or not args.apply:
        print(json.dumps(summary,indent=2))
        return
    if not summary["revision_ready"]:
        raise ValueError("Current campaign revision differs from explicit expected revision")
    manifest=json.loads((BASE/"patch_manifest.json").read_text())
    archive=read(manifest["archive_plan_ref"])
    source,destination=path(archive["source_directory"]),path(archive["destination_directory"])
    from arggnn.benchmarking.paper_campaign import processes
    if processes.session_members(archive["original_task_row"]["pid"]):
        raise ValueError("Failed session reappeared before archive")
    if source.exists():
        destination.parent.mkdir(parents=True,exist_ok=True)
        os.rename(source,destination)
    if source.exists() or inventory(destination)!=archive["files"]:
        raise ValueError("Failed scope archive did not preserve exact inventory")
    actual=dict(schema="AP_prediction_failed_scope_archive_actual_v1",
        status="archived_exact_failed_scope",source_directory=archive["source_directory"],
        archive_directory=archive["destination_directory"],archived_bytes=archive["archived_bytes"],
        files=archive["files"],archive_plan_ref=manifest["archive_plan_ref"],
        original_task_row=archive["original_task_row"],failure_log_ref=archive["failure_log_ref"],
        completed_fit_untouched=True,checkpoint_ref=summary["checkpoint_ref"])
    request["archive_manifest_ref"]=save_exact(BASE/"archive_actual.json",actual)
    latest=json.loads((DIRECTORY/"state.json").read_text())
    if (latest["plan_ref"]!=request["expected_plan_ref"]
            or latest.get("plan_revision",0)!=args.expected_plan_revision
            or any(latest["tasks"][name].get("attempt",1)!=attempt
                   for name,attempt in request["expected_attempts"].items())):
        raise ValueError("Campaign revision/attempts changed before publication")
    # Complete the actual controller validator against a research-only pointer.
    # The running controller never scans this inert directory.
    from arggnn.benchmarking.paper_campaign import repairs
    dry_directory=BASE/"inert_controller_preflight"
    dry_body=dry_directory/"repair_requests"/(REQUEST_ID+".request.json")
    dry_pointer=dry_directory/"repair_requests"/(REQUEST_ID+".ref.json")
    dry_ref=save_exact(dry_body,request)
    save_exact(dry_pointer,dry_ref)
    candidate,receipt=repairs._prepare(effective_plan(latest),latest,
                                     dry_directory,dry_pointer)
    if set(receipt["reset_task_ids"])!={TARGET,SCORE} or receipt["status"]!="applied":
        raise ValueError("Actual repairs._prepare reset scope differs")
    save_exact(BASE/"repair_prepare_preflight.json",dict(
        schema="AP_actual_repair_prepare_inert_validation_v1",
        status="passed_before_live_pointer_publication",
        expected_plan_revision=args.expected_plan_revision,
        request_ref=dry_ref,resulting_plan_sha256=repairs._digest(candidate),
        reset_task_ids=receipt["reset_task_ids"],training_repeated=False))
    body=DIRECTORY/"repair_requests"/(REQUEST_ID+".request.json")
    submission=DIRECTORY/"repair_requests"/(REQUEST_ID+".ref.json")
    body_ref=save_exact(body,request)
    submission_ref=save_exact(submission,body_ref)
    print(json.dumps(dict(status="repair_submitted_to_existing_controller",
        request_ref=body_ref,submission_ref=submission_ref,
        direct_workers_launched=0,training_repeated=False),indent=2))


if __name__=="__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps(dict(status="rejected",error_type=type(error).__name__,error=str(error))),flush=True)
        raise
