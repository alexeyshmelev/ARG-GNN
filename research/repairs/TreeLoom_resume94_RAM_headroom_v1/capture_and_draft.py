"""Capture the failed worker durable94 boundary and prepare an inert RAM-only repair."""
from pathlib import Path
import copy
import hashlib
import json
import os
import shutil
import statistics
import time
import torch
from arggnn.benchmarking import contracts as c
from research.repairs.TreeLoom_lazy_compact_prefetch_20261024_v1 import training

BASE=Path("research/repairs/TreeLoom_resume94_RAM_headroom_v1")
GIB=2**30

def ref(path):
    p=Path(path)
    need(p.is_file() and not p.is_symlink(),"Ordinary existing file required")
    h=hashlib.sha256()
    with p.open("rb") as stream:
        for block in iter(lambda:stream.read(2**20),b""):h.update(block)
    return dict(path=str(p),sha256=h.hexdigest())

def need(ok,message):
    if not ok:raise ValueError(message)

def write(name,value):
    p=BASE/name
    with p.open("x") as stream:
        json.dump(value,stream,sort_keys=True,indent=2);stream.write("\n")
        stream.flush();os.fsync(stream.fileno())
    return ref(p)

def main():
    torch.set_num_threads(1)
    state=json.loads(Path(".runtime/benchmarking/paper_campaign_20261024_v2/state.json").read_text())
    row=state["tasks"]["paper_american_treeloom_fit"]
    need(row["status"]=="failed" and row["attempt"]==2 and row["actual_wait"] is True
         and row["owned_process_group_absent"] is True and row["actual_returncode"]==1,
         "Exact failed and actually waited attempt2 required")
    proc=Path("/proc")/str(row["pid"])/"stat"
    if proc.exists():
        fields=proc.read_text().rsplit(") ",1)[1].split()
        need(fields[19]!=row["process_start_ticks"],"Original failed numerical worker must be absent")
    job_path=Path(".runtime/benchmarking/paper_campaign_20261024_v2/jobs/paper_american_treeloom_fit.attempt2.json")
    job=json.loads(job_path.read_text());arguments=job["call"]["spec"]["arguments"]
    out=Path(arguments["output_directory"])
    latest_path=out/"latest_checkpoint.json"
    latest=json.loads(latest_path.read_text())
    need(latest["complete_parent_updates"]==94,"Actual latest durable94 metadata required")
    source=Path(latest["checkpoint_ref"]["path"])
    need(source.stat().st_size<=8*2**20 and ref(source)==latest["checkpoint_ref"],"Bounded exact original latest checkpoint required")
    target=BASE/"checkpoint.pt"
    with source.open("rb") as src,target.open("xb") as dest:
        shutil.copyfileobj(src,dest,2**20);dest.flush();os.fsync(dest.fileno())
    captured=ref(target)
    need(ref(source)==latest["checkpoint_ref"] and captured["sha256"]==latest["checkpoint_ref"]["sha256"],"Source must remain unchanged during capture")
    registry=training.fresh.FreshParentRegistry.from_ref(arguments["parent_registry_ref"])
    model=training.fresh.FreshTreeLoom(registry).build_model()
    saved=torch.load(target,map_location="cpu",weights_only=True,mmap=True)
    steps=training.validate_saved_state(saved,model,registry,expected_steps=94)
    schedule=[(epoch,parent) for epoch in range(1,31) for parent in registry.epoch_order(epoch)]
    need(saved["next_schedule_index"]==steps and saved["schedule_identity"]==training.schedule_identity(schedule)
         and saved["last_completed_parent"]==schedule[steps-1][1],"Exact original schedule94 boundary required")
    sources=saved["execution_requirements"]["source_refs"]
    for reference in sources:need(ref(reference["path"])==reference,"Actual failed-worker source pin changed")
    snapshot_ref=write("failed_attempt_snapshot.json",dict(schema="TreeLoom_failed_attempt2_capture_snapshot_v1",
        observed_unix=time.time(),plan_ref=state["plan_ref"],plan_revision=state.get("plan_revision",0),task=copy.deepcopy(row),
        original_job_ref=ref(job_path),original_latest_metadata_ref=ref(latest_path),original_checkpoint_ref=latest["checkpoint_ref"]))
    evidence=dict(schema="treeloom_durable_optimizer_boundary_v2",captured_unix=time.time(),
        checkpoint_ref=captured,original_checkpoint_ref=latest["checkpoint_ref"],original_job_ref=ref(job_path),
        failed_attempt_snapshot_ref=snapshot_ref,parent_registry_ref=registry.reference,
        complete_parent_updates=steps,next_schedule_index=steps,original_last_completed_parent=seq_parent(schedule,steps-1),
        next_parent=seq_parent(schedule,steps),observation_barrier_ref=saved["observation_barrier_ref"],
        durable_complete_parent_optimizer_boundary=True,no_inflight_update_in_serialized_state=True,
        owned_numerical_session_absent=True,source_refs=sources,VAL_truth_read=False,TEST_truth_read=False,
        unrelated_sessions_signalled=False,previous_resume_checkpoint_ref=saved["resume_checkpoint_ref"],
        previous_resume_evidence_ref=saved["resume_evidence_ref"])
    evidence_ref=write("evidence.json",evidence)
    successor=copy.deepcopy(job)
    successor["max_rss_bytes"]=768*GIB
    args=successor["call"]["spec"]["arguments"]
    args["max_rss_bytes"]=768*GIB;args["resume_checkpoint_ref"]=captured;args["resume_evidence_ref"]=evidence_ref
    need(args["observation_cache_bytes"]==352*GIB and args["preparation_headroom_bytes"]==args["training_headroom_bytes"]==128*GIB,
         "Only process RAM ceiling and resume boundary may change")
    probe=copy.deepcopy(args);probe["output_directory"]="experiments/results/TreeLoom_resume94_metadata_probe/native"
    spec=training.build_spec(**probe)
    need(training.validate_resume_checkpoint(saved,model,registry,spec)==94,"Captured94 continuation must pass full existing resume validator")
    replacement_ref=write("replacement_task.json",successor)
    events=[json.loads(line) for line in (out/"progress.jsonl").read_text().splitlines()]
    updates=[r for r in events if r["event"]=="complete_parent_update"]
    ready=[r for r in events if r["event"]=="compact_parent_ready"]
    last_ready=ready[-1]["observed_unix"]
    warm=[r for r in updates if r["observed_unix"]>last_ready]
    warm_gaps=[b["observed_unix"]-a["observed_unix"] for a,b in zip(warm,warm[1:])]
    last=updates[-1]
    report=dict(schema="TreeLoom_resume94_RAM_headroom_delivery_v1",status="qualified_metadata_ready_for_ROOT_owned_repair",
        recorded_unix=time.time(),checkpoint_ref=captured,evidence_ref=evidence_ref,replacement_task_ref=replacement_ref,
        failed_attempt_snapshot_ref=snapshot_ref,source_ref=ref(__file__),
        existing_frozen_source_pins_ref=ref("research/repairs/TreeLoom_lazy_compact_prefetch_20261024_v1/source_pins.json"),
        actual_saved_updates=94,optimizer_states=49,all_saved_Adam_steps=94,next_parent=schedule[94][1],remaining_updates=506,total_budget=600,
        full_resume_validator_passed=True,metadata_build_spec_passed=True,GPU_initialized=torch.cuda.is_initialized(),
        failure_before_update95_labels_and_train_step=True,failure="Complete-parent training headroom admission at512GiB after next bank restore",
        observed_failed_worker_PSS_bytes=row["observed_pss_bytes"],admission_old_RSS_threshold_bytes=(512-128)*GIB,
        compact_records_ready=len(ready),compact_actual_charged_bytes=ready[-1]["pool_charged_bytes"],
        previous_latest_parent_peak_RSS_bytes=json.loads((out/"training_live.json").read_text())["peak_RSS_bytes"],
        recommended_process_RSS_limit_bytes=768*GIB,reserves_and_cache_unchanged=True,
        concurrency_conservative_upper_bytes=373741732426+40197312104+2*128*GIB,
        recommendation="768GiB admits the observed mature pool plus complete-parent headroom and conservatively permits one producer/training overlap. Preserve fail-stop guards. This is an operational allocation, not a proven global allocator peak.",
        latest_parent_stage_seconds=last["stage_seconds"],warm_cache_update_gap_median_seconds=statistics.median(warm_gaps),
        warm_cache_updates_observed=len(warm),warm_cache_gap_count=len(warm_gaps),
        cold_first20_preparation_lost_with_failed_process=True,
        numerical_updates_lost=0,source_changed=False,model_objective_schedule_domain_unchanged=True,
        observation_disk_cache=False,production_GPU_utilization_percent_proven=False,
        no_signals_no_archive_no_launch=True,same_output_paths_and_prediction_dependencies=True,
        operational_limitations="The process-local compact banks died with attempt2; restart must prepare them again. Wall-clock stages do not establish GPU occupancy. ROOT must account for other host reservations and archive the old scope before repair.")
    delivery_ref=write("delivery.json",report)
    print(json.dumps(dict(delivery_ref=delivery_ref,checkpoint_ref=captured,evidence_ref=evidence_ref,
        replacement_task_ref=replacement_ref,next_parent=schedule[94][1],remaining_updates=506),indent=2))

def seq_parent(schedule,index):return schedule[index][1]

if __name__=="__main__":main()
