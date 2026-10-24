"""Build an inert replacement task; never signal, launch or alter live outputs."""
import copy,json
from pathlib import Path
from arggnn.benchmarking import contracts as c
from research.repairs.TreeLoom_lazy_compact_prefetch_20261024_v1 import training
base=Path("research/repairs/TreeLoom_lazy_compact_prefetch_20261024_v1")
plan=json.loads(Path("configs/benchmarks/paper_campaign_20261024_v2.json").read_text())
state=json.loads(Path(".runtime/benchmarking/paper_campaign_20261024_v2/state.json").read_text())
old=next(t for t in plan["tasks"] if t["id"]=="paper_american_treeloom_fit")
task=copy.deepcopy(old)
task["call"]["module"]="research.repairs.TreeLoom_lazy_compact_prefetch_20261024_v1.training"
task["call"]["function"]="run_task"
task["cpu_threads"]=1
arguments=task["call"]["spec"]["arguments"]
assert arguments["max_rss_bytes"]==task["max_rss_bytes"]==512*2**30
assert arguments["observation_cache_bytes"]==352*2**30
assert arguments["preparation_headroom_bytes"]==arguments["training_headroom_bytes"]==128*2**30
probe=copy.deepcopy(arguments)
probe["output_directory"]="experiments/results/paper_american_20261024/TreeLoom_repair_metadata_probe/native"
spec=training.build_spec(**probe)
by_path={r["path"]:r for r in task["source_refs"]}
for reference in spec["requirements"]["source_refs"]:
    if reference["path"] in by_path:assert by_path[reference["path"]]==reference
    by_path[reference["path"]]=reference
task["source_refs"]=[by_path[k] for k in sorted(by_path)]
(base/"replacement_task.json").write_text(json.dumps(task,indent=2)+"\n")
qualified=c.read_json("research/reviews/workflow_progress_diagnosis_20261018/treeloom/storage_bound_v2.json")
report=dict(schema="TreeLoom_lazy_compact_prefetch_transition_draft_v1",status="awaiting_independent_review_and_ROOT_owned_migration",
    replacement_task_ref=c.file_ref(base/"replacement_task.json"),source_pins_ref=c.file_ref(base/"source_pins.json"),
    CPU_qualification_ref=c.file_ref(base/"cpu_smoke_v2/completed.json"),original_plan_ref=state["plan_ref"],
    observed_plan_revision=state.get("plan_revision",0),old_task_id=task["id"],
    observed_old_PID=state["tasks"][task["id"]]["pid"],
    observed_old_start_ticks=state["tasks"][task["id"]]["process_start_ticks"],
    output_scope_preserved=task["output_directory"],native_output_preserved=arguments["output_directory"],
    no_prediction_path_rewrite_required=True,resume_checkpoint_ref=arguments["resume_checkpoint_ref"],
    resume_evidence_ref=arguments["resume_evidence_ref"],saved_complete_parent_updates=58,
    next_parent="argformer_american_train_013",remaining_updates=542,max_seconds=task["max_seconds"],
    max_rss_bytes=task["max_rss_bytes"],compact_all20_bound_plus_metadata_bytes=qualified["compact_all20_upper_bound_bytes"]+20*160000,
    compact_cap_bytes=arguments["observation_cache_bytes"],no_disk_observation_cache=True,
    metadata_probe_passed=True,metadata_probe_output_created=False,
    old_output_must_be_archived_by_ROOT_before_repair=True,old_process_not_signalled=True,jobs_started=0,
    qualifications_are_CPU_synthetic_only=True,production_GPU_speed_claimed=False)
(base/"transition_draft.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(dict(replacement_task_ref=c.file_ref(base/"replacement_task.json"),
    transition_draft_ref=c.file_ref(base/"transition_draft.json"),source_refs=len(task["source_refs"]),build_spec_passed=True),indent=2))
