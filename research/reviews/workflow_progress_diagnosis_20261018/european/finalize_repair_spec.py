"""Finalize the reviewed metadata-only repair record. No scientific stages."""
from pathlib import Path
import copy
import hashlib
import json
from datetime import datetime,timezone

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
OUT=ROOT/"research/reviews/workflow_progress_diagnosis_20261018/european"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path):
    path=Path(path)
    path=path if path.is_absolute() else ROOT/path
    if not path.is_relative_to(ROOT) or any(
            "truth" in v.lower() or "oracle" in v.lower() for v in path.parts):
        raise ValueError("Only project metadata and observed paths allowed")
    return dict(path=str(path.relative_to(ROOT)),sha256=digest(path))


def write(name,value):
    path=OUT/name
    with path.open("x") as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write("\n")
    return ref(path)


def run():
    raw_path=OUT/"observed_coordinate_cap_smoke.json"
    raw=json.loads(raw_path.read_text())
    state_path=ROOT/".runtime/benchmarking/workflow_20261018_v2/state.json"
    state_bytes=state_path.read_bytes()
    state=json.loads(state_bytes)
    plan=json.loads((ROOT/state["plan_ref"]["path"]).read_text())
    selected={}
    anchors={}
    for row in raw["parents"]:
        failure=state["tasks"][row["failed_task_id"]]
        generation=state["tasks"][row["generation_task_id"]]
        if (failure["job_ref"]!=row["failed_job_ref"]
                or generation["result_ref"]!=row["generation_result_ref"]
                or generation["status"]!="succeeded"):
            raise ValueError("Original failed attempt or successful anchor changed")
        for name in ("failed_actual_wait_ref","failed_stage_log_ref"):
            record=row[name]
            if ref(ROOT/record["path"])!=record:
                raise ValueError("Preserved failure evidence changed")
        selected[row["failed_task_id"]]=failure
        anchors[row["generation_task_id"]]=generation
    snapshot_path=OUT/"failed_attempts_snapshot.json"
    if snapshot_path.exists():
        previous=json.loads(snapshot_path.read_text())
        if previous["failed_tasks"]!=selected or previous["successful_generation_tasks"]!=anchors:
            raise ValueError("Existing immutable failure snapshot differs")
        snapshot_ref=ref(snapshot_path)
    else:
        snapshot_ref=write("failed_attempts_snapshot.json",dict(
            schema="European_cap_repair_immutable_failure_snapshot_v1",
            captured_utc=datetime.now(timezone.utc).isoformat(),
            plan_ref=state["plan_ref"],failed_tasks=selected,successful_generation_tasks=anchors,
            live_state_read_fingerprint=dict(path=str(state_path.relative_to(ROOT)),
                                            sha256=hashlib.sha256(state_bytes).hexdigest()),
            only_original_failed_inference_and_completed_generation_entries=True))
    reviewed=copy.deepcopy(raw)
    reviewed["schema"]="European_observed_coordinate_cap_reviewed_smoke_v1"
    reviewed["original_execution_report_ref"]=ref(raw_path)
    reviewed["live_state_read_fingerprint"]=reviewed.pop("state_snapshot_ref")
    reviewed["live_state_fingerprint_is_not_a_replayable_immutable_ref"]=True
    reviewed["state_snapshot_ref"]=snapshot_ref
    reviewed["state_snapshot_kind"]="immutable subset of all9failed inference attempts and9successful generation anchors"
    reviewed["review_receipt_correction_only_no_smoke_result_changed"]=True
    reviewed["finalization_source_ref"]=ref(Path(__file__))
    reviewed_ref=write("reviewed_coordinate_cap_smoke.json",reviewed)

    anchor_ids=set(anchors)
    european=[t for t in plan["tasks"]
              if t["id"].startswith(("AP__","C4__","EMP__"))]
    downstream=[t for t in european if t["id"] not in anchor_ids]
    if len(european)!=89 or len(downstream)!=80:
        raise ValueError("Exact original89-stage European/empirical branch required")
    required=[]
    for task in downstream:
        infer=task["id"] in selected
        required.append(dict(task_id=task["id"],resource=task["resource"],
            dependencies=task["dependencies"],factory_module=task["factory_module"],
            factory_function=task["factory_function"],
            original_task_sha256=hashlib.sha256(json.dumps(task,sort_keys=True,
                separators=(",",":"),allow_nan=False).encode()).hexdigest(),
            fresh_output_scheme="<ROOT-v3-OUTPUT>/"+task["output_directory"].split("/release_v2/",1)[-1],
            required_input_metadata_cap_bytes=256*2**20 if infer else None,
            outputs_reused=False))
    record=dict(native_site_id=1342572,VCF_position_1based=200000000,
                original_position_0based=199999999,local_position_0based=199999999.0,
                original_site_id=1342572)
    one=len(json.dumps(dict(post_five_iterations=dict(site_join=[record])),indent=2,sort_keys=True))
    two=len(json.dumps(dict(post_five_iterations=dict(site_join=[record,record])),indent=2,sort_keys=True))
    site_join_bound=one+(max(r["expected_sites"] for r in raw["parents"])-1)*(two-one)
    spec=dict(schema="European_metadata_read_cap_repair_spec_v1",
        status="reviewed_metadata_only_not_activated",source_plan_ref=state["plan_ref"],
        reviewed_smoke_ref=reviewed_ref,immutable_failure_snapshot_ref=snapshot_ref,
        root_cause=dict(file="src/arggnn/benchmarking/european_native_relate.py",
            line=31,caller_line=97,
            exception="ValueError: Observed metadata exceeds its explicit read cap",
            failed_input="site_coordinate_manifest.json",
            old_cap_bytes=128*2**20,validated_input_cap_bytes=256*2**20,
            observed_manifest_bytes=[min(r["size_bytes"] for r in raw["parents"]),
                                     max(r["size_bytes"] for r in raw["parents"])]),
        parser_result=dict(parents_passed=9,
            per_parent_sites=[min(r["expected_sites"] for r in raw["parents"]),
                              max(r["expected_sites"] for r in raw["parents"])],
            exact_schema_and_all_row_predicates=True,
            original_reader_used=True,only_one_manifest_object_held_at_a_time=True,
            coordinate_SHA_before_and_after_matches_original=True),
        successful_generation_inputs=[dict(parent_id=r["parent_id"],
            original_seeds=r["original_seeds"],original_recipe_ref=r["original_recipe_ref"],
            generation_task_id=r["generation_task_id"],
            generation_result_ref=r["generation_result_ref"],
            observed_input_refs=r["observed_input_refs"],coordinate_ref=r["coordinate_ref"],
            coordinate_bytes=r["size_bytes"]) for r in raw["parents"]],
        scientific_preservation=dict(resimulation=False,seed_changes=False,
            recipe_changes=False,observed_input_hash_changes=False,
            reference_or_query_selection_changes=False,Relate_parameter_changes=False,
            training_or_model_changes=False,
            failed_files_deleted_or_overwritten=False,truth_access=False),
        activation=dict(owner="ROOT",route="integrate into reviewed single-controller v3",
            separate80task_controller_requested=False,
            preserve_all139ROOT_registered_successes=True,
            generation_anchors_to_reuse=sorted(anchor_ids),
            fresh_scope_scheme="<ROOT-v3-OUTPUT>/<original family>/<parent>/<stage>",
            every_failed_inference_scope_preserved_read_only=True,
            launch_policy_or_Git_performed=False),
        required_downstream_task_count=80,required_downstream_tasks=required,
        downstream_receipt_budget=dict(
            current_coordinate256MiB_smoke_does_not_validate_future_inference_receipts=True,
            source_site_join_upper_bytes_if_all_observed_sites_retained=site_join_bound,
            bound_uses_exact_pretty_print_structure_and_max_observed_site_count=True,
            source_writer="src/arggnn/prepare_ancestralpaths_full_panel_discovery_v1.py::json_write",
            recommended_downstream_metadata_cap_bytes=512*2**20,
            recommendation_is_IO_ceiling_only_and_requires_ROOT_activation=True))
    spec_ref=write("repair_specification.json",spec)
    print(json.dumps(dict(reviewed_smoke_ref=reviewed_ref,
        repair_specification_ref=spec_ref,required_downstream_tasks=80,
        native_commands_started=0),indent=2),flush=True)
    return spec_ref,reviewed_ref


if __name__=="__main__":
    run()
