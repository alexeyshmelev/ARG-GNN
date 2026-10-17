"""Exact pretruth interfaces only. No ancestry truth reader or scorer."""
from . import native_common_v1 as c
PARENTS=tuple("american_c3_val_%03d"%i for i in range(5))

def completed_native_fit_bundle_v1(record):
    b=c.read(record);c.require(b.get("root_authorizer")=="root_agent","ROOT-consumed actual new pool job outcome")
    completed=c.read(b["completed_ref"]);wait=c.read(b["actual_wait_ref"]);outer=c.read(b["pool_outer_consumed_ref"])
    # Completed artifacts are readable after their work authority expires.
    # Authenticate historical provenance without reviving execution authority.
    spec=c.read(wait["input_ref"]);request=c.read(spec["root_stage_request_ref"])
    cfg,_,_,_=c.science_v1(request);protocol=cfg["scientific_protocol"]
    review=c.read(request["source_review_ref"])
    c.require(spec["source_sha256"]=={r["path"]:r["sha256"]for r in c.source_refs_v1()}
        and request["adapter_manifest_ref"]==c.ref(c.MANIFEST)
        and review.get("independent") is True and review.get("verdict")=="SOURCE_PASS"
        and review.get("source_manifest_ref")==c.ref(c.MANIFEST)
        and wait.get("actual_authority_seconds_remaining",0)>150,
        "Exact source-bound original completed authority, no expiry reset or renewed execution")
    registry=c.read(outer["immutable_pool_registry_snapshot_ref"])
    jobs=[j for j in registry["jobs"]if j["logical_job"]==spec["job_id"]]
    c.require(len(jobs)==1,"Same actual declared native pool job")
    job=jobs[0]
    c.require(completed.get("parent_id")==request["parent_id"] and completed.get("preparation_ref")==request["preparation_ref"]
        and completed.get("original_parent_cardinality_ref")==request["counts_ref"]
        and completed.get("completed_epochs")==5 and completed.get("actual_optimizer_updates")==protocol["optimizer_updates"]
        and completed.get("initializer_checkpoint_ref") is None and completed.get("resume_checkpoint_ref") is None
        and completed.get("query_ancestry_truth_opened") is False
        and completed.get("fresh_optimizer_evidence",{}).get("optimizer_state_empty") is True
        and completed.get("active_input_ref")==wait["input_ref"]
        and ("child_input_ref" not in completed or completed["child_input_ref"]==wait["input_ref"])
        and wait.get("adapter_manifest_ref")==c.ref(c.MANIFEST)
        and wait.get("returncode")==0 and wait.get("actual_OS_wait_status")==0
        and wait.get("actual_direct_worker_wait") is True and wait.get("complete_owned_roster_known") is True
        and wait.get("all_owned_processes_and_groups_drained") is True and wait.get("operational_failure") is None
        and wait.get("completed_ref")==b["completed_ref"]
        and job["state"]=="succeeded" and job["exit_code"]==0 and job["leader_wait_status_known"] is True
        and job["owned_drain_observed"] is True
        and outer.get("root_authorizer")=="root_agent" and outer.get("actual_pool_logical_outer_OS_wait_consumed") is True
        and outer.get("actual_exit_code")==0 and outer.get("actual_terminal_consumed") is True
        and outer.get("job_id")==spec["job_id"]
        and outer.get("actual_wait_ref")==b["actual_wait_ref"]
        and registry.get("capability_ref")==request["postdrain_pool_capability_ref"]
        and outer.get("pool_capability_ref")==request["postdrain_pool_capability_ref"]
        and outer.get("original_standalone_or_historical_wait_relabelled") is False,
        "Genuine full parent fit + inner native direct0/drain + actual pool-owned outer leader0/drain, never relabel old UNKNOWNs")
    return dict(parent_id=request["parent_id"],completed_ref=b["completed_ref"],actual_wait_ref=b["actual_wait_ref"],pool_outer_consumed_ref=b["pool_outer_consumed_ref"],original_preparation_ref=request["preparation_ref"],counts_ref=request["counts_ref"],expected_optimizer_updates=protocol["optimizer_updates"],private_truth_access_authorized=False)

def all_parent_pretruth_barrier_v1(records,verify_prediction):
    # A separately reviewed label-blind prediction producer verifies each full
    # product and immutable completion/wait/outer bundle. This gate grants no
    # truth access or evaluator choice on its own.
    c.require(type(records)is dict and set(records)==set(PARENTS),"All five original parent prediction products before any ancestry scoring")
    result={}
    for parent in PARENTS:
        row=verify_prediction(records[parent])
        original=c.read(c.ref(c.PARENT_REGISTRY))["parents"][parent]
        c.require(row["parent_id"]==parent and row["query_focal_count"]==10000
            and row["evaluation_length_bp"]==30000000 and row["query_focals_ref"]==original["query_focals_ref"]
            and row["observation_ref"]==original["observation_ref"] and row["query_ancestry_truth_opened"] is False
            and row["full_author_per_parent_pipeline_completed"] is True
            and row["source_bound_prediction_bytes_verified"] is True
            and row["genuine_completion_direct0_owned_drain_outer0_verified"] is True,
            "Original full parent/focal identities and genuine source-bound pretruth prediction")
        result[parent]=row
    return dict(schema="native_original_five_parent_pretruth_barrier_v1",parents=result,all_five_native_predictions_verified=True,ancestry_truth_access_authorized=False,scoring_or_selection_performed=False,separate_frozen_evaluator_decision_required=True)
