"""Stdlib control gates for a separately reviewed observation-reuse successor."""
from __future__ import annotations
import hashlib
import json
import math
from research.reviews.frozen_TreeLoom_bounded_training_source_20261017_v1 import contract_v1 as c
HERE="research/reviews/TreeLoom_preparation_reuse_source_20261017_v1"
MANIFEST=HERE+"/source_manifest.v1.json"
PARENTS=["american_c3_train_%03d"%i for i in range(20)]


def source_pass(record,manifest_ref):
    review=c.read(record)
    c.require(review.get("independent")is True and review.get("verdict")=="SOURCE_PASS"
        and review.get("blockers")==[]
        and (review.get("candidate_manifest_ref")==manifest_ref or review.get("source_manifest_ref")==manifest_ref),
        "Exact independently reviewed preparation-reuse successor")


def validate_plan(record,parent_refs,manifest_ref):
    """Metadata only: cache payloads are authenticated/read by the worker later."""
    plan=c.read(record)
    c.require(plan["schema"]=="ROOT_TreeLoom_actual_observed_reuse_plan_v1"
        and plan["root_authorizer"]=="root_agent"and plan["ROOT_approved"]is True
        and plan["preparation_source_manifest_ref"]==manifest_ref
        and plan["project_limit_bytes"]==10**12 and plan["owned_RSS_cap_bytes"]==384*2**30
        and plan["retention"]=="immutable_fixed_subset_all30epochs_no_eviction"
        and plan["training_parent_order_changed"]is False
        and plan["truth_or_learned_state_cached"]is False
        and list(plan["parent_refs"])==PARENTS and plan["parent_refs"]==parent_refs,
        "Actual measured fixed-parent cache plan under original1TB/384GiB caps")
    selected=plan["selected_parent_ids"]
    c.require(type(selected)is list and selected==sorted(set(selected))
        and set(selected)<=set(PARENTS)and list(plan["entries"])==selected,
        "Exact fixed reusable subset; no eviction, LRU, fallback or substituted parent")
    total=65536 if selected else 0
    for parent in selected:
        entry=plan["entries"][parent]
        inp=c.read(entry["producer_input_ref"]);done=c.read(entry["producer_completed_ref"])
        wait=c.read(entry["producer_wait_ref"]);outer=c.read(entry["producer_outer_terminal_ref"])
        accepted=c.read(entry["producer_close_accepted_ref"]);receipt=c.read(accepted["receipt_ref"])
        rev=c.read(entry["producer_permanent_revocation_ref"])
        request=c.read(inp["root_stage_request_ref"])
        c.require(request["source_manifest_ref"]==manifest_ref
            and inp["source_sha256"]==request["source_sha256"]==c.read(manifest_ref)["source_sha256"]
            and entry["producer_completed_ref"]==c.ref(inp["output_directory"]+"/completed.json")
            and inp["parent_id"]==parent and inp["stage"]=="prepare_cache"
            and inp["GPU_authorized"]is False and inp["query_truth_access_authorized"]is False,
            "Exact reviewed admitted CPU-only cache producer")
        source_pass(request["source_review_ref"],manifest_ref)
        c.require(done["schema"]=="TreeLoom_observed_preparation_cache_producer_completion_v1"
            and done["status"]=="completed_whole_observed_preparation_archive_and_exact_replay"
            and done["parent_id"]==parent and done["parent_refs"]==parent_refs[parent]
            and done["source_manifest_ref"]==manifest_ref
            and done["observation_only"]is True and done["query_truth_opened"]is False
            and done["model_executed"]is False and entry["archive_ref"]==done["archive_ref"]
            and entry["archive_ref"]["path"]==inp["output_directory"]+"/observed_preparation.cache",
            "Actual observed-only archive and exact replay result")
        c.require(wait["input_ref"]==entry["producer_input_ref"]
            and wait["completed_ref"]==entry["producer_completed_ref"]
            and wait["actual_direct_worker_wait"]is True and wait["actual_direct_launcher_wait"]is True
            and type(wait["returncode"])is int and wait["returncode"]==0
            and type(wait["actual_OS_wait_status"])is int and wait["actual_OS_wait_status"]==0
            and wait["all_owned_processes_and_groups_drained"]is True
            and wait["operational_failure"]is None and wait["cleanup_errors"]==[]
            and all(row["returncode"]==0 for row in wait["adopted_waits"])
            and outer["root_authorizer"]=="root_agent"
            and outer["actual_supervisor_OS_terminal_consumed"]is True
            and type(outer["actual_exit_code"])is int and outer["actual_exit_code"]==0
            and outer["input_ref"]==entry["producer_input_ref"]
            and outer["completed_ref"]==entry["producer_completed_ref"]
            and outer["actual_worker_wait_ref"]==entry["producer_wait_ref"]
            and outer["all_owned_processes_and_groups_drained"]is True,
            "Actual direct/supervisor terminal0 and owned drain, never completed.json alone")
        c.require(accepted["parent_id"]==parent and accepted["stage"]=="prepare_cache"
            and accepted["actual_direct_OS_wait0"]is True
            and accepted["all_owned_processes_and_groups_drained"]is True
            and receipt["key"]==inp["storage"]["reservation"]["key"]
            and rev["reservation_key"]==receipt["key"]
            and rev["input_ref"]==entry["producer_input_ref"]
            and rev["completed_ref"]==entry["producer_completed_ref"]
            and rev["actual_outer_terminal_ref"]==entry["producer_outer_terminal_ref"]
            and rev["all_original_launch_resume_retry_write_routes_permanently_revoked"]is True,
            "Cache producer genuinely closed, retained and all writes permanently revoked")
        m=done["measurement"]
        c.require(m["actual_full_parent_measurement"]is True
            and m["roundtrip_all_tensor_bytes_verified"]is True
            and m["truth_opened"]is False
            and m["actual_measured_worker_peak_RSS_bytes"]<=384*2**30
            and m["actual_measured_replay_worker_peak_RSS_bytes"]<=384*2**30,
            "Actual byte-size/replay/RSS measurement, no source fit assertion")
        caps=c.read(inp["caps_ref"])
        for key in inp["caps_locator"]:caps=caps[key]
        c.require(0<caps["rss_bytes"]<=384*2**30,"Original actual owned RSS cap enforced by admitted CPU supervisor")
        for name in("cold_prepare_seconds","actual_verified_replay_seconds"):
            c.require(type(m[name])in(int,float)and math.isfinite(m[name])and m[name]>0,
                "Actual finite measured cold/replay seconds")
        c.require(entry["tensor_nbytes"]==m["tensor_nbytes"]
            and entry["max_archive_bytes"]==m["serialized_upper_bound_bytes"]
            and entry["key"]==done["key"]and entry["identity"]==done["identity"],
            "Exact measured replay limits and immutable observation/focal/source key")
        total+=m["per_entry_charge_upper_bound_bytes"]
    c.require(total<=plan["actual_retained_cache_charge_upper_bound_bytes"]
        <=plan["explicitly_accounted_retained_cache_bytes"],
        "All selected retained caches fit actual existing charged scopes; no all20 fit pretense")
    return plan


def validate_input_extension(q,storage):
    manifest_ref=c.ref(MANIFEST)
    c.require(q["preparation_reuse_manifest_ref"]==manifest_ref,
        "Exact separate reuse source manifest")
    source_pass(q["preparation_reuse_source_review_ref"],manifest_ref)
    own=c.read(manifest_ref)
    for path,sha in own["source_sha256"].items():c.raw(dict(path=path,sha256=sha))
    c.require(manifest_ref in storage["reservation"]["consumer_bindings"],
        "New source closure bound inside original<=64-role storage contract")
    if q["phase"]=="qualification":
        c.require(q["preparation_cache_plan_ref"]is None and q["preparation_cold_qualification"]is True,
            "Truthful cache-cold fullTRAIN000 technical qualification; no replay or warmstart")
        return None
    c.require(q["phase"]=="full_training"and q["preparation_cold_qualification"]is False,
        "Original full600 training phase")
    plan=validate_plan(q["preparation_cache_plan_ref"],q["parents"],manifest_ref)
    bundle=c.read(q["ROOT_ready_control_bundle_ref"])
    c.require(bundle["preparation_cache_plan_ref"]==q["preparation_cache_plan_ref"]
        and bundle["preparation_reuse_manifest_ref"]==manifest_ref,
        "Actual cache plan consolidated into existing direct ready-control binding")
    return plan
