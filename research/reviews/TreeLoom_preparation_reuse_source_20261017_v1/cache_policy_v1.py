"""Measured-size, fixed-parent cache selection. No resource operation or estimate fabrication."""
from __future__ import annotations
import math
from .cache_codec_v1 import need
PARENTS=["american_c3_train_%03d"%i for i in range(20)]


def select_fixed_subset_v1(measurements,capacity,*,source_manifest_ref):
    """A selected parent remains cached for all30 passes; there is no eviction/LRU.

    All20 is accepted only if actual measured charges fit. Otherwise a fixed
    benefit/byte subset still produces29 genuine reuse opportunities per parent
    under the original interleaved epoch order. Zero eligible entries means no
    cache claim and no assumed speedup. ROOT must separately approve full600 time.
    """
    need(capacity["root_authorizer"]=="root_agent"
        and capacity["actual_current_observation"]is True
        and capacity["source_only_estimate"]is False,"ROOT supplied actual capacity, never source fit claim")
    budget=min(capacity["actual_admissible_new_bytes"],capacity["cache_reservation_bytes"])
    need(type(budget)is int and budget>0
        and capacity["project_limit_bytes"]==10**12
        and capacity["peak_owned_RSS_cap_bytes"]==384*2**30,"Original1TB and384GiB caps")
    budget-=65536
    rows=[]
    for parent,m in measurements.items():
        need(parent in PARENTS and m["parent_id"]==parent
            and m["actual_full_parent_measurement"]is True
            and m["preparation_source_manifest_ref"]==source_manifest_ref
            and m["sequence_length_bp"]==30000000 and m["haplotypes"]==1600
            and m["native_focals"]==10000 and m["truth_opened"]is False
            and m["roundtrip_all_tensor_bytes_verified"]is True,
            "Genuine exact observed-parent size/replay measurement only")
        need(m["actual_original_owned_RSS_cap_enforced"]is True
            and m["actual_measured_worker_peak_RSS_bytes"]<=384*2**30
            and m["actual_measured_replay_worker_peak_RSS_bytes"]<=384*2**30,
            "Actual cold/replay owned RSS measured under original cap")
        charge=m["per_entry_charge_upper_bound_bytes"]
        cold,replay=m["cold_prepare_seconds"],m["actual_verified_replay_seconds"]
        need(type(charge)is int and charge>0 and type(cold)in(int,float)and type(replay)in(int,float)
            and math.isfinite(cold)and math.isfinite(replay)
            and cold>0 and replay>=0,"Actual finite size/time measurements")
        benefit=29*(cold-replay)
        if benefit>0 and charge<=budget:rows.append((benefit/charge,parent,charge,benefit))
    selected=[];used=0;benefit=0.
    for _,parent,charge,saved in sorted(rows,key=lambda x:(-x[0],x[1])):
        if used+charge<=budget:selected.append(parent);used+=charge;benefit+=saved
    selected.sort()
    return dict(schema="TreeLoom_fixed_observed_preparation_cache_policy_v1",
        preparation_source_manifest_ref=source_manifest_ref,selected_parent_ids=selected,
        cache_charge_upper_bound_bytes=used+65536 if selected else 0,available_cache_bytes=budget+65536,
        measurements={p:measurements[p]for p in selected},
        retention="immutable_fixed_subset_all30epochs_no_eviction",
        per_selected_parent_total_reads=30,per_selected_parent_avoided_repreparations=29,
        all20_cached=selected==PARENTS,source_only_fit_claimed=False,
        measured_preparation_seconds_saved_lower_estimate=benefit,
        training_parent_order_changed=False,training_examples_dropped=False,
        learned_tensors_targets_optimizer_or_RNG_cached=False,
        measured_full600_plan_and_genuine_qualification_still_required=True)


def size_candidates_v1(measurements,capacity,*,source_manifest_ref):
    """Prospective size choice only; no archive/replay or full600 admission claim."""
    need(capacity["root_authorizer"]=="root_agent"
        and capacity["actual_current_observation"]is True and capacity["source_only_estimate"]is False
        and capacity["project_limit_bytes"]==10**12,"Actual ROOT1TB capacity evidence")
    budget=min(capacity["actual_admissible_new_bytes"],capacity["cache_reservation_bytes"])-65536
    need(type(budget)is int and budget>=0,"Explicit nonnegative cache budget")
    rows=[]
    for parent,m in measurements.items():
        need(parent in PARENTS and m["parent_id"]==parent
            and m["actual_full_parent_measurement"]is True
            and m["source_manifest_ref"]==source_manifest_ref
            and m["sequence_length_bp"]==30000000 and m["haplotypes"]==1600
            and m["native_focals"]==10000 and m["truth_opened"]is False,
            "Actual complete size-only observation measurement")
        charge=m["per_entry_charge_upper_bound_bytes"]
        seconds=m["cold_prepare_seconds"]
        need(type(charge)is int and charge>0 and math.isfinite(seconds)and seconds>0,
            "Measured finite size/cold cost")
        rows.append((seconds/charge,parent,charge))
    selected=[];used=0
    for _,parent,charge in sorted(rows,key=lambda x:(-x[0],x[1])):
        if used+charge<=budget:selected.append(parent);used+=charge
    selected.sort()
    return dict(schema="TreeLoom_prospective_measured_size_cache_candidates_v1",
        selected_parent_ids=selected,charge_upper_bound_bytes=used+65536 if selected else 0,
        all20_size_candidates=selected==PARENTS,prospective_only=True,
        exact_archive_replay_and_genuine_producer_closes_still_required=True,
        full600_authorized=False,cache_speedup_or_runtime_fit_claimed=False)
