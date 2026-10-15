"""Pure successful-v4 qualification evidence gate; no I/O or execution authority."""
STAGES=("make_chunks","paint_c0","build_topology_c0","find_equivalent_branches_c0",
        "infer_branch_lengths_c0","combine_sections_c0","finalize","convert")
def need(ok,message):
    if not ok:raise ValueError(message)
def verify_v4(records,refs,expected_driver_ref,expected_manifest_ref):
    c,r,w,a,q,f=(records[k] for k in ("completed","controller_result","controller_wait","admitted","request","facts"))
    need(c["schema"]=="H018_Relate_synthetic_qualification_completed_v1"
      and c["status"]=="qualified_tiny_synthetic_native_path_only" and c["synthetic_only"] is True
      and c["fixture_sites"]==512 and c["haplotypes"]==40 and c["constant_sites"]==6
      and c["sites_dropped"]==0,"Genuine complete fixed synthetic qualification")
    need(c["admitted_ref"]==refs["admitted"] and c["facts_ref"]==refs["facts"]
      and a["request_ref"]==refs["request"] and a["driver_ref"]==expected_driver_ref
      and q["driver_ref"]==expected_driver_ref and q["source_manifest_ref"]==expected_manifest_ref,
      "Exact frozen v4 qualification producer closure")
    need(r["success"] is True and r["completed_ref"]==refs["completed"]
      and r["facts_ref"]==refs["facts"] and r["admitted_ref"]==refs["admitted"],
      "Actual successful qualification controller result")
    need(w["actual_direct_controller_wait"] is True and w["returncode"]==0
      and w["group_absent_after_wait"] is True and isinstance(w["controller_pid"],int)
      and w["controller_pid"]>0,"Root actual qualification wait0 and group drain")
    need(f["status"]=="qualified_tiny_synthetic_native_path_only" and f["failures"]==[]
      and f["all_fixture_sites_preserved"] is True
      and f["exact_six_decimal_age_tokens"] is True
      and f["combined_binary64_sixdecimal_age_bound"] is True
      and f["exact_parsed_anc_to_binary_sample_time"] is True
      and f["no_converter_chronology_adjustment_trigger"] is True
      and f["finite_strict_native_edge_chronology"] is True,
      "Actual successful fullsite/fractional-date/native chronology facts")
    need(set(refs["waits"])==set(STAGES) and c["actual_native_direct_wait_refs"]==refs["waits"]
      and f["native_direct_wait_refs"]==refs["waits"] and set(records["waits"])==set(STAGES),
      "Exact eight actual native wait receipts")
    for stage in STAGES:
        x=records["waits"][stage]
        need(x["stage"]==stage and x["actual_direct_wait_observed"] is True
          and x["actual_exit_code"]==0 and x["process_group_drained"] is True
          and x["owned_group_absent_after_wait"] is True and x["control_error"] is None
          and x["supervisor_pid"]==w["controller_pid"],"Actual qualifying wait0/drain: "+stage)
    return dict(status="verified_successful_v4_qualification_only",fixture_sites=512,haplotypes=40,
      empirical_ancestry_truth_available=False,empirical_native_execution_authority=False)
