"""Pure gate for preserved v4 native outputs and root corrected verification."""
STAGES=("make_chunks","paint_c0","build_topology_c0","find_equivalent_branches_c0",
        "infer_branch_lengths_c0","combine_sections_c0","finalize","convert")
def need(ok,message):
    if not ok:raise ValueError(message)
def verify_v4_corrected(records,refs,expected_driver_ref,expected_manifest_ref,expected_candidate_ref):
    r,w,a,q,o,z,v,f=(records[k] for k in ("controller_result","controller_wait","admitted","request",
      "original_facts","corrected_registration","corrected_result","facts"))
    need(a["request_ref"]==refs["request"] and a["driver_ref"]==expected_driver_ref
      and a["source_manifest_ref"]==expected_manifest_ref and a["synthetic_only"] is True
      and q["driver_ref"]==expected_driver_ref and q["source_manifest_ref"]==expected_manifest_ref,
      "Exact original synthetic v4 producer closure")
    need(r["success"] is False and r["completed_ref"] is None
      and r["status"]=="failed_or_unsupported_preserved"
      and r["facts_ref"]==refs["original_facts"] and r["admitted_ref"]==refs["admitted"]
      and r["all_partial_outputs_preserved"] is True and r["reservation_remains_fully_charged"] is True
      and r["replay_or_phase_resume_authorized"] is False,
      "Preserve genuine original failed controller, null completion and full charge")
    need(w["actual_direct_controller_wait"] is True and w["returncode"]==1
      and w["group_absent_after_wait"] is True and type(w["controller_pid"]) is int and w["controller_pid"]>0,
      "Genuine original root controller wait1 and group drain")
    need(o["status"]=="unsupported_native_facts_preserved"
      and o["failures"]==["no_unreviewed_internal_time_changes"]
      and o["expected_sites"]==512 and o["haplotypes"]==40
      and len(o["internal_time_mismatch_facts"])==39,
      "Exactly the preserved source-type oracle failure")
    need(set(refs["waits"])==set(STAGES) and set(records["waits"])==set(STAGES)
      and o["native_direct_wait_refs"]==refs["waits"],
      "Exact original eight actual native wait references")
    need(r["actual_native_wait_observations"]==[{k:v for k,v in records["waits"][stage].items() if k!="admitted_ref"} for stage in STAGES],
      "Original result observations equal the actual native receipts")
    for stage in STAGES:
        x=records["waits"][stage]
        need(x["stage"]==stage and x["admitted_ref"]==refs["admitted"] and x["actual_direct_wait_observed"] is True
          and x["actual_exit_code"]==0 and x["process_group_drained"] is True
          and x["owned_group_absent_after_wait"] is True and x["control_error"] is None
          and x["supervisor_pid"]==w["controller_pid"],"Actual original native wait0/drain: "+stage)
    need(z["schema"]=="H018_v4_verification_only_root_registration_v1"
      and z["candidate_ref"]==expected_candidate_ref and z["original_facts_ref"]==refs["original_facts"]
      and z["original_result_ref"]==refs["controller_result"] and z["original_controller_exit_stays"]==1
      and z["native_rerun"] is False and z["native_or_empirical_inputs_modified"] is False,
      "Exact registered verification-only correction, preserving old failure")
    need(v["schema"]=="H018_v4_corrected_verification_root_result_v1"
      and v["candidate_ref"]==expected_candidate_ref and v["registration_ref"]==refs["corrected_registration"]
      and v["facts_ref"]==refs["facts"] and v["all_original_artifacts_unchanged"] is True
      and v["status"]=="qualified_tiny_synthetic_native_path_only" and v["failures"]==[]
      and v["original39_mismatches"]==39 and v["remaining_internal_time_mismatches"]==0
      and v["native_commands_rerun"]==0 and v["original_controller_exit"]==1
      and v["observed2074_inference_executed"] is False,
      "Genuine successful corrected preserved-output verification")
    native=v["native_refs"]
    need(len(native)==3 and {x["path"].rsplit("/",1)[-1] for x in native}=={"native.anc","native.mut","native.trees"}
      and all(x["path"].startswith("data/H018_native_Relate_synthetic_qualification_v4/work/")
              and len(x["sha256"])==64 for x in native),"Exact retained synthetic native outputs")
    flags=("all_fixture_sites_preserved","all20480_biological_nucleotide_and_phase_calls",
      "all_exact_local_bp1_sites_no_jitter","every_fixture_site_has_full_tree_support",
      "all_raw_intervals_carry_all40_samples","exact_six_decimal_age_tokens",
      "combined_binary64_sixdecimal_age_bound","exact_parsed_anc_to_binary_sample_time",
      "no_converter_chronology_adjustment_trigger","finite_strict_native_edge_chronology",
      "native_mut_every_identity_position_allele_order","native_mut_exact_row_count",
      "sample_node_insertion_and_count","tree_sequence_exact_site_count",
      "native_tree_local_internal_node_layout","no_unreviewed_internal_time_changes")
    need(f["schema"]=="H018_Relate_synthetic_native_facts_v1"
      and f["status"]=="qualified_tiny_synthetic_native_path_only" and f["failures"]==[]
      and f["synthetic_only"] is True and f["expected_sites"]==512 and f["haplotypes"]==40
      and f["native_mut_rows"]==512 and all(f[k] is True for k in flags)
      and len(f["six_constant_vector_facts"])==6 and f["internal_time_mismatch_facts"]==[]
      and f["invalid_edges"]==[] and f["vector_errors"]==[]
      and f["native_anc_float32_nonpositive_branches"]==[]
      and f["native_converter_chronology_adjustment_triggered"] is False
      and f["verification_only_no_native_rerun"] is True and f["original_controller_exit"]==1
      and f["installed_binary_behavior_verified_on_fixture"] is True
      and f["exact_source_build_identity_proven"] is False
      and f["qualifier_modified_native_genotypes_or_times"] is False
      and f["biological_or_accuracy_claim"] is False and f["observed2074_baseline_authorized"] is False,
      "Complete corrected synthetic format/date/site/support facts only")
    return dict(status="verified_original_native_waits_and_corrected_v4_facts",
      fixture_sites=512,haplotypes=40,original_controller_exit=1,
      original_completion_manufactured=False,native_commands_rerun=0,
      installed_binary_fixture_behavior_verified=True,exact_source_build_identity_proven=False,
      empirical_ancestry_truth_available=False,empirical_native_execution_authority=False)
