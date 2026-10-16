"""Standard-library JSON/source resource counts. No scientific imports or payload reads."""
from pathlib import Path
import hashlib,json,math,random

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/H058_full_parent_pilot_source_20261016_v2"
CACHE=dict(path="data/expanded_full_panel_pyg_cache_v5/cache/manifest.json",sha256="6144bcc237cf2916fa8b4c90964f6633fa56989d73e2c8f4295675d2239469e8")
COUNTS=dict(path="research/reviews/H043_observed_count_manifest_root_v1/counts.json",sha256="5ea2eedbb44fe34bf89083e15387533b8886cf0230e10f72bf30241c1893e9f8")


def read_metadata(record):
    path=ROOT/record["path"]
    if path.suffix!=".json" or path.resolve()!=path:
        raise ValueError("Exact canonical source JSON only")
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=record["sha256"]:
        raise ValueError("Actual metadata SHA changed")
    return json.loads(raw)


def one_parent_bounds_v1(n,t,s,readouts):
    h,q,r,a,d=952,686,266,4,32
    # All bank integer/float field shapes from the unchanged dataclass.
    # Four level collections contain N+3*(N-T) int64 entries.
    fixed=100*n-24*t+8*t*h+164*s+4*t+8*readouts+8*q+12*h
    layout=4*(7*n-2*t+2*t*h+4*t*q+3*t*r)
    full32=n*d*4
    query32=t*q*d*4
    return dict(native_nodes=n,native_holds=t,original_SNPs=s,original_readouts=readouts,
        CPU_GPU_original_bank_tensor_bytes_lower=fixed+s*h,
        CPU_GPU_original_bank_tensor_bytes_upper=fixed+8*s*h,
        genotype_element_size_not_read=True,genotype_accepted_byte_range=[1,8],
        one_packed_layout_numeric_bytes=layout,one_packed_layout_Tensor_objects=15*t+1,
        one_packed_layout_named_preparation_allowance_bytes=80*n+32*t*h,
        original_H045_episode_tensor_upper_bytes=18*n+4*t,
        original_H045_episode_named_preparation_allowance_bytes=80*n+16*t,
        native_TRAIN_target_FP32_bytes=t*q*4*4,
        one_native_VAL_probability_FP32_bytes=t*q*4*4,
        one_N_by32_FP32_state_bytes=full32,
        F1_query_residual_list_bytes=query32,F1_query_residual_concat_bytes=query32,
        F1_int64_GPU_query_scatter_bytes=t*q*8,
        maximum_global_marker_blocks_upper=math.ceil(s/16)+t,
        three_global_block_Q24_FP32_summary_upper_bytes=(math.ceil(s/16)+t)*q*24*4*3,
        original_full_genomic_GRU_input_bytes=t*h*35*4,
        original_full_genomic_GRU_bidirectional_output_bytes=t*h*64*4,
        original_classifier_context_history_concat_bytes=t*h*96*4,
        ten_live_N32_named_state_allowance_bytes=10*full32,
        # Transparent planning allowances, not a Torch allocator certificate.
        staged_backbone_autograd_named_allowance_bytes=32*full32+20*t*h*d*4)


def build_source_counts_v1():
    cache=read_metadata(CACHE);counts=read_metadata(COUNTS)
    ids=tuple("ap_expanded_train_%03d"%i for i in range(32))+tuple("ap_expanded_val_%03d"%i for i in range(8))
    if cache["observations_only"] is not True or cache["targets_cached"] is not False:
        raise ValueError("Only frozen observation metadata")
    parents={}
    for pid in ids:
        meta=cache["parents"][pid]["observations_receipt"]
        prep=meta["preprocessing"];known=counts["parents"][pid]
        n,t,s=prep["vertices"],prep["genuine_holds"],prep["sites_in_observed_coverage"]
        if (n!=known["native_nodes"] or t!=known["native_holds"]
                or meta["input_samples"]!=952 or meta["actual_query_readouts"]!=686
                or meta["targets_read"] is not False):
            raise ValueError("Actual complete metadata counts disagree")
        parents[pid]=one_parent_bounds_v1(n,t,s,len(prep["readout_positions_bp0"]))
    train=[parents[p] for p in ids[:32]];val=[parents[p] for p in ids[32:]]
    maximum=lambda key:max(p[key] for p in parents.values())
    order=list(ids[:32]);random.Random(102).shuffle(order)
    vmax,b=4096,16
    transient_message=vmax*b*8*4*4
    transient_view=8*(5*vmax-2+2*686+3*266+952)+266*8*4
    return dict(schema="H058_full_parent_source_resource_counts_v1",
        status="SOURCE_METADATA_COUNTS_AND_EXPLICIT_UNMEASURED_ALLOWANCES",
        metadata_refs=[CACHE,COUNTS],parents=parents,
        coverage=dict(TRAIN_parents=32,VAL_parents=8,region_bp=2000000,haplotypes=952,queries=686,
            TRAIN_native_holds_range=[min(p["native_holds"] for p in train),max(p["native_holds"] for p in train)],
            TRAIN_nodes_range=[min(p["native_nodes"] for p in train),max(p["native_nodes"] for p in train)],
            TRAIN_SNP_range=[min(p["original_SNPs"] for p in train),max(p["original_SNPs"] for p in train)]),
        phase_aggregate=dict(TRAIN_original_bank_tensor_upper_bytes=sum(p["CPU_GPU_original_bank_tensor_bytes_upper"] for p in train),
            VAL_original_bank_tensor_upper_bytes=sum(p["CPU_GPU_original_bank_tensor_bytes_upper"] for p in val),
            TRAIN_native_target_tensor_bytes=sum(p["native_TRAIN_target_FP32_bytes"] for p in train),
            TRAIN_all_episode_tensor_upper_bytes=sum(p["original_H045_episode_tensor_upper_bytes"] for p in train),
            new96_native_prediction_bytes=12*sum(p["one_native_VAL_probability_FP32_bytes"] for p in val),
            optional48_control_prediction_bytes=6*sum(p["one_native_VAL_probability_FP32_bytes"] for p in val)),
        one_parent_limits=dict(packed_cache_parent_count=1,simultaneous_packed_arm_count=1,
            max_original_bank_tensor_upper_bytes=maximum("CPU_GPU_original_bank_tensor_bytes_upper"),
            max_packed_layout_numeric_bytes=maximum("one_packed_layout_numeric_bytes"),
            max_packed_Tensor_objects=maximum("one_packed_layout_Tensor_objects"),
            max_packed_named_preparation_allowance_bytes=maximum("one_packed_layout_named_preparation_allowance_bytes"),
            F1_comparison_CPU_scratch_allowance_bytes=24*2**20,
            two_original_CPU_DataLoader_workers=True,prefetch_factor=1,pin_memory=False,
            redundant_worker_edge_index_bytes_per_queued_parent_upper=16*max(p["native_nodes"]-p["native_holds"] for p in parents.values())),
        added_route_activation=dict(reviewed_runtime_max_hold_nodes=vmax,all_vertices_retained_or_capacity_rejected=True,
            marker_block_size=b,one_FP32_VB8A_message_bytes=transient_message,
            planning_allowance_32_message_buffers_bytes=32*transient_message,
            one_transient_int64_layout_and_group_view_bytes=transient_view,
            two_simultaneous_transient_views_allowance_bytes=2*transient_view,
            device_views_created_inside_original_checkpoints=True,
            full_bank_and_full_history_gradients_detached=False,
            full_T_by_Q_residual_copies_counted=True,
            full_backbone_and_genomic_GRU_workspace_not_certified_by_route_allowance=True),
        inherited_H050_first_scheduled_parent=dict(parent_id="ap_expanded_train_004",
            control_peak_allocated_bytes=14621469184,shared_allocator_peak_reserved_bytes=20451426304,
            measured_for_H050_only=True),
        first_scheduled_full_parent=order[0],
        suggested_first_scheduled_GPU0_reserved_cap_bytes=28*2**30,
        proposed_total_CPU_owned_RSS_cap_bytes=64*2**30,
        sixteen_GiB_assessment=dict(fit_measured=False,fit_proved=False,
            plausible_with_stage_and_nested_checkpoints=True,
            source_blocker="Historical uncheckpointed allocator peak exceeds16GiB; framework/autograd/workspace/fragmentation are unmeasured for new staged model.",
            source_action="Retain every original observation and gradient; record the first actual scheduled step without extra discarded updates.",
            no_reduced_data_fallback=True),
        no_runtime_or_numerical_execution=True,scientific_imports=False,payloads_opened_or_hashed=False)
