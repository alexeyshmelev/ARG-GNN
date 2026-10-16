"""Frozen registered H058 science and inert metadata contract. No execution."""
from __future__ import annotations

HYPOTHESIS = "H058"
ARMS = ("same_marker_aligned", "same_marker_alignment_null")
CONTINUATION = "h045_continued_training"
TRAIN = tuple("ap_expanded_train_%03d" % i for i in range(32))
VAL = tuple("ap_expanded_val_%03d" % i for i in range(8))
PASSES = (5, 10, 15, 20, 25, 30)
MODEL = dict(hidden_dim=32, expected_haplotypes=952, site_chunk_size=256, checkpoint_sites=True)
PARAMETERS = {ARMS[0]: 32526, ARMS[1]: 32526, CONTINUATION: 31812}
SELECTION = "earliest_maximum_equal_parent_Present182_exact_span_accuracy_then_lower_NLL_then_earlier_pass"
OBSERVATION = dict(path="data/expanded_full_panel_pyg_cache_v5/cache/manifest.json", sha256="6144bcc237cf2916fa8b4c90964f6633fa56989d73e2c8f4295675d2239469e8")
WARM_CHECKPOINT_REF = dict(path="experiments/results/H045_GPU5_episode_train_seed101_v1/own_state.pass_30.pt", sha256="11dc132ac1958c452fa1d2651bd261258aac6c036ebbdbe7175f235148905331")
WARM_COMPLETION_REF = dict(path="experiments/results/H045_GPU5_episode_train_seed101_v1/completed.json", sha256="258c7da76aa1f693c684094569100e19a3430566c46fe9452e841eb5127f955e")
PROTOCOL_REF = dict(path="research/reviews/biological_operator_next_fresh_20261016_v1/PROTOCOL_source_only_v2.md", sha256="f24b7deacdd5faff00793ba4153a3a26142746705b09afad883deeefab5f83cb")
F1_REVIEW_REF = dict(path="research/reviews/same_marker_full_parent_binding_successor_independent_review_20261016_v1/REPORT.md", sha256="0d3476b0060fb20ded391c376f49ca9c13b1402db79e64f216e73207b45feb6e")
PACKED_CORE_REF = dict(path="research/reviews/biological_operator_next_fresh_20261016_v1/packed_csr_v2/same_marker_packed_csr_v2.py", sha256="081acadf8f55adec0a5d16c6430c30fe5771ee4a307f5387d32138e1659a5e61")
F1_BINDING_REF = dict(path="research/reviews/same_marker_full_parent_binding_successor_source_20261016_v1/observation_binding_v3.py", sha256="b071b47c2f3381edb0c1c38d6a43de427655740439a20158cbde88c725b01ae3")
F1_INTEGRATION_REF = dict(path="research/reviews/same_marker_full_parent_binding_successor_source_20261016_v1/integration_hooks_v3.py", sha256="82933740e0ecbe059a9e8eb2f16bf238bfbda770159b1941d0f55989c8afe78f")
H050_GPU_QUALIFICATION_REF = dict(path="experiments/results/H050_GPU0_direct_readout_train_seed101_v1/cuda_qualification/qualification.json", sha256="279c701d8120e21a5fd8998a5e74a57427373dbb95c1c96e0a89c92771167b2f")
H050_CPU_RUNTIME_REF = dict(path="research/reviews/H050_CPU_evaluation_failed_CLI_successor_source_20261016_v1/root_actual_metadata_v1/runtime.json", sha256="562649fa6ba8b6c180a6884cd00b06ed8b149b3ad4f54e5f828d5a3ef10a87b6")
H050_CPU_EVAL_REF = dict(path="experiments/results/H050_CPU_full192_savedfit_failedCLI_root_v1/completed.json", sha256="2e3dcb14affee5f87d9e0607745a0039647d0ca55daffa0e276bb5599a4a6763")
H050_TRAIN_COMPLETION_REF = dict(path="experiments/results/H050_GPU0_direct_readout_train_seed101_v1/completed.json", sha256="8df6cdcf507a067e48a9a23769cbb6718d485323f29ead6f4d767a64a55fbb8c")
H050_INPUT_REF = dict(path="research/reviews/GPU0_shared_root_integration_20261016_v1/admitted_H050_GPU0_train_v1/input.json", sha256="1100941aaf8b1431cb0c4ea963daf568843ed1b76caa397c98bedb0fe3e57576")
H050_TRAIN_BINDINGS_REF = dict(path="research/reviews/GPU0_shared_root_integration_20261016_v1/admitted_H050_GPU0_train_v1/phase_target_bindings.json", sha256="3aa66440ef0b40f7a4572fac36b78114211aefc3a0b83d14d3c7d3254ca8e1fe")
H050_VAL_BINDINGS_REF = dict(path="research/reviews/H050_CPU_evaluation_failed_CLI_successor_source_20261016_v1/root_actual_metadata_v1/VAL_target_bindings.json", sha256="adcd0dac6fe2c131109244e6ba2e10d96a9b2386920893a29bcf63b7bfa2cc9e")
NULL_SOURCE_REF = dict(path="research/reviews/biological_operator_next_fresh_20261016_v1/source_handoff/same_marker_allele_support_v1.py", sha256="a933d39033312acbccc74f63e5c76789b220ae530d983d2935d81fd681e37490")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def fixed_protocol_v1():
    return dict(schema="H058_full_parent_scientific_protocol_v1", hypothesis_id=HYPOTHESIS,
        arms=list(ARMS), parameters_per_arm={a: PARAMETERS[a] for a in ARMS},
        MODEL=MODEL, TRAIN_parent_ids=list(TRAIN), VAL_parent_ids=list(VAL),
        seed=101, dtype="float32", passes=30, updates_per_pass=32, updates_per_arm=960,
        updates_all_new_arms=1920, checkpoint_passes=list(PASSES), scheduled_checkpoint_files=12,
        VAL_prediction_products_required_before_truth=96, original_grid_bp=100,
        observed_haplotypes=952, query_haplotypes=686, observed_references=266,
        donor_group_counts=[48,18,90,76,2,10,12,10], class_order=["Ana","CHG","WHG","EHG"],
        region_length_bp=2000000, learning_rate=.001, weight_decay=.0001,
        AdamW_betas=[.9,.999], AdamW_epsilon=1e-8, gradient_clip_norm=1.,
        parent_order="shuffle_seed101_plus_pass", warmstart="completed_H045_own_state_pass30",
        common_decoder="unchanged_H045_own_state_full_tree_BiGRU_direct_classifier",
        all_parameters_trainable=True, independent_reset_AdamW=True, early_stopping=False,
        first_step_qualification="first_scheduled_step_retained_in_960_no_extra_updates",
        loss="unchanged_H045_complete_native_hold_original_exact_span_loss",
        selection=SELECTION, VAL_influences_optimization=False, outer_truth_read=False,
        null="seed101_parentwide_complete_reference_diploid_permutation_within_source_age_family",
        initial_gamma=.95, added_route_marker_block=16, added_route_rank=8,
        no_generic_operator_priority_claim=True, reused8VAL_development_evidence=True,
        utility=dict(boundary_F1_gain_min=.01, accuracy_delta_min=-.0005,
            mean_NLL_regression_max=0., controls=["same_marker_alignment_null", CONTINUATION]))


def phase_arms(phase):
    require(phase in ("train", "evaluate", "train_continuation", "evaluate_continuation"), "Explicit scientific phase")
    return (CONTINUATION,) if phase.endswith("_continuation") else ARMS
