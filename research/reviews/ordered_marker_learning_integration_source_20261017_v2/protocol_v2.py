"""Inert full-workload ordered-marker placement protocol. No execution."""
from __future__ import annotations
from research.reviews.H058_full_parent_CPU_execution_source_20261017_v1.protocol_v1 import (
    TRAIN,VAL,PASSES,MODEL,SELECTION,OBSERVATION,WARM_CHECKPOINT_REF,WARM_COMPLETION_REF,
    H050_INPUT_REF,H050_TRAIN_BINDINGS_REF,H050_VAL_BINDINGS_REF,require,
    fixed_protocol_v1 as original_h058_protocol_v1)

HYPOTHESIS="ordered_marker_placement_v2"
ARMS=("ordered_marker_tree_feedback","ordered_marker_sample_readout")
PARAMETERS={a:33966 for a in ARMS}


def fixed_protocol_v2():
    value=original_h058_protocol_v1()
    value.update(schema="ordered_marker_placement_scientific_protocol_v2",hypothesis_id=HYPOTHESIS,
        arms=list(ARMS),parameters_per_arm=PARAMETERS,
        null=None,controls=[ARMS[1]],
        comparison="identical_ordered_evidence_before_vs_after_second_complete_tree_sweep",
        common_decoder="unchanged_H045_own_state_full_tree_BiGRU_direct_classifier",
        marker_recurrence="width8_forward_and_reverse_GRUs_reset_only_at_native_hold",
        marker_evidence="original24_channels_all_observed_rows_in_original_order",
        marker_checkpoint="feature_and_directional_GRU_consumer_checkpointed_per_original16marker_block",
        streaming_marker_evidence=True,physical_marker_gaps_used=False,
        native_hold_output_and_original100bp_lookup_unchanged=True,
        utility=dict(boundary_F1_gain_min=.01,accuracy_delta_min=-.0005,
            mean_NLL_regression_max=0.,uncensored_true_short_recall_delta_min=0.,
            comparison=ARMS[0]+"_minus_"+ARMS[1]),
        candidate_registered_at_authoring=False,execution_authority=False,
        no_speed_efficacy_or_novelty_claim=True)
    return value
