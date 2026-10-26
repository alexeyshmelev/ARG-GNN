"""Observed-tree features for the separately trained four-path experiment."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from ...ancestralpaths_native_bridge import ObservedPanel, extract_native_features
from . import COHORT_FEATURE_ORDER, FEATURE_NAMES, QUERY_COHORTS

# Native columns: Bronze_Age, BAA, Yam, Neo, WHG, EHG, Ana, CHG, age.
_RETAINED_NATIVE_COLUMNS = (0, 2, 3, 4, 5, 6, 7, 8)


@dataclass(frozen=True)
class _DistributionMask:
    """Minimal read-only view accepted by the authentic feature extractor.

    Eight is the extractor's existing excluded-group sentinel. This changes
    distribution eligibility only. It does not rename a sample, mutate the
    observed panel, prune a node or alter any genealogy.
    """

    haplotype_node_ids: tuple[int, ...]
    group_by_node: Mapping[int, int]


def extract_four_path_features(observed_tree: Any, query_node_ids: Sequence[int],
                               panel: ObservedPanel, *,
                               age_mode: str = "training_int",
                               dtype: Any = np.float64) -> np.ndarray:
    """Return [Q,5,8] with BAA excluded before shell selection and normalization.

    Recompute from the original observed tree. Deleting a column from a cached
    [Q,5,9] tensor is insufficient: BAA-only ancestors must be skipped and the
    next eligible ancestors may occur beyond the five originally retained.
    Original sample IDs and requested query order remain exact. This function
    never reads population truth, census nodes or query ancestry labels.
    """
    output_dtype = np.dtype(dtype)
    if not np.issubdtype(output_dtype, np.floating):
        raise ValueError("four-path feature dtype must be floating")
    ids = tuple(query_node_ids)
    if any(panel.cohort_by_node.get(u) not in QUERY_COHORTS for u in ids):
        raise ValueError("four-path queries must be observed non-BAA admixed cohorts")
    allowed = set(COHORT_FEATURE_ORDER) | {"BAA", "present_day", "GBR"}
    if any(cohort not in allowed for cohort in panel.cohort_by_node.values()):
        raise ValueError("undeclared observed cohort in four-path distribution")
    groups = dict(panel.group_by_node)
    for node, cohort in panel.cohort_by_node.items():
        if cohort == "BAA":
            groups[node] = 8
    masked = _DistributionMask(panel.haplotype_node_ids, MappingProxyType(groups))
    native = extract_native_features(observed_tree, ids, masked,
                                     age_mode=age_mode, dtype=np.float64)
    return native[:, :, _RETAINED_NATIVE_COLUMNS].astype(output_dtype, copy=False)


def feature_spec() -> dict[str, Any]:
    return dict(
        paper_section="Supplementary Note 3.1, printed pp. 43-44",
        input_shape=[5, 8], feature_names=list(FEATURE_NAMES),
        eligible_ancient_cohorts=list(COHORT_FEATURE_ORDER),
        excluded_distribution_cohort="BAA", excluded_query_cohort="BAA",
        genealogies_pruned=False, genealogies_reinferred=False,
        sample_ids_and_order_preserved=True,
        masking_precedes_shell_selection_and_denominator=True,
        empty_eligible_shells_skipped=True, maximum_informative_ancestors=5,
        ancient_query_self_inclusion_retained=True,
        repeated_addition_float64_before_final_float32=True,
        root_padding=-15, training_age="int(node_time) / 1500",
        cached_native_six_features_are_insufficient=True,
        author_feature_function="arggnn.ancestralpaths_native_bridge.extract_native_features",
        four_class_recipe_is_disclosed_reconstruction=True)
