"""Pure full-panel native-baseline contracts. No data readers or execution entrypoint.

Caller-supplied metadata/declarations are checked here. Future guarded readers must
verify referenced file bytes and enforce these decisions before opening any truth.
These functions do not create execution/storage/GPU admission, prepare native
inputs, import author models, or establish methodological equivalence.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

SCHEMA = "full_panel_native_baseline_contracts_v1"
SPLIT_SHA256 = "1ab8b72387e51bbb0f2d429410d31f0a5f007b38b7a6fa11809f3d55c0da2cc7"
CLASS_ORDER = ("Ana", "CHG", "WHG", "EHG")
AP6_TO_C4 = ((0, 4), (1, 5), (2,), (3,))
PARENT_ROLES = {
    "ap_full_train_000": "TRAIN",
    "ap_full_train_001": "TRAIN",
    "ap_full_train_002": "INTERNAL_VAL",
    "ap_full_discovery_holdout_000": "OUTER_DISCOVERY",
    "ap_full_discovery_holdout_001": "OUTER_DISCOVERY",
}
COHORT_HAPLOTYPES = {
    "present_day": 182, "Bronze_Age": 162, "BAA": 14, "Neo": 302,
    "Yam": 26, "WHG": 102, "EHG": 86, "Ana": 50, "CHG": 28,
}
REFERENCE_CLASSES = {"Ana": 0, "CHG": 1, "WHG": 2, "EHG": 3}
OLDER_REFERENCE_COUNTS = {"WHG": 12, "EHG": 10, "Ana": 2, "CHG": 10}
SCAN_BUDGET = {
    "present_day": 100000, "Bronze_Age": 100000, "BAA": 25000,
    "Neo": 50000, "Yam": 25000,
}
REQUIRED_OUTER_PRODUCTS = (
    "ancestralpaths_dense6", "candidate_ap6_denseskip",
    "argmix_current_joint_c4", "argformer_retrieval_k1_c4",
    "argformer_retrieval_k5_c4",
)
TRUTH_PURPOSES = {
    "TRAIN_SUPERVISED_ROW_SELECTION": "TRAIN",
    "TRAIN_GRADIENTS": "TRAIN",
    "NATIVE_INTERNAL_VAL_SELECTION": "INTERNAL_VAL",
    "SEALED_OUTER_SCORE": "OUTER_DISCOVERY",
}


class ContractError(ValueError):
    """A scientific/schema contract failed; never silently alter the method."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _int(value: Any, field: str) -> int:
    _require(type(value) is int, f"{field} must be an integer (not bool)")
    return value


def _sha(value: Any, field: str = "sha256") -> str:
    _require(isinstance(value, str) and len(value) == 64 and
             all(c in "0123456789abcdef" for c in value), f"Invalid {field}")
    return value


def _number(value: Any, field: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool),
             f"{field} must be numeric")
    result = float(value)
    _require(math.isfinite(result), f"{field} must be finite")
    return result


def _decode(payload: bytes, field: str) -> dict[str, Any]:
    _require(type(payload) is bytes, f"{field} must be original bytes")
    try:
        result = json.loads(payload)
    except (UnicodeError, ValueError) as error:
        raise ContractError(f"Invalid {field} JSON") from error
    _require(isinstance(result, dict), f"{field} must be a JSON object")
    return result


def validate_split_bytes(payload: bytes) -> dict[str, Any]:
    """Bind the exact root-frozen split, including nominal TRAIN002 provenance."""
    _require(hashlib.sha256(payload).hexdigest() == SPLIT_SHA256,
             "Downstream split bytes differ from the root freeze")
    split = _decode(payload, "split")
    _require(split.get("schema") == "full_panel_native_comparison_downstream_split_v1",
             "Unknown split schema")
    parents = split.get("parents")
    _require(isinstance(parents, list) and len(parents) == 5,
             "Exactly five registered parents are required")
    _require({r.get("parent_id"): r.get("downstream_role") for r in parents} == PARENT_ROLES,
             "Frozen downstream parent roles differ")
    _require(split.get("class_order") == list(CLASS_ORDER) and
             split.get("probability_collapse") == [[1, 5], [2, 6], [3], [4]],
             "Frozen C4 source ontology differs")
    _require(split.get("original_generation_tags_seeds_and_bytes_preserved") is True and
             split.get("original_paper_reproduction") is False,
             "Generation provenance/fidelity must remain explicit")
    val = next(r for r in parents if r["parent_id"] == "ap_full_train_002")
    _require(val["split"] == "train", "Nominal TRAIN002 generation tag must be retained")
    return split


def validate_policy_bytes(payload: bytes, split_payload: bytes) -> dict[str, Any]:
    """Check a planned-only policy, without admitting execution or reading files."""
    split = validate_split_bytes(split_payload)
    cfg = _decode(payload, "policy")
    _require(cfg.get("schema") == "full_panel_native_baselines_v2",
             "Unknown baseline policy schema")
    _require(cfg.get("status") == "planned_not_execution_guard_ready" and
             cfg.get("execution_guard_ready") is False and
             cfg.get("execution_authorized") is False,
             "This contract does not admit execution")
    _require(cfg.get("split_manifest") == {
        "path": "research/reviews/full_panel_native_comparison_split_v1/split.json",
        "sha256": SPLIT_SHA256}, "Unbound split manifest")
    for key in ("generation_config", "generation_source"):
        _require(cfg.get(key) == split[key], f"Frozen {key} lineage differs")
    _require(cfg.get("class_order") == list(CLASS_ORDER) and
             cfg.get("probability_collapse") == [[1, 5], [2, 6], [3], [4]],
             "Policy source ontology differs")
    _require(cfg.get("required_outer_products") == list(REQUIRED_OUTER_PRODUCTS),
             "All prespecified outer prediction products must remain required")
    _require(cfg.get("native_internal_val_selection_methods") ==
             ["argmix_current_joint_c4"], "Only native ARGMix selects on query VAL truth")
    ap = cfg["native_method_contracts"]["ancestralpaths"]
    _require(ap["source_num_seq_in_per_parent_stride"] == 3 and
             ap["shells"] == 5 and ap["head_classes"] == 6 and
             ap["epochs"] == 5 and ap["checkpoint_selection"] == "final_epoch",
             "Native AP shell/head/stride/selection contract differs")
    mix = cfg["native_method_contracts"]["argmix_current"]
    expected = {
        "num_refs": 15, "num_ancs": 0, "head_classes": 4,
        "hidden_size": 128, "num_layers": 4, "num_heads": 4,
        "dim_feedforward_mult": 4, "gt_output_dim": 16,
        "dropout_prob": 0.25, "attn_dropout_prob": 0.1,
        "batch_size": 16, "max_epochs": 20, "optimizer": "AdamW",
        "learning_rate": 0.0002, "weight_decay": 0.0001,
        "label_smoothing": 0.1, "loss_weighting": "none",
        "early_stop_patience": 2, "gru_hidden_size": 128,
        "gru_bidirectional": True, "gru_num_layers": 1,
        "log_scale_tmrca": True, "max_rel_distance": 14,
        "chunk_rows": 1024, "chunk_stride": 512, "source_seed": 50,
        "train_drop_top_m_nearest": 1, "val_infer_drop_top_m_nearest": 0,
        "checkpoint_selection": "native_masked_smoothed_unweighted_VAL_batch_mean",
        "source_integer_times": True, "subgraph_merge_enabled": False,
        "backend": "fast", "truth_label_mode": "first_snp",
        "truth_min_label_purity": None, "train_val_exclude_edge_trees": True,
        "inference_exclude_edge_trees": False, "stored_cm_feature_count": 2,
        "active_cm_feature_count": 1, "include_delta_cm_prev": False,
        "native_final_output": "arithmetic_mean_of_overlapping_chunk_softmax",
        "base_hdf5_precision": "uint16_half_float_bits",
        "unused_base_classifier_is_ancestry_result": False,
    }
    _require(all(mix.get(k) == v for k, v in expected.items()),
             "Native current-release ARGMix extraction/head/selection changed")
    former = cfg["native_method_contracts"]["argformer"]
    _require(former["fitting_regime"] == "within_ARG_observed_only_transductive" and
             former["logical_mlm_batch"] == 512 and
             former["logical_contrastive_batch"] == 4096 and
             former["query_ancestry_used_for_fit"] is False and
             former["native_classifier_used_for_ancestry"] is False and
             former["primary_retrieval_k"] == 1 and former["sensitivity_retrieval_k"] == 5 and
             former["retrieval_corpus"] == "all_unique_full_pure_reference_paths",
             "Native ARGformer observed-only transduction/retrieval changed")
    _require(all(stage["status"] == "planned_unimplemented"
                 for stage in cfg.get("stages", [])) and cfg.get("stages"),
             "Planned stages must not imply runnable admission")
    return cfg


def collapse_ap6_to_c4(probabilities: Any) -> Any:
    """Sum native six probabilities in source order, never collapse native argmax.

    Derived C4 arithmetic uses float64. Input probabilities are validated without
    clipping, normalization, smoothing, masking, or fitting.
    """
    import numpy as np
    p = _probabilities(probabilities, 6)
    return np.stack([p[..., list(columns)].sum(axis=-1)
                     for columns in AP6_TO_C4], axis=-1)


def ap6_c4_argmax(probabilities: Any) -> Any:
    """C4 call after probability collapse; ties use first C4 class in fixed order."""
    import numpy as np
    return np.argmax(collapse_ap6_to_c4(probabilities), axis=-1)


def _probabilities(values: Any, classes: int) -> Any:
    import numpy as np
    p = np.asarray(values, dtype=np.float64)
    _require(p.ndim >= 1 and p.shape[-1] == classes and p.size > 0,
             f"Expected nonempty final probability dimension {classes}")
    _require(bool(np.isfinite(p).all()) and bool((p >= 0).all()) and
             bool((p <= 1).all()), "Probability values must be finite and in [0,1]")
    _require(bool(np.allclose(p.sum(axis=-1), 1, atol=1e-6, rtol=0)),
             "Probability rows must sum to one without renormalization")
    return p


@dataclass(frozen=True)
class ObservedRoster:
    query_nodes: tuple[int, ...]
    reference_nodes: tuple[int, ...]
    reference_classes: tuple[int, ...]
    all_sample_nodes: tuple[int, ...]
    older_reference_counts: tuple[tuple[str, int], ...]
    stock_argmix_id_correspondence: bool

    @property
    def reference_class_by_node(self) -> dict[int, int]:
        return dict(zip(self.reference_nodes, self.reference_classes))


def validate_observed_roster(rows: Sequence[Mapping[str, Any]], *,
                             require_argmix_native_ids: bool = False) -> ObservedRoster:
    """Validate full role/order/age eligibility; do not claim census labels for donors.

    The observed reader must independently hash the exact supplied age/template
    bytes. This validates schema/demography eligibility, not arbitrary-row provenance.
    """
    _require(len(rows) == 952, "All 952 observed haplotypes must remain")
    expected_cohorts = tuple(c for c, count in COHORT_HAPLOTYPES.items()
                             for _ in range(count))
    query_nodes, reference_nodes, classes, nodes = [], [], [], []
    old = dict.fromkeys(OLDER_REFERENCE_COUNTS, 0)
    sample_ids = []
    for index, (row, cohort) in enumerate(zip(rows, expected_cohorts)):
        _require(_int(row.get("sample_index"), "sample_index") == index,
                 "Original ordered genotype slots differ")
        node = _int(row.get("inferred_tree_node_id"), "inferred_tree_node_id")
        _require(node >= 0 and row.get("observed_node_id") == node,
                 "Actual inferred ID/alias differs")
        _require(row.get("observed_cohort") == cohort, "Original cohort order differs")
        _require(_int(row.get("haplotype"), "haplotype") == index % 2,
                 "Original diploid haplotype order differs")
        _require(row.get("sample_id") == f"tsk_{index // 2}",
                 "Original 476 diploid sample order differs")
        sample_ids.append(row["sample_id"])
        age = _number(row.get("age_generations"), "age_generations")
        _require(age >= 0, "Observed age cannot be negative")
        if cohort in REFERENCE_CLASSES:
            _require(row.get("role") == "pure_source_reference" and
                     type(row.get("observed_reference_class")) is int and
                     row["observed_reference_class"] == REFERENCE_CLASSES[cohort],
                     "Only genuine source cohorts are eligible C4 donors")
            split_age = 600 if cohort in ("WHG", "EHG") else 800
            _require(age < split_age, "Donor must exist in its source deme at sampling")
            old[cohort] += int(age > 350)
            reference_nodes.append(node)
            classes.append(REFERENCE_CLASSES[cohort])
        else:
            _require(row.get("role") == "query" and
                     row.get("observed_reference_class") is None,
                     "Queries, including BAA, must never receive donor ancestry labels")
            pulse = {"Bronze_Age": 166, "BAA": 169, "Neo": 259, "Yam": 177}
            _require(age == 0 if cohort == "present_day" else age < pulse[cohort],
                     "Query sampling age must precede its frozen backward admixture")
            query_nodes.append(node)
        nodes.append(node)
    _require(len(set(nodes)) == 952 and len(set(sample_ids)) == 476,
             "Actual node/sample identities must remain unique")
    _require(len(query_nodes) == 686 and len(reference_nodes) == 266,
             "Full query/reference panel differs")
    _require(old == OLDER_REFERENCE_COUNTS,
             "Retain exactly34 frozen older source prototypes, without census targets")
    stock_ids = nodes == list(range(952))
    if require_argmix_native_ids:
        _require(stock_ids, "Stock ARGMix truth resolver requires node=2*sample_row+hap; no reindexing")
    return ObservedRoster(tuple(query_nodes), tuple(reference_nodes), tuple(classes),
                          tuple(nodes), tuple(old.items()), stock_ids)


def parent_namespace(parent_id: str, inferred_tree_sha256: str) -> str:
    _require(parent_id in PARENT_ROLES, "Unknown registered biological parent")
    return parent_id + ":" + _sha(inferred_tree_sha256, "inferred tree hash")


def validate_path_namespace(actual: str, *, parent_id: str,
                            inferred_tree_sha256: str) -> str:
    expected = parent_namespace(parent_id, inferred_tree_sha256)
    _require(actual == expected, "ARG-specific vocabulary/checkpoint namespace differs")
    return expected


def full_path_identity(namespace: str, nodes: Sequence[int]) -> tuple[str, tuple[int, ...]]:
    """Identity includes leaf before token skipping/truncation, within one ARG."""
    _require(isinstance(namespace, str) and ":" in namespace, "Missing ARG namespace")
    parent, digest = namespace.split(":", 1)
    validate_path_namespace(namespace, parent_id=parent, inferred_tree_sha256=digest)
    path = tuple(_int(node, "full path node") for node in nodes)
    _require(path and all(node >= 0 for node in path) and len(set(path)) == len(path),
             "Full leaf-to-root path is empty, invalid, or cyclic")
    return namespace, path


def validate_physical_support(coordinate: Mapping[str, Any], *,
                              original_centers_bp: Sequence[float] | None = None
                              ) -> dict[str, Any]:
    """Check metadata-only observed support and construct exact clipped bin edges."""
    _require(coordinate.get("condition") == "post_five_size_iterations_primary",
             "Primary post-size Relate condition required")
    _require(coordinate.get("normalization_length_bp") == 2000000 and
             coordinate.get("original_parent_interval_bp0") == [0, 2000000] and
             coordinate.get("samples") == 952 and coordinate.get("fixed_grid_total") == 20000,
             "Full observed parent/support schema differs")
    coverage = coordinate.get("coverage_original_bp0")
    _require(isinstance(coverage, (list, tuple)) and len(coverage) == 2,
             "Coverage must be an interval")
    left, right = (_number(x, "coverage") for x in coverage)
    _require(0 <= left < right <= 2000000, "Invalid half-open observed coverage")
    _require(_number(coordinate.get("local_to_original_offset_bp"), "offset") == left and
             _number(coordinate.get("sequence_length_local_bp"), "local length") == right - left,
             "Original/local position transform differs")
    indexes = coordinate.get("retained_fixed100bp_indexes")
    _require(isinstance(indexes, (list, tuple)), "Explicit retained center indexes required")
    indexes = tuple(_int(i, "retained grid index") for i in indexes)
    expected = tuple(i for i in range(20000) if left <= 50 + 100 * i < right)
    _require(indexes == expected and len(indexes) > 0 and
             coordinate.get("fixed_grid_retained") == len(indexes),
             "Retain every observation-defined center inside coverage, without target filtering")
    centers = tuple(float(50 + 100 * i) for i in indexes)
    if original_centers_bp is not None:
        supplied = tuple(_number(x, "original center") for x in original_centers_bp)
        _require(supplied == centers, "Grid bytes/metadata position join differs")
    edges = tuple(max(left, min(right, x)) for x in
                  (centers[0] - 50, *(center + 50 for center in centers)))
    _require(all(b > a for a, b in zip(edges, edges[1:])), "Clipped bins must have positive width")
    return {"grid_indexes": indexes, "original_centers_bp": centers,
            "local_centers_bp": tuple(x - left for x in centers),
            "clipped_bin_edges_bp": edges,
            "coverage_original_bp0": (left, right),
            "observed_coverage_bp": right - left,
            "scored_support_bp": edges[-1] - edges[0],
            "full_parent_excluded_bp": 2000000 - (edges[-1] - edges[0]),
            "within_observed_unscored_bp": right - left - (edges[-1] - edges[0]),
            "query_bp_denominator": 686 * (edges[-1] - edges[0]),
            "no_tail_extrapolation": True}


def validate_site_coordinate_join(site_join: Sequence[Mapping[str, Any]], *,
                                  coverage_original_bp0: Sequence[float],
                                  declared_site_count: int) -> tuple[tuple[int, int, int], ...]:
    """Validate native/local/original/VCF SNP identities, using observations only."""
    _require(len(site_join) == _int(declared_site_count, "site count") and len(site_join) > 0,
             "Complete native site join required")
    _require(len(coverage_original_bp0) == 2, "Coverage interval required")
    left, right = (_number(x, "coverage") for x in coverage_original_bp0)
    _require(0 <= left < right <= 2000000, "Invalid original SNP coverage")
    result = []
    previous_original, previous_source = -1, -1
    for i, row in enumerate(site_join):
        _require(_int(row.get("native_site_id"), "native site ID") == i,
                 "Native site IDs must be complete and ordered")
        original = _int(row.get("original_position_0based"), "original SNP bp0")
        source = _int(row.get("original_site_id"), "source site ID")
        local = _number(row.get("local_position_0based"), "local SNP bp0")
        _require(local.is_integer() and local == original - left and
                 left <= original < right and
                 _int(row.get("VCF_position_1based"), "VCF position") == original + 1,
                 "Exact local/original/VCF SNP join differs")
        _require(original > previous_original and source > previous_source,
                 "SNP/source IDs must stay unique, ordered, and preserve attrition")
        previous_original, previous_source = original, source
        result.append((i, source, original))
    _require(result[0][2] == left and result[-1][2] + 1 == right,
             "Observed coverage must be first retained SNP through last SNP+1")
    return tuple(result)


def authorize_truth_use(parent_id: str, purpose: str, *, method: str,
                        policy: Mapping[str, Any], selection_frozen: bool = False,
                        outer_prediction_bindings: Mapping[tuple[str, str], Mapping[str, str]]
                        | None = None) -> dict[str, Any]:
    """Check truth intent against the frozen split. Does not open/verify target files.

    Outer bindings are declarations only: a guarded scorer must independently
    verify every prediction/selection file hash before obtaining real truth access.
    """
    _require(parent_id in PARENT_ROLES and purpose in TRUTH_PURPOSES,
             "Unknown parent or truth purpose")
    _require(PARENT_ROLES[parent_id] == TRUTH_PURPOSES[purpose],
             "Truth capability violates downstream parent role")
    methods = policy.get("required_outer_products")
    _require(methods == list(REQUIRED_OUTER_PRODUCTS) and method in methods,
             "Unknown or changed comparison method")
    if purpose == "NATIVE_INTERNAL_VAL_SELECTION":
        _require(policy.get("native_internal_val_selection_methods") == ["argmix_current_joint_c4"] and
                 method == "argmix_current_joint_c4",
                 "Only the declared native VAL objective may consume query VAL labels")
    if purpose in ("TRAIN_GRADIENTS", "TRAIN_SUPERVISED_ROW_SELECTION"):
        _require(method not in ("argformer_retrieval_k1_c4", "argformer_retrieval_k5_c4"),
                 "ARGformer fits only observed nodes/reference labels, never query ancestry")
    if purpose == "SEALED_OUTER_SCORE":
        _require(selection_frozen is True, "Selection must be frozen before outer truth")
        required = {(p, m) for p, role in PARENT_ROLES.items()
                    if role == "OUTER_DISCOVERY" for m in methods}
        _require(outer_prediction_bindings is not None and
                 set(outer_prediction_bindings) == required,
                 "Every method product on both outer parents must be declared sealed")
        for binding in outer_prediction_bindings.values():
            _require(isinstance(binding, Mapping) and
                     isinstance(binding.get("path"), str) and binding["path"] and
                     not binding["path"].startswith("/") and
                     ".." not in binding["path"].split("/") and
                     not any(x in binding["path"].split("/") for x in ("truth", "oracle")),
                     "Prediction seal must name a contained non-truth artifact")
            _sha(binding.get("sha256"), "sealed prediction hash")
    return {"parent_id": parent_id, "downstream_role": PARENT_ROLES[parent_id],
            "purpose": purpose, "method": method,
            "truth_file_opened": False, "execution_admission_created": False,
            "requires_guarded_reader_and_verified_bytes": True}


def native_ap_stride(site_count: int, cohort: str, *, traversal: str) -> int:
    """Exact frozen source /3 per-parent TRAIN geometry; zero strides fail."""
    sites = _int(site_count, "inferred site count")
    _require(sites > 0 and cohort in SCAN_BUDGET, "Unknown query cohort or empty sites")
    haplotypes = COHORT_HAPLOTYPES[cohort]
    if traversal == "TRAIN":
        denominator = Fraction(SCAN_BUDGET[cohort]) + Fraction(10000, 3)
    elif traversal == "HOLDOUT":
        denominator = Fraction(20000)
    else:
        raise ContractError("Traversal must be TRAIN or HOLDOUT, not a changed schedule")
    value = Fraction(sites * haplotypes) / denominator
    stride = value.numerator // value.denominator
    _require(stride >= 1, "Native initial stride is zero: fail cell, no grid/stream replacement")
    return stride


def validate_argformer_transduction(*, parent_id: str, inferred_tree_sha256: str,
                                   vocabulary_namespace: str, encoder_namespace: str,
                                   roster: ObservedRoster,
                                   mlm_leaf_nodes: Sequence[int],
                                   contrastive_leaf_nodes: Sequence[int],
                                   query_ancestry_used: bool,
                                   logical_contrastive_batch: int = 4096) -> dict[str, Any]:
    """Check metadata describing a per-ARG observed-only fit, including outer ARGs."""
    validate_path_namespace(vocabulary_namespace, parent_id=parent_id,
                            inferred_tree_sha256=inferred_tree_sha256)
    validate_path_namespace(encoder_namespace, parent_id=parent_id,
                            inferred_tree_sha256=inferred_tree_sha256)
    _require(query_ancestry_used is False, "No query ancestry may enter MLM/contrastive fit")
    mlm = {_int(n, "MLM leaf") for n in mlm_leaf_nodes}
    contrastive = {_int(n, "contrastive leaf") for n in contrastive_leaf_nodes}
    _require(mlm == set(roster.all_sample_nodes),
             "MLM must retain all full-panel observed leaves within its own ARG")
    _require(contrastive == set(roster.reference_nodes),
             "Contrastive labels may come only from all266 pure donor leaves")
    _require(_int(logical_contrastive_batch, "logical contrastive batch") == 4096,
             "Do not reduce the native contrastive negative set")
    return {"parent_id": parent_id, "namespace": vocabulary_namespace,
            "within_ARG_observed_only_transductive": True,
            "biological_parent_holdout_is_not_frozen_encoder_transfer": True,
            "query_ancestry_used": False, "execution_admission_created": False}


def raw_probability_nll(probabilities: Any, truth_mass: Any, weights: Any) -> dict[str, Any]:
    """Unfloored proper score with fractional truth/query-bp weights, no file access."""
    import numpy as np
    p_input = np.asarray(probabilities)
    _require(p_input.ndim >= 2, "NLL requires rows plus class dimension")
    p = _probabilities(p_input, p_input.shape[-1])
    t = _probabilities(truth_mass, p.shape[-1])
    w = np.asarray(weights, dtype=np.float64)
    _require(t.shape == p.shape and w.shape == p.shape[:-1],
             "Prediction/truth/physical-weight join differs")
    _require(bool(np.isfinite(w).all()) and bool((w >= 0).all()) and float(w.sum()) > 0,
             "Physical weights must have finite positive total")
    zero = (p == 0) & (t > 0) & (w[..., None] > 0)
    zero_mass = float(np.sum(np.where(zero, t * w[..., None], 0)))
    total = float(w.sum())
    if bool(zero.any()):
        return {"nll": None, "nll_is_infinite": True,
                "zero_probability_positive_truth_query_bp": zero_mass,
                "denominator_query_bp": total, "probability_floor": None}
    active = (t > 0) & (w[..., None] > 0)
    contribution = np.zeros_like(p)
    contribution[active] = -t[active] * np.log(p[active])
    return {"nll": float(np.sum(contribution * w[..., None]) / total),
            "nll_is_infinite": False,
            "zero_probability_positive_truth_query_bp": 0.0,
            "denominator_query_bp": total, "probability_floor": None}

def run_handmade_checks(policy: Mapping[str, Any], split_payload: bytes) -> dict[str, str]:
    """Decisive synthetic contract checks; no actual parent files or targets opened."""
    import copy
    import numpy as np
    checks: dict[str, str] = {}

    def rejects(function: Any, *args: Any, **kwargs: Any) -> None:
        try:
            function(*args, **kwargs)
        except ContractError:
            return
        raise AssertionError("Expected contract rejection")

    validate_split_bytes(split_payload)
    rejects(validate_split_bytes, split_payload + b" ")
    checks["exact_root_split_bytes"] = "passed"

    p6 = np.array([[.30, .34, .02, .01, .30, .03]])
    assert int(np.argmax(p6)) == 1
    np.testing.assert_allclose(collapse_ap6_to_c4(p6), [[.60, .37, .02, .01]])
    assert ap6_c4_argmax(p6).tolist() == [0]
    rejects(collapse_ap6_to_c4, [[1, 0, 0, 0]])
    checks["source_AP6_probability_collapse_before_argmax"] = "passed"

    rows = []
    for cohort, count in COHORT_HAPLOTYPES.items():
        for within in range(count):
            index = len(rows)
            reference = cohort in REFERENCE_CLASSES
            age = 200 if reference else {"present_day": 0, "Bronze_Age": 165,
                                       "BAA": 168, "Neo": 258, "Yam": 176}[cohort]
            if reference and within < OLDER_REFERENCE_COUNTS[cohort]:
                age = 360 if cohort == "Ana" else 400
            rows.append({"sample_index": index, "sample_id": f"tsk_{index // 2}",
                         "haplotype": index % 2, "inferred_tree_node_id": index + 1000,
                         "observed_node_id": index + 1000, "observed_cohort": cohort,
                         "age_generations": age,
                         "role": "pure_source_reference" if reference else "query",
                         "observed_reference_class": REFERENCE_CLASSES.get(cohort)})
    roster = validate_observed_roster(rows)
    assert len(roster.reference_nodes) == 266 and sum(dict(roster.older_reference_counts).values()) == 34
    assert [roster.reference_classes.count(i) for i in range(4)] == [50, 28, 102, 86]
    rejects(validate_observed_roster, rows, require_argmix_native_ids=True)
    bad = copy.deepcopy(rows)
    bad[344]["observed_reference_class"] = 0
    rejects(validate_observed_roster, bad)
    bad = copy.deepcopy(rows)
    bad[686]["age_generations"] = 200
    rejects(validate_observed_roster, bad)
    checks["unequal_older_source_prototypes_and_native_ID_boundary"] = "passed"

    digest = "1" * 64
    ns0 = parent_namespace("ap_full_train_000", digest)
    ns1 = parent_namespace("ap_full_train_001", digest)
    assert full_path_identity(ns0, [1000, 2000]) != full_path_identity(ns1, [1000, 2000])
    rejects(validate_path_namespace, ns0, parent_id="ap_full_train_001",
            inferred_tree_sha256=digest)
    validate_argformer_transduction(parent_id="ap_full_train_000",
        inferred_tree_sha256=digest, vocabulary_namespace=ns0, encoder_namespace=ns0,
        roster=roster, mlm_leaf_nodes=roster.all_sample_nodes,
        contrastive_leaf_nodes=roster.reference_nodes, query_ancestry_used=False)
    rejects(validate_argformer_transduction, parent_id="ap_full_train_000",
        inferred_tree_sha256=digest, vocabulary_namespace=ns0, encoder_namespace=ns0,
        roster=roster, mlm_leaf_nodes=roster.all_sample_nodes,
        contrastive_leaf_nodes=roster.reference_nodes, query_ancestry_used=True)
    checks["ARG_parent_namespace_and_observed_only_transduction"] = "passed"

    for parent, purpose, method in [
        ("ap_full_train_000", "TRAIN_GRADIENTS", "ancestralpaths_dense6"),
        ("ap_full_train_002", "NATIVE_INTERNAL_VAL_SELECTION", "argmix_current_joint_c4"),
    ]:
        authorize_truth_use(parent, purpose, method=method, policy=policy)
    for parent, purpose, method in [
        ("ap_full_train_002", "TRAIN_GRADIENTS", "ancestralpaths_dense6"),
        ("ap_full_discovery_holdout_000", "TRAIN_GRADIENTS", "argmix_current_joint_c4"),
        ("ap_full_train_000", "TRAIN_GRADIENTS", "argformer_retrieval_k1_c4"),
        ("ap_full_train_002", "NATIVE_INTERNAL_VAL_SELECTION", "ancestralpaths_dense6"),
    ]:
        rejects(authorize_truth_use, parent, purpose, method=method, policy=policy)
    declarations = {(p, m): {"path": f"synthetic_seals/{p}/{m}.json", "sha256": "2" * 64}
                    for p, role in PARENT_ROLES.items() if role == "OUTER_DISCOVERY"
                    for m in REQUIRED_OUTER_PRODUCTS}
    authorize_truth_use("ap_full_discovery_holdout_000", "SEALED_OUTER_SCORE",
        method="ancestralpaths_dense6", policy=policy, selection_frozen=True,
        outer_prediction_bindings=declarations)
    rejects(authorize_truth_use, "ap_full_discovery_holdout_000", "SEALED_OUTER_SCORE",
        method="ancestralpaths_dense6", policy=policy, selection_frozen=False,
        outer_prediction_bindings=declarations)
    declarations.pop(next(iter(declarations)))
    rejects(authorize_truth_use, "ap_full_discovery_holdout_000", "SEALED_OUTER_SCORE",
        method="ancestralpaths_dense6", policy=policy, selection_frozen=True,
        outer_prediction_bindings=declarations)
    checks["TRAIN_VAL_OUTER_truth_gates_and_complete_seal_declarations"] = "passed"

    assert native_ap_stride(2024, "BAA", traversal="TRAIN") == 1
    rejects(native_ap_stride, 2023, "BAA", traversal="TRAIN")
    assert native_ap_stride(1429, "BAA", traversal="HOLDOUT") == 1
    rejects(native_ap_stride, 1428, "BAA", traversal="HOLDOUT")
    checks["native_AP_exact_divisor3_and_zero_stride_failures"] = "passed"

    coordinate = {"condition": "post_five_size_iterations_primary",
        "normalization_length_bp": 2000000, "original_parent_interval_bp0": [0, 2000000],
        "samples": 952, "fixed_grid_total": 20000,
        "coverage_original_bp0": [59, 459], "local_to_original_offset_bp": 59,
        "sequence_length_local_bp": 400, "retained_fixed100bp_indexes": [1, 2, 3, 4],
        "fixed_grid_retained": 4}
    support = validate_physical_support(coordinate, original_centers_bp=[150, 250, 350, 450])
    assert support["scored_support_bp"] == 359 and support["within_observed_unscored_bp"] == 41
    assert support["query_bp_denominator"] == 359 * 686
    joins = [{"native_site_id": i, "original_site_id": source,
              "original_position_0based": pos, "local_position_0based": pos - 59,
              "VCF_position_1based": pos + 1}
             for i, (source, pos) in enumerate([(1, 59), (3, 159), (5, 458)])]
    validate_site_coordinate_join(joins, coverage_original_bp0=[59, 459], declared_site_count=3)
    joins[0]["VCF_position_1based"] = 59
    rejects(validate_site_coordinate_join, joins, coverage_original_bp0=[59, 459],
            declared_site_count=3)
    checks["exact_clipped_physical_support_and_VCF_local_original_join"] = "passed"

    score = raw_probability_nll([[1, 0, 0, 0]], [[.5, .5, 0, 0]], [2])
    assert score["nll"] is None and score["nll_is_infinite"]
    assert score["zero_probability_positive_truth_query_bp"] == 1
    assert raw_probability_nll([[1, 0, 0, 0]], [[1, 0, 0, 0]], [2])["nll"] == 0
    checks["raw_zero_vote_infinite_NLL_without_floor"] = "passed"
    return checks
