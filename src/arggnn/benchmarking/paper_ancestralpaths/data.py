"""Frozen observed row interfaces; only load_training opens TRAIN supervision."""
from __future__ import annotations
from .. import european_native_models as io
from .. import european_sampling as sampling


def checked_plan(plan_ref):
    plan = io._read(plan_ref, observed=True)
    io._need(plan == sampling.paper_sampling_plan(
        plan["train_parent_ids"], plan["test_parent_ids"], seed=plan["seed"]),
        "Exact frozen three-TRAIN/five-TEST paper-count plan required")
    return plan


def load_feature(feature_ref, plan, *, parent, split):
    import numpy as np
    feature = io._read(feature_ref, observed=True)
    registry = io._read(feature["registry_ref"], observed=True)
    expected = plan["parents"][parent]
    io._need(feature["schema"] == "ap_paper_native_features_v1"
        and registry["schema"] == "ap_paper_sampled_registry_v1"
        and feature["parent_id"] == registry["parent_id"] == parent
        and feature["split"] == registry["split"] == expected["split"] == split
        and registry["cohort_quotas"] == expected["cohort_quotas"]
        and feature["rows"] == registry["examples"] == sum(expected["cohort_quotas"].values())
        and feature["feature_shape"] == [5, 9]
        and feature["original_native_feature_arithmetic"] is True
        and feature["source_repeated_add_float64_final_input_float32"] is True
        and feature["features_contain_no_query_truth"] is True
        and feature["all_sampled_rows_retained"] is True
        and registry["labels_present"] is False
        and registry["query_truth_read"] is False
        and registry["observation_ref"] == feature["inference_ref"],
        "Complete observed native TRAIN/TEST feature and registry provenance required")
    a = io._arrays(feature["arrays_ref"], cap=32 * 2**20, observed=True)
    n = feature["rows"]
    io._need(a["features"].shape == (n, 5, 9) and a["features"].dtype == np.float32
        and np.isfinite(a["features"]).all()
        and np.array_equal(a["example_indices"], np.arange(n)),
        "Exact finite Float32 features and canonical IDs required")
    sampling.registry_arrays(registry)
    return feature, registry, a["features"]


def load_training(plan_ref, products):
    import numpy as np
    plan = checked_plan(plan_ref)
    io._need(set(products) == set(plan["train_parent_ids"]), "All three TRAIN products required")
    xs, ys, ids = [], [], []
    cohorts = {c: 0 for c in sampling.COHORTS}
    for parent in plan["train_parent_ids"]:
        refs = products[parent]
        f, r, x = load_feature(refs["features_ref"], plan, parent=parent, split="train")
        t = io._read(refs["targets_ref"])
        io._need(t["schema"] == "ap_paper_TRAIN_native_targets_v1"
            and t["parent_id"] == parent and t["split"] == "train"
            and t["features_ref"] == refs["features_ref"] and t["registry_ref"] == f["registry_ref"]
            and t["no_label_dependent_selection"] is True
            and t["native_path_names"] == list(sampling.NATIVE_PATHS) and t["rows"] == len(x),
            "TRAIN-only immutable same-row supervision required")
        y = io._arrays(t["arrays_ref"], cap=2 * 2**20)["native_labels_0based"]
        io._need(y.shape == (len(x),) and y.dtype.kind in "iu"
            and ((y >= 0) & (y < 6)).all(), "All native-six TRAIN labels required")
        xs.append(x); ys.append(y)
        ids.extend(parent + ":" + str(i) for i in range(len(x)))
        for c, n in r["cohort_quotas"].items():
            cohorts[c] += n
    io._need(set(cohorts.values()) == {10000} and len(ids) == len(set(ids)) == 50000,
             "Exactly 50000 unique TRAIN identities, 10000 per cohort required")
    return np.concatenate(xs).reshape(50000, 1, 5, 9), np.concatenate(ys), ids, cohorts
