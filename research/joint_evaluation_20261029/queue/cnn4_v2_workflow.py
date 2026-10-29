"""CNN4 preparation with exact measured bounds for the eight pinned manifests.

All original feature extraction, target selection and model code is reused.
The original workflow and the failed original queue attempt remain unchanged.
"""
from collections import Counter
from arggnn.benchmarking.paper_ancestralpaths_four import CONDITION, features
from arggnn.benchmarking.paper_ancestralpaths_four import workflow as original
from arggnn.benchmarking.paper_ancestralpaths_four.workflow import selected_records, extract_rows
from arggnn.benchmarking import european_native_models as io
from arggnn.benchmarking.paper_ancestralpaths import data as original_data
from arggnn.ancestralpaths_native_bridge import ObservedPanel

INFERENCE_METADATA_BYTES = {'1c8383e62f9b3bf914622ca6ac9f014d4142c6bb71870b8cd0508f49b8722e78': 293476778, '940b7e389e74964deaea82bf89febea68bb6b3c0e62012910507303b04058d66': 293096357, '04d22d16294b98e3df81b65df8b441146cf57e7273bf23bda7d0377b3b10d56f': 293529805, '7c6043bcd4f350b0361e6969b231d009500e0ed7ebe6e772a07925176d74be15': 293977705, '12d332cd72741e448d21869a5621fd3a97b853e5f52a43d960906272f1a72d0b': 293881779, '9e6a7d01ec963e7f5e5624d85950b5891dffc0dbcee6340727430aa1040dce5b': 293893047, '3b8361cf05a36d124c965c08794368ae0de2150492a853cdc6adcf9c01d97832': 294354022, 'e9fde75c7666d305c868aa44f8ddb10a3ee20b62293a92d641a67d9d780c6e4b': 294244015}


def prepare(source_plan_ref, registry_ref, inference_ref, output_directory, *,
            guard, execution_check, write_headroom_bytes):
    import numpy as np
    import tskit
    io._ready(guard, execution_check, write_headroom_bytes)
    plan = original_data.checked_plan(source_plan_ref)
    registry = io._read(registry_ref, observed=True)
    records, indices = selected_records(registry, plan)
    inferred = io._read(inference_ref, observed=True, cap=INFERENCE_METADATA_BYTES[inference_ref["sha256"]])
    io._need(registry["observation_ref"] == inference_ref
             and inferred["parent_id"] == registry["parent_id"]
             and inferred["split"] == registry["split"]
             and inferred["truth_or_oracle_read"] is False,
             "Original same-parent observed genealogy required")
    rows = io._read(inferred["observed_sample_manifest"], observed=True)
    panel = ObservedPanel(tuple(r["sample_id"] for r in rows[::2]),
                          tuple(r["observed_node_id"] for r in rows),
                          {r["observed_node_id"]: r["observed_cohort"] for r in rows})
    by_node = {r["observed_node_id"]: r for r in rows}
    for record in records:
        observed = by_node[record["query_node_id"]]
        io._need(observed["sample_id"] == record["sample_id"]
                 and observed["observed_cohort"] == record["cohort"]
                 and observed["sample_index"] == record["sample_index"]
                 and observed["haplotype"] == record["haplotype"]
                 and observed["inferred_tree_node_id"] == record["query_node_id"],
                 "Focal sample, haplotype and inferred node must match the observation manifest")
    ts = tskit.load(io._checked(inferred["inferred_tree"], observed=True))
    io._need(np.array_equal(ts.samples(), panel.haplotype_node_ids),
             "Observed sample order must match the original inferred tree")
    offsets = {r["position_original_bp"] - r["position_local_bp"] for r in records}
    io._need(len(offsets) == 1, "One original-to-local coordinate offset required")
    offset = next(iter(offsets))
    io._need(0 <= offset and offset + ts.sequence_length <= plan["sequence_length_bp"],
             "Inferred coverage must stay inside the original simulated chromosome")
    io._need(all(0 <= r["native_site_id"] < ts.num_sites
                 and ts.site(r["native_site_id"]).position == r["position_local_bp"] for r in records),
             "Every focal coordinate must match its original observed SNP")
    x = extract_rows(ts, panel, records, check=lambda: (execution_check(), guard(0)))
    output = io._new_output(output_directory)
    arrays = io._npz(output / "features.npz", dict(features=x, example_indices=indices), guard)
    return io._json(output / "features.json", dict(
        schema="ap_four_features_v1", condition=CONDITION,
        source_plan_ref=source_plan_ref, registry_ref=registry_ref, inference_ref=inference_ref,
        parent_id=registry["parent_id"], split=registry["split"], rows=len(records),
        rows_per_cohort=dict(Counter(r["cohort"] for r in records)),
        arrays_ref=arrays, feature_shape=[5, 8], feature_spec=features.feature_spec(),
        query_truth_read=False, row_selection="original non-BAA observed cohort rows",
        source_example_indices_preserved=True), guard)



def execute_stage(spec, *, resource_check, execution_check, bounded_runner=None):
    if spec["stage"] != "prepare":
        return original.execute_stage(spec, resource_check=resource_check,
                                      execution_check=execution_check, bounded_runner=bounded_runner)
    return prepare(output_directory=spec["output_directory"], **spec["arguments"],
                   guard=resource_check, execution_check=execution_check)
