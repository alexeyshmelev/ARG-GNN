"""Synthetic five-parent complete scorer and pre-truth barrier qualification."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from . import CONDITION
from .workflow import score
from .. import european_native_models as io, european_sampling as sampling


def run(output_directory):
    root = io._new_output(output_directory)
    guard = lambda size: None
    check = lambda: None
    parents = ["synthetic_test_" + str(i) for i in range(5)]
    plan = sampling.paper_sampling_plan(["synthetic_train_"+str(i) for i in range(3)],
                                        parents, seed=17)
    def write(path, value):
        return io._json(path, value, guard)
    plan_ref = write(root / "plan.json", plan)
    methods = [CONDITION, "ancestralpaths_dense_control", "treeloom"]
    predictions = {m: {} for m in methods}
    targets = {}
    queries = []
    for cohort, count in zip(sampling.COHORTS, sampling.QUERY_COUNTS):
        for _ in range(count):
            i = len(queries)
            queries.append(dict(sample_index=i, sample_id="q"+str(i//2),
                haplotype=i%2, query_node_id=i, cohort=cohort))
    labels = np.asarray([4+i%2 if r["cohort"] == "BAA" else i%4
                         for i,r in enumerate(queries)], dtype=np.int64)
    for parent in parents:
        directory = root / parent
        directory.mkdir()
        private = directory / "truth"
        private.mkdir()
        records = []
        for cohort in sampling.COHORTS:
            members = [r for r in queries if r["cohort"] == cohort]
            for j in range(2000):
                records.append(dict(members[j%len(members)], example_index=len(records),
                    native_site_id=0, position_local_bp=1., position_original_bp=1.))
        registry = dict(schema="ap_paper_sampled_registry_v1", parent_id=parent, split="test",
            examples=10000, native_path_names=list(sampling.NATIVE_PATHS),
            labels_present=False, query_truth_read=False, query_identities=queries,
            cohort_quotas={c:2000 for c in sampling.COHORTS}, records=records)
        reg_ref = write(directory / "registry.json", registry)
        y = np.asarray([labels[r["sample_index"]] for r in records])
        for index, method in enumerate(methods):
            predicted = y.copy()
            predicted[np.arange(index, 10000, 10+index)] = (predicted[np.arange(index,10000,10+index)]+1)%6
            p = np.full((10000,6), .02, np.float32)
            p[np.arange(10000),predicted] = .9
            values = dict(probabilities=p, example_indices=np.arange(10000,dtype=np.int64),
                example_ids=np.asarray([parent+":"+str(i) for i in range(10000)]))
            arrays_ref = io._npz(directory / (method+".npz"),values,guard)
            product = dict(parent_id=parent, split="test", rows=10000,
                    native_path_names=list(sampling.NATIVE_PATHS), query_truth_read=False,
                    registry_ref=reg_ref, condition=method, predictions_ref=arrays_ref,
                    query_ancestry_truth_opened=False)
            if method == "treeloom":
                del product["query_truth_read"]
            predictions[method][parent] = write(directory / (method+".json"), product)
        truth_ref = io._npz(private / "full_truth_intervals.npz",
            dict(intervals_bp=np.asarray([[0.,200000000.]]), native_paths_1based=labels[None,:]+1,
                 query_raw_node_ids=np.arange(686,dtype=np.int64)),guard)
        schema_ref = write(private / "target_schema.json",dict(parent_id=parent,split="test",
            native_paths_1based=list(range(1,7)),query_sample_indices=list(range(686)),
            query_keys=[[r["sample_id"],r["haplotype"]] for r in queries]))
        gen_ref = write(directory / "generation.json",dict(parent_id=parent,split="test",
            files={"truth/full_truth_intervals.npz":{"sha256":truth_ref["sha256"]},
                   "truth/target_schema.json":{"sha256":schema_ref["sha256"]}}))
        targets[parent] = dict(generation_ref=gen_ref,target_schema_ref=schema_ref,truth_ref=truth_ref)
    calls = []
    original_read = io._read
    private_paths = {v[k]["path"] for v in targets.values() for k in v}
    def tracking_read(ref, **kwargs):
        if ref["path"] in private_paths:
            calls.append(ref["path"])
        return original_read(ref, **kwargs)
    io._read = tracking_read
    broken = {m:dict(v) for m,v in predictions.items()}
    broken["treeloom"][parents[-1]] = dict(path=str((root/"missing.json").relative_to(io.ROOT)),
                                         sha256="0"*64)
    try:
        try:
            score(plan_ref,broken,targets,methods,root/"must_not_score",
                guard=guard,execution_check=check,write_headroom_bytes=1024,
                max_truth_uncompressed_bytes=2**20)
        except ValueError:
            pass
        else:
            raise AssertionError("Incomplete prediction barrier did not fail")
        io._need(not calls and not (root/"must_not_score").exists(),
                 "Every prediction must pass before any private-target read or score output")
        reference = score(plan_ref,predictions,targets,methods,root/"complete",
            guard=guard,execution_check=check,write_headroom_bytes=1024,
            max_truth_uncompressed_bytes=2**20)
    finally:
        io._read = original_read
    comparison = io._read(reference)
    io._need(comparison["all_method_all_parent_prediction_barrier_before_truth"]
        and comparison["test_parents"] == 5 and len(calls) == 10,
        "Full synthetic five-parent scoring and private-target read counts required")
    value = comparison["summary"][CONDITION]["overall"]["native6"]["accuracy"]
    io._need(abs(value["mean"]-.9) < 1e-12 and value["sample_SD"] == 0.,
             "Known complete synthetic scorer values required")
    receipt = dict(schema="paper_ancestralpaths_synthetic_scoring_smoke_v1",status="PASS",
        synthetic_only=True, actual_TEST_truth_read=False, methods=methods,
        synthetic_test_parents=5, synthetic_rows_per_parent=10000,
        missing_last_prediction_rejected_before_any_truth_read=True,
        complete_scoring_passed=True, known_accuracy=.9, sample_SD=0.,
        native6_and_derived_C4=True, per_cohort_endpoints=True, comparison_ref=reference)
    write(root/"receipt.json",receipt)
    return receipt
