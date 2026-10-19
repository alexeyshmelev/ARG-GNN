"""V8 exact cut-root gather hoist with fresh qualified source and lineage bindings."""
from __future__ import annotations
import copy
from . import contracts as c
from . import treeloom_preparation_v4 as frozen
from . import treeloom_preparation_v5 as lineage
from . import treeloom_preparation_v6 as lifecycle
from research.reviews.benchmark_repairs_20261019.treeloom import integration_v1 as repaired
from research.reviews.workflow_progress_diagnosis_20261018.treeloom import prepare_integration_v2 as preparation

MODULE = "arggnn.benchmarking.treeloom_preparation_v8"
PARENT = frozen.PARENT
GIB = 2**30
REPAIR = "research/reviews/benchmark_repairs_20261019/treeloom/"


def _need(ok, message):
    if not ok:
        raise c.ContractError(message)


def candidate_source_refs():
    refs = [*frozen.candidate_source_refs(), c.file_ref(__file__),
            c.file_ref(REPAIR + "cut_hoist_v1.py"), c.file_ref(REPAIR + "integration_v1.py")]
    return list({ref["path"]: ref for ref in refs}.values())


def _source_refs():
    refs = [*frozen._source_refs(), c.file_ref(lineage.__file__),
            c.file_ref(lifecycle.__file__), *candidate_source_refs()]
    return list({ref["path"]: ref for ref in refs}.values())


def _descriptor(function, spec):
    return dict(module=MODULE, function=function, spec=spec,
                python=".runtime/env/bin/python", source_refs=_source_refs())


def qualification_factory(*, preparation_ref, generation_ref, output_directory, max_rss_bytes,
                          max_seconds=7200, max_output_bytes=16*2**20):
    # Keep the authenticated v5 raw/inferred lineage and v6 fresh task/native scope.
    descriptor = lifecycle.qualification_factory(preparation_ref=preparation_ref,
        generation_ref=generation_ref, output_directory=output_directory,
        max_rss_bytes=max_rss_bytes, max_seconds=max_seconds, max_output_bytes=max_output_bytes)
    spec = dict(descriptor["spec"], candidate_source_refs=candidate_source_refs())
    return _descriptor("run_qualification", spec)


def _require_pass(qualification_ref, observed_input_refs):
    result = c.read_ref(qualification_ref)
    _need(result.get("schema") == "one_full_AMtrain000_exact_compact_CPU_qualification_v2"
          and result.get("status") == "passed_complete_parent_exact_preparation_and_two_reuses"
          and result.get("parent_id") == PARENT
          and result.get("observed_input_refs") == observed_input_refs
          and result.get("candidate_source_refs") == candidate_source_refs()
          and result.get("private_truth_opened") is False
          and result.get("models_executed") is False and result.get("CUDA_called") is False
          and result.get("scientific_outputs_created") == 0 and result.get("disk_cache_bytes") == 0
          and result.get("full_parent_preparation_count") == 1 and result.get("exact_restores") == 2
          and result.get("all_original_tensor_and_metadata_signatures_equal") is True
          and result.get("original_complete_parent_retained") is True
          and 0 < result.get("compact_charged_bytes", 0) <= frozen._parent_bound()
          and 0 < result.get("observed_peak_rss_bytes", 0) <= 128*GIB
          and 0 < result.get("elapsed_seconds", 0) <= 7200,
          "Actual bounded full-parent PASS with the exact v8 cut-hoist sources required before fitting")
    return result


def run_qualification(spec, *, resource_check):
    _need(spec["schema"] == frozen.QUALIFICATION_SCHEMA and spec["parent_id"] == PARENT
          and spec["candidate_source_refs"] == candidate_source_refs()
          and spec["CPU_only"] is True and spec["truth_or_model_allowed"] is False
          and spec["disk_cache_bytes"] == 0, "Exact CPU-only v8 qualification/source closure")
    result = repaired.qualify(spec, resource_check=resource_check)
    ref = c.file_ref(c.contained_path(spec["output_directory"]) / "qualification.json")
    _need(c.read_ref(ref) == result, "Exact saved v8 CPU qualification receipt")
    _require_pass(ref, spec["observed_input_refs"])
    return ref


def fit_factory(*, preparation_refs, generation_refs, output_directory, max_rss_bytes, qualification_ref):
    from . import american_workflow as american
    registry = c.file_ref(american.REGISTRY)
    parents = {row["parent_id"] for row in c.read_ref(registry)["parents"] if row["split"] == "train"}
    _need(set(preparation_refs) == set(generation_refs) == parents and len(parents) == 20
          and max_rss_bytes == 512*GIB, "All20 exact TRAIN parents and original512GiB fit cap required")
    observed = american.observed_binding(preparation_refs[PARENT])
    _require_pass(qualification_ref, observed)
    train = {parent: american.train_binding(parent, preparation_refs[parent], generation_refs[parent])
             for parent in sorted(parents)}
    base = preparation.build_spec(parent_registry_ref=registry, output_directory=output_directory,
                                  train_inputs=train, max_rss_bytes=max_rss_bytes)
    spec = dict(schema=frozen.FIT_SCHEMA, training_spec=base,
                qualification_ref=copy.deepcopy(qualification_ref), candidate_source_refs=candidate_source_refs(),
                gpu_worker_and_actual_wait_owned_by_ROOT=True, disk_cache_bytes=0)
    return _descriptor("run_fit", spec)


def run_fit(spec, *, resource_check):
    from . import treeloom_fresh as fresh
    expected = dict(schema=frozen.FIT_SCHEMA, training_spec=spec["training_spec"],
                    qualification_ref=spec["qualification_ref"], candidate_source_refs=candidate_source_refs(),
                    gpu_worker_and_actual_wait_owned_by_ROOT=True, disk_cache_bytes=0)
    _need(spec == expected, "Exact v8 cut-hoist full fit/source closure")
    _require_pass(spec["qualification_ref"],
                  {key: spec["training_spec"]["train_inputs"][PARENT][key] for key in fresh.INPUT_REFS})
    result = repaired.run_training(spec["training_spec"], resource_check=resource_check)
    completion_ref = c.file_ref(c.contained_path(spec["training_spec"]["output_directory"]) /
                                "training_completed.json")
    _need(c.read_ref(completion_ref) == result, "Actual final600-update v8 training receipt differs")
    return dict(result, completion_ref=completion_ref)
