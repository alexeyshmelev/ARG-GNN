"""V9 exact European integer preparation, CPU qualification and deferred GPU fit."""
from __future__ import annotations
import copy
from . import contracts as c
from . import european_training as original
from . import european_c4_training as c4
from . import european_treeloom as observed
from research.reviews.benchmark_repairs_20261019.european_v9 import integration_v1 as integration
from research.reviews.benchmark_repairs_20261019.european_v9 import qualification_v1 as qualification

MODULE = "arggnn.benchmarking.european_preparation_v9"
ROOT = "research/reviews/benchmark_repairs_20261019/"
QUAL_SCHEMA = "european_v9_CPU_qualification_spec_v1"
FIT_SCHEMA = "european_v9_full_fit_wrapper_v1"
PREDICT_SCHEMA = "european_v9_prediction_wrapper_v1"
OBSERVED = ("inference_ref", "observed_vcf_ref", "sampled_registry_ref")


def need(ok, message):
    if not ok:
        raise c.ContractError(message)


def candidate_source_refs():
    names = [__file__, ROOT+"european_v9/integration_v1.py", ROOT+"european_v9/qualification_v1.py",
             ROOT+"treeloom/cut_hoist_v1.py", "src/arggnn/benchmarking/execution.py",
             "research/reviews/workflow_progress_diagnosis_20261018/treeloom/candidate_v1.py",
             "research/reviews/workflow_progress_diagnosis_20261018/treeloom/indices_v1.py",
             "research/reviews/workflow_progress_diagnosis_20261018/treeloom/cache_v1.py",
             "research/reviews/workflow_progress_diagnosis_20261018/treeloom/resident_pool_v2.py"]
    return [c.file_ref(name) for name in names]


def _descriptor(function, spec):
    refs = [*c4.requirements()["source_refs"], *candidate_source_refs()]
    return dict(module=MODULE, function=function, spec=spec, python=".runtime/env/bin/python",
                source_refs=list({ref["path"]: ref for ref in refs}.values()))


def qualification_factory(*, protocol_id, sampling_plan_ref, parent_id, observed_inputs,
        output_directory, max_sites, max_metadata_bytes, preparation_limits,
        max_rss_bytes=128*2**30, max_seconds=7200, max_output_bytes=16*2**20):
    bound = integration.backend(protocol_id)
    plan = bound.parent_plan(sampling_plan_ref, protocol_id)
    need(parent_id == plan["train_parent_ids"][0] and set(observed_inputs) == set(OBSERVED),
         "One exact first TRAIN parent and only its observed inputs required")
    for ref in observed_inputs.values():
        original.gpu._public_ref(ref)
    need(type(max_rss_bytes) is int and 0 < max_rss_bytes <= 128*2**30
         and type(max_seconds) is int and 0 < max_seconds <= 14400
         and 2*2**20 <= max_output_bytes <= 16*2**20,
         "Finite128GiB/14400s/16MiB European CPU qualification caps required")
    original._limits(max_sites=max_sites, max_metadata_bytes=max_metadata_bytes,
        preparation_limits=preparation_limits, max_output_bytes=max_output_bytes, parents=1)
    output = c.contained_path(output_directory)
    need(output.is_relative_to(c.ROOT/"experiments/results") and output.name == "native" and not output.exists(),
         "Fresh supplied task/native scope required")
    spec = dict(schema=QUAL_SCHEMA, protocol_id=protocol_id, parent_id=parent_id,
        sampling_plan_ref=copy.deepcopy(sampling_plan_ref), observed_inputs=copy.deepcopy(observed_inputs),
        output_directory=str(output.relative_to(c.ROOT)), max_sites=max_sites,
        max_metadata_bytes=max_metadata_bytes, preparation_limits=copy.deepcopy(preparation_limits),
        max_rss_bytes=max_rss_bytes, max_seconds=max_seconds, max_output_bytes=max_output_bytes-2**20,
        candidate_source_refs=candidate_source_refs(), CPU_only=True)
    return _descriptor("run_qualification", spec)


def _require_pass(ref, scientific_spec, observed_inputs=None):
    result = c.read_ref(ref)
    need(result.get("schema") == "european_v9_complete_parent_CPU_qualification_v1"
         and result.get("status") == "passed_full_observed_parent_CPU_preparation"
         and result.get("protocol_id") == scientific_spec["protocol_id"]
         and result.get("sampling_plan_ref") == scientific_spec["sampling_plan_ref"]
         and result.get("candidate_source_refs") == candidate_source_refs()
         and result.get("preparation_contract") == qualification.contract(scientific_spec)
         and result.get("all_native_trees_nodes_sites_samples_retained") is True
         and result.get("sampled_rows_and_repeated_draws_preserved") is True
         and result.get("private_truth_opened") is False and result.get("training_labels_opened") is False
         and result.get("model_instantiated") is False and result.get("CUDA_called") is False
         and result.get("GPU_fit_qualified") is False and result.get("full_parent_preparation_count") == 1
         and result.get("disk_cache_bytes") == result.get("scientific_outputs_created") == 0
         and 0 < result.get("observed_peak_rss_bytes", 0) <= 128*2**30
         and 0 < result.get("elapsed_seconds", 0) <= 14400
         and 0 < result.get("observed_tensor_bytes", 0) <= 128*2**30
         and isinstance(result.get("complete_signature_sha256"), str)
         and len(result["complete_signature_sha256"]) == 64,
         "Actual same-protocol/plan full observed parent CPU qualification with exact v9 sources required")
    bound = integration.backend(scientific_spec["protocol_id"])
    plan = bound.parent_plan(scientific_spec["sampling_plan_ref"], scientific_spec["protocol_id"])
    need(result.get("parent_id") == plan["train_parent_ids"][0], "Same exact qualified first TRAIN parent required")
    if observed_inputs is not None:
        need(result.get("observed_inputs") == observed_inputs, "Qualification must bind exact actual observed refs")
    return result


def run_qualification(spec, *, resource_check):
    need(spec["schema"] == QUAL_SCHEMA and spec["CPU_only"] is True
         and spec["candidate_source_refs"] == candidate_source_refs(), "Exact v9 CPU/source specification")
    result = qualification.qualify(spec, resource_check=resource_check)
    ref = c.file_ref(c.contained_path(spec["output_directory"])/"qualification.json")
    need(c.read_ref(ref) == result, "Actual saved European qualification differs")
    _require_pass(ref, spec, spec["observed_inputs"])
    return ref


def fit_factory(*, qualification_ref, **kwargs):
    bound = integration.backend(kwargs["protocol_id"])
    spec = bound.build_fit_spec(**kwargs)
    plan = bound.parent_plan(spec["sampling_plan_ref"], spec["protocol_id"])
    first = spec["train_inputs"][plan["train_parent_ids"][0]]
    _require_pass(qualification_ref, spec, {key: first[key] for key in OBSERVED})
    return _descriptor("run_fit", dict(schema=FIT_SCHEMA, scientific_spec=spec,
        qualification_ref=copy.deepcopy(qualification_ref), candidate_source_refs=candidate_source_refs()))


def prediction_factory(*, qualification_ref, **kwargs):
    bound = integration.backend(kwargs["protocol_id"])
    spec = bound.build_prediction_spec(**kwargs)
    _require_pass(qualification_ref, spec)
    return _descriptor("run_predictions", dict(schema=PREDICT_SCHEMA, scientific_spec=spec,
        qualification_ref=copy.deepcopy(qualification_ref), candidate_source_refs=candidate_source_refs()))


def _validate(spec, schema):
    need(spec == dict(schema=schema, scientific_spec=spec["scientific_spec"],
         qualification_ref=spec["qualification_ref"], candidate_source_refs=candidate_source_refs()),
         "Exact v9 scientific wrapper/source specification required")
    scientific = spec["scientific_spec"]
    bound = integration.backend(scientific["protocol_id"])
    inputs = None
    if schema == FIT_SCHEMA:
        plan = bound.parent_plan(scientific["sampling_plan_ref"], scientific["protocol_id"])
        inputs = {key: scientific["train_inputs"][plan["train_parent_ids"][0]][key] for key in OBSERVED}
    _require_pass(spec["qualification_ref"], scientific, inputs)
    return bound, scientific


def run_fit(spec, *, resource_check, acquire_gpu0):
    bound, scientific = _validate(spec, FIT_SCHEMA)
    return bound.run_fit(scientific, resource_check=resource_check, acquire_gpu0=acquire_gpu0)


def run_predictions(spec, *, resource_check):
    bound, scientific = _validate(spec, PREDICT_SCHEMA)
    return bound.run_predictions(scientific, resource_check=resource_check)
