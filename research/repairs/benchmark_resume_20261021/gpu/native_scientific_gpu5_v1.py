"""GPU5 visibility binding for the unchanged author-native ARGformer stage.

The original spec validation, preparation, native argv, scientific completion
checks, limits, writer and stage result schema are retained exactly.
"""
import copy
import json
import os
from arggnn.benchmarking import american_competitors as api
from .shared_gpu5_v1 import GPU5_UUID


def run_argformer(spec, *, bounded_runner=None, resource_check):
    out = api.validate_spec(spec)
    api.need(spec["method"] == "argformer" and callable(resource_check), "Admitted ARGformer stage callback required")
    if spec["stage"] == "prepare":
        return api._argformer_prepare(spec, out, resource_check)
    api.need(callable(bounded_runner), "Injected ROOT bounded native runner required")
    api.need(os.environ.get("CUDA_VISIBLE_DEVICES") == GPU5_UUID and os.environ.get("ARGGNN_PHYSICAL_GPU") == "5",
             "Native stages must inherit ROOT-admitted physical GPU5 visibility")
    resource_check("before_native_stage_scope")
    out.mkdir(parents=True, exist_ok=False)
    writer = api.Writer(out, spec["limits"]["max_output_bytes"], resource_check, create=False)
    spec_ref = writer.json("native_stage_spec.json", spec)
    env = dict(os.environ,
               PYTHONPATH=str(api.ROOT/"src")+":"+str(api.ROOT), PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
               TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")
    wait = bounded_runner([str(api.ROOT/api.NATIVE_PYTHON), "-B", "-s", "-m", api.WORKER,
                           "--spec-ref", json.dumps(spec_ref, sort_keys=True)], str(api.ROOT), env,
                          copy.deepcopy(spec["limits"]), str(out/"native.log"))
    api.validate_wait(wait)
    native = api.read(api.ref(out/"scientific_completed.json"))
    api.need(native["parent_id"] == spec["parent_id"] and native["stage"] == spec["stage"]
             and native["preparation_ref"] == spec["preparation_ref"]
             and native["focal_registry_ref"] == spec["focal_registry_ref"]
             and native["query_ancestry_truth_opened"] is False, "Exact native scientific return required")
    if spec["stage"] in ("mlm", "contrastive"):
        api.need(native.get("completed_epochs") == (5 if spec["stage"] == "mlm" else 1)
                 and native.get("actual_optimizer_updates") == spec["expected_updates"]
                 and native.get("optimizer_unique_step_values") == [spec["expected_updates"]],
                 "Actual complete scientific training return required")
    writer = api.Writer(out, spec["limits"]["max_output_bytes"], resource_check, create=False)
    return writer.json("stage_completed.json", dict(native, schema="american_fresh_ARGformer_stage_completed_v1",
                                                    native_wait=wait, outer_worker_wait_still_required=True))
