"""Workflow-facing American prediction binding with genuine candidate adoption.

This preserves the original workflow keyword arguments and forwards their exact
scientific inputs and memory settings to the ordinary low-level prediction
adapter. An adopted completion reference is metadata, not a changed checkpoint.
No process, GPU, prediction or ancestry truth reader is launched here.
"""
from arggnn.benchmarking import contracts as c
from . import american_prediction as model

REGISTRY = "configs/benchmarks/fresh_american_parent_registry.json"
GIB = 2**30


def _fit_result(value):
    if type(value) is dict and set(value) == {"path", "sha256"}:
        reference = model._public_ref(value)
        complete = c.read_ref(reference)
        model._need(type(complete) is dict
                    and complete.get("schema") == "treeloom_fresh_full_training_completion_v1"
                    and complete.get("status") == "completed_full20parents30passes600updates",
                    "The adopted artifact must be a genuine full fixed-final training completion")
        model._need("completion_ref" not in complete or complete["completion_ref"] == reference,
                    "Adopted completion cannot name another completion artifact")
        return dict(complete, completion_ref=dict(reference))
    model._need(type(value) is dict and {"checkpoint_ref", "completion_ref"} <= set(value),
                "Original workflow fit result or its genuine adopted completion reference is required")
    return value


def _observed_binding(preparation_ref):
    model._public_ref(preparation_ref)
    prepared = c.read_ref(preparation_ref)
    return dict(observation_ref=prepared["observation_ref"],
                focal_registry_ref=prepared["focals_ref"],
                native_preparation_ref=preparation_ref)


def prediction_factory(*, fit_result, preparation_refs, output_directory, max_rss_bytes):
    complete = _fit_result(fit_result)
    model._need(type(preparation_refs) is dict and preparation_refs,
                "Original requested held-out preparation references are required")
    call = model.prediction_factory(parent_registry_ref=c.file_ref(REGISTRY),
        checkpoint_ref=complete["checkpoint_ref"], training_completion_ref=complete["completion_ref"],
        heldout_inputs={parent:_observed_binding(ref) for parent,ref in preparation_refs.items()},
        output_directory=output_directory, max_rss_bytes=max_rss_bytes,
        preparation_headroom_bytes=128*GIB, prediction_headroom_bytes=64*GIB,
        observation_cache_bytes=0, max_output_bytes=64*2**20)
    references = [*call["source_refs"], c.file_ref(__file__)]
    return dict(call, source_refs=list({ref["path"]:ref for ref in references}.values()))
