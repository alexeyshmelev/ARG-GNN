"""Extend canonical all-parent scoring for explicitly named corrected ARGformer calls."""
from contextlib import contextmanager
from . import protocol as p

def corrected_argformer_seal(parent, record, registry, focal_preparation):
    """Validate corrected lineage without pretending that PCA is native top-one retrieval."""
    value = p.read(record)
    if "completion_ref" in value:
        record = value["completion_ref"]
        value = p.read(record)
    if (value.get("schema") != "paper_american_native_completion_v1"
            or value.get("stage") != "predict" or value.get("variant") != "paper_aligned"
            or value.get("parent_id") != parent or value.get("query_ancestry_truth_opened") is not False
            or value.get("records") != 10000 or value.get("class_order") != p.CLASSES
            or value.get("focal_registry_ref") != focal_preparation["focals_ref"]
            or value.get("evaluator") != "reference_PCA10_nearest_class_centroid"
            or value.get("query_truth_used_for_fit") is not False
            or value.get("probability_semantics") != "hard_call_one_hot_transport"):
        raise ValueError("Complete corrected per-ARG PCA10 hard-call seal required")
    prep = p.read(value["preparation_ref"])
    if (prep["observation_ref"] != focal_preparation["observation_ref"]
            or prep["focals_ref"] != focal_preparation["focals_ref"]
            or prep["parent_id"] != parent or prep["query_ancestry_truth_opened"] is not False):
        raise ValueError("Exactly the shared parent, observed inputs and full focal support required")
    executed = p.read(value["self_spec_ref"])
    contrastive = p.read(executed["checkpoint_completion_ref"])
    if (contrastive["stage"] != "contrastive" or contrastive["variant"] != "paper_aligned"
            or contrastive["preparation_ref"] != value["preparation_ref"]
            or contrastive.get("full_budget_completed") is not True):
        raise ValueError("Complete same-parent corrected contrastive producer required")
    return dict(seal_ref=record, prediction_ref=value["prediction_ref"],
                semantics="paper_oriented_ARGformer_reference_PCA10_centroid_hard_call",
                posterior_metrics=False)

@contextmanager
def canonical_extension():
    """Preserve the original registry, prediction barrier, truth join and metrics."""
    from arggnn.benchmarking import american_scoring as original
    old_seal, old_requirements = original._seal, original.requirements
    def seal(method, parent, record, registry, fp):
        if method == "argformer":
            return corrected_argformer_seal(parent, record, registry, fp)
        return old_seal(method, parent, record, registry, fp)
    def requirements():
        value = old_requirements()
        value["source_refs"] = value["source_refs"] + [p.ref(__file__)]
        value["argformer_evaluator"] = "reference_PCA10_nearest_class_centroid"
        value["argformer_is_native_top1"] = False
        return value
    original._seal, original.requirements = seal, requirements
    try:
        yield original
    finally:
        original._seal, original.requirements = old_seal, old_requirements

def build_spec(**kwargs):
    with canonical_extension() as original:
        return original.build_spec(**kwargs)

def run(spec, *, resource_check):
    with canonical_extension() as original:
        return original.run(spec, resource_check=resource_check)

def score_factory(**kwargs):
    spec = build_spec(**kwargs)
    return dict(module=__name__, function="run", spec=spec, python=".runtime/env/bin/python",
                source_refs=spec["requirements"]["source_refs"])
