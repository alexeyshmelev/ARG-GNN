"""Resume the existing exact TreeLoom boundary into a fresh nested output leaf."""
from arggnn.benchmarking.paper_american import protocol as p
from research.repairs.TreeLoom_lazy_compact_prefetch_20261024_v1 import training

OUTPUT = "experiments/results/paper_american_20261024/treeloom_fit/native"
NEW_LEAF = "resume_20261026_update238"


def run_task(request, *, resource_check):
    if set(request) != {"stage", "arguments"} or request["stage"] != "fit":
        raise ValueError("Exact fit continuation request required")
    arguments = dict(request["arguments"])
    if arguments["output_directory"] != OUTPUT:
        raise ValueError("Original stopped native output required")
    out = p.path(OUTPUT)
    if not out.is_dir() or (out / "training_completed.json").exists():
        raise ValueError("Preserved interrupted native output without completion required")
    evidence = p.read(arguments["resume_evidence_ref"])
    if evidence["complete_parent_updates"] != 238:
        raise ValueError("Explicit durable update238 required")
    arguments["output_directory"] = str(out / NEW_LEAF)
    result = training.run_task(dict(stage="fit", arguments=arguments), resource_check=resource_check)
    if result["complete_parent_updates"] != 600 or result["resumed_from_complete_parent_updates"] != 238:
        raise ValueError("Full fixed update600 completion from update238 required")
    completion = {key: value for key, value in result.items() if key != "completion_ref"}
    completion["nested_completion_ref"] = result["completion_ref"]
    resource_check("TreeLoom_publish_preserved_namespace_completion")
    result["completion_ref"] = p.save(completion, out / "training_completed.json")
    return result
