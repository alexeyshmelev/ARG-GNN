"""Read-only tiny failed graph diagnostic, no tsdate/tsinfer imports or inference."""
from pathlib import Path
import ast
import builtins
import copy
import hashlib
import json
import sys

original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split(".")[0] in {"tsdate", "tsinfer"}:
        raise AssertionError("native inference/date import is forbidden in diagnosis")
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import

import numpy as np
import tskit

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
OUT = ROOT/"research/reviews/biological_paths_tsinfer_native_failure_v1"
BASE = ROOT/"research/reviews/biological_paths_tsinfer_v1_source_audit/root_cpu_qualification_v1"
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):
    return {"path":str(Path(p).relative_to(ROOT)), "sha256":sha(p)}
def write(name, value):
    with (OUT/name).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")

source = OUT/"installed_tsdate_rescaling.py"
names = {"_fixed_changepoints", "_count_mutations", "count_mutations",
         "mutational_area", "mutational_timescale", "piecewise_scale_point_estimate"}
tree = ast.parse(source.read_text())
nodes = [copy.deepcopy(node) for node in tree.body
         if isinstance(node, ast.FunctionDef) and node.name in names]
assert {node.name for node in nodes} == names
for node in nodes:
    node.decorator_list = []  # strip JIT decorators only; every function body unchanged
environment = {"np":np, "tskit":tskit}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), environment)

graphs = {}
comparisons = {}
for name in ["modern_preprocessed.trees", "modern_dated.trees", "all_preprocessed.trees"]:
    path = BASE/name
    before = sha(path)
    ts = tskit.load(path)
    samples = ts.samples()
    edge_stats, mutation_edges = environment["count_mutations"](ts, size_biased=True)
    edge_stats[:, 1] *= 1.25e-8
    fixed = np.zeros(ts.num_nodes, dtype=bool)
    fixed[samples] = True
    counts, offset, duration, indexes = environment["mutational_area"](
        ts.nodes_time, edge_stats, ts.edges_parent, ts.edges_child)
    breaks = np.append(0.0, np.cumsum(duration))
    graphs[name] = {
        "reference":ref(path), "interpretation":"preserved input/output times, not failed EP posterior",
        "sequence_length":ts.sequence_length, "num_nodes":ts.num_nodes, "num_edges":ts.num_edges,
        "num_sites":ts.num_sites, "num_mutations":ts.num_mutations, "num_trees":ts.num_trees,
        "samples":samples.tolist(), "sample_times":ts.nodes_time[samples].tolist(),
        "mutations_node":ts.mutations_node.tolist(), "mutation_child_times":ts.nodes_time[ts.mutations_node].tolist(),
        "mutations_edge":mutation_edges.tolist(),
        "terminal_sample_edges_with_mutations":int(np.sum(np.isin(ts.edges_child[mutation_edges],samples))),
        "epoch_breaks":breaks.tolist(), "epoch_duration":duration.tolist(),
        "epoch_sizebiased_instantaneous_counts":counts.tolist(),
        "epoch_mutational_offset":offset.tolist(),
        "zero_count_positive_offset_epochs":[int(i) for i in np.flatnonzero((counts==0)&(offset>0))],
        "node_times":ts.nodes_time.tolist(),
    }
    results = []
    for intervals in [1000, 1]:
        changepoints = np.unique(environment["_fixed_changepoints"](offset*duration, intervals))
        original_breaks, rescaled_breaks = environment["mutational_timescale"](
            ts.nodes_time, edge_stats, fixed, ts.edges_parent, ts.edges_child, intervals)
        value={"requested_intervals":intervals,
               "actual_unique_changepoint_indexes":changepoints.tolist(),
               "actual_intervals":len(changepoints)-1,
               "original_breaks":original_breaks.tolist(), "rescaled_breaks":rescaled_breaks.tolist(),
               "rescaled_increments":np.diff(rescaled_breaks).tolist(),
               "strictly_positive_rescaled_increments":bool(np.all(np.diff(rescaled_breaks)>0))}
        try:
            rescaled=environment["piecewise_scale_point_estimate"](
                ts.nodes_time, fixed, original_breaks, rescaled_breaks)
            value.update(point_scale_status="completed", fixed_samples_unchanged=bool(np.array_equal(rescaled[samples],ts.nodes_time[samples])))
        except AssertionError as error:
            value.update(point_scale_status="reproduced_assertion_on_preserved_times", error=str(error))
        results.append(value)
    comparisons[name]=results
    assert sha(path)==before

# Tiny algebraic mechanism test: two positive-span time bins, no counts in the first.
point=np.array([0.,10.,20.])
fixed=np.array([True,True,False])
likelihoods=np.array([[0.,1.],[1.,1.]])
parents=np.array([1,2],dtype=np.int32)
children=np.array([0,1],dtype=np.int32)
handmade=[]
for intervals in [1000,1]:
    original,rescaled=environment["mutational_timescale"](point,likelihoods,fixed,parents,children,intervals)
    item={"intervals":intervals,"original_breaks":original.tolist(),"rescaled_breaks":rescaled.tolist()}
    try:
        environment["piecewise_scale_point_estimate"](point,fixed,original,rescaled)
        item["status"]="completed"
    except AssertionError as error:
        item.update(status="assertion",error=str(error))
    handmade.append(item)

write("graph_summary.json",graphs)
write("pure_helper_rescaling_diagnostic.json",{
    "kind":"source_extracted_nonjit_rescaling_arithmetic_diagnostic_v1",
    "source":ref(source), "helper_names":sorted(names), "only_removed_AST":"JIT decorators",
    "tsdate_tsinfer_imported":False, "native_inference_or_dating_rerun":False,
    "failed_EP_posterior_available":False,
    "important_limit":"Input graph times and modern dated output times are inspected; actual failing all_dated EP posterior/breakpoints were not preserved and are not reconstructed.",
    "preserved_graph_time_comparisons":comparisons, "handmade_zero_count_positive_span_mechanism":handmade,
    "not_a_parameter_sweep":"Only frozen1000 and the proposed single-interval fixture setting1 are compared by deterministic helper arithmetic."})
print(json.dumps({"graphs":{k:{"nodes":v["num_nodes"],"mutations":v["num_mutations"],"zero_count_positive_offset_epochs":v["zero_count_positive_offset_epochs"]} for k,v in graphs.items()},
"comparisons":{k:[{"intervals":x["requested_intervals"],"actual":x["actual_intervals"],"status":x["point_scale_status"],"nonpositive":sum(y<=0 for y in x["rescaled_increments"])} for x in v] for k,v in comparisons.items()},
"handmade":handmade},indent=2))
