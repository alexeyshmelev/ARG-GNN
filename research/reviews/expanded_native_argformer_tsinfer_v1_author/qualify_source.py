"""Source-only qualification: no actual arrays, trees, inference, tensors or guards."""
from pathlib import Path
import ast
import json
import sys
import types
from arggnn import expanded_native_argformer_tsinfer_v1 as native
from arggnn import prepare_expanded_native_argformer_tsinfer_v1 as supervisor

config = native.default_config_v1()
namespace = native._native_namespace()
previous = native._frozen()
unchanged = ("_joins", "_gpu_runtime", "_scientific_checks", "_model", "_drain_loader", "_train_phase")
for name in unchanged:
    assert namespace[name].__code__.co_code == getattr(previous, name).__code__.co_code, name
    assert namespace[name].__code__.co_consts == getattr(previous, name).__code__.co_consts, name


def schema_normalize(value):
    if isinstance(value, str):
        return value.replace("expanded_native_argformer_tsinfer_", "expanded_native_argformer_")
    if isinstance(value, types.CodeType):
        return value.co_code, tuple(schema_normalize(v) for v in value.co_consts)
    return value


for name in ("fit_parent_v1", "infer_parent_v1"):
    old = getattr(previous, name).__code__
    new = namespace[name].__code__
    assert new.co_code == old.co_code, name
    assert tuple(schema_normalize(v) for v in new.co_consts) == tuple(schema_normalize(v) for v in old.co_consts), name

functions = ("parse_vcf", "grid_indexes", "check_samples", "check_genotypes",
    "strict_times", "connected_identity", "require_identity", "sanitize_clip", "pipeline")
pinned = native._checked(config["observation"]["pipeline_ref"]).read_text()
recipe_ast = ast.parse(pinned)
g = native._recipe_namespace(config, lambda planned=0: True)
for node in recipe_ast.body:
    if isinstance(node, ast.FunctionDef) and node.name in functions:
        single = dict(g)
        exec(compile(ast.Module(body=[node], type_ignores=[]), native.PIPELINE, "exec"), single)
        assert single[node.name].__code__.co_code == g[node.name].__code__.co_code

# Handwritten strings validate coordinate/phase contracts without reading data.
coordinates = [dict(site_id=0, VCF_position_1based=165, tree_position_0based=164)]
vcf = ["##source=tskit 1.0.3\n",
       "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample0\n",
       "22\t165\t0\tA\tG\t.\tPASS\t.\tGT\t0|1\n"]
assert g["parse_vcf"](vcf, ["sample0"], coordinates, 2000000,
    expected_writer_source="tskit 1.0.3")[0]["position"] == 164
for malformed in (
    vcf[:-1] + [vcf[-1].replace("0|1", "0/1")],
    vcf[:-1] + [vcf[-1].replace("\t165\t", "\t166\t")],
    vcf[:-1] + [vcf[-1].replace("0|1", ".|1")]):
    try:
        g["parse_vcf"](malformed, ["sample0"], coordinates, 2000000,
                       expected_writer_source="tskit 1.0.3")
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid phase/position accepted")

caps = {key: config["resource_planning"][key] for key in (
    "per_parent_output_cap_bytes", "CPU_RSS_cap_bytes", "per_stage_wall_seconds",
    "maximum_native_nodes", "maximum_native_trees", "maximum_native_edges", "maximum_native_sites",
    "maximum_full_path_occurrences", "maximum_unique_full_paths")}
native._limits(config, caps)
for malformed in ({}, dict(caps, maximum_native_nodes=0),
                  dict(caps, maximum_native_nodes=caps["maximum_native_nodes"] + 1)):
    try:
        native._limits(config, malformed)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid root cap accepted")

assert config["MLM"] == previous.default_config_v1()["MLM"]
assert config["contrastive"] == previous.default_config_v1()["contrastive"]
assert config["retrieval"] == previous.default_config_v1()["retrieval"]
assert config["observation"]["dating"]["rescaling_intervals"] == 1000
assert config["primary_fit_parent_ids"] == list(native.VAL_PARENTS) and len(native.VAL_PARENTS) == 8
draft = supervisor.admission_draft_v1()
assert draft["execution_scope"] == "canary" and draft["execution_parent_ids"] == ["ap_expanded_val_000"]
assert len(draft["completion_refs"]) == 40 and draft["truth_authorized"] is False
assert draft["fit_authorized"] is False and draft["GPU_authorized"] is False
assert "all40_fit_and_prediction_planning_bytes" not in config["resource_planning"]
assert native._ref(native.FROZEN)["sha256"] == native.FROZEN_SHA
assert not any(name in sys.modules for name in ("numpy", "tskit", "tsinfer", "tsdate", "torch"))
print(json.dumps(dict(status="passed_source_and_mock_checks_only",
    unchanged_native_functions=list(unchanged), native_fit_and_retrieval_only_schema_constants_changed=True,
    qualified_recipe_function_bytecode_preserved=list(functions),
    literal_MLM512_contrastive4096_and_k1_k5=True, primary_fit_scope="eight_independent_VAL_ARGs",
    initial_CPU_scope="full952_VAL000_canary", all40_registry_metadata=True,
    malformed_coordinate_phase_and_cap_rejected=True, packages_arrays_trees_models_loaded=False,
    storage_guards_or_journal_used=False, actual_native_capability_verified=False), indent=2, sort_keys=True))
