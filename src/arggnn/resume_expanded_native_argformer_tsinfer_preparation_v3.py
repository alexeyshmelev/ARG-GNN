"""Deferred VAL000 native preparation v3: metadata reservation, sealed binary proof.

Source-only default. Reuses exact original v2 stats/observation/native preparation;
no statistics, inference or training command exists here. ROOT owns any later
650MB admission and execution, after H039 capacity/active-guard constraints.
"""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path
from . import resume_expanded_native_argformer_tsinfer_preparation_v2 as previous

ROOT = previous.ROOT
SOURCE = "src/arggnn/resume_expanded_native_argformer_tsinfer_preparation_v3.py"
PREVIOUS = previous.SOURCE
PREVIOUS_SHA = "d8f888aa577cbeb6ed034b9a1701923aade14a9996c811d7fe2b72740f7c4b99"
STATS = "research/reviews/expanded_full_panel_v1_root/argformer_VAL000_resume_preparation_v2/stats.json"
STATS_SHA = "79741bcafc5e82883dec7e578d4e6c6638dc02995fc555199edf300e28e98bc3"
SCHEMA = "native_argformer_VAL000_resume_preparation_v3_root_admission"
OUTPUT_CAP = 650000000
_SOURCE = None


def ref(path):
    return previous.ref(path)


def read(record):
    return previous.read(record)


def metadata_consumer_refs_v3(spec):
    """Budget consumer bindings contain source/JSON metadata; never binary data."""
    refs = [spec[k] for k in ("source_ref", "scientific_config_ref", "observation_ref",
                             "sealed_roster_ref", "runtime_ref", "stats_ref", "storage_adapter_ref")]
    refs += [dict(path=p, sha256=h) for p, h in spec["source_sha256"].items()]
    result = list({(r["path"], r["sha256"]): r for r in refs}.values())
    previous.v1._require(all(Path(r["path"]).suffix in (".py", ".json") for r in result),
                        "Reservation closure must be source/JSON metadata only")
    return result


def admission_draft_v3(stats_ref=None):
    previous.v1._require(ref(PREVIOUS)["sha256"] == PREVIOUS_SHA, "Frozen v2 must remain unchanged")
    if stats_ref is None:
        stats_ref = dict(path=STATS, sha256=STATS_SHA)
    previous.v1._require(stats_ref == dict(path=STATS, sha256=STATS_SHA),
                        "Exact already executed v2 stats required; no stats reconstruction/rerun")
    result = previous.admission_draft_v2(stats_ref)
    stats = read(stats_ref)
    previous.v1._require(stats["prepared_retained_byte_bound_before_margin"] == 557765233
        and stats["suggested_preparation_RSS_cap_bytes"] == 8589934592
        and stats["suggested_preparation_wall_seconds"] == 600
        and stats["unique_full_paths"] == 1726406 and stats["full_path_occurrences"] == 4997048,
        "Exact previously measured corpus/resource proof required")
    result.update(schema=SCHEMA, source_ref=ref(SOURCE),
        source_sha256={SOURCE: ref(SOURCE)["sha256"], **result["source_sha256"]},
        job_id="H038-native-argformer-VAL000-resume-preparation-v3",
        output_directory="data/expanded_native_argformer_tsinfer_VAL000_prepared_v3",
        output_cap_bytes=OUTPUT_CAP, reservation_closure_kind="source_and_JSON_metadata_only",
        binary_input_verification="exact_spec_and_sealed_observation_manifest_hash_before_native_load",
        prior_v2_source_ref=ref(PREVIOUS), prior800MB_scope_reused=False,
        retained_byte_bound_before_margin=557765233, retained_output_margin_bytes=92234767,
        additional_budget_metadata_charge_bytes=65536,
        projected_new_scope_charge_bytes=650065536,
        scheduling="deferred_until_ROOT_capacity_after_H039_and_compatible_active_guard_completion")
    result["metadata_consumer_refs"] = metadata_consumer_refs_v3(result)
    return result


def verify_sealed_inputs_v3(spec, observed):
    """Validate the binary separately; its JSON manifest remains budget-bound."""
    previous.v1._require(spec["observation_ref"] == dict(path=previous.OBSERVATION, sha256=previous.OBS_SHA)
        and spec["sealed_tree_ref"] == observed["inferred_tree_ref"]
        and spec["sealed_roster_ref"] == observed["roster_ref"],
        "Exact spec raw-input refs must equal the sealed successful observation manifest")
    previous.v1._checked(spec["sealed_tree_ref"])
    previous.v1._checked(spec["sealed_roster_ref"])


def _execution_namespace():
    """Clone only v2 infrastructure, with explicit nonnumerical replacements."""
    previous.v1._require(ref(PREVIOUS)["sha256"] == PREVIOUS_SHA, "Frozen v2 source changed")
    source = previous.v1._checked(ref(PREVIOUS)).read_text()
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "execute_v2")
    code = ast.get_source_segment(source, function)
    replacements = [
        ("def execute_v2(input_ref):", "def execute_v3(input_ref):"),
        ('expected = admission_draft_v2(spec["stats_ref"])', 'expected = admission_draft_v3(spec["stats_ref"])'),
        ('config, receipt = science(); v1._CPU_runtime("prepare", config)',
         'config, receipt = science(); verify_sealed_inputs_v3(spec, receipt); v1._CPU_runtime("prepare", config)'),
        ('    needed = [spec[k] for k in ("source_ref", "scientific_config_ref", "observation_ref", "sealed_tree_ref",\n'
         '                                "sealed_roster_ref", "runtime_ref", "stats_ref", "storage_adapter_ref")]\n'
         '    needed += [dict(path=p, sha256=h) for p, h in spec["source_sha256"].items()]',
         '    needed = metadata_consumer_refs_v3(spec)'),
        ("Exact fresh retired-scopes source/data/runtime/measured output reservation required",
         "Exact fresh retired-scopes source/JSON-metadata/runtime/measured output reservation required"),
        ('v1_unique_path_planning_cap_superseded_explicitly=True)',
         'v1_unique_path_planning_cap_superseded_explicitly=True,\n'
         '                         metadata_reservation_closure_v3=True,\n'
         '                         binary_tree_verified_via_sealed_manifest_and_exact_hash=True)'),
        ("native_argformer_VAL000_resumed_preparation_completed_v2",
         "native_argformer_VAL000_resumed_preparation_completed_v3"),
    ]
    for old, new in replacements:
        previous.v1._require(code.count(old) == 1, "Exact frozen infrastructure replacement required")
        code = code.replace(old, new)
    g = dict(previous.__dict__, SOURCE=SOURCE, SCHEMA=SCHEMA,
        admission_draft_v3=admission_draft_v3, metadata_consumer_refs_v3=metadata_consumer_refs_v3,
        verify_sealed_inputs_v3=verify_sealed_inputs_v3)
    # Preserve the module-import information used by Python's call compiler.
    # These are the same cached stdlib/v1 modules, not new scientific imports.
    imports = ("from datetime import datetime, timezone\nimport time\n"
               "from . import expanded_native_argformer_tsinfer_v1 as v1\n")
    exec(compile(imports + code, SOURCE, "exec"), g)
    return g


def execute_v3(input_ref):
    return _execution_namespace()["execute_v3"](input_ref)


def source_report_v3():
    draft = admission_draft_v3()
    return dict(status="source_only_deferred_no_admission_execution", source_ref=ref(SOURCE),
        previous_source_ref=ref(PREVIOUS), preserved_stats_ref=draft["stats_ref"],
        original_observation_ref=draft["observation_ref"], admission_draft=draft,
        reuse_existing800MB_scope_valid=False,
        reason="New v3 source is not in immutable reserved consumer bindings; no amendment API",
        stats_rerun=False, inference_rerun=False, model_changed=False, guards_changed=False,
        ROOT_scheduling_priority="H039650MB_first_baseline_prep_later_after_capacity",
        native_output_cap_bytes=650000000, projected_scope_charge_bytes=650065536)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--admission-draft", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--input")
    parser.add_argument("--expected-input-sha256")
    args = parser.parse_args(argv)
    if args.admission_draft:
        value = admission_draft_v3()
    elif args.execute:
        if not args.input or not args.expected_input_sha256:
            parser.error("ROOT active exact input and SHA required; admission is deferred")
        value = execute_v3(dict(path=args.input, sha256=args.expected_input_sha256))
    else:
        value = source_report_v3()
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
