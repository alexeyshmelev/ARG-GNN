"""Isolated short development native pipeline; no launch on import."""
from __future__ import annotations
import math
from arggnn.benchmarking.european_native_relate import _observed_assets, _need, OBSERVED_FILES
from . import execution_stages_v2 as stages
from .short_plan_v1 import validate_parent, validate_native_config, plan_ref


def infer_parent(recipe, parent, observed_directory, destination, *,
                 native_config, expected_input_refs, native_step,
                 guard, execution_check, write_headroom_bytes,
                 max_metadata_bytes):
    """Run the exact five-stage native AP inference pipeline at declared length.

    native_step has the existing signature
    (argv, work, journal, label, config, guard). The supplied runner must supervise
    genuine child waits and the configured finite timeout. It must append one
    genuine terminal command record, or raise. The current outer worker owns
    whole-stage RSS, wall, output and process-group limits. No fallback, crop,
    repeat, sampler, feature cache or ancestry truth read occurs here.
    """
    from .aggregate_guard_v2 import AggregateOwner
    from .native_supervisor_v2 import NativeSupervisor
    _need(type(guard) is AggregateOwner and type(native_step) is NativeSupervisor
          and native_step.owner is guard and execution_check is guard.execution_check,
          'One admitted aggregate owner and its exact native supervisor required')
    destination = guard.path(destination)
    observed_directory = guard.path(observed_directory)
    validate_parent(parent)
    validate_native_config(native_config)
    length = stages._approved(recipe, guard, execution_check, write_headroom_bytes)
    _need(callable(native_step), "Caller-owned bounded native-step runner required")
    _need(length == 10_000_000, "Exact frozen 10 Mb development AP inference required")
    settings = recipe["parameters"]["native_inference"]
    _need(settings["initial_haploid_effective_size"] == 30000
          and settings["mutation_rate"] == 1.25e-8
          and settings["population_size_iterations"] == 5
          and settings["final_branch_length_reestimate"] is True
          and settings["years_per_generation"] == 28,
          "Unchanged AP native inference settings required")
    timeout = native_config["relate"]["timeout_seconds_per_step"]
    _need(type(timeout) in (int, float) and math.isfinite(timeout) and timeout > 0
          and type(native_config["native_step_headroom_bytes"]) is int
          and native_config["native_step_headroom_bytes"] > 0,
          "Explicit finite native per-step timeout and write headroom required")
    seeds = parent.get("seeds", {})
    _need(all(type(seeds.get(k)) is int and 0 < seeds[k] < 2**31 - 6
              for k in ("relate_initial", "relate_size_base"))
          and seeds["relate_initial"] != seeds["relate_size_base"],
          "Explicit fresh native inference seed streams required")
    _, _, full, helper = stages._runtime()
    assets = _observed_assets(
        recipe, parent, observed_directory, expected_input_refs, full=full,
        guard=guard, max_metadata_bytes=max_metadata_bytes)
    work = full.confined(destination)
    _need(not work.exists() and not work.is_relative_to(full.confined(observed_directory)),
          "Fresh separately admitted native output directory required")
    native = full.native_tools(native_config)
    guard(16384)
    work.mkdir()
    (work / "tmp").mkdir()
    full.text_write(work / "NOTICE.txt",
        "Relate1.2.4\nCopyright © 2018, University of Oxford\n"
        "Previously authorized internal academic research use.\n", guard)
    eligible, counts = stages.prepare_relate_inputs(
        recipe, assets, work / "inputs", guard=guard,
        execution_check=execution_check, write_headroom_bytes=write_headroom_bytes)
    inputs, journal = work / "inputs", []
    formats = native["bin/RelateFileFormats"]

    def run(argv, label):
        execution_check()
        guard(native_config["native_step_headroom_bytes"])
        before = len(journal)
        native_step(argv, work, journal, label, native_config, guard)
        _need(len(journal) == before + 1
              and journal[-1].get("label") == label
              and journal[-1].get("argv") == argv
              and journal[-1].get("returncode") == 0
              and journal[-1].get("status") == "complete",
              "One genuine successful native step record required")

    run([formats, "--mode", "ConvertFromVcf", "--haps", str(inputs / "input.haps"),
         "--sample", str(inputs / "converted.sample"), "-i", str(inputs / "input")],
        "convert_vcf")
    full.text_write(inputs / "input.sample",
        helper.canonical_sample_text((inputs / "converted.sample").read_text(),
                                     assets["names"]), guard)
    retained = full.haps_check(inputs / "input.haps", eligible)
    run([formats, "--mode", "GenerateSNPAnnotations", "--haps", str(inputs / "input.haps"),
         "--sample", str(inputs / "input.sample"), "--poplabels", str(inputs / "input.poplabels"),
         "-o", str(inputs / "input")], "annotate_observed_cohorts")
    run(["bash", "-x", native["scripts/RelateParallel/RelateParallel.sh"],
         "-m", "1.25e-8", "-N", "30000", "--threads", "8", "--memory", "2",
         "--seed", str(seeds["relate_initial"]), "--haps", str(inputs / "input.haps"),
         "--sample", str(inputs / "input.sample"), "--sample_ages",
         str(inputs / "sample_ages.txt"), "--map", str(inputs / "input.map"),
         "--annot", str(inputs / "input.annot"), "-o", "pre_size"], "relate_initial")
    pre_refs = {name: full.ref(str((work / name).relative_to(full.ROOT)))
                for name in ("pre_size.anc", "pre_size.mut")}
    run(["bash", "-x", native["scripts/EstimatePopulationSize/EstimatePopulationSize.sh"],
         "-i", "pre_size", "-o", "post_size", "-m", "1.25e-8", "--threads", "8",
         "--num_iter", "5", "--threshold", "0.5", "--years_per_gen", "28",
         "--seed", str(seeds["relate_size_base"]), "--poplabels",
         str(inputs / "input.poplabels"), "--noplot"], "five_population_size_iterations")
    run([formats, "--mode", "ConvertToTreeSequence", "-i", "post_size", "-o", "post_size"],
        "convert_post_size")
    guard(max(write_headroom_bytes,
              2 * (work / "post_size.trees").stat().st_size + 1048576))
    # The unchanged cleaner uses this container only for SNP membership.
    # HAPS order was already checked. A set avoids quadratic full-length joins.
    post = stages.clean_converted_tree(
        recipe, work / "post_size.trees", assets, set(retained),
        work / "post_size.observed.trees", guard=guard,
        execution_check=execution_check, write_headroom_bytes=write_headroom_bytes,
        include_auxiliary_grid=False)
    mapping = post["sample_index_to_inferred_tree_node_id"]
    _need(len(mapping) == len(assets["rows"]) == 952
          and post["samples"] == 952 and len(post["query_keys"]) == 686,
          "All AP inferred samples and native-six query identities required")
    inferred_rows = []
    for row, mapped in zip(assets["rows"], mapping):
        _need(row["sample_index"] == mapped["sample_index"], "Native inferred sample mapping changed")
        item = dict(row)
        item["source_observed_node_id"] = item.pop("observed_node_id")
        item["inferred_tree_node_id"] = mapped["inferred_tree_node_id"]
        item["observed_node_id"] = mapped["inferred_tree_node_id"]
        inferred_rows.append(item)
    full.json_write(work / "observed_sample_manifest.json", inferred_rows, guard)
    _need(assets["source_sha256"] == {name: full.sha(path) for name, path in assets["paths"].items()},
          "Original observed input bytes changed during native inference")
    _need(all(record["sha256"] == full.sha(full.confined(record["path"]))
              for record in pre_refs.values()),
          "Native size iterations modified original initial tree files")
    receipt = dict(
        schema="AP_short_development_native_inference_v1", parent_id=parent["parent_id"],
        short_development_plan_ref=plan_ref(), development_model_selection_only=True,
        split=parent["split"], status="post_five_iterations_inferred",
        sequence_length_bp=length, truth_or_oracle_read=False,
        inferred_tree=full.ref(str((work / "post_size.observed.trees").relative_to(full.ROOT))),
        observed_sample_manifest=full.ref(str((work / "observed_sample_manifest.json").relative_to(full.ROOT))),
        post_five_iterations=post, input_observed_sha256=assets["source_sha256"],
        filter_counts=counts, native_converter_sites=len(retained),
        converter_additional_attrition=len(eligible) - len(retained),
        commands=full.ref(str((work / "commands.json").relative_to(full.ROOT))),
        outer_command_returncodes=[record["returncode"] for record in journal],
        native_status_scope="outer helper returncodes; supplied native runner retains trace",
        native_tools={key: {"path": path, "sha256": full.sha(path)}
                      for key, path in native.items()},
        pre_size_native_files_unchanged=pre_refs,
        all_nine_cohorts_retained=True, native_six_path_endpoint_retained=True,
        sampled_registry_prepared=False, dense_grid_loaded=False,
        operational_adapter=dict(threads=8, native_memory_gb=2,
            explicit_seeds=True, explicit_released_defaults=dict(threshold=0.5,
                years_per_generation=28), plotting_disabled=True,
            observed_preprocessing="existing source-pinned VCF filter and canonical sample formatting",
            coordinate_cleanup="existing full-length minus-one and actual-coverage clipping"),
        paper_comparison_completed=False)
    full.json_write(work / "inference.json", receipt, guard)
    return receipt
