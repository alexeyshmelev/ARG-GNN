"""Full-length observed AP assets and native Relate stage orchestration.

No process or scientific runtime starts at import. The caller supplies current
approval/resource checks and an explicitly bounded native-step runner. Existing
licensed binaries, VCF filtering, HAPS checks, sample ordering and tree cleanup
are reused. This adapter reads only the observed whitelist, never truth.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from . import european_execution as stages

OBSERVED_FILES = (
    "genotypes.vcf", "sample_manifest.json", "sample_order.txt",
    "sample_ages.generations", "population_labels.txt",
    "site_coordinate_manifest.json", "uniform_recombination_map.txt",
    "scoring_coordinates.npz", "observation_manifest.json")
ROSTER_FIELDS = ("sample_index", "sample_id", "haplotype", "observed_cohort",
                 "age_generations", "role", "observed_reference_class")


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _text(path, cap):
    _need(path.stat().st_size <= cap, "Observed metadata exceeds its explicit read cap")
    return path.read_text()


def _json(path, cap):
    return json.loads(_text(path, cap))


def _observed_assets(recipe, parent, observed_directory, expected_input_refs, *,
                     full, guard, max_metadata_bytes):
    """Length-aware replacement for the historical 2 Mb observed-only loader."""
    _need(type(max_metadata_bytes) is int and max_metadata_bytes > 0,
          "Explicit bounded observed-metadata read size required")
    observed = full.confined(observed_directory)
    _need(observed.is_dir() and observed.resolve() == observed,
          "Canonical observed input directory required")
    paths = {name: observed / name for name in OBSERVED_FILES}
    _need(isinstance(expected_input_refs, dict)
          and set(expected_input_refs) == set(OBSERVED_FILES),
          "Exact observed whitelist bindings required")
    actual_refs = {}
    for name, path in paths.items():
        _need(not path.is_symlink() and path.is_file() and path.resolve() == path,
              "Observed whitelist file must be canonical and regular")
        guard(0)
        actual_refs[name] = {
            "path": str(path.relative_to(full.ROOT)), "sha256": full.sha(path)}
    _need(actual_refs == expected_input_refs,
          "Observed input hashes differ from immutable admitted bindings")
    length = recipe["parameters"]["length_bp"]
    manifest = _json(paths["observation_manifest.json"], max_metadata_bytes)
    expected = full.roster()
    _need(manifest.get("parent_id") == parent["parent_id"]
          and manifest.get("split") == parent["split"]
          and manifest.get("config_kind") == "ancestralpaths_european"
          and manifest.get("length_bp") == length
          and manifest.get("samples") == 952 and manifest.get("diploids") == 476
          and manifest.get("query_haplotypes") == 686
          and manifest.get("pure_reference_haplotypes") == 266
          and manifest.get("census_or_ancestry_annotations_present") is False
          and manifest.get("cleaned_oracle_exported") is False
          and manifest.get("Relate_executed") is False
          and manifest.get("sample_template_sha256") == full.digest(expected),
          "Complete canonical AP observed-only parent metadata required")
    rows = _json(paths["sample_manifest.json"], max_metadata_bytes)
    _need(isinstance(rows, list) and len(rows) == len(expected) == 952
          and all(all(row.get(k) == source[k] for k in ROSTER_FIELDS)
                  for row, source in zip(rows, expected)),
          "All AP sample/date/cohort/reference identities must remain unchanged")
    _need(all(type(r.get("observed_node_id")) is int and r["observed_node_id"] >= 0
              for r in rows)
          and len({r["observed_node_id"] for r in rows}) == 952,
          "Actual observed sample node identities required")
    names = [r["sample_id"] for r in rows[::2]]
    _need(_text(paths["sample_order.txt"], max_metadata_bytes).splitlines() == names,
          "Canonical AP diploid order required")
    ages = [r["age_generations"] for r in rows]
    _need([float(v) for v in _text(paths["sample_ages.generations"],
                                  max_metadata_bytes).split()] == ages,
          "All AP sample ages must retain their original order")
    labels = [line.split() for line in _text(paths["population_labels.txt"],
                                             max_metadata_bytes).splitlines()]
    _need(labels == [["sample", "population", "group", "sex"]] + [
        [r["sample_id"], r["observed_cohort"], r["observed_cohort"], "NA"]
        for r in rows[::2]], "All nine AP cohort labels must remain unchanged")

    coordinate = _json(paths["site_coordinate_manifest.json"], max_metadata_bytes)
    _need(coordinate.get("length_bp") == length and coordinate.get("contig_id") == "22"
          and isinstance(coordinate.get("rows"), list),
          "Complete observed site domain and coordinate manifest required")
    positions, previous = {}, 0
    for index, row in enumerate(coordinate["rows"]):
        pos = row.get("VCF_position_1based")
        _need(type(pos) is int and previous < pos <= length
              and row.get("tree_position_0based") == pos - 1
              and row.get("site_id") == index,
              "Unique sorted full-domain observed SNP coordinates required")
        positions[pos] = row
        previous = pos
    _need(len(positions) > 0 and len(positions) == manifest.get("sites"),
          "Complete nonempty observed SNP registry required")
    rate = recipe["parameters"]["recombination_rate"]
    map_text = _text(paths["uniform_recombination_map.txt"], max_metadata_bytes).split()
    expected_map = ["Position(bp)", "Rate(cM/Mb)", "Map(cM)", "1",
                    f"{rate * 1e8:g}", "0", str(length + 1),
                    f"{rate * 1e8:g}", f"{length * rate * 100:g}"]
    _need(map_text == expected_map, "Canonical complete-domain AP uniform map required")
    # Native sampled training does not read/materialize the diagnostic grid.
    return dict(paths=paths, rows=rows, names=names, ages=ages,
                position_join=positions,
                source_sha256={name: item["sha256"] for name, item in actual_refs.items()})


def load_observed_assets(recipe, parent, observed_directory, expected_input_refs, *,
                         guard, execution_check, write_headroom_bytes,
                         max_metadata_bytes):
    """Load only source-bound observed assets, with no truth or native process."""
    stages._approved(recipe, guard, execution_check, write_headroom_bytes)
    _, _, full, _ = stages._runtime()
    return _observed_assets(
        recipe, parent, observed_directory, expected_input_refs, full=full,
        guard=guard, max_metadata_bytes=max_metadata_bytes)


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
    length = stages._approved(recipe, guard, execution_check, write_headroom_bytes)
    _need(callable(native_step), "Caller-owned bounded native-step runner required")
    _need(length == 200_000_000, "Complete canonical 200 Mb AP inference required")
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
        schema="AP_full200mb_native_inference_v1", parent_id=parent["parent_id"],
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
