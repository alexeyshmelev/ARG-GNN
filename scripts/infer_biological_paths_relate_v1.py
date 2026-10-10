#!/usr/bin/env python
"""Observed-only biological Relate chain; no inference without root --execute."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import tskit

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
NOTICE = "Copyright © 2018, University of Oxford"
SOURCE_FILES = ("genotypes.vcf", "sample_manifest.json", "sample_order.txt",
                "sample_ages.generations", "population_labels.txt",
                "site_coordinate_manifest.json", "uniform_recombination_map.txt",
                "scoring_coordinates.npz", "observation_manifest.json")
PARENTS = tuple(f"bio_paths_pilot_{split}_{i:03d}" for split in ("train", "val") for i in range(2))


def confined(value):
    p = Path(value)
    p = (ROOT / p if not p.is_absolute() else p).resolve()
    if ROOT not in p.parents:
        raise ValueError("path must remain inside project")
    return p


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    with Path(path).open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def tools(config):
    # Existing verifier checks previously authorized archive/binaries/licence.
    from arggnn.relate_pipeline import verified_relate_installation
    installation = verified_relate_installation()
    root = confined(installation["release_root"])
    result = {}
    for relative, expected in config["tool_sha256"].items():
        p = root / relative
        if sha(p) != expected:
            raise ValueError("installed native tool/source drift: " + relative)
        result[relative] = str(p)
    return installation, result


def validate_sample_assets(folder, parent_id):
    if folder != confined("data/biological_paths_execution_pilot_v1") / parent_id / "observed":
        raise ValueError("only the dedicated observed namespace is allowed")
    # Explicit whitelist: no oracle, parent truth manifest, targets or true trees.
    paths = {name: folder / name for name in SOURCE_FILES}
    if any(p.is_symlink() or not p.is_file() for p in paths.values()):
        raise ValueError("required observed assets missing/nonregular")
    metadata = json.loads(paths["observation_manifest.json"].read_text())
    if (metadata["parent_id"] != parent_id or metadata["samples"] != 280
            or metadata["diploids"] != 140 or metadata["query_haplotypes"] != 120
            or metadata["pure_reference_haplotypes"] != 96):
        raise ValueError("biological observed140/280 panel differs")
    rows = json.loads(paths["sample_manifest.json"].read_text())
    names = paths["sample_order.txt"].read_text().splitlines()
    ages = np.loadtxt(paths["sample_ages.generations"], ndmin=1)
    if len(rows) != 280 or len(names) != 140 or len(set(names)) != 140 or ages.shape != (280,):
        raise ValueError("observed sample/age count differs")
    if not np.isfinite(ages).all() or (ages < 0).any():
        raise ValueError("invalid observed ages")
    if [r["sample_index"] for r in rows] != list(range(280)):
        raise ValueError("observed haplotype order differs")
    for i, row in enumerate(rows):
        if row["sample_id"] != names[i // 2] or row["haplotype"] != i % 2 or row["age_generations"] != ages[i]:
            raise ValueError("sample/name/haplotype/age join differs")
        if row["role"] != "pure_source_reference" and row["observed_reference_class"] is not None:
            raise ValueError("query/mixed reference carries a local source label")
    for i in range(0, 280, 2):
        if rows[i]["observed_cohort"] != rows[i + 1]["observed_cohort"] or ages[i] != ages[i + 1]:
            raise ValueError("diploid pair cohort/date differs")
    counts = {role: sum(r["role"] == role for r in rows) for role in
              ("pure_source_reference", "mixed_reference_only", "query")}
    if counts != {"pure_source_reference": 96, "mixed_reference_only": 64, "query": 120}:
        raise ValueError("observed role counts differ")
    labels = [line.split() for line in paths["population_labels.txt"].read_text().splitlines()]
    expected = [["sample", "population", "group", "sex"]] + [
        [names[i], rows[2*i]["observed_cohort"], rows[2*i]["observed_cohort"], "NA"] for i in range(140)]
    if labels != expected:
        raise ValueError("observed diploid population-label order differs")
    mapping = json.loads(paths["site_coordinate_manifest.json"].read_text())
    positions = {}
    for record in mapping["rows"]:
        vcf = record["VCF_position_1based"]
        if vcf != record["tree_position_0based"] + 1 or not 1 <= vcf <= 2000000 or vcf in positions:
            raise ValueError("observed SNP/VCF+1 coordinate join differs")
        positions[vcf] = record
    original_map = [line.split() for line in paths["uniform_recombination_map.txt"].read_text().splitlines()]
    if original_map != [["Position(bp)", "Rate(cM/Mb)", "Map(cM)"], ["1", "1", "0"], ["2000001", "1", "2"]]:
        raise ValueError("observed synthetic map differs")
    with np.load(paths["scoring_coordinates.npz"], allow_pickle=False) as grid:
        expected_positions = np.arange(50, 2000000, 100, dtype=np.float64)
        if not np.array_equal(grid["positions_bp"], expected_positions):
            raise ValueError("observed fixed100bp scoring coordinates differ")
    return {"paths": paths, "rows": rows, "names": names, "ages": ages,
            "position_join": positions, "metadata": metadata,
            "source_sha256": {name: sha(p) for name, p in paths.items()}}


def filter_vcf(lines, names, position_join):
    """Observed-only biallelic segregating nucleotide/MAC1 filtering."""
    header, kept, counts = [], [], {"input": 0, "multiallelic_or_nonnucleotide": 0, "monomorphic": 0}
    saw_header = False
    previous = 0
    for line in lines:
        if line.startswith("#"):
            if line.startswith("#CHROM"):
                if line.rstrip().split("\t")[9:] != names:
                    raise ValueError("original VCF diploid order differs")
                saw_header = True
            header.append(line.rstrip())
            continue
        if not saw_header:
            raise ValueError("VCF has no ordered sample header")
        fields = line.rstrip().split("\t")
        if len(fields) != 9 + len(names):
            raise ValueError("VCF sample columns differ")
        counts["input"] += 1
        pos = int(fields[1])
        if fields[0] != "22" or pos <= previous or pos not in position_join:
            raise ValueError("VCF/SNP coordinate join or chromosome differs")
        previous = pos
        if fields[3] not in "ACGT" or len(fields[3]) != 1 or fields[4] not in "ACGT" or len(fields[4]) != 1 or fields[3] == fields[4]:
            counts["multiallelic_or_nonnucleotide"] += 1
            continue
        fmt = fields[8].split(":")
        if "GT" not in fmt:
            raise ValueError("VCF lacks genotype field")
        gt_index = fmt.index("GT")
        genotypes = []
        for value in fields[9:]:
            gt = value.split(":")[gt_index]
            if gt not in ("0|0", "0|1", "1|0", "1|1"):
                raise ValueError("unphased/missing/invalid observed genotype")
            genotypes.extend((int(gt[0]), int(gt[2])))
        if not 0 < sum(genotypes) < len(genotypes):
            counts["monomorphic"] += 1
            continue
        kept.append({"fields": fields, "pos1": pos, "genotypes": genotypes,
                     "original_site_id": position_join[pos]["site_id"]})
    if not kept:
        raise ValueError("no eligible observed SNPs")
    counts["eligible"] = len(kept)
    return header, kept, counts


def canonical_sample_text(text, names):
    """Keep original converter file; normalize only ID2 missing/duplicate schema."""
    lines = [x.split() for x in text.splitlines() if x.strip()]
    if lines[:2] != [["ID_1", "ID_2", "missing"], ["0", "0", "0"]] or len(lines[2:]) != len(names):
        raise ValueError("native .sample schema/count differs")
    for fields, name in zip(lines[2:], names):
        if len(fields) != 3 or not (
                (fields[0] == name and fields[1] in (name, "NA")) or
                (fields[1] == name and fields[0] == "NA")):
            raise ValueError("native converter changed observed sample identities/order")
    return "ID_1 ID_2 missing\n0 0 0\n" + "".join(f"{name} {name} 0\n" for name in names)


def prepare_inputs(assets, destination):
    destination.mkdir()
    with assets["paths"]["genotypes.vcf"].open() as stream:
        header, rows, counts = filter_vcf(stream, assets["names"], assets["position_join"])
    with (destination / "input.vcf").open("x") as stream:
        stream.write("\n".join(header) + "\n")
        stream.write("\n".join("\t".join(r["fields"]) for r in rows) + "\n")
    # Header-only map adapter: exact numeric entries and VCF+1 origin unchanged.
    (destination / "input.map").write_text("pos COMBINED_rate Genetic_Map\n1 1 0\n2000001 1 2\n")
    (destination / "sample_ages.txt").write_text(assets["paths"]["sample_ages.generations"].read_text())
    (destination / "input.poplabels").write_text(assets["paths"]["population_labels.txt"].read_text())
    write_json(destination / "original_sample_manifest.json", assets["rows"])
    write_json(destination / "preparation.json", {
        "observed_source_sha256": assets["source_sha256"], "site_filter_counts": counts,
        "map_adapter": "only canonical Relate header; exact original map numeric rows",
        "polarization": "original observed VCF REF at retained simulated mutation sites",
        "no_ancestral_FASTA_or_unobserved_polarization_invented": True,
        "truth_or_oracle_files_read": False,
    })
    return rows, counts


def verify_native_haps(path, eligible):
    expected = {row["pos1"]: row for row in eligible}
    retained = []
    previous = 0
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) != 5 + 280:
            raise ValueError("native haps lost haplotypes")
        pos = int(fields[2])
        if pos <= previous or pos not in expected:
            raise ValueError("native haps SNP position/order differs")
        previous = pos
        row = expected[pos]
        if fields[3:5] != row["fields"][3:5] or list(map(int, fields[5:])) != row["genotypes"]:
            raise ValueError("native haps changed observed alleles/phase")
        retained.append(pos)
    if not retained:
        raise ValueError("native converter exported no sites")
    return retained


def run_step(argv, work, journal, label, timeout):
    log = work / f"{len(journal):02d}_{label}.log"
    record = {"label": label, "argv": argv, "command_sha256": digest(argv),
              "cwd": str(work), "log": str(log), "status": "started"}
    journal.append(record)
    write_json(work / "commands.json", {"steps": journal, "copyright_notice": NOTICE})
    started = time.monotonic()
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", TMPDIR=str(ROOT / ".runtime/tmp"))
    try:
        with log.open("x") as stream:
            result = subprocess.run(argv, cwd=work, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                                    timeout=timeout, check=False)
        record.update(returncode=result.returncode, status="complete" if result.returncode == 0 else "failed")
    except Exception as error:
        record.update(status="exception", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic() - started
        if log.exists():
            record["log_sha256"] = sha(log)
        write_json(work / "commands.json", {"steps": journal, "copyright_notice": NOTICE})
    if record["returncode"] != 0:
        raise RuntimeError(f"{label} failed; preserve {log}")


def inferred_sample_mapping(inferred, assets):
    """Map observed haplotype columns to actual flagged converter node IDs.

    ConvertToTreeSequence appends sample nodes in .anc haplotype order; .anc
    inherits the checked .haps column and .sample diploid order. Enumerating
    actual flagged node rows retains that ordering without assuming ID 0..279.
    """
    sample_ids = inferred.samples().astype(np.int64)
    if len(sample_ids) != len(assets["rows"]):
        raise ValueError("converted flagged sample count differs")
    if not np.allclose(inferred.tables.nodes.time[sample_ids], assets["ages"], atol=1e-9, rtol=0):
        raise ValueError("converted sample ages differ in verified converter order")
    return [{"sample_index": row["sample_index"], "inferred_tree_node_id": int(node)}
            for row, node in zip(assets["rows"], sample_ids)]


def write_inferred_roster(assets, mapping, destination):
    rows = []
    for source, joined in zip(assets["rows"], mapping):
        if source["sample_index"] != joined["sample_index"]:
            raise ValueError("observed roster/mapping order differs")
        row = dict(source)
        row["source_observed_node_id"] = row.pop("observed_node_id", None)
        row["inferred_tree_node_id"] = joined["inferred_tree_node_id"]
        row["observed_node_id"] = joined["inferred_tree_node_id"]
        rows.append(row)
    write_json(destination, rows)
    return {"path": str(destination), "sha256": sha(destination), "rows": len(rows),
            "source_manifest": {"path": str(assets["paths"]["sample_manifest.json"]),
                                "sha256": assets["source_sha256"]["sample_manifest.json"]},
            "observed_node_id_alias": "actual inferred_tree_node_id for this condition"}


def clean_converted_tree(native_file, assets, retained_vcf, destination, condition):
    """Convert retained VCF coordinates to clipped original physical support."""
    inferred = tskit.load(native_file)
    mapping = inferred_sample_mapping(inferred, assets)
    site_positions = inferred.tables.sites.position
    if not np.array_equal(site_positions, site_positions.astype(np.int64)):
        raise ValueError("converter SNP positions are not exact integers")
    actual = site_positions.astype(np.int64).tolist()
    if not actual or any(p not in retained_vcf for p in actual) or len(set(actual)) != len(actual):
        raise ValueError("converted SNPs cannot join observed haps")
    left, right = float(min(actual) - 1), float(max(actual))
    if not 0 <= left < right <= 2000000:
        raise ValueError("converted physical coverage invalid")
    tables = inferred.dump_tables()
    edge = tables.edges
    shifted_left = np.maximum(edge.left - 1, left)
    shifted_right = np.minimum(edge.right - 1, right)
    keep = shifted_right > shifted_left
    parent, child = edge.parent[keep], edge.child[keep]
    tables.edges.clear()
    tables.edges.set_columns(left=shifted_left[keep] - left, right=shifted_right[keep] - left,
                             parent=parent, child=child)
    tables.sites.position = site_positions - 1 - left
    tables.sequence_length = right - left
    tables.nodes.population = np.full(inferred.num_nodes, tskit.NULL, dtype=np.int32)
    tables.nodes.individual = np.full(inferred.num_nodes, tskit.NULL, dtype=np.int32)
    tables.nodes.flags = tables.nodes.flags & tskit.NODE_IS_SAMPLE
    for table in (tables.nodes, tables.edges, tables.sites, tables.mutations):
        table.drop_metadata()
    tables.migrations.clear()
    tables.populations.clear()
    tables.individuals.clear()
    tables.provenances.clear()
    tables.metadata_schema = tskit.MetadataSchema(None)
    tables.metadata = b""
    tables.sort()
    clean = tables.tree_sequence()
    if inferred_sample_mapping(clean, assets) != mapping or clean.num_nodes != inferred.num_nodes:
        raise ValueError("clipping changed native inferred node IDs/order")
    clean.dump(destination)
    positions = np.arange(50, 2000000, 100, dtype=np.float64)
    retained_grid = np.flatnonzero((positions >= left) & (positions < right))
    query_rows = [r for r in assets["rows"] if r["role"] == "query"]
    result = {
        "condition": condition, "native_tree_sha256": sha(native_file), "clean_tree_sha256": sha(destination),
        "native_tree": str(native_file), "clean_tree": str(destination),
        "coverage_original_bp0": [left, right], "physical_coverage_fraction": (right-left)/2000000,
        "local_to_original_offset_bp": left, "sequence_length_local_bp": clean.sequence_length,
        "original_parent_interval_bp0": [0, 2000000], "normalization_length_bp": 2000000,
        "physical_coordinate_transform": "physical_original = local_tree_position + coverage_left",
        "sample_index_to_inferred_tree_node_id": mapping,
        "mapping_basis": "actual NODE_IS_SAMPLE rows in converter insertion order; verified VCF diploid header, .sample order, phased .haps columns, and observed ages",
        "node_ids_preserved_during_clipping": True,
        "converter_VCF_to_original_offset_bp": -1,
        "retained_fixed100bp_indexes": retained_grid.tolist(), "fixed_grid_total": 20000,
        "fixed_grid_retained": len(retained_grid), "samples": 280,
        "sample_age_order_preserved": True, "sites": clean.num_sites, "trees": clean.num_trees,
        "query_keys": [[r["sample_id"], r["haplotype"]] for r in query_rows],
        "source_class_labels": "observed pure-reference labels only; no target loaded",
        "site_join": [{"native_site_id": i, "VCF_position_1based": p,
                      "original_position_0based": p-1, "local_position_0based": p-1-left,
                      "original_site_id": assets["position_join"][p]["site_id"]}
                     for i, p in enumerate(actual)],
    }
    return result


def infer_parent(config, parent, target, native):
    source = confined(config["source_root"]) / parent / "observed"
    assets = validate_sample_assets(source, parent)
    work = target / parent
    work.mkdir()
    (work / "NOTICE.txt").write_text("Relate1.2.4\n" + NOTICE + "\nPreviously authorized internal academic research use.\n")
    inputs = work / "inputs"
    eligible, counts = prepare_inputs(assets, inputs)
    journal = []
    f = native["bin/RelateFileFormats"]
    timeout = config["timeout_seconds"]
    run_step([f, "--mode", "ConvertFromVcf", "--haps", str(inputs/"input.haps"),
              "--sample", str(inputs/"converted.sample"), "-i", str(inputs/"input")], work, journal, "convert_vcf", timeout)
    (inputs/"input.sample").write_text(canonical_sample_text((inputs/"converted.sample").read_text(), assets["names"]))
    retained = verify_native_haps(inputs/"input.haps", eligible)
    run_step([f, "--mode", "GenerateSNPAnnotations", "--haps", str(inputs/"input.haps"),
              "--sample", str(inputs/"input.sample"), "--poplabels", str(inputs/"input.poplabels"),
              "-o", str(inputs/"input")], work, journal, "annotate_observed_cohorts", timeout)
    seeds = config["parent_seeds"][parent]
    run_step(["bash", "-x", native["scripts/RelateParallel/RelateParallel.sh"],
              "-m", "1.25e-8", "-N", "30000", "--threads", "4", "--memory", "2",
              "--seed", str(seeds["initial"]), "--haps", str(inputs/"input.haps"),
              "--sample", str(inputs/"input.sample"), "--sample_ages", str(inputs/"sample_ages.txt"),
              "--map", str(inputs/"input.map"), "--annot", str(inputs/"input.annot"),
              "-o", "pre_size"], work, journal, "relate_initial", timeout)
    run_step([f, "--mode", "ConvertToTreeSequence", "-i", "pre_size", "-o", "pre_size"], work, journal, "convert_pre_size", timeout)
    pre = clean_converted_tree(work/"pre_size.trees", assets, retained, work/"pre_size.observed.trees", "pre_size_fixed_haploid_N30000")
    pre_assets = {str(p): sha(p) for p in [work/"pre_size.anc", work/"pre_size.mut", work/"pre_size.trees"]}
    run_step(["bash", "-x", native["scripts/EstimatePopulationSize/EstimatePopulationSize.sh"],
              "-i", "pre_size", "-o", "post_size", "-m", "1.25e-8", "--threads", "1",
              "--num_iter", "5", "--threshold", "0", "--years_per_gen", "28",
              "--seed", str(seeds["size_base"]), "--poplabels", str(inputs/"input.poplabels"),
              "--noplot"], work, journal, "five_population_size_iterations", timeout)
    run_step([f, "--mode", "ConvertToTreeSequence", "-i", "post_size", "-o", "post_size"], work, journal, "convert_post_size", timeout)
    post = clean_converted_tree(work/"post_size.trees", assets, retained, work/"post_size.observed.trees", "post_five_size_iterations_primary")
    pre["observed_sample_manifest"] = write_inferred_roster(
        assets, pre["sample_index_to_inferred_tree_node_id"], work/"pre_size.observed_sample_manifest.json")
    post["observed_sample_manifest"] = write_inferred_roster(
        assets, post["sample_index_to_inferred_tree_node_id"], work/"observed_sample_manifest.json")
    ordering = {
        "VCF_header_diploid_order": {"path": str(assets["paths"]["sample_order.txt"]),
                                     "sha256": assets["source_sha256"]["sample_order.txt"]},
        "original_VCF": {"path": str(assets["paths"]["genotypes.vcf"]),
                         "sha256": assets["source_sha256"]["genotypes.vcf"]},
        "filtered_VCF": {"path": str(inputs/"input.vcf"), "sha256": sha(inputs/"input.vcf")},
        "native_haps": {"path": str(inputs/"input.haps"), "sha256": sha(inputs/"input.haps")},
        "original_converter_sample": {"path": str(inputs/"converted.sample"), "sha256": sha(inputs/"converted.sample")},
        "canonical_converter_sample": {"path": str(inputs/"input.sample"), "sha256": sha(inputs/"input.sample")},
        "sample_ages_generations": {"path": str(inputs/"sample_ages.txt"), "sha256": sha(inputs/"sample_ages.txt")},
        "haplotype_column_order": "sample_index=2*diploid_header_index+haplotype(0,1)",
        "converter_sample_node_order": "actual NODE_IS_SAMPLE node rows, following verified .anc/.haps haplotype insertion order",
        "coordinate_source_reference": config["source_coordinate_reference"],
        "source_reference_scope": "ordering/coordinate implementation evidence; unsigned source commit is not asserted as binary build identity",
    }
    pre["converter_order_provenance"] = ordering
    post["converter_order_provenance"] = ordering
    if pre_assets != {name: sha(name) for name in pre_assets}:
        raise ValueError("population-size chain modified original pre-size native outputs")
    common_left=max(pre["coverage_original_bp0"][0],post["coverage_original_bp0"][0])
    common_right=min(pre["coverage_original_bp0"][1],post["coverage_original_bp0"][1])
    if common_right<=common_left:
        raise ValueError("pre/post conditions have no common observed coverage")
    # Original observed inputs are immutable, even though adapters live separately.
    if assets["source_sha256"] != {name: sha(p) for name,p in assets["paths"].items()}:
        raise ValueError("original observed assets changed during native chain")
    record = {"parent_id": parent, "split": "train" if "_train_" in parent else "val",
              "status": "inferred_both_conditions", "truth_or_oracle_read": False,
              "inferred_tree": {"path": post["clean_tree"], "sha256": post["clean_tree_sha256"],
                                "condition": post["condition"]},
              "pre_size_inferred_tree": {"path": pre["clean_tree"], "sha256": pre["clean_tree_sha256"],
                                         "condition": pre["condition"]},
              "observed_sample_manifest": post["observed_sample_manifest"],
              "sample_index_to_inferred_tree_node_id": post["sample_index_to_inferred_tree_node_id"],
              "original_observed_sample_manifest": post["observed_sample_manifest"]["source_manifest"],
              "coordinate_contract": {
                  "original_parent_interval_bp0": [0, 2000000], "normalization_length_bp": 2000000,
                  "coverage_original_bp0": post["coverage_original_bp0"],
                  "local_to_original_offset_bp": post["local_to_original_offset_bp"],
                  "physical_coordinate_transform": post["physical_coordinate_transform"],
                  "uncovered_tails": "excluded; no extrapolation",
                  "fixed_grid": {"path": str(assets["paths"]["scoring_coordinates.npz"]),
                                 "sha256": assets["source_sha256"]["scoring_coordinates.npz"],
                                 "center_formula_bp0": "50 + 100*grid_index", "total": 20000},
                  "retained_fixed100bp_indexes": post["retained_fixed100bp_indexes"],
                  "site_join_location": "post_five_iterations.site_join"},
              "input_observed_sha256": assets["source_sha256"], "filter_counts": counts,
              "native_converter_sites": len(retained), "converter_additional_attrition": len(eligible)-len(retained),
              "pre_size": pre, "post_five_iterations": post,
              "common_coverage_original_bp0": [common_left, common_right],
              "common_retained_fixed100bp_indexes": sorted(set(pre["retained_fixed100bp_indexes"]) & set(post["retained_fixed100bp_indexes"])),
              "pre_native_outputs_unchanged": pre_assets, "command_returncodes": [r["returncode"] for r in journal],
              "native_status_scope": "outer native/helper returncodes; bash-x internal trace retained, background child statuses not separately asserted",
              "commands_sha256": sha(work/"commands.json"),
              "commands_provenance": {"path": str(work/"commands.json"), "sha256": sha(work/"commands.json")},
              "native_input_provenance": ordering,
              "native_outputs": {p.name: {"sha256":sha(p),"bytes":p.stat().st_size}
                                 for p in work.iterdir() if p.is_file()},
              "copyright_notice": NOTICE}
    write_json(work/"inference.json", record)
    record["completion_provenance"] = {"path": str(work/"inference.json"), "sha256": sha(work/"inference.json")}
    return record


def handmade_schemas():
    names = ["a", "b"]
    text = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ta\tb\n22\t1\t0\tA\tG\t.\tPASS\t.\tGT\t0|1\t0|0\n22\t2\t1\tA\tG,T\t.\tPASS\t.\tGT\t0|1\t0|0\n"
    _, rows, counts = filter_vcf(io.StringIO(text), names, {1:{"site_id":0},2:{"site_id":1}})
    assert len(rows)==1 and rows[0]["pos1"]-1==0 and counts["multiallelic_or_nonnucleotide"]==1
    sample = canonical_sample_text("ID_1 ID_2 missing\n0 0 0\na NA 0\nb NA 0\n",names)
    assert sample.endswith("a a 0\nb b 0\n")
    tables = tskit.TableCollection(1)
    for flag, age in [(0, 20), (tskit.NODE_IS_SAMPLE, 0), (0, 30), (tskit.NODE_IS_SAMPLE, 10)]:
        tables.nodes.add_row(flags=flag, time=age)
    joined = inferred_sample_mapping(tables.tree_sequence(), {
        "rows": [{"sample_index": 0}, {"sample_index": 1}], "ages": np.array([0., 10.])})
    assert joined == [{"sample_index": 0, "inferred_tree_node_id": 1},
                      {"sample_index": 1, "inferred_tree_node_id": 3}]
    return {"handmade_VCF_plus1_and_sample_schema_checked": True,
            "handmade_noncontiguous_flagged_node_mapping_checked": True, "native_commands_executed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/biological_paths_relate_v1.json")
    parser.add_argument("--expected-config-sha256")
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--check-handmade-schemas",action="store_true")
    args = parser.parse_args()
    p = confined(args.config)
    config = json.loads(p.read_text())
    if config["parents"] != list(PARENTS) or config["population_size_iterations"] != 5:
        parser.error("only the four observed biological development parents/five iterations are supported")
    expected_options={"source_root":"data/biological_paths_execution_pilot_v1",
                      "output_root":"data/biological_paths_execution_pilot_v1/relate_inferred_v1", "mutation_rate":1.25e-8,
                      "initial_effective_haploid_size":30000,"initial_threads":4,"memory_gb":2,
                      "size_threads":1,"size_tree_mutation_threshold":0,
                      "years_per_generation_for_coal_outputs":28,"population_size_plotting":False,
                      "mutation_rate_normalization":False,"native_tool_release":"1.2.4"}
    if any(config.get(k)!=v for k,v in expected_options.items()):
        parser.error("fixed observed-only native execution options changed")
    installation,native = tools(config)
    if args.expected_config_sha256 is not None and sha(p)!=args.expected_config_sha256:
        parser.error("root-reviewed config bytes changed")
    if args.check_handmade_schemas:
        if args.execute:
            parser.error("schema check is separate from native execution")
        print(json.dumps(handmade_schemas()))
        return
    assets = [validate_sample_assets(confined(config["source_root"])/parent/"observed",parent) for parent in PARENTS]
    if not args.execute:
        print(json.dumps({"observed_parents_validated":4,"native_commands_executed":False,
                          "config_sha256":sha(p),"genotype_data_records_not_parsed":True,
                          "installation_version":installation["version"]},indent=2))
        return
    if not args.expected_config_sha256:
        parser.error("--execute requires root-reviewed --expected-config-sha256")
    target=confined(config["output_root"])
    target.mkdir(exist_ok=False)
    manifest={"kind":"biological_paths_relate_v1","schema_version":1,"status":"running","config_sha256":sha(p),
              "primary_condition":"post_five_size_iterations_primary",
              "original_parent_interval_bp0":[0,2000000],"normalization_length_bp":2000000,
              "copyright_notice":NOTICE,
              "tool_installation":installation,"actual_tools":{k:{"path":v,"sha256":sha(v)} for k,v in native.items()},
              "implementation_sha256":sha(Path(__file__)),"replicates":[],"truth_or_oracle_read":False}
    try:
        for parent in PARENTS:
            record=infer_parent(config,parent,target,native)
            manifest["replicates"].append(record)
            write_json(target/"manifest.partial.json",manifest)
            print(json.dumps({"parent_id":parent,"status":record["status"],"sites_post":record["post_five_iterations"]["sites"]}),flush=True)
        manifest["status"]="four_development_parents_inferred_pre_and_post_five_iterations"
    except Exception as error:
        manifest.update(status="failed_partial_preserved",failure_type=type(error).__name__,failure_message=str(error))
        write_json(target/"manifest.json",manifest)
        raise
    write_json(target/"manifest.json",manifest)


if __name__=="__main__":
    main()
