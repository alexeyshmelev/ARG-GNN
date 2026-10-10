"""Paired genotype-to-Relate inference with explicit coordinate/sample manifests.

The export reads mutations and observed sample roles only. Census ancestry stays
in external labels. Inferred inputs are packed from inferred topology and ages.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any, Sequence

import msprime
import numpy as np
import torch
import tskit

from .simulate import (
    SimulatedDataset, _project_path, load_replicate, nearest_tmrca_probabilities,
    pack_tree_sequence, save_replicate,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOL_ROOT = PROJECT_ROOT / ".runtime" / "tools" / "relate"
RELATE_VERSION = "1.2.4"
OFFICIAL_ARCHIVE = "https://myersgroup.github.io/relate/download/relate_v1.2.4_x86_64_static.tgz"
ARCHIVE_SHA256 = "9d6fe0f25846da9a846099b0d7c8d1f9f529aee0042e92803aef72b82e771438"
COPYRIGHT_NOTICE = "Copyright © 2018, University of Oxford"
LICENSE_URL = "https://raw.githubusercontent.com/MyersGroup/relate/b54ede259cbb0be095bc9c9a8bd18cdaf7e88b74/20180411_15670_RELATE_Academic_Usage_Licence.pdf"


@dataclass
class ObservedGenotypes:
    genotypes: np.ndarray             # [S,H], ancestral=0 / alternative=1
    positions: np.ndarray             # [S], exact integer zero-based positions
    site_ids: list[int]
    ancestral_alleles: list[str]
    alternative_alleles: list[str]
    haplotypes: list[dict[str, Any]]    # declared sample order, 0 then 1 per diploid
    sample_ids: list[str]
    sequence_length: int
    filter_counts: dict[str, int]


@dataclass(frozen=True)
class RelateConfig:
    mutation_rate: float = 1.25e-8
    effective_haploid_size: float = 30_000.
    memory_gb: float = .25
    seed: int = 1
    timeout_seconds: int = 180
    threads: int = 1

    def validate(self) -> None:
        if any(not np.isfinite(value) or value <= 0 for value in (self.mutation_rate, self.effective_haploid_size, self.memory_gb)):
            raise ValueError("Relate rate, haploid effective size and memory must be positive")
        if not 1 <= self.seed <= 2**31 - 1 or self.timeout_seconds < 1 or self.threads < 1:
            raise ValueError("Relate requires a positive signed-31-bit seed, timeout and thread count")


def load_compressed_tree_sequence(path: str | Path) -> tskit.TreeSequence:
    """tskit needs a file descriptor, so decompress in a confined scratch file."""
    path = Path(path)
    if path.suffix != ".gz":
        return tskit.load(path)
    scratch = TOOL_ROOT / "tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=scratch, suffix=".trees") as handle:
        with gzip.open(path, "rb") as source:
            shutil.copyfileobj(source, handle, length=1024 * 1024)
        handle.flush()
        return tskit.load(handle.name)


def load_ancient_parent(directory: str | Path) -> tuple[SimulatedDataset, tskit.TreeSequence]:
    directory = Path(directory)
    parent = load_replicate(directory)
    observed = directory / "observed.trees.gz"
    if not observed.is_file():
        raise ValueError("paired inference requires the cleaned observed.trees.gz, never raw census truth")
    return parent, load_compressed_tree_sequence(observed)


def _ordered_haplotypes(metadata: dict[str, Any], ts: tskit.TreeSequence, sample_order: Sequence[str] | None) -> tuple[list[dict[str, Any]], list[str]]:
    rows = metadata.get("samples")
    if not rows:
        raise ValueError("a diploid/haplotype sample manifest is required")
    groups: dict[str, dict[int, dict[str, Any]]] = {}
    for original in rows:
        row = dict(original)
        name, hap = str(row["sample_id"]), int(row["haplotype"])
        if not name or any(character.isspace() for character in name) or hap not in (0, 1):
            raise ValueError("invalid sample ID or haplotype index")
        if hap in groups.setdefault(name, {}):
            raise ValueError("duplicate sample/haplotype key")
        groups[name][hap] = row
    order = list(groups) if sample_order is None else list(sample_order)
    if len(order) != len(set(order)) or set(order) != set(groups):
        raise ValueError("sample order must name every diploid exactly once")
    ordered = []
    for name in order:
        if set(groups[name]) != {0, 1}:
            raise ValueError("each diploid must have both haplotypes")
        pair = [groups[name][0], groups[name][1]]
        if pair[0]["age_generations"] != pair[1]["age_generations"]:
            raise ValueError("diploid haplotypes have different sampling dates")
        if pair[0]["cohort_role"] != pair[1]["cohort_role"] or pair[0].get("reference_class") != pair[1].get("reference_class"):
            raise ValueError("diploid roles or reference classes disagree")
        ordered.extend(pair)
    nodes = [int(row["observed_node_id"]) for row in ordered]
    if len(set(nodes)) != len(nodes) or set(nodes) != set(int(u) for u in ts.samples()):
        raise ValueError("sample manifest does not match all observed tree leaves")
    for row in ordered:
        if not np.isclose(ts.node(int(row["observed_node_id"])).time, row["age_generations"], atol=1e-8, rtol=0):
            raise ValueError("observed sample age disagrees with manifest")
        if row["cohort_role"] == "query" and row.get("reference_class") is not None:
            raise ValueError("query local ancestry cannot be supplied as a reference class")
    return ordered, order


def collect_observed_genotypes(
    parent: SimulatedDataset, observed_ts: tskit.TreeSequence,
    *, sample_order: Sequence[str] | None = None, min_mac: int = 1,
) -> ObservedGenotypes:
    """Collect actually segregating, biallelic nucleotide variants, without rounding."""
    if isinstance(min_mac, bool) or not isinstance(min_mac, int) or min_mac < 1:
        raise ValueError("min_mac must be a positive integer")
    length = observed_ts.sequence_length
    if not float(length).is_integer() or length < 1:
        raise ValueError("VCF export requires an integer sequence length")
    if np.any(observed_ts.tables.nodes.flags & msprime.NODE_IS_CEN_EVENT):
        raise ValueError("export requires a census-free observation artifact")
    haplotypes, sample_ids = _ordered_haplotypes(parent.metadata, observed_ts, sample_order)
    nodes = [int(row["observed_node_id"]) for row in haplotypes]
    genotypes, positions, site_ids, ancestral, alternative = [], [], [], [], []
    counts = {"input_sites": observed_ts.num_sites, "monomorphic": 0, "non_biallelic": 0,
              "non_nucleotide": 0, "below_mac": 0, "exported_sites": 0}
    for variant in observed_ts.variants(samples=nodes, isolated_as_missing=True):
        position = variant.site.position
        if not float(position).is_integer():
            raise ValueError("noninteger mutation position cannot be silently rounded into VCF")
        raw = np.asarray(variant.genotypes)
        if np.any(raw < 0):
            raise ValueError("missing genotypes require a separate explicit observation policy")
        observed_states = {variant.alleles[int(g)] for g in raw}
        anc = variant.site.ancestral_state
        if len(observed_states) < 2:
            counts["monomorphic"] += 1
            continue
        if len(observed_states) != 2 or anc not in observed_states:
            counts["non_biallelic"] += 1
            continue
        alt = next(state for state in observed_states if state != anc)
        if anc not in "ACGT" or alt not in "ACGT" or len(anc) != 1 or len(alt) != 1:
            counts["non_nucleotide"] += 1
            continue
        encoded = np.array([int(variant.alleles[int(g)] == alt) for g in raw], dtype=np.int8)
        mac = min(int(encoded.sum()), len(encoded) - int(encoded.sum()))
        if mac < min_mac:
            counts["below_mac"] += 1
            continue
        genotypes.append(encoded)
        positions.append(int(position))
        site_ids.append(int(variant.site.id))
        ancestral.append(anc)
        alternative.append(alt)
    counts["exported_sites"] = len(positions)
    if not positions:
        raise ValueError("no eligible observed biallelic nucleotide variants")
    if len(positions) != len(set(positions)):
        raise ValueError("duplicate mutation coordinates are unsupported")
    return ObservedGenotypes(np.stack(genotypes), np.asarray(positions, dtype=np.int64), site_ids,
                             ancestral, alternative, haplotypes, sample_ids, int(length), counts)


def constant_map_rows(sequence_length: int, recombination_rate: float) -> list[tuple[int, float, float]]:
    if sequence_length < 1 or not np.isfinite(recombination_rate) or recombination_rate < 0:
        raise ValueError("invalid chromosome length or recombination rate")
    # p_VCF = p_original + 1, so original coordinate zero has cumulative cM zero at 1.
    rate_cm_per_mb = recombination_rate * 1e8
    return [(1, rate_cm_per_mb, 0.), (sequence_length + 1, rate_cm_per_mb, recombination_rate * sequence_length * 100)]


def prepare_relate_inputs(
    parent: SimulatedDataset, observed_ts: tskit.TreeSequence, directory: str | Path,
    *, sample_order: Sequence[str] | None = None, min_mac: int = 1,
    recombination_rate: float | None = None,
) -> dict[str, Any]:
    """Write VCF, already-polarized haps, sample ages, maps and auditable manifests."""
    directory = _project_path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    variants = collect_observed_genotypes(parent, observed_ts, sample_order=sample_order, min_mac=min_mac)
    rate = parent.metadata["config"]["recombination_rate"] if recombination_rate is None else recombination_rate
    paths = {name: directory / filename for name, filename in {
        "vcf": "input.vcf", "haps": "input.haps", "sample": "input.sample",
        "ages": "sample_ages.txt", "map": "constant.map", "poplabels": "input.poplabels",
        "sites": "site_manifest.tsv", "haplotypes": "haplotype_manifest.json",
    }.items()}
    with paths["vcf"].open("w") as vcf, paths["haps"].open("w") as haps, paths["sites"].open("w") as sites:
        vcf.write("##fileformat=VCFv4.2\n")
        vcf.write(f"##contig=<ID=1,length={variants.sequence_length}>\n")
        vcf.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Phased genotype">\n')
        vcf.write("\t".join(["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", *variants.sample_ids]) + "\n")
        sites.write("original_site_id\toriginal_bp0\tvcf_bp1\tancestral_allele\talternative_allele\tderived_count\n")
        for i, row in enumerate(variants.genotypes):
            position, anc, alt = int(variants.positions[i]), variants.ancestral_alleles[i], variants.alternative_alleles[i]
            identifier = f"site_{variants.site_ids[i]}"
            phased = [f"{int(row[j])}|{int(row[j+1])}" for j in range(0, len(row), 2)]
            vcf.write("\t".join(["1", str(position + 1), identifier, anc, alt, ".", "PASS", ".", "GT", *phased]) + "\n")
            haps.write(" ".join(["1", identifier, str(position + 1), anc, alt, *[str(int(g)) for g in row]]) + "\n")
            sites.write(f"{variants.site_ids[i]}\t{position}\t{position+1}\t{anc}\t{alt}\t{int(row.sum())}\n")
    with paths["sample"].open("w") as sample, paths["poplabels"].open("w") as labels:
        sample.write("ID_1 ID_2 missing\n0 0 0\n")
        labels.write("sample population group sex\n")
        for index, name in enumerate(variants.sample_ids):
            sample.write(f"{name} {name} 0\n")
            row = variants.haplotypes[2 * index]
            group = row.get("source_name") or "QUERY"
            labels.write(f"{name} {group} {group} NA\n")
    paths["ages"].write_text("".join(f"{row['age_generations']:.17g}\n" for row in variants.haplotypes))
    with paths["map"].open("w") as handle:
        handle.write("pos COMBINED_rate Genetic_Map\n")
        for position, r, cm in constant_map_rows(variants.sequence_length, float(rate)):
            handle.write(f"{position} {r:.17g} {cm:.17g}\n")
    paths["haplotypes"].write_text(json.dumps(variants.haplotypes, indent=2) + "\n")
    manifest = {
        "schema_version": 1, "parent_benchmark": parent.metadata["benchmark"],
        "split": parent.metadata["split"], "replicate": parent.metadata["replicate"],
        "sequence_length": variants.sequence_length, "sample_order": variants.sample_ids,
        "haplotypes": variants.haplotypes, "num_haplotypes": len(variants.haplotypes),
        "num_sites": len(variants.positions), "site_positions0": variants.positions.tolist(),
        "filter_counts": variants.filter_counts, "min_mac": min_mac,
        "coordinate_convention": "VCF/HAPS position = exact original integer position + 1",
        "map_convention": "cM(p_vcf)=100*r*(p_vcf-1)", "recombination_rate": float(rate),
        "polarization": "simulated ancestral allele at observed mutation sites only",
        "unobserved_ancestral_alleles": "not supplied; no invented ancestor FASTA",
        "paths": {name: str(path) for name, path in paths.items()},
        "files_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
        "genotype_matrix_sha256": hashlib.sha256(variants.genotypes.tobytes()).hexdigest(),
    }
    (directory / "preparation.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def read_relate_sample_order(path: str | Path) -> list[str]:
    lines = [line.split() for line in Path(path).read_text().splitlines() if line.strip()]
    if len(lines) < 3 or lines[0] != ["ID_1", "ID_2", "missing"] or lines[1] != ["0", "0", "0"]:
        raise ValueError("unexpected Relate diploid .sample header")
    if any(len(line) != 3 or line[0] != line[1] or line[1] == "NA" for line in lines[2:]):
        raise ValueError("this protocol requires unchanged diploid sample IDs")
    order = [line[0] for line in lines[2:]]
    if len(order) != len(set(order)):
        raise ValueError("duplicate .sample IDs")
    return order


def paired_inferred_dataset(
    inferred_ts: tskit.TreeSequence, parent: SimulatedDataset, preparation: dict[str, Any],
    *, inferred_sample_order: Sequence[str] | None = None,
    coordinate_offset: float = 1., inference_name: str = "Relate v1.2.4",
) -> SimulatedDataset:
    """Align inference leaves and score the exact external parent grid in SNP span.

    The offset is 1 for official Relate conversion, which retains VCF coordinates
    in tskit. Zero supports methods that infer directly in original coordinates.
    Both chromosome ends outside the observed variant span are masked explicitly.
    """
    order = preparation["sample_order"] if inferred_sample_order is None else list(inferred_sample_order)
    if len(order) != len(set(order)) or set(order) != set(preparation["sample_order"]):
        raise ValueError("inferred sample order does not preserve every parent diploid")
    rows = {(str(row["sample_id"]), int(row["haplotype"])): row for row in preparation["haplotypes"]}
    keys = [(name, hap) for name in order for hap in (0, 1)]
    leaves = [int(u) for u in inferred_ts.samples()]
    if len(leaves) != len(keys) or set(keys) != set(rows):
        raise ValueError("inferred haplotype count/identity mismatch")
    key_to_leaf = dict(zip(keys, leaves))
    for key, leaf in key_to_leaf.items():
        if not np.isclose(inferred_ts.node(leaf).time, rows[key]["age_generations"], rtol=0, atol=1e-5):
            raise ValueError("inferred sample ages do not match the original ordered age manifest")
    original_key_by_node = {int(row["observed_node_id"]): key for key, row in rows.items()}
    query_keys = [original_key_by_node[int(u)] for u in parent.sequence.query_node_ids]
    query_leaves = [key_to_leaf[key] for key in query_keys]
    references = {key_to_leaf[key]: int(row["reference_class"]) for key, row in rows.items() if row["cohort_role"] == "reference"}
    if set(query_leaves) & set(references):
        raise ValueError("queries leaked into inference references")
    length = int(preparation["sequence_length"])
    input_positions = np.asarray(preparation["site_positions0"], dtype=np.float64)
    if input_positions.size == 0:
        raise ValueError("inference has no observed variant span")
    left, right = float(input_positions.min()), float(input_positions.max() + 1)
    # Converter source preserves VCF coordinates, but never extend terminal trees
    # into unobserved chromosome tails or shift target truth coordinates silently.
    tables = inferred_ts.dump_tables()
    site_positions = tables.sites.position - coordinate_offset
    if np.any(site_positions < left) or np.any(site_positions >= right) or not np.all(np.isin(site_positions, input_positions)):
        raise ValueError("inferred site positions do not match the explicit export coordinate mapping")
    if np.any(tables.nodes.flags & msprime.NODE_IS_CEN_EVENT):
        raise ValueError("inferred inputs cannot contain census events")
    old = tables.edges
    edge_left = np.maximum(old.left - coordinate_offset, left)
    edge_right = np.minimum(old.right - coordinate_offset, right)
    keep_edges = edge_right > edge_left
    parents, children = old.parent[keep_edges], old.child[keep_edges]
    tables.edges.clear()
    tables.edges.set_columns(left=edge_left[keep_edges] - left, right=edge_right[keep_edges] - left,
                             parent=parents, child=children)
    tables.sites.position = site_positions - left
    tables.nodes.population = np.full(inferred_ts.num_nodes, tskit.NULL, dtype=np.int32)
    tables.nodes.individual = np.full(inferred_ts.num_nodes, tskit.NULL, dtype=np.int32)
    tables.nodes.flags = tables.nodes.flags & tskit.NODE_IS_SAMPLE
    for table in (tables.nodes, tables.edges, tables.sites, tables.mutations):
        table.drop_metadata()
    tables.populations.clear()
    tables.individuals.clear()
    tables.migrations.clear()
    tables.provenances.clear()
    tables.metadata_schema = tskit.MetadataSchema(None)
    tables.metadata = b""
    tables.sequence_length = right - left
    tables.sort()
    observation = tables.tree_sequence()
    for tree in observation.trees():
        if len(tree.roots) != 1 or set(tree.samples()) != set(leaves):
            raise ValueError("inference does not provide a complete single-root marginal tree over the scoring span")
    original_positions = parent.sequence.positions.cpu().numpy()
    physical_positions = original_positions * length
    retained = np.flatnonzero((physical_positions >= left) & (physical_positions < right))
    if not len(retained):
        raise ValueError("no original fixed scoring coordinate lies in observed variant coverage")
    local_positions = (physical_positions[retained] - left) / (right - left)
    sequence, landmarks = pack_tree_sequence(
        observation, references, query_leaves, local_positions,
        num_classes=int(parent.metadata["num_classes"]),
        landmarks_per_source=int(parent.metadata["config"]["landmarks_per_source"]),
        max_graph_nodes=int(parent.metadata["config"].get("max_graph_nodes", 500_000)),
    )
    # Keep the parent's original chromosome coordinates and exact external labels.
    sequence.intervals = (sequence.intervals * (right - left) + left) / length
    sequence.positions = parent.sequence.positions[torch.from_numpy(retained)]
    sequence.window_bounds = sequence.intervals.new_tensor([left / length, right / length])
    sequence.validate()
    baseline = nearest_tmrca_probabilities(observation, references, query_leaves,
                                           physical_positions[retained] - left, int(parent.metadata["num_classes"]))
    metadata = dict(parent.metadata)
    sample_rows = []
    for key in keys:
        row = dict(rows[key])
        row["oracle_observed_node_id"] = row["observed_node_id"]
        row["observed_node_id"] = key_to_leaf[key]
        sample_rows.append(row)
    metadata.update({
        "benchmark": "ancient-europe-stdpopsim-0.3.0-relate-inferred",
        "parent_benchmark": parent.metadata["benchmark"],
        "input_kind": "inferred_genealogy", "ancestry_inference": inference_name,
        "samples": sample_rows, "reference_classes": {str(u): c for u, c in references.items()},
        "landmark_node_ids": landmarks, "num_trees": observation.num_trees,
        "packed_nodes": int(sequence.x.shape[0]), "num_sites": observation.num_sites,
        "inferred_sample_order": order, "coordinate_offset_from_export": coordinate_offset,
        "coverage_policy": "observed variant span; no chromosome-tail extrapolation",
        "coverage_physical_bp": [left, right], "physical_coverage_fraction": (right - left) / length,
        "parent_grid_points": len(original_positions), "retained_parent_grid_indexes": retained.tolist(),
        "grid_coverage_fraction": len(retained) / len(original_positions),
        "internal_population_annotations_present": False,
        "parent_target_sha256": hashlib.sha256(parent.labels.cpu().numpy().tobytes()).hexdigest(),
    })
    return SimulatedDataset(sequence, parent.labels[torch.from_numpy(retained)], torch.from_numpy(baseline), observation, metadata)


def verified_relate_installation() -> dict[str, Any]:
    path = TOOL_ROOT / "installation.json"
    if not path.exists():
        raise ValueError("official Relate installation provenance is missing")
    installation = json.loads(path.read_text())
    if installation.get("version") != RELATE_VERSION or installation.get("archive_sha256") != ARCHIVE_SHA256:
        raise ValueError("Relate release provenance differs from the pinned official archive")
    license_path = _project_path(installation["license_file"])
    if hashlib.sha256(license_path.read_bytes()).hexdigest() != installation["license_sha256"]:
        raise ValueError("reviewed Relate licence checksum mismatch")
    for name in ("Relate", "RelateFileFormats"):
        record = installation["binaries"][name]
        binary = _project_path(record["path"])
        if TOOL_ROOT not in binary.parents or not binary.is_file() or hashlib.sha256(binary.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Relate binary checksum or location mismatch")
    for record in installation.get("scripts", {}).values():
        script = _project_path(record["path"])
        if TOOL_ROOT not in script.parents or not script.is_file() or hashlib.sha256(script.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Relate script checksum or location mismatch")
    return installation


def relate_commands(preparation: dict[str, Any], directory: str | Path, installation: dict[str, Any], config: RelateConfig) -> list[list[str]]:
    config.validate()
    directory = _project_path(directory)
    paths = preparation["paths"]
    prefix = Path("inferred")  # Relate requires a basename in its working directory.
    if "RelateParallel" in installation.get("scripts", {}):
        first = ["bash", installation["scripts"]["RelateParallel"]["path"]]
        extra = ["--threads", str(config.threads)]
    else:
        first = [installation["binaries"]["Relate"]["path"], "--mode", "All"]
        extra = []
    commands = [first + ["-m", str(config.mutation_rate), "-N", str(config.effective_haploid_size),
                         "--haps", paths["haps"], "--sample", paths["sample"], "--sample_ages", paths["ages"],
                         "--map", paths["map"], "--memory", str(config.memory_gb), "--seed", str(config.seed), "-o", str(prefix)] + extra,
                [installation["binaries"]["RelateFileFormats"]["path"], "--mode", "ConvertToTreeSequence", "-i", str(prefix), "-o", str(prefix)]]
    return commands


def run_paired_relate(
    parent: SimulatedDataset, preparation: dict[str, Any], directory: str | Path,
    *, config: RelateConfig | None = None,
) -> SimulatedDataset:
    """One explicit CPU inference run after reviewed-licence authorization.

    Human authorization on 2026-10-10 covers this research use, not redistribution.
    This function does not download tools, accept new terms, or submit GPU work.
    """
    config = config or RelateConfig()
    directory = _project_path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any((directory / name).exists() for name in ("inferred.anc", "inferred.mut", "inferred.trees", "inferred")):
        raise ValueError("choose a fresh inference directory, existing outputs are preserved")
    installation = verified_relate_installation()
    (directory / "NOTICE.txt").write_text(f"Relate {RELATE_VERSION}\n{COPYRIGHT_NOTICE}\nReviewed academic research licence: {LICENSE_URL}\n")
    if read_relate_sample_order(preparation["paths"]["sample"]) != preparation["sample_order"]:
        raise ValueError(".sample order changed after preparation")
    for name, expected_hash in preparation["files_sha256"].items():
        if hashlib.sha256(Path(preparation["paths"][name]).read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"prepared input changed: {name}")
    ages = np.loadtxt(preparation["paths"]["ages"], ndmin=1)
    expected = np.array([row["age_generations"] for row in preparation["haplotypes"]])
    if ages.shape != expected.shape or not np.allclose(ages, expected, atol=1e-8, rtol=0):
        raise ValueError("haplotype ages changed after preparation")
    commands = relate_commands(preparation, directory, installation, config)
    records = []
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", TMPDIR=str(PROJECT_ROOT / ".runtime" / "tmp"))
    for index, command in enumerate(commands):
        started = time.perf_counter()
        log = directory / f"step{index}.log"
        with log.open("w") as handle:
            result = subprocess.run(command, cwd=directory, env=environment, stdout=handle, stderr=subprocess.STDOUT,
                                    timeout=config.timeout_seconds, check=False)
        records.append({"argv": command, "cwd": str(directory), "returncode": result.returncode, "elapsed_seconds": time.perf_counter() - started, "log": str(log)})
        (directory / "execution.json").write_text(json.dumps({"steps": records, "relate_installation": installation,
                                                             "copyright_notice": COPYRIGHT_NOTICE}, indent=2) + "\n")
        if result.returncode != 0:
            raise RuntimeError(f"Relate step {index} failed, see {log}")
    inferred = tskit.load(directory / "inferred.trees")
    paired = paired_inferred_dataset(inferred, parent, preparation,
                                      inferred_sample_order=read_relate_sample_order(preparation["paths"]["sample"]))
    paired.metadata["inference_provenance"] = {"steps": records, "relate_installation": installation, "copyright_notice": COPYRIGHT_NOTICE}
    destination = directory / "dataset"
    paired.metadata["inference_tree_file"] = str(directory / "inferred.trees")
    packed_only = SimulatedDataset(paired.sequence, paired.labels, paired.baseline_probabilities, None, paired.metadata)
    save_replicate(packed_only, destination)
    raw_observation = destination / "observed.trees"
    paired.tree_sequence.dump(raw_observation)
    with raw_observation.open("rb") as source, gzip.open(destination / "observed.trees.gz", "wb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
    raw_observation.unlink()
    return paired


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run", action="store_true", help="Execute one paired CPU Relate run using already verified tools")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    parent, observed = load_ancient_parent(args.parent)
    preparation = prepare_relate_inputs(parent, observed, Path(args.output) / "inputs")
    if args.run:
        paired = run_paired_relate(parent, preparation, Path(args.output) / "inference", config=RelateConfig(seed=args.seed))
        print(json.dumps({"num_trees": paired.sequence.num_graphs, "retained_grid_points": len(paired.sequence.positions),
                          "grid_coverage_fraction": paired.metadata["grid_coverage_fraction"]}))
    else:
        print(json.dumps({"prepared_sites": preparation["num_sites"], "haplotypes": preparation["num_haplotypes"], "status": "prepared, inference not run"}))


if __name__ == "__main__":
    main()
