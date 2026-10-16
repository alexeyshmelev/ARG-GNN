"""Prospective ARGMix-count IID generation and common-support bridge.

Source-only packet. Root owns admission, supervision and native commands.
No numerical import or scientific work occurs at import time. Every producer
requires root's actual guard(required_write_bytes) callback. No GPU work.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import sys

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
HERE = Path(__file__).resolve().parent
CONFIG = HERE / "generator_contract.proposed.v1.json"
CLASSES = ("Ana", "CHG", "WHG", "EHG")
IID_SPLITS = ("train", "val", "iid")


def need(ok, message):
    if not ok:
        raise ValueError(message)


def confined(value):
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    need(path.resolve() == path and ROOT in path.parents, "Canonical repository path required")
    return path


def sha(path):
    result = hashlib.sha256()
    with confined(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def file_ref(path):
    path = confined(path)
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size)


def verify_ref(record):
    path = confined(record["path"])
    need(path.is_file() and not path.is_symlink() and sha(path) == record["sha256"],
         "Pinned input drift: " + str(path))
    return path


def metadata(path):
    path = confined(path)
    need(path.suffix == ".json" and path.stat().st_size <= 8 * 1024 * 1024,
         "Bounded JSON metadata required")
    return json.loads(path.read_text())


class Writes:
    """Finite producer byte ceiling, with root's actual resource callback."""
    def __init__(self, limit, guard):
        need(type(limit) is int and limit > 0 and callable(guard),
             "Root guard and finite positive byte ceiling required")
        self.limit, self.used, self.guard = limit, 0, guard

    def before(self, count):
        need(type(count) is int and count >= 0 and self.used + count <= self.limit,
             "Finite stage output ceiling exceeded. Preserve partial outputs, do not redraw")
        self.guard(count)
        self.used += count

    def after(self):
        self.guard(0)

    def raw(self, path, data):
        self.before(len(data))
        with confined(path).open("xb") as stream:
            stream.write(data)
        self.after()

    def text(self, path, value):
        self.raw(path, value.encode())

    def json(self, path, value):
        self.text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")

    def tree(self, path, tree_sequence):
        # A conservative envelope for fixed-schema kastore plus table data.
        ceiling = 2 * int(tree_sequence.nbytes) + 1024 * 1024
        self.before(ceiling)
        tree_sequence.dump(confined(path))
        need(confined(path).stat().st_size <= ceiling, "Tree serialization exceeded reserved envelope")
        self.after()


def checked_contract(config_path=CONFIG):
    config = metadata(config_path)
    need(config["schema"] == "ARGMix_prospective1611_generator_contract_v1", "Wrong contract")
    for record in config["input_refs"].values():
        verify_ref(record)
    proposal = metadata(config["input_refs"]["proposal"]["path"])
    registry = metadata(config["input_refs"]["registry"]["path"])
    binding = metadata(config["input_refs"]["map_binding"]["path"])
    need(binding["length_bp"] == config["map"]["length_bp"] and
         binding["source_interval_bp"] == config["map"]["source_interval_bp"] and
         binding["rebased_map_sha256"] == config["input_refs"]["map"]["sha256"],
         "Exact authenticated chr22 map binding required")
    need(config["demography"] == proposal["demography"] and config["roster"] == proposal["roster"],
         "Prospective demographic/roster choices differ from reviewed proposal")
    values = [v for row in registry["parents"] for v in row["new_seed_values"].values()]
    values += list(registry["new_fit_seeds"].values())
    need(len(values) == 77 and len(set(values)) == 77, "Do not replace registered 77 streams")
    need(sys.version_info[:2] == (3, 11), "Frozen CPython3.11 sampler contract")
    return config, registry


def parent_record(config_path, parent_id):
    config, registry = checked_contract(config_path)
    matches = [p for p in registry["parents"] if p["parent_id"] == parent_id]
    need(len(matches) == 1 and matches[0]["split"] in IID_SPLITS,
         "Only the frozen five TRAIN, one VAL and three IID parents are implemented")
    return config, matches[0]


def roster(config):
    rows = []
    for cohort in config["roster"]["cohorts"]:
        for diploid in range(cohort["diploids"]):
            older = cohort["id"] == "NF" and diploid == cohort["diploids"] - 1
            age = (185 if older else 180) if cohort["id"] == "NF" else cohort["age_generations"]
            name = f'{cohort["id"]}_{diploid:04d}'
            for hap in (0, 1):
                rows.append(dict(sample_index=len(rows), diploid_index=len(rows) // 2,
                                 sample_id=name, haplotype=hap,
                                 stable_haplotype_id=f"{name}:h{hap}",
                                 observed_cohort=cohort["id"],
                                 sampling_population=cohort["population"],
                                 age_generations=age, reference_eligible=cohort["donor"],
                                 query_eligible=cohort["query"],
                                 observed_reference_class=CLASSES.index(cohort["id"]) if cohort["id"] in CLASSES else None,
                                 synthetic_older_NF_proxy=older, identified_iceman=False))
    expected = config["counts"]
    need(len(rows) == expected["haplotypes"] and len(rows) // 2 == expected["diploids"] and
         sum(r["reference_eligible"] for r in rows) == expected["donor_haplotypes"] and
         sum(r["query_eligible"] for r in rows) == expected["query_haplotypes"] and
         sum(r["observed_reference_class"] is not None for r in rows) == expected["pure_founder_haplotypes"],
         "Declared roster arithmetic differs")
    return rows


def eligible_donor_indices(rows, focal_index):
    """Identity exclusion of BOTH focal-diploid haplotypes, before any ranking."""
    focal = rows[focal_index]
    need(focal["sample_index"] == focal_index and focal["query_eligible"], "Explicit query required")
    return [r["sample_index"] for r in rows if r["reference_eligible"] and
            r["sample_id"] != focal["sample_id"]]


def nearest_donors(tree, rows, inferred_node_by_sample, focal_index, k):
    """Prospective native-distance adapter only, no local ancestry truth read."""
    need(type(k) is int and k > 0, "Positive donor count required")
    focal_node = inferred_node_by_sample[focal_index]
    eligible = eligible_donor_indices(rows, focal_index)
    ranked = []
    for index in eligible:
        value = float(tree.tmrca(focal_node, inferred_node_by_sample[index]))
        need(math.isfinite(value), "Every eligible donor must share an inferred root")
        ranked.append((value, rows[index]["stable_haplotype_id"], index))
    need(len(ranked) >= k, "Insufficient distinct eligible donor haplotypes")
    return [item[2] for item in sorted(ranked)[:k]]


def demographic_model(config, msprime):
    model = msprime.Demography()
    spec = config["demography"]
    for name in sorted(spec["populations_initial_size"]):
        model.add_population(name=name, initial_size=spec["populations_initial_size"][name])
    for event in spec["admixtures"]:
        model.add_admixture(**event)
    for event in spec["splits"]:
        model.add_population_split(**event)
    model.add_census(time=spec["census_time"])
    model.sort_events()
    model.validate()
    return model


def empirical_rate_map(config, msprime, np):
    path = verify_ref(config["input_refs"]["map"])
    positions, cm = [], []
    with path.open() as stream:
        need(next(stream).split() == ["position", "COMBINED_rate(cM/Mb)", "Genetic_Map(cM)"],
             "Frozen map columns required")
        for line in stream:
            p, _, g = line.split()
            positions.append(int(p))
            cm.append(float(g))
    pos = np.asarray(positions, dtype=np.float64)
    genetic = np.asarray(cm, dtype=np.float64)
    need(len(pos) == config["map"]["rows"] and pos[0] == 0 and pos[-1] == config["map"]["length_bp"]
         and genetic[0] == 0 and np.all(np.diff(pos) > 0) and np.all(np.diff(genetic) >= 0),
         "Map support and monotonicity differ")
    need(abs(float(genetic[-1]) - config["map"]["genetic_length_cM"]) < 1e-10,
         "Map genetic length differs")
    rates = np.diff(genetic) / (100.0 * np.diff(pos))
    return msprime.RateMap(position=pos, rate=rates), positions, cm


def _fresh_directory(output, guard):
    output = confined(output)
    need(not output.exists() and output.parent.is_dir(), "Fresh output with existing project parent required")
    guard(16384)
    output.mkdir()
    return output


def generate_parent(config_path, parent_id, output, *, guard):
    """One independent parent. Root must supervise and admit before calling."""
    config, parent = parent_record(config_path, parent_id)
    import msprime
    import numpy as np
    import tskit
    rows = roster(config)
    budget = Writes(config["bounds"]["generation_bytes"], guard)
    output = _fresh_directory(output, guard)
    obs, target = output / "observed", output / "truth"
    obs.mkdir()
    target.mkdir()
    rate_map, _, _ = empirical_rate_map(config, msprime, np)
    model = demographic_model(config, msprime)
    samples = [msprime.SampleSet(1, ploidy=2, population=r["sampling_population"],
                                time=r["age_generations"]) for r in rows[::2]]
    guard(0)
    truth = msprime.sim_ancestry(samples=samples, demography=model, recombination_rate=rate_map,
                                model="hudson", discrete_genome=True, record_migrations=False,
                                record_provenance=False, random_seed=parent["new_seed_values"]["ancestry"])
    need(truth.sequence_length == config["map"]["length_bp"] and
         list(map(int, truth.samples())) == list(range(len(rows))), "Native sample/interval order differs")
    need(truth.nbytes <= config["bounds"]["truth_table_bytes"] and
         truth.num_trees <= config["bounds"]["truth_trees"], "Truth capacity cap exceeded. Preserve and stop")
    for r in rows:
        node = truth.node(r["sample_index"])
        need(node.time == r["age_generations"] and model.populations[node.population].name == r["sampling_population"],
             "Sample age/population order differs")
    for i in range(0, len(rows), 2):
        need(truth.node(i).individual != tskit.NULL and truth.node(i).individual == truth.node(i+1).individual,
             "Native diploid pairing differs")
    # Only the segregated target namespace contains the census topology/populations.
    budget.tree(target / "census_ancestry.trees", truth)
    clean, mapping = truth.simplify(samples=list(truth.samples()), map_nodes=True,
                                   keep_unary=False, filter_populations=False,
                                   filter_individuals=False, record_provenance=False)
    need(not np.any(clean.tables.nodes.flags & msprime.NODE_IS_CEN_EVENT),
         "Census node survived observation cleanup")
    observed = msprime.sim_mutations(clean, rate=config["mutation_rate"],
                                    model=msprime.BinaryMutationModel(state_independent=False),
                                    discrete_genome=True, record_provenance=False,
                                    random_seed=parent["new_seed_values"]["mutation"])
    need(observed.num_sites <= config["bounds"]["mutation_sites"],
         "Mutation-site capacity cap exceeded. Do not shorten, redraw or subsample")
    sample_nodes = [int(mapping[r["sample_index"]]) for r in rows]
    need(sample_nodes == list(map(int, observed.samples())), "Cleaned sample join differs")
    # No true trees or clean oracle trees are exported into observed assets.
    observed_rows = [{k:v for k,v in r.items() if k != "sampling_population"} for r in rows]
    names = [r["sample_id"] for r in rows[::2]]
    budget.json(obs / "sample_manifest.json", observed_rows)
    budget.text(obs / "sample_order.txt", "\n".join(names) + "\n")
    budget.text(obs / "sample_ages.generations", "\n".join(str(r["age_generations"]) for r in rows) + "\n")
    budget.text(obs / "population_labels.txt", "sample population group sex\n" + "".join(
        f'{r["sample_id"]} {r["observed_cohort"]} {r["observed_cohort"]} NA\n' for r in rows[::2]))
    header = '##fileformat=VCFv4.2\n##contig=<ID=22,length=' + str(config["map"]["length_bp"]) + '>\n'
    header += '##FORMAT=<ID=GT,Number=1,Type=String,Description="Phased binary simulated genotype">\n'
    header += '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t' + '\t'.join(names) + '\n'
    rng = np.random.Generator(np.random.PCG64(parent["new_seed_values"]["genotype_error"]))
    counts = dict(mutation_sites=observed.num_sites, exported_sites=0, dropped_invariant_after_error=0,
                  flipped_alleles_all_mutation_sites=0)
    buffers = [bytearray(header.encode()), bytearray(b"observed_site_id\tmutation_site_id\tposition0\tposition1\tmap_source_bp\n"),
               bytearray(), bytearray()]
    paths = [obs / "genotypes.vcf", obs / "variants.tsv", obs / "positions.u4", obs / "genotypes.packbits"]
    handles = [p.open("xb") for p in paths]
    def flush():
        budget.before(sum(len(b) for b in buffers))
        for handle, data in zip(handles, buffers):
            handle.write(data)
            data.clear()
        budget.after()
    try:
        for variant in observed.variants(samples=sample_nodes, isolated_as_missing=False):
            raw = variant.genotypes
            need(np.all(raw >= 0), "No missing latent genotypes allowed")
            alleles = [variant.alleles[int(g)] for g in np.unique(raw)]
            need(set(alleles) <= {"0", "1"}, "Unexpected binary allele")
            state_lut = np.asarray([0 if a is None else int(a) for a in variant.alleles], dtype=np.uint8)
            genotype = state_lut[raw]
            flips = rng.random(len(rows)) < config["error_rate"]
            genotype ^= flips.astype(np.uint8)
            counts["flipped_alleles_all_mutation_sites"] += int(flips.sum())
            if not 0 < int(genotype.sum()) < len(rows):
                counts["dropped_invariant_after_error"] += 1
                continue
            position = int(variant.site.position)
            need(position == variant.site.position and 0 <= position < config["map"]["length_bp"],
                 "Discrete site position outside frozen interval")
            index = counts["exported_sites"]
            # Binary state0->A and state1->C is an explicit format encoding, not a new mutation model.
            # Vectorized ASCII layout avoids billions of per-haplotype Python string conversions.
            ascii_gt = np.empty(2 * len(rows), dtype=np.uint8)
            ascii_gt[0::4], ascii_gt[1::4] = genotype[0::2] + 48, 124
            ascii_gt[2::4], ascii_gt[3::4] = genotype[1::2] + 48, 9
            ascii_gt[-1] = 10
            buffers[0].extend(f'22\t{position+1}\tp{index:07d}\tA\tC\t.\tPASS\t.\tGT\t'.encode())
            buffers[0].extend(ascii_gt.tobytes())
            buffers[1].extend(f'{index}\t{variant.site.id}\t{position}\t{position+1}\t{position+config["map"]["source_interval_bp"][0]}\n'.encode())
            buffers[2].extend(struct.pack('<I', position))
            buffers[3].extend(np.packbits(genotype, bitorder='little').tobytes())
            counts["exported_sites"] += 1
            if sum(len(b) for b in buffers) >= config["bounds"]["io_batch_bytes"]:
                flush()
        flush()
    finally:
        for handle in handles:
            handle.close()
    need(counts["exported_sites"] > 1, "Insufficient observed segregating sites. Stop, no redraw")
    exports = {p.name:file_ref(p) for p in obs.iterdir() if p.is_file()}
    budget.json(obs / "observation_manifest.json", dict(
        schema="ARGMix_prospective1611_observations_v1", parent_id=parent_id, split=parent["split"],
        title="Prospective synthetic chr22 common-data panel", config_ref=file_ref(config_path),
        source_ref=file_ref(__file__), counts=config["counts"], sites=counts["exported_sites"], filters=counts,
        length_bp=config["map"]["length_bp"], coordinate="VCF position = rebased tree position + 1",
        source_map_ref=config["input_refs"]["map"], observed_files=exports,
        genotype_error=dict(rate=config["error_rate"], applied_once=True, rng="numpy.Generator(PCG64)",
                            seed=parent["new_seed_values"]["genotype_error"],
                            order="all mutation sites in tskit order, all roster haplotypes per site"),
        genotypes=dict(layout="site-major bit-packed uint8, bitorder=little, zero pad to byte", shape=[counts["exported_sites"],len(rows)],
                       bytes_per_site=(len(rows)+7)//8, positions="little-endian uint32 positions.u4", allele_encoding={"0":"A","1":"C"}),
        no_oracle_tree_exported=True, census_or_target_features_exported=False,
        original_paper_reproduction=False))
    budget.json(target / "truth_manifest.json", dict(
        schema="ARGMix_prospective1611_census_truth_v1", parent_id=parent_id, split=parent["split"],
        config_ref=file_ref(config_path), census_time=config["demography"]["census_time"],
        ancestry_file=file_ref(target / "census_ancestry.trees"), class_order=list(CLASSES),
        population_id_to_name={str(i):p.name for i,p in enumerate(model.populations)},
        sample_index_to_raw_node_id=list(map(int, truth.samples())),
        truth_boundaries_forbidden_as_observed_model_inputs=True))
    receipt = dict(status="prospective_observations_and_segregated_census_truth_complete", parent_id=parent_id,
                   split=parent["split"], config_ref=file_ref(config_path), source_ref=file_ref(__file__),
                   observation_manifest_ref=file_ref(obs / "observation_manifest.json"),
                   truth_manifest_ref=file_ref(target / "truth_manifest.json"), counts=counts,
                   true_trees=truth.num_trees, true_table_bytes=int(truth.nbytes),
                   runtime=dict(python=sys.version.split()[0], numpy=np.__version__, tskit=tskit.__version__, msprime=msprime.__version__),
                   Relate_executed=False, model_fit_performed=False)
    budget.json(output / "generation.json", receipt)
    return receipt


def observed_assets(config_path, parent_id, parent_output):
    config, parent = parent_record(config_path, parent_id)
    folder = confined(parent_output) / "observed"
    manifest = metadata(folder / "observation_manifest.json")
    need(manifest["schema"] == "ARGMix_prospective1611_observations_v1" and
         manifest["parent_id"] == parent_id and manifest["split"] == parent["split"] and
         manifest["config_ref"] == file_ref(config_path), "Observed provenance differs")
    for record in manifest["observed_files"].values():
        path = verify_ref(record)
        need(path.parent == folder, "Only observed whitelist assets may be opened")
    original = roster(config)
    rows = metadata(folder / "sample_manifest.json")
    need(rows == [{k:v for k,v in r.items() if k != "sampling_population"} for r in original],
         "Observed metadata roster differs")
    return config, parent, folder, manifest, rows


def prepare_relate(config_path, parent_id, parent_output, output, *, guard):
    """Write native format inputs and exact argv. No subprocess is launched."""
    config, parent, obs, observation, rows = observed_assets(config_path, parent_id, parent_output)
    import numpy as np
    budget = Writes(config["bounds"]["relate_input_bytes"], guard)
    output = _fresh_directory(output, guard)
    names = [r["sample_id"] for r in rows[::2]]
    budget.text(output / "input.sample", "ID_1 ID_2 missing\n0 0 0\n" +
                "".join(f'{name} {name} 0\n' for name in names))
    budget.text(output / "sample_ages.txt", (obs / "sample_ages.generations").read_text())
    budget.text(output / "input.poplabels", (obs / "population_labels.txt").read_text())
    _, positions, cm = empirical_rate_map(config, NoneMap(), np)
    lines = ["pos COMBINED_rate Genetic_Map"]
    for i, (position, genetic) in enumerate(zip(positions, cm)):
        rate = 0 if i == len(positions)-1 else 1e6*(cm[i+1]-genetic)/(positions[i+1]-position)
        lines.append(f'{position+1} {rate:.17g} {genetic:.17g}')
    budget.text(output / "input.map", "\n".join(lines) + "\n")
    count, width = observation["sites"], (len(rows)+7)//8
    need((obs / "positions.u4").stat().st_size == 4*count and
         (obs / "genotypes.packbits").stat().st_size == width*count, "Packed genotype shape differs")
    buffer, previous = bytearray(), -1
    with (obs / "positions.u4").open("rb") as posfile, (obs / "genotypes.packbits").open("rb") as genfile, (output / "input.haps").open("xb") as handle:
        for index in range(count):
            position = struct.unpack('<I', posfile.read(4))[0]
            genotype = np.unpackbits(np.frombuffer(genfile.read(width), dtype=np.uint8), bitorder='little')[:len(rows)]
            need(previous < position < config["map"]["length_bp"] and 0 < int(genotype.sum()) < len(rows),
                 "Observed position/order or segregating state differs")
            previous = position
            ascii_haps = np.empty(2 * len(rows), dtype=np.uint8)
            ascii_haps[0::2], ascii_haps[1::2] = genotype + 48, 32
            ascii_haps[-1] = 10
            buffer.extend(f'22 p{index:07d} {position+1} A C '.encode())
            buffer.extend(ascii_haps.tobytes())
            if len(buffer) >= config["bounds"]["io_batch_bytes"]:
                budget.before(len(buffer)); handle.write(buffer); buffer.clear(); budget.after()
        budget.before(len(buffer)); handle.write(buffer); budget.after()
    installation = metadata(config["input_refs"]["relate_installation"]["path"])
    need(installation["version"] == "1.2.4" and installation["authorization"]["human_message"] == "I reviewed, use it",
         "Pinned authorized Relate installation required")
    binaries = {key:verify_ref(value) for key,value in installation["binaries"].items()}
    inferred = output / "inferred"
    commands = [[str(binaries["Relate"]), "--mode", "All", "-m", str(config["mutation_rate"]),
                 "-N", "30000", "--memory", str(config["relate_memory_gb"]),
                 "--seed", str(parent["new_seed_values"]["relate_initial"]),
                 "--haps", str(output / "input.haps"), "--sample", str(output / "input.sample"),
                 "--sample_ages", str(output / "sample_ages.txt"), "--map", str(output / "input.map"),
                 "-o", "inferred"],
                [str(binaries["RelateFileFormats"]), "--mode", "ConvertToTreeSequence", "-i", "inferred", "-o", "inferred"]]
    receipt = dict(schema="ARGMix_prospective1611_relate_commands_v1", status="inputs_ready_native_execution_pending",
                   parent_id=parent_id, config_ref=file_ref(config_path), source_ref=file_ref(__file__),
                   observation_manifest_ref=file_ref(obs / "observation_manifest.json"),
                   input_refs={p.name:file_ref(p) for p in output.iterdir() if p.is_file()},
                   cwd=str(output), argv=commands, native_trees=str(inferred)+".trees",
                   truth_or_true_tree_read=False, original_paper_reproduction=False,
                   explicit_postprocessing="none", population_size_estimation_iterations=0,
                   mutation_branch_estimation="Relate1.2.4 --mode All native defaults, no additional external re-estimation",
                   native_stage_total_output_ceiling_bytes=config["bounds"]["native_relate_bytes"],
                   root_supervisor_timeout_seconds=config["bounds"]["native_timeout_seconds"])
    budget.json(output / "commands.json", receipt)
    return receipt


class NoneMap:
    """Map-parser reuse during input export, without importing msprime."""
    @staticmethod
    def RateMap(**kwargs):
        return None


def finalize_relate(config_path, parent_id, parent_output, native_work, terminal_receipt, output, *, guard):
    """Consume root-confirmed native success, clip only to observed SNP support."""
    config, parent, obs, observation, rows = observed_assets(config_path, parent_id, parent_output)
    import numpy as np
    import tskit
    native_work = confined(native_work)
    commands = metadata(native_work / "commands.json")
    terminal = metadata(terminal_receipt)
    need(commands["parent_id"] == parent_id and commands["config_ref"] == file_ref(config_path), "Native command provenance differs")
    need(terminal["commands_ref"] == file_ref(native_work / "commands.json") and
         terminal["actual_wait_consumed"] is True and terminal["returncodes"] == [0,0],
         "Root must supply actual successful waits for both unchanged native argv")
    source = confined(commands["native_trees"])
    need(source == native_work / "inferred.trees" and source.stat().st_size <= config["bounds"]["native_relate_bytes"],
         "Native output path/cap differs")
    inferred = tskit.load(source)
    sample_ids = list(map(int, inferred.samples()))
    need(len(sample_ids) == len(rows), "Native sample count differs")
    need(np.allclose(inferred.tables.nodes.time[sample_ids], [r["age_generations"] for r in rows], atol=1e-9, rtol=0),
         "Native sample age/order differs")
    # The root executed commands use direct .haps columns and explicit .sample IDs, so no converter reordering is inferred.
    positions = np.fromfile(obs / "positions.u4", dtype='<u4')
    native_sites = inferred.tables.sites.position
    need(len(native_sites) > 1 and np.array_equal(native_sites, native_sites.astype(np.int64)) and
         np.array_equal(native_sites.astype(np.int64), positions.astype(np.int64)+1),
         "Native SNP subset/drop changed common support. Review explicitly before proceeding")
    left, right = float(positions[0]), float(positions[-1]+1)
    need(0 <= left < right <= config["map"]["length_bp"], "Invalid inferred original support")
    tables = inferred.dump_tables()
    edges = tables.edges
    lo, hi = np.maximum(edges.left-1, left), np.minimum(edges.right-1, right)
    keep = hi > lo
    par, child = edges.parent[keep], edges.child[keep]
    tables.edges.clear()
    tables.edges.set_columns(left=lo[keep]-left, right=hi[keep]-left, parent=par, child=child)
    tables.sites.position = native_sites-1-left
    tables.sequence_length = right-left
    tables.nodes.population = np.full(inferred.num_nodes, tskit.NULL, dtype=np.int32)
    tables.nodes.individual = np.full(inferred.num_nodes, tskit.NULL, dtype=np.int32)
    tables.nodes.flags = tables.nodes.flags & tskit.NODE_IS_SAMPLE
    for table in (tables.nodes, tables.edges, tables.sites, tables.mutations):
        table.drop_metadata()
    tables.migrations.clear(); tables.populations.clear(); tables.individuals.clear(); tables.provenances.clear()
    tables.metadata_schema = tskit.MetadataSchema(None)
    tables.metadata = b""
    tables.sort()
    clean = tables.tree_sequence()
    need(list(map(int, clean.samples())) == sample_ids and clean.num_nodes == inferred.num_nodes,
         "Clipping changed inferred identities")
    need(all(tree.num_roots == 1 for tree in clean.trees()), "Uncovered/disconnected inferred interval")
    budget = Writes(config["bounds"]["inferred_bridge_bytes"], guard)
    output = _fresh_directory(output, guard)
    budget.tree(output / "observed.trees", clean)
    joined = [dict(r, inferred_tree_node_id=node, observed_node_id=node) for r,node in zip(rows,sample_ids)]
    budget.json(output / "sample_manifest.json", joined)
    receipt = dict(schema="ARGMix_prospective1611_inferred_observation_v1", parent_id=parent_id, split=parent["split"],
                   config_ref=file_ref(config_path), source_ref=file_ref(__file__),
                   observed_manifest_ref=file_ref(obs / "observation_manifest.json"),
                   native_commands_ref=file_ref(native_work / "commands.json"), native_terminal_ref=file_ref(terminal_receipt),
                   native_tree_ref=file_ref(source), tree_ref=file_ref(output / "observed.trees"),
                   roster_ref=file_ref(output / "sample_manifest.json"), trees=clean.num_trees, sites=clean.num_sites,
                   observed_coverage_bp0=[left,right], local_to_rebased_offset_bp=left,
                   chromosome_map_length_bp=config["map"]["length_bp"],
                   no_tree_extrapolation=True, original_observed_genotypes_authoritative=True,
                   converted_mutations_forbidden_as_genotype_features=True, truth_read=False)
    budget.json(output / "completed.json", receipt)
    return receipt


def sample_targets(config_path, parent_id, inferred_output, output, *, guard):
    """Freeze exact shared rows from observations only, before any truth join."""
    config, parent = parent_record(config_path, parent_id)
    import tskit
    inferred_output = confined(inferred_output)
    bridge = metadata(inferred_output / "completed.json")
    need(bridge["parent_id"] == parent_id and bridge["config_ref"] == file_ref(config_path), "Inferred provenance differs")
    tree_file, roster_file = verify_ref(bridge["tree_ref"]), verify_ref(bridge["roster_ref"])
    trees = tskit.load(tree_file)
    rows = metadata(roster_file)
    expected = roster(config)
    need(len(rows) == len(expected) and
         all(all(k in r and r[k] == v for k,v in e.items() if k != "sampling_population") for r,e in zip(rows,expected)) and
         [r["inferred_tree_node_id"] for r in rows] == list(map(int, trees.samples())),
         "Inferred roster or sample-node order differs")
    queries = [r for r in rows if r["query_eligible"]]
    step = {"train":10,"val":100,"iid":1}[parent["split"]]
    intervals = [(t.index,t.interval.left,t.interval.right) for t in trees.trees() if t.index % step == 0 and t.span > 0]
    need(len(intervals)*len(queries) >= parent["planned_examples"], "Fewer than frozen100000 distinct support pairs")
    choices = random.Random(parent["new_seed_values"]["sampler"]).sample(range(len(intervals)*len(queries)),parent["planned_examples"])
    cohort_counts = Counter()
    budget = Writes(config["bounds"]["target_bytes"], guard)
    output = _fresh_directory(output, guard)
    lines = ["target_row\tinferred_tree_index\tsample_index\tsample_id\thaplotype\tcohort\tlocal_midpoint_bp\trebased_midpoint_bp\n"]
    for index,flat in enumerate(choices):
        tree_slot, query_slot = divmod(flat,len(queries))
        tree_index,left,right = intervals[tree_slot]
        q = queries[query_slot]
        local = (left+right)/2
        physical = local+bridge["local_to_rebased_offset_bp"]
        cohort_counts[q["observed_cohort"]] += 1
        lines.append(f'{index}\t{tree_index}\t{q["sample_index"]}\t{q["sample_id"]}\t{q["haplotype"]}\t{q["observed_cohort"]}\t{local:.17g}\t{physical:.17g}\n')
    need(set(cohort_counts) == {c["id"] for c in config["roster"]["cohorts"] if c["query"]}, "A required query cohort has zero targets. Stop, do not resample")
    budget.text(output / "shared_targets.tsv", "".join(lines))
    receipt = dict(schema="ARGMix_prospective1611_shared_support_v1", parent_id=parent_id, split=parent["split"],
                   config_ref=file_ref(config_path), inferred_bridge_ref=file_ref(inferred_output / "completed.json"),
                   targets_ref=file_ref(output / "shared_targets.tsv"), examples=len(choices), cohort_counts=dict(cohort_counts),
                   support_pairs=len(intervals)*len(queries), tree_step=step, includes_edge_trees=True,
                   sampler="CPython3.11 random.Random(seed).sample(range(N),100000)", seed=parent["new_seed_values"]["sampler"],
                   row_order="unmodified sampler return order", truth_or_labels_read=False,
                   target_coordinate="inferred half-open interval midpoint in rebased physical bp")
    budget.json(output / "completed.json", receipt)
    return receipt


def label_targets(config_path, parent_id, parent_output, support_output, output, *, guard):
    """Segregated scoring/supervision join. Never call from observed feature code."""
    config, parent = parent_record(config_path,parent_id)
    import msprime
    import tskit
    truth_dir = confined(parent_output) / "truth"
    manifest = metadata(truth_dir / "truth_manifest.json")
    need(manifest["parent_id"] == parent_id and manifest["config_ref"] == file_ref(config_path), "Truth parent provenance differs")
    truth = tskit.load(verify_ref(manifest["ancestry_file"]))
    support = metadata(confined(support_output) / "completed.json")
    need(support["parent_id"] == parent_id and support["config_ref"] == file_ref(config_path) and support["truth_or_labels_read"] is False,
         "Exact observed-only frozen support required")
    target_rows = []
    with verify_ref(support["targets_ref"]).open() as stream:
        names = next(stream).rstrip().split('\t')
        for line in stream:
            target_rows.append(dict(zip(names,line.rstrip().split('\t'))))
    expected = roster(config)
    need(len(target_rows) == parent["planned_examples"], "Frozen target count differs")
    labels = [None]*len(target_rows)
    tree = truth.first()
    for index in sorted(range(len(target_rows)),key=lambda i:float(target_rows[i]["rebased_midpoint_bp"])):
        row = target_rows[index]
        need(int(row["target_row"]) == index, "Shared target row order differs")
        sample = expected[int(row["sample_index"])]
        need(sample["query_eligible"] and [row["sample_id"],int(row["haplotype"]),row["cohort"]] ==
             [sample["sample_id"],sample["haplotype"],sample["observed_cohort"]], "Target identity join differs")
        position = float(row["rebased_midpoint_bp"])
        need(0 <= position < truth.sequence_length,"Target coordinate outside full truth support")
        tree.seek(position)  # tskit half-open intervals give the declared right-continuous lookup.
        node = manifest["sample_index_to_raw_node_id"][sample["sample_index"]]
        while node != tskit.NULL and not (truth.node(node).flags & msprime.NODE_IS_CEN_EVENT):
            node = tree.parent(node)
        need(node != tskit.NULL and truth.node(node).time == manifest["census_time"],"Missing census350 ancestor")
        population = manifest["population_id_to_name"][str(truth.node(node).population)]
        need(population in CLASSES,"Census lineage outside four founder classes")
        labels[index] = CLASSES.index(population)
    budget = Writes(config["bounds"]["target_bytes"],guard)
    output = _fresh_directory(output,guard)
    budget.raw(output / "classes.u1",bytes(labels))
    receipt = dict(schema="ARGMix_prospective1611_targets_with_external_truth_v1",parent_id=parent_id,split=parent["split"],
                   config_ref=file_ref(config_path),support_manifest_ref=file_ref(confined(support_output)/"completed.json"),
                   truth_manifest_ref=file_ref(truth_dir/"truth_manifest.json"),labels_ref=file_ref(output/"classes.u1"),
                   class_order=list(CLASSES),examples=len(labels),row_order="identical to frozen shared_targets.tsv",
                   ancestry_definition="first census350 ancestor, four founder lineage",
                   features_exported=False,training_labels_permitted_only_for_split="train")
    budget.json(output/"completed.json",receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe",action="store_true")
    args=parser.parse_args()
    need(args.describe,"Root must call producer API with its actual guard callback")
    print(json.dumps(dict(status="source_only_no_scientific_execution",contract=str(CONFIG),
                         stages=["generate_parent","prepare_relate","root executes native argv",
                                 "finalize_relate","sample_targets","label_targets"],
                         execution_authority_created=False),indent=2))


if __name__ == "__main__":
    main()
