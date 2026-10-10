#!/usr/bin/env python
"""Source-only by default; root-admitted observed CPU tsinfer+tsdate transfer."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
PARENTS = tuple(f"bio_paths_pilot_{split}_{i:03d}" for split in ("train", "val") for i in range(2))
FILES = ("genotypes.vcf", "sample_manifest.json", "sample_order.txt",
         "sample_ages.generations", "population_labels.txt",
         "site_coordinate_manifest.json", "uniform_recombination_map.txt",
         "scoring_coordinates.npz", "observation_manifest.json")
ROLES = {"pure_source_reference": 96, "mixed_reference_only": 64, "query": 120}
CONDITION = "official_documented_heterochronous_tsinfer_tsdate_task_transfer_v1"
VERSIONS = {"numpy": "2.4.6", "tskit": "1.0.3", "tsinfer": "0.5.1", "tsdate": "0.2.7"}


def confined(value):
    lexical = Path(value)
    lexical = ROOT / lexical if not lexical.is_absolute() else lexical
    p = lexical.resolve()
    if ROOT not in p.parents:
        raise ValueError("path must remain strictly inside project")
    return p


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ref(path):
    return {"path": str(Path(path).relative_to(ROOT)), "sha256": sha(path)}


def checked_ref(value):
    if set(value) != {"path", "sha256"} or len(value["sha256"]) != 64:
        raise ValueError("exact path/SHA256 reference required")
    lexical = Path(value["path"])
    lexical = ROOT / lexical if not lexical.is_absolute() else lexical
    if lexical.is_symlink():
        raise ValueError("artifact reference cannot be a symlink")
    p = confined(lexical)
    if not p.is_file() or sha(p) != value["sha256"]:
        raise ValueError("artifact bytes changed: " + str(p))
    return p


def write_json(path, value, *, new=False):
    with Path(path).open("x" if new else "w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def validate_config(c):
    fixed = {
        "kind": "biological_paths_tsinfer_v1", "condition": CONDITION,
        "source_root": "data/biological_paths_execution_pilot_v1",
        "output_root": "data/biological_paths_execution_pilot_v1/tsinfer_tsdate_observed_v1",
        "qualification_root": "research/reviews/biological_paths_tsinfer_v1_source_audit/root_cpu_qualification_v1",
        "runtime_prefix": ".runtime/envs/biological_tsinfer_v1",
        "runtime_versions": VERSIONS, "python_minor": "3.11",
        "parents": list(PARENTS), "panel": ROLES,
        "length_bp": 2000000, "score_bin_bp": 100, "mutation_rate": 1.25e-8,
        "class_order": ["ANA", "CHG", "WHG", "EHG"],
        "num_threads": 0, "recombination_rate": None, "mismatch_ratio": None,
        "path_compression": True, "post_process": True,
        "preprocess": {"delete_intervals": [], "split_disjoint": True,
                       "filter_populations": False, "filter_individuals": False,
                       "filter_sites": False},
        "dating": {"method": "variational_gamma", "max_iterations": 25,
                   "rescaling_intervals": 1000, "rescaling_iterations": 5,
                   "match_segregating_sites": False, "singletons_phased": True,
                   "constr_iterations": None, "min_branch_length": 1e-8,
                   "allow_unary": False, "set_metadata": True,
                   "time_units": "generations"},
        "site_age": {"unconstrained": True, "node_selection": "child", "min_time": 1},
        "proxy": {"allow_mutation": True, "epsilon": None},
        "force_sample_times": True, "minimum_connected_internal_reuse": 1,
        "storage_warning_bytes": 450000000000, "storage_limit_bytes": 500000000000,
        "max_added_bytes": 20000000000,
    }
    for key, value in fixed.items():
        if c.get(key) != value:
            raise ValueError("fixed transfer contract changed: " + key)
    if c.get("polarization") != "REF_is_ancestral_only_under_pinned_tskit_writer_defaults":
        raise ValueError("unsupported ancestral-allele assumption")
    for item in c["source_evidence"].values():
        checked_ref(item)
    checked_ref(c["observed_asset_pins"])
    return c


def source_only(c, config_path):
    # Source bytes and an observation-only metadata lock, never VCF/NPZ/tree arrays.
    validate_config(c)
    return {"kind": "biological_paths_tsinfer_v1_source_check",
            "status": "source_only_unexecuted_not_paper_reproduction",
            "condition": CONDITION, "config": ref(config_path), "implementation": ref(Path(__file__)),
            "runtime_imported": False, "observed_arrays_opened": False,
            "inference_dating_executed": False,
            "capability_warning": "Official historical tsinfer+variational_gamma workflow is not extensively tested; CPU qualification and actual identity/genotype/date checks are pending."}


def validate_roster(rows, names, ages, *, roles=ROLES):
    if len(rows) != 2 * len(names) or len(ages) != len(rows) or len(set(names)) != len(names):
        raise ValueError("diploid/haplotype counts differ")
    if any(not math.isfinite(x) or x < 0 for x in ages):
        raise ValueError("invalid observed sample age")
    counts = {key: 0 for key in roles}
    for i, row in enumerate(rows):
        if (row["sample_index"] != i or row["sample_id"] != names[i // 2]
                or row["haplotype"] != i % 2 or row["age_generations"] != ages[i]):
            raise ValueError("sample/date/haplotype order differs")
        role = row["role"]
        if role not in counts:
            raise ValueError("unknown observed role")
        counts[role] += 1
        label = row["observed_reference_class"]
        if (role == "pure_source_reference" and (type(label) is not int or label not in range(4))
                or role != "pure_source_reference" and label is not None):
            raise ValueError("query/mixed labels or pure-reference class invalid")
        if i % 2 and (ages[i] != ages[i-1] or row["observed_cohort"] != rows[i-1]["observed_cohort"]
                      or role != rows[i-1]["role"] or label != rows[i-1]["observed_reference_class"]):
            raise ValueError("diploid pair metadata differs")
    if counts != roles:
        raise ValueError("fixed observed panel roles differ")


def parse_vcf(lines, names, coordinate_rows, length, *, expected_writer_source=None):
    """Retain every original site. Invalid/missing phase aborts, never filters."""
    join = {}
    for i, row in enumerate(coordinate_rows):
        pos = row["VCF_position_1based"]
        if (type(pos) is not int or row["site_id"] != i or pos != row["tree_position_0based"] + 1
                or not 1 <= pos <= length or pos in join):
            raise ValueError("original coordinate/site join invalid")
        join[pos] = row
    sites, header, previous, writer_sources = [], False, 0, []
    for line in lines:
        if line.startswith("##"):
            if line.startswith("##source="):
                writer_sources.append(line.rstrip("\n").removeprefix("##source="))
            continue
        if line.startswith("#CHROM"):
            fields = line.rstrip("\n").split("\t")
            if header or fields[:9] != ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"] or fields[9:] != names:
                raise ValueError("VCF header/order differs")
            header = True
            continue
        if line.startswith("#") or not header:
            raise ValueError("VCF missing/unsupported header")
        f = line.rstrip("\n").split("\t")
        if len(f) != 9 + len(names) or f[0] != "22":
            raise ValueError("VCF columns/contig differ")
        pos = int(f[1])
        if pos <= previous or pos not in join or f[2] != str(join[pos]["site_id"]):
            raise ValueError("VCF position/site identity differs")
        previous = pos
        alleles = [f[3]] + ([] if f[4] == "." else f[4].split(","))
        if len(set(alleles)) != len(alleles) or any(a not in ("A", "C", "G", "T") for a in alleles):
            raise ValueError("unsupported original nucleotide alleles")
        fmt = f[8].split(":")
        if fmt.count("GT") != 1:
            raise ValueError("VCF requires exactly one GT")
        gt_index = fmt.index("GT")
        g = []
        for cell in f[9:]:
            parts = cell.split(":")
            if len(parts) <= gt_index:
                raise ValueError("missing GT")
            pair = parts[gt_index].split("|")
            if len(pair) != 2 or any(not x.isdecimal() for x in pair):
                raise ValueError("unphased/missing genotype")
            values = [int(x) for x in pair]
            if any(x >= len(alleles) for x in values):
                raise ValueError("genotype allele index out of range")
            g.extend(values)
        sites.append({"site_id": join[pos]["site_id"], "position": pos-1,
                      "vcf_position_1based": pos, "alleles": alleles, "genotypes": g,
                      "ancestral_allele": 0})
    if expected_writer_source is not None and writer_sources != [expected_writer_source]:
        raise ValueError("pinned VCF writer/version header differs")
    if not sites or len(sites) != len(join) or {x["vcf_position_1based"] for x in sites} != set(join):
        raise ValueError("VCF must preserve all coordinate-manifest sites")
    return sites


def grid_indexes(left, right, length=2000000, bin_bp=100):
    if not (math.isfinite(left) and math.isfinite(right) and 0 <= left < right <= length):
        raise ValueError("invalid original physical coverage")
    return [i for i in range(length // bin_bp) if left <= bin_bp/2 + bin_bp*i < right]


def load_assets(c, parent, pin):
    import numpy as np
    folder = confined(c["source_root"]) / parent / "observed"
    if set(pin["observed_files_sha256"]) != set(FILES):
        raise ValueError("observed input lock must be exact whitelist")
    paths = {}
    for name in FILES:
        p = folder / name
        if p.resolve() != p.absolute() or not p.is_file() or sha(p) != pin["observed_files_sha256"][name]:
            raise ValueError("original observed asset drift: " + name)
        paths[name] = p
    rows = json.loads(paths["sample_manifest.json"].read_text())
    names = paths["sample_order.txt"].read_text().splitlines()
    ages = [float(x) for x in paths["sample_ages.generations"].read_text().splitlines()]
    validate_roster(rows, names, ages)
    if len(names) != 140:
        raise ValueError("fixed 140-diploid panel differs")
    labels = [x.split() for x in paths["population_labels.txt"].read_text().splitlines()]
    expected = [["sample", "population", "group", "sex"]] + [
        [names[i], rows[2*i]["observed_cohort"], rows[2*i]["observed_cohort"], "NA"] for i in range(len(names))]
    if labels != expected:
        raise ValueError("observed cohort roster differs")
    m = json.loads(paths["observation_manifest.json"].read_text())
    for key, value in {"parent_id": parent, "samples": 280, "diploids": 140,
                       "query_haplotypes": 120, "pure_reference_haplotypes": 96,
                       "scoring_bin_bp": 100}.items():
        if m.get(key) != value:
            raise ValueError("observed manifest differs: " + key)
    coords = json.loads(paths["site_coordinate_manifest.json"].read_text())
    if coords["build"] != "synthetic" or coords["contig_id"] != "22" or coords["length_bp"] != c["length_bp"]:
        raise ValueError("synthetic coordinate contract differs")
    if paths["uniform_recombination_map.txt"].read_text().split() != ["Position(bp)", "Rate(cM/Mb)", "Map(cM)", "1", "1", "0", "2000001", "1", "2"]:
        raise ValueError("original uniform map differs")
    with np.load(paths["scoring_coordinates.npz"], allow_pickle=False) as grid:
        if set(grid.files) != {"bin_edges_bp", "positions_bp"} or not np.array_equal(grid["positions_bp"], np.arange(50, 2000000, 100)) or not np.array_equal(grid["bin_edges_bp"], np.arange(0, 2000001, 100)):
            raise ValueError("original fixed scoring centers/edges differ")
    with paths["genotypes.vcf"].open() as stream:
        sites = parse_vcf(stream, names, coords["rows"], c["length_bp"], expected_writer_source="tskit 1.0.3")
    left, right = pin["comparison_coverage_original_bp0"]
    expected_grid = grid_indexes(left, right)
    if expected_grid != pin["comparison_retained_fixed100bp_indexes"]:
        raise ValueError("comparison coverage/grid metadata differs")
    if not sites[0]["position"] <= left < right <= sites[-1]["position"] + 1:
        raise ValueError("common comparison support exceeds observed SNP support")
    return {"rows": rows, "names": names, "ages": ages, "sites": sites, "paths": paths, "pin": pin,
            "identity_support_original_bp0": [left, right]}


def runtime_inventory(prefix):
    """Complete non-pyc runtime inventory; no imports of inference packages."""
    prefix = confined(prefix)
    files = {}
    for directory, dirs, names in os.walk(prefix, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for name in sorted(names):
            p = Path(directory) / name
            if p.suffix == ".pyc":
                continue
            if p.is_symlink() and ROOT not in p.resolve().parents:
                raise ValueError("runtime symlink escapes project")
            if p.is_file():
                files[str(p.relative_to(prefix))] = sha(p)
    return files


def project_bytes():
    total = 0
    for directory, dirs, names in os.walk(ROOT, followlinks=False):
        dirs[:] = [d for d in dirs if not (Path(directory)/d).is_symlink()]
        for name in names:
            p = Path(directory)/name
            if not p.is_symlink():
                total += p.stat().st_size
    return total


def storage_check(c, initial=None):
    used = project_bytes()
    if used >= c["storage_warning_bytes"] or used + c["max_added_bytes"] >= c["storage_limit_bytes"]:
        raise ValueError("project storage warning/reservation exceeded")
    if initial is not None and used - initial > c["max_added_bytes"]:
        raise ValueError("run storage reservation exceeded")
    return used


def admission(c, args, config_path):
    if not args.expected_config_sha256 or not args.expected_source_sha256 or not args.admission or not args.expected_admission_sha256:
        raise ValueError("actual CPU modes require root config/source/admission SHA256 opt-in")
    if sha(config_path) != args.expected_config_sha256 or sha(Path(__file__)) != args.expected_source_sha256:
        raise ValueError("root-reviewed implementation/config bytes differ")
    p = checked_ref({"path": args.admission, "sha256": args.expected_admission_sha256})
    a = json.loads(p.read_text())
    if (a.get("kind") != "biological_paths_tsinfer_v1_root_admission"
            or a.get("authorizer") != "root_agent" or a.get("mode") != args.mode
            or a.get("condition") != CONDITION or a.get("truth_or_oracle_inputs_authorized") is not False):
        raise ValueError("root CPU admission scope differs")
    if a.get("config") != ref(config_path) or a.get("implementation") != ref(Path(__file__)) or a.get("observed_asset_pins") != c["observed_asset_pins"]:
        raise ValueError("admission does not bind frozen config/source/input metadata lock")
    runtime_path = checked_ref(a["runtime"])
    storage_path = checked_ref(a["storage"])
    runtime = json.loads(runtime_path.read_text())
    if (runtime.get("kind") != "biological_paths_tsinfer_v1_isolated_runtime"
            or runtime.get("prefix") != c["runtime_prefix"]
            or Path(sys.prefix).resolve() != confined(c["runtime_prefix"])
            or f"{sys.version_info.major}.{sys.version_info.minor}" != c["python_minor"]
            or sha(sys.executable) != runtime.get("python_executable_sha256")
            or runtime.get("versions") != VERSIONS):
        raise ValueError("isolated runtime interpreter/versions differ")
    if runtime_inventory(c["runtime_prefix"]) != runtime.get("files_sha256"):
        raise ValueError("complete frozen runtime inventory changed")
    for name, expected in VERSIONS.items():
        if importlib.metadata.version(name) != expected:
            raise ValueError("installed distribution version differs: " + name)
    storage = json.loads(storage_path.read_text())
    if (storage.get("kind") != "biological_paths_tsinfer_v1_storage_admission"
            or storage.get("project_root") != str(ROOT)
            or storage.get("max_added_bytes") != c["max_added_bytes"]
            or storage.get("limit_bytes") != c["storage_limit_bytes"]
            or type(storage.get("measured_project_bytes")) is not int
            or storage["measured_project_bytes"] >= c["storage_warning_bytes"]):
        raise ValueError("storage admission differs")
    initial = storage_check(c)
    if args.mode == "infer":
        q = json.loads(checked_ref(a["cpu_qualification"]).read_text())
        if (q.get("kind") != "biological_paths_tsinfer_v1_cpu_qualification"
                or q.get("status") != "passed"
                or q.get("config") != a["config"] or q.get("implementation") != a["implementation"]
                or q.get("runtime") != a["runtime"]):
            raise ValueError("same-code/config/runtime handmade CPU qualification required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU mode must explicitly hide CUDA devices")
    return a, p, initial


def check_samples(ts, ages, *, diploid=True):
    import numpy as np
    samples = ts.samples().tolist()
    if len(samples) != len(ages) or not np.array_equal(ts.tables.nodes.time[samples], np.asarray(ages)):
        raise ValueError("actual flagged sample order/age differs")
    if diploid:
        for i in range(0, len(samples), 2):
            a, b = ts.node(samples[i]).individual, ts.node(samples[i+1]).individual
            if a < 0 or a != b or set(ts.individual(a).nodes) != {samples[i], samples[i+1]}:
                raise ValueError("actual sample diploid pairing differs")
    return samples


def check_genotypes(ts, sites, indexes=None):
    indexes = list(range(ts.num_samples)) if indexes is None else indexes
    actual = ts.variants(samples=ts.samples(), isolated_as_missing=True)
    if ts.num_sites != len(sites):
        raise ValueError("native inference/site count changed")
    for var, site in zip(actual, sites):
        if var.site.position != site["position"] or var.site.ancestral_state != site["alleles"][0]:
            raise ValueError("native original position/ancestral allele differs")
        decoded = []
        for g in var.genotypes:
            if g < 0:
                raise ValueError("native missing genotype")
            decoded.append(var.alleles[int(g)])
        expected = [site["alleles"][site["genotypes"][i]] for i in indexes]
        if decoded != expected:
            raise ValueError("native changed original phased alleles/order")


def strict_times(ts, ages, *, diploid=True):
    check_samples(ts, ages, diploid=diploid)
    times = ts.tables.nodes.time
    if any(not math.isfinite(float(x)) or x < 0 for x in times):
        raise ValueError("nonfinite/negative native node times")
    for edge in ts.edges():
        if not times[edge.parent] > times[edge.child]:
            raise ValueError("parent must be strictly older than child")
    for mut in ts.mutations():
        if not math.isfinite(mut.time) or mut.time < times[mut.node]:
            raise ValueError("invalid native mutation time")
        if mut.edge != -1 and mut.time > times[ts.edge(mut.edge).parent]:
            raise ValueError("mutation older than its parent")


def connected_identity(ts, *, support=None):
    """Account for connected ancestry only over an explicitly covered interval."""
    left, right = (0, ts.sequence_length) if support is None else support
    if not (math.isfinite(left) and math.isfinite(right) and 0 <= left < right <= ts.sequence_length):
        raise ValueError("invalid identity support")
    samples = set(ts.samples().tolist())
    unique, occurrences, adjacent, previous = set(), 0, [], None
    counted_trees, cursor = 0, left
    for tree in ts.trees():
        interval_left = max(left, tree.interval.left)
        interval_right = min(right, tree.interval.right)
        if interval_right <= interval_left:
            continue
        if interval_left != cursor:
            raise ValueError("native topology does not cover entire identity support")
        current, roots = set(), set()
        for sample in samples:
            node = sample
            while tree.parent(node) != -1:
                node = tree.parent(node)
                if node not in samples:
                    current.add(node)
            roots.add(node)
        if len(roots) != 1 or next(iter(roots)) in samples:
            raise ValueError("connected samples do not have one inferred ancestral root inside identity support")
        counted_trees += 1
        cursor = interval_right
        unique.update(current)
        occurrences += len(current)
        if previous is not None:
            adjacent.append(len(previous & current))
        previous = current
    if cursor != right:
        raise ValueError("native topology does not cover entire identity support")
    return {"trees": counted_trees, "native_trees_total": ts.num_trees,
            "evaluated_support_bp0": [left, right],
            "uncovered_interval_inside_support_admitted": False,
            "connected_internal_occurrences": occurrences,
            "unique_connected_internal_ids": len(unique), "internal_reuse_occurrences": occurrences-len(unique),
            "adjacent_shared_internal_min": min(adjacent) if adjacent else None,
            "adjacent_shared_internal_max": max(adjacent) if adjacent else None,
            "method": "ancestor union from actual samples via tree.parent; isolated non-sample nodes excluded"}


def require_identity(ts, minimum, *, support=None):
    result = connected_identity(ts, support=support)
    if result["internal_reuse_occurrences"] < minimum:
        raise ValueError("native connected internal-ID reuse inadequate; no ID repair/fallback")
    return result


def sanitize_clip(ts, ages, left, right):
    """Clip once without simplify, time repair or relabeling any native nodes."""
    import numpy as np
    import tskit
    before_samples = check_samples(ts, ages)
    tables = ts.dump_tables()
    e = tables.edges
    l, r = np.maximum(e.left, left), np.minimum(e.right, right)
    keep = r > l
    tables.edges.set_columns(left=l[keep]-left, right=r[keep]-left,
                             parent=e.parent[keep], child=e.child[keep])
    # keep_intervals performs no simplification, but the edge-only clip above
    # requires removing sites/mutations outside the same explicit support.
    tables.delete_sites(np.flatnonzero((tables.sites.position < left) | (tables.sites.position >= right)))
    tables.sites.position = tables.sites.position - left
    tables.sequence_length = right-left
    tables.nodes.population = np.full(ts.num_nodes, tskit.NULL, dtype=np.int32)
    tables.nodes.individual = np.full(ts.num_nodes, tskit.NULL, dtype=np.int32)
    tables.nodes.flags = tables.nodes.flags & tskit.NODE_IS_SAMPLE
    for table in (tables.nodes, tables.edges, tables.sites, tables.mutations):
        table.drop_metadata()
        table.metadata_schema = tskit.MetadataSchema(None)
    tables.populations.clear()
    tables.individuals.clear()
    tables.migrations.clear()
    tables.provenances.clear()
    tables.metadata_schema = tskit.MetadataSchema(None)
    tables.metadata = b""
    tables.sort()
    clean = tables.tree_sequence()
    if clean.num_nodes != ts.num_nodes or clean.samples().tolist() != before_samples or not np.array_equal(clean.tables.nodes.time, ts.tables.nodes.time):
        raise ValueError("sanitation changed native node IDs/times/order")
    if clean.num_populations or clean.num_individuals or clean.num_migrations or clean.num_provenances or clean.metadata:
        raise ValueError("sanitation incomplete")
    strict_times(clean, ages, diploid=False)
    return clean


def pipeline(c, assets, work, initial):
    import numpy as np
    import tsinfer
    import tsdate
    import tskit
    prefix = confined(c["runtime_prefix"])
    for module in (np, tsinfer, tsdate, tskit):
        if prefix not in Path(module.__file__).resolve().parents:
            raise ValueError("inference import originated outside frozen runtime")
    journal = []

    def stage(name, callback, tree=False):
        storage_check(c, initial)
        entry = {"stage": name, "status": "started"}
        journal.append(entry)
        write_json(work/"stages.json", journal)
        started = time.monotonic()
        try:
            value = callback()
            if tree:
                out = work/(name+".trees")
                value.dump(out)
                entry["output"] = ref(out)
                if value.sequence_length != c["length_bp"]:
                    raise ValueError("native physical sequence length changed")
            entry["status"] = "complete"
            return value
        except Exception as error:
            entry.update(status="failed_preserved", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            entry["elapsed_seconds"] = time.monotonic()-started
            write_json(work/"stages.json", journal)

    all_data = tsinfer.SampleData(path=str(work/"all.samples"), sequence_length=c["length_bp"])
    for i, age in enumerate(assets["ages"][::2]):
        individual, ids = all_data.add_individual(ploidy=2, time=age)
        if individual != i or list(ids) != [2*i, 2*i+1]:
            raise ValueError("SampleData insertion order differs")
    for site in assets["sites"]:
        all_data.add_site(site["position"], site["genotypes"], alleles=site["alleles"], ancestral_allele=0)
    all_data.finalise()
    modern_ids = [i for i, age in enumerate(assets["ages"][::2]) if age == 0]
    if len(modern_ids) < 2:
        raise ValueError("documented two-step workflow needs at least two actual modern diploids")
    modern_indexes = [j for i in modern_ids for j in (2*i, 2*i+1)]
    modern = all_data.subset(individuals=modern_ids, path=str(work/"modern.samples"))
    matching = {"recombination_rate": c["recombination_rate"], "mismatch_ratio": c["mismatch_ratio"],
                "path_compression": c["path_compression"], "num_threads": c["num_threads"]}
    modern_ts = stage("modern_inferred", lambda: tsinfer.infer(modern, **matching,
                      post_process=c["post_process"], record_provenance=True), True)
    check_samples(modern_ts, [0]*len(modern_indexes))
    check_genotypes(modern_ts, assets["sites"], modern_indexes)

    def preprocess(ts):
        return tsdate.preprocess_ts(ts, **c["preprocess"], record_provenance=True)

    def date(ts):
        return tsdate.date(ts, mutation_rate=c["mutation_rate"], **c["dating"], record_provenance=True)

    modern_pre = stage("modern_preprocessed", lambda: preprocess(modern_ts), True)
    check_genotypes(modern_pre, assets["sites"], modern_indexes)
    dated_modern = stage("modern_dated", lambda: date(modern_pre), True)
    strict_times(dated_modern, [0]*len(modern_indexes))
    check_genotypes(dated_modern, assets["sites"], modern_indexes)
    site_times = stage("modern_site_ages", lambda: tsdate.sites_time_from_ts(dated_modern, **c["site_age"]))
    if len(site_times) != len(assets["sites"]) or any(np.isinf(site_times)):
        raise ValueError("native site-age output invalid")
    timed_data = stage("ancient_site_bounds", lambda: tsdate.add_sampledata_times(all_data, site_times))
    bounds = all_data.min_site_times(individuals_only=True)
    actual_times = timed_data.sites_time[:]
    finite = np.isfinite(actual_times)
    if np.any(actual_times[finite] < bounds[finite]) or np.any(actual_times[finite] < 0):
        raise ValueError("native ancient site-age constraints violated")
    write_json(work/"site_age_transfer.json", {
        "modern_individual_indexes": modern_ids,
        "modern_haplotype_indexes": modern_indexes,
        "all_observed_ages_preserved": assets["ages"],
        "native_site_times": [float(x) if np.isfinite(x) else None for x in actual_times],
        "native_nonvariable_modern_site_times_are_nan": int(np.isnan(actual_times).sum()),
        "nan_site_policy": "official tsinfer excludes from ancestor inference, preserves via native overlay; no custom time or allele fallback",
        "native_bound_function": "tsdate.add_sampledata_times; np.maximum(estimates, observed derived-carrier sample-age bounds)"})
    ancestors = stage("ancestors_generated", lambda: tsinfer.generate_ancestors(
        timed_data, path=str(work/"generated.ancestors"), num_threads=c["num_threads"]))
    if ancestors.num_sites < 1:
        raise ValueError("no native inference sites; artificial-star fallback is not admitted")
    proxies = stage("historical_proxy_ancestors", lambda: ancestors.insert_proxy_samples(
        timed_data, path=str(work/"proxy.ancestors"), **c["proxy"]))
    ancestor_ts = stage("ancestors_matched", lambda: tsinfer.match_ancestors(
        timed_data, proxies, path_compression=True, num_threads=c["num_threads"],
        time_units="generations", record_provenance=True), True)
    inferred = stage("all_samples_matched", lambda: tsinfer.match_samples(
        timed_data, ancestor_ts, **matching, force_sample_times=True,
        post_process=True, overlay_non_inference_sites=True, record_provenance=True), True)
    check_samples(inferred, assets["ages"])
    check_genotypes(inferred, assets["sites"])
    predated = stage("all_preprocessed", lambda: preprocess(inferred), True)
    check_samples(predated, assets["ages"])
    check_genotypes(predated, assets["sites"])
    dated = stage("all_dated", lambda: date(predated), True)
    strict_times(dated, assets["ages"])
    check_genotypes(dated, assets["sites"])
    identity = require_identity(dated, c["minimum_connected_internal_reuse"],
                                support=assets["identity_support_original_bp0"])
    write_json(work/"native_verification.json", {"all_original_sites_preserved": len(assets["sites"]),
               "exact_observed_genotypes_and_ages": True, "strict_native_times": True,
               "native_connected_identity": identity,
               "identity_support_original_bp0": assets["identity_support_original_bp0"],
               "native_empty_flanks_outside_identity_support_ignored": True,
               "source_labels_passed_to_inference": False,
               "truth_or_oracle_tree_read": False})
    for value in (modern, timed_data, proxies, ancestors, all_data):
        value.close()
    return dated


def infer_parent(c, parent, target, pin, initial):
    assets = load_assets(c, parent, pin)
    work = target/parent
    work.mkdir()
    native = pipeline(c, assets, work, initial)
    left, right = pin["comparison_coverage_original_bp0"]
    clean = sanitize_clip(native, assets["ages"], left, right)
    clean_sites = [{**s, "position": s["position"]-left} for s in assets["sites"] if left <= s["position"] < right]
    check_genotypes(clean, clean_sites)
    identity = require_identity(clean, c["minimum_connected_internal_reuse"])
    clean_site_ids = {site["site_id"]: i for i, site in enumerate(clean_sites)}
    out = work/"observed.trees"
    clean.dump(out)
    sample_ids = clean.samples().tolist()
    roster = []
    for row, node in zip(assets["rows"], sample_ids):
        joined = dict(row)
        joined["source_observed_node_id"] = joined.pop("observed_node_id")
        joined["observed_node_id"] = joined["inferred_tree_node_id"] = node
        roster.append(joined)
    roster_path = work/"observed_sample_manifest.json"
    write_json(roster_path, roster, new=True)
    for name, p in assets["paths"].items():
        if sha(p) != pin["observed_files_sha256"][name]:
            raise ValueError("source observations changed during inference")
    result = {"parent_id": parent, "condition": CONDITION, "status": "qualified_observed_cpu_inference",
              "split": "train" if "_train_" in parent else "val",
              "inferred_tree": {**ref(out), "condition": CONDITION},
              "observed_sample_manifest": {**ref(roster_path), "rows": len(roster)},
              "original_observed_sample_manifest": {"path": str(assets["paths"]["sample_manifest.json"]), "sha256": pin["observed_files_sha256"]["sample_manifest.json"]},
              "input_observed_sha256": pin["observed_files_sha256"],
              "sample_index_to_inferred_tree_node_id": [{"sample_index": i, "inferred_tree_node_id": n} for i, n in enumerate(sample_ids)],
              "coordinate_contract": {"original_parent_interval_bp0": [0, c["length_bp"]],
                 "normalization_length_bp": c["length_bp"], "coverage_original_bp0": [left, right],
                 "local_to_original_offset_bp": left, "physical_coordinate_transform": "physical_original = local_tree_position + coverage_left",
                 "retained_fixed100bp_indexes": grid_indexes(left, right),
                 "fixed_grid": {"path": str(assets["paths"]["scoring_coordinates.npz"]), "sha256": pin["observed_files_sha256"]["scoring_coordinates.npz"],
                                "center_formula_bp0": "50 + 100*grid_index", "total": 20000},
                 "uncovered_tails": "excluded identically to frozen Relate comparison support",
                 "site_join_location": "site_join"},
              "site_join": [{"native_full_site_id": s["site_id"], "original_site_id": s["site_id"],
                            "clean_site_id": clean_site_ids.get(s["site_id"]),
                            "VCF_position_1based": s["vcf_position_1based"],
                            "original_position_0based": s["position"],
                            "local_position_0based": s["position"]-left if left <= s["position"] < right else None,
                            "in_scored_support": left <= s["position"] < right} for s in assets["sites"]],
              "connected_identity": identity, "native_node_ids_and_times_preserved_during_sanitation": True,
              "native_full_sites": len(assets["sites"]), "sites_in_scored_support": clean.num_sites,
              "no_input_variant_filtering": True, "truth_or_oracle_read": False,
              "native_paper_reproduction_claimed": False,
              "native_artifacts": {str(p.relative_to(work)): {"sha256": sha(p), "bytes": p.stat().st_size}
                                   for p in sorted(work.rglob("*")) if p.is_file()}}
    write_json(work/"inference.json", result, new=True)
    result["completion_provenance"] = ref(work/"inference.json")
    return result


def handmade_assets():
    # Fixed hand-authored genotype table. No simulator, randomness, real data or target.
    names = ["m0", "m1", "a0", "a1"]
    ages = [0, 0, 0, 0, 10, 10, 20, 20]
    patterns = [[0,0,1,1,0,1,0,1], [0,1,0,1,1,0,1,0],
                [0,1,1,0,1,0,0,1], [1,1,0,0,1,0,1,0],
                [0,0,1,1,1,1,1,0], [0,1,0,1,0,1,1,1]]
    sites = [{"site_id": i, "position": 10+i*10, "alleles": ["A", "G"],
              "genotypes": g, "ancestral_allele": 0} for i, g in enumerate(patterns)]
    return {"names": names, "ages": ages, "sites": sites,
            "identity_support_original_bp0": [10, 61]}


def qualification_summary(c, case, dated):
    """The handmade mode uses its fixed observed-site span, never empty tails."""
    support = case["identity_support_original_bp0"]
    return {"status": "passed",
            "identity": require_identity(dated, c["minimum_connected_internal_reuse"], support=support),
            "identity_support_original_bp0": support,
            "actual_observed_arrays_opened": False, "handmade_only": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/biological_paths_tsinfer_v1.json")
    parser.add_argument("--mode", choices=["source-only", "qualify-cpu", "infer"], default="source-only")
    parser.add_argument("--expected-config-sha256")
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--admission")
    parser.add_argument("--expected-admission-sha256")
    args = parser.parse_args()
    try:
        config_path = confined(args.config)
        if args.expected_config_sha256 and sha(config_path) != args.expected_config_sha256:
            raise ValueError("expected config bytes changed")
        if args.expected_source_sha256 and sha(Path(__file__)) != args.expected_source_sha256:
            raise ValueError("expected source bytes changed")
        c = json.loads(config_path.read_text())
        report = source_only(c, config_path)
        if args.mode == "source-only":
            print(json.dumps(report, indent=2))
            return
        a, admission_path, initial = admission(c, args, config_path)
        # All nonstdlib inference imports and native execution occur after admission.
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        sys.dont_write_bytecode = True
        for name in ("TMPDIR", "NUMBA_CACHE_DIR", "XDG_CACHE_HOME"):
            dest = confined(".runtime/cache/biological_tsinfer_v1/"+name.lower())
            dest.mkdir(parents=True, exist_ok=True)
            os.environ[name] = str(dest)
        target = confined(c["qualification_root"] if args.mode == "qualify-cpu" else c["output_root"])
        target.mkdir(exist_ok=False)
        manifest = {"kind": "biological_paths_tsinfer_v1_cpu_qualification" if args.mode == "qualify-cpu" else "biological_paths_tsinfer_v1",
                    "status": "running", "condition": CONDITION, "primary_condition": CONDITION,
                    "original_parent_interval_bp0": [0, c["length_bp"]],
                    "normalization_length_bp": c["length_bp"], "config": ref(config_path),
                    "implementation": ref(Path(__file__)), "runtime": a["runtime"],
                    "admission": ref(admission_path), "storage": a["storage"],
                    "observed_asset_pins": c["observed_asset_pins"], "replicates": [],
                    "truth_or_oracle_read": False, "native_paper_reproduction_claimed": False}
        write_json(target/"manifest.partial.json", manifest, new=True)
        try:
            if args.mode == "qualify-cpu":
                case = handmade_assets()
                write_json(target/"handmade_inputs.json", case, new=True)
                dated = pipeline(c, case, target, initial)
                manifest.update(qualification_summary(c, case, dated))
            else:
                lock = json.loads(checked_ref(c["observed_asset_pins"]).read_text())
                if lock.get("kind") != "biological_paths_tsinfer_v1_observed_asset_pins" or set(lock["parents"]) != set(PARENTS):
                    raise ValueError("four-parent observation metadata lock differs")
                checked_ref({k: v for k, v in lock["basis"].items() if k in {"path", "sha256"}})
                for parent in PARENTS:
                    manifest["active_parent"] = parent
                    write_json(target/"manifest.partial.json", manifest)
                    manifest["replicates"].append(infer_parent(c, parent, target, lock["parents"][parent], initial))
                    write_json(target/"manifest.partial.json", manifest)
                manifest["status"] = "four_observed_parents_cpu_inferred_and_qualified"
            manifest["artifact_sha256"] = {str(p.relative_to(target)): sha(p) for p in sorted(target.rglob("*"))
                                           if p.is_file() and p.name not in {"manifest.json", "manifest.partial.json"}}
            storage_check(c, initial)
            if runtime_inventory(c["runtime_prefix"]) != json.loads(checked_ref(a["runtime"]).read_text())["files_sha256"]:
                raise ValueError("isolated runtime changed during execution")
        except Exception as error:
            manifest.update(status="failed_partial_preserved", error_type=type(error).__name__, error=str(error))
            write_json(target/"manifest.json", manifest)
            raise
        write_json(target/"manifest.json", manifest, new=True)
        print(json.dumps(ref(target/"manifest.json"), indent=2))
    except (ValueError, KeyError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
