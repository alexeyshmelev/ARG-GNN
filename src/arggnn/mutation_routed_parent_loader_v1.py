"""Dedicated observed-only loader for the direct whole-tree H036 candidate.

Verified original VCF, observation rosters, native inferred TS and coordinates
only. No emitter, teacher, target binding, true tree or population payload.
Full-parent native holds and observed raw float64 dates survive without tiling.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import io
import json
import tempfile

import numpy as np

from .mutation_routed_tree_observations_v1 import (
    NativeObservedTreeV1, validate_observed_sample_join_v1, observed_sham_geometry_v1,
    fit_sham_bin_edges_v1, prepare_mutation_routed_tree_observations_v1,
)

ROOT = Path(__file__).resolve().parents[2]
PARENTS = tuple(f"bio_paths_pilot_{split}_{index:03d}" for split in ("train", "val") for index in range(2))
PRODUCER = "official_documented_heterochronous_tsinfer_tsdate_task_transfer_v1"
PRIMARY_REFS = {"prepared_manifest", "observed_parent_entry", "observed_roster"}


class ObservationOnlyParentReaderV1:
    """Exact project-local hash boundary around fit.ConsumerReaderV1."""
    def __init__(self, reader):
        self.reader = reader
        self.root = Path(reader.root).resolve()
        if self.root != ROOT:
            raise ValueError("The existing project observation reader is required")
        self.evidence = {}

    def record(self, ref):
        if (not isinstance(ref, dict) or set(ref) != {"path", "sha256"}
                or not isinstance(ref["path"], str) or not isinstance(ref["sha256"], str)
                or len(ref["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in ref["sha256"])):
            raise ValueError("Exact path and SHA256 observation reference required")
        p = Path(ref["path"])
        if ".." in p.parts:
            raise ValueError("Observation path traversal forbidden")
        if p.is_absolute():
            try:
                p = p.relative_to(self.root)
            except ValueError:
                raise ValueError("External observation path forbidden") from None
        if not p.parts or str(p) == ".":
            raise ValueError("Observation file path required")
        parts = [part.lower() for part in p.parts]
        if ("truth" in parts or any("oracle" in part for part in parts)
                or any("target_bindings" in part or "census" in part for part in parts)
                or p.name.lower() in {"population_labels.txt", "source_classes.npy", "native_paths_1based.npy"}):
            raise ValueError("Observed parent preparation cannot read ancestry/oracle payloads")
        return {"path": p.as_posix(), "sha256": ref["sha256"]}

    def read(self, ref):
        ref = self.record(ref)
        data = self.reader.read(ref)
        if not isinstance(data, bytes) or hashlib.sha256(data).hexdigest() != ref["sha256"]:
            raise ValueError("Observed bytes do not match the pinned SHA256")
        self.evidence[ref["path"]] = {"sha256": ref["sha256"], "bytes": len(data)}
        return data

    def decode_json(self, ref):
        return json.loads(self.read(ref))


def _reader(reader):
    return reader if isinstance(reader, ObservationOnlyParentReaderV1) else ObservationOnlyParentReaderV1(reader)


def _ref(reader, ref):
    return reader.record({key: ref[key] for key in ("path", "sha256")})


def _join_observed_rosters_v1(source, derived, mapping, native_nodes, native_times):
    """Keep original Ana spelling, with ANA canonicalized only in join checks.

    The qualified source manifest uses Ana for the class0 cohort. The frozen
    array helper uses ANA. No class, role, date, identity or feature changes.
    """
    canonical_source, canonical_derived = [], []
    for original, inferred in zip(source, derived):
        raw, row = dict(original), dict(inferred)
        if (raw.get("role") == "pure_source_reference"
                and raw.get("observed_reference_class") == 0
                and raw.get("observed_cohort") == "Ana"):
            if row.get("observed_cohort") != "Ana":
                raise ValueError("Original and inferred pure-source cohort spelling differs")
            raw["observed_cohort"] = row["observed_cohort"] = "ANA"
        canonical_source.append(raw); canonical_derived.append(row)
    return validate_observed_sample_join_v1(canonical_source, canonical_derived,
        mapping, native_nodes, native_times)


def _source_bindings_v1(parent_id, refs, reader):
    """Metadata/roster-only inspection; no original genotype or tree arrays."""
    if parent_id not in PARENTS or set(refs) != PRIMARY_REFS:
        raise ValueError("The four frozen development parents and only observation refs are accepted")
    reader = _reader(reader)
    primary = {key: reader.record(value) for key, value in refs.items()}
    prepared = reader.decode_json(primary["prepared_manifest"])
    entry = reader.decode_json(primary["observed_parent_entry"])
    derived = reader.decode_json(primary["observed_roster"])
    split = "train" if "_train_" in parent_id else "val"
    if (prepared.get("schema") != "official_argformer_prepared_observed_arg_v1"
            or prepared.get("status") != "prepared_observed_paths_only"
            or prepared.get("parent_id") != parent_id or prepared.get("parent_split") != split
            or prepared.get("class_order") != ["ANA", "CHG", "WHG", "EHG"]
            or prepared.get("true_query_ancestry_read") is not False
            or entry.get("parent_id") != parent_id or entry.get("split") != split
            or entry.get("status") != "qualified_observed_cpu_inference"
            or entry.get("condition") != PRODUCER or entry.get("inferred_tree", {}).get("condition") != PRODUCER
            or entry.get("truth_or_oracle_read") is not False
            or entry.get("no_input_variant_filtering") is not True
            or prepared.get("files_sha256", {}).get("observed_roster.json") != primary["observed_roster"]["sha256"]
            or prepared.get("files_sha256", {}).get("observed_parent_entry.json") != primary["observed_parent_entry"]["sha256"]):
        raise ValueError("Qualified tsinfer producer, exact original parent, source labels and input lineage required")
    source_ref = _ref(reader, entry["original_observed_sample_manifest"])
    tree_ref = _ref(reader, entry["inferred_tree"])
    source = reader.decode_json(source_ref)
    if (prepared.get("source_observed_roster_sha256") != source_ref["sha256"]
            or prepared.get("inferred_tree_sha256") != tree_ref["sha256"]
            or prepared.get("derived_inferred_roster_sha256") != entry["observed_sample_manifest"]["sha256"]
            or _ref(reader, entry["observed_sample_manifest"]["source_manifest"]) != source_ref):
        raise ValueError("Source/native roster and inferred-tree hash lineage differs")
    observed_folder = f"data/biological_paths_execution_pilot_v1/{parent_id}/observed"
    hashes = entry["input_observed_sha256"]
    original_refs = {key: reader.record({"path": observed_folder + "/" + filename, "sha256": hashes[filename]})
        for key, filename in (("vcf", "genotypes.vcf"), ("site_coordinates", "site_coordinate_manifest.json"),
                              ("source_roster", "sample_manifest.json"), ("scoring_coordinates", "scoring_coordinates.npz"))}
    if original_refs["source_roster"] != source_ref:
        raise ValueError("Original VCF/coordinate observations and graph source roster differ")
    coordinates = prepared["coordinate_contract"]
    if (coordinates != entry["coordinate_contract"]
            or coordinates.get("original_parent_interval_bp0") != [0, 2_000_000]
            or coordinates.get("normalization_length_bp") != 2_000_000
            or coordinates.get("physical_coordinate_transform") != "physical_original = local_tree_position + coverage_left"
            or coordinates.get("fixed_grid", {}).get("center_formula_bp0") != "50 + 100*grid_index"
            or coordinates.get("fixed_grid", {}).get("total") != 20000
            or _ref(reader, coordinates["fixed_grid"]) != original_refs["scoring_coordinates"]
            or prepared.get("grid_sha256") != original_refs["scoring_coordinates"]["sha256"]):
        raise ValueError("Exact original physical fixed-grid coordinate lineage required")
    coverage = np.asarray(coordinates["coverage_original_bp0"], dtype=np.float64)
    if (coverage.shape != (2,) or not np.isfinite(coverage).all()
            or not 0 <= coverage[0] < coverage[1] <= 2_000_000
            or coordinates["local_to_original_offset_bp"] != coverage[0]):
        raise ValueError("Positive observed coverage and original-coordinate offset required")
    # At this stage validate the row/schema join only. Native dates are verified
    # against the actual inferred TS separately during load_parent_observations.
    mapping = entry["sample_index_to_inferred_tree_node_id"]
    nodes = np.asarray([row["inferred_tree_node_id"] for row in mapping], dtype=np.int64)
    ages = np.asarray([row["age_generations"] for row in source], dtype=np.float64)
    join = _join_observed_rosters_v1(source, derived, mapping, nodes, ages)
    return dict(parent_id=parent_id, split=split, reader=reader, prepared=prepared,
        entry=entry, source=source, derived=derived, join=join, coverage=coverage,
        offset=float(coverage[0]), refs=dict(primary, **original_refs, native_tree=tree_ref))


def inspect_parent_observation_bindings_v1(parent_id, refs, reader):
    """Small metadata/roster receipt for source integration without array reads."""
    value = _source_bindings_v1(parent_id, refs, reader)
    return dict(parent_id=parent_id, split=value["split"], observations_only=True,
        native_dates_verified_against_TS=False, actual_tree_or_genotype_arrays_read=False,
        refs=value["refs"], observed_sample_count=len(value["source"]),
        pure_source_count=int(np.count_nonzero(value["join"].reference_class >= 0)),
        unknown_input_ancestry_count=int(np.count_nonzero(value["join"].reference_class < 0)),
        actual_query_count=len(value["join"].query_id), coverage_bp0=value["coverage"].tolist(),
        retained_fixed_grid_count=len(value["prepared"]["coordinate_contract"]["retained_fixed100bp_indexes"]))


def extract_native_tree_holds_v1(ts, join, coverage, *, offset):
    """All genuine marginal holds, all sampled paths, no administrative rows.

    Only sample IDs, parent relations and native node times are consulted.
    Mutation tables, population annotations and ancestral alleles are unused.
    """
    coverage = np.asarray(coverage, dtype=np.float64)
    native_samples = np.asarray(ts.samples(), dtype=np.int64)
    if (set(native_samples.tolist()) != set(join.native_sample_node.tolist())
            or len(native_samples) != len(join.sample_id) or len(set(native_samples.tolist())) != len(native_samples)
            or float(ts.sequence_length) != coverage[1] - coverage[0] or offset != coverage[0]):
        raise ValueError("All and only observed native samples and full original coverage required")
    raw_times = np.asarray(ts.nodes_time, dtype=np.float64)
    if not np.array_equal(raw_times[join.native_sample_node], join.sample_age):
        raise ValueError("Actual native TS sampled dates differ from original observations")
    trees, intervals = [], []
    unary_count = 0; rounded_ties = 0; native_edge_count = 0
    for marginal in ts.trees():
        retained = set()
        for sample in join.native_sample_node:
            node = int(sample)
            while node >= 0 and node not in retained:
                retained.add(node); node = int(marginal.parent(node))
        native_rows = np.asarray(sorted(retained), dtype=np.int64)
        at = {node: row for row, node in enumerate(native_rows.tolist())}
        parent = np.asarray([at[int(marginal.parent(node))] if marginal.parent(node) >= 0 else -1
                             for node in native_rows], dtype=np.int64)
        times = raw_times[native_rows].copy()
        sample_rows = np.asarray([at[int(node)] for node in join.native_sample_node], dtype=np.int64)
        if np.count_nonzero(parent == -1) != 1:
            raise ValueError("Every native hold must have one connected observed sampled-ancestor union")
        nonroot = parent >= 0
        durations = times[parent[nonroot]] - times[nonroot]
        if not np.isfinite(times).all() or np.any(times < 0) or np.any(durations <= 0):
            raise ValueError("Raw CPU float64 native chronology must be strictly positive")
        child_counts = np.bincount(parent[nonroot], minlength=len(parent))
        unary_count += int(np.count_nonzero(child_counts == 1))
        clocks32 = times.astype(np.float32)
        rounded_ties += int(np.count_nonzero(clocks32[parent[nonroot]] == clocks32[nonroot]))
        native_edge_count += int(nonroot.sum())
        trees.append(NativeObservedTreeV1(parent, times, sample_rows, join.sample_id.copy()))
        intervals.append([float(marginal.interval.left) + offset, float(marginal.interval.right) + offset])
    intervals = np.asarray(intervals, dtype=np.float64)
    if (not len(trees) or len(trees) != int(ts.num_trees) or intervals.shape != (len(trees), 2)
            or intervals[0, 0] != coverage[0] or intervals[-1, 1] != coverage[1]
            or np.any(intervals[:, 1] <= intervals[:, 0])
            or not np.array_equal(intervals[:-1, 1], intervals[1:, 0])):
        raise ValueError("Every genuine original marginal interval must be preserved in order")
    return trees, intervals, dict(native_holds=len(trees), native_edge_occurrences=native_edge_count,
        unary_vertex_occurrences_preserved=unary_count, rounded_parent_child_clock_ties=rounded_ties,
        raw_chronology_validated_cpu_float64=True, analytic_cut_vertices_inserted=False,
        administrative_table_rows_excluded=True, original_holds_coalesced=False,
        trees_resampled_at_scoring_centers=False, native_populations_or_mutations_used=False)


def _verified_scratch_path_v1(root, scratch_directory):
    """Resolve an explicit project-local nonsymlink temporary directory."""
    root = Path(root).resolve()
    tmp = (root / ".runtime" / "mutation_routed_parent_loader_v1" / "tmp"
           if scratch_directory is None else Path(scratch_directory))
    if ".." in tmp.parts:
        raise ValueError("Observation scratch path traversal forbidden")
    if not tmp.is_absolute():
        tmp = root / tmp
    try:
        relative = tmp.relative_to(root)
    except ValueError:
        raise ValueError("Observation scratch directory must remain inside the project") from None
    if not relative.parts:
        raise ValueError("A dedicated observation scratch directory is required")
    for index in range(1, len(relative.parts) + 1):
        if root.joinpath(*relative.parts[:index]).is_symlink():
            raise ValueError("Observation scratch directory cannot contain symlinks")
    tmp.resolve().relative_to(root)
    return tmp


def _load_verified_ts_v1(data, root, *, scratch_directory=None):
    """Decode verified bytes in the worker's auto-deleted scratch file."""
    import tskit
    tmp = _verified_scratch_path_v1(root, scratch_directory)
    tmp.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=tmp, suffix=".observed.trees") as file:
        file.write(data); file.flush()
        return tskit.load(file.name)


def _scoring_grid_v1(data, contract):
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        if set(archive.files) != {"positions_bp", "bin_edges_bp"}:
            raise ValueError("Original observation scoring coordinate arrays required")
        positions, edges = archive["positions_bp"].copy(), archive["bin_edges_bp"].copy()
    if (positions.dtype != np.float64 or edges.dtype != np.float64
            or not np.array_equal(positions, np.arange(50., 2_000_000., 100.))
            or not np.array_equal(edges, np.arange(0., 2_000_001., 100.))):
        raise ValueError("Original full fixed100bp physical centers and bins required")
    coverage = np.asarray(contract["coverage_original_bp0"], dtype=np.float64)
    retained = np.flatnonzero((positions >= coverage[0]) & (positions < coverage[1]))
    if retained.tolist() != contract["retained_fixed100bp_indexes"] or not len(retained):
        raise ValueError("Exact original retained scoring centers required")
    clipped = edges[np.r_[retained, retained[-1] + 1]].copy()
    clipped[0] = max(clipped[0], coverage[0]); clipped[-1] = min(clipped[-1], coverage[1])
    return positions[retained], clipped, retained


def _scoring_parent_v1(value, positions, edges, retained):
    source = value["source"]; join = value["join"]
    query = join.query_id.copy()
    raw = np.asarray([source[int(i)]["observed_node_id"] for i in query], dtype=np.int64)
    cohorts = np.asarray([source[int(i)]["observed_cohort"] for i in query])
    required = {"present_day": 64, "Bronze_Age": 16, "BAA": 8, "Neo": 16, "Yam": 16}
    if len(query) != 120 or {key: int(np.count_nonzero(cohorts == key)) for key in required} != required:
        raise ValueError("Original Present64 and Ancient56 scoring strata required")
    return dict(parent_id=value["parent_id"], split=value["split"], samples=source,
        query_slots=query, raw_query_nodes=raw, cohorts=cohorts, positions=positions, edges=edges,
        retained_indexes=retained, coverage_original=value["coverage"].tolist(), query_ids=query.copy(),
        positions_original=positions, bin_edges_original=edges, query_sample_index=query,
        original_query_nodes=raw, parent_manifest_ref=value["refs"]["prepared_manifest"],
        observed_parent_entry_ref=value["refs"]["observed_parent_entry"])


@dataclass(frozen=True)
class LoadedMutationRoutedParentV1:
    parent_id: str
    split: str
    trees: list
    intervals_bp: np.ndarray
    positions_bp0: np.ndarray
    genotype: np.ndarray
    allele_count: np.ndarray
    sample_ids: np.ndarray
    reference_class: np.ndarray
    sample_age: np.ndarray
    query_ids: np.ndarray
    readout_x_bp: np.ndarray
    readout_bin_edges_bp: np.ndarray
    scoring: dict
    receipt: dict


def load_parent_observations_v1(parent_id, refs, reader, *, scratch_directory=None):
    """Authorized full observed-parent load; never invoked by source-only tests."""
    from .observed_variant_residual_v1 import parse_observed_vcf_v1
    value = _source_bindings_v1(parent_id, refs, reader)
    reader = value["reader"]
    ts = _load_verified_ts_v1(reader.read(value["refs"]["native_tree"]), reader.root,
        scratch_directory=scratch_directory)
    join = _join_observed_rosters_v1(value["source"], value["derived"],
        value["entry"]["sample_index_to_inferred_tree_node_id"], np.asarray(ts.samples(), dtype=np.int64),
        np.asarray(ts.nodes_time, dtype=np.float64)[np.asarray(ts.samples(), dtype=np.int64)])
    value["join"] = join
    trees, intervals, topology = extract_native_tree_holds_v1(ts, join, value["coverage"], offset=value["offset"])
    del ts
    coordinates = reader.decode_json(value["refs"]["site_coordinates"])
    alleles = parse_observed_vcf_v1(reader.read(value["refs"]["vcf"]).decode("utf-8"), value["source"], coordinates)
    if len(alleles.positions_bp0) != value["entry"]["native_full_sites"]:
        raise ValueError("All original variants, including multiallelic, must be retained")
    positions, edges, retained = _scoring_grid_v1(reader.read(value["refs"]["scoring_coordinates"]), value["prepared"]["coordinate_contract"])
    scoring = _scoring_parent_v1(value, positions, edges, retained)
    receipt = dict(schema="mutation_routed_loaded_parent_v1", parent_id=parent_id, split=value["split"],
        observations_only=True, targets_read=False, refs=value["refs"], producer=PRODUCER,
        topology=topology, alleles=alleles.metadata, input_samples=280, labeled_pure_proxies=96,
        unknown_input_ancestry=184, actual_query_readouts=120, raw_dates_cpu_float64=True,
        pure_cohort_spelling="original_Ana_retained;ANA_alias_only_for_frozen_join_helper",
        scoring_support="original_fixed100bp_centers_and_original_clipped_bins",
        genuine_coverage_bp0=value["coverage"].tolist(), scoring_bin_extent_bp0=[float(edges[0]),float(edges[-1])],
        unscored_coverage_tail_bp=float(edges[0]-value["coverage"][0]+value["coverage"][1]-edges[-1]),
        truth_binding_read=False, teacher_or_emitter_array_read=False)
    return LoadedMutationRoutedParentV1(parent_id, value["split"], trees, intervals,
        alleles.positions_bp0, alleles.genotypes, alleles.allele_counts, join.sample_id,
        join.reference_class, join.sample_age, join.query_id, positions, edges, scoring, receipt)


def prepare_parents_v1(parent_refs, reader, *, scratch_directory=None):
    """Load all observations, freeze TRAIN-only global bins, then prepare banks.

    The calling driver must seal this returned receipt before opening TRAIN
    targets. This routine neither receives nor opens target bindings.
    """
    from .mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeInputsV1, validate_inputs_v1
    if set(parent_refs) != set(PARENTS):
        raise ValueError("Exactly the four frozen TRAIN/VAL development parents required; no TEST")
    reader = _reader(reader)
    loaded = {pid: load_parent_observations_v1(pid, parent_refs[pid], reader,
        scratch_directory=scratch_directory) for pid in PARENTS}
    train_ids = [pid for pid in PARENTS if loaded[pid].split == "train"]
    val_ids = [pid for pid in PARENTS if loaded[pid].split == "val"]
    training_duration, training_count = [], []
    for pid in train_ids:
        duration, count = observed_sham_geometry_v1(loaded[pid].trees)
        training_duration.append(duration); training_count.append(count)
    bins = fit_sham_bin_edges_v1(np.concatenate(training_duration), np.concatenate(training_count), split="TRAIN", bins=4)
    del training_duration, training_count
    parents, parent_receipts = {}, {}
    for pid in PARENTS:
        observed = loaded.pop(pid)
        bank = prepare_mutation_routed_tree_observations_v1(observed.trees, observed.intervals_bp,
            observed.positions_bp0, observed.genotype, observed.allele_count, observed.sample_ids,
            observed.reference_class, observed.sample_age, observed.query_ids, observed.readout_x_bp,
            sham_bin_edges=bins)
        # Scoring-bin tails can differ from inference coverage. Preserve the
        # original edges in scoring/receipt while retaining all genuine trees.
        model_inputs = MutationRoutedWholeTreeInputsV1.from_arrays(bank.model_arrays())
        validation = validate_inputs_v1(model_inputs)
        receipt = dict(observed.receipt, preprocessing=bank.metadata, model_schema_validation=validation,
            original_scoring_bin_edges_bp0=observed.readout_bin_edges_bp.tolist())
        parents[pid] = dict(model_inputs=model_inputs, observations_receipt=receipt, scoring=observed.scoring)
        parent_receipts[pid] = receipt
        del bank, observed
    bins_json = asdict(bins)
    receipt = dict(schema="mutation_routed_prepared_parent_collection_v1", observations_only=True,
        targets_read=False, observations_prepared_before_target_access=True,
        sham_bin_fit_calls=1, sham_bins_fit_from_parent_ids=train_ids,
        sham_bins_applied_unchanged_to_parent_ids=list(PARENTS), sham_bin_edges=bins_json,
        parents=parent_receipts, source_byte_evidence=reader.evidence,
        teacher_or_emitter_array_read=False, target_bindings_read=False)
    return dict(parents=parents, train_parent_ids=train_ids, val_parent_ids=val_ids,
                sham_bin_edges=bins_json, receipt=receipt)
