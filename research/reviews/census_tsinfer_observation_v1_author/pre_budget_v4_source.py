"""Separate same-tsinfer observation bridge. Default mode is standard-library source-only.

Only fresh root phase admissions can decode real observations. Inspection regenerates
x48 from native tsinfer trees using the unchanged source extractor. Numerical rebuild
uses the unchanged Census bank and five-control CPU kernel. No target or fitting API.
"""
from __future__ import annotations
import argparse
import ast
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
CONFIG = "configs/census_tsinfer_observation_v1.json"
SOURCE = "src/arggnn/census_tsinfer_observation_v1.py"
OUTER_CONDITION = "current_tsinfer_tsdate_observed_heterochronous_task_transfer_v2"
HELPER_CONDITION = "official_documented_heterochronous_tsinfer_tsdate_task_transfer_v1"
CONTROLS = ["proposed", "eta0", "permuted_groups", "raw_diagonal", "local"]
ROSTER_FIELDS = {"sample_index", "sample_id", "haplotype", "observed_node_id",
                 "observed_cohort", "age_generations", "role", "observed_reference_class"}
AUTH_REFS = {"config", "review", "runtime", "storage", "launcher", "registration"}

def digest(value):
    return hashlib.sha256(value).hexdigest()

def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()

def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)

def _utc(value):
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError):
        raise ValueError("Explicit timezone-aware UTC required") from None
    if result.tzinfo is None:
        raise ValueError("Explicit timezone required")
    return result.astimezone(timezone.utc)

class VerifiedObservationReaderV1:
    """No decoder here. Each exact project-local byte record is SHA verified."""
    def __init__(self, root=ROOT, max_file_bytes=8_000_000_000):
        self.root = Path(root).resolve()
        self.max_file_bytes = max_file_bytes
        self.evidence = {}

    def relative(self, value):
        if not isinstance(value, str):
            raise ValueError("Explicit project file path required")
        p = Path(value)
        if ".." in p.parts:
            raise ValueError("Traversal forbidden")
        if p.is_absolute():
            try:
                p = p.relative_to(self.root)
            except ValueError:
                raise ValueError("External file forbidden") from None
        if not p.parts or str(p) == ".":
            raise ValueError("A file path is required")
        return p.as_posix()

    def record(self, record):
        if set(record) != {"path", "sha256"} or not _sha(record["sha256"]):
            raise ValueError("Exact path/SHA record required")
        return {"path": self.relative(record["path"]), "sha256": record["sha256"]}

    def read(self, record, *, runtime=False):
        record = self.record(record)
        path = self.root / record["path"]
        if not runtime and any((self.root / Path(*Path(record["path"]).parts[:i])).is_symlink()
                               for i in range(1, len(Path(record["path"]).parts) + 1)):
            raise ValueError("Source/observation symlink forbidden")
        if not path.is_file() or path.stat().st_size > self.max_file_bytes:
            raise ValueError("Missing/nonregular/over-budget source file")
        if not runtime:
            path.resolve().relative_to(self.root)
        data = path.read_bytes()
        if digest(data) != record["sha256"]:
            raise ValueError("Source bytes changed: " + record["path"])
        self.evidence[record["path"]] = {"sha256": record["sha256"], "bytes": len(data)}
        return data

    def decode_json(self, record):
        return json.loads(self.read(record))

class ExclusiveOutputV1:
    """Fresh bounded root output. All partial products remain on failures."""
    def __init__(self, relative, cfg, *, root=ROOT):
        self.root = Path(root).resolve()
        p = Path(relative)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError("New relative output required")
        self.path = self.root / p
        if self.path.exists() or self.path.is_symlink():
            raise ValueError("Exclusive new output required")
        self.path.resolve().relative_to(self.root)
        for parent in self.path.parents:
            if parent == self.root:
                break
            if parent.is_symlink():
                raise ValueError("Output parent symlink forbidden")
        self.path.mkdir(parents=True)
        self.limit = cfg["output_cap_bytes"]
        self.diagnostic = cfg["terminal_diagnostic_bytes"]
        self.written = 0

    def write(self, relative, data, *, terminal=False):
        if not isinstance(data, bytes) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Contained bytes output required")
        ceiling = self.limit if terminal else self.limit - self.diagnostic
        if self.written + len(data) > ceiling:
            raise ValueError("Local output reservation exceeded")
        p = self.path / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        fs = os.statvfs(self.path)
        if fs.f_bavail * fs.f_frsize < len(data) + self.diagnostic:
            raise ValueError("Local filesystem headroom exhausted")
        with p.open("xb") as stream:
            stream.write(data)
        self.written += len(data)
        return {"path": p.relative_to(self.root).as_posix(), "sha256": digest(data)}

    def write_json(self, relative, value, **kwargs):
        return self.write(relative, json_bytes(value), **kwargs)

def load_config_v1():
    cfg = json.loads((ROOT / CONFIG).read_text())
    if (cfg["kind"] != "census_tsinfer_observation_v1" or cfg["fixture_mode"] is not False
            or cfg["role_counts"] != {"query": 120, "mixed_reference_only": 64, "pure_source_reference": 96}
            or cfg["feature_columns"] != list(range(7)) + list(range(8, 48))
            or cfg["expected_tiles"] != 32 or cfg["normalization_length_bp"] != 2_000_000
            or cfg["tile_bp"] != 250_000 or cfg["census"] != 350
            or cfg["references_per_class"] != 24 or cfg["numerical_policy"]["controls"] != CONTROLS
            or cfg["scientific_fallback"] is not False or cfg["query_truth_authorized"] is not False
            or cfg["fit_authorized"] is not False
            or cfg["query_cohort_counts"] != {"present_day":64,"Bronze_Age":16,"BAA":8,"Neo":16,"Yam":16}):
        raise ValueError("Frozen same-tsinfer scientific policy changed")
    return cfg

def source_report_v1(cfg=None):
    """Standard library only. No runtime, observation decoder, model or cache."""
    cfg = cfg or load_config_v1()
    for path, expected in cfg["source_sha256"].items():
        if digest((ROOT / path).read_bytes()) != expected:
            raise ValueError("Pinned dependency changed: " + path)
    tree = ast.parse((ROOT / cfg["extractor"]["source"]).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == cfg["extractor"]["function"])
    ast_sha = digest(ast.dump(node, include_attributes=False).encode())
    if ast_sha != cfg["extractor"]["AST_sha256"]:
        raise ValueError("Original native extractor AST changed")
    old = json.loads((ROOT / "configs/census_native_qualification_runner_v1.json").read_text())
    if cfg["numerical_policy"] != {k: old[k] for k in cfg["numerical_policy"]}:
        raise ValueError("Unchanged five-control numerical policy changed")
    return {"schema": "census_tsinfer_source_report_v1", "status": "source_only_no_actual_admission",
            "source_sha256": digest(Path(__file__).read_bytes()),
            "config_sha256": digest((ROOT / CONFIG).read_bytes()),
            "extractor_AST_sha256": ast_sha, "unchanged_extractor_AST": True,
            "feature_semantic_adaptation": False, "column7": "observed encoded branch duration, dropped",
            "actual_arrays_opened": False, "neural_model_created": False, "GPU_used": False}

def _dependencies_v1(cfg=None):
    """Called only behind actual authority or directly by handmade fixture APIs."""
    cfg = cfg or load_config_v1()
    source_report_v1(cfg)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Explicit empty CUDA visibility required")
    import numpy as np
    import torch
    import tskit
    from .data import GraphSequence
    from .reference_encoder import ReferenceFeatureAdapter
    from . import census_biological_bank_v1 as bank
    from . import census_native_qualification_runner_v1 as numerical
    torch.set_num_threads(1)
    for module, path in ((bank, "src/arggnn/census_biological_bank_v1.py"),
                         (numerical, "src/arggnn/census_native_qualification_runner_v1.py")):
        if Path(module.__file__).resolve() != (ROOT / path).resolve():
            raise ValueError("Loaded scientific module origin changed")
    original = ast.parse((ROOT / cfg["extractor"]["source"]).read_text())
    node = next(n for n in original.body if isinstance(n, ast.FunctionDef)
                and n.name == cfg["extractor"]["function"])
    capsule = {"np": np, "torch": torch, "math": math, "GraphSequence": GraphSequence,
               "ReferenceFeatureAdapter": ReferenceFeatureAdapter}
    exec(compile(ast.Module(body=[node], type_ignores=[]),
                 str(ROOT / cfg["extractor"]["source"]), "exec"), capsule)
    return np, torch, tskit, bank, numerical, capsule["pack_observed_tile"]

def _observation_record_v1(reader, record):
    ref = reader.record({"path": record["path"], "sha256": record["sha256"]})
    if any(any(w in part.lower() for w in ("truth", "oracle", "target"))
           for part in Path(ref["path"]).parts):
        raise ValueError("Target/oracle/truth paths are never observation capabilities")
    return ref

def metadata_inputs_v1(cfg, reader):
    """Verify exact frozen four-parent and genuine ARGformer JSON lineage."""
    values = {p: reader.decode_json({"path": p, "sha256": h})
              for p, h in cfg["metadata_input_sha256"].items()}
    combined = values[cfg["combined_manifest_ref"]["path"]]
    summary = values[cfg["ARGformer_summary_ref"]["path"]]
    if (combined.get("kind") != "biological_paths_tsinfer_observed_v2"
            or combined.get("status") != "four_observed_development_parents_inferred1000"
            or combined.get("condition") != OUTER_CONDITION
            or combined.get("source_helper_condition") != HELPER_CONDITION
            or combined.get("truth_or_oracle_read") is not False
            or combined.get("source_labels_passed_to_inference") is not False
            or summary.get("status") != "all_four_full_native_corpora_qualified_no_model_fit_or_retrieval"
            or summary.get("query_truth_used") is not False
            or summary.get("model_fit_or_retrieval_executed") is not False):
        raise ValueError("Exact completed tsinfer/ARGformer observed lineage required")
    expected = {p["parent_id"]: p for p in cfg["parents"]}
    rows = combined["replicates"]
    if (len(rows) != 4 or {r["parent_id"] for r in rows} != set(expected)
            or set(summary["parents"]) != set(expected)):
        raise ValueError("Exact four independent development parents required")
    helpers = []
    for row in rows:
        h = row["frozen_helper_result"]
        pin = expected[row["parent_id"]]
        dispatcher = values[pin["ARGformer_dispatcher_ref"]["path"]]
        qualification = values[pin["ARGformer_qualification_ref"]["path"]]
        prepared = values[pin["ARGformer_prepared_manifest_ref"]["path"]]
        native = _observation_record_v1(reader, h["inferred_tree"])
        if (h["parent_id"] != pin["parent_id"] or h["split"] != pin["split"]
                or row["condition"] != OUTER_CONDITION or h["condition"] != HELPER_CONDITION
                or h["inferred_tree"]["condition"] != HELPER_CONDITION
                or h["status"] != "qualified_observed_cpu_inference"
                or h["truth_or_oracle_read"] is not False
                or h["native_node_ids_and_times_preserved_during_sanitation"] is not True
                or native != pin["tree_ref"]
                or dispatcher["namespace"] != pin["namespace"]
                or dispatcher["combined_observed_manifest"] != cfg["combined_manifest_ref"]
                or dispatcher["actual_native_preparation_completed"] is not True
                or dispatcher["query_truth_used"] is not False
                or dispatcher["model_fit_or_retrieval_executed"] is not False
                or dispatcher["status"] != "qualified_full_native_preparation"
                or dispatcher["corpus_qualification"] != pin["ARGformer_qualification_ref"]
                or dispatcher["prepared_manifest"] != pin["ARGformer_prepared_manifest_ref"]
                or qualification["status"] != "qualified_full_native_preparation"
                or qualification["query_truth_used"] is not False
                or qualification["native_original_node_IDs_used"] is not True):
            raise ValueError("Same native ARG namespace/receipt condition changed")
        joins = {"inferred_tree_sha256": native["sha256"],
                 "derived_inferred_roster_sha256": h["observed_sample_manifest"]["sha256"],
                 "source_observed_roster_sha256": h["original_observed_sample_manifest"]["sha256"],
                 "grid_sha256": h["coordinate_contract"]["fixed_grid"]["sha256"],
                 "completion_provenance_sha256": h["completion_provenance"]["sha256"],
                 "input_manifest_sha256": cfg["combined_manifest_ref"]["sha256"],
                 "namespace": pin["namespace"], "parent_id": pin["parent_id"],
                 "parent_split": pin["split"], "true_query_ancestry_read": False,
                 "class_order": [c.upper() for c in cfg["classes"]]}
        if any(prepared.get(k) != v for k, v in joins.items()) or prepared["coordinate_contract"] != h["coordinate_contract"]:
            raise ValueError("Native ARGformer prepared roster/tree/grid lineage differs")
        if h["coordinate_contract"]["coverage_original_bp0"] != pin["coverage_bp"]:
            raise ValueError("Original common physical support changed")
        helpers.append((h, pin))
    return helpers

def role_join_v1(ts, original, derived, mapping, cfg=None, *, fixture=False):
    """Only observed proxy class at R. Query cohorts never enter the feature parent."""
    cfg = cfg or load_config_v1()
    count = len(original)
    wanted = cfg["role_counts"]
    if fixture:
        wanted = {k: sum(r.get("role") == k for r in original) for k in wanted}
    if (count != sum(wanted.values()) or len(derived) != count or len(mapping) != count
            or [m.get("sample_index") for m in mapping] != list(range(count))):
        raise ValueError("Complete ordered source/native roster required")
    np, torch, _, bank, _, _ = _dependencies_v1(cfg)
    native_ids = []
    roles = []
    for i, (raw, row, joined) in enumerate(zip(original, derived, mapping)):
        if set(raw) != ROSTER_FIELDS or raw["sample_index"] != i or type(raw["sample_index"]) is not int:
            raise ValueError("Observation-only original roster field whitelist required")
        if set(joined) != {"sample_index", "inferred_tree_node_id"}:
            raise ValueError("Explicit native sample mapping required")
        native = joined["inferred_tree_node_id"]
        if type(native) is not int or native < 0 or type(raw["observed_node_id"]) is not int:
            raise ValueError("Observed/native ID types differ")
        expected = dict(raw)
        expected["source_observed_node_id"] = expected.pop("observed_node_id")
        expected["observed_node_id"] = expected["inferred_tree_node_id"] = native
        if row != expected:
            raise ValueError("Source-to-native row/date/identity changed")
        role, cls, age = raw["role"], raw["observed_reference_class"], raw["age_generations"]
        if role not in wanted or type(age) not in (int, float) or not math.isfinite(age) or age < 0:
            raise ValueError("Exact observed role/date required")
        if role == "pure_source_reference":
            if type(cls) is not int or cls not in range(4) or raw["observed_cohort"] != cfg["classes"][cls]:
                raise ValueError("Pure observed proxy class/cohort differs")
        elif cls is not None or age >= 350:
            raise ValueError("Query/context ancestry label or ineligible date forbidden")
        if float(ts.node(native).time) != age:
            raise ValueError("Direct native sample date changed")
        if not isinstance(raw["sample_id"], str) or not raw["sample_id"] or raw["haplotype"] not in (0, 1):
            raise ValueError("Observed diploid/haplotype identity required")
        roles.append(bank.ObservedSampleRoleV1(i, native, role, age, cls))
        native_ids.append(native)
    if (len(set(native_ids)) != count or set(native_ids) != set(map(int, ts.samples()))
            or {k: sum(r.role == k for r in roles) for k in wanted} != wanted):
        raise ValueError("All and only observed native samples/roles required")
    if not fixture:
        cohorts = {c: sum(r["role"] == "query" and r["observed_cohort"] == c for r in original)
                   for c in cfg["query_cohort_counts"]}
        if cohorts != cfg["query_cohort_counts"]:
            raise ValueError("Exact present64/ancient56 query cohort contract changed")
        for i in range(0, count, 2):
            a, b = original[i:i + 2]
            if (a["sample_id"] != b["sample_id"] or [a["haplotype"], b["haplotype"]] != [0, 1]
                    or any(a[k] != b[k] for k in ("role", "age_generations", "observed_cohort", "observed_reference_class"))):
                raise ValueError("Original diploid pairing changed")
    nref = count if fixture else cfg["references_per_class"]
    counts = [sum(r.role == "pure_source_reference" and r.observed_reference_class == c for r in roles)
              for c in range(4)]
    if not counts or min(counts) < 1 or len(set(counts)) != 1 or (not fixture and counts != [nref] * 4):
        raise ValueError("Four equally represented pure proxy classes required")
    return roles

def physical_contract_v1(helper, positions, edges, cfg=None):
    cfg = cfg or load_config_v1()
    np, _, _, _, _, _ = _dependencies_v1(cfg)
    c = helper["coordinate_contract"]
    coverage, offset = c["coverage_original_bp0"], c["local_to_original_offset_bp"]
    if (positions.dtype != np.float64 or edges.dtype != np.float64
            or not np.array_equal(positions, np.arange(50., 2_000_000., 100.))
            or not np.array_equal(edges, np.arange(0., 2_000_001., 100.))
            or c["original_parent_interval_bp0"] != [0, 2_000_000]
            or c["normalization_length_bp"] != 2_000_000 or not 0 <= coverage[0] < coverage[1] <= 2_000_000
            or offset != coverage[0] or c["physical_coordinate_transform"] != "physical_original = local_tree_position + coverage_left"):
        raise ValueError("Original physical2Mb/fixed100bp coordinate contract required")
    retained = np.flatnonzero((positions >= coverage[0]) & (positions < coverage[1]))
    if retained.tolist() != c["retained_fixed100bp_indexes"] or not len(retained):
        raise ValueError("Exact original retained center support required")
    selected_edges = edges[np.r_[retained, retained[-1] + 1]].copy()
    selected_edges[0] = max(selected_edges[0], coverage[0])
    selected_edges[-1] = min(selected_edges[-1], coverage[1])
    return retained, selected_edges

def extract_native_tsinfer_tile_v1(ts, roles, positions, coverage, tile_id, cfg=None, *, fixture=False):
    """Genuine unmodified AST extraction on THIS supplied native TS, never a cache."""
    cfg = cfg or load_config_v1()
    np, torch, _, bank, _, extractor = _dependencies_v1(cfg)
    if type(tile_id) is not int or tile_id not in range(8):
        raise ValueError("Original tile ID required")
    if not fixture and len(roles) != 280:
        raise ValueError("All280 observed samples required")
    if len({r.actual_id for r in roles}) != len(roles):
        raise ValueError("Persistent sampled IDs are unique join keys")
    if set(r.native_sample_node for r in roles) != set(map(int, ts.samples())):
        raise ValueError("Native sample roster changed")
    left, right = max(tile_id * 250000., coverage[0]), min((tile_id + 1) * 250000., coverage[1])
    if not left < right or ts.sequence_length != coverage[1] - coverage[0]:
        raise ValueError("Native local span/common coverage differ")
    # Complete observed class flags, not simulation population fields.
    parent = {"observation": {"nodes": np.array([r.native_sample_node for r in roles], dtype=np.int64),
                              "offset": coverage[0]},
              "dynamics_slots": np.array([i for i, r in enumerate(roles) if r.role != "pure_source_reference"]),
              "query_slots": np.array([i for i, r in enumerate(roles) if r.role == "query"]),
              "reference_slots": np.array([i for i, r in enumerate(roles) if r.role == "pure_source_reference"]),
              "samples": [{"observed_reference_class": r.observed_reference_class} for r in roles],
              "positions": positions}
    sequence, indexes, tree_ids, landmarks = extractor(ts, parent, left, right)
    raw = {"sequence": sequence.as_dict(), "labels_in_input": False}
    expected_mask = {r.native_sample_node for r in roles if r.role == "query"}
    refclasses = {r.native_sample_node: r.observed_reference_class for r in roles if r.role == "pure_source_reference"}
    for node, x in zip(sequence.node_ids.tolist(), sequence.x):
        expected = torch.zeros(4, dtype=torch.float32)
        if node in refclasses:
            expected[refclasses[node]] = 1.
        if not torch.equal(x[:4], expected) or float(x[4]) != float(node in expected_mask):
            raise ValueError("Pure-reference leaf ancestry mask/query-context marker differs")
    if sequence.x.dtype != torch.float32 or sequence.intervals.dtype != torch.float64:
        raise ValueError("Original feature/native physical dtype policy differs")
    return raw, indexes, tree_ids, landmarks, left, right

def raw_sequence_bytes_v1(raw, cfg=None):
    _, torch, _, _, _, _ = _dependencies_v1(cfg)
    stream = io.BytesIO()
    torch.save(raw, stream)
    return stream.getvalue()

@dataclass(frozen=True)
class CertifiedCensusTsinferObservationV1:
    native_certificate: object
    lineage: dict
    certificate_sha256: str

def _outer_content_v1(inner, lineage):
    return {"schema": "census_tsinfer_observation_certificate_v1",
            "inner_certificate_sha256": inner.certificate_sha256, "lineage": lineage}

def certify_native_tile_v1(ts, roles, positions, coverage, tile_id, source_bindings,
                           lineage, cfg=None, *, fixture=False, cached_raw=None):
    """Certify fixed materialized x48 by independent unchanged native-bank replay."""
    cfg = cfg or load_config_v1()
    np, torch, _, bank, _, _ = _dependencies_v1(cfg)
    fresh, indexes, trees, landmarks, left, right = extract_native_tsinfer_tile_v1(
        ts, roles, positions, coverage, tile_id, cfg, fixture=fixture)
    raw = fresh if cached_raw is None else cached_raw
    adapter_cfg = bank.load_census_biological_config_v1()
    if fixture:
        adapter_cfg.update(fixture_mode=True,
                           role_counts={k: sum(r.role == k for r in roles) for k in cfg["role_counts"]},
                           references_per_class=sum(r.role == "pure_source_reference" for r in roles) // 4)
    inner = bank.prepare_census_biological_bank_v1(
        ts, raw, roles, left_bp=left, right_bp=right, offset_bp=coverage[0],
        positions_bp=torch.tensor(positions[indexes], dtype=torch.float64),
        original_grid_indexes=torch.tensor(indexes, dtype=torch.long),
        source_bindings=source_bindings, parent_id=lineage["parent_id"], split=lineage["split"],
        tile_id=tile_id, config=adapter_cfg)
    if inner.provenance["original_tree_ids"] != trees or inner.provenance["landmark_native_nodes"] != landmarks:
        raise ValueError("Same native original-tree/landmark extraction changed")
    meta = dict(lineage, original_grid_indexes=indexes.tolist(), original_tree_ids=trees,
                features="unchanged_native_extractor_x48_then_F47_columns0-6_and8-47",
                sampled_IDs="joins_only", internal_IDs="native_inferred_ancestors_not_genomes_or_learned_IDs",
                direct_native_float64_times=True, no_copied_Relate_features=True)
    outer = CertifiedCensusTsinferObservationV1(
        inner, meta, bank.fingerprint_sha256_v1(_outer_content_v1(inner, meta)))
    return outer, fresh

def validate_tsinfer_certificate_v1(cert, expected_sha256, cfg=None):
    _, _, _, bank, _, _ = _dependencies_v1(cfg)
    bank.validate_census_biological_certificate_v1(
        cert.native_certificate, cert.native_certificate.certificate_sha256)
    if (cert.certificate_sha256 != expected_sha256
            or bank.fingerprint_sha256_v1(_outer_content_v1(cert.native_certificate, cert.lineage)) != expected_sha256):
        raise ValueError("Changed same-tsinfer source/role/native/input certificate")
    return cert

def forward_certified_tsinfer_v1(model, cert, *, expected_certificate_sha256, control="proposed",
                                  group_permutation=None, readout_chunk_size=128, cfg=None):
    """Future root consumer: same unchanged model, after both certificate guards. No fit."""
    validate_tsinfer_certificate_v1(cert, expected_certificate_sha256, cfg)
    _, _, _, bank, _, _ = _dependencies_v1(cfg)
    inner = cert.native_certificate
    return bank.forward_certified_census_biological_v1(
        model, inner, expected_certificate_sha256=inner.certificate_sha256,
        control=control, group_permutation=group_permutation, readout_chunk_size=readout_chunk_size)


def admit_phase_v1(admission_ref, phase, reader=None):
    """Source/metadata authority before scientific imports or observation decoding."""
    reader = reader or VerifiedObservationReaderV1()
    a = reader.decode_json(admission_ref)
    required = {"schema", "status", "authorizer", "job_id", "phase", "created_at_utc",
                "expires_at_utc", "refs", "source_sha256", "observed_artifact_sha256",
                "output_directory", "inspection_manifest", "expected_certificates",
                "derived_artifact_sha256", "initial_state_sha256_by_dtype", "group_permutation",
                "query_truth_authorized", "fit_authorized"}
    if (set(a) != required or a["schema"] != "census_tsinfer_observation_admission_v1"
            or a["status"] != "active" or a["authorizer"] != "root_agent"
            or a["phase"] != phase or phase not in ("inspect", "numerical")
            or a["query_truth_authorized"] is not False or a["fit_authorized"] is not False
            or set(a["refs"]) != AUTH_REFS):
        raise ValueError("Fresh exact root observation-only phase authority required")
    cfg = reader.decode_json(a["refs"]["config"])
    if reader.record(a["refs"]["config"]) != {"path": CONFIG, "sha256": digest((ROOT / CONFIG).read_bytes())} or cfg != load_config_v1():
        raise ValueError("Exact shipped new bridge configuration required")
    source_report_v1(cfg)
    clock = datetime.now(timezone.utc)
    if (not _utc(a["created_at_utc"]) <= clock < _utc(a["expires_at_utc"])
            or (_utc(a["expires_at_utc"]) - _utc(a["created_at_utc"])).total_seconds() > cfg["admission_max_age_seconds"]
            or (clock - _utc(a["created_at_utc"])).total_seconds() > cfg["admission_max_age_seconds"]):
        raise ValueError("Fresh bounded UTC authority required")
    if not isinstance(a["job_id"], str) or not a["job_id"] or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in a["job_id"]):
        raise ValueError("Canonical root job ID required")
    expected_sources = dict(cfg["source_sha256"], **{SOURCE: digest(Path(__file__).read_bytes())})
    if a["source_sha256"] != expected_sources or a["observed_artifact_sha256"] != cfg["observed_artifact_sha256"]:
        raise ValueError("Exact complete source/observed closure required")
    for p, h in a["source_sha256"].items():
        reader.read({"path": p, "sha256": h})
    review = reader.decode_json(a["refs"]["review"])
    if (review.get("schema") != "census_tsinfer_observation_independent_review_v1"
            or review.get("review_role") != "independent_reviewer" or review.get("verdict") != "PASS"
            or review.get("fixture_mode") is not False or review.get("actual_admission") is not False
            or review.get("source_sha256") != expected_sources[SOURCE]
            or review.get("config_sha256") != a["refs"]["config"]["sha256"]):
        raise ValueError("Fresh independently reviewed exact source/config required")
    registration = reader.decode_json(a["refs"]["registration"])
    if (registration.get("schema") != "census_tsinfer_observation_root_registration_v1"
            or registration.get("authorizer") != "root_agent" or registration.get("status") != "registered"
            or registration.get("condition") != cfg["condition"] or not registration.get("hypothesis_id")
            or registration.get("phase") != phase or registration.get("job_id") != a["job_id"]
            or registration.get("query_truth_authorized") is not False or registration.get("fit_authorized") is not False):
        raise ValueError("New root experiment registration required")
    prefix = cfg["inspection_output_prefix"] + a["job_id"]
    if a["output_directory"] != prefix:
        raise ValueError("Exact exclusive new output namespace required")
    storage = reader.decode_json(a["refs"]["storage"])
    if (storage.get("schema") != "census_tsinfer_observation_storage_v1"
            or storage.get("status") != "active" or storage.get("job_id") != a["job_id"]
            or storage.get("output_directory") != prefix
            or type(storage.get("reserved_new_bytes")) is not int or storage["reserved_new_bytes"] < cfg["output_cap_bytes"]
            or type(storage.get("used_project_bytes")) is not int or storage["used_project_bytes"] < 0
            or type(storage.get("concurrent_reserve_bytes")) is not int or storage["concurrent_reserve_bytes"] < 0
            or storage["used_project_bytes"] + storage["reserved_new_bytes"] + storage["concurrent_reserve_bytes"] >= 450_000_000_000
            or storage.get("project_limit_bytes") != 500_000_000_000
            or not _utc(storage["created_at_utc"]) <= clock < _utc(storage["expires_at_utc"])):
        raise ValueError("Fresh bounded root storage authority required")
    snap = reader.decode_json(storage["snapshot_ref"])
    if (snap.get("schema") != "census_native_root_storage_snapshot_v1" or snap.get("status") != "complete"
            or snap.get("returncode") != 0 or snap.get("project_root") != str(ROOT)
            or snap.get("used_project_bytes") != storage["used_project_bytes"]
            or not 0 <= (clock - _utc(snap["created_at_utc"])).total_seconds() <= cfg["storage_snapshot_max_age_seconds"]):
        raise ValueError("Fresh complete root storage measurement required")
    launcher = reader.decode_json(a["refs"]["launcher"])
    if launcher != {"schema": "census_tsinfer_observation_CPU_launcher_v1", "job_id": a["job_id"],
                    "python": ".runtime/env/bin/python", "CUDA_VISIBLE_DEVICES": "",
                    "PYTHONDONTWRITEBYTECODE": "1", "GPU_admitted": False}:
        raise ValueError("Exact CPU launcher authority required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise ValueError("CPU-only disabled-bytecode runtime required")
    if Path(sys.executable).resolve() != (ROOT / ".runtime/env/bin/python").resolve():
        raise ValueError("Pinned main project Python required")
    runtime = reader.decode_json(a["refs"]["runtime"])
    if (runtime.get("schema") != "census_tsinfer_observation_CPU_runtime_v1"
            or set(runtime.get("runtime_file_sha256", {})) != set(cfg["runtime_files"])):
        raise ValueError("Exact compiled CPU runtime file closure required")
    for p, h in runtime["runtime_file_sha256"].items():
        reader.read({"path": p, "sha256": h}, runtime=True)
    _, _, _, _, numerical, _ = _dependencies_v1(cfg)
    if runtime.get("identity") != numerical.runtime_identity_v1():
        raise ValueError("Actual CPU runtime identity differs")
    if phase == "inspect":
        if (a["inspection_manifest"] is not None or a["expected_certificates"] or a["derived_artifact_sha256"]
                or a["initial_state_sha256_by_dtype"] or a["group_permutation"]):
            raise ValueError("Inspection cannot grant itself numerical admission")
    else:
        if (not a["inspection_manifest"] or len(a["expected_certificates"]) != 32
                or set(a["initial_state_sha256_by_dtype"]) != {"float64", "float32"}
                or any(not _sha(h) for h in a["initial_state_sha256_by_dtype"].values())
                or any(type(i) is not int for i in a["group_permutation"])
                or sorted(a["group_permutation"]) != list(range(184))):
            raise ValueError("Independent frozen inspection/weights/global permutation required")
    return a, cfg, reader

def _load_parent_v1(h, pin, cfg, reader, tmp):
    np, _, tskit, _, _, _ = _dependencies_v1(cfg)
    refs = {"native": _observation_record_v1(reader, h["inferred_tree"]),
            "source": _observation_record_v1(reader, h["original_observed_sample_manifest"]),
            "derived": _observation_record_v1(reader, h["observed_sample_manifest"]),
            "grid": _observation_record_v1(reader, h["coordinate_contract"]["fixed_grid"]),
            "completion": _observation_record_v1(reader, h["completion_provenance"])}
    original, derived = reader.decode_json(refs["source"]), reader.decode_json(refs["derived"])
    completion = reader.decode_json(refs["completion"])
    if completion != {k: v for k, v in h.items() if k != "completion_provenance"}:
        raise ValueError("Original tsinfer completion changed")
    data = reader.read(refs["native"])
    with tempfile.NamedTemporaryFile(dir=tmp, suffix=".observed.trees") as f:
        f.write(data)
        f.flush()
        ts = tskit.load(f.name)
    roles = role_join_v1(ts, original, derived, h["sample_index_to_inferred_tree_node_id"], cfg)
    with np.load(io.BytesIO(reader.read(refs["grid"])), allow_pickle=False) as z:
        if set(z.files) != {"positions_bp", "bin_edges_bp"}:
            raise ValueError("Original observed grid whitelist required")
        positions, edges = z["positions_bp"].copy(), z["bin_edges_bp"].copy()
    retained, selected_edges = physical_contract_v1(h, positions, edges, cfg)
    coverage = h["coordinate_contract"]["coverage_original_bp0"]
    if ts.sequence_length != coverage[1] - coverage[0]:
        raise ValueError("Exact same-tsinfer local span changed")
    lineage = {"parent_id": pin["parent_id"], "split": pin["split"], "namespace": pin["namespace"],
               "native_tree_ref": refs["native"], "original_roster_ref": refs["source"],
               "derived_roster_ref": refs["derived"], "grid_ref": refs["grid"],
               "completion_ref": refs["completion"], "ARGformer_dispatcher_ref": pin["ARGformer_dispatcher_ref"],
               "ARGformer_qualification_ref": pin["ARGformer_qualification_ref"],
               "ARGformer_prepared_manifest_ref": pin["ARGformer_prepared_manifest_ref"],
               "config_ref": {"path": CONFIG, "sha256": digest((ROOT / CONFIG).read_bytes())},
               "source_sha256": digest(Path(__file__).read_bytes()),
               "extractor_AST_sha256": cfg["extractor"]["AST_sha256"],
               "scored_actual_ids": [r.actual_id for r in roles if r.role == "query"],
               "dynamics_actual_ids": [r.actual_id for r in roles if r.role != "pure_source_reference"],
               "sample_index_to_native_id": [[r.actual_id, r.native_sample_node] for r in roles],
               "native_sample_dates": [r.age_generations for r in roles],
               "ordered_subject_haplotypes": [[r["sample_id"], r["haplotype"]] for r in original],
               "scoring_strata_observed_only": {str(i): r["observed_cohort"] for i, r in enumerate(original) if r["role"] == "query"},
               "retained_original_grid_indexes": retained.tolist(), "bin_support_bp": [float(selected_edges[0]), float(selected_edges[-1])],
               "denominator_present64_query_bp": float(selected_edges[-1] - selected_edges[0]) * 64,
               "coverage_original_bp": coverage}
    loaded = (ts, roles, positions, coverage[0], coverage, refs, lineage)
    return loaded

def _producer_payload_v1(raw, lineage, tile_id, cfg, raw_ref):
    _, _, _, bank, _, _ = _dependencies_v1(cfg)
    return {"schema": "census_tsinfer_native_x48_materialization_v1", "status": "fresh_native_source_extraction",
            "parent_id": lineage["parent_id"], "tile_id": tile_id, "namespace": lineage["namespace"],
            "native_tree_ref": lineage["native_tree_ref"], "source_sha256": lineage["source_sha256"],
            "extractor_AST_sha256": cfg["extractor"]["AST_sha256"], "config_ref": lineage["config_ref"],
            "raw_ref": raw_ref, "raw_value_fingerprint_sha256": bank.fingerprint_sha256_v1(raw),
            "feature_columns": cfg["feature_columns"], "mixed_query_flag": 0,
            "query_population_or_ancestry_used": False, "cache_from_Relate": False,
            "native_internal_IDs_preserved": True}

def run_phase_v1(admission_ref, phase):
    a, cfg, reader = admit_phase_v1(admission_ref, phase)
    helpers = metadata_inputs_v1(cfg, reader)
    # Verify the entire original file closure before the first native/grid decoder.
    for p, h in a["observed_artifact_sha256"].items():
        reader.read(_observation_record_v1(reader, {"path": p, "sha256": h}))
    frozen = None
    frozen_cells = {}
    if phase == "numerical":
        frozen = reader.decode_json(a["inspection_manifest"])
        if (frozen.get("schema") != "census_tsinfer_observation_evidence_v1"
                or frozen.get("phase") != "inspect" or frozen.get("status") != "inspection_complete_pending_root_freeze"
                or frozen.get("failed_cells") or len(frozen.get("cells", [])) != 32
                or frozen.get("source_sha256") != a["source_sha256"]
                or frozen.get("observed_artifact_sha256") != a["observed_artifact_sha256"]
                or frozen.get("metadata_input_sha256") != cfg["metadata_input_sha256"]):
            raise ValueError("Exact complete new-input inspection freeze required")
        frozen_cells = {f"{c['parent_id']}:{c['tile_id']}": c for c in frozen["cells"]}
        if len(frozen_cells) != 32 or {k: c["certificate_sha256"] for k, c in frozen_cells.items()} != a["expected_certificates"]:
            raise ValueError("Frozen new-input per-cell certificates differ")
        closure = {r["path"]: r["sha256"] for c in frozen["cells"] for r in (c["raw_ref"], c["producer_ref"])}
        if a["derived_artifact_sha256"] != closure:
            raise ValueError("Frozen inspected materialization closure changed")
        for p, h in closure.items():
            if not p.startswith(cfg["inspection_output_prefix"]):
                raise ValueError("Copied/foreign cache path forbidden")
            reader.read({"path": p, "sha256": h})
    output = ExclusiveOutputV1(a["output_directory"], cfg)
    tmp = output.path / "tmp"
    tmp.mkdir()
    np, torch, _, bank, numerical, _ = _dependencies_v1(cfg)
    result = {"schema": "census_tsinfer_observation_evidence_v1", "phase": phase, "job_id": a["job_id"],
              "status": "running", "source_sha256": a["source_sha256"],
              "metadata_input_sha256": cfg["metadata_input_sha256"],
              "observed_artifact_sha256": a["observed_artifact_sha256"],
              "admission_ref": reader.record(admission_ref), "fixture_mode": False,
              "biological_targets_opened": False, "fit_performed": False, "GPU_used": False,
              "feature_semantic_adaptation": False, "old_Relate_arm_unchanged": True,
              "cells": [], "failed_cells": [], "expected_cells": 32}
    try:
        tasks = [(h, pin, tile) for h, pin in helpers for tile in range(8)]
        if phase == "numerical":
            selection = bank.select_census_numerical_tiles_v1(
                [{k: c[k] for k in ("parent_id", "split", "tile_id", "num_trees")} for c in frozen["cells"]])
            priority = [(selection[k]["parent_id"], selection[k]["tile_id"]) for k in ("minimum", "maximum")]
            tasks.sort(key=lambda x: (0 if (x[0]["parent_id"], x[2]) in priority else 1,
                                     priority.index((x[0]["parent_id"], x[2])) if (x[0]["parent_id"], x[2]) in priority else 0,
                                     x[0]["parent_id"], x[2]))
            result["prespecified_TRAIN_numerical_cells"] = selection
        current, loaded = None, None
        for h, pin, tile in tasks:
            cell = {"parent_id": h["parent_id"], "split": h["split"], "tile_id": tile}
            identity = f"{h['parent_id']}:{tile}"
            try:
                if datetime.now(timezone.utc) >= _utc(a["expires_at_utc"]):
                    raise ValueError("Active phase expired")
                if current != h["parent_id"]:
                    loaded = _load_parent_v1(h, pin, cfg, reader, tmp)
                    current = h["parent_id"]
                ts, roles, positions, offset, coverage, refs, base_lineage = loaded
                fresh, indexes, tree_ids, landmarks, left, right = extract_native_tsinfer_tile_v1(
                    ts, roles, positions, coverage, tile, cfg)
                if phase == "inspect":
                    relative = f"{h['parent_id']}/tile_{tile:02d}"
                    raw_ref = output.write(relative + ".whole_observed.pt", raw_sequence_bytes_v1(fresh, cfg))
                    producer = _producer_payload_v1(fresh, base_lineage, tile, cfg, raw_ref)
                    producer_ref = output.write_json(relative + ".producer.json", producer)
                    raw = fresh
                else:
                    previous = frozen_cells[identity]
                    raw_ref, producer_ref = previous["raw_ref"], previous["producer_ref"]
                    producer = reader.decode_json(producer_ref)
                    raw = torch.load(io.BytesIO(reader.read(raw_ref)), map_location="cpu", weights_only=True)
                    if producer != _producer_payload_v1(fresh, base_lineage, tile, cfg, raw_ref):
                        raise ValueError("Fresh native extraction/source or cached observation contaminated")
                lineage = dict(base_lineage, raw_ref=raw_ref, producer_ref=producer_ref)
                bindings = {"native_observed_ts": refs["native"], "serialized_whole_observed": raw_ref,
                            "observed_roster": refs["derived"], "observed_grid": refs["grid"],
                            "producer_receipt": producer_ref}
                cert, _ = certify_native_tile_v1(ts, roles, positions, coverage, tile,
                                                bindings, lineage, cfg, cached_raw=raw)
                inner = cert.native_certificate
                structural = numerical.structural_cell_v1(inner, hidden_dim=cfg["numerical_policy"]["model"]["hidden_dim"])
                collision = numerical.event_collision_counts_v1(ts, left, right, offset, positions[indexes])
                cell.update(num_trees=len(inner.native_trees), certificate_sha256=cert.certificate_sha256,
                            inner_certificate_sha256=inner.certificate_sha256, raw_ref=raw_ref,
                            producer_ref=producer_ref, lineage=cert.lineage, structural=structural,
                            event_collision_counts=collision)
                if phase == "inspect":
                    cell["status"] = "inspected_pending_root_freeze"
                else:
                    previous = frozen_cells[identity]
                    validate_tsinfer_certificate_v1(cert, a["expected_certificates"][identity], cfg)
                    for key in ("inner_certificate_sha256", "lineage", "structural", "event_collision_counts", "num_trees"):
                        if cell[key] != previous[key]:
                            raise ValueError("Rebuilt frozen input/certificate/geometry differs: " + key)
                    cell["native_oracles"] = numerical.native_oracle_checks_v1(inner, loaded)
                    q = numerical.numerical_cell_v1(inner, inner.certificate_sha256, cfg["numerical_policy"],
                                                   torch.tensor(a["group_permutation"], dtype=torch.long),
                                                   a["initial_state_sha256_by_dtype"])
                    wanted = [[dtype, c] for dtype in ("float64", "float32") for c in CONTROLS]
                    if q.get("status") != "qualified" or q.get("performed_controls") != wanted or len(q.get("checks", [])) != 10:
                        raise ValueError("All unchanged five-control/dtype checks required")
                    cell.update(status="numerically_qualified", qualification=q)
            except Exception as error:
                cell.update(status="failed", error_type=type(error).__name__, error=str(error))
                result["failed_cells"].append(identity)
                result["cells"].append(cell)
                if phase == "numerical":
                    raise
                continue
            result["cells"].append(cell)
        if len(result["cells"]) != 32 or result["failed_cells"]:
            result["status"] = "failed_complete_inventory_preserved"
        elif phase == "inspect":
            result["status"] = "inspection_complete_pending_root_freeze"
            result["prespecified_TRAIN_numerical_cells"] = bank.select_census_numerical_tiles_v1(
                [{k: c[k] for k in ("parent_id", "split", "tile_id", "num_trees")} for c in result["cells"]])
        else:
            result["status"] = "numerical_complete_no_fitting_or_predictive_claim"
    except Exception as error:
        result.update(status="failed_partial_preserved", error_type=type(error).__name__, error=str(error))
    result["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["verified_file_bytes"] = reader.evidence
    result["sampled_output_bytes_before_receipt"] = output.written
    try:
        receipt = output.write_json("evidence.json", result)
    except Exception as error:
        terminal = {"schema": result["schema"], "phase": phase, "job_id": a["job_id"],
                    "status": "failed_output_cap_partial_products_preserved",
                    "failed_cells": result["failed_cells"], "completed_cell_ids": [
                        f"{c['parent_id']}:{c['tile_id']}" for c in result["cells"]],
                    "error": str(error), "biological_targets_opened": False, "fit_performed": False}
        receipt = output.write_json("failure.json", terminal, terminal=True)
        result["status"] = terminal["status"]
    return result, receipt

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--phase", choices=("inspect", "numerical"))
    p.add_argument("--admission")
    p.add_argument("--expected-admission-sha256")
    args = p.parse_args()
    if not args.execute:
        print(json.dumps(source_report_v1(), indent=2, sort_keys=True))
        return
    if not args.phase or not args.admission or not _sha(args.expected_admission_sha256):
        raise ValueError("Explicit phase and independently supplied active admission SHA required")
    result, receipt = run_phase_v1({"path": args.admission, "sha256": args.expected_admission_sha256}, args.phase)
    print(json.dumps({"status": result["status"], "receipt": receipt}, indent=2))
    if result["status"] not in ("inspection_complete_pending_root_freeze", "numerical_complete_no_fitting_or_predictive_claim"):
        raise SystemExit(1)

if __name__ == "__main__":
    main()
