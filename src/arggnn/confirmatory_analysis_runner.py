"""Admission-first CPU joined analysis. No simulation, prediction, fitting or GPU use."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
from .resources import ROOT
from . import confirmatory_lock as lock_api
from .confirmatory_metrics import logit_metrics, paired_parent_report, holm_adjust
from .dense_tract_metrics import DenseGrid, TractPartition, dense_tract_metrics

SEEDS = [101, 202, 303]
ANALYSIS_SEED = 20261115
REMAINING = ["robustness paired regimes and demographic OOD", "query-count draws and two-workflow Holm family",
             "finite-source and tile sensitivity", "prespecified illustrations"]
CODE_PATHS = {"joined_analysis": "src/arggnn/confirmatory_analysis_runner.py",
              "analysis_cli": "scripts/analyze_confirmatory.py",
              "metrics": "src/arggnn/confirmatory_metrics.py",
              "dense_tract_metrics": "src/arggnn/dense_tract_metrics.py"}


def _numpy_scalar(value):
    if isinstance(value,np.generic):return value.item()
    raise TypeError("Non-JSON analysis value: "+type(value).__name__)


def _canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False,default=_numpy_scalar)+"\n").encode()


def _sha(content):
    return hashlib.sha256(content).hexdigest()


def _pins():
    return {key: {"path": path, "sha256": _sha((ROOT/path).read_bytes())} for key, path in CODE_PATHS.items()}


_IMPORTED_PINS = _pins()

def analysis_schema_values():
    """Prospective schema and current code pins, never a test-data or lock artifact."""
    if _pins()!=_IMPORTED_PINS:raise ValueError("Imported analysis source drift")
    return {"schema_version": 1, "kind": "prespecified_joined_analysis_artifact_schema",
        "status": "frozen_before_test_preparation", "analysis_phase": "id_primary_secondary_and_descriptive",
        "code_artifacts": _pins(),
        "roster": {"kind": "sealed_confirmatory_prediction_roster", "status": "sealed_before_target_join",
            "fields": ["schema_version", "kind", "status", "lock_sha256", "parent_seed_registry_sha256",
                       "admission_ledger_sha256", "cohort", "namespace", "entries"],
            "entry_fields": ["method_id", "seed", "parent_id", "status", "prediction", "checkpoint", "reason"],
            "cell_statuses": ["success", "failed", "unresolved"], "terminal_cell_statuses": ["success", "failed"],
            "matrix": "candidate,strongest_required_baseline,all locked controls x101,202,303 xall admitted ID parents"},
        "admission_ledger": {"kind": "sealed_confirmatory_admission_ledger", "status": "sealed_before_model_scoring",
            "fields": ["schema_version", "kind", "status", "lock_sha256", "parent_seed_registry_sha256",
                       "cohort", "namespace", "target_admitted_parents", "maximum_attempted_parents", "attempts"],
            "attempt_fields": ["parent_id", "status", "reason", "label_blind", "observed_identity",
                               "primary_labels", "census_tracts"],
            "policy": "contiguous registry attempts, first160 admitted, all192 attempted before a shortfall"},
        "observed_identity": {"kind": "sealed_confirmatory_observed_parent_identity",
            "fields": ["schema_version", "kind", "lock_sha256", "namespace", "parent_id", "parent_seeds",
                       "source_artifacts", "query_ids", "diploid_pairs", "primary_grid", "dense_grid",
                       "artificial_tile_edges_bp", "truth_free_observed_inputs"],
            "grids": "DenseGrid.identity_record, original128/4096 midpoint grids in physical bp, coverage clipped"},
        "observed_npz": {"arrays": ["metadata_json","x","edge_index","node_graph","node_ids","clades",
                                   "intervals","query_node_ids","window_bounds"],
            "header_fields": ["schema_version","kind","lock_sha256","namespace","parent_id","feature_names",
                              "time_scale","truth_free_observed_inputs"],
            "kind": "sealed_confirmatory_observed_graph_sequence",
            "feature_names": ["reference_ANA","reference_CHG","reference_WHG","reference_EHG",
                              "is_query_leaf","is_leaf","log_time","log_parent_branch"],
            "time_scale": 10000., "coordinate_units": "intervals/window_bounds float64 global bp/1e6",
            "dtypes": "x/clades float32 or float64; index and ID arrays int64",
            "identity_sources": "exactly one observed-only NPZ; analysis verifies hashes, frozen prediction workflow validates topology/reference adapter"},
        "prediction_npz": {"arrays": ["metadata_json", "primary_logits", "dense_logits", "primary_positions_bp",
                                     "dense_positions_bp", "query_ids"],
            "format": "np.load allow_pickle=False, finite float32/float64 original raw logits, no probabilities"},
        "primary_labels_npz": {"arrays": ["metadata_json", "labels", "positions_bp", "query_ids"],
                              "format": "external integer class-slot labels, exact ordered identity"},
        "census_json": {"fields": ["schema_version", "kind", "lock_sha256", "namespace", "parent_id",
                                   "observed_identity_sha256", "class_order", "query_ids", "domain_bp", "tracts"],
                        "format": "external exact full-domain TractPartition triples in original query order"},
        "policy": {"training_seeds": SEEDS, "analysis_seed": ANALYSIS_SEED, "bootstrap_resamples": 10000,
            "practical_improvement_nats": .010, "target_t_halfwidth_nats": .005, "alpha": .05,
            "calibration_bins": 15, "uncapped_float64_raw_primary_nll": True,
            "mandatory_pre_test_val_temperatures": True, "target_reads_after_complete_sealed_admission": True,
            "target_join_requires_terminal_roster": True,
            "positive_confirmation_requires_complete_predictions_and_target_precision": True},
        "remaining_coverage": REMAINING, "full_protocol_readiness": False}


def _keys(value, names, message):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError(message)


def _integer(value, message, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(message)
    return value


def _digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Lowercase SHA256 required")
    return value


def _no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _decode(content):
    return json.loads(content, object_pairs_hook=_no_duplicates,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


class _Capture:
    """Hash and stat-bound project artifacts; target refs remain unopened until admitted."""
    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        self.files = {}

    def path(self, value):
        if not isinstance(value, (str, Path)) or not str(value) or "\\" in str(value) or ".." in Path(value).parts:
            raise ValueError("Explicit in-project path without traversal required")
        p = Path(value)
        p = (p if p.is_absolute() else self.root/p).resolve(strict=False)
        if p == self.root or not p.is_relative_to(self.root):
            raise ValueError("Artifact escapes project")
        return p

    def ref(self, value):
        _keys(value, ("path", "sha256"), "Exact artifact path/SHA reference required")
        return {"path": str(self.path(value["path"]).relative_to(self.root)), "sha256": _digest(value["sha256"])}

    @staticmethod
    def _stat(path):
        s = path.stat()
        return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)

    def read(self, value, *, json_file=True):
        ref = self.ref(value); path = self.root/ref["path"]
        if str(path) in self.files and self.files[str(path)]["sha256"] != ref["sha256"]:
            raise ValueError("Conflicting artifact hashes")
        before = self._stat(path)
        if before[2] > (64*1024**2 if json_file else 256*1024**2):
            raise ValueError("Artifact exceeds bounded reader size")
        content = path.read_bytes()
        if before != self._stat(path) or _sha(content) != ref["sha256"]:
            raise ValueError("Artifact hash or stat drift")
        self.files[str(path)] = {"path": ref["path"], "sha256": ref["sha256"], "bytes": len(content), "stat": before}
        return _decode(content) if json_file else content

    def hash_file(self, value):
        ref = self.ref(value); path = self.root/ref["path"]; before = self._stat(path)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4*1024**2), b""):
                digest.update(chunk)
        if before != self._stat(path) or digest.hexdigest() != ref["sha256"]:
            raise ValueError("Observed source hash or stat drift")
        prior = self.files.get(str(path))
        if prior is not None and prior["sha256"] != ref["sha256"]:
            raise ValueError("Conflicting source identity")
        self.files[str(path)] = {"path": ref["path"], "sha256": ref["sha256"], "bytes": before[2], "stat": before}

    def check(self):
        for path, record in self.files.items():
            if self._stat(Path(path)) != record["stat"]:
                raise ValueError("Captured analysis artifact drift")

    def provenance(self):
        return [{key: value for key, value in record.items() if key != "stat"} for record in sorted(self.files.values(), key=lambda x: x["path"])]


def _grid(value):
    _keys(value, ("grid_id", "domain_bp", "coverage_bp", "grid_points", "original_indices", "positions_bp", "sha256"),
          "Exact observed grid identity required")
    grid = DenseGrid(**{key: v for key, v in value.items() if key != "sha256"})
    if grid.identity_record() != value:
        raise ValueError("Observed grid fingerprint differs")
    return grid


def _methods(payload):
    result = {"candidate": payload["candidate"], "strongest_required_baseline": payload["strongest_required_baseline"],
              **payload["secondary_controls"]}
    if len(result) != 2+len(payload["secondary_controls"]) or set(payload["secondary_controls"]) not in (
            {"gain_matched_diagonal", "static_local"},
            {"gain_matched_diagonal", "static_local", "matched_scalar_age_full"},
            {"gain_matched_diagonal", "static_local", "noncontextual"}):
        raise ValueError("Exact conditional locked method roster required")
    for bundle in result.values():
        if [row["seed"] for row in bundle["seed_bundles"]] != SEEDS:
            raise ValueError("Exact three locked seed bundles required")
    return result


def _seed_bundle(bundle, seed):
    return next(row for row in bundle["seed_bundles"] if row["seed"] == seed)


def _state(admission):
    return _canonical({"lock": admission.lock.payload, "schema": admission.schema,
        "registry": admission.registry, "temperatures": admission.temperatures,
        "ledger": admission.ledger, "roster": admission.roster, "identities": admission.identities,
        "parents": admission.parents, "entries": [list(key)+[value] for key,value in sorted(admission.entries.items())],
        "roster_ref": admission.roster_ref, "ledger_ref": admission.ledger_ref})


@dataclass
class AnalysisAdmission:
    lock: object
    capture: _Capture
    schema: dict
    registry: dict
    temperatures: dict
    ledger: dict
    roster: dict
    identities: dict
    methods: dict
    parents: list
    entries: dict
    roster_ref: dict
    ledger_ref: dict
    content: bytes = b""

    def check(self):
        if _state(self) != self.content or self.methods != _methods(self.lock.payload):
            raise ValueError("In-memory analysis admission drift")
        self.lock.check()
        if self.schema["code_artifacts"] != _pins() or _pins()!=_IMPORTED_PINS:
            raise ValueError("Analysis implementation drift")
        self.capture.check()


def admit_analysis(lock_path, expected_lock_sha256, roster_path, expected_roster_sha256,
                   ledger_path, expected_ledger_sha256, *, project_root=ROOT):
    """Admit metadata and observed identities, without opening target or prediction arrays."""
    capture = _Capture(project_root)
    for path in (lock_path, roster_path, ledger_path):
        capture.path(path)
    validator = getattr(lock_api, "validate_confirmatory_lock", None)
    if validator is None:
        raise ValueError("Public frozen-lock validator is unavailable")
    frozen = validator(lock_path, expected_lock_sha256, project_root=project_root)
    if frozen.sha256 != _digest(expected_lock_sha256) or frozen.payload.get("kind") != "confirmatory_provenance_lock":
        raise ValueError("A validated actual frozen lock is required")
    payload = frozen.payload
    if payload.get("lock_execution_scope") != "id_confirmation_only":
        raise ValueError("This runner accepts only the frozen bounded ID phase")
    if payload.get("training_seeds") != SEEDS:
        raise ValueError("Prescribed training seeds required")
    for name, version in payload["analysis_environment_versions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Frozen analysis environment differs")
    refs = payload["bound_references"]
    schema = capture.read(refs["analysis_schema"])
    if schema != analysis_schema_values():
        raise ValueError("Exact pre-test artifact schema/current code pins required")
    for ref in schema["code_artifacts"].values():
        capture.hash_file(ref)
    code = payload["code_artifacts"]["joined_analysis"]
    if {key: code[key] for key in ("path", "sha256")} not in (schema["code_artifacts"]["joined_analysis"], schema["code_artifacts"]["analysis_cli"]):
        raise ValueError("Frozen joined-analysis entry source differs")
    registry = capture.read(refs["parent_seed_registry"])
    protocol = capture.read(refs["protocol_json"])
    contract = capture.read(refs["analysis_contract"])
    if (contract["confirmation"] != protocol["confirmation"] or
            contract["separate_prespecified_analyses"] != protocol["separate_prespecified_analyses"] or
            registry["analysis_seed"] != ANALYSIS_SEED or registry["training_seeds"] != SEEDS):
        raise ValueError("Exact registry/analysis prespecification required")
    if (protocol["confirmation"]["target_admitted_parents"] != 160 or
            protocol["confirmation"]["max_attempted_parents"] != 192 or
            protocol["confirmation"]["primary"]["practical_difference_nats"] != .010 or
            protocol["confirmation"]["primary"]["target_95_ci_halfwidth_nats"] != .005):
        raise ValueError("This bounded runner implements the frozen v1 ID policy")
    cohort = registry["cohorts"]["id_confirmation"]
    if len(cohort["parents"]) != 192 or [row["parent_id"] for row in cohort["parents"]] != list(range(192)):
        raise ValueError("Full prespecified ID attempt registry required")
    methods = _methods(payload)
    temperatures = capture.read(refs["temperature_manifest"])
    if (temperatures.get("kind") != "prespecified_validation_temperature_manifest" or
            temperatures.get("status") != "frozen_before_test_preparation" or
            temperatures.get("training_seeds") != SEEDS or temperatures.get("final_test_access") is not False):
        raise ValueError("Pre-test frozen VAL temperatures required")
    temp_keys = set()
    for row in temperatures["records"]:
        key = (row["method_id"], _integer(row["seed"], "Integer temperature seed required"))
        if key in temp_keys or key[0] not in methods or key[1] not in SEEDS:
            raise ValueError("Exact unique frozen method/seed temperatures required")
        temp_keys.add(key); selected = _seed_bundle(methods[key[0]], key[1]); t = row["temperature"]
        if (isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or t <= 0 or
                row["fit_scope"] != "fresh_development_validation_only" or
                row["checkpoint"] != selected["checkpoint"] or row["development_prediction"] != selected["development_prediction"]):
            raise ValueError("Temperature provenance or positive value differs")
    if temp_keys != {(method, seed) for method in methods for seed in SEEDS}:
        raise ValueError("All selected method/seed temperatures required")
    ledger_ref = {"path": str(ledger_path), "sha256": expected_ledger_sha256}
    roster_ref = {"path": str(roster_path), "sha256": expected_roster_sha256}
    ledger = capture.read(ledger_ref); roster = capture.read(roster_ref)
    _keys(ledger, schema["admission_ledger"]["fields"], "Exact sealed admission ledger required")
    _keys(roster, schema["roster"]["fields"], "Exact sealed prediction roster required")
    for record, specification in ((ledger, schema["admission_ledger"]), (roster, schema["roster"])):
        if (type(record["schema_version"]) is not int or record["schema_version"] != 1 or
                record["kind"] != specification["kind"] or record["status"] != specification["status"] or
                record["lock_sha256"] != frozen.sha256 or record["cohort"] != "id_confirmation" or
                record["namespace"] != cohort["namespace"] or
                record["parent_seed_registry_sha256"] != refs["parent_seed_registry"]["sha256"]):
            raise ValueError("Roster/ledger lock, cohort, registry or seal differs")
    if roster["admission_ledger_sha256"] != expected_ledger_sha256:
        raise ValueError("Prediction seal must bind exact admission ledger")
    if ledger["target_admitted_parents"] != 160 or ledger["maximum_attempted_parents"] != 192:
        raise ValueError("Fixed admission budget differs")
    attempts = ledger["attempts"]
    if not isinstance(attempts, list) or not attempts or len(attempts) > 192:
        raise ValueError("Nonempty bounded attempt ledger required")
    parents = []; by_parent = {}
    for index, row in enumerate(attempts):
        _keys(row, schema["admission_ledger"]["attempt_fields"], "Exact admission attempt fields required")
        if (type(row["parent_id"]) is not int or row["parent_id"] != index or row["label_blind"] is not True or
                row["status"] not in ("admitted", "rejected") or not isinstance(row["reason"], str) or not row["reason"]):
            raise ValueError("Ordered label-blind admission attempt and reason required")
        if row["status"] == "admitted":
            if len(parents) == 160:
                raise ValueError("Cannot continue admission after the first160 admitted parents")
            parents.append(index); by_parent[index] = row
            for name in ("observed_identity", "primary_labels", "census_tracts"):
                capture.ref(row[name])
        elif any(row[name] is not None for name in ("observed_identity", "primary_labels", "census_tracts")):
            raise ValueError("Rejected parent cannot declare joined target artifacts")
    if not parents or (len(parents) < 160 and len(attempts) != 192) or (len(parents) == 160 and attempts[-1]["status"] != "admitted"):
        raise ValueError("Admission must reach160 or exhaust all192 attempts; never stop by outcomes")
    entries = {}
    expected = {(method, seed, parent) for method in methods for seed in SEEDS for parent in parents}
    for row in roster["entries"]:
        _keys(row, schema["roster"]["entry_fields"], "Exact prediction cell fields required")
        key = (row["method_id"], _integer(row["seed"], "Integer prediction seed required"),
               _integer(row["parent_id"], "Integer prediction parent required"))
        if key not in expected or key in entries or row["status"] not in ("success", "failed", "unresolved"):
            raise ValueError("Duplicate, extra or unknown prediction cell")
        if row["checkpoint"] != _seed_bundle(methods[key[0]], key[1])["checkpoint"]:
            raise ValueError("No checkpoint/baseline replacement is allowed")
        if not isinstance(row["reason"], str) or (row["status"] != "success" and not row["reason"]):
            raise ValueError("Preserved failure/unresolved reason required")
        if row["status"] == "success":
            capture.ref(row["prediction"])
        elif row["prediction"] is not None:
            raise ValueError("Failed/unresolved cells cannot substitute predictions")
        entries[key] = row
    if set(entries) != expected:
        raise ValueError("Complete sealed method/seed/parent roster required before target access")
    targets = {capture.ref(row[name])["path"] for row in by_parent.values() for name in ("primary_labels", "census_tracts")}
    predictions = {capture.ref(row["prediction"])["path"] for row in entries.values() if row["status"] == "success"}
    identities = {}
    for parent, row in by_parent.items():
        if capture.ref(row["observed_identity"])["path"] in targets | predictions:
            raise ValueError("Target/prediction cannot substitute observed identity metadata")
        identity = capture.read(row["observed_identity"])
        _keys(identity, schema["observed_identity"]["fields"], "Exact observed identity fields required")
        registry_row = cohort["parents"][parent]
        if (type(identity["schema_version"]) is not int or identity["schema_version"] != 1 or
                type(identity["parent_id"]) is not int or identity["kind"] != schema["observed_identity"]["kind"] or
                identity["parent_id"] != parent or identity["namespace"] != cohort["namespace"] or
                identity["lock_sha256"] != frozen.sha256 or identity["truth_free_observed_inputs"] is not True or
                _canonical(identity["parent_seeds"]) != _canonical({key: registry_row[key] for key in ("ancestry_seed", "mutation_seed", "inference_seed")})):
            raise ValueError("Observed parent source/registry identity differs")
        primary = _grid(identity["primary_grid"]); dense = _grid(identity["dense_grid"])
        domain = [0., float(protocol["common_generation"]["sequence_length_bp"])]
        if (list(primary.domain_bp) != domain or primary.domain_bp != dense.domain_bp or
                primary.coverage_bp != dense.coverage_bp or primary.grid_points != 128 or dense.grid_points != 4096 or
                len(primary.positions_bp) < 120 or (primary.coverage_bp[1]-primary.coverage_bp[0])/domain[1] < .95):
            raise ValueError("Original128/4096 grids and frozen coverage admission required")
        queries = identity["query_ids"]; pairs = np.asarray(identity["diploid_pairs"])
        if (not isinstance(queries, list) or len(queries) != 64 or
                any(type(q) is not int or q < 0 for q in queries) or len(set(queries)) != 64 or
                pairs.shape != (32, 2) or not np.issubdtype(pairs.dtype, np.integer) or
                sorted(pairs.ravel().tolist()) != list(range(64))):
            raise ValueError("Explicit64 observed query IDs and32 nonoverlapping diploid index pairs required")
        edges = np.asarray(identity["artificial_tile_edges_bp"], dtype=np.float64)
        if edges.shape != (7,) or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0) or np.any(edges <= 0) or np.any(edges >= domain[1]):
            raise ValueError("Explicit artificial eight-tile edge identity required")
        if (not isinstance(identity["source_artifacts"], list) or len(identity["source_artifacts"])!=1 or
                Path(capture.ref(identity["source_artifacts"][0])["path"]).suffix!=".npz"):
            raise ValueError("Exactly one observed-only NPZ source required")
        for source in identity["source_artifacts"]:
            if capture.ref(source)["path"] in targets | predictions:
                raise ValueError("Targets/predictions cannot serve as observed input source")
            capture.hash_file(source)
        identities[parent] = identity
    admission = AnalysisAdmission(frozen, capture, schema, registry, temperatures, ledger, roster, identities,
                                  methods, parents, entries, capture.ref(roster_ref), capture.ref(ledger_ref))
    admission.content = _state(admission); admission.check()
    return admission


def _npz(capture, ref, names):
    body = capture.read(ref, json_file=False)
    with np.load(io.BytesIO(body), allow_pickle=False) as data:
        if set(data.files) != set(names):
            raise ValueError("Exact safe NPZ array schema required")
        arrays = {name: data[name] for name in names}
    metadata = arrays.pop("metadata_json")
    if metadata.shape != () or metadata.dtype.kind not in ("U", "S"):
        raise ValueError("Scalar JSON-string NPZ metadata required")
    text = metadata.item()
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    return _decode(text), arrays


def _prediction(admission, method, seed, parent):
    entry = admission.entries[(method, seed, parent)]; identity = admission.identities[parent]
    header, arrays = _npz(admission.capture, entry["prediction"], admission.schema["prediction_npz"]["arrays"])
    bundle = admission.methods[method]
    expected = {"schema_version": 1, "kind": "sealed_confirmatory_raw_prediction",
        "lock_sha256": admission.lock.sha256, "namespace": admission.roster["namespace"],
        "parent_id": parent, "method_id": method, "training_seed": seed,
        "configuration_id": bundle["configuration_id"], "family_id": bundle["family_id"],
        "protocol": bundle["protocol"], "checkpoint": entry["checkpoint"],
        "observed_identity_sha256": next(row["observed_identity"]["sha256"] for row in admission.ledger["attempts"] if row["parent_id"] == parent),
        "primary_grid_sha256": identity["primary_grid"]["sha256"], "dense_grid_sha256": identity["dense_grid"]["sha256"],
        "class_order": ["ANA", "CHG", "WHG", "EHG"], "no_gradients_or_optimization": True}
    if _canonical(header) != _canonical(expected):
        raise ValueError("Prediction header source/checkpoint/method/grid identity differs")
    if arrays["query_ids"].dtype.kind not in ("i", "u") or arrays["query_ids"].tolist() != identity["query_ids"]:
        raise ValueError("Prediction original query order differs")
    for prefix in ("primary", "dense"):
        positions = arrays[prefix+"_positions_bp"]; logits = arrays[prefix+"_logits"]
        if (positions.dtype != np.dtype("float64") or not np.array_equal(positions, identity[prefix+"_grid"]["positions_bp"]) or
                logits.dtype not in (np.dtype("float32"), np.dtype("float64")) or
                logits.shape != (len(positions), 64, 4) or not np.isfinite(logits).all()):
            raise ValueError("Finite original raw-logit/coordinate array contract differs")
    return arrays


def _targets(admission, parent):
    row = next(row for row in admission.ledger["attempts"] if row["parent_id"] == parent)
    identity = admission.identities[parent]
    header, arrays = _npz(admission.capture, row["primary_labels"], admission.schema["primary_labels_npz"]["arrays"])
    expected = {"schema_version": 1, "kind": "sealed_external_primary_labels", "lock_sha256": admission.lock.sha256,
        "namespace": admission.roster["namespace"], "parent_id": parent,
        "observed_identity_sha256": row["observed_identity"]["sha256"],
        "primary_grid_sha256": identity["primary_grid"]["sha256"], "class_order": ["ANA", "CHG", "WHG", "EHG"]}
    y = arrays["labels"]
    if (_canonical(header) != _canonical(expected) or arrays["positions_bp"].dtype != np.dtype("float64") or
            not np.array_equal(arrays["positions_bp"], identity["primary_grid"]["positions_bp"]) or
            arrays["query_ids"].dtype.kind not in ("i", "u") or arrays["query_ids"].tolist() != identity["query_ids"] or
            y.shape != (len(identity["primary_grid"]["positions_bp"]), 64) or
            y.dtype.kind not in ("i", "u") or np.any(y < 0) or np.any(y >= 4)):
        raise ValueError("External primary labels/coordinate/query identity differs")
    census = admission.capture.read(row["census_tracts"])
    _keys(census, admission.schema["census_json"]["fields"], "Exact external census fields required")
    for key in ("schema_version", "lock_sha256", "namespace", "parent_id", "observed_identity_sha256", "class_order"):
        if _canonical(census[key]) != _canonical(expected[key]):
            raise ValueError("Census target source identity differs")
    if (census["kind"] != "sealed_external_census_tract_partitions" or census["query_ids"] != identity["query_ids"] or
            census["domain_bp"] != identity["primary_grid"]["domain_bp"]):
        raise ValueError("Census query/domain contract differs")
    partition = TractPartition(tuple(census["query_ids"]), tuple(census["domain_bp"]), 4, tuple(tuple(row) for row in census["tracts"]))
    # Independently extracted primary labels and census tracts must agree at exact readout coordinates.
    positions = np.asarray(identity["primary_grid"]["positions_bp"])
    for index, tracts in enumerate(partition.tracts):
        boundaries = np.asarray([tract.right_bp for tract in tracts])
        slots = np.asarray([tract.class_slot for tract in tracts])
        census_y = slots[np.searchsorted(boundaries, positions, side="right")]
        if not np.array_equal(census_y, y[:, index]):
            raise ValueError("External primary labels disagree with exact census partition")
    return y, partition


def _ece(metrics):
    return math.fsum(row["observations"]*abs(row["mean_confidence"]-row["accuracy"])
                     for row in metrics["calibration"] if row["observations"])/metrics["observations"]


def _interpret(report, *, complete, target):
    if report is None:
        return {"robust_positive_confirmation": False, "reason": "Fewer than two complete paired parents",
                "statistical_favorability": False, "practical_advantage_established": False,
                "precision_shortfall": True}
    upper = report["t_interval95"][1]; lower = report["t_interval95"][0]
    precision = bool(report["independent_parents"] < target or report["achieved_t_interval_halfwidth"] > .005)
    favorable = bool(upper < 0); agreement = report["t_upper_below_zero"] == report["bootstrap_upper_below_zero"]
    return {"statistical_favorability": favorable, "practical_advantage_established": bool(upper < -.010),
        "promising_but_practical_precision_insufficient": bool(report["difference_candidate_minus_baseline"] <= -.010 and -.010 <= upper < 0),
        "no_material_advantage_supported": bool(lower > -.010), "equivalence_established": False,
        "inconclusive": bool(lower <= 0 <= upper or precision), "precision_shortfall": precision,
        "t_bootstrap_sign_agreement": bool(agreement),
        "interpretation_scope": "full_admitted_roster" if complete else "descriptive_complete_case_sensitivity",
        "robust_positive_confirmation": bool(complete and not precision and favorable and report["bootstrap_upper_below_zero"]),
        "reason": "Complete fixed roster, achieved precision and both intervals required for robust positive confirmation",
        "thresholds_nats": {"practical_improvement": .010, "target_halfwidth": .005}}


def _paired(admission, losses, baseline):
    parents = [parent for parent in admission.parents if all((method, seed, parent) in losses for method in ("candidate", baseline) for seed in SEEDS)]
    if len(parents) < 2:
        return parents, None
    candidate = [[losses[("candidate", seed, parent)] for seed in SEEDS] for parent in parents]
    comparator = [[losses[(baseline, seed, parent)] for seed in SEEDS] for parent in parents]
    report = paired_parent_report(candidate, comparator, parent_ids=parents, training_seeds=SEEDS,
        bootstrap_seed=ANALYSIS_SEED, bootstrap_resamples=10000, practical_improvement=.010)
    return parents, report


def _publish(output, content, check):
    pending = None; identity = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=output.parent, prefix=".analysis_pending_", delete=False) as handle:
            pending = Path(handle.name); stat = os.fstat(handle.fileno()); identity = (stat.st_dev, stat.st_ino)
            handle.write(content); handle.flush(); os.fsync(handle.fileno())
        check(); stat = pending.lstat()
        if (stat.st_dev, stat.st_ino) != identity:
            raise ValueError("Owned analysis output inode changed")
        os.link(pending, output, follow_symlinks=False)
    finally:
        if pending is not None and identity is not None:
            try:
                stat = pending.lstat()
            except FileNotFoundError:
                pass
            else:
                if (stat.st_dev, stat.st_ino) == identity:
                    pending.unlink()


def analyze_confirmatory(admission, *, output_path=None):
    """One deterministic joined ID phase, preserving every absent/failing cell and parent."""
    if not isinstance(admission, AnalysisAdmission):
        raise ValueError("Actual frozen-lock/roster/ledger admission required")
    admission.check()
    if any(row["status"]=="unresolved" for row in admission.entries.values()):
        raise ValueError("Complete terminal prediction roster required before any target join; unresolved cells remain in sealed roster")
    output = None if output_path is None else admission.capture.path(output_path)
    if output is not None and (output.exists() or not output.parent.is_dir()):
        raise ValueError("Fresh exclusive analysis output with existing parent required")
    # All success artifacts are CPU-validated and hash-sealed before opening any target.
    for key, entry in admission.entries.items():
        if entry["status"] == "success":
            _prediction(admission, *key)
    admission.check()
    losses = {}; diagnostics = []; failures = []
    temperatures = {(row["method_id"], row["seed"]): row["temperature"] for row in admission.temperatures["records"]}
    for parent in admission.parents:
        successful = [key for key, entry in admission.entries.items() if key[2] == parent and entry["status"] == "success"]
        if successful:
            labels, census = _targets(admission, parent)
        for method, seed, _ in [key for key in admission.entries if key[2] == parent]:
            entry = admission.entries[(method, seed, parent)]
            if entry["status"] != "success":
                failures.append(dict(entry)); continue
            arrays = _prediction(admission, method, seed, parent); identity = admission.identities[parent]
            raw = logit_metrics(arrays["primary_logits"], labels, bins=15, diploid_pairs=identity["diploid_pairs"])
            raw["top_label_ece"] = _ece(raw); losses[(method, seed, parent)] = raw["nll"]
            calibrated = logit_metrics(arrays["primary_logits"].astype(np.float64)/temperatures[(method, seed)],
                labels, bins=15, diploid_pairs=identity["diploid_pairs"])
            calibrated["top_label_ece"] = _ece(calibrated)
            calibrated["nll_policy"] = "Descriptive float64 logits divided by pre-test VAL-frozen temperature"
            dense = dense_tract_metrics(arrays["dense_logits"], query_ids=identity["query_ids"],
                grid=_grid(identity["dense_grid"]), true_partitions=census,
                artificial_tile_edges_bp=identity["artificial_tile_edges_bp"],
                boundary_tolerance_bp=1000., short_tract_cutoff_bp=10000., tile_edge_radius_bp=10000.)
            diagnostics.append({"method_id": method, "seed": seed, "parent_id": parent, "raw_primary": raw,
                "validation_temperature": temperatures[(method, seed)], "val_temperature_primary": calibrated,
                "raw_dense_tracts": dense})
    all_success = not failures
    primary_parents, primary = _paired(admission, losses, "strongest_required_baseline")
    primary_complete = primary_parents == admission.parents
    secondary = []; p_values = []
    for control in admission.lock.payload["secondary_controls"]:
        parents, report = _paired(admission, losses, control); complete = parents == admission.parents
        p_values.append(report["two_sided_t_pvalue"] if complete and report is not None else 1.)
        secondary.append({"control_id": control, "all_admitted_parents_complete": complete,
                          "complete_case_parent_ids": parents, "report": report,
                          "inference_eligible": all_success and complete and report is not None})
    for record, adjusted in zip(secondary, holm_adjust(p_values)):
        record["holm_two_sided_pvalue"] = adjusted
        record["holm_reject_alpha005"] = bool(record["inference_eligible"] and adjusted < .05)
        record["favorable_adjusted_result"] = bool(record["holm_reject_alpha005"] and record["report"]["t_upper_below_zero"])
    interpretation = _interpret(primary, complete=all_success and primary_complete, target=160)
    result = {"schema_version": 1, "kind": "joined_confirmatory_id_analysis_report",
        "status": "id_phase_analyzed" if all_success else "blocked_positive_confirmation_with_descriptive_complete_cases",
        "analysis_phase": admission.schema["analysis_phase"], "lock_sha256": admission.lock.sha256,
        "roster_sha256": admission.roster_ref["sha256"], "ledger_sha256": admission.ledger_ref["sha256"],
        "namespace": admission.roster["namespace"], "training_seeds": SEEDS, "analysis_seed": ANALYSIS_SEED,
        "admitted_parent_ids": admission.parents, "attempted_parent_ids": [row["parent_id"] for row in admission.ledger["attempts"]],
        "admission_attempts": admission.ledger["attempts"], "admission_probability": len(admission.parents)/len(admission.ledger["attempts"]),
        "target_admitted_parents": 160, "admission_target_shortfall": len(admission.parents) < 160,
        "all_predictions_successful": all_success, "prediction_failures_or_unresolved": failures,
        "method_seed_failure_rates": [{"method_id": method, "seed": seed,
            "failed_or_unresolved_parents": sum(admission.entries[(method, seed, parent)]["status"] != "success" for parent in admission.parents),
            "denominator_admitted_parents": len(admission.parents),
            "failure_rate": sum(admission.entries[(method, seed, parent)]["status"] != "success" for parent in admission.parents)/len(admission.parents)}
            for method in admission.methods for seed in SEEDS],
        "method_parent_seed_nll": [{"method_id": method, "parent_id": parent,
            "training_seeds": SEEDS, "nll": [losses.get((method, seed, parent)) for seed in SEEDS],
            "mean_three_seed_nll": math.fsum(losses[(method, seed, parent)] for seed in SEEDS)/3 if all((method, seed, parent) in losses for seed in SEEDS) else None}
            for method in admission.methods for parent in admission.parents],
        "primary": {"candidate": admission.lock.payload["candidate"]["choice"],
            "strongest_required_baseline": admission.lock.payload["strongest_required_baseline"]["choice"],
            "all_admitted_parents_complete": primary_complete, "complete_case_parent_ids": primary_parents,
            "omitted_from_complete_case_sensitivity": [p for p in admission.parents if p not in primary_parents],
            "report": primary, "interpretation": interpretation,
            "scope": "Primary inference only on full frozen admitted roster. Incomplete cases are descriptive sensitivity."},
        "conditional_secondary_holm": secondary, "conditional_holm_family_size": len(secondary),
        "descriptive_parent_seed_diagnostics": diagnostics,
        "captured_artifacts": admission.capture.provenance(), "code_artifacts": admission.schema["code_artifacts"],
        "remaining_coverage": REMAINING, "full_protocol_readiness": False,
        "primary_selection_changed": False, "authorizes_generation_or_refitting": False,
        "scope": "ID primary/conditional and raw,VAL-temperature,dense descriptive phase. Statistical output is conditional on frozen development and checkpoints; no novelty or general seed-population claim."}
    admission.check(); content = _canonical(result)
    if output is not None:
        _publish(output, content, admission.check)
    return {"values": result, "bytes": content, "sha256": _sha(content)}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True); parser.add_argument("--expected-lock-sha256", required=True)
    parser.add_argument("--roster", required=True); parser.add_argument("--expected-roster-sha256", required=True)
    parser.add_argument("--admission-ledger", required=True); parser.add_argument("--expected-admission-ledger-sha256", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    admitted = admit_analysis(args.lock, args.expected_lock_sha256, args.roster, args.expected_roster_sha256,
        args.admission_ledger, args.expected_admission_ledger_sha256)
    result = analyze_confirmatory(admitted, output_path=args.output)
    print(json.dumps({"event": "joined_id_analysis", "sha256": result["sha256"],
        "all_predictions_successful": result["values"]["all_predictions_successful"],
        "full_protocol_readiness": False, "output_written": args.output is not None}), flush=True)


if __name__ == "__main__":
    main()
