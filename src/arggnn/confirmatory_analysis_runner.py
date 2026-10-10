"""Admission-first CPU joined analysis. No simulation, prediction, fitting or GPU use."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import io
import json
import math
import os
import stat
import uuid
from pathlib import Path
import tempfile

import numpy as np
from .resources import ROOT, ALLOWED_GPUS, PROCESS_TITLE
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

def _terminal_sources():
    return {key:{"path":path,"sha256":_sha((ROOT/path).read_bytes())} for key,path in {
        "prediction":"src/arggnn/confirmatory_prediction_runner.py","prediction_cli":"scripts/predict_confirmatory.py"}.items()}


def analysis_schema_values():
    """Prospective schema and current code pins, never a test-data or lock artifact."""
    if _pins()!=_IMPORTED_PINS:raise ValueError("Imported analysis source drift")
    return {"schema_version": 1, "kind": "prespecified_joined_analysis_artifact_schema",
        "status": "frozen_before_test_preparation", "analysis_phase": "id_primary_secondary_and_descriptive",
        "code_artifacts": _pins(),
        "roster": {"kind": "sealed_confirmatory_prediction_roster", "status": "sealed_before_target_join",
            "fields": ["schema_version", "kind", "status", "lock_sha256", "parent_seed_registry_sha256",
                       "admission_ledger_sha256", "cohort", "namespace", "entries"],
            "entry_fields": ["method_id", "seed", "parent_id", "status", "prediction", "checkpoint", "reason", "terminal_receipt"],
            "cell_statuses": ["success", "failed", "unresolved"], "terminal_cell_statuses": ["success", "failed"],
            "matrix": "candidate,strongest_required_baseline,all locked controls x101,202,303 xall admitted ID parents"},
        "terminal_receipt": {"kind":"confirmatory_prediction_cell_receipt","source_artifacts":_terminal_sources(),
            "terminal_statuses":["success","failed"],"unresolved_receipt":None,
            "success_requires":{"terminal":True,"cpu_validation_only":False,"cuda_execution_performed":True,
                "target_artifacts_opened":False,"no_gradients_or_optimization":True},
            "runtime":"complete CUDA initialization; exact frozen software/hardware/strictpolicy; allowed physicalGPU/title; canonical runtime=visible=launcherUUID",
            "failed":"actual terminal failure receipt, exact cell/ledger/lock/source, null scored prediction, preserved error; never positive confirmation"},
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
            "grids": "DenseGrid.identity_record, original128/4096 midpoint grids in physical bp, coverage clipped",
            "artificial_tile_edges_bp":[125000.*i for i in range(1,8)]},
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
        "roster_ref": admission.roster_ref, "ledger_ref": admission.ledger_ref, "terminal_receipts":admission.terminal_receipts})


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
    terminal_receipts: dict
    content: bytes = b""

    def check(self):
        if _state(self) != self.content or self.methods != _methods(self.lock.payload):
            raise ValueError("In-memory analysis admission drift")
        self.lock.check()
        if self.schema!=analysis_schema_values() or _pins()!=_IMPORTED_PINS:
            raise ValueError("Analysis implementation drift")
        self.capture.check()


def _expected_cell(frozen_lock,method_id,seed,parent_id,identity,identity_ref):
    methods=_methods(frozen_lock.payload)
    if method_id not in methods or type(seed) is not int or seed not in SEEDS or type(parent_id) is not int:
        raise ValueError("Exact locked terminal cell identifiers required")
    bundle=methods[method_id];selected=_seed_bundle(bundle,seed)
    return {"schema_version":1,"kind":"sealed_confirmatory_raw_prediction","lock_sha256":frozen_lock.sha256,
        "namespace":identity["namespace"],"parent_id":parent_id,"method_id":method_id,"training_seed":seed,
        "configuration_id":bundle["configuration_id"],"family_id":bundle["family_id"],
        "protocol":bundle["protocol"],"checkpoint":selected["checkpoint"],
        "observed_identity_sha256":identity_ref["sha256"],"primary_grid_sha256":identity["primary_grid"]["sha256"],
        "dense_grid_sha256":identity["dense_grid"]["sha256"],"class_order":["ANA","CHG","WHG","EHG"],
        "no_gradients_or_optimization":True}


def _cuda_uuid(value):
    if not isinstance(value,str):raise ValueError("Actual CUDA UUID string required")
    plain=value[4:] if value.startswith("GPU-") else value
    try:normalized=str(uuid.UUID(plain))
    except (ValueError,AttributeError) as error:raise ValueError("Malformed actual CUDA UUID") from error
    if normalized!=plain.lower():raise ValueError("Canonical actual CUDA UUID required")
    return normalized


def _qualified_runtime(runtime,payload,capture):
    if not isinstance(runtime,dict) or runtime.get("device_kind")!="cuda" or runtime.get("initialization_stage")!="complete":
        raise ValueError("Complete actual CUDA runtime required; CPU validation cannot qualify")
    frozen=payload["captured_runtime"]
    if runtime.get("process_title")!=PROCESS_TITLE or frozen.get("process_title")!=PROCESS_TITLE:
        raise ValueError("Required operational research process title differs")
    lock_api._same_policy(runtime.get("deterministic_execution_policy"))
    if _canonical(runtime["deterministic_execution_policy"])!=_canonical(payload["numerical_execution_policy"]):
        raise ValueError("Frozen strict runtime policy differs")
    software=runtime.get("software")
    if not isinstance(software,dict) or any(not isinstance(software.get(k),str) or not software[k].strip()
                                           for k in ("python","python_executable","torch","torch_cuda","torch_build")):
        raise ValueError("Actual CUDA software/build evidence required")
    if any(software.get(k)!=frozen["software"].get(k) for k in ("python","python_executable","torch","torch_cuda","torch_build")):
        raise ValueError("CUDA terminal software differs from frozen stack")
    device=runtime.get("device")
    if (not isinstance(device,dict) or type(device.get("logical_index")) is not int or device["logical_index"]!=0 or
        type(device.get("total_memory_bytes")) is not int or device["total_memory_bytes"]<=0 or
        any(device.get(k)!=frozen["device"].get(k) for k in ("name","compute_capability"))):
        raise ValueError("CUDA terminal hardware differs from frozen stack")
    physical=runtime.get("physical_gpu")
    if type(physical) is not int or physical not in ALLOWED_GPUS:
        raise ValueError("Allowed physical GPU2..5 required")
    if not isinstance(runtime.get("visible_gpu_uuid"),str) or not runtime["visible_gpu_uuid"].startswith("GPU-"):
        raise ValueError("Actual GPU-prefixed visible UUID required")
    actual=_cuda_uuid(device.get("runtime_uuid"));visible=_cuda_uuid(runtime.get("visible_gpu_uuid"))
    if actual!=visible or device.get("canonical_runtime_uuid")!=actual or runtime.get("canonical_visible_gpu_uuid")!=actual:
        raise ValueError("Actual runtime and visible CUDA UUID evidence differs")
    launcher=runtime.get("launcher_metadata")
    if (not isinstance(launcher,dict) or type(launcher.get("gpu")) is not int or launcher["gpu"]!=physical or
        _cuda_uuid(launcher.get("uuid"))!=actual or not isinstance(launcher.get("job"),str) or not launcher["job"]):
        raise ValueError("Actual launcher GPU/UUID/job evidence required")
    expected=capture.root/".runtime/locks"/f"gpu_{physical}.lock"
    if runtime.get("lease_path")!=str(expected) or type(runtime.get("lease_fd")) is not int or runtime["lease_fd"]<0:
        raise ValueError("Archived operational launcher lease identity differs")


def validate_prediction_terminal(capture,terminal_ref,*,frozen_lock,method_id,seed,parent_id,
                                observed_identity,observed_identity_ref,admission_ledger_ref,
                                expected_status=None,expected_prediction=None):
    """Metadata-only terminal qualification under an already validated lock; no targets/devices."""
    identity_ref=capture.ref(observed_identity_ref);ledger_ref=capture.ref(admission_ledger_ref)
    ref=capture.ref(terminal_ref);receipt=capture.read(ref)
    base={"schema_version","kind","lock_sha256","admission_ledger","cell","created_utc",
          "no_gradients_or_optimization","target_artifacts_opened","raw_primary_dense_separate_forward_passes",
          "imported_source_sha256","status","terminal","phase","runtime","prediction","reason","cpu_validation_only"}
    if not isinstance(receipt,dict) or receipt.get("status") not in ("success","failed"):
        raise ValueError("Actual success/failed terminal prediction receipt required")
    status=receipt["status"]
    fields=base|({"captured_artifacts","cuda_execution_performed","cuda_allocator_peak_bytes"} if status=="success"
                 else {"orphan_prediction","error_type"})
    # Genuine CPU success has no allocator record and is rejected explicitly before the exact CUDA field check.
    if status=="success" and (receipt.get("cpu_validation_only") is not False or receipt.get("cuda_execution_performed") is not True):
        raise ValueError("CPU-validation prediction cannot qualify for joined analysis")
    _keys(receipt,fields,"Exact operational terminal receipt fields required")
    if (type(receipt["schema_version"]) is not int or receipt["schema_version"]!=1 or
        receipt["kind"]!="confirmatory_prediction_cell_receipt" or receipt["terminal"] is not True or
        receipt["lock_sha256"]!=frozen_lock.sha256 or capture.ref(receipt["admission_ledger"])!=ledger_ref or
        receipt["no_gradients_or_optimization"] is not True or receipt["target_artifacts_opened"] is not False or
        receipt["raw_primary_dense_separate_forward_passes"] is not True or
        not isinstance(receipt["created_utc"],str) or not receipt["created_utc"] or
        _canonical(receipt["cell"])!=_canonical(_expected_cell(frozen_lock,method_id,seed,parent_id,observed_identity,identity_ref))):
        raise ValueError("Terminal lock/ledger/source/checkpoint/cell identity differs")
    if expected_status is not None and expected_status!=status:raise ValueError("Roster/terminal status differs")
    sources=receipt["imported_source_sha256"];required={value["path"]:value["sha256"] for value in _terminal_sources().values()}
    required.update({path:digest for path,digest in frozen_lock.payload["binding_implementation_sha256"].items()
                     if path.startswith("src/arggnn/")})
    if not isinstance(sources,dict) or not sources or any(sources.get(path)!=digest for path,digest in required.items()):
        raise ValueError("Actual frozen prediction/source implementation evidence required")
    workflow=frozen_lock.payload["code_artifacts"]["prediction"]
    if sources.get(workflow["path"])!=workflow["sha256"]:raise ValueError("Frozen prediction workflow differs")
    for path,digest in sources.items():
        if not (path.startswith("src/arggnn/") and Path(path).suffix==".py" or
                path in ("AGENTS.md","scripts/launch_gpu.py","scripts/project_env.sh","scripts/predict_confirmatory.py")):
            raise ValueError("Only imported project implementation sources may be declared")
        capture.hash_file({"path":path,"sha256":digest})
    if status=="failed":
        if receipt["prediction"] is not None or not isinstance(receipt["reason"],str) or not receipt["reason"] or not isinstance(receipt["error_type"],str) or not receipt["error_type"] or not isinstance(receipt["runtime"],dict):
            raise ValueError("Actual preserved failure and null scored prediction required")
        if type(receipt["cpu_validation_only"]) is not bool:raise ValueError("Typed failure CPU-validation flag required")
        return {"values":receipt,"terminal_receipt":ref,"status":status,"prediction":None}
    if receipt["phase"]!="complete" or receipt["reason"]!="":raise ValueError("Complete terminal success phase required")
    prediction=capture.ref(receipt["prediction"])
    if expected_prediction is not None and prediction!=capture.ref(expected_prediction):
        raise ValueError("Terminal prediction ref differs from sealed roster")
    _qualified_runtime(receipt["runtime"],frozen_lock.payload,capture)
    peaks=receipt["cuda_allocator_peak_bytes"]
    _keys(peaks,("allocated","reserved"),"Actual CUDA allocator peak evidence required")
    if any(type(value) is not int or value<0 for value in peaks.values()):raise ValueError("Nonnegative actual CUDA allocator counts required")
    provenance=receipt["captured_artifacts"];pairs=set()
    if not isinstance(provenance,list) or not provenance:raise ValueError("Actual prediction admission provenance required")
    for record in provenance:
        _keys(record,("path","sha256","bytes"),"Exact captured prediction provenance required")
        normalized=capture.ref({key:record[key] for key in ("path","sha256")})
        if type(record["bytes"]) is not int or record["bytes"]<0:raise ValueError("Actual provenance byte count required")
        pairs.add((normalized["path"],normalized["sha256"]))
    known=[ledger_ref,identity_ref,*[capture.ref(source) for source in observed_identity["source_artifacts"]]]
    if any((value["path"],value["sha256"]) not in pairs for value in known):
        raise ValueError("Terminal observed-source/identity/ledger provenance differs")
    return {"values":receipt,"terminal_receipt":ref,"status":status,"prediction":prediction}


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
    for ref in list(schema["code_artifacts"].values())+list(schema["terminal_receipt"]["source_artifacts"].values()):
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
        if row["status"] in ("success","failed"):capture.ref(row["terminal_receipt"])
        elif row["terminal_receipt"] is not None:raise ValueError("Unresolved cell cannot claim terminal evidence")
        if row["status"] == "success":
            capture.ref(row["prediction"])
        elif row["prediction"] is not None:
            raise ValueError("Failed/unresolved cells cannot substitute predictions")
        entries[key] = row
    if set(entries) != expected:
        raise ValueError("Complete sealed method/seed/parent roster required before target access")
    targets = {capture.ref(row[name])["path"] for row in by_parent.values() for name in ("primary_labels", "census_tracts")}
    predictions = {capture.ref(row["prediction"])["path"] for row in entries.values() if row["status"] == "success"}
    terminals={capture.ref(row["terminal_receipt"])["path"] for row in entries.values() if row["terminal_receipt"] is not None}
    if terminals & (targets|predictions):raise ValueError("Target/prediction cannot substitute terminal receipt")
    identities = {}
    for parent, row in by_parent.items():
        if capture.ref(row["observed_identity"])["path"] in targets | predictions | terminals:
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
        if edges.shape != (7,) or not np.array_equal(edges,[125000.*i for i in range(1,8)]):
            raise ValueError("Exact125000bp-spaced eight-tile edge identity required")
        if (not isinstance(identity["source_artifacts"], list) or len(identity["source_artifacts"])!=1 or
                Path(capture.ref(identity["source_artifacts"][0])["path"]).suffix!=".npz"):
            raise ValueError("Exactly one observed-only NPZ source required")
        for source in identity["source_artifacts"]:
            if capture.ref(source)["path"] in targets | predictions | terminals:
                raise ValueError("Targets/predictions cannot serve as observed input source")
            capture.hash_file(source)
        identities[parent] = identity
    terminal_values={}
    for key,row in entries.items():
        if row["status"]=="unresolved":continue
        method,seed,parent=key
        terminal=validate_prediction_terminal(capture,row["terminal_receipt"],frozen_lock=frozen,
            method_id=method,seed=seed,parent_id=parent,observed_identity=identities[parent],
            observed_identity_ref=by_parent[parent]["observed_identity"],admission_ledger_ref=capture.ref(ledger_ref),
            expected_status=row["status"],expected_prediction=row["prediction"])
        if row["reason"]!=terminal["values"]["reason"]:raise ValueError("Roster/actual terminal reason differs")
        terminal_values["|".join(map(str,key))]=terminal["values"]
    admission = AnalysisAdmission(frozen, capture, schema, registry, temperatures, ledger, roster, identities,
                                  methods, parents, entries, capture.ref(roster_ref), capture.ref(ledger_ref),terminal_values)
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


def _file_stamp(s):
    return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)


def _verify_staged(path,identity,content,expected_stamp=None):
    lexical=path.lstat()
    if (lexical.st_dev,lexical.st_ino)!=identity:raise ValueError("Owned analysis output inode changed")
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        before=os.fstat(fd)
        if (before.st_dev,before.st_ino)!=identity or not stat.S_ISREG(before.st_mode):
            raise ValueError("Owned analysis output inode changed")
        if expected_stamp is not None and _file_stamp(before)!=expected_stamp:
            raise ValueError("Owned staged analysis output stat drift")
        digest=hashlib.sha256()
        with os.fdopen(fd,"rb",closefd=False) as stream:
            for chunk in iter(lambda:stream.read(4*1024**2),b""):digest.update(chunk)
        after=os.fstat(fd)
        if _file_stamp(before)!=_file_stamp(after) or before.st_size!=len(content) or digest.hexdigest()!=_sha(content):
            raise ValueError("Owned staged analysis output byte/hash drift")
    finally:os.close(fd)


def _unlink_owned(path,identity):
    try:current=path.lstat()
    except FileNotFoundError:return
    if (current.st_dev,current.st_ino)==identity:path.unlink()


def _publish(output, content, check):
    # Hold original FD through cleanup so an unlinked inode cannot be reused by a foreign replacement.
    with tempfile.NamedTemporaryFile(mode="wb",dir=output.parent,prefix=".analysis_pending_",delete=False) as handle:
        pending=Path(handle.name);current=os.fstat(handle.fileno());identity=(current.st_dev,current.st_ino)
        published=False
        try:
            handle.write(content);handle.flush();os.fsync(handle.fileno());stamp=_file_stamp(os.fstat(handle.fileno()))
            check();_verify_staged(pending,identity,content,stamp)
            os.link(pending,output,follow_symlinks=False);published=True
            _verify_staged(output,identity,content)
        except BaseException:
            if published:_unlink_owned(output,identity)
            raise
        finally:_unlink_owned(pending,identity)


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
        "prediction_terminal_receipts":admission.terminal_receipts,
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
