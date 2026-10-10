"""Lock-first observed admission and external-target custody packaging.

No generation, inference, fitting or operating-system permission isolation.
Input artifacts already exist; every real operation requires the public lock validator.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import stat
import tempfile
import zipfile

import numpy as np
from .resources import ROOT
from . import confirmatory_lock as lock_api
from . import confirmatory_analysis_runner as analysis
from .dense_tract_metrics import TractPartition
from .kron_interface import precompute_leaf_interfaces
from .tree_canonicalization import canonicalize_observed_trees

SEEDS = [101, 202, 303]
CLASS_ORDER = ["ANA", "CHG", "WHG", "EHG"]
CODE_PATHS = {
    "custodian": "src/arggnn/confirmatory_custodian.py",
    "custodian_cli": "scripts/prepare_confirmatory_release.py",
    "observed_electrical": "src/arggnn/kron_interface.py",
    "observed_canonicalization": "src/arggnn/tree_canonicalization.py",
    "observed_graph": "src/arggnn/data.py",
    "grid_and_targets": "src/arggnn/dense_tract_metrics.py",
    "artifact_schema": "src/arggnn/confirmatory_analysis_runner.py",
    "observed_validator": "src/arggnn/confirmatory_prediction_runner.py",
}
REQUEST_FIELDS = ["schema_version", "kind", "status", "lock_sha256",
    "parent_seed_registry_sha256", "cohort", "namespace", "attempts"]
ATTEMPT_FIELDS = ["parent_id", "status", "reason", "label_blind", "native_inference_returncode",
    "observed_identity", "primary_labels", "census_tracts"]


def _canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()


def _sha(content):
    return hashlib.sha256(content).hexdigest()


def _pins():
    return {key: {"path": name, "sha256": _sha((ROOT/name).read_bytes())}
            for key, name in CODE_PATHS.items()}


_IMPORTED_PINS = _pins()


def custodian_schema_values():
    """Prospective importer contract only, never a lock or final input example."""
    if _pins() != _IMPORTED_PINS:
        raise ValueError("Imported custodian implementation drift")
    return {"schema_version": 1, "kind": "prespecified_confirmatory_custodian_schema",
        "status": "frozen_before_test_preparation", "code_artifacts": _pins(),
        "request_fields": REQUEST_FIELDS, "attempt_fields": ATTEMPT_FIELDS,
        "request_kind": "confirmatory_custodian_import_request",
        "request_status": "awaiting_label_blind_release",
        "policy": {"cohort": "id_confirmation", "target_admitted_parents": 160,
            "maximum_attempted_parents": 192, "minimum_primary_points": 120,
            "minimum_coverage_fraction": .95, "max_graph_nodes": 1000000,
            "max_condition": 1e12, "training_seeds": SEEDS,
            "observed_admission_before_target_bytes": True,
            "target_integrity_failure_aborts_entire_release": True},
        "outputs": ["admission_ledger.json", "prediction_roster_initial.json", "release_receipt.json"],
        "terminal_roster": {"entry_terminal_receipt": "success/failed actual ref; unresolved null",
            "operational_success_requires_qualified_CUDA": True,
            "CPU_validation_predictions_cannot_join": True, "pending_cells_preserved": True,
            "outputs": ["prediction_roster.json", "roster_seal_receipt.json"]},
        "generation_implemented": False, "operational_permissions_enforced": False}


def _prediction_api():
    return importlib.import_module(".confirmatory_prediction_runner", __package__)


@dataclass
class CustodianAdmission:
    lock: object
    capture: object
    registry: dict
    protocol: dict
    schema: dict
    methods: dict
    content: bytes = b""

    def state(self):
        return _canonical({"lock": self.lock.payload, "registry": self.registry,
            "protocol": self.protocol, "schema": self.schema, "methods": self.methods})

    def check(self, *, full=False):
        if self.state() != self.content or _pins() != _IMPORTED_PINS:
            raise ValueError("In-memory custodian admission or implementation drift")
        self.lock.check(full=full)
        self.capture.check()
        if full:
            for record in list(self.capture.files.values()):
                self.capture.hash_file({key: record[key] for key in ("path", "sha256")})


def admit_custodian(lock_path, expected_lock_sha256, *, project_root=ROOT):
    """Real published-lock validation precedes every import, observed or target read."""
    frozen = lock_api.validate_confirmatory_lock(lock_path, expected_lock_sha256, project_root=project_root)
    payload = frozen.payload
    if (frozen.sha256 != analysis._digest(expected_lock_sha256) or
            payload.get("kind") != "confirmatory_provenance_lock" or
            payload.get("lock_execution_scope") != "id_confirmation_only" or
            payload.get("training_seeds") != SEEDS):
        raise ValueError("Actual validated published ID lock required")
    capture = analysis._Capture(project_root)
    refs = payload["bound_references"]
    registry = capture.read(refs["parent_seed_registry"])
    protocol = capture.read(refs["protocol_json"])
    schema = capture.read(refs["analysis_schema"])
    if schema != analysis.analysis_schema_values():
        raise ValueError("Frozen exact observed/target/roster artifact schema required")
    pins = custodian_schema_values()["code_artifacts"]
    workflow = payload["code_artifacts"]["custodian"]
    if {key: workflow[key] for key in ("path", "sha256")} not in (pins["custodian"], pins["custodian_cli"]):
        raise ValueError("Published custodian workflow source differs")
    # The lock must capture every implementation used by observed admission.
    captured = {(record["path"], record["sha256"]) for record in payload["captured_artifacts"]}
    for ref in pins.values():
        if (ref["path"], ref["sha256"]) not in captured:
            raise ValueError("Custodian support code lacks published captured provenance")
        capture.hash_file(ref)
    cohort = registry["cohorts"]["id_confirmation"]
    if (registry["training_seeds"] != SEEDS or cohort["namespace"] != protocol["confirmation"]["namespace"] or
            cohort["target_admitted_parents"] != 160 or
            protocol["confirmation"]["target_admitted_parents"] != 160 or
            protocol["confirmation"]["max_attempted_parents"] != 192 or
            protocol["common_generation"]["sequence_length_bp"] != 1000000 or
            len(cohort["parents"]) != 192 or
            [row["parent_id"] for row in cohort["parents"]] != list(range(192))):
        raise ValueError("Exact frozen 160/192 ID registry and physical domain required")
    for row in cohort["parents"]:
        if any(type(row[key]) is not int or row[key] < 0
               for key in ("parent_id", "ancestry_seed", "mutation_seed", "inference_seed")):
            raise ValueError("Integer frozen parent seeds required")
    methods = analysis._methods(payload)
    result = CustodianAdmission(frozen, capture, registry, protocol, schema, deepcopy(methods))
    result.content = result.state()
    result.check()
    return result


class _ObservedRejection(ValueError):
    """Valid observed identity misses a label-blind admission threshold."""


def _identity(admission, row, forbidden):
    capture = admission.capture; ref = capture.ref(row["observed_identity"])
    if ref["path"] in forbidden:
        raise ValueError("Target artifact cannot substitute observed identity")
    identity = capture.read(ref)
    analysis._keys(identity, admission.schema["observed_identity"]["fields"], "Exact observed identity required")
    parent = row["parent_id"]; cohort = admission.registry["cohorts"]["id_confirmation"]
    registered = cohort["parents"][parent]
    if (type(identity["schema_version"]) is not int or identity["schema_version"] != 1 or
            identity["kind"] != "sealed_confirmatory_observed_parent_identity" or
            type(identity["parent_id"]) is not int or identity["parent_id"] != parent or
            identity["namespace"] != cohort["namespace"] or identity["lock_sha256"] != admission.lock.sha256 or
            identity["truth_free_observed_inputs"] is not True or
            _canonical(identity["parent_seeds"]) != _canonical(
                {key: registered[key] for key in ("ancestry_seed", "mutation_seed", "inference_seed")})):
        raise ValueError("Observed identity registry/lock differs")
    primary = analysis._grid(identity["primary_grid"]); dense = analysis._grid(identity["dense_grid"])
    if (primary.domain_bp != (0., 1e6) or dense.domain_bp != primary.domain_bp or
            dense.coverage_bp != primary.coverage_bp or primary.grid_points != 128 or dense.grid_points != 4096):
        raise ValueError("Observed exact global grid/domain identity differs")
    if len(primary.positions_bp) < 120 or (primary.coverage_bp[1]-primary.coverage_bp[0])/1e6 < .95:
        raise _ObservedRejection("Observed original-grid coverage admission failed")
    queries = identity["query_ids"]; pairs = np.asarray(identity["diploid_pairs"])
    if (not isinstance(queries, list) or len(queries) != 64 or
            any(type(q) is not int or q < 0 for q in queries) or len(set(queries)) != 64 or
            pairs.shape != (32, 2) or pairs.dtype.kind not in ("i", "u") or
            sorted(pairs.ravel().tolist()) != list(range(64)) or
            identity["artificial_tile_edges_bp"] != [125000., 250000., 375000., 500000., 625000., 750000., 875000.]):
        raise ValueError("Exact query/diploid/eight-tile correspondence required")
    sources = identity["source_artifacts"]
    if not isinstance(sources, list) or len(sources) != 1:
        raise ValueError("Exactly one observed graph NPZ required")
    source = capture.ref(sources[0])
    if Path(source["path"]).suffix != ".npz" or source["path"] in forbidden:
        raise ValueError("Observed NPZ must be separate from all target artifacts")
    body = capture.read(source, json_file=False)  # Hash failures abort; never relabel them as biological QA.
    return ref, identity, body


def _observed_qa(identity, body, lock_sha256):
    """All computations are observed-only and cover every graph before tiling."""
    seq = _prediction_api().validate_observed_npz(body, identity, lock_sha256)
    counts = np.bincount(seq.node_graph.numpy())
    if counts.max() > 1000000:
        raise ValueError("Full-parent graph node cap exceeded")
    bank = precompute_leaf_interfaces(seq, time_column=6, parent_branch_column=7,
        time_scale=10000., max_condition=1e12)
    canonicalize_observed_trees(seq, time_column=6, parent_branch_column=7,
        base_feature_count=8, time_scale=10000., merge_identical_adjacent=False)
    return {"max_graph_nodes": int(counts.max()), "graphs": len(counts),
        "maximum_interior_condition": max(float(d.interior_condition_number) for d in bank.diagnostics),
        "full_parent_electrical_admissible": True, "full_parent_canonicalization_admissible": True}


def _target_npz(capture, ref):
    names = ["metadata_json", "labels", "positions_bp", "query_ids"]
    body = capture.read(ref, json_file=False)
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        members = archive.infolist()
        if (len(members) != len(names) or len({m.filename for m in members}) != len(names) or
                set(m.filename for m in members) != {name+".npy" for name in names} or
                sum(m.file_size for m in members) > 256*1024**2):
            raise ValueError("Exact bounded target NPZ member schema required")
    with np.load(io.BytesIO(body), allow_pickle=False) as data:
        if set(data.files) != set(names):
            raise ValueError("Exact target NPZ arrays required")
        arrays = {name: data[name] for name in names}
    metadata = arrays.pop("metadata_json")
    if metadata.shape != () or metadata.dtype.kind != "U":
        raise ValueError("Scalar Unicode target metadata required")
    return analysis._decode(metadata.item()), arrays


def _target_integrity(admission, row, identity):
    capture = admission.capture
    header, arrays = _target_npz(capture, row["primary_labels"])
    expected = {"schema_version": 1, "kind": "sealed_external_primary_labels",
        "lock_sha256": admission.lock.sha256, "namespace": identity["namespace"],
        "parent_id": row["parent_id"], "observed_identity_sha256": row["observed_identity"]["sha256"],
        "primary_grid_sha256": identity["primary_grid"]["sha256"], "class_order": CLASS_ORDER}
    labels = arrays["labels"]
    if (_canonical(header) != _canonical(expected) or labels.dtype != np.dtype("int64") or
            labels.shape != (len(identity["primary_grid"]["positions_bp"]), 64) or
            np.any(labels < 0) or np.any(labels >= 4) or arrays["positions_bp"].dtype != np.dtype("float64") or
            not np.array_equal(arrays["positions_bp"], identity["primary_grid"]["positions_bp"]) or
            arrays["query_ids"].dtype != np.dtype("int64") or arrays["query_ids"].tolist() != identity["query_ids"]):
        raise ValueError("External labels/physical positions/query identity differs")
    census = capture.read(row["census_tracts"])
    analysis._keys(census, admission.schema["census_json"]["fields"], "Exact external census schema required")
    for key in ("schema_version", "lock_sha256", "namespace", "parent_id", "observed_identity_sha256", "class_order"):
        if _canonical(census[key]) != _canonical(expected[key]):
            raise ValueError("Census target identity differs")
    if (census["kind"] != "sealed_external_census_tract_partitions" or
            census["query_ids"] != identity["query_ids"] or census["domain_bp"] != [0., 1e6]):
        raise ValueError("Census query/domain identity differs")
    partition = TractPartition(tuple(census["query_ids"]), tuple(census["domain_bp"]), 4,
        tuple(tuple(tracts) for tracts in census["tracts"]))
    positions = np.asarray(identity["primary_grid"]["positions_bp"])
    for index, tracts in enumerate(partition.tracts):
        right = np.asarray([t.right_bp for t in tracts]); slots = np.asarray([t.class_slot for t in tracts])
        if not np.array_equal(slots[np.searchsorted(right, positions, side="right")], labels[:, index]):
            raise ValueError("Primary labels disagree with exact full-domain census partitions")


def _write_file(handle, content):
    handle.write(content); handle.flush(); os.fsync(handle.fileno())


def _stat_identity(value):
    return (value.st_dev, value.st_ino)


def _stat_record(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _verify_output_file(path, body, identity, expected_stat=None):
    """Hash the owned inode through a no-follow descriptor, with bounded reads."""
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or _stat_identity(before) != identity:
        raise ValueError("Staged/published output inode drift")
    if expected_stat is not None and _stat_record(before) != expected_stat:
        raise ValueError("Staged output stat drift")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        if _stat_record(opened) != _stat_record(before) or opened.st_size != len(body):
            raise ValueError("Staged/published output size or inode drift")
        digest = hashlib.sha256(); remaining = len(body)+1
        while remaining:
            chunk = os.read(fd, min(1024*1024, remaining))
            if not chunk:break
            digest.update(chunk); remaining -= len(chunk)
        if digest.hexdigest() != _sha(body) or _stat_record(os.fstat(fd)) != _stat_record(opened):
            raise ValueError("Staged/published output content drift")
    finally:
        os.close(fd)
    if _stat_record(path.lstat()) != _stat_record(before):
        raise ValueError("Staged/published output changed during verification")
    return _stat_record(before)


def _cleanup_owned_files(directory, directory_identity, owned):
    try:current = directory.lstat()
    except FileNotFoundError:return
    if not stat.S_ISDIR(current.st_mode) or _stat_identity(current) != directory_identity:return
    for name, identity in owned.items():
        try:current = (directory/name).lstat()
        except FileNotFoundError:continue
        if _stat_identity(current) == identity:(directory/name).unlink()
    try:directory.rmdir()
    except OSError:pass  # Foreign replacement files and directories survive.


def _publish(output, contents, check):
    """Recheck exact bytes/inodes; receipt is last; cleanup preserves foreign replacements."""
    stage = Path(tempfile.mkdtemp(prefix=".custodian_pending_", dir=output.parent))
    stage_identity = _stat_identity(stage.lstat())
    staged = {}; staged_owned = {}; output_identity = None; published_owned = {}; handles = []; failed = False
    def verify_staged():
        current = stage.lstat()
        if not stat.S_ISDIR(current.st_mode) or _stat_identity(current) != stage_identity:
            raise ValueError("Staged output directory ownership drift")
        for name, body in contents.items():
            _verify_output_file(stage/name, body, staged[name]["identity"], staged[name]["stat"])
    def verify_published():
        if _stat_identity(output.lstat()) != output_identity or not stat.S_ISDIR(output.lstat().st_mode):
            raise ValueError("Owned release directory drift")
        for name, identity in published_owned.items():
            _verify_output_file(output/name, contents[name], identity)
    try:
        for name, body in contents.items():
            handle = (stage/name).open("xb"); handles.append(handle)
            identity = _stat_identity(os.fstat(handle.fileno()))
            staged_owned[name] = identity  # Hold the inode open through cleanup, preventing inode reuse.
            _write_file(handle, body)
            staged[name] = {"identity": identity,
                "stat": _verify_output_file(stage/name, body, identity)}
        check(); verify_staged()
        output.mkdir(); output_identity = _stat_identity(output.lstat())
        for name, body in contents.items():
            check(); verify_staged(); verify_published()
            os.link(stage/name, output/name, follow_symlinks=False)
            published_owned[name] = _stat_identity((output/name).lstat())
            if published_owned[name] != staged[name]["identity"]:
                raise ValueError("Published output differs from owned staged inode")
            # A legitimate hard link changes ctime, so rebaseline it only after a
            # successful identity+content verification of the original inode.
            staged[name]["stat"] = _verify_output_file(stage/name, body, staged[name]["identity"])
            verify_published()
        check(full=True); verify_staged(); verify_published()
    except BaseException:
        failed = True
        if output_identity is not None:_cleanup_owned_files(output, output_identity, published_owned)
        raise
    finally:
        try:_cleanup_owned_files(stage, stage_identity, staged_owned)
        finally:
            for handle in handles:handle.close()
            # NFS may retain a silly-rename entry until the owned descriptor
            # closes; retry removal using the same ownership guard afterward.
            _cleanup_owned_files(stage, stage_identity, staged_owned)
            if failed and output_identity is not None:
                _cleanup_owned_files(output, output_identity, published_owned)


@dataclass(frozen=True)
class CustodianRelease:
    receipt: dict
    ledger: dict
    roster: dict
    sha256: str
    path: str


def prepare_confirmatory_release(lock_path, expected_lock_sha256, request_path, expected_request_sha256,
                                *, output_directory, project_root=ROOT):
    """Package existing sources after real lock validation; never generate or fit."""
    admission = admit_custodian(lock_path, expected_lock_sha256, project_root=project_root)
    capture = admission.capture; root = capture.root
    output = capture.path(output_directory)
    if output.exists() or not output.parent.is_dir():
        raise ValueError("Fresh exclusive release directory with existing in-project parent required")
    request_ref = capture.ref({"path": str(request_path), "sha256": expected_request_sha256})
    request = capture.read(request_ref)
    analysis._keys(request, REQUEST_FIELDS, "Exact custodian import request required")
    namespace = admission.registry["cohorts"]["id_confirmation"]["namespace"]
    if (type(request["schema_version"]) is not int or request["schema_version"] != 1 or
            request["kind"] != "confirmatory_custodian_import_request" or
            request["status"] != "awaiting_label_blind_release" or request["cohort"] != "id_confirmation" or
            request["namespace"] != namespace or request["lock_sha256"] != admission.lock.sha256 or
            request["parent_seed_registry_sha256"] != admission.lock.payload["bound_references"]["parent_seed_registry"]["sha256"]):
        raise ValueError("Import request does not bind frozen lock/registry/cohort")
    supplied = request["attempts"]
    if not isinstance(supplied, list) or not supplied or len(supplied) > 192:
        raise ValueError("Ordered bounded attempted-parent request required")
    forbidden = set()
    for index, row in enumerate(supplied):
        analysis._keys(row, ATTEMPT_FIELDS, "Exact custodian attempt required")
        if (type(row["parent_id"]) is not int or row["parent_id"] != index or
                row["status"] not in ("admitted", "rejected") or row["label_blind"] is not True or
                not isinstance(row["reason"], str) or not row["reason"]):
            raise ValueError("Contiguous label-blind attempt order/reason required")
        if row["status"] == "rejected":
            if any(row[name] is not None for name in ("observed_identity", "primary_labels", "census_tracts")):
                raise ValueError("Rejected attempts cannot contain input/target references")
        else:
            for name in ("primary_labels", "census_tracts"):forbidden.add(capture.ref(row[name])["path"])
    attempts = []; identities = {}; qa = []
    for row in supplied:
        if len(identities) == 160:
            raise ValueError("Request continues beyond first160 admitted parents")
        result = {name: deepcopy(row[name]) for name in admission.schema["admission_ledger"]["attempt_fields"]}
        if row["status"] == "admitted":
            if type(row["native_inference_returncode"]) is not int:
                raise ValueError("Explicit integer native inference return code required")
            if row["native_inference_returncode"] != 0:
                result.update(status="rejected", reason="Observed native inference nonzero return code",
                    observed_identity=None, primary_labels=None, census_tracts=None)
            else:
                try:
                    ref, identity, body = _identity(admission, row, forbidden)
                except _ObservedRejection as error:
                    result.update(status="rejected", reason="Observed-only coverage QA rejected: "+str(error),
                        observed_identity=None, primary_labels=None, census_tracts=None)
                    attempts.append(result)
                    admission.check()
                    continue
                try:diagnostics = _observed_qa(identity, body, admission.lock.sha256)
                except ValueError as error:
                    result.update(status="rejected", reason="Observed-only full-parent QA rejected: "+str(error),
                        observed_identity=None, primary_labels=None, census_tracts=None)
                else:
                    identities[row["parent_id"]] = identity
                    for name in ("observed_identity", "primary_labels", "census_tracts"):
                        result[name] = capture.ref(row[name])
                    qa.append({"parent_id": row["parent_id"], **diagnostics})
        attempts.append(result)
        admission.check()
    if len(identities) < 160 and len(attempts) != 192:
        raise ValueError("Admission shortfall must exhaust all192 attempts before release")
    ledger = {"schema_version": 1, "kind": "sealed_confirmatory_admission_ledger",
        "status": "sealed_before_model_scoring", "lock_sha256": admission.lock.sha256,
        "parent_seed_registry_sha256": request["parent_seed_registry_sha256"],
        "cohort": "id_confirmation", "namespace": namespace, "target_admitted_parents": 160,
        "maximum_attempted_parents": 192, "attempts": attempts}
    ledger_bytes = _canonical(ledger); ledger_sha = _sha(ledger_bytes)
    roster = {"schema_version": 1, "kind": "sealed_confirmatory_prediction_roster",
        "status": "sealed_before_target_join", "lock_sha256": admission.lock.sha256,
        "parent_seed_registry_sha256": request["parent_seed_registry_sha256"],
        "admission_ledger_sha256": ledger_sha, "cohort": "id_confirmation", "namespace": namespace,
        "entries": [{"method_id": method, "seed": seed, "parent_id": parent, "status": "unresolved",
            "prediction": None, "terminal_receipt": None, "checkpoint": deepcopy(analysis._seed_bundle(bundle, seed)["checkpoint"]),
            "reason": "Awaiting frozen observed-only prediction"}
            for method, bundle in admission.methods.items() for seed in SEEDS for parent in identities]}
    roster_bytes = _canonical(roster)
    # The complete admission decision and expected roster are sealed in canonical
    # bytes before opening ANY target arrays. Targets cannot change admission.
    identity_state = _canonical(identities)
    def release_check(*, full=False):
        if (_canonical(ledger) != ledger_bytes or _canonical(roster) != roster_bytes or
                _canonical(identities) != identity_state):
            raise ValueError("Sealed admission decision, roster or observed identity drift")
        admission.check(full=full)
    release_check(full=True)
    for row in attempts:
        if row["status"] == "admitted":
            _target_integrity(admission, row, identities[row["parent_id"]])
            release_check()
    release_check(full=True)
    prefix = str(output.relative_to(root))
    receipt = {"schema_version": 1, "kind": "sealed_confirmatory_custodian_release",
        "status": "sealed_before_model_scoring", "lock_sha256": admission.lock.sha256,
        "import_request": request_ref,
        "admission_ledger": {"path": prefix+"/admission_ledger.json", "sha256": ledger_sha},
        "initial_prediction_roster": {"path": prefix+"/prediction_roster_initial.json", "sha256": _sha(roster_bytes)},
        "attempted_parents": len(attempts), "admitted_parents": len(identities),
        "precision_shortfall": len(identities) < 160, "expected_prediction_cells": len(roster["entries"]),
        "label_blind_admission_sealed_before_target_integrity_checks": True,
        "external_targets_checked_for_integrity_only": True, "observed_qa": qa,
        "captured_artifacts": capture.provenance(), "code_artifacts": custodian_schema_values()["code_artifacts"],
        "generation_performed": False, "prediction_performed": False, "target_analysis_performed": False,
        "positive_confirmation_authorized": False, "operational_permissions_enforced": False}
    receipt_bytes = _canonical(receipt)
    _publish(output, {"admission_ledger.json": ledger_bytes, "prediction_roster_initial.json": roster_bytes,
        "release_receipt.json": receipt_bytes}, release_check)
    return CustodianRelease(receipt, ledger, roster, _sha(receipt_bytes), str(output/"release_receipt.json"))



@dataclass(frozen=True)
class PredictionRosterSeal:
    roster: dict
    receipt: dict
    sha256: str
    path: str


def seal_confirmatory_prediction_roster(lock_path, expected_lock_sha256,
        ledger_path, expected_ledger_sha256, initial_roster_path, expected_initial_roster_sha256,
        *, terminal_receipts=(), output_directory, project_root=ROOT):
    """Seal operational terminal records or explicit pending cells; never open targets."""
    custody = admit_custodian(lock_path, expected_lock_sha256, project_root=project_root)
    admission = analysis.admit_analysis(lock_path, expected_lock_sha256,
        initial_roster_path, expected_initial_roster_sha256, ledger_path, expected_ledger_sha256,
        project_root=project_root)
    capture = admission.capture
    if custody.lock.sha256 != admission.lock.sha256:
        raise ValueError("Custody and prediction roster locks differ")
    if any(row["status"] != "unresolved" or row["prediction"] is not None or
           row.get("terminal_receipt") is not None for row in admission.roster["entries"]):
        raise ValueError("Exact untouched initial unresolved roster required")
    output = capture.path(output_directory)
    if output.exists() or not output.parent.is_dir():
        raise ValueError("Fresh exclusive roster seal directory required")
    targets = {capture.ref(row[name])["path"] for row in admission.ledger["attempts"]
        if row["status"] == "admitted" for name in ("primary_labels", "census_tracts")}
    observed = {capture.ref(row["observed_identity"])["path"] for row in admission.ledger["attempts"]
        if row["status"] == "admitted"}
    observed.update(capture.ref(source)["path"] for identity in admission.identities.values()
        for source in identity["source_artifacts"])
    entries = deepcopy(admission.entries); seen = set(); paths = set()
    parent_rows = {row["parent_id"]: row for row in admission.ledger["attempts"] if row["status"] == "admitted"}
    checker = getattr(analysis, "validate_prediction_terminal", None)
    if checker is None:
        raise ValueError("Frozen operational terminal receipt validator unavailable")
    for value in terminal_receipts:
        ref = capture.ref(value)
        if ref["path"] in targets | observed or ref["path"] in paths:
            raise ValueError("Unique terminal record cannot alias observed/target artifacts")
        paths.add(ref["path"])
        body = capture.read(ref)
        cell = body.get("cell", {})
        key = (cell.get("method_id"), cell.get("training_seed"), cell.get("parent_id"))
        if (type(key[1]) is not int or type(key[2]) is not int or key not in entries or key in seen):
            raise ValueError("Duplicate, unknown or noninteger terminal prediction cell")
        seen.add(key); method, seed, parent = key
        result = checker(capture, ref, frozen_lock=admission.lock, method_id=method,
            seed=seed, parent_id=parent, observed_identity=admission.identities[parent],
            observed_identity_ref=parent_rows[parent]["observed_identity"],
            admission_ledger_ref=admission.ledger_ref)
        if result["status"] not in ("success", "failed"):
            raise ValueError("Actual terminal success or failure required")
        row = entries[key]
        row.update(status=result["status"], prediction=result["prediction"],
            terminal_receipt=result["terminal_receipt"], reason=result["values"]["reason"])
        if row["status"] == "success":
            prediction = capture.ref(row["prediction"])
            if prediction["path"] in targets | observed | paths:
                raise ValueError("Prediction cannot alias observed, target or terminal artifacts")
            # This prediction-only view checks NPZ header/query/grid/raw-logit
            # identity without changing the admitted initial roster or targets.
            view = SimpleNamespace(**{**admission.__dict__, "entries": entries})
            analysis._prediction(view, method, seed, parent)
        elif row["prediction"] is not None:
            raise ValueError("Failed terminal cell cannot provide a prediction")
    roster = deepcopy(admission.roster)
    roster["entries"] = [entries[(row["method_id"], row["seed"], row["parent_id"])]
        for row in admission.roster["entries"]]
    roster_bytes = _canonical(roster)
    def check(*, full=False):
        if _canonical(roster) != roster_bytes or _pins() != _IMPORTED_PINS:
            raise ValueError("Sealed terminal roster or implementation drift")
        custody.check(full=full)
        if analysis._state(admission) != admission.content:
            raise ValueError("Initial admitted prediction matrix drift")
        admission.lock.check(full=full); capture.check()
        if full:
            for record in list(capture.files.values()):
                capture.hash_file({name: record[name] for name in ("path", "sha256")})
    check(full=True)
    counts = {status: sum(row["status"] == status for row in roster["entries"])
        for status in ("success", "failed", "unresolved")}
    prefix = str(output.relative_to(capture.root))
    receipt = {"schema_version": 1, "kind": "sealed_confirmatory_prediction_roster_receipt",
        "lock_sha256": admission.lock.sha256, "admission_ledger": admission.ledger_ref,
        "initial_roster": admission.roster_ref,
        "roster": {"path": prefix+"/prediction_roster.json", "sha256": _sha(roster_bytes)},
        "cell_counts": counts, "matrix_terminal": counts["unresolved"] == 0,
        "target_artifacts_opened": False, "positive_confirmation_authorized": False,
        "operational_permissions_enforced": False, "captured_artifacts": capture.provenance(),
        "code_artifacts": custodian_schema_values()["code_artifacts"]}
    receipt_bytes = _canonical(receipt)
    _publish(output, {"prediction_roster.json": roster_bytes, "roster_seal_receipt.json": receipt_bytes}, check)
    return PredictionRosterSeal(roster, receipt, _sha(roster_bytes), str(output/"prediction_roster.json"))

