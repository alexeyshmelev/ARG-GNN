"""Reuse a certified fixed forest through the unchanged Census encoder branch.

Shipped source is inactive. Real profile/fit calls need a fresh exact-source
reuse capability plus the original root gates. No data readers or scheduler
changes are introduced here.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import weakref
import torch
from . import census_biological_bank_v1 as bank_api
from . import census_biological_fit_v2 as fit_api
from . import census_tree_encoder_v1 as tree_api
from .census_cut_exchange_v1 import permuted_group_control_v1

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = "configs/census_certified_forest_reuse_v1.json"
CONFIG_SHA256 = "36d467af33d4fa24d32df71d0f88960d17ca4c39243aeed76d5a973fb8d14af2"
SOURCE_PATH = "src/arggnn/census_certified_forest_reuse_v1.py"
ORIGINAL_FORWARD = bank_api.forward_certified_census_biological_v1
ORIGINAL_ENCODER_FORWARD = tree_api.CensusWholeTreeEncoderV1.forward
ORIGINAL_MODEL_FORWARD = tree_api.CensusTreeGNNV1.forward
CONTROLS = ("proposed", "eta0", "permuted_groups", "raw_diagonal", "local")


def load_config_v1():
    data = (ROOT / CONFIG_PATH).read_bytes()
    if hashlib.sha256(data).hexdigest() != CONFIG_SHA256:
        raise ValueError("Certified forest reuse protocol bytes changed")
    cfg = json.loads(data)
    for path, expected in cfg["original_source_sha256"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
            raise ValueError("Sealed original source changed: " + path)
    return cfg


class CertifiedForestReuseV1:
    """Single-certificate weak binding; no cross-tile bank or neural cache."""
    def __init__(self, *, synchronize=None):
        self._synchronize = synchronize
        self._certificate = None
        self._expected = None
        self._routing = False
        self.calls = 0
        self.bindings = 0
        self.reuses = 0
        self.phase_seconds = {"certificate_SHA": 0., "forest_binding_or_checks": 0.,
                              "forward": 0.}

    def _forest(self, cert, expected):
        if type(cert) is not bank_api.CertifiedCensusBiologicalBankV1:
            raise ValueError("Original certified Census bank type required")
        if type(cert.forest) is not tree_api.RetainedObservedForestV1:
            raise ValueError("Original fixed retained forest type required")
        bound = self._certificate() if self._certificate is not None else None
        if bound is cert:
            if expected != self._expected:
                raise ValueError("Already bound certificate hash changed")
            # Preserve the exact per-use retained input checks performed by the
            # original forest builder, including its feature/time predicates.
            for inputs in cert.inputs:
                tree_api.retained_observed_features_v1(inputs, time_scale=10000.)
            self.reuses += 1
        else:
            reference = tree_api.prepare_retained_observed_forest_v1(
                cert.inputs, time_scale=10000.)
            if (bank_api.fingerprint_sha256_v1(reference) !=
                    bank_api.fingerprint_sha256_v1(cert.forest)):
                raise ValueError("Certified forest differs from original input-derived forest")
            self._certificate = weakref.ref(cert)
            self._expected = expected
            self.bindings += 1
        return cert.forest

    @contextmanager
    def _route(self, model, cert, forest):
        if (type(model) is not tree_api.CensusTreeGNNV1 or
                type(model.encoder) is not tree_api.CensusWholeTreeEncoderV1 or
                model.encoder.time_scale != 10000. or
                "forward" in model.__dict__ or
                model.forward.__func__ is not ORIGINAL_MODEL_FORWARD):
            raise ValueError("Original Census model and exact time_scale10000 required")
        encoder = model.encoder
        if (self._routing or "forward" in encoder.__dict__ or
                encoder.forward.__func__ is not ORIGINAL_ENCODER_FORWARD):
            raise ValueError("Nested or overridden encoder routing is unsupported")
        self._routing = True
        calls = [0]
        def routed(inputs):
            if inputs is not cert.inputs or calls[0] != 0:
                raise ValueError("Only the exact certified original input tuple may route once")
            calls[0] += 1
            return ORIGINAL_ENCODER_FORWARD(encoder, forest)
        encoder.forward = routed
        try:
            yield
            if calls[0] != 1:
                raise ValueError("Unchanged Census forward did not consume its forest once")
        finally:
            del encoder.forward
            self._routing = False

    def __call__(self, model, cert, *, expected_certificate_sha256,
                 control="proposed", group_permutation=None, readout_chunk_size=128):
        if self._routing:
            raise ValueError("Recursive certified forest reuse is unsupported")
        begin = time.perf_counter()
        bank_api.validate_census_biological_certificate_v1(cert, expected_certificate_sha256)
        self.phase_seconds["certificate_SHA"] += time.perf_counter() - begin
        begin = time.perf_counter()
        forest = self._forest(cert, expected_certificate_sha256)
        self.phase_seconds["forest_binding_or_checks"] += time.perf_counter() - begin
        bank = cert.bank
        if control == "permuted_groups":
            if group_permutation is None:
                raise ValueError("Fixed global group permutation required")
            bank = permuted_group_control_v1(bank, group_permutation)
        elif group_permutation is not None:
            raise ValueError("Membership permutation belongs only to its declared contrast")
        if self._synchronize:
            self._synchronize()
        begin = time.perf_counter()
        with self._route(model, cert, forest):
            result = model(cert.inputs, bank, control=control,
                           readout_chunk_size=readout_chunk_size)
        if self._synchronize:
            self._synchronize()
        self.phase_seconds["forward"] += time.perf_counter() - begin
        self.calls += 1
        return result

    def report(self):
        return {"calls": self.calls, "forest_bindings": self.bindings,
                "forest_reuses": self.reuses, "phase_seconds": dict(self.phase_seconds),
                "timing_semantics": ("synchronized future profile phases" if self._synchronize else
                                    "host elapsed; GPU phase attribution needs synchronized future profile"),
                "speedup_claim": False}


@contextmanager
def installed_fit_forward_v1(adapter):
    """Use only in a fresh isolated worker, never a currently running worker."""
    if type(adapter) is not CertifiedForestReuseV1:
        raise ValueError("Original dedicated reuse adapter required")
    if fit_api.forward_certified_census_biological_v1 is not ORIGINAL_FORWARD:
        raise ValueError("Original fit forward helper already replaced")
    fit_api.forward_certified_census_biological_v1 = adapter
    try:
        yield
    finally:
        fit_api.forward_certified_census_biological_v1 = ORIGINAL_FORWARD


def admit_reuse_v1(record, *, phase, base_admission_ref):
    """Small extra exact-source gate; original fit/numerical gates remain separate."""
    if phase not in ("fit", "observed_profile"):
        raise ValueError("Declared reuse phase required")
    reader = fit_api.VerifiedReaderV1()
    admission = reader.decode_json(record)
    required = {"schema", "status", "fit_authorized", "observed_profile_authorized",
                "not_before_utc", "expires_at_utc", "base_admission_ref",
                "config_ref", "source_ref", "review_ref"}
    if (set(admission) != required or
            admission["schema"] != "census_certified_forest_reuse_admission_v1" or
            admission["status"] != "active" or
            admission[phase + "_authorized"] is not True or
            admission["base_admission_ref"] != base_admission_ref):
        raise ValueError("Fresh root exact reuse capability required")
    now = datetime.now(timezone.utc)
    times = [datetime.fromisoformat(admission[k].replace("Z", "+00:00"))
             for k in ("not_before_utc", "expires_at_utc")]
    if any(t.tzinfo is None for t in times) or not times[0] <= now < times[1]:
        raise ValueError("Reuse capability is inactive or expired")
    cfg = load_config_v1()
    if (admission["config_ref"] != {"path": CONFIG_PATH, "sha256": CONFIG_SHA256} or
            reader.decode_json(admission["config_ref"]) != cfg or
            admission["source_ref"].get("path") != SOURCE_PATH):
        raise ValueError("Exact reuse source/config bindings required")
    source = reader.read(admission["source_ref"])
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != hashlib.sha256(source).hexdigest():
        raise ValueError("Running reuse implementation differs from reviewed source")
    reader.read(base_admission_ref)
    review = reader.decode_json(admission["review_ref"])
    expected = {SOURCE_PATH: admission["source_ref"]["sha256"], CONFIG_PATH: CONFIG_SHA256}
    if (review.get("status") != "PASS" or review.get("actual_admission") is not False or
            any(review.get("artifact_sha256", {}).get(p) != h for p, h in expected.items())):
        raise ValueError("Fresh independent exact reuse source/config PASS required")
    return admission


def run_original_fit_v1(base_admission_ref, reuse_admission_ref):
    admit_reuse_v1(reuse_admission_ref, phase="fit", base_admission_ref=base_admission_ref)
    adapter = CertifiedForestReuseV1()
    with installed_fit_forward_v1(adapter):
        result = fit_api.run_fit_v1(base_admission_ref)
    return {"original_fit_receipt": result, "forest_reuse_report": adapter.report()}


def observed_timing_v1(model, prepare_certificate, *, expected_certificate_sha256,
                       base_admission_ref,
                       reuse_admission_ref, control="proposed", group_permutation=None,
                       readout_chunk_size=128):
    """Future observed-only phase timing, with synthetic backward and no optimizer.

    The root-provided preparation callable must be observation-only. This entry
    reads no target, invokes no fit loop, makes no optimizer, and applies no step.
    The expected certificate hash is supplied separately from the prepared object.
    """
    admit_reuse_v1(reuse_admission_ref, phase="observed_profile",
                  base_admission_ref=base_admission_ref)
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError("Observed diagnostic requires initially absent parameter gradients")
    def sync():
        device = next(model.parameters()).device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    sync()
    begin = time.perf_counter()
    cert = prepare_certificate()
    sync()
    preparation = time.perf_counter() - begin
    adapter = CertifiedForestReuseV1(synchronize=sync)
    try:
        sync()
        output = adapter(model, cert, expected_certificate_sha256=expected_certificate_sha256,
                         control=control, group_permutation=group_permutation,
                         readout_chunk_size=readout_chunk_size)
        sync()
        before_backward = time.perf_counter()
        objective = output.logits.square().mean()
        objective.backward()
        sync()
        backward = time.perf_counter() - before_backward
        bank_api.validate_census_biological_certificate_v1(cert, expected_certificate_sha256)
        return {"preparation_seconds": preparation, **adapter.report(),
                "synthetic_backward_seconds": backward,
                "backward_objective": "observed_logits_squared_mean_no_target_no_optimizer",
                "TRAIN_loss_or_fit": False, "speedup_claim": False}
    finally:
        for parameter in model.parameters():
            parameter.grad = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-admission")
    parser.add_argument("--expected-fit-admission-sha256")
    parser.add_argument("--reuse-admission")
    parser.add_argument("--expected-reuse-admission-sha256")
    args = parser.parse_args()
    if not all(vars(args).values()):
        raise ValueError("Inactive source requires both fresh exact root fit/reuse admission refs")
    base = {"path": args.fit_admission, "sha256": args.expected_fit_admission_sha256}
    reuse = {"path": args.reuse_admission, "sha256": args.expected_reuse_admission_sha256}
    result = run_original_fit_v1(base, reuse)
    print(json.dumps({"status": result["original_fit_receipt"]["status"],
                      "forest_reuse_report": result["forest_reuse_report"]}, sort_keys=True))


if __name__ == "__main__":
    main()
