"""Source-only actual-bank adapter and qualification helpers for compact action.

No main entry, fit, scientific labels, bank generation, cache rewrite or CUDA
admission is performed by import. Actual file loading requires a separately
hashed root admission. The frozen candidate and compact core remain unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import copy
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Callable

import torch
from torch import Tensor

from .anchor_kernel import (
    AnchorKernelBank, AnchorKernelContext, AnchorKernelModel,
    anchor_kernel_action)
from .anchor_spectral_compact_v1 import (
    CompactAnchorLayoutV1, CompactSpectralGeometryV1,
    anchor_spectral_compact_action_v1, compact_anchor_messages_v1,
    compact_layout_from_anchor_v1)

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
TRAIN_PARENTS = ("bio_paths_pilot_train_000", "bio_paths_pilot_train_001")
PROCESS_TITLE = "AI Auto Research Pipeline Test"
COUPLINGS = ("full", "scalar", "diagonal", "gain_matched_diagonal")


def sha256_file(p: Path) -> str:
    digest = hashlib.sha256()
    with p.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_file(record: dict, root: Path = ROOT) -> Path:
    """Reject traversal/symlinks before reading a hash-bound repository file."""
    root = root.resolve()
    supplied = Path(record["path"])
    p = supplied if supplied.is_absolute() else root / supplied
    if ".." in p.parts:
        raise ValueError("Parent traversal is forbidden")
    relative = p.relative_to(root)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Bound paths cannot contain symlinks")
    expected = record["sha256"]
    if (not isinstance(expected, str) or len(expected) != 64
            or any(c not in "0123456789abcdef" for c in expected)
            or not p.is_file() or sha256_file(p) != expected):
        raise ValueError("Missing or changed bound file: " + str(p))
    return p


def select_train_tiles_v1(manifest: dict) -> list[dict]:
    """Metadata-only deterministic extrema, before opening any bank.

    Pick minimum G, maximum G, minimum P in that order. Ties use parent/tile ID.
    Selection does not inspect logits, losses, labels, geometry or elapsed time.
    """
    if (manifest.get("status") != "complete_observed_only_Q184_R96_actualQ120"
            or manifest.get("truth_opened") is not False):
        raise ValueError("Require the sealed observed-only prepared manifest")
    rows = manifest["parents"]
    parents = [p for p in rows if p.get("split") == "train"]
    if (len(parents) != 2 or {p["parent_id"] for p in parents} != set(TRAIN_PARENTS)
            or any(p["parent_id"] in TRAIN_PARENTS and p["split"] != "train" for p in rows)):
        raise ValueError("Exactly the two declared TRAIN parents required")
    candidates = []
    seen = set()
    for parent in parents:
        for tile in parent["tiles"]:
            key = (parent["parent_id"], tile["tile_id"])
            g, p = len(tile["original_tree_ids"]), len(tile["position_indexes"])
            if key in seen or not isinstance(tile["tile_id"], int) or min(g, p) < 1:
                raise ValueError("Invalid or duplicate TRAIN tile metadata")
            seen.add(key)
            if (tile.get("query_dynamics") != 184 or tile.get("pure_reference_sources") != 96
                    or tile.get("supervised_outputs") != 120
                    or tile.get("mixed_latent_outputs") != 64
                    or tile.get("mixed_query_flag") != 0
                    or tile.get("all_tree_nodes_retained") is not True
                    or tile.get("all280_leaves_retained") is not True
                    or tile.get("exact_serialization_roundtrip_checked") is not True
                    or tile.get("all_original_age_cache_fields_retained") is not True):
                raise ValueError("Original full observed-bank contract differs")
            candidates.append(dict(parent_id=key[0], tile_id=key[1], graphs=g,
                                   readouts=p, bank=tile["bank"],
                                   full_bank_field_fingerprint_sha256=
                                   tile["full_bank_field_fingerprint_sha256"]))
    if not candidates:
        raise ValueError("No eligible TRAIN tiles")
    result = []
    for reason, key in (
            ("minimum_tree_count", lambda x: (x["graphs"], x["parent_id"], x["tile_id"])),
            ("maximum_tree_count", lambda x: (-x["graphs"], x["parent_id"], x["tile_id"])),
            ("minimum_readout_count", lambda x: (x["readouts"], x["parent_id"], x["tile_id"]))):
        item = min(candidates, key=key)
        existing = next((r for r in result if (r["parent_id"], r["tile_id"]) ==
                         (item["parent_id"], item["tile_id"])), None)
        if existing is None:
            result.append(dict(item, selection_reasons=[reason]))
        else:
            existing["selection_reasons"].append(reason)
    return result


@dataclass(frozen=True)
class ActualQualificationAdmissionV1:
    """Hash-bound capability issued by root, not this source-only artifact."""
    config: dict
    prepared_manifest: dict
    selected: tuple[dict, ...]
    root: Path
    binding: dict


def admit_actual_qualification_v1(config_record: dict, admission_record: dict,
                                  *, root: Path = ROOT) -> ActualQualificationAdmissionV1:
    cfg = json.loads(checked_file(config_record, root).read_text())
    admission = json.loads(checked_file(admission_record, root).read_text())
    if (cfg.get("kind") != "anchor_compact_actual_bank_protocol_v1"
            or cfg.get("status") != "source_only_actual_execution_not_admitted"
            or cfg.get("fit_allowed") is not False or cfg.get("truth_allowed") is not False):
        raise ValueError("Wrong source-only protocol")
    if (admission.get("kind") != "root_compact_actual_bank_qualification_admission_v1"
            or admission.get("status") != "active_root_admission"
            or admission.get("authorizer") != "root_agent"
            or admission.get("config") != config_record
            or admission.get("project_root") != str(root.resolve())
            or admission.get("truth_allowed") is not False
            or admission.get("fit_allowed") is not False
            or admission.get("devices") != ["cpu_float64", "cuda_float32"]
            or admission.get("stages") != ["parity", "profile_after_parity_pass"]
            or admission.get("expected_process_title") != PROCESS_TITLE):
        raise ValueError("Actual qualification needs an exact active root receipt")
    for name, digest in cfg["source_sha256"].items():
        checked_file({"path": name, "sha256": digest}, root)
    review_record = admission["independent_review"]
    review = json.loads(checked_file(review_record, root).read_text())
    if (review.get("verdict") != "PASS"
            or review.get("adapter_sha256") != cfg["source_sha256"][
                "src/arggnn/anchor_compact_actual_bank_v1.py"]
            or review.get("protocol_sha256") != config_record["sha256"]
            or review.get("actual_execution_admitted") is not False):
        raise ValueError("Need independent source/handmade review of exact adapter/protocol")
    # Root owns runtime/storage/lease receipts. This binds their frozen evidence
    # but does not claim to validate their domain-specific content or grant CUDA.
    for name in ("runtime_admission", "storage_admission", "launcher_admission"):
        checked_file(admission[name], root)
    manifest = json.loads(checked_file(cfg["prepared_manifest"], root).read_text())
    if (manifest["config_sha256"] != cfg["original_candidate_config"]["sha256"]
            or manifest["observed_manifest_sha256"] != cfg["observed_manifest"]["sha256"]
            or manifest["source_sha256"] != cfg["original_preparation_source_sha256"]):
        raise ValueError("Original prepared source/config/observation binding changed")
    checked_file(cfg["original_candidate_config"], root)
    observed = json.loads(checked_file(cfg["observed_manifest"], root).read_text())
    if observed.get("truth_or_oracle_read") is not False:
        raise ValueError("Observed provenance cannot contain truth")
    selected = select_train_tiles_v1(manifest)
    if admission.get("selected_train_tiles") != selected:
        raise ValueError("Root selection differs from frozen metadata-only rule")
    return ActualQualificationAdmissionV1(cfg, manifest, tuple(selected), root.resolve(),
        dict(config=config_record, admission=admission_record,
             independent_review=review_record))


def recheck_admission_v1(admission: ActualQualificationAdmissionV1) -> None:
    current = admit_actual_qualification_v1(admission.binding["config"],
                                           admission.binding["admission"],
                                           root=admission.root)
    if (current.selected != admission.selected
            or current.prepared_manifest != admission.prepared_manifest
            or current.config != admission.config):
        raise ValueError("Bound actual qualification changed")


def load_admitted_train_bank_v1(admission: ActualQualificationAdmissionV1,
                                selection_index: int) -> tuple[AnchorKernelBank, dict]:
    """Read one original bank only after provenance admission, retaining hashes.

    Full-file SHA and full-field fingerprint are retained on every load.
    No inferred tree, truth array, VAL bank or on-disk converted cache is opened.
    """
    from .biological_candidate import fingerprint_sha
    recheck_admission_v1(admission)
    if type(selection_index) is not int or not 0 <= selection_index < len(admission.selected):
        raise ValueError("Select only a frozen admitted TRAIN tile")
    selected = admission.selected[selection_index]
    parent = next(p for p in admission.prepared_manifest["parents"]
                  if p["parent_id"] == selected["parent_id"] and p["split"] == "train")
    tile = next(t for t in parent["tiles"] if t["tile_id"] == selected["tile_id"])
    filename = checked_file(tile["bank"], admission.root)
    saved = torch.load(filename, map_location="cpu", weights_only=False)
    if (saved.get("schema") != "biological_candidate_full_age_bank_v1"
            or fingerprint_sha(saved) != tile["full_bank_field_fingerprint_sha256"]):
        raise ValueError("Full original-bank schema/fingerprint changed")
    prepared = saved["fixed_prepared"]
    bank, cache = prepared.bank, prepared.bank.query_cache
    mapping = parent["relate_entry"]["sample_index_to_inferred_tree_node_id"]
    if ([x["sample_index"] for x in mapping] != list(range(280))
            or len({x["inferred_tree_node_id"] for x in mapping}) != 280):
        raise ValueError("Original sample identity mapping changed")
    nodes = torch.tensor([x["inferred_tree_node_id"] for x in mapping], dtype=torch.long)
    dynamic = torch.tensor(parent["dynamics_sample_indexes"], dtype=torch.long)
    reference = torch.tensor(parent["reference_sample_indexes"], dtype=torch.long)
    actual = torch.tensor(parent["actual_query_sample_indexes"], dtype=torch.long)
    output = torch.tensor(parent["actual_query_output_indexes"], dtype=torch.long)
    if (dynamic.shape != (184,) or reference.shape != (96,) or actual.shape != (120,)
            or output.shape != (120,) or torch.unique(output).numel() != 120
            or not torch.equal(dynamic[output], actual)
            or not torch.equal(torch.sort(torch.cat((dynamic, reference))).values,
                               torch.arange(280))
            or not torch.equal(nodes[dynamic], torch.tensor(parent["dynamics_query_node_ids"]))
            or not torch.equal(nodes[actual], torch.tensor(parent["query_node_ids"]))
            or not torch.equal(bank.query_node_ids, nodes[dynamic])
            or not torch.equal(cache.query_ids, nodes[dynamic])
            or not torch.equal(cache.reference_ids, nodes[reference])
            or not torch.equal(torch.sort(bank.sample_ids).values, torch.sort(nodes).values)
            or not torch.equal(bank.sample_ids[cache.query_indices], nodes[dynamic])
            or not torch.equal(bank.sample_ids[cache.reference_indices], nodes[reference])):
        raise ValueError("Original Q184/R96/Q120 identity/gather joins changed")
    indexes = torch.tensor(tile["position_indexes"], dtype=torch.long)
    expected_positions = torch.tensor(parent["positions_bp"], dtype=torch.float64)[indexes] / 2000000
    if (bank.leaf_x.shape[1:] != (280, 47) or prepared.num_scales != 3
            or prepared.layout.anchors.numel() != 32
            or bank.feature_columns != tuple(c for c in range(48) if c != 7)
            or not torch.equal(bank.positions, expected_positions)
            or not torch.equal(torch.stack((bank.intervals[:, 0].min(), bank.intervals[:, 1].max())),
                               torch.tensor(tile["normalized_interval"], dtype=torch.float64))
            or not torch.equal(bank.window_bounds,
                               torch.tensor(tile["normalized_interval"], dtype=torch.float64))
            or bank.positions.numel() != selected["readouts"]
            or bank.leaf_x.shape[0] != selected["graphs"]):
        raise ValueError("Original dimensions/features/physical readout join changed")
    flags = torch.isin(bank.sample_ids, nodes[actual]).to(bank.leaf_x.dtype)
    if not torch.equal(bank.leaf_x[:, :, 4], flags.expand_as(bank.leaf_x[:, :, 4])):
        raise ValueError("Mixed64 rows must retain zero actual-query feature")
    if sha256_file(filename) != tile["bank"]["sha256"]:
        raise ValueError("Bank changed during load/validation")
    return prepared, dict(selected, actual_query_output_indexes=output.tolist(),
                          full_file_and_full_field_hash_checked=True,
                          labels_opened=False, bank_modified=False)


@dataclass(frozen=True)
class CompactActualBankV1:
    """Minimal alias view, not an immutable or newly serialized bank."""
    leaf_x: Tensor
    normalized_response: Tensor
    query_indices: Tensor
    reference_indices: Tensor
    target_graph: Tensor
    layout: CompactAnchorLayoutV1
    geometry: CompactSpectralGeometryV1
    geometry_name: str
    coupling: str
    num_scales: int

    def to(self, device: torch.device | str, dtype: torch.dtype) -> CompactActualBankV1:
        return replace(self,
            leaf_x=self.leaf_x.to(device=device, dtype=dtype),
            normalized_response=self.normalized_response.to(device=device, dtype=dtype),
            query_indices=self.query_indices.to(device=device),
            reference_indices=self.reference_indices.to(device=device),
            target_graph=self.target_graph.to(device=device),
            layout=self.layout.to(device), geometry=self.geometry.to(device, dtype))


def derive_compact_actual_bank_v1(prepared: AnchorKernelBank, geometry: str,
                                  coupling: str = "full") -> CompactActualBankV1:
    """Extract exact required original fields on CPU before dtype conversion."""
    if geometry not in prepared.operators or coupling not in COUPLINGS:
        raise ValueError("Unknown exact geometry/coupling")
    bank, cache = prepared.bank, prepared.bank.query_cache
    if bank.leaf_x.device.type != "cpu" or cache.normalized_response.device.type != "cpu":
        raise ValueError("Derive compact alias view from the sealed CPU bank")
    layout = compact_layout_from_anchor_v1(prepared.layout)
    return CompactActualBankV1(bank.leaf_x, cache.normalized_response,
        cache.query_indices, cache.reference_indices, bank.target_graph, layout,
        CompactSpectralGeometryV1.from_operators(prepared.operators[geometry],
            include_diagonal=coupling in {"diagonal", "gain_matched_diagonal"}),
        geometry, coupling, prepared.num_scales)


class AnchorCompactActualBankModelV1(AnchorKernelModel):
    """Same inherited parameter names, encoder/gates/head and all Q target rows."""
    def __init__(self, *args, readout_chunk_size: int = 128, **kwargs):
        super().__init__(*args, **kwargs)
        if type(readout_chunk_size) is not int or readout_chunk_size < 1:
            raise ValueError("Positive readout chunk size required")
        self.readout_chunk_size = readout_chunk_size

    def encode_context(self, prepared: CompactActualBankV1) -> AnchorKernelContext:
        layout = prepared.layout
        weight = self.encoder.input_linear.weight
        if (prepared.num_scales != self.num_scales or len(layout.anchors) != self.num_anchors
                or prepared.geometry_name != self.geometry or prepared.coupling != self.coupling
                or prepared.leaf_x.shape[-1] != self.input_dim
                or prepared.leaf_x.dtype != weight.dtype or prepared.leaf_x.device != weight.device
                or not bool(torch.isfinite(prepared.leaf_x).all())):
            raise ValueError("Compact bank/model/source contract differs")
        h = self.encoder(prepared.leaf_x, prepared.normalized_response)
        query, reference = h[:, prepared.query_indices], h[:, prepared.reference_indices]
        anchors, targets = query[layout.anchor_graph], query[prepared.target_graph]
        fixed_reference = reference[layout.anchor_graph]
        messages = []
        for layer in self.layers:
            message = compact_anchor_messages_v1(
                layout, prepared.geometry, layer.query_projection(anchors).tanh(),
                layer.reference_projection(fixed_reference).tanh(),
                layer.rates, layer.damping, layer.mixture, coupling=self.coupling,
                readout_chunk_size=self.readout_chunk_size)
            anchors = layer.update(anchors, message.anchors)
            targets = layer.update(targets, message.targets)
            messages.append(message)
        return AnchorKernelContext(anchors, targets, tuple(messages))



def qualification_models_v1(geometry: str, state: str) -> tuple[
        AnchorKernelModel, AnchorCompactActualBankModelV1]:
    """Prespecified CPU seed101 candidate-sized pair, no checkpoint selection."""
    if geometry not in {"plain", "age"} or state not in {"initial", "rate_mixture_stress"}:
        raise ValueError("Frozen geometry/state required")
    args = dict(input_dim=47, num_classes=4, hidden_dim=32, num_layers=2,
                latent_dim=8, kernel_depth=2, num_scales=3, num_anchors=32,
                initial_coupling=(5., 20., 80.), initial_damping=1.,
                geometry=geometry, coupling="full", rate_parameterization="log")
    # Initialization uses CPU RNG and restores caller RNG state. No CUDA API.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(101)
        native = AnchorKernelModel(**args)
        compact = AnchorCompactActualBankModelV1(**args, readout_chunk_size=128)
    if state == "rate_mixture_stress":
        with torch.no_grad():
            for layer in native.layers:
                layer.raw_coupling.copy_(torch.linspace(-.7, .6, 24).reshape(3, 8))
                layer.raw_damping.copy_(torch.linspace(.4, -.3, 24).reshape(3, 8))
                layer.mixture_logits.copy_(torch.linspace(-.8, .7, 24).reshape(3, 8))
    compact.load_state_dict(native.state_dict(), strict=True)
    return native, compact

def comparison_v1(actual: Tensor, expected: Tensor, *, gradient: bool = False) -> dict:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("Parity shape/dtype mismatch")
    if not actual.numel() or not bool(torch.isfinite(actual).all() & torch.isfinite(expected).all()):
        raise ValueError("Parity requires nonempty finite tensors")
    if actual.dtype == torch.float64:
        rtol, atol = (1e-8, 1e-10) if gradient else (1e-10, 1e-12)
    elif actual.dtype == torch.float32:
        rtol, atol = (1e-3, 1e-5) if gradient else (2e-4, 2e-5)
    else:
        raise ValueError("Only frozen float64/float32 parity")
    error = (actual.detach() - expected.detach()).abs()
    units = error / (atol + rtol * expected.detach().abs())
    return dict(passed=bool((units <= 1).all()), max_absolute_error=float(error.max()),
                max_acceptance_units=float(units.max()), rtol=rtol, atol=atol,
                elements=actual.numel())


def qualification_objective_v1(logits: Tensor, context: AnchorKernelContext) -> Tensor:
    """Deterministic no-label cotangent on ALL Q184 and all positive row masses."""
    weights = torch.linspace(-.7, .9, logits.numel(), dtype=logits.dtype,
                             device=logits.device).reshape_as(logits)
    loss = (weights * logits).mean()
    for messages in context.layers:
        if not bool(torch.isfinite(messages.mass).all() & (messages.mass > 0).all()):
            raise ValueError("All Q184 normalization masses must be finite and positive")
        loss = loss + .013 * messages.mass.log().mean()
    return 1000 * loss


def optimizer_v1(model: AnchorKernelModel) -> torch.optim.AdamW:
    special = list(model.rate_and_mixture_parameters())
    identities = {id(p) for p in special}
    groups = [dict(params=[p for p in model.parameters() if id(p) not in identities],
                   weight_decay=.0001), dict(params=special, weight_decay=0.)]
    return torch.optim.AdamW(groups, lr=.001, foreach=False)


def compare_model_step_v1(native_model: AnchorKernelModel,
                           compact_model: AnchorCompactActualBankModelV1,
                           native_bank: AnchorKernelBank,
                           compact_bank: CompactActualBankV1) -> dict:
    """One deterministic full-Q VJP and actively clipped matched AdamW step.

    Only private deep copies of models and feature inputs change. No training,
    ancestry labels, prediction score or checkpoint/early-stopping identity.
    """
    left, right = copy.deepcopy(native_model), copy.deepcopy(compact_model)
    right.load_state_dict(left.state_dict(), strict=True)
    left.eval()
    right.eval()
    left_bank = replace(native_bank, bank=replace(native_bank.bank,
                        leaf_x=native_bank.bank.leaf_x.detach().clone().requires_grad_(True)))
    right_bank = replace(compact_bank,
                         leaf_x=compact_bank.leaf_x.detach().clone().requires_grad_(True))
    lc, rc = left.encode_context(left_bank), right.encode_context(right_bank)
    ll, rl = left.head(lc.targets), right.head(rc.targets)
    checks = {"logits_all_Q": comparison_v1(rl, ll),
              "final_anchors": comparison_v1(rc.anchors, lc.anchors),
              "final_targets": comparison_v1(rc.targets, lc.targets)}
    for i, (a, b) in enumerate(zip(lc.layers, rc.layers)):
        for name in ("anchors", "targets", "numerator", "mass", "sources", "mass_sources"):
            checks[f"layer{i}.{name}"] = comparison_v1(getattr(b, name), getattr(a, name))
        # Per-scale raw action is qualified separately from mixed masses.
        sources = torch.stack((a.sources, a.mass_sources), dim=-1)
        raw_left = anchor_kernel_action(native_bank.layout, native_bank.operators[left.geometry],
                                       sources, left.layers[i].rates, left.layers[i].damping,
                                       coupling=left.coupling)
        raw_right = anchor_spectral_compact_action_v1(
            compact_bank.layout, compact_bank.geometry, sources,
            left.layers[i].rates, left.layers[i].damping, coupling=left.coupling,
            readout_chunk_size=right.readout_chunk_size)
        checks[f"layer{i}.raw_per_scale"] = comparison_v1(raw_right, raw_left)
    lo, ro = optimizer_v1(left), optimizer_v1(right)
    qualification_objective_v1(ll, lc).backward()
    qualification_objective_v1(rl, rc).backward()
    checks["input_feature_VJP"] = comparison_v1(right_bank.leaf_x.grad,
                                                left_bank.bank.leaf_x.grad, gradient=True)
    lp, rp = dict(left.named_parameters()), dict(right.named_parameters())
    if lp.keys() != rp.keys() or left.state_dict().keys() != right.state_dict().keys():
        raise ValueError("Inherited parameter/state-dict identity changed")
    for name in lp:
        if lp[name].grad is None or rp[name].grad is None:
            raise ValueError("Missing expected model VJP: " + name)
        checks[name + ".VJP"] = comparison_v1(rp[name].grad, lp[name].grad, gradient=True)
    ln = torch.nn.utils.clip_grad_norm_(left.parameters(), 1., error_if_nonfinite=True)
    rn = torch.nn.utils.clip_grad_norm_(right.parameters(), 1., error_if_nonfinite=True)
    if not float(ln) > 1 or not float(rn) > 1:
        raise ValueError("Frozen objective did not exercise active gradient clipping")
    checks["preclip_norm"] = comparison_v1(rn, ln, gradient=True)
    # A failed parity must stop BEFORE optimizer mutation.
    if not all(x["passed"] for x in checks.values()):
        return dict(status="parity_failed_optimizer_not_run", checks=checks,
                    all_Q_checked=True, labels_opened=False)
    lo.step()
    ro.step()
    for name in lp:
        checks[name + ".post_AdamW"] = comparison_v1(rp[name], lp[name], gradient=True)
        for key in ("step", "exp_avg", "exp_avg_sq"):
            checks[name + ".optimizer." + key] = comparison_v1(
                ro.state[rp[name]][key], lo.state[lp[name]][key], gradient=True)
    return dict(status="PASS" if all(x["passed"] for x in checks.values()) else
                "parity_failed_after_private_optimizer_step", checks=checks,
                all_Q_checked=True, labels_opened=False,
                preclip_native_norm=float(ln), preclip_compact_norm=float(rn),
                input_models_modified=False)


def _profile_cuda_guard_v1(admission, runtime, lease, parity_receipt):
    from .native_argformer_exec import validate_lease, apply_and_validate_title
    recheck_admission_v1(admission)
    if (parity_receipt.get("status") != "PASS"
            or parity_receipt.get("binding") != admission.binding
            or parity_receipt.get("all_frozen_cases_passed") is not True):
        raise ValueError("No profiling before exact bound qualification PASS")
    if (validate_lease() != lease or apply_and_validate_title() != PROCESS_TITLE
            or runtime.get("actual_process_title") != PROCESS_TITLE
            or runtime.get("pid") != os.getpid()
            or runtime.get("actual_context_uuid") != lease["canonical_uuid"]
            or runtime.get("device") != "cuda"
            or runtime.get("execution_policy") is None):
        raise ValueError("Root-qualified own runtime/UUID/title/lease required")
    # The inherited runtime must already be qualified by the root launcher.
    if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
        raise ValueError("One leased logical CUDA0 required")


def cuda_profile_pair_v1(factories: dict[str, Callable[[], Callable[[], object]]],
                         admission: ActualQualificationAdmissionV1,
                         runtime: dict, lease: dict, parity_receipt: dict,
                         *, stage: str, repetitions: int = 10, warmups: int = 3) -> dict:
    """Synchronized alternating ABBA pair timing and own allocator peaks.

    Each factory creates fresh equivalent state outside its timed callable.
    Pair order is native/compact, compact/native, repeated five times.
    Stage setup is excluded and reported separately by the caller.
    """
    _profile_cuda_guard_v1(admission, runtime, lease, parity_receipt)
    from .native_argformer_exec import validate_lease
    if (type(repetitions) is not int or repetitions != 10
            or type(warmups) is not int or warmups != 3
            or set(factories) != {"native", "compact"}
            or stage not in {"encoder", "source", "action", "decoder",
                             "forward_with_grad", "backward", "optimizer", "full_step"}):
        raise ValueError("Frozen stages, pair names, three warmups and ten repetitions required")
    for _ in range(warmups):
        for name in ("native", "compact"):
            function = factories[name]()
            value = function()
            torch.cuda.synchronize()
            del value, function
    samples = []
    for repetition in range(repetitions):
        order = ("native", "compact") if repetition % 2 == 0 else ("compact", "native")
        for order_index, name in enumerate(order):
            if validate_lease() != lease:
                raise ValueError("Lease changed during profiling")
            setup_begin = time.perf_counter()
            function = factories[name]()
            torch.cuda.synchronize()
            setup_seconds = time.perf_counter() - setup_begin
            baseline_allocated = torch.cuda.memory_allocated()
            baseline_reserved = torch.cuda.memory_reserved()
            torch.cuda.reset_peak_memory_stats()
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin = time.perf_counter()
            start.record()
            value = function()
            stop.record()
            torch.cuda.synchronize()
            seconds = time.perf_counter() - begin
            peak_allocated, peak_reserved = (torch.cuda.max_memory_allocated(),
                                             torch.cuda.max_memory_reserved())
            samples.append(dict(implementation=name, repetition=repetition, order_index=order_index,
                                setup_synchronized_wall_seconds=setup_seconds,
                                synchronized_wall_seconds=seconds,
                                CUDA_event_milliseconds=start.elapsed_time(stop),
                                baseline_allocated_bytes=baseline_allocated,
                                baseline_reserved_bytes=baseline_reserved,
                                peak_allocated_bytes=peak_allocated, peak_reserved_bytes=peak_reserved,
                                incremental_peak_allocated_bytes=peak_allocated-baseline_allocated,
                                incremental_peak_reserved_bytes=peak_reserved-baseline_reserved))
            del value, function, start, stop
    return dict(status="measured", stage=stage, samples=samples, warmups=warmups,
                repetitions_per_implementation=repetitions,
                memory_scope="own_process_Torch_allocator", order="alternating_ABBA",
                excludes_other_process_memory=True, excludes_load_hash_transfer=True,
                measured_speedup_claim=False)


def compare_action_VJPs_v1(model: AnchorKernelModel, native: AnchorKernelBank,
                           compact: CompactActualBankV1) -> dict:
    """Actual encoded first-layer q/r signal and raw rate/damping/mixture VJPs.

    Signals are detached and made independent VJP inputs. Fixed geometry is
    identical. All evaluation rows and all branches remain in the objective.
    """
    if model.rate_parameterization != "log":
        raise ValueError("Frozen actual qualification uses native log-rate model")
    with torch.no_grad():
        h = model.encoder(native.bank.leaf_x, native.bank.query_cache.normalized_response)
        q = h[:, native.bank.query_cache.query_indices][native.layout.anchor_graph]
        r = h[:, native.bank.query_cache.reference_indices][native.layout.anchor_graph]
        layer = model.layers[0]
        initial = (layer.query_projection(q).tanh(), layer.reference_projection(r).tanh(),
                   layer.raw_coupling, layer.raw_damping, layer.mixture_logits)
    def execute(kind):
        query, reference, raw_rate, raw_damping, logits = tuple(
            x.detach().clone().requires_grad_(True) for x in initial)
        rates = layer.initial_coupling * raw_rate.exp()
        damping = layer.initial_damping * raw_damping.exp()
        cqr = native.operators[model.geometry].cqr[native.layout.anchor_graph]
        sources = query[:, None] - (cqr @ reference[:, None]) * rates[None, :, None, :]
        mass_sources = torch.ones_like(sources) - cqr.sum(-1)[..., None] * rates[None, :, None, :]
        bundle = torch.stack((sources, mass_sources), dim=-1)
        if kind == "native":
            raw = anchor_kernel_action(native.layout, native.operators[model.geometry],
                                       bundle, rates, damping, coupling=model.coupling)
        else:
            raw = anchor_spectral_compact_action_v1(compact.layout, compact.geometry,
                bundle, rates, damping, coupling=model.coupling, readout_chunk_size=128)
        mixed = (raw * logits.softmax(0)[None, :, None, :, None]).sum(1)
        numerator, mass = mixed.unbind(-1)
        if not bool(torch.isfinite(mass).all() & (mass > 0).all()):
            raise ValueError("All mixed row masses must remain finite and positive")
        normalized = numerator / mass
        weights = torch.linspace(-.7, .9, normalized.numel(), dtype=normalized.dtype,
                                 device=normalized.device).reshape_as(normalized)
        objective = (normalized * weights).mean() + .013 * mass.log().mean()
        inputs = (query, reference, raw_rate, raw_damping, logits)
        gradients = torch.autograd.grad(objective, inputs)
        return (raw, numerator, mass, normalized), gradients
    left, lg = execute("native")
    right, rg = execute("compact")
    checks = {name: comparison_v1(b, a) for name, a, b in zip(
        ("raw_per_scale", "numerator", "mass", "normalized_all_Q"), left, right)}
    for name, a, b in zip(("query_signal", "reference_signal", "raw_coupling",
                          "raw_damping", "mixture_logits"), lg, rg):
        checks[name + ".VJP"] = comparison_v1(b, a, gradient=True)
    return dict(status="PASS" if all(x["passed"] for x in checks.values()) else "FAIL",
                checks=checks, all_Q_checked=True, labels_opened=False,
                input_model_modified=False)


def cuda_isolated_memory_v1(factories: dict[str, Callable[[], object]],
                            admission: ActualQualificationAdmissionV1,
                            runtime: dict, lease: dict, parity_receipt: dict) -> dict:
    """Three independent own-allocator footprint probes for each implementation.

    Factories must close over CPU-only banks/models and allocate/transfer fresh
    device state inside the call, run the full gradient/clipped-AdamW step,
    then return its own live state/output. No overlapping implementation bank
    is allowed. This probe uses cache flushing outside any timing comparison.
    """
    import gc
    from .native_argformer_exec import validate_lease
    _profile_cuda_guard_v1(admission, runtime, lease, parity_receipt)
    if set(factories) != {"native", "compact"}:
        raise ValueError("Native/compact footprint factories required")
    samples = []
    for repetition in range(3):
        order = ("native", "compact") if repetition % 2 == 0 else ("compact", "native")
        for index, name in enumerate(order):
            if validate_lease() != lease:
                raise ValueError("Lease changed during isolated footprint probe")
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            baseline = torch.cuda.memory_allocated()
            if baseline > 1048576:
                raise ValueError("Isolated footprint requires at most1MiB live context tensors")
            reserved = torch.cuda.memory_reserved()
            torch.cuda.reset_peak_memory_stats()
            value = factories[name]()
            torch.cuda.synchronize()
            samples.append(dict(implementation=name, repetition=repetition, order_index=index,
                baseline_allocated_bytes=baseline, baseline_reserved_bytes=reserved,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                incremental_peak_allocated_bytes=torch.cuda.max_memory_allocated()-baseline,
                incremental_peak_reserved_bytes=torch.cuda.max_memory_reserved()-reserved))
            del value
    return dict(status="measured", samples=samples,
                cache_flushed_outside_timing=True, live_baseline_guard_bytes=1048576,
                required_factory_scope="CPU_to_device_transfer_model_forward_backward_optimizer",
                factory_scope_is_caller_contract=True,
                memory_scope="own_process_Torch_allocator", measured_speedup_claim=False)
