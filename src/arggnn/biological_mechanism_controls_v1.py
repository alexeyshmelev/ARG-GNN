"""Separate source-bound biological mechanism controls on frozen H020 banks."""
from __future__ import annotations

import sys
# Module launcher guard must precede Torch, candidate and control imports.
if __name__ == "__main__" and "--train" in sys.argv:
    requested_device = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--device=")),
        sys.argv[sys.argv.index("--device")+1] if "--device" in sys.argv else "cuda")
    if requested_device == "cuda":
        from .native_argformer_exec import validate_lease, apply_and_validate_title
        validate_lease()
        apply_and_validate_title()

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import Tensor

from . import biological_candidate as frozen
from .anchor_kernel import AnchorKernelModel, AnchorKernelBank
from .anchor_controls import NoncontextualAnchorKernelModel
from .age_intervention_controls import SplitAgeAnchorKernelModel

ROOT = frozen.ROOT
BASE_CONFIG_PATH = "configs/biological_candidate_pilot_v1.json"
BASE_CONFIG_SHA256 = "4151e7e156db3f188512bcc738b43597407d7707bc77c420509165c2e9bfc2f1"
EXTRA_SOURCE_PATHS = (
    "src/arggnn/biological_mechanism_controls_v1.py",
    "scripts/train_biological_mechanism_controls_v1.py",
    "src/arggnn/anchor_controls.py",
    "src/arggnn/age_intervention_controls.py",
)
CONTROL_DEPENDENCY_SHA256 = {
    "src/arggnn/anchor_controls.py": "f4939b4515bafb7ba77373980d2be44f7c9be1a5b6881f2166ca0a2d6265063e",
    "src/arggnn/age_intervention_controls.py": "e0ebce68121af8c9bea250d90e28b40df84747ad25c8ce8f13cb71ddcbc41e28",
}
CONTROL_NAMES = (
    "bio_mc_v1_scalar_age_d2",
    "bio_mc_v1_age_propagation_only_d2",
    "bio_mc_v1_age_source_only_d2",
    "bio_mc_v1_age_gainmatched_d2",
    "bio_mc_v1_noncontext_age_d2",
    "bio_mc_v1_age_d1",
    "bio_mc_v1_age_d1_plus_pointwise",
)
CONTROL_SPECS = {
    CONTROL_NAMES[0]: {"geometry": "matched_scalar_age", "coupling": "full", "kernel_depth": 2,
        "control": "scalar_age", "reference": "candidate_age_d2", "parameters": 10420},
    CONTROL_NAMES[1]: {"geometry": "age", "coupling": "full", "kernel_depth": 2,
        "control": "propagation_only", "reference": "candidate_age_d2", "parameters": 10420},
    CONTROL_NAMES[2]: {"geometry": "age", "coupling": "full", "kernel_depth": 2,
        "control": "source_only", "reference": "candidate_age_d2", "parameters": 10420},
    CONTROL_NAMES[3]: {"geometry": "age", "coupling": "gain_matched_diagonal", "kernel_depth": 2,
        "control": "gain_matched", "reference": "candidate_age_d2", "parameters": 10420},
    CONTROL_NAMES[4]: {"geometry": "age", "coupling": "full", "kernel_depth": 2,
        "control": "noncontextual", "reference": "candidate_age_d2", "parameters": 10420},
    CONTROL_NAMES[5]: {"geometry": "age", "coupling": "full", "kernel_depth": 1,
        "control": "depth_one", "reference": "candidate_age_d2", "parameters": 8188},
    CONTROL_NAMES[6]: {"geometry": "age", "coupling": "full", "kernel_depth": 1,
        "pointwise_depth": 1, "learned_layer_schema_depth": 2,
        "control": "one_kernel_plus_pointwise", "reference": "candidate_age_d2", "parameters": 10420},
}


@dataclass(frozen=True)
class PointwiseMessage:
    message: Tensor
    source: Tensor
    mass_source: Tensor
    branch_message: Tensor
    bounded_gain: Tensor


@dataclass(frozen=True)
class OneKernelPointwiseContext:
    anchors: Tensor
    targets: Tensor
    kernel_message: object
    pointwise_anchors: PointwiseMessage
    pointwise_targets: PointwiseMessage


def pointwise_reference_message(layer, cqr: Tensor, query: Tensor, reference: Tensor) -> PointwiseMessage:
    """Query-local second update, with no QQ action or coordinate aggregation.

    S_s=q(U)-lambda_s Cqr_s r(R), m_s=1-lambda_s Cqr_s 1.
    M=sum_s pi_s [d_s/(1+d_s)] [S_s/m_s].
    Both source and mass use the same fixed QR. Damping is an active bounded
    local gain here, not a decay or an extra inactive capacity parameter.
    """
    rates, damping = layer.rates, layer.damping
    if (not bool(torch.isfinite(rates).all()) or not bool(torch.isfinite(damping).all())
            or bool((rates <= 0).any()) or bool((damping <= 0).any())):
        raise ValueError("Pointwise rates and gains must be finite and positive")
    if (cqr.ndim != 4 or query.ndim != 3 or reference.ndim != 3
            or cqr.shape[0] != query.shape[0] or cqr.shape[2] != query.shape[1]
            or cqr.shape[0] != reference.shape[0] or cqr.shape[3] != reference.shape[1]
            or cqr.shape[1] != rates.shape[0]):
        raise ValueError("Pointwise QR/query/reference dimensions differ")
    if (not bool(torch.isfinite(cqr).all()) or not bool(torch.isfinite(query).all())
            or not bool(torch.isfinite(reference).all()) or bool((cqr > 0).any())):
        raise ValueError("Pointwise observed QR/features must be finite and QR nonpositive")
    q_signal = layer.query_projection(query).tanh()
    r_signal = layer.reference_projection(reference).tanh()
    source = q_signal[:, None] - (cqr @ r_signal[:, None]) * rates[None, :, None, :]
    mass = torch.ones_like(source) - cqr.sum(-1)[..., None] * rates[None, :, None, :]
    if (not bool(torch.isfinite(source).all()) or not bool(torch.isfinite(mass).all())
            or bool((mass <= 0).any())):
        raise ValueError("Pointwise reference-inclusive mass is nonpositive/nonfinite")
    gain = damping / (1 + damping)
    branch = source / mass
    message = (branch * gain[None, :, None, :] * layer.mixture[None, :, None, :]).sum(1)
    if not bool(torch.isfinite(message).all()):
        raise ValueError("Pointwise normalized message is nonfinite")
    return PointwiseMessage(message, source, mass, branch, gain)


class OneKernelPointwiseModel(AnchorKernelModel):
    """One age/full kernel plus a same-schema active query-local residual block."""

    def __init__(self, *args, **kwargs):
        requested_depth = kwargs.pop("kernel_depth", 2)
        if requested_depth != 2:
            raise ValueError("Pointwise comparator requires exactly two learned layer schemas")
        if kwargs.get("geometry", "plain") != "age" or kwargs.get("coupling", "full") != "full":
            raise ValueError("Pointwise comparator requires age/full first kernel")
        super().__init__(*args, kernel_depth=2, **kwargs)

    def encode_context(self, prepared: AnchorKernelBank) -> OneKernelPointwiseContext:
        bank, layout = prepared.bank, prepared.layout
        if (prepared.num_scales != self.num_scales or layout.anchors.numel() != self.num_anchors
                or self.geometry not in prepared.operators):
            raise ValueError("Prepared branch/anchor count or geometry differs from model")
        weight = self.encoder.input_linear.weight
        if (bank.leaf_x.shape[-1] != self.input_dim or bank.leaf_x.dtype != weight.dtype
                or bank.leaf_x.device != weight.device or not bool(torch.isfinite(bank.leaf_x).all())):
            raise ValueError("Prepare finite shared features with matching model dimension/device/dtype")
        h = self.encoder(bank.leaf_x, bank.query_cache.normalized_response)
        query = h[:, bank.query_cache.query_indices]
        reference = h[:, bank.query_cache.reference_indices]
        anchors, targets = query[layout.anchor_graph], query[bank.target_graph]
        fixed_anchor_reference = reference[layout.anchor_graph]
        first = self.layers[0].messages(prepared, anchors, fixed_anchor_reference, self.geometry, "full")
        anchors = self.layers[0].update(anchors, first.anchors)
        targets = self.layers[0].update(targets, first.targets)
        cqr = prepared.operators[self.geometry].cqr
        second_anchors = pointwise_reference_message(self.layers[1], cqr[layout.anchor_graph],
            anchors, fixed_anchor_reference)
        second_targets = pointwise_reference_message(self.layers[1], cqr[bank.target_graph],
            targets, reference[bank.target_graph])
        return OneKernelPointwiseContext(
            self.layers[1].update(anchors, second_anchors.message),
            self.layers[1].update(targets, second_targets.message),
            first, second_anchors, second_targets)

    def configuration_manifest(self, prepared=None):
        result = super().configuration_manifest(prepared)
        result["core_configuration_sha256"] = result.pop("configuration_sha256")
        result.update(
            kind="one_age_kernel_plus_active_pointwise_residual_control",
            kernel_depth=1, pointwise_depth=1, learned_layer_schema_depth=2,
            pointwise_damping_meaning="positive bounded gain d/(1+d), not temporal decay",
            pointwise_source="current query at each coordinate and original observed reference of containing graph",
            pointwise_normalization="per-branch S/m using same selected QR for numerator/ones, bounded gain then global mixture",
            pointwise_query_context_exchange=False, pointwise_cross_coordinate_action=False,
            second_layer_kernel_called=False, pointwise_parameter_padding=False,
            decoder="first exact kernel readout then coordinate-local reference-conditioned residual update",
            rate_scope="kernel layer0 global coupling/decay and pointwise layer1 reference coupling/bounded gain",
            prior_state_exchange_by_layer=[True, False],
            control_interpretation="one kernel plus matched additional query-local nonlinear capacity versus two kernels",
            limitations="equal parameter count/schema does not mean equal function class, rate meaning or optimization",
            controls_implementation_sha256=frozen.sha(Path(__file__)))
        result["configuration_sha256"] = hashlib.sha256(
            json.dumps(result, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return result


def make_control(name, *, input_dim=47, num_classes=4, hidden_dim=32, num_layers=2,
                 latent_dim=8, num_anchors=32, initial_coupling=(5., 20., 80.), initial_damping=1.):
    """General mathematical constructors, with fixed biological defaults in the runner."""
    if name not in CONTROL_SPECS:
        raise ValueError("Unknown registered biological mechanism condition")
    spec = CONTROL_SPECS[name]
    common = dict(input_dim=input_dim, num_classes=num_classes, hidden_dim=hidden_dim,
        num_layers=num_layers, latent_dim=latent_dim, num_scales=3, num_anchors=num_anchors,
        initial_coupling=initial_coupling, initial_damping=initial_damping,
        geometry=spec["geometry"], coupling=spec["coupling"], rate_parameterization="log")
    control = spec["control"]
    if control in ("source_only", "propagation_only"):
        return SplitAgeAnchorKernelModel(**common, kernel_depth=2, age_intervention=control)
    if control == "noncontextual":
        return NoncontextualAnchorKernelModel(**common, kernel_depth=2)
    if control == "one_kernel_plus_pointwise":
        return OneKernelPointwiseModel(**common, kernel_depth=2)
    return AnchorKernelModel(**common, kernel_depth=spec["kernel_depth"])


def model_factory(name, cfg):
    if cfg["models"] != list(CONTROL_NAMES) or cfg["model_specs"] != CONTROL_SPECS:
        raise ValueError("Frozen control matrix changed")
    model = make_control(name)
    if sum(p.numel() for p in model.parameters()) != CONTROL_SPECS[name]["parameters"]:
        raise ValueError("Frozen active-layer parameter count changed")
    return model


def extension_protocol(base_cfg, extra_hashes):
    if set(extra_hashes) != set(EXTRA_SOURCE_PATHS):
        raise ValueError("Require exact additional source binding set")
    if any(extra_hashes[name] != expected for name, expected in CONTROL_DEPENDENCY_SHA256.items()):
        raise ValueError("Existing attribution control source drift")
    result = deepcopy(base_cfg)
    result.update(
        kind="biological_mechanism_controls_v1", hypothesis_id="H020",
        status="source_prepared_controls_not_executed",
        models=list(CONTROL_NAMES), model_specs=deepcopy(CONTROL_SPECS),
        prior_hypothesis_ids=["H004", "H005", "H006", "H020"],
        reference_config={"path": BASE_CONFIG_PATH, "sha256": BASE_CONFIG_SHA256},
        prepared_bank_policy="reuse_exact_v1_lossless_fixed_prepared_only_no_preparation",
        scientific_scope="development_attribution_not_confirmation_or_novelty",
        condition_registration="root_must_register_new_condition_reasons_before_execution",
        claims={"observed_preparation_performed_by_extension":False,
                "frozen_bank_reuse_required":True, "optimizer_fit_completed":False,
                "final_TEST_evaluation":False, "superiority_or_novelty_claim":False})
    result["source_sha256"].update(extra_hashes)
    return result


def load_protocol(config_path, expected_config_sha256, expected_base_config_sha256):
    if not expected_config_sha256 or frozen.sha(config_path) != expected_config_sha256:
        raise ValueError("Root-reviewed exact extension config SHA256 required")
    if expected_base_config_sha256 != BASE_CONFIG_SHA256:
        raise ValueError("Exact frozen H020 reference config SHA256 required")
    base_cfg, helpers = frozen.load_protocol(frozen.path(BASE_CONFIG_PATH), BASE_CONFIG_SHA256)
    cfg = json.loads(config_path.read_text())
    extra_hashes = {name: cfg.get("source_sha256", {}).get(name) for name in EXTRA_SOURCE_PATHS}
    if cfg != extension_protocol(base_cfg, extra_hashes):
        raise ValueError("Frozen biological extension method/configuration changed")
    frozen.verify_sources(cfg)
    return cfg, base_cfg, helpers


def require_output_namespace(output):
    """Prevent writes to any existing/frozen preparation or v1 output namespace."""
    resolved = output.resolve()
    resolved.relative_to(ROOT)
    results = ROOT / "results"
    if resolved.parent != results or not resolved.name.startswith("biological_mechanism_controls_v1_"):
        raise ValueError("Use a fresh direct results/biological_mechanism_controls_v1_* directory")
    if output.exists() or output.is_symlink():
        raise ValueError("Extension output must be fresh")


def verify_bindings(cfg, config_path, expected_config, manifest_path, expected_manifest,
                    prepared_path, prepared_sha):
    for file, expected in ((config_path, expected_config), (manifest_path, expected_manifest),
            (prepared_path, prepared_sha), (frozen.path(BASE_CONFIG_PATH), BASE_CONFIG_SHA256)):
        if not expected or frozen.sha(file) != expected:
            raise ValueError("Frozen extension/reference/observed/prepared binding changed")
    if cfg != json.loads(config_path.read_text()):
        raise ValueError("In-memory extension config differs from its bound bytes")
    frozen.verify_sources(cfg)


def validate_prepared_records(prepared, base_cfg, expected_manifest, parents):
    """Metadata-only admission, no bank extraction or truth capability."""
    if (prepared["status"] != "complete_observed_only_Q184_R96_actualQ120"
            or prepared["config_sha256"] != BASE_CONFIG_SHA256
            or prepared["observed_manifest_sha256"] != expected_manifest
            or prepared["source_sha256"] != base_cfg["source_sha256"]
            or prepared["truth_opened"] is not False):
        raise ValueError("Original observed prepared source/config/provenance binding differs")
    if len(parents) != len(frozen.PARENTS) or {p["parent_id"] for p in parents} != set(frozen.PARENTS):
        raise ValueError("Exactly the original observed parents are required")
    records = {r["parent_id"]: r for r in prepared["parents"]}
    if len(prepared["parents"]) != len(frozen.PARENTS) or set(records) != set(frozen.PARENTS):
        raise ValueError("Exactly the original prepared parents are required")
    for parent in parents:
        row = records[parent["parent_id"]]
        checks = (
            ("relate_entry", parent["observation"]["entry"]), ("split", parent["split"]),
            ("query_node_ids", parent["observation"]["nodes"][parent["query_slots"]].tolist()),
            ("dynamics_query_node_ids", parent["observation"]["nodes"][parent["dynamics_slots"]].tolist()),
            ("positions_bp", parent["positions"].tolist()), ("bin_edges_bp", parent["edges"].tolist()),
            ("actual_query_output_indexes", parent["output_indexes"].tolist()),
            ("actual_query_sample_indexes", parent["query_slots"].tolist()),
            ("dynamics_sample_indexes", parent["dynamics_slots"].tolist()),
            ("retained_fixed100bp_indexes", parent["retained_indexes"].tolist()))
        if any(row[key] != expected for key, expected in checks):
            raise ValueError("Prepared role/output/grid joins changed")
        for tile in row["tiles"]:
            frozen.checked(tile["bank"])
            frozen.checked(tile["whole_observed_graph"])
    return records


def seal_prediction(model, record, parent, cfg, output, storage, device, bindings):
    probabilities, seal = frozen.sealed_prediction(model, record, parent, cfg, output, storage, device)
    return probabilities, dict(seal, **bindings, model_configuration=model.configuration_manifest(),
        actual_coordinate_units="original physical bp/2000000")


from .biological_candidate import (
    sha, checked, write_json, fingerprint_sha, AccountedStorage, configure_fit, check_fit_lease,
    load_bank, optimizer_groups, ExternalTruthCapability, exact_probability_nll,
)

def train(cfg, base_cfg, helpers, config_path, expected_config, manifest_path, expected_manifest,
          prepared_path, prepared_sha, seed, output, snapshot_path, snapshot_sha, device_name="cuda"):
    if seed not in cfg["seeds"]:
        raise ValueError("Only frozen seeds101,202,303 are supported")
    require_output_namespace(output)
    verify_bindings(cfg, config_path, expected_config, manifest_path, expected_manifest,
                    prepared_path, prepared_sha)
    parents = frozen.observed_contract(cfg, manifest_path, expected_manifest, helpers)
    prepared = json.loads(prepared_path.read_text())
    records = validate_prepared_records(prepared, base_cfg, expected_manifest, parents)
    verify_bindings(cfg, config_path, expected_config, manifest_path, expected_manifest,
                    prepared_path, prepared_sha)
    prediction_bindings = dict(extension_config_sha256=expected_config,
        reference_config_sha256=BASE_CONFIG_SHA256, observed_manifest_sha256=expected_manifest,
        prepared_manifest_sha256=prepared_sha, source_sha256=cfg["source_sha256"])
    required = (cfg["storage"]["training_prediction_bytes_per_parent_epoch"]
                *2*cfg["training"]["epochs"]*len(cfg["models"])
                +cfg["storage"]["training_fixed_bytes"])
    storage = AccountedStorage.admit(snapshot_path,snapshot_sha,required,cfg)
    output.mkdir(exist_ok=False)
    device,gpu_runtime,lease=configure_fit(device_name,cfg,output)
    receipt = dict(kind="biological_mechanism_controls_fit_v1",hypothesis_id=cfg["hypothesis_id"],status="running",
                   seed=seed, config_sha256=expected_config, observed_manifest_sha256=expected_manifest,
                   prepared_manifest_sha256=prepared_sha, source_sha256=cfg["source_sha256"],
                   models={}, device=str(device), storage=storage.record(), TEST_parents=0,
                   GPU_qualification=gpu_runtime,
                   target_scope="actual query120 only; mixed64 latent outputs never labeled",
                   runtime={"torch":str(torch.__version__),"threads":torch.get_num_threads(),
                            "device":str(device),"feature_dtype":"float32","coordinate_dtype":"float64"})
    write_json(output/"receipt.json",receipt)
    states = {}
    for name in cfg["models"]:
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        model = model_factory(name,cfg).to(device)
        states[name] = dict(model=model,
            optimizer=torch.optim.AdamW(optimizer_groups(model,cfg),lr=cfg["training"]["learning_rate"]),
            best=float("inf"), best_epoch=None, stale=0, active=True, history=[],
            initial_weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},steps=0)
        receipt["models"][name] = dict(status="pending",parameters=sum(p.numel() for p in model.parameters()),
                                      configuration=model.configuration_manifest(),
                                      actual_coordinate_units="original physical bp/2000000",
                                      initial_parameter_fingerprint_sha256=fingerprint_sha(model.state_dict()),
                                      query_dynamics=184,reference_sources=96,actual_supervised_outputs=120,
                                      mixed_latent_outputs=64,mixed_query_feature_flag=0,
                                      optimizer_groups=[{"parameters":sum(p.numel() for p in g["params"]),
                                                        "weight_decay":g["weight_decay"]}
                                                        for g in optimizer_groups(model,cfg)])
    train_truth, val_truth = {}, {}
    train_parents = [p for p in parents if p["split"]=="train"]
    val_parents = [p for p in parents if p["split"]=="val"]
    for p in train_parents:
        capability = ExternalTruthCapability(p,cfg,"TRAIN_loss")
        train_truth[p["parent_id"]] = capability.target_mass(p,helpers)
    tiles = [(p,tile) for p in train_parents for tile in records[p["parent_id"]]["tiles"]]
    started = time.monotonic()
    try:
        for epoch in range(1,cfg["training"]["epochs"]+1):
            check_fit_lease(lease)
            verify_bindings(cfg, config_path, expected_config, manifest_path, expected_manifest,
                            prepared_path, prepared_sha)
            active = [n for n,s in states.items() if s["active"]]
            if not active:
                break
            order = list(range(len(tiles)))
            random.Random(seed+epoch).shuffle(order)
            losses = {n:[] for n in active}
            for index in order:
                parent,tile = tiles[index]
                bank = load_bank(tile,cfg,parent,device)
                positions = np.asarray(tile["position_indexes"],dtype=int)
                mass = torch.from_numpy(train_truth[parent["parent_id"]][positions]).to(device,dtype=torch.float32)
                outputs = torch.as_tensor(parent["output_indexes"],dtype=torch.long,device=device)
                for name in active:
                    state = states[name];model=state["model"];model.train()
                    state["optimizer"].zero_grad(set_to_none=True)
                    logits = model(bank).index_select(1,outputs)
                    if logits.shape != mass.shape:
                        raise ValueError("Loss must address exactly actual120 query outputs")
                    loss = -(torch.log_softmax(logits,dim=-1)*mass).sum()/mass.sum()
                    if not bool(torch.isfinite(loss)):
                        raise ValueError("Nonfinite exact TRAIN query-bp loss")
                    loss.backward()
                    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                        raise ValueError("Nonfinite candidate gradient")
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    state["optimizer"].step()
                    state["steps"]+=1
                    losses[name].append(float(loss.detach()))
                del bank
            for name in active:
                state=states[name];val_cells=[]
                for parent in val_parents:
                    pfile=output/f"{name}.epoch_{epoch:02d}.{parent['parent_id']}.probabilities.npz"
                    probabilities,sealed=seal_prediction(state["model"],records[parent["parent_id"]],
                                                           parent,cfg,pfile,storage,device,prediction_bindings)
                    if parent["parent_id"] not in val_truth:
                        val_truth[parent["parent_id"]]=ExternalTruthCapability(
                            parent,cfg,"VAL_after_sealed_prediction",sealed)
                    mass=val_truth[parent["parent_id"]].target_mass(parent,helpers)
                    primary=np.flatnonzero(parent["cohorts"]=="present_day")
                    nll,infinite=exact_probability_nll(probabilities,mass,primary)
                    val_cells.append(dict(parent_id=parent["parent_id"],nll=nll,
                                          infinite_nll_query_bp=infinite,sealed_prediction=sealed))
                mean_nll=float(np.mean([v["nll"] for v in val_cells]))
                row=dict(epoch=epoch, mean_parent_VAL_present64_NLL=mean_nll, VAL=val_cells,
                         mean_tile_TRAIN_query120_NLL=float(np.mean(losses[name])),
                         selection_scope="exact retained-center clipped-bin query-bp per parent; unweighted parent mean",
                         optimizer_steps=state["steps"],
                         maximum_parameter_delta_from_initial=max(float((v.detach().cpu()-state["initial_weights"][k]).abs().max())
                              for k,v in state["model"].state_dict().items() if v.is_floating_point()),
                         dynamic_controls=state["model"].configuration_manifest())
                state["history"].append(row)
                if mean_nll < state["best"]:
                    checkpoint=output/f"{name}.best_epoch_{epoch:02d}.pt"
                    torch.save({"model_state":state["model"].state_dict(),
                        "optimizer_state":state["optimizer"].state_dict(),"epoch":epoch,"seed":seed,
                        "config_sha256":expected_config,"observed_manifest_sha256":expected_manifest,
                        "prepared_manifest_sha256":prepared_sha,"selection_VAL_NLL":mean_nll,
                         "reference_config_sha256":BASE_CONFIG_SHA256,"source_sha256":cfg["source_sha256"],
                         "model_name":name,"condition":CONTROL_SPECS[name]},checkpoint)
                    storage.account(checkpoint)
                    state.update(best=mean_nll,best_epoch=epoch,stale=0,
                                 checkpoint={"path":str(checkpoint),"sha256":sha(checkpoint)})
                else:
                    state["stale"]+=1
                state["active"]=state["stale"]<cfg["training"]["patience"]
                receipt["models"][name].update(status="fitting",history=state["history"],
                    best_epoch=state["best_epoch"],best_mean_parent_VAL_present64_NLL=state["best"],
                    checkpoint=state.get("checkpoint"))
            verify_bindings(cfg, config_path, expected_config, manifest_path, expected_manifest,
                            prepared_path, prepared_sha)
            receipt["storage"]=storage.record()
            write_json(output/"receipt.json",receipt)
            print(json.dumps({"stage":"fit_epoch","seed":seed,"epoch":epoch,"active_models":active}),flush=True)
        for name,state in states.items():
            if state["best_epoch"] is None:
                raise ValueError("No finite VAL-selected checkpoint for "+name)
            saved=torch.load(checked(state["checkpoint"]),map_location="cpu",weights_only=False)
            if any(saved[key] != expected for key, expected in (
                    ("config_sha256", expected_config), ("reference_config_sha256", BASE_CONFIG_SHA256),
                    ("observed_manifest_sha256", expected_manifest), ("prepared_manifest_sha256", prepared_sha),
                    ("source_sha256", cfg["source_sha256"]), ("model_name", name),
                    ("condition", CONTROL_SPECS[name]), ("seed", seed), ("epoch", state["best_epoch"]))):
                raise ValueError("Selected control checkpoint identity differs")
            state["model"].load_state_dict(saved["model_state"])
            selected_delta=max(float((v.detach().cpu()-state["initial_weights"][k]).abs().max())
                               for k,v in state["model"].state_dict().items() if v.is_floating_point())
            if state["steps"]<=0 or not selected_delta>0:
                raise ValueError("Selected candidate checkpoint has no actual changed-weight optimizer fit")
            cells=[]
            for parent in val_parents:
                pfile=output/f"{name}.selected.{parent['parent_id']}.probabilities.npz"
                probabilities,sealed=seal_prediction(state["model"],records[parent["parent_id"]],
                                                       parent,cfg,pfile,storage,device,prediction_bindings)
                capability=val_truth[parent["parent_id"]]
                groups={"primary_present_day_64":np.flatnonzero(parent["cohorts"]=="present_day"),
                        "secondary_ancient_56":np.flatnonzero(parent["cohorts"]!="present_day"),
                        "all_queries_120":np.arange(120)}
                metrics={group:helpers.full_span_score(probabilities,parent,capability,columns,cfg["metrics"])
                         for group,columns in groups.items()}
                result=dict(parent_id=parent["parent_id"],split="val",seed=seed,model=name,
                    selected_epoch=state["best_epoch"],sealed_prediction=sealed,
                    external_truth_provenance=capability.provenance,C4_exact_support_metrics=metrics,
                    no_calibration_fit=True,mixed_latent_outputs_labeled=False)
                metric_path=output/f"{name}.{parent['parent_id']}.metrics.json"
                write_json(metric_path,result);storage.account(metric_path)
                cells.append({"parent_id":parent["parent_id"],"path":str(metric_path),"sha256":sha(metric_path)})
            receipt["models"][name].update(status="completed",metrics=cells,
                optimizer_fit_completed=True,optimizer_steps=state["steps"],
                selected_maximum_parameter_delta_from_initial=selected_delta,
                configuration=state["model"].configuration_manifest(),
                actual_coordinate_units="original physical bp/2000000")
        verify_bindings(cfg, config_path, expected_config, manifest_path, expected_manifest,
                        prepared_path, prepared_sha)
        check_fit_lease(lease)
        if device.type=="cuda":
            torch.cuda.synchronize()
            receipt["peak_cuda_allocated_bytes"]=torch.cuda.max_memory_allocated()
        receipt.update(status="completed_seven_controls_one_seed",elapsed_seconds=time.monotonic()-started,
                       storage=storage.record())
    except Exception as error:
        receipt.update(status="failed_partial_preserved",error_type=type(error).__name__,error=str(error),
                       elapsed_seconds=time.monotonic()-started,storage=storage.record())
        write_json(output/"receipt.json",receipt)
        raise
    write_json(output/"receipt.json",receipt)
    return receipt


def main():
    import runpy
    sys.modules.setdefault("arggnn.biological_mechanism_controls_v1", sys.modules[__name__])
    runpy.run_path(str(ROOT / "scripts/train_biological_mechanism_controls_v1.py"), run_name="__main__")


if __name__ == "__main__":
    main()
