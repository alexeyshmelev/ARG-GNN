"""Frozen train/validation-only fixed-anchor kernel GNN study."""
from __future__ import annotations
from .resources import ROOT, project_path, require_storage, set_research_title
set_research_title()
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time
import numpy as np
import torch
from .anchor_kernel import AnchorKernelModel, prepare_anchor_kernel_bank
from .age_dataset import load_age_caches
from .boundary_dataset import load_boundary_dataset
from .models import count_parameters
from .train import set_seed, dump_json
from .train_study import grid_sequence_metrics, sha
from .metrics import replicate_summary


def require_device(device):
    if device == "cuda":
        if os.environ.get("ARGGNN_PHYSICAL_GPU") not in {"2", "3", "4", "5"}:
            raise RuntimeError("Use the project GPU launcher")
        lease = project_path(os.environ["ARGGNN_GPU_LEASE"])
        if os.fstat(int(os.environ["ARGGNN_GPU_LEASE_FD"])).st_ino != lease.stat().st_ino:
            raise RuntimeError("GPU lease descriptor mismatch")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Must expose exactly one reserved GPU")
    elif torch.cuda.is_available():
        raise RuntimeError("CPU runs require CUDA_VISIBLE_DEVICES empty")


def checked_protocol(path):
    config = json.loads(path.read_text())
    if config["splits"].get("test", 0) != 0:
        raise RuntimeError("Mechanism runner does not load or score test data")
    for key in ("boundary_dataset_manifest_sha256", "boundary_preparation_sha256"):
        value = config.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise RuntimeError(f"Frozen SHA256 required: {key}")
    if config.get("selection") != "minimum mean parent-validation NLL":
        raise RuntimeError("Declare parent-validation NLL selection")
    return config


def evaluate(model, cases, device_banks):
    model.eval()
    records, probabilities = [], []
    with torch.no_grad():
        for case in cases:
            parent = None
            seen = torch.zeros(case.labels.shape[0], dtype=torch.bool)
            for tile in case.tiles:
                bank = device_banks[id(tile)]
                p = model(bank).softmax(-1).cpu()
                indexes = tile.parent_position_indices
                if (indexes.dtype != torch.long or indexes.device.type != "cpu"
                    or indexes.ndim != 1 or indexes.numel() != p.shape[0]
                    or torch.unique(indexes).numel() != indexes.numel()
                    or bool(((indexes < 0) | (indexes >= case.labels.shape[0])).any())):
                    raise RuntimeError("Invalid parent fixed-coordinate mapping")
                if (not torch.equal(tile.labels, case.labels[indexes])
                    or not torch.equal(tile.bank.positions, case.positions[indexes])
                    or not torch.equal(tile.bank.query_node_ids, case.query_node_ids)):
                    raise RuntimeError("Tile targets differ from parent coordinate/query contract")
                if p.shape[:2] != tile.labels.shape or bool(seen[indexes].any()):
                    raise RuntimeError("Window predictions do not partition parent targets")
                if parent is None:
                    parent = p.new_empty((*case.labels.shape, p.shape[-1]))
                parent[indexes] = p
                seen[indexes] = True
            if parent is None or not bool(seen.all()):
                raise RuntimeError("Some parent fixed coordinates lack predictions")
            records.append({"replicate": case.metadata["replicate"],
                            **grid_sequence_metrics(parent, case.labels)})
            probabilities.append(parent)
    return records, probabilities




def checked_anchor_protocol(path):
    config = checked_protocol(path)
    models, specs = config.get("models"), config.get("model_specs")
    if (not isinstance(models, list) or not models or len(set(models)) != len(models)
            or not isinstance(specs, dict) or set(specs) != set(models)):
        raise RuntimeError("Require distinct frozen models and exact specification keys")
    if config.get("rate_weight_decay") != 0.:
        raise RuntimeError("Frozen anchor protocol requires zero rate/mixture weight decay")
    for name, spec in specs.items():
        if (set(spec) != {"geometry", "coupling", "kernel_depth", "rate_parameterization"}
                or spec["geometry"] not in {"plain", "age", "matched_scalar_age"}
                or spec["coupling"] not in {"full", "scalar", "diagonal", "gain_matched_diagonal"}
                or spec["kernel_depth"] not in {1,2}
                or spec["rate_parameterization"] not in {"log", "softplus"}):
            raise RuntimeError("Unsupported frozen anchor specification")
    for key in ("num_anchors", "num_scales", "num_classes", "epochs", "patience"):
        value = config.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise RuntimeError("Require positive frozen integer dimensions/budgets")
    if config["num_scales"] not in {1,3}:
        raise RuntimeError("Current prototype has one or three fixed branches")
    return config

def make_anchor_model(name, input_dim, num_classes, settings):
    specs = settings["model_specs"]
    if name not in specs:
        raise ValueError("Missing frozen anchor model specification")
    spec = dict(specs[name])
    return AnchorKernelModel(input_dim=input_dim, num_classes=num_classes,
        hidden_dim=settings["hidden_dim"], num_layers=settings["num_layers"],
        latent_dim=settings["latent_dim"], num_anchors=settings["num_anchors"],
        num_scales=settings["num_scales"],
        initial_coupling=settings["initial_coupling"],
        initial_damping=settings["initial_damping"], **spec)

def optimizer_groups(model, weight_decay):
    rate_parameters = list(model.rate_and_mixture_parameters())
    identities = {id(p) for p in rate_parameters}
    ordinary = [p for p in model.parameters() if id(p) not in identities]
    if len(identities) != len(rate_parameters) or not identities <= {id(p) for p in model.parameters()}:
        raise RuntimeError("Invalid/duplicate kernel rate parameter group")
    return [{"params": ordinary, "weight_decay": weight_decay},
            {"params": rate_parameters, "weight_decay": 0.}]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--age-data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    require_device(args.device)
    config_path, directory = project_path(args.config), project_path(args.data)
    settings = checked_anchor_protocol(config_path)
    manifest_path = directory / "manifest.json"
    if sha(manifest_path) != settings["boundary_dataset_manifest_sha256"]:
        raise RuntimeError("Boundary manifest differs from frozen study")
    dataset = load_boundary_dataset(directory,
        expected_manifest_sha256=settings["boundary_dataset_manifest_sha256"],
        expected_preparation_sha256=settings["boundary_preparation_sha256"],
        expected_source_manifest_sha256=settings["source_dataset_manifest_sha256"])
    cases = {split: dataset.cases(split) for split in ("train", "val")}
    age_caches = None
    if args.age_data:
        age_caches = load_age_caches(args.age_data, dataset,
            expected_manifest_sha256=settings["age_dataset_manifest_sha256"],
            expected_preparation_sha256=settings["age_preparation_sha256"])
    elif any(spec["geometry"] != "plain" for spec in settings["model_specs"].values()):
        raise RuntimeError("Frozen age modes require a verified age dataset")
    for split in cases:
        if any(c.metadata["num_classes"] != settings["num_classes"] for c in cases[split]):
            raise RuntimeError("Class count differs from parent metadata")
        if len(cases[split]) != settings["splits"][split]:
            raise RuntimeError("Independent parent count differs from frozen protocol")
    requested = args.models or settings["models"]
    if any(name not in settings["models"] for name in requested) or len(set(requested)) != len(requested):
        raise ValueError("Choose distinct models in the frozen protocol")
    output = project_path(args.output)
    if output.exists():
        raise RuntimeError("Choose a fresh run directory")
    require_storage(headroom_bytes=200_000_000)
    output.mkdir(parents=True)
    (output/"protocol.json").write_text(config_path.read_text())
    (output/"dataset_manifest.json").write_text(manifest_path.read_text())
    device = torch.device(args.device)
    started_transfer = time.monotonic()
    device_banks = {id(tile): prepare_anchor_kernel_bank(tile.bank,
        None if age_caches is None else age_caches[tile.provenance["tile_id"]],
        num_anchors=settings["num_anchors"], num_scales=settings["num_scales"]
        ).to(device, dtype=torch.float32)
        for group in cases.values() for case in group for tile in case.tiles}
    if device.type == "cuda":
        torch.cuda.synchronize()
    transfer_seconds = time.monotonic() - started_transfer
    tiles = [tile for case in cases["train"] for tile in case.tiles]
    example = cases["train"][0].tiles[0].bank
    config_sha = sha(config_path)
    code_paths = ["train_anchor.py", "anchor_kernel.py", "train_boundary.py",
        "boundary_memory.py", "boundary_dataset.py", "kron_interface.py",
        "models.py", "data.py", "reference_encoder.py", "genealogy_baselines.py",
        "simulate.py", "train_study.py", "train.py", "metrics.py", "resources.py"]
    if age_caches is not None:
        code_paths += ["age_interface.py", "age_dataset.py"]
    initial_code_hashes = {p: sha(ROOT/"src/arggnn"/p) for p in code_paths}
    results = {"stage": settings["stage"], "interpretation": settings["interpretation"],
        "evaluation_split": "validation_only", "training_seed": args.seed,
        "protocol_sha256": config_sha, "boundary_dataset_manifest_sha256": sha(manifest_path),
        "boundary_preparation_sha256": settings["boundary_preparation_sha256"],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "implementation_sha256": initial_code_hashes,
        "anchor_source_measure": "fixed equal-width physical midpoint anchors",
        "age_dataset_manifest_sha256": settings.get("age_dataset_manifest_sha256"),
        "rate_weight_decay": 0.,
        "physical_gpu": os.environ.get("ARGGNN_PHYSICAL_GPU"),
        "bank_transfer_seconds": transfer_seconds,
        "environment": {"torch": str(torch.__version__), "cuda": torch.version.cuda},
        "selection": settings["selection"], "models": {}}
    checkpoint_id = hashlib.sha256(str(output.relative_to(ROOT)).encode()).hexdigest()[:16]
    checkpoint_root = project_path(Path("checkpoints") / (output.name+"_"+checkpoint_id))
    checkpoint_root.mkdir(parents=True)
    for name in requested:
        set_seed(args.seed)
        model = make_anchor_model(name, example.leaf_x.shape[-1], settings["num_classes"], settings).to(device)
        optimizer = torch.optim.AdamW(optimizer_groups(model, settings["weight_decay"]),
                                      lr=settings["learning_rate"])
        initial_rates = [layer.rates.detach().clone() for layer in model.layers]
        initial_damping = [layer.damping.detach().clone() for layer in model.layers]
        best_nll, best_epoch, stale = float("inf"), -1, 0
        history = []
        checkpoint = checkpoint_root / (name+".pt")
        started = time.monotonic()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for epoch in range(settings["epochs"]):
            require_storage(headroom_bytes=100_000_000)
            model.train()
            order = list(range(len(tiles)))
            random.shuffle(order)
            losses = []
            gradient_norms = {"coupling": [], "damping": [], "mixture": []}
            for index in order:
                tile = tiles[index]
                logits = model(device_banks[id(tile)])
                y = tile.labels.to(device)
                loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Nonfinite training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                for key, attribute in (("coupling", "raw_coupling"),
                        ("damping", "raw_damping"), ("mixture", "mixture_logits")):
                    gradients = [getattr(layer, attribute).grad for layer in model.layers]
                    if any(grad is None or not bool(torch.isfinite(grad).all()) for grad in gradients):
                        raise RuntimeError("Missing or nonfinite kernel optimization gradient")
                    norm = torch.stack([grad.detach().square().sum() for grad in gradients]).sum().sqrt()
                    gradient_norms[key].append(float(norm))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                losses.append(float(loss.detach()))
            val, _ = evaluate(model, cases["val"], device_banks)
            nll = float(np.mean([r["nll_uniform_grid"] for r in val]))
            record = {"epoch": epoch, "train_nll": float(np.mean(losses)), "val_nll": nll,
                      "val_accuracy": float(np.mean([r["accuracy_uniform_grid"] for r in val])),
                      "elapsed_seconds": time.monotonic()-started}
            record["kernel_optimization"] = {
                "gradient_norm_before_clip_mean": {key: float(np.mean(values))
                    for key, values in gradient_norms.items()},
                "gradient_norm_before_clip_max": {key: max(values)
                    for key, values in gradient_norms.items()},
                "maximum_relative_rate_change": max(float((layer.rates.detach()/initial-1).abs().max())
                    for layer, initial in zip(model.layers, initial_rates)),
                "maximum_relative_damping_change": max(float((layer.damping.detach()/initial-1).abs().max())
                    for layer, initial in zip(model.layers, initial_damping))}
            history.append(record)
            print(json.dumps({"event": "epoch", "model": name, "seed": args.seed, **record}), flush=True)
            if nll < best_nll:
                best_nll, best_epoch, stale = nll, epoch, 0
                torch.save({"model": model.state_dict(), "epoch": epoch, "validation_nll": nll,
                            "protocol_sha256": config_sha,
                            "boundary_dataset_manifest_sha256": sha(manifest_path)}, checkpoint)
            else:
                stale += 1
                if stale >= settings["patience"]:
                    break
        chosen = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(chosen["model"])
        val, predictions = evaluate(model, cases["val"], device_banks)
        prediction_path = checkpoint_root / (name+"_validation_predictions.pt")
        torch.save({"probabilities": predictions,
                    "parent_replicates": [c.metadata["replicate"] for c in cases["val"]],
                    "positions": [c.positions for c in cases["val"]],
                    "query_node_ids": [c.query_node_ids for c in cases["val"]],
                    "protocol_sha256": config_sha}, prediction_path)
        result = {"parameters": count_parameters(model), "best_epoch": best_epoch,
                  "validation_nll": best_nll, "training_seconds": time.monotonic()-started,
                  "history": history, "per_replicate": val,
                  "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
                  "validation_predictions": str(prediction_path),
                  "validation_predictions_sha256": sha(prediction_path),
                  "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if device.type=="cuda" else None,
                  **replicate_summary(val, args.seed)}
        if initial_code_hashes != {p: sha(ROOT/"src/arggnn"/p) for p in code_paths}:
            raise RuntimeError("Active anchor implementation changed during training")
        result["model_configuration"] = model.configuration_manifest()
        results["models"][name] = result
        dump_json(output/(name+".json"), result)
        dump_json(output/"progress.json", results)
        print(json.dumps({"event": "model_complete", "model": name,
                          "validation_accuracy": result["mean_accuracy"]}), flush=True)
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
    dump_json(output/"results.json", results)
    print(json.dumps({"event": "run_complete", "output": str(output),
                      "storage_bytes": require_storage()}), flush=True)


if __name__ == "__main__":
    main()
