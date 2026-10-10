"""Read-only bridge to the pinned author's ARGMix component; no reimplementation."""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

from .resources import ROOT

AUTHOR_COMMIT = "c2481248eb0de5fda659d68f59c51e0e27becd20"
AUTHOR_REPOSITORY = "https://github.com/AI-sandbox/argmix"
AUTHOR_ARCHIVE_SHA256 = "98c2cda680ca07c62723177599f2e7f37dc5b15d3ab05d2aa8c4a8f7f075747e"
AUTHOR_TREE_SHA256 = "4fbcc41801027983a9654920d143f053d5b74ffea020b751a13757c0ae327ae8"
AUTHOR_FILE_COUNT = 86
DEFAULT_AUTHOR_ROOT = ROOT / ".runtime/author_sources/argmix_c2481248eb0d"


def _canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha(content):
    return hashlib.sha256(content).hexdigest()


def _contained(path):
    path = Path(path)
    if ".." in path.parts:
        raise ValueError("Traversal is not an author-source path")
    if not path.is_absolute():
        path = ROOT / path
    try:
        relative = path.relative_to(ROOT)
    except ValueError as error:
        raise ValueError("Author source must remain inside the project") from error
    current = ROOT
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Symlinks are not admitted author-source paths")
    return path


@dataclass(frozen=True)
class OfficialARGMixSource:
    root: Path
    file_hashes: tuple[tuple[str, str], ...]

    def check(self):
        current = verify_author_source(self.root)
        if current.file_hashes != self.file_hashes:
            raise ValueError("Admitted author source changed")
        return self

    def reference(self, relative_path):
        hashes = dict(self.file_hashes)
        if relative_path not in hashes:
            raise ValueError("Unknown pinned author source")
        return {"path": str((self.root / relative_path).relative_to(ROOT)),
                "sha256": hashes[relative_path]}


def verify_author_source(author_root=DEFAULT_AUTHOR_ROOT):
    """Verify all 86 unchanged files before executing any upstream Python."""
    root = _contained(author_root)
    if not root.is_dir():
        raise ValueError("Pinned author-source directory is required")
    hashes = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Symlinks are not admitted inside author source")
        if path.is_file():
            hashes[str(path.relative_to(root))] = _sha(path.read_bytes())
    if len(hashes) != AUTHOR_FILE_COUNT or _sha(_canonical(hashes)) != AUTHOR_TREE_SHA256:
        raise ValueError("Pinned complete author-source inventory/hash differs")
    receipt = json.loads((root / "ARG_GNN_SOURCE_RECEIPT.json").read_text())
    if (receipt["repository"] != AUTHOR_REPOSITORY or receipt["commit"] != AUTHOR_COMMIT
            or receipt["archive_sha256"] != AUTHOR_ARCHIVE_SHA256):
        raise ValueError("Author-source retrieval receipt differs")
    return OfficialARGMixSource(root, tuple(sorted(hashes.items())))


def dependency_inventory():
    """Read installed metadata/specs only; do not install/import native pipelines."""
    modules = ("torch", "numpy", "yaml", "pandas", "polars", "h5py", "tskit",
               "ancpath", "maturin", "snakemake", "snputils", "transformers", "wandb")
    distributions = {"yaml": "PyYAML", "ancpath": "ancpath"}
    inventory = {}
    for name in modules:
        spec = importlib.util.find_spec(name)
        try:
            version = importlib.metadata.version(distributions.get(name, name))
        except importlib.metadata.PackageNotFoundError:
            version = None
        inventory[name] = {"available": spec is not None, "version": version,
                           "origin": None if spec is None else spec.origin}
    return {"python": sys.version.split()[0], "python_executable": sys.executable,
            "modules": inventory,
            "executables": {name: shutil.which(name) for name in
                            ("pixi", "cargo", "rustc", "maturin", "Relate", "snakemake", "bcftools", "plink2")}}


def inspect_official_source(author_root=DEFAULT_AUTHOR_ROOT):
    source = verify_author_source(author_root)
    model_path = source.root / "src/tmrca_attn.py"
    tree = ast.parse(model_path.read_text())
    for name in ("TmrcaAttn", "SpatialBiasAttention", "EncoderLayer"):
        if not any(isinstance(node, ast.ClassDef) and node.name == name for node in tree.body):
            raise ValueError("Native author architecture is absent")
    source.check()
    return {"kind": "official_argmix_source_preflight", "author_repository": AUTHOR_REPOSITORY,
            "author_commit": AUTHOR_COMMIT, "author_archive_sha256": AUTHOR_ARCHIVE_SHA256,
            "author_tree_sha256": AUTHOR_TREE_SHA256, "author_file_count": AUTHOR_FILE_COUNT,
            "author_source_root": str(source.root.relative_to(ROOT)),
            "native_component": source.reference("src/tmrca_attn.py"),
            "retrieval_receipt": source.reference("ARG_GNN_SOURCE_RECEIPT.json"),
            "dependencies": dependency_inventory(),
            "source_unchanged": True, "upstream_code_execution": False,
            "native_full_pipeline_executed": False, "native_model_smoke_executed": False,
            "paper_v1_reproduction_established": False, "scientific_readiness": False,
            "version_limit": "Current Python/Rust build path forbids ancestor features; current CLI trains transformer+GRU. Paper-v1 25-reference/3-ancestor execution needs version review.",
            "data_limit": "No real cohort/labels/checkpoints were supplied, generated or accessed.",
            "pipeline_limit": "Stock extraction/build/train/infer/export needs a reviewed contained native environment and build, biological input panel and split/evaluation contract."}


def native_model_smoke(author_root=DEFAULT_AUTHOR_ROOT):
    """Execute unchanged author's TmrcaAttn on synthetic CPU tokens, without fitting.

    This component smoke does not execute Rust extraction, the full joint GRU
    workflow, biological trees, labels, model training or a real checkpoint.
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Native component smoke requires explicitly hidden CUDA")
    source = verify_author_source(author_root)
    import torch
    import yaml
    config = yaml.safe_load((source.root / "templates/forward_blank/argmix.yaml").read_text())
    cfg = config["model"]
    classes = config["ancestry"]["class_labels"]
    special_node = next(node for node in ast.parse((source.root / "src/tree_sampler.py").read_text()).body
                        if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == "SPECIAL_TOKENS" for target in node.targets))
    special = ast.literal_eval(special_node.value)
    # These are synthetic fixture IDs, not a biological token/feature extractor.
    vocabulary = special + sorted(set(config["ancestry"]["reference_pops"]))
    model_source = source.root / "src/tmrca_attn.py"
    module_name = "_arggnn_native_argmix_c2481248_tmrca"
    spec = importlib.util.spec_from_file_location(module_name, model_source)
    if spec is None or spec.loader is None:
        raise ValueError("Native source cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    previous_module = sys.modules.get(module_name)
    previous_bytecode = sys.dont_write_bytecode
    old_threads = torch.get_num_threads()
    try:
        sys.dont_write_bytecode = True
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        source.check()
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261010)
            kwargs = dict(vocab_size=len(vocabulary), num_classes=len(classes),
                          num_layer=int(cfg["num_layers"]), d_model=int(cfg["hidden_size"]),
                          num_heads=int(cfg["num_heads"]), dim_feedforward_mult=int(cfg["dim_feedforward_mult"]),
                          dropout=float(cfg["dropout_prob"]), attn_dropout=float(cfg["attn_dropout_prob"]),
                          max_rel_distance=int(cfg["max_rel_distance"]), gt_output_dim=int(cfg["gt_output_dim"]))
            model = module.TmrcaAttn(**kwargs).cpu().eval()
            nodes = 1 + int(cfg["num_refs"])
            ids = torch.full((2, nodes), vocabulary.index("<SAMPLE>"), dtype=torch.long)
            reference_ids = [vocabulary.index(pop) for pop in config["ancestry"]["reference_pops"]]
            for column in range(1, nodes):
                ids[:, column] = reference_ids[(column - 1) % len(reference_ids)]
            distances = torch.arange(nodes * nodes, dtype=torch.long).reshape(nodes, nodes)
            distances = (distances % (int(cfg["max_rel_distance"]) + 1)).repeat(2, 1, 1)
            distances.diagonal(dim1=1, dim2=2).zero_()
            mask = torch.ones_like(ids, dtype=torch.bool)
            with torch.no_grad():
                logits = model(ids, distances, mask)
                compiled = torch.jit.script(model)
                scripted = compiled(ids, distances, mask)
            if not torch.isfinite(logits).all() or not torch.equal(logits, scripted):
                raise ValueError("Native CPU/TorchScript component output differs or is nonfinite")
            if any(parameter.device.type != "cpu" for parameter in model.parameters()):
                raise ValueError("Native smoke escaped CPU")
            result = {"kind": "official_argmix_native_component_cpu_smoke",
                      "author_commit": AUTHOR_COMMIT, "author_tree_sha256": AUTHOR_TREE_SHA256,
                      "native_component": source.reference("src/tmrca_attn.py"),
                      "native_class": "TmrcaAttn", "native_constructor": kwargs,
                      "model_parameter_count": sum(p.numel() for p in model.parameters()),
                      "input_ids_shape": list(ids.shape), "tmrca_shape": list(distances.shape),
                      "mask_shape": list(mask.shape), "raw_logits_shape": list(logits.shape),
                      "raw_logits_dtype": str(logits.dtype),
                      "raw_logits_sha256": _sha(logits.contiguous().numpy().tobytes()),
                      "torchscript_exact_match": True, "finite": True,
                      "software": {"python": sys.version.split()[0], "torch": torch.__version__},
                      "synthetic_inputs_not_biological_extraction": True,
                      "untrained_random_weights": True, "optimization_performed": False,
                      "cuda_execution_performed": False, "source_unchanged": True,
                      "full_native_pipeline_executed": False, "real_data_or_labels_accessed": False,
                      "paper_v1_reproduction_established": False, "scientific_readiness": False}
        source.check()
        return result
    finally:
        torch.set_num_threads(old_threads)
        sys.dont_write_bytecode = previous_bytecode
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-source", default=str(DEFAULT_AUTHOR_ROOT))
    parser.add_argument("--native-model-smoke", action="store_true",
                        help="Execute only unchanged native TmrcaAttn on explicit synthetic CPU fixtures")
    args = parser.parse_args(argv)
    result = inspect_official_source(args.author_source)
    if args.native_model_smoke:
        result["native_model_smoke"] = native_model_smoke(args.author_source)
        result["native_model_smoke_executed"] = True
        result["upstream_code_execution"] = True
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False), flush=True)
