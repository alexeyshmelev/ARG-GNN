"""Per-ARG native ARGformer fitting and explicit C4 readout on observed inputs.

No truth loader exists here. GPU work is root-scheduled through the fixed native
interpreter. Source-default backbones/epochs/logical batches remain intact.
"""
from __future__ import annotations
import argparse
from contextlib import redirect_stdout
import hashlib
import importlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import pickle
import random
import sys
import time
import traceback

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
CONFIG = ROOT / "configs/official_argformer_biological_pilot_v1.json"
SCRIPT = ROOT / "src/arggnn/train_official_argformer_biological.py"
PREFIX = ROOT / ".runtime/conda/envs/argformer_native_v1"
SOURCE = ROOT / ".runtime/author_sources/argformer_f7545be7bca0"
HELPERS = ROOT / "scripts/qualify_official_argformer_runtime.py"
GUARD = ROOT / "src/arggnn/native_argformer_exec.py"
HELPERS_SHA = "8fe652f87848f4a296e3d88555998fb388f11ae5c04d0858095e70c34b10b31c"
GUARD_SHA = "84ed405379ac56511c2f4722c130e8b884226018bbf8e2949ea2bb4dceeb99d0"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def json_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def observed_path(value):
    path = Path(value)
    if ".." in path.parts or any("truth" in part.lower() or "oracle" in part.lower()
                                for part in path.parts):
        raise ValueError("Truth/oracle/traversal paths are forbidden")
    path = path if path.is_absolute() else ROOT / path
    if path.resolve() != path or not path.is_relative_to(ROOT):
        raise ValueError("Canonical observed project path required")
    return path


def load_frozen(path, digest, name):
    if sha(path) != digest:
        raise RuntimeError("Frozen runtime helper changed")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def policy(expected_sha=None):
    if expected_sha is not None and sha(CONFIG) != expected_sha:
        raise ValueError("Root-reviewed biological policy changed")
    config = json.loads(CONFIG.read_text())
    checks = {
        "observed-only unchanged source": config["source_unchanged"] is True and
            config["query_ancestry_targets_read"] is False,
        "fixed four-parent roster": config["parent_ids"] == [
            "bio_paths_pilot_train_000", "bio_paths_pilot_train_001",
            "bio_paths_pilot_val_000", "bio_paths_pilot_val_001"],
        "completed post-size producer": config["inferred_manifest_kind"] == "biological_paths_relate_v1" and
            config["inferred_manifest_status"] ==
            "four_development_parents_inferred_pre_and_post_five_iterations" and
            config["inferred_condition"] == "post_five_size_iterations_primary",
        "source commit": config["source_commit"] == "f7545be7bca0200fb39aeb7f94a8e5f0b6bf5596",
        "source epochs and seeds": config["pretraining"]["epochs"] == 5 and
            config["contrastive"]["epochs"] == 1 and config["pretraining"]["seed"] == 17 and
            config["contrastive"]["seed"] == 36,
        "source logical batches": config["pretraining"]["logical_batch"] == 512 and
            config["pretraining"]["microbatch"] == 64 and
            config["contrastive"]["logical_batch"] == config["contrastive"]["microbatch"] == 4096,
        "native tokenization and within-ARG split": config["sequence_length"] == 128 and
            config["skip_extant_tokens"] is True and config["path_split_seed"] == 42 and
            config["path_split_train_fraction"] == .9,
        "C4 observed reference panel": config["class_order"] == ["ANA", "CHG", "WHG", "EHG"] and
            config["pure_reference_haplotypes"] == 96 and config["query_haplotypes"] == 120 and
            config["total_observed_haplotypes"] == 280,
        "native contrastive execution": config["contrastive"]["temperature"] == .3 and
            config["contrastive"]["precision"] == "fp32" and
            config["contrastive"]["gradient_cache_chunk"] == 64 and
            config["contrastive"]["phantom_classifier_num_labels"] == 3 and
            config["contrastive"]["classifier_used_for_ancestry"] is False,
        "actual parity prerequisite": config["gradient_cache_parity"]["real_paths"] == 16 and
            config["gradient_cache_parity"]["chunk"] == 4 and
            config["gradient_cache_parity"]["evaluation_mode_dropout_disabled"] is True and
            config["gradient_cache_parity"]["required_before_cache_fit"] is True,
        "uncalibrated local readout": config["readout"]["temperature"] == .3 and
            config["readout"]["embedding_batch"] == 128 and config["readout"]["coordinate_chunk"] == 32 and
            config["readout"]["tail_extrapolation"] is False and
            config["readout"]["calibration_fitted"] is False,
    }
    for requirement, admitted in checks.items():
        if not admitted:
            raise ValueError("Biological policy differs: " + requirement)
    return config


def prepare_paths(manifest_path, parent_id, output, expected_manifest_sha, expected_config_sha):
    """Root-only CPU preparation from completed explicit inferred observations."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Observed path preparation is CPU-only")
    config = policy(expected_config_sha)
    if parent_id not in config["parent_ids"]:
        raise ValueError("Assigned four-parent pilot ID required")
    manifest_path = observed_path(manifest_path)
    if sha(manifest_path) != expected_manifest_sha:
        raise ValueError("Completed inference manifest changed")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("kind") != config["inferred_manifest_kind"] or
            manifest.get("status") != config["inferred_manifest_status"]):
        raise ValueError("All explicit completed inferred parents required")
    replicates = manifest["replicates"]
    if (len(replicates) != 4 or {row["parent_id"] for row in replicates} != set(config["parent_ids"]) or
            any(row["status"] != "inferred_both_conditions" or
                row.get("truth_or_oracle_read") is not False for row in replicates)):
        raise ValueError("All four unique completed observed-only parents required")
    entries = [row for row in replicates if row["parent_id"] == parent_id]
    if len(entries) != 1:
        raise ValueError("One explicit inferred parent mapping required")
    entry = entries[0]
    if entry["status"] != "inferred_both_conditions" or (
            entry["inferred_tree"]["condition"] != config["inferred_condition"]):
        raise ValueError("Completed post-five-iteration observed graph required")
    tree_path = observed_path(entry["inferred_tree"]["path"])
    roster_path = observed_path(entry["observed_sample_manifest"]["path"])
    provenance_path = observed_path(entry["completion_provenance"]["path"])
    source_roster = entry["observed_sample_manifest"]["source_manifest"]
    source_roster_path = observed_path(source_roster["path"])
    for path, field in ((tree_path, entry["inferred_tree"]),
                        (roster_path, entry["observed_sample_manifest"]),
                        (source_roster_path, source_roster),
                        (provenance_path, entry["completion_provenance"])):
        if sha(path) != field["sha256"]:
            raise ValueError("Observed graph/roster/completion provenance changed")
    rows = json.loads(roster_path.read_text())
    mapping_rows = entry["sample_index_to_inferred_tree_node_id"]
    if any(type(row["sample_index"]) is not int or type(row["inferred_tree_node_id"]) is not int
           for row in mapping_rows):
        raise ValueError("Typed explicit inferred sample mapping required")
    mapping = {row["sample_index"]: row["inferred_tree_node_id"] for row in mapping_rows}
    if (len(rows) != 280 or len(mapping_rows) != 280 or set(mapping) != set(range(280)) or
            len(set(mapping.values())) != 280 or
            entry["observed_sample_manifest"]["rows"] != 280):
        raise ValueError("Explicit unique mapping for all 280 observed haplotypes required")
    if [row["sample_index"] for row in rows] != list(range(280)):
        raise ValueError("Observed sample roster order differs")
    for row in rows:
        if (mapping[row["sample_index"]] != row["inferred_tree_node_id"] or
                row["observed_node_id"] != row["inferred_tree_node_id"]):
            raise ValueError("Actual inferred node mapping differs from roster")
        if row["role"] != "pure_source_reference" and row["observed_reference_class"] is not None:
            raise ValueError("Only declared pure refs may supply ancestry class labels")
    refs = [row for row in rows if row["role"] == "pure_source_reference"]
    queries = [row for row in rows if row["role"] == "query"]
    if (len(refs) != 96 or len(queries) != 120 or
            sum(row["role"] == "mixed_reference_only" for row in rows) != 64):
        raise ValueError("Fixed 96 pure references, 64 mixed references and 120 queries required")
    if entry["split"] != ("train" if "_train_" in parent_id else "val"):
        raise ValueError("Biological parent split differs")
    ref_classes = {row["inferred_tree_node_id"]: row["observed_reference_class"] for row in refs}
    if any(type(label) is not int or not 0 <= label < 4 for label in ref_classes.values()) or any(
            sum(label == klass for label in ref_classes.values()) != 24 for klass in range(4)):
        raise ValueError("Exactly24 observed pure reference haplotypes per C4 class required")
    coordinate = entry["coordinate_contract"]
    if (coordinate["original_parent_interval_bp0"] != [0, 2000000] or
            coordinate["normalization_length_bp"] != 2000000 or
            coordinate["fixed_grid"]["total"] != 20000 or
            coordinate["fixed_grid"]["center_formula_bp0"] != "50 + 100*grid_index"):
        raise ValueError("Original two-Mb physical coordinate contract required")
    grid_path = observed_path(coordinate["fixed_grid"]["path"])
    if sha(grid_path) != coordinate["fixed_grid"]["sha256"]:
        raise ValueError("Observed coordinate grid changed")
    import numpy as np
    import tskit
    from arggnn.official_argformer_bridge import extract_observed_paths, build_vocabulary
    with np.load(grid_path, allow_pickle=False) as archive:
        original_grid = archive["positions_bp"].copy()
    if not np.array_equal(original_grid, np.arange(50, 2000000, 100, dtype=np.float64)):
        raise ValueError("Actual fixed100bp center grid required")
    if any(type(value) is not int for value in coordinate["retained_fixed100bp_indexes"]):
        raise ValueError("Integer retained fixed-grid indexes required")
    indexes = np.asarray(coordinate["retained_fixed100bp_indexes"], dtype=np.int64)
    if (indexes.ndim != 1 or not len(indexes) or np.any(np.diff(indexes) <= 0) or
            np.any(indexes < 0) or np.any(indexes >= len(original_grid))):
        raise ValueError("Declared retained ordered coverage indexes required")
    original_positions = original_grid[indexes]
    offset = float(coordinate["local_to_original_offset_bp"])
    local_positions = original_positions - offset
    ts = tskit.load(tree_path)
    if set(ts.samples()) != set(mapping.values()):
        raise ValueError("Actual flagged inferred samples differ from explicit genotype mapping")
    if (ts.num_migrations or ts.num_individuals or ts.num_populations or ts.num_provenances or
            np.any(ts.tables.nodes.population != tskit.NULL) or
            np.any((ts.tables.nodes.flags != 0) & (ts.tables.nodes.flags != tskit.NODE_IS_SAMPLE))):
        raise ValueError("Only cleaned inferred sample flags/topology are admitted")
    if any(not math.isfinite(float(row["age_generations"])) or
           abs(ts.node(row["inferred_tree_node_id"]).time - float(row["age_generations"])) > 1e-6
           for row in rows):
        raise ValueError("Actual inferred sample ages differ from observed roster")
    coverage = coordinate["coverage_original_bp0"]
    if (len(coverage) != 2 or not all(math.isfinite(float(value)) for value in coverage) or
            not 0 <= float(coverage[0]) < float(coverage[1]) <= 2000000 or
            float(coverage[0]) != offset or not math.isclose(
            ts.sequence_length, float(coverage[1]) - offset, rel_tol=0, abs_tol=1e-7) or
            np.any(local_positions < 0) or np.any(local_positions >= ts.sequence_length)):
        raise ValueError("Local observed intervals/offset/coverage differ; no extrapolation")
    namespace = parent_id + ":" + entry["inferred_tree"]["sha256"]
    corpus = extract_observed_paths(ts, namespace=namespace,
                                   sample_ids=[mapping[row["sample_index"]] for row in rows])
    vocabulary = build_vocabulary(corpus).official_values()
    path_ids = list(range(len(corpus.paths)))
    random.Random(42).shuffle(path_ids)
    cut = int(.9 * len(path_ids))
    splits = {"train": path_ids[:cut], "val": path_ids[cut:]}
    output = observed_path(output)
    if output.exists() or not output.parent.is_dir():
        raise ValueError("Fresh prepared directory with existing parent required")
    output.mkdir()
    for split, ids in splits.items():
        for task in ("mlm", "contrastive"):
            selected = ids if task == "mlm" else [i for i in ids if corpus.paths[i][0] in ref_classes]
            directory = output / task / split
            directory.mkdir(parents=True)
            with (directory / ("raw_" + split + "_sequences_and_vocab.pkl")).open("xb") as stream:
                pickle.dump({"raw_sequences": [list(corpus.paths[i]) for i in selected],
                             "vocab": vocabulary}, stream, protocol=4)
            if task == "contrastive":
                with (directory / "labels.pkl").open("xb") as stream:
                    pickle.dump([ref_classes[corpus.paths[i][0]] for i in selected], stream, protocol=4)
            json_new(directory / "path_ids.json", selected)
    directory = output / "embedding/all"
    directory.mkdir(parents=True)
    with (directory / "raw_all_sequences_and_vocab.pkl").open("xb") as stream:
        pickle.dump({"raw_sequences": [list(path) for path in corpus.paths], "vocab": vocabulary},
                    stream, protocol=4)
    with (output / "corpus.pkl").open("xb") as stream:
        pickle.dump({"paths": corpus.paths, "occurrences": corpus.occurrences,
                     "namespace": namespace}, stream, protocol=4)
    query_nodes = [row["inferred_tree_node_id"] for row in queries]
    ref_nodes = [row["inferred_tree_node_id"] for row in refs]
    np.savez_compressed(output / "coordinate_path_joins.npz", grid_indexes=indexes,
                        original_positions_bp=original_positions, local_positions_bp=local_positions,
                        query_nodes=np.asarray(query_nodes), reference_nodes=np.asarray(ref_nodes),
                        reference_classes=np.asarray([ref_classes[node] for node in ref_nodes]),
                        query_path_ids=corpus.readout_path_ids(local_positions, query_nodes),
                        reference_path_ids=corpus.readout_path_ids(local_positions, ref_nodes))
    json_new(output / "observed_roster.json", rows)
    json_new(output / "observed_parent_entry.json", entry)
    counts = {task: {split: len(json.loads((output / task / split / "path_ids.json").read_text()))
                     for split in ("train", "val")} for task in ("mlm", "contrastive")}
    prepared = {"schema": "official_argformer_prepared_observed_arg_v1", "parent_id": parent_id,
                "parent_split": entry["split"], "namespace": namespace,
                "input_manifest_sha256": expected_manifest_sha, "policy_sha256": sha(CONFIG),
                "inferred_tree_sha256": sha(tree_path),
                "derived_inferred_roster_sha256": sha(roster_path),
                "source_observed_roster_sha256": sha(source_roster_path),
                "completion_provenance_sha256": sha(provenance_path),
                "grid_sha256": sha(grid_path), "coordinate_contract": coordinate,
                "path_counts": counts, "unique_full_paths": len(corpus.paths),
                "vocab_size": len(vocabulary["special"]) + len(vocabulary["node_id_tokens"]),
                "paths_without_ancestor": sum(len(path) < 2 for path in corpus.paths),
                "true_query_ancestry_read": False, "within_ARG_transductive": True,
                "data_preparation_tskit": tskit.__version__,
                "class_order": config["class_order"], "status": "prepared_observed_paths_only",
                "files_sha256": {str(path.relative_to(output)): sha(path)
                                 for path in sorted(output.rglob("*")) if path.is_file()}}
    json_new(output / "prepared_manifest.json", prepared)
    return prepared


def prepared_inputs(directory, expected_sha, config_sha):
    directory = observed_path(directory)
    path = directory / "prepared_manifest.json"
    if sha(path) != expected_sha:
        raise ValueError("Root-reviewed observed corpus manifest changed")
    prepared = json.loads(path.read_text())
    if (prepared["schema"] != "official_argformer_prepared_observed_arg_v1" or
            prepared["policy_sha256"] != config_sha or prepared["true_query_ancestry_read"]):
        raise ValueError("Prepared observed-only policy differs")
    for relative, digest in prepared["files_sha256"].items():
        if sha(observed_path(directory / relative)) != digest:
            raise ValueError("Prepared observed artifact changed: " + relative)
    return prepared


def cache_loss(base, z, labels):
    """Literal author inline contrastive algebra, with real author _similarity.

    A loss-only entry point is absent in the released source. This transcription
    is admitted only by a root-run native full-forward/gradient parity receipt.
    """
    import torch
    sim = base._similarity(z, z) / base.temperature
    size = z.shape[0]
    if size <= 1:
        return sim.new_zeros(()), sim.detach()
    eye = torch.eye(size, dtype=torch.bool, device=sim.device)
    no_self = sim.masked_fill(eye, float("-inf"))
    positives = labels.view(-1, 1).eq(labels.view(1, -1)) & (~eye)
    counts = positives.sum(dim=1)
    log_prob = sim - torch.logsumexp(no_self, dim=1, keepdim=True)
    per_anchor = -(log_prob * positives.float()).sum(dim=1) / torch.clamp(counts.float(), min=1.)
    valid = counts > 0
    if valid.any() and base.class_weights is not None:
        weights = torch.tensor([base.class_weights.get(int(label), 1.) for label in labels[valid]],
                               device=sim.device, dtype=per_anchor.dtype)
        weights = weights / weights.mean()
        loss = (per_anchor[valid] * weights).mean()
    elif valid.any():
        loss = per_anchor[valid].mean()
    else:
        loss = sim.new_zeros(())
    return loss, sim.detach()


def cache_adapter(native, chunk_size):
    """Standard GradCache: full logical negatives, two-pass chunk RNG replay."""
    import torch
    from composer import ComposerModel

    def rng():
        return torch.get_rng_state().clone(), torch.cuda.get_rng_state(0).clone()

    def restore(states):
        torch.set_rng_state(states[0]); torch.cuda.set_rng_state(states[1], 0)

    def encode(base, batch):
        return base.encode(batch["input_ids"], batch["attention_mask"], batch.get("token_type_ids"))

    def chunks(batch, size):
        return [{key: value[start:start + size] for key, value in batch.items()}
                for start in range(0, len(batch["labels"]), size)]

    class GradCache(torch.autograd.Function):
        @staticmethod
        def forward(ctx, model, batch, size, *parameters):
            ctx.model, ctx.parameters = model, parameters
            ctx.batches = chunks(batch, size)
            ctx.rng_states, ctx.embeddings = [], []
            for micro in ctx.batches:
                ctx.rng_states.append(rng())
                ctx.embeddings.append(encode(model.base, micro).detach())
            ctx.post_forward_rng = rng()
            with torch.enable_grad():
                z = torch.cat(ctx.embeddings).detach().requires_grad_(True)
                loss, sim = cache_loss(model.base, z, batch["labels"])
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite full-batch cached source loss")
                ctx.embedding_gradients = torch.autograd.grad(loss, z)[0].detach()
            ctx.mark_non_differentiable(sim)
            return loss.detach(), sim

        @staticmethod
        def backward(ctx, loss_gradient, ignored_sim_gradient):
            gradients = [None] * len(ctx.parameters)
            cursor = 0
            try:
                for micro, saved_rng, saved_z in zip(ctx.batches, ctx.rng_states, ctx.embeddings):
                    restore(saved_rng)
                    with torch.enable_grad():
                        z = encode(ctx.model.base, micro)
                        if not torch.equal(z.detach(), saved_z):
                            raise RuntimeError("Actual gradient-cache dropout replay differs")
                        dz = ctx.embedding_gradients[cursor:cursor + len(z)] * loss_gradient
                        current = torch.autograd.grad(z, ctx.parameters, grad_outputs=dz,
                                                      allow_unused=True)
                    cursor += len(z)
                    for index, value in enumerate(current):
                        if value is not None:
                            gradients[index] = value if gradients[index] is None else gradients[index] + value
            finally:
                restore(ctx.post_forward_rng)
            return (None, None, None, *gradients)

    class Adapter(ComposerModel):
        def __init__(self):
            super().__init__()
            self.native = native

        def forward(self, batch):
            if self.training and torch.is_grad_enabled():
                loss, sim = GradCache.apply(self.native, batch, chunk_size,
                                            *tuple(self.native.parameters()))
            else:
                with torch.no_grad():
                    z = torch.cat([encode(self.native.base, micro)
                                   for micro in chunks(batch, chunk_size)])
                    loss, sim = cache_loss(self.native.base, z, batch["labels"])
            return {"loss": loss, "sim": sim}

        def loss(self, outputs, batch):
            return outputs["loss"]

        def get_metrics(self, is_train):
            return self.native.get_metrics(is_train)

        def eval_forward(self, batch, outputs=None):
            return self.forward(batch) if outputs is None else outputs

        def update_metric(self, batch, outputs, metric):
            return self.native.update_metric(batch, outputs, metric)

    return Adapter(), GradCache


def object_tensor_manifest(value, helpers):
    """Hash optimizer/state tensors without dropping dtype or scalar values."""
    import torch
    if torch.is_tensor(value):
        return helpers.tensor_info(value)
    if isinstance(value, dict):
        return {str(key): object_tensor_manifest(item, helpers) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [object_tensor_manifest(item, helpers) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("Unsupported actual optimizer state value: " + type(value).__name__)


def native_runtime(output, guard, helpers):
    if Path(sys.prefix).resolve() != PREFIX or sys.version_info[:3] != (3, 11, 9):
        raise RuntimeError("Fixed admitted native interpreter required")
    title = guard.apply_and_validate_title()
    lease = guard.validate_lease()
    admission = json.loads((output / "exec_admission.json").read_text())
    if admission.get("lease") != lease or admission.get("implementation_sha256") != sha(SCRIPT):
        raise RuntimeError("Native exec admission differs")
    runtime = {"process_title": title, "lease": lease, "interpreter": sys.executable,
               "source": helpers.validate_source(), "actual_cuda": False,
               "environment_variant": "packaging_repairs_v2_official_FA2_wheel"}
    if os.environ.get("ARGGNN_BIOLOGICAL_NATIVE_CODE_SHA256") != sha(SCRIPT):
        raise RuntimeError("Root-scheduler exec code provenance required")
    import importlib.metadata
    for name, expected in helpers.EXPECTED.items():
        actual = importlib.metadata.version(name)
        if actual.split("+")[0] != expected:
            raise RuntimeError("Admitted actual native distribution changed: " + name)
    runtime["distributions"] = {item.metadata["Name"]: item.version
                               for item in importlib.metadata.distributions()}
    import torch
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one actual leased GPU required")
    anchor = torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    runtime["actual_cuda"] = True
    actual_uuid = helpers.actual_cuda_uuid(output, runtime)
    if actual_uuid != lease["canonical_uuid"]:
        raise RuntimeError("Actual driver context UUID differs from lease")
    helpers.nvtop_snapshot(output, lease, runtime)
    properties = torch.cuda.get_device_properties(0)
    runtime["device"] = {"name": properties.name, "compute_capability": [properties.major, properties.minor],
                         "actual_uuid": actual_uuid, "total_memory_bytes": properties.total_memory}
    runtime["torch"] = {"version": torch.__version__, "cuda": torch.version.cuda,
                        "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
                        "build": torch.__config__.show()}
    sys.path.insert(0, str(SOURCE))
    return runtime, anchor


def source_model(mode, prepared_dir, output, pretrained=None):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(SOURCE / "yamls" / (mode + ".yaml"))
    cfg.data_local = str(prepared_dir / ("mlm" if mode == "mlm" else "contrastive"))
    cfg.model.pretrained_model_name = str(output / "base_config")
    cfg.loggers = {}
    cfg.load_path = None
    cfg.autoresume = False
    cfg.run_name = "native_" + prepared_dir.name + "_" + mode
    cfg.save_folder = None
    if mode == "mlm":
        cfg.device_train_microbatch_size = 64
        module = importlib.import_module("main")
        model = module.build_model(cfg.model)
        loader = module.build_dataloader(cfg=cfg.train_loader, tokenizer=model.tokenizer,
                                        device_batch_size=512, device_microbatch_size=64)
        evaluation = module.build_dataloader(cfg=cfg.eval_loader, tokenizer=model.tokenizer,
                                            device_batch_size=512)
    else:
        cfg.model.pretrained_checkpoint = str(pretrained) if pretrained else None
        module = importlib.import_module("sequence_contrastive")
        model = module.build_contrastive_model(cfg.model)
        if model.base.class_weights is not None or model.base.backbone.config.num_labels != 3:
            raise RuntimeError("Source uniform fallback and unused three-class layer required")
        loader = module.build_contrastive_dataloader(cfg.train_loader, device_batch_size=4096)
        evaluation = module.build_contrastive_dataloader(cfg.eval_loader, device_batch_size=4096)
    return model, loader, evaluation, module, cfg


def assert_backbone(model, mode, helpers):
    backbone = model.model if mode == "mlm" else model.base.backbone
    layers = backbone.bert.encoder.layers
    cfg = backbone.config
    if len(layers) != 12 or cfg.hidden_size != 768 or cfg.intermediate_size != 3072 or (
            cfg.num_attention_heads != 12 or not cfg.use_fa2):
        raise RuntimeError("Actual full released source backbone required")
    attention = importlib.import_module("src.bert_layers.attention")
    rotary = importlib.import_module("src.bert_layers.rotary")
    if not attention.IMPL_USE_FLASH2 or attention.IMPL_USE_FLASH3:
        raise RuntimeError("Actual FA2-only backend required")
    for i, layer in enumerate(layers):
        expected = (-1, -1) if mode == "contrastive" or i % 3 == 0 else (64, 64)
        if type(layer.attn) is not attention.FlexBertUnpadRopeAttention or (
                type(layer.attn.rotary_emb) is not rotary.UnpaddedRotaryEmbedding or
                tuple(layer.attn.sliding_window) != expected):
            raise RuntimeError("Source layer/RoPE/window differs")
    if cfg.unpad_embeddings != (mode == "mlm"):
        raise RuntimeError("Released config separation differs")
    if mode == "mlm":
        losses = importlib.import_module("flash_attn.losses.cross_entropy")
        if type(backbone.loss_fn) is not losses.CrossEntropyLoss:
            raise RuntimeError("Actual source fused MLM CE required")
    for module in (attention, rotary, importlib.import_module("src.flex_bert")):
        helpers.module_origin(module, SOURCE)
    if mode == "contrastive" and (not model.base.normalize or model.base.temperature != .3 or
                                      cfg.pooling_type != "mean"):
        raise RuntimeError("Actual released normalized contrastive mean pooling required")
    return backbone


def audit_loader(loader, directory, path_ids):
    """Observe the source sampler's actual single index stream without reseeding."""
    from torch.utils.data import Sampler
    raw = loader.dataloader if hasattr(loader, "dataloader") else loader
    original = raw.batch_sampler.sampler
    class OrderTrace(Sampler):
        calls = 0
        def __iter__(self):
            values = [int(value) for value in iter(original)]
            record = {"sampler_class": type(original).__name__,
                      "source_sampler_epoch": int(getattr(original, "epoch", 0)),
                      "actual_dataset_indices": values,
                      "actual_global_path_ids": [path_ids[index] for index in values],
                      "batch_drop_last": raw.drop_last, "logical_batch": raw.batch_size}
            json_new(directory / ("epoch_order_" + str(self.calls) + ".json"), record)
            self.calls += 1
            return iter(values)
        def __len__(self):
            return len(original)
    raw.batch_sampler.sampler = OrderTrace()
    return raw


def audit_callback(output, mode, native, helpers):
    import torch
    from composer import Callback
    class Audit(Callback):
        updates = 0
        def before_train_batch(self, state, logger):
            labels = state.batch["labels"]
            if mode == "contrastive":
                self.label_counts = [int((labels == klass).sum()) for klass in range(4)]
                if labels.shape != (4096,) or sum(self.label_counts) != 4096:
                    raise RuntimeError("Full4096 pure-ref logical contrastive batch required")
            else:
                self.label_counts = None
            self.batch_labels_sha = helpers.tensor_info(labels)["sha256"]
        def after_forward(self, state, logger):
            if not torch.isfinite(state.outputs["loss"]):
                raise RuntimeError("Actual native training produced nonfinite loss")
        def after_backward(self, state, logger):
            gradients = [p.grad for p in native.parameters() if p.grad is not None]
            if not gradients or any(not torch.isfinite(value).all() for value in gradients):
                raise RuntimeError("Actual native gradients are not finite")
            if mode == "contrastive" and any(
                    p.grad is not None for p in native.base.backbone.classifier.parameters()):
                raise RuntimeError("Phantom classifier must remain unused")
        def after_train_batch(self, state, logger):
            self.updates += 1
            record = {"optimizer_update": self.updates,
                      "epoch": int(state.timestamp.epoch.value),
                      "last_microbatch_loss": float(state.outputs["loss"].detach()),
                      "actual_batch_labels_sha256": self.batch_labels_sha,
                      "actual_C4_label_counts": self.label_counts,
                      "learning_rates": [group["lr"] for optimizer in state.optimizers
                                         for group in optimizer.param_groups]}
            with (output / "updates.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
    return Audit()


def train_phase(mode, native, loader, evaluation, module, cfg, directory,
                helpers, path_ids, execution="full_batch"):
    import torch
    from composer import Trainer, Evaluator
    directory.mkdir()
    initial = helpers.state_manifest(native)
    json_new(directory / "initial_model_state.json", initial)
    json_new(directory / "resolved_model_config.json",
             (native.model if mode == "mlm" else native.base.backbone).config.to_dict())
    from omegaconf import OmegaConf
    json_new(directory / "resolved_source_invocation.json", OmegaConf.to_container(cfg, resolve=True))
    audit_loader(loader, directory, path_ids)
    trained_model = native
    if mode == "contrastive" and execution == "gradient_cache":
        trained_model, _ = cache_adapter(native, 64)
    optimizer = module.build_optimizer(cfg.optimizer, trained_model)
    scheduler = module.build_scheduler(cfg.scheduler)
    callbacks = [audit_callback(directory, mode, native, helpers)]
    algorithms = [module.build_algorithm(name, settings)
                  for name, settings in cfg.get("algorithms", {}).items()]
    trainer = Trainer(model=trained_model, train_dataloader=loader,
                      eval_dataloader=Evaluator(label="eval", dataloader=evaluation,
                                                device_eval_microbatch_size=cfg.device_eval_microbatch_size),
                      seed=cfg.seed, run_name=cfg.run_name, device="gpu",
                      optimizers=optimizer, schedulers=scheduler, algorithms=algorithms,
                      callbacks=callbacks, loggers=[], precision=cfg.precision,
                      max_duration=cfg.max_duration, eval_interval=cfg.eval_interval,
                      eval_subset_num_batches=cfg.eval_subset_num_batches,
                      device_train_microbatch_size=cfg.device_train_microbatch_size,
                      progress_bar=False, log_to_console=True, autoresume=False)
    trainer.fit()
    checkpoint = directory / "composer_final.pt"
    trainer.save_checkpoint(str(checkpoint))
    final = helpers.state_manifest(native)
    if initial["sha256"] == final["sha256"] or callbacks[0].updates < 1:
        raise RuntimeError("Actual native optimizer updates required")
    json_new(directory / "final_model_state.json", final)
    optimizer_manifest = object_tensor_manifest(optimizer.state_dict(), helpers)
    json_new(directory / "optimizer_state_manifest.json", optimizer_manifest)
    record = {"mode": mode, "execution": execution, "optimizer_class": type(optimizer).__name__,
              "actual_optimizer_updates": callbacks[0].updates,
              "completed_epochs": int(trainer.state.timestamp.epoch.value),
              "initial_model_state_sha256": initial["sha256"],
              "final_model_state_sha256": final["sha256"], "checkpoint_sha256": sha(checkpoint),
              "optimizer_state_sha256": hashlib.sha256(canonical(optimizer_manifest)).hexdigest(),
              "backbone_unchanged": True, "true_query_ancestry_read": False}
    json_new(directory / "phase_receipt.json", record)
    if hasattr(trainer, "close"):
        trainer.close()
    return checkpoint, record


def memory_or_parity(action, native, loader, config, helpers, output, module, cfg, result):
    import torch
    raw = loader.dataloader if hasattr(loader, "dataloader") else loader
    if len(raw) < 1:
        raise RuntimeError("Not enough unique pure-ref training paths for source batch4096")
    batch = next(iter(raw))
    batch = {key: value.cuda() for key, value in batch.items()}
    native = native.cuda()
    before = helpers.state_manifest(native)["sha256"]
    if action == "probe-memory":
        native.train()
        snapshot = {name: value.detach().cpu().clone() for name, value in native.state_dict().items()}
        optimizer = module.build_optimizer(cfg.optimizer, native)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        result["full_batch_probe_attempted"] = True
        result["full_batch_probe_logical_batch"] = 4096
        loss = native(batch)["loss"]
        if not torch.isfinite(loss):
            raise RuntimeError("Source memory-probe loss is nonfinite")
        loss.backward()
        gradients = [parameter.grad for parameter in native.parameters() if parameter.grad is not None]
        if not gradients or any(not torch.isfinite(value).all() for value in gradients):
            raise RuntimeError("Source memory-probe gradients are not finite")
        if any(parameter.grad is not None for parameter in native.base.backbone.classifier.parameters()):
            raise RuntimeError("Source memory-probe phantom classifier is unexpectedly used")
        optimizer.step()
        torch.cuda.synchronize()
        optimizer_manifest = object_tensor_manifest(optimizer.state_dict(), helpers)
        json_new(output / "probe_optimizer_state_manifest.json", optimizer_manifest)
        updated_state = helpers.state_manifest(native)["sha256"]
        if updated_state == before or not optimizer.state:
            raise RuntimeError("Actual native optimizer state/update required for admission")
        record = {"action": action, "logical_batch": 4096, "execution": "actual_author_full_batch",
                  "loss": float(loss.detach()), "peak_memory_bytes": torch.cuda.max_memory_allocated(),
                  "native_optimizer_class": type(optimizer).__name__, "optimizer_step": True,
                  "temporary_updated_state_sha256": updated_state,
                  "native_optimizer_state_sha256": hashlib.sha256(canonical(optimizer_manifest)).hexdigest(),
                  "probe_is_not_a_biological_fit": True, "actual_class_counts":
                  [int((batch["labels"] == klass).sum()) for klass in range(4)]}
        native.load_state_dict(snapshot, strict=True)
        optimizer.zero_grad(set_to_none=True)
        record["temporary_probe_weights_restored"] = True
    else:
        n = config["gradient_cache_parity"]["real_paths"]
        batch = {key: value[:n] for key, value in batch.items()}
        native.eval()
        parameters = tuple(native.parameters())
        full_loss = native(batch)["loss"]
        full_grad = torch.autograd.grad(full_loss, parameters, allow_unused=True)
        _, function = cache_adapter(native, config["gradient_cache_parity"]["chunk"])
        cache_value, _ = function.apply(native, batch, config["gradient_cache_parity"]["chunk"], *parameters)
        cache_grad = torch.autograd.grad(cache_value, parameters, allow_unused=True)
        difference_sq, norm_sq = 0., 0.
        for first, second in zip(full_grad, cache_grad):
            if (first is None) != (second is None):
                raise RuntimeError("Full/cache parameter loss participation differs")
            if first is not None:
                difference_sq += float((first.double() - second.double()).square().sum())
                norm_sq += float(first.double().square().sum())
        relative = math.sqrt(difference_sq / max(norm_sq, 1e-30))
        loss_error = abs(float(full_loss.detach() - cache_value.detach()))
        if relative > config["gradient_cache_parity"]["global_gradient_relative_l2_tolerance"] or (
                loss_error > config["gradient_cache_parity"]["loss_absolute_tolerance"]):
            raise RuntimeError("Actual author full/cache loss-gradient parity failed")
        record = {"action": action, "status": "passed", "actual_native_encoder_and_loss": True,
                  "real_observed_paths": n, "dropout_disabled_for_full_batch_parity": True,
                  "loss_absolute_error": loss_error, "global_gradient_relative_l2": relative,
                  "training_chunk_RNG_replay_check": "bitwise in custom backward",
                  "optimizer_step": False}
    if helpers.state_manifest(native)["sha256"] != before:
        raise RuntimeError("Probe/parity changed model weights")
    record["unchanged_model_state_sha256"] = before
    return record


def readout(native, prepared_dir, output, config, helpers):
    import numpy as np
    import torch
    from src.arg_data import ARGDataset
    dataset = ARGDataset(local=str(prepared_dir / "embedding"), split="all",
                         max_seq_len=128, skip_extant_tokens=True)
    native.cuda().eval()
    vectors = []
    with torch.no_grad():
        for start in range(0, len(dataset), 128):
            items = [dataset[index] for index in range(start, min(start + 128, len(dataset)))]
            ids = torch.tensor([item["input_ids"] for item in items], device="cuda")
            mask = torch.tensor([item["attention_mask"] for item in items], device="cuda")
            z = native.base.encode(ids, mask)
            if not torch.isfinite(z).all():
                raise RuntimeError("Native biological embedding contains nonfinite values")
            vectors.append(z.cpu())
    embeddings = torch.cat(vectors)
    torch.save({"embeddings": embeddings, "native_pooling": "trained_author_mean_head"},
               output / "native_path_embeddings.pt")
    with np.load(prepared_dir / "coordinate_path_joins.npz", allow_pickle=False) as archive:
        joins = {key: archive[key].copy() for key in archive.files}
    gpu_vectors = embeddings.cuda()
    scores, probabilities = [], []
    with torch.no_grad():
        for start in range(0, len(joins["grid_indexes"]), 32):
            q = gpu_vectors[torch.as_tensor(joins["query_path_ids"][start:start + 32], device="cuda")]
            r = gpu_vectors[torch.as_tensor(joins["reference_path_ids"][start:start + 32], device="cuda")]
            similarity = torch.bmm(q, r.transpose(1, 2)) / .3
            labels = torch.as_tensor(joins["reference_classes"], device="cuda")
            logits = torch.stack([torch.logsumexp(similarity[:, :, labels == klass], dim=-1) -
                                  math.log(int((labels == klass).sum())) for klass in range(4)], dim=-1)
            probs = torch.softmax(logits, dim=-1)
            if not torch.isfinite(logits).all() or not torch.isfinite(probs).all():
                raise RuntimeError("Downstream readout is nonfinite")
            scores.append(logits.cpu()); probabilities.append(probs.cpu())
    logits, probs = torch.cat(scores), torch.cat(probabilities)
    np.savez_compressed(output / "downstream_C4_local_readout.npz",
                        logits=logits.numpy(), probabilities=probs.numpy(),
                        grid_indexes=joins["grid_indexes"],
                        positions_bp=joins["original_positions_bp"],
                        local_positions_bp=joins["local_positions_bp"],
                        query_nodes=joins["query_nodes"], query_path_ids=joins["query_path_ids"],
                        reference_nodes=joins["reference_nodes"],
                        reference_path_ids=joins["reference_path_ids"])
    return {"native_embedding_sha256": helpers.tensor_info(embeddings)["sha256"],
            "downstream_logits_sha256": helpers.tensor_info(logits)["sha256"],
            "uncalibrated_probabilities_sha256": helpers.tensor_info(probs)["sha256"],
            "class_order": config["class_order"], "readout_temperature": .3,
            "native_classifier_used": False, "downstream_adapter": True,
            "external_VAL_calibration_pending": True, "no_tail_extrapolation": True,
            "normalization_length_bp": 2000000}


def verify_encoder_initializer(native, checkpoint, helpers):
    """Inspect every actual loaded encoder key, not the source permissive guard."""
    import torch
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    original = payload["state"]["model"]
    mapped = {key.removeprefix("model."): value for key, value in original.items()}
    actual = native.base.backbone.state_dict()
    missing, unequal, loaded = [], [], []
    for key in actual:
        if key.startswith("bert."):
            if key not in mapped:
                missing.append(key)
            elif not torch.equal(actual[key].detach().cpu(), mapped[key]):
                unequal.append(key)
            else:
                loaded.append(key)
    if missing or unequal:
        raise RuntimeError("Trained MLM encoder initializer not fully loaded: " + str((missing, unequal)))
    return {"all_encoder_keys_loaded_equal": loaded, "missing_encoder": missing,
            "unequal_encoder": unequal, "checkpoint_sha256": sha(checkpoint),
            "source_checkpoint_only_keys": sorted(set(mapped) - set(actual)),
            "new_classification_backbone_keys": sorted(set(actual) - set(mapped)),
            "author_head_key_overlap_preserved": sorted(key for key in set(actual) & set(mapped)
                                                       if key.startswith("head."))}


def execute_native(args, output, guard, helpers, result):
    config = policy(args.expected_config_sha256)
    prepared_dir = observed_path(args.prepared_dir)
    prepared = prepared_inputs(prepared_dir, args.expected_prepared_sha256, sha(CONFIG))
    runtime, anchor = native_runtime(output, guard, helpers)
    result.update(action=args.action, parent_id=prepared["parent_id"],
                  namespace=prepared["namespace"], runtime=runtime,
                  config_sha256=sha(CONFIG), prepared_manifest_sha256=args.expected_prepared_sha256,
                  implementation_sha256=sha(SCRIPT), true_query_ancestry_read=False,
                  fitting_regime="within_ARG_transductive", calibration_pending_external_VAL=True,
                  root_actual_execution=True)
    json_new(output / "runtime_start.json", runtime)
    base = output / "base_config"
    base.mkdir()
    if sha(helpers.BASE) != helpers.BASE_SHA256:
        raise RuntimeError("Frozen BERT base configuration changed")
    (base / "config.json").write_bytes(helpers.BASE.read_bytes())
    import torch
    from composer.utils import reproducibility
    if prepared["path_counts"]["contrastive"]["train"] < 4096:
        raise RuntimeError("Observed pure-ref corpus has fewer than4096 paths; source drop-last yields no update")
    if args.action != "fit":
        reproducibility.seed_all(36)
        native, loader, _, module, cfg = source_model("contrastive", prepared_dir, output)
        assert_backbone(native, "contrastive", helpers)
        result["result"] = memory_or_parity(args.action, native, loader, config, helpers, output, module, cfg, result)
        result["optimizer_step"] = args.action == "probe-memory"
    else:
        if prepared["path_counts"]["mlm"]["train"] < 512 or not prepared["path_counts"]["mlm"]["val"]:
            raise RuntimeError("Observed corpus is insufficient for source MLM batch/split")
        if args.contrastive_execution == "gradient_cache":
            parity_path = observed_path(args.cache_parity_receipt)
            if sha(parity_path) != args.expected_cache_parity_sha256:
                raise RuntimeError("Root-reviewed actual cache parity receipt changed")
            parity = json.loads(parity_path.read_text())
            if (parity.get("status") != "passed" or parity.get("action") != "cache-parity" or
                    parity.get("prepared_manifest_sha256") != args.expected_prepared_sha256 or
                    parity.get("implementation_sha256") != sha(SCRIPT) or
                    parity.get("result", {}).get("status") != "passed" or
                    not parity.get("runtime", {}).get("actual_cuda")):
                raise RuntimeError("Actual native same-ARG author loss/gradient parity required before cache fit")
            memory_path = observed_path(args.full_batch_memory_receipt)
            if sha(memory_path) != args.expected_memory_receipt_sha256:
                raise RuntimeError("Actual full-batch memory evidence changed")
            memory = json.loads(memory_path.read_text())
            if (memory.get("action") != "probe-memory" or memory.get(
                    "prepared_manifest_sha256") != args.expected_prepared_sha256 or
                    memory.get("implementation_sha256") != sha(SCRIPT) or
                    memory.get("config_sha256") != sha(CONFIG) or
                    memory.get("status") not in {"passed", "failed"} or
                    not memory.get("runtime", {}).get("actual_cuda") or
                    not memory.get("full_batch_probe_attempted")):
                raise RuntimeError("Real same-ARG source4096 optimizer-memory attempt required")
            if memory.get("status") == "failed":
                detail = memory.get("failure", {})
                if "out of memory" not in detail.get("message", "").lower():
                    raise RuntimeError("Cache cannot silently correct a non-memory source failure")
            result["gradient_cache_admission"] = {"parity_receipt_sha256": sha(parity_path),
                                                  "full_batch_memory_receipt_sha256": sha(memory_path),
                                                  "logical_batch_unchanged": 4096}
        reproducibility.seed_all(17)
        mlm, loader, evaluation, module, cfg = source_model("mlm", prepared_dir, output)
        assert_backbone(mlm, "mlm", helpers)
        mlm_ids = json.loads((prepared_dir / "mlm/train/path_ids.json").read_text())
        mlm_checkpoint, mlm_record = train_phase(
            "mlm", mlm, loader, evaluation, module, cfg, output / "mlm", helpers, mlm_ids)
        result["pretraining"] = mlm_record
        del mlm, loader, evaluation
        import gc
        gc.collect(); torch.cuda.empty_cache()
        reproducibility.seed_all(36)
        captured = io.StringIO()
        with redirect_stdout(captured):
            native, loader, evaluation, module, cfg = source_model(
                "contrastive", prepared_dir, output, pretrained=mlm_checkpoint)
        (output / "contrastive_construction.stdout").write_text(captured.getvalue())
        assert_backbone(native, "contrastive", helpers)
        result["encoder_initialization"] = verify_encoder_initializer(native, mlm_checkpoint, helpers)
        classifier_before = helpers.state_manifest(native.base.backbone.classifier)["sha256"]
        ids = json.loads((prepared_dir / "contrastive/train/path_ids.json").read_text())
        _, contrastive_record = train_phase(
            "contrastive", native, loader, evaluation, module, cfg,
            output / "contrastive", helpers, ids, execution=args.contrastive_execution)
        if helpers.state_manifest(native.base.backbone.classifier)["sha256"] != classifier_before:
            raise RuntimeError("Unused phantom classifier unexpectedly changed")
        result["contrastive"] = contrastive_record
        result["classifier_unchanged_unused"] = True
        source_checkpoint = output / "contrastive_source_compatible.pt"
        torch.save({"state": {"model": {"model." + key: value.detach().cpu()
                                       for key, value in native.base.backbone.state_dict().items()}},
                    "arggnn_namespace": prepared["namespace"],
                    "prepared_manifest_sha256": args.expected_prepared_sha256,
                    "resolved_model_config": native.base.backbone.config.to_dict()},
                   source_checkpoint)
        result["source_compatible_checkpoint_sha256"] = sha(source_checkpoint)
        result["readout"] = readout(native, prepared_dir, output, config, helpers)
    if guard.validate_lease() != runtime["lease"]:
        raise RuntimeError("Actual scheduler reservation changed")
    prepared_inputs(prepared_dir, args.expected_prepared_sha256, sha(CONFIG))
    helpers.validate_source()
    result.update(status="passed", published_native_task_reproduction=False,
                  root_actual_execution=True, exit_code=0)
    del anchor
    return result


def bootstrap(args, guard, admission):
    output = guard.project_path(args.output_dir)
    if output.exists() or not output.parent.is_dir():
        raise RuntimeError("Fresh biological run directory with existing parent required")
    output.mkdir()
    admission["owns_output"] = True
    if Path(__file__).resolve() != SCRIPT or SCRIPT.is_symlink():
        raise RuntimeError("Fixed biological runner source required")
    lease = guard.validate_lease()
    guard.apply_and_validate_title()
    from arggnn.resources import require_storage
    require_storage(headroom_bytes=5_000_000_000)
    if (PREFIX.resolve() != PREFIX or not (PREFIX / "bin/python").is_file() or
            not (PREFIX / "bin/python").resolve().is_relative_to(PREFIX / "bin")):
        raise RuntimeError("Fixed isolated ARGformer interpreter required")
    env = os.environ.copy()
    env.update(PYTHONPATH="", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", WANDB_MODE="disabled",
               ARGGNN_BIOLOGICAL_NATIVE_CODE_SHA256=sha(SCRIPT))
    env.pop("PYTHONHOME", None)
    env["PATH"] = str(PREFIX / "bin") + os.pathsep + env.get("PATH", "")
    for name, relative in {
        "TMPDIR": "tmp", "XDG_CACHE_HOME": "cache", "XDG_CONFIG_HOME": "config",
        "XDG_DATA_HOME": "data", "HF_HOME": "cache/huggingface",
        "TORCH_HOME": "cache/torch", "TRITON_CACHE_DIR": "cache/triton",
        "CUDA_CACHE_PATH": "cache/cuda", "TORCHINDUCTOR_CACHE_DIR": "cache/torchinductor",
        "NUMBA_CACHE_DIR": "cache/numba", "MPLCONFIGDIR": "config/matplotlib",
        "WANDB_DIR": "logs/wandb", "WANDB_CACHE_DIR": "cache/wandb",
        "WANDB_CONFIG_DIR": "config/wandb"}.items():
        directory = output / relative
        directory.mkdir(parents=True, exist_ok=True)
        env[name] = str(directory)
    os.set_inheritable(lease["lease_fd"], True)
    if not os.get_inheritable(lease["lease_fd"]) or guard.validate_lease() != lease:
        raise RuntimeError("Lease inheritance admission failed")
    json_new(output / "exec_admission.json", {"lease": lease, "implementation_sha256": sha(SCRIPT),
                                            "action": args.action, "pid": os.getpid()})
    argv = [str(PREFIX / "bin/python"), "-B", str(SCRIPT), "--native", *sys.argv[1:]]
    os.execve(PREFIX / "bin/python", argv, env)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("fit", "probe-memory", "cache-parity"), required=True)
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--expected-prepared-sha256", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--contrastive-execution", choices=("full_batch", "gradient_cache"),
                        default="full_batch")
    parser.add_argument("--cache-parity-receipt")
    parser.add_argument("--expected-cache-parity-sha256")
    parser.add_argument("--full-batch-memory-receipt")
    parser.add_argument("--expected-memory-receipt-sha256")
    parser.add_argument("--native", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.action == "fit" and args.contrastive_execution == "gradient_cache" and not all(
        (args.cache_parity_receipt, args.expected_cache_parity_sha256,
         args.full_batch_memory_receipt, args.expected_memory_receipt_sha256)):
        parser.error("Cache fit requires actual same-ARG parity and source4096 memory receipts")
    guard = load_frozen(GUARD, GUARD_SHA, "_frozen_argformer_lease_guard")
    if not args.native:
        admission = {"owns_output": False}
        try:
            bootstrap(args, guard, admission)
            raise RuntimeError("Exec unexpectedly returned")
        except BaseException as error:
            output = guard.project_path(args.output_dir)
            if admission["owns_output"] and output.is_dir():
                json_new(output / "bootstrap_failure.json", {
                    "status": "bootstrap_failed", "error_type": type(error).__name__,
                    "error": str(error), "actual_training": False, "exit_code": 1})
                (output / "exit_status").write_text("1\n")
            raise
    output = guard.project_path(args.output_dir)
    if not (output / "exec_admission.json").is_file() or (output / "receipt.json").exists():
        raise RuntimeError("Fresh scheduler-admitted native output required")
    result = {"status": "failed", "action": args.action, "implementation_sha256": sha(SCRIPT),
              "prepared_manifest_sha256": args.expected_prepared_sha256,
              "root_actual_execution": False, "true_query_ancestry_read": False, "exit_code": 1}
    started = time.time()
    try:
        helpers = load_frozen(HELPERS, HELPERS_SHA, "_frozen_argformer_runtime_helpers")
        result = execute_native(args, output, guard, helpers, result)
    except BaseException as error:
        result["failure"] = {"type": type(error).__name__, "message": str(error),
                             "traceback": traceback.format_exc()}
        traceback.print_exc()
    finally:
        result["elapsed_seconds"] = time.time() - started
        result["artifacts_sha256"] = {str(path.relative_to(output)): sha(path)
            for path in sorted(output.rglob("*")) if path.is_file() and
            not set(path.relative_to(output).parts) & {"cache", "tmp"} and
            path.name not in {"receipt.json", "exit_status"}}
        json_new(output / "receipt.json", result)
        (output / "exit_status").write_text(str(result["exit_code"]) + "\n")
        print(json.dumps({"action": args.action, "status": result["status"],
                          "exit_code": result["exit_code"], "output": str(output)}), flush=True)
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
