"""H038 shared full952 ARGformer current-release adapter; source-only default.

Every parent has its own observed node vocabulary and fresh native MLM+contrastive
fits. Root supplies storage/runtime/GPU admission; no GPU action at import.
Paper-original AmericanAdmixture4B18 reproduction is a separate benchmark.
"""
from __future__ import annotations
import argparse
import ast
import copy
import gc
import hashlib
import importlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE = "src/arggnn/expanded_native_argformer_v1.py"
AUTHOR = ".runtime/author_sources/argformer_f7545be7bca0"
NATIVE_PREFIX = ".runtime/conda/envs/argformer_native_v1"
AP_CONFIG = "research/reviews/expanded_native_ancestralpaths_v1_author/config.draft.json"
HELPERS = {
    "observed_reader": "src/arggnn/expanded_native_ancestralpaths_v1.py",
    "path_bridge": "src/arggnn/official_argformer_bridge.py",
    "fit_definitions": "src/arggnn/train_official_argformer_biological.py",
    "qualifier": "scripts/qualify_official_argformer_runtime.py",
    "process_title": "src/arggnn/native_argformer_exec.py",
}
CACHE_ENV = ("XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "HF_HOME",
    "TORCH_HOME", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH", "TORCHINDUCTOR_CACHE_DIR",
    "NUMBA_CACHE_DIR", "MPLCONFIGDIR", "WANDB_DIR", "WANDB_CACHE_DIR", "WANDB_CONFIG_DIR")


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _path(value):
    path = Path(value)
    if path.is_absolute():
        path = path.relative_to(ROOT)
    _require(bool(path.parts) and ".." not in path.parts, "Dedicated project artifact required")
    for n in range(1, len(path.parts) + 1):
        _require(not ROOT.joinpath(*path.parts[:n]).is_symlink(), "Project symlink forbidden")
    result = ROOT / path
    result.resolve().relative_to(ROOT)
    return result


def _ref(value):
    path = _path(value)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=h.hexdigest())


def _checked(ref):
    _require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}, "Exact artifact ref required")
    _require(ref == _ref(ref["path"]), "Bound artifact bytes differ")
    return _path(ref["path"])


def _read(ref):
    return json.loads(_checked(ref).read_text())


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _module(ref, name):
    spec = importlib.util.spec_from_file_location(name, _checked(ref))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _guard(guard, planned=0):
    _require(Path.cwd().resolve() == ROOT and callable(guard) and guard(int(planned)) is not False,
             "Root-admitted project write guard required")


def _directory(value, guard):
    output = _path(value)
    _guard(guard, 16384)
    output.mkdir(parents=True, exist_ok=False)
    return output


def _bytes(path, payload, guard):
    _guard(guard, len(payload) + 4096)
    with path.open("xb") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    return _ref(path)


def _json(path, value, guard):
    return _bytes(path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(), guard)


def _pickle(path, value, guard):
    # Only own primitive lists/tuples/dicts/ints are authored for native loaders.
    return _bytes(path, pickle.dumps(value, protocol=4), guard)


def _npz(path, arrays, guard):
    import numpy as np
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return _bytes(path, stream.getvalue(), guard)


def default_config_v1():
    return dict(schema="expanded_native_argformer_v1_config", hypothesis_id="H038",
        status="source_only_root_stage_admission_required", source_ref=_ref(SOURCE),
        author_root=AUTHOR, author_commit="f7545be7bca0200fb39aeb7f94a8e5f0b6bf5596",
        author_receipt_ref=_ref(AUTHOR + "/ARG_GNN_SOURCE_RECEIPT.json"),
        author_tree_sha256="2295d86ace359ac1a7d0522c1001b6f2822b30cf9b16200517ea3473f234ca5b",
        author_archive_ref=_ref(".runtime/author_sources/argformer_f7545be7bca0.tar.gz"),
        helper_refs={name: _ref(path) for name, path in HELPERS.items()},
        observed_reader_config_ref=_ref(AP_CONFIG),
        mlm_yaml_ref=_ref(AUTHOR + "/yamls/mlm.yaml"),
        contrastive_yaml_ref=_ref(AUTHOR + "/yamls/contrastive.yaml"),
        retrieval_source_ref=_ref(AUTHOR + "/retrieve.py"),
        base_config_ref=_ref(".runtime/tmp/official_argformer_native_environment_candidate_v1/hf_base_config/config.json"),
        native_prefix=NATIVE_PREFIX, class_order=["Ana", "CHG", "WHG", "EHG"],
        split=dict(TRAIN=32, INTERNAL_VAL=8, outer_prepared=0),
        within_ARG_transductive=True, cross_ARG_checkpoint_transfer=False, query_truth_used=False,
        paths=dict(split_seed=42, train_fraction=.9, max_seq_len=128, skip_extant=True,
                   deduplicate_full_paths_before_tokenization=True, all952_MLM=True, pure266_contrastive=True),
        MLM=dict(epochs=5, seed=17, logical_batch=512, microbatch=512, precision="amp_bf16"),
        contrastive=dict(epochs=1, seed=36, logical_batch=4096, microbatch=4096, precision="fp32",
                         temperature=.3, source_uniform_weight_fallback=True, unused_classifier_classes=3),
        retrieval=dict(primary_k=1, sensitivity_k=5, q_batch=1024, e_chunk=100000,
            embedding_batch=128, pooling="trained_author_mean_head", source_torch_topk_ties=True,
            all_reference_paths_train_plus_val=True, no_max_samples_truncation=True,
            hard_votes_no_probability_floor=True),
        published_original_task="AmericanAdmixture4B18_1600samples_30Mb_3sources_tsinfer_tsdate",
        shared_task="expanded_Relate_full952_2Mb_4sources_actual_ages",
        paper_original_numerical_reproduction=False, release_algorithm_fidelity=True)


def _policy(config):
    _require(config == default_config_v1(), "Frozen release/shared-panel policy and source refs required")
    # Inventory is the pinned 60-file release, including original license notices.
    records = []
    for path in sorted(_path(AUTHOR).rglob("*")):
        _require(not path.is_symlink(), "Author archive symlink forbidden")
        if path.is_file() and path.name != "ARG_GNN_SOURCE_RECEIPT.json":
            records.append(dict(path=path.relative_to(_path(AUTHOR)).as_posix(), sha256=_ref(path)["sha256"]))
    _require(len(records) == 60 and _digest(records) == config["author_tree_sha256"],
             "Unchanged pinned native release inventory required")
    reader_cfg = _read(config["observed_reader_config_ref"])
    registry = _read(reader_cfg["registration_ref"])["parent_registry"]
    roles = {r["parent_id"]: r["split"] for r in registry}
    _require([list(roles.values()).count(s) for s in ("train", "val", "outer")] == [32, 8, 8],
             "Exact expanded32TRAIN/8VAL registry required")
    return reader_cfg, roles


def _joins(corpus, positions, nodes):
    """Exact occurrence projection with interval slices, no full-grid mask loop."""
    import numpy as np
    result = np.full((len(positions), len(nodes)), -1, dtype=np.int64)
    columns = {node: i for i, node in enumerate(nodes)}
    for path_id, leaf, left, right in corpus.occurrences:
        if leaf not in columns:
            continue
        lo, hi = np.searchsorted(positions, (left, right))
        column = columns[leaf]
        _require(np.all(result[lo:hi, column] == -1), "Native occurrence overlap at readout")
        result[lo:hi, column] = path_id
    _require(np.all(result >= 0), "Every query/coordinate must retain its native full path")
    return result


def prepare_parent_v1(parent_id, completion_ref, output_directory, *, config, guard):
    """CPU-only complete paths/vocabulary/observed labels; no query truth input."""
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only path preparation required")
    reader_cfg, roles = _policy(config)
    _require(roles.get(parent_id) in ("train", "val"), "Registered inner parent only")
    reader = importlib.import_module("arggnn.expanded_native_ancestralpaths_v1")
    observed = reader.read_completed_parent_v1(parent_id, completion_ref, config=reader_cfg)
    import numpy as np
    from .official_argformer_bridge import extract_observed_paths, build_vocabulary
    started = time.monotonic()
    namespace = parent_id + ":" + observed["tree_ref"]["sha256"]
    samples = [int(r["inferred_tree_node_id"]) for r in observed["rows"]]
    corpus = extract_observed_paths(observed["ts"], namespace=namespace, sample_ids=samples)
    vocabulary = build_vocabulary(corpus).official_values()
    queries = [r for r in observed["rows"] if r["role"] == "query"]
    donors = {int(r["inferred_tree_node_id"]): int(r["observed_reference_class"])
              for r in observed["rows"] if r["role"] == "pure_source_reference"}
    _require(len(samples) == 952 and len(queries) == 686 and len(donors) == 266
        and [list(donors.values()).count(c) for c in range(4)] == [50, 28, 102, 86],
        "Complete unequal full-panel donor/query roles required")
    ids = list(range(len(corpus.paths))); random.Random(42).shuffle(ids)
    cut = int(.9 * len(ids)); splits = dict(train=ids[:cut], val=ids[cut:])
    counts = {task: {split: len(values) if task == "mlm" else
        sum(corpus.paths[i][0] in donors for i in values) for split, values in splits.items()}
        for task in ("mlm", "contrastive")}
    _require(counts["mlm"]["train"] >= 512 and counts["contrastive"]["train"] >= 4096
        and counts["mlm"]["val"] > 0 and counts["contrastive"]["val"] > 0,
        "Native batch/split shortfall: fail, never reduce source batches")
    output = _directory(output_directory, guard)
    files = {}
    for split, values in splits.items():
        for task in ("mlm", "contrastive"):
            selected = values if task == "mlm" else [i for i in values if corpus.paths[i][0] in donors]
            directory = output / task / split
            _guard(guard, 8192); directory.mkdir(parents=True)
            path = directory / ("raw_" + split + "_sequences_and_vocab.pkl")
            files[path.relative_to(output).as_posix()] = _pickle(path,
                dict(raw_sequences=[list(corpus.paths[i]) for i in selected], vocab=vocabulary), guard)
            if task == "contrastive":
                path = directory / "labels.pkl"
                files[path.relative_to(output).as_posix()] = _pickle(path,
                    [donors[corpus.paths[i][0]] for i in selected], guard)
            files[(directory / "path_ids.json").relative_to(output).as_posix()] = _json(
                directory / "path_ids.json", selected, guard)
    directory = output / "embedding/all"
    _guard(guard, 8192); directory.mkdir(parents=True)
    files["embedding/all/raw_all_sequences_and_vocab.pkl"] = _pickle(
        directory / "raw_all_sequences_and_vocab.pkl",
        dict(raw_sequences=[list(path) for path in corpus.paths], vocab=vocabulary), guard)
    files["corpus.pkl"] = _pickle(output / "corpus.pkl",
        dict(paths=corpus.paths, occurrences=corpus.occurrences, namespace=namespace), guard)
    support = observed["support"]; offset = support["coverage_original_bp0"][0]
    local_grid = np.asarray(support["local_centers_bp"], dtype=np.float64)
    original_sites = np.asarray([r["original_position_0based"] for r in observed["site_join"]], dtype=np.float64)
    nodes = [int(r["inferred_tree_node_id"]) for r in queries]
    reference_path_ids = [i for i, path in enumerate(corpus.paths) if path[0] in donors]
    files["coordinate_path_joins.npz"] = _npz(output / "coordinate_path_joins.npz", dict(
        grid_path_ids=_joins(corpus, local_grid, nodes),
        SNP_path_ids=_joins(corpus, original_sites - offset, nodes),
        positions_bp=np.asarray(support["original_centers_bp"]), site_positions_bp=original_sites,
        bin_edges_bp=np.asarray(support["clipped_bin_edges_bp"]), query_nodes=np.asarray(nodes),
        query_sample_indices=np.asarray([r["sample_index"] for r in queries]),
        raw_query_nodes=np.asarray([r["source_observed_node_id"] for r in queries]),
        retained_fixed100bp_indexes=np.asarray(support["grid_indexes"]),
        coverage_original_bp0=np.asarray(support["coverage_original_bp0"]),
        reference_full_path_ids=np.asarray(reference_path_ids, dtype=np.int64),
        reference_full_path_classes=np.asarray([donors[corpus.paths[i][0]] for i in reference_path_ids])), guard)
    files["observed_roster.json"] = _json(output / "observed_roster.json", observed["rows"], guard)
    result = dict(schema="expanded_native_argformer_prepared_v1", parent_id=parent_id, split=roles[parent_id],
        namespace=namespace, source_ref=config["source_ref"], config_digest=_digest(config),
        completion_ref=completion_ref, inferred_tree_ref=observed["tree_ref"], files=files,
        path_counts=counts, unique_full_paths=len(corpus.paths),
        vocabulary_size=len(vocabulary["special"]) + len(vocabulary["node_id_tokens"]),
        full_path_occurrences=len(corpus.occurrences), class_order=config["class_order"],
        query_keys=[[r["sample_id"], r["haplotype"]] for r in queries],
        input_samples=952, query_haplotypes=686, reference_haplotypes=266, BAA_queries=14,
        observed_ages_verified=True, dates_encoded_as_tokens=False,
        within_ARG_transductive=True, true_query_ancestry_read=False, outer_included=False,
        corpus_is_full_before_skip_truncation=True, preparation_elapsed_seconds=time.monotonic() - started,
        MLM_optimizer_updates=5 * (counts["mlm"]["train"] // 512),
        contrastive_optimizer_updates=counts["contrastive"]["train"] // 4096)
    result["manifest_ref"] = _json(output / "prepared_manifest.json", result, guard)
    return result


def _prepared(ref, config):
    _, roles = _policy(config)
    value = _read(ref)
    _require(value["schema"] == "expanded_native_argformer_prepared_v1"
        and roles.get(value["parent_id"]) in ("train", "val")
        and value["split"] == roles[value["parent_id"]]
        and value["source_ref"] == config["source_ref"] and value["config_digest"] == _digest(config)
        and value["namespace"] == value["parent_id"] + ":" + value["inferred_tree_ref"]["sha256"]
        and value["true_query_ancestry_read"] is False and value["input_samples"] == 952
        and value["query_haplotypes"] == 686 and value["reference_haplotypes"] == 266,
        "Exact full observed parent vocabulary/corpus required")
    directory = _checked(ref).parent
    for relative, record in value["files"].items():
        _require(_checked(record) == directory / relative, "Prepared path binding differs")
    return value, directory


def _gpu_runtime(output, prepared, config, guard, gpu_authorizer):
    """Root callback owns shared leases/two-device/memory admission before CUDA."""
    _require(callable(gpu_authorizer), "Root GPU resource authorizer required")
    _require(Path(sys.prefix).resolve() == _path(NATIVE_PREFIX) and sys.version_info[:3] == (3, 11, 9),
             "Qualified native interpreter required")
    approved = gpu_authorizer(dict(source_ref=config["source_ref"], namespace=prepared["namespace"],
        parent_id=prepared["parent_id"], model="released_ARGformer_full512_full4096"))
    _require(approved.get("physical_gpu") in (2, 3, 4, 5)
        and approved.get("held_lease_verified") is True
        and approved.get("two_physical_device_limit_checked") is True
        and approved.get("logical_memory_reservation_verified") is True,
        "Root shared-lease/device/memory authority required")
    title = _module(config["helper_refs"]["process_title"], "arggnn._expanded_argformer_title")
    title.apply_and_validate_title()
    qualifier = _module(config["helper_refs"]["qualifier"], "arggnn._expanded_argformer_qualifier")
    for name, version in qualifier.EXPECTED.items():
        _require(importlib.metadata.version(name).split("+")[0] == version, "Qualified distribution drift: " + name)
    _guard(guard, 1024 * 1024)
    cache = output / "runtime_cache"; cache.mkdir()
    for name in CACHE_ENV:
        directory = cache / name.lower(); directory.mkdir()
        os.environ[name] = str(directory)
    os.environ.update(WANDB_MODE="disabled", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false", TMPDIR=str(cache), PYTHONDONTWRITEBYTECODE="1")
    import tempfile
    tempfile.tempdir = str(cache)
    import torch
    _require(torch.cuda.device_count() == 1
        and os.environ.get("ARGGNN_PHYSICAL_GPU") == str(approved["physical_gpu"]),
        "Exactly one admitted eligible physical CUDA device required")
    anchor = torch.zeros(1, device="cuda"); torch.cuda.synchronize()
    runtime = {}
    _require(qualifier.actual_cuda_uuid(output, runtime) == approved.get("canonical_uuid"),
             "Actual CUDA context must match the held root lease UUID")
    _json(output / "native_cuda_identity.json", runtime, guard)
    del anchor
    sys.path.insert(0, str(_path(AUTHOR)))
    # Root performs once-per-corpus memory qualification; no fallback is supplied.
    return qualifier, approved


def _scientific_checks(config):
    source = _checked(config["helper_refs"]["fit_definitions"]).read_text()
    wanted = {"assert_backbone", "verify_encoder_initializer"}
    nodes = [copy.deepcopy(n) for n in ast.parse(source).body
             if isinstance(n, ast.FunctionDef) and n.name in wanted]
    _require(len(nodes) == 2, "Pinned native scientific definitions required")
    namespace = dict(importlib=importlib, SOURCE=_path(AUTHOR), sha=lambda p: _ref(p)["sha256"])
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 str(_path(HELPERS["fit_definitions"])), "exec"), namespace)
    return namespace


def _model(mode, prepared_dir, output, *, pretrained=None):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(_path(AUTHOR) / "yamls" / (mode + ".yaml"))
    cfg.data_local = str(prepared_dir / ("mlm" if mode == "mlm" else "contrastive"))
    cfg.model.pretrained_model_name = str(output / "base_config")
    cfg.loggers = {}; cfg.load_path = None; cfg.autoresume = False
    cfg.run_name = "expanded_native_" + prepared_dir.name + "_" + mode
    cfg.save_folder = None
    if mode == "mlm":
        module = importlib.import_module("main")
        native = module.build_model(cfg.model)
        loader = module.build_dataloader(cfg=cfg.train_loader, tokenizer=native.tokenizer,
                                        device_batch_size=512, device_microbatch_size=512)
        evaluation = module.build_dataloader(cfg=cfg.eval_loader, tokenizer=native.tokenizer, device_batch_size=512)
        _require(cfg.device_train_microbatch_size == 512 and cfg.max_duration == "5ep",
                 "Literal author MLM512/five epochs required")
    else:
        cfg.model.pretrained_checkpoint = str(pretrained) if pretrained else None
        module = importlib.import_module("sequence_contrastive")
        native = module.build_contrastive_model(cfg.model)
        _require(native.base.class_weights is None and native.base.backbone.config.num_labels == 3,
                 "Preserve released uniform fallback and unused classifier")
        loader = module.build_contrastive_dataloader(cfg.train_loader, device_batch_size=4096)
        evaluation = module.build_contrastive_dataloader(cfg.eval_loader, device_batch_size=4096)
        _require(cfg.device_train_microbatch_size == 4096 and cfg.max_duration == "1ep",
                 "Literal source full4096 contrastive batch/one epoch required")
    return native, loader, evaluation, module, cfg


def _drain_loader(loader):
    raw = loader.dataloader if hasattr(loader, "dataloader") else loader
    iterator = getattr(raw, "_iterator", None)
    if iterator is not None:
        workers = tuple(getattr(iterator, "_workers", ()))
        iterator._shutdown_workers()
        for worker in workers:
            worker.join(timeout=5)
        _require(not any(worker.is_alive() for worker in workers), "Owned native loader workers did not drain")


def _train_phase(mode, native, loader, evaluation, module, cfg, output, guard):
    import torch
    from composer import Trainer, Evaluator, Callback
    from omegaconf import OmegaConf
    _guard(guard, sum(p.numel() for p in native.parameters()) * 24 + 256 * 1024 * 1024)
    output.mkdir()
    _json(output / "resolved_release_config.json", OmegaConf.to_container(cfg, resolve=True), guard)
    class NativeCounts(Callback):
        updates = 0
        def after_train_batch(self, state, logger):
            self.updates += 1
            _require(torch.isfinite(state.outputs["loss"]).all().item(), "Native training loss became nonfinite")
    callback = NativeCounts()
    optimizer = module.build_optimizer(cfg.optimizer, native)
    scheduler = module.build_scheduler(cfg.scheduler)
    trainer = Trainer(model=native, train_dataloader=loader,
        eval_dataloader=Evaluator(label="eval", dataloader=evaluation,
                                  device_eval_microbatch_size=cfg.device_eval_microbatch_size),
        seed=cfg.seed, run_name=cfg.run_name, device="gpu",
        optimizers=optimizer, schedulers=scheduler,
        algorithms=[module.build_algorithm(n, v) for n, v in cfg.get("algorithms", {}).items()],
        callbacks=[callback], loggers=[], precision=cfg.precision, max_duration=cfg.max_duration,
        eval_interval=cfg.eval_interval, eval_subset_num_batches=cfg.eval_subset_num_batches,
        device_train_microbatch_size=cfg.device_train_microbatch_size,
        progress_bar=False, log_to_console=True, autoresume=False)
    started = time.monotonic()
    try:
        trainer.fit()
        epochs = int(trainer.state.timestamp.epoch.value)
    finally:
        # Guarded checkpoint publication follows own worker drain, so transient
        # native multiprocessing IPC entries are absent at the scope boundary.
        _drain_loader(loader); _drain_loader(evaluation)
    _require(callback.updates > 0 and epochs == (5 if mode == "mlm" else 1), "Native epoch/update shortfall")
    checkpoint = output / "composer_final.pt"
    _guard(guard, sum(p.numel() for p in native.parameters()) * 24 + 64 * 1024 * 1024)
    trainer.save_checkpoint(str(checkpoint))
    record = dict(mode=mode, checkpoint_ref=_ref(checkpoint), epochs=epochs,
        optimizer_updates=callback.updates, logical_batch=512 if mode == "mlm" else 4096,
        microbatch=cfg.device_train_microbatch_size, precision=cfg.precision, seed=cfg.seed,
        elapsed_seconds=time.monotonic() - started, same_ARG_only=True,
        source_batch_reduced=False, gradient_cache_used=False, query_truth_used=False)
    _json(output / "phase_receipt.json", record, guard)
    if hasattr(trainer, "close"):
        trainer.close()
    return checkpoint, record


def fit_parent_v1(prepared_ref, output_directory, *, config, guard, gpu_authorizer):
    """Fresh same-ARG source MLM512 then full4096 contrastive; no old checkpoint."""
    prepared, directory = _prepared(prepared_ref, config)
    output = _directory(output_directory, guard)
    qualifier, approved = _gpu_runtime(output, prepared, config, guard, gpu_authorizer)
    _guard(guard, 32768)
    base = output / "base_config"; base.mkdir()
    _bytes(base / "config.json", _checked(config["base_config_ref"]).read_bytes(), guard)
    checks = _scientific_checks(config)
    import torch
    from composer.utils import reproducibility
    reproducibility.seed_all(17)
    mlm, loader, evaluation, module, cfg = _model("mlm", directory, output)
    checks["assert_backbone"](mlm, "mlm", qualifier)
    mlm_checkpoint, mlm_record = _train_phase("mlm", mlm, loader, evaluation, module, cfg, output / "mlm", guard)
    del mlm, loader, evaluation; gc.collect(); torch.cuda.empty_cache()
    reproducibility.seed_all(36)
    native, loader, evaluation, module, cfg = _model("contrastive", directory, output, pretrained=mlm_checkpoint)
    checks["assert_backbone"](native, "contrastive", qualifier)
    initializer = checks["verify_encoder_initializer"](native, mlm_checkpoint, qualifier)
    classifier_before = qualifier.state_manifest(native.base.backbone.classifier)["sha256"]
    checkpoint, contrastive_record = _train_phase(
        "contrastive", native, loader, evaluation, module, cfg, output / "contrastive", guard)
    _require(qualifier.state_manifest(native.base.backbone.classifier)["sha256"] == classifier_before,
             "Unused native classifier changed")
    result = dict(schema="expanded_native_argformer_fit_v1", parent_id=prepared["parent_id"],
        namespace=prepared["namespace"], prepared_ref=prepared_ref, source_ref=config["source_ref"],
        config_digest=_digest(config), MLM=mlm_record, contrastive=contrastive_record,
        final_checkpoint_ref=_ref(checkpoint), encoder_initializer=initializer,
        classifier_unchanged_unused=True, root_GPU_authority=approved,
        fresh_same_ARG_fit=True, old_checkpoint_reused=False,
        within_ARG_transductive=True, query_truth_used=False, paper_original_numerical_reproduction=False)
    result["manifest_ref"] = _json(output / "fit_manifest.json", result, guard)
    return result


def infer_parent_v1(fit_ref, prepared_ref, output_directory, *, config, guard, gpu_authorizer):
    """Literal native retrieval twice (k1/k5), every donor path and query center."""
    prepared, directory = _prepared(prepared_ref, config)
    fit = _read(fit_ref)
    _require(fit["schema"] == "expanded_native_argformer_fit_v1" and fit["prepared_ref"] == prepared_ref
        and fit["namespace"] == prepared["namespace"] and fit["source_ref"] == config["source_ref"]
        and fit["config_digest"] == _digest(config) and fit["fresh_same_ARG_fit"] is True,
        "Only this exact parent's fresh final fit may infer")
    output = _directory(output_directory, guard)
    qualifier, approved = _gpu_runtime(output, prepared, config, guard, gpu_authorizer)
    _guard(guard, 32768); (output / "base_config").mkdir()
    _bytes(output / "base_config/config.json", _checked(config["base_config_ref"]).read_bytes(), guard)
    native, loader, evaluation, _, cfg = _model("contrastive", directory, output)
    import torch
    payload = torch.load(_checked(fit["final_checkpoint_ref"]), map_location="cpu", weights_only=False)
    native.load_state_dict(payload["state"]["model"], strict=True)
    _scientific_checks(config)["assert_backbone"](native, "contrastive", qualifier)
    native.cuda().eval()
    from src.arg_data import ARGDataset
    dataset = ARGDataset(local=str(directory / "embedding"), split="all", max_seq_len=128, skip_extant_tokens=True)
    _require(len(dataset) == prepared["unique_full_paths"], "Complete full-path embeddings required")
    import numpy as np
    embeddings = torch.empty((len(dataset), 768), dtype=torch.float32)
    started = time.monotonic()
    with torch.no_grad():
        for start in range(0, len(dataset), 128):
            items = [dataset[i] for i in range(start, min(start + 128, len(dataset)))]
            ids = torch.tensor([item["input_ids"] for item in items], device="cuda")
            mask = torch.tensor([item["attention_mask"] for item in items], device="cuda")
            z = native.base.encode(ids, mask)
            _require(torch.isfinite(z).all().item(), "Native path embedding became nonfinite")
            embeddings[start:start + len(items)] = z.cpu()
    with np.load(_checked(prepared["files"]["coordinate_path_joins.npz"]), allow_pickle=False) as archive:
        joins = {key: archive[key].copy() for key in archive.files}
    donor_ids = joins["reference_full_path_ids"]
    query_ids = np.unique(np.concatenate((joins["grid_path_ids"].ravel(), joins["SNP_path_ids"].ravel())))
    retrieval = _module(config["retrieval_source_ref"], "_expanded_author_argformer_retrieve")
    E = retrieval.ensure_l2_normalized(embeddings[torch.as_tensor(donor_ids)])
    Q = retrieval.ensure_l2_normalized(embeddings[torch.as_tensor(query_ids)])
    _require(len(donor_ids) >= 5, "Native k5 reference corpus shortfall")
    products, neighbor_refs = {}, {}
    for k in (1, 5):
        # Independent literal calls preserve source topk tie semantics at each k.
        scores, indexes = retrieval.topk_indices_streaming(Q, E, device=torch.device("cuda"),
                                                          k=k, q_batch=1024, e_chunk=100000)
        classes = joins["reference_full_path_classes"][indexes.numpy()]
        votes = np.stack([(classes == c).mean(axis=1) for c in range(4)], axis=-1).astype(np.float32)
        lookup_grid = np.searchsorted(query_ids, joins["grid_path_ids"])
        lookup_sites = np.searchsorted(query_ids, joins["SNP_path_ids"])
        products[str(k)] = _npz(output / ("C4_k" + str(k) + "_full_support.npz"), dict(
            probabilities= votes[lookup_grid], SNP_probabilities=votes[lookup_sites],
            positions_bp=joins["positions_bp"], site_positions_bp=joins["site_positions_bp"],
            bin_edges_bp=joins["bin_edges_bp"], query_nodes=joins["query_nodes"],
            query_sample_indices=joins["query_sample_indices"], raw_query_nodes=joins["raw_query_nodes"],
            retained_fixed100bp_indexes=joins["retained_fixed100bp_indexes"],
            coverage_original_bp0=joins["coverage_original_bp0"]), guard)
        neighbor_refs[str(k)] = _npz(output / ("native_k" + str(k) + "_neighbors.npz"), dict(
            query_full_path_ids=query_ids, neighbor_full_path_ids=donor_ids[indexes.numpy()],
            native_cosine_scores=scores.numpy(), raw_class_votes=votes), guard)
    _guard(guard, embeddings.numel() * 4 + 65536)
    torch.save(dict(embeddings=embeddings, namespace=prepared["namespace"],
                   pooling="trained_author_mean_head", all_full_paths=True), output / "full_path_embeddings.pt")
    result = dict(schema="expanded_native_argformer_prediction_seal_v1",
        parent_id=prepared["parent_id"], namespace=prepared["namespace"], fit_ref=fit_ref,
        prepared_ref=prepared_ref, source_ref=config["source_ref"], config_digest=_digest(config),
        class_order=config["class_order"], primary_k=1, sensitivity_k=5,
        prediction_refs=products, native_neighbor_refs=neighbor_refs,
        full_embeddings_ref=_ref(output / "full_path_embeddings.pt"),
        all266_donor_unique_paths_across_train_val=True, reference_full_paths=len(donor_ids),
        query_full_paths=len(query_ids), query_haplotypes=686, BAA_queries=14,
        source_torch_topk_device="cuda", exact_ties_use_source_topk=True,
        duplicate_embedding_vectors_retained_by_full_path_ID=True,
        raw_vote_zero_probabilities_retained=True, NLL_may_be_infinite=True,
        no_tail_extrapolation=True, no_query_truth_read=True, calibration_fitted=False,
        old_checkpoint_reused=False, within_ARG_transductive=True,
        inference_elapsed_seconds=time.monotonic() - started, root_GPU_authority=approved,
        paper_original_numerical_reproduction=False)
    result["manifest_ref"] = _json(output / "prediction_seal.json", result, guard)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(default_config_v1(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
