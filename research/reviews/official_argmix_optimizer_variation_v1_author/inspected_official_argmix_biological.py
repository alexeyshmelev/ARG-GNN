"""Mechanical bridge to unchanged current-release ARGMix; root execution only."""
from __future__ import annotations
from contextlib import redirect_stdout, redirect_stderr
import ctypes
import hashlib
import importlib
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import traceback
import uuid

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
AUTHOR = ROOT / ".runtime/author_sources/argmix_c2481248eb0d"
WORK = ROOT / ".runtime/official_argmix_abi210_v1/cu129"
PREFIX = WORK / "env"
SOURCE = WORK / "source"
CLI = SOURCE / "src/rust/ancpath/target/cli/release/build_training"
MODULE = ROOT / "src/arggnn/official_argmix_biological.py"
SCRIPT = ROOT / "scripts/train_official_argmix_biological_v1.py"
CONFIG = ROOT / "configs/official_argmix_biological_pilot_v1.json"
GUARD = ROOT / "src/arggnn/native_argformer_exec.py"
HELPERS = ROOT / "scripts/qualify_official_argformer_runtime.py"
GUARD_SHA = "84ed405379ac56511c2f4722c130e8b884226018bbf8e2949ea2bb4dceeb99d0"
HELPERS_SHA = "8fe652f87848f4a296e3d88555998fb388f11ae5c04d0858095e70c34b10b31c"
SOURCE_TREE_SHA = "4fbcc41801027983a9654920d143f053d5b74ffea020b751a13757c0ae327ae8"

OBSERVED_MANIFEST = {"path": "data/biological_paths_execution_pilot_v1/relate_inferred_v1/manifest.json",
                     "sha256": "6872222127c96b3defbfb1b9edb83ba1d5b11b0b60e6888956369bc507cefe28"}
DATASET_MANIFEST = {"path": "data/biological_paths_execution_pilot_v1/manifest.json",
                    "sha256": "d404f4308466b09d7fa9a296b12bb76932bf84142bebbd03503cb2d2445665ea"}
PARENT_SPLITS = {"bio_paths_pilot_train_000": "train", "bio_paths_pilot_train_001": "train",
                 "bio_paths_pilot_val_000": "val", "bio_paths_pilot_val_001": "val"}
NATIVE_MODEL = {"type": "tmrca", "num_classes": 4, "num_refs": 15, "hidden_size": 128,
                "gt_output_dim": 16, "dim_feedforward_mult": 4, "num_heads": 4, "num_layers": 4,
                "dropout_prob": 0.25, "attn_dropout_prob": 0.1, "max_rel_distance": 14,
                "log_scale_tmrca": True}
JOINT_MODEL = {"enabled": True, "base_encoder": "tmrca_transformer", "chunk_rows": 1024,
    "chunk_stride": 512, "cm_features": {"include_delta_cm_prev": False,
        "row_span_transform": {"mode": "raw", "median_span_cM": None,
                               "clamp_min": 1e-6, "clamp_max": 10.0}},
    "gru": {"hidden_size": 128, "num_layers": 1, "bidirectional": True, "dropout": 0.1},
    "train": {"batch_size": 16, "num_epochs": 20, "learning_rate": 2e-4, "weight_decay": 1e-4,
        "label_smoothing": 0.1, "early_stop_patience": 2, "loss_weighting": {"mode": "none"},
        "sample_seed": 0, "max_train_subgraphs": None, "max_val_subgraphs": None,
        "reference_selection_augmentation": {"train": {"enabled": True, "drop_top_m_nearest": 1},
                                             "val": {"enabled": False, "drop_top_m_nearest": 0}}}}
ADMITTED_NATIVE_FILES = {}
METADATA_REPAIR_V3 = {'kind': 'official_argmix_biological_metadata_compatibility_v3', 'prior_config_sha256': 'beb49bccf0f4780d6d76a3e738e1cba85dc5a1de4c846811aea2a20962029819', 'prior_owned_code_sha256': {'scripts/train_official_argmix_biological_v1.py': '4261fb1195d16f73370b5fa0144ba0f59afe6a67726af6567634b0bd282e223d', 'src/arggnn/official_argmix_biological.py': 'cde961f2a9ed7aa98b6d24af644ef83d5657abed7ac5d8029afe46cebafc1665'}, 'inherited_products': {'official_argmix_observed_frames_v1': {'path': 'experiments/results/official_argmix_biological_pilot_v2/observed/observed_frames.json', 'sha256': 'b8de4b503bfebcf7eb71b6861879826f24731deb37aabe68b5283f6f07f07531'}, 'official_argmix_native_biological_archives_v1': {'path': 'experiments/results/official_argmix_biological_pilot_v2/native/native_prepared.json', 'sha256': '2c15bd908ae89b81e1a8c9d07809e82c46d6b1b41a2031ac93013dc54534961c'}}, 'change_scope': 'add_bound_observed_references_pointer_for_stock_counting_and_logging_only', 'metadata_reference_parent': 'bio_paths_pilot_train_000'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def local(value, observed=False):
    lexical = Path(value)
    if ".." in lexical.parts or (observed and any(
            "truth" in part.lower() or "oracle" in part.lower() for part in lexical.parts)):
        raise ValueError("Observed-path truth/traversal forbidden")
    path = lexical if lexical.is_absolute() else ROOT / lexical
    if path.resolve() != path or not path.is_relative_to(ROOT):
        raise ValueError("Canonical repository-local path required")
    return path


def new_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def checked(record, observed=False):
    path = local(record["path"], observed)
    if sha(path) != record["sha256"]:
        raise ValueError("Bound artifact changed: " + str(path))
    return path


def load_frozen(path, digest, name):
    if sha(path) != digest:
        raise RuntimeError("Frozen helper changed")
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def source_inventory(require_copy=True):
    files = {str(path.relative_to(AUTHOR)): sha(path)
             for path in sorted(AUTHOR.rglob("*")) if path.is_file()}
    if len(files) != 86 or hashlib.sha256(canonical(files) + b"\n").hexdigest() != SOURCE_TREE_SHA:
        raise RuntimeError("Immutable current-author source inventory changed")
    if require_copy:
        for name, digest in files.items():
            path = local(SOURCE / name)
            if sha(path) != digest:
                raise RuntimeError("Ordinary execution-copy source changed: " + name)
    return files


def policy(expected_sha=None):
    if expected_sha and sha(CONFIG) != expected_sha:
        raise ValueError("Root-reviewed ARGMix biological policy changed")
    cfg = json.loads(CONFIG.read_text())
    expected = {"kind": "official_argmix_biological_pilot_v1", "schema_version": 1,
        "author_commit": "c2481248eb0de5fda659d68f59c51e0e27becd20",
        "source_tree_sha256": SOURCE_TREE_SHA, "class_order": ["ANA", "CHG", "WHG", "EHG"],
        "runtime_prefix": str(PREFIX.relative_to(ROOT)), "source_copy": str(SOURCE.relative_to(ROOT)),
        "published_April_paper_reproduction": False, "parent_ids": list(PARENT_SPLITS),
        "parent_splits": PARENT_SPLITS, "observed_manifest": OBSERVED_MANIFEST,
        "dataset_manifest": DATASET_MANIFEST, "native_model": NATIVE_MODEL, "joint_model": JOINT_MODEL,
        "inference_only_exclude_edge_trees": False, "full_support_missing_policy": "unavailable_no_denominator_reduction",
        "source_seed": 50, "external_VAL_use": "native_selection_without_gradients_and_sealed_descriptive_scores",
        "normalization_length_bp": 2000000, "active_cm_feature_count": 1, "stored_cm_feature_count": 2,
        "compatibility_amendment": METADATA_REPAIR_V3}
    if set(cfg) != set(expected) | {"owned_code_sha256"} or any(
            canonical(cfg[key]) != canonical(value) for key, value in expected.items()):
        raise ValueError("Current-release source/default/model/data policy differs")
    expected_paths = {str(SCRIPT.relative_to(ROOT)), str(MODULE.relative_to(ROOT))}
    if set(cfg["owned_code_sha256"]) != expected_paths:
        raise ValueError("Both owned source files must be pinned")
    for name, digest in cfg["owned_code_sha256"].items():
        if sha(local(name)) != digest:
            raise ValueError("Root-reviewed owned bridge changed")
    if sha(GUARD) != GUARD_SHA or sha(HELPERS) != HELPERS_SHA:
        raise RuntimeError("Frozen lease/NVTOP helper changed")
    source_inventory(require_copy=False)
    return cfg


def stage(record, kind, cfg, observed=False):
    result = json.loads(checked(record, observed=observed).read_text())
    if result.get("kind") != kind:
        raise ValueError("Prior stage kind differs")
    if (result.get("config_sha256") == sha(CONFIG) and
            result.get("owned_code_sha256") == cfg["owned_code_sha256"]):
        return result
    amendment = cfg["compatibility_amendment"]
    inherited = amendment["inherited_products"].get(kind)
    if (inherited is None or record != inherited or
            result.get("config_sha256") != amendment["prior_config_sha256"] or
            result.get("owned_code_sha256") != amendment["prior_owned_code_sha256"]):
        raise ValueError("Prior stage is not the exact admitted inherited product")
    # Return original declarations unchanged. Only two exact products are admitted.
    return result


def verify_frames(record, cfg):
    frames = stage(record, "official_argmix_observed_frames_v1", cfg, observed=True)
    if frames["manifest"] != cfg["observed_manifest"] or frames["dataset_manifest"] != cfg["dataset_manifest"]:
        raise ValueError("Prepared root manifests differ from policy")
    original = json.loads(checked(frames["manifest"], observed=True).read_text())
    dataset = json.loads(checked(frames["dataset_manifest"]).read_text())
    entries = {entry["parent_id"]: entry for entry in original["replicates"]}
    raw = {entry["parent_id"]: entry for entry in dataset["replicates"]}
    if set(frames["parents"]) != set(PARENT_SPLITS):
        raise ValueError("Prepared four-parent scope differs")
    for parent, frame in frames["parents"].items():
        if (frame["parent_id"] != parent or frame["split"] != PARENT_SPLITS[parent] or
                frame["source_observed_entry"] != entries[parent] or frame["external_dataset_entry"] != raw[parent] or
                frame["tree"] != entries[parent]["inferred_tree"] or
                frame["external_dataset_directory"] != raw[parent]["directory"]):
            raise ValueError("Prepared original parent/split provenance differs")
        for name in ("tree", "references", "native_sample", "coordinates", "roster"):
            checked(frame[name], observed=True)
        checked(entries[parent]["observed_sample_manifest"], observed=True)
        checked(entries[parent]["original_observed_sample_manifest"], observed=True)
        checked(entries[parent]["coordinate_contract"]["fixed_grid"], observed=True)
        if json.loads(checked(frame["roster"], observed=True).read_text()) != json.loads(
                checked(entries[parent]["observed_sample_manifest"], observed=True).read_text()):
            raise ValueError("Copied observed roster differs from original")
    checked(frames["local_map"], observed=True)
    return frames


def verify_prepared(record, cfg):
    prepared = stage(record, "official_argmix_native_biological_archives_v1", cfg)
    verify_frames(prepared["frames"], cfg)
    for key in ("backend", "train_pairs", "val_pairs", "local_map"):
        checked(prepared[key], observed=key == "local_map")
    for split in ("train", "val"):
        checked(prepared["archives"][split])
    for record in prepared["truth_provenance"].values():
        checked(record)
        checked(record["native_truth"])
    return prepared


def verify_backend(backend, cfg, inference=False, references_record=None):
    expected = backend_config(cfg, ROOT)
    if inference:
        expected["processing"]["exclude_edge_trees"] = False
    if set(backend) != set(expected):
        raise ValueError("Native backend sections differ")
    for name in expected:
        if name not in {"paths", "relate"} and canonical(backend[name]) != canonical(expected[name]):
            raise ValueError("Native source backend setting differs: " + name)
    original_paths = {"train_pairs", "val_pairs", "save_dir"}
    if set(backend["paths"]) not in (original_paths, original_paths | {"references"}) or set(backend["relate"]) != {"recomb_map"}:
        raise ValueError("Native path schema differs")
    if references_record is not None and "references" not in backend["paths"]:
        raise ValueError("Bound fitted/inference backend requires stock metadata references")
    if "references" in backend["paths"]:
        if references_record is None or backend["paths"]["references"] != str(checked(references_record, observed=True)):
            raise ValueError("Stock metadata references pointer lacks its exact observed binding")
    for value in [*backend["paths"].values(), backend["relate"]["recomb_map"]]:
        local(value)
    return backend


def runtime_admission(build_record, cpu_record, cfg):
    global ADMITTED_NATIVE_FILES
    build = json.loads(checked(build_record).read_text())
    cpu = json.loads(checked(cpu_record).read_text())
    if (build["status"] != "stock_native_build_passed_CPU_runtime_admission_pending" or
            build.get("compiled") is not True or build.get("CLI_discovery_path_preserved") is not True or
            build.get("copied_author_files_unchanged") is not True or
            build.get("immutable_author_files_unchanged") is not True):
        raise RuntimeError("Completed unchanged stock native build required")
    for artifact in build["native_artifacts"]:
        checked(artifact)
    if not any(local(item["path"]) == CLI for item in build["native_artifacts"]):
        raise RuntimeError("Stock CLI discovery path absent from build receipt")
    if (cpu.get("status") != "passed" or
            cpu.get("actual_CPU_Python_to_Rust_TorchScript_probe") is not True or
            cpu.get("Rust_device") != "Cpu" or cpu.get("torch_version") != "2.10.0+cu129"):
        raise RuntimeError("Actual qualified Python-to-Rust CPU library admission required")
    extension_paths = {local(item["path"]) for item in build["native_artifacts"] if local(item["path"]) != CLI}
    actual_extensions = {local(path) for path in cpu.get("native_extension_paths", [])}
    torchlib = PREFIX / "lib/python3.12/site-packages/torch/lib"
    loaded_torch = [local(path) for path in cpu.get("loaded_library_paths", [])
                    if "libtorch" in path or "libc10" in path]
    if (not extension_paths or actual_extensions != extension_paths or not loaded_torch or
            any(not path.is_relative_to(torchlib) for path in loaded_torch) or
            cpu.get("GPU_context_requested") is not False):
        raise RuntimeError("Actual CPU extension/Torch mappings differ from stock build and wheel")
    checked(build["closure"])
    checked(build["admission_commands"]["helper"])
    checked(build["admission_commands"]["command"])
    ADMITTED_NATIVE_FILES = {str(local(item["path"])): item["sha256"] for item in build["native_artifacts"]}
    source_inventory()
    return {"build": build_record, "CPU_library_admission": cpu_record,
            "native_artifacts": build["native_artifacts"]}


def source_imports():
    if Path(sys.prefix).resolve() != PREFIX or sys.version_info[:2] != (3, 12):
        raise RuntimeError("Fixed qualified Python3.12 native interpreter required")
    if "LIBTORCH_BYPASS_VERSION_CHECK" in os.environ:
        raise RuntimeError("Any version-guard bypass presence forbidden")
    import torch
    import torchvision
    import ancpath
    if (torch.__version__ != "2.10.0+cu129" or torchvision.__version__ != "0.25.0+cu129" or
            torch.version.cuda != "12.9"):
        raise RuntimeError("Root-qualified ABI210/cu129 tuple required")
    if not Path(torch.__file__).resolve().is_relative_to(PREFIX):
        raise RuntimeError("Torch module outside native prefix")
    sys.path.insert(0, str(SOURCE / "src"))
    joint = importlib.import_module("joint_interval_model")
    trainer = importlib.import_module("train_joint_model")
    loader = importlib.import_module("config_loader")
    for module in (joint, trainer, loader, importlib.import_module("tree_sampler"),
                   importlib.import_module("tmrca_attn")):
        if not Path(module.__file__).resolve().is_relative_to(SOURCE / "src"):
            raise RuntimeError("Loaded author module outside unchanged execution copy")
    maps = Path("/proc/self/maps").read_text()
    torchlib = Path(torch.__file__).resolve().parent / "lib"
    libraries = sorted({line.split()[-1] for line in maps.splitlines()
                        if "/" in line and ("libtorch" in line or "libc10" in line)})
    if not libraries or any(not Path(path).resolve().is_relative_to(torchlib) for path in libraries):
        raise RuntimeError("Actual loaded Torch libraries differ from wheel admission")
    extensions = {line.split()[-1] for line in maps.splitlines() if "ancpath" in line and ".so" in line}
    expected_extensions = set(ADMITTED_NATIVE_FILES) - {str(CLI)}
    if not expected_extensions or extensions != expected_extensions or any(
            sha(path) != digest for path, digest in ADMITTED_NATIVE_FILES.items()):
        raise RuntimeError("Actually loaded ancpath mappings differ from admitted stock build")
    return torch, ancpath, joint, trainer, loader, libraries


def gpu_runtime(output):
    guard = load_frozen(GUARD, GUARD_SHA, "_argmix_frozen_GPU_lease")
    helpers = load_frozen(HELPERS, HELPERS_SHA, "_argmix_frozen_NVTOP")
    lease = guard.validate_lease()
    title = guard.apply_and_validate_title()
    torch, ancpath, joint, trainer, loader, libraries = source_imports()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one actually leased GPU required")
    anchor = torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    maps = Path("/proc/self/maps").read_text()
    paths = sorted({line.split()[-1] for line in maps.splitlines()
                    if "/" in line and "libcuda.so" in line})
    if len(paths) != 1:
        raise RuntimeError("Unique actually loaded CUDA driver required")
    driver = ctypes.CDLL(paths[0])
    current = ctypes.c_int(-1)
    get_device = driver.cuCtxGetDevice
    get_device.argtypes = [ctypes.POINTER(ctypes.c_int)]
    get_device.restype = ctypes.c_int
    if get_device(ctypes.byref(current)) != 0 or current.value != 0:
        raise RuntimeError("Actual CUDA current context is not logical device0")
    class UUID(ctypes.Structure):
        _fields_ = [("bytes", ctypes.c_ubyte * 16)]
    actual = UUID()
    get_uuid = getattr(driver, "cuDeviceGetUuid_v2", driver.cuDeviceGetUuid)
    get_uuid.argtypes = [ctypes.POINTER(UUID), ctypes.c_int]
    get_uuid.restype = ctypes.c_int
    if get_uuid(ctypes.byref(actual), current.value) != 0:
        raise RuntimeError("Actual driver UUID request failed")
    device_uuid = str(uuid.UUID(bytes=bytes(actual.bytes)))
    if device_uuid != lease["canonical_uuid"]:
        raise RuntimeError("Actual CUDA driver UUID differs from scheduler lease")
    runtime = {"actual_cuda": True, "lease": lease, "process_title": title,
               "actual_uuid": device_uuid, "torch": torch.__version__,
               "compiled_cuda": torch.version.cuda, "loaded_Torch_libraries": libraries,
               "author_source_tree_sha256": SOURCE_TREE_SHA}
    (output / "loaded_library_maps.txt").write_text(maps)
    helpers.nvtop_snapshot(output, lease, runtime)
    return runtime, anchor, (torch, ancpath, joint, trainer, loader)


def backend_config(cfg, root, train_pairs=None, val_pairs=None):
    return {
        "populations": {"reference": cfg["class_order"], "predict": ["QUERY"],
                        "class_labels": {i: name for i, name in enumerate(cfg["class_order"])}},
        "model": dict(cfg["native_model"]), "joint_model": cfg["joint_model"],
        "processing": {"backend_name": "fast", "exclude_edge_trees": True,
                       "subgraph_merge": {"enabled": False, "signature_mode": "ids_only"}},
        "supervision": {"label_source": "npz_truth", "truth_dataset": "truth",
                        "truth_label_mode": "first_snp", "truth_min_label_purity": None,
                        "truth_collect_diagnostics": True},
        "paths": {"train_pairs": str(train_pairs or root / "train_pairs.yaml"),
                  "val_pairs": str(val_pairs or root / "val_pairs.yaml"),
                  "save_dir": str(root / "checkpoints")},
        "relate": {"recomb_map": str(root / "local_uniform_map.tsv")},
        "wandb": {"enabled": False}, "inference": {"device": "cuda"},
        "runtime": {}, "testing": {},
    }


def run_author(name, args, output):
    """Run exact stock script main in the titled, UUID-leased process."""
    old = sys.argv[:]
    log = output / (name.removesuffix(".py") + ".log")
    loggers = [logging.getLogger()] + [item for item in logging.Logger.manager.loggerDict.values()
                                      if isinstance(item, logging.Logger)]
    saved = {item: (item.handlers[:], item.level, item.propagate, item.disabled) for item in loggers}
    old_handlers = {handler for item in loggers for handler in item.handlers}
    for item in loggers:
        item.handlers = []
    started = time.monotonic()
    code = 0
    try:
        sys.argv = [str(SOURCE / "src" / name), *map(str, args)]
        with log.open("x") as stream, redirect_stdout(stream), redirect_stderr(stream):
            try:
                runpy.run_path(sys.argv[0], run_name="__main__")
            except SystemExit as result:
                code = result.code if type(result.code) is int else (0 if result.code is None else 1)
                if code and not isinstance(result.code, int):
                    print(result.code, file=sys.stderr)
            except BaseException:
                code = 1
                traceback.print_exc()
                raise
            finally:
                active = [logging.getLogger()] + [item for item in logging.Logger.manager.loggerDict.values()
                                                 if isinstance(item, logging.Logger)]
                for item in active:
                    for handler in item.handlers[:]:
                        item.removeHandler(handler)
                        if handler not in old_handlers:
                            handler.flush(); handler.close()
                    if item in saved:
                        item.handlers, item.level, item.propagate, item.disabled = saved[item]
    except BaseException:
        code = 1
        raise
    finally:
        sys.argv = old
        for item, values in saved.items():
            item.handlers, item.level, item.propagate, item.disabled = values
        new_json(output / (name + ".command.json"),
                 {"stock_source": str((SOURCE / "src" / name).relative_to(ROOT)),
                  "source_sha256": sha(SOURCE / "src" / name), "argv": list(map(str, args)),
                  "exit_code": code, "elapsed_seconds": time.monotonic() - started,
                  "log_sha256": sha(log)})
    if code:
        raise RuntimeError("Stock author command failed: " + name)


def frame_prepare(manifest_record, dataset_record, output, cfg):
    """CPU observations only; no truth arrays, native labels or model construction."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Observed frame preparation is CPU-only")
    import numpy as np
    import tskit
    if manifest_record != cfg["observed_manifest"] or dataset_record != cfg["dataset_manifest"]:
        raise ValueError("Exact reviewed completed root manifest bindings required")
    manifest = json.loads(checked(manifest_record, observed=True).read_text())
    if (manifest["kind"] != "biological_paths_relate_v1" or manifest["status"] !=
            "four_development_parents_inferred_pre_and_post_five_iterations"):
        raise ValueError("All four completed post-five-iteration parents required")
    entries = manifest["replicates"]
    if len(entries) != 4 or {entry["parent_id"] for entry in entries} != set(cfg["parent_ids"]):
        raise ValueError("Exact four-parent cohort required")
    dataset = json.loads(checked(dataset_record).read_text())
    raw_entries = {row["parent_id"]: row for row in dataset["replicates"]}
    if (len(raw_entries) != 4 or set(raw_entries) != set(PARENT_SPLITS) or
            any(row["split"] != PARENT_SPLITS[row["parent_id"]] for row in dataset["replicates"])):
        raise ValueError("Exact declared development split partition required")
    prepared = {"kind": "official_argmix_observed_frames_v1", "parents": {},
                "manifest": manifest_record, "dataset_manifest": dataset_record,
                "truth_opened": False, "config_sha256": sha(CONFIG), "owned_code_sha256": cfg["owned_code_sha256"]}
    for entry in entries:
        parent = entry["parent_id"]
        if entry["split"] != PARENT_SPLITS[parent]:
            raise ValueError("Completed observed parent split differs")
        if entry["status"] != "inferred_both_conditions" or entry["inferred_tree"]["condition"] != "post_five_size_iterations_primary":
            raise ValueError("Explicit primary inferred condition required")
        tree_path = checked(entry["inferred_tree"], observed=True)
        roster_path = checked(entry["observed_sample_manifest"], observed=True)
        rows = json.loads(roster_path.read_text())
        mapping = entry["sample_index_to_inferred_tree_node_id"]
        if (len(rows) != 280 or any(type(row["sample_index"]) is not int or
                type(row["inferred_tree_node_id"]) is not int or type(row["observed_node_id"]) is not int for row in rows) or
                [item["sample_index"] for item in mapping] != list(range(280)) or
                any(type(item["sample_index"]) is not int or type(item["inferred_tree_node_id"]) is not int for item in mapping) or
                [row["sample_index"] for row in rows] != list(range(280)) or
                len(mapping) != 280 or any(item["inferred_tree_node_id"] != item["sample_index"] for item in mapping) or
                any(row["inferred_tree_node_id"] != row["sample_index"] or row["observed_node_id"] != row["inferred_tree_node_id"]
                    for row in rows)):
            raise ValueError("Actual mapping differs from stock truth resolver's 2*sample-row+hap rule; no reindexing")
        refs = [row for row in rows if row["role"] == "pure_source_reference"]
        queries = [row for row in rows if row["role"] == "query"]
        if (len(refs) != 96 or len(queries) != 120 or
                sum(row["role"] == "mixed_reference_only" for row in rows) != 64 or
                any(row["role"] not in {"query", "pure_source_reference", "mixed_reference_only"} for row in rows) or
                any(type(row["observed_reference_class"]) is not int or not 0 <= row["observed_reference_class"] < 4 for row in refs) or
                any(not math.isfinite(float(row["age_generations"])) or row["age_generations"] < 0 for row in rows) or any(
                row["role"] != "pure_source_reference" and row["observed_reference_class"] is not None for row in rows)):
            raise ValueError("Only 96 observed pure references may carry class labels; 120 queries required")
        if any(sum(row["observed_reference_class"] == c for row in refs) != 24 for c in range(4)):
            raise ValueError("Exactly24 pure haplotypes per observed class required")
        contract = entry["coordinate_contract"]
        grid = checked(contract["fixed_grid"], observed=True)
        with np.load(grid, allow_pickle=False) as z:
            positions = z["positions_bp"].copy()
        if not np.array_equal(positions, np.arange(50, 2000000, 100, dtype=np.float64)):
            raise ValueError("Original fixed100bp grid differs")
        retained = contract["retained_fixed100bp_indexes"]
        if any(type(index) is not int or not 0 <= index < len(positions) for index in retained):
            raise ValueError("Retained grid indexes must be actual in-range integers")
        indexes = np.asarray(retained, dtype=np.int64)
        if len(indexes) == 0 or np.any(np.diff(indexes) != 1):
            raise ValueError("Exact continuous retained-center grid contract required")
        offset = float(contract["local_to_original_offset_bp"])
        coverage = np.asarray(contract["coverage_original_bp0"], dtype=np.float64)
        if (coverage.shape != (2,) or not np.isfinite(coverage).all() or not 0 <= coverage[0] < coverage[1] <= 2000000 or
                contract["normalization_length_bp"] != 2000000 or
                np.any(positions[indexes] < coverage[0]) or np.any(positions[indexes] >= coverage[1])):
            raise ValueError("Retained local/original coordinate contract differs")
        ts = tskit.load(tree_path)
        tables = ts.tables
        if (tables.metadata or ts.num_individuals or np.any(tables.nodes.population != -1) or
                np.any(tables.nodes.individual != -1) or any(len(table.metadata) for table in
                (tables.nodes, tables.edges, tables.sites, tables.mutations, tables.individuals, tables.populations, tables.migrations))):
            raise ValueError("Observed graph must have removed all ancestry/population/individual annotations")
        if (set(ts.samples()) != set(range(280)) or ts.sequence_length != coverage[1] - coverage[0] or
                coverage[0] != offset or ts.num_populations or ts.num_migrations or
                any(abs(ts.node(row["inferred_tree_node_id"]).time - row["age_generations"]) > 1e-6 for row in rows)):
            raise ValueError("Observed graph, actual ages, sample IDs or physical coverage differ")
        d = output / parent; d.mkdir()
        references = d / "references.tsv"
        with references.open("x") as stream:
            for row in rows:
                pop = cfg["class_order"][row["observed_reference_class"]] if row["role"] == "pure_source_reference" else (
                    "QUERY" if row["role"] == "query" else "UNLABELED_REF")
                stream.write(f"{row['inferred_tree_node_id']} {pop} {row['age_generations']}\n")
        sample = d / "native_sample.sample"
        with sample.open("x") as stream:
            stream.write("ID_1 ID_2 missing\n0 0 0\n")
            for index in range(0, 280, 2):
                left, right = rows[index:index + 2]
                if left["sample_id"] != right["sample_id"] or (left["haplotype"], right["haplotype"]) != (0, 1):
                    raise ValueError("Observed diploid order differs")
                stream.write(f"{left['sample_id']} {left['sample_id']} 0\n")
        if (len({row["sample_id"] for row in rows}) != 140 or
                sum(row["observed_cohort"] == "present_day" for row in queries) != 64):
            raise ValueError("Exact140 diploids and64 present/56 ancient queries required")
        sites = ts.tables.sites.position.copy()
        if (not np.array_equal(sites, sites.astype(np.int64)) or not len(sites) or
                np.any(np.diff(sites) <= 0) or np.any(sites < 0) or np.any(sites >= ts.sequence_length)):
            raise ValueError("Native SNP labels require exact integer local positions")
        edges = np.r_[positions[indexes] - 50, positions[indexes[-1]] + 50]
        edges = np.clip(edges, coverage[0], coverage[1])
        np.savez_compressed(d / "observed_coordinates.npz", grid_indexes=indexes, positions_bp=positions[indexes],
                            local_positions_bp=positions[indexes] - offset, bin_edges_bp=edges,
                            local_site_positions_bp=sites.astype(np.int64), original_site_positions_bp=sites + offset,
                            query_nodes=np.asarray([row["inferred_tree_node_id"] for row in queries], dtype=np.int32))
        new_json(d / "observed_roster.json", rows)
        prepared["parents"][parent] = {"parent_id": parent, "split": entry["split"],
            "tree": entry["inferred_tree"], "references": {"path": str(references.relative_to(ROOT)), "sha256": sha(references)},
            "native_sample": {"path": str(sample.relative_to(ROOT)), "sha256": sha(sample)},
            "coordinates": {"path": str((d / "observed_coordinates.npz").relative_to(ROOT)),
                            "sha256": sha(d / "observed_coordinates.npz")},
            "roster": {"path": str((d / "observed_roster.json").relative_to(ROOT)), "sha256": sha(d / "observed_roster.json")},
            "source_observed_entry": entry, "external_dataset_entry": raw_entries[parent],
            "external_dataset_directory": raw_entries[parent]["directory"],
            "local_offset_bp": offset, "coverage_original_bp0": coverage.tolist(),
            "normalization_length_bp": 2000000, "pure_refs": 96, "queries": 120,
            "stock_native_ID_correspondence_verified": True, "truth_opened": False}
    (output / "local_uniform_map.tsv").write_text("position Genetic_Map(cM)\n0 0\n2000000 2\n")
    prepared["local_map"] = {"path": str((output / "local_uniform_map.tsv").relative_to(ROOT)),
                             "sha256": sha(output / "local_uniform_map.tsv")}
    prepared["map_adaptation"] = "uniform1cM/Mb translated to local origin; cM-span/delta differences preserved"
    new_json(output / "observed_frames.json", prepared)
    verify_frames({"path": str((output / "observed_frames.json").relative_to(ROOT)), "sha256": sha(output / "observed_frames.json")}, cfg)
    return prepared


def external_truth(frame, purpose, seal=None):
    """Explicit external capability; no ancestral truth enters observed inputs."""
    import numpy as np
    if purpose == "TRAIN_gradients" and frame["split"] != "train":
        raise ValueError("TRAIN gradient supervision requires TRAIN parent")
    if purpose in {"native_VAL_selection_only", "sealed_external_VAL_score"} and frame["split"] != "val":
        raise ValueError("VAL capability requires declared VAL parent")
    if purpose == "sealed_external_VAL_score":
        checked({"path": seal["path"], "sha256": seal["sha256"]}, observed=True)
    if purpose not in {"TRAIN_gradients", "native_VAL_selection_only", "sealed_external_VAL_score"}:
        raise ValueError("Undeclared truth capability")
    rows = json.loads(checked(frame["roster"], observed=True).read_text())
    queries = [row for row in rows if row["role"] == "query"]
    raw_nodes = [row["source_observed_node_id"] for row in queries]
    record = frame["external_dataset_entry"]["files"]["truth/full_truth_intervals.npz"]
    path = checked({"path": str(local(frame["external_dataset_directory"]) / "truth/full_truth_intervals.npz"),
                    "sha256": record["sha256"]})
    with np.load(path, allow_pickle=False) as z:
        intervals = z["intervals_bp"].copy()
        labels = z["source_classes_0based"].copy()
        actual_nodes = z["query_raw_node_ids"].copy()
    if (intervals.ndim != 2 or intervals.shape[1:] != (2,) or not len(intervals) or
            not np.isfinite(intervals).all() or labels.shape != (len(intervals), 120) or not np.array_equal(actual_nodes, raw_nodes) or
            intervals[0, 0] != 0 or intervals[-1, 1] != 2000000 or
            not np.array_equal(intervals[:-1, 1], intervals[1:, 0]) or
            np.any(intervals[:, 1] <= intervals[:, 0]) or labels.dtype.kind not in "iu" or
            np.any((labels < 0) | (labels > 3))):
        raise ValueError("Exact external full-span C4/query correspondence required")
    return intervals, labels, queries, {"path": str(path.relative_to(ROOT)), "sha256": sha(path), "purpose": purpose}


def native_archives(frames_record, output, cfg):
    """Stock Rust labels/extraction/chunking; VAL labels select, never enter gradients."""
    import numpy as np
    import yaml
    frames = verify_frames(frames_record, cfg)
    pairs = {"train": [], "val": []}
    provenance = {}
    for parent, frame in frames["parents"].items():
        intervals, labels, queries, truth = external_truth(
            frame, "TRAIN_gradients" if frame["split"] == "train" else "native_VAL_selection_only")
        with np.load(checked(frame["coordinates"], observed=True), allow_pickle=False) as z:
            local_sites, physical_sites = z["local_site_positions_bp"], z["original_site_positions_bp"]
        indexes = np.searchsorted(intervals[:, 1], physical_sites, side="right")
        if np.any(indexes >= len(intervals)) or np.any(physical_sites < 0):
            raise ValueError("Original observed SNP lies outside external complete domain")
        path = output / (parent + ".native_truth.npz")
        np.savez_compressed(path, variant_pos=local_sites.astype(np.int64),
                            truth__hap_labels=labels[indexes].astype(np.int16),
                            truth__sample_ids=np.asarray([row["sample_id"] for row in queries[::2]], dtype=str),
                            ancestry_names=np.asarray(cfg["class_order"], dtype=str))
        pairs[frame["split"]].append({"rel": str(checked(frame["tree"], observed=True)),
            "references": str(checked(frame["references"], observed=True)),
            "relate_sample": str(checked(frame["native_sample"], observed=True)), "truth_npz": str(path)})
        provenance[parent] = dict(truth, native_truth={"path": str(path.relative_to(ROOT)), "sha256": sha(path)},
                                  label_frame="observed SNP local bp0; labels joined at original SNP bp0",
                                  labels_used_for_feature_or_query_eligibility=False)
    train_pairs, val_pairs = output / "train_pairs.yaml", output / "val_pairs.yaml"
    train_pairs.write_text(yaml.safe_dump(pairs["train"], sort_keys=False))
    val_pairs.write_text(yaml.safe_dump(pairs["val"], sort_keys=False))
    (output / "local_uniform_map.tsv").write_bytes(checked(frames["local_map"], observed=True).read_bytes())
    backend = verify_backend(backend_config(cfg, output, train_pairs, val_pairs), cfg)
    config_path = output / "backend.yaml"; config_path.write_text(yaml.safe_dump(backend, sort_keys=False))
    commands = []
    for split in ("train", "val"):
        archive = output / (split + ".pt")
        argv = [str(CLI), "--config", str(config_path), "--split", split, "--output", str(archive)]
        env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = ""
        started = time.monotonic()
        with (output / (split + ".Rust.log")).open("xb") as log:
            result = subprocess.run(argv, cwd=SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT)
        command = {"argv": argv, "exit_code": result.returncode, "elapsed_seconds": time.monotonic() - started,
                   "stock_binary_sha256": sha(CLI), "log_sha256": sha(output / (split + ".Rust.log"))}
        commands.append(command)
        new_json(output / (split + ".command.json"), command)
        if result.returncode or not archive.is_file():
            raise RuntimeError("Actual unchanged Rust build_training failed: " + split)
    _, _, joint, trainer, loader, libraries = source_imports()
    loader.validate_config(backend)
    archive_info = {}
    expected_vocab_size = len(joint.build_token_to_id_from_config(backend))
    for split, code in (("train", 0), ("val", 1)):
        payload = trainer._load_gt_gru_split_pt(str(output / (split + ".pt")), expected_split_code=code)
        if (payload["input_ids"].shape[1:] != (1024, 16) or not payload["row_mask"].any() or
                payload["base_encoder"] != "tmrca_transformer" or payload["format_version"] != 2 or
                payload["pad_id"] != 0 or payload["vocab_size"] != expected_vocab_size or
                payload["cm_features"].shape[-1] != 2):
            raise ValueError("Actual source chunk/row/sequence contract differs")
        if (not np.isfinite(payload["cm_features"].numpy()).all() or
                np.any(payload["cm_features"].numpy()[..., 0] < 0)):
            raise ValueError("Stored source map span channels invalid")
        valid = payload["labels"][payload["row_mask"]]
        if (valid < 0).any() or (valid >= 4).any():
            raise ValueError("Native supervised archive labels are not C4")
        archive_info[split] = {"path": str((output / (split + ".pt")).relative_to(ROOT)),
            "sha256": sha(output / (split + ".pt")), "chunks": payload["num_chunks"],
            "valid_rows_including_overlap": int(payload["row_mask"].sum()), "metadata": payload["metadata"].tolist()}
    prepared = {"kind": "official_argmix_native_biological_archives_v1", "frames": frames_record,
                "config_sha256": sha(CONFIG), "owned_code_sha256": cfg["owned_code_sha256"], "backend": {"path": str(config_path.relative_to(ROOT)),
                "sha256": sha(config_path)}, "archives": archive_info, "commands": commands,
                "train_pairs": {"path": str(train_pairs.relative_to(ROOT)), "sha256": sha(train_pairs)},
                "val_pairs": {"path": str(val_pairs.relative_to(ROOT)), "sha256": sha(val_pairs)},
                "local_map": {"path": str((output / "local_uniform_map.tsv").relative_to(ROOT)),
                              "sha256": sha(output / "local_uniform_map.tsv")},
                "truth_provenance": provenance, "native_VAL_selection": "source smoothed unweighted mean of batch masked losses",
                "full_support_external_VAL_metrics": "pending sealed native inference",
                "stock_source_unchanged": True}
    verify_frames(frames_record, cfg)
    new_json(output / "native_prepared.json", prepared)
    return prepared


def state_hash(model):
    records = {}
    for name, value in model.state_dict().items():
        array = value.detach().cpu().contiguous().numpy()
        records[name] = {"dtype": str(array.dtype), "shape": list(array.shape),
                         "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
    return hashlib.sha256(canonical(records)).hexdigest()


def qualify_gpu(cfg, output, runtime, modules):
    torch, ancpath, joint, trainer, loader = modules
    config = backend_config(cfg, output)
    loader.validate_config(config)
    reproducibility = importlib.import_module("utils.reproducibility")
    reproducibility.seed_everything(50)
    model = joint.build_joint_model_from_config(config, device=torch.device("cuda"))
    before = state_hash(model)
    vocab = joint.build_token_to_id_from_config(config)
    ids = torch.tensor([[vocab["<SAMPLE>"]] + [vocab[cfg["class_order"][i % 4]] for i in range(15)]],
                       device="cuda").expand(1024, 16).reshape(1, 1024, 16).clone()
    tmrca = torch.ones((1, 1024, 16, 16), dtype=torch.int16, device="cuda")
    mask = torch.ones((1, 1024), dtype=torch.bool, device="cuda")
    cm = torch.zeros((1, 1024, 2), device="cuda"); cm[..., 0] = .001
    labels = torch.arange(1024, device="cuda").remainder(4).reshape(1, 1024)
    _, logits = model.forward_chunks(ids, tmrca, mask, cm, pad_id=vocab["<PAD>"])
    loss = trainer._masked_joint_loss(logits, labels, mask, cm, label_smoothing=.1,
                                     loss_weighting={"mode": "none"})
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not torch.isfinite(loss) or not gradients or any(not torch.isfinite(g).all() for g in gradients):
        raise RuntimeError("Full actual source transformer+GRU CUDA forward/backward failed")
    if state_hash(model) != before:
        raise RuntimeError("No-optimizer qualification altered weights")
    fixture = output / "UNTRAINED_qualification_checkpoint.pt"
    joint.save_joint_model_checkpoint(fixture, model=model, config=config, epoch=0,
                                     val_loss=0., run_name="qualification_not_a_fit")
    import yaml
    config_path = output / "qualification_backend.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    export = output / "qualification_base.pt"
    run_author("export_torchscript.py", ["--model_checkpoint", fixture, "--config", config_path,
        "--output", export, "--device", "cpu", "--export_mode", "logits_only",
        "--ref_pops", *cfg["class_order"], "--predict_pops", "QUERY"], output)
    if not export.is_file() or not Path(str(export).replace(".pt", "_vocab.json")).is_file():
        raise RuntimeError("Stock export swallowed a failure; required file absent")
    model.base_model.eval()
    with torch.no_grad():
        expected = model.base_model(ids[0, :1], tmrca[0, :1], ids[0, :1].ne(0)).argmax(-1).tolist()
    native = ancpath.RustTorchScriptInference()
    native.load_model(str(export), str(export).replace(".pt", "_vocab.json"), True)
    tokens = [["<SAMPLE>"] + [cfg["class_order"][i % 4] for i in range(15)]]
    actual = native.predict_batch(tokens, [[[1] * 16 for _ in range(16)]])
    if actual != expected or native.get_device_info() != "Cuda(0)":
        raise RuntimeError("Stock Rust TorchScript actual GPU fixture/device mismatch")
    return {"status": "passed", "phase": "qualify-gpu", "actual_source_joint_GPU_forward_backward": True,
            "actual_stock_Rust_TorchScript_GPU_fixture": True, "actual_cuda": True, "runtime": runtime,
            "config_sha256": sha(CONFIG), "owned_code_sha256": cfg["owned_code_sha256"],
            "state_unchanged_sha256": before, "fixture_chunks": 1, "fixture_rows": 1024,
            "optimizer_step": False, "biological_data_read": False, "source_TREE_sha256": SOURCE_TREE_SHA}


def reference_metadata_amendment(prepared, cfg):
    """Verify original pairs and observation roles before supplying stock count metadata."""
    import yaml
    frames = verify_frames(prepared["frames"], cfg)
    expected_counts = {name: 24 for name in cfg["class_order"]}
    expected_counts.update(QUERY=120, UNLABELED_REF=64)
    reference_records, summaries, eligibility_signatures = {}, {}, []
    for parent in PARENT_SPLITS:
        frame = frames["parents"][parent]
        reference_path = checked(frame["references"], observed=True)
        rows = []
        for line in reference_path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 3:
                raise ValueError("Observed reference metadata requires exactly three fields")
            node, pop, age = int(parts[0]), parts[1], float(parts[2])
            if not math.isfinite(age) or age < 0:
                raise ValueError("Observed reference age must be finite and nonnegative")
            rows.append((node, pop, age))
        roster = json.loads(checked(frame["roster"], observed=True).read_text())
        expected = []
        for row in roster:
            role, klass = row["role"], row["observed_reference_class"]
            if role == "pure_source_reference":
                if type(klass) is not int or not 0 <= klass < 4:
                    raise ValueError("Pure reference class must be an observed C4 role")
                pop = cfg["class_order"][klass]
            elif role in {"query", "mixed_reference_only"} and klass is None:
                pop = "QUERY" if role == "query" else "UNLABELED_REF"
            else:
                raise ValueError("Query/mixed reference eligibility changed")
            expected.append((row["inferred_tree_node_id"], pop, float(row["age_generations"])))
        counts = {pop: sum(value[1] == pop for value in rows) for pop in {r[1] for r in rows}}
        if (rows != expected or len(rows) != 280 or {r[0] for r in rows} != set(range(280))
                or counts != expected_counts):
            raise ValueError("Observed QUERY120/pure96/unlabeled64 metadata/roster eligibility differs")
        eligibility = [[node, pop] for node, pop, age in rows]
        digest = hashlib.sha256(canonical(eligibility)).hexdigest()
        eligibility_signatures.append(digest)
        reference_records[parent] = frame["references"]
        summaries[parent] = {"reference_record": frame["references"], "counts": counts,
            "rows": len(rows), "unique_node_ids": len({r[0] for r in rows}),
            "eligibility_sha256": digest, "exact_observed_roster_join": True}
    if len(set(eligibility_signatures)) != 1:
        raise ValueError("Parent reference eligibility is not equivalent for stock counting")
    for split in ("train", "val"):
        actual = yaml.safe_load(checked(prepared[split + "_pairs"]).read_text())
        expected = []
        for parent in PARENT_SPLITS:
            if PARENT_SPLITS[parent] != split:
                continue
            frame = frames["parents"][parent]
            expected.append({
                "rel": str(checked(frame["tree"], observed=True)),
                "references": str(checked(frame["references"], observed=True)),
                "relate_sample": str(checked(frame["native_sample"], observed=True)),
                "truth_npz": str(checked(prepared["truth_provenance"][parent]["native_truth"]))})
        if canonical(actual) != canonical(expected):
            raise ValueError("Native pair tree/reference/sample/label pointers changed")
    chosen = cfg["compatibility_amendment"]["metadata_reference_parent"]
    return {"kind": "official_argmix_stock_reference_metadata_v3",
        "selected_parent": chosen, "selected_reference": reference_records[chosen],
        "parents": summaries, "stock_available_predict_haplotypes": 120,
        "scope": "stock startup count and training-data log; no model/archive/eligibility alteration",
        "inherited_products": {
            "frames": {"record": prepared["frames"], "original_config_sha256": frames["config_sha256"],
                "original_owned_code_sha256": frames["owned_code_sha256"]},
            "native_prepared": {"original_config_sha256": prepared["config_sha256"],
                "original_owned_code_sha256": prepared["owned_code_sha256"]}}}


def stock_train(prepared_record, output, cfg, modules):
    import yaml
    torch, ancpath, joint, trainer, loader = modules
    prepared = verify_prepared(prepared_record, cfg)
    backend_path = checked(prepared["backend"])
    backend = verify_backend(yaml.safe_load(backend_path.read_text()), cfg)
    if (backend["paths"]["train_pairs"] != str(checked(prepared["train_pairs"])) or
            backend["paths"]["val_pairs"] != str(checked(prepared["val_pairs"])) or
            backend["relate"]["recomb_map"] != str(checked(prepared["local_map"], observed=True))):
        raise ValueError("Actual backend pair/map bindings differ from native preparation")
    metadata_info = reference_metadata_amendment(prepared, cfg)
    metadata_info["inherited_products"]["native_prepared"]["record"] = prepared_record
    backend["paths"]["references"] = str(checked(metadata_info["selected_reference"], observed=True))
    verify_backend(backend, cfg, references_record=metadata_info["selected_reference"])
    new_json(output / "references_metadata_amendment.json", metadata_info)
    backend["paths"]["save_dir"] = str(output / "checkpoints")
    actual_config = output / "training_backend.yaml"; actual_config.write_text(yaml.safe_dump(backend, sort_keys=False))
    importlib.import_module("utils.reproducibility").seed_everything(50)
    initial = joint.build_joint_model_from_config(backend, device=torch.device("cpu"))
    initial_sha = state_hash(initial)
    initial_classifier_sha = state_hash(initial.base_model.classifier); del initial
    pointer = output / "stock_checkpoint.path"
    run_author("train_joint_model.py", ["--config", actual_config,
        "--train-pt", checked(prepared["archives"]["train"]),
        "--val-pt", checked(prepared["archives"]["val"]), "--checkpoint-output", pointer], output)
    checkpoint = local(pointer.read_text().strip())
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model, meta = joint.load_joint_model_checkpoint(checkpoint, device=torch.device("cpu"))
    final_sha = state_hash(model)
    checkpoint_meta = payload["joint_model"]
    if (payload.get("format") != "joint_interval_gru_v1" or
            canonical(payload["config"]) != canonical(backend) or checkpoint_meta != meta["joint_model"] or
            initial_sha == final_sha or checkpoint_meta["epoch"] < 1 or
            not math.isfinite(float(checkpoint_meta["best_val_loss"])) or
            state_hash(model.base_model.classifier) != initial_classifier_sha):
        raise RuntimeError("Actual unchanged native optimizer fit/checkpoint selection not established")
    verify_prepared(prepared_record, cfg)
    return {"kind": "official_argmix_biological_stock_fit_v1", "status": "passed",
            "checkpoint": {"path": str(checkpoint.relative_to(ROOT)), "sha256": sha(checkpoint)},
            "metadata_reference_counting": metadata_info,
            "compatibility_amendment": cfg["compatibility_amendment"],
            "native_prepared": prepared_record, "native_backend": {"path": str(actual_config.relative_to(ROOT)), "sha256": sha(actual_config)},
            "initial_model_state_sha256": initial_sha, "selected_model_state_sha256": final_sha,
            "native_selected_epoch": int(checkpoint_meta["epoch"]), "native_selection_loss": float(checkpoint_meta["best_val_loss"]),
            "unused_base_classifier_unchanged_sha256": initial_classifier_sha,
            "config_sha256": sha(CONFIG), "owned_code_sha256": cfg["owned_code_sha256"],
            "source_epochs_cap": 20, "source_patience": 2, "source_seed": 50,
            "selection": "unchanged source smoothed masked unweighted VAL batch-mean loss",
            "external_full_support_VAL_selection_or_scoring": "not substituted for native inner selection",
            "standalone_base_classifier": "unused by joint training; refined C4 output is primary"}


def stock_streamed_replay_chunks(rows, joint, recomb_map_path):
    """Source helper plus stock stream's exact-end trailing overlap; no model change."""
    import numpy as np
    import torch
    grouped = joint.group_row_indices(rows, group_across_shards_by_node_id=False)
    natural = {}
    for index, row in enumerate(rows):
        natural.setdefault((str(row["shard_id"]), int(row["node_id"])), []).append(index)
    if grouped != natural:
        raise RuntimeError("HDF5 per-query rows are not in stock streamed chronological order")
    chunks = joint.build_joint_feature_chunks(rows, recomb_map_path=recomb_map_path,
                                               chunk_rows=1024, chunk_stride=512)
    tensors = ("features", "labels", "row_mask", "cm_features", "row_indices")
    extras = 0
    for row_indexes in grouped.values():
        size = len(row_indexes)
        if size >= 1024 and size % 512 == 0:
            tail_indexes = row_indexes[size - 512:]
            tail = joint.build_joint_feature_chunks([rows[index] for index in tail_indexes],
                recomb_map_path=recomb_map_path, chunk_rows=1024, chunk_stride=512)
            if tail["num_chunks"] != 1 or int(tail["row_mask"].sum()) != 512:
                raise RuntimeError("Stock exact-end tail helper geometry differs")
            tail["row_indices"][0, :512] = torch.as_tensor(tail_indexes, dtype=torch.long)
            for key in tensors:
                chunks[key] = torch.cat([chunks[key], tail[key]], dim=0)
            extras += 1
    chunks["num_chunks"] = int(chunks["features"].shape[0])
    # The stock stream retains previous-row delta across overlaps; the standalone
    # helper resets it at each chunk. Restore the stored second channel exactly,
    # although the unchanged default consumes raw span only.
    previous = {}
    for row_indexes in grouped.values():
        for index, prior in zip(row_indexes[1:], row_indexes[:-1]):
            previous[index] = prior
    recomb = joint.RecombinationMapInterpolator.from_path(recomb_map_path)
    start_cm = recomb.at_bp(np.asarray([row["tree_start"] for row in rows], dtype=np.float64)) * 100.0
    for chunk in range(chunks["num_chunks"]):
        row_index = int(chunks["row_indices"][chunk, 0])
        prior = previous.get(row_index)
        chunks["cm_features"][chunk, 0, 1] = 0.0 if prior is None else float(start_cm[row_index] - start_cm[prior])
    return chunks, {"standalone_source_helper": "build_joint_feature_chunks",
                    "native_geometry": "unchanged streamed HDF5 full chunks plus final trailing overlap",
                    "exact_end_extra_trailing_chunks": extras,
                    "stored_delta_channel_matches_stock_stream": True,
                    "active_cm_features": ["cM_span"]}


def infer_and_seal(fit_record, output, cfg, modules):
    """Stock export -> native Rust HDF5 -> stock GRU; preserve original chunk logits."""
    import numpy as np
    import h5py
    import yaml
    torch, ancpath, joint, trainer, loader = modules
    fit = stage(fit_record, "official_argmix_biological_stock_fit_v1", cfg)
    checkpoint = checked(fit["checkpoint"])
    prepared = verify_prepared(fit["native_prepared"], cfg)
    frames = verify_frames(prepared["frames"], cfg)
    metadata_info = reference_metadata_amendment(prepared, cfg)
    metadata_info["inherited_products"]["native_prepared"]["record"] = fit["native_prepared"]
    if fit.get("metadata_reference_counting") != metadata_info or fit.get("compatibility_amendment") != cfg["compatibility_amendment"]:
        raise ValueError("Fitted metadata-reference amendment provenance differs")
    backend = verify_backend(yaml.safe_load(checked(fit["native_backend"]).read_text()), cfg,
        references_record=metadata_info["selected_reference"])
    base_export = output / "source_base_logits_and_final.pt"
    run_author("export_torchscript.py", ["--model_checkpoint", checkpoint,
        "--config", checked(fit["native_backend"]), "--output", base_export,
        "--device", "cpu", "--export_mode", "logits_and_final",
        "--ref_pops", *cfg["class_order"], "--predict_pops", "QUERY"], output)
    if not base_export.is_file() or not Path(str(base_export).replace(".pt", "_vocab.json")).is_file():
        raise RuntimeError("Source exporter swallowed failure; expected artifacts absent")
    seals = []
    for parent, frame in frames["parents"].items():
        if frame["split"] != "val":
            continue
        d = output / parent; d.mkdir()
        pair_path = d / "inference_pairs.yaml"
        pair_path.write_text(yaml.safe_dump([{"rel": str(checked(frame["tree"], observed=True)),
            "references": str(checked(frame["references"], observed=True))}], sort_keys=False))
        infer_cfg = dict(backend)
        infer_cfg["processing"] = dict(backend["processing"], exclude_edge_trees=False)
        verify_backend(infer_cfg, cfg, inference=True, references_record=metadata_info["selected_reference"])
        config_path = d / "full_support_inference.yaml"
        config_path.write_text(yaml.safe_dump(infer_cfg, sort_keys=False))
        base_h5 = d / "source_base.h5"
        run_author("run_inference.py", ["--config", config_path, "--val_pairs", pair_path,
            "--references", checked(frame["references"], observed=True),
            "--torchscript_model", base_export,
            "--vocab_file", str(base_export).replace(".pt", "_vocab.json"),
            "--checkpoint", checkpoint, "--output_dir", d, "--output_name", "source_base",
            "--probs_h5", base_h5, "--h5_feature_source", "final_layer",
            "--h5_include_probs", "true", "--h5_final_layer_dim", 16,
            "--device", "cuda", "--num_data_workers", 1,
            "--max_count", 0, "--max_predict_haplotypes", 0, "--drop_top_m_nearest", 0], d)
        if not base_h5.is_file():
            raise RuntimeError("Stock Rust HDF5 inference output absent")
        refined_dir = d / "stock_refinement"; refined_dir.mkdir()
        run_author("run_joint_inference.py", ["--config", config_path, "--checkpoint", checkpoint,
            "--input_h5", base_h5, "--output_dir", refined_dir,
            "--output_prefix", "source_refined", "--device", "cuda"], d)
        refined_h5 = refined_dir / "source_refined.h5"
        rows = joint.load_joint_rows_from_h5(base_h5)
        chunks, replay_geometry = stock_streamed_replay_chunks(rows, joint, backend["relate"]["recomb_map"])
        model, metadata = joint.load_joint_model_checkpoint(checkpoint, device=torch.device("cuda"))
        before = state_hash(model)
        logits = []
        with torch.no_grad():
            for start in range(0, int(chunks["num_chunks"]), 256):
                value = model.refine_feature_chunks(chunks["features"][start:start + 256].cuda(),
                    chunks["row_mask"][start:start + 256].cuda(),
                    chunks["cm_features"][start:start + 256].cuda())
                if value.dtype != torch.float32 or not torch.isfinite(value).all():
                    raise RuntimeError("Original source per-chunk refined logits invalid")
                logits.append(value.cpu())
        chunk_logits = torch.cat(logits)
        chunk_probs = torch.softmax(chunk_logits, dim=-1).numpy()
        aggregated = joint.aggregate_chunk_probabilities(chunk_probs,
            chunks["row_indices"].numpy(), chunks["row_mask"].numpy(), num_rows=len(rows))
        with h5py.File(refined_h5, "r") as h5:
            native_probs = np.asarray(h5["refined_probs"], dtype=np.float32)
            nodes = np.asarray(h5["node_id"], dtype=np.int32)
            left = np.asarray(h5["tree_start"], dtype=np.float64)
            right = np.asarray(h5["tree_end"], dtype=np.float64)
        if (native_probs.shape != (len(rows), 4) or not np.isfinite(native_probs).all() or
                np.any(native_probs < 0) or np.any(native_probs > 1) or
                not np.isfinite(left).all() or not np.isfinite(right).all() or np.any(right <= left) or
                not np.allclose(native_probs.sum(-1), 1, atol=1e-5, rtol=0) or
                not np.array_equal(nodes, [row["node_id"] for row in rows]) or
                not np.array_equal(left, [row["tree_start"] for row in rows]) or
                not np.array_equal(right, [row["tree_end"] for row in rows])):
            raise RuntimeError("Stock refined HDF5 row/probability identity differs")
        error = float(np.max(np.abs(native_probs.astype(np.float64) - aggregated)))
        if error > 1e-6 or state_hash(model) != before:
            raise RuntimeError("Source chunk logit replay differs from stock aggregation or changed weights")
        logits_path = d / "original_refined_chunk_logits_float32.npz"
        with logits_path.open("xb") as writer:
            np.savez_compressed(writer, logits=chunk_logits.numpy(),
                row_indices=chunks["row_indices"].numpy(), row_mask=chunks["row_mask"].numpy(),
                node_ids=nodes, local_tree_left_bp=left, local_tree_right_bp=right)
        with np.load(checked(frame["coordinates"], observed=True), allow_pickle=False) as z:
            coords = {key: z[key].copy() for key in z.files}
        grid = np.full((len(coords["grid_indexes"]), 120, 4), np.nan, dtype=np.float32)
        present = np.zeros(grid.shape[:2], dtype=bool)
        columns = {int(node): index for index, node in enumerate(coords["query_nodes"])}
        for i, node in enumerate(nodes):
            if int(node) not in columns:
                raise RuntimeError("Stock inference returned a non-query node")
            column = columns[int(node)]
            mask = (coords["local_positions_bp"] >= left[i]) & (coords["local_positions_bp"] < right[i])
            if np.any(present[mask, column]):
                raise RuntimeError("Overlapping native output interval for one observed query")
            grid[mask, column] = native_probs[i]
            present[mask, column] = True
        path = d / "sealed_C4_grid_probabilities.npz"
        with path.open("xb") as writer:
            np.savez_compressed(writer, probabilities=grid, supported=present, **coords)
        seal = {"parent_id": parent, "grid_prediction": {"path": str(path.relative_to(ROOT)), "sha256": sha(path)},
            "original_chunk_logits": {"path": str(logits_path.relative_to(ROOT)), "sha256": sha(logits_path)},
            "stock_base_HDF5": {"path": str(base_h5.relative_to(ROOT)), "sha256": sha(base_h5)},
            "stock_refined_HDF5": {"path": str(refined_h5.relative_to(ROOT)), "sha256": sha(refined_h5)},
            "checkpoint": fit["checkpoint"], "frame": frame,
            "native_replay_max_probability_error": error, "replay_geometry": replay_geometry,
            "original_raw_logits_are_per_chunk_not_log_of_averaged_probability": True,
            "native_final_estimand": "source arithmetic average of overlapping-chunk softmax probabilities",
            "source_HDF5_base_feature_precision": "uint16 half-float bits decoded by stock source",
            "full_support_complete": bool(present.all()),
            "missing_grid_query_cells": int(np.sum(~present)),
            "truth_opened_for_external_score": False, "inference_only_edge_option": False}
        new_json(d / "prediction_seal.json", seal)
        seals.append(seal)
        del model, chunks, chunk_logits
        torch.cuda.empty_cache()
    if len(seals) != 2:
        raise RuntimeError("Both observed VAL parents must be sealed before external scoring")
    verify_prepared(fit["native_prepared"], cfg)
    checked(fit["checkpoint"])
    new_json(output / "all_VAL_predictions_sealed.json", {"seals": seals, "truth_scoring_started": False})
    return seals


def external_score(seal, output, cfg):
    """Full retained support proper score; support gaps never silently change denominator."""
    import numpy as np
    coordinates = checked(seal["grid_prediction"], observed=True)
    with np.load(coordinates, allow_pickle=False) as z:
        probs = z["probabilities"].astype(np.float64)
        supported = z["supported"]
        edges = z["bin_edges_bp"]
        indexes = z["grid_indexes"]
    frame = seal["frame"]
    intervals, labels, queries, provenance = external_truth(frame, "sealed_external_VAL_score", seal=seal["grid_prediction"])
    # Exact intersection join, independent of truth labels and label purity.
    bins, rows, spans = [], [], []
    i = j = 0
    while i < len(edges) - 1 and j < len(intervals):
        left, right = max(edges[i], intervals[j, 0]), min(edges[i + 1], intervals[j, 1])
        if right > left:
            bins.append(i); rows.append(j); spans.append(right - left)
        if edges[i + 1] <= intervals[j, 1]:
            i += 1
        else:
            j += 1
    bins = np.asarray(bins, dtype=np.int64)
    rows = np.asarray(rows, dtype=np.int64)
    spans = np.asarray(spans, dtype=np.float64)
    if abs(spans.sum() - (edges[-1] - edges[0])) > 1e-8:
        raise ValueError("Exact retained bin support does not join complete truth domain")
    cohorts = np.asarray([row["observed_cohort"] for row in queries])
    if sum(cohorts == "present_day") != 64 or len(cohorts) != 120:
        raise ValueError("External query cohort sizes differ")
    groups = {"primary_present_day_64": np.flatnonzero(cohorts == "present_day"),
              "secondary_ancient_56": np.flatnonzero(cohorts != "present_day"),
              "all_queries_120": np.arange(120)}
    reports = {}
    for name, columns in groups.items():
        total = float(spans.sum() * len(columns))
        missing = float(np.sum(spans[:, None] * ~supported[bins][:, columns]))
        report = {"queries": int(len(columns)), "full_retained_support_denominator_query_bp": total,
                  "missing_prediction_query_bp": missing, "complete_support": missing == 0,
                  "normalization_length_bp": 2000000, "uncovered_tail_extrapolation": False,
                  "no_truth_purity_or_eligibility_filter": True}
        if missing:
            report.update(NLL=None, accuracy=None, Brier=None, ECE=None,
                          metric_status="full-support score unavailable; missing predictions preserved, denominator unchanged")
            reports[name] = report
            continue
        nll = brier = good = infinity = 0.
        reliability = np.zeros((15, 3), dtype=np.float64)
        confusion = np.zeros((4, 4), dtype=np.float64)
        for start in range(0, len(bins), 1024):
            cut = slice(start, start + 1024)
            p = probs[bins[cut]][:, columns]
            y = labels[rows[cut]][:, columns]
            weight = spans[cut, None]
            selected = np.take_along_axis(p, y[..., None], -1)[..., 0]
            positive = selected > 0
            infinity += float(np.sum(weight * ~positive))
            nll += float(np.sum(weight * -np.log(np.where(positive, selected, 1))))
            brier += float(np.sum(weight * (np.sum(p * p, -1) + 1 - 2 * selected)))
            predicted = p.argmax(-1)
            good += float(np.sum(weight * (predicted == y)))
            np.add.at(confusion, (y.ravel(), predicted.ravel()), np.broadcast_to(weight, y.shape).ravel())
            confidence = p.max(-1)
            slots = np.minimum((confidence * 15).astype(int), 14)
            for slot in range(15):
                mask = slots == slot
                reliability[slot, 0] += np.sum(weight * mask)
                reliability[slot, 1] += np.sum(weight * confidence * mask)
                reliability[slot, 2] += np.sum(weight * (predicted == y) * mask)
        report.update(NLL=None if infinity else nll / total, NLL_is_infinite=bool(infinity),
            infinite_NLL_query_bp=infinity, probability_floor=False, accuracy=good / total,
            Brier=brier / total, ECE=float(np.abs(reliability[:, 2] - reliability[:, 1]).sum() / total),
            confusion_truth_rows_prediction_columns_query_bp=confusion.tolist(),
            calibration_bin_count=15, metric_status="complete full-support native final probability score")
        reports[name] = report
    result = {"parent_id": frame["parent_id"], "groups": reports, "truth_provenance": provenance,
              "prediction_seal": seal, "retained_original_grid_indexes": indexes.tolist(),
              "coverage_original_bp0": frame["coverage_original_bp0"],
              "score_selection": "external descriptive proper score; no change to stock inner checkpoint selection",
              "paper_reproduction": False, "final_TEST_evaluation": False}
    new_json(output / (frame["parent_id"] + "_external_full_support_metrics.json"), result)
    return result


def gpu_receipt(record, cfg):
    receipt = json.loads(checked(record).read_text())
    result = receipt.get("result", receipt)
    if (receipt.get("status") != "passed" or result.get("phase") != "qualify-gpu" or
            result.get("actual_source_joint_GPU_forward_backward") is not True or
            result.get("actual_stock_Rust_TorchScript_GPU_fixture") is not True or
            result.get("actual_cuda") is not True or result.get("config_sha256") != sha(CONFIG) or
            result.get("owned_code_sha256") != cfg["owned_code_sha256"]):
        raise RuntimeError("Same-policy/code actual full-source GPU and stock Rust FFI admission required")
    return record


def bootstrap(args, gpu, ownership):
    """Existing scheduler holds GPUs; this changes only interpreter/environment."""
    output = local(args.output)
    if output.exists() or not output.parent.is_dir():
        raise RuntimeError("Fresh project output with existing parent required")
    output.mkdir(); ownership["created"] = True
    guard = load_frozen(GUARD, GUARD_SHA, "_argmix_bootstrap_guard")
    lease = guard.validate_lease() if gpu else None
    if gpu:
        guard.apply_and_validate_title()
    elif os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU action requires hidden CUDA")
    from arggnn.resources import require_storage
    used = require_storage(headroom_bytes=5_000_000_000)
    env = os.environ.copy()
    for key in ("PYTHONHOME", "LIBTORCH_BYPASS_VERSION_CHECK", "LIBTORCH", "LIBTORCH_INCLUDE",
                "LIBTORCH_LIB", "LIBTORCH_STATIC", "LIBTORCH_CXX11_ABI", "TORCH_CUDA_VERSION"):
        env.pop(key, None)
    env.update(PYTHONPATH=str(ROOT / "src"), PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
               CONDA_PREFIX=str(PREFIX), WANDB_MODE="disabled", LIBTORCH_USE_PYTORCH="1",
               HF_HUB_OFFLINE="1", HDF5_DIR=str(PREFIX),
               LD_LIBRARY_PATH=str(PREFIX / "lib") + ":" + str(PREFIX / "lib/python3.12/site-packages/torch/lib"),
               ARGGNN_ARGMIX_BIOLOGICAL_CODE_SHA256=sha(MODULE))
    env["PATH"] = str(PREFIX / "bin") + os.pathsep + env.get("PATH", "")
    short = local(ROOT / ".runtime/tmp/ambi" / hashlib.sha256(str(output).encode()).hexdigest()[:12])
    if len(os.fsencode(short / "pymp-12345678/listener-12345678")) > 107:
        raise RuntimeError("Short contained IPC path exceeds socket budget")
    short.mkdir(parents=True, exist_ok=False)
    for key in ("TMPDIR", "TMP", "TEMP"):
        env[key] = str(short)
    for key, directory in {"XDG_CACHE_HOME": "cache", "XDG_CONFIG_HOME": "config",
            "XDG_DATA_HOME": "data", "HF_HOME": "cache/huggingface", "TORCH_HOME": "cache/torch",
            "MPLCONFIGDIR": "cache/matplotlib", "WANDB_DIR": "logs/wandb",
            "WANDB_CACHE_DIR": "cache/wandb", "WANDB_CONFIG_DIR": "config/wandb",
            "NUMBA_CACHE_DIR": "cache/numba", "TRITON_CACHE_DIR": "cache/triton",
            "CUDA_CACHE_PATH": "cache/cuda", "TORCHINDUCTOR_CACHE_DIR": "cache/torchinductor"}.items():
        target = output / directory; target.mkdir(parents=True, exist_ok=True); env[key] = str(target)
    if lease:
        os.set_inheritable(lease["lease_fd"], True)
        if guard.validate_lease() != lease:
            raise RuntimeError("Actual lease changed before native exec")
    new_json(output / "exec_admission.json", {"phase": args.phase, "lease": lease,
        "module_sha256": sha(MODULE), "script_sha256": sha(SCRIPT), "interpreter": str(PREFIX / "bin/python"),
        "TMPDIR": str(short), "storage_bytes": used, "headroom_bytes": 5_000_000_000})
    os.execve(PREFIX / "bin/python", [str(PREFIX / "bin/python"), "-B", str(SCRIPT), "--native", *sys.argv[1:]], env)


def main():
    spec = importlib.util.spec_from_file_location("_argmix_owned_biological_cli", SCRIPT)
    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    return cli.main()


if __name__ == "__main__":
    raise SystemExit(main())
