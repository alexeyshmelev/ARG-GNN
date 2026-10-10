#!/usr/bin/env python
"""Qualify a full pinned author ARGformer on handmade inputs only.

Source-only mode performs standard-library source/fixture checks. Native mode
requires the fixed isolated interpreter and the project runner's actual lease.
No biological data, optimizer step, author source edit or AST replacement occurs.
"""
from __future__ import annotations
import argparse
import ast
from contextlib import nullcontext, redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
import traceback

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE = ROOT / ".runtime/author_sources/argformer_f7545be7bca0"
PREFIX = ROOT / ".runtime/conda/envs/argformer_native_v1"
RUNNER = ROOT / "src/arggnn/native_argformer_exec.py"
SCRIPT = ROOT / "scripts/qualify_official_argformer_runtime.py"
BASE = ROOT / ".runtime/tmp/official_argformer_native_environment_candidate_v1/hf_base_config/config.json"
RECIPES = ROOT / "configs/official_argformer_native_environment_v1"
NVTOP = ROOT / ".runtime/env/bin/nvtop"
COMMIT = "f7545be7bca0200fb39aeb7f94a8e5f0b6bf5596"
TREE_SHA256 = "2295d86ace359ac1a7d0522c1001b6f2822b30cf9b16200517ea3473f234ca5b"
BASE_SHA256 = "7160e1553ad2ca51d8c1cb066be533db31826e12d173824c1bb0cb1a4f187d20"
LOCK_SHA256 = "e14f152b61e624fb926b86c9f8d6125d8e998aacf896f13c96f6e2a2804dcda3"
EXPECTED = {"torch": "2.4.0", "numpy": "1.26.4", "transformers": "4.44.1",
            "mosaicml": "0.24.1", "mosaicml-streaming": "0.8.0",
            "omegaconf": "2.3.0", "torchmetrics": "1.4.0.post0",
            "einops": "0.8.0", "triton": "3.0.0", "flash-attn": "2.6.3",
            "setproctitle": "1.3.8"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    body = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x") as handle:
        handle.write(body)
    return hashlib.sha256(body.encode()).hexdigest()


def load_guard():
    if RUNNER.is_symlink() or RUNNER.resolve() != RUNNER:
        raise RuntimeError("Assigned runner source required")
    spec = importlib.util.spec_from_file_location("_arggnn_native_argformer_guard", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_source():
    if SOURCE.resolve() != SOURCE or SOURCE.is_symlink():
        raise RuntimeError("Canonical pinned source directory required")
    origin = json.loads((SOURCE / "ARG_GNN_SOURCE_RECEIPT.json").read_text())
    if origin.get("commit") != COMMIT or origin.get("archive_sha256") != (
            "44c1b76f7806dfc3063d707eae1950601eb4caa3ebe2fa20277bb98e77f1890c"):
        raise RuntimeError("Source archive identity differs")
    files = []
    for path in sorted(SOURCE.rglob("*")):
        if path.is_symlink():
            raise RuntimeError("Pinned source contains a symlink")
        if path.is_file() and path.name != "ARG_GNN_SOURCE_RECEIPT.json":
            files.append({"path": path.relative_to(SOURCE).as_posix(), "sha256": sha(path)})
    digest = hashlib.sha256(canonical(files)).hexdigest()
    if len(files) != 60 or digest != TREE_SHA256:
        raise RuntimeError("Exact unchanged 60-file author source required")
    return {"commit": COMMIT, "source_directory": str(SOURCE),
            "source_tree_sha256": digest, "files": files}


def prepare_inputs(output):
    base = BASE.read_bytes()
    if hashlib.sha256(base).hexdigest() != BASE_SHA256:
        raise RuntimeError("Frozen BERT base config differs")
    config = json.loads(base)
    for name, expected in {"hidden_size": 768, "intermediate_size": 3072,
                           "num_attention_heads": 12, "num_hidden_layers": 12}.items():
        if config.get(name) != expected:
            raise RuntimeError("BERT base dimensions differ")
    base_dir = output / "base_config"
    base_dir.mkdir()
    (base_dir / "config.json").write_bytes(base)
    if sha(SOURCE / "pixi.lock") != LOCK_SHA256:
        raise RuntimeError("Author dependency lock differs")
    released = {}
    for mode in ("mlm", "contrastive"):
        body = (SOURCE / "yamls" / (mode + ".yaml")).read_bytes()
        copied = output / ("released_" + mode + ".yaml")
        copied.write_bytes(body)
        released[mode] = {"path": str(copied), "sha256": sha(copied)}
    recipes = {str(path.relative_to(RECIPES)): sha(path)
               for path in sorted(RECIPES.rglob("*")) if path.is_file()}
    if not recipes or json.loads((RECIPES / "lock-package-manifest.json").read_text()).get(
            "lock_sha256") != LOCK_SHA256:
        raise RuntimeError("Primary-owned pinned recipe manifest required")
    vocab = {"special": {"[PAD]": 0, "[MASK]": 1, "[CLS]": 2, "[SEP]": 3},
             "node_id_tokens": {"node_" + str(node): node + 4 for node in range(138)}}
    paths = [[leaf] + list(range(6, 138))[:length]
             for leaf, length in enumerate((130, 110, 90, 70, 40, 16))]
    labels = [0, 0, 1, 1, 2, 2]
    fixture = output / "handmade_fixture"
    for split in ("train", "val"):
        directory = fixture / split
        directory.mkdir(parents=True)
        with (directory / ("raw_" + split + "_sequences_and_vocab.pkl")).open("xb") as handle:
            pickle.dump({"vocab": vocab, "raw_sequences": paths}, handle, protocol=4)
        with (directory / "labels.pkl").open("xb") as handle:
            pickle.dump(labels, handle, protocol=4)
    description = {"kind": "handmade_nonbiological_fixture", "node_namespace":
                   "native_runtime_qualification_only", "vocab": vocab,
                   "raw_sequences": paths, "synthetic_pair_labels": labels,
                   "sequence_length": 128, "skip_extant_tokens": True,
                   "biological_labels_or_cohorts": False}
    write_json(output / "fixture.json", description)
    return {"base_config_directory": str(base_dir), "base_config_sha256": BASE_SHA256,
            "released_configs": released, "pinned_recipes_sha256": recipes,
            "fixture_directory": str(fixture), "fixture_sha256": {
                str(path.relative_to(output)): sha(path) for path in fixture.rglob("*")
                if path.is_file()}, "synthetic_labels": labels}


def module_origin(module, directory):
    path = Path(module.__file__).resolve()
    if not path.is_relative_to(directory):
        raise RuntimeError("Module resolved outside assigned directory: " + module.__name__)
    return {"module": module.__name__, "path": str(path), "sha256": sha(path)}


def tensor_info(tensor):
    cpu = tensor.detach().cpu().contiguous()
    data = cpu.reshape(-1).view(importlib.import_module("torch").uint8).numpy().tobytes()
    return {"shape": list(cpu.shape), "dtype": str(cpu.dtype),
            "sha256": hashlib.sha256(data).hexdigest()}


def state_manifest(model):
    rows = []
    for name, value in sorted(model.state_dict().items()):
        rows.append({"name": name, **tensor_info(value)})
    return {"tensors": rows, "sha256": hashlib.sha256(canonical(rows)).hexdigest()}




def actual_cuda_uuid(output, runtime):
    """Read the real current driver-context UUID without cudaDeviceProp ABI guesses.

    Torch2.4 does not expose uuid. CUDA12.4 cudart has no cudaDeviceGetUuid;
    UUID lookup is the stable driver API on the driver Torch already loaded.
    """
    import ctypes
    import uuid
    choices = [PREFIX / "lib/libcudart.so.12",
               PREFIX / "targets/x86_64-linux/lib/libcudart.so.12"]
    library_path = next((path for path in choices
                         if path.is_file() and path.resolve().is_relative_to(PREFIX)), None)
    if library_path is None:
        raise RuntimeError("Actual isolated CUDA12 runtime library required")
    identity = {"source": "actual_loaded_cuda_driver_current_context_uuid",
                "runtime_library": str(library_path),
                "resolved_runtime_library": str(library_path.resolve()),
                "runtime_library_sha256": sha(library_path.resolve()), "calls": {}}
    runtime["cuda_uuid_identity"] = identity
    library = ctypes.CDLL(str(library_path))
    current = ctypes.c_int(-1)
    runtime_version = ctypes.c_int(-1)
    for name, value in (("cudaGetDevice", current),
                        ("cudaRuntimeGetVersion", runtime_version)):
        function = getattr(library, name)
        function.argtypes = [ctypes.POINTER(ctypes.c_int)]
        function.restype = ctypes.c_int
        code = int(function(ctypes.byref(value)))
        identity["calls"][name] = {"return_code": code, "value": int(value.value)}
        if code:
            raise RuntimeError("Actual isolated CUDA identity call failed: " + name)
    if current.value != 0 or runtime_version.value != 12040:
        raise RuntimeError("Actual isolated CUDA12.4 logical device0 required")
    maps = Path("/proc/self/maps").read_text()
    (output / "cuda_library_maps.txt").write_text(maps)
    driver_paths = set()
    for line in maps.splitlines():
        columns = line.split(maxsplit=5)
        if len(columns) == 6 and Path(columns[5]).name.startswith("libcuda.so."):
            path = Path(columns[5])
            if not path.is_absolute() or not path.is_file():
                raise RuntimeError("Actual loaded CUDA driver file unavailable")
            driver_paths.add(path.resolve())
    if len(driver_paths) != 1:
        raise RuntimeError("One actual loaded CUDA driver from own process maps required")
    driver_path = driver_paths.pop()
    identity.update(driver_library=str(driver_path), driver_library_sha256=sha(driver_path),
                    driver_origin="own_actual_process_maps_after_torch_context",
                    process_maps_sha256=sha(output / "cuda_library_maps.txt"))
    driver = ctypes.CDLL(str(driver_path))
    device = ctypes.c_int(-1)
    driver_version = ctypes.c_int(-1)
    for name, value in (("cuCtxGetDevice", device), ("cuDriverGetVersion", driver_version)):
        function = getattr(driver, name)
        function.argtypes = [ctypes.POINTER(ctypes.c_int)]
        function.restype = ctypes.c_int
        code = int(function(ctypes.byref(value)))
        identity["calls"][name] = {"return_code": code, "value": int(value.value)}
        if code:
            raise RuntimeError("Actual loaded driver identity call failed: " + name)
    class CudaUUID(ctypes.Structure):
        _fields_ = [("bytes", ctypes.c_ubyte * 16)]
    actual = CudaUUID()
    function_name = "cuDeviceGetUuid_v2" if hasattr(driver, "cuDeviceGetUuid_v2") else "cuDeviceGetUuid"
    function = getattr(driver, function_name)
    function.argtypes = [ctypes.POINTER(CudaUUID), ctypes.c_int]
    function.restype = ctypes.c_int
    code = int(function(ctypes.byref(actual), device.value))
    identity["calls"][function_name] = {"return_code": code, "context_device": int(device.value)}
    if code or driver_version.value < 12040:
        raise RuntimeError("Actual CUDA driver context UUID lookup failed")
    identity.update(logical_runtime_device=int(current.value),
                    runtime_version=int(runtime_version.value),
                    driver_version=int(driver_version.value),
                    driver_context_device=int(device.value),
                    uuid_function=function_name, uuid_bytes_hex=bytes(actual.bytes).hex(),
                    canonical_uuid=str(uuid.UUID(bytes=bytes(actual.bytes))))
    return identity["canonical_uuid"]


def nvtop_snapshot(output, lease, runtime):
    if not NVTOP.is_file() or not NVTOP.resolve().is_relative_to(ROOT / ".runtime/env"):
        raise RuntimeError("Actual project NVTOP executable required")
    runtime["nvtop"] = {"executed": True, "executable": str(NVTOP)}
    try:
        result = subprocess.run([str(NVTOP), "--snapshot"], capture_output=True, timeout=30)
    except subprocess.TimeoutExpired as error:
        (output / "nvtop_snapshot.json").write_bytes(error.stdout or b"")
        (output / "nvtop_snapshot.stderr").write_bytes(error.stderr or b"")
        (output / "nvtop.exit_status").write_text("timeout\n")
        runtime["nvtop"].update(exit_code=None, timed_out=True,
                               stdout_sha256=sha(output / "nvtop_snapshot.json"),
                               stderr_sha256=sha(output / "nvtop_snapshot.stderr"))
        raise
    (output / "nvtop_snapshot.json").write_bytes(result.stdout)
    (output / "nvtop_snapshot.stderr").write_bytes(result.stderr)
    (output / "nvtop.exit_status").write_text(str(result.returncode) + "\n")
    runtime["nvtop"] = {"executed": True, "exit_code": result.returncode,
                       "executable": str(NVTOP), "stdout_sha256":
                       sha(output / "nvtop_snapshot.json"), "stderr_sha256":
                       sha(output / "nvtop_snapshot.stderr")}
    if result.returncode:
        raise RuntimeError("Actual NVTOP snapshot failed")
    snapshot = json.loads(result.stdout)
    physical = lease["physical_gpu"]
    if not isinstance(snapshot, list) or physical >= len(snapshot):
        raise RuntimeError("NVTOP physical device ordering unavailable")
    matches = [item for item in snapshot[physical].get("processes", [])
               if str(item.get("pid")) == str(os.getpid())]
    if len(matches) != 1 or matches[0].get("kind") != "compute":
        raise RuntimeError("Own compute PID missing from reserved physical NVTOP device")
    if matches[0].get("cmdline", "").strip() != "AI Auto Research Pipeline Test":
        raise RuntimeError("Actual NVTOP process title differs")
    runtime["nvtop"].update(physical_gpu=physical, own_process=matches[0],
                           actual_pid_title_and_index_verified=True)


def native_qualification(args, output, guard, inputs, receipt):
    runtime = receipt["runtime"]
    runtime["stage"] = "native_interpreter"
    if sys.version_info[:3] != (3, 11, 9):
        raise RuntimeError("Pinned Python3.11.9 required")
    if Path(sys.prefix).resolve() != PREFIX or not Path(sys.executable).resolve().is_relative_to(
            PREFIX / "bin"):
        raise RuntimeError("Fixed isolated native Python prefix required")
    admission = json.loads((output / "exec_admission.json").read_text())
    if (os.environ.get("ARGGNN_NATIVE_OUTPUT_DIR") != str(output) or
            os.environ.get("ARGGNN_NATIVE_SCRIPT_SHA256") != sha(SCRIPT) or
            admission.get("script_sha256") != sha(SCRIPT) or
            os.environ.get("ARGGNN_NATIVE_RUNNER_SHA256") != sha(RUNNER)):
        raise RuntimeError("Scheduler exec provenance differs")
    runtime["stage"] = "title_before_cuda"
    runtime["process_title"] = guard.apply_and_validate_title()
    lease = guard.validate_lease()
    if admission.get("lease") != lease:
        raise RuntimeError("Actual inherited lease differs after exec")
    if not os.get_inheritable(lease["lease_fd"]):
        raise RuntimeError("Native inherited lease FD is not inheritable")
    runtime["lease"] = lease
    runtime["stage"] = "dependency_inventory"
    runtime["distributions"] = {item.metadata["Name"]: item.version
                               for item in importlib.metadata.distributions()}
    for name, expected in EXPECTED.items():
        actual = importlib.metadata.version(name)
        if actual.split("+")[0] != expected:
            raise RuntimeError("Pinned native distribution differs: " + name + "=" + actual)
    repairs_path = RECIPES / "packaging_repairs_v2.json"
    repairs = json.loads(repairs_path.read_text())
    if not repairs.get("original_author_lock_preserved") or not repairs.get("model_source_unchanged"):
        raise RuntimeError("Explicit unchanged-model packaging variant required")
    runtime["environment_variant"] = {
        "label": "author_source_model_unchanged_packaging_repairs_v2_official_FA2_wheel",
        "exact_author_dependency_lock_reproduction": False,
        "author_lock_sha256": LOCK_SHA256, "repairs_manifest_sha256": sha(repairs_path),
        "actual_repaired_distributions": {}, "FA_install_route": "official_prebuilt_wheel",
        "FA_source_build_claimed": False,
        "FA_wheel_recipe_sha256": sha(RECIPES / "official_flash_wheel_v2.txt")}
    for name, repair in repairs["repairs"].items():
        actual = importlib.metadata.version(name)
        runtime["environment_variant"]["actual_repaired_distributions"][name] = {
            "version": actual, "declared_version": repair["version"],
            "wheel_sha256": repair["sha256"]}
        if actual != repair["version"]:
            raise RuntimeError("Actual repaired distribution differs: " + name + "=" + actual)
    runtime["conda_records_scope"] = "nominal package-manager records; actual pip overlays are recorded separately"
    runtime["conda_records"] = [{"name": data.get("name"), "version": data.get("version"),
                               "build": data.get("build"), "url": data.get("url"),
                               "sha256": data.get("sha256"), "md5": data.get("md5"),
                               "metadata_sha256": sha(path)}
                              for path in sorted((PREFIX / "conda-meta").glob("*.json"))
                              for data in [json.loads(path.read_text())]]
    write_json(output / "runtime_environment.json", runtime)
    runtime["stage"] = "torch_import"
    import torch
    runtime["torch"] = {"version": torch.__version__, "cuda": torch.version.cuda,
                        "build": torch.__config__.show(),
                        "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
                        "default_dtype": str(torch.get_default_dtype()),
                        "interpreter": sys.executable, "prefix": sys.prefix,
                        "python": sys.version}
    if torch.cuda.is_initialized() or torch.version.cuda != "12.4":
        raise RuntimeError("Uninitialized CUDA12.4 native Torch required")
    runtime["stage"] = "actual_cuda_context"
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one reserved visible CUDA device required")
    context_anchor = torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    receipt["actual_cuda"] = True
    properties = torch.cuda.get_device_properties(0)
    property_uuid = getattr(properties, "uuid", None)
    actual_uuid = actual_cuda_uuid(output, runtime)
    if actual_uuid != lease["canonical_uuid"]:
        raise RuntimeError("Actual CUDA runtime UUID differs from scheduler reservation")
    if property_uuid is not None and guard.canonical_uuid(str(property_uuid)) != actual_uuid:
        raise RuntimeError("Torch property UUID and actual CUDA runtime UUID differ")
    runtime["device"] = {"logical_index": 0, "runtime_uuid": actual_uuid,
                         "torch_property_uuid": str(property_uuid) if property_uuid is not None else None,
                         "uuid_source": "actual_loaded_cuda_driver_current_context_uuid",
                         "name": properties.name, "compute_capability":
                         [properties.major, properties.minor],
                         "total_memory_bytes": properties.total_memory}
    if properties.major < 8:
        raise RuntimeError("FlashAttention2 Ampere/Ada/Hopper device required")
    guard.apply_and_validate_title()
    nvtop_snapshot(output, lease, runtime)
    driver = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,driver_version",
                             "--format=csv,noheader"], capture_output=True, text=True,
                            timeout=30)
    (output / "nvidia_smi_identity.stdout").write_text(driver.stdout)
    (output / "nvidia_smi_identity.stderr").write_text(driver.stderr)
    (output / "nvidia_smi.exit_status").write_text(str(driver.returncode) + "\n")
    if driver.returncode:
        raise RuntimeError("Actual physical GPU identity query failed")
    rows = [line.split(",") for line in driver.stdout.splitlines() if line.strip()]
    matching = [row for row in rows if int(row[0].strip()) == lease["physical_gpu"]]
    if len(matching) != 1 or matching[0][1].strip() != lease["visible_gpu_uuid"]:
        raise RuntimeError("Physical GPU UUID mapping differs")
    runtime["driver_version"] = matching[0][2].strip()
    runtime["stage"] = "actual_author_import"
    if any(name == "src" or name.startswith("src.") for name in sys.modules):
        raise RuntimeError("Fresh author src namespace required")
    sys.path.insert(0, str(SOURCE))
    from omegaconf import OmegaConf
    flex = importlib.import_module("src.flex_bert")
    data_module = importlib.import_module("src.arg_data")
    attention = importlib.import_module("src.bert_layers.attention")
    rotary = importlib.import_module("src.bert_layers.rotary")
    author_models = importlib.import_module("src.bert_layers.model")
    losses = importlib.import_module("flash_attn.losses.cross_entropy")
    extension = importlib.import_module("flash_attn_2_cuda")
    if not attention.IMPL_USE_FLASH2 or attention.IMPL_USE_FLASH3:
        raise RuntimeError("Actual FA2-only author backend required")
    runtime["author_modules"] = [module_origin(importlib.import_module("src"), SOURCE)] + [
        module_origin(module, SOURCE) for module in
        (flex, data_module, attention, rotary, author_models)]
    runtime["native_dependency_modules"] = [module_origin(module, PREFIX) for module in
        (torch, losses, extension, importlib.import_module("transformers"),
         importlib.import_module("composer"), importlib.import_module("triton"))]
    config = OmegaConf.load(SOURCE / "yamls" / (args.mode + ".yaml"))
    config.data_local = inputs["fixture_directory"]
    config.model.pretrained_model_name = inputs["base_config_directory"]
    config.model.model_config.dataset.local = inputs["fixture_directory"]
    if args.mode == "contrastive":
        config.model.pretrained_checkpoint = None
    receipt["actual_full_author_import"] = True
    runtime["stage"] = "actual_factory"
    seed = 17 if args.mode == "mlm" else 36
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    captured = io.StringIO()
    with redirect_stdout(captured):
        if args.mode == "mlm":
            wrapper = flex.create_flex_bert_mlm(
                pretrained_model_name=config.model.pretrained_model_name,
                model_config=config.model.model_config, tokenizer_name="argformer",
                recompute_metric_loss=False)
            backbone = wrapper.model
            if type(backbone) is not author_models.FlexBertForMaskedLM or (
                    type(backbone.loss_fn) is not losses.CrossEntropyLoss):
                raise RuntimeError("Actual author MLM and FA fused loss required")
        else:
            contrastive = importlib.import_module("sequence_contrastive")
            runtime["author_modules"].append(module_origin(contrastive, SOURCE))
            wrapper = contrastive.build_contrastive_model(config.model)
            backbone = wrapper.base.backbone
            if type(backbone) is not author_models.FlexBertForSequenceClassification:
                raise RuntimeError("Actual author contrastive backbone required")
            if wrapper.base.class_weights is not None:
                raise RuntimeError("Released model-only config uniform fallback changed")
            receipt["contrastive_uniform_weights"] = {
                "requested_in_yaml": True, "actual_class_weights": None,
                "author_missing_model_train_loader_fallback_preserved": True,
                "temperature": wrapper.base.temperature,
                "constructor_stdout": captured.getvalue()}
            if wrapper.base.temperature != 0.3:
                raise RuntimeError("Released contrastive temperature differs")
    (output / "model_construction.stdout").write_text(captured.getvalue())
    author_roots = {"src", "bert_layers", "bert_padding", "hf_bert", "mosaic_bert"}
    runtime["author_modules"] = [module_origin(module, SOURCE)
        for name, module in sorted(sys.modules.items())
        if (name.split(".")[0] in author_roots or name == "sequence_contrastive")
        and getattr(module, "__file__", None)]
    layers = backbone.bert.encoder.layers
    resolved = backbone.config.to_dict()
    if (len(layers) != 12 or resolved["hidden_size"] != 768 or
            resolved["intermediate_size"] != 3072 or resolved["num_attention_heads"] != 12 or
            not resolved["use_fa2"] or resolved["max_seq_len"] != 128):
        raise RuntimeError("Full released encoder dimensions/backend required")
    windows = []
    for index, layer in enumerate(layers):
        if (type(layer.attn) is not attention.FlexBertUnpadRopeAttention or
                type(layer.attn.rotary_emb) is not rotary.UnpaddedRotaryEmbedding or
                not layer.attn.use_fa2 or layer.attn.use_fa3):
            raise RuntimeError("Actual native RoPE/FA2 encoder layer required")
        expected_window = (-1, -1) if args.mode == "contrastive" or index % 3 == 0 else (64, 64)
        if tuple(layer.attn.sliding_window) != expected_window:
            raise RuntimeError("Released per-layer attention window differs")
        windows.append({"zero_based_layer": index, "class": type(layer).__name__,
                        "attention_class": type(layer.attn).__name__,
                        "rotary_class": type(layer.attn.rotary_emb).__name__,
                        "window": list(layer.attn.sliding_window)})
    if resolved["unpad_embeddings"] != (args.mode == "mlm"):
        raise RuntimeError("Separately released unpadding configuration differs")
    receipt["resolved_model_config_sha256"] = write_json(
        output / ("resolved_" + args.mode + "_model_config.json"), resolved)
    receipt["resolved_invocation_config_sha256"] = write_json(
        output / ("resolved_" + args.mode + "_invocation.json"),
        OmegaConf.to_container(config, resolve=True))
    receipt["model"] = {"seed": seed, "parameter_count": sum(p.numel() for p in wrapper.parameters()),
                        "actual_layers": windows, "loss_class": type(backbone.loss_fn).__name__
                        if args.mode == "mlm" else "author_contrastive_similarity_loss",
                        "class": type(backbone).__name__, "random_initialization": True,
                        "trained_comparator": False}
    initial_state = state_manifest(wrapper)
    write_json(output / "initial_state_manifest.json", initial_state)
    receipt["initial_state_sha256"] = initial_state["sha256"]
    dataset = data_module.ARGDataset(local=inputs["fixture_directory"], split="val",
                                    max_seq_len=128, skip_extant_tokens=True)
    items = [dataset[index] for index in range(len(dataset))]
    batch = {key: torch.tensor([item[key] for item in items], dtype=torch.long)
             for key in ("input_ids", "attention_mask", "token_type_ids")}
    if args.mode == "mlm":
        labels = torch.full_like(batch["input_ids"], -100)
        for row in range(len(items)):
            positions = [1, int(batch["attention_mask"][row].sum()) // 2]
            for position in positions:
                if batch["input_ids"][row, position] < 4:
                    raise RuntimeError("Actual handmade ancestor token required for masking")
                labels[row, position] = batch["input_ids"][row, position]
                batch["input_ids"][row, position] = 1
        batch["labels"] = labels
        receipt["masking"] = {"kind": "explicit_handmade_token_reconstruction",
                              "masked_count": int((labels != -100).sum()),
                              "not_author_training_collator_reproduction": True}
    else:
        batch["labels"] = torch.tensor(inputs["synthetic_labels"], dtype=torch.long)
    torch.save(batch, output / "handmade_native_batch.pt")
    receipt["input_tensors"] = {key: tensor_info(value) for key, value in batch.items()}
    wrapper = wrapper.cuda().eval()
    batch = {key: value.cuda() for key, value in batch.items()}
    torch.cuda.reset_peak_memory_stats()
    runtime["stage"] = "actual_native_forward"
    started = time.perf_counter()
    precision = torch.autocast("cuda", dtype=torch.bfloat16) if args.mode == "mlm" else nullcontext()
    with (nullcontext() if args.backward else torch.no_grad()), precision:
        outputs = wrapper(batch)
        loss = outputs["loss"]
    if loss.ndim or not torch.isfinite(loss).all():
        raise RuntimeError("Finite actual scalar native loss required")
    tensors = {"loss": loss.detach().cpu()}
    key = "logits" if args.mode == "mlm" else "sim"
    if not torch.isfinite(outputs[key]).all():
        raise RuntimeError("Actual native output contains nonfinite values")
    tensors[key] = outputs[key].detach().cpu()
    if args.mode == "contrastive":
        with torch.no_grad():
            tensors["embeddings"] = wrapper.base.encode(
                batch["input_ids"], batch["attention_mask"],
                batch.get("token_type_ids")).detach().cpu()
        if tensors["embeddings"].shape != (6, 768) or not torch.isfinite(
                tensors["embeddings"]).all():
            raise RuntimeError("Finite complete native embeddings required")
    torch.save(tensors, output / "native_outputs.pt")
    receipt["output_tensors"] = {name: tensor_info(value) for name, value in tensors.items()}
    receipt["loss_value"] = float(tensors["loss"])
    receipt["actual_full_native_forward"] = True
    if args.backward:
        runtime["stage"] = "actual_native_backward"
        loss.backward()
        grads = [(name, parameter.grad) for name, parameter in wrapper.named_parameters()
                 if parameter.grad is not None]
        if not grads or any(not torch.isfinite(gradient).all() for _, gradient in grads):
            raise RuntimeError("Finite actual native gradients required")
        if any(not any(parameter.grad is not None for parameter in layer.parameters())
               for layer in layers):
            raise RuntimeError("All twelve layers must participate in native backward")
        receipt["backward"] = {"executed": True, "gradient_parameter_count": len(grads),
                               "all_twelve_layers_have_gradients": True,
                               "optimizer_step": False}
        if args.mode == "contrastive":
            if any(p.grad is not None for p in backbone.classifier.parameters()) or not any(
                    p.grad is not None for p in backbone.head.parameters()):
                raise RuntimeError("Author pooler/classifier loss path differs")
            receipt["backward"]["classifier_unused_by_author_loss"] = True
    else:
        receipt["backward"] = {"executed": False, "optimizer_step": False}
    torch.cuda.synchronize()
    receipt["forward_elapsed_seconds_including_optional_backward"] = time.perf_counter() - started
    receipt["peak_memory_bytes"] = torch.cuda.max_memory_allocated()
    final_state = state_manifest(wrapper)
    write_json(output / "final_state_manifest.json", final_state)
    if initial_state["sha256"] != final_state["sha256"]:
        raise RuntimeError("Parameters changed without an authorized optimizer step")
    receipt["final_state_sha256"] = final_state["sha256"]
    if guard.validate_lease() != lease:
        raise RuntimeError("Scheduler lease changed during native qualification")
    validate_source()
    runtime["stage"] = "complete"
    receipt.update(actual_full_author_import=True, actual_full_native_forward=True,
                   actual_cuda=True, status="passed_handmade_native_runtime_only")
    del context_anchor


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("mlm", "contrastive"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args(argv)
    guard = load_guard()
    output = guard.project_path(args.output_dir)
    if args.source_only:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or args.backward:
            parser.error("Source-only mode requires empty CUDA_VISIBLE_DEVICES and no backward")
        if output.exists() or not output.parent.is_dir():
            parser.error("Fresh source-only output directory with existing parent required")
        output.mkdir()
    elif (not output.is_dir() or not (output / "exec_admission.json").is_file()):
        parser.error("Native mode requires project exec runner output admission")
    receipt = {"schema_version": 1, "kind": "official_argformer_runtime_qualification_v1",
               "started_utc": datetime.now(timezone.utc).isoformat(),
               "mode": args.mode, "source_only": args.source_only,
               "status": "started", "actual_full_author_import": False,
               "actual_full_native_forward": False, "actual_cuda": False,
               "actual_nvtop": False, "published_task_reproduction": False,
               "biological_labels_or_cohorts": False, "planned_held_out_reads": False,
               "optimizer_step": False, "runtime": {"stage": "source_validation"}}
    code = 1
    try:
        if SCRIPT.resolve() != SCRIPT or Path(__file__).resolve() != SCRIPT:
            raise RuntimeError("Assigned fixed qualification script required")
        receipt["source"] = validate_source()
        receipt["inputs"] = prepare_inputs(output)
        receipt["code_sha256"] = {"runner": sha(RUNNER), "qualifier": sha(SCRIPT)}
        if args.source_only:
            parsed = {}
            for path in (RUNNER, SCRIPT, SOURCE / "src/flex_bert.py",
                         SOURCE / "src/bert_layers/attention.py",
                         SOURCE / "src/bert_layers/model.py", SOURCE / "sequence_contrastive.py"):
                tree = ast.parse(path.read_text())
                parsed[str(path)] = {"sha256": sha(path), "definitions": [
                    node.name for node in ast.walk(tree)
                    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]}
            receipt["source_ast"] = parsed
            receipt["runtime"]["stage"] = "source_only_complete"
            receipt["status"] = "passed_source_only_no_native_execution"
        else:
            native_qualification(args, output, guard, receipt["inputs"], receipt)
            receipt["actual_nvtop"] = receipt["runtime"].get("nvtop", {}).get(
                "actual_pid_title_and_index_verified", False)
        code = 0
    except BaseException as error:
        receipt["status"] = "failed"
        receipt["failure"] = {"type": type(error).__name__, "message": str(error),
                              "traceback": traceback.format_exc()}
        traceback.print_exc()
    finally:
        receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
        receipt["exit_code"] = code
        receipt["actual_nvtop"] = receipt["runtime"].get("nvtop", {}).get(
            "actual_pid_title_and_index_verified", False)
        receipt["artifacts_sha256"] = {str(path.relative_to(output)): sha(path)
                                      for path in sorted(output.rglob("*"))
                                      if path.is_file() and "cache" not in
                                      path.relative_to(output).parts and "tmp" not in
                                      path.relative_to(output).parts and path.name not in
                                      {"receipt.json", "exit_status"}}
        write_json(output / "receipt.json", receipt)
        (output / "exit_status").write_text(str(code) + "\n")
        print(json.dumps({"status": receipt["status"], "exit_code": code,
                          "mode": args.mode, "source_only": args.source_only,
                          "actual_full_native_forward": receipt["actual_full_native_forward"],
                          "output_dir": str(output)}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
