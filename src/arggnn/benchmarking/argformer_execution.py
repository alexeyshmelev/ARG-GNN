"""Full native ARGformer contrastive phase for the current guarded worker.

Import/spec construction use only source and compact metadata. The worker owns
GPU0 lease/title, admission, process supervision and actual waits. This module
owns no scheduler, accounting, launcher, retry or historical-controller route.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import tempfile

from .contracts import ROOT, ContractError, contained_path, file_ref, read_ref

GPU0_UUID = "GPU-ff06ca1f-403a-f2d0-b584-71e177452667"
AUTHOR = ".runtime/author_sources/argformer_f7545be7bca0"
NATIVE_PYTHON = ".runtime/conda/envs/argformer_native_v1/bin/python"
DATA = "experiments/results/American_C3_val000_full30mb_shared_v1/native_paths"
SCHEMA = "benchmark_argformer_full_contrastive_v1"
SETTINGS = dict(epochs=1, updates=2135, batch=4096, microbatch=4096,
                precision="fp32", seed=36, temperature=0.3,
                uniform_class_weight_fallback=True, max_seq_len=128,
                save_interval="20ba", save_num_checkpoints_to_keep=1)
PINS = {'author_manifest': {'path': 'research/reviews/American_C3_native_consumer_source_v1/author_source_manifest.json', 'sha256': 'bed570db438f2c4981ebf606460dc4ff72a97b148ba371e3873ab0141376736a'}, 'preparation': {'path': 'experiments/results/American_C3_val000_full30mb_shared_v1/native_paths/completed.json', 'sha256': '40a3b588d0aa2b474c3f4aab492e34feb0a753a0f8cb73c20d9c0059d8a8c16b'}, 'counts': {'path': 'experiments/results/American_C3_native_val000_full30mb_counts980_namespace_v1/completed.json', 'sha256': '502c111b443c46eea789dc03ee3ff8b6454546ce6e86e3f9fcddba25e82489a7'}, 'acceptance': {'path': 'research/reviews/research_execution_ROOT_20261017_v34/native_saved_MLM_artifact_acceptance.root_active_v1.json', 'sha256': 'a74bcce6bab94e01d367a7e32795e45942d9639bfd56bf637fb3ac634d0dec8e'}, 'checkpoint': {'path': 'experiments/results/native_VAL000_verified22000_continuation_v12_root01_oct16/checkpoints/ep5-ba227990-rank0.pt', 'sha256': '762d005aa403a787520f389f285a24b59a249c06f8f1145bc00db2ad060c499a'}, 'base_config': {'path': 'research/reviews/C3_bert_base_configuration_root_v1/config.json', 'sha256': '7160e1553ad2ca51d8c1cb066be533db31826e12d173824c1bb0cb1a4f187d20'}, 'bridge': {'path': 'research/reviews/native_VAL000_saved_MLM_acceptance_bridge_source_20261017_v1/saved_mlm_bridge_v1.py', 'sha256': '5810041797592fa414d57747607f646a3eee1050d0373df5412b7dd4b1b63d97'}, 'integration': {'path': 'research/reviews/native_VAL000_contrastive_saved_MLM_integration_source_20261017_v2/saved_mlm_integration_v1.py', 'sha256': '651e18d11d99083a60face804a3d50b9f1d48bcc83d3b6e0c2dbf827ae12b324'}, 'ipc': {'path': 'research/reviews/American_C3_native_MLM_amendment_v12/abstract_unix_bootstrap_v1.py', 'sha256': '4550ead44f36eeb0fda8a817a2002082c3d067cf1a84fd817301532577d8b014'}, 'runtime_binding': {'path': 'research/reviews/native_ARGformer_VAL000_GPU0_shared_execution_20261016_v1/mlm_v1/runtime_mlm.root_active_v1.json', 'sha256': '58aa23a7fcadb1b1cddc169cf7f0e1c63a17877d85bdef6405c34a48a5064e61'}}


def _need(ok, message):
    if not ok:
        raise ContractError(message)


def _verify_sources():
    manifest = read_ref(PINS["author_manifest"])
    for ref in [r for k, r in PINS.items() if k != "checkpoint"] + manifest["refs"]:
        _need(file_ref(ref["path"]) == ref, "ARGformer source/metadata changed: " + ref["path"])


def requirements():
    """Concrete native runtime, source closure and full-phase inputs; no payload I/O."""
    _verify_sources()
    prep, counts = read_ref(PINS["preparation"]), read_ref(PINS["counts"])
    acceptance = read_ref(PINS["acceptance"])
    _need(acceptance["contrastive_only_artifact_use_accepted"] is True
          and acceptance["saved_MLM_checkpoint_ref"] == PINS["checkpoint"]
          and acceptance["historical_MLM_outer_exit_code"] == 1,
          "Exact accepted saved MLM initializer; preserve failed historical supervisor")
    _need(prep["parent_id"] == counts["parent_id"] == "american_c3_val_000"
          and prep["query_ancestry_truth_opened"] is False
          and counts["query_ancestry_truth_opened"] is False
          and counts["contrastive_optimizer_updates"] == 2135
          and counts["datasets"]["contrastive/train"]["rows"] == 8748479
          and counts["datasets"]["contrastive/val"]["rows"] == 972757,
          "Exact full observed native contrastive corpus")
    inputs = [prep["vocab_ref"]]
    for split in ("train", "val"):
        item = counts["datasets"]["contrastive/" + split]
        _need(item["dataset_ref"] == prep["native_dataset_refs"]["contrastive/" + split],
              "Preparation/count corpus binding")
        inputs.extend((item["dataset_ref"], counts["reference_label_refs"][split]))
    sources = list(PINS.values()) + read_ref(PINS["author_manifest"])["refs"]
    # Checkpoint bytes are checked once by the original inline initializer.
    sources = [r for r in sources if r != PINS["checkpoint"]]
    return dict(native_python=NATIVE_PYTHON, physical_gpu=0, gpu_uuid=GPU0_UUID,
                author_callable="sequence_contrastive.train",
                author_source_ref=next(r for r in sources if r["path"] == AUTHOR + "/sequence_contrastive.py"),
                source_refs=list({r["path"]: r for r in sources}.values()),
                input_refs=inputs, initializer_checkpoint_ref=copy.deepcopy(PINS["checkpoint"]),
                runtime_inventory_ref=read_ref(PINS["runtime_binding"])["native_inventory_ref"],
                expected_updates=2135, expected_epochs=1, settings=copy.deepcopy(SETTINGS),
                scientific_payloads_opened=False, exact_paper_reproduction=False,
                paper_conflicts=["released temperature 0.30 versus paper default 0.05",
                                 "released uniform class-weight fallback versus paper inverse-frequency loss",
                                 "saved MLM uses released 80/10/10 masking versus literal paper MASK replacement"],
                phase_scope="contrastive_only_no_embedding_retrieval_or_scoring")


def _output(value):
    path = contained_path(value)
    _need(path.resolve() == path and path.is_relative_to(ROOT / "experiments/results")
          and path != ROOT / "experiments/results", "Fresh canonical result namespace required")
    return path


def build_spec(output_directory):
    """Return the complete phase specification for a separately admitted worker."""
    out = _output(output_directory)
    _need(not out.exists(), "New output namespace required; no continuation or replay")
    return _spec(out)


def _spec(out):
    return dict(schema=SCHEMA, parent_id="american_c3_val_000",
                output_directory=str(out.relative_to(ROOT)), requirements=requirements(),
                self_source_ref=file_ref(Path(__file__).resolve()))


def _validate_spec(spec):
    out = _output(spec["output_directory"])
    _need(spec == _spec(out), "Only the frozen complete native contrastive phase is supported")
    return out


def _load_source(role):
    ref = PINS[role]
    _need(file_ref(ref["path"]) == ref, "Pinned scientific helper changed")
    name = "_benchmark_argformer_" + role
    module_spec = importlib.util.spec_from_file_location(name, ROOT / ref["path"])
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    module_spec.loader.exec_module(module)
    return module


def _payload_ref(path, resource_check):
    """Authenticate a real consumed input/output without decoding it."""
    path = contained_path(path)
    digest = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        _need(stat.S_ISREG(before.st_mode), "Regular native payload required")
        while True:
            resource_check("ARGformer_payload_authentication")
            block = stream.read(16 * 2**20)
            if not block:
                break
            digest.update(block)
        after = os.fstat(stream.fileno())
    stamp = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    _need(stamp(before) == stamp(after) == stamp(path.lstat()), "Native payload changed during authentication")
    return dict(path=str(path.relative_to(ROOT)), sha256=digest.hexdigest())


def _checkpoint_counters(state):
    """Verify actual serialized Composer and Adam counters, never filename counters."""
    _need(isinstance(state, dict) and isinstance(state.get("model"), dict) and state["model"],
          "Actual nonempty saved model state required")
    outer = state.get("timestamp")
    _need(type(outer) is dict and set(outer) == {"Timestamp"}, "Installed Composer Timestamp schema required")
    timestamp = outer["Timestamp"]
    _need(type(timestamp) is dict and type(timestamp.get("epoch")) is int
          and type(timestamp.get("batch")) is int
          and timestamp["epoch"] == 1 and timestamp["batch"] == 2135,
          "Actual full epoch1/update2135 timestamp required")
    optimizers = state.get("optimizers")
    _need(type(optimizers) is dict and set(optimizers) == {"DecoupledAdamW"},
          "Actual author DecoupledAdamW state required")
    states = optimizers["DecoupledAdamW"].get("state")
    _need(isinstance(states, dict) and states, "Actual nonempty optimizer parameter states required")
    for item in states.values():
        _need(isinstance(item, dict) and {"step", "exp_avg", "exp_avg_sq"}.issubset(item),
              "Actual Adam moments and step required for every parameter")
        step = item["step"].item() if hasattr(item["step"], "item") else item["step"]
        _need(type(step) in (int, float) and step == 2135, "Every actual Adam step must equal2135")
    return dict(completed_epochs=timestamp["epoch"], actual_optimizer_updates=timestamp["batch"],
                optimizer_parameter_state_count=len(states), optimizer_unique_step_values=[2135],
                timestamp_schema="state.timestamp.Timestamp")


def _final_checkpoint(directory, torch, resource_check):
    regular, alias = [], None
    for path in directory.iterdir():
        info = path.lstat()
        if path.name == "latest-rank0.pt" and stat.S_ISLNK(info.st_mode):
            alias = os.readlink(path)
            _need(Path(alias).name == alias and alias not in ("", ".", ".."), "Local checkpoint alias required")
        else:
            _need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
                  and path.suffix == ".pt" and 0 < info.st_size <= 3 * 2**30,
                  "Bounded newly generated checkpoint files required")
            regular.append(path)
    _need(0 < len(regular) <= 2, "Keep1 current/previous checkpoint file bound")
    candidates = []
    for path in regular:
        resource_check("ARGformer_final_checkpoint_load")
        saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        state = saved.get("state", {})
        ts = state.get("timestamp", {}).get("Timestamp", {})
        if ts.get("epoch") == 1 and ts.get("batch") == 2135:
            counters = _checkpoint_counters(state)
            candidates.append(dict(checkpoint_ref=_payload_ref(path, resource_check), **counters))
        del saved
    _need(len(candidates) == 1, "One actual final checkpoint at full scientific budget required")
    _need(alias is None or alias == Path(candidates[0]["checkpoint_ref"]["path"]).name,
          "Latest alias must name the verified final regular checkpoint")
    return candidates[0]


def run_contrastive(spec, *, resource_check):
    """Run full original author training inside the current GPU0 guarded child."""
    _need(callable(resource_check), "Current admitted worker resource callback required")
    resource_check("ARGformer_before_science")
    out = _validate_spec(spec)
    _need(Path(sys.executable).resolve() == (ROOT / NATIVE_PYTHON).resolve()
          and sys.dont_write_bytecode and sys.flags.no_user_site,
          "Exact isolated native interpreter with -B -s required")
    _need(os.environ.get("CUDA_VISIBLE_DEVICES") == GPU0_UUID
          and os.environ.get("ARGGNN_PHYSICAL_GPU") == "0", "Only admitted physical GPU0 visibility")
    _need(not out.exists(), "Fresh guarded native output; no saved phase replay")
    resource_check("ARGformer_before_native_output")
    out.mkdir()
    for split in ("train", "val"):
        _need(not (ROOT / DATA / "contrastive" / split /
                  ("tokenized_" + split + "_sequences_and_vocab.pkl")).exists(),
              "Preserve exact raw author dataset selection")
    for ref in spec["requirements"]["input_refs"]:
        _need(_payload_ref(ROOT / ref["path"], resource_check) == ref, "Exact native observed input required")
    for name in ("TMPDIR", "TMP", "TEMP", "HF_HOME", "XDG_CACHE_HOME", "TORCH_HOME", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH"):
        path = out / (".tmp" if name in ("TMPDIR", "TMP", "TEMP") else ".cache/" + name.lower())
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    tempfile.tempdir = str(out / ".tmp")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", WANDB_MODE="disabled", CUDA_CACHE_MAXSIZE=str(256 * 2**20))
    # IPC address allocation only; GPU policy belongs to the current worker.
    # Calling this unchanged function avoids the historical GPU4/5 bootstrap.
    bootstrap = _load_source("ipc")
    bootstrap.install_abstract_unix_v1("benchmark_argformer_cl")
    sys.path.insert(0, str(ROOT / AUTHOR))
    sys.path.insert(0, str(ROOT))
    os.chdir(out)
    resource_check("ARGformer_before_native_imports")
    import torch
    from omegaconf import OmegaConf
    _need(torch.cuda.device_count() == 1, "Exactly one visible CUDA device required")
    cap = int(os.environ.get("ARGGNN_MAX_CUDA_RESERVED_BYTES", "0"))
    total = torch.cuda.get_device_properties(0).total_memory
    _need(0 < cap <= min(74 * 2**30, total), "Current worker finite CUDA allocator cap required")
    torch.cuda.set_per_process_memory_fraction(cap / total, 0)
    module = importlib.import_module("sequence_contrastive")
    _need(Path(module.__file__).resolve() == ROOT / AUTHOR / "sequence_contrastive.py",
          "Pinned unchanged author contrastive origin")
    for name, imported in tuple(sys.modules.items()):
        origin = getattr(imported, "__file__", None)
        if (name == "src" or name.startswith("src.")) and origin:
            _need(Path(origin).resolve().is_relative_to(ROOT / AUTHOR), "Author src namespace must remain isolated")
    integration = _load_source("integration")
    bridge = integration.bridge_v1()
    eligible = bridge.read_eligibility_v1()
    cfg = OmegaConf.load(ROOT / AUTHOR / "yamls/contrastive.yaml")
    cfg.data_local = str(ROOT / DATA / "contrastive")
    cfg.run_name = "C3_native_contrastive"
    cfg.save_folder = str(out / "checkpoints")
    cfg.load_path = None
    cfg.autoresume = False
    cfg.loggers = {}
    cfg.device = "gpu"
    cfg.model.pretrained_model_name = str((ROOT / PINS["base_config"]["path"]).parent)
    cfg.model.pretrained_checkpoint = str(ROOT / PINS["checkpoint"]["path"])
    cfg.save_num_checkpoints_to_keep = 1
    OmegaConf.save(cfg, out / "resolved_io_config.yaml", resolve=True)
    resource_check("ARGformer_before_full2135_author_updates")
    initialization = integration.run_original_keep1_contrastive_train_v1(
        cfg, eligible, author_train=module.train)
    resource_check("ARGformer_after_full_author_train_return")
    result = dict(schema="benchmark_argformer_full_contrastive_completion_v1",
                  status="completed_full_native_contrastive", parent_id=spec["parent_id"],
                  **_final_checkpoint(out / "checkpoints", torch, resource_check),
                  initializer_checkpoint_ref=copy.deepcopy(PINS["checkpoint"]),
                  initialization=initialization, author_code_modified=False,
                  settings=copy.deepcopy(SETTINGS), query_ancestry_truth_opened=False,
                  exact_paper_reproduction=False, historical_MLM_outer_exit_code=1,
                  downstream_embedding_retrieval_scoring_performed=False,
                  actual_worker_wait_and_owned_drain_still_required=True)
    resource_check("ARGformer_before_completion_metadata")
    with (out / "native_contrastive_completed.json").open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    resource_check("ARGformer_after_completion_metadata")
    return result
