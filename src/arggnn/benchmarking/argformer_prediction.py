"""Native ARGformer prediction stages, released after genuine producer completion.

Only actual native completion, direct worker wait0 and a succeeded queue record
can supply a checkpoint or embedding input. No historical authority or receipt
schema is synthesized. There is no truth reader, scoring, admission or launcher.
Each callable belongs in a fresh, separately released guarded native worker.
"""
from __future__ import annotations

import copy
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types

from . import argformer_execution as training
from .contracts import ROOT, ContractError, file_ref, read_ref

PINS = {'embeddings': {'path': '.runtime/author_sources/argformer_f7545be7bca0/embeddings.py', 'sha256': 'f85c8c3b039b75ad1118d2a4904d3c77d5a253227032b9def551f09ab0144869'}, 'retrieve': {'path': '.runtime/author_sources/argformer_f7545be7bca0/retrieve.py', 'sha256': '520f1961c396432a875a40be253a01648e7debf1fad8c9a274a7522d3809e7ea'}, 'concat': {'path': 'src/arggnn/american_c3_native_concat_v1.py', 'sha256': 'a9de71c71baed134f829619b23ca11b910eefb63be92af24d8993e232b70c7bb'}, 'evaluator': {'path': 'src/arggnn/american_c3_evaluate_v1.py', 'sha256': '1ada0ab40fe9f198b89a1953bb7c0688e68f1dd44b34dc118600ac25eefc70d3'}, 'ram_io': {'path': 'research/reviews/native_VAL000_downstream_IO_efficiency_source_20261016_v1/efficient_downstream_v1.py', 'sha256': 'bab7057a958e9cfd51f503b5680034c0067226a661b820eb42bb16c7133a94b9'}, 'frozen_training_adapter': {'path': 'src/arggnn/benchmarking/argformer_execution.py', 'sha256': '7e28463a549c1052b23ed65dff52fadc3033f913dd5a4cd00dc52fe7b7e23f2c'}}
ROWS = {"contrastive/train": 8748479, "contrastive/val": 972757,
        "query_paths/train": 9003, "query_paths/val": 997}
SCHEMA = "benchmark_argformer_prediction_stage_v1"


def _need(ok, message):
    if not ok:
        raise ContractError(message)


def requirements():
    """Inspect source/compact metadata only; no future checkpoint is invented."""
    base = training.requirements()
    for ref in PINS.values():
        _need(file_ref(ref["path"]) == ref, "Pinned native prediction source changed")
    return dict(native_python=training.NATIVE_PYTHON, physical_gpu=0,
                gpu_uuid=training.GPU0_UUID,
                source_refs=list({r["path"]: r for r in base["source_refs"] +
                                  list(PINS.values()) +
                                  [file_ref(Path(__file__).resolve())]}.values()),
                rows=copy.deepcopy(ROWS), embedding_dim=768, max_seq_len=128,
                batch_size=256, pooling="head_norm", layer=None, max_samples=0,
                include_splits="one_split_per_process_then_train_val_concatenation",
                retrieval=dict(topk=1, q_batch=1024, e_chunk=100000),
                common_query_records=10000, reference_rows=9721236,
                producer_record_keys=["completion_ref", "actual_wait_ref", "queue_result_ref"],
                native_embedding_bytes_approx=(9721236 * (768*4 + 2*128*8 + 2*8)
                                              + 10000 * (768*4 + 2*128*8 + 8)),
                duplicated_combined_embedding_files_written=False,
                exact_paper_reproduction=False, scientific_payloads_opened=False,
                private_truth_access=False, scoring_performed=False)


def _producer(bundle, stage):
    """Use actual current-worker completion/wait and a succeeded queue record."""
    _need(type(bundle) is dict and set(bundle) == {
        "completion_ref", "actual_wait_ref", "queue_result_ref"},
        "Actual completion, worker wait and successful queue record required")
    completion, wait, queue = (read_ref(bundle[k]) for k in
                                ("completion_ref", "actual_wait_ref", "queue_result_ref"))
    entries = queue.get("jobs", [queue])
    matches = [entry for entry in entries if entry.get("admission", {}).get("job_id") == wait["job_id"]]
    _need(len(matches) == 1 and matches[0].get("status") == "succeeded"
          and matches[0].get("actual_returncode") == 0 and matches[0].get("error") is None,
          "This exact producer must have genuinely succeeded in the current queue")
    job = read_ref(matches[0]["admission"]["input_ref"])
    expected = ("arggnn.benchmarking.argformer_execution", "run_contrastive") if stage == "contrastive" else (
        "arggnn.benchmarking.argformer_prediction", "run_embedding")
    _need(job.get("schema") == "benchmark_guarded_worker_spec_v1"
          and (job.get("callable_module"), job.get("callable_name")) == expected
          and wait.get("job_id") == job["job_id"] and wait.get("actual_direct_wait") is True
          and wait.get("actual_direct_returncode") == 0 and wait.get("supervisor_error") is None,
          "Exact native producer family and actual current direct wait0 required")
    expected_source = PINS["frozen_training_adapter"] if stage == "contrastive" else file_ref(Path(__file__).resolve())
    _need(expected_source in job.get("source_refs", []), "Actual producer must bind its exact executed source")
    native = training._output(job["scientific_spec"]["output_directory"])
    filename = "native_contrastive_completed.json" if stage == "contrastive" else "embedding_completed.json"
    _need(bundle["completion_ref"]["path"] == str((native / filename).relative_to(ROOT))
          and bundle["actual_wait_ref"]["path"] ==
          str((ROOT / job["output_directory"] / "actual_wait.json").relative_to(ROOT)),
          "Exact native completion/current worker output joins")
    _need(completion.get("parent_id") == "american_c3_val_000"
          and completion.get("query_ancestry_truth_opened") is False,
          "Only the exact observation-only original parent")
    if stage == "contrastive":
        _need(completion.get("schema") == "benchmark_argformer_full_contrastive_completion_v1"
              and completion.get("status") == "completed_full_native_contrastive"
              and completion.get("completed_epochs") == 1
              and completion.get("actual_optimizer_updates") == 2135
              and completion.get("optimizer_unique_step_values") == [2135]
              and type(completion.get("optimizer_parameter_state_count")) is int
              and completion["optimizer_parameter_state_count"] > 0
              and completion.get("initializer_checkpoint_ref") == training.PINS["checkpoint"],
              "Actual full2135 saved-counter completion required")
    else:
        key = completion.get("view", "") + "/" + completion.get("split", "")
        _need(completion.get("schema") == "benchmark_argformer_single_split_embedding_completion_v1"
              and key in ROWS and completion.get("rows") == ROWS[key]
              and completion.get("pooling") == "head_norm" and completion.get("max_samples") == 0
              and completion.get("checkpoint_ref") == job["scientific_spec"]["checkpoint_ref"],
              "Complete unmodified single-split embedding producer")
    return completion



def _spec(stage, out, **fields):
    return dict(schema=SCHEMA, stage=stage, parent_id="american_c3_val_000",
                output_directory=str(out.relative_to(ROOT)), requirements=requirements(), **fields)


def build_embedding_spec(producer, *, view, split, output_directory):
    """Bind a future genuine full contrastive producer to one complete split."""
    key = view + "/" + split
    _need(key in ROWS, "One registered complete native view/split required")
    completion = _producer(producer, "contrastive")
    out = training._output(output_directory)
    _need(not out.exists(), "Fresh native embedding output required")
    return _spec("embedding", out, producer=producer,
                 checkpoint_ref=completion["checkpoint_ref"], view=view, split=split,
                 expected_rows=ROWS[key])


def build_retrieval_spec(embedding_producers, *, output_directory):
    """Require all four actual complete embedding producers before native top1."""
    _need(type(embedding_producers) is dict and set(embedding_producers) == set(ROWS),
          "All four genuine single-split producers required")
    products = {key: _producer(ref, "embedding") for key, ref in embedding_producers.items()}
    first = products["contrastive/train"]["checkpoint_ref"]
    for key, item in products.items():
        _need(item["view"] + "/" + item["split"] == key and item["checkpoint_ref"] == first,
              "Four complete observed views must use the same genuine final checkpoint")
    out = training._output(output_directory)
    _need(not out.exists(), "Fresh native retrieval output required")
    return _spec("retrieve_and_seal", out, embedding_producers=embedding_producers,
                 checkpoint_ref=first)


def _validated(spec, stage):
    out = training._output(spec["output_directory"])
    if stage == "embedding":
        expected = build_embedding_spec(spec["producer"], view=spec["view"],
                                        split=spec["split"], output_directory=out)
    else:
        expected = build_retrieval_spec(spec["embedding_producers"], output_directory=out)
    _need(spec == expected, "Frozen complete native prediction specification required")
    return out


def _native(out, resource_check):
    resource_check("ARGformer_prediction_before_native")
    _need(Path(sys.executable).resolve() == (ROOT / training.NATIVE_PYTHON).resolve()
          and sys.dont_write_bytecode and sys.flags.no_user_site,
          "Exact native interpreter -B -s required")
    _need(os.environ.get("CUDA_VISIBLE_DEVICES") == training.GPU0_UUID
          and os.environ.get("ARGGNN_PHYSICAL_GPU") == "0", "Current admitted physicalGPU0 required")
    out.mkdir()
    for name in ("TMPDIR", "TMP", "TEMP", "HF_HOME", "XDG_CACHE_HOME",
                 "TORCH_HOME", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH"):
        path = out / (".tmp" if name in ("TMPDIR", "TMP", "TEMP") else ".cache/" + name.lower())
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    tempfile.tempdir = str(out / ".tmp")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", WANDB_MODE="disabled")
    training._load_source("ipc").install_abstract_unix_v1("benchmark_argformer_predict")
    sys.path.insert(0, str(ROOT / training.AUTHOR))
    sys.path.insert(0, str(ROOT))
    os.chdir(out)
    import torch
    cap = int(os.environ.get("ARGGNN_MAX_CUDA_RESERVED_BYTES", "0"))
    _need(torch.cuda.device_count() == 1 and 0 < cap <= 74 * 2**30,
          "One visible GPU0 and finite worker CUDA cap required")
    total = torch.cuda.get_device_properties(0).total_memory
    _need(cap <= total, "CUDA cap exceeds actual device")
    torch.cuda.set_per_process_memory_fraction(cap / total, 0)
    return torch


def _author(name):
    module = importlib.import_module(name)
    _need(Path(module.__file__).resolve() == ROOT / training.AUTHOR / (name + ".py"),
          "Exact unchanged author prediction module required")
    return module


def _write_json(path, value, resource_check):
    resource_check("ARGformer_prediction_metadata_write")
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return file_ref(path)


def run_embedding(spec, *, resource_check):
    """Original learned-head embedding for every row of one native split."""
    _need(callable(resource_check), "Current worker resource callback required")
    out = _validated(spec, "embedding")
    prep = read_ref(training.PINS["preparation"])
    counts = read_ref(training.PINS["counts"])
    key = spec["view"] + "/" + spec["split"]
    inputs = [prep["native_dataset_refs"][key], prep["vocab_ref"],
              prep["full_path_id_order_refs"][key]]
    if spec["view"] == "contrastive":
        inputs.append(counts["reference_label_refs"][spec["split"]])
    for ref in inputs + [spec["checkpoint_ref"]]:
        _need(training._payload_ref(ROOT / ref["path"], resource_check) == ref,
              "Exact actual native prediction input bytes required")
    _need(not (ROOT / training.DATA / spec["view"] / spec["split"] /
              ("tokenized_" + spec["split"] + "_sequences_and_vocab.pkl")).exists(),
          "Native raw corpus selection must remain unchanged")
    torch = _native(out, resource_check)
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(ROOT / training.AUTHOR / "yamls/contrastive.yaml")
    cfg.data_local = str(ROOT / training.DATA / spec["view"])
    cfg.run_name = "C3_native_embedding"
    cfg.save_folder = str(out / "unused_checkpoints")
    cfg.load_path = None
    cfg.autoresume = False
    cfg.loggers = {}
    cfg.device = "gpu"
    cfg.model.pretrained_model_name = str((ROOT / training.PINS["base_config"]["path"]).parent)
    cfg.model.pretrained_checkpoint = None
    config_path = out / "resolved_io_config.yaml"
    OmegaConf.save(cfg, config_path, resolve=True)
    module = _author("embeddings")
    model, loaded_cfg = module.load_checkpoint_model(str(config_path), str(ROOT / spec["checkpoint_ref"]["path"]))
    resource_check("ARGformer_before_complete_native_embedding")
    vectors = module.embed_arg_dataset(
        model, loaded_cfg, output_path=str(out / "embeddings.pt"), batch_size=256,
        pooling_strategy="head_norm", layer_index=None, max_samples=0,
        is_classification=True, include_splits=spec["split"],
        use_split_labels=spec["view"] == "contrastive")
    _need(tuple(vectors.shape) == (spec["expected_rows"], 768)
          and vectors.dtype == torch.float32 and bool(torch.isfinite(vectors).all()),
          "Every complete finite native FP32 learned-head embedding required")
    resource_check("ARGformer_after_complete_native_embedding")
    result = dict(schema="benchmark_argformer_single_split_embedding_completion_v1",
                  parent_id=spec["parent_id"], view=spec["view"], split=spec["split"],
                  rows=spec["expected_rows"], checkpoint_ref=spec["checkpoint_ref"],
                  embedding_ref=training._payload_ref(out / "embeddings.pt", resource_check),
                  raw_dataset_ref=prep["native_dataset_refs"][key],
                  row_order_ref=prep["full_path_id_order_refs"][key], pooling="head_norm",
                  max_samples=0, query_ancestry_truth_opened=False,
                  actual_worker_wait_and_owned_drain_still_required=True)
    _write_json(out / "embedding_completed.json", result, resource_check)
    return result


class _Writer:
    def __init__(self, path, resource_check):
        self.path, self.resource_check = path, resource_check

    def save(self, name, value, kind=None):
        self.resource_check("ARGformer_native_seal_write")
        path = self.path / name
        _need(path.parent == self.path and not path.exists(), "Fresh bounded native seal output")
        if kind == "npz":
            import numpy as np
            with path.open("xb") as stream:
                np.savez_compressed(stream, **value)
            return training._payload_ref(path, self.resource_check)
        _need(kind is None, "Only native seal metadata and arrays are written")
        return _write_json(path, value, self.resource_check)

    def npz(self, name, **arrays):
        return self.save(name, arrays, "npz")

    def json(self, name, value):
        return self.save(name, value)


def run_retrieval(spec, *, resource_check):
    """Full native top1 and unchanged common10k observation-only prediction seal."""
    _need(callable(resource_check), "Current worker resource callback required")
    out = _validated(spec, "retrieve_and_seal")
    products = {key: _producer(ref, "embedding")
                for key, ref in spec["embedding_producers"].items()}
    for item in products.values():
        ref = item["embedding_ref"]
        _need(training._payload_ref(ROOT / ref["path"], resource_check) == ref,
              "Exact read-only native embedding product required")
    torch = _native(out, resource_check)
    import numpy as np
    concat = importlib.import_module("arggnn.american_c3_native_concat_v1")
    io = importlib.import_module(
        "research.reviews.native_VAL000_downstream_IO_efficiency_source_20261016_v1.efficient_downstream_v1")
    evaluator = importlib.import_module("arggnn.american_c3_evaluate_v1")
    prep = read_ref(training.PINS["preparation"])
    writer = _Writer(out, resource_check)
    ram = io.RAMCombinationWriter(writer, {"outputs": products, "checkpoint_ref": spec["checkpoint_ref"]}, prep)
    for view in ("query_paths", "contrastive"):
        parts = []
        for split in ("train", "val"):
            resource_check("ARGformer_complete_native_embedding_read")
            key = view + "/" + split
            item = products[key]
            _need(item["raw_dataset_ref"] == prep["native_dataset_refs"][key]
                  and item["row_order_ref"] == prep["full_path_id_order_refs"][key],
                  "Exact full-path sidecars from the same observed preparation required")
            order = concat.row_order(item["row_order_ref"], parent_id=spec["parent_id"], reference=view == "contrastive")
            value = torch.load(ROOT / item["embedding_ref"]["path"], map_location="cpu", weights_only=True, mmap=True)
            n, meta = ROWS[key], value["metadata"]
            expected_keys = {"embeddings", "input_ids", "attention_masks", "extant_node_ids", "metadata"}
            if view == "contrastive":
                expected_keys.add("labels")
            _need(set(value) == expected_keys and meta["splits"] == [split]
                  and meta["num_samples"] == n and meta["embedding_dim"] == 768
                  and meta["pooling_strategy"] == "head_norm" and meta["layer_index"] is None,
                  "Complete original single-split metadata/tensor fields required")
            _need(value["embeddings"].shape == (n, 768) and value["embeddings"].dtype == torch.float32
                  and bool(torch.isfinite(value["embeddings"]).all())
                  and value["input_ids"].shape == value["attention_masks"].shape == (n, 128)
                  and value["input_ids"].dtype == value["attention_masks"].dtype == torch.long
                  and value["extant_node_ids"].dtype == torch.long
                  and np.array_equal(value["extant_node_ids"].reshape(-1).numpy(), order["sample_nodes"]),
                  "Exact complete native tensors and sidecar sample row order required")
            if view == "contrastive":
                _need(value["labels"].dtype == torch.long and
                      np.array_equal(value["labels"].reshape(-1).numpy(), order["reference_classes_0based"]),
                      "Only exact observed reference labels are permitted")
            parts.append(value)
        _need({k:v for k,v in parts[0]["metadata"].items() if k not in ("splits", "num_samples")} ==
              {k:v for k,v in parts[1]["metadata"].items() if k not in ("splits", "num_samples")},
              "Compatible native train/val metadata")
        combined = {name: torch.cat((parts[0][name], parts[1][name]), dim=0)
                    for name in parts[0] if name != "metadata"}
        combined["metadata"] = dict(parts[0]["metadata"], splits=["train", "val"],
                                   num_samples=sum(p["metadata"]["num_samples"] for p in parts))
        ram.values[view] = combined
        del parts, combined
    module = _author("retrieve")
    tokens = {"RAM_native_reference_train_then_val": ram.values["contrastive"],
              "RAM_native_query_train_then_val": ram.values["query_paths"]}
    def load_ram(path):
        _need(path in tokens, "Only complete validated native RAM tensors may be retrieved")
        return tokens[path]
    native_main = types.FunctionType(module.main.__code__, dict(module.main.__globals__, load_embeddings=load_ram),
                                     module.main.__name__, module.main.__defaults__, module.main.__closure__)
    previous = sys.argv
    try:
        sys.argv = [str(ROOT / training.AUTHOR / "retrieve.py"), "--corpus", "RAM_native_reference_train_then_val",
                    "--queries", "RAM_native_query_train_then_val", "--out", str(out / "retrieval.pt"),
                    "--topk", "1", "--q_batch", "1024", "--e_chunk", "100000"]
        resource_check("ARGformer_before_original_full_top1")
        native_main()
    finally:
        sys.argv = previous
    resource_check("ARGformer_after_original_full_top1")
    retrieval_ref = training._payload_ref(out / "retrieval.pt", resource_check)
    registry = evaluator.registry(prep["focals_ref"])
    _need(len(registry["record_ids"]) == 10000, "Exact complete common10000 focal registry required")
    seal_out = out / "seal"
    seal_out.mkdir()
    cfg = dict(evaluator.DEFAULTS, mode="native_top1_seal", method="native_ARGformer_C3_release_per_ARG",
               preparation_ref=training.PINS["preparation"], focal_registry_ref=prep["focals_ref"],
               native_query_embeddings_ref=ram.descriptor("query_paths", "RAM_tensors"),
               native_corpus_embeddings_ref=ram.descriptor("contrastive", "RAM_tensors"),
               native_retrieval_ref=retrieval_ref, external_truth_ref=None, prediction_ref=None)
    # Existing reviewed adaptation changes only the embedding-load expression.
    seal_ref = io._original_RAM_seal_v1({}, ram)(cfg, _Writer(seal_out, resource_check), registry)
    result = dict(schema="benchmark_argformer_native_top1_prediction_completion_v1",
                  parent_id=spec["parent_id"], checkpoint_ref=spec["checkpoint_ref"],
                  retrieval_ref=retrieval_ref, prediction_seal_ref=seal_ref, records=10000,
                  reference_rows=9721236, query_rows=10000, topk=1,
                  combined_embedding_files_written=False,
                  native_normalization_topk_ties_and_scores_unchanged=True,
                  query_ancestry_truth_opened=False, scoring_performed=False,
                  actual_worker_wait_and_owned_drain_still_required=True)
    _write_json(out / "retrieval_completed.json", result, resource_check)
    return result
