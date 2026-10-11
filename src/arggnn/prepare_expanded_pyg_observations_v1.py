"""Root-admitted CPU preparation of the registered expanded inner PyG cache.

Default CLI reports source pins only. Execution preloads each ready observed
TRAIN/VAL parent once with eight spawn workers, retains it in host RAM, fits the
unchanged legacy bins on all 32 TRAIN, then prepares all 40 complete parents.
No producer, inference, outer parent, target decoder or GPU is invoked here.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import signal
import sys
import tempfile
import threading
import time
import traceback

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE = "src/arggnn/prepare_expanded_pyg_observations_v1.py"
CONFIG = "configs/expanded_full_panel_v1.json"
REGISTRATION = "research/reviews/expanded_full_panel_v1_root/registration.json"
PRODUCER = "src/arggnn/prepare_expanded_full_panel_v1.py"
ORIGINAL_PRODUCER = "src/arggnn/prepare_ancestralpaths_full_panel_discovery_v1.py"
STORAGE = "src/arggnn/project_storage_cpu_closed_scopes_v1.py"
BASE_STORAGE = "src/arggnn/project_storage_closed_scopes_v1.py"
CACHE_SOURCES = tuple("src/arggnn/" + name for name in (
    "prepare_expanded_pyg_observations_v1.py",
    "mutation_routed_full_panel_observations_v1.py",
    "mutation_routed_full_panel_cache_v1.py",
    "mutation_routed_tree_observations_v1.py",
    "mutation_routed_parent_loader_v1.py",
    "mutation_routed_whole_tree_gnn_v1.py",
    "mutation_routed_pyg_dataset_v1.py",
    "mutation_routed_observation_cache_v1.py",
    "observed_variant_residual_v1.py",
    "full_panel_native_baseline_contracts_v1.py",
    "census_tsinfer_fit_v1.py",
    "census_tsinfer_observation_v1.py",
))
SOURCE_PATHS = CACHE_SOURCES + (STORAGE, BASE_STORAGE)
SCHEMA = "expanded_pyg_observations_v1_root_admission"


def _path(value):
    if not isinstance(value, (str, Path)):
        raise ValueError("Explicit project path required")
    path = Path(value)
    if ".." in path.parts:
        raise ValueError("Path traversal forbidden")
    if path.is_absolute():
        path = path.relative_to(ROOT)
    if not path.parts:
        raise ValueError("Dedicated project path required")
    for count in range(1, len(path.parts) + 1):
        if ROOT.joinpath(*path.parts[:count]).is_symlink():
            raise ValueError("Project symlink forbidden")
    result = ROOT / path
    result.resolve().relative_to(ROOT)
    return result


def _sha(path):
    digest = hashlib.sha256()
    with _path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ref(path):
    path = _path(path)
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": _sha(path)}


def _read(ref, expected_path=None):
    if (not isinstance(ref, dict) or set(ref) != {"path", "sha256"}
            or (expected_path is not None and ref["path"] != expected_path)):
        raise ValueError("Exact project path/SHA256 binding required")
    path = _path(ref["path"])
    payload = path.read_bytes()
    if (path.relative_to(ROOT).as_posix() != ref["path"]
            or hashlib.sha256(payload).hexdigest() != ref["sha256"]):
        raise ValueError("Exact project path/SHA256 binding required")
    return json.loads(payload)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _rss(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except FileNotFoundError:
        pass
    return 0


def _admit(spec):
    expiry = datetime.fromisoformat(spec["expires_at_utc"].replace("Z", "+00:00"))
    if (spec.get("schema") != SCHEMA or spec.get("root_authorizer") != "root_agent"
            or spec.get("execution_authorized") is not True or spec.get("status") != "active"
            or spec.get("CPU_workers") != 8 or spec.get("outer_included") is not False
            or spec.get("targets_read") is not False or spec.get("GPU_authorized") is not False
            or expiry.tzinfo is None or expiry <= datetime.now(timezone.utc)
            or os.environ.get("CUDA_VISIBLE_DEVICES") != ""):
        raise ValueError("Active exact CPU-only root admission required")
    for key in ("max_cpu_rss_bytes", "output_cap_bytes", "readiness_timeout_seconds"):
        if type(spec.get(key)) is not int or spec[key] <= 0:
            raise ValueError("Positive admitted CPU RAM, output cap and readiness timeout required")
    if not 0.5 <= spec.get("poll_seconds", 2.0) <= 30:
        raise ValueError("Readiness polling must be between 0.5 and 30 seconds")
    sources = spec["source_sha256"]
    if not isinstance(sources, dict) or set(sources) != set(SOURCE_PATHS):
        raise ValueError("Exact observation/reader/storage source closure required")
    for path, digest in sources.items():
        if _ref(path) != {"path": path, "sha256": digest}:
            raise ValueError("Admitted implementation source bytes differ")
    if (spec["source_ref"] != _ref(SOURCE) or spec["storage_adapter_ref"] != _ref(STORAGE)
            or spec["producer_ref"] != _ref(PRODUCER)
            or spec["original_producer_ref"] != _ref(ORIGINAL_PRODUCER)):
        raise ValueError("Exact preparation/producer/storage sources required")
    config = _read(spec["config_ref"], CONFIG)
    registration = _read(spec["registration_ref"], REGISTRATION)
    registry = registration["parent_registry"]
    if (registry != config["parent_registry"]
            or registration["registry_sha256"] != _digest(registry)
            or registration["source_ref"] != spec["producer_ref"]
            or registration["config_ref"] != spec["config_ref"]
            or registration["outer_evaluation_authorized"] is not False
            or config["output_root"] != "data/expanded_full_panel_v1"):
        raise ValueError("Frozen generation config/registration/registry binding differs")
    _read(spec["runtime_ref"])
    output = _path(spec["output_directory"])
    storage = spec["storage"]
    if (storage.get("accounting_mode") != "journal_cpu_completed_scopes_credit_v1"
            or storage["output_directory"] != spec["output_directory"]):
        raise ValueError("Admitted CPU-overlay output guard required")
    required = [spec[key] for key in ("source_ref", "config_ref", "registration_ref",
                "runtime_ref", "storage_adapter_ref", "producer_ref", "original_producer_ref")]
    required.extend({"path": path, "sha256": digest} for path, digest in sources.items())
    if any(ref not in storage["reservation"]["consumer_bindings"] for ref in required):
        raise ValueError("Source/config/registration/runtime closure must be reserved")
    # Input includes storage authority. Verify its CLI hash separately to avoid
    # a circular reservation binding to a hash of the reservation itself.
    return config, registry, output


def _ready_entry(row, spec, config):
    parent_id = row["parent_id"]
    base = config["output_root"] + "/" + parent_id
    completed = _path(base + "/expanded_parent_completed.json")
    if completed.exists():
        value = json.loads(completed.read_text())
        if (value.get("schema") != "expanded_full_panel_v1_parent_completion"
                or value.get("parent_id") != parent_id or value.get("split") != row["split"]
                or value.get("generation_completed") is not True
                or value.get("Relate_completed") is not True
                or value.get("source_ref") != spec["producer_ref"]
                or value.get("original_producer_ref") != spec["original_producer_ref"]
                or value.get("config_content_sha256") != _digest(config)
                or value.get("parent_registry_sha256") != _digest(config["parent_registry"])
                or value.get("outer_truth_evaluated") is not False):
            raise ValueError("Completed parent source/config/registry binding differs: " + parent_id)
        entry_ref = value["observation_entry_ref"]
        entry = _read(entry_ref, base + "/full_panel_observed_entry.json")
        if (entry.get("observations_only") is not True or entry.get("truth_or_oracle_read") is not False
                or entry.get("parent_id") != parent_id or entry.get("split") != row["split"]):
            raise ValueError("Registered observed-only entry required: " + parent_id)
        expected = dict(source_roster="observed/sample_manifest.json",
            vcf="observed/genotypes.vcf", site_coordinates="observed/site_coordinate_manifest.json",
            scoring_coordinates="observed/scoring_coordinates.npz",
            observation_manifest="observed/observation_manifest.json",
            native_tree="relate/post_size.observed.trees",
            derived_roster="relate/observed_sample_manifest.json",
            inference_receipt="full_panel_observed_inference_receipt.json")
        if (set(entry["refs"]) != set(expected)
                or any(entry["refs"][key]["path"] != base + "/" + relative
                       for key, relative in expected.items())
                or entry["original_native_inference_receipt"]["path"] != base + "/relate/inference.json"):
            raise ValueError("Fixed observed-only parent asset paths required: " + parent_id)
        # Metadata/file size only. No target or native tree bytes are read here.
        native = _path(entry["refs"]["native_tree"]["path"])
        return entry_ref, native.stat().st_size
    failure = _path("research/reviews/expanded_full_panel_v1_root/" + parent_id + "/launcher_exit.json")
    if failure.exists():
        value = json.loads(failure.read_text())
        if value.get("returncode", 0) != 0:
            raise RuntimeError("Registered producer failed: " + parent_id)
    return None


def _worker_init(scratch):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch
    import multiprocessing.connection as connection
    from .mutation_routed_pyg_dataset_v1 import _worker_ipc_address_v1
    torch.set_num_threads(1)
    connection.arbitrary_address = _worker_ipc_address_v1
    directory = _path(Path(scratch) / ("worker_" + str(os.getpid())))
    directory.mkdir(exist_ok=False)
    os.environ["TMPDIR"] = str(directory)
    tempfile.tempdir = str(directory)


def _load(row, entry_ref, scratch):
    from .census_tsinfer_fit_v1 import ConsumerReaderV1
    from .mutation_routed_full_panel_observations_v1 import load_full_parent_v1
    started = time.monotonic()
    directory = _path(Path(scratch) / row["parent_id"])
    directory.mkdir(exist_ok=False)
    try:
        observed = load_full_parent_v1(row, entry_ref, ConsumerReaderV1(), scratch_directory=directory)
    finally:
        # The loader unlinks its own native snapshot. Never prune foreign files.
        directory.rmdir()
    stats = dict(parent_id=row["parent_id"], split=row["split"], worker_pid=os.getpid(),
        load_seconds=time.monotonic() - started, worker_peak_rss_bytes=_peak_rss(),
        native_holds=len(observed.trees), original_sites=len(observed.positions_bp0),
        forest_vertices=sum(len(tree.parent) for tree in observed.trees),
        haplotypes=len(observed.sample_ids), query_haplotypes=len(observed.query_ids))
    return observed, stats


def _prepare(observed, bins):
    from . import mutation_routed_full_panel_observations_v1 as observations
    from .mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeInputsV1
    started = time.monotonic()
    bank = observations._full_prepare(observed.trees, observed.intervals_bp,
        observed.positions_bp0, observed.genotype, observed.allele_count,
        observed.sample_ids, observed.reference_class, observed.sample_age,
        observed.query_ids, observed.readout_x_bp, sham_bin_edges=bins, fixture=False)
    inputs = MutationRoutedWholeTreeInputsV1.from_arrays(bank.model_arrays())
    observations.validate_full_bank_v1(inputs)
    item = dict(model_inputs=inputs, scoring=observed.scoring,
        observations_receipt=dict(observed.receipt, preprocessing=bank.metadata))
    return item, dict(parent_id=observed.parent_id, prepare_seconds=time.monotonic() - started,
                     prepare_worker_pid=os.getpid(), prepare_worker_peak_rss_bytes=_peak_rss())


class _Output:
    def __init__(self, directory, guard):
        self.directory, self.guard = directory, guard
        guard(65536)
        directory.mkdir(parents=True, exist_ok=False)
        self.events = directory / "events.jsonl"
        self.events.touch(exist_ok=False)
        self.lock = threading.Lock()

    def event(self, kind, **fields):
        payload = (json.dumps(dict(utc=_utc(), event=kind, **fields), sort_keys=True,
                              allow_nan=False) + "\n").encode()
        with self.lock:
            self.guard(len(payload))
            with self.events.open("ab") as stream:
                stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        print(payload.decode().rstrip(), flush=True)

    def save(self, name, value):
        from .mutation_routed_observation_cache_v1 import _write_fresh_atomic
        payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        self.guard(len(payload))
        _write_fresh_atomic(self.directory / name, payload)
        return _ref(self.directory / name)


class _Monitor:
    """Sample only the main process and this executor registered children."""
    def __init__(self, executor, limit):
        self.executor, self.limit = executor, limit
        self.peak_total = self.peak_main = self.peak_worker = 0
        self.breach = None
        self.stop = threading.Event()
        self.old_handler = signal.signal(signal.SIGUSR1, self._raise)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _raise(self, *_):
        raise MemoryError(self.breach or "Admitted CPU RSS limit exceeded")

    def _run(self):
        while not self.stop.wait(0.5):
            # No global process scan. The executor itself owns this PID registry.
            workers = list((self.executor._processes or {}).values())
            main = _rss(os.getpid())
            values = [_rss(process.pid) for process in workers if process.pid]
            total = main + sum(values)
            self.peak_main = max(self.peak_main, main)
            self.peak_worker = max(self.peak_worker, max(values, default=0))
            self.peak_total = max(self.peak_total, total)
            if total > self.limit:
                self.breach = f"Owned process RSS sum {total} exceeds admitted {self.limit} bytes"
                os.kill(os.getpid(), signal.SIGUSR1)
                return

    def close(self):
        self.stop.set(); self.thread.join(timeout=2)
        signal.signal(signal.SIGUSR1, self.old_handler)


def _drain(executor, failed):
    owned = list((executor._processes or {}).values())
    if failed:
        for process in owned:
            if process.is_alive():
                process.terminate()
        for process in owned:
            process.join(timeout=5)
            if process.is_alive():
                process.kill(); process.join(timeout=5)
    executor.shutdown(wait=True, cancel_futures=True)
    if any(process.is_alive() for process in owned):
        raise RuntimeError("Owned CPU workers did not drain")


def execute_v1(spec, input_ref):
    config, registry, directory = _admit(spec)
    import numpy as np
    import torch
    import multiprocessing.connection as connection
    from . import mutation_routed_full_panel_observations_v1 as observations
    from . import mutation_routed_tree_observations_v1 as old
    from .mutation_routed_full_panel_cache_v1 import write_full_cache_v1, read_full_cache_index_v1
    from .census_tsinfer_fit_v1 import ConsumerReaderV1
    from .mutation_routed_pyg_dataset_v1 import _worker_ipc_address_v1
    roles = observations.validate_registry_v1(registry)
    rows = [row for row in registry if row["split"] != "outer"]
    source_refs = {path: {"path": path, "sha256": spec["source_sha256"][path]} for path in CACHE_SOURCES}
    if directory.exists():
        completed = directory / "completed.json"
        if not completed.is_file():
            raise FileExistsError("Incomplete output preserved; a fresh admitted output is required")
        result = json.loads(completed.read_text())
        if (result.get("input_ref") != input_ref or result.get("targets_read") is not False
                or result.get("outer_included") is not False or result.get("CPU_workers") != 8):
            raise ValueError("Existing completion belongs to a different admission")
        entries = {}
        for row in rows:
            ready = _ready_entry(row, spec, config)
            if ready is None:
                raise ValueError("Completed cache registered parent is no longer ready")
            entries[row["parent_id"]] = ready[0]
        read_full_cache_index_v1(result["cache_manifest_ref"], ConsumerReaderV1(),
            registry=registry, entry_refs=entries, source_refs=source_refs)
        return dict(result, reused_completed_cache=True)
    adapter = importlib.import_module("arggnn.project_storage_cpu_closed_scopes_v1")
    # Resource cap may exceed the generation-only cap. No producer config edit.
    resource_config = dict(config, output_cap_bytes=spec["output_cap_bytes"])
    adapter.validate_storage_v9(spec, resource_config, spec["storage"], source_ref=spec["source_ref"],
                               config_ref=spec["config_ref"], runtime_ref=spec["runtime_ref"])
    expiry = datetime.fromisoformat(spec["expires_at_utc"].replace("Z", "+00:00"))
    guarded_bytes = 0
    def guard(planned=0):
        nonlocal guarded_bytes
        if datetime.now(timezone.utc) >= expiry:
            raise ValueError("CPU preparation admission expired")
        value = adapter.budget_check_v9(spec["storage"], boundary=True, required_write_bytes=int(planned))
        guarded_bytes += int(planned)
        return value
    output = _Output(directory, guard)
    output.save("started.json", dict(schema="expanded_pyg_observations_v1_started", input_ref=input_ref,
        source_ref=spec["source_ref"], registry_sha256=_digest(registry), started_utc=_utc(), CPU_workers=8))
    scratch = directory / "worker_tmp"
    guard(16 * 8192)
    scratch.mkdir()
    os.environ["TMPDIR"] = str(scratch); tempfile.tempdir = str(scratch)
    torch.set_num_threads(1)
    connection.arbitrary_address = _worker_ipc_address_v1
    started = time.monotonic()
    executor = ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context("spawn"),
                                   initializer=_worker_init, initargs=(str(scratch),))
    monitor = _Monitor(executor, spec["max_cpu_rss_bytes"])
    failed = True
    entries, loaded, stats, pending, submitted = {}, {}, {}, {}, set()
    cache_manifest = None
    def log_failure(error, kind="failed"):
        detail = dict(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
                      submitted_parent_ids=sorted(submitted), main_peak_rss_bytes=_peak_rss(),
                      owned_peak_rss_sum_bytes=monitor.peak_total)
        print(json.dumps(dict(utc=_utc(), event=kind, **detail)), file=sys.stderr, flush=True)
        try:
            output.event(kind, **detail)
            if not (directory / "failed.json").exists():
                output.save("failed.json", dict(schema="expanded_pyg_observations_v1_failure",
                                               input_ref=input_ref, **detail))
        except BaseException as logging_error:
            print("Failure receipt write rejected: " + str(logging_error), file=sys.stderr, flush=True)
    try:
        output.event("preload_started", ready_policy="registered_completed_observations_only", CPU_workers=8)
        deadline = started + spec["readiness_timeout_seconds"]
        next_poll = 0.0
        while len(loaded) < 40:
            if time.monotonic() >= deadline:
                raise TimeoutError("Registered inner observed parent readiness/load timeout")
            if time.monotonic() >= next_poll:
                for row in rows:
                    parent_id = row["parent_id"]
                    if parent_id in submitted:
                        continue
                    ready = _ready_entry(row, spec, config)
                    if ready is None:
                        continue
                    if len(pending) >= 8:
                        break
                    entry_ref, native_bytes = ready
                    guard(native_bytes + sum(value[1] for value in pending.values()) + 8192)
                    entries[parent_id] = entry_ref
                    future = executor.submit(_load, row, entry_ref, str(scratch))
                    pending[future] = (parent_id, native_bytes)
                    submitted.add(parent_id)
                    output.event("parent_load_submitted", parent_id=parent_id, entry_ref=entry_ref)
                next_poll = time.monotonic() + spec.get("poll_seconds", 2.0)
            done, _ = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
            for future in done:
                parent_id, _ = pending.pop(future)
                observed, detail = future.result()
                if observed.parent_id != parent_id or parent_id in loaded:
                    raise ValueError("Parent load identity or exactly-once contract differs")
                loaded[parent_id], stats[parent_id] = observed, detail
                output.event("parent_loaded", loaded_parent_count=len(loaded), **detail)
            if not pending:
                time.sleep(0.5)
        preload_seconds = time.monotonic() - started
        output.event("all_inner_parents_loaded", parent_count=40, train_count=32, val_count=8,
                     preload_elapsed_seconds=preload_seconds)
        fit_started = time.monotonic()
        duration, count = [], []
        for parent_id in roles["train"]:
            d, c = old.observed_sham_geometry_v1(loaded[parent_id].trees)
            duration.append(d); count.append(c)
        bins = old.fit_sham_bin_edges_v1(np.concatenate(duration), np.concatenate(count), split="TRAIN", bins=4)
        del duration, count, d, c
        bin_seconds = time.monotonic() - fit_started
        output.event("legacy_bins_fitted", TRAIN_parent_ids=roles["train"], legacy_metadata_only=True,
                     fit_seconds=bin_seconds, training_branch_count=bins.training_branch_count)
        prepare_started = time.monotonic()
        items, pending = {}, {}
        remaining = iter(row["parent_id"] for row in rows)
        def submit_prepare():
            parent_id = next(remaining, None)
            if parent_id is None:
                return False
            observed = loaded.pop(parent_id)
            pending[executor.submit(_prepare, observed, bins)] = parent_id
            return True
        for _ in range(8):
            submit_prepare()
        while pending:
            done, _ = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
            for future in done:
                parent_id = pending.pop(future)
                item, detail = future.result()
                items[parent_id] = item
                stats[parent_id].update(detail)
                output.event("parent_prepared", prepared_parent_count=len(items), **detail)
                submit_prepare()
        prepare_seconds = time.monotonic() - prepare_started
        if loaded or len(items) != 40 or len(submitted) != 40:
            raise ValueError("Complete exactly-once inner observation collection required")
        collection = dict(parents={row["parent_id"]: items[row["parent_id"]] for row in rows},
            registry=registry, entry_refs=entries, sham_bin_edges=asdict(bins),
            receipt=dict(schema="mutation_routed_full_panel_prepared_collection_v1", targets_read=False,
                observations_only=True, sham_bins_fit_from_parent_ids=roles["train"],
                original_native_holds_preserved=True, outer_included=False,
                elapsed_observed_preparation_seconds=time.monotonic() - started,
                placement_sham_computed=False, sham_bins_legacy_metadata_only=True))
        publication_started = time.monotonic()
        cache_directory = directory / "cache"
        publication_timings, publication_clock = [], [publication_started]
        def publication_guard(planned):
            parent_number = len(publication_timings)
            now = time.monotonic()
            detail = dict(parent_id=rows[parent_number]["parent_id"] if parent_number < 40 else None,
                artifact_kind="parent" if parent_number < 40 else "manifest", compressed_bytes=int(planned),
                boundary_interval_seconds=now - publication_clock[0],
                cumulative_publication_seconds=now - publication_started)
            publication_timings.append(detail)
            publication_clock[0] = now
            output.event("cache_publication_boundary", **detail)
            # The compressed-byte check follows the log write, so both growths
            # remain inside the cap before the unchanged writer publishes.
            guard(planned)
            return True
        cache_manifest = write_full_cache_v1(collection, cache_directory, source_refs=source_refs,
                                             guard_exact_bytes=publication_guard)
        publication_seconds = time.monotonic() - publication_started
        output.event("cache_published", cache_manifest_ref=cache_manifest,
                     publication_elapsed_seconds=publication_seconds,
                     compressed_bytes=sum(row["compressed_bytes"] for row in publication_timings))
        failed = False
    except BaseException as error:
        log_failure(error)
        raise
    finally:
        monitor.close()
        try:
            _drain(executor, failed)
            # Only owned empty directories are removed after tracked children exit.
            for process_directory in scratch.glob("worker_*"):
                if process_directory.is_dir() and not process_directory.is_symlink():
                    process_directory.rmdir()
            scratch.rmdir()
        except BaseException as error:
            log_failure(error, "worker_cleanup_failed")
            if not failed:
                raise
    result = dict(schema="expanded_pyg_observations_v1_completed", input_ref=input_ref,
        source_ref=spec["source_ref"], config_ref=spec["config_ref"], registration_ref=spec["registration_ref"],
        source_refs=source_refs, runtime_ref=spec["runtime_ref"], registry_sha256=_digest(registry),
        cache_manifest_ref=cache_manifest, completed_utc=_utc(), CPU_workers=8, parent_count=40,
        TRAIN_count=32, VAL_count=8, outer_included=False, targets_read=False, observations_only=True,
        inference_regenerated=False, GPU_used=False, owned_workers_drained=True,
        preload_seconds=preload_seconds, legacy_bin_fit_seconds=bin_seconds,
        prepare_seconds=prepare_seconds, publication_seconds=publication_seconds,
        total_elapsed_seconds=time.monotonic() - started, parent_stats=stats,
        publication_timings=publication_timings,
        publication_boundary_intervals_include_previous_write_and_current_serialization=True,
        compressed_bytes=sum(row["compressed_bytes"] for row in publication_timings),
        main_peak_rss_bytes=_peak_rss(), owned_peak_rss_sum_bytes=monitor.peak_total,
        worker_peak_rss_bytes=max([monitor.peak_worker] +
            [max(row["worker_peak_rss_bytes"], row["prepare_worker_peak_rss_bytes"]) for row in stats.values()]),
        RSS_sum_includes_shared_page_double_counting=True,
        guarded_planned_bytes=guarded_bytes, reused_completed_cache=False)
    output.event("completed", cache_manifest_ref=cache_manifest, parent_count=40, owned_workers_drained=True)
    output.save("completed.json", result)
    return result


def source_report_v1():
    return dict(status="source_only_no_preparation", schema=SCHEMA, source_ref=_ref(SOURCE),
        source_sha256={path: _sha(path) for path in SOURCE_PATHS},
        config_ref=_ref(CONFIG), registration_ref=_ref(REGISTRATION),
        producer_ref=_ref(PRODUCER), original_producer_ref=_ref(ORIGINAL_PRODUCER),
        storage_adapter_ref=_ref(STORAGE), CPU_workers=8, TRAIN_count=32, VAL_count=8,
        outer_included=False, targets_read=False, GPU_used=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--input")
    parser.add_argument("--expected-input-sha256")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(source_report_v1(), indent=2, sort_keys=True)); return
    if not args.input or not args.expected_input_sha256:
        parser.error("Execution requires root --input and --expected-input-sha256")
    input_ref = {"path": args.input, "sha256": args.expected_input_sha256}
    spec = _read(input_ref)
    result = execute_v1(spec, input_ref)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
