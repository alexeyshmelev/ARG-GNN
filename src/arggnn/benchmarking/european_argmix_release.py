"""Available-release ARGMix on the complete common AP200Mb C4 condition.

These are new length-aware I/O adapters. The author source, model, first-SNP
truth resolver, augmentation, chunk coverage, objective, validation selection
and inference routines remain unchanged. No stage runs at import.
"""
from __future__ import annotations

from contextlib import contextmanager

import importlib
import json
import os
from pathlib import Path
import sys
import time
import types
from . import european_native_models as m
from . import european_sampling as sampling
from .execution import GPU0_UUID, TITLE
from .. import official_argmix_biological as native
from .. import expanded_native_argmix_v1 as archives

CLASSES = ("Ana", "CHG", "WHG", "EHG")
NATIVE_CLASSES = ("ANA", "CHG", "WHG", "EHG")
LENGTH = 200_000_000
SCHEMA = "argmix_available_release_common_data_v2"


def config(record):
    value = m._read(record, observed=True)
    m._need(value.get("schema") == SCHEMA
        and value["native_model"] == native.NATIVE_MODEL
        and value["joint_model"] == native.JOINT_MODEL
        and value["source_seed"] == 50
        and value["class_order"] == list(NATIVE_CLASSES)
        and value["author_source_tree_sha256"] == native.SOURCE_TREE_SHA
        and value["published_ARGMix_paper_reproduction"] is False
        and value["processing"] == dict(exclude_edge_trees=True,
            subgraph_merge=dict(enabled=False, signature_mode="ids_only"))
        and value["all_native_training_chunks"] is True
        and value["inference_changes_native_edge_exclusion"] is False,
        "Exact current author defaults and truthful separate C4 condition required")
    scope = value["data_condition"]
    m._need(scope["length_bp"] == LENGTH and scope["haplotypes"] == 952
        and scope["query_haplotypes"] == 686
        and scope["reference_haplotypes"] == 266
        and len(scope["train_parent_ids"]) == 3
        and len(scope["test_parent_ids"]) == 5,
        "Complete common 200Mb sample and independent parent condition required")
    return value


def _roster(inference, *, max_metadata_bytes):
    from ..full_panel_native_baseline_contracts_v1 import validate_observed_roster
    rows = m._read(inference["observed_sample_manifest"], observed=True,
                   cap=max_metadata_bytes)
    checked = validate_observed_roster(rows, require_argmix_native_ids=True)
    return rows, checked


def prepare_frame(inference_ref, output_directory, *, guard, execution_check,
                  write_headroom_bytes, max_metadata_bytes):
    """Only sanitized observed TS, exact native IDs, original dates and SNPs."""
    m._ready(guard, execution_check, write_headroom_bytes)
    m._need(os.environ.get("CUDA_VISIBLE_DEVICES") == "",
            "Observed full-length frame stage is CPU-only")
    inference = m._read(inference_ref, observed=True, cap=max_metadata_bytes)
    m._need(inference.get("schema") == "AP_full200mb_native_inference_v1"
        and inference["status"] == "post_five_iterations_inferred"
        and inference["sequence_length_bp"] == LENGTH
        and inference["truth_or_oracle_read"] is False,
        "Actual complete native observed inference receipt required")
    import numpy as np
    import tskit
    rows, roster = _roster(inference, max_metadata_bytes=max_metadata_bytes)
    ts = tskit.load(m._checked(inference["inferred_tree"], observed=True))
    post = inference["post_five_iterations"]
    coverage = np.asarray(post["coverage_original_bp0"], dtype=np.float64)
    m._need(coverage.shape == (2,) and np.isfinite(coverage).all()
        and 0 <= coverage[0] < coverage[1] <= LENGTH
        and post["local_to_original_offset_bp"] == coverage[0]
        and post["normalization_length_bp"] == LENGTH
        and ts.sequence_length == coverage[1] - coverage[0]
        and np.array_equal(ts.samples(), np.arange(952))
        and np.allclose(ts.nodes_time[ts.samples()],
            [r["age_generations"] for r in rows], rtol=0., atol=1e-6),
        "Native IDs, dates and actual clipped coordinate identity required")
    tables = ts.tables
    m._need(not tables.metadata and not ts.num_individuals and not ts.num_populations
        and not ts.num_migrations and np.all(tables.nodes.population == -1)
        and np.all(tables.nodes.individual == -1)
        and not any(len(t.metadata) for t in
            (tables.nodes, tables.edges, tables.sites, tables.mutations,
             tables.individuals, tables.populations, tables.migrations)),
        "Observed tree must contain no private ancestry annotations")
    sites = np.asarray(tables.sites.position)
    m._need(len(sites) > 0 and np.array_equal(sites, sites.astype(np.int64))
        and (np.diff(sites) > 0).all() and (sites >= 0).all()
        and (sites < ts.sequence_length).all(),
        "Every retained native SNP must have an exact local integer coordinate")
    join = post["site_join"]
    m._need(len(join) == len(sites)
        and [r["native_site_id"] for r in join] == list(range(len(sites)))
        and np.array_equal([r["local_position_0based"] for r in join], sites)
        and np.array_equal([r["original_position_0based"] for r in join],
                           sites + coverage[0])
        and all(r["VCF_position_1based"] == r["original_position_0based"] + 1
                for r in join),
        "Exact VCF/local/original site join required without new sampling")
    output = m._new_output(output_directory)
    slot_rows, _ = archives.validate_slot_metadata_v1(rows)
    refs = "".join("%d %s %s\n" % row for row in slot_rows).encode()
    reference_ref = _write(output / "references.tsv", refs, guard)
    sample = "ID_1 ID_2 missing\n0 0 0\n" + "".join(
        "%s %s 0\n" % (rows[i]["sample_id"], rows[i]["sample_id"])
        for i in range(0, 952, 2))
    sample_ref = _write(output / "native_sample.sample", sample.encode(), guard)
    coords = m._npz(output / "observed_coordinates.npz", dict(
        local_site_positions_bp=sites.astype(np.int64),
        original_site_positions_bp=sites + coverage[0],
        query_nodes=np.asarray(roster.query_nodes, dtype=np.int32)), guard)
    roster_ref = m._json(output / "observed_roster.json", rows, guard)
    map_ref = _write(output / "local_uniform_map.tsv",
                    b"position Genetic_Map(cM)\n0 0\n200000000 200\n", guard)
    return m._json(output / "frame.json", dict(
        schema="european_release_argmix_observed_frame_v1",
        parent_id=inference["parent_id"], split=inference["split"],
        inference_ref=inference_ref, tree=inference["inferred_tree"],
        references=reference_ref, native_sample=sample_ref, coordinates=coords,
        roster=roster_ref, local_map=map_ref,
        local_offset_bp=float(coverage[0]), coverage_original_bp0=coverage.tolist(),
        normalization_length_bp=LENGTH, native_tree_count=ts.num_trees,
        samples=952, queries=686, pure_refs=266,
        stock_native_ID_correspondence_verified=True, truth_opened=False), guard)


def prepare_c4_registry(plan_ref, inference_ref, output_directory, *,
                        guard, execution_check, write_headroom_bytes,
                        max_metadata_bytes):
    """Freeze common C4 SNP support using native edge eligibility before truth."""
    m._ready(guard, execution_check, write_headroom_bytes)
    import copy
    import numpy as np
    import tskit
    plan = m._read(plan_ref, observed=True)
    inference = m._read(inference_ref, observed=True, cap=max_metadata_bytes)
    m._need(inference["truth_or_oracle_read"] is False
        and inference["parent_id"] in plan["parents"],
        "Actual observed inference on registered independent parent required")
    ts = tskit.load(m._checked(inference["inferred_tree"], observed=True))
    m._need(ts.num_trees >= 3, "Native default requires at least one interior tree")
    left = float(ts.at_index(0).interval.right)
    right = float(ts.at_index(ts.num_trees - 1).interval.left)
    m._need(0 < left < right < ts.sequence_length,
            "Genuine first/last tree exclusion must leave interior support")
    rows = m._read(inference["observed_sample_manifest"], observed=True,
                   cap=max_metadata_bytes)
    original = inference["post_five_iterations"]["site_join"]
    eligible = [r for r in original if left <= r["local_position_0based"] < right]
    m._need(eligible, "Observed interior tree support contains no retained SNPs")
    metadata = copy.deepcopy(inference["post_five_iterations"])
    metadata["site_join"] = [dict(row, native_site_id=i) for i,row in enumerate(eligible)]
    registry = sampling.build_sampled_registry(plan, inference["parent_id"], rows,
        metadata, observation_ref=inference_ref)
    for record in registry["records"]:
        record["native_site_id"] = eligible[record["native_site_id"]]["native_site_id"]
    offset = inference["post_five_iterations"]["local_to_original_offset_bp"]
    registry["sampling_policy"]["site_policy"] = (
        "midpoints of equal bins in retained SNP rank within native interior trees")
    registry["native_C4_eligible_support"] = dict(
        native_exclude_edge_trees=True, definition="first and last marginal tree omitted",
        eligible_interval_local_bp0=[left,right],
        eligible_interval_original_bp0=[left+offset,right+offset],
        native_trees=ts.num_trees, eligible_trees=ts.num_trees-2,
        total_retained_SNPs=len(original), eligible_retained_SNPs=len(eligible),
        excluded_edge_SNPs=len(original)-len(eligible),
        excluded_edge_bp=ts.sequence_length-(right-left),
        eligibility_uses_predictions_or_truth=False,
        source_ref=m._ref(native.AUTHOR/"src/rust/ancpath/src/fast_lca/engine.rs"),
        support_frozen_before_C4_supervision_or_prediction=True,
        paper_native6_endpoint_unchanged=True)
    output = m._new_output(output_directory)
    ref = m._json(output/"registry.json", registry, guard)
    array_ref = m._npz(output/"registry_arrays.npz",
                      sampling.registry_arrays(registry),guard)
    return dict(registry_ref=ref, arrays_ref=array_ref,
        parent_id=registry["parent_id"], examples=registry["examples"],
        coverage=registry["native_C4_eligible_support"])


def _write(path, payload, guard):
    guard(len(payload) + 8192)
    with path.open("xb") as stream:
        stream.write(payload)
    guard(0)
    return m._ref(path)


def _child_startup_source():
    """Python startup IO binding; original source files and argv remain exact."""
    return f"""import os, sys, traceback
def _arggnn_native_startup():
    import setproctitle
    setproctitle.setproctitle({TITLE!r})
    name = os.path.basename(sys.argv[0])
    if name not in ("export_torchscript.py", "run_inference.py", "run_joint_inference.py"):
        return
    if os.path.realpath(sys.argv[0]) != os.path.join({str(native.SOURCE / "src")!r}, name):
        raise RuntimeError("Native child script must retain its admitted source origin")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != {GPU0_UUID!r} or os.environ.get("ARGGNN_PHYSICAL_GPU") != "0":
        raise RuntimeError("Native child must inherit admitted physical GPU0")
    cap = int(os.environ.get("ARGGNN_MAX_CUDA_RESERVED_BYTES", "0"))
    if not 0 < cap <= 74 * 2**30:
        raise RuntimeError("Native child needs a finite admitted CUDA cap")
    device = sys.argv[sys.argv.index("--device") + 1]
    if name == "export_torchscript.py":
        if device != "cpu":
            raise RuntimeError("Native TorchScript export must stay CPU-only")
        return
    if device != "cuda":
        raise RuntimeError("Native GPU inference must retain its declared device")
    import torch
    if torch.cuda.device_count() != 1:
        raise RuntimeError("One inherited admitted CUDA device required")
    total = torch.cuda.get_device_properties(0).total_memory
    if cap > total:
        raise RuntimeError("Native child CUDA cap exceeds physical memory")
    torch.cuda.set_per_process_memory_fraction(cap / total, 0)
try:
    _arggnn_native_startup()
except BaseException:
    traceback.print_exc(file=sys.stderr)
    os._exit(78)
""".encode()


@contextmanager
def _native_python_startup(output, guard):
    """Only this fresh stage's children receive the bounded startup hook."""
    directory = output / "python_startup"
    directory.mkdir()
    ref = _write(directory / "sitecustomize.py", _child_startup_source(), guard)
    previous = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = str(directory) + (os.pathsep + previous if previous else "")
    try:
        yield ref
    finally:
        if previous is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = previous


@contextmanager
def _native_local_runner(output, bounded_runner, execution_check):
    """Private command IO binding; child argv and native command bytecode stay exact."""
    m._need(callable(bounded_runner), "ROOT-owned inherited-group native runner required")
    module = importlib.import_module("commands.infer")
    m._need(Path(module.__file__).resolve() == native.SOURCE / "src/commands/infer.py",
            "Only admitted unchanged native inference module may be bound")
    original = module.cmd_infer
    waits = []
    def run_local(command, *, cwd=native.SOURCE):
        execution_check()
        wait = bounded_runner(command, cwd=cwd, env={}, limits={},
            log_path=output / ("native_child_%02d.log" % (len(waits) + 1)))
        m._need(wait.get("actual_direct_wait") is True
            and wait.get("actual_direct_returncode") == 0
            and wait.get("native_process_absent") is True
            and wait.get("outer_owned_group_cleanup_pending") is True,
            "Actual successful inherited-group native child wait required")
        waits.append(dict(argv=list(command), cwd=str(cwd), actual_wait=wait))
        execution_check()
    bound = types.FunctionType(original.__code__,
        dict(original.__globals__, _run_local=run_local), original.__name__,
        original.__defaults__, original.__closure__)
    m._need(bound.__code__ is original.__code__,
            "Original native cmd_infer code object must remain unchanged")
    module.cmd_infer = bound
    try:
        yield waits
    finally:
        module.cmd_infer = original


def _frame(record, *, max_metadata_bytes):
    result = m._read(record, observed=True, cap=max_metadata_bytes)
    m._need(result.get("schema") == "european_release_argmix_observed_frame_v1"
        and result["normalization_length_bp"] == LENGTH
        and result["samples"] == 952 and result["queries"] == 686
        and result["pure_refs"] == 266 and result["truth_opened"] is False
        and result["stock_native_ID_correspondence_verified"] is True,
        "Complete genuine current-condition observed frame required")
    for key in ("tree", "references", "native_sample", "coordinates", "roster", "local_map"):
        m._checked(result[key], observed=True)
    return result


def _truth(registry_or_frame, generation_ref, target_schema_ref, truth_ref, *,
           allowed_splits, max_truth_uncompressed_bytes):
    """Only TRAIN or declared independent VAL may reach private paths here."""
    parent, split = registry_or_frame["parent_id"], registry_or_frame["split"]
    m._need(split in allowed_splits,
            "Only declared TRAIN/independent validation supervision may open private targets")
    generated, schema = m._read(generation_ref), m._read(target_schema_ref)
    m._need(generated["parent_id"] == schema["parent_id"] == parent
        and generated["split"] == schema["split"] == split
        and generated["files"]["truth/full_truth_intervals.npz"]["sha256"] == truth_ref["sha256"]
        and generated["files"]["truth/target_schema.json"]["sha256"] == target_schema_ref["sha256"],
        "Actual same-parent producer-authenticated original targets required")
    base = m._checked(generation_ref).parent
    m._need(m._path(truth_ref["path"]) == base / "truth/full_truth_intervals.npz"
        and m._path(target_schema_ref["path"]) == base / "truth/target_schema.json",
        "Exact same-parent private target namespace required")
    truth = m._arrays(truth_ref, cap=max_truth_uncompressed_bytes)
    return schema, truth


def sampled_source_labels(registry, schema, truth):
    """Exact ground-truth ontology conversion, never a prediction-head collapse."""
    import numpy as np
    native_labels = sampling.native_truth_labels(registry, schema, truth)
    source = np.asarray(truth["source_classes_0based"])
    intervals = np.asarray(truth["intervals_bp"])
    raw = np.asarray(truth["query_raw_node_ids"])
    m._need(source.dtype.kind in "iu" and source.shape == (len(intervals), 686)
        and ((source >= 0) & (source < 4)).all()
        and schema["native_to_C4"] == [0, 1, 2, 3, 0, 1],
        "Original direct source truth and exact path/source ontology required")
    by_sample = {int(sample): col for col, sample in enumerate(raw)}
    result = np.empty(registry["examples"], dtype=np.int64)
    for record in registry["records"]:
        row = int(np.searchsorted(intervals[:, 1],
                                 record["position_original_bp"], side="right"))
        result[record["example_index"]] = source[row, by_sample[record["sample_index"]]]
    m._need(np.array_equal(result, np.asarray([0, 1, 2, 3, 0, 1])[native_labels]),
            "Direct source truth must equal the documented census-path ontology")
    return result


def prepare_sampled_c4_targets(registry_ref, generation_ref, target_schema_ref,
                              truth_ref, output_directory, *, guard, execution_check,
                              write_headroom_bytes, max_truth_uncompressed_bytes):
    m._ready(guard, execution_check, write_headroom_bytes)
    registry = m._read(registry_ref, observed=True)
    m._need(registry.get("schema") == "ap_paper_sampled_registry_v1"
        and registry.get("labels_present") is False and registry.get("split") == "train",
        "Only the original frozen observed TRAIN sampled registry is allowed")
    schema, truth = _truth(registry, generation_ref, target_schema_ref, truth_ref,
        allowed_splits={"train"}, max_truth_uncompressed_bytes=max_truth_uncompressed_bytes)
    labels = sampled_source_labels(registry, schema, truth)
    output = m._new_output(output_directory)
    arrays_ref = m._npz(output / "source_targets.npz",
                       dict(source_labels_0based=labels), guard)
    return m._json(output / "targets.json", dict(
        schema="european_release_C4_sampled_targets_v1",
        parent_id=registry["parent_id"], split="train", registry_ref=registry_ref,
        generation_ref=generation_ref, target_schema_ref=target_schema_ref,
        truth_ref=truth_ref, arrays_ref=arrays_ref, rows=len(labels),
        source_class_order=list(CLASSES), no_label_dependent_selection=True,
        direct_source_truth_verified=True,
        six_head_predictions_or_weights_converted=False), guard)


def native_snp_truth(intervals, labels, actual_nodes, queries, local_sites,
                     original_sites, *, max_snp_truth_bytes):
    """Pure complete same-SNP C4 join for the unchanged Rust first-SNP resolver."""
    import numpy as np
    raw = np.asarray([r["source_observed_node_id"] for r in queries])
    m._need(intervals.dtype == np.float64 and intervals.ndim == 2
        and intervals.shape[1:] == (2,) and len(intervals) > 0
        and np.isfinite(intervals).all() and intervals[0, 0] == 0
        and intervals[-1, 1] == LENGTH
        and np.array_equal(intervals[:-1, 1], intervals[1:, 0])
        and (intervals[:, 1] > intervals[:, 0]).all()
        and labels.shape == (len(intervals), 686) and labels.dtype.kind in "iu"
        and ((labels >= 0) & (labels < 4)).all()
        and np.array_equal(actual_nodes, raw) and len(queries) == 686
        and all(r["role"] == "query" for r in queries),
        "Complete original 200Mb C4 truth and actual biological query order required")
    m._need(local_sites.ndim == original_sites.ndim == 1
        and len(local_sites) == len(original_sites) > 0
        and np.array_equal(local_sites, local_sites.astype(np.int64))
        and (np.diff(local_sites) > 0).all()
        and np.isfinite(original_sites).all()
        and (np.diff(original_sites) > 0).all()
        and (original_sites >= 0).all() and (original_sites < LENGTH).all()
        and type(max_snp_truth_bytes) is int
        and len(local_sites) * 686 * 2 <= max_snp_truth_bytes,
        "Every observed SNP must fit the declared supervision cap without cropping")
    indexes = np.searchsorted(intervals[:, 1], original_sites, side="right")
    out = np.empty((len(local_sites), 686), dtype=np.int16)
    for first in range(0, len(local_sites), 4096):
        out[first:first + 4096] = labels[indexes[first:first + 4096]]
    return dict(variant_pos=local_sites.astype(np.int64), truth__hap_labels=out,
        truth__sample_ids=np.asarray([r["sample_id"] for r in queries[::2]], dtype=str),
        ancestry_names=np.asarray(NATIVE_CLASSES, dtype=str))


def prepare_native_targets(frame_ref, generation_ref, target_schema_ref, truth_ref,
                           output_directory, *, guard, execution_check,
                           write_headroom_bytes, max_metadata_bytes,
                           max_truth_uncompressed_bytes, max_snp_truth_bytes):
    m._ready(guard, execution_check, write_headroom_bytes)
    frame = _frame(frame_ref, max_metadata_bytes=max_metadata_bytes)
    schema, truth = _truth(frame, generation_ref, target_schema_ref, truth_ref,
        allowed_splits={"train", "validation"},
        max_truth_uncompressed_bytes=max_truth_uncompressed_bytes)
    import numpy as np
    rows = m._read(frame["roster"], observed=True)
    queries = [r for r in rows if r["role"] == "query"]
    m._need(schema["query_keys"] == [[r["sample_id"], r["haplotype"]] for r in queries]
        and schema["query_sample_indices"] == [r["source_observed_node_id"] for r in queries],
        "Native true source columns require the exact semantic diploid/haplotype join")
    coords = m._arrays(frame["coordinates"], observed=True, cap=max_metadata_bytes)
    guard(max_snp_truth_bytes + write_headroom_bytes)
    arrays = native_snp_truth(truth["intervals_bp"], truth["source_classes_0based"],
        truth["query_raw_node_ids"], queries, coords["local_site_positions_bp"],
        coords["original_site_positions_bp"], max_snp_truth_bytes=max_snp_truth_bytes)
    output = m._new_output(output_directory)
    native_ref = m._npz(output / "native_truth.npz", arrays, guard)
    return m._json(output / "targets.json", dict(
        schema="european_release_argmix_native_targets_v1",
        parent_id=frame["parent_id"], split=frame["split"], frame_ref=frame_ref,
        native_truth_ref=native_ref, source_class_order=list(NATIVE_CLASSES),
        generation_ref=generation_ref, target_schema_ref=target_schema_ref,
        truth_ref=truth_ref, every_observed_SNP_retained=True,
        native_label_resolver="first_snp", label_purity_filter=None), guard)


def _backend(cfg, output, frames, pairs, guard):
    import yaml
    m._need(frames and all(f["local_map"]["sha256"] == frames[0]["local_map"]["sha256"]
                         for f in frames), "Same full-length map required")
    _write(output / "local_uniform_map.tsv",
           m._checked(frames[0]["local_map"], observed=True).read_bytes(), guard)
    pair_paths = {}
    for split in ("train", "val"):
        path = output / (split + "_pairs.yaml")
        _write(path, yaml.safe_dump(pairs[split], sort_keys=False).encode(), guard)
        pair_paths[split] = path
    backend = native.backend_config(cfg, output, pair_paths["train"], pair_paths["val"])
    # This pointer is used only by stock startup counting/logging. Every native
    # pair retains that parent's actual full observed reference metadata.
    backend["paths"]["references"] = str(m._checked(frames[0]["references"], observed=True))
    path = output / "backend.yaml"
    _write(path, yaml.safe_dump(backend, sort_keys=False).encode(), guard)
    return m._ref(path)


def _pair(frame, target):
    m._need(target["parent_id"] == frame["parent_id"]
        and target["split"] == frame["split"]
        and target["schema"] == "european_release_argmix_native_targets_v1",
        "Same-parent native training archive inputs required")
    return dict(rel=str(m._checked(frame["tree"], observed=True)),
        references=str(m._checked(frame["references"], observed=True)),
        relate_sample=str(m._checked(frame["native_sample"], observed=True)),
        truth_npz=str(m._checked(target["native_truth_ref"])))


def admit_native(build_ref, cpu_ref, *, require_gpu=False):
    """Reuse the existing exact source/build/library admission, no installations."""
    native.runtime_admission(build_ref, cpu_ref, {})
    result = native.source_imports()
    if require_gpu:
        import setproctitle
        m._need(setproctitle.getproctitle() == "AI Auto Research Pipeline Test",
                "Author code must run inside the correctly titled GPU worker")
        m._need(os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "")
                and result[0].cuda.device_count() == 1,
                "ROOT-owned single physical GPU0 admission required")
        cap = int(os.environ.get("ARGGNN_MAX_CUDA_RESERVED_BYTES", "0"))
        total = result[0].cuda.get_device_properties(0).total_memory
        m._need(0 < cap <= min(74 * 2**30, total),
                "Finite ROOT-admitted CUDA allocator cap required")
        result[0].cuda.set_per_process_memory_fraction(cap / total, 0)
    else:
        m._need(os.environ.get("CUDA_VISIBLE_DEVICES") == "",
                "Native archive construction exposes no CUDA device")
    return result


def build_archive(config_ref, frame_ref, targets_ref, output_directory, *,
                  build_ref, cpu_ref, bounded_runner, command_limits,
                  max_archive_bytes, guard, execution_check,
                  write_headroom_bytes, max_metadata_bytes):
    m._ready(guard, execution_check, write_headroom_bytes)
    cfg = config(config_ref)
    frame = _frame(frame_ref, max_metadata_bytes=max_metadata_bytes)
    m._need(frame["split"] in ("train", "validation"),
            "Only TRAIN/independent VAL native archives allowed")
    target = m._read(targets_ref)
    m._need(target["frame_ref"] == frame_ref, "Exact native frame target binding required")
    pair = _pair(frame, target)
    admit_native(build_ref, cpu_ref)
    output = m._new_output(output_directory)
    split = "train" if frame["split"] == "train" else "val"
    pairs = {s: [pair] if s == split else [] for s in ("train", "val")}
    guard(write_headroom_bytes)
    backend_ref = _backend(cfg, output, [frame], pairs, guard)
    archive = output / (split + ".pt")
    m._need(type(max_archive_bytes) is int and max_archive_bytes > 0,
            "Explicit fail-stop full native archive output bound required")
    # A conservative bound from actual observed tree count is available now.
    # Reject before native allocation rather than crop the default corpus.
    import math
    chunk_upper = 686 * (math.ceil(frame["native_tree_count"] / 512) + 1)
    archive_upper = chunk_upper * 1024 * (16*8 + 16*16*2 + 8 + 1 + 2*4) + 8_000_000
    m._need(archive_upper <= max_archive_bytes,
            "Complete native observed-tree-derived archive bound exceeds caller cap: "
            + str(archive_upper) + " > " + str(max_archive_bytes))
    guard(archive_upper)
    argv = [str(native.CLI), "--config", str(m._checked(backend_ref)),
            "--split", split, "--output", str(archive)]
    result = bounded_runner(argv, cwd=native.SOURCE,
        env=dict(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"),
        limits=dict(command_limits), log_path=output / "native_Rust.log")
    execution_check()
    m._need(result.get("actual_direct_wait") is True
        and result.get("native_process_absent") is True
        and result.get("actual_direct_returncode") == 0
        and archive.is_file() and archive.stat().st_size <= max_archive_bytes,
        "Actual successful bounded unchanged native Rust wait required")
    layout = archives.archive_layout_v1(archive)
    m._need(layout["metadata"][1] == (0 if split == "train" else 1),
            "Actual native training/validation split metadata required")
    guard(0)
    return m._json(output / "archive.json", dict(
        schema="european_release_argmix_archive_v1", parent_id=frame["parent_id"],
        split=frame["split"], frame_ref=frame_ref, targets_ref=targets_ref,
        backend_ref=backend_ref, archive_ref=m._ref(archive), archive_layout=layout,
        observed_tree_derived_archive_upper_bytes=archive_upper,
        pair=pair, command=dict(argv=argv, actual_wait=result,
            binary_sha256=m._sha(native.CLI)), config_ref=config_ref), guard)


def fit_release(config_ref, archive_products, output_directory, *, build_ref, cpu_ref,
                guard, execution_check,
                write_headroom_bytes, max_metadata_bytes):
    m._ready(guard, execution_check, write_headroom_bytes)
    cfg = config(config_ref)
    scope = cfg["data_condition"]
    train_ids, val_id = scope["train_parent_ids"], scope["validation_parent"]["parent_id"]
    m._need(set(archive_products) == set(train_ids + [val_id]),
            "All three complete TRAIN plus independent VAL archives required")
    modules = admit_native(build_ref, cpu_ref, require_gpu=True)
    products, frames, pairs = {}, [], {"train": [], "val": []}
    for parent in train_ids + [val_id]:
        product = m._read(archive_products[parent])
        expected_split = "train" if parent in train_ids else "validation"
        m._need(product["schema"] == "european_release_argmix_archive_v1"
            and product["parent_id"] == parent and product["split"] == expected_split
            and product["config_ref"] == config_ref,
            "Exact same-condition native complete parent archive required")
        frame = _frame(product["frame_ref"], max_metadata_bytes=max_metadata_bytes)
        target = m._read(product["targets_ref"])
        m._need(product["pair"] == _pair(frame, target),
                "Native pairs must retain original parent references/sample/truth")
        products[parent] = product
        frames.append(frame)
        pairs["train" if expected_split == "train" else "val"].append(product["pair"])
    output = m._new_output(output_directory)
    guard(write_headroom_bytes)
    backend_ref = _backend(cfg, output, frames, pairs, guard)
    from . import european_argmix_shards as shards
    # Original ZIP_STORED members are mapped in place. The logical corpus keeps
    # full global chunk order and stock random shuffling without a duplicate.
    train_refs = [products[p]["archive_ref"] for p in train_ids]
    logical_train = shards.write_manifest(output/"logical_train.json",train_ids,
        train_refs,split_code=0,guard=guard)
    logical_val = shards.write_manifest(output/"logical_val.json",[val_id],
        [products[val_id]["archive_ref"]],split_code=1,guard=guard)
    checkpoint_pointer = output / "trained_checkpoint.path"
    shards.run_original_main(modules[3], [
        "--config", str(m._checked(backend_ref)),
        "--train-pt", str(m._checked(logical_train)),
        "--val-pt", str(m._checked(logical_val)),
        "--checkpoint-output", str(checkpoint_pointer)], output,
        guard=guard,execution_check=execution_check)
    execution_check()
    m._need(checkpoint_pointer.is_file() and checkpoint_pointer.stat().st_size < 8192,
            "Actual native reported checkpoint pointer required")
    checkpoint = m._path(checkpoint_pointer.read_text().strip())
    m._need(checkpoint.is_relative_to(output / "checkpoints") and checkpoint.is_file()
        and not checkpoint.is_symlink(), "Native selected checkpoint must remain in current fit scope")
    guard(0)
    return m._json(output / "fit.json", dict(
        schema="european_release_argmix_fit_v1", config_ref=config_ref,
        training_parent_ids=train_ids, validation_parent_id=val_id,
        archive_products=archive_products, backend_ref=backend_ref,
        logical_train_manifest_ref=logical_train, logical_validation_manifest_ref=logical_val,
        original_native_archives_mapped_in_place=True, merged_archive_written=False,
        checkpoint_ref=m._ref(checkpoint),
        checkpoint_pointer_ref=m._ref(checkpoint_pointer),
        author_commit=cfg["author_commit"], author_source_tree_sha256=native.SOURCE_TREE_SHA,
        stock_author_training_main_bytecode_unchanged=True, training_seed=50,
        archive_IO_adapter="logical concatenation of memory mapped original stored members",
        checkpoint_selection="stock minimum validation loss, patience2, at most20epochs",
        all_native_training_chunks=True, test_truth_read=False,
        published_ARGMix_paper_reproduction=False), guard)


def infer_release(config_ref, fit_ref, frame_ref, output_directory, *,
                  build_ref, cpu_ref, bounded_runner, guard, execution_check,
                  write_headroom_bytes, max_metadata_bytes):
    m._ready(guard, execution_check, write_headroom_bytes)
    cfg = config(config_ref)
    fit, frame = m._read(fit_ref, observed=True), _frame(
        frame_ref, max_metadata_bytes=max_metadata_bytes)
    m._need(fit["schema"] == "european_release_argmix_fit_v1"
        and fit["config_ref"] == config_ref
        and frame["split"] == "test"
        and frame["parent_id"] in cfg["data_condition"]["test_parent_ids"],
        "Frozen native fit and declared independent TEST observed frame required")
    admit_native(build_ref, cpu_ref, require_gpu=True)
    output = m._new_output(output_directory)
    guard(write_headroom_bytes)
    backend_ref = _backend(cfg, output, [frame], {"train": [], "val": []}, guard)
    argv = ["infer", "--trees", str(m._checked(frame["tree"], observed=True)),
        "--references", str(m._checked(frame["references"], observed=True)),
        "--recomb-map", str(m._checked(frame["local_map"], observed=True)),
        "--checkpoint", str(m._checked(fit["checkpoint_ref"], observed=True)),
        "--output-dir", str(output), "--predict-pops", "QUERY",
        "--relate-sample", str(m._checked(frame["native_sample"], observed=True)),
        "--config", str(m._checked(backend_ref, observed=True)),
        "--output-prefix", "lai", "--device", "cuda"]
    m._need("--truth-npz" not in argv, "No truth may enter native inference")
    with _native_python_startup(output, guard) as child_startup_ref, \
         _native_local_runner(output, bounded_runner, execution_check) as child_waits:
        native.run_author("argmix_cli.py", argv, output)
    execution_check()
    h5 = output / "lai.h5"
    m._need(h5.is_file(), "Genuine author joint interval prediction H5 required")
    guard(0)
    return m._json(output / "inference.json", dict(
        schema="european_release_argmix_interval_predictions_v1",
        parent_id=frame["parent_id"], split="test", frame_ref=frame_ref,
        fit_ref=fit_ref, config_ref=config_ref, h5_ref=m._ref(h5),
        backend_ref=backend_ref, native_edge_exclusion_preserved=True,
        child_startup_ref=child_startup_ref,
        child_resource_binding="stage-local sitecustomize; title and admitted CUDA allocator before source main",
        native_subprocess_argv_and_source_unchanged=True,
        explicit_CPU_export_initializes_CUDA=False,
        parent_PYTHONPATH_restored_after_native_CLI=True,
        native_child_waits=child_waits,
        native_children_inherit_owned_process_group=True,
        native_cmd_infer_main_code_object_unchanged=True,
        source_class_order=list(CLASSES), test_truth_read=False), guard)


def match_sampled_intervals(registry, chunks):
    """Stream author rows onto every frozen focal SNP, without denominator edits."""
    import numpy as np
    canonical = sorted(registry["records"], key=lambda r: r["example_index"])
    n = registry["examples"]
    m._need([r["example_index"] for r in canonical] == list(range(n)),
            "Complete canonical sampled record indices required")
    by_node = {}
    for record in canonical:
        by_node.setdefault(record["query_node_id"], []).append(
            (record["position_local_bp"], record["example_index"]))
    by_node = {node: (np.asarray([p for p, _ in sorted(rows)]),
                     np.asarray([i for _, i in sorted(rows)], dtype=np.int64))
               for node, rows in by_node.items()}
    out, seen = np.empty((n, 4), dtype=np.float32), np.zeros(n, dtype=bool)
    for nodes, starts, ends, probs in chunks:
        m._need(nodes.ndim == starts.ndim == ends.ndim == 1
            and starts.shape == ends.shape == nodes.shape
            and probs.shape == (len(nodes), 4)
            and np.isfinite(starts).all() and np.isfinite(ends).all()
            and (ends > starts).all() and np.isfinite(probs).all()
            and ((probs >= 0) & (probs <= 1)).all()
            and np.allclose(probs.sum(1), 1., atol=1e-6, rtol=1e-5),
            "Raw valid original author half-open interval probabilities required")
        for node in np.unique(nodes):
            focal = by_node.get(int(node))
            if focal is None:
                continue
            positions, indices = focal
            selected = np.flatnonzero(nodes == node)
            # Only the small set of focal positions is queried in each row.
            left = np.searchsorted(positions, starts[selected], side="left")
            right = np.searchsorted(positions, ends[selected], side="left")
            for row, a, b in zip(selected, left, right):
                ids = indices[a:b]
                m._need(not seen[ids].any(),
                        "Ambiguous repeated prediction support cannot be averaged")
                out[ids] = probs[row]
                seen[ids] = True
    m._need(bool(seen.all()),
            "Native default lacks required common sampled support, no denominator reduction allowed")
    return out


def seal_sampled_predictions(interval_predictions_ref, registry_ref, output_directory, *,
                             guard, execution_check, write_headroom_bytes,
                             max_h5_chunk_rows):
    m._ready(guard, execution_check, write_headroom_bytes)
    product, registry = m._read(interval_predictions_ref, observed=True), m._read(
        registry_ref, observed=True)
    m._need(product["schema"] == "european_release_argmix_interval_predictions_v1"
        and registry["schema"] == "ap_paper_sampled_registry_v1"
        and product["parent_id"] == registry["parent_id"]
        and product["split"] == registry["split"] == "test"
        and registry["labels_present"] is False
        and type(max_h5_chunk_rows) is int and 0 < max_h5_chunk_rows <= 262144,
        "Same-parent frozen observed TEST support and bounded H5 chunks required")
    import h5py
    import numpy as np
    with h5py.File(m._checked(product["h5_ref"], observed=True), "r") as h5:
        keys = ("node_id", "tree_start", "tree_end", "refined_probs")
        m._need(all(k in h5 for k in keys)
            and len({len(h5[k]) for k in keys}) == 1,
            "Original author joint H5 interval schema required")
        def chunks():
            for start in range(0, len(h5["node_id"]), max_h5_chunk_rows):
                execution_check()
                guard(0)
                yield tuple(h5[k][start:start + max_h5_chunk_rows] for k in keys)
        probabilities = match_sampled_intervals(registry, chunks())
    output = m._new_output(output_directory)
    arrays_ref = m._npz(output / "predictions.npz", dict(
        example_indices=np.arange(registry["examples"], dtype=np.int64),
        probabilities=probabilities), guard)
    return m._json(output / "predictions.json", dict(
        schema="european_release_C4_predictions_v1", method="argmix",
        parent_id=registry["parent_id"], split="test",
        registry_ref=registry_ref, rows=registry["examples"],
        predictions_ref=arrays_ref, source_class_order=list(CLASSES),
        interval_predictions_ref=interval_predictions_ref,
        no_support_intersection_or_extrapolation=True,
        native_edge_exclusion_preserved=True, test_truth_read=False), guard)


def c4_metrics(probabilities, labels):
    import numpy as np
    p, y = np.asarray(probabilities), np.asarray(labels)
    m._need(p.shape == (len(y), 4) and len(y) > 0
        and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        and np.allclose(p.sum(1), 1., atol=1e-6, rtol=1e-5)
        and y.dtype.kind in "iu" and ((y >= 0) & (y < 4)).all(),
        "Complete actual C4 probabilities and labels required")
    calls = p.argmax(1)
    confusion = np.bincount(y * 4 + calls, minlength=16).reshape(4, 4)
    per_class = {}
    for i, name in enumerate(CLASSES):
        tp, actual, called = int(confusion[i,i]), int(confusion[i].sum()), int(confusion[:,i].sum())
        per_class[name] = dict(precision=tp/called if called else 0.,
            recall=tp/actual if actual else 0.,
            F1=2*tp/(actual+called) if actual+called else 0.,
            actual_count=actual, predicted_count=called)
    true_p = p[np.arange(len(y)), y]
    zeros = int((true_p == 0).sum())
    return dict(accuracy=float((calls == y).mean()),
        macro_F1=float(np.mean([v["F1"] for v in per_class.values()])),
        per_class=per_class, confusion=confusion.tolist(),
        confusion_convention="truth rows, prediction columns", records=len(y),
        raw_NLL=None if zeros else float(-np.log(true_p).mean()),
        raw_NLL_is_infinite=bool(zeros), zero_probability_truth_count=zeros,
        Brier=float(((p - np.eye(4)[y])**2).sum(1).mean()),
        probabilities_clipped_or_renormalized=False)


def verify_cross_endpoint_barrier(plan, native6_predictions, c4_predictions):
    """Validate ALL declared predictions before either endpoint opens TEST truth."""
    import numpy as np
    parents = plan["test_parent_ids"]
    m._need(len(parents) == 5
        and set(native6_predictions) == {"ancestralpaths", "treeloom"}
        and set(c4_predictions) == {"argmix", "treeloom"}
        and all(set(v) == set(parents) for group in
                (native6_predictions, c4_predictions) for v in group.values()),
        "Every registered method, endpoint and all five TEST parents must seal first")
    for parent in parents:
        for group, classes in ((native6_predictions, sampling.NATIVE_PATHS),
                               (c4_predictions, CLASSES)):
            registry_ref = None
            for method, mapping in group.items():
                value = m._read(mapping[parent], observed=True)
                current = value["registry_ref"]
                m._need(value["parent_id"] == parent and value["split"] == "test"
                    and value["rows"] == 10000
                    and (registry_ref is None or current == registry_ref),
                    "Each endpoint must preserve identical canonical TEST sample/site support")
                registry_ref = current
                actual_classes = value.get("source_class_order", value.get("native_path_names"))
                m._need(actual_classes == list(classes), "Declared actual prediction endpoint required")
                arrays = m._arrays(value["predictions_ref"], observed=True, cap=8*2**20)
                p = arrays["probabilities"]
                m._need(np.array_equal(arrays["example_indices"], np.arange(10000))
                    and p.shape == (10000, len(classes))
                    and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
                    and np.allclose(p.sum(1), 1., atol=1e-6, rtol=1e-5),
                    "Complete original sealed canonical probabilities required before TEST truth")


def score_c4(plan_ref, predictions, native6_predictions, test_targets,
             output_directory, *, guard, execution_check,
             write_headroom_bytes, max_truth_uncompressed_bytes):
    m._ready(guard, execution_check, write_headroom_bytes)
    plan = m._read(plan_ref, observed=True)
    verify_cross_endpoint_barrier(plan, native6_predictions, predictions)
    parents = plan["test_parent_ids"]
    m._need(set(test_targets) == set(parents), "All five exact target bindings required")
    import numpy as np
    rows = {name: {} for name in predictions}
    for parent in parents:
        execution_check()
        products = {name: m._read(refs[parent], observed=True)
                    for name, refs in predictions.items()}
        registry = m._read(products["treeloom"]["registry_ref"], observed=True)
        refs = test_targets[parent]
        schema, truth = _truth(registry, **refs, allowed_splits={"test"},
            max_truth_uncompressed_bytes=max_truth_uncompressed_bytes)
        y = sampled_source_labels(registry, schema, truth)
        del truth
        canonical = sorted(registry["records"], key=lambda r: r["example_index"])
        cohorts = np.asarray([r["cohort"] for r in canonical])
        for method, product in products.items():
            p = m._arrays(product["predictions_ref"], observed=True,
                          cap=8*2**20)["probabilities"]
            rows[method][parent] = dict(overall=c4_metrics(p,y), cohorts={
                c: c4_metrics(p[cohorts == c], y[cohorts == c])
                for c in sampling.COHORTS})
    summary, differences = {}, {}
    for method in rows:
        summary[method] = {}
        for group in ("overall", *sampling.COHORTS):
            values = [rows[method][p]["overall"] if group == "overall"
                      else rows[method][p]["cohorts"][group] for p in parents]
            summary[method][group] = {}
            for metric in ("accuracy", "macro_F1", "Brier", "raw_NLL"):
                actual = [v[metric] for v in values]
                summary[method][group][metric] = dict(
                    mean=None if any(v is None for v in actual) else float(np.mean(actual)),
                    sample_SD=None if any(v is None for v in actual) else float(np.std(actual,ddof=1)),
                    independent_parents=5)
    for group in ("overall", *sampling.COHORTS):
        delta = [((rows["treeloom"][p]["overall"] if group == "overall"
                  else rows["treeloom"][p]["cohorts"][group])["accuracy"] -
                 (rows["argmix"][p]["overall"] if group == "overall"
                  else rows["argmix"][p]["cohorts"][group])["accuracy"])
                 for p in parents]
        differences[group] = dict(per_parent=delta, mean=float(np.mean(delta)),
            sample_SD=float(np.std(delta,ddof=1)), direction="TreeLoom minus ARGMix")
    output = m._new_output(output_directory)
    return m._json(output / "comparison.json", dict(
        schema="european_release_C4_paired_comparison_v1", plan_ref=plan_ref,
        predictions=predictions, native6_predictions=native6_predictions,
        test_targets=test_targets, rows=rows, summary=summary,
        paired_accuracy_differences=differences, source_class_order=list(CLASSES),
        all_registered_endpoint_predictions_before_any_TEST_truth=True,
        records_per_parent=10000, uncertainty="equal-parent mean and sample SD, ddof1",
        separate_genuine_four_head_TreeLoom_training=True,
        published_ARGMix_paper_reproduction=False,
        scope="available release on common AP200Mb data, separate from native-six AP benchmark"), guard)
