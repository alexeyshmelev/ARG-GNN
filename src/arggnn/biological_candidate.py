"""Observed-only whole-tree biological candidate; targets are external capabilities."""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import datetime as dt
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import time

# Launcher entry applies the actual title and inherited lease before importing Torch.
# Direct script entry performs the same admission before importing this module.
import sys
if __name__ == "__main__" and "--train" in sys.argv:
    requested_device = next((a.split("=",1)[1] for a in sys.argv if a.startswith("--device=")),
                            sys.argv[sys.argv.index("--device")+1] if "--device" in sys.argv else "cuda")
    if requested_device == "cuda":
        from .native_argformer_exec import validate_lease, apply_and_validate_title
        validate_lease()
        apply_and_validate_title()

import numpy as np
import torch

from .data import GraphSequence
from .reference_encoder import ReferenceFeatureAdapter
from .age_interface import AgeInterfaceBank, precompute_age_interfaces
from .anchor_kernel import AnchorKernelModel, prepare_anchor_kernel_bank
from .anchor_temporal_baselines import AnchorGRUModel

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
PARENTS = tuple(f"bio_paths_pilot_{s}_{i:03d}" for s in ("train", "val") for i in range(2))
CLASSES = ("Ana", "CHG", "WHG", "EHG")
BASE_FEATURES = tuple(f"observed_reference_{c}" for c in CLASSES) + (
    "actual_query_flag", "observed_sample_flag", "encoded_age", "encoded_parent_branch")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def path(value):
    p = (ROOT / value).resolve()
    p.relative_to(ROOT)
    return p


def jsonable(value):
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, torch.Tensor):
        return jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(p, value):
    p.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n")


def checked(record):
    p = path(record["path"])
    if p.is_symlink() or not p.is_file() or sha(p) != record["sha256"]:
        raise ValueError("Missing/changed bound file: " + str(p))
    return p


def configure_cpu():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Explicitly empty CUDA visibility is required")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)


def verify_sources(cfg):
    for name, expected in cfg["source_sha256"].items():
        if sha(path(name)) != expected:
            raise ValueError("Frozen source drift: " + name)



def configure_fit(device_name, cfg, output):
    """Root-only qualified lease/title/device admission, before any scientific CUDA."""
    from .development_execution import configure_deterministic_development
    from .native_argformer_exec import validate_lease, apply_and_validate_title, canonical_uuid
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    policy = configure_deterministic_development()
    if device_name == "cpu":
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ValueError("CPU fit requires empty CUDA visibility")
        return torch.device("cpu"), {"device":"cpu","execution_policy":policy}, None
    if device_name != "cuda":
        raise ValueError("Only explicit cpu/cuda fit devices supported")
    lease = validate_lease()
    title = apply_and_validate_title()
    # All CUDA calls follow actual lease/title and deterministic policy admission.
    if torch.cuda.device_count() != 1:
        raise ValueError("Root launcher must expose exactly the one leased GPU UUID")
    device = torch.device("cuda:0")
    keep_context = torch.empty(1,device=device)
    torch.cuda.synchronize()
    if torch.cuda.current_device() != 0:
        raise ValueError("Leased single logical CUDA0 required")
    qualification = output/"cuda_qualification"
    qualification.mkdir()
    runtime = {"device":"cuda","pid":os.getpid(),"lease":lease,"actual_process_title":title,
               "execution_policy":policy,"torch":str(torch.__version__),
               "torch_cuda_version":torch.version.cuda}
    # Same stable driver-context UUID approach as the frozen native qualifier;
    # no cudaDeviceProp ABI or foreign prefix/cudart/version is assumed.
    import ctypes
    import uuid
    maps = Path("/proc/self/maps").read_text()
    maps_path = qualification/"cuda_library_maps.txt"
    maps_path.write_text(maps)
    drivers = set()
    for line in maps.splitlines():
        parts = line.split(maxsplit=5)
        if len(parts)==6 and Path(parts[5]).name.startswith("libcuda.so."):
            p=Path(parts[5])
            if not p.is_absolute() or not p.is_file():
                raise ValueError("Actual loaded CUDA driver file unavailable")
            drivers.add(p.resolve())
    if len(drivers)!=1:
        raise ValueError("Exactly one own loaded CUDA driver required")
    driver_path=drivers.pop()
    driver=ctypes.CDLL(str(driver_path))
    actual_device=ctypes.c_int(-1)
    ctx=driver.cuCtxGetDevice
    ctx.argtypes=[ctypes.POINTER(ctypes.c_int)];ctx.restype=ctypes.c_int
    if ctx(ctypes.byref(actual_device)):
        raise ValueError("Actual CUDA driver context device lookup failed")
    class CudaUUID(ctypes.Structure):
        _fields_=[("bytes",ctypes.c_ubyte*16)]
    identity=CudaUUID()
    function_name="cuDeviceGetUuid_v2" if hasattr(driver,"cuDeviceGetUuid_v2") else "cuDeviceGetUuid"
    function=getattr(driver,function_name)
    function.argtypes=[ctypes.POINTER(CudaUUID),ctypes.c_int];function.restype=ctypes.c_int
    if function(ctypes.byref(identity),actual_device.value):
        raise ValueError("Actual CUDA driver context UUID lookup failed")
    actual_uuid=str(uuid.UUID(bytes=bytes(identity.bytes)))
    properties=torch.cuda.get_device_properties(0)
    property_uuid=getattr(properties,"uuid",None)
    if actual_uuid!=lease["canonical_uuid"] or (
            property_uuid is not None and canonical_uuid(str(property_uuid))!=actual_uuid):
        raise ValueError("Actual Torch/driver UUID differs from inherited scheduler lease")
    runtime.update(actual_context_uuid=actual_uuid,torch_property_uuid=str(property_uuid),
                   driver_library=str(driver_path),driver_library_sha256=sha(driver_path),
                   process_maps_sha256=sha(maps_path),uuid_function=function_name,
                   device_name=properties.name,total_memory_bytes=properties.total_memory)
    helper_path=path(cfg["gpu_runtime"]["nvtop_helper"])
    spec=importlib.util.spec_from_file_location("arggnn_biological_nvtop_helpers",helper_path)
    nvtop=importlib.util.module_from_spec(spec);spec.loader.exec_module(nvtop)
    # Frozen helper verifies own compute PID, physical device index and exact title.
    nvtop.nvtop_snapshot(qualification,lease,runtime)
    if validate_lease()!=lease:
        raise ValueError("Inherited lease changed during actual GPU qualification")
    write_json(qualification/"qualification.json",runtime)
    del keep_context
    return device,runtime,lease


def check_fit_lease(lease):
    if lease is not None:
        from .native_argformer_exec import validate_lease, apply_and_validate_title
        if validate_lease()!=lease:
            raise ValueError("Inherited GPU UUID lease changed during fit")
        apply_and_validate_title()


def load_protocol(config_path, expected_hash):
    if not expected_hash or sha(config_path) != expected_hash:
        raise ValueError("A root-reviewed exact config SHA256 is required")
    cfg = json.loads(config_path.read_text())
    if (cfg["kind"] != "biological_candidate_pilot_v1" or cfg["hypothesis_id"] != "H020"
            or cfg["parents"] != list(PARENTS) or cfg["seeds"] != [101, 202, 303]
            or cfg["models"] != ["candidate_age_d2", "candidate_plain_d2", "matched_gru_d2"]
            or cfg["training"]["epochs"] != 30 or cfg["training"]["patience"] != 8
            or cfg["geometry"] != {"tile_bp": 250000, "normalization_length_bp": 2000000,
                                   "num_anchors": 32, "num_scales": 3,
                                   "age_scales": [350.0, 1400.0], "age_strength": 3.0,
                                   "time_scale": 10000.0, "resistance_scale": 0.0001,
                                   "max_condition": 1e12}):
        raise ValueError("Frozen biological pilot method changed")
    if (cfg["storage"]["concurrent_write_reserve_bytes"] < 50000000000
            or cfg["storage"]["project_limit_bytes"] > 450000000000):
        raise ValueError("Require50GB concurrent-write reserve and<=450GB project ceiling")
    verify_sources(cfg)
    # Import reviewed definitions only. Do not call AP protocol(), run() or TruthReader.
    metric_path = path(cfg["metric_helper"])
    spec = importlib.util.spec_from_file_location("arggnn_biological_ap_helpers", metric_path)
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    return cfg, helpers


def observed_contract(cfg, manifest_path, expected_hash, helpers):
    if not expected_hash or sha(manifest_path) != expected_hash:
        raise ValueError("Frozen completed Relate manifest SHA256 is required")
    native = json.loads(manifest_path.read_text())
    by_id = {r["parent_id"]: r for r in native.get("replicates", [])}
    if len(by_id) != 4 or set(by_id) != set(PARENTS):
        raise ValueError("Exactly the four biological development parents are required")
    parents = []
    for parent_id in PARENTS:
        r = by_id[parent_id]
        directory = path("data/biological_paths_execution_pilot_v1") / parent_id
        roster_path = directory / "observed/sample_manifest.json"
        if sha(roster_path) != cfg["observed_roster_sha256"][parent_id]:
            raise ValueError("Frozen source observed roster changed")
        rows = json.loads(roster_path.read_text())
        if len(rows) != 280 or [r["sample_index"] for r in rows] != list(range(280)):
            raise ValueError("Ordered 280-haplotype roster required")
        query = np.array([i for i, row in enumerate(rows) if row["role"] == "query"], dtype=np.int64)
        mixed = [i for i, row in enumerate(rows) if row["role"] == "mixed_reference_only"]
        reference = [i for i, row in enumerate(rows) if row["role"] == "pure_source_reference"]
        dynamic = np.array([i for i, row in enumerate(rows)
                            if row["role"] != "pure_source_reference"], dtype=np.int64)
        if (len(query), len(mixed), len(reference), len(dynamic)) != (120, 64, 96, 184):
            raise ValueError("Require actual Q120, latent mixed64, dynamics Q184 and pure R96")
        for row in rows:
            c = row["observed_reference_class"]
            if row["role"] == "pure_source_reference":
                if c not in range(4) or row["observed_cohort"] != CLASSES[c]:
                    raise ValueError("Pure-reference cohort/class alignment differs")
            elif c is not None:
                raise ValueError("Queries/mixed leaves must have null observed reference class")
        if [sum(rows[i]["observed_reference_class"] == c for i in reference) for c in range(4)] != [24]*4:
            raise ValueError("Require 24 pure observed reference haplotypes per class")
        for i in range(0, 280, 2):
            a, b = rows[i:i+2]
            if (a["sample_id"] != b["sample_id"] or (a["haplotype"], b["haplotype"]) != (0, 1)
                    or any(a[k] != b[k] for k in ("role", "observed_cohort", "age_generations"))):
                raise ValueError("Observed diploid pairs disagree")
        cohorts = np.array([rows[i]["observed_cohort"] for i in query])
        if np.sum(cohorts == "present_day") != 64:
            raise ValueError("Require present64 primary / ancient56 secondary queries")
        grid_record = r["coordinate_contract"]["fixed_grid"]
        grid_path = checked(grid_record)
        with np.load(grid_path, allow_pickle=False) as z:
            positions, edges = z["positions_bp"].copy(), z["bin_edges_bp"].copy()
        if not np.array_equal(positions, np.arange(50, 2000000, 100, dtype=float)) or not np.array_equal(
                edges, np.arange(0, 2000001, 100, dtype=float)):
            raise ValueError("Original complete physical 100bp grid required")
        parent = dict(parent_id=parent_id, split="train" if "_train_" in parent_id else "val",
                      directory=directory, samples=rows, query_slots=query, cohorts=cohorts,
                      dynamics_slots=dynamic, reference_slots=reference,
                      output_indexes=np.array([int(np.flatnonzero(dynamic == i)[0]) for i in query]),
                      raw_query_nodes=np.array([rows[i]["observed_node_id"] for i in query]),
                      positions=positions, edges=edges,
                      row={"files": {"observed/sample_manifest.json": {"sha256": sha(roster_path)},
                                    "observed/scoring_coordinates.npz": {"sha256": grid_record["sha256"]}}})
        parents.append(parent)
    # Audited helper consumes only producer manifest, observed rosters/grid, and receipts.
    observations = helpers.observations(manifest_path, parents, cfg)
    for parent in parents:
        obs = observations[parent["parent_id"]]
        parent["observation"] = obs
        parent["positions"] = obs["positions"]
        parent["edges"] = obs["bin_edges"]
        parent["coverage_original"] = obs["coverage"]
        parent["retained_indexes"] = obs["retained_indexes"]
    return parents


@dataclass
class AccountedStorage:
    """One complete root snapshot; direct writes and cheap free-space checks only."""
    baseline: int
    reserved_new: int
    reserve: int
    limit: int
    snapshot_record: dict
    added: int = 0

    @classmethod
    def admit(cls, snapshot_path, expected_sha, required_bytes, cfg):
        if not expected_sha or sha(snapshot_path) != expected_sha:
            raise ValueError("A hashed complete root storage snapshot is required")
        receipt = json.loads(snapshot_path.read_text())
        if (receipt.get("schema") != "biological_candidate_storage_snapshot_v1"
                or receipt.get("status") != "complete" or receipt.get("returncode") != 0
                or receipt.get("project_root") != str(ROOT)
                or receipt.get("command") != ["du", "-s", "-B1", str(ROOT)]
                or type(receipt.get("used_bytes")) is not int or receipt["used_bytes"] < 0):
            raise ValueError("Storage snapshot must prove a complete successful project-root scan")
        completed = dt.datetime.fromisoformat(receipt["completed_utc"])
        if completed.tzinfo is None or completed > dt.datetime.now(dt.timezone.utc):
            raise ValueError("Root storage snapshot needs a timezone-aware completed timestamp")
        storage = cfg["storage"]
        result = cls(receipt["used_bytes"], int(required_bytes), storage["concurrent_write_reserve_bytes"],
                     storage["project_limit_bytes"],
                     {"path": str(snapshot_path), "sha256": expected_sha, "receipt": receipt})
        result.check()
        return result

    def check(self):
        if self.baseline + self.reserved_new + self.reserve >= self.limit or self.added > self.reserved_new:
            raise ValueError("Conservative project storage budget/headroom exceeded")
        available = os.statvfs(ROOT)
        if self.reserved_new-self.added+self.reserve >= available.f_bavail*available.f_frsize:
            raise ValueError("Filesystem free-space headroom exceeded")

    def account(self, p):
        self.added += max(p.stat().st_size, p.stat().st_blocks*512) + 8192
        self.check()

    def record(self):
        return dict(self.snapshot_record, baseline_bytes=self.baseline,
                    reserved_new_bytes=self.reserved_new, accounted_write_bytes=self.added,
                    concurrent_write_reserve_bytes=self.reserve, per_epoch_root_scans=0)


def pack_observed_tile(ts, parent, left, right):
    """Complete observed trees in a clipped physical tile, with original 2Mb coordinates."""
    obs = parent["observation"]
    nodes = obs["nodes"]
    query_ids = nodes[parent["dynamics_slots"]]
    actual_ids = set(map(int, nodes[parent["query_slots"]]))
    reference = {int(nodes[i]): parent["samples"][i]["observed_reference_class"]
                 for i in parent["reference_slots"]}
    all_samples = set(map(int, nodes))
    landmarks = []
    for c in range(4):
        candidates = [int(nodes[i]) for i in parent["reference_slots"]
                      if parent["samples"][i]["observed_reference_class"] == c]
        landmarks.extend(candidates[i] for i in np.linspace(0, len(candidates)-1, 4, dtype=int))
    position_indexes = np.flatnonzero((parent["positions"] >= left) & (parent["positions"] < right))
    if not len(position_indexes):
        raise ValueError("A clipped physical tile has no original retained center; no fallback permitted")
    xs, clades, ids, graphs, edges, intervals, original_tree_ids = [], [], [], [], [], [], []
    time_norm = math.log1p(10000)
    for tree in ts.trees():
        start = max(tree.interval.left + obs["offset"], left)
        stop = min(tree.interval.right + obs["offset"], right)
        if stop <= start:
            continue
        if tree.num_roots != 1 or set(tree.leaves()) != all_samples:
            raise ValueError("Existing bank requires complete connected trees and all280 sampled leaves")
        packed = list(tree.nodes(order="postorder"))
        lookup = {u: len(xs)+i for i, u in enumerate(packed)}
        graph = len(intervals)
        for u in packed:
            p = tree.parent(u)
            age = tree.time(u)
            branch = 0.0 if p == -1 else tree.time(p)-age
            if not np.isfinite(age) or age < 0 or branch < 0:
                raise ValueError("Observed tree ages/branches invalid")
            feature = np.zeros(8, dtype=np.float32)
            if u in reference:
                feature[reference[u]] = 1
            # Q184 is a dynamics partition only. Actual query flag stays Q120.
            feature[4] = float(u in actual_ids)
            feature[5] = float(u in all_samples)
            feature[6] = math.log1p(age)/time_norm
            feature[7] = math.log1p(branch)/time_norm
            xs.append(feature)
            clades.append([float(tree.is_descendant(landmark, u)) for landmark in landmarks])
            ids.append(int(u) if u in all_samples else -1)
            graphs.append(graph)
            if p != -1:
                edges.extend([(lookup[p], lookup[u]), (lookup[u], lookup[p])])
        intervals.append([start/2000000, stop/2000000])
        original_tree_ids.append(tree.index)
    interval_array = np.asarray(intervals, dtype=np.float64)
    positions = parent["positions"][position_indexes]/2000000
    target_graph = np.searchsorted(interval_array[:, 1], positions, side="right")
    sequence = GraphSequence(
        x=torch.from_numpy(np.asarray(xs, dtype=np.float32)),
        edge_index=torch.tensor(edges, dtype=torch.long).reshape(-1, 2).T.contiguous(),
        node_graph=torch.tensor(graphs, dtype=torch.long), node_ids=torch.tensor(ids, dtype=torch.long),
        clades=torch.tensor(clades, dtype=torch.float32), intervals=torch.from_numpy(interval_array),
        query_node_ids=torch.as_tensor(query_ids.copy(), dtype=torch.long),
        positions=torch.from_numpy(positions), target_graph=torch.as_tensor(target_graph, dtype=torch.long),
        window_bounds=torch.tensor([left/2000000, right/2000000], dtype=torch.float64))
    sequence.validate()
    adapter = ReferenceFeatureAdapter(8, reference_columns=(0,1,2,3),
                                     time_column=6, branch_column=7,
                                     kernel_scales=(.05,.1,.2))
    augmented = adapter.precompute(sequence)
    if augmented.x.shape[1] != 48:
        raise ValueError("Existing observed reference feature contract must have48 fields")
    return augmented, position_indexes, original_tree_ids, landmarks


def field_fingerprint(value):
    """Lossless serialization evidence for every original cached field/tensor."""
    if isinstance(value, torch.Tensor):
        flat = value.detach().cpu().contiguous().reshape(-1)
        h = hashlib.sha256()
        for start in range(0, flat.numel(), 1048576):
            h.update(memoryview(flat[start:start+1048576].numpy()).cast("B"))
        return {"dtype": str(value.dtype), "shape": list(value.shape),
                "stride": list(value.stride()), "sha256": h.hexdigest()}
    if is_dataclass(value):
        return {f.name: field_fingerprint(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {str(k): field_fingerprint(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [field_fingerprint(v) for v in value]
    if isinstance(value, float):
        return {"float_hex": value.hex()}
    return value


def fingerprint_sha(value):
    return hashlib.sha256(json.dumps(field_fingerprint(value), sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def prepare(cfg, helpers, config_path, expected_config, manifest_path, expected_manifest,
            output, snapshot_path, snapshot_sha):
    import tskit
    configure_cpu()
    parents = observed_contract(cfg, manifest_path, expected_manifest, helpers)
    native = json.loads(manifest_path.read_text())
    graphs = sum(r["post_five_iterations"]["trees"] + 7 for r in native["replicates"])
    required = graphs*cfg["storage"]["preparation_bytes_per_graph"] + cfg["storage"]["preparation_fixed_bytes"]
    storage = AccountedStorage.admit(snapshot_path, snapshot_sha, required, cfg)
    output.mkdir(exist_ok=False)
    receipt = dict(kind="biological_candidate_prepared_v1", status="running",
                   config_sha256=expected_config, observed_manifest_path=str(manifest_path),
                   observed_manifest_sha256=expected_manifest, source_sha256=cfg["source_sha256"],
                   truth_opened=False, labels_in_features_or_banks=False, parents=[],
                   geometry=cfg["geometry"], storage=storage.record(),
                   runtime={"torch":str(torch.__version__),"threads":torch.get_num_threads(),
                            "device":"cpu","cache_dtype":"float64"})
    write_json(output/"manifest.partial.json", receipt)
    try:
        for parent in parents:
            directory = output/parent["parent_id"]
            directory.mkdir()
            obs = parent["observation"]
            ts = tskit.load(obs["tree_path"])
            if (ts.num_trees != obs["entry"]["post_five_iterations"]["trees"]
                    or ts.sequence_length != obs["coverage"][1]-obs["coverage"][0]
                    or set(ts.samples()) != set(obs["nodes"])):
                raise ValueError("Actual inferred tree length/sample join differs")
            if not np.allclose(ts.tables.nodes.time[obs["nodes"]],
                               [row["age_generations"] for row in parent["samples"]], atol=1e-6, rtol=0):
                raise ValueError("Actual observed ages differ")
            record = dict(parent_id=parent["parent_id"], split=parent["split"],
                          relate_entry=obs["entry"], dynamics_sample_indexes=parent["dynamics_slots"],
                          actual_query_sample_indexes=parent["query_slots"],
                          reference_sample_indexes=parent["reference_slots"],
                          actual_query_output_indexes=parent["output_indexes"],
                          query_node_ids=obs["nodes"][parent["query_slots"]],
                          dynamics_query_node_ids=obs["nodes"][parent["dynamics_slots"]],
                          coverage_original_bp0=obs["coverage"],
                          retained_fixed100bp_indexes=parent["retained_indexes"],
                          positions_bp=parent["positions"], bin_edges_bp=parent["edges"], tiles=[])
            seen_trees, seen_positions = set(), []
            for tile_number in range(8):
                left = max(tile_number*250000, float(obs["coverage"][0]))
                right = min((tile_number+1)*250000, float(obs["coverage"][1]))
                if right <= left:
                    continue
                sequence, indexes, tree_ids, landmarks = pack_observed_tile(ts, parent, left, right)
                seen_trees.update(tree_ids)
                seen_positions.extend(indexes.tolist())
                # Unmodified precomputation produces plain plus age350/age1400 branches.
                age_bank = precompute_age_interfaces(
                    sequence, time_column=6, parent_branch_column=7, time_scale=10000,
                    resistance_scale=.0001, age_scales=(350.,1400.), strength=3.,
                    reference_ids=torch.as_tensor(obs["nodes"][parent["reference_slots"]].copy(), dtype=torch.long),
                    max_condition=1e12)
                bank = age_bank.plain_bank
                if bank.leaf_x.shape[1:] != (280, 47) or bank.query_node_ids.numel() != 184:
                    raise ValueError("Whole-tree boundary/selected feature/dynamics count changed")
                flags = bank.leaf_x[:, :, bank.feature_columns.index(4)]
                expected_flags = torch.isin(bank.sample_ids, torch.as_tensor(
                    obs["nodes"][parent["query_slots"]].copy(), dtype=torch.long)).double()
                if not torch.equal(flags, expected_flags.expand_as(flags)):
                    raise ValueError("Mixed64 flag must be zero; actual-query120 flag must be one")
                graph_file = directory/f"tile_{tile_number:02d}.whole_observed.pt"
                bank_file = directory/f"tile_{tile_number:02d}.bank.pt"
                torch.save({"sequence": sequence.as_dict(), "labels_in_input": False}, graph_file)
                storage.account(graph_file)
                fixed = prepare_anchor_kernel_bank(bank, age_bank.operator_cache,
                                                   num_anchors=32, num_scales=3)
                bundle = {"schema": "biological_candidate_full_age_bank_v1",
                          "age_bank": age_bank.as_dict(), "fixed_prepared": fixed}
                expected_fingerprint = fingerprint_sha(bundle)
                torch.save(bundle, bank_file)
                storage.account(bank_file)
                # No field dropping, compression, matrix reconstruction or quantization.
                # Torch preserves shared tensor storage losslessly; verify every field.
                del bundle, fixed, age_bank, bank
                gc.collect()
                restored = torch.load(bank_file, map_location="cpu", weights_only=False)
                AgeInterfaceBank.from_dict(restored["age_bank"])
                if fingerprint_sha(restored) != expected_fingerprint:
                    raise ValueError("Original exact age/plain bank serialization roundtrip changed a field")
                del restored
                record["tiles"].append(dict(
                    tile_id=tile_number, physical_interval_bp0=[left,right],
                    normalized_interval=[left/2000000,right/2000000],
                    position_indexes=indexes, original_grid_indexes=parent["retained_indexes"][indexes],
                    original_tree_ids=tree_ids, all_tree_nodes_retained=True, all280_leaves_retained=True,
                    landmarks_observed_node_ids=landmarks,
                    whole_observed_graph={"path":str(graph_file),"sha256":sha(graph_file)},
                    bank={"path":str(bank_file),"sha256":sha(bank_file)},
                    serialized_bytes=graph_file.stat().st_size+bank_file.stat().st_size,
                    query_dynamics=184, pure_reference_sources=96, supervised_outputs=120,
                    mixed_latent_outputs=64, mixed_query_flag=0,
                    all_original_age_cache_fields_retained=True,
                    exact_serialization_roundtrip_checked=True,
                    full_bank_field_fingerprint_sha256=expected_fingerprint))
                print(json.dumps({"stage":"prepared_tile", "parent_id":parent["parent_id"],
                                  "tile":tile_number,"trees":sequence.num_graphs}), flush=True)
                del sequence
                gc.collect()
            if seen_trees != set(range(ts.num_trees)) or seen_positions != list(range(len(parent["positions"]))):
                raise ValueError("Physical tiling omitted a tree or retained fixed-grid center")
            record["original_trees_preserved"] = ts.num_trees
            receipt["parents"].append(record)
            receipt["storage"] = storage.record()
            write_json(output/"manifest.partial.json", receipt)
        if sha(config_path) != expected_config or sha(manifest_path) != expected_manifest:
            raise ValueError("Bound source config/manifest changed during preparation")
        verify_sources(cfg)
        receipt["status"] = "complete_observed_only_Q184_R96_actualQ120"
    except Exception as error:
        receipt.update(status="failed_partial_preserved", error_type=type(error).__name__, error=str(error))
        write_json(output/"manifest.json", receipt)
        raise
    write_json(output/"manifest.json", receipt)
    return receipt


def load_bank(tile, cfg, parent, device):
    p = checked(tile["bank"])
    saved = torch.load(p, map_location="cpu", weights_only=False)
    if (saved["schema"] != "biological_candidate_full_age_bank_v1"
            or tile["exact_serialization_roundtrip_checked"] is not True
            or tile["all_original_age_cache_fields_retained"] is not True):
        raise ValueError("Exact full bank schema/roundtrip provenance differs")
    # Source precomputation, eigensystem/partition validation, anchor preparation
    # and full-field roundtrip occurred before sealing this hashed artifact.
    # Training reuses the fixed prepared object; no repeated eigensolve or raw
    # matrix validation is introduced into each neural forward.
    prepared = saved["fixed_prepared"].to("cpu", dtype=torch.float32)
    if (prepared.bank.leaf_x.shape[1:] != (280,47)
            or prepared.bank.query_node_ids.numel() != 184
            or prepared.bank.query_cache.reference_ids.numel() != 96
            or prepared.num_scales != 3 or prepared.layout.anchors.numel() != 32):
        raise ValueError("Frozen whole-tree model bank dimensions changed")
    nodes=parent["observation"]["nodes"]
    if (not torch.equal(prepared.bank.query_node_ids, torch.as_tensor(
            nodes[parent["dynamics_slots"]].copy(),dtype=torch.long))
            or not torch.equal(prepared.bank.query_cache.reference_ids,torch.as_tensor(
            nodes[parent["reference_slots"]].copy(),dtype=torch.long))
            or set(prepared.bank.sample_ids.tolist()) != set(nodes.tolist())
            or prepared.bank.feature_columns != tuple(c for c in range(48) if c != 7)):
        raise ValueError("Prepared actual inferred identities/reference roles/features changed")
    indexes=np.asarray(tile["position_indexes"],dtype=int)
    if not np.array_equal(prepared.bank.positions.numpy(),parent["positions"][indexes]/2000000):
        raise ValueError("Prepared original normalized2Mb readouts changed")
    actual=torch.as_tensor(nodes[parent["query_slots"]].copy(),dtype=torch.long)
    expected_flags=torch.isin(prepared.bank.sample_ids,actual).float()
    if not torch.equal(prepared.bank.leaf_x[:,:,4],expected_flags.expand_as(prepared.bank.leaf_x[:,:,4])):
        raise ValueError("Prepared mixed64 flag became an actual-query flag")
    return prepared.to(device,dtype=torch.float32)


class ExternalTruthCapability:
    """Only actual query120 targets, after TRAIN loss or sealed VAL score admission."""
    def __init__(self, parent, cfg, purpose, prediction_record=None):
        if purpose not in ("TRAIN_loss", "VAL_after_sealed_prediction"):
            raise ValueError("Undeclared external label capability")
        if parent["split"] != ("train" if purpose == "TRAIN_loss" else "val"):
            raise ValueError("External capability split differs")
        if purpose == "VAL_after_sealed_prediction":
            if prediction_record is None:
                raise ValueError("VAL truth requires an already sealed prediction")
            checked(prediction_record)
        p = path(cfg["external_truth_manifest"]["path"])
        if sha(p) != cfg["external_truth_manifest"]["sha256"]:
            raise ValueError("Frozen external target manifest changed")
        metadata = json.loads(p.read_text())
        rows = [r for r in metadata["replicates"] if r["parent_id"] == parent["parent_id"]]
        if len(rows) != 1 or rows[0]["split"] != parent["split"]:
            raise ValueError("External truth parent/split mismatch")
        source = path("data/biological_paths_execution_pilot_v1")/parent["parent_id"]/"truth/full_truth_intervals.npz"
        expected = rows[0]["files"]["truth/full_truth_intervals.npz"]["sha256"]
        if sha(source) != expected:
            raise ValueError("Frozen external intervals changed")
        with np.load(source, allow_pickle=False) as z:
            self.intervals = z["intervals_bp"].copy()
            self.c4 = z["source_classes_0based"].copy()
            original_ids = z["query_raw_node_ids"].copy()
        if (self.intervals.ndim != 2 or self.intervals.shape[1] != 2
                or self.intervals[0,0] != 0 or self.intervals[-1,1] != 2000000
                or not np.array_equal(self.intervals[:-1,1], self.intervals[1:,0])
                or np.any(self.intervals[:,1] <= self.intervals[:,0])
                or self.c4.shape != (len(self.intervals),120) or self.c4.dtype.kind not in "iu"
                or np.any((self.c4 < 0)|(self.c4 > 3))
                or not np.array_equal(original_ids, parent["raw_query_nodes"])):
            raise ValueError("External full-domain C4/query120 truth correspondence invalid")
        self.provenance = {"path":str(source),"sha256":expected,"purpose":purpose}
        self.mass = None

    def target_mass(self, parent, helpers):
        if self.mass is None:
            bins, rows, spans = helpers.intersections(parent["edges"], self.intervals)
            self.mass = np.zeros((len(parent["positions"]),120,4), dtype=np.float64)
            query = np.arange(120)[None,:]
            np.add.at(self.mass, (bins[:,None], query, self.c4[rows]), spans[:,None])
            widths = np.diff(parent["edges"])[:,None]
            if not np.allclose(self.mass.sum(-1), widths, atol=1e-8, rtol=0):
                raise ValueError("Exact external target mass does not cover scored bins")
        return self.mass


def model_factory(name, cfg):
    common = dict(input_dim=47, num_classes=4, hidden_dim=32, num_layers=2,
                  latent_dim=8, num_scales=3, num_anchors=32, initial_coupling=(5.,20.,80.))
    if name == "matched_gru_d2":
        return AnchorGRUModel(**common, temporal_depth=2, dropout=0.)
    return AnchorKernelModel(**common, kernel_depth=2, initial_damping=1.,
                             geometry="age" if name == "candidate_age_d2" else "plain",
                             coupling="full", rate_parameterization="log")


def optimizer_groups(model, cfg):
    rate = list(model.rate_and_mixture_parameters())
    identities = {id(p) for p in rate}
    if len(identities)!=len(rate) or not identities <= {id(p) for p in model.parameters()}:
        raise ValueError("Existing model rate/mixture parameter group is duplicated or invalid")
    return [{"params":[p for p in model.parameters() if id(p) not in identities],
             "weight_decay":cfg["training"]["weight_decay"]},
            {"params":rate,"weight_decay":0.}]


def sealed_prediction(model, record, parent, cfg, output, storage, device):
    logits = np.empty((len(parent["positions"]),120,4), dtype=np.float32)
    seen = np.zeros(len(parent["positions"]), dtype=bool)
    model.eval()
    with torch.no_grad():
        for tile in record["tiles"]:
            prepared = load_bank(tile,cfg,parent,device)
            outputs=torch.as_tensor(parent["output_indexes"],dtype=torch.long,device=device)
            result = model(prepared).index_select(1,outputs)
            indexes = np.asarray(tile["position_indexes"],dtype=int)
            if result.shape != (len(indexes),120,4) or seen[indexes].any():
                raise ValueError("Actual-query output/grid correspondence changed")
            logits[indexes] = result.detach().cpu().numpy()
            seen[indexes] = True
            del prepared
    if not seen.all() or not np.isfinite(logits).all():
        raise ValueError("Sealed query120 prediction incomplete/nonfinite")
    probabilities = torch.softmax(torch.from_numpy(logits),dim=-1).numpy()
    if (not np.isfinite(probabilities).all() or np.any(probabilities<0)
            or not np.allclose(probabilities.sum(-1),1,atol=1e-6,rtol=1e-5)):
        raise ValueError("Raw C4 probabilities invalid; no floor or renormalization")
    with output.open("xb") as stream:
        np.savez_compressed(stream, logits=logits, C4_probabilities=probabilities,
            positions_bp=parent["positions"], bin_edges_bp=parent["edges"],
            retained_fixed100bp_indexes=parent["retained_indexes"],
            coverage_original_bp0=parent["coverage_original"],
            normalization_length_bp=np.asarray(2000000),
            query_sample_indexes=parent["query_slots"], actual_query_output_indexes=parent["output_indexes"],
            query_node_ids=parent["observation"]["nodes"][parent["query_slots"]])
    storage.account(output)
    return probabilities, {"path":str(output),"sha256":sha(output)}


def exact_probability_nll(probabilities, target_mass, columns):
    p, mass = probabilities[:,columns].astype(np.float64), target_mass[:,columns]
    zero = (p == 0)&(mass > 0)
    if zero.any():
        return float("inf"), float(mass[zero].sum())
    return float(-(mass*np.log(np.where(p>0,p,1))).sum()/mass.sum()), 0.0


def train(cfg, helpers, config_path, expected_config, manifest_path, expected_manifest,
          prepared_path, prepared_sha, seed, output, snapshot_path, snapshot_sha, device_name="cuda"):
    if seed not in cfg["seeds"]:
        raise ValueError("Only frozen seeds101,202,303 are supported")
    parents = observed_contract(cfg,manifest_path,expected_manifest,helpers)
    if not prepared_sha or sha(prepared_path) != prepared_sha:
        raise ValueError("Frozen observed prepared manifest hash required")
    prepared = json.loads(prepared_path.read_text())
    if (prepared["status"] != "complete_observed_only_Q184_R96_actualQ120"
            or prepared["config_sha256"] != expected_config
            or prepared["observed_manifest_sha256"] != expected_manifest
            or prepared["source_sha256"] != cfg["source_sha256"]
            or prepared["truth_opened"] is not False):
        raise ValueError("Observed prepared source/config/provenance binding differs")
    records = {r["parent_id"]:r for r in prepared["parents"]}
    if set(records) != set(PARENTS):
        raise ValueError("Prepared parents differ")
    for parent in parents:
        r = records[parent["parent_id"]]
        if (r["relate_entry"] != parent["observation"]["entry"]
                or r["split"] != parent["split"]
                or r["query_node_ids"] != parent["observation"]["nodes"][parent["query_slots"]].tolist()
                or r["dynamics_query_node_ids"] != parent["observation"]["nodes"][parent["dynamics_slots"]].tolist()
                or r["positions_bp"] != parent["positions"].tolist()
                or r["bin_edges_bp"] != parent["edges"].tolist()
                or r["actual_query_output_indexes"] != parent["output_indexes"].tolist()
                or r["actual_query_sample_indexes"] != parent["query_slots"].tolist()
                or r["dynamics_sample_indexes"] != parent["dynamics_slots"].tolist()
                or r["retained_fixed100bp_indexes"] != parent["retained_indexes"].tolist()):
            raise ValueError("Prepared role/output/grid joins changed")
        for tile in r["tiles"]:
            checked(tile["bank"])
            checked(tile["whole_observed_graph"])
    required = (cfg["storage"]["training_prediction_bytes_per_parent_epoch"]
                *2*cfg["training"]["epochs"]*len(cfg["models"])
                +cfg["storage"]["training_fixed_bytes"])
    storage = AccountedStorage.admit(snapshot_path,snapshot_sha,required,cfg)
    output.mkdir(exist_ok=False)
    device,gpu_runtime,lease=configure_fit(device_name,cfg,output)
    receipt = dict(kind="biological_candidate_fit_v1",hypothesis_id="H020",status="running",
                   seed=seed, config_sha256=expected_config, observed_manifest_sha256=expected_manifest,
                   prepared_manifest_sha256=prepared_sha, source_sha256=cfg["source_sha256"],
                   models={}, device=str(device), storage=storage.record(), TEST_parents=0,
                   GPU_qualification=gpu_runtime,
                   target_scope="actual query120 only; mixed64 latent outputs never labeled",
                   runtime={"torch":str(torch.__version__),"threads":torch.get_num_threads(),
                            "device":str(device),"feature_dtype":"float32","coordinate_dtype":"float64"})
    write_json(output/"receipt.json",receipt)
    states = {}
    for name in cfg["models"]:
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        model = model_factory(name,cfg).to(device)
        states[name] = dict(model=model,
            optimizer=torch.optim.AdamW(optimizer_groups(model,cfg),lr=cfg["training"]["learning_rate"]),
            best=float("inf"), best_epoch=None, stale=0, active=True, history=[],
            initial_weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},steps=0)
        receipt["models"][name] = dict(status="pending",parameters=sum(p.numel() for p in model.parameters()),
                                      configuration=model.configuration_manifest(),
                                      actual_coordinate_units="original physical bp/2000000",
                                      initial_parameter_fingerprint_sha256=fingerprint_sha(model.state_dict()),
                                      query_dynamics=184,reference_sources=96,actual_supervised_outputs=120,
                                      mixed_latent_outputs=64,mixed_query_feature_flag=0,
                                      optimizer_groups=[{"parameters":sum(p.numel() for p in g["params"]),
                                                        "weight_decay":g["weight_decay"]}
                                                        for g in optimizer_groups(model,cfg)])
    train_truth, val_truth = {}, {}
    train_parents = [p for p in parents if p["split"]=="train"]
    val_parents = [p for p in parents if p["split"]=="val"]
    for p in train_parents:
        capability = ExternalTruthCapability(p,cfg,"TRAIN_loss")
        train_truth[p["parent_id"]] = capability.target_mass(p,helpers)
    tiles = [(p,tile) for p in train_parents for tile in records[p["parent_id"]]["tiles"]]
    started = time.monotonic()
    try:
        for epoch in range(1,cfg["training"]["epochs"]+1):
            check_fit_lease(lease)
            active = [n for n,s in states.items() if s["active"]]
            if not active:
                break
            order = list(range(len(tiles)))
            random.Random(seed+epoch).shuffle(order)
            losses = {n:[] for n in active}
            for index in order:
                parent,tile = tiles[index]
                bank = load_bank(tile,cfg,parent,device)
                positions = np.asarray(tile["position_indexes"],dtype=int)
                mass = torch.from_numpy(train_truth[parent["parent_id"]][positions]).to(device,dtype=torch.float32)
                outputs = torch.as_tensor(parent["output_indexes"],dtype=torch.long,device=device)
                for name in active:
                    state = states[name];model=state["model"];model.train()
                    state["optimizer"].zero_grad(set_to_none=True)
                    logits = model(bank).index_select(1,outputs)
                    if logits.shape != mass.shape:
                        raise ValueError("Loss must address exactly actual120 query outputs")
                    loss = -(torch.log_softmax(logits,dim=-1)*mass).sum()/mass.sum()
                    if not bool(torch.isfinite(loss)):
                        raise ValueError("Nonfinite exact TRAIN query-bp loss")
                    loss.backward()
                    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                        raise ValueError("Nonfinite candidate gradient")
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    state["optimizer"].step()
                    state["steps"]+=1
                    losses[name].append(float(loss.detach()))
                del bank
            for name in active:
                state=states[name];val_cells=[]
                for parent in val_parents:
                    pfile=output/f"{name}.epoch_{epoch:02d}.{parent['parent_id']}.probabilities.npz"
                    probabilities,sealed=sealed_prediction(state["model"],records[parent["parent_id"]],
                                                           parent,cfg,pfile,storage,device)
                    if parent["parent_id"] not in val_truth:
                        val_truth[parent["parent_id"]]=ExternalTruthCapability(
                            parent,cfg,"VAL_after_sealed_prediction",sealed)
                    mass=val_truth[parent["parent_id"]].target_mass(parent,helpers)
                    primary=np.flatnonzero(parent["cohorts"]=="present_day")
                    nll,infinite=exact_probability_nll(probabilities,mass,primary)
                    val_cells.append(dict(parent_id=parent["parent_id"],nll=nll,
                                          infinite_nll_query_bp=infinite,sealed_prediction=sealed))
                mean_nll=float(np.mean([v["nll"] for v in val_cells]))
                row=dict(epoch=epoch, mean_parent_VAL_present64_NLL=mean_nll, VAL=val_cells,
                         mean_tile_TRAIN_query120_NLL=float(np.mean(losses[name])),
                         selection_scope="exact retained-center clipped-bin query-bp per parent; unweighted parent mean",
                         optimizer_steps=state["steps"],
                         maximum_parameter_delta_from_initial=max(float((v.detach().cpu()-state["initial_weights"][k]).abs().max())
                              for k,v in state["model"].state_dict().items() if v.is_floating_point()),
                         dynamic_controls=state["model"].configuration_manifest())
                state["history"].append(row)
                if mean_nll < state["best"]:
                    checkpoint=output/f"{name}.best_epoch_{epoch:02d}.pt"
                    torch.save({"model_state":state["model"].state_dict(),
                        "optimizer_state":state["optimizer"].state_dict(),"epoch":epoch,"seed":seed,
                        "config_sha256":expected_config,"observed_manifest_sha256":expected_manifest,
                        "prepared_manifest_sha256":prepared_sha,"selection_VAL_NLL":mean_nll},checkpoint)
                    storage.account(checkpoint)
                    state.update(best=mean_nll,best_epoch=epoch,stale=0,
                                 checkpoint={"path":str(checkpoint),"sha256":sha(checkpoint)})
                else:
                    state["stale"]+=1
                state["active"]=state["stale"]<cfg["training"]["patience"]
                receipt["models"][name].update(status="fitting",history=state["history"],
                    best_epoch=state["best_epoch"],best_mean_parent_VAL_present64_NLL=state["best"],
                    checkpoint=state.get("checkpoint"))
            receipt["storage"]=storage.record()
            write_json(output/"receipt.json",receipt)
            print(json.dumps({"stage":"fit_epoch","seed":seed,"epoch":epoch,"active_models":active}),flush=True)
        for name,state in states.items():
            if state["best_epoch"] is None:
                raise ValueError("No finite VAL-selected checkpoint for "+name)
            saved=torch.load(checked(state["checkpoint"]),map_location="cpu",weights_only=False)
            state["model"].load_state_dict(saved["model_state"])
            selected_delta=max(float((v.detach().cpu()-state["initial_weights"][k]).abs().max())
                               for k,v in state["model"].state_dict().items() if v.is_floating_point())
            if state["steps"]<=0 or not selected_delta>0:
                raise ValueError("Selected candidate checkpoint has no actual changed-weight optimizer fit")
            cells=[]
            for parent in val_parents:
                pfile=output/f"{name}.selected.{parent['parent_id']}.probabilities.npz"
                probabilities,sealed=sealed_prediction(state["model"],records[parent["parent_id"]],
                                                       parent,cfg,pfile,storage,device)
                capability=val_truth[parent["parent_id"]]
                groups={"primary_present_day_64":np.flatnonzero(parent["cohorts"]=="present_day"),
                        "secondary_ancient_56":np.flatnonzero(parent["cohorts"]!="present_day"),
                        "all_queries_120":np.arange(120)}
                metrics={group:helpers.full_span_score(probabilities,parent,capability,columns,cfg["metrics"])
                         for group,columns in groups.items()}
                result=dict(parent_id=parent["parent_id"],split="val",seed=seed,model=name,
                    selected_epoch=state["best_epoch"],sealed_prediction=sealed,
                    external_truth_provenance=capability.provenance,C4_exact_support_metrics=metrics,
                    no_calibration_fit=True,mixed_latent_outputs_labeled=False)
                metric_path=output/f"{name}.{parent['parent_id']}.metrics.json"
                write_json(metric_path,result);storage.account(metric_path)
                cells.append({"parent_id":parent["parent_id"],"path":str(metric_path),"sha256":sha(metric_path)})
            receipt["models"][name].update(status="completed",metrics=cells,
                optimizer_fit_completed=True,optimizer_steps=state["steps"],
                selected_maximum_parameter_delta_from_initial=selected_delta,
                configuration=state["model"].configuration_manifest(),
                actual_coordinate_units="original physical bp/2000000")
        if sha(config_path)!=expected_config or sha(manifest_path)!=expected_manifest or sha(prepared_path)!=prepared_sha:
            raise ValueError("Frozen config/observation/prepared manifest changed during fit")
        verify_sources(cfg)
        check_fit_lease(lease)
        if device.type=="cuda":
            torch.cuda.synchronize()
            receipt["peak_cuda_allocated_bytes"]=torch.cuda.max_memory_allocated()
        receipt.update(status="completed_three_conditions_one_seed",elapsed_seconds=time.monotonic()-started,
                       storage=storage.record())
    except Exception as error:
        receipt.update(status="failed_partial_preserved",error_type=type(error).__name__,error=str(error),
                       elapsed_seconds=time.monotonic()-started,storage=storage.record())
        write_json(output/"receipt.json",receipt)
        raise
    write_json(output/"receipt.json",receipt)
    return receipt


def main():
    """Module entry used by the unchanged project launcher; script owns argparse."""
    import runpy
    sys.modules.setdefault("arggnn.biological_candidate",sys.modules[__name__])
    runpy.run_path(str(ROOT/"scripts/train_biological_candidate_v1.py"),run_name="__main__")


if __name__=="__main__":
    main()
