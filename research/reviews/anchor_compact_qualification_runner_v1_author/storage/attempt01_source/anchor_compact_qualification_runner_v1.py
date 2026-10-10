"""Frozen source-only runner. Import and inactive templates confer no admission."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Callable

import torch

from . import anchor_compact_actual_bank_v1 as sealed

ROOT = sealed.ROOT
RUNNER = "src/arggnn/anchor_compact_qualification_runner_v1.py"
PROTOCOL = "configs/anchor_compact_qualification_runner_v1.json"
ADAPTER_PROTOCOL = {"path": "configs/anchor_compact_actual_bank_v1.json",
                    "sha256": "fb03f822587dfdb3f9ca9d605a2bde9714e3cc3ada9980d53242be05c86e9894"}
ADAPTER_REVIEW = {
    "path": "research/reviews/anchor_compact_actual_bank_v1_independent/review_receipt_v2.json",
    "sha256": "e117a87d0cfdd1ce42a947493d52d7b3df65460e6c56956eb50ea8362876e188"}
STAGES = ("encoder", "source", "action", "decoder", "forward_with_grad",
          "backward", "optimizer", "full_step")
DEVICES = ("cpu_float64", "cuda_float32")
PROBES = ("independent_signal_action_VJPs", "full_model_VJPs_and_clipped_AdamW")
EXPECTED_CONFIG = {
    "kind": "anchor_compact_qualification_runner_protocol_v1",
    "status": "source_only_actual_execution_not_admitted",
    "hypothesis_id": "H020", "fit_allowed": False, "truth_allowed": False,
    "adapter_protocol": ADAPTER_PROTOCOL, "adapter_independent_review": ADAPTER_REVIEW,
    "scientific_protocol_changes": False, "evidence_max_initial_age_seconds": 120,
    "proposed_output_cache_log_cap_bytes": 1000000000,
    "stage_boundary_checks_are_continuous_quotas": False,
    "existing_pool_allowlist_admits_runner": False,
}
BOUNDARIES = {
    "encoder": "finite feature guard and inherited encoder on all 280 leaves",
    "source": "both layers' query/reference projections, tanh, Cqr source and mass construction; cached layer inputs",
    "action": "both layers' raw finite actions, mixing, finite/positive mass guard and normalization; cached sources",
    "decoder": "both layers' anchor/target gate+projection+LayerNorm updates and final all-Q head; cached layer inputs/messages",
    "forward_with_grad": "inherited/compact complete context and all-Q head with autograd, no objective/backward",
    "backward": "frozen all-Q objective backward; fresh forward+objective outside timed boundary",
    "optimizer": "AdamW step only; fresh forward/backward/active clipping outside timed boundary",
    "full_step": "zero_grad(set_to_none), forward, all-Q objective, backward, active norm1 clipping, AdamW",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_bound(record, root=ROOT):
    return json.loads(sealed.checked_file(record, root).read_text())


def canonical_path(value, root=ROOT):
    path = Path(value)
    path = path if path.is_absolute() else root / path
    require(".." not in path.parts and path.is_relative_to(root)
            and path == path.resolve(), "Canonical in-repository path required")
    return path


def validate_domains(runtime, storage, launcher, *, config_record, source_sha256,
                     output, live, now, initial=True):
    """Validate contents, including current independently observed own context."""
    from .development_execution import DETERMINISTIC_POLICY
    from .native_argformer_exec import canonical_uuid
    for record, kind in ((runtime, "root_compact_runner_runtime_v1"),
                         (storage, "root_compact_runner_storage_v1"),
                         (launcher, "root_compact_runner_launcher_v1")):
        require(record.get("kind") == kind and record.get("status") == "active_root_admission"
                and record.get("authorizer") == "root_agent"
                and record.get("project_root") == str(ROOT), "Wrong domain receipt")
        require(record.get("runner_config") == config_record, "Domain configuration differs")
        stamp = record.get("issued_unix_seconds")
        require(type(stamp) in (int, float) and math.isfinite(stamp)
                and 0 <= now - stamp and (not initial or now - stamp <= 120),
                "Domain evidence is future, missing or initially stale")
    require(runtime.get("source_sha256") == source_sha256, "Runtime source-byte map differs")
    for name in ("python", "torch", "torch_cuda", "driver_version", "python_executable",
                 "python_executable_sha256", "actual_process_title", "pid",
                 "actual_context_uuid", "device", "execution_policy"):
        require(runtime.get(name) == live.get(name) and name in live,
                "Live runtime evidence differs: " + name)
    require(runtime["python_executable"] == str(ROOT / ".runtime/env/bin/python")
            and runtime["device"] == "cuda" and runtime["pid"] == os.getpid()
            and runtime["actual_process_title"] == sealed.PROCESS_TITLE
            and runtime["execution_policy"] == DETERMINISTIC_POLICY,
            "Own fixed interpreter/title/determinism required")
    require(live.get("logical_device_count") == 1 and live.get("logical_device") == 0,
            "Exactly one leased logical GPU0 required")
    lease = live["lease"]
    require(launcher.get("lease") == lease and launcher.get("pid") == os.getpid()
            and launcher.get("actual_context_uuid") == lease["canonical_uuid"]
            and runtime["actual_context_uuid"] == lease["canonical_uuid"]
            and canonical_uuid(lease["visible_gpu_uuid"]) == lease["canonical_uuid"]
            and lease["physical_gpu"] in (2, 3, 4, 5)
            and launcher.get("title_before_cuda") == sealed.PROCESS_TITLE
            and launcher.get("title_before_cuda_verified") is True
            and launcher.get("root_pool_runner_allowlist_integrated") is True
            and launcher.get("launch_after_prior_pools_natural_drain") is True,
            "Fresh reviewed root launcher/lease evidence required")
    require(launcher.get("nvtop_snapshot") == live.get("nvtop_snapshot_record")
            and launcher.get("nvtop_exit_code") == 0
            and launcher.get("nvtop_executable") == str(ROOT / ".runtime/env/bin/nvtop"),
            "Hash-bound successful actual NVTOP evidence required")
    snapshot = live["nvtop_snapshot"]
    require(isinstance(snapshot, list), "NVTOP array required")
    own, project_devices = [], set()
    for physical, gpu in enumerate(snapshot):
        for process in gpu.get("processes", []):
            if str(process.get("pid")) == str(os.getpid()):
                own.append((physical, process))
            if process.get("cmdline", "").strip() == sealed.PROCESS_TITLE:
                require(process.get("kind") == "compute", "Project title on noncompute process")
                project_devices.add(physical)
    require(len(own) == 1 and own[0][0] == lease["physical_gpu"]
            and own[0][1].get("kind") == "compute"
            and own[0][1].get("cmdline", "").strip() == sealed.PROCESS_TITLE,
            "Own actual PID/title on leased physical NVTOP device required")
    require(0 < len(project_devices) <= 2 and project_devices <= {2, 3, 4, 5}
            and launcher.get("project_physical_gpus") == sorted(project_devices),
            "Current project inventory violates eligible/max-two GPU policy")
    require(storage.get("reservation_scope") == [str(output)]
            and storage.get("output_cache_log_cap_bytes") == 1000000000
            and storage.get("raw_tensors_or_checkpoints_retained") is False
            and storage.get("global_scan_used") is False
            and storage.get("reservation_active") is True
            and type(storage.get("reservation_expires_unix_seconds")) in (int, float)
            and now < storage["reservation_expires_unix_seconds"],
            "Exact live output/cache/log reservation scope required")
    require(storage.get("accounting_scope") == "entire_repository_including_ignored_environment"
            and storage.get("unaccounted_writer_scope") == []
            and storage.get("warning_threshold_bytes") == 450000000000
            and storage.get("hard_limit_bytes") == 500000000000,
            "Complete current repository storage accounting required")
    base = storage.get("observed_repository_bytes")
    reservations = storage.get("outstanding_growth_reservations")
    require(type(base) is int and base >= 0 and isinstance(reservations, list),
            "Invalid repository storage projection")
    seen, own_reservations, total = set(), 0, base
    for item in reservations:
        require(isinstance(item.get("id"), str) and item["id"] not in seen
                and type(item.get("bytes")) is int and item["bytes"] >= 0
                and item.get("status") == "reserved", "Invalid growth reservation")
        seen.add(item["id"])
        total += item["bytes"]
        require(isinstance(item.get("scope"), list) and bool(item["scope"]),
                "Reservation lacks explicit repository scope")
        for scope in item["scope"]:
            canonical_path(scope)
        if item["scope"] == [str(output)]:
            require(item["bytes"] == 1000000000, "Runner reservation must cover proposed1GB")
            own_reservations += 1
    require(own_reservations == 1 and total == storage.get("projected_repository_peak_bytes")
            and total < 450000000000, "Current reserved storage projection reaches warning")
    require(storage.get("current_reservation_inventory") == reservations,
            "Current reservation inventory and projection differ")


@dataclass(frozen=True)
class RunnerAdmissionV1:
    adapter: object
    runner_config: dict
    root_receipt: dict
    binding: dict
    runtime: dict
    storage: dict
    launcher: dict
    source_sha256: dict
    output: Path


def observe_live(launcher):
    """Read own live lease/title, driver context, precision and NVTOP each boundary."""
    import ctypes
    from .development_execution import deterministic_development_manifest
    from .native_argformer_exec import validate_lease, canonical_uuid
    lease = validate_lease()
    title = Path("/proc/self/cmdline").read_bytes().split(b"\0")[0].decode().strip()
    require(title == sealed.PROCESS_TITLE, "Root must set title before any CUDA context")
    policy = deterministic_development_manifest()
    require(torch.cuda.is_initialized(), "Root must qualify its own CUDA context first")
    require(torch.cuda.device_count() == 1 and torch.cuda.current_device() == 0,
            "Single own logical CUDA0 required")
    driver = ctypes.CDLL("libcuda.so.1")
    device = ctypes.c_int()
    require(driver.cuCtxGetDevice(ctypes.byref(device)) == 0, "No current CUDA driver context")
    class UUID(ctypes.Structure):
        _fields_ = [("bytes", ctypes.c_ubyte * 16)]
    value = UUID()
    driver.cuDeviceGetUuid.argtypes = [ctypes.POINTER(UUID), ctypes.c_int]
    require(driver.cuDeviceGetUuid(ctypes.byref(value), device) == 0,
            "Actual driver context UUID unavailable")
    import uuid
    actual_uuid = str(uuid.UUID(bytes=bytes(value.bytes)))
    require(actual_uuid == canonical_uuid(lease["visible_gpu_uuid"]), "Own driver UUID differs")
    nvtop_file = sealed.checked_file(launcher["nvtop_snapshot"], ROOT)
    admitted_snapshot = json.loads(nvtop_file.read_text())
    result = subprocess.run([str(ROOT / ".runtime/env/bin/nvtop"), "--snapshot"],
                            capture_output=True, check=True, timeout=30)
    snapshot = json.loads(result.stdout)
    # Root-issued raw snapshot is parsed separately from current observation.
    require(isinstance(admitted_snapshot, list), "Admitted raw NVTOP content missing")
    def own_rows(rows):
        return [(i, p) for i, gpu in enumerate(rows) for p in gpu.get("processes", [])
                if str(p.get("pid")) == str(os.getpid())]
    a, b = own_rows(admitted_snapshot), own_rows(snapshot)
    require(len(a) == len(b) == 1 and a[0][0] == b[0][0]
            and all(a[0][1].get(k) == b[0][1].get(k) for k in ("pid", "kind", "cmdline")),
            "Admitted raw NVTOP own PID/title/index differs from live")
    result = subprocess.run(["nvidia-smi", "--id=" + lease["visible_gpu_uuid"],
                             "--query-gpu=driver_version", "--format=csv,noheader"],
                            capture_output=True, text=True, check=True, timeout=15)
    executable = ROOT / ".runtime/env/bin/python"
    require(Path(sys.executable) == executable, "Exact project interpreter required")
    return dict(python=platform.python_version(), torch=torch.__version__,
        torch_cuda=torch.version.cuda, driver_version=result.stdout.strip(),
        python_executable=str(executable), python_executable_sha256=sealed.sha256_file(executable),
        actual_process_title=title, pid=os.getpid(), actual_context_uuid=actual_uuid,
        device="cuda", execution_policy=policy, logical_device_count=torch.cuda.device_count(),
        logical_device=torch.cuda.current_device(), lease=lease,
        nvtop_snapshot=snapshot, nvtop_snapshot_record=launcher["nvtop_snapshot"])


def admit_runner_v1(config_record, admission_record, *, live_observer=observe_live):
    """No path fallback, old runtime receipt reuse, or active template accepted."""
    require(Path(__file__).resolve() == ROOT / RUNNER, "Only canonical reviewed runner admitted")
    cfg = read_bound(config_record)
    require(config_record["path"] == PROTOCOL and cfg == EXPECTED_CONFIG, "Exact frozen runner protocol required")
    root = read_bound(admission_record)
    require(root.get("kind") == "root_compact_qualification_runner_v1_admission"
            and root.get("status") == "active_root_admission"
            and root.get("authorizer") == "root_agent"
            and root.get("project_root") == str(ROOT)
            and root.get("runner_config") == config_record
            and root.get("fit_allowed") is False and root.get("truth_allowed") is False,
            "Fresh active root runner admission required")
    source = {"path": RUNNER, "sha256": sealed.sha256_file(ROOT / RUNNER)}
    require(root.get("runner_source") == source, "Root runner source bytes differ")
    review = read_bound(root["independent_runner_review"])
    require(review.get("verdict") == "PASS" and review.get("runner_sha256") == source["sha256"]
            and review.get("protocol_sha256") == config_record["sha256"]
            and review.get("actual_execution_admitted") is False, "Exact independent runner review required")
    adapter_receipt = read_bound(root["adapter_admission"])
    require(adapter_receipt.get("independent_review") == ADAPTER_REVIEW,
            "Only sealed independent adapter v2 PASS accepted")
    adapter = sealed.admit_actual_qualification_v1(ADAPTER_PROTOCOL, root["adapter_admission"])
    source_map = dict(adapter.config["source_sha256"], **{
        RUNNER: source["sha256"], PROTOCOL: config_record["sha256"]})
    for path, digest in source_map.items():
        sealed.checked_file({"path": path, "sha256": digest})
    domains = {}
    for name in ("runtime_admission", "storage_admission", "launcher_admission"):
        require(root.get(name) == adapter_receipt.get(name), "Runner/adapter domain binding differs")
        domains[name] = read_bound(root[name])
    output = canonical_path(root["output_scope"])
    require(output != ROOT and output.is_dir() and not (output / "report.json").exists(),
            "Root-created unused owned output scope required")
    result = RunnerAdmissionV1(adapter, cfg, root, dict(config=config_record, admission=admission_record),
        domains["runtime_admission"], domains["storage_admission"], domains["launcher_admission"],
        source_map, output)
    validate_domains(result.runtime, result.storage, result.launcher, config_record=config_record,
        source_sha256=source_map, output=output, live=live_observer(result.launcher),
        now=time.time(), initial=True)
    return result


def recheck_runner(admission):
    sealed.recheck_admission_v1(admission.adapter)
    require(read_bound(admission.binding["config"]) == admission.runner_config
            and read_bound(admission.binding["admission"]) == admission.root_receipt,
            "Runner admission/config changed")
    for name, expected in (("runtime_admission", admission.runtime),
                            ("storage_admission", admission.storage),
                            ("launcher_admission", admission.launcher)):
        require(read_bound(admission.root_receipt[name]) == expected, "Domain receipt changed")
    read_bound(admission.root_receipt["independent_runner_review"])
    for path, digest in admission.source_sha256.items():
        sealed.checked_file({"path": path, "sha256": digest})
    validate_domains(admission.runtime, admission.storage, admission.launcher,
        config_record=admission.binding["config"], source_sha256=admission.source_sha256,
        output=admission.output, live=observe_live(admission.launcher), now=time.time(), initial=False)


def check_owned_storage(output):
    """Only walk this new reservation. Stage boundaries do not enforce a quota."""
    for name in ("TMPDIR", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME",
                 "CUDA_CACHE_PATH", "TORCH_HOME", "TORCHINDUCTOR_CACHE_DIR",
                 "TRITON_CACHE_DIR", "NUMBA_CACHE_DIR", "MPLCONFIGDIR",
                 "PYTHONPYCACHEPREFIX"):
        require(name in os.environ and canonical_path(os.environ[name]).is_relative_to(output),
                "Root must redirect temporary/cache/config paths into reservation: " + name)
    total = 0
    for parent, dirs, files in os.walk(output, followlinks=False):
        for name in dirs + files:
            path = Path(parent) / name
            require(not path.is_symlink(), "No symlink in owned output scope")
        total += sum((Path(parent) / name).stat().st_size for name in files)
    require(total <= 1000000000, "Owned output/cache/log stage cap exceeded")
    return {"owned_scope": str(output), "observed_bytes": total,
            "cap_bytes": 1000000000, "continuous_or_OS_quota": False}


def frozen_cells(selected):
    require(1 <= len(selected) <= 3, "Metadata-only deduplicated selection has1..3 tiles")
    minimum = [i for i, tile in enumerate(selected)
               if "minimum_tree_count" in tile["selection_reasons"]]
    require(len(minimum) == 1, "Unique minimum-tree selection required")
    cells = []
    def append(index, geometry, state, coupling, device, probes):
        tile = selected[index]
        for probe in probes:
            cells.append(dict(selection_index=index, parent_id=tile["parent_id"],
                tile_id=tile["tile_id"], graphs=tile["graphs"], readouts=tile["readouts"],
                geometry=geometry, state=state, coupling=coupling, device_condition=device,
                seed=101, probe=probe))
    for i in range(len(selected)):
        for geometry in ("plain", "age"):
            for device in DEVICES:
                append(i, geometry, "initial", "full", device, PROBES)
    for device in DEVICES:
        append(minimum[0], "age", "rate_mixture_stress", "full", device, PROBES)
        for control in sealed.COUPLINGS[1:]:
            append(minimum[0], "age", "initial", control, device, PROBES[:1])
    require(len(cells) == 8 * len(selected) + 10
            and len({json.dumps(x, sort_keys=True) for x in cells}) == len(cells),
            "Frozen coverage missing or duplicated")
    return cells


def expected_checks(cell):
    if cell["probe"] == PROBES[0]:
        return {"raw_per_scale", "numerator", "mass", "normalized_all_Q"} | {
            name + ".VJP" for name in ("query_signal", "reference_signal",
                                      "raw_coupling", "raw_damping", "mixture_logits")}
    model, _ = sealed.qualification_models_v1(cell["geometry"], cell["state"])
    names = {"logits_all_Q", "final_anchors", "final_targets", "input_feature_VJP", "preclip_norm"}
    for i in range(2):
        names |= {f"layer{i}.{name}" for name in ("anchors", "targets", "numerator",
                              "mass", "sources", "mass_sources", "raw_per_scale")}
    for name, _ in model.named_parameters():
        names |= {name + suffix for suffix in (".VJP", ".post_AdamW", ".optimizer.step",
                                                ".optimizer.exp_avg", ".optimizer.exp_avg_sq")}
    return names


def validate_performed(cell, result):
    """Never let requested labels replace a mismatched performed helper identity."""
    require(result.get("performed_request") == cell, "Performed tile/seed/state/request differs")
    require(result.get("seeded_constructor") == (
        "qualification_direct_control_models_v2" if cell["coupling"] != "full"
        else "qualification_models_v1"), "Wrong seeded/control constructor")
    expected = dict(geometry=cell["geometry"], coupling=cell["coupling"],
        native_geometry=cell["geometry"], compact_geometry=cell["geometry"],
        native_coupling=cell["coupling"], compact_coupling=cell["coupling"],
        dtype="torch.float64" if cell["device_condition"] == "cpu_float64" else "torch.float32",
        device="cpu" if cell["device_condition"] == "cpu_float64" else "cuda:0",
        graphs=cell["graphs"], readouts=cell["readouts"], Q=184, R=96,
        anchors=32, scales=3, latent_dim=8, kernel_depth=2, probe=cell["probe"])
    require(result.get("cell_identity") == expected, "Returned performed helper identity differs")
    require(result.get("all_Q_checked") is True and result.get("labels_opened") is False,
            "Full-Q/no-label evidence missing")
    checks = result.get("checks", {})
    require(set(checks) == expected_checks(cell), "Required numerical/optimizer families missing")
    for name, metric in checks.items():
        require(type(metric.get("elements")) is int and metric["elements"] > 0,
                "Empty numerical family: " + name)
        for key in ("max_absolute_error", "max_acceptance_units", "rtol", "atol"):
            require(type(metric.get(key)) in (int, float) and math.isfinite(metric[key])
                    and metric[key] >= 0, "Invalid numerical metric: " + name)
        gradient = (name.endswith(".VJP") or ".optimizer." in name
                    or name.endswith(".post_AdamW") or name in ("preclip_norm", "input_feature_VJP"))
        f32 = cell["device_condition"] == "cuda_float32" or name.endswith(".optimizer.step")
        tolerance = ((1e-3, 1e-5) if gradient else (2e-4, 2e-5)) if f32 else (
                     (1e-8, 1e-10) if gradient else (1e-10, 1e-12))
        require((metric["rtol"], metric["atol"]) == tolerance,
                "Frozen tolerance differs: " + name)
        require(type(metric.get("passed")) is bool
                and metric["passed"] == (metric["max_acceptance_units"] <= 1),
                "Numerical PASS flag contradicts metric: " + name)
    require(result.get("status") == "PASS" and all(m["passed"] for m in checks.values()),
            "Numerical cell failed")
    if cell["probe"] == PROBES[1]:
        require(result.get("input_models_modified") is False
                and result.get("preclip_native_norm", 0) > 1
                and result.get("preclip_compact_norm", 0) > 1,
                "Matched private active clipped optimizer evidence missing")


def perform_cell(admission, cell):
    sealed.recheck_admission_v1(admission.adapter)
    cpu, metadata = sealed.load_admitted_train_bank_v1(admission.adapter, cell["selection_index"])
    selected = admission.adapter.selected[cell["selection_index"]]
    require(all(metadata.get(k) == v for k, v in selected.items())
            and metadata.get("full_file_and_full_field_hash_checked") is True
            and metadata.get("labels_opened") is False and metadata.get("bank_modified") is False,
            "Performed loaded tile/integrity differs")
    if cell["coupling"] == "full":
        native, compact = sealed.qualification_models_v1(cell["geometry"], cell["state"])
        constructor = "qualification_models_v1"
    else:
        native, compact = sealed.qualification_direct_control_models_v2(cell["geometry"], cell["coupling"])
        constructor = "qualification_direct_control_models_v2"
    # The sealed CPU seed101 constructor and explicit stress state precede transfer.
    for model in (native, compact):
        require(model.hidden_dim == 32 and model.num_layers == 2 and model.input_dim == 47
                and model.rate_parameterization == "log" and model.num_classes == 4,
                "Frozen backbone changed")
        for layer in model.layers:
            for name, endpoints in (("raw_coupling", (-.7, .6)),
                                    ("raw_damping", (.4, -.3)),
                                    ("mixture_logits", (-.8, .7))):
                value = getattr(layer, name)
                expected = torch.zeros_like(value) if cell["state"] == "initial" else (
                    torch.linspace(*endpoints, 24).reshape(3, 8))
                require(torch.equal(value, expected), "Performed state differs")
    view = sealed.derive_compact_actual_bank_v1(cpu, cell["geometry"], cell["coupling"])
    device, dtype = ("cpu", torch.float64) if cell["device_condition"] == "cpu_float64" else (
                    "cuda:0", torch.float32)
    bank, view = cpu.to(device, dtype), view.to(device, dtype)
    native, compact = native.to(device=device, dtype=dtype), compact.to(device=device, dtype=dtype)
    if cell["probe"] == PROBES[0]:
        result = sealed.compare_action_VJPs_v1(native, bank, view)
    else:
        result = sealed.compare_model_step_v1(native, compact, bank, view)
    return dict(result, performed_request=cell, seeded_constructor=constructor,
                loaded_integrity=metadata)


def qualify_cells(selected, perform, save):
    cells, records = frozen_cells(selected), []
    for i, cell in enumerate(cells):
        result = None
        try:
            result = perform(cell)
            validate_performed(cell, result)
            record = dict(request=cell, validation="PASS", result=result)
        except Exception as error:
            record = dict(request=cell, validation="FAIL", result=result,
                          error_type=type(error).__name__, error=str(error))
        records.append(record)
        save(i, record)
    passed = len(records) == len(cells) and all(r["validation"] == "PASS" for r in records)
    return dict(status="PASS" if passed else "FAIL", all_frozen_cases_passed=passed,
                planned_cell_count=len(cells), records=records,
                complete_frozen_cell_coverage=True)


def summarize_pair(report):
    samples = report["samples"]
    require(len(samples) == 20, "Missing ten ABBA samples per implementation")
    for repetition in range(10):
        rows = [x for x in samples if x["repetition"] == repetition]
        order = ["native", "compact"] if repetition % 2 == 0 else ["compact", "native"]
        require([x["implementation"] for x in rows] == order
                and [x["order_index"] for x in rows] == [0, 1], "ABBA order differs")
    def stats(values):
        require(values and all(math.isfinite(x) and x >= 0 for x in values), "Invalid raw sample")
        return dict(median=statistics.median(values), min=min(values), max=max(values))
    grouped = {}
    for name in ("native", "compact"):
        rows = [x for x in samples if x["implementation"] == name]
        grouped[name] = {key: stats([x[key] for x in rows]) for key in rows[0]
                         if key not in ("implementation", "repetition", "order_index")}
    ratios = []
    for i in range(10):
        pair = {x["implementation"]: x["synchronized_wall_seconds"]
                for x in samples if x["repetition"] == i}
        require(pair["compact"] > 0, "Zero compact duration")
        ratios.append(pair["native"] / pair["compact"])
    return dict(report, summary=grouped, paired_native_over_compact_wall_ratios=ratios,
                paired_ratio_summary=stats(ratios), measured_speedup_claim=False)


def full_step(model, bank, optimizer):
    optimizer.zero_grad(set_to_none=True)
    context = model.encode_context(bank)
    logits = model.head(context.targets)
    objective = sealed.qualification_objective_v1(logits, context)
    objective.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    require(float(norm) > 1, "Profiling must preserve active frozen clipping")
    optimizer.step()
    return model, bank, optimizer, context, logits, objective, norm


def _staged_parts(model, bank, implementation):
    from .anchor_kernel import AnchorLayerMessages, anchor_kernel_action
    from .anchor_spectral_compact_v1 import anchor_spectral_compact_action_v1
    layout = bank.layout
    features = bank.bank.leaf_x if implementation == "native" else bank.leaf_x
    response = bank.bank.query_cache.normalized_response if implementation == "native" else bank.normalized_response
    query = bank.bank.query_cache.query_indices if implementation == "native" else bank.query_indices
    reference = bank.bank.query_cache.reference_indices if implementation == "native" else bank.reference_indices
    graph = bank.bank.target_graph if implementation == "native" else bank.target_graph
    operators = bank.operators[model.geometry] if implementation == "native" else bank.geometry
    def encoder():
        require(bool(torch.isfinite(features).all()), "Finite encoder input required")
        return model.encoder(features, response)
    h = encoder()
    anchors, targets, fixed = h[:, query][layout.anchor_graph], h[:, query][graph], h[:, reference][layout.anchor_graph]
    def source(layer, a):
        q, r = layer.query_projection(a).tanh(), layer.reference_projection(fixed).tanh()
        cqr, rates = operators.cqr[layout.anchor_graph], layer.rates
        values = q[:, None] - (cqr @ r[:, None]) * rates[None, :, None, :]
        masses = torch.ones_like(values) - cqr.sum(-1)[..., None] * rates[None, :, None, :]
        return values, masses
    def action(layer, bundle):
        values, masses = bundle
        packed = torch.stack((values, masses), -1)
        if implementation == "native":
            raw = anchor_kernel_action(layout, operators, packed, layer.rates, layer.damping,
                                       coupling=model.coupling)
        else:
            raw = anchor_spectral_compact_action_v1(layout, operators, packed, layer.rates,
                layer.damping, coupling=model.coupling, readout_chunk_size=128)
        mixed = (raw * layer.mixture[None, :, None, :, None]).sum(1)
        numerator, mass = mixed.unbind(-1)
        require(bool(torch.isfinite(mixed).all()) and bool((mass > 0).all()),
                "Unchanged finite positive action normalization")
        normalized = numerator / mass
        return AnchorLayerMessages(normalized[layout.anchor_output], normalized[layout.target_output],
                                   numerator, mass, values, masses)
    cached = []
    for layer in model.layers:
        bundle = source(layer, anchors)
        message = action(layer, bundle)
        cached.append((layer, anchors, targets, bundle, message))
        anchors, targets = layer.update(anchors, message.anchors), layer.update(targets, message.targets)
    def decoder():
        outputs = [(layer.update(a, message.anchors), layer.update(t, message.targets))
                   for layer, a, t, _, message in cached]
        return outputs, model.head(outputs[-1][1])
    return {"encoder": encoder,
            "source": lambda: [source(layer, a) for layer, a, _, _, _ in cached],
            "action": lambda: [action(layer, bundle) for layer, _, _, bundle, _ in cached],
            "decoder": decoder}


def stage_factory(cpu, geometry, implementation, stage):
    """CPU-only closure; model/bank transfer and fresh state are untimed setup."""
    def factory():
        pair = sealed.qualification_models_v1(geometry, "initial")
        model = pair[0 if implementation == "native" else 1].to(device="cuda:0", dtype=torch.float32)
        del pair
        bank = cpu if implementation == "native" else sealed.derive_compact_actual_bank_v1(cpu, geometry)
        bank = bank.to("cuda:0", torch.float32)
        model.eval()
        if stage in STAGES[:4]:
            return _staged_parts(model, bank, implementation)[stage]
        if stage == "forward_with_grad":
            return lambda: (lambda context: (context, model.head(context.targets)))(model.encode_context(bank))
        optimizer = sealed.optimizer_v1(model)
        if stage == "full_step":
            return lambda: full_step(model, bank, optimizer)
        context = model.encode_context(bank)
        logits = model.head(context.targets)
        objective = sealed.qualification_objective_v1(logits, context)
        if stage == "backward":
            def backward():
                objective.backward()
                return model, bank, context, logits, objective
            return backward
        objective.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        require(float(norm) > 1, "Optimizer setup must actively clip")
        def step():
            optimizer.step()
            return model, bank, optimizer, context, logits, objective, norm
        return step
    return factory


def retained_bytes(value):
    """Exact unique backing-storage bytes, not allocator/process memory."""
    from dataclasses import fields, is_dataclass
    seen, objects = set(), set()
    def walk(item):
        if id(item) in objects:
            return 0
        objects.add(id(item))
        if isinstance(item, torch.Tensor):
            storage = item.untyped_storage()
            key = (str(item.device), storage.data_ptr(), storage.nbytes())
            if key in seen:
                return 0
            seen.add(key)
            return storage.nbytes()
        if is_dataclass(item):
            return sum(walk(getattr(item, f.name)) for f in fields(item))
        if isinstance(item, dict):
            return sum(walk(x) for x in item.values())
        if isinstance(item, (tuple, list)):
            return sum(walk(x) for x in item)
        return 0
    return walk(value)


def timed_guarded_load(admission, index):
    """Diagnostic constituents plus measured authoritative sealed load path.

    Diagnostic hash/deserialize/field probes are not added to guarded total.
    Every diagnostic deserialization is fully file+field checked before discard.
    """
    from .biological_candidate import fingerprint_sha
    sealed.recheck_admission_v1(admission.adapter)
    selected = admission.adapter.selected[index]
    filename = canonical_path(selected["bank"]["path"])
    diagnostics = {}
    def timed(name, function):
        start = time.perf_counter()
        result = function()
        diagnostics[name] = time.perf_counter() - start
        return result
    digest = timed("file_hash", lambda: sealed.sha256_file(filename))
    require(digest == selected["bank"]["sha256"], "Diagnostic file bytes differ")
    saved = timed("deserialize", lambda: torch.load(filename, map_location="cpu", weights_only=False))
    field = timed("full_field_hash", lambda: fingerprint_sha(saved))
    require(saved.get("schema") == "biological_candidate_full_age_bank_v1"
            and field == selected["full_bank_field_fingerprint_sha256"]
            and sealed.sha256_file(filename) == digest, "Diagnostic full-file/field integrity differs")
    del saved
    start = time.perf_counter()
    cpu, meta = sealed.load_admitted_train_bank_v1(admission.adapter, index)
    guarded = time.perf_counter() - start
    require(meta.get("full_file_and_full_field_hash_checked") is True, "Guarded load integrity missing")
    return cpu, diagnostics, guarded


def profile_load_to_step(admission, index, geometry, boundary):
    samples = []
    for repetition in range(3):
        order = ("native", "compact") if repetition % 2 == 0 else ("compact", "native")
        for order_index, implementation in enumerate(order):
            boundary("matched_load_start")
            cpu, diagnostic, guarded_seconds = timed_guarded_load(admission, index)
            # Matched total includes only the actual common sealed guarded load,
            # not the independent diagnostic constituents above.
            start = time.perf_counter()
            metadata_start = time.perf_counter()
            bank = cpu if implementation == "native" else sealed.derive_compact_actual_bank_v1(cpu, geometry)
            metadata_seconds = time.perf_counter() - metadata_start
            dtype_start = time.perf_counter()
            bank = bank.to("cpu", torch.float32)
            dtype_seconds = time.perf_counter() - dtype_start
            pair = sealed.qualification_models_v1(geometry, "initial")
            model = pair[0 if implementation == "native" else 1].to(dtype=torch.float32)
            del pair
            torch.cuda.synchronize()
            transfer_start = time.perf_counter()
            bank, model = bank.to("cuda:0", torch.float32), model.to("cuda:0")
            torch.cuda.synchronize()
            transfer_seconds = time.perf_counter() - transfer_start
            optimizer = sealed.optimizer_v1(model)
            compute_start = time.perf_counter()
            value = full_step(model, bank, optimizer)
            torch.cuda.synchronize()
            compute_seconds = time.perf_counter() - compute_start
            matched_seconds = guarded_seconds + time.perf_counter() - start
            samples.append(dict(implementation=implementation, repetition=repetition, order_index=order_index,
                diagnostic_load_stages_seconds=diagnostic, common_guarded_load_seconds=guarded_seconds,
                compact_metadata_seconds=metadata_seconds, dtype_conversion_seconds=dtype_seconds,
                synchronized_transfer_seconds=transfer_seconds, matched_guarded_load_to_step_seconds=matched_seconds,
                matched_compute_full_step_seconds=compute_seconds,
                full_file_and_full_field_integrity=True))
            del value, optimizer, model, bank, cpu
            gc.collect()
            boundary("matched_load_end")
    summary = {}
    ratios = []
    for implementation in ("native", "compact"):
        rows = [r for r in samples if r["implementation"] == implementation]
        keys = [k for k in rows[0] if k.endswith("_seconds") and k != "diagnostic_load_stages_seconds"]
        summary[implementation] = {k: dict(median=statistics.median([r[k] for r in rows]),
            min=min(r[k] for r in rows), max=max(r[k] for r in rows)) for k in keys}
        summary[implementation]["diagnostic_load_stages_seconds"] = {
            k: dict(median=statistics.median([r["diagnostic_load_stages_seconds"][k] for r in rows]),
                    min=min(r["diagnostic_load_stages_seconds"][k] for r in rows),
                    max=max(r["diagnostic_load_stages_seconds"][k] for r in rows))
            for k in ("file_hash", "deserialize", "full_field_hash")}
    for repetition in range(3):
        pair = {r["implementation"]: r["matched_guarded_load_to_step_seconds"]
                for r in samples if r["repetition"] == repetition}
        ratios.append(pair["native"] / pair["compact"])
    return dict(samples=samples, summary=summary, paired_native_over_compact_wall_ratios=ratios,
        paired_ratio_summary=dict(median=statistics.median(ratios), min=min(ratios), max=max(ratios)),
        matched_total_scope="common sealed guarded load+metadata+dtype+fresh model+sync transfer+optimizer construction+full_step",
        diagnostic_stages_overlap_guarded_load=True, diagnostic_stages_added_to_matched_total=False,
        disk_cold_cache_claim=False, original_native_fit_load_path_measured=False, measured_speedup_claim=False)


def profile_cell(admission, index, geometry, parity, boundary, save):
    cpu, metadata = sealed.load_admitted_train_bank_v1(admission.adapter, index)
    view = sealed.derive_compact_actual_bank_v1(cpu, geometry)
    provenance = dict(selected=metadata, geometry=geometry, state="initial", seed=101,
        coupling="full", dtype="torch.float32", chunk_size=128, Q=184, R=96, A=32, S=3,
        H=32, latent_dim=8, depth=2, native_segments=cpu.layout.segment_graph.numel(),
        compact_segments=view.layout.num_segments, native_CPU_retained_storage_bytes=retained_bytes(cpu),
        compact_CPU_retained_storage_bytes=retained_bytes(view),
        compact_aliases_native_CPU_storage=True, clipping_active_required=True)
    del view
    records = []
    for stage in STAGES:
        boundary("profile_" + stage + "_start")
        factories = {name: stage_factory(cpu, geometry, name, stage) for name in ("native", "compact")}
        result = sealed.cuda_profile_pair_v1(factories, admission.adapter, admission.runtime,
            admission.launcher["lease"], parity, stage=stage, repetitions=10, warmups=3)
        result = dict(summarize_pair(result), exact_boundary=BOUNDARIES[stage], provenance=provenance)
        save(stage, result)
        records.append(result)
        del factories, result
        gc.collect()
        boundary("profile_" + stage + "_end")
    matched = profile_load_to_step(admission, index, geometry, boundary)
    save("matched_load_to_step", dict(matched, provenance=provenance))
    gc.collect()
    # Factories retain only original CPU input and create/return each full device state.
    def footprint(name):
        def run():
            return stage_factory(cpu, geometry, name, "full_step")()()
        return run
    boundary("isolated_footprint_start")
    memory = sealed.cuda_isolated_memory_v1({name: footprint(name) for name in ("native", "compact")},
        admission.adapter, admission.runtime, admission.launcher["lease"], parity)
    require(len(memory["samples"]) == 6, "Missing isolated three-per-implementation probes")
    memory = dict(memory, provenance=provenance, exact_boundary="CPU bank/model construction+transfer through full clipped AdamW step")
    save("isolated_allocator", memory)
    boundary("isolated_footprint_end")
    return dict(provenance=provenance, stages=records, matched_load_to_step=matched, isolated_allocator=memory)


@dataclass
class Dependencies:
    """Explicit handmade test injection. The CLI always uses production defaults."""
    boundary: Callable
    perform: Callable
    profile: Callable
    save: Callable


def production_dependencies(admission):
    def boundary(stage):
        recheck_runner(admission)
        return dict(stage=stage, **check_owned_storage(admission.output))
    def save(relative, record):
        path = canonical_path(admission.output / relative)
        require(not path.exists(), "Never overwrite preserved attempt evidence")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return Dependencies(boundary, lambda cell: perform_cell(admission, cell),
        lambda i, geometry, parity, callback: profile_cell(admission, i, geometry, parity, boundary, callback),
        save)


def run_qualified_v1(admission, *, dependencies=None):
    deps = dependencies if dependencies is not None else production_dependencies(admission)
    report = dict(kind="anchor_compact_qualification_runner_v1_attempt", status="running",
        binding=admission.binding, source_sha256=admission.source_sha256,
        actual_fitting=False, raw_tensors_or_checkpoints_retained=False,
        measured_speedup_claim=False, profile_blocked=True, storage_checks=[])
    try:
        report["storage_checks"].append(deps.boundary("before_first_array"))
        def perform(cell):
            report["storage_checks"].append(deps.boundary("numerical_cell_start"))
            result = deps.perform(cell)
            report["storage_checks"].append(deps.boundary("numerical_cell_end"))
            return result
        parity = qualify_cells(admission.adapter.selected, perform,
            lambda i, result: deps.save(f"cells/cell_{i:03d}.json", result))
        parity["binding"] = admission.adapter.binding
        report["parity"] = parity
        deps.save("parity.json", parity)
        require(parity["status"] == "PASS" and parity["all_frozen_cases_passed"],
                "Complete frozen numerical PASS required; profiling blocked")
        report["profile_blocked"] = False
        report["profiles"] = []
        for i in range(len(admission.adapter.selected)):
            for geometry in ("plain", "age"):
                report["storage_checks"].append(deps.boundary("profile_cell_start"))
                record = deps.profile(i, geometry, parity,
                    lambda stage, result, i=i, geometry=geometry: deps.save(
                        f"profiles/tile_{i}_{geometry}/{stage}.json", result))
                report["profiles"].append(record)
                report["storage_checks"].append(deps.boundary("profile_cell_end"))
        report["storage_checks"].append(deps.boundary("completion_source_and_storage_recheck"))
        report["status"] = "completed_descriptive_qualification"
    except Exception as error:
        report.update(status="FAIL", error_type=type(error).__name__, error=str(error))
        try:
            report["storage_checks"].append(deps.boundary("failure_final_recheck"))
        except Exception as final_error:
            report["final_recheck_failure"] = dict(error_type=type(final_error).__name__, error=str(final_error))
    deps.save("report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--admission", required=True)
    parser.add_argument("--admission-sha256", required=True)
    args = parser.parse_args(argv)
    admission = admit_runner_v1({"path": args.config, "sha256": args.config_sha256},
        {"path": args.admission, "sha256": args.admission_sha256})
    result = run_qualified_v1(admission)
    print(json.dumps({"status": result["status"], "profile_blocked": result["profile_blocked"],
                      "output_scope": str(admission.output), "measured_speedup_claim": False}))
    return 0 if result["status"] == "completed_descriptive_qualification" else 1


if __name__ == "__main__":
    raise SystemExit(main())
