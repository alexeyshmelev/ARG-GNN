"""Handmade CPU orchestration tests. No project banks, inference arrays or GPU calls."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from arggnn import anchor_compact_qualification_runner_v1 as m
from arggnn.development_execution import DETERMINISTIC_POLICY

@pytest.fixture(autouse=True)
def cpu_only():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    torch.set_num_threads(1)
    yield
    assert not torch.cuda.is_initialized()


def selected(n=3):
    return [dict(parent_id=f"handmade_train_{i}", tile_id=i, graphs=2 + i,
                 readouts=7 + i, selection_reasons=[[
                     "minimum_tree_count", "maximum_tree_count", "minimum_readout_count"][i]])
            for i in range(n)]


def passed(cell):
    names = m.expected_checks(cell)
    checks = {}
    for name in names:
        gradient = name.endswith(".VJP") or ".optimizer." in name or name.endswith(".post_AdamW") or name in (
            "preclip_norm", "input_feature_VJP")
        f32 = cell["device_condition"] == "cuda_float32" or name.endswith(".optimizer.step")
        rtol, atol = (((1e-3, 1e-5) if gradient else (2e-4, 2e-5)) if f32 else
                      ((1e-8, 1e-10) if gradient else (1e-10, 1e-12)))
        checks[name] = dict(elements=1, passed=True, max_absolute_error=0.,
                            max_acceptance_units=0., rtol=rtol, atol=atol)
    return dict(status="PASS", performed_request=dict(cell),
        seeded_constructor="qualification_models_v1" if cell["coupling"] == "full" else
                           "qualification_direct_control_models_v2",
        cell_identity=dict(geometry=cell["geometry"], coupling=cell["coupling"],
            native_geometry=cell["geometry"], compact_geometry=cell["geometry"],
            native_coupling=cell["coupling"], compact_coupling=cell["coupling"],
            dtype="torch.float64" if cell["device_condition"] == "cpu_float64" else "torch.float32",
            device="cpu" if cell["device_condition"] == "cpu_float64" else "cuda:0",
            graphs=cell["graphs"], readouts=cell["readouts"], Q=184, R=96,
            anchors=32, scales=3, latent_dim=8, kernel_depth=2, probe=cell["probe"]),
        checks=checks, all_Q_checked=True, labels_opened=False,
        input_models_modified=False, preclip_native_norm=2., preclip_compact_norm=2.)


def admitted():
    return SimpleNamespace(adapter=SimpleNamespace(selected=tuple(selected()),
               binding={"handmade_adapter_binding": True}), binding={"handmade": True},
               source_sha256={"handmade": "a" * 64})


@pytest.mark.parametrize("n", [1, 2, 3])
def test_exact_complete_plan(n):
    rows = selected(n)
    if n == 1:
        rows[0]["selection_reasons"] += ["maximum_tree_count", "minimum_readout_count"]
    cells = m.frozen_cells(rows)
    assert len(cells) == 8 * n + 10
    assert len({json.dumps(x, sort_keys=True) for x in cells}) == len(cells)
    assert sum(c["state"] == "rate_mixture_stress" for c in cells) == 4
    assert sum(c["coupling"] != "full" for c in cells) == 6
    assert all(c["seed"] == 101 for c in cells)
    assert all(c["selection_index"] == 0 and c["geometry"] == "age" for c in cells
               if c["state"] != "initial" or c["coupling"] != "full")


@pytest.mark.parametrize("probe", m.PROBES)
@pytest.mark.parametrize("device", m.DEVICES)
def test_complete_returned_identity_and_checks(probe, device):
    cell = next(c for c in m.frozen_cells(selected()) if c["probe"] == probe and c["device_condition"] == device)
    m.validate_performed(cell, passed(cell))


@pytest.mark.parametrize("field", ["geometry", "coupling", "native_geometry", "compact_geometry",
    "native_coupling", "compact_coupling", "dtype", "device", "graphs", "readouts",
    "Q", "R", "anchors", "scales", "latent_dim", "kernel_depth", "probe"])
def test_requested_labels_cannot_replace_returned_performed_identity(field):
    cell = m.frozen_cells(selected())[0]
    result = passed(cell)
    result["cell_identity"][field] = "wrong"
    with pytest.raises(ValueError, match="helper identity"):
        m.validate_performed(cell, result)


@pytest.mark.parametrize("field", ["seed", "state", "parent_id", "tile_id", "selection_index"])
def test_seed_state_tile_performed_request_cannot_drift(field):
    cell = m.frozen_cells(selected())[0]
    result = passed(cell)
    result["performed_request"][field] = "wrong"
    with pytest.raises(ValueError, match="tile/seed/state"):
        m.validate_performed(cell, result)


@pytest.mark.parametrize("failure", ["missing_family", "loosened_tolerance", "nonfinite",
    "contradictory_flag", "nonpositive_elements", "no_active_clipping", "labels", "wrong_constructor"])
def test_incomplete_or_false_pass_rejected(failure):
    cell = next(c for c in m.frozen_cells(selected()) if c["probe"] == m.PROBES[1])
    result = passed(cell)
    name = next(iter(result["checks"]))
    if failure == "missing_family":
        del result["checks"][name]
    elif failure == "loosened_tolerance":
        result["checks"][name]["rtol"] = .5
    elif failure == "nonfinite":
        result["checks"][name]["max_absolute_error"] = float("nan")
    elif failure == "contradictory_flag":
        result["checks"][name]["max_acceptance_units"] = 2.
    elif failure == "nonpositive_elements":
        result["checks"][name]["elements"] = 0
    elif failure == "no_active_clipping":
        result["preclip_native_norm"] = 1.
    elif failure == "labels":
        result["labels_opened"] = True
    else:
        result["seeded_constructor"] = "full_model_labeled_as_control"
    with pytest.raises(ValueError):
        m.validate_performed(cell, result)


@pytest.mark.parametrize("failure", ["numerical", "exception", "wrong_returned_control"])
def test_failures_preserved_and_no_profile_calls(failure):
    receipts, profiles = {}, []
    calls = []
    def perform(cell):
        calls.append(cell)
        result = passed(cell)
        if len(calls) == 1:
            if failure == "exception":
                raise RuntimeError("handmade failure before probe")
            if failure == "numerical":
                name = next(iter(result["checks"]))
                result["status"] = "FAIL"
                result["checks"][name].update(passed=False, max_acceptance_units=2.)
            else:
                result["cell_identity"]["compact_coupling"] = "scalar"
        return result
    deps = m.Dependencies(lambda stage: {"stage": stage}, perform,
        lambda *args: profiles.append(args), lambda name, r: receipts.setdefault(name, deepcopy(r)))
    result = m.run_qualified_v1(admitted(), dependencies=deps)
    assert result["status"] == "FAIL" and result["profile_blocked"]
    assert len(calls) == 34 and len(result["parity"]["records"]) == 34
    assert not profiles
    assert receipts["cells/cell_000.json"]["validation"] == "FAIL"
    assert receipts["report.json"]["status"] == "FAIL"
    assert receipts["parity.json"]["status"] == "FAIL"


def test_profile_only_after_all34_pass_and_final_boundary_rechecked():
    events, saved = [], {}
    def boundary(stage):
        events.append(stage)
        return {"stage": stage}
    def profile(i, geometry, parity, save):
        assert parity["status"] == "PASS" and len(parity["records"]) == 34
        assert parity["binding"] == admitted().adapter.binding
        events.append((i, geometry))
        save("handmade", {"scope": "source_semantics_only"})
        return {"source_semantics_only": True}
    deps = m.Dependencies(boundary, passed, profile, lambda name, r: saved.setdefault(name, deepcopy(r)))
    result = m.run_qualified_v1(admitted(), dependencies=deps)
    assert result["status"] == "completed_descriptive_qualification"
    assert len(result["profiles"]) == 6 and not result["profile_blocked"]
    assert events[-1] == "completion_source_and_storage_recheck"
    assert len([name for name in saved if name.startswith("cells/")]) == 34
    assert not result["measured_speedup_claim"]


def test_final_source_change_preserved():
    saved = {}
    def boundary(stage):
        if stage in ("completion_source_and_storage_recheck", "failure_final_recheck"):
            raise ValueError("handmade source changed")
        return {"stage": stage}
    result = m.run_qualified_v1(admitted(), dependencies=m.Dependencies(
        boundary, passed, lambda *args: {}, lambda name, r: saved.setdefault(name, deepcopy(r))))
    assert result["status"] == "FAIL" and "final_recheck_failure" in result


def domains():
    output = m.ROOT / "research/reviews/anchor_compact_qualification_runner_v1_author/handmade_output"
    config = {"path": m.PROTOCOL, "sha256": "a" * 64}
    source = {"handmade": "b" * 64}
    lease = {"physical_gpu": 2, "canonical_uuid": "00000000-0000-0000-0000-000000000002",
             "visible_gpu_uuid": "GPU-00000000-0000-0000-0000-000000000002"}
    common = dict(status="active_root_admission", authorizer="root_agent",
                  project_root=str(m.ROOT), runner_config=config, issued_unix_seconds=990.)
    own = dict(pid=os.getpid(), kind="compute", cmdline=m.sealed.PROCESS_TITLE)
    snapshot = [{"processes": []} for _ in range(7)]
    snapshot[2]["processes"] = [own]
    snapshot_record = {"path": "handmade_nvtop.json", "sha256": "c" * 64}
    live = dict(python="handmade_python", torch="handmade_torch", torch_cuda="handmade_cuda",
        driver_version="handmade_driver", python_executable=str(m.ROOT / ".runtime/env/bin/python"),
        python_executable_sha256="d" * 64, actual_process_title=m.sealed.PROCESS_TITLE,
        pid=os.getpid(), actual_context_uuid=lease["canonical_uuid"], device="cuda",
        execution_policy=deepcopy(DETERMINISTIC_POLICY), logical_device_count=1, logical_device=0,
        lease=lease, nvtop_snapshot=snapshot, nvtop_snapshot_record=snapshot_record)
    runtime = dict(common, kind="root_compact_runner_runtime_v1", source_sha256=source,
        **{k: v for k, v in live.items() if k not in (
            "logical_device_count", "logical_device", "lease", "nvtop_snapshot", "nvtop_snapshot_record")})
    reservations = [
        dict(id="handmade_other", bytes=20000000000, scope=[str(m.ROOT / "handmade_other")], status="reserved"),
        dict(id="handmade_runner", bytes=1000000000, scope=[str(output)], status="reserved")]
    storage = dict(common, kind="root_compact_runner_storage_v1", reservation_scope=[str(output)],
        output_cache_log_cap_bytes=1000000000, raw_tensors_or_checkpoints_retained=False,
        global_scan_used=False, reservation_active=True, reservation_expires_unix_seconds=2000.,
        accounting_scope="entire_repository_including_ignored_environment", unaccounted_writer_scope=[],
        warning_threshold_bytes=450000000000, hard_limit_bytes=500000000000,
        observed_repository_bytes=410000000000, outstanding_growth_reservations=reservations,
        projected_repository_peak_bytes=431000000000, current_reservation_inventory=deepcopy(reservations))
    launcher = dict(common, kind="root_compact_runner_launcher_v1", lease=lease, pid=os.getpid(),
        actual_context_uuid=lease["canonical_uuid"], title_before_cuda=m.sealed.PROCESS_TITLE,
        title_before_cuda_verified=True, root_pool_runner_allowlist_integrated=True,
        launch_after_prior_pools_natural_drain=True, nvtop_snapshot=snapshot_record,
        nvtop_exit_code=0, nvtop_executable=str(m.ROOT / ".runtime/env/bin/nvtop"),
        project_physical_gpus=[2])
    return runtime, storage, launcher, dict(config_record=config, source_sha256=source,
                                           output=output, live=live, now=1000.)


def test_content_validated_with_exact_storage_reservation_and_live_identity():
    runtime, storage, launcher, args = domains()
    m.validate_domains(runtime, storage, launcher, **args)


@pytest.mark.parametrize("failure", ["old_pid", "UUID", "TF32", "missing_source", "stale",
    "expired", "bad_arithmetic", "warn_threshold", "missing_scope", "unaccounted",
    "third_GPU", "excluded_GPU", "own_title", "title_before_cuda", "pool_not_integrated",
    "wrong_inventory", "no_actual_NVTOP", "shared_cache_scope", "missing_reservation"])
def test_domain_contents_fail_closed_even_if_receipt_hash_were_valid(failure):
    runtime, storage, launcher, args = domains()
    if failure == "old_pid":
        runtime["pid"] = os.getpid() + 1
    elif failure == "UUID":
        runtime["actual_context_uuid"] = "wrong"
    elif failure == "TF32":
        runtime["execution_policy"]["cuda_matmul_allow_tf32"] = True
    elif failure == "missing_source":
        runtime["source_sha256"] = {}
    elif failure == "stale":
        storage["issued_unix_seconds"] = 0.
    elif failure == "expired":
        storage["reservation_expires_unix_seconds"] = 999.
    elif failure == "bad_arithmetic":
        storage["projected_repository_peak_bytes"] += 1
    elif failure == "warn_threshold":
        storage["observed_repository_bytes"] = 429000000000
        storage["projected_repository_peak_bytes"] = 450000000000
    elif failure == "missing_scope":
        storage["outstanding_growth_reservations"][0]["scope"] = []
    elif failure == "unaccounted":
        storage["unaccounted_writer_scope"] = ["handmade_unknown"]
    elif failure in ("third_GPU", "excluded_GPU"):
        devices = [2, 3, 4] if failure == "third_GPU" else [0, 2]
        for physical in devices:
            args["live"]["nvtop_snapshot"][physical]["processes"].append(
                dict(pid=os.getpid() + physical + 1, kind="compute", cmdline=m.sealed.PROCESS_TITLE))
        launcher["project_physical_gpus"] = devices
    elif failure == "own_title":
        args["live"]["nvtop_snapshot"][2]["processes"][0]["cmdline"] = "wrong"
    elif failure == "title_before_cuda":
        launcher["title_before_cuda_verified"] = False
    elif failure == "pool_not_integrated":
        launcher["root_pool_runner_allowlist_integrated"] = False
    elif failure == "wrong_inventory":
        storage["current_reservation_inventory"] = []
    elif failure == "no_actual_NVTOP":
        launcher["nvtop_exit_code"] = 1
    elif failure == "shared_cache_scope":
        storage["reservation_scope"] = [str(m.ROOT)]
    else:
        storage["outstanding_growth_reservations"].pop()
    with pytest.raises((ValueError, KeyError)):
        m.validate_domains(runtime, storage, launcher, **args)


def test_scope_path_never_falls_back_outside_repo(tmp_path):
    with pytest.raises(ValueError):
        m.canonical_path(tmp_path)
    with pytest.raises(ValueError):
        m.canonical_path(m.ROOT / ".." / "escape")


def pair_samples():
    rows = []
    for repetition in range(10):
        order = ["native", "compact"] if repetition % 2 == 0 else ["compact", "native"]
        for index, implementation in enumerate(order):
            rows.append(dict(implementation=implementation, repetition=repetition, order_index=index,
                synchronized_wall_seconds=2. if implementation == "native" else 1.,
                CUDA_event_milliseconds=2., baseline_allocated_bytes=3,
                peak_allocated_bytes=5, incremental_peak_allocated_bytes=2))
    return {"samples": rows, "stage": "full_step"}


def test_all_raw_ABBA_ratios_and_median_min_max_retained():
    report = pair_samples()
    result = m.summarize_pair(report)
    assert result["samples"] == report["samples"]
    assert result["paired_native_over_compact_wall_ratios"] == [2.] * 10
    assert result["paired_ratio_summary"] == dict(median=2., min=2., max=2.)
    assert not result["measured_speedup_claim"]


@pytest.mark.parametrize("failure", ["missing_sample", "wrong_ABBA", "nonfinite"])
def test_malformed_profile_samples_fail(failure):
    report = pair_samples()
    if failure == "missing_sample":
        report["samples"].pop()
    elif failure == "wrong_ABBA":
        report["samples"][0]["implementation"] = "compact"
    else:
        report["samples"][0]["synchronized_wall_seconds"] = float("nan")
    with pytest.raises(ValueError):
        m.summarize_pair(report)


def test_inactive_root_template_rejected_before_any_live_observation(monkeypatch):
    template = json.loads((m.ROOT / "research/reviews/anchor_compact_qualification_runner_v1_author/root_admission_template.json").read_text())
    monkeypatch.setattr(m, "read_bound", lambda record: m.EXPECTED_CONFIG if record["path"] == m.PROTOCOL else template)
    with pytest.raises(ValueError, match="active root runner admission"):
        m.admit_runner_v1({"path": m.PROTOCOL, "sha256": "a" * 64},
            {"path": "handmade_inactive", "sha256": "b" * 64},
            live_observer=lambda _: pytest.fail("inactive template must not inspect CUDA"))


def test_retained_storage_alias_count_is_not_allocator_memory():
    tensor = torch.ones(3, dtype=torch.float64)
    assert m.retained_bytes({"a": tensor, "alias": tensor[1:]}) == 24
