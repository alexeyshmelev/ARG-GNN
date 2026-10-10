"""CPU/mocked launch-contract tests. Never create a CUDA context or controller."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from arggnn import anchor_compact_cuda_bootstrap_v2 as boot

PROJECT = Path('/workspace-SR003.nfs2/ARG-GNN')


@pytest.fixture
def admission_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(boot, "ROOT", tmp_path)
    monkeypatch.setattr(boot, "__file__", str(tmp_path / boot.SOURCE))
    monkeypatch.setattr(boot.time, "time", lambda: 1000.0)
    config_record = {"path": boot.PROTOCOL, "sha256": boot.PROTOCOL_SHA256}
    permit_record = {"path": "research/root_permit.json", "sha256": "a" * 64}
    output = tmp_path / "output"
    pool_log = tmp_path / ".runtime/gpup/p/jobs/j.log"
    reservations = [
        dict(id="own", bytes=1000000000, scope=[str(output)], status="reserved"),
        dict(id="pool_log", bytes=1048576, scope=[str(pool_log)], status="reserved")]
    storage = dict(kind="root_compact_runner_storage_v1", status="active_root_admission",
        authorizer="root_agent", project_root=str(tmp_path), runner_config=boot.RUNNER_CONFIG,
        issued_unix_seconds=999.0, reservation_scope=[str(output)],
        output_cache_log_cap_bytes=1000000000, reservation_active=True,
        reservation_expires_unix_seconds=2000.0, raw_tensors_or_checkpoints_retained=False,
        global_scan_used=False, accounting_scope="entire_repository_including_ignored_environment",
        unaccounted_writer_scope=[], warning_threshold_bytes=450000000000, hard_limit_bytes=500000000000,
        observed_repository_bytes=1000, outstanding_growth_reservations=reservations,
        current_reservation_inventory=copy.deepcopy(reservations),
        projected_repository_peak_bytes=1000 + 1000000000 + 1048576)
    uuid = "GPU-00000000-0000-0000-0000-000000000002"
    lease = dict(physical_gpu=2, visible_gpu_uuid=uuid, canonical_uuid=uuid[4:],
                 launcher_metadata=dict(launcher_pid=123, job="gpu_pool_v1:p:gpu2"))
    runtime = dict(python="3.11.17", torch="2.6.0+cu124", torch_cuda="12.4",
                   python_executable=str(tmp_path / ".runtime/env/bin/python"),
                   python_executable_sha256="a" * 64, driver_version="fixed")
    config = dict(kind="anchor_compact_cuda_bootstrap_protocol_v2",
        status="source_only_actual_execution_not_admitted", actual_execution_admitted=False,
        root_delegated_live_evidence_issuance_requires_fresh_permit=True,
        source_sha256={}, expected_runtime={k: v for k, v in runtime.items() if k != "driver_version"})
    source = {"path": boot.SOURCE, "sha256": "b" * 64}
    extension = dict(source={"path": "scripts/gpu_pool_research_v3.py", "sha256": "c" * 64},
        config={"path": "configs/gpu_pool_research_v3.json", "sha256": "d" * 64},
        independent_review={"path": "research/reviews/gpu_pool_research_v3_independent/review_receipt.json",
                            "sha256": "e" * 64})
    bootstrap_review = {"path": "research/reviews/anchor_compact_cuda_bootstrap_v2_independent/review_receipt.json",
                        "sha256": "f" * 64}
    experiment_record = {"path": "research/root_experiment.json", "sha256": "1" * 64}
    experiment = dict(kind="root_compact_runner_experiment_v1", status="active_root_admission",
        authorizer="root_agent", project_root=str(tmp_path), hypothesis_id="H020",
        fit_allowed=False, truth_allowed=False, ledger_consulted=True, decisions_registered=True,
        repeat_or_new_condition_reason="source fixture only", issued_unix_seconds=999.0, expires_unix_seconds=2000.0,
        ledger={"path": "research/experiment_ledger.md", "sha256": "2" * 64},
        decisions={"path": "research/experiment_decisions.json", "sha256": "3" * 64})
    permit = dict(kind="root_compact_cuda_bootstrap_v1_permit", status="active_root_admission",
        authorizer="root_agent", project_root=str(tmp_path), bootstrap_config=config_record,
        root_delegate_live_evidence_issuance=True, fit_allowed=False, truth_allowed=False,
        launch_after_prior_pools_natural_drain=True, issued_unix_seconds=999.0, expires_unix_seconds=2000.0,
        bootstrap_source=source, independent_bootstrap_review=bootstrap_review, pool_extension=extension,
        output_scope=str(output), pool="p", logical_job="j",
        controller_identity=dict(pid=123, uid=os.getuid(), start_ticks=77),
        lease=lease, storage_fields=storage, runtime_expected=runtime, experiment_admission=experiment_record,
        project_physical_gpus=[2], project_compute_identities=[], adapter_admission_fields={})
    records = {
        boot.PROTOCOL: config, permit_record["path"]: permit, experiment_record["path"]: experiment,
        bootstrap_review["path"]: dict(verdict="PASS", actual_execution_admitted=False,
            bootstrap_sha256=source["sha256"], bootstrap_config_sha256=config_record["sha256"]),
        extension["config"]["path"]: dict(allowed_modules=sorted(boot.pool.MODULES | {boot.RUNNER,
            "arggnn.anchor_compact_cuda_bootstrap_v2"}), bootstrap_worker=dict(source=source, config=config_record)),
        extension["independent_review"]["path"]: dict(verdict="PASS", actual_execution_admitted=False,
            launcher_sha256=extension["source"]["sha256"], config_sha256=extension["config"]["sha256"],
            bootstrap_sha256=source["sha256"], bootstrap_config_sha256=config_record["sha256"])}
    monkeypatch.setattr(boot, "read_bound", lambda r: copy.deepcopy(records[r["path"]]))
    monkeypatch.setattr(boot, "bound_bytes", lambda r: b"fixed source-only fixture")
    monkeypatch.setattr(boot.guard, "validate_lease", lambda: copy.deepcopy(lease))
    own = dict(pid=os.getpid(), uid=os.getuid(), start_ticks=88, ppid=123,
               pgid=os.getpid(), sid=os.getpid(), state="S")
    controller = dict(pid=123, uid=os.getuid(), start_ticks=77, ppid=1, pgid=123, sid=123, state="S")
    monkeypatch.setattr(boot.pool, "proc_identity", lambda pid: copy.deepcopy(own if pid == os.getpid() else controller))
    rows = [dict(index=i, uuid=f"GPU-00000000-0000-0000-0000-{i:012d}", used_mib=0,
                 total_mib=81920, utilization=0, compute=[]) for i in (2, 3, 4, 5)]
    monkeypatch.setattr(boot.pool, "real_inventory", lambda: rows)
    monkeypatch.setattr(boot, "check_runtime", lambda *a: None)
    permit["pool_submission"] = dict(job="j", model="H020_compact_qualification", seed=101,
        module="arggnn.anchor_compact_cuda_bootstrap_v2", output=str(output), peak_mib=12288,
        storage_bytes=1001048576, policy="reject", log_limit_bytes=1048576)
    expected_args = ["--config", config_record["path"], "--config-sha256", config_record["sha256"],
        "--permit", permit_record["path"], "--permit-sha256", permit_record["sha256"], "--output", str(output)]
    registry = dict(schema="gpu_pool_registry_v1", pool="p", lifecycle="running", physical_gpu=2,
        controller=controller, jobs=[dict(logical_job="j", state="running",
            requested=dict(permit["pool_submission"], args=expected_args),
            argv=[runtime["python_executable"], "-m", permit["pool_submission"]["module"], *expected_args],
            module_source_sha256=source["sha256"], pid=os.getpid(), leader_start_ticks=88)])
    monkeypatch.setattr(boot.pool, "read_json", lambda path: copy.deepcopy(registry))
    monkeypatch.setenv("ARGGNN_LOGICAL_JOB_NAME", "j")
    return SimpleNamespace(config_record=config_record, permit_record=permit_record, output=output,
        permit=permit, config=config, records=records, experiment=experiment, lease=lease, rows=rows,
        own=own, controller=controller, registry=registry, admit=lambda: boot.admit_precontext(config_record, permit_record, str(output)))


def test_valid_precontext_checks_create_no_output_or_context(admission_fixture, monkeypatch):
    fixture = admission_fixture
    with mock.patch.object(boot, "load_cuda_modules") as cuda, mock.patch.object(boot, "prepare_scope") as scope:
        result = fixture.admit()
    assert result["lease"] == fixture.lease
    assert not fixture.output.exists()
    cuda.assert_not_called()
    scope.assert_not_called()


@pytest.mark.parametrize("change", [
    ("status", "inactive_template_root_must_issue_fresh_receipt"), ("authorizer", "author"),
    ("root_delegate_live_evidence_issuance", 1), ("fit_allowed", 0), ("truth_allowed", 0),
    ("launch_after_prior_pools_natural_drain", False), ("issued_unix_seconds", True),
    ("issued_unix_seconds", 1001.0), ("issued_unix_seconds", 879.0),
    ("expires_unix_seconds", 1000.0), ("expires_unix_seconds", float("inf")),
    ("project_physical_gpus", [2, 3, 4]), ("project_physical_gpus", [0, 2]),
    ("project_physical_gpus", [True, 2]), ("logical_job", "another")])
def test_root_permit_rejections_precede_context(admission_fixture, change):
    fixture = admission_fixture
    fixture.permit[change[0]] = change[1]
    with mock.patch.object(boot, "load_cuda_modules") as cuda:
        with pytest.raises((ValueError, KeyError, TypeError)):
            fixture.admit()
    cuda.assert_not_called()
    assert not fixture.output.exists()


@pytest.mark.parametrize("target,key,value", [
    ("bootstrap_review", "verdict", "FAIL"), ("bootstrap_review", "bootstrap_sha256", "0" * 64),
    ("bootstrap_review", "actual_execution_admitted", True), ("extension_review", "config_sha256", "0" * 64),
    ("extension_review", "bootstrap_config_sha256", "0" * 64), ("experiment", "ledger_consulted", 1),
    ("experiment", "decisions_registered", False), ("experiment", "hypothesis_id", "H019"),
    ("experiment", "issued_unix_seconds", 800.0)])
def test_reviews_and_current_experiment_reject(admission_fixture, target, key, value):
    fixture = admission_fixture
    record = {"bootstrap_review": fixture.permit["independent_bootstrap_review"],
              "extension_review": fixture.permit["pool_extension"]["independent_review"],
              "experiment": fixture.permit["experiment_admission"]}[target]
    fixture.records[record["path"]][key] = value
    with pytest.raises(ValueError):
        fixture.admit()
    assert not fixture.output.exists()


@pytest.mark.parametrize("key,value", [
    ("output_cache_log_cap_bytes", 999999999), ("reservation_active", 1),
    ("reservation_expires_unix_seconds", 1000.0), ("projected_repository_peak_bytes", 449999999999),
    ("observed_repository_bytes", True), ("global_scan_used", True), ("unaccounted_writer_scope", ["missing"])])
def test_storage_content_and_type_rejections(admission_fixture, key, value):
    admission_fixture.permit["storage_fields"][key] = value
    with pytest.raises(ValueError):
        admission_fixture.admit()


def test_storage_exact_pool_log_and_reservation_inventory(admission_fixture):
    fixture = admission_fixture
    fixture.permit["storage_fields"]["outstanding_growth_reservations"][1]["bytes"] -= 1
    with pytest.raises(ValueError):
        fixture.admit()


def test_lease_parent_foreign_compute_and_incomplete_inventory_reject(admission_fixture, monkeypatch):
    fixture = admission_fixture
    fixture.own["pgid"] += 1
    with pytest.raises(ValueError):
        fixture.admit()
    fixture.own["pgid"] -= 1
    fixture.rows[0]["compute"] = [dict(pid=999, used_mib=1)]
    with pytest.raises(ValueError):
        fixture.admit()
    fixture.rows.pop()
    with pytest.raises(ValueError):
        fixture.admit()


def test_source_hash_path_type_symlink_and_size_rejection(tmp_path, monkeypatch):
    monkeypatch.setattr(boot, "ROOT", tmp_path)
    path = tmp_path / "source.py"
    path.write_bytes(b"sealed")
    record = {"path": str(path), "sha256": hashlib.sha256(b"sealed").hexdigest()}
    assert boot.bound_bytes(record) == b"sealed"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError):
        boot.bound_bytes(record)
    for wrong in (dict(record, sha256=True), dict(record, extra=1), dict(record, path="../source.py")):
        with pytest.raises(ValueError):
            boot.bound_bytes(wrong)
    link = tmp_path / "link.py"
    link.symlink_to(path)
    with pytest.raises(ValueError):
        boot.bound_bytes(dict(record, path=str(link)))
    path.write_bytes(b"x" * (1024 * 1024 + 1))
    with pytest.raises(ValueError):
        boot.bound_bytes(dict(record, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))


def mocked_run(fixture, monkeypatch):
    admission = fixture.admit()
    events, written = [], {}
    monkeypatch.setattr(boot, "prepare_scope", lambda output: events.append("scope"))
    monkeypatch.setattr(boot.guard, "apply_and_validate_title", lambda: events.append("title") or boot.guard.PROCESS_TITLE)
    torch = SimpleNamespace(__version__="2.6.0+cu124", version=SimpleNamespace(cuda="12.4"),
                            cuda=SimpleNamespace(is_initialized=lambda: False))
    policy = SimpleNamespace(DETERMINISTIC_POLICY={"strict": True})
    policy.configure_deterministic_development = lambda: events.append("policy") or policy.DETERMINISTIC_POLICY
    live = dict(fixture.permit["runtime_expected"], lease=fixture.lease, pid=os.getpid(),
        actual_process_title=boot.guard.PROCESS_TITLE, actual_context_uuid=fixture.lease["canonical_uuid"],
        execution_policy=policy.DETERMINISTIC_POLICY, logical_device_count=1, logical_device=0,
        nvtop_snapshot=[{}, {}, dict(processes=[dict(pid=os.getpid(), kind="compute", cmdline=boot.guard.PROCESS_TITLE)])])
    source_path = "src/arggnn/anchor_compact_qualification_runner_v1.py"
    fixture.config["source_sha256"][source_path] = "4" * 64
    admission["config"]["source_sha256"][source_path] = "4" * 64
    runner = SimpleNamespace(sealed=SimpleNamespace(), check_owned_storage=lambda output: events.append("storage"),
        observe_live=lambda launcher: events.append("live") or dict(live, nvtop_snapshot_record=launcher["nvtop_snapshot"]))
    def validate(runtime, storage, launcher, **kwargs):
        events.append("domains")
        assert runtime["source_sha256"] == {source_path: "4" * 64, boot.RUNNER_CONFIG["path"]: boot.RUNNER_CONFIG["sha256"]}
        assert storage == fixture.permit["storage_fields"]
        assert launcher["bootstrap_execution"]["root_permit"] == fixture.permit_record
        assert runtime["pid"] == launcher["pid"] == os.getpid()
    runner.validate_domains = validate
    def main(argv):
        events.append("runner")
        assert argv[:4] == ["--config", boot.RUNNER_CONFIG["path"], "--config-sha256", boot.RUNNER_CONFIG["sha256"]]
        assert argv[4] == "--admission" and argv[6] == "--admission-sha256"
        return 0
    runner.main = main
    monkeypatch.setattr(boot, "load_cuda_modules", lambda: events.append("import") or (torch, policy, runner))
    monkeypatch.setattr(boot, "validate_adapter_fields", lambda *a: events.append("metadata") or {"source_sha256": {}})
    monkeypatch.setattr(boot, "recheck_inputs", lambda a, initial=False: events.append("recheck_initial" if initial else "recheck"))
    monkeypatch.setattr(boot, "initialize_context", lambda torch: events.append("context"))
    monkeypatch.setattr(boot.subprocess, "run", lambda *a, **kw: events.append("nvtop") or SimpleNamespace(stdout=b"[]"))
    def write(output, name, value, raw=False):
        events.append("write:" + name)
        written[name] = copy.deepcopy(value)
        return {"path": str(output / ".launch" / name), "sha256": "5" * 64}
    monkeypatch.setattr(boot, "write_evidence", write)
    return admission, events, written, torch, live


def test_same_pid_title_policy_precontext_checks_and_immutable_receipt_shapes(admission_fixture, monkeypatch):
    admission, events, written, torch, live = mocked_run(admission_fixture, monkeypatch)
    assert boot.run_bootstrap(admission) == 0
    assert events.index("title") < events.index("import") < events.index("policy") < events.index("metadata")
    assert events.index("recheck_initial") < events.index("context") < events.index("nvtop") < events.index("live")
    assert events.index("domains") < events.index("runner")
    assert events[-2:] == ["recheck", "storage"]
    assert set(written) == {"nvtop_initial.json", "runtime_admission.json", "storage_admission.json",
                           "launcher_admission.json", "adapter_admission.json", "runner_admission.json"}
    for key in ("runtime_admission", "storage_admission", "launcher_admission"):
        assert written["adapter_admission.json"][key] == written["runner_admission.json"][key]


@pytest.mark.parametrize("failure", ["initialized", "runtime", "policy", "live_pid", "live_uuid", "live_title", "devices", "logical_device"])
def test_bootstrap_failures_block_runner_and_domain_issuance(admission_fixture, monkeypatch, failure):
    admission, events, written, torch, live = mocked_run(admission_fixture, monkeypatch)
    if failure == "initialized":
        torch.cuda.is_initialized = lambda: True
    elif failure == "runtime":
        torch.__version__ = "wrong"
    elif failure == "policy":
        def failed():
            raise RuntimeError("policy")
        # Failed completed-source/policy gate before context.
        monkeypatch.setattr(boot, "validate_adapter_fields", failed)
    elif failure == "live_pid":
        live["pid"] += 1
    elif failure == "live_uuid":
        live["actual_context_uuid"] = "00000000-0000-0000-0000-000000000099"
    elif failure == "logical_device":
        live["logical_device"] = 1
    elif failure == "live_title":
        live["actual_process_title"] = "wrong"
    else:
        live["nvtop_snapshot"].append(dict(processes=[dict(pid=999, kind="compute", cmdline=boot.guard.PROCESS_TITLE)]))
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        boot.run_bootstrap(admission)
    assert "runner" not in events
    assert "domains" not in events
    if failure in {"initialized", "runtime", "policy"}:
        assert "context" not in events


def test_context_initializer_single_visible_device_order(monkeypatch):
    events = []
    cuda = SimpleNamespace(is_initialized=lambda: False, device_count=lambda: 1,
        set_device=lambda value: events.append(("device", value)),
        init=lambda: events.append("init"), synchronize=lambda: events.append("sync"))
    torch = SimpleNamespace(cuda=cuda, empty=lambda *a, **kw: events.append(("empty", kw["device"])) or object())
    boot.initialize_context(torch)
    assert events == [("device", 0), "init", ("empty", "cuda:0"), "sync"]
    cuda.device_count = lambda: 2
    events.clear()
    with pytest.raises(ValueError):
        boot.initialize_context(torch)
    assert events == []


def test_owned_scope_cache_log_and_exclusive_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(boot, "ROOT", tmp_path)
    output = tmp_path / "owned"
    with mock.patch.object(boot.os, "dup2") as redirect:
        boot.prepare_scope(output)
    assert redirect.call_count == 2
    for name in boot.CACHE_PATHS:
        assert Path(os.environ[name]).is_relative_to(output)
        assert Path(os.environ[name]).is_dir()
    record = boot.write_evidence(output, "test.json", {"source_only": True})
    assert hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest() == record["sha256"]
    with pytest.raises(FileExistsError):
        boot.write_evidence(output, "test.json", {})
    with pytest.raises(FileExistsError):
        boot.prepare_scope(output)


def test_inactive_author_template_and_import_scope_are_inert():
    source = (PROJECT / boot.SOURCE).read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [n.name for n in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            assert all("torch" not in name and "qualification_runner" not in name for name in names)
    template = json.loads((PROJECT / "research/reviews/anchor_compact_cuda_bootstrap_v2_author/root_permit_template.json").read_text())
    assert template["status"] == "inactive_template_root_must_issue_fresh_receipt"
    assert template["root_delegate_live_evidence_issuance"] is False


@pytest.mark.parametrize("field,value", [("pid", True), ("uid", False), ("start_ticks", True), ("physical_gpu", True)])
def test_root_compute_identity_integer_types_reject(admission_fixture, field, value):
    fixture = admission_fixture
    fixture.permit["project_compute_identities"] = [dict(pid=123, uid=os.getuid(), start_ticks=77,
        physical_gpu=2, uuid=fixture.lease["visible_gpu_uuid"])]
    fixture.permit["project_compute_identities"][0][field] = value
    with pytest.raises(ValueError):
        fixture.admit()


def test_root_inventory_requires_current_compute_rows_on_all_project_devices(admission_fixture):
    fixture = admission_fixture
    fixture.permit["project_physical_gpus"] = [2, 3]
    fixture.rows[1]["compute"] = [dict(pid=999, used_mib=1)]
    with pytest.raises(ValueError):
        fixture.admit()


@pytest.mark.parametrize("key,value", [("peak_mib", 1), ("storage_bytes", 1000000000), ("seed", True),
    ("log_limit_bytes", 2000000), ("module", "arggnn.anchor_compact_qualification_runner_v1")])
def test_root_permit_binds_actual_registered_worker_reservation(admission_fixture, key, value):
    fixture = admission_fixture
    fixture.registry["jobs"][0]["requested"][key] = value
    with pytest.raises(ValueError):
        fixture.admit()


@pytest.mark.parametrize("key,value", [("pid", 999), ("leader_start_ticks", 99), ("module_source_sha256", "0" * 64),
    ("argv", ["python", "-m", "arbitrary"]), ("state", "succeeded")])
def test_registered_own_pid_argv_source_state_reject(admission_fixture, key, value):
    admission_fixture.registry["jobs"][0][key] = value
    with pytest.raises(ValueError):
        admission_fixture.admit()


@pytest.fixture
def handmade_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(boot, "ROOT", tmp_path)
    pins = dict(boot.METADATA_MANIFEST_SHA256)
    monkeypatch.setattr(boot, "METADATA_MANIFEST_SHA256", pins)
    def write(relative, payload):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        pins[relative] = digest
        return path, {"path": relative, "sha256": digest}
    return tmp_path, pins, write


@pytest.mark.parametrize("relative", list(boot.METADATA_MANIFEST_SHA256))
def test_original_metadata_paths_accept_complete_handmade_json_above_1mib(handmade_metadata, monkeypatch, relative):
    root, pins, write = handmade_metadata
    # A scalar-only handmade JSON string. No actual metadata or arrays are read.
    payload = json.dumps({"handmade_metadata_only": "x" * (1024 * 1024 + 257)}).encode()
    path, record = write(relative, payload)
    complete_hash_calls, read_limits = [], []
    real_sha256, real_decode, real_open = hashlib.sha256, json.loads, Path.open
    def hash_complete(data):
        complete_hash_calls.append((len(data), data == payload))
        return real_sha256(data)
    def decode_after_full_hash(data):
        assert complete_hash_calls == [(len(payload), True)]
        assert data == payload
        return real_decode(data)
    class TrackedFile:
        def __init__(self, file):
            self.file = file
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.file.close()
        def fileno(self):
            return self.file.fileno()
        def read(self, limit):
            read_limits.append(limit)
            return self.file.read(limit)
    monkeypatch.setattr(boot.hashlib, "sha256", hash_complete)
    monkeypatch.setattr(boot.json, "loads", decode_after_full_hash)
    monkeypatch.setattr(Path, "open", lambda self, *a, **kw: TrackedFile(real_open(self, *a, **kw)))
    result = boot.read_bound(record)
    assert len(result["handmade_metadata_only"]) == 1024 * 1024 + 257
    assert read_limits == [64 * 1024 * 1024 + 1]
    assert all(0 < limit <= 64 * 1024 * 1024 + 1 for limit in read_limits)


@pytest.mark.parametrize("relative", list(boot.METADATA_MANIFEST_SHA256))
def test_metadata_full_file_checksum_failure_never_decodes(handmade_metadata, monkeypatch, relative):
    root, pins, write = handmade_metadata
    payload = json.dumps({"handmade": "x" * (1024 * 1024 + 32)}).encode()
    path, record = write(relative, payload)
    changed = payload[:-1] + b" "
    path.write_bytes(changed)
    real_sha256 = hashlib.sha256
    hashed = []
    monkeypatch.setattr(boot.hashlib, "sha256", lambda data: hashed.append(data) or real_sha256(data))
    with mock.patch.object(boot.json, "loads") as decoder:
        with pytest.raises(ValueError, match="SHA256 differs"):
            boot.read_bound(record)
    assert hashed == [changed]
    decoder.assert_not_called()


@pytest.mark.parametrize("relative", list(boot.METADATA_MANIFEST_SHA256))
def test_wrong_metadata_declared_pin_rejects_before_decoder(handmade_metadata, relative):
    root, pins, write = handmade_metadata
    path, record = write(relative, b'{"handmade":true}')
    record["sha256"] = "0" * 64
    with mock.patch.object(boot.json, "loads") as decoder:
        with pytest.raises(ValueError, match="Pinned metadata manifest SHA256 differs"):
            boot.read_bound(record)
    decoder.assert_not_called()


@pytest.mark.parametrize("relative", list(boot.METADATA_MANIFEST_SHA256))
def test_64mib_plus_one_metadata_rejects_size_before_read_hash_or_decoder(handmade_metadata, monkeypatch, relative):
    root, pins, write = handmade_metadata
    path, record = write(relative, b"handmade fixture")
    with path.open("r+b") as handle:
        handle.truncate(64 * 1024 * 1024 + 1)
    real_open = Path.open
    reads = []
    class NoAllocationFile:
        def __init__(self, file):
            self.file = file
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.file.close()
        def fileno(self):
            return self.file.fileno()
        def read(self, limit):
            reads.append(limit)
            raise AssertionError("oversize stat must reject before read/allocation")
    monkeypatch.setattr(Path, "open", lambda self, *a, **kw: NoAllocationFile(real_open(self, *a, **kw)))
    with mock.patch.object(boot.json, "loads") as decoder, mock.patch.object(boot.hashlib, "sha256") as hasher:
        with pytest.raises(ValueError, match="size limit 67108864 bytes") as error:
            boot.read_bound(record)
    assert "SHA256" not in str(error.value) and reads == []
    decoder.assert_not_called()
    hasher.assert_not_called()


@pytest.mark.parametrize("relative", ["src/arggnn/source.py", "research/root_permit.json",
    "research/reviews/independent/review_receipt.json", "data/unlisted_manifest.json"])
def test_source_permit_review_and_unlisted_metadata_keep_1mib_cap(tmp_path, monkeypatch, relative):
    monkeypatch.setattr(boot, "ROOT", tmp_path)
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"x" * (1024 * 1024 + 1)
    path.write_bytes(payload)
    record = {"path": relative, "sha256": hashlib.sha256(payload).hexdigest()}
    with mock.patch.object(boot.json, "loads") as decoder, mock.patch.object(boot.hashlib, "sha256") as hasher:
        with pytest.raises(ValueError, match="size limit 1048576 bytes") as error:
            boot.read_bound(record)
    assert "SHA256" not in str(error.value)
    decoder.assert_not_called()
    hasher.assert_not_called()


def test_growth_after_size_observation_is_caught_by_bounded_sentinel_read(tmp_path, monkeypatch):
    monkeypatch.setattr(boot, "ROOT", tmp_path)
    path = tmp_path / "permit.json"
    payload = b"x" * (1024 * 1024 + 1)
    path.write_bytes(payload)
    record = {"path": "permit.json", "sha256": hashlib.sha256(payload).hexdigest()}
    monkeypatch.setattr(boot.os, "fstat", lambda fd: SimpleNamespace(st_size=1))
    with mock.patch.object(boot.json, "loads") as decoder, mock.patch.object(boot.hashlib, "sha256") as hasher:
        with pytest.raises(ValueError, match="size limit 1048576 bytes"):
            boot.read_bound(record)
    decoder.assert_not_called()
    hasher.assert_not_called()


def test_default_bounded_checksum_error_is_distinct_from_size_error(tmp_path, monkeypatch):
    monkeypatch.setattr(boot, "ROOT", tmp_path)
    path = tmp_path / "permit.json"
    path.write_bytes(b'{"handmade":true}')
    with mock.patch.object(boot.json, "loads") as decoder:
        with pytest.raises(ValueError, match="SHA256 differs") as error:
            boot.read_bound({"path": "permit.json", "sha256": "0" * 64})
    assert "size limit" not in str(error.value)
    decoder.assert_not_called()


def test_all_unchanged_bootstrap_helpers_match_frozen_v1_ast():
    old = ast.parse((PROJECT / "src/arggnn/anchor_compact_cuda_bootstrap_v1.py").read_text())
    new = ast.parse((PROJECT / "src/arggnn/anchor_compact_cuda_bootstrap_v2.py").read_text())
    class NormalizeIdentities(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                node.value = node.value.replace("anchor_compact_cuda_bootstrap_v2", "anchor_compact_cuda_bootstrap_v1")
                node.value = node.value.replace("anchor_compact_cuda_bootstrap_protocol_v2", "anchor_compact_cuda_bootstrap_protocol_v1")
                node.value = node.value.replace("gpu_pool_research_v3", "gpu_pool_research_v2")
            return node
    old_helpers = {node.name: ast.dump(node, include_attributes=False)
                   for node in old.body if isinstance(node, ast.FunctionDef)}
    new_helpers = {node.name: ast.dump(NormalizeIdentities().visit(node), include_attributes=False)
                   for node in new.body if isinstance(node, ast.FunctionDef)}
    assert set(old_helpers) == set(new_helpers)
    assert [name for name in old_helpers if old_helpers[name] != new_helpers[name]] == ["bound_bytes"]


def test_caps_and_original_metadata_pins_match_unchanged_adapter_config():
    adapter = json.loads((PROJECT / "configs/anchor_compact_actual_bank_v1.json").read_text())
    expected = {adapter[key]["path"]: adapter[key]["sha256"] for key in ("prepared_manifest", "observed_manifest")}
    assert boot.METADATA_MANIFEST_SHA256 == expected
    config = json.loads((PROJECT / boot.PROTOCOL).read_text())
    assert config["metadata_manifest_sha256"] == expected
    assert config["default_bound_file_limit_bytes"] == boot.DEFAULT_BOUND_FILE_LIMIT_BYTES == 1048576
    assert config["metadata_manifest_limit_bytes"] == boot.METADATA_MANIFEST_LIMIT_BYTES == 67108864
