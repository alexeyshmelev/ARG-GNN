"""Focused source/config and v1 IPC fixtures. No live controller, lease or GPU."""
import contextlib
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import tempfile
import threading
from types import SimpleNamespace
from unittest import mock

import pytest
from arggnn import gpu_pool_v1 as pool

PROJECT = Path("/workspace-SR003.nfs2/ARG-GNN")
spec = importlib.util.spec_from_file_location("gpu_pool_research_v2", PROJECT / "scripts/gpu_pool_research_v2.py")
extension = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extension)


@pytest.fixture
def bound_fixture(tmp_path):
    cfg = json.loads((PROJECT / extension.CONFIG).read_text())
    records = [cfg["base_controller"], cfg["extension_worker"]["source"], cfg["extension_worker"]["config"],
               cfg["extension_worker"]["independent_review"], cfg["bootstrap_worker"]["source"],
               cfg["bootstrap_worker"]["config"], {"path": extension.CONFIG}]
    records += [{"path": path} for path in cfg["unchanged_entrypoint_and_guards"]]
    for record in records:
        path = tmp_path / record["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((PROJECT / record["path"]).read_bytes())
    return tmp_path, cfg


def submit_argv(module=None):
    return ["submit", "--pool", "p", "--job", "j", "--model", "H020_compact_qualification",
            "--seed", "101", "--output", "research/outputs/fresh", "--peak-mib", "12288",
            "--storage-bytes", "1001048576", "--policy", "reject", "--log-limit-bytes", "1048576",
            module or extension.BOOTSTRAP, "--config", "configs/anchor_compact_cuda_bootstrap_v1.json",
            "--config-sha256", "fixed", "--permit", "research/fresh_root_permit.json",
            "--permit-sha256", "fixed", "--output", "research/outputs/fresh"]


def start_argv():
    return ["start", "--pool", "p", "--gpu", "2", "--headroom-mib", "8192",
            "--storage-snapshot", "research/storage.json", "--storage-snapshot-sha256", "a" * 64,
            "--external-storage-reserve-bytes", "1000000000"]


def test_exact_allowlist_bindings_and_no_global_rebinding(bound_fixture):
    root, cfg = bound_fixture
    original = pool.MODULES
    assert extension.verify_bindings(root) == cfg
    assert extension.MODULES == pool.MODULES | {extension.WORKER, extension.BOOTSTRAP}
    assert pool.MODULES is original
    assert extension.WORKER not in original and extension.BOOTSTRAP not in original
    assert cfg["actual_execution_admitted"] is False


@pytest.mark.parametrize("path", [
    extension.CONFIG, "src/arggnn/gpu_pool_v1.py", "scripts/gpu_pool_v1.py", "scripts/launch_gpu.py",
    "src/arggnn/native_argformer_exec.py", "src/arggnn/anchor_compact_qualification_runner_v1.py",
    "configs/anchor_compact_qualification_runner_v1.json",
    "research/reviews/anchor_compact_qualification_runner_v1_independent/review_receipt.json",
    "src/arggnn/anchor_compact_cuda_bootstrap_v1.py", "configs/anchor_compact_cuda_bootstrap_v1.json"])
def test_each_pinned_source_config_review_rejects_mutation(bound_fixture, path):
    root, cfg = bound_fixture
    with (root / path).open("ab") as handle:
        handle.write(b"\nchanged\n")
    with pytest.raises(ValueError, match="SHA256 differs"):
        extension.verify_bindings(root)


def test_unknown_module_direct_runner_and_config_failure_never_reach_ipc(monkeypatch):
    with mock.patch.object(pool, "request") as ipc:
        with pytest.raises(SystemExit):
            extension.main(submit_argv("arggnn.arbitrary"))
        with pytest.raises(ValueError, match="Direct runner"):
            extension.main(submit_argv(extension.WORKER))
        monkeypatch.setattr(extension, "verify_bindings", mock.Mock(side_effect=ValueError("binding")))
        with pytest.raises(ValueError, match="binding"):
            extension.main(submit_argv())
    ipc.assert_not_called()


def test_start_selects_exact_unchanged_class_with_explicit_modules(monkeypatch):
    controller = SimpleNamespace(start=lambda: controller, status=lambda: {"source_only_mock": True},
                                 run=mock.Mock(), shutdown_requested=False)
    with mock.patch.object(pool, "Controller", return_value=controller) as constructor, \
            mock.patch.object(extension.signal, "signal") as handlers:
        assert extension.main(start_argv()) == 0
        kwargs = constructor.call_args.kwargs
        assert kwargs == dict(root=pool.ROOT, pool="p", gpu=2, headroom_mib=8192,
            storage_snapshot="research/storage.json", storage_snapshot_sha256="a" * 64,
            external_storage_reserve_bytes=1000000000,
            controller_storage_reserve_bytes=pool.MIN_CONTROLLER_RESERVE,
            storage_limit_bytes=450000000000, modules=extension.MODULES)
        handlers.call_args.args[1](15, None)
        assert controller.shutdown_requested is True
    controller.run.assert_called_once_with()


def test_start_binding_failure_precedes_constructor(monkeypatch):
    monkeypatch.setattr(extension, "verify_bindings", mock.Mock(side_effect=ValueError("changed")))
    with mock.patch.object(pool, "Controller") as constructor:
        with pytest.raises(ValueError):
            extension.main(start_argv())
    constructor.assert_not_called()


def test_submit_preserves_literal_v1_request_and_reservations():
    argv = submit_argv()
    with mock.patch.object(pool, "request", return_value={"mock": True}) as request:
        assert extension.main(argv) == 0
    path, request_value = request.call_args.args
    assert path == pool.ROOT / ".runtime/gpup/p.sock"
    assert request_value == dict(v=1, op="submit", spec=dict(job="j", model="H020_compact_qualification",
        seed=101, output="research/outputs/fresh", peak_mib=12288, storage_bytes=1001048576,
        policy="reject", log_limit_bytes=1048576, module=extension.BOOTSTRAP,
        args=argv[argv.index(extension.BOOTSTRAP) + 1:]))


@pytest.mark.parametrize("argv", [["status", "--pool", "p"], ["wait", "--pool", "p", "--job", "j"],
                                  ["shutdown", "--pool", "p"], ["shutdown", "--pool", "p", "--terminate"]])
def test_status_wait_shutdown_are_unchanged_clients(argv):
    with mock.patch.object(pool, "main", return_value=17) as main, \
            mock.patch.object(extension, "verify_bindings") as verify:
        assert extension.main(argv) == 17
    main.assert_called_once_with(argv)
    verify.assert_not_called()


def bare_controller(tmp_path):
    controller = pool.Controller.__new__(pool.Controller)
    controller.root, controller.modules = tmp_path, extension.MODULES
    controller.jobs, controller.headroom = {}, 8192
    controller.device = dict(used_mib=0, total_mib=81920)
    (tmp_path / "outputs").mkdir()
    return controller


def worker_spec(tmp_path):
    return dict(job="j", model="H020_compact_qualification", seed=101, module=extension.BOOTSTRAP,
                args=["--config", "fixed", "--permit", "inactive"], output=str(tmp_path / "outputs/new"),
                peak_mib=12288, storage_bytes=1001048576, policy="reject", log_limit_bytes=1048576)


@contextlib.contextmanager
def ipc_dispatch(controller):
    # An isolated AF_UNIX fixture, no production .runtime path or Controller.start.
    parent = PROJECT / "research/reviews/gpu_pool_research_v2_independent/t"
    parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="", dir=parent) as directory:
        path = Path(directory) / "s"
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(path))
        server.listen(1)
        failures = []
        def serve():
            try:
                with server.accept()[0] as connection:
                    request = bytearray()
                    while b"\n" not in request:
                        request.extend(connection.recv(8192))
                    try:
                        result = dict(ok=True, result=controller.dispatch(json.loads(request.split(b"\n")[0])))
                    except Exception as error:
                        result = dict(ok=False, error=f"{type(error).__name__}: {error}")
                    connection.sendall(pool.encoded(result) + b"\n")
            except Exception as error:
                failures.append(error)
        thread = threading.Thread(target=serve)
        thread.start()
        try:
            yield path
        finally:
            thread.join(5)
            server.close()
            assert not thread.is_alive() and not failures


def test_real_v1_ipc_accepts_only_reviewed_extension_spec_without_execution(tmp_path):
    controller = bare_controller(tmp_path)
    controller.submit = controller._validate_spec
    original = worker_spec(tmp_path)
    with ipc_dispatch(controller) as path:
        result = pool.request(path, dict(v=1, op="submit", spec=original))
    assert result == original
    assert controller.jobs == {} and not Path(result["output"]).exists()


@pytest.mark.parametrize("change", [
    ("module", "arggnn.arbitrary"), ("seed", True), ("peak_mib", True), ("storage_bytes", 1.0),
    ("log_limit_bytes", True), ("args", "--not-a-vector"), ("args", ["bad\0argument"]),
    ("output", "../outside"), ("output", "outputs/../escape"), ("policy", "optin"), ("extra", "unreviewed")])
def test_v1_ipc_retains_base_type_path_argument_rejection(tmp_path, change):
    controller = bare_controller(tmp_path)
    controller.submit = controller._validate_spec
    value = worker_spec(tmp_path)
    value[change[0]] = change[1]
    with ipc_dispatch(controller) as path:
        with pytest.raises(RuntimeError):
            pool.request(path, dict(v=1, op="submit", spec=value))
    assert controller.jobs == {} and not (tmp_path / "outputs/new").exists()


@pytest.mark.parametrize("version", [True, "1", 2])
def test_v1_ipc_protocol_type_is_not_relaxed(tmp_path, version):
    controller = bare_controller(tmp_path)
    with ipc_dispatch(controller) as path:
        with pytest.raises(RuntimeError, match="Protocol version"):
            pool.request(path, dict(v=version, op="submit", spec=worker_spec(tmp_path)))


@pytest.mark.parametrize("change", ["worker_source", "output"])
def test_base_rechecks_new_bootstrap_source_and_output_before_popen(tmp_path, change):
    controller = bare_controller(tmp_path)
    source = tmp_path / "src/arggnn/anchor_compact_cuda_bootstrap_v1.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"sealed bootstrap fixture")
    spec = worker_spec(tmp_path)
    job = dict(state="queued", requested=spec,
               module_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    controller.jobs = {"j": job}
    controller.lifecycle, controller.safety_error = "running", None
    controller._process_scan = mock.Mock()
    controller._memory = mock.Mock(return_value={"blocked_by": []})
    controller._lease_check = mock.Mock()
    if change == "worker_source":
        source.write_bytes(b"changed")
    else:
        Path(spec["output"]).mkdir()
    with mock.patch.object(pool.subprocess, "Popen") as popen:
        controller._launch_queue()
    assert job["state"] == "failed" and "changed" in job["error"]
    popen.assert_not_called()
    controller._lease_check.assert_not_called()
