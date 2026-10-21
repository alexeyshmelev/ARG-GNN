"""Explicit operational bindings around unchanged admitted ARGMix functions.

No scientific function is copied or replaced. Each temporary I/O binding is
restored in finally. The worker owns one scientific call at a time.
"""
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import hashlib
import importlib
import logging
import os
from pathlib import Path
import sys
import time
import traceback

from arggnn.benchmarking import european_argmix_release as release
from arggnn.benchmarking import european_argmix_shards as shards

m = release.m
native = release.native
GPU_UUID = "GPU-3db85505-d061-e45d-d154-e7e8c1cad497"
TITLE = "AI Auto Research Pipeline Test"
STARTUP = Path(__file__).with_name("sitecustomize_gpu5_v1.py")
STARTUP_SHA256 = "9dda9d73110ca488b09d9ec0dfdcf5e4040ff64759b6dc01a04da89cbdcf9f7a"
ORIGINAL_ADMIT = release.admit_native
ORIGINAL_TRAIN_RUNNER = shards.run_original_main
ORIGINAL_STARTUP = release._native_python_startup
ORIGINAL_LOCAL_RUNNER = release._native_local_runner
ORIGINAL_AUTHOR_RUNNER = native.run_author


def require_gpu5_environment():
    m._need(os.environ.get("CUDA_VISIBLE_DEVICES") == GPU_UUID
        and os.environ.get("ARGGNN_PHYSICAL_GPU") == "5",
        "Exact ROOT-admitted physical GPU5 UUID and assignment required before CUDA")
    import setproctitle
    m._need(setproctitle.getproctitle() == TITLE,
        "Author code must run inside the correctly titled GPU worker")


def admit_native(build_ref, cpu_ref, *, require_gpu=False):
    if not require_gpu:
        return ORIGINAL_ADMIT(build_ref, cpu_ref, require_gpu=False)
    require_gpu5_environment()
    native.runtime_admission(build_ref, cpu_ref, {})
    result = native.source_imports()
    m._need(result[0].cuda.device_count() == 1,
        "ROOT-owned single physical GPU5 admission required")
    cap = int(os.environ.get("ARGGNN_MAX_CUDA_RESERVED_BYTES", "0"))
    total = result[0].cuda.get_device_properties(0).total_memory
    m._need(0 < cap <= min(74 * 2**30, total),
        "Finite ROOT-admitted CUDA allocator cap required")
    result[0].cuda.set_per_process_memory_fraction(cap / total, 0)
    return result


def _detach_logging():
    loggers = [logging.getLogger()] + [value for value in
        logging.Logger.manager.loggerDict.values() if isinstance(value, logging.Logger)]
    saved = {value: (value.handlers[:], value.level, value.propagate, value.disabled)
        for value in loggers}
    old_handlers = {handler for value in loggers for handler in value.handlers}
    for value in loggers:
        value.handlers = []
    return saved, old_handlers


def _restore_logging(saved, old_handlers):
    active = [logging.getLogger()] + [value for value in
        logging.Logger.manager.loggerDict.values() if isinstance(value, logging.Logger)]
    for value in active:
        for handler in value.handlers[:]:
            value.removeHandler(handler)
            if handler not in old_handlers:
                target = getattr(handler, "stream", None)
                if target is None or not getattr(target, "closed", False):
                    handler.flush()
                handler.close()
        if value in saved:
            value.handlers, value.level, value.propagate, value.disabled = saved[value]


def run_original_main(author_module, argv, output, *, guard, execution_check):
    """Call the exact imported main, temporarily binding its archive I/O name."""
    original = author_module.main
    m._need(original.__globals__ is vars(author_module)
        and Path(author_module.__file__).resolve() == native.SOURCE / "src/train_joint_model.py",
        "Exact imported admitted trainer main and module namespace required")
    original_code = original.__code__
    old_loader = author_module._load_gt_gru_split_pt

    def load_split(path, expected_split_code):
        return shards.load_sharded_split(path, expected_split_code=expected_split_code,
            guard=guard, execution_check=execution_check)

    old_argv = sys.argv[:]
    saved, old_handlers = _detach_logging()
    started = time.monotonic()
    code = 0
    try:
        author_module._load_gt_gru_split_pt = load_split
        sys.argv = [str(Path(author_module.__file__)), *map(str, argv)]
        with (output / "train_joint_model.log").open("x") as stream, \
                redirect_stdout(stream), redirect_stderr(stream):
            try:
                original()
            except SystemExit as error:
                code = error.code if isinstance(error.code, int) else (0 if error.code is None else 1)
                if code:
                    raise
    except BaseException:
        code = 1
        raise
    finally:
        author_module._load_gt_gru_split_pt = old_loader
        sys.argv = old_argv
        _restore_logging(saved, old_handlers)
        m._json(output / "train_joint_model.command.json", dict(
            source_ref=m._ref(author_module.__file__), argv=list(map(str, argv)),
            exit_code=code, elapsed_seconds=time.monotonic() - started,
            original_main_code_object_unchanged=(author_module.main is original
                and original.__code__ is original_code),
            replaced_global_names=["_load_gt_gru_split_pt"],
            adaptation="memory mapped original stored NPZ shards, logical concatenation",
            dense_merged_or_extracted_tensor_corpus_written=False,
            original_TensorDataset_DataLoader_seed_shuffle_loss_optimizer_selection=True), guard)


@contextmanager
def native_python_startup(output, guard):
    """Copy only the reviewed fixed startup source; preserve the inherited environment."""
    payload = STARTUP.read_bytes()
    m._need(hashlib.sha256(payload).hexdigest() == STARTUP_SHA256,
        "Pinned static GPU5 startup source required")
    directory = output / "python_startup"
    directory.mkdir()
    ref = release._write(directory / "sitecustomize.py", payload, guard)
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
def native_local_runner(output, bounded_runner, execution_check):
    """Keep cmd_infer itself unchanged and temporarily bind its local process I/O."""
    m._need(callable(bounded_runner), "ROOT-owned inherited-group native runner required")
    module = importlib.import_module("commands.infer")
    original = module.cmd_infer
    original_code = original.__code__
    m._need(Path(module.__file__).resolve() == native.SOURCE / "src/commands/infer.py"
        and original.__globals__ is vars(module),
        "Only admitted unchanged native inference module may be bound")
    previous = module._run_local
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

    module._run_local = run_local
    try:
        yield waits
        m._need(module.cmd_infer is original and original.__code__ is original_code,
            "Original native cmd_infer callable and code must remain unchanged")
    finally:
        module._run_local = previous


def run_author(name, args, output):
    """Ordinary imported stock CLI main, with the original argv and result semantics."""
    m._need(name == "argmix_cli.py", "Only the exact admitted inference CLI is bound")
    module = importlib.import_module("argmix_cli")
    original = module.main
    m._need(Path(module.__file__).resolve() == native.SOURCE / "src/argmix_cli.py"
        and original.__globals__ is vars(module), "Exact admitted stock CLI main required")
    old_argv = sys.argv[:]
    saved, old_handlers = _detach_logging()
    log = output / "argmix_cli.log"
    started = time.monotonic()
    code = 0
    try:
        sys.argv = [str(native.SOURCE / "src" / name), *map(str, args)]
        with log.open("x") as stream, redirect_stdout(stream), redirect_stderr(stream):
            try:
                raise SystemExit(original())
            except SystemExit as result:
                code = result.code if type(result.code) is int else (0 if result.code is None else 1)
                if code and not isinstance(result.code, int):
                    print(result.code, file=sys.stderr)
            except BaseException:
                code = 1
                traceback.print_exc()
                raise
    except BaseException:
        code = 1
        raise
    finally:
        sys.argv = old_argv
        _restore_logging(saved, old_handlers)
        native.new_json(output / (name + ".command.json"), dict(
            stock_source=str((native.SOURCE / "src" / name).relative_to(native.ROOT)),
            source_sha256=native.sha(native.SOURCE / "src" / name),
            argv=list(map(str, args)), exit_code=code,
            elapsed_seconds=time.monotonic() - started, log_sha256=native.sha(log)))
    if code:
        raise RuntimeError("Stock author command failed: " + name)


def fit_release(**arguments):
    m._need(release.admit_native is ORIGINAL_ADMIT
        and shards.run_original_main is ORIGINAL_TRAIN_RUNNER,
        "Fresh single-call trainer I/O binding required")
    release.admit_native = admit_native
    shards.run_original_main = run_original_main
    try:
        return release.fit_release(**arguments)
    finally:
        release.admit_native = ORIGINAL_ADMIT
        shards.run_original_main = ORIGINAL_TRAIN_RUNNER


def infer_release(**arguments):
    m._need(release.admit_native is ORIGINAL_ADMIT
        and release._native_python_startup is ORIGINAL_STARTUP
        and release._native_local_runner is ORIGINAL_LOCAL_RUNNER
        and native.run_author is ORIGINAL_AUTHOR_RUNNER,
        "Fresh single-call inference I/O bindings required")
    release.admit_native = admit_native
    release._native_python_startup = native_python_startup
    release._native_local_runner = native_local_runner
    native.run_author = run_author
    try:
        return release.infer_release(**arguments)
    finally:
        release.admit_native = ORIGINAL_ADMIT
        release._native_python_startup = ORIGINAL_STARTUP
        release._native_local_runner = ORIGINAL_LOCAL_RUNNER
        native.run_author = ORIGINAL_AUTHOR_RUNNER
