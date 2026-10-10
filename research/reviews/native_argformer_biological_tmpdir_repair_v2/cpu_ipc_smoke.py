"""Synthetic CPU IPC test of the short TMPDIR; no author model or biology."""
from pathlib import Path
import hashlib, json, os, socket, stat, sys, time, traceback

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
REVIEW = ROOT / "research/reviews/native_argformer_biological_tmpdir_repair_v2"
PREFIX = ROOT / ".runtime/conda/envs/argformer_native_v1"
result = {"status": "failed", "exit_code": 1, "GPU_used": False,
          "biological_data_read": False, "author_model_or_training": False}
started = time.monotonic()
iterator = None
test_socket = None
try:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or Path(sys.prefix).resolve() != PREFIX:
        raise RuntimeError("Native qualified interpreter with empty CUDA visibility required")
    identity = {"output_dir": str(REVIEW / "CPU_smoke"), "job": "afbio-v2-cpu-ipc-smoke"}
    job_hash = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]
    short_tmp = ROOT / ".runtime/tmp/afbio" / job_hash
    short_tmp.mkdir(parents=True, exist_ok=False)
    os.environ["TMPDIR"] = str(short_tmp)
    import tempfile
    tempfile.tempdir = str(short_tmp)
    test_directory = short_tmp / "pymp-12345678"
    test_directory.mkdir()
    test_socket = test_directory / "listener-12345678"
    address_bytes = len(os.fsencode(test_socket))
    if address_bytes > 107:
        raise RuntimeError("Actual AF_UNIX path exceeds Linux usable length")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(test_socket))
        server.listen(1)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(str(test_socket))
            peer, _ = server.accept()
            with peer:
                client.sendall(b"repo-local-ipc")
                if peer.recv(64) != b"repo-local-ipc":
                    raise RuntimeError("Actual AF_UNIX byte transfer differs")
    test_socket.unlink()
    result["actual_AF_UNIX_bind_and_transfer"] = True
    result["actual_AF_UNIX_path_bytes"] = address_bytes
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU smoke imported initialized CUDA")
    torch.set_num_threads(1)
    if torch.multiprocessing.get_sharing_strategy() != "file_descriptor":
        raise RuntimeError("Qualified source-default Torch sharing strategy changed")
    ids = torch.arange(32 * 128, dtype=torch.long).reshape(32, 128)
    masks = torch.ones_like(ids)
    labels = torch.arange(32, dtype=torch.long) % 4
    dataset = TensorDataset(ids, masks, labels)
    loader = DataLoader(dataset, batch_size=8, num_workers=4, pin_memory=True,
                        prefetch_factor=2, persistent_workers=True, timeout=20)
    iterator = iter(loader)
    seen, checksum, batches = 0, 0, 0
    sockets = {}
    for batch in iterator:
        if not all(item.device.type == "cpu" and item.is_shared() for item in batch):
            raise RuntimeError("Actual CPU Torch batches were not transferred through shared memory")
        seen += len(batch[0]); checksum += int(batch[0].sum()); batches += 1
        for path in short_tmp.rglob("*"):
            try:
                if stat.S_ISSOCK(path.stat().st_mode):
                    sockets[str(path)] = len(os.fsencode(path))
            except FileNotFoundError:
                pass
    if seen != 32 or checksum != int(ids.sum()) or batches != 4:
        raise RuntimeError("Actual Torch DataLoader transfer count/checksum differs")
    if not sockets or max(sockets.values()) > 107:
        raise RuntimeError("Actual resource-sharer AF_UNIX socket evidence missing/too long")
    if torch.cuda.is_initialized():
        raise RuntimeError("Synthetic CPU IPC check initialized CUDA")
    result.update(status="passed", exit_code=0, TMPDIR=str(short_tmp),
                  actual_Torch_shared_memory_DataLoader=True,
                  sharing_strategy=torch.multiprocessing.get_sharing_strategy(),
                  actual_resource_sharer_sockets=sockets, native_torch_version=torch.__version__,
                  CUDA_initialized=False, batches=batches, examples=seen, checksum=checksum,
                  synthetic_dataset=True, model_instantiated=False)
except BaseException as error:
    result["failure"] = {"type": type(error).__name__, "message": str(error),
                         "traceback": traceback.format_exc()}
    traceback.print_exc()
finally:
    if iterator is not None:
        iterator._shutdown_workers()
    result["elapsed_seconds"] = time.monotonic() - started
    (REVIEW / "CPU_ipc_smoke_receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    (REVIEW / "CPU_ipc_smoke.exit_status").write_text(str(result["exit_code"]) + "\n")
    print(json.dumps({key: result.get(key) for key in ["status", "exit_code", "TMPDIR",
          "actual_AF_UNIX_path_bytes", "batches", "examples", "CUDA_initialized"]}), flush=True)
raise SystemExit(result["exit_code"])
