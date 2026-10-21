"""Fixed stage-local startup for unchanged author children on physical GPU 5."""
import os
import sys
import traceback

TITLE = "AI Auto Research Pipeline Test"
GPU_UUID = "GPU-3db85505-d061-e45d-d154-e7e8c1cad497"
SOURCE_ROOT = "/workspace-SR003.nfs2/ARG-GNN/.runtime/official_argmix_abi210_v1/cu129/source/src"
CHILD_NAMES = ("export_torchscript.py", "run_inference.py", "run_joint_inference.py")


def admit_child():
    import setproctitle
    setproctitle.setproctitle(TITLE)
    name = os.path.basename(sys.argv[0])
    if name not in CHILD_NAMES:
        return
    if os.path.realpath(sys.argv[0]) != os.path.join(SOURCE_ROOT, name):
        raise RuntimeError("Native child script must retain its admitted source origin")
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != GPU_UUID
            or os.environ.get("ARGGNN_PHYSICAL_GPU") != "5"):
        raise RuntimeError("Native child must inherit admitted physical GPU5")
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


if __name__ == "sitecustomize":
    try:
        admit_child()
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        os._exit(78)
