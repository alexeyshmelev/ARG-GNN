"""Small GPU/process-label verification, not a training experiment."""
from __future__ import annotations
import json
import os
import time
from .resources import ROOT, PROCESS_TITLE, require_storage, set_research_title

def main():
    set_research_title()
    import torch
    require_storage()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Probe must see exactly the reserved GPU")
    x = torch.randn(1024, 1024, device="cuda")
    lease_fd = int(os.environ["ARGGNN_GPU_LEASE_FD"])
    os.fstat(lease_fd)
    print(json.dumps({"event": "probe_ready", "pid": os.getpid(), "lease_fd": lease_fd,
                      "lease_path": os.environ["ARGGNN_GPU_LEASE"],
                      "physical_gpu": os.environ["ARGGNN_PHYSICAL_GPU"]}), flush=True)
    start = time.monotonic()
    while time.monotonic() - start < 35:
        _ = x @ x
        torch.cuda.synchronize()
        time.sleep(0.1)
    print(json.dumps({
        "kind": "infrastructure_probe", "pid": os.getpid(),
        "physical_gpu": os.environ["ARGGNN_PHYSICAL_GPU"],
        "title": PROCESS_TITLE, "torch": torch.__version__,
        "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
    }), flush=True)

if __name__ == "__main__":
    main()
