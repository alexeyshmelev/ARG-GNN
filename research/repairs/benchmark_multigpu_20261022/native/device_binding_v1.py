"""Exact inherited job-device binding for authorized physical GPUs 2, 3 and 5."""
import os

UUID_BY_DEVICE = {
    "2": "GPU-976a5cd3-df12-2535-fa51-e3ea063688ba",
    "3": "GPU-df11de7f-7a15-ebe2-085d-9c7d907f3786",
    "5": "GPU-3db85505-d061-e45d-d154-e7e8c1cad497",
}
TITLE = "AI Auto Research Pipeline Test"


def require_job_device(*, require_title=False):
    physical = os.environ.get("ARGGNN_PHYSICAL_GPU")
    expected_uuid = UUID_BY_DEVICE.get(physical)
    if expected_uuid is None or os.environ.get("CUDA_VISIBLE_DEVICES") != expected_uuid:
        raise ValueError("Exact inherited job physical GPU2/3/5 and matching UUID required before CUDA")
    if require_title:
        import setproctitle
        if setproctitle.getproctitle() != TITLE:
            raise ValueError("Author code must run inside the correctly titled GPU worker")
    return int(physical), expected_uuid
