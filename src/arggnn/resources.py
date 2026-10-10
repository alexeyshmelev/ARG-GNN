"""Project-scoped resource guards. No CUDA contexts are created here."""
from __future__ import annotations
import json
import os
import subprocess
import warnings
from pathlib import Path

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
MAX_BYTES = 500_000_000_000
WARN_BYTES = 450_000_000_000
ALLOWED_GPUS = (2, 3, 4, 5)
PROCESS_TITLE = "AI Auto Research Pipeline Test"

def project_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if path != ROOT and ROOT not in path.parents:
        raise ValueError(f"Output path is outside project: {path}")
    return path

def storage_bytes() -> int:
    result = subprocess.run(["du", "-s", "-B1", str(ROOT)], check=True,
                            capture_output=True, text=True)
    return int(result.stdout.split()[0])

def require_storage(headroom_bytes: int = 0) -> int:
    used = storage_bytes()
    if headroom_bytes < 0:
        raise ValueError("Headroom must be nonnegative")
    if used + headroom_bytes >= MAX_BYTES:
        raise RuntimeError(f"Project storage budget exceeded: {used} bytes used")
    if used + headroom_bytes >= WARN_BYTES:
        warnings.warn(f"Project approaching storage budget: {used} bytes", stacklevel=2)
    return used

def gpu_inventory() -> list[dict]:
    result = subprocess.run([
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], check=True, capture_output=True, text=True)
    apps = subprocess.run([
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ], check=True, capture_output=True, text=True)
    busy_uuids = {line.split(",")[0].strip() for line in apps.stdout.splitlines() if line.strip()}
    records = []
    for line in result.stdout.splitlines():
        idx, uuid, used, total, util = [part.strip() for part in line.split(",")]
        records.append({
            "index": int(idx), "uuid": uuid, "memory_used_mib": int(used),
            "memory_total_mib": int(total), "utilization_percent": int(util),
            "has_compute_process": uuid in busy_uuids,
        })
    return records

def eligible_free(record: dict) -> bool:
    return (record["index"] in ALLOWED_GPUS
            and not record["has_compute_process"]
            and record["memory_used_mib"] <= 100
            and record["utilization_percent"] == 0)

def set_research_title() -> None:
    import setproctitle
    setproctitle.setproctitle(PROCESS_TITLE)
