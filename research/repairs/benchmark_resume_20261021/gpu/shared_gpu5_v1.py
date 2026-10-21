"""Read-only physical GPU5 inventory and exact PID-generation metadata."""
from pathlib import Path
import subprocess
import time

PHYSICAL_GPU = 5
GPU5_UUID = "GPU-3db85505-d061-e45d-d154-e7e8c1cad497"


def need(value, message):
    if not value:
        raise RuntimeError(message)


def identity(pid):
    fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=int(pid), start_ticks=int(fields[19]))


def gpu_sample(run=subprocess.run):
    result = run(['nvidia-smi', '--id=5',
        '--query-gpu=index,uuid,memory.total,memory.used,memory.free',
        '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=10)
    rows = [line.split(',') for line in result.stdout.splitlines() if line.strip()]
    need(len(rows) == 1 and len(rows[0]) == 5, 'One exact physical GPU5 inventory row required')
    index, uuid, total, used, free = [x.strip() for x in rows[0]]
    need(index == '5' and uuid == GPU5_UUID, 'Unexpected physical device identity')
    total, used, free = [int(x) * 2**20 for x in (total, used, free)]
    reserved = total-used-free
    need(0 <= used <= total and 0 <= free <= total and -2*2**20 <= reserved <= total,
         'Incomplete or inconsistent GPU5 memory inventory')
    return dict(physical_gpu=PHYSICAL_GPU, uuid=uuid, total_bytes=total, used_bytes=used,
                free_bytes=free, driver_reserved_bytes=max(0,reserved),
                driver_reserved_bytes_derived_from_inventory=True, monotonic=time.monotonic())
