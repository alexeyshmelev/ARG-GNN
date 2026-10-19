"""Observation-only per-batch CUDA receipts; no numerical probes or schedule changes."""
import json
import os
from pathlib import Path
import time


def memory_callback(Callback, torch, output, resource_check):
    output = Path(output)
    class MemoryReceipt(Callback):
        def __init__(self):
            self.begin = None
            self.first = False
        def batch_start(self, state, logger):
            self.begin = time.monotonic()
        def batch_end(self, state, logger):
            resource_check("native_full_batch_completed")
            cap = int(os.environ["ARGGNN_MAX_CUDA_RESERVED_BYTES"])
            reserved = torch.cuda.memory_reserved(0)
            if reserved > cap:
                raise RuntimeError("Shared native CUDA allocator cap exceeded")
            row = dict(schema="shared_native_full_batch_memory_v1",
                batch=int(state.timestamp.batch.value),
                batch_wall_seconds=None if self.begin is None else time.monotonic()-self.begin,
                allocated_bytes=torch.cuda.memory_allocated(0),reserved_bytes=reserved,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
                cuda_cap_bytes=cap,full_native_batch_and_microbatch_preserved=True,
                observation_only=True)
            destinations = [output / "latest_batch_memory.json"]
            if not self.first:
                destinations.append(output / "first_full_batch_memory.json")
                self.first = True
            for path in destinations:
                temporary = path.with_suffix(".pending")
                temporary.write_text(json.dumps(row,sort_keys=True)+"\n")
                os.replace(temporary,path)
    return MemoryReceipt()
