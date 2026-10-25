"""Bounded atomic parent-boundary recovery state, independent of final selection."""
from __future__ import annotations
import os
from pathlib import Path
import random
import numpy as np
import torch

SCHEMA = "paper_gst_lai_parent_recovery_v1"
MAX_CHECKPOINT_BYTES = 16*2**20
PEAK_CHECKPOINT_BYTES = 2*MAX_CHECKPOINT_BYTES


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key:cpu_copy(item) for key,item in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    return value


def random_state():
    numpy = np.random.get_state()
    return dict(python=random.getstate(), torch_cpu=torch.get_rng_state(),
        numpy=dict(kind=numpy[0], keys=torch.from_numpy(numpy[1].astype(np.int64)),
                   position=numpy[2], has_gaussian=numpy[3], cached_gaussian=numpy[4]),
        torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def restore_random_state(saved):
    random.setstate(saved["python"])
    numpy = saved["numpy"]
    np.random.set_state((numpy["kind"], numpy["keys"].numpy().astype(np.uint32),
        numpy["position"], numpy["has_gaussian"], numpy["cached_gaussian"]))
    torch.set_rng_state(saved["torch_cpu"])
    if saved["torch_cuda"]:
        if not torch.cuda.is_initialized():
            raise ValueError("Resume requires the original CUDA RNG device condition")
        torch.cuda.set_rng_state_all(saved["torch_cuda"])


def tensor_bytes(value):
    if torch.is_tensor(value):
        return value.numel()*value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple,list)):
        return sum(tensor_bytes(item) for item in value)
    return 0


class CappedWriter:
    """Enforce the temporary file cap while torch writes, before atomic replace."""
    def __init__(self, stream):
        self.stream = stream
    def write(self, value):
        if self.stream.tell()+len(value) > MAX_CHECKPOINT_BYTES:
            raise ValueError("GST recovery checkpoint exceeds 16 MiB")
        return self.stream.write(value)
    def flush(self):
        self.stream.flush()


def save_latest(output, payload, resource_check):
    resource_check(PEAK_CHECKPOINT_BYTES)
    if tensor_bytes(payload) > MAX_CHECKPOINT_BYTES-2**20:
        raise ValueError("GST recovery tensors leave insufficient serialization headroom")
    output = Path(output)
    staging = output/"recovery_latest.staging.pt"
    latest = output/"recovery_latest.pt"
    if staging.exists():
        raise ValueError("Unconsumed recovery staging evidence must be preserved")
    with staging.open("xb") as stream:
        torch.save(cpu_copy(payload), CappedWriter(stream))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(staging, latest)
    directory_fd = os.open(output, os.O_RDONLY|os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return latest


def validate_progress(saved, plan, settings, train_inputs, source_refs, proof):
    if saved["schema"] != SCHEMA or saved["settings"] != settings:
        raise ValueError("Exact recovery schema and original scientific settings required")
    if saved["train_inputs"] != train_inputs or saved["source_refs"] != source_refs:
        raise ValueError("Exact unchanged data and isolated source bindings required")
    if saved["observation_proof"] != proof:
        raise ValueError("Exact reauthenticated original observed parents required")
    progress = saved["progress"]
    epoch = progress["epoch"]
    if not 1 <= epoch <= 30:
        raise ValueError("Original 30-epoch budget required")
    order = list(plan["train_parent_ids"])
    random.Random(settings["seed"]+epoch).shuffle(order)
    done = progress["completed_parents"]
    if done != order[:len(done)] or len(done)>len(order):
        raise ValueError("Recovery must be an exact original parent-order prefix")
    expected = (epoch-1)*len(order)+len(done)
    if saved["optimizer_updates"] != expected or len(saved["history"]) != expected:
        raise ValueError("Recovery cannot repeat or invent completed parent updates")
    steps = [int(row["step"].item()) for row in saved["optimizer_state"]["state"].values()]
    parameter_count=sum(len(group["params"]) for group in saved["optimizer_state"]["param_groups"])
    if not 1 <= expected <= 90 or len(steps) != parameter_count or set(steps) != {expected}:
        raise ValueError("All optimizer counters must match the completed parent boundary")
    expected_history=[]
    for previous_epoch in range(1,epoch+1):
        previous_order=list(plan["train_parent_ids"])
        random.Random(settings["seed"]+previous_epoch).shuffle(previous_order)
        if previous_epoch == epoch:
            previous_order=previous_order[:len(done)]
        for parent in previous_order:
            expected_history.append((previous_epoch,parent,len(expected_history)+1))
    actual_history=[(row["epoch"],row["parent_id"],row["optimizer_updates"])
                    for row in saved["history"]]
    if actual_history != expected_history:
        raise ValueError("Recovery history must preserve the full original update order")
    if set(progress["counts"]) != set(order):
        raise ValueError("Recovery must retain counts for every original TRAIN parent")
    for parent in order:
        counts=progress["counts"][parent]
        target=proof[parent] if parent in done else dict(trees=0,sites=0)
        if counts["trees"] != target["trees"] or counts["sites"] != target["sites"]:
            raise ValueError("Only complete committed parents may be skipped on recovery")
    totals=progress["completed_epoch_totals"]
    if totals["trees"] != (epoch-1)*sum(item["trees"] for item in proof.values()):
        raise ValueError("Recovery complete-epoch native tree counts disagree")
    if totals["sites"] != (epoch-1)*sum(item["sites"] for item in proof.values()):
        raise ValueError("Recovery complete-epoch native site counts disagree")
    return progress
