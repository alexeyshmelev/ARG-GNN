"""ROOT-admitted full observed European parent CPU qualification; no launch API."""
from __future__ import annotations
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import traceback
from . import european_stage as stage
from . import european_observations as observed
from .common import need


def contract(spec):
    return {key: spec[key] for key in ("max_sites", "max_metadata_bytes", "preparation_limits")}


def qualify(spec, *, resource_check):
    from arggnn.benchmarking import contracts as c
    from arggnn.benchmarking import treeloom_training as memory
    from research.reviews.workflow_progress_diagnosis_20261018.treeloom import cache_v1 as codec
    from research.reviews.workflow_progress_diagnosis_20261018.treeloom.resident_pool_v2 import array_hash
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only European qualification required")
    need(0 < spec["max_rss_bytes"] <= 128*2**30 and 0 < spec["max_seconds"] <= 14400
         and 0 < spec["max_output_bytes"] <= 16*2**20, "Finite full-parent CPU/RAM/output bounds required")
    refs = spec["observed_inputs"]
    need(set(refs) == {"inference_ref", "observed_vcf_ref", "sampled_registry_ref"},
         "Exactly observed inference/VCF/sample refs; no TRAIN labels or truth capability")
    out = c.contained_path(spec["output_directory"])
    need(out.is_relative_to(c.ROOT/"experiments/results") and out.name == "native" and not out.exists(),
         "Fresh supplied task/native qualification scope required")
    out.mkdir()
    started = time.monotonic(); peak = [0]; last = [None, started]
    def rss():
        return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    def write(name, value):
        encoded = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
        need(sum(path.stat().st_size for path in out.iterdir()) + len(encoded) <= spec["max_output_bytes"],
             "Bounded European qualification metadata output")
        with (out/name).open("ab") as stream:
            stream.write(encoded);stream.flush();os.fsync(stream.fileno())
    def check(stage):
        resource_check(stage)
        now = time.monotonic(); current = rss(); peak[0] = max(peak[0], current)
        need(now-started <= spec["max_seconds"] and current <= spec["max_rss_bytes"],
             "Full European CPU qualification exceeded admitted RAM/wall cap")
        if stage != last[0] or now-last[1] >= 30:
            write("progress.jsonl", dict(stage=stage, elapsed_seconds=now-started, rss_bytes=current,
                                         parent_id=spec["parent_id"], protocol_id=spec["protocol_id"]))
            last[:] = [stage, now]
    def alarm(number, frame):
        raise TimeoutError("Full European CPU qualification wall cap")
    signal.signal(signal.SIGALRM, alarm);signal.alarm(spec["max_seconds"])
    result = dict(schema="european_readable_complete_parent_CPU_qualification_v1",
        protocol_id=spec["protocol_id"], parent_id=spec["parent_id"],
        sampling_plan_ref=spec["sampling_plan_ref"], observed_inputs=refs,
        preparation_contract=contract(spec), candidate_source_refs=spec["candidate_source_refs"],
        private_truth_opened=False, training_labels_opened=False, model_instantiated=False,
        CUDA_called=False, full_parent_preparation_count=0, scientific_outputs_created=0,
        disk_cache_bytes=0, GPU_fit_qualified=False)
    timings = {}
    try:
        import torch
        torch.set_num_threads(8)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        plan = stage._parent_plan(spec["sampling_plan_ref"], spec["protocol_id"])
        need(spec["parent_id"] == plan["train_parent_ids"][0], "First exact registered TRAIN parent qualification")
        check("European_qualification_load"); at = time.monotonic()
        state = observed.load_relate_observation(refs["inference_ref"], refs["observed_vcf_ref"],
            protocol_id=spec["protocol_id"], max_sites=spec["max_sites"],
            max_metadata_bytes=spec["max_metadata_bytes"], resource_check=check)
        timings["load_observation_seconds"] = time.monotonic()-at
        at = time.monotonic();result["full_parent_preparation_count"] = 1
        bundle, metadata = observed.prepare_observation(state, protocol_id=spec["protocol_id"],
            parent_id=spec["parent_id"], split="train", source_refs=state["source_refs"],
            sampled_registry_ref=refs["sampled_registry_ref"], resource_check=check,
            max_sampled_metadata_bytes=spec["max_metadata_bytes"], **spec["preparation_limits"])
        timings["complete_prepare_seconds"] = time.monotonic()-at
        bundle.validate();check("European_qualification_full_signature")
        arrays, structures = codec.flatten(bundle.bank, bundle.episodes, bundle.cut)
        arrays["focal_hold"] = bundle.focal_hold.numpy()
        arrays["focal_sample_column"] = bundle.focal_sample_column.numpy()
        hashes = {}
        for key, array in arrays.items():
            check("European_signature_" + key);hashes[key] = array_hash(array)
        scalars = {field.name: codec._plain(getattr(bundle.original, field.name))
                   for field in fields(bundle.original)
                   if field.name not in ("bank", "episodes", "cut") and not field.name.endswith("stamps")}
        signature = dict(arrays=hashes, structures=structures, original=scalars,
                         focal_identity_sha256=bundle.focal_identity_sha256,
                         sampled_registry_ref=bundle.sampled_registry_ref, metadata=metadata)
        digest = hashlib.sha256(json.dumps(signature, sort_keys=True, allow_nan=False).encode()).hexdigest()
        check("European_qualification_signature_complete")
        result.update(status="passed_full_observed_parent_CPU_preparation", all_native_trees_nodes_sites_samples_retained=True,
            native_holds=bundle.native_holds, packed_nodes=bundle.bank.parent.numel(),
            haplotypes=bundle.bank.sample_index.shape[1], native_sites=bundle.bank.genotype.shape[0],
            focal_records=bundle.focal_hold.numel(), focal_identity_sha256=bundle.focal_identity_sha256,
            observed_tensor_bytes=memory._cpu_bytes(bundle, torch), complete_signature_sha256=digest,
            sampled_rows_and_repeated_draws_preserved=True)
    except BaseException as error:
        result.update(status="failed_bounded_European_CPU_qualification", error_type=type(error).__name__,
                      error=str(error), traceback=traceback.format_exc()[-16384:])
    finally:
        signal.alarm(0)
        result.update(elapsed_seconds=time.monotonic()-started, observed_peak_rss_bytes=peak[0], timings=timings)
        write("qualification.json", result)
    return result
