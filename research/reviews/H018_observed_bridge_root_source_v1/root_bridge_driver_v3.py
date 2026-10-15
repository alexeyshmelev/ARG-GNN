"""Root-only bounded H018 baseline GT preparation. Every action is inert by default.

No native inference, Relate invocation, model, ancestry target or HS realization.
The unchanged credit-v3 guard owns accounting. There is no release/credit path.
"""
from pathlib import Path
from datetime import datetime, timedelta, timezone
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import resource
import selectors
import signal
import stat
import subprocess
import sys
import time

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
REL = "research/reviews/H018_observed_bridge_root_source_v1"
SOURCE = REL + "/root_bridge_driver_v3.py"
CONFIG = REL + "/config_v3.json"
MANIFEST = REL + "/source_manifest_v3.json"
PACKET = REL + "/admission_v2"
AUX = REL + "/worker_aux"
OUTPUT = "data/H018_R16_observed_bridge_v1"
JOB = "H018-observed-baseline-GT-preparation-v1"
KEY = "H018/observed_baseline_GT_preparation_v1"
CONDITION = "H018_R16_20_30Mb_extension_v1"
CAP, OUTPUT_CAP, AUX_WRITE = 50_000_000, 32_000_000, 3_000_000
CORE_PAYLOAD_CAP, INPUT_CAP, LOG_CAP = 25_000_000, 80_000_000, 2_000_000
MEMORY, COMBINED_RSS = 1_000_000_000, 2_000_000_000
WORKER_SECONDS, SUPERVISOR_SECONDS, GRACE_SECONDS = 300, 600, 10
COMMON = dict(path="research/reviews/H018_map_acquisition_root_source_v1/root_map_driver.py",
              sha256="4c957db44a1bf6c9cd8e76fd6cd5e31b979b7c618554c6a73c57cce8099af65d")
BRIDGE = dict(path="research/reviews/H018_observed_bridge_source_v1/bridge_core.py",
              sha256="a860ebf2f046834871d06a1938f0a1de0431cc20f698c1c5096b87bdc4ef89d2")
INPUT_REFS = dict(path="research/reviews/H018_observed_bridge_source_v1/input_refs.inactive.json",
                  sha256="dbc7ad96c9d91574c6c3137f0c0767380c4238116c15b7bb7c2dc953e13b26b7")
PYTHON = dict(path=".runtime/env/bin/python3.11",
              sha256="463a65c4055c18d88f6ed863d227383593395c2f7f70976e38df24f17a836289")
PYTHON_VERSION = "3.11.17 (main, Oct  7 2026, 00:02:24) [GCC 15.3.0]"
CORE_NAMES = {"baseline.haps", "baseline.sample", "sample_ages.generations",
              "frozen_site_identities.json", "polarization_equivalence.json", "bridge_preparation.json"}
FINAL_NAMES = CORE_NAMES | {"qualified.regional.map", "NOTICE.txt", "provenance.json"}


def require(ok, text):
    if not ok:
        raise ValueError(text)


def utc():
    return datetime.now(timezone.utc).isoformat()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def common():
    raw = (ROOT / COMMON["path"]).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == COMMON["sha256"], "Original admission primitive source drift")
    return load("_H018_bridge_unchanged_root_primitives", ROOT / COMMON["path"])


def syncdir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save(b, folder, name, value):
    record = b.save(folder, name, value)
    syncdir(folder)
    return record


def identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def bounded_file(b, record, cap, *, retain=True):
    """Canonical non-symlink bounded bytes or streaming output/source reference.

    Unlike common.ref this accepts bounded artifacts exceeding four MB. Never
    call this on observed VCF/QC before the worker's original guard boundary.
    """
    require(set(record) in ({"path", "sha256"}, {"path", "sha256", "bytes"}), "Exact bounded reference shape")
    path = b.local(record["path"])
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    parts, digest, total = [], hashlib.sha256(), 0
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_size <= cap, "Bounded regular input/artifact")
        while True:
            chunk = os.read(fd, min(1_048_576, cap - total + 1))
            if not chunk:
                break
            total += len(chunk)
            require(total <= cap, "Input/artifact byte cap")
            digest.update(chunk)
            if retain:
                parts.append(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    require(identity(before) == identity(after) == identity(path.lstat()), "Input/artifact changed during read")
    require(total == before.st_size and digest.hexdigest() == record["sha256"], "Pinned input/artifact SHA and size")
    require("bytes" not in record or total == record["bytes"], "Completion artifact byte count")
    return b"".join(parts) if retain else dict(path=record["path"], sha256=digest.hexdigest(), bytes=total)


def error_record(exc):
    return dict(type=type(exc).__name__, message=str(exc)[:2048])


def native_gate():
    return dict(native_execution_authorized=False, native_inference_executed=False,
                separate_native_policy_and_admission_required=True,
                fractional_calBP1950_div28_native_age_roundtrip_policy_resolved=False,
                HS_draws_materialized=False, GPU_used=False, model_prediction=False,
                ancestry_truth_available=False)


DIRECT_BINDING_SUFFIXES = (".json", ".py", ".sh")


def direct_authority_refs(manifest, closure=None):
    """Retain every ref transitively; only original allowed suffixes bind directly."""
    groups = ("source_refs", "metadata_refs")
    excluded = {group: {name: rec for name, rec in manifest[group].items()
                       if Path(rec["path"]).suffix not in DIRECT_BINDING_SUFFIXES}
                for group in groups}
    require(manifest["direct_binding_suffixes"] == list(DIRECT_BINDING_SUFFIXES)
            and manifest["transitive_only_refs"] == excluded,
            "Exact original allowed direct suffixes and full excluded-reference manifest required")
    if closure is not None:
        require(all(closure[group] == manifest[group] for group in groups)
                and closure["direct_binding_suffixes"] == manifest["direct_binding_suffixes"]
                and closure["transitive_only_refs"] == excluded,
                "Actual direct JSON closure must preserve every complete source/metadata/excluded ref")
    return [rec for group in groups for rec in manifest[group].values()
            if Path(rec["path"]).suffix in DIRECT_BINDING_SUFFIXES]


def verify_request(b, request_ref):
    r = b.read(request_ref)
    require(r["schema"] == "H018_observed_bridge_root_request_v1" and r["status"] == "active"
            and r["root_authorizer"] == "root_agent" and r["registered_before_execution"] is True,
            "Explicit root activation and genuine frozen registration required")
    expected = dict(job_id=JOB, condition_id=CONDITION, output_directory=OUTPUT,
                    packet_directory=PACKET, auxiliary_directory=AUX, reserved_bytes=CAP,
                    GPU_authorized=False, native_execution_authorized=False,
                    HS_execution_authorized=False, replay_authorized=False)
    require(all(r.get(k) == v for k, v in expected.items()), "Exact preparation-only root request")
    require(r["driver_ref"] == b.ref(SOURCE) and r["config_ref"] == b.ref(CONFIG)
            and r["source_manifest_ref"] == b.ref(MANIFEST), "Actual driver/config/manifest pins")
    cfg, manifest = b.read(r["config_ref"]), b.read(r["source_manifest_ref"])
    exact = dict(schema="H018_observed_bridge_root_config_v1", job_id=JOB, condition_id=CONDITION,
        output_directory=OUTPUT, packet_directory=PACKET, auxiliary_directory=AUX,
        reserved_bytes=CAP, output_batch_cap_bytes=OUTPUT_CAP, core_payload_cap_bytes=CORE_PAYLOAD_CAP,
        auxiliary_guard_growth_bytes=AUX_WRITE, input_file_cap_bytes=INPUT_CAP, log_cap_bytes=LOG_CAP,
        worker_seconds=WORKER_SECONDS, supervisor_seconds=SUPERVISOR_SECONDS,
        termination_grace_seconds=GRACE_SECONDS, address_space_per_process_bytes=MEMORY,
        combined_RSS_cap_bytes=COMBINED_RSS, CPU_threads=1, physical_gpu=None,
        common_ref=COMMON, bridge_ref=BRIDGE, input_refs_ref=INPUT_REFS,
        python_binary_ref=PYTHON, python_version=PYTHON_VERSION,
        guard_ref=b.GUARD, policy_ref=b.POLICY, repair_authority_ref=b.REPAIR,
        native_execution_authorized=False, HS_execution_authorized=False,
        fresh_single_attempt=True, warning_limit_bytes=950_000_000_000,
        hard_limit_bytes=1_000_000_000_000)
    require(cfg == exact, "Exact bounded preparation configuration")
    require(manifest["schema"] == "H018_observed_bridge_root_source_manifest_v1"
            and manifest["source_only"] is True and manifest["source_refs"]["driver"] == r["driver_ref"]
            and manifest["source_refs"]["config"] == r["config_ref"], "Exact source closure")
    direct_authority_refs(manifest)  # Classification never skips the original full SHA checks below.
    for rec in manifest["source_refs"].values():
        require(b.ref(rec["path"]) == rec, "Source closure drift")
    for rec in manifest["metadata_refs"].values():
        require(b.ref(rec["path"]) == rec, "Qualified compact metadata drift")
    inputs = b.read(INPUT_REFS)
    require(inputs["schema"] == "H018_baseline_bridge_source_inputs_v1"
            and inputs["root_metadata_original_OS_exit"] == "unknown"
            and inputs["root_metadata_original_wait0_available"] is False
            and inputs["root_metadata_original_reservation_bytes_retained"] == 5_000_000
            and inputs["root_metadata_replay_or_credit_authorized"] is False,
            "Preserve original metadata-controller unknown exit/full-charge caveat")
    refs = inputs["refs"]
    require(manifest["metadata_refs"] == {k: refs[k] for k in manifest["metadata_refs"]},
            "Exact transitive qualified inputs")
    caveat = b.read(refs["metadata_independent_result_validation"])
    require(caveat["status"] == "all_outputs_verified_original_writer_absent_actual_child_OS_exit_unknown"
            and caveat["all_producer_asset_hashes_verified"] is True
            and caveat["actual_direct_wait0_claim"] is False
            and caveat["reservation_remains_fully_charged"] is True, "Honest original metadata evidence")
    completed = b.read(refs["metadata_completion"])
    require(completed["status"] == "completed_source_qualified_map_and_sample_metadata"
            and completed["map_rows"] == 12280 and completed["no_internal_OS_exit_or_wait0_claim"] is True,
            "Qualified metadata producer completion, without inferred OS exit")
    for name in ("qualified.regional.map", "planned.sample", "planned.sample_ages.generations",
                 "planned_haplotype_roster.json"):
        require(completed["assets"][name] == refs[name], "Exact independently verified producer assets")
    qualification = b.read(refs["map_qualification.json"])
    require(qualification["source_semantics_qualified"] is True
            and qualification["uniform_map_substitution"] is False
            and qualification["output_rows"] == 12280
            and qualification["map_sha256"] == refs["qualified.regional.map"]["sha256"], "Genuine qualified map")
    # No VCF/QC is opened by request verification, admission, launch or source checks.
    registration = b.read(r["registration_ref"])
    entry = next((x for x in registration["decisions"] if x.get("id") == "H018"), None)
    require(entry is not None and JOB in json.dumps(entry, sort_keys=True), "Root H018 registration explicitly names this fresh job")
    require(Path(sys.executable).resolve() == ROOT / PYTHON["path"] and sys.version == PYTHON_VERSION,
            "Exact existing project Python runtime")
    bounded_file(b, PYTHON, 32_000_000, retain=False)
    return r, cfg, manifest, refs


def admit(record):
    b = common()
    r, cfg, manifest, refs = verify_request(b, record)
    packet, output, aux = (b.local(x) for x in (PACKET, OUTPUT, AUX))
    require(not any(p.exists() for p in (packet, output, aux)), "Fresh single-use packet/output/aux required")
    packet.mkdir(); syncdir(packet.parent)
    expiry = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
    runtime = save(b, packet, "runtime.json", dict(schema="H018_observed_bridge_CPU_runtime_v1",
        job_id=JOB, python_executable=str(ROOT / PYTHON["path"]), python_binary_ref=PYTHON,
        python_version=sys.version, CPU_threads=1, max_address_space_bytes=MEMORY,
        combined_RSS_cap_bytes=COMBINED_RSS, worker_seconds=WORKER_SECONDS,
        supervisor_seconds=SUPERVISOR_SECONDS, log_cap_bytes=LOG_CAP, device="cpu", physical_gpu=None,
        output_directory=OUTPUT, auxiliary_directory=AUX, native_execution_authorized=False,
        created_at_utc=utc(), expires_at_utc=expiry))
    closure = save(b, packet, "consumer_manifest.json", dict(schema="H018_observed_bridge_consumer_v1",
        request_ref=record, source_manifest_ref=r["source_manifest_ref"], runtime_ref=runtime,
        registration_ref=r["registration_ref"], source_refs=manifest["source_refs"],
        metadata_refs=manifest["metadata_refs"], direct_binding_suffixes=manifest["direct_binding_suffixes"],
        transitive_only_refs=manifest["transitive_only_refs"],
        observed_payload_refs={k: refs[k] for k in ("source_VCF", "source_QC")},
        observed_payloads_opened=False, native_execution_authorized=False,
        metadata_original_OS_exit="unknown", metadata_original_reservation_retained_bytes=5_000_000))
    closure_direct = direct_authority_refs(manifest, b.read(closure))
    m = b.guard()
    lock = b.local("research/reviews/American_C3_short_train2mb_wave_source_v1/scheduler/admission.lock")
    with lock.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        view = m.inspect_v1(expected_source_sha256=b.CORE, expected_launcher_sha256=b.BASE)
        candidates = list({x["path"]: x for x in [record, r["config_ref"], r["source_manifest_ref"],
            r["registration_ref"], runtime, closure, *manifest["source_refs"].values(),
            *manifest["metadata_refs"].values(), b.GUARD, b.POLICY, b.REPAIR,
            *view["bindings"].values()]}.values())
        consumers = [x for x in candidates if Path(x["path"]).suffix in DIRECT_BINDING_SUFFIXES]
        authorities = [record, r["config_ref"], r["source_manifest_ref"], r["registration_ref"],
                       runtime, closure, *closure_direct, b.GUARD, b.POLICY, b.REPAIR,
                       *view["bindings"].values()]
        require(all(Path(x["path"]).suffix in DIRECT_BINDING_SUFFIXES and x in consumers
                    for x in authorities), "Every actual runtime/source/guard authority remains directly bound")
        require(all(x not in consumers for group in manifest["transitive_only_refs"].values()
                    for x in group.values()), "Excluded full refs remain transitive, never direct authorities")
        require(len(consumers) <= 64, "Original direct-reference bound")
        operation = m.operation_draft_v1("reserve", dict(key=KEY, output=str(output),
            auxiliary_scopes=[str(aux)], reserved_bytes=CAP, consumer_bindings=consumers),
            expected_before=m._pins(view), bindings=view["bindings"])
        operation.update(active=True, expires_at_utc=expiry)
        receipt = m.apply_reserve_v1(save(b, packet, "reserve_authority.json", operation),
            expected_source_sha256=b.CORE, expected_launcher_sha256=b.BASE)
    save(b, packet, "reserve_receipt.json", receipt)
    storage = save(b, packet, "storage.json", dict(kind="project_budget_v9_storage_record", active=True,
        consumer_uses_v9_guard=True, accounting_mode=m.MODE, job_id=JOB, output_directory=OUTPUT,
        reserved_new_bytes=CAP, reservation=receipt["value"], expires_at_utc=expiry,
        initial_ledger_pins=receipt["initial_ledger_pins"], completed_overlay_policy_ref=b.POLICY,
        budget_components=dict(implementation=b.ref(m.budget.SOURCE_REL),
            launcher=b.ref(m.budget.LAUNCHER_REL), consumer_adapter=b.GUARD)))
    admitted = save(b, packet, "admitted.json", dict(schema="H018_observed_bridge_admitted_v1",
        request_ref=record, driver_ref=b.ref(SOURCE), runtime_ref=runtime, storage_ref=storage,
        consumer_manifest_ref=closure, source_manifest_ref=r["source_manifest_ref"],
        job_id=JOB, output_directory=OUTPUT, auxiliary_directory=AUX, reserved_bytes=CAP,
        expires_at_utc=expiry, consumer_binding_count=len(consumers), execution_started=False,
        native_execution_authorized=False, reservation_remains_fully_charged=True))
    print(json.dumps(dict(admitted_ref=admitted, reserved_bytes=CAP)), flush=True)
    return admitted


def admitted_inputs(record):
    b = common()
    ad = b.read(record)
    require(record["path"] == PACKET + "/admitted.json" and ad["schema"] == "H018_observed_bridge_admitted_v1"
            and ad["driver_ref"] == b.ref(SOURCE), "Exact original admission")
    r, cfg, manifest, refs = verify_request(b, ad["request_ref"])
    require(ad["source_manifest_ref"] == r["source_manifest_ref"] and ad["job_id"] == JOB
            and ad["output_directory"] == OUTPUT and ad["auxiliary_directory"] == AUX
            and ad["reserved_bytes"] == CAP and ad["native_execution_authorized"] is False,
            "Exact original output/aux/cap admission")
    stamp = datetime.fromisoformat(ad["expires_at_utc"])
    require(stamp.tzinfo is not None and stamp > datetime.now(timezone.utc), "Unexpired admission")
    runtime, storage = b.read(ad["runtime_ref"]), b.read(ad["storage_ref"])
    closure = b.read(ad["consumer_manifest_ref"])
    require(runtime["job_id"] == JOB and runtime["python_executable"] == str(ROOT / PYTHON["path"])
            and runtime["python_binary_ref"] == PYTHON and runtime["python_version"] == sys.version
            and runtime["device"] == "cpu" and runtime["physical_gpu"] is None
            and runtime["native_execution_authorized"] is False
            and runtime["expires_at_utc"] == ad["expires_at_utc"], "Actual admitted runtime")
    own = storage["reservation"]
    require(storage["job_id"] == JOB and storage["output_directory"] == OUTPUT
            and storage["reserved_new_bytes"] == CAP and own["key"] == KEY
            and own["output"] == str(b.local(OUTPUT)) and own["auxiliary_scopes"] == [str(b.local(AUX))]
            and own["reserved_bytes"] == CAP and storage["expires_at_utc"] == ad["expires_at_utc"],
            "Original combined output-plus-aux reservation")
    require(closure["request_ref"] == ad["request_ref"] and closure["runtime_ref"] == ad["runtime_ref"]
            and closure["source_manifest_ref"] == r["source_manifest_ref"], "Actual bound consumer manifest")
    closure_direct = direct_authority_refs(manifest, closure)
    required = [ad["request_ref"], ad["runtime_ref"], ad["consumer_manifest_ref"], r["config_ref"],
                r["source_manifest_ref"], r["registration_ref"], *closure_direct,
                b.GUARD, b.POLICY, b.REPAIR, *storage["budget_components"].values()]
    require(all(Path(x["path"]).suffix in DIRECT_BINDING_SUFFIXES and x in own["consumer_bindings"]
                for x in required), "Actual producer/runtime/source/guard authorities must remain direct bindings")
    require(all(x not in own["consumer_bindings"] for group in manifest["transitive_only_refs"].values()
                for x in group.values()), "All excluded refs retained only in bound manifest/closure JSON")
    return b, ad, r, refs, storage


def limit_process():
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                      MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
               PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT / "src"))
    affinity = os.sched_getaffinity(0)
    require(affinity, "Available CPU affinity")
    os.sched_setaffinity(0, {min(affinity)})
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY, MEMORY))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def write_bytes(folder, name, raw):
    require(Path(name).name == name and name not in ("", ".", ".."), "Flat exclusive artifact name")
    fd = os.open(folder / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def deadline(signum, frame):
    raise TimeoutError("300-second preparation deadline exceeded")


def worker(record):
    b = None
    output = None
    output_write_admitted = False
    validated_counts = None
    try:
        b, ad, r, refs, storage = admitted_inputs(record)
        packet, output = b.local(PACKET), b.local(OUTPUT)
        authority = b.read(b.ref(packet / "worker_authority.json"))
        require(authority["admitted_ref"] == record and authority["supervisor_pid"] == os.getppid()
                and authority["argv_action"] == "worker" and os.getsid(0) == os.getpid()
                and os.getpgrp() == os.getpid(), "Genuine direct child of admitted root supervisor")
        require(not output.exists(), "Preserve any previous output; no replay")
        save(b, packet, "worker_claim.json", dict(admitted_ref=record, worker_pid=os.getpid(),
            supervisor_pid=os.getppid(), started_utc=utc(), retries_authorized=False))
        limit_process()
        m = b.guard()
        # This unchanged preflight precedes the first observed payload read.
        m.budget_check_v9(storage, first=True, boundary=True, required_write_bytes=OUTPUT_CAP)
        started = time.monotonic()
        prior = signal.signal(signal.SIGALRM, deadline)
        signal.setitimer(signal.ITIMER_REAL, WORKER_SECONDS)
        try:
            vcf = bounded_file(b, refs["source_VCF"], INPUT_CAP)
            qc = bounded_file(b, refs["source_QC"], INPUT_CAP)
            roster = b.read(refs["planned_haplotype_roster.json"])
            qualified_map = bounded_file(b, refs["qualified.regional.map"], 2_000_000)
            core = load("_H018_pinned_pure_observed_bridge", ROOT / BRIDGE["path"])
            payloads, report = core.assemble_baseline(vcf, qc, roster,
                expected_vcf_sha256=refs["source_VCF"]["sha256"], expected_qc_sha256=refs["source_QC"]["sha256"])
            require(set(payloads) == CORE_NAMES and all(isinstance(x, bytes) for x in payloads.values())
                    and sum(map(len, payloads.values())) <= CORE_PAYLOAD_CAP, "Bounded exact pure bridge batch")
            require(report["retained_sites"] == 4148 and report["candidate_sites"] == 94892
                    and report["source_GT_biological_nucleotides_and_phase_unchanged"] is True
                    and report["HS_draws_materialized"] is False and report["AP_used"] is False,
                    "All frozen sites, unchanged biological GT/phase, no HS/AP")
            for name, original in (("baseline.sample", "planned.sample"),
                                   ("sample_ages.generations", "planned.sample_ages.generations")):
                require(hashlib.sha256(payloads[name]).hexdigest() == refs[original]["sha256"],
                        "Exact qualified sample order and fractional dates")
            mono = report["monomorphic_sites"]
            require(isinstance(mono, list) and len(set(mono)) == len(mono) and len(mono) <= 4148,
                    "Honest monomorphic site list")
            validated_counts = dict(counts_complete=True, monomorphic_site_count=len(mono),
                                    unsupported_site_count=0)
            report.update(native_gate())
            report.update(counts_complete=True, monomorphic_site_count=len(mono),
                          unsupported_site_count=0, unsupported_policy="Any unsupported retained site fails preparation; never drop",
                          native_gate_blockers=["separate_native_policy_and_admission", "fractional_native_age_roundtrip_policy"]
                              + (["monomorphic_actual40_sites"] if mono else []))
            # Keep the complete monomorphic identities in the bounded scientific preparation report.
            payloads["bridge_preparation.json"] = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
            payloads["qualified.regional.map"] = qualified_map
            payloads["NOTICE.txt"] = bounded_file(b, b.read(refs["metadata_completion"])["assets"]["NOTICE.txt"], 64_000)
            payloads["provenance.json"] = b.encoded(dict(schema="H018_observed_bridge_preparation_provenance_v1",
                condition_id=CONDITION, admitted_ref=record, bridge_ref=BRIDGE, input_refs_ref=INPUT_REFS,
                source_VCF_ref=refs["source_VCF"], source_QC_ref=refs["source_QC"],
                map_ref=refs["qualified.regional.map"], roster_ref=refs["planned_haplotype_roster.json"],
                metadata_completion_ref=refs["metadata_completion"],
                metadata_independent_validation_ref=refs["metadata_independent_result_validation"],
                metadata_original_OS_exit="unknown", metadata_original_wait0_available=False,
                metadata_original_reservation_retained_bytes=5_000_000,
                original_2Mb_failure=dict(retained=434, candidates=13312, below_1000=True, replaced=False),
                coordinate_basis="GRCh37 chr22 absolute bp0 [20000000,30000000); Oxford/map local bp1",
                dates="source calBP1950 / 28, fractional, no 2026 offset, rounding or clamp",
                map_curve="qualified piecewise public source curve copied unchanged; no uniform substitution",
                extension_normalization_bp=10_000_000, future_GNN_span_denominator="log(2000001), unchanged",
                **native_gate()))
            require(set(payloads) == FINAL_NAMES, "Exact flat preparation artifact set")
            assets = {name: dict(path=OUTPUT+"/"+name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
                      for name, raw in payloads.items()}
            completed = dict(schema="H018_observed_bridge_completed_v1",
                status="completed_baseline_GT_preparation_not_native_admitted", job_id=JOB,
                condition_id=CONDITION, admitted_ref=record, assets=assets, candidate_sites=94892,
                retained_sites=4148, haplotypes=40, reference_haplotypes=16, query_haplotypes=24,
                counts_complete=True, monomorphic_site_count=len(mono), unsupported_site_count=0,
                monomorphic_details_ref=assets["bridge_preparation.json"], sites_dropped=0,
                source_GT_biological_nucleotides_and_phase_unchanged=True,
                metadata_original_OS_exit="unknown", metadata_original_wait0_available=False,
                metadata_original_reservation_retained_bytes=5_000_000,
                preparation_elapsed_seconds=time.monotonic()-started,
                peak_worker_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                address_space_cap_bytes=MEMORY, single_CPU_affinity=sorted(os.sched_getaffinity(0)),
                no_internal_OS_exit_or_wait0_claim=True, reservation_remains_fully_charged=True,
                historical_charges_unchanged=True, finished_utc=utc(), **native_gate())
            terminal = b.encoded(completed)
            require(sum(map(len, payloads.values())) + len(terminal) + 131072 <= OUTPUT_CAP,
                    "Whole output batch and filesystem slack <=32MB")
            m.budget_check_v9(storage, boundary=True, required_write_bytes=OUTPUT_CAP)
            output_write_admitted = True
            output.mkdir(); syncdir(output.parent)
            for name, raw in payloads.items():
                write_bytes(output, name, raw)
            # The completed marker is last, after durable validated inputs.
            write_bytes(output, "completed.json", terminal)
            syncdir(output); syncdir(output.parent)
            completion_ref = b.ref(output / "completed.json")
            print(json.dumps(dict(completed_ref=completion_ref, native_execution_authorized=False)), flush=True)
            return 0
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, prior)
    except BaseException as exc:
        failure = dict(schema="H018_observed_bridge_failed_v1", job_id=JOB, admitted_ref=record,
            status="failed_attempt_preserved", error=error_record(exc), counts_complete=False,
            monomorphic_site_count=None, unsupported_site_count=None,
            unsupported_count_reason="Unknown until full retained-site validation completes",
            output_write_boundary_passed=output_write_admitted, original_2Mb_failure_replaced=False,
            metadata_original_OS_exit="unknown", metadata_original_wait0_available=False,
            reservation_remains_fully_charged=True, partial_files_preserved=True, no_replay_or_credit=True,
            finished_utc=utc(), **native_gate())
        if validated_counts is not None:
            failure.update(validated_counts, unsupported_count_reason="All retained sites validated before later write/control failure")
        # If no output boundary passed, only existing compact root packet metadata
        # is eligible. Never create an unguarded data output after a failed guard.
        if b is not None:
            try:
                destination = output if output_write_admitted and output.exists() else b.local(PACKET)
                save(b, destination, "failed.json" if destination == output else "worker_failed.json", failure)
            except BaseException as persist_exc:
                failure["failure_receipt_write_error"] = error_record(persist_exc)
        print(json.dumps(failure, allow_nan=False), flush=True)
        return 1


def group_absent(pgid):
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def rss(pid):
    # Only the exact owned live worker is inspected, never an unrelated process scan.
    try:
        with open("/proc/" + str(pid) + "/status", encoding="ascii") as stream:
            for line in stream:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])*1024
    except FileNotFoundError:
        return 0
    return 0


def direct_wait_owned(proc, started, control_error):
    """Reap this exact child using its actual OS wait status, never a guessed rc.

    WNOHANG leaves a live leader unreaped while deadline/group signals may be
    needed. Once waitpid returns this child, no group signal is sent again.
    """
    terminate_at = None
    killed = False
    while True:
        try:
            now = time.monotonic()
            if control_error is not None or now-started > SUPERVISOR_SECONDS:
                if control_error is None:
                    control_error = dict(type="TimeoutError", message="Owned child outlived supervisor deadline")
                if terminate_at is None:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    terminate_at = now
                elif now-terminate_at >= GRACE_SECONDS and not killed:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    killed = True
            waited_pid, status = os.waitpid(proc.pid, os.WNOHANG)
            if waited_pid == proc.pid:
                require(os.WIFEXITED(status) or os.WIFSIGNALED(status), "Terminal genuine child OS wait status")
                rc = os.waitstatus_to_exitcode(status)
                proc.returncode = rc  # Prevent Popen destructor from attempting a second reap.
                return dict(actual_exit_code=rc, actual_OS_wait_status=status,
                            actual_direct_wait_observed=True, control_error=control_error)
            require(waited_pid == 0, "waitpid returned an unrelated child")
            time.sleep(0.05)
        except ChildProcessError as exc:
            # External reaping cannot be converted to wait0 or a drain claim.
            return dict(actual_exit_code=None, actual_OS_wait_status=None,
                        actual_direct_wait_observed=False, control_error=error_record(exc))
        except (InterruptedError, KeyboardInterrupt, SystemExit) as exc:
            control_error = error_record(exc)
            continue
        except BaseException as exc:
            # Preserve ownership and continue to the exact OS wait after control
            # errors. Repeated inability to signal cannot justify a fake exit.
            control_error = error_record(exc)
            if terminate_at is None:
                terminate_at = time.monotonic()-GRACE_SECONDS


def supervise(proc):
    """All monitoring setup is inside the direct-wait finally protection."""
    captured, total, peak, control_error = bytearray(), 0, 0, None
    started = time.monotonic()
    selector = None
    try:
        selector = selectors.DefaultSelector()
        os.set_blocking(proc.stdout.fileno(), False)
        selector.register(proc.stdout, selectors.EVENT_READ)
        eof = False
        while not eof:
            if time.monotonic()-started > SUPERVISOR_SECONDS:
                raise TimeoutError("600-second supervisor deadline exceeded")
            current = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 + rss(proc.pid)
            peak = max(peak, current)
            require(current <= COMBINED_RSS, "Two-process monitored RSS cap")
            for key, events in selector.select(timeout=0.25):
                chunk = os.read(proc.stdout.fileno(), 65536)
                if not chunk:
                    eof = True
                    break
                total += len(chunk)
                room = LOG_CAP-len(captured)
                captured.extend(chunk[:max(0, room)])
                require(total <= LOG_CAP, "Bounded combined worker stdout/stderr")
    except BaseException as exc:
        control_error = error_record(exc)
    finally:
        waited = direct_wait_owned(proc, started, control_error)
        # The leader is reaped, or an honest external-wait error was returned.
        # Do not signal the numeric group after either event.
        drained = group_absent(proc.pid) if waited["actual_direct_wait_observed"] else False
        observation = dict(schema="H018_observed_bridge_direct_wait_observation_v1", worker_pid=proc.pid,
            supervisor_pid=os.getpid(), process_group=proc.pid, **waited,
            owned_group_absent_after_wait=drained, process_group_drained=drained,
            observed_utc=utc(), observed_before_risky_postprocessing=True,
            log_total_bytes=total, log_captured_bytes=len(captured), peak_combined_RSS_bytes=peak)
        cleanup_errors = []
        for item in (selector, proc.stdout):
            if item is not None:
                try:
                    item.close()
                except BaseException as exc:
                    cleanup_errors.append(error_record(exc))
        if cleanup_errors:
            observation["cleanup_errors"] = cleanup_errors
            observation["control_error"] = observation["control_error"] or cleanup_errors[0]
        # Print immediately after genuine wait/drain before any guard/log/output
        # postprocessing. The caller next persists the same compact observation.
        try:
            print(json.dumps(dict(direct_wait_observation=observation)), flush=True)
        except BaseException as exc:
            observation["stdout_receipt_error"] = error_record(exc)
    return observation, bytes(captured)


def verify_output(b, admitted_ref):
    terminal = b.ref(b.local(OUTPUT) / "completed.json")
    result = b.read(terminal)
    require(result["schema"] == "H018_observed_bridge_completed_v1"
            and result["status"] == "completed_baseline_GT_preparation_not_native_admitted"
            and result["admitted_ref"] == admitted_ref and result["retained_sites"] == 4148
            and result["counts_complete"] is True and result["unsupported_site_count"] == 0
            and result["sites_dropped"] == 0 and result["native_execution_authorized"] is False
            and result["native_inference_executed"] is False
            and result["metadata_original_OS_exit"] == "unknown"
            and result["metadata_original_wait0_available"] is False, "Genuine preparation-only producer completion")
    require(set(result["assets"]) == FINAL_NAMES, "Exact completed artifact closure")
    actual_names = {p.name for p in b.local(OUTPUT).iterdir()}
    require(actual_names == FINAL_NAMES | {"completed.json"}, "No unexpected or failed output artifact")
    total = 0
    for name, rec in result["assets"].items():
        require(rec["path"] == OUTPUT+"/"+name, "Actual output artifact scope")
        checked = bounded_file(b, rec, OUTPUT_CAP, retain=False)
        total += checked["bytes"]
    require(total + b.local(terminal["path"]).stat().st_size + 131072 <= OUTPUT_CAP, "Whole verified output <=32MB")
    return terminal


def launch(record):
    b, ad, r, refs, storage = admitted_inputs(record)
    # Resolve/check every reference and every destination before Popen. All Path
    # arguments passed to common.ref/save are absolute, avoiding the old pitfall.
    packet, output, aux = (b.local(x) for x in (PACKET, OUTPUT, AUX))
    script, python = b.local(SOURCE), b.local(PYTHON["path"])
    driver_ref = b.ref(script)
    launch_path, wait_path, log_path = aux/"launch.json", aux/"actual_wait.json", aux/"worker.log"
    observation_path = packet/"direct_wait_observation.json"
    require(all(p.is_absolute() and p.resolve() == p for p in
                (packet, output, aux, script, python, launch_path, wait_path, log_path, observation_path)),
            "All controller paths resolved before Popen")
    require(not output.exists() and not aux.exists(), "Preserve old output/aux; no replay")
    argv = [str(python), "-B", str(script), "worker", "--execute", "--record", record["path"],
            "--sha256", record["sha256"]]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
               PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT / "src"))
    require(signal.getsignal(signal.SIGCHLD) != signal.SIG_IGN, "Child OS wait must remain available")
    limit_process()
    m = b.guard()
    m.budget_check_v9(storage, first=True, boundary=True, required_write_bytes=AUX_WRITE)
    save(b, packet, "launch_claim.json", dict(admitted_ref=record, supervisor_pid=os.getpid(),
        started_utc=utc(), retries_authorized=False, native_execution_authorized=False))
    authority = save(b, packet, "worker_authority.json", dict(admitted_ref=record, supervisor_pid=os.getpid(),
        argv_action="worker", driver_ref=driver_ref, issued_utc=utc(), native_execution_authorized=False))
    aux.mkdir(); syncdir(aux.parent)
    prelaunch = dict(schema="H018_observed_bridge_prelaunch_v1", admitted_ref=record, driver_ref=driver_ref,
        worker_authority_ref=authority, supervisor_pid=os.getpid(), argv=argv, cwd=str(ROOT),
        single_CPU_affinity=sorted(os.sched_getaffinity(0)), started_utc=utc(), native_execution_authorized=False)
    save(b, aux, "prelaunch.json", prelaunch)
    # This try/finally encloses the first statement after successful Popen. The
    # only child launched anywhere is this exact CPU preparation worker.
    proc = None
    observation, log = None, b""
    try:
        proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True,
            close_fds=True)
    except BaseException as exc:
        save(b, packet, "failed.json", dict(schema="H018_observed_bridge_launch_failed_v1",
            admitted_ref=record, worker_spawned=False, error=error_record(exc), failed_utc=utc(),
            reservation_remains_fully_charged=True, partial_files_preserved=True))
        return 1
    finally:
        if proc is not None:
            observation, log = supervise(proc)  # Always waits even when monitor setup fails.
    # Genuine direct observation is durable before riskier guard/output validation.
    direct_receipt_error = None
    try:
        save(b, packet, "direct_wait_observation.json", dict(observation, admitted_ref=record))
    except BaseException as exc:
        direct_receipt_error = error_record(exc)
        print(json.dumps(dict(direct_wait_observation=observation,
                              durable_direct_wait_receipt_error=direct_receipt_error)), flush=True)
    final = dict(schema="H018_observed_bridge_actual_wait_v1", admitted_ref=record,
        **{k: v for k, v in observation.items() if k != "schema"},
        success=False, completed_ref=None, reservation_remains_fully_charged=True,
        partial_files_preserved=True, native_execution_authorized=False,
        metadata_original_OS_exit="unknown", metadata_original_wait0_available=False)
    post_error = direct_receipt_error
    try:
        require(direct_receipt_error is None, "Direct wait receipt durability failed; retain stdout observation")
        m.budget_check_v9(storage, boundary=True, required_write_bytes=AUX_WRITE)
        write_bytes(aux, "worker.log", log); syncdir(aux)
        launch_ref = save(b, aux, "launch.json", dict(prelaunch, schema="H018_observed_bridge_launch_v1",
            worker_pid=proc.pid, process_group=proc.pid, actual_spawn_recorded_after_wait=True))
        final["launch_ref"] = launch_ref
        require(observation["actual_direct_wait_observed"] is True
                and observation["actual_exit_code"] == 0 and observation["process_group_drained"] is True
                and observation["control_error"] is None, "Actual direct wait0 and owned-group drain required")
        final["completed_ref"] = verify_output(b, record)
        final["success"] = True
    except BaseException as exc:
        post_error = error_record(exc)
    final["postprocessing_error"] = post_error
    final["status"] = "completed_controller_wait_and_preparation_verified" if final["success"] else "failed_attempt_preserved"
    final["finished_utc"] = utc()
    # Root metadata fallback preserves the genuine direct observation if an aux
    # guard/log failure makes an auxiliary-scope final receipt ineligible.
    destination = aux if post_error is None else packet
    result = save(b, destination, "actual_wait.json", final)
    if not final["success"]:
        save(b, packet, "failed.json", dict(schema="H018_observed_bridge_controller_failed_v1",
            status="failed_attempt_preserved", actual_wait_ref=result,
            actual_exit_code=observation["actual_exit_code"],
            actual_direct_wait_observed=observation["actual_direct_wait_observed"],
            process_group_drained=observation["process_group_drained"],
            control_error=observation["control_error"], postprocessing_error=post_error,
            reservation_remains_fully_charged=True, partial_files_preserved=True,
            no_replay_or_credit=True, native_execution_authorized=False, failed_utc=utc()))
    print(json.dumps(dict(actual_wait_ref=result, success=final["success"],
                          reservation_remains_fully_charged=True)), flush=True)
    return 0 if final["success"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("admit", "launch", "worker"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--record")
    parser.add_argument("--sha256")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="inactive_source_only", root_owns_execution=True,
                              observed_payloads_opened=False, native_execution_authorized=False)))
        return 0
    require(args.record and args.sha256 and Path.cwd().resolve() == ROOT,
            "Explicit exact root request/admission ref and project cwd")
    record = dict(path=args.record, sha256=args.sha256)
    if args.action == "admit":
        admit(record)
        return 0
    return launch(record) if args.action == "launch" else worker(record)


if __name__ == "__main__":
    raise SystemExit(main())
