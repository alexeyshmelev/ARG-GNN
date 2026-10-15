"""H018 exact-three-asset root CPU acquisition. Default CLI is inert.

No subprocess, GPU launcher, package installation, genealogy/model payload, memo
adapter or accounting replacement. Only root explicitly calls admit/run.
"""
from pathlib import Path
from datetime import datetime, timedelta, timezone
from urllib.request import Request, build_opener, HTTPRedirectHandler, ProxyHandler
import argparse, hashlib, importlib.util, json, os, resource, signal, sys, time

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
REL = "research/reviews/H018_map_acquisition_root_source_v1"
HERE = ROOT / REL
SOURCE = REL + "/root_map_driver.py"
CONFIG = REL + "/config.json"
PACKET = REL + "/admission_v1"
OUTPUT = REL + "/acquired_v1"
JOB = "H018_map_sources_chr22_b37_v1"
KEY = "H018/map_sources_chr22_b37_v1"
CAP, NET_CAP, EXPECTED_BYTES = 5_000_000, 400_000, 329673
SECONDS, MEMORY = 120, 1_000_000_000
MAP_RAM_PLAN = 64_000_000  # Original unmeasured data-buffer forecast; guard/control overhead is separate.
CORE = "483d78bf2ef3c917d65ecf1dde5a7809bfadc21194c9f9a90394ce5c532fdd5e"
BASE = "a8049bc103afb1cbbbce6f8fc3707a21de9613b23034599641b688994fc5caa3"
GUARD = dict(path="research/reviews/concurrent_completed_scope_credits_v3/completed_scope_credits_v3.py",
             sha256="ea14c12c41b9385a86182aa9c24058f7b1276107fc5706c70e47828e8b2c31ef")
POLICY = dict(path="research/reviews/completed_scope_credits_root_activation_20261015_v1/activation_policy.json",
              sha256="a0885b5209172dd79a835d442e0ad3495f5c1a0eac61a1c5c08154ec877ecf85")
REPAIR = dict(path="research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json",
              sha256="f6f2d628199e8be73db8b5d90d428570d8afdefeab9dc5cd94bb5eafb39247db")
COMMIT = "103ca071ea9e4e648192916142e5ed353ac94e27"
RAW_BASE = "https://raw.githubusercontent.com/odelaneau/GLIMPSE/" + COMMIT + "/"
ASSETS = (
    dict(filename="chr22.b37.gmap.gz", source_path="maps/genetic_maps.b37/chr22.b37.gmap.gz",
         bytes=322186, git_blob_sha1="eec8ceff2fdff0d70badcb09c18e5c6cc5b04854"),
    dict(filename="gmap_reader.cpp", source_path="phase/src/io/gmap_reader.cpp",
         bytes=2584, git_blob_sha1="24a6fdf902c14c8e0b372910ef67c5aa37666fe3"),
    dict(filename="variant_map.cpp", source_path="phase/src/containers/variant_map.cpp",
         bytes=4903, git_blob_sha1="e00b1fc7bc5c2da0329ef73a659dab5a73a5ab2f"),
)
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def local(value):
    p = Path(value)
    require(not p.is_absolute() and ".." not in p.parts and p.as_posix() == value,
            "Canonical project-relative path required")
    p = ROOT / p
    require(p != ROOT and p.resolve() == p, "Canonical project path without symlinks")
    return p


def ref(path):
    p = local(path) if isinstance(path, str) else path
    require(ROOT in p.parents and p.resolve() == p and p.is_file() and not p.is_symlink(),
            "Existing regular source/metadata file")
    require(p.stat().st_size <= 4_000_000, "Small source/metadata cap")
    return dict(path=p.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def read(record):
    require(set(record) == {"path", "sha256"} and ref(record["path"]) == record,
            "Exact frozen source/metadata reference")
    return json.loads(local(record["path"]).read_bytes())


def encoded(value):
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    require(len(data) <= 65536, "Compact root receipt <=64KiB")
    return data


def save(folder, name, value):
    path = folder / name
    with path.open("xb") as stream:
        stream.write(encoded(value)); stream.flush(); os.fsync(stream.fileno())
    return ref(path)


def source_closure(request):
    cfg = read(request["config_ref"])
    manifest = read(request["source_manifest_ref"])
    require(request["config_ref"]["path"] == CONFIG
            and manifest["schema"] == "H018_exact_map_source_manifest_v1"
            and manifest["source_refs"].get("driver") == ref(SOURCE)
            and manifest["source_refs"].get("config") == request["config_ref"],
            "Exact admitted driver/config source closure")
    for record in manifest["source_refs"].values():
        require(ref(record["path"]) == record, "Source closure drift")
    require(cfg["assets"] == list(ASSETS) and cfg["reserved_bytes"] == CAP
            and cfg["network_cap_bytes"] == NET_CAP and cfg["expected_network_bytes"] == EXPECTED_BYTES
            and cfg["download_timeout_seconds"] == SECONDS and cfg["max_address_space_bytes"] == MEMORY
            and cfg["CPU_threads"] == 1 and cfg["network_retries"] == 0
            and cfg["map_data_RAM_planning_cap_bytes"] == MAP_RAM_PLAN
            and cfg["control_RSS_cap_bytes"] == MEMORY
            and cfg["guard_ref"] == GUARD and cfg["policy_ref"] == POLICY
            and cfg["repair_authority_ref"] == REPAIR
            and cfg["qualification_helper_execution_authorized"] is False,
            "Exact three immutable sources and bounded CPU settings")
    return cfg, manifest


def guard():
    for record in (GUARD, POLICY, REPAIR):
        require(ref(record["path"]) == record, "Original guard/policy/repair source drift")
    spec = importlib.util.spec_from_file_location("_H018_map_original_guard", ROOT / GUARD["path"])
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.configure_v2(REPAIR, policy_ref=POLICY)
    return module


def admit(request_ref):
    r = read(request_ref)
    require(r["schema"] == "H018_exact_map_root_request_v1" and r["status"] == "active"
            and r["root_authorizer"] == "root_agent" and r["registered_before_execution"] is True
            and r["job_id"] == JOB and r["packet_directory"] == PACKET
            and r["output_directory"] == OUTPUT and r["reserved_bytes"] == CAP
            and r["physical_gpu"] is None and r["map_semantics_qualified"] is False,
            "Explicit active exact root request")
    cfg, manifest = source_closure(r)
    decisions = r["experiment_decisions_ref"]
    read(decisions)  # Root's genuine frozen registration, never invented by source task.
    packet, output = local(PACKET), local(OUTPUT)
    require(not packet.exists() and not output.exists(), "Fresh single-use root packet/output")
    packet.mkdir()
    now = datetime.now(timezone.utc); expiry = (now + timedelta(hours=2)).isoformat()
    runtime = save(packet, "runtime.json", dict(schema="H018_exact_map_CPU_runtime_v1", active=True,
        device="cpu", physical_gpu=None, python_executable=sys.executable, python_version=sys.version,
        job_id=JOB, output_directory=OUTPUT, CPU_threads=1, max_address_space_bytes=MEMORY,
        control_RSS_cap_bytes=MEMORY, map_data_RAM_planning_cap_bytes=MAP_RAM_PLAN,
        data_RAM_forecast_measured=False,
        download_timeout_seconds=SECONDS, network_retries=0, created_at_utc=now.isoformat(),
        expires_at_utc=expiry, completed_overlay_policy_ref=POLICY))
    closure = save(packet, "consumer_manifest.json", dict(schema="H018_exact_map_consumer_v1",
        request_ref=request_ref, config_ref=r["config_ref"], source_manifest_ref=r["source_manifest_ref"],
        runtime_ref=runtime, experiment_decisions_ref=decisions, source_refs=manifest["source_refs"],
        no_VCF_tree_model_target_checkpoint_reads=True, qualification_helper_execution_authorized=False))
    m = guard()
    view = m.inspect_v1(expected_source_sha256=CORE, expected_launcher_sha256=BASE)
    consumers = list({x["path"]: x for x in [
        request_ref, r["config_ref"], r["source_manifest_ref"], runtime, closure, decisions,
        *manifest["source_refs"].values(), GUARD, POLICY, REPAIR, *view["bindings"].values()
    ]}.values())
    require(len(consumers) <= 64, "Original direct-reference bound")
    operation = m.operation_draft_v1("reserve",
        dict(key=KEY, output=str(output), auxiliary_scopes=[], reserved_bytes=CAP,
             consumer_bindings=consumers), expected_before=m._pins(view), bindings=view["bindings"])
    operation.update(active=True, expires_at_utc=expiry)
    receipt = m.apply_reserve_v1(save(packet, "reserve_authority.json", operation),
        expected_source_sha256=CORE, expected_launcher_sha256=BASE)
    save(packet, "reserve_receipt.json", receipt)
    storage = save(packet, "storage.json", dict(kind="project_budget_v9_storage_record", active=True,
        consumer_uses_v9_guard=True, accounting_mode=m.MODE, job_id=JOB, output_directory=OUTPUT,
        reserved_new_bytes=CAP, reservation=receipt["value"], expires_at_utc=expiry,
        initial_ledger_pins=receipt["initial_ledger_pins"], completed_overlay_policy_ref=POLICY,
        budget_components=dict(implementation=ref(m.budget.SOURCE_REL),
            launcher=ref(m.budget.LAUNCHER_REL), consumer_adapter=GUARD)))
    admitted = dict(schema="H018_exact_map_root_admitted_v1", driver_ref=ref(SOURCE),
        request_ref=request_ref, runtime_ref=runtime, consumer_manifest_ref=closure, storage_ref=storage,
        source_manifest_ref=r["source_manifest_ref"], job_id=JOB, output_directory=OUTPUT,
        reserved_bytes=CAP, expires_at_utc=expiry, consumer_binding_count=len(consumers),
        execution_started=False, reservation_remains_fully_charged=True)
    record = save(packet, "admitted.json", admitted)
    print(json.dumps(dict(admitted_ref=record, reserved_bytes=CAP)), flush=True)
    return record


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Redirect/fallback forbidden for exact immutable URL")


def verify_body(asset, raw):
    require(len(raw) == asset["bytes"], "Exact source body size differs")
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    require(blob == asset["git_blob_sha1"], "Exact Git source blob differs")
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), git_blob_sha1=blob)


def _deadline(signum, frame):
    raise TimeoutError("120-second bounded acquisition clock")


def acquire_in_RAM(payloads, requests):
    opener = build_opener(ProxyHandler({}), NoRedirect())
    started = time.monotonic(); total = 0
    previous = signal.signal(signal.SIGALRM, _deadline)
    signal.setitimer(signal.ITIMER_REAL, SECONDS)
    try:
        for asset in ASSETS:
            remaining = SECONDS - (time.monotonic() - started)
            require(remaining > 0 and total + asset["bytes"] + 1 <= NET_CAP,
                    "Whole acquisition deadline/body-byte cap")
            url = RAW_BASE + asset["source_path"]
            row = dict(url=url, filename=asset["filename"], expected_bytes=asset["bytes"],
                       started_utc=utc(), response_body_bytes=0, verified=False)
            requests.append(row)
            req = Request(url, headers={"User-Agent": "ARG-GNN-H018-map-source/1",
                                        "Accept-Encoding": "identity"})
            with opener.open(req, timeout=min(30, remaining)) as response:
                row.update(status=response.status, final_url=response.url,
                    Content_Length=response.headers.get("Content-Length"),
                    Content_Encoding=response.headers.get("Content-Encoding"),
                    ETag=response.headers.get("ETag"), Last_Modified=response.headers.get("Last-Modified"))
                require(response.status == 200 and response.url == url, "Exact HTTP200 immutable URL")
                require(response.headers.get("Content-Encoding") in (None, "identity"),
                        "No transfer content encoding")
                length = response.headers.get("Content-Length")
                require(length is None or length == str(asset["bytes"]), "Exact source Content-Length")
                raw = response.read(asset["bytes"] + 1)
                total += len(raw); row["response_body_bytes"] = len(raw)
                payloads[asset["filename"]] = raw  # Preserve received failure bytes under failed names.
                require(total <= NET_CAP, "Cumulative response-body byte cap")
                row.update(verify_body(asset, raw), verified=True, completed_utc=utc())
        require(len(requests) == 3 and total == EXPECTED_BYTES, "Exactly three verified immutable sources")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, previous)
    return total, time.monotonic() - started


def run(admitted_ref):
    ad = read(admitted_ref); r = read(ad["request_ref"]); cfg, manifest = source_closure(r)
    require(ad["schema"] == "H018_exact_map_root_admitted_v1" and ad["driver_ref"] == ref(SOURCE)
            and admitted_ref["path"] == PACKET + "/admitted.json"
            and ad["job_id"] == JOB and ad["output_directory"] == OUTPUT
            and ad["reserved_bytes"] == CAP and r["status"] == "active",
            "Exact original single-use admission")
    stamp = datetime.fromisoformat(ad["expires_at_utc"])
    require(stamp > datetime.now(timezone.utc), "Unexpired root admission")
    runtime, storage = read(ad["runtime_ref"]), read(ad["storage_ref"])
    require(runtime["job_id"] == JOB and runtime["device"] == "cpu"
            and runtime["physical_gpu"] is None and storage["reserved_new_bytes"] == CAP,
            "Exact admitted CPU runtime/storage")
    require(not local(OUTPUT).exists(), "Preserve any prior output; no replay")
    # Root packet metadata is covered by existing per-event metadata allowance.
    save(local(PACKET), "run_claim.json", dict(schema="H018_exact_map_single_attempt_v1",
        admitted_ref=admitted_ref, process_pid=os.getpid(), started_utc=utc(), retries_authorized=False))
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                      MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    available = os.sched_getaffinity(0); require(available, "Available CPU affinity")
    os.sched_setaffinity(0, {min(available)})
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY, MEMORY))
    m = guard()
    # One unchanged actual download preflight. No network/output has occurred.
    m.budget_check_v9(storage, first=True, boundary=True, required_write_bytes=CAP)
    require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= MEMORY, "1GB control-process RSS cap")
    received, requests, error, elapsed = {}, [], None, None
    start = time.monotonic(); began = utc()
    try:
        total, elapsed = acquire_in_RAM(received, requests)
    except Exception as exc:
        error = dict(type=type(exc).__name__, message=str(exc)[:2048])
        total = sum(len(x) for x in received.values()); elapsed = time.monotonic()-start
    report = dict(schema="H018_exact_map_source_acquisition_v1", status="verified" if error is None else "failed",
        condition_id="H018_R16_20_30Mb_extension_v1", job_id=JOB, admitted_ref=admitted_ref,
        source_commit=COMMIT, requests=requests, attempts=len(requests), received_body_bytes=total,
        network_cap_bytes=NET_CAP, expected_body_bytes=EXPECTED_BYTES, network_retries=0,
        download_timeout_seconds=SECONDS, download_elapsed_seconds=elapsed,
        single_CPU_affinity=list(os.sched_getaffinity(0)), address_space_cap_bytes=MEMORY,
        control_RSS_cap_bytes=MEMORY, peak_control_RSS_bytes_at_download_end=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        map_data_RAM_planning_cap_bytes=MAP_RAM_PLAN, data_RAM_forecast_measured=False,
        bounded_source_body_buffers_bytes=sum(len(x) for x in received.values()),
        started_utc=began, finished_utc=utc(), error=error, map_semantics_qualified=False,
        metadata_or_map_helper_executed=False, genotype_tree_model_checkpoint_target_read=False,
        child_processes_launched=False, GPU_used=False, install_performed=False)
    payloads = {name if error is None else "received_" + name: raw for name, raw in received.items()}
    payloads["acquisition.json"] = encoded(report)
    terminal_name = "completed.json" if error is None else "failed.json"
    terminal = dict(schema="H018_map_source_acquisition_completed_v1" if error is None else "H018_map_source_acquisition_failed_v1",
        status="completed_exact3_source_assets_only" if error is None else "failed_attempt_preserved",
        job_id=JOB, admitted_ref=admitted_ref, completed_utc=utc() if error is None else None,
        assets={name: dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
                for name, raw in payloads.items()}, source_bytes_verified=error is None,
        map_semantics_qualified=False, metadata_or_map_helper_executed=False, error=error,
        reservation_remains_fully_charged=True, historical_charges_unchanged=True,
        synchronous_single_process=True, no_internal_OS_exit_or_wait0_claim=True)
    terminal_bytes = encoded(terminal)
    require(sum(len(x) for x in payloads.values()) + len(terminal_bytes) + 65536 < CAP,
            "Entire durable batch fits the single fully charged reserve")
    # One unchanged actual preflight for the entire final durable batch.
    m.budget_check_v9(storage, boundary=True, required_write_bytes=CAP)
    require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= MEMORY, "1GB control-process RSS cap")
    output = local(OUTPUT); output.mkdir()
    for name, raw in payloads.items():
        with (output/name).open("xb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    # Genuine completed marker follows successful verification and durable source writes.
    with (output/terminal_name).open("xb") as stream:
        stream.write(terminal_bytes); stream.flush(); os.fsync(stream.fileno())
    fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    terminal_ref = ref(output/terminal_name)
    print(json.dumps(dict(terminal_ref=terminal_ref, success=error is None,
                          reservation_remains_fully_charged=True)), flush=True)
    return 0 if error is None else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("admit", "run"))
    p.add_argument("--execute", action="store_true"); p.add_argument("--record"); p.add_argument("--sha256")
    args = p.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="source_only_not_executed", root_owns_execution=True))); return 0
    require(args.record and args.sha256, "Exact active root request/admission path and SHA256")
    record = dict(path=args.record, sha256=args.sha256)
    if args.action == "admit":
        admit(record); return 0
    return run(record)


if __name__ == "__main__":
    raise SystemExit(main())
