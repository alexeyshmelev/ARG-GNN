"""Root-only CPU caller for unchanged H055 v2. Import is inert and stdlib only.

No admission/lease/storage framework is implemented here. Root supplies one
exact, already-admitted input reference and genuine original V9 storage.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, importlib.util, json, os, re, resource, signal, stat, sys, time

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
HERE = "research/reviews/H055_CPU_runtime_source_20261016_v1"
SOURCE = HERE + "/runtime_v1.py"
SCIENCE = "research/reviews/Danish_genotype_corroboration_source_20261016_v2"
FIXED_REFS = {
    "producer": {"path": SCIENCE + "/corroboration_v2.py", "sha256": "fdd145e0a34dc00c30999c5047f5469f92bf75d88703a5783cc7f3f3e1390a36"},
    "protocol": {"path": SCIENCE + "/protocol.source_frozen.inactive.v2.json", "sha256": "2e48ece48e69d8600cc6c88649e451ffe0ff5034e2322d499db1ce4d9de5da6a"},
    "roster": {"path": SCIENCE + "/candidate_roster.metadata.v1.tsv", "sha256": "eefcb18bb4c59d959b26d1e6b8a4a62c4eb0c9bbcfa89a7602797716e6d3cf46"},
    "registration_snapshot": {"path": "research/reviews/H055_root_integration_20261016_v1/registered_H055_frozen_before_data.root_v1.json", "sha256": "bcc9606966a2b2da05c7067eaaaf1ea468ba7e702e237495339b70bf74bc8fac"},
    "guard": {"path": "research/reviews/prospective_980GB_capacity_source_v1/prospective_980gb_v1.py", "sha256": "fc71030865e0a3d5da31120d4daa2caa0b2d39ec7dd212b8c46c476b5a866d56"},
    "policy": {"path": "research/reviews/prospective_980GB_capacity_source_v1/activation_policy.root_active_v1.json", "sha256": "eb3408004a508136aa434c0a2e02a9d14a5598a96b8483c264c93026f9326018"},
    "checker": {"path": "research/reviews/GPU45_TRAIN_controller_execution_override_20261016_v1/controller_metadata_scope_check_v1.py", "sha256": "2621353de0dc56f328f7f308ba17a341425ef2fc02d54739b6ccfc7d1a6ce2cf"},
    "renewal": {"path": "research/reviews/H039_marker_repair_renewal_source_20261016_v1/marker_repair_renewal_v1.py", "sha256": "491236877e8d71cae8bd7619880858ac501c57dd93e59479ca32dacacab20468"},
    "renewal_certificate": {"path": "research/reviews/H039_marker_repair_renewal_source_20261016_v1/renewal_certificate.root_active_v1.json", "sha256": "66006202094d7a3ef2b29cbec28eb93f334505b72589f09d86e4b57498162db7"},
    "repair_authority": {"path": "research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json", "sha256": "f6f2d628199e8be73db8b5d90d428570d8afdefeab9dc5cd94bb5eafb39247db"},
    "memo": {"path": "research/reviews/resource_guard_hotpath_source_audit_v1/transaction_memo_adapter_v2.py", "sha256": "c972e9a779d424ceafdb258848614bf40994d1004e2c73fc5cd225ae99621364"},
    "v4": {"path": "src/arggnn/project_budget_v4.py", "sha256": "a093c7d86f57fee569cb99f4712250e9505a1119f3950fa44282c7ba8c57df47"},
}
SCIENCE_CAP, CHARGED_CAP, MAX_ROWS = 16 * 2**20, 32 * 2**20, 100000
RSS_CAP, DEADLINE, ROOT_METADATA_CAP = 2 * 2**30, 4 * 3600, 512 * 1024
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
_RAN = False


def require(ok, message):
    if not ok:
        raise ValueError(message)


def path(name):
    require(isinstance(name, str) and name and not Path(name).is_absolute()
            and ".." not in Path(name).parts, "Relative contained project pathname")
    p = ROOT / name
    require(p != ROOT and p.resolve() == p and not p.is_symlink(), "Canonical non-symlink project path")
    return p


def exact_ref(ref):
    require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}
            and isinstance(ref["sha256"], str) and re.fullmatch("[0-9a-f]{64}", ref["sha256"]),
            "Exact relative path/SHA256 reference")
    path(ref["path"])
    return ref


def raw(ref, limit=32 * 2**20):
    exact_ref(ref)
    p = path(ref["path"])
    before = p.lstat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_nlink, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size <= limit,
            "Bounded singly linked metadata/source")
    with os.fdopen(os.open(p, os.O_RDONLY | os.O_NOFOLLOW), "rb") as f:
        require(identity(os.fstat(f.fileno())) == identity(before), "Unchanged bound open identity")
        data = f.read(limit + 1)
        require(identity(os.fstat(f.fileno())) == identity(before), "Unchanged bound read identity")
    require(len(data) <= limit and identity(p.lstat()) == identity(before)
            and hashlib.sha256(data).hexdigest() == ref["sha256"], "Exact unchanged bound bytes")
    return data


def read(ref, limit=32 * 2**20):
    def pairs(items):
        result = {}
        for k, v in items:
            require(k not in result, "No duplicate JSON keys")
            result[k] = v
        return result
    def invalid(value):
        raise ValueError("No nonfinite JSON constant: " + value)
    return json.loads(raw(ref, limit), object_pairs_hook=pairs, parse_constant=invalid)


def source_ref():
    require(Path(__file__).resolve() == path(SOURCE), "Actual runtime source path")
    return {"path": SOURCE, "sha256": hashlib.sha256(path(SOURCE).read_bytes()).hexdigest()}


def unique(refs):
    result = {}
    for ref in refs:
        exact_ref(ref)
        require(ref["path"] not in result or result[ref["path"]] == ref["sha256"], "No conflicting source pins")
        result[ref["path"]] = ref["sha256"]
    return [{"path": p, "sha256": h} for p, h in sorted(result.items())]


def load(ref, label):
    data = raw(ref)
    spec = importlib.util.spec_from_file_location(label, path(ref["path"]))
    module = importlib.util.module_from_spec(spec)
    sys.modules[label] = module
    # Execute the authenticated source bytes, without producer/guard pyc output.
    exec(compile(data, str(path(ref["path"])), "exec"), module.__dict__)
    return module


def run_v1(root_input_ref):
    """Single fresh CPU process; root owns the genuine admission and input pin."""
    global _RAN
    require(not _RAN and sys.platform.startswith("linux"), "One fresh Linux CPU caller invocation")
    _RAN = True
    started = time.monotonic()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    require(not any(n in sys.modules for n in ("torch", "torch_geometric", "tensorflow", "jax", "cupy")),
            "Fresh interpreter without model/GPU imports")
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    # A stricter virtual-address ceiling also bounds self RSS between callbacks.
    limit = min([RSS_CAP] + [v for v in (soft, hard) if v != resource.RLIM_INFINITY])
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    spec = read(root_input_ref, 65536)
    required_keys = {"schema", "status", "root_authorizer", "hypothesis_id", "registered", "execution_authorized",
                     "job_id", "output_directory", "created_at_utc", "expires_at_utc", "runtime_entry_ref", "refs"}
    require(set(spec) == required_keys and spec["schema"] == "H055_CPU_root_input_v1"
            and spec["status"] == "active" and spec["root_authorizer"] == "root_agent"
            and spec["hypothesis_id"] == "H055" and spec["registered"] is True
            and spec["execution_authorized"] is True and isinstance(spec["job_id"], str)
            and NAME.fullmatch(spec["job_id"]) and spec["runtime_entry_ref"] == source_ref()
            and set(spec["refs"]) == {"config", "runtime", "source_closure", "storage"},
            "Strict root H055 CPU input and actual runtime binding")
    issued = datetime.fromisoformat(spec["created_at_utc"].replace("Z", "+00:00"))
    expires = datetime.fromisoformat(spec["expires_at_utc"].replace("Z", "+00:00"))
    require(issued.tzinfo is not None and expires.tzinfo is not None and issued <= datetime.now(timezone.utc) < expires
            and 0 < (expires - issued).total_seconds() <= DEADLINE, "Current absolute authority at most four hours")
    end = expires.timestamp()
    out = path(spec["output_directory"])
    require(spec["output_directory"].startswith("experiments/results/") and out.parent.is_dir()
            and not out.exists(), "Fresh result output and existing charged parent")
    metadata_folder = path(spec["refs"]["config"]["path"]).parent
    require(metadata_folder != out and out not in metadata_folder.parents and metadata_folder not in out.parents,
            "Separate preexisting root metadata and fresh scientific scope")
    require(all(path(r["path"]).parent == metadata_folder for r in spec["refs"].values()),
            "Config/runtime/closure/actual storage in one fresh root metadata folder")
    for ref in FIXED_REFS.values():
        raw(ref)
    protocol = read(FIXED_REFS["protocol"])
    for ref in protocol["source_refs"]:
        raw(ref)
    require(protocol["roster_ref"] == FIXED_REFS["roster"], "Frozen transitive roster reference")
    snapshot = read(FIXED_REFS["registration_snapshot"])
    require(snapshot["schema"] == "H055_root_registered_pre_data_snapshot_v1"
            and snapshot["root_authorizer"] == "root_agent" and snapshot["hypothesis_id"] == "H055"
            and snapshot["registered"] is True and snapshot["execution_authorized"] is False
            and snapshot["snapshot_is_not_execution_authority"] is True
            and snapshot["scientific_source_ref"] == FIXED_REFS["producer"]
            and snapshot["scientific_protocol_ref"] == FIXED_REFS["protocol"] and snapshot["roster_ref"] == FIXED_REFS["roster"]
            and all(snapshot[k] == protocol[k] for k in ("subjects", "earlier", "later", "contrasts"))
            and snapshot["windows"] == {"chromosomes": ["20", "21", "22"], "GRCh37_bp0_half_open": [20000000, 30000000]}
            and snapshot["no_mutable_decisions_file_dependency"] is True, "Frozen pre-data H055 registration snapshot")
    config = dict(schema="H055_CPU_config_v1", root_authorizer="root_agent", hypothesis_id="H055",
                  fixed_refs=FIXED_REFS, device="cpu", physical_gpu=None, cuda_visible_devices="",
                  max_cuda_reserved_bytes=0, max_cpu_rss_bytes=RSS_CAP, global_wall_seconds=DEADLINE,
                  scientific_output_cap_bytes=SCIENCE_CAP, max_metadata_eligible_SNPs=MAX_ROWS,
                  charged_scope_cap_bytes=CHARGED_CAP, auxiliary_scopes=[],
                  original_scope_checker_reuse_root_authorized=True)
    require(read(spec["refs"]["config"]) == config, "Exact fixed source and CPU/scientific/charged-scope caps")
    runtime = {k: spec[k] for k in ("job_id", "output_directory", "created_at_utc", "expires_at_utc")}
    runtime.update(schema="H055_CPU_runtime_v1", root_authorizer="root_agent", device="cpu",
                   physical_gpu=None, cuda_visible_devices="", global_wall_seconds=DEADLINE)
    require(read(spec["refs"]["runtime"]) == runtime, "Exact job/output/expiry runtime metadata")
    closure = read(spec["refs"]["source_closure"])
    bound_refs = dict(FIXED_REFS, runtime_entry=spec["runtime_entry_ref"],
                      config=spec["refs"]["config"], runtime=spec["refs"]["runtime"])
    require(set(closure) == {"schema", "root_authorizer", "bound_refs", "original_bindings"}
            and closure["schema"] == "H055_CPU_source_closure_v1" and closure["root_authorizer"] == "root_agent"
            and closure["bound_refs"] == bound_refs and isinstance(closure["original_bindings"], dict)
            and closure["original_bindings"], "Root exact source closure plus all actual original state bindings")
    transitive = [r for r in protocol["source_refs"] if Path(r["path"]).suffix in {".py", ".json", ".sh"}]
    # Original v4 permits PY/SH/JSON direct bindings only; TSV/TXT stay authenticated
    # transitively through exact protocol/snapshot/closure. No proxy binding.
    direct = unique([*closure["original_bindings"].values(),
                     *(r for r in bound_refs.values() if Path(r["path"]).suffix in {".py", ".json", ".sh"}),
                     spec["refs"]["source_closure"], *transitive])
    require(len(direct) <= 64 and all(Path(r["path"]).suffix in {".py", ".json", ".sh"} for r in direct),
            "Unchanged original 64 direct metadata/source binding limit")
    storage = read(spec["refs"]["storage"])
    own = storage["reservation"]
    require(storage.get("kind") == "project_budget_v9_storage_record" and storage.get("active") is True
            and storage.get("consumer_uses_v9_guard") is True and storage.get("accounting_mode") == "prospective_980gb_v1"
            and storage["job_id"] == spec["job_id"] and storage["output_directory"] == spec["output_directory"]
            and storage["expires_at_utc"] == spec["expires_at_utc"]
            and type(storage["reserved_new_bytes"]) is int and storage["reserved_new_bytes"] == own["reserved_bytes"] == CHARGED_CAP
            and own["output"] == str(out) and own["auxiliary_scopes"] == []
            and isinstance(own["key"], str) and own["key"].startswith("H055/")
            and isinstance(own["consumer_bindings"], list) and len(own["consumer_bindings"]) <= 64
            and unique(own["consumer_bindings"]) == direct
            and storage["completed_overlay_policy_ref"] == FIXED_REFS["policy"],
            "Actual original open fully charged32MiB/noaux job/output/cap/expiry/direct bindings")
    require(metadata_folder.name and not out.exists(), "Output remains absent before producer Writer")
    filesystem = os.statvfs(out.parent)
    allocation_unit = max(4096, filesystem.f_frsize, filesystem.f_bsize)
    # At most nine producer files (including a failure), two root receipts and
    # one flat directory. Whole-unit overhead stays in the extra charged slack.
    allocation_allowance = 12 * allocation_unit
    require(allocation_unit <= 2**20 and SCIENCE_CAP + 4096 + ROOT_METADATA_CAP + allocation_allowance <= CHARGED_CAP
            and CHARGED_CAP <= filesystem.f_bavail * filesystem.f_frsize, "Conservative file/block/directory slack and free bytes")
    counters = dict(directory_bytes=0, producer_serialized_bytes=0, root_serialized_bytes=0,
                    write_callbacks=0, request_callbacks=0, maximum_requested_response_bytes=0)
    checker = None
    previous_handler = signal.getsignal(signal.SIGALRM)
    require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), "Fresh process without another wall timer")
    def timeout(signum, frame):
        raise TimeoutError("H055 global wall deadline or absolute root authority expiry")
    remaining = min(DEADLINE - (time.monotonic() - started), end - time.time())
    require(remaining > 0, "Current global deadline before installing wall timer")
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    def runtime_check():
        require(time.monotonic() - started < DEADLINE and time.time() < end, "Global four-hour/absolute root expiry")
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not any(
            n in sys.modules for n in ("torch", "torch_geometric", "tensorflow", "jax", "cupy")),
            "CPU-only empty CUDA and no model/GPU import")
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= RSS_CAP,
                "Linux peak self RSS at most2GiB")
    try:
        runtime_check()
        sys.dont_write_bytecode = True
        sys.path[:0] = [str(ROOT), str(ROOT / "src")]
        renewal = load(FIXED_REFS["renewal"], "_H055_original_marker_renewal")
        renewal.install_renewal_v1(FIXED_REFS["renewal_certificate"])
        certificate, _ = renewal.validate_certificate_v1(FIXED_REFS["renewal_certificate"])
        require(end < datetime.fromisoformat(certificate["renewed_expires_at_utc"].replace("Z", "+00:00")).timestamp(),
                "Job absolute expiry fits original authentic renewal")
        guard = load(FIXED_REFS["guard"], "_H055_original_980_guard")
        guard.configure_v2(FIXED_REFS["repair_authority"], policy_ref=FIXED_REFS["policy"])
        checker_module = load(FIXED_REFS["checker"], "_H055_original_scope_checker")
        checker = checker_module.ControllerMetadataScopeV1(guard, storage)
        # Constructor performs exactly one full original startup guard.
        require(checker.bindings == closure["original_bindings"], "All actual original full-state bindings unchanged")
        preflight = checker.check(required=SCIENCE_CAP + ROOT_METADATA_CAP)
        runtime_check()
        require(not out.exists(), "Only the original producer Writer may create output")
        last_charge, last_check = 0, time.monotonic()
        scope_checks = 1
        latest_scope = preflight
        def charge():
            return counters["directory_bytes"] + counters["producer_serialized_bytes"] + counters["root_serialized_bytes"]
        def boundary(pending=0, force=False):
            nonlocal last_charge, last_check, scope_checks, latest_scope
            runtime_check()
            require(charge() + pending + allocation_allowance <= CHARGED_CAP, "Root serialized/mkdir/block overhead counter fits32MiB")
            now = time.monotonic()
            if force or charge() + pending - last_charge >= 2**20 or now - last_check >= 60:
                latest_scope = checker.check(required=pending + allocation_allowance)
                last_charge, last_check = charge() + pending, time.monotonic()
                scope_checks += 1
                runtime_check()
        def before_request(size):
            require(type(size) is int and 0 <= size <= 2**20 + 1, "Bounded original HEAD/Range response request")
            boundary()
            counters["request_callbacks"] += 1
            counters["maximum_requested_response_bytes"] += size
        def before_write(size):
            require(type(size) is int and 0 <= size <= SCIENCE_CAP, "Exact producer write/mkdir count")
            if counters["write_callbacks"] == 0:
                require(size == 4096 and not out.exists(), "Original Writer creates the still-fresh directory")
                boundary(size)
                counters["directory_bytes"] = size
            else:
                require(counters["producer_serialized_bytes"] + size <= SCIENCE_CAP, "Scientific serialized content remains16MiB")
                boundary(size)
                counters["producer_serialized_bytes"] += size
            counters["write_callbacks"] += 1
        producer = load(FIXED_REFS["producer"], "_H055_exact_corroboration_v2")
        require(producer.CAP == SCIENCE_CAP and producer.MAX_ROWS == MAX_ROWS, "Original scientific producer limits")
        authorization = dict(hypothesis_id="H055", registered=True, execution_authorized=True,
                             source_sha256=FIXED_REFS["producer"]["sha256"], protocol_sha256=FIXED_REFS["protocol"]["sha256"],
                             output_cap_bytes=SCIENCE_CAP)
        result = producer.run_v1(path(FIXED_REFS["protocol"]["path"]), out, root_authorization=authorization,
                                 before_request=before_request, before_write=before_write)
        # No root durable write occurs until the unchanged producer returns success.
        runtime_check()
        def produced(name):
            p = out / name
            require(p.is_file() and not p.is_symlink() and p.stat().st_size <= ROOT_METADATA_CAP, "Bounded original completion metadata")
            ref = {"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            return ref, read(ref, ROOT_METADATA_CAP)
        completed_ref, completed = produced("completed.json")
        summary_ref, summary = produced("summary.json")
        require(completed["status"] == "completed_registered_genotype_only_corroboration"
                and completed["protocol_sha256"] == FIXED_REFS["protocol"]["sha256"]
                and completed["cap_bytes"] == SCIENCE_CAP and completed["record_bytes"] == 160
                and type(completed["retained_rows"]) is int and 0 < completed["retained_rows"] <= MAX_ROWS
                and summary == result and summary["metadata_eligible_retained_SNPs"] == completed["retained_rows"]
                and not (out / "failed.json").exists(), "Actual original successful completion and returned summary")
        boundary(force=True)
        def save_root(name, value):
            require(name in {"root_runtime_binding.json", "root_runtime_receipt.json"}, "Separately named root receipts only")
            data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
            require(counters["root_serialized_bytes"] + len(data) <= ROOT_METADATA_CAP, "Root receipt512KiB total cap")
            boundary(len(data), force=True)
            p = out / name
            with os.fdopen(os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "wb") as f:
                f.write(data); f.flush(); os.fsync(f.fileno())
                written = os.fstat(f.fileno())
            named = p.lstat()
            require(stat.S_ISREG(named.st_mode) and named.st_nlink == 1 and named.st_size == len(data)
                    and (written.st_dev, written.st_ino) == (named.st_dev, named.st_ino), "Exclusive root receipt identity")
            counters["root_serialized_bytes"] += len(data)
            return {"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(data).hexdigest()}
        binding_ref = save_root("root_runtime_binding.json", dict(
            schema="H055_actual_CPU_runtime_binding_v1", input_ref=root_input_ref, storage_ref=spec["refs"]["storage"],
            runtime_entry_ref=spec["runtime_entry_ref"], config_ref=spec["refs"]["config"], runtime_ref=spec["refs"]["runtime"],
            source_closure_ref=spec["refs"]["source_closure"], fixed_refs=FIXED_REFS, job_id=spec["job_id"],
            output_directory=spec["output_directory"], device="cpu", CUDA_VISIBLE_DEVICES="", model_or_torch_imported=False,
            original_producer_returned_successfully=True, producer_completed_ref=completed_ref, producer_summary_ref=summary_ref,
            scientific_output_cap_bytes=SCIENCE_CAP, charged_scope_cap_bytes=CHARGED_CAP,
            original_scope_checker_reuse_root_authorized=True, all_original_direct_bindings_verified=True))
        final_ref = save_root("root_runtime_receipt.json", dict(
            schema="H055_CPU_root_runtime_receipt_v1", status="completed_H055_original_producer_and_root_binding",
            job_id=spec["job_id"], completed_utc=datetime.now(timezone.utc).isoformat(), input_ref=root_input_ref,
            storage_ref=spec["refs"]["storage"], runtime_binding_ref=binding_ref,
            producer_completed_ref=completed_ref, producer_summary_ref=summary_ref,
            counters_before_this_receipt=dict(counters), allocation_allowance_bytes=allocation_allowance,
            original_full_startup_guard_passed_once=True, scope_checks_before_this_receipt=scope_checks,
            latest_own_scope_check_before_this_receipt=latest_scope,
            subsequent_global_projection_recomputed=False, periodic_growth_bytes=2**20, periodic_seconds=60,
            peak_self_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            elapsed_seconds=time.monotonic() - started, CPU_only=True, model_or_torch_imported=False,
            producer_completed_and_summary_preserved=True, postwrite_scope_check_required_for_caller_success=True))
        runtime_check()
        final_scope = checker.check(required=0)
        scope_checks += 1
        runtime_check()
        return dict(status="completed_H055_CPU_caller", root_receipt_ref=final_ref,
                    producer_completed_ref=completed_ref, final_own_scope_check=final_scope,
                    scope_checks=scope_checks, counters=dict(counters), actual_execution=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if checker is not None:
            checker.memo.uninstall()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--input")
    p.add_argument("--expected-input-sha256")
    args = p.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="source_only_inactive", hypothesis_id="H055", device="cpu")))
        return 0
    require(args.input and args.expected_input_sha256, "Explicit root input and exact externally supplied SHA256")
    result = run_v1({"path": args.input, "sha256": args.expected_input_sha256})
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
