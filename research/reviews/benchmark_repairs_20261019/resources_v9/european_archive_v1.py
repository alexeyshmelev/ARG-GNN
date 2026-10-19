"""Lossless closed-European-intermediate archival, never automatically activated.

Only exact completed inference scopes are eligible. Later maintained consumers
read inference.json, observed_sample_manifest.json and post_size.observed.trees.
Original observed inputs live in the generation scope and are untouched. The
allowlisted native conversion inputs are redundant after the five native steps.
Every byte remains recoverable from gzip plus an original-SHA/size manifest.
"""
from __future__ import annotations
import copy
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import stat

SOURCE = "research/reviews/benchmark_repairs_20261019/resources_v9/european_archive_v1.py"
SHIM = "src/arggnn/benchmarking/european_retention_v9.py"
MODULE = "arggnn.benchmarking.european_retention_v9"
NEW_SOURCES = (SOURCE, SHIM)
MAX_TEMP = 8 * 2**30
METADATA_CAP = 32 * 2**20
MAX_OUTPUT = 16 * 2**30 + METADATA_CAP
CANDIDATES = (
    "pre_size.anc", "pre_size.mut", "post_size.trees", "post_size.pairwise.bin",
    "inputs/input.vcf", "inputs/input.haps",
    "02_relate_initial.log", "03_five_population_size_iterations.log",
)
REQUIRED = ("pre_size.anc", "pre_size.mut", "post_size.trees")
PROTECTED = ("post_size.observed.trees", "observed_sample_manifest.json",
             "inference.json", "commands.json")
CONSUMER_SOURCES = (
    "src/arggnn/benchmarking/european_workflow.py",
    "src/arggnn/benchmarking/european_native_models.py",
    "src/arggnn/benchmarking/european_argmix_release.py",
    "src/arggnn/benchmarking/european_native_relate.py",
    "src/arggnn/benchmarking/european_treeloom.py",
    "src/arggnn/benchmarking/european_training.py",
    "research/reviews/frozen_benchmark_training_ready_20261017_v1/european_frozen_treeloom_v1.py",
)

def need(ok, message):
    if not ok:
        raise ValueError(message)

def _root(repo_root=None):
    return Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[4]

def _path(repo, value):
    p = Path(value)
    p = p if p.is_absolute() else repo / p
    need(p.resolve().is_relative_to(repo), "Archive path escaped repository")
    q = p
    while q != repo:
        need(not q.is_symlink(), "Archive path contains symlink")
        q = q.parent
    return p

def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 2**20), b""):
            h.update(block)
    return h.hexdigest()

def _ref(repo, path):
    path = _path(repo, path)
    need(path.is_file() and not path.is_symlink(), "Regular file required")
    return dict(path=str(path.relative_to(repo)), sha256=_sha(path))

def _read(repo, reference, cap=512 * 2**20):
    path = _path(repo, reference["path"])
    need(path.is_file() and not path.is_symlink() and path.stat().st_size <= cap,
         "Bounded regular metadata required")
    need(_ref(repo, path) == reference, "Archive evidence changed")
    with path.open() as handle:
        return json.load(handle)

def _identity(path):
    x = path.lstat()
    need(stat.S_ISREG(x.st_mode) and x.st_nlink == 1, "Single-link regular candidate required")
    return (x.st_dev, x.st_ino, x.st_size, x.st_blocks, x.st_mtime_ns, x.st_ctime_ns, x.st_mode)

def _unchanged_identity(before, after):
    # Delayed NFS allocation may update st_blocks after a completed write.
    # Identity, logical size, nanosecond mtime/ctime and mode must stay exact.
    return before[:3] == after[:3] and before[4:] == after[4:]

def validate(spec, repo_root=None):
    repo = _root(repo_root)
    need(spec.get("schema") == "closed_european_lossless_archive_v1", "Explicit archive schema required")
    need(spec.get("root_current_scope_writer_revocation_required") is True,
         "ROOT must revoke future writes to the completed producer")
    owner = _path(repo, spec["owner_scope"])
    need("benchmark_campaign_20261018_workflow" in owner.parts and owner.is_dir(),
         "Only this campaign's exact successful workflow scope")
    inference_path = owner / "native/inference.json"
    need(_path(repo, spec["inference_ref"]["path"]) == inference_path, "Inference must belong to owner")
    inf = _read(repo, spec["inference_ref"])
    need(inf.get("schema") == "AP_full200mb_native_inference_v1"
         and inf.get("status") == "post_five_iterations_inferred"
         and inf.get("sequence_length_bp") == 200_000_000
         and inf.get("truth_or_oracle_read") is False
         and inf.get("outer_command_returncodes") == [0, 0, 0, 0, 0],
         "Exact genuine full native inference required")
    wait = _read(repo, spec["actual_wait_ref"], 2**20)
    need(_path(repo, spec["actual_wait_ref"]["path"]) == owner / "actual_wait.json"
         and wait.get("actual_direct_wait") is True and wait.get("actual_direct_returncode") == 0
         and wait.get("owned_process_group_absent") is True and wait.get("supervisor_error") is None,
         "Successful original direct wait and actual group drain required")
    job = _read(repo, wait["job_ref"], 32 * 2**20)
    need(job["task"]["id"] == spec["owner_task_id"]
         and _path(repo, job["task"]["output_directory"]) == owner,
         "Original wait must authenticate exact task and output scope")
    need(spec.get("max_output_bytes") == MAX_OUTPUT, "Exact total archival output cap required")
    need(type(spec["max_temporary_bytes"]) is int and 2**20 <= spec["max_temporary_bytes"] <= MAX_TEMP,
         "Explicit at-most-8GiB temporary allowance required")
    protected = [_path(repo, inf[key]["path"]) for key in ("inferred_tree", "observed_sample_manifest")]
    need(protected == [owner / "native/post_size.observed.trees",
                       owner / "native/observed_sample_manifest.json"], "Preserve exact future consumer files")
    for name in REQUIRED:
        candidate = _path(repo, owner / "native" / name)
        _identity(candidate)
    need(tuple(reference["path"] for reference in spec["consumer_source_refs"]) == CONSUMER_SOURCES,
         "Complete exact future-consumer source closure required")
    for reference in spec["consumer_source_refs"]:
        _read_source(repo, reference)
    return inf, owner

def _read_source(repo, reference):
    path = _path(repo, reference["path"])
    need(path.is_file() and path.stat().st_size <= 8 * 2**20 and _ref(repo, path) == reference,
         "Maintained future consumer source changed")

def factory(*, inference_ref, owner_scope, owner_task_id, output_directory,
            max_temporary_bytes=MAX_TEMP):
    repo = _root()
    owner = _path(repo, owner_scope)
    spec = dict(schema="closed_european_lossless_archive_v1",
                inference_ref=inference_ref, owner_scope=str(owner),
                owner_task_id=owner_task_id,
                actual_wait_ref=_ref(repo, owner / "actual_wait.json"),
                output_directory=output_directory, max_temporary_bytes=max_temporary_bytes,
                max_output_bytes=MAX_OUTPUT,
                consumer_source_refs=[_ref(repo, value) for value in CONSUMER_SOURCES],
                root_current_scope_writer_revocation_required=True)
    validate(spec, repo)
    return dict(module=MODULE, function="run", python=".runtime/env/bin/python", spec=spec,
                source_refs=[_ref(repo, value) for value in NEW_SOURCES + CONSUMER_SOURCES])

def _save(path, value):
    payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    need(len(payload) <= METADATA_CAP, "Archive metadata exceeded allowance")
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)

def _fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

class _BoundedWriter:
    def __init__(self, handle, limit):
        self.handle, self.limit = handle, limit
        self.bytes = 0
        self.hash = hashlib.sha256()
    def write(self, value):
        need(self.bytes + len(value) + 8192 <= self.limit, "Compressed temporary cap reached")
        written = self.handle.write(value)
        need(written == len(value), "Short archive write")
        self.hash.update(value)
        self.bytes += written
        return written
    def flush(self):
        return self.handle.flush()

def archive_one(source, output, max_temporary_bytes, check, before_unlink,
                expected_original_sha256=None):
    """Stream one file; an incompressible file stays intact and earns no credit."""
    source, output = Path(source), Path(output)
    before = _identity(source)
    archive = output.with_suffix(".gz")
    temporary = archive.with_name(archive.name + ".pending")
    need(not archive.exists() and not temporary.exists(), "Archive output already exists")
    original_hash, original_bytes = hashlib.sha256(), 0
    try:
        with temporary.open("xb") as target:
            writer = _BoundedWriter(target, max_temporary_bytes)
            with gzip.GzipFile(filename="", mode="wb", compresslevel=6, fileobj=writer, mtime=0) as compressed:
                with source.open("rb") as original:
                    for block in iter(lambda: original.read(8 * 2**20), b""):
                        check("lossless_archive_stream")
                        original_hash.update(block)
                        original_bytes += len(block)
                        compressed.write(block)
            target.flush()
            os.fsync(target.fileno())
            archive_hash, archive_bytes = writer.hash.hexdigest(), writer.bytes
        need(_unchanged_identity(before, _identity(source)) and original_bytes == before[2], "Original changed during compression")
        need(expected_original_sha256 is None or original_hash.hexdigest() == expected_original_sha256,
             "Original content differs from frozen native provenance")
        verified_hash, verified_bytes = hashlib.sha256(), 0
        with gzip.open(temporary, "rb") as restored:
            for block in iter(lambda: restored.read(8 * 2**20), b""):
                check("lossless_archive_verify")
                verified_hash.update(block)
                verified_bytes += len(block)
        need(verified_bytes == original_bytes and verified_hash.digest() == original_hash.digest(),
             "Archive decompression verification failed")
        temporary_stat = temporary.stat()
        original_charge = max(before[2], before[3] * 512) + 4096
        compressed_charge = max(temporary_stat.st_size, temporary_stat.st_blocks * 512) + 4096
        entry = dict(original_path=str(source), original_sha256=original_hash.hexdigest(),
                     original_bytes=original_bytes, original_mode=stat.S_IMODE(before[6]),
                     original_mtime_ns=before[4], archive_path=str(archive),
                     archive_sha256=archive_hash, archive_bytes=archive_bytes,
                     original_conservative_charge_bytes=original_charge,
                     archive_conservative_charge_bytes=compressed_charge,
                     decompressed_hash_and_size_verified=True)
        if compressed_charge >= original_charge:
            temporary.unlink()
            return dict(entry, state="original_kept_no_storage_gain", credited_bytes=0)
        os.rename(temporary, archive)
        _fsync_directory(archive.parent)
        entry["state"] = "verified_archive_original_preserved"
        # Durable restoration proof must precede removal of the redundant raw path.
        _save(output, entry)
        before_unlink()
        check("before_lossless_raw_replacement")
        need(_unchanged_identity(before, _identity(source)), "Original identity changed before replacement")
        need(_sha(archive) == archive_hash, "Archive changed before replacement")
        source.unlink()
        _fsync_directory(source.parent)
        entry.update(state="archive_committed", credited_bytes=original_charge-compressed_charge)
        _save(output, entry)
        return entry
    except BaseException:
        # The raw source survives every pre-commit error. Partial compressed data
        # are this task's new temporary artifacts and earn no accounting credit.
        raise

def run(spec, *, resource_check, execution_check=None, bounded_runner=None):
    repo = _root()
    inf, owner = validate(spec, repo)
    check_execution = execution_check or (lambda: None)
    out = _path(repo, spec["output_directory"])
    out.mkdir(parents=True, exist_ok=False)
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Archival worker must remain CPU-only")
    lock_directory = _path(repo, ".runtime/locks")
    lock_directory.mkdir(parents=True, exist_ok=True)
    lock_key = hashlib.sha256(str(owner).encode()).hexdigest()
    lock_path = _path(repo, lock_directory / ("european_retention_v9_" + lock_key + ".lock"))
    entries = []
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(lock_fd, "r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for index, name in enumerate(CANDIDATES):
            source = _path(repo, owner / "native" / name)
            if not source.exists():
                need(name not in REQUIRED, "Required native intermediate missing")
                continue
            _identity(source)
            need(name not in PROTECTED, "Protected future consumer file")
            check_execution()
            def before_unlink():
                check_execution()
                # Every original wait and source ref remains pinned. Required
                # candidates may already be archived by this same current task.
                wait = _read(repo, spec["actual_wait_ref"], 2**20)
                need(wait.get("actual_direct_returncode") == 0
                     and wait.get("owned_process_group_absent") is True,
                     "Original completion evidence changed")
                need(_ref(repo, owner / "native/inference.json") == spec["inference_ref"],
                     "Inference metadata changed")
            current_bytes = sum(max((Path(base) / child).lstat().st_size,
                                    (Path(base) / child).lstat().st_blocks * 512) + 4096
                                for base, dirs, files in os.walk(out, followlinks=False)
                                for child in dirs + files)
            remaining = spec["max_output_bytes"] - current_bytes - METADATA_CAP
            need(remaining >= 2**20, "No admitted space remains for another archive")
            expected = inf.get("pre_size_native_files_unchanged", {}).get(name, {}).get("sha256")
            entry = archive_one(source, out / ("file_%02d.json" % index),
                                min(spec["max_temporary_bytes"], remaining),
                                resource_check, before_unlink, expected)
            entries.append(entry)
        for name in PROTECTED:
            need((owner / "native" / name).is_file(), "Future consumer payload disappeared")
        manifest = dict(schema=spec["schema"], inference_ref=spec["inference_ref"],
                        actual_wait_ref=spec["actual_wait_ref"],
                        owner_scope=str(owner), owner_task_id=spec["owner_task_id"],
                        archive_output_directory=str(out), entries=entries,
                        archives_inside_current_task_output=True,
                        compressed_content_preserves_every_original_byte=True,
                        final_observed_tree_and_inputs_unchanged=True,
                        raw_path_replacement_only_after_verified_durable_archive=True,
                        restoration="For each archive_committed entry: admit original_bytes temporary space; gzip-decompress to a fresh same-directory file; verify original_sha256 and original_bytes; set original_mode and original_mtime_ns; atomically rename to original_path. Preserve archive until restoration is verified.",
                        observed_saved_bytes=sum(x["credited_bytes"] for x in entries))
        _save(out / "restore_manifest.json", manifest)
    return _ref(repo, out / "restore_manifest.json")

def add_archive_tasks(plan, output_root):
    """Pure task insertion. Existing inference science and budgets stay exact."""
    changed = copy.deepcopy(plan)
    ids = {task["id"] for task in changed["tasks"]}
    additions, skipped_external = [], []
    for task in plan["tasks"]:
        identifier = task["id"]
        if identifier.startswith("C4__") and identifier.endswith("__validation_infer"):
            skipped_external.append(identifier)
            continue
        if not (identifier.startswith("AP__") and identifier.endswith("__ap_infer")):
            continue
        if task.get("resource") == "external" or not task.get("output_directory"):
            skipped_external.append(identifier)
            continue
        archive_id = identifier + "__lossless_archive_v9"
        need(archive_id not in ids, "Archive task already exists")
        additions.append(dict(
            id=archive_id, dependencies=[identifier], resource="cpu",
            max_rss_bytes=16 * 2**30, max_output_bytes=MAX_OUTPUT,
            max_seconds=86400,
            output_directory=str(Path(output_root) / "retention" / archive_id),
            factory_module=MODULE, factory_function="factory",
            factory_arguments=dict(inference_ref={"$task": identifier, "read": False},
                                   owner_scope=task["output_directory"], owner_task_id=identifier,
                                   max_temporary_bytes=MAX_TEMP),
            remeasure_succeeded_tasks=[identifier],
            root_current_scope_writer_revocation_required=True,
            temporary_write_scope="current_retirement_task_output",
            temporary_write_limit_bytes=MAX_TEMP,
        ))
    changed["tasks"] = additions + changed["tasks"]
    return changed, dict(archive_tasks_added=[task["id"] for task in additions],
                         skipped_external_owner_task_ids=skipped_external,
                         prospective_savings_credited_bytes=0,
                         max_temporary_bytes_per_worker=MAX_TEMP,
                         max_total_output_bytes_per_worker=MAX_OUTPUT,
                         final_observed_trees_and_original_observed_inputs_preserved=True)
