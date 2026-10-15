"""Source/stdlib/in-memory mocks only. Never import driver or any guard.

Reads only exact source and compact metadata refs, not observed VCF/QC. No
admission, process observation, Popen, scientific worker or native invocation.
"""
from pathlib import Path
from types import SimpleNamespace
import ast
import hashlib
import io
import json
import os
import stat
import sys

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
HERE = ROOT / "research/reviews/H018_observed_bridge_root_source_v1"
results = []


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    results.append(dict(name=name, passed=True))


def expect_error(name, operation, kind=ValueError):
    try:
        operation()
    except kind:
        check(name, True)
        return
    raise AssertionError(name + " did not reject")


def compact(record):
    p = ROOT / record["path"]
    assert not (record["path"].endswith(".vcf.gz") or record["path"].endswith("regional_site_QC.tsv"))
    raw = p.read_bytes()
    assert len(raw) <= 4_000_000 and hashlib.sha256(raw).hexdigest() == record["sha256"]
    return json.loads(raw)


text = (HERE / "root_bridge_driver.py").read_text()
tree = ast.parse(text)
compile(tree, str(HERE / "root_bridge_driver.py"), "exec")
check("driver_compiles_without_import_or_execution", True)
functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
constants = [n for n in tree.body if isinstance(n, ast.Assign)]
assert all(not isinstance(c, ast.Call) or isinstance(c.func, ast.Name) and c.func.id in ("Path", "dict")
           for n in constants for c in ast.walk(n))
selected = [functions[x] for x in ("require", "identity", "error_record", "native_gate", "verify_request",
                                  "bounded_file", "group_absent", "direct_wait_owned", "supervise")]
ns = dict(Path=Path, hashlib=hashlib, json=json, stat=stat, sys=sys, utc=lambda: "synthetic-clock")
exec(compile(ast.Module(body=constants+selected, type_ignores=[]), "<AST-selected-in-memory-checks>", "exec"), ns)
check("combined_reservation_forecast_has_15MB_plus_slack", ns["OUTPUT_CAP"] + ns["AUX_WRITE"] + 1_000_000 < ns["CAP"])
check("native_gate_always_requires_separate_admission", ns["native_gate"]()["native_execution_authorized"] is False
      and ns["native_gate"]()["separate_native_policy_and_admission_required"] is True)
worker = ast.get_source_segment(text, functions["worker"])
check("worker_has_no_subprocess_or_native_calls", not any(x in worker for x in
      ("subprocess", "Popen", "os.system", "Relate(", "RelateFileFormats(", "decode_fixed_HS_pair(")))
check("observed_reads_follow_original_guard", worker.index("m.budget_check_v9(storage, first=True") < worker.index('refs["source_VCF"]'))
check("final_output_mkdir_follows_32MB_guard", worker.index("output.mkdir()") > worker.rindex("required_write_bytes=OUTPUT_CAP"))
check("completion_follows_all_payload_writes", worker.index('write_bytes(output, "completed.json"') > worker.index("for name, raw in payloads.items():"))
check("failed_count_unknown_until_full_validation", "validated_counts = None" in worker and "if validated_counts is not None:" in worker)
launch = ast.get_source_segment(text, functions["launch"])
check("all_paths_and_argv_resolved_before_Popen", launch.index("All controller paths resolved before Popen") < launch.index("subprocess.Popen")
      and launch.index("argv =") < launch.index("subprocess.Popen") and launch.index("env =") < launch.index("subprocess.Popen"))
check("Popen_is_inside_wait_finally", any(isinstance(n, ast.Try) and n.finalbody
      and any(isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute) and x.func.attr == "Popen" for a in n.body for x in ast.walk(a))
      and any(isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id == "supervise" for a in n.finalbody for x in ast.walk(a))
      for n in ast.walk(functions["launch"])))
check("large_artifact_validation_streams_instead_of_common_ref", 'bounded_file(b, rec, OUTPUT_CAP, retain=False)' in ast.get_source_segment(text, functions["verify_output"]))
check("no_credit_release_delete_implementation", not any(x in text for x in
      ("apply_close_completed_v1(", "unlink(", "rmtree(", ".remove(", "capacity_v2.configure", "memoize(")))
check("one_Popen_call_for_preparation_worker", sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "Popen" for n in ast.walk(tree)) == 1)
check("direct_OS_wait_no_synthesized_subprocess_returncode", "os.waitpid(proc.pid, os.WNOHANG)" in text
      and not any(x in text for x in ("proc.poll(", "proc.wait(", "proc.communicate(")))

manifest = json.loads((HERE / "source_manifest.json").read_bytes())
request = json.loads((HERE / "root_request.inactive.json").read_bytes())
config = json.loads((HERE / "config.json").read_bytes())
all_refs = list(manifest["source_refs"].values())+list(manifest["metadata_refs"].values())
for rec in all_refs:
    p = ROOT / rec["path"]
    raw = p.read_bytes()
    assert p.resolve() == p and len(raw) <= 4_000_000 and hashlib.sha256(raw).hexdigest() == rec["sha256"]
check("all_source_and_compact_metadata_pins_match", True)
inputs = compact(ns["INPUT_REFS"])
check("observed_payload_refs_only_transitive", not any(x["path"] in
      (inputs["refs"]["source_VCF"]["path"], inputs["refs"]["source_QC"]["path"]) for x in all_refs))
check("unchanged_bridge_core_pin", manifest["source_refs"]["pure_bridge"] == ns["BRIDGE"])
check("metadata_original_exit_unknown_and_full_charge_preserved", inputs["root_metadata_original_OS_exit"] == "unknown"
      and inputs["root_metadata_original_wait0_available"] is False
      and inputs["root_metadata_original_reservation_bytes_retained"] == 5_000_000)

# Exercise request semantics using only an in-memory fake reference reader.
fake_registration_ref = dict(path="synthetic/registration.json", sha256="0"*64)
active = dict(request, status="active", root_authorizer="root_agent", registered_before_execution=True,
              registration_ref=fake_registration_ref)
request_ref = dict(path="synthetic/request.json", sha256="1"*64)
objects = {ns["INPUT_REFS"]["path"]: inputs, ns["CONFIG"]: config, ns["MANIFEST"]: manifest,
           fake_registration_ref["path"]: dict(decisions=[dict(id="H018", job_id=ns["JOB"])])}
objects.update({k["path"]: compact(k) for k in manifest["metadata_refs"].values()
                if k["path"].endswith(".json")})
refs_by_path = {r["path"]: r for r in all_refs}
refs_by_path[ns["MANIFEST"]] = request["source_manifest_ref"]
fake_b = SimpleNamespace(read=lambda r: objects[r["path"]], ref=lambda p: refs_by_path[str(p)],
    GUARD=config["guard_ref"], POLICY=config["policy_ref"], REPAIR=config["repair_authority_ref"])
objects[request_ref["path"]] = request
real_bounded = ns["bounded_file"]
ns["bounded_file"] = lambda *a, **k: None
expect_error("inactive_request_rejected_before_any_guard", lambda: ns["verify_request"](fake_b, request_ref))
objects[request_ref["path"]] = active
ns["verify_request"](fake_b, request_ref)
check("synthetic_active_request_and_exact_config_closure_accepted", True)
objects[request_ref["path"]] = dict(active, native_execution_authorized=True)
expect_error("synthetic_native_authorization_rejected", lambda: ns["verify_request"](fake_b, request_ref))
objects[request_ref["path"]] = active
ns["bounded_file"] = real_bounded

# No OS file/process operations occur below. Every os/resource/selector/time is fake.
class FakeClock:
    def __init__(self, step=0.1):
        self.value, self.step = 0, step
    def monotonic(self):
        self.value += self.step
        return self.value
    def sleep(self, amount):
        self.value += amount


class FakePipe:
    def fileno(self): return 7
    def close(self): pass


class FakeOS:
    WNOHANG = 1
    WIFEXITED = staticmethod(os.WIFEXITED)
    WIFSIGNALED = staticmethod(os.WIFSIGNALED)
    waitstatus_to_exitcode = staticmethod(os.waitstatus_to_exitcode)
    def __init__(self, chunks, waits):
        self.chunks, self.waits, self.signals = list(chunks), list(waits), []
    def set_blocking(self, fd, value): pass
    def read(self, fd, amount): return self.chunks.pop(0) if self.chunks else b""
    def waitpid(self, pid, flags):
        value = self.waits.pop(0)
        if isinstance(value, BaseException): raise value
        return (pid, value) if value is not None else (0, 0)
    def killpg(self, pid, sig):
        if sig == 0: raise ProcessLookupError()
        self.signals.append(sig)
    def getpid(self): return 12345


class FakeSelector:
    def register(self, *args): pass
    def select(self, **kwargs): return [(None, 1)]
    def close(self): pass


def simulate(name, chunks, waits, setup_fail=False, clock_step=0.1, high_rss=False, broken_print=False):
    fake_os = FakeOS(chunks, waits)
    clock = FakeClock(clock_step)
    def make_selector():
        if setup_fail: raise RuntimeError("synthetic selector setup failure")
        return FakeSelector()
    lines = []
    def fake_print(value, **kwargs):
        if broken_print: raise BrokenPipeError("synthetic outer logging failure")
        lines.append(json.loads(value))
    ns.update(os=fake_os, time=clock, selectors=SimpleNamespace(DefaultSelector=make_selector, EVENT_READ=1),
        resource=SimpleNamespace(RUSAGE_SELF=0, getrusage=lambda x: SimpleNamespace(ru_maxrss=10)),
        signal=SimpleNamespace(SIGTERM=15, SIGKILL=9), rss=lambda pid: 3_000_000_000 if high_rss else 10,
        print=fake_print)
    proc = SimpleNamespace(pid=98765, stdout=FakePipe(), returncode=None)
    observation, log = ns["supervise"](proc)
    check(name, True)
    return observation, log, fake_os.signals, lines

obs, log, signals, lines = simulate("mock_normal_wait0_and_pipe_drain", [b"hello", b""], [0])
check("mock_wait_records_raw_actual_OS_status", obs["actual_direct_wait_observed"] and obs["actual_exit_code"] == 0
      and obs["actual_OS_wait_status"] == 0 and obs["process_group_drained"] and log == b"hello" and not signals)
obs, log, signals, _ = simulate("mock_nonzero_exit_retained", [b""], [7 << 8])
check("mock_nonzero_never_becomes_wait0", obs["actual_exit_code"] == 7)
obs, log, signals, _ = simulate("mock_monitor_setup_failure_still_waits", [], [15], setup_fail=True)
check("mock_setup_failure_owned_TERM_and_actual_wait", signals == [15] and obs["actual_exit_code"] == -15 and obs["control_error"])
obs, log, signals, _ = simulate("mock_RSS_cap_still_waits", [], [15], high_rss=True)
check("mock_RSS_violation_preserved", obs["control_error"] and obs["actual_direct_wait_observed"])
obs, log, signals, _ = simulate("mock_log_overflow_still_waits", [b"x"*65536]*32, [15])
check("mock_log_capture_never_exceeds_2MB", len(log) == ns["LOG_CAP"] and obs["log_total_bytes"] > ns["LOG_CAP"] and obs["actual_exit_code"] == -15)
obs, log, signals, _ = simulate("mock_deadline_TERM_then_KILL_and_wait", [], [None, None, 9], setup_fail=True, clock_step=20)
check("mock_timeout_signals_owned_group_before_reaping", signals == [15, 9] and obs["actual_exit_code"] == -9)
obs, log, signals, _ = simulate("mock_ECHILD_preserves_unknown_exit", [b""], [ChildProcessError("synthetic external reaping")])
check("mock_ECHILD_never_claims_wait0_or_drain", obs["actual_exit_code"] is None and not obs["actual_direct_wait_observed"] and not obs["process_group_drained"])
obs, log, signals, _ = simulate("mock_outer_stdout_failure_keeps_known_wait", [b""], [0], broken_print=True)
check("mock_known_OS_exit_survives_print_failure", obs["actual_exit_code"] == 0 and obs["stdout_receipt_error"])

# Bounded file/hash tests use in-memory fake descriptors and stat objects only.
class FakePath:
    def __init__(self, info): self.info = info
    def lstat(self): return self.info


def file_case(raw, cap, expected_sha=None, mutate=False, retain=True):
    info = SimpleNamespace(st_dev=1, st_ino=2, st_size=len(raw), st_mtime_ns=3, st_ctime_ns=4, st_mode=stat.S_IFREG)
    after = SimpleNamespace(**vars(info))
    if mutate: after.st_mtime_ns += 1
    remaining = [raw]
    def read(fd, size):
        data = remaining[0][:size]
        remaining[0] = remaining[0][size:]
        return data
    stat_calls = [info, after]
    fake_os = SimpleNamespace(O_RDONLY=0, O_NOFOLLOW=1, open=lambda *a: 99,
        close=lambda fd: None, fstat=lambda fd: stat_calls.pop(0), read=read)
    ns["os"] = fake_os
    b = SimpleNamespace(local=lambda p: FakePath(after))
    rec = dict(path="synthetic/artifact", sha256=expected_sha or hashlib.sha256(raw).hexdigest())
    return ns["bounded_file"](b, rec, cap, retain=retain)

check("mock_large_artifact_streaming_above_common_4MB_cap", file_case(b"z"*4_100_000, 5_000_000, retain=False)["bytes"] == 4_100_000)
expect_error("mock_artifact_cap_rejects", lambda: file_case(b"too large", 2))
expect_error("mock_artifact_SHA_rejects", lambda: file_case(b"abc", 10, expected_sha="f"*64))
expect_error("mock_artifact_stability_rejects", lambda: file_case(b"abc", 10, mutate=True))
check("mock_retained_bytes_exact", file_case(b"abc", 10) == b"abc")

report = dict(schema="H018_observed_bridge_root_source_checks_v1", status="PASS_SOURCE_AND_IN_MEMORY_MOCKS_ONLY",
    driver_ref=dict(path=str((HERE/"root_bridge_driver.py").relative_to(ROOT)),sha256=hashlib.sha256(text.encode()).hexdigest()),
    checks=results, count=len(results), source_only=True, driver_imported=False, guard_imported_or_invoked=False,
    admission_or_accounting_performed=False, observed_VCF_QC_or_genotypes_read=False,
    actual_process_checks_or_Popen_performed=False, native_or_scientific_runtime_invoked=False,
    real_data_result_or_counts_claimed=False)
with (HERE/"source_checks.json").open("x") as stream:
    json.dump(report, stream, sort_keys=True, indent=2); stream.write("\n")
print(json.dumps(dict(status=report["status"], checks=len(results))))
