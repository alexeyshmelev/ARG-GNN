"""Bounded admitted CPU driver for an independently qualified official FLARE JAR."""
from pathlib import Path
import hashlib
import os
import re
import json
import resource
import signal
import subprocess
import time

from .american_c3_common_v1 import require, read, bound, cli
from .american_c3_flare_common_v1 import (
    PIN_REPOSITORY, PIN_TAG, PIN_COMMIT, EXPECTED_VERSION, NATIVE_OPTIONS, PROJECTION,
    stage_completion, native_mac_threshold, native_file_hash, gzip_bound,
)
SOURCE = "src/arggnn/american_c3_flare_run_v2.py"
DEFAULTS = dict(
    schema="american_c3_flare_native_run_v1", execution_enabled=False,
    conversion_ref=None, jar_ref=None, binary_provenance_ref=None, external_truth_ref=None,
    java_executable=None, java_executable_sha256=None, java_version_output_sha256=None,
    native_options=NATIVE_OPTIONS, projection=PROJECTION, source_commit=PIN_COMMIT,
    heap_gib=16, timeout_seconds=14400, max_native_vcf_bytes=40000000000,
    max_java_tmp_bytes=1073741824, output_cap_bytes=42000000000,
)


def _qualified_native_jar(cfg, storage):
    # Immutable storage guards admit metadata/source suffixes, not a binary JAR.
    # The admitted JSON manifest pins its JAR, which is separately byte-verified.
    manifest_ref = cfg["binary_provenance_ref"]
    require(isinstance(manifest_ref, dict) and set(manifest_ref) == {"path", "sha256"}
            and Path(manifest_ref["path"]).suffix == ".json"
            and manifest_ref in storage["reservation"]["consumer_bindings"],
            "Exact JSON binary provenance manifest must be bound by root admission")
    binary = read(manifest_ref)
    require(binary.get("schema") == "root_qualified_FLARE_binary_v1"
            and binary.get("root_authorizer") == "root_agent"
            and binary.get("repository") == PIN_REPOSITORY and binary.get("tag") == PIN_TAG
            and binary.get("source_commit") == PIN_COMMIT
            and binary.get("native_sources_modified") is False
            and binary.get("source_commit_build_verified") is True
            and binary.get("reported_version") == EXPECTED_VERSION
            and binary.get("jar_ref") == cfg["jar_ref"],
            "Root-qualified unmodified pinned-source FLARE binary required; no assumed unversioned JAR")
    jar = bound(cfg["jar_ref"])
    require(jar.suffix == ".jar", "Actual project-local byte-verified native JAR required")
    return jar


def execute(cfg, w):
    require(cfg["schema"] == DEFAULTS["schema"] and cfg["external_truth_ref"] is None
            and cfg["native_options"] == NATIVE_OPTIONS and cfg["projection"] == PROJECTION
            and cfg["source_commit"] == PIN_COMMIT and cfg["heap_gib"] == 16
            and type(cfg["timeout_seconds"]) is int and cfg["timeout_seconds"] > 0,
            "Frozen native CPU reconstruction protocol required")
    converted = stage_completion(cfg["conversion_ref"], "completed_C3_FLARE_observation_conversion")
    require(converted["native_source_commit"] == PIN_COMMIT
            and converted["native_options"] == NATIVE_OPTIONS
            and converted["projection"] == PROJECTION
            and converted["common_focal_registry_unchanged"] is True
            and converted["query_ancestry_truth_opened"] is False,
            "Exact pre-truth conversion/marker/focal contract required")
    native_mac_threshold(converted["reference_haplotypes"])
    jar = _qualified_native_jar(cfg, w.storage)
    java = Path(cfg["java_executable"])
    require(java.is_absolute() and java.is_file() and os.access(java, os.X_OK)
            and native_file_hash(java.resolve()) == cfg["java_executable_sha256"],
            "Actual root-qualified Java executable/hash required")
    require(all(not os.environ.get(name) for name in
                ("JAVA_TOOL_OPTIONS", "_JAVA_OPTIONS", "JDK_JAVA_OPTIONS")),
            "Unrecorded JVM option injection rejected")
    # Resource/placement flags only. Native FLARE/application options remain unchanged.
    qualified_tmp = w.prepare("java_tmp/.unwritten_directory_marker", 0).parent
    qualified_home = w.prepare("java_home/.unwritten_directory_marker", 0).parent
    jvm_operational_flags = [
        "-XX:-UsePerfData", "-XX:ActiveProcessorCount=4",
        "-Djava.io.tmpdir="+str(qualified_tmp), "-Duser.home="+str(qualified_home),
        "-XX:ErrorFile="+str(qualified_home/"hs_err_pid%p.log"),
    ]
    version_probe_argv = [str(java), *jvm_operational_flags, "-version"]
    def no_core_dump():
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    version_process = subprocess.run(version_probe_argv, check=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=20, cwd=w.path, preexec_fn=no_core_dump,
        start_new_session=True)
    version = version_process.stdout
    require(len(version) <= 65536, "Bounded actual Java version output")
    major = re.search(rb'(?:version\s+"?|openjdk\s+)([0-9]+)', version)
    require(major and int(major.group(1)) >= 11
            and hashlib.sha256(version).hexdigest() == cfg["java_version_output_sha256"],
            "Exact qualified Java11-or-later runtime required")
    path = w.prepare("java_version.txt", len(version))
    with path.open("xb") as stream: stream.write(version)
    java_version_ref = w.finish(path, len(version))

    markers = converted["native_expected_markers"]
    samples = converted["query_haplotypes"] // 2
    require(markers > 0 and samples > 0, "Complete nonempty native observation dimensions")
    vcf_plan = gzip_bound(4194304 + markers*(256+64*samples))
    require(vcf_plan <= cfg["max_native_vcf_bytes"], "Native posterior output count bound exceeds cap")
    caps = {
        ".anc.vcf.gz": vcf_plan, ".model": 1048576,
        ".global.anc.gz": 1048576, ".log": 16777216,
    }
    stdout_cap = 16777216
    planned = sum(caps.values()) + stdout_cap + cfg["max_java_tmp_bytes"] + 16777216
    require(planned <= cfg["output_cap_bytes"], "Total native output/tmp bound exceeds admitted cap")
    expected = {suffix: w.prepare("native/flare"+suffix, cap) for suffix, cap in caps.items()}
    prefix = w.path/"native/flare"
    tmp = qualified_tmp
    stdout_path = w.prepare("native_stdout.txt", stdout_cap)
    argv = [
        str(java), "-Xmx16g", "-XX:ActiveProcessorCount=4",
        "-Duser.language=en", "-Duser.country=US", "-Djava.io.tmpdir="+str(tmp),
        "-jar", str(jar),
        "ref="+str(bound(converted["ref_vcf_ref"])),
        "ref-panel="+str(bound(converted["ref_panel_ref"])),
        "gt="+str(bound(converted["gt_vcf_ref"])),
        "map="+str(bound(converted["map_ref"])), "out="+str(prefix),
    ]
    argv += [key+"="+(str(value).lower() if isinstance(value, bool) else str(value))
             for key, value in NATIVE_OPTIONS.items()]
    argv[1:1] = jvm_operational_flags
    argv_ref = w.json("native_argv.json", dict(
        argv=argv, conversion_ref=cfg["conversion_ref"], jar_ref=cfg["jar_ref"],
        binary_provenance_ref=cfg["binary_provenance_ref"], java_version_ref=java_version_ref,
        source_commit=PIN_COMMIT, expected_version=EXPECTED_VERSION,
        native_options=NATIVE_OPTIONS, no_model_or_gt_ancestries_supplied=True,
        JVM_operational_flags=jvm_operational_flags, version_probe_argv=version_probe_argv,
        version_probe_returncode=version_process.returncode,
        CPU_only=True, planned_output_and_tmp_bytes=planned, native_output_caps=caps,
    ))
    w.adapter.budget_check_v9(w.storage, boundary=True, required_write_bytes=planned)
    child = None
    start = time.monotonic()
    operation_error = None
    actual_rc = None
    group_absent = True
    try:
        with stdout_path.open("xb") as stream:
            try:
                child = subprocess.Popen(argv, cwd=w.path, stdout=stream, stderr=subprocess.STDOUT,
                    start_new_session=True, preexec_fn=no_core_dump)
                while child.poll() is None:
                    require(time.monotonic()-start <= cfg["timeout_seconds"],
                            "Native CPU time limit exceeded; preserve partial output")
                    require(stdout_path.stat().st_size <= stdout_cap, "Native stdout cap exceeded")
                    for suffix, path in expected.items():
                        require(not path.exists() or path.stat().st_size <= caps[suffix],
                                "Native file output cap exceeded: "+suffix)
                    temporary_bytes = 0
                    for item in tmp.rglob("*"):
                        try:
                            if item.is_file(): temporary_bytes += item.stat().st_size
                        except FileNotFoundError:
                            continue
                    require(temporary_bytes <= cfg["max_java_tmp_bytes"], "JVM temporary-file cap exceeded")
                    # No global storage mutex or stable-scope scan while native files grow.
                    # Entire declared growth was checked immediately before Popen.
                    time.sleep(1)
            except BaseException as error:
                operation_error = error
            finally:
                if child is not None:
                    def signal_owned_group(signum):
                        try: os.killpg(child.pid, signum)
                        except ProcessLookupError: pass
                    if operation_error is not None: signal_owned_group(signal.SIGTERM)
                    try:
                        actual_rc = child.wait(timeout=10 if operation_error is not None else 5)
                    except BaseException as error:
                        operation_error = operation_error or error
                        signal_owned_group(signal.SIGKILL)
                        actual_rc = child.wait()
                    group_absent = False
                    for _ in range(100):
                        try: os.killpg(child.pid, 0)
                        except ProcessLookupError:
                            group_absent = True
                            break
                        signal_owned_group(signal.SIGKILL)
                        time.sleep(.1)
    except BaseException as error:
        operation_error = operation_error or error
    wait = dict(
        schema="C3_native_FLARE_actual_wait_v1", returncode=actual_rc,
        actual_direct_native_wait=child is not None, owned_process_group_ESRCH=group_absent,
        operational_failure=None if operation_error is None else dict(
            type=type(operation_error).__name__, message=str(operation_error)[:4096]),
        native_argv_ref=argv_ref, seconds=time.monotonic()-start,
    )
    try: print(json.dumps(dict(native_actual_wait=wait), sort_keys=True), flush=True)
    except BaseException: pass
    require(group_absent, "Native group remains live; defer scope boundary/writes to root")
    # This original guarded writer is reached only after all owned writers are terminal.
    native_wait_ref = w.json("native_actual_wait.json", wait)
    require(operation_error is None and actual_rc == 0,
            "Official native FLARE failed; preserve partial outputs, no fallback/options change")
    require(set(p.name for p in (w.path/"native").iterdir())
            == {"flare"+suffix for suffix in expected}, "Unexpected or missing native products")
    refs = {suffix: w.finish(path, caps[suffix]) for suffix, path in expected.items()}
    require(EXPECTED_VERSION in bound(refs[".log"]).read_text(),
            "Reported native version differs from pinned official source")
    stdout_ref = w.finish(stdout_path, stdout_cap)
    return w.json("completed.json", dict(
        status="completed_C3_native_FLARE", conversion_ref=cfg["conversion_ref"],
        parent_id=converted["parent_id"], class_order=converted["class_order"],
        focal_registry_ref=converted["focal_registry_ref"], projection=PROJECTION,
        native_output_refs=refs, native_stdout_ref=stdout_ref, native_argv_ref=argv_ref,
        jar_ref=cfg["jar_ref"], binary_provenance_ref=cfg["binary_provenance_ref"],
        java_version_ref=java_version_ref, source_commit=PIN_COMMIT, native_version=EXPECTED_VERSION,
        native_options=NATIVE_OPTIONS, native_actual_wait_ref=native_wait_ref,
        JVM_operational_flags=jvm_operational_flags, seconds=time.monotonic()-start,
        query_ancestry_truth_opened=False, native_outputs_modified=False,
        author_Table1_exact_reproduction=False,
    ))


if __name__ == "__main__":
    cli(SOURCE, DEFAULTS, execute, dependencies=(
        "src/arggnn/american_c3_flare_common_v1.py",
        "src/arggnn/american_c3_flare_convert_v1.py",
    ))
