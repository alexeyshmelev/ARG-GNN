"""Explicit one-time continuation after verified MakeChunks and Paint.

This is a new readable driver. It never rewrites the pinned original driver,
recipe, prepared inputs, or completed chunk/painting files. ROOT must first
archive the failed native_run log/receipt leaf and submit the reviewed branch
repair. The retained native inventory must match exactly before any native work.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import json
from pathlib import Path
import re
import subprocess
import threading
import time

from arggnn.benchmarking.paper_argmix import paper_argmix_generator as g
from arggnn.benchmarking.paper_argmix.paper_argmix_features import ScopeGuard

SCHEMA = "paper_argmix_verified_paint_resume_v1"
ORIGINAL_DRIVER_SHA256 = "7b575d90a29b6fa2df630d16778f572a7e223d16cc997ff751ed23bca81262c3"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def stage_plan(commands, binary, chunk, sections):
    """Return the unchanged original post-Paint native argv, in stage order."""
    argv = commands["argv"][0]
    def option(flag):
        return argv[argv.index(flag) + 1]
    prefix = option("-o")
    blocks = []
    for mode in ("BuildTopology", "InferBranchLengths"):
        entries = []
        for section in range(sections):
            args = ["--mode", mode, "--chunk_index", str(chunk),
                    "--first_section", str(section), "--last_section", str(section)]
            if mode == "BuildTopology":
                args += ["--painting", "0.001,1", "-N", option("-N")]
            else:
                args += ["-m", option("-m"), "-N", option("-N")]
            args += ["--sample_ages", option("--sample_ages"), "--seed", option("--seed"), "-o", prefix]
            entries.append((f"chunk{chunk}_{mode}_section{section}", [binary] + args))
        blocks.append(entries)
        if mode == "BuildTopology":
            blocks.append([(f"chunk{chunk}_FindEquivalentBranches",
                            [binary, "--mode", "FindEquivalentBranches", "--chunk_index", str(chunk), "-o", prefix])])
    blocks.append([(f"chunk{chunk}_CombineSections",
                    [binary, "--mode", "CombineSections", "-N", option("-N"), "--chunk_index", str(chunk),
                     "--sample_ages", option("--sample_ages"), "-o", prefix])])
    return blocks


def verify_resume(spec):
    """Authenticate the full retained observed-native state without mutation."""
    expected_keys = {"schema", "config_ref", "commands_ref", "original_driver_ref",
                     "original_native_run_directory", "archived_native_run_directory",
                     "archived_native_waits_ref", "archived_outer_wait_ref",
                     "native_work_directory", "retained_inventory", "expected_chunks",
                     "expected_sections_by_chunk", "workers"}
    need(type(spec) is dict and set(spec) == expected_keys and spec["schema"] == SCHEMA,
         "Exact explicit paint-resume specification required")
    config_path = g.verify_ref(spec["config_ref"])
    config = g.metadata(config_path)
    commands_path = g.verify_ref(spec["commands_ref"])
    commands = g.metadata(commands_path)
    original = g.verify_ref(spec["original_driver_ref"])
    need(g.sha(original) == ORIGINAL_DRIVER_SHA256,
         "Unchanged pinned original stage driver required")
    work = g.confined(spec["native_work_directory"])
    original_output = g.confined(spec["original_native_run_directory"])
    archive = g.confined(spec["archived_native_run_directory"])
    need(work == commands_path.parent == g.confined(commands["cwd"])
         and original_output != archive and archive.is_dir()
         and commands["config_ref"] == g.file_ref(config_path),
         "Exact prepared native workspace and separately preserved failed attempt required")
    need(spec["workers"] == config["relate_execution"]["workers"] == 8,
         "Original eight-worker execution partition required")
    install = g.metadata(g.verify_ref(config["input_refs"]["relate_installation"]))
    binary = str(g.verify_ref(install["binaries"]["Relate"]))
    converter = str(g.verify_ref(install["binaries"]["RelateFileFormats"]))
    source_wrapper = g.verify_ref(install["scripts"]["RelateParallel"])
    for reference in commands["input_refs"].values():
        g.verify_ref(reference)
    argv = commands["argv"][0]
    def option(flag):
        return argv[argv.index(flag) + 1]
    need(len(commands["argv"]) == 2 and argv[0] == binary
         and commands["argv"][1][0] == converter and option("--mode") == "All"
         and option("-o") == "inferred", "Original complete native commands required")
    prefix = work / "inferred"
    need(prefix.is_dir() and not prefix.is_symlink(), "Retained ordinary native prefix required")
    inventory = spec["retained_inventory"]
    need(type(inventory) is list and inventory, "Complete sealed retained inventory required")
    relative_names = [item["relative_path"] for item in inventory]
    need(len(relative_names) == len(set(relative_names))
         and all(type(name) is str and name and not Path(name).is_absolute()
                 and ".." not in Path(name).parts for name in relative_names),
         "Unique contained native product paths required")
    actual_names = sorted(str(path.relative_to(prefix)) for path in prefix.rglob("*") if path.is_file())
    need(actual_names == sorted(relative_names),
         "Retained prefix changed or contains unproven topology outputs; preserve and review separately")
    for item in inventory:
        path = prefix / item["relative_path"]
        need(not path.is_symlink() and path.is_file() and path.stat().st_size == item["bytes"]
             and g.sha(path) == item["sha256"], "Retained native product changed: " + str(path))
    chunks = sorted(int(re.fullmatch(r"parameters_c(\d+)\.bin", path.name).group(1))
                    for path in prefix.glob("parameters_c*.bin"))
    need(chunks == spec["expected_chunks"] == [0], "This continuation binds the original single completed painted chunk")
    sections = {}
    for chunk in chunks:
        names = sorted(int(re.fullmatch(r"relate_(\d+)\.bin", path.name).group(1))
                       for path in (prefix / f"chunk_{chunk}" / "paint").glob("*.bin"))
        count = spec["expected_sections_by_chunk"][str(chunk)]
        need(names == list(range(count)) and count > 0, "Every original painting section must remain")
        sections[chunk] = count
    waits_path = g.verify_ref(spec["archived_native_waits_ref"])
    outer_path = g.verify_ref(spec["archived_outer_wait_ref"])
    need(waits_path.parent == outer_path.parent == archive,
         "Historical native and supervisor waits must remain in the archived leaf")
    outer = g.metadata(outer_path)
    need(outer["actual_wait"] is True and outer["owned_process_group_absent"] is True
         and outer["status"] == "failed" and outer["error"] == "Resource reservation exceeded",
         "Exact failed, drained resource-boundary attempt required")
    from arggnn.benchmarking.paper_campaign import processes
    need(not processes.session_members(outer["pid"]), "Failed native session must remain empty")
    prior = [json.loads(line) for line in waits_path.read_text().splitlines() if line]
    expected = [
        ("MakeChunks", [binary, "--mode", "MakeChunks", "--haps", option("--haps"),
                        "--sample", option("--sample"), "--map", option("--map"),
                        "--memory", option("--memory"), "-o", "inferred"]),
        ("chunk0_Paint", [binary, "--mode", "Paint", "--painting", "0.001,1", "-o", "inferred", "--chunk_index", "0"]),
    ]
    need(len(prior) == len(expected), "Exactly the two proven original native waits required")
    reused = []
    for record, (name, exact) in zip(prior, expected):
        need(record["stage"] == name and record["argv"] == exact and record["returncode"] == 0
             and record["actual_wait_consumed"] is True,
             "Only exact successful original MakeChunks/Paint stages may be reused")
        old_log = g.confined(record["log_ref"]["path"])
        need(old_log.parent == original_output and old_log.name == name + ".log",
             "Original stage log identity differs")
        log = archive / old_log.name
        need(log.is_file() and not log.is_symlink()
             and log.stat().st_size == record["log_ref"]["bytes"]
             and g.sha(log) == record["log_ref"]["sha256"], "Archived successful native log changed")
        copied = copy.deepcopy(record)
        copied["historical_original_log_ref"] = copied["log_ref"]
        copied["log_ref"] = g.file_ref(log)
        copied["reused_completed_wait"] = True
        copied["historical_waits_ref"] = g.file_ref(waits_path)
        reused.append(copied)
    return config_path, config, commands_path, commands, binary, source_wrapper, work, chunks, sections, reused


def run_resume(spec_path, output, *, guard):
    spec_path = g.confined(spec_path)
    spec = g.metadata(spec_path)
    (config_path, config, commands_path, commands, binary, source_wrapper,
     work, chunks, sections, reused) = verify_resume(spec)
    need(g.confined(output) == g.confined(spec["original_native_run_directory"]), "Preserve the fixed scientific output namespace")
    output = g._fresh_directory(output, guard)
    started = time.monotonic()
    lock = threading.Lock()
    records = list(reused)
    with (output / "actual_native_waits.jsonl").open("x") as stream:
        for record in reused:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    def run(name, exact):
        guard(0)
        print(json.dumps(dict(stage=name, argv=exact, cwd=str(work))), flush=True)
        begin = time.monotonic()
        with (output / (name + ".log")).open("xb") as log:
            process = subprocess.run(exact, cwd=work, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, check=False,
                                     timeout=max(1, config["bounds"]["native_timeout_seconds"] - (time.monotonic()-started)))
        record = dict(stage=name, argv=exact, returncode=process.returncode,
                      actual_wait_consumed=True, seconds=time.monotonic()-begin,
                      log_ref=g.file_ref(output / (name + ".log")))
        with lock:
            records.append(record)
            with (output / "actual_native_waits.jsonl").open("a") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        guard(0)
        need(process.returncode == 0, "Relate stage failed: " + name)
    for chunk in chunks:
        for block in stage_plan(commands, binary, chunk, sections[chunk]):
            if len(block) == 1 and "section" not in block[0][0]:
                run(*block[0])
            else:
                with ThreadPoolExecutor(max_workers=spec["workers"]) as executor:
                    futures = [executor.submit(run, *entry) for entry in block]
                    for future in as_completed(futures):
                        future.result()
    first = commands["argv"][0]
    sample_ages = first[first.index("--sample_ages") + 1]
    run("Finalize", [binary, "--mode", "Finalize", "--sample_ages", sample_ages, "-o", "inferred"])
    converter = commands["argv"][1]
    run("ConvertToTreeSequence", converter)
    receipt = dict(schema="paper_argmix_native_terminal_v1", commands_ref=g.file_ref(commands_path),
                   actual_wait_consumed=True, returncodes=[0, 0],
                   returncode_semantics="all original reused and continued staged waits succeeded; converter returned zero",
                   native_stage_waits=records, source_ref=g.file_ref(__file__),
                   original_driver_ref=spec["original_driver_ref"], resume_spec_ref=g.file_ref(spec_path),
                   upstream_wrapper_ref=g.file_ref(source_wrapper), execution_mode="bounded_staged_verified_paint_resume",
                   workers=spec["workers"], chunks=len(chunks), successful=True,
                   reused_completed_stages=[item["stage"] for item in reused],
                   seconds=time.monotonic()-started, logs_removed=False)
    g.Writes(4*1024*1024, guard).json(output / "terminal.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-output-bytes", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    args = parser.parse_args()
    spec = g.metadata(args.spec)
    config = g.metadata(g.verify_ref(spec["config_ref"]))
    import setproctitle
    from arggnn.benchmarking.paper_campaign.runtime import TITLE
    setproctitle.setproctitle(TITLE)
    output_guard = ScopeGuard(args.output, args.max_output_bytes, args.timeout_seconds)
    work_guard = ScopeGuard(spec["native_work_directory"], config["bounds"]["native_relate_bytes"], args.timeout_seconds)
    def guard(required_write_bytes=0):
        output_guard(required_write_bytes)
        work_guard(0)
    result = run_resume(args.spec, args.output, guard=guard)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
