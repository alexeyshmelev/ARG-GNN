"""VAL000-only native ARGformer resume preparation v2; source-only default.

Reuses the sealed successful v1 observation. --stats is an explicitly ROOT-owned
CPU corpus/serialization diagnostic, stdout only. --execute needs a new guarded
ROOT admission. No tsinfer/tsdate inference, training, GPU or old-output writes.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import resource
import sys
import time
from . import expanded_native_argformer_tsinfer_v1 as v1

ROOT = v1.ROOT
SOURCE = "src/arggnn/resume_expanded_native_argformer_tsinfer_preparation_v2.py"
OLD = v1.SOURCE
OLD_SHA = "b3704e780a5c3dbac073ed1cea7cd3fecf660caa54164b9996ee658479b3e600"
PARENT = "ap_expanded_val_000"
OBSERVATION = "data/expanded_native_argformer_tsinfer_canary_v1/parents/ap_expanded_val_000/observation/observation_manifest.json"
OBS_SHA = "bea9e2078652f752f814bf8fbac3503bb50026c3fc369d50a810e03dc7689fe6"
SCIENCE = "research/reviews/expanded_native_argformer_tsinfer_v1_author/config.draft.json"
SCIENCE_SHA = "929b48db93b7cac2c2be3e33c9a8c5f15a14f53124c138279e4e984c00adc024"
RUNTIME = "research/reviews/expanded_full_panel_v1_root/native_launch_helpers_v1/admitted_v1/argformer-canary/preparation_runtime.json"
RUNTIME_SHA = "e3429d1829131ecd1d01d53b6d755d5943d595fc0024bf74c7958109b989f135"
BASE = "data/expanded_native_argformer_tsinfer_canary_v1/parents/ap_expanded_val_000/"
STORAGE = "src/arggnn/project_storage_terminal_retired_scopes_v1.py"
STORAGE_SHA = "bbbdf26206e4e1725c79d2558b4426af848df4cbbd5ac2d0f2bc7051c0606bab"
MODE = "journal_terminal_retired_scopes_credit_v1"
SCHEMA = "native_argformer_VAL000_resume_preparation_v2_root_admission"
DIAGNOSTIC_RSS_CAP = 8589934592
DIAGNOSTIC_SECONDS = 600
SOURCE_PATHS = (SOURCE, OLD, v1.FROZEN, v1.CONTRACTS, v1.PRODUCER, v1.ORIGINAL,
    "src/arggnn/official_argformer_bridge.py", STORAGE,
    "src/arggnn/project_storage_terminal_success_scopes_v1.py",
    "src/arggnn/project_storage_terminal_scopes_v1.py",
    "src/arggnn/project_storage_cpu_closed_scopes_v1.py",
    "src/arggnn/project_storage_closed_scopes_v1.py", "scripts/project_budget_v9.py",
    "scripts/launch_reserved_gpu_v6.py", "src/arggnn/project_budget_v9_consumer.py")


def ref(path):
    return v1._ref(path)


def read(record):
    return v1._read(record)


def science():
    for path, expected in ((OLD, OLD_SHA), (SCIENCE, SCIENCE_SHA), (RUNTIME, RUNTIME_SHA)):
        v1._require(ref(path)["sha256"] == expected, "Frozen successful-observation/scientific runtime pin changed")
    config = read(dict(path=SCIENCE, sha256=SCIENCE_SHA))
    v1._require(config == v1.default_config_v1(), "Unchanged v1 scientific condition required")
    observed = read(dict(path=OBSERVATION, sha256=OBS_SHA))
    v1._require(observed["parent_id"] == PARENT and observed["split"] == "val"
        and observed["source_ref"] == config["source_ref"]
        and observed["config_digest"] == v1._digest(config)
        and observed["all952_actual_dates"] is True and observed["all_original_genotypes_and_sites_verified"] is True
        and observed["truth_or_oracle_read"] is False and observed["native_nodes"] == 8585
        and observed["native_trees"] == 5249 and observed["native_sites_in_support"] == 13661,
        "Exact sealed successful full952 VAL000 observation required")
    return config, observed


def plan_v2():
    config, observed = science()
    return dict(status="source_only_no_diagnostic_or_resume_run", source_ref=ref(SOURCE),
        scientific_adapter_ref=ref(OLD), scientific_config_ref=ref(SCIENCE),
        observation_ref=dict(path=OBSERVATION, sha256=OBS_SHA),
        sealed_tree_ref=observed["inferred_tree_ref"], sealed_roster_ref=observed["roster_ref"],
        runtime_ref=ref(RUNTIME), parent_id=PARENT, mode="resume_only_from_sealed_native_observation",
        source_sha256={path: ref(path)["sha256"] for path in SOURCE_PATHS},
        observed_facts=dict(nodes=8585, trees=5249, original_sites=13661, common_SNPs=13631,
            native_occurrences_exact=952 * 5249, vocabulary_size_exact=8585 + 4,
            unique_full_paths_lower_bound_exclusive=1000000,
            unique_full_paths_upper_bound_inclusive=952 * 5249,
            inference_seconds=208.31832185946405, inference_sampled_RSS_bytes=817336320,
            failed_preparation_seconds=33.898640672676265, failed_preparation_sampled_RSS_bytes=2210861056),
        failed_cap="maximum_unique_full_paths=1000000", persisted_exact_unique_count=False,
        stats_scope="one_sealed_VAL000_exact_native_path_and_serialized_size_counts_stdout_only",
        diagnostic_RSS_cap_bytes=DIAGNOSTIC_RSS_CAP, diagnostic_wall_seconds=DIAGNOSTIC_SECONDS,
        future_output_cap="derive_from_actual_protocol4_sizes_and_uncompressed_join_bound_plus_margin",
        future_CPU_RSS_cap="derive_from_diagnostic_peak_plus_max_payload_and_join_buffers_min8GiB",
        no_reinference=True, no_paths_or_occurrences_removed=True, no_training=True, GPU_used=False)


def peak_RSS():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def diagnostic_boundary(started):
    v1._require(peak_RSS() <= DIAGNOSTIC_RSS_CAP, "Exact native diagnostic exceeds8GiB RSS stop bound")
    v1._require(time.monotonic() - started <= DIAGNOSTIC_SECONDS, "Exact native diagnostic exceeds600s stop bound")


class ByteCounter:
    def __init__(self):
        self.bytes = 0
    def write(self, block):
        self.bytes += len(block)
        return len(block)


def pickle_bytes(value):
    counter = ByteCounter()
    pickle.Pickler(counter, protocol=4).dump(value)
    return counter.bytes


def json_ids_bytes(values):
    return 3 if not values else sum(len(str(value)) for value in values) + 4 * len(values) + 3


def stats_v2(*, root_authorized):
    v1._require(root_authorized is True, "Explicit ROOT-owned stats invocation required")
    config, receipt = science(); v1._CPU_runtime("prepare", config)
    started = time.monotonic()
    observed = v1.read_observed_parent_v1(PARENT, dict(path=OBSERVATION, sha256=OBS_SHA),
                                        config=config["observation"])
    from .official_argformer_bridge import extract_observed_paths, build_vocabulary
    namespace = PARENT + ":" + observed["tree_ref"]["sha256"]
    samples = [int(row["inferred_tree_node_id"]) for row in observed["rows"]]
    corpus = extract_observed_paths(observed["ts"], namespace=namespace, sample_ids=samples)
    diagnostic_boundary(started)
    vocabulary = build_vocabulary(corpus).official_values()
    donors = {int(row["inferred_tree_node_id"]): int(row["observed_reference_class"])
              for row in observed["rows"] if row["role"] == "pure_source_reference"}
    ids = list(range(len(corpus.paths))); random.Random(42).shuffle(ids)
    cut = int(.9 * len(ids)); splits = dict(train=ids[:cut], val=ids[cut:])
    counts, sizes, JSON_sizes = {}, {}, {}
    for split, values in splits.items():
        counts[split] = {}
        for task in ("mlm", "contrastive"):
            chosen = values if task == "mlm" else [i for i in values if corpus.paths[i][0] in donors]
            counts[split][task] = len(chosen)
            name = task + "/" + split
            sizes[name + "/raw_sequences_and_vocab.pkl"] = pickle_bytes(
                dict(raw_sequences=[list(corpus.paths[i]) for i in chosen], vocab=vocabulary))
            JSON_sizes[name + "/path_ids.json"] = json_ids_bytes(chosen)
            if task == "contrastive":
                sizes[name + "/labels.pkl"] = pickle_bytes([donors[corpus.paths[i][0]] for i in chosen])
            diagnostic_boundary(started)
    sizes["embedding/all/raw_all_sequences_and_vocab.pkl"] = pickle_bytes(
        dict(raw_sequences=[list(path) for path in corpus.paths], vocab=vocabulary))
    diagnostic_boundary(started)
    sizes["corpus.pkl"] = pickle_bytes(dict(paths=corpus.paths, occurrences=corpus.occurrences, namespace=namespace))
    diagnostic_boundary(started)
    reference_paths = sum(path[0] in donors for path in corpus.paths)
    grid, sites, queries = len(observed["support"]["grid_indexes"]), len(observed["site_join"]), 686
    join_bytes = 8 * ((grid + sites) * queries + 3 * grid + sites + 3 * queries + 3 + 2 * reference_paths)
    roster_bytes = len((json.dumps(observed["rows"], sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
    primitive_bytes = sum(sizes.values()) + sum(JSON_sizes.values()) + roster_bytes
    retained_bound = primitive_bytes + math.ceil(1.01 * join_bytes) + 65536
    output_cap = math.ceil((1.25 * retained_bound + 64000000) / 100000000) * 100000000
    RSS_cap = math.ceil(max(DIAGNOSTIC_RSS_CAP, peak_RSS() + max(sizes.values()) + 4 * join_bytes + 1073741824)
                        / 1073741824) * 1073741824
    v1._require(len(corpus.occurrences) == 4997048 and len(vocabulary["special"]) +
        len(vocabulary["node_id_tokens"]) == 8589, "Exact observed corpus occurrence/vocabulary count changed")
    result = dict(schema="native_argformer_VAL000_corpus_stats_v2", source_ref=ref(SOURCE),
        observation_ref=dict(path=OBSERVATION, sha256=OBS_SHA), scientific_config_ref=ref(SCIENCE),
        namespace=namespace, parent_id=PARENT, samples=952, queries=686, donors=266,
        unique_full_paths=len(corpus.paths), full_path_occurrences=len(corpus.occurrences),
        vocabulary_size=8589, reference_unique_full_paths=reference_paths,
        full_path_node_tokens=sum(map(len, corpus.paths)), minimum_full_path_length=min(map(len, corpus.paths)),
        maximum_full_path_length=max(map(len, corpus.paths)), native_seed42_path_split_counts=counts,
        protocol4_serialized_bytes=sizes, exact_path_ID_JSON_bytes=JSON_sizes,
        coordinate_joins_uncompressed_payload_bound_bytes=join_bytes,
        prepared_retained_byte_bound_before_margin=retained_bound, suggested_new_output_cap_bytes=output_cap,
        diagnostic_peak_RSS_bytes=peak_RSS(), suggested_preparation_RSS_cap_bytes=RSS_cap,
        elapsed_seconds=time.monotonic() - started,
        suggested_preparation_wall_seconds=max(600, math.ceil(10 * (time.monotonic() - started))),
        full_corpus_before_skip_truncation=True,
        no_array_pickle_tree_outputs_written=True, native_inference_rerun=False,
        model_or_numeric_tests=False, truth_read=False, GPU_used=False)
    diagnostic_boundary(started)
    return result


def admission_draft_v2(stats_ref):
    stats = read(stats_ref)
    plan = plan_v2()
    v1._require(stats["schema"] == "native_argformer_VAL000_corpus_stats_v2"
        and stats["source_ref"] == plan["source_ref"] and stats["observation_ref"] == plan["observation_ref"]
        and stats["scientific_config_ref"] == plan["scientific_config_ref"]
        and stats["full_path_occurrences"] == 4997048 and stats["vocabulary_size"] == 8589
        and 1000000 < stats["unique_full_paths"] <= 4997048 and stats["truth_read"] is False
        and stats["native_inference_rerun"] is False and stats["GPU_used"] is False,
        "Actual same-source exact native diagnostic required")
    return dict(schema=SCHEMA, status="draft_not_admitted", root_authorizer="root_agent",
        execution_authorized=False, fit_authorized=False, GPU_authorized=False, truth_authorized=False,
        parent_id=PARENT, source_ref=plan["source_ref"], source_sha256=plan["source_sha256"],
        scientific_config_ref=plan["scientific_config_ref"], observation_ref=plan["observation_ref"],
        sealed_tree_ref=plan["sealed_tree_ref"], sealed_roster_ref=plan["sealed_roster_ref"],
        runtime_ref=plan["runtime_ref"], stats_ref=stats_ref, storage_adapter_ref=ref(STORAGE),
        job_id="H038-native-argformer-VAL000-resume-preparation-v2",
        output_directory="data/expanded_native_argformer_tsinfer_VAL000_prepared_v2",
        output_cap_bytes=stats["suggested_new_output_cap_bytes"],
        CPU_RSS_cap_bytes=stats["suggested_preparation_RSS_cap_bytes"],
        CPU_wall_seconds=stats["suggested_preparation_wall_seconds"],
        maximum_unique_full_paths=stats["unique_full_paths"], maximum_full_path_occurrences=4997048,
        expires_at_utc="ROOT_SET_ACTIVE_EXPIRY", storage="ROOT_BIND_FRESH_RETIRED_STORAGE",
        original_successful_observation_reused=True, reinference_authorized=False)


def execute_v2(input_ref):
    spec = read(input_ref); expected = admission_draft_v2(spec["stats_ref"])
    expiry = datetime.fromisoformat(spec["expires_at_utc"].replace("Z", "+00:00"))
    for key, value in expected.items():
        if key not in ("status", "execution_authorized", "expires_at_utc", "storage"):
            v1._require(spec.get(key) == value, "Exact measured one-parent resume input required: " + key)
    v1._require(spec["status"] == "active" and spec["execution_authorized"] is True
        and expiry.tzinfo is not None and expiry > datetime.now(timezone.utc),
        "Active ROOT resume-preparation authority required")
    config, receipt = science(); v1._CPU_runtime("prepare", config)
    v1._require(ref(STORAGE)["sha256"] == STORAGE_SHA, "Frozen retired guard required")
    from . import project_storage_terminal_retired_scopes_v1 as storage
    record = spec["storage"]
    needed = [spec[k] for k in ("source_ref", "scientific_config_ref", "observation_ref", "sealed_tree_ref",
                                "sealed_roster_ref", "runtime_ref", "stats_ref", "storage_adapter_ref")]
    needed += [dict(path=p, sha256=h) for p, h in spec["source_sha256"].items()]
    v1._require(record["accounting_mode"] == MODE and record["output_directory"] == spec["output_directory"]
        and record["reserved_new_bytes"] == spec["output_cap_bytes"]
        and all(item in record["reservation"]["consumer_bindings"] for item in needed),
        "Exact fresh retired-scopes source/data/runtime/measured output reservation required")
    storage.validate_storage_v9(spec, dict(output_cap_bytes=spec["output_cap_bytes"]), record,
        source_ref=spec["source_ref"], config_ref=spec["scientific_config_ref"], runtime_ref=spec["runtime_ref"])
    output = v1._path(spec["output_directory"])
    v1._require(not output.exists(), "Fresh resumed preparation output required; v1 artifacts preserved")
    started = time.monotonic()
    def guard(planned=0):
        v1._require(datetime.now(timezone.utc) < expiry and peak_RSS() <= spec["CPU_RSS_cap_bytes"]
            and time.monotonic() - started <= spec["CPU_wall_seconds"],
                    "ROOT resume expiry/RSS cap exceeded")
        return storage.budget_check_v9(record, boundary=True, required_write_bytes=int(planned))
    # Fresh isolated release globals: change resource admission only, no algorithm.
    old_limits = {key: config["resource_planning"][key] for key in (
        "per_parent_output_cap_bytes", "CPU_RSS_cap_bytes", "per_stage_wall_seconds", "maximum_native_nodes",
        "maximum_native_trees", "maximum_native_edges", "maximum_native_sites",
        "maximum_full_path_occurrences", "maximum_unique_full_paths")}
    native = v1._native_namespace(old_limits)
    def complete_corpus(corpus, unused):
        v1._require(len(corpus.paths) == spec["maximum_unique_full_paths"]
            and len(corpus.occurrences) == spec["maximum_full_path_occurrences"],
            "Exact complete measured native corpus changed; no semantic reduction")
        guard()
    native["_validate_corpus_shape"] = complete_corpus
    original_json = native["_json"]
    def publish(path, value, callback):
        if path.name == "prepared_manifest.json":
            value = dict(value, preparation_driver_ref=spec["source_ref"], resource_policy_input_ref=input_ref,
                         resource_policy_stats_ref=spec["stats_ref"], sealed_observation_reused=True,
                         v1_unique_path_planning_cap_superseded_explicitly=True)
        return original_json(path, value, callback)
    native["_json"] = publish
    original_directory = native["_directory"]
    def directory(value, callback):
        result = original_directory(value, callback)
        callback(8192)
        tmp = result / "runtime_tmp"; tmp.mkdir(exist_ok=False)
        os.environ.update(TMPDIR=str(tmp), NUMBA_CACHE_DIR=str(tmp), XDG_CACHE_HOME=str(tmp),
                          PYTHONDONTWRITEBYTECODE="1")
        import tempfile
        tempfile.tempdir = str(tmp)
        return result
    native["_directory"] = directory
    prepared = native["prepare_parent_v1"](PARENT, spec["observation_ref"], output, config=config, guard=guard)
    result = dict(schema="native_argformer_VAL000_resumed_preparation_completed_v2", source_ref=spec["source_ref"],
        input_ref=input_ref, observation_ref=spec["observation_ref"], stats_ref=spec["stats_ref"],
        scientific_config_ref=spec["scientific_config_ref"], prepared_ref=prepared["manifest_ref"],
        parent_id=PARENT, unique_full_paths=prepared["unique_full_paths"], vocabulary_size=prepared["vocabulary_size"],
        full_path_occurrences=prepared["full_path_occurrences"], path_counts=prepared["path_counts"],
        peak_RSS_bytes=peak_RSS(), elapsed_seconds=time.monotonic() - started,
        old_observation_reused=True, inference_rerun=False, frozen_v1_sources_unchanged=True,
        all_paths_and_occurrences_preserved=True, GPU_used=False, fit_started=False, truth_read=False)
    result["manifest_ref"] = v1._save(output / "resume_completed.json", result, guard)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--stats", action="store_true")
    mode.add_argument("--admission-draft", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--root-authorized", action="store_true")
    parser.add_argument("--stats-ref")
    parser.add_argument("--expected-stats-sha256")
    parser.add_argument("--input")
    parser.add_argument("--expected-input-sha256")
    args = parser.parse_args(argv)
    if args.stats:
        value = stats_v2(root_authorized=args.root_authorized)
    elif args.admission_draft:
        if not args.stats_ref or not args.expected_stats_sha256:
            parser.error("Actual ROOT stats JSON and exact SHA required")
        value = admission_draft_v2(dict(path=args.stats_ref, sha256=args.expected_stats_sha256))
    elif args.execute:
        if not args.input or not args.expected_input_sha256:
            parser.error("Active ROOT input and exact SHA required")
        value = execute_v2(dict(path=args.input, sha256=args.expected_input_sha256))
    else:
        value = plan_v2()
    print(json.dumps(value, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
