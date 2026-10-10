#!/usr/bin/env python3
"""Fresh64 TRAIN/16 VAL ancestry preparation. Defaults to read-only preflight.

No shared module is patched. Preparation uses the existing ancient simulator,
Relate pairing, boundary banks and age sidecars. Only --execute can generate
development data. Fixed identity/count/path checks exclude final confirmation.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import stat
import sys
import time

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
PLAN = "configs/fresh_development_preparation_v1.json"
FREEZE = "configs/fresh_development_execution_freeze_v1.json"
COUNTS = {"train": 64, "val": 16, "test": 0}
LAYOUT = {
    "oracle": "data/ancient_development64_v1",
    "inferred": "data/ancient_inferred_development64_v1",
    "boundary": "data/boundary_inferred_development64_v1",
    "age": "data/age_inferred_development64_v1",
    "receipts": "experiments/results/fresh_development_preparation_v1",
}
CORE = ("ancient_simulate.py", "simulate.py", "data.py", "resources.py",
        "reference_encoder.py", "prepare_study.py", "prepare_relate_study.py",
        "relate_pipeline.py", "boundary_dataset.py", "kron_interface.py",
        "age_dataset.py", "age_interface.py", "tree_canonicalization.py",
        "train_study.py", "models.py", "genealogy_baselines.py")
SEED_KEYS = {"ancestry", "mutations", "ancestry_seed", "mutation_seed", "mutations_seed",
             "inference_seed", "seed", "random_seed"}
_EVENT_LOG = None
_CURRENT_PARENT = None


def confined(value, root=ROOT):
    path = Path(value)
    path = (root / path if not path.is_absolute() else path).resolve()
    if path != root.resolve() and root.resolve() not in path.parents:
        raise ValueError(f"Path escapes project: {path}")
    return path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = confined(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def utc():
    return datetime.now(timezone.utc).isoformat()


def configure_cpu():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["TMPDIR"] = str(ROOT / ".runtime/tmp")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "4"
    # No package installs, Git commands, CUDA contexts or GPU launcher.
    import torch
    torch.set_num_threads(4)


def load_plan(path=PLAN):
    path = confined(path)
    plan = read_json(path)
    if (plan.get("kind") != "fresh_development_preparation_v1"
            or plan.get("stage") != "development_only"
            or plan.get("confirmatory_access") is not False
            or plan.get("expected_splits") != COUNTS or plan.get("paths") != LAYOUT):
        raise ValueError("Only fixed fresh development64/16/TEST0 paths are permitted")
    source = confined(plan["oracle_config"])
    oracle = read_json(source)
    if (oracle.get("splits") != COUNTS or oracle.get("master_seed") != 20261101
            or plan["expected_simulation_master_seed"] != 20261101
            or plan["relate_master_seed"] != 20261102
            or oracle.get("benchmark") != "ancient-europe-stdpopsim-0.3.0-oracle"):
        raise ValueError("Development counts, benchmark or fixed master seeds changed")
    simulation = oracle["simulation"]
    expected = {"sequence_length": 1000000, "query_diploids": 32,
                "reference_diploids_per_source": 3, "grid_points": 128,
                "landmarks_per_source": 3, "max_graph_nodes": 1000000,
                "recombination_rate": 1e-8, "mutation_rate": 1.25e-8,
                "reference_times": [260., 300., 250., 250.], "census_time": 350.,
                "required_stdpopsim_version": "0.3.0"}
    if simulation != expected or oracle["windows"] != 8:
        raise ValueError("Frozen ancient development observation contract changed")
    inf = plan["inference"]
    if (inf["master_seed"] != 20261102 or inf["method"] != "Relate"
            or inf["version"] != "1.2.4" or inf["threads"] != 4
            or inf["coordinate_offset"] != 1.):
        raise ValueError("Pinned Relate development settings changed")
    if (plan["storage"]["stop_before_project_bytes"] != 450_000_000_000
            or plan["storage"]["hard_project_bytes"] != 500_000_000_000
            or plan["storage"]["planning_multiplier"] < 2
            or plan["storage"]["scratch_reserve_bytes"] < 10_000_000_000
            or plan["storage"]["concurrent_project_margin_bytes"] < 20_000_000_000):
        raise ValueError("Development storage limits/reserves cannot be relaxed")
    return plan, oracle


def expected_ids():
    return [(split, i) for split in ("train", "val") for i in range(COUNTS[split])]


def validate_parent_records(records):
    ids = [(r["split"], r["replicate"]) for r in records]
    if len(ids) != len(set(ids)) or set(ids) != set(expected_ids()):
        raise RuntimeError("Missing/duplicate development parents or unsupported TEST split")


def planned_registry():
    from arggnn.ancient_simulate import ancient_replicate_seeds
    from arggnn.prepare_relate_study import inference_seed
    return [{"split": split, "replicate": replicate,
             "seeds": ancient_replicate_seeds(20261101, split, replicate),
             "inference_seed": inference_seed(20261102, split, replicate)}
            for split, replicate in expected_ids()]


def seed_uses(value, prefix="$"):
    """Find declared simulation/inference seeds recursively, including bindings."""
    result = []
    if isinstance(value, dict):
        for key, child in value.items():
            location = f"{prefix}.{key}"
            if key in SEED_KEYS and isinstance(child, int) and not isinstance(child, bool):
                result.append({"value": child, "location": location})
            if key == "seeds" and isinstance(child, list):
                result.extend({"value": seed, "location": location + f"[{i}]"}
                              for i, seed in enumerate(child)
                              if isinstance(seed, int) and not isinstance(seed, bool))
            if key == "inference_configuration" and isinstance(child, dict):
                seed = child.get("seed")
                if isinstance(seed, int) and not isinstance(seed, bool):
                    result.append({"value": seed, "location": location + ".seed"})
            result.extend(seed_uses(child, location))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.extend(seed_uses(child, f"{prefix}[{index}]"))
    return result


def manifest_inventory(root=ROOT, exclude_roots=()):
    """ALL repository manifests, including partials/fixtures, plus orphan metadata.

    Never follow symlinks. A malformed prior manifest fails closed. A final
    confirmatory data path blocks the scan without opening its files.
    """
    root = root.resolve()
    excluded = tuple(Path(p).resolve() for p in exclude_roots)
    found = []
    for current, directories, names in os.walk(root, followlinks=False):
        current = Path(current)
        directories[:] = [n for n in directories if n != ".git"
                          and not (current / n).is_symlink()
                          and not any((current / n).resolve() == p for p in excluded)]
        for name in names:
            path = current / name
            if path.is_symlink() or any(path == p or p in path.parents for p in excluded):
                continue
            is_manifest = "manifest" in name.lower() and name.endswith(".json")
            is_metadata = name == "metadata.json" and root / "data" in path.parents
            if not (is_manifest or is_metadata):
                continue
            relative = path.relative_to(root).as_posix()
            if relative.startswith("data/") and any(
                    x in relative.lower() for x in ("confirmatory", "final_test", "final-test")):
                raise RuntimeError("Held-out final data exists; refusing access during development seed audit")
            try:
                payload = read_json(path)
            except (ValueError, UnicodeError) as error:
                raise RuntimeError(f"Unreadable/malformed prior manifest: {relative}") from error
            found.append({"path": relative, "sha256": sha(path), "seed_uses": seed_uses(payload)})
    return sorted(found, key=lambda r: r["path"])


def reject_collisions(registry, inventory):
    proposed = []
    for row in registry:
        for key, value in row["seeds"].items():
            proposed.append((value, f'{row["split"]}:{row["replicate"]}:{key}'))
        proposed.append((row["inference_seed"], f'{row["split"]}:{row["replicate"]}:inference'))
    values = [v for v, _ in proposed]
    if len(values) != len(set(values)):
        raise RuntimeError("Proposed ancestry/mutation/inference seed collision across streams")
    prior = {}
    for record in inventory:
        for use in record["seed_uses"]:
            prior.setdefault(use["value"], []).append(record["path"] + ":" + use["location"])
    collisions = [{"seed": v, "proposed": identity, "prior": prior[v]}
                  for v, identity in proposed if v in prior]
    if collisions:
        raise RuntimeError("Fresh development seed collision: " + json.dumps(collisions))
    return len(prior)


def scan_storage(root=ROOT):
    """Conservative allocated/logical bytes, no symlink traversal/deletion."""
    total = 0
    for current, directories, names in os.walk(root, followlinks=False):
        directories[:] = [n for n in directories if not (Path(current) / n).is_symlink()]
        for name in [*directories, *names]:
            path = Path(current) / name
            try:
                info = path.lstat()
            except FileNotFoundError:
                continue
            total += max(info.st_size, getattr(info, "st_blocks", 0) * 512)
    return total


def storage_guard(plan, used=None, free=None):
    settings = plan["storage"]
    used = scan_storage() if used is None else used
    if free is None:
        fs = os.statvfs(ROOT)
        free = fs.f_bavail * fs.f_frsize
    expected = math_ceil(80 * settings["pilot_combined_bytes_per_parent"]
                         * settings["planning_multiplier"])
    reserve = expected + settings["scratch_reserve_bytes"] + settings["concurrent_project_margin_bytes"]
    if used + reserve >= settings["stop_before_project_bytes"]:
        raise RuntimeError(f"Storage preflight rejected: used{used}+reserve{reserve} reaches450GB")
    if used + reserve >= settings["hard_project_bytes"] or reserve >= free:
        raise RuntimeError("Project hard limit or filesystem free space admission failed")
    return {"used_bytes": used, "development_payload_estimate_bytes": expected,
            "scratch_and_concurrent_margin_bytes": reserve - expected,
            "projected_bytes_with_reserves": used + reserve, "filesystem_free_bytes": free,
            "stop_before_bytes": settings["stop_before_project_bytes"]}


def math_ceil(value):
    import math
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Storage estimate must be finite and positive")
    return math.ceil(value)


def code_pins():
    paths = [ROOT / "scripts/prepare_fresh_development.py",
             *[ROOT / "src/arggnn" / name for name in CORE]]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def cpu_available():
    entries = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, rest = line.split(":", 1)
        if name == "MemAvailable":
            entries[name] = int(rest.split()[0]) * 1024
    if "MemAvailable" not in entries:
        raise RuntimeError("Cannot establish CPU memory headroom")
    return entries["MemAvailable"]


def preflight(plan_path=PLAN, frozen_inventory=()):
    plan, oracle = load_plan(plan_path)
    registry = planned_registry()
    excludes = [confined(plan["paths"][k]) for k in ("oracle", "inferred", "boundary", "age")]
    inventory = manifest_inventory(exclude_roots=excludes)
    # Frozen history still prevents collisions if old fixture/data files vanish.
    unique_prior = reject_collisions(registry, [*inventory, *frozen_inventory])
    from arggnn.ancient_simulate import AncientSimulationConfig, pinned_demography
    from arggnn.relate_pipeline import verified_relate_installation
    from arggnn.prepare_relate_study import SEED_SCHEME
    config = AncientSimulationConfig(**oracle["simulation"])
    config.validate()
    _, provenance = pinned_demography(config)  # Creates demography only; no simulation.
    installation = verified_relate_installation()  # Reads sealed authorization/binary hashes only.
    if (installation["archive_sha256"] != plan["inference"]["archive_sha256"]
            or plan["inference"]["seed_scheme"] != SEED_SCHEME):
        raise RuntimeError("Relate archive or exact seed scheme differs from preparation freeze")
    available = cpu_available()
    if available < plan["storage"]["minimum_cpu_available_bytes"]:
        raise RuntimeError("Need40GB available CPU RAM for existing eager boundary/age loaders")
    result = {"schema_version": 1, "kind": "fresh_development_preparation_execution_freeze",
              "created_utc": utc(), "not_confirmatory_test_freeze": True,
              "data_generated_by_freeze": False, "confirmatory_access": False,
              "config_files": {str(confined(plan_path).relative_to(ROOT)): sha(confined(plan_path)),
                               plan["oracle_config"]: sha(confined(plan["oracle_config"]))},
              "plan": plan, "simulation_protocol": oracle, "parent_registry": registry,
              "source_code_sha256": code_pins(), "prior_inventory": inventory,
              "prior_unique_seed_values": unique_prior,
              "prior_inventory_sha256": digest(inventory),
              "demography_sha256": provenance["model_demography_sha256"],
              "relate_installation": installation,
              "storage": storage_guard(plan), "cpu_available_bytes": available,
              "stage_order": ["oracle", "inferred", "boundary", "age", "verify"],
              "execution_permission": "Only explicit --execute with this freeze's SHA256. Root owns scheduling."}
    return result


def verify_freeze(path, expected_sha):
    if not expected_sha or sha(confined(path)) != expected_sha:
        raise RuntimeError("Explicit frozen-manifest SHA256 missing or changed")
    frozen = read_json(confined(path))
    if frozen.get("kind") != "fresh_development_preparation_execution_freeze":
        raise RuntimeError("Not a development preparation execution freeze")
    if frozen.get("confirmatory_access") is not False or frozen.get("not_confirmatory_test_freeze") is not True:
        raise RuntimeError("Final-test access boundary changed")
    for name, expected in frozen["config_files"].items():
        if sha(confined(name)) != expected:
            raise RuntimeError("Frozen development config drift: " + name)
    if code_pins() != frozen["source_code_sha256"]:
        raise RuntimeError("Frozen preparation implementation drift")
    if frozen["parent_registry"] != planned_registry():
        raise RuntimeError("Frozen parent identity/seed registry drift")
    return frozen


def assert_live_pins(frozen, plan):
    if code_pins() != frozen["source_code_sha256"]:
        raise RuntimeError("Preparation implementation changed during stage")
    for name, expected in frozen["config_files"].items():
        if sha(confined(name)) != expected:
            raise RuntimeError("Preparation config changed during stage")
    return storage_guard(plan)


def report_event(value):
    encoded = json.dumps({"event_utc": utc(), **value}, sort_keys=True, allow_nan=False)
    if _EVENT_LOG is not None:
        _EVENT_LOG.write(encoded + "\n")
        _EVENT_LOG.flush()
    print(encoded, flush=True)


def parent_started(stage, split, replicate):
    global _CURRENT_PARENT
    _CURRENT_PARENT = {"stage": stage, "split": split, "replicate": replicate}
    report_event({"event": "development_parent_attempt_started", **_CURRENT_PARENT})


def save_adapted(case, directory, oracle):
    import torch
    from arggnn.reference_encoder import ReferenceFeatureAdapter
    from arggnn.ancient_simulate import save_ancient_replicate
    record = save_ancient_replicate(case, directory)
    record["tensor_sha256"] = sha(record["tensor_file"])
    record["metadata_sha256"] = sha(record["metadata"])
    adapter = ReferenceFeatureAdapter(case.sequence.x.shape[1], **oracle["reference_adapter"])
    prepared = adapter.precompute(case.sequence)
    cached = directory / "prepared.pt"
    provenance = adapter.manifest_provenance(case.metadata["feature_names"], record["tensor_sha256"])
    torch.save({"sequence": prepared.as_dict(), "adapter": provenance}, cached)
    record.update(prepared_tensor_file=str(cached), prepared_tensor_sha256=sha(cached),
                  adapter_configuration_sha256=provenance["configuration_sha256"],
                  true_switches=case.metadata["total_true_switches"],
                  grid_switches=sum(case.metadata["grid_switch_counts_per_query"]))
    return record


def stage_oracle(plan, oracle, frozen):
    from arggnn.ancient_simulate import AncientSimulationConfig, simulate_ancient_replicate
    target = confined(plan["paths"]["oracle"])
    if target.exists():
        raise RuntimeError("Fresh oracle output exists; no overwrite/resume")
    target.mkdir(parents=True)
    manifest = {"schema_version": 2, "protocol": oracle,
                "protocol_sha256": sha(confined(plan["oracle_config"])),
                "git_commit": None, "git_metadata_policy": "No Git commands. Exact source hashes are pinned.",
                "git_dirty_at_generation": None, "generation_code_sha256": frozen["source_code_sha256"],
                "development_freeze_file_sha256": frozen.get("_file_sha256"),
                "development_freeze_content_sha256": digest({k:v for k,v in frozen.items() if k != "_file_sha256"}),
                "replicates": []}
    config = AncientSimulationConfig(**oracle["simulation"])
    for row in frozen["parent_registry"]:
        assert_live_pins(frozen, plan)
        split, replicate = row["split"], row["replicate"]
        parent_started("oracle", split, replicate)
        case = simulate_ancient_replicate(config, master_seed=20261101, split=split, replicate=replicate)
        if case.metadata["seeds"] != row["seeds"]:
            raise RuntimeError("Simulator seeds disagree with frozen registry")
        record = save_adapted(case, target / split / f"rep{replicate:03d}", oracle)
        manifest["replicates"].append(record)
        # Mutable checkpoint only, never an accepted final manifest.
        (target / "manifest.partial.json").write_text(json.dumps(manifest, indent=2) + "\n")
        report_event({"event": "development_oracle_parent_ready", "split": split,
                      "replicate": replicate, "seeds": record["seeds"]})
        del case
    validate_parent_records(manifest["replicates"])
    verification = verify_panel(manifest, plan, inferred=False)
    write_new(target / "verification.json", verification)
    manifest.update(verification_file=str(target / "verification.json"),
                    verification_sha256=sha(target / "verification.json"))
    assert_live_pins(frozen, plan)
    write_new(target / "manifest.json", manifest)
    return {"manifest": str(target / "manifest.json"), "manifest_sha256": sha(target / "manifest.json")}


def resolved_inferred_protocol(plan, oracle):
    source_path = confined(plan["paths"]["oracle"]) / "manifest.json"
    source = read_json(source_path)
    validate_parent_records(source["replicates"])
    if source["protocol"] != oracle or source["protocol_sha256"] != sha(confined(plan["oracle_config"])):
        raise RuntimeError("Oracle manifest protocol binding differs from frozen development config")
    from arggnn.prepare_relate_study import COMMON_CHOICES
    settings = {key: oracle[key] for key in COMMON_CHOICES}
    settings.update(stage="fresh_development64_paired_inferred_v1_train_validation_only",
                    benchmark="ancient-europe-stdpopsim-0.3.0-relate-inferred",
                    evaluation="Exact original128midpoints in observed SNP coverage, original64query order, no tails.",
                    interpretation="Fresh64TRAIN/16VAL/TEST0 development preparation only, no final-test access.",
                    source_parent={"config": plan["oracle_config"],
                                   "config_sha256": sha(confined(plan["oracle_config"])),
                                   "manifest": str(source_path),
                                   "manifest_sha256": sha(source_path),
                                   "benchmark": oracle["benchmark"],
                                   "split_policy": "Same64TRAIN/16VALfresh parents, TEST0."},
                    inference=plan["inference"])
    return source, settings


def coverage_admission(record, plan):
    import math
    policy = plan["admission"]
    reasons = []
    fraction = record["physical_coverage_fraction"]
    grid, nodes = record["retained_grid_points"], record["packed_nodes"]
    if record["native_returncodes"] != [0, 0]:
        reasons.append("native_returncodes_nonzero")
    if not math.isfinite(fraction) or not policy["min_physical_coverage_fraction"] <= fraction <= 1:
        reasons.append("physical_coverage_below_threshold_or_invalid")
    if isinstance(grid, bool) or not isinstance(grid, int) or not policy["min_retained_grid_points"] <= grid <= 128:
        reasons.append("retained_grid_count_below_threshold_or_invalid")
    if isinstance(nodes, bool) or not isinstance(nodes, int) or not 0 < nodes <= policy["max_packed_nodes"]:
        reasons.append("packed_node_count_outside_limit")
    if reasons:
        observation = {k:record[k] for k in ("native_returncodes", "physical_coverage_fraction",
                                           "retained_grid_points", "packed_nodes")}
        report_event({"event": "development_parent_rejected", "parent": _CURRENT_PARENT,
                      "label_blind_reasons": reasons,
                      "observation": {k: (v if not isinstance(v,float) or math.isfinite(v) else repr(v))
                                      for k,v in observation.items()},
                      "no_parent_removed_or_replaced": True})
        raise RuntimeError("Whole development parent rejected by label-blind admission: " + ",".join(reasons))


def stage_inferred(plan, oracle, frozen):
    """Generic development adapter around verified v1.2.4 pairing primitives."""
    import numpy as np
    import torch
    import tskit
    from arggnn.reference_encoder import ReferenceFeatureAdapter
    from arggnn.prepare_relate_study import parent_binding, verify_pair
    from arggnn.relate_pipeline import (
        COPYRIGHT_NOTICE, RelateConfig, load_ancient_parent, prepare_relate_inputs, run_paired_relate)
    target = confined(plan["paths"]["inferred"])
    if target.exists():
        raise RuntimeError("Fresh inferred output exists; no overwrite/resume")
    source, settings = resolved_inferred_protocol(plan, oracle)
    source_records = sorted(source["replicates"], key=lambda r: (r["split"] != "train", r["replicate"]))
    # Every source hash is checked before the first native inference.
    bindings = [parent_binding(r, settings) for r in source_records]
    registry = {(r["split"], r["replicate"]): r for r in frozen["parent_registry"]}
    target.mkdir(parents=True)
    protocol_path = target / "protocol.json"
    write_new(protocol_path, settings)
    manifest = {"schema_version": 2, "protocol": settings, "protocol_sha256": sha(protocol_path),
                "git_commit": None, "git_dirty_at_generation": None,
                "git_metadata_policy": "No Git commands. Exact source hashes are pinned.",
                "generation_code_sha256": frozen["source_code_sha256"],
                "source_parent": settings["source_parent"],
            "development_freeze_file_sha256": frozen.get("_file_sha256"),
            "relate_installation": frozen["relate_installation"],
                "versions": {"python": sys.version, "torch": str(torch.__version__),
                             "numpy": np.__version__, "tskit": tskit.__version__},
                "copyright_notice": COPYRIGHT_NOTICE, "replicates": []}
    for source_record, binding in zip(source_records, bindings):
        assert_live_pins(frozen, plan)
        started = time.monotonic()
        split, replicate = source_record["split"], source_record["replicate"]
        parent_started("inferred", split, replicate)
        planned = registry[(split, replicate)]
        if source_record["seeds"] != planned["seeds"]:
            raise RuntimeError("Source simulation seeds differ from frozen development registry")
        parent, observed = load_ancient_parent(binding["directory"])
        directory = target / split / f"rep{replicate:03d}"
        inf = plan["inference"]
        for name in ("mutation_rate", "recombination_rate"):
            if parent.metadata["config"][name] != inf[name]:
                raise RuntimeError("Paired inference rate changed: " + name)
        prep = prepare_relate_inputs(parent, observed, directory / "inputs",
                                    min_mac=inf["min_mac"], recombination_rate=inf["recombination_rate"])
        native = RelateConfig(**{k: inf[k] for k in
            ("mutation_rate", "effective_haploid_size", "memory_gb", "timeout_seconds", "threads")},
            seed=planned["inference_seed"])
        case = run_paired_relate(parent, prep, directory / "inference", config=native)
        alignment = verify_pair(parent, case)
        obs = case.tree_sequence
        if (obs.tables.nodes.metadata.size or obs.tables.edges.metadata.size
                or bool((obs.tables.nodes.population != tskit.NULL).any()) or obs.num_migrations):
            raise RuntimeError("Internal ancestry annotations survived inferred observation cleaning")
        raw = directory / "inference/dataset"
        tensor, metadata = raw / "tensors.pt.gz", raw / "metadata.json"
        case.metadata.update(
            benchmark=settings["benchmark"], parent_benchmark=parent.metadata["benchmark"],
            source_parent=binding, paired_alignment=alignment,
            inference_configuration=asdict(native), inference_seed=planned["inference_seed"],
            inference_seed_scheme=inf["seed_scheme"], inference_master_seed=20261102,
            protocol_sha256=manifest["protocol_sha256"],
            grid_switch_counts_per_query=(case.labels[1:] != case.labels[:-1]).sum(0).tolist(),
            grid_class_counts=torch.bincount(case.labels.flatten(), minlength=4).tolist())
        case.metadata["files"] = {p.name: p.stat().st_size for p in raw.iterdir()
                                  if p.is_file() and p.name != "metadata.json"}
        metadata.write_text(json.dumps(case.metadata, indent=2) + "\n")
        adapter = ReferenceFeatureAdapter(case.sequence.x.shape[1], **oracle["reference_adapter"])
        prepared = adapter.precompute(case.sequence)
        tensor_hash = sha(tensor)
        provenance = adapter.manifest_provenance(case.metadata["feature_names"], tensor_hash)
        cached = raw / "prepared.pt"
        torch.save({"sequence": prepared.as_dict(), "adapter": provenance}, cached)
        steps = case.metadata["inference_provenance"]["steps"]
        record = {"directory": str(raw), "metadata": str(metadata), "metadata_sha256": sha(metadata),
                  "tensor_file": str(tensor), "tensor_sha256": tensor_hash,
                  "prepared_tensor_file": str(cached), "prepared_tensor_sha256": sha(cached),
                  "adapter_configuration_sha256": provenance["configuration_sha256"],
                  "split": split, "replicate": replicate, "seeds": parent.metadata["seeds"],
                  "source_parent": binding, "inference_seed": planned["inference_seed"],
                  "inference_configuration": asdict(native),
                  "observed_tree_file": str(raw / "observed.trees.gz"),
                  "observed_tree_sha256": sha(raw / "observed.trees.gz"),
                  "native_inferred_tree_sha256": sha(directory / "inference/inferred.trees"),
                  "preparation_sha256": sha(directory / "inputs/preparation.json"),
                  "execution_sha256": sha(directory / "inference/execution.json"),
                  "parent_grid_points": case.metadata["parent_grid_points"],
                  "retained_grid_points": len(case.sequence.positions),
                  "physical_coverage_bp": case.metadata["coverage_physical_bp"],
                  "physical_coverage_fraction": case.metadata["physical_coverage_fraction"],
                  "grid_coverage_fraction": case.metadata["grid_coverage_fraction"],
                  "paired_alignment": alignment, "exported_sites": prep["num_sites"],
                  "site_filter_counts": prep["filter_counts"],
                  "inferred_trees": case.sequence.num_graphs, "packed_nodes": len(case.sequence.x),
                  "native_returncodes": [s["returncode"] for s in steps],
                  "inference_elapsed_seconds": sum(s["elapsed_seconds"] for s in steps),
                  "preparation_elapsed_seconds": time.monotonic() - started,
                  "bytes": sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())}
        coverage_admission(record, plan)
        manifest["replicates"].append(record)
        (target / "manifest.partial.json").write_text(json.dumps(manifest, indent=2) + "\n")
        report_event({"event": "development_inferred_parent_ready", "split": split,
                      "replicate": replicate, "physical_coverage_fraction": record["physical_coverage_fraction"],
                      "retained_grid_points": record["retained_grid_points"]})
        del parent, observed, case, prepared
    validate_parent_records(manifest["replicates"])
    verification = verify_panel(manifest, plan, inferred=True)
    write_new(target / "verification.json", verification)
    manifest.update(verification_file=str(target / "verification.json"),
                    verification_sha256=sha(target / "verification.json"),
                    summary={"train_parents": 64, "validation_parents": 16, "test_parents": 0,
                             "all_native_returncodes_zero": True,
                             "total_retained_grid_points": sum(r["retained_grid_points"] for r in manifest["replicates"]),
                             "total_parent_grid_points": 80 * 128})
    assert_live_pins(frozen, plan)
    write_new(target / "manifest.json", manifest)
    return {"manifest": str(target / "manifest.json"), "manifest_sha256": sha(target / "manifest.json"),
            "protocol": str(protocol_path), "protocol_sha256": sha(protocol_path),
            "parents": 80, "test_parents": 0}


def verify_panel(manifest, plan, *, inferred):
    import torch
    from arggnn.simulate import load_replicate, split_fixed_windows
    from arggnn.train_study import load_cases
    from arggnn.prepare_relate_study import verify_pair
    from arggnn.tree_canonicalization import canonicalize_observed_trees
    validate_parent_records(manifest["replicates"])
    records = {(r["split"], r["replicate"]): r for r in manifest["replicates"]}
    checked = []
    for split in ("train", "val"):
        cases = load_cases(manifest, split)  # Existing hash/prefix/adapter verification.
        for case in cases:
            parent_started("verify_inferred" if inferred else "verify_oracle", split, case.metadata["replicate"])
            row = records[(split, case.metadata["replicate"])]
            if case.metadata["master_seed"] != 20261101 or case.metadata["seeds"] != row["seeds"]:
                raise RuntimeError("Fresh development parent identity changed")
            if case.sequence.query_node_ids.numel() != 64 or len(case.metadata["samples"]) != 88:
                raise RuntimeError("Development query/reference sample count changed")
            if inferred:
                coverage_admission(row, plan)
                parent = load_replicate(row["source_parent"]["directory"])
                alignment = verify_pair(parent, case)
                if list(Path(row["directory"]).glob("truth.trees*")):
                    raise RuntimeError("Raw truth genealogy leaked into inferred dataset directory")
                del parent
            else:
                if len(case.sequence.positions) != 128:
                    raise RuntimeError("Oracle fixed midpoint count changed")
                alignment = {"original_grid_count": 128}
            # Full observed parent, before tiling; canonicalization sees sequence only.
            canonical = canonicalize_observed_trees(case.sequence, time_column=6,
                parent_branch_column=7, base_feature_count=8, time_scale=10000.,
                merge_identical_adjacent=True)
            if (not torch.equal(canonical.sequence.positions, case.sequence.positions)
                    or not torch.equal(canonical.sequence.query_node_ids, case.sequence.query_node_ids)):
                raise RuntimeError("Canonicalization changed queries or scoring coordinates")
            tiles = split_fixed_windows(case, 8)
            if (len(tiles) != 8 or not torch.equal(torch.cat([t.labels for t in tiles]), case.labels)
                    or not torch.equal(torch.cat([t.sequence.positions for t in tiles]), case.sequence.positions)):
                raise RuntimeError("Fixed tiles do not cover retained targets exactly once and in order")
            report_event({"event": "development_parent_verification_admitted", "split": split,
                          "replicate": case.metadata["replicate"], "inferred": inferred,
                          "retained_grid_points": len(case.sequence.positions), "tiles": len(tiles)})
            checked.append({"split": split, "replicate": case.metadata["replicate"],
                            "prepared_feature_dim": case.sequence.x.shape[1], "tiles": len(tiles),
                            "canonicalization_admitted": True, **alignment})
            del canonical
        del cases
    return {"train_parents": 64, "validation_parents": 16, "test_parents": 0,
            "all_parent_hashes_and_adapter_bindings_verified": True,
            "all_targets_queries_coordinates_and_tiles_exact": True,
            "full_observed_parent_canonicalization_admitted": True, "parents": checked}


def stage_boundary(plan, oracle, frozen):
    from arggnn.boundary_dataset import prepare_boundary_dataset
    source = confined(plan["paths"]["inferred"])
    protocol = source / "protocol.json"
    manifest = read_json(source / "manifest.json")
    validate_parent_records(manifest["replicates"])
    # Full-parent electrical admission gets its own durable attempt/failure ID.
    verify_panel(manifest, plan, inferred=True)
    from arggnn.boundary_dataset import _bank, _preparation, _reference_ids
    from arggnn.train_study import load_cases
    settings = _preparation(manifest, windows=8, **plan["boundary"])
    for split in ("train", "val"):
        cases = load_cases(manifest, split)
        for case in cases:
            parent_started("electrical_admission", split, case.metadata["replicate"])
            assert_live_pins(frozen, plan)
            bank = _bank(case.sequence, settings)
            _reference_ids(case.sequence, bank, settings["reference_columns"])
            report_event({"event": "development_electrical_parent_admitted", "split": split,
                          "replicate": case.metadata["replicate"], "graphs_checked": len(bank.diagnostics),
                          "maximum_interior_condition": max(d.interior_condition_number for d in bank.diagnostics)})
            del bank
        del cases
    def progress(event):
        if event["event"] == "parent_admissible" or event.get("window_index") == 0:
            parent_started("boundary", event["split"], event["replicate"])
            assert_live_pins(frozen, plan)
        report_event(event)
    result = prepare_boundary_dataset(source / "manifest.json", confined(plan["paths"]["boundary"]),
        source_protocol_path=protocol, windows=8, **plan["boundary"],
        expected_source_manifest_sha256=sha(source / "manifest.json"),
        expected_source_protocol_sha256=sha(protocol), progress=progress)
    return {"manifest": str(result.directory / "manifest.json"),
            "manifest_sha256": result.manifest_sha256,
            "preparation_sha256": result.preparation_sha256,
            "parents": len(result.parent_cases), "tiles": len(result.records)}


def stage_age(plan, oracle, frozen):
    from arggnn.age_dataset import prepare_age_dataset
    boundary = confined(plan["paths"]["boundary"])
    manifest = read_json(boundary / "manifest.json")
    counter = 0
    def progress(event):
        nonlocal counter
        counter += 1
        if counter % 8 == 1:
            global _CURRENT_PARENT
            identity = event["tile_id"].rsplit(":window", 1)[0]
            _CURRENT_PARENT = {"stage": "age", "parent_id": identity}
            report_event({"event": "development_age_parent_attempt_started", **_CURRENT_PARENT})
            assert_live_pins(frozen, plan)
        report_event(event)
    prepare_age_dataset(boundary, confined(plan["paths"]["age"]),
        expected_boundary_manifest_sha256=sha(boundary / "manifest.json"),
        expected_boundary_preparation_sha256=manifest["preparation_sha256"],
        **plan["age"], progress=progress)
    path = confined(plan["paths"]["age"]) / "manifest.json"
    return {"manifest": str(path), "manifest_sha256": sha(path),
            "preparation_sha256": read_json(path)["preparation_sha256"]}


def stage_verify(plan, oracle, frozen):
    from arggnn.boundary_dataset import load_boundary_dataset
    from arggnn.age_dataset import load_age_caches
    result = {"all_stages_complete": False, "test_parents": 0, "stage_manifests": {}}
    for stage in ("oracle", "inferred", "boundary", "age"):
        path = confined(plan["paths"][stage]) / "manifest.json"
        payload = read_json(path)
        fact = {"path": str(path), "sha256": sha(path)}
        for field in ("protocol_sha256", "preparation_sha256", "source_manifest_sha256",
                      "boundary_manifest_sha256", "boundary_preparation_sha256", "verification_sha256"):
            if field in payload:
                fact[field] = payload[field]
        if "source" in payload:
            fact["source"] = payload["source"]
        if "source_parent" in payload:
            fact["source_parent"] = payload["source_parent"]
        if stage == "inferred":
            fact["protocol_path"] = str(path.parent / "protocol.json")
        result["stage_manifests"][stage] = fact
    verify_panel(read_json(confined(plan["paths"]["oracle"]) / "manifest.json"), plan, inferred=False)
    verify_panel(read_json(confined(plan["paths"]["inferred"]) / "manifest.json"), plan, inferred=True)
    boundary_path = confined(plan["paths"]["boundary"]) / "manifest.json"
    boundary = load_boundary_dataset(boundary_path.parent,
        expected_manifest_sha256=sha(boundary_path))
    if len(boundary.parent_cases) != 80 or len(boundary.records) != 640:
        raise RuntimeError("Development boundary parent/tile counts changed")
    age_path = confined(plan["paths"]["age"]) / "manifest.json"
    caches = load_age_caches(age_path.parent, boundary,
        expected_manifest_sha256=sha(age_path),
        expected_preparation_sha256=read_json(age_path)["preparation_sha256"])
    if len(caches) != 640:
        raise RuntimeError("Development age sidecars do not cover every tile")
    assert_live_pins(frozen, plan)
    result.update(all_stages_complete=True, train_parents=64, validation_parents=16,
                  inferred_boundary_and_age_tiles=640,
                  verified_at_utc=utc(), numerical_ancestry_performance_claim=False,
                  development_execution_freeze_file_sha256=frozen.get("_file_sha256"),
                  preparation_source_code_sha256=frozen["source_code_sha256"],
                  parent_registry_sha256=digest(frozen["parent_registry"]),
                  admission_policy=plan["admission"],
                  ledger_directory=plan["paths"]["receipts"],
                  training_or_confirmatory_selection_frozen=False)
    # Tracked compact completion facts for root's subsequent bound train configs.
    write_new(confined(plan["paths"]["receipts"]) / "completion.json", result)
    return result


STAGES = {"oracle": stage_oracle, "inferred": stage_inferred,
          "boundary": stage_boundary, "age": stage_age, "verify": stage_verify}


def execute(stage, plan, oracle, frozen, freeze_sha):
    global _EVENT_LOG, _CURRENT_PARENT
    # Durable failed-run receipts are outside dataset manifests, never overwrite.
    _CURRENT_PARENT = None
    receipts = confined(plan["paths"]["receipts"])
    receipts.mkdir(parents=True, exist_ok=True)
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + f"_{os.getpid()}_{stage}.json"
    receipt = {"stage": stage, "started_utc": utc(), "freeze_sha256": freeze_sha,
               "development_only": True, "test_parents": 0, "status": "started",
               "parent_attempt_ledger": str(receipts / (name + ".events.jsonl"))}
    _EVENT_LOG = (receipts / (name + ".events.jsonl")).open("x")
    try:
        assert_live_pins(frozen, plan)
        receipt["result"] = STAGES[stage](plan, oracle, frozen)
        receipt["status"] = "complete"
    except Exception as error:
        receipt.update(status="failed", exception_type=type(error).__name__, exception=str(error),
                       failed_or_last_active_parent=_CURRENT_PARENT)
        report_event({"event": "development_stage_failed", "stage": stage,
                      "parent": _CURRENT_PARENT, "exception_type": type(error).__name__,
                      "reason": str(error), "no_parent_removed_or_replaced": True})
        raise
    finally:
        receipt["finished_utc"] = utc()
        write_new(receipts / name, receipt)
        report_event({"event": "development_stage_receipt", "stage": stage,
                      "status": receipt["status"], "receipt": str(receipts / name)})
        _EVENT_LOG.close()
        _EVENT_LOG = None
        _CURRENT_PARENT = None
        gc.collect()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true", help="Default: no dataset writes or simulation")
    mode.add_argument("--freeze", action="store_true", help="Write a fresh development preparation freeze only")
    mode.add_argument("--execute", action="store_true", help="Root-only explicit CPU development preparation")
    parser.add_argument("--config", default=PLAN)
    parser.add_argument("--freeze-manifest", default=FREEZE)
    parser.add_argument("--freeze-sha256")
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    args = parser.parse_args(argv)
    configure_cpu()
    if args.execute:
        frozen = verify_freeze(args.freeze_manifest, args.freeze_sha256)
        frozen["_file_sha256"] = args.freeze_sha256
        current = preflight(args.config, frozen_inventory=frozen["prior_inventory"])
        plan, oracle = load_plan(args.config)
        if current["config_files"] != frozen["config_files"]:
            raise RuntimeError("Requested development config differs from frozen execution config")
        stages = list(STAGES) if args.stage == "all" else [args.stage]
        for stage in stages:
            execute(stage, plan, oracle, frozen, args.freeze_sha256)
    else:
        result = preflight(args.config)
        if args.freeze:
            if any(confined(result["plan"]["paths"][name]).exists()
                   for name in ("oracle", "inferred", "boundary", "age")):
                raise RuntimeError("Initial development freeze requires all dataset directories fresh")
            write_new(args.freeze_manifest, result)
        report_event({"event": "development_preparation_freeze" if args.freeze else "development_preflight",
            "data_generated": False, "test_parents": 0, "parent_count": len(result["parent_registry"]),
            "prior_manifests_and_metadata_scanned": len(result["prior_inventory"]),
            "prior_unique_seed_values": result["prior_unique_seed_values"],
            "proposed_unique_seed_values": 240, "prior_inventory_sha256": result["prior_inventory_sha256"],
            "storage": result["storage"], "cpu_available_bytes": result["cpu_available_bytes"],
            **({"freeze_manifest": str(confined(args.freeze_manifest)),
                "freeze_sha256": sha(confined(args.freeze_manifest))} if args.freeze else {})})


if __name__ == "__main__":
    main()
