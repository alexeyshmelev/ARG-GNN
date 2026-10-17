"""Authenticate closed compact producers, assemble all20 metadata, never payloads.

build_values is pure metadata authentication; build writes exactly three bounded
JSON files under the existing charged compact admin scope. No reserve, launch,
close, cleanup, scientific payload read or model execution is performed here.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import re
import stat
import sys
from research.reviews.frozen_TreeLoom_bounded_training_source_20261017_v1 import contract_v1 as c
from research.reviews.TreeLoom_compact_original_focal_source_20261017_v1 import compact_focals_v1 as focal

HERE = "research/reviews/TreeLoom_compact_all20_full600_control_source_20261017_v1"
SOURCE = HERE + "/readiness_ROOT_v1.py"
MANIFEST = HERE + "/source_manifest.v1.json"
PARENTS = ["american_c3_train_%03d" % i for i in range(20)]
COMPACT = "compact_original_identity_focals_without_native_corpus"
NATIVE = "existing_full_native_corpus"
READINESS_SCHEMA = "TreeLoom_all20_training_readiness_with_compact_focals_v1"
MAX_JSON = 65536
MAX_TOTAL = 256 * 1024
SELECTOR = dict(path="research/reviews/Census_observation_ROOT_activation_20261017_v1/explicit_child_selector.root_active_v1.json",
    sha256="fd05d8934ed29eed39f7e60121a33e48fbeb0106991e76690664ca96355692e4")


def short_ref_v1(value):
    c.require(type(value) is dict and set(value) == {"path", "sha256"}
        and type(value["path"]) is str and type(value["sha256"]) is str
        and re.fullmatch("[0-9a-f]{64}", value["sha256"]) is not None,
        "Exact compact immutable ref")
    c.local(value["path"])
    return dict(value)


def source_pass_v1(record, manifest_ref):
    review = c.read(short_ref_v1(record))
    c.require(review.get("independent") is True and review.get("verdict") == "SOURCE_PASS"
        and review.get("blockers") == []
        and (review.get("candidate_manifest_ref") == manifest_ref
             or review.get("source_manifest_ref") == manifest_ref),
        "Exact independent source pass for this producer/adapter manifest")
    return review


def closed_v1(record, parent, *, input_ref=None, stage=None):
    accepted = c.read(short_ref_v1(record))
    receipt = c.read(short_ref_v1(accepted["receipt_ref"]))
    c.require(accepted["parent_id"] == parent
        and accepted["actual_direct_OS_wait0"] is True
        and accepted["all_owned_processes_and_groups_drained"] is True
        and accepted.get("UI_tool_terminal_claimed", False) is False
        and type(receipt["key"]) is str and bool(receipt["key"]),
        "Genuine successful ordinary producer close acceptance")
    if stage == "compact_focals":
        c.require(accepted["stage"] == stage, "Exact new compact closed producer stage")
    elif stage is not None:
        # Existing ROOT TRAIN001 close acceptance may predate its stage field.
        c.require(accepted.get("stage", stage) == stage and input_ref is not None,
            "Legacy inference acceptance still requires its exact admitted producer")
    if input_ref is not None:
        inp = c.read(short_ref_v1(input_ref))
        c.require(receipt["key"] == inp["storage"]["reservation"]["key"]
            and inp["parent_id"] == parent
            and (stage is None or inp["stage"] == stage),
            "Close receipt belongs to this exact admitted parent/stage producer")
    return accepted


def observation_v1(record, parent):
    observed = c.read(short_ref_v1(record))
    c.require(observed["schema"] == "american_c3_observed_v1"
        and observed["parent_id"] == parent and observed["split"] == "train"
        and observed["sequence_length_bp"] == 30000000
        and observed["genealogy_kind"] == focal.GENEALOGY
        and observed["class_order"] == list(focal.CLASSES)
        and observed["query_ancestry_truth_in_inputs"] is False
        and observed["sites_and_sample_identities_verified"] is True,
        "Actual complete original sanitized inferred30Mb observation metadata")
    # The observation producer authenticated these payloads. Do not hash/open them.
    for key in ("tree_ref", "genotypes_ref", "positions_ref", "alleles_ref"):
        short_ref_v1(observed[key])
    roster = c.read(short_ref_v1(observed["roster_ref"]))
    rows = roster["samples"]
    c.require(roster["parent_id"] == parent and roster["class_order"] == list(focal.CLASSES)
        and len(rows) == 1600
        and [r["sample_index"] for r in rows] == list(range(1600))
        and [r["role"] for r in rows] == ["pure_source_reference"] * 600 + ["query"] * 1000
        and [r["observed_reference_class"] for r in rows] == [2] * 200 + [0] * 200 + [1] * 200 + [None] * 1000,
        "Exact original ordered1600sample roster; no query ancestry")
    return observed


def simulation_v1(record, parent):
    done = c.read(short_ref_v1(record))
    c.require(done["parent_id"] == parent and done["status"] == "completed_C3_simulation_reconstruction",
        "Original same-parent simulation completion provenance")
    private = short_ref_v1(done["private_truth_ref"])
    c.require(private["path"].endswith(".npz") and "/truth/" in private["path"]
        and not any(x in private["path"].lower() for x in ("val_", "test_")),
        "Original TRAIN-only truth handle; payload remains unopened")
    return private


def compact_parent_v1(record, parent, parameters):
    ready_ref = short_ref_v1(record)
    ready = c.read(ready_ref)
    c.require(ready["schema"] == "TreeLoom_compact_focal_parent_ready_v1"
        and ready["parent_id"] == parent and parent in PARENTS[1:]
        and ready["focal_preparation_kind"] == COMPACT
        and ready["query_truth_opened"] is False and ready["output_frozen"] is True
        and ready["source_manifest_ref"] == parameters["compact_preparation_manifest_ref"],
        "Exact genuinely completed compact focal frontier")
    inp_ref = short_ref_v1(ready["compact_input_ref"])
    inp = c.read(inp_ref)
    request = c.read(short_ref_v1(inp["root_stage_request_ref"]))
    manifest = c.read(parameters["compact_preparation_manifest_ref"])
    review_ref = parameters["compact_preparation_source_review_ref"]
    source_pass_v1(review_ref, parameters["compact_preparation_manifest_ref"])
    expected_sources = {r["path"]: r["sha256"] for r in manifest["refs"]}
    c.require(request["source_manifest_ref"] == ready["source_manifest_ref"]
        and request["source_review_ref"] == review_ref
        and inp["source_sha256"] == request["source_sha256"] == expected_sources,
        "Complete compact producer source closure")
    c.require(inp["parent_id"] == parent and inp["stage"] == "compact_focals"
        and inp["phase"] == "compact_focals_" + parent.removeprefix("american_c3_")
        and inp["GPU_authorized"] is False and inp["query_truth_access_authorized"] is False
        and inp["TEST_access_authorized"] is False
        and inp["GNN_optimization_authorized"] is False
        and inp["refs"]["config"] == ready["compact_config_ref"],
        "Exact admitted compact CPU-only producer")
    cfg = c.read(short_ref_v1(ready["compact_config_ref"]))
    observed_ref = short_ref_v1(ready["observation_ref"])
    c.require(cfg == focal.make_config_v1(parent, observed_ref, execution_enabled=True),
        "Unchanged original parent focal seed/traversal failstops")
    done_ref = short_ref_v1(ready["compact_completed_ref"])
    c.require(done_ref == ready["native_preparation_ref"]
        and done_ref == c.ref(inp["output_directory"] + "/completed.json"),
        "Compatibility preparation key binds real compact completion")
    done = c.read(done_ref)
    focals_ref = short_ref_v1(ready["focal_registry_ref"])
    seal_ref = short_ref_v1(ready["focal_seal_ref"])
    identity = ready["focal_identity_sha256"]
    c.require(type(identity) is str and re.fullmatch("[0-9a-f]{64}", identity) is not None,
        "Exact original six-field focal identity SHA")
    forbidden = {"native_dataset_refs", "full_path_id_order_refs", "vocab_ref", "sequence_to_tree_indices_ref"}
    c.require(done["schema"] == focal.COMPLETION_SCHEMA and done["status"] == focal.STATUS
        and done["parent_id"] == parent and done["observation_ref"] == observed_ref
        and done["genealogy_kind"] == focal.GENEALOGY and done["class_order"] == list(focal.CLASSES)
        and done["focals_ref"] == focals_ref and done["focal_seal_ref"] == seal_ref
        and done["focal_identity_sha256"] == identity and done["query_focal_count"] == 10000
        and done["path_sampling_seed"] == cfg["path_sampling_seed"]
        and done["full_original_path_traversal"] is True
        and done["reference_paths_counted_in_global_record_ids"] is True
        and done["native_corpus_materialized"] is False and done["native_dataset_files_written"] == 0
        and done["native_vocabulary_materialized"] is False
        and done["native_train_val_split_executed"] is False
        and done["sequence_to_tree_occurrence_mapping_materialized"] is False
        and done["query_ancestry_truth_opened"] is False
        and not forbidden.intersection(done),
        "Honest original-identity10000focal preparation without any native corpus claim")
    seal = c.read(seal_ref)
    c.require(seal["schema"] == "TreeLoom_original_native_focal_identity_seal_v1"
        and seal["parent_id"] == parent and seal["observation_ref"] == observed_ref
        and seal["focal_registry_ref"] == focals_ref and seal["focal_identity_sha256"] == identity
        and seal["focal_records"] == 10000 and seal["query_truth_opened"] is False,
        "Exact original-format focal identity seal, no focal payload read")
    observed = observation_v1(observed_ref, parent)
    wait = c.read(short_ref_v1(ready["direct_wait_ref"]))
    supervisor = c.read(short_ref_v1(ready["actual_supervisor_wait_ref"]))
    outer = c.read(short_ref_v1(ready["actual_outer_terminal_ref"]))
    c.require(wait["schema"] == "American_C3_full_original_CPU_actual_wait_v1"
        and wait["parent_id"] == parent and wait["phase"] == inp["phase"]
        and wait["input_ref"] == inp_ref and wait["completed_ref"] == done_ref
        and wait["actual_direct_worker_wait"] is True and wait["actual_direct_launcher_wait"] is True
        and type(wait["actual_OS_wait_status"]) is int and wait["actual_OS_wait_status"] == 0
        and type(wait["returncode"]) is int and wait["returncode"] == 0
        and wait["all_owned_processes_and_groups_drained"] is True
        and wait["cleanup_errors"] == [] and wait["operational_failure"] is None
        and all(x["returncode"] == 0 for x in wait["adopted_waits"]),
        "Actual compact worker/launcher wait0 and owned drain")
    c.require(supervisor["actual_direct_wait"] is True
        and type(supervisor["actual_OS_wait_status"]) is int and supervisor["actual_OS_wait_status"] == 0
        and type(supervisor["actual_returncode"]) is int and supervisor["actual_returncode"] == 0
        and supervisor["all_owned_processes_and_groups_drained"] is True
        and supervisor["pid"] == wait["supervisor_pid"]
        and supervisor["input_ref"] == inp_ref and supervisor["completed_ref"] == done_ref
        and supervisor["direct_worker_wait_ref"] == ready["direct_wait_ref"],
        "Real compact supervisor direct OS0")
    c.require(outer["schema"] == "ROOT_batch_unit_actual_OS_terminal_v1"
        and outer["root_authorizer"] == "root_agent"
        and outer["actual_supervisor_OS_terminal_consumed"] is True
        and type(outer["actual_exit_code"]) is int and outer["actual_exit_code"] == 0
        and outer["UI_tool_terminal_claimed"] is False
        and outer["actual_worker_wait_ref"] == ready["direct_wait_ref"]
        and outer["actual_supervisor_wait_ref"] == ready["actual_supervisor_wait_ref"]
        and outer["input_ref"] == inp_ref and outer["completed_ref"] == done_ref
        and outer["all_owned_processes_and_groups_drained"] is True,
        "Genuine controller-consumed compact supervisor terminal")
    rev = c.read(short_ref_v1(ready["permanent_revocation_ref"]))
    c.require(rev["input_ref"] == inp_ref and rev["completed_ref"] == done_ref
        and rev["direct_wait_ref"] == ready["direct_wait_ref"]
        and rev["actual_outer_terminal_ref"] == ready["actual_outer_terminal_ref"]
        and rev["all_original_launch_resume_retry_write_routes_permanently_revoked"] is True,
        "All compact producer write/retry routes permanently revoked")
    closed_v1(ready["focal_preparation_close_accepted_ref"], parent, input_ref=inp_ref, stage="compact_focals")
    inferred = c.read(short_ref_v1(ready["inference_ready_ref"]))
    c.require(inferred["schema"] == "TreeLoom_inference_only_parent_ready_v1"
        and inferred["parent_id"] == parent and inferred["observation_ref"] == observed_ref
        and inferred["roster_ref"] == observed["roster_ref"]
        and inferred["class_order"] == list(focal.CLASSES)
        and inferred["sequence_length_bp"] == 30000000
        and inferred["inference_close_accepted_ref"] == ready["inference_close_accepted_ref"]
        and inferred["simulation_completion_ref"] == ready["simulation_completion_ref"]
        and inferred["query_truth_opened"] is False and inferred["output_frozen"] is True,
        "Original complete inferred-observation frontier remains bound")
    inferred_input = c.read(short_ref_v1(inferred["input_ref"]))
    inferred_done = c.read(short_ref_v1(inferred["completed_ref"]))
    c.require(inferred_input["parent_id"] == parent and inferred_input["stage"] == "infer"
        and inferred_done["status"] == "completed_current_C3_inference"
        and inferred_done["observation_ref"] == observed_ref,
        "Actual completed whole-observation producer is separate from compact focal completion")
    closed_v1(ready["inference_close_accepted_ref"], parent, input_ref=inferred["input_ref"], stage="infer")
    row = dict(parent_id=parent, observation_ref=observed_ref, native_preparation_ref=done_ref,
        focal_registry_ref=focals_ref, focal_identity_sha256=identity, focal_preparation_kind=COMPACT,
        producer_ready_ref=ready_ref, observation_only=True,
        all_native_trees_nodes_sites_samples_retained=True,
        inference_close_accepted_ref=ready["inference_close_accepted_ref"],
        focal_preparation_close_accepted_ref=ready["focal_preparation_close_accepted_ref"])
    return row, ready["simulation_completion_ref"]


def native000_v1(parameters):
    parent = PARENTS[0]
    metadata_ref = short_ref_v1(parameters["TRAIN000_preparation_metadata_ref"])
    metadata = c.read(metadata_ref)
    observed_ref = short_ref_v1(metadata["observation_ref"])
    prepared_ref = short_ref_v1(metadata["native_preparation_ref"])
    focals_ref = short_ref_v1(metadata["focal_registry_ref"])
    observation_v1(observed_ref, parent)
    prepared = c.read(prepared_ref)
    c.require(metadata["parent_id"] == parent and metadata["observation_only"] is True
        and metadata["all_native_trees_nodes_sites_samples_retained"] is True
        and prepared["status"] == "completed_native_C3_path_preparation"
        and prepared["parent_id"] == parent and prepared["observation_ref"] == observed_ref
        and prepared["focals_ref"] == focals_ref and prepared["query_focal_count"] == 10000
        and prepared["genealogy_kind"] == focal.GENEALOGY
        and prepared["class_order"] == list(focal.CLASSES)
        and prepared["query_ancestry_truth_opened"] is False,
        "Reuse actual original TRAIN000 full-native preparation and unchanged focal identity")
    identity = metadata["focal_identity_sha256"]
    c.require(type(identity) is str and re.fullmatch("[0-9a-f]{64}", identity) is not None,
        "Actual original TRAIN000 focal identity metadata")
    inference_close = short_ref_v1(parameters["TRAIN000_inference_close_accepted_ref"])
    native_close = short_ref_v1(parameters["TRAIN000_native_close_accepted_ref"])
    closed_v1(inference_close, parent)
    closed_v1(native_close, parent)
    return dict(parent_id=parent, observation_ref=observed_ref, native_preparation_ref=prepared_ref,
        focal_registry_ref=focals_ref, focal_identity_sha256=identity,
        focal_preparation_kind=NATIVE, producer_ready_ref=metadata_ref,
        observation_only=True, all_native_trees_nodes_sites_samples_retained=True,
        inference_close_accepted_ref=inference_close,
        focal_preparation_close_accepted_ref=native_close)


def parameters_v1(record, *, require_active):
    p = c.read(short_ref_v1(record))
    start = datetime.fromisoformat(p["created_at_utc"])
    end = datetime.fromisoformat(p["expires_at_utc"])
    now = datetime.now(timezone.utc)
    c.require(p["schema"] == "ROOT_TreeLoom_compact_all20_build_parameters_v1"
        and p["root_authorizer"] == "root_agent" and p["active"] is True
        and p["build_only"] is True and p["admission_or_execution_authorized"] is False
        and start.tzinfo is not None and end.tzinfo is not None
        and 0 < (end-start).total_seconds() <= 900 and start <= now,
        "Finite explicit ROOT metadata-only build authority")
    if require_active:
        c.require(now < end, "Fresh unexpired build-only directive")
    own_ref = c.ref(MANIFEST)
    own = c.read(own_ref)
    c.require(p["source_manifest_ref"] == own_ref
        and p["compact_preparation_manifest_ref"] == own["compact_CPU_manifest_ref"],
        "Exact sealed adapter and compact CPU producer")
    source_pass_v1(p["source_review_ref"], own_ref)
    source_pass_v1(p["compact_preparation_source_review_ref"], p["compact_preparation_manifest_ref"])
    for path, sha in own["source_sha256"].items():
        c.raw(dict(path=path, sha256=sha))
    c.require(list(p["compact_parent_ready_refs"]) == PARENTS[1:],
        "All nineteen actual compact frontiers, exact original order")
    return p


def build_values(parameters_ref, *, require_active=True):
    """Authenticate actual closed producer metadata; never read private payload."""
    p = parameters_v1(parameters_ref, require_active=require_active)
    parents = {PARENTS[0]: native000_v1(p)}
    simulation_refs = {PARENTS[0]: short_ref_v1(p["TRAIN000_simulation_completion_ref"])}
    for parent in PARENTS[1:]:
        row, simulation_ref = compact_parent_v1(p["compact_parent_ready_refs"][parent], parent, p)
        parents[parent] = row
        simulation_refs[parent] = short_ref_v1(simulation_ref)
    handles = {parent: dict(parent_id=parent, truth_ref=simulation_v1(simulation_refs[parent], parent))
        for parent in PARENTS}
    readiness = dict(schema=READINESS_SCHEMA, root_authorizer="root_agent",
        source_manifest_ref=p["source_manifest_ref"], source_review_ref=p["source_review_ref"],
        parameters_ref=parameters_ref, TRAIN_parent_ids=PARENTS, parents=parents,
        class_order=list(focal.CLASSES), sequence_length_bp=30000000,
        all20_complete=True, query_truth_opened=False,
        all_native_trees_nodes_sites_samples_retained=True,
        all20_complete_observation_metadata_authenticated=True,
        complete_observation_barrier_separate_from_compact_focal_stage=True,
        original_barrier_semantics="all20_metadata_before_this_full_training_truth;whole_parent_encode_before_own_target_join",
        compact_native_corpus_claimed=False, original_TRAIN000_native_preparation_reused=True,
        automatic_retry=False)
    truth = dict(schema="TreeLoom_all20_training_truth_handles_v1", root_authorizer="root_agent",
        TRAIN_parent_ids=PARENTS, parents=handles, class_order=list(focal.CLASSES),
        query_truth_opened=False, simulation_completion_refs=simulation_refs, truth_payloads_read=False)
    return p, readiness, truth


def verify_values_v1(readiness_ref, readiness, truth_ref, truth):
    """Reauthenticate immutable producer joins when full600 consumes the barrier."""
    p, expected_ready, expected_truth = build_values(readiness["parameters_ref"], require_active=False)
    c.require(readiness == expected_ready and truth == dict(expected_truth, readiness_ref=readiness_ref),
        "Exact actual all20 compact-aware readiness and original truth-handle joins")
    short_ref_v1(truth_ref)
    return p


def _write_v1(directory, name, value, guard, storage):
    data = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    c.require(len(data) <= MAX_JSON, "Each all20 metadata JSON fits original64KiB")
    used = 4096
    existing = list(directory.iterdir())
    c.require(len(existing) < 3, "At most three flat all20 metadata files")
    for entry in existing:
        s = entry.lstat()
        c.require(stat.S_ISREG(s.st_mode) and not entry.is_symlink() and s.st_nlink == 1
            and entry.suffix == ".json" and s.st_size <= MAX_JSON, "Only own bounded all20 JSON")
        used += s.st_size + 4096
    c.require(used + len(data) + 4096 <= MAX_TOTAL, "All all20 metadata inside declared256KiB")
    guard.budget_check_v9(storage, boundary=True, required_write_bytes=len(data) + 8192)
    fd = os.open(directory/name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
    with os.fdopen(fd, "wb") as stream:
        c.require(stream.write(data) == len(data), "Complete exclusive all20 JSON write")
        stream.flush()
        os.fsync(stream.fileno())
    return c.ref(str((directory/name).relative_to(c.ROOT)))


def build(parameters_ref):
    p, readiness, truth = build_values(parameters_ref)
    admin = c.read(short_ref_v1(p["compact_admin_input_ref"]))
    storage = c.read(short_ref_v1(p["admin_storage_ref"]))
    authority = c.read(short_ref_v1(admin["authority_ref"]))
    c.active(authority)
    c.require(admin["storage"] == storage and admin["admin_storage_ref"] == p["admin_storage_ref"]
        and admin["source_manifest_ref"] == p["compact_preparation_manifest_ref"]
        and authority["all20_adapter_source_ref"] == c.ref(SOURCE)
        and authority["all20_adapter_manifest_ref"] == p["source_manifest_ref"]
        and authority["all20_adapter_output_cap_bytes"] == MAX_TOTAL
        and storage["active"] is True and storage["consumer_uses_v9_guard"] is True
        and storage["reservation"]["reserved_bytes"] == storage["reserved_new_bytes"] == 33554432
        and storage["output_directory"] == authority["run_directory"],
        "Exact open charged compact admin and explicitly bound all20 adapter")
    output = c.local(p["output_directory"])
    c.require(str(output.relative_to(c.ROOT)) == authority["run_directory"] + "/all20_ready_control"
        and authority["all20_adapter_output_directory"] == p["output_directory"]
        and not output.exists(), "Fresh dedicated all20 metadata child in charged admin output")
    guard = c.load(c.GUARD, "_TreeLoom_compact_all20_current_guard")
    c.require(guard.load_selected_guard_v1(SELECTOR) is guard
        and guard.MODE == "Census_observation_two_scope_retention_v1",
        "Existing exact current Census guard; no new quota framework")
    guard.budget_check_v9(storage, boundary=True, required_write_bytes=MAX_TOTAL)
    output.mkdir(mode=0o700)
    rr = _write_v1(output, "all20_training_readiness.json", readiness, guard, storage)
    tr = _write_v1(output, "all20_training_truth_handles.json", dict(truth, readiness_ref=rr), guard, storage)
    cr = _write_v1(output, "completed.json", dict(
        schema="ROOT_TreeLoom_compact_all20_build_completion_v1",
        status="ALL20_COMPLETE_OBSERVATIONS_AND_ORIGINAL_FOCALS_READY",
        readiness_ref=rr, truth_handles_ref=tr, parameters_ref=parameters_ref,
        query_truth_opened=False, model_executed=False, admission=False, execution=False,
        full600_still_requires_original_qualification_owner_measured_window_and_actual_admission=True), guard, storage)
    fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return dict(status="ALL20_COMPACT_AWARE_METADATA_READY", readiness_ref=rr,
        truth_handles_ref=tr, completed_ref=cr, full600_admitted=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-ref")
    args = parser.parse_args(argv)
    if not args.build_ref:
        print(json.dumps(dict(status="SOURCE_ONLY_INACTIVE", build=False, admission=False, execution=False)))
        return 0
    c.require(sys.dont_write_bytecode and sys.flags.no_user_site and sys.flags.optimize == 0
        and Path.cwd() == c.ROOT and not any(n in sys.modules for n in ("torch", "torch_geometric")),
        "Fresh stdlib-only ROOT metadata builder")
    print(json.dumps(build(json.loads(args.build_ref)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
