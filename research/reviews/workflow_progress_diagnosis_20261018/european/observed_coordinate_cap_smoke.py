"""Observed-only bounded JSON smoke for the failed European inference attempts.

No inference stage, native command, model, truth array or training is executed.
"""
from __future__ import annotations
import gc
import hashlib
import json
from pathlib import Path
import resource
import time
from datetime import datetime,timezone

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
OUT=ROOT/"research/reviews/workflow_progress_diagnosis_20261018/european"
OLD_CAP=128*2**20
NEW_CAP=256*2**20


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda:stream.read(2**20),b""):
            h.update(part)
    return h.hexdigest()


def path(value):
    p=Path(value)
    p=p if p.is_absolute() else ROOT/p
    if (not p.is_relative_to(ROOT) or p.resolve()!=p
            or any("truth" in part.lower() or "oracle" in part.lower() for part in p.parts)):
        raise ValueError("Only canonical repository metadata/observed paths are allowed")
    return p


def ref(value):
    p=path(value)
    return dict(path=str(p.relative_to(ROOT)),sha256=sha(p))


def read_ref(record):
    p=path(record["path"])
    if p.stat().st_size>4*2**20 or sha(p)!=record["sha256"]:
        raise ValueError("Exact bounded operational metadata reference required")
    return json.loads(p.read_text())


def write(name,value):
    p=OUT/name
    with p.open("x") as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write("\n")
    return ref(p)


def run():
    from arggnn.benchmarking import european_native_relate as native
    state_ref=ref(".runtime/benchmarking/workflow_20261018_v2/state.json")
    state=read_ref(state_ref)
    plan=read_ref(state["plan_ref"])
    tasks={t["id"]:t for t in plan["tasks"]}
    recipe_ref=ref("configs/benchmarks/recipes/ancestralpaths_european.json")
    recipe=read_ref(recipe_ref)
    if recipe["parameters"]["length_bp"]!=200_000_000:
        raise ValueError("Original full200Mb recipe required")
    failures=[(key,value) for key,value in state["tasks"].items()
        if value["status"]=="failed" and (key.startswith("AP__") or key.startswith("C4__"))
        and key.endswith(("__ap_infer","__validation_infer"))]
    if len(failures)!=9:
        raise ValueError("Exactly the original nine failed inference attempts required")
    reports=[]
    for key,entry in sorted(failures):
        started=time.monotonic()
        job=read_ref(entry["job_ref"])
        arguments=job["call"]["spec"]["arguments"]
        if arguments["max_metadata_bytes"]!=OLD_CAP:
            raise ValueError("Original failed read budget differs")
        task=tasks[key]
        parent=arguments["parent"]
        generation_ids=[dep for dep in task["dependencies"]
                        if dep.endswith(("__ap_generate","__validation_generate"))]
        if len(generation_ids)!=1:
            raise ValueError("Exactly one same-parent completed generation dependency required")
        generation_id=generation_ids[0]
        generation_entry=state["tasks"][generation_id]
        if (generation_entry["status"]!="succeeded"
                or generation_entry["actual_controller_wait"]!=0
                or generation_entry["owned_process_group_absent"] is not True):
            raise ValueError("Actual successful original generation required")
        generation=read_ref(generation_entry["result_ref"])["value"]
        generation_job=read_ref(generation_entry["job_ref"])
        generator_parent=generation_job["call"]["spec"]["arguments"]["parent"]
        if parent!=generator_parent or generation["parent_id"]!=parent["parent_id"]:
            raise ValueError("Simulation/native seeds and biological parent identity changed")
        expected=arguments["expected_input_refs"]
        if expected!=generation["observed_input_refs"]:
            raise ValueError("Original observed input hash bindings changed")
        coordinate_ref=expected["site_coordinate_manifest.json"]
        coordinate_path=path(coordinate_ref["path"])
        if coordinate_path!=path(arguments["observed_directory"])/"site_coordinate_manifest.json":
            raise ValueError("Exact declared observed coordinate path required")
        before=coordinate_path.stat()
        if sha(coordinate_path)!=coordinate_ref["sha256"]:
            raise ValueError("Original generated coordinate bytes changed")
        try:
            native._json(coordinate_path,OLD_CAP)
        except ValueError as error:
            if str(error)!="Observed metadata exceeds its explicit read cap":
                raise
            original_cap_rejected=True
        else:
            raise ValueError("Expected existing128MiB rejection was not reproduced")
        # Only this one full coordinate object is held in memory.
        coordinate=native._json(coordinate_path,NEW_CAP)
        if (set(coordinate)!={"build","contig_id","length_bp","rows"}
                or coordinate["build"]!="synthetic" or coordinate["contig_id"]!="22"
                or coordinate["length_bp"]!=200_000_000
                or not isinstance(coordinate["rows"],list)):
            raise ValueError("Exact original observed coordinate schema required")
        previous=0
        for index,row in enumerate(coordinate["rows"]):
            pos=row.get("VCF_position_1based")
            if (set(row)!={"VCF_position_1based","site_id","tree_position_0based"}
                    or type(pos) is not int or not previous<pos<=200_000_000
                    or row["tree_position_0based"]!=pos-1 or row["site_id"]!=index):
                raise ValueError("Original reader coordinate predicate failed")
            previous=pos
        count=len(coordinate["rows"])
        observation=read_ref(expected["observation_manifest.json"])
        if count==0 or count!=observation["sites"]:
            raise ValueError("All original observed SNP rows must be retained")
        first=coordinate["rows"][0]["VCF_position_1based"]
        last=coordinate["rows"][-1]["VCF_position_1based"]
        del coordinate
        gc.collect()
        after=coordinate_path.stat()
        if ((before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns)
                or sha(coordinate_path)!=coordinate_ref["sha256"]):
            raise ValueError("Observed bytes changed during metadata smoke")
        stage=path(task["output_directory"])
        wait_ref=ref(stage/"actual_wait.json")
        wait=read_ref(wait_ref)
        stage_log_ref=ref(stage/"stage.log")
        if "ValueError: Observed metadata exceeds its explicit read cap" not in (stage/"stage.log").read_text():
            raise ValueError("Expected original failure traceback must be preserved")
        if (wait["actual_direct_returncode"]!=1 or wait["actual_direct_wait"] is not True
                or wait["owned_process_group_absent"] is not True
                or wait["supervisor_error"] is not None
                or (stage/"native").exists()):
            raise ValueError("Original failure/native-start evidence differs")
        reports.append(dict(parent_id=parent["parent_id"],split=parent["split"],
            failed_task_id=key,failed_job_ref=entry["job_ref"],
            original_seeds=parent["seeds"],original_recipe_ref=recipe_ref,
            generation_task_id=generation_id,generation_result_ref=generation_entry["result_ref"],
            observed_input_refs=expected,coordinate_ref=coordinate_ref,size_bytes=before.st_size,
            expected_sites=count,first_VCF_position_1based=first,last_VCF_position_1based=last,
            old_cap_bytes=OLD_CAP,proposed_cap_bytes=NEW_CAP,
            old_cap_rejection_reproduced=original_cap_rejected,
            original_json_loader_parse_succeeded=True,exact_schema_passed=True,
            sequential_all_row_predicate_passed=True,coordinate_SHA_before_after_unchanged=True,
            recipe_seeds_all_observed_hash_bindings_unchanged=True,
            failed_actual_wait_ref=wait_ref,failed_stage_log_ref=stage_log_ref,
            failed_files_preserved=True,native_inference_started=False,
            elapsed_seconds=time.monotonic()-started))
        print(json.dumps(dict(parent_id=parent["parent_id"],sites=count,
            size_bytes=before.st_size,proposed_cap_parse="passed")),flush=True)
    result=dict(schema="European_observed_coordinate_cap_smoke_v1",
        completed_utc=datetime.now(timezone.utc).isoformat(),
        source_ref=ref(__file__),reader_source_ref=ref(native.__file__),
        state_snapshot_ref=state_ref,plan_ref=state["plan_ref"],recipe_ref=recipe_ref,
        parents=reports,parents_passed=len(reports),
        peak_process_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        largest_manifest_bytes=max(r["size_bytes"] for r in reports),
        original_cap_bytes=OLD_CAP,proposed_cap_bytes=NEW_CAP,
        one_coordinate_object_at_a_time=True,
        completed_generation_products_reused_without_resimulation=True,
        truth_arrays_opened=False,models_executed=False,
        native_commands_started=0,scientific_outputs_created=0,
        source_or_live_plan_changed=False,
        scope="Only coordinate JSON parsing and existing operational metadata, no inference qualification")
    smoke_ref=write("observed_coordinate_cap_smoke.json",result)
    print(json.dumps(dict(smoke_ref=smoke_ref,parents_passed=len(reports))),flush=True)
    return result,smoke_ref


if __name__=="__main__":
    run()
