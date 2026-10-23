"""Expand actual current cohort products into ready specs and dependency-resolved tasks."""
import argparse
import json
from . import protocol as p

def expand(state_path, plan_path, output_directory):
    state_bytes = p.path(state_path).read_bytes()
    state = json.loads(state_bytes)
    plan = json.loads(p.path(plan_path).read_text())
    planned = {x["id"]: x for x in plan["tasks"]}
    out = p.path(output_directory)
    out.mkdir(parents=True, exist_ok=True)
    tasks, reusable, controls = [], [], []
    state_tasks = state["tasks"]
    ready_count = 0
    for split, count in (("validation", 5), ("test", 10)):
        for index in range(count):
            parent = "argformer_american_" + split + "_" + str(index).zfill(3)
            prefix = "AM_" + parent
            prep_job = state_tasks[prefix + "_argformer_prepare"]
            flare_job = state_tasks[prefix + "_flare_native"]
            focal_job = state_tasks[prefix + "_focals"]
            if flare_job["status"] != "succeeded" or focal_job["status"] != "succeeded":
                raise ValueError("Completed actual native FLARE and shared focal preparation required")
            focal_ref = p.read(focal_job["result_ref"])["value"]
            focal = p.read(focal_ref)
            flare_ref = p.read(flare_job["result_ref"])["value"]
            flare = p.read(flare_ref)
            converted = p.read(flare["conversion_ref"])
            if (flare["focal_registry_ref"] != focal["focals_ref"]
                    or converted["observation_ref"] != focal["observation_ref"]
                    or focal["parent_id"] != parent
                    or flare["native_AN1_AN2_calls_used_unchanged"] is not True
                    or flare["query_ancestry_truth_opened"] is not False):
                raise ValueError("Same actual observed parent, focal identities and untouched calls required")
            base = "experiments/results/paper_american_20261024/" + parent
            prep_ref, prep_path, prep_dependency = None, None, []
            historical_prep_ref = p.read(prep_job["result_ref"])["value"] if prep_job["status"] == "succeeded" else None
            historical_prep = p.read(historical_prep_ref) if historical_prep_ref else None
            payloads_present = historical_prep is not None and all(p.path(r["path"]).is_file() for r in
                [*historical_prep["native_dataset_refs"].values(), *historical_prep["full_path_id_order_refs"].values(), historical_prep["vocab_ref"]])
            if payloads_present:
                prep_ref = historical_prep_ref
                prep = historical_prep
                if prep["focals_ref"] != focal["focals_ref"] or prep["observation_ref"] != focal["observation_ref"]:
                    raise ValueError("Native path preparation belongs to another cohort")
                ready_count += 1
            else:
                from arggnn.benchmarking.american_competitors import build_argformer_spec
                original = planned[prefix + "_argformer_prepare"]
                arguments = dict(original.get("factory_arguments", dict(stage="prepare",
                    limits=dict(max_output_bytes=20*2**30, max_rss_bytes=256*2**30, max_seconds=86400),
                    path_limits=dict(max_path_occurrences=250000000, max_path_tokens=500000000,
                                     max_raw_path_nodes=1024, max_unique_paths=32000000, max_vocabulary_nodes=250000))))
                arguments["focal_preparation_ref"] = focal_ref
                arguments["inference_ref"] = focal["input_ref"]
                arguments["output_directory"] = base + "/prepare"
                actual_spec = build_argformer_spec(**arguments)
                prep_id = "paper_american_" + parent + "_prepare"
                tasks.append(dict(id=prep_id, dependencies=[], resource="cpu",
                                  callable_module="arggnn.benchmarking.american_competitors",
                                  callable_name="run_argformer", scientific_spec=actual_spec,
                                  python_executable=".runtime/env/bin/python",
                                  output_directory=base + "/prepare",
                                  expected_output_bytes=arguments["limits"]["max_output_bytes"],
                                  max_rss_bytes=arguments["limits"]["max_rss_bytes"],
                                  max_seconds=arguments["limits"]["max_seconds"], CPU_threads=8))
                prep_path = base + "/prepare/preparation_completed.json"
                prep_dependency = [prep_id]
            reusable.append(dict(parent_id=parent, preparation_ref=prep_ref,
                                 preparation_completion_path=prep_path,
                                 historical_preparation_ref=historical_prep_ref,
                                 historical_raw_payloads_present=payloads_present,
                                 focal_registry_ref=focal["focals_ref"], observation_ref=focal["observation_ref"],
                                 flare_prediction_seal_ref=flare_ref, flare_result_ref=flare_job["result_ref"]))
            previous = None
            for stage in ("mlm", "contrastive", "predict"):
                job_id = "paper_american_" + parent + "_" + stage
                result_dir = base + "/" + stage
                request = dict(stage=stage, variant="paper_aligned", output_directory=result_dir, num_workers=8)
                if prep_ref is not None:
                    request["preparation_ref"] = prep_ref
                else:
                    request["preparation_completion_path"] = prep_path
                if previous is not None:
                    request["checkpoint_completion_path"] = previous["output_directory"] + "/completed.json"
                task = dict(id=job_id, dependencies=prep_dependency if previous is None else [previous["id"]],
                            resource="gpu", callable_module="arggnn.benchmarking.paper_american.worker",
                            callable_name="run_task", scientific_spec=request,
                            python_executable=str(p.NATIVE_PYTHON.relative_to(p.ROOT)),
                            output_directory=result_dir,
                            expected_output_bytes=8 * 2**30 if stage != "predict" else 32 * 2**20,
                            max_rss_bytes=128 * 2**30, gpu_memory_policy="full_device",
                            max_seconds=7 * 86400 if stage == "mlm" else 2 * 86400,
                            CPU_threads=8, numerical_batch_size=512 if stage=="mlm" else 4096 if stage=="contrastive" else 256,
                            data_sampling_or_epoch_reduction=False)
                if stage == "mlm" and prep_ref is not None:
                    ready = p.build_spec(stage=stage, preparation_ref=prep_ref, output_directory=result_dir)
                    task["ready_spec_ref"] = p.save(ready, out / (job_id + ".json"))
                tasks.append(task)
                previous = task
            for suffix in ("argformer_mlm", "argformer_contrastive", "argformer_embeddings", "argformer_retrieval"):
                job = state_tasks.get(prefix + "_" + suffix)
                if job:
                    controls.append(dict(parent_id=parent, stage=suffix, status=job["status"],
                                         result_ref=job.get("result_ref"),
                                         reusable_as_corrected_initializer=False,
                                         reusable_as_named_release_control=job["status"]=="succeeded"))
    snapshot = out / "source_state_snapshot.json"
    with snapshot.open("xb") as stream:
        stream.write(state_bytes)
    result = dict(schema="paper_american_executable_DAG_v1", source_state_snapshot_ref=p.ref(snapshot),
                  source_plan_ref=p.ref(plan_path), methods=["ARGformer_paper_oriented_extant", "FLARE_native_reused"],
                  parents=reusable, reused_native_FLARE_count=15, ready_corrected_MLM_specs=ready_count,
                  new_native_path_preparation_tasks=15-ready_count, tasks=tasks,
                  release_control_records=controls, truth_access=False,
                  output_reference_embeddings_stored=False, corrected_checkpoints_reused=False,
                  root_owns_scheduling_and_storage=True)
    return p.save(result, out / "task_dag.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--state", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args()
    print(json.dumps(expand(args.state, args.plan, args.output_directory)))
