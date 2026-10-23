
"""Metadata-only16-node frozen-American TreeLoom queue builder.

ROOT supplies an immutable durable checkpoint and its exact boundary evidence.
This module opens JSON/source metadata only; it imports no Torch/model/weights.
"""
from __future__ import annotations
import argparse
import copy
import json
from arggnn.benchmarking import contracts as c

MODULE="research.benchmark_rebuild_20261024.american.treeloom_speed.prediction_v2"
LIVE_JOB=".runtime/benchmarking/workflow_20261022_gpu345_v2/jobs/AM_treeloom_fit.json"
ORIGINAL_JOB=".runtime/benchmarking/workflow_20261019_v8/jobs/AM_treeloom_fit.json"
PINS="research/benchmark_rebuild_20261024/american/treeloom_speed/source_pins.json"
OLD_STATE=".runtime/benchmarking/workflow_20261022_gpu345_v2/state.json"
GIB=2**30

def need(ok,message):
    if not ok:raise c.ContractError(message)

def build(*,resume_checkpoint_ref,resume_evidence_ref,
          output_root="experiments/results/paper_american_20261024"):
    need(type(resume_checkpoint_ref) is dict and set(resume_checkpoint_ref)=={"path","sha256"}
         and type(resume_evidence_ref) is dict and set(resume_evidence_ref)=={"path","sha256"},
         "Explicit immutable ROOT capture and matching durable-boundary evidence refs required")
    live=c.read_json(LIVE_JOB)
    original=c.read_json(ORIGINAL_JOB)["call"]["spec"]["training_spec"]
    pins=c.read_json(PINS)
    references=pins["source_refs"]+[c.file_ref(PINS),c.file_ref(__file__)]
    registry_ref=original["parent_registry_ref"]
    root=str(c.contained_path(output_root).relative_to(c.ROOT))
    fit_output=root+"/treeloom_fit"
    fit_id="paper_american_treeloom_fit"
    fit_arguments=dict(parent_registry_ref=registry_ref,train_inputs=copy.deepcopy(original["train_inputs"]),
        output_directory=fit_output,max_rss_bytes=512*GIB,
        preparation_headroom_bytes=128*GIB,training_headroom_bytes=128*GIB,
        observation_cache_bytes=352*GIB,max_truth_uncompressed_bytes=original["max_truth_uncompressed_bytes"],
        resume_checkpoint_ref=copy.deepcopy(resume_checkpoint_ref),resume_evidence_ref=copy.deepcopy(resume_evidence_ref))
    tasks=[dict(id=fit_id,dependencies=[],resource="gpu",callable_module=MODULE,callable_name="run_task",
        scientific_spec=dict(stage="fit",arguments=fit_arguments),
        python_executable=".runtime/env/bin/python",output_directory=fit_output,
        expected_output_bytes=64*2**20,max_rss_bytes=512*GIB,max_seconds=30*86400,
        gpu_memory_policy="full_device",CPU_threads=8,source_refs=references)]
    state=c.read_json(OLD_STATE)
    for split,count in (("validation",5),("test",10)):
        for index in range(count):
            parent="argformer_american_"+split+"_"+str(index).zfill(3)
            row=state["tasks"]["AM_"+parent+"_focals"]
            need(row["status"]=="succeeded","Every actual held-out focal preparation must exist")
            fp_ref=c.read_ref(row["result_ref"])["value"]
            fp=c.read_ref(fp_ref)
            need(fp["parent_id"]==parent and fp["query_ancestry_truth_opened"] is False,
                 "Exact sanitized held-out parent preparation required")
            heldout={parent:dict(observation_ref=fp["observation_ref"],focal_registry_ref=fp["focals_ref"],
                                native_preparation_ref=fp_ref)}
            output=root+"/"+parent+"/treeloom_predict"
            arguments=dict(parent_registry_ref=registry_ref,heldout_inputs=heldout,
                output_directory=output,max_rss_bytes=256*GIB,
                preparation_headroom_bytes=128*GIB,prediction_headroom_bytes=64*GIB,
                observation_cache_bytes=0,max_output_bytes=16*2**20)
            tasks.append(dict(id="paper_american_"+parent+"_treeloom_predict",dependencies=[fit_id],
                resource="gpu",callable_module=MODULE,callable_name="run_task",
                scientific_spec=dict(stage="predict",arguments=arguments,
                    training_completion_path=fit_output+"/training_completed.json"),
                python_executable=".runtime/env/bin/python",output_directory=output,
                expected_output_bytes=16*2**20,max_rss_bytes=256*GIB,max_seconds=2*86400,
                gpu_memory_policy="full_device",CPU_threads=8,source_refs=references))
    need(len(tasks)==16,"Exactly one fit plus fifteen held-out prediction tasks required")
    return dict(schema="paper_american_static_TreeLoom16_DAG_v2",tasks=tasks,
        original_live_job_ref=c.file_ref(LIVE_JOB),original_scientific_job_ref=c.file_ref(ORIGINAL_JOB),
        source_pins_ref=c.file_ref(PINS),resume_checkpoint_ref=resume_checkpoint_ref,resume_evidence_ref=resume_evidence_ref,
        requires_ROOT_native_subdirectory_rewrite=True,metadata_only=True,weights_opened=False,
        all20_train_parents=True,optimizer_budget=600,fixed_final_epoch=30,
        prediction_task_ids=[row["id"] for row in tasks[1:]],jobs_started=0)

def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--resume-checkpoint-ref-json",required=True)
    parser.add_argument("--resume-evidence-ref-json",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    payload=build(resume_checkpoint_ref=json.loads(args.resume_checkpoint_ref_json),
                  resume_evidence_ref=json.loads(args.resume_evidence_ref_json))
    out=c.contained_path(args.output)
    with out.open("x") as stream:json.dump(payload,stream,indent=2);stream.write("\n")
    print(json.dumps(dict(path=str(out.relative_to(c.ROOT)),sha256=c.file_ref(out)["sha256"])))

if __name__=="__main__":main()
