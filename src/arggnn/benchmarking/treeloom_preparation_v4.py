"""Unpinned v4 American TreeLoom CPU qualification and compact-RAM fit factories."""
from __future__ import annotations
import copy
import json
from pathlib import Path
from . import contracts as c

MODULE="arggnn.benchmarking.treeloom_preparation_v4"
PARENT="argformer_american_train_000"
GIB=2**30
REVIEW=Path("research/reviews/workflow_progress_diagnosis_20261018/treeloom")
QUALIFICATION_SCHEMA="treeloom_full_AMtrain000_CPU_qualification_v4"
FIT_SCHEMA="treeloom_full20_compact_RAM_fit_v4"


def _need(ok,message):
    if not ok:raise c.ContractError(message)


def candidate_source_refs():
    names=("indices_v1.py","candidate_v1.py","cache_v1.py","resident_pool_v2.py",
           "prepare_integration_v2.py","qualify_parent_v2.py","storage_bound_v2.json")
    return [c.file_ref(__file__),*[c.file_ref(REVIEW/name) for name in names]]


def _source_refs():
    from . import treeloom_training as original
    return list({r["path"]:r for r in [*original.requirements()["source_refs"],
        c.file_ref("src/arggnn/benchmarking/american_workflow.py"),*candidate_source_refs()]}.values())


def _descriptor(function,spec):
    return dict(module=MODULE,function=function,spec=spec,python=".runtime/env/bin/python",
                source_refs=_source_refs())


def _parent_bound():
    bounds=json.loads(c.contained_path(REVIEW/"storage_bound_v2.json").read_text())
    return next(row["complete_compact_cache_upper_bound_bytes"]+160000
                for row in bounds["parents"] if row["parent_id"]==PARENT)


def qualification_factory(*,preparation_ref,generation_ref,output_directory,max_rss_bytes,
                          max_seconds=7200,max_output_bytes=16*2**20):
    from . import american_workflow as american
    from . import treeloom_training as original
    _need(max_rss_bytes==128*GIB and 0<max_seconds<=7200
          and 2*2**20<=max_output_bytes<=16*2**20,
          "One full-parent CPU qualification has finite128GiB/7200s/16MiB caps")
    refs=american.observed_binding(preparation_ref)
    ready=c.read_ref(generation_ref);observed=c.read_ref(refs["observation_ref"])
    _need(ready["parent_id"]==observed["parent_id"]==PARENT
          and ready["split"]==observed["split"]=="train"
          and ready["observation_ref"]==refs["observation_ref"]
          and observed["query_ancestry_truth_in_inputs"] is False,
          "Same existing AMtrain000 public observed generation/focal bindings required")
    output=c.contained_path(output_directory)
    _need(output.is_relative_to(c.ROOT/"experiments/results") and not output.exists(),
          "Fresh dedicated qualification output scope")
    spec=dict(schema=QUALIFICATION_SCHEMA,parent_id=PARENT,
        observed_input_refs=copy.deepcopy(refs),preparation_limits=copy.deepcopy(original.LIMITS),
        parent_registry_ref=c.file_ref(american.REGISTRY),
        output_directory=str(output.relative_to(c.ROOT)),max_rss_bytes=max_rss_bytes,
        max_seconds=max_seconds,max_output_bytes=max_output_bytes-2**20,
        max_parent_compact_bytes=_parent_bound(),candidate_source_refs=candidate_source_refs(),
        CPU_only=True,truth_or_model_allowed=False,disk_cache_bytes=0)
    return _descriptor("run_qualification",spec)


def run_qualification(spec,*,resource_check):
    from research.reviews.workflow_progress_diagnosis_20261018.treeloom.qualify_parent_v2 import qualify
    _need(spec["schema"]==QUALIFICATION_SCHEMA and spec["parent_id"]==PARENT
          and spec["candidate_source_refs"]==candidate_source_refs()
          and spec["CPU_only"] is True and spec["truth_or_model_allowed"] is False
          and spec["disk_cache_bytes"]==0,"Exact CPU-only qualification/source closure")
    result=qualify(spec,resource_check=resource_check)
    ref=c.file_ref(c.contained_path(spec["output_directory"])/"qualification.json")
    _need(c.read_ref(ref)==result,"Exact saved CPU qualification receipt")
    _require_pass(ref,spec["observed_input_refs"])
    return ref


def _require_pass(qualification_ref,observed_input_refs):
    result=c.read_ref(qualification_ref)
    _need(result.get("schema")=="one_full_AMtrain000_exact_compact_CPU_qualification_v2"
          and result.get("status")=="passed_complete_parent_exact_preparation_and_two_reuses"
          and result.get("parent_id")==PARENT
          and result.get("observed_input_refs")==observed_input_refs
          and result.get("candidate_source_refs")==candidate_source_refs()
          and result.get("private_truth_opened") is False
          and result.get("models_executed") is False and result.get("CUDA_called") is False
          and result.get("scientific_outputs_created")==0 and result.get("disk_cache_bytes")==0
          and result.get("full_parent_preparation_count")==1 and result.get("exact_restores")==2
          and result.get("all_original_tensor_and_metadata_signatures_equal") is True
          and result.get("original_complete_parent_retained") is True
          and 0<result.get("compact_charged_bytes",0)<=_parent_bound()
          and 0<result.get("observed_peak_rss_bytes",0)<=128*GIB
          and 0<result.get("elapsed_seconds",0)<=7200,
          "Actual complete-parent bounded CPU preparation/reuse PASS must precede fitting")
    return result


def fit_factory(*,preparation_refs,generation_refs,output_directory,max_rss_bytes,qualification_ref):
    from . import american_workflow as american
    from research.reviews.workflow_progress_diagnosis_20261018.treeloom import prepare_integration_v2 as integration
    registry=c.file_ref(american.REGISTRY)
    parents={row["parent_id"] for row in c.read_ref(registry)["parents"] if row["split"]=="train"}
    _need(set(preparation_refs)==set(generation_refs)==parents and len(parents)==20
          and max_rss_bytes==512*GIB,"All20 exact TRAIN parents and original512GiB fit cap required")
    observed=american.observed_binding(preparation_refs[PARENT])
    _require_pass(qualification_ref,observed)
    train={p:american.train_binding(p,preparation_refs[p],generation_refs[p]) for p in sorted(parents)}
    base=integration.build_spec(parent_registry_ref=registry,output_directory=output_directory,
                               train_inputs=train,max_rss_bytes=max_rss_bytes)
    spec=dict(schema=FIT_SCHEMA,training_spec=base,qualification_ref=copy.deepcopy(qualification_ref),
              candidate_source_refs=candidate_source_refs(),
              gpu_worker_and_actual_wait_owned_by_ROOT=True,disk_cache_bytes=0)
    return _descriptor("run_fit",spec)


def run_fit(spec,*,resource_check):
    from research.reviews.workflow_progress_diagnosis_20261018.treeloom import prepare_integration_v2 as integration
    from . import treeloom_fresh as fresh
    expected=dict(schema=FIT_SCHEMA,training_spec=spec["training_spec"],
                  qualification_ref=spec["qualification_ref"],candidate_source_refs=candidate_source_refs(),
                  gpu_worker_and_actual_wait_owned_by_ROOT=True,disk_cache_bytes=0)
    _need(spec==expected,"Exact reviewed compact-RAM full fit/source closure")
    _require_pass(spec["qualification_ref"],
                  {k:spec["training_spec"]["train_inputs"][PARENT][k] for k in fresh.INPUT_REFS})
    result=integration.run_training(spec["training_spec"],resource_check=resource_check)
    completion_ref=c.file_ref(c.contained_path(spec["training_spec"]["output_directory"])/"training_completed.json")
    _need(c.read_ref(completion_ref)==result,"Actual final600-update compact-RAM training receipt differs")
    return dict(result,completion_ref=completion_ref)
