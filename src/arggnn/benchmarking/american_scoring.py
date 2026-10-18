"""Shared American scoring after every15-parent x3-method prediction seals.

Planning reads only public metadata. Execution authenticates every actual10k
prediction and exact focal identity before resolving ANY held-out truth payload.
Truth remains outside all model inputs. Native one-hot transports are hard calls,
not posterior probabilities. There is no launch, training, tuning or model import.
"""
from __future__ import annotations
import copy
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import resource
import statistics
import time
import zipfile

from . import american_focals as focals
from . import american_preparation as generation
from . import treeloom_fresh as fresh

ROOT=generation.ROOT
SCHEMA="american_fresh_all15_common_score_v1"
METHODS=("treeloom","argformer","flare")
IDENTITY=("record_ids","sample_nodes","sample_indices","positions_bp0","parent_id","class_order")
TRUTH_KEYS=("intervals_bp","query_sample_indices","source_classes_0based","parent_id","class_order")
EVALUATOR_REF=dict(path="src/arggnn/american_c3_evaluate_v1.py",sha256="1ada0ab40fe9f198b89a1953bb7c0688e68f1dd44b34dc118600ac25eefc70d3")


def need(ok,message):
    if not ok:raise ValueError(message)


def requirements():
    need(generation._reference(EVALUATOR_REF["path"])==EVALUATOR_REF,"Original common metric/identity source changed")
    refs=[EVALUATOR_REF,generation._reference(__file__),generation._reference(fresh.__file__),
          *focals.requirements()["source_refs"]]
    return dict(source_refs=list({r["path"]:r for r in refs}.values()),
                all45_predictions_before_any_truth=True,methods=list(METHODS),
                hard_call_metrics=["accuracy","precision_per_class","recall_per_class","F1_per_class","macro_F1","confusion"],
                probabilistic_metrics_only_for_genuine_posteriors=True,scientific_payloads_opened=False)


def _read(ref):return focals._read(ref)


def _registry(ref):
    registry=fresh.FreshParentRegistry.from_ref(ref)
    need([len(registry.ids(k)) for k in ("train","validation","test")]==[20,5,10],
         "Exactly20TRAIN/5validation/10TEST independent parents required")
    return registry


def _seal(method,parent,record,registry,fp):
    value=_read(record)
    if method=="treeloom" and value.get("schema")=="treeloom_fresh_heldout_prediction_completion_v1":
        need(value.get("query_ancestry_truth_opened") is False
             and value.get("parent_registry_ref")==registry.reference,"Same-registry TreeLoom completion required")
        record=value["prediction_seal_refs"][parent];value=_read(record)
    elif method=="argformer" and "prediction_seal_ref" in value:
        need(value.get("parent_id")==parent and value.get("query_ancestry_truth_opened") is False,
             "Same-parent native retrieval completion required")
        record=value["prediction_seal_ref"];value=_read(record)
    need(value.get("parent_id")==parent and value.get("query_ancestry_truth_opened") is False
         and value.get("records")==10000 and value.get("class_order")==["EUR","EAS","AFR"]
         and value.get("focal_registry_ref")==fp["focals_ref"],"Complete same-parent10k prediction seal required")
    if method=="treeloom":
        need(value.get("schema")=="treeloom_fresh_prediction_seal_v1"
             and value.get("parent_registry_ref")==registry.reference
             and value.get("split")==registry.split_for(parent)
             and value.get("observation_ref")==fp["observation_ref"]
             and value.get("all_native_trees_nodes_sites_samples_retained") is True
             and value.get("checkpoint_selection")=="fixed_final_epoch30", "Actual complete fixed-final TreeLoom softmax seal required")
        semantics="genuine_softmax_posterior"
    elif method=="argformer":
        need(value.get("status")=="sealed_native_C3_top1"
             and value.get("scores_and_retrieval_indices_modified") is False,"Original native top1 hard-call seal required")
        prep=_read(value["preparation_ref"])
        need(prep.get("parent_id")==parent and prep.get("observation_ref")==fp["observation_ref"]
             and prep.get("focals_ref")==fp["focals_ref"] and prep.get("query_ancestry_truth_opened") is False,
             "Same actual native observed ARG/focal preparation required")
        semantics="native_top1_hard_call_one_hot_transport"
    else:
        need(value.get("status")=="sealed_C3_native_FLARE_calls_and_raw_posteriors"
             and value.get("common_focal_registry_unchanged") is True
             and value.get("native_AN1_AN2_calls_used_unchanged") is True
             and value.get("one_hot_is_native_call_transport") is True
             and value.get("rounded_ANP_renormalized") is False
             and value.get("raw_rounded_ANP_modified") is False,"Original native FLARE hard calls and raw rounded ANP required")
        converted=_read(value["conversion_ref"])
        need(converted.get("parent_id")==parent and converted.get("observation_ref")==fp["observation_ref"]
             and converted.get("focal_registry_ref")==fp["focals_ref"]
             and converted.get("query_ancestry_truth_opened") is False,"Same shared FLARE genotype/focal conversion required")
        semantics="native_AN1_AN2_hard_call_one_hot_transport"
    focals._public_ref(value["prediction_ref"])
    return dict(seal_ref=record,prediction_ref=value["prediction_ref"],semantics=semantics,
                posterior_metrics=method=="treeloom")


def build_spec(*,parent_registry_ref,prediction_refs_by_method,generation_ready_refs_by_parent,
               focal_preparation_refs_by_parent,output_directory,generation_completion_paths_by_parent=None,
               max_truth_uncompressed_bytes=8*2**30,max_public_array_uncompressed_bytes=8*2**20,
               max_output_bytes=16*2**20,max_seconds=86400,max_rss_bytes=16*2**30):
    registry=_registry(parent_registry_ref)
    parents=registry.ids("validation")+registry.ids("test")
    need(type(prediction_refs_by_method) is dict and set(prediction_refs_by_method)==set(METHODS)
         and all(type(v) is dict and set(v)==set(parents) for v in prediction_refs_by_method.values())
         and set(generation_ready_refs_by_parent)==set(focal_preparation_refs_by_parent)==set(parents),
         "All15 evaluation parents x3 methods must seal before any truth access")
    for value in (max_truth_uncompressed_bytes,max_public_array_uncompressed_bytes,max_output_bytes,max_seconds,max_rss_bytes):
        need(type(value) is int and value>0,"Explicit finite scoring truth/array/output/wall/RSS caps required")
    need(max_output_bytes>=4*2**20,"Complete45-result paired score metadata must fit output cap")
    private_paths={}
    for parent in parents:
        gen=_read(generation_ready_refs_by_parent[parent]);fp=_read(focal_preparation_refs_by_parent[parent])
        split={"validation":"val","test":"test"}[registry.split_for(parent)]
        need(gen.get("schema")=="american_fresh_shared_observation_ready_v1"
             and gen.get("parent_id")==parent and gen.get("split")==split
             and gen.get("query_truth_in_training_inputs") is False
             and gen.get("truth_scoring_authorized") is False
             and fp.get("schema")==focals.RESULT_SCHEMA and fp.get("status")=="completed_compact_query_focals"
             and fp.get("parent_id")==parent and fp.get("split")==split
             and fp.get("query_focal_count")==10000 and fp.get("original_record_ids_preserved") is True
             and fp.get("query_ancestry_truth_opened") is False
             and fp.get("class_order")==["EUR","EAS","AFR"]
             and fp.get("genealogy_kind")==focals.observation.GENEALOGY,
             "Actual complete fresh generation/compact focal lineage required")
        inferred=_read(fp["input_ref"])
        observed=_read(fp["observation_ref"])
        need(inferred.get("schema")=="american_fresh_inference_ready_v1"
             and inferred.get("parent_id")==parent and inferred.get("split")==split
             and inferred.get("requires_tsinfer_tsdate") is False
             and inferred.get("input_ref")==generation_ready_refs_by_parent[parent]
             and inferred.get("observation_ref")==fp["observation_ref"]
             and inferred.get("query_truth_in_training_inputs") is False
             and observed.get("parent_id")==parent and observed.get("split")==split
             and observed.get("class_order")==["EUR","EAS","AFR"]
             and observed.get("genealogy_kind")==focals.observation.GENEALOGY
             and observed.get("sites_and_sample_identities_verified") is True
             and observed.get("source_observation_ref")==gen["observation_ref"]
             and observed.get("query_ancestry_truth_in_inputs") is False
             and observed.get("sequence_length_bp")==30000000,"Inferred focal namespace must descend from this exact generated parent")
        for method in METHODS:_seal(method,parent,prediction_refs_by_method[method][parent],registry,fp)
        default=generation._path(generation_ready_refs_by_parent[parent]["path"]).parent/"private/generation_completed.json"
        value=default if generation_completion_paths_by_parent is None else generation._path(generation_completion_paths_by_parent[parent])
        need(value==default,"Only this exact fresh generator's private completion may resolve held-out truth")
        private_paths[parent]=str(value.relative_to(ROOT))
    out=generation._path(output_directory)
    need(out.is_relative_to(ROOT/"experiments/results") and not out.exists(),"Fresh dedicated all15-parent score output required")
    return dict(schema=SCHEMA,parent_registry_ref=copy.deepcopy(parent_registry_ref),
        prediction_refs_by_method=copy.deepcopy(prediction_refs_by_method),
        generation_ready_refs_by_parent=copy.deepcopy(generation_ready_refs_by_parent),
        focal_preparation_refs_by_parent=copy.deepcopy(focal_preparation_refs_by_parent),
        generation_completion_paths_by_parent=private_paths,output_directory=str(out.relative_to(ROOT)),
        max_truth_uncompressed_bytes=max_truth_uncompressed_bytes,max_public_array_uncompressed_bytes=max_public_array_uncompressed_bytes,
        max_output_bytes=max_output_bytes,max_seconds=max_seconds,max_rss_bytes=max_rss_bytes,
        requirements=requirements(),all45_predictions_required_before_truth=True)


def _load_npz(record,cap,*,keys=None,public=True):
    import numpy as np
    if public:focals._public_ref(record)
    path=generation._path(record["path"])
    need(path.is_file() and path.stat().st_size<=cap+2**20 and generation._reference(path)==record,"Exact bounded actual NPZ payload required")
    with zipfile.ZipFile(path) as archive:
        members=archive.infolist();names=[m.filename for m in members]
        need(len(names)==len(set(names)) and all(n.endswith(".npy") and "/" not in n for n in names)
             and sum(m.file_size for m in members)<=cap
             and (keys is None or set(names)=={k+".npy" for k in keys}),"Bounded exact NPZ members; no duplicates or decompression overflow")
    with np.load(path,allow_pickle=False) as archive:return {k:archive[k] for k in archive.files}


def _identity_sha(records):
    import numpy as np
    h=hashlib.sha256()
    for key in IDENTITY:
        a=np.ascontiguousarray(records[key]);h.update(key.encode());h.update(a.dtype.str.encode())
        h.update(json.dumps(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def join_truth(records,truth,*,resource_check):
    """Exact full-domain half-open join; reusable with tiny synthetic arrays."""
    import numpy as np
    need(set(truth)==set(TRUTH_KEYS) and np.array_equal(truth["parent_id"],records["parent_id"])
         and np.array_equal(truth["class_order"],records["class_order"]),"Same-parent/class external truth required")
    intervals,samples,labels=(truth[k] for k in TRUTH_KEYS[:3])
    need(intervals.dtype==np.float64 and intervals.ndim==2 and intervals.shape[1]==2 and len(intervals)>0
         and np.isfinite(intervals).all() and intervals[0,0]==0 and intervals[-1,1]==30000000
         and (intervals[:,1]>intervals[:,0]).all() and np.array_equal(intervals[:-1,1],intervals[1:,0])
         and samples.dtype==np.int64 and np.array_equal(samples,np.arange(600,1600,dtype=np.int64))
         and labels.dtype==np.uint8 and labels.shape==(len(intervals),1000),"Complete30Mb half-open truth and all1000 query columns required")
    for start in range(0,len(labels),32768):
        need((labels[start:start+32768]<3).all(),"Every full-domain ancestry class must be valid")
        resource_check("all_truth_classes_validated")
    pos,query=records["positions_bp0"],records["sample_indices"]
    need(((pos>=0)&(pos<30000000)).all() and ((query>=600)&(query<1600)).all(),"No coordinate extrapolation or reference/query drift")
    rows=np.searchsorted(intervals[:,1],pos,side="right")
    return labels[rows,query-600].astype(np.int64,copy=True)


def score_arrays(true,prediction,*,genuine_posterior):
    import numpy as np
    from arggnn.american_c3_evaluate_v1 import metrics
    p=prediction["probabilities"];calls=p.argmax(-1)
    confusion=np.zeros((3,3),dtype=np.int64);np.add.at(confusion,(true,calls),1)
    result=metrics(confusion)
    if genuine_posterior:
        chosen=p[np.arange(len(true)),true].astype(np.float64)
        zeros=int((chosen==0).sum())
        nll=None if zeros else float(-np.log(chosen).mean())
        target=np.eye(3,dtype=np.float64)[true]
        result["probabilistic"]=dict(available=True,NLL=nll,NLL_status="positive_infinity" if zeros else "finite",
            zero_true_probability_records=zeros,Brier=float(np.square(p.astype(np.float64)-target).sum(-1).mean()),
            NLL_unit="nats_per_focal",Brier_unit="sum_squared_class_probability_error_per_focal",
            probability_floor=False,renormalization=False)
    else:
        result["probabilistic"]=dict(available=False,reason="native_one_hot_hard_call_transport_is_not_a_posterior")
    return result


def _flat(result):
    out={k:result[k] for k in ("accuracy","macro_F1")}
    for cls,values in result["per_class"].items():
        for metric in ("precision","recall","F1"):out[cls+"_"+metric]=values[metric]
    if result["probabilistic"]["available"]:
        out["Brier"]=result["probabilistic"]["Brier"]
        out["NLL"]=result["probabilistic"]["NLL"]
    return out


def _summary(values,metric):
    if any(v is None for v in values):return dict(parents=len(values),mean=None,sample_SD=None,status="positive_infinity",unit="nats_per_focal")
    result=dict(parents=len(values),mean=statistics.fmean(values),sample_SD=statistics.stdev(values) if len(values)>1 else None,
                unit="nats_per_focal" if metric=="NLL" else "sum_squared_probability_error_per_focal" if metric=="Brier" else "fraction")
    if metric=="accuracy":result.update(mean_percent=100*result["mean"],sample_SD_percentage_points=100*result["sample_SD"] if result["sample_SD"] is not None else None)
    return result


def aggregate(per_method,groups):
    summaries,paired={},{}
    for split,parents in groups.items():
        summaries[split]={}
        for method in METHODS:
            rows=[_flat(per_method[method][p]["result"]) for p in parents]
            summaries[split][method]={k:_summary([r[k] for r in rows],k) for k in rows[0]}
        paired[split]={}
        for a,b in combinations(METHODS,2):
            aa=[_flat(per_method[a][p]["result"]) for p in parents]
            bb=[_flat(per_method[b][p]["result"]) for p in parents]
            metrics=set(aa[0])&set(bb[0])
            paired[split][a+"_minus_"+b]={k:dict(_summary([x[k]-y[k] for x,y in zip(aa,bb)],k),
                per_parent={p:x[k]-y[k] for p,x,y in zip(parents,aa,bb)},direction="positive_favors_first_method") for k in sorted(metrics)}
    return summaries,paired


def _resolve_truth(spec,parent):
    """Called only AFTER the complete all45-actual-predictions artifact exists."""
    gen=_read(spec["generation_ready_refs_by_parent"][parent])
    path=generation._path(spec["generation_completion_paths_by_parent"][parent])
    completion_ref=generation._reference(path)
    completion=generation._read(completion_ref)
    need(completion.get("status")=="completed_C3_simulation_reconstruction"
         and completion.get("parent_id")==parent and completion.get("class_order")==["EUR","EAS","AFR"]
         and completion.get("observation_ref")==gen["observation_ref"],"Actual same-parent private generation completion required")
    truth_ref=completion["private_truth_ref"]
    need(generation._path(truth_ref["path"]).is_relative_to(path.parent/"truth"),"Only this generator's separate private truth payload may be scored")
    return truth_ref,completion_ref


def run(spec,*,resource_check):
    need(callable(resource_check) and os.environ.get("CUDA_VISIBLE_DEVICES")=="","Actual admitted CPU-only scoring callback required")
    args={k:v for k,v in spec.items() if k not in ("schema","requirements","all45_predictions_required_before_truth")}
    need(spec==build_spec(**args),"Exact frozen all15-parent scoring specification required")
    deadline=time.monotonic()+spec["max_seconds"]
    def check(boundary):
        need(time.monotonic()<deadline and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=spec["max_rss_bytes"],"Scoring wall/RSS cap reached")
        resource_check(boundary)
    registry=_registry(spec["parent_registry_ref"])
    parents=registry.ids("validation")+registry.ids("test")
    import numpy as np
    from arggnn.american_c3_evaluate_v1 import verify_prediction
    prepared={};predictions={m:{} for m in METHODS};barrier={}
    cap=spec["max_public_array_uncompressed_bytes"]
    # ALL45 payloads authenticate, validate and enter this sealed RAM bank first.
    for parent in parents:
        fp=_read(spec["focal_preparation_refs_by_parent"][parent])
        records=_load_npz(fp["focals_ref"],cap,keys=(*IDENTITY,"first_occurrence_intervals_bp"))
        need(len(records["record_ids"])==10000 and records["parent_id"].item()==parent
             and records["class_order"].tolist()==["EUR","EAS","AFR"]
             and len(set(records["record_ids"].tolist()))==10000
             and records["parent_id"].shape==()
             and (records["record_ids"]>=0).all() and (records["sample_nodes"]>=0).all()
             and all(records[k].dtype==np.int64 and records[k].shape==(10000,) for k in IDENTITY[:3])
             and records["positions_bp0"].dtype==np.float64 and records["positions_bp0"].shape==(10000,)
             and np.isfinite(records["positions_bp0"]).all()
             and ((records["sample_indices"]>=600)&(records["sample_indices"]<1600)).all(),"All exact original10k raw-path focal identities required")
        intervals=records["first_occurrence_intervals_bp"]
        need(intervals.dtype==np.float64 and intervals.shape==(10000,2) and np.isfinite(intervals).all()
             and ((intervals[:,0]>=0)&(intervals[:,1]<=30000000)&(intervals[:,1]>intervals[:,0])).all()
             and np.array_equal((intervals[:,0]+intervals[:,1])/2,records["positions_bp0"]),"Unchanged first-occurrence focal midpoints required")
        prepared[parent]=records
        barrier[parent]=dict(focal_registry_ref=fp["focals_ref"],focal_identity_sha256=_identity_sha(records),methods={})
        for method in METHODS:
            seal=_seal(method,parent,spec["prediction_refs_by_method"][method][parent],registry,fp)
            prediction=_load_npz(seal["prediction_ref"],cap,keys=(*IDENTITY,"probabilities"))
            calls=verify_prediction(prediction,records)
            need(calls.shape==(10000,) and (prediction["probabilities"]<=1).all(),"Every full focal call/probability must seal")
            if not seal["posterior_metrics"]:
                need(np.array_equal(prediction["probabilities"],np.eye(3,dtype=np.float32)[calls]),"Native hard-call transports must remain exact one-hot")
            predictions[method][parent]=(prediction,seal)
            barrier[parent]["methods"][method]=seal
            check("actual_prediction_authenticated_before_any_truth")
    writer=generation._Writer(generation._path(spec["output_directory"]),spec["max_output_bytes"],check)
    barrier_ref=writer.json("all45_prediction_barrier.json",dict(schema="american_all45_actual_prediction_barrier_v1",
        parent_registry_ref=registry.reference,parents=barrier,records_per_parent=10000,methods=list(METHODS),
        heldout_truth_payloads_opened=0,all15parents_all3methods_actual_predictions_validated=True))
    per_method={m:{} for m in METHODS};truth_refs={}
    for parent in parents:
        truth_ref,generation_completion_ref=_resolve_truth(spec,parent)
        check("before_first_or_next_separately_scored_truth_payload")
        truth=_load_npz(truth_ref,spec["max_truth_uncompressed_bytes"],keys=TRUTH_KEYS,public=False)
        true=join_truth(prepared[parent],truth,resource_check=check)
        del truth
        truth_refs[parent]=dict(truth_ref=truth_ref,generation_completion_ref=generation_completion_ref)
        for method in METHODS:
            prediction,seal=predictions[method][parent]
            result=score_arrays(true,prediction,genuine_posterior=seal["posterior_metrics"])
            per_method[method][parent]=dict(parent_id=parent,split=registry.split_for(parent),records=10000,
                prediction_ref=seal["prediction_ref"],prediction_seal_ref=seal["seal_ref"],semantics=seal["semantics"],
                focal_registry_ref=barrier[parent]["focal_registry_ref"],result=result)
        check("complete_same_parent_three_method_scores")
    groups={"validation":registry.ids("validation"),"test":registry.ids("test"),"all_evaluation":parents}
    summaries,paired=aggregate(per_method,groups)
    summary=dict(schema=SCHEMA,status="completed_shared15parent_paired_scores",parent_registry_ref=registry.reference,
        all45_prediction_barrier_ref=barrier_ref,methods=list(METHODS),per_parent=per_method,
        equal_parent_summaries=summaries,paired_deltas=paired,truth_provenance=truth_refs,
        parents=15,validation_parents=5,test_parents=10,records_per_parent=10000,records_per_method=150000,
        truth_rule="recorded_admixture_pulse12_lineage_destination",truth_join="complete_half_open_searchsorted_right_original_query_column",
        all_predictions_sealed_before_any_heldout_truth=True,probabilities_modified=False,probability_floor=False,
        aggregation="equal_independent_parent_mean_and_sample_SD_ddof1",sample_SD_unit="metric_fraction_except_accuracy_percentage_points_also_reported",
        native_one_hot_transports_excluded_from_NLL_Brier=True,model_or_hyperparameter_changes_performed=False,
        exact_paper_reproduction=False,source_refs=spec["requirements"]["source_refs"])
    return writer.json("summary.json",summary)


def score_factory(**kwargs):
    spec=build_spec(**kwargs)
    return dict(module="arggnn.benchmarking.american_scoring",function="run",spec=spec,
                python=".runtime/env/bin/python",source_refs=spec["requirements"]["source_refs"])


def factory(**kwargs):
    """Concrete ROOT workflow factory alias."""
    return score_factory(**kwargs)
