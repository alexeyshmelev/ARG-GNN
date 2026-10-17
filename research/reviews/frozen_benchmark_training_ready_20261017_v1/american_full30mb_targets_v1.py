"""Separate TRAIN-only C3 focal targets. No private read at import/constructor."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import hashlib,json,zipfile
import numpy as np
import torch
from arggnn.american_c3_common_v1 import CLASSES,require,read,bound
from arggnn.american_c3_evaluate_v1 import registry
REGISTRY_REF=dict(path="research/reviews/ARGformer_American_C3_implementation_20261015_v1/parent_seed_registry_inactive.json",
 sha256="608da56508757646a0584cb653f7f6af4c910f05d560d6da1b99c0c1ea691f9a")
TRAIN=tuple("american_c3_train_%03d"%i for i in range(20))
VAL=tuple("american_c3_val_%03d"%i for i in range(5))
TEST=tuple("american_c3_test_%03d"%i for i in range(10))
IDENTITY=("record_ids","sample_nodes","sample_indices","positions_bp0","parent_id","class_order")
def identity_sha(records):
    h=hashlib.sha256()
    for key in IDENTITY:
        value=np.ascontiguousarray(records[key]);h.update(key.encode());h.update(str(value.dtype).encode())
        h.update(json.dumps(list(value.shape)).encode());h.update(value.tobytes())
    return h.hexdigest()
def authoritative_registry_v1(record=REGISTRY_REF):
    require(record==REGISTRY_REF,"Exact original full30Mb parent seed registry")
    value=read(record);rows=value["parents"]
    require(value["schema"]=="american_c3_parent_split_seeds_v1"
        and value["split_unit"]=="independent simulation parent before path/window sampling"
        and value["train_parents"]==20 and value["val_parents"]==5
        and value["test_parents"]==10,"Frozen original independent parent split")
    require(tuple(r["parent_id"] for r in rows if r["split"]=="train")==TRAIN
        and tuple(r["parent_id"] for r in rows if r["split"]=="val")==VAL
        and tuple(r["parent_id"] for r in rows if r["split"]=="test")==TEST,
        "Original20TRAIN/5VAL/10TEST registry identities")
    return {r["parent_id"]:r for r in rows if r["split"]=="train"}
from arggnn.american_c3_gnn_targets_v1 import TrainFocalTargetVectorV1
class TrainTargetJoinV1:
    """The resolver is invoked only after TRAIN membership and observation seal."""
    def __init__(self,resolve_train_truth:Callable[[str],dict],*,max_truth_uncompressed_bytes:int,train_truth_refs:dict,registry_ref=REGISTRY_REF):
        require(callable(resolve_train_truth) and type(max_truth_uncompressed_bytes)is int
            and max_truth_uncompressed_bytes>0,"Explicit separate bounded TRAIN truth capability")
        self.rows=authoritative_registry_v1(registry_ref)
        require(isinstance(train_truth_refs,dict) and set(train_truth_refs)==set(TRAIN)
            and all(isinstance(r,dict) and set(r)=={"path","sha256"}
                and isinstance(r["path"],str) and r["path"].startswith("experiments/results/")
                and len(r["sha256"])==64 for r in train_truth_refs.values()),
            "ROOT-authenticated exact truth refs for all original20 full30Mb TRAIN parents")
        self.train_truth_refs={p:dict(r) for p,r in train_truth_refs.items()}
        self.resolve=resolve_train_truth;self.cap=max_truth_uncompressed_bytes
    def join(self,parent_id,*,split,plan,observation_seal):
        # This check deliberately precedes resolver invocation and any private path inspection.
        require(split=="train" and parent_id in self.rows,"VAL/TEST/short2Mb parent denied before truth resolution")
        row=self.rows[parent_id]
        require(plan["parent_id"]==parent_id and plan["sequence_length_bp"]==30000000
            and observation_seal["parent_id"]==parent_id and observation_seal["observation_only"] is True
            and observation_seal["all_blocks_sealed_before_truth"] is True
            and observation_seal["plan_ref"]==plan["plan_ref"],"Complete observation-only plan seal before truth")
        r=registry(plan["focal_registry_ref"])
        require(r["parent_id"].item()==parent_id and r["class_order"].tolist()==list(CLASSES)
            and len(r["record_ids"])==10000 and identity_sha(r)==observation_seal["focal_identity_sha256"],
            "Exact predeclared10000 C3 query focal identities")
        truth_ref=self.resolve(parent_id)
        expected=self.train_truth_refs[parent_id]
        require(isinstance(truth_ref,dict) and truth_ref==expected,"Same-parent TRAIN-only private truth path")
        path=bound(truth_ref)
        with zipfile.ZipFile(path) as z:
            names=z.namelist()
            require(set(names)=={k+".npy" for k in ("intervals_bp","query_sample_indices",
                "source_classes_0based","parent_id","class_order")}
                and len(names)==len(set(names))==5 and sum(i.file_size for i in z.infolist())<=self.cap,
                "Bounded exact external truth NPZ, no duplicate members")
        with np.load(path,allow_pickle=False) as z:t={k:z[k] for k in z.files}
        require(set(t)=={"intervals_bp","query_sample_indices","source_classes_0based","parent_id","class_order"}
            and np.array_equal(t["parent_id"],r["parent_id"]) and np.array_equal(t["class_order"],r["class_order"]),
            "External C3 TRAIN truth schema/parent/class identity")
        intervals,samples,labels=t["intervals_bp"],t["query_sample_indices"],t["source_classes_0based"]
        require(intervals.dtype==np.float64 and intervals.ndim==2 and intervals.shape[1]==2 and len(intervals)>0
            and np.isfinite(intervals).all() and intervals[0,0]==0 and intervals[-1,1]==30000000
            and (intervals[:,1]>intervals[:,0]).all() and np.array_equal(intervals[:-1,1],intervals[1:,0])
            and samples.dtype==np.int64 and samples.ndim==1 and len(samples)==1000
            and np.array_equal(samples,np.arange(600,1600,dtype=np.int64))
            and labels.dtype==np.uint8 and labels.shape==(len(intervals),len(samples)) and (labels<3).all(),
            "Complete contiguous half-open truth and original1000 query sample columns")
        positions=r["positions_bp0"];require(((positions>=0)&(positions<30000000)).all(),"No truth extrapolation")
        lookup={int(v):i for i,v in enumerate(samples)}
        require(all(int(v) in lookup for v in r["sample_indices"]),"All original query columns must be present")
        ix=np.searchsorted(intervals[:,1],positions,side="right")
        columns=np.asarray([lookup[int(v)] for v in r["sample_indices"]],dtype=np.int64)
        target=np.asarray(labels[ix,columns],dtype=np.int64)
        target.setflags(write=False)
        receipt=dict(schema="C3_external_TRAIN_truth_focal_join_v1",parent_id=parent_id,split="train",
            truth_ref=truth_ref,focal_registry_ref=plan["focal_registry_ref"],focal_identity_sha256=identity_sha(r),
            records=len(target),class_order=list(CLASSES),interval_rule="half_open_searchsorted_right_original_sample_index",
            labels_in_observation_bank=False,purity_filter=False,nearest_or_majority_fallback=False)
        return TrainFocalTargetVectorV1(parent_id,dict(plan["focal_registry_ref"]),identity_sha(r),
            torch.tensor(target.copy(),dtype=torch.long,device="cpu"),receipt)
