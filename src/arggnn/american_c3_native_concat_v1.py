"""Admitted CPU counts, row-preserving native concatenation and existing seal.

No Torch/NumPy import at module load. No model, GPU, truth reader, corpus
sampling, token/embedding matching, normalization or metric change is added.
"""
from __future__ import annotations
import argparse,io,json,os,pickle
from pathlib import Path
from .american_c3_common_v1 import ROOT,CLASSES,require,local,ref,bound
from .american_c3_native_contract_v1 import (
    DEFAULTS,SPLITS,VIEWS,AUTHOR_MANIFEST_REF,metadata,validate_active_input,
    load_root_context,preparation_v1,counts_v1,fit_completion_v1)
SOURCE="src/arggnn/american_c3_native_concat_v1.py"

class _PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self,module,name):raise ValueError("Raw path preparation uses primitive pickle data only")
def primitive_pickle(record,cap):
    p=bound(record);require(p.stat().st_size<=cap,"Bounded original raw-path pickle")
    with p.open("rb") as stream:return _PrimitiveUnpickler(stream).load()
def row_order(record,*,parent_id,reference):
    import numpy as np
    with np.load(bound(record),allow_pickle=False) as z:out={k:z[k].copy() for k in z.files}
    keys={"record_ids","sample_nodes","sample_indices","parent_id","class_order"}
    if reference:keys.add("reference_classes_0based")
    n=len(out["record_ids"])
    require(set(out)==keys and n>0 and all(out[k].dtype==np.int64 and out[k].shape==(n,)
        for k in keys-{"parent_id","class_order"}) and len(set(out["record_ids"].tolist()))==n
        and out["parent_id"].shape==() and out["parent_id"].item()==parent_id
        and out["class_order"].tolist()==list(CLASSES),"Exact raw-full-path row sidecar schema/identity")
    if reference:require(bool(((out["reference_classes_0based"]>=0)&(out["reference_classes_0based"]<3)).all()),
        "Known reference C3 classes only")
    return out
def count_preparation_v1(spec,cfg,ctx):
    """Observation-only actual counts; zero-batch corpora get an ineligible receipt."""
    import numpy as np
    from .american_c3_evaluate_v1 import registry
    prep,pcfg,directory=preparation_v1(spec);records=registry(prep["focals_ref"])
    observation=metadata(prep["observation_ref"]);roster=metadata(observation["roster_ref"]);rows=roster["samples"]
    require(observation["query_ancestry_truth_in_inputs"] is False and len(rows)==1600
        and [r["sample_index"] for r in rows]==list(range(1600))
        and [r["sample_index"] for r in rows if r["role"]=="query"]==list(range(600,1600))
        and prep["parent_id"]==roster["parent_id"],
        "Original1600 C3 observed haplotypes, with only known reference labels")
    lookup={r["observed_node_id"]:r for r in rows};datasets={};labels_refs={};vocabulary=None;query_orders=[]
    for view in VIEWS:
        for split in SPLITS:
            ctx["fast_guard"]("CPU_count_dataset")
            key=view+"/"+split;dataset_ref=prep["native_dataset_refs"][key]
            payload=primitive_pickle(dataset_ref,cfg["max_preparation_file_bytes"])
            require(set(payload)=={"raw_sequences","vocab"} and isinstance(payload["raw_sequences"],list),
                "Complete original native raw full-path dataset")
            vocab=payload["vocab"];sequences=payload["raw_sequences"]
            if vocabulary is None:vocabulary=vocab
            require(vocab==vocabulary and set(vocab)=={"special","node_id_tokens"}
                and vocab["special"]=={"[PAD]":0,"[MASK]":1,"[CLS]":2,"[SEP]":3}
                and sorted(vocab["node_id_tokens"].values())==list(range(4,4+prep["vocabulary_nodes"])),
                "One exact per-ARG observed vocabulary across all native views/splits")
            order=row_order(prep["full_path_id_order_refs"][key],parent_id=prep["parent_id"],reference=view=="contrastive")
            require(len(sequences)==len(order["record_ids"]),"Every raw sequence retains its exact source sidecar row")
            tokens=0;longest=0
            for i,seq in enumerate(sequences):
                if i%10000==0:ctx["fast_guard"]("CPU_count_full_paths")
                require(isinstance(seq,list) and seq and all(type(v) is int and 0<=v<2**31 for v in seq)
                    and seq[0]==int(order["sample_nodes"][i])
                    and lookup[seq[0]]["sample_index"]==int(order["sample_indices"][i]),
                    "Only exact full-path/sidecar row association, never truncated token matching")
                role=lookup[seq[0]]["role"]
                if view=="contrastive":
                    require(role=="pure_source_reference"
                        and lookup[seq[0]]["observed_reference_class"]==int(order["reference_classes_0based"][i]),
                        "Native supervision uses known references only")
                if view=="query_paths":require(role=="query","All query embedding rows remain unlabelled")
                tokens+=len(seq);longest=max(longest,len(seq))
            if view=="contrastive":
                lr=ref(directory/view/split/"labels.pkl")
                labels=primitive_pickle(lr,cfg["max_preparation_file_bytes"])
                require(isinstance(labels,list) and labels==order["reference_classes_0based"].tolist(),
                    "Exact original reference labels.pkl is now SHA-bound")
                labels_refs[split]=lr
            if view=="query_paths":query_orders.append(order)
            datasets[key]=dict(dataset_ref=dataset_ref,row_order_ref=prep["full_path_id_order_refs"][key],
                rows=len(sequences),raw_path_tokens=tokens,max_raw_path_nodes=longest,
                raw_pickle_bytes=bound(dataset_ref).stat().st_size)
            del payload,sequences,order
    q={k:np.concatenate([x[k] for x in query_orders]) for k in ("record_ids","sample_nodes","sample_indices")}
    require(len(q["record_ids"])==10000 and len(set(q["record_ids"].tolist()))==10000
        and set(q["record_ids"].tolist())==set(records["record_ids"].tolist()),"All and only frozen10000 common query full-path IDs")
    ix={int(v):i for i,v in enumerate(q["record_ids"])}
    join=np.asarray([ix[int(v)] for v in records["record_ids"]])
    require(all(np.array_equal(q[k][join],records[k]) for k in ("record_ids","sample_nodes","sample_indices")),
        "Original complete focal identity join before any native fit")
    mlm=5*(datasets["mlm/train"]["rows"]//512);contrastive=datasets["contrastive/train"]["rows"]//4096
    return ctx["writer"].save("completed.json",dict(schema="american_c3_native_actual_counts_v1",
        status="completed_C3_native_observation_only_counts",source_ref=ref(SOURCE),parent_id=prep["parent_id"],
        preparation_ref=spec["refs"]["preparation"],preparation_config_ref=spec["refs"]["preparation_config"],
        active_input_ref=ctx["input_ref"],config_ref=spec["refs"]["config"],runtime_ref=spec["refs"]["runtime"],
        class_order=list(CLASSES),haplotypes=1600,query_focal_count=10000,datasets=datasets,
        reference_label_refs=labels_refs,vocabulary_nodes=prep["vocabulary_nodes"],
        tokenizer_vocab_size=prep["vocabulary_nodes"]+4,MLM_optimizer_updates=mlm,
        contrastive_optimizer_updates=contrastive,native_training_eligible=(mlm>0 and contrastive>0),
        drop_last_release_batches_preserved=True,source_corpus_capped_or_sampled=False,
        query_ancestry_truth_opened=False,counts_are_not_RAM_GPU_or_timing_qualification=True))

def concatenate_embeddings_v1(spec,cfg,ctx):
    import numpy as np,torch
    prep,_,_=preparation_v1(spec);counts=counts_v1(spec,prep)
    fit=fit_completion_v1(spec["refs"]["fit_completion"],spec)
    completion=metadata(spec["refs"]["embedding_completion"])
    require(completion["status"]=="completed_C3_native_four_single_split_embeddings"
        and completion["fit_completion_ref"]==spec["refs"]["fit_completion"]
        and completion["preparation_ref"]==spec["refs"]["preparation"]
        and completion["counts_ref"]==spec["refs"]["counts"] and completion["parent_id"]==spec["parent_id"]
        and completion["checkpoint_ref"]==fit["phases"]["contrastive"]["checkpoint_ref"]
        and set(completion["outputs"])=={v+"/"+s for v in ("query_paths","contrastive") for s in SPLITS},
        "All four unmodified same-checkpoint single-split producers")
    combined={};order_refs={}
    for view in ("query_paths","contrastive"):
        parts=[];orders=[]
        for split in SPLITS:
            ctx["fast_guard"]("CPU_native_embedding_read")
            key=view+"/"+split;item=completion["outputs"][key]
            producer=metadata(item["child_completion_ref"]);wait=metadata(item["actual_wait_ref"])
            require(wait["returncode"]==0 and wait["actual_direct_child_wait"] is True
                and wait["process_group_ESRCH"] is True and wait["child_completion_ref"]==item["child_completion_ref"]
                and producer["embedding_ref"]==item["embedding_ref"] and producer["view"]==view
                and producer["split"]==split and producer["checkpoint_ref"]==completion["checkpoint_ref"]
                and producer["parent_id"]==spec["parent_id"]
                and producer["raw_dataset_ref"]==prep["native_dataset_refs"][key]
                and producer["row_order_ref"]==prep["full_path_id_order_refs"][key]
                and producer["query_ancestry_truth_opened"] is False
                and producer["child_input_ref"]==wait["child_input_ref"],
                "Exact actual native producer/directwait/drain provenance")
            path=bound(item["embedding_ref"]);require(path.stat().st_size<=cfg["max_embedding_input_bytes"],"Bounded actual native CPU input")
            value=torch.load(path,map_location="cpu",weights_only=True,mmap=True)
            meta=value["metadata"];n=counts["datasets"][key]["rows"]
            require(meta["splits"]==[split] and meta["num_samples"]==n and meta["embedding_dim"]==768
                and meta["pooling_strategy"]=="head_norm" and meta["layer_index"] is None,
                "Exact complete single-split row count and native pooling/layer/dimension")
            order=row_order(prep["full_path_id_order_refs"][key],parent_id=spec["parent_id"],reference=view=="contrastive")
            expected={"embeddings","input_ids","attention_masks","extant_node_ids","metadata"}
            if view=="contrastive":expected.add("labels")
            require(set(value)==expected and all(torch.is_tensor(t) and t.device.type=="cpu" and t.ndim>=1
                and t.shape[0]==n for k,t in value.items() if k!="metadata"),"Complete compatible native CPU row tensors")
            require(value["embeddings"].dtype==torch.float32 and value["embeddings"].shape==(n,768)
                and bool(torch.isfinite(value["embeddings"]).all())
                and value["input_ids"].dtype==value["attention_masks"].dtype==torch.long
                and value["input_ids"].shape==value["attention_masks"].shape==(n,128)
                and value["extant_node_ids"].dtype==torch.long
                and np.array_equal(value["extant_node_ids"].numpy().reshape(-1),order["sample_nodes"]),
                "Preserved complete native tensor dtypes/axes and exact sidecar extant IDs")
            if view=="contrastive":
                require(value["labels"].dtype==torch.long
                    and np.array_equal(value["labels"].numpy().reshape(-1),order["reference_classes_0based"]),
                    "Original known reference labels only")
            parts.append(value);orders.append(order)
        require(set(parts[0])==set(parts[1])
            and {k:v for k,v in parts[0]["metadata"].items() if k not in ("splits","num_samples")}==
                {k:v for k,v in parts[1]["metadata"].items() if k not in ("splits","num_samples")},
            "Compatible native metadata/mappings across train and val")
        out={}
        for name in parts[0]:
            if name=="metadata":continue
            a,b=parts[0][name],parts[1][name]
            require(a.dtype==b.dtype and a.shape[1:]==b.shape[1:],"Same exact tensor key/dtype/trailing shape")
            out[name]=torch.cat((a,b),dim=0)
        out["metadata"]=dict(parts[0]["metadata"],splits=["train","val"],
            num_samples=sum(p["metadata"]["num_samples"] for p in parts))
        full_order={k:np.concatenate([o[k] for o in orders]) for k in orders[0] if k not in ("parent_id","class_order")}
        full_order.update(parent_id=orders[0]["parent_id"],class_order=orders[0]["class_order"])
        require(len(set(full_order["record_ids"].tolist()))==len(full_order["record_ids"]),"No duplicate full raw-path IDs across splits")
        combined[view]=ctx["writer"].save(view+"_train_then_val_embeddings.pt",out,kind="torch")
        order_refs[view]=ctx["writer"].save(view+"_train_then_val_full_path_order.npz",full_order,kind="npz")
        del out,parts,orders
    return ctx["writer"].save("completed.json",dict(status="completed_C3_native_CPU_row_preserving_concat",
        parent_id=spec["parent_id"],class_order=list(CLASSES),source_ref=ref(SOURCE),
        preparation_ref=spec["refs"]["preparation"],counts_ref=spec["refs"]["counts"],
        fit_completion_ref=spec["refs"]["fit_completion"],embedding_completion_ref=spec["refs"]["embedding_completion"],
        checkpoint_ref=completion["checkpoint_ref"],combined_embeddings_refs=combined,combined_full_path_order_refs=order_refs,
        all_four_original_outputs=completion["outputs"],splits=["train","val"],all_tensor_values_preserved_bit_for_bit=True,
        sort_filter_normalization_or_cap=False,query_ancestry_truth_opened=False))

class _SealWriter:
    def __init__(self,writer):self.writer=writer
    def json(self,name,value):return self.writer.save(name,value)
    def npz(self,name,**arrays):return self.writer.save(name,arrays,kind="npz")
def prediction_seal_v1(spec,cfg,ctx):
    from .american_c3_evaluate_v1 import DEFAULTS as EVAL,native_top1_seal,registry
    prep,_,_=preparation_v1(spec);counts_v1(spec,prep)
    combined=metadata(spec["refs"]["combined_completion"]);retrieval=metadata(spec["refs"]["retrieval_completion"])
    require(retrieval["status"]=="completed_C3_native_top1_retrieval"
        and retrieval["combined_completion_ref"]==spec["refs"]["combined_completion"]
        and retrieval["parent_id"]==combined["parent_id"]==spec["parent_id"]
        and combined["preparation_ref"]==spec["refs"]["preparation"]
        and combined["counts_ref"]==spec["refs"]["counts"]
        and combined["status"]=="completed_C3_native_CPU_row_preserving_concat",
        "Exact completed same-parent native retrieval and unchanged combined outputs")
    wait=metadata(retrieval["actual_wait_ref"])
    require(wait["returncode"]==0 and wait["actual_direct_child_wait"] is True
        and wait["process_group_ESRCH"] is True
        and wait["child_completion_ref"]==retrieval["child_completion_ref"],
        "Actual retrieval wait0/drain before native prediction seal")
    child=metadata(retrieval["child_completion_ref"])
    require(child["retrieval_ref"]==retrieval["retrieval_ref"]
        and child["corpus_ref"]==combined["combined_embeddings_refs"]["contrastive"]
        and child["queries_ref"]==combined["combined_embeddings_refs"]["query_paths"]
        and child["child_input_ref"]==wait["child_input_ref"],
        "Unchanged actual native top1 product bound to exact combined query/reference tensors")
    config=dict(EVAL,mode="native_top1_seal",method="native_ARGformer_C3_release_per_ARG",
        preparation_ref=spec["refs"]["preparation"],focal_registry_ref=prep["focals_ref"],
        native_query_embeddings_ref=combined["combined_embeddings_refs"]["query_paths"],
        native_corpus_embeddings_ref=combined["combined_embeddings_refs"]["contrastive"],
        native_retrieval_ref=retrieval["retrieval_ref"],external_truth_ref=None,prediction_ref=None)
    return native_top1_seal(config,_SealWriter(ctx["writer"]),registry(prep["focals_ref"]))

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--execute",action="store_true")
    parser.add_argument("--input");parser.add_argument("--expected-input-sha256");args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="source_only_inactive_C3_native_CPU_contract",source_ref=ref(SOURCE),
            phases=["counts","concat","prediction_seal"],scientific_payloads_opened=False),indent=2));return 0
    require(args.input and args.expected_input_sha256,"Actual root CPU input path/SHA required")
    record=dict(path=args.input,sha256=args.expected_input_sha256);spec=metadata(record)
    cfg,_=validate_active_input(record,spec,SOURCE,GPU=False)
    require(os.environ.get("CUDA_VISIBLE_DEVICES")=="","Explicit CPU-only native serialization stage")
    ctx=load_root_context(record,spec,CPU=True);ctx["input_ref"]=record
    ctx["boundary_guard"]("before_CPU_stage",required_write_bytes=cfg["phase_write_budget_bytes"])
    ctx["fast_guard"]("before_CPU_scientific_imports")
    result={"counts":count_preparation_v1,"concat":concatenate_embeddings_v1,
        "prediction_seal":prediction_seal_v1}[spec["phase"]](spec,cfg,ctx)
    ctx["boundary_guard"]("after_CPU_stage",required_write_bytes=0)
    print(json.dumps(dict(completed_ref=result)),flush=True);return 0
if __name__=="__main__":raise SystemExit(main())
