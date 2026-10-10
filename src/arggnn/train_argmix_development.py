"""Dedicated fair fresh-development full-tree ARGMix equation adaptation runner."""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch

from .argmix_development_bindings import (ADAPTER,PATHS,SEEDS,SELECTION,STATUS,ROOT,
    capture_training_snapshot,development_path,implementation_hashes,safe_id,sha)
from .data import GraphSequence,slice_genomic_window
from .models import query_node_index
from .study_models import make_study_model
from .train_study import load_cases
from .confirmatory_metrics import logit_metrics
from .resources import require_storage,set_research_title
from .train import dump_json,set_seed

_IMPORTED_PINS=implementation_hashes()
FEATURE_NAMES=["reference_ANA","reference_CHG","reference_WHG","reference_EHG",
               "is_query_leaf","is_leaf","log_time","log_parent_branch"]


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


@dataclass
class ObservedTile:
    sequence: GraphSequence
    parent_position_indices: torch.Tensor
    window_index: int


@dataclass
class TreeTrainingParent:
    sequence: GraphSequence
    tiles: list[ObservedTile]
    labels: torch.Tensor                 # external targets, never model inputs
    parent_id: str
    metadata: dict


def observed_fixed_tiles(sequence,windows=8):
    """Slice raw topology/features by physical coverage, without any targets."""
    if type(windows) is not int or windows<1:raise ValueError("Positive integer windows required")
    sequence.validate()
    coverage=sequence.window_bounds
    lo,hi=(0.,1.) if coverage is None else (float(coverage[0]),float(coverage[1]))
    tiles=[];seen=torch.zeros(len(sequence.positions),dtype=torch.bool)
    for index in range(windows):
        left,right=max(lo,index/windows),min(hi,(index+1)/windows)
        if left>=right or not bool(((sequence.positions>=left)&(sequence.positions<right)).any()):continue
        part,indices=slice_genomic_window(sequence,left,right)
        if bool(seen[indices].any()):raise RuntimeError("Duplicate parent readout assignment")
        if not torch.equal(part.positions,sequence.positions[indices]) or not torch.equal(part.query_node_ids,sequence.query_node_ids):
            raise RuntimeError("Observed tile changed parent position/query correspondence")
        seen[indices]=True;tiles.append(ObservedTile(part,indices.cpu(),index))
    if not tiles or not bool(seen.all()):raise RuntimeError("Every retained parent readout must appear exactly once")
    return tiles


def verify_parent_contract(case,record,num_queries=64):
    """Check targets/roles/paired identity after the source/prefix-verified loader."""
    sequence=case.sequence;sequence.validate()
    metadata=case.metadata
    if (metadata.get("split")!=record["split"] or metadata.get("replicate")!=record["replicate"]
        or metadata.get("seeds")!=record["seeds"] or metadata.get("master_seed")!=20261101
        or metadata.get("num_classes")!=4 or metadata.get("feature_names")!=FEATURE_NAMES
        or metadata.get("feature_time_scale")!=10000
        or metadata.get("internal_population_annotations_present") is not False
        or sequence.x.shape[1]!=48 or sequence.num_queries!=num_queries):
        raise RuntimeError("Fresh observed parent feature/identity/schema contract failed")
    labels=case.labels
    if labels.dtype!=torch.long or labels.shape!=(len(sequence.positions),num_queries) or not labels.numel() or bool(((labels<0)|(labels>=4)).any()):
        raise RuntimeError("External four-class target shape/dtype/support contract failed")
    alignment=record["paired_alignment"]
    if (alignment.get("targets_equal") is not True or alignment.get("positions_equal") is not True
        or alignment.get("query_order_equal") is not True or metadata.get("paired_alignment")!=alignment
        or tensor_sha(labels)!=alignment["retained_target_sha256"]
        or tensor_sha(sequence.positions)!=alignment["retained_positions_sha256"]):
        raise RuntimeError("Paired external target/coordinate correspondence failed")
    indexes=alignment["retained_parent_grid_indexes"]
    if (not isinstance(indexes,list) or len(indexes)!=len(sequence.positions)
        or any(type(i) is not int or not 0<=i<128 for i in indexes)
        or any(a>=b for a,b in zip(indexes,indexes[1:]))
        or not torch.equal(sequence.positions,(torch.tensor(indexes,dtype=torch.float64)+.5)/128)):
        raise RuntimeError("Original retained128-midpoint correspondence failed")
    rows=metadata["samples"]
    mapping={row["observed_node_id"]:row for row in rows}
    if len(mapping)!=len(rows):raise RuntimeError("Duplicate sample identity records")
    keys=[]
    for query in sequence.query_node_ids.tolist():
        row=mapping[query]
        if row["cohort_role"]!="query" or row.get("reference_class") is not None:
            raise RuntimeError("Query source label leaked into sample metadata")
        keys.append([row["sample_id"],row["haplotype"]])
    if keys!=alignment["query_haplotype_keys"]:raise RuntimeError("Paired query identity/order correspondence failed")
    panel=sequence.x[:,:4]
    if not bool(((panel==0)|(panel==1)).all()) or bool((panel.sum(-1)>1).any()):
        raise RuntimeError("Observed reference labels must be one-hot")
    queries=query_node_index(sequence)
    if bool(panel[queries].any()) or not bool((sequence.x[queries,4:6]==1).all()):
        raise RuntimeError("Query ancestry label leaked into observed features")
    for graph in range(sequence.num_graphs):
        if not torch.equal(panel[sequence.node_graph==graph].sum(0),panel.new_tensor([6.,6.,6.,6.])):
            raise RuntimeError("Require all24 observed labeled references in every full tree")
    internal=sequence.node_ids<0
    if bool(sequence.x[internal,4:6].any()):raise RuntimeError("Internal nodes marked as observed leaves/queries")
    reference=panel.sum(-1)>0
    if bool((sequence.node_ids[reference]<0).any()) or not bool((sequence.x[reference,5]==1).all()):
        raise RuntimeError("Internal/source-label leakage into reference features")
    return sequence,labels,keys


def load_verified_parents(manifest,snapshot):
    """Reuse frozen raw-prefix/adapter loader, then bind only external targets."""
    if snapshot.implementation_sha256!=_IMPORTED_PINS:raise RuntimeError("Source changed after module imports")
    manifest_sha=snapshot.controls[str(development_path(PATHS["inferred"]))][0]
    records={(r["split"],r["replicate"]):r for r in manifest["replicates"]}
    parents={}
    for split in ("train","val"):
        snapshot.check();parents[split]=[]
        for case in load_cases(manifest,split):
            snapshot.check()
            record=records[(split,case.metadata["replicate"])]
            sequence,labels,keys=verify_parent_contract(case,record)
            parents[split].append(TreeTrainingParent(sequence,observed_fixed_tiles(sequence,8),labels,
                f"{manifest_sha}:{split}:{record['replicate']}",{"split":split,
                    "replicate":record["replicate"],"query_haplotype_keys":keys,
                    "source_tensor_sha256":record["tensor_sha256"],"source_prepared_sha256":record["prepared_tensor_sha256"],
                    "retained_target_sha256":record["paired_alignment"]["retained_target_sha256"]}))
    snapshot.check(full_sources=True)
    return parents


def factory(configuration,settings):
    safe_id(configuration["id"])
    options={"hidden_dim":configuration["hidden_dim"],"dropout":configuration["dropout"],
             "num_layers":settings["num_layers"],"reference_adapter":settings["reference_adapter"]}
    model=make_study_model("argmix_full_tree",48,4,options)
    audit=model.configuration()
    fixed={"scientific_status":STATUS,"input_dim":48,"num_classes":4,"num_layers":2,"num_heads":4,
        "ffn_multiplier":4,"token_scope":"full_tree","max_rel_distance":1000,
        "relation_kind":"elapsed_tmrca","time_column":6,"branch_column":7,"time_scale":10000.,
        "cross_tree_aggregation":"none","learned_node_id_embeddings":False,"truth_features":[],"external_pretraining":False}
    if any(audit.get(key)!=value for key,value in fixed.items()):
        raise RuntimeError("Existing ARGMix factory differs from frozen full-tree equation adaptation")
    return model


def check_tile_correspondence(parent,tile,seen):
    indices=tile.parent_position_indices
    if (indices.dtype!=torch.long or indices.device.type!="cpu" or indices.ndim!=1
        or not indices.numel() or torch.unique(indices).numel()!=indices.numel()
        or bool(((indices<0)|(indices>=len(parent.sequence.positions))).any())
        or bool(seen[indices].any()) or not torch.equal(tile.sequence.positions,parent.sequence.positions[indices])
        or not torch.equal(tile.sequence.query_node_ids,parent.sequence.query_node_ids)):
        raise RuntimeError("Prediction tile/parent coordinate/query correspondence failed")
    return indices


def evaluate(model,parents,device):
    model.eval();records=[];predictions=[]
    with torch.no_grad():
        for parent in parents:
            p,q=parent.labels.shape
            logits=torch.empty((p,q,4),dtype=torch.float64)
            seen=torch.zeros(p,dtype=torch.bool)
            for tile in parent.tiles:
                indices=check_tile_correspondence(parent,tile,seen)
                value=model(tile.sequence.to(device)).detach().cpu().double()
                if value.shape!=(len(indices),q,4) or not bool(torch.isfinite(value).all()):
                    raise RuntimeError("Finite aligned raw tile logits required")
                logits[indices]=value;seen[indices]=True
            if not bool(seen.all()):raise RuntimeError("A retained parent readout lacks a prediction")
            records.append({"parent_id":parent.parent_id,**logit_metrics(logits.numpy(),parent.labels.numpy(),bins=15)})
            predictions.append(logits)
    return records,predictions


def train_configuration(configuration,settings,parents,device,seed,checkpoint,binding,integrity_check=None,callback=None):
    """One independently initialized frozen configuration/seed, no extra starts."""
    safe_id(configuration["id"])
    if seed not in SEEDS:raise ValueError("Only fixed101/202/303 training seeds are allowed")
    if integrity_check:integrity_check()
    set_seed(seed);model=factory(configuration,settings).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=configuration["learning_rate"],weight_decay=settings["weight_decay"])
    tiles=[(parent,tile) for parent in parents["train"] for tile in parent.tiles]
    if not tiles or not parents["val"]:raise RuntimeError("Training/validation parents required")
    history=[];best=float("inf");best_epoch=-1;stale=0;started=time.monotonic()
    if device.type=="cuda":torch.cuda.reset_peak_memory_stats()
    for epoch in range(settings["epochs"]):
        if integrity_check:integrity_check()
        model.train();order=list(range(len(tiles)));random.shuffle(order);losses=[]
        for index in order:
            parent,tile=tiles[index]
            indices=check_tile_correspondence(parent,tile,torch.zeros(len(parent.sequence.positions),dtype=torch.bool))
            logits=model(tile.sequence.to(device))
            labels=parent.labels[indices].to(device)
            if logits.shape!=(*labels.shape,4):raise RuntimeError("Training logits/target correspondence failed")
            loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),labels.flatten())
            if not bool(torch.isfinite(loss)):raise RuntimeError("Nonfinite ARGMix training loss")
            optimizer.zero_grad(set_to_none=True);loss.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise RuntimeError("Nonfinite ARGMix gradient")
            torch.nn.utils.clip_grad_norm_(model.parameters(),settings["gradient_clip_norm"])
            optimizer.step();losses.append(float(loss.detach()))
        validation,_=evaluate(model,parents["val"],device)
        nll=float(np.mean([row["nll"] for row in validation]))
        history.append({"epoch":epoch,"train_nll":float(np.mean(losses)),"validation_nll":nll,"val_nll":nll,
                        "elapsed_seconds":time.monotonic()-started})
        if callback:callback(history[-1])
        if integrity_check:integrity_check()
        if nll<best:
            best,best_epoch,stale=nll,epoch,0
            torch.save({"model":model.state_dict(),"epoch":epoch,"validation_nll":best,
                "seed":seed,"configuration_id":configuration["id"],"binding":binding,**binding},checkpoint)
        else:
            stale+=1
            if stale>=settings["patience"]:break
    if integrity_check:integrity_check()
    chosen=torch.load(checkpoint,map_location=device,weights_only=True)
    if chosen["binding"]!=binding or chosen["seed"]!=seed or chosen["configuration_id"]!=configuration["id"]:
        raise RuntimeError("Selected checkpoint source/config/seed identity mismatch")
    model.load_state_dict(chosen["model"])
    validation,predictions=evaluate(model,parents["val"],device)
    if integrity_check:integrity_check()
    selected_nll=float(np.mean([row["nll"] for row in validation]))
    if selected_nll!=chosen["validation_nll"]:raise RuntimeError("Reloaded selected checkpoint validation NLL differs")
    result={"configuration_id":configuration["id"],"configuration":configuration,"training_seed":seed,
        **binding,"validation_nll":selected_nll,"best_epoch":chosen["epoch"],"history":history,
        "per_parent":validation,"parameters":sum(p.numel() for p in model.parameters()),
        "model_configuration":model.configuration(),"checkpoint":str(checkpoint),"checkpoint_sha256":sha(checkpoint),
        "training_seconds":time.monotonic()-started,"selection":SELECTION,
        "ending_config_sha256":binding["protocol_sha256"],"ending_implementation_sha256":binding["implementation_sha256"],
        "peak_cuda_memory_bytes":torch.cuda.max_memory_allocated() if device.type=="cuda" else None}
    return result,predictions


def select_family(records,settings,expected_config_sha256):
    from .argmix_development_bindings import valid_sha
    valid_sha(expected_config_sha256)
    expected={(c["id"],seed) for c in settings["configurations"] for seed in SEEDS}
    ids=[(row["configuration_id"],row["training_seed"]) for row in records]
    if len(ids)!=len(set(ids)) or set(ids)!=expected:raise ValueError("All24 frozen configuration/seed starts required")
    configurations={c["id"]:c for c in settings["configurations"]}
    for row in records:
        if row.get("configuration")!=configurations[row["configuration_id"]]:
            raise ValueError("Family record differs from exact frozen configuration specification")
        if (row.get("protocol_sha256")!=expected_config_sha256
            or row.get("dataset_manifest_sha256")!=settings["dataset_manifest_sha256"]
            or row.get("dataset_protocol_sha256")!=settings["dataset_protocol_sha256"]
            or row.get("completion_receipt_sha256")!=settings["completion_receipt_sha256"]
            or row.get("implementation_sha256")!=settings["bound_implementation_sha256"]):
            raise ValueError("Family records differ from exact config/source/code bindings")
    summaries=[]
    for config in settings["configurations"]:
        rows=[row for row in records if row["configuration_id"]==config["id"]]
        counts={row["parameters"] for row in rows}
        expected_parameters=sum(p.numel() for p in factory(config,settings).parameters())
        if counts!={expected_parameters} or not all(np.isfinite(row["validation_nll"]) for row in rows):
            raise ValueError("Finite NLL and matching per-configuration capacity required")
        summaries.append({"configuration_id":config["id"],"parameters":counts.pop(),
                          "mean_validation_nll":float(np.mean([row["validation_nll"] for row in rows]))})
    minimum=min(row["mean_validation_nll"] for row in summaries)
    tied=[row for row in summaries if row["mean_validation_nll"]<=minimum+settings["family_tie_tolerance_nats"]]
    return {"selected":min(tied,key=lambda row:(row["parameters"],row["configuration_id"])),"all_configurations":summaries}


def require_device(device):
    set_research_title()
    if device=="cuda":
        if os.environ.get("ARGGNN_PHYSICAL_GPU") not in {"2","3","4","5"}:raise RuntimeError("Use project GPU launcher")
        lease=development_path(os.environ["ARGGNN_GPU_LEASE"])
        if os.fstat(int(os.environ["ARGGNN_GPU_LEASE_FD"])).st_ino!=lease.stat().st_ino or torch.cuda.device_count()!=1:
            raise RuntimeError("GPU lease/visible-device mismatch")
    elif torch.cuda.is_available():raise RuntimeError("CPU runs require CUDA_VISIBLE_DEVICES empty")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    parser.add_argument("--expected-config-sha256",required=True)
    parser.add_argument("--output")
    parser.add_argument("--seed",type=int,default=101)
    parser.add_argument("--configurations",nargs="+")
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    parser.add_argument("--validate-config",action="store_true")
    args=parser.parse_args(argv);torch.set_num_threads(4)
    if args.seed not in SEEDS:raise ValueError("Only fixed101/202/303 seeds allowed")
    if args.configurations:
        for name in args.configurations:safe_id(name)
    config_path=development_path(args.config)
    settings,manifest,snapshot=capture_training_snapshot(config_path,args.expected_config_sha256)
    if snapshot.implementation_sha256!=_IMPORTED_PINS:raise RuntimeError("Source changed after module imports")
    names=args.configurations or [config["id"] for config in settings["configurations"]]
    if len(names)!=len(set(names)) or not set(names).issubset({config["id"] for config in settings["configurations"]}):
        raise ValueError("Distinct frozen family configuration IDs required")
    if args.validate_config:
        for config in settings["configurations"]:
            if config["id"] in names:
                model=factory(config,settings)
                print(json.dumps({"event":"argmix_bound_config_valid","configuration_id":config["id"],
                    "parameters":sum(p.numel() for p in model.parameters()),"config_sha256":args.expected_config_sha256}),flush=True)
        snapshot.check(full_sources=True);return
    if not args.output:parser.error("--output is required for fitting")
    require_device(args.device)
    output=development_path(args.output)
    if output.exists():raise RuntimeError("Choose a fresh development run directory")
    identity=hashlib.sha256(str(output.relative_to(ROOT)).encode()).hexdigest()[:16]
    checkpoint_root=development_path(Path("checkpoints/argmix_development_v1")/
        f"{output.name}_{identity}_{args.expected_config_sha256[:12]}_s{args.seed}")
    if checkpoint_root.exists():raise RuntimeError("Fresh checkpoint namespace required")
    started=time.monotonic();parents=load_verified_parents(manifest,snapshot)
    snapshot.check(full_sources=True);require_storage(headroom_bytes=500_000_000)
    output.mkdir(parents=True);checkpoint_root.mkdir(parents=True)
    (output/"protocol.json").write_bytes(snapshot.bytes_for(config_path))
    (output/"dataset_manifest.json").write_bytes(snapshot.bytes_for(settings["dataset_manifest"]))
    (output/"dataset_protocol.json").write_bytes(snapshot.bytes_for(settings["dataset_protocol"]))
    (output/"completion.json").write_bytes(snapshot.bytes_for(settings["completion_receipt"]))
    binding={"protocol_sha256":args.expected_config_sha256,"dataset_manifest_sha256":settings["dataset_manifest_sha256"],
        "dataset_protocol_sha256":settings["dataset_protocol_sha256"],"completion_receipt_sha256":settings["completion_receipt_sha256"],
        "implementation_sha256":snapshot.implementation_sha256,
        **{key:settings[key] for key in ("source_dataset_manifest_sha256","boundary_dataset_manifest_sha256",
            "boundary_preparation_sha256","age_dataset_manifest_sha256","age_preparation_sha256","preparation_execution_freeze_sha256")}}
    results={"stage":settings["stage"],"family":"argmix_full_tree","evaluation_split":"fresh_development_validation_only",
        "test_accessed":False,"training_seed":args.seed,"training_seeds":SEEDS,"development_namespace":"development-v1",
        "nll_policy":settings["nll_policy"],"initial_implementation_sha256":snapshot.implementation_sha256,**binding,"scientific_status":STATUS,
        "family_selection_complete":False,"models":{},"source_artifact_sha256":snapshot.sources,
        "integrity_checks":"exact config/plan/completion/manifests/protocol/freeze/code hashes and source stat identities through preparation/configurations/epochs; full source hashes at major boundaries",
        "preparation_seconds":time.monotonic()-started,"parents":{split:[{"parent_id":p.parent_id,
            "positions":len(p.sequence.positions),"queries":p.sequence.num_queries,"tiles":len(p.tiles),**p.metadata} for p in group] for split,group in parents.items()},
        "environment":{"torch":str(torch.__version__),"cuda":torch.version.cuda,"device":args.device},
        "physical_gpu":os.environ.get("ARGGNN_PHYSICAL_GPU"),"interpretation":settings["interpretation"]}
    dump_json(output/"preparation.json",results)
    active=None
    try:
        for config in settings["configurations"]:
            if config["id"] not in names:continue
            active=config["id"];snapshot.check(full_sources=True);require_storage(headroom_bytes=100_000_000)
            checkpoint=checkpoint_root/(active+".pt")
            def callback(item):
                require_storage(headroom_bytes=100_000_000)
                print(json.dumps({"event":"epoch","configuration_id":active,"seed":args.seed,**item}),flush=True)
            result,logits=train_configuration(config,settings,parents,torch.device(args.device),args.seed,
                checkpoint,binding,snapshot.check,callback)
            snapshot.check(full_sources=True)
            prediction=checkpoint_root/(active+"_validation_logits.pt")
            torch.save({"logits":logits,"parent_ids":[p.parent_id for p in parents["val"]],
                "positions":[p.sequence.positions for p in parents["val"]],
                "query_node_ids":[p.sequence.query_node_ids for p in parents["val"]],
                "query_haplotype_keys":[p.metadata["query_haplotype_keys"] for p in parents["val"]],
                "seed":args.seed,"training_seed":args.seed,"nll_policy":settings["nll_policy"],"configuration_id":active,**binding},prediction)
            result.update(validation_logits=str(prediction),validation_logits_sha256=sha(prediction),
                source_artifact_sha256=snapshot.sources,ending_config_sha256=args.expected_config_sha256,
                ending_implementation_sha256=snapshot.implementation_sha256)
            results["models"][active]=result
            dump_json(output/(active+".json"),result);dump_json(output/"progress.json",results)
        snapshot.check(full_sources=True)
        results.update(ending_config_sha256=args.expected_config_sha256,ending_implementation_sha256=snapshot.implementation_sha256)
        dump_json(output/"results.json",results)
    except Exception as error:
        dump_json(output/"failure.json",{"configuration_id":active,"training_seed":args.seed,**binding,
            "reason":str(error),"exception_type":type(error).__name__,"test_accessed":False})
        raise
    print(json.dumps({"event":"argmix_development_seed_complete","output":str(output),"family_selection_complete":False}),flush=True)


if __name__=="__main__":main()
