"""Fresh64/16 completion-bound full canonical-tree GNN+all-record GRU training."""
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

from . import argmix_development_bindings as shared
from .canonical_development_bindings import (AGGREGATION,FAMILY,INPUT_SCOPE,NLL_POLICY,SOURCE_KEYS,
    canonical_path,capture_training_snapshot,implementation_hashes,settings_from_original_bytes)
from .development_execution import (DETERMINISTIC_POLICY,configure_deterministic_development,deterministic_development_manifest)
from .canonical_gru import (CanonicalPreparedParent,CanonicalTile,CanonicalTreeGNNGRU,
    canonical_fixed_tiles,prepare_canonical_parent)
from .train_argmix_development import FEATURE_NAMES,load_verified_parents,require_device
from .confirmatory_metrics import logit_metrics
from .resources import require_storage
from .train import dump_json,set_seed

_IMPORTED_PINS=implementation_hashes()


@dataclass
class CanonicalDevelopmentParent:
    prepared: CanonicalPreparedParent
    tiles: list[CanonicalTile]
    labels: torch.Tensor               # external targets, joined after preprocessing
    parent_id: str
    metadata: dict


def prepare_observed_parent(sequence,*,parent_id,split,source_tensor_sha256,source_prepared_sha256):
    """Observed sequence/options only: no dataset, labels or truth-bearing handle."""
    return prepare_canonical_parent(sequence,shared.ADAPTER,base_feature_count=8,
        source_provenance={"parent_id":parent_id,"split":split,"source_tensor_sha256":source_tensor_sha256,
                           "source_prepared_sha256":source_prepared_sha256},
        original_feature_names=FEATURE_NAMES,source_tensor_sha256=source_tensor_sha256)


def join_external_targets(prepared,labels,parent_id,metadata,windows=8):
    sequence=prepared.sequence
    if labels.dtype!=torch.long or labels.shape!=(len(sequence.positions),sequence.num_queries) or bool(((labels<0)|(labels>=4)).any()):
        raise RuntimeError("External canonical target shape/dtype/class correspondence failed")
    if sequence.x.shape[1]!=48:raise RuntimeError("Fresh canonical observed F48 required")
    return CanonicalDevelopmentParent(prepared,canonical_fixed_tiles(prepared,windows),labels,parent_id,dict(metadata))


def prepare_verified_parents(manifest,snapshot):
    """Use the shared raw-source catalog/loader, then canonicalize observed data."""
    snapshot.check(full_sources=True)
    if snapshot.implementation_sha256!=_IMPORTED_PINS:
        raise RuntimeError("Canonical source changed after module imports")
    raw=load_verified_parents(manifest,snapshot.shared)
    parents={}
    for split in ("train","val"):
        parents[split]=[]
        for parent in sorted(raw[split],key=lambda value:value.metadata["replicate"]):
            snapshot.check();started=time.monotonic()
            prepared=prepare_observed_parent(parent.sequence,parent_id=parent.parent_id,split=split,
                source_tensor_sha256=parent.metadata["source_tensor_sha256"],
                source_prepared_sha256=parent.metadata["source_prepared_sha256"])
            if not torch.equal(prepared.sequence.positions,parent.sequence.positions) or not torch.equal(prepared.sequence.query_node_ids,parent.sequence.query_node_ids):
                raise RuntimeError("Canonical preparation changed original position/query order")
            metadata={**parent.metadata,"canonical_preparation_seconds":time.monotonic()-started,
                "canonicalization":prepared.correspondence.provenance,"adapter":prepared.adapter_provenance,
                "source_graphs":parent.sequence.num_graphs,"canonical_graphs":prepared.sequence.num_graphs}
            parents[split].append(join_external_targets(prepared,parent.labels,parent.parent_id,metadata,8))
            snapshot.check()
    snapshot.check(full_sources=True)
    return parents


def factory(configuration,settings):
    shared.safe_id(configuration["id"])
    model=CanonicalTreeGNNGRU(48,4,configuration["hidden_dim"],settings["num_layers"],
        settings["temporal_layers"],configuration["dropout"])
    audit=model.configuration_manifest()
    if audit["GNN_depth"]!=2 or audit["GRU_depth"]!=2 or audit["input_dim"]!=48 or not audit["bidirectional"]:
        raise RuntimeError("Frozen full canonical-tree source/temporal schema changed")
    return model


def check_tile(parent,tile,seen):
    indices=tile.parent_position_indices;sequence=tile.model_input.sequence
    if (indices.dtype!=torch.long or indices.device.type!="cpu" or indices.ndim!=1 or not indices.numel()
        or torch.unique(indices).numel()!=indices.numel()
        or bool(((indices<0)|(indices>=len(parent.prepared.sequence.positions))).any())
        or bool(seen[indices].any()) or not torch.equal(sequence.positions,parent.prepared.sequence.positions[indices])
        or not torch.equal(sequence.query_node_ids,parent.prepared.sequence.query_node_ids)):
        raise RuntimeError("Canonical tile/parent coordinate/query scatter correspondence failed")
    return indices


def evaluate(model,parents,device):
    model.eval();records=[];predictions=[]
    with torch.no_grad():
        for parent in parents:
            p,q=parent.labels.shape;logits=torch.empty((p,q,4),dtype=torch.float64)
            seen=torch.zeros(p,dtype=torch.bool)
            for tile in parent.tiles:
                indices=check_tile(parent,tile,seen)
                value=model(tile.model_input.to(device)).detach().cpu().double()
                if value.shape!=(len(indices),q,4) or not bool(torch.isfinite(value).all()):
                    raise RuntimeError("Finite aligned raw canonical logits required")
                logits[indices]=value;seen[indices]=True
            if not bool(seen.all()):raise RuntimeError("A retained original canonical readout lacks prediction")
            records.append({"parent_id":parent.parent_id,**logit_metrics(logits.numpy(),parent.labels.numpy(),bins=15)})
            predictions.append(logits)
    return records,predictions


def train_configuration(configuration,settings,parents,device,seed,checkpoint,binding,integrity_check=None,callback=None):
    """Independent configuration/seed. Raw equal-parent VAL NLL selects weights."""
    shared.safe_id(configuration["id"])
    if type(seed) is not int or seed not in shared.SEEDS:raise ValueError("Only fixed101/202/303 seeds permitted")
    def check_live():
        if integrity_check:integrity_check()
        declared=binding.get("deterministic_policy")
        if declared is not None and deterministic_development_manifest()!=declared:
            raise RuntimeError("Frozen canonical deterministic policy drift")
    if device.type=="cuda" and binding.get("deterministic_policy")!=DETERMINISTIC_POLICY:
        raise RuntimeError("Declared strict deterministic development policy required before CUDA fitting")
    check_live()
    set_seed(seed);model=factory(configuration,settings).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=configuration["learning_rate"],weight_decay=settings["weight_decay"])
    tiles=[(parent,tile) for parent in parents["train"] for tile in parent.tiles]
    if not tiles or not parents["val"]:raise RuntimeError("Training and validation parents required")
    history=[];best=float("inf");stale=0;started=time.monotonic()
    if device.type=="cuda":torch.cuda.reset_peak_memory_stats()
    for epoch in range(settings["epochs"]):
        check_live()
        model.train();order=list(range(len(tiles)));random.shuffle(order);losses=[]
        for index in order:
            parent,tile=tiles[index]
            indices=check_tile(parent,tile,torch.zeros(len(parent.prepared.sequence.positions),dtype=torch.bool))
            logits=model(tile.model_input.to(device));labels=parent.labels[indices].to(device)
            if logits.shape!=(*labels.shape,4):raise RuntimeError("Canonical training logits/targets differ")
            loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),labels.flatten())
            if not bool(torch.isfinite(loss)):raise RuntimeError("Nonfinite canonical development loss")
            optimizer.zero_grad(set_to_none=True);loss.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise RuntimeError("Nonfinite canonical development gradient")
            torch.nn.utils.clip_grad_norm_(model.parameters(),settings["gradient_clip_norm"])
            optimizer.step();losses.append(float(loss.detach()))
        validation,_=evaluate(model,parents["val"],device)
        nll=float(np.mean([row["nll"] for row in validation]))
        history.append({"epoch":epoch,"train_nll":float(np.mean(losses)),"val_nll":nll,
                        "elapsed_seconds":time.monotonic()-started})
        if callback:callback(history[-1])
        check_live()
        if nll<best:
            best,stale=nll,0
            torch.save({"model":model.state_dict(),"epoch":epoch,"validation_nll":nll,"seed":seed,
                "configuration_id":configuration["id"],"binding":binding,**binding},checkpoint)
        else:
            stale+=1
            if stale>=settings["patience"]:break
    check_live()
    chosen=torch.load(checkpoint,map_location=device,weights_only=True)
    if chosen["binding"]!=binding or chosen["seed"]!=seed or chosen["configuration_id"]!=configuration["id"]:
        raise RuntimeError("Canonical selected checkpoint config/source/seed identity changed")
    model.load_state_dict(chosen["model"])
    validation,logits=evaluate(model,parents["val"],device)
    check_live()
    nll=float(np.mean([row["nll"] for row in validation]))
    original=float(chosen["validation_nll"]);delta=nll-original
    if nll!=original:
        error=RuntimeError("Reloaded canonical checkpoint NLL differs under strict deterministic execution")
        error.replay_evidence={"original_selected_validation_nll":original,"reloaded_validation_nll":nll,
            "checkpoint_replay_nll_delta":delta,"deterministic_policy":binding.get("deterministic_policy")}
        raise error
    result={"configuration_id":configuration["id"],"configuration":configuration,"training_seed":seed,**binding,
        "parameters":sum(p.numel() for p in model.parameters()),"validation_nll":nll,"best_epoch":chosen["epoch"],
        "original_selected_validation_nll":original,"reloaded_validation_nll":nll,"checkpoint_replay_nll_delta":delta,
        "per_parent":validation,"history":history,"checkpoint":str(checkpoint),"checkpoint_sha256":shared.sha(checkpoint),
        "model_configuration":model.configuration_manifest(),"selection":shared.SELECTION,
        "input_scope":INPUT_SCOPE,"cross_tree_aggregation":AGGREGATION,
        "training_seconds":time.monotonic()-started,"ending_config_sha256":binding["protocol_sha256"],
        "ending_implementation_sha256":binding["implementation_sha256"],
        "peak_cuda_memory_bytes":torch.cuda.max_memory_allocated() if device.type=="cuda" else None}
    return result,logits


def select_family(records,config_bytes,expected_config_sha256):
    """Original byte identity is mandatory, never caller-mutated settings/JSON SHA."""
    settings=settings_from_original_bytes(config_bytes,expected_config_sha256)
    configurations={row["id"]:row for row in settings["configurations"]}
    expected={(name,seed) for name in configurations for seed in shared.SEEDS}
    ids=[(row["configuration_id"],row["training_seed"]) for row in records]
    if len(configurations)!=8 or any(type(seed) is not int for _,seed in ids) or len(ids)!=len(set(ids)) or set(ids)!=expected:
        raise ValueError("Exactly24 prescribed canonical configuration/seed starts required")
    summaries=[]
    for configuration in settings["configurations"]:
        shared.safe_id(configuration["id"])
        rows=[row for row in records if row["configuration_id"]==configuration["id"]]
        capacity=sum(p.numel() for p in factory(configuration,settings).parameters())
        for row in rows:
            if (row.get("protocol_sha256")!=expected_config_sha256 or row.get("configuration")!=configuration
                or row.get("parameters")!=capacity or row.get("implementation_sha256")!=settings["bound_implementation_sha256"]
                or row.get("source_artifact_sha256")!=settings["source_artifact_sha256"]
                or any(row.get(key)!=settings[key] for key in (*SOURCE_KEYS,"dataset_manifest_sha256","dataset_protocol_sha256"))):
                raise ValueError("Canonical family record differs from original config/spec/capacity/source/code bindings")
            if isinstance(row["validation_nll"],bool) or not np.isfinite(row["validation_nll"]) or row["validation_nll"]<0:
                raise ValueError("Finite nonnegative selected raw NLL required")
        summaries.append({"configuration_id":configuration["id"],"parameters":capacity,
            "mean_three_seed_nll":float(np.mean([row["validation_nll"] for row in rows]))})
    best=min(row["mean_three_seed_nll"] for row in summaries)
    tied=[row for row in summaries if row["mean_three_seed_nll"]<=best+settings["family_tie_tolerance_nats"]]
    return {"selected":min(tied,key=lambda row:(row["parameters"],row["configuration_id"])),"all_configurations":summaries}


def tensor_bytes(value):
    if isinstance(value,torch.Tensor):return value.numel()*value.element_size()
    if isinstance(value,dict):return sum(tensor_bytes(item) for item in value.values())
    if isinstance(value,(list,tuple)):return sum(tensor_bytes(item) for item in value)
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True);parser.add_argument("--expected-config-sha256",required=True)
    parser.add_argument("--output");parser.add_argument("--seed",type=int,default=101)
    parser.add_argument("--configurations",nargs="+");parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    parser.add_argument("--validate-config",action="store_true")
    args=parser.parse_args(argv);torch.set_num_threads(4)
    if args.seed not in shared.SEEDS:raise ValueError("Only fixed101/202/303 seeds permitted")
    if args.configurations:
        for name in args.configurations:shared.safe_id(name)
    config_path=canonical_path(args.config)
    settings,manifest,snapshot=capture_training_snapshot(config_path,args.expected_config_sha256)
    if snapshot.implementation_sha256!=_IMPORTED_PINS:raise RuntimeError("Canonical source changed after imports")
    names=args.configurations or [row["id"] for row in settings["configurations"]]
    if len(names)!=len(set(names)) or not set(names).issubset({row["id"] for row in settings["configurations"]}):
        raise ValueError("Distinct prescribed canonical configuration IDs required")
    # Completion/freeze/raw sources must pass before parent inputs, models or CUDA.
    started=time.monotonic();parents=prepare_verified_parents(manifest,snapshot)
    snapshot.check(full_sources=True)
    applied_policy=configure_deterministic_development()
    if applied_policy!=settings["deterministic_policy"]:raise RuntimeError("Applied policy differs from captured protocol")
    snapshot.shared.bind_runtime_configuration(config_path,settings)
    if args.validate_config:
        for row in settings["configurations"]:
            if row["id"] in names:
                model=factory(row,settings)
                print(json.dumps({"event":"canonical_bound_config_valid","configuration_id":row["id"],
                    "parameters":sum(p.numel() for p in model.parameters()),"config_sha256":args.expected_config_sha256,
                    "deterministic_policy":applied_policy}),flush=True)
        snapshot.check(full_sources=True);return
    if not args.output:parser.error("--output is required for fitting")
    require_device(args.device)
    snapshot.check(full_sources=True)
    output=canonical_path(args.output)
    if output.exists():raise RuntimeError("Choose a fresh canonical development run directory")
    identity=hashlib.sha256(str(output.relative_to(shared.ROOT)).encode()).hexdigest()[:16]
    checkpoint_root=canonical_path(Path("checkpoints/canonical_development_v1")/
        f"{output.name}_{identity}_{args.expected_config_sha256[:12]}_s{args.seed}")
    if checkpoint_root.exists():raise RuntimeError("Fresh canonical checkpoint namespace required")
    storage_estimate=sum(tensor_bytes(parent.prepared.as_dict()) for group in parents.values() for parent in group)
    require_storage(headroom_bytes=max(500_000_000,int(storage_estimate*1.2)+100_000_000))
    snapshot.check(full_sources=True)
    output.mkdir(parents=True);checkpoint_root.mkdir(parents=True)
    for name,path in (("protocol.json",config_path),("dataset_manifest.json",settings["dataset_manifest"]),
                      ("dataset_protocol.json",settings["dataset_protocol"]),("completion.json",settings["completion_receipt"])):
        (output/name).write_bytes(snapshot.bytes_for(path))
    receipts=[]
    for split,group in parents.items():
        for parent in group:
            filename=f"canonical_input_{split}_rep{parent.metadata['replicate']:03d}.pt"
            path=checkpoint_root/filename
            snapshot.check();torch.save(parent.prepared.as_dict(),path)
            receipts.append({"parent_id":parent.parent_id,"input_cache":str(path),"input_cache_sha256":shared.sha(path),
                "scoring_positions":len(parent.prepared.sequence.positions),"query_haplotypes":parent.prepared.sequence.num_queries,
                "tiles":len(parent.tiles),**parent.metadata})
    snapshot.check(full_sources=True)
    binding={"protocol_sha256":args.expected_config_sha256,"dataset_manifest_sha256":settings["dataset_manifest_sha256"],
        "dataset_protocol_sha256":settings["dataset_protocol_sha256"],"implementation_sha256":snapshot.implementation_sha256,
        "deterministic_policy":applied_policy,
        **{key:settings[key] for key in SOURCE_KEYS}}
    results={"stage":settings["stage"],"family":FAMILY,"development_namespace":"development-v1",
        "evaluation_split":"fresh_development_validation_only","training_seed":args.seed,"training_seeds":shared.SEEDS,
        "nll_policy":NLL_POLICY,"test_accessed":False,"family_selection_complete":False,**binding,
        "initial_implementation_sha256":snapshot.implementation_sha256,"source_artifact_sha256":snapshot.sources,
        "input_scope":INPUT_SCOPE,"cross_tree_aggregation":AGGREGATION,"preparation":receipts,
        "preparation_seconds":time.monotonic()-started,"models":{},"interpretation":settings["interpretation"],
        "environment":{"torch":str(torch.__version__),"cuda":torch.version.cuda,"device":args.device},
        "physical_gpu":os.environ.get("ARGGNN_PHYSICAL_GPU"),
        "integrity_checks":"shared exact control/source catalog plus canonical code pins; controls/code/stat identities through prep/configurations/epochs, full raw/prepared/metadata hashes at major boundaries"}
    dump_json(output/"preparation.json",results)
    active=None
    try:
        for configuration in settings["configurations"]:
            if configuration["id"] not in names:continue
            active=configuration["id"];snapshot.check(full_sources=True);require_storage(headroom_bytes=100_000_000)
            def callback(item):
                require_storage(headroom_bytes=100_000_000)
                print(json.dumps({"event":"epoch","configuration_id":active,"seed":args.seed,**item}),flush=True)
            checkpoint=checkpoint_root/(active+".pt")
            result,logits=train_configuration(configuration,settings,parents,torch.device(args.device),args.seed,
                checkpoint,binding,snapshot.check,callback)
            snapshot.check(full_sources=True)
            prediction=checkpoint_root/(active+"_validation_logits.pt")
            torch.save({"logits":logits,"parent_ids":[p.parent_id for p in parents["val"]],
                "positions":[p.prepared.sequence.positions for p in parents["val"]],
                "query_node_ids":[p.prepared.sequence.query_node_ids for p in parents["val"]],
                "query_haplotype_keys":[p.metadata["query_haplotype_keys"] for p in parents["val"]],
                "seed":args.seed,"configuration_id":active,"nll_policy":NLL_POLICY,**binding},prediction)
            result.update(validation_logits=str(prediction),validation_logits_sha256=shared.sha(prediction),
                source_artifact_sha256=snapshot.sources,ending_config_sha256=args.expected_config_sha256,
                ending_implementation_sha256=snapshot.implementation_sha256)
            results["models"][active]=result
            dump_json(output/(active+".json"),result);dump_json(output/"progress.json",results)
        snapshot.check(full_sources=True)
        results.update(ending_config_sha256=args.expected_config_sha256,ending_implementation_sha256=snapshot.implementation_sha256)
        dump_json(output/"results.json",results)
    except Exception as error:
        dump_json(output/"failure.json",{"configuration_id":active,"training_seed":args.seed,**binding,
            "exception_type":type(error).__name__,"reason":str(error),"test_accessed":False,
            **getattr(error,"replay_evidence",{})})
        raise
    print(json.dumps({"event":"canonical_development_seed_complete","output":str(output),"family_selection_complete":False}),flush=True)


if __name__=="__main__":main()
