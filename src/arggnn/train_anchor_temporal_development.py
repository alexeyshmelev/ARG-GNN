"""Exact-bound fresh64/16 temporal/smoother training. No final-test input path."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch

from .resources import ROOT,project_path,require_storage,set_research_title
set_research_title()
from .development_training_bindings import (
    checked_protocol,check_live,admit_run,no_test_path,snapshot,sha,SEEDS)
from .anchor_kernel import prepare_anchor_kernel_bank
from .anchor_temporal_baselines import (
    AnchorGRUModel,AnchorTemporalTransformerModel,ObservedReferenceCoordinateSmoother)
from .train_anchor_temporal import optimizer_groups,evaluate_temporal
from .train_anchor_development import require_device
from .train import set_seed,dump_json

_IMPORTED_PINS=snapshot()


def factory(name,settings,input_dim=47):
    spec=dict(settings["model_specs"][name])
    family=spec.pop("family")
    if family=="reference_smoother":
        return ObservedReferenceCoordinateSmoother(num_classes=4,num_anchors=32,**spec)
    constructor=AnchorGRUModel if family=="gru" else AnchorTemporalTransformerModel
    return constructor(input_dim,4,num_layers=2,latent_dim=8,num_scales=3,num_anchors=32,
        initial_coupling=[5.,20.,80.],max_attention_score_bytes=settings["max_attention_score_bytes"],**spec)


def captured_protocol(path,expected):
    path=no_test_path(path)
    content=Path(path).read_bytes()
    value=hashlib.sha256(content).hexdigest()
    if value!=expected:raise RuntimeError("Configuration differs from exact supplied freeze")
    settings=checked_protocol(path,value)
    if Path(path).read_bytes()!=content:raise RuntimeError("Configuration changed during capture")
    return content,value,settings


def checkpoint_directory(output,config_sha,seed):
    identity=hashlib.sha256(str(output.relative_to(ROOT)).encode()).hexdigest()[:16]
    return project_path(Path("checkpoints")/"temporal_development_v1"/f"{output.name}_{identity}_{config_sha[:12]}_s{seed}")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    parser.add_argument("--expected-config-sha256",required=True)
    parser.add_argument("--output")
    parser.add_argument("--seed",type=int,default=101)
    parser.add_argument("--models",nargs="+")
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    parser.add_argument("--validate-config",action="store_true")
    args=parser.parse_args(argv)
    torch.set_num_threads(4)
    path=no_test_path(args.config)
    content,config_sha,settings=captured_protocol(path,args.expected_config_sha256)
    if args.seed not in SEEDS:raise RuntimeError("Seed outside fixed101/202/303 development list")
    names=args.models or settings["models"]
    if len(names)!=len(set(names)) or any(n not in settings["models"] for n in names):
        raise RuntimeError("Select distinct names within the frozen family grid")
    initial=settings["bound_implementation_sha256"]
    if initial!=snapshot() or initial!=_IMPORTED_PINS:
        raise RuntimeError("Source changed after module imports or dependency lock differs")
    if args.validate_config:
        for name in names:
            model=factory(name,settings)
            print(json.dumps({"event":"bound_development_config_valid","model":name,
                "config_sha256":config_sha,"parameters":sum(p.numel() for p in model.parameters())}),flush=True)
        check_live(settings,path,config_sha)
        return
    if not args.output:parser.error("--output is required for fitting")
    completion,dataset,age_caches,source_lock=admit_run(settings,path,config_sha,imported_pins=_IMPORTED_PINS)
    # Complete cohort/config/dependency admission before any CUDA query.
    require_device(args.device)
    del age_caches
    cases={s:dataset.cases(s) for s in ("train","val")}
    output=project_path(args.output)
    if output.exists():raise RuntimeError("Choose a fresh development run directory")
    checkpoints=checkpoint_directory(output,config_sha,args.seed)
    if checkpoints.exists():raise RuntimeError("Checkpoint namespace already exists")
    require_storage(headroom_bytes=200_000_000)
    output.mkdir(parents=True);checkpoints.mkdir(parents=True)
    (output/"protocol.json").write_bytes(content)
    dump_json(output/"dataset_binding.json",{
        "completion_receipt":settings["completion_receipt"],
        "completion_receipt_sha256":settings["completion_receipt_sha256"],
        "source_dataset_manifest_sha256":settings["source_dataset_manifest_sha256"],
        "boundary_manifest_sha256":dataset.manifest_sha256,
        "boundary_preparation_sha256":dataset.preparation_sha256,
        "age_manifest_sha256":settings["age_dataset_manifest_sha256"],
        "parents":{s:[c.parent_id for c in cases[s]] for s in cases},"test_parents":0,
        "source_snapshot":source_lock.manifest()})
    device=torch.device(args.device)
    try:
        prepared={id(tile):prepare_anchor_kernel_bank(tile.bank,num_anchors=32,num_scales=3).to(device,torch.float32)
            for group in cases.values() for case in group for tile in case.tiles}
    except Exception as error:
        dump_json(output/"failure.json",{"stage":"bank_transfer","seed":args.seed,
            "protocol_sha256":config_sha,"initial_implementation_sha256":initial,
            "error_type":type(error).__name__,"reason":str(error),"test_access":False})
        raise
    tiles=[tile for case in cases["train"] for tile in case.tiles]
    results={"stage":settings["stage"],"development_namespace":"development-v1",
        "evaluation_split":"fresh_development_validation_only","training_seed":args.seed,
        "training_seeds":SEEDS,"family":settings["family"],"protocol_sha256":config_sha,
        "completion_receipt_sha256":settings["completion_receipt_sha256"],
        "boundary_dataset_manifest_sha256":dataset.manifest_sha256,
        "initial_implementation_sha256":initial,"selection":settings["selection"],
        "nll_policy":settings["nll_policy"],"models":{},
        "physical_gpu":os.environ.get("ARGGNN_PHYSICAL_GPU"),
        "interpretation":settings["interpretation"],
        "environment":{"torch":str(torch.__version__),"cuda":torch.version.cuda,"device":str(device)}}
    active=None
    try:
        for name in names:
            active=name;check_live(settings,path,config_sha,source_snapshot=source_lock,full_sources=True)
            set_seed(args.seed);model=factory(name,settings).to(device)
            parameters=list(model.parameters());started=time.monotonic();history=[]
            checkpoint=None;best_epoch=None
            if device.type=="cuda":torch.cuda.reset_peak_memory_stats()
            if parameters:
                optimizer=torch.optim.AdamW(optimizer_groups(model,settings["weight_decay"]),
                    lr=settings["model_learning_rates"][name])
                best=float("inf");stale=0;checkpoint=checkpoints/(name+".pt")
                for epoch in range(48):
                    check_live(settings,path,config_sha,source_snapshot=source_lock,full_sources=False);require_storage(headroom_bytes=100_000_000)
                    model.train();order=list(range(len(tiles)));random.shuffle(order);losses=[]
                    for index in order:
                        tile=tiles[index];logits=model(prepared[id(tile)])
                        loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),tile.labels.to(device).flatten())
                        if not bool(torch.isfinite(loss)):raise RuntimeError("Nonfinite development training loss")
                        optimizer.zero_grad(set_to_none=True);loss.backward()
                        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):
                            raise RuntimeError("Nonfinite development gradient")
                        torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step();losses.append(float(loss.detach()))
                    validation,_=evaluate_temporal(model,cases["val"],prepared)
                    nll=float(np.mean([r["nll"] for r in validation]))
                    history.append({"epoch":epoch,"train_nll":float(np.mean(losses)),
                        "val_nll":nll,"elapsed_seconds":time.monotonic()-started})
                    print(json.dumps({"event":"epoch","model":name,"seed":args.seed,**history[-1]}),flush=True)
                    if nll<best:
                        best,best_epoch,stale=nll,epoch,0
                        torch.save({"model":model.state_dict(),"epoch":epoch,"validation_nll":best,
                            "protocol_sha256":config_sha,"seed":args.seed,
                            "boundary_manifest_sha256":dataset.manifest_sha256,
                            "completion_receipt_sha256":settings["completion_receipt_sha256"],
                            "implementation_sha256":initial},checkpoint)
                    else:
                        stale+=1
                        if stale>=8:break
                saved=torch.load(checkpoint,map_location=device,weights_only=True)
                if saved["protocol_sha256"]!=config_sha:raise RuntimeError("Selected checkpoint config binding changed")
                model.load_state_dict(saved["model"])
            validation,logits=evaluate_temporal(model,cases["val"],prepared)
            prediction=checkpoints/(name+"_validation_logits.pt")
            torch.save({"logits":[v.double() for v in logits],"parent_ids":[c.parent_id for c in cases["val"]],
                "positions":[c.positions for c in cases["val"]],"query_node_ids":[c.query_node_ids for c in cases["val"]],
                "protocol_sha256":config_sha,"seed":args.seed,"nll_policy":settings["nll_policy"],
                "completion_receipt_sha256":settings["completion_receipt_sha256"]},prediction)
            ending=check_live(settings,path,config_sha,source_snapshot=source_lock,full_sources=True)
            result={"parameters":sum(p.numel() for p in parameters),"best_epoch":best_epoch,
                "learning_rate":settings["model_learning_rates"].get(name),
                "validation_nll":float(np.mean([r["nll"] for r in validation])),
                "per_parent":validation,"history":history,"training_seconds":time.monotonic()-started,
                "model_configuration":model.configuration_manifest(),
                "checkpoint":None if checkpoint is None else str(checkpoint),
                "checkpoint_sha256":None if checkpoint is None else sha(checkpoint),
                "validation_logits":str(prediction),"validation_logits_sha256":sha(prediction),
                "ending_config_sha256":config_sha,"ending_implementation_sha256":ending,
                "smoother_fitting":"none; seed-independent deterministic predictor" if not parameters else None,
                "peak_cuda_memory_bytes":torch.cuda.max_memory_allocated() if device.type=="cuda" else None}
            results["models"][name]=result
            dump_json(output/(name+".json"),result);dump_json(output/"progress.json",results)
            del model
            if parameters:del optimizer
            if device.type=="cuda":torch.cuda.empty_cache()
        results["ending_config_sha256"]=config_sha
        results["ending_implementation_sha256"]=check_live(settings,path,config_sha,source_snapshot=source_lock,full_sources=True)
        results["ending_input_artifact_sha256"]=source_lock.hashes
        dump_json(output/"results.json",results)
    except Exception as error:
        dump_json(output/"failure.json",{"model":active,"seed":args.seed,
            "protocol_sha256":config_sha,"initial_implementation_sha256":initial,
            "error_type":type(error).__name__,"reason":str(error),
            "phase":"fresh_development_validation_only","test_access":False})
        raise


if __name__=="__main__":main()
