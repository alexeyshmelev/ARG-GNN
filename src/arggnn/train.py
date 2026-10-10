"""Run bounded oracle-pilot experiments. No publication claims are generated."""
from __future__ import annotations
from .resources import ROOT, project_path, require_storage, set_research_title
set_research_title()
import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from .metrics import classification_metrics, replicate_summary
from .models import build_model, count_parameters
from .simulate import load_replicate, split_fixed_windows

def make_model(name, input_dim, num_classes, settings):
    kwargs = dict(input_dim=input_dim,num_classes=num_classes,
                  hidden_dim=settings["hidden_dim"],num_layers=settings["num_layers"],validate=False)
    if name in {"late_transformer","supra_mpnn"}:
        from .temporal_baselines import LeafTemporalTransformer, LeafCorrespondenceSupraMPNN
        cls = LeafTemporalTransformer if name == "late_transformer" else LeafCorrespondenceSupraMPNN
        return cls(**kwargs)
    return build_model(name,**kwargs)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate(model, cases, device, windows):
    model.eval()
    records=[]
    with torch.no_grad():
        for case in cases:
            probabilities=[]
            for tile in split_fixed_windows(case,windows):
                probabilities.append(model(tile.sequence.to(device)).softmax(-1).cpu())
            p=torch.cat(probabilities)
            metrics=classification_metrics(p,case.labels)
            records.append({"replicate":case.metadata["replicate"],**metrics})
    return records

def dump_json(path,value):
    path=project_path(path)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2)+"\n")
    temporary.replace(path)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    parser.add_argument("--data",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--seed",type=int,default=101)
    parser.add_argument("--models",nargs="+")
    parser.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    args=parser.parse_args()
    torch.set_num_threads(4)
    if args.device=="cuda":
        if os.environ.get("ARGGNN_PHYSICAL_GPU") not in {"2","3","4","5"} or not os.environ.get("ARGGNN_GPU_LEASE"):
            raise RuntimeError("GPU training requires the project resource launcher")
        if torch.cuda.device_count()!=1:
            raise RuntimeError("A job must see exactly one reserved GPU")
        torch.cuda.reset_peak_memory_stats()
    else:
        if torch.cuda.is_available():
            raise RuntimeError("CPU checks must set CUDA_VISIBLE_DEVICES empty")
    device=torch.device(args.device)
    settings=json.loads(project_path(args.config).read_text())
    data_dir=project_path(args.data)
    manifest=json.loads((data_dir/"manifest.json").read_text())
    config_sha=hashlib.sha256(project_path(args.config).read_bytes()).hexdigest()
    if manifest["protocol_sha256"]!=config_sha:
        raise RuntimeError("Dataset and training protocol hashes differ")
    output=project_path(args.output)
    if (output/"results.json").exists():
        raise RuntimeError("Completed output exists. Choose a fresh versioned run.")
    output.mkdir(parents=True,exist_ok=True)
    cases={split:[load_replicate(r["directory"]) for r in manifest["replicates"] if r["split"]==split]
           for split in ["train","val","test"]}
    if any(not v for v in cases.values()):
        raise RuntimeError("Train, validation and test must all be nonempty")
    for split,values in cases.items():
        if any(c.metadata["split"]!=split for c in values):
            raise RuntimeError("Manifest/replicate split mismatch")
    windows=settings["windows"]
    train_tiles=[tile for c in cases["train"] for tile in split_fixed_windows(c,windows)]
    example=cases["train"][0]
    selected=args.models or settings["models"]
    results={"stage":settings["stage"],"interpretation":settings["interpretation"],
             "input_kind":"oracle_true_genealogies","evaluation":settings["evaluation"],
             "training_seed":args.seed,"protocol_sha256":config_sha,
             "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
             "physical_gpu":os.environ.get("ARGGNN_PHYSICAL_GPU"),
             "environment":{"torch":torch.__version__,"cuda":torch.version.cuda},
             "models":{}}
    nearest=[classification_metrics(c.baseline_probabilities,c.labels) for c in cases["test"]]
    results["nearest_tmrca"]={"per_replicate":nearest,**replicate_summary(nearest,args.seed)}
    checkpoint_root=project_path(Path("checkpoints")/output.name)
    checkpoint_root.mkdir(parents=True,exist_ok=True)
    for name in selected:
        set_seed(args.seed)
        model=make_model(name,example.sequence.x.shape[1],example.metadata["num_classes"],settings).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=settings["learning_rate"],weight_decay=settings["weight_decay"])
        best_nll=float("inf")
        best_epoch=-1
        stale=0
        history=[]
        started=time.monotonic()
        checkpoint=checkpoint_root/(name+".pt")
        if device.type=="cuda":
            torch.cuda.reset_peak_memory_stats()
        for epoch in range(settings["epochs"]):
            require_storage(headroom_bytes=100_000_000)
            model.train()
            order=list(range(len(train_tiles)))
            random.shuffle(order)
            losses=[]
            for index in order:
                tile=train_tiles[index]
                seq=tile.sequence.to(device)
                y=tile.labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits=model(seq)
                loss=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1))
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            val=evaluate(model,cases["val"],device,windows)
            val_nll=float(np.mean([r["nll_uniform_grid"] for r in val]))
            record={"epoch":epoch,"train_nll":float(np.mean(losses)),
                    "val_nll":val_nll,"val_accuracy":float(np.mean([r["accuracy_uniform_grid"] for r in val])),
                    "elapsed_seconds":time.monotonic()-started}
            history.append(record)
            print(json.dumps({"event":"epoch","model":name,"seed":args.seed,**record}),flush=True)
            if val_nll<best_nll:
                best_nll=val_nll
                best_epoch=epoch
                stale=0
                torch.save({"model":model.state_dict(),"epoch":epoch,"validation_nll":val_nll},checkpoint)
            else:
                stale+=1
                if stale>=settings["patience"]:
                    break
        chosen=torch.load(checkpoint,map_location=device,weights_only=True)
        model.load_state_dict(chosen["model"])
        tests=evaluate(model,cases["test"],device,windows)
        result={"parameters":count_parameters(model),"best_epoch":best_epoch,"validation_nll":best_nll,
                "training_seconds":time.monotonic()-started,"history":history,"per_replicate":tests,
                "peak_cuda_memory_bytes":torch.cuda.max_memory_allocated() if device.type=="cuda" else None,
                **replicate_summary(tests,args.seed)}
        results["models"][name]=result
        dump_json(output/(name+".json"),result)
        dump_json(output/"progress.json",results)
        print(json.dumps({"event":"model_complete","model":name,"seed":args.seed,"accuracy":result["mean_accuracy"]}),flush=True)
        del model,optimizer
        if device.type=="cuda":
            torch.cuda.empty_cache()
    dump_json(output/"results.json",results)
    print(json.dumps({"event":"run_complete","output":str(output),"storage_bytes":require_storage()}),flush=True)

if __name__=="__main__":
    main()
