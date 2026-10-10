"""CPU train/validation forecasting pilot, leaving final test unassessed."""
from __future__ import annotations
from .resources import ROOT,project_path,require_storage,set_research_title
set_research_title()
import argparse
from dataclasses import replace
import hashlib
import json
import random
import subprocess
import time
import numpy as np
import torch
from .chickenpox import (ARCHIVE_NAME,ForecastProtocol,TrainingScaler,forecast_sequence,
    load_chickenpox_archive,last_value_forecast,seasonal_forecast,LinearRidgeForecaster)
from .models import count_parameters
from .study_models import make_regression_model
from .train import dump_json,set_seed


def regression_metrics(prediction,truth):
    error=np.asarray(prediction,dtype=float)-np.asarray(truth,dtype=float)
    return {"mae_cases":float(np.abs(error).mean()),"rmse_cases":float(np.sqrt(np.square(error).mean())),
            "per_county_mae":np.abs(error).mean(axis=0).tolist(),"forecast_weeks":len(error)}


def no_geography(seq):
    node=torch.arange(len(seq.x))
    return replace(seq,edge_index=torch.stack((node,node)))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--seed",type=int,default=101)
    parser.add_argument("--models",nargs="+")
    args=parser.parse_args()
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError("This small-graph CPU runner requires CUDA_VISIBLE_DEVICES empty")
    require_storage(headroom_bytes=500_000_000)
    source=project_path(args.config)
    settings=json.loads(source.read_text())
    protocol=ForecastProtocol(**{k:settings[k] for k in ("history","train_end","validation_end","observations")})
    data=load_chickenpox_archive(project_path("data/raw/chickenpox")/ARCHIVE_NAME)
    if data.archive_sha256!=settings["source_archive_sha256"]:
        raise RuntimeError("Source/protocol hash mismatch")
    scaler=TrainingScaler.fit(data.counts,protocol.train_end)
    train=protocol.targets("train")
    val=protocol.targets("validation")
    output=project_path(args.output)
    if output.exists():
        raise RuntimeError("Use a new versioned run directory")
    output.mkdir(parents=True)
    (output/"protocol.json").write_text(source.read_text())
    results={"stage":settings["stage"],"training_seed":args.seed,
             "protocol_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
             "source_archive_sha256":data.archive_sha256,
             "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
             "input_kind":"fixed_graph_weekly_observations","evaluation_split":"validation_only",
             "interpretation":settings["interpretation"],"test_evaluated":False,"models":{}}
    naive={"last_value":last_value_forecast,"seasonal_lag52":seasonal_forecast}
    results["classical_controls"]={name:regression_metrics(np.stack([function(data.counts[t-protocol.history:t]) for t in val]),data.counts[val])
                                   for name,function in naive.items()}
    ridge=[]
    for alpha in settings["ridge_candidates"]:
        fitted=LinearRidgeForecaster.fit_training(data,protocol,scaler,alpha)
        score=regression_metrics(np.stack([fitted.predict(data.counts[t-protocol.history:t]) for t in val]),data.counts[val])
        ridge.append({"ridge":alpha,**score})
    results["ridge_validation_search"]={"candidates":ridge,"selected":min(ridge,key=lambda r:r["mae_cases"])}
    sequences={int(t):forecast_sequence(data,scaler,int(t),protocol.history) for t in np.concatenate((train,val))}
    standardized=scaler.transform(data.counts)
    targets={int(t):torch.tensor(standardized[t],dtype=torch.float32) for t in train}
    checkpoint_id=hashlib.sha256(str(output.relative_to(ROOT)).encode()).hexdigest()[:16]
    checkpoint_root=project_path("checkpoints")/(output.name+"_"+checkpoint_id)
    checkpoint_root.mkdir()
    for name in (args.models or settings["models"]):
        if name not in settings["models"]:
            raise ValueError("Model outside frozen protocol")
        set_seed(args.seed)
        model=make_regression_model(name,1,settings)
        optimizer=torch.optim.AdamW(model.parameters(),lr=settings["learning_rate"],weight_decay=settings["weight_decay"])
        prepared={t:no_geography(seq) for t,seq in sequences.items()} if name.endswith("_no_geography") else sequences
        best=float("inf")
        best_epoch=-1
        stale=0
        history=[]
        checkpoint=checkpoint_root/(name+".pt")
        started=time.monotonic()
        for epoch in range(settings["epochs"]):
            require_storage(headroom_bytes=50_000_000)
            model.train()
            order=list(map(int,train))
            random.shuffle(order)
            losses=[]
            for t in order:
                optimizer.zero_grad(set_to_none=True)
                prediction=model(prepared[t])[0,:,0]
                loss=torch.nn.functional.mse_loss(prediction,targets[t])
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite regression loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
                losses.append(float(loss.detach()))
            model.eval()
            with torch.no_grad():
                prediction=np.stack([scaler.inverse(model(prepared[int(t)])[0,:,0].numpy()) for t in val])
            metrics=regression_metrics(prediction,data.counts[val])
            record={"epoch":epoch,"train_mse_standardized":float(np.mean(losses)),**metrics,
                    "elapsed_seconds":time.monotonic()-started}
            history.append(record)
            print(json.dumps({"event":"epoch","model":name,"seed":args.seed,**record}),flush=True)
            if metrics["mae_cases"]<best:
                best,best_epoch,stale=metrics["mae_cases"],epoch,0
                torch.save({"model":model.state_dict(),"epoch":epoch,"validation_mae":best,
                            "protocol_sha256":results["protocol_sha256"]},checkpoint)
            else:
                stale+=1
                if stale>=settings["patience"]:
                    break
        model.load_state_dict(torch.load(checkpoint,weights_only=True)["model"])
        model.eval()
        with torch.no_grad():
            prediction=np.stack([scaler.inverse(model(prepared[int(t)])[0,:,0].numpy()) for t in val])
        saved=checkpoint_root/(name+"_validation_predictions.npz")
        np.savez_compressed(saved,prediction=prediction,target_indices=val)
        result={"parameters":count_parameters(model),"best_epoch":best_epoch,
                "training_seconds":time.monotonic()-started,"history":history,
                "checkpoint":str(checkpoint),"checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "validation_predictions":str(saved),"validation_predictions_sha256":hashlib.sha256(saved.read_bytes()).hexdigest(),
                **regression_metrics(prediction,data.counts[val])}
        results["models"][name]=result
        dump_json(output/(name+".json"),result)
        dump_json(output/"progress.json",results)
    dump_json(output/"results.json",results)
    print(json.dumps({"event":"run_complete","output":str(output)}),flush=True)


if __name__=="__main__":
    main()
