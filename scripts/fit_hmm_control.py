"""Select a classical control using recorded validation predictions only."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from arggnn.resources import project_path,require_storage
from arggnn.simulate import load_replicate
from arggnn.temporal_hmm import markov_smooth
from arggnn.train_study import grid_sequence_metrics,sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",required=True)
    parser.add_argument("--model",default="feature_mlp")
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    require_storage(headroom_bytes=100_000_000)
    run=project_path(args.run)
    destination=project_path(args.output)
    if destination.exists():
        raise RuntimeError("Versioned output already exists")
    manifest=json.loads((run/"dataset_manifest.json").read_text())
    source=json.loads((run/(args.model+".json")).read_text())
    prediction_path=project_path(source["validation_predictions"])
    if sha(prediction_path)!=source["validation_predictions_sha256"]:
        raise RuntimeError("Prediction integrity failure")
    bundle=torch.load(prediction_path,weights_only=True,map_location="cpu")
    train=[]
    val=[]
    for record in manifest["replicates"]:
        if record["split"] not in {"train","val"}:
            continue
        directory=project_path(record["directory"])
        if (directory/"tensors.pt.gz").resolve()!=project_path(record["tensor_file"]) or sha(record["tensor_file"])!=record["tensor_sha256"]:
            raise RuntimeError("Parent data binding failure")
        case=load_replicate(directory)
        (train if record["split"]=="train" else val).append(case)
    if [c.metadata["replicate"] for c in val]!=bundle["parent_replicates"]:
        raise RuntimeError("Prediction/parent order mismatch")
    k=train[0].metadata["num_classes"]
    counts=sum((torch.bincount(c.labels.flatten(),minlength=k) for c in train),torch.zeros(k,dtype=torch.long))
    prior=(counts.double()+1)/(counts.sum()+k)
    trials=[]
    for rate in (0.,.1,.3,1.,3.,10.,30.,100.):
        for temperature in (.5,1.,2.,4.):
            metrics=[{"replicate":c.metadata["replicate"],**grid_sequence_metrics(markov_smooth(p,c.sequence.positions,prior,rate,temperature),c.labels)}
                     for p,c in zip(bundle["probabilities"],val)]
            trials.append({"rate_per_normalized_chromosome":rate,"temperature":temperature,
                           "mean_validation_nll":float(np.mean([r["nll_uniform_grid"] for r in metrics])),
                           "mean_validation_accuracy":float(np.mean([r["accuracy_uniform_grid"] for r in metrics])),
                           "per_replicate":metrics})
    chosen=min(trials,key=lambda r:r["mean_validation_nll"])
    result={"stage":"validation_only_classical_smoothing_control","source_model":args.model,
            "source_run":str(run),"prediction_sha256":sha(prediction_path),
            "training_class_counts":counts.tolist(),"training_class_prior":prior.tolist(),
            "context":"entire retained fixed-coordinate parent grid, larger than neural fixed-window context",
            "selection":"minimum mean parent-validation NLL over32 prespecified candidates",
            "final_test_evaluated":False,"chosen":chosen,"trials":trials}
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"output":str(destination),"chosen_rate":chosen["rate_per_normalized_chromosome"],"temperature":chosen["temperature"],"validation_accuracy":chosen["mean_validation_accuracy"]}),flush=True)


if __name__=="__main__":
    main()
