"""Training/validation-only common-input study. Test evaluation is separate."""
from __future__ import annotations
from .resources import ROOT,project_path,require_storage,set_research_title
set_research_title()
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time
import numpy as np
import torch
from .data import GraphSequence
from .metrics import classification_metrics,replicate_summary
from .models import count_parameters
from .simulate import load_replicate,split_fixed_windows
from .study_models import make_study_model
from .train import set_seed,dump_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def grid_sequence_metrics(p,labels):
    result=classification_metrics(p,labels)
    pred=p.argmax(-1)
    true_edges=labels[1:]!=labels[:-1]
    pred_edges=pred[1:]!=pred[:-1]
    # Maximum-cardinality ordered matching within one fixed grid interval.
    # This is a grid diagnostic, not a full-resolution tract metric.
    matches=0
    for q in range(labels.shape[1]):
        actual=torch.nonzero(true_edges[:,q],as_tuple=False).flatten().tolist()
        predicted=torch.nonzero(pred_edges[:,q],as_tuple=False).flatten().tolist()
        i=j=0
        while i<len(actual) and j<len(predicted):
            if abs(actual[i]-predicted[j])<=1:
                matches+=1
                i+=1
                j+=1
            elif predicted[j]<actual[i]-1:
                j+=1
            else:
                i+=1
    true_count=int(true_edges.sum())
    pred_count=int(pred_edges.sum())
    result.update(grid_true_switches=true_count,grid_predicted_switches=pred_count,
                  grid_boundary_matches_tolerance_one=matches,
                  grid_boundary_precision=matches/pred_count if pred_count else None,
                  grid_boundary_recall=matches/true_count if true_count else None,
                  grid_boundary_f1=2*matches/(true_count+pred_count) if true_count+pred_count else None,
                  mean_absolute_switch_count_error=float((true_edges.sum(0)-pred_edges.sum(0)).abs().float().mean()))
    return result


def evaluate(model,cases,device,windows):
    model.eval()
    records=[]
    predictions=[]
    with torch.no_grad():
        for case in cases:
            probabilities=[]
            for tile in split_fixed_windows(case,windows):
                probabilities.append(model(tile.sequence.to(device)).softmax(-1).cpu())
            p=torch.cat(probabilities)
            if p.shape[:2] != case.labels.shape:
                raise RuntimeError("Fixed-grid prediction/label shape mismatch")
            records.append({"replicate":case.metadata["replicate"],**grid_sequence_metrics(p,case.labels)})
            predictions.append(p)
    return records,predictions


def load_cases(manifest,split):
    cases=[]
    for record in manifest["replicates"]:
        if record["split"]!=split:
            continue
        directory=project_path(record["directory"])
        if (directory/"tensors.pt.gz").resolve()!=project_path(record["tensor_file"]):
            raise RuntimeError("Directory is not bound to the verified raw tensor")
        if (directory/"metadata.json").resolve()!=project_path(record["metadata"]):
            raise RuntimeError("Directory is not bound to the verified parent metadata")
        if sha(project_path(record["metadata"]))!=record["metadata_sha256"]:
            raise RuntimeError("Parent metadata integrity failure")
        if sha(project_path(record["tensor_file"]))!=record["tensor_sha256"]:
            raise RuntimeError("Raw tensor integrity failure")
        cached=project_path(record["prepared_tensor_file"])
        if sha(cached)!=record["prepared_tensor_sha256"]:
            raise RuntimeError("Prepared tensor integrity failure")
        case=load_replicate(directory)
        if case.metadata["replicate"]!=record["replicate"] or case.metadata["seeds"]!=record["seeds"]:
            raise RuntimeError("Parent replicate/seed identity mismatch")
        if case.metadata["split"]!=split:
            raise RuntimeError("Parent split mismatch")
        bundle=torch.load(cached,map_location="cpu",weights_only=True)
        if bundle["adapter"]["source_tensor_sha256"]!=record["tensor_sha256"]:
            raise RuntimeError("Prepared inputs are not derived from this raw parent")
        if bundle["adapter"]["configuration_sha256"]!=record["adapter_configuration_sha256"]:
            raise RuntimeError("Adapter configuration mismatch")
        from .reference_encoder import ReferenceFeatureAdapter
        adapter=ReferenceFeatureAdapter(case.sequence.x.shape[1],**manifest["protocol"]["reference_adapter"])
        expected=adapter.manifest_provenance(case.metadata["feature_names"],record["tensor_sha256"])
        if expected["configuration_sha256"]!=bundle["adapter"]["configuration_sha256"]:
            raise RuntimeError("Prepared adapter differs from the frozen protocol")
        prepared=GraphSequence.from_dict(bundle["sequence"])
        if prepared.x.shape[1]!=adapter.output_dim or not torch.equal(prepared.x[:,:adapter.input_dim],case.sequence.x):
            raise RuntimeError("Prepared original feature prefix differs from raw parent")
        for field,value in case.sequence.as_dict().items():
            if field=="x":
                continue
            actual=getattr(prepared,field)
            if (value is None)!=(actual is None) or (value is not None and not torch.equal(value,actual)):
                raise RuntimeError(f"Prepared {field} differs from raw parent")
        case.sequence=prepared
        cases.append(case)
    if not cases:
        raise ValueError(f"No {split} parents")
    return cases


def verify_reused_dataset_protocol(settings,manifest):
    """Bind data reuse explicitly while permitting a separate frozen model study."""
    if settings.get("dataset_source_protocol_sha256")!=manifest["protocol_sha256"]:
        raise RuntimeError("Dataset source protocol hash is not explicitly bound")
    training_fields={"stage","models","hidden_dim","num_layers","epochs","patience",
                     "learning_rate","weight_decay","dropout","selection","evaluation",
                     "interpretation","comparison_note","dataset_source_protocol_sha256",
                     "model_options"}
    source=manifest["protocol"]
    for key in (set(settings)|set(source))-training_fields:
        if settings.get(key)!=source.get(key):
            raise RuntimeError(f"Reused dataset field changed: {key}")


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
        if os.environ.get("ARGGNN_PHYSICAL_GPU") not in {"2","3","4","5"}:
            raise RuntimeError("Use project GPU launcher")
        lease=project_path(os.environ["ARGGNN_GPU_LEASE"])
        fd=int(os.environ["ARGGNN_GPU_LEASE_FD"])
        if os.fstat(fd).st_ino!=lease.stat().st_ino:
            raise RuntimeError("GPU lease descriptor mismatch")
        if torch.cuda.device_count()!=1:
            raise RuntimeError("Must see exactly one reserved GPU")
    elif torch.cuda.is_available():
        raise RuntimeError("CPU runs require CUDA_VISIBLE_DEVICES empty")
    device=torch.device(args.device)
    config_path=project_path(args.config)
    settings=json.loads(config_path.read_text())
    manifest_path=project_path(args.data)/"manifest.json"
    manifest=json.loads(manifest_path.read_text())
    config_sha=sha(config_path)
    if manifest["protocol_sha256"]!=config_sha:
        verify_reused_dataset_protocol(settings,manifest)
    if settings["splits"].get("test",0)!=0:
        raise RuntimeError("This study runner permits train/validation-only protocols")
    output=project_path(args.output)
    if output.exists():
        raise RuntimeError("Choose a fresh run directory")
    output.mkdir(parents=True)
    (output/"dataset_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (output/"protocol.json").write_text(config_path.read_text())
    cases={split:load_cases(manifest,split) for split in ("train","val")}
    windows=settings["windows"]
    tiles=[t for c in cases["train"] for t in split_fixed_windows(c,windows)]
    nearest=[{"replicate":c.metadata["replicate"],**grid_sequence_metrics(c.baseline_probabilities,c.labels)}
             for c in cases["val"]]
    results={"stage":settings["stage"],"interpretation":settings["interpretation"],
             "input_kind":settings["benchmark"],"evaluation_split":"validation_only",
             "training_seed":args.seed,"protocol_sha256":config_sha,
             "dataset_source_protocol_sha256":manifest["protocol_sha256"],
             "dataset_manifest_sha256":sha(manifest_path),"dataset_directory":str(manifest_path.parent),
             "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
             "physical_gpu":os.environ.get("ARGGNN_PHYSICAL_GPU"),
             "environment":{"torch":torch.__version__,"cuda":torch.version.cuda},
             "nearest_tmrca_validation":{"per_replicate":nearest,**replicate_summary(nearest,args.seed)},
             "models":{}}
    checkpoint_id=hashlib.sha256(str(output.relative_to(ROOT)).encode()).hexdigest()[:16]
    checkpoint_root=project_path(Path("checkpoints")/(output.name+"_"+checkpoint_id))
    checkpoint_root.mkdir(parents=True)
    example=cases["train"][0]
    for name in (args.models or settings["models"]):
        if name not in settings["models"]:
            raise ValueError("Model is not in frozen protocol")
        set_seed(args.seed)
        model=make_study_model(name,example.sequence.x.shape[1],example.metadata["num_classes"],settings).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=settings["learning_rate"],weight_decay=settings["weight_decay"])
        best_nll=float("inf")
        best_epoch=-1
        stale=0
        history=[]
        checkpoint=checkpoint_root/(name+".pt")
        started=time.monotonic()
        if device.type=="cuda":
            torch.cuda.reset_peak_memory_stats()
        for epoch in range(settings["epochs"]):
            require_storage(headroom_bytes=100_000_000)
            model.train()
            order=list(range(len(tiles)))
            random.shuffle(order)
            losses=[]
            for index in order:
                tile=tiles[index]
                seq=tile.sequence.to(device)
                y=tile.labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits=model(seq)
                loss=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1))
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
                losses.append(float(loss.detach()))
            val,_=evaluate(model,cases["val"],device,windows)
            nll=float(np.mean([r["nll_uniform_grid"] for r in val]))
            record={"epoch":epoch,"train_nll":float(np.mean(losses)),"val_nll":nll,
                    "val_accuracy":float(np.mean([r["accuracy_uniform_grid"] for r in val])),
                    "elapsed_seconds":time.monotonic()-started}
            history.append(record)
            print(json.dumps({"event":"epoch","model":name,"seed":args.seed,**record}),flush=True)
            if nll<best_nll:
                best_nll,best_epoch,stale=nll,epoch,0
                torch.save({"model":model.state_dict(),"epoch":epoch,"validation_nll":nll,
                            "protocol_sha256":config_sha,"dataset_manifest_sha256":sha(manifest_path)},checkpoint)
            else:
                stale+=1
                if stale>=settings["patience"]:
                    break
        chosen=torch.load(checkpoint,map_location=device,weights_only=True)
        model.load_state_dict(chosen["model"])
        val,predictions=evaluate(model,cases["val"],device,windows)
        prediction_path=checkpoint_root/(name+"_validation_predictions.pt")
        torch.save({"probabilities":predictions,"parent_replicates":[c.metadata["replicate"] for c in cases["val"]],
                    "protocol_sha256":config_sha},prediction_path)
        result={"parameters":count_parameters(model),"best_epoch":best_epoch,"validation_nll":best_nll,
                "training_seconds":time.monotonic()-started,"history":history,"per_replicate":val,
                "checkpoint":str(checkpoint),"checkpoint_sha256":sha(checkpoint),
                "validation_predictions":str(prediction_path),"validation_predictions_sha256":sha(prediction_path),
                "peak_cuda_memory_bytes":torch.cuda.max_memory_allocated() if device.type=="cuda" else None,
                **replicate_summary(val,args.seed)}
        results["models"][name]=result
        dump_json(output/(name+".json"),result)
        dump_json(output/"progress.json",results)
        print(json.dumps({"event":"model_complete","model":name,"validation_accuracy":result["mean_accuracy"]}),flush=True)
        del model,optimizer
        if device.type=="cuda":
            torch.cuda.empty_cache()
    dump_json(output/"results.json",results)
    print(json.dumps({"event":"run_complete","output":str(output),"storage_bytes":require_storage()}),flush=True)


if __name__=="__main__":
    main()
