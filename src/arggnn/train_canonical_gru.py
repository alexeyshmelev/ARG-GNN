"""Dedicated canonical-tree GNN+GRU train/validation pilot, with no test access."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import re
import subprocess
import time

import numpy as np
import torch

from .canonical_gru import (CanonicalPreparedParent,CanonicalTile,CanonicalTreeGNNGRU,
                            canonical_fixed_tiles,prepare_canonical_parent)
from .metrics import classification_metrics,replicate_summary
from .confirmatory_metrics import logit_metrics
from .resources import ROOT,project_path,require_storage,set_research_title
from .train import dump_json,set_seed
from .train_study import load_cases


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def implementation_paths():
    names = ("train_canonical_gru","canonical_gru","tree_canonicalization","reference_encoder",
             "genealogy_baselines","models","data","train_study","simulate","metrics","confirmatory_metrics","train","resources")
    return {f"src/arggnn/{name}.py":Path(__file__).with_name(name+".py") for name in names}


def implementation_hashes():
    return {name:sha(path) for name,path in implementation_paths().items()}


def safe_component(value,label="ID"):
    """Accept one bounded ASCII filename component before constructing any path."""
    if not isinstance(value,str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}",value) is None:
        raise ValueError(f"Unsafe {label}: require1-128 ASCII letters/digits/underscores/hyphens, starting with letter/digit")
    return value


def require_sha256(value,label):
    if not isinstance(value,str) or re.fullmatch(r"[0-9a-f]{64}",value) is None:
        raise ValueError(f"Invalid {label}: require lowercase SHA256")
    return value


def pilot_path(value):
    """Reject held-out namespaces before opening config/manifest/source files."""
    raw=Path(value)
    def forbidden(parts):
        return any(part.lower()=="test" or any(token in part.lower() for token in
            ("confirmatory","confirmation","final_test","final-test")) for part in parts)
    if forbidden(raw.parts):raise ValueError("Held-out paths are forbidden in canonical train/validation")
    path=project_path(raw)
    if forbidden(path.relative_to(ROOT).parts):
        raise ValueError("Held-out paths are forbidden in canonical train/validation")
    return path


def verify_protocol(settings,manifest,manifest_path,manifest_sha256=None):
    manifest_path=pilot_path(manifest_path)
    for configuration in settings["configurations"]:
        safe_component(configuration["id"],"configuration ID")
    parent_ids=[]
    for record in manifest["replicates"]:
        parent_ids.append((record["split"],safe_component(str(record["replicate"]),"parent ID")))
    if len(parent_ids)!=len(set(parent_ids)):
        raise ValueError("Duplicate split/parent IDs would overwrite input caches")
    if (manifest_sha256 if manifest_sha256 is not None else sha(manifest_path))!=settings["dataset"]["manifest_sha256"] or manifest["protocol_sha256"]!=settings["dataset"]["protocol_sha256"]:
        raise RuntimeError("Canonical pilot dataset hash binding failed")
    if pilot_path(manifest_path).parent!=pilot_path(settings["dataset"]["directory"]):
        raise RuntimeError("Canonical pilot dataset directory differs from frozen source")
    source = manifest["protocol"]
    for key in ("reference_adapter","windows","splits"):
        if settings[key]!=source[key]:
            raise RuntimeError(f"Canonical pilot changed observed dataset field: {key}")
    if source["benchmark"]!=settings["benchmark"] or source["simulation"]["query_diploids"]*2!=settings["num_queries"]:
        raise RuntimeError("Canonical pilot source/Q64 schema mismatch")
    if settings["splits"].get("test",0)!=0 or any(r["split"] not in {"train","val"} for r in manifest["replicates"]):
        raise RuntimeError("Canonical pilot accepts train/validation only, no test records")
    for split in ("train","val"):
        if sum(r["split"]==split for r in manifest["replicates"])!=settings["splits"][split]:
            raise RuntimeError("Parent split counts differ from frozen protocol")
    if settings["base_feature_count"]!=8 or settings["input_dim"]!=48 or settings["num_classes"]!=4 or settings["canonicalization"]!={"merge_identical_adjacent":True,"suppress_anonymous_unary":True,"fresh_reference_adapter_after_canonicalization":True}:
        raise RuntimeError("Canonical preprocessing/source prefix schema differs from protocol")
    ids=[v["id"] for v in settings["configurations"]]
    tuples={(v["hidden_dim"],v["dropout"],v["learning_rate"]) for v in settings["configurations"]}
    if len(ids)!=8 or len(set(ids))!=8 or tuples!=set(itertools.product((32,64),(0.,.1),(.0003,.001))):
        raise RuntimeError("Canonical family requires the frozen8 width/dropout/LR configurations")
    if settings["training_seeds"]!=[101,202,303] or settings["num_layers"]!=2 or settings["temporal_layers"]!=2 or not 1<=settings["epochs"]<=48 or settings["patience"]!=8:
        raise RuntimeError("Canonical family tuning budget/depth/seeds differ from frozen design")


def source_artifacts(manifest):
    """Only the validated TRAIN/VAL source artifacts, never test paths."""
    paths={}
    for record in manifest["replicates"]:
        if record["split"] not in {"train","val"}:
            raise RuntimeError("Source snapshots permit train/validation only")
        safe_component(str(record["replicate"]),"parent ID")
        for field,hash_field in (("metadata","metadata_sha256"),("tensor_file","tensor_sha256"),
                                 ("prepared_tensor_file","prepared_tensor_sha256")):
            path=pilot_path(record[field])
            digest=require_sha256(record[hash_field],hash_field)
            if str(path) in paths and paths[str(path)]!=digest:
                raise RuntimeError("Conflicting source artifact hash bindings")
            paths[str(path)]=digest
    return paths


def source_signature(path):
    stat=Path(path).stat()
    return (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)


@dataclass(frozen=True)
class RunSnapshot:
    config_path: Path
    manifest_path: Path
    config_bytes: bytes
    manifest_bytes: bytes
    config_sha256: str
    manifest_sha256: str
    implementation_sha256: dict
    source_sha256: dict
    source_signatures: dict

    def check(self,full_sources=False):
        """Fail closed on changed inputs/code; hash sources at major boundaries."""
        try:
            if sha(self.config_path)!=self.config_sha256:
                raise RuntimeError("Run integrity failure: config changed")
            if sha(self.manifest_path)!=self.manifest_sha256:
                raise RuntimeError("Run integrity failure: dataset manifest changed")
            if implementation_hashes()!=self.implementation_sha256:
                raise RuntimeError("Run integrity failure: implementation/dependency source changed")
            for path,digest in self.source_sha256.items():
                if source_signature(path)!=self.source_signatures[path]:
                    raise RuntimeError("Run integrity failure: dataset source artifact changed")
                if full_sources:
                    if sha(path)!=digest:
                        raise RuntimeError("Run integrity failure: dataset source artifact hash changed")
                    if source_signature(path)!=self.source_signatures[path]:
                        raise RuntimeError("Run integrity failure: dataset source artifact changed during hashing")
        except OSError as error:
            raise RuntimeError("Run integrity failure: a frozen artifact is unavailable") from error


def capture_run_snapshot(config_path,manifest_path,expected_config_sha256=None):
    """Capture exact bytes and reviewed hashes before outputs or preprocessing."""
    config_path=pilot_path(config_path)
    manifest_path=pilot_path(manifest_path)
    config_bytes=config_path.read_bytes()
    config_sha256=hashlib.sha256(config_bytes).hexdigest()
    if expected_config_sha256 is not None:
        require_sha256(expected_config_sha256,"expected config SHA256")
        if config_sha256!=expected_config_sha256:
            raise RuntimeError("Supplied expected config SHA256 differs from original config")
    manifest_bytes=manifest_path.read_bytes()
    manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest()
    settings=json.loads(config_bytes)
    manifest=json.loads(manifest_bytes)
    verify_protocol(settings,manifest,manifest_path,manifest_sha256)
    implementations=implementation_hashes()
    expected=settings.get("implementation_sha256")
    if not isinstance(expected,dict) or expected!=implementations:
        raise RuntimeError("Reviewed implementation/dependency hash binding failed")
    for name,digest in expected.items(): require_sha256(digest,name)
    sources=source_artifacts(manifest)
    if settings["dataset"].get("source_artifact_sha256")!=sources:
        raise RuntimeError("Reviewed raw/prepared/metadata source hash binding failed")
    snapshot=RunSnapshot(config_path,manifest_path,config_bytes,manifest_bytes,config_sha256,
        manifest_sha256,implementations,sources,{path:source_signature(path) for path in sources})
    snapshot.check(full_sources=True)
    return snapshot


@dataclass
class CanonicalTrainingParent:
    prepared: CanonicalPreparedParent
    tiles: list[CanonicalTile]
    labels: torch.Tensor         # external to preprocessing/model input
    parent_id: str
    metadata: dict


def bind_training_parent(prepared,labels,metadata,settings):
    """Join external targets only AFTER the observed-only preparation stage."""
    parent_id=safe_component(str(metadata["replicate"]),"parent ID")
    sequence=prepared.sequence
    if labels.shape!=(sequence.positions.numel(),sequence.num_queries) or metadata["num_classes"]!=settings["num_classes"] or sequence.num_queries!=settings["num_queries"]:
        raise RuntimeError("External labels/head/query schema mismatch")
    if sequence.x.shape[-1]!=settings["input_dim"]:
        raise RuntimeError("Canonical reference feature schema differs from frozen F48")
    return CanonicalTrainingParent(prepared,canonical_fixed_tiles(prepared,settings["windows"]),labels,
        parent_id,{"split":metadata["split"],"num_classes":metadata["num_classes"]})


def logits_metrics(logits,labels):
    logits=logits.detach().cpu().double()
    labels=labels.detach().cpu().long()
    if logits.shape[:-1]!=labels.shape or not torch.isfinite(logits).all():
        raise ValueError("Finite aligned logits and labels required")
    logp=torch.log_softmax(logits,dim=-1)
    details=logit_metrics(logits.numpy(),labels.numpy(),bins=15)
    result=classification_metrics(logp.exp(),labels)
    result.update(nll_stable_log_softmax=details["nll"],raw_logit_metrics=details,
        primary_nll_method="float64 log_softmax logits, no probability clipping",
        per_class_support=torch.bincount(labels.flatten(),minlength=logits.shape[-1]).tolist())
    return result


def evaluate(model,parents,device):
    model.eval()
    records,predictions=[],[]
    with torch.no_grad():
        for parent in parents:
            p,q=parent.labels.shape
            logits=torch.empty(p,q,model.num_classes,dtype=model.head.weight.dtype)
            assigned=torch.zeros(p,dtype=torch.bool)
            for tile in parent.tiles:
                value=model(tile.model_input.to(device)).detach().cpu()
                indices=tile.parent_position_indices
                if value.shape!=(len(indices),q,model.num_classes) or bool(assigned[indices].any()):
                    raise RuntimeError("Tile prediction/scatter correspondence failed")
                logits[indices]=value
                assigned[indices]=True
            if not bool(assigned.all()):
                raise RuntimeError("A parent scoring position was not predicted")
            records.append({"replicate":parent.parent_id,**logits_metrics(logits,parent.labels)})
            predictions.append(logits)
    return records,predictions


def train_configuration(configuration,settings,parents,device,seed,checkpoint,protocol_sha,manifest_sha,callback=None,integrity_check=None):
    """One frozen configuration/seed. Labels enter loss/evaluation only."""
    safe_component(configuration["id"],"configuration ID")
    if integrity_check: integrity_check()
    set_seed(seed)
    model=CanonicalTreeGNNGRU(settings["input_dim"],settings["num_classes"],configuration["hidden_dim"],
        settings["num_layers"],settings["temporal_layers"],configuration["dropout"]).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=configuration["learning_rate"],weight_decay=settings["weight_decay"])
    training=[(parent,tile) for parent in parents["train"] for tile in parent.tiles]
    best,best_epoch,stale=float("inf"),-1,0
    history=[]
    started=time.monotonic()
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats()
    for epoch in range(settings["epochs"]):
        if integrity_check: integrity_check()
        model.train()
        order=list(range(len(training)))
        random.shuffle(order)
        losses=[]
        for index in order:
            parent,tile=training[index]
            optimizer.zero_grad(set_to_none=True)
            logits=model(tile.model_input.to(device))
            labels=parent.labels[tile.parent_position_indices].to(device)
            loss=torch.nn.functional.cross_entropy(logits.reshape(-1,model.num_classes),labels.reshape(-1))
            if not bool(torch.isfinite(loss)): raise RuntimeError("Nonfinite canonical training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),settings["gradient_clip_norm"])
            optimizer.step()
            losses.append(float(loss.detach()))
        val,_=evaluate(model,parents["val"],device)
        nll=float(np.mean([r["nll_stable_log_softmax"] for r in val]))
        item={"epoch":epoch,"train_nll":float(np.mean(losses)),"val_nll":nll,
            "val_accuracy":float(np.mean([r["accuracy_uniform_grid"] for r in val])),"elapsed_seconds":time.monotonic()-started}
        history.append(item)
        if callback: callback(item)
        if integrity_check: integrity_check()
        if nll<best: # Strict improvement gives earliest epoch on an exact tie.
            best,best_epoch,stale=nll,epoch,0
            torch.save({"model":model.state_dict(),"epoch":epoch,"validation_nll":nll,
                "protocol_sha256":protocol_sha,"dataset_manifest_sha256":manifest_sha},checkpoint)
        else:
            stale+=1
            if stale>=settings["patience"]: break
    if integrity_check: integrity_check()
    chosen=torch.load(checkpoint,map_location=device,weights_only=True)
    model.load_state_dict(chosen["model"])
    val,predictions=evaluate(model,parents["val"],device)
    if integrity_check: integrity_check()
    result={"configuration_id":configuration["id"],"configuration":configuration,"training_seed":seed,
        "protocol_sha256":protocol_sha,"dataset_manifest_sha256":manifest_sha,
        "parameters":sum(p.numel() for p in model.parameters()),"model_manifest":model.configuration_manifest(),
        "best_epoch":best_epoch,"validation_nll":best,"history":history,"per_replicate":val,
        "training_seconds":time.monotonic()-started,"checkpoint":str(checkpoint),"checkpoint_sha256":sha(checkpoint),
        "peak_cuda_memory_bytes":torch.cuda.max_memory_allocated() if device.type=="cuda" else None,
        **replicate_summary(val,seed)}
    return result,predictions


def select_family_configuration(records,settings,expected_config_sha256,*,config_bytes):
    """Require exact original config bytes/spec/code/raw-source locks for all starts."""
    require_sha256(expected_config_sha256,"expected original config SHA256")
    if not isinstance(config_bytes,bytes) or hashlib.sha256(config_bytes).hexdigest()!=expected_config_sha256:
        raise ValueError("Family selection requires exact original config bytes SHA256")
    if json.loads(config_bytes)!=settings:
        raise ValueError("Family settings differ from captured original config bytes")
    configurations={safe_component(c["id"],"configuration ID"):c for c in settings["configurations"]}
    if not configurations or len(configurations)!=len(settings["configurations"]):
        raise ValueError("Distinct frozen family configurations required")
    expected={(name,seed) for name in configurations for seed in settings["training_seeds"]}
    observed=[(r["configuration_id"],r["training_seed"]) for r in records]
    if len(observed)!=len(set(observed)) or set(observed)!=expected:
        raise ValueError("Family selection requires every frozen configuration/seed exactly once")
    implementations=settings.get("implementation_sha256")
    sources=settings["dataset"].get("source_artifact_sha256")
    if not isinstance(implementations,dict) or not implementations or not isinstance(sources,dict) or not sources:
        raise ValueError("Reviewed implementation/dependency and raw source hashes required")
    for value in (*implementations.values(),*sources.values()):require_sha256(value,"frozen source SHA256")
    for path in sources:pilot_path(path)
    counts={name:sum(p.numel() for p in CanonicalTreeGNNGRU(settings["input_dim"],settings["num_classes"],
        c["hidden_dim"],settings["num_layers"],settings["temporal_layers"],c["dropout"]).parameters())
        for name,c in configurations.items()}
    for row in records:
        name=row["configuration_id"]
        if row.get("protocol_sha256")!=expected_config_sha256:
            raise ValueError("Family record differs from exact expected original config SHA256")
        if row.get("configuration")!=configurations[name] or row.get("parameters")!=counts[name]:
            raise ValueError("Family record configuration/capacity differs from frozen specification")
        if row.get("dataset_manifest_sha256")!=settings["dataset"]["manifest_sha256"]:
            raise ValueError("Family record differs from frozen dataset manifest")
        if row.get("implementation_sha256")!=implementations:
            raise ValueError("Family record differs from reviewed implementation/dependency hashes")
        if row.get("source_artifact_sha256")!=sources:
            raise ValueError("Family record differs from reviewed raw/prepared/metadata source hashes")
    summaries=[]
    for name in configurations:
        rows=[r for r in records if r["configuration_id"]==name]
        if not all(not isinstance(r["validation_nll"],bool) and np.isfinite(r["validation_nll"]) and r["validation_nll"]>=0 for r in rows):
            raise ValueError("Invalid family validation NLL")
        summaries.append({"configuration_id":name,"mean_validation_nll":float(np.mean([r["validation_nll"] for r in rows])),"parameters":counts[name]})
    best=min(r["mean_validation_nll"] for r in summaries)
    tied=[r for r in summaries if r["mean_validation_nll"]<=best+settings["family_tie_tolerance_nats"]]
    return {"selected":min(tied,key=lambda r:(r["parameters"],r["configuration_id"])),"all_configurations":summaries}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    parser.add_argument("--expected-config-sha256")
    parser.add_argument("--data",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--seed",type=int,default=101)
    parser.add_argument("--configurations",nargs="+")
    parser.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    parser.add_argument("--prepare-only",action="store_true")
    args=parser.parse_args()
    set_research_title()
    torch.set_num_threads(4)
    require_storage(headroom_bytes=500_000_000)
    config_path=pilot_path(args.config)
    manifest_path=pilot_path(args.data)/"manifest.json"
    snapshot=capture_run_snapshot(config_path,manifest_path,args.expected_config_sha256)
    settings=json.loads(snapshot.config_bytes)
    manifest=json.loads(snapshot.manifest_bytes)
    if args.seed not in settings["training_seeds"]: raise ValueError("Seed outside frozen family")
    selected=args.configurations or [c["id"] for c in settings["configurations"]]
    for name in selected: safe_component(name,"selected configuration ID")
    if len(selected)!=len(set(selected)) or not set(selected).issubset({c["id"] for c in settings["configurations"]}): raise ValueError("Unknown/duplicate family configuration")
    output=pilot_path(args.output)
    if output.exists(): raise RuntimeError("Choose a fresh versioned run directory")
    snapshot.check(full_sources=True)
    if args.device=="cuda":
        if os.environ.get("ARGGNN_PHYSICAL_GPU") not in {"2","3","4","5"}:
            raise RuntimeError("Use the project GPU launcher")
        lease=project_path(os.environ["ARGGNN_GPU_LEASE"])
        if os.fstat(int(os.environ["ARGGNN_GPU_LEASE_FD"])).st_ino!=lease.stat().st_ino or torch.cuda.device_count()!=1:
            raise RuntimeError("GPU lease/visible-device mismatch")
    elif torch.cuda.is_available(): raise RuntimeError("CPU runs require CUDA_VISIBLE_DEVICES empty")
    output.mkdir(parents=True)
    (output/"protocol.json").write_bytes(snapshot.config_bytes)
    (output/"dataset_manifest.json").write_bytes(snapshot.manifest_bytes)
    parents={}
    receipts=[]
    records={(r["split"],str(r["replicate"])):r for r in manifest["replicates"]}
    for split in ("train","val"):
        parents[split]=[]
        snapshot.check()
        for case in load_cases(manifest,split):
            snapshot.check()
            record=records[(split,str(case.metadata["replicate"]))]
            started=time.monotonic()
            safe_provenance={"parent_id":str(case.metadata["replicate"]),"split":split,
                "source_tensor_sha256":record["tensor_sha256"],"source_prepared_sha256":record["prepared_tensor_sha256"]}
            prepared=prepare_canonical_parent(case.sequence,settings["reference_adapter"],
                base_feature_count=settings["base_feature_count"],source_provenance=safe_provenance,
                original_feature_names=case.metadata["feature_names"],source_tensor_sha256=record["tensor_sha256"])
            if not torch.equal(prepared.sequence.positions,case.sequence.positions) or not torch.equal(prepared.sequence.query_node_ids,case.sequence.query_node_ids):
                raise RuntimeError("Canonical preprocessing changed external scoring/query correspondence")
            parent=bind_training_parent(prepared,case.labels,case.metadata,settings)
            saved=output/f"canonical_input_{split}_{parent.parent_id}.pt"
            snapshot.check()
            torch.save(parent.prepared.as_dict(),saved) # Observed input/correspondence only; no labels.
            receipts.append({"split":split,"parent_id":parent.parent_id,"input_cache":str(saved),"input_cache_sha256":sha(saved),
                "source_tensor_sha256":record["tensor_sha256"],"source_prepared_sha256":record["prepared_tensor_sha256"],
                "canonicalization":parent.prepared.correspondence.provenance,"adapter":parent.prepared.adapter_provenance,
                "preparation_seconds":time.monotonic()-started,"scoring_positions":parent.prepared.sequence.positions.numel(),
                "query_haplotypes":parent.prepared.sequence.num_queries})
            parents[split].append(parent)
    snapshot.check(full_sources=True)
    results={"stage":settings["stage"],"interpretation":settings["interpretation"],"evaluation_split":"validation_only",
        "test_accessed":False,"training_seed":args.seed,"protocol_sha256":snapshot.config_sha256,"dataset_manifest_sha256":snapshot.manifest_sha256,
        "dataset_source_protocol_sha256":manifest["protocol_sha256"],"implementation_sha256":snapshot.implementation_sha256,
        "source_artifact_sha256":snapshot.source_sha256,"integrity_checks":{"config_manifest_code":"preparation/configuration boundaries and each epoch",
            "source_artifacts":"stat identity each preparation boundary/epoch; full SHA256 before/after preparation/configurations"},
        "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "physical_gpu":os.environ.get("ARGGNN_PHYSICAL_GPU"),"environment":{"torch":str(torch.__version__),"cuda":torch.version.cuda},
        "preparation":receipts,"configurations":{},"family_selection_complete":False}
    dump_json(output/"preparation.json",results)
    if args.prepare_only:
        snapshot.check(full_sources=True)
        print(json.dumps({"event":"canonical_preparation_complete","output":str(output)}),flush=True)
        return
    for configuration in settings["configurations"]:
        name=configuration["id"]
        if name not in selected: continue
        snapshot.check(full_sources=True)
        require_storage(headroom_bytes=100_000_000)
        checkpoint=output/(name+".pt")
        def callback(item):
            print(json.dumps({"event":"epoch","configuration_id":name,"seed":args.seed,**item}),flush=True)
        result,predictions=train_configuration(configuration,settings,parents,torch.device(args.device),args.seed,
            checkpoint,results["protocol_sha256"],results["dataset_manifest_sha256"],callback,snapshot.check)
        snapshot.check(full_sources=True)
        saved=output/(name+"_validation_logits.pt")
        torch.save({"logits":predictions,"parent_ids":[p.parent_id for p in parents["val"]],
            "positions":[p.prepared.sequence.positions for p in parents["val"]],
            "query_ids":[p.prepared.sequence.query_node_ids for p in parents["val"]],
            "protocol_sha256":results["protocol_sha256"]},saved)
        result.update(validation_predictions=str(saved),validation_predictions_sha256=sha(saved),implementation_sha256=results["implementation_sha256"],source_artifact_sha256=snapshot.source_sha256)
        results["configurations"][name]=result
        dump_json(output/(name+".json"),result)
        dump_json(output/"progress.json",results)
    snapshot.check(full_sources=True)
    dump_json(output/"results.json",results)
    print(json.dumps({"event":"canonical_pilot_complete","output":str(output)}),flush=True)


if __name__=="__main__": main()
