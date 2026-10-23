"""Memory-mapped per-subgraph training and truth-free prediction."""
from __future__ import annotations

import contextlib
import json
import math
import time
import numpy as np
import torch
from . import paper_argmix_generator as g
from .paper_argmix_gpu import validate_gpu_binding


def build_model(config, condition):
    spec = dict(config["model"])
    if condition == "author_key_i":
        from .paper_argmix_author_model import TmrcaAttn
    else:
        g.need(condition in ("equation_corrected", "hidden128"), "Unregistered model condition")
        from .paper_argmix_model import TmrcaAttn
    if condition == "hidden128":
        spec["d_model"] = 128
    return TmrcaAttn(**spec), spec


class CompactCorpus:
    """Keep compact maps on CPU and convert only selected batch rows to tensors."""
    def __init__(self, feature_manifests, label_manifests=None, *, config_ref=None, allowed_splits=None):
        self.parts, self.counts, self.refs = [], [], []
        self.labels = label_manifests is not None
        g.need(not self.labels or len(label_manifests) == len(feature_manifests), "Feature/label parent count differs")
        for i, path in enumerate(feature_manifests):
            feature = g.metadata(path)
            g.need(feature["schema"] == "paper_argmix_compact_features_v1" and feature["truth_read"] is False,
                   "Compact observed-only features required")
            if config_ref is not None:
                g.need(feature["config_ref"] == config_ref, "Feature recipe differs")
            if allowed_splits is not None:
                g.need(feature["split"] in allowed_splits, "Forbidden split in this stage")
            arrays = {name:np.load(g.verify_ref(feature["arrays"][name]),mmap_mode="r",allow_pickle=False)
                      for name in ("tokens","distances")}
            n, width = arrays["tokens"].shape
            g.need(n == feature["examples"] and arrays["distances"].shape == (n,width,width) and
                   arrays["tokens"].dtype == np.uint8 and arrays["distances"].dtype == np.uint16,
                   "Compact array shape or dtype differs")
            if self.labels:
                labels = g.metadata(label_manifests[i])
                g.need(labels["parent_id"] == feature["parent_id"] and labels["split"] == feature["split"] and
                       labels["config_ref"] == feature["config_ref"] and labels["class_order"] == list(g.CLASSES) and
                       labels["support_manifest_ref"] == feature["support_ref"] and labels["examples"] == n,
                       "Label/support identity differs")
                arrays["labels"] = np.memmap(g.verify_ref(labels["labels_ref"]),mode="r",dtype=np.uint8,shape=(n,))
                g.need(np.all(arrays["labels"] < 4), "Invalid founder class")
            self.parts.append(arrays)
            self.counts.append(n)
            self.refs.append(dict(features=g.file_ref(path), labels=g.file_ref(label_manifests[i]) if self.labels else None,
                                  parent_id=feature["parent_id"], split=feature["split"]))
        self.offsets = np.cumsum([0]+self.counts)
        self.examples = int(self.offsets[-1])

    def batch(self, indices, device):
        indices = np.asarray(indices,dtype=np.int64)
        width = self.parts[0]["tokens"].shape[1]
        x = np.empty((len(indices),width),dtype=np.int64)
        d = np.empty((len(indices),width,width),dtype=np.int64)
        y = np.empty(len(indices),dtype=np.int64) if self.labels else None
        for i, part in enumerate(self.parts):
            slots = np.flatnonzero((indices >= self.offsets[i]) & (indices < self.offsets[i+1]))
            local = indices[slots]-self.offsets[i]
            x[slots], d[slots] = part["tokens"][local],part["distances"][local]
            if y is not None:
                y[slots] = part["labels"][local]
        tokens = torch.from_numpy(x).to(device)
        distances = torch.from_numpy(d).to(device)
        labels = torch.from_numpy(y).to(device) if y is not None else None
        return tokens, distances, tokens != 2, labels


def autocast(device):
    return torch.autocast("cuda",dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()


def epoch(model, corpus, batch_size, device, *, order, optimizer=None, clip_norm=1.0, guard):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0,0,0
    for start in range(0,len(order),batch_size):
        x,d,mask,y = corpus.batch(order[start:start+batch_size],device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training),autocast(device):
            logits = model(x,d,mask)
            loss = torch.nn.functional.cross_entropy(logits.float(),y)
        g.need(torch.isfinite(loss).item(), "Non-finite loss; stop without replacement data")
        if training:
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),clip_norm,error_if_nonfinite=True)
            optimizer.step()
        total_loss += loss.item()*len(y)
        correct += int((logits.argmax(1) == y).sum().item())
        count += len(y)
        guard(0)
    return dict(loss=total_loss/count,accuracy=correct/count,examples=count)


def train(config_path, fit_manifest_path, output, *, condition, device_name, cpu_threads, guard):
    config = g.metadata(config_path)
    fit = g.metadata(fit_manifest_path)
    reference = g.file_ref(config_path)
    g.need(fit["schema"] == "paper_argmix_fit_inputs_v1" and fit["config_ref"] == reference, "Wrong fit manifest")
    train_corpus = CompactCorpus([g.verify_ref(r["features"]) for r in fit["train"]],
                                 [g.verify_ref(r["labels"]) for r in fit["train"]],config_ref=reference,allowed_splits={"train"})
    val = fit["validation"]
    val_corpus = CompactCorpus([g.verify_ref(val["features"])],[g.verify_ref(val["labels"])],
                              config_ref=reference,allowed_splits={"val"})
    expected = {p["parent_id"] for p in config["parents"] if p["split"] == "train"}
    g.need({p["parent_id"] for p in train_corpus.refs} == expected and len(train_corpus.refs)==5 and
           all(n==100000 for n in train_corpus.counts) and train_corpus.examples==500000 and val_corpus.examples==100000,
           "Full five-parent 500000 TRAIN / 100000 VAL contract required")
    g.need(cpu_threads > 0, "Positive CPU thread allocation required")
    torch.set_num_threads(cpu_threads)
    torch.manual_seed(config["training"]["seed"])
    np_rng = np.random.Generator(np.random.PCG64(config["training"]["seed"]))
    device = torch.device(device_name)
    g.need(device.type in ("cpu","cuda"), "Only explicit CPU/CUDA devices supported")
    if device.type == "cuda":
        validate_gpu_binding()
        g.need(torch.cuda.device_count() == 1, "Root must expose exactly the admitted campaign GPU UUID")
        torch.cuda.reset_peak_memory_stats(device)
    model, model_spec = build_model(config,condition)
    model.to(device)
    options = config["training"]
    optimizer = torch.optim.AdamW(model.parameters(),lr=options["lr"],betas=tuple(options["betas"]),
                                 eps=options["eps"],weight_decay=options["weight_decay"])
    output = g._fresh_directory(output,guard)
    budget = g.Writes(config["bounds"]["training_output_bytes"],guard)
    best_accuracy, best_epoch, best_weights, stale = -1.0,-1,None,0
    started = time.monotonic()
    records=[]
    with (output/"progress.jsonl").open("x") as progress:
        for index in range(options["epochs"]):
            train_result=epoch(model,train_corpus,options["batch_size"],device,order=np_rng.permutation(train_corpus.examples),
                               optimizer=optimizer,clip_norm=options["clip_grad_norm"],guard=guard)
            with torch.inference_mode():
                val_result=epoch(model,val_corpus,options["batch_size"],device,order=np.arange(val_corpus.examples),guard=guard)
            if val_result["accuracy"] > best_accuracy:
                best_accuracy,best_epoch,stale=val_result["accuracy"],index,0
                best_weights={name:weight.detach().cpu().clone() for name,weight in model.state_dict().items()}
            else:
                stale+=1
            record=dict(epoch=index,train=train_result,validation=val_result,best_epoch=best_epoch,
                        elapsed_seconds=time.monotonic()-started)
            text=json.dumps(record,sort_keys=True)+"\n"
            budget.before(len(text.encode())); progress.write(text); progress.flush(); budget.after()
            print(text,end="",flush=True)
            records.append(record)
            if stale >= options["patience"]:
                break
    g.need(best_weights is not None, "No validation checkpoint selected")
    checkpoint=output/"selected.pt"
    guard(config["bounds"]["checkpoint_bytes"])
    torch.save(dict(schema="paper_argmix_checkpoint_v1",condition=condition,model_config=model_spec,
                    model=best_weights,config_ref=reference,fit_manifest_ref=g.file_ref(fit_manifest_path),
                    selected_epoch=best_epoch,validation_accuracy=best_accuracy,
                    model_source_ref=g.file_ref(__file__.replace("paper_argmix_train.py",
                        "paper_argmix_author_model.py" if condition=="author_key_i" else "paper_argmix_model.py"))),checkpoint)
    g.need(checkpoint.stat().st_size <= config["bounds"]["checkpoint_bytes"], "Checkpoint cap exceeded")
    guard(0)
    receipt=dict(schema="paper_argmix_selected_model_v1",condition=condition,config_ref=reference,
                 source_ref=g.file_ref(__file__),fit_manifest_ref=g.file_ref(fit_manifest_path),
                 checkpoint_ref=g.file_ref(checkpoint),selection="earliest highest validation example accuracy",
                 selected_epoch=best_epoch,validation_accuracy=best_accuracy,epochs_completed=len(records),
                 training_examples=500000,validation_examples=100000,model_config=model_spec,
                 parameter_count=sum(p.numel() for p in model.parameters()),seconds=time.monotonic()-started,
                 device=str(device),torch_version=torch.__version__,cpu_threads=cpu_threads,
                 cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else 0,
                 test_or_ood_read=False,resume_supported=False)
    budget.json(output/"completed.json",receipt)
    return receipt


def predict(config_path, selected_path, feature_path, output, *, device_name, cpu_threads, guard):
    config = g.metadata(config_path)
    selected,feature = g.metadata(selected_path),g.metadata(feature_path)
    g.need(selected["config_ref"]==feature["config_ref"]==g.file_ref(config_path) and feature["split"] in ("iid","misspec"),
           "Frozen selected model and evaluation features required")
    torch.set_num_threads(cpu_threads)
    device=torch.device(device_name)
    if device.type=="cuda":
        validate_gpu_binding()
        g.need(torch.cuda.device_count()==1,"Expose only admitted campaign GPU UUID")
    checkpoint=torch.load(g.verify_ref(selected["checkpoint_ref"]),map_location="cpu",weights_only=True)
    model,spec=build_model(config,selected["condition"])
    g.need(checkpoint["model_config"]==spec and checkpoint["config_ref"]==g.file_ref(config_path),"Checkpoint config differs")
    model.load_state_dict(checkpoint["model"],strict=True)
    model.to(device).eval()
    corpus=CompactCorpus([feature_path],config_ref=g.file_ref(config_path),allowed_splits={"iid","misspec"})
    output=g._fresh_directory(output,guard)
    guard(4*4*corpus.examples+1024)
    probabilities=np.lib.format.open_memmap(output/"probabilities.npy",mode="w+",dtype=np.float32,shape=(corpus.examples,4))
    started=time.monotonic()
    with torch.inference_mode():
        for start in range(0,corpus.examples,config["training"]["batch_size"]):
            end=min(corpus.examples,start+config["training"]["batch_size"])
            x,d,mask,_=corpus.batch(np.arange(start,end),device)
            with autocast(device):
                logits=model(x,d,mask)
            result=logits.float().softmax(-1).cpu().numpy()
            g.need(np.all(np.isfinite(result)),"Non-finite probabilities")
            probabilities[start:end]=result
            guard(0)
    probabilities.flush()
    receipt=dict(schema="paper_argmix_predictions_v1",parent_id=feature["parent_id"],split=feature["split"],
                 config_ref=g.file_ref(config_path),selected_model_ref=g.file_ref(selected_path),
                 feature_ref=g.file_ref(feature_path),support_ref=feature["support_ref"],
                 probabilities_ref=g.file_ref(output/"probabilities.npy"),examples=corpus.examples,
                 class_order=list(g.CLASSES),truth_read=False,seconds=time.monotonic()-started)
    g.Writes(1024*1024,guard).json(output/"completed.json",receipt)
    return receipt
