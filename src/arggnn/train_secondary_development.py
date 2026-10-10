"""Fresh matched attribution-control fitting after strict main-study admission."""
from __future__ import annotations
from .resources import ROOT,set_research_title,require_storage
set_research_title()
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time
import torch
from .anchor_kernel import AnchorKernelModel,prepare_anchor_kernel_bank
from .anchor_controls import StaticLocalAnchorModel,NoncontextualAnchorKernelModel
from .train_anchor_development import optimizer_groups,evaluate,require_device
from .train import set_seed,dump_json
from .development_execution import configure_deterministic_development,deterministic_development_manifest
from . import secondary_control_bindings as bindings
from . import development_selection as selection
from .development_selection_bindings import guard_development_path

_IMPORTED_PINS=bindings.implementation_hashes()


def make_control(name,settings):
    spec=dict(settings["model_specs"][name]);control=spec.pop("control")
    cls=StaticLocalAnchorModel if control=="static_local" else NoncontextualAnchorKernelModel if control=="noncontextual" else AnchorKernelModel
    return cls(**{key:settings[key] for key in ("input_dim","num_classes","hidden_dim","num_layers","latent_dim","num_scales","num_anchors","initial_coupling","initial_damping")},**spec)


def validate(model,cases,prepared):
    rows,logits=evaluate(model,cases,prepared)
    records=[{"parent_id":case.parent_id,"nll":row["nll"],"nll_policy":selection.PARENT_NLL,
        "weighting":selection.PARENT_WEIGHT} for case,row in zip(cases,rows)]
    return math.fsum(row["nll"] for row in records)/len(records),records,logits


def fit_configuration(model,cases,prepared,*,name,seed,settings,config_sha,checkpoint,prediction,live_check):
    """Pure fitting utility; production budget/settings are admitted by the caller."""
    spec=settings["model_specs"][name];binding={key:settings[key] for key in selection.BINDINGS}
    main_sha=settings["main_selection_receipt"]["sha256"]
    optimizer=torch.optim.AdamW(optimizer_groups(model,settings["weight_decay"]),lr=settings["model_learning_rates"][name])
    tiles=[tile for case in cases["train"] for tile in case.tiles]
    best=float("inf");best_epoch=None;stale=0;history=[];started=time.monotonic()
    for epoch in range(settings["epochs"]):
        live_check(False);deterministic_development_manifest()
        model.train();order=list(range(len(tiles)));random.shuffle(order);losses=[]
        for index in order:
            tile=tiles[index];logits=model(prepared[id(tile)])
            loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),tile.labels.to(logits.device).flatten())
            if not bool(torch.isfinite(loss)):raise RuntimeError("Nonfinite secondary training loss")
            optimizer.zero_grad(set_to_none=True);loss.backward()
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise RuntimeError("Missing or nonfinite secondary gradient")
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
        nll,rows,logits=validate(model,cases["val"],prepared)
        live_check(False);deterministic_development_manifest()
        history.append({"epoch":epoch,"val_nll":nll,"train_nll":math.fsum(losses)/len(losses)})
        if nll<best:
            best,best_epoch,stale=nll,epoch,0
            torch.save({"model":model.state_dict(),"epoch":epoch,"seed":seed,"validation_nll":nll,
                "protocol_sha256":config_sha,"completion_receipt_sha256":settings["completion_receipt_sha256"],
                "boundary_manifest_sha256":settings["boundary_dataset_manifest_sha256"],
                "implementation_sha256":settings["bound_implementation_sha256"],"source_binding":binding,
                "main_selection_receipt_sha256":main_sha,"control_spec":spec,
                "deterministic_execution_policy":settings["deterministic_execution_policy"],
                "original_selected_logit_sha256":[selection.tensor_fingerprint(z) for z in logits]},checkpoint)
        else:
            stale+=1
            if stale>=settings["patience"]:break
    device=next(model.parameters()).device
    chosen=torch.load(checkpoint,map_location=device,weights_only=True)
    if chosen["protocol_sha256"]!=config_sha or chosen["control_spec"]!=spec or chosen["implementation_sha256"]!=settings["bound_implementation_sha256"]:
        raise RuntimeError("Secondary checkpoint binding changed")
    model.load_state_dict(chosen["model"],strict=True)
    nll,rows,logits=validate(model,cases["val"],prepared)
    live_check(True);deterministic_development_manifest()
    if nll!=chosen["validation_nll"] or [selection.tensor_fingerprint(z) for z in logits]!=chosen["original_selected_logit_sha256"]:
        raise RuntimeError("Deterministic secondary checkpoint replay differs")
    torch.save({"logits":logits,"positions":[case.positions for case in cases["val"]],
        "query_node_ids":[case.query_node_ids for case in cases["val"]],"parent_ids":[case.parent_id for case in cases["val"]],
        "protocol_sha256":config_sha,"seed":seed,"nll_policy":selection.RAW_NLL,
        "completion_receipt_sha256":settings["completion_receipt_sha256"],"source_binding":binding,
        "main_selection_receipt_sha256":main_sha,"control_spec":spec,
        "implementation_sha256":settings["bound_implementation_sha256"],
        "deterministic_execution_policy":settings["deterministic_execution_policy"]},prediction)
    ending=live_check(True);deterministic_development_manifest()
    return {"parameters":sum(p.numel() for p in model.parameters()),"best_epoch":best_epoch,
        "learning_rate":settings["model_learning_rates"][name],"control_spec":spec,"history":history,
        "validation_nll":nll,"per_parent":rows,"model_configuration":model.configuration_manifest(),
        "original_selected_validation_nll":chosen["validation_nll"],"reloaded_validation_nll":nll,
        "checkpoint_replay_nll_delta":0.,"checkpoint_replay_max_abs_logit_delta":0.,
        "checkpoint":str(checkpoint),"checkpoint_sha256":bindings.common.sha(checkpoint),
        "validation_predictions":str(prediction),"validation_predictions_sha256":bindings.common.sha(prediction),
        "ending_config_sha256":config_sha,"ending_implementation_sha256":ending,"ending_source_binding":binding,
        "training_seconds":time.monotonic()-started}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True);parser.add_argument("--expected-config-sha256",required=True)
    parser.add_argument("--output",required=True);parser.add_argument("--seed",type=int,default=101)
    parser.add_argument("--models",nargs="+");parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    args=parser.parse_args(argv)
    output=guard_development_path(args.output)
    if output.exists():raise ValueError("Fresh secondary run directory required")
    admission=bindings.admit_secondary(args.config,args.expected_config_sha256)
    settings=admission.settings
    if settings["bound_implementation_sha256"]!=_IMPORTED_PINS:raise ValueError("Secondary code changed after imports")
    if args.seed not in settings["training_seeds"]:raise ValueError("Prescribed secondary seed required")
    names=args.models or settings["models"]
    if not names or len(set(names))!=len(names) or any(name not in settings["models"] for name in names):raise ValueError("Prescribed distinct secondary models required")
    cases={split:admission.dataset.cases(split) for split in ("train","val")}
    for split,parents in cases.items():
        if len(parents)!=settings["splits"][split] or any(tile.bank.leaf_x.shape[-1]!=47 for case in parents for tile in case.tiles):raise ValueError("Exact observed secondary parent/source schema required")
    cases["val"].sort(key=lambda case:int(case.parent_id.rsplit(":",1)[1]))
    execution=configure_deterministic_development();require_device(args.device)
    torch.set_num_threads(4);device=torch.device(args.device)
    identity=hashlib.sha256(str(output.relative_to(ROOT)).encode()).hexdigest()[:16]
    checkpoints=ROOT/"checkpoints/secondary_development"/(identity+"_"+admission.sha256[:12]+f"_s{args.seed}")
    if checkpoints.exists():raise ValueError("Fresh secondary checkpoint namespace required")
    require_storage(headroom_bytes=200_000_000)
    output.mkdir(parents=True);checkpoints.mkdir(parents=True)
    (output/"protocol.json").write_bytes(admission.content)
    dump_json(output/"source_binding.json",{"source_snapshot":admission.source_lock.manifest(),"main_selection_receipt":settings["main_selection_receipt"]})
    binding={key:settings[key] for key in selection.BINDINGS}
    result={"kind":"secondary_development_run","family":"secondary_controls","development_namespace":"development-v1",
        "training_seed":args.seed,"training_seeds":settings["training_seeds"],"protocol_sha256":admission.sha256,
        "main_selection_receipt_sha256":settings["main_selection_receipt"]["sha256"],"source_binding":binding,
        "initial_implementation_sha256":_IMPORTED_PINS,"deterministic_execution_policy":execution,"models":{}}
    active=None;phase="bank_transfer"
    try:
        prepared={id(tile):prepare_anchor_kernel_bank(tile.bank,
            None if settings["selected_candidate_spec"]["geometry"]=="plain" else admission.ages[tile.provenance["tile_id"]],
            num_anchors=32,num_scales=3).to(device,torch.float32)
            for parents in cases.values() for case in parents for tile in case.tiles}
        phase="fitting"
        def live(full):
            deterministic_development_manifest();require_storage(headroom_bytes=100_000_000)
            return admission.check(full)
        for name in names:
            active=name;live(True);set_seed(args.seed)
            body=fit_configuration(make_control(name,settings).to(device),cases,prepared,name=name,seed=args.seed,
                settings=settings,config_sha=admission.sha256,checkpoint=checkpoints/(name+".pt"),
                prediction=checkpoints/(name+"_validation_predictions.pt"),live_check=live)
            body["ending_input_artifact_sha256"]=dict(admission.source_lock.hashes)
            result["models"][name]=body;dump_json(output/(name+".json"),body);dump_json(output/"progress.json",result)
        live(True);dump_json(output/"results.json",result)
    except BaseException as error:
        dump_json(output/"failure.json",{"kind":"secondary_development_failure_record","configuration_id":active,
            "seed":args.seed,"protocol_sha256":admission.sha256,"main_selection_receipt_sha256":settings["main_selection_receipt"]["sha256"],
            "attempt_id":f"run_{identity}_s{args.seed}"+("_"+active if active else ""),
            "phase":phase,"error_type":type(error).__name__,"reason":str(error),"terminal":True,"test_access":False})
        raise


if __name__=="__main__":main()
