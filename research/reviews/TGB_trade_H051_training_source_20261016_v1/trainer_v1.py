"""Fixed full-budget primary comparison; root supplies data, writer and guard."""
from __future__ import annotations
from dataclasses import dataclass
import copy,hashlib
import torch
from .preparation_v1 import prepare_sequence_v1,require
from .model_v1 import TradeOrderedModelV1,PRIMARY_ARMS,parameter_report_v1
from .targets_and_metrics_v1 import prepare_train_targets_v1,equal_year_soft_cross_entropy_v1

@dataclass(frozen=True)
class SealedPrimaryPredictionsV1:
    scores:torch.Tensor
    seeds:tuple
    arms:tuple
    years:tuple
    seal_ref:dict
    fit_receipt_ref:dict
    score_payload_sha256:str

def _cpu_state(model):
    return {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}

def run_primary_v1(*,dataset,load_train_target_year,writer,runtime_check,device,protocol):
    """No held-out target capability, resume, early stopping or model selection."""
    cfg=copy.deepcopy(protocol);campaign=cfg["primary_campaign"]
    require(campaign["arms"]==list(PRIMARY_ARMS) and campaign["seeds"]==[17]
        and campaign["epochs"]==100 and campaign["optimizer_updates_per_epoch"]==1,
        "Frozen three-arm seed17 final-epoch100 primary budget")
    require(cfg["architecture"]["width"]==64 and cfg["architecture"]["destinations"]==255
        and callable(runtime_check),"Frozen architecture and root runtime callback")
    runtime_check("before_scientific_allocation")
    executed_protocol_ref=writer.begin_run(cfg)
    # No AMP. Root qualifies the actual CUDA0 implementation; no device selection here.
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    runtime_check("before_causal_graph_preparation")
    sequence=prepare_sequence_v1(dataset,builder_sha256=cfg["source_refs"]["dataset_core"]["sha256"])
    sequence=sequence.to(device)
    train_steps=tuple(step for step in sequence.steps if step.year<=2009)
    require(tuple(step.year for step in train_steps)==tuple(range(1987,2010)),"All 23 TRAIN years")
    # All graph inputs/schedules/plans and TRAIN targets stay device-resident.
    targets=prepare_train_targets_v1(load_train_target_year,device)
    runtime_check("all_inputs_resident_before_epochs")
    years=tuple(range(2010,2017));all_scores=[];checkpoints=[];parameter_reports={}
    for seed in campaign["seeds"]:
        torch.manual_seed(seed)
        template=TradeOrderedModelV1("tree_only",width=64,destinations=255).float()
        initial=_cpu_state(template)
        initial_ref=writer.save_initial(seed,initial,dict(seed=seed,identical_parameter_names_shapes=True))
        del template
        seed_scores=[]
        for arm in campaign["arms"]:
            runtime_check("before_model_"+arm)
            torch.manual_seed(seed)
            model=TradeOrderedModelV1(arm,width=64,destinations=255).float()
            model.load_state_dict(initial,strict=True);model=model.to(device=device,dtype=torch.float32)
            report=parameter_report_v1(model)
            require(report["total_learned_parameters"]==116351
                and report["structurally_unused_carry_parameters"]==(12288 if arm=="tree_only" else 0),
                "Exact frozen parameter accounting")
            parameter_reports[arm]=report
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=.0001)
            history=[]
            for epoch in range(1,101):
                runtime_check("before_epoch_%s_%d"%(arm,epoch))
                model.train();optimizer.zero_grad(set_to_none=True);state=None;losses=[]
                # Full chronological BPTT through all 23 years, reset every epoch.
                for step in train_steps:
                    runtime_check("TRAIN_year_%d"%step.year)
                    logits,state=model.forward_step(step,state)
                    losses.append(equal_year_soft_cross_entropy_v1(logits,targets[step.year]))
                loss=torch.stack(losses).mean()
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                history.append(dict(epoch=epoch,optimizer_update=epoch,equal_year_soft_CE=float(loss.detach().cpu()),
                    gradient_norm_before_clip=float(norm.detach().cpu())))
            require(len(history)==100,"Entire frozen budget completed")
            runtime_check("before_final_checkpoint_"+arm)
            checkpoint=writer.save_final(seed,arm,_cpu_state(model),
                dict(seed=seed,arm=arm,final_epoch=100,optimizer_updates=100,training_years=list(range(1987,2010)),
                    common_initial_ref=initial_ref,parameter_report=report,no_VAL_TEST_selection=True),history)
            checkpoints.append(checkpoint)
            # Replay the complete past forcing with FINAL frozen weights. TRAIN
            # outputs are discarded; no teacher forcing or held-out fitting.
            del state,losses,loss,norm
            model.eval();state=None;scores=[]
            with torch.no_grad():
                for step in sequence.steps:
                    runtime_check("forecast_year_%d"%step.year)
                    logits,state=model.forward_step(step,state)
                    probability=torch.softmax(logits,dim=1)
                    if step.year>=2010:scores.append(probability)
            heldout=torch.stack(scores).detach().cpu()
            require(heldout.shape==(7,255,255) and bool(torch.isfinite(heldout).all()),"All held-out countries and periods")
            seed_scores.append(heldout)
            del model,optimizer,state,probability,heldout,logits
        all_scores.append(torch.stack(seed_scores))
    scores=torch.stack(all_scores).contiguous()
    runtime_check("before_sealing_all_heldout_predictions")
    payload_sha=hashlib.sha256(memoryview(scores.numpy().reshape(-1)).cast("B")).hexdigest()
    seal_ref=writer.seal_predictions(scores,dict(
        axes=["seed","arm","forecast_year","source_country","destination"],shape=list(scores.shape),
        dtype=str(scores.dtype),score_payload_sha256=payload_sha,score_semantics="FP32 softmax destination proportions",
        seeds=campaign["seeds"],arms=campaign["arms"],forecast_years=list(years),
        all_255_source_predictions=True,official_query_source_selection_performed=False,
        heldout_target_loader_invoked_by_trainer=False,fit_target_years=list(range(1987,2010)),
        chosen_final_epoch=100,checkpoint_refs=checkpoints,executed_protocol_ref=executed_protocol_ref,
        catalogue_sha256=sequence.catalogue_sha256,observation_digests=list(sequence.observation_digests)))
    receipt=writer.write_fit_receipt(dict(schema="TGB_trade_H051_primary_fit_completed_v1",
        primary_full_budget_complete=True,total_models=3,total_optimizer_updates=300,
        TRAIN_graph_forwards=6900,final_replay_graph_forwards=90,parameter_reports=parameter_reports,
        cut_change_years=[step.year for step in sequence.steps if step.cut_changed],
        TRAIN_cut_change_years=[step.year for step in train_steps if step.cut_changed],
        seed_count=1,optimizer_seed_uncertainty_estimable=False,
        retained_temporal_state_shape=[255,64],leaf_state_compression_claim=False,
        executed_protocol_ref=executed_protocol_ref,prediction_seal_ref=seal_ref,
        VAL_or_TEST_metric_computed=False,confirmation_started=False))
    return SealedPrimaryPredictionsV1(scores,(17,),tuple(campaign["arms"]),years,seal_ref,receipt,payload_sha)
