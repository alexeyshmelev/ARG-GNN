
"""Synthetic full512/max30-token native resource qualification; never corpus training."""
import math,time,pickle
def qualify(module,cfg,directory,torch,OmegaConf,ctx,runtime):
    # The fixed synthetic vocabulary exercises the exact observed model dimensions.
    # It is not a biological vocabulary or a transferable scientific checkpoint.
    synthetic=directory/"synthetic_only"/"val";synthetic.mkdir(parents=True)
    vocab={"special":{"[PAD]":0,"[MASK]":1,"[CLS]":2,"[SEP]":3},
        "node_id_tokens":{"node_"+str(n):n+4 for n in range(106769)}}
    with (synthetic/"raw_val_sequences_and_vocab.pkl").open("xb") as stream:
        pickle.dump(dict(raw_sequences=[[0]*29],vocab=vocab),stream,protocol=pickle.HIGHEST_PROTOCOL)
    cfg.data_local=str(synthetic.parent)
    from composer.utils import reproducibility
    from transformers import DataCollatorForLanguageModeling
    reproducibility.seed_all(17)
    model=module.build_model(cfg.model)
    parameters=sum(p.numel() for p in model.parameters())
    if parameters!=167655192 or model.model.config.vocab_size!=106776:
        raise ValueError("Exact native512 big-vocabulary parameter dimensions required")
    if not model.model.config.use_fa2 or not model.model.config.unpad_embeddings or model.model.config.masked_prediction:
        raise ValueError("Exact native FA2/unpadded/all-token decoder")
    model.to(torch.device("cuda",0));model.train()
    optimizer=module.build_optimizer(cfg.optimizer,model)
    collator=DataCollatorForLanguageModeling(tokenizer=model.tokenizer,mlm=True,mlm_probability=.3)
    # 512 paths, each exactly30 true tokens at source padded length128.
    examples=[dict(input_ids=[2]+[4+((j*28+k)%106769) for k in range(28)]+[3]+[0]*98,
        attention_mask=[1]*30+[0]*98,token_type_ids=[0]*128) for j in range(512)]
    elapsed=[];losses=[];torch.cuda.reset_peak_memory_stats()
    for step in range(3):
        ctx["fast_guard"]("before_synthetic_qualification_step")
        batch={k:v.to("cuda") for k,v in collator(examples).items()}
        optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.monotonic()
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
            outputs=model(batch);loss=outputs["loss"]
        if not bool(torch.isfinite(loss).item()):raise ValueError("Nonfinite qualification loss")
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
        optimizer.step();torch.cuda.synchronize()
        elapsed.append(time.monotonic()-start);losses.append(float(loss.detach().cpu()))
        ctx["fast_guard"]("after_synthetic_qualification_step")
        del batch,outputs,loss,norm
    from pathlib import Path
    import sys
    if any(n.startswith("src.") and getattr(v,"__file__",None) and not Path(v.__file__).resolve().is_relative_to(Path(module.__file__).parent)
        for n,v in list(sys.modules.items())):raise ValueError("Author module origin changed")
    identity=ctx["actual_device_receipt"]()
    result=dict(status="passed_synthetic_MLM_resource_qualification",batch=512,microbatch=512,padded_length=128,
        true_tokens_per_path=30,vocabulary=106776,parameters=parameters,optimizer_steps=3,precision="amp_bf16",seed=17,layers=12,heads=12,hidden_size=768,intermediate_size=3072,
        losses_finite=all(math.isfinite(v) for v in losses),losses=losses,step_seconds=elapsed,
        actual_context_uuid=identity["actual_device_identity"]["actual_context_uuid"],
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        full_budget_ETA_claimed=False,scientific_training_or_checkpoint_claimed=False,
        source="unchanged_author_build_model_and_optimizer_synthetic_dimension_qualification",
        optimizer_configuration_unchanged=True,dataloader_corpus_RAM_qualified=False)
    del optimizer,model
    return result
