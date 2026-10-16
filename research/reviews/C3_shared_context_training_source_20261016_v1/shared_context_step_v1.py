"""SOURCE ONLY: same-objective FP32 shared contexts and exact encoder VJPs.

No launcher, admission, CLI, numerical import at module import, or authority.
Root must separately admit the future complete training family and its bounds.
"""
from pathlib import Path
import hashlib,json,resource
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/C3_shared_context_training_source_20261016_v1"
SOURCE=HERE+"/shared_context_step_v1.py"
ORIGINAL_STEP_REF=dict(path="src/arggnn/american_c3_gnn_stream_train_isolated_v2.py",
    sha256="09394390199ceee020e4123d25a4cfcdcee38229450abc3ddd24dde55773b912")
MODEL_REF=dict(path="src/arggnn/american_c3_gnn_model_v1.py",
    sha256="f1a64bf56a6edc6e2b5ead5d0b06dda1d64034bdfaa5f8fc166f7b5ddffb994d")
CAP_KEYS=("max_context_cpu_bytes","max_metadata_and_boundary_cpu_bytes",
    "max_block_copy_bytes","max_context_device_copy_bytes")

def require(ok,message):
    if not ok:raise ValueError(message)

def source_ref(path):
    p=ROOT/path
    require(p.resolve()==p and p.is_file() and not p.is_symlink() and p.suffix in (".py",".json")
        and p.stat().st_size<=1048576,"Bounded canonical source/metadata only")
    return dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def memory_forecast_v1(plan,*,haplotypes=1600,hidden_dim=32,focals=10000):
    """Count explicitly owned tensor storage, not opaque autograd/workspace/RSS."""
    blocks=plan["blocks"];trees=plan["trees"]
    require(type(trees)is int and trees>0 and blocks and all(type(v)is int and v>0
        for v in (haplotypes,hidden_dim,focals)),"Positive explicit dimensions")
    spans=[b["stop_tree"]-b["start_tree"] for b in blocks]
    require(all(type(t)is int and t>0 for t in spans) and sum(spans)==trees,"Complete block coverage")
    values=trees*haplotypes*hidden_dim*4
    largest=max(spans)*haplotypes*hidden_dim*4
    hidden=haplotypes*hidden_dim*4
    feature=max(100*b["nodes"]+8*t*haplotypes+(haplotypes+130)*b["sites"]
        +24*t+20*haplotypes+24*focals for b,t in zip(blocks,spans))
    metadata=24*focals+12*trees
    boundaries=4*(len(blocks)+1)*hidden
    focal_buffers=16*focals*hidden_dim*4
    context_device=3*largest+max(spans)*haplotypes*(hidden_dim+3)*4+2*hidden+12*max(spans)
    return dict(schema="C3_shared_context_explicit_tensor_planning_v1",trees=trees,blocks=len(blocks),
        context_values_cpu_bytes=values,context_leaf_grad_cpu_bytes=values,
        context_cpu_transient_allowance_bytes=2*values+2*largest,
        max_context_cpu_bytes=4*values+2*largest,
        focal_index_and_geometry_cpu_bytes=metadata,
        directional_CPU_boundary_bytes=boundaries,focal_CPU_buffer_planning_bytes=focal_buffers,
        max_metadata_and_boundary_cpu_bytes=metadata+boundaries+focal_buffers,
        max_block_copy_bytes=feature,max_context_device_copy_bytes=context_device,
        largest_context_bytes=largest,
        excludes_existing_feature_cache_model_optimizer_autograd_bookkeeping_and_GPU_workspaces=True,
        whole_process_RSS_and_GPU_cap_still_required=True,measured_fit=False)

def validate_memory_caps_v1(caps,forecast):
    require(isinstance(caps,dict) and set(caps)==set(CAP_KEYS),"Exact explicit shared-context tensor caps")
    for key in CAP_KEYS:
        require(type(caps[key])is int and caps[key]>0 and forecast[key]<=caps[key],
            "Prospective tensor bound exceeds root cap: "+key)
    return dict(caps)

def _runtime():
    import torch
    from torch.nn import functional as F
    from torch.utils.checkpoint import checkpoint
    return torch,F,checkpoint

def require_supported_model_v1(model,device):
    """Fail closed on stochastic modules, buffers, hooks, mixed precision or aliases."""
    torch,_,_=_runtime()
    from torch import nn
    from torch_geometric.nn.aggr import SumAggregation
    from arggnn.american_c3_gnn_model_v1 import AmericanC3WholeTreeGNNV1
    from arggnn.mutation_routed_pyg_gnn_v1 import _GatedUpwardMessagePassingV1,_ExcludedCavityMessagePassingV1
    require(source_ref(MODEL_REF["path"])==MODEL_REF and source_ref(ORIGINAL_STEP_REF["path"])==ORIGINAL_STEP_REF,
        "Exact original model and reference-step source")
    require(type(model)is AmericanC3WholeTreeGNNV1 and model.class_order==("EUR","EAS","AFR")
        and model.hidden_dim==32 and model.donor_cohorts==2 and model.num_classes==3
        and model.site_chunk_size==256 and model.checkpoint_sites is True,
        "Only original fresh C3 width32/site256 checkpointed baseline")
    actual=torch.device(device)
    require(torch.is_grad_enabled() and torch.get_default_dtype()==torch.float32
        and not torch.is_autocast_enabled() and not torch.is_autocast_enabled("cpu")
        and torch.get_float32_matmul_precision()=="highest","FP32 only; no AMP or reduced matmul precision")
    if actual.type=="cuda":
        require(actual.index is not None and not torch.backends.cuda.matmul.allow_tf32
            and not torch.backends.cudnn.allow_tf32,"One explicit CUDA device with TF32 disabled")
    else:require(actual.type=="cpu","Only CPU or one explicit CUDA device")
    allowed={AmericanC3WholeTreeGNNV1,nn.Sequential,nn.Linear,nn.GELU,nn.Tanh,nn.LayerNorm,
        nn.GRU,_GatedUpwardMessagePassingV1,_ExcludedCavityMessagePassingV1,SumAggregation}
    require(not list(model.named_buffers()),"No trainable or running-state buffers supported")
    for module in model.modules():
        require(type(module)in allowed and "forward" not in vars(module),"Unsupported/overridden module")
        require(all(not value for key,value in vars(module).items()
            if key.endswith("_hooks") and isinstance(value,dict)),"No side-effecting module hooks")
        if type(module)is nn.GRU:
            require(module.num_layers==1 and not module.bidirectional and module.dropout==0,
                "Original deterministic one-layer directional GRU only")
        if type(module)is nn.GELU:require(module.approximate=="none","Original GELU equation")
    params=list(model.parameters())
    require(params and all(p.dtype==torch.float32 and p.device==actual and p.requires_grad
        and not getattr(p,"_backward_hooks",None) for p in params),"One-device FP32 trainable parameters, no hooks")
    named=list(model.named_parameters(remove_duplicate=False))
    require(len({id(p) for _,p in named})==len(named),"No parameter aliases across encoder/GRU/head")
    return actual

def _rng_state(torch,device):
    return (torch.random.get_rng_state().clone(),
        torch.cuda.get_rng_state(device).cpu().clone() if device.type=="cuda" else None)

def _same_rng(torch,device,before):
    after=_rng_state(torch,device)
    require(torch.equal(before[0],after[0]) and
        (before[1] is None if after[1] is None else torch.equal(before[1],after[1])),
        "Checkpoint body consumed RNG; unsupported stochastic behavior")

def _two_phase_backward_v1(model,get_block,plan,labels_cpu,*,device,resource_check,
        focal_chunk_size,memory_caps,capture_logits=False):
    """Internal gradient engine. No optimizer step; fixture may use tiny H/N.

    Deliberate context leaf cut is completed by J_encoder^T * accumulated_context_grad.
    Recurrent states are never detached, truncated or reset at block boundaries.
    """
    torch,F,checkpoint=_runtime()
    from arggnn.american_c3_gnn_dataset_isolated_v2 import BANK_FIELDS,BLOCK_TENSORS
    from arggnn.american_c3_gnn_stream_train_isolated_v2 import _tensors
    actual=require_supported_model_v1(model,device)
    require(model.training,"Original training-mode checkpoint behavior required")
    n=labels_cpu.numel();h=model.expected_haplotypes;d=model.hidden_dim
    require(type(h)is int and h>0 and labels_cpu.device.type=="cpu" and labels_cpu.dtype==torch.long
        and labels_cpu.shape==(n,) and n>0 and bool(((labels_cpu>=0)&(labels_cpu<3)).all())
        and type(focal_chunk_size)is int and focal_chunk_size>0,"Exact CPU class targets/chunk")
    forecast=memory_forecast_v1(plan,haplotypes=h,hidden_dim=d,focals=n)
    validate_memory_caps_v1(memory_caps,forecast)
    versions=tuple(p._version for p in model.parameters());rng=_rng_state(torch,actual)
    contexts=[];metadata=[];dummy=torch.zeros((),device=actual,dtype=torch.float32,requires_grad=True)
    counts=dict(encoder_value_forwards=0,encoder_VJP_forwards=0,encoder_VJP_backwards=0,
        directional_GRU_forward_or_replay_entries=0,classifier_forward_or_replay_entries=0,
        observation_get_calls=0,whole_parent_loss_backwards=0)
    def unchanged():
        require(tuple(p._version for p in model.parameters())==versions,
            "No parameter update/mutation allowed between context and encoder-VJP phases")
    def get(index):
        counts["observation_get_calls"]+=1
        cpu=get_block(index)
        require(all(t.device.type=="cpu" and (not t.is_floating_point() or t.dtype==torch.float32)
            for key in BANK_FIELDS+BLOCK_TENSORS for t in _tensors(cpu[key])),
            "Authenticated CPU observations retain original FP32/integer types")
        return cpu
    try:
        # Phase1: only context values survive; no encoder graph is retained.
        for index,spec in enumerate(plan["blocks"]):
            resource_check("shared_context_values_block_"+str(index));unchanged()
            cpu=get(index);t=spec["stop_tree"]-spec["start_tree"]
            require(tuple(cpu.sample_index.shape)==(t,h) and cpu.genomic_geometry.shape==(t,3),
                "Complete block/sample cardinality")
            meta=dict(rows=cpu.focal_rows.clone(),hold=cpu.focal_hold.clone(),
                columns=cpu.focal_sample_column.clone(),geometry=cpu.genomic_geometry.clone())
            require(meta["rows"].dtype==meta["hold"].dtype==meta["columns"].dtype==torch.long
                and meta["rows"].shape==meta["hold"].shape==meta["columns"].shape
                and bool(((meta["hold"]>=0)&(meta["hold"]<t)).all())
                and bool(((meta["columns"]>=0)&(meta["columns"]<h)).all()),"Exact focal readout metadata")
            block=cpu.clone().to(actual)
            with torch.no_grad():
                value=model.encode_bank(block.to_model_inputs())
                require(value.dtype==torch.float32 and value.shape==(t,h,d)
                    and bool(torch.isfinite(value).all()),"Finite complete FP32 block context")
                leaf=value.detach().to("cpu").contiguous().requires_grad_(True)
            contexts.append(leaf);metadata.append(meta);counts["encoder_value_forwards"]+=1
            del cpu,block,value,leaf
        _same_rng(torch,actual,rng);unchanged()
        def scan(direction):
            hidden=torch.zeros((1,h,d),dtype=torch.float32,device="cpu")
            rows_parts=[];history_parts=[];local_parts=[]
            order=range(len(contexts)) if direction=="forward" else range(len(contexts)-1,-1,-1)
            for index in order:
                meta=metadata[index]
                def evaluate(_dummy,context_cpu,state,physical=index,scan_direction=direction):
                    resource_check("shared_context_GRU_"+scan_direction+"_block_"+str(physical))
                    counts["directional_GRU_forward_or_replay_entries"]+=1
                    m=metadata[physical]
                    history,next_state=model.scan_context(context_cpu.to(actual),m["geometry"].to(actual),
                        direction=scan_direction,hidden=state.to(actual))
                    return next_state.to("cpu"),history[m["hold"].to(actual),m["columns"].to(actual)].to("cpu")
                hidden,history=checkpoint(evaluate,dummy,contexts[index],hidden,
                    use_reentrant=False,preserve_rng_state=True)
                rows_parts.append(meta["rows"]);history_parts.append(history)
                if direction=="forward":
                    local_parts.append(contexts[index][meta["hold"],meta["columns"]])
            joined=torch.cat(rows_parts);permutation=torch.argsort(joined)
            require(torch.equal(joined[permutation],torch.arange(n)),"One-to-one complete focal row join")
            histories=torch.cat(history_parts).index_select(0,permutation)
            local=torch.cat(local_parts).index_select(0,permutation) if local_parts else None
            return local,histories
        # Phase2: full directional BPTT plus direct head path accumulate onto shared leaves.
        local,forward=scan("forward");_,reverse=scan("reverse")
        losses=[];debug=[]
        for start in range(0,n,focal_chunk_size):
            stop=min(start+focal_chunk_size,n)
            def classify(_dummy,x,f,r,y):
                resource_check("shared_context_classifier_or_replay")
                counts["classifier_forward_or_replay_entries"]+=1
                logits=model.classify_context(x.to(actual),f.to(actual),r.to(actual))
                return F.cross_entropy(logits,y.to(actual),reduction="sum").to("cpu")
            x,f,r=local[start:stop],forward[start:stop],reverse[start:stop]
            losses.append(checkpoint(classify,dummy,x,f,r,labels_cpu[start:stop],
                use_reentrant=False,preserve_rng_state=True))
            if capture_logits:
                # Fixture-only diagnostic; excluded from production cost proof.
                with torch.no_grad():debug.append(model.classify_context(x.to(actual),f.to(actual),r.to(actual)).cpu())
        loss=torch.stack(losses).sum()/n
        require(bool(torch.isfinite(loss)),"Finite original all-focal mean CE")
        resource_check("shared_context_full_parent_loss_backward");loss.backward()
        counts["whole_parent_loss_backwards"]+=1
        _same_rng(torch,actual,rng);unchanged()
        require(all(leaf.grad is not None and leaf.grad.shape==leaf.shape
            and leaf.grad.dtype==torch.float32 and bool(torch.isfinite(leaf.grad).all())
            for leaf in contexts),"Both directional and direct-head context adjoints accumulated")
        # Phase3: J_encoder^T*g per block, at unchanged weights, accumulates encoder parameters.
        for index in range(len(contexts)):
            resource_check("shared_context_encoder_VJP_block_"+str(index));unchanged()
            cpu=get(index);block=cpu.clone().to(actual)
            value=model.encode_bank(block.to_model_inputs());counts["encoder_VJP_forwards"]+=1
            leaf=contexts[index]
            require(value.shape==leaf.shape and value.dtype==torch.float32
                and bool(torch.isfinite(value).all()),"Same finite original encoder shape")
            torch.autograd.backward(value,grad_tensors=leaf.grad.to(actual))
            counts["encoder_VJP_backwards"]+=1
            leaf.grad=None;contexts[index]=None
            del cpu,block,value,leaf
        _same_rng(torch,actual,rng);unchanged()
        require(counts["encoder_value_forwards"]==counts["encoder_VJP_forwards"]==len(plan["blocks"])
            and counts["encoder_VJP_backwards"]==len(plan["blocks"]),"Exactly2B encoder calls andB encoder VJPs")
        return dict(loss=loss.detach(),logits=torch.cat(debug) if capture_logits else None,
            counts=counts,memory_forecast=forecast)
    finally:
        for leaf in contexts:
            if leaf is not None:leaf.grad=None
        contexts.clear();metadata.clear()

def shared_context_complete_parent_step_v1(model,provider,targets,optimizer,*,device,
        resource_check,memory_caps,gradient_clip=1.,focal_chunk_size=1024,first_update=False):
    """Drop-in scientific step API; root supplies separately admitted explicit caps."""
    torch,_,_=_runtime()
    from arggnn.american_c3_gnn_stream_train_isolated_v2 import (
        require_baseline_v1,SealedObservationProviderV1,_parameter_sha)
    from arggnn.american_c3_gnn_targets_v1 import TrainFocalTargetVectorV1
    require_baseline_v1(model)
    require(isinstance(provider,SealedObservationProviderV1) and provider.seal_record is not None
        and isinstance(targets,TrainFocalTargetVectorV1) and targets.parent_id==provider.plan["parent_id"]
        and targets.focal_identity_sha256==provider.seal_record["focal_identity_sha256"]
        and targets.labels_cpu.dtype==torch.long and targets.labels_cpu.shape==(10000,)
        and provider.plan["haplotypes"]==model.expected_haplotypes==1600
        and provider.plan["focal_records"]==10000 and provider.plan["sequence_length_bp"]==2000000
        and len(provider)==len(provider.plan["blocks"]) and gradient_clip==1. and callable(resource_check),
        "Original sealed full TRAIN parent and externally joined all10000 targets")
    actual=require_supported_model_v1(model,device)
    resource_check("complete_parent_begin");model.train();optimizer.zero_grad(set_to_none=True)
    before=_parameter_sha(model) if first_update else None
    if actual.type=="cuda":torch.cuda.reset_peak_memory_stats(actual)
    try:
        result=_two_phase_backward_v1(model,provider.get,provider.plan,targets.labels_cpu,device=actual,
            resource_check=resource_check,focal_chunk_size=focal_chunk_size,memory_caps=memory_caps)
        grads=[p.grad for p in model.parameters() if p.requires_grad]
        require(all(g is not None and bool(torch.isfinite(g).all()) for g in grads)
            and any(bool((g!=0).any()) for g in grads),"Finite complete shared-context gradient for every parameter")
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),gradient_clip,error_if_nonfinite=True)
        require(bool(torch.isfinite(norm)),"Finite original global gradient norm")
        resource_check("complete_parent_optimizer");optimizer.step()
        require(all(bool(torch.isfinite(p).all()) for p in model.parameters()),"Finite updated FP32 parameters")
        after=_parameter_sha(model) if first_update else None
        if first_update:require(before!=after,"First real complete-parent update changes fresh parameters")
        resource_check("complete_parent_end")
        return dict(parent_id=targets.parent_id,records=10000,loss=float(result["loss"]),gradient_norm=float(norm),
            condition="fresh_C3_supervised_short_independent_complete_parents",training_length_bp=2000000,
            whole_parent_optimizer_updates=1,forward_blocks=len(provider),reverse_blocks=len(provider),
            shared_context_two_phase_encoder_VJP=True,checkpointed_encoder_and_GRU=False,
            checkpointed_GRU=True,checkpointed_sites=True,block_reset_or_truncated_BPTT=False,
            detached_recurrent_history=False,all_trees_and_samples_retained=True,
            bitwise_dense_gradient_equivalence_claimed=False,cost_counts=result["counts"],
            shared_context_memory_forecast=result["memory_forecast"],
            first_real_update_checks=(dict(finite_loss=True,all_parameter_gradients_finite=True,
                global_gradient_nonzero=True,updated_parameters_finite=True,parameter_sha_before=before,
                parameter_sha_after=after,checked_inside_full_training=True) if first_update else None),
            memory=dict(process_peak_RSS_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,
                CUDA_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(actual)) if actual.type=="cuda" else None,
                CUDA_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(actual)) if actual.type=="cuda" else None,
                CPU_cache_cap_bytes=provider.cache_bytes,CPU_boundary_state_graph_scales_with_blocks=True),
            paper_native_training_equivalent=False,novel_architecture_claim=False)
    finally:
        optimizer.zero_grad(set_to_none=True);provider.release()
