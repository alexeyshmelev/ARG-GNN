"""Future FP32 C3 step. Whole1600-leaf graphs, query-only independent GRUs.
Existing qualified two-phase shared-context/VJP algorithm, new query/device axes.
No numerical source imports, execution, launcher or admission at module import.
"""
from pathlib import Path
import hashlib,importlib.util,json,resource,time
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/C3_raw_cache_transfer_diagnosis_source_20261016_v1"
SOURCE=HERE+"/efficient_step_v2.staged.py"
BASE_REF=dict(path="research/reviews/C3_shared_context_training_source_20261016_v1/shared_context_step_v1.py",
 sha256="bfea21ea1f18d01aa0b01644295b539c25ac80e42928a471cf1a8e3d922ab786")
CACHE_REF={'path': 'research/reviews/C3_raw_cache_transfer_diagnosis_source_20261016_v1/single_parent_device_cache_v2.staged.py', 'sha256': '98c4dc26d1703ba2b871638b3a8cd6f66a0aa659a3dc9c216f464888f7328242'}
CAP_KEYS=("max_parent_trees","max_parent_blocks","max_active_device_feature_bytes","max_block_feature_bytes",
 "max_query_context_device_bytes","max_recurrent_metadata_device_bytes","max_progress_record_bytes",
 "max_successor_cuda_reserved_bytes")
_BASE=None

def require(ok,message):
    if not ok:raise ValueError(message)

def ref(path):
    p=ROOT/path
    require(p.resolve()==p and p.is_file() and not p.is_symlink() and p.stat().st_size<=1048576,
            "Canonical bounded source or metadata")
    return dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def _load(record,name):
    require(ref(record["path"])==record,"Exact pinned source before deferred runtime import")
    spec=importlib.util.spec_from_file_location(name,ROOT/record["path"])
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def _base():
    global _BASE
    require(ref(BASE_REF["path"])==BASE_REF,"Original qualified shared-context source unchanged")
    if _BASE is None:_BASE=_load(BASE_REF,"_C3_efficient_original_shared_context_v1")
    return _BASE

def forecast_v1(plan,*,queries=1000,hidden=32,focals=10000,haplotypes=1600):
    trees=plan["trees"];blocks=plan["blocks"];b=len(blocks)
    spans=[x["stop_tree"]-x["start_tree"] for x in blocks]
    require(type(trees)is int and trees>0 and blocks and sum(spans)==trees
            and all(type(t)is int and 0<t<=32 for t in spans),"Exact complete parent block coverage")
    v=trees*queries*hidden*4;m=max(spans)*queries*hidden*4
    hidden_bytes=queries*hidden*4
    metadata=24*focals+12*trees+4*(b+1)*hidden_bytes+16*focals*hidden*4
    nodes=max(x["nodes"] for x in blocks);sites=max(x["sites"] for x in blocks)
    # Explicit provisional tensor allowances, not an autograd/cuDNN workspace bound.
    activation_allowance=128*nodes*hidden*4+64*sites*haplotypes*hidden*4
    return dict(trees=trees,blocks=b,query_context_value_bytes=v,
        largest_block_nodes=nodes,largest_block_sites=sites,
        activation_tensor_planning_allowance_bytes=activation_allowance,
        activation_allowance_formula="128*Nmax*D*4+64*Smax*H*D*4",
        activation_allowance_is_provisional_not_proven_peak_bound=True,
        context_and_adjoint_and_transient_bound_bytes=4*v+2*m,
        recurrent_index_boundary_focal_planning_bytes=metadata,
        excludes_opaque_autograd_workspaces_and_existing_model_optimizer=True,measured_fit=False)

def validate_caps_v1(caps,forecast):
    require(isinstance(caps,dict) and set(caps)==set(CAP_KEYS)
            and all(type(v)is int and v>0 for v in caps.values()),"Exact prospective explicit caps")
    require(forecast["trees"]<=caps["max_parent_trees"] and forecast["blocks"]<=caps["max_parent_blocks"]
            and forecast["context_and_adjoint_and_transient_bound_bytes"]<=caps["max_query_context_device_bytes"]
            and forecast["recurrent_index_boundary_focal_planning_bytes"]<=caps["max_recurrent_metadata_device_bytes"],
            "Explicit query/context/boundary counts fit root bounds")
    require(caps["max_successor_cuda_reserved_bytes"]<=40*2**30,
            "Extra root-chosen abort cap never raises original40GiB maximum")
    planned=(caps["max_active_device_feature_bytes"]+forecast["context_and_adjoint_and_transient_bound_bytes"]
             +forecast["recurrent_index_boundary_focal_planning_bytes"]+forecast["activation_tensor_planning_allowance_bytes"])
    require(planned<caps["max_successor_cuda_reserved_bytes"],
            "Named tensor planning allowances below additional cap, opaque peak still unproven")
    return dict(caps)

def bounded_resource_check_v1(original,torch,actual,caps):
    require(callable(original),"Original resource guard remains mandatory")
    limit=caps["max_successor_cuda_reserved_bytes"]
    require(type(limit)is int and 0<limit<=40*2**30,"Root-chosen additional CUDA abort ceiling")
    if getattr(original,"_C3_additional_cuda_cap",None)==(str(actual),limit):return original
    def check(stage):
        original(stage)
        if actual.type=="cuda":
            require(torch.cuda.memory_reserved(actual)<=limit and torch.cuda.max_memory_reserved(actual)<=limit,
                    "Additional efficient-successor CUDA reserved/observed-peak abort cap")
    check._C3_additional_cuda_cap=(str(actual),limit)
    return check

def _backward_v1(model,provider,labels_cpu,*,device,resource_check,caps,focal_chunk_size,
                 expected_queries,capture_logits=False):
    """No optimizer action. Fixture may use small explicit H/Q/N, production may not."""
    base=_base();require(ref(CACHE_REF["path"])==CACHE_REF,"Exact owned devicecache source");torch,F,checkpoint=base._runtime()
    from arggnn.american_c3_gnn_dataset_isolated_v2 import BANK_FIELDS,BLOCK_TENSORS
    SingleParentDeviceFeatureCacheV1=_load(CACHE_REF,"_C3_efficient_single_parent_cache_private").SingleParentDeviceFeatureCacheV1
    actual=base.require_supported_model_v1(model,device)
    require(model.training and type(expected_queries)is int and expected_queries>0,"Training mode and explicit query count")
    h=model.expected_haplotypes;d=model.hidden_dim;n=labels_cpu.numel()
    require(type(h)is int and h>expected_queries and labels_cpu.device.type=="cpu"
            and labels_cpu.dtype==torch.long and labels_cpu.shape==(n,) and n>0
            and bool(((labels_cpu>=0)&(labels_cpu<3)).all()),"All original CPU targets")
    forecast=forecast_v1(provider.plan,queries=expected_queries,hidden=d,focals=n,haplotypes=h)
    validate_caps_v1(caps,forecast)
    resource_check=bounded_resource_check_v1(resource_check,torch,actual,caps)
    resource_check("before_efficient_step_allocations")
    cache=SingleParentDeviceFeatureCacheV1(provider,torch=torch,device=actual,
        fields=BANK_FIELDS+BLOCK_TENSORS,max_bytes=caps["max_active_device_feature_bytes"],
        max_block_bytes=caps["max_block_feature_bytes"],check=resource_check)
    versions=tuple(p._version for p in model.parameters());rng=base._rng_state(torch,actual)
    counts=dict(encoder_value_forwards=0,encoder_VJP_forwards=0,encoder_VJP_backwards=0,
                directional_GRU_forward_or_replay_entries=0,classifier_forward_or_replay_entries=0,
                whole_parent_loss_backwards=0)
    contexts=[];metadata=[];timings={}
    def unchanged():
        require(tuple(p._version for p in model.parameters())==versions,
                "No optimizer or weight mutation between context values and exact encoderVJPs")
    started=time.perf_counter()
    try:
        t0=time.perf_counter();cache.prepare();timings["raw_feature_cache_seconds"]=time.perf_counter()-t0
        # Export existing host-call timers through the unchanged progress hook.
        # provider.get includes hashing/context work, not only hash throughput.
        timings["raw_feature_provider_get_seconds"]=cache.hash_get_seconds
        timings["raw_feature_copy_call_seconds"]=cache.copy_seconds
        query_columns=None;query_device=None
        t0=time.perf_counter()
        for index,spec in enumerate(provider.plan["blocks"]):
            resource_check("efficient_context_values_"+str(index));unchanged()
            block=cache.get(index);meta=cache.metadata[index]
            t=spec["stop_tree"]-spec["start_tree"]
            require(tuple(block.sample_index.shape)==(t,h),"All original full-graph sample columns")
            q=meta["query_column"]
            require(q.device.type=="cpu" and q.dtype==torch.long and q.shape==(expected_queries,)
                    and torch.unique(q).numel()==expected_queries and bool(((q>=0)&(q<h)).all())
                    and bool((meta["reference_class"][q]==-1).all())
                    and int((meta["reference_class"]<0).sum())==expected_queries,
                    "Every and only original query column, no reference/query substitution")
            if query_columns is None:
                query_columns=q.clone();query_device=q.to(actual)
            else:require(torch.equal(q,query_columns),"Same persistent query columns across all trees")
            inverse=torch.full((h,),-1,dtype=torch.long)
            inverse[q]=torch.arange(expected_queries)
            rows,hold,columns=meta["focal_rows"],meta["focal_hold"],meta["focal_sample_column"]
            require(rows.dtype==hold.dtype==columns.dtype==torch.long
                    and rows.shape==hold.shape==columns.shape
                    and bool(((hold>=0)&(hold<t)).all())
                    and bool(((columns>=0)&(columns<h)).all())
                    and bool((inverse[columns]>=0).all()),"All focal rows remain original query/sample pairs")
            metadata.append(dict(rows=rows,hold=hold.to(actual),columns=inverse[columns].to(actual),
                                 geometry=block.genomic_geometry))
            with torch.no_grad():
                full=model.encode_bank(block.to_model_inputs())
                require(full.shape==(t,h,d) and full.dtype==torch.float32 and bool(torch.isfinite(full).all()),
                        "Original whole1600-haplotype encoder before readout selection")
                leaf=full.index_select(1,query_device).detach().contiguous().requires_grad_(True)
            contexts.append(leaf);counts["encoder_value_forwards"]+=1;cache.unchanged(index)
            del full,leaf,block
        timings["encoder_value_seconds"]=time.perf_counter()-t0
        base._same_rng(torch,actual,rng);unchanged()
        dummy=torch.zeros((),device=actual,dtype=torch.float32,requires_grad=True)
        def scan(direction):
            hidden=torch.zeros((1,expected_queries,d),device=actual,dtype=torch.float32)
            rows=[];histories=[];locals_=[]
            order=range(len(contexts)) if direction=="forward" else range(len(contexts)-1,-1,-1)
            for index in order:
                meta=metadata[index]
                def evaluate(_dummy,context,state,physical=index,scan_direction=direction):
                    resource_check("efficient_query_GRU_"+scan_direction+"_"+str(physical));unchanged()
                    counts["directional_GRU_forward_or_replay_entries"]+=1
                    m=metadata[physical]
                    history,next_state=model.scan_context(context,m["geometry"],direction=scan_direction,hidden=state)
                    return next_state,history[m["hold"],m["columns"]]
                hidden,history=checkpoint(evaluate,dummy,contexts[index],hidden,use_reentrant=False,preserve_rng_state=True)
                rows.append(meta["rows"]);histories.append(history)
                if direction=="forward":locals_.append(contexts[index][meta["hold"],meta["columns"]])
            joined=torch.cat(rows);permutation=torch.argsort(joined)
            require(torch.equal(joined[permutation],torch.arange(n)),"All10000original focal rows exactly once")
            order_device=permutation.to(actual)
            return (torch.cat(locals_).index_select(0,order_device) if locals_ else None,
                    torch.cat(histories).index_select(0,order_device))
        t0=time.perf_counter()
        local,forward=scan("forward");_,reverse=scan("reverse")
        losses=[];debug=[];labels=labels_cpu.to(actual)
        for start in range(0,n,focal_chunk_size):
            stop=min(n,start+focal_chunk_size)
            def classify(_dummy,x,f,r,y):
                resource_check("efficient_classifier_or_replay");unchanged()
                counts["classifier_forward_or_replay_entries"]+=1
                # Preserve original sum-perfocalchunk and CPU scalar mean reduction.
                return F.cross_entropy(model.classify_context(x,f,r),y,reduction="sum").to("cpu")
            x,f,r=local[start:stop],forward[start:stop],reverse[start:stop]
            losses.append(checkpoint(classify,dummy,x,f,r,labels[start:stop],
                                     use_reentrant=False,preserve_rng_state=True))
            if capture_logits:
                with torch.no_grad():debug.append(model.classify_context(x,f,r).cpu())
        loss=torch.stack(losses).sum()/n
        require(bool(torch.isfinite(loss)),"Finite unchanged all-focal meanCE")
        loss.backward();counts["whole_parent_loss_backwards"]+=1
        timings["full_bidirectional_query_GRU_head_backward_seconds"]=time.perf_counter()-t0
        base._same_rng(torch,actual,rng);unchanged()
        require(all(x.grad is not None and x.grad.shape==x.shape and x.grad.dtype==torch.float32
                    and bool(torch.isfinite(x.grad).all()) for x in contexts),
                "Both complete directional and directhead context adjoints")
        t0=time.perf_counter()
        for index,leaf in enumerate(contexts):
            resource_check("efficient_encoder_VJP_"+str(index));unchanged()
            block=cache.get(index)
            full=model.encode_bank(block.to_model_inputs());counts["encoder_VJP_forwards"]+=1
            value=full.index_select(1,query_device)
            require(value.shape==leaf.shape and bool(torch.isfinite(value).all()),"Exact original encoder queryVJP")
            torch.autograd.backward(value,grad_tensors=leaf.grad)
            counts["encoder_VJP_backwards"]+=1
            leaf.grad=None;contexts[index]=None;cache.unchanged(index)
            del block,full,value,leaf
        timings["encoder_VJP_seconds"]=time.perf_counter()-t0
        base._same_rng(torch,actual,rng);unchanged()
        B=len(provider.plan["blocks"])
        require(counts["encoder_value_forwards"]==counts["encoder_VJP_forwards"]==counts["encoder_VJP_backwards"]==B,
                "Exactly2B originalencoderentries plus B exactVJPs")
        counts.update(cache.diagnostics(),full_graph_haplotypes=h,recurrent_query_columns=expected_queries,
                      original_focal_labels=n)
        timings["gradient_engine_seconds"]=time.perf_counter()-started
        return dict(loss=loss.detach(),logits=torch.cat(debug) if capture_logits else None,
                    counts=counts,timings=timings,memory_forecast=forecast)
    finally:
        for leaf in contexts:
            if leaf is not None:leaf.grad=None
        contexts.clear();metadata.clear();cache.clear()

def complete_parent_step_v1(model,provider,targets,optimizer,*,device,resource_check,caps,
                            gradient_clip=1.,focal_chunk_size=1024,first_update=False):
    base=_base();torch,_,_=base._runtime()
    from arggnn.american_c3_gnn_stream_train_isolated_v2 import require_baseline_v1,SealedObservationProviderV1,_parameter_sha
    from arggnn.american_c3_gnn_targets_v1 import TrainFocalTargetVectorV1
    require_baseline_v1(model)
    require(isinstance(provider,SealedObservationProviderV1) and provider.seal_record is not None
            and isinstance(targets,TrainFocalTargetVectorV1)
            and targets.parent_id==provider.plan["parent_id"]
            and targets.focal_identity_sha256==provider.seal_record["focal_identity_sha256"]
            and targets.labels_cpu.shape==(10000,) and targets.labels_cpu.dtype==torch.long
            and provider.plan["haplotypes"]==model.expected_haplotypes==1600
            and provider.plan["focal_records"]==10000 and provider.plan["sequence_length_bp"]==2000000
            and len(provider)==len(provider.plan["blocks"]) and gradient_clip==1.,
            "Original complete2Mb training parent/all1600graphs/all10000labels")
    actual=base.require_supported_model_v1(model,device)
    resource_check("complete_parent_begin");model.train();optimizer.zero_grad(set_to_none=True)
    before=_parameter_sha(model) if first_update else None
    if actual.type=="cuda":torch.cuda.reset_peak_memory_stats(actual)
    resource_check=bounded_resource_check_v1(resource_check,torch,actual,caps)
    resource_check("additional_successor_cap_before_update")
    start=time.perf_counter()
    try:
        result=_backward_v1(model,provider,targets.labels_cpu,device=actual,resource_check=resource_check,
            caps=caps,focal_chunk_size=focal_chunk_size,expected_queries=1000)
        grads=[p.grad for p in model.parameters() if p.requires_grad]
        require(all(g is not None and bool(torch.isfinite(g).all()) for g in grads)
                and any(bool((g!=0).any()) for g in grads),"All parameter gradients finite and global gradient nonzero")
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),gradient_clip,error_if_nonfinite=True)
        require(bool(torch.isfinite(norm)),"Original finite global norm")
        resource_check("complete_parent_optimizer");t0=time.perf_counter();optimizer.step()
        optimizer_seconds=time.perf_counter()-t0
        require(all(bool(torch.isfinite(p).all()) for p in model.parameters()),"Finite updated FP32parameters")
        after=_parameter_sha(model) if first_update else None
        if first_update:require(before!=after,"Actual firstupdate changes initializedweights")
        resource_check("complete_parent_end")
        return dict(parent_id=targets.parent_id,records=10000,loss=float(result["loss"]),gradient_norm=float(norm),
            condition="fresh_C3_supervised_short_independent_complete_parents",training_length_bp=2000000,
            whole_parent_optimizer_updates=1,forward_blocks=len(provider),reverse_blocks=len(provider),
            block_reset_or_truncated_BPTT=False,detached_recurrent_history=False,
            all_trees_and_samples_retained=True,full_graph_haplotypes=1600,recurrent_query_columns=1000,
            shared_context_two_phase_encoder_VJP=True,bounded_single_parent_raw_device_cache=True,
            checkpointed_GRU=True,checkpointed_sites=True,bitwise_dense_gradient_equivalence_claimed=False,
            cost_counts=result["counts"],memory_forecast=result["memory_forecast"],
            timings=dict(result["timings"],optimizer_call_seconds=optimizer_seconds,
                         complete_parent_step_seconds=time.perf_counter()-start),
            first_real_update_checks=(dict(finite_loss=True,all_parameter_gradients_finite=True,
                global_gradient_nonzero=True,updated_parameters_finite=True,parameter_sha_before=before,
                parameter_sha_after=after,checked_inside_full_training=True) if first_update else None),
            memory=dict(process_peak_RSS_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,
             CUDA_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(actual)) if actual.type=="cuda" else None,
             CPU_cache_cap_bytes=provider.cache_bytes,
             additional_cuda_reserved_abort_cap_bytes=caps["max_successor_cuda_reserved_bytes"],
             original_cuda_reserved_ceiling_bytes=40*2**30,
             block_activation_or_workspace_fit_not_measured=True),
            paper_native_training_equivalent=False,novel_architecture_claim=False)
    finally:
        optimizer.zero_grad(set_to_none=True);provider.release()
