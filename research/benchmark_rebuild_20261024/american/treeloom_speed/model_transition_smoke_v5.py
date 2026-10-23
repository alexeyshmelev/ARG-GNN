import traceback
try:

    """Small FP32 production-model optimizer continuation and history transition smoke."""
    from pathlib import Path
    from types import MethodType
    import copy,json,time
    import numpy as np
    import torch,tskit
    from arggnn.benchmarking import contracts as c
    from arggnn.american_c3_gnn_train_v1 import _cpu_snapshot
    from research.benchmark_rebuild_20261024.american.treeloom_speed import runtime_v2 as fresh
    from research.benchmark_rebuild_20261024.american.treeloom_speed import training_v2 as training
    from research.benchmark_rebuild_20261024.american.treeloom_speed.stable_leaf_history_v2 import stable_leaf_history_v2
    base=c.ROOT/"research/benchmark_rebuild_20261024/american/treeloom_speed/model_transition_smoke_v5"
    base.mkdir()
    torch.set_num_threads(1);torch.manual_seed(71)
    started=time.monotonic()
    job=c.read_json(".runtime/benchmarking/workflow_20261019_v8/jobs/AM_treeloom_fit.json")
    original=job["call"]["spec"]["training_spec"]
    registry=fresh.FreshParentRegistry.from_ref(original["parent_registry_ref"])
    runtime=fresh.FreshTreeLoom(registry);parent=registry.ids("train")[0]
    ages=np.asarray(([0.]*100+[1000.]*100)*3+[0.]*1000)
    tables=tskit.TableCollection(sequence_length=30000000)
    current=[tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE,time=age) for age in ages]
    clock=5000.
    while len(current)>1:
        parents=[]
        for first in range(0,len(current),2):
            parent_node=tables.nodes.add_row(time=clock)
            for child in current[first:first+2]:
                tables.edges.add_row(0,30000000,parent_node,child)
            parents.append(parent_node)
        current=parents;clock*=2
    edge_parents=tuple(tables.edges.parent);edge_children=tuple(tables.edges.child)
    tables.edges.clear()
    for left,right in ((0,500),(500,30000000)):
        for up,down in zip(edge_parents,edge_children):
            tables.edges.add_row(left,right,int(up),int(down))
    positions=np.asarray([10.,30.,15000060.,15000090.])
    for position in positions:tables.sites.add_row(position,"A")
    tables.sort();ts=tables.tree_sequence()
    nodes=np.asarray(ts.samples(),dtype=np.int64)
    focal_indices=600+np.arange(10000,dtype=np.int64)%1000
    focal_positions=np.concatenate((np.linspace(.05,499.95,5000),np.linspace(1000.,29999900.,5000)))
    focals=dict(record_ids=np.arange(10000,dtype=np.int64),sample_nodes=nodes[focal_indices],
        sample_indices=focal_indices,positions_bp0=focal_positions,parent_id=np.asarray(parent),
        class_order=np.asarray(fresh.CLASSES))
    np.savez(base/"focals.npz",**focals)
    manifest=dict(schema="american_c3_observed_v1",genealogy_kind=runtime.frozen.GENEALOGY,
        query_ancestry_truth_in_inputs=False,class_order=list(fresh.CLASSES),parent_id=parent,
        sequence_length_bp=30000000,synthetic_CPU_smoke=True)
    (base/"observation.json").write_text(json.dumps(manifest)+"\n")
    (base/"preparation.json").write_text(json.dumps(dict(synthetic_CPU_smoke=True))+"\n")
    refs=dict(observation_ref=c.file_ref(base/"observation.json"),
        focal_registry_ref=c.file_ref(base/"focals.npz"),native_preparation_ref=c.file_ref(base/"preparation.json"))
    state=dict(manifest=manifest,ts=ts,nodes=nodes,labels=np.asarray([2]*200+[0]*200+[1]*200+[-1]*1000),
        query_columns=np.arange(600,1600),focals=focals,genotype=np.random.default_rng(71).binomial(1,.2,size=(4,1600)).astype(np.int8),
        positions=positions,allele_count=np.full(4,2,dtype=np.int8),ages=ages,
        spans=np.asarray([500.,29999500.]),focal_tree=(focal_positions>=500).astype(np.int64))
    bundle,metadata=runtime.prepare_observation(state,split="train",resource_check=lambda stage:None,
        **refs,max_nodes=100000,max_holds=4,max_episode_payload_bytes=16*2**20)
    serial=runtime.build_model()
    parallel=runtime.build_model()
    live_job=c.read_json(".runtime/benchmarking/workflow_20261022_gpu345_v2/jobs/AM_treeloom_fit.json")
    live_output=c.contained_path(live_job["call"]["spec"]["output_directory"])
    initial_ref=c.read_json(live_output/"latest_checkpoint.json")["checkpoint_ref"]
    assert training._hash(c.contained_path(initial_ref["path"]))==initial_ref
    initial_saved=torch.load(c.contained_path(initial_ref["path"]),map_location="cpu",weights_only=True)
    initial_steps=training.validate_saved_state(initial_saved,serial,registry)
    serial.load_state_dict(initial_saved["model_state"],strict=True)
    parallel.load_state_dict(initial_saved["model_state"],strict=True)
    assert training._hash(c.contained_path(initial_ref["path"]))==initial_ref
    def fast_history(self,forcing,spans,plan):
        return stable_leaf_history_v2(self,forcing,self.ell_theta,spans,plan,device="cpu",
                                      holds_per_window=1024,resource_check=lambda stage:None)
    parallel.memory._history=MethodType(fast_history,parallel.memory)
    parallel.memory.checkpoint_memory=False
    serial.memory.checkpoint_memory=False
    labels=torch.arange(10000)%3
    def update(model,optimizer):
        model.train();optimizer.zero_grad(set_to_none=True)
        if model is serial:
            logits=model(bundle)
            loss=torch.nn.functional.cross_entropy(logits,labels)
            loss.backward()
        else:
            from research.benchmark_rebuild_20261024.american.treeloom_speed.decoder_v2 import decode_parent_backward, geometry_from_spans
            with torch.no_grad():
                logits=model(bundle)
            context=model.encode_full_parent_v1(bundle)
            loss,context_gradient,receipt=decode_parent_backward(model,context,geometry_from_spans(bundle.bank.spans),
                bundle.focal_hold,bundle.focal_sample_column,labels,tile_haplotypes=64)
            assert receipt["native_focals"]==10000
            context.backward(context_gradient)
        gradients={name:param.grad.detach().clone() for name,param in model.named_parameters()}
        assert len(gradients)==49 and all(torch.isfinite(value).all() for value in gradients.values())
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        return logits.detach(),loss.detach(),gradients
    def optimizer(model):
        return torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001,betas=(.9,.999),
            eps=1e-8,maximize=False,amsgrad=False,foreach=False,fused=False,capturable=False,differentiable=False)
    a,b=optimizer(serial),optimizer(parallel)
    a.load_state_dict(copy.deepcopy(initial_saved["optimizer_state"]))
    b.load_state_dict(copy.deepcopy(initial_saved["optimizer_state"]))
    ref_logits,ref_loss,ref_gradients=update(serial,a)
    new_logits,new_loss,new_gradients=update(parallel,b)
    max_output=float((ref_logits-new_logits).abs().max())
    max_gradient=max(float((ref_gradients[key]-new_gradients[key]).abs().max()) for key in ref_gradients)
    max_update=max(float((left-right).abs().max()) for left,right in zip(serial.parameters(),parallel.parameters()))
    assert torch.allclose(ref_logits,new_logits,rtol=1e-4,atol=1e-5)
    assert max_gradient<1e-4 and max_update<1e-5
    assert abs(float(ref_gradients["memory.ell_theta"]))>0
    # Save/reload the complete actual49-state Adam boundary, then compare the next
    # update with an uninterrupted copy using exactly the same candidate history.
    payload=dict(schema="treeloom_fresh_full_checkpoint_v1",epoch=(initial_steps+20)//20,complete_parent_updates=initial_steps+1,
        model_seed=70001,parent_registry_ref=registry.reference,settings=training.SETTINGS,
        contract=parallel.checkpoint_contract(),checkpoint_selection="fixed_final_epoch30",
        VAL_truth_read=False,TEST_truth_read=False,model_state=_cpu_snapshot(parallel.state_dict()),
        optimizer_state=_cpu_snapshot(b.state_dict()))
    with (base/"step1.pt").open("xb") as stream:torch.save(payload,stream)
    saved=torch.load(base/"step1.pt",map_location="cpu",weights_only=True)
    resumed=runtime.build_model();resumed.memory._history=MethodType(fast_history,resumed.memory)
    resumed.memory.checkpoint_memory=False
    training.validate_saved_state(saved,resumed,registry,expected_steps=initial_steps+1)
    resumed.load_state_dict(saved["model_state"],strict=True)
    resumed_optimizer=optimizer(resumed);resumed_optimizer.load_state_dict(saved["optimizer_state"])
    update(parallel,b);update(resumed,resumed_optimizer)
    continuation_error=max(float((left-right).abs().max()) for left,right in zip(parallel.parameters(),resumed.parameters()))
    assert continuation_error==0
    assert set(training._optimizer_counts(b))==set(training._optimizer_counts(resumed_optimizer))=={initial_steps+2}
    result=dict(schema="TreeLoom_production_FP32_optimizer_transition_CPU_smoke_v2",passed=True,CPU_only=True,
        samples=1600,queries=1000,native_holds=bundle.native_holds,focal_targets=10000,observed_SNPs=4,
        trainable_tensors=49,parameters=32132,serial_vs_parallel_logit_max_error=max_output,
        serial_vs_parallel_gradient_max_error=max_gradient,serial_vs_parallel_update_max_error=max_update,
        actual_saved_adam_states=49,actual_saved_step=initial_steps+1,resumed_and_uninterrupted_step=initial_steps+2,
        original_live_checkpoint_ref=initial_ref,original_completed_updates=initial_steps,
        resumed_vs_uninterrupted_parameter_max_error=continuation_error,all49_finite_gradients=True,learned_history_theta_gradient=float(ref_gradients["memory.ell_theta"]),
        production_streamed_decoder_64haps_exercised=True,
        source_ref=c.file_ref(__file__),seconds=time.monotonic()-started,
        limitations="Synthetic two native holds and FP32 CPU only; no production GPU speed or full-parent fit measured")
    with (base/"completed.json").open("x") as stream:json.dump(result,stream,indent=2);stream.write("\n")
    print(json.dumps(result,indent=2))

except BaseException:
    with open("research/benchmark_rebuild_20261024/american/treeloom_speed/model_transition_v5_failure.txt","x") as stream:
        stream.write(traceback.format_exc())
    raise
