"""One actual FP32 CPU graph-model forward/backward qualification, no optimizer."""
from pathlib import Path
from datetime import datetime,timezone
import sys,json,hashlib,traceback,os
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES")==""
    sys.path[:0]=[str(ROOT),str(ROOT/"src")]
    import torch,numpy as np
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(17)
    from research.reviews.TGB_trade_PyG_dataset_source_20261016_v1.trade_tree_core_v1 import CountryCatalogueV1,observed_year_v1
    from research.reviews.TGB_trade_PyG_dataset_source_20261016_v1.trade_tree_pyg_v1 import _TradeTreeDatasetCoreV1
    from research.reviews.TGB_trade_PyG_dataset_source_20261016_v1.whole_country_tree_encoder_v1 import make_full_tree_schedule_v1
    from research.reviews.TGB_trade_H051_training_source_20261016_v1.model_v1 import TradeOrderedModelV1,parameter_report_v1
    from research.reviews.TGB_trade_H051_training_source_20261016_v1.preparation_v1 import PreparedYearV1,canonical_cut_v1
    from research.reviews.H051_conditional_innovation_design_20261016_v1.conditional_innovation_v1 import make_plan_v1
    catalogue=CountryCatalogueV1.synthetic_for_fixture(tuple("synthetic_country_%03d"%i for i in range(255)))
    years=(1986,1987,1988);observations={}
    for y in years:
        weights=np.zeros((255,255),dtype=np.float32)
        for i in range(255):weights[i,(i%2 if y!=1987 else (i//2)%2)]=1.
        observations[y]=observed_year_v1(y,weights,np.ones(255,dtype=bool))
    dataset=_TradeTreeDatasetCoreV1(catalogue,observations.__getitem__,available_edge_years=years,forecast_years=(1987,1988,1989))
    graphs=[dataset[i] for i in range(3)]
    cuts=[canonical_cut_v1(g.cut_node_index) for g in graphs]
    provenance=dict(schema="H051_observation_only_cut_source_v1",observed_trees_sha256="a"*64,builder_sha256="b"*64,targets_or_outcomes_used=False,identity_namespace_sha256=None)
    plans=[None,make_plan_v1(cuts[0],cuts[0],cuts[1],provenance=provenance),make_plan_v1(*cuts,provenance=provenance)]
    steps=[PreparedYearV1(1987+i,g,make_full_tree_schedule_v1(g),plans[i],i>0) for i,g in enumerate(graphs)]
    template=TradeOrderedModelV1("tree_only").float()
    original={k:v.detach().clone() for k,v in template.state_dict().items()}
    results={};outputs={}
    for arm in ("tree_only","first_order","conditional_triple"):
        model=TradeOrderedModelV1(arm).float();model.load_state_dict(original,strict=True)
        report=parameter_report_v1(model);assert report["total_learned_parameters"]==116351
        assert report["structurally_unused_carry_parameters"]==(12288 if arm=="tree_only" else 0)
        with torch.no_grad():
            zero_logits,_=model.forward_step(steps[0],None)
            expected=torch.log(steps[0].graph.leaf_profile+1e-8)
            assert torch.equal(zero_logits,expected)
            # Fixed nonzero head enables the entire encoder/GRU path to receive
            # a gradient in this engineering case. No fitting/update is performed.
            w=model.readout[-1].weight
            w.copy_(torch.linspace(-.03,.03,w.numel(),dtype=torch.float32).reshape_as(w))
        state=None
        for step in steps:logits,state=model.forward_step(step,state)
        outputs[arm]=logits.detach()
        loss=logits.square().mean();loss.backward()
        gradients={name:dict(present=p.grad is not None,finite=bool(torch.isfinite(p.grad).all()) if p.grad is not None else None,max_abs=float(p.grad.abs().max()) if p.grad is not None else None) for name,p in model.named_parameters()}
        assert all(p.dtype==torch.float32 for p in model.parameters())
        assert all(x["present"] and x["finite"] for x in gradients.values())
        assert any(x["max_abs"]>0 for n,x in gradients.items() if n.startswith("encoder."))
        assert any(x["max_abs"]>0 for n,x in gradients.items() if n.startswith("temporal_cell."))
        if arm=="tree_only":assert gradients["temporal_cell.weight_hh"]["max_abs"]==0
        results[arm]=dict(parameter_report=report,zero_residual_equals_fixed_prior=True,all_parameter_gradients_finite=True,encoder_and_GRU_nonzero_gradient=True,synthetic_logit_shape=list(logits.shape),gradient_parameter_count=len(gradients))
    delta=float((outputs["conditional_triple"]-outputs["first_order"]).abs().max())
    assert delta>1e-8 and not torch.cuda.is_initialized()
    result=dict(schema="TGB_H051_actual_model_FP32_CPU_qualification_v1",status="PASS",completed_utc=datetime.now(timezone.utc).isoformat(),synthetic_graphs="255 aligned synthetic leaves, changing A-B-A two-cell partitions, full hierarchy",actual_trainable_encoder_GRU_head_forward_and_backward=True,nondefault_head_used_only_to_test_gradient_reachability=True,frozen_zero_head_initialization_verified_before_debug_change=True,optimizer_steps=0,training_runs=0,real_observations_or_targets_read=False,GPU_context_created=False,scientific_metric_or_accuracy=False,arms=results,conditional_vs_first_order_logit_max_abs=delta)
    p=ROOT/"research/reviews/TGB_trade_H051_data_integration_root_20261016_v1/actual_model_FP32_CPU_qualification.root_v1.json"
    with p.open("x") as stream:json.dump(result,stream,indent=2);stream.write("\n")
    print(json.dumps(dict(status="PASS",model_parameters=116351,arms=3,conditional_vs_first_order_logit_max_abs=delta,optimizer_steps=0,GPU=False)),flush=True)
    return 0
if __name__=="__main__":
    try:raise SystemExit(main())
    except SystemExit:raise
    except BaseException:traceback.print_exc(file=sys.stdout);raise SystemExit(1)
