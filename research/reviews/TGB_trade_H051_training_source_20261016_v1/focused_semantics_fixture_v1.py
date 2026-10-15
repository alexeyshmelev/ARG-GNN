"""ROOT-only model-free integration semantics fixture; default is inert."""
from pathlib import Path
import argparse,ast,hashlib,json,math,os,sys,types

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=Path(__file__).resolve().parent

def execute_fixture(output):
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    import torch
    sys.path.insert(0,str(ROOT))
    from research.reviews.H051_conditional_innovation_design_20261016_v1.conditional_innovation_v1 import (
        make_plan_v1,ConditionalInnovationTransportV1)
    from research.reviews.TGB_trade_H051_training_source_20261016_v1.targets_and_metrics_v1 import (
        TargetYearV1,prepare_train_targets_v1,equal_year_soft_cross_entropy_v1)
    checks=[]
    def ck(name,value):
        if not value:raise AssertionError(name)
        checks.append(name)
    def require(value,message):
        if not value:raise ValueError(message)
    ns=dict(torch=torch,hashlib=hashlib,json=json,require=require,make_plan_v1=make_plan_v1,
        make_full_tree_schedule_v1=lambda graph:None,
        PreparedYearV1=lambda year,graph,schedule,plan,changed:types.SimpleNamespace(
            year=year,graph=graph,schedule=schedule,event_plan=plan,cut_changed=changed),
        PreparedSequenceV1=lambda steps,catalogue,digests,builder:types.SimpleNamespace(steps=steps))
    src=(HERE/"preparation_v1.py").read_text();tree=ast.parse(src)
    for name in ("canonical_cut_v1","prepare_sequence_v1"):
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        exec(compile(ast.Module(body=[node],type_ignores=[]),"exact_preparation_source","exec"),ns)
    canonical=ns["canonical_cut_v1"]
    ck("anonymous_cut_renaming_does_not_create_an_event",torch.equal(
        canonical(torch.tensor([2,2,5,1])),canonical(torch.tensor([100,100,-1,50]))))
    graphs=[]
    for year in range(1987,2017):
        cut=(torch.arange(255)//128 if year<=1988 else torch.arange(255)%2 if year<=1990 else torch.arange(255)%3)
        if year in (1988,1990):cut=cut+1000
        graphs.append(types.SimpleNamespace(leaf_profile=torch.zeros(255,255),
            country_witness_index=torch.arange(255),observation_year=torch.tensor([year-1]),
            catalogue_sha256="a"*64,observation_sha256=hashlib.sha256(str(year).encode()).hexdigest(),
            cut_node_index=cut))
    class FakeDataset:
        forecast_years=tuple(range(1987,2017))
        def __getitem__(self,index):return graphs[index]
    prepared=ns["prepare_sequence_v1"](FakeDataset(),builder_sha256="b"*64)
    ck("only_genuine_cut_changes_create_transport_events",
        [s.year for s in prepared.steps if s.cut_changed]==[1989,1991])
    ck("unchanged_annual_holds_keep_their_separate_forcing_steps",len(prepared.steps)==30 and
        all(s.event_plan is None for s in prepared.steps if s.year not in (1989,1991)))
    first_plan=prepared.steps[2].event_plan
    state255=torch.arange(255,dtype=torch.float32)[:,None]
    zero=ConditionalInnovationTransportV1(gate="fixed",alpha=0.)
    triple=ConditionalInnovationTransportV1(gate="fixed",alpha=1.)
    ck("first_distinct_cut_bootstrap_has_identical_memory_transport",
        torch.allclose(zero(state255,first_plan,basis="leaf"),triple(state255,first_plan,basis="leaf")))
    ck("later_event_retains_two_last_DISTINCT_cut_episodes",prepared.steps[4].event_plan.old_cells==4
        and prepared.steps[4].event_plan.new_cells==6)
    # Execute exact forward_step body with no learned encoder/GRU/head instance.
    text=(HERE/"model_v1.py").read_text();klass=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=="TradeOrderedModelV1")
    method=next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=="forward_step")
    scope=dict(torch=torch)
    exec(compile(ast.Module(body=[method],type_ignores=[]),"exact_model_forward_step_with_stubs","exec"),scope)
    forward=scope["forward_step"]
    forcing=torch.tensor([[.1],[.2],[.3],[.4]])
    state=torch.tensor([[1.],[3.],[5.],[7.]])
    provenance=dict(schema="H051_observation_only_cut_source_v1",observed_trees_sha256="c"*64,
        builder_sha256="d"*64,targets_or_outcomes_used=False,identity_namespace_sha256=None)
    plan=make_plan_v1(torch.tensor([0,0,1,1]),torch.tensor([0,1,0,1]),torch.tensor([0,0,0,1]),provenance=provenance)
    prior_profile=torch.tensor([[.8,.2],[0.,1.],[0.,0.],[.5,.5]])
    fake_graph=types.SimpleNamespace(leaf_profile=prior_profile)
    event=types.SimpleNamespace(graph=fake_graph,schedule=None,event_plan=plan)
    hold=types.SimpleNamespace(graph=fake_graph,schedule=None,event_plan=None)
    def stub(arm):
        return types.SimpleNamespace(arm=arm,prior_epsilon=1e-8,
            encoder=lambda graph,schedule:types.SimpleNamespace(leaf_state=forcing),
            temporal_cell=lambda current,carry:current+carry,readout=lambda value:torch.zeros(value.shape[0],2),
            transport=triple if arm=="conditional_triple" else zero)
    output={}
    for arm in ("tree_only","first_order","conditional_triple"):
        unused,value=forward(stub(arm),event,state);output[arm]=value
        unused,initial=forward(stub(arm),event,None)
        ck("all_arms_share_zero_state_bootstrap",torch.equal(initial,forcing))
        unused,value=forward(stub(arm),hold,state)
        ck("annual_hold_semantics",torch.equal(value,forcing if arm=="tree_only" else forcing+state))
    logits,unused=forward(stub("tree_only"),event,state)
    expected=(prior_profile+1e-8)/(prior_profile+1e-8).sum(1,keepdim=True)
    ck("zero_residual_is_smoothed_persistence_including_uniform_missing_rows",
        torch.allclose(torch.softmax(logits,1),expected,atol=1e-7,rtol=1e-6))
    ck("zero_residual_preserves_same_FP32_persistence_rank_order",
        torch.equal(torch.argsort(logits,dim=1),torch.argsort(prior_profile,dim=1)))
    ck("zero_init_final_residual_layer_explicit_in_actual_model_source",
        "nn.init.zeros_(self.readout[-1].weight)" in text and "nn.init.zeros_(self.readout[-1].bias)" in text)
    ck("tree_only_has_no_past_partition_or_state_forcing",torch.equal(output["tree_only"],forcing))
    first_expected=torch.tensor([[3.],[5.],[3.],[5.]])+forcing
    triple_expected=torch.tensor([[3.],[3.],[3.],[7.]])+forcing
    ck("first_order_is_previous_partition_mean_at_event",torch.equal(output["first_order"],first_expected))
    ck("conditional_triple_uses_H051_innovation_at_same_event",torch.equal(output["conditional_triple"],triple_expected))
    ck("conditional_and_first_order_differ_only_in_carry",not torch.equal(output["first_order"],output["conditional_triple"]))
    calls=[]
    truth=torch.zeros((255,255),dtype=torch.float64);truth[:3,0]=1.
    present=torch.arange(255)<3
    def load_train(year):
        calls.append(year)
        if year>2009:raise AssertionError("Heldout target capability accessed")
        return TargetYearV1(year,truth,present)
    targets=prepare_train_targets_v1(load_train,"cpu")
    ck("supervision_requests_exactly_23_TRAIN_years_only",calls==list(range(1987,2010)))
    ck("training_targets_preserve_official_source_mask_and_normalization",
        targets[1987][0].shape==(3,255) and torch.allclose(targets[1987][0].sum(1),torch.ones(3)))
    loss=equal_year_soft_cross_entropy_v1(torch.zeros(255,255),targets[1987])
    ck("soft_CE_uses_all_destination_scores_and_only_official_queries",abs(float(loss)-math.log(255))<1e-6)
    ck("source_parameter_formula_matches_frozen_counts",20*64**2+2*255*64+24*64+255==116351
        and 3*64**2==12288 and 116351-12288==104063)
    trainer=(HERE/"trainer_v1.py").read_text()
    ck("all_inputs_resident_before_repeated_epoch_loop",trainer.index("sequence=sequence.to(device)")<
        trainer.index("targets=prepare_train_targets_v1")<trainer.index("for epoch in range(1,101)"))
    ck("primary_driver_cannot_select_future_half_count_or_confirmation_seeds",
        'campaign["seeds"]==[17]' in trainer and 'campaign["arms"]==list(PRIMARY_ARMS)' in trainer)
    receipt=dict(schema="TGB_trade_H051_model_free_semantics_fixture_v1",status="PASS",checks=checks,
        check_count=len(checks),distinct_checks=sorted(set(checks)),Torch_version=torch.__version__,
        exact_forward_step_executed_with_model_free_stubs=True,param_free_H051_transport_executed=True,
        trainable_encoder_GRU_head_constructed_or_forwarded=False,optimizer_or_fit=False,
        actual_label_dataset_checkpoint_read=False,GPU_API_call=False,worker_or_admission=False,
        source_refs=[dict(path=str(HERE/name),sha256=hashlib.sha256((HERE/name).read_bytes()).hexdigest())
            for name in ("preparation_v1.py","model_v1.py","targets_and_metrics_v1.py","trainer_v1.py",Path(__file__).name)])
    with Path(output).open("x") as f:json.dump(receipt,f,indent=2,sort_keys=True);f.write("\n")
    print(json.dumps(dict(status="PASS",check_count=len(checks),distinct_checks=len(set(checks)))))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--execute-fixture-root",action="store_true");p.add_argument("--output")
    a=p.parse_args()
    if not a.execute_fixture_root:
        print(json.dumps(dict(status="source_only_fixture_root_deferred",Torch_imported=False)));return 0
    if not a.output:raise ValueError("New root fixture receipt outside sealed packet required")
    execute_fixture(a.output);return 0

if __name__=="__main__":raise SystemExit(main())
