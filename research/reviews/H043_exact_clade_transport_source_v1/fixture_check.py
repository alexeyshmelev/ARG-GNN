"""Handmade H043 source validation only: no dataset/target/checkpoint reader."""
from pathlib import Path
import copy,hashlib,json,os,sys,traceback
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=ROOT/"research/reviews/H043_exact_clade_transport_source_v1"
sys.path.insert(0,str(ROOT/"src"))
for relative in ("tmp","cache","cache/torch","cache/extensions"):
    (HERE/relative).mkdir(parents=True,exist_ok=True)
for variable,relative in (("TMPDIR","tmp"),("XDG_CACHE_HOME","cache"),
                          ("TORCH_HOME","cache/torch"),("TORCH_EXTENSIONS_DIR","cache/extensions")):
    os.environ[variable]=str(HERE/relative)
import numpy as np
import torch
torch.set_num_threads(1)
import torch_geometric
from torch_geometric.nn import MessagePassing
from arggnn.anonymous_clade_flag_control_v1 import (
    build_clade_flag_arrays_v1,AnonymousCladeFlagControlGNNV1,RankEightCladeFlagAdapterV1,
)
from arggnn.anonymous_exact_clade_transport_v1 import (
    SOURCE,ARMS,ARRAY_FIELDS,INDEX_FIELDS,build_clade_transport_arrays_v1,
    AnonymousExactCladeTransportV1,RankEightExactCladeTransportV1,
    AnonymousExactCladeTransportGNNV1,
)


def forest(trees,n):
    parent,rows,masks = [],[],[]
    for tree in trees:
        leaves,by_mask = {},{}
        def visit(spec):
            if isinstance(spec,int):
                assert spec not in leaves
                node=len(parent);parent.append(-1);leaves[spec]=node
                mask=frozenset((spec,))
            else:
                children=[visit(s) for s in spec]
                node=len(parent);parent.append(-1)
                mask=frozenset().union(*(part[1] for part in children))
                for child,_ in children:parent[child]=node
            by_mask.setdefault(mask,node)  # deepest representative, postorder
            return node,mask
        visit(tree)
        assert set(leaves)==set(range(n))
        rows.append([leaves[i] for i in range(n)]);masks.append(by_mask)
    return np.asarray(parent,dtype=np.int32),np.asarray(rows,dtype=np.int32),masks


def construct(parent,samples):
    arrays,meta=build_clade_transport_arrays_v1(parent,samples,max_payload_bytes=16777216)
    reference,reference_meta=build_clade_flag_arrays_v1(parent,samples)
    np.testing.assert_array_equal(arrays["node_flags"],reference)
    assert meta["flag_meta"]==reference_meta
    assert set(arrays)==set(ARRAY_FIELDS) and meta["exact_bitsets_retained"] is False
    assert meta["clade_postorder_node_visits"]==parent.size
    assert meta["root_parent_link_steps"]<=parent.size
    assert meta["nearest_traversal_node_visits"]<=2*parent.size
    payload=AnonymousExactCladeTransportV1.from_verified_arrays(arrays,meta)
    assert payload.tensor_bytes()==meta["raw_array_bytes"]
    assert payload.tensor_bytes()==6*parent.size+24*meta["event_count"]
    return arrays,meta,payload


def records(arrays,inverse=None):
    result={}
    for i in range(len(arrays["weight"])):
        a,b=int(arrays["old_node"][i]),int(arrays["new_node"][i])
        if inverse is not None:
            a=int(inverse[a]) if a>=0 else -1
            b=int(inverse[b]) if b>=0 else -1
        key=(int(arrays["boundary_index"][i]),a,b)
        result[key]=(int(arrays["leaf_support"][i]),int(arrays["intersection_count"][i]),
                     float(arrays["weight"][i]))
    return result


def rejects(call):
    try:call()
    except (ValueError,TypeError):return
    raise AssertionError("Required rejection did not occur")


def run():
    assert os.environ.get("CUDA_VISIBLE_DEVICES")==""
    checks=[]
    parent,samples,masks=forest([((0,1),2),(0,(1,2))],3)
    arrays,meta,payload=construct(parent,samples)
    a=masks[0][frozenset((0,1))];b=masks[1][frozenset((1,2))]
    assert set(records(arrays))=={(0,a,-1),(0,a,b),(0,-1,b)}
    assert np.all(arrays["leaf_support"]==1)
    expected_weights={(0,a,-1):1/np.sqrt(2),(0,a,b):.5,(0,-1,b):1/np.sqrt(2)}
    for key,record in records(arrays).items():
        np.testing.assert_allclose(record[2],expected_weights[key],rtol=1e-6)
    assert meta["boundary_audit"][0]["one_sided_leaf_support"]==2
    checks.append("three_leaf_one_sided_events_and_exact_support")

    hidden=[(((0,1),(2,3)),(4,5)),(((0,2),(1,4)),(3,5))]
    parent,samples,masks=forest(hidden,6)
    arrays,meta,payload=construct(parent,samples)
    old_outer=masks[0][frozenset((0,1,2,3))]
    new_outer=masks[1][frozenset((0,1,2,4))]
    assert arrays["node_flags"][old_outer,1]==arrays["node_flags"][new_outer,0]==1
    assert old_outer not in arrays["old_node"] and new_outer not in arrays["new_node"]
    assert meta["boundary_audit"][0]["removed_without_nearest_support"]==1
    assert meta["boundary_audit"][0]["added_without_nearest_support"]==1
    assert meta["event_count"]==meta["raw_leaf_events"]==6
    checks.append("hidden_changed_ancestors_retained_as_H042_flags")

    perm=np.random.default_rng(101).permutation(parent.size)
    inverse=np.argsort(perm)
    relabelled=np.full_like(parent,-1)
    for node,ancestor in enumerate(parent):
        relabelled[perm[node]]=perm[ancestor] if ancestor>=0 else -1
    moved,moved_meta,moved_payload=construct(relabelled,perm[samples].astype(np.int32))
    assert records(moved,inverse)==records(arrays)
    np.testing.assert_array_equal(moved["node_flags"][perm],arrays["node_flags"])
    np.testing.assert_array_equal(moved["node_size"][perm],arrays["node_size"])
    checks.append("root_internal_and_leaf_node_ID_relabelling")

    columns=np.asarray([2,0,5,1,4,3])
    leafperm,_,_=construct(parent,samples[:,columns])
    for name in ARRAY_FIELDS:np.testing.assert_array_equal(leafperm[name],arrays[name])
    checks.append("joint_stable_leaf_identity_permutation")

    unary_parent,unary_samples,_=forest([(((0,1),),2),(0,(1,2))],3)
    _,unary_meta,_=construct(unary_parent,unary_samples)
    assert unary_meta["flag_meta"]["unary_equivalences_per_tree"]==[1,0]
    star_parent,star_samples,_=forest([(0,1,2),((0,1),2)],3)
    star,star_meta,_=construct(star_parent,star_samples)
    assert star_meta["event_count"]==1 and int(star["old_node"][0])==-1
    assert int(star["leaf_support"][0])==2
    assert float(star["weight"][0])==1.
    single_parent,single_samples,_=forest([((0,1),2)],3)
    _,single_meta,_=construct(single_parent,single_samples)
    assert single_meta["event_count"]==0
    checks.append("unary_collapse_polytomy_and_single_hold")

    def comb(order):
        value=(order[0],order[1])
        for leaf in order[2:]:value=(value,leaf)
        return value
    order=list(range(64));other=[0,2,1,*range(3,64)]
    comb_parent,comb_samples,_=forest([comb(order),comb(other)],64)
    _,comb_meta,_=construct(comb_parent,comb_samples)
    assert comb_meta["nearest_traversal_node_visits"]<=2*comb_parent.size
    checks.append("comb_memoized_roots_and_linear_nearest_traversal_visit_bound")

    rejects(lambda:build_clade_transport_arrays_v1(parent,samples,
        max_payload_bytes=meta["raw_array_bytes"]-1))
    altered={k:v.copy() for k,v in arrays.items()};altered["leaf_support"][0]+=1
    rejects(lambda:AnonymousExactCladeTransportV1.from_verified_arrays(altered,meta))
    cycle=parent.copy();cycle[0]=0
    rejects(lambda:build_clade_transport_arrays_v1(cycle,samples))
    checks.append("no_subsampling_cap_SHA_and_cycle_rejections")

    torch.manual_seed(101)
    layer=RankEightExactCladeTransportV1(event_block_size=2,checkpoint_event_blocks=False)
    reference_layer=RankEightCladeFlagAdapterV1();reference_layer.load_state_dict(layer.state_dict())
    assert sum(p.numel() for p in layer.parameters())==1024
    zero=torch.zeros(parent.size,32,dtype=torch.float32,requires_grad=True)
    rank=layer.rank_residual(zero,payload,arm="true_partner")
    assert rank.detach().abs().sum()>0
    output=layer(zero,payload,arm="true_partner")
    assert torch.count_nonzero(output)==0
    probe=torch.arange(output.numel(),dtype=torch.float32).reshape_as(output)/output.numel()
    (output*probe).sum().backward()
    assert layer.O.weight.grad is not None and layer.O.weight.grad.abs().sum()>0
    for projection in (layer.U,layer.V,layer.W):
        assert projection.weight.grad is not None
        assert torch.count_nonzero(projection.weight.grad)==0
    checks.append("zero_state_geometry_signal_and_zero_output_warm_start")

    with torch.no_grad():layer.O.weight.normal_(0.,.05)
    upward=torch.randn(parent.size,32,dtype=torch.float32)
    pu,pv,pw=layer.U(upward),layer.V(upward),layer.W(upward)
    torch.testing.assert_close(layer.local_rank_v1(pu,pv,pw,payload.flags.node_flags),
        reference_layer.rank_residual(upward,payload.flags),rtol=0,atol=0)
    checks.append("exact_original_H042_local_rank_term")

    outputs={}
    projection_calls={}
    for arm in ARMS:
        layer.zero_grad(set_to_none=True)
        value=upward.clone().requires_grad_(True)
        calls={name:0 for name in ("U","V","W","O")}
        hooks=[]
        for name in calls:
            def count(_module,_args,_output,name=name):calls[name]+=1
            hooks.append(getattr(layer,name).register_forward_hook(count))
        out=layer(value,payload,arm=arm)
        loss=out.square().sum()+(out*probe).sum()
        loss.backward()
        for hook in hooks:hook.remove()
        assert out.dtype==value.grad.dtype==torch.float32
        assert torch.isfinite(out).all() and torch.isfinite(value.grad).all()
        assert value.grad.abs().sum()>0
        for parameter in layer.parameters():
            assert parameter.grad is not None and parameter.grad.dtype==torch.float32
            assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum()>0
        assert calls==dict(U=1,V=1,W=1,O=1)
        outputs[arm]=out.detach();projection_calls[arm]=calls
    assert not torch.allclose(outputs["true_partner"],outputs["own_state"])
    checks.append("both_arms_finite_FP32_input_and_all_projection_gradients")
    checks.append("identical_once_per_node_projection_work_and_distinct_foreign_gather")

    relabelled_up=torch.empty_like(upward);relabelled_up[torch.from_numpy(perm)]=upward
    for arm in ARMS:
        torch.testing.assert_close(layer(relabelled_up,moved_payload,arm=arm)[torch.from_numpy(perm)],
            layer(upward,payload,arm=arm),rtol=1e-5,atol=1e-6)
    checks.append("nonzero_layer_output_node_relabelling_equivariance")

    check_layer=RankEightExactCladeTransportV1(event_block_size=1,checkpoint_event_blocks=True)
    check_layer.load_state_dict(layer.state_dict())
    for arm in ARMS:
        layer.zero_grad(set_to_none=True);check_layer.zero_grad(set_to_none=True)
        left=upward.clone().requires_grad_(True);right=upward.clone().requires_grad_(True)
        result=layer(left,payload,arm=arm);checked=check_layer(right,payload,arm=arm)
        torch.testing.assert_close(result,checked,rtol=1e-5,atol=1e-6)
        (result.square().sum()+(result*probe).sum()).backward()
        (checked.square().sum()+(checked*probe).sum()).backward()
        torch.testing.assert_close(left.grad,right.grad,rtol=1e-5,atol=1e-6)
        for p,q in zip(layer.parameters(),check_layer.parameters()):
            torch.testing.assert_close(p.grad,q.grad,rtol=1e-5,atol=1e-6)
    checks.append("checkpointed_block_forward_backward_matches_uncheckpointed")

    torch.manual_seed(101);baseline=AnonymousCladeFlagControlGNNV1()
    baseline_rng=torch.get_rng_state().clone()
    models={}
    for arm in ARMS:
        torch.manual_seed(101)
        model=AnonymousExactCladeTransportGNNV1(arm=arm)
        assert isinstance(model.pyg_upward,MessagePassing) and isinstance(model.pyg_downward,MessagePassing)
        assert sum(p.numel() for p in model.parameters())==30788
        assert sum(p.numel() for p in model.clade_coupling.parameters())==1024
        assert set(model.state_dict())==set(baseline.state_dict())
        for key,value in baseline.state_dict().items():assert torch.equal(model.state_dict()[key],value)
        assert torch.equal(torch.get_rng_state(),baseline_rng)
        models[arm]=dict(total_parameters=30788,added_parameters=1024,
            identical_H042_initial_state_and_CPU_rng=True,actual_inherited_PyG_MessagePassing=True)
    checks.append("both_full_models_matched_30788_parameters_1024_adapter")
    checks.append("original_H042_initial_parameters_and_rng_preserved")
    return dict(checks=checks,models=models,projection_calls=projection_calls,
        hidden_ancestor_audit=meta["boundary_audit"],comb_traversal_counts={
            k:comb_meta[k] for k in ("native_nodes","root_parent_link_steps",
            "clade_postorder_node_visits","nearest_traversal_node_visits")})


def main():
    result=run()
    def ref(path):
        raw=path.read_bytes()
        return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
    result.update(schema="H043_handmade_CPU_source_fixture_receipt_v1",status="passed",
        source_ref=ref(ROOT/SOURCE),fixture_ref=ref(Path(__file__)),
        torch=str(torch.__version__),torch_geometric=torch_geometric.__version__,
        CPU_threads=torch.get_num_threads(),CUDA_VISIBLE_DEVICES=os.environ["CUDA_VISIBLE_DEVICES"],
        handmade_max_panel_leaves=64,full_model_forward_or_training_run=False,
        observation_targets_checkpoints_or_H040_VAL_read=False,real_data_arrays_used=False,
        guard_admission_GPU_install_or_Git_actions=False)
    path=HERE/"fixture_receipt.json"
    path.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True))
if __name__=="__main__":
    try:main()
    except Exception:
        print(traceback.format_exc(),flush=True)
        raise
