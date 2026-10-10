"""H021 inherited whole-tree backbone: handmade CPU fixtures, no fitting."""
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path

import pytest
import torch

from arggnn.census_cut_exchange_v1 import (
    ObservedDatedTreeV1, prepare_whole_tree_encoder_inputs_v1,
    prepare_census_exchange_bank_v1, permuted_group_control_v1, build_census_cut_partition_v1)
from arggnn.census_tree_encoder_v1 import (
    CensusWholeTreeEncoderV1, CensusTreeGNNV1, retained_observed_features_v1,
    encode_whole_tree_contexts_v1, connected_observed_tree_v1, prepare_connected_whole_tree_inputs_v1,
    prepare_retained_observed_forest_v1)

Q = torch.tensor([101, 103, 107, 109])
R = torch.tensor([201, 203])
SCORED = Q[:2]
INTERVALS = torch.tensor([[0., .5], [.5, 1.]], dtype=torch.float64)
POSITIONS = torch.tensor([.5, .0, .12, .5, .49, .82, 1/6, .999], dtype=torch.float64)


@pytest.fixture(autouse=True)
def handmade_cpu_only():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    torch.set_num_threads(1)


def tree(which=0):
    parent = [6, 6, 7, 7, 8, 9, 8, 9, 9, -1]
    times = [0., 100., 200., 300., 400., 600., 300., 450., 550., 800.]
    if which:
        parent[:4] = [7, 6, 6, 7]
        times[6:8] = [330., 500.]
    return ObservedDatedTreeV1(torch.tensor(parent), torch.tensor(times, dtype=torch.float64), torch.cat((Q, R)), torch.arange(6))


def features(t, dtype=torch.float64):
    generator = torch.Generator().manual_seed(4241)
    value = .2 * torch.randn(len(t.parent), 47, generator=generator, dtype=dtype)
    value[:, 6] = (torch.log1p(t.node_times)/math.log1p(10000.)).to(dtype)
    return value


def prepared(which=0, dtype=torch.float64, gradients=False):
    t = tree(which)
    value = features(t, dtype).requires_grad_(gradients)
    return prepare_whole_tree_encoder_inputs_v1(t, value), value


def close(actual, expected, dtype=torch.float64):
    rt, at = (1e-10, 1e-12) if dtype == torch.float64 else (2e-4, 2e-5)
    torch.testing.assert_close(actual, expected, rtol=rt, atol=at)


def model(dtype=torch.float64):
    torch.manual_seed(171)
    return CensusTreeGNNV1(7, 3, 4, 2, 3, initial_exchange=(.7, 1.1, 1.8), initial_damping=.5).to(dtype=dtype)


def bank(dtype=torch.float64):
    return prepare_census_exchange_bank_v1([tree(0), tree(1)], INTERVALS, POSITIONS, Q, R, SCORED, num_anchors=3).to("cpu", dtype)


def loss(output):
    weight = torch.linspace(-.7, 1.1, output.numel(), dtype=output.dtype).reshape_as(output)
    return (weight * output).sum() + .13 * output.square().sum()


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_explicit_observed_node_type_age_and_edge_time_features(dtype):
    inputs, _ = prepared(dtype=dtype)
    observed = retained_observed_features_v1(inputs)
    assert observed.node.shape == (10, 51) and observed.edge_time.shape == (10, 1)
    torch.testing.assert_close(observed.node[:, :47], inputs.features, rtol=0, atol=0)
    close(observed.node[:, 47], (torch.log1p(inputs.node_times)/math.log1p(10000.)).to(dtype), dtype)
    assert observed.node[:, 48].tolist() == [0.]*9+[1.]
    assert observed.node[:, 49].tolist() == [1.]*6+[0.]*4
    assert observed.node[:, 50].tolist() == [0.]*6+[1.]*4
    close(observed.edge_time[:, 0], torch.log1p(inputs.branch_duration)/math.log1p(10000.))
    edges = [(parent, child) for parent, children in enumerate(observed.children) for child in children]
    assert len(edges) == 9 and {child for _, child in edges} == set(range(9))
    renamed = replace(inputs, sample_ids=inputs.sample_ids + 10000)
    same = retained_observed_features_v1(renamed)
    torch.testing.assert_close(same.node, observed.node, rtol=0, atol=0)
    torch.testing.assert_close(same.edge_time, observed.edge_time, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_every_retained_node_and_message_parameter_has_meaningful_gradient(dtype):
    inputs, original = prepared(dtype=dtype, gradients=True)
    torch.manual_seed(291)
    encoder = CensusWholeTreeEncoderV1(7).to(dtype=dtype)
    q, r = encode_whole_tree_contexts_v1(encoder, [inputs], Q, R)
    loss(torch.cat((q.flatten(), r.flatten()))).backward()
    assert original.grad is not None and torch.isfinite(original.grad).all()
    assert (original.grad.abs().sum(1) > 1e-9).all()  # all leaves and every branching interior/root
    for name, parameter in encoder.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 1e-9, name


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_branching_interior_and_older_reference_change_query_contexts(dtype):
    inputs, _ = prepared(dtype=dtype)
    torch.manual_seed(37)
    encoder = CensusWholeTreeEncoderV1(7).to(dtype=dtype)
    base_q, _ = encode_whole_tree_contexts_v1(encoder, [inputs], Q, R)
    for changed_node in (8, 9, 5):  # branching interiors/root and age600 pure R
        value = inputs.features.clone()
        value[changed_node, 3] += 2.
        changed = replace(inputs, features=value)
        q, _ = encode_whole_tree_contexts_v1(encoder, [changed], Q, R)
        assert (q - base_q).abs().max() > 1e-5
    changed_tree = tree()
    ages = changed_tree.node_times.clone()
    ages[8] += 60.
    changed_tree = replace(changed_tree, node_times=ages)
    changed_inputs = prepare_whole_tree_encoder_inputs_v1(changed_tree, features(changed_tree, dtype))
    q, _ = encode_whole_tree_contexts_v1(encoder, [changed_inputs], Q, R)
    assert (q-base_q).abs().max() > 1e-6  # explicit native age and additive endpoint edge time
    assert inputs.node_times[inputs.sample_nodes[-2:]].tolist() == [400., 600.]


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_native_node_and_sample_order_permutation_preserves_contexts_and_gradients(dtype):
    t = tree()
    original_features = features(t, dtype).requires_grad_()
    inputs = prepare_whole_tree_encoder_inputs_v1(t, original_features)
    node_order = torch.tensor([8, 3, 9, 1, 6, 5, 0, 7, 2, 4])
    inverse = torch.argsort(node_order)
    native_parent = t.parent[node_order]
    parent = torch.where(native_parent < 0, native_parent, inverse[native_parent.clamp_min(0)])
    sample_order = torch.tensor([4, 0, 5, 2, 1, 3])
    permuted_tree = ObservedDatedTreeV1(parent, t.node_times[node_order], t.sample_ids[sample_order], inverse[t.sample_nodes[sample_order]])
    permuted = prepare_whole_tree_encoder_inputs_v1(permuted_tree, original_features[node_order])
    torch.manual_seed(711)
    encoder = CensusWholeTreeEncoderV1(7).to(dtype=dtype)
    base = encode_whole_tree_contexts_v1(encoder, [inputs], Q, R)
    query_order, reference_order = torch.tensor([2, 0, 3, 1]), torch.tensor([1, 0])
    changed = encode_whole_tree_contexts_v1(encoder, [permuted], Q[query_order], R[reference_order])
    close(changed[0], base[0][:, query_order], dtype)
    close(changed[1], base[1][:, reference_order], dtype)
    objectives = [sum(x.square().sum() for x in pair) for pair in (base, changed)]
    gradients = [torch.autograd.grad(value, (original_features, *tuple(encoder.parameters())), retain_graph=True) for value in objectives]
    for left, right in zip(*gradients):
        close(left, right, dtype)
    renamed = replace(inputs, sample_ids=inputs.sample_ids+50000)
    same = encode_whole_tree_contexts_v1(encoder, [renamed], Q+50000, R+50000)
    for left, right in zip(base, same):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_anonymous_serial_unary_insertion_is_exact_after_checked_suppression():
    t = tree()
    parent = torch.cat((t.parent.clone(), torch.tensor([8])))
    parent[6] = 10
    serial = replace(t, parent=parent, node_times=torch.cat((t.node_times, torch.tensor([350.], dtype=torch.float64))))
    original_features = features(t)
    serial_features = torch.cat((original_features, torch.full((1,47), 50., dtype=torch.float64)))
    serial_features[10,6] = math.log1p(350.)/math.log1p(10000.)
    before = prepare_whole_tree_encoder_inputs_v1(t, original_features)
    after = prepare_whole_tree_encoder_inputs_v1(serial, serial_features)
    torch.manual_seed(277)
    encoder = CensusWholeTreeEncoderV1(7).double()
    left, right = encoder(before), encoder(after)
    torch.testing.assert_close(left, right, rtol=0, atol=0)
    gradients = [torch.autograd.grad(loss(value), tuple(encoder.parameters()), retain_graph=True) for value in (left,right)]
    for a,b in zip(*gradients):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert after.branch_duration[6] == 250. and 10 not in after.native_vertex_index.tolist()


def test_unary_root_is_retained_and_processed_but_sampled_interior_is_rejected():
    t = tree()
    parent = torch.cat((t.parent.clone(), torch.tensor([-1])))
    parent[9] = 10
    augmented = replace(t, parent=parent, node_times=torch.cat((t.node_times, torch.tensor([900.], dtype=torch.float64))))
    value = torch.cat((features(t), torch.ones(1,47,dtype=torch.float64)))
    value[10,6] = math.log1p(900.)/math.log1p(10000.)
    inputs = prepare_whole_tree_encoder_inputs_v1(augmented, value)
    assert len(inputs.parent) == 11 and inputs.parent[10] == -1 and inputs.parent[9] == 10
    encoder = CensusWholeTreeEncoderV1(7).double()
    original = encoder(prepare_whole_tree_encoder_inputs_v1(t, features(t)))
    changed = encoder(inputs)
    assert (changed[:6]-original[:6]).abs().max() > 1e-6
    sampled_interior = replace(t, sample_nodes=torch.tensor([6,1,2,3,4,5]))
    with pytest.raises(ValueError, match="all and only observed sampled leaves"):
        prepare_whole_tree_encoder_inputs_v1(sampled_interior, features(t))
    with pytest.raises(ValueError, match="all and only observed sampled leaves"):
        encoder(replace(inputs, sample_nodes=torch.tensor([6,1,2,3,4,5])))


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("control", ["proposed", "eta0", "permuted_groups", "raw_diagonal", "local"])
def test_shared_full_gnn_two_depth_all_controls_have_active_and_inactive_gradients(dtype, control):
    inputs_and_features = [prepared(which, dtype, True) for which in (0,1)]
    inputs = [x[0] for x in inputs_and_features]
    b = bank(dtype)
    if control == "permuted_groups":
        b = permuted_group_control_v1(b, torch.tensor([2,3,1,0]))
    gnn = model(dtype)
    output = gnn(inputs, b, control=control, readout_chunk_size=2)
    assert output.logits.shape == (8,2,4) and output.anchor_hidden.shape == (3,4,7)
    assert output.target_hidden.shape == (8,4,7) and output.scored_query_ids.tolist() == SCORED.tolist()
    assert output.encoder_reference_context.shape == (2,2,7)
    loss(output.logits).backward()
    for _, value in inputs_and_features:
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert (value.grad.abs().sum(1) > 1e-10).all()
    for name, parameter in gnn.named_parameters():
        inactive = (name.endswith("raw_eta") and control in {"eta0","local"}) or (name.endswith("raw_damping") and control == "local")
        if inactive:
            assert parameter.grad is None, name
        else:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name
    assert all(not name.startswith("id") and "population" not in name for name,_ in gnn.named_parameters())
    # No optimizer, fitting loop or actual labels are supplied.


def test_control_forwards_share_exact_encoder_source_projections_gates_and_decoder():
    inputs = [prepared(which)[0] for which in (0,1)]
    b = bank()
    gnn = model()
    objects = tuple(id(parameter) for parameter in gnn.parameters())
    calls, anchor_inputs = [], []
    hook = gnn.encoder.register_forward_hook(lambda *args: calls.append(1))
    source_hook = gnn.layers[0].query_projection.register_forward_pre_hook(lambda module,args: anchor_inputs.append(args[0].detach().clone()))
    outputs = {}
    for control in ("proposed", "eta0", "permuted_groups", "raw_diagonal", "local"):
        control_bank = permuted_group_control_v1(b, torch.tensor([2,3,1,0])) if control=="permuted_groups" else b
        start = len(anchor_inputs)
        outputs[control] = gnn(inputs, control_bank, control=control)
        torch.testing.assert_close(anchor_inputs[start], outputs["proposed"].encoder_query_context[b.layout.anchor_graph], rtol=0, atol=0)
        torch.testing.assert_close(outputs[control].encoder_query_context, outputs["proposed"].encoder_query_context, rtol=0,atol=0)
        torch.testing.assert_close(outputs[control].encoder_reference_context, outputs["proposed"].encoder_reference_context, rtol=0,atol=0)
        assert tuple(id(parameter) for parameter in gnn.parameters()) == objects
    hook.remove();source_hook.remove()
    assert len(calls) == 5
    for control in ("eta0","permuted_groups","raw_diagonal","local"):
        assert (outputs[control].logits-outputs["proposed"].logits).abs().max() > 1e-8
    shuffled_local = gnn(inputs, permuted_group_control_v1(b, torch.tensor([2,3,1,0])), control="local")
    torch.testing.assert_close(shuffled_local.logits, outputs["local"].logits, rtol=0,atol=0)


def test_full_gnn_locality_and_finite_genomic_dependence():
    inputs = [prepared(which)[0] for which in (0,1)]
    b=bank();gnn=model()
    changed_features = inputs[1].features.clone()
    changed_features[4,2] += 4.  # pure reference in the other physical hold
    changed_inputs = [inputs[0], replace(inputs[1], features=changed_features)]
    local = gnn(inputs,b,control="local")
    changed_local = gnn(changed_inputs,b,control="local")
    in_graph0 = b.target_graph==0
    torch.testing.assert_close(local.logits[in_graph0], changed_local.logits[in_graph0], rtol=0,atol=0)
    assert (local.logits[~in_graph0]-changed_local.logits[~in_graph0]).abs().max()>1e-5
    finite = gnn(inputs,b)
    changed_finite = gnn(changed_inputs,b)
    assert (finite.logits[in_graph0]-changed_finite.logits[in_graph0]).abs().max()>1e-6


def test_malformed_retained_geometry_sweeps_rosters_and_bank_dates_fail_closed():
    inputs,_=prepared();encoder=CensusWholeTreeEncoderV1(7).double()
    bad_duration=inputs.branch_duration.clone();bad_duration[0]+=1.
    with pytest.raises(ValueError,match="endpoint differences"):
        encoder(replace(inputs,branch_duration=bad_duration))
    with pytest.raises(ValueError,match="every retained node"):
        encoder(replace(inputs,upward_order=inputs.upward_order[:-1]))
    with pytest.raises(ValueError,match="every retained parent-child"):
        encoder(replace(inputs,downward_order=tuple(reversed(inputs.downward_order))))
    with pytest.raises(ValueError,match="provenance"):
        encoder(replace(inputs,native_vertex_index=torch.zeros(10,dtype=torch.long)))
    with pytest.raises(ValueError,match="checked Q/R roster"):
        encode_whole_tree_contexts_v1(encoder,[inputs],Q,R[:1])
    gnn=model();b=bank();both=[inputs,prepared(1)[0]]
    with pytest.raises(ValueError,match="every serialized"):
        gnn([inputs],b)
    with pytest.raises(ValueError,match="Unknown H021"):
        gnn(both,b,control="unregistered")
    changed=tree(1);times=changed.node_times.clone();times[0]=1.
    changed=replace(changed,node_times=times)
    wrong=prepare_whole_tree_encoder_inputs_v1(changed,features(changed))
    with pytest.raises(ValueError,match="queries/native ages"):
        gnn([inputs,wrong],b)
    parts=list(b.partitions);parts[1]=replace(parts[1],scored_query_mask=~parts[1].scored_query_mask)
    with pytest.raises(ValueError,match="Scored query roster"):
        gnn(both,replace(b,partitions=tuple(parts)))


def test_previously_sealed_exchange_core_remains_exactly_bound():
    root=Path(__file__).resolve().parents[1]
    binding=json.loads((root/'research/reviews/census_cut_exchange_v1_author/core_binding.json').read_text())
    for path,expected in binding['core_sha256'].items():
        assert hashlib.sha256((root/path).read_bytes()).hexdigest()==expected
    receipt=root/'research/reviews/census_cut_exchange_v1_author/receipt.json'
    assert hashlib.sha256(receipt.read_bytes()).hexdigest()==binding['receipt_sha256']



def test_unrelated_isolated_marginal_nodes_are_excluded_from_encoder_and_census_groups():
    t=tree();value=features(t)
    unrelated_parent=torch.tensor([-1,-1,-1,14,-1])
    unrelated_times=torch.tensor([900.,1200.,750.,1000.,1400.],dtype=torch.float64)
    enlarged=replace(t,parent=torch.cat((t.parent,unrelated_parent)),node_times=torch.cat((t.node_times,unrelated_times)))
    irrelevant=torch.full((5,47),100.,dtype=torch.float64)
    irrelevant[:,6]=torch.log1p(unrelated_times)/math.log1p(10000.)
    enlarged_features=torch.cat((value,irrelevant))
    connected=connected_observed_tree_v1(enlarged,enlarged_features)
    assert connected.excluded_non_sample_nodes==5 and connected.native_vertex_index.tolist()==list(range(10))
    original=prepare_whole_tree_encoder_inputs_v1(t,value)
    selected=prepare_connected_whole_tree_inputs_v1(enlarged,enlarged_features)
    torch.manual_seed(351);encoder=CensusWholeTreeEncoderV1(7).double()
    torch.testing.assert_close(encoder(original),encoder(selected),rtol=0,atol=0)
    before=build_census_cut_partition_v1(t,Q,R,SCORED)
    after=build_census_cut_partition_v1(connected.tree,Q,R,SCORED)
    assert before.group_members==after.group_members
    assert after.query_ages.tolist()==before.query_ages.tolist()
    with pytest.raises(ValueError,match="single complete"):
        prepare_whole_tree_encoder_inputs_v1(enlarged,enlarged_features)
    # Two sample-bearing components cannot be repaired by discarding a sample.
    broken=t.parent.clone();broken[4]=-1
    with pytest.raises(ValueError,match="single complete"):
        connected_observed_tree_v1(replace(t,parent=broken),value)


def test_connected_union_adapter_preserves_sparse_native_provenance_and_feature_gradients():
    t=tree();value=features(t).requires_grad_()
    # Interleave unused roots and reorder all connected vertices.
    order=torch.tensor([8,3,9,1,6,5,0,7,2,4]);inverse=torch.argsort(order)
    positions=torch.tensor([1,3,4,6,7,9,10,11,13,14])
    parent=torch.full((16,),-1,dtype=torch.long)
    times=torch.full((16,),1700.,dtype=torch.float64)
    f=torch.zeros(16,47,dtype=torch.float64)
    native_parent=t.parent[order]
    parent[positions]=torch.where(native_parent<0,native_parent,positions[inverse[native_parent.clamp_min(0)]])
    times[positions]=t.node_times[order]
    f[:,6]=torch.log1p(times)/math.log1p(10000.)
    f[positions]=value[order]
    supplied=ObservedDatedTreeV1(parent,times,t.sample_ids,positions[inverse[t.sample_nodes]])
    selected=prepare_connected_whole_tree_inputs_v1(supplied,f)
    assert selected.native_vertex_index.tolist()==positions.tolist()
    torch.manual_seed(33);encoder=CensusWholeTreeEncoderV1(7).double()
    original=prepare_whole_tree_encoder_inputs_v1(t,value)
    base=encode_whole_tree_contexts_v1(encoder,[original],Q,R)
    changed=encode_whole_tree_contexts_v1(encoder,[selected],Q,R)
    for left,right in zip(base,changed):close(left,right)
    grads=[torch.autograd.grad(sum(x.square().sum() for x in pair),value,retain_graph=True)[0] for pair in (base,changed)]
    close(grads[0],grads[1])
    bad=parent.clone();bad[positions[inverse[0]]]=positions[inverse[0]]
    with pytest.raises(ValueError,match="invalid parent or cycle"):
        connected_observed_tree_v1(replace(supplied,parent=bad),f)



def test_full_gnn_float64_five_point_directional_derivative_for_features_and_all_parameters():
    from torch.func import functional_call
    pairs=[prepared(which,gradients=True) for which in (0,1)]
    inputs=[x[0] for x in pairs];values=[x[1] for x in pairs]
    gnn=model();b=bank();named=list(gnn.named_parameters())
    base=loss(gnn(inputs,b).logits)
    gradients=torch.autograd.grad(base,(*values,*(parameter for _,parameter in named)))
    generator=torch.Generator().manual_seed(551)
    directions=[torch.randn(value.shape,dtype=torch.float64,generator=generator) for value in (*values,*(parameter for _,parameter in named))]
    for direction in directions[:2]:direction[:,6]=0.  # native observed age is a fixed input constraint
    expected=sum((gradient*direction).sum() for gradient,direction in zip(gradients,directions))
    step=2e-4
    points={}
    for k in (-2,-1,1,2):
        shifted_inputs=[replace(item,features=item.features.detach()+k*step*direction) for item,direction in zip(inputs,directions[:2])]
        parameters={name:parameter.detach()+k*step*direction for (name,parameter),direction in zip(named,directions[2:])}
        points[k]=loss(functional_call(gnn,parameters,(shifted_inputs,b)).logits)
    difference=(points[-2]-8*points[-1]+8*points[1]-points[2])/(12*step)
    torch.testing.assert_close(expected,difference,rtol=2e-8,atol=1e-9)


def test_tensor_census_bank_alignment_and_prequalified_donor_contract(monkeypatch):
    import arggnn.census_cut_exchange_v1 as core
    inputs=[prepared(which)[0] for which in (0,1)];b=bank();gnn=model()
    with pytest.raises(ValueError,match="observed census groups"):
        gnn([inputs[1],inputs[0]],b)
    shuffled=permuted_group_control_v1(b,torch.tensor([2,3,1,0]))
    with pytest.raises(ValueError,match="Observed H021 controls"):
        gnn(inputs,shuffled,control="proposed")
    monkeypatch.setattr(core,"observed_tmrca_donor_weights_v1",lambda *a,**k: (_ for _ in ()).throw(AssertionError("forward MRCA recomputation")))
    assert torch.isfinite(gnn(inputs,b).logits).all()


def test_forest_level_batches_match_separate_whole_trees_and_cover_every_edge():
    inputs=[prepared(which)[0] for which in (0,1)]
    forest=prepare_retained_observed_forest_v1(inputs)
    assert forest.graph_offsets==(0,10,20)
    assert sum(len(x) for x in forest.upward_levels)==sum(len(x) for x in forest.downward_levels)==20
    assert sum(len(x) for x in forest.upward_children)==18
    assert set(torch.cat(forest.upward_children).tolist())==set(range(20))-{9,19}
    torch.manual_seed(481);encoder=CensusWholeTreeEncoderV1(7).double()
    separate=[encoder(item) for item in inputs]
    shapes=[]
    hooks=[layer.register_forward_pre_hook(lambda module,args:shapes.append(tuple(args[0].shape))) for layer in encoder.modules() if isinstance(layer,torch.nn.Linear)]
    encoder(inputs[0])
    single_tree_call_count=len(shapes)
    shapes.clear()
    together=encoder(forest)
    for hook in hooks:hook.remove()
    assert len(shapes)==single_tree_call_count  # doubling trees doubles rows, not neural launch count
    assert all(len(shape)==2 for shape in shapes)
    assert any(shape[0]==20 for shape in shapes) and max(shape[0] for shape in shapes)==20
    # Neural call count depends on levels, not the20 vertices.
    for left,right in zip(separate,together):close(left,right)
    gradient_left=torch.autograd.grad(sum(x.square().sum() for x in separate),tuple(encoder.parameters()),retain_graph=True)
    gradient_right=torch.autograd.grad(sum(x.square().sum() for x in together),tuple(encoder.parameters()))
    for left,right in zip(gradient_left,gradient_right):close(left,right)



def test_handmade_tskit_marginal_tree_excludes_other_marginal_internal_nodes():
    import tskit
    tables=tskit.TableCollection(sequence_length=1.)
    t=tree();extra_times=[330.,500.,550.,800.,900.]
    for node,age in enumerate(t.node_times.tolist()+extra_times):
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE if node<6 else 0,time=age)
    for child,parent in enumerate(t.parent.tolist()):
        if parent>=0:tables.edges.add_row(0.,.5,parent,child)
    right_parent={0:11,1:10,2:10,3:11,4:12,5:13,10:12,11:13,12:13}
    for child,parent in right_parent.items():tables.edges.add_row(.5,1.,parent,child)
    tables.sort();ts=tables.tree_sequence()
    assert ts.num_samples==6 and ts.num_nodes==15 and ts.num_trees==2
    marginal=ts.at(.25)
    raw=ObservedDatedTreeV1(torch.tensor([marginal.parent(node) for node in range(ts.num_nodes)]),
        torch.tensor([ts.node(node).time for node in range(ts.num_nodes)],dtype=torch.float64),t.sample_ids,t.sample_nodes)
    base_features=features(t)
    extra=torch.zeros(5,47,dtype=torch.float64)
    extra[:,6]=torch.log1p(torch.tensor(extra_times,dtype=torch.float64))/math.log1p(10000.)
    value=torch.cat((base_features,extra))
    connected=connected_observed_tree_v1(raw,value)
    assert connected.excluded_non_sample_nodes==5 and len(connected.tree.parent)==10
    assert all(raw.parent[node]==-1 for node in range(10,15))
    original=prepare_whole_tree_encoder_inputs_v1(t,base_features)
    selected=prepare_connected_whole_tree_inputs_v1(raw,value)
    torch.manual_seed(611);encoder=CensusWholeTreeEncoderV1(7).double()
    torch.testing.assert_close(encoder(original),encoder(selected),rtol=0,atol=0)
    before=build_census_cut_partition_v1(t,Q,R,SCORED)
    after=build_census_cut_partition_v1(connected.tree,Q,R,SCORED)
    assert before.group_members==after.group_members
