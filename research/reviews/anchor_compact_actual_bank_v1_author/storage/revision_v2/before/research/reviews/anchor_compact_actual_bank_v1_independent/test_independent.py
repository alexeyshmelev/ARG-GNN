"""Independent handmade CPU evidence. No project arrays or actual CUDA APIs."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import copy
import hashlib
import json
import os
import pytest
import torch
import arggnn.anchor_compact_actual_bank_v1 as m
from arggnn.anchor_kernel import AnchorKernelBank, AnchorKernelModel, AnchorOperators, _layout
from arggnn.kron_interface import LeafInterfaceBank, QueryInterfaceCache
from arggnn.anchor_spectral_compact_v1 import anchor_spectral_compact_action_v1

OUT = Path(__file__).parent
METRICS = []

@pytest.fixture(scope='session', autouse=True)
def boundary():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    yield
    assert not torch.cuda.is_initialized()
    (OUT/'independent_metrics.json').write_text(json.dumps(METRICS, indent=2)+'\n')


def handmade(q=7, r=4, features=5, anchors=5):
    gen = torch.Generator().manual_seed(937)
    intervals = torch.tensor([[.5,.875],[.125,.29],[.29,.5]], dtype=torch.float64)
    positions = torch.tensor([.65,.125,.5,.29,.500000001,.8,.875-1e-9,.65], dtype=torch.float64)
    starts = torch.argsort(intervals[:,0])
    target = starts[torch.searchsorted(intervals[starts,0].contiguous(),positions,right=True)-1]
    matrices=[]
    for g in range(3):
        upper=torch.triu(torch.rand(q,q,generator=gen,dtype=torch.float64), diagonal=1)
        adjacency=(upper+upper.T)*(1.0/q+.03)
        leakage=.2+.2*torch.rand(q,generator=gen,dtype=torch.float64)
        base=torch.diag(adjacency.sum(-1)+leakage)-adjacency
        matrices.append(torch.stack((base,.85*base+.12*torch.eye(q),1.2*base+.05*torch.eye(q))))
    cqq=torch.stack(matrices)
    fractions=torch.rand(3,3,q,r,generator=gen,dtype=torch.float64)+.1
    fractions=fractions/fractions.sum(-1,keepdim=True)
    cqr=-cqq.sum(-1)[...,None]*fractions
    eig,vec=torch.linalg.eigh(cqq)
    nodes=torch.roll(torch.arange(q+r,dtype=torch.long)*11+1001, shifts=3)
    dynamics=torch.cat((torch.arange(0,q,2),torch.arange(1,q,2)))
    references=torch.arange(q+r-1,q-1,-1)
    sample_ids=torch.sort(nodes).values
    qi=torch.searchsorted(sample_ids,nodes[dynamics])
    ri=torch.searchsorted(sample_ids,nodes[references])
    order=torch.cat((qi,ri))
    qr=torch.zeros(3,q+r,q+r,dtype=torch.float64)
    qr[:,:q,:q]=cqq[:,0]
    qr[:,:q,q:]=cqr[:,0]
    qr[:,q:,:q]=cqr[:,0].transpose(-1,-2)
    qr[:,q:,q:]=torch.diag_embed(-cqr[:,0].sum(-2))
    response=torch.empty_like(qr)
    response[:,order[:,None],order[None,:]]=qr
    widths=intervals[:,1]-intervals[:,0]
    cache=QueryInterfaceCache(response,torch.ones(3),cqq[:,0],cqr[:,0],eig[:,0],vec[:,0],
        qi,ri,nodes[dynamics],nodes[references],widths/widths.sum(),cqq[:,0].mean(0),eig[0,0],vec[0,0])
    indexes=torch.arange(3*(q+r)*features,dtype=torch.float64).reshape(3,q+r,features)
    leaf=torch.sin(indexes/19)+.3*torch.cos(indexes/37)
    output=torch.roll(torch.arange(q),shifts=17)[:min(120,q)]
    actual=dynamics[output]
    if features==47:
        leaf[:,:,4]=torch.isin(sample_ids,nodes[actual]).double()
    columns=tuple(c for c in range(48) if c!=7) if features==47 else tuple(range(features))
    bank=LeafInterfaceBank(response,sample_ids,torch.arange(3*(q+r)).reshape(3,q+r),leaf,
        columns,intervals,nodes[dynamics],positions,target,torch.tensor([.125,.875],dtype=torch.float64),
        (),.0001,10000.,query_cache=cache)
    age=AnchorOperators(cqq,cqr,eig,vec)
    plain=AnchorOperators(cqq[:,0:1].expand(-1,3,-1,-1),cqr[:,0:1].expand(-1,3,-1,-1),
        eig[:,0:1].expand(-1,3,-1),vec[:,0:1].expand(-1,3,-1,-1))
    prepared=AnchorKernelBank(bank,_layout(bank,anchors),{'plain':plain,'age':age},3)
    return prepared,dict(nodes=nodes,dynamics=dynamics,references=references,actual=actual,output=output)


def pair(prepared,dtype,coupling='full',geometry='age',stress=False,candidate=False):
    if candidate:
        native,compact=m.qualification_models_v1(geometry,'rate_mixture_stress' if stress else 'initial')
        native.coupling=compact.coupling=coupling
    else:
        args=dict(input_dim=prepared.bank.leaf_x.shape[-1],hidden_dim=9,num_layers=2,
            latent_dim=3,kernel_depth=2,num_scales=3,num_anchors=prepared.layout.anchors.numel(),
            geometry=geometry,coupling=coupling,initial_coupling=(1.2,3.,5.))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(641)
            native=AnchorKernelModel(**args)
            compact=m.AnchorCompactActualBankModelV1(**args,readout_chunk_size=2)
        if stress:
            with torch.no_grad():
                for layer in native.layers:
                    layer.raw_coupling.copy_(torch.linspace(-.7,.6,9).reshape(3,3))
                    layer.raw_damping.copy_(torch.linspace(.4,-.3,9).reshape(3,3))
                    layer.mixture_logits.copy_(torch.linspace(-.8,.7,9).reshape(3,3))
        compact.load_state_dict(native.state_dict())
    view=m.derive_compact_actual_bank_v1(prepared,geometry,coupling).to('cpu',dtype)
    return native.to(dtype=dtype),compact.to(dtype=dtype),prepared.to('cpu',dtype),view


def dense_oracle(layout,operators,sources,rates,damping,coupling):
    """Direct weighted impulse sum using matrix_exp/solve, no spectral scan."""
    S,Q,H,V=sources.shape[1:]
    identity=torch.eye(Q,dtype=sources.dtype)
    full=operators.cqq[:, :, None]*rates[None,:, :,None,None]+identity*damping[None,:,:,None,None]
    if coupling=='full':
        K=full
    elif coupling=='scalar':
        K=identity*(rates+damping)[None,:,:,None,None]
        K=K.expand(operators.cqq.shape[0],-1,-1,-1,-1)
    else:
        diagonal=operators.cqq.diagonal(dim1=-2,dim2=-1)[...,None]*rates[None,:,None,:]+damping[None,:,None,:]
        K=torch.diag_embed(diagonal.permute(0,1,3,2))
    coordinates=layout.coordinates.tolist()
    evaluation=[]
    for output in range(layout.num_evaluations):
        event=layout.output_at_event.index(output)
        evaluation.append(coordinates[event])
    result=[]
    for t in evaluation:
        value=torch.zeros(S,H,Q,V,dtype=sources.dtype)
        for a,x in enumerate(layout.anchors.tolist()):
            signal=sources[a].permute(0,2,1,3)*layout.weights[a]
            if coupling=='gain_matched_diagonal':
                g=int(layout.anchor_graph[a])
                signal=K[g]@torch.linalg.solve(full[g],signal)
            if t>x:
                segments=[j for j in range(len(coordinates)-1) if coordinates[j]>=x and coordinates[j+1]<=t]
            else:
                segments=[j for j in range(len(coordinates)-1) if coordinates[j]>=t and coordinates[j+1]<=x][::-1]
            for j in segments:
                g=int(layout.segment_graph[j])
                signal=torch.matrix_exp(-layout.segment_duration[j]*K[g])@signal
            value=value+signal
        result.append(value.permute(0,2,1,3))
    return torch.stack(result)


@pytest.mark.parametrize('dtype',[torch.float64,torch.float32])
@pytest.mark.parametrize('coupling',m.COUPLINGS)
def test_direct_dense_noncommuting_oracle_and_gradients(dtype,coupling):
    prepared,_=handmade()
    native,compact,bank,view=pair(prepared,dtype,coupling,stress=True)
    ops=bank.operators['age']
    assert torch.linalg.matrix_norm(ops.cqq[0,0]@ops.cqq[1,0]-ops.cqq[1,0]@ops.cqq[0,0])>.01
    signal=torch.sin(torch.arange(5*3*7*3*2,dtype=dtype).reshape(5,3,7,3,2)/11)
    def run(which):
        sources=signal.clone().requires_grad_(True)
        raw=native.layers[0].raw_coupling.detach().clone().requires_grad_(True)
        damp=native.layers[0].raw_damping.detach().clone().requires_grad_(True)
        rates=native.layers[0].initial_coupling*raw.exp()
        damping=native.layers[0].initial_damping*damp.exp()
        if which=='oracle':
            result=dense_oracle(bank.layout,ops,sources,rates,damping,coupling)
        else:
            result=anchor_spectral_compact_action_v1(view.layout,view.geometry,sources,rates,damping,
                coupling=coupling,readout_chunk_size=2)
        weight=torch.cos(torch.arange(result.numel(),dtype=dtype).reshape_as(result)/17)
        grads=torch.autograd.grad((result*weight).mean(),(sources,raw,damp))
        return result,grads
    expected,eg=run('oracle')
    actual,ag=run('compact')
    checks={'output':m.comparison_v1(actual,expected)}
    for name,a,b in zip(('source_VJP','raw_coupling_VJP','raw_damping_VJP'),ag,eg):
        checks[name]=m.comparison_v1(a,b,gradient=True)
    assert all(c['passed'] for c in checks.values()),checks
    model_step=m.compare_model_step_v1(native,compact,bank,view)
    action=m.compare_action_VJPs_v1(native,bank,view)
    assert model_step['status']=='PASS',model_step
    assert action['status']=='PASS',action
    METRICS.append(dict(kind='direct_dense_oracle_small_full_model',dtype=str(dtype),coupling=coupling,
        q=7,r=4,checks=checks,model_step=model_step,action=action))


@pytest.mark.parametrize('dtype',[torch.float64,torch.float32])
@pytest.mark.parametrize('geometry,stress',[('plain',False),('age',False),('age',True)])
def test_candidate_sized_handmade_Q184_R96_full_Q_and_clipped_step(dtype,geometry,stress):
    prepared,roles=handmade(184,96,47,32)
    before_rng=torch.random.get_rng_state().clone()
    native,compact,bank,view=pair(prepared,dtype,geometry=geometry,stress=stress,candidate=True)
    assert torch.equal(before_rng,torch.random.get_rng_state())
    assert native.state_dict().keys()==compact.state_dict().keys()
    original={k:v.clone() for k,v in native.state_dict().items()}
    step=m.compare_model_step_v1(native,compact,bank,view)
    assert step['status']=='PASS',step
    action=m.compare_action_VJPs_v1(native,bank,view)
    assert action['status']=='PASS',action
    assert step['checks']['logits_all_Q']['elements']==8*184*4
    with torch.no_grad():
        logits=compact(view)
        assert logits.shape==(8,184,4)
        changed_features=view.leaf_x.clone()
        changed_features[:,view.reference_indices[-1],0]+=2
        donor_logits=compact(replace(view,leaf_x=changed_features))
        effect=(donor_logits[:,roles['output']]-logits[:,roles['output']]).abs().max()
        assert effect>0
    for name,value in native.state_dict().items():
        assert torch.equal(value,original[name])
    METRICS.append(dict(kind='candidate_sized_handmade',dtype=str(dtype),geometry=geometry,stress=stress,
        q=184,r=96,actual_q120_map=roles['output'].tolist(),donor_effect=float(effect),model_step=step,action=action))


@pytest.mark.parametrize('coupling',m.COUPLINGS[1:])
def test_reproduce_false_PASS_when_direct_control_view_uses_full_native_model(coupling,monkeypatch):
    prepared,_=handmade()
    native,_,bank,_=pair(prepared,torch.float64)
    view=m.derive_compact_actual_bank_v1(prepared,'age',coupling)
    seen=[]
    original_native=m.anchor_kernel_action
    original_compact=m.anchor_spectral_compact_action_v1
    def trace_native(*a,**kw):
        seen.append(('native',kw['coupling']))
        return original_native(*a,**kw)
    def trace_compact(*a,**kw):
        seen.append(('compact',kw['coupling']))
        return original_compact(*a,**kw)
    monkeypatch.setattr(m,'anchor_kernel_action',trace_native)
    monkeypatch.setattr(m,'anchor_spectral_compact_action_v1',trace_compact)
    result=m.compare_action_VJPs_v1(native,bank,view)
    assert native.coupling=='full' and view.coupling==coupling
    assert result['status']=='PASS'
    assert seen==[('native','full'),('compact','full')]
    native.coupling=coupling
    seen.clear()
    explicit=m.compare_action_VJPs_v1(native,bank,view)
    assert explicit['status']=='PASS'
    assert seen==[('native',coupling),('compact',coupling)]
    METRICS.append(dict(kind='blocking_false_control_PASS',requested_coupling=coupling,
        returned_status=result['status'],wrong_executed_couplings=['full','full'],
        explicitly_set_control_status=explicit['status'],required_behavior='reject_mismatched_model_view_contract'))


def test_reproduce_direct_geometry_label_mismatch_PASS():
    prepared,_=handmade()
    native,_,bank,view=pair(prepared,torch.float64)
    mislabeled=replace(view,geometry_name='plain')
    result=m.compare_action_VJPs_v1(native,bank,mislabeled)
    assert result['status']=='PASS'
    METRICS.append(dict(kind='same_missing_direct_contract_guard',model_geometry='age',
        view_geometry_name='plain',returned_status=result['status']))


def test_minimal_storage_and_coordinate_semantics():
    prepared,_=handmade()
    full=m.derive_compact_actual_bank_v1(prepared,'age')
    for name,source in [('leaf_x',prepared.bank.leaf_x),('normalized_response',prepared.bank.query_cache.normalized_response),
        ('query_indices',prepared.bank.query_cache.query_indices),('reference_indices',prepared.bank.query_cache.reference_indices),
        ('target_graph',prepared.bank.target_graph)]:
        assert getattr(full,name).data_ptr()==source.data_ptr()
    assert full.geometry.diagonal is None
    diagonal=m.derive_compact_actual_bank_v1(prepared,'age','diagonal')
    assert diagonal.geometry.diagonal.untyped_storage().data_ptr()==prepared.operators['age'].cqq.untyped_storage().data_ptr()
    converted=full.to('cpu',torch.float32)
    assert converted.geometry.eigenvectors.dtype==torch.float32
    assert not hasattr(converted,'operators') and not hasattr(converted,'cqq')
    for name in ('anchors','coordinates','weights','segment_duration','evaluations','evaluation_left','evaluation_right'):
        assert getattr(converted.layout,name).dtype==torch.float64
    assert torch.equal(converted.layout.target_output,full.layout.target_output)
    assert converted.layout.target_output[0]==converted.layout.target_output[-1]
    METRICS.append(dict(kind='memory_scope',CPU_alias_preserved=True,diagonal_aliases_full_Cqq_storage=True,
        CPU_original_still_retained=True,neural_dtype='float32',coordinate_dtype='float64',device_memory_unmeasured=True))


def synthetic_load(tmp_path,monkeypatch,change=None):
    from arggnn.biological_candidate import fingerprint_sha
    prepared,roles=handmade(184,96,47,32)
    bank=prepared.bank
    if change=='gather':
        bad=bank.query_cache.query_indices.clone();bad[0]=bad[1]
        bank=replace(bank,query_cache=replace(bank.query_cache,query_indices=bad))
    elif change=='flag':
        bad=bank.leaf_x.clone();bad[:,bank.query_cache.query_indices[-1],4]=2
        bank=replace(bank,leaf_x=bad)
    elif change=='features':
        bank=replace(bank,feature_columns=tuple(range(47)))
    elif change=='positions':
        bank=replace(bank,positions=bank.positions+.00001)
    elif change=='graph_count':
        bank=replace(bank,leaf_x=bank.leaf_x[:2])
    prepared=replace(prepared,bank=bank)
    saved=dict(schema='biological_candidate_full_age_bank_v1',fixed_prepared=prepared)
    path=tmp_path/'handmade_bytes_only.pt';path.write_bytes(b'handmade sentinel, never actual data')
    rec=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    selected=dict(parent_id=m.TRAIN_PARENTS[0],tile_id=11,graphs=3,readouts=8,bank=rec,
        full_bank_field_fingerprint_sha256=fingerprint_sha(saved),selection_reasons=['minimum_tree_count'])
    tile=dict(selected,position_indexes=list(range(8)),normalized_interval=[.125,.875])
    parent=dict(parent_id=m.TRAIN_PARENTS[0],split='train',tiles=[tile],
        relate_entry={'sample_index_to_inferred_tree_node_id':[dict(sample_index=i,inferred_tree_node_id=int(v)) for i,v in enumerate(roles['nodes'])]},
        dynamics_sample_indexes=roles['dynamics'].tolist(),reference_sample_indexes=roles['references'].tolist(),
        actual_query_sample_indexes=roles['actual'].tolist(),actual_query_output_indexes=roles['output'].tolist(),
        dynamics_query_node_ids=roles['nodes'][roles['dynamics']].tolist(),query_node_ids=roles['nodes'][roles['actual']].tolist(),
        positions_bp=(prepared.bank.positions*2000000 if change!='positions' else (prepared.bank.positions-.00001)*2000000).tolist())
    if change=='output_map':
        parent['actual_query_output_indexes'][0]=parent['actual_query_output_indexes'][1]
    if change=='role_overlap':
        parent['reference_sample_indexes'][0]=parent['dynamics_sample_indexes'][0]
    if change=='identity':
        parent['relate_entry']['sample_index_to_inferred_tree_node_id'][0]['inferred_tree_node_id']=1000
    admission=m.ActualQualificationAdmissionV1({},dict(parents=[parent]),(selected,),tmp_path,{})
    monkeypatch.setattr(m,'recheck_admission_v1',lambda a: None)
    def handmade_load(filename,**kw):
        assert Path(filename)==path and kw==dict(map_location='cpu',weights_only=False)
        if change=='midload_hash_drift':path.write_bytes(b'changed during handmade mocked load')
        return saved
    monkeypatch.setattr(torch,'load',handmade_load)
    return admission


def test_handmade_loader_full_identity_feature_role_and_bank_joins(tmp_path,monkeypatch):
    admission=synthetic_load(tmp_path,monkeypatch)
    prepared,provenance=m.load_admitted_train_bank_v1(admission,0)
    assert prepared.bank.query_cache.query_ids.numel()==184
    assert prepared.bank.query_cache.reference_ids.numel()==96
    assert len(provenance['actual_query_output_indexes'])==120
    assert provenance['full_file_and_full_field_hash_checked'] and not provenance['labels_opened']
    METRICS.append(dict(kind='handmade_mocked_loader_positive',Q184_R96_Q120=True,
        actual_arrays=False,real_deserialization=False,synthetic_guarded_joins=True))


@pytest.mark.parametrize('change',['gather','flag','features','positions','graph_count','output_map','role_overlap','identity','midload_hash_drift'])
def test_handmade_loader_rejects_changed_joins(tmp_path,monkeypatch,change):
    admission=synthetic_load(tmp_path,monkeypatch,change)
    with pytest.raises((ValueError,IndexError)):
        m.load_admitted_train_bank_v1(admission,0)
    METRICS.append(dict(kind='handmade_mocked_loader_negative',change=change,rejected=True))


def test_cuda_profiling_entry_rejects_unqualified_parity_before_device_API(monkeypatch):
    calls=[]
    monkeypatch.setattr(m,'recheck_admission_v1',lambda a:None)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:calls.append('device_count'))
    admission=SimpleNamespace(binding={'exact':'handmade'})
    with pytest.raises(ValueError,match='profiling'):
        m.cuda_profile_pair_v1({},admission,{},{},{},stage='full_step')
    with pytest.raises(ValueError,match='profiling'):
        m.cuda_isolated_memory_v1({},admission,{},{},{})
    assert calls==[]


def mock_allocator(monkeypatch):
    import arggnn.native_argformer_exec as runtime
    trace=[]
    lease={'synthetic':True}
    monkeypatch.setattr(m,'_profile_cuda_guard_v1',lambda *a:None)
    monkeypatch.setattr(runtime,'validate_lease',lambda:lease)
    for name in ('synchronize','empty_cache','reset_peak_memory_stats'):
        monkeypatch.setattr(torch.cuda,name,lambda name=name:trace.append(name))
    for name,value in [('memory_allocated',0),('memory_reserved',0),('max_memory_allocated',1234),('max_memory_reserved',2048)]:
        monkeypatch.setattr(torch.cuda,name,lambda value=value:value)
    class FakeEvent:
        def __init__(self,**kw):assert kw==dict(enable_timing=True)
        def record(self):trace.append('event_record')
        def elapsed_time(self,other):return 0.
    monkeypatch.setattr(torch.cuda,'Event',FakeEvent)
    return trace,lease


def test_mocked_profile_ABBA_fresh_factory_and_synchronization(monkeypatch):
    trace,lease=mock_allocator(monkeypatch)
    setups=[]
    def factory(name):
        setups.append(name);trace.append('setup_'+name)
        def execute():trace.append('execute_'+name);return object()
        return execute
    result=m.cuda_profile_pair_v1({name:lambda name=name:factory(name) for name in ('native','compact')},
        None,{},lease,{},stage='full_step')
    assert len(setups)==26 and len(result['samples'])==20
    assert [x['implementation'] for x in result['samples']]==list(('native','compact','compact','native')*5)
    assert all(s['peak_allocated_bytes']==1234 and s['baseline_allocated_bytes']==0 for s in result['samples'])
    for j,event in enumerate(trace):
        if event=='event_record' and j>0 and trace[j-1].startswith('execute_'):
            assert trace[j+1]=='synchronize'
    assert result['excludes_load_hash_transfer'] and not result['measured_speedup_claim']
    METRICS.append(dict(kind='CPU_mock_profile_contract',actual_CUDA_API=False,fresh_factories=26,
        measured_samples=20,ABBA=True,results_not_timings=True,stage_workspace_not_pipeline_footprint=True))


def test_mocked_isolated_probe_live_baseline_guard_and_full_factory_contract(monkeypatch):
    trace,lease=mock_allocator(monkeypatch)
    factories={name:lambda name=name:trace.append('CPU_handmade_factory_'+name) for name in ('native','compact')}
    result=m.cuda_isolated_memory_v1(factories,None,{},lease,{})
    assert len(result['samples'])==6 and trace.count('empty_cache')==6
    assert result['factory_scope_is_caller_contract']
    monkeypatch.setattr(torch.cuda,'memory_allocated',lambda:1048577)
    with pytest.raises(ValueError,match='1MiB'):
        m.cuda_isolated_memory_v1(factories,None,{},lease,{})
    METRICS.append(dict(kind='CPU_mock_isolated_allocator',actual_CUDA_API=False,probe_count=6,
        baseline_over_1MiB_rejected=True,CPU_only_factory_inputs_and_full_step_unenforced_caller_contract=True))
