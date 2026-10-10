"""Fresh independent runner review. Handmade CPU arrays and mocked permits only."""
from copy import deepcopy
from types import SimpleNamespace
from pathlib import Path
import hashlib
import json
import math
import os
import pytest
import torch
from arggnn import anchor_compact_qualification_runner_v1 as r
from arggnn.anchor_kernel import AnchorKernelBank, AnchorOperators, _layout
from arggnn.kron_interface import LeafInterfaceBank, QueryInterfaceCache
from arggnn.development_execution import DETERMINISTIC_POLICY

OUT=Path(__file__).parent
METRICS=[]

@pytest.fixture(scope='session',autouse=True)
def no_cuda():
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    yield
    assert not torch.cuda.is_initialized()
    p=OUT/os.environ['ARGGNN_INDEPENDENT_METRICS_FILE']
    assert not p.exists()
    p.write_text(json.dumps(METRICS,indent=2,sort_keys=True,allow_nan=False)+'\n')


def selection(n=3):
    reasons=['minimum_tree_count','maximum_tree_count','minimum_readout_count']
    return [dict(parent_id='handmade_'+str(i),tile_id=12+i,graphs=2+i,readouts=7+i,
                 selection_reasons=[reasons[i]]) for i in range(n)]


def action_families():
    return {'raw_per_scale','numerator','mass','normalized_all_Q'}|{n+'.VJP' for n in
        ('query_signal','reference_signal','raw_coupling','raw_damping','mixture_logits')}


def model_families(cell):
    model,_=r.sealed.qualification_models_v1(cell['geometry'],cell['state'])
    names={'logits_all_Q','final_anchors','final_targets','input_feature_VJP','preclip_norm'}
    names|={f'layer{i}.{n}' for i in range(2) for n in
            ('anchors','targets','numerator','mass','sources','mass_sources','raw_per_scale')}
    names|={n+s for n,_ in model.named_parameters() for s in
            ('.VJP','.post_AdamW','.optimizer.step','.optimizer.exp_avg','.optimizer.exp_avg_sq')}
    return names


def valid_result(cell):
    names=action_families() if cell['probe']==r.PROBES[0] else model_families(cell)
    metrics={}
    for n in names:
        gradient=n.endswith('.VJP') or '.optimizer.' in n or n.endswith('.post_AdamW') or n in ('input_feature_VJP','preclip_norm')
        f32=cell['device_condition']=='cuda_float32' or n.endswith('.optimizer.step')
        tolerance=((.001,.00001) if gradient else (.0002,.00002)) if f32 else ((1e-8,1e-10) if gradient else (1e-10,1e-12))
        metrics[n]=dict(elements=1,max_absolute_error=0.,max_acceptance_units=0.,passed=True,rtol=tolerance[0],atol=tolerance[1])
    return dict(status='PASS',performed_request=deepcopy(cell),seeded_constructor=
        'qualification_models_v1' if cell['coupling']=='full' else 'qualification_direct_control_models_v2',
        all_Q_checked=True,labels_opened=False,checks=metrics,input_models_modified=False,
        preclip_native_norm=3.,preclip_compact_norm=3.,cell_identity=dict(
            geometry=cell['geometry'],native_geometry=cell['geometry'],compact_geometry=cell['geometry'],
            coupling=cell['coupling'],native_coupling=cell['coupling'],compact_coupling=cell['coupling'],
            dtype='torch.float64' if cell['device_condition']=='cpu_float64' else 'torch.float32',
            device='cpu' if cell['device_condition']=='cpu_float64' else 'cuda:0',
            graphs=cell['graphs'],readouts=cell['readouts'],Q=184,R=96,anchors=32,scales=3,
            latent_dim=8,kernel_depth=2,probe=cell['probe']))


@pytest.mark.parametrize('n',[1,2,3])
def test_independent_plan_cross_product(n):
    rows=selection(n)
    if n==1: rows[0]['selection_reasons']+=['maximum_tree_count','minimum_readout_count']
    observed=r.frozen_cells(rows)
    key=lambda c:(c['selection_index'],c['geometry'],c['state'],c['coupling'],c['device_condition'],c['probe'])
    expected={(i,g,'initial','full',d,p) for i in range(n) for g in ('plain','age')
              for d in ('cpu_float64','cuda_float32') for p in r.PROBES}
    expected|={(0,'age','rate_mixture_stress','full',d,p) for d in r.DEVICES for p in r.PROBES}
    expected|={(0,'age','initial',c,d,r.PROBES[0]) for c in ('scalar','diagonal','gain_matched_diagonal') for d in r.DEVICES}
    assert len(observed)==len(expected)==8*n+10
    assert set(map(key,observed))==expected and len(set(map(key,observed)))==len(observed)
    assert all(c['seed']==101 and c['parent_id']==rows[c['selection_index']]['parent_id'] and
               c['tile_id']==rows[c['selection_index']]['tile_id'] for c in observed)
    assert set(r.expected_checks(observed[0]))==action_families()
    assert set(r.expected_checks(observed[1]))==model_families(observed[1])


@pytest.mark.parametrize('which',['absent','duplicate'])
def test_minimum_selection_required(which):
    rows=selection()
    rows[0]['selection_reasons']=[]
    if which=='duplicate':
        rows[0]['selection_reasons']=['minimum_tree_count'];rows[1]['selection_reasons']=['minimum_tree_count']
    with pytest.raises(ValueError): r.frozen_cells(rows)


@pytest.mark.parametrize('field',['geometry','native_geometry','compact_geometry','coupling','native_coupling','compact_coupling','dtype','device','graphs','readouts','Q','R','anchors','scales','latent_dim','kernel_depth','probe'])
def test_mislabeled_helper_rejected(field):
    c=r.frozen_cells(selection())[0]; result=valid_result(c);result['cell_identity'][field]='corrupt'
    with pytest.raises(ValueError):r.validate_performed(c,result)


@pytest.mark.parametrize('field',['selection_index','parent_id','tile_id','graphs','readouts','geometry','state','coupling','device_condition','seed','probe'])
def test_performed_request_rejected(field):
    c=r.frozen_cells(selection())[0];result=valid_result(c);result['performed_request'][field]='corrupt'
    with pytest.raises(ValueError):r.validate_performed(c,result)


@pytest.mark.parametrize('failure',['missing_family','extra_family','wrong_constructor','wrong_tolerance','negative_error','nan_units','infinite_error','false_PASS','zero_elements','bool_elements','allQ_false','labels_true','model_modified','inactive_clipping','failed_status'])
def test_numerical_gate_rejected(failure):
    c=r.frozen_cells(selection())[1];result=valid_result(c);n='logits_all_Q';metric=result['checks'][n]
    if failure=='missing_family':del result['checks']['input_feature_VJP']
    elif failure=='extra_family':result['checks']['unexpected']=deepcopy(metric)
    elif failure=='wrong_constructor':result['seeded_constructor']='wrong'
    elif failure=='wrong_tolerance':metric['rtol']=1e-3
    elif failure=='negative_error':metric['max_absolute_error']=-1
    elif failure=='nan_units':metric['max_acceptance_units']=float('nan')
    elif failure=='infinite_error':metric['max_absolute_error']=float('inf')
    elif failure=='false_PASS':metric['max_acceptance_units']=1.001
    elif failure=='zero_elements':metric['elements']=0
    elif failure=='bool_elements':metric['elements']=True
    elif failure=='allQ_false':result['all_Q_checked']=False
    elif failure=='labels_true':result['labels_opened']=True
    elif failure=='model_modified':result['input_models_modified']=True
    elif failure=='inactive_clipping':result['preclip_compact_norm']=1.
    elif failure=='failed_status':result['status']='FAIL'
    with pytest.raises(ValueError):r.validate_performed(c,result)


def mocked_admission(n=3):
    return SimpleNamespace(adapter=SimpleNamespace(selected=selection(n),binding={'handmade':True}),
        binding={'mock_admission':True},source_sha256={'handmade':'a'*64})


@pytest.mark.parametrize('failure',['numerical','wrong_control','exception','cell_end'])
def test_failed_cell_not_reachable_to_profile_and_retained(failure):
    calls=[];profiles=[];saved={};first=True
    def perform(c):
        calls.append(deepcopy(c));value=valid_result(c)
        if len(calls)==1:
            if failure=='exception':raise RuntimeError('independent handmade failure')
            if failure=='numerical':value['checks']['mass'].update(passed=False,max_acceptance_units=2.)
            if failure=='wrong_control':value['cell_identity']['coupling']='diagonal'
        return value
    def boundary(stage):
        nonlocal first
        if failure=='cell_end' and stage=='numerical_cell_end' and first:
            first=False;raise RuntimeError('independent boundary failure')
        return {'stage':stage}
    result=r.run_qualified_v1(mocked_admission(),dependencies=r.Dependencies(boundary,perform,
        lambda *args:profiles.append(args),lambda name,value:saved.setdefault(name,deepcopy(value))))
    assert result['status']=='FAIL' and result['profile_blocked'] and not profiles
    assert len(calls)==34 and len(result['parity']['records'])==34
    assert saved['cells/cell_000.json']['validation']=='FAIL' and saved['report.json']['status']=='FAIL'
    assert saved['parity.json']['all_frozen_cases_passed'] is False
    assert calls==r.frozen_cells(selection())


@pytest.mark.parametrize('fail_at',['before_first_array','completion_source_and_storage_recheck'])
def test_boundary_failure_retains_fail_report(fail_at):
    calls=[];saved={};profiles=[]
    def boundary(stage):
        if stage==fail_at:raise ValueError('simulated bound bytes changed')
        return {'stage':stage}
    result=r.run_qualified_v1(mocked_admission(),dependencies=r.Dependencies(boundary,
        lambda c:(calls.append(c) or valid_result(c)),lambda *a:(profiles.append(a) or {}),
        lambda name,value:saved.setdefault(name,deepcopy(value))))
    assert result['status']=='FAIL' and result['profile_blocked']
    assert len(calls)==(0 if fail_at=='before_first_array' else 34)
    assert len(profiles)==(0 if fail_at=='before_first_array' else 6)
    assert saved['report.json']['error']=='simulated bound bytes changed'


def test_complete_gate_then_exact_profile_tiles():
    profiles=[];events=[];saved={}
    def profile(i,g,parity,save):
        assert len(parity['records'])==34 and all(x['validation']=='PASS' for x in parity['records'])
        assert parity['binding']=={'handmade':True};profiles.append((i,g));save('mock',{'not_measured':True});return {'not_measured':True}
    result=r.run_qualified_v1(mocked_admission(),dependencies=r.Dependencies(
        lambda stage:(events.append(stage) or {'stage':stage}),valid_result,profile,
        lambda name,value:saved.setdefault(name,deepcopy(value))))
    assert profiles==[(i,g) for i in range(3) for g in ('plain','age')]
    assert result['status']=='completed_descriptive_qualification' and events[-1]=='completion_source_and_storage_recheck'
    assert len([n for n in saved if n.startswith('cells/')])==34


def domain_fixture():
    output=OUT/'handmade_domain_output';output.mkdir(exist_ok=True)
    config={'path':r.PROTOCOL,'sha256':'a'*64};sources={r.RUNNER:'b'*64}
    lease=dict(physical_gpu=3,canonical_uuid='00000000-0000-0000-0000-000000000003',visible_gpu_uuid='GPU-00000000-0000-0000-0000-000000000003')
    common=dict(status='active_root_admission',authorizer='root_agent',project_root=str(r.ROOT),runner_config=config,issued_unix_seconds=950.)
    snapshot=[{'processes':[]} for _ in range(7)];snapshot[3]['processes']=[dict(pid=os.getpid(),kind='compute',cmdline=r.sealed.PROCESS_TITLE)]
    record={'path':str(OUT/'nonexistent_handmade_snapshot.json'),'sha256':'c'*64}
    live=dict(python='mock_python',torch='mock_torch',torch_cuda='mock_cuda',driver_version='mock_driver',python_executable=str(r.ROOT/'.runtime/env/bin/python'),python_executable_sha256='d'*64,actual_process_title=r.sealed.PROCESS_TITLE,pid=os.getpid(),actual_context_uuid=lease['canonical_uuid'],device='cuda',execution_policy=deepcopy(DETERMINISTIC_POLICY),logical_device_count=1,logical_device=0,lease=lease,nvtop_snapshot=snapshot,nvtop_snapshot_record=record)
    runtime=dict(common,kind='root_compact_runner_runtime_v1',source_sha256=sources,
                 **{k:v for k,v in live.items() if k not in ('logical_device_count','logical_device','lease','nvtop_snapshot','nvtop_snapshot_record')})
    reservations=[dict(id='mock_runner',bytes=1000000000,scope=[str(output)],status='reserved'),dict(id='mock_other',bytes=20000000000,scope=[str(OUT/'other_scope')],status='reserved')]
    storage=dict(common,kind='root_compact_runner_storage_v1',reservation_scope=[str(output)],output_cache_log_cap_bytes=1000000000,raw_tensors_or_checkpoints_retained=False,global_scan_used=False,reservation_active=True,reservation_expires_unix_seconds=2000.,accounting_scope='entire_repository_including_ignored_environment',unaccounted_writer_scope=[],warning_threshold_bytes=450000000000,hard_limit_bytes=500000000000,observed_repository_bytes=420000000000,outstanding_growth_reservations=reservations,current_reservation_inventory=deepcopy(reservations),projected_repository_peak_bytes=441000000000)
    launcher=dict(common,kind='root_compact_runner_launcher_v1',lease=lease,pid=os.getpid(),actual_context_uuid=lease['canonical_uuid'],title_before_cuda=r.sealed.PROCESS_TITLE,title_before_cuda_verified=True,root_pool_runner_allowlist_integrated=True,launch_after_prior_pools_natural_drain=True,nvtop_snapshot=record,nvtop_exit_code=0,nvtop_executable=str(r.ROOT/'.runtime/env/bin/nvtop'),project_physical_gpus=[3])
    return runtime,storage,launcher,dict(config_record=config,source_sha256=sources,output=output,live=live,now=1000.)


@pytest.mark.parametrize('field',['python','torch','torch_cuda','driver_version','python_executable','python_executable_sha256','actual_process_title','pid','actual_context_uuid','device','execution_policy','source_sha256'])
def test_wrong_runtime_domain(field):
    runtime,storage,launcher,args=domain_fixture();runtime[field]='changed'
    with pytest.raises((ValueError,KeyError)):r.validate_domains(runtime,storage,launcher,**args)


@pytest.mark.parametrize('failure',['future','stale','expiry','inventory','arithmetic','warn','missing_reservation','duplicate_id','wrong_scope','missing_accounted','wrong_lease','wrong_pid','excluded_GPU','three_GPUs','duplicate_own_PID','own_noncompute','own_wrong_title','no_NVTOP','no_allowlist','no_drain','logical_count','logical_device','false_title_before'])
def test_wrong_storage_launcher_domains(failure):
    runtime,storage,launcher,args=domain_fixture()
    if failure=='future':runtime['issued_unix_seconds']=1001
    elif failure=='stale':storage['issued_unix_seconds']=879
    elif failure=='expiry':storage['reservation_expires_unix_seconds']=1000
    elif failure=='inventory':storage['current_reservation_inventory']=[]
    elif failure=='arithmetic':storage['projected_repository_peak_bytes']+=1
    elif failure=='warn':storage['observed_repository_bytes']=429000000000;storage['projected_repository_peak_bytes']=450000000000
    elif failure=='missing_reservation':storage['outstanding_growth_reservations']=[]
    elif failure=='duplicate_id':storage['outstanding_growth_reservations'][1]['id']='mock_runner'
    elif failure=='wrong_scope':storage['reservation_scope']=[str(r.ROOT)]
    elif failure=='missing_accounted':storage['unaccounted_writer_scope']=['mock_unknown_writer']
    elif failure=='wrong_lease':launcher['lease']={}
    elif failure=='wrong_pid':launcher['pid']=os.getpid()+1
    elif failure in ('excluded_GPU','three_GPUs'):
        devices=[0,3] if failure=='excluded_GPU' else [2,3,4]
        for i in devices:
            if i!=3:args['live']['nvtop_snapshot'][i]['processes']=[dict(pid=os.getpid()+i+1,kind='compute',cmdline=r.sealed.PROCESS_TITLE)]
        launcher['project_physical_gpus']=devices
    elif failure=='duplicate_own_PID':args['live']['nvtop_snapshot'][4]['processes']=deepcopy(args['live']['nvtop_snapshot'][3]['processes'])
    elif failure=='own_noncompute':args['live']['nvtop_snapshot'][3]['processes'][0]['kind']='graphics'
    elif failure=='own_wrong_title':args['live']['nvtop_snapshot'][3]['processes'][0]['cmdline']='wrong'
    elif failure=='no_NVTOP':launcher['nvtop_exit_code']=1
    elif failure=='no_allowlist':launcher['root_pool_runner_allowlist_integrated']=False
    elif failure=='no_drain':launcher['launch_after_prior_pools_natural_drain']=False
    elif failure=='logical_count':args['live']['logical_device_count']=2
    elif failure=='logical_device':args['live']['logical_device']=1
    elif failure=='false_title_before':launcher['title_before_cuda_verified']=False
    with pytest.raises((ValueError,KeyError)):r.validate_domains(runtime,storage,launcher,**args)


def test_valid_domain_and_boundary_age_contract():
    runtime,storage,launcher,args=domain_fixture();r.validate_domains(runtime,storage,launcher,**args)
    args['now']=1300.;r.validate_domains(runtime,storage,launcher,**args,initial=False)
    with pytest.raises(ValueError):r.validate_domains(runtime,storage,launcher,**args,initial=True)


def test_real_inactive_template_stops_before_adapter_or_live(monkeypatch):
    config={'path':r.PROTOCOL,'sha256':'0c8f27a35b2b1725bc817e6b0426346ae36634c7eed0c3e3244b994df424b7b5'}
    template={'path':'research/reviews/anchor_compact_qualification_runner_v1_author/root_admission_template.json','sha256':'2e748b7e60e960752788b773d56384ef70d1d57fb577bf553490aa03ddcefec0'}
    monkeypatch.setattr(r.sealed,'admit_actual_qualification_v1',lambda *a:pytest.fail('inactive reached adapter'))
    with pytest.raises(ValueError,match='active root runner admission'):
        r.admit_runner_v1(config,template,live_observer=lambda *a:pytest.fail('inactive reached live'))


def test_wrong_source_hash_fails_without_arrays():
    with pytest.raises(ValueError):r.sealed.checked_file({'path':r.RUNNER,'sha256':'0'*64})


def test_owned_scope_and_no_overwrite(monkeypatch,tmp_path):
    output=tmp_path/'owned';output.mkdir()
    names=('TMPDIR','XDG_CACHE_HOME','XDG_CONFIG_HOME','XDG_DATA_HOME','CUDA_CACHE_PATH','TORCH_HOME','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','NUMBA_CACHE_DIR','MPLCONFIGDIR','PYTHONPYCACHEPREFIX')
    for name in names:monkeypatch.setenv(name,str(output/name))
    assert r.check_owned_storage(output)['observed_bytes']==0
    admission=SimpleNamespace(output=output);deps=r.production_dependencies(admission)
    deps.save('one.json',{'handmade':True});before=(output/'one.json').read_bytes()
    with pytest.raises(ValueError,match='overwrite'):deps.save('one.json',{'handmade':False})
    assert (output/'one.json').read_bytes()==before
    monkeypatch.setenv('TORCH_HOME',str(r.ROOT/'.runtime/cache/torch'))
    with pytest.raises(ValueError):r.check_owned_storage(output)


def test_scope_rejects_symlink_and_traversal(tmp_path):
    actual=tmp_path/'real';actual.mkdir();link=tmp_path/'link';link.symlink_to(actual)
    with pytest.raises(ValueError):r.canonical_path(link)
    with pytest.raises(ValueError):r.canonical_path(tmp_path/'..'/'escape')


def handmade():
    """New analytic block-rotation fixture, disordered Q/R joins and duplicate readouts."""
    q=184;rr=96;g=2;f=47;s=3
    intervals=torch.tensor([[.4,1.],[0.,.4]],dtype=torch.float64)
    positions=torch.tensor([0.,.4,.2,.8,.4,.125,.999],dtype=torch.float64)
    target=torch.where(positions<.4,1,0)
    mats=[];eigs=[];vectors=[]
    for graph in range(g):
        gs=[];es=[];vs=[]
        for scale in range(s):
            angle=.11+.14*graph+.03*scale;c=math.cos(angle);z=math.sin(angle)
            v=torch.eye(q,dtype=torch.float64)
            for i in range(0,q,2):v[i:i+2,i:i+2]=torch.tensor([[c,-z],[z,c]],dtype=torch.float64)
            e=.8+torch.arange(q,dtype=torch.float64)/600+scale*.12
            es.append(e);vs.append(v);gs.append(v@torch.diag(e)@v.T)
        mats.append(torch.stack(gs));eigs.append(torch.stack(es));vectors.append(torch.stack(vs))
    cqq=torch.stack(mats);eig=torch.stack(eigs);vec=torch.stack(vectors)
    cqr=-cqq.sum(-1)[...,None].expand(-1,-1,-1,rr)/rr
    sample_ids=torch.arange(q+rr)*13+2003
    query=torch.cat((torch.arange(0,q,2),torch.arange(1,q,2)))
    reference=torch.arange(q+rr-1,q-1,-1)
    joins=torch.cat((query,reference))
    canonical=torch.zeros(g,q+rr,q+rr,dtype=torch.float64)
    canonical[:,:q,:q]=cqq[:,0];canonical[:,:q,q:]=cqr[:,0]
    canonical[:,q:,:q]=cqr[:,0].transpose(-1,-2);canonical[:,q:,q:]=torch.diag_embed(-cqr[:,0].sum(-2))
    response=torch.empty_like(canonical);response[:,joins[:,None],joins[None,:]]=canonical
    widths=intervals[:,1]-intervals[:,0]
    cache=QueryInterfaceCache(response,torch.ones(g),cqq[:,0],cqr[:,0],eig[:,0],vec[:,0],query,reference,sample_ids[query],sample_ids[reference],widths/widths.sum(),cqq[:,0].mean(0),eig[0,0],vec[0,0])
    indexes=torch.arange(g*(q+rr)*f,dtype=torch.float64).reshape(g,q+rr,f)
    leaf=.7*torch.sin(indexes/23)+.2*torch.cos(indexes/41)
    leaf[:,:,4]=torch.isin(sample_ids,sample_ids[query[:120]]).double()
    bank=LeafInterfaceBank(response,sample_ids,torch.arange(g*(q+rr)).reshape(g,q+rr),leaf,
        tuple(i for i in range(48) if i!=7),intervals,sample_ids[query],positions,target,
        torch.tensor([0.,1.],dtype=torch.float64),(),.0001,10000.,query_cache=cache)
    age=AnchorOperators(cqq,cqr,eig,vec)
    plain=AnchorOperators(*(x[:,0:1].expand(-1,3,*x.shape[2:]) for x in (cqq,cqr,eig,vec)))
    return AnchorKernelBank(bank,_layout(bank,32),{'plain':plain,'age':age},3)

@pytest.fixture(scope='module')
def handbank():return handmade()


@pytest.mark.parametrize('geometry,state,coupling,probe',[
    ('plain','initial','full',r.PROBES[1]),
    ('age','rate_mixture_stress','full',r.PROBES[0]),
    ('age','initial','gain_matched_diagonal',r.PROBES[0])])
def test_default_performer_new_handmade_math(handbank,monkeypatch,geometry,state,coupling,probe):
    row=dict(parent_id='independent_handmade_train',tile_id=8,graphs=2,readouts=7)
    meta=dict(row,full_file_and_full_field_hash_checked=True,labels_opened=False,bank_modified=False)
    admission=SimpleNamespace(adapter=SimpleNamespace(selected=[row]))
    monkeypatch.setattr(r.sealed,'recheck_admission_v1',lambda *a:None)
    monkeypatch.setattr(r.sealed,'load_admitted_train_bank_v1',lambda *a:(handbank,deepcopy(meta)))
    cell=dict(row,selection_index=0,geometry=geometry,state=state,coupling=coupling,probe=probe,device_condition='cpu_float64',seed=101)
    result=r.perform_cell(admission,cell);r.validate_performed(cell,result)
    METRICS.append(dict(kind='independent_default_performer_Q184_R96',cell=cell,result=result))


@pytest.mark.parametrize('implementation',['native','compact'])
@pytest.mark.parametrize('dtype',[torch.float64,torch.float32])
def test_stage_sources_action_decoder_new_fixture(handbank,implementation,dtype):
    pair=r.sealed.qualification_models_v1('age','initial');model=pair[implementation=='compact'].to(dtype=dtype)
    bank=(handbank if implementation=='native' else r.sealed.derive_compact_actual_bank_v1(handbank,'age')).to('cpu',dtype)
    parts=r._staged_parts(model,bank,implementation);context=model.encode_context(bank)
    sources=parts['source']();actions=parts['action']();decoded,logits=parts['decoder']()
    checks=[]
    features=bank.bank.leaf_x if implementation=='native' else bank.leaf_x
    response=bank.bank.query_cache.normalized_response if implementation=='native' else bank.normalized_response
    assert torch.equal(parts['encoder'](),model.encoder(features,response))
    assert logits.shape==(7,184,4)
    for i in range(2):
        for observed,expected in zip(sources[i],(context.layers[i].sources,context.layers[i].mass_sources)):
            assert torch.equal(observed,expected);checks.append(r.sealed.comparison_v1(observed,expected))
        for n in ('anchors','targets','numerator','mass'):
            observed=getattr(actions[i],n);expected=getattr(context.layers[i],n)
            assert torch.equal(observed,expected);checks.append(r.sealed.comparison_v1(observed,expected))
    assert torch.equal(decoded[-1][0],context.anchors) and torch.equal(decoded[-1][1],context.targets)
    assert torch.equal(logits,model.head(context.targets))
    checks.append(r.sealed.comparison_v1(logits,model.head(context.targets)))
    METRICS.append(dict(kind='independent_stage_math',implementation=implementation,dtype=str(dtype),checks=checks))


def test_fullstep_against_independent_first_adamw_step(handbank):
    actual,_=r.sealed.qualification_models_v1('age','initial');actual=actual.double().eval()
    oracle=deepcopy(actual);bank=handbank.to('cpu',torch.float64)
    context=oracle.encode_context(bank);logits=oracle.head(context.targets)
    weights=torch.linspace(-.7,.9,logits.numel(),dtype=logits.dtype).reshape_as(logits)
    objective=1000*((weights*logits).mean()+.013*sum(m.mass.log().mean() for m in context.layers))
    objective.backward()
    parameters=dict(oracle.named_parameters())
    norm=torch.sqrt(sum(p.grad.square().sum() for p in parameters.values()))
    factor=min(1.,float(1/(norm+1e-6)));assert float(norm)>1
    special={id(p) for p in oracle.rate_and_mixture_parameters()}
    expected={};moments={}
    for n,p in parameters.items():
        grad=p.grad*factor;decay=0. if id(p) in special else .0001
        expected[n]=p.detach()*(1-.001*decay)-.001*grad/(grad.abs()+1e-8)
        moments[n]=(grad*.1,grad.square()*.001)
    for p in actual.parameters():p.grad=torch.ones_like(p)*99
    optimizer=r.sealed.optimizer_v1(actual);result=r.full_step(actual,bank,optimizer)
    torch.testing.assert_close(result[-1],norm,rtol=1e-12,atol=1e-12)
    checks=[]
    for n,p in actual.named_parameters():
        torch.testing.assert_close(p,expected[n],rtol=1e-10,atol=1e-12)
        state=optimizer.state[p];assert float(state['step'])==1
        torch.testing.assert_close(state['exp_avg'],moments[n][0],rtol=1e-10,atol=1e-12)
        torch.testing.assert_close(state['exp_avg_sq'],moments[n][1],rtol=1e-10,atol=1e-12)
        checks.append(r.sealed.comparison_v1(p,expected[n],gradient=True))
    METRICS.append(dict(kind='independent_explicit_AdamW',preclip_norm=float(norm),checks=checks))
