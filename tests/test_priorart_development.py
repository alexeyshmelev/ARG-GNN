"""Synthetic CPU adaptation, immutable source admission and selection evidence."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from arggnn import priorart_development_bindings as bindings
from arggnn import train_priorart_development as runner
from arggnn.train_argmix_development import TreeTrainingParent,observed_fixed_tiles
from arggnn.reference_encoder import ReferenceFeatureAdapter
from test_argmix_development import completed_files,save
from test_anchor_temporal_baselines import seq
from test_development_selection import study,synth as synth_selection,write_json as write_selection_json,write_tensor

torch.set_num_threads(1)


@pytest.fixture(autouse=True)
def deterministic_cpu_policy():
    runner.configure_deterministic_development()


@pytest.fixture
def prior_completed(completed_files):
    f=completed_files;real_root=Path(__file__).resolve().parents[1]
    amendment=json.loads((real_root/bindings.AMENDMENT).read_bytes())
    save(f.root/bindings.AMENDMENT,amendment)
    plan=bindings.make_search_plan();planpath=f.root/bindings.PLAN;plansha=save(planpath,plan)
    return SimpleNamespace(base=f,plan=plan,planpath=planpath,plansha=plansha)


def tiny_settings(family,epochs=2,patience=8):
    return {'family':family,'num_layers':2,'reference_adapter':bindings.shared.ADAPTER,
        'epochs':epochs,'patience':patience,'weight_decay':.0001,'gradient_clip_norm':1.,
        'deterministic_policy':deepcopy(runner.DETERMINISTIC_POLICY)}


def tiny_configuration(family):return {'id':family+'_tiny','hidden_dim':8,'dropout':0.,'learning_rate':.001}


def tiny_parent():
    value=seq();value=replace(value,x=value.x.float(),clades=value.clades.float())
    labels=torch.tensor([[i%4,(i+1)%4] for i in range(len(value.positions))])
    return TreeTrainingParent(value,observed_fixed_tiles(value,2),labels,'synthetic-source:val:0',{})


@pytest.mark.parametrize('family',bindings.FAMILIES)
def test_existing_factory_configuration_gradients_and_full_observed_feature_scope(family):
    torch.manual_seed(133)
    value=seq();value.x.requires_grad_()
    net=runner.factory(tiny_configuration(family),tiny_settings(family)).double().eval()
    output=net(value);assert output.shape==(7,2,4) and torch.isfinite(output).all()
    output.square().mean().backward()
    for name,parameter in net.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(),name
    assert torch.isfinite(value.x.grad).all() and value.x.grad[value.node_ids<0].abs().max()>0
    assert value.x.grad[:,8:].abs().max()>0
    assert net.head[-1].weight.grad.abs().max()>0 if family=='gst_ca_span' else net.head.weight.grad.abs().max()>0
    audit=runner.model_configuration(net,family)
    assert audit['input_dim']==48 and audit['num_layers']==2
    assert audit['external_pretraining'] is False and audit['truth_features']==[]
    assert audit['parameters']==sum(p.numel() for p in net.parameters())
    if family=='gst_ca_span':assert audit['position_encodings'] is False and audit['token_weighting']=='span'
    else:assert audit['state_dim']==16 and audit['directions_share_parameters'] and audit['discretization']=='zoh'


@pytest.mark.parametrize('family',bindings.FAMILIES)
def test_intervening_hold_without_scored_target_and_reference_internal_counterfactual(family):
    torch.manual_seed(12)
    original=seq();original=replace(original,positions=torch.tensor([.04,.96],dtype=torch.float64),target_graph=torch.tensor([0,2]))
    net=runner.factory(tiny_configuration(family),tiny_settings(family)).double().eval();expected=net(original)
    original.x.requires_grad_();net(original)[0].square().sum().backward()
    assert original.x.grad[original.node_graph==1].abs().max()>1e-12
    internal=(original.node_graph==1)&(original.node_ids<0)
    changed=original.x.detach().clone();changed[internal,8:]+=.5
    assert (net(replace(original,x=changed))-expected).abs().max()>1e-9
    # Reference role changes must refresh the global observed F40 summaries.
    # Two local GraphSAGE blocks do not directly reach arbitrary distant refs.
    raw=original.x.detach()[:,:8].clone();refs=raw[:,:4].sum(-1)>0
    raw[refs,:4]=raw[refs,:4].roll(1,-1)
    refreshed=ReferenceFeatureAdapter(8,**bindings.shared.ADAPTER).precompute(replace(original,x=raw))
    assert (net(refreshed)-expected).abs().max()>1e-9


@pytest.mark.parametrize('family',bindings.FAMILIES)
def test_readout_query_identity_serialization_and_annotations(family):
    torch.manual_seed(19);value=seq();net=runner.factory(tiny_configuration(family),tiny_settings(family)).double().eval();expected=net(value)
    points=torch.cat((value.positions,torch.tensor([.12,.41,.83,.22],dtype=torch.float64)))
    target=((points[:,None]>=value.intervals[None,:,0])&(points[:,None]<value.intervals[None,:,1])).long().argmax(1)
    expanded=replace(value,positions=points,target_graph=target)
    torch.testing.assert_close(net(expanded)[:7],expected,rtol=3e-11,atol=3e-11)
    torch.testing.assert_close(net(replace(value,query_node_ids=value.query_node_ids.flip(0))),expected.flip(1),rtol=3e-11,atol=3e-11)
    torch.testing.assert_close(net(seq(serialize=True)),expected,rtol=3e-11,atol=3e-11)
    torch.testing.assert_close(net(replace(value,clades=torch.zeros_like(value.clades))),expected,rtol=0,atol=0)
    class Guard:
        def __getattr__(self,name):
            if name in ('labels','population','metadata','truth','baseline_probabilities','clades'):raise AssertionError('Annotation access')
            return getattr(value,name)
    torch.testing.assert_close(net(Guard()),expected,rtol=0,atol=0)
    stream=io.BytesIO();torch.save(net.state_dict(),stream);stream.seek(0)
    restored=runner.factory(tiny_configuration(family),tiny_settings(family)).double().eval()
    restored.load_state_dict(torch.load(stream,weights_only=True))
    torch.testing.assert_close(restored(value),expected,rtol=0,atol=0)


@pytest.mark.parametrize('family',bindings.FAMILIES)
def test_tiny_cpu_fitting_selects_real_hash_bound_checkpoint_and_raw_parent_logits(tmp_path,family):
    parent=tiny_parent();settings=tiny_settings(family);configuration=tiny_configuration(family)
    binding={'protocol_sha256':'1'*64,'completion_receipt_sha256':'2'*64,
        'boundary_manifest_sha256':'3'*64,'implementation_sha256':{'actual-synthetic-producer.py':'4'*64}}
    checks=[];checkpoint=tmp_path/(family+'.pt')
    record,logits=runner.train_configuration(configuration,settings,{'train':[parent],'val':[parent]},
        torch.device('cpu'),101,checkpoint,binding,integrity_check=lambda **kw:checks.append(kw['full_sources']))
    saved=torch.load(checkpoint,weights_only=True)
    assert len(record['history'])==2 and saved['seed']==101
    assert record['checkpoint_sha256']==bindings.shared.sha(checkpoint)
    assert saved['implementation_sha256']==binding['implementation_sha256']
    assert record['best_epoch']==min(range(2),key=lambda i:(record['history'][i]['val_nll'],i))
    assert logits[0].dtype==torch.float64 and logits[0].shape==(7,2,4)
    assert checks[0] and checks[-1]
    assert record['checkpoint_replay_nll_delta']==0.
    assert record['historical_validation_nll']==record['reloaded_validation_nll']
    assert record['deterministic_policy']==runner.DETERMINISTIC_POLICY
    assert record['per_parent'][0]['nll_policy']=='float64 stable log-softmax of uncalibrated logits'


@pytest.mark.parametrize('family',bindings.FAMILIES)
def test_external_labels_change_only_metrics_not_source_predictions(family):
    parent=tiny_parent();torch.manual_seed(41)
    net=runner.factory(tiny_configuration(family),tiny_settings(family)).eval()
    a,first=runner.evaluate(net,[parent],torch.device('cpu'))
    altered=replace(parent,labels=(parent.labels+1)%4)
    b,second=runner.evaluate(net,[altered],torch.device('cpu'))
    torch.testing.assert_close(first[0],second[0],rtol=0,atol=0)
    assert a[0]['nll']!=b[0]['nll']


@pytest.mark.parametrize('fault',['test','pilot','seed','epochs','patience','weight_decay','depth','width','dropout','lr','grid','id','code','scope','policy','amendment'])
def test_exact_plan_scope_and_budget_guards(fault):
    plan=bindings.make_search_plan();family='gst_ca_span'
    if fault=='test':plan['splits']['test']=1
    elif fault=='pilot':plan['splits']['train']=16
    elif fault=='seed':plan['training_seeds'].append(404)
    elif fault in ('epochs','patience','depth'):plan['num_layers' if fault=='depth' else fault]=1
    elif fault=='weight_decay':plan[fault]=0.
    elif fault=='width':plan['families'][family]['configurations'][0]['hidden_dim']=32.
    elif fault=='dropout':plan['families'][family]['configurations'][0]['dropout']=False
    elif fault=='lr':plan['families'][family]['configurations'][0]['learning_rate']=.002
    elif fault=='grid':plan['families'][family]['configurations'].pop()
    elif fault=='id':plan['families'][family]['configurations'][0]['id']='../escape'
    elif fault=='code':plan['reviewed_implementation_sha256'].pop('src/arggnn/gst_baseline.py')
    elif fault=='scope':plan['families'][family]['input_scope']='target graph only'
    elif fault=='policy':plan['nll_policy']='clipped calibrated NLL'
    else:plan['scope_amendment']['path']='data/final_test/scope.json'
    with pytest.raises((ValueError,RuntimeError)):bindings.check_plan(plan)


def test_full_completed_source_binding_emits_both_immutable_configs_and_startup_parents(prior_completed):
    f=prior_completed;output=f.base.root/'configs/priorart_bound'
    settings,contents,digests=bindings.bind_completed_development(f.base.completion_path,f.base.completion_sha,
        f.planpath,f.plansha,output_dir=output)
    assert set(settings)==set(bindings.FAMILIES)
    for family in bindings.FAMILIES:
        path=output/(family+'.json')
        assert path.read_bytes()==contents[family] and bindings.shared.sha(path)==digests[family]
        value,parents,snapshot=bindings.capture_training_snapshot(path,digests[family])
        assert value==settings[family] and len(parents['train'])==64 and len(parents['val'])==16
        assert all(len(parent.tiles)==8 for group in parents.values() for parent in group)
        assert parents['val'][0].parent_id.startswith(value['source_dataset_manifest_sha256']+':val:')
        assert snapshot.source.implementation_sha256==bindings.shared.implementation_hashes()
        assert snapshot.implementation_sha256==bindings.implementation_hashes()
        snapshot.check(full_sources=True)
    with pytest.raises(RuntimeError,match='Fresh'):bindings.bind_completed_development(f.base.completion_path,f.base.completion_sha,
        f.planpath,f.plansha,output_dir=output)


@pytest.mark.parametrize('fault',['incomplete','test','source','targets','prefix'])
def test_bad_completion_raw_source_targets_and_feature_prefix_prevent_config_output(prior_completed,fault):
    f=prior_completed;completion=deepcopy(f.base.completion);completion_sha=f.base.completion_sha
    if fault in ('incomplete','test'):
        if fault=='incomplete':completion['all_stages_complete']=False
        else:completion['test_parents']=1
        completion_sha=save(f.base.completion_path,completion)
    elif fault=='source':Path(f.base.records[0]['tensor_file']).write_bytes(b'corrupt')
    else:
        path=Path(f.base.records[0]['prepared_tensor_file']);payload=torch.load(path,weights_only=True)
        if fault=='prefix':payload['sequence']['x'][0,0]=1.
        else:
            path=Path(f.base.records[0]['metadata']);payload=json.loads(path.read_bytes());payload['paired_alignment']['query_order_equal']=False
            new_sha=save(path,payload);f.base.manifest['replicates'][0]['metadata_sha256']=new_sha
        if fault=='prefix':
            torch.save(payload,path);f.base.manifest['replicates'][0]['prepared_tensor_sha256']=bindings.shared.sha(path)
        completion['stage_manifests']['inferred']['sha256']=save(f.base.manifest_path,f.base.manifest)
        completion_sha=save(f.base.completion_path,completion)
    output=f.base.root/'must_not_bind'
    with pytest.raises((ValueError,RuntimeError)):bindings.bind_completed_development(f.base.completion_path,completion_sha,
        f.planpath,f.plansha,output_dir=output)
    assert not output.exists()


def test_composed_snapshot_captures_config_bytes_once_and_rejects_control_code_source_drift(tmp_path):
    path=tmp_path/'control.json';path.write_text('{"counter":1}');digest=bindings.shared.sha(path)
    snapshot=bindings.PriorArtSnapshot();old=Path.read_bytes;counts={}
    def counted(p):counts[str(p)]=counts.get(str(p),0)+1;return old(p)
    with patch.object(Path,'read_bytes',counted):
        first=snapshot.control(path,digest);second=snapshot.control(path,digest)
    assert first==second and counts[str(path)]==1
    path.write_text('{"counter":2}')
    with pytest.raises(RuntimeError,match='drift'):snapshot.check()
    clean=bindings.PriorArtSnapshot()
    changed=dict(clean.implementation_sha256);changed['src/arggnn/gst_baseline.py']='a'*64
    with patch.object(bindings,'implementation_hashes',return_value=changed),pytest.raises(RuntimeError,match='drift'):clean.check()


def test_code_guard_and_heldout_paths_reject_before_any_control_or_source_read(tmp_path):
    changed=dict(bindings.implementation_hashes());changed['src/arggnn/gst_baseline.py']='a'*64
    with patch.object(bindings,'implementation_hashes',return_value=changed),patch.object(Path,'read_bytes',side_effect=AssertionError('No input read allowed')):
        with pytest.raises(RuntimeError,match='drift'):bindings.capture_training_snapshot(tmp_path/'config.json','1'*64)
    for path in ('../escape','data/final_test/config.json','data/test/config.json','data/confirmatory_inputs/config.json'):
        with patch.object(Path,'read_bytes',side_effect=AssertionError('No heldout read allowed')):
            with pytest.raises(ValueError):bindings.PriorArtSnapshot().control(path,'1'*64)


def test_main_full_source_admission_precedes_first_device_query(tmp_path):
    args=['--config',str(tmp_path/'settings.json'),'--expected-config-sha256','1'*64,'--output',str(tmp_path/'run')]
    calls=[]
    def fail(*a,**kw):calls.append('admission');raise RuntimeError('Synthetic admission failure')
    with patch.object(runner,'capture_training_snapshot',side_effect=fail),patch.object(runner,'require_device',side_effect=AssertionError('Device before input admission')),patch.object(runner,'configure_deterministic_development',side_effect=AssertionError('Policy before input admission')):
        with pytest.raises(RuntimeError,match='admission failure'):runner.main(args)
    assert calls==['admission'] and not (tmp_path/'run').exists()


def test_fitting_failure_preserves_explicit_family_seed_config_source_record(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner,'development_path',lambda value:Path(value) if Path(value).is_absolute() else tmp_path/value)
    class Snapshot:
        implementation_sha256={'real-source.py':'5'*64};sources={'source.pt':'6'*64}
        def check(self,**kw):pass
        def bytes_for(self,path):return b'{}'
    configuration=tiny_configuration('gst_ca_span')
    settings={**tiny_settings('gst_ca_span'),'stage':'synthetic','configurations':[configuration],
        'completion_receipt_sha256':'2'*64,'boundary_dataset_manifest_sha256':'3'*64,
        'input_scope':bindings.SCOPE['gst_ca_span'],'cross_tree_aggregation':bindings.AGGREGATION['gst_ca_span'],
        'scope_amendment':{'path':'synthetic','sha256':'4'*64},'interpretation':'synthetic only'}
    output=tmp_path/'failed_run'
    with patch.object(runner,'require_storage'),patch.object(runner,'train_configuration',side_effect=RuntimeError('Synthetic numerical failure')):
        with pytest.raises(RuntimeError,match='numerical failure'):runner.execute_run(settings,{},Snapshot(),tmp_path/'config.json','1'*64,
            output,101,[configuration['id']],torch.device('cpu'))
    record=json.loads((output/'failure.json').read_bytes())
    assert record['family']=='gst_ca_span' and record['configuration_id']==configuration['id']
    assert record['training_seed']==101 and record['test_access'] is False and record['protocol_sha256']=='1'*64
    assert not (output/'results.json').exists()


def test_duplicate_tile_or_bad_original_query_scatter_is_rejected():
    parent=tiny_parent();net=runner.factory(tiny_configuration('graphssm_zoh'),tiny_settings('graphssm_zoh'))
    parent.tiles.append(parent.tiles[0])
    with pytest.raises(RuntimeError,match='correspondence'):runner.evaluate(net,[parent],torch.device('cpu'))


def test_one_global_generic_selector_matrix_accepts_required_priorart_list_protocols(study):
    root=study['root'];source_refs=[ref for ref in study['refs'] if ref['family_id']=='gru']
    for family in bindings.FAMILIES:
        protocol=deepcopy(study['configs']['gru']);protocol.update(family=family,input_scope=bindings.SCOPE[family],
            cross_tree_aggregation=bindings.AGGREGATION[family],configurations=bindings.grid(family))
        for key in ('models','model_specs','model_learning_rates'):protocol.pop(key)
        protocol_ref=write_selection_json(root,'configs/'+family+'.json',protocol)
        study['plan']['families'][family]={'role':family,'grid_kind':'width_dropout_lr','protocol':protocol_ref,
            'expected_implementation_sha256':protocol['bound_implementation_sha256'],
            'input_scope':protocol['input_scope'],'cross_tree_aggregation':protocol['cross_tree_aggregation']}
        study['plan']['required_baseline_families'].append(family)
        for original_ref in source_refs:
            run=json.loads((root/original_ref['path']).read_text());run['family']=family;run['protocol_sha256']=protocol_ref['sha256'];models={}
            for configuration,old_name in zip(protocol['configurations'],study['configs']['gru']['models']):
                name=configuration['id'];body=run['models'][old_name];body['ending_config_sha256']=protocol_ref['sha256']
                for pathkey,hashkey,suffix in (('checkpoint','checkpoint_sha256','checkpoint'),('validation_logits','validation_logits_sha256','logits')):
                    payload=torch.load(root/body[pathkey],weights_only=True);payload['protocol_sha256']=protocol_ref['sha256']
                    artifact=write_tensor(root,f'artifacts/{family}/{name}_s{run["training_seed"]}_{suffix}.pt',payload)
                    body[pathkey]=artifact['path'];body[hashkey]=artifact['sha256']
                models[name]=body
            run['models']=models
            study['refs'].append({'family_id':family,**write_selection_json(root,f'runs/{family}_s{run["training_seed"]}.json',run)})
    study['plan_ref']=write_selection_json(root,'plan.json',study['plan'])
    result=synth_selection(study).payload
    assert result['matrix_counts']=={'success':192,'failed':0,'missing':0} and result['baseline_selection_eligible']
    assert all(result['families'][family]['family_selection_eligible'] for family in bindings.FAMILIES)


def test_policy_drift_prevents_optimizer_or_device_transfer():
    settings=tiny_settings('gst_ca_span');settings['deterministic_policy']['warn_only']=True
    with patch.object(runner,'factory',side_effect=AssertionError('No model transfer before policy validation')):
        with pytest.raises(RuntimeError,match='strict deterministic'):runner.train_configuration(tiny_configuration('gst_ca_span'),
            settings,{},torch.device('cpu'),101,Path('unused'),{},integrity_check=lambda **kw:None)


def test_genuine_nonfinite_synthetic_training_fails_without_selected_checkpoint(tmp_path):
    parent=tiny_parent();first=parent.tiles[0];broken=first.sequence.x.clone();broken[0,8]=float('nan')
    parent.tiles[0]=replace(first,sequence=replace(first.sequence,x=broken))
    checkpoint=tmp_path/'must_not_select.pt'
    with pytest.raises(RuntimeError,match='Nonfinite'):
        runner.train_configuration(tiny_configuration('gst_ca_span'),tiny_settings('gst_ca_span'),
            {'train':[parent],'val':[parent]},torch.device('cpu'),101,checkpoint,{},integrity_check=lambda **kw:None)
    assert not checkpoint.exists()
