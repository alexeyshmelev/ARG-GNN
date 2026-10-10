"""Synthetic fresh-source guards, exact budgets and all-record runner joins."""
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
from arggnn.development_execution import configure_deterministic_development
from test_development_execution import restore_execution_flags


@pytest.fixture(autouse=True)
def strict_cpu_execution(restore_execution_flags):
    configure_deterministic_development()

from test_allrecord_boundary_temporal import bank,model,readouts
from arggnn import allrecord_development_bindings as binding
from arggnn import train_allrecord_development as runner
from arggnn.confirmatory_metrics import logit_metrics
from test_development_selection import study, synth as synth_selection, write_json as write_selection_json, write_tensor


def write_json(path,value):
    content=(json.dumps(value,indent=2,sort_keys=True)+'\n').encode();path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


class Lock:
    def __init__(self):self.hashes={};self.signatures={};self.checks=[]
    def check(self,full=False):self.checks.append(full)
    def manifest(self):return {'artifact_sha256':self.hashes}


@pytest.fixture
def synthetic(tmp_path):
    # Temporary project artifacts only, with a mocked verified data provider.
    plan=binding.make_search_plan();planpath=tmp_path/'plan.json';plansha=write_json(planpath,plan)
    source={stage:{'path':str(tmp_path/(stage+'_manifest.json')),'sha256':hashlib.sha256(stage.encode()).hexdigest()}
        for stage in ('oracle','inferred','boundary','age')}
    source['boundary']['preparation_sha256']=hashlib.sha256(b'boundary-prep').hexdigest()
    source['age']['preparation_sha256']=hashlib.sha256(b'age-prep').hexdigest()
    completion={'all_stages_complete':True,'train_parents':64,'validation_parents':16,'test_parents':0,
        'inferred_boundary_and_age_tiles':640,'training_or_confirmatory_selection_frozen':False,
        'stage_manifests':source,'development_execution_freeze_file_sha256':hashlib.sha256(b'freeze').hexdigest(),
        'development_execution_freeze_path':str(tmp_path/'freeze.json')}
    completionpath=tmp_path/'completion.json';completionsha=write_json(completionpath,completion)
    configs=binding.bound_configs(plan,planpath,plansha,completion,completionpath,completionsha)
    paths={family:tmp_path/(family+'.json') for family in binding.FAMILIES}
    refs={family:write_json(paths[family],settings) for family,settings in configs.items()}
    q=torch.arange(64,dtype=torch.long)
    toybank=SimpleNamespace(feature_columns=binding.FEATURE_COLUMNS,
        leaf_x=torch.zeros(1,88,47),query_node_ids=q,query_cache=SimpleNamespace(reference_ids=torch.arange(64,88)))
    groups={split:[SimpleNamespace(metadata={'split':split,'num_classes':4},query_node_ids=q,
        tiles=[SimpleNamespace(bank=toybank) for _ in range(8)]) for _ in range(count)]
        for split,count in (('train',64),('val',16))}
    dataset=SimpleNamespace(records=[None]*640,cases=lambda split:groups[split])
    return SimpleNamespace(root=tmp_path,plan=plan,planpath=planpath,plansha=plansha,
        completion=completion,completionpath=completionpath,completionsha=completionsha,
        configs=configs,paths=paths,refs=refs,dataset=dataset,lock=Lock())


def test_exact_two_eight_configuration_grids_common_pins_and_family_budget(synthetic):
    expected={(h,d,lr) for h in (32,64) for d in (0.,.1) for lr in (.0003,.001)}
    all_ids=[]
    for family,settings in synthetic.configs.items():
        binding.validate_settings(settings)
        actual={(settings['model_specs'][name]['hidden_dim'],settings['model_specs'][name]['dropout'],settings['model_learning_rates'][name]) for name in settings['models']}
        assert actual==expected and len(settings['models'])==8
        assert settings['epochs']==48 and settings['patience']==8 and settings['weight_decay']==.0001
        assert settings['training_seeds']==[101,202,303] and settings['splits']['test']==0
        assert settings['source_depth']==2 and settings['temporal_depth']==2
        assert settings['bound_implementation_sha256']==binding.common.implementation_snapshot(settings)
        all_ids+=settings['models']
        for name in settings['models']:
            net=runner.factory(name,settings)
            assert net.source_depth==2 and net.temporal_depth==2 and net.hidden_dim in (32,64)
    assert len(all_ids)==len(set(all_ids))==16


@pytest.mark.parametrize('change',['epochs','patience','weight_decay','source_depth','temporal_depth','width','dropout','lr','unsafe_id','scope','seed','test','pin','extra','selection'])
def test_mutated_source_grid_budget_and_scope_rejected_before_admission(synthetic,change):
    settings=deepcopy(synthetic.configs['allrecord_gru']);name=settings['models'][0]
    if change in ('epochs','patience','source_depth','temporal_depth'):settings[change]=1
    elif change=='weight_decay':settings[change]=0.
    elif change=='width':settings['model_specs'][name]['hidden_dim']=32.
    elif change=='dropout':settings['model_specs'][name]['dropout']=False
    elif change=='lr':settings['model_learning_rates'][name]=.002
    elif change=='unsafe_id':settings['models'][0]='../escape'
    elif change=='scope':settings['input_scope']='anchor-only'
    elif change=='seed':settings['training_seeds']=[101,202]
    elif change=='test':settings['splits']['test']=1
    elif change=='pin':settings['bound_implementation_sha256']['allrecord_boundary_temporal.py']='1'*64
    elif change=='extra':settings['extra_budget']=1000
    else:settings['selection']='minimum calibrated probability-clipped NLL'
    with patch.object(binding.common,'admit_run',side_effect=AssertionError('Must fail before admission')):
        with pytest.raises(RuntimeError):binding.validate_settings(settings)


def test_initial_config_bytes_captured_once_and_live_drift_detected(synthetic):
    path=synthetic.paths['allrecord_gru'];old=Path.read_bytes;counts={}
    def counted(p):counts[str(p)]=counts.get(str(p),0)+1;return old(p)
    with patch.object(Path,'read_bytes',counted):captured=binding.capture_settings(path,synthetic.refs['allrecord_gru'])
    assert counts[str(path)]==1 and captured.content==old(path)
    with path.open('ab') as handle:handle.write(b' ')
    with pytest.raises(RuntimeError,match='changed'):captured.check()


def test_plan_schema_and_typed_grid_reject_changes(synthetic):
    plan=deepcopy(synthetic.plan);name=plan['families']['allrecord_gru']['models'][0]
    plan['families']['allrecord_gru']['model_specs'][name]['hidden_dim']=32.
    with pytest.raises(RuntimeError,match='Typed'):binding.validate_plan(plan)
    plan=deepcopy(synthetic.plan);plan['epochs']=24
    with pytest.raises(RuntimeError):binding.validate_plan(plan)
    plan=deepcopy(synthetic.plan);plan['reviewed_implementation_sha256'].pop('allrecord_boundary_temporal.py')
    with pytest.raises(RuntimeError,match='source/dependency'):binding.validate_plan(plan)


def test_completed_binding_calls_real_verifier_then_writes_fresh_hash_bound_configs(synthetic):
    output=synthetic.root/'bound'
    with patch.object(binding.common,'verify_completion',return_value=(synthetic.completion,synthetic.dataset,{},synthetic.lock)) as verifier:
        values,contents,digests=binding.bind_completed_development(synthetic.completionpath,synthetic.completionsha,
            synthetic.planpath,synthetic.plansha,output_dir=output)
    verifier.assert_called_once_with(synthetic.completionpath,synthetic.completionsha,with_snapshot=True)
    assert synthetic.lock.checks==[True,True]
    for family in binding.FAMILIES:
        data=(output/(family+'.json')).read_bytes()
        assert data==contents[family] and digests[family]==hashlib.sha256(data).hexdigest()
        assert values[family]==synthetic.configs[family]
    with patch.object(binding.common,'verify_completion',side_effect=AssertionError('No reread for existing output')):
        with pytest.raises(RuntimeError,match='Fresh'):binding.bind_completed_development(synthetic.completionpath,synthetic.completionsha,
            synthetic.planpath,synthetic.plansha,output_dir=output)


def test_missing_completion_or_bad_dataset_cannot_create_executable_configs(synthetic):
    output=synthetic.root/'blocked'
    with patch.object(binding.common,'verify_completion',side_effect=RuntimeError('Not completed')):
        with pytest.raises(RuntimeError):binding.bind_completed_development(synthetic.completionpath,synthetic.completionsha,
            synthetic.planpath,synthetic.plansha,output_dir=output)
    assert not output.exists()
    synthetic.dataset.records.pop()
    with patch.object(binding.common,'verify_completion',return_value=(synthetic.completion,synthetic.dataset,{},synthetic.lock)):
        with pytest.raises(RuntimeError,match='640'):binding.bind_completed_development(synthetic.completionpath,synthetic.completionsha,
            synthetic.planpath,synthetic.plansha,output_dir=output)
    assert not output.exists()


def test_production_run_uses_shared_mandatory_admission_and_captured_plan(synthetic):
    family='allrecord_gru';captured=binding.capture_settings(synthetic.paths[family],synthetic.refs[family])
    with patch.object(binding.common,'admit_run',return_value=(synthetic.completion,synthetic.dataset,{},synthetic.lock)) as admission:
        result=binding.admit_captured_run(captured,imported_pins=synthetic.configs[family]['bound_implementation_sha256'])
    admission.assert_called_once_with(captured.values,captured.path,captured.sha256,imported_pins=captured.values['bound_implementation_sha256'])
    assert result[1] is synthetic.dataset
    assert synthetic.lock.hashes[str(synthetic.planpath)]==synthetic.plansha


def test_runtime_wrapper_uses_stat_only_epoch_checks_and_full_seals(synthetic):
    family='allrecord_gru';captured=binding.capture_settings(synthetic.paths[family],synthetic.refs[family])
    with patch.object(binding.common,'check_live',return_value=captured.values['bound_implementation_sha256']) as live:
        binding.check_live(captured.values,captured,synthetic.lock,full=False)
        binding.check_live(captured.values,captured,synthetic.lock,full=True)
    assert [call.kwargs['full_sources'] for call in live.call_args_list]==[False,True]
    assert all(call.kwargs['source_snapshot'] is synthetic.lock for call in live.call_args_list)


def test_heldout_and_escaped_paths_blocked_before_read_or_source_admission(synthetic):
    for path in (synthetic.root/'data/final_test/receipt.json',synthetic.root/'data/test/receipt.json',synthetic.root/'confirmatory_dataset/config.json','../escape.json'):
        with patch.object(Path,'read_bytes',side_effect=AssertionError('Must reject before file read')):
            with pytest.raises(RuntimeError):binding.capture_control(path,synthetic.plansha)
    escaped=synthetic.root/'escape';escaped.symlink_to('/tmp',target_is_directory=True)
    with pytest.raises(RuntimeError):binding.development_path(escaped/'receipt.json')
    with patch.object(binding.common,'verify_completion',side_effect=AssertionError('Must reject before data loading')):
        with pytest.raises(RuntimeError):binding.bind_completed_development(synthetic.completionpath,synthetic.completionsha,
            synthetic.planpath,synthetic.plansha,output_dir=synthetic.root/'final_test/configs')


def one_parent_cases():
    value=bank();labels=torch.zeros(len(value.positions),2,dtype=torch.long)
    tile=SimpleNamespace(bank=value,parent_position_indices=torch.arange(len(value.positions)),labels=labels)
    case=SimpleNamespace(parent_id='synthetic-source:val:0',metadata={'replicate':0},positions=value.positions,
        query_node_ids=value.query_node_ids,labels=labels,tiles=[tile])
    return case,{id(tile):value}


def test_runner_joins_labels_only_after_forward_and_rejects_overlap_correspondence():
    case,banks=one_parent_cases();net=model('allrecord_gru').eval()
    records,logits=runner.evaluate_allrecord(net,[case],banks)
    expected=logit_metrics(net(case.tiles[0].bank).detach().numpy(),case.labels.numpy())
    assert records[0]['parent_id']==case.parent_id and records[0]['nll']==expected['nll']
    assert logits[0].dtype==torch.float64
    case.tiles.append(case.tiles[0])
    with pytest.raises(RuntimeError,match='correspondence'):runner.evaluate_allrecord(net,[case],banks)


def test_label_counterfactual_changes_evaluation_score_but_never_source_predictions():
    case,banks=one_parent_cases();net=model('allrecord_transformer').eval()
    a,first=runner.evaluate_allrecord(net,[case],banks)
    case.labels=torch.ones_like(case.labels)
    b,second=runner.evaluate_allrecord(net,[case],banks)
    torch.testing.assert_close(first[0],second[0],rtol=0,atol=0)
    assert a[0]['nll']!=b[0]['nll']


def test_earliest_checkpoint_tie_budget_and_flattened_bindings_on_synthetic_cpu(tmp_path):
    torch.manual_seed(9)
    case,banks=one_parent_cases();case.tiles[0].labels=case.labels
    traincase=SimpleNamespace(tiles=case.tiles)
    class Tiny(torch.nn.Module):
        def __init__(self):super().__init__();self.bias=torch.nn.Parameter(torch.zeros(4,dtype=torch.float64))
        def forward(self,value):return self.bias.expand(len(value.positions),2,4)
        def configuration_manifest(self):return {'synthetic':True}
    net=Tiny();settings={'model_learning_rates':{'synthetic':.001}}
    actual={'protocol_sha256':'1'*64,'completion_receipt_sha256':'2'*64,
        'boundary_manifest_sha256':'3'*64,'implementation_sha256':{'source.py':'4'*64}}
    checkpoint=tmp_path/'selected.pt';checks=[]
    record={'parent_id':case.parent_id,'nll':1.25}
    predictions=[torch.zeros(len(case.positions),2,4,dtype=torch.float64)]
    with patch.object(runner,'evaluate_allrecord',return_value=([record],predictions)):
        result,_=runner.train_configuration(net,settings,'synthetic',101,{'train':[traincase],'val':[case]},
            banks,checkpoint,actual,integrity_check=lambda **kw:checks.append(kw['full']))
    assert result['best_epoch']==0 and len(result['history'])==9
    assert all(row['val_nll']==1.25 for row in result['history'])
    assert checks==[False]*18+[True]*2
    saved=torch.load(checkpoint,weights_only=True)
    assert saved['seed']==101 and saved['epoch']==0 and saved['configuration_id']=='synthetic'
    assert all(saved[key]==value for key,value in actual.items())


def test_parent_average_is_equally_weighted_not_record_count():
    assert runner.parent_mean([{'nll':1.},{'nll':3.}])==2.
    with pytest.raises(RuntimeError):runner.parent_mean([{'nll':float('nan')}])


def test_main_source_admission_precedes_any_device_query(synthetic):
    family='allrecord_gru';args=['--config',str(synthetic.paths[family]),'--expected-config-sha256',synthetic.refs[family],
        '--output',str(synthetic.root/'run'),'--device','cpu']
    calls=[]
    def reject_admission(*a,**kw):calls.append('admit');raise RuntimeError('Synthetic source rejection')
    with patch.object(runner,'admit_captured_run',side_effect=reject_admission),patch.object(runner,'require_device',side_effect=AssertionError('Device queried before admission')):
        with pytest.raises(RuntimeError,match='source rejection'):runner.main(args)
    assert calls==['admit'] and not (synthetic.root/'run').exists()
    def admit(*a,**kw):calls.append('admit');return synthetic.completion,synthetic.dataset,{},synthetic.lock
    def stop_at_device(*a,**kw):calls.append('device');raise RuntimeError('Stop before output/fitting')
    calls=[]
    with patch.object(runner,'admit_captured_run',side_effect=admit),patch.object(runner,'require_device',side_effect=stop_at_device):
        with pytest.raises(RuntimeError,match='Stop before output'):runner.main(args)
    assert calls==['admit','device'] and not (synthetic.root/'run').exists()


def test_extensible_generic_selector_includes_both_allrecord_families_in_full_matrix(study):
    # Real small artifact files with synthetic scores; no actual models fitted.
    root=study['root'];source_family='gru'
    source_refs=[ref for ref in study['refs'] if ref['family_id']==source_family]
    for family in binding.FAMILIES:
        protocol=deepcopy(study['configs'][source_family]);protocol.update(family=family,**binding.family_grid(family))
        protocol_ref=write_selection_json(root,'configs/'+family+'.json',protocol)
        study['plan']['families'][family]={'role':'allrecord_boundary_'+family.removeprefix('allrecord_'),
            'grid_kind':'width_dropout_lr','protocol':protocol_ref,
            'expected_implementation_sha256':protocol['bound_implementation_sha256'],
            'input_scope':protocol['input_scope'],'cross_tree_aggregation':protocol['cross_tree_aggregation']}
        study['plan']['required_baseline_families'].append(family)
        for original_ref in source_refs:
            run=json.loads((root/original_ref['path']).read_text());run['family']=family
            run['protocol_sha256']=protocol_ref['sha256'];models={}
            for new_name,old_name in zip(protocol['models'],study['configs'][source_family]['models']):
                body=run['models'][old_name]
                body['ending_config_sha256']=protocol_ref['sha256']
                for pathkey,hashkey,suffix in (('checkpoint','checkpoint_sha256','checkpoint'),('validation_logits','validation_logits_sha256','logits')):
                    payload=torch.load(root/body[pathkey],weights_only=True);payload['protocol_sha256']=protocol_ref['sha256']
                    artifact=write_tensor(root,f'artifacts/{family}/{new_name}_s{run["training_seed"]}_{suffix}.pt',payload)
                    body[pathkey]=artifact['path'];body[hashkey]=artifact['sha256']
                models[new_name]=body
            run['models']=models
            study['refs'].append({'family_id':family,**write_selection_json(root,f'runs/{family}_s{run["training_seed"]}.json',run)})
    study['plan_ref']=write_selection_json(root,'plan.json',study['plan'])
    receipt=synth_selection(study).payload
    assert receipt['matrix_counts']=={'success':192,'failed':0,'missing':0}
    assert receipt['baseline_selection_eligible'] and len(receipt['required_baseline_families'])==7
    for family in binding.FAMILIES:
        assert receipt['families'][family]['family_selection_eligible']
        assert receipt['families'][family]['input_scope']==binding.INPUT_SCOPE
