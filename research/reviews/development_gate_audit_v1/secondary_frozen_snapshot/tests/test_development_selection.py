"""Synthetic CPU evidence matrix, frozen artifacts, and fair family selection."""
from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

import arggnn.development_selection as selection
from arggnn.development_selection import (
    synthesize_development_selection,load_development_selection_plan,
    tensor_fingerprint,RAW_NLL,PARENT_NLL,PARENT_WEIGHT,SEEDS,
)

torch.set_num_threads(1)


def write_json(root,name,value):
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
    content=(json.dumps(value,sort_keys=True,indent=2)+'\n').encode();path.write_bytes(content)
    return {'path':name,'sha256':hashlib.sha256(content).hexdigest()}


def write_tensor(root,name,value):
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
    payload=io.BytesIO();torch.save(value,payload);content=payload.getvalue();path.write_bytes(content)
    return {'path':name,'sha256':hashlib.sha256(content).hexdigest()}


def rebind_run(study,ref,mutate):
    content=json.loads((study['root']/ref['path']).read_text())
    mutate(content)
    return {'family_id':ref['family_id'],**write_json(study['root'],ref['path'],content)}


@pytest.fixture
def study(tmp_path):
    root=tmp_path
    producer=root/'producer.py';producer.write_text('# Synthetic receipt producer bytes, never executed.\n')
    pins={'producer.py':hashlib.sha256(producer.read_bytes()).hexdigest()}
    freeze=write_json(root,'freeze.json',{'kind':'fresh_development_preparation_execution_freeze',
        'confirmatory_access':False,'not_confirmatory_test_freeze':True})
    source=write_json(root,'source.json',{'protocol':{'splits':{'train':64,'val':16,'test':0},
        'master_seed':20261101,'inference':{'master_seed':20261102}},
        'replicates':[{'split':s,'replicate':i} for s,n in (('train',64),('val',16)) for i in range(n)]})
    bprep=hashlib.sha256(b'boundary-preparation').hexdigest();aprep=hashlib.sha256(b'age-preparation').hexdigest()
    boundary=write_json(root,'boundary.json',{'status':'complete','preparation_sha256':bprep,
        'source':{'manifest_sha256':source['sha256']}})
    age=write_json(root,'age.json',{'status':'complete','preparation_sha256':aprep,
        'source_manifest_sha256':source['sha256'],'boundary_manifest_sha256':boundary['sha256'],
        'boundary_preparation_sha256':bprep})
    completion=write_json(root,'completion.json',{'all_stages_complete':True,'train_parents':64,
        'validation_parents':16,'test_parents':0,'inferred_boundary_and_age_tiles':640,
        'training_or_confirmatory_selection_frozen':False,
        'development_execution_freeze_file_sha256':freeze['sha256'],
        'stage_manifests':{'inferred':source,'boundary':boundary,'age':age}})
    binding={'completion_receipt_sha256':completion['sha256'],'source_dataset_manifest_sha256':source['sha256'],
        'boundary_dataset_manifest_sha256':boundary['sha256'],'boundary_preparation_sha256':bprep,
        'age_dataset_manifest_sha256':age['sha256'],'age_preparation_sha256':aprep,
        'preparation_execution_freeze_sha256':freeze['sha256']}
    x=torch.tensor([.25,.75],dtype=torch.float64);q=torch.tensor([10,20])
    parents=[{'parent_id':source['sha256']+':val:'+str(i),'positions_sha256':tensor_fingerprint(x),
              'query_ids_sha256':tensor_fingerprint(q),'logit_shape':[2,2,4]} for i in range(16)]
    roles={'candidate':'candidate','gru':'same_source_gru','transformer':'same_source_transformer',
           'canonical':'canonical_tree_gru','argmix':'argmix_adaptation','smoother':'observed_reference_smoother'}
    families={};configs={}
    for family,role in roles.items():
        config={'family':family,'development_namespace':'development-v1','training_seeds':list(SEEDS),
            'splits':{'train':64,'val':16,'test':0},'nll_policy':RAW_NLL,**binding,
            'epochs':48,'patience':8,'weight_decay':.0001,
            'selection':'minimum mean parent-validation uncalibrated logit NLL',
            'input_scope':'Synthetic '+family+' observed-input contract',
            'cross_tree_aggregation':'Synthetic '+family+' aggregation contract',
            'models':[],'model_specs':{},
            'model_learning_rates':{},'bound_implementation_sha256':pins}
        if family=='candidate':
            grid=[(g,d,lr) for g in ('plain','age') for d in (1,2) for lr in (.0003,.001)]
            kind='candidate'
            for i,(g,d,lr) in enumerate(grid):
                name=family+'_'+str(i);config['models'].append(name)
                config['model_specs'][name]={'geometry':g,'kernel_depth':d,'coupling':'full','rate_parameterization':'log'}
                config['model_learning_rates'][name]=lr
        elif family=='smoother':
            kind='bandwidth_pseudocount'
            for i,(bw,pc) in enumerate((bw,pc) for bw in (10000,30000,100000,300000) for pc in (.001,.01)):
                name=family+'_'+str(i);config['models'].append(name)
                config['model_specs'][name]={'bandwidth_bp':bw,'pseudocount':pc}
        else:
            kind='width_dropout_lr'
            for i,(h,d,lr) in enumerate((h,d,lr) for h in (32,64) for d in (0.,.1) for lr in (.0003,.001)):
                name=family+'_'+str(i);config['models'].append(name)
                config['model_specs'][name]={'hidden_dim':h,'dropout':d};config['model_learning_rates'][name]=lr
        config_ref=write_json(root,'configs/'+family+'.json',config)
        families[family]={'role':role,'grid_kind':kind,'protocol':config_ref,'expected_implementation_sha256':pins,
            'input_scope':config['input_scope'],'cross_tree_aggregation':config['cross_tree_aggregation']}
        configs[family]=config
    plan={'schema_version':1,'kind':'fresh_development_family_selection_plan','development_namespace':'development-v1',
        'training_seeds':list(SEEDS),'splits':{'train':64,'val':16,'test':0},'nll_policy':RAW_NLL,
        'tie_tolerance':.0001,'source_binding':binding,'source_artifacts':{'completion':completion,
        'source_manifest':source,'boundary_manifest':boundary,'age_manifest':age,'preparation_execution_freeze':freeze},
        'validation_parents':parents,'families':families,'required_baseline_families':[f for f in roles if f!='candidate'],
        'implementation_artifacts':{'producer.py':{'path':'producer.py','sha256':pins['producer.py']}}}
    plan_ref=write_json(root,'plan.json',plan)
    refs=[]
    for family_index,(family,config) in enumerate(configs.items()):
        for seed_index,seed in enumerate(SEEDS):
            run={'development_namespace':'development-v1','evaluation_split':'fresh_development_validation_only',
                'training_seeds':list(SEEDS),'training_seed':seed,'family':family,'nll_policy':RAW_NLL,
                'protocol_sha256':families[family]['protocol']['sha256'],
                'completion_receipt_sha256':completion['sha256'],'boundary_dataset_manifest_sha256':boundary['sha256'],
                'initial_implementation_sha256':pins,'models':{}}
            for i,name in enumerate(config['models']):
                target=1.1+.03*family_index+.001*i+(.0001*seed_index if family!='smoother' else 0.)
                values=[target+.001*(j-7.5) for j in range(16)]
                logits=[]
                for score in values:
                    z=torch.zeros(2,2,4,dtype=torch.float64)
                    z[:,:,0]=-math.log(math.expm1(score)/3)
                    logits.append(z)
                prediction=write_tensor(root,f'artifacts/{family}/{name}_s{seed}_logits.pt',{
                    'logits':logits,'positions':[x.clone() for _ in parents],'query_node_ids':[q.clone() for _ in parents],
                    'parent_ids':[p['parent_id'] for p in parents],'protocol_sha256':run['protocol_sha256'],
                    'seed':seed,'nll_policy':RAW_NLL,'completion_receipt_sha256':completion['sha256']})
                count=0 if family=='smoother' else 100+100*family_index+i
                checkpoint=None
                if count:
                    checkpoint=write_tensor(root,f'artifacts/{family}/{name}_s{seed}_checkpoint.pt',{
                        'model':{'weight':torch.zeros(count,dtype=torch.float64)},'epoch':1,'seed':seed,
                        'validation_nll':target,'protocol_sha256':run['protocol_sha256'],
                        'completion_receipt_sha256':completion['sha256'],'boundary_manifest_sha256':boundary['sha256'],
                        'implementation_sha256':pins})
                run['models'][name]={'parameters':count,'validation_nll':target,
                    'per_parent':[{'parent_id':p['parent_id'],'nll':score,'nll_policy':PARENT_NLL,'weighting':PARENT_WEIGHT}
                                  for p,score in zip(parents,values)],
                    'best_epoch':None if not count else 1,
                    'history':[] if not count else [{'epoch':0,'val_nll':target+.1},{'epoch':1,'val_nll':target},{'epoch':2,'val_nll':target}],
                    'checkpoint':None if checkpoint is None else checkpoint['path'],
                    'checkpoint_sha256':None if checkpoint is None else checkpoint['sha256'],
                    'validation_logits':prediction['path'],'validation_logits_sha256':prediction['sha256'],
                    'ending_config_sha256':run['protocol_sha256'],'ending_implementation_sha256':pins,
                    'smoother_fitting':'none; seed-independent deterministic predictor' if not count else None}
            refs.append({'family_id':family,**write_json(root,f'runs/{family}_s{seed}.json',run)})
    return {'root':root,'plan':plan,'plan_ref':plan_ref,'refs':refs,'configs':configs,'parents':parents}


def synth(study,refs=None,ledgers=(),output=None):
    return synthesize_development_selection(study['plan_ref']['path'],study['plan_ref']['sha256'],
        run_receipts=study['refs'] if refs is None else refs,failure_ledgers=ledgers,
        output_path=output,project_root=study['root'])


def ledger(study,entries):
    return write_json(study['root'],'ledger.json',{'schema_version':1,'kind':'development_failure_ledger',
        'plan_sha256':study['plan_ref']['sha256'],'entries':entries})


def failed_entry(seed=101,terminal=True,classification='numerical'):
    return {'family_id':'gru','configuration_id':'gru_0','seed':seed,'attempt_id':'attempt_1',
        'terminal':terminal,'failure_class':classification,'reason':'Synthetic deliberate failure','test_access':False}


def test_full_exact_matrix_choice_and_immutable_receipt_hash(study):
    frozen=load_development_selection_plan(study['plan_ref']['path'],study['plan_ref']['sha256'],project_root=study['root'])
    assert frozen.sha256==study['plan_ref']['sha256']
    result=synth(study,output='selection.json')
    assert result.payload['matrix_counts']=={'success':144,'failed':0,'missing':0}
    assert result.payload['matrix_resolved'] and result.payload['baseline_selection_eligible']
    assert result.payload['candidate_choice']['configuration_id']=='candidate_0'
    assert result.payload['strongest_required_baseline']['configuration_id']=='gru_0'
    assert result.sha256==hashlib.sha256((study['root']/'selection.json').read_bytes()).hexdigest()
    assert result.payload['promotes_final_test_phase'] is False
    with pytest.raises(FileExistsError):synth(study,output='selection.json')


def test_missing_seed_blocks_complete_family_and_no_successful_seed_average(study):
    refs=study['refs'][3:]
    result=synth(study,refs)
    assert result.payload['matrix_counts']['missing']==24
    assert result.payload['candidate_choice'] is None
    partial=[r for r in study['refs'] if not(r['family_id']=='argmix' and '_s303' in r['path'])]
    receipt=synth(study,partial).payload
    assert receipt['baseline_selection_eligible'] is False and receipt['strongest_required_baseline'] is None
    item=receipt['families']['argmix']['configurations'][0]
    assert item['mean_three_seed_nll'] is None and item['seed_scores']['303'] is None


def test_terminal_failure_disqualifies_config_preserves_other_scores(study):
    refs=deepcopy(study['refs']); ref=refs[3]
    refs[3]=rebind_run(study,ref,lambda run:run['models'].pop('gru_0'))
    receipt=synth(study,refs,[ledger(study,[failed_entry()])]).payload
    assert receipt['matrix_counts']=={'success':143,'failed':1,'missing':0}
    item=receipt['families']['gru']['configurations'][0]
    assert not item['eligible'] and item['mean_three_seed_nll'] is None
    assert item['seed_scores']['202'] is not None and item['seed_statuses']['101']=='failed'
    assert receipt['strongest_required_baseline']['configuration_id']=='gru_1'


def test_nonterminal_infrastructure_retry_is_preserved_without_extra_search_cell(study):
    receipt=synth(study,ledgers=[ledger(study,[failed_entry(terminal=False,classification='infrastructure')])]).payload
    assert receipt['matrix_counts']['success']==144 and len(receipt['failures_and_retry_ledger'])==1
    assert receipt['strongest_required_baseline'] is not None
    with pytest.raises(ValueError):synth(study,ledgers=[ledger(study,[failed_entry()])])


def test_duplicate_success_and_extra_seed_rejected(study):
    with pytest.raises(ValueError,match='duplicate'):synth(study,study['refs']+[study['refs'][0]])
    refs=deepcopy(study['refs'])
    refs[0]=rebind_run(study,refs[0],lambda run:run.update(training_seed=404))
    with pytest.raises(ValueError,match='seed'):synth(study,refs)


@pytest.mark.parametrize('field',['score','policy','parents','epoch','pins','ending_hash'])
def test_unfair_or_unbound_success_metadata_rejected(study,field):
    refs=deepcopy(study['refs'])
    def mutate(run):
        body=run['models']['candidate_0']
        if field=='score':body['validation_nll']+=.01
        elif field=='policy':body['per_parent'][0]['nll_policy']='probability-clipped NLL'
        elif field=='parents':body['per_parent'].reverse()
        elif field=='epoch':body['best_epoch']=2
        elif field=='pins':body['ending_implementation_sha256']={'producer.py':'1'*64}
        else:body['ending_config_sha256']='1'*64
    refs[0]=rebind_run(study,refs[0],mutate)
    with pytest.raises(ValueError):synth(study,refs)


@pytest.mark.parametrize('artifact_kind',['checkpoint','prediction'])
def test_real_artifact_hash_and_metadata_validation(study,artifact_kind):
    ref=study['refs'][0];run=json.loads((study['root']/ref['path']).read_text());body=run['models']['candidate_0']
    key='checkpoint' if artifact_kind=='checkpoint' else 'validation_logits'
    path=study['root']/body[key]
    with path.open('ab') as handle:handle.write(b'corruption')
    with pytest.raises(ValueError,match='hash mismatch'):synth(study)


def test_prediction_query_order_and_float64_contract_checked_after_hash(study):
    refs=deepcopy(study['refs']);run=json.loads((study['root']/refs[0]['path']).read_text());body=run['models']['candidate_0']
    artifact=study['root']/body['validation_logits'];payload=torch.load(artifact,weights_only=True)
    payload['query_node_ids'][0]=payload['query_node_ids'][0].flip(0)
    changed=write_tensor(study['root'],body['validation_logits'],payload)
    refs[0]=rebind_run(study,refs[0],lambda run:run['models']['candidate_0'].update(validation_logits_sha256=changed['sha256']))
    with pytest.raises(ValueError,match='query order'):synth(study,refs)


def test_single_global_tie_window_no_chained_family_tolerance():
    configurations=[{'family_id':'x','configuration_id':'a','eligible':True,'parameters':100,'mean_three_seed_nll':1.},
        {'family_id':'y','configuration_id':'b','eligible':True,'parameters':20,'mean_three_seed_nll':1.00009},
        {'family_id':'y','configuration_id':'c','eligible':True,'parameters':1,'mean_three_seed_nll':1.00018},
        {'family_id':'z','configuration_id':'d','eligible':False,'parameters':0,'mean_three_seed_nll':None}]
    winner=selection._winner(configurations)
    assert winner['configuration_id']=='b'
    assert ('y','c') not in winner['tie_ids']


def test_config_bytes_captured_once_and_drift_before_seal_rejected(study):
    counts={};old=Path.read_bytes
    def counted(path):
        key=str(path)
        counts[key]=counts.get(key,0)+1
        return old(path)
    with patch.object(Path,'read_bytes',counted):synth(study)
    for family in study['configs']:
        assert counts[str(study['root']/'configs'/f'{family}.json')]==1
    original=selection._cell;changed=False
    def mutate_during(*args,**kwargs):
        nonlocal changed
        result=original(*args,**kwargs)
        if not changed:
            with (study['root']/'configs/candidate.json').open('a') as handle:handle.write(' ')
            changed=True
        return result
    with patch.object(selection,'_cell',mutate_during),pytest.raises(ValueError,match='before selection sealing'):synth(study)


def test_path_confinement_safe_ids_and_heldout_phase_paths(study):
    with pytest.raises(ValueError):synthesize_development_selection('../outside.json',study['plan_ref']['sha256'],project_root=study['root'])
    with pytest.raises(ValueError):synth(study,output='../outside-selection.json')
    held=study['root']/'data/ancient_confirmatory_cohort';held.mkdir(parents=True)
    heldfile=held/'receipt.json';heldfile.write_text('{}')
    with pytest.raises(ValueError,match='Final-test'):selection._Capture(study['root']).path(heldfile)
    link=study['root']/'escape';link.symlink_to(study['root'].parent,target_is_directory=True)
    with pytest.raises(ValueError):selection._Capture(study['root']).path('escape')


def test_legacy_parent_and_prediction_field_aliases_require_real_bindings(study):
    refs=deepcopy(study['refs']);run=json.loads((study['root']/refs[0]['path']).read_text())
    for body in run['models'].values():
        body['per_replicate']=[{**row,'replicate':i} for i,row in enumerate(body.pop('per_parent'))]
        for row in body['per_replicate']:row.pop('parent_id')
        body['validation_predictions']=body.pop('validation_logits')
        body['validation_predictions_sha256']=body.pop('validation_logits_sha256')
        payload=torch.load(study['root']/body['validation_predictions'],weights_only=True)
        payload['parent_replicates']=list(range(16));payload.pop('parent_ids')
        ref=write_tensor(study['root'],body['validation_predictions'],payload)
        body['validation_predictions_sha256']=ref['sha256']
    refs[0]={'family_id':'candidate',**write_json(study['root'],refs[0]['path'],run)}
    assert synth(study,refs).payload['matrix_counts']['success']==144
    refs[0]=rebind_run(study,refs[0],lambda r:r.pop('completion_receipt_sha256'))
    with pytest.raises(ValueError,match='completion'):synth(study,refs)


def update_plan(study):
    study['plan_ref']=write_json(study['root'],'plan.json',study['plan'])


def replace_protocol(study,family,config):
    """Rebind synthetic factual config/artifact hashes; never fill attestations."""
    protocol=write_json(study['root'],'configs/'+family+'.json',config)
    study['plan']['families'][family]['protocol']=protocol
    update_plan(study)
    refs=[]
    for ref in study['refs']:
        if ref['family_id']!=family:
            refs.append(ref);continue
        run=json.loads((study['root']/ref['path']).read_text())
        run['protocol_sha256']=protocol['sha256']
        for body in run['models'].values():
            body['ending_config_sha256']=protocol['sha256']
            for pathkey,hashkey in (('checkpoint','checkpoint_sha256'),('validation_logits','validation_logits_sha256')):
                if body[pathkey] is None:continue
                payload=torch.load(study['root']/body[pathkey],weights_only=True)
                payload['protocol_sha256']=protocol['sha256']
                artifact=write_tensor(study['root'],body[pathkey],payload)
                body[hashkey]=artifact['sha256']
        refs.append({'family_id':family,**write_json(study['root'],ref['path'],run)})
    study['refs']=refs


def test_all_cells_failed_family_cannot_be_omitted_from_baseline_choice(study):
    refs=[ref for ref in study['refs'] if ref['family_id']!='argmix']
    failures=[{**failed_entry(seed),'family_id':'argmix','configuration_id':name}
        for name in study['configs']['argmix']['models'] for seed in SEEDS]
    receipt=synth(study,refs,[ledger(study,failures)]).payload
    assert receipt['matrix_resolved'] and receipt['matrix_counts']['failed']==24
    assert receipt['families']['argmix']['matrix_resolved']
    assert receipt['families']['argmix']['family_selection_eligible'] is False
    assert receipt['baseline_selection_eligible'] is False and receipt['strongest_required_baseline'] is None


def test_parameterizable_families_explicit_requirement_and_reported_input_scope(study):
    config=deepcopy(study['configs']['gru']);family='boundary_gru'
    config['family']=family
    config['input_scope']='Every observed tree with the shared observed boundary encoder'
    config['cross_tree_aggregation']='Ordered all-tree GRU tokens'
    config['models']=[name.replace('gru_',family+'_') for name in config['models']]
    config['model_specs']={name.replace('gru_',family+'_'):value for name,value in config['model_specs'].items()}
    config['model_learning_rates']={name.replace('gru_',family+'_'):value for name,value in config['model_learning_rates'].items()}
    spec=deepcopy(study['plan']['families']['gru'])
    spec.update(role='all_boundary_gru',input_scope=config['input_scope'],cross_tree_aggregation=config['cross_tree_aggregation'],
                protocol=write_json(study['root'],'configs/'+family+'.json',config))
    study['plan']['families'][family]=spec
    study['plan']['required_baseline_families'].append(family);update_plan(study)
    receipt=synth(study,refs=[]).payload
    assert receipt['matrix_counts']=={'success':0,'failed':0,'missing':168}
    assert receipt['families'][family]['input_scope']==config['input_scope']
    assert family in receipt['required_baseline_families']
    study['plan']['required_baseline_families'].remove(family);update_plan(study)
    with pytest.raises(ValueError,match='silent omission'):synth(study,refs=[])


def test_configurations_list_normalization_preserves_original_hash_and_bytes(study):
    config=deepcopy(study['configs']['argmix'])
    config['configurations']=[{'id':name,**config['model_specs'][name],'learning_rate':config['model_learning_rates'][name]}
        for name in config.pop('models')]
    config.pop('model_specs');config.pop('model_learning_rates')
    replace_protocol(study,'argmix',config)
    before=(study['root']/'configs/argmix.json').read_bytes()
    receipt=synth(study).payload
    assert receipt['matrix_counts']['success']==144
    assert receipt['families']['argmix']['protocol']['sha256']==hashlib.sha256(before).hexdigest()
    assert (study['root']/'configs/argmix.json').read_bytes()==before


def test_relative_dependency_pin_aliases_capture_same_source_once(study):
    config=deepcopy(study['configs']['argmix']);pin='src/arggnn/producer.py'
    digest=config['bound_implementation_sha256'].pop('producer.py')
    config['bound_implementation_sha256'][pin]=digest
    study['plan']['families']['argmix']['expected_implementation_sha256']={pin:digest}
    study['plan']['implementation_artifacts'][pin]={'path':'producer.py','sha256':digest}
    study['plan']['families']['argmix']['protocol']=write_json(study['root'],'configs/argmix.json',config)
    update_plan(study)
    receipt=synth(study,refs=[]).payload
    artifacts=[a for a in receipt['source_artifacts'] if a['path']=='producer.py']
    assert len(artifacts)==1
    study['plan']['families']['argmix']['expected_implementation_sha256']={'../producer.py':digest}
    config['bound_implementation_sha256']={'../producer.py':digest}
    study['plan']['families']['argmix']['protocol']=write_json(study['root'],'configs/argmix.json',config)
    update_plan(study)
    with pytest.raises(ValueError,match='relative dependency'):synth(study,refs=[])


@pytest.mark.parametrize('setting',['selection','width','scope','pin','candidate_role'])
def test_bound_protocol_and_scope_hardening(study,setting):
    family='gru';config=deepcopy(study['configs'][family])
    if setting=='selection':config['selection']='minimum calibrated accuracy loss'
    elif setting=='width':config['model_specs']['gru_0']['hidden_dim']=32.
    elif setting=='scope':config['input_scope']='Different information access'
    elif setting=='pin':config['bound_implementation_sha256']={'producer.py':'f'*64}
    else:study['plan']['families'][family]['role']='candidate'
    study['plan']['families'][family]['protocol']=write_json(study['root'],'configs/'+family+'.json',config)
    update_plan(study)
    with pytest.raises(ValueError):synth(study,refs=[])


@pytest.mark.parametrize('artifact_kind',['checkpoint','prediction'])
def test_artifact_metadata_mismatch_rejected_even_after_hash_rebound(study,artifact_kind):
    refs=deepcopy(study['refs']);run=json.loads((study['root']/refs[0]['path']).read_text());body=run['models']['candidate_0']
    key='checkpoint' if artifact_kind=='checkpoint' else 'validation_logits'
    digestkey='checkpoint_sha256' if artifact_kind=='checkpoint' else 'validation_logits_sha256'
    payload=torch.load(study['root']/body[key],weights_only=True);payload['seed']=202
    changed=write_tensor(study['root'],body[key],payload)
    refs[0]=rebind_run(study,refs[0],lambda run:run['models']['candidate_0'].update({digestkey:changed['sha256']}))
    with pytest.raises(ValueError,match='seed mismatch'):synth(study,refs)


def test_failure_attempt_duplicates_and_unlogged_model_retry_rejected(study):
    entry=failed_entry(terminal=False,classification='infrastructure')
    with pytest.raises(ValueError,match='Duplicate ledgered attempt'):synth(study,ledgers=[ledger(study,[entry,entry])])
    with pytest.raises(ValueError,match='Only logged infrastructure'):synth(study,ledgers=[ledger(study,[failed_entry(terminal=False)])])


def test_no_fit_smoother_requires_actual_cross_seed_prediction_identity(study):
    refs=deepcopy(study['refs']);index=next(i for i,ref in enumerate(refs) if ref['family_id']=='smoother')
    run=json.loads((study['root']/refs[index]['path']).read_text());body=run['models']['smoother_0']
    payload=torch.load(study['root']/body['validation_logits'],weights_only=True)
    payload['logits'][0][0,0,0]+=.001
    changed=write_tensor(study['root'],body['validation_logits'],payload)
    refs[index]=rebind_run(study,refs[index],lambda run:run['models']['smoother_0'].update(validation_logits_sha256=changed['sha256']))
    with pytest.raises(ValueError,match='Seed-independent'):synth(study,refs)


def test_history_rejects_boolean_epoch_metadata(study):
    refs=deepcopy(study['refs'])
    refs[0]=rebind_run(study,refs[0],lambda run:run['models']['candidate_0']['history'][1].update(epoch=True))
    with pytest.raises(ValueError,match='ordered training history'):synth(study,refs)


def test_prescribed_smoother_cannot_attest_fitted_parameters(study):
    refs=deepcopy(study['refs']);index=next(i for i,ref in enumerate(refs) if ref['family_id']=='smoother')
    refs[index]=rebind_run(study,refs[index],lambda run:run['models']['smoother_0'].update(parameters=1))
    with pytest.raises(ValueError,match='zero fitted parameters'):synth(study,refs)
