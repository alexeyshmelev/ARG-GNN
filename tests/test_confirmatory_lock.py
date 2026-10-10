"""Synthetic temporary-project evidence only. No real lock, test cohort or GPU."""
from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from arggnn import confirmatory_lock as lock
from arggnn import development_selection as selection
from arggnn import development_selection_bindings as combined
from arggnn import secondary_control_bindings as secondary
from arggnn import development_execution as numerical
from arggnn.development_selection import tensor_fingerprint, RAW_NLL, PARENT_NLL, PARENT_WEIGHT, SEEDS

ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(1)

def write_json(root,name,value):
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
    content=(json.dumps(value,sort_keys=True,indent=2)+'\n').encode();path.write_bytes(content)
    return {'path':name,'sha256':hashlib.sha256(content).hexdigest()}

def write_tensor(root,name,value):
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
    payload=io.BytesIO();torch.save(value,payload);content=payload.getvalue();path.write_bytes(content)
    return {'path':name,'sha256':hashlib.sha256(content).hexdigest()}

def make_synthetic_study(tmp_path):
    root=tmp_path
    producer=root/'producer.py';producer.write_text('# Synthetic receipt producer bytes, never executed.\n')
    pins={'producer.py':hashlib.sha256(producer.read_bytes()).hexdigest()}
    freeze=write_json(root,'freeze.json',{'kind':'fresh_development_preparation_execution_freeze',
        'confirmatory_access':False,'not_confirmatory_test_freeze':True})
    source=write_json(root,'source.json',{'protocol':{'splits':{'train':64,'val':16,'test':0},
        'master_seed':20261101,'inference':{'master_seed':20261102}},
        'replicates':[{'split':s,'replicate':i,'seeds':{'ancestry':10000+offset+i,'mutations':20000+offset+i},'inference_seed':30000+offset+i} for s,n,offset in (('train',64,0),('val',16,64)) for i in range(n)]})
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
    roles=dict(combined.ROLES)
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
        elif family=='reference_smoother':
            kind='bandwidth_pseudocount'
            for i,(bw,pc) in enumerate((bw,pc) for bw in (10000,30000,100000,300000) for pc in (.001,.01)):
                name=family+'_'+str(i);config['models'].append(name)
                config['model_specs'][name]={'bandwidth_bp':bw,'pseudocount':pc}
        else:
            kind='width_dropout_lr'
            for i,(h,d,lr) in enumerate((h,d,lr) for h in (32,64) for d in (0.,.1) for lr in (.0003,.001)):
                name=family+'_'+str(i);config['models'].append(name)
                config['model_specs'][name]={'hidden_dim':h,'dropout':d};config['model_learning_rates'][name]=lr
        config[combined.POLICY_FIELDS[family]]=deepcopy(numerical.DETERMINISTIC_POLICY)
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
            run[combined.POLICY_FIELDS[family]]=deepcopy(numerical.DETERMINISTIC_POLICY)
            for i,name in enumerate(config['models']):
                target=1.1+.03*family_index+.001*i+(.0001*seed_index if family!='reference_smoother' else 0.)
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
                count=0 if family=='reference_smoother' else 100+100*family_index+i
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
                    'smoother_fitting':'none; seed-independent deterministic predictor' if not count else None,
                    'original_selected_validation_nll':target if count else None,'reloaded_validation_nll':target if count else None,
                    'checkpoint_replay_nll_delta':0. if count else None}
            refs.append({'family_id':family,**write_json(root,f'runs/{family}_s{seed}.json',run)})
    return {'root':root,'plan':plan,'plan_ref':plan_ref,'refs':refs,'configs':configs,'parents':parents}


def toy_secondary_select(root, config_ref, run_receipts, failure_ledgers=()):
    """Real captured synthetic checkpoints/logits through production _cell.

    Production additionally re-admits full datasets/factories via its existing
    separately tested selector. This stub isolates the final lock integration.
    """
    capture=selection._Capture(root);settings=capture.read(config_ref)
    spec={'role':'secondary_control','protocol':config_ref,'expected_implementation_sha256':settings['bound_implementation_sha256']}
    plan={'families':{'secondary_controls':spec},'source_binding':settings['source_binding'],'validation_parents':settings['validation_parents']}
    cells=[]
    for ref in run_receipts:
        run=capture.read(ref)
        for name,body in run['models'].items():
            cells.append(selection._cell(capture,plan,{'secondary_controls':settings},'secondary_controls',name,run['training_seed'],body,ref['sha256']))
    assert not failure_ledgers
    controls={}
    for control in settings['controls']:
        configurations=[]
        for name in settings['models']:
            if settings['model_specs'][name]['control']!=control:continue
            rows=[row for row in cells if row['configuration_id']==name]
            configurations.append({'family_id':control,'configuration_id':name,'eligible':len(rows)==3,
                'parameters':rows[0]['parameters'],'mean_three_seed_nll':math.fsum(row['validation_nll'] for row in rows)/3})
        controls[control]={'matrix_resolved':True,'configurations':configurations,'choice':selection._winner(configurations)}
    return {'kind':'secondary_development_attribution_selection_receipt','schema_version':1,'config_sha256':config_ref['sha256'],
        'main_selection_plan':settings['main_selection_plan'],'main_selection_receipt':settings['main_selection_receipt'],
        'selected_candidate':settings['selected_candidate'],'primary_strongest_baseline':settings['primary_strongest_baseline'],
        'controls':controls,'cells':cells,'failures':[],'optimizer_starts_recorded':len(cells),'maximum_optimizer_starts':len(cells),
        'matrix_resolved':True,'source_binding':settings['source_binding'],'deterministic_execution_policy':deepcopy(numerical.DETERMINISTIC_POLICY),
        'captured_artifacts':capture.provenance(),'interpretation':'SYNTHETIC UNIT TEST ONLY',
        'primary_selection_changed':False,'final_test_access':False,'authorizes_confirmatory_generation':False}


@pytest.fixture(scope='module')
def evidence(tmp_path_factory):
    root=tmp_path_factory.mktemp('synthetic_lock_study')
    study=make_synthetic_study(root)
    patches=pytest.MonkeyPatch()
    patches.setenv('CUDA_VISIBLE_DEVICES','')
    patches.setattr(combined,'_family_contract',lambda family:{
        'input_scope':study['configs'][family]['input_scope'],'cross_tree_aggregation':study['configs'][family]['cross_tree_aggregation'],
        'pins':study['plan']['families'][family]['expected_implementation_sha256']})
    def producer_rebind(*args, **kwargs):
        assert kwargs['output_directory'] is None
        assert len(kwargs['family_configs'])==10
        assert {ref['family_id'] for ref in kwargs['family_configs']}==set(combined.FAMILIES)
        return SimpleNamespace(plan=deepcopy(study['plan']))
    patches.setattr(combined,'bind_development_selection',producer_rebind)
    def primary_model(family,name,settings):
        index=list(study['configs']).index(family);offset=settings['models'].index(name)
        count=0 if family=='reference_smoother' else 100+100*index+offset
        model=torch.nn.Module()
        if count:model.register_parameter('weight',torch.nn.Parameter(torch.zeros(count,dtype=torch.float64)))
        return model
    patches.setattr(lock,'_primary_model',primary_model)
    main_receipt=selection.synthesize_development_selection(root/study['plan_ref']['path'],study['plan_ref']['sha256'],
        run_receipts=study['refs'],project_root=root).payload
    main_ref=write_json(root,'main_selection.json',main_receipt)
    selected_spec=study['configs']['candidate']['model_specs'][main_receipt['candidate_choice']['configuration_id']]
    controls=secondary.controls_for(selected_spec)
    settings={'main_selection_plan':study['plan_ref'],'main_selection_receipt':main_ref,'controls':list(controls),
        'models':[],'model_specs':{},'source_binding':study['plan']['source_binding'],
        'bound_implementation_sha256':study['plan']['families']['candidate']['expected_implementation_sha256'],
        'validation_parents':study['parents'],'selected_candidate':main_receipt['candidate_choice'],
        'primary_strongest_baseline':main_receipt['strongest_required_baseline']}
    for control in controls:
        for lr in (0,1):
            name=f'{control}_lr{lr}';settings['models'].append(name);settings['model_specs'][name]=controls[control]
    config_ref=write_json(root,'secondary_config.json',settings)
    secondary_runs=[]
    for seed in SEEDS:
        run={'training_seed':seed,'models':{}}
        for index,name in enumerate(settings['models']):
            body=deepcopy(next(iter(study['configs']['candidate']['models'])) and
                json.loads((root/study['refs'][0]['path']).read_text())['models']['candidate_0'])
            body['parameters']=200+index
            body['ending_config_sha256']=config_ref['sha256']
            checkpoint=write_tensor(root,f'artifacts/controls/{name}_s{seed}_state.pt',{
                'model':{'weight':torch.zeros(200+index,dtype=torch.float64)},'epoch':body['best_epoch'],'seed':seed,
                'validation_nll':body['validation_nll'],'protocol_sha256':config_ref['sha256'],
                'completion_receipt_sha256':study['plan']['source_binding']['completion_receipt_sha256'],
                'boundary_manifest_sha256':study['plan']['source_binding']['boundary_dataset_manifest_sha256'],
                'implementation_sha256':settings['bound_implementation_sha256']})
            prediction=torch.load(root/body['validation_logits'],weights_only=True)
            prediction['protocol_sha256']=config_ref['sha256'];prediction['seed']=seed
            pred_ref=write_tensor(root,f'artifacts/controls/{name}_s{seed}_logits.pt',prediction)
            body['checkpoint']=checkpoint['path'];body['checkpoint_sha256']=checkpoint['sha256']
            body['validation_logits']=pred_ref['path'];body['validation_logits_sha256']=pred_ref['sha256']
            run['models'][name]=body
        secondary_runs.append(write_json(root,f'runs/controls_s{seed}.json',run))
    sec_receipt=toy_secondary_select(root,config_ref,secondary_runs)
    sec_ref=write_json(root,'secondary_selection.json',sec_receipt)
    calls=[]
    def replay_secondary(config_path, expected_sha256, *, run_receipts, failure_ledgers, project_root):
        calls.append((run_receipts,failure_ledgers))
        ref={'path':str(Path(config_path).relative_to(project_root)),'sha256':expected_sha256}
        return {'values':toy_secondary_select(Path(project_root),ref,run_receipts,failure_ledgers)}
    patches.setattr(secondary,'synthesize_secondary_selection',replay_secondary)
    protocol=json.loads((ROOT/'research/protocols/confirmatory_design_v1.json').read_text())
    protocol_ref=write_json(root,'research/protocols/confirmatory_design_v1.json',protocol)
    md_path=root/'research/protocols/confirmatory_design_v1.md'
    md_path.write_bytes((ROOT/'research/protocols/confirmatory_design_v1.md').read_bytes())
    md_ref={'path':str(md_path.relative_to(root)),'sha256':hashlib.sha256(md_path.read_bytes()).hexdigest()}
    amendment_ref=write_json(root,combined.SCOPE_AMENDMENT,json.loads((ROOT/combined.SCOPE_AMENDMENT).read_text()))
    support={}
    for name,module in (('metrics',lock.metrics),('numerical_policy',numerical)):
        path=root/'src/arggnn'/Path(module.__file__).name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(Path(module.__file__).read_bytes())
        support[name]={'path':str(path.relative_to(root)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    workflow={}
    for name in lock.WORKFLOWS:
        path=root/f'{name}.py';path.write_text('# Synthetic unit-test workflow stand-in, not production.\ndef main(records):\n    return [dict(row) for row in records]\n')
        workflow[name]={'path':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'entry_point':'main'}
    code={**support,**workflow}
    runtime={'device_kind':'cuda','physical_gpu':4,'process_title':lock.PROCESS_TITLE,'visible_gpu_uuid':'GPU-synthetic-unit-test',
        'device':{'name':'synthetic device, never queried'},'deterministic_execution_policy':deepcopy(numerical.DETERMINISTIC_POLICY),
        'software':{'torch':lock.importlib.metadata.version('torch'),'python_executable':sys.executable,
            'python':sys.version,'torch_cuda':'synthetic mock','torch_build':'synthetic unit-test mock; no device'}}
    runtime_ref=write_json(root,'runtime.json',runtime)
    analysis={'schema_version':1,'kind':'prespecified_confirmatory_analysis_contract','status':'prespecified_before_test_preparation',
        'protocol_sha256':protocol_ref['sha256'],'training_seeds':list(SEEDS),'nll_policy':RAW_NLL,
        'confirmation':protocol['confirmation'],'separate_prespecified_analyses':protocol['separate_prespecified_analyses'],
        'code_artifacts':code,'runtime_manifest':runtime_ref,'final_test_access':False}
    analysis_ref=write_json(root,'analysis_contract.json',analysis)
    evidence_ref=write_json(root,'workflow_cpu_evidence.json',{'kind':'synthetic_unit_test_evidence_only','actual_final_test_access':False})
    validation={'schema_version':1,'kind':'confirmatory_workflow_validation_receipt','status':'passed','final_test_access':False,
        'synthetic_development_validation_only':True,'code_artifacts':code,'checks':{name:True for name in lock.WORKFLOW_CHECKS},
        'evidence_artifacts':[evidence_ref]}
    validation_ref=write_json(root,'workflow_validation.json',validation)
    cohorts={}
    known={'id_confirmation':(192,160,protocol['confirmation']),
           'robustness_mother':(64,64,protocol['separate_prespecified_analyses']['cohort']),
           'demographic_ood':(64,64,{'namespace':'SYNTHETIC-OOD-REGISTRY','ancestry_mutation_master_seed':20261131,'relate_master_seed':20261132})}
    for index,(name,(count,target,spec)) in enumerate(known.items()):
        cohorts[name]={key:spec[key] for key in ('namespace','ancestry_mutation_master_seed','relate_master_seed')}
        cohorts[name].update(target_admitted_parents=target,parents=[{'parent_id':i,'ancestry_seed':100000+1000*index+i,
            'mutation_seed':200000+1000*index+i,'inference_seed':300000+1000*index+i,
            'perturbation_seeds':{key:400000+10000*index+100*i+j for j,key in enumerate(lock.PERTURBATIONS)} if name=='robustness_mother' else {}} for i in range(count)])
    registry={'schema_version':1,'kind':'prespecified_parent_seed_registry','status':'prespecified_before_test_preparation',
        'protocol_sha256':protocol_ref['sha256'],'training_seeds':list(SEEDS),'analysis_seed':20261115,'cohorts':cohorts}
    registry_ref=write_json(root,'registry.json',registry)
    source_ref=study['plan']['source_artifacts']['source_manifest'];source=json.loads((root/source_ref['path']).read_text())
    prior={'schema_version':1,'kind':'prior_parent_seed_registry','all_previous_and_development_parents_declared':True,
        'final_test_parents':0,'source_artifacts':[source_ref],
        'parents':[{'ancestry_seed':row['seeds']['ancestry'],'mutation_seed':row['seeds']['mutations'],'inference_seed':row['inference_seed']} for row in source['replicates']]}
    prior_ref=write_json(root,'prior_registry.json',prior)
    schema={'schema_version':1,'kind':'prespecified_joined_analysis_artifact_schema','status':'frozen_before_test_preparation',
        'analysis_phase':'id_primary_secondary_and_descriptive','code_artifacts':{'joined_analysis':{k:workflow['joined_analysis'][k] for k in ('path','sha256')},
          'analysis_cli':{k:workflow['prediction'][k] for k in ('path','sha256')},'metrics':support['metrics'],'dense_tract_metrics':support['metrics']},
        'roster_schema':{'synthetic_unit_test_only':True},'ledger_schema':{'synthetic_unit_test_only':True},
        'prediction_npz':{'allow_pickle':False},'primary_labels_npz':{'allow_pickle':False},'census_json':{'external_only':True},
        'remaining_coverage':['robustness','query_count','demographic_ood','illustrations']}
    schema_ref=write_json(root,'analysis_schema.json',schema)
    patches.setattr(lock,'expected_analysis_schema',lambda capture,received:deepcopy(schema))
    capture=lock._Capture(root)
    primary={'candidate':lock._bundle(capture,main_receipt['candidate_choice'],main_receipt['cells'],study['plan']['families']['candidate']['protocol']),
        'strongest_required_baseline':lock._bundle(capture,main_receipt['strongest_required_baseline'],main_receipt['cells'],study['plan']['families']['gru']['protocol'])}
    control_bundles={control:lock._bundle(capture,row['choice'],sec_receipt['cells'],config_ref,family=control) for control,row in sec_receipt['controls'].items()}
    fit_code=root/'synthetic_temperature_fitter.py';fit_code.write_text('# Synthetic unit-test source, no real fitting.\ndef fit_temperature(records):\n    return 1.0\n')
    fit_code_ref={'path':fit_code.name,'sha256':hashlib.sha256(fit_code.read_bytes()).hexdigest()}
    temperatures=[]
    for method,bundle in (primary|control_bundles).items():
        for seed in bundle['seed_bundles']:
            row={'method_id':method,'seed':seed['seed'],'temperature':1.0,'checkpoint':seed['checkpoint'],
                'development_prediction':seed['development_prediction'],'validation_parent_ids':main_receipt['ordered_parent_ids'],
                'fit_scope':'fresh_development_validation_only'}
            fit_record=write_json(root,f'fit_record_{method}_s{seed["seed"]}.json',{'scope':'SYNTHETIC UNIT TEST ONLY','temperature':1.0})
            fit={'schema_version':1,'kind':'validation_temperature_fit_receipt','status':'complete',
                **{key:row[key] for key in ('method_id','seed','temperature','checkpoint','development_prediction','fit_scope')},
                'main_selection_receipt_sha256':main_ref['sha256'],'secondary_selection_receipt_sha256':sec_ref['sha256'],
                'source_binding':study['plan']['source_binding'],'validation_parents':study['parents'],
                'unchanged_model_weights':True,'final_test_access':False,'fit_code_artifacts':[fit_code_ref],'fit_record_artifacts':[fit_record]}
            row['fit_evidence']=write_json(root,f'fit_evidence_{method}_s{seed["seed"]}.json',fit)
            temperatures.append(row)
    manifest={'schema_version':1,'kind':'prespecified_validation_temperature_manifest','status':'frozen_before_test_preparation',
        'main_selection_receipt_sha256':main_ref['sha256'],'secondary_selection_receipt_sha256':sec_ref['sha256'],
        'training_seeds':list(SEEDS),'records':temperatures,'final_test_access':False}
    temperature_ref=write_json(root,'temperature_manifest.json',manifest)
    request={'schema_version':1,'kind':'confirmatory_lock_request','status':'reviewed_prespecification_before_test_preparation',
        'test_preparation_started':False,'final_test_access':False,'workflow_code':workflow,'support_code':support,
        'protocol_json':protocol_ref,'protocol_markdown':md_ref,'scope_amendment':amendment_ref,
        'main_selection_plan':study['plan_ref'],'main_selection_receipt':main_ref,'secondary_configuration':config_ref,
        'secondary_selection_receipt':sec_ref,'runtime_manifest':runtime_ref,'analysis_contract':analysis_ref,
        'parent_seed_registry':registry_ref,'prior_parent_seed_registry':prior_ref,'workflow_validation':validation_ref,
        'temperature_manifest':temperature_ref,'analysis_schema':schema_ref}
    request_ref=write_json(root,'request.json',request)
    yield {'root':root,'study':study,'request':request,'request_ref':request_ref,'calls':calls,
           'main':main_receipt,'secondary':sec_receipt,'registry':registry,'prior':prior,'protocol':protocol,'temperatures':manifest,'schema':schema}
    patches.undo()


def bind(evidence, request=None, output=None):
    ref=evidence['request_ref'] if request is None else write_json(evidence['root'],'variant_request.json',request)
    return lock.bind_confirmatory_lock(evidence['root']/ref['path'],ref['sha256'],output_path=output,project_root=evidence['root'])


def test_full_captured_main_and_secondary_replay_without_emission(evidence):
    result=bind(evidence)
    assert result.path is None
    assert result.payload['main_matrix_counts']=={'success':240,'failed':0,'missing':0}
    assert result.payload['candidate']['family_id']=='candidate'
    assert result.payload['strongest_required_baseline']['family_id']=='gru'
    assert set(result.payload['secondary_controls'])=={'gain_matched_diagonal','static_local'}
    assert len(result.payload['candidate']['seed_bundles'])==3
    assert all(row['checkpoint'] for row in result.payload['candidate']['seed_bundles'])
    assert [row['seed'] for row in result.payload['candidate']['seed_bundles']]==list(SEEDS)
    assert result.payload['test_preparation_performed'] is False
    assert result.payload['operational_permissions_enforced'] is False
    assert evidence['calls'] and len(evidence['calls'][-1][0])==3
    assert result.payload['prespecified_registry']['cohort_attempts']=={'id_confirmation':192,'robustness_mother':64,'demographic_ood':64}
    assert not torch.cuda.is_initialized()


def test_exclusive_unit_test_lock_output(evidence):
    path=evidence['root']/'unit_test_lock_only.json'
    result=bind(evidence,output=path)
    assert path.read_bytes()==result.content
    assert hashlib.sha256(result.content).hexdigest()==result.sha256
    with pytest.raises(ValueError,match='Fresh exclusive'):
        bind(evidence,output=path)


@pytest.mark.parametrize('key,value', [('status','hypothetical_proposal'),('test_preparation_started',True),('final_test_access',True)])
def test_proposal_or_post_access_request_cannot_emit(evidence,key,value):
    request=deepcopy(evidence['request']);request[key]=value
    with pytest.raises(ValueError,match='Reviewed pre-test'):
        bind(evidence,request)


def test_forged_main_winner_rejected_by_replay(evidence):
    receipt=deepcopy(evidence['main']);receipt['candidate_choice']['configuration_id']='candidate_7'
    ref=write_json(evidence['root'],'forged_main_selection.json',receipt)
    request=deepcopy(evidence['request']);request['main_selection_receipt']=ref
    with pytest.raises(ValueError,match='does not replay'):
        bind(evidence,request)


def test_incomplete_main_selection_rejected(evidence):
    receipt=deepcopy(evidence['main']);receipt['matrix_resolved']=False
    request=deepcopy(evidence['request']);request['main_selection_receipt']=write_json(evidence['root'],'incomplete_main.json',receipt)
    with pytest.raises(ValueError,match='resolved main'):
        bind(evidence,request)


def test_changed_actual_checkpoint_bytes_rejected(evidence):
    row=evidence['main']['cells'][0];path=evidence['root']/row['checkpoint'];before=path.read_bytes()
    try:
        path.write_bytes(before+b'corruption')
        with pytest.raises(ValueError,match='hash mismatch'):
            bind(evidence)
    finally:
        path.write_bytes(before)


def test_missing_workflow_implementation_rejected(evidence):
    request=deepcopy(evidence['request']);del request['workflow_code']['custodian']
    with pytest.raises(ValueError,match='mandatory'):
        bind(evidence,request)


def test_metric_helper_cannot_substitute_for_joined_analysis(evidence):
    request=deepcopy(evidence['request']);request['workflow_code']['joined_analysis']=request['support_code']['metrics']|{'entry_point':'paired_parent_report'}
    with pytest.raises(ValueError,match='not workflow implementations'):
        bind(evidence,request)


def test_placeholder_workflow_is_rejected(evidence):
    path=evidence['root']/'placeholder.py';path.write_text('def main():\n    raise NotImplementedError\n')
    request=deepcopy(evidence['request']);request['workflow_code']['prediction']={'path':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'entry_point':'main'}
    with pytest.raises(ValueError,match='Placeholder workflow'):
        bind(evidence,request)


def test_cpu_runtime_cannot_be_claimed_as_cuda_evidence(evidence):
    request=deepcopy(evidence['request']);runtime=json.loads((evidence['root']/request['runtime_manifest']['path']).read_text());runtime['device_kind']='cpu'
    request['runtime_manifest']=write_json(evidence['root'],'cpu_runtime.json',runtime)
    with pytest.raises(ValueError,match='CUDA runtime'):
        bind(evidence,request)


def test_analysis_contract_change_rejected(evidence):
    request=deepcopy(evidence['request']);analysis=json.loads((evidence['root']/request['analysis_contract']['path']).read_text());analysis['confirmation']['primary']['practical_difference_nats']=.005
    request['analysis_contract']=write_json(evidence['root'],'changed_analysis.json',analysis)
    with pytest.raises(ValueError,match='prespecification differs'):
        bind(evidence,request)


def test_secondary_summary_with_forged_choice_cannot_lock(evidence):
    request=deepcopy(evidence['request']);receipt=deepcopy(evidence['secondary']);receipt['controls']['gain_matched_diagonal']['choice']['parameters']+=1
    request['secondary_selection_receipt']=write_json(evidence['root'],'changed_secondary.json',receipt)
    with pytest.raises(ValueError,match='does not replay from actual'):
        bind(evidence,request)


def test_missing_secondary_choice_rejected(evidence):
    request=deepcopy(evidence['request']);receipt=deepcopy(evidence['secondary']);receipt['controls']['static_local']['choice']=None
    request['secondary_selection_receipt']=write_json(evidence['root'],'incomplete_secondary.json',receipt)
    with pytest.raises(ValueError,match='resolved eligible choice'):
        bind(evidence,request)


@pytest.mark.parametrize('mutation,match', [
    ('seed_reuse','reuses a prior'),('missing_parent','attempt registry'),('wrong_training_seed','registry binding'),
    ('missing_draw','query-draw seeds'),('wrong_master','master seeds')])
def test_prespecified_registry_rejections(evidence,mutation,match):
    registry=deepcopy(evidence['registry'])
    if mutation=='seed_reuse':registry['cohorts']['id_confirmation']['parents'][0]['ancestry_seed']=10000
    elif mutation=='missing_parent':registry['cohorts']['id_confirmation']['parents'].pop()
    elif mutation=='wrong_training_seed':registry['training_seeds']=[101,202,404]
    elif mutation=='missing_draw':del registry['cohorts']['robustness_mother']['parents'][0]['perturbation_seeds']['query_draw3']
    elif mutation=='wrong_master':registry['cohorts']['id_confirmation']['relate_master_seed']+=1
    with pytest.raises(ValueError,match=match):
        lock.validate_registries(registry,evidence['prior'],evidence['protocol'],evidence['request']['protocol_json']['sha256'])


def test_smoother_bundle_has_three_predictions_and_no_fabricated_checkpoint(evidence):
    capture=lock._Capture(evidence['root']);choice=evidence['main']['families']['reference_smoother']['winner']
    result=lock._bundle(capture,choice,evidence['main']['cells'],evidence['study']['plan']['families']['reference_smoother']['protocol'],smoother=True)
    assert len(result['seed_bundles'])==3 and all(row['checkpoint'] is None for row in result['seed_bundles'])
    assert all(row['development_prediction']['sha256'] for row in result['seed_bundles'])


def test_unique_secondary_run_provenance_required(evidence):
    capture=lock._Capture(evidence['root'])
    original=evidence['secondary']['captured_artifacts'][0]
    alias=evidence['root']/'ambiguous_alias.json';alias.write_bytes((evidence['root']/original['path']).read_bytes())
    one=lock._actual_evidence_refs(capture,[original,original|{'path':alias.name}])
    with pytest.raises(ValueError,match='Unique actual'):
        one(original['sha256'])


@pytest.mark.parametrize('relative',['data/test/manifest.json','data/confirmatory_data/manifest.json','data/final_test_labels.json','../outside.json'])
def test_heldout_and_traversal_paths_rejected_before_open(evidence,relative):
    with pytest.raises(ValueError):
        lock.guard_lock_path(relative,project_root=evidence['root'],must_exist=False)


def test_imported_source_drift_is_rejected(monkeypatch,evidence):
    monkeypatch.setattr(lock,'_IMPORTED_PINS',{**lock._IMPORTED_PINS,str(Path(lock.__file__).resolve()):'0'*64})
    with pytest.raises(ValueError,match='implementation drift'):
        bind(evidence)


def test_stat_drift_during_capture_rejected(evidence):
    capture=lock._Capture(evidence['root']);ref=write_json(evidence['root'],'capture_race.json',{'before':True});capture.read(ref)
    (evidence['root']/ref['path']).write_text('changed')
    with pytest.raises(ValueError,match='changed before'):
        capture.verify_stable()


def test_complete_producer_contract_admission_is_mandatory(monkeypatch,evidence):
    def rejected(*args,**kwargs):
        raise ValueError('candidate producer rejects malformed dimensions or held-out path')
    monkeypatch.setattr(combined,'bind_development_selection',rejected)
    with pytest.raises(ValueError,match='producer rejects'):
        bind(evidence)


def test_rebound_producer_plan_must_match_captured_plan(monkeypatch,evidence):
    value=deepcopy(evidence['study']['plan']);value['tie_tolerance']=.002
    monkeypatch.setattr(combined,'bind_development_selection',lambda *args,**kwargs:SimpleNamespace(plan=value))
    with pytest.raises(ValueError,match='producer-contract admission'):
        bind(evidence)


@pytest.mark.parametrize('key,value',[('warn_only',0),('cublas_workspace_config',':16:8'),('deterministic_algorithms',False)])
def test_typed_strict_policy_rejects_alias_and_drift(key,value):
    policy=deepcopy(numerical.DETERMINISTIC_POLICY);policy[key]=value
    with pytest.raises(ValueError,match='typed strict'):
        lock._same_policy(policy)


@pytest.mark.parametrize('spec',[{'geometry':'age','kernel_depth':1,'coupling':'full','rate_parameterization':'log'},
                                  {'geometry':'plain','kernel_depth':2,'coupling':'full','rate_parameterization':'log'}])
def test_eligible_third_control_cannot_be_omitted(evidence,spec):
    capture=lock._Capture(evidence['root'])
    with pytest.raises(ValueError,match='Every conditional secondary control'):
        lock.replay_secondary(capture,evidence['request']['secondary_configuration'],evidence['request']['secondary_selection_receipt'],
                              evidence['request']['main_selection_plan'],evidence['request']['main_selection_receipt'],spec)


def test_boolean_schema_alias_is_rejected(evidence):
    registry=deepcopy(evidence['registry']);registry['schema_version']=True
    with pytest.raises(ValueError,match='prespecified parent/seed registry'):
        lock.validate_registries(registry,evidence['prior'],evidence['protocol'],evidence['request']['protocol_json']['sha256'])


def test_selected_bundle_cannot_omit_a_training_seed(evidence):
    capture=lock._Capture(evidence['root']);choice=evidence['main']['candidate_choice']
    cells=[row for row in evidence['main']['cells'] if not(row['family_id']=='candidate' and row['configuration_id']==choice['configuration_id'] and row['seed']==303)]
    with pytest.raises(ValueError,match='all three successful fixed seeds'):
        lock._bundle(capture,choice,cells,evidence['study']['plan']['families']['candidate']['protocol'])


def test_primary_factory_capacity_check_is_mandatory(monkeypatch,evidence):
    def wrong_capacity(*args):
        model=torch.nn.Module();model.register_parameter('weight',torch.nn.Parameter(torch.zeros(1,dtype=torch.float64)));return model
    monkeypatch.setattr(lock,'_primary_model',wrong_capacity)
    with pytest.raises(ValueError,match='capacity differs'):
        bind(evidence)


def test_primary_checkpoint_schema_check_is_mandatory(monkeypatch,evidence):
    def wrong_schema(family,name,settings):
        index=list(evidence['study']['configs']).index(family);offset=settings['models'].index(name)
        model=torch.nn.Module();model.register_parameter('weight',torch.nn.Parameter(torch.zeros((1,100+100*index+offset),dtype=torch.float64)));return model
    monkeypatch.setattr(lock,'_primary_model',wrong_schema)
    with pytest.raises(ValueError,match='keys/shapes/dtypes'):
        bind(evidence)


def test_primary_replay_requires_exact_selected_and_reloaded_score(evidence):
    capture=lock._Capture(evidence['root']);choice=evidence['main']['candidate_choice']
    bundle=lock._bundle(capture,choice,evidence['main']['cells'],evidence['study']['plan']['families']['candidate']['protocol'])
    one=lock._actual_evidence_refs(capture,evidence['main']['source_artifacts']);ref=one(bundle['seed_bundles'][0]['run_receipt_sha256'])
    body=capture.read(ref)['models'][choice['configuration_id']];before=body['reloaded_validation_nll']
    try:
        body['reloaded_validation_nll']=before+.001
        with pytest.raises(ValueError,match='Exact zero primary'):
            lock.validate_primary_bundle(capture,bundle,evidence['main'])
    finally:body['reloaded_validation_nll']=before


def test_frozen_temperature_missing_method_seed_rejected(evidence):
    request=deepcopy(evidence['request']);manifest=deepcopy(evidence['temperatures']);manifest['records'].pop()
    request['temperature_manifest']=write_json(evidence['root'],'missing_temperature.json',manifest)
    with pytest.raises(ValueError,match='Every selected method'):
        bind(evidence,request)


@pytest.mark.parametrize('value',[0.,-1.,True])
def test_temperature_must_be_positive_finite_scalar(evidence,value):
    request=deepcopy(evidence['request']);manifest=deepcopy(evidence['temperatures']);manifest['records'][0]['temperature']=value
    request['temperature_manifest']=write_json(evidence['root'],'invalid_temperature.json',manifest)
    with pytest.raises(ValueError,match='positive finite validation temperature'):
        bind(evidence,request)


def test_temperature_cannot_fit_another_parent_or_checkpoint(evidence):
    request=deepcopy(evidence['request']);manifest=deepcopy(evidence['temperatures']);manifest['records'][0]['validation_parent_ids'][0]='unfrozen-parent'
    request['temperature_manifest']=write_json(evidence['root'],'wrong_temperature_parents.json',manifest)
    with pytest.raises(ValueError,match='all sixteen VAL parents'):
        bind(evidence,request)


def test_analysis_schema_must_replay_actual_source_helper(evidence):
    request=deepcopy(evidence['request']);schema=deepcopy(evidence['schema']);schema['prediction_npz']['allow_pickle']=True
    request['analysis_schema']=write_json(evidence['root'],'changed_schema.json',schema)
    with pytest.raises(ValueError,match='actual public metadata helper'):
        bind(evidence,request)


def test_published_lock_validator_before_any_target_reads(evidence):
    path=evidence['root']/'unit_test_published_id_lock.json';result=bind(evidence,output=path)
    verified=lock.validate_confirmatory_lock(path,result.sha256,project_root=evidence['root'])
    assert verified.payload==result.payload and verified.content==result.content
    assert verified.check()==result.sha256
    assert verified.payload['lock_execution_scope']=='id_confirmation_only'
    assert verified.payload['analysis_coverage']['remaining_coverage']==evidence['schema']['remaining_coverage']
    assert len(verified.payload['validation_temperatures'])==12
    assert not torch.cuda.is_initialized()
    verified.payload['training_seeds'].append(404)
    with pytest.raises(ValueError,match='In-memory published lock'):
        verified.check()


def test_published_lock_detects_changed_captured_artifacts(evidence):
    path=evidence['root']/'unit_test_published_drift_lock.json';result=bind(evidence,output=path)
    code=evidence['root']/evidence['request']['workflow_code']['custodian']['path'];before=code.read_bytes()
    try:
        code.write_bytes(before+b'# changed\n')
        with pytest.raises(ValueError,match='hash mismatch'):
            lock.validate_confirmatory_lock(path,result.sha256,project_root=evidence['root'])
    finally:code.write_bytes(before)


def test_published_lock_rejects_analysis_library_version_drift(monkeypatch,evidence):
    path=evidence['root']/'unit_test_published_library_lock.json';result=bind(evidence,output=path)
    version=lock.importlib.metadata.version
    monkeypatch.setattr(lock.importlib.metadata,'version',lambda name:'changed-synthetic-version' if name=='numpy' else version(name))
    with pytest.raises(ValueError,match='analysis environment versions differ'):
        lock.validate_confirmatory_lock(path,result.sha256,project_root=evidence['root'])
