"""One public selector consumes six actual tiny CPU fit cells under synthetic facts.
Four full matrices of hand-made fixture artifacts only scaffold the other families.
No producer admission is certified. No actual cohort data or CUDA are accessed.
"""
from pathlib import Path
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import hashlib,json,sys,torch
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
OUT=ROOT/'research/reviews/development_producer_audit_v1'
TEMP=ROOT/'.runtime/tmp/development_producer_audit_v1/cross_producer_v2'
TEMP.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(ROOT/'tests'))
from test_development_selection import study,write_json,write_tensor
from test_argformer_control import example_sequence
from test_anchor_temporal_baselines import seq
from test_allrecord_boundary_temporal import bank
from test_canonical_gru import genuine_sequence
from arggnn.development_selection import synthesize_development_selection,SEEDS,RAW_NLL
from arggnn.development_execution import configure_deterministic_development,DETERMINISTIC_POLICY
from arggnn import argmix_development_bindings as rb,canonical_development_bindings as cb,priorart_development_bindings as pb,allrecord_development_bindings as ab
from arggnn import train_argmix_development as rr,train_canonical_development as cr,train_priorart_development as pr,train_allrecord_development as ar

torch.set_num_threads(1);configure_deterministic_development()
fixture=study.__wrapped__(TEMP);plan=fixture['plan'];old=fixture['configs']['gru']
for name in ('canonical','argmix'):
    plan['families'].pop(name)
plan['required_baseline_families']=[f for f in plan['required_baseline_families'] if f not in ('canonical','argmix')]
refs=[r for r in fixture['refs'] if r['family_id'] not in ('canonical','argmix')]
positions=torch.tensor([.25,.75],dtype=torch.float64);queries=torch.tensor([10,20],dtype=torch.long)
checks={}
def actual_sequence(value):
    intervals=value.intervals
    indexes=((positions[:,None]>=intervals[:,0])&(positions[:,None]<intervals[:,1])).long().argmax(1)
    identities=value.node_ids.clone()
    for original,replacement in zip(value.query_node_ids.tolist(),queries.tolist()):
        identities[value.node_ids==original]=replacement
    return replace(value,positions=positions,target_graph=indexes,query_node_ids=queries,node_ids=identities,x=value.x.float(),clades=value.clades.float())
def raw_parents(value):
    training=rr.TreeTrainingParent(value,rr.observed_fixed_tiles(value,8),torch.zeros(2,2,dtype=torch.long),'synthetic_train',{})
    validation=[rr.TreeTrainingParent(value,rr.observed_fixed_tiles(value,8),torch.full((2,2),3,dtype=torch.long),p['parent_id'],{}) for p in plan['validation_parents']]
    return {'train':[training],'val':validation}
def add_implementation(pins):
    for name,digest in pins.items():
        source=ROOT/name if '/' in name else ROOT/'src/arggnn'/name
        assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
        relative='implementation/src/arggnn/'+source.name if str(source.parent)==str(ROOT/'src/arggnn') else 'implementation/scripts/'+source.name
        path=TEMP/relative;path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():assert path.read_bytes()==source.read_bytes()
        else:path.write_bytes(source.read_bytes())
        plan['implementation_artifacts'][name]={'path':relative,'sha256':digest}
def protocol(family,pins,grid):
    values={k:deepcopy(v) for k,v in old.items() if k not in ('models','model_specs','model_learning_rates')}
    values.update(family=family,bound_implementation_sha256=pins,num_layers=2,temporal_layers=2,
        reference_adapter=rb.ADAPTER,gradient_clip_norm=1.,input_dim=48,num_classes=4)
    if family=='argmix_full_tree':scope,aggregation='all raw observed marginal-tree nodes/topology and F48, original F8 prefix with verified full-parent reference adapter; no canonicalization or truth features','none'
    elif family==cb.FAMILY:scope,aggregation=cb.INPUT_SCOPE,cb.AGGREGATION
    elif family in pb.FAMILIES:scope,aggregation=pb.SCOPE[family],pb.AGGREGATION[family]
    else:scope,aggregation=ab.INPUT_SCOPE,ab.AGGREGATION[family]
    values.update(input_scope=scope,cross_tree_aggregation=aggregation,selection=rb.SELECTION)
    if family in (cb.FAMILY,*pb.FAMILIES):values['deterministic_policy']=deepcopy(DETERMINISTIC_POLICY)
    else:values['deterministic_execution_policy']=deepcopy(DETERMINISTIC_POLICY)
    if family in ab.FAMILIES:values.update(deepcopy(grid),max_attention_score_bytes=268435456,input_dim=47)
    else:values['configurations']=grid
    ref=write_json(TEMP,'configs/'+family+'.json',values)
    plan['families'][family]={'role':family,'grid_kind':'width_dropout_lr','protocol':ref,'expected_implementation_sha256':pins,'input_scope':scope,'cross_tree_aggregation':aggregation}
    plan['required_baseline_families'].append(family);add_implementation(pins)
    return values,ref
for family in ('argmix_full_tree',cb.FAMILY,*pb.FAMILIES,*ab.FAMILIES):
    configure_deterministic_development()
    pins=rb.implementation_hashes() if family=='argmix_full_tree' else cb.implementation_hashes() if family==cb.FAMILY else pb.implementation_hashes() if family in pb.FAMILIES else ab.implementation_hashes()
    grid=rb.grid() if family=='argmix_full_tree' else cb.grid() if family==cb.FAMILY else pb.grid(family) if family in pb.FAMILIES else ab.family_grid(family)
    settings,config_ref=protocol(family,pins,grid)
    binding={'protocol_sha256':config_ref['sha256'],'implementation_sha256':pins,**plan['source_binding']}
    if family in (cb.FAMILY,*pb.FAMILIES):binding['deterministic_policy']=deepcopy(DETERMINISTIC_POLICY)
    path=TEMP/'actual'/family;path.mkdir(parents=True);checkpoint=path/'selected.pt'
    if family in ab.FAMILIES:
        value=bank(actual_sequence(seq()))
        tiles=lambda:SimpleNamespace(bank=value,parent_position_indices=torch.arange(2),labels=torch.zeros(2,2,dtype=torch.long))
        def case(pid,labels):
            tile=tiles();tile.labels=labels
            return SimpleNamespace(parent_id=pid,metadata={'replicate':0},positions=positions,query_node_ids=queries,labels=labels,tiles=[tile])
        training=case('synthetic_train',torch.zeros(2,2,dtype=torch.long))
        validation=[case(p['parent_id'],torch.full((2,2),3,dtype=torch.long)) for p in plan['validation_parents']]
        cases={'train':[training],'val':validation};banks={id(tile):tile.bank for group in cases.values() for parent in group for tile in parent.tiles}
        name=settings['models'][0];model=ar.factory(name,settings)
        result,logits=ar.train_configuration(model,settings,name,101,cases,banks,checkpoint,binding,integrity_check=lambda **kw:None)
    elif family==cb.FAMILY:
        value=actual_sequence(genuine_sequence(dtype=torch.float32))
        prepared=cr.prepare_observed_parent(value,parent_id='synthetic',split='train',source_tensor_sha256='a'*64,source_prepared_sha256='b'*64)
        training=cr.join_external_targets(prepared,torch.zeros(2,2,dtype=torch.long),'synthetic_train',{},8)
        validation=[cr.join_external_targets(prepared,torch.full((2,2),3,dtype=torch.long),p['parent_id'],{},8) for p in plan['validation_parents']]
        parents={'train':[training],'val':validation};configuration=grid[0];name=configuration['id']
        result,logits=cr.train_configuration(configuration,settings,parents,torch.device('cpu'),101,checkpoint,binding)
    else:
        parents=raw_parents(actual_sequence(example_sequence() if family=='argmix_full_tree' else seq()))
        configuration=grid[0];name=configuration['id']
        if family=='argmix_full_tree':result,logits=rr.train_configuration(configuration,settings,parents,torch.device('cpu'),101,checkpoint,binding)
        else:result,logits=pr.train_configuration(configuration,settings,parents,torch.device('cpu'),101,checkpoint,binding,integrity_check=lambda **kw:None)
    prediction=write_tensor(TEMP,'actual/'+family+'/logits.pt',{'logits':logits,'positions':[positions for _ in range(16)],
        'query_node_ids':[queries for _ in range(16)],'parent_ids':[p['parent_id'] for p in plan['validation_parents']],
        'seed':101,'nll_policy':RAW_NLL,**binding})
    result.update(validation_logits=prediction['path'],validation_logits_sha256=prediction['sha256'],ending_config_sha256=config_ref['sha256'],ending_implementation_sha256=pins)
    run={'family':family,'development_namespace':'development-v1','evaluation_split':'fresh_development_validation_only',
        'training_seeds':list(SEEDS),'training_seed':101,'nll_policy':RAW_NLL,**binding,'initial_implementation_sha256':pins,'models':{name:result}}
    ref={'family_id':family,**write_json(TEMP,'runs/'+family+'_actual101.json',run)};refs.append(ref)
    checks[family]={'actual_neural_CPU_fit':True,'configuration_id':name,'parameters':result['parameters'],'history_epochs':len(result['history']),
        'best_epoch':result['best_epoch'],'checkpoint_replay_nll_delta':result['checkpoint_replay_nll_delta'],'protocol_sha256':config_ref['sha256'],'receipt':ref}
    (OUT/'cross_producer_progress.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(family+': '+json.dumps(checks[family],sort_keys=True),flush=True)
plan_ref=write_json(TEMP,'actual_producer_selection_plan.json',plan)
receipt=synthesize_development_selection(plan_ref['path'],plan_ref['sha256'],run_receipts=refs,project_root=TEMP,output_path='public_selector_receipt.json')
summary={'kind':'six_actual_tiny_CPU_producer_cells_with_synthetic_source_facts','plan_sha256':plan_ref['sha256'],'selector_receipt_sha256':receipt.sha256,
    'selector_receipt_path':receipt.path,'actual_cells':checks,'matrix_counts':receipt.payload['matrix_counts'],
    'matrix_resolved':receipt.payload['matrix_resolved'],'baseline_selection_eligible':receipt.payload['baseline_selection_eligible'],
    'actual_baseline_families':[f for f in checks],
    'scaffolding':'96 hand-made fixture success cells for candidate/GRU/Transformer/smoother; they are not actual trained producer outputs',
    'source_admission':'all completion/source manifest facts are hand-built; actual producer source admission not certified by this replay',
    'promotes_final_test_phase':receipt.payload['promotes_final_test_phase'],'final_test_access':receipt.payload['final_test_access'],
    'cuda_initialized':torch.cuda.is_initialized()}
assert receipt.payload['matrix_counts']=={'success':102,'failed':0,'missing':138}
assert receipt.payload['baseline_selection_eligible'] is False and receipt.payload['strongest_required_baseline'] is None
assert receipt.payload['promotes_final_test_phase'] is False and not torch.cuda.is_initialized()
(OUT/'cross_producer.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
print('PUBLIC_SELECTOR: '+json.dumps({k:summary[k] for k in ('matrix_counts','matrix_resolved','baseline_selection_eligible','promotes_final_test_phase','cuda_initialized')},sort_keys=True))
