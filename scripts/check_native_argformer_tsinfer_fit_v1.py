"""Author verification: handmade fixtures and source ASTs only, never actual data."""
from __future__ import annotations
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
import types
from unittest.mock import patch

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
SOURCE=ROOT/'src/arggnn/native_argformer_tsinfer_fit_v1.py'
REVIEW=ROOT/'research/reviews/native_argformer_tsinfer_fit_v1_author'
sys.dont_write_bytecode=True
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='4'
spec=importlib.util.spec_from_file_location('_source_only_native_fit_author',SOURCE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
c=m.policy();recipe=m.verify_source(c)
checks=[]
test_only_dependency=None
if importlib.util.find_spec('threadpoolctl') is None:
    dep=ROOT/'.runtime/env/lib/python3.11/site-packages/threadpoolctl.py'
    ds=importlib.util.spec_from_file_location('threadpoolctl',dep)
    dm=importlib.util.module_from_spec(ds);sys.modules['threadpoolctl']=dm;ds.loader.exec_module(dm)
    test_only_dependency=m.reference(dep)
def check(name,condition):
    if not condition:raise AssertionError(name)
    checks.append(name)
def reject(name,call):
    try:call()
    except (ValueError,RuntimeError,KeyError,TypeError,FileNotFoundError):checks.append(name);return
    raise AssertionError('unexpected acceptance: '+name)
def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n');return m.reference(path)


def fixture(directory,np,torch):
    parent=c['parent_ids'][0];base=directory/'input'/parent;prepared_dir=base/'prepared';prepared_dir.mkdir(parents=True)
    nodes=[7+3*i for i in range(280)]
    roster=[{'sample_index':i,'inferred_tree_node_id':node,'observed_node_id':node,
             'source_observed_node_id':i+2000,'role':('pure_source_reference' if i<96 else 'mixed_reference_only' if i<160 else 'query'),
             'observed_reference_class':i//24 if i<96 else None} for i,node in enumerate(nodes)]
    paths=tuple((node,10000+j) for node in nodes for j in range(48))
    occurrences=tuple((i*48+j,node,float(j*100),float((j+1)*100)) for i,node in enumerate(nodes) for j in range(48))
    namespace=parent+':'+('6'*64)
    coordinate={'retained_fixed100bp_indexes':list(range(48)),'coverage_original_bp0':[0,4800],
       'local_to_original_offset_bp':0,'normalization_length_bp':2000000,'original_parent_interval_bp0':[0,2000000],
       'fixed_grid':{'total':20000,'center_formula_bp0':'50 + 100*grid_index'}}
    split_ids=list(range(len(paths)));cut=int(.9*len(paths));train=split_ids[:cut];val=split_ids[cut:]
    # A deterministic fixture-only split with source-sized pure-reference train.
    import random
    random.Random(42).shuffle(split_ids);train=split_ids[:cut];val=split_ids[cut:]
    refset=set(nodes[:96]);ptrain=[i for i in train if paths[i][0] in refset];pval=[i for i in val if paths[i][0] in refset]
    check('handmade_full_reference_train_supports_native4096',len(ptrain)>=4096)
    with (prepared_dir/'corpus.pkl').open('wb') as stream:pickle.dump({'paths':paths,'occurrences':occurrences,'namespace':namespace},stream,protocol=4)
    vocab={'special':{'<PAD>':0,'<CLS>':1,'<SEP>':2,'<MASK>':3},'node_id_tokens':{str(n):i+4 for i,n in enumerate(sorted(set(x for p in paths for x in p)))}}
    for task,split,ids in [('mlm','train',train),('mlm','val',val),('contrastive','train',ptrain),('contrastive','val',pval),('embedding','all',list(range(len(paths))))]:
        p=prepared_dir/task/split;p.mkdir(parents=True)
        write_json(p/'path_ids.json',ids)
        with (p/'raw_all_sequences_and_vocab.pkl').open('wb') as stream:pickle.dump({'raw_sequences':[list(paths[i]) for i in ids],'vocab':vocab},stream,protocol=4)
    positions=np.arange(50,4800,100,dtype=np.float64)
    joins={'grid_indexes':np.arange(48,dtype=np.int64),'original_positions_bp':positions,'local_positions_bp':positions,
       'query_nodes':np.asarray(nodes[160:],dtype=np.int64),'reference_nodes':np.asarray(nodes[:96],dtype=np.int64),
       'reference_classes':np.asarray([i//24 for i in range(96)],dtype=np.int64),
       'query_path_ids':np.asarray([[i*48+j for i in range(160,280)] for j in range(48)],dtype=np.int64),
       'reference_path_ids':np.asarray([[i*48+j for i in range(96)] for j in range(48)],dtype=np.int64)}
    np.savez_compressed(prepared_dir/'coordinate_path_joins.npz',**joins)
    write_json(prepared_dir/'observed_roster.json',roster)
    counts={'unique_full_paths':len(paths),'MLM_train':len(train),'MLM_val':len(val),'pure_reference_train':len(ptrain),
       'pure_reference_val':len(pval),'query_full_paths_scored':5760}
    budget={'actual_counts':counts,'MLM_optimizer_updates':5*(len(train)//512),'contrastive_optimizer_updates':len(ptrain)//4096,
      'pure_reference_corpus_paths_all_train_plus_val':4608,'exhaustive_query_corpus_pairs':5760*4608}
    prepared={'schema':'official_argformer_prepared_observed_arg_v1','status':'prepared_observed_paths_only','parent_id':parent,
      'policy_sha256':c['native_recipe']['sha256'],'namespace':namespace,'inferred_tree_sha256':'6'*64,
      'true_query_ancestry_read':False,'within_ARG_transductive':True,'class_order':['ANA','CHG','WHG','EHG'],
      'unique_full_paths':len(paths),'vocab_size':len(vocab['node_id_tokens'])+4,'coordinate_contract':coordinate,
      'path_counts':{'mlm':{'train':len(train),'val':len(val)},'contrastive':{'train':len(ptrain),'val':len(pval)}},
      'files_sha256':{str(p.relative_to(prepared_dir)):m.sha(p) for p in prepared_dir.rglob('*') if p.is_file()}}
    pref=write_json(prepared_dir/'prepared_manifest.json',prepared)
    q={'status':'qualified_full_native_preparation','native_original_node_IDs_used':True,'query_truth_used':False,
       'model_fit_or_retrieval_executed':False,'full_readout_occurrence_partitions_validated':True,'budget':budget,
       'source_node_vocabulary_size':prepared['vocab_size']}
    qref=write_json(base/'corpus_qualification.json',q)
    comp={'status':'original_observed_tree_loaded_without_conversion','native_IDs_preserved':True,
      'conversion_upgrade_fallback':False,'input_tree':{'path':str(directory/'never_opened_observed.trees'),'sha256':'6'*64}}
    cref=write_json(base/'input_tree_compatibility.json',comp)
    d={'kind':'official_argformer_tsinfer_parent_v1_receipt','status':'qualified_full_native_preparation','parent_id':parent,
       'condition':c['condition'],'implementation':c['prepared_producer']['implementation'],'config':c['prepared_producer']['config'],
       'bridge':{'config':c['native_recipe']},'actual_native_preparation_completed':True,
       'model_fit_or_retrieval_executed':False,'query_truth_used':False,'scientific_fallback':False,'exit_code':0,
       'namespace':namespace,'prepared_manifest':pref,'corpus_qualification':qref,'input_tree_compatibility':cref}
    dref=write_json(base/'dispatcher_receipt.json',d)
    fc=copy.deepcopy(c);fc['prepared_root']=str(directory/'input')
    fc['prepared_dispatchers'][parent]=dref;fc['qualified_corpus_metadata'][parent]=qref
    data=m.prepared_metadata(fc,parent,dref)
    check('handmade_complete_scalar_preparation_dispatch_chain',data['prepared']==prepared)
    return fc,data,joins,roster,prepared_dir,np,torch


def verify_source_wiring(directory,fc,data,torch):
    """Dependency stubs exercise unchanged execute_native ordering and IO only.

    These are handmade CPU placeholders. No released model is instantiated and no
    scientific objective is executed. Their receipts never enter actual scopes.
    """
    ns=m.compile_fit(c);events=[]
    original_source_model=ns['source_model']
    class Config:
        num_labels=3
        def to_dict(self):return {'fixture_placeholder':True,'num_labels':3}
    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__();self.bert=torch.nn.Linear(1,1,bias=False);self.classifier=torch.nn.Linear(1,3,bias=False);self.config=Config()
    class MLM(torch.nn.Module):
        def __init__(self):super().__init__();self.model=Backbone();self.tokenizer='fixture_only'
    class Contrast(torch.nn.Module):
        def __init__(self,checkpoint):
            super().__init__();self.base=torch.nn.Module();self.base.backbone=Backbone();self.base.class_weights=None
            payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
            mapping={k.removeprefix('model.'):v for k,v in payload['state']['model'].items()}
            self.base.backbone.load_state_dict(mapping,strict=True)
    main=types.ModuleType('main');contrast=types.ModuleType('sequence_contrastive')
    def mlm_factory(cfg):events.append(('construct_mlm',cfg.pretrained_model_name));return MLM()
    def contrast_factory(cfg):events.append(('construct_contrastive',cfg.pretrained_checkpoint));return Contrast(cfg.pretrained_checkpoint)
    def loader(**kwargs):
        events.append(('mlm_loader',kwargs['device_batch_size'],kwargs.get('device_microbatch_size')))
        return types.SimpleNamespace(fixture=True)
    def cloader(cfg,device_batch_size):events.append(('contrastive_loader',device_batch_size));return types.SimpleNamespace(fixture=True)
    main.build_model=mlm_factory;main.build_dataloader=loader
    contrast.build_contrastive_model=contrast_factory;contrast.build_contrastive_dataloader=cloader
    repro=types.ModuleType('composer.utils.reproducibility');repro.seed_all=lambda seed:events.append(('seed',seed))
    composer_utils=types.ModuleType('composer.utils');composer_utils.reproducibility=repro
    def state_manifest(model):
        h=hashlib.sha256()
        for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().numpy().tobytes())
        return {'sha256':h.hexdigest()}
    helper_base=directory/'fixture_base.json';helper_base.write_text('{}')
    helpers=types.SimpleNamespace(BASE=helper_base,BASE_SHA256=m.sha(helper_base),state_manifest=state_manifest,
        validate_source=lambda:events.append(('recheck_author_source',)))
    output=directory/'handmade_native_wiring';output.mkdir()
    ns['policy']=lambda expected:recipe
    ns['prepared_inputs']=lambda *args:data['prepared']
    ns['native_runtime']=lambda *args:({'lease':{},'fixture_dependency_only':True},None)
    ns['assert_backbone']=lambda model,mode,helpers:events.append(('assert_backbone',mode))
    def train(mode,native,loader,evaluation,module,cfg,path,helpers,ids,execution='full_batch'):
        events.append(('train',mode,str(cfg.max_duration),int(cfg.seed),int(cfg.device_train_microbatch_size),execution,len(ids)))
        path.mkdir();initial=state_manifest(native)['sha256']
        backbone=native.model if mode=='mlm' else native.base.backbone
        with torch.no_grad():backbone.bert.weight.add_(1)
        checkpoint=path/'composer_final.pt'
        torch.save({'state':{'model':{'model.'+k:v.detach().cpu() for k,v in backbone.state_dict().items()}}},checkpoint)
        return checkpoint,{'mode':mode,'execution':execution,'completed_epochs':5 if mode=='mlm' else 1,
          'actual_optimizer_updates':data['budget']['MLM_optimizer_updates'] if mode=='mlm' else data['budget']['contrastive_optimizer_updates'],
          'initial_model_state_sha256':initial,'final_model_state_sha256':state_manifest(native)['sha256'],
          'backbone_unchanged':True,'true_query_ancestry_read':False}
    ns['train_phase']=train
    ns['readout']=lambda native,*args:events.append(('export_native_mean_head_after_contrastive',)) or {'fixture_dependency_only':True}
    guard=types.SimpleNamespace(validate_lease=lambda:{})
    args=types.SimpleNamespace(action='fit',prepared_dir=str(data['dispatcher']['prepared_manifest']['path']).removesuffix('/prepared_manifest.json'),
      expected_prepared_sha256=data['dispatcher']['prepared_manifest']['sha256'],expected_config_sha256=m.sha(m.CONFIG),contrastive_execution='full_batch')
    with patch.dict(sys.modules,{'main':main,'sequence_contrastive':contrast,'composer.utils':composer_utils,'composer.utils.reproducibility':repro}):
        result=ns['execute_native'](args,output,guard,helpers,{})
    check('unchanged_execute_native_MLM17_then_contrastive36_seed_order',[e for e in events if e[0]=='seed']==[('seed',17),('seed',36)])
    check('unchanged_source_model_MLM512_micro64_and_eval512',[e for e in events if e[0]=='mlm_loader']==[('mlm_loader',512,64),('mlm_loader',512,None)])
    check('unchanged_source_model_contrastive_full4096_train_and_eval',[e for e in events if e[0]=='contrastive_loader']==[('contrastive_loader',4096),('contrastive_loader',4096)])
    trains=[e for e in events if e[0]=='train']
    check('unchanged_MLM5epoch_native_seed_precision_schedule',trains[0][1:6]==('mlm','5ep',17,64,'full_batch'))
    check('unchanged_contrastive1epoch_full4096_native_source_schedule',trains[1][1:6]==('contrastive','1ep',36,4096,'full_batch'))
    check('complete_MLM_checkpoint_encoder_initializer_verified',bool(result['encoder_initialization']['all_encoder_keys_loaded_equal']))
    check('phantom_classifier_preserved_unused_by_execution_wiring',result['classifier_unchanged_unused'])
    check('auxiliary_mean_head_export_order_after_contrastive',events.index(('export_native_mean_head_after_contrastive',))>events.index(trains[1]))
    check('capsule_globals_do_not_rebind_original_training_module',ns['SCRIPT']==m.SCRIPT and original_source_model.__globals__ is ns)


def verify_retrieval_integration(directory,fc,data,joins,roster,prepared_dir,np,torch):
    # Actual readout wrapper and unchanged seal kernel on a completely handmade corpus.
    parent=c['parent_ids'][0];run='handmade_fixture_only';stage='retrieve'
    scope,output,tmp,log=m.scopes(c,parent,stage,run)
    fake_scope=directory/'results'/parent;fake_output=fake_scope/'retrieve'/run;fake_output.mkdir(parents=True)
    fake_fit=fake_scope/'fit'/'handmade';fake_fit.mkdir(parents=True)
    vectors=torch.zeros((data['prepared']['unique_full_paths'],768),dtype=torch.float32)
    vectors[:,0]=1;vectors[0]=0
    torch.save({'embeddings':vectors,'native_pooling':'trained_author_mean_head'},fake_fit/'native_path_embeddings.pt')
    np.savez_compressed(fake_fit/'downstream_C4_local_readout.npz',fixture_placeholder=np.asarray([1]))
    result={'status':'passed','action':'fit','exit_code':0,'root_actual_execution':True,'true_query_ancestry_read':False,
      'implementation_sha256':m.sha(m.SCRIPT),'config_sha256':m.sha(m.CONFIG),'parent_id':parent,
      'namespace':data['prepared']['namespace'],'prepared_manifest_sha256':data['dispatcher']['prepared_manifest']['sha256'],
      'published_native_task_reproduction':False,'classifier_unchanged_unused':True,
      'encoder_initialization':{'all_encoder_keys_loaded_equal':['fixture.bert.weight'],'missing_encoder':[],'unequal_encoder':[]},
      'readout':{'native_embedding_sha256':hashlib.sha256(vectors.numpy().tobytes()).hexdigest()},
      'artifacts_sha256':{p.name:m.sha(p) for p in fake_fit.iterdir()}}
    for phase,mode,epochs,key in [('pretraining','mlm',5,'MLM_optimizer_updates'),('contrastive','contrastive',1,'contrastive_optimizer_updates')]:
        result[phase]={'mode':mode,'execution':'full_batch','completed_epochs':epochs,'actual_optimizer_updates':data['budget'][key],
            'backbone_unchanged':True,'true_query_ancestry_read':False,'initial_model_state_sha256':'1'*64,'final_model_state_sha256':'2'*64}
    fitref=write_json(fake_fit/'receipt.json',result)
    rc=copy.deepcopy(c);rc['result_root']=str(directory/'results')
    storage=types.SimpleNamespace(check=lambda:{'handmade_storage_stub':True})
    args=types.SimpleNamespace(parent=parent,stage=stage,run_id=run)
    seal=m.execute_retrieval(rc,args,{'completed_fit_receipt':fitref},data,storage,{})
    check('full_retrieval_wrapper_handmade_complete_dispatch_to_native_seal',seal['status']=='passed')
    s=seal['prediction_seal'];diag=s['diagnostics']
    check('full_pure_corpus_both_native_train_and_val_splits_retained',diag['pure_reference_full_paths']==4608)
    check('every_handmade_query_path_exhaustively_retrieved',diag['unique_query_paths']==5760 and diag['exhaustive_pairs']==26542080)
    check('embedding_zero_row_disclosed_without_normalization_substitution',diag['native_embedding_zero_or_epsilon_rows']==1)
    check('duplicate_embedding_rows_retained_and_exact_cross_class_ties_disclosed',diag['top1_cross_class_exact_max_ties']==5760)
    with np.load(m.bound(s['neighbors']),allow_pickle=False) as z:
        check('exact_score_ties_use_original_full_path_ID',np.array_equal(z['neighbor_full_path_ids'][0],np.arange(1,7)))
        check('all_neighbor_IDs_are_pure_reference_full_paths',np.all(z['neighbor_full_path_ids']<4608))
    with np.load(m.bound({'path':s['path'],'sha256':s['sha256']}),allow_pickle=False) as z:
        check('primary_k1_and_sensitivity_k5_full_C4_grid_sealed',z['C4_probabilities_raw_k1'].shape==(48,120,4) and z['C4_probabilities_raw_k5'].shape==(48,120,4))
        check('raw_vote_zeros_are_retained',np.any(z['C4_probabilities_raw_k1']==0))
        check('original_nonconsecutive_native_sample_IDs_exported',np.array_equal(z['query_nodes'],np.asarray([7+3*i for i in range(160,280)])))
        check('original_100bp_support_and_edges_preserved',np.array_equal(z['positions_bp'],np.arange(50,4800,100)) and np.array_equal(z['bin_edges_bp'],np.arange(0,4801,100)))
    check('retrieval_truth_remains_separate_and_unauthorized',not seal['external_truth_access_authorized'] and not s['truth_opened'])
    bad=copy.deepcopy(result);bad['contrastive']['execution']='gradient_cache'
    reject('reject_gradient_cache_fit_substitution',lambda:m.verify_completed_fit(c,data,bad))
    bad=copy.deepcopy(result);bad['pretraining']['actual_optimizer_updates']-=1
    reject('reject_partial_MLM_optimizer_budget',lambda:m.verify_completed_fit(c,data,bad))
    bad=copy.deepcopy(result);bad['prepared_manifest_sha256']='3'*64
    reject('reject_cross_ARG_or_stale_prepared_fit',lambda:m.verify_completed_fit(c,data,bad))
    bad=copy.deepcopy(joins);bad['query_path_ids'][0,0]=-1
    ns=m.compile_retrieval(c)
    with (prepared_dir/'corpus.pkl').open('rb') as stream:corpus=pickle.load(stream)
    reject('reject_missing_query_occurrence_support',lambda:ns['validate_support'](corpus['paths'],corpus['occurrences'],corpus['namespace'],corpus['namespace'],bad,4800))
    reject('reject_cross_ARG_corpus_namespace',lambda:ns['validate_support'](corpus['paths'],corpus['occurrences'],'wrong',corpus['namespace'],joins,4800))
    check('no_CUDA_context_after_handmade_tensor_retrieval',not torch.cuda.is_initialized())


def verify_storage_and_memory(directory,data):
    fc=copy.deepcopy(c);fc['result_root']=str(directory/'storage_results');fc['short_tmp_root']=str(directory/'storage_tmp')
    lock=directory/'scheduler.lock';lock.write_bytes(b'')
    ledger=directory/'ledger.json'
    fc['storage']['scheduler_lock']=str(lock);fc['storage']['legacy_ledger_path']=str(ledger)
    parent=c['parent_ids'][0];scope,output,tmp,log=m.scopes(fc,parent,'fit','fixture')
    scope.mkdir(parents=True);tmp.mkdir(parents=True)
    binding={'schema':'gpu_pool_storage_ledger_v1','root':str(ROOT),'snapshot_sha256':'4'*64,
       'baseline_bytes':1024,'external_reserve_bytes':4096,'limit_bytes':450000000000}
    key=fc['storage']['reservation_key_prefix']+'/'+parent
    state={**binding,'jobs':{key:{'reserved_bytes':20000000000,'output':str(scope)}},'controllers':{}}
    write_json(ledger,state)
    receipt={'kind':'native_argformer_tsinfer_fit_v1_root_storage','authorizer':'root_agent','active':True,
      'implementation':m.reference(m.SCRIPT),'config':m.reference(m.CONFIG),'scope_parents':[parent],
      'reservation_key':key,'reserved_bytes':20000000000,'output':str(scope),'write_scopes':[str(scope),str(tmp)],
      'mode':'legacy450','legacy_binding':binding,'legacy_ledger_sha256_at_admission':m.sha(ledger),
      'stage_boundary_cap_not_OS_quota':True}
    storage=m.Storage(fc,parent,'fit','fixture',receipt)
    check('legacy_exact_live_first_read_admission',storage.last['legacy_ledger_sha256']==m.sha(ledger))
    first=storage.last['legacy_ledger_sha256'];state['jobs']['unrelated_legitimate']={'reserved_bytes':1024,'output':str(directory/'other')};write_json(ledger,state)
    check('parallel_unrelated_reservation_additions_allowed_after_initial_bind',storage.check()['legacy_ledger_sha256']!=first)
    state['jobs'].pop('unrelated_legitimate');write_json(ledger,state)
    check('parallel_unrelated_reservation_settlements_allowed',storage.check()['projected_bytes']==20000005120)
    state['unexpected_ledger_field']=True;write_json(ledger,state)
    reject('reject_unreviewed_legacy_schema_extension',storage.check)
    state.pop('unexpected_ledger_field');write_json(ledger,state)
    bad=copy.deepcopy(receipt);bad['legacy_ledger_sha256_at_admission']='5'*64
    reject('stale_initial_live_ledger_SHA_rejected',lambda:m.Storage(fc,parent,'fit','fixture',bad))
    state['jobs'][key]['reserved_bytes']=100;write_json(ledger,state)
    reject('cannot_borrow_or_change_own_reservation',storage.check)
    state['jobs'][key]['reserved_bytes']=20000000000;state['baseline_bytes']+=1;write_json(ledger,state)
    reject('no_opportunistic_ledger_baseline_rebase',storage.check)
    state['baseline_bytes']-=1;state['limit_bytes']=950000000000;write_json(ledger,state)
    reject('no_opportunistic_legacy450GB_ceiling_raise',storage.check)
    state['limit_bytes']=450000000000;write_json(ledger,state)
    with patch.object(m,'local_bytes',return_value=20000000001):reject('combined_scope_sampled_cap_enforced',storage.check)
    storage.high_water=0
    link=scope/'bad_link';link.symlink_to(lock)
    reject('owned_scope_symlinks_rejected',storage.check);link.unlink()
    check('output_and_short_tmp_are_both_in_own_write_scopes',receipt['write_scopes']==[str(scope),str(tmp)])
    p=data['dispatcher'];mem={'status':'passed','action':'probe-memory','exit_code':0,
      'implementation_sha256':m.sha(m.SCRIPT),'config_sha256':m.sha(m.CONFIG),
      'prepared_manifest_sha256':p['prepared_manifest']['sha256'],'parent_id':parent,'namespace':data['prepared']['namespace'],
      'full_batch_probe_attempted':True,'full_batch_probe_logical_batch':4096,
      'runtime':{'actual_cuda':True,'device':{'actual_uuid':'fixture-UUID'}},
      'result':{'logical_batch':4096,'execution':'actual_author_full_batch','optimizer_step':True,
        'probe_is_not_a_biological_fit':True,'temporary_probe_weights_restored':True,'peak_memory_bytes':1024}}
    runtime={'maximum_GPU_memory_bytes':2048,'qualified_GPU_UUID':'fixture-UUID'}
    m.verify_memory(c,data,mem,runtime);checks.append('complete_handmade_native4096_memory_gate_schema')
    for field,value in [('status','failed'),('full_batch_probe_attempted',False),('full_batch_probe_logical_batch',2048),('prepared_manifest_sha256','9'*64)]:
        bad=copy.deepcopy(mem);bad[field]=value
        reject('memory_gate_rejects_'+field,lambda bad=bad:m.verify_memory(c,data,bad,runtime))
    bad=copy.deepcopy(mem);bad['result']['temporary_probe_weights_restored']=False
    reject('memory_gate_requires_model_weight_restoration',lambda:m.verify_memory(c,data,bad,runtime))
    reject('memory_gate_requires_exact_reserved_VRAM',lambda:m.verify_memory(c,data,mem,dict(runtime,maximum_GPU_memory_bytes=1000)))
    reject('memory_gate_requires_same_scheduled_GPU_UUID',lambda:m.verify_memory(c,data,mem,dict(runtime,qualified_GPU_UUID='different')))
    check('actual_probe_never_substituted_for_biological_fit',mem['result']['probe_is_not_a_biological_fit'])
    reject('unbound_additive_helper_cannot_use_main_environment',lambda:m.additive_helper_paths({'additive_helpers':[{'module':'threadpoolctl','version':'3.6.0','implementation':m.reference(ROOT/'.runtime/env/lib/python3.11/site-packages/threadpoolctl.py'),'wheel':m.reference(lock),'directory_inventory':m.reference(ledger)}]}))
    reject('unrelated_additive_module_rejected',lambda:m.additive_helper_paths({'additive_helpers':[{'module':'torch'}]}))
    check('no_unrequested_runtime_helpers_default',m.additive_helper_paths({})==[])


def verify_original_launcher_fixture(directory):
    """Unchanged existing launcher plus genuine lease validator, fake inventory/child."""
    launch_path=ROOT/'scripts/launch_gpu.py'
    tree=ast.parse(launch_path.read_text())
    ns={'__name__':'_handmade_existing_launcher','__file__':str(launch_path)}
    module=ast.Module(body=[n for n in tree.body if not isinstance(n,ast.If)],type_ignores=[])
    exec(compile(module,str(launch_path),'exec'),ns)
    ns['ROOT']=directory/'launcher_fixture';ns['ROOT'].mkdir()
    lockdir=ns['ROOT']/'.runtime/locks';lockdir.mkdir(parents=True)
    ns['require_storage']=lambda headroom_bytes:12345
    records=[]
    ns['gpu_inventory']=lambda:[{'index':i,'uuid':'GPU-11111111-1111-1111-1111-11111111111'+str(i),
       'memory_used_mib':0,'memory_total_mib':80000,'utilization_percent':0,'has_compute_process':False} for i in (2,3,4,5)]
    guard=m.load_frozen_module(c,'src/arggnn/native_argformer_exec.py','_handmade_original_guard')
    guard.ROOT=ns['ROOT']
    def popen(command,**kwargs):
        proof=guard.validate_lease(kwargs['env'])
        records.append((command,kwargs,proof))
        return types.SimpleNamespace(wait=lambda:0,poll=lambda:0,pid=999999)
    import fcntl,subprocess,signal
    held3=(lockdir/'gpu_3.lock').open('a+');fcntl.flock(held3,fcntl.LOCK_EX|fcntl.LOCK_NB)
    argv=['launch_gpu.py','--gpu','2','--job','native_argformer_fixture_lease','arggnn.native_argformer_tsinfer_fit_v1','--execute']
    with patch.object(sys,'argv',argv),patch.object(subprocess,'Popen',side_effect=popen),patch.object(signal,'signal'),patch.dict(os.environ,{'ARGGNN_LOGICAL_JOB_NAME':'fixture_logical_job'}):
        check('original_launcher_fixture_can_add_GPU2_while_GPU3_is_held',ns['main']()==0)
    check('genuine_original_guard_validates_launcher_inherited_OFD',records[0][2]['physical_gpu']==2)
    check('original_launcher_preserves_exact_logical_job_environment',records[0][1]['env']['ARGGNN_LOGICAL_JOB_NAME']=='fixture_logical_job')
    check('original_launcher_passes_lease_FD_to_child',records[0][2]['lease_fd'] in records[0][1]['pass_fds'])
    check('existing_launcher_does_not_register_duplicate_legacy_output',not (ns['ROOT']/'.runtime/gpup/storage_reservations.json').exists())
    held4=(lockdir/'gpu_4.lock').open('a+');fcntl.flock(held4,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with patch.object(sys,'argv',argv),patch.object(subprocess,'Popen',side_effect=popen),patch.object(signal,'signal'):
        reject('unchanged_launcher_two_physical_GPU_ceiling',ns['main'])
    held4.close();held3.close()
    template=json.loads((ROOT/'research/reviews/project_budget_v3_author/root_reserve_admission_template.json').read_text())
    for reservation,parent in zip(template['reservations'],c['parent_ids']):
        scope,_,tmp,_=m.scopes(c,parent,'probe-memory','fixture')
        check('supplemental_exact_output_scope_'+parent,reservation['output']==str(scope))
        check('supplemental_exact_short_tmp_scope_'+parent,reservation['auxiliary_scopes']==[str(tmp)])


def verify_bootstrap_fixture(directory):
    fc=copy.deepcopy(c);fc['result_root']=str(directory/'bootstrap_results');fc['short_tmp_root']='.runtime/tmp/afts1'
    parent=c['parent_ids'][0];args=types.SimpleNamespace(parent=parent,stage='retrieve',run_id='fixture',admission=str(directory/'bootstrap_admission.json'))
    write_json(Path(args.admission),{'handmade_fixture_only':True})
    _,output,tmp,_=m.scopes(fc,parent,'retrieve','fixture')
    ownership={'owns_output':False};storage=types.SimpleNamespace(check=lambda:{'fixture_only':True})
    runtime={'python_executable_sha256':m.sha(ROOT/c['runtime']['interpreter']),'additive_helpers':[]}
    invoked=[]
    def fake_exec(path,argv,environ):
        invoked.append((path,argv,environ));raise RuntimeError('fixture intentional exec stop')
    with patch.dict(os.environ,dict(os.environ),clear=True),patch.object(os,'execve',side_effect=fake_exec):
        reject('handmade_bootstrap_stops_before_actual_exec',lambda:m.bootstrap(fc,args,{},runtime,storage,ownership))
    check('bootstrap_owns_only_fresh_fixture_stage_output',ownership['owns_output'] and output.is_dir())
    admission=json.loads((output/'exec_admission.json').read_text())
    check('bootstrap_seals_exact_implementation_and_root_admission',admission['implementation_sha256']==m.sha(m.SCRIPT) and admission['root_admission']==m.reference(args.admission))
    check('bootstrap_execs_only_pinned_isolated_python',invoked[0][0]==ROOT/c['runtime']['interpreter'])
    check('bootstrap_child_hides_CUDA_for_CPU_retrieval',invoked[0][2]['CUDA_VISIBLE_DEVICES']=='')
    check('bootstrap_child_uses_exact_short_TMPDIR',invoked[0][2]['TMPDIR']==str(tmp))
    check('bootstrap_child_excludes_unbound_environment_libraries',invoked[0][2]['PYTHONPATH']=='')
    check('bootstrap_child_preserves_offline_and_no_bytecode_policy',invoked[0][2]['HF_HUB_OFFLINE']=='1' and invoked[0][2]['PYTHONDONTWRITEBYTECODE']=='1')
    reject('bootstrap_rejects_output_reuse',lambda:m.bootstrap(fc,args,{},runtime,storage,{'owns_output':False}))
    tmp.rmdir()
    check('pinned_conda_python_alias_is_explicitly_confined',m.native_executable(c).resolve().parent==ROOT/c['runtime']['prefix']/'bin')
    bad=copy.deepcopy(c);bad['runtime']['interpreter']='.runtime/env/bin/python'
    reject('alternate_environment_interpreter_rejected',lambda:m.native_executable(bad))


with tempfile.TemporaryDirectory(prefix='handmade_',dir=REVIEW) as temp:
    directory=Path(temp)
    import numpy as np
    import torch
    fc,data,joins,roster,prepared_dir,np,torch=fixture(directory,np,torch)
    proof=m.ast_proof(c)
    check('all19_native_fit_scientific_AST_definitions_frozen',len(proof[m.FIT])==19)
    check('all11_native_retrieval_AST_definitions_frozen',len(proof[m.RETRIEVAL])==11)
    check('source_check_does_not_decode_actual_arrays',m.source_report(c)['actual_arrays_opened'] is False)
    for stage in m.STAGES:
        t=m.templates(c,stage=stage)
        check('inactive_'+stage+'_admission_template',t['admission']['active'] is False)
        check('inactive_'+stage+'_runtime_template',t['runtime']['active'] is False)
        check('inactive_'+stage+'_storage_template',t['storage']['active'] is False)
        reject('inactive_'+stage+'_template_cannot_admit',lambda:m.validate_admission(c,c['parent_ids'][0],stage,'root_assign_fresh_run',t['admission'],{},t['runtime'],{}))
    a=m.admission_bindings(c,c['parent_ids'][0],'fit','fixture')
    review={'kind':'native_argformer_tsinfer_fit_v1_independent_review','review_role':'independent_reviewer','verdict':'PASS','implementation':a['implementation'],'config':a['config']}
    inventory_ref=write_json(directory/'fixture_runtime_inventory.json',{'handmade_fixture':True})
    runtime={'kind':'native_argformer_tsinfer_fit_v1_root_runtime','authorizer':'root_agent','active':True,'CPU_only':False,**c['runtime'],'python_executable_sha256':'1'*64,'environment_inventory_receipt':inventory_ref}
    reg={'kind':'native_argformer_tsinfer_fit_v1_root_experiment_registration','authorizer':'root_agent','active':True,
      'condition':c['condition'],'scope_parents':[c['parent_ids'][0]],'stage':'fit','implementation':a['implementation'],'config':a['config'],
      'query_truth_authorized':False,'hypothesis_id':'H013','repeat_reason':'fixture only'}
    check('exact_handmade_root_admission_positive_schema',m.validate_admission(c,c['parent_ids'][0],'fit','fixture',a,review,runtime,reg)==a)
    for field,value in [('active',False),('query_truth_authorized',True),('scientific_fallback',True),('contrastive_execution','gradient_cache'),('stage','retrieve'),('scope_parents',[c['parent_ids'][1]]),('maximum_walltime_seconds',1),('output',str(directory/'unassigned'))]:
        bad=copy.deepcopy(a);bad[field]=value
        reject('reject_admission_mutation_'+field,lambda bad=bad:m.validate_admission(c,c['parent_ids'][0],'fit','fixture',bad,review,runtime,reg))
    for field,value in [('verdict','PENDING'),('review_role','author'),('implementation',{'path':'x','sha256':'1'*64})]:
        bad=copy.deepcopy(review);bad[field]=value
        reject('reject_review_mutation_'+field,lambda bad=bad:m.validate_admission(c,c['parent_ids'][0],'fit','fixture',a,bad,runtime,reg))
    for path in ['../bad','/tmp/external','data/query_truth.json','data/oracle_file.pkl']:
        reject('reject_unsafe_path_'+path,lambda path=path:m.confined(path))
    f=directory/'immutable.json';r=write_json(f,{'fixture':1});f.write_text('{}')
    reject('reject_changed_immutable_metadata',lambda:m.read_json(r))
    reject('reject_old_preparation_only_root_admission',lambda:m.validate_admission(c,c['parent_ids'][0],'fit','fixture',{'kind':'official_argformer_tsinfer_parent_v1_root_admission','active':True},review,runtime,reg))
    for path in [directory/'missing.npy',directory/'metadata.pkl']:
        path.write_bytes(b'handmade')
        reject('reject_non_JSON_admission_payload',lambda path=path:m.read_json(m.reference(path)))
    symlink=directory/'symbolic';symlink.symlink_to(f)
    reject('reject_input_symlink',lambda:m.confined(symlink))
    for stage in m.STAGES:
        _,_,tmp,_=m.scopes(c,c['parent_ids'][0],stage,'fixture')
        check('short_native_socket_'+stage,len(os.fsencode(tmp/'pymp-12345678/listener-12345678'))<=107)
    verify_bootstrap_fixture(directory)
    verify_original_launcher_fixture(directory)
    verify_storage_and_memory(directory,data)
    verify_source_wiring(directory,fc,data,torch)
    verify_retrieval_integration(directory,fc,data,joins,roster,prepared_dir,np,torch)

report={'kind':'native_argformer_tsinfer_fit_v1_author_handmade_checks','status':'passed','checks':checks,'check_count':len(checks),
  'implementation':m.reference(SOURCE),'config':m.reference(m.CONFIG),'test_source':m.reference(__file__),
  'native_model_executed':False,'native_fit_executed':False,'actual_scientific_arrays_opened':False,
  'handmade_CPU_tensors_and_primitive_pickles_only':True,'query_truth_opened':False,'GPU':False,
  'execution_wiring_dependencies_stubbed_only_in_test_namespace':True,
  'test_only_existing_project_threadpoolctl_module_if_missing_from_native_env':test_only_dependency,
  'test_only_threadpoolctl_version':sys.modules['threadpoolctl'].__version__,'real_retrieval_wrapper_and_AST_kernels_exercised':True}
write_json(REVIEW/'handmade_checks.json',report)
print(json.dumps(report,indent=2))
